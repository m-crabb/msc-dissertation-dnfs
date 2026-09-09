"""Modal wrapper for DNFS Ising baseline training. Delegates to `run.train`
so the remote and local code paths share one implementation.

The container image installs pixi inside the container and runs
`pixi install --environment dev --locked` against `pixi.lock`, so the remote
runtime stack matches local dev byte-for-byte. `CONDA_OVERRIDE_CUDA=12.4` is
needed because the build container has no GPU and pixi's `__cuda` virtual
package check would otherwise fail.

Usage (after `modal token new` and `modal secret create wandb-secret ...`):
    pixi run -e dev modal run -m \\
        experiments.dnfs_baseline_01.modal_app::main \\
        --cfg-name stage_1_d4 --seed 42

    # `--detach` is required: without it the ephemeral app stops when the
    # entrypoint returns and spawned FunctionCalls are cancelled.
    pixi run -e dev modal run --detach -m \\
        experiments.dnfs_baseline_01.modal_app::batch --scale all
"""

import os
import time

import modal
from experiments.dnfs_baseline_01.configs import CONFIGS

PROJECT_DIR = "/repo"
# Modal app name. The constrained track has its own hardcoded app name in
# `experiments.constrained_soft_02.modal_app`.
APP_NAME = os.environ.get("DNFS_MODAL_APP", "dnfs-baseline")
PIXI_ENV_BIN = f"{PROJECT_DIR}/.pixi/envs/cuda/bin"

# Local-only directories are excluded from the upload: `.pixi/` in particular
# causes "modified during build" failures when a local pixi-run command
# touches conda metadata while the Modal uploader is still streaming.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://pixi.sh/install.sh | bash",
        "ln -s /root/.pixi/bin/pixi /usr/local/bin/pixi",
    )
    .workdir(PROJECT_DIR)
    .add_local_dir(
        ".",
        PROJECT_DIR,
        copy=True,
        ignore=[
            ".pixi/**",
            "results/**",
            "wandb/**",
            ".git/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            ".venv/**",
            "*.pdf",
        ],
    )
    .run_commands(
        f"cd {PROJECT_DIR} && CONDA_OVERRIDE_CUDA=12.4 "
        "pixi install --environment cuda --locked",
        # torch.compile needs the CUDA driver-API header: inductor compiles a
        # small cuda_utils.c with the system gcc and the locked pixi env ships
        # no CUDA dev headers. The runtime wheel carries include/cuda.h;
        # --no-deps leaves the locked env's torch/nvidia libs untouched, and
        # CPATH below puts the header on gcc's search path.
        f"{PROJECT_DIR}/.pixi/envs/cuda/bin/python -m ensurepip && "
        f"{PROJECT_DIR}/.pixi/envs/cuda/bin/python -m pip install "
        "--no-deps nvidia-cuda-runtime-cu12",
    )
    .env(
        {
            "PATH": (
                f"{PIXI_ENV_BIN}:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "CPATH": (
                f"{PROJECT_DIR}/.pixi/envs/cuda/lib/python3.11/"
                "site-packages/nvidia/cuda_runtime/include"
            ),
            # Venue parity with the DoC sbatch scripts, which export
            # this. Allocator headroom, not a speed lever.
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

# Persistent volume for run artefacts (training_log.csv, checkpoints, eval
# tensors); one volume across runs lets us re-pull anything via
# `modal volume get`.
volume = modal.Volume.from_name("dnfs-results", create_if_missing=True)

# wandb API key lives in a Modal secret, not in the image. `modal secret
# create wandb-secret WANDB_API_KEY=...` makes it available as an env var.
wandb_secret = modal.Secret.from_name("wandb-secret")

app = modal.App(APP_NAME, image=image)


def _validate_cfg_name(cfg_name: str) -> None:
    if cfg_name not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown cfg_name {cfg_name!r}. Valid configs: {valid}")


@app.function(
    # A100-80GB by default: b512 at d=256 needs the 80 GB card, and the
    # hard-recipe ladder's 8x8/16x16 siblings ran on it. The decorator is
    # evaluated where `modal run` executes, so this is a launch-time lever
    # (running apps keep their spawn-time spec); non-attention stages fit an L4.
    gpu=os.environ.get("DNFS_TRAIN_GPU", "A100-80GB"),
    volumes={"/results": volume},
    secrets=[wandb_secret],
    # 24h is generous for D=10; tighten if cost matters.
    timeout=24 * 60 * 60,
    # Note: `nonpreemptible=True` is not supported for GPU workloads on Modal
    # (rejected at deploy time). Long runs preempt-and-auto-retry; the retry
    # resumes from `checkpoints/resume.pt` only when the caller minted a
    # stable `tag` (see train_remote's docstring).
)
def train_remote(cfg_name: str, seed: int = 42, tag: str = ""):
    """Run a single training config on Modal and persist artefacts to the
    volume. Importable but typically launched via `modal run`.

    `tag` replaces the run dir's timestamp suffix (see `run.train`); "" is
    the "no tag" sentinel because Modal's CLI cannot pass None. A preemption
    re-runs this function with identical inputs, so a tag fixed at spawn lands
    the retry in the same run dir, where `checkpoints/resume.pt` continues
    from the last outer-cycle boundary. Without one the retry mints a fresh
    timestamped sibling and restarts from step 0 (2-Sep hard-recipe 8x8/16x16
    seeds each left two dirs this way).
    """
    import sys

    # /repo is the mount point of `add_local_dir`; on sys.path so the
    # `experiments.dnfs_baseline_01.run` import resolves.
    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_01.configs import CONFIGS
    from experiments.dnfs_baseline_01.run import train

    train(
        CONFIGS[cfg_name],
        seed=seed,
        output_dir="/results",
        tag=tag or None,
        # A preemption gets no chance to flush, so the resume checkpoint is
        # committed to the volume the moment it is written; otherwise the
        # retry finds nothing and restarts from step 0.
        on_checkpoint=volume.commit,
    )
    # commit() makes the artefacts visible to subsequent `modal volume get`
    # calls. Without this, the volume is reverted on container shutdown.
    volume.commit()


@app.function(
    # L4 variant, for small-footprint cells. The eval protocol sets peak
    # memory, not training: an unchunked 5000-draw eval through the dense
    # readout is a (5000, 4, d, 2d) score tensor, 9.77 GiB at d=256, which
    # OOMs the L4's 22 GiB. Check n_eval_samples x heads x d x 2d x 4B
    # against ~20 GiB before routing a cell here; d <= 100 cells all fit.
    gpu="L4",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_remote_l4(cfg_name: str, seed: int = 42):
    """train_remote on an L4; see train_remote for the body contract."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_01.configs import CONFIGS
    from experiments.dnfs_baseline_01.run import train

    train(CONFIGS[cfg_name], seed=seed, output_dir="/results")
    volume.commit()


@app.function(
    # A100, matching train_remote: the archived evals were drawn on the
    # training card, so a re-eval on the same device leaves whatever the
    # re-eval changed as the single moved variable. On CPU, device and draw
    # would move at once and the read becomes unattributable.
    gpu="A100",
    volumes={"/results": volume},
    # Eval-only: one 5000-draw batch, minutes at D=10; 2h is generous.
    timeout=2 * 60 * 60,
)
def eval_remote(
    run_dir_name: str,
    redraw: bool = False,
    redraw_seed: int = 0,
    n_euler_override: int = 0,
):
    """Re-run the end-of-run eval for a run dir already on the volume.

    Mirrors `constrained_hard_03.modal_app.eval_remote` but calls the baseline
    `eval_only`: the hard app's eval path reads hard-experiment configs and
    cannot recover these runs.

    `redraw=False` rescores the saved tensors; `redraw=True` draws a fresh
    eval batch from `checkpoints/final.pt` (archiving the stale `eval/` to
    `eval_archived_pre_redraw/` on the volume first), needed when the archived
    draw itself was wrong, since a wrong x0 is baked into the saved
    log-weights and no rescoring can remove it.

    `n_euler_override > 0` re-draws on that Euler grid instead of the run's
    own (artefacts to eval_ne<k>/, frozen eval/ untouched; see run.eval_only);
    0 is the "no override" sentinel because Modal's CLI cannot pass None.
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_01.run import eval_only

    metrics = eval_only(
        Path("/results") / run_dir_name,
        redraw=redraw,
        redraw_seed=redraw_seed,
        n_euler_override=n_euler_override,
    )
    print(f"[eval_remote] {run_dir_name}: {json.dumps(metrics, indent=2)}")
    volume.commit()


@app.function(gpu="A100", timeout=2 * 60 * 60)
def compile_bench_remote(
    cfg_name: str = "stage_4_d10", n_steps: int = 400, tail: int = 200
):
    """Same-container eager-vs-compiled bench of the flip-route trainer
    (method in compile_bench.py — both arms in one container so the ratio
    is same-device by construction)."""
    from experiments.dnfs_baseline_01.probes.compile_bench import run_bench

    return run_bench(cfg_name=cfg_name, n_steps=n_steps, tail=tail)


@app.local_entrypoint()
def compile_bench(cfg_name: str = "stage_4_d10", n_steps: int = 400, tail: int = 200):
    """Blocking local CLI entry for the compile bench."""
    compile_bench_remote.remote(cfg_name=cfg_name, n_steps=n_steps, tail=tail)


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 42):
    """Local CLI entry: spawns `train_remote` as a remote Modal call.

    The tag is minted here, once, so a preemption retry lands in the same
    run dir; `batch_seeds` takes it from the caller for the same reason.
    """
    _validate_cfg_name(cfg_name)
    train_remote.remote(
        cfg_name=cfg_name, seed=seed, tag=time.strftime("%Y%m%d-%H%M%S")
    )


@app.local_entrypoint()
def main_l4(cfg_name: str, seed: int = 42):
    """Spawn (not block on) an L4 run — pair with `modal run --detach` so
    a long cheap run survives the client exiting."""
    _validate_cfg_name(cfg_name)
    call = train_remote_l4.spawn(cfg_name=cfg_name, seed=seed)
    print(f"spawned {cfg_name} seed {seed} on L4: {call.object_id}")


@app.local_entrypoint()
def batch(scale: str = "all", seed: int = 42):
    """Fire off the stage 0-2 configs in parallel. scale: 'all' | 'd4' | 'd10'."""
    d4 = ["stage_0_d4", "stage_0_d4_cv", "stage_1_d4", "stage_2_d4"]
    d10 = ["stage_0_d10", "stage_0_d10_cv", "stage_1_d10", "stage_2_d10"]
    groups = {"all": d4 + d10, "d4": d4, "d10": d10}
    if scale not in groups:
        valid = ", ".join(sorted(groups))
        raise ValueError(f"Unknown scale {scale!r}. Valid scales: {valid}")
    configs = groups[scale]
    for cfg in configs:
        _validate_cfg_name(cfg)
    for cfg in configs:
        train_remote.spawn(cfg_name=cfg, seed=seed)
    print(f"spawned {len(configs)} jobs: {configs}")


@app.local_entrypoint()
def batch_configs(configs: str = "", seed: int = 42):
    """Spawn arbitrary comma-separated configs in parallel.

    Example: `--configs stage_1_d10,stage_2_d10` to relaunch only the leMLP D=10 runs.
    """
    cfg_list = [c.strip() for c in configs.split(",") if c.strip()]
    for cfg in cfg_list:
        _validate_cfg_name(cfg)
    for cfg in cfg_list:
        train_remote.spawn(cfg_name=cfg, seed=seed)
    print(f"spawned {len(cfg_list)} jobs: {cfg_list}")


@app.local_entrypoint()
def batch_eval(
    run_dirs: str = "",
    redraw: bool = False,
    redraw_seed: int = 0,
    n_euler_override: int = 0,
):
    """Spawn one eval per comma-separated run dir, in parallel.

    Spawned, not called, so the set survives the client exiting -- pair with
    `--detach`.
    """
    dirs = [d.strip() for d in run_dirs.split(",") if d.strip()]
    for run_dir in dirs:
        eval_remote.spawn(
            run_dir_name=run_dir,
            redraw=redraw,
            redraw_seed=redraw_seed,
            n_euler_override=n_euler_override,
        )
    print(f"spawned {len(dirs)} evals at ne{n_euler_override or 'native'}: {dirs}")


@app.local_entrypoint()
def batch_seeds(cfg_name: str, seeds: str = "42", tag: str = ""):
    """Spawn one config across multiple seeds in parallel.

    Example: `--cfg-name stage_4_d10_sc_hardrecipe_efc --seeds "42,43,44,45"
    --tag 20260907-d10-hardrecipe`. Pass a tag: it is what makes a
    preemption retry resume instead of restarting.
    """
    _validate_cfg_name(cfg_name)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    for seed in seed_list:
        train_remote.spawn(cfg_name=cfg_name, seed=seed, tag=tag)
    print(
        f"spawned {len(seed_list)} jobs: cfg={cfg_name}, seeds={seed_list} "
        f"tag={tag or '<timestamp>'}"
    )
