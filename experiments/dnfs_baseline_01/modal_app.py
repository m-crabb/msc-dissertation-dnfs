"""Modal wrapper for DNFS Ising baseline training. Delegates the actual
training to `run.train` so the remote and local code paths share a single
implementation.

The container image is built by installing pixi inside the container and
running `pixi install --environment dev --locked` against the project's
`pixi.lock`, so the remote runtime stack (pytorch, numpy, ...) matches
local dev byte-for-byte. One source of truth is `pixi.lock` -- there is
no second dep list living in this file.

The `CONDA_OVERRIDE_CUDA=12.4` env var is required because the build
container has no GPU and pixi's `__cuda` virtual package check would
otherwise fail; runtime containers get a real GPU from Modal.

Usage (after `modal token new` and `modal secret create wandb-secret ...`):
    # Single config:
    pixi run -e dev modal run -m \\
        experiments.dnfs_baseline_01.modal_app::main \\
        --cfg-name stage_1_d4 --seed 42

    # All 8 post-redo configs in parallel. `--detach` is REQUIRED:
    # without it, the ephemeral app stops when the entrypoint returns and
    # all spawned FunctionCalls are cancelled before any container runs.
    pixi run -e dev modal run --detach -m \\
        experiments.dnfs_baseline_01.modal_app::batch --scale all
"""
import os

import modal

from experiments.dnfs_baseline_01.configs import CONFIGS

PROJECT_DIR = "/repo"
# Modal app name. The constrained track has its own hardcoded app name in
# `experiments.constrained_soft_02.modal_app`.
APP_NAME = os.environ.get("DNFS_MODAL_APP", "dnfs-baseline")
PIXI_ENV_BIN = f"{PROJECT_DIR}/.pixi/envs/cuda/bin"

# Build the container image from `pixi.lock`. The repo is added at /repo;
# pixi installs the dev env in-place; PATH points at the pixi env's bin.
#
# Local-only directories (`.pixi/` env tree, past `results/`, `wandb/` run
# logs, `.git`, caches) are excluded from the upload. They are large,
# irrelevant on the remote, and `.pixi/` in particular causes "modified
# during build" failures when any local pixi-run command touches conda
# metadata while the Modal uploader is still streaming bytes.
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
        "pixi install --environment cuda --locked"
    )
    .env({"PATH": f"{PIXI_ENV_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)

# Persistent volume for run artefacts (training_log.csv, checkpoints, eval
# tensors). One volume across runs lets us re-pull anything later via
# `modal volume get`.
volume = modal.Volume.from_name("dnfs-results", create_if_missing=True)

# wandb API key lives in a Modal secret, NOT in the image. `modal secret
# create wandb-secret WANDB_API_KEY=...` makes it available as an env var.
wandb_secret = modal.Secret.from_name("wandb-secret")

app = modal.App(APP_NAME, image=image)


def _validate_cfg_name(cfg_name: str) -> None:
    if cfg_name not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown cfg_name {cfg_name!r}. Valid configs: {valid}")


@app.function(
    # A100 for Stage 4 leTF re-launch (attention-bound; 2x faster wall-clock
    # vs L4 at d=100). Earlier MLP/leconv stages ran fine on L4; if cost
    # matters for non-attention runs, downgrade per-launch by editing here.
    gpu="A100",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    # 24h is generous for D=10; tighten if cost matters.
    timeout=24 * 60 * 60,
    # Note: `nonpreemptible=True` is not supported for GPU workloads on Modal
    # (rejected at deploy time). Long runs may preempt-and-auto-retry; clean up
    # truncated wandb runs (`_step < n_steps - 1`) post-completion.
)
def train_remote(cfg_name: str, seed: int = 42):
    """Run a single training config on Modal and persist artefacts to the
    volume. Importable but typically launched via `modal run`."""
    import sys

    # /repo is the mount point of `add_local_dir`. Inserting it onto sys.path
    # lets `experiments.dnfs_baseline_01.run` import resolve correctly.
    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_01.run import train
    from experiments.dnfs_baseline_01.configs import CONFIGS

    train(CONFIGS[cfg_name], seed=seed, output_dir="/results")
    # commit() makes the artefacts visible to subsequent `modal volume get`
    # calls. Without this, the volume is reverted on container shutdown.
    volume.commit()


@app.function(
    # L4 variant (2026-08-18): the cheap card for SMALL-footprint cells.
    # Sizing lesson from the d16 control's failed first launch: the eval
    # protocol sets peak memory, not training — an unchunked 5000-draw
    # eval through the dense readout is a (5000, 4, d, 2d) score tensor,
    # 9.77 GiB at d=256, which OOMs the L4's 22 GiB. Check
    # n_eval_samples x heads x d x 2d x 4B against ~20 GiB before
    # routing a cell here; d <= 100 cells all fit.
    gpu="L4",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_remote_l4(cfg_name: str, seed: int = 42):
    """train_remote on an L4; see train_remote for the body contract."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_01.run import train
    from experiments.dnfs_baseline_01.configs import CONFIGS

    train(CONFIGS[cfg_name], seed=seed, output_dir="/results")
    volume.commit()


@app.function(
    # A100 deliberately, matching train_remote: the archived evals were drawn
    # on the training card, so a re-eval on the SAME device isolates whatever
    # the re-eval changed (e.g. the corrected base draw for the 2026-06-17
    # matched-base runs) as the single moved variable. A CPU re-run would
    # move device and draw at once and the read becomes unattributable.
    gpu="A100",
    volumes={"/results": volume},
    # Eval-only: one 5000-draw batch, minutes at D=10; 2h is generous.
    timeout=2 * 60 * 60,
)
def eval_remote(run_dir_name: str, redraw: bool = False, redraw_seed: int = 0):
    """Re-run the end-of-run eval for a run dir already on the volume.

    Mirrors `constrained_hard_03.modal_app.eval_remote` but calls the
    BASELINE `eval_only`, which is what the baseline and soft cells actually
    use — the hard app's eval path reads hard-experiment configs and cannot
    recover these runs. This gap (an eval recovery path existing only for
    the hard chapter) is part of why the bugged 2026-06-17 matched-base
    eval went unchallenged for six weeks.

    `redraw=False` rescores the saved tensors; `redraw=True` draws a fresh
    eval batch from `checkpoints/final.pt` (archiving the stale `eval/` to
    `eval_archived_pre_redraw/` on the volume first) — required when the
    archived DRAW itself was wrong, since a wrong x0 is baked into the
    saved log-weights and no rescoring can remove it.
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_01.run import eval_only

    metrics = eval_only(
        Path("/results") / run_dir_name, redraw=redraw, redraw_seed=redraw_seed
    )
    print(f"[eval_remote] {run_dir_name}: {json.dumps(metrics, indent=2)}")
    volume.commit()


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 42):
    """Local CLI entry: spawns `train_remote` as a remote Modal call."""
    _validate_cfg_name(cfg_name)
    train_remote.remote(cfg_name=cfg_name, seed=seed)


@app.local_entrypoint()
def main_l4(cfg_name: str, seed: int = 42):
    """Spawn (not block on) an L4 run — pair with `modal run --detach` so
    a long cheap run survives the client exiting."""
    _validate_cfg_name(cfg_name)
    call = train_remote_l4.spawn(cfg_name=cfg_name, seed=seed)
    print(f"spawned {cfg_name} seed {seed} on L4: {call.object_id}")


@app.local_entrypoint()
def batch(scale: str = "all", seed: int = 42):
    """Fire off post-redo configs in parallel. scale: 'all' | 'd4' | 'd10'."""
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
def batch_seeds(cfg_name: str, seeds: str = "42"):
    """Spawn one config across multiple seeds in parallel.

    Example: `--cfg-name stage_4_d10_paper_probe_warmup --seeds "42,43,44,45"`.
    """
    _validate_cfg_name(cfg_name)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    for seed in seed_list:
        train_remote.spawn(cfg_name=cfg_name, seed=seed)
    print(f"spawned {len(seed_list)} jobs: cfg={cfg_name}, seeds={seed_list}")
