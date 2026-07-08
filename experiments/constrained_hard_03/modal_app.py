"""Modal wrapper for hard-constraint swap-CTMC training. Delegates the
actual training to `experiments.constrained_hard_03.run.train` so the
remote and local code paths share a single implementation.

The container image is built by installing pixi inside the container and
running `pixi install --environment dev --locked` against the project's
`pixi.lock`, so the remote runtime stack matches local dev byte-for-byte.

Usage (after `modal token new` and `modal secret create wandb-secret ...`):
    # Single config, blocking (used for smoke checks):
    pixi run -e dev modal run -m \\
        experiments.constrained_hard_03.modal_app::main \\
        --cfg-name H2_d16_c50_s010_letf_dh --seed 42 --smoke

    # One config across multiple seeds, fire-and-forget:
    pixi run -e dev modal run --detach -m \\
        experiments.constrained_hard_03.modal_app::batch_seeds \\
        --cfg-name H2_d16_c50_s010_letf_dh --seeds "42,43,44"

    # All three sigma-ladder cells x seeds, fire-and-forget:
    pixi run -e dev modal run --detach -m \\
        experiments.constrained_hard_03.modal_app::ladder \\
        --seeds "42,43,44"

    # The 4x4 exact-enumeration gate over the trained run dirs already on
    # the volume (blocking so logs stream; writes /results/gate_4x4):
    pixi run -e dev modal run -m \\
        experiments.constrained_hard_03.modal_app::gate
"""
import modal
from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.run import HEAD_KINDS

PROJECT_DIR = "/repo"
APP_NAME = "dnfs-hard"
PIXI_ENV_BIN = f"{PROJECT_DIR}/.pixi/envs/cuda/bin"

# The sigma-ladder cells that `ladder()` fans out over: same D=16,
# c_target=0.5, leTF, doubly_hollow head; sigma is the only thing that
# varies (subcritical / critical / supercritical, sigma_c ~= 0.22305).
LADDER_CFGS = (
    "H2_d16_c50_s010_letf_dh",
    "H2_d16_c50_s223_letf_dh",
    "H2_d16_c50_s040_letf_dh",
)

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
    .env(
        {
            "PATH": (
                f"{PIXI_ENV_BIN}:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            )
        }
    )
)

volume = modal.Volume.from_name("dnfs-results", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")
app = modal.App(APP_NAME, image=image)


def _validate_cfg_name(cfg_name: str) -> None:
    if cfg_name not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown cfg_name {cfg_name!r}. Valid configs: {valid}")


def _resolve_head_kind(head_kind: str) -> str | None:
    """Modal's CLI can't pass `None`, so entrypoints take `""` as the "use
    the config's default head_kind" sentinel; anything else must be a valid
    `run.HEAD_KINDS` member."""
    resolved = head_kind or None
    if resolved is not None and resolved not in HEAD_KINDS:
        valid = ", ".join(HEAD_KINDS)
        raise ValueError(f"Unknown head_kind {resolved!r}. Valid: {valid}")
    return resolved


@app.function(
    # A100 for seed runs (decision 2026-07-06): the perf profile showed the
    # workload bandwidth-bound (layernorm/copies), where the L4 is weakest;
    # the win concentrates in the eval slices. bench_remote stays on L4 so
    # benchmark numbers remain comparable with the recorded baselines.
    gpu="A100",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_remote(
    cfg_name: str, seed: int = 42, head_kind: str | None = None, smoke: bool = False
):
    """Run a single hard-constraint training config on Modal."""
    import sys
    from dataclasses import replace

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.run import smoke_config, train

    cfg = CONFIGS[cfg_name]
    if head_kind is not None:
        cfg = replace(cfg, head_kind=head_kind)
    if smoke:
        cfg = smoke_config(cfg)

    train(cfg, seed=seed, output_dir="/results")
    volume.commit()


@app.function(
    # Same L4 as train_remote: the gate re-loads the trained d=16 heads and
    # draws 5k-sample swap-CTMC evals -- tiny on GPU, ~2h40m on local CPU.
    gpu="L4",
    volumes={"/results": volume},
    timeout=4 * 60 * 60,
)
def gate_remote(
    seeds: str = "42,43,44", n_samples: int = 5000, skip_controls: bool = False
):
    """Run the 4x4 exact-enumeration go/no-go gate against the trained run
    dirs already on the volume; writes verdict.json + plot to /results/gate_4x4."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.gate_4x4 import main as gate_main

    argv = [
        "--results-dir", "/results",
        "--device", "cuda",
        "--out", "/results/gate_4x4",
        "--seeds", seeds,
        "--n-samples", str(n_samples),
    ]
    if skip_controls:
        argv.append("--skip-controls")
    gate_main(argv)
    volume.commit()


@app.function(gpu="L4", volumes={"/results": volume}, timeout=2 * 60 * 60)
def demo_remote(seeds: str = "42,43,44", n_samples: int = 5000,
                n_replicates: int = 5):
    """GPU stage of the 4x4 demo analysis (2026-07-08): gate fidelity +
    neural replicate estimates for the 10k MA/MO cells; writes
    /results/demo_4x4/neural_estimates.json. L4 suffices at d=16."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.demo_4x4 import main as demo_main

    demo_main([
        "gpu", "--results-dir", "/results", "--device", "cuda",
        "--seeds", seeds, "--n-samples", str(n_samples),
        "--n-replicates", str(n_replicates),
        "--out", "/results/demo_4x4",
    ])
    volume.commit()


@app.function(gpu="L4", volumes={"/results": volume}, timeout=60 * 60)
def phi_hist_remote(seeds: str = "42,43,44", n_samples: int = 5000):
    """Light GPU stage: pooled IS-weighted phi histograms for the demo cells
    (mode-coverage exhibit); writes /results/demo_4x4/phi_hists.json."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.demo_4x4 import main as demo_main

    demo_main([
        "phi", "--results-dir", "/results", "--device", "cuda",
        "--seeds", seeds, "--n-samples", str(n_samples),
        "--out", "/results/demo_4x4",
    ])
    volume.commit()


@app.function(
    # A100 like train_remote: the eval slices are exactly where the perf
    # profile showed the A100 win concentrating.
    gpu="A100",
    volumes={"/results": volume},
    timeout=2 * 60 * 60,
)
def eval_remote(run_dir_name: str, multi_event: bool = False):
    """Re-run the end-of-run eval for a run dir already on the volume
    (recovery for trainings whose final eval died, e.g. the 2026-07-06
    d=64 OOMs before final_eval chunked its draw). `multi_event=True` is
    the --compare-multi-event probe: same checkpoint and draw protocol,
    matching step instead of one-event, artefacts to eval_multi_event/."""
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.run import eval_only

    eval_only(Path("/results") / run_dir_name, multi_event=multi_event)
    volume.commit()


@app.function(gpu="A100", volumes={"/results": volume}, timeout=60 * 60)
def scout_remote(run_dir_name: str, D: int, head_kind: str = "mask_one"):
    """Run the Euler-budget scout (`scout_euler_budget.from_checkpoint`) on a
    trained checkpoint already on the volume; writes /results/scout."""
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.scout_euler_budget import from_checkpoint

    ckpt = Path("/results") / run_dir_name / "checkpoints" / "final.pt"
    from_checkpoint(ckpt, D, head_kind, "/results/scout")
    volume.commit()


@app.function(gpu="L4", timeout=60 * 60)
def bench_remote(argv: str = ""):
    """Run the profile/benchmark harness on the L4. `argv` is the
    space-separated profile_swap CLI string, e.g.
    "--mode eval --d 64 --batch 256 --n-euler-steps 128"."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.profile_swap import main as bench_main

    bench_main(argv.split())


@app.local_entrypoint()
def bench(argv: str = ""):
    """Local CLI entry for the profiling harness: blocking so the timing
    tables stream back to the local terminal (dev box is CPU-only)."""
    bench_remote.remote(argv=argv)


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 42, head_kind: str = "", smoke: bool = False):
    """Local CLI entry: blocking single `train_remote` call (used for smoke
    checks)."""
    _validate_cfg_name(cfg_name)
    resolved_head_kind = _resolve_head_kind(head_kind)
    train_remote.remote(
        cfg_name=cfg_name, seed=seed, head_kind=resolved_head_kind, smoke=smoke
    )


@app.local_entrypoint()
def gate(seeds: str = "42,43,44", n_samples: int = 5000, skip_controls: bool = False):
    """Local CLI entry for the gate: blocking `.remote()` so the per-run
    progress prints stream back to the local terminal."""
    gate_remote.remote(seeds=seeds, n_samples=n_samples, skip_controls=skip_controls)


@app.local_entrypoint()
def demo(seeds: str = "42,43,44", n_samples: int = 5000, n_replicates: int = 5):
    """Blocking local CLI entry for the 4x4 demo GPU stage (per-cell progress
    prints stream back to the local terminal)."""
    demo_remote.remote(
        seeds=seeds, n_samples=n_samples, n_replicates=n_replicates
    )


@app.local_entrypoint()
def phihist(seeds: str = "42,43,44", n_samples: int = 5000):
    """Blocking local CLI entry for the phi-histogram stage."""
    phi_hist_remote.remote(seeds=seeds, n_samples=n_samples)


@app.local_entrypoint()
def evalonly(run_dirs: str, multi_event: bool = False):
    """Spawn eval-only recovery over comma-separated run dir names on the
    volume (fire-and-forget: launch with --detach)."""
    names = [n.strip() for n in run_dirs.split(",") if n.strip()]
    for name in names:
        eval_remote.spawn(run_dir_name=name, multi_event=multi_event)
    print(f"spawned {len(names)} eval-only jobs (multi_event={multi_event}): {names}")


@app.local_entrypoint()
def scout(run_dir: str, d_side: int, head_kind: str = "mask_one"):
    """Blocking Euler-budget scout on a trained checkpoint (streams the
    report table back to the local terminal)."""
    scout_remote.remote(run_dir_name=run_dir, D=d_side, head_kind=head_kind)


@app.local_entrypoint()
def batch_seeds(cfg_name: str, seeds: str = "42", head_kind: str = ""):
    """Spawn one hard-constraint config across multiple seeds in parallel."""
    _validate_cfg_name(cfg_name)
    resolved_head_kind = _resolve_head_kind(head_kind)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    for seed in seed_list:
        train_remote.spawn(cfg_name=cfg_name, seed=seed, head_kind=resolved_head_kind)
    print(f"spawned {len(seed_list)} jobs for {cfg_name}: seeds={seed_list}")


@app.local_entrypoint()
def ladder(seeds: str = "42,43,44", head_kind: str = ""):
    """Spawn all three sigma-ladder cells (subcritical/critical/supercritical)
    across the given seeds in parallel -- 9 jobs at the default seeds."""
    resolved_head_kind = _resolve_head_kind(head_kind)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    spawned = []
    for cfg_name in LADDER_CFGS:
        for seed in seed_list:
            train_remote.spawn(
                cfg_name=cfg_name, seed=seed, head_kind=resolved_head_kind
            )
            spawned.append((cfg_name, seed))
    print(f"spawned {len(spawned)} jobs across {LADDER_CFGS}: seeds={seed_list}")
