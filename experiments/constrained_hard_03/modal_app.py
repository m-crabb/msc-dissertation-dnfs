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
    .env({"PATH": f"{PIXI_ENV_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
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
    # d=16, hidden_dim<=32 is tiny -- the A100 in the baseline app is for
    # d=100 attention-bound runs. Bump per-launch if L4 proves slow.
    gpu="L4",
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
