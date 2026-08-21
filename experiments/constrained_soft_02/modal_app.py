"""Modal wrapper for constrained-soft Ising training. Delegates the actual
training to `experiments.dnfs_baseline_01.run.train` so the remote and
local code paths share a single implementation.

The container image is built by installing pixi inside the container and
running `pixi install --environment dev --locked` against the project's
`pixi.lock`, so the remote runtime stack matches local dev byte-for-byte.

Usage (after `modal token new` and `modal secret create wandb-secret ...`):
    # Single config:
    pixi run -e dev modal run -m \\
        experiments.constrained_soft_02.modal_app::main \\
        --cfg-name S2_d4_c05_l50_letf --seed 42

    # Multi-seed for d=4:
    pixi run -e dev modal run --detach -m \\
        experiments.constrained_soft_02.modal_app::batch_seeds \\
        --cfg-name S2_d4_c05_l50_letf --seeds "42,43,44,45"
"""
import modal

from experiments.constrained_soft_02.configs import CONFIGS

PROJECT_DIR = "/repo"
APP_NAME = "dnfs-constraints"
PIXI_ENV_BIN = f"{PROJECT_DIR}/.pixi/envs/cuda/bin"

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


@app.function(
    # Pinned to the 80GB SKU rather than the bare `gpu="A100"` (= 40GB), which
    # Modal may serve from more than one card variant. Nothing here needs the
    # memory: the reason is that the dominant known nuisance channel on this
    # codebase is cross-device FP non-determinism -- `seeding.py` sets
    # `cudnn.deterministic` but never `torch.use_deterministic_algorithms`, and
    # leTF's two `nn.Embedding` backwards use order-dependent atomic
    # scatter-adds, so a 2-ULP step-0 gradient difference decorrelates a 50k
    # run entirely. Identical config + seed has been measured at ESS/N 0.423 vs
    # 0.899 across venues, which is the size of the effects these cells are
    # asked to resolve. Leaving the SKU free would let that channel vary WITHIN
    # a seed-replicate comparison, i.e. inside the very measurement meant to
    # bound it. `train` records `get_device_name(0)` so homogeneity stays
    # checkable after the fact instead of assumed.
    gpu="A100-80GB",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_remote(cfg_name: str, seed: int = 42, tag: str = ""):
    """Run a single constrained-soft training config on Modal.

    `tag` replaces the run dir's timestamp suffix (see `run.train`); "" is the
    "no tag" sentinel because Modal's CLI cannot pass None. It exists so a
    seed-replicate family is greppable as one unit: several of these cells
    already have more than one archived dir at the same seed, and a bare
    timestamp leaves the analysis picking the right one by date.
    """
    import sys

    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_01.run import train
    from experiments.constrained_soft_02.configs import CONFIGS

    train(CONFIGS[cfg_name], seed=seed, output_dir="/results", tag=tag or None)
    volume.commit()


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 42):
    """Local CLI entry: spawns `train_remote` as a remote Modal call."""
    _validate_cfg_name(cfg_name)
    train_remote.remote(cfg_name=cfg_name, seed=seed)


@app.local_entrypoint()
def batch_seeds(cfg_name: str, seeds: str = "42", tag: str = ""):
    """Spawn one constrained config across multiple seeds in parallel."""
    _validate_cfg_name(cfg_name)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    for seed in seed_list:
        train_remote.spawn(cfg_name=cfg_name, seed=seed, tag=tag)
    print(f"spawned {len(seed_list)} jobs for {cfg_name}: seeds={seed_list} tag={tag or '<timestamp>'}")
