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
        --cfg-name S2_d4_c03_l50 --seed 42

    # Multi-seed for d=4:
    pixi run -e dev modal run --detach -m \\
        experiments.constrained_soft_02.modal_app::batch_seeds \\
        --cfg-name S2_d4_c03_l50 --seeds "42,43,44,45"
"""
import modal

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


@app.function(
    gpu="A100",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_remote(cfg_name: str, seed: int = 42):
    """Run a single constrained-soft training config on Modal."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_01.run import train
    from experiments.constrained_soft_02.configs import CONFIGS

    train(CONFIGS[cfg_name], seed=seed, output_dir="/results")
    volume.commit()


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 42):
    """Local CLI entry: spawns `train_remote` as a remote Modal call."""
    train_remote.remote(cfg_name=cfg_name, seed=seed)


@app.local_entrypoint()
def batch_seeds(cfg_name: str, seeds: str = "42"):
    """Spawn one constrained config across multiple seeds in parallel."""
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    for seed in seed_list:
        train_remote.spawn(cfg_name=cfg_name, seed=seed)
    print(f"spawned {len(seed_list)} jobs for {cfg_name}: seeds={seed_list}")
