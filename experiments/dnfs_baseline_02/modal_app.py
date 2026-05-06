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
    pixi run -e dev modal run -m \\
        experiments.dnfs_baseline_02.modal_app::train_remote \\
        --cfg-name stage_1_d4_xl --seed 0
"""
import modal

PROJECT_DIR = "/repo"
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

app = modal.App("dnfs-baseline", image=image)


@app.function(
    # L4 is roughly the same wall-clock as A100 for Stage 1's MLP-bound
    # workload, at about half the cost. Reconsider for Stage 3 (LeT) where
    # attention layers benefit more from A100/H100-class compute.
    gpu="L4",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    # 24h is generous for D=10; tighten if cost matters.
    timeout=24 * 60 * 60,
)
def train_remote(cfg_name: str, seed: int = 0):
    """Run a single training config on Modal and persist artefacts to the
    volume. Importable but typically launched via `modal run`."""
    import sys

    # /repo is the mount point of `add_local_dir`. Inserting it onto sys.path
    # lets `experiments.dnfs_baseline_02.run` import resolve correctly.
    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_02.run import train

    train(cfg_name, seed=seed, output_dir="/results")
    # commit() makes the artefacts visible to subsequent `modal volume get`
    # calls. Without this, the volume is reverted on container shutdown.
    volume.commit()


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 0):
    """Local CLI entry: spawns `train_remote` as a remote Modal call."""
    train_remote.remote(cfg_name=cfg_name, seed=seed)
