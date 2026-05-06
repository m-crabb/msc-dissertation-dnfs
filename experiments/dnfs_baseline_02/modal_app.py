"""Modal wrapper for DNFS Ising baseline training. Delegates the actual
training to `run.train` so the remote and local code paths share a single
implementation.

Usage (after `modal token new` and `modal secret create wandb-secret ...`):
    pixi run -e dev modal run \\
        experiments.dnfs_baseline_02.modal_app::train_remote \\
        --cfg-name stage_1_d10 --seed 0
"""
import modal


# Build the container image from the project's pyproject.toml so deps are
# locked to whatever the local pixi env tracks. The repo itself is mounted
# at /repo so imports resolve identically to a local checkout.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir(".", remote_path="/repo")
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
    gpu="A100",
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
