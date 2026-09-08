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

    # Multi-seed packed onto one card (wall-clock from packed runs is
    # contention-contaminated, see train_pack_remote):
    pixi run -e dev modal run --detach -m \\
        experiments.constrained_soft_02.modal_app::batch_seeds_packed \\
        --cfg-name S2_d8_c0250_l50_letf_ne128_house --seeds "42,43,44,45" \\
        --tag 20260831-softhouse-d64
"""

import time

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
        "pixi install --environment cuda --locked",
        # torch.compile header fix, as in the hard app: the
        # runtime wheel carries include/cuda.h for inductor's gcc step;
        # --no-deps leaves the locked env's torch/nvidia libs untouched.
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
            # Venue parity with the DoC sbatch scripts, which export this.
            # Allocator headroom, not a speed lever.
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
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


@app.function(
    # Pinned to the 80GB SKU rather than the bare `gpu="A100"` (= 40GB), which
    # Modal may serve from more than one card variant. Not for the memory: the
    # dominant nuisance channel here is cross-device FP non-determinism --
    # `seeding.py` sets `cudnn.deterministic` but never
    # `torch.use_deterministic_algorithms`, and leTF's two `nn.Embedding`
    # backwards use order-dependent atomic scatter-adds, so a 2-ULP step-0
    # gradient difference decorrelates a 50k run entirely. Identical config +
    # seed measured ESS/N 0.423 vs 0.899 across venues, the size of the effects
    # these cells must resolve, so a free SKU would let that channel vary within
    # a seed-replicate comparison. `train` records `get_device_name(0)` so
    # homogeneity stays checkable after the fact.
    gpu="A100-80GB",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_remote(cfg_name: str, seed: int = 42, tag: str = ""):
    """Run a single constrained-soft training config on Modal.

    `tag` replaces the run dir's timestamp suffix (see `run.train`); "" is the
    "no tag" sentinel because Modal's CLI cannot pass None. It keeps a
    seed-replicate family greppable as one unit (several cells have more than
    one archived dir at the same seed), and a preemption re-runs this function
    with identical inputs, so a stable tag lands the retry in the same run dir,
    where `checkpoints/resume.pt` continues from the last outer-cycle boundary
    instead of step 0.
    """
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_soft_02.configs import CONFIGS
    from experiments.dnfs_baseline_01.run import train

    train(
        CONFIGS[cfg_name],
        seed=seed,
        output_dir="/results",
        tag=tag or None,
        # A preemption gets no chance to flush, so the resume checkpoint is
        # committed to the volume the moment it is written; otherwise the retry
        # finds nothing and restarts from step 0.
        on_checkpoint=volume.commit,
    )
    volume.commit()


@app.function(
    # Same SKU pin as train_remote, same reason: the determinism channel is
    # device class, and packing seeds onto one card does not vary it.
    gpu="A100-80GB",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_pack_remote(cfg_name: str, seeds: str, tag: str = ""):
    """Run several seeds of one config concurrently on the one rented card.

    Modal cannot cohabit containers on a GPU, so co-residency happens inside the
    container: one subprocess per seed sharing the A100 this function rents. At
    d64 the leTF cells are small and launch-bound, so four co-resident runs
    overlap well and the pack cuts the bill ~4x while keeping the SKU pin.

    Two accepted costs. Wall-clock columns from packed runs are
    contention-contaminated and must not be quoted; ESS, fidelity and the
    analytic FLOP/es (the house table's cost column) are untouched. And a
    preemption interrupts every co-resident seed at once; the fixed tag plus the
    commit loop below make the retry resume each seed from its last outer-cycle
    boundary rather than step 0.
    """
    import subprocess
    import sys
    import time as time_module

    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    procs = {}
    for seed in seed_list:
        procs[seed] = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.constrained_soft_02.run",
                "--cfg",
                cfg_name,
                "--seed",
                str(seed),
                "--output-dir",
                "/results",
                *(["--tag", tag] if tag else []),
            ],
            cwd=PROJECT_DIR,
        )
    # The subprocess CLI cannot pass on_checkpoint=volume.commit, so the
    # parent commits on a timer instead: resume.pt lands on the volume
    # within a minute of being written, close enough to the single-run
    # per-checkpoint granularity for preemption recovery.
    while any(p.poll() is None for p in procs.values()):
        time_module.sleep(60)
        volume.commit()
    volume.commit()
    failed = {seed: p.returncode for seed, p in procs.items() if p.returncode != 0}
    if failed:
        raise RuntimeError(f"packed seeds failed (seed: exit): {failed}")


@app.function(
    # Same SKU pin as train_remote: the Richardson pair combines two
    # independent draws, so venue is not a within-pair confound, but one SKU
    # keeps every drawn number on one device class.
    gpu="A100-80GB",
    volumes={"/results": volume},
    timeout=60 * 60,
)
def redraw_remote(run_dir_name: str, n_euler: int):
    """One eval-grid redraw against a volume run dir (run.eval_only).

    Writes eval_ne<k>/ beside the frozen eval/ (which stays byte-untouched;
    see eval_only). redraw_seed=45 is the grid-offset convention every
    side-grid F(c) draw shares.
    Skip-if-exists makes a re-run of the batch idempotent, mirroring the
    DoC sbatch this replaces. Needs config.json, checkpoints/final.pt and
    training_log.csv in the run dir (the trailing-ESS block reads the log).
    """
    import sys

    sys.path.insert(0, "/repo")
    from pathlib import Path

    from experiments.dnfs_baseline_01.run import eval_only

    run_dir = Path("/results") / run_dir_name
    if (run_dir / f"eval_ne{n_euler}" / "metrics.json").exists():
        print(f"skip {run_dir_name} ne{n_euler} (exists)")
        return
    eval_only(run_dir, redraw=True, redraw_seed=45, n_euler_override=n_euler)
    volume.commit()


@app.function(
    # Same SKU pin as train_remote: sweep rows are compared seed-to-seed
    # across the four camort replicates, so the device class stays fixed
    # within the comparison even though the training itself ran on DoC A30s
    # (that venue boundary is unavoidable — the checkpoints are imports).
    gpu="A100-80GB",
    volumes={"/results": volume},
    timeout=60 * 60,
)
def sweep_remote(run_dir_name: str, checkpoint: str = "final.pt", force: bool = False):
    """Per-composition request-grid sweep of one amortised run dir on the
    volume (run.composition_sweep; one CRN-paired eval row per requested c).

    `checkpoint` keys the artefact dir as composition_sweep does: final.pt ->
    eval/composition_sweep.json, final_ema.pt -> eval_ema/composition_sweep.json.
    Skip-if-exists makes a batch re-run idempotent; `force` redoes a recorded
    sweep, for sweeps filed before the per-composition frames were kept. Needs
    config.json and checkpoints/<checkpoint> in the run dir.
    """
    import sys

    sys.path.insert(0, "/repo")
    from pathlib import Path

    from experiments.dnfs_baseline_01.run import composition_sweep

    run_dir = Path("/results") / run_dir_name
    sweep_dir = "eval_ema" if checkpoint == "final_ema.pt" else "eval"
    if (run_dir / sweep_dir / "composition_sweep.json").exists() and not force:
        print(f"skip {run_dir_name} {checkpoint} (sweep exists)")
        return
    composition_sweep(run_dir, checkpoint=checkpoint)
    volume.commit()


@app.local_entrypoint()
def sweep_batch(
    run_dirs: str, checkpoints: str = "final.pt,final_ema.pt", force: bool = False
):
    """Fan out sweep_remote over run_dirs x checkpoints, one container per
    sweep (both args comma-separated; Modal's CLI takes strings)."""
    names = [s.strip() for s in run_dirs.split(",") if s.strip()]
    ckpts = [c.strip() for c in checkpoints.split(",") if c.strip()]
    calls = [
        sweep_remote.spawn(run_dir_name=name, checkpoint=ckpt, force=force)
        for name in names
        for ckpt in ckpts
    ]
    print(f"spawned {len(calls)} sweeps: {len(names)} run dirs x {ckpts}")
    for call in calls:
        call.get()


@app.function(
    # Smallest card Modal serves: D=4 gate runs only. A gate is a single-seed
    # pass/fail against a coarse bar (~0.99 vs collapse), so the SKU homogeneity
    # the A100-80GB pin buys is not part of the measurement. Never route a cell
    # that will be compared seed-to-seed against A100 runs through here.
    gpu="T4",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=6 * 60 * 60,
)
def train_gate_remote(cfg_name: str, seed: int = 42, tag: str = ""):
    """train_remote's body on a small card — see the gpu comment for scope."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_soft_02.configs import CONFIGS
    from experiments.dnfs_baseline_01.run import train

    train(
        CONFIGS[cfg_name],
        seed=seed,
        output_dir="/results",
        tag=tag or None,
        on_checkpoint=volume.commit,
    )
    volume.commit()


@app.function(
    # Small-card diagnostics only (same scope logic as the T4 gate above):
    # pass/fail against a coarse bar, here "trains vs the 0.001-0.01 dead
    # floor". L4 not T4 because a d64 50k cell needs headroom under the timeout.
    # Never route a cell that will be compared seed-to-seed against A100 runs
    # through here; `train` records the card name.
    gpu="L4",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=12 * 60 * 60,
)
def train_diag_remote(cfg_name: str, seed: int = 42, tag: str = ""):
    """train_remote's body on a small card — see the gpu comment for scope."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_soft_02.configs import CONFIGS
    from experiments.dnfs_baseline_01.run import train

    train(
        CONFIGS[cfg_name],
        seed=seed,
        output_dir="/results",
        tag=tag or None,
        on_checkpoint=volume.commit,
    )
    volume.commit()


@app.local_entrypoint()
def batch_seeds_diag(cfg_name: str, seeds: str = "42", tag: str = ""):
    """Spawn train_diag_remote (L4, diagnostics scope) across seeds."""
    _validate_cfg_name(cfg_name)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    for seed in seed_list:
        train_diag_remote.spawn(cfg_name=cfg_name, seed=seed, tag=tag)
    print(
        f"spawned {len(seed_list)} diag jobs for {cfg_name}: "
        f"seeds={seed_list} tag={tag}"
    )


@app.local_entrypoint()
def redraw_batch(run_dirs: str, grids: str):
    """Fan out redraw_remote over run_dirs x grids, one container per eval.

    Both args comma-separated (Modal's CLI takes strings): every named run
    dir is drawn on every grid. Windows with a different grid pair (the
    ne64-trained c=0.30 fallback) go in a second invocation.
    """
    names = [s.strip() for s in run_dirs.split(",") if s.strip()]
    n_eulers = [int(g.strip()) for g in grids.split(",") if g.strip()]
    calls = [
        redraw_remote.spawn(run_dir_name=name, n_euler=n_euler)
        for name in names
        for n_euler in n_eulers
    ]
    for call in calls:
        call.get()


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 42):
    """Local CLI entry: spawns `train_remote` as a remote Modal call.

    The tag is minted here, once, rather than defaulted inside the container: a
    preemption re-runs `train_remote` with identical inputs, so a tag fixed at
    spawn time lands the retry in the same run dir and lets it resume, while a
    container-side timestamp would mint a fresh sibling dir and start over.
    `batch_seeds` takes the tag from the caller for the same reason.
    """
    _validate_cfg_name(cfg_name)
    train_remote.remote(
        cfg_name=cfg_name,
        seed=seed,
        tag=time.strftime("%Y%m%d-%H%M%S"),
    )


@app.local_entrypoint()
def batch_seeds(cfg_name: str, seeds: str = "42", tag: str = ""):
    """Spawn one constrained config across multiple seeds in parallel."""
    _validate_cfg_name(cfg_name)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    for seed in seed_list:
        train_remote.spawn(cfg_name=cfg_name, seed=seed, tag=tag)
    print(
        f"spawned {len(seed_list)} jobs for {cfg_name}: seeds={seed_list} tag={tag or '<timestamp>'}"
    )


@app.local_entrypoint()
def batch_seeds_packed(cfg_name: str, seeds: str = "42", tag: str = ""):
    """One container, all seeds co-resident (see train_pack_remote)."""
    _validate_cfg_name(cfg_name)
    train_pack_remote.spawn(cfg_name=cfg_name, seeds=seeds, tag=tag)
    print(
        f"spawned packed container for {cfg_name}: "
        f"seeds={seeds} tag={tag or '<timestamp>'}"
    )
