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
import os
import time

import modal
from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.run import HEAD_KINDS

PROJECT_DIR = "/repo"
APP_NAME = "dnfs-hard"
PIXI_ENV_BIN = f"{PROJECT_DIR}/.pixi/envs/cuda/bin"

# The sigma-ladder cells that `ladder()` fans out over: same D=16,
# c_target=0.5, leTF, doubly_hollow head; sigma is the only thing that
# varies (subcritical / critical / supercritical, operating sigma_c = 0.22305).
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
            ".claude/**",  # agent worktrees carry their own .pixi
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
        # small cuda_utils.c with the system gcc, and the locked pixi env
        # ships no CUDA dev headers (probe s59: no cuda.h anywhere in the
        # image). The runtime wheel carries include/cuda.h; installed --no-deps
        # so the locked env's torch/nvidia libs are untouched, and CPATH below
        # puts the header on gcc's search path.
        # (env python called directly: `pixi run` would re-validate the cuda
        # virtual package, which no build container can satisfy; the env
        # ships no pip, so ensurepip bootstraps it first)
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
            # B5 (optimisation decision, 2026-08-24): venue parity with the
            # DoC sbatch scripts, which export this. Allocator headroom on
            # the 40 GB Modal A100s, not a speed lever.
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
    # the win concentrates in the eval slices. (bench_remote moved to A100
    # too on 2026-07-23, so eval-cost numbers match production hardware.)
    # 80GB spelled out 2026-08-14: Modal's bare "A100" is the 40 GB variant,
    # and the d=256 cells were sized on the DoC cluster's 80 GB a100s. A
    # 40 GB card here is not hardware-matched to any archived d=256 run and
    # will OOM the enlarged-rollout knobs (c_t_batch=512 peaks ~20 GB on top
    # of training state). Small-lattice apps elsewhere in the repo keep the
    # cheaper default deliberately.
    # DNFS_TRAIN_GPU (2026-08-26): launch-time venue override, read where
    # `modal run` executes. A100-80GB stays the default; the d64 w2 mo
    # migration runs A100-40GB (sm_80-identical to the archived namesakes,
    # which trained on 40 GB Modal A100s) and d256 follow-on seeds may pass
    # H100/H200 — record the card in the run's provenance when overridden.
    gpu=os.environ.get("DNFS_TRAIN_GPU", "A100-80GB"),
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_remote(
    cfg_name: str,
    seed: int = 42,
    head_kind: str | None = None,
    smoke: bool = False,
    tag: str = "",
):
    """Run a single hard-constraint training config on Modal.

    `tag` is minted ONCE at spawn time by the local entrypoints: a Modal
    preemption retry re-runs this function with identical inputs, so a stable
    tag makes the retry land in the same run dir and resume from
    checkpoints/resume.pt instead of training from scratch (the 2026-07-23
    MO 100k recall restarted from step 0 for want of exactly this).
    `volume.commit` rides along as the checkpoint hook so resume state is on
    the volume even if a preemption skips the death-flush."""
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

    train(
        cfg,
        seed=seed,
        output_dir="/results",
        tag=tag or None,
        on_checkpoint=volume.commit,
    )
    volume.commit()


@app.function(
    # Same L4 as train_remote: the gate re-loads the trained d=16 heads and
    # draws 5k-sample swap-CTMC evals -- tiny on GPU, ~2h40m on local CPU.
    gpu="L4",
    volumes={"/results": volume},
    timeout=4 * 60 * 60,
)
def gate_remote(
    seeds: str = "42,43,44", n_samples: int = 5000, skip_controls: bool = False,
    cells: str = "", out: str = "/results/gate_4x4",
):
    """Run the 4x4 exact-enumeration go/no-go gate against the trained run
    dirs already on the volume; writes verdict.json + plot to /results/gate_4x4."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.gate_4x4 import main as gate_main

    argv = [
        "--results-dir", "/results",
        "--device", "cuda",
        "--out", out,
        "--seeds", seeds,
        "--n-samples", str(n_samples),
    ]
    if skip_controls:
        argv.append("--skip-controls")
    if cells:
        argv += ["--cells", cells]
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


def _resolve_multi_event_trit(multi_event: int):
    """Map the CLI-facing trit onto eval_only's three-valued multi_event.

    Modal's CLI has no way to pass None, and eval_only's None is the value
    that matters most: it defers to the cell's own canonical trajectory
    step, which is what makes recovery evals and grid probes land in the
    canonical eval/ (or eval_ne<k>/) dirs that frozen comparisons read.
    A bool default therefore cannot work — False would silently force the
    one-event step on a matching-step cell — and the `or None` trick the
    float/int sentinels use would make an explicit one-event probe
    (False) unrequestable. Hence a trit: -1 -> None (canonical, default),
    0 -> False (force one-event), 1 -> True (force matching step)."""
    trit_to_multi_event = {-1: None, 0: False, 1: True}
    if multi_event not in trit_to_multi_event:
        raise ValueError(
            f"multi_event must be -1 (cell's canonical step), 0 (force "
            f"one-event) or 1 (force matching step); got {multi_event}"
        )
    return trit_to_multi_event[multi_event]


@app.function(
    # A100-80GB like train_remote: the eval slices are exactly where the
    # perf profile showed the A100 win concentrating, and the d=256 eval
    # chunk sizes were tuned against 80 GB cards (see train_remote).
    gpu="A100-80GB",
    volumes={"/results": volume},
    # 6 h: eval draw time scales linearly with the sampling grid, and the
    # grid-decoupling probe re-draws d=256 checkpoints at up to ne=1024 —
    # ~8x the ~29 min measured at the ne=128 default.
    timeout=6 * 60 * 60,
)
def eval_remote(
    run_dir_name: str, multi_event: int = -1, smc_tau: float = 0.0,
    n_euler_override: int = 0, stage_best: int = -1,
):
    """Re-run the end-of-run eval for a run dir already on the volume
    (recovery for trainings whose final eval died, e.g. the 2026-07-06
    d=64 OOMs before final_eval chunked its draw). `multi_event` is a
    three-state flag because the underlying eval has three behaviours and
    Modal's CLI cannot pass None: -1 (default) defers to the cell's own
    canonical trajectory step, so recovery/probe evals land in the
    canonical eval/ (or eval_ne<k>/) dirs the frozen comparisons read;
    0 forces the one-event step and 1 the matching step, each writing a
    contrast dir (eval_one_event*/ or eval_multi_event*/) when it is the
    non-canonical choice for the cell. A plain bool cannot carry this: an
    explicit False silently forces one-event on a matching-step cell,
    diverting the eval away from its canonical dir. `smc_tau > 0` runs
    the SMC-resampled eval instead (artefacts to eval_smc_tau<τ>/,
    alongside the untouched plain-IS eval/); 0.0 is the "plain eval"
    sentinel — a τ=0 trigger never fires anyway, so the sentinel can't
    collide with a real sweep point. `n_euler_override > 0` re-draws on
    that sampling grid instead of the cell's own (artefacts to
    eval_ne<k>/; the grid-decoupling probe — see run.eval_only); 0 is the
    same can't-collide sentinel. `stage_best >= 0` draws from
    best_stage<k>.pt instead of final.pt (artefacts to eval_stage<k>/,
    the pre-registered checkpoint-selection read); -1 is its sentinel,
    since stage 0 is a real stage and cannot serve as one."""
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.run import eval_only

    eval_only(
        Path("/results") / run_dir_name,
        multi_event=_resolve_multi_event_trit(multi_event),
        smc_tau=smc_tau or None,
        n_euler_override=n_euler_override or None,
        stage_best=None if stage_best < 0 else stage_best,
    )
    volume.commit()


@app.function(
    # Same A100-80GB class as eval_remote: this draws the eval's own sample
    # count on the eval's own grid, and additionally holds the full
    # (n_euler_steps + 1, chunk, d) trajectory, so it is strictly heavier
    # than the eval it mirrors.
    gpu="A100-80GB",
    volumes={"/results": volume},
    timeout=6 * 60 * 60,
)
def transport_decomposition_remote(run_dir_name: str, n_samples: int = 0):
    """Split a run's bond-correlation transport into gross vs net.

    The eval reports only the NET endpoint gap closed, which cannot tell a
    sampler whose swaps are individually small (a TARGETING limit, fixed in
    the architecture) from one whose swaps are large but undo each other (a
    CANCELLATION limit, fixed in the rate field). This draws trajectories with
    `return_all_states=True` and accumulates both, writing
    transport_decomposition.json beside the run. It never touches eval/: the
    trajectory mode withholds importance weights by construction, so this
    draw is a diagnostic and is not an eval."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.analysis_transport_decomposition import (
        decompose_run,
    )

    run_dir = Path("/results") / run_dir_name
    result = decompose_run(run_dir, n_samples=n_samples or None)
    print(json.dumps(result, indent=2))
    (run_dir / "transport_decomposition.json").write_text(
        json.dumps(result, indent=2)
    )
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


@app.function(gpu="L4", volumes={"/results": volume}, timeout=60 * 60)
def mdns_gate_remote(argv: str = ""):
    """Run the budget-masked MDNS gate driver remotely. `argv` is the
    space-separated mdns_budget_gate_4x4 CLI string; point --results-dir
    inside /results so the run dirs and verdict JSON persist on the
    volume. L4 deliberately: the gate's MLP is tiny and its bands are
    statistical, not device-paired."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.mdns_budget_gate_4x4 import (
        main as mdns_gate_main,
    )

    mdns_gate_main(argv.split())
    volume.commit()


@app.local_entrypoint()
def mdns_gate(argv: str = ""):
    """Blocking local CLI entry so the per-arm progress prints stream
    back to the local terminal."""
    mdns_gate_remote.remote(argv=argv)


@app.function(gpu="A100-80GB", timeout=2 * 60 * 60)
def bench_remote(argv: str = ""):
    """Run the profile/benchmark harness on the production GPU. `argv` is
    the space-separated profile_swap CLI string, e.g.
    "--mode eval --d 64 --batch 256 --n-euler-steps 128". Several
    configurations separated by ";" run back to back in the one container,
    so a whole head ladder pays the cold start once.

    A100-80GB is spelled out deliberately: Modal's bare "A100" is the 40 GB
    variant, which OOMs the large-batch arms this harness exists to measure
    (masked_attention peaks 5.0 GB at d=256 B=32, so a B=512 arm wants
    ~80 GB). It also matches the DoC cluster's a100 partition, so benched
    costs stay comparable to the recorded run wall-clocks — which is the
    whole point of benching on production hardware."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.profile_swap import main as bench_main

    for one in argv.split(";"):
        bench_main(one.split())
        print(flush=True)


@app.local_entrypoint()
def bench(argv: str = ""):
    """Local CLI entry for the profiling harness: blocking so the timing
    tables stream back to the local terminal (dev box is CPU-only)."""
    bench_remote.remote(argv=argv)


@app.function(
    # A100-80GB, spelled out: Modal's bare "A100" is the 40 GB variant, and
    # every other function in this file inherits that 40 GB default. The
    # distinction bites on d=256 batch sizing — a masked-attention forward
    # peaks 5.0 GB at B=32, so a B=512 arm wants ~80 GB and silently OOMs a
    # 40 GB card. Measured on Modal 2026-08-14; the DoC cluster's a100
    # partition is 80 GB, so cluster-tuned batch sizes do NOT transfer to a
    # bare gpu="A100" Modal function.
    # The dev Mac's MPS is the wrong home for this entirely: one d=256 arm
    # runs about an hour there and starves the machine.
    gpu="A100-80GB",
    volumes={"/results": volume},
    timeout=4 * 60 * 60,
)
def resolution_sweep_remote(argv: str = ""):
    """Run the n_euler resolution sweep against a checkpoint staged on the
    volume. `argv` is the space-separated CLI string, e.g.
    "--run-dir /results/<run> --n-euler 128,256,384 --n-draws 512"."""
    import sys

    sys.path.insert(0, "/repo")
    from scripts.n_euler_resolution_sweep import main as sweep_main

    sys.argv = ["n_euler_resolution_sweep", *argv.split()]
    sweep_main()


@app.local_entrypoint()
def resolution_sweep(argv: str = ""):
    """Local CLI entry: blocking so the sweep table streams back."""
    resolution_sweep_remote.remote(argv=argv)


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 42, head_kind: str = "", smoke: bool = False):
    """Local CLI entry: blocking single `train_remote` call (used for smoke
    checks)."""
    _validate_cfg_name(cfg_name)
    resolved_head_kind = _resolve_head_kind(head_kind)
    train_remote.remote(
        cfg_name=cfg_name, seed=seed, head_kind=resolved_head_kind, smoke=smoke,
        tag=time.strftime("%Y%m%d-%H%M%S"),
    )


@app.local_entrypoint()
def gate(
    seeds: str = "42,43,44", n_samples: int = 5000, skip_controls: bool = False,
    cells: str = "", out: str = "/results/gate_4x4",
):
    """Local CLI entry for the gate: blocking `.remote()` so the per-run
    progress prints stream back to the local terminal. `cells` = comma-
    separated CONFIGS names to gate instead of the dh ladder (pass `out` too
    so the ladder verdict is not overwritten)."""
    gate_remote.remote(
        seeds=seeds, n_samples=n_samples, skip_controls=skip_controls,
        cells=cells, out=out,
    )


@app.function(gpu="A100-80GB", volumes={"/results": volume}, timeout=60 * 60)
def residue_probe_remote(run_dirs: str, n_states: int = 256):
    """Forward-only compile-vs-eager residue probe on trained checkpoints
    already on the volume (s70 HOLD mechanism leg; method and instruments
    in compile_residue_probe.py). A100-80GB deliberately: the question is
    whether TRAINING-VENUE inductor numerics enlarge the factorised head's
    cancellation residue, so the probe must run on the training GPU class —
    a CPU or L4 read answers a different question."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.compile_residue_probe import (
        main as probe_main,
    )

    probe_main([
        "--results-dir", "/results", "--run-dirs", run_dirs,
        "--device", "cuda", "--n-states", str(n_states),
        "--out", "/results/compile_residue_probe/report.json",
    ])
    volume.commit()


@app.local_entrypoint()
def residue_probe(run_dirs: str, n_states: int = 256):
    """Blocking local CLI entry so the per-checkpoint reports stream back."""
    residue_probe_remote.remote(run_dirs=run_dirs, n_states=n_states)


@app.function(gpu="A100-80GB", timeout=45 * 60)
def compile_gate_remote():
    """GPU-stack compile certification gate (A1, s60): the s59 CPU-passed
    gate re-run once on the training venue's stack, because inductor
    generates different kernels per backend. Raises on failure so the
    calling entrypoint fails loudly."""
    from experiments.constrained_hard_03.compile_gate import main as gate_main

    if gate_main() != 0:
        raise RuntimeError("compile gate FAILED on the GPU stack")


@app.local_entrypoint()
def compile_gate():
    """Blocking local CLI entry for the GPU-stack compile gate."""
    compile_gate_remote.remote()


@app.function(gpu="A100-80GB", timeout=60 * 60)
def compile_profile_remote(cfg_name: str = "", microbatch: int = 128,
                           rollout_batch: int = 512, rollout_steps: int = 16):
    """Post-compile region profile of the production swap stack (method and
    region list in compile_profile.py). Prints tables; nothing on the
    volume."""
    from experiments.constrained_hard_03.compile_profile import (
        THP2_CELL,
        run_profile,
    )

    run_profile(cfg_name=cfg_name or THP2_CELL, microbatch=microbatch,
                rollout_batch=rollout_batch, rollout_steps=rollout_steps)


@app.local_entrypoint()
def compile_profile(cfg_name: str = "", microbatch: int = 128,
                    rollout_batch: int = 512, rollout_steps: int = 16):
    """Blocking local CLI entry for the post-compile profile."""
    compile_profile_remote.remote(
        cfg_name=cfg_name, microbatch=microbatch,
        rollout_batch=rollout_batch, rollout_steps=rollout_steps,
    )


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
def evalonly(
    run_dirs: str, multi_event: int = -1, smc_tau: float = 0.0,
    n_euler_override: int = 0,
):
    """Spawn eval-only recovery over comma-separated run dir names on the
    volume (fire-and-forget: launch with --detach). `multi_event` is a
    trit, because Modal's CLI cannot pass None and the eval has three
    behaviours: -1 (default) uses each cell's own canonical trajectory
    step so artefacts land in the canonical eval/ (or eval_ne<k>/) dirs;
    0 forces the one-event step; 1 forces the matching step (the
    non-canonical choice writes a contrast dir instead). `smc_tau > 0`
    runs the SMC-resampled eval variant instead of the plain-IS one;
    `n_euler_override > 0` the grid-decoupling probe (eval_ne<k>/)."""
    _resolve_multi_event_trit(multi_event)  # fail fast locally on bad values
    names = [n.strip() for n in run_dirs.split(",") if n.strip()]
    for name in names:
        eval_remote.spawn(
            run_dir_name=name, multi_event=multi_event, smc_tau=smc_tau,
            n_euler_override=n_euler_override,
        )
    print(
        f"spawned {len(names)} eval-only jobs "
        f"(multi_event={multi_event}, smc_tau={smc_tau}, "
        f"n_euler_override={n_euler_override}): {names}"
    )


@app.local_entrypoint()
def scout(run_dir: str, d_side: int, head_kind: str = "mask_one"):
    """Blocking Euler-budget scout on a trained checkpoint (streams the
    report table back to the local terminal)."""
    scout_remote.remote(run_dir_name=run_dir, D=d_side, head_kind=head_kind)


@app.local_entrypoint()
def batch_seeds(cfg_name: str, seeds: str = "42", head_kind: str = "", tag: str = ""):
    """Spawn one hard-constraint config across multiple seeds in parallel.

    `tag` defaults to a launch timestamp; pass a fixed campaign tag (e.g.
    20260825-hard-w2) so every cell of a wave lands under one label and a
    resubmission after preemption resumes into the same run dirs."""
    _validate_cfg_name(cfg_name)
    resolved_head_kind = _resolve_head_kind(head_kind)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    for seed in seed_list:
        train_remote.spawn(
            cfg_name=cfg_name, seed=seed, head_kind=resolved_head_kind, tag=tag
        )
    print(f"spawned {len(seed_list)} jobs for {cfg_name}: seeds={seed_list} tag={tag}")


@app.local_entrypoint()
def ladder(seeds: str = "42,43,44", head_kind: str = ""):
    """Spawn all three sigma-ladder cells (subcritical/critical/supercritical)
    across the given seeds in parallel -- 9 jobs at the default seeds."""
    resolved_head_kind = _resolve_head_kind(head_kind)
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    tag = time.strftime("%Y%m%d-%H%M%S")
    spawned = []
    for cfg_name in LADDER_CFGS:
        for seed in seed_list:
            train_remote.spawn(
                cfg_name=cfg_name, seed=seed, head_kind=resolved_head_kind, tag=tag
            )
            spawned.append((cfg_name, seed))
    print(f"spawned {len(spawned)} jobs across {LADDER_CFGS}: seeds={seed_list}")
