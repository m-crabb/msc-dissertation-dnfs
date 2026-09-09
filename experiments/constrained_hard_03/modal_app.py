"""Modal wrapper for hard-constraint swap-CTMC training. Training itself is
`experiments.constrained_hard_03.run.train`, so remote and local share one
implementation. The image installs pixi against the project's `pixi.lock`
(`pixi install --environment dev --locked`), matching the local dev stack.

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

# Cells `ladder()` fans out over: D=16, c_target=0.5, leTF, doubly_hollow;
# sigma alone varies (sub/critical/supercritical, sigma_c = 0.22305).
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
            ".claude/**",  # tool worktrees carry their own .pixi
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
        # torch.compile's inductor builds cuda_utils.c with the system gcc and
        # the locked env ships no cuda.h; the runtime wheel carries one, added
        # --no-deps so torch/nvidia stay put, with CPATH below pointing gcc at
        # it. Env python directly: `pixi run` would re-validate the cuda virtual
        # package, unsatisfiable in a build container; ensurepip supplies pip.
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
            # Venue parity with the DoC sbatch scripts; allocator headroom on
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
    # L4 (2026-08-30): the GFN 4x4 cells are tiny (d=16, hidden 128, 10k steps)
    # and latency-bound on the sequential 16-step sampler, where the A100's
    # bandwidth buys nothing.
    gpu="L4",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=2 * 60 * 60,
)
def train_gfn_remote(cfg_name: str, seed: int = 42, tag: str = ""):
    """Run a single GFN comparator cell on Modal (small-lattice venue).

    Same stable-tag preemption contract as train_remote."""
    _train_gfn_on_volume(cfg_name, seed, tag)


@app.function(
    # The d256 GFN cells: ~9 h per 100k seed on an A30. Same card family as the
    # archived d256 swap cells; DNFS_TRAIN_GPU overrides as for train_remote.
    gpu=os.environ.get("DNFS_TRAIN_GPU", "A100-80GB"),
    volumes={"/results": volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def train_gfn_remote_a100(cfg_name: str, seed: int = 42, tag: str = ""):
    """train_gfn_remote on the production card, for the d256 cells."""
    _train_gfn_on_volume(cfg_name, seed, tag)


def _train_gfn_on_volume(cfg_name: str, seed: int, tag: str) -> None:
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS
    from experiments.constrained_hard_03.run_gfn import train_gfn

    train_gfn(
        GFN_CONFIGS[cfg_name],
        seed=seed,
        output_dir="/results",
        tag=tag or None,
        on_checkpoint=volume.commit,
    )
    volume.commit()


@app.function(
    # A100 for seed runs (2026-07-06): the workload is bandwidth-bound
    # (layernorm/copies), where the L4 is weakest, and the win concentrates in
    # the eval slices. 80GB spelled out 2026-08-14: Modal's bare "A100" is the
    # 40 GB variant, but the d=256 cells were sized on the DoC cluster's 80 GB
    # a100s and a 40 GB card OOMs the enlarged-rollout knobs (c_t_batch=512
    # peaks ~20 GB on top of training state). DNFS_TRAIN_GPU (2026-08-26) is a
    # launch-time venue override read where `modal run` executes; record the
    # card in the run's provenance when overridden.
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

    `tag` is minted once at spawn time by the local entrypoints: a preemption
    retry re-runs this function with identical inputs, so a stable tag lands it
    in the same run dir and resumes from checkpoints/resume.pt instead of
    training from scratch. `volume.commit` rides the checkpoint hook so resume
    state survives a preemption that skips the death-flush."""
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
    seeds: str = "42,43,44",
    n_samples: int = 5000,
    skip_controls: bool = False,
    cells: str = "",
    out: str = "/results/gate_4x4",
):
    """Run the 4x4 exact-enumeration pass/fail gate against the trained run
    dirs already on the volume; writes verdict.json + plot to /results/gate_4x4."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.probes.gate_4x4 import main as gate_main

    argv = [
        "--results-dir",
        "/results",
        "--device",
        "cuda",
        "--out",
        out,
        "--seeds",
        seeds,
        "--n-samples",
        str(n_samples),
    ]
    if skip_controls:
        argv.append("--skip-controls")
    if cells:
        argv += ["--cells", cells]
    gate_main(argv)
    volume.commit()


@app.function(gpu="L4", volumes={"/results": volume}, timeout=2 * 60 * 60)
def demo_remote(seeds: str = "42,43,44", n_samples: int = 5000, n_replicates: int = 5):
    """GPU stage of the 4x4 demo analysis (2026-07-08): gate fidelity +
    neural replicate estimates for the 10k MA/MO cells; writes
    /results/demo_4x4/neural_estimates.json. L4 suffices at d=16."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.analysis.demo_4x4 import main as demo_main

    demo_main(
        [
            "gpu",
            "--results-dir",
            "/results",
            "--device",
            "cuda",
            "--seeds",
            seeds,
            "--n-samples",
            str(n_samples),
            "--n-replicates",
            str(n_replicates),
            "--out",
            "/results/demo_4x4",
        ]
    )
    volume.commit()


@app.function(gpu="L4", volumes={"/results": volume}, timeout=60 * 60)
def phi_hist_remote(seeds: str = "42,43,44", n_samples: int = 5000):
    """Light GPU stage: pooled IS-weighted phi histograms for the demo cells
    (mode-coverage exhibit); writes /results/demo_4x4/phi_hists.json."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.analysis.demo_4x4 import main as demo_main

    demo_main(
        [
            "phi",
            "--results-dir",
            "/results",
            "--device",
            "cuda",
            "--seeds",
            seeds,
            "--n-samples",
            str(n_samples),
            "--out",
            "/results/demo_4x4",
        ]
    )
    volume.commit()


def _resolve_multi_event_trit(multi_event: int):
    """Map the CLI-facing trit onto eval_only's three-valued multi_event.

    Modal's CLI cannot pass None, and None is the value that matters most: it
    defers to the cell's own canonical trajectory step, so recovery evals and
    grid probes land in the canonical eval/ (or eval_ne<k>/) dirs that frozen
    comparisons read. A bool cannot carry that, since False would silently
    force the one-event step on a matching-step cell. Hence a trit:
    -1 -> None (canonical, default), 0 -> False (force one-event),
    1 -> True (force matching step)."""
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
    run_dir_name: str,
    multi_event: int = -1,
    smc_tau: float = 0.0,
    n_euler_override: int = 0,
    stage_best: int = -1,
    use_ema: bool = False,
):
    """Re-run the end-of-run eval for a run dir already on the volume (recovery
    for trainings whose final eval died, e.g. the 2026-07-06 d=64 OOMs before
    final_eval chunked its draw).

    Sentinels, since Modal's CLI cannot pass None: `multi_event` -1 defers to
    the cell's own canonical trajectory step (see _resolve_multi_event_trit),
    0 forces the one-event step and 1 the matching step, each writing a contrast
    dir (eval_one_event*/ or eval_multi_event*/) when non-canonical for the
    cell. `smc_tau > 0` runs the SMC-resampled eval into eval_smc_tau<τ>/
    beside the untouched plain-IS eval/; a τ=0 trigger never fires, so 0.0
    cannot collide with a real sweep point. `n_euler_override > 0` re-draws on
    that sampling grid into eval_ne<k>/ (the grid-decoupling probe; see
    run.eval_only). `stage_best >= 0` draws best_stage<k>.pt into
    eval_stage<k>/, -1 being its sentinel since stage 0 is a real stage.
    `use_ema` draws from final_ema.pt."""
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
        # Plain `use_ema` (no grid override) is the died-before-landing
        # recovery for a missing eval_ema/ — eval_only refuses it whenever
        # the frozen EMA eval actually exists, so this cannot overwrite one.
        use_ema=use_ema,
    )
    volume.commit()


@app.function(
    # Same A100-80GB class as eval_remote and strictly heavier: it draws the
    # eval's own samples on the eval's own grid while holding the full
    # (n_euler_steps + 1, chunk, d) trajectory.
    gpu="A100-80GB",
    volumes={"/results": volume},
    timeout=6 * 60 * 60,
)
def transport_decomposition_remote(run_dir_name: str, n_samples: int = 0):
    """Split a run's bond-correlation transport into gross vs net.

    Separate small swap effects (targeting) from effects that undo each other
    (cancellation); see probes/transport_decomposition.py for the method.
    `return_all_states=True` withholds importance weights, so this diagnostic
    writes transport_decomposition.json beside the run and leaves eval/ intact.
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.probes.transport_decomposition import (
        decompose_run,
    )

    run_dir = Path("/results") / run_dir_name
    result = decompose_run(run_dir, n_samples=n_samples or None)
    print(json.dumps(result, indent=2))
    (run_dir / "transport_decomposition.json").write_text(json.dumps(result, indent=2))
    volume.commit()


@app.function(gpu="A100", volumes={"/results": volume}, timeout=60 * 60)
def scout_remote(run_dir_name: str, D: int, head_kind: str = "mask_one"):
    """Run the Euler-budget scout (`scout_euler_budget.from_checkpoint`) on a
    trained checkpoint already on the volume; writes /results/scout."""
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.probes.scout_euler_budget import (
        from_checkpoint,
    )

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
    from experiments.constrained_hard_03.probes.mdns_budget_gate_4x4 import (
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
def bench_remote(argv: str = "", isolate: bool = True):
    """Run the profile/benchmark harness on the production GPU. `argv` is the
    space-separated profile_swap CLI string, e.g. "--mode eval --d 64 --batch
    256 --n-euler-steps 128"; several configurations separated by ";" run back
    to back in the one container, so a head ladder pays one cold start.

    `isolate` (the default) gives each configuration a fresh subprocess inside
    that container, because torch state that corrupts a benched row is
    per-process. The dynamo recompile budget (profile_swap._CompileGaveUp) is
    spent per forward code object, which interval / masked_attention / stencil
    share, so four configurations exhaust it and every later one silently runs
    eager -- measured on an A100 2026-08-29 as 26.4 ms / 2.96 GB against
    10.1 ms / 1.76 GB for the identical interval d=256 B=32 row benched first.
    `--tf32` leaks the same way, leaving set_float32_matmul_precision("high")
    on for every following row. Pass isolate=False only to reproduce an
    in-process roster on purpose.

    A100-80GB is spelled out because Modal's bare "A100" is the 40 GB variant,
    which OOMs the large-batch arms this harness exists to measure
    (masked_attention peaks 5.0 GB at d=256 B=32, so a B=512 arm wants ~80 GB);
    it also matches the DoC cluster's a100 partition, keeping benched costs
    comparable to the recorded run wall-clocks."""
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/repo")
    configs = [one.split() for one in argv.split(";")]
    if not isolate:
        from experiments.constrained_hard_03.probes.profile_swap import (
            main as bench_main,
        )

        for one in configs:
            bench_main(one)
            print(flush=True)
        return

    env = {**os.environ, "PYTHONPATH": PROJECT_DIR}
    for one in configs:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.constrained_hard_03.probes.profile_swap",
                *one,
            ],
            cwd=PROJECT_DIR,
            env=env,
            check=True,
        )
        print(flush=True)


@app.local_entrypoint()
def bench(argv: str = "", isolate: bool = True):
    """Local CLI entry for the profiling harness: blocking so the timing
    tables stream back to the local terminal (dev box is CPU-only)."""
    bench_remote.remote(argv=argv, isolate=isolate)


@app.function(gpu="A100-80GB", volumes={"/results": volume}, timeout=2 * 60 * 60)
def training_flops_remote(argv: str = ""):
    """Run the training-FLOP measurement harness (measure_training_flops)
    on A100-80GB, matching d256 training (bare "A100" selects 40 GB).
    `argv` is a space-separated CLI string, e.g.
    "--cfg H2_d256_... --out /results/training_flops_d256_thp2.json".
    Keep --out in /results for persistence and --scratch container-local:
    scratch run dirs are wiped between horizons and must not shadow real runs.
    """
    import sys

    sys.path.insert(0, PROJECT_DIR)
    from experiments.constrained_hard_03.probes.measure_training_flops import (
        main as training_flops_main,
    )

    training_flops_main(argv.split())
    volume.commit()


@app.local_entrypoint()
def training_flops(argv: str = ""):
    """Blocking local CLI entry so the per-horizon progress and the final
    measured/derived comparison stream back to the local terminal."""
    training_flops_remote.remote(argv=argv)


@app.function(
    # A100-80GB, spelled out: Modal's bare "A100" is the 40 GB variant. The
    # distinction bites on d=256 batch sizing — a masked-attention forward
    # peaks 5.0 GB at B=32, so a B=512 arm wants ~80 GB and silently OOMs a
    # 40 GB card (measured on Modal 2026-08-14). The DoC a100 partition is
    # 80 GB, so cluster-tuned batch sizes do not transfer to a bare "A100".
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


@app.function(
    # A100-80GB spelled out for the same reason as resolution_sweep_remote: a
    # bare gpu="A100" is the 40 GB variant, and this holds a (T, B, d)
    # trajectory alongside the head's per-pair activations at d=256.
    gpu="A100-80GB",
    volumes={"/results": volume},
    timeout=4 * 60 * 60,
)
def zero_shot_transfer_remote(argv: str = ""):
    """Zero-shot coupling/composition transfer probe against a checkpoint
    staged on the volume. Samples without gradients, optimisation or new
    weights. `argv` is the space-separated CLI string; see
    `probe_zero_shot_transfer` for derivations and the grid rationale."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.constrained_hard_03.probes.probe_zero_shot_transfer import (
        main as probe_main,
    )

    sys.argv = ["probe_zero_shot_transfer", *argv.split()]
    probe_main()
    volume.commit()


@app.function(gpu="L4", volumes={"/results": volume}, timeout=2 * 60 * 60)
def cuau16_amortised_sweep_remote(argv: str = ""):
    """Roll the 16-site composition-amortised Cu-Au checkpoints out on every
    slice (experiments.alloy_ce.probes.cuau16_amortised_sweep). Minutes on a
    GPU; the Mac measured 115 s per 500 draws per slice."""
    import sys

    sys.path.insert(0, "/repo")
    from experiments.alloy_ce.probes.cuau16_amortised_sweep import main as sweep_main

    sweep_main(argv.split())
    volume.commit()


@app.local_entrypoint()
def zero_shot_transfer(
    seeds: str = "42,43,44",
    run_template: str = (
        "H2_d256_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w3_seed{seed}"
        "_20260826-d256-sc"
    ),
    checkpoint: str = "final_ema.pt",
    compositions: str = "0.5,0.46875,0.4375,0.375,0.3125,0.25,0.625",
    # Exact Euler grid points k/127 for k = 16, 32, 58, 76, 95, 111, 127, so no
    # stop time snaps and every row's coupling is exact. k=58 is the closest the
    # production grid comes to the certified sigma=0.1 reference (0.100629, a
    # 0.6% offset -- fine for ESS, but a correlation comparison there wants a
    # reference regenerated at 0.100629). Composition 0.625 is the Z2 mirror of
    # 0.375: the target family is symmetric under the global flip and thp is not
    # equivariant by construction, so the pair measures the trained head's Z2
    # symmetry. Measured 2026-08-27: symmetric within seed noise.
    stop_times: str = (
        "0.125984252,0.251968504,0.456692913,0.598425197,0.748031496,0.874015748,1.0"
    ),
    n_samples: int = 5000,
    n_euler_steps: int = 128,
    sample_chunk: int = 500,
    out_name: str = "zero_shot_transfer.json",
):
    """Local CLI entry: one spawned container per seed, so the three run
    concurrently rather than serialised behind one cold start.

    `checkpoint` defaults to final_ema.pt because the published 16x16 table
    cell is the EMA eval -- the three seeds read 0.805/0.838/0.836, mean 0.826,
    which is the printed number. final.pt instead shows a ~2.5-point deficit at
    the t*=1, c=0.5 anchor that is checkpoint choice, not failed transfer."""
    handles = []
    for seed in seeds.split(","):
        run_dir = "/results/" + run_template.format(seed=seed.strip())
        argv = (
            f"--run-dir {run_dir} --checkpoint {checkpoint} "
            f"--compositions {compositions} --stop-times {stop_times} "
            f"--n-samples {n_samples} --n-euler-steps {n_euler_steps} "
            f"--sample-chunk {sample_chunk} "
            f"--out {run_dir}/{out_name}"
        )
        handles.append(zero_shot_transfer_remote.spawn(argv=argv))
        print(f"[zero_shot_transfer] spawned seed {seed.strip()} -> {run_dir}")
    for handle in handles:
        handle.get()


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
        cfg_name=cfg_name,
        seed=seed,
        head_kind=resolved_head_kind,
        smoke=smoke,
        tag=time.strftime("%Y%m%d-%H%M%S"),
    )


@app.local_entrypoint()
def gate(
    seeds: str = "42,43,44",
    n_samples: int = 5000,
    skip_controls: bool = False,
    cells: str = "",
    out: str = "/results/gate_4x4",
):
    """Local CLI entry for the gate: blocking `.remote()` so the per-run
    progress prints stream back to the local terminal. `cells` = comma-
    separated CONFIGS names to gate instead of the dh ladder (pass `out` too
    so the ladder's verdict.json is not overwritten)."""
    gate_remote.remote(
        seeds=seeds,
        n_samples=n_samples,
        skip_controls=skip_controls,
        cells=cells,
        out=out,
    )


@app.function(gpu="A100-80GB", timeout=45 * 60)
def compile_gate_remote():
    """Run the CPU-passed compile gate on the training GPU stack: Inductor
    generates different kernels per backend. Raise on a failed gate."""
    from experiments.constrained_hard_03.probes.compile_gate import main as gate_main

    if gate_main() != 0:
        raise RuntimeError("compile gate FAILED on the GPU stack")


@app.local_entrypoint()
def compile_gate():
    """Blocking local CLI entry for the GPU-stack compile gate."""
    compile_gate_remote.remote()


@app.function(gpu="A100-80GB", timeout=60 * 60)
def compile_profile_remote(
    cfg_name: str = "",
    microbatch: int = 128,
    rollout_batch: int = 512,
    rollout_steps: int = 16,
):
    """Post-compile region profile of the production swap stack (method and
    region list in compile_profile.py). Prints tables; nothing on the
    volume."""
    from experiments.constrained_hard_03.probes.compile_profile import (
        THP2_CELL,
        run_profile,
    )

    run_profile(
        cfg_name=cfg_name or THP2_CELL,
        microbatch=microbatch,
        rollout_batch=rollout_batch,
        rollout_steps=rollout_steps,
    )


@app.local_entrypoint()
def compile_profile(
    cfg_name: str = "",
    microbatch: int = 128,
    rollout_batch: int = 512,
    rollout_steps: int = 16,
):
    """Blocking local CLI entry for the post-compile profile."""
    compile_profile_remote.remote(
        cfg_name=cfg_name,
        microbatch=microbatch,
        rollout_batch=rollout_batch,
        rollout_steps=rollout_steps,
    )


@app.local_entrypoint()
def demo(seeds: str = "42,43,44", n_samples: int = 5000, n_replicates: int = 5):
    """Blocking local CLI entry for the 4x4 demo GPU stage (per-cell progress
    prints stream back to the local terminal)."""
    demo_remote.remote(seeds=seeds, n_samples=n_samples, n_replicates=n_replicates)


@app.local_entrypoint()
def phihist(seeds: str = "42,43,44", n_samples: int = 5000):
    """Blocking local CLI entry for the phi-histogram stage."""
    phi_hist_remote.remote(seeds=seeds, n_samples=n_samples)


@app.local_entrypoint()
def evalonly(
    run_dirs: str,
    multi_event: int = -1,
    smc_tau: float = 0.0,
    n_euler_override: int = 0,
):
    """Spawn eval-only recovery over comma-separated run dir names on the
    volume (fire-and-forget: launch with --detach). `multi_event` is a trit
    (see _resolve_multi_event_trit): -1 uses each cell's own canonical
    trajectory step, 0 forces the one-event step, 1 the matching step.
    `smc_tau > 0` runs the SMC-resampled eval variant instead of the plain-IS
    one; `n_euler_override > 0` the grid-decoupling probe (eval_ne<k>/)."""
    _resolve_multi_event_trit(multi_event)  # fail fast locally on bad values
    names = [n.strip() for n in run_dirs.split(",") if n.strip()]
    for name in names:
        eval_remote.spawn(
            run_dir_name=name,
            multi_event=multi_event,
            smc_tau=smc_tau,
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


@app.local_entrypoint()
def gfn_batch_seeds(cfg_name: str, seeds: str = "42,43,44", tag: str = ""):
    """Spawn one GFN cell across seeds on the production card (the d256
    cells; `gfn_d16` keeps the L4 for the 4x4 cells). Pass the campaign
    tag so a preemption retry resumes into the same run dirs."""
    from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS

    if cfg_name not in GFN_CONFIGS:
        raise ValueError(f"unknown GFN cell {cfg_name!r}")
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    for seed in seed_list:
        train_gfn_remote_a100.spawn(cfg_name=cfg_name, seed=seed, tag=tag)
    print(f"spawned {len(seed_list)} jobs for {cfg_name}: seeds={seed_list} tag={tag}")


@app.local_entrypoint()
def gfn_d16(seeds: str = "42,43,44", tag: str = "", suffix: str = ""):
    """Spawn the 4x4 GFN comparator cells (tb/fldb x s010/s220) across the
    given seeds. Pass an explicit tag so the run dirs carry a known label.
    `suffix` restricts to cells whose name ends with it (e.g. `_par` = the
    parity cells only); empty spawns every registered cell."""
    from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS

    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    spawned = []
    for cfg_name in sorted(n for n in GFN_CONFIGS if n.endswith(suffix)):
        for seed in seed_list:
            handle = train_gfn_remote.spawn(cfg_name=cfg_name, seed=seed, tag=tag)
            spawned.append((cfg_name, seed, handle.object_id))
    for cfg_name, seed, object_id in spawned:
        print(f"spawned {cfg_name} seed={seed} -> {object_id}")
    print(f"tag={tag}; {len(spawned)} jobs total")
