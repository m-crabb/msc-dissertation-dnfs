"""Wall-clock + torch.profiler harness for the swap-CTMC pipeline.

Modes isolate the pipeline layers so the eval/train cost ranking (backbone
passes vs readout vs Euler sampling logic vs syncs vs Python dispatch) is
measured rather than guessed:

    head        one LeTFMaskOneSwapHead forward (no_grad): the (d*B)-row pass
    train_step  one inner gradient step: loss_swap forward + backward + AdamW
    gfn_update  GFlowNet loss forward + backward + AdamW on a prepared batch
    eval        one eval slice: sample_swap_ctmc(return_log_weights=True)
    components  per-piece timings inside one Euler step (head / gather /
                log-ratio / xi / one-event step / matching step / base draw)

Timing protocol: fixed seeds, `--warmup` calls (default one), then `--repeats`
timed calls with torch.cuda.synchronize() around each on CUDA. Peak counters
are reset after warmup, excluding compilation/setup peaks. Reports median/min
seconds and additional CUDA memory; BENCH_JSON also records every timing and
total allocated peak, including model, prepared batch and optimiser state.
`--profile` additionally wraps one call in torch.profiler
and prints the top ops by self time (CPU table locally, CUDA table on GPU).

Usage (local CPU):
    pixi run -e dev python -m experiments.constrained_hard_03.profile_swap \\
        --mode eval --d 64 --batch 32 --n-euler-steps 16 --repeats 3

On the Modal L4 (see modal_app.bench):
    pixi run -e dev modal run -m experiments.constrained_hard_03.modal_app::bench \\
        --argv "--mode eval --d 64 --batch 256 --n-euler-steps 128"
"""
import argparse
import json
import logging
import statistics
import time

import torch
import torch.nn.functional as F

from discrete_flow_sampler.constraints.factorised_swap_head import (
    FactorisedSwapHead,
)
from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import (
    DoublyHollowSwapHead,
    LeTFMaskOneSwapHead,
)
from discrete_flow_sampler.constraints.two_hole_patch_swap_head import (
    TwoHolePatchSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.swap_ctmc import (
    _euler_step_swap,
    _euler_step_swap_matching,
    compute_xi_t_swap,
    sample_swap_ctmc,
)
from discrete_flow_sampler.samplers.swap_kolmogorov import (
    loss_swap, loss_swap_backward_microbatched)
from discrete_flow_sampler.targets.ising import (
    SIGMA_C, FixedCompositionIsingTarget)


def build_head_and_target(
    d: int, device: torch.device, anchor_chunk: int | None, use_sdpa: bool = False,
    head_kind: str = "mask_one",
    site_orderings: tuple[str, ...] = ("row",),
    exterior_combiner: str = "mlp",
    interior_band: str | None = None,
    patch_radius: int = 1,
    rope_patch_size: int | None = None,
    gather_triu_pairs: bool = False,
    separable_band_scores: bool = False,
):
    """Production-shape head/target (hidden 32, 2 layers, 4 heads, sigma_c).

    head_kind defaults to mask_one so every recorded baseline stays
    comparable; "interval" benches the one-pass spike head (K3 A/B);
    "masked_attention" benches the reported exclusion-mask head; "stencil"
    benches the MA head with the 5-point lattice-stencil band family (the
    ladder's 0.8046/0.86034 cell); "factorised" benches the rank-8
    bilinear+global head at its fab8-arm defaults; "naive" benches the
    O(d^2) doubly-hollow oracle (mask BOTH sites of every ordered pair,
    sequential loop — bit-exact to mask_one at the d=16 gate, so it
    prices the naive rung of the forward-pass ladder rather than shipping
    as a sampler; measurable only at small d); "two_hole_patch" benches the
    ordering-free patch + pooled-levels head at radius `patch_radius`.
    Parity with the production cells is pinned by
    tests/test_profile_swap_heads.py."""
    side = int(round(d**0.5))
    if side * side != d:
        raise ValueError(f"--d must be a square lattice site count, got {d}")
    torch.manual_seed(42)
    target = FixedCompositionIsingTarget(
        D=side, sigma=SIGMA_C, target_composition=0.5, device=device
    )
    backbone = LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=32, n_layers=2, n_heads=4,
        use_sdpa_readout=use_sdpa,
    ).to(device)
    if head_kind == "interval":
        head = IntervalSwapHead(
            backbone, pair_offsets=(1, side), exterior_combiner=exterior_combiner,
            gather_triu_pairs=gather_triu_pairs,
            site_orderings=site_orderings, lattice_side=side,
        ).to(device)
    elif head_kind == "masked_attention":
        head = MaskedAttentionSwapHead(
            backbone, pair_offsets=(1, side), exterior_combiner=exterior_combiner,
            gather_triu_pairs=gather_triu_pairs,
            separable_band_scores=separable_band_scores,
            site_orderings=site_orderings, lattice_side=side,
        ).to(device)
    elif head_kind == "stencil":
        head = MaskedAttentionSwapHead(
            backbone, pair_offsets=(1, side), use_stencil=True, lattice_side=side,
            gather_triu_pairs=gather_triu_pairs,
            separable_band_scores=separable_band_scores,
        ).to(device)
    elif head_kind == "factorised":
        head = FactorisedSwapHead(
            backbone, site_orderings=site_orderings, lattice_side=side,
            interior_band=interior_band, gather_triu_pairs=gather_triu_pairs,
        ).to(device)
    elif head_kind == "naive":
        head = DoublyHollowSwapHead(backbone).to(device)
    elif head_kind == "two_hole_patch":
        head = TwoHolePatchSwapHead(
            backbone, lattice_side=side, patch_radius=patch_radius,
        ).to(device)
    else:
        head = LeTFMaskOneSwapHead(backbone, anchor_chunk_size=anchor_chunk)
    return head, target


def _timed(fn, repeats: int, device: torch.device, warmup: int = 1) -> list[float]:
    for _ in range(warmup):
        fn()  # compilation, allocator, autotune, lazy optimiser state
    if device.type == "cuda":
        torch.cuda.synchronize()
        # Report steady-state update memory, excluding compilation/setup peaks.
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def _report(name: str, times: list[float], device: torch.device,
            baseline_bytes: int = 0) -> None:
    """`baseline_bytes` is what was already resident when the peak counter was
    reset, and is subtracted so the column reports this configuration's own
    transient cost.

    `reset_peak_memory_stats()` resets the peak, not the allocator: a later
    `max_memory_allocated()` still counts every tensor alive at reset time.
    That is a real hazard whenever `main` is called more than once in a
    process, and the subtraction also removes this configuration's own
    parameters, which is the right call at ~100k of them (0.4 MB) -- what the
    column is for is the transient cost of a forward, not the checkpoint size.

    It is NOT, however, what inflated the multi-configuration bench: measured
    2026-08-29, subtracting the baseline moved the interval d=256 reading from
    2.97 GB to 2.96 GB against 1.76 GB for the same configuration benched
    first. Predecessors were being freed. The 1.2 GB was torch.compile giving
    up and falling back to eager -- see _CompileGaveUp.
    """
    peak_gb = (
        (torch.cuda.max_memory_allocated() - baseline_bytes) / 1e9
        if device.type == "cuda" else 0.0
    )
    print(
        f"{name:24s} median {statistics.median(times):9.4f}s  "
        f"min {min(times):9.4f}s  peak_mem {peak_gb:6.2f} GB"
    )
    print("BENCH_JSON " + json.dumps({
        "name": name, "seconds": times,
        "median_seconds": statistics.median(times),
        "peak_additional_gb": peak_gb,
        "peak_total_gb": (torch.cuda.max_memory_allocated() / 1e9
                          if device.type == "cuda" else None),
        "device": (torch.cuda.get_device_name(device)
                   if device.type == "cuda" else str(device)),
        "torch": torch.__version__,
    }))


def _profile_once(fn, device: torch.device) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))


def _runners(args, head, target, device: torch.device) -> dict:
    """Named zero-arg callables for the requested mode."""
    d, batch = target.d, args.batch
    torch.manual_seed(7)
    x = target.sample_base(batch, device=device)
    t = torch.full((batch,), 0.5, device=device)
    ts = torch.linspace(0.0, 1.0, args.n_euler_steps, device=device)
    step_dt = ts[1] - ts[0]
    pairs = upper_tri_pairs(d, device)

    if args.mode == "head":
        return {"head_forward": lambda: head(x, t)}

    if args.mode == "train_step":
        optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3)
        c_t = torch.zeros(batch, device=device)

        train_autocast_kwargs = dict(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=args.train_autocast_bf16,
        )

        def run_train_step():
            with torch.autocast(**train_autocast_kwargs):
                loss = loss_swap(x, t, c_t, head, target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        def run_train_step_microbatched():
            # The production path when loss_microbatch_size is set: the same
            # gradient, accumulated over row slices instead of materialising
            # one graph over all of them.
            optimiser.zero_grad()
            with torch.autocast(**train_autocast_kwargs):
                loss_swap_backward_microbatched(
                    x, t, c_t, head, target,
                    microbatch_size=args.loss_microbatch,
                )
            optimiser.step()

        if args.loss_microbatch is None:
            return {"train_step": run_train_step}
        return {f"train_step_mb{args.loss_microbatch}":
                run_train_step_microbatched}

    if args.mode == "eval":
        autocast_kwargs = dict(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=args.eval_autocast_bf16,
        )

        def run_eval_slice():
            x0 = target.sample_base(batch, device=device)
            with torch.autocast(**autocast_kwargs):
                sample_swap_ctmc(
                    head, x0, ts, return_log_weights=True, target=target,
                    multi_event=args.multi_event,
                )

        def eval_quality_diagnostics():
            """One seeded draw: ESS fraction + composition for the flag-on vs
            flag-off within-noise comparison (Tier-2 evidence)."""
            from discrete_flow_sampler.diagnostics.metrics import (
                ess_from_log_weights,
            )

            torch.manual_seed(123)
            x0 = target.sample_base(batch, device=device)
            with torch.autocast(**autocast_kwargs):
                x_final, log_w = sample_swap_ctmc(
                    head, x0, ts, return_log_weights=True, target=target,
                    multi_event=args.multi_event,
                )
            ess_frac = ess_from_log_weights(log_w).item() / batch
            composition = ((x_final > 0).float().mean(dim=1))
            print(
                f"eval_quality: ess_frac {ess_frac:.4f}  "
                f"composition mean {composition.mean():.4f} "
                f"(target {target.target_composition})  "
                f"log_w mean {log_w.mean():.4f} std {log_w.std():.4f}"
            )

        return {"eval_slice": run_eval_slice,
                "_quality": eval_quality_diagnostics}

    def run_matching_step():
        try:
            _euler_step_swap_matching(head, x, t, step_dt)
        except RuntimeError as error:  # pre-fix: CPU/CUDA device mismatch
            print(f"  [matching step failed: {error}]")

    return {
        "sample_base": lambda: target.sample_base(batch, device=device),
        "head_forward": lambda: head(x, t),
        "gather_relu": lambda: F.relu(gather_pair_scores(head(x, t), pairs)),
        "swap_log_ratio": lambda: target.swap_log_ratio(x, t, pairs),
        "xi_t_full": lambda: compute_xi_t_swap(x, t, head, target),
        "euler_one_event": lambda: _euler_step_swap(head, x, t, step_dt),
        "euler_matching": run_matching_step,
    }


class _CompileGaveUp(logging.Handler):
    """Tell whether torch.compile silently abandoned this configuration.

    `nn.Module.compile()` caches per FORWARD CODE OBJECT, never per module
    instance, and `MaskedAttentionSwapHead` subclasses `IntervalSwapHead`
    without overriding `forward` -- so interval, masked_attention and stencil
    share ONE dynamo budget of `torch._dynamo.config.recompile_limit` (8 by
    default). Each freshly built head spends TWO entries of it, because
    `LeTFRateMatrix._cached_causal_mask` sets `_causal_mask` lazily: the first
    trace guards the attribute ABSENT, the call after it appears fails that
    guard and retraces. Four configurations of that family therefore exhaust
    the budget, after which dynamo marks the code object skipped for the rest
    of the PROCESS and every later configuration runs eager -- while still
    printing `compile=True`, which is what makes the corruption invisible.

    Measured on an A100 2026-08-29, `--mode head --batch 32 --sdpa --d 256
    --head-kind interval`: 10.1 ms / 1.76 GB as the first configuration of a
    process, 26.4 ms / 2.96 GB as the sixth, with nothing but roster position
    changed. Raising the limit to 256 restored 10.1 ms / 1.74 GB in the same
    six-configuration process, which is the measurement that isolates it.
    The compiled figure is the true one: production cells set
    `compile_head=True` and build exactly one head per process.

    `torch._dynamo.reset()` before each `head.compile()` is the fix; this
    handler is the alarm that says the fix stopped working, because a wrong
    number here is otherwise perfectly plausible.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.gave_up = False

    def emit(self, record):
        if "recompile_limit" in record.getMessage():
            self.gave_up = True

    def arm(self) -> "_CompileGaveUp":
        """Watch the next configuration. Installed once and re-armed rather
        than re-added: `main` is called in a loop by `modal_app.bench_remote`,
        and a handler per call would pile up on the logger."""
        self.gave_up = False
        logger = logging.getLogger("torch._dynamo")
        if self not in logger.handlers:
            logger.addHandler(self)
        return self


_COMPILE_WATCH = _CompileGaveUp()


def _run_gfn_bench(args, device: torch.device) -> None:
    """GFN comparator rows for tab:head-cost-ladder (s100).

    The GFN has no per-Euler-step head forward: its sampler IS the
    KV-cached autoregressive rollout (d sequential one-token steps, EAGER
    BY CONSTRUCTION -- the per-step cache shapes are recompile territory),
    so the "forward" this mode prices is the WHOLE per-sample rollout at
    the bench batch, and the table's caption must say so. The policy is
    built from the registry's d64 `_par` recipe exactly as the comparator
    cells run it (hidden 64/2/4 = measured-param parity with the
    masked-attention head; a different --d re-realises the same policy at
    that lattice, the same convention the head rows use for sizes no cell
    was trained at). Train step = eager rollout + scoring loss
    forward/backward (compiled under --compile, the shipped
    compile_policy=True configuration) + AdamW with the arm's own split
    lr groups.

    gfn_update instead prepares and detaches one batch before timing, then
    reuses it for loss evaluation, backward and AdamW. This matches the
    swap train_step boundary: neither includes trajectory generation. It
    measures update cost, not total training cost or convergence speed.
    """
    from dataclasses import replace as dataclass_replace

    from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS
    from experiments.constrained_hard_03.run_gfn import (
        _loss_and_train_diagnostics,
        build_optimiser,
        build_target_and_policy,
    )

    side = int(round(args.d ** 0.5))
    if side * side != args.d:
        raise ValueError(f"--d must be a square lattice site count, got {args.d}")
    # Prefer a SIZE-NATIVE registered parity cell: parity is measured
    # params PER RUNG, so if a 16x16 GFN arm is ever registered at its own
    # capacity, this bench re-points to it automatically and the
    # cost-ladder row re-benches at the architecture that actually runs.
    # Until then, sizes without a cell price the d64 `_par` recipe
    # re-realised at that lattice — the same convention the head rows use
    # for sizes no cell was trained at — and say so in the output line.
    native = [
        name for name in GFN_CONFIGS
        if name.startswith(f"GFN_d{args.d}_c50_s220_{args.gfn_objective}_")
        and name.endswith("_par")
    ]
    if native:
        cfg = dataclass_replace(GFN_CONFIGS[native[0]], batch_size=args.batch)
    else:
        cfg = dataclass_replace(
            GFN_CONFIGS[f"GFN_d64_c50_s220_{args.gfn_objective}_50k_par"],
            D=side, batch_size=args.batch,
        )
        if args.d != 64:
            print(f"no registered GFN d{args.d} parity cell: pricing the "
                  f"d64 `_par` recipe re-realised at D={side} — re-bench if "
                  f"a size-native cell lands at different capacity")
    torch.manual_seed(42)
    target, policy = build_target_and_policy(cfg, str(device))

    if args.mode == "gfn_rollout":
        def runner():
            with torch.no_grad():
                policy.sample(args.batch)
    else:  # gfn_train_step or gfn_update
        if args.compile:
            policy.site_log_probs = torch.compile(policy.site_log_probs)
            if cfg.with_flow_head:
                policy.site_log_probs_and_flow_residuals = torch.compile(
                    policy.site_log_probs_and_flow_residuals
                )
        optimiser = build_optimiser(cfg, policy)
        prepared_spins = None
        if args.mode == "gfn_update":
            with torch.no_grad():
                prepared_spins, _ = policy.sample(cfg.batch_size, epsilon=cfg.epsilon)
            prepared_spins = prepared_spins.detach()

        def runner():
            if prepared_spins is None:
                spins, _ = policy.sample(cfg.batch_size, epsilon=cfg.epsilon)
            else:
                spins = prepared_spins
            loss, _ = _loss_and_train_diagnostics(cfg, policy, target, spins)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    baseline_bytes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.max_memory_allocated()
    print(f"mode={args.mode} d={args.d} batch={args.batch} "
          f"compile={args.compile} warmup={args.warmup} repeats={args.repeats} "
          f"gfn_objective={cfg.objective} hidden={cfg.hidden_dim}")
    times = _timed(runner, args.repeats, device, warmup=args.warmup)
    _report(
        f"{args.mode}_{args.gfn_objective}_d{args.d}_B{args.batch}",
        times, device, baseline_bytes,
    )
    if args.profile:
        _profile_once(runner, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=("head", "train_step", "eval", "components",
                 "gfn_rollout", "gfn_train_step", "gfn_update"),
    )
    parser.add_argument(
        "--gfn-objective", default="tb", choices=("tb", "fldb"),
        help="gfn_* modes only: which comparator arm's registry recipe to "
             "price (costs differ only by the fldb flow head's linear).",
    )
    parser.add_argument("--d", type=int, default=64, help="site count (D*D)")
    parser.add_argument(
        "--head-kind", default="mask_one",
        choices=("mask_one", "interval", "masked_attention", "stencil",
                 "factorised", "naive", "two_hole_patch"),
    )
    parser.add_argument(
        "--patch-radius", type=int, default=1,
        help="two_hole_patch only: hollow window radius R (2R+1 <= D).",
    )
    parser.add_argument(
        "--site-orderings", default="row",
        help="comma-separated causal stream orderings. Each extra ordering "
             "is a full extra backbone pass and adds no modules, so this is "
             "the sweep axis of the ladder: 'row' is the one-sweep arm and "
             "'row,col' the two-sweep one (`ivmo2` / `mamo2`). Honoured by "
             "the interval, masked_attention and factorised heads; ignored by "
             "mask_one, naive and two_hole_patch, which have no causal streams.",
    )
    parser.add_argument(
        "--exterior-combiner", default="mlp", choices=("mlp", "bilinear"),
        help="interval / masked_attention only: 'bilinear' is the literal "
             "mab / ivb cell (only [P,S] moved into a rank-8 product).",
    )
    parser.add_argument(
        "--interior-band", default=None, choices=(None, "prefix", "attention"),
        help="factorised only: the fib / fatt / fimo2 interior mechanism.",
    )
    parser.add_argument(
        "--gather-triu-pairs", action="store_true",
        help="interval / masked_attention / factorised only: run the per-pair "
             "nonlinear work on the d(d-1)/2 unordered pairs instead of the "
             "d^2 grid (the D=16 memory lever). No-op for the other heads.",
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--anchor-chunk", type=int, default=None)
    parser.add_argument("--n-euler-steps", type=int, default=128)
    parser.add_argument("--multi-event", action="store_true")
    parser.add_argument("--eval-autocast-bf16", action="store_true")
    parser.add_argument("--sdpa", action="store_true")
    parser.add_argument(
        "--rope-patch-size", type=int, default=None,
        help="swap the leTF backbone for the periodic-RoPE / patch-key one at "
             "this patch size (1 or 2 = the rope1/rope2 cells). None = leTF, "
             "every archived row. A drop-in LeTFRateMatrix subclass, so the "
             "position code is the only variable and any head composes.",
    )
    parser.add_argument(
        "--separable-band-scores", action="store_true",
        help=(
            "compute the masked-attention band without its (B, d^2, n) score "
            "tensor -- an EXACT rewrite of the same function, so this prices "
            "memory and time only; quality is equivalence-tested, not benched"
        ),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--train-autocast-bf16", action="store_true",
        help="train_step only: run the loss forward/backward under a "
             "bf16 autocast. Unlike --tf32 this halves the BYTES of the "
             "(B, d, d, f) pair slab, which is what a bandwidth-bound "
             "step is actually waiting on. Exploratory: the production "
             "trainer autocasts the EVAL only.",
    )
    parser.add_argument(
        "--loss-microbatch", type=int, default=None,
        help="train_step only: slice the backward over this many rows "
             "at a time (the production `loss_microbatch_size`). This is "
             "gradient accumulation and it is gradient-EXACT, so it "
             "trades wall clock for peak memory and moves no number. "
             "None = the single-shot backward.",
    )
    parser.add_argument(
        "--tf32", action="store_true",
        help="run fp32 matmuls in TF32 (10-bit mantissa inputs, fp32 "
             "accumulate). Reports the exactness of the TARGET's x @ A "
             "alongside, because that matmul carries the closed-form "
             "swap log-ratio and therefore the importance weights.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.repeats < 1:
        parser.error("--warmup and --repeats must be positive")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.mode.startswith("gfn_"):
        _run_gfn_bench(args, device)
        return
    head, target = build_head_and_target(
        args.d, device, args.anchor_chunk, use_sdpa=args.sdpa,
        head_kind=args.head_kind,
        site_orderings=tuple(args.site_orderings.split(",")),
        exterior_combiner=args.exterior_combiner,
        interior_band=args.interior_band,
        patch_radius=args.patch_radius,
        rope_patch_size=args.rope_patch_size,
        gather_triu_pairs=args.gather_triu_pairs,
        separable_band_scores=args.separable_band_scores,
    )
    if args.tf32:
        # THE CORRECTNESS GATE, reported rather than assumed. TF32 is a
        # GLOBAL matmul setting, so it reaches the target's `h = x @ A` --
        # the closed-form Kawasaki field sum behind the swap log-ratio, and
        # so behind every importance weight. The argument that this is safe
        # is that both operands are tiny exactly-representable integers (x
        # is +-1, A is the 0/1 torus adjacency counted twice per edge) and
        # A100 TF32 accumulates in fp32, so the rounding TF32 applies to its
        # inputs has nothing to round. That is a claim about one operator,
        # and a claim is worth what its check is worth, so print the
        # residual. A nonzero value here means TF32 moves the weights and
        # the flag is an ESTIMATOR change, not a speed lever.
        probe = target.sample_base(min(args.batch, 64), device=device).float()
        adjacency = target.A.float()
        torch.set_float32_matmul_precision("high")
        tf32_field = probe @ adjacency
        torch.set_float32_matmul_precision("highest")
        exact_field = probe @ adjacency
        residual = (tf32_field - exact_field).abs().max().item()
        verdict = "EXACT" if residual == 0.0 else "INEXACT -- weights move"
        print(f"tf32 target x@A max |residual| vs fp32: {residual:.3e} ({verdict})")
        # Leave TF32 on for the timing that follows.
        torch.set_float32_matmul_precision("high")

    compile_watch = None
    if args.compile:
        # Fresh dynamo state per configuration -- see _CompileGaveUp. The
        # caches are keyed by forward CODE OBJECT, so a roster benched in one
        # process spends one shared budget; this hands each row its own.
        torch._dynamo.reset()
        compile_watch = _COMPILE_WATCH.arm()
        head.compile()
    print(
        f"mode={args.mode} head_kind={args.head_kind} "
        f"exterior_combiner={args.exterior_combiner} interior_band={args.interior_band} "
        f"site_orderings={args.site_orderings} patch_radius={args.patch_radius} "
        f"d={args.d} batch={args.batch} "
        f"anchor_chunk={args.anchor_chunk} n_euler_steps={args.n_euler_steps} "
        f"gather_triu_pairs={args.gather_triu_pairs} "
        f"separable_band_scores={args.separable_band_scores} "
        f"multi_event={args.multi_event} "
        f"eval_autocast_bf16={args.eval_autocast_bf16} sdpa={args.sdpa} "
        f"compile={args.compile} tf32={args.tf32} "
        f"rope_patch_size={args.rope_patch_size} "
        f"loss_microbatch={args.loss_microbatch} "
        f"train_bf16={args.train_autocast_bf16} "
        f"device={device} torch={torch.__version__}"
    )

    grad_free = args.mode != "train_step"
    with torch.no_grad() if grad_free else torch.enable_grad():
        runners = _runners(args, head, target, device)
        quality_fn = runners.pop("_quality", None)
        for name, fn in runners.items():
            baseline = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
            _report(name, _timed(fn, args.repeats, device, warmup=args.warmup),
                    device, baseline)
        if compile_watch is not None and compile_watch.gave_up:
            print(
                "  *** COMPILE ABANDONED: dynamo hit its recompile limit, so "
                "the numbers above are EAGER, not compiled. Bench this "
                "configuration in a fresh process. ***"
            )
        if quality_fn is not None:
            quality_fn()  # once, seeded -- not a timing target
        if args.profile:
            first_name, first_fn = next(iter(runners.items()))
            print(f"\ntorch.profiler: {first_name}")
            _profile_once(first_fn, device)


if __name__ == "__main__":
    main()
