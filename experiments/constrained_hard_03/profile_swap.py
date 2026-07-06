"""Wall-clock + torch.profiler harness for the swap-CTMC pipeline.

Modes isolate the pipeline layers so the eval/train cost ranking (backbone
passes vs readout vs Euler sampling logic vs syncs vs Python dispatch) is
measured rather than guessed:

    head        one LeTFMaskOneSwapHead forward (no_grad): the (d*B)-row pass
    train_step  one inner gradient step: loss_swap forward + backward + AdamW
    eval        one eval slice: sample_swap_ctmc(return_log_weights=True)
    components  per-piece timings inside one Euler step (head / gather /
                log-ratio / xi / one-event step / matching step / base draw)

Timing protocol: fixed seeds, one warmup call, `--repeats` timed calls with
torch.cuda.synchronize() around each on CUDA; reports median/min seconds and
CUDA peak memory. `--profile` additionally wraps one call in torch.profiler
and prints the top ops by self time (CPU table locally, CUDA table on GPU).

Usage (local CPU):
    pixi run -e dev python -m experiments.constrained_hard_03.profile_swap \\
        --mode eval --d 64 --batch 32 --n-euler-steps 16 --repeats 3

On the Modal L4 (see modal_app.bench):
    pixi run -e dev modal run -m experiments.constrained_hard_03.modal_app::bench \\
        --argv "--mode eval --d 64 --batch 256 --n-euler-steps 128"
"""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from discrete_flow_sampler.constraints.swap_readout import LeTFMaskOneSwapHead
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
from discrete_flow_sampler.samplers.swap_kolmogorov import loss_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def build_head_and_target(
    d: int, device: torch.device, anchor_chunk: int | None, use_sdpa: bool = False
):
    """Production-shape head/target (hidden 32, 2 layers, 4 heads, sigma_c)."""
    side = int(round(d**0.5))
    if side * side != d:
        raise ValueError(f"--d must be a square lattice site count, got {d}")
    torch.manual_seed(42)
    target = FixedCompositionIsingTarget(
        D=side, sigma=0.223, target_composition=0.5, device=device
    )
    backbone = LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=32, n_layers=2, n_heads=4,
        use_sdpa_readout=use_sdpa,
    ).to(device)
    head = LeTFMaskOneSwapHead(backbone, anchor_chunk_size=anchor_chunk)
    return head, target


def _timed(fn, repeats: int, device: torch.device) -> list[float]:
    fn()  # warmup (allocator, autotune, lazy init)
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


def _report(name: str, times: list[float], device: torch.device) -> None:
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    )
    print(
        f"{name:24s} median {statistics.median(times):9.4f}s  "
        f"min {min(times):9.4f}s  peak_mem {peak_gb:6.2f} GB"
    )


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

        def run_train_step():
            loss = loss_swap(x, t, c_t, head, target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return {"train_step": run_train_step}

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
            flag-off within-noise comparison (Tier-2 evidence, plan Task 7)."""
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=("head", "train_step", "eval", "components"),
    )
    parser.add_argument("--d", type=int, default=64, help="site count (D*D)")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--anchor-chunk", type=int, default=None)
    parser.add_argument("--n-euler-steps", type=int, default=128)
    parser.add_argument("--multi-event", action="store_true")
    parser.add_argument("--eval-autocast-bf16", action="store_true")
    parser.add_argument("--sdpa", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    head, target = build_head_and_target(
        args.d, device, args.anchor_chunk, use_sdpa=args.sdpa
    )
    print(
        f"mode={args.mode} d={args.d} batch={args.batch} "
        f"anchor_chunk={args.anchor_chunk} n_euler_steps={args.n_euler_steps} "
        f"multi_event={args.multi_event} "
        f"eval_autocast_bf16={args.eval_autocast_bf16} sdpa={args.sdpa} "
        f"device={device} torch={torch.__version__}"
    )

    grad_free = args.mode != "train_step"
    with torch.no_grad() if grad_free else torch.enable_grad():
        runners = _runners(args, head, target, device)
        quality_fn = runners.pop("_quality", None)
        for name, fn in runners.items():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _report(name, _timed(fn, args.repeats, device), device)
        if quality_fn is not None:
            quality_fn()  # once, seeded -- not a timing target
        if args.profile:
            first_name, first_fn = next(iter(runners.items()))
            print(f"\ntorch.profiler: {first_name}")
            _profile_once(first_fn, device)


if __name__ == "__main__":
    main()
