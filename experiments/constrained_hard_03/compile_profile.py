"""Post-compile profile of the production swap stack.

Profile the compiled thp2 record cell at production shapes on one Modal
A100 container. The four regions below show which costs remain after
compilation (candidate follow-ups: einsum readout, level-loop collapse,
rank-one zeroed patch, scatter alternatives).

Regions (each warmed up past compilation before profiling):
  1. head forward, no-grad, B = inner microbatch — readout, level loop,
     patch construction.
  2. loss_swap forward + backward, B = inner microbatch — the training
     update unit the 2.21x compile speed-up was measured on.
  3. rollout slice, production batch and PRODUCTION dt (a leading slice of
     the 128-step grid, so thinning probabilities match production) —
     matching rounds, .any() syncs, apply_swaps.
  4. xi_t_swap_from_scores on pre-gathered scores — the ~12-kernel
     RNG-free chain (the cached gather is live here).

Output is printed tables (CUDA-time-sorted key_averages + peak memory per
region); nothing lands on the volume — copy the log by hand.

Run on Modal:
    pixi run -e dev modal run -m \
        experiments.constrained_hard_03.modal_app::compile_profile
"""
import time

import torch
from torch.profiler import ProfilerActivity, profile

from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.swap_ctmc import (
    sample_swap_ctmc,
    xi_t_swap_from_scores,
)
from discrete_flow_sampler.samplers.swap_kolmogorov import loss_swap

THP2_CELL = "H2_d256_c50_s223_letf_thp2_70k_curr_b512_ne128_cv2"
TABLE_ROWS = 48


def _profile_region(label: str, fn, warmups: int = 2, actives: int = 3):
    """Warm past compilation, then profile `actives` repeats of `fn`."""
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
    ) as prof:
        for _ in range(actives):
            fn()
        torch.cuda.synchronize()
    wall_s = (time.perf_counter() - started) / actives
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n===== region: {label} — {wall_s * 1e3:.1f} ms/iter, "
          f"peak {peak_gb:.2f} GB =====")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=TABLE_ROWS
    ))


def run_profile(cfg_name: str = THP2_CELL, microbatch: int = 128,
                rollout_batch: int = 512, rollout_steps: int = 16):
    from experiments.constrained_hard_03.configs import (
        CONFIGS,
        optimised_recipe,
    )
    from experiments.constrained_hard_03.run import build_target_and_head

    device = "cuda"
    print(f"[compile_profile] cfg={cfg_name} "
          f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    cfg = optimised_recipe(CONFIGS[cfg_name])  # compile_head=True
    target, head = build_target_and_head(cfg, device)
    head.eval()

    d = target.d
    pairs = upper_tri_pairs(d, device)
    torch.manual_seed(0)
    x_micro = target.sample_base(microbatch, device=device)
    t_micro = torch.rand(microbatch, device=device)
    dt_log_Zt = torch.zeros(microbatch, device=device)
    x_roll = target.sample_base(rollout_batch, device=device)
    t_roll = torch.rand(rollout_batch, device=device)
    # Leading slice of the production grid: production dt (1/(ne-1)),
    # so thinning probabilities — and thus matching-round counts — match
    # a real rollout rather than a coarsened one.
    production_dt = 1.0 / (cfg.ctmc.n_euler_steps - 1)
    ts_slice = torch.arange(
        rollout_steps + 1, device=device, dtype=torch.float32
    ) * production_dt

    with torch.no_grad():
        scores_roll = gather_pair_scores(head(x_roll, t_roll), pairs)

    @torch.no_grad()
    def region_head_forward():
        head(x_micro, t_micro)

    def region_loss_update():
        head.zero_grad(set_to_none=True)
        loss_swap(x_micro, t_micro, dt_log_Zt, head, target).backward()

    @torch.no_grad()
    def region_rollout_slice():
        sample_swap_ctmc(
            head, x_roll, ts_slice,
            multi_event=cfg.ctmc.use_matching_step,
        )

    @torch.no_grad()
    def region_xi_chain():
        xi_t_swap_from_scores(scores_roll, x_roll, t_roll, target, pairs)

    _profile_region(f"head forward B={microbatch}", region_head_forward)
    _profile_region(f"loss_swap fwd+bwd B={microbatch}", region_loss_update)
    _profile_region(
        f"rollout slice b={rollout_batch} x {rollout_steps} steps "
        f"(production dt)", region_rollout_slice,
    )
    _profile_region(f"xi chain B={rollout_batch}", region_xi_chain)


if __name__ == "__main__":
    run_profile()
