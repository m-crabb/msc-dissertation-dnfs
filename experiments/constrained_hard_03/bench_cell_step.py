"""Per-phase wall-clock decomposition of one REGISTERED hard-constraint cell.

WHY THIS EXISTS. The six 64-site Cu-Au cells run on 2026-09-03 (tag
20260903-cuau64-house, A100-80GB) cost 3.56 s per logged step, while the
`wall_clock_step_s` column -- which times only the inner loss update -- read
0.235 s. So 93% of a run's wall clock is spent outside the only column the
training log records, and an arithmetic estimate from the 8x8 Ising head
bench (~39 ms per mask-one forward at batch 128) predicts 0.6-0.8 s/step, a
factor of ~5 short. `profile_swap.py` cannot settle this: it builds its own
square-torus Ising target from `--d` and prices head / train_step / eval in
isolation, not the PHASES of `train_swap` on the target a CONFIGS cell
actually constructs. This script prices those phases on the real cell, via
`run.build_target_and_head`, so the missing time can be attributed rather
than guessed.

WHAT ONE LOGGED STEP COSTS. `train_swap` is an outer/inner loop: every
`inner_steps_per_outer` logged steps it rebuilds the replay buffer, which
costs one rollout plus one c_t grid, and every `eval.eval_every` steps it
additionally draws `n_eval_samples_training or n_eval_samples` samples in
`eval_sample_chunk` slices. So, with n = inner_steps_per_outer and
e = eval_every,

    t_step = t_inner + (t_rollout + t_c_t_grid + t_dt_traj) / n
             + (t_eval_slice * n_chunks + t_rate_diag + t_ckpt) / e     (1)

and the phases below are exactly the terms of (1). The one that the
arithmetic estimate omits entirely is `c_t_grid`: with
`estimator="control_variate"` and `c_t_grid_chunk_rows=None` it is a
SEQUENTIAL loop of `n_euler_steps` xi_t calls at `outer_batch` rows each --
a second full pass of head forwards per outer cycle, on top of the rollout's.

TIMING PROTOCOL. Fixed seed, one warmup call, `--repeats` timed calls with
`torch.cuda.synchronize()` around each; median reported. `--trainer-steps N`
additionally runs `train_swap` itself for N steps into a scratch dir and
reports the measured seconds per logged step, which is the end-to-end check
that (1) is complete -- if the phases do not sum to it, the residue is in
the bookkeeping (resume-state writes at `resume_every_outer`, checkpoint
saves, wandb) and not in the sampler.

Usage (local CPU smoke, tiny):
    pixi run -e dev python -m experiments.constrained_hard_03.bench_cell_step \\
        --cfg H2_cuau64_c25_T500_mask_one_50k_curr --repeats 1 \\
        --n-euler-steps 4 --batch 4 --eval-chunk 4 --device cpu

On the Modal A100-80GB (see modal_app.bench_cell):
    pixi run -e dev modal run -m \\
        experiments.constrained_hard_03.modal_app::bench_cell \\
        --argv "--cfg H2_cuau64_c25_T500_mask_one_50k_curr"
"""

import argparse
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch
from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.run import build_target_and_head

from discrete_flow_sampler.ema import ExponentialMovingAverage
from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs
from discrete_flow_sampler.samplers.swap_ctmc import (
    compute_c_t_grid_swap,
    sample_swap_ctmc,
)
from discrete_flow_sampler.samplers.swap_kolmogorov import (
    loss_swap_backward_microbatched,
)
from discrete_flow_sampler.samplers.swap_training import (
    _swap_rate_diagnostics,
    train_swap,
)


def _timed(fn, repeats: int, device: torch.device) -> float:
    """Median seconds over `repeats` synchronised calls, after one warmup."""
    fn()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _phase_runners(cfg, target, head, device):
    """Zero-arg callables for every term of (1), plus reference probes.

    Each closure is written to match what `train_swap` does at that point:
    same batch, same grid, same flags. Where the trainer holds a tensor
    across phases (the rollout trajectory feeding the c_t grid) the closure
    holds it too, so the c_t timing is not measuring a fresh rollout.
    """
    n_grid = cfg.ctmc.n_euler_steps
    batch = cfg.train.batch_size
    eval_chunk = cfg.eval.eval_sample_chunk or cfg.eval.n_eval_samples
    t_grid = torch.linspace(0.0, 1.0, n_grid, device=device)
    multi_event = cfg.ctmc.use_matching_step

    torch.manual_seed(42)
    x_initial = target.sample_base(batch, device=device)
    with torch.no_grad():
        x_traj = sample_swap_ctmc(
            head,
            x_initial,
            t_grid,
            return_all_states=True,
            multi_event=multi_event,
            target=target,
        )
    x_traj_flat = x_traj.reshape(n_grid * batch, target.d)
    t_flat = t_grid.repeat_interleave(batch)

    # Inner-step state: one buffer draw, held fixed, so the timing is the
    # loss/backward/step and not the buffer indexing.
    optimiser = torch.optim.AdamW(head.parameters(), lr=cfg.train.lr)
    ema = ExponentialMovingAverage(head.parameters(), decay=cfg.ema_decay)
    x_sample = x_traj[0]
    t_sample = t_grid[:1].expand(batch)
    c_t_sample = torch.zeros(batch, device=device)

    def rollout():
        with torch.no_grad():
            sample_swap_ctmc(
                head,
                target.sample_base(batch, device=device),
                t_grid,
                return_all_states=True,
                multi_event=multi_event,
                target=target,
            )

    def c_t_grid():
        compute_c_t_grid_swap(
            t_grid,
            x_traj,
            target,
            head,
            mode=cfg.estimator,
            chunk_rows=cfg.train.c_t_grid_chunk_rows,
        )

    def dt_traj():
        with torch.no_grad():
            target.dt_log_p_tilde_t(x_traj_flat, t_flat)

    def inner_step():
        optimiser.zero_grad()
        loss_swap_backward_microbatched(
            x_sample,
            t_sample,
            c_t_sample,
            head,
            target,
            microbatch_size=cfg.train.loss_microbatch_size,
        )
        torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.train.grad_clip_max_norm)
        optimiser.step()
        ema.update()

    eval_autocast = dict(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=cfg.eval.eval_autocast_bf16,
    )

    def eval_slice():
        with torch.no_grad(), torch.autocast(**eval_autocast):
            sample_swap_ctmc(
                head,
                target.sample_base(eval_chunk, device=device),
                t_grid,
                return_log_weights=True,
                target=target,
                multi_event=multi_event,
            )

    def rate_diag():
        with torch.no_grad():
            _swap_rate_diagnostics(
                head,
                x_sample,
                t_sample,
                step_dt=1.0 / max(n_grid - 1, 1),
                target=target,
            )

    x_probe = target.sample_base(eval_chunk, device=device)
    t_probe = torch.full((eval_chunk,), 0.5, device=device)
    pairs = upper_tri_pairs(target.d, device)

    def swap_log_ratio():
        with torch.no_grad():
            target.swap_log_ratio(x_probe, t_probe, pairs)

    def head_forward():
        with torch.no_grad():
            head(x_probe, t_probe)

    runners = {
        f"rollout (B={batch}, {n_grid} euler)": rollout,
        f"c_t_grid ({cfg.estimator}, {n_grid}x{batch})": c_t_grid,
        f"dt_log_p_tilde traj ({n_grid * batch} rows)": dt_traj,
        f"inner_step (B={batch})": inner_step,
        f"eval_slice (B={eval_chunk}, {n_grid} euler)": eval_slice,
        f"rate_diag (B={batch})": rate_diag,
        f"swap_log_ratio (B={eval_chunk})": swap_log_ratio,
        f"head_forward (B={eval_chunk})": head_forward,
    }
    if hasattr(target, "spec"):

        def swap_energy_change():
            with torch.no_grad():
                target.spec.swap_energy_change(x_probe)

        runners[f"swap_energy_change (B={eval_chunk})"] = swap_energy_change
    return runners


def _project(cfg, phase_s: dict, args) -> None:
    """Print the per-1000-step decomposition of (1) and the cost to finish.

    Scenarios: (a) the config as registered, (b) n_eval_samples_training cut
    to one chunk, (c) in-training eval off, (d) whatever levers this
    invocation was given (they are already inside the measured phases).
    """
    inner_per_outer = cfg.train.inner_steps_per_outer
    eval_every = cfg.eval.eval_every
    n_train_eval = cfg.eval.n_eval_samples_training or cfg.eval.n_eval_samples
    eval_chunk = cfg.eval.eval_sample_chunk or n_train_eval

    def named(prefix):
        return next(v for k, v in phase_s.items() if k.startswith(prefix))

    cycle_s = named("rollout") + named("c_t_grid") + named("dt_log_p_tilde")
    inner_s = named("inner_step")
    eval_fixed_s = named("rate_diag")
    slice_s = named("eval_slice")

    print("\nper-1000-step decomposition (s):")
    per_1000 = {}
    for label, n_eval in (
        ("(a) as registered", n_train_eval),
        (f"(b) n_eval_samples_training={eval_chunk}", eval_chunk),
        ("(c) in-training eval off", 0),
    ):
        n_chunks = -(-n_eval // eval_chunk) if n_eval else 0
        cycles = 1000 / inner_per_outer
        evals = 1000 / eval_every if n_eval else 0
        # The eval BLOCK still fires when n_eval is 0 only if eval_every
        # divides the step; scenario (c) means the block is skipped whole,
        # so its rate diagnostics and checkpoint write go with it.
        eval_s = evals * (n_chunks * slice_s + eval_fixed_s) if n_eval else 0.0
        total = 1000 * inner_s + cycles * cycle_s + eval_s
        per_1000[label] = total
        print(
            f"  {label:38s} inner {1000 * inner_s:8.1f}  "
            f"buffer {cycles * cycle_s:8.1f}  eval {eval_s:8.1f}  "
            f"TOTAL {total:8.1f}  ({total / 1000:5.3f} s/step)"
        )

    print(f"\nremaining {args.remaining_steps} steps of one cell:")
    for label, total in per_1000.items():
        hours = total * args.remaining_steps / 1000 / 3600
        print(f"  {label:38s} {hours:6.2f} h   ${hours * args.gpu_cost_per_hour:7.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="CONFIGS cell name")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default=None)
    # Overrides: the CPU smoke shrinks the cell; the lever rows change one
    # knob of the registered cell and re-time every phase under it.
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--eval-chunk", type=int, default=None)
    parser.add_argument("--n-euler-steps", type=int, default=None)
    parser.add_argument("--compile-head", action="store_true")
    parser.add_argument("--eval-autocast-bf16", action="store_true")
    parser.add_argument("--anchor-chunk", type=int, default=None)
    parser.add_argument(
        "--c-t-grid-chunk-rows",
        type=int,
        default=None,
        help="flatten the c_t grid into row-chunks of this size instead of "
        "the per-slot sequential loop (gradient-free, parity-tested).",
    )
    parser.add_argument("--remaining-steps", type=int, default=41_000)
    parser.add_argument("--gpu-cost-per-hour", type=float, default=2.50)
    parser.add_argument(
        "--trainer-steps",
        type=int,
        default=0,
        help="also run train_swap itself for this many logged steps and "
        "report measured s/step (the end-to-end check on (1)).",
    )
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = CONFIGS[args.cfg]
    if args.anchor_chunk is not None:
        cfg = replace(cfg, anchor_chunk_size=args.anchor_chunk)
    if args.eval_autocast_bf16:
        cfg = replace(cfg, eval=replace(cfg.eval, eval_autocast_bf16=True))
    if args.c_t_grid_chunk_rows is not None:
        cfg = replace(
            cfg,
            train=replace(cfg.train, c_t_grid_chunk_rows=args.c_t_grid_chunk_rows),
        )
    if args.n_euler_steps is not None:
        cfg = replace(cfg, ctmc=replace(cfg.ctmc, n_euler_steps=args.n_euler_steps))
    if args.batch is not None:
        cfg = replace(cfg, train=replace(cfg.train, batch_size=args.batch))
    if args.eval_chunk is not None:
        cfg = replace(cfg, eval=replace(cfg.eval, eval_sample_chunk=args.eval_chunk))

    torch.manual_seed(cfg.train.seed)
    target, head = build_target_and_head(cfg, str(device))
    if args.compile_head:
        torch._dynamo.reset()
        head.compile()
    print(
        f"cfg={args.cfg} d={target.d} target={type(target).__name__} "
        f"head={cfg.head_kind} hidden={cfg.model.hidden_dim}x{cfg.model.n_layers} "
        f"batch={cfg.train.batch_size} n_euler={cfg.ctmc.n_euler_steps} "
        f"eval_chunk={cfg.eval.eval_sample_chunk} "
        f"inner_per_outer={cfg.train.inner_steps_per_outer} "
        f"estimator={cfg.estimator} c_t_chunk={cfg.train.c_t_grid_chunk_rows} "
        f"compile={args.compile_head} bf16_eval={cfg.eval.eval_autocast_bf16} "
        f"anchor_chunk={cfg.anchor_chunk_size} device={device} "
        f"torch={torch.__version__}"
    )

    phase_s = {}
    for name, fn in _phase_runners(cfg, target, head, device).items():
        phase_s[name] = _timed(fn, args.repeats, device)
        print(f"  {name:44s} {phase_s[name] * 1e3:10.1f} ms")
    _project(cfg, phase_s, args)

    if args.trainer_steps:
        with tempfile.TemporaryDirectory() as scratch:
            short = replace(cfg, train=replace(cfg.train, n_steps=args.trainer_steps))
            torch.manual_seed(short.train.seed)
            target, head = build_target_and_head(short, str(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            train_swap(
                head,
                target,
                short.train,
                short.ctmc,
                short.eval,
                Path(scratch),
                use_wandb=False,
                estimator_mode=short.estimator,
                sigma_curriculum=(
                    short.curriculum.stages if short.curriculum else None
                ),
                ema_decay=short.ema_decay,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        print(
            f"\ntrain_swap end-to-end: {args.trainer_steps} steps in "
            f"{elapsed:.1f} s = {elapsed / args.trainer_steps:.3f} s/step "
            f"(includes {1 + args.trainer_steps // cfg.eval.eval_every} evals "
            f"and {-(-args.trainer_steps // cfg.train.inner_steps_per_outer)} "
            f"outer cycles)"
        )


if __name__ == "__main__":
    main()
