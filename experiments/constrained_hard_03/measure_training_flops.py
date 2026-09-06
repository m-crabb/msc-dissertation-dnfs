"""Measure a cell's training FLOPs at short horizons, extrapolate, and check
the extrapolation against the count the loop structure derives.

Why this exists. The house table's FLOP/es column prices SAMPLING only, by
design -- "training cost is amortised and lives in the appendix recipe table".
But the amortisation defence of that column is the claim that one training run
serves many targets, and that claim is unpriceable without a training number.
Nothing in `run.py` records one.

Method, and why it is a fit rather than a single measurement:

  * The loop is exactly periodic -- `n_outer` cycles of (one `n_euler_steps`
    rollout at `outer_batch`) + (`inner_steps_per_outer` updates at
    `batch_size`) -- so total FLOPs should be `fixed + per_cycle * n_outer`.
  * Running at three horizons and fitting recovers `per_cycle` while CANCELLING
    the fixed prefix. That prefix is not negligible: `replay_buffer_cycles` is 8
    on the production cells, so the first eight cycles run with a partly-filled
    buffer and are not the steady state being extrapolated. Measuring once and
    dividing folds that distortion straight into the per-cycle rate.
  * The fit's residual is the extrapolation's licence. If per-cycle cost drifts,
    the residual says so instead of the slope quietly absorbing it.

Horizons come from `valid_measurement_horizons`, i.e. multiples of
lcm(inner_steps_per_outer, eval_every). Total FLOPs are a STEP function of
n_steps -- they jump when a rollout lands and again when the periodic
in-training eval fires -- so a horizon that cuts a cycle or straddles an eval
sits off the line for reasons unrelated to per-cycle cost.

The cross-check, which is the actual output. `training_run_flops` derives the
same total from `measured_forward_flops` at the two batch shapes times the
forward count the recipe implies. It has to assume backward = 2x forward, and
that nothing is doing forwards the recipe does not mention. The measured leg
has to assume linearity. Comparing them tests both assumptions at once, and a
gap is a FINDING about the loop -- an unaccounted forward, a backward that is
not 2x, a c_t grid pass that is not actually being skipped -- rather than two
numbers to average.

One assumption the short horizons cannot test, stated rather than hidden: the
sigma curriculum's first boundary is at step 5,000, well past any measurement
horizon, so the measured slope is the FIRST stage's per-cycle rate. It is taken
as the whole run's rate because sigma is a scalar multiplying the target and
changes no tensor shape -- later stages do the identical arithmetic. What a
stage boundary does change is the replay-buffer flush, which alters what the
buffer holds but not how many forwards happen.

Nothing here modifies the trainer. FlopCounterMode is a dispatch mode, so it
wraps the existing `train_swap` call: the trainer is untouched and archived
runs stay byte-identical. The cost is that the counter's interception
slows the loop, which is exactly why this runs at short horizons.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.run import build_target_and_head
from torch.utils.flop_counter import FlopCounterMode

from discrete_flow_sampler.diagnostics.flops import (
    diagnostic_eval_flops,
    fit_flop_scaling,
    measured_forward_flops,
    training_forward_counts,
    training_run_flops,
    valid_measurement_horizons,
)
from discrete_flow_sampler.samplers.swap_training import train_swap


def curriculum_within(curriculum, horizon: int):
    """Stages that start inside the horizon — the smoke-cell pattern
    (`_SMOKE12K_SIGMA_LADDER`): the trainer's validator correctly rejects
    stages starting at or beyond n_steps, and passing the full production
    ladder to a 500-step measurement trips it. Truncation changes nothing
    about what is measured: every valid horizon sits inside the first
    stage (first boundary 5,000 vs horizons ≤ 1,500), and the module
    docstring already scopes the measured slope to the first stage's
    rate."""
    if curriculum is None:
        return None
    return tuple(s for s in curriculum.stages if s.start_step < horizon)


def measure_horizon(cfg, horizon: int, output_dir: Path, device: str) -> int:
    """Total FLOPs for a fresh training run of `horizon` inner steps.

    Fresh head and optimiser every horizon, and a scratch output dir wiped
    first: `train_swap` resumes from `checkpoints/resume.pt` if it finds one, so
    a reused directory would silently continue the previous horizon and the
    "measurement" would be of a few steps, not of the horizon.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    torch.manual_seed(cfg.train.seed)
    target, head = build_target_and_head(cfg, device)
    counter = FlopCounterMode(display=False)
    with counter:
        train_swap(
            head,
            target,
            replace(cfg.train, n_steps=horizon),
            cfg.ctmc,
            cfg.eval,
            output_dir,
            use_wandb=False,
            estimator_mode=cfg.estimator,
            sigma_curriculum=curriculum_within(cfg.curriculum, horizon),
        )
    return counter.get_total_flops()


def derived_flops(cfg, head, target, device: str, n_steps: int) -> dict:
    """The loop-structure count, priced by two measured per-forward readings.

    Two readings because the rollout runs at `outer_batch` and the update at
    `batch_size`, and the counter's number is not linear in batch -- per-call
    fixed work does not scale with rows.
    """
    outer_batch = cfg.train.outer_batch_size or cfg.train.batch_size

    def forward_flops(batch_size: int) -> int:
        x = target.sample_base(batch_size, device=device)
        t = torch.rand(batch_size, device=device)
        return measured_forward_flops(head, (x, t))

    rollout_flops = forward_flops(outer_batch)
    update_flops = forward_flops(cfg.train.batch_size)
    counts = training_forward_counts(
        n_steps,
        cfg.train.inner_steps_per_outer,
        cfg.ctmc.n_euler_steps,
        c_t_from_rollout=cfg.train.c_t_from_rollout,
    )
    return {
        "rollout_batch": outer_batch,
        "sampling_flops_per_eval_draw_set": (
            cfg.ctmc.n_euler_steps
            * update_flops
            * ((cfg.eval.n_eval_samples or 5000) / cfg.train.batch_size)
        ),
        "update_batch": cfg.train.batch_size,
        "rollout_forward_flops": rollout_flops,
        "update_forward_flops": update_flops,
        "forward_counts": counts,
        "total_flops": training_run_flops(
            rollout_flops,
            update_flops,
            n_steps=n_steps,
            inner_steps_per_outer=cfg.train.inner_steps_per_outer,
            n_euler_steps=cfg.ctmc.n_euler_steps,
            c_t_from_rollout=cfg.train.c_t_from_rollout,
        ),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cfg", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--n-horizons", type=int, default=3)
    parser.add_argument(
        "--horizon-scale",
        type=int,
        default=1,
        help="multiply every horizon; raise if the fixed prefix dominates",
    )
    parser.add_argument("--scratch", type=Path, default=Path("results/flop_probe"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = CONFIGS[args.cfg]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = [
        h * args.horizon_scale
        for h in valid_measurement_horizons(
            cfg.train.inner_steps_per_outer,
            getattr(cfg.eval, "eval_every", None),
            args.n_horizons,
        )
    ]
    print(f"[flops] {args.cfg}: horizons {horizons} (full run = {cfg.train.n_steps})")

    totals = []
    for horizon in horizons:
        total = measure_horizon(cfg, horizon, args.scratch / f"h{horizon}", device)
        totals.append(total)
        print(f"[flops]   {horizon:>7d} steps -> {total:.4e} FLOPs")

    cycles = [h // cfg.train.inner_steps_per_outer for h in horizons]
    fit = fit_flop_scaling(cycles, totals)
    full_cycles = cfg.train.n_steps // cfg.train.inner_steps_per_outer
    extrapolated = fit["extrapolate"](full_cycles)

    torch.manual_seed(cfg.train.seed)
    target, head = build_target_and_head(cfg, device)
    with torch.no_grad():
        derived = derived_flops(cfg, head, target, device, cfg.train.n_steps)

    # Price training in frozen-eval draw sets: each additional target served
    # by one checkpoint amortises this cost. Use the same measured forward
    # at the same batch as the training leg; per-call fixed work means the
    # counter is not linear in batch, so a per-sample price would misstate it.
    one_eval_draw_set = derived["sampling_flops_per_eval_draw_set"]

    # Three-way split (decided 2026-08-31, after the d64 certification
    # reconciled the measured-vs-derived gap to the instrument within
    # 0.8%): training-proper is the ALGORITHM's bill and the printed
    # appendix number; the periodic in-training eval is severable
    # instrumentation priced beside it, never folded in; and the
    # certification ratio compares the measurement against the SUM, since
    # the counter necessarily measured both.
    training_proper = derived["total_flops"]
    diagnostic = diagnostic_eval_flops(
        derived["update_forward_flops"],
        update_batch_size=cfg.train.batch_size,
        n_euler_steps=cfg.ctmc.n_euler_steps,
        n_steps=cfg.train.n_steps,
        eval_every=getattr(cfg.eval, "eval_every", None),
        n_eval_draws=(
            getattr(cfg.eval, "n_eval_samples_training", None)
            or cfg.eval.n_eval_samples
        ),
    )
    payload = {
        "cfg": args.cfg,
        "horizons": horizons,
        "measured_totals": totals,
        "fixed_flops": fit["fixed_flops"],
        "flops_per_outer_cycle": fit["flops_per_outer_cycle"],
        "max_relative_residual": fit["max_relative_residual"],
        "extrapolated_training_flops": extrapolated,
        "training_proper_flops": training_proper,
        "diagnostic_eval_flops": diagnostic,
        "as_instrumented_flops": training_proper + diagnostic,
        "measured_over_accounted": extrapolated / (training_proper + diagnostic),
        "derived_detail": derived,
        "one_eval_draw_set_flops": one_eval_draw_set,
        "training_proper_in_eval_draw_sets": training_proper / one_eval_draw_set,
    }
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k != "derived_detail"},
            indent=2,
            default=str,
        )
    )
    print(
        f"[flops] residual {fit['max_relative_residual']:.2%} "
        f"| measured/accounted {payload['measured_over_accounted']:.3f} "
        f"| training-proper = {payload['training_proper_in_eval_draw_sets']:.0f} "
        f"x ({cfg.eval.n_eval_samples}-draw eval) "
        f"| instrument +{diagnostic / training_proper:.0%}"
    )
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
