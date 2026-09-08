"""Record one raw flip trajectory with the learned per-site rates and the
closed-form flip response, for the flip-family (baseline or soft) rate strip.

The flip analogue of record_swap_trajectory: rates are one number per site,
so no anchor is marked, and the channel row is the closed-form flip response
Delta_i(x) = log pi(x^(i)) - log pi(x) (eq:exact-field), the Ising term plus the
penalty term where a penalty is present.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.seeding import seed_everything

from .sample_checkpoint import load_bundle, positive_int


def flip_rate(model, state, time):
    """R_t(x, x^(i)) for every site of one state, shape (d,).

    Locally equivariant models return pre-relu scores G(tau, i | x) over the
    spin values tau; the flip rate is the positive part of the score for the
    other spin value (ctmc._euler_step_lenet). Other models return rates.
    """
    scores = model(state[None], time[None])[0]
    if getattr(model, "is_locally_equivariant", False):
        flipped_index = ((1.0 - state) / 2).long()  # index 0 is spin -1, index 1 is +1
        scores = scores.gather(-1, flipped_index[:, None])[:, 0].relu()
    return scores


def flip_response(model, target, state):
    """Delta_i(x) for every site of one state, shape (d,)."""
    exact_field = getattr(model, "exact_field", None)
    if exact_field is not None:
        return exact_field(state[None])[0]
    return target.base_flip_log_ratio(state[None])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--threads", type=positive_int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; choose a new archive path")
    if args.out.suffix != ".npz":
        parser.error("--out must have a .npz extension")
    torch.set_num_threads(args.threads)
    cfg, target, model, manifest = load_bundle(args.bundle, "cpu")
    if manifest["family"] not in {"baseline", "soft"}:
        parser.error("requires a flip-family Ising checkpoint")
    seed_everything(args.seed)
    # The flip trainers count grid points, the swap trainer intervals.
    grid = torch.linspace(0, 1, cfg.ctmc.n_euler_steps)
    with torch.no_grad():
        trajectory = sample_ctmc(
            model,
            target.sample_base(1, "cpu"),
            grid,
            return_all_states=True,
            target=target,
        )[:, 0]
        rates, channels = [], []
        for state, time in zip(trajectory, grid, strict=True):
            rates.append(flip_rate(model, state, time).numpy())
            channels.append(flip_response(model, target, state).numpy())
    metadata = {
        "family": manifest["family"],
        "side": cfg.ising.D,
        "sigma": cfg.ising.sigma,
        "penalty_strength": cfg.ising.composition_penalty_strength,
        "target_composition": cfg.ising.target_composition,
        "base_composition": cfg.ising.base_composition,
        "seed": args.seed,
        "n_draws": 1,
        "displayed_draw": 0,
        "checkpoint": manifest["checkpoint"],
        "source_run": manifest["source_run"],
        "source_sha256": manifest["sha256"],
        "n_intervals": len(grid) - 1,
        "step_kind": "flip",
        "device": "cpu",
        "dtype": "float32",
        "threads": args.threads,
        "torch_version": torch.__version__,
        "draw_kind": "one raw proposal trajectory; no selection or resampling",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        states=trajectory.numpy(),
        rates=np.asarray(rates),
        channels=np.asarray(channels),
        times=grid.numpy(),
        metadata=json.dumps(metadata),
    )
    print(f"Saved {len(grid)} grid states to {args.out}")


if __name__ == "__main__":
    main()
