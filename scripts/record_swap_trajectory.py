"""Record one raw Ising swap trajectory and anchor rates for the README animation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from experiments.constrained_hard_03.analysis.rate_field_strip import channel_for_anchor

from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.seeding import seed_everything

from .sample_checkpoint import load_bundle, positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--threads", type=positive_int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; choose a new archive path")
    if args.out.suffix != ".npz":
        parser.error("--out must have a .npz extension")
    torch.set_num_threads(args.threads)
    cfg, target, model, manifest = load_bundle(args.bundle, "cpu")
    if manifest["family"] != "hard" or cfg.target_kind != "ising":
        parser.error("requires a hard Ising checkpoint")
    seed_everything(args.seed)
    side = cfg.ising.D
    anchor = (side // 2) * side + side // 2
    grid = torch.linspace(0, 1, cfg.ctmc.n_euler_steps + 1)
    with torch.no_grad():
        trajectory = sample_swap_ctmc(
            model,
            target.sample_base(1, "cpu"),
            grid,
            return_all_states=True,
            target=target,
            multi_event=cfg.ctmc.use_matching_step,
        )[:, 0]
        rates, channels = [], []
        js = torch.arange(target.d)
        for state, time in zip(trajectory, grid, strict=True):
            field = model(state[None], time[None])[0]
            rate = torch.where(js > anchor, field[anchor, js], field[js, anchor]).relu()
            rate[anchor] = 0
            rates.append(rate.numpy())
            channels.append(
                channel_for_anchor(
                    state, anchor, target.A.float(), cfg.ising.sigma
                ).numpy()
            )
    metadata = {
        "side": side,
        "anchor": anchor,
        "sigma": cfg.ising.sigma,
        "seed": args.seed,
        "n_draws": 1,
        "displayed_draw": 0,
        "checkpoint": manifest["checkpoint"],
        "source_run": manifest["source_run"],
        "source_sha256": manifest["sha256"],
        "n_intervals": len(grid) - 1,
        "step_kind": "matching" if cfg.ctmc.use_matching_step else "one_event",
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
