"""Decompose the sampler's bond-correlation transport into GROSS and NET.

WHAT THIS MEASURES, AND WHY THE NET NUMBER ALONE IS NOT ENOUGH
--------------------------------------------------------------
The fixed-composition sampler has to move the mean bond alignment

    S(x) = sum_{<ij>} x_i x_j = 0.5 * x^T A x

from its base value to the target's.  Per site, and using the code's own
nearest-neighbour correlation C(x) = x^T A x / A.sum() (numerator and
denominator both double-count each undirected bond, so C is the mean
correlation PER BOND and S = 2 d C on a D x D torus, where 2 = bonds per
site = z/2):

    required edge-units/site = 2 (C_target - C_base)

The eval already reports the NET achievement -- the endpoint gap actually
closed -- and the jump counters report how many state-changing swaps were
fired.  Dividing one by the other gives "edge-units achieved per swap", which
at d64 measures 3.907.  But that ratio cannot distinguish two completely
different samplers:

  * TARGETING problem -- every swap individually achieves little.  The head
    is putting its rate mass on low-|Delta S| pairs.  The fix lives in the
    architecture / rate parameterisation (can the head even represent the
    long-range pair structure it needs?).
  * CANCELLATION problem -- individual swaps are large but undo one another,
    so transport is diffusive rather than directed.  The fix lives in the
    reference-process weight construction.

They call for different work, and the net ratio is blind to the difference.
This script separates them by accumulating, along each trajectory,

    net   = S(x_N) - S(x_0)                     (what the eval already sees)
    gross = sum_k |S(x_{k+1}) - S(x_k)|         (the work actually done)
    forward / backward = sum_k (Delta S)^+ / (Delta S)^-

    cancellation fraction = 1 - |net| / gross

A cancellation fraction near 0 means ballistic transport: essentially every
swap contributes to the endpoint.  Near 1 means diffusive: the sampler is
doing a random walk in S with a small drift, and most of its per-swap
capacity is spent undoing itself.  With `gross / n_state_changing_swaps` you
also get the honest per-swap MAGNITUDE, which is the number the "targeting"
reading needs and which the net ratio understates by exactly the
cancellation factor.

DESIGN CHOICE: Delta S FROM CONSECUTIVE STATES, NOT A CLOSED FORM
-----------------------------------------------------------------
Delta S for a swap of sites i, j has an exact closed form,
(x_j - x_i)(h_i - h_j) - A_ij (x_j - x_i)^2 with h = A x, and using it would
be cheaper.  It is deliberately NOT used here.  Recomputing S from the stored
states makes the measurement independent of the step kind (one-event or
matching), of how many swaps a step fired, and -- the reason that matters
most -- of any assumption that the base density is constant.  A warm-base
run would silently break a closed-form shortcut in exactly the regime this
diagnostic is meant to interrogate.  The cost is one (B, d) @ (d, d) product
per Euler step, which is negligible beside the head call already happening
there.

REJECTED ALTERNATIVE: accumulating Delta S inside `_euler_step_swap` as a new
`matching_stats` counter.  That would touch a hot production path shared by
training and eval to serve a diagnostic, and it would need the adjacency
threaded into the step function.  `return_all_states=True` already exists and
gives a strictly more informative result (the full per-step distribution, not
just sums) for zero production risk.

FAILURE MODE GUARDED: `return_all_states=True` is mutually exclusive with
`return_log_weights=True` (the trajectory mode deliberately withholds
log-weights, which post-reset are a per-segment residue).  So this draw
produces NO importance weights and its samples must never be quoted as an
eval: they are a fresh draw for a transport diagnostic only.  The script
therefore writes to `transport_decomposition.json` and never touches `eval/`.

Run remotely:  modal run modal_app.py::transport_decomposition_remote \
                   --run-dir-name <dir>
Run locally:   python analysis_transport_decomposition.py <run_dir>
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiments.constrained_hard_03.configs import CONFIGS, HardStageCfg
from experiments.constrained_hard_03.run import (
    _backfill_missing_defaults,
    build_target_and_head,
)
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.seeding import seed_everything


def bond_alignment(states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """S(x) = 0.5 x^T A x for a (..., d) batch of states, shape (...,).

    Half the quadratic form because the symmetrised adjacency counts each
    undirected bond twice -- the same convention `nn_correlation` divides out.
    """
    x = states.float()
    return 0.5 * ((x @ adjacency) * x).sum(dim=-1)


def decompose_run(
    run_dir: Path, *, n_samples: int | None = None, device: str | None = None
) -> dict:
    """Draw trajectories off a run's final checkpoint and split gross vs net.

    Mirrors `run.eval_only`'s reconstruction (config.json -> CONFIGS entry ->
    drift guard -> `build_target_and_head` -> final.pt) so the model here can
    never drift from the one the frozen eval numbers were read off.
    """
    saved = json.loads((run_dir / "config.json").read_text())
    _backfill_missing_defaults(saved, HardStageCfg)
    cfg = CONFIGS[saved["name"]]
    cfg = replace(
        cfg,
        head_kind=saved["head_kind"],
        train=replace(cfg.train, seed=saved["train"]["seed"]),
    )
    if json.loads(json.dumps(asdict(cfg))) != saved:
        raise ValueError(
            f"config.json in {run_dir} does not match CONFIGS[{saved['name']!r}]"
        )

    seed_everything(cfg.train.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target, head = build_target_and_head(cfg, device)
    head.load_state_dict(
        torch.load(run_dir / "checkpoints" / "final.pt",
                   map_location=device, weights_only=True)
    )
    head.eval()

    total = n_samples or cfg.eval.n_eval_samples
    chunk = cfg.eval.eval_sample_chunk or total
    ts = torch.linspace(0.0, 1.0, cfg.ctmc.n_euler_steps + 1, device=device)
    adjacency = target.A.to(device).float()
    n_sites = target.d

    transport_stats: dict = {}
    net_all, gross_all, forward_all, backward_all = [], [], [], []
    largest_step = torch.tensor(0.0, device=device)
    remaining = total
    with torch.no_grad():
        while remaining > 0:
            batch = min(chunk, remaining)
            x_initial = target.sample_base(batch, device=device)
            trajectory = sample_swap_ctmc(
                head, x_initial, ts, return_all_states=True, target=target,
                multi_event=cfg.ctmc.use_matching_step,
                matching_stats=transport_stats,
            )                                        # (n_steps + 1, B, d)
            alignment = bond_alignment(trajectory, adjacency)   # (n_steps+1, B)
            steps = alignment[1:] - alignment[:-1]              # (n_steps, B)
            net_all.append(alignment[-1] - alignment[0])
            gross_all.append(steps.abs().sum(dim=0))
            forward_all.append(steps.clamp(min=0.0).sum(dim=0))
            backward_all.append((-steps.clamp(max=0.0)).sum(dim=0))
            largest_step = torch.maximum(largest_step, steps.abs().max())
            remaining -= batch

    net = torch.cat(net_all) / n_sites
    gross = torch.cat(gross_all) / n_sites
    forward = torch.cat(forward_all) / n_sites
    backward = torch.cat(backward_all) / n_sites

    # EXACTLY run.py's normalisation (its jumps_per_site_* block), so this
    # number is directly comparable with the eval metrics: state_steps
    # accumulates batch_size per Euler step, so it equals N * n_euler_steps
    # and the norm reduces to 1 / (N * d) -- swaps per site per trajectory.
    per_site_trajectory_norm = cfg.ctmc.n_euler_steps / (
        float(transport_stats["state_steps"]) * target.d
    )
    swaps_per_site = (
        float(transport_stats["accepted_state_changing"])
        * per_site_trajectory_norm
    )
    net_mean, gross_mean = float(net.mean()), float(gross.mean())
    return {
        "run_dir": run_dir.name,
        "cell": saved["name"],
        "n_samples": int(total),
        "n_euler_steps": int(cfg.ctmc.n_euler_steps),
        "multi_event": bool(cfg.ctmc.use_matching_step),
        "net_edge_units_per_site": net_mean,
        "gross_edge_units_per_site": gross_mean,
        "forward_edge_units_per_site": float(forward.mean()),
        "backward_edge_units_per_site": float(backward.mean()),
        "cancellation_fraction": 1.0 - abs(net_mean) / gross_mean,
        "swaps_per_site_state_changing": swaps_per_site,
        "net_per_swap": net_mean / swaps_per_site if swaps_per_site else None,
        "gross_per_swap": gross_mean / swaps_per_site if swaps_per_site else None,
        "largest_single_step_delta_s": float(largest_step),
        "net_std_over_draws": float(net.std()),
        "transport_stats": {
            key: float(value) for key, value in transport_stats.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n-samples", type=int, default=0)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    result = decompose_run(
        run_dir,
        n_samples=args.n_samples or None,
        device=args.device or None,
    )
    print(json.dumps(result, indent=2))
    (run_dir / "transport_decomposition.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
