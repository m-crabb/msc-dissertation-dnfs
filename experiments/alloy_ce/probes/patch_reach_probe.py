"""Reach probe for the two-hole patch head on a cluster-expansion cell.

The head's window sees one neighbour shell (2.7 A on fcc), the Cu-Au
expansion carries pair terms out to 9.3 A, so unlike nearest-neighbour Ising
the exact swap log-ratio is not inside the one-shell function class: the
pooled levels must carry the far field. Before a GPU run, measure how much of
that log-ratio each window reach can represent.

Supervised fit, no sampler. The physical pair score G[min, max]
and the swap log-ratio Delta_ij = -beta DeltaE_swap share a symmetry class
(label-symmetric, odd under the state swap), so regress one on the other:
minimise the MSE of G over unlike pairs against Delta on uniform random
fixed-composition states, and report held-out R^2. A constant scale is
absorbed by the readout, so R^2, not the MSE, is the reach statistic.
Uniform slice states are the base distribution; ordered states at 500 K are
harder for a local head, so this is an optimistic bound on reach.

Usage (CPU, minutes):
    pixi run -e dev python -m experiments.alloy_ce.probes.patch_reach_probe \\
        --spec data/ce/cuau_fcc_4x4x4.json --composition 0.25 --shells 1 2
"""

import argparse
import json
import math

import torch
from experiments.constrained_hard_03.configs import cuau_sigma

from discrete_flow_sampler.constraints.two_hole_patch_swap_head import (
    TwoHolePatchSwapHead,
    bravais_patch_geometry,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.targets.cluster_expansion import (
    BinaryExpansionSpec,
    FixedCompositionClusterExpansionTarget,
)


def ordered_states(spec, phase):
    """The L1_0 (six) or L1_2 (four) ordered states of an fcc supercell whose
    positions are in Cartesian A with the conventional cube edge = the second
    neighbour distance: layer parity along an axis for L1_0, one simple-cubic
    sublattice for L1_2. (n_states, n_sites) in {-1, +1}."""
    positions = torch.tensor(spec.positions, dtype=torch.float64)
    distances = torch.cdist(positions, positions)
    nearest = distances[distances > 1e-6].min()  # raw, no wrap needed
    cube_edge = nearest * math.sqrt(2.0)  # fcc: a = sqrt(2) d_nn
    parity = torch.round(2 * positions / cube_edge).long() % 2  # (n, 3)
    if phase == "l10":
        states = []
        for axis in range(3):
            s = torch.where(parity[:, axis] == 0, 1.0, -1.0)
            states += [s, -s]
        return torch.stack(states)
    sublattice = parity[:, 0] * 2 + parity[:, 1]
    return torch.stack([torch.where(sublattice == k, 1.0, -1.0) for k in range(4)])


def near_ordered_states(spec, phase, n_swaps, n_states, generator):
    """Ordered states with `n_swaps` random unlike-pair swaps applied: the
    domain-wall-ridden neighbourhood the flow must anneal through."""
    refs = ordered_states(spec, phase)
    out = []
    for i in range(n_states):
        x = (
            refs[torch.randint(0, len(refs), (1,), generator=generator)]
            .clone()
            .flatten()
        )
        for _ in range(n_swaps):
            plus = torch.nonzero(x > 0).flatten()
            minus = torch.nonzero(x < 0).flatten()
            a = plus[torch.randint(0, len(plus), (1,), generator=generator)]
            b = minus[torch.randint(0, len(minus), (1,), generator=generator)]
            x[a], x[b] = -1.0, 1.0
        out.append(x)
    return torch.stack(out)


def random_slice_states(n_states, n_sites, n_plus, generator):
    """Uniform fixed-composition states in {-1, +1}, (n_states, n_sites)."""
    scores = torch.rand(n_states, n_sites, generator=generator)
    top = scores.topk(n_plus, dim=1).indices
    states = torch.full((n_states, n_sites), -1.0)
    states.scatter_(1, top, 1.0)
    return states


def unlike_pair_targets(target, states, pairs):
    """Delta_ij at t = 1 over the listed pairs, and the unlike-pair mask."""
    t = torch.ones(states.shape[0])
    delta = target.swap_log_ratio(states, t, pairs)
    unlike = states[:, pairs[:, 0]] != states[:, pairs[:, 1]]
    return delta, unlike


def r_squared(prediction, truth, mask):
    residual = ((prediction - truth) ** 2)[mask].sum()
    total = ((truth - truth[mask].mean()) ** 2)[mask].sum()
    return float(1.0 - residual / total)


def fit_head(
    head, target, pairs, train_states, held_out_states, steps, batch, lr, seed
):
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    held_delta, held_unlike = unlike_pair_targets(target, held_out_states, pairs)
    for step in range(steps):
        index = torch.randint(0, train_states.shape[0], (batch,), generator=generator)
        x = train_states[index]
        delta, unlike = unlike_pair_targets(target, x, pairs)
        scores = gather_pair_scores(head(x, torch.ones(batch)), pairs)
        loss = (((scores - delta) ** 2) * unlike).sum() / unlike.sum()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                held = gather_pair_scores(
                    head(held_out_states, torch.ones(held_out_states.shape[0])), pairs
                )
                print(
                    f"    step {step:5d} train mse {loss.item():.4f} "
                    f"held-out R^2 {r_squared(held, held_delta, held_unlike):.3f}",
                    flush=True,
                )
    with torch.no_grad():
        held = gather_pair_scores(
            head(held_out_states, torch.ones(held_out_states.shape[0])), pairs
        )
    return r_squared(held, held_delta, held_unlike)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="data/ce/cuau_fcc_4x4x4.json")
    parser.add_argument("--composition", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=500.0)
    parser.add_argument("--shells", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-held-out", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--states",
        default="uniform",
        help="'uniform' (the base), or 'l10+K' / 'l12+K': the ordered phase "
        "with K random unlike swaps applied (near-ordered, domain-wall states)",
    )
    args = parser.parse_args(argv)

    spec = BinaryExpansionSpec.from_json(args.spec)
    beta = 2.0 * cuau_sigma(args.temperature)
    target = FixedCompositionClusterExpansionTarget(
        spec, beta=beta, target_composition=args.composition
    )
    generator = torch.Generator().manual_seed(args.seed)
    if args.states == "uniform":
        train_states = random_slice_states(
            args.n_train, spec.n_sites, target.n_plus_target, generator
        )
        held_out_states = random_slice_states(
            args.n_held_out, spec.n_sites, target.n_plus_target, generator
        )
    else:
        phase, n_swaps = args.states.split("+")
        train_states = near_ordered_states(
            spec, phase, int(n_swaps), args.n_train, generator
        )
        held_out_states = near_ordered_states(
            spec, phase, int(n_swaps), args.n_held_out, generator
        )
    pairs = upper_tri_pairs(spec.n_sites, train_states.device)
    held_delta, held_unlike = unlike_pair_targets(target, held_out_states, pairs)
    print(
        f"{args.spec}: {spec.n_sites} sites, c={args.composition}, "
        f"T={args.temperature} K, states={args.states}, "
        f"held-out Delta std {held_delta[held_unlike].std():.3f} "
        f"over {int(held_unlike.sum())} unlike pairs"
    )

    results = {}
    for shells in args.shells:
        torch.manual_seed(args.seed)
        geometry = bravais_patch_geometry(
            spec.positions, spec.cell, patch_shells=shells
        )
        backbone = LeTFRateMatrix(
            d=spec.n_sites,
            vocab_size=2,
            hidden_dim=args.hidden_dim,
            n_layers=3,
            n_heads=4,
        )
        head = TwoHolePatchSwapHead(
            backbone, geometry=geometry, feature_dim=args.feature_dim
        )
        print(
            f"  shells={shells}: window {head.n_patch} sites, "
            f"pooled balls {geometry.level_sizes}"
        )
        results[shells] = fit_head(
            head,
            target,
            pairs,
            train_states,
            held_out_states,
            args.steps,
            args.batch,
            args.lr,
            args.seed,
        )
    print("held-out R^2 by shells:", {k: round(v, 3) for k, v in results.items()})
    if args.out:
        json.dump(
            {"args": vars(args), "held_out_r2": results}, open(args.out, "w"), indent=2
        )


if __name__ == "__main__":
    main()
