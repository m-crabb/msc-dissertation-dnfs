"""Classical references for the alloy cells: Metropolis chains and a beta-ladder TI.

WHY. On the Ising torus the references are certified elsewhere (Wolff pools,
numba Kawasaki, mchammer, Ferdinand-Fisher). The Cu-Au expansion has none of
those, and mchammer cannot read its CLEASE-format ECIs, so the reference is
built here on the SAME exported energy the samplers use (`BinaryExpansionSpec`),
which removes the energy-convention surface entirely: any disagreement between
sampler and chain is then a sampling statement, not a units one.

TWO ENSEMBLES, ONE MOVE EACH. Free composition (the unconstrained and the
penalised targets): single-site Metropolis, propose one flip per chain per
step, accept with min(1, exp(delta log p)). Fixed composition (canonical):
Kawasaki, propose one unlike pair per chain, accept the same way; the move set
conserves the count by construction, exactly as the swap sampler does. Both
use `target.log_prob`, so the penalised target is the same chain with the
penalty inside delta log p -- the VC-SGC chain in its penalty form.

ABSOLUTE FREE ENERGY BY THERMODYNAMIC INTEGRATION IN beta. A chain returns
averages, never a normaliser, so F needs a path from a solvable point:

    d log Z / d beta = -<E>_beta,   log Z(0) = log |Omega|,
    log Z(beta) = log |Omega| - int_0^beta <E>_b db,

with |Omega| = 2^d for the free ensemble and C(d, n_plus) on a slice. This is
the coupling-ladder integration icet's ThermodynamicIntegrationEnsemble runs
(there from the ideal solution up to the real energy), the comparator the
hard sampler is measured against: one draw per composition against a ladder
of chains per composition. Composite Simpson on a uniform beta grid; the
integrand is analytic on a finite cell; the half-grid shift is reported next
to the statistical error so quadrature bias cannot hide inside it.

FAILURE MODES GUARDED. Ordering at 500 K freezes single-flip and swap chains
(the alloy's own critical slowing): every point runs R independent chains
from random starts, reports the cross-chain SE, and the TI ladder walks beta
UPWARD so each rung starts from the previous rung's equilibrated states (the
frozen competitor rule: burn-in then thinned records). On the 16-site cell
everything here is checked against exact enumeration in the tests.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from discrete_flow_sampler.targets.cluster_expansion import (
    BinaryExpansionSpec,
    ClusterExpansionTarget,
    FixedCompositionClusterExpansionTarget,
)

K_B_EV = 8.617333262e-5


def build_target(spec, temperature_K, composition=None, penalty=0.0, canonical=False):
    beta = 1.0 / (K_B_EV * temperature_K)
    if canonical:
        return FixedCompositionClusterExpansionTarget(spec, beta=beta, target_composition=composition)
    return ClusterExpansionTarget(
        spec, beta=beta, target_composition=composition,
        composition_penalty_strength=penalty,
    )


def metropolis_sweeps(target, x, n_sweeps, generator, canonical):
    """Advance every chain in `x` by n_sweeps sweeps (d proposals per sweep), in place.

    Vectorised over chains: one proposal per chain per step. Flip: a random
    site. Swap: a random unlike pair, drawn by picking a random up-site and a
    random down-site per chain (uniform over unlike pairs, which is what the
    swap sampler's move set is too). Acceptance uses the full log_prob
    difference from two energy evaluations, so the penalty, the bias and the
    expansion all enter through the one function the samplers use.
    """
    n_chains, d = x.shape
    rows = torch.arange(n_chains)
    log_p = target.log_prob(x)
    for _ in range(n_sweeps * d):
        y = x.clone()
        if canonical:
            up = (x > 0).float()
            i = torch.multinomial(up, 1, generator=generator).squeeze(1)
            j = torch.multinomial(1.0 - up, 1, generator=generator).squeeze(1)
            y[rows, i], y[rows, j] = x[rows, j], x[rows, i]
        else:
            i = torch.randint(d, (n_chains,), generator=generator)
            y[rows, i] = -x[rows, i]
        log_p_new = target.log_prob(y)
        accept = torch.log(torch.rand(n_chains, generator=generator)) < (log_p_new - log_p)
        x[accept] = y[accept]
        log_p = torch.where(accept, log_p_new, log_p)
    return x


def run_chains(target, n_chains, burn_in_sweeps, n_records, thin_sweeps, seed, canonical,
               x0=None):
    """Thinned records from n_chains independent chains: (n_records, n_chains, d)."""
    generator = torch.Generator().manual_seed(seed)
    x = target.sample_base(n_chains, device="cpu") if x0 is None else x0.clone()
    metropolis_sweeps(target, x, burn_in_sweeps, generator, canonical)
    records = []
    for _ in range(n_records):
        metropolis_sweeps(target, x, thin_sweeps, generator, canonical)
        records.append(x.clone())
    return torch.stack(records), x


def observables(target, states):
    """Per-chain means of energy per site, composition and NN bond correlation."""
    n_records, n_chains, d = states.shape
    flat = states.reshape(-1, d)
    energy = target.spec.energy(flat).reshape(n_records, n_chains) / d
    composition = ((flat + 1) / 2).mean(-1).reshape(n_records, n_chains)
    quadratic = (flat @ target.A * flat).sum(-1).reshape(n_records, n_chains)
    bond = quadratic / target.A.sum()
    return {
        "energy_per_site": energy.mean(0), "composition": composition.mean(0),
        "nn_correlation": bond.mean(0),
    }


def beta_ladder_free_energy(spec, temperature_K, composition, canonical, n_grid, n_chains,
                            burn_in_sweeps, n_records, thin_sweeps, seed):
    """log Z(beta) at the target temperature by Simpson over a uniform beta grid.

    Returns F per site in eV, its cross-chain SE, and the half-grid shift.
    Chains are carried up the ladder (each rung starts from the last rung's
    final states) so the cold end is reached through equilibrated states.
    """
    beta_max = 1.0 / (K_B_EV * temperature_K)
    if n_grid % 2 == 0:
        raise ValueError("Simpson needs an odd number of grid points")
    grid = np.linspace(0.0, beta_max, n_grid)
    d = spec.n_sites
    mean_energy, se_energy = [], []
    x = None
    for k, beta in enumerate(grid):
        target = build_target(spec, 1.0 / (K_B_EV * max(beta, 1e-12)), composition,
                              canonical=canonical)
        if beta == 0.0:
            # uniform on the state space (or the slice): the exact mean energy
            # of the base is a sample average over a large independent draw
            x = target.sample_base(n_chains, device="cpu")
            states = torch.stack([target.sample_base(n_chains, device="cpu")
                                  for _ in range(n_records)])
        else:
            states, x = run_chains(target, n_chains, burn_in_sweeps, n_records,
                                   thin_sweeps, seed + k, canonical, x0=x)
        per_chain = observables(target, states)["energy_per_site"] * d
        mean_energy.append(float(per_chain.mean()))
        se_energy.append(float(per_chain.std(unbiased=True) / math.sqrt(n_chains)))
    mean_energy, se_energy = np.array(mean_energy), np.array(se_energy)

    def simpson(values, xs):
        h = xs[1] - xs[0]
        weights = np.ones_like(values)
        weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
        return h / 3.0 * float(np.dot(weights, values)), h / 3.0 * weights

    integral, weights = simpson(mean_energy, grid)
    log_omega = (math.lgamma(d + 1) - math.lgamma(round(composition * d) + 1)
                 - math.lgamma(d - round(composition * d) + 1)) if canonical else d * math.log(2.0)
    log_z = log_omega - integral
    free_energy_per_site = -log_z / beta_max / d
    se = float(np.sqrt(np.sum((weights * se_energy) ** 2))) / beta_max / d
    coarse, _ = simpson(mean_energy[::2], grid[::2])
    half_grid_shift = (coarse - integral) / beta_max / d
    return {
        "temperature_K": temperature_K, "composition": composition, "canonical": canonical,
        "free_energy_per_site_eV": free_energy_per_site, "se": se,
        "half_grid_shift": half_grid_shift, "log_z": log_z,
        "grid": grid.tolist(), "mean_energy": mean_energy.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--mode", choices=["chain", "ti"], required=True)
    parser.add_argument("--temperature", type=float, default=500.0)
    parser.add_argument("--composition", type=float, default=None)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=2000, help="sweeps")
    parser.add_argument("--n-records", type=int, default=200)
    parser.add_argument("--thin", type=int, default=10, help="sweeps between records")
    parser.add_argument("--n-grid", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = BinaryExpansionSpec.from_json(args.spec)
    if args.mode == "chain":
        target = build_target(spec, args.temperature, args.composition, args.penalty,
                              args.canonical)
        states, _ = run_chains(target, args.n_chains, args.burn_in, args.n_records,
                               args.thin, args.seed, args.canonical)
        obs = observables(target, states)
        payload = {k: {"mean": float(v.mean()), "se": float(v.std(unbiased=True) / math.sqrt(len(v)))}
                   for k, v in obs.items()}
        payload["n_states"] = int(states.shape[0] * states.shape[1])
        torch.save(states.reshape(-1, spec.n_sites).to(torch.int8), args.out.with_suffix(".pt"))
    else:
        payload = beta_ladder_free_energy(
            spec, args.temperature, args.composition, args.canonical, args.n_grid,
            args.n_chains, args.burn_in, args.n_records, args.thin, args.seed)
    payload.update(spec=str(args.spec), temperature_K=args.temperature, mode=args.mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: v for k, v in payload.items() if k not in ("grid", "mean_energy")}, indent=1))


if __name__ == "__main__":
    main()
