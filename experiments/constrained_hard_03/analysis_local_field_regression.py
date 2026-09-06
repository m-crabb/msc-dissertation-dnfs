"""How much of a trained 4x4 swap head is the hole-excluded local field?

For binary tokens exact swap antisymmetry forces G(i,j|x) = (x_i - x_j) S_ij(x)
with S blind to both holes, and for the Ising target the exact energy change
under a swap is  Delta E = sigma (x_i - x_j)(h_i - h_j)  with
h_i = sum_{k in N(i), k != j} x_k  the local field at i EXCLUDING the swap
partner (the i-j bond is invariant under the swap). So the equilibrium
log-ratio is a rank-1, linear, hole-subtracted per-site field difference.

This script asks how much of each trained head's S is (a) that linear field,
(b) a held-out cell-mean lookup per pair over the <=8 neighbour spins, and
(c) the lookup residual's association with exterior energy. Enumerates the
whole fixed-composition slice (C(16,8) = 12,870 states), so the linear fits
cover that space up to numerical error. Lookup scores still depend on the
fit split, finite cell counts and unseen-cell fallback; their residuals
are not by construction non-local. Archived lookup scores used centred
residual variance; recomputed scores use held-out SSE/SST. CPU, seconds.
"""

import argparse
import itertools
import json
from pathlib import Path

import torch

from experiments.constrained_hard_03.gate_4x4 import load_run
from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs


def enumerate_slice(d, n_plus):
    rows = []
    for ups in itertools.combinations(range(d), n_plus):
        row = torch.full((d,), -1.0)
        row[list(ups)] = 1.0
        rows.append(row)
    return torch.stack(rows)


def r_squared(target, design):
    """Least-squares R^2 of `target` (N,) on columns of `design` (N, k).

    float64 with standardised columns: powers like E_ext^2 * delta_h span
    orders of magnitude and an fp32 solve returned a superset design scoring
    BELOW its subset (conditioning, not signal).
    """
    design = design.double()
    scale = design.std(0).clamp(min=1e-12)
    scale[design.std(0) == 0] = 1.0
    coef = torch.linalg.lstsq(design / scale, target.double().unsqueeze(1)).solution
    residual = target.double() - ((design / scale) @ coef).squeeze(1)
    return float(1.0 - residual.var() / target.double().var())


def lookup_r_squared(target, keys, fit_mask):
    """Held-out cell means; unseen keys predict the fit-set global mean.

    R^2 = 1 - sum(residual^2) / sum((held_y - mean(held_y))^2).
    Held-out residuals need not average zero, so their centred variance
    would hide prediction bias. This split-dependent fit is not an exact
    bound on the capacity of arbitrary local functions.
    """
    unique, inverse = torch.unique(keys, return_inverse=True)
    fit = fit_mask.float()
    sums = torch.zeros(len(unique)).index_add_(0, inverse, target * fit)
    counts = torch.zeros(len(unique)).index_add_(0, inverse, fit)
    means = torch.where(counts > 0, sums / counts.clamp(min=1), target[fit_mask].mean())
    held = ~fit_mask
    residual = target[held] - means[inverse][held]
    total = (target[held] - target[held].mean()).square().sum()
    return float(1.0 - residual.square().sum() / total), len(unique), residual


def analyse(run_dir, t_value, device="cpu"):
    head, target = load_run(Path(run_dir), device)
    d = target.d
    A = target.A
    x = enumerate_slice(d, d // 2)
    pairs = upper_tri_pairs(d, device)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    t = torch.full((x.shape[0],), t_value)
    with torch.no_grad():
        G = torch.cat([head(xb, tb) for xb, tb in zip(x.split(1024), t.split(1024))])
    G = G[:, i_idx, j_idx]                                           # (N, P)
    xi, xj = x[:, i_idx], x[:, j_idx]
    differing = xi != xj
    S = (G / (xi - xj))[differing]
    # hole-excluded local fields: field at i minus the partner's bond
    field = x @ A                                                    # (N, d)
    h_i = field[:, i_idx] - A[i_idx, j_idx] * xj
    h_j = field[:, j_idx] - A[i_idx, j_idx] * xi
    delta_h, sum_h = (h_i - h_j)[differing], (h_i + h_j)[differing]
    adjacent = A[i_idx, j_idx].expand_as(xi)[differing]
    # exterior energy: all bonds not touching either hole (blind, a hole-
    # subtracted BOND sum); E = E_ext + (x_i - x_j)(h_i - h_j)/2 - A_ij on
    # differing pairs, so (E_ext, h_i, h_j, A_ij) carry everything about E.
    energy = -0.5 * (x * field).sum(1, keepdim=True)                 # -sum_bonds x_k x_l
    E_ext = (energy + xi * h_i + xj * h_j + A[i_idx, j_idx] * xi * xj)[differing]
    ones = torch.ones_like(S)
    linear_field = torch.stack([ones, delta_h], 1)
    local_quadratic = torch.stack(
        [ones, delta_h, sum_h, delta_h**2, delta_h * sum_h, adjacent, adjacent * delta_h], 1
    )
    with_exterior_energy = torch.cat(
        [local_quadratic, torch.stack([E_ext, E_ext * delta_h, E_ext**2, E_ext**2 * delta_h], 1)], 1
    )
    # Lookup over the neighbourhood spins of i and j (holes excluded).
    neigh_mask = ((A[i_idx] + A[j_idx]) > 0).float()                 # (P, d)
    neigh_mask[torch.arange(len(i_idx)), i_idx] = 0
    neigh_mask[torch.arange(len(i_idx)), j_idx] = 0
    bits = ((x.unsqueeze(1) > 0).float() * neigh_mask.unsqueeze(0))  # (N, P, d)
    powers = 2.0 ** torch.arange(d)
    pattern = (bits * powers).sum(-1)
    pair_id = torch.arange(len(i_idx)).expand(x.shape[0], -1)
    keys = (pair_id * 2**d + pattern)[differing].long()
    fit_mask = torch.rand(S.shape[0], generator=torch.Generator().manual_seed(0)) < 0.5
    local_r2, n_cells, local_residual = lookup_r_squared(S, keys, fit_mask)
    # Regress the held-out lookup residual on exterior energy and Delta h.
    # This association also includes lookup estimation error; the legacy
    # JSON key below is retained without asserting a purely nonlocal share.
    held = ~fit_mask
    E_h, dh_h = E_ext[held], delta_h[held]
    energy_design = torch.stack(
        [torch.ones_like(E_h), E_h, E_h**2, E_h * dh_h, E_h**2 * dh_h], 1
    )
    residual_on_energy = r_squared(local_residual, energy_design)
    return {
        "run": Path(run_dir).name, "t": t_value,
        "lookup_protocol": "pair_bitmask_heldout_sse_v2",
        "r2_linear_field": r_squared(S, linear_field),
        "r2_local_quadratic": r_squared(S, local_quadratic),
        "r2_with_exterior_energy": r_squared(S, with_exterior_energy),
        "r2_any_local": local_r2, "n_local_cells": n_cells,
        "nonlocal_share_explained_by_E_ext": residual_on_energy,
        "S_std": float(S.std()),
        "G_same_token_max": float(G[~differing].abs().max()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--t", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = [analyse(r, t) for r in args.run_dirs for t in args.t]
    print(f"{'run':46s} {'t':>4s} {'linear':>7s} {'quad':>7s} {'+E_ext':>7s} "
          f"{'local':>7s} {'resid~E':>8s} {'S_std':>6s}")
    for r in rows:
        print(f"{r['run'][16:62]:46s} {r['t']:4.1f} {r['r2_linear_field']:7.3f} "
              f"{r['r2_local_quadratic']:7.3f} {r['r2_with_exterior_energy']:7.3f} "
              f"{r['r2_any_local']:7.3f} {r['nonlocal_share_explained_by_E_ext']:8.3f} "
              f"{r['S_std']:6.3f}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
