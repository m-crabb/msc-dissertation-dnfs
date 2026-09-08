"""Absolute on-slice free energy by thermodynamic integration (slice TI).

At 4x4 the gate scores the sampler against exact slice enumeration; at 8x8
the slice has C(64,32) ~ 1.8e18 states, so the absolute reference is built
here instead. The sampler's own estimate is a variational bound,
log Z >= E_Q[w] (paper Eq. 37), so its F sits at or above the truth (8x8
sigma_c archived replicates: F/d = -1.89703 +/- 0.00014).

On the fixed-composition slice log p_tilde(x) = sigma * x^T A x with A the
double-counted torus adjacency (x^T A x = 2 * sum_<ij> s_i s_j, matching
`initial_log_prob_ising` and `IsingTarget.A`). Then

    d(log Z_slice)/d(sigma) = <x^T A x>_sigma          (TI identity)
    log Z_slice(0)          = log C(d, n_plus)          (uniform slice)
    F(sigma)/d              = -log Z_slice(sigma) / (2 sigma d)

so integrating chain estimates of the mean energy over a sigma-grid from 0
to the operating point gives the reference in the convention of
`free_energy_lb_estimate` / `on_slice_free_energy_reference` (the factor
1/(2 sigma d) is the paper's beta = 2 sigma; the tests pin it at -2.98043
for 4x4, sigma=0.10).

Engine: the non-local numba Kawasaki runner, tau_int(energy) ~ 10 sweeps at
both 8x8 operating points; the local NN-swap runner (tau_int ~ 55 at
sigma_c) is an independent-dynamics cross-check. Quadrature is composite
Simpson with a half-grid check reported next to the statistical error, and
burn-in is max(1e4 sweeps, 20 * tau_int) with mode-seeded inits at sigma_c
as an equilibration probe.

Stages: `--stage validate4x4` runs the chain pipeline where the exact
answer is enumerable and prints estimate vs exact; `--stage run8x8`
produces the production reference at (0.10, 8x8) and (0.223, 8x8) plus the
neural-bias statement.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from experiments.constrained_hard_03.probe_analysis_8x8 import (
    batch_means_tau_int,
    kawasaki_burn_in_sweeps,
)

from discrete_flow_sampler.mcmc.kawasaki import (
    init_phase_separated,
    init_random_at_composition,
    run_chain,
    run_local_swap_chain_snapshots,
)

NEURAL_F_PER_SITE_8X8_SC = -1.89703  # archived replicate estimate (mean)
NEURAL_F_SE_8X8_SC = 0.00014  # cross-replicate standard error
BURN_IN_FLOOR_SWEEPS = 10_000

SEED_BASE = 7_000  # disjoint from probe seed ranges
CHAIN_SWEEPS = 300_000
REPLICATE_CHAINS = 4


# --------------------------------------------------------------------------
# quadrature and closed-form pieces
# --------------------------------------------------------------------------


def composite_simpson(grid, values):
    """Composite Simpson on a uniform grid; grid must have an even number of
    equal intervals (odd point count). Raises ValueError otherwise -- a
    silent fallback to trapezoid would change the quadrature bias order."""
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    n_intervals = len(grid) - 1
    if n_intervals < 2 or n_intervals % 2 != 0:
        raise ValueError("composite Simpson needs an even interval count >= 2")
    h = np.diff(grid)
    if not np.allclose(h, h[0], rtol=1e-10, atol=1e-14):
        raise ValueError("composite Simpson expects a uniform grid")
    return float(
        h[0]
        / 3.0
        * (
            values[0]
            + values[-1]
            + 4.0 * values[1:-1:2].sum()
            + 2.0 * values[2:-1:2].sum()
        )
    )


def _simpson_weights(grid):
    n = len(grid)
    h = (grid[-1] - grid[0]) / (n - 1)
    weights = np.full(n, 2.0)
    weights[1:-1:2] = 4.0
    weights[0] = weights[-1] = 1.0
    return weights * h / 3.0


def quadrature_error(grid, standard_errors):
    """SE of the Simpson integral for independent point estimates:
    Var = sum_i (w_i * SE_i)^2. Independence holds by construction (fresh
    chains per grid point, disjoint seeds)."""
    weights = _simpson_weights(np.asarray(grid, dtype=float))
    ses = np.asarray(standard_errors, dtype=float)
    return float(np.sqrt(((weights * ses) ** 2).sum()))


def uniform_slice_mean_energy(lattice_side):
    """<x^T A x> under the uniform half-filled slice (the sigma = 0 endpoint),
    in closed form. The fixed total magnetisation has zero variance, so
    sum_{i != j} E[x_i x_j] = -d, giving E[x_i x_j] = -1/(d-1) and
    <x^T A x>_0 = -(sum_ij A_ij)/(d-1) = -4d/(d-1) on the double-counted
    torus (4d directed bonds). Exact, zero statistical error."""
    d = lattice_side * lattice_side
    return -4.0 * d / (d - 1.0)


def lattice_energy_double_counted(states, lattice_side):
    """x^T A x for a batch of flat +/-1 states, via torus rolls: each site
    times the sum of its 4 neighbours (double-counted, matching target.A)."""
    grids = np.asarray(states, dtype=np.float64).reshape(-1, lattice_side, lattice_side)
    neighbour_sum = (
        np.roll(grids, 1, axis=1)
        + np.roll(grids, -1, axis=1)
        + np.roll(grids, 1, axis=2)
        + np.roll(grids, -1, axis=2)
    )
    return (grids * neighbour_sum).sum(axis=(1, 2))


def slice_ti_free_energy_per_site(grid, integrand, lattice_side, n_plus):
    """F(sigma_target)/d = -[log C(d, n_plus) + int_0^sigma <x^T A x>] / (2 sigma d)."""
    d = lattice_side * lattice_side
    sigma_target = float(grid[-1])
    log_z = (
        math.lgamma(d + 1)
        - math.lgamma(n_plus + 1)
        - math.lgamma(d - n_plus + 1)
        + composite_simpson(grid, integrand)
    )
    return -log_z / (2.0 * sigma_target * d)


# --------------------------------------------------------------------------
# chain estimation of the integrand
# --------------------------------------------------------------------------


def _one_chain_mean_energy(lattice_side, sigma, seed, sweeps, init="random"):
    """Post-burn-in mean of x^T A x from one non-local Kawasaki chain.

    `run_chain` records log_prob = sigma * x^T A x per proposal; dividing by
    sigma recovers the energy (sigma = 0 is the closed-form endpoint, never a
    chain). The trace is thinned to per-sweep before tau/burn-in analysis so
    tau_int is in sweeps, the burn-in rule's unit."""
    d = lattice_side * lattice_side
    rng = np.random.default_rng(seed)
    if init == "random":
        x0 = init_random_at_composition(d, 0.5, rng)
    else:  # "mode0" / "mode1": phase-separated, the equilibration probe
        x0 = init_phase_separated(lattice_side, 0 if init == "mode0" else 1)
    trace, _, _ = run_chain(x0, lattice_side, sigma, sweeps * d, seed)
    energy_per_sweep = np.asarray(trace, dtype=np.float64)[d - 1 :: d] / sigma
    tail = energy_per_sweep[BURN_IN_FLOOR_SWEEPS:]
    tau, _, _ = batch_means_tau_int(tail)
    burn = kawasaki_burn_in_sweeps(tau)
    kept = energy_per_sweep[burn:]
    if len(kept) < sweeps // 4:
        raise RuntimeError(
            f"burn-in {burn} sweeps consumed >3/4 of a {sweeps}-sweep chain "
            f"(tau_int {tau:.1f}); lengthen the chain rather than accept a "
            f"starved mean"
        )
    return float(kept.mean()), tau, burn


def grid_point_estimate(
    lattice_side, sigma, point_index, sweeps=CHAIN_SWEEPS, chains=REPLICATE_CHAINS
):
    """Mean +/- cross-chain SE of <x^T A x>_sigma from `chains` fresh chains."""
    means, taus = [], []
    for replicate in range(chains):
        mean, tau, _ = _one_chain_mean_energy(
            lattice_side,
            sigma,
            seed=SEED_BASE + 100 * point_index + replicate,
            sweeps=sweeps,
        )
        means.append(mean)
        taus.append(tau)
    means = np.asarray(means)
    se = means.std(ddof=1) / math.sqrt(chains)
    return float(means.mean()), float(se), float(np.max(taus))


def ti_reference(
    lattice_side,
    n_plus,
    sigma_target,
    n_intervals,
    label,
    seed_offset,
    sweeps=CHAIN_SWEEPS,
):
    """Full TI estimate at one operating point, with the half-grid check."""
    grid = np.linspace(0.0, sigma_target, n_intervals + 1)
    integrand = np.empty(len(grid))
    ses = np.empty(len(grid))
    taus = []
    integrand[0], ses[0] = uniform_slice_mean_energy(lattice_side), 0.0
    for index, sigma in enumerate(grid[1:], start=1):
        integrand[index], ses[index], tau = grid_point_estimate(
            lattice_side,
            float(sigma),
            point_index=seed_offset + index,
            sweeps=sweeps,
        )
        taus.append(tau)
    f_per_site = slice_ti_free_energy_per_site(grid, integrand, lattice_side, n_plus)
    d = lattice_side * lattice_side
    se_f = quadrature_error(grid, ses) / (2.0 * sigma_target * d)
    coarse = slice_ti_free_energy_per_site(
        grid[::2], integrand[::2], lattice_side, n_plus
    )
    return {
        "label": label,
        "lattice_side": lattice_side,
        "sigma": sigma_target,
        "grid_points": len(grid),
        "chain_sweeps": sweeps,
        "chains_per_point": REPLICATE_CHAINS,
        "f_per_site": f_per_site,
        "f_se": se_f,
        "half_grid_shift": coarse - f_per_site,
        "max_tau_int_sweeps": max(taus),
        "integrand": integrand.tolist(),
        "integrand_se": ses.tolist(),
        "grid": grid.tolist(),
    }


def sigma_c_cross_checks(result, lattice_side=8, sweeps=CHAIN_SWEEPS):
    """Two equilibration probes at the hardest grid point (sigma_c):
    mode-seeded non-local chains (phase-separated inits, both sides), and the
    local NN-swap runner -- different dynamics, same target."""
    sigma = result["sigma"]
    d = lattice_side * lattice_side
    mode_means = []
    for init in ("mode0", "mode1"):
        mean, _, _ = _one_chain_mean_energy(
            lattice_side,
            sigma,
            seed=SEED_BASE + 9_000 + len(mode_means),
            sweeps=sweeps,
            init=init,
        )
        mode_means.append(mean)
    local_means = []
    for replicate in range(2):
        rng = np.random.default_rng(SEED_BASE + 9_500 + replicate)
        x0 = init_random_at_composition(d, 0.5, rng).astype(np.int8)
        snapshots, _, _ = run_local_swap_chain_snapshots(
            x0,
            lattice_side,
            sigma,
            sweeps * d,
            SEED_BASE + 9_500 + replicate,
            thin=d,
        )
        energies = lattice_energy_double_counted(snapshots, lattice_side)
        tail = energies[BURN_IN_FLOOR_SWEEPS:]
        tau, _, _ = batch_means_tau_int(tail)
        local_means.append(float(energies[kawasaki_burn_in_sweeps(tau) :].mean()))
    return {"mode_seeded_means": mode_means, "local_runner_means": local_means}


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def validate_4x4():
    """Run the chain pipeline at 4x4 and score it against exact enumeration."""
    from experiments.constrained_hard_03.gate_4x4 import (
        on_slice_free_energy_reference,
    )

    from discrete_flow_sampler.diagnostics.metrics import (
        conditional_pmf_at_composition,
        enumerate_states,
        exact_log_probs,
    )
    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

    rows = []
    for sigma_target, n_intervals in ((0.10, 8), (0.223, 16)):
        estimate = ti_reference(
            4,
            8,
            sigma_target,
            n_intervals,
            label=f"v4x4-{sigma_target}",
            seed_offset=0 if sigma_target == 0.10 else 20,
        )
        target = FixedCompositionIsingTarget(
            D=4, sigma=sigma_target, target_composition=0.5
        )
        all_states = enumerate_states(16)
        log_pi = exact_log_probs(target, all_states)
        slice_states, _ = conditional_pmf_at_composition(
            all_states, log_pi, target.n_plus_target
        )
        exact = float(on_slice_free_energy_reference(target, slice_states.float()))
        rows.append(
            {
                **{
                    k: estimate[k]
                    for k in (
                        "sigma",
                        "f_per_site",
                        "f_se",
                        "half_grid_shift",
                        "max_tau_int_sweeps",
                    )
                },
                "f_exact": exact,
                "deviation": estimate["f_per_site"] - exact,
                "deviation_over_se": (estimate["f_per_site"] - exact)
                / estimate["f_se"],
            }
        )
        print(
            f"[slice-ti] 4x4 sigma={sigma_target}: "
            f"TI {estimate['f_per_site']:.5f} +/- {estimate['f_se']:.5f} "
            f"vs exact {exact:.5f} "
            f"({rows[-1]['deviation_over_se']:+.2f} SE); "
            f"half-grid shift {estimate['half_grid_shift']:+.1e}"
        )
    return rows


def run_8x8():
    results = []
    for sigma_target, n_intervals in ((0.10, 8), (0.223, 16)):
        result = ti_reference(
            8,
            32,
            sigma_target,
            n_intervals,
            label=f"p8x8-{sigma_target}",
            seed_offset=40 if sigma_target == 0.10 else 60,
        )
        print(
            f"[slice-ti] 8x8 sigma={sigma_target}: "
            f"F/d = {result['f_per_site']:.5f} +/- {result['f_se']:.5f} "
            f"(half-grid shift {result['half_grid_shift']:+.1e}, "
            f"max tau {result['max_tau_int_sweeps']:.1f} sweeps)"
        )
        results.append(result)
    checks = sigma_c_cross_checks(results[-1])
    print(
        f"[slice-ti] sigma_c cross-checks: mode-seeded means "
        f"{checks['mode_seeded_means']}, local-runner means "
        f"{checks['local_runner_means']} "
        f"(grid-point estimate {results[-1]['integrand'][-1]:.3f})"
    )
    bias = NEURAL_F_PER_SITE_8X8_SC - results[-1]["f_per_site"]
    combined = math.hypot(NEURAL_F_SE_8X8_SC, results[-1]["f_se"])
    print(
        f"[slice-ti] neural bias at sigma_c: {bias:+.5f} +/- {combined:.5f} "
        f"per site ({bias / combined:+.1f} combined SE); "
        f"variational bound predicts bias >= 0"
    )
    return {
        "points": results,
        "sigma_c_cross_checks": checks,
        "neural_bias_sc": {"bias": bias, "se": combined},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["validate4x4", "run8x8"])
    parser.add_argument("--out", default="results/03_hard/slice_ti_8x8")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = validate_4x4() if args.stage == "validate4x4" else run_8x8()
    out_path = out_dir / f"{args.stage}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[slice-ti] wrote {out_path}")


if __name__ == "__main__":
    main()
