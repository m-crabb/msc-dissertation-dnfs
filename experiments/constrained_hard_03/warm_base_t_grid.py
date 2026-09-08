"""Shape of the drive variance along the annealing path, and the conditional
per-swap ceiling.

The warm-base design measures Var[D] only at t = 1 (reference draws) and t = 0
(base draws), which gives the integral

    Delta c = c_1 - c_0 = int_0^1 Var_{p_t}[D] dt

exactly, but says nothing about the shape, and the shape decides the design:

    warm base    Delta c / Var_{p_1}[D] = 66.1 / 38.7  = 1.71x
    uniform base                        = 135.0/170.2 = 0.79x

(measured at d256 by warm_base_offline_table.py), i.e. the warm base's drive
variance must average 1.7x its t=1 value across the path while the uniform
base's averages 0.8x.  Its difficulty is therefore front-loaded near t = 0,
exactly where the design has no measurement and where three other hazards live
(the closed-form swap_log_ratio break, the epsilon-floor clamp stress and the
extra |Dlog eta| range all peak at t->0).

Measured at d256 by this script: the warm base sits below the uniform base at
every t, so the design is not overturned, but the ratio runs from 0.70x at
t = 0 to 0.24x at t = 1, path-averaged 0.49x.  The two curves peak at opposite
ends (uniform back-loaded at t ~ 0.8, warm front-loaded at t ~ 0.4), so an
endpoint-only comparison at t = 1 pits the uniform base near its worst against
the warm base at its best.  Quote the path-averaged number.

The annealing path is the exponential family

    p_t(x)  ∝  p~_t(x) = eta(x)^(1-t) rho(x)^t = exp( log eta(x) + t D(x) ),
    D(x) = log rho(x) - log eta(x) = sigma x^T A x - log eta(x)

with D as the sufficient statistic, so

    c_t      = d/dt log Z_t = E_{p_t}[D]
    dc_t/dt  = Var_{p_t}[D]  >= 0.

Endpoint evaluation gives int_0^1 Var dt exactly; this probe measures the
integrand pointwise by running a composition-preserving (Kawasaki) Metropolis
chain against p~_t at each t on a grid, then checks the trapezoid integral back
against the endpoint Delta c.  If the measured curve does not integrate to the
independently known Delta c, the chain has not mixed and no shape claim is made.

Swapping sites i, j with x_i = a != b = x_j leaves the i-j cross term 2 A_ij a b
unchanged (it becomes 2 A_ij b a), so

    Delta(x^T A x) = 2 (b - a) (h_i - h_j) - 2 A_ij (b - a)^2,   h = A x

the same identity the production closed form uses, but applied here only to the
target term sigma x^T A x, never to the base.  The base's contribution to the
acceptance ratio comes from a full `log_density` call, i.e. the marginal mixture
logsumexp; using eta_k for the component a state was "drawn from" would break
normalisation and every weight downstream.

Conditional ceiling: max Delta S per swap = 2z = 16 on the square torus (12 if
restricted to adjacent pairs), attained only when |h_i - h_j| = 8 -- two
isolated point defects in two oppositely-saturated domains at once, rare at
sigma_c.  So the measured 3.44 achieved edge-units per swap ("22% of ceiling")
is measured against a bound that may never bind.  This probe instead reports,
for states actually drawn from p_t, the distribution of Delta S over all valid
pairs: the oracle max is 9.18 at d64 and 14.44 at d256, a uniformly random valid
swap destroys 7.61 / 9.39 edge-units, so an achieved +3.44 captures 65.8% (d64)
/ 53.8% (d256) of the random-to-oracle range -- a capture that falls with
lattice size.

Run:  pixi run -e default python warm_base_t_grid.py [--side 8] [--steps 6000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from warm_base_offline_table import load_d64, load_d256
from warm_base_reference import (
    BlockOccupancyBase,
    UniformSliceBase,
    quadratic_form,
    torus_adjacency,
)


def torus_neighbours(lattice_side: int) -> np.ndarray:
    """(d, 4) array of the four torus neighbours of every site.

    Same site indexing as `torus_adjacency` (row-major flat), so `h = A x`
    equals `x[neighbours].sum(axis=-1)` site-wise.
    """
    d = lattice_side * lattice_side
    rows, cols = np.divmod(np.arange(d), lattice_side)
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    return np.stack(
        [
            ((rows + dr) % lattice_side) * lattice_side + (cols + dc) % lattice_side
            for dr, dc in shifts
        ],
        axis=1,
    )


class AnnealedKawasakiChain:
    """Metropolis on the fixed-composition slice targeting p~_t.

    One proposal per chain per step: pick a uniformly random up-site and a
    uniformly random down-site and swap them.  Composition-preserving, so every
    state stays on the slice and no off-slice proposal ever tests the base's
    support condition.

    Acceptance uses log p~_t = log eta + t D, i.e.

        Dlog p~_t = t * sigma * Delta(x^T A x) + (1 - t) * Dlog eta

    with the target term from the closed form above and the base term from a
    full `log_density` call on the proposed batch.
    """

    def __init__(
        self,
        base,
        lattice_side: int,
        sigma: float,
        n_chains: int,
        rng: np.random.Generator,
    ) -> None:
        self.base = base
        self.lattice_side = lattice_side
        self.d = lattice_side * lattice_side
        self.sigma = sigma
        self.rng = rng
        self.adjacency = torus_adjacency(lattice_side)
        self.neighbours = torus_neighbours(lattice_side)
        self.is_uniform_base = isinstance(base, UniformSliceBase)

        self.spins = base.sample(n_chains, rng).astype(np.int8)
        self.n_chains = n_chains
        self.field = self.spins[:, self.neighbours].sum(axis=-1).astype(np.int16)
        self.log_eta = base.log_density(self.spins)
        self.n_accepted = 0
        self.n_proposed = 0

    def _propose_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """One (up-site, down-site) pair per chain, uniformly at random.

        argsort of per-row uniforms is a uniform random permutation, so taking
        the first up-site and first down-site it encounters is a uniform draw
        from each group -- the idiom the production `sample_base` uses.
        """
        keys = self.rng.random((self.n_chains, self.d))
        up_keys = np.where(self.spins > 0, keys, np.inf)
        down_keys = np.where(self.spins < 0, keys, np.inf)
        up_first = np.argmin(up_keys, axis=1)
        down_first = np.argmin(down_keys, axis=1)
        return up_first, down_first

    def step(self, t: float) -> None:
        site_up, site_down = self._propose_pair()
        rows = np.arange(self.n_chains)

        # a = x[site_up] = +1, b = x[site_down] = -1, so (b - a) = -2.
        spin_delta = -2
        field_up = self.field[rows, site_up].astype(np.int32)
        field_down = self.field[rows, site_down].astype(np.int32)
        adjacent = self.adjacency[site_up, site_down].astype(np.int32)
        delta_quadratic = (
            2 * spin_delta * (field_up - field_down) - 2 * adjacent * spin_delta**2
        )
        delta_log_target = t * self.sigma * delta_quadratic

        if self.is_uniform_base:
            delta_log_eta = np.zeros(self.n_chains)
            proposed_log_eta = self.log_eta
        else:
            proposed = self.spins.copy()
            proposed[rows, site_up] = -1
            proposed[rows, site_down] = 1
            proposed_log_eta = self.base.log_density(proposed)
            delta_log_eta = proposed_log_eta - self.log_eta

        delta_log_p = delta_log_target + (1.0 - t) * delta_log_eta
        accept = np.log(self.rng.random(self.n_chains)) < delta_log_p
        self.n_proposed += self.n_chains
        self.n_accepted += int(accept.sum())
        if not accept.any():
            return

        moved = rows[accept]
        up_moved, down_moved = site_up[accept], site_down[accept]
        self.spins[moved, up_moved] = -1
        self.spins[moved, down_moved] = 1
        # h_k += (b - a) (A_ki - A_kj); np.add.at because i and j may share
        # neighbours, in which case the two updates must both land.
        np.add.at(self.field, (moved[:, None], self.neighbours[up_moved]), spin_delta)
        np.add.at(
            self.field, (moved[:, None], self.neighbours[down_moved]), -spin_delta
        )
        self.log_eta = (
            np.where(accept, proposed_log_eta, self.log_eta)
            if not self.is_uniform_base
            else self.log_eta
        )

    def drive(self) -> np.ndarray:
        """D(x) = log rho(x) - log eta(x) for the current ensemble."""
        return self.sigma * quadratic_form(self.spins, self.adjacency) - self.log_eta


def run_t_grid(
    base,
    lattice_side: int,
    sigma: float,
    t_grid: np.ndarray,
    n_chains: int,
    n_steps: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for t in t_grid:
        chain = AnnealedKawasakiChain(base, lattice_side, sigma, n_chains, rng)
        drives = []
        for step_index in range(n_steps):
            chain.step(float(t))
            if step_index >= burn_in and (step_index - burn_in) % thin == 0:
                drives.append(chain.drive())
        samples = np.concatenate(drives)
        out.append(
            {
                "t": float(t),
                "c_t": float(samples.mean()),
                "var_D": float(samples.var()),
                "n_samples": int(samples.size),
                "accept_rate": chain.n_accepted / max(chain.n_proposed, 1),
            }
        )
        print(
            f"    t={t:4.2f}  c_t={samples.mean():12.3f}  "
            f"Var[D]={samples.var():10.3f}  acc={out[-1]['accept_rate']:.3f}",
            flush=True,
        )
    return out


def conditional_ceiling(
    spins: np.ndarray, lattice_side: int, max_states: int, rng: np.random.Generator
) -> dict:
    """Distribution of Delta S over all valid (up, down) pairs, per state.

    Delta S = (b - a)(h_i - h_j) - A_ij (b - a)^2 with (b - a) = -2 for an
    (up, down) ordered pair.  Returns the oracle max, the uniform-random mean,
    and upper quantiles -- the denominator the "22% of ceiling" claim needs.
    """
    adjacency = torus_adjacency(lattice_side)
    neighbours = torus_neighbours(lattice_side)
    if spins.shape[0] > max_states:
        spins = spins[rng.choice(spins.shape[0], max_states, replace=False)]
    field = spins[:, neighbours].sum(axis=-1).astype(np.int32)

    per_state_max, per_state_mean, pooled = [], [], []
    for state, h in zip(spins, field):
        up = np.flatnonzero(state > 0)
        down = np.flatnonzero(state < 0)
        # (b - a) = -2 for every (up, down) pair.
        delta_s = (
            -2 * (h[up][:, None] - h[down][None, :]) - 4 * adjacency[np.ix_(up, down)]
        )
        per_state_max.append(delta_s.max())
        per_state_mean.append(delta_s.mean())
        pooled.append(delta_s.ravel())
    pooled = np.concatenate(pooled)
    return {
        "n_states": int(spins.shape[0]),
        "theoretical_max": 16,
        "oracle_max_mean": float(np.mean(per_state_max)),
        "oracle_max_p50": float(np.median(per_state_max)),
        "random_pair_mean": float(np.mean(per_state_mean)),
        "pooled_p99": float(np.quantile(pooled, 0.99)),
        "pooled_p999": float(np.quantile(pooled, 0.999)),
        "pooled_max": float(pooled.max()),
        "achieved_per_swap": 3.44,
    }


def summarise(
    name: str, grid: list[dict], var_at_one: float, delta_c_endpoint: float
) -> dict:
    t = np.array([row["t"] for row in grid])
    var = np.array([row["var_D"] for row in grid])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    integral = float(trapezoid(var, t))
    return {
        "base": name,
        "integral_trapezoid": integral,
        "delta_c_endpoint": delta_c_endpoint,
        "integral_over_endpoint": integral / delta_c_endpoint
        if delta_c_endpoint
        else None,
        "var_at_t1": var_at_one,
        "mean_var_over_var_at_t1": float(var.mean() / var_at_one)
        if var_at_one
        else None,
        "front_loading_ratio": integral / var_at_one if var_at_one else None,
        "grid": grid,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=8, choices=(8, 16))
    ap.add_argument("--sigma", type=float, default=0.223)
    ap.add_argument("--chains", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--burn-in", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=50)
    ap.add_argument("--t-points", type=int, default=11)
    ap.add_argument("--ceiling-states", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    started = time.time()
    if args.side == 8:
        reference, _ = load_d64()
    else:
        reference = load_d256()
    n_up = int((reference[0] > 0).sum())
    print(
        f"reference: {reference.shape[0]} draws, D={args.side}, N_A={n_up}, "
        f"sigma={args.sigma}"
    )

    bases = [
        ("uniform", UniformSliceBase(args.side, n_up)),
        ("B(2,w) K=4", BlockOccupancyBase.fit(reference, args.side, 2, n_offsets=4)),
    ]
    t_grid = np.linspace(0.0, 1.0, args.t_points)
    adjacency = torus_adjacency(args.side)
    log_rho_ref = args.sigma * quadratic_form(reference, adjacency)

    results = {"config": vars(args), "bases": []}
    for name, base in bases:
        print(f"\n{name}:")
        # Endpoint Delta c, the independent value the curve must integrate to.
        log_eta_ref = base.log_density(reference)
        rng = np.random.default_rng(args.seed + 1)
        base_draws = base.sample(min(20000, reference.shape[0]), rng)
        drive_at_one = log_rho_ref - log_eta_ref
        drive_at_zero = args.sigma * quadratic_form(
            base_draws, adjacency
        ) - base.log_density(base_draws)
        delta_c = float(drive_at_one.mean() - drive_at_zero.mean())
        var_at_one = float(drive_at_one.var())
        print(f"    endpoint Delta c = {delta_c:.3f}, Var_p1[D] = {var_at_one:.3f}")

        grid = run_t_grid(
            base,
            args.side,
            args.sigma,
            t_grid,
            args.chains,
            args.steps,
            args.burn_in,
            args.thin,
            args.seed,
        )
        summary = summarise(name, grid, var_at_one, delta_c)
        ceiling_rng = np.random.default_rng(args.seed + 2)
        summary["conditional_ceiling_at_t1"] = conditional_ceiling(
            reference, args.side, args.ceiling_states, ceiling_rng
        )
        results["bases"].append(summary)

        print(
            f"    trapezoid int Var dt = {summary['integral_trapezoid']:.3f} "
            f"vs endpoint {delta_c:.3f} "
            f"(ratio {summary['integral_over_endpoint']:.3f})"
        )
        print(
            f"    FRONT-LOADING  int Var dt / Var_p1[D] = "
            f"{summary['front_loading_ratio']:.3f}"
        )

    print(f"\nelapsed {time.time() - started:.1f}s")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
