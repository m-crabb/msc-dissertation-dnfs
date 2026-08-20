"""Self-checks for `warm_base_reference.py`, run before any number is believed.

The two that gate every downstream number:
  (i)  logsumexp of the log-density over ALL slice states at D = 4 equals 0
       -- i.e. the closed form really is exactly normalised on the slice, which
       is requirement (ii) of the warm-base contract (log Z_0 = 0 is assumed by
       `free_energy_lb_estimate` and `smc_log_z_estimate`);
  (ii) empirical draw frequencies match the density (chi-square) at D = 4
       -- i.e. the backwards-DP sampler really samples the density it claims,
       which is requirement (i) (an approximate base makes every importance
       weight wrong by an unmeasurable per-particle amount).

Plus the structural properties the design leans on: Z2 symmetry, translation
invariance of the K = b^2 mixture, agreement of the adjacency with the
library's, positivity on the whole slice, and the degenerate-tiling identity
b = D  =>  uniform on the slice.

Run:  pixi run -e default python warm_base_selftest.py
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chisquare

from warm_base_reference import (
    BlockOccupancyBase,
    UniformSliceBase,
    log_binom,
    nn_correlation,
    torus_adjacency,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name, detail))


def all_slice_states(lattice_side: int, n_up: int) -> np.ndarray:
    d = lattice_side * lattice_side
    combos = list(itertools.combinations(range(d), n_up))
    states = -np.ones((len(combos), d), dtype=np.int8)
    for row, sites in enumerate(combos):
        states[row, list(sites)] = 1
    return states


def translate(spins: np.ndarray, lattice_side: int, dr: int, dc: int) -> np.ndarray:
    grid = spins.reshape(-1, lattice_side, lattice_side)
    return np.roll(np.roll(grid, dr, axis=1), dc, axis=2).reshape(spins.shape)


# --------------------------------------------------------------------------
# 0. adjacency agrees with the library's
# --------------------------------------------------------------------------
try:
    import torch

    from discrete_flow_sampler.targets.ising import IsingTarget

    for side in (4, 8):
        target = IsingTarget(D=side, sigma=0.223, device=torch.device("cpu"))
        same = np.array_equal(target.A.cpu().numpy(), torus_adjacency(side))
        check(f"adjacency matches ising.py at D={side}", same,
              f"sum(A) = {torus_adjacency(side).sum():.0f} = 4d")
except Exception as exc:                                     # pragma: no cover
    check("adjacency matches ising.py", False, f"could not import: {exc}")

# --------------------------------------------------------------------------
# 1. exact normalisation on the slice at D = 4
# --------------------------------------------------------------------------
SIDE, N_UP = 4, 8
states = all_slice_states(SIDE, N_UP)
n_states = states.shape[0]

rng = np.random.default_rng(0)
A4 = torus_adjacency(SIDE)
# The fitting population is an EXACT draw from the D = 4 slice Ising law at the
# project's sigma_c -- at d = 16 the slice is enumerable, so no MCMC is needed
# and the fitted w is exactly the population w the production probe fits by
# Monte Carlo.  Using the real target (rather than an arbitrary tilt) keeps the
# self-test in the regime the design is actually evaluated in.
SIGMA_SELFTEST = 0.223
log_target = SIGMA_SELFTEST * (states.astype(np.float64) @ A4 * states).sum(axis=1)
target_probs = np.exp(log_target - logsumexp(log_target))
fit_pop = states[rng.choice(n_states, size=20_000, p=target_probs)]

for block_side, n_offsets in ((2, 1), (2, 4), (4, 16)):
    base = BlockOccupancyBase.fit(fit_pop, SIDE, block_side, n_offsets=n_offsets)
    total = logsumexp(base.log_density(states))
    check(f"logsumexp log_eta over all {n_states} slice states = 0 "
          f"[b={block_side}, K={n_offsets}]",
          abs(total) < 1e-9, f"logsumexp = {total:+.3e}")
    check(f"log_eta finite on the whole slice [b={block_side}, K={n_offsets}]",
          np.isfinite(base.log_density(states)).all(),
          f"min log_eta = {base.log_density(states).min():.3f}")

# --------------------------------------------------------------------------
# 2. sampler matches the density (chi-square) at D = 4
# --------------------------------------------------------------------------
state_key = {tuple(s.tolist()): i for i, s in enumerate(states)}
for block_side, n_offsets in ((2, 1), (2, 4)):
    base = BlockOccupancyBase.fit(fit_pop, SIDE, block_side, n_offsets=n_offsets)
    log_p = base.log_density(states)
    n_draws = 400_000
    draws = base.sample(n_draws, np.random.default_rng(12345))
    idx = np.fromiter((state_key[tuple(s.tolist())] for s in draws),
                      dtype=np.int64, count=n_draws)
    observed = np.bincount(idx, minlength=n_states).astype(float)
    expected = np.exp(log_p) * n_draws
    # Adaptive binning: a chi-square on raw cells is invalid when most of the
    # 12 870 slice states carry expected count << 1 (the statistic then has
    # nothing like a chi-square law and reads absurdly LOW, which would hide a
    # real sampler bug rather than expose one).  Merge cells, in density order,
    # until every bin has expected count >= MIN_EXPECTED.
    MIN_EXPECTED = 25.0
    order = np.argsort(expected)
    obs_bins, exp_bins, acc_o, acc_e = [], [], 0.0, 0.0
    for i in order:
        acc_o += observed[i]
        acc_e += expected[i]
        if acc_e >= MIN_EXPECTED:
            obs_bins.append(acc_o); exp_bins.append(acc_e); acc_o = acc_e = 0.0
    if acc_e > 0:
        obs_bins[-1] += acc_o; exp_bins[-1] += acc_e
    stat, pvalue = chisquare(obs_bins, exp_bins)
    check(f"sampler == density, chi-square [b={block_side}, K={n_offsets}]",
          pvalue > 0.001,
          f"chi2 = {stat:.1f} on {len(obs_bins) - 1} bins-1 dof, p = {pvalue:.3f}")

# --------------------------------------------------------------------------
# 3. Z2 and translation symmetry
# --------------------------------------------------------------------------
for block_side, n_offsets in ((2, 1), (2, 4), (4, 16)):
    base = BlockOccupancyBase.fit(fit_pop, SIDE, block_side, n_offsets=n_offsets)
    lp = base.log_density(states)
    lp_flipped = base.log_density(-states)
    check(f"Z2: log_eta(x) == log_eta(-x) [b={block_side}, K={n_offsets}]",
          np.allclose(lp, lp_flipped, atol=1e-12),
          f"max |diff| = {np.abs(lp - lp_flipped).max():.2e}")

    lp_shift = base.log_density(translate(states, SIDE, 1, 0))
    shifted_ok = np.allclose(lp, lp_shift, atol=1e-12)
    expect = (n_offsets == block_side * block_side)
    check(f"translation by 1 site {'invariant' if expect else 'NOT invariant'} "
          f"[b={block_side}, K={n_offsets}]",
          shifted_ok == expect,
          f"max |diff| = {np.abs(lp - lp_shift).max():.2e}")

# --------------------------------------------------------------------------
# 4. degenerate tiling b = D reduces to uniform on the slice
# --------------------------------------------------------------------------
base_full = BlockOccupancyBase.fit(fit_pop, SIDE, SIDE, n_offsets=1)
uniform = UniformSliceBase(SIDE, N_UP)
check("b = D collapses to uniform on the slice",
      np.allclose(base_full.log_density(states), uniform.log_density(states)),
      f"log_eta = {base_full.log_density(states)[0]:.6f} vs "
      f"-log C(16,8) = {-uniform.log_normaliser:.6f}")

# --------------------------------------------------------------------------
# 5. the uniform base's exact nn-correlation -1/(d-1)
# --------------------------------------------------------------------------
draws = uniform.sample(200_000, np.random.default_rng(7))
emp = nn_correlation(draws, A4).mean()
check("uniform-slice nn-correlation == -1/(d-1)",
      abs(emp - uniform.exact_nn_correlation()) < 4e-3,
      f"empirical {emp:+.5f} vs exact {uniform.exact_nn_correlation():+.5f}")

# --------------------------------------------------------------------------
# 6. DP normaliser against brute force at a size where brute force is possible
# --------------------------------------------------------------------------
base = BlockOccupancyBase.fit(fit_pop, SIDE, 2, n_offsets=4)
s, B, N = base.tile_sites, base.n_tiles, base.n_up
brute = logsumexp([
    sum(base.log_weights[m] for m in occ)
    for occ in itertools.product(range(s + 1), repeat=B) if sum(occ) == N
])
check("DP log Z_w == brute-force enumeration",
      abs(brute - base.log_normaliser) < 1e-10,
      f"DP {base.log_normaliser:.10f} vs brute {brute:.10f}")

# --------------------------------------------------------------------------
# 7. sampled composition is exactly on the slice at production sizes
# --------------------------------------------------------------------------
for side, n_up, block_side, n_offsets in ((8, 32, 2, 4), (16, 128, 2, 4), (16, 128, 4, 16)):
    d = side * side
    pop = UniformSliceBase(side, n_up).sample(2000, np.random.default_rng(1))
    b = BlockOccupancyBase.fit(pop, side, block_side, n_offsets=n_offsets)
    dr = b.sample(3000, np.random.default_rng(2))
    check(f"draws on the slice at D={side}, b={block_side}",
          bool(((dr > 0).sum(axis=1) == n_up).all()) and set(np.unique(dr)) == {-1, 1},
          f"n_up unique = {np.unique((dr > 0).sum(axis=1))}")

# --------------------------------------------------------------------------
# 8. Delta c = c_1 - c_0 = int_0^1 Var_{p_t}[D] dt, and the Monte-Carlo
#    estimator used by warm_base_offline_table.py, both against exact enumeration.
#    This is the check that decides which Delta c column is right: the full
#    (3.3) definition or an energy-only surrogate.
# --------------------------------------------------------------------------
base = BlockOccupancyBase.fit(fit_pop, SIDE, 2, n_offsets=4)
log_rho = SIGMA_SELFTEST * (states.astype(np.float64) @ A4 * states).sum(axis=1)
log_eta = base.log_density(states)
drive = log_rho - log_eta


def c_of_t(t: float) -> tuple[float, float]:
    """c_t = E_{p_t}[D] and Var_{p_t}[D] for p_t propto eta * exp(t D)."""
    logits = log_eta + t * drive
    p = np.exp(logits - logsumexp(logits))
    mean = float(p @ drive)
    return mean, float(p @ (drive - mean) ** 2)


c0, _ = c_of_t(0.0)
c1, _ = c_of_t(1.0)
grid = np.linspace(0.0, 1.0, 2001)
integral = np.trapezoid([c_of_t(t)[1] for t in grid], grid)
check("Delta c == int_0^1 Var_{p_t}[D] dt (exact enumeration)",
      abs((c1 - c0) - integral) < 1e-6 * max(1.0, abs(c1 - c0)),
      f"c1 - c0 = {c1 - c0:.6f}, integral = {integral:.6f}")

rng_mc = np.random.default_rng(99)
p1 = np.exp(log_rho - logsumexp(log_rho))
ref_draws = states[rng_mc.choice(n_states, size=400_000, p=p1)]
base_draws = base.sample(400_000, rng_mc)
mc_delta_c = (
    (SIGMA_SELFTEST * (ref_draws.astype(np.float64) @ A4 * ref_draws).sum(axis=1)
     - base.log_density(ref_draws)).mean()
    - (SIGMA_SELFTEST * (base_draws.astype(np.float64) @ A4 * base_draws).sum(axis=1)
       - base.log_density(base_draws)).mean()
)
check("Monte-Carlo Delta c estimator == exact c1 - c0",
      abs(mc_delta_c - (c1 - c0)) < 0.02,
      f"MC {mc_delta_c:.4f} vs exact {c1 - c0:.4f}")

energy_only = SIGMA_SELFTEST * (
    (ref_draws.astype(np.float64) @ A4 * ref_draws).sum(axis=1).mean()
    - (base_draws.astype(np.float64) @ A4 * base_draws).sum(axis=1).mean()
)
check("energy-only surrogate OVERSTATES Delta c for a warm base",
      energy_only > (c1 - c0) + 0.05,
      f"energy-only {energy_only:.4f} vs true {c1 - c0:.4f} "
      f"(base-entropy term {energy_only - mc_delta_c:.4f})")

# --------------------------------------------------------------------------
print()
for status, name, detail in results:
    print(f"[{status}] {name:<62s}  {detail}")
n_fail = sum(1 for s, _, _ in results if s == FAIL)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
raise SystemExit(1 if n_fail else 0)
