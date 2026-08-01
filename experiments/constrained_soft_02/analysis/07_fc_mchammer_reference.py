"""Native mchammer free-energy reference F(c) for the soft-constraint campaign.

This is the ground-truth canonical free-energy curve the DNFS soft F(c) gets
measured against. It uses *literal* mchammer (icet) machinery rather than a
hand-rolled integrator, per the campaign's "what practitioners actually run"
framing:

  - `ThermodynamicIntegrationEnsemble` runs the canonical (fixed-composition,
    swap-move) ensemble with the coupled Hamiltonian H(lambda) = lambda * H_CE,
    sampling lambda from 0 (fully disordered reference, free energy = ideal
    mixing entropy, known in closed form) to 1 (our cluster expansion).
  - `free_energy_tools.get_free_energy_thermodynamic_integration` returns the
    *absolute* canonical free energy A(c) at each composition: no offset/anchor
    fudge, and forward vs backward integration brackets the hysteresis error.

Embedding (validated 2026-06-13; lives in
`discrete_flow_sampler.mcmc.mchammer_ising`, pinned by
`tests/test_mchammer_ising.py`): the 2D Ising
torus is a single-layer vacuum-padded cell; the target
log p(x) = 2*sigma*sum_<ij> x_i x_j + bias*sum_i x_i maps to a binary CE with
ECI = [0, -bias, -4*sigma]. We work in natural units (kT = 1: temperature = 1,
boltzmann_constant = 1) so exp(-E/kT) = exp(-E) = p, matching the DNFS target.

The free energy returned by mchammer is total (not per site); we report F/site to
match the DNFS `free_energy_per_site` convention. NOTE this is the *canonical*
free energy; reconciling it with the DNFS soft IS-logZ (Laplace / kappa-match) is
a separate step. Here we build and validate the reference itself.

Validation (`--validate`, D<=4): the canonical sector enumerates (2^(D*D) states),
so F_exact(c) = -log sum_{c(x)=c} p(x) is available and the TI curve is checked
against it bit-for-comparable.

Example:
    python -m experiments.constrained_soft_02.analysis.07_fc_mchammer_reference \
        --D 4 --sigma 0.1 --compositions 0.5 0.625 --seeds 0 1 --validate

    python -m experiments.constrained_soft_02.analysis.07_fc_mchammer_reference \
        --D 10 --sigma 0.1 \
        --compositions 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
        --seeds 0 1 2 --n_steps 200000 --out results/02_constrained_soft/fc_ref_d10.npz
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import ThermodynamicIntegrationEnsemble
from mchammer.free_energy_tools import get_free_energy_thermodynamic_integration

from discrete_flow_sampler.mcmc.mchammer_ising import (
    ising_cluster_expansion,
    ising_supercell as _supercell,
)

# natural units: kT = 1 so exp(-E/kT) = exp(-E) = p(x)
_KB = 1.0
_T0 = 1.0


def exact_canonical_F(cs, ce, sc0, n_up: int) -> float:
    """-log Z_can over the fixed-composition sector by enumeration (small D)."""
    from scipy.special import logsumexp
    base = sc0.copy()
    N = len(base)
    log_terms = []
    for up in itertools.combinations(range(N), n_up):
        syms = ["Ag"] * N
        for i in up:
            syms[i] = "Au"
        base.set_chemical_symbols(syms)
        # ce.predict is per-atom; total energy = per-atom * N; p = exp(-E)
        log_terms.append(-ce.predict(base) * N)
    return float(-logsumexp(log_terms))


def ti_canonical_F(cs, ce, sc0, n_up: int, n_steps: int, seed: int):
    """Absolute canonical free energy (total) at fixed composition via native TI.

    Runs forward and backward; returns (mean over the two directions, hysteresis
    gap). The free energy at lambda=1 is the entry whose temperature equals T0.
    """
    sc = sc0.copy()
    N = len(sc)
    syms = ["Au"] * n_up + ["Ag"] * (N - n_up)
    np.random.default_rng(seed).shuffle(syms)
    sc.set_chemical_symbols(syms)
    calc = ClusterExpansionCalculator(sc, ce)
    ends = []
    for fwd in (True, False):
        ens = ThermodynamicIntegrationEnsemble(
            structure=sc, calculator=calc, temperature=_T0, n_steps=n_steps,
            forward=fwd, boltzmann_constant=_KB, ensemble_data_write_interval=10,
            random_seed=seed,
        )
        ens.run()
        temps, F = get_free_energy_thermodynamic_integration(
            ens.data_container, cs, forward=fwd, boltzmann_constant=_KB)
        temps = np.asarray(temps, float)
        F = np.asarray(F, float)
        good = np.isfinite(temps) & np.isfinite(F)
        j = np.argmin(np.abs(temps[good] - _T0))
        ends.append(F[good][j])
    return float(np.mean(ends)), float(abs(ends[0] - ends[1]))


def fc_reference(D, sigma, compositions, seeds, n_steps, validate=False):
    prim, cs, ce = ising_cluster_expansion(sigma)
    sc0 = _supercell(prim, D)
    N = len(sc0)
    rows = []
    for c in compositions:
        n_up = int(round(c * N))
        c_eff = n_up / N  # composition actually realised on this lattice
        per_seed, hysts = [], []
        for s in seeds:
            F, hy = ti_canonical_F(cs, ce, sc0, n_up, n_steps, s)
            per_seed.append(F)
            hysts.append(hy)
        F_mean = float(np.mean(per_seed))
        F_sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
        row = dict(c_target=c, c_eff=c_eff, n_up=n_up, N=N,
                   F_total=F_mean, F_per_site=F_mean / N,
                   F_sd_total=F_sd, F_sd_per_site=F_sd / N,
                   hysteresis=float(np.mean(hysts)))
        if validate and D * D <= 18:
            Fe = exact_canonical_F(cs, ce, sc0, n_up)
            row["F_exact_total"] = Fe
            row["F_exact_per_site"] = Fe / N
            row["diff_per_site"] = (F_mean - Fe) / N
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--D", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--compositions", nargs="+", type=float, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n_steps", type=int, default=200_000,
                   help="number of lambda points in the TI sweep")
    p.add_argument("--validate", action="store_true",
                   help="(D<=4) compare the TI curve against exact enumeration")
    p.add_argument("--out", type=Path, default=None, help="npz output path")
    args = p.parse_args()

    rows = fc_reference(args.D, args.sigma, args.compositions, args.seeds,
                        args.n_steps, validate=args.validate)

    has_exact = any("F_exact_per_site" in r for r in rows)
    print(f"=== mchammer canonical F(c) reference : D={args.D} sigma={args.sigma} "
          f"seeds={args.seeds} n_steps={args.n_steps} ===")
    header = (f"{'c_t':>6} {'c_eff':>6} {'F/site (mean +/- sd)':>22} {'hyst/site':>10}")
    if has_exact:
        header += f" {'F/site exact':>13} {'diff/site':>10}"
    print(header)
    for r in rows:
        line = (f"{r['c_target']:>6.3f} {r['c_eff']:>6.3f} "
                f"{r['F_per_site']:>11.5f} +/- {r['F_sd_per_site']:<6.5f} "
                f"{r['hysteresis'] / r['N']:>10.5f}")
        if has_exact and "F_exact_per_site" in r:
            line += f" {r['F_exact_per_site']:>13.5f} {r['diff_per_site']:>+10.5f}"
        print(line)

    # save first, so a cosmetic print issue can never lose the swept data
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        cols = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
        np.savez(args.out, **cols)
        print(f"\nwrote {args.out}")

    # Z2 check F(c) = F(1-c) where both present (round keys consistently)
    Fmap = {round(r["c_eff"], 4): r["F_per_site"] for r in rows}
    pairs = sorted({(round(min(c, 1 - c), 4), round(max(c, 1 - c), 4))
                    for c in Fmap
                    if round(1 - c, 4) in Fmap and abs(c - 0.5) > 1e-6})
    if pairs:
        print("\n--- Z2 check  F(c) vs F(1-c) (no field => should match) ---")
        for lo, hi in pairs:
            print(f"  F({lo:.3f})={Fmap[lo]:+.5f}  F({hi:.3f})={Fmap[hi]:+.5f}  "
                  f"|gap|={abs(Fmap[lo] - Fmap[hi]):.5f}")


if __name__ == "__main__":
    main()
