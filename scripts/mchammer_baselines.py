"""Run the literal mchammer MCMC baselines at the thesis operating points.

These are the *reported* DNFS-vs-MCMC sampling baselines: mchammer's own
VCSGCEnsemble (soft leg) and CanonicalEnsemble (hard leg) on the validated
cluster-expansion embedding, timed per effective sample. The numba scripts
(`vcsgc_mcmc_validation.py`, `kawasaki_sweep.py`) stay as independent
cross-checks only. Wall-clock per effective sample is the comparison
currency against DNFS eval (`eval_draw_seconds`, `nfe_per_effective_sample`)
because an MCMC sweep has no NFE analogue.

Layout mirrors the bespoke runs, one dir per (composition, seed):

    results/mchammer_vcsgc/D10_s0.1_l50.0_c0.30_seed42/
        summary.json  composition.npy  potential.npy  [spins.npy]

Examples:
    # soft leg, matched to the S=2 D=10 lambda=50 campaign
    python scripts/mchammer_baselines.py vcsgc --D 10 --sigma 0.1 --lam 50 \
        --compositions 0.30 0.50 0.55 0.60 0.65 0.80

    # hard leg, matched to the d64 (D=8) sigma_c cells
    python scripts/mchammer_baselines.py canonical --D 8 --sigma 0.223 \
        --compositions 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np

from discrete_flow_sampler.mcmc.mchammer_ising import (
    run_canonical, run_sgc, run_vcsgc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ensemble", choices=["vcsgc", "canonical", "sgc"])
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument(
        "--lam", type=float, default=50.0,
        help="soft composition penalty strength (vcsgc only; kappa = lam)",
    )
    parser.add_argument(
        "--compositions", type=float, nargs="+", required=True,
        help="target composition (vcsgc/canonical); for sgc the chain's "
             "STARTING composition only -- at Delta-mu = 0 it then floats",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--n-steps", type=int, default=1_000_000)
    parser.add_argument("--write-interval", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--record-spins", action="store_true",
        help="vcsgc/sgc only: also save spins.npy (post-burn-in "
             "configurations, int8) so profile observables can be scored "
             "against the chain",
    )
    args = parser.parse_args()

    out_root = args.out or Path("results") / f"mchammer_{args.ensemble}"
    for composition in args.compositions:
        for seed in args.seeds:
            if args.ensemble == "vcsgc":
                summary = run_vcsgc(
                    D=args.D, sigma=args.sigma, penalty_strength=args.lam,
                    target_composition=composition, n_steps=args.n_steps,
                    seed=seed, bias=args.bias,
                    data_write_interval=args.write_interval,
                    record_spins=args.record_spins,
                )
                # Composition at 3 decimals: the 8x8 house windows include
                # 0.375, which the old 2-decimal name would round to c0.38
                # — a lossy dir name the analysis globs would then have to
                # guess back. Sigma via %g so a sigma_c chain does not get
                # a 17-digit dir name. Archived D10 dirs (2-decimal) are
                # never regenerated, so no collision.
                run_name = (
                    f"D{args.D}_s{args.sigma:g}_l{args.lam}"
                    f"_c{composition:.3f}_seed{seed}"
                )
            elif args.ensemble == "sgc":
                summary = run_sgc(
                    D=args.D, sigma=args.sigma,
                    initial_composition=composition, n_steps=args.n_steps,
                    seed=seed, bias=args.bias,
                    data_write_interval=args.write_interval,
                    record_spins=args.record_spins,
                )
                # c is a start, not a constraint -- tagged c0 so an SGC
                # dir is never mistaken for a composition-pinned one.
                run_name = (
                    f"D{args.D}_s{args.sigma}_c0{composition:.2f}_seed{seed}"
                )
            else:
                summary = run_canonical(
                    D=args.D, sigma=args.sigma,
                    target_composition=composition, n_steps=args.n_steps,
                    seed=seed, bias=args.bias,
                    data_write_interval=args.write_interval,
                )
                run_name = (
                    f"D{args.D}_s{args.sigma}_c{composition:.2f}_seed{seed}"
                )

            run_dir = out_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            traces = summary.pop("traces")
            for name, trace in traces.items():
                np.save(run_dir / f"{name}.npy", trace)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

            potential = summary["observables"]["potential"]
            print(
                f"{run_name}: {summary['steps_per_second']:.0f} steps/s, "
                f"potential ESS {potential['ess']:.0f} "
                f"({potential['seconds_per_effective_sample']:.2e} s/eff), "
                + (
                    f"<c> {summary['observables']['composition']['mean']:.4f}"
                    f" +- {summary['observables']['composition']['std']:.4f}"
                    if args.ensemble in ("vcsgc", "sgc")
                    else f"c fixed at {summary['composition_realised']:.4f}"
                )
            )


if __name__ == "__main__":
    main()
