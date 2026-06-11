"""Soft-constraint figure: the penalty's variance injection, seed by seed.

The experiments chapter claims the lambda^2 Var[delta_P] term (appendix
derivation) is what makes soft-constrained training unstable at D=10: the
estimator-integrand variance starts ~100x the unconstrained level at matched
settings, and only the seed in which it eventually collapses produces a usable
sampler. This figure shows exactly that, from the training logs already on
disk: var_estimator_integrand over training for the four soft
S2_d10_c05_l50_letf_ne64 seeds vs the four matched unconstrained
stage_4_d10_budget seeds (identical architecture/budget/settings, no penalty).

The full Var[delta_I] / lambda^2 Var[delta_P] decomposition is NOT logged, so
this is the two-trace version: total integrand variance, soft vs unconstrained.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOG_CADENCE_STEPS = 500  # var_estimator_integrand only refreshes at eval cadence


def variance_trace(run_dir: Path) -> pd.DataFrame:
    log = pd.read_csv(run_dir / "training_log.csv",
                      usecols=["step", "var_estimator_integrand"])
    return log[log.step % LOG_CADENCE_STEPS == 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--soft_runs", required=True, nargs="+", type=Path,
                        help="S2_d10_c05_l50_letf_ne64 run dirs (seeds 42-45)")
    parser.add_argument("--baseline_runs", required=True, nargs="+", type=Path,
                        help="matched stage_4_d10_budget run dirs (seeds 42-45)")
    parser.add_argument("--healthy_soft_seed", default="seed44",
                        help="substring naming the soft seed that trained")
    parser.add_argument("--out", type=Path, default=Path("penalty_variance_traces.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    for run_dir in args.baseline_runs:
        trace = variance_trace(run_dir)
        ax.plot(trace.step, trace.var_estimator_integrand, color="C2", lw=1.0,
                alpha=0.8)
    for run_dir in args.soft_runs:
        trace = variance_trace(run_dir)
        healthy = args.healthy_soft_seed in run_dir.name
        ax.plot(trace.step, trace.var_estimator_integrand,
                color="C1" if not healthy else "C3", lw=1.2)
        if healthy:
            final = trace.iloc[-1]
            ax.annotate("the one soft seed that trains",
                        xy=(final.step, final.var_estimator_integrand),
                        xytext=(0.38, 0.20), textcoords="axes fraction",
                        fontsize=9, color="C3",
                        arrowprops={"arrowstyle": "->", "color": "C3", "lw": 1.0})

    # Proxy artists so the legend has one entry per group, not per seed.
    ax.plot([], [], color="C1", lw=1.2, label=r"soft, $\lambda=50$ (stuck seeds)")
    ax.plot([], [], color="C3", lw=1.2, label=r"soft, $\lambda=50$ (healthy seed)")
    ax.plot([], [], color="C2", lw=1.0, label="unconstrained, matched config")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("estimator-integrand variance")
    ax.set_title(r"$10\times10$, $\sigma=0.1$: the penalty's variance injection,"
                 " per seed", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved figure to {args.out}")


if __name__ == "__main__":
    main()
