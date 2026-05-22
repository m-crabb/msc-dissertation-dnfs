"""Overlay training-time diagnostics for two runs to test the
two-phase (constraint-fit then Ising-fit) hypothesis.

Plots loss, training ESS, estimator-integrand variance, and grad-norm
on log-x axes so the early-phase knee is visible. If the two-phase
picture is right, expect a sharp knee at a few thousand steps where
the constraint fits, then a slow grind where the Ising-side correlations
do (or do not) get fit.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(run_dir):
    return pd.read_csv(Path(run_dir) / "training_log.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_l50", default="results/02_constrained_soft/S2_d10_c03_l50_letf_ne128_relaunch_seed42")
    p.add_argument("--run_l10", default="results/02_constrained_soft/S2_d10_c03_l10_letf_ne128_seed42_20260522-120742")
    p.add_argument("--out", default="results/vcsgc_mcmc/training_trajectories_l50_vs_l10.png")
    p.add_argument("--smooth", type=int, default=200, help="rolling-mean window for plotting")
    args = p.parse_args()

    df50 = load(args.run_l50)
    df10 = load(args.run_l10)

    def smooth(s):
        return s.rolling(args.smooth, min_periods=1, center=True).mean()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    ax = axes[0, 0]
    ax.plot(df50["step"], smooth(df50["loss"]), color="C0", lw=1.5, label="λ=50")
    ax.plot(df10["step"], smooth(df10["loss"]), color="C3", lw=1.5, label="λ=10")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("loss (smoothed)")
    ax.set_title("training loss")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    ax.plot(df50["step"], smooth(df50["ess"]), color="C0", lw=1.5, label="λ=50")
    ax.plot(df10["step"], smooth(df10["ess"]), color="C3", lw=1.5, label="λ=10")
    ax.set_xscale("log")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("training-time ESS (smoothed)")
    ax.set_title("training-time ESS")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    ax.plot(df50["step"], smooth(df50["var_estimator_integrand"]),
            color="C0", lw=1.5, label="λ=50")
    ax.plot(df10["step"], smooth(df10["var_estimator_integrand"]),
            color="C3", lw=1.5, label="λ=10")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("var(estimator integrand)")
    ax.set_title("estimator-integrand variance")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(df50["step"], smooth(df50["grad_norm"]), color="C0", lw=1.5, label="λ=50")
    ax.plot(df10["step"], smooth(df10["grad_norm"]), color="C3", lw=1.5, label="λ=10")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("‖grad‖ (smoothed)")
    ax.set_title("gradient norm")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"DNFS training trajectories: S2_d10_c03 (seed 42)  |  "
        f"smoothed with window={args.smooth} steps"
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # Knee detection: find step where loss has dropped within 10x of its final value.
    def find_knee(df, frac=0.1):
        loss = df["loss"].values
        final = np.median(loss[-200:])
        threshold = final * (1.0 / frac) if final > 0 else (final + abs(final) * (1 - frac))
        if final > 0:
            mask = loss < threshold
        else:
            mask = loss < final * frac
        if mask.any():
            return int(df["step"].values[np.argmax(mask)])
        return None

    knee50 = find_knee(df50)
    knee10 = find_knee(df10)

    print(f"λ=50 knee (loss within 10× of final): step {knee50}")
    print(f"λ=10 knee (loss within 10× of final): step {knee10}")
    print(f"λ=50 final loss : {df50['loss'].iloc[-200:].median():+.4f}")
    print(f"λ=10 final loss : {df10['loss'].iloc[-200:].median():+.4f}")
    print(f"λ=50 final ess  : {df50['ess'].iloc[-200:].median():.1f}")
    print(f"λ=10 final ess  : {df10['ess'].iloc[-200:].median():.1f}")
    print(f"λ=50 final var  : {df50['var_estimator_integrand'].iloc[-200:].median():.3e}")
    print(f"λ=10 final var  : {df10['var_estimator_integrand'].iloc[-200:].median():.3e}")

    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
