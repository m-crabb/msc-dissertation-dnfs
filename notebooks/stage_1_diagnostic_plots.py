"""Generate Stage 1 diagnostic figures from finished run directories.

Reads `training_log.csv` and `eval/metrics.json` from each run dir under
`results/02_baseline/`, then emits:

- Per-run 4-panel summary (loss / ESS-fraction / var_dt_log_p_tilde /
  wall_clock_step) saved into `<run_dir>/diagnostic_summary.png`.
- Cross-run comparison overlay (loss + var_dt_log_p_tilde + ESS-fraction
  on a shared x-axis) saved into `results/02_baseline/stage_1_comparison.png`.
- A summary table printed to stdout reporting the paper-faithful eval
  metrics (paper Appendix D.1, Table 2): ESS-fraction, F/D, E/D, S/D,
  and (where available) bias-vs-exact at D ≤ 20.
- Energy-histogram-vs-Gibbs verdict figure for D=10 runs (paper Figure 5
  right panel structural analog), saved to
  `results/02_baseline/stage_1_d10_energy_vs_gibbs.png` when the Gibbs
  reference at `gibbs_chain_d10_sigma01.pt` is on disk.
- Energy-histogram-vs-exact verdict figure for D=4 runs (analytical
  ground truth via 2^16-state enumeration; sharper than the d10 Gibbs
  reference because the reference has no sampling noise), saved to
  `results/02_baseline/stage_1_d4_energy_vs_exact.png`.

Usage:
    pixi run -e dev python notebooks/stage_1_diagnostic_plots.py

The script is idempotent and discovery-driven: it picks up whichever
runs are on disk, so re-running it after a fresh Modal pull just
refreshes the figures.

Stage 1's narrative claim: the naive ∂_t log Z_t estimator is too
high-variance to train the flow. The figures here pin that claim to
two empirical observations: (a) `var_dt_log_p_tilde` does not decrease
over training, and (b) ESS climbs anyway -- the dishonest signal that
paper §3 motivates Stage 2's control variate (Eq. 8) to repair.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    exact_log_probs,
)
from discrete_flow_sampler.targets.ising import IsingTarget

RESULTS_DIR = Path("results/02_baseline")
GIBBS_REF_PATH = RESULTS_DIR / "gibbs_chain_d10_sigma01.pt"


def _load_run(run_dir: Path) -> dict:
    """Pull training-log + eval-metrics into a single dict per run.

    Returns None if essential artefacts are missing (run is mid-flight or
    incomplete) so the caller can skip it without crashing the figure
    generation pipeline.
    """
    csv_path = run_dir / "training_log.csv"
    if not csv_path.exists():
        return None
    config = json.loads((run_dir / "config.json").read_text())
    log = pd.read_csv(csv_path)
    metrics_path = run_dir / "eval" / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    )
    n_eval_samples = config["eval"]["n_eval_samples"]
    eval_rows = log.dropna(subset=["ess"]).copy()
    eval_rows["ess_fraction"] = eval_rows["ess"] / n_eval_samples
    return {
        "name": run_dir.name,
        "label": config["name"],
        "run_dir": run_dir,
        "config": config,
        "log": log,
        "eval_rows": eval_rows,
        "metrics": metrics,
    }


def _plot_per_run(run: dict) -> None:
    """Four-panel diagnostic figure saved next to the run's artefacts."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    log = run["log"]
    eval_rows = run["eval_rows"]

    axes[0, 0].plot(log.step, log.loss, lw=0.5, alpha=0.4, label="per-step")
    rolling_window = max(1, len(log) // 200)
    axes[0, 0].plot(
        log.step,
        log.loss.rolling(rolling_window, min_periods=1).mean(),
        lw=1.5,
        color="C3",
        label=f"rolling mean (w={rolling_window})",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("loss (log)")
    axes[0, 0].set_title("Kolmogorov residual loss")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        eval_rows.step, eval_rows.ess_fraction, marker="o", lw=1, color="C2"
    )
    axes[0, 1].axhline(1.0, ls="--", color="grey", lw=0.5)
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("ESS / N_eval")
    axes[0, 1].set_title("ESS fraction at eval (self-consistency, NOT coverage)")
    axes[0, 1].set_ylim(0, 1.05)

    axes[1, 0].plot(
        log.step, log.var_dt_log_p_tilde, lw=0.5, alpha=0.5, color="C4"
    )
    axes[1, 0].plot(
        log.step,
        log.var_dt_log_p_tilde.rolling(rolling_window, min_periods=1).mean(),
        lw=1.5,
        color="C3",
    )
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel(r"$\mathrm{Var}\,\partial_t \log \tilde{p}$")
    axes[1, 0].set_title("Estimator-integrand variance (Stage 2 target)")

    axes[1, 1].plot(log.step, log.wall_clock_step_s * 1000, lw=0.5, color="C5")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("wall-clock per step (ms)")
    axes[1, 1].set_title("Wall-clock per training step")

    metrics = run["metrics"]
    title = f"{run['label']}  ({run['name']})"
    if metrics is not None and "ess_fraction" in metrics:
        parts = [f"ESS-frac = {metrics['ess_fraction']:.3f}"]
        if "free_energy_per_site" in metrics:
            parts.append(f"F/D = {metrics['free_energy_per_site']:.4f}")
        if "internal_energy_per_site" in metrics:
            parts.append(f"E/D = {metrics['internal_energy_per_site']:.4f}")
        if "entropy_per_site" in metrics:
            parts.append(f"S/D = {metrics['entropy_per_site']:.4f}")
        if "free_energy_per_site_bias" in metrics:
            parts.append(
                f"bias F/D = {metrics['free_energy_per_site_bias']:+.4f}"
            )
        title += "\n" + "  | ".join(parts)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = run["run_dir"] / "diagnostic_summary.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_comparison(runs: list[dict]) -> None:
    """Cross-run overlay: loss / ESS-fraction / var_dt_log_p_tilde."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for run in runs:
        log = run["log"]
        eval_rows = run["eval_rows"]
        rolling_window = max(1, len(log) // 200)
        axes[0].plot(
            log.step,
            log.loss.rolling(rolling_window, min_periods=1).mean(),
            label=run["label"],
            lw=1.5,
        )
        axes[1].plot(
            eval_rows.step,
            eval_rows.ess_fraction,
            marker="o",
            ms=3,
            label=run["label"],
            lw=1.0,
        )
        axes[2].plot(
            log.step,
            log.var_dt_log_p_tilde.rolling(rolling_window, min_periods=1).mean(),
            label=run["label"],
            lw=1.5,
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (log, rolling)")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=8)

    axes[1].axhline(1.0, ls="--", color="grey", lw=0.5)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("ESS / N_eval")
    axes[1].set_title("ESS fraction")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("step")
    axes[2].set_ylabel(r"$\mathrm{Var}\,\partial_t \log \tilde{p}$")
    axes[2].set_title("Integrand variance (rolling)")
    axes[2].legend(fontsize=8)

    fig.suptitle("Stage 1 cross-run comparison", fontsize=12)
    fig.tight_layout()
    out_path = RESULTS_DIR / "stage_1_comparison.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _print_summary_table(runs: list[dict]) -> None:
    """Tabular summary for cut-and-paste into the dissertation prose.

    Columns mirror the paper-faithful suite (Appendix D.1, Table 2):
    ESS-fraction (Eq. 42), F/D (Eq. 37), E/D (Eq. 38), S/D = 2σ(E - F)/D,
    and where available the signed bias against the exact enumeration
    reference (D ≤ 20). Training-side columns (loss, var_dt_log_p_tilde)
    are kept as mechanism diagnostics; they're not paper-headline but
    are the substance of Stage 2's claim that the integrand variance is
    the optimisation bottleneck.
    """
    print()
    print("=" * 110)
    print("STAGE 1 SUMMARY")
    print("=" * 110)
    rows = []
    for run in runs:
        log = run["log"]
        eval_rows = run["eval_rows"]
        metrics = run["metrics"] or {}

        def _fmt(key: str, fmt: str = "{:.4f}") -> str:
            value = metrics.get(key)
            return fmt.format(value) if value is not None else "—"

        rows.append(
            {
                "run": run["label"],
                "steps_run": len(log),
                "loss_tail500": f"{log.loss.tail(500).mean():.3f}",
                "var_dt_tail500": f"{log.var_dt_log_p_tilde.tail(500).mean():.3f}",
                "ess_frac_final": (
                    f"{eval_rows.ess_fraction.iloc[-1]:.3f}"
                    if len(eval_rows) else "—"
                ),
                "F/D": _fmt("free_energy_per_site"),
                "E/D": _fmt("internal_energy_per_site"),
                "S/D": _fmt("entropy_per_site"),
                "F/D_bias": _fmt("free_energy_per_site_bias", "{:+.4f}"),
                "E/D_bias": _fmt("internal_energy_per_site_bias", "{:+.4f}"),
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()


def _plot_energy_vs_gibbs(d10_runs: list[dict]) -> None:
    """Energy-histogram verdict figure for D=10 runs vs Gibbs Oracle.

    Structural analog of the paper's Figure 5 right panel: overlays the
    target-evaluated energy H(x) = -log p̃(x) of DNFS samples against a
    long-run Gibbs reference. Skip cleanly if the reference is missing.

    Mirrors `_plot_energy_vs_exact`'s two-panel layout: left = unweighted
    CTMC samples (proposal-quality probe), right = IS-weighted samples
    (paper Eq. 13, target-fit headline). All curves as probability mass
    per bin so cross-run N differences are silently normalised. Note: the
    Gibbs reference itself has small sampling noise (long-chain MCMC, not
    enumeration), so the "Oracle" curve is *near*-exact rather than exact.
    """
    if not GIBBS_REF_PATH.exists():
        print(f"  skipping energy-vs-Gibbs: no reference at {GIBBS_REF_PATH}")
        return
    if not d10_runs:
        return

    gibbs_payload = torch.load(GIBBS_REF_PATH, weights_only=True)
    gibbs_samples = gibbs_payload["samples"]
    cfg = d10_runs[0]["config"]["ising"]
    target = IsingTarget(D=cfg["D"], sigma=cfg["sigma"], bias=cfg["bias"])
    gibbs_energy = (-target.log_prob(gibbs_samples)).numpy()

    dnfs_curves: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for run in d10_runs:
        samples_path = run["run_dir"] / "eval" / "samples.pt"
        log_w_path = run["run_dir"] / "eval" / "log_weights.pt"
        if not samples_path.exists() or not log_w_path.exists():
            continue
        samples = torch.load(samples_path, weights_only=True)
        log_w = torch.load(log_w_path, weights_only=True).numpy()
        norm_weights = np.exp(log_w - log_w.max())
        norm_weights = norm_weights / norm_weights.sum()
        energies = (-target.log_prob(samples)).numpy()
        dnfs_curves.append((run, energies, norm_weights))
    if not dnfs_curves:
        return

    energy_min = min(
        [gibbs_energy.min()] + [e.min() for _, e, _ in dnfs_curves]
    )
    energy_max = max(
        [gibbs_energy.max()] + [e.max() for _, e, _ in dnfs_curves]
    )
    bins = np.linspace(energy_min, energy_max, 50)

    gibbs_counts, _ = np.histogram(gibbs_energy, bins=bins)
    gibbs_mass = gibbs_counts / len(gibbs_energy)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, mode in zip(axes, ("unweighted", "is_weighted")):
        ax.step(
            bins[:-1], gibbs_mass, where="post", lw=2.0, color="black",
            label="Oracle (Gibbs, long-run)",
        )
        for run, energy, weights in dnfs_curves:
            if mode == "unweighted":
                counts, _ = np.histogram(energy, bins=bins)
                mass = counts / len(energy)
            else:
                mass, _ = np.histogram(energy, bins=bins, weights=weights)
            ax.step(
                bins[:-1], mass, where="post", lw=1.5,
                label=run["label"], alpha=0.85,
            )
        ax.set_xlabel(r"Energy $= -\log\tilde{p}(x)$")
        ax.set_title(
            "Unweighted CTMC samples (proposal quality)" if mode == "unweighted"
            else "IS-weighted samples (paper Eq. 13, target fit)"
        )
        ax.legend(fontsize=8)
    axes[0].set_ylabel("probability mass per bin")
    fig.suptitle(
        f"D={cfg['D']}×{cfg['D']}, σ={cfg['sigma']}: "
        "energy distribution vs Gibbs Oracle (paper Fig. 5 right)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = RESULTS_DIR / "stage_1_d10_energy_vs_gibbs.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_energy_vs_exact(d4_runs: list[dict]) -> None:
    """Energy-histogram verdict figure for D=4 runs vs analytical exact density.

    At D=4 (16 sites, 2^16 = 65k states) the target is fully enumerable so the
    reference is *analytical*: each enumerated state contributes its probability
    mass to its energy bin. There is no sampling noise on the reference side,
    so any DNFS-vs-exact deviation is real bias, not statistical fluctuation.

    Two left/right panels:
      • Left  — unweighted CTMC samples: a *proposal-quality* check. Tells us
                how close the learned R_t is to being a self-sufficient sampler.
      • Right — IS-weighted samples (paper Eq. 13): the headline target-fit
                check. Reweighting absorbs proposal bias as long as the
                proposal has support over the target's high-mass region.

    All curves plotted as probability mass per bin (sum to 1) so cross-run N
    differences are silently normalised.
    """
    if not d4_runs:
        return

    cfg = d4_runs[0]["config"]["ising"]
    target = IsingTarget(D=cfg["D"], sigma=cfg["sigma"], bias=cfg["bias"])
    states = enumerate_states(D=target.d).float()
    log_p_exact = exact_log_probs(target, states)
    p_exact = log_p_exact.exp().numpy()
    energies_exact = (-target.log_prob(states)).numpy()

    dnfs_curves: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for run in d4_runs:
        samples_path = run["run_dir"] / "eval" / "samples.pt"
        log_w_path = run["run_dir"] / "eval" / "log_weights.pt"
        if not samples_path.exists() or not log_w_path.exists():
            continue
        samples = torch.load(samples_path, weights_only=True)
        log_w = torch.load(log_w_path, weights_only=True).numpy()
        # Self-normalised weights: subtract max for numerical stability.
        norm_weights = np.exp(log_w - log_w.max())
        norm_weights = norm_weights / norm_weights.sum()
        energies = (-target.log_prob(samples)).numpy()
        dnfs_curves.append((run, energies, norm_weights))
    if not dnfs_curves:
        return

    energy_min = min(
        [energies_exact.min()] + [e.min() for _, e, _ in dnfs_curves]
    )
    energy_max = max(
        [energies_exact.max()] + [e.max() for _, e, _ in dnfs_curves]
    )
    bins = np.linspace(energy_min, energy_max, 50)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    p_per_bin_exact, _ = np.histogram(energies_exact, bins=bins, weights=p_exact)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, mode in zip(axes, ("unweighted", "is_weighted")):
        ax.plot(
            bin_centres, p_per_bin_exact, lw=2.0, color="black",
            label="Exact (analytical)",
        )
        for run, energy, weights in dnfs_curves:
            if mode == "unweighted":
                counts, _ = np.histogram(energy, bins=bins)
                mass = counts / len(energy)
            else:
                mass, _ = np.histogram(energy, bins=bins, weights=weights)
            ax.step(
                bins[:-1], mass, where="post", lw=1.5,
                label=run["label"], alpha=0.85,
            )
        ax.set_xlabel(r"Energy $= -\log\tilde{p}(x)$")
        ax.set_title(
            "Unweighted CTMC samples (proposal quality)" if mode == "unweighted"
            else "IS-weighted samples (paper Eq. 13, target fit)"
        )
        ax.legend(fontsize=8)
    axes[0].set_ylabel("probability mass per bin")
    fig.suptitle(
        f"D={cfg['D']}×{cfg['D']}, σ={cfg['sigma']}: "
        "energy distribution vs exact enumeration",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = RESULTS_DIR / "stage_1_d4_energy_vs_exact.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    run_dirs = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    runs = [r for r in (_load_run(d) for d in run_dirs) if r is not None]
    if not runs:
        print(f"No runs found under {RESULTS_DIR}")
        return
    print(f"Found {len(runs)} run(s):")
    for run in runs:
        print(f"  - {run['name']}")
        _plot_per_run(run)
    _plot_comparison(runs)
    d4_runs = [r for r in runs if r["config"]["ising"]["D"] == 4]
    _plot_energy_vs_exact(d4_runs)
    d10_runs = [r for r in runs if r["config"]["ising"]["D"] == 10]
    _plot_energy_vs_gibbs(d10_runs)
    _print_summary_table(runs)


if __name__ == "__main__":
    main()
