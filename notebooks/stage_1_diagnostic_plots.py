"""Generate Stage 1 diagnostic figures from finished run directories.

Reads `training_log.csv` and `eval/metrics.json` from each run dir under
`results/02_baseline/`, then emits:

- Per-run 4-panel summary (loss / ESS-fraction / var_dt_log_p_tilde /
  wall_clock_step) saved into `<run_dir>/diagnostic_summary.png`.
- Cross-run comparison overlay (loss + var_dt_log_p_tilde + ESS-fraction
  on a shared x-axis) saved into `results/02_baseline/stage_1_comparison.png`.

Usage:
    pixi run -e dev python notebooks/stage_1_diagnostic_plots.py

The script is intentionally idempotent and discovery-driven: it picks up
whichever runs are on disk, so re-running it after a fresh Modal pull
just refreshes the figures.

Stage 1's narrative claim: the naive ∂_t log Z_t estimator is too
high-variance to train the flow. The figures here pin that claim to two
empirical observations: (a) `var_dt_log_p_tilde` does not decrease over
training, and (b) ESS climbs anyway -- the dishonest signal that paper
§3.1 motivates Stage 2's control variate to repair.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results/02_baseline")


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
    if metrics is not None and "tvd" in metrics:
        title += (
            f"\nfinal TVD = {metrics['tvd']:.3f}  | ESS-frac = "
            f"{metrics['ess_fraction']:.3f}  | reverse-KL = "
            f"{metrics.get('kl_reverse'):.3f}  | logp-W1 = "
            f"{metrics.get('log_prob_w1'):.3f}"
        )
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
    """Tabular summary for cut-and-paste into the dissertation prose."""
    print()
    print("=" * 88)
    print("STAGE 1 SUMMARY")
    print("=" * 88)
    rows = []
    for run in runs:
        log = run["log"]
        eval_rows = run["eval_rows"]
        metrics = run["metrics"] or {}
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
                "tvd": (
                    f"{metrics.get('tvd'):.3f}"
                    if metrics.get("tvd") is not None else "—"
                ),
                "kl_reverse": (
                    f"{metrics.get('kl_reverse'):.3f}"
                    if metrics.get("kl_reverse") is not None else "—"
                ),
                "logp_w1": (
                    f"{metrics.get('log_prob_w1'):.3f}"
                    if metrics.get("log_prob_w1") is not None else "—"
                ),
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()


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
    _print_summary_table(runs)


if __name__ == "__main__":
    main()
