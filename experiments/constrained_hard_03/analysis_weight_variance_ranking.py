"""Model selection by the batch variance of log importance weights.

Why this statistic exists in the project
----------------------------------------
The archive has been ranked by frozen-eval ESS/N. At sigma_c on the 16x16
lattice the log-weight variance sits at Var[log w] >= 17, where the
self-normalised effective sample count collapses to 1-47 out of 5000 draws.
An ESS estimate whose value is set by the one or two largest weights carries
essentially no ranking information: reorderings between arms are within
noise, so those comparisons were statistically empty.

Var[log w] is the natural replacement in exactly this regime. It is a plain
second central moment over ALL N draws — every draw contributes equally, so
its own sampling error stays controlled even when the normalised weights
have degenerated onto a handful of draws. Lower Var[log w] means the model's
sampling path measure is closer (in the log-density sense) to the reference
path measure, which is the quantity training actually shrinks. Selecting
models by batch log-weight variance rather than by degenerate ESS follows
the practice of the masked diffusion neural sampler literature, where the
same collapse arises.

What is computed per (run, eval subdirectory)
---------------------------------------------
Given the 5000 archived eval log-weights log w_i:

* Var[log w]      — POPULATION variance (ddof=0). Chosen over the sample
                    (ddof=1) variance because the statistic is defined as the
                    batch's mean squared deviation — a moment of the empirical
                    distribution, not an unbiased estimate of some
                    infinite-population variance. At N=5000 the two differ by
                    0.02%, far below the bootstrap CI width, so the choice is
                    about definition, not numerics.
* Var[log w]/site — divided by d = D^2 lattice sites (config.json ising.D),
                    making runs at different lattice sizes comparable: log w
                    is extensive, so its variance grows ~linearly with d for
                    a fixed per-site model quality.
* ESS/N           — self-normalised: (sum w)^2 / (N sum w^2) with a
                    max-log-weight shift before exponentiating so the largest
                    weight is exp(0)=1 and nothing overflows.
* chi^2           — 1/(ESS/N) - 1. The identity ESS/N = 1/(1+chi^2), where
                    chi^2 is the empirical chi-square divergence of the
                    self-normalised weights from uniform, is exact and
                    assumption-free (it is algebra, not an approximation), so
                    the chi-square column comes free from the ESS column.
* nats/site       — log(1+chi^2)/d, the per-site log chi-square gap; a second
                    size-normalised quality axis derived from the identity.
* top-weight      — largest normalised weight. Direct degeneracy witness:
                    values near 1 mean the "5000-draw" estimate is one draw.

Uncertainty: bootstrap, not analytic
------------------------------------
Both Var[log w] and ESS/N get a 95% percentile CI from the same 2000
nonparametric bootstrap resamples (fixed seed, so the table is reproducible).
Bootstrap rather than an analytic CI because the analytic standard error of a
variance needs the FOURTH moment (kurtosis) of log w — and heavy tails, the
very pathology under study, are where empirical fourth moments are least
trustworthy. The percentile bootstrap assumes only exchangeability of the
draws, and using the identical resample indices for both statistics puts the
two CIs on exactly equal footing, which is the comparison this script exists
to make. Caveat kept honest: when a single weight dominates, the bootstrap
CI for ESS/N is itself unreliable (each resample either contains that draw
or not) — that unreliability is part of the finding, not a bug to hide.

Resolution criterion
--------------------
Two rows are called RESOLVED under a statistic when their 95% CIs are
disjoint. Disjoint 95% intervals imply a difference significant beyond the
5% level, so this is a conservative call: overlapping CIs are recorded as
unresolved even though a sharper paired test might separate them. For
ranking claims in a thesis, conservative is the right default — a pair
marked resolved here stays resolved under any reasonable test.

Output
------
* results/03_hard/weight_variance_ranking.csv — every eval found, ranked by
  Var[log w]/site ascending, with both CIs and adjacent-pair resolution
  marks (adjacency taken within the same lattice size, since cross-size
  rankings are only meaningful after the per-site normalisation and are not
  the comparisons the archive was making).
* A printed focus table and pairwise resolution matrices for the 16x16
  sigma_c-class runs, under Var[log w] and under ESS/N side by side.

Run with:  pixi run -e dev python experiments/constrained_hard_03/analysis_weight_variance_ranking.py
CPU-only; loads nothing but 5000-float tensors and JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "03_hard"
OUTPUT_CSV = RESULTS_ROOT / "weight_variance_ranking.csv"

N_BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260819  # fixed so the table (and its CIs) is reproducible
CI_PERCENTILES = (2.5, 97.5)

# Only these subdirectory patterns hold pure importance-sampling weights on
# the standard frozen-eval protocol. SMC and replicate evals are excluded:
# post-resampling SMC weights are near-uniform by construction, so their
# variance measures the resampler, not the model.
ELIGIBLE_EVAL_SUBDIR_PATTERNS = ("eval", "eval_ema", "eval_ne*")


@dataclass
class EvalRow:
    run_name: str
    eval_subdir: str
    n_sites: int
    n_draws: int
    var_log_w: float
    var_log_w_ci: tuple[float, float]
    ess_over_n: float
    ess_over_n_ci: tuple[float, float]
    top_weight_fraction: float
    ess_fraction_recorded: float | None  # metrics.json value, cross-check only


def self_normalised_ess_fraction(log_weights: np.ndarray) -> float:
    """ESS/N = (sum w)^2 / (N sum w^2), computed with a max-shift.

    Subtracting max(log w) before exponentiating makes the largest weight
    exactly 1; without it, log-weights in the hundreds (routine at d=256,
    where log w is extensive) overflow float64 immediately. The shift
    cancels exactly in the ratio, so this is a numerical guard, not an
    approximation.
    """
    shifted = log_weights - log_weights.max()
    weights = np.exp(shifted)
    return float(weights.sum() ** 2 / (len(weights) * (weights**2).sum()))


def top_normalised_weight(log_weights: np.ndarray) -> float:
    """Largest self-normalised weight — the single-draw-domination witness."""
    shifted = log_weights - log_weights.max()
    weights = np.exp(shifted)
    return float(weights.max() / weights.sum())


def bootstrap_confidence_intervals(
    log_weights: np.ndarray, rng: np.random.Generator
) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% percentile CIs for (Var[log w], ESS/N) from shared resamples.

    Identical indices support comparison of the two statistics' pairwise
    resolution. Vectorised (B, N) arrays at B=2000, N=5000 use ~80 MB
    each for int64 indices or float64 values.
    """
    n_draws = len(log_weights)
    resample_indices = rng.integers(0, n_draws, size=(N_BOOTSTRAP_RESAMPLES, n_draws))
    resampled = log_weights[resample_indices]  # (B, N)

    variance_per_resample = resampled.var(axis=1)  # population variance, ddof=0

    shifted = resampled - resampled.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    ess_per_resample = weights.sum(axis=1) ** 2 / (n_draws * (weights**2).sum(axis=1))

    variance_ci = tuple(np.percentile(variance_per_resample, CI_PERCENTILES))
    ess_ci = tuple(np.percentile(ess_per_resample, CI_PERCENTILES))
    return variance_ci, ess_ci


def lattice_sites_from_config(run_dir: Path) -> int | None:
    """d = D^2 from config.json (ising.D is the lattice SIDE, e.g. 16 -> 256)."""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        side_length = json.loads(config_path.read_text())["ising"]["D"]
    except (KeyError, json.JSONDecodeError, TypeError):
        return None
    return int(side_length) ** 2


def recorded_ess_fraction(eval_dir: Path) -> float | None:
    """The eval's own metrics.json ess_fraction, used only as a wiring check."""
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        return float(json.loads(metrics_path.read_text())["ess_fraction"])
    except (KeyError, json.JSONDecodeError, ValueError):
        return None


def collect_rows() -> tuple[list[EvalRow], int, int]:
    """Walk the archive; return (rows, runs_skipped_no_weights, runs_skipped_no_config)."""
    rows: list[EvalRow] = []
    runs_without_weights = 0
    runs_without_config = 0

    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_dirs = [
            candidate
            for pattern in ELIGIBLE_EVAL_SUBDIR_PATTERNS
            for candidate in sorted(run_dir.glob(pattern))
            if candidate.is_dir() and (candidate / "log_weights.pt").is_file()
        ]
        if not eval_dirs:
            runs_without_weights += 1
            continue

        n_sites = lattice_sites_from_config(run_dir)
        if n_sites is None:
            runs_without_config += 1
            continue

        for eval_dir in eval_dirs:
            log_weights = (
                torch.load(eval_dir / "log_weights.pt", map_location="cpu", weights_only=True)
                .double()
                .numpy()
                .ravel()
            )
            # Seed by eval identity so adding archive runs cannot change a row's CI.
            per_eval_rng = np.random.default_rng(
                [BOOTSTRAP_SEED, *f"{run_dir.name}/{eval_dir.name}".encode()]
            )
            variance_ci, ess_ci = bootstrap_confidence_intervals(log_weights, per_eval_rng)
            rows.append(
                EvalRow(
                    run_name=run_dir.name,
                    eval_subdir=eval_dir.name,
                    n_sites=n_sites,
                    n_draws=len(log_weights),
                    var_log_w=float(log_weights.var()),  # population variance
                    var_log_w_ci=variance_ci,
                    ess_over_n=self_normalised_ess_fraction(log_weights),
                    ess_over_n_ci=ess_ci,
                    top_weight_fraction=top_normalised_weight(log_weights),
                    ess_fraction_recorded=recorded_ess_fraction(eval_dir),
                )
            )
    return rows, runs_without_weights, runs_without_config


def build_table(rows: list[EvalRow]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "run": [r.run_name for r in rows],
            "eval_subdir": [r.eval_subdir for r in rows],
            "d": [r.n_sites for r in rows],
            "n_draws": [r.n_draws for r in rows],
            "var_log_w": [r.var_log_w for r in rows],
            "var_log_w_ci_lo": [r.var_log_w_ci[0] for r in rows],
            "var_log_w_ci_hi": [r.var_log_w_ci[1] for r in rows],
            "ess_over_n": [r.ess_over_n for r in rows],
            "ess_over_n_ci_lo": [r.ess_over_n_ci[0] for r in rows],
            "ess_over_n_ci_hi": [r.ess_over_n_ci[1] for r in rows],
            "top_weight_fraction": [r.top_weight_fraction for r in rows],
            "ess_fraction_recorded": [r.ess_fraction_recorded for r in rows],
        }
    )
    table["var_per_site"] = table["var_log_w"] / table["d"]
    table["var_per_site_ci_lo"] = table["var_log_w_ci_lo"] / table["d"]
    table["var_per_site_ci_hi"] = table["var_log_w_ci_hi"] / table["d"]
    # Exact identity columns: ESS/N = 1/(1+chi^2)  =>  chi^2 = 1/(ESS/N) - 1,
    # and log(1+chi^2) = -log(ESS/N) is the chi-square gap in nats.
    table["chi_squared"] = 1.0 / table["ess_over_n"] - 1.0
    table["nats_per_site"] = np.log1p(table["chi_squared"]) / table["d"]

    table = table.sort_values("var_per_site", ascending=True, kind="stable").reset_index(drop=True)

    # Compare adjacent rows within each lattice size, matching the archive's
    # within-size ranking claims.
    table["var_resolved_vs_next_same_d"] = _adjacent_resolution(
        table, "var_log_w_ci_lo", "var_log_w_ci_hi"
    )
    table["ess_resolved_vs_next_same_d"] = _adjacent_resolution(
        table, "ess_over_n_ci_lo", "ess_over_n_ci_hi"
    )
    return table


def _adjacent_resolution(table: pd.DataFrame, lo_col: str, hi_col: str) -> list[object]:
    """For each row: True/False if its CI is disjoint/overlapping with the
    next row of the SAME lattice size in the ranking; None for the last row
    of each size group."""
    marks: list[object] = [None] * len(table)
    for _, group in table.groupby("d", sort=False):
        positions = group.index.to_list()
        for earlier, later in zip(positions, positions[1:]):
            disjoint = intervals_disjoint(
                (table.at[earlier, lo_col], table.at[earlier, hi_col]),
                (table.at[later, lo_col], table.at[later, hi_col]),
            )
            marks[earlier] = bool(disjoint)
    return marks


def intervals_disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[1] < b[0] or b[1] < a[0]


def pairwise_resolution_report(table: pd.DataFrame, label: str) -> None:
    """Print each pair's resolved/unresolved status under Var[log w] and ESS/N."""
    print(f"\n=== Pairwise resolution, {label} ({len(table)} evals) ===")
    resolved_var = resolved_ess = 0
    n_pairs = 0
    disagreements: list[str] = []
    rows = table.reset_index(drop=True)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows.iloc[i], rows.iloc[j]
            var_ok = intervals_disjoint(
                (a.var_log_w_ci_lo, a.var_log_w_ci_hi), (b.var_log_w_ci_lo, b.var_log_w_ci_hi)
            )
            ess_ok = intervals_disjoint(
                (a.ess_over_n_ci_lo, a.ess_over_n_ci_hi), (b.ess_over_n_ci_lo, b.ess_over_n_ci_hi)
            )
            n_pairs += 1
            resolved_var += var_ok
            resolved_ess += ess_ok
            if var_ok != ess_ok:
                disagreements.append(
                    f"  {'VAR-only' if var_ok else 'ESS-only'}: "
                    f"{shorten(a.run)}/{a.eval_subdir}  vs  {shorten(b.run)}/{b.eval_subdir}"
                )
    print(f"pairs: {n_pairs}   resolved by Var[log w] CI: {resolved_var}   by ESS/N CI: {resolved_ess}")
    if disagreements:
        print("pairs resolved under one statistic only:")
        print("\n".join(disagreements))


def shorten(run_name: str) -> str:
    """Trim the shared H2_d256_c50_s223_letf_ prefix for readable printing."""
    for prefix in ("H2_d256_c50_s223_letf_", "H2_d256_", "H2_"):
        if run_name.startswith(prefix):
            return run_name[len(prefix):]
    return run_name


def main() -> None:
    rows, runs_without_weights, runs_without_config = collect_rows()
    table = build_table(rows)
    table.to_csv(OUTPUT_CSV, index=False)

    print(f"evals analysed: {len(table)} across {table['run'].nunique()} runs")
    print(f"run dirs skipped (no eval-like log_weights.pt): {runs_without_weights}")
    if runs_without_config:
        print(f"run dirs skipped (weights present but no parsable config.json): {runs_without_config}")
    print(f"ranked table written to {OUTPUT_CSV}")

    # Cross-check archived ESS/N, allowing float32/float64 accumulation drift.
    checkable = table.dropna(subset=["ess_fraction_recorded"])
    if len(checkable):
        relative_gap = (
            (checkable["ess_over_n"] - checkable["ess_fraction_recorded"]).abs()
            / checkable["ess_fraction_recorded"]
        )
        print(
            f"ESS/N cross-check vs recorded metrics.json ({len(checkable)} evals): "
            f"max relative gap {relative_gap.max():.2e}"
        )

    # Focus group: the 16x16 sigma_c-class full-horizon runs — the arms whose
    # ESS/N rankings were shown to be statistically empty.
    focus = table[(table["d"] == 256) & table["run"].str.contains("50k_curr")].copy()
    focus["run_short"] = focus["run"].map(shorten)

    display_columns = {
        "run_short": "run",
        "eval_subdir": "eval",
        "var_log_w": "Var[log w]",
        "var_per_site": "Var/site",
        "var_per_site_ci_lo": "Var/site lo",
        "var_per_site_ci_hi": "Var/site hi",
        "ess_over_n": "ESS/N",
        "ess_over_n_ci_lo": "ESS/N lo",
        "ess_over_n_ci_hi": "ESS/N hi",
        "nats_per_site": "nats/site",
        "top_weight_fraction": "top-w",
    }
    print("\n=== 16x16 sigma_c-class runs, ranked by Var[log w]/site (ascending = better) ===")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(
            focus[list(display_columns)]
            .rename(columns=display_columns)
            .to_string(index=False, float_format=lambda value: f"{value:.4g}")
        )

    pairwise_resolution_report(focus, "16x16 sigma_c-class (d=256, 50k_curr)")

    # Rank-inversion check across the whole table, within each lattice size:
    # under heavy tails the ESS ordering can invert the variance ordering
    # (a run with LOWER Var[log w] showing WORSE ESS/N because one lucky draw
    # inflated a rival's ESS). Surface any strong disagreements explicitly.
    print("\n=== Var-rank vs ESS-rank disagreements (|rank gap| >= 3 within a lattice size) ===")
    any_inversion = False
    for n_sites, group in table.groupby("d"):
        group = group.copy()
        group["var_rank"] = group["var_per_site"].rank(method="min")
        group["ess_rank"] = group["ess_over_n"].rank(method="min", ascending=False)
        group["rank_gap"] = group["ess_rank"] - group["var_rank"]
        strong = group[group["rank_gap"].abs() >= 3]
        for _, row in strong.iterrows():
            any_inversion = True
            print(
                f"  d={n_sites}: {shorten(row['run'])}/{row['eval_subdir']}  "
                f"var-rank {int(row['var_rank'])} vs ess-rank {int(row['ess_rank'])}  "
                f"(Var/site {row['var_per_site']:.4f}, ESS/N {row['ess_over_n']:.4g}, "
                f"top-w {row['top_weight_fraction']:.2f})"
            )
    if not any_inversion:
        print("  none at this threshold")


if __name__ == "__main__":
    main()
