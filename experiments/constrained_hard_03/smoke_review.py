"""Five-arm smoke-wave review harness: one table per arm, computed
identically, so the reviewed arms (naive PASS, unclip FAIL) double as the
harness's own validation set.

Measures exactly the stated criteria (configs.py smoke-ladder comment),
nothing else:

- (i) rung-0 escape: grad_norm median on steps [2500, 3500]; bar ~500-scale
  (the passing arm read 293; an order of magnitude above is the failing one).
- (ii) 10k sigma=0.17 survival: least-squares loss slope on [10000, 12000];
  bar <= 0. Transition excursion reported alongside (pre-boundary median
  [9500,10000), peak on [10000,10600), final median [11500,12000)).
- Estimator lens: loss@~5k, train-ESS trajectory (bar: > 15/256 and rising
  at ~5k), var_integrand / var_dt_log_p_tilde ratio (CV mechanism).
- Sim lens: proposal_drop_frac, events/site/step, lambda_dt_p99.
- Clip health: lambda_dt_clipped_frac, log_ratio_clamp_frac.
- Per-rung medians over the last 500 steps of each rung (rungs 0/5k/10k,
  the truncated smoke view of the 50k ladder).

The harness measures and annotates against the bars; the pass/fail call
stays a human judgement (the naive precedent: one failed bar stated
honestly without failing the arm — a harness that decided automatically
would flatten exactly that nuance).
"""
import argparse
import csv
import math
from pathlib import Path

RUNG_BOUNDARIES = (0, 5_000, 10_000, 12_000)   # truncated smoke ladder
ESCAPE_WINDOW = (2_500, 3_500)
SURVIVAL_WINDOW = (10_000, 12_000)
EXCURSION_PRE = (9_500, 10_000)
EXCURSION_PEAK = (10_000, 10_600)
FINAL_WINDOW = (11_500, 12_000)
TRAIN_ESS_BAR = 15.0                            # of 256, estimator lens
ESCAPE_SCALE_BAR = 500.0


def read_log(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if value in ("", None):
                    parsed[key] = math.nan
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = math.nan
            rows.append(parsed)
    return rows


def column(rows, key, lo=None, hi=None):
    """Non-NaN values of one column, optionally step-windowed [lo, hi)."""
    out = []
    for r in rows:
        step = r.get("step", math.nan)
        if lo is not None and not (lo <= step < hi):
            continue
        v = r.get(key, math.nan)
        if not math.isnan(v):
            out.append(v)
    return out


def median(values):
    if not values:
        return math.nan
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def least_squares_slope(steps, values):
    n = len(steps)
    if n < 2:
        return math.nan
    mean_s = sum(steps) / n
    mean_v = sum(values) / n
    cov = sum((s - mean_s) * (v - mean_v) for s, v in zip(steps, values))
    var = sum((s - mean_s) ** 2 for s in steps)
    return cov / var


def arm_report(run_dir):
    rows = read_log(Path(run_dir) / "training_log.csv")
    name = Path(run_dir).name

    escape_grad = median(column(rows, "grad_norm", *ESCAPE_WINDOW))

    survival = [
        (r["step"], r["loss"]) for r in rows
        if SURVIVAL_WINDOW[0] <= r["step"] < SURVIVAL_WINDOW[1]
        and not math.isnan(r["loss"])
    ]
    slope = least_squares_slope([s for s, _ in survival],
                                [v for _, v in survival])
    pre = median(column(rows, "loss", *EXCURSION_PRE))
    peak_values = column(rows, "loss", *EXCURSION_PEAK)
    peak = max(peak_values) if peak_values else math.nan
    final_loss = median(column(rows, "loss", *FINAL_WINDOW))
    final_grad = median(column(rows, "grad_norm", *FINAL_WINDOW))

    per_rung = []
    for rung in range(len(RUNG_BOUNDARIES) - 1):
        end = RUNG_BOUNDARIES[rung + 1]
        window = (end - 500, end)
        per_rung.append({
            "rung": rung,
            "loss": median(column(rows, "loss", *window)),
            "grad": median(column(rows, "grad_norm", *window)),
            "ess": median(column(rows, "ess", RUNG_BOUNDARIES[rung], end)),
        })

    ess_all = column(rows, "ess")
    ess_around_5k = median(column(rows, "ess", 4_000, 6_000))
    ess_final_rung = column(rows, "ess", 10_000, 12_000)
    ess_final_slope = math.nan
    ess_steps = [r["step"] for r in rows
                 if 10_000 <= r["step"] < 12_000
                 and not math.isnan(r.get("ess", math.nan))]
    if ess_steps:
        ess_final_slope = least_squares_slope(ess_steps, ess_final_rung)

    var_ratio_rows = [
        (r["step"],
         r["var_estimator_integrand"] / r["var_dt_log_p_tilde"])
        for r in rows
        if not math.isnan(r.get("var_estimator_integrand", math.nan))
        and not math.isnan(r.get("var_dt_log_p_tilde", math.nan))
        and r["var_dt_log_p_tilde"] != 0.0
    ]
    var_ratio_final = median(
        [v for s, v in var_ratio_rows if 11_000 <= s < 12_000]
    )

    def stat(key, agg):
        vals = column(rows, key)
        if not vals:
            return math.nan
        return {"median": median(vals), "max": max(vals),
                "p95": sorted(vals)[int(0.95 * (len(vals) - 1))]}[agg]

    return {
        "name": name,
        "n_rows": len(rows),
        "last_step": max((r["step"] for r in rows
                          if not math.isnan(r["step"])), default=math.nan),
        "escape_grad_3k": escape_grad,
        "escape_annotation": (
            "pass-scale" if escape_grad < 2 * ESCAPE_SCALE_BAR
            else "ABOVE 500-scale"
        ),
        "survival_slope": slope,
        "survival_annotation": "pass" if slope <= 0 else "FAIL (rising)",
        "excursion": (pre, peak, final_loss),
        "final_grad": final_grad,
        "per_rung": per_rung,
        "loss_at_5k": median(column(rows, "loss", 4_800, 5_000)),
        "ess_around_5k": ess_around_5k,
        "ess_max": max(ess_all) if ess_all else math.nan,
        "ess_final_slope_per_step": ess_final_slope,
        "ess_annotation": (
            "never > 15/256" if (ess_all and max(ess_all) <= TRAIN_ESS_BAR)
            else "clears 15/256 at least once"
        ),
        "var_ratio_final_rung": var_ratio_final,
        "proposal_drop_median": stat("proposal_drop_frac", "median"),
        "proposal_drop_p95": stat("proposal_drop_frac", "p95"),
        "events_median": stat("events_per_site_per_step", "median"),
        "lambda_dt_p99_median": stat("lambda_dt_p99", "median"),
        "clip_frac_median": stat("lambda_dt_clipped_frac", "median"),
        "clip_frac_max": stat("lambda_dt_clipped_frac", "max"),
        "log_ratio_clamp_max": stat("log_ratio_clamp_frac", "max"),
    }


def format_report(report):
    pre, peak, final_loss = report["excursion"]
    rungs = " / ".join(
        f"{r['loss']:.3g}" for r in report["per_rung"]
    ) + " (loss), " + " / ".join(
        f"{r['grad']:.3g}" for r in report["per_rung"]
    ) + " (grad)"
    lines = [
        f"### {report['name']} "
        f"(rows {report['n_rows']}, last step {report['last_step']:.0f})",
        "",
        "| lens | bar | measured | annotation |",
        "|---|---|---|---|",
        f"| (i) rung-0 escape | grad ~500-scale by 3k "
        f"| median {report['escape_grad_3k']:.3g} on [2.5k,3.5k) "
        f"| {report['escape_annotation']} |",
        f"| (ii) 10k survival | loss slope <= 0 on 10k-12k "
        f"| {report['survival_slope']:+.2e}/step "
        f"| {report['survival_annotation']} |",
        f"| transition excursion | contained "
        f"| pre {pre:.3g} -> peak {peak:.3g} -> final {final_loss:.3g} "
        f"| — |",
        f"| estimator: loss@5k | < 5 (twin: 20) "
        f"| {report['loss_at_5k']:.3g} | — |",
        f"| estimator: train ESS | > 15/256, rising at ~5k "
        f"| median {report['ess_around_5k']:.3g} at 5k; max "
        f"{report['ess_max']:.3g}; final-rung slope "
        f"{report['ess_final_slope_per_step']:+.2e}/step "
        f"| {report['ess_annotation']} |",
        f"| estimator: CV mechanism | var ratio ~1 when CV dead/healthy "
        f"| {report['var_ratio_final_rung']:.3g} median on 11k-12k | — |",
        f"| sim: proposal_drop_frac | median < 0.01 "
        f"| {report['proposal_drop_median']:.2g} "
        f"(p95 {report['proposal_drop_p95']:.2g}) | — |",
        f"| sim: events/site/step | < 0.1 "
        f"| {report['events_median']:.2g} | — |",
        f"| sim: lambda_dt_p99 | logged, sane "
        f"| median {report['lambda_dt_p99_median']:.3g} | — |",
        f"| clip health | — | clipped_frac median "
        f"{report['clip_frac_median']:.3g} max {report['clip_frac_max']:.3g}"
        f"; log_ratio_clamp max {report['log_ratio_clamp_max']:.3g} | — |",
        "",
        f"Per-rung medians (last 500 steps each): {rungs}; "
        f"final grad {report['final_grad']:.3g}.",
        "",
    ]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", default=None,
                        help="markdown output path (default: stdout)")
    args = parser.parse_args(argv)
    lines = ["# Smoke-wave deep-review measurements "
             "(harness: smoke_review.py)", ""]
    for run_dir in args.run_dirs:
        lines += format_report(arm_report(run_dir))
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"[smoke-review] wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
