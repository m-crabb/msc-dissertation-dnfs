"""Aggregate per-seed eval metrics into mean +/- std rows for the report tables.

The report's headline numbers need error bars, so we run each consolidation
config across seeds 42-45 and summarise here. This is pure post-hoc aggregation: it reads the scalar fields
already written to each run's `eval/metrics.json` by `run.py` (ESS fraction,
energy-marginal biases, composition mean/std, ...) and reports
`mean +/- std` with the sample standard deviation (ddof=1, i.e. statistics.stdev).
No metric is (re)defined here; if a number isn't in metrics.json it isn't here.

Run-dir convention (see run.py): `<output_dir>/<cfg>_seed<seed>_<timestamp>`.
We glob `<cfg>_seed<seed>*` per seed and, if several match (e.g. a relaunch
plus the original), take the most recently modified and warn.

Usage:
    python -m scripts.aggregate_seeds \\
        --exp results/02_constrained_soft --cfg S2_d4_c05_l50_letf
    python -m scripts.aggregate_seeds \\
        --exp results/01_baseline --cfg stage_4_d10_budget --seeds 42,43,44,45
"""
import argparse
import json
import statistics
from pathlib import Path

# Curated ordering so the most report-relevant rows print first; any scalar key
# not listed here still gets aggregated, appended alphabetically afterwards.
HEADLINE_ORDER = [
    "ess_fraction",
    "ess",
    "free_energy_per_site_bias",
    "internal_energy_per_site_bias",
    "entropy_per_site_bias",
    "composition_mean",
    "composition_std",
    "composition_abs_error_mean",
    "composition_sq_violation_mean",
    "magnetisation_mean",
    "magnetisation_std",
]


def find_run_dir(exp: Path, cfg: str, seed: int) -> Path | None:
    """Return the run dir for one seed (latest by mtime if several match)."""
    matches = sorted(
        (p for p in exp.glob(f"{cfg}_seed{seed}*")
         if (p / "eval" / "metrics.json").is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [warn] seed {seed}: {len(matches)} matching run dirs, using "
              f"newest ({matches[-1].name})")
    return matches[-1]


def load_metrics(exp: Path, cfg: str, seeds: list[int]) -> dict[int, dict]:
    """Map each seed to its metrics.json dict, skipping seeds with no run."""
    out = {}
    for seed in seeds:
        run_dir = find_run_dir(exp, cfg, seed)
        if run_dir is None:
            print(f"  [warn] seed {seed}: no run dir with eval/metrics.json found")
            continue
        out[seed] = json.loads((run_dir / "eval" / "metrics.json").read_text())
    return out


def ordered_scalar_keys(per_seed: dict[int, dict]) -> list[str]:
    """Scalar keys present in *every* seed, headline keys first then alpha."""
    if not per_seed:
        return []
    def scalar_keys(m: dict) -> set[str]:
        return {k for k, v in m.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}

    shared = set.intersection(*(scalar_keys(m) for m in per_seed.values()))
    headline = [k for k in HEADLINE_ORDER if k in shared]
    rest = sorted(shared - set(headline))
    return headline + rest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, type=Path,
                   help="experiment results dir, e.g. results/02_constrained_soft")
    p.add_argument("--cfg", required=True,
                   help="config name, e.g. S2_d4_c05_l50_letf")
    p.add_argument("--seeds", default="42,43,44,45",
                   help="comma-separated seeds (default 42,43,44,45)")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Aggregating {args.cfg} over seeds {seeds} under {args.exp}")
    per_seed = load_metrics(args.exp, args.cfg, seeds)
    found = sorted(per_seed)
    if not found:
        print("No runs found; nothing to aggregate.")
        return
    print(f"Found {len(found)}/{len(seeds)} seeds: {found}\n")

    keys = ordered_scalar_keys(per_seed)
    name_w = max(len(k) for k in keys)
    header = f"{'metric':<{name_w}}  {'mean +/- std':>22}   per-seed"
    print(header)
    print("-" * len(header))
    for key in keys:
        vals = [per_seed[s][key] for s in found]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        per = ", ".join(f"{v:.4g}" for v in vals)
        print(f"{key:<{name_w}}  {mean:>10.5g} +/- {std:<8.3g}   [{per}]")


if __name__ == "__main__":
    main()
