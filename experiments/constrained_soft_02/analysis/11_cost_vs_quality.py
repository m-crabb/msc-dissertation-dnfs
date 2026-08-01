"""Cost-vs-quality grid: seconds per effective sample, DNFS against mchammer.

The supervisor's ask is a grid of time-per-good-sample rather than two
unrelated quality tables, because wall-clock is the only currency the neural
sampler and the MCMC reference actually share. Quality alone flatters whichever
method was given more compute; cost alone ignores that a fast sampler drawing
correlated junk is worthless. The joining quantity is therefore

    seconds per effective sample = wall-clock to produce a batch / ESS of it

which both sides can report honestly, because both already carry an ESS whose
denominator means "how many independent draws was this batch worth".

The two sides compute that ESS differently, and the difference is the point:

  * DNFS draws i.i.d. samples and pays in *weight* variance, so its ESS is the
    self-normalised importance-sampling ESS, `ess_fraction * n_eval_samples`.
    Cost is `eval_draw_seconds`, the measured time to draw the eval batch.
  * mchammer draws correlated samples and pays in *autocorrelation*, so its ESS
    is `n_frames / tau_int_frames`. `seconds_per_effective_sample` is already
    computed per observable in its summary.

A consequence worth stating rather than hiding: the mchammer figure is
per-observable (composition mixes faster than potential, so quoting the
composition number alone understates its cost for thermodynamic estimands).
Both are reported.

HARDWARE PAIRING — mchammer runs on an Apple M-series CPU, DNFS on an NVIDIA
a30 GPU, and this script prints the hostnames rather than letting the reader
assume a common machine. That pairing is deliberate and conservative: mchammer
is framed as a reference, not a rival, so timing it on the fastest hardware to
hand makes the baseline as strong as possible and understates any DNFS
advantage. The same cells on a cluster CPU ran ~3x slower, which would have
flattered DNFS.

Example:
    python -m experiments.constrained_soft_02.analysis.11_cost_vs_quality \\
        --results-dir results/02_constrained_soft --D 10
"""
import argparse
import json
from pathlib import Path

import pandas as pd

MCHAMMER_SOFT = Path("results/mchammer_vcsgc")


def _seconds_per_effective_sample(metrics: dict) -> float | None:
    """DNFS cost, or None when the run predates the timing fields.

    Guarded rather than defaulted: a missing `eval_draw_seconds` means the run
    was never timed, and inventing a zero (or reusing another run's) would put
    a fabricated number in the one table whose whole purpose is cost.
    """
    seconds = metrics.get("eval_draw_seconds")
    ess = metrics.get("ess")
    if seconds is None or not ess:
        return None
    return seconds / ess


def collect_dnfs(
    results_dir: Path, *, D: int, seeds: list[int] | None = None
) -> pd.DataFrame:
    """One row per (run, composition) that carries timing.

    Reads both artefact shapes: a swept amortised run contributes one row per
    composition from `composition_sweep.json`, a specialist contributes its
    single `metrics.json` row at its own target composition.

    `seeds` restricts which runs are read, and it exists because cost here is
    1/ESS: a seed that collapsed contributes a huge number that dominates the
    seed-mean, so the averaged row describes no run that was ever performed.
    The amortised cells mix collapsed and healthy seeds, and the two questions
    "what does this recipe cost when it works" and "how often does it work"
    have to be answered separately. Quote a filtered row only alongside the
    survival rate — on its own it is cherry-picking.
    """
    rows = []
    for config_path in sorted(results_dir.glob("*/config.json")):
        run_dir = config_path.parent
        cfg = json.loads(config_path.read_text())
        if cfg["ising"]["D"] != D:
            continue
        if seeds is not None and cfg["train"]["seed"] not in seeds:
            continue
        sweep_path = run_dir / "eval" / "composition_sweep.json"
        metrics_path = run_dir / "eval" / "metrics.json"
        if sweep_path.exists():
            entries = [
                (row["composition"], row)
                for row in json.loads(sweep_path.read_text())
            ]
        elif metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            entries = [(cfg["ising"].get("target_composition"), metrics)]
        else:
            continue
        for composition, metrics in entries:
            cost = _seconds_per_effective_sample(metrics)
            if cost is None:
                continue
            rows.append({
                "cell": cfg["name"],
                "seed": cfg["train"]["seed"],
                "composition": composition,
                "ess_fraction": metrics.get("ess_fraction"),
                "dnfs_s_per_eff": cost,
                "nfe_per_eff": metrics.get("nfe_per_effective_sample"),
                "device": metrics.get("eval_device"),
            })
    return pd.DataFrame(rows)


def collect_mchammer(baseline_dir: Path, *, D: int) -> pd.DataFrame:
    """One row per mchammer cell, both observables kept separate."""
    rows = []
    for summary_path in sorted(baseline_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary["D"] != D:
            continue
        observables = summary["observables"]
        rows.append({
            "composition": summary["target_composition"],
            "seed": summary["seed"],
            "mcmc_s_per_eff_composition": observables.get(
                "composition", {}
            ).get("seconds_per_effective_sample"),
            "mcmc_s_per_eff_potential": observables.get(
                "potential", {}
            ).get("seconds_per_effective_sample"),
            "steps_per_second": summary["steps_per_second"],
            "hostname": summary["hostname"],
        })
    return pd.DataFrame(rows)


def build_grid(dnfs: pd.DataFrame, mcmc: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean each side, join on composition, and form the cost ratio.

    Seed-meaned before joining because the two sides do not share a seed
    axis — DNFS seed 42 and mchammer seed 42 are unrelated random streams, so
    pairing them row-wise would imply a correspondence that does not exist.

    The ratio is mchammer over DNFS, i.e. how many seconds of MCMC one second
    of DNFS is worth per effective sample; >1 favours DNFS. It is deliberately
    not computed against the potential column as well, because a single
    headline ratio invites quoting the flattering observable — the potential
    cost is on the page for the reader to form that ratio themselves.
    """
    if dnfs.empty or mcmc.empty:
        return pd.DataFrame()
    dnfs_agg = dnfs.groupby(["cell", "composition"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        ess_fraction=("ess_fraction", "mean"),
        dnfs_s_per_eff=("dnfs_s_per_eff", "mean"),
        nfe_per_eff=("nfe_per_eff", "mean"),
    )
    mcmc_agg = mcmc.groupby("composition", as_index=False).agg(
        mcmc_s_per_eff_composition=("mcmc_s_per_eff_composition", "mean"),
        mcmc_s_per_eff_potential=("mcmc_s_per_eff_potential", "mean"),
    )
    grid = dnfs_agg.merge(mcmc_agg, on="composition", how="outer")
    grid["mcmc_over_dnfs"] = (
        grid["mcmc_s_per_eff_composition"] / grid["dnfs_s_per_eff"]
    )
    return grid.sort_values(["cell", "composition"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/02_constrained_soft")
    parser.add_argument("--baseline-dir", default=str(MCHAMMER_SOFT))
    parser.add_argument("--D", type=int, default=10)
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        help="Restrict the DNFS side to these seeds. Quote such a row only "
             "next to the survival rate of the cell it came from.",
    )
    parser.add_argument("--out", help="Optional CSV path for the grid")
    args = parser.parse_args()

    dnfs = collect_dnfs(Path(args.results_dir), D=args.D, seeds=args.seeds)
    mcmc = collect_mchammer(Path(args.baseline_dir), D=args.D)

    print(f"DNFS timed runs at D={args.D}: {len(dnfs)} rows")
    if not dnfs.empty:
        print(f"  devices: {sorted(dnfs.device.dropna().unique())}")
    print(f"mchammer cells at D={args.D}: {len(mcmc)} rows")
    if not mcmc.empty:
        print(f"  hosts: {sorted(mcmc.hostname.unique())}")
    print()

    grid = build_grid(dnfs, mcmc)
    if grid.empty:
        print(
            "No grid: one side is empty. DNFS timing fields "
            "(`eval_draw_seconds`) exist only on runs from 2026-08-01 onward, "
            "so archived specialists cannot be costed — only re-run or "
            "in-flight cells appear here."
        )
        return
    print(grid.to_string(index=False))
    if args.out:
        grid.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
