"""Per-composition table: one amortised model against six specialists.

The amortisation claim is that a single conditioned model, trained once,
matches a fleet of per-composition specialists at the compositions they were
each trained for — and keeps working between them, where no specialist exists.
This script assembles that comparison from artefacts already on disk:

  * amortised side — `eval/composition_sweep.json` in each amortised run dir,
    written by `dnfs_baseline_01.run.composition_sweep`, one row per
    composition (the six specialist values plus four held-out points).
  * specialist side — `eval/metrics.json` in each archived per-composition run
    dir, which carries `ess_fraction` and `composition_mean` under the same
    keys, so the two sides are directly comparable without re-deriving
    anything.

Specialists are selected by their **config**, not their directory name: D, σ,
the final penalty strength, the presence of the λ anneal, an unconditioned
model, and a uniform base. The last of those matters — the c = 0.80 window has
a `matched_anneal` sibling that trained against a Bernoulli(0.8) base, which is
a different experiment (it failed its own ESS gate) and must not be quoted as
the specialist comparator. Run dirs predating the `base_composition` field are
read as the then-default 0.5.

The comparator set is NOT uniform in Euler budget: c = 0.50 has ne128 seeds,
the other windows are ne64, and ne128 alone moved c = 0.50 from 0.699 to 0.918
mean ESS fraction. The Euler budget is therefore printed per row rather than
averaged away, and an amortised (ne128) row compared against an ne64
specialist is a comparison across two budgets — say so when quoting it.

Example:
    python -m experiments.constrained_soft_02.analysis.10_amortised_vs_specialist \\
        --results-dir results/02_constrained_soft \\
        --cells S2_d10_camort_l50_letf_ne128_anneal \\
                S2_d10_cgrid_l50_letf_ne128_anneal
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from experiments.dnfs_baseline_01.run import HELD_OUT_COMPOSITIONS

REPORTED = ["ess_fraction", "composition_mean"]
# Declared so that "nothing collected" is still a frame with these columns —
# the sweeps land after the specialists, so the half-populated table is the
# normal state of this script, not an edge case.
# `model_kind`, `hidden_dim` and `n_steps` are carried through to the detail
# CSV rather than used for filtering: they are the confounds most likely to
# creep into an "amortised vs specialist" claim, and they belong on the page
# where a reader can check them.
SPECIALIST_COLUMNS = [
    "composition", "seed", "n_euler_steps", "model_kind", "hidden_dim",
    "n_steps", "run", *REPORTED,
]
AMORTISED_COLUMNS = ["cell", "held_out", *SPECIALIST_COLUMNS]


def _run_dirs(results_dir: Path):
    for config_path in sorted(results_dir.glob("*/config.json")):
        yield config_path.parent, json.loads(config_path.read_text())


def _descriptors(cfg: dict) -> dict:
    return {
        "seed": cfg["train"]["seed"],
        "n_euler_steps": cfg["ctmc"]["n_euler_steps"],
        "model_kind": cfg["model"]["kind"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "n_steps": cfg["train"]["n_steps"],
    }


def collect_specialists(
    results_dir: Path,
    *,
    D: int,
    sigma: float,
    penalty: float,
    require_anneal: bool = True,
    model_kind: str = "let",
) -> pd.DataFrame:
    """One row per archived specialist run: its composition and its metrics.

    `model_kind` is a filter, not just a label: the c = 0.30 window at D = 4
    has both a leTF and an `lemlp` specialist, and quoting the wrong one turns
    an amortisation result into an architecture comparison.

    `require_anneal` follows the comparator, not the code. At D = 10 the λ
    anneal is the recipe that took seed survival from 1/4 to 4/4, so the
    archived non-annealed cells are not the thing to beat. At D = 4, λ = 50
    trained 4/4 from scratch and the archived comparators have no anneal at
    all — requiring one there would silently return an empty comparator set.
    """
    rows = []
    for run_dir, cfg in _run_dirs(results_dir):
        ising = cfg["ising"]
        is_specialist = (
            cfg.get("composition") is None
            and not cfg["model"].get("condition_on_composition", False)
            and ising.get("target_composition") is not None
            and ising["D"] == D
            and ising["sigma"] == sigma
            and ising["composition_penalty_strength"] == penalty
            and ising.get("base_composition", 0.5) == 0.5
            and cfg["model"]["kind"] == model_kind
            and (cfg.get("lambda_curriculum") is not None) == require_anneal
        )
        metrics_path = run_dir / "eval" / "metrics.json"
        if not (is_specialist and metrics_path.exists()):
            continue
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "composition": ising["target_composition"],
                **_descriptors(cfg),
                "run": run_dir.name,
                **{key: metrics.get(key) for key in REPORTED},
            }
        )
    return pd.DataFrame(rows, columns=SPECIALIST_COLUMNS)


def collect_amortised(results_dir: Path, cells: list[str]) -> pd.DataFrame:
    """One row per (amortised run, swept composition)."""
    rows = []
    for run_dir, cfg in _run_dirs(results_dir):
        sweep_path = run_dir / "eval" / "composition_sweep.json"
        if cfg["name"] not in cells or not sweep_path.exists():
            continue
        for row in json.loads(sweep_path.read_text()):
            rows.append(
                {
                    "cell": cfg["name"],
                    "composition": row["composition"],
                    "held_out": row["held_out"],
                    **_descriptors(cfg),
                    "run": run_dir.name,
                    **{key: row.get(key) for key in REPORTED},
                }
            )
    return pd.DataFrame(rows, columns=AMORTISED_COLUMNS)


def _one_run_per_seed(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep the newest run dir per (group, seed); report what was dropped.

    The archive holds repeated run dirs for the same cell and seed — a relaunch
    that superseded an earlier attempt, or the same run copied under a new
    timestamp. Averaging over rows would weight such a seed twice, inflating
    (or deflating) the seed mean while `n_seeds` still reports the honest
    distinct-seed count, so the discrepancy is invisible in the output.

    Newest wins because run dirs carry a trailing timestamp and a relaunch is
    the later word on that seed. Duplicates are printed rather than silently
    collapsed: if two dirs for one seed disagree, that is something to look at,
    not something for this function to decide quietly.
    """
    subset = [*keys, "seed"]
    ordered = frame.sort_values("run")
    duplicated = ordered.duplicated(subset=subset, keep="last")
    for _, row in ordered[duplicated].iterrows():
        print(f"  (superseded, excluded from the mean: {row['run']})")
    return ordered[~duplicated]


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Seed-mean of the reported metrics, with the seed count kept visible.

    `n_euler_steps` is always a grouping key, never averaged over: c = 0.50 and
    c = 0.80 each have both ne64 and ne128 specialists, and pooling them would
    fold a known 0.699 → 0.918 effect into the seed mean of the very quantity
    the amortised model is being judged on.

    The seed count is not decoration either: this leg's documented failure mode
    is that some seeds never learn the physics at all (the c = 0.60 ne64 group
    has a 0.000 among its four), so a mean quoted without n hides whether it
    describes a working recipe or one survivor.
    """
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "n_seeds", *REPORTED])
    deduped = _one_run_per_seed(frame, keys)
    aggregated = deduped.groupby(keys, as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{key: (key, "mean") for key in REPORTED},
    )
    return aggregated.sort_values(keys).reset_index(drop=True)


def build_table(
    amortised: pd.DataFrame, specialists: pd.DataFrame
) -> pd.DataFrame:
    """Join the two sides on composition; missing sides stay as NaN.

    An outer join on purpose: the held-out compositions have no specialist by
    construction, and a cell whose sweep has not run yet should show as absent
    rather than silently drop the composition from the table. Where a
    composition has specialists at two Euler budgets it contributes two
    comparator rows, so the budget the amortised model is being compared
    against is always on the page.
    """
    amortised_agg = _aggregate(
        amortised, ["cell", "composition", "n_euler_steps"]
    )
    specialist_agg = _aggregate(specialists, ["composition", "n_euler_steps"])
    if amortised_agg.empty:
        table = specialist_agg.assign(cell=None)
    elif specialist_agg.empty:
        table = amortised_agg
    else:
        table = amortised_agg.merge(
            specialist_agg, on="composition", how="outer",
            suffixes=("_amortised", "_specialist"),
        )
    table["held_out"] = table["composition"].isin(HELD_OUT_COMPOSITIONS)
    return table.sort_values(["cell", "composition"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/02_constrained_soft")
    parser.add_argument(
        "--cells", nargs="+",
        default=[
            "S2_d10_camort_l50_letf_ne128_anneal",
            "S2_d10_cgrid_l50_letf_ne128_anneal",
        ],
    )
    parser.add_argument("--D", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--penalty", type=float, default=50.0)
    parser.add_argument("--model-kind", default="let")
    parser.add_argument(
        "--require-anneal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Comparators must (default) or must not carry the λ anneal — "
             "match whichever recipe the amortised cell was cloned from "
             "(D=10 anneals, D=4 does not)",
    )
    parser.add_argument(
        "--out", help="Optional CSV path for the per-run detail rows"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    specialists = collect_specialists(
        results_dir, D=args.D, sigma=args.sigma, penalty=args.penalty,
        require_anneal=args.require_anneal, model_kind=args.model_kind,
    )
    amortised = collect_amortised(results_dir, args.cells)

    if amortised.empty:
        print(
            "No composition_sweep.json found for "
            f"{', '.join(args.cells)} under {results_dir} — showing the "
            "specialist comparators only.\n"
        )
    print(build_table(amortised, specialists).to_string(index=False))

    if args.out:
        detail = pd.concat(
            [
                specialists.assign(source="specialist"),
                amortised.assign(source="amortised"),
            ],
            ignore_index=True,
        )
        detail.to_csv(args.out, index=False)
        print(f"\nPer-run detail written to {args.out}")


if __name__ == "__main__":
    main()
