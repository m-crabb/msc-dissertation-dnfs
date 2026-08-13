"""MDNS-vs-DNFS 4x4 side-by-side: the Amendment-01 comparison table.

Reads the budget-masked MDNS gate verdict (Amendment-01 rerun) and the
ARCHIVED DNFS 4x4 gate verdict (results/03_hard/gate_4x4/verdict.json —
the per-seed artefact numbers, deliberately not the prose quotes in
hard.tex) and prints one table per shared operating point over the four
shared instruments: energy-marginal TV, within-level max excess, IS-ESS
fraction, and per-site free-energy bias. All four are computed by the
same code for both families (gate_4x4's constructions, reused by the MDNS
driver), so rows differ by sampler, not by instrument.

Cost currencies are stated per family and NEVER divided (bracket
discipline, the 2026-08-13 framing ruling): MDNS pays d = 16 network
calls per sample (one per site revelation); DNFS pays n_euler = 128 rate
evaluations per sample. Eval sample counts also differ (65,536 vs 5,000)
and are printed, not normalised away — ESS FRACTIONS are the comparable
quantity, absolute ESS is not.
"""
import argparse
import json
from pathlib import Path

SHARED_METRICS = [
    ("energy_tv", "energy TV"),
    ("max_level_excess", "within-level max excess"),
    ("ess_fraction", "ESS fraction"),
    ("free_energy_bias", "free-energy bias /site"),
]
# json float keys as the driver writes them <-> DNFS verdict rung names
OPERATING_POINTS = {"sigma_0.1": "s010", "sigma_0.223": "s223"}
MDNS_ARM_NAMES = {"a": "MDNS A (V0)", "b": "MDNS B (none)",
                  "c": "MDNS C (unconstr.)"}


def mean(values):
    return sum(values) / len(values)


def format_row(label, per_seed_metric_lists):
    cells = []
    for values in per_seed_metric_lists:
        cells.append(
            f"{mean(values):+.4f} [{', '.join(f'{v:+.4f}' for v in values)}]"
        )
    return f"| {label} | " + " | ".join(cells) + " |"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mdns-verdict",
        default="results/03_hard/mdns_budget_gate2_4x4/"
                "verdict_20260813-gate2-10k.json")
    parser.add_argument(
        "--dnfs-verdict", default="results/03_hard/gate_4x4/verdict.json")
    args = parser.parse_args(argv)

    mdns = json.loads(Path(args.mdns_verdict).read_text())
    dnfs = json.loads(Path(args.dnfs_verdict).read_text())

    for sigma_key, rung in OPERATING_POINTS.items():
        if sigma_key not in mdns:
            print(f"(no MDNS results for {sigma_key} — skipped)")
            continue
        reports = mdns[sigma_key]["reports"]
        seeds = sorted({report["seed"] for report in reports.values()})
        print(f"\n## Operating point {rung} "
              f"(sigma = {sigma_key.split('_')[1]}) — "
              f"mean [per-seed {seeds}]")
        header = "| sampler | " + " | ".join(
            label for _, label in SHARED_METRICS) + " |"
        print(header)
        print("|" + "---|" * (len(SHARED_METRICS) + 1))

        dnfs_per_seed = dnfs["rungs"][rung]["per_seed"]
        print(format_row(
            "DNFS swap CTMC (dh)",
            [[seed_row[key] for seed_row in dnfs_per_seed]
             for key, _ in SHARED_METRICS],
        ))
        for arm, arm_label in MDNS_ARM_NAMES.items():
            rows = [reports[f"{arm}_seed{seed}"] for seed in seeds
                    if f"{arm}_seed{seed}" in reports]
            if not rows:
                continue
            print(format_row(
                arm_label,
                [[row[key] for row in rows] for key, _ in SHARED_METRICS],
            ))

        dnfs_n = dnfs_per_seed[0]["n_samples"]
        mdns_n = next(iter(reports.values()))["eval_rollouts"]
        print(f"\nCost currencies (stated, never divided): "
              f"DNFS = 128 rate evaluations/sample x {dnfs_n} samples/seed; "
              f"MDNS = 16 network calls/sample x {mdns_n} samples/seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
