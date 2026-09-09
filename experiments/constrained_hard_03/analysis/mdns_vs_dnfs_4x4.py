"""MDNS-vs-DNFS 4x4 side-by-side: the second-pass comparison table.

DNFS anchor: the 10k budget-matched head-twin gate record — configs
H2_d16_c50_{s010,s223}_letf_{ma,mo}_10k, per-seed artefacts in
results/03_hard/demo_4x4/observables_table.md. The `ma` (masked_attention)
head is the reported cell in hard.tex; `mo` (mask_one) is the reference row.
The earlier anchor (results/03_hard/gate_4x4/verdict.json) is a stale
artefact of the old dh-head gate and must not anchor this comparison — its
s223 numbers belong to that early experiment.

Instruments: energy-marginal TV, within-level max excess, and IS-ESS fraction
are per-seed in the archived table and computed by the same constructions the
MDNS driver reuses (gate_4x4's). Per-site free-energy bias for the DNFS cells
exists in the pack only as recorded per-point bounds (recomputed from the
runs' final-eval log-weights: <= 0.003/site at sigma = 0.10, <= 0.007/site at
sigma_c), so the DNFS column prints those bounds, not per-seed values.

Cost currencies are stated per family and never divided: the DNFS record's
eval spends 5,000 IS draws per replicate at ~5e5 backbone rows per replicate
(backbone-row currency, neff_table.json); MDNS pays d = 16 network calls per
sample over 65,536 rollouts. ESS fractions are the comparable quantity,
absolute ESS is not.
"""

import argparse
import json
import re
from pathlib import Path

SHARED_METRICS = [
    ("energy_tv", "energy TV"),
    ("max_level_excess", "within-level max excess"),
    ("ess_fraction", "ESS fraction"),
    ("free_energy_bias", "free-energy bias /site"),
]
# json float keys as the driver writes them <-> demo-pack cell name stems
OPERATING_POINTS = {"sigma_0.1": "s010", "sigma_0.223": "s223"}
MDNS_ARM_NAMES = {"a": "MDNS A (V0)", "b": "MDNS B (none)", "c": "MDNS C (unconstr.)"}
DNFS_HEAD_LABELS = {
    "ma": "DNFS letf ma 10k (reported)",
    "mo": "DNFS letf mo 10k (reference)",
}
# Recorded per-point bounds, provenance doc §2b (no per-seed artefact).
DNFS_FREE_ENERGY_BIAS_BOUND = {"s010": 0.003, "s223": 0.007}


def parse_demo_pack_table(path):
    """Per-seed {(point, head): {seed: {metric: value}}} from the archived
    markdown table (| cell | seed | energy_tv | ess_frac | max_level_excess |).
    The artefact stays the source of truth."""
    pattern = re.compile(
        r"\|\s*H2_d16_c50_(s\d+)_letf_(ma|mo)_10k\s*\|\s*(\d+)\s*\|"
        r"\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|"
    )
    table = {}
    for line in Path(path).read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            point, head, seed, tv, ess, excess = match.groups()
            table.setdefault((point, head), {})[int(seed)] = {
                "energy_tv": float(tv),
                "ess_fraction": float(ess),
                "max_level_excess": float(excess),
            }
    return table


def mean(values):
    return sum(values) / len(values)


def numeric_cell(values):
    return f"{mean(values):+.4f} [{', '.join(f'{v:+.4f}' for v in values)}]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mdns-verdict",
        default="results/03_hard/mdns_budget_gate2_4x4/verdict_20260813-gate2-10k.json",
    )
    parser.add_argument(
        "--dnfs-table", default="results/03_hard/demo_4x4/observables_table.md"
    )
    args = parser.parse_args(argv)

    mdns = json.loads(Path(args.mdns_verdict).read_text())
    dnfs = parse_demo_pack_table(args.dnfs_table)

    print(
        "DNFS anchor: 10k head-twin gate record "
        "(H2_d16_c50_{s010,s223}_letf_{ma,mo}_10k, "
        "results/03_hard/demo_4x4/) — ma reported, mo reference."
    )
    for sigma_key, point in OPERATING_POINTS.items():
        if sigma_key not in mdns:
            print(f"(no MDNS results for {sigma_key} — skipped)")
            continue
        reports = mdns[sigma_key]["reports"]
        seeds = sorted({report["seed"] for report in reports.values()})
        print(
            f"\n## Operating point {point} "
            f"(sigma = {sigma_key.split('_')[1]}) — mean [per-seed]"
        )
        print("| sampler | " + " | ".join(label for _, label in SHARED_METRICS) + " |")
        print("|" + "---|" * (len(SHARED_METRICS) + 1))

        for head, label in DNFS_HEAD_LABELS.items():
            per_seed = dnfs[(point, head)]
            dnfs_seeds = sorted(per_seed)
            cells = [
                numeric_cell([per_seed[s][key] for s in dnfs_seeds])
                for key, _ in SHARED_METRICS[:3]
            ]
            cells.append(
                f"<= {DNFS_FREE_ENERGY_BIAS_BOUND[point]:.3f} (recorded bound)"
            )
            print(f"| {label} | " + " | ".join(cells) + " |")
        for arm, arm_label in MDNS_ARM_NAMES.items():
            rows = [
                reports[f"{arm}_seed{seed}"]
                for seed in seeds
                if f"{arm}_seed{seed}" in reports
            ]
            if not rows:
                continue
            cells = [
                numeric_cell([row[key] for row in rows]) for key, _ in SHARED_METRICS
            ]
            print(f"| {arm_label} | " + " | ".join(cells) + " |")

        mdns_n = next(iter(reports.values()))["eval_rollouts"]
        print(
            f"\nCost currencies (stated, never divided): "
            f"DNFS = 5,000 IS draws/replicate at ~5e5 backbone rows "
            f"per replicate (backbone-row currency); "
            f"MDNS = 16 network calls/sample x {mdns_n} samples/seed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
