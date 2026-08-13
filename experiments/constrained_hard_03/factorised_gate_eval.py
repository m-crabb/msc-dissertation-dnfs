"""Fidelity pass for the factorised-head 4x4 gate cells.

Re-rolls each trained facgate cell (fab8 / fab16 / fbil / fglo at both sigma
points, seeds 42-44) through the exact-enumeration gate, so the factorised
arms read in the SAME currency as the archived MA/MO demo observables table:
energy_tv, ess_fraction, max_level_excess (plus antisym_violation and the
free-energy bias as sanity columns). run_gate / load_run are reused verbatim
-- this script only iterates and tabulates; the pass/partial/fail bands are
frozen elsewhere and applied by the reader, not here.

The archived demo table used n_samples=5000 with the cell's own
n_euler_steps; those defaults are kept so numbers are comparable. The demo
numbers were measured on a cluster GPU -- a CPU rerun differs only through
sampling noise at 5000 rollouts, but disclose the device next to any
side-by-side.
"""

import argparse
import json
from pathlib import Path

from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.gate_4x4 import (
    latest_run_dir,
    load_run,
    run_gate,
)

ARMS = ["fab8", "fab16", "fbil", "fglo"]
SIGMA_TAGS = ["s010", "s223"]

REPORT_COLUMNS = [
    "energy_tv", "ess_fraction", "max_level_excess",
    "antisym_violation", "free_energy_bias",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/03_hard")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/03_hard/factorised_gate_4x4")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for arm in ARMS:
        for sigma_tag in SIGMA_TAGS:
            cfg_name = f"H2_d16_c50_{sigma_tag}_letf_{arm}_10k"
            n_euler_steps = CONFIGS[cfg_name].ctmc.n_euler_steps
            for seed in seeds:
                run_dir = latest_run_dir(args.results_dir, cfg_name, seed)
                print(f"[facgate-eval] {cfg_name} seed {seed}: {run_dir.name}",
                      flush=True)
                head, target = load_run(run_dir, args.device)
                metrics = run_gate(
                    head, target, args.n_samples, n_euler_steps, seed
                )
                row = {
                    "cell": cfg_name, "arm": arm, "sigma_tag": sigma_tag,
                    "seed": seed, "run_dir": run_dir.name,
                    "device": args.device, "n_samples": args.n_samples,
                }
                row.update({c: float(metrics[c]) for c in REPORT_COLUMNS})
                rows.append(row)
                print(
                    f"[facgate-eval]   ess_frac {row['ess_fraction']:.3f} "
                    f"energy_tv {row['energy_tv']:.4f} "
                    f"max_level_excess {row['max_level_excess']:.4f}",
                    flush=True,
                )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fidelity.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "| cell | seed | energy_tv | ess_frac | max_level_excess |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['cell']} | {row['seed']} | {row['energy_tv']:.4f} "
        f"| {row['ess_fraction']:.3f} | {row['max_level_excess']:.4f} |"
        for row in rows
    ]
    (out_dir / "observables_table.md").write_text("\n".join(lines) + "\n")
    print(f"[facgate-eval] wrote {out_dir / 'fidelity.json'}", flush=True)


if __name__ == "__main__":
    main()
