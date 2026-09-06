"""Fill pass for the zero-shot transfer tables (sec:hard-zero-shot).

Two tables from the probe JSONs (`probe_zero_shot_transfer.py` output,
`zero_shot_transfer.json` in each run dir):

  * TEMPERATURE (tab:zero-shot-coupling): the trained slice (c = 0.5) down
    the coupling column at 16x16 — one sigma_c checkpoint early-stopped at
    t* = sigma'/sigma serves every subcritical coupling. Rows are the
    probe's EXACT Euler grid points (k/127), so no coupling is a rounded
    fiction; the k=58 row sits at 0.1006, a 0.6% offset from the certified
    0.1 reference, which matters to correlation comparisons but not to
    ESS.
  * COMPOSITION (tab:zero-shot-composition): the composition grid at
    t* = 1, three columns — 16x16 at sigma_c (the collapse), 16x16 at
    sigma' = 0.1006 (nearly free: the criticality x distance interaction),
    and 8x8 at sigma_c (much milder: the finite-size cutoff of the
    correlation length). An optional fourth column is emitted when the
    amortised (mixture-trained) checkpoints' probe output exists — the
    trained arm beside its null, never before it lands.

RELIABILITY DAGGERS, and why they are principled rather than cosmetic.
The probe stores exp(-Var[log w]) as `ess_fraction_predicted`; when the
identity measured ~= predicted holds the estimator is healthy, and when
measured/predicted >> 1 the measured ESS is a finite-draw FLOOR (at 5000
draws a dead cell cannot read below ~1/5000 x the weight ceiling), i.e.
an upper bound and not a measurement. Cells whose 3-seed mean ratio
exceeds RELIABILITY_BAR are daggered and the caption says what the dagger
means. The d256 sigma_c column trips it below n+ = 112 (ratios 2.0 / 6.5
/ ~116 at n+ = 96 / 80 / 64); every other column is clean.

Seeds are aggregated mean +- SD, matching every house table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

RELIABILITY_BAR = 1.5
SEEDS = (42, 43, 44)

D256_TEMPLATE = ("H2_d256_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w3"
                 "_seed{seed}_20260826-d256-sc")
D64_TEMPLATE = ("H2_d64_c50_s220_letf_thp_50k_curr_w2"
                "_seed{seed}_20260825-hard-w2-d64")
# Per-slice c_t baseline. The 20260831-camort-d64 twin pooled the baseline
# across composition slices; its ESS read within 0.008 of this one on every
# slice, but only this run is scored.
CAMORT_TEMPLATE = ("H2_d64_camort_s220_letf_thp_50k_curr"
                   "_seed{seed}_20260905-camort-d64-perslice")
# The 16x16 amortised confirmation: one seed by design, thp2 on the same
# five-slice mixture; c = 0.25 is outside the mixture there too. This is the
# pooled-c_t-baseline run; its per-slice twin (tag
# 20260905-camort-d256-perslice) replaces this template once its run dir exists.
D256_CAMORT_TEMPLATE = ("H2_d256_camort_s220_letf_thp2_100k_curr"
                        "_seed{seed}_20260831-camort-d256")

# The cross-chapter composition spine: fractions
# realisable at EVERY rung (n+ = 4/6/8 at d16, 16/24/32 at d64, 64/96/128
# at d256), shared verbatim by any future soft-chapter zero-shot table.
# The finer d256 probe fractions (0.46875, 0.4375, 0.3125) stay in the
# JSONs and resolve the collapse ONSET — a prose point, not rows (0.46875
# does not even exist at d16). 0.625 (the Z2 mirror of 0.375) is also NOT
# a row: it measures the trained head's Z2 symmetry, not transfer.
COMPOSITION_ROWS = (0.25, 0.375, 0.5)   # ascending, as every house table in the thesis


def load_rows(results_dir, template, seeds=SEEDS):
    """rows[(composition, stop_time)] -> list over seeds of the probe row.
    Missing run dirs return None (the column is simply not emitted)."""
    per_seed = []
    for seed in seeds:
        path = (results_dir / template.format(seed=seed)
                / "zero_shot_transfer.json")
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        per_seed.append(data["rows"] if isinstance(data, dict) else data)
    rows = {}
    for seed_rows in per_seed:
        for row in seed_rows:
            key = (round(row["composition"], 6), round(row["stop_time"], 6))
            rows.setdefault(key, []).append(row)
    return rows


def cell(seed_rows):
    """ESS mean +- SD with a dagger when the measured/predicted identity
    breaks (finite-draw floor: the number is an upper bound)."""
    ess = [r["ess_fraction"] for r in seed_rows]
    ratio = np.mean([
        r["ess_fraction"] / max(r["ess_fraction_predicted"], 1e-12)
        for r in seed_rows
    ])
    dagger = "^{\\dagger}" if ratio > RELIABILITY_BAR else ""
    if len(ess) == 1:   # a single-seed column prints its value with the one-seed mark
        return f"${ess[0]:.3f}^{{\\S}}{dagger}$"
    return f"${np.mean(ess):.3f} \\pm {np.std(ess):.3f}{dagger}$"


def temperature_table(d256):
    """One row per stopping time on the trained slice, largest sigma' last."""
    keys = sorted({k for k in d256 if k[0] == 0.5}, key=lambda k: k[1])
    lines = []
    for key in keys:
        seed_rows = d256[key]
        coupling = seed_rows[0]["coupling"]
        stop_time = seed_rows[0]["stop_time"]
        lines.append(
            f"        ${stop_time:.3f}$ & ${coupling:.4f}$ & "
            f"{cell(seed_rows)} \\\\"
        )
    return "\n".join(lines)


def composition_table(columns):
    """One row per composition at t* = 1; `columns` is an ordered dict of
    label -> (rows, stop_time_of_that_column)."""
    lines = []
    for composition in COMPOSITION_ROWS:
        cells = []
        for rows, stop_time in columns.values():
            key = (round(composition, 6), round(stop_time, 6))
            cells.append(cell(rows[key]) if rows and key in rows else "--")
        n_plus_256 = round(composition * 256)
        lines.append(
            f"        ${composition:.5g}$ & ${n_plus_256}$ & "
            + " & ".join(cells) + " \\\\"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=REPO_ROOT / "results" / "03_hard")
    args = parser.parse_args(argv)

    d256 = load_rows(args.results_dir, D256_TEMPLATE)
    d64 = load_rows(args.results_dir, D64_TEMPLATE)
    camort = load_rows(args.results_dir, CAMORT_TEMPLATE)
    camort_d256 = load_rows(args.results_dir, D256_CAMORT_TEMPLATE, seeds=(42,))
    if d256 is None:
        sys.exit("d256 probe JSONs missing; nothing to emit")

    print("% --- tab:zero-shot-coupling body ---")
    print(temperature_table(d256))

    # The subcritical d256 column sits at the k=58 grid point (0.456693).
    columns = {
        "d256_sc": (d256, 1.0),
        "d256_sub": (d256, 0.456693),
    }
    if camort_d256 is not None:
        columns["d256_camort_sc"] = (camort_d256, 1.0)
        print("% d256 camort column INCLUDED (single seed)")
    columns["d64_sc"] = (d64, 1.0)
    if camort is not None:
        columns["d64_camort_sc"] = (camort, 1.0)
        print("% camort column INCLUDED (trained-arm probe found)")
    else:
        print("% camort column omitted (trained-arm probe not on disk yet)")
    print("% --- tab:zero-shot-composition body ---")
    print(composition_table(columns))


if __name__ == "__main__":
    main()
