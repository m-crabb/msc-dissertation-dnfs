"""Fill pass for the 24x24 rung of the house evaluation table.

The thinnest rung: one coupling (sigma_c), two cells (patch radius 3 and 4),
three seeds each, tag 20260903-d576-sc, launched 2026-09-03 on the DoC a100
partition and landed 2026-09-04/06. It is a scaling probe of the d400 sigma_c
recipe, and the table reads as one.

  * One coupling by construction: no sigma = 0.1 wave was run at d576 (every
    rung below saturates there), so the body has one five-column half rather
    than an empty sigma = 0.1 half that would read as "not yet landed".
  * The R=3 cell is the d400 sigma_c R=3 bf16 cell moved to the lattice
    (tests/test_configs.py pins two changed fields: the lattice and a loss
    microbatch of 128, gradient-exact and there for memory only). R=4 moves
    the radius and nothing else; capacity is held.
  * bf16 training only, fp32 evaluation, as at d400.

Reference `kawasaki_ref_d576_sc` (2026-09-03, 42 s on the Mac): same generator
and sweep budget as the rungs below (8 chains, 100k burn-in + 102,400 sampling
sweeps, thinned at 2x the worst chain's tau), sigma recorded at exactly
SIGMA_C. tau 21.6-26.2 sweeps per chain against the D^1.5 prediction of ~25
from the d400 pool's 18.9; 15,176 stored draws; no external mchammer anchor
(a d256 property), certification on the internal checks alone.

FLOP/es: every lattice-bound helper is imported from the 20x20 fill with the
side passed explicitly; at their defaults they would bill the reference at
(20/24)^2 = 0.69 of its proposals and score the profiles on the wrong lattice.
Pinned by tests/test_house_table_24x24.py.

GFN rows (added 2026-09-08, tag 20260906-gfn-d576-sc): the 20x20 block
carried up, TB only, hidden 76 at this lattice; FL-DB stays "--" as at 20x20.
The bill is one KV-cached rollout plus one target eval, handed to
`neural_cell` as `flops_raw` exactly as at 20x20.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from experiments.constrained_hard_03.analysis.house_table_8x8 import (
    _sci,
    aggregate,
    flop_billing_config,
    fmt,
    gfn_registry_config_for,
    reference_standard_error,
    registry_config_for,
    sampling_floor_from_reference,
)
from experiments.constrained_hard_03.analysis.house_table_16x16 import reference_row
from experiments.constrained_hard_03.analysis.house_table_20x20 import (
    ERROR_COLUMNS,
    energy_per_site,
    find_cells,
    load_reference,
    neural_cell,
    reference_trial_counts,
)

from discrete_flow_sampler.diagnostics.flops import (
    ising_energy_eval_flops,
    measured_forward_flops,
)
from discrete_flow_sampler.targets.ising import SIGMA_C

REPO_ROOT = Path(__file__).resolve().parents[3]


L = 24
D_SITES = L * L
SIGMA_LABELS = ("s220",)
SIGMA = {"s220": SIGMA_C}
TAGS = {"s220": "20260903-d576-sc"}
REFERENCE_DIRS = {"s220": REPO_ROOT / "results" / "kawasaki_ref_d576_sc"}

ARMS = {
    "thp3_w5bf16": "two-hole patch head, $R=3$",
    "thp4_w5bf16": "two-hole patch head, $R=4$",
}
ARM_CONFIGS = {
    f"{radius}_w5bf16": {
        "s220": f"H2_d576_c50_s220_letf_{radius}_100k_curr_b512_ne128_cv2_w5bf16",
    }
    for radius in ("thp3", "thp4")
}
# GFlowNet comparator rows, outside the bold comparison (`best` runs over
# ARMS, which these are not in). The fldb cell is registered but was never
# launched above 16x16.
GFN_ARMS = {
    "gfn_tb": "GFlowNet, trajectory balance",
    "gfn_fldb": "GFlowNet, forward-looking DB",
}
GFN_TAG = {"gfn_tb": "20260906-gfn-d576-sc", "gfn_fldb": "20260906-gfn-d576-sc"}
GFN_CELL_NAME = {"s220": "GFN_d576_c50_s220_{objective}_100k_par"}

LATEX_ROWS = (
    ("reference", "Kawasaki (mchammer), certified reference"),
    ("floor", "sampling floor at $N=5000$"),
    None,
    *((arm, label) for arm, label in ARMS.items()),
    None,
    *((arm, label) for arm, label in GFN_ARMS.items()),
)


def latex_table(table, n_draws=5000):
    """One five-column half; bold = best measured swap-head value, as at
    every rung, without implying statistical separation."""
    sigma_label = SIGMA_LABELS[0]

    def key_for(arm):
        if arm == "reference":
            return f"reference_{sigma_label}"
        if arm == "floor":
            return f"floor{n_draws}_{sigma_label}"
        return f"{arm}_{sigma_label}"

    def cell(key, column, sci=False):
        entry = table.get(key)
        if entry is None or column not in entry:
            return "--"
        mean, sd = entry[column]
        if sci:
            return _sci(mean)
        if key.startswith("reference"):
            return f"($ {mean * 100:.1f} $)".replace(" ", "")
        if key.startswith("floor"):
            return f"${mean * 100:.1f}$"
        return f"${mean * 100:.1f} \\pm {sd * 100:.1f}$"

    def bold(text):
        return f"$\\mathbf{{{text.strip('$')}}}$"

    arms = [a for a in ARMS if key_for(a) in table]
    best = {}
    if arms:
        best["ESS"] = max(arms, key=lambda a: table[key_for(a)]["ESS"][0])
        for column in ERROR_COLUMNS + ("FLOP/es",):
            best[column] = min(arms, key=lambda a: table[key_for(a)][column][0])

    lines = []
    for row in LATEX_ROWS:
        if row is None:
            lines.append("        \\midrule")
            continue
        arm, label = row
        key = key_for(arm)
        entry = table.get(key)
        ess = (
            "/"
            if arm in ("reference", "floor")
            else (
                f"${entry['ESS'][0]:.3f} \\pm {entry['ESS'][1]:.3f}$" if entry else "--"
            )
        )
        if best.get("ESS") == arm:
            ess = bold(ess)
        cells = [ess]
        for column in ERROR_COLUMNS:
            value = cell(key, column)
            if best.get(column) == arm and value != "--":
                value = bold(value)
            cells.append(value)
        flops = "--" if arm == "floor" else cell(key, "FLOP/es", sci=True)
        if best.get("FLOP/es") == arm:
            flops = bold(flops)
        cells.append(flops)
        lines.append(f"        {label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "03_hard"
    )
    parser.add_argument(
        "--eval-subdir",
        default="eval_ema",
        help="eval_ema (default, the frozen convention) or eval",
    )
    parser.add_argument("--n-splits", type=int, default=64)
    parser.add_argument("--n-floor-replicates", type=int, default=200)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "03_hard" / "w2_24x24_house"
    )
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args(argv)

    from experiments.constrained_hard_03.run import build_target_and_head

    sigma_label = SIGMA_LABELS[0]
    chains, provenance = load_reference(
        REFERENCE_DIRS[sigma_label], sigma_label, lattice_side=L
    )
    reference = torch.cat(chains)

    probe_dirs = find_cells(
        args.results_dir, ARM_CONFIGS[next(iter(ARMS))][sigma_label], TAGS[sigma_label]
    )
    if not probe_dirs:
        sys.exit(f"no d576 run dirs under {args.results_dir}")
    target, _ = build_target_and_head(registry_config_for(probe_dirs[0]), device="cpu")
    chain_energies = [energy_per_site(target, c) for c in chains]
    reference_energy = torch.cat(chain_energies)

    table = {
        f"reference_{sigma_label}": {
            **reference_row(
                chains,
                chain_energies,
                reference_trial_counts(provenance, lattice_side=L),
            ),
            **{
                k: (v, 0.0)
                for k, v in reference_standard_error(
                    chains, L, args.n_splits, seed=0, chain_energies=chain_energies
                ).items()
            },
        }
    }
    for arm in ARMS:
        run_dirs = find_cells(
            args.results_dir, ARM_CONFIGS[arm][sigma_label], TAGS[sigma_label]
        )
        if not run_dirs:
            continue
        cfg = registry_config_for(run_dirs[0])
        _, head = build_target_and_head(flop_billing_config(cfg), device="cpu")
        per_forward = measured_forward_flops(
            head, (reference[:1], torch.full((1,), 0.5))
        )
        n_draws = cfg.eval.n_eval_samples
        rows = [
            neural_cell(
                d,
                target,
                reference,
                reference_energy,
                per_forward,
                cfg.ctmc.n_euler_steps,
                eval_subdir=args.eval_subdir,
                lattice_side=L,
            )
            for d in run_dirs
        ]
        cell = aggregate(rows)
        cell["per_forward_flops"] = per_forward
        cell["n_seeds"] = len(run_dirs)
        cell["per_seed_ess"] = [r["ESS"] for r in rows]
        table[f"{arm}_{sigma_label}"] = cell

        floor_key = f"floor{n_draws}_{sigma_label}"
        if floor_key not in table:
            table[floor_key] = {
                k: (v, 0.0)
                for k, v in sampling_floor_from_reference(
                    reference,
                    L,
                    n_draws,
                    args.n_floor_replicates,
                    seed=0,
                    reference_energy=reference_energy,
                ).items()
            }

    for gfn_arm in GFN_ARMS:
        from experiments.constrained_hard_03.run_gfn import build_target_and_policy

        name = GFN_CELL_NAME[sigma_label].format(objective=gfn_arm.removeprefix("gfn_"))
        run_dirs = [
            d
            for d in find_cells(args.results_dir, name, GFN_TAG[gfn_arm])
            if (d / args.eval_subdir / "metrics.json").is_file()
        ]
        if not run_dirs:
            print(f"no landed cells for {gfn_arm} at {sigma_label}", file=sys.stderr)
            continue
        gfn_cfg = gfn_registry_config_for(run_dirs[0])
        assert abs(gfn_cfg.sigma - target.sigma) < 1e-9, (
            f"{name}: trains at sigma={gfn_cfg.sigma} "
            f"against the reference's {target.sigma}"
        )
        _, policy = build_target_and_policy(gfn_cfg, "cpu")
        flops_per_raw = measured_forward_flops(
            policy.sample, (1,)
        ) + ising_energy_eval_flops(D_SITES)
        rows = [
            neural_cell(
                d,
                target,
                reference,
                reference_energy,
                per_forward=None,
                n_euler=None,
                eval_subdir=args.eval_subdir,
                lattice_side=L,
                flops_raw=flops_per_raw,
            )
            for d in run_dirs
        ]
        cell = aggregate(rows)
        cell["per_sample_flops"] = flops_per_raw
        cell["n_seeds"] = len(run_dirs)
        cell["per_seed_ess"] = [r["ESS"] for r in rows]
        table[f"{gfn_arm}_{sigma_label}"] = cell

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "house_table_24x24.json").write_text(json.dumps(table, indent=2))

    if args.latex:
        print(latex_table(table))
        return
    print(
        f"{'row':28} {'ESS':>14} {'dMag':>16} {'dCorr':>16} {'EW2':>16} {'FLOP/es':>14}"
    )
    for key, cell in table.items():
        ess = fmt(*cell["ESS"]) if "ESS" in cell else "/"
        flops = fmt(*cell["FLOP/es"], sci=True) if "FLOP/es" in cell else "--"
        print(
            f"{key:28} {ess:>14} {fmt(*cell['dMag']):>16} "
            f"{fmt(*cell['dCorr']):>16} {fmt(*cell['EW2']):>16} {flops:>14}"
        )


if __name__ == "__main__":
    main()
