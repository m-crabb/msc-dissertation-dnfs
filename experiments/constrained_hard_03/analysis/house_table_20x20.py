"""Fill pass for the 20x20 rung of the house evaluation table.

A thin table by design: the d400 wave (tag 20260827-d400-s010) is four cells --
two patch radii x two eval precisions -- at two seeds and one coupling. It is a
scaling probe, not a head survey, and the table should read as one.

Differences from every rung below:

  * Two couplings since 2026-08-30 (single-coupling before that: an empty
    sigma_c half would have read as "not yet landed" rather than "never run").
    The sigma_c wave (tag 20260829-d400-sc, 12 cells, 100k steps, curriculum
    ending at exact SIGMA_C from step 30k) landed with its own certified
    reference. The two waves carry different config names (50k flat vs
    100k_curr) and different tags, so cells are pinned per (row, coupling) in
    ARM_CONFIGS rather than globbed from one template.
  * Seed counts are mixed and recorded per cell (`n_seeds`): s010 thp2 rows
    carry three seeds since 2026-08-30 (DoC 280229/280230), s010 thp3 rows two,
    every sigma_c cell three. No claim should rest on a two-seed spread.
  * The arms are a radius x training-precision grid, not different heads. Every
    cell is the two-hole patch head, and `w4` vs `w4bf16` is
    `train.train_autocast_bf16` and nothing else. It is a training lever, not an
    evaluation one: `eval.eval_autocast_bf16` is True on all four cells, so
    every row here is evaluated identically. Labelling the pair "bf16
    evaluation" would imply the w4 rows evaluate in fp32, which they do not.

The references, one per coupling, same generator and sweep budget (8 chains,
100k burn-in + 102,400 sampling sweeps, thinned at 2x the worst chain's tau,
every draw at exactly 200 up-spins):

  * `kawasaki_ref_d400_s010` (2026-08-29, 30.4 s): tau 2.10 sweeps at D = 20
    against 2.11 at D = 16 -- no size penalty, because the tau ~ D^1.5 growth
    in fig:kawasaki-slowing is a critical phenomenon and sigma = 0.1 is far
    from it. 136,536 stored draws, Gelman-Rubin 0.999985.
  * `kawasaki_ref_d400_sc` (2026-08-30, ~5 min): tau 18.9 sweeps -- right on
    the D^1.5 prediction from the d256 sc pool's 13.8 -- still swallowed whole
    by the fixed 100k burn-in. 21,560 stored draws (the tau enters through
    thinning, not wall clock), Gelman-Rubin 0.99993, sigma recorded at exact
    SIGMA_C (the d256 sc pool is on record mislabelled at 0.22305).

Neither coupling has an external anchor, and both certifications say so: the
mchammer nn anchor is a property of (sigma_c, d256) jointly
(`external_nn_anchor` gates on both since 2026-08-30), so the s010 pool sits off
the anchor's coupling and the sc pool off its lattice. Certification therefore
rests on the internal checks -- Gelman-Rubin, start-condition agreement, the
energy-convention gap and the exact-composition assertion.

GFN rows (added 2026-09-07, tag 20260904-gfn-d400-tb). The 16x16 GFN block
carried up: TB only, hidden 72 = the parity re-size at this lattice (155,522
params vs thp2 158,848 / thp3 159,616; hidden 68 carried up would sit 11.6%
under both). FL-DB was dead at 256-step trajectories and was never launched at
400, so its row stays "--" with the 16x16 verdict as the footnote. The bill is
one KV-cached rollout plus one target eval for the IS weight, not an Euler grid,
so the GFN rows hand `neural_cell` their raw-sample price directly.

FLOP/es. The reference's algorithmic bill is derived from this rung's lattice
(`reference_trial_counts`), never from the d256 default that
`house_table_16x16.chain_trial_counts` carries -- taking that default would
under-bill by (16/20)^2 = 0.64. Neural cells are billed from their own saved
config rebuilt on today's code, masked-attention heads at the separable band,
exactly as the rungs below.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Shared reference statistics and FLOP provenance.
from experiments.constrained_hard_03.analysis.house_table_8x8 import (
    _sci,
    aggregate,
    flop_billing_config,
    fmt,
    gfn_registry_config_for,
    is_composition_exact,
    reference_standard_error,
    registry_config_for,
    sampling_floor_from_reference,
)
from experiments.constrained_hard_03.analysis.house_table_16x16 import (
    reference_row,
    split_pooled_into_chains,
)

from discrete_flow_sampler.diagnostics.flops import (
    ising_energy_eval_flops,
    measured_forward_flops,
    neural_sampling_flops_per_sample,
    per_effective_sample,
)
from discrete_flow_sampler.diagnostics.metrics import (
    correlation_profile_error,
    energy_wasserstein2,
    magnetisation_profile_error,
)
from discrete_flow_sampler.targets.ising import SIGMA_C

L = 20
D_SITES = L * L
SIGMA_LABELS = ("s010", "s220")
SIGMA = {"s010": 0.1, "s220": SIGMA_C}
TAGS = {"s010": "20260827-d400-s010", "s220": "20260829-d400-sc"}
REFERENCE_DIRS = {
    "s010": REPO_ROOT / "results" / "kawasaki_ref_d400_s010",
    "s220": REPO_ROOT / "results" / "kawasaki_ref_d400_sc",
}

# ARM_CONFIGS pins each (row, coupling) to its registry name: the campaigns
# use different recipes, 50k flat at s010 and 100k_curr at s220.
ARMS = {
    "thp2_w4": "two-hole patch head, $R=2$",
    "thp3_w4": "two-hole patch head, $R=3$",
    "thp2_w4bf16": "\\quad $R=2$, bf16 training",
    "thp3_w4bf16": "\\quad $R=3$, bf16 training",
}
ARM_CONFIGS = {
    f"{radius}_{precision}": {
        "s010": f"H2_d400_c50_s010_letf_{radius}_50k_b512_ne128_cv2_{precision}",
        "s220": (
            f"H2_d400_c50_s220_letf_{radius}_100k_curr_b512_ne128_cv2_{precision}"
        ),
    }
    for radius in ("thp2", "thp3")
    for precision in ("w4", "w4bf16")
}

# GFlowNet comparator rows, outside the bold comparison (`best` runs over
# ARMS, which these are not in). Tags are per arm as at 16x16; the fldb cell
# is registered but was never launched at this size.
GFN_ARMS = {
    "gfn_tb": "GFlowNet, trajectory balance",
    "gfn_fldb": "GFlowNet, forward-looking DB",
}
GFN_TAG = {"gfn_tb": "20260904-gfn-d400-tb", "gfn_fldb": "20260904-gfn-d400-tb"}
GFN_CELL_NAME = {
    "s010": "GFN_d400_c50_s010_{objective}_50k_par",
    "s220": "GFN_d400_c50_s220_{objective}_100k_par",
}

# The 16x16/20x20/24x24 pools come from the thesis's own swap generator; only
# the d256 sigma_c pool carries the mchammer anchor, so the label says so.
LATEX_ROWS = (
    ("reference", "Kawasaki (thesis engine), certified reference"),
    ("floor", "sampling floor at $N=5000$"),
    None,
    *((arm, label) for arm, label in ARMS.items()),
    None,
    *((arm, label) for arm, label in GFN_ARMS.items()),
)
ERROR_COLUMNS = ("dMag", "dCorr", "EW2")


def reference_trial_counts(provenance, lattice_side=L):
    """Swap proposals per chain at this rung's lattice, burn-in included.

    Not `house_table_16x16.chain_trial_counts(provenance)`: that function's
    `lattice_edge` defaults to 16, and taking the default here would under-bill
    the reference chain by (16/20)^2 = 0.64 with nothing in the output to show
    it. Pinned by test_house_table_20x20.
    """
    per_chain = (
        provenance["burn_in_sweeps"] + provenance["sampling_sweeps_per_chain"]
    ) * lattice_side**2
    return [per_chain] * provenance["n_chains"]


def load_reference(directory, sigma_key, lattice_side=L):
    """Certified chains for the one coupling, one tensor per chain.

    Asserts the pool's recorded lattice as well as its sigma: at this rung a
    d256 pool would otherwise load, split and score without complaint while
    measuring a different system.
    """
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["lattice_side"] == lattice_side, (
        f"{directory.name} records D={provenance['lattice_side']}, not {lattice_side}"
    )
    assert abs(provenance["sigma"] - SIGMA[sigma_key]) < 1e-9, (
        f"{directory.name} records sigma={provenance['sigma']}, not "
        f"{SIGMA[sigma_key]}: couplings must never be mixed in one column"
    )
    pooled = torch.load(directory / "samples.pt", weights_only=True).float()
    assert is_composition_exact(pooled, lattice_side**2 // 2), (
        f"{directory.name}: reference left the c=0.5 slice"
    )
    return split_pooled_into_chains(pooled, provenance["n_chains"]), provenance


def energy_per_site(target, states, chunk=4096):
    """Chunked: the pool is 1.4e5 states at d=400. Site count read off the
    states so the 24x24 fill can share this."""
    n_sites = states.shape[1]
    parts = [
        -target.log_prob(states[i : i + chunk]) / (2 * target.sigma * n_sites)
        for i in range(0, states.shape[0], chunk)
    ]
    return torch.cat(parts)


def find_cells(results_dir, config_name, tag):
    """Run dirs for one (cell, coupling), seed order; empty rather than
    raising. The config name pins the cell and the tag pins the campaign, so
    the glob needs no head-token anchoring of the kind the rungs below need to
    stop `thp` matching `thp2`.
    """
    found = []
    for run_dir in sorted(Path(results_dir).glob(f"{config_name}_seed*_{tag}")):
        if (run_dir / "cv_inversion_halt.json").exists():
            print(
                f"dropped (cv-inversion tripwire halt): {run_dir.name}", file=sys.stderr
            )
            continue
        found.append(run_dir)
    return found


def neural_cell(
    run_dir,
    target,
    reference,
    reference_energy,
    per_forward,
    n_euler,
    eval_subdir="eval",
    lattice_side=L,
    flops_raw=None,
):
    """`lattice_side` reshapes the flat states for the profile errors and
    sets the per-sample FLOP bill; the 24x24 fill passes 24 (a d576 state
    reshaped as 20x20 raises, so a forgotten argument fails loudly).
    `flops_raw` replaces the Euler-grid bill for rows that run no grid: a
    GFN row pays one cached rollout plus a target eval, which
    per_forward x n_euler cannot express, so the caller prices it."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / eval_subdir / "metrics.json").read_text())
    samples = torch.load(
        run_dir / eval_subdir / "samples.pt", weights_only=True
    ).float()
    log_w = torch.load(run_dir / eval_subdir / "log_weights.pt", weights_only=True)
    weights = torch.softmax(log_w, dim=0)
    ess = metrics["ess_fraction"]
    w_ref = torch.full((reference.shape[0],), 1.0 / reference.shape[0])
    if flops_raw is None:
        flops_raw = neural_sampling_flops_per_sample(
            per_forward, n_euler, lattice_side**2
        )
    return {
        "ESS": ess,
        "dMag": magnetisation_profile_error(
            samples, weights, reference, lattice_side, reference_weights=w_ref
        ),
        "dCorr": correlation_profile_error(
            samples, weights, reference, lattice_side, reference_weights=w_ref
        ),
        "EW2": energy_wasserstein2(
            energy_per_site(target, samples),
            weights,
            reference_energy,
            reference_weights=w_ref,
        ),
        "FLOP/es": per_effective_sample(flops_raw, ess),
    }


def latex_table(table, n_draws=5000):
    """Two-coupling body, same conventions as tab:eval-hard-16x16. Where the
    error columns sit at the sampling floor (the whole s010 half) a bolded cell
    marks the smallest number measured, not a separation; the caption says so."""

    def key_for(arm, sigma_label):
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

    best = {}
    for sigma_label in SIGMA_LABELS:
        arms = [
            a
            for a, _ in (r for r in LATEX_ROWS if r)
            if a in ARMS and key_for(a, sigma_label) in table
        ]
        if not arms:
            continue
        best[(sigma_label, "ESS")] = max(
            arms, key=lambda a: table[key_for(a, sigma_label)]["ESS"][0]
        )
        for column in ERROR_COLUMNS + ("FLOP/es",):
            best[(sigma_label, column)] = min(
                arms, key=lambda a: table[key_for(a, sigma_label)][column][0]
            )

    lines = []
    for row in LATEX_ROWS:
        if row is None:
            lines.append("        \\midrule")
            continue
        arm, label = row
        cells = []
        for sigma_label in SIGMA_LABELS:
            key = key_for(arm, sigma_label)
            entry = table.get(key)
            ess = (
                "/"
                if arm in ("reference", "floor")
                else (
                    f"${entry['ESS'][0]:.3f} \\pm {entry['ESS'][1]:.3f}$"
                    if entry
                    else "--"
                )
            )
            flops = "--" if arm == "floor" else cell(key, "FLOP/es", sci=True)
            if best.get((sigma_label, "ESS")) == arm:
                ess = f"$\\mathbf{{{ess.strip('$')}}}$"
            if best.get((sigma_label, "FLOP/es")) == arm:
                flops = f"$\\mathbf{{{flops.strip('$')}}}$"
            cells.append(ess)
            for column in ERROR_COLUMNS:
                value = cell(key, column)
                if best.get((sigma_label, column)) == arm and value != "--":
                    value = f"$\\mathbf{{{value.strip('$')}}}$"
                cells.append(value)
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
        help="eval_ema (default, the frozen convention at the rungs above) or eval",
    )
    parser.add_argument("--n-splits", type=int, default=64)
    parser.add_argument("--n-floor-replicates", type=int, default=200)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "03_hard" / "w2_20x20_house"
    )
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args(argv)

    from experiments.constrained_hard_03.run import build_target_and_head

    table = {}
    for sigma_label in SIGMA_LABELS:
        chains, provenance = load_reference(REFERENCE_DIRS[sigma_label], sigma_label)
        reference = torch.cat(chains)

        probe = ARM_CONFIGS[next(iter(ARMS))][sigma_label]
        probe_dirs = find_cells(args.results_dir, probe, TAGS[sigma_label])
        if not probe_dirs:
            print(f"no run dirs under {args.results_dir} for {probe}", file=sys.stderr)
            continue
        target, _ = build_target_and_head(
            registry_config_for(probe_dirs[0]), device="cpu"
        )

        chain_energies = [energy_per_site(target, c) for c in chains]
        reference_energy = torch.cat(chain_energies)

        table[f"reference_{sigma_label}"] = {
            **reference_row(chains, chain_energies, reference_trial_counts(provenance)),
            **{
                k: (v, 0.0)
                for k, v in reference_standard_error(
                    chains, L, args.n_splits, seed=0, chain_energies=chain_energies
                ).items()
            },
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
                )
                for d in run_dirs
            ]
            cell = aggregate(rows)
            cell["per_forward_flops"] = per_forward
            cell["n_seeds"] = len(run_dirs)
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

            name = GFN_CELL_NAME[sigma_label].format(
                objective=gfn_arm.removeprefix("gfn_")
            )
            run_dirs = [
                d
                for d in find_cells(args.results_dir, name, GFN_TAG[gfn_arm])
                if (d / args.eval_subdir / "metrics.json").is_file()
            ]
            if not run_dirs:
                print(
                    f"no landed cells for {gfn_arm} at {sigma_label}", file=sys.stderr
                )
                continue
            gfn_cfg = gfn_registry_config_for(run_dirs[0])
            assert abs(gfn_cfg.sigma - target.sigma) < 1e-9, (
                f"{name}: trains at sigma={gfn_cfg.sigma} against the "
                f"reference's {target.sigma}"
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
                    flops_raw=flops_per_raw,
                )
                for d in run_dirs
            ]
            cell = aggregate(rows)
            cell["per_sample_flops"] = flops_per_raw
            cell["n_seeds"] = len(run_dirs)
            cell["cells"] = [d.name for d in run_dirs]
            table[f"{gfn_arm}_{sigma_label}"] = cell

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "house_table_20x20.json").write_text(json.dumps(table, indent=2))

    if args.latex:
        print(latex_table(table))
        return

    print(
        f"{'row':56} {'ESS':>14} {'dMag':>16} {'dCorr':>16} {'EW2':>16} {'FLOP/es':>14}"
    )
    for key, cell in table.items():
        ess = fmt(*cell["ESS"]) if "ESS" in cell else "/"
        flops = fmt(*cell["FLOP/es"], sci=True) if "FLOP/es" in cell else "--"
        print(
            f"{key:56} {ess:>14} {fmt(*cell['dMag']):>16} "
            f"{fmt(*cell['dCorr']):>16} {fmt(*cell['EW2']):>16} {flops:>14}"
        )


if __name__ == "__main__":
    main()
