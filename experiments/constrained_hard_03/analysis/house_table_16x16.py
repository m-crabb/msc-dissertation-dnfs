"""Fill pass for tab:eval-hard-16x16 (the house evaluation table, top rung).

Reads the d256 w3 wave (tags `20260826-d256-s010` at sigma = 0.1 and
`20260826-d256-sc` / `-sc-ma` at sigma_c: four heads x two couplings x seeds
42/43/44) and prints the same house columns the 8x8 fill prints one rung
down.

The lattice-generic machinery -- half-split standard error, sampling floor,
error metrics, and the FLOP provenance that reads each cell's own saved config
and asserts it against the live registry -- is imported from the 8x8 fill.
Defined locally: only what closes over this rung's `L` / `D_SITES`
(`energy_per_site`, `neural_cell`, `latex_table`) plus this rung's reference
loader.

Four differences from the 8x8 fill, each a way to print a wrong number:

  * The reference ships pooled, not per chain.
    `generate_kawasaki_reference_d256.py` concatenates the thinned chains into
    one tensor, so chain identity -- the half-split's unit of independence --
    is recovered by equal-width slicing. The blocks are equal because every
    chain records the same count and is thinned by the same interval.

  * The FLOP bill is in different units. The 8x8 npz records `n_trial_steps`,
    a count of swap proposals; the d256 provenance records sweeps, and one
    sweep is `N_SITES = 256` proposals
    (generate_kawasaki_reference_d256.py:137). Passing sweeps straight to
    `kawasaki_run_flops` would under-price the reference chain by 256x, so
    `chain_trial_counts` converts explicitly and is tested on both rungs'
    site counts.

  * The coupling is asserted, not assumed. `kawasaki_ref_d256_sc` is
    mislabelled: it sits at sigma = 0.22305 rather than the frozen sigma_c =
    0.220343, and it certifies cleanly against the legacy 0.588 anchor, so
    only its provenance reveals the problem. The directories read here are
    `kawasaki_ref_d256_s010` (sigma = 0.1, 136,536 draws) and
    `kawasaki_ref_d256_s220` (exact sigma_c, 31,512 draws, certified at
    nn-correlation 0.5790 +- 0.0003 against the mchammer anchor 0.578756).

  * Four heads and no rejection row: the exact-field lever is already answered
    at 4x4 and 8x8, and neither the unconstrained nor the soft chapter has a
    d256 case to reject off.

tau_int is measured with `integrated_autocorr` on each chain's stored
energy-per-site series and floored at 1.0, as in the 8x8 fill. The
provenance's tau in sweeps would have to be divided by the thinning interval
(twice the worst chain's tau) and would land below one.

Four sigma = 0.1 cells carry `cv_inversion_halt.json` and stopped at step 5000
of 50000; their evals read ESS fraction ~0.0009 against their relaunched
twins' 0.898, so they are dropped. The relaunches carry the `-r2` tag.

Per-site energy follows the chapter's convention E/d = -log p~(x) /
(2 sigma d). Neural cells aggregate mean +- SD over the three seeds.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
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

from discrete_flow_sampler.diagnostics.flops import (
    chain_per_effective_sample,
    ising_energy_eval_flops,
    kawasaki_run_flops,
    measured_forward_flops,
    neural_sampling_flops_per_sample,
    per_effective_sample,
)
from discrete_flow_sampler.diagnostics.metrics import (
    correlation_profile_error,
    energy_wasserstein2,
    integrated_autocorr,
    magnetisation_profile_error,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# The lattice-generic half of the 8x8 fill, imported rather than restated.


L = 16
D_SITES = L * L
SEEDS = (42, 43, 44)
SIGMA_LABELS = ("s010", "s220")
SIGMA = {"s010": 0.1, "s220": 0.22034339675488573}

ARMS = {
    "ma": "masked-attention band, one sweep",
    "thp": "two-hole patch head",
    "thp2": "two-hole patch head, $R=2$",
    # Raster-ordering ladder at this rung (tag 20260830-rasterord-d256). `masep`
    # runs at the floor only; separable is an exact rewrite of `ma`, so the
    # archived dense sigma_c failure stands as the anchor.
    "masep": "masked-attention band, one sweep (separable twin)",
    "mamo2": "masked-attention band, two sweeps",
    "mamo2ef": "masked-attention band, two sweeps + exact field",
    "iv": "prefix-sum band, one sweep",
    "ivmo2": "prefix-sum band, two sweeps",
    "ivmo2ef": "prefix-sum band, two sweeps + exact field",
}

# Row order of tab:eval-hard-16x16, flat like the 4x4 and 8x8 tables (the
# nested "two orderings / \quad + exact-field" pair was retired 2026-08-27).
# `None` is a \midrule. No rejection row at this rung -- see the docstring.
LATEX_ROWS = (
    ("reference", "Kawasaki (mchammer), certified reference"),
    ("floor", "sampling floor at $N=5000$"),
    None,
    ("ma", "masked-attention band, one sweep"),
    ("thp", "two-hole patch head"),
    ("thp2", "\\quad $R=2$"),
    None,
    # masep (the separable-trained twin) is not a row (decided 2026-09-03): one
    # masked-attention row per rung, billed separable; the twin's 2/3 collapsed
    # floor seeds are a caveat in the Experimental Setup, not a cell.
    ("mamo2", "masked-attention band, two sweeps"),
    ("mamo2ef", "\\quad + exact field"),
    ("iv", "prefix-sum band, one sweep"),
    ("ivmo2", "prefix-sum band, two sweeps"),
    ("ivmo2ef", "\\quad + exact field"),
    None,
    # Different sampling paradigm, so outside the bold comparison: `best` is
    # computed over ARMS, which the GFN arms are not in.
    ("gfn_tb", "GFlowNet, trajectory balance"),
    ("gfn_fldb", "GFlowNet, forward-looking DB"),
)
ERROR_COLUMNS = ("dMag", "dCorr", "EW2")

# GFlowNet comparator rows at the 256-step fairness rung. The bill is the
# KV-cached rollout plus one target eval for the IS weight (no Euler grid).
# Tags are per arm: the first wave's TB centres (tag 20260831-gfn-d256) stalled
# with log Z pinned at 100 by AdamW's default weight decay
# (run_gfn.build_optimiser) and are re-run under their own tag; the FL-DB loss
# carries no log Z, so those cells are kept.
GFN_ARMS = {
    "gfn_tb": "GFlowNet, trajectory balance",
    "gfn_fldb": "GFlowNet, forward-looking DB",
}
GFN_TAG = {"gfn_tb": "20260903-gfn-d256-tb", "gfn_fldb": "20260831-gfn-d256"}
GFN_CELL_NAME = {
    "s010": "GFN_d256_c50_s010_{objective}_50k_par",
    "s220": "GFN_d256_c50_s220_{objective}_100k_par",
}

# Matched on size, composition, coupling, head and wave, not on the budget or
# optimiser infixes: those differ legitimately across the wave (sigma_c cells
# train 100k with the widening curriculum, sigma = 0.1 floor cells 50k flat),
# so pinning them would silently drop cells rather than fail loudly. The
# coupling token excludes the legacy archive (`s223`) and `_w3_` excludes
# anything older at the same coupling.
CELL_GLOB = "H2_d{d}_c50_{sigma}_letf_{arm}_*_w3_seed*"

# Seeds excluded as degenerate, and daggered in the table where excluded.
# Distinct from the tripwire drops: this run trained its full 50k steps and
# still collapsed its weights. Its largest self-normalised weight is 0.1839
# (one draw of 5000 carrying 18% of the mass) where every other cell across the
# 50 landed cells at both rungs sits at or below 0.0053; its ESS fraction is
# 0.0032 against its siblings' 0.805 and 0.918.
DEGENERATE_SEEDS = {
    ("ma", "s010", 43),
    # ivmo2 seed 44 at sigma_c: raw 0.790 but EMA 0.001, a floor read on the EMA
    # table this row prints (decided 2026-09-03: every seed prints unless its
    # frozen ESS is at the floor).
    ("ivmo2", "s220", 44),
}


# --- the reference --------------------------------------------------------


def split_pooled_into_chains(pooled, n_chains):
    """Recover chain blocks from the single pooled reference tensor.

    Contiguous slices, never a stride: the generator concatenates whole chains
    in index order, so a strided split would put one snapshot of every chain
    into each block and make the half-split read within-chain correlation as
    between-chain, collapsing the reference's stated error toward zero. Any
    remainder is dropped rather than smeared.
    """
    block = len(pooled) // n_chains
    return [pooled[i * block : (i + 1) * block] for i in range(n_chains)]


def load_reference(directory, sigma_key):
    """Certified chains for one coupling, one tensor per chain.

    Asserts the pool's own recorded sigma against the column's before
    returning anything: `kawasaki_ref_d256_sc` is mislabelled at 0.22305
    and certifies cleanly against the legacy anchor, so the directory name
    is not evidence of the coupling.
    """
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    assert abs(provenance["sigma"] - SIGMA[sigma_key]) < 1e-9, (
        f"{directory.name} records sigma={provenance['sigma']}, not "
        f"{SIGMA[sigma_key]}: couplings must never be mixed in one column"
    )
    pooled = torch.load(directory / "samples.pt", weights_only=True).float()
    assert is_composition_exact(pooled, D_SITES // 2), (
        f"{directory.name}: reference left the c=0.5 slice"
    )
    return split_pooled_into_chains(pooled, provenance["n_chains"]), provenance


def chain_trial_counts(provenance, lattice_edge=L):
    """Swap proposals per chain, burn-in included.

    One sweep is `lattice_edge**2` proposals; the bill prices proposals, and
    burn-in is paid before the first usable record, so it belongs in the total
    (kawasaki_run_flops' convention). The site factor is derived, not
    hard-coded.
    """
    per_chain = (
        (provenance["burn_in_sweeps"] + provenance["sampling_sweeps_per_chain"])
        * lattice_edge
        * lattice_edge
    )
    return [per_chain] * provenance["n_chains"]


def reference_row(chains, chain_energies, trial_counts):
    """The certified chain as both reference and classical baseline.

    Error cells carry the reference's own standard error (scored against
    itself it would be identically zero, which would claim the chain is
    exact). tau_int is measured on the stored series and floored at 1.0.
    """
    taus = [max(1.0, integrated_autocorr(e.numpy())) for e in chain_energies]
    per_chain = [
        chain_per_effective_sample(kawasaki_run_flops(n_trials), chain.shape[0], tau)
        for chain, tau, n_trials in zip(chains, taus, trial_counts)
    ]
    return {
        "FLOP/es": (float(np.mean(per_chain)), float(np.std(per_chain))),
        "tau_int_snapshots": (float(np.mean(taus)), float(np.std(taus))),
        "n_chains": len(chains),
        "n_effective": float(sum(c.shape[0] for c in chains) / np.mean(taus)),
    }


# --- cells ----------------------------------------------------------------


def find_cells(results_dir, sigma_key, arm, eval_subdir="eval_ema"):
    """Healthy run dirs for one head at one coupling, seed order.

    Drops cells the cold-CV tripwire halted, and matches the head token between
    underscores so `thp` cannot sweep up `thp2` -- separate rows of the table.
    A head whose runs have not landed returns empty rather than raising, so a
    missing arm leaves a blank row instead of aborting the fill.
    """
    found = []
    for run_dir in sorted(
        Path(results_dir).glob(CELL_GLOB.format(d=D_SITES, sigma=sigma_key, arm=arm))
    ):
        # Right-anchors the token the glob already anchored on the left.
        if f"_{arm}_" not in run_dir.name:
            continue
        if (run_dir / "cv_inversion_halt.json").exists():
            print(
                f"dropped (cv-inversion tripwire halt): {run_dir.name}", file=sys.stderr
            )
            continue
        if not (run_dir / eval_subdir / "samples.pt").exists():
            # Still training (or a partial pull): not landed, not a cell.
            print(f"skipped (no {eval_subdir} yet): {run_dir.name}", file=sys.stderr)
            continue
        seed = int(run_dir.name.split("_seed")[1][:2])
        if (arm, sigma_key, seed) in DEGENERATE_SEEDS:
            print(f"dropped (degenerate seed): {run_dir.name}", file=sys.stderr)
            continue
        found.append(run_dir)
    return found


def has_excluded_seed(arm, sigma_key):
    """Whether this cell lost a seed to degeneracy, so the row is daggered."""
    return any(a == arm and s == sigma_key for a, s, _ in DEGENERATE_SEEDS)


def energy_per_site(target, states, chunk=4096):
    """Chunked: the sigma = 0.1 reference pool is 1.4e5 states at d=256."""
    parts = [
        -target.log_prob(states[i : i + chunk]) / (2 * target.sigma * D_SITES)
        for i in range(0, states.shape[0], chunk)
    ]
    return torch.cat(parts)


def neural_cell(
    run_dir, target, reference, reference_energy, flops_per_raw, eval_subdir="eval"
):
    """One seed's row. The per-raw-sample bill is the caller's: a swap cell
    pays per_forward x n_euler Euler forwards, a GFN cell one cached rollout
    plus a target eval."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / eval_subdir / "metrics.json").read_text())
    samples = torch.load(
        run_dir / eval_subdir / "samples.pt", weights_only=True
    ).float()
    log_w = torch.load(run_dir / eval_subdir / "log_weights.pt", weights_only=True)
    weights = torch.softmax(log_w, dim=0)
    ess = metrics["ess_fraction"]
    w_ref = torch.full((reference.shape[0],), 1.0 / reference.shape[0])
    return {
        "ESS": ess,
        "dMag": magnetisation_profile_error(
            samples, weights, reference, L, reference_weights=w_ref
        ),
        "dCorr": correlation_profile_error(
            samples, weights, reference, L, reference_weights=w_ref
        ),
        "EW2": energy_wasserstein2(
            energy_per_site(target, samples),
            weights,
            reference_energy,
            reference_weights=w_ref,
        ),
        "FLOP/es": per_effective_sample(flops_per_raw, ess),
    }


# --- LaTeX ----------------------------------------------------------------


def latex_table(table, n_draws=5000):
    """Emit the table body, bolding the best neural cell in each column.

    Same conventions as tab:eval-hard-8x8. Where the reference's own standard
    error is comparable to the spread across heads, a bolded error cell marks
    the smallest number measured, not a separation from the others; the caption
    says so.
    """

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
        if any(has_excluded_seed(arm, s) for s in SIGMA_LABELS):
            label += "$^{\\dagger}$"
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
            errors = []
            for column in ERROR_COLUMNS:
                text = cell(key, column)
                if best.get((sigma_label, column)) == arm and text != "--":
                    text = f"$\\mathbf{{{text.strip('$')}}}$"
                errors.append(text)
            cells += [ess] + errors + [flops]
        lines.append(f"        {label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


# --- driver ---------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "03_hard"
    )
    parser.add_argument("--reference-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument(
        "--eval-subdir", default="eval_ema", choices=("eval", "eval_ema")
    )
    parser.add_argument("--n-splits", type=int, default=64)
    parser.add_argument("--n-floor-replicates", type=int, default=200)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "03_hard" / "w3_16x16_house"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="emit the tab:eval-hard-16x16 body instead of the console summary",
    )
    args = parser.parse_args(argv)

    from experiments.constrained_hard_03.run import build_target_and_head

    table, n_draws = {}, 5000
    for sigma_label in SIGMA_LABELS:
        chains, provenance = load_reference(
            args.reference_dir / f"kawasaki_ref_d256_{sigma_label}", sigma_label
        )
        reference = torch.cat(chains)

        # One target per coupling, built from a landed cell's own config so the
        # coupling on the neural side provably matches the reference's.
        probe = next(
            (
                find_cells(args.results_dir, sigma_label, a, args.eval_subdir)
                for a in ARMS
                if find_cells(args.results_dir, sigma_label, a, args.eval_subdir)
            ),
            None,
        )
        if not probe:
            print(f"no landed cells at {sigma_label}; skipping", file=sys.stderr)
            continue
        target, _ = build_target_and_head(registry_config_for(probe[0]), device="cpu")
        assert abs(target.sigma - provenance["sigma"]) < 1e-9, (
            f"{sigma_label}: cells train at sigma={target.sigma} against a "
            f"reference at {provenance['sigma']}"
        )

        chain_energies = [energy_per_site(target, c) for c in chains]
        reference_energy = torch.cat(chain_energies)

        table[f"reference_{sigma_label}"] = {
            **reference_row(chains, chain_energies, chain_trial_counts(provenance, L)),
            **{
                k: (v, 0.0)
                for k, v in reference_standard_error(
                    chains, L, args.n_splits, seed=0, chain_energies=chain_energies
                ).items()
            },
        }

        for arm in ARMS:
            run_dirs = find_cells(args.results_dir, sigma_label, arm, args.eval_subdir)
            if not run_dirs:
                print(f"no cells for {arm} at {sigma_label}", file=sys.stderr)
                continue
            cfg = registry_config_for(run_dirs[0])
            # Billed separable for a masked-attention head; see
            # house_table_8x8.flop_billing_config.
            _, head = build_target_and_head(flop_billing_config(cfg), device="cpu")
            per_forward = measured_forward_flops(
                head, (reference[:1], torch.full((1,), 0.5))
            )
            n_draws = cfg.eval.n_eval_samples
            flops_per_raw = neural_sampling_flops_per_sample(
                per_forward, cfg.ctmc.n_euler_steps, D_SITES
            )
            rows = [
                neural_cell(
                    d,
                    target,
                    reference,
                    reference_energy,
                    flops_per_raw,
                    eval_subdir=args.eval_subdir,
                )
                for d in run_dirs
            ]
            cell = aggregate(rows)
            cell["per_forward_flops"] = per_forward
            cell["n_seeds"] = len(run_dirs)
            cell["cells"] = [d.name for d in run_dirs]
            table[f"{arm}_{sigma_label}"] = cell

        for gfn_arm in GFN_ARMS:
            from experiments.constrained_hard_03.run_gfn import build_target_and_policy

            name = GFN_CELL_NAME[sigma_label].format(
                objective=gfn_arm.removeprefix("gfn_")
            )
            run_dirs = [
                args.results_dir / f"{name}_seed{seed}_{GFN_TAG[gfn_arm]}"
                for seed in SEEDS
            ]
            if not all(
                (d / args.eval_subdir / "metrics.json").is_file() for d in run_dirs
            ):
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
                    flops_per_raw,
                    eval_subdir=args.eval_subdir,
                )
                for d in run_dirs
            ]
            cell = aggregate(rows)
            cell["per_sample_flops"] = flops_per_raw
            cell["n_seeds"] = len(run_dirs)
            cell["cells"] = [d.name for d in run_dirs]
            table[f"{gfn_arm}_{sigma_label}"] = cell

        table[f"floor{n_draws}_{sigma_label}"] = {
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

    if args.latex:
        print(latex_table(table, n_draws))
    else:
        for key in sorted(table):
            entry = table[key]
            summary = "  ".join(
                f"{c}={fmt(*entry[c], sci=(c == 'FLOP/es'))}"
                for c in ("ESS",) + ERROR_COLUMNS + ("FLOP/es",)
                if c in entry and isinstance(entry[c], tuple)
            )
            print(f"{key:28} {summary}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"table_{args.eval_subdir}.json").write_text(
        json.dumps(table, indent=2, default=str)
    )
    return table


if __name__ == "__main__":
    main()
