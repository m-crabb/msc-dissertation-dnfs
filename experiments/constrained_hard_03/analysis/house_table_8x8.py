"""Fill pass for tab:eval-hard-8x8 (the house evaluation table, 8x8 rung).

Reads the Wave-2 d64 matrix (tag 20260825-hard-w2-d64: five arms x two
sigma x seeds 42/43/44, plus the `_w2e` eager twins of the two factorised
arms at sigma_c) and prints the house columns of
tab:eval-unconstrained-10x10 for each cell, exactly as house_table_4x4.py
does one rung down.

WHAT CHANGES FROM THE 4x4 FILL, and why it is not a cosmetic port.

The 4x4 slice is exactly enumerable -- C(16,8) = 12,870 states -- so that
fill's reference IS the true conditional: its error cells are zero by
construction and only the sampler side needs a floor. Here the slice holds
C(64,32) ~ 1.8e18 states, enumeration is out, and the reference becomes the
certified mchammer Kawasaki chain pool. Three consequences:

  * THE REFERENCE HAS ITS OWN PRECISION. It cannot print zero without
    claiming the chain is exact. Its error cells carry a standard error
    instead, which is what the float's caption already promises.
  * THE REFERENCE AND THE CHAIN BASELINE ARE THE SAME ENGINE, so they are
    one row rather than the 4x4 fill's two. At 4x4 the "Kawasaki, run long"
    row was a separate object from the enumerated reference; here scoring
    the chain against itself would be identically zero, and the honest
    single row reports the reference's SE in the error columns and its own
    algorithmic bill in FLOP/es.
  * THE FLOOR IS ESTIMATED, NOT DRAWN FROM AN EXACT PMF. It resamples the
    reference pool at the neural cells' own N.

THE REFERENCE STANDARD ERROR: a HALF-SPLIT estimator, not a bootstrap.
Partition the chain pool into two disjoint halves, measure the error metric
BETWEEN the halves, halve it. Each half-size reference carries sqrt(2) times
the full pool's noise and the separation combines two of them, so the split
distance is ~2x the full pool's SE. The rejected alternative -- bootstrapping
the reference against itself -- answers "how much does this reference wobble
under resampling", which is not the quantity an error column needs; the
column needs "how far apart would two independent references land".

THE FLOOR'S SELF-REFERENCE, and why it is tolerable. The floor draws N from
the same pool it scores against, so draw and reference overlap. Measured on
the shipped chains the pool is 120,015 post-burn-in snapshots against
N = 5,000 draws -- a 4% overlap, an order below the 0.02 scale the table
quotes. Holding chains out would remove that 4% at the cost of a less
precise reference for every row, which is the worse trade; the 4x4 fill's
floor draws from the exact pmf for the same structural reason and the two
scripts stay readable side by side.

WHY THE POOL IS PRECISE ENOUGH TO BE A REFERENCE AT ALL. Snapshots are
thinned 100 trials apart, so the integrated autocorrelation of the
energy-per-site series is 1.08 snapshots at sigma = 0.1 and 2.81 at
sigma_c. The 15-chain pool therefore carries 110,741 and 42,686 effective
draws against the neural cells' N = 5,000 -- 22x and 8.5x -- and certifies
at Gelman-Rubin 1.0000 and 1.0001 on energy (split-half 1.0000 / 1.0004)
against the caption's claimed <= 1.01. Magnetisation R-hat is formally
infinite because composition is exact on the slice: all 300,030 shipped
snapshots carry net magnetisation identically zero, so within-chain
variance vanishes. That is the constraint holding, not a diagnostic
failure.

FLOP/es PROVENANCE. Each cell's architecture is read from its OWN saved
config.json and asserted equal to the live registry entry before the
forward is measured, so a cell trained before a lever landed can never be
billed at today's architecture. Audited 2026-08-27 across all 36 d64 cells
and all 36 printed 4x4 cells: zero drift.

Per-site energy follows the chapter's convention E/d = -log p~(x) /
(2 sigma d). Neural cells aggregate mean +- SD over the three seeds.
"""
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from discrete_flow_sampler.diagnostics.flops import (
    chain_per_effective_sample, kawasaki_run_flops, measured_forward_flops,
    neural_sampling_flops_per_sample, per_effective_sample)
from discrete_flow_sampler.diagnostics.metrics import (
    correlation_profile_error, energy_wasserstein2, integrated_autocorr,
    magnetisation_profile_error)

L = 8
D_SITES = L * L
TAG = "20260825-hard-w2-d64"
ARMS = {
    "mo": "mask-one head",
    "ma": "masked-attention head",
    "fimo2ef": "factorised head, prefix band + exact field",
    "fmo2ef": "factorised head, global interior + exact field",
    "thp": "two-hole patch head",
}
SIGMA_LABELS = ("s010", "s220")
SEEDS = (42, 43, 44)

# Config sigma labels (sigma*100, zero-padded) vs the kawasaki npz tags
# (round(sigma*1000)): 0.10 -> s010 vs s100; SIGMA_C -> s220 in both by the
# accident of round(220.343) = 220. Same mapping as the 4x4 fill.
KAWASAKI_TAG = {"s010": "s100", "s220": "s220"}

# The d64 config name differs by coupling: the floor cells train flat, the
# critical cells carry the sigma ladder (the `_curr` infix).
CELL_NAME = {
    "s010": "H2_d64_c50_s010_letf_{arm}_50k_w2",
    "s220": "H2_d64_c50_s220_letf_{arm}_50k_curr_w2",
}
# The two factorised arms at sigma_c print from their EAGER twins, the
# decision-(c) convention inherited from the 4x4 table. At d64 the compiled
# and eager twins agree to within the declared ~0.02 FP caveat (compiled
# 0.843/0.840 against eager 0.828/0.835), so this is convention-matching
# across rungs rather than a repair -- recorded because a reader comparing
# the two tables will ask whether the dagger means the same thing at both.
EAGER_TWIN = {("fimo2ef", "s220"), ("fmo2ef", "s220")}

FLOP_BEARING_FIELDS = ("head_kind", "gather_triu_pairs", "compile_head",
                       "compile_model", "use_sdpa_readout")


# --- reference ------------------------------------------------------------

def load_reference_chains(kawasaki_dir, lattice_edge, npz_tag,
                          burn_in_fraction=0.2):
    """Post-burn-in snapshots of each certified chain, one tensor per chain.

    Kept per chain rather than pre-pooled: the half-split standard error
    needs the chain as the unit of independence, since snapshots within a
    chain are not independent and snapshots across chains are.
    """
    chains = []
    for npz_path in sorted(kawasaki_dir.glob(
            f"kawasaki_D{lattice_edge}_{npz_tag}_seed*.npz")):
        spins = torch.from_numpy(np.load(npz_path)["spins"]).float()
        chains.append(spins[int(len(spins) * burn_in_fraction):])
    return chains


def chain_trial_counts(kawasaki_dir, lattice_edge, npz_tag):
    """Total trials per chain, for the algorithmic FLOP bill (burn-in
    included -- it is paid before the first usable record)."""
    return [int(np.load(p)["n_trial_steps"]) for p in sorted(
        kawasaki_dir.glob(f"kawasaki_D{lattice_edge}_{npz_tag}_seed*.npz"))]


def is_composition_exact(states, n_plus):
    """Every state sits on the c = n_plus/d slice. A reference that drifted
    off the slice would make every error column measure the composition gap
    rather than the structure the columns are meant to compare."""
    return bool(((states > 0).sum(dim=-1) == n_plus).all())


def _metrics_between(sampler, reference, lattice_edge,
                     sampler_energy=None, reference_energy=None):
    """The three error columns for one (sampler, reference) pair, both
    unweighted -- chain snapshots and floor draws carry no IS weights."""
    w_s = torch.full((sampler.shape[0],), 1.0 / sampler.shape[0])
    w_r = torch.full((reference.shape[0],), 1.0 / reference.shape[0])
    out = {
        "dMag": magnetisation_profile_error(
            sampler, w_s, reference, lattice_edge, reference_weights=w_r),
        "dCorr": correlation_profile_error(
            sampler, w_s, reference, lattice_edge, reference_weights=w_r),
    }
    if sampler_energy is not None and reference_energy is not None:
        out["EW2"] = energy_wasserstein2(
            sampler_energy, w_s, reference_energy, reference_weights=w_r)
    return out


def reference_standard_error(chains, lattice_edge, n_splits=64, seed=0,
                             chain_energies=None):
    """How far apart two independent references would land, halved.

    See the module docstring: the half-split distance carries ~2x the full
    pool's standard error, because each half is a sqrt(2)-noisier reference
    and the separation combines two of them.
    """
    generator = torch.Generator().manual_seed(seed)
    n_chains = len(chains)
    half = n_chains // 2
    replicates = []
    for _ in range(n_splits):
        order = torch.randperm(n_chains, generator=generator).tolist()
        left, right = order[:half], order[half:]
        pack = lambda idx, src: torch.cat([src[i] for i in idx])
        energies = (
            (pack(left, chain_energies), pack(right, chain_energies))
            if chain_energies is not None else (None, None)
        )
        replicates.append(_metrics_between(
            pack(left, chains), pack(right, chains), lattice_edge,
            *energies))
    return {k: float(np.mean([r[k] for r in replicates])) / 2.0
            for k in replicates[0]}


def sampling_floor_from_reference(reference, lattice_edge, n_draws,
                                  n_replicates=200, seed=0,
                                  reference_energy=None):
    """The error a PERFECT sampler still shows at the neural cells' own N.

    Draws are taken from the reference pool itself; at the shipped sizes
    they overlap it by ~4% (see the module docstring), which is an order
    below the scale the table quotes. A cell at or below this row is
    indistinguishable from the reference at its own draw count.
    """
    generator = torch.Generator().manual_seed(seed)
    replicates = []
    for _ in range(n_replicates):
        idx = torch.randint(0, reference.shape[0], (n_draws,),
                            generator=generator)
        draws = reference[idx]
        replicates.append(_metrics_between(
            draws, reference, lattice_edge,
            None if reference_energy is None else reference_energy[idx],
            reference_energy))
    return {k: float(np.mean([r[k] for r in replicates]))
            for k in replicates[0]}


# --- FLOP provenance (Tier 4.8) -------------------------------------------

def run_dir_config(run_dir):
    """The run's OWN saved config, defaults backfilled the way the eval-only
    route backfills them, so a config written before a field existed still
    compares against today's dataclass."""
    from experiments.constrained_hard_03.configs import HardStageCfg
    from experiments.constrained_hard_03.run import _backfill_missing_defaults

    saved = json.loads((Path(run_dir) / "config.json").read_text())
    _backfill_missing_defaults(saved, HardStageCfg)
    return saved


def _flatten(mapping, prefix=""):
    flat = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def config_drift(saved, live, fields=FLOP_BEARING_FIELDS):
    """Fields where the run's saved config disagrees with the live registry.

    Names the drifted fields rather than picking a side: a cell whose saved
    architecture differs from today's registry is a provenance finding, and
    silently billing either one would hide it.
    """
    flat_saved, flat_live = _flatten(saved), _flatten(live)
    return {
        key: (flat_saved.get(key), flat_live.get(key))
        for key in set(flat_saved) | set(flat_live)
        if key.split(".")[-1] in fields
        and flat_saved.get(key) != flat_live.get(key)
    }


def registry_config_for(run_dir):
    """The registry entry for a run, asserted against the run's own config.

    Returns the dataclass (needed to build the head for the FLOP forward)
    only when the two agree on every FLOP-bearing field; otherwise raises,
    so the table can never print a bill measured at the wrong architecture.
    """
    from experiments.constrained_hard_03.configs import CONFIGS

    saved = run_dir_config(run_dir)
    cfg = CONFIGS[saved["name"]]
    cfg = replace(cfg, head_kind=saved["head_kind"],
                  train=replace(cfg.train, seed=saved["train"]["seed"]))
    drift = config_drift(saved, json.loads(json.dumps(asdict(cfg))))
    if drift:
        raise ValueError(
            f"{Path(run_dir).name}: saved config disagrees with the registry "
            f"on FLOP-bearing fields {drift}"
        )
    return cfg


# --- cells ----------------------------------------------------------------

def energy_per_site(target, states, chunk=4096):
    """Chunked because the reference pool is ~1.2e5 states at d=64."""
    parts = [-target.log_prob(states[i:i + chunk]) / (2 * target.sigma * D_SITES)
             for i in range(0, states.shape[0], chunk)]
    return torch.cat(parts)


def neural_cell(run_dir, target, reference, reference_energy, per_forward,
                n_euler, eval_subdir="eval"):
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / eval_subdir / "metrics.json").read_text())
    samples = torch.load(run_dir / eval_subdir / "samples.pt",
                         weights_only=True).float()
    log_w = torch.load(run_dir / eval_subdir / "log_weights.pt",
                       weights_only=True)
    weights = torch.softmax(log_w, dim=0)
    ess = metrics["ess_fraction"]
    w_ref = torch.full((reference.shape[0],), 1.0 / reference.shape[0])
    flops_raw = neural_sampling_flops_per_sample(per_forward, n_euler, D_SITES)
    return {
        "ESS": ess,
        "dMag": magnetisation_profile_error(
            samples, weights, reference, L, reference_weights=w_ref),
        "dCorr": correlation_profile_error(
            samples, weights, reference, L, reference_weights=w_ref),
        "EW2": energy_wasserstein2(
            energy_per_site(target, samples), weights, reference_energy,
            reference_weights=w_ref),
        "FLOP/es": per_effective_sample(flops_raw, ess),
    }


def reference_row(chains, chain_energies, trial_counts):
    """The certified chain, as BOTH reference and classical baseline.

    Error cells are the reference's own standard error (the reference row
    scored against itself would be identically zero and would claim the
    chain is exact). FLOP/es is the algorithmic chain bill over ALL trials
    -- burn-in included, mirroring the other chain bills -- divided by the
    effective record count n_kept / tau_int, with tau_int measured per chain
    on the energy-per-site series.
    """
    taus = [max(1.0, integrated_autocorr(e.numpy())) for e in chain_energies]
    per_chain = [
        chain_per_effective_sample(kawasaki_run_flops(n_trials),
                                   chain.shape[0], tau)
        for chain, tau, n_trials in zip(chains, taus, trial_counts)
    ]
    return {
        "FLOP/es": (float(np.mean(per_chain)), float(np.std(per_chain))),
        "tau_int_snapshots": (float(np.mean(taus)), float(np.std(taus))),
        "n_chains": len(chains),
        "n_effective": float(sum(c.shape[0] for c in chains) / np.mean(taus)),
    }


def aggregate(rows):
    return {key: (float(np.mean([r[key] for r in rows])),
                  float(np.std([r[key] for r in rows])))
            for key in rows[0]}


def fmt(mean, sd, sci=False):
    return f"{mean:.2e}+-{sd:.1e}" if sci else f"{mean:.3f}+-{sd:.3f}"


# --- LaTeX ----------------------------------------------------------------

# Row order and labels of tab:eval-hard-8x8, matching the flat naming the
# printed 4x4 table uses. `None` is a \midrule.
LATEX_ROWS = (
    ("reference", "Kawasaki (mchammer), certified reference"),
    ("floor", "sampling floor at $N=5000$"),
    None,
    ("mo", "mask-one head"),
    ("ma", "masked-attention head"),
    ("fimo2ef", "factorised head, prefix band $+$ exact field"),
    ("fmo2ef", "factorised head, global interior $+$ exact field"),
    ("thp", "two-hole patch head"),
    None,
    ("reject_off_soft", "reject off soft \\gls{dnfs}"),
)
ERROR_COLUMNS = ("dMag", "dCorr", "EW2")


def _sci(value):
    exponent = int(np.floor(np.log10(value)))
    return f"${value / 10 ** exponent:.1f}\\times10^{{{exponent}}}$"


def latex_table(table, n_draws=5000):
    """Emit the table body, bolding the best neural cell in every column.

    Uniform with tab:eval-hard-4x4 (decided 2026-08-27). Note what the
    bold does and does not claim in the error columns: at sigma_c the
    reference's own standard error (2.7) is comparable to the whole spread
    across heads (5.1-6.6), so a bolded error cell marks the smallest number
    measured, NOT a separation from the others. The caption says so.
    """
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

    def key_for(arm, sigma_label):
        if arm == "reference":
            return f"reference_{sigma_label}"
        if arm == "floor":
            return f"floor{n_draws}_{sigma_label}"
        return f"{arm}_{sigma_label}"

    best = {}
    for sigma_label in SIGMA_LABELS:
        arms = [a for a, _ in (r for r in LATEX_ROWS if r) if a in ARMS
                and key_for(a, sigma_label) in table]
        if not arms:
            continue
        best[(sigma_label, "ESS")] = max(
            arms, key=lambda a: table[key_for(a, sigma_label)]["ESS"][0])
        for column in ERROR_COLUMNS + ("FLOP/es",):
            best[(sigma_label, column)] = min(
                arms, key=lambda a: table[key_for(a, sigma_label)][column][0])

    lines = []
    for row in LATEX_ROWS:
        if row is None:
            lines.append("        \\midrule")
            continue
        arm, label = row
        if arm in EAGER_TWIN_ARMS:
            label += "$^{\\dagger}$"
        cells = []
        for sigma_label in SIGMA_LABELS:
            key = key_for(arm, sigma_label)
            entry = table.get(key)
            ess = "/" if arm in ("reference", "floor") else (
                f"${entry['ESS'][0]:.3f} \\pm {entry['ESS'][1]:.3f}$"
                if entry else "--")
            flops = "--" if arm == "floor" else cell(key, "FLOP/es", sci=True)
            if best.get((sigma_label, "ESS")) == arm:
                ess = f"$\\mathbf{{{ess.strip('$')}}}$"
            if best.get((sigma_label, "FLOP/es")) == arm:
                flops = f"$\\mathbf{{{flops.strip('$')}}}$"
            errors = []
            for column in ERROR_COLUMNS:
                value = cell(key, column)
                if best.get((sigma_label, column)) == arm:
                    value = f"$\\mathbf{{{value.strip('$')}}}$"
                errors.append(value)
            cells += [ess] + errors + [flops]
        lines.append(f"        {label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


# Arms whose sigma_c cells print from their eager twins (dagger in the
# caption). Derived from EAGER_TWIN so the two can never disagree.
EAGER_TWIN_ARMS = {arm for arm, _ in EAGER_TWIN}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=REPO_ROOT / "results" / "03_hard")
    parser.add_argument("--kawasaki-dir", type=Path,
                        default=REPO_ROOT / "results" / "03_hard" / "kawasaki_w2")
    parser.add_argument("--burn-in-fraction", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=64)
    parser.add_argument("--n-floor-replicates", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results" / "03_hard" / "w2_8x8_house")
    parser.add_argument("--latex", action="store_true",
                        help="emit the tab:eval-hard-8x8 body instead of the "
                             "console summary")
    args = parser.parse_args(argv)

    from experiments.constrained_hard_03.run import build_target_and_head

    table = {}
    for sigma_label in SIGMA_LABELS:
        chains = load_reference_chains(args.kawasaki_dir, L,
                                       KAWASAKI_TAG[sigma_label],
                                       args.burn_in_fraction)
        if not chains:
            print(f"no reference chains for {sigma_label}; skipping")
            continue

        # One target per coupling, built from an arm's own saved config.
        probe_dir = (args.results_dir /
                     f"{CELL_NAME[sigma_label].format(arm='mo')}_seed42_{TAG}")
        target, _ = build_target_and_head(registry_config_for(probe_dir),
                                          device="cpu")

        chain_energies = [energy_per_site(target, c) for c in chains]
        reference = torch.cat(chains)
        reference_energy = torch.cat(chain_energies)
        assert is_composition_exact(reference, D_SITES // 2), \
            f"{sigma_label}: reference left the c=0.5 slice"

        table[f"reference_{sigma_label}"] = {
            **reference_row(chains, chain_energies,
                            chain_trial_counts(args.kawasaki_dir, L,
                                               KAWASAKI_TAG[sigma_label])),
            **{k: (v, 0.0) for k, v in reference_standard_error(
                chains, L, args.n_splits, seed=0,
                chain_energies=chain_energies).items()},
        }

        for arm in ARMS:
            suffix = "e" if (arm, sigma_label) in EAGER_TWIN else ""
            name = CELL_NAME[sigma_label].format(arm=arm) + suffix
            run_dirs = [args.results_dir / f"{name}_seed{seed}_{TAG}"
                        for seed in SEEDS]
            cfg = registry_config_for(run_dirs[0])
            _, head = build_target_and_head(cfg, device="cpu")
            per_forward = measured_forward_flops(
                head, (reference[:1], torch.full((1,), 0.5)))
            n_draws = cfg.eval.n_eval_samples

            for subdir in ("eval", "eval_ema"):
                rows = [neural_cell(d, target, reference, reference_energy,
                                    per_forward, cfg.ctmc.n_euler_steps,
                                    eval_subdir=subdir)
                        for d in run_dirs]
                cell = aggregate(rows)
                cell["per_forward_flops"] = per_forward
                cell["eager_twin"] = suffix == "e"
                key = f"{arm}_{sigma_label}" + ("_ema" if subdir == "eval_ema" else "")
                table[key] = cell

            floor_key = f"floor{n_draws}_{sigma_label}"
            if floor_key not in table:
                table[floor_key] = {
                    k: (v, 0.0) for k, v in sampling_floor_from_reference(
                        reference, L, n_draws, args.n_floor_replicates,
                        seed=0, reference_energy=reference_energy).items()}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "house_table_8x8.json").write_text(json.dumps(table, indent=2))

    if args.latex:
        print(latex_table(table))
        return

    print(f"{'row':34} {'ESS':>14} {'dMag':>16} {'dCorr':>16} "
          f"{'EW2':>16} {'FLOP/es':>14}")
    for key, cell in table.items():
        ess = fmt(*cell["ESS"]) if "ESS" in cell else "/"
        flops = fmt(*cell["FLOP/es"], sci=True) if "FLOP/es" in cell else "--"
        print(f"{key:34} {ess:>14} {fmt(*cell['dMag']):>16} "
              f"{fmt(*cell['dCorr']):>16} {fmt(*cell['EW2']):>16} {flops:>14}")


if __name__ == "__main__":
    main()
