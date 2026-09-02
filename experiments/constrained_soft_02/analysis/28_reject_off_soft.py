"""Rejecting off the soft sampler at every window the soft chapter trains.

The soft chapter's closing section asks: given the trained soft sampler at a
window, what does it cost to recover the EXACT fixed-composition ensemble by
keeping only the draws on the slice c(x) = c_target?  The hard chapter's
`rejection_rows.py` prices this at c = 0.5 only, against a Kawasaki reference,
for its own tables.  This scorer prices it at every soft window and both
couplings, at both soft sizes (4x4 exact-enumeration rung, 8x8 production
rung), and reports the three numbers the argument needs and nothing else:

  * acceptance       -- fraction of draws on the slice (measured, all seeds),
  * ESS on the slice -- self-normalised ESS of the SURVIVORS' original weights,
                        which is an ESS against the hard target because the
                        penalty is identically zero on the slice (see the
                        docstring of rejection_rows.py for the derivation),
  * FLOP/es          -- the sampler's per-draw bill divided by BOTH the
                        acceptance and the ESS, so a rejected draw is charged.

Beside the measured acceptance sits the ANALYTIC prediction from the soft
target's composition envelope: grouped by composition the penalty multiplies
the target by exp(-lambda d (c - c_t)^2), and on the composition grid
c = k/d the mass on the exact slice under the envelope alone is

    p_slice = 1 / sum_k exp(-lambda d (k/d - c_t)^2)        (k over 0..d)

which is 0.92 at d=16 and 0.50 at d=64 for lambda=50 -- the 4x4 "near-free"
overhead of 1.09x becomes 2x at the production size, and continues as
sqrt(pi d / (2 lambda)) for large d.  The envelope drops the canonical density
of states Z_can(c), which tilts the true marginal toward half filling, so the
measured acceptance at an off-centre window is expected to sit BELOW the
envelope prediction; the gap is the entropic tilt, not sampler error.

The soft sampler priced at each window is the chapter's delivered recipe (the
F(c) convention): the matched-base specialist off centre, the house specialist
at the centre, both couplings, seeds 42-45, raw final weights.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "constrained_hard_03" / "analysis"))

from rejection_rows import SEEDS, kept_draws  # noqa: E402
from discrete_flow_sampler.diagnostics.flops import (  # noqa: E402
    measured_forward_flops, neural_sampling_flops_per_sample,
    per_effective_sample)

RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
LAMBDA = 50.0
WINDOWS = (0.25, 0.375, 0.5)
COUPLINGS = ("s010", "s220")

# (n_sites, coupling, c_target) -> run-dir glob.  Off-centre 8x8 windows are
# the matched-base cells; the centre has no matched-base twin (Bernoulli(1/2)
# is already matched there), so it prints the house specialist.
def run_glob(n_sites, coupling, c_target):
    sc = "_sc" if coupling == "s220" else ""
    c_tag = f"c{int(round(c_target * 1000)):04d}"
    if n_sites == 16:
        return f"S2_d4_{c_tag}_10k_l50_letf_house{sc}_seed{{seed}}_20260902-softhouse-d16-10k"
    if c_target == 0.5:
        return f"S2_d8_{c_tag}_l50_letf_ne128_house{sc}_seed{{seed}}_20260831-softhouse-d64"
    return f"S2_d8_{c_tag}_l50_letf_ne128_house_mb{sc}_seed{{seed}}_20260831-softmb-d64"


def envelope_slice_mass(n_sites, c_target, penalty=LAMBDA):
    """Mass of the exact slice under the penalty envelope on the c = k/d grid."""
    compositions = np.arange(n_sites + 1) / n_sites
    weights = np.exp(-penalty * n_sites * (compositions - c_target) ** 2)
    return float(weights[np.isclose(compositions, c_target)].sum() / weights.sum())


def score_cell(n_sites, coupling, c_target):
    from experiments.dnfs_baseline_01.run import _build_model, _rebuild_from_run_dir

    n_plus = int(round(c_target * n_sites))
    per_seed, run_dir = [], None
    for seed in SEEDS:
        matches = sorted(RESULTS.glob(run_glob(n_sites, coupling, c_target).format(seed=seed)))
        if not matches:
            continue
        run_dir = matches[0]
        samples, log_w, drawn = kept_draws(run_dir, n_plus)
        survivors = torch.softmax(log_w, dim=0)
        per_seed.append({
            "acceptance": samples.shape[0] / drawn,
            "ESS": float(1.0 / (survivors.pow(2).sum() * samples.shape[0])),
            "n_kept": int(samples.shape[0]),
            "n_drawn": drawn,
        })
    if run_dir is None:
        return None

    cfg, target, _device = _rebuild_from_run_dir(run_dir)
    model = _build_model(cfg, target)
    probe = torch.ones(1, n_sites)
    per_forward = measured_forward_flops(model, (probe, torch.full((1,), 0.5)))
    raw = neural_sampling_flops_per_sample(per_forward, cfg.ctmc.n_euler_steps, n_sites)
    for row in per_seed:
        row["FLOP/es"] = per_effective_sample(raw / row["acceptance"], row["ESS"])

    summary = {key: (float(np.mean([r[key] for r in per_seed])),
                     float(np.std([r[key] for r in per_seed])))
               for key in ("acceptance", "ESS", "FLOP/es")}
    summary["n_kept_per_seed"] = [r["n_kept"] for r in per_seed]
    summary["n_drawn_per_seed"] = [r["n_drawn"] for r in per_seed]
    summary["envelope_prediction"] = envelope_slice_mass(n_sites, c_target)
    summary["per_forward_flops"] = per_forward
    summary["n_euler_steps"] = cfg.ctmc.n_euler_steps
    summary["run_glob"] = run_glob(n_sites, coupling, c_target)
    return summary


def flop_cell(value):
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def latex_rows(table, n_sites):
    """One row per window: measured acceptance, envelope prediction, ESS, FLOP/es, both couplings."""
    lines = []
    for c_target in WINDOWS:
        cells = []
        for coupling in COUPLINGS:
            cell = table.get(f"soft_{n_sites}_{coupling}_c{c_target}")
            if cell is None:
                cells += ["--"] * 3
                continue
            acc, acc_sd = cell["acceptance"]
            ess, ess_sd = cell["ESS"]
            cells += [f"${acc:.2f} \\pm {acc_sd:.2f}$",
                      f"${ess:.2f} \\pm {ess_sd:.2f}$",
                      flop_cell(cell["FLOP/es"][0])]
        prediction = envelope_slice_mass(n_sites, c_target)
        lines.append(f"        $c_\\text{{target}} = {c_target}$ & ${prediction:.2f}$ & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "reject_off_soft.json")
    parser.add_argument("--latex", type=int, choices=(16, 64))
    args = parser.parse_args(argv)

    table = {}
    for n_sites in (16, 64):
        for coupling in COUPLINGS:
            for c_target in WINDOWS:
                cell = score_cell(n_sites, coupling, c_target)
                if cell is not None:
                    table[f"soft_{n_sites}_{coupling}_c{c_target}"] = cell
                    print(f"{n_sites:3d} {coupling} c={c_target:<5} acc {cell['acceptance'][0]:.3f}"
                          f" (env {cell['envelope_prediction']:.3f})  ESS {cell['ESS'][0]:.3f}"
                          f"  FLOP/es {cell['FLOP/es'][0]:.2e}  kept {cell['n_kept_per_seed']}")
    args.out.write_text(json.dumps(table, indent=2))
    print(f"wrote {args.out}")
    if args.latex:
        print(latex_rows(table, args.latex))


if __name__ == "__main__":
    main()
