"""Rejection rows for the hard house tables: sample off-manifold, then filter.

Two routes reach the fixed-composition target WITHOUT a composition-preserving
sampler, and both belong in the hard tables as the null the swap CTMC is
measured against:

  * UNCONSTRAINED + REJECT -- draw from the plain Ising sampler of Chapter 3
    and keep only the draws that happen to land on c = 1/2.
  * REJECT OFF SOFT -- draw from the penalised sampler of Chapter 4 and do
    the same.

BOTH ARE EXACT, which is the point worth stating before any cost claim. For
importance-weighted draws {(x, w)} targeting pi, restricting to the event
{x in C} and keeping the same weights is importance sampling for the
conditional pi(. | C): the restriction changes the normaliser, and the
normaliser cancels in the self-normalised estimator. For the SOFT sampler the
argument needs one extra step and gives a stronger conclusion: on the manifold
the penalty term lambda*d*(c_+(x) - c_target)^2 is IDENTICALLY ZERO because
c_+(x) = c_target exactly, so

    pi_soft(x) restricted to C  ==  pi_Ising(x) restricted to C  ==  pi_C(x),

i.e. filtering the soft sampler targets the HARD chapter's distribution with
no penalty residue at all. That is why the kept draws score an ESS fraction in
the nineties rather than paying for the penalty.

WHICH TARGET THE ESS REFERS TO -- an examiner will ask, and the answer is
"both, and they are the same number". The stored log-weights are
log pi~(x) - log q(x) for the sampler's ORIGINAL target pi~. Targeting the
constrained pi_C = pi~ 1[x in C] / Z_C with the rejection-filtered proposal
q_C = q 1[x in C] / q(C) needs

    w_C(x) = pi_C(x) / q_C(x) = [pi~(x)/Z_C] [q(C)/q(x)] = (q(C)/Z_C) w(x),

i.e. the ORIGINAL weights times a constant. Self-normalised ESS is invariant
under w -> c w, so no reweighting is required and the printed number is the
ESS with respect to the constrained target. Verified 2026-08-27 on the soft
c=0.5 cell: 0.973194 from the stored weights, 0.973194 rebuilt against the
hard chapter's own log_prob, 0.973194 after adding an arbitrary +7.3 offset.
Two identities make the constant exactly constant rather than nearly so, both
measured on the kept draws: the soft penalty's contribution is 0.00e+00 on
the manifold, and log pi_Ising - log pi_hard has spread 0.00e+00 there (the
fixed-composition base differs by a pure constant).

BUT THE ESS PRICES THE SURVIVORS, NOT THE REJECTION -- never quote it alone
for these rows. At 8x8 sigma_c it reads 0.936 on the 16 draws per seed that
lived and is silent on the 4,984 that did not; all of that cost sits in
FLOP/es through the 1/acceptance factor. Read alone the column suggests the
unconstrained route is competitive at criticality, where the paired
FLOP/es says 4.2e12 against the two-hole patch head's 3.2e9. An ESS
estimated from 16 weights is also noisy: 0.936 +- 0.038 means "consistent
with exact", not a measurement.

WHAT SEPARATES THE ROUTES IS WASTE, NOT BIAS, and the measurement is stark.
Acceptance on the shipped evals:

    unconstrained + reject   11.2% (4x4 sigma=0.1)   1.24% (4x4 sigma_c)
                                                     0.33% (8x8 sigma_c)
    reject off soft          91.0% (4x4 sigma=0.1)
    swap CTMC                100% by construction, every size and coupling

The collapse at criticality is the Ising model concentrating on ORDERED
configurations, which are exactly the ones far from balanced: a uniform
sampler would land on the 4x4 manifold 19.6% of the time
(C(16,8)/2^16), and the critical target manages 1.24%. FLOP/es therefore
carries the whole argument -- it is the raw per-sample bill divided by BOTH
the acceptance rate and the ESS fraction, so a rejected draw is charged for.

WHY ONE CELL IS BLANK. The unconstrained 8x8 cell at sigma = 0.1 was never
run (the baseline chapter's 8x8 rung is sigma_c only); every soft cell now
exists at both sizes and couplings (the soft chapter moved its production
size to 8x8 and filled its 4x4 sigma_c half on 2026-09-02). The blank is
left in the table as the reminder of which one.

WHAT THE 8x8 SOFT ROW SAYS. The on-slice acceptance is ~0.50 at 64 sites
against 0.91 at 16 (the envelope exp(-lambda d (c - c_t)^2) on the c = k/d
grid puts 0.919 / 0.499 of its mass on the exact slice at d = 16 / 64), so
the "near-free" 1.09x overhead of the enumerable size is 2x at production
and grows as sqrt(pi d / lambda). The survivors' ESS equals the specialist's
own ESS in the soft table, which is the weights-are-already-correct
argument above made measurable.

THE ERROR COLUMNS ARE SCORED AT THE RUNG'S OWN DRAW COUNT, NOT AT WHATEVER
REJECTION HAPPENED TO LEAVE. Rejection changes N by the acceptance rate --
2,247 and 18,207 kept at the 4x4 floor coupling against the neural rows' 512,
249 and 66 at criticality -- and profile errors scale with N, so scoring a
rejection cell at its own kept count would let a route look accurate purely
for having survived more draws (or inaccurate purely for having survived
fewer). Every filled cell is therefore SUBSAMPLED WITHOUT REPLACEMENT to the
rung's own draw count and averaged over replicates, exactly as the floor row
resamples, so the numbers sit on one scale across every row of the table.

Where the kept count falls BELOW the rung's draw count the cell cannot be
equalised downward and prints "--": subsampling up is not a thing, and the fix
is a larger eval. `draws_needed` records how large -- about 41,000 raw draws
to reach 512 kept at 4x4 sigma_c (8x the current eval) and about 1.5 million
to reach 5,000 at 8x8 sigma_c (300x). Both are re-evals of trained
checkpoints, not retrains.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from discrete_flow_sampler.diagnostics.flops import (
    measured_forward_flops, neural_sampling_flops_per_sample,
    per_effective_sample)
from discrete_flow_sampler.diagnostics.metrics import (
    correlation_profile_error, energy_wasserstein2,
    magnetisation_profile_error)

# (paradigm, rung, coupling) -> run-dir glob, or None where no run exists.
# Rungs are named by SITE COUNT to match the hard chapter (d16 = 4x4,
# d64 = 8x8); the baseline and soft chapters name their cells by lattice
# EDGE (d4, d8), which is why the globs read d4/d8 for the same lattices.
CELLS = {
    ("unconstrained", 16, "s010"): ("01_baseline", "stage_4_d4_seed{seed}_20260609-*"),
    ("unconstrained", 16, "s220"): ("01_baseline", "stage_4_d4_critical_sc_seed{seed}_20260824-wave1-sc"),
    ("unconstrained", 64, "s010"): None,
    ("unconstrained", 64, "s220"): ("01_baseline", "stage_4_d8_critical_paper_curriculum_sc_seed{seed}_20260824-wave1-sc"),
    # Soft cells are the SAME runs the soft chapter's house tables print
    # (tab:eval-soft-4x4 = the 10k house family, tab:eval-soft-8x8 = the 8x8
    # house specialists), so the rejection row and the soft table are priced
    # off identical draws. The old 50k anneal cell (S2_d4_c05_50k_l50_letf_
    # anneal_offset_clip50) was a different sampler from the one soft prints.
    ("soft", 16, "s010"): ("02_constrained_soft", "S2_d4_c0500_10k_l50_letf_house_seed{seed}_20260902-softhouse-d16-10k"),
    ("soft", 16, "s220"): ("02_constrained_soft", "S2_d4_c0500_10k_l50_letf_house_sc_seed{seed}_20260902-softhouse-d16-10k"),
    ("soft", 64, "s010"): ("02_constrained_soft", "S2_d8_c0500_l50_letf_ne128_house_seed{seed}_20260831-softhouse-d64"),
    ("soft", 64, "s220"): ("02_constrained_soft", "S2_d8_c0500_l50_letf_ne128_house_sc_seed{seed}_20260831-softhouse-d64"),
}
SEEDS = (42, 43, 44, 45)
# The draw count each rung's neural rows are evaluated at; a rejection cell
# whose KEPT count falls below this cannot support the error columns.
FLOOR_DRAWS = {16: 512, 64: 5000}


def kept_draws(run_dir, n_plus):
    """Draws on the manifold, with their original log-weights.

    The weights are NOT recomputed: restricting an importance sample to an
    event and keeping its weights is importance sampling for the conditional
    (see the module docstring), so the stored weights are already the right
    ones and any renormalisation cancels in the self-normalised estimator.
    """
    samples = torch.load(run_dir / "eval" / "samples.pt",
                         weights_only=True).float()
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    on_manifold = (samples > 0).sum(dim=-1) == n_plus
    return samples[on_manifold], log_w[on_manifold], int(samples.shape[0])


def _seed_errors(samples, weights, reference, reference_weights,
                 reference_energy, energy_of, lattice_edge, floor_draws,
                 n_replicates, generator):
    """The three error columns for ONE seed, scored at `floor_draws`.

    Subsampled without replacement and averaged over replicates so the number
    sits at the rung's own draw count rather than at whatever rejection left
    (see the module docstring).
    """
    w_ref = reference_weights if reference_weights is not None else \
        torch.full((reference.shape[0],), 1.0 / reference.shape[0])
    energies = energy_of(samples)
    replicates = []
    for _ in range(n_replicates):
        idx = torch.randperm(samples.shape[0], generator=generator)[:floor_draws]
        sub_w = weights[idx] / weights[idx].sum()
        replicates.append({
            "dMag": magnetisation_profile_error(
                samples[idx], sub_w, reference, lattice_edge,
                reference_weights=w_ref),
            "dCorr": correlation_profile_error(
                samples[idx], sub_w, reference, lattice_edge,
                reference_weights=w_ref),
            "EW2": energy_wasserstein2(
                energies[idx], sub_w, reference_energy,
                reference_weights=w_ref),
        })
    return {k: float(np.mean([r[k] for r in replicates])) for k in replicates[0]}


def rejection_cell(results_root, glob, n_sites, reference, reference_energy,
                   reference_weights, energy_of, lattice_edge, floor_draws,
                   n_replicates=64, seed=0):
    """One (paradigm, rung, coupling) cell as mean +- SD over seeds.

    PER SEED, not pooled, so the row reports the same statistic as every
    neural row above it. Pooling was checked and is safe here -- the raw
    log-weight medians agree across seeds to 0.02-0.15 nats, so no run
    dominates a pooled softmax, and the pooled ESS matches the per-seed mean
    to about 0.001 -- but it discards the seed spread, which is exactly what
    tells a reader that the 8x8 critical cell rests on 15-18 draws per seed.
    """
    from experiments.dnfs_baseline_01.run import (_build_model,
                                                  _rebuild_from_run_dir)

    generator = torch.Generator().manual_seed(seed)
    rows, n_kept_total, n_drawn_total, run_dir = [], 0, 0, None
    for seed_value in SEEDS:
        matches = sorted(Path(results_root).glob(glob.format(seed=seed_value)))
        if not matches:
            continue
        run_dir = matches[0]
        samples, log_w, drawn = kept_draws(run_dir, n_sites // 2)
        n_kept = samples.shape[0]
        n_kept_total += n_kept
        n_drawn_total += drawn
        weights = torch.softmax(log_w, dim=0)
        row = {"ESS": float(1.0 / (weights.pow(2).sum() * n_kept)),
               "acceptance": n_kept / drawn, "n_kept": n_kept}
        if n_kept >= floor_draws:
            row |= _seed_errors(samples, weights, reference, reference_weights,
                                reference_energy, energy_of, lattice_edge,
                                floor_draws, n_replicates, generator)
        rows.append(row)
    if run_dir is None:
        return None

    # FLOP/es charges for the REJECTED draws: the raw per-sample bill is paid
    # on every draw, and only `acceptance` of them survive to be weighted.
    cfg, target, _device = _rebuild_from_run_dir(run_dir)
    model = _build_model(cfg, target)
    per_forward = measured_forward_flops(
        model, (reference[:1], torch.full((1,), 0.5)))
    raw = neural_sampling_flops_per_sample(per_forward, cfg.ctmc.n_euler_steps,
                                           n_sites)
    for row in rows:
        row["FLOP/es"] = per_effective_sample(raw / row["acceptance"],
                                              row["ESS"])

    cell = {key: (float(np.mean([r[key] for r in rows])),
                  float(np.std([r[key] for r in rows])))
            for key in rows[0] if key not in ("n_kept",)}
    cell["n_kept"] = n_kept_total
    cell["n_kept_per_seed"] = [r["n_kept"] for r in rows]
    cell["n_drawn"] = n_drawn_total
    cell["per_forward_flops"] = per_forward
    if "dMag" in rows[0]:
        cell["scored_at_draws"] = floor_draws
    else:
        cell["draws_needed"] = int(np.ceil(
            floor_draws / (n_kept_total / n_drawn_total)))
    return cell


def references_for(rung, sigma_label, results_dir):
    """(reference states, their per-site energies, energy_of) for a rung.

    Reuses each rung's own house-table machinery so the rejection rows are
    scored against exactly what the neural rows are scored against: the
    enumerated conditional at 4x4, the certified chain pool at 8x8.
    """
    if rung == 16:
        from experiments.constrained_hard_03.analysis import house_table_4x4 as h4
        from experiments.constrained_hard_03.configs import CONFIGS

        cfg = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_mo_10k_w2"]
        target, _head, states, probs = h4.exact_reference(cfg)
        energy_of = lambda x: h4.energy_per_site(target, x)
        return states, energy_of(states), energy_of, probs

    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8
    from experiments.constrained_hard_03.run import build_target_and_head

    probe = (results_dir /
             f"{h8.CELL_NAME[sigma_label].format(arm='mo')}_seed42_{h8.TAG}")
    target, _ = build_target_and_head(h8.registry_config_for(probe), "cpu")
    chains = h8.load_reference_chains(
        REPO_ROOT / "results" / "03_hard" / "kawasaki_w2", h8.L,
        h8.KAWASAKI_TAG[sigma_label], 0.2)
    reference = torch.cat(chains)
    energy_of = lambda x: h8.energy_per_site(target, x)
    return reference, energy_of(reference), energy_of, None


LATEX_LABEL = {
    "unconstrained": r"unconstrained \gls{dnfs} $+$ reject",
    "soft": r"reject off soft \gls{dnfs}",
}
ERROR_COLUMNS = ("dMag", "dCorr", "EW2")


def latex_rows(table, rung, print_errors=False):
    """The two rejection rows for one rung: ESS and FLOP/es only.

    THE ERROR COLUMNS ARE DELIBERATELY NOT PRINTED, even for the two cells
    where the kept count supports them. Three reasons, and the third is the
    decisive one:

      * they do not discriminate. Equalised to the rung's own N every route
        sits at the floor -- unconstrained + reject reads 5.8/12.2/6.5 and
        reject off soft 7.4/11.8/6.9 against a floor of 7.2/10.9/6.5 -- so
        the cells restate the floor row rather than separating anything;
      * these rows exist for the COST argument, which ESS and FLOP/es carry
        in full: both routes are exact, and what separates them is acceptance
        (11.24% / 1.24% / 0.33% against the swap CTMC's 100%);
      * printing them at only ONE coupling, which is all the draw counts
        allow, leaves a half-filled block whose asymmetry a reader must chase
        into the caption. A uniformly blank error block says one thing.

    The numbers are still computed and land in the JSON; `print_errors=True`
    emits them if the disposition is ever revisited.

    Nothing here is bolded: these rows are the null the neural rows are
    measured against, not competitors for a best-in-column mark.
    """
    lines = []
    for paradigm in ("unconstrained", "soft"):
        cells = []
        for sigma_label in ("s010", "s220"):
            cell = table.get(f"{paradigm}_{rung}_{sigma_label}")
            if cell is None:
                cells += ["--"] * 5
                continue
            cells.append(f"${cell['ESS'][0]:.3f} \\pm {cell['ESS'][1]:.3f}$")
            cells += [f"${cell[c][0] * 100:.1f} \\pm {cell[c][1] * 100:.1f}$"
                      if print_errors and c in cell else "--"
                      for c in ERROR_COLUMNS]
            mean, sd = cell["FLOP/es"]
            exponent = int(np.floor(np.log10(mean)))
            cells.append(f"${mean / 10 ** exponent:.1f}"
                         f"\\times10^{{{exponent}}}$")
        lines.append(f"        {LATEX_LABEL[paradigm]} & "
                     + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results" / "03_hard" / "rejection_rows.json")
    parser.add_argument("--latex", type=int, choices=(16, 64),
                        help="emit this rung's two rejection rows and exit")
    parser.add_argument("--print-errors", action="store_true",
                        help="also print the equalised error columns "
                             "(off in print; see latex_rows)")
    args = parser.parse_args(argv)

    hard_dir = args.results_root / "03_hard"
    table = {}
    for rung in (16, 64):
        for sigma_label in ("s010", "s220"):
            wanted = [k for k in CELLS
                      if k[1] == rung and k[2] == sigma_label and CELLS[k]]
            if not wanted:
                continue
            reference, reference_energy, energy_of, ref_probs = references_for(
                rung, sigma_label, hard_dir)
            for key in wanted:
                subdir, glob = CELLS[key]
                cell = rejection_cell(
                    args.results_root / subdir, glob, rung, reference,
                    reference_energy, ref_probs, energy_of, int(rung ** 0.5),
                    FLOOR_DRAWS[rung])
                if cell:
                    table[f"{key[0]}_{rung}_{sigma_label}"] = cell

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2))

    if args.latex:
        print(latex_rows(table, args.latex, args.print_errors))
        return

    print(f"{'cell':28} {'accept':>8} {'kept/seed':>18} {'ESS':>16} "
          f"{'FLOP/es':>10}  errors")
    for key, c in table.items():
        errs = ("filled" if "dMag" in c
                else f"-- (needs {c['draws_needed']:,} raw draws)")
        ess = f"{c['ESS'][0]:.3f} +- {c['ESS'][1]:.3f}"
        print(f"{key:28} {c['acceptance'][0]*100:7.2f}% "
              f"{str(c['n_kept_per_seed']):>18} {ess:>16} "
              f"{c['FLOP/es'][0]:10.2e}  {errs}")
    for key, spec in CELLS.items():
        if spec is None:
            print(f"{key[0]}_{key[1]}_{key[2]:6} NO RUN -- blank in print")


if __name__ == "__main__":
    main()
