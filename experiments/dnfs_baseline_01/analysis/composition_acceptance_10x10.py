"""Composition-acceptance taster at 10x10: the cost of asking the
unconstrained sampler for a fixed composition by rejection.

The baseline chapter's sampler targets the plain Ising model, so a request
for composition c = n_plus / d has to be met by drawing and discarding: keep
the draws whose +1 count equals n_plus, with their original importance
weights (restricting an importance sample to an event and keeping its weights
is importance sampling for the conditional -- the normaliser cancels in the
self-normalised estimator; see rejection_rows.py in the hard chapter). The
table this script feeds is the taster for the constrained chapters.

At sigma = 0.1 the target is disordered, so its composition marginal is a
near-binomial peaked at c = 1/2: acceptance is largest at the centre (about
5%, below the uniform sampler's 8% because the ferromagnetic coupling already
fattens the tails) and falls with |c - 1/2| to about 0.03% by c = 0.25. At
sigma_c the picture inverts: the critical target has moved its mass towards
the ordered tails, so the balanced slice is starved (about 0.3%) and
acceptance is flat-to-rising across 0.25 <= c <= 0.5 rather than peaked.
Rejection is cheapest exactly where it is least needed.

Two acceptances are computed and the table prints the raw one. The raw
acceptance n_kept / n_drawn is the proposal's hit rate on the manifold, and
it is what the overhead 1 / acceptance charges for: every rejected draw cost
a full forward pass. The weighted acceptance, the normalised-weight mass on
the slice, is instead the sampler's estimate of the target's composition
mass pi(C). The hard chapter's rejection rows print the acceptance of draws
(rejection_rows.py's convention) and this table follows it; both land in the
JSON; at ESS fractions of 0.99 (sigma = 0.1) and 0.90 (sigma_c) they agree to
the printed precision at sigma = 0.1 and differ by about one seed SD (0.28%
raw vs 0.23% weighted at c = 1/2) at sigma_c.

Caveat for the caption: the sigma_c cells rest on 50-90 kept draws pooled
over four seeds, so their spread is dominated by counting noise and the
row-to-row ordering at sigma_c should not be over-read.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from experiments.constrained_hard_03.analysis.rejection_rows import kept_draws

REPO_ROOT = Path(__file__).resolve().parents[3]


RESULTS = REPO_ROOT / "results" / "01_baseline"
N_SITES = 100  # D=10 -> d = 100
SEEDS = (42, 43, 44, 45)
COUPLINGS = {  # label -> (run-dir glob, sigma, LaTeX column header)
    "s010": ("stage_4_d10_budget_seed{seed}_*", 0.1, r"$\sigma=0.1$"),
    "sc": (
        "stage_4_d10_sc_hardrecipe_efc_seed{seed}_*",
        0.220343,
        r"$\sigma_c$",
    ),
}
COMPOSITIONS = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25)
TABLE_COMPOSITIONS = (0.50, 0.40, 0.30)


def uniform_acceptance(n_plus):
    """The no-model line: a uniform sampler lands on c = n_plus/d with
    probability C(d, n_plus) / 2^d (0.080 at d = 100, c = 1/2)."""
    return math.comb(N_SITES, n_plus) / 2**N_SITES


def seed_acceptance(run_dir, n_plus):
    """Raw and weighted acceptance of one seed's eval on the c-manifold."""
    _kept_samples, kept_log_w, n_drawn = kept_draws(run_dir, n_plus)
    all_log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    # normalised-weight mass on the slice = exp(lse(kept) - lse(all))
    weighted = (
        (torch.logsumexp(kept_log_w, 0) - torch.logsumexp(all_log_w, 0)).exp().item()
        if kept_log_w.numel()
        else 0.0
    )
    return {
        "n_kept": int(kept_log_w.numel()),
        "n_drawn": n_drawn,
        "raw": kept_log_w.numel() / n_drawn,
        "weighted": weighted,
    }


def composition_cell(run_dirs, n_plus):
    """One (coupling, composition) cell: per-seed rows, mean +- SD over the
    four seeds (np.std, population, as rejection_rows does), and the pooled
    fraction the earlier hand check quoted."""
    per_seed = [seed_acceptance(run_dir, n_plus) for run_dir in run_dirs]
    raw = np.array([row["raw"] for row in per_seed])
    weighted = np.array([row["weighted"] for row in per_seed])
    n_kept = sum(row["n_kept"] for row in per_seed)
    n_drawn = sum(row["n_drawn"] for row in per_seed)
    return {
        "n_plus": n_plus,
        "n_kept": n_kept,
        "n_drawn": n_drawn,
        "n_kept_per_seed": [row["n_kept"] for row in per_seed],
        "pooled_acceptance": n_kept / n_drawn,
        "raw_acceptance": (float(raw.mean()), float(raw.std())),
        "weighted_acceptance": (float(weighted.mean()), float(weighted.std())),
        "overhead": float(1.0 / raw.mean()) if raw.mean() > 0 else math.inf,
    }


def latex_rows(table):
    """Compact taster rows: c, then (acceptance, overhead) per coupling."""

    def pair(cell):
        mean, sd = cell["raw_acceptance"]
        return (
            f"${100 * mean:.2f} \\pm {100 * sd:.2f}\\%$ & "
            f"${cell['overhead']:.0f}\\times$"
        )

    lines = []
    for composition in TABLE_COMPOSITIONS:
        key = f"{composition:.2f}"
        lines.append(
            f"        {composition:.2f} & "
            + " & ".join(pair(table[label][key]) for label in COUPLINGS)
            + r" \\"
        )
    uniform = table["uniform"]["0.50"]
    lines.append(r"        \midrule")
    lines.append(
        f"        uniform draws, $c=0.5$ & \\multicolumn{{4}}{{c}}"
        f"{{${100 * uniform:.2f}\\%$, ${1 / uniform:.0f}\\times$}} \\\\"
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--out", type=Path, default=RESULTS / "composition_acceptance_10x10.json"
    )
    args = parser.parse_args(argv)

    table = {
        "uniform": {
            f"{c:.2f}": uniform_acceptance(round(c * N_SITES)) for c in COMPOSITIONS
        }
    }
    for label, (glob, sigma, _header) in COUPLINGS.items():
        run_dirs = [
            sorted(args.results.glob(glob.format(seed=seed)))[0] for seed in SEEDS
        ]
        table[label] = {"sigma": sigma, "runs": [d.name for d in run_dirs]}
        for composition in COMPOSITIONS:
            table[label][f"{composition:.2f}"] = composition_cell(
                run_dirs, round(composition * N_SITES)
            )
    args.out.write_text(json.dumps(table, indent=2))

    print(
        f"{'coupling':8} {'c':>5} {'pooled':>8} {'raw mean+-SD':>17} "
        f"{'weighted mean+-SD':>19} {'kept':>6} {'overhead':>9} {'uniform':>8}"
    )
    for label in COUPLINGS:
        for composition in COMPOSITIONS:
            key = f"{composition:.2f}"
            cell = table[label][key]
            raw, weighted = cell["raw_acceptance"], cell["weighted_acceptance"]
            print(
                f"{label:8} {composition:5.2f} {cell['pooled_acceptance']:8.4f} "
                f"{raw[0]:8.4f} +- {raw[1]:.4f} "
                f"{weighted[0]:10.4f} +- {weighted[1]:.4f} {cell['n_kept']:6d} "
                f"{cell['overhead']:8.0f}x {table['uniform'][key]:8.4f}"
            )
    print("\nLaTeX rows (columns: c, acc sigma=0.1, overhead, acc sigma_c, overhead):")
    print(latex_rows(table))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
