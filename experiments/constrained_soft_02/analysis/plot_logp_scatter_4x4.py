"""Per-configuration exact-recovery scatter, soft 4x4 (app:logp-scatters).

ARCHIVED FIGURE (6 September 2026). The rationale below is historical:
DNFS path weights cannot generally be inverted into an endpoint log-density.
For configuration-probability validation use scripts/configuration_calibration_4x4.py
and scripts/plot_configuration_calibration_4x4.py instead. This script remains
only to reproduce the retired image, whose density interpretation was incorrect.

The third and last section of the appendix; it mirrors
experiments/constrained_hard_03/analysis/plot_logp_scatter_4x4.py and
experiments/dnfs_baseline_01/analysis/plot_logp_scatter_4x4.py so the three
figures read alike.

The estimator is the importance-weight identity

    w(x) = pi~(x) / q(x)   =>   log q(x) = log pi~(x) - log w(x),

with q the sampler's own (normalised) path-marginal density, so the y-axis
needs no fitted constant if the stored log-weights are the raw ratio. In
practice the eval's log-weights carry one common additive shift (base and
time-grid constants cancel in the normalised weights but ride along in the
raw ones), so the script prints the median offset per panel and removes it
before plotting. A healthy sampler shows a tight cloud on the diagonal;
probability mass misallocated between equal-energy configurations --
invisible to every energy-based instrument -- shows as vertical scatter that
no offset can hide.

How the soft panels differ from the other two sections':

  * The target is penalised, not constrained. The reference is the full
    2^16 = 65,536-state enumeration of

        log pi(x) = base_log_prob(x) - lambda * d * (c_+(x) - c_target)^2

    normalised to one, so every state the sampler can emit carries an exact
    probability and there is no feasible slice to condition on -- as in the
    unconstrained section, and unlike the hard section, whose x-axis is the
    enumerated conditional over the C(16,8) = 12,870 feasible states.
    The exhibit therefore asks whether the sampler recovers the penalised
    target; whether that target's composition is the requested one is a
    separate question, owned by the delivered-composition instruments.
  * The panel axis is composition, not coupling. The soft 4x4 family is run
    at one coupling (sigma = 0.1) and one penalty strength (lambda = 50);
    what varies across the chapter's 4x4 cells is the requested composition.
    Panels are therefore c = 0.50 and c = 0.80: the symmetric case, where
    the penalty's minimum coincides with the unpenalised mode and costs the
    sampler nothing, and the furthest composition the chapter requests,
    where the penalty pulls hard against the Ising energy and the two terms
    of the target disagree. Choosing the two sigma values instead would
    print the same distribution twice.

Reads the per-composition specialist cells (the same artefacts
tab:amort-4x4 scores), seeds 42-45 pooled, 5000 draws each.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.diagnostics.metrics import enumerate_states, exact_log_probs
from discrete_flow_sampler.targets.ising import IsingTarget

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]


L = 4
D_SITES = L * L
SEEDS = (42, 43, 44, 45)
# (run-name stem, requested composition, panel title). Both cells carry
# sigma = 0.1 and lambda = 50; see the module docstring for the panel choice.
PANELS = (
    ("S2_d4_c05_50k_l50_letf_anneal_offset_clip50", 0.5, r"$c = 0.50$"),
    ("S2_d4_c08_50k_l50_letf_anneal_offset_clip50", 0.8, r"$c = 0.80$"),
)
SIGMA = 0.1
PENALTY_STRENGTH = 50.0
COLOUR = "tab:red"


def state_keys(states):
    """Pack +-1 states to integer keys for exact log-prob lookup."""
    bits = (states > 0).long()
    powers = 2 ** torch.arange(states.shape[1])
    return bits @ powers


def panel_series(results_dir=None):
    """Plot-ready panels, shared by main() and the combined app:logp-scatters
    figure so the two cannot disagree about what is plotted.

    One entry per panel (compositions, not couplings), each carrying a single
    series with the per-panel median offset removed.
    """
    results_dir = Path(results_dir or REPO_ROOT / "results" / "02_constrained_soft")
    all_states = enumerate_states(D_SITES)
    panels = []
    for stem, composition, panel_title in PANELS:
        target = IsingTarget(
            D=L,
            sigma=SIGMA,
            bias=0.0,
            target_composition=composition,
            composition_penalty_strength=PENALTY_STRENGTH,
        )
        log_pi = exact_log_probs(target, all_states)
        key_to_logp = dict(zip(state_keys(all_states).tolist(), log_pi.tolist()))
        xs, ys = [], []
        for seed in SEEDS:
            matches = sorted(results_dir.glob(f"{stem}_seed{seed}_*"))
            if not matches:
                raise FileNotFoundError(f"no run dir for seed {seed} matching {stem}")
            run_dir = matches[0]
            samples = torch.load(
                run_dir / "eval" / "samples.pt", weights_only=True
            ).float()
            log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
            xs.append(
                torch.tensor([key_to_logp[k] for k in state_keys(samples).tolist()])
            )
            ys.append(target.log_prob(samples) - log_w)
        x, y = torch.cat(xs), torch.cat(ys)
        offset = (y - x).median()
        print(
            f"[scatter] c={composition:.2f}: {len(x)} points, median offset "
            f"{offset:.3f} nats, residual sd after removal "
            f"{(y - x - offset).std():.3f} nats"
        )
        # Limits from the enumerated support actually visited, padded, so the
        # diagonal spans the plotted cloud rather than the full 2^16 tail.
        lims = (
            min(x.min().item(), (y - offset).min().item()) - 0.3,
            max(x.max().item(), (y - offset).max().item()) + 0.3,
        )
        panels.append(
            {
                "title": panel_title,
                "lims": lims,
                "xlabel": r"exact $\log \pi(x)$",
                "series": [("soft specialist", COLOUR, x, y - offset)],
            }
        )
    return panels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "02_constrained_soft",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output PNG path (the Overleaf assets file)",
    )
    args = parser.parse_args(argv)

    # Style annex: in-figure labels 9pt, annotations 8pt.
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(4.54, 2.5))

    for ax, panel in zip(axes, panel_series(args.results_dir)):
        for _label, colour, x, y in panel["series"]:
            ax.scatter(x, y, s=4, alpha=0.25, lw=0, color=colour, rasterized=True)
        ax.plot(panel["lims"], panel["lims"], color="black", lw=0.8, zorder=0)
        ax.set_xlim(panel["lims"])
        ax.set_ylim(panel["lims"])
        ax.set_title(panel["title"], fontsize=9)
        ax.set_xlabel(panel["xlabel"])

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"[scatter] wrote {args.out}")


if __name__ == "__main__":
    main()
