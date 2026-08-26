"""Per-configuration exact-recovery scatter, UNCONSTRAINED 4x4 (app:logp-scatters).

The house tables deliberately exclude TV/KL over configurations; the agreed
replacement at enumerable sizes is this scatter, one section per results
chapter. This is the unconstrained chapter's section; the hard chapter's
counterpart is experiments/constrained_hard_03/analysis/plot_logp_scatter_4x4.py
and this script deliberately mirrors it so the two figures read alike.

The estimator is the importance-weight identity

    w(x) = pi~(x) / q(x)   =>   log q(x) = log pi~(x) - log w(x),

with q the sampler's own (normalised) path-marginal density, so the y-axis
needs no fitted constant IF the stored log-weights are the raw ratio. In
practice the eval's log-weights carry one common additive shift (base and
time-grid constants cancel in the normalised weights but ride along in the
raw ones), so the script PRINTS the median offset per family and removes it
before plotting. A healthy sampler then shows a tight cloud on the diagonal,
and probability mass misallocated between equal-energy configurations --
invisible to every energy-based instrument -- shows as vertical scatter that
no offset can hide.

WHY THIS IS THE UNCONSTRAINED VERSION, AND HOW IT DIFFERS FROM THE HARD ONE:
the hard chapter conditions on a fixed composition, so its x-axis is the
enumerated CONDITIONAL over the C(16,8) = 12,870 feasible states. Here there
is no constraint, so the reference is the full 2^16 = 65,536-state
enumeration normalised to one (`exact_log_probs`, logsumexp = 0). That makes
this the stronger exhibit of the two: every state the sampler can emit has an
exact probability, with no slice to condition on.

Panels are the chapter's two operating points. The sigma_c panel reads the
Wave-1 `_sc` retrains at the ONE critical coupling SIGMA_C = 0.220343 (they
passed their pre-registered bands 4/4, final fp32 5000-draw eval ESS
0.986 +- 0.004 against a 0.93 floor); the sigma = 0.1 panel is untouched by
that migration and reads the original family. Never mix couplings in one
comparison.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states, exact_log_probs)
from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget

L = 4
D_SITES = L * L
SEEDS = (42, 43, 44, 45)
# One family per panel: (glob, sigma, panel title). The sigma = 0.1 family
# predates the run-tag convention, hence the timestamped glob.
PANELS = (
    ("stage_4_d4_seed{seed}_20260609-*", 0.1, r"$\sigma = 0.1$"),
    ("stage_4_d4_critical_sc_seed{seed}_20260824-wave1-sc", SIGMA_C,
     r"$\sigma = \sigma_c$"),
)
COLOUR = "tab:purple"


def state_keys(states):
    """Pack +-1 states to integer keys for exact log-prob lookup."""
    bits = (states > 0).long()
    powers = 2 ** torch.arange(states.shape[1])
    return bits @ powers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=REPO_ROOT / "results" / "01_baseline")
    parser.add_argument("--out", type=Path, required=True,
                        help="output PNG path (the Overleaf assets file)")
    args = parser.parse_args(argv)

    # Style annex: in-figure labels 9pt, annotations 8pt; \textwidth two-panel.
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1))

    all_states = enumerate_states(D_SITES)
    summary = {}
    for ax, (pattern, sigma, panel_title) in zip(axes, PANELS):
        target = IsingTarget(D=L, sigma=sigma, bias=0.0)
        log_pi = exact_log_probs(target, all_states)
        key_to_logp = dict(zip(state_keys(all_states).tolist(),
                               log_pi.tolist()))

        xs, ys = [], []
        for seed in SEEDS:
            matches = sorted(args.results_dir.glob(pattern.format(seed=seed)))
            if not matches:
                raise FileNotFoundError(
                    f"no run dir for seed {seed} matching {pattern}")
            run_dir = matches[0]
            samples = torch.load(run_dir / "eval" / "samples.pt",
                                 weights_only=True).float()
            log_w = torch.load(run_dir / "eval" / "log_weights.pt",
                               weights_only=True)
            log_q = target.log_prob(samples) - log_w
            xs.append(torch.tensor([key_to_logp[k]
                                    for k in state_keys(samples).tolist()]))
            ys.append(log_q)

        x, y = torch.cat(xs), torch.cat(ys)
        offset = (y - x).median()
        residual_sd = (y - x - offset).std()
        summary[panel_title] = (float(offset), float(residual_sd), len(x))
        print(f"[scatter] sigma={sigma:.6f}: {len(x)} points, median offset "
              f"{offset:.3f} nats, residual sd after removal "
              f"{residual_sd:.3f} nats")

        # Limits from the enumerated support actually visited, padded, so the
        # diagonal spans the plotted cloud rather than the full 2^16 tail.
        lims = (min(x.min().item(), (y - offset).min().item()) - 0.3,
                max(x.max().item(), (y - offset).max().item()) + 0.3)
        ax.scatter(x, y - offset, s=4, alpha=0.25, lw=0, color=COLOUR,
                   rasterized=True)
        ax.plot(lims, lims, color="black", lw=0.8, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(r"exact $\log \pi(x)$")
    axes[0].set_ylabel(r"estimated $\log \hat q(x)$ (offset removed)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"[scatter] wrote {args.out}")
    return summary


if __name__ == "__main__":
    main()
