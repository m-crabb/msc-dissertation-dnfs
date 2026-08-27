"""Per-configuration exact-recovery scatter at the 4x4 gate (app:logp-scatters).

The house tables deliberately exclude TV/KL over configurations; the agreed
replacement at enumerable sizes is this scatter: estimated sampler
log-density against the exactly enumerated conditional, one point per drawn
sample. The estimator is the importance-weight identity

    w(x) = pi~(x) / q(x)   =>   log q(x) = log pi~(x) - log w(x),

with q the sampler's own (normalised) path-marginal density, so the y-axis
needs no fitted constant IF the stored log-weights are the raw ratio. In
practice the swap eval's log-weights carry one common additive shift (the
on-manifold base constant and any time-grid constant cancel in the
normalised weights but ride along in the raw ones -- see the App. C ELBO
note), so the script PRINTS the weighted mean offset per run and removes
the per-arm MEDIAN offset before plotting; a healthy sampler then shows a
tight cloud on the diagonal, and mass misallocation shows as vertical
scatter that no offset can hide. The x-axis is log pi_cond = log pi~ -
logsumexp over the enumerated C(16,8) slice, exact by construction.

Reads the Wave-2 runs (tag 20260825-hard-w2) for the three printed arms;
the two held factorised sigma_c cells are omitted on purpose -- this is a
print exhibit and they are held from print.
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
    conditional_pmf_at_composition, enumerate_states, exact_log_probs)

L = 4
D_SITES = L * L
TAG = "20260825-hard-w2"
SEEDS = (42, 43, 44)
# Printed arms only (fimo2ef/fmo2ef sigma_c are HELD from print s70).
ARM_STYLE = {
    "mo": ("mask-one", "tab:green"),
    "ma": ("masked-attention", "tab:blue"),
    "thp": ("two-hole patch", "tab:orange"),
}
SIGMA_PANELS = (("s010", r"$\sigma = 0.1$"), ("s220", r"$\sigma = \sigma_c$"))


def exact_reference(cfg):
    from experiments.constrained_hard_03.run import build_target_and_head

    target, _ = build_target_and_head(cfg, device="cpu")
    all_states = enumerate_states(D_SITES)
    log_pi = exact_log_probs(target, all_states)
    slice_states, log_p_cond = conditional_pmf_at_composition(
        all_states, log_pi, target.n_plus_target
    )
    return target, slice_states.float(), log_p_cond


def state_keys(states):
    """Pack +-1 states to integer keys for exact log-prob lookup."""
    bits = (states > 0).long()
    powers = 2 ** torch.arange(states.shape[1])
    return bits @ powers


def panel_series(results_dir=None):
    """Plot-ready panels, shared by main() and the combined app:logp-scatters
    figure so the two can never disagree about what is being plotted.

    One entry per panel: the exact-reference limits, the axis label, and one
    (label, colour, x, y) series per printed arm with the per-arm median
    offset already removed.
    """
    from experiments.constrained_hard_03.configs import CONFIGS

    results_dir = Path(results_dir or REPO_ROOT / "results" / "03_hard")
    panels = []
    for sigma_label, panel_title in SIGMA_PANELS:
        cfg_any = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_mo_10k_w2"]
        target, slice_states, log_p_cond = exact_reference(cfg_any)
        key_to_logp = dict(zip(state_keys(slice_states).tolist(),
                               log_p_cond.tolist()))
        lims = (log_p_cond.min().item() - 0.5, log_p_cond.max().item() + 0.5)

        series = []
        for arm, (arm_label, colour) in ARM_STYLE.items():
            cfg = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_{arm}_10k_w2"]
            xs, ys = [], []
            for seed in SEEDS:
                run_dir = results_dir / f"{cfg.name}_seed{seed}_{TAG}"
                samples = torch.load(run_dir / "eval" / "samples.pt",
                                     weights_only=True).float()
                log_w = torch.load(run_dir / "eval" / "log_weights.pt",
                                   weights_only=True)
                xs.append(torch.tensor([key_to_logp[k] for k in
                                        state_keys(samples).tolist()]))
                ys.append(target.log_prob(samples) - log_w)
            x, y = torch.cat(xs), torch.cat(ys)
            offset = (y - x).median()
            print(f"[scatter] {sigma_label} {arm}: median offset "
                  f"{offset:.3f}, residual sd after removal "
                  f"{(y - x - offset).std():.3f}")
            series.append((arm_label, colour, x, y - offset))
        panels.append({"title": panel_title, "lims": lims, "series": series,
                       "xlabel": r"exact $\log \pi_{\mathrm{cond}}(x)$"})
    return panels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=REPO_ROOT / "results" / "03_hard")
    parser.add_argument("--out", type=Path, required=True,
                        help="output PNG path (the Overleaf assets file)")
    args = parser.parse_args(argv)

    # Style annex: in-figure labels 9pt, annotations 8pt.
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(4.54, 2.5), sharey=False)

    for ax, panel in zip(axes, panel_series(args.results_dir)):
        for label, colour, x, y in panel["series"]:
            ax.scatter(x, y, s=4, alpha=0.25, lw=0, color=colour,
                       label=label, rasterized=True)
        ax.plot(panel["lims"], panel["lims"], color="black", lw=0.8, zorder=0)
        ax.set_xlim(panel["lims"])
        ax.set_ylim(panel["lims"])
        ax.set_title(panel["title"], fontsize=9)
        ax.set_xlabel(panel["xlabel"])
    axes[0].set_ylabel(r"estimated $\log \hat q(x)$ (offset removed)")
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.1)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"[scatter] wrote {args.out}")


if __name__ == "__main__":
    main()
