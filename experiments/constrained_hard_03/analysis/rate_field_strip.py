"""One trajectory of the trained swap process, base to terminal state, with the learned rate
field drawn at a few grid times: what the generator looks like, as a picture.

Three rows per displayed time t_k of ONE rollout of the trained head from a uniform-on-slice
base state:
  1. the state x_{t_k}, with a fixed anchor site marked;
  2. the learned one-way rate [G(a, j | x, t)]_+ the sampler puts on swapping the anchor a with
     every other site j (zero on like-spin partners by antisymmetry; G is index-antisymmetric,
     so the pair's rate is the relu of its upper-triangle entry) -- the non-local rate
     field of tab:rate-field, site by site;
  3. the closed-form channel sigma * Delta_aj(x) of eq:swap-log-ratio for the same anchor,
     Delta_aj = 2 (x_j - x_a)(h~_a - h~_j) with the hole-excluded fields -- the linear part the
     regression table scores the head against, drawn signed.
Rows 2 and 3 are the two things tab:local-field compares: how much of the learned rate is the
exact field, and what the non-local remainder looks like on the lattice.

Colour follows the job: the rate is a magnitude (one hue, light -> dark, the sampler blue);
the channel is signed (two poles about a neutral mid-grey); the state uses the house spin
colours. Runs on CPU in seconds at 8x8 (one head forward per displayed time, plus the rollout).
"""
import argparse
import json
from pathlib import Path

from discrete_flow_sampler.diagnostics.figure_style import (
    CLASSICAL_HUE, FONT_SIZE_ANNOTATION, FULL_WIDTH_IN, GRID, MUTED,
    SAMPLER_HUE, SAVEFIG_DPI, SPIN_CMAP, use_house_style)
from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from experiments.constrained_hard_03.gate_4x4 import load_run
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import torch


def channel_for_anchor(x, anchor, A, sigma):
    """sigma * Delta_aj for every j: the exact-field channel's score at unit gain."""
    field = x @ A
    h_a = field[anchor] - A[anchor] * x            # anchor's field with partner j removed
    h_j = field - A[anchor] * x[anchor]            # partner's field with the anchor removed
    delta = 2.0 * (x - x[anchor]) * (h_a - h_j)
    delta[anchor] = 0.0
    return sigma * delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--anchor", type=int, default=None,
                        help="anchor site index (default: the centre site)")
    parser.add_argument("--times", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("rate_field_strip.png"))
    args = parser.parse_args()
    use_house_style()
    torch.manual_seed(args.seed)

    head, target = load_run(args.run_dir, "cpu")
    cfg = json.loads((args.run_dir / "config.json").read_text())
    side, sigma = cfg["ising"]["D"], cfg["ising"]["sigma"]
    d = side * side
    n_euler = cfg["ctmc"]["n_euler_steps"]
    anchor = args.anchor if args.anchor is not None else (side // 2) * side + side // 2
    A = target.A.float()

    ts = torch.linspace(0.0, 1.0, n_euler + 1)
    x0 = target.sample_base(1, "cpu").float()
    with torch.no_grad():
        trajectory = sample_swap_ctmc(head, x0, ts, return_all_states=True, target=target,
                                      multi_event=bool(cfg["ctmc"].get("use_matching_step", False)))
    pairs = upper_tri_pairs(d, "cpu")
    columns = []
    for t_value in args.times:
        k = int(round(t_value * n_euler))
        x = trajectory[k, 0]
        with torch.no_grad():
            G = head(x.unsqueeze(0), ts[k].expand(1))[0]
        # G is index-antisymmetric (G[j,i] = -G[i,j]); the one-way rate of the unordered
        # pair {a, j} is the relu of the UPPER-triangle entry, so read G[min, max].
        js = torch.arange(d)
        signed = torch.where(js > anchor, G[anchor, js], G[js, anchor])
        rate = torch.relu(signed)
        rate[anchor] = 0.0
        total_rate = torch.relu(G[pairs[:, 0], pairs[:, 1]]).sum().item()
        channel = channel_for_anchor(x, anchor, A, sigma)
        columns.append(dict(t=ts[k].item(), x=x, rate=rate, channel=channel, total_rate=total_rate))

    rate_cmap = LinearSegmentedColormap.from_list("rate", ["#f4f4f1", SAMPLER_HUE])
    diverging = LinearSegmentedColormap.from_list("channel", [CLASSICAL_HUE, "#e6e5df", SAMPLER_HUE])
    rate_max = max(c["rate"].max().item() for c in columns)
    chan_max = max(c["channel"].abs().max().item() for c in columns)

    n = len(columns)
    fig, axes = plt.subplots(3, n, figsize=(FULL_WIDTH_IN, 0.78 * FULL_WIDTH_IN * 3 / n + 0.5),
                             gridspec_kw=dict(wspace=0.08, hspace=0.12))
    row_labels = ["state $x_t$", "learned rate\n$[G(a,j\\mid x,t)]_+$",
                  "closed form\n$\\sigma\\Delta_{aj}(x)$"]
    ar, ac = divmod(anchor, side)
    for col, c in enumerate(columns):
        grids = [c["x"].view(side, side).numpy(),
                 c["rate"].view(side, side).numpy(),
                 c["channel"].view(side, side).numpy()]
        for row, grid in enumerate(grids):
            ax = axes[row, col]
            if row == 0:
                ax.imshow(grid, cmap=SPIN_CMAP, vmin=-1, vmax=1, interpolation="nearest")
            elif row == 1:
                im_rate = ax.imshow(grid, cmap=rate_cmap, vmin=0, vmax=rate_max, interpolation="nearest")
            else:
                im_chan = ax.imshow(grid, cmap=diverging, norm=TwoSlopeNorm(0, -chan_max, chan_max),
                                    interpolation="nearest")
            ax.add_patch(plt.Rectangle((ac - 0.5, ar - 0.5), 1, 1, fill=False, lw=1.6,
                                       edgecolor="#1a1a19"))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=FONT_SIZE_ANNOTATION - 1)
        axes[0, col].set_title(f"$t = {c['t']:.2f}$   $\\Lambda = {c['total_rate']:.1f}$",
                               fontsize=FONT_SIZE_ANNOTATION, color=MUTED)
    cb1 = fig.colorbar(im_rate, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01)
    cb2 = fig.colorbar(im_chan, ax=axes[2, :].tolist(), fraction=0.02, pad=0.01)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=FONT_SIZE_ANNOTATION - 1)
        cb.outline.set_edgecolor(GRID)
    fig.savefig(args.out, dpi=SAVEFIG_DPI, bbox_inches="tight")

    # numbers for the caption: how non-local is the anchor's rate at the end?
    x_end, rate_end = columns[-1]["x"], columns[-1]["rate"]
    adjacent = A[anchor] > 0
    unlike = x_end != x_end[anchor]
    print(f"anchor {anchor} (row {ar}, col {ac}); Lambda along the strip: "
          + ", ".join(f"{c['total_rate']:.1f}" for c in columns))
    print(f"t=1: rate on the anchor's {int((adjacent & unlike).sum())} unlike neighbours "
          f"{rate_end[adjacent].sum():.3f} vs {int((~adjacent & unlike).sum())} unlike distant sites "
          f"{rate_end[~adjacent].sum():.3f}; corr(rate, relu(channel)) = "
          f"{np.corrcoef(rate_end.numpy(), torch.relu(columns[-1]['channel']).numpy())[0, 1]:.2f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
