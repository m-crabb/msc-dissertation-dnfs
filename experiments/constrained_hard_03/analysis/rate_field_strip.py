"""One trajectory of the trained swap process, base to terminal state, with the
learned rate field drawn at a few grid times.

Approved 20x20 figure, without rerunning the model:
    python -m experiments.constrained_hard_03.analysis.rate_field_strip
        --recorded assets/hard_rate_field_strip_20x20.npz --out figure.pdf
The archive includes the displayed frames and checkpoint/rollout provenance.
For the dense README recording, use hard_rate_field_strip_24x24.npz with
--recorded-stride 32 to display five times.

Three rows per displayed time t_k of one rollout from a uniform-on-slice base:
  1. the state x_{t_k}, with a fixed anchor site marked;
  2. the learned one-way rate [G(a, j | x, t)]_+ for swapping anchor a with site j
     (zero on like-spin partners by antisymmetry; G is index-antisymmetric, so
     the pair's rate is the relu of its upper-triangle entry) -- the non-local
     rate field of tab:rate-field, site by site;
  3. the closed-form channel sigma * Delta_aj(x) of eq:swap-log-ratio,
     Delta_aj = 2 (x_j - x_a)(h~_a - h~_j), using hole-excluded fields -- the
     linear part the regression table scores the head against, drawn signed.
The closed form describes the terminal target, not the time-t path ratio,
which has another factor t. It is not a rate: learned swaps may raise energy.

The rate is a magnitude (light -> dark sampler blue); the channel is signed
(two poles about neutral mid-grey); the state uses the house spin colours.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.constrained_hard_03.gate_4x4 import load_run
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from discrete_flow_sampler.diagnostics.figure_style import (
    CLASSICAL_HUE,
    FONT_SIZE_ANNOTATION,
    FULL_WIDTH_IN,
    GRID,
    SAMPLER_HUE,
    SAVEFIG_DPI,
    SPIN_CMAP,
    use_house_style,
)
from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc


def channel_for_anchor(x, anchor, A, sigma):
    """sigma * Delta_aj for every j: the exact-field channel's score at unit gain."""
    field = x @ A
    h_a = field[anchor] - A[anchor] * x  # anchor's field with partner j removed
    h_j = field - A[anchor] * x[anchor]  # partner's field with the anchor removed
    delta = 2.0 * (x - x[anchor]) * (h_a - h_j)
    delta[anchor] = 0.0
    return sigma * delta


SWAP_ROW_LABELS = ["Configuration", "Learned\nswap rate", "Closed-form\nlog ratio"]
FLIP_ROW_LABELS = ["Configuration", "Learned\nflip rate", "Closed-form\nlog ratio"]


def plot_strip(columns, side, anchor, out, row_labels=SWAP_ROW_LABELS):
    """State, learned rate and terminal log ratio; explanations live in the caption.

    `anchor=None` draws no marked site: the flip-family strip, where the rate
    and channel rows are per-site fields rather than one anchor's partners.

    Each row uses a common scale across time. The two fields have different
    units and separate colour scales. Read unordered-pair rates before calling
    this function; G[a,j] alone has the wrong sign whenever j < a.
    """
    use_house_style()
    rate_cmap = LinearSegmentedColormap.from_list("rate", ["#f4f4f1", SAMPLER_HUE])
    diverging = LinearSegmentedColormap.from_list(
        "channel", [CLASSICAL_HUE, "#e6e5df", SAMPLER_HUE]
    )
    rate_max = max(float(np.asarray(c["rate"]).max()) for c in columns)
    chan_max = max(float(np.abs(np.asarray(c["channel"])).max()) for c in columns)
    n = len(columns)
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 3.65))
    grid = fig.add_gridspec(
        3,
        n + 1,
        width_ratios=[1] * n + [0.055],
        left=0.13,
        right=0.92,
        bottom=0.04,
        top=0.93,
        wspace=0.10,
        hspace=0.15,
    )
    anchor_row, anchor_col = divmod(anchor, side) if anchor is not None else (0, 0)
    for row, key in enumerate(("x", "rate", "channel")):
        for col, values in enumerate(columns):
            ax = fig.add_subplot(grid[row, col])
            field = np.asarray(values[key]).reshape(side, side)
            if row == 0:
                im = ax.imshow(
                    field, cmap=SPIN_CMAP, vmin=-1, vmax=1, interpolation="nearest"
                )
            elif row == 1:
                im = ax.imshow(
                    field,
                    cmap=rate_cmap,
                    vmin=0,
                    vmax=max(rate_max, 1e-12),
                    interpolation="nearest",
                )
            else:
                limit = max(chan_max, 1e-12)
                im = ax.imshow(
                    field,
                    cmap=diverging,
                    norm=TwoSlopeNorm(0, -limit, limit),
                    interpolation="nearest",
                )
            anchor_outline = (
                [("white", 2.4), ("#1a1a19", 1.2)] if anchor is not None else []
            )
            for colour, width in anchor_outline:
                ax.add_patch(
                    plt.Rectangle(
                        (anchor_col - 0.5, anchor_row - 0.5),
                        1,
                        1,
                        fill=False,
                        lw=width,
                        edgecolor=colour,
                    )
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)
            if col == 0:
                ax.set_ylabel(
                    row_labels[row], fontsize=FONT_SIZE_ANNOTATION, labelpad=8
                )
            if row == 0:
                ax.set_title(f"$t = {values['t']:.2f}$", fontsize=FONT_SIZE_ANNOTATION)
        if row:
            colourbar = fig.colorbar(im, cax=fig.add_subplot(grid[row, n]))
            colourbar.ax.tick_params(labelsize=FONT_SIZE_ANNOTATION - 1)
            colourbar.outline.set_edgecolor(GRID)
    fig.savefig(out, dpi=SAVEFIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument(
        "--recorded",
        type=Path,
        help="render a saved figure archive without sampling a checkpoint",
    )
    parser.add_argument(
        "--recorded-stride",
        type=int,
        default=1,
        help="display every Nth archived frame (default: all)",
    )
    parser.add_argument(
        "--recorded-frames",
        type=int,
        nargs="+",
        help="archived frame indices to display, instead of --recorded-stride",
    )
    parser.add_argument(
        "--anchor",
        type=int,
        default=None,
        help="anchor site index (default: the centre site)",
    )
    parser.add_argument(
        "--times", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("rate_field_strip.png"))
    args = parser.parse_args()
    if args.recorded_stride < 1:
        parser.error("--recorded-stride must be positive")
    if args.recorded is not None:
        if args.run_dir is not None:
            parser.error("choose a run directory or --recorded, not both")
        with np.load(args.recorded, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            frames = args.recorded_frames or list(
                range(0, len(data["times"]), args.recorded_stride)
            )
            columns = [
                dict(
                    t=data["times"][k],
                    x=data["states"][k],
                    rate=data["rates"][k],
                    channel=data["channels"][k],
                )
                for k in frames
            ]
        anchor = metadata.get("anchor")
        plot_strip(
            columns,
            metadata["side"],
            anchor,
            args.out,
            row_labels=SWAP_ROW_LABELS if anchor is not None else FLIP_ROW_LABELS,
        )
        print(f"saved {args.out} from {args.recorded}")
        return
    if args.run_dir is None:
        parser.error("provide a run directory or --recorded")
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
        trajectory = sample_swap_ctmc(
            head,
            x0,
            ts,
            return_all_states=True,
            target=target,
            multi_event=bool(cfg["ctmc"].get("use_matching_step", False)),
        )
    pairs = upper_tri_pairs(d, "cpu")
    columns = []
    for t_value in args.times:
        k = int(round(t_value * n_euler))
        x = trajectory[k, 0]
        with torch.no_grad():
            G = head(x.unsqueeze(0), ts[k].expand(1))[0]
        # G is index-antisymmetric (G[j,i] = -G[i,j]); the one-way rate of the unordered
        # pair {a, j} is the relu of the upper-triangle entry, so read G[min, max].
        js = torch.arange(d)
        signed = torch.where(js > anchor, G[anchor, js], G[js, anchor])
        rate = torch.relu(signed)
        rate[anchor] = 0.0
        total_rate = torch.relu(G[pairs[:, 0], pairs[:, 1]]).sum().item()
        channel = channel_for_anchor(x, anchor, A, sigma)
        columns.append(
            dict(t=ts[k].item(), x=x, rate=rate, channel=channel, total_rate=total_rate)
        )

    plot_strip(columns, side, anchor, args.out)

    # numbers for the caption: how non-local is the anchor's rate at the end?
    x_end, rate_end = columns[-1]["x"], columns[-1]["rate"]
    adjacent = A[anchor] > 0
    unlike = x_end != x_end[anchor]
    correlation = np.corrcoef(
        rate_end.numpy(), torch.relu(columns[-1]["channel"]).numpy()
    )[0, 1]
    print(
        f"anchor {anchor} (row {anchor // side}, col {anchor % side}); "
        "Lambda along the strip: "
        + ", ".join(f"{c['total_rate']:.1f}" for c in columns)
    )
    print(
        f"t=1: rate on the anchor's {int((adjacent & unlike).sum())} unlike neighbours "
        f"{rate_end[adjacent].sum():.3f} vs {int((~adjacent & unlike).sum())} "
        "unlike distant sites "
        f"{rate_end[~adjacent].sum():.3f}; corr(rate, relu(channel)) = "
        f"{correlation:.2f}"
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
