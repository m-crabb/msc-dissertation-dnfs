"""Animate recorded grid states of the README's 24x24 swap trajectory.

Reads the recorded arrays only: no checkpoint loading, sampling or interpolation.
The rate panel uses one colour scale across all times. These are proposal-path
snapshots, not importance-resampled draws from the target distribution.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from discrete_flow_sampler.diagnostics.figure_style import (
    GRID,
    REFERENCE_INK,
    SAMPLER_HUE,
    SPIN_CMAP,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded",
        type=Path,
        default=Path("assets/hard_rate_field_strip_24x24.npz"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=None,
        help="frame duration (default: 80 ms for dense recordings, 1200 ms otherwise)",
    )
    args = parser.parse_args()
    if args.out.suffix.lower() != ".gif":
        parser.error("--out must have a .gif extension")
    if args.frame_ms is not None and args.frame_ms < 10:
        parser.error("--frame-ms must be at least 10")

    with np.load(args.recorded, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        states, rates, times = data["states"], data["rates"], data["times"]
    side = metadata["side"]
    if states.shape != (len(times), side * side) or rates.shape != states.shape:
        raise ValueError("Recorded states and rates must match the lattice and times")
    if not np.isin(states, [-1, 1]).all() or not np.isfinite(rates).all():
        raise ValueError("Expected binary spins and finite rates")
    if (rates < 0).any() or not (np.diff(times) > 0).all():
        raise ValueError("Expected nonnegative rates and increasing times")
    counts = (states == 1).sum(axis=1)
    if not (counts == counts[0]).all():
        raise ValueError("Recorded trajectory does not preserve composition")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.1), dpi=100)
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.19, top=0.80, wspace=0.15)
    fig.patch.set_facecolor("white")
    title = fig.suptitle("", fontsize=20, color=REFERENCE_INK, y=0.96)
    fig.text(
        0.5,
        0.865,
        f"{len(times)} recorded states from one neural swap trajectory",
        ha="center",
        fontsize=12,
        color=REFERENCE_INK,
    )
    state_image = axes[0].imshow(
        states[0].reshape(side, side),
        cmap=SPIN_CMAP,
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    rate_cmap = LinearSegmentedColormap.from_list("rate", ["#f4f4f1", SAMPLER_HUE])
    rate_image = axes[1].imshow(
        rates[0].reshape(side, side),
        cmap=rate_cmap,
        vmin=0,
        vmax=max(float(rates.max()), 1e-12),
        interpolation="nearest",
    )
    anchor_row, anchor_col = divmod(metadata["anchor"], side)
    for ax, label in zip(
        axes, ["Configuration", "Swap rates from the marked site"], strict=True
    ):
        ax.set_title(label, fontsize=12, pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        for colour, width in [("white", 2.5), (REFERENCE_INK, 1.2)]:
            ax.add_patch(
                plt.Rectangle(
                    (anchor_col - 0.5, anchor_row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=colour,
                    linewidth=width,
                )
            )
    colourbar = fig.colorbar(rate_image, cax=fig.add_axes([0.925, 0.19, 0.015, 0.61]))
    colourbar.ax.tick_params(labelsize=10)
    fig.text(
        0.5,
        0.095,
        f"{side} × {side} Ising lattice  ·  {counts[0]} gold / "
        f"{side * side - counts[0]} indigo sites at every time",
        ha="center",
        fontsize=12,
        color=REFERENCE_INK,
    )
    fig.text(
        0.5,
        0.04,
        "Recorded grid states; no interpolation. Loop restarts at t = 0.",
        ha="center",
        fontsize=10,
        color=REFERENCE_INK,
    )
    frames = []
    for state, rate, time in zip(states, rates, times, strict=True):
        state_image.set_data(state.reshape(side, side))
        rate_image.set_data(rate.reshape(side, side))
        title.set_text(
            f"Fixed composition, evolving configurations   |   t = {time:.2f}"
        )
        fig.canvas.draw()
        frames.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
        )
    frame_ms = args.frame_ms or (80 if len(frames) > 10 else 1200)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=[frame_ms] * (len(frames) - 1) + [2400],
        loop=0,
        disposal=2,
    )
    plt.close(fig)
    print(f"Saved {len(frames)} recorded frames to {args.out}")


if __name__ == "__main__":
    main()
