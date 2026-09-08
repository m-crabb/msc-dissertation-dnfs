"""Build a cross-size warm-start state dict for the swap head (e.g. d64 -> d256).

The 16x16 rung diverged from cold initialisation: ~96% of its step-0 loss is
the head's own init noise, an unnormalised coherent sum over d(d-1)/2 pair
scores (review 2026-08-11). The converged 8x8 checkpoint already encodes a
near-Kolmogorov-consistent rate field, and that field is local and intensive
per pair (the swap log-ratio is a 5-point stencil function), so its weights are
the right starting basin at any lattice size the architecture can address.

All but a handful of parameter tensors are shape-identical across D. The
backbone is a set-attention architecture over sites, so every
Linear/attention/LayerNorm tensor is (hidden_dim, hidden_dim) shaped; the
band-family MLPs are indexed by relative offset, so the delta=D "column
neighbour" family maps onto the new D's column neighbour with identical
weights. The only d-dependent tensors are learned positional tables:

    backbone.{fwd,bwd}_stack.blocks.*.pos_embed   (1 + d, hidden_dim)
    backbone.attention_readout.pos_embed          (d, d_k)
    pair_position_embedding.weight   (interval / masked-attention heads)
    site_position_embedding.weight   (factorised head)         (d, position_dim)

Those live on the D x D grid and are resampled by `_resize_grid_rows`.
Tables carrying a leading non-grid row (the cond_t slot at row 0) keep that
row verbatim.

Resampling is bicubic with align_corners=False over a circularly padded grid,
because the lattice is a torus (see `_resize_grid_rows`).

`skip_prefixes` defaults to empty: transfer everything that fits.
`backbone.attention_readout.*` is the live compute path for the mask_one and
doubly_hollow heads, so dropping it there would leave the only module that
reads the lattice at fresh init on top of a fully transferred trunk; for the
one-pass band heads those tensors are inert and cost only checkpoint bytes.

The archived 2026-08-11 d64 -> d256 transfer predates these defaults: it was
built with bilinear/edge resampling and with the attention_readout tensors
skipped, so it is not reproduced byte-for-byte here; that checkpoint file is
its own record.

Usage:
    pixi run python -m scripts.warm_start_swap_head \
        --source results/03_hard/<d64_run>/checkpoints/final.pt \
        --source_cfg H2_d64_c50_s223_letf_ma_100k_curr \
        --target_cfg H2_d256_smoke12k_warm \
        --out results/03_hard/warm_start_d64_to_d256.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.run import build_target_and_head


def _resize_grid_rows(rows: torch.Tensor, d_src: int, d_tgt: int) -> torch.Tensor:
    """Resample (d_src, C) grid-flattened rows to (d_tgt, C) on the torus.

    Rows are raster-ordered over a sqrt(d) x sqrt(d) lattice. Interpolation is
    bicubic with align_corners=False, the operation used to transfer learned
    ViT position embeddings between input resolutions: a position table is a
    smooth field read at new sample points, where the cubic kernel's continuous
    first derivative avoids the faceting bilinear leaves at the original grid
    lines, and align_corners=False keeps the mapping a pure rescaling of pixel
    centres.

    The lattice has periodic boundary conditions, so the field being resampled
    is periodic and no site is distinguished; PyTorch's default edge handling
    replicates the boundary row/column into the kernel's support and would
    manufacture such a distinguished edge. Circular padding is implemented by
    tiling the grid 3x3 and cropping the centre after resampling: with
    align_corners=False the scale factor is unchanged by the tiling
    (3*L_dst / 3*L_src), so the centre block sees exactly the periodic
    extension of the source at every sample point, for any L_src -> L_dst
    ratio. Padding by a fixed 2-pixel collar is only correct for integer scale
    factors.

    Commutes with the 180-degree raster reversal -- the kernel is symmetric
    and the sampling grid is centred -- which is what makes it correct to
    resample the bwd stack's table, whose site rows run in reversed raster
    order because that stack reads x.flip(1), as an ordinary grid.

    Callers must strip any leading conditioning row before calling this; the
    assertion below catches only a non-square row count, not a mis-sliced
    one.
    """
    side_src, side_tgt = int(round(d_src ** 0.5)), int(round(d_tgt ** 0.5))
    assert side_src ** 2 == d_src and side_tgt ** 2 == d_tgt, "non-square grid"
    if side_src == side_tgt:
        return rows
    grid = rows.reshape(side_src, side_src, -1).permute(2, 0, 1).unsqueeze(0)
    tiled = grid.repeat(1, 1, 3, 3)
    resized = F.interpolate(
        tiled, size=(3 * side_tgt, 3 * side_tgt), mode="bicubic",
        align_corners=False,
    )[:, :, side_tgt:2 * side_tgt, side_tgt:2 * side_tgt]
    return resized.squeeze(0).permute(1, 2, 0).reshape(d_tgt, -1)


def build_transfer(
    source_sd: dict,
    target_sd: dict,
    d_src: int,
    d_tgt: int,
    skip_prefixes: tuple[str, ...] = (),
) -> tuple[dict, list, list]:
    """Adapt `source_sd` (lattice d_src) to load into `target_sd` (d_tgt).

    Returns (transfer, interpolated_keys, skipped). Shape-identical tensors
    are copied verbatim; a 2-D tensor whose leading dim grows by exactly
    d_tgt - d_src is treated as a positional table over the lattice and
    resampled, with a leading (d_src + 1)-row form understood as carrying a
    non-site conditioning row at index 0 that is passed through untouched.
    d_src == d_tgt therefore takes the copy branch for every tensor, making
    same-size transfer bit-exact by construction rather than by tolerance.
    """
    transfer, interpolated, skipped = {}, [], []
    for key, src in source_sd.items():
        if key not in target_sd:
            skipped.append((key, "absent in target"))
            continue
        tgt_shape = target_sd[key].shape
        if any(key.startswith(p) for p in skip_prefixes):
            skipped.append((key, "caller-skipped; left at fresh init"))
            continue
        if src.shape == tgt_shape:
            transfer[key] = src.clone()
        elif (
            src.dim() == 2 and src.shape[1] == tgt_shape[1]
            and src.shape[0] in (d_src, d_src + 1)
            and tgt_shape[0] - src.shape[0] == d_tgt - d_src
        ):
            has_cond_row = src.shape[0] == d_src + 1
            grid_rows = src[1:] if has_cond_row else src
            resized = _resize_grid_rows(grid_rows, d_src, d_tgt)
            transfer[key] = (
                torch.cat([src[:1], resized]) if has_cond_row else resized
            )
            interpolated.append(key)
        else:
            skipped.append((key, f"unhandled shape {tuple(src.shape)} -> "
                                 f"{tuple(tgt_shape)}"))
    return transfer, interpolated, skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source_cfg", required=True, choices=list(CONFIGS))
    p.add_argument("--target_cfg", required=True, choices=list(CONFIGS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--skip-prefix", action="append", default=[], metavar="PREFIX",
        help="Leave every parameter under PREFIX at fresh init (repeatable). "
             "Default: transfer everything that fits.",
    )
    args = p.parse_args()

    src_cfg, tgt_cfg = CONFIGS[args.source_cfg], CONFIGS[args.target_cfg]
    d_src, d_tgt = src_cfg.ising.D ** 2, tgt_cfg.ising.D ** 2
    _, tgt_head = build_target_and_head(tgt_cfg, "cpu")
    source_sd = torch.load(args.source, map_location="cpu", weights_only=True)
    target_sd = tgt_head.state_dict()

    skip_prefixes = tuple(args.skip_prefix)
    transfer, interpolated, skipped = build_transfer(
        source_sd, target_sd, d_src, d_tgt, skip_prefixes=skip_prefixes,
    )
    # The transfer must be loadable and must leave nothing accidentally
    # fresh: every target key is either transferred or knowingly skipped.
    unaccounted = set(target_sd) - set(transfer) - {
        k for k in target_sd if any(k.startswith(p) for p in skip_prefixes)
    }
    if unaccounted:
        raise SystemExit(f"target keys neither transferred nor knowingly "
                         f"skipped: {sorted(unaccounted)}")

    print(f"copied {len(transfer) - len(interpolated)} shape-identical keys")
    for key in interpolated:
        print(f"interpolated {key}: {tuple(source_sd[key].shape)} -> "
              f"{tuple(target_sd[key].shape)}")
    for key, why in skipped:
        print(f"skipped {key}: {why}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(transfer, args.out)
    print(f"wrote {args.out} ({len(transfer)} keys)")


if __name__ == "__main__":
    main()
