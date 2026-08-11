"""Build a cross-size warm-start state dict for the swap head (d64 -> d256).

Why this exists: the 16x16 rung diverged from COLD initialisation -- ~96% of
its step-0 loss is the head's own init noise, an unnormalised coherent sum
over d(d-1)/2 pair scores (adversarial review 2026-08-11). The converged 8x8
checkpoint already encodes a near-Kolmogorov-consistent rate field, and the
target rate field is local and intensive per pair (the swap log-ratio is a
5-point stencil function), so its weights are the right starting basin at any
lattice size the architecture can address.

Why it is legitimate: 110 of 116 parameter tensors are shape-identical across
D (the backbone is a set-attention architecture over sites; band-family MLPs
are indexed by RELATIVE offset, so the delta=D "column neighbour" family maps
onto the new D's column neighbour with identical weights). The only
d-dependent tensors are learned positional tables, which live on the D x D
grid and transfer by bilinear interpolation -- the standard treatment for
resizing learned positional embeddings. Tables carrying a leading non-grid
row (the cond_t slot at row 0) keep that row verbatim. The one table the swap
head never reads (the backbone's attention_readout, a single-site-route
leftover) is deliberately NOT emitted, so the target model keeps its fresh
init there and `load_state_dict(strict=False)` reports it as missing -- that
printed report is the transfer's audit trail.

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
    """Bilinearly resize (d_src, C) grid-flattened rows to (d_tgt, C).

    Rows are raster-ordered over a sqrt(d) x sqrt(d) lattice; bilinear
    interpolation commutes with the 180-degree raster flip the bwd stack
    stores, so one helper serves fwd, bwd and pair tables alike.
    """
    D_src, D_tgt = int(d_src ** 0.5), int(d_tgt ** 0.5)
    assert D_src * D_src == d_src and D_tgt * D_tgt == d_tgt, "non-square grid"
    grid = rows.reshape(D_src, D_src, -1).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        grid, size=(D_tgt, D_tgt), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).permute(1, 2, 0).reshape(d_tgt, -1)


def build_transfer(source_sd: dict, target_sd: dict, d_src: int, d_tgt: int,
                   skip_prefixes: tuple[str, ...]) -> tuple[dict, list, list]:
    transfer, interpolated, skipped = {}, [], []
    for key, src in source_sd.items():
        if key not in target_sd:
            skipped.append((key, "absent in target"))
            continue
        tgt_shape = target_sd[key].shape
        if any(key.startswith(p) for p in skip_prefixes):
            skipped.append((key, "dead weight for the swap head; fresh init"))
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
    args = p.parse_args()

    src_cfg, tgt_cfg = CONFIGS[args.source_cfg], CONFIGS[args.target_cfg]
    d_src, d_tgt = src_cfg.ising.D ** 2, tgt_cfg.ising.D ** 2
    _, tgt_head = build_target_and_head(tgt_cfg, "cpu")
    source_sd = torch.load(args.source, map_location="cpu", weights_only=True)
    target_sd = tgt_head.state_dict()

    transfer, interpolated, skipped = build_transfer(
        source_sd, target_sd, d_src, d_tgt,
        skip_prefixes=("backbone.attention_readout.",),
    )
    # The transfer must be loadable and must leave nothing accidentally
    # fresh: every target key is either transferred or knowingly skipped.
    unaccounted = set(target_sd) - set(transfer) - {
        k for k in target_sd if any(
            k.startswith(p) for p in ("backbone.attention_readout.",))
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
