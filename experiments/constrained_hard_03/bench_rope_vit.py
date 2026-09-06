"""Cost of the periodic-RoPE / patch-key backbone against leTF, behind the
fimo2 head (factorised + prefix band + row/col orderings), plus the two-hole
patch head on leTF (which never runs the stacks), at the production
width (hidden 32, 2 layers, 4 heads): parameters, counted forward FLOPs, and
CPU wall time of one head forward. CPU only, small batch; the numbers are
per-sample and relative, not a GPU throughput claim.

    pixi run -e dev python -m experiments.constrained_hard_03.bench_rope_vit
"""

import argparse
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from discrete_flow_sampler.constraints.factorised_swap_head import FactorisedSwapHead
from discrete_flow_sampler.constraints.two_hole_patch_swap_head import (
    TwoHolePatchSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.models.rope_vit import RoPEViTRateMatrix


def _analytic_attention_gflops(backbone, lattice_side, n_layers=2, hidden=32):
    """Score + value matmuls of one causal stack, 2 * 2 * queries * keys *
    hidden per layer: leTF keys = 1 + d (dense causal); RoPE keys =
    1 + pL + d / p^2 (cond + near window + far patches). Counted by hand
    because the FLOP counter does not see nn.MultiheadAttention's fused
    kernel, so the counted column below excludes attention for leTF only."""
    d = lattice_side**2
    if isinstance(backbone, RoPEViTRateMatrix):
        keys = 1 + backbone.patch_size * lattice_side + d // backbone.patch_size**2
    else:
        keys = 1 + d
    return n_layers * 4 * d * keys * hidden / 1e9


def _fimo2(backbone, lattice_side):
    return FactorisedSwapHead(
        backbone,
        interior_band="prefix",
        site_orderings=("row", "col"),
        pair_offsets=(1, lattice_side),
        lattice_side=lattice_side,
    ).eval()


def _arms(lattice_side):
    """(name, backbone, head) triples: fimo2 on leTF, fimo2 on RoPE p=1/2/4,
    and the two-hole patch head (R=1) on leTF -- the three rows of the
    symmetry-heads table on ONE harness (head-total FLOPs at the same batch)."""
    d = lattice_side * lattice_side
    common = dict(d=d, vocab_size=2, hidden_dim=32, n_layers=2, n_heads=4)
    torch.manual_seed(0)
    letf = LeTFRateMatrix(**common)
    yield "letf", letf, _fimo2(letf, lattice_side)
    for patch_size in (1, 2, 4):
        if lattice_side % patch_size == 0:
            torch.manual_seed(0)
            rope = RoPEViTRateMatrix(patch_size=patch_size, **common)
            yield f"rope_p{patch_size}", rope, _fimo2(rope, lattice_side)
    torch.manual_seed(0)
    yield (
        "thp_R1",
        letf,
        TwoHolePatchSwapHead(
            letf, lattice_side=lattice_side, patch_radius=1, feature_dim=32
        ).eval(),
    )


@torch.no_grad()
def bench(lattice_side: int, batch: int, repeats: int):
    d = lattice_side * lattice_side
    torch.manual_seed(1)
    x = (torch.randint(0, 2, (batch, d)) * 2 - 1).float()
    t = torch.rand(batch)
    print(
        f"\n=== {lattice_side}x{lattice_side} (d={d}), batch {batch}, CPU threads {torch.get_num_threads()} ==="
    )
    print(
        f"{'arm':10s} {'backbone params':>16s} {'head total':>11s} {'stack GFLOP/sample':>19s} {'head GFLOP/sample':>18s} {'attn GFLOP analytic':>20s} {'ms/sample':>10s}"
    )
    for name, backbone, head in _arms(lattice_side):
        backbone_params = sum(p.numel() for p in backbone.parameters())
        total_params = sum(p.numel() for p in head.parameters())
        # Counted in train mode: eval-mode nn.MultiheadAttention takes the
        # fused fast path, which the counter does not see (dropout is 0, so
        # the arithmetic is identical either way).
        head.train()
        with FlopCounterMode(display=False) as stack_flops:
            backbone.fwd_stack(torch.randn(batch, 1 + d, 32))
        with FlopCounterMode(display=False) as head_flops:
            head(x, t)
        head.eval()
        head(x, t)  # warm
        start = time.perf_counter()
        for _ in range(repeats):
            head(x, t)
        ms = (time.perf_counter() - start) / repeats / batch * 1e3
        print(
            f"{name:10s} {backbone_params:16,d} {total_params:11,d} "
            f"{stack_flops.get_total_flops() / batch / 1e9:19.4f} "
            f"{head_flops.get_total_flops() / batch / 1e9:18.4f} "
            f"{_analytic_attention_gflops(backbone, lattice_side):20.4f} {ms:10.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sides", type=int, nargs="+", default=[8, 16])
    args = parser.parse_args()
    for side in args.sides:
        bench(side, args.batch, args.repeats)
