"""Bench-vs-production head parity for the eval-time repricing bench.

profile_swap's cost table is only evidence about the production heads if
`build_head_and_target` constructs the same modules the d64 cells train.
These tests pin that: for each benchable head kind, the bench head must have
the same class and state-dict schema (keys and shapes) as the head
`configs.build_swap_head` builds from the corresponding 50k-curriculum cell.
Schema comparison is deliberate — e.g. forgetting `use_stencil=True` drops an
attention family and the `band_stencil_features` MLP, so the key set itself
diverges and the mismatch is caught structurally.
"""
import pytest
import torch

from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head
from experiments.constrained_hard_03.profile_swap import build_head_and_target

# Bench --head_kind -> the d64 cell whose cost it claims to measure.
BENCH_KIND_TO_CELL = {
    "mask_one": "H2_d64_c50_s223_letf_mo_50k_curr",
    "interval": "H2_d64_c50_s223_letf_iv_50k_curr",
    "masked_attention": "H2_d64_c50_s223_letf_ma_50k_curr",
    "stencil": "H2_d64_c50_s223_letf_ma_stencil_50k_curr",
}


@pytest.mark.parametrize("bench_kind,cell_name", BENCH_KIND_TO_CELL.items())
def test_bench_head_matches_production_cell(bench_kind, cell_name):
    device = torch.device("cpu")
    bench_head, _ = build_head_and_target(
        d=64, device=device, anchor_chunk=None, head_kind=bench_kind
    )
    cfg = CONFIGS[cell_name]
    production_head = build_swap_head(cfg, bench_head.backbone)

    assert type(bench_head) is type(production_head)
    bench_schema = {k: v.shape for k, v in bench_head.state_dict().items()}
    production_schema = {
        k: v.shape for k, v in production_head.state_dict().items()
    }
    assert bench_schema == production_schema


def test_stencil_bench_head_routes_stencil_band():
    head, _ = build_head_and_target(
        d=64, device=torch.device("cpu"), anchor_chunk=None, head_kind="stencil"
    )
    assert head.use_stencil
    assert head.stencil_side == 8
