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

# Bench --head_kind -> the cell whose cost it claims to measure. The cfg
# contributes head knobs, not dimensions (the backbone is shared), so the
# factorised row can pin against the 4x4 fab8 gate cell while no factorised
# d64 cell exists yet.
BENCH_KIND_TO_CELL = {
    "mask_one": "H2_d64_c50_s223_letf_mo_50k_curr",
    "interval": "H2_d64_c50_s223_letf_iv_50k_curr",
    "masked_attention": "H2_d64_c50_s223_letf_ma_50k_curr",
    "stencil": "H2_d64_c50_s223_letf_ma_stencil_50k_curr",
    "factorised": "H2_d16_c50_s223_letf_fab8_10k",
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


def test_naive_bench_head_is_the_doubly_hollow_oracle():
    """The naive rung has no trained cell (mask_one is bit-exact equal, so
    it never ships) — its bench claim is the ORACLE's cost, so the pin is
    class identity + shared backbone rather than a cell schema."""
    from discrete_flow_sampler.constraints.swap_readout import (
        DoublyHollowSwapHead,
    )

    head, _ = build_head_and_target(
        d=64, device=torch.device("cpu"), anchor_chunk=None, head_kind="naive"
    )
    assert type(head) is DoublyHollowSwapHead
    assert head.backbone.d == 64


@pytest.mark.parametrize("head_kind", ["interval", "masked_attention"])
def test_site_orderings_reach_the_raster_bench_heads(head_kind):
    """The sweep axis of the cost table must actually reach the head.

    Until 2026-08-29 `site_orderings` was forwarded to the factorised branch
    alone, so `--head-kind interval --site-orderings row,col` built a
    ONE-sweep head, exited 0, and printed `site_orderings=row,col` in the
    run banner. A cost table filled from that would have priced `ivmo2` /
    `mamo2` at their one-sweep parents' cost and made the second backbone
    pass look free. Mirrors test_raster_head_orderings'
    `test_config_flag_reaches_the_raster_heads` for the config path.
    """
    head, _ = build_head_and_target(
        d=64, device=torch.device("cpu"), anchor_chunk=None,
        head_kind=head_kind, site_orderings=("row", "col"),
    )
    assert head.site_orderings == ("row", "col")
    # The permutation buffer is what an extra sweep actually costs; a head
    # that recorded the tuple but registered nothing would still be a null.
    assert hasattr(head, "_order_col")


@pytest.mark.parametrize("head_kind", ["interval", "masked_attention"])
def test_one_sweep_stays_the_default_on_the_bench(head_kind):
    """Every archived bench row was taken at one sweep; the fix must not
    silently re-price them."""
    head, _ = build_head_and_target(
        d=64, device=torch.device("cpu"), anchor_chunk=None, head_kind=head_kind,
    )
    assert head.site_orderings == ("row",)
    assert not hasattr(head, "_order_col")
