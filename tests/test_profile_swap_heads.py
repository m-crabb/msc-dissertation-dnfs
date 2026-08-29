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


def test_reported_peak_memory_excludes_the_resident_baseline():
    """A config benched after others must report its OWN peak.

    `reset_peak_memory_stats()` resets the peak but not the allocator, so
    `max_memory_allocated()` still counts whatever was alive at reset.
    Any caller that invokes `main` more than once in a process is exposed to
    this. It is not what caused the 2026-08-29 multi-config inflation, though
    -- subtracting the baseline moved the interval d=256 reading only from
    2.97 GB to 2.96 GB, and the remaining 1.2 GB over the 1.76 GB standalone
    figure was the eager fallback the compile-isolation tests below cover.
    """
    import inspect

    from experiments.constrained_hard_03 import profile_swap

    src = inspect.getsource(profile_swap._report)
    assert "baseline_bytes" in inspect.signature(profile_swap._report).parameters
    assert "- baseline_bytes" in src, "peak must be reported net of the baseline"

    caller = inspect.getsource(profile_swap.main)
    reset = caller.index("reset_peak_memory_stats")
    grab = caller.index("memory_allocated()", reset)
    report = caller.index("_report(", reset)
    assert reset < grab < report, (
        "the baseline must be captured AFTER the reset and BEFORE the timed run"
    )


def test_masked_attention_shares_the_interval_forward_code_object():
    """The fact the compile-isolation fix exists for.

    torch.compile caches per forward CODE OBJECT, not per module, so heads
    that inherit a forward share one `recompile_limit` budget across the
    whole process. `MaskedAttentionSwapHead` (and therefore the stencil
    family) inherits `IntervalSwapHead.forward`, which is why benching an MA
    row could silently de-optimise a later interval row. If a future MA head
    overrides forward this assertion should be updated, not deleted -- the
    grouping is what the isolation protocol is sized against.
    """
    from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
    from discrete_flow_sampler.constraints.masked_attention_swap_head import (
        MaskedAttentionSwapHead,
    )

    assert (MaskedAttentionSwapHead.forward.__code__
            is IntervalSwapHead.forward.__code__)


def test_compiled_configs_reset_dynamo_before_compiling():
    """Each benched row must start from clean dynamo state.

    Without the reset the budget is spent across configurations: a fresh head
    costs two cache entries (the lazily set `_causal_mask` invalidates the
    first trace), so the fifth interval-family configuration in a process
    exhausts the default limit of 8 and every later one runs EAGER while
    still printing compile=True. Measured on an A100 2026-08-29: 26.4 ms /
    2.96 GB against 10.1 ms / 1.76 GB for the same interval d=256 B=32 row.
    """
    import inspect

    from experiments.constrained_hard_03 import profile_swap

    src = inspect.getsource(profile_swap.main)
    reset = src.index("torch._dynamo.reset()")
    compile_call = src.index("head.compile()", reset)
    assert reset < compile_call, "reset must precede the compile it protects"


def test_compile_gave_up_watch_arms_once_and_rearms():
    import logging

    from experiments.constrained_hard_03.profile_swap import _COMPILE_WATCH

    logger = logging.getLogger("torch._dynamo")
    before = list(logger.handlers)
    try:
        _COMPILE_WATCH.arm()
        _COMPILE_WATCH.arm()
        assert logger.handlers.count(_COMPILE_WATCH) == 1, "handler piled up"

        record = logging.LogRecord(
            "torch._dynamo", logging.WARNING, __file__, 0,
            "torch._dynamo hit config.recompile_limit (8)", None, None,
        )
        logger.handle(record)
        assert _COMPILE_WATCH.gave_up, "the eager fallback must be detected"
        assert not _COMPILE_WATCH.arm().gave_up, "arming must clear the flag"
    finally:
        logger.handlers[:] = before


def test_bench_remote_isolates_each_config_in_a_subprocess_by_default():
    import inspect

    from experiments.constrained_hard_03 import modal_app

    fn = modal_app.bench_remote.get_raw_f()
    assert inspect.signature(fn).parameters["isolate"].default is True
    src = inspect.getsource(fn)
    assert "subprocess.run" in src and "profile_swap" in src
