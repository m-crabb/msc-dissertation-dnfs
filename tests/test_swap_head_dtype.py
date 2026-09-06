"""Swap-head dtype contract: pair scores stay fp32 under bf16 autocast.

Regression for a d64 twin-cell crash: einsum sits on autocast's
lower-precision list, so an einsum readout emits bf16 G inside the
eval_autocast_bf16 block -- crashing the fp32-only torch.quantile rate
diagnostic at step 0 and departing from the dtype path the bf16-eval flags were
validated on. The mask-one readout never had the bug because elementwise
mul+sum is not autocast-listed. The D=4 gate cells and the smoke configs run
with the flag OFF, so only a d64-flag run exercises this path -- hence a
dedicated CPU-autocast test rather than relying on cell smokes.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import LeTFMaskOneSwapHead
from discrete_flow_sampler.models.letf import LeTFRateMatrix


def _backbone(d=9):
    torch.manual_seed(42)
    return LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=8, n_layers=2, n_heads=2,
        use_sdpa_readout=False,
    )


@pytest.mark.parametrize(
    "build_head",
    [
        pytest.param(lambda b: LeTFMaskOneSwapHead(b), id="mask_one"),
        pytest.param(lambda b: IntervalSwapHead(b, pair_offsets=(1, 3)), id="interval"),
        pytest.param(
            lambda b: MaskedAttentionSwapHead(b, pair_offsets=(1, 3)),
            id="masked_attention",
        ),
    ],
)
@torch.no_grad()
def test_pair_scores_fp32_under_bf16_autocast(build_head):
    head = build_head(_backbone())
    head.eval()
    torch.manual_seed(1)
    x = (torch.randint(0, 2, (2, 9)) * 2 - 1).float()
    x[:, 0], x[:, 1] = 1.0, -1.0  # guarantee active pairs
    t = torch.rand(2)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        G = head(x, t)

    assert G.dtype == torch.float32, f"G is {G.dtype} under bf16 autocast"
    # The exact call that crashed the d64 run's step-0 diagnostics.
    torch.quantile(torch.relu(G).reshape(-1), 0.99)
