"""Pins for the 4x4 demo analysis (demo_4x4.py): the N_eff estimator on
deterministic inputs, phi's exact symmetry through the enumerated moments,
and the pass-row counter's mask_one vs masked_attention ratio.
"""
import numpy as np
import pytest
import torch

from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head
from experiments.constrained_hard_03.demo_4x4 import (
    BackbonePassRowCounter,
    exact_moments,
    n_eff_observable,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def test_n_eff_observable_deterministic():
    """Var 4.0, two estimates at +/-2 from truth -> MSE 4 -> N_eff exactly 1;
    equal leave-one-out MSEs -> jackknife SE exactly 0."""
    estimates = np.array([12.0, 8.0])
    n_eff, se = n_eff_observable(estimates, exact_mean=10.0, exact_var=4.0)
    assert n_eff == pytest.approx(1.0)
    assert se == pytest.approx(0.0)


def test_exact_phi_mean_is_zero_by_symmetry():
    """E_pi[phi] = 0 on the 50/50 slice: the global spin flip is a slice
    bijection that preserves energy and negates phi."""
    target = FixedCompositionIsingTarget(
        D=4, sigma=0.223, target_composition=0.5, bias=0.0, device="cpu"
    )
    moments = exact_moments(target)
    phi_mean, phi_var = moments["phi"]
    assert abs(phi_mean) < 1e-6
    assert phi_var > 0.0


def _tiny_head(head_kind):
    cfg_key = (
        "H2_d16_c50_s010_letf_ma_10k" if head_kind == "masked_attention"
        else "H2_d16_c50_s010_letf_mo_10k"
    )
    backbone = LeTFRateMatrix(
        d=16, vocab_size=2, hidden_dim=32, n_layers=2, n_heads=4
    )
    return build_swap_head(CONFIGS[cfg_key], backbone), backbone


def test_pass_row_counter_mask_one_is_d_times_masked_attention():
    """The network-pass currency counts rows through fwd_stack: mask_one
    stacks d anchor passes per head call, masked_attention runs one."""
    batch, d = 3, 16
    x = torch.ones(batch, d)
    x[:, : d // 2] = -1.0
    t = torch.full((batch,), 0.5)
    rows = {}
    for head_kind in ("masked_attention", "mask_one"):
        head, backbone = _tiny_head(head_kind)
        counter = BackbonePassRowCounter(backbone)
        with torch.no_grad():
            head(x, t)
        counter.close()
        rows[head_kind] = counter.rows
    assert rows["masked_attention"] == batch
    assert rows["mask_one"] == d * batch
