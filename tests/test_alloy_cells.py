"""The Cu-Au alloy cells build the right target on every rung.

Free and penalised rungs go through the baseline constructor with
`expansion_json` set; the canonical rung through the hard constructor with
`target_kind = "cluster_expansion"` and the geometry-free mask_one head. What
must hold: d comes from the file (16 or 64 sites), sigma is beta/2 at 1200 K
at the start of the curriculum and beta/2 at 500 K at its end, the penalty
and composition settings are the ones the name promises, and the hard head
emits antisymmetric pair scores of the right shape on the fcc cell.
"""
import pytest
import torch

from discrete_flow_sampler.targets.cluster_expansion import (
    ClusterExpansionTarget,
    FixedCompositionClusterExpansionTarget,
)
from experiments.constrained_hard_03.configs import CONFIGS as HARD_CONFIGS
from experiments.constrained_hard_03.configs import cuau_sigma
from experiments.constrained_hard_03.run import build_target_and_head
from experiments.constrained_soft_02.configs import CONFIGS as SOFT_CONFIGS
from experiments.dnfs_baseline_01.run import _construct_target

SIGMA_500K = 1.0 / (2.0 * 8.617333262e-5 * 500.0)


@pytest.mark.parametrize("sites", [16, 64])
def test_free_and_penalised_cells(sites):
    free = SOFT_CONFIGS[f"A1_cuau{sites}_T500_letf_{10 if sites == 16 else 50}k_curr"]
    target = _construct_target(free.ising, device="cpu")
    assert isinstance(target, ClusterExpansionTarget)
    assert target.d == sites
    assert free.ising.composition_penalty_strength == 0.0
    assert free.ising.target_composition is None
    assert free.curriculum.stages[-1].sigma == pytest.approx(SIGMA_500K)
    assert target.sigma == pytest.approx(cuau_sigma(1200.0))
    assert free.model.condition_on_composition is False

    soft = SOFT_CONFIGS[f"S2_cuau{sites}_c25_l50_T500_letf_{10 if sites == 16 else 50}k_curr"]
    target = _construct_target(soft.ising, device="cpu")
    assert soft.ising.composition_penalty_strength == 50.0
    assert soft.ising.target_composition == 0.25
    assert soft.ising.base_matches_composition
    x = target.sample_base(64, device="cpu")
    assert x.shape == (64, sites)
    assert torch.isfinite(target.log_prob(x)).all()


@pytest.mark.parametrize("sites", [16, 64])
def test_canonical_cells_build_an_antisymmetric_head(sites):
    cfg = HARD_CONFIGS[f"H2_cuau{sites}_c25_T500_mask_one_{10 if sites == 16 else 50}k_curr"]
    assert cfg.target_kind == "cluster_expansion"
    assert cfg.head_kind == "mask_one"
    target, head = build_target_and_head(cfg, device="cpu")
    assert isinstance(target, FixedCompositionClusterExpansionTarget)
    assert target.d == sites
    assert target.n_plus_target == sites // 4
    x = target.sample_base(3, device="cpu")
    target.assert_on_manifold(x)
    t = torch.full((3,), 0.5)
    with torch.no_grad():
        scores = head(x, t)
    assert scores.shape == (3, sites, sites)
    # the swap potential is antisymmetric under the STATE swap (the head's
    # defining property, not index transposition): G(i,j|x) = -G(i,j|Swap_ij x)
    unlike = (x[:, :, None] != x[:, None, :]).nonzero()[:6]
    for b, i, j in unlike.tolist():
        y = x.clone()
        y[b, i], y[b, j] = x[b, j], x[b, i]
        with torch.no_grad():
            swapped = head(y[b : b + 1], t[b : b + 1])
        assert swapped[0, i, j].item() == pytest.approx(-scores[b, i, j].item(), abs=1e-5)
