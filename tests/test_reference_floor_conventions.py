"""An iid floor measures draw noise, even when every chain/block is balanced."""
import pytest
import torch

from experiments.constrained_soft_02.analysis import composition_marginal_overlay_8x8 as soft
from experiments.dnfs_baseline_01.analysis import unconstrained_clean_demo as baseline


def binary_pmf(samples, weights):
    return torch.stack((weights[samples[:, 0] < 0].sum(),
                        weights[samples[:, 0] > 0].sum()))


@pytest.mark.parametrize("module,width", [(soft, 64), (baseline, 100)])
def test_two_iid_draws_from_balanced_pool_have_known_tv(module, width, monkeypatch):
    # K ~ Binomial(2, 1/2); E|K/2 - 1/2| = 1/4. Every ten-frame
    # block is exactly balanced, so a block bootstrap would wrongly give zero.
    reference = torch.tensor([-1., 1.] * 10)[:, None].expand(-1, width)
    monkeypatch.setattr(module, "N_FLOOR_BOOTSTRAP", 1000)
    if module is soft:
        monkeypatch.setattr(module, "N_EVAL", 2)
        actual = module.reference_tv_floor(reference, binary_pmf)
    else:
        actual = module.reference_tv_floor(reference, 1, binary_pmf, n_draws=2)
    assert actual == pytest.approx(0.25, abs=0.025)


def test_split_uncertainty_keeps_independent_chains_intact():
    from experiments.constrained_hard_03.analysis.house_table_8x8 import reference_standard_error

    # Two constant but incompatible chains: site means differ by two.
    # dMag = 4 on a 2x2 grid; half the split distance must be 2, not zero
    # as it would be if frames were shuffled between the halves.
    chains = [-torch.ones(20, 4), torch.ones(20, 4)]
    result = reference_standard_error(chains, 2, n_splits=1)
    assert result["dMag"] == pytest.approx(2.0)
    assert result["dCorr"] == pytest.approx(0.0)


def test_hard_marginal_floor_excludes_chain_pool_uncertainty():
    from experiments.constrained_hard_03.analysis.hard_results_cell import _floor

    chains = [-torch.ones(10, 4), torch.ones(10, 4)]
    def pmf(samples):
        return binary_pmf(samples, torch.full((len(samples),), 1 / len(samples))).numpy()
    # A hierarchical chain bootstrap would add uncertainty in the pool and
    # yield 3/8, whereas two iid draws from the balanced pool give 1/4.
    assert _floor(chains, 2, 1000, 0, pmf) == pytest.approx(0.25, abs=0.025)
