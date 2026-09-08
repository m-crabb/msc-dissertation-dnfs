"""Matched-base amortised soft design.

The flag `base_matches_composition` on IsingTarget makes the base distribution
eta read the composition the run is conditioned on — the bound per-cycle vector
during amortised training, else the target's own scalar c* — instead of the
static `base_composition`.

Properties pinned:

1. Per-row alignment: base_log_eta scores row b against row b's c (the b-major
   rule, composition.py), against a brute-force per-site product.
2. Draw/path agreement: draw and path share one binding, pinned per-row by
   log p_tilde_0 == base_log_eta on sample_base's own draws (the archived-eval
   bug was a silent 6.9-nat log w0 hole at d=64).
3. Flag off is byte-identical to today's behaviour, RNG consumption included,
   and at a bound c = 0.5 the matched draw equals the house draw bit-for-bit.
4. A bias-only Ising with bias = log(c/(1-c))/2 is the matched Bernoulli base
   (log-densities equal up to a constant), so p_tilde_t is t-invariant up to a
   constant and R = 0 satisfies Kolmogorov (Eq. 7) exactly.
5. The camort cells differ from their house parents in exactly the three
   declared levers: conditioning, spine draw, matched base.
"""

import json
import math
from dataclasses import asdict

import pytest
import torch

from discrete_flow_sampler.samplers.kolmogorov import residual_general
from discrete_flow_sampler.targets.ising import IsingTarget

SPINE = (0.25, 0.375, 0.5)


def _matched_target(D=4, **kwargs):
    defaults = dict(
        sigma=0.1,
        target_composition=0.5,
        composition_penalty_strength=50.0,
        base_matches_composition=True,
    )
    defaults.update(kwargs)
    return IsingTarget(D=D, **defaults)


def _brute_force_log_eta(x, p_per_row):
    site_p = torch.where(x > 0, p_per_row.unsqueeze(-1), 1.0 - p_per_row.unsqueeze(-1))
    return site_p.log().sum(dim=-1)


# -- 1. per-row base density ------------------------------------------------


def test_base_log_eta_per_row_matches_brute_force_mixed_c():
    target = _matched_target(D=2)
    x = torch.tensor(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    composition = torch.tensor([0.25, 0.375, 0.5, 0.25])
    with target.composition_batch(composition):
        got = target.base_log_eta(x)
    torch.testing.assert_close(got, _brute_force_log_eta(x, composition))


def test_base_log_eta_unbound_uses_target_composition():
    target = _matched_target(D=2, target_composition=0.25)
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, -1.0]])
    expected = _brute_force_log_eta(x, torch.full((2,), 0.25))
    torch.testing.assert_close(target.base_log_eta(x), expected)


def test_base_log_eta_normalises_per_row():
    # Sum over all 2^d states must be 0 (log 1) at every bound c, or the
    # RND weights carry a phantom constant per composition.
    target = _matched_target(D=2, target_composition=0.375)
    states = torch.cartesian_prod(*[torch.tensor([-1.0, 1.0])] * 4)
    for c in SPINE:
        with target.composition_batch(torch.full((states.shape[0],), c)):
            total = torch.logsumexp(target.base_log_eta(states), dim=0)
        torch.testing.assert_close(total, torch.tensor(0.0), atol=1e-5, rtol=0)


# -- 2. draw/path agreement (the de9db7c bug class) -------------------------


def test_sample_base_matched_mean_tracks_bound_composition():
    target = _matched_target(D=10, target_composition=0.5)
    n = 20_000
    for c in (0.25, 0.375):
        torch.manual_seed(0)
        with target.composition_batch(torch.full((n,), c)):
            x = target.sample_base(n, device="cpu")
        mean_composition = target.composition_fraction(x).mean().item()
        # Binomial SE at n*d = 2e6 site draws is ~3e-4; 5 SE bound.
        assert abs(mean_composition - c) < 5 * math.sqrt(
            c * (1 - c) / (n * target.d)
        ), (c, mean_composition)


def test_path_t0_equals_matched_base_on_its_own_draws():
    target = _matched_target(D=4, target_composition=0.5)
    n = 64
    composition = torch.tensor([0.25, 0.375, 0.5, 0.25]).repeat(n // 4)
    with target.composition_batch(composition):
        x = target.sample_base(n, device="cpu")
        t0 = torch.zeros(n)
        torch.testing.assert_close(target.log_p_tilde_t(x, t0), target.base_log_eta(x))


def test_path_t1_equals_penalised_target_per_row():
    target = _matched_target(D=4, target_composition=0.5)
    composition = torch.tensor([0.25, 0.5])
    x = torch.stack([torch.ones(16) * -1.0, torch.ones(16)])
    with target.composition_batch(composition):
        t1 = torch.ones(2)
        torch.testing.assert_close(target.log_p_tilde_t(x, t1), target.log_prob(x))


# -- 3. archived behaviour + centre anchor ----------------------------------


def test_flag_off_ignores_binding_and_matches_static_base():
    plain = IsingTarget(D=4, sigma=0.1, base_composition=0.5)
    bound = IsingTarget(
        D=4,
        sigma=0.1,
        base_composition=0.5,
        target_composition=0.25,
        composition_penalty_strength=50.0,
    )
    x = torch.tensor([[1.0] * 16, [-1.0] * 16])
    with bound.composition_batch(torch.tensor([0.25, 0.25])):
        got = bound.base_log_eta(x)
    torch.testing.assert_close(got, plain.base_log_eta(x))
    torch.manual_seed(7)
    expected_draw = plain.sample_base(50, device="cpu")
    torch.manual_seed(7)
    with bound.composition_batch(torch.tensor([0.25])):
        got_draw = bound.sample_base(50, device="cpu")
    torch.testing.assert_close(got_draw, expected_draw)


def test_matched_draw_at_half_is_bit_identical_to_house_base():
    plain = IsingTarget(D=4, sigma=0.1, base_composition=0.5)
    matched = _matched_target(D=4, target_composition=0.5)
    torch.manual_seed(11)
    expected = plain.sample_base(100, device="cpu")
    torch.manual_seed(11)
    with matched.composition_batch(torch.tensor([0.5])):
        got = matched.sample_base(100, device="cpu")
    torch.testing.assert_close(got, expected)


def test_flag_requires_a_composition_to_match():
    with pytest.raises(ValueError):
        IsingTarget(D=4, sigma=0.1, base_matches_composition=True)


# -- 4. analytic Kolmogorov pin ---------------------------------------------


class _ZeroRateModel:
    def __call__(self, x, t):
        return torch.zeros_like(x).float()


def test_residual_zero_when_target_is_the_matched_base():
    # bias = log(c/(1-c))/2 and sigma = 0 make log_prob(x) equal
    # log eta_c(x) up to the x-independent constant d/2*log(c(1-c)), so
    # p_tilde_t is the same distribution at every t and R = 0 satisfies
    # Kolmogorov with dt_log_Zt = that constant's negation.
    c = 0.25
    target = IsingTarget(
        D=2,
        sigma=0.0,
        bias=0.5 * math.log(c / (1 - c)),
        target_composition=c,
        base_matches_composition=True,
    )
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, -1.0]])
    t = torch.tensor([0.3, 0.8])
    with target.composition_batch(torch.full((2,), c)):
        dt_log_Zt = -torch.tensor(target.d / 2.0 * math.log(c * (1 - c)))
        got = residual_general(x, t, dt_log_Zt, _ZeroRateModel(), target)
    torch.testing.assert_close(got, torch.zeros(2), atol=1e-5, rtol=0)


# -- 5. config cells: declared levers only ----------------------------------

CAMORT_LEVER_FIELDS_TOP = {"name", "ising", "model", "composition"}


@pytest.mark.parametrize("sigma_suffix", ["", "_sc"])
def test_d8_camort_cell_is_three_declared_levers_off_house_centre(
    sigma_suffix,
):
    from experiments.constrained_soft_02.configs import CONFIGS

    parent = asdict(CONFIGS[f"S2_d8_c0500_l50_letf_ne128_house{sigma_suffix}"])
    cell = asdict(CONFIGS[f"S2_d8_camort_l50_letf_ne128_house{sigma_suffix}"])
    top = {key for key in parent if parent[key] != cell[key]}
    assert top == CAMORT_LEVER_FIELDS_TOP, top
    ising_diff = {
        key for key in parent["ising"] if parent["ising"][key] != cell["ising"][key]
    }
    assert ising_diff == {"base_matches_composition"}, ising_diff
    assert cell["ising"]["base_matches_composition"] is True
    model_diff = {
        key for key in parent["model"] if parent["model"][key] != cell["model"][key]
    }
    assert model_diff == {"condition_on_composition"}, model_diff
    # The quantised continuum: every
    # realisable composition in [0.25, 0.5] at d=64, uniform draw.
    assert cell["composition"]["values"] == tuple(sites / 64 for sites in range(16, 33))
    assert all((c * 64) == int(c * 64) for c in cell["composition"]["values"])
    assert cell["composition"]["centre"] == 0.5
    assert cell["composition"]["half_width"] == 0.0
    assert cell["composition"]["curriculum"] is None


@pytest.mark.parametrize("sigma_suffix", ["", "_sc"])
def test_d8_camort_spine3_is_one_lever_off_the_17_value_cell(sigma_suffix):
    """One lever off the 17-value cell: the draw set back to the gate's spine
    {0.25, 0.375, 0.5}, everything else byte-identical, both couplings. It
    separates "mixed draws at criticality" from "the post-gate densification
    to 17 values", confounded in the 17-value cell (healthy at sigma=0.1,
    collapsed on all four seeds at sigma_c)."""
    from experiments.constrained_soft_02.configs import CONFIGS

    parent = asdict(CONFIGS[f"S2_d8_camort_l50_letf_ne128_house{sigma_suffix}"])
    cell = asdict(CONFIGS[f"S2_d8_camort_spine3_l50_letf_ne128_house{sigma_suffix}"])
    top = {key for key in parent if parent[key] != cell[key]}
    assert top == {"name", "composition"}, top
    composition_diff = {
        key
        for key in parent["composition"]
        if parent["composition"][key] != cell["composition"][key]
    }
    assert composition_diff == {"values"}, composition_diff
    assert cell["composition"]["values"] == (0.25, 0.375, 0.5)


def test_d8_camort_spine1_is_one_lever_off_spine3_sc():
    """One lever off spine3 sc: the full amortised machinery (conditioning
    channel, matched base, per-cycle draw-and-bind) at a single value {0.5}.
    spine3 sc collapsed identically to the 17-value cell, so this splits
    machinery from mixture — dying means the machinery breaks at sigma_c with
    no mixing, matching the mb c=0.5 specialist (~0.96) means the mixture is
    the poison."""
    from experiments.constrained_soft_02.configs import CONFIGS

    parent = asdict(CONFIGS["S2_d8_camort_spine3_l50_letf_ne128_house_sc"])
    cell = asdict(CONFIGS["S2_d8_camort_spine1_l50_letf_ne128_house_sc"])
    top = {key for key in parent if parent[key] != cell[key]}
    assert top == {"name", "composition"}, top
    composition_diff = {
        key
        for key in parent["composition"]
        if parent["composition"][key] != cell["composition"][key]
    }
    assert composition_diff == {"values"}, composition_diff
    assert cell["composition"]["values"] == (0.5,)


def test_d8_camort_spine3_rb1_is_one_lever_off_spine3_sc():
    """One lever off spine3 sc: replay_buffer_cycles=1, so every inner batch is
    scored against cycle-fresh c_t and trajectories rather than its own cycle's
    frozen c_t up to 4 cycles stale — the one structural asymmetry vs the
    specialist path, which always uses the latest grid."""
    from experiments.constrained_soft_02.configs import CONFIGS

    parent = asdict(CONFIGS["S2_d8_camort_spine3_l50_letf_ne128_house_sc"])
    cell = asdict(CONFIGS["S2_d8_camort_spine3_rb1_l50_letf_ne128_house_sc"])
    top = {key for key in parent if parent[key] != cell[key]}
    assert top == {"name", "train"}, top
    train_diff = {
        key for key in parent["train"] if parent["train"][key] != cell["train"][key]
    }
    assert train_diff == {"replay_buffer_cycles"}, train_diff
    assert cell["train"]["replay_buffer_cycles"] == 1


def test_d4_camort_mb_gate_cell_spine_and_matched_base():
    from experiments.constrained_soft_02.configs import CONFIGS

    cell = CONFIGS["S2_d4_camort_mb_50k_l50_letf_house"]
    assert cell.ising.base_matches_composition is True
    assert cell.model.condition_on_composition is True
    assert cell.composition.values == SPINE
    assert cell.composition.half_width == 0.0
    assert cell.composition.curriculum is None
    # Every spine c is an integer site count at d=16 (4/6/8 sites).
    for c in cell.composition.values:
        assert (c * cell.ising.D**2) == int(c * cell.ising.D**2)


def test_d4_gate_differs_from_wave3_camort_in_declared_set_only():
    from experiments.constrained_soft_02.configs import CONFIGS

    parent = asdict(CONFIGS["S2_d4_camort_50k_l50_letf_house"])
    cell = asdict(CONFIGS["S2_d4_camort_mb_50k_l50_letf_house"])
    top = {key for key in parent if parent[key] != cell[key]}
    assert top == {"name", "ising", "composition"}, top
    ising_diff = {
        key for key in parent["ising"] if parent["ising"][key] != cell["ising"][key]
    }
    assert ising_diff == {"base_matches_composition"}, ising_diff


# -- wire: one construction helper serves train and rebuild-eval ------------


def test_construct_target_forwards_matched_base_flag():
    from experiments.dnfs_baseline_01.configs import IsingCfg
    from experiments.dnfs_baseline_01.run import _construct_target

    cfg = IsingCfg(
        D=2,
        target_composition=0.5,
        composition_penalty_strength=50.0,
        base_matches_composition=True,
    )
    target = _construct_target(cfg, device="cpu")
    assert target.base_matches_composition is True
    assert (
        _construct_target(IsingCfg(D=2), device="cpu").base_matches_composition is False
    )


def test_rebuild_from_run_dir_forwards_matched_base_flag(tmp_path):
    # A rebuild that silently drops the flag would eval a matched-base run
    # against a uniform base (the pre-de9db7c failure), so go via config.json.
    from dataclasses import asdict as cfg_asdict

    from experiments.constrained_soft_02.configs import CONFIGS
    from experiments.dnfs_baseline_01.run import _rebuild_from_run_dir

    cfg = CONFIGS["S2_d4_camort_mb_50k_l50_letf_house"]
    (tmp_path / "config.json").write_text(json.dumps(cfg_asdict(cfg)))
    _cfg, target, _device = _rebuild_from_run_dir(tmp_path)
    assert target.base_matches_composition is True
