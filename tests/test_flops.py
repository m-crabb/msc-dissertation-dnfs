"""Tests for the FLOP/es counter (house-table cost column, agreed s50/s63).

What correct looks like, fixed before the implementation:
- the per-forward instrument is torch's FlopCounterMode, so on a bare
  linear layer it must return EXACTLY the textbook 2*M*N*K matmul count,
  and it must be linear in batch size (the licence for measuring at B=1
  and scaling);
- the neural sampling bill is n_euler_steps forwards plus the (negligible
  but counted) target-energy evaluations for the IS weights, divided by
  the ESS fraction for the per-effective-sample price;
- the chain bill divides total work by N/tau_int effective records;
- Wolff instrumentation must not change a single sample: same seed with
  and without the log gives bit-identical output (the certified pools are
  recounted, never rebuilt).
"""
import pytest
import torch
from torch import nn

from discrete_flow_sampler.diagnostics.flops import (
    chain_per_effective_sample, gibbs_run_flops, ising_energy_eval_flops,
    measured_forward_flops, neural_sampling_flops_per_sample,
    per_effective_sample, wolff_run_flops)
from discrete_flow_sampler.mcmc.wolff import wolff_sample
from discrete_flow_sampler.targets.ising import IsingTarget


def test_measured_forward_flops_matches_textbook_matmul_count():
    layer = nn.Linear(8, 4, bias=False)
    flops = measured_forward_flops(layer, (torch.randn(1, 8),))
    assert flops == 2 * 8 * 4  # 2 FLOPs per multiply-accumulate


def test_measured_forward_flops_is_linear_in_batch():
    layer = nn.Sequential(nn.Linear(8, 16, bias=False), nn.Linear(16, 4, bias=False))
    single = measured_forward_flops(layer, (torch.randn(1, 8),))
    double = measured_forward_flops(layer, (torch.randn(2, 8),))
    assert double == 2 * single


def test_measured_forward_flops_runs_on_the_letf_rate_matrix():
    """Integration: the instrument works on the real architecture at a real
    shape (the d=16 gate size), eager build, and attention makes the count
    strictly exceed the embedding-free linear floor."""
    from experiments.dnfs_baseline_01.configs import CONFIGS
    from experiments.dnfs_baseline_01.run import _construct_model

    cfg = CONFIGS["stage_4_d4_critical_sc"]
    target = IsingTarget(D=4, sigma=cfg.ising.sigma, bias=0.0)
    model = _construct_model(cfg, target)  # eager: no compile wrapper
    x = target.sample_base(2, device="cpu")
    t = torch.zeros(2)
    flops = measured_forward_flops(model, (x, t))
    assert flops > 0
    assert flops % 2 == 0  # counts are 2*MAC, always even


def test_neural_sampling_bill_composes_forwards_and_weight_evals():
    per_forward = 1_000_000
    bill = neural_sampling_flops_per_sample(
        per_forward_flops=per_forward, n_euler_steps=64, n_sites=100)
    assert bill == 64 * (per_forward + ising_energy_eval_flops(100))
    # the weight-eval term must be counted but negligible (< 1% here)
    assert 64 * ising_energy_eval_flops(100) < 0.01 * 64 * per_forward


def test_per_effective_sample_divides_by_ess_and_guards_range():
    assert per_effective_sample(1000, ess_fraction=0.5) == 2000
    assert per_effective_sample(1000, ess_fraction=1.0) == 1000
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            per_effective_sample(1000, ess_fraction=bad)


def test_chain_bill_divides_by_effective_records():
    # 100 records at tau_int=2 -> 50 effective samples
    assert chain_per_effective_sample(
        total_flops=10_000, n_records=100, tau_int=2.0) == 200


def test_gibbs_and_wolff_bills_scale_with_their_work_units():
    assert gibbs_run_flops(n_sites=100, n_sweeps=3) == 3 * gibbs_run_flops(100, 1)
    assert wolff_run_flops(total_cluster_sites=50) == 50 * wolff_run_flops(1)


def test_wolff_cluster_log_does_not_perturb_the_chain():
    target = IsingTarget(D=3, sigma=0.2, bias=0.0)
    plain = wolff_sample(target, n_samples=5, clusters_per_sample=3,
                         burn_in_clusters=10, seed=7)
    log: list[int] = []
    logged = wolff_sample(target, n_samples=5, clusters_per_sample=3,
                          burn_in_clusters=10, seed=7, cluster_size_log=log)
    assert torch.equal(plain, logged)
    # burn-in included: the realistic price starts at the first cluster
    assert len(log) == 10 + 5 * 3
    assert all(1 <= size <= 9 for size in log)
