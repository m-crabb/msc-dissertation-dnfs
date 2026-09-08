"""Per-slice c_t on composition mixtures (hard amortisation).

On a slice mixture the Kolmogorov residual (Eq. 10) for a row on slice C
needs ∂_t log Z_t^{(C)}, that slice's own normaliser derivative: swap
dynamics hold every slice's mass fixed, so only the conditional on each
slice evolves and there is no single mixture-level ∂_t log Z_t the residual
could use. The control-variate estimate (Eq. 8) c_t = mean_m ξ_t(x_m) is
unbiased within a slice for any rates (Stein), so
the correction is a within-slice mean plus a (time, slice) lookup. An
earlier version of the hard trainer pooled the mean over every rollout row and
handed every replay row that one scalar, leaving each row an offset
∂_t log Z_t^{(C)} − mean_C ∂_t log Z_t^{(C)} that is ~2 nats at d16 and
~18 nats at d256 from the binomial base constant alone.

Single-slice targets must stay byte-identical: the archived specialist
numbers are pinned at the batch-blocking 1e-5 class and their EMA resume
state is a (T,) grid.
"""

from pathlib import Path

import pytest
import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers import swap_training
from discrete_flow_sampler.samplers.swap_ctmc import (
    compute_c_t_grid_swap,
    mean_per_slice,
    n_slices,
    slice_index_of,
)
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    MixtureCompositionIsingTarget,
)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _tiny_head():
    return DoublyHollowSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _mixture_target():
    # d=16: slices n_plus=8 (12,870 states) and n_plus=4 (1,820 states).
    return MixtureCompositionIsingTarget(D=4, sigma=0.3, compositions=(0.5, 0.25))


# --------------------------------------------------------------------------
# Slice bookkeeping
# --------------------------------------------------------------------------


def test_slice_index_reads_the_registered_count_off_each_row():
    target = _mixture_target()
    x = torch.full((3, 16), -1.0)
    x[0, :8] = 1.0  # slice 0 (composition 0.5)
    x[1, :4] = 1.0  # slice 1 (composition 0.25)
    x[2, 3:11] = 1.0  # slice 0 again, different sites
    assert slice_index_of(target, x).tolist() == [0, 1, 0]
    assert n_slices(target) == 2


def test_slice_index_is_zero_and_single_slice_without_a_grid():
    target = FixedCompositionIsingTarget(D=4, sigma=0.3, target_composition=0.5)
    x = target.sample_base(5, device="cpu")
    assert slice_index_of(target, x).tolist() == [0] * 5
    assert n_slices(target) == 1


def test_mean_per_slice_matches_an_explicit_group_by():
    torch.manual_seed(0)
    values = torch.randn(6, 40)
    slice_idx = torch.randint(0, 3, (40,))
    got = mean_per_slice(values, slice_idx, 3)
    assert got.shape == (6, 3)
    for k in range(3):
        want = values[:, slice_idx == k].mean(dim=-1)
        assert torch.allclose(got[:, k], want, atol=1e-6)


def test_mean_per_slice_falls_back_to_the_pooled_mean_on_an_empty_slice():
    """A slice no rollout row landed on (probability ~K·(1−1/K)^M, i.e.
    ~1e-12 at M=128) must not poison the grid with NaN; the pooled slot
    mean is the least-wrong finite value and the next cycle redraws."""
    values = torch.randn(4, 10)
    slice_idx = torch.zeros(10, dtype=torch.long)  # slice 1 never drawn
    got = mean_per_slice(values, slice_idx, 2)
    assert torch.allclose(got[:, 0], values.mean(dim=-1))
    assert torch.allclose(got[:, 1], values.mean(dim=-1))


def test_single_slice_reduction_is_bitwise_the_plain_mean():
    values = torch.randn(7, 33)
    got = mean_per_slice(values, torch.zeros(33, dtype=torch.long), 1)
    assert got.shape == (7, 1)
    assert torch.equal(got[:, 0], values.mean(dim=-1))


# --------------------------------------------------------------------------
# The estimator on the mixture: exact per-slice normaliser derivatives
# --------------------------------------------------------------------------


def _exact_slice_conditional(target, states, t):
    """p_t^{(C)} over the enumerated states of one slice at time t."""
    t_col = torch.full((states.shape[0],), t)
    log_p_tilde = target.log_p_tilde_t(states, t_col)
    return torch.softmax(log_p_tilde, dim=0)


@torch.no_grad()
def test_c_t_grid_on_a_mixture_recovers_each_slice_normaliser_derivative():
    """Per-slice c_t must match the enumerated ∂_t log Z_t^{(C)} on both
    slices, while the pooled mean the old code returned matches neither
    by more than the Monte-Carlo tolerance."""
    torch.manual_seed(0)
    target = _mixture_target()
    all_states = enumerate_states(16).float()
    n_plus = ((all_states + 1) * 0.5).sum(dim=-1).long()
    slices = [all_states[n_plus == count] for count in target.n_plus_values]
    t_grid = torch.tensor([0.25, 0.6, 0.9])
    rows_per_slice = 4000

    x_traj, exact = [], []
    for t in t_grid.tolist():
        rows, exact_at_t = [], []
        for states in slices:
            p_t = _exact_slice_conditional(target, states, t)
            dt_log = target.dt_log_p_tilde_t(states, torch.full((states.shape[0],), t))
            exact_at_t.append((p_t * dt_log).sum())
            draw = torch.multinomial(p_t, rows_per_slice, replacement=True)
            rows.append(states[draw])
        x_traj.append(torch.cat(rows))
        exact.append(torch.stack(exact_at_t))
    x_traj = torch.stack(x_traj)  # (T, 2M, 16)
    exact = torch.stack(exact)  # (T, 2)

    c_t_grid, integrand = compute_c_t_grid_swap(
        t_grid, x_traj, target, head=None, mode="naive_mc"
    )
    assert c_t_grid.shape == (3, 2) and integrand.shape == (3, 2 * rows_per_slice)

    monte_carlo_tol = 0.08
    assert torch.allclose(c_t_grid, exact, atol=monte_carlo_tol), (c_t_grid, exact)
    pooled = integrand.mean(dim=-1)  # the pre-fix grid
    slice_gap = (exact[:, 0] - exact[:, 1]).abs()
    assert (slice_gap > 4 * monte_carlo_tol).all(), slice_gap
    assert ((pooled[:, None] - exact).abs() > 2 * monte_carlo_tol).all()


def test_c_t_grid_shape_is_unchanged_for_a_single_slice_target():
    """The archived (T,) contract for specialists, both estimator modes."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.3, target_composition=0.5)
    head = _tiny_head()
    t_grid = torch.linspace(0.0, 1.0, 5)
    x_traj = torch.stack([target.sample_base(6, device="cpu") for _ in range(5)])
    for mode in ("naive_mc", "control_variate"):
        c_t, integrand = compute_c_t_grid_swap(t_grid, x_traj, target, head, mode=mode)
        assert c_t.shape == (5,)
        assert torch.equal(c_t, integrand.mean(dim=-1))


# --------------------------------------------------------------------------
# The trainer: every replay row gets its own slice's baseline
# --------------------------------------------------------------------------


def _tiny_cfgs(**train_extra):
    train_cfg = _Cfg(
        n_steps=4,
        batch_size=48,
        outer_batch_size=48,
        inner_steps_per_outer=2,
        lr=1e-3,
        seed=0,
        replay_buffer_cycles=1,
        grad_clip_max_norm=500.0,
        warmup_steps=0,
        **train_extra,
    )
    ctmc_cfg = _Cfg(n_euler_steps=4)
    eval_cfg = _Cfg(eval_every=100, n_eval_samples=8)
    return train_cfg, ctmc_cfg, eval_cfg


def _capture_loss_calls(monkeypatch):
    """Wrap the trainer's loss entry point to record (x, t, c_t) per call."""
    calls = []
    original = swap_training.loss_swap_backward_microbatched

    def recording(x, t, dt_log_Zt, head, target, **kw):
        calls.append(
            (x.detach().clone(), t.detach().clone(), dt_log_Zt.detach().clone())
        )
        return original(x, t, dt_log_Zt, head, target, **kw)

    monkeypatch.setattr(swap_training, "loss_swap_backward_microbatched", recording)
    return calls


def _same_time_pairs(x, t, c_t):
    """Pairs of rows sharing a time slot, split by whether they share a slice."""
    n_plus = ((x + 1) * 0.5).sum(dim=-1).long()
    same_slice, cross_slice = [], []
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            if t[i] != t[j]:
                continue
            gap = (c_t[i] - c_t[j]).abs().item()
            (same_slice if n_plus[i] == n_plus[j] else cross_slice).append(gap)
    return same_slice, cross_slice


@pytest.mark.parametrize("c_t_from_rollout", [False, True])
def test_train_swap_hands_each_row_its_own_slice_baseline(
    tmp_path, monkeypatch, c_t_from_rollout
):
    """Both grid paths the camort cells use: the standalone estimator and
    the rollout-integrand reuse (c_t_from_rollout). Rows at one time slot on one slice
    share a baseline; rows at one time slot on different slices do not."""
    torch.manual_seed(0)
    calls = _capture_loss_calls(monkeypatch)
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs(c_t_from_rollout=c_t_from_rollout)
    train_swap(
        _tiny_head(),
        _mixture_target(),
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        Path(tmp_path),
        use_wandb=False,
        estimator_mode="control_variate",
    )
    assert calls
    same_slice, cross_slice = [], []
    for x, t, c_t in calls:
        assert torch.isfinite(c_t).all()
        same, cross = _same_time_pairs(x, t, c_t)
        same_slice += same
        cross_slice += cross
    assert same_slice and cross_slice, "batch never collided in a slot"
    assert max(same_slice) == 0.0
    assert min(cross_slice) > 0.1  # ~2 nats from the base constant alone


def test_train_swap_specialist_rows_at_one_slot_share_one_baseline(
    tmp_path, monkeypatch
):
    torch.manual_seed(0)
    calls = _capture_loss_calls(monkeypatch)
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
    target = FixedCompositionIsingTarget(D=4, sigma=0.3, target_composition=0.5)
    train_swap(
        _tiny_head(),
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        Path(tmp_path),
        use_wandb=False,
        estimator_mode="control_variate",
    )
    same_slice, cross_slice = [], []
    for x, t, c_t in calls:
        same, cross = _same_time_pairs(x, t, c_t)
        same_slice += same
        cross_slice += cross
    assert same_slice and not cross_slice
    assert max(same_slice) == 0.0
