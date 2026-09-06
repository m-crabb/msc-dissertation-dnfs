"""Tests for the CTMC Euler-step trajectory sampler.

These pin behavioural invariants, not numerical correctness of the IS
weight formula -- the latter requires choosing a specific xi_t form
(see paper Eq. 8 / 13) and is exercised by the higher-value Kolmogorov
test (test_kolmogorov.py).

Invariants pinned here:
    1) zero rate -> trajectory is identity (state unchanged).
    2) output preserves shape and the {-1, +1} value set.
    3) `return_log_weights=True` returns a (state, log_w) tuple of the
       expected shapes.
    4) zero-rate trajectories still produce finite log-weights when a
       target is supplied (xi_t reduces to dt_log_p_tilde, which is finite).
"""
import pytest
import torch

from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.targets.ising import IsingTarget


class ZeroRateModel:
    """R_t == 0 everywhere. Trajectory should be the identity."""

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x).float()


class ConstantRateModel:
    """Constant flip rate per site, independent of (x, t).

    Useful for shape tests; doesn't pin any specific dynamic behaviour.
    """

    def __init__(self, flip_rate: float):
        self.flip_rate = flip_rate

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.flip_rate, dtype=torch.float)


def test_zero_rate_preserves_state():
    x0 = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    time_grid = torch.linspace(0, 1, 10)
    x_final = sample_ctmc(ZeroRateModel(), x0, time_grid)
    assert torch.equal(x_final, x0)


def test_output_shape_and_value_set():
    """Output keeps (B, d) shape and stays in the binary {-1, +1} state set."""
    x0 = torch.randint(0, 2, (8, 16)).float() * 2 - 1
    time_grid = torch.linspace(0, 1, 50)
    x_final = sample_ctmc(ConstantRateModel(flip_rate=0.5), x0, time_grid)
    assert x_final.shape == x0.shape
    assert torch.all((x_final == 1) | (x_final == -1))


def test_log_weights_returned_when_requested():
    """With return_log_weights=True, output is (x_final, log_w) and shapes
    are (B, d) and (B,) respectively."""
    target = IsingTarget(D=4, sigma=0.1)
    x0 = torch.randint(0, 2, (4, 16)).float() * 2 - 1
    time_grid = torch.linspace(0, 1, 20)
    out = sample_ctmc(
        ConstantRateModel(flip_rate=0.5),
        x0,
        time_grid,
        return_log_weights=True,
        target=target,
    )
    assert isinstance(out, tuple) and len(out) == 2
    x_final, log_w = out
    assert x_final.shape == (4, 16)
    assert log_w.shape == (4,)


def test_zero_rate_finite_weights():
    """R=0 freezes the state, but the log-weight still accumulates the
    target's dt_log_p_tilde contribution. Only finiteness is asserted here --
    correctness of the xi_t formula is checked by the Kolmogorov test
    (test_kolmogorov.py), not here."""
    target = IsingTarget(D=2, sigma=0.1)
    x0 = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    time_grid = torch.linspace(0, 1, 10)
    x_final, log_w = sample_ctmc(
        ZeroRateModel(),
        x0,
        time_grid,
        return_log_weights=True,
        target=target,
    )
    assert torch.equal(x_final, x0)
    assert torch.isfinite(log_w).all()


def test_return_all_states_shape_and_endpoints():
    """`return_all_states=True` returns the (T, B, D) trajectory.

    Pinned invariants:
      - shape (T, B, D), where T = len(ts).
      - traj[0] == x0 (the supplied initial state at ts[0]).
      - traj[-1] equals what the default scalar-return path produces under
        the same RNG seed -- the trajectory must agree with the existing
        single-endpoint path at its final step (otherwise the buffer's
        last time slot would silently drift from `sample_ctmc`'s endpoint).
    """
    torch.manual_seed(0)
    n_dims = 4
    batch_size = 8
    n_grid = 10
    flip_rate = 0.3
    model = ConstantRateModel(flip_rate=flip_rate)
    x0 = torch.randint(0, 2, (batch_size, n_dims)).float() * 2 - 1
    ts = torch.linspace(0.0, 1.0, n_grid)

    torch.manual_seed(123)
    traj = sample_ctmc(model, x0, ts, return_all_states=True)

    assert traj.shape == (n_grid, batch_size, n_dims)
    assert torch.equal(traj[0], x0), "first slot must be the input x0"

    torch.manual_seed(123)
    x_final_default = sample_ctmc(model, x0, ts)
    assert torch.equal(traj[-1], x_final_default), (
        "final slot must equal the default sample_ctmc endpoint under same seed"
    )


def test_return_all_states_value_set_and_states_ordered_by_grid():
    """All trajectory states stay in {-1, +1} and the grid axis is preserved
    in order (no transposition of T-vs-B axes)."""
    torch.manual_seed(1)
    n_dims = 6
    batch_size = 4
    n_grid = 12
    model = ConstantRateModel(flip_rate=0.5)
    x0 = torch.randint(0, 2, (batch_size, n_dims)).float() * 2 - 1
    ts = torch.linspace(0.0, 1.0, n_grid)

    traj = sample_ctmc(model, x0, ts, return_all_states=True)
    assert torch.all((traj == 1) | (traj == -1))
    # Axis-order sanity: traj[k] has shape (B, D), not (D, B).
    assert traj[0].shape == (batch_size, n_dims)


def test_return_all_states_is_incompatible_with_log_weights():
    """Log-weights are deliberately not cross-implemented with all-states
    return -- the buffer-construction path doesn't need IS weights, and
    silently returning a 3-tuple would be a footgun."""
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    x0 = torch.randint(0, 2, (4, 4)).float() * 2 - 1
    ts = torch.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="return_all_states"):
        sample_ctmc(
            ConstantRateModel(flip_rate=0.2),
            x0, ts,
            return_all_states=True,
            return_log_weights=True,
            target=target,
        )


def test_sample_ctmc_lenet_path_runs_and_preserves_state_set():
    """sample_ctmc with a leMLP must run end-to-end and produce states
    in the expected support {-1, +1}^D. Doesn't pin distributional
    accuracy (training-loop integration tests handle that); just shape
    + support."""
    from discrete_flow_sampler.models.lemlp import LeMLPRateMatrix
    torch.manual_seed(0)
    D = 4
    model = LeMLPRateMatrix(d=D, vocab_size=2, hidden_dim=16, n_summands=2)
    x0 = torch.randint(0, 2, (8, D)).float() * 2 - 1
    ts = torch.linspace(0.0, 1.0, 20)
    x_final = sample_ctmc(model, x0, ts)
    assert x_final.shape == (8, D)
    unique = torch.unique(x_final)
    assert set(unique.tolist()).issubset({-1.0, 1.0}), (
        f"sample_ctmc produced out-of-support values: {unique}"
    )
