"""Zero-shot transfer probe: what a trained swap sampler delivers off-target.

Two axes, both evaluated on an UNCHANGED trained head:

  * coupling.     On the fixed-composition slice base_log_eta is a constant
                  (-log C(d, N_A)), so the geometric path
                  log p~_t = (1-t)*base_log_eta + t*log_prob has its (1-t)
                  term cancel out of every ratio, leaving log p~_t ∝ t*sigma*xAx.
                  The path IS a coupling anneal: stopping at t* lands exactly on
                  the target at coupling t*·sigma. No retraining, no new base.
  * composition.  Swapping the target's slice changes sample_base and the
                  slice constant. The head never sees c except through x, so it
                  transfers unchanged; the IS weights re-target by construction.

Both are EXACT whatever the model does — the learned rates are only a proposal,
and xi_t is evaluated against whichever target is handed in. Only the efficiency
(ESS) varies. That is what makes this probe safe to run before any new training.

What these tests pin, in order of how much the probe rests on them:

1. `running_log_weights` reconstructs, from ONE sampler pass, the weights a
   truncated run would have produced at every grid time. This is the whole
   reason the probe is cheap (one pass per composition, not one per stopping
   time), and it is a mechanical claim, not a statistical one.
2. The early-stopped weighted ensemble targets p~_{t*}, checked at 4x4 against
   exact enumeration of the slice. This is the claim the physics rests on.
3. The same holds on a slice the model was never trained on.
4. Draws never leave the requested manifold — a transferred composition must be
   delivered bit-exactly, or the "constraint is free" story is false.

4x4 is used throughout because the fixed-c slice is enumerable there
(C(16,8) = 12,870 at half filling), so every claim is checked against exact
ground truth rather than against another sampler.
"""
import math

import pytest
import torch

from discrete_flow_sampler.constraints.two_hole_patch_swap_head import (
    TwoHolePatchSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

from experiments.constrained_hard_03.probe_zero_shot_transfer import (
    check_sampling_provenance,
    exact_slice_statistics,
    running_log_weights,
    slice_states,
    transfer_grid,
    weighted_diagnostics,
)

D = 4
SIGMA = 0.220343
N_EULER = 24


def _head(seed: int = 0, device: str = "cpu"):
    """A small thp head at 4x4. Untrained: the probe's correctness must not
    depend on the model being good, only on the weights re-targeting."""
    torch.manual_seed(seed)
    backbone = LeTFRateMatrix(
        d=D * D, vocab_size=2, hidden_dim=16, n_layers=1, n_heads=2
    )
    return TwoHolePatchSwapHead(
        backbone, lattice_side=D, patch_radius=1, feature_dim=8
    ).to(device)


def _target(composition: float = 0.5, sigma: float = SIGMA):
    return FixedCompositionIsingTarget(
        D=D, sigma=sigma, target_composition=composition
    )


# --- 1. the one-pass reconstruction --------------------------------


@pytest.mark.parametrize("stop_index", [1, 7, N_EULER - 1])
def test_running_weights_match_a_truncated_run(stop_index):
    """log w at grid time k from one full pass == the weight a run stopped at k
    reports. Same seed, so the first k steps consume identical randomness and
    the two integrate the same increments.

    This is the claim that makes the probe one pass instead of len(stop_times)
    passes. If it fails, every stopping time needs its own run.

    NOT bit-exact, and the tolerance is the point rather than a concession:
    `cumsum` and the sampler's sequential `log_weights += xi*dt` sum the same
    float32 increments in different orders, which lands ~5e-7 apart. That is
    associativity, not a different quantity — a genuine off-by-one or a wrong
    integrand would move the weights by O(nats), several million times this
    band, so a loose absolute tolerance still discriminates completely.
    """
    head, target = _head(), _target()
    ts = torch.linspace(0.0, 1.0, N_EULER)

    torch.manual_seed(1234)
    x0 = target.sample_base(64, device="cpu")
    torch.manual_seed(99)
    _, running = running_log_weights(head, target, x0, ts)

    torch.manual_seed(99)
    _, truncated = sample_swap_ctmc(
        head, x0, ts[: stop_index + 1], target=target, return_log_weights=True
    )

    assert running.shape == (len(ts), 64)
    torch.testing.assert_close(running[stop_index], truncated, rtol=1e-5, atol=1e-5)


def test_running_weights_start_at_zero():
    """At t=0 the ensemble IS the base, so the weight integral is empty. A
    non-zero weight here would mean the cumsum is off by one step and every
    reported stopping time is shifted."""
    head, target = _head(), _target()
    ts = torch.linspace(0.0, 1.0, N_EULER)
    x0 = target.sample_base(32, device="cpu")
    _, running = running_log_weights(head, target, x0, ts)
    assert torch.all(running[0] == 0.0)


# --- 2. the early-stopped ensemble targets the intermediate coupling ----------


@pytest.mark.parametrize("stop_time", [0.4538, 1.0])
def test_early_stopped_weights_target_the_intermediate_coupling(stop_time):
    """Self-normalised IS at t* must reproduce the EXACT slice expectation at
    coupling t*·sigma, not at sigma.

    t* = 0.4538 is the stopping time that lands on sigma = 0.1 for a sigma_c
    path — the one intermediate coupling with a certified d256 reference, so
    this is the point the production probe will be judged at.

    Tolerance is 4 standard errors of the self-normalised estimator, computed
    from the run's own effective sample size rather than assumed: an untrained
    head gives dispersed weights and a fixed absolute tolerance would either
    be vacuous or flaky.
    """
    head, target = _head(), _target()
    ts = torch.linspace(0.0, 1.0, N_EULER)
    torch.manual_seed(7)
    x0 = target.sample_base(4096, device="cpu")
    trajectory, running = running_log_weights(head, target, x0, ts)

    index = int(torch.argmin((ts - stop_time).abs()))
    log_w, states = running[index], trajectory[index]
    weights = torch.softmax(log_w, dim=0)

    energy = (states @ target.A * states).sum(-1)
    estimate = (weights * energy).sum()

    exact = exact_slice_statistics(
        D=D, sigma=SIGMA, composition=0.5, stop_time=float(ts[index])
    )
    ess = float(weighted_diagnostics(log_w, states, target)["ess"])
    spread = float(((weights * (energy - estimate) ** 2).sum()).sqrt())
    tolerance = 4.0 * spread / math.sqrt(ess)

    assert abs(float(estimate) - exact["mean_quadratic"]) < tolerance

    # And the intermediate target must be genuinely DIFFERENT from the endpoint,
    # or the test above passes for the wrong reason.
    if stop_time < 1.0:
        endpoint = exact_slice_statistics(
            D=D, sigma=SIGMA, composition=0.5, stop_time=1.0
        )
        assert abs(exact["mean_quadratic"] - endpoint["mean_quadratic"]) > tolerance


# --- 3. transfer to a slice the model never saw -------------------------------


def test_transfer_to_an_unseen_composition_targets_that_slice():
    """The head is untouched; only the target's slice moves. The weighted
    ensemble must land on the exact conditional of the NEW slice."""
    head = _head()
    target = _target(composition=0.375)  # 6 of 16 up — never the training slice
    ts = torch.linspace(0.0, 1.0, N_EULER)
    torch.manual_seed(11)
    x0 = target.sample_base(4096, device="cpu")
    trajectory, running = running_log_weights(head, target, x0, ts)

    log_w, states = running[-1], trajectory[-1]
    weights = torch.softmax(log_w, dim=0)
    energy = (states @ target.A * states).sum(-1)
    estimate = (weights * energy).sum()

    exact = exact_slice_statistics(
        D=D, sigma=SIGMA, composition=0.375, stop_time=1.0
    )
    ess = float(weighted_diagnostics(log_w, states, target)["ess"])
    spread = float(((weights * (energy - estimate) ** 2).sum()).sqrt())
    assert abs(float(estimate) - exact["mean_quadratic"]) < 4.0 * spread / math.sqrt(ess)


def test_transferred_draws_never_leave_the_requested_manifold():
    """Composition is enforced by the move set, so a transferred slice must be
    delivered bit-exactly on every draw at every grid time — not on average.
    If this fails, the hard leg's central claim does not survive transfer."""
    head = _head()
    for composition in (0.25, 0.375, 0.625):
        target = _target(composition=composition)
        ts = torch.linspace(0.0, 1.0, N_EULER)
        torch.manual_seed(3)
        x0 = target.sample_base(256, device="cpu")
        trajectory, _ = running_log_weights(head, target, x0, ts)
        for step in range(len(ts)):
            target.assert_on_manifold(trajectory[step])


# --- 4. the exact-enumeration reference itself --------------------------------


def test_slice_enumeration_has_the_binomial_size():
    for k in (4, 6, 8):
        states = slice_states(D=D, composition=k / (D * D))
        assert states.shape[0] == math.comb(D * D, k)
        assert torch.all(((states + 1) / 2).sum(-1) == k)


def test_exact_statistics_obey_the_z2_mirror():
    """The slice at c and at 1-c are exact global-spin-flip images, and every
    observable here is even under that flip, so the enumerated values must
    match. Zero-cost control on the reference the probe is judged against."""
    for composition in (0.25, 0.375):
        low = exact_slice_statistics(D, SIGMA, composition, stop_time=1.0)
        high = exact_slice_statistics(D, SIGMA, 1.0 - composition, stop_time=1.0)
        assert low["mean_quadratic"] == pytest.approx(high["mean_quadratic"], rel=1e-10)
        assert low["nn_correlation"] == pytest.approx(
            high["nn_correlation"], rel=1e-10
        )


def test_stop_time_zero_is_the_uniform_slice():
    """At t=0 the tilt vanishes and the conditional is uniform on the slice, so
    the mean energy is the plain slice average. Anchors the coupling axis at
    the end where the answer is known without any Ising physics."""
    exact = exact_slice_statistics(D, SIGMA, composition=0.5, stop_time=0.0)
    # float64 on both sides: the reference is an enumeration, and at 12,870
    # states a float32 sum loses ~1e-7 relative — larger than the identity being
    # certified. The exact value here is rational (-64/15).
    states = slice_states(D, composition=0.5).double()
    target = _target()
    uniform_mean = float((states @ target.A.double() * states).sum(-1).mean())
    assert exact["mean_quadratic"] == pytest.approx(uniform_mean, rel=1e-10)


# --- 5. the grid wrapper ------------------------------------------------------


def test_transfer_grid_covers_both_axes_and_reports_the_coupling():
    """The wrapper's contract: one row per (composition, stop_time), each
    carrying the PHYSICAL coupling t*·sigma rather than the bare stop time,
    because that is the number a reference chain is generated at."""
    head = _head()
    rows = transfer_grid(
        head,
        D=D,
        sigma=SIGMA,
        compositions=(0.5, 0.375),
        stop_times=(0.4538, 1.0),
        n_samples=256,
        n_euler_steps=N_EULER,
        seed=5,
    )
    assert len(rows) == 4
    for row in rows:
        assert row["coupling"] == pytest.approx(row["stop_time"] * SIGMA)
        assert 0.0 < row["ess_fraction"] <= 1.0
        assert row["n_plus"] == round(row["composition"] * D * D)
        assert math.isfinite(row["var_log_w_per_site"])


# --- 6. provenance -----------------------------------------------------------


def test_training_side_drift_is_reported_not_fatal():
    """A finished cell whose TRAINING config has since moved must still be
    probeable: no optimiser runs here, so the training subtree cannot reach the
    sampled measure. The drift is returned so the caller logs it rather than
    swallowing it.

    Live case this encodes: `halt_on_cv_inversion_after` was cleared to None
    after the thp2 sigma_c cells launched. It is a cold-CV screening halt that
    never fired on them, and run.py's whole-config launch guard would refuse
    every one of those checkpoints over it.
    """
    saved = {"name": "cell", "ising": {"sigma": 0.22}, "ctmc": {"n_euler_steps": 128},
             "train": {"seed": 42, "halt_on_cv_inversion_after": 5000}}
    current = {"name": "cell", "ising": {"sigma": 0.22}, "ctmc": {"n_euler_steps": 128},
               "train": {"seed": 42, "halt_on_cv_inversion_after": None}}
    assert check_sampling_provenance(saved, current) == {
        "halt_on_cv_inversion_after": (5000, None)
    }


@pytest.mark.parametrize(
    "field, value",
    [
        ("ising", {"sigma": 0.1}),
        ("ctmc", {"n_euler_steps": 64}),
        ("head_kind", "masked_attention"),
    ],
)
def test_drift_in_what_is_sampled_is_fatal(field, value):
    """Sigma, the Euler grid and the head all change what is being sampled. A
    silent mismatch would be read as failed transfer rather than as the wrong
    model, which is the one way this probe can produce a confidently wrong
    conclusion — so it must refuse rather than warn."""
    saved = {"name": "cell", "ising": {"sigma": 0.22},
             "ctmc": {"n_euler_steps": 128}, "head_kind": "two_hole_patch",
             "train": {"seed": 42}}
    current = dict(saved, **{field: value})
    with pytest.raises(ValueError, match="change what is sampled"):
        check_sampling_provenance(saved, current)
