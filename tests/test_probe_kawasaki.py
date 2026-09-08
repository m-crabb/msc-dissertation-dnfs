"""Tests for the 8x8 Kawasaki mixing-probe machinery.

1. The local nearest-neighbour-swap snapshot runner is composition-preserving
   by construction: every recorded snapshot sits exactly on the c = 0.5 slice.
   An off-slice snapshot means the move set is not a swap, and the hard
   constraint has silently become soft.
2. At sigma = 0 every unlike-pair proposal is accepted (delta log p = 0), so
   the acceptance fraction equals the probability that a uniformly drawn
   directed NN bond is unlike-spin. Under the uniform slice distribution that
   is d / (2*(d-1)) for any fixed bond (hypergeometric: pick the partner spin
   from the remaining d-1 sites, of which d/2 are opposite). The chain kernel
   preserves uniformity at sigma = 0, so the long-run acceptance pins both the
   proposal distribution and the accept rule.
3. Split-half R-hat halves each chain before the multi-chain R-hat, so a
   drift shared by all chains (none stationary, all agreeing) is caught as a
   first-half/second-half discrepancy that plain R-hat is blind to.
4. The mchammer canonical probe runner respects the explicit initial state
   (snapshot 0 is the seeded configuration), keeps composition constant,
   records the exact snapshot count, and its data-container `potential`
   equals -sigma * x^T A x recomputed from our snapshots -- the cross-engine
   check that fails if the supercell atom order stops being readable as a
   row-major torus flattening.
"""

import numpy as np
import pytest

from discrete_flow_sampler.diagnostics.metrics import (
    gelman_rubin,
    split_half_gelman_rubin,
)
from discrete_flow_sampler.mcmc.kawasaki import (
    init_phase_separated,
    init_random_at_composition,
    initial_log_prob_ising,
    left_minus_right,
    run_local_swap_chain_snapshots,
)
from discrete_flow_sampler.mcmc.mchammer_ising import run_canonical_probe

D_SMALL = 4
d_SMALL = D_SMALL * D_SMALL


# ---------------------------------------------------------------------------
# (1) numba local snapshot runner: composition exact on every snapshot
# ---------------------------------------------------------------------------


def test_local_snapshot_runner_preserves_composition_on_every_snapshot():
    rng = np.random.default_rng(0)
    x = init_random_at_composition(d_SMALL, 0.5, rng)
    n_steps = 500 * d_SMALL  # 500 sweeps
    thin = d_SMALL  # snapshot every sweep
    snapshots, x_final, n_accept = run_local_swap_chain_snapshots(
        x, D_SMALL, 0.223, n_steps, 7, thin
    )
    assert snapshots.dtype == np.int8
    assert snapshots.shape == (n_steps // thin, d_SMALL)
    assert np.all(np.abs(snapshots) == 1)
    assert np.all((snapshots == 1).sum(axis=1) == d_SMALL // 2)
    assert int((x_final == 1).sum()) == d_SMALL // 2
    assert 0 <= n_accept <= n_steps


def test_local_snapshot_runner_first_snapshot_is_initial_state():
    x = init_phase_separated(D_SMALL, side=0)
    snapshots, _, _ = run_local_swap_chain_snapshots(
        x.copy(), D_SMALL, 0.10, 10 * d_SMALL, 3, d_SMALL
    )
    np.testing.assert_array_equal(snapshots[0], x.astype(np.int8))


# ---------------------------------------------------------------------------
# (2) sigma = 0: acceptance = P(directed NN bond is unlike) = d / (2*(d-1))
# ---------------------------------------------------------------------------


def test_local_snapshot_runner_sigma0_acceptance_matches_unlike_bond_rate():
    rng = np.random.default_rng(1)
    x = init_random_at_composition(d_SMALL, 0.5, rng)
    n_steps = 200_000
    _, _, n_accept = run_local_swap_chain_snapshots(
        x, D_SMALL, 0.0, n_steps, 11, d_SMALL
    )
    # P(unlike) for a fixed bond under the uniform slice measure:
    # 2 * (d/2) * (d/2) / (d * (d-1)) = d / (2*(d-1)) = 8/15 at d = 16.
    expected = d_SMALL / (2 * (d_SMALL - 1))
    assert abs(n_accept / n_steps - expected) < 0.02


# ---------------------------------------------------------------------------
# (3) split-half R-hat wrapper
# ---------------------------------------------------------------------------


def test_split_half_rhat_near_one_for_matching_stationary_chains():
    rng = np.random.default_rng(2)
    chains = rng.standard_normal((2, 20_000))
    assert split_half_gelman_rubin(chains) < 1.01


def test_split_half_rhat_large_for_chains_with_different_means():
    rng = np.random.default_rng(3)
    chains = rng.standard_normal((2, 5_000))
    chains[1] += 10.0
    assert split_half_gelman_rubin(chains) > 2.0


def test_split_half_rhat_catches_shared_drift_plain_rhat_misses():
    # Both chains drift upward together: plain R-hat compares only chain
    # means (equal), split-half sees each chain's own halves disagreeing.
    rng = np.random.default_rng(4)
    drift = np.linspace(0.0, 5.0, 4_000)
    chains = rng.standard_normal((2, 4_000)) + drift
    split_rhat = split_half_gelman_rubin(chains)
    assert split_rhat > 1.5
    assert gelman_rubin(chains) < split_rhat


def test_split_half_rhat_odd_length_drops_middle_sample():
    rng = np.random.default_rng(5)
    chains = rng.standard_normal((2, 10_001))
    assert split_half_gelman_rubin(chains) < 1.01


# ---------------------------------------------------------------------------
# (4) mchammer canonical probe runner (D = 4 tiny run)
# ---------------------------------------------------------------------------


def test_run_canonical_probe_tiny_run():
    sigma = 0.10
    initial = init_phase_separated(D_SMALL, side=0)
    snapshot_interval = 10 * d_SMALL  # 10 sweeps between snapshots
    n_proposals = 20 * snapshot_interval  # exactly 20 snapshots
    result = run_canonical_probe(
        D=D_SMALL,
        sigma=sigma,
        initial_spins=initial,
        n_proposals=n_proposals,
        snapshot_interval=snapshot_interval,
        seed=123,
    )
    snapshots = result["snapshots"]
    assert snapshots.shape == (20, d_SMALL)
    assert snapshots.dtype == np.int8

    # composition constant across every snapshot (swap moves, no penalty)
    assert np.all((snapshots == 1).sum(axis=1) == d_SMALL // 2)
    assert result["composition_is_constant"]

    # initial spins respected: snapshot 0 is the seeded state, so the
    # seeded phi mode (side=0 -> +domain in the left half-columns) survives
    # the spins -> symbols -> spins round trip.
    np.testing.assert_array_equal(snapshots[0], initial.astype(np.int8))
    assert left_minus_right(snapshots[0].astype(np.int64), D_SMALL) > 0

    # exact proposal-count currency
    assert result["n_proposals"] == n_proposals
    assert 0 <= result["n_accepted"] <= n_proposals
    assert result["wall_seconds_run"] > 0.0

    # cross-engine energy check: mchammer's recomputed potential at each
    # snapshot must equal -log p_tilde(x) = -sigma * x^T A x from OUR
    # adjacency reading of the same snapshot (natural units kT = 1).
    potential = result["potential_per_snapshot"]
    assert potential.shape == (20,)
    for k in range(20):
        log_p_tilde = initial_log_prob_ising(
            snapshots[k].astype(np.int64), D_SMALL, sigma
        )
        assert potential[k] == pytest.approx(-log_p_tilde, abs=1e-6)


def test_run_canonical_probe_rejects_partial_snapshot_interval():
    # A trailing partial interval would silently undercount acceptance
    # (acceptance_ratio rows only land on full write intervals), so the
    # contract is proposals % interval == 0, enforced loudly.
    with pytest.raises(ValueError):
        run_canonical_probe(
            D=D_SMALL,
            sigma=0.10,
            initial_spins=init_phase_separated(D_SMALL, side=0),
            n_proposals=170,
            snapshot_interval=160,
            seed=1,
        )
