"""Tests for composition conditioning — the amortised sampler.

An amortised sampler is conditioned on the target composition c, so one trained
model serves many compositions instead of one specialist per composition. Two
halves.

Target. `IsingTarget.composition_batch(c)` binds a (B,) composition vector for
the duration of a block, so `composition_penalty` becomes per-row:

    penalty(x_b) = λ · d · (c₊(x_b) − c_b)²                [was: scalar c]

The load-bearing property is the b-major expansion rule. Every batch-expanding
call site in this codebase (`_log_p_tilde_at_neighbours`,
`kolmogorov.residual_general`, `ctmc._compute_xi_t_general`) expands the batch
axis b-major by an integer factor and rides `t` along with
`t.repeat_interleave(k)`. A bound composition vector must ride along by exactly
the same rule, or row b's penalty silently gets row b''s target composition — a
bias that would never raise, only degrade.

Model. c is embedded by a second `TimestepEmbedder` and summed into the existing
`(B, 1, h)` conditioning tensor. Three properties matter:

  1. Hollowness (Def. 3) survives — c is independent of x entirely, so the
     existing slice-and-mask argument is untouched.
  2. Local equivariance (Eq. 20) survives, for the same reason.
  3. Zero-init ⇒ the c-channel is inert at initialisation, so an
     unconditioned specialist checkpoint warm-starts a conditioned model
     with identical outputs. This is what makes "conditioning off" a free
     ablation rather than a second code path.
"""

import pytest
import torch

from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._neighbours import _log_p_tilde_at_neighbours
from discrete_flow_sampler.targets.ising import IsingTarget

# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

LAMBDA = 50.0


def _soft_target(D=4, target_composition=0.5):
    return IsingTarget(
        D=D,
        sigma=0.1,
        target_composition=target_composition,
        composition_penalty_strength=LAMBDA,
    )


def _random_spins(batch_size, d, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch_size, d), generator=generator).float() * 2 - 1


def _penalty_by_loop(target, x, compositions):
    """Reference: evaluate one row at a time with a scalar-c target."""
    original = target.target_composition
    values = []
    for row, composition in zip(x, compositions):
        target.target_composition = float(composition)
        values.append(target.composition_penalty(row[None, :])[0])
    target.target_composition = original
    return torch.stack(values)


# --------------------------------------------------------------------------
# Target — per-row composition
# --------------------------------------------------------------------------


def test_unbound_penalty_is_unchanged_by_the_new_machinery():
    """With nothing bound, every specialist run is untouched.

    The six archived specialists in results/02_constrained_soft were trained
    against the scalar-c penalty. If this test fails, previously reported
    numbers are no longer reproducible from this code.
    """
    target = _soft_target(target_composition=0.3)
    x = _random_spins(8, target.d)
    expected = LAMBDA * target.d * (target.composition_fraction(x) - 0.3).pow(2)
    torch.testing.assert_close(target.composition_penalty(x), expected)


def test_bound_penalty_matches_per_row_scalar_evaluation():
    """A mixed-c batch equals looping row-by-row with a scalar-c target."""
    target = _soft_target()
    x = _random_spins(6, target.d, seed=1)
    compositions = torch.tensor([0.30, 0.35, 0.50, 0.575, 0.65, 0.80])

    with target.composition_batch(compositions):
        got = target.composition_penalty(x)

    torch.testing.assert_close(got, _penalty_by_loop(target, x, compositions))


def test_bound_composition_expands_b_major():
    """An expanded batch inherits its parent row's c.

    `x.repeat_interleave(k, dim=0)` is the expansion every neighbour helper
    performs. Row (b*k + j) must carry composition c[b] for every j — the
    same rule `t.repeat_interleave(k)` already obeys.
    """
    target = _soft_target()
    x = _random_spins(4, target.d, seed=2)
    compositions = torch.tensor([0.30, 0.50, 0.65, 0.80])
    expansion = 3

    x_expanded = x.repeat_interleave(expansion, dim=0)
    with target.composition_batch(compositions):
        got = target.composition_penalty(x_expanded)

    expected = _penalty_by_loop(
        target, x_expanded, compositions.repeat_interleave(expansion)
    )
    torch.testing.assert_close(got, expected)


def test_bound_composition_rejects_non_divisible_batch():
    """A batch that is not an integer multiple of the bound vector is a bug.

    Silently broadcasting here would give some rows the wrong target
    composition, which shows up only as a quietly worse ESS. Raise instead.
    """
    target = _soft_target()
    x = _random_spins(7, target.d)
    with target.composition_batch(torch.tensor([0.3, 0.5, 0.8])):
        with pytest.raises(ValueError, match="b-major|divis"):
            target.composition_penalty(x)


def test_composition_batch_restores_scalar_on_exit_and_on_exception():
    target = _soft_target(target_composition=0.3)
    x = _random_spins(4, target.d)
    scalar_penalty = target.composition_penalty(x)

    with target.composition_batch(torch.full((4,), 0.8)):
        pass
    torch.testing.assert_close(target.composition_penalty(x), scalar_penalty)

    with pytest.raises(RuntimeError):
        with target.composition_batch(torch.full((4,), 0.8)):
            raise RuntimeError("boom")
    torch.testing.assert_close(target.composition_penalty(x), scalar_penalty)


def test_bound_composition_reaches_the_annealing_path():
    """The penalty is reached via log_prob → log_p_tilde_t / dt_log_p_tilde_t.

    Binding must change both, and must not change either at t=0, where the
    path is pure base η and stays composition-independent.
    """
    target = _soft_target(target_composition=0.5)
    x = _random_spins(4, target.d, seed=3)
    compositions = torch.tensor([0.30, 0.50, 0.65, 0.80])
    t_mid = torch.full((4,), 0.5)
    t_zero = torch.zeros(4)

    baseline_mid = target.log_p_tilde_t(x, t_mid)
    baseline_zero = target.log_p_tilde_t(x, t_zero)
    baseline_dt = target.dt_log_p_tilde_t(x, t_mid)

    with target.composition_batch(compositions):
        bound_mid = target.log_p_tilde_t(x, t_mid)
        bound_zero = target.log_p_tilde_t(x, t_zero)
        bound_dt = target.dt_log_p_tilde_t(x, t_mid)

    # Row 1 is bound to the scalar the target already held → unchanged.
    torch.testing.assert_close(bound_mid[1], baseline_mid[1])
    assert not torch.allclose(bound_mid[0], baseline_mid[0])
    assert not torch.allclose(bound_dt[0], baseline_dt[0])
    # t=0 is the base η, which carries no composition.
    torch.testing.assert_close(bound_zero, baseline_zero)


def test_neighbour_helper_respects_bound_composition():
    """Integration: the (B, d, S) neighbour expansion keeps rows aligned.

    This is where the b-major rule is exercised in anger — the helper reshapes
    to (B*d*S, d) and repeat_interleaves t by d*S. Compared against a per-row
    loop with a scalar-c target, which is unambiguous.
    """
    target = _soft_target()
    x = _random_spins(3, target.d, seed=4)
    compositions = torch.tensor([0.30, 0.55, 0.80])
    t = torch.tensor([0.4, 0.7, 0.9])

    with target.composition_batch(compositions):
        got = _log_p_tilde_at_neighbours(x, t, target, vocab_size=2)

    original = target.target_composition
    expected_rows = []
    for row, composition, t_row in zip(x, compositions, t):
        target.target_composition = float(composition)
        expected_rows.append(
            _log_p_tilde_at_neighbours(row[None, :], t_row[None], target, vocab_size=2)[
                0
            ]
        )
    target.target_composition = original

    torch.testing.assert_close(got, torch.stack(expected_rows))


# --------------------------------------------------------------------------
# Model — c summed into cond_t
# --------------------------------------------------------------------------


def _conditioned_model(d=9, hidden_dim=16, seed=0):
    torch.manual_seed(seed)
    model = LeTFRateMatrix(
        d=d,
        vocab_size=2,
        hidden_dim=hidden_dim,
        n_layers=2,
        n_heads=2,
        condition_on_composition=True,
    )
    model.eval()
    return model


def _excite_composition_channel(model, scale=1.0):
    """Undo the zero-init so the c-channel actually influences the output.

    At initialisation the composition embedding is identically zero by design;
    a test of "does c matter?" must first put the model in the state training
    would reach.
    """
    with torch.no_grad():
        final_linear = model.comp_embedder.mlp[-1]
        final_linear.weight.normal_(std=scale)
        final_linear.bias.normal_(std=scale)


def test_composition_channel_is_inert_at_initialisation():
    """Zero-init: any two compositions give identical rates before training."""
    model = _conditioned_model()
    x = _random_spins(4, model.d, seed=5)
    t = torch.rand(4)

    low = model(x, t, torch.full((4,), 0.30))
    high = model(x, t, torch.full((4,), 0.80))
    torch.testing.assert_close(low, high)


def test_conditioner_preserves_shared_initialisation_and_rng_stream():
    """A same-seed conditioned model must be a paired specialist at step 0.

    The composition module stays registered before the stacks for optimizer
    resume compatibility, but its private initialization must not advance the
    RNG used by any shared tensor or by code after model construction.
    """

    def build(conditioned):
        torch.manual_seed(1234)
        model = LeTFRateMatrix(
            d=9,
            vocab_size=2,
            hidden_dim=16,
            n_layers=2,
            n_heads=2,
            condition_on_composition=conditioned,
        )
        return model, torch.get_rng_state().clone()

    specialist, specialist_rng = build(False)
    conditioned, conditioned_rng = build(True)
    assert torch.equal(specialist_rng, conditioned_rng)

    specialist_state = specialist.state_dict()
    conditioned_state = conditioned.state_dict()
    for name, value in specialist_state.items():
        assert torch.equal(value, conditioned_state[name]), name
    assert set(conditioned_state) - set(specialist_state) == {
        "comp_embedder.mlp.0.weight",
        "comp_embedder.mlp.0.bias",
        "comp_embedder.mlp.2.weight",
        "comp_embedder.mlp.2.bias",
    }

    # Parameter registration order is how optimizer state is mapped on resume,
    # so parity must not be obtained by moving the module.
    names = [name for name, _ in conditioned.named_parameters()]
    assert names.index("comp_embedder.mlp.0.weight") < names.index(
        "fwd_stack.blocks.0.proj_in.weight"
    )


def test_unconditioned_checkpoint_warm_starts_a_conditioned_model():
    """The point of zero-init: a specialist checkpoint transfers exactly.

    Loading an unconditioned state_dict with strict=False leaves comp_embedder
    at its zero-init, so the conditioned model reproduces the specialist's
    rates bit-for-bit — a legitimate warm start, not an approximation.
    """
    torch.manual_seed(7)
    plain = LeTFRateMatrix(
        d=9,
        vocab_size=2,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
    )
    plain.eval()
    conditioned = _conditioned_model(seed=99)  # deliberately different init

    missing, unexpected = conditioned.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys when warm-starting: {unexpected}"
    assert all(key.startswith("comp_embedder.") for key in missing), (
        f"only the composition embedder may be missing, got {missing}"
    )

    x = _random_spins(4, 9, seed=6)
    t = torch.rand(4)
    torch.testing.assert_close(conditioned(x, t, torch.full((4,), 0.65)), plain(x, t))


def test_composition_changes_rates_once_the_channel_is_trained():
    """Guard against c being accepted and silently dropped."""
    model = _conditioned_model()
    _excite_composition_channel(model)
    x = _random_spins(4, model.d, seed=8)
    t = torch.rand(4)

    low = model(x, t, torch.full((4,), 0.30))
    high = model(x, t, torch.full((4,), 0.80))
    assert (low - high).abs().max() > 1e-4, (
        "composition input does not affect the rates — c is being dropped"
    )


def test_conditioned_model_is_hollow():
    """Def. 3 survives conditioning: G(·, i | x) must not depend on x_i.

    Hollowness is what makes Eq. (10)'s single-forward-pass residual valid;
    an excited c-channel must not perturb it.
    """
    model = _conditioned_model()
    _excite_composition_channel(model)
    x = _random_spins(2, model.d, seed=9)
    t = torch.rand(2)
    c = torch.tensor([0.30, 0.80])

    body = model.compute_body(x, t, c)
    for site in range(model.d):
        x_flipped = x.clone()
        x_flipped[:, site] *= -1
        body_flipped = model.compute_body(x_flipped, t, c)
        drift = (body[:, site] - body_flipped[:, site]).abs().max().item()
        assert drift < 1e-5, (
            f"hollowness broken at site {site}: body moved by {drift:.2e} "
            "when only x_site changed"
        )


def test_conditioned_model_is_locally_equivariant():
    """Eq. (20): G(τ, i | x) = −G(x_i, i | Swap(x, i, τ)), c held fixed."""
    model = _conditioned_model()
    _excite_composition_channel(model)
    x = _random_spins(2, model.d, seed=10)
    t = torch.rand(2)
    c = torch.tensor([0.35, 0.70])

    G = model(x, t, c)
    for site in range(model.d):
        x_swapped = x.clone()
        x_swapped[:, site] *= -1
        G_swapped = model(x_swapped, t, c)
        x_idx = ((x[:, site] + 1) / 2).long()
        tau_idx = 1 - x_idx
        forward = G[torch.arange(2), site, tau_idx]
        reverse = G_swapped[torch.arange(2), site, x_idx]
        torch.testing.assert_close(forward, -reverse, atol=1e-5, rtol=1e-4)


def test_conditioning_flag_and_argument_must_agree():
    """Mismatches are configuration bugs, not silent no-ops."""
    conditioned = _conditioned_model()
    x = _random_spins(2, conditioned.d)
    t = torch.rand(2)

    with pytest.raises(ValueError, match="composition"):
        conditioned(x, t)

    torch.manual_seed(0)
    plain = LeTFRateMatrix(d=9, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    with pytest.raises(ValueError, match="composition"):
        plain(x, t, torch.full((2,), 0.5))
