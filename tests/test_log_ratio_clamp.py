"""The neighbour log-ratio clamp under a composition-penalised target.

WHY THIS FILE EXISTS
--------------------
`residual_lenet` (the loss) and `_compute_xi_t_lenet` (the control-variate
integrand) both clamp

    log_ratio(x, i, tau) = log p̃_t(y) - log p̃_t(x),   y = x with site i -> tau

at a ceiling, following DNFS App. E.1.1, which fixes it at 5. That constant
was calibrated for an *unmodified* Ising target, whose single-flip neighbour
log-ratios are O(a few). Adding the VCSGC-style composition penalty

    penalty(x) = lambda * d * (c(x) - c_target)^2,   c(x) = fraction of +1

changes that scale, because flipping one site moves c by exactly 1/d:

    penalty(y) - penalty(x) = lambda * d * [(Delta -+ 1/d)^2 - Delta^2]
                            = -+ 2 * lambda * Delta  +  lambda / d
                                                       with Delta = c(x) - c_target

and log p̃_t carries the penalty scaled by t (Eq. 4), so the penalty's
contribution to the neighbour log-ratio is t * (-+ 2*lambda*Delta + lambda/d).

The consequence, which is the load-bearing claim of the amortisation
post-mortem: the ceiling binds as soon as

    Delta > ceiling / (2 * lambda * t),

i.e. at t = 1 a threshold Delta* = ceiling / (2*lambda) that is INDEPENDENT
OF d. At lambda = 50 and the paper's ceiling of 5 that is Delta* = 0.05 —
smaller than the obedience error a conditioned sampler actually achieves,
so the clamp binds during normal training rather than on rare outliers.

Binding is a BIAS, not noise: once the ceiling is active the inflow term is
computed against exp(ceiling) instead of the true ratio, so no rate field
makes the residual zero and the loss acquires an irreducible floor.

These tests pin that arithmetic so the dissertation's claim cannot silently
drift from the code, and pin the contract that the loss and the control
variate must clamp identically — a mismatch there would bias the CV against
the objective it is supposed to be a control for, with no visible symptom.
"""

import pytest
import torch

from discrete_flow_sampler.samplers._neighbours import _log_p_tilde_at_neighbours
from discrete_flow_sampler.samplers.ctmc import _compute_xi_t_lenet
from discrete_flow_sampler.samplers.kolmogorov import (
    DEFAULT_LOG_RATIO_CLAMP,
    residual_lenet,
)
from discrete_flow_sampler.targets.ising import IsingTarget


def _target(lam: float, c_target: float = 0.5, D: int = 4, clamp: float | None = None):
    kwargs = {} if clamp is None else {"log_ratio_clamp": clamp}
    return IsingTarget(
        D=D,
        sigma=0.1,
        target_composition=c_target,
        composition_penalty_strength=lam,
        **kwargs,
    )


class _ConstantRateModel(torch.nn.Module):
    """Locally equivariant stand-in emitting a fixed G, so the only thing
    varying between assertions is the target's neighbour log-ratio.

    Sign matters. `site_terms = [G]_+ - [-G]_+ * exp(log_ratio)`, so only the
    NEGATIVE part of G multiplies the clamped ratio. A model emitting G > 0
    everywhere has zero inflow term and is completely blind to the clamp —
    which is why the tests below that exercise the ceiling pass `value < 0`.
    """

    is_locally_equivariant = True
    vocab_size = 2

    def __init__(self, n_sites: int, value: float = 0.3):
        super().__init__()
        self.n_sites = n_sites
        self.value = value

    def forward(self, x, t):
        G = torch.full(
            (x.shape[0], self.n_sites, self.vocab_size), self.value, dtype=x.dtype
        )
        # The tau = x_i slot is zero by construction in the real leTF readout.
        x_idx = ((x + 1.0) * 0.5).long()
        G.scatter_(-1, x_idx[..., None], 0.0)
        return G


def test_paper_default_is_five():
    """The shipped default must remain the paper's App. E.1.1 value, so that
    turning the knob is an explicit experimental act and never a silent
    change to the replication."""
    assert DEFAULT_LOG_RATIO_CLAMP == 5.0
    assert _target(lam=0.0).log_ratio_clamp == 5.0


@pytest.mark.parametrize("lam", [10.0, 25.0, 50.0])
@pytest.mark.parametrize("delta", [0.10, 0.25])
def test_penalty_contribution_to_neighbour_log_ratio(lam, delta):
    """The derivation itself: at t = 1 the penalty shifts the neighbour
    log-ratio by exactly -+2*lambda*Delta + lambda/d relative to the
    unpenalised target.

    Measured as a difference against a lambda = 0 target on the SAME state,
    so the Ising part cancels and only the penalty term survives.
    """
    D = 4
    d = D * D
    c_target = 0.5
    n_plus = round((c_target + delta) * d)
    x = torch.cat([torch.ones(n_plus), -torch.ones(d - n_plus)])[None, :]
    t = torch.ones(1)

    achieved = ((x + 1.0) * 0.5).mean().item()
    delta_exact = achieved - c_target

    def ratios(target):
        lp_n = _log_p_tilde_at_neighbours(x, t, target, vocab_size=2)
        return lp_n - target.log_p_tilde_t(x, t)[:, None, None]

    shift = ratios(_target(lam, c_target, D)) - ratios(_target(0.0, None, D))

    plus_idx = ((x[0] + 1.0) * 0.5).long()  # 1 where site is +1
    # Flipping a +1 site to -1 lowers c; with Delta > 0 that moves toward
    # target, so the penalty falls and the log-ratio RISES by 2*lambda*Delta.
    down_flips = shift[0, plus_idx == 1, 0]
    up_flips = shift[0, plus_idx == 0, 1]

    expected_down = 2 * lam * delta_exact - lam / d
    expected_up = -2 * lam * delta_exact - lam / d
    assert torch.allclose(
        down_flips, torch.full_like(down_flips, expected_down), atol=1e-3
    )
    assert torch.allclose(up_flips, torch.full_like(up_flips, expected_up), atol=1e-3)


@pytest.mark.parametrize("lam", [10.0, 25.0, 50.0])
def test_clamp_threshold_is_delta_star_and_is_d_independent(lam):
    """Delta* = ceiling / (2*lambda) predicts the onset of clamping, and the
    same Delta* holds at d = 16 and d = 100.

    This is the claim that explains why the identical recipe survives at
    4x4 and dies at 10x10 *without* invoking a d-dependent loss scale: the
    threshold does not move with d, so what differs between the sizes is the
    obedience error the model actually achieves, not the threshold it must
    stay under.
    """
    ceiling = DEFAULT_LOG_RATIO_CLAMP
    delta_star = ceiling / (2 * lam)
    t = torch.ones(1)

    for D in (4, 10):
        d = D * D
        target = _target(lam, 0.5, D)

        def clamped_fraction(delta):
            n_plus = round((0.5 + delta) * d)
            x = torch.cat([torch.ones(n_plus), -torch.ones(d - n_plus)])[None, :]
            lp_n = _log_p_tilde_at_neighbours(x, t, target, vocab_size=2)
            log_ratio = lp_n - target.log_p_tilde_t(x, t)[:, None, None]
            return (log_ratio > ceiling).float().mean().item()

        # Comfortably inside the threshold: the Ising part alone is O(a few),
        # so nothing saturates.
        assert clamped_fraction(0.5 * delta_star) == 0.0
        # Well outside it: the penalty term alone is >= 2x the ceiling.
        assert clamped_fraction(2.0 * delta_star) > 0.0


def test_loss_and_control_variate_clamp_identically():
    """`residual_lenet` and `_compute_xi_t_lenet` must use the same ceiling.

    They are the objective and its control variate; if they disagreed, the
    CV would be centred on a different quantity than the loss it corrects,
    biasing training with no diagnostic that would show it. The two differ
    only by the `- dt_log_Zt` shift, so passing dt_log_Zt = 0 makes them
    equal by construction whenever the clamps match.
    """
    D, lam = 4, 50.0
    d = D * D
    target = _target(lam, 0.5, D, clamp=17.0)
    model = _ConstantRateModel(d)

    torch.manual_seed(0)
    x = torch.where(torch.rand(8, d) < 0.75, 1.0, -1.0)
    t = torch.rand(8)

    residual = residual_lenet(x, t, torch.zeros(()), model, target)
    xi = _compute_xi_t_lenet(x, t, model, target)
    assert torch.allclose(residual, xi, atol=1e-5)


def test_raising_the_clamp_changes_the_residual_only_where_it_bound():
    """Turning the knob is inert when nothing saturates and material when
    something does — which is what makes `log_ratio_clamp_frac` readable as
    an early warning rather than merely correlated with trouble."""
    D, lam = 4, 50.0
    d = D * D
    model = _ConstantRateModel(d, value=-0.3)  # negative: see class docstring
    t = torch.ones(4)

    # c = 0.5 exactly: Delta = 0, penalty differences vanish, Ising part is
    # O(a few) so no ceiling in [5, 20] can bind.
    obedient = torch.cat([torch.ones(d // 2), -torch.ones(d // 2)])[None, :].repeat(
        4, 1
    )
    r5 = residual_lenet(obedient, t, torch.zeros(()), model, _target(lam, 0.5, D, 5.0))
    r20 = residual_lenet(
        obedient, t, torch.zeros(()), model, _target(lam, 0.5, D, 20.0)
    )
    assert torch.allclose(r5, r20, atol=1e-5)

    # Delta = 0.25, five times Delta* = 0.05: the ceiling is load-bearing, and
    # the size of the gap is the point. The true ratio at the favourable
    # neighbours is exp(2*lambda*Delta) = exp(25), so relaxing the ceiling
    # does not gently correct the residual — it swaps a bounded bias for an
    # astronomically large term. Both regimes are unusable, which is why the
    # deployable fix has to flatten the penalty rather than raise the ceiling.
    n_plus = round(0.75 * d)
    disobedient = torch.cat([torch.ones(n_plus), -torch.ones(d - n_plus)])[
        None, :
    ].repeat(4, 1)
    b5 = residual_lenet(
        disobedient, t, torch.zeros(()), model, _target(lam, 0.5, D, 5.0)
    )
    b20 = residual_lenet(
        disobedient, t, torch.zeros(()), model, _target(lam, 0.5, D, 20.0)
    )
    assert not torch.allclose(b5, b20, atol=1.0)
    assert b20.abs().max() > 1e3 * b5.abs().max()


def test_clamp_must_be_positive():
    with pytest.raises(ValueError, match="log_ratio_clamp"):
        _target(lam=50.0, clamp=0.0)
