"""Falsification tests for the separable band-score lever (opt-in, default OFF).

Written BEFORE the lever: these encode what correct looks like independently
of how the separation is implemented.

WHAT THE LEVER IS -- and what it is NOT.

It is an EXACT algebraic identity on the masked-attention band, not a
modelling change. The word "factorised" is deliberately avoided in the flag
name because this codebase already spends it twice on something else: the
factorised swap head is an APPROXIMATION with a measured price (a constant
multiplicative factor on per-site log-weight variance, 3.1x at d=16 and
2.6x at d=64 against the masked-attention twin), and the print claims about
that head overstate how much per-pair work it removes. This lever computes
the SAME FUNCTION the archived band computes, so it has no variance price
and nothing to trade off; it only declines to build one tensor.

THE IDENTITY. Per family, the band scores a pair (i, j) against term k as

    s_ijk = <W_q [rho_i || rho_j] + b, key_k> * scale

The query is a single `nn.Linear` on a CONCATENATION, so W_q splits
column-wise into [W1 | W2] and the score is an OUTER SUM

    s_ijk = A_ik + B_jk,   A_ik = <rho_i W1^T + b, key_k> * scale
                           B_jk = <rho_j W2^T,     key_k> * scale

The visibility mask separates too, in BOTH windows -- interval as
1[k + min(O) > i] * 1[k + max(O) < j], lattice as
prod_o 1[k+o != i] * prod_o 1[k+o != j]. So with alpha_ik = u_ik e^{A_ik}
and beta_jk = w_jk e^{B_jk}, both halves of the masked softmax are matrix
products:

    Z_ij     = sum_k alpha_ik beta_jk
    out_ijf  = (1/Z_ij) sum_k alpha_ik beta_jk feat_kf

and the (B, d^2, n) score tensor -- this head's largest object, 5.00 GB at
B=32/d=256, and the reason the module docstring says "d = 256 wants pair
chunking" -- never exists.

WHAT COULD GO WRONG, and therefore what is pinned below:

  * THERE IS NO SOFTMAX SHIFT, AND THAT IS A DECISION. The separable form
    would have to stabilise with the upper bound max_k A_ik + max_k B_jk,
    and that row maximum ranges over u_i -- a set CONTAINING the partner
    hole k = j. A common shift cancels analytically but not bit-for-bit
    (exp(s - c) rounds differently for different c), so the hole's token
    value reaches the answer in the last bits: measured 2.98e-8 before the
    shift was dropped. Blindness is worth more than the shift, whose only
    job is keeping exp inside the exponent budget -- so that budget becomes
    a MONITORED quantity (fp32 overflows near +88, a product of halves
    underflows near -87, against a trained checkpoint's measured [-6.65,
    10.69]) rather than a structural guarantee. Pinned by a range test.

  * EMPTY BANDS ARE A 0/0. Adjacent pairs, and any pair with
    j - i < delta + 2, see no visible term. The archived path softmaxes a
    fully-masked row to a discarded uniform (finite fill, never -inf) and
    overwrites it with exact 0.0. Here Z_ij is exactly 0, so the division
    must be guarded BEFORE `torch.where` -- an unguarded 0/0 returns NaN,
    and `torch.where` propagates NaN through the UNSELECTED branch in
    BACKWARD even though forward looks clean. That is the failure mode the
    gradient test below exists to catch, and it is why the empty-band case
    is constructed explicitly rather than left to chance.

  * BLINDNESS MUST STAY BIT-EXACT. The archived path relies on
    exp(-1e9 - max) underflowing to +0.0. The separable path masks
    MULTIPLICATIVELY, so an excluded term contributes exactly zero by
    construction -- a strictly stronger guarantee, and it is asserted as
    equality rather than as a tolerance.

  * THE FLAG MUST BE FREE WHEN OFF. No parameter, no buffer, no RNG draw,
    so every checkpoint on disk still loads and every archived cell still
    reproduces bit-for-bit.

NOT COVERED, deliberately: `pair_position_mode="relative"`. There the query
is one embedding vector per PAIR, not a linear map on a concatenation, so
the identity is false. The head must refuse the combination rather than
silently compute a different function; that refusal is the only thing
tested about it.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix

# Equivalence bar. The two paths differ only in contraction order and in
# whether the softmax is stabilised at all, so the drift is fp round-off on
# quantities of order 1 -- the same bit-class the triu-pair gather is held to,
# and two orders tighter than the suite's blindness ATOL.
SEPARABLE_ATOL = 1e-5

# fp32 denormals vanish near exp(-103) and overflow arrives at exp(+88), so
# +-87 is the practical single-precision exponent budget the UNSHIFTED pool
# spends. bf16 has the same exponent range, so it buys no extra room.
UNDERFLOW_KNEE = 87.0

LATTICE_SIDE = 4
D = LATTICE_SIDE * LATTICE_SIDE


def _backbone(d=D, seed=42):
    torch.manual_seed(seed)
    return LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=8, n_layers=2, n_heads=2,
        use_sdpa_readout=False,
    )


def _head(separable, **kw):
    torch.manual_seed(0)
    return MaskedAttentionSwapHead(
        _backbone(), pair_offsets=(1, LATTICE_SIDE), band_feature_dim=6,
        position_dim=6, attention_dim=6, lattice_side=LATTICE_SIDE,
        separable_band_scores=separable, **kw,
    ).eval()


# One case per code path the lever touches: both windows (their masks
# separate for DIFFERENT reasons -- an inequality pair vs a product of
# disequalities), the stencil family (five offsets, so the widest support
# and the most conservative straddle exclusion), the triu-pair gather (the
# pair axes collapse to a list, so alpha and beta must be GATHERED rather
# than outer-multiplied), and the bilinear exterior (which changes what
# consumes the band, not the band itself -- a regression guard).
HEAD_CASES = {
    "interval": {},
    "lattice": {"attention_window": "lattice"},
    "interval_stencil": {"use_stencil": True},
    "lattice_stencil": {"attention_window": "lattice", "use_stencil": True},
    "interval_gathered": {"gather_triu_pairs": True},
    "lattice_gathered": {"attention_window": "lattice", "gather_triu_pairs": True},
    "interval_bilinear": {"exterior_combiner": "bilinear"},
}


def _pair(case):
    """The same head built twice, flag off and flag on. Same seed and no RNG
    draw from the flag => identical parameters, so any output difference is
    the lever and nothing else."""
    return _head(False, **HEAD_CASES[case]), _head(True, **HEAD_CASES[case])


def _state(batch=2, d=D, seed=7):
    """Fixed-composition spins: both values present in every row, so the
    band features are never degenerate."""
    torch.manual_seed(seed)
    half = torch.cat([torch.ones(d // 2), -torch.ones(d - d // 2)])
    return torch.stack([half[torch.randperm(d)] for _ in range(batch)])


def _drift(a, b):
    return (a - b).abs().max().item()


def _train_scale(head, gain=3.0):
    """Stand-in for trained weights. The identity is exact at any weight, but
    the EXPONENT BUDGET is not scale-free: without a stabilising shift the
    live score range is what has to clear +-87. Init weights are the easy
    case; a trained 4x4 checkpoint measures [-6.65, 10.69]. Scaling the
    query/key projections reproduces the hard case without depending on a
    results directory that is not in the repo."""
    with torch.no_grad():
        for projection in (*head.band_query_projections,
                           *head.band_key_projections):
            projection.weight.mul_(gain)
            projection.bias.mul_(gain)
    return head


# --------------------------------------------------------------------------
# Equivalence: the lever computes the same function.
# --------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_separable_band_matches_dense(case):
    """The band summaries themselves -- the tensor the lever rewrites."""
    dense, separable = _pair(case)
    x, t = _state(), torch.rand(2)
    drift = _drift(separable.band_summaries(x, t), dense.band_summaries(x, t))
    assert drift < SEPARABLE_ATOL, f"{case}: band moved by {drift:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_separable_scores_match_dense(case):
    """End to end: G is the same matrix either way. Guards against the band
    matching while something downstream consumes it differently."""
    dense, separable = _pair(case)
    x, t = _state(), torch.rand(2)
    drift = _drift(separable(x, t), dense(x, t))
    assert drift < SEPARABLE_ATOL, f"{case}: G moved by {drift:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_separable_matches_dense_at_trained_weight_scale(case):
    """The equivalence must not be an init-only artefact. Widening the score
    range is exactly what stresses the non-separable softmax shift."""
    dense, separable = _train_scale(_pair(case)[0]), _train_scale(_pair(case)[1])
    x, t = _state(), torch.rand(2)
    drift = _drift(separable.band_summaries(x, t), dense.band_summaries(x, t))
    assert drift < SEPARABLE_ATOL, f"{case}: band moved by {drift:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("case", ["interval", "lattice"])
def test_separable_matches_dense_under_bf16_autocast(case):
    """bf16 has fp32's exponent range, so the shift bound is no tighter
    there; only the mantissa is, which is why the bar is loosened rather
    than the test dropped."""
    dense, separable = _pair(case)
    x, t = _state(), torch.rand(2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        drift = _drift(separable.band_summaries(x, t), dense.band_summaries(x, t))
    assert drift < 5e-2, f"{case}: band moved by {drift:.2e} under bf16"


# --------------------------------------------------------------------------
# The two structural claims the identity rests on.
# --------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("window", ["interval", "lattice"])
@pytest.mark.parametrize("offsets", [(0,), (0, 1), (-LATTICE_SIDE, -1, 0, 1, LATTICE_SIDE)])
def test_visibility_mask_is_separable(window, offsets):
    """CLAIM 1: visible(i, j, k) = u(i, k) AND w(j, k).

    This is what lets the mask multiply into alpha and beta instead of
    filling scores. If it failed for any window or support the whole lever
    would be wrong, so it is checked against the head's OWN predicate rather
    than against a re-derivation."""
    head = _head(True, attention_window=window)
    n_terms = D - max(0, max(offsets))
    slot = torch.arange(n_terms)
    site_i, site_j = torch.arange(D).view(D, 1, 1), torch.arange(D).view(1, D, 1)
    dense = head._family_visibility(slot.view(1, 1, -1), site_i, site_j, offsets)
    u, w = head._family_visibility_halves(slot, offsets)
    assert torch.equal(u.unsqueeze(1) & w.unsqueeze(0), dense)


@torch.no_grad()
def test_excluded_terms_carry_exactly_zero_weight():
    """CLAIM 2, strengthened: the archived path leans on exp(-1e9 - max)
    underflowing to +0.0; the separable path multiplies by a boolean, so
    exclusion is zero by construction. Blindness is therefore bit-exact
    without a numerical argument -- assert equality, not a tolerance."""
    head = _train_scale(_head(True))
    x, t = _state(), torch.rand(2)
    band = head.band_summaries(x, t)
    flipped = x.clone()
    flipped[:, 3] *= -1        # move a hole's token value
    flipped[:, 11] *= -1
    moved = head.band_summaries(flipped, t)
    assert torch.equal(band[:, 3, 11], moved[:, 3, 11])
    assert torch.equal(band[:, 11, 3], moved[:, 11, 3])


# --------------------------------------------------------------------------
# Empty bands: the 0/0 the archived path never had to face.
# --------------------------------------------------------------------------


@torch.no_grad()
def test_empty_bands_are_exactly_zero_and_finite():
    """Adjacent pairs see no visible term under the interval window, so
    Z_ij = 0 exactly. The archived path softmaxes a fully-masked row to a
    discarded uniform; here the division must be guarded."""
    head = _train_scale(_head(True))
    band = head.band_summaries(_state(), torch.rand(2))
    assert torch.isfinite(band).all()
    adjacent = torch.arange(D - 1)
    assert torch.equal(band[:, adjacent, adjacent + 1], torch.zeros_like(
        band[:, adjacent, adjacent + 1]))


def test_gradients_are_finite_with_empty_bands_present():
    """THE failure this lever can introduce. An unguarded 0/0 gives NaN, and
    `torch.where(cond, nan, 0.0)` is clean in FORWARD but propagates NaN
    through the unselected branch in BACKWARD. Adjacent pairs guarantee
    empty bands are present, so this exercises the guard rather than hoping
    to hit it.

    `sum(G**2)`, not `sum(G)`: G is exactly antisymmetric, so its plain sum
    is identically zero as a function of the parameters and every gradient
    would vanish -- a dead objective that passes a dead module."""
    head = _train_scale(_head(True)).train()
    loss = head(_state(), torch.rand(2)).pow(2).sum()
    loss.backward()
    for name, parameter in head.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f"non-finite grad: {name}"


@pytest.mark.parametrize("case", ["interval", "lattice", "interval_gathered"])
def test_separable_gradients_match_dense(case):
    """Untested before this file: a forward identity to 1e-7 does not by
    itself pin BACKWARD. The two paths take different routes to the same
    scalar, so the gradients must be compared parameter by parameter."""
    dense, separable = _pair(case)
    x, t = _state(), torch.rand(2)
    for head in (dense.train(), separable.train()):
        head(x, t).pow(2).sum().backward()
    dense_grads = dict(dense.named_parameters())
    for name, parameter in separable.named_parameters():
        reference = dense_grads[name].grad
        if reference is None:
            continue
        drift = _drift(parameter.grad, reference)
        scale = reference.abs().max().clamp_min(1.0).item()
        assert drift / scale < SEPARABLE_ATOL, (
            f"{case}: grad of {name} moved by {drift:.2e}"
        )


# --------------------------------------------------------------------------
# The numerical caveat, measured rather than asserted away.
# --------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("window", ["interval", "lattice"])
def test_score_range_keeps_exponent_headroom(window):
    """Dropping the softmax shift buys bit-exact blindness and spends the
    float's exponent budget instead, so that budget has to be watched. The
    weight scale here exceeds the trained 4x4 checkpoint's measured range of
    [-6.65, 10.69] several times over and must still clear the knee with room
    to spare."""
    head = _train_scale(_head(True, attention_window=window), gain=4.0)
    widest = head.band_score_range(_state())
    assert widest < UNDERFLOW_KNEE / 2.0, (
        f"{window}: scores span {widest:.1f} nats against a {UNDERFLOW_KNEE:.0f} "
        f"budget -- the unshifted pool is close to overflow"
    )


@torch.no_grad()
def test_score_range_reads_the_halves_not_the_pair_tensor():
    """The monitor must not reintroduce the object the lever removes: it is
    O(d n), so it stays affordable at the d where it actually matters."""
    head = _head(True)
    assert head.band_score_range(_state()) > 0.0


# --------------------------------------------------------------------------
# The flag must be free when off, and must refuse what it cannot compute.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_flag_adds_no_state_and_no_rng_draw(case):
    """Default OFF buys archived-checkpoint reproducibility; it only holds
    if the flag draws no parameter and registers no buffer."""
    dense, separable = _pair(case)
    assert list(dense.state_dict()) == list(separable.state_dict())
    for name, parameter in dense.state_dict().items():
        assert torch.equal(parameter, separable.state_dict()[name]), name


def test_separable_refuses_the_relative_pair_position_code():
    """With a relative code the query is ONE embedding row per pair, not a
    linear map on [rho_i || rho_j], so s_ijk is not an outer sum and the
    identity is false. Refuse loudly: silently computing a different
    function is the one outcome an exactness claim cannot survive."""
    with pytest.raises(ValueError, match="separable"):
        _head(True, pair_position_mode="relative")


def test_config_flag_reaches_the_head():
    """`cfg.separable_band_scores` must arrive at the head it is set for."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    cfg = replace(
        CONFIGS["H2_d16_c50_s010_letf_ma_10k"], separable_band_scores=True
    )
    head = build_swap_head(cfg, _backbone())
    assert head.separable_band_scores is True
