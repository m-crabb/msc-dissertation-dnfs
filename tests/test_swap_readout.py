"""Tests for the paired-swap antisymmetric readout (P1.1, 2026-06-29)."""

import pytest
import torch

from discrete_flow_sampler.constraints.swap_readout import (
    DoublyHollowSwapHead,
    LeTFMaskOneSwapHead,
    _masked_body,
    antisymmetrise,
    swap2,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix

ATOL = 1e-5


def _backbone(d=9, seed=42, hidden_dim=8, n_heads=2, n_layers=2, use_sdpa=False):
    torch.manual_seed(seed)
    m = LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=hidden_dim, n_layers=n_layers,
        n_heads=n_heads, use_sdpa_readout=use_sdpa,
    )
    m.eval()
    return m


def _state(d=9, seed=1):
    """A (1, d) ±1 state guaranteed to contain both spins."""
    torch.manual_seed(seed)
    x = (torch.randint(0, 2, (1, d)) * 2 - 1).float()
    if bool((x > 0).all()) or bool((x < 0).all()):
        x[0, 0] *= -1
    return x


def test_swap2_is_involution_and_exchanges():
    x = _state(d=9)
    i, j = 2, 5
    y = swap2(x, i, j)
    assert y[0, i].item() == x[0, j].item()
    assert y[0, j].item() == x[0, i].item()
    # off {i,j} unchanged
    mask = [k for k in range(9) if k not in (i, j)]
    assert torch.equal(y[0, mask], x[0, mask])
    # involution
    assert torch.equal(swap2(y, i, j), x)


@torch.no_grad()
@pytest.mark.parametrize("use_sdpa", [False, True])
def test_masked_body_double_hollow(use_sdpa):
    """H from masking anchor i: H[:,j,:] is blind to x_i (masked) and x_j (hollow)."""
    m = _backbone(d=9, use_sdpa=use_sdpa)
    x = _state(d=9)
    t = torch.rand(1)
    i, j = 2, 5
    H = _masked_body(m, x, t, (i,))
    base = H[:, j, :].clone()

    # flip x_i (masked) -> H[:,j,:] unchanged
    x_fi = x.clone()
    x_fi[0, i] *= -1
    d_i = (_masked_body(m, x_fi, t, (i,))[:, j, :] - base).abs().max().item()
    # flip x_j (hollow at j) -> H[:,j,:] unchanged
    x_fj = x.clone()
    x_fj[0, j] *= -1
    d_j = (_masked_body(m, x_fj, t, (i,))[:, j, :] - base).abs().max().item()

    if use_sdpa:
        # SDPA masking is structural (weight exactly 0): both blindness
        # properties must be EXACT flag-on, per the Tier-2 evidence bar.
        assert d_i == 0.0 and d_j == 0.0, (
            f"SDPA double-hollowness not exact: d_i={d_i:.2e}, d_j={d_j:.2e}"
        )
    assert d_i < ATOL, f"H_ij not blind to x_i: {d_i:.2e}"
    assert d_j < ATOL, f"H_ij not hollow in x_j: {d_j:.2e}"


def _active_pairs(x):
    d = x.shape[1]
    return [(i, j) for i in range(d) for j in range(i + 1, d) if x[0, i] != x[0, j]]


def _all_pairs(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def _state_batch(d=16, batch=3, seed=7):
    """A (batch, d) ±1 state batch; every row guaranteed to contain both spins."""
    torch.manual_seed(seed)
    x = (torch.randint(0, 2, (batch, d)) * 2 - 1).float()
    for b in range(batch):
        if bool((x[b] > 0).all()) or bool((x[b] < 0).all()):
            x[b, 0] *= -1
    return x


@torch.no_grad()
def test_doubly_hollow_antisymmetric():
    """Assertion 1: state-swap antisymmetry, bit-exact (< atol)."""
    for d in (9, 16):
        m = _backbone(d=d)
        head = DoublyHollowSwapHead(m)
        x = _state(d=d)
        t = torch.rand(1)
        G = head(x, t)
        worst = 0.0
        for i, j in _active_pairs(x):
            y = swap2(x, i, j)
            Gy = head(y, t)
            worst = max(worst, (G[0, i, j] + Gy[0, i, j]).abs().item())
        assert worst < ATOL, f"d={d}: antisymmetry residual {worst:.2e}"


@torch.no_grad()
def test_doubly_hollow_trivial_swap_vanishes():
    """Assertion 3: same-spin pair => G_swap == 0 exactly."""
    m = _backbone(d=9)
    head = DoublyHollowSwapHead(m)
    x = _state(d=9)
    t = torch.rand(1)
    G = head(x, t)
    same = [(i, j) for i in range(9) for j in range(i + 1, 9) if x[0, i] == x[0, j]]
    assert same, "fixture must contain at least one same-spin pair"
    worst = max(G[0, i, j].abs().item() for (i, j) in same)
    assert worst < ATOL, f"trivial-swap nonzero: {worst:.2e}"


@torch.no_grad()
def test_doubly_hollow_finite():
    """Assertion 6 (part): no NaN/Inf."""
    m = _backbone(d=9)
    head = DoublyHollowSwapHead(m)
    G = head(_state(d=9), torch.rand(1))
    assert torch.isfinite(G).all()


@torch.no_grad()
def test_mask_one_antisymmetric():
    """Assertion 5: the real climax head is state-swap antisymmetric, bit-exact."""
    for d in (9, 16):
        m = _backbone(d=d)
        head = LeTFMaskOneSwapHead(m)
        x = _state(d=d)
        t = torch.rand(1)
        G = head(x, t)
        worst = 0.0
        for i, j in _active_pairs(x):
            y = swap2(x, i, j)
            Gy = head(y, t)
            worst = max(worst, (G[0, i, j] + Gy[0, i, j]).abs().item())
        assert worst < ATOL, f"d={d}: mask-one antisymmetry residual {worst:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("use_sdpa", [False, True])
def test_mask_one_blind_to_anchor(use_sdpa):
    """Assertion 5 (part): G_swap(i,j) is invariant to flipping x_i then fixing omega.

    Flip x_i AND keep the readout's omega_{x_i} fixed by comparing the body
    contribution only: assert the masked body H[:,j,:] (anchor i) is unchanged
    when x_i flips (structural blindness, not incidental).
    """
    m = _backbone(d=9, use_sdpa=use_sdpa)
    x = _state(d=9)
    t = torch.rand(1)
    i, j = _active_pairs(x)[0]
    base = _masked_body(m, x, t, (i,))[:, j, :].clone()
    x_fi = x.clone()
    x_fi[0, i] *= -1
    drift = (_masked_body(m, x_fi, t, (i,))[:, j, :] - base).abs().max().item()
    assert drift < ATOL, f"mask-one body not structurally blind to x_i: {drift:.2e}"


@torch.no_grad()
def test_mask_one_label_asymmetry_pinned():
    """Assertion 6: the mask-one head is NOT label-symmetric (H_ij != H_ji).

    This is expected and documents why the downstream residual must order swap
    pairs by site index (i<j), not by spin. State-swap antisymmetry (above) is
    unaffected; here we PIN the label asymmetry so a future 'fix' that makes it
    symmetric is caught and reconsidered.
    """
    m = _backbone(d=9)
    head = LeTFMaskOneSwapHead(m)
    x = _state(d=9)
    t = torch.rand(1)
    i, j = _active_pairs(x)[0]
    G = head(x, t)
    label_resid = (G[0, i, j] + G[0, j, i]).abs().item()
    assert label_resid > 1e-4, (
        f"mask-one head unexpectedly label-symmetric ({label_resid:.2e}); "
        "the i<j ordering convention assumption needs revisiting"
    )
    assert torch.isfinite(G).all()


def _naive_factoring(model, x, t):
    """Negative control: G(x_j,i|x) + G(x_i,j|x) from the real single-site readout.

    Provably breaks paired-swap antisymmetry. Built as a raw
    (B, d, d) callable for both the negative control and the antisymmetrise
    oracle. B=1 assumed (test fixture).
    """
    G = model(x, t)  # (B, d, S)
    x_idx = ((x + 1) / 2).long()
    batch, d = x.shape
    out = x.new_zeros(batch, d, d)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            out[:, i, j] = G[0, i, x_idx[0, j]] + G[0, j, x_idx[0, i]]
    return out


@torch.no_grad()
def test_naive_factoring_breaks_antisymmetry():
    """Assertion 2: the naive factoring is NOT antisymmetric.

    Negative control: max-over-pairs floor + separation from the bit-exact head.
    """
    m = _backbone(d=9)
    x = _state(d=9)
    t = torch.rand(1)

    base = _naive_factoring(m, x, t)
    naive_worst = 0.0
    for i, j in _active_pairs(x):
        sw = _naive_factoring(m, swap2(x, i, j), t)
        naive_worst = max(naive_worst, (base[0, i, j] + sw[0, i, j]).abs().item())

    head = DoublyHollowSwapHead(m)
    G = head(x, t)
    hollow_worst = 0.0
    for i, j in _active_pairs(x):
        Gy = head(swap2(x, i, j), t)
        hollow_worst = max(hollow_worst, (G[0, i, j] + Gy[0, i, j]).abs().item())

    # Explicit floor (robust; observed naive max ~1e-3 at seed 42) ...
    assert naive_worst > 1e-4, f"negative control too weak: {naive_worst:.2e}"
    # ... and clean separation from the (bit-exact) doubly-hollow head.
    assert hollow_worst < ATOL
    assert naive_worst > 100 * max(hollow_worst, 1e-12)


@torch.no_grad()
def test_antisymmetrise_fixes_arbitrary_head():
    """Assertion 4a: antisymmetrise(non-antisymmetric raw head) is antisymmetric."""
    m = _backbone(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    A = antisymmetrise(lambda xx, tt: _naive_factoring(m, xx, tt), x, t)
    worst = 0.0
    for i, j in _active_pairs(x):
        Ay = antisymmetrise(
            lambda xx, tt: _naive_factoring(m, xx, tt), swap2(x, i, j), t
        )
        worst = max(worst, (A[0, i, j] + Ay[0, i, j]).abs().item())
    assert worst < ATOL, f"antisymmetrise did not enforce antisymmetry: {worst:.2e}"


@torch.no_grad()
def test_brute_force_matches_mask_one():
    """Assertion 4b: brute-force mask-both and leTF mask-one agree (read at j)."""
    m = _backbone(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    G_bf = DoublyHollowSwapHead(m)(x, t)
    G_m1 = LeTFMaskOneSwapHead(m)(x, t)
    diff = 0.0
    for i, j in _active_pairs(x):
        diff = max(diff, (G_bf[0, i, j] - G_m1[0, i, j]).abs().item())
    assert diff < ATOL, f"brute-force vs mask-one disagree: {diff:.2e}"


@torch.no_grad()
def test_brute_force_matches_mask_one_d16_batch_all_pairs():
    """Assertion 4b, extended: d=16 (gate dim), batch>1, ALL i<j pairs.

    test_brute_force_matches_mask_one above only pins d=9, batch 1, active
    (opposite-spin) pairs. Before mask_one is used as an O(d) drop-in for the
    O(d^2) reference head in the d=16 gate (task 9), the oracle must also cover
    the gate's own dimension, more than one state at once, and same-spin pairs
    (where both heads should agree trivially at exactly 0).
    """
    d = 16
    m = _backbone(d=d)
    x = _state_batch(d=d, batch=3)
    t = torch.rand(3)
    G_bf = DoublyHollowSwapHead(m)(x, t)
    G_m1 = LeTFMaskOneSwapHead(m)(x, t)

    pairs = _all_pairs(d)
    idx_i = torch.tensor([i for i, _ in pairs])
    idx_j = torch.tensor([j for _, j in pairs])
    bf_vals = G_bf[:, idx_i, idx_j]  # (batch, n_pairs)
    m1_vals = G_m1[:, idx_i, idx_j]

    # masking j is a float32 no-op (module docstring), so this is observed
    # bit-exact; fall back to atol if a future backbone change makes the
    # paths diverge slightly.
    if not torch.equal(bf_vals, m1_vals):
        diff = (bf_vals - m1_vals).abs().max().item()
        assert diff < ATOL, f"d=16 batch all-pairs oracle: max diff {diff:.2e}"


@torch.no_grad()
def test_masked_body_matches_real_readout_path():
    """Drift guard: with no masking, _masked_body equals the model's real readout body.

    Pins the hand-rolled replication of compute_body + the output_norm/time line to
    LeTFRateMatrix, so a future change to letf.py's readout path fails loudly here
    instead of silently desyncing both swap heads (which share _masked_body).
    """
    m = _backbone(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    expected = m.output_norm(m.compute_body(x, t)) + m.time_embedder(t).unsqueeze(1)
    got = _masked_body(m, x, t, ())
    assert (got - expected).abs().max().item() < ATOL
