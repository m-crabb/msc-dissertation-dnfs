"""Bit-exact regression tests for the Tier-1 performance refactors.

The reference implementations below are VERBATIM copies of the pre-refactor
code (swap_ctmc.py @ commit 7259301). The refactored library functions must
match them exactly — torch.equal on CPU float32 — under identical RNG seeds.
That is a fair test because the refactors add/remove/reorder no RNG
consumption (`multinomial` / `bernoulli` / `rand` calls are untouched); only
deterministic tensor arithmetic is restructured.

Second-return contract note: the step functions' second return changed from
relu'd forward rates to raw gathered pair scores (so the eval loop can reuse
one head call for the IS integrand). Assertions compare `F.relu(second)` on
both sides — relu is idempotent, so the same assertion pins both the old and
the new contract.
"""
import torch
import torch.nn.functional as F

from discrete_flow_sampler.constraints.swap_readout import (
    LeTFMaskOneSwapHead,
    swap2,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.swap_ctmc import (
    _apply_swaps,
    _euler_step_swap,
    _euler_step_swap_matching,
    _vertex_disjoint_matching,
    compute_xi_t_swap,
    sample_swap_ctmc,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

# --------------------------------------------------------------------------
# Reference implementations (verbatim pre-refactor copies)
# --------------------------------------------------------------------------


def _reference_euler_step_swap(head, state, t_per_batch, step_dt):
    batch_size, d = state.shape
    pairs = upper_tri_pairs(d, state.device)
    n_pairs = pairs.shape[0]
    forward_rates = F.relu(gather_pair_scores(head(state, t_per_batch), pairs))
    step_probs = (forward_rates * step_dt).clamp(0.0, 1.0)
    stay_prob = (1.0 - step_probs.sum(dim=-1)).clamp(0.0, 1.0)
    categorical = torch.cat([step_probs, stay_prob[:, None]], dim=-1)
    choice = torch.multinomial(categorical, num_samples=1).squeeze(-1)

    new_state = state.clone()
    fired = choice < n_pairs
    if fired.any():
        rows = torch.nonzero(fired, as_tuple=False).squeeze(-1)
        chosen = pairs[choice[rows]]
        site_i, site_j = chosen[:, 0], chosen[:, 1]
        spin_i = new_state[rows, site_i].clone()
        new_state[rows, site_i] = new_state[rows, site_j]
        new_state[rows, site_j] = spin_i
    return new_state, forward_rates


def _reference_apply_swaps(state, accepted, pairs):
    batch_size, d = state.shape
    perm = torch.arange(d, device=state.device).expand(batch_size, d).clone()
    rows, cols = accepted.nonzero(as_tuple=True)
    site_i, site_j = pairs[cols, 0], pairs[cols, 1]
    perm[rows, site_i] = site_j
    perm[rows, site_j] = site_i
    return state.gather(1, perm)


def _reference_euler_step_swap_matching(head, state, t_per_batch, step_dt):
    batch_size, d = state.shape
    pairs = upper_tri_pairs(d, state.device)
    forward_rates = F.relu(gather_pair_scores(head(state, t_per_batch), pairs))
    fire_prob = (forward_rates * step_dt).clamp(0.0, 1.0)
    proposed = torch.bernoulli(fire_prob).bool()
    priority = torch.rand(batch_size, pairs.shape[0], device=state.device)
    accepted = _vertex_disjoint_matching(proposed, priority, pairs, d)
    return _reference_apply_swaps(state, accepted, pairs), forward_rates


def _reference_compute_xi_t_swap(x, t, head, target):
    from discrete_flow_sampler.samplers._swap_neighbours import (
        SWAP_LOG_RATIO_CLAMP,
    )

    pairs = upper_tri_pairs(x.shape[1], x.device)
    G_edge = gather_pair_scores(head(x, t), pairs)
    G_plus = F.relu(G_edge)
    neg_G_plus = F.relu(-G_edge)
    log_ratio = target.swap_log_ratio(x, t, pairs).clamp(max=SWAP_LOG_RATIO_CLAMP)
    outflow = G_plus.sum(dim=-1)
    inflow = (neg_G_plus * log_ratio.exp()).sum(dim=-1)
    return target.dt_log_p_tilde_t(x, t) + outflow - inflow


def _reference_sample_swap_ctmc(head, x0, ts, *, target, multi_event):
    state = x0.clone()
    batch_size, _ = state.shape
    log_weights = torch.zeros(batch_size, dtype=state.dtype, device=state.device)
    step_fn = (
        _reference_euler_step_swap_matching if multi_event
        else _reference_euler_step_swap
    )
    for step in range(len(ts) - 1):
        step_dt = ts[step + 1] - ts[step]
        t_per_batch = ts[step].expand(batch_size)
        new_state, _ = step_fn(head, state, t_per_batch, step_dt)
        xi_t = _reference_compute_xi_t_swap(state, t_per_batch, head, target)
        log_weights = log_weights + xi_t * step_dt
        state = new_state
    return state, log_weights


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class _ConstScoreHead:
    """Every ordered pair scores `value`: forces the no-fire (value<=0) and
    saturated all-fire (value*dt >= 1) corners of the step logic."""

    def __init__(self, d, value):
        self.d = d
        self.value = value

    def __call__(self, x, t):
        return torch.full(
            (x.shape[0], self.d, self.d), self.value, dtype=x.dtype
        )


def _small_head_and_target(d_side=4, seed=11):
    torch.manual_seed(seed)
    target = FixedCompositionIsingTarget(
        D=d_side, sigma=0.223, target_composition=0.5
    )
    backbone = LeTFRateMatrix(
        d=target.d, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
    )
    return LeTFMaskOneSwapHead(backbone), target


# --------------------------------------------------------------------------
# Task 2: static-tensor caching
# --------------------------------------------------------------------------


def test_upper_tri_pairs_values_and_cache_identity():
    pairs = upper_tri_pairs(64, "cpu")
    expected = torch.combinations(torch.arange(64), r=2)
    assert torch.equal(pairs, expected)
    assert upper_tri_pairs(64, "cpu") is pairs
    assert upper_tri_pairs(64, torch.device("cpu")) is pairs


def test_letf_mask_caches_not_in_state_dict_and_stable():
    torch.manual_seed(0)
    model = LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    x = (torch.randint(0, 2, (3, 16)) * 2 - 1).float()
    t = torch.rand(3)
    keys_before = sorted(model.state_dict())
    with torch.no_grad():
        first = model(x, t)
        second = model(x, t)
    assert torch.equal(first, second)
    assert sorted(model.state_dict()) == keys_before


# --------------------------------------------------------------------------
# Tasks 3 + 4: Euler steps, swap application, sampler log-weights
# --------------------------------------------------------------------------


@torch.no_grad()
def test_one_event_step_bit_exact_vs_reference():
    _, target = _small_head_and_target()
    real_head, _ = _small_head_and_target()
    heads = {
        "real": real_head,
        "no_fire": _ConstScoreHead(target.d, 0.0),
        "all_fire": _ConstScoreHead(target.d, 50.0),
    }
    torch.manual_seed(3)
    x = target.sample_base(16, device="cpu")
    t = torch.full((16,), 0.5)
    step_dt = torch.tensor(0.1)
    for label, head in heads.items():
        torch.manual_seed(21)
        want_state, want_second = _reference_euler_step_swap(head, x, t, step_dt)
        torch.manual_seed(21)
        got_state, got_second = _euler_step_swap(head, x, t, step_dt)
        assert torch.equal(got_state, want_state), label
        assert torch.equal(F.relu(got_second), F.relu(want_second)), label


@torch.no_grad()
def test_matching_step_bit_exact_vs_reference():
    _, target = _small_head_and_target()
    real_head, _ = _small_head_and_target()
    heads = {
        "real": real_head,
        "dense_conflicts": _ConstScoreHead(target.d, 5.0),
        "no_fire": _ConstScoreHead(target.d, 0.0),
    }
    torch.manual_seed(3)
    x = target.sample_base(16, device="cpu")
    t = torch.full((16,), 0.5)
    step_dt = torch.tensor(0.5)
    for label, head in heads.items():
        torch.manual_seed(22)
        want_state, want_second = _reference_euler_step_swap_matching(
            head, x, t, step_dt
        )
        torch.manual_seed(22)
        got_state, got_second = _euler_step_swap_matching(head, x, t, step_dt)
        assert torch.equal(got_state, want_state), label
        assert torch.equal(F.relu(got_second), F.relu(want_second)), label


@torch.no_grad()
def test_apply_swaps_bit_exact_vs_reference():
    d = 16
    pairs = upper_tri_pairs(d, "cpu")
    torch.manual_seed(5)
    state = (torch.randint(0, 2, (32, d)) * 2 - 1).float()
    proposed = torch.rand(32, pairs.shape[0]) < 0.3  # dense -> real matchings
    priority = torch.rand(32, pairs.shape[0])
    accepted = _vertex_disjoint_matching(proposed, priority, pairs, d)
    accepted[0] = False  # exercise the no-swap row
    assert torch.equal(
        _apply_swaps(state, accepted, pairs),
        _reference_apply_swaps(state, accepted, pairs),
    )


@torch.no_grad()
def test_sampler_log_weights_bit_exact_vs_reference():
    head, target = _small_head_and_target()
    ts = torch.linspace(0.0, 1.0, 12)
    for multi_event in (False, True):
        torch.manual_seed(3)
        x0 = target.sample_base(8, device="cpu")
        torch.manual_seed(4)
        want_x, want_w = _reference_sample_swap_ctmc(
            head, x0, ts, target=target, multi_event=multi_event
        )
        torch.manual_seed(4)
        got_x, got_w = sample_swap_ctmc(
            head, x0, ts, return_log_weights=True, target=target,
            multi_event=multi_event,
        )
        assert torch.equal(got_x, want_x), f"multi_event={multi_event}"
        assert torch.equal(got_w, want_w), f"multi_event={multi_event}"


@torch.no_grad()
def test_compute_xi_t_swap_unchanged_behaviour():
    head, target = _small_head_and_target()
    x = target.sample_base(8, device="cpu")
    t = torch.full((8,), 0.3)
    assert torch.equal(
        compute_xi_t_swap(x, t, head, target),
        _reference_compute_xi_t_swap(x, t, head, target),
    )


# --------------------------------------------------------------------------
# Task 7: invariants under bf16 autocast (opt-in eval path)
# --------------------------------------------------------------------------


@torch.no_grad()
def test_antisymmetry_exact_under_bf16_autocast():
    """Swap-antisymmetry must hold EXACTLY under autocast: both calls feed
    bit-identical inputs to the masked body, and the readout dot product just
    negates one operand -- IEEE negation is exact at any precision."""
    head, target = _small_head_and_target()
    torch.manual_seed(9)
    x = target.sample_base(6, device="cpu")
    t = torch.full((6,), 0.5)
    pairs = upper_tri_pairs(target.d, "cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        G = head(x, t)
        for i, j in pairs.tolist()[:20]:
            G_swapped = head(swap2(x, i, j), t)
            assert torch.equal(G[:, i, j].float(), -G_swapped[:, i, j].float())


@torch.no_grad()
def test_sampler_on_manifold_and_fp32_log_weights_under_bf16_autocast():
    """Composition preservation is dtype-independent (swaps are index
    permutations), and the IS log-weight accumulator must stay fp32 so the
    128-step running sum does not lose precision."""
    head, target = _small_head_and_target()
    torch.manual_seed(6)
    x0 = target.sample_base(8, device="cpu")
    ts = torch.linspace(0.0, 1.0, 12)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        x_final, log_weights = sample_swap_ctmc(
            head, x0, ts, return_log_weights=True, target=target
        )
    target.assert_on_manifold(x_final)
    assert log_weights.dtype == torch.float32


@torch.no_grad()
def test_head_scores_match_fp32_within_bf16_tolerance():
    """Value-level (not structural) agreement: bf16 has an 8-bit mantissa,
    so per-element relative error ~2^-8. Bound calibrated on the seed-11
    init-scale head (observed max abs diff 1.7e-4 at max |G| 1.9e-2)."""
    head, target = _small_head_and_target()
    torch.manual_seed(9)
    x = target.sample_base(6, device="cpu")
    t = torch.full((6,), 0.5)
    G_fp32 = head(x, t)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        G_bf16 = head(x, t)
    assert torch.allclose(G_bf16.float(), G_fp32, atol=1e-3, rtol=2e-2)


# --------------------------------------------------------------------------
# Task 8: SDPA readout (opt-in flag, default OFF)
# --------------------------------------------------------------------------


def _letf_pair(d: int, seed: int = 13) -> tuple[LeTFRateMatrix, LeTFRateMatrix]:
    """Same-seed model pair differing ONLY in the SDPA readout flag (the flag
    consumes no RNG, so the parameters are identical)."""
    models = []
    for use_sdpa in (False, True):
        torch.manual_seed(seed)
        model = LeTFRateMatrix(
            d=d, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2,
            use_sdpa_readout=use_sdpa,
        )
        model.eval()
        models.append(model)
    return models[0], models[1]


@torch.no_grad()
def test_sdpa_readout_matches_manual_and_keeps_state_dict():
    """SDPA computes the same masked softmax-attention as the manual
    scores/masked_fill/softmax/matmul block without materialising the
    (B, n_heads, d, 2d) score buffer. Only the kernel's reduction order
    differs, so G agrees to fp32 tolerance; no new parameters or buffers,
    so checkpoints are interchangeable across the flag."""
    for d in (16, 64):
        manual, sdpa = _letf_pair(d)
        assert sorted(manual.state_dict()) == sorted(sdpa.state_dict())
        torch.manual_seed(2)
        x = (torch.randint(0, 2, (4, d)) * 2 - 1).float()
        t = torch.rand(4)
        assert torch.allclose(sdpa(x, t), manual(x, t), atol=1e-5), f"d={d}"


@torch.no_grad()
def test_sdpa_flag_reaches_swap_head():
    """The mask-one swap head routes through AttentionReadout via
    _masked_body, so the flag must reach that path (its d-anchor-stacked
    score buffer is the d=256 memory wall) while preserving G_swap."""
    manual, sdpa = _letf_pair(16)
    assert sdpa.attention_readout.use_sdpa
    torch.manual_seed(3)
    x = (torch.randint(0, 2, (4, 16)) * 2 - 1).float()
    t = torch.rand(4)
    G_manual = LeTFMaskOneSwapHead(manual)(x, t)
    G_sdpa = LeTFMaskOneSwapHead(sdpa)(x, t)
    assert torch.allclose(G_sdpa, G_manual, atol=1e-5)


def test_model_cfg_sdpa_default_off():
    from experiments.dnfs_baseline_01.configs import ModelCfg

    assert ModelCfg().use_sdpa_readout is False


# --------------------------------------------------------------------------
# Task 9: torch.compile on the head (opt-in flag, default OFF)
# --------------------------------------------------------------------------


@torch.no_grad()
def test_compile_head_flag_matches_uncompiled_and_keeps_state_dict():
    """`compile_head=True` must use IN-PLACE nn.Module.compile: identical
    state_dict keys (no `_orig_mod.` prefix, so checkpoints round-trip) and
    inductor-vs-eager output agreement at fp32 tolerance. The cfg's d need
    not match the backbone (build_swap_head only reads head_kind /
    anchor_chunk_size / compile_head), so a small backbone keeps the test's
    runtime cost at one compile."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import (
        CONFIGS,
        HardStageCfg,
        build_swap_head,
    )

    assert HardStageCfg.__dataclass_fields__["compile_head"].default is False

    base_cfg = CONFIGS["H2_d64_c50_s223_letf_mo"]
    assert base_cfg.compile_head is False
    torch.manual_seed(0)
    backbone = LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    backbone.eval()

    plain_head = build_swap_head(base_cfg, backbone)
    keys_before = sorted(plain_head.state_dict())
    torch.manual_seed(1)
    x = (torch.randint(0, 2, (2, 16)) * 2 - 1).float()
    t = torch.rand(2)
    want = plain_head(x, t)

    compiled_head = build_swap_head(replace(base_cfg, compile_head=True), backbone)
    assert sorted(compiled_head.state_dict()) == keys_before
    got = compiled_head(x, t)
    assert torch.allclose(got, want, atol=1e-5)
