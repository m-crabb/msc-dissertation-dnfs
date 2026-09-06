"""Falsification tests for the triu-pair gather lever (opt-in, default OFF).

Written BEFORE the lever: these encode what correct looks like independently
of how the gather is implemented.

What the lever is. Every one-pass swap head assembles a pair context H that
is label-SYMMETRIC (H_ji := H_ij, the convention that makes index
antisymmetry G[j,i] = -G[i,j] an identity), and then runs its per-pair
nonlinear work -- the pair-readout MLP, the masked-attention band, the
hole-subtracted global MLP -- on all d^2 ordered pairs before mirroring. Half
of that work is the mirror image of the other half, and the diagonal is dead
(forward reads H against omega_{x_i} - omega_{x_j}, identically zero at
i = j). The lever gathers the d(d-1)/2 pairs with i < j BEFORE the nonlinear
work and scatters the result back, so the (B, d^2, F) activation slabs -- the
heads' memory footprint at d = 256 -- become (B, d(d-1)/2, F).

The correctness claim is therefore a pure re-indexing claim, and it is tested
as one:

  * G must match the dense path to fp round-off (the per-pair maps are
    row-independent, so only GEMM shape changes);
  * H must match on i != j. The DIAGONAL is deliberately not reproduced: it
    never reaches G, and recomputing it would reinstate d of the rows the
    lever exists to drop;
  * index antisymmetry and the zero diagonal of G stay EXACT, not merely
    close -- they are identities of the mirror, and a scatter that wrote the
    two triangles from different tensors would break them;
  * the flag adds no parameter, no persistent buffer and no RNG draw, so a
    flag-off head is byte-identical to every checkpoint on disk (that is what
    "default OFF" buys, and archived-run reproducibility depends on it);
  * blindness is untouched -- pinned in each head's own suite, where the
    blindness probes are parametrised over the flag.

Heads NOT covered here, and why (audited 2026-08-26):
  * two_hole_patch: H_ij = P_ij - P_ji needs BOTH orderings of every pair
    (z_ij routes the two holes through different linear maps), so its per-pair
    MLP is genuinely d^2-shaped; only the diagonal is waste.
  * mask_one / grouped_anchor: H_ij = H^{group(i)}[:, j, :] is not symmetric,
    and the per-pair work is a mul+sum readout, not an MLP -- the cost there
    is the masked body passes, which are per-SITE.
  * the exact-field channel: sigma * Delta_ij is symmetric in (i, j), but it
    is elementwise arithmetic at width 1, not a per-pair map -- gathering it
    would trade three (B, d, d) temporaries for three (B, P) ones and a
    scatter, which is not where the memory is.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.exact_field_channel import (
    ExactFieldSwapHead,
)
from discrete_flow_sampler.constraints.factorised_swap_head import (
    FactorisedSwapHead,
)
from discrete_flow_sampler.constraints.interval_swap_head import (
    IntervalSwapHead,
    triu_pair_indices,
)
from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.targets.ising import (
    SIGMA_C,
    FixedCompositionIsingTarget,
)

# Re-indexing bar: the per-pair maps are row-independent, so the only source
# of drift is GEMM/reduction shape (a (B, d^2, F) matmul vs a (B, P, F) one).
# That is fp round-off on quantities of order 1, not a numerical method
# change, so the bar is two orders tighter than the suite's blindness ATOL.
GATHER_ATOL = 1e-6

LATTICE_SIDE = 4
D = LATTICE_SIDE * LATTICE_SIDE


def _backbone(d=D, seed=42):
    torch.manual_seed(seed)
    return LeTFRateMatrix(
        d=d,
        vocab_size=2,
        hidden_dim=8,
        n_layers=2,
        n_heads=2,
        use_sdpa_readout=False,
    )


def _interval(gather, **kw):
    return IntervalSwapHead(
        _backbone(),
        pair_offsets=(1, LATTICE_SIDE),
        band_feature_dim=6,
        gather_triu_pairs=gather,
        **kw,
    )


def _masked_attention(gather, **kw):
    return MaskedAttentionSwapHead(
        _backbone(),
        pair_offsets=(1, LATTICE_SIDE),
        band_feature_dim=6,
        attention_dim=6,
        lattice_side=LATTICE_SIDE,
        gather_triu_pairs=gather,
        **kw,
    )


def _factorised(gather, **kw):
    return FactorisedSwapHead(
        _backbone(),
        bilinear_rank=3,
        factor_dim=4,
        global_feature_dim=6,
        position_dim=5,
        band_feature_dim=6,
        attention_dim=6,
        pair_offsets=(1, LATTICE_SIDE),
        lattice_side=LATTICE_SIDE,
        gather_triu_pairs=gather,
        **kw,
    )


# One case per code path the lever touches: both exterior combiners of the
# shared interval/MA readout, the MA band with and without the stencil family
# (an extra family = an extra attention block), and the factorised global term
# alone, with each interior band, and under multi-order streams (whose
# bilinear einsum stays dense -- the gather is inside the global term only).
HEAD_CASES = {
    "interval": lambda g: _interval(g),
    "interval_bilinear": lambda g: _interval(g, exterior_combiner="bilinear"),
    "masked_attention": lambda g: _masked_attention(g),
    "masked_attention_bilinear": lambda g: _masked_attention(
        g, exterior_combiner="bilinear"
    ),
    "masked_attention_stencil": lambda g: _masked_attention(g, use_stencil=True),
    "factorised": lambda g: _factorised(g),
    "factorised_global_only": lambda g: _factorised(g, use_bilinear=False),
    "factorised_prefix_band": lambda g: _factorised(g, interior_band="prefix"),
    "factorised_attention_band": lambda g: _factorised(g, interior_band="attention"),
    "factorised_multi_order": lambda g: _factorised(
        g,
        interior_band="attention",
        site_orderings=("row", "col"),
    ),
}


def _pair(case, gather_on=True):
    """The same head built twice, flag off and flag on. Same seed and no RNG
    draw from the flag => identical parameters, so any output difference is
    the lever and nothing else."""
    dense = HEAD_CASES[case](False).eval()
    gathered = HEAD_CASES[case](gather_on).eval()
    return dense, gathered


def _state(batch=2, d=D, seed=7):
    torch.manual_seed(seed)
    spins = (torch.randint(0, 2, (batch, d)) * 2 - 1).float()
    spins[:, 0] = 1.0  # guarantee both spins present in every row
    spins[:, 1] = -1.0
    return spins


def _drift(a, b):
    return (a - b).abs().max().item()


@torch.no_grad()
@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_gathered_scores_match_dense(case):
    """The lever is a re-indexing: G is the same matrix either way."""
    dense, gathered = _pair(case)
    x, t = _state(), torch.rand(2)
    drift = _drift(gathered(x, t), dense(x, t))
    assert drift < GATHER_ATOL, f"{case}: G moved by {drift:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_gathered_pair_context_matches_dense_off_diagonal(case):
    """H agrees wherever it is READ. The diagonal is exempt by design: the
    token difference is zero there, so H_ii never reaches G, and the gathered
    path leaves it at zero rather than paying for d dead rows."""
    dense, gathered = _pair(case)
    x, t = _state(), torch.rand(2)
    off_diagonal = ~torch.eye(D, dtype=torch.bool).view(1, D, D, 1)
    H_dense = dense.compute_pair_context(x, t)
    H_gathered = gathered.compute_pair_context(x, t)
    assert H_gathered.shape == H_dense.shape
    drift = _drift(
        H_gathered.masked_select(off_diagonal), H_dense.masked_select(off_diagonal)
    )
    assert drift < GATHER_ATOL, f"{case}: H moved by {drift:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_gathered_context_is_exactly_symmetric(case):
    """The scatter must write ONE tensor into both triangles: a mirror built
    from two separately-computed halves would leave index antisymmetry
    approximate, which is the property the swap CTMC's reverse rate leans on."""
    _, gathered = _pair(case)
    H = gathered.compute_pair_context(_state(), torch.rand(2))
    assert torch.equal(H, H.transpose(1, 2))


@torch.no_grad()
@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_gathered_scores_stay_exactly_antisymmetric(case):
    """G[j,i] = -G[i,j] and G_ii = 0, at exactly 0.0, on the gathered path."""
    _, gathered = _pair(case)
    G = gathered(_state(), torch.rand(2))
    assert torch.equal(G, -G.transpose(1, 2))
    assert torch.equal(torch.diagonal(G, dim1=1, dim2=2), torch.zeros(2, D))


@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_flag_adds_no_state_and_no_rng_draw(case):
    """Default OFF is the archived-run contract: the flag is a plain bool
    attribute, so state_dict, parameter values and RNG consumption at
    construction are identical with it set."""
    dense, gathered = _pair(case)
    assert dense.state_dict().keys() == gathered.state_dict().keys()
    for key, value in dense.state_dict().items():
        assert torch.equal(value, gathered.state_dict()[key]), key


@torch.no_grad()
@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_gathered_matches_dense_under_bf16_autocast(case):
    """The hard recipe evaluates under a bf16 autocast block; the gather sits
    inside it, so the two paths must still agree there (and G must stay fp32,
    the dtype contract every head keeps through its readout)."""
    dense, gathered = _pair(case)
    x, t = _state(), torch.rand(2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        G_dense, G_gathered = dense(x, t), gathered(x, t)
    assert G_dense.dtype == G_gathered.dtype == torch.float32
    # bf16 has ~3 decimal digits, so the bar is the dtype's, not the lever's.
    assert _drift(G_gathered, G_dense) < 1e-2


@pytest.mark.parametrize("case", list(HEAD_CASES))
def test_gathered_gradients_reach_every_head_parameter(case):
    """The scatter must be differentiable and must not orphan a module: an
    index_put that dropped its input from the graph would train silently
    wrong, which no forward-value test would catch."""
    _, gathered = _pair(case)
    gathered.train()
    gathered(_state(), torch.rand(2)).sum().backward()
    for name, parameter in gathered.named_parameters():
        if "backbone" in name or "attention_readout" in name:
            continue
        assert parameter.grad is not None, f"{case}: {name} dead in graph"


@torch.no_grad()
@pytest.mark.parametrize(
    "build", [_interval, _masked_attention], ids=["interval", "masked_attention"]
)
def test_band_summaries_pair_form_matches_dense_triangle(build):
    """The band's gathered form is the lever's load-bearing new API: for the
    masked-attention head it moves the (B, d^2, n_terms) attention SCORES --
    the head's largest tensor, and the one its docstring flags as the d = 256
    price -- onto the i < j pairs. Both heads define the band only for i < j,
    so the pair form must equal the dense upper triangle exactly where it is
    defined."""
    head = build(True).eval()
    x, t = _state(), torch.rand(2)
    rows, cols = triu_pair_indices(D, x.device)
    dense = head.band_summaries(x, t)[:, rows, cols]
    gathered = head.band_summaries(x, t, (rows, cols))
    assert gathered.shape == dense.shape
    assert _drift(gathered, dense) < GATHER_ATOL


@torch.no_grad()
def test_exact_field_channel_composes_with_the_gather():
    """The fimo2ef arm is factorised + exact field; the channel wraps the head,
    so the lever must survive the wrapper untouched (the channel's own
    sigma * Delta term is left dense -- see the module docstring)."""
    target = FixedCompositionIsingTarget(
        D=LATTICE_SIDE, sigma=SIGMA_C, target_composition=0.5
    )
    dense = ExactFieldSwapHead(_factorised(False, interior_band="attention"), target)
    gathered = ExactFieldSwapHead(_factorised(True, interior_band="attention"), target)
    dense.eval(), gathered.eval()
    x, t = _state(), torch.rand(2)
    assert _drift(gathered(x, t), dense(x, t)) < GATHER_ATOL


@pytest.mark.parametrize("head_kind", ["interval", "masked_attention", "factorised"])
def test_config_flag_reaches_the_head(head_kind):
    """`cfg.gather_triu_pairs` must arrive at the head it is set for -- a
    silently-dropped memory lever would look like a null result at D=16."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    cfg = replace(
        CONFIGS["H2_d16_c50_s223_letf_fab8_10k"],
        head_kind=head_kind,
        gather_triu_pairs=True,
    )
    head = build_swap_head(cfg, _backbone())
    assert head.gather_triu_pairs is True


def test_config_flag_is_a_no_op_for_the_heads_without_a_symmetric_slab():
    """House pattern for head-specific knobs: cells that do not read the flag
    build byte-identically with it set. two_hole_patch needs BOTH orderings of
    every pair (H_ij = P_ij - P_ji), mask_one's H is not label-symmetric at
    all, so neither has a triangle to drop."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    base = CONFIGS["H2_d16_c50_s223_letf_fab8_10k"]
    for head_kind in ("two_hole_patch", "mask_one"):
        cfg = replace(base, head_kind=head_kind)
        plain = build_swap_head(cfg, _backbone())
        flagged = build_swap_head(replace(cfg, gather_triu_pairs=True), _backbone())
        x, t = _state(), torch.rand(2)
        with torch.no_grad():
            assert torch.equal(plain(x, t), flagged(x, t)), head_kind


@torch.no_grad()
def test_triu_pair_indices_enumerate_the_upper_triangle():
    """The index table itself, pinned: d(d-1)/2 pairs, strictly i < j, in the
    row-major order the sampler's own pair table uses."""
    rows, cols = triu_pair_indices(5, torch.device("cpu"))
    assert rows.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
    assert cols.tolist() == [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
