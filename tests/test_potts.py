"""Potts target: S=2 reduction to Ising, closed-form swap ratio vs the generic
oracle, multiset-manifold base, and the claim that the swap sampler is
S-agnostic.

`test_s2_potts_swap_ratio_matches_ising` pins the factor-2 convention and the
energy form against a module already validated to the paper, on the quantity
that drives sampling. `test_closed_form_swap_ratio_matches_generic_oracle`
checks the O(B·d·S) closed form against the materialise-and-re-evaluate path
for S >= 3, where no Ising cross-check exists.
"""

import math

import pytest
import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)
from discrete_flow_sampler.targets.potts import (
    FixedCompositionPottsTarget,
    PottsTarget,
)


def _random_slice_states(target, n, seed=0):
    torch.manual_seed(seed)
    return target.sample_base(n, "cpu")


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def test_index_roundtrip_covers_all_species():
    """x = 2·label − 1 must round-trip for every label, not just {0,1}.

    This is the property that lets every head keep its inline
    `((x + 1) / 2).long()` unchanged under S > 2.
    """
    target = PottsTarget(D=2, sigma=0.3, n_states=4)
    labels = torch.arange(4).repeat(2, 1)  # (2, 4)
    spins = target.from_index(labels)

    assert torch.equal(spins, torch.tensor([[-1.0, 1.0, 3.0, 5.0]] * 2))
    assert torch.equal(target.to_index(spins), labels)


def test_head_index_idiom_agrees_with_to_index():
    """The inline idiom the heads use must equal the target's own map; if
    they diverge the heads silently embed the wrong species."""
    target = PottsTarget(D=2, sigma=0.3, n_states=3)
    spins = target.sample_base(8, "cpu")

    assert torch.equal(((spins + 1) / 2).long(), target.to_index(spins))


def test_composition_fraction_rejected_for_potts():
    """The binary scalar composition must fail loudly, not return nonsense."""
    target = PottsTarget(D=2, sigma=0.3, n_states=3)
    with pytest.raises(NotImplementedError, match="composition_counts"):
        target.composition_fraction(target.sample_base(2, "cpu"))


# --------------------------------------------------------------------------
# Energy: the S=2 reduction to Ising
# --------------------------------------------------------------------------


def test_s2_potts_energy_matches_ising_up_to_constant():
    """S=2 Potts at 2σ = Ising at σ + C, with C independent of the state.

    δ(s_i,s_j) = (1 + x_i x_j)/2  ⇒  2σ·Σ A δ = σ·Σ A x x + σ·Σ A.
    Only the difference across states is physical, so assert the offset is
    constant rather than zero — and separately that it equals the predicted
    σ·Σ_ij A_ij.
    """
    sigma = 0.35
    ising = FixedCompositionIsingTarget(D=3, sigma=sigma, target_composition=1 / 3)
    potts = PottsTarget(D=3, sigma=2 * sigma, n_states=2)

    states = _random_slice_states(ising, 16, seed=1)
    offset = potts.base_log_prob(states) - ising.base_log_prob(states)

    assert torch.allclose(offset, offset[0].expand_as(offset), atol=1e-5)
    assert offset[0].item() == pytest.approx(sigma * ising.A.sum().item(), rel=1e-5)


@pytest.mark.parametrize("D", [2, 3, 8])
def test_potts_reuses_the_ising_adjacency_and_its_edge_double_counting(D):
    """Potts must share IsingTarget's A exactly, double counting included.

    Both sums run over ordered (i, j), so each of the torus's 2d undirected
    edges contributes twice: A.sum() == 4d. A divergence would surface as a
    silent coupling rescale in the S=2 correspondence rather than as a failure.

    D=2 is included deliberately: on a 2-cycle a site's left and right
    neighbours coincide, so A carries entries of 2 rather than 1 while the
    total still equals 4d. Ising has the same degeneracy — it is a property of
    the L=2 torus, not of this class.
    """
    ising = IsingTarget(D=D, sigma=1.0)
    potts = PottsTarget(D=D, sigma=1.0, n_states=3)

    assert torch.equal(potts.A, ising.A)
    assert potts.A.sum().item() == pytest.approx(4 * D * D)
    assert torch.equal(potts.A, potts.A.T)
    assert torch.all(potts.A.diagonal() == 0)


def test_potts_energy_is_maximised_by_alignment():
    """A fully-aligned lattice maximises Σ A δ — a sign/orientation check that
    catches an inverted coupling, which would train happily and sample the
    wrong phase."""
    potts = PottsTarget(D=3, sigma=0.4, n_states=3)
    aligned = potts.from_index(torch.zeros(1, 9, dtype=torch.long))
    mixed = potts.from_index(torch.arange(9).remainder(3).view(1, 9))

    assert potts.base_log_prob(aligned).item() > potts.base_log_prob(mixed).item()
    # Every one of the 2·(2·d) directed edges agrees when aligned.
    assert potts.base_log_prob(aligned).item() == pytest.approx(
        0.4 * potts.A.sum().item(), rel=1e-6
    )


# --------------------------------------------------------------------------
# Swap log-ratio
# --------------------------------------------------------------------------


def test_s2_potts_swap_ratio_matches_ising():
    """The sampling-relevant quantity must agree exactly at S=2.

    The additive energy constant cancels in a log-ratio, so unlike the energy
    test above this one admits no offset. If the factor-2 convention or the
    −2·A_ij double-count correction is wrong, this is what catches it.
    """
    sigma = 0.3
    ising = FixedCompositionIsingTarget(D=3, sigma=sigma, target_composition=1 / 3)
    potts = FixedCompositionPottsTarget(
        D=3, sigma=2 * sigma, composition=(2 / 3, 1 / 3)
    )
    states = _random_slice_states(ising, 8, seed=2)
    t = torch.rand(states.shape[0])
    pairs = upper_tri_pairs(9, states.device)

    assert torch.allclose(
        potts.swap_log_ratio(states, t, pairs),
        ising.swap_log_ratio(states, t, pairs),
        atol=1e-5,
    )


@pytest.mark.parametrize("n_states", [3, 4])
def test_closed_form_swap_ratio_matches_generic_oracle(n_states):
    """Closed form vs materialise-and-re-evaluate, for S with no Ising analogue.

    The generic `IsingTarget.swap_log_ratio` builds every swapped neighbour and
    calls `log_p_tilde_t` — encoding-agnostic, so it is a valid oracle for any
    S. It is also O(B·P·d) memory, which is why the closed form exists.
    """
    composition = tuple([1 / n_states] * n_states)
    potts = FixedCompositionPottsTarget(D=n_states, sigma=0.3, composition=composition)
    states = _random_slice_states(potts, 6, seed=3)
    t = torch.rand(states.shape[0])
    pairs = upper_tri_pairs(potts.d, states.device)

    closed_form = potts.swap_log_ratio(states, t, pairs)
    oracle = super(FixedCompositionPottsTarget, potts).swap_log_ratio(states, t, pairs)

    assert torch.allclose(closed_form, oracle, atol=1e-4)


def test_same_label_swaps_have_zero_log_ratio():
    """Swapping identical labels is the identity move, so the ratio is exactly 0.

    Not approximately: these columns feed `exp()` in the ξ_t inflow term, and a
    spurious nonzero here would inject a fake rate on a move that does nothing.
    """
    potts = FixedCompositionPottsTarget(D=2, sigma=0.5, composition=(0.5, 0.5))
    # Sites 0,1 share a label and sites 2,3 share the other.
    states = potts.from_index(torch.tensor([[0, 0, 1, 1]]))
    pairs = upper_tri_pairs(4, states.device)
    ratio = potts.swap_log_ratio(states, torch.tensor([0.7]), pairs)

    labels = potts.to_index(states)
    same = labels[0, pairs[:, 0]] == labels[0, pairs[:, 1]]
    assert torch.all(ratio[0, same] == 0.0)
    assert torch.any(ratio[0, ~same] != 0.0)


# --------------------------------------------------------------------------
# Fixed-composition base and manifold
# --------------------------------------------------------------------------


def test_base_lands_on_multiset_manifold():
    """Every drawn state must carry exactly the target species counts."""
    potts = FixedCompositionPottsTarget(
        D=3, sigma=0.3, composition=(1 / 3, 1 / 3, 1 / 3)
    )
    states = _random_slice_states(potts, 64, seed=4)

    counts = potts.composition_counts(states)
    assert torch.equal(counts, torch.tensor([[3, 3, 3]]).expand_as(counts))
    potts.assert_on_manifold(states)


def test_base_log_eta_is_the_multinomial_slice_size():
    """−log|C| with |C| = d!/∏N_a! — a multinomial, not Ising's binomial.

    Getting this wrong shifts log Z by a constant, which is invisible in
    sampling and silently wrong in any reported free energy.
    """
    potts = FixedCompositionPottsTarget(D=2, sigma=0.3, composition=(0.5, 0.25, 0.25))
    states = _random_slice_states(potts, 4, seed=5)

    slice_size = math.factorial(4) / (math.factorial(2) * math.factorial(1) ** 2)
    expected = -math.log(slice_size)
    assert torch.allclose(
        potts.base_log_eta(states), torch.full((4,), expected), atol=1e-6
    )


def test_base_covers_more_than_one_slice_state():
    """The base must be uniform over C, not a fixed arrangement: a permutation
    bug returning the sorted multiset every time would still pass the manifold
    test above."""
    potts = FixedCompositionPottsTarget(D=2, sigma=0.3, composition=(0.5, 0.5))
    states = _random_slice_states(potts, 64, seed=6)

    assert torch.unique(states, dim=0).shape[0] > 1


def test_non_integral_composition_rejected():
    """No exact slice exists unless every c_a·d is an integer."""
    with pytest.raises(ValueError, match="not integral"):
        FixedCompositionPottsTarget(D=3, sigma=0.3, composition=(0.5, 0.5))


def test_composition_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        FixedCompositionPottsTarget(D=2, sigma=0.3, composition=(0.5, 0.25))


def test_off_manifold_states_rejected():
    potts = FixedCompositionPottsTarget(D=2, sigma=0.3, composition=(0.5, 0.5))
    wrong = potts.from_index(torch.zeros(1, 4, dtype=torch.long))
    with pytest.raises(AssertionError):
        potts.assert_on_manifold(wrong)


# --------------------------------------------------------------------------
# The premise: the swap stack is S-agnostic
# --------------------------------------------------------------------------


@torch.no_grad()
def test_swap_sampler_runs_unchanged_on_potts():
    """`sample_swap_ctmc` + a stock head must run on S=3 with no code changes.

    The move set only permutes positions, and the readout indexes an
    `nn.Embedding(vocab_size, ·)`. Composition is conserved exactly — a swap
    cannot change a multiset — so the hard constraint holds for S species for
    free.
    """
    torch.manual_seed(7)
    potts = FixedCompositionPottsTarget(D=2, sigma=0.3, composition=(0.5, 0.25, 0.25))
    backbone = LeTFRateMatrix(
        d=potts.d, vocab_size=potts.n_states, hidden_dim=16, n_layers=2, n_heads=2
    )
    head = DoublyHollowSwapHead(backbone)

    x0 = potts.sample_base(8, "cpu")
    ts = torch.linspace(0, 1, 6)
    final, log_weights = sample_swap_ctmc(
        head, x0, ts, return_log_weights=True, target=potts
    )

    potts.assert_on_manifold(final)
    assert log_weights.shape == (8,)
    assert torch.isfinite(log_weights).all()
