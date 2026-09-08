"""Potts target distribution: the S-species generalisation of the Ising target.

Each site carries one of S labels and the coupling rewards agreement:

    log p(s) = σ · Σ_{i,j} A_ij · δ(s_i, s_j)                          (Potts)
    log p(x) = σ · Σ_{i,j} A_ij · x_i x_j    + bias·Σ_i x_i            (Ising)

A_ij is the periodic D×D adjacency IsingTarget builds, reused as is: symmetric
with zero diagonal, so each undirected edge is counted twice in both sums
(`A.sum() == 4d`). The effective per-bond coupling is therefore 2σ, already
baked into the project's σ_c ≈ 0.223 (the 2D Ising critical coupling
ln(1+√2)/2 ≈ 0.4407, halved). D=2 is degenerate: on a 2-cycle a site's two
neighbours are the same site, so A has entries of 2 rather than 1 (Ising
identically); it is used only in convention-agnostic plumbing tests.

S=2 correspondence. For x ∈ {−1,+1}, δ(s_i, s_j) = (1 + x_i x_j) / 2, so

    σ_P · Σ A_ij δ(s_i,s_j) = (σ_P/2)·Σ A_ij x_i x_j + (σ_P/2)·Σ A_ij.

S=2 Potts at σ_P = 2σ equals Ising at σ plus an x-independent constant
(σ·Σ A_ij = σ·2·|edges|) that cancels in every log-ratio. This is the
regression test in test_potts.py, and why `sigma` here is the Potts coupling:
reproducing an Ising run at σ means passing 2σ. `IsingTarget.base_log_prob`
cannot be reused for S > 2: `x_i x_j` on labels {−1,1,3,...} is not an
indicator, and nothing would raise.

State encoding: x = 2·label − 1, label ∈ {0, …, S−1}, stored as a float
tensor like Ising's {−1,+1}. Every head recovers the index with
`((x + 1) / 2).long()` (13 call sites), which inverts this map for any S, so
the swap/head stack runs on Potts unchanged; `to_index` / `from_index` own
the convention.

Not carried over: `bias` (on the fixed-composition manifold a per-species
field depends only on the frozen species counts, so it is a constant); the
soft composition penalty (`composition_penalty` is inherited returning zeros);
`composition_fraction` (a Potts composition is an S-vector, so it raises; use
`composition_counts`).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.targets.ising import IsingTarget


class PottsTarget(IsingTarget):
    """Periodic D×D Potts lattice with the shared annealing path.

    Inherits the adjacency construction, `set_sigma`, the annealing path
    `log_p_tilde_t` / `dt_log_p_tilde_t` (Eq. 4), `log_prob` and the generic
    materialise-and-evaluate `swap_log_ratio`, all encoding-agnostic. Only
    the pieces that read `x` as a number rather than a label are overridden.

    Args:
        D: lattice side; d = D².
        sigma: Potts coupling. Note the factor-2 convention above — the Ising
            run at coupling σ corresponds to S=2 Potts at 2σ.
        n_states: number of species S (S=2 reproduces Ising; S≥3 is new).
        device: torch device for the adjacency matrix.
    """

    def __init__(
        self,
        D: int,
        sigma: float,
        n_states: int,
        device: torch.device | str = "cpu",
    ):
        if n_states < 2:
            raise ValueError(f"n_states must be at least 2, got {n_states}")
        super().__init__(
            D,
            sigma,
            bias=0.0,
            device=device,
            target_composition=None,
            composition_penalty_strength=0.0,
        )
        self.n_states = n_states

    def to_index(self, x: Tensor) -> Tensor:
        """Label indices 0…S−1 from stored spins, shape (B, d) long.

        Inverse of `from_index`; the map every head applies inline.
        """
        return ((x + 1) / 2).long()

    def from_index(self, index: Tensor) -> Tensor:
        """Stored spins x = 2·index − 1 from label indices, shape (B, d) float."""
        return (index * 2 - 1).float()

    def composition_counts(self, x: Tensor) -> Tensor:
        """Per-species site counts, shape (B, S); rows sum to d.

        The Potts analogue of `composition_fraction`, and the quantity a swap
        conserves exactly (it permutes labels).
        """
        return F.one_hot(self.to_index(x), self.n_states).sum(dim=1)

    def composition_fraction(self, x: Tensor) -> Tensor:
        """Not defined for S species — use `composition_counts`."""
        raise NotImplementedError(
            "composition_fraction is binary-only (fraction of +1 spins); a "
            "Potts composition is an S-vector — use composition_counts(x)."
        )

    def base_log_prob(self, x: Tensor) -> Tensor:
        """Unnormalised Potts log-density, shape (B,).

            log p(x) = σ · Σ_{i,j} A_ij · δ(s_i, s_j)

        Each undirected edge is counted twice (A symmetric, zero diagonal),
        matching IsingTarget so the S=2 correspondence is exact.

        With the one-hot Ω ∈ {0,1}^{B×d×S} of labels,
        Σ_{i,j} A_ij δ(s_i,s_j) = Σ_a (Ω_a)ᵀ A (Ω_a): one batched matmul
        rather than a (B, d, d) pairwise-equality tensor.
        """
        onehot = F.one_hot(self.to_index(x), self.n_states).to(x.dtype)  # (B, d, S)
        neighbour_counts = torch.matmul(self.A, onehot)  # (B, d, S)
        return self.sigma * (onehot * neighbour_counts).sum(dim=(1, 2))

    def base_log_eta(self, x: Tensor) -> Tensor:
        """Log-density of the uniform base η on {0…S−1}^d, shape (B,).

        η is uniform over all S^d states, so log η ≡ −d·log S, constant in x
        (the S-species analogue of Ising's −d·log 2 uniform base).
        """
        return torch.full(
            (x.shape[0],),
            -self.d * math.log(self.n_states),
            device=x.device,
            dtype=x.dtype,
        )

    def sample_base(self, n: int, device) -> Tensor:
        """Draw n states uniformly from {0…S−1}^d, shape (n, d) float spins."""
        return self.from_index(
            torch.randint(0, self.n_states, (n, self.d), device=device)
        )


class FixedCompositionPottsTarget(PottsTarget):
    """Potts on the fixed-composition manifold C = {n_a(s) = N_a ∀ species a}.

    A swap permutes two labels and so preserves the label multiset, which is
    this manifold (Ising's scalar n_plus generalised to an S-vector of
    counts); `samplers/swap_ctmc.py` needs no changes.

    Differs from PottsTarget only in the base: `sample_base` draws uniformly
    over the multiset slice, and `base_log_eta` is the constant −log|C| with
    |C| the multinomial coefficient d! / ∏_a N_a! (Ising: binomial C(d, N_A)).

    Args:
        composition: per-species fractions, length S, summing to 1. Each
            c_a·d must be integral or no exact slice exists (same requirement
            IsingTarget imposes on its scalar composition).
    """

    def __init__(
        self,
        D: int,
        sigma: float,
        composition: tuple[float, ...],
        device: torch.device | str = "cpu",
    ):
        super().__init__(D, sigma, n_states=len(composition), device=device)
        if abs(sum(composition) - 1.0) > 1e-9:
            raise ValueError(f"composition must sum to 1, got {sum(composition)}")
        counts = []
        for species, fraction in enumerate(composition):
            exact = fraction * self.d
            rounded = round(exact)
            if abs(exact - rounded) > 1e-9:
                raise ValueError(
                    f"composition[{species}]={fraction} * d={self.d} = {exact} "
                    "is not integral; no exact fixed-composition slice exists."
                )
            counts.append(rounded)
        self.target_counts = tuple(counts)
        # log|C| = log d! − Σ_a log N_a!  (multinomial coefficient)
        self._log_slice_size = math.lgamma(self.d + 1) - sum(
            math.lgamma(count + 1) for count in counts
        )

    def sample_base(self, n: int, device) -> Tensor:
        """Uniform over the multiset slice, shape (n, d).

        `argsort` of per-row uniforms is a uniform random permutation (as in
        IsingTarget.sample_base), so scattering the sorted multiset through
        it is uniform on C.
        """
        multiset = torch.repeat_interleave(
            torch.arange(self.n_states, device=device),
            torch.tensor(self.target_counts, device=device),
        )  # (d,) sorted labels with the target counts
        permuted_sites = torch.rand(n, self.d, device=device).argsort(dim=1)
        labels = torch.empty((n, self.d), dtype=torch.long, device=device)
        labels.scatter_(1, permuted_sites, multiset.expand(n, self.d))
        return self.from_index(labels)

    def base_log_eta(self, x: Tensor) -> Tensor:
        """Constant log-density −log|C| on the slice, shape (B,).

        Constant in x, which is why the (1−t) term of the annealing path
        drops out of `swap_log_ratio`.
        """
        return torch.full(
            (x.shape[0],), -self._log_slice_size, device=x.device, dtype=x.dtype
        )

    def assert_on_manifold(self, x: Tensor) -> None:
        """Raise AssertionError if any row's species counts differ from target."""
        counts = self.composition_counts(x)
        expected = torch.tensor(self.target_counts, device=x.device)
        off_manifold = (counts != expected).any(dim=-1)
        if off_manifold.any():
            raise AssertionError(
                f"off-manifold states: expected counts={self.target_counts}, "
                f"got e.g. {counts[off_manifold][:5].tolist()}"
            )

    def swap_log_ratio(self, x: Tensor, t: Tensor, pairs: Tensor) -> Tensor:
        """Closed-form Potts swap log-ratio on the slice, shape (B, P).

        Mirrors `FixedCompositionIsingTarget.swap_log_ratio`: the base is
        constant on C so the (1−t) term cancels, leaving t·σ·ΔE.

        Derivation. Write E(s) = Σ_k n_k(s_k) with n_k(a) = Σ_l A_kl·δ(s_l, a)
        the adjacency-weighted count of neighbours of k carrying label a.
        Swapping a = s_i and b = s_j changes only terms touching i or j. The
        {i, j} pair's own term 2·A_ij·δ(a,b) is symmetric in a,b and cancels;
        with the exclusive counts m_k(·) = n_k(·) − A_ki·δ(s_i,·) − A_kj·δ(s_j,·):

            ΔE = 2·[ m_i(b) − m_i(a) + m_j(a) − m_j(b) ]

        Re-expanding m in terms of n (using A_ii = 0, and δ(a,b) = 0 since
        a ≠ b for any non-trivial swap) gives the computable form

            ΔE = 2·[ (n_i(b) − n_i(a)) + (n_j(a) − n_j(b)) − 2·A_ij ]

        so   log p̃_t(Swap2(x,i,j)) − log p̃_t(x) = t · σ · ΔE.

        The −2·A_ij term removes the i–j bond that n_i and n_j each already
        include. Same-label pairs (a = b) are the identity move and must give
        exactly 0, but the formula assumed a ≠ b: the brackets vanish while
        −2·A_ij survives, so adjacent same-label pairs would get a spurious
        −4σt that feeds `exp()` in the ξ_t inflow term. They are masked.

        Must agree with the inherited generic `swap_log_ratio`, the oracle in
        test_potts.py.
        """
        labels = self.to_index(x)  # (B, d)
        onehot = F.one_hot(labels, self.n_states).to(x.dtype)  # (B, d, S)
        neighbour_counts = torch.matmul(self.A, onehot)  # (B, d, S) = n_k(c)

        site_i, site_j, adjacent_ij = self._pair_columns(pairs)  # (P,) each
        label_i, label_j = labels[:, site_i], labels[:, site_j]  # (B, P) = a, b

        # Flat (site, species) indexing keeps every gather at (B, P) instead of
        # materialising (B, P, S).
        flat_counts = neighbour_counts.reshape(x.shape[0], -1)  # (B, d*S)

        def count_at(site: Tensor, label: Tensor) -> Tensor:
            return flat_counts.gather(1, site.unsqueeze(0) * self.n_states + label)

        delta_energy = 2.0 * (
            (count_at(site_i, label_j) - count_at(site_i, label_i))
            + (count_at(site_j, label_i) - count_at(site_j, label_j))
            - 2.0 * adjacent_ij
        )
        delta_energy = delta_energy.masked_fill(label_i == label_j, 0.0)
        return t[:, None] * self.sigma * delta_energy
