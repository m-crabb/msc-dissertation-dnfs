"""Potts target distribution: the S-species generalisation of the Ising target.

Scope + rationale: `docs/design/2026-07-31-potts-extension-scope.md`.

Why Potts
---------
Ising fixes S = 2. The Potts model keeps the same lattice and the same
annealing path but lets each site carry one of S labels, with the coupling
rewarding *agreement* rather than a product of spins:

    log p(s) = σ · Σ_{i,j} A_ij · δ(s_i, s_j)                          (Potts)
    log p(x) = σ · Σ_{i,j} A_ij · x_i x_j    + bias·Σ_i x_i            (Ising)

A_ij is the same periodic D×D adjacency IsingTarget builds — reused verbatim,
not rebuilt. It is symmetric with zero diagonal, so each undirected edge is
counted TWICE in both sums (`A.sum() == 4d` for the 2d edges of the torus).
Sharing that convention is what makes the S=2 correspondence below exact: any
edge-counting difference would show up as a coupling rescale, not an error.

Two consequences of the double counting worth knowing:

* The effective per-bond coupling is 2σ, since σ·Σ_{i,j} A_ij(·) = 2σ·Σ_⟨ij⟩(·).
  This is already baked into the project's σ_c ≈ 0.223 gate value (the 2D Ising
  critical coupling ln(1+√2)/2 ≈ 0.4407, halved).
* **D=2 is degenerate and should not be used for physics.** On a 2-cycle a
  site's left and right neighbours are the SAME site, so the loop writes that
  edge twice before symmetrisation doubles it again: A has entries of 2 rather
  than 1 (`A.sum()` still equals 4d). That is a genuine property of the L=2
  torus as a multigraph, not a bug, and it affects Ising identically — but it
  makes D=2 an atypical lattice. It is used here only in plumbing tests whose
  assertions are convention-agnostic.

The Ising form is not a special case of the Potts form by substitution — it
is a special case by *identity*. For x ∈ {−1,+1},

    δ(s_i, s_j) = (1 + x_i x_j) / 2

so    σ_P · Σ A_ij δ(s_i,s_j) = (σ_P/2)·Σ A_ij x_i x_j + (σ_P/2)·Σ A_ij.

Hence **S=2 Potts at coupling σ_P = 2σ equals Ising at coupling σ, plus an
x-independent constant** (σ·Σ A_ij = σ·2·|edges|). The constant shifts log Z
by a known amount and cancels identically in every log-ratio, so it never
reaches the sampler. This is the regression test in test_potts.py, and it is
the reason `sigma` here means the *Potts* coupling: a caller reproducing an
Ising run at σ must pass 2σ.

Copying `IsingTarget.base_log_prob` would be silently wrong for S > 2 — the
quadratic `x_i x_j` on labels {−1,1,3,5,...} takes values {1,−1,−3,9,...},
which is not an indicator and not any Potts model. No error would be raised.

State encoding
--------------
States are stored in the codebase's existing affine convention

    x = 2·label − 1,    label ∈ {0, …, S−1}    ⇒    x ∈ {−1, 1, 3, …, 2S−3}

as a float tensor, exactly as Ising stores {−1,+1}. This is deliberate: every
head and backbone recovers the embedding index with `((x + 1) / 2).long()`
(13 call sites), which inverts this map for ANY S. So the whole swap/head
stack runs on Potts unchanged — see the scope doc §"Why this is cheaper than
expected". `to_index` / `from_index` below make the convention explicit and
give a single place to change it later.

What is NOT carried over
------------------------
* **No `bias` field.** On the fixed-composition manifold a per-species field
  Σ_i h_{s_i} depends only on the species COUNTS, which are frozen by the
  constraint — so it is an additive constant there and does exactly nothing.
  Omitted rather than implemented-and-ignored.
* **No soft composition penalty.** The hard route enforces composition through
  the swap move set. `composition_penalty` is inherited returning zeros
  (target_composition=None), which is what the hard cells already rely on.
* **`composition_fraction` is meaningless for S > 2** — a Potts composition is
  an S-vector, not a scalar — so it is overridden to raise. Use
  `composition_counts`.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.targets.ising import IsingTarget


class PottsTarget(IsingTarget):
    """Periodic D×D Potts lattice with the shared annealing path.

    Subclasses IsingTarget to REUSE, not to specialise: the periodic adjacency
    construction, `set_sigma`, the annealing path `log_p_tilde_t` /
    `dt_log_p_tilde_t` (Eq. 4 — linear in log, so t-independent derivative),
    `log_prob`, and the generic materialise-and-evaluate `swap_log_ratio` are
    all encoding-agnostic and correct as inherited. Only the pieces that read
    `x` as a NUMBER rather than a LABEL are overridden here.

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

        Inverse of `from_index`; the same map every head applies inline as
        `((x + 1) / 2).long()`. Centralised here so the encoding has one
        owner — see the module docstring.
        """
        return ((x + 1) / 2).long()

    def from_index(self, index: Tensor) -> Tensor:
        """Stored spins x = 2·index − 1 from label indices, shape (B, d) float."""
        return (index * 2 - 1).float()

    def composition_counts(self, x: Tensor) -> Tensor:
        """Per-species site counts, shape (B, S).

        The Potts analogue of `composition_fraction`. Rows sum to d. This is
        the quantity the swap move set conserves exactly (a swap permutes
        labels, so it cannot change the multiset), which is why the hard
        constraint generalises to S > 2 for free.
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
        matching IsingTarget's convention so the S=2 correspondence in the
        module docstring is exact.

        Implementation note: build the one-hot Ω ∈ {0,1}^{B×d×S} of labels;
        then Σ_{i,j} A_ij δ(s_i,s_j) = Σ_a (Ω_a)ᵀ A (Ω_a), i.e. contract the
        adjacency against each species channel and sum. That is one batched
        matmul rather than a (B, d, d) pairwise-equality tensor.
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

    The hard-constraint counterpart, and the reason the Potts extension is
    cheap: a swap permutes two labels, so it preserves the label MULTISET
    exactly. That is precisely this manifold, generalised from Ising's single
    scalar n_plus to an S-vector of counts. `samplers/swap_ctmc.py` therefore
    needs no changes at all.

    Differs from PottsTarget only in the base:
      * `sample_base` draws uniformly over the multiset slice (a random
        permutation of a fixed label multiset), not i.i.d. uniform labels;
      * `base_log_eta` is the constant −log|C|, where |C| is the MULTINOMIAL
        coefficient d! / ∏_a N_a! — not Ising's binomial C(d, N_A).

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

        A uniform random permutation of the fixed label multiset is a uniform
        draw from C. `argsort` of per-row uniforms IS a uniform random
        permutation (the same trick IsingTarget.sample_base uses for its
        N_A-subset), so scattering the sorted multiset through it is uniform
        on the slice.
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

        Every state in C is equiprobable under the uniform base, so the base
        contributes no x-dependence — which is why the (1−t) term of the
        annealing path drops out of `swap_log_ratio` entirely.
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
        constant on C so the (1−t) term cancels, leaving only t·σ·ΔE.

        Derivation. Write the energy as E(s) = Σ_k n_k(s_k) where
        n_k(a) = Σ_l A_kl·δ(s_l, a) is the adjacency-weighted count of
        neighbours of site k carrying label a. Swapping labels a = s_i and
        b = s_j changes only terms touching i or j. Splitting off the
        {i, j} pair itself (whose contribution 2·A_ij·δ(a,b) is symmetric in
        a,b and therefore CANCELS), and writing the exclusive neighbour counts
        m_k(·) = n_k(·) − A_ki·δ(s_i,·) − A_kj·δ(s_j,·):

            ΔE = 2·[ m_i(b) − m_i(a) + m_j(a) − m_j(b) ]

        Re-expanding m in terms of n (using A_ii = 0, and δ(a,b) = 0 since
        a ≠ b for any non-trivial swap) gives the computable form

            ΔE = 2·[ (n_i(b) − n_i(a)) + (n_j(a) − n_j(b)) − 2·A_ij ]

        so   log p̃_t(Swap2(x,i,j)) − log p̃_t(x) = t · σ · ΔE.

        The −2·A_ij correction is the part that is easy to drop: it removes
        the double-counted i–j bond that n_i and n_j each already include.

        Same-label pairs (a = b) are the identity move and must give EXACTLY
        0, but the formula above does NOT deliver that for free: the count
        brackets vanish while the −2·A_ij term survives, so adjacent
        same-label pairs would pick up a spurious −4σt. The derivation assumed
        a ≠ b (it used δ(a,b) = 0), so a = b must be masked explicitly. These
        columns feed `exp()` in the ξ_t inflow term, where a fake rate on a
        no-op move would silently bias the sampler.

        Cost: one (B, d, S) neighbour-count tensor via `A @ onehot`, then O(B·P)
        flat gathers — the (B, P, S) intermediate is deliberately avoided.

        MUST agree with the inherited generic `swap_log_ratio` (which
        materialises the swapped states and re-evaluates); that generic path
        is the oracle in test_potts.py.
        """
        labels = self.to_index(x)  # (B, d)
        onehot = F.one_hot(labels, self.n_states).to(x.dtype)  # (B, d, S)
        neighbour_counts = torch.matmul(self.A, onehot)  # (B, d, S) = n_k(c)

        site_i, site_j = pairs[:, 0], pairs[:, 1]
        label_i, label_j = labels[:, site_i], labels[:, site_j]  # (B, P) = a, b

        # Flat (site, species) indexing keeps every gather at (B, P) instead of
        # materialising (B, P, S), which is the whole point of the closed form.
        flat_counts = neighbour_counts.reshape(x.shape[0], -1)  # (B, d*S)

        def count_at(site: Tensor, label: Tensor) -> Tensor:
            return flat_counts.gather(1, site.unsqueeze(0) * self.n_states + label)

        delta_energy = 2.0 * (
            (count_at(site_i, label_j) - count_at(site_i, label_i))
            + (count_at(site_j, label_i) - count_at(site_j, label_j))
            - 2.0 * self.A[site_i, site_j]
        )
        delta_energy = delta_energy.masked_fill(label_i == label_j, 0.0)
        return t[:, None] * self.sigma * delta_energy
