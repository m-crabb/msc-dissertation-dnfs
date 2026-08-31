"""Ising target distribution. Paper Eq. (11): p(x) ∝ exp(x^T J x + b · Σx). """

import math
from contextlib import contextmanager

import torch
from torch import Tensor

from discrete_flow_sampler.composition import expand_b_major
from discrete_flow_sampler.samplers._neighbours import DEFAULT_LOG_RATIO_CLAMP

# THE project's critical coupling (canonical since the s58 migration,
# 2026-08-24): exact 2D Ising criticality under THIS repo's double-counted
# convention (x^T J x picks up each edge twice, so the per-bond coupling is
# 2*sigma): beta_c = ln(1+sqrt(2))/2 = 0.44069 gives sigma_c = ln(1+sqrt(2))/4.
# Every NEW cell, reference pool and figure uses this value.
SIGMA_C = math.log(1.0 + math.sqrt(2.0)) / 4.0  # = 0.220343...
SIGMA_C_EXACT = SIGMA_C  # alias kept for the s58 finding's test/readers

# The LEGACY critical coupling every archived pre-s58 "s223"/sigma_c run was
# trained and evaluated at, inherited from DNFS Table 2. The s58 finding
# (test_ising_exact.py): DNFS's own Table 2 "optimal" column at this label is
# in fact evaluated at SIGMA_C, so 0.22305 was never anyone's exact value.
# Archived cell definitions keep this literal (their stored configs and the
# eval config-drift guard are pinned to it, and it is what those models were
# trained at); scheduled retrain waves replace them with SIGMA_C cells.
SIGMA_C_LEGACY = 0.22305


class IsingTarget:
    """Periodic-boundary DxD Ising lattice with annealing path.

    Target distribution (paper Eq. 11):

        p(x) ∝ exp( x^T J x + bias · Σ_i x_i ),    x ∈ {-1, +1}^d, d = D²

    where J = sigma · A_D and A_D is the adjacency matrix of the DxD grid
    with periodic boundaries (the lattice is a torus, no edge effects).
    The convention is that A_D is symmetric with zeros on the diagonal,
    each nearest-neighbour pair {i, j} contributing entry sigma in BOTH
    A_D[i, j] and A_D[j, i] — so the quadratic form x^T J x picks up
    each edge twice. This matches the paper's igraph-based construction.

    Annealing path (paper Eq. 4):

        log p̃_t(x) = (1 - t) · log η(x) + t · log p(x)

    with η = uniform on {-1, +1}^d (so log η ≡ -d · log 2, constant in x)
    and log p(x) = log_prob(x) - log Z. Working with the unnormalised
    log_prob is fine because the constant log Z cancels in derivatives.

    Because the path is linear in log, the time-derivative is path-position
    independent:

        ∂_t log p̃_t(x) = log p(x) - log η(x)         [t-independent]

    This is the term DNFS estimates the expectation of (under p̃_t) to form
    the ∂_t log Z_t signal in the Kolmogorov-residual training loss.
    """

    def __init__(
        self,
        D: int,
        sigma: float,
        bias: float = 0.0,
        device: torch.device | str = "cpu",
        target_composition: float | None = None,
        composition_penalty_strength: float = 0.0,
        base_composition: float = 0.5,
        log_ratio_clamp: float = DEFAULT_LOG_RATIO_CLAMP,
    ):
        if log_ratio_clamp <= 0.0:
            raise ValueError(
                "log_ratio_clamp must be positive, got " f"{log_ratio_clamp}"
            )
        if not 0.0 < base_composition < 1.0:
            raise ValueError(
                "base_composition must be in the open interval (0, 1), "
                f"got {base_composition}"
            )
        if target_composition is not None and not 0.0 <= target_composition <= 1.0:
            raise ValueError(
                "target_composition must be in [0, 1], "
                f"got {target_composition}"
            )
        if composition_penalty_strength < 0.0:
            raise ValueError(
                "composition_penalty_strength must be non-negative, "
                f"got {composition_penalty_strength}"
            )
        if composition_penalty_strength > 0.0 and target_composition is None:
            raise ValueError(
                "target_composition must be set when "
                "composition_penalty_strength is nonzero"
            )

        self.D = D
        self.d = D * D
        self.sigma = sigma
        self.bias = bias
        self.device = torch.device(device)
        self.target_composition = target_composition
        self.composition_penalty_strength = composition_penalty_strength
        self.base_composition = base_composition
        # Ceiling for log p̃_t(y)/p̃_t(x) at single-flip neighbours. The
        # composition penalty contributes ∓2λ·(c(x)−c_target) to that ratio,
        # so the paper's 5 binds once the obedience error exceeds 5/(2λ) —
        # 0.05 at λ=50, inside the error a conditioned sampler achieves.
        self.log_ratio_clamp = log_ratio_clamp
        # Per-row composition bound by `composition_batch`. None means the
        # scalar `target_composition` is in force — the specialist path.
        self._bound_composition: Tensor | None = None

        A = torch.zeros((self.d, self.d), device=self.device)

        for r in range(self.D):
            for c in range(self.D):
                i = r * self.D + c                          # (r, c)             -> flat
                right = r * self.D + (c + 1) % self.D       # (r, (c+1) % D)     -> flat
                down = ((r + 1) % self.D) * self.D + c      # ((r+1) % D, c)     -> flat
                A[i, right] = 1.0
                A[i, down] = 1.0

        # symmetrise so the matrix is symmetric (undirected edges)
        A = A + A.T
        self.A = A                    # kept for `set_sigma` rescaling
        self.J = self.sigma * A

    def set_sigma(self, sigma: float) -> None:
        """Mutate σ in place; rescales J = σ · A.

        Used by temperature curricula to move through easier intermediate
        targets without rebuilding model or optimizer state.
        """
        self.sigma = sigma
        self.J = sigma * self.A

    def set_composition_penalty_strength(self, strength: float) -> None:
        """Mutate λ in place.

        Used by λ-annealing curricula to tighten the soft composition
        constraint without rebuilding model or optimizer state.
        """
        if strength < 0.0:
            raise ValueError(
                "composition_penalty_strength must be non-negative, "
                f"got {strength}"
            )
        if strength > 0.0 and self.target_composition is None:
            raise ValueError(
                "target_composition must be set when "
                "composition_penalty_strength is nonzero"
            )
        self.composition_penalty_strength = strength

    # Denominator of the realisable compositions, or None if c is free. The
    # soft penalty accepts any real c, so an amortised trainer may draw from a
    # continuum. A fixed-composition target cannot — there c·d must be an
    # integer or no exact slice exists — so it sets this to `d` and the
    # trainer quantises its draws onto that lattice.
    composition_quantum: int | None = None

    def composition_fraction(self, x: Tensor) -> Tensor:
        """Fraction of +1 spins in each state, shape (B,)."""
        return ((x + 1.0) * 0.5).mean(dim=-1)

    def base_log_eta(self, x: Tensor) -> Tensor:
        """Log-density of the per-site Bernoulli base η, shape (B,).

        η(x) = ∏_i p^{[x_i=+1]} (1-p)^{[x_i=-1]}, p = base_composition.
        For the uniform base (p=0.5) this is the constant -d·log2 for all x;
        we return that exact expression so the annealing path stays
        byte-identical to a uniform base.
        """
        if self.base_composition == 0.5:
            return torch.full(
                (x.shape[0],), -self.d * math.log(2),
                device=x.device, dtype=x.dtype,
            )
        n_plus = ((x + 1.0) * 0.5).sum(dim=-1)
        return (
            n_plus * math.log(self.base_composition)
            + (self.d - n_plus) * math.log(1.0 - self.base_composition)
        )

    def sample_base(self, n: int, device) -> Tensor:
        """Draw n states from the base η, shape (n, d), entries in {-1, +1}.

        At p=0.5 this is a plain randint draw (identical RNG consumption, so
        existing runs reproduce bit-for-bit).
        """
        if self.base_composition == 0.5:
            return torch.randint(0, 2, (n, self.d), device=device).float() * 2 - 1
        return (
            (torch.rand(n, self.d, device=device) < self.base_composition)
            .float() * 2 - 1
        )

    def base_log_prob(self, x: Tensor) -> Tensor:
        """Unnormalised Ising log-density before optional soft constraints.

        x: (B, d) float tensor with entries in {-1, +1}.
        Returns: (B,) tensor.

            log_prob(x) = x^T J x  +  bias · Σ_i x_i

        Note: omits the log-partition-function constant log Z; this is the
        un-normalised log p.
        """
        return (x @ self.J * x).sum(dim=-1)  + (self.bias * x.sum(dim=1))

    @contextmanager
    def composition_batch(self, composition: Tensor):
        """Bind a per-row target composition for the duration of the block.

        Used by the amortised sampler, where one model is trained to serve
        many compositions, so a training batch carries a *different* c per
        row:

            log p_c(x_b) = x_b^T J x_b − λ · d · (c_+(x_b) − c_b)^2

        c is bound rather than passed as an argument because the penalty is
        reached indirectly through `log_prob` → `log_p_tilde_t` /
        `dt_log_p_tilde_t`, which are called from ~15 sites across
        `kolmogorov.py`, `_neighbours.py`, `ctmc.py` and the swap stack.
        Threading an argument through all of them would churn signatures the
        hard leg also depends on.

        Args:
            composition: (n_blocks,) tensor of target compositions in [0, 1],
                one per row of the batch this block will evaluate. Batches
                that are an integer multiple of `n_blocks` are expanded
                b-major — see `_row_composition`.

        On exit the previous binding is restored, including when the block
        raises, so no run can leak a bound vector into a later evaluation.

        This is the shared seam for both constraint routes. Here the bound
        composition feeds the soft penalty; a fixed-composition subclass is
        expected to honour the same binding wherever it currently reads its
        scalar composition (its base sampler, its slice-size constant, its
        off-manifold check), so an amortised trainer needs no knowledge of
        which route it is driving.
        """
        composition = torch.as_tensor(
            composition, dtype=torch.float, device=self.device
        )
        if composition.ndim != 1:
            raise ValueError(
                f"composition must be 1-D (one entry per block), got shape "
                f"{tuple(composition.shape)}"
            )
        if not ((composition >= 0.0) & (composition <= 1.0)).all():
            raise ValueError("composition entries must lie in [0, 1]")

        previous = self._bound_composition
        self._bound_composition = composition
        try:
            yield
        finally:
            self._bound_composition = previous

    def _row_composition(self, x: Tensor) -> Tensor | float | None:
        """Target composition for each row of `x`. Scalar when nothing is bound.

        A bound vector is aligned to `x` by `composition.expand_b_major`,
        which documents why the expansion rule matters and why a ragged
        batch is a hard error rather than a broadcast.

        Returns:
            None when no composition is configured at all, the scalar
            `self.target_composition` when nothing is bound, else a (B,)
            tensor aligned row-for-row with `x`.
        """
        if self._bound_composition is None:
            return self.target_composition

        return expand_b_major(self._bound_composition, x.shape[0])

    def composition_penalty(self, x: Tensor) -> Tensor:
        """Extensive soft-composition penalty, shape (B,).

        The form mirrors VCSGC-style concentration control by scaling the
        squared composition deviation by the number of sites:

            λ · d · (c_+(x) - c_target)^2

        It is subtracted from `log_prob`, equivalently added to the target
        energy. Under `composition_batch` the scalar c_target becomes a
        per-row vector; the expression is otherwise unchanged.
        """
        composition = self._row_composition(x)
        if composition is None or self.composition_penalty_strength == 0.0:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        diff = self.composition_fraction(x) - composition
        return self.composition_penalty_strength * self.d * diff.pow(2)

    def log_prob(self, x: Tensor) -> Tensor:
        """Un-normalised target log-density.

        x: (B, d) float tensor with entries in {-1, +1}.
        Returns: (B,) tensor.

            log_prob(x) = base_log_prob(x) - composition_penalty(x)

        With no soft-composition constraint, this reduces to the base Ising
        log-density.
        """
        return self.base_log_prob(x) - self.composition_penalty(x)

    def log_p_tilde_t(self, x: Tensor, t: Tensor) -> Tensor:
        """Annealing-path log-density at time t (paper Eq. 4).

        x: (B, d) float tensor.
        t: (B,) tensor with entries in [0, 1].
        Returns: (B,) tensor.

            log p̃_t(x) = (1 - t) · log η(x) + t · log p(x)
                       = (1 - t) · base_log_eta(x) + t · log_prob(x)

        η is the per-site Bernoulli base (uniform when base_composition=0.5,
        in which case base_log_eta is the constant -d · log 2).
        At t=0: returns base_log_eta(x) (the base).
        At t=1: returns log_prob(x) (full target).
        """
        return (1 - t) * self.base_log_eta(x) + (t * self.log_prob(x))

    def dt_log_p_tilde_t(self, x: Tensor, t: Tensor) -> Tensor:
        """Time-derivative of the annealing log-density. t-independent.

        x: (B, d) float tensor.
        t: (B,) tensor (unused; kept in signature to match log_p_tilde_t).
        Returns: (B,) tensor.

            ∂_t log p̃_t(x) = log p(x) - log η(x)
                            = log_prob(x) - base_log_eta(x)

        For the uniform base (base_composition=0.5) base_log_eta is -d · log 2,
        so this reduces to log_prob(x) + d · log 2. The expression has no
        t-dependence - that's the consequence of a linear-in-log annealing path.
        """
        return self.log_prob(x) - self.base_log_eta(x)

    def swap_log_ratio(self, x: Tensor, t: Tensor, pairs: Tensor) -> Tensor:
        """log p̃_t(Swap2(x, i, j)) − log p̃_t(x) for each pair, shape (B, P).

        Generic fallback: materialise the swapped states and evaluate the
        annealing density directly. Correct for any target (the same
        build-and-evaluate path as `_log_p_tilde_at_swap_neighbours`), and it
        doubles as the oracle the closed-form overrides are tested against.
        Subclasses on a fixed-composition slice override this with a closed form
        that skips the (B, P, d) materialisation.
        """
        batch_size, d = x.shape
        n_pairs = pairs.shape[0]
        i_col = pairs[:, 0].view(1, n_pairs, 1).expand(batch_size, n_pairs, 1)
        j_col = pairs[:, 1].view(1, n_pairs, 1).expand(batch_size, n_pairs, 1)
        y = x[:, None, :].expand(batch_size, n_pairs, d).clone()
        spin_i = y.gather(2, i_col)
        spin_j = y.gather(2, j_col)
        y.scatter_(2, i_col, spin_j)
        y.scatter_(2, j_col, spin_i)
        neighbours = self.log_p_tilde_t(
            y.reshape(batch_size * n_pairs, d), t.repeat_interleave(n_pairs)
        ).reshape(batch_size, n_pairs)
        return neighbours - self.log_p_tilde_t(x, t)[:, None]

    def _pair_columns(self, pairs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(site_i, site_j, A_ij) for a pairs tensor, cached by identity.

        B4 (optimisation decision, 2026-08-24): the closed-form swap
        log-ratios gather A[site_i, site_j] from the frozen adjacency on
        every call, and the production samplers pass the same module-level
        `upper_tri_pairs` cache object every step — so the (P,) gather is
        identical each time. A single-slot cache keyed by tensor identity
        (`is`, not value equality, so the hit stays O(1)) pays it once;
        any other pairs tensor recomputes correctly through the miss path.
        """
        cached = getattr(self, "_pair_columns_cache", None)
        if cached is not None and cached[0] is pairs:
            return cached[1], cached[2], cached[3]
        site_i, site_j = pairs[:, 0], pairs[:, 1]
        adjacent = self.A[site_i, site_j]
        self._pair_columns_cache = (pairs, site_i, site_j, adjacent)
        return site_i, site_j, adjacent


class FixedCompositionIsingTarget(IsingTarget):
    """Ising target on the fixed-composition manifold C = {n_plus = N_A}.

    Hard-constraint counterpart of IsingTarget: composition is enforced by the
    swap move set, not a soft penalty, so this target carries no
    composition_penalty. It differs from IsingTarget only in the base:

      * sample_base draws uniformly over configs with exactly N_A up-spins (the
        canonical fixed-N base), NOT product-Bernoulli.
      * base_log_eta returns the constant -log C(d, N_A) on the slice.

    The config knob is the composition fraction c (target_composition);
    N_A = round(c * d) is its integer realisation. On the slice every fixed-N
    config is equiprobable under any product base, so c drops out of
    base_log_eta, leaving only the slice-size constant.
    """

    def __init__(self, D, sigma, target_composition, bias=0.0, device="cpu"):
        n_plus_float = target_composition * (D * D)
        n_plus_target = round(n_plus_float)
        if abs(n_plus_float - n_plus_target) > 1e-9:
            raise ValueError(
                f"target_composition={target_composition} * d={D * D} = "
                f"{n_plus_float} is not integral; no exact fixed-N slice exists."
            )
        super().__init__(
            D, sigma, bias=bias, device=device,
            target_composition=target_composition,
            composition_penalty_strength=0.0,
        )
        self.n_plus_target = n_plus_target
        self.composition_quantum = self.d
        self._log_slice_size = (
            math.lgamma(self.d + 1)
            - math.lgamma(n_plus_target + 1)
            - math.lgamma(self.d - n_plus_target + 1)
        )  # log C(d, N_A)

    def sample_base(self, n, device):
        """Uniform over fixed-N configs: a random N_A-subset of sites set to +1.

        argsort of per-row uniforms is a uniform random permutation; its first
        N_A columns are a uniform random N_A-subset of site indices.
        """
        permuted_sites = torch.rand(n, self.d, device=device).argsort(dim=1)
        x = torch.full((n, self.d), -1.0, device=device)
        up_sites = permuted_sites[:, : self.n_plus_target]
        x.scatter_(1, up_sites, 1.0)
        return x

    def base_log_eta(self, x):
        """Constant log-density -log C(d, N_A) on the slice, shape (B,)."""
        return torch.full(
            (x.shape[0],), -self._log_slice_size,
            device=x.device, dtype=x.dtype,
        )

    def assert_on_manifold(self, x):
        """Raise AssertionError if any row has n_plus != N_A."""
        n_plus = ((x + 1) * 0.5).sum(dim=-1)
        if not torch.all(n_plus == self.n_plus_target):
            bad = n_plus[n_plus != self.n_plus_target]
            raise AssertionError(
                f"off-manifold states: expected n_plus={self.n_plus_target}, "
                f"got e.g. {bad[:5].tolist()}"
            )

    def swap_log_ratio(self, x, t, pairs):
        """Closed-form swap log-ratio on the fixed-N slice, shape (B, P).

        Because base_log_eta is constant on the slice (so the (1 − t) term
        cancels) and bias·Σx is swap-invariant, only the t·σ·Δ(xᵀAx) term
        survives:

            log p̃_t(Swap2(x, i, j)) − log p̃_t(x)
                = t·σ·[ 2(x_j − x_i)(h_i − h_j) − 2(x_j − x_i)²·A_ij ],   h = x·A

        the batched form of the single-move Kawasaki ΔE. Same-spin pairs
        (x_i = x_j ⇒ diff = 0) give 0 for free. Cost O(B·d² + B·P), with no
        (B, P, d) neighbour materialisation.
        """
        h = x @ self.A                                    # (B, d) neighbour sums
        site_i, site_j, adjacent = self._pair_columns(pairs)  # (P,) each
        diff = x[:, site_j] - x[:, site_i]                # (B, P)
        delta_quadratic = (
            2.0 * diff * (h[:, site_i] - h[:, site_j]) - 2.0 * diff * diff * adjacent
        )
        return t[:, None] * self.sigma * delta_quadratic


class MixtureCompositionIsingTarget(FixedCompositionIsingTarget):
    """Ising target on a MIXTURE of fixed-composition slices, for the
    composition-amortisation campaign (one head trained across slices).

    THE ALGEBRA THIS CLASS RESTS ON. Swap moves conserve n_plus row-wise,
    so a trajectory never leaves the slice its base draw started on: the
    mixture lives entirely in `sample_base`, and each batch element's
    annealing path, rates and IS weights are exact against ITS OWN slice
    conditional. Three consequences, one per override:

      * `sample_base` draws a slice per element uniformly from the grid,
        then a uniform configuration on that slice — that is the whole
        amortisation;
      * `base_log_eta` is computed FROM x as -log C(d, n_plus(x)) instead
        of a stored constant, which is exactly what makes the per-slice
        geometric path log p~_t = (1-t)*base_log_eta(x) + t*log_prob(x)
        correct for every row simultaneously;
      * `assert_on_manifold` checks membership of the registered slice
        SET, not one count.

    `swap_log_ratio` is inherited untouched: the swapped and unswapped
    states always share a slice, so the base constant cancels pairwise
    exactly as in the single-slice closed form.

    NO conditioning channel is added to any head: composition is conserved
    and visible in x (the spin count), so the head amortises implicitly
    through its input. An explicit c-input was rejected — it duplicates
    information the head already has and breaks checkpoint compatibility
    with every archived head.

    `compositions[0]` is the ANCHOR slice: the inherited scalar attributes
    (`n_plus_target`, `_log_slice_size`, `target_composition`) refer to it,
    which keeps single-slice diagnostics meaningful. Put the trained
    chapter composition (0.5) first.
    """

    def __init__(self, D, sigma, compositions, bias=0.0, device="cpu"):
        if not compositions:
            raise ValueError("compositions must name at least one slice")
        super().__init__(
            D, sigma, target_composition=compositions[0], bias=bias,
            device=device,
        )
        n_plus_values = []
        for c in compositions:
            n_plus_float = c * self.d
            n_plus = round(n_plus_float)
            if abs(n_plus_float - n_plus) > 1e-9:
                raise ValueError(
                    f"composition {c} * d={self.d} = {n_plus_float} is not "
                    "integral; no exact fixed-N slice exists."
                )
            n_plus_values.append(n_plus)
        self.compositions = tuple(compositions)
        self.n_plus_values = tuple(n_plus_values)
        # log C(d, n) for every n in 0..d, so base_log_eta is a lookup on
        # the per-row count. d+1 floats; float64 because at d=256 the
        # binomial coefficients differ across the grid by ~40 nats and the
        # table is the one place slice constants must stay exact.
        counts = torch.arange(self.d + 1, dtype=torch.float64)
        self._log_binomial_table = (
            math.lgamma(self.d + 1)
            - torch.lgamma(counts + 1)
            - torch.lgamma(self.d - counts + 1)
        )

    def sample_base(self, n, device):
        """Uniform slice choice per element, then uniform on that slice:
        the first n_plus[i] columns of a per-row random permutation are a
        uniform random subset of that size."""
        slice_index = torch.randint(
            len(self.n_plus_values), (n,), device=device)
        counts = torch.tensor(
            self.n_plus_values, device=device)[slice_index]     # (n,)
        permuted_sites = torch.rand(n, self.d, device=device).argsort(dim=1)
        rank = torch.arange(self.d, device=device).expand(n, -1)
        values = torch.where(rank < counts[:, None], 1.0, -1.0)
        x = torch.empty(n, self.d, device=device)
        x.scatter_(1, permuted_sites, values)
        return x

    def base_log_eta(self, x):
        """-log C(d, n_plus(x)) per row, shape (B,): each row's own slice
        constant, read off the state."""
        n_plus = ((x + 1.0) * 0.5).sum(dim=-1).long()
        table = self._log_binomial_table.to(x.device)
        return (-table[n_plus]).to(x.dtype)

    def assert_on_manifold(self, x):
        """Raise AssertionError if any row's n_plus is outside the
        registered slice set."""
        n_plus = ((x + 1) * 0.5).sum(dim=-1)
        allowed = torch.tensor(
            self.n_plus_values, device=x.device, dtype=n_plus.dtype)
        on_a_slice = (n_plus[:, None] == allowed[None, :]).any(dim=1)
        if not on_a_slice.all():
            bad = n_plus[~on_a_slice]
            raise AssertionError(
                f"off-manifold states: expected n_plus in "
                f"{self.n_plus_values}, got e.g. {bad[:5].tolist()}"
            )
