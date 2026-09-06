"""The block-occupancy base B(b, w) of the warm-base design, §2.2.

WHAT THIS IS
------------
A probability law on the fixed-composition slice
    S(d, N_A) = { x in {-1,+1}^d : #{i : x_i = +1} = N_A }
of the D x D torus (d = D^2) that is

  (i)   exactly samplable,
  (ii)  exactly normalised with a closed-form log-density,
  (iii) supported on the whole slice,
  (iv)  spatially ordered (nn-correlation strictly positive), and
  (v)   Z2- and translation-symmetric,

i.e. the five requirements a replacement for the current uniform-on-slice base
has to satisfy.

WHY THE CONSTRUCTION LOOKS LIKE THIS
------------------------------------
The natural warm base is a site-dependent product Bernoulli conditioned on the
slice.  Its normaliser
    sum_{x in S} prod_i p_i^{[x_i=+1]} (1-p_i)^{[x_i=-1]}
is a permanent-like sum with no closed form in general.  It IS closed form when
the field p_i is piecewise constant on a partition of the lattice: the
conditional law then factorises into "how many up-spins in each block" times
"uniform within each block", and the block-occupancy normaliser is a
one-dimensional convolution computable exactly by dynamic programming.

So: partition the torus into B = (D/b)^2 tiles of s = b^2 sites; draw the
occupancy vector (m_1, ..., m_B), m_t in {0..s}, sum_t m_t = N_A, with
probability proportional to prod_t w(m_t); then place m_t up-spins on a
uniformly random m_t-subset of tile t.  The density is

    eta_k(x) = prod_t w(m_t(x)) / ( Z_w * prod_t C(s, m_t(x)) ) ,
    Z_w      = sum_{ sum_t m_t = N_A } prod_t w(m_t) .                    (2.2)

Normalisation check (this is the whole point of the C(s, m) denominator):
    sum_{x in S} eta_k(x)
      = sum_{m : sum m_t = N_A} (#x with those occupancies) * prod_t w(m_t)
                                / (Z_w prod_t C(s, m_t))
      = sum_m prod_t C(s, m_t) * prod_t w(m_t) / (Z_w prod_t C(s, m_t))
      = Z_w / Z_w = 1 .

SYMMETRY
--------
Z2: impose w(m) = w(s - m).  The global flip maps m_t -> s - m_t and
N_A -> d - N_A, which at c = 0.5 is N_A again, so EACH component is
individually Z2-invariant -- no mixture is needed for this.

Translation: a single tiling is invariant only under translations by multiples
of b.  Mixing uniformly over the K = b^2 offsets of the tiling (its translation
orbit) restores exact translation invariance while keeping the density a
closed-form finite sum and keeping sampling exact (draw k uniformly, then run
the exact DP sampler for tiling k):

    eta(x) = (1/K) sum_{k=1..K} eta_k(x) .

FLOOR
-----
w <- (1 - eps) w + eps / (s + 1), eps = 1e-3, so every slice configuration keeps
strictly positive mass (requirement iii) and log eta is finite everywhere.
Without this an occupancy value never seen in the fitting draws would get
w(m) = 0 and a whole face of the slice would be assigned -inf log-density --
a coverage failure that the importance weights would report as NaN, not as a
number.

FAILURE MODE GUARDED AGAINST
----------------------------
Everything here is done in the log domain with logsumexp.  Z_w at d = 256 is a
sum over the compositions of 128 into 64 parts each <= 4; the linear-domain
value overflows float64 long before the DP finishes.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp


def torus_adjacency(lattice_side: int) -> np.ndarray:
    """The project's own adjacency, duplicated verbatim from
    `src/discrete_flow_sampler/targets/ising.py` (the A-construction in
    `IsingTarget.__init__`).

    Each site owns one "right" and one "down" bond; the matrix is then
    symmetrised, so every undirected bond contributes 2 to sum(A) and

        x^T A x = 2 * sum_{undirected <ij>} x_i x_j ,   sum(A) = 4d ,
        C(x)    = x^T A x / sum(A) .

    Duplicated rather than imported so this probe is self-contained; the
    self-test asserts it is elementwise identical to the library's.
    """
    side = lattice_side
    d = side * side
    A = np.zeros((d, d))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            A[i, r * side + (c + 1) % side] = 1.0
            A[i, ((r + 1) % side) * side + c] = 1.0
    return A + A.T


def nn_correlation(spins: np.ndarray, A: np.ndarray) -> np.ndarray:
    """C(x) = x^T A x / sum(A), matching `diagnostics/metrics.py:451-462`."""
    return quadratic_form(spins, A) / A.sum()


def quadratic_form(spins: np.ndarray, A: np.ndarray) -> np.ndarray:
    """x^T A x -- the sigma-free part of log rho(x) = sigma * x^T A x."""
    x = spins.astype(np.float64)
    return ((x @ A) * x).sum(axis=1)   # BLAS matmul, not einsum: 400k x 256 is
                                       # 2 GFLOP and einsum would not use BLAS


def log_binom(n: int, k: np.ndarray | int) -> np.ndarray:
    k = np.asarray(k)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


class BlockOccupancyBase:
    """B(b, w) on the c = N_A/d slice of a `lattice_side` x `lattice_side` torus.

    Parameters
    ----------
    lattice_side : D
    block_side   : b, must divide D
    n_up         : N_A
    weights      : the occupancy weight w(0..s).  `fit` builds it from draws.
    n_offsets    : K.  b*b for the full translation orbit (the design point),
                   1 for the single-tiling ablation.
    epsilon      : the floor.
    """

    def __init__(
        self,
        lattice_side: int,
        block_side: int,
        n_up: int,
        weights: np.ndarray,
        n_offsets: int | None = None,
        epsilon: float = 1e-3,
    ) -> None:
        if lattice_side % block_side:
            raise ValueError("block_side must divide lattice_side")
        self.lattice_side = lattice_side
        self.block_side = block_side
        self.d = lattice_side * lattice_side
        self.n_up = n_up
        self.tile_sites = block_side * block_side              # s
        self.n_tiles = self.d // self.tile_sites               # B
        self.epsilon = epsilon

        full_orbit = self._offset_list(block_side)
        self.n_offsets = block_side * block_side if n_offsets is None else n_offsets
        self.offsets = full_orbit[: self.n_offsets]

        self.log_weights = self._prepare_weights(weights)
        self.log_tile_multiplicity = log_binom(self.tile_sites,
                                               np.arange(self.tile_sites + 1))
        self.forward_dp = self._build_dp()                     # (B+1, N_A+1)
        self.log_normaliser = self.forward_dp[self.n_tiles, self.n_up]
        if not np.isfinite(self.log_normaliser):
            raise ValueError("Z_w is zero: no occupancy vector reaches N_A")

        # site -> tile map and its inverse, one per offset
        self.tile_of_site = np.stack(
            [self._tile_map(dr, dc) for dr, dc in self.offsets]
        )                                                       # (K, d)
        self.sites_of_tile = np.stack(
            [np.argsort(m, kind="stable").reshape(self.n_tiles, self.tile_sites)
             for m in self.tile_of_site]
        )                                                       # (K, B, s)

    # ---------------- construction helpers ----------------

    @staticmethod
    def _offset_list(block_side: int) -> list[tuple[int, int]]:
        return [(dr, dc) for dr in range(block_side) for dc in range(block_side)]

    def _prepare_weights(self, weights: np.ndarray) -> np.ndarray:
        """Z2-symmetrise w(m) = w(s - m), normalise, then floor.

        The flat floor preserves Z2 symmetry; a non-flat floor would make the
        order of symmetrisation and flooring matter.
        """
        w = np.asarray(weights, dtype=np.float64).copy()
        if w.shape != (self.tile_sites + 1,):
            raise ValueError(f"weights must have shape ({self.tile_sites + 1},)")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        w = 0.5 * (w + w[::-1])                      # Z2: w(m) = w(s - m)
        w = w / w.sum()
        w = (1.0 - self.epsilon) * w + self.epsilon / (self.tile_sites + 1)
        return np.log(w)

    def _tile_map(self, dr: int, dc: int) -> np.ndarray:
        """Flat site index -> tile index for the tiling shifted by (dr, dc).

        The shift is taken modulo D so the partition stays a partition of the
        torus (a tile may wrap around the boundary; that is exactly what makes
        the offset orbit a symmetry rather than an edge effect).
        """
        side, b = self.lattice_side, self.block_side
        tiles_per_row = side // b
        rows = np.arange(side)[:, None]
        cols = np.arange(side)[None, :]
        tr = ((rows - dr) % side) // b
        tc = ((cols - dc) % side) // b
        return (tr * tiles_per_row + tc).reshape(-1)

    def _build_dp(self) -> np.ndarray:
        """forward_dp[t, n] = log sum over occupancies of the first t tiles
        totalling n up-spins of prod w(m).  O(B * N_A * s), all in log space.
        """
        s, B, N = self.tile_sites, self.n_tiles, self.n_up
        dp = np.full((B + 1, N + 1), -np.inf)
        dp[0, 0] = 0.0
        for t in range(1, B + 1):
            prev = dp[t - 1]
            # contrib[m, n] = prev[n - m] + logw[m]
            contrib = np.full((s + 1, N + 1), -np.inf)
            for m in range(min(s, N) + 1):
                contrib[m, m:] = prev[: N + 1 - m] + self.log_weights[m]
            dp[t] = logsumexp(contrib, axis=0)
        return dp

    # ---------------- density ----------------

    def tile_occupancies(self, spins: np.ndarray, offset_index: int) -> np.ndarray:
        """m_t(x) for every tile, shape (N, B)."""
        up = (spins > 0)
        return up[:, self.sites_of_tile[offset_index]].sum(axis=2)

    def log_density_single(self, spins: np.ndarray, offset_index: int) -> np.ndarray:
        """log eta_k(x) for one tiling, shape (N,).  Eq. (2.2)."""
        m = self.tile_occupancies(spins, offset_index)
        return (
            self.log_weights[m].sum(axis=1)
            - self.log_tile_multiplicity[m].sum(axis=1)
            - self.log_normaliser
        )

    def log_density(self, spins: np.ndarray) -> np.ndarray:
        """log eta(x) = log( (1/K) sum_k eta_k(x) ), shape (N,).

        Kept as a logsumexp over the K components rather than a linear average:
        at d = 256 each eta_k is ~e^{-160}, so the linear mixture underflows.
        """
        spins = np.atleast_2d(spins)
        per_offset = np.stack(
            [self.log_density_single(spins, k) for k in range(self.n_offsets)]
        )
        return logsumexp(per_offset, axis=0) - np.log(self.n_offsets)

    # ---------------- exact sampling ----------------

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Exact draws from eta, shape (n, d), entries in {-1, +1}.

        Two stages, both exact:
          1. the DP run backwards -- draw m_B proportional to
             w(m) * exp(forward_dp[B-1, n_rem - m]), then m_{B-1}, ...;
          2. a uniformly random m_t-subset of each tile, done by ranking i.i.d.
             uniform keys within the tile (the vectorised form of the
             `randperm` idiom in `sample_base`).
        """
        which_offset = rng.integers(self.n_offsets, size=n_samples)
        spins = np.empty((n_samples, self.d), dtype=np.int8)
        for k in range(self.n_offsets):
            rows = np.flatnonzero(which_offset == k)
            if rows.size:
                spins[rows] = self._sample_one_tiling(rows.size, k, rng)
        return spins

    def _sample_one_tiling(self, n_samples: int, offset_index: int,
                           rng: np.random.Generator) -> np.ndarray:
        s, B, N = self.tile_sites, self.n_tiles, self.n_up
        occupancy = np.empty((n_samples, B), dtype=np.int64)
        remaining = np.full(n_samples, N, dtype=np.int64)
        m_grid = np.arange(s + 1)

        for t in range(B, 0, -1):
            back = remaining[:, None] - m_grid[None, :]         # (n, s+1)
            feasible = back >= 0
            logits = np.where(
                feasible,
                self.forward_dp[t - 1][np.clip(back, 0, N)] + self.log_weights[m_grid],
                -np.inf,
            )
            # Gumbel-max: exact categorical sampling without normalising.
            gumbel = -np.log(-np.log(rng.random((n_samples, s + 1))))
            drawn = np.argmax(logits + gumbel, axis=1)
            occupancy[:, t - 1] = drawn
            remaining -= drawn
        assert (remaining == 0).all(), "DP sampler failed to spend the budget"

        keys = rng.random((n_samples, B, s))
        rank_within_tile = np.argsort(np.argsort(keys, axis=2), axis=2)
        is_up = rank_within_tile < occupancy[:, :, None]

        spins = np.empty((n_samples, self.d), dtype=np.int8)
        sites = self.sites_of_tile[offset_index]                # (B, s)
        spins[:, sites.reshape(-1)] = np.where(
            is_up.reshape(n_samples, -1), 1, -1
        ).astype(np.int8)
        return spins

    # ---------------- fitting ----------------

    @classmethod
    def fit(
        cls,
        spins: np.ndarray,
        lattice_side: int,
        block_side: int,
        n_offsets: int | None = None,
        epsilon: float = 1e-3,
    ) -> "BlockOccupancyBase":
        """Moment-match w to the target's own tile-occupancy marginal, pooled
        over the FULL K = b^2 offset orbit (so the fitted w is
        translation-symmetric by construction even for the K = 1 ablation).

        This is the moment-matched choice within the "independent tiles,
        uniform inside a tile" family, and is close to but not exactly the
        KL-optimum: the sum_t m_t = N_A conditioning couples the tiles, an
        O(1/B) mismatch.
        """
        n_up = int((spins[0] > 0).sum())
        s = block_side * block_side
        scaffold = cls(lattice_side, block_side, n_up,
                       weights=np.ones(s + 1), n_offsets=None, epsilon=epsilon)
        counts = np.zeros(s + 1)
        for k in range(scaffold.n_offsets):
            m = scaffold.tile_occupancies(spins, k)
            counts += np.bincount(m.reshape(-1), minlength=s + 1)
        return cls(lattice_side, block_side, n_up, weights=counts,
                   n_offsets=n_offsets, epsilon=epsilon)


class UniformSliceBase:
    """The base actually in force today: uniform on the slice.

    `FixedCompositionIsingTarget.sample_base` (`ising.py:390-400`) draws a
    uniform N_A-subset, so log eta = -log C(d, N_A) for every slice state and
    Var[log eta] = 0 exactly.  Its nn-correlation is -1/(d-1) exactly (not
    asymptotically): exchangeability plus sum_{i != j} x_i x_j = M^2 - d with
    M = 2 N_A - d constant on the slice.
    """

    def __init__(self, lattice_side: int, n_up: int) -> None:
        self.lattice_side = lattice_side
        self.d = lattice_side * lattice_side
        self.n_up = n_up
        self.log_normaliser = float(log_binom(self.d, n_up))

    def log_density(self, spins: np.ndarray) -> np.ndarray:
        return np.full(np.atleast_2d(spins).shape[0], -self.log_normaliser)

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        keys = rng.random((n_samples, self.d))
        rank = np.argsort(np.argsort(keys, axis=1), axis=1)
        return np.where(rank < self.n_up, 1, -1).astype(np.int8)

    def exact_nn_correlation(self) -> float:
        """(M^2 - d) / (d (d-1)) with M = 2 N_A - d; = -1/(d-1) at c = 0.5."""
        magnetisation = 2 * self.n_up - self.d
        return (magnetisation**2 - self.d) / (self.d * (self.d - 1))
