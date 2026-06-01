"""Kawasaki (composition-preserving swap) MCMC for the hard-constrained Ising
canonical ensemble.

The move exchanges the spins of one +1 site and one -1 site chosen uniformly at
random anywhere on the torus (non-local swap — the practitioner-standard
canonical-ensemble move, and a deliberately strong baseline). Composition
c(x) = #{+1}/d is therefore invariant by construction: there is no penalty term,
and the hard constraint c(x) = c_target holds exactly at every step. This is the
§3.1 counterpart to the soft VCSGC sampler in scripts/vcsgc_mcmc_validation.py.

Energy convention matches IsingTarget.base_log_prob (bias=0, swap-invariant):

    log_prob_ising(x) = sigma * x^T A x = sigma * sum_i x_i * neighbour_sum(i),

with A the symmetric DxD-torus adjacency (each edge counted twice). Acceptance
is min(1, exp(delta log_prob_ising)).

Derivation of the swap Δ. For a swap of opposite sites i, j with old values
a = x_i and b = x_j = -a, write the change as two simultaneous single-site
changes δ_i = b - a at i and δ_j = a - b = -δ_i at j. The quadratic form changes
by the two single-site terms plus a cross term for the shared bond:

    Δ(x^T A x) = 2 δ_i N(i) + 2 δ_j N(j) + 2 A_ij δ_i δ_j
               = 2 (b - a) [N(i) - N(j)] - 2 * 1[i~j] * (b - a)^2,

where N(k) = sum_{n in nbr(k)} x_n is the OLD neighbour-sum (computed on the
current x, including the other swapped site). The correction -2·1[i~j]·(b-a)^2
removes the double-counted shared bond: its product x_i x_j is invariant under
the exchange, but each neighbour-sum counts the other site. The bias term
bias·Σx_i is unchanged by a swap (composition fixed ⇒ Σx_i fixed), so it never
enters Δ. See tests/test_kawasaki.py::test_kawasaki_dE_matches_recompute_adjacent.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:                                   # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def neighbour_sum(x, i, D):
    """Sum of x over the 4 nearest neighbours of site i on a DxD torus."""
    r = i // D
    c = i % D
    up = ((r - 1) % D) * D + c
    down = ((r + 1) % D) * D + c
    left = r * D + (c - 1) % D
    right = r * D + (c + 1) % D
    return x[up] + x[down] + x[left] + x[right]


@njit(cache=True)
def _is_adjacent(i, j, D):
    r = i // D
    c = i % D
    return (j == ((r - 1) % D) * D + c or j == ((r + 1) % D) * D + c
            or j == r * D + (c - 1) % D or j == r * D + (c + 1) % D)


@njit(cache=True)
def kawasaki_delta_log_prob(x, i, j, D, sigma):
    """Δlog_prob_ising for swapping opposite-spin sites i and j."""
    diff = x[j] - x[i]                                # = x_i' - x_i = -(x_j' - x_j)
    delta_quadratic = 2.0 * diff * (neighbour_sum(x, i, D) - neighbour_sum(x, j, D))
    if _is_adjacent(i, j, D):
        delta_quadratic -= 2.0 * diff * diff
    return sigma * delta_quadratic


@njit(cache=True)
def initial_log_prob_ising(x, D, sigma):
    """log_prob_ising(x) = sigma * sum_i x_i * neighbour_sum(i)."""
    total = 0.0
    for i in range(D * D):
        total += x[i] * neighbour_sum(x, i, D)
    return sigma * total


def init_random_at_composition(d, c_target, rng):
    """±1 int64 array of length d with exactly round(c_target * d) +1 sites."""
    n_plus = max(0, min(d, int(round(c_target * d))))
    x = -np.ones(d, dtype=np.int64)
    idx = rng.choice(d, size=n_plus, replace=False)
    x[idx] = 1
    return x


@njit(cache=True)
def run_chain(x, D, sigma, n_steps, seed):
    """Non-local Kawasaki chain. Returns (energy_trace, x_final, n_accept).

    energy_trace[t] = log_prob_ising after step t (units for τ_int are single
    swap-attempt steps). Maintains plus/minus index arrays so each step draws a
    +site and a -site in O(1); on accept the chosen indices switch arrays.
    """
    np.random.seed(seed)
    d = D * D
    plus = np.empty(d, dtype=np.int64)
    minus = np.empty(d, dtype=np.int64)
    n_plus = 0
    n_minus = 0
    for k in range(d):
        if x[k] == 1:
            plus[n_plus] = k
            n_plus += 1
        else:
            minus[n_minus] = k
            n_minus += 1

    L = initial_log_prob_ising(x, D, sigma)
    energy_trace = np.empty(n_steps, dtype=np.float64)
    n_accept = 0

    for step in range(n_steps):
        pi = np.random.randint(n_plus)
        mi = np.random.randint(n_minus)
        i = plus[pi]
        j = minus[mi]
        delta = kawasaki_delta_log_prob(x, i, j, D, sigma)
        if delta >= 0.0 or np.random.random() < np.exp(delta):
            x[i] = -1
            x[j] = 1
            plus[pi] = j
            minus[mi] = i
            L += delta
            n_accept += 1
        energy_trace[step] = L

    return energy_trace, x, n_accept
