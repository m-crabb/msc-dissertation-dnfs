"""Kawasaki (composition-preserving swap) MCMC for the hard-constrained Ising
canonical ensemble.

Two move sets live here. run_chain / run_chain_order_param exchange the spins
of one +1 site and one -1 site chosen uniformly at random anywhere on the
torus (non-local swap — the practitioner-standard canonical-ensemble move, and
a deliberately strong baseline). run_local_swap_chain_snapshots swaps a
uniform random nearest-neighbour bond instead (local Kawasaki — the textbook
conserved-order-parameter dynamics, and the mixing probe's slow competitor).
Either way composition c(x) = #{+1}/d is invariant by construction: there is
no penalty term, and the hard constraint c(x) = c_target holds exactly at
every step. This is the §3.1 counterpart to the soft VCSGC sampler in
scripts/vcsgc_mcmc_validation.py.

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
except ImportError:  # pragma: no cover
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
    return (
        j == ((r - 1) % D) * D + c
        or j == ((r + 1) % D) * D + c
        or j == r * D + (c - 1) % D
        or j == r * D + (c + 1) % D
    )


@njit(cache=True)
def kawasaki_delta_log_prob(x, i, j, D, sigma):
    """Δlog_prob_ising for swapping opposite-spin sites i and j."""
    diff = x[j] - x[i]  # = x_i' - x_i = -(x_j' - x_j)
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


@njit(cache=True)
def left_minus_right(x, D):
    """Mode-sensitive order parameter: mean(x | left-half columns) minus
    mean(x | right-half columns) on a DxD lattice.

    Labels WHICH way the system has phase-separated at fixed composition:
    +domain on the left → ~ +2, on the right → ~ -2, symmetric/mixed → ~ 0.
    NOTE `diagnostics.metrics.half_magnetisation_order_parameter` is the same
    observable scaled by 1/2 (phi in [-1, 1]) — the two figure families are
    on different axes scales.
    Unlike energy, this distinguishes spatial modes, so its between-chain R̂
    detects ergodicity breaking (the §3.1 mode-coverage failure).
    """
    half = D // 2
    left_sum = 0.0
    right_sum = 0.0
    n_left = 0
    n_right = 0
    for i in range(D * D):
        if i % D < half:
            left_sum += x[i]
            n_left += 1
        else:
            right_sum += x[i]
            n_right += 1
    return left_sum / n_left - right_sum / n_right


def init_phase_separated(D, side):
    """c=0.5 phase-separated config: all +1 in the left (side=0) or right
    (side=1) half-columns, -1 elsewhere. For even D this is exactly d/2 +1
    sites. Used to seed chains in DIFFERENT modes for the ergodicity test."""
    if D % 2 != 0:
        raise ValueError("init_phase_separated assumes even D for c=0.5")
    d = D * D
    half = D // 2
    x = -np.ones(d, dtype=np.int64)
    for i in range(d):
        in_left = (i % D) < half
        if (side == 0 and in_left) or (side == 1 and not in_left):
            x[i] = 1
    return x


@njit(cache=True)
def neighbour_site(i, direction, D):
    """Index of the up/down/left/right (direction 0-3) torus neighbour of i."""
    r = i // D
    c = i % D
    if direction == 0:
        return ((r - 1) % D) * D + c
    if direction == 1:
        return ((r + 1) % D) * D + c
    if direction == 2:
        return r * D + (c - 1) % D
    return r * D + (c + 1) % D


@njit(cache=True)
def run_local_swap_chain_snapshots(x, D, sigma, n_steps, seed, thin):
    """LOCAL nearest-neighbour-swap Kawasaki chain recording full int8 spin
    snapshots every `thin` proposals. Returns (snapshots, x_final, n_accept).

    Move set — deliberately DIFFERENT from run_chain / run_chain_order_param,
    which swap arbitrary unlike pairs (non-local Kawasaki, the mchammer
    CanonicalEnsemble move): here a site i is drawn uniformly, then one of its
    4 torus neighbours j uniformly, i.e. a uniform random directed NN bond.
    Local composition-conserving dynamics transports magnetisation
    diffusively, so domain coarsening and mode traversal near criticality are
    drastically slower than under non-local swaps — that gap is the quantity
    the mixing probe measures, so silently reusing the non-local move here
    would erase the phenomenon under study.

    Trial-step currency: EVERY proposal costs one step, including like-spin
    bonds where the swap is the identity (the sampler cannot know a bond is
    like-spin without touching it — this is the standard local-Kawasaki
    accounting). Identity proposals are counted as rejections; the rejected
    alternative (proposing only unlike bonds) needs a live unlike-bond list,
    which is a different, rejection-free algorithm with a different currency.
    Unlike-pair swaps use the same closed-form Metropolis delta as the
    non-local runner (`kawasaki_delta_log_prob`; its shared-bond correction is
    always active here since i ~ j by construction), so acceptance is
    min(1, exp(sigma * Delta(x^T A x))) and composition is invariant exactly.

    snapshots[k] is the state after k*thin proposals — snapshots[0] is the
    initial configuration, matching run_chain_order_param's record-at-top
    convention; the final state is returned separately, not recorded.
    """
    np.random.seed(seed)
    d = D * D
    n_record = n_steps // thin
    snapshots = np.empty((n_record, d), dtype=np.int8)
    rec = 0
    n_accept = 0

    for step in range(n_steps):
        if step % thin == 0 and rec < n_record:
            for k in range(d):
                snapshots[rec, k] = x[k]
            rec += 1
        i = np.random.randint(d)
        j = neighbour_site(i, np.random.randint(4), D)
        if x[i] == x[j]:
            continue  # identity proposal: rejected
        delta = kawasaki_delta_log_prob(x, i, j, D, sigma)
        if delta >= 0.0 or np.random.random() < np.exp(delta):
            spin_i = x[i]
            x[i] = x[j]
            x[j] = spin_i
            n_accept += 1

    return snapshots[:rec], x, n_accept


@njit(cache=True)
def run_nonlocal_swap_chain_snapshots(x, D, sigma, n_steps, seed, thin):
    """NON-local Kawasaki chain recording full int8 spin snapshots every
    `thin` proposals. Returns (snapshots, x_final, n_accept).

    Same move set and accept rule as run_chain (uniform unlike-pair swap
    anywhere on the torus, Metropolis on sigma * x^T A x, plus/minus index
    arrays for O(1) proposals) — only the record differs: full configurations
    rather than the scalar energy, because reference-sample generation needs
    the stored draws themselves, not just a trace. The snapshot convention
    matches run_local_swap_chain_snapshots: snapshots[k] is the state after
    k*thin proposals, so snapshots[0] is the initial configuration and the
    final state is returned separately, not recorded. Passing thin = n_steps
    therefore turns this into a burn-in runner that stores a single snapshot.
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

    n_record = n_steps // thin
    snapshots = np.empty((n_record, d), dtype=np.int8)
    rec = 0
    n_accept = 0

    for step in range(n_steps):
        if step % thin == 0 and rec < n_record:
            for k in range(d):
                snapshots[rec, k] = x[k]
            rec += 1
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
            n_accept += 1

    return snapshots[:rec], x, n_accept


@njit(cache=True)
def run_chain_order_param(x, D, sigma, n_steps, seed, thin):
    """Kawasaki chain recording the left_minus_right order parameter every
    `thin` steps. Returns (phi_trace, x_final, n_accept). Same move and accept
    rule as run_chain; only the recorded observable differs (mode label, not
    energy), so different-init chains' R̂(phi) measures ergodicity breaking."""
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

    n_record = n_steps // thin
    phi_trace = np.empty(n_record, dtype=np.float64)
    rec = 0
    n_accept = 0

    for step in range(n_steps):
        # Record at the top so phi_trace[k] is the state after k*thin steps;
        # phi_trace[0] is the initial config (before any move).
        if step % thin == 0 and rec < n_record:
            phi_trace[rec] = left_minus_right(x, D)
            rec += 1
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
            n_accept += 1

    return phi_trace[:rec], x, n_accept
