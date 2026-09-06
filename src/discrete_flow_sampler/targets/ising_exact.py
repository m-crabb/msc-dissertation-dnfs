"""Exact finite-torus 2D Ising thermodynamics (Kaufman / Ferdinand--Fisher).

Kaufman (1949) closed the partition function of the finite m x n torus that
Onsager solved in the thermodynamic limit; Ferdinand & Fisher (1969) used it
for finite-size analysis, and DNFS Table 2 prints its per-site values as the
"Optimal Value" rows the baseline replication is measured against. This
module computes those values so the thesis carries its own exact reference
instead of citing the paper's numbers.

The formula, with K the PER-BOND coupling beta*J (this repo's convention has
log p = x^T J x double-counting each edge, so K = 2 * sigma):

    Z = (1/2) (2 sinh 2K)^{mn/2} (Z1 + Z2 + Z3 + Z4)

    Z1 = prod_{r=0}^{n-1} 2 cosh(m gamma_{2r+1} / 2)
    Z2 = prod_{r=0}^{n-1} 2 sinh(m gamma_{2r+1} / 2)
    Z3 = prod_{r=0}^{n-1} 2 cosh(m gamma_{2r}   / 2)
    Z4 = prod_{r=0}^{n-1} 2 sinh(m gamma_{2r}   / 2)

    cosh gamma_l = cosh 2K coth 2K - cos(pi l / n)   (gamma_l >= 0, l >= 1)
    gamma_0      = 2K + ln tanh K                    (SIGNED)

gamma_0 is the only signed angle and it changes sign exactly at criticality
(tanh K_c = e^{-2K_c}); below K_c the Z4 product is negative, at K_c it is
zero, above it is positive. Mishandling that sign is the classic bug, so the
four products are accumulated as (log magnitude, sign) pairs and combined by
a signed logsumexp. Everything is done in logs: at 10x10, log Z ~ 73, and
the naive products overflow float64 at modest sizes.

The internal energy is -d(log Z)/dK. Rather than differentiating the angles
analytically (twice the code for no accuracy the tests can see), it is a
5-point central difference on the closed form: truncation O(h^4) and
float64 cancellation balance near 1e-9 at h = 1e-4 * max(1, K), validated
against exact enumeration at 4x4 to 1e-6 in tests/test_ising_exact.py.
"""

import math


def _log_2cosh(x: float) -> float:
    """log(2 cosh x), stable for large |x|."""
    ax = abs(x)
    return ax + math.log1p(math.exp(-2.0 * ax))


def _log_abs_2sinh(x: float) -> tuple[float, float]:
    """(log |2 sinh x|, sign), stable for large |x|; sign 0 at x = 0."""
    if x == 0.0:
        return float("-inf"), 0.0
    ax = abs(x)
    return ax + math.log1p(-math.exp(-2.0 * ax)), math.copysign(1.0, x)


def log_partition_torus(n_rows: int, n_cols: int, bond_coupling: float) -> float:
    """log Z of the n_rows x n_cols periodic Ising lattice at per-bond
    coupling K = bond_coupling (ferromagnetic, K > 0, zero field).

    beta*H = -K * sum over each unordered torus edge of s_i s_j, so
    Z = sum_x exp(K * pair_sum(x)) — the convention the enumeration test
    pins directly.
    """
    if bond_coupling <= 0:
        raise ValueError("ferromagnetic coupling required: bond_coupling > 0")
    m, n, K = n_rows, n_cols, bond_coupling

    cosh_term = math.cosh(2 * K) / math.tanh(2 * K)  # cosh 2K coth 2K

    def gamma(l: int) -> float:
        if l == 0:
            return 2 * K + math.log(math.tanh(K))  # signed
        return math.acosh(cosh_term - math.cos(math.pi * l / n))

    odd_angles = [gamma(2 * r + 1) for r in range(n)]
    even_angles = [gamma(2 * r) for r in range(n)]

    log_z1 = sum(_log_2cosh(m * g / 2) for g in odd_angles)
    log_z3 = sum(_log_2cosh(m * g / 2) for g in even_angles)

    def signed_log_prod_sinh(angles):
        log_mag, sign = 0.0, 1.0
        for g in angles:
            piece_log, piece_sign = _log_abs_2sinh(m * g / 2)
            log_mag += piece_log
            sign *= piece_sign
        return log_mag, sign

    log_z2, sign_z2 = signed_log_prod_sinh(odd_angles)  # always +1
    log_z4, sign_z4 = signed_log_prod_sinh(even_angles)  # sign of gamma_0

    # Signed logsumexp of the four products (the total is always positive).
    terms = [(log_z1, 1.0), (log_z2, sign_z2), (log_z3, 1.0), (log_z4, sign_z4)]
    peak = max(log_mag for log_mag, sign in terms if sign != 0.0)
    total = sum(sign * math.exp(log_mag - peak) for log_mag, sign in terms)
    log_sum = peak + math.log(total)

    return -math.log(2.0) + (m * n / 2) * math.log(2 * math.sinh(2 * K)) + log_sum


def ferdinand_fisher_per_site(D: int, sigma: float) -> dict[str, float]:
    """Exact per-site F, E, S for the D x D torus at the repo's sigma.

    Units are DNFS Table 2's (energy in units of J, k_B = 1), under the
    double-counted convention K = 2 * sigma:

        F/D^2 = -log Z / (K D^2)
        E/D^2 = -d(log Z)/dK / D^2
        S/D^2 =  K (E - F) / D^2      [their S = 2 sigma (E - F)]
    """
    K = 2.0 * sigma
    n_sites = D * D
    log_z = log_partition_torus(D, D, K)

    h = 1e-4 * max(1.0, K)
    stencil = (
        -log_partition_torus(D, D, K + 2 * h)
        + 8 * log_partition_torus(D, D, K + h)
        - 8 * log_partition_torus(D, D, K - h)
        + log_partition_torus(D, D, K - 2 * h)
    ) / (12 * h)

    free_energy = -log_z / (K * n_sites)
    internal_energy = -stencil / n_sites
    entropy = K * (internal_energy - free_energy)
    return {
        "free_energy": free_energy,
        "internal_energy": internal_energy,
        "entropy": entropy,
    }
