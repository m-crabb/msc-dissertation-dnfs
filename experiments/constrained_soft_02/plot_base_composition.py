"""Base-vs-target composition marginal for the soft chapter.

The three path identities (eq:matched-drive/weight/difficulty/trade) say the
base lowers the path's difficulty only through the drive g = log rho~ -
log eta, and for the soft target g is a function of the composition c(x)
alone. The figure draws c's marginal under each base and under the target,
so the gap the flow must close is visible on one axis.

  D=10, sigma=0.10, c_t=0.80, lambda=50 (the stress window).
  Bases: uniform = Binomial(d, 1/2); matched = Binomial(d, 0.80).
  Target and Ising-only marginals by Metropolis single-flip chains
  (256 chains x 3000 post-burn-in sweeps). The unpenalised Ising marginal
  is drawn as a guide so the penalty's role is visible: it, not the
  physics, moves the target.

Two features of the soft-target (ink) curve are REAL, not plotting
artefacts; both were checked 2026-08-29 against the pmf this script builds.

  1. Its peak sits LEFT of the c_t = 0.80 guide line. The soft marginal is
     p(c) ~ Z_can(c) * exp(-lambda*d*(c - c_t)^2), and the combinatorial
     factor in Z_can(c) falls steeply on the c > 1/2 side: d/dc of
     log C(d, dc) = d*log((1-c)/c) = -138.6 per unit c at c = 0.8, against
     the penalty curvature 2*lambda*d = 1e4. Balancing the two displaces
     the mode by 138.6/1e4 = 0.0139, i.e. 1.4 lattice sites, and the
     continuum root of log((1-c)/c) = 2*lambda*(c - c_t) is c = 0.7869.
     Measured: mode at N = 79 (c = 0.79, p = 0.3572, against p = 0.3547 at
     c = 0.80), mean c = 0.7949, and the marginal is skewed the same way
     (p(0.78) = 0.1289 > p(0.81) = 0.1251). Entropy pulls the alloy back
     toward half-filling; the penalty alone would centre it on 0.80. It is
     NOT an axis offset: the compositions are the exact support N/d, and
     the matched-base binomial on the same axis peaks exactly on 0.80.
  2. Its top is FLAT rather than rounded. The support spacing is 1/d = 0.01
     and the target's own width is 1/sqrt(2*lambda*d) = 0.01 -- exactly one
     step -- so the peak is carried by two points of nearly equal height
     (0.3572 and 0.3547, 0.7% apart) and the piecewise-linear curve through
     them is flat by construction. Nothing is clipped: the y limit is
     1.6 * peak = 0.57 against a peak of 0.357.

Output: assets/soft_base_composition.png under --out-dir.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from discrete_flow_sampler.diagnostics.figure_style import (
    ANALYTIC_GUIDE,
    NEURAL_COMPARATOR_HUE,
    REFERENCE_INK,
    SAMPLER_HUE,
    style_axes,
    use_house_style,
)


def _neighbour_field(spins, side):
    """Sum of the four torus neighbours, (chains, d)."""
    grid = spins.reshape(-1, side, side)
    field = (
        np.roll(grid, 1, 1)
        + np.roll(grid, -1, 1)
        + np.roll(grid, 1, 2)
        + np.roll(grid, -1, 2)
    )
    return field.reshape(spins.shape)


def soft_target_composition_marginal(
    d, sigma, c_target, penalty, rng, n_chains=256, n_sweeps=4000, burn_in=1000
):
    """Composition marginal of exp(sigma x^T A x - penalty d (c - c_t)^2) by
    Metropolis single-flip chains, histogrammed over chains and sweeps.

    Random-site sequential updates, vectorised ACROSS chains: the penalty
    couples every site through c(x), so a checkerboard sweep (which flips
    half the lattice at once) would not be a valid single-site kernel here.
    At sigma = 0.1 the flip chain mixes in tens of sweeps; burn-in is 1000.
    (The naive slice estimator Z_slice(N) = C(d,N) E_unif[exp(sigma x^T A x)]
    is heavy-tailed -- one rare ordered draw dominates -- and was replaced
    by this, 2026-08-23.)"""
    side = int(round(d**0.5))
    spins = rng.choice([-1, 1], size=(n_chains, d)).astype(np.int8)
    n_up = (spins > 0).sum(1)
    rows = np.arange(n_chains)
    counts = np.zeros(d + 1)
    for sweep in range(n_sweeps):
        for _ in range(d):
            site = rng.integers(d, size=n_chains)
            field = _neighbour_field(spins, side)[rows, site]
            old = spins[rows, site]
            delta_quadratic = -2.0 * old * field  # change in x^T A x / 2 * 2
            new_up = n_up - old  # -1 -> +1 adds one up-spin
            c_old, c_new = n_up / d, new_up / d
            delta_penalty = (
                penalty * d * ((c_new - c_target) ** 2 - (c_old - c_target) ** 2)
            )
            log_accept = 2.0 * sigma * delta_quadratic - delta_penalty
            accept = np.log(rng.random(n_chains)) < log_accept
            spins[rows[accept], site[accept]] = -old[accept]
            n_up = np.where(accept, new_up, n_up)
        if sweep >= burn_in:
            counts += np.bincount(n_up, minlength=d + 1)
    return counts / counts.sum()


def soft_panel(ax, d=100, sigma=0.10, c_target=0.80, penalty=50.0, seed=0):
    rng = np.random.default_rng(seed)
    compositions = np.arange(d + 1) / d
    uniform = binom.pmf(np.arange(d + 1), d, 0.5)
    matched = binom.pmf(np.arange(d + 1), d, c_target)
    target = soft_target_composition_marginal(d, sigma, c_target, penalty, rng)
    ising_only = soft_target_composition_marginal(d, sigma, c_target, 0.0, rng)
    ax.plot(
        compositions,
        ising_only,
        color=ANALYTIC_GUIDE,
        lw=1.2,
        ls="--",
        label="Ising, no penalty",
    )
    ax.plot(compositions, uniform, color=SAMPLER_HUE, lw=1.6, label="uniform base")
    ax.plot(
        compositions, matched, color=NEURAL_COMPARATOR_HUE, lw=1.6, label="matched base"
    )
    ax.plot(compositions, target, color=REFERENCE_INK, lw=1.6, label="soft target")
    ax.axvline(c_target, color=ANALYTIC_GUIDE, lw=0.8, ls="--")
    ax.set_xlabel("composition $c(x)$")
    ax.set_ylabel("probability")
    ax.set_xlim(0.3, 1.0)
    return {
        "ising": ising_only,
        "uniform": uniform,
        "matched": matched,
        "target": target,
    }


def _finish(ax):
    style_axes(ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)  # headroom for the legend
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=2,
        fontsize=7.5,
        handlelength=1.4,
        columnspacing=1.0,
    )
    ax.figure.tight_layout()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="assets")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    use_house_style()

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    soft = soft_panel(ax, seed=args.seed)
    _finish(ax)
    fig.savefig(out / "soft_base_composition.png")
    plt.close(fig)

    # Numbers for the captions: where each law sits. The mode is printed
    # alongside the mean because for the soft target the two disagree, and
    # that disagreement is the entropic pull documented at the top of this
    # file -- the ink curve peaking left of c_t is the physics, not a bug.
    c = np.arange(101) / 100
    for name, p in soft.items():
        print(
            f"soft {name:8s} mean c {np.sum(c * p):.3f}  "
            f"mode c {c[p.argmax()]:.2f} (p {p.max():.4f})"
        )


if __name__ == "__main__":
    main()
