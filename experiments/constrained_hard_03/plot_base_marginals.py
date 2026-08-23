"""Base-vs-target marginals of the drive's sufficient statistic, one panel
per chapter (2026-08-23).

The three path identities (soft chapter, eq:matched-drive/weight/difficulty/
trade) say the base lowers the path's difficulty only through the drive
g = log rho~ - log eta, and g is a function of ONE scalar statistic in each
chapter. The figure draws that statistic's marginal under each base and
under the target, so the gap the flow must close is visible on one axis.

  soft  (D=10, sigma=0.10, c_t=0.80, lambda=50): statistic = composition c.
        Bases: uniform = Binomial(d, 1/2); matched = Binomial(d, 0.80).
        Target marginal p(c) ∝ C(d,N) E_unif-slice[exp(sigma x^T A x)]
        exp(-lambda d (c - c_t)^2), the slice expectation estimated by
        sampling the uniform slice (sigma = 0.1 is weak coupling, so the
        estimator is low-variance; n_slice draws per N).
  hard  (D=8, sigma_c=0.223, c=1/2): statistic = energy per site
        e = -x^T A x / (2 d) (bonds counted once).  Bases: uniform on the
        slice (exact), block-occupancy B(2, w), K=4 (exact, w fitted to the
        reference draws).  Target: the certified Kawasaki reference chains.

Output: assets/soft_base_composition.png and assets/hard_base_energy.png
under --out-dir (the Overleaf assets directory).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom

from discrete_flow_sampler.diagnostics.figure_style import (
    ANALYTIC_GUIDE, MUTED, NEURAL_COMPARATOR_HUE, REFERENCE_INK, SAMPLER_HUE,
    style_axes, use_house_style,
)
from experiments.constrained_hard_03.block_occupancy_base import (
    BlockOccupancyBase, UniformSliceBase, quadratic_form, torus_adjacency,
)

REFERENCE_DIR = Path("results/kawasaki_probe/reference/sc")


# ---------------------------------------------------------------- soft panel
def soft_target_composition_marginal(d, sigma, c_target, penalty, rng, n_slice=20_000):
    """log p(c) up to a constant for every N = 0..d, via the slice identity
    Z_slice(N) = C(d, N) E_unif-slice[exp(sigma x^T A x)]."""
    side = int(round(d ** 0.5))
    A = torus_adjacency(side)
    log_marginal = np.empty(d + 1)
    for n_up in range(d + 1):
        log_binomial = gammaln(d + 1) - gammaln(n_up + 1) - gammaln(d - n_up + 1)
        if n_up in (0, d):
            log_slice_mean = sigma * quadratic_form(np.ones((1, d)), A)[0]
        else:
            draws = UniformSliceBase(side, n_up).sample(n_slice, rng)
            log_slice_mean = logsumexp(sigma * quadratic_form(draws, A)) - np.log(n_slice)
        composition = n_up / d
        log_marginal[n_up] = (
            log_binomial + log_slice_mean - penalty * d * (composition - c_target) ** 2
        )
    return np.exp(log_marginal - logsumexp(log_marginal))


def soft_panel(ax, d=100, sigma=0.10, c_target=0.80, penalty=50.0, seed=0):
    rng = np.random.default_rng(seed)
    compositions = np.arange(d + 1) / d
    uniform = binom.pmf(np.arange(d + 1), d, 0.5)
    matched = binom.pmf(np.arange(d + 1), d, c_target)
    target = soft_target_composition_marginal(d, sigma, c_target, penalty, rng)
    ax.plot(compositions, uniform, color=SAMPLER_HUE, lw=1.6, label="uniform base")
    ax.plot(compositions, matched, color=NEURAL_COMPARATOR_HUE, lw=1.6, label="matched base")
    ax.plot(compositions, target, color=REFERENCE_INK, lw=1.6, label="target")
    ax.axvline(c_target, color=ANALYTIC_GUIDE, lw=0.8, ls="--")
    ax.set_xlabel("composition $c(x)$")
    ax.set_ylabel("probability")
    ax.set_xlim(0.3, 1.0)
    return {"uniform": uniform, "matched": matched, "target": target}


# ---------------------------------------------------------------- hard panel
def load_reference(thin=28):
    spins = [np.load(p)["spins"][::thin] for p in sorted(REFERENCE_DIR.glob("chain_*/snapshots.npz"))]
    return np.concatenate(spins).astype(np.int8)


def energy_per_site(spins, A):
    return -quadratic_form(spins, A) / (2 * spins.shape[1])


def hard_panel(ax, side=8, n_draws=200_000, seed=0):
    rng = np.random.default_rng(seed)
    A = torus_adjacency(side)
    reference = load_reference()
    n_up = int((reference[0] > 0).sum())
    uniform = UniformSliceBase(side, n_up).sample(n_draws, rng)
    warm = BlockOccupancyBase.fit(reference, side, 2, n_offsets=4).sample(n_draws, rng)
    series = [
        ("uniform base", energy_per_site(uniform, A), SAMPLER_HUE),
        ("block base $B(2,w)$", energy_per_site(warm, A), NEURAL_COMPARATOR_HUE),
        ("target (Kawasaki reference)", energy_per_site(reference, A), REFERENCE_INK),
    ]
    # On the even-sided torus the attainable energies form a lattice of
    # spacing 4/d (every swap or flip changes the bond sum by a multiple of
    # 4); draw the exact pmf on those levels as a curve (the soft panel's
    # grammar) rather than a histogram whose bins would alias it.
    d = side * side
    levels = np.arange(-2.0, 0.5 + 1e-9, 4 / d)
    for label, values, hue in series:
        index = np.rint((values + 2.0) * d / 4).astype(int)
        pmf = np.bincount(index, minlength=len(levels))[: len(levels)] / len(values)
        ax.plot(levels, pmf, color=hue, lw=1.6, label=label)
    ax.set_xlabel("energy per site $E(x)/d$")
    ax.set_ylabel("probability")
    ax.set_xlim(-1.5, 0.45)
    return {label: values for label, values, _ in series}


def _finish(ax):
    style_axes(ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)  # headroom for the legend
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=7.5,
              handlelength=1.4, columnspacing=1.0)
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

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    hard = hard_panel(ax, seed=args.seed)
    _finish(ax)
    fig.savefig(out / "hard_base_energy.png")
    plt.close(fig)

    # Numbers for the captions: where each law sits.
    c = np.arange(101) / 100
    for name, p in soft.items():
        print(f"soft {name:8s} mean c {np.sum(c * p):.3f}")
    for name, e in hard.items():
        print(f"hard {name:28s} mean E/d {e.mean():+.3f}  std {e.std():.3f}")


if __name__ == "__main__":
    main()
