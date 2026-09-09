"""Wolff single-cluster sampler for the unconstrained torus Ising model.

Ground-truth cross-check for the baseline chapter's Gibbs reference (Wolff
1989). At criticality single-site dynamics has dynamic exponent z ~ 2.17
while Wolff sits near 0.25, so a Wolff pool decorrelates ~L^2 faster per
sweep-equivalent; its agreement certifies the Gibbs pool. Same cluster
family as the Swendsen--Wang ground truths of MDNS and DASBS (one seeded
cluster per move, no lattice-wide bond percolation).

The move: pick a random seed site, grow a cluster over aligned neighbours,
adding each tried bond with

    p_add = 1 - exp(-2 K) = 1 - exp(-4 sigma)

(K = 2 sigma is the per-bond coupling under this repo's double-counted
`log p = x^T J x` convention — using 1 - exp(-2 sigma) here is the silent
bug the 3x3 enumeration test exists to catch), then flip the whole cluster.
Every bond on the cluster boundary was rejected with probability exp(-2K)
during growth, which is exactly the factor detailed balance needs, so the
flip is accepted with probability one. The construction requires zero
external field: a bias term breaks the up/down symmetry of the cluster flip
and there is no cheap correction, hence the hard ValueError.

Ergodicity: a cluster can be a single site (all four bonds rejected), which
is a single-spin flip, so every configuration is reachable. Cluster flips
change M by +-2|C|, so the chain tunnels freely between the Z2 sectors that
trap single-site dynamics at sigma_c — the failure mode under audit.

Growth is breadth-first with the whole frontier processed per step: each
frontier site tries its 4 torus neighbours once (numpy-vectorised), aligned
non-member neighbours join with p_add. A site rejected through one bond can
be re-tried later through another, which is the correct Wolff rule (each
bond is tried at most once, sites may be offered repeatedly).
"""

import numpy as np
import torch
from torch import Tensor


def wolff_sample(
    target,
    n_samples: int,
    clusters_per_sample: int = 20,
    burn_in_clusters: int = 500,
    seed: int = 0,
    cluster_size_log: list | None = None,
) -> Tensor:
    """Draw n_samples configurations from an unconstrained IsingTarget.

    cluster_size_log: pass a list to have every flipped cluster's site count
      appended (burn-in included, as part of the sampling price). The RNG
      stream is untouched, so the same seed yields bit-identical samples
      with or without the log; diagnostics/flops.py::wolff_run_flops relies
      on this to FLOP-count certified pools without rebuilding them.

    target: duck-types IsingTarget — needs `.D`, `.sigma`, `.bias`, `.d`.
      Must have bias == 0 (see module docstring).
    clusters_per_sample: cluster flips between recorded samples. At the
      operating sigma_c a critical cluster spans an O(1) fraction of the
      lattice, so tens of clusters decorrelate a 10x10 configuration; the
      default is deliberately generous for reference-pool use.
    Returns: (n_samples, D*D) float tensor in {-1, +1} on CPU, matching the
      layout of `gibbs_sample` so downstream observable code is shared.
    """
    if getattr(target, "bias", 0.0) != 0.0:
        raise ValueError(
            "Wolff requires zero external field: the cluster flip is only "
            f"detailed-balanced at bias = 0 (target has bias={target.bias})"
        )
    D = target.D
    n_sites = D * D
    p_add = 1.0 - np.exp(-4.0 * target.sigma)

    # Torus neighbour table, (n_sites, 4): right, left, down, up.
    site = np.arange(n_sites)
    row, col = site // D, site % D
    neighbours = np.stack(
        [
            row * D + (col + 1) % D,
            row * D + (col - 1) % D,
            ((row + 1) % D) * D + col,
            ((row - 1) % D) * D + col,
        ],
        axis=1,
    )

    rng = np.random.default_rng(seed)
    spins = rng.integers(0, 2, n_sites).astype(np.int8) * 2 - 1
    in_cluster = np.zeros(n_sites, dtype=bool)

    def flip_one_cluster() -> None:
        """Grow one Wolff cluster from a random seed site and flip it.

        Breadth-first over the torus neighbour table: each aligned, not yet
        included neighbour joins with probability p_add = 1 - exp(-4 sigma),
        which makes the cluster flip rejection-free at zero field.
        """
        seed_site = rng.integers(n_sites)
        cluster_spin = spins[seed_site]
        in_cluster[:] = False
        in_cluster[seed_site] = True
        frontier = np.array([seed_site])
        while frontier.size:
            candidates = neighbours[frontier].ravel()
            tried = candidates[
                (spins[candidates] == cluster_spin) & ~in_cluster[candidates]
            ]
            accepted = tried[rng.random(tried.size) < p_add]
            # A site accepted through two bonds at once appears twice.
            frontier = np.unique(accepted)
            in_cluster[frontier] = True
        spins[in_cluster] *= -1
        if cluster_size_log is not None:
            cluster_size_log.append(int(in_cluster.sum()))

    for _ in range(burn_in_clusters):
        flip_one_cluster()

    samples = np.empty((n_samples, n_sites), dtype=np.int8)
    for sample_index in range(n_samples):
        for _ in range(clusters_per_sample):
            flip_one_cluster()
        samples[sample_index] = spins

    return torch.from_numpy(samples.astype(np.float32))
