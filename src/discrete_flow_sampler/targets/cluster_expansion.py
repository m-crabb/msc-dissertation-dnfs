"""Cluster-expansion alloy targets: a real materials energy on every rung.

WHY THIS EXISTS. The three chapters run the ensemble ladder (free, penalised,
fixed composition) on the nearest-neighbour Ising model read as a binary
alloy. This module lets the same samplers run on a real cluster expansion --
the MetaDNS Cu-Au expansion on its 64-site fcc cell, or the square-grid toy in
icet-ce/ -- so the ladder can be shown on an alloy a materials reader
recognises, with mchammer as the reference on the same energy.

THE ENERGY. On a fixed periodic cell with two species, any cluster expansion
is exactly a polynomial in spins s_i in {-1, +1} (Au = +1, Cu = -1):

    E(s) = J_0 + sum_k c_k sum_{tuples T in class k} prod_{i in T} s_i,           (1)

with the tuple lists and coefficients exported once by
`experiments/alloy_ce/export_binary_expansion.py`, which fits (1) to the
library's own energies and refuses to write unless the residual is at
floating-point precision. The Boltzmann target is p(s) ∝ exp[-beta E(s)]; to
keep every downstream estimator unchanged the project's beta = 2 sigma
convention is kept, so `sigma` here is beta/2 and
`free_energy_lb_estimate(log_w, sigma, d)` returns F/d in the expansion's own
energy units (eV per site for the alloy files).

CLOSED-FORM MOVES. Flipping site i negates every product that contains i, so

    Delta E_flip(i)   = -2 E_i,           E_i  := sum_{T ∋ i} c_T prod_T s,         (2)
    Delta E_swap(i,j) = -2 E_i - 2 E_j + 4 E_ij,   E_ij := sum_{T ∋ i,j} c_T prod_T s, (3)

for s_i != s_j (a like-spin swap is the identity, so it gets 0): flipping both
negates the products containing exactly one of the two sites and leaves the
ones containing both alone. For the pair-only Ising expansion (3) is the
existing Kawasaki closed form in `FixedCompositionIsingTarget.swap_log_ratio`,
which is what the parity test pins. Cost is one gather-product-scatter per
tuple class, O(B * n_tuples), with no (B, d, d, d) or (B, P, d) materialisation
-- the reason the swap sampler could not simply call the library.

FAILURE MODES GUARDED. Wrong species sign or basis convention: absorbed by the
export's fit, and caught by the reference-energy test. Image multiplicity
(a shell at exactly half the cell): the tuple lists carry the duplicate on
purpose and `index_add_` accumulates it. Slice constant: the fixed-composition
class reuses the Ising slice methods verbatim, so log C(d, n_plus) enters the
path weight exactly as it does for Ising.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)


@dataclass
class BinaryExpansionSpec:
    """The exported expansion (1): tuple classes, coefficients, cell metadata."""

    n_sites: int
    constant: float
    terms: list[dict]
    nearest_neighbour_pairs: list
    positions: list | None = None
    cell: list | None = None
    source: str = ""
    energy_units: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "BinaryExpansionSpec":
        raw = json.loads(Path(path).read_text())
        return cls(
            n_sites=raw["n_sites"], constant=raw["constant"], terms=raw["terms"],
            nearest_neighbour_pairs=raw["nearest_neighbour_pairs"],
            positions=raw.get("positions"), cell=raw.get("cell"),
            source=raw.get("source", ""), energy_units=raw.get("energy_units", ""),
        )

    def _tensors(self, device) -> list[tuple[Tensor, float]]:
        """Per class: (tuples (n_k, order_k) long, coefficient), cached per device."""
        key = str(device)
        if key not in self._cache:
            self._cache[key] = [
                (torch.tensor(term["tuples"], dtype=torch.long, device=device),
                 float(term["coefficient"]))
                for term in self.terms
            ]
        return self._cache[key]

    def _class_products(self, x: Tensor):
        """Yield (tuples, c * prod_T s) for every class, prod shape (B, n_k)."""
        for tuples, coefficient in self._tensors(x.device):
            yield tuples, coefficient * x[:, tuples].prod(dim=-1)

    def energy(self, x: Tensor) -> Tensor:
        """E(s) of (1), shape (B,), in x.dtype."""
        total = torch.full((x.shape[0],), self.constant, dtype=x.dtype, device=x.device)
        for _, products in self._class_products(x):
            total = total + products.sum(dim=-1)
        return total

    def site_energies(self, x: Tensor) -> Tensor:
        """E_i of (2): the summed weight of every tuple containing site i, (B, d)."""
        site = torch.zeros(x.shape[0], self.n_sites, dtype=x.dtype, device=x.device)
        for tuples, products in self._class_products(x):
            for position in range(tuples.shape[1]):
                site.index_add_(1, tuples[:, position], products)
        return site

    def pair_energies(self, x: Tensor) -> Tensor:
        """E_ij of (3): the summed weight of tuples containing both i and j, (B, d, d)."""
        d = self.n_sites
        flat = torch.zeros(x.shape[0], d * d, dtype=x.dtype, device=x.device)
        for tuples, products in self._class_products(x):
            for a, b in itertools.combinations(range(tuples.shape[1]), 2):
                flat.index_add_(1, tuples[:, a] * d + tuples[:, b], products)
                flat.index_add_(1, tuples[:, b] * d + tuples[:, a], products)
        return flat.view(x.shape[0], d, d)

    def flip_energy_change(self, x: Tensor) -> Tensor:
        """Delta E for flipping each site, (B, d), Eq. (2)."""
        return -2.0 * self.site_energies(x)

    def swap_energy_change(self, x: Tensor) -> Tensor:
        """Delta E for swapping each unlike pair, (B, d, d), Eq. (3); 0 for like pairs."""
        site = self.site_energies(x)
        delta = -2.0 * site[:, :, None] - 2.0 * site[:, None, :] + 4.0 * self.pair_energies(x)
        unlike = (x[:, :, None] != x[:, None, :]).to(x.dtype)
        return delta * unlike

    def nn_adjacency(self, device="cpu") -> Tensor:
        """Symmetric nearest-neighbour count matrix, the `A` the diagnostics read."""
        A = torch.zeros(self.n_sites, self.n_sites, device=device)
        for i, j in self.nearest_neighbour_pairs:
            A[i, j] += 1.0
            A[j, i] += 1.0
        return A


class ClusterExpansionTarget(IsingTarget):
    """p(s) ∝ exp[-beta E(s) + bias sum s] for an exported expansion.

    Subclasses IsingTarget so the soft-composition penalty, the annealing path,
    the matched base and the generic swap fallback are inherited unchanged; only
    the energy is replaced. `sigma` = beta/2 (see the module docstring), and
    `set_sigma` is the temperature curriculum: sigma -> beta/2 at the new T.
    `D` is a label only (the torus loop is bypassed by passing the adjacency).
    """

    def __init__(
        self,
        spec: BinaryExpansionSpec,
        beta: float,
        bias: float = 0.0,
        device: torch.device | str = "cpu",
        **ising_kwargs,
    ):
        self.spec = spec
        side = math.isqrt(spec.n_sites)
        super().__init__(
            D=side if side * side == spec.n_sites else spec.n_sites,
            sigma=beta / 2.0, bias=bias, device=device,
            adjacency=spec.nn_adjacency(device), **ising_kwargs,
        )

    @property
    def beta(self) -> float:
        return 2.0 * self.sigma

    def base_log_prob(self, x: Tensor) -> Tensor:
        """-beta E(s) + bias * sum_i s_i, shape (B,)."""
        return -self.beta * self.spec.energy(x) + self.bias * x.sum(dim=1)


class FixedCompositionClusterExpansionTarget(ClusterExpansionTarget):
    """The canonical rung on an expansion: uniform slice base, swap moves only.

    The slice machinery is FixedCompositionIsingTarget's verbatim (same base
    draw, same -log C(d, n_plus) constant, same manifold check); only the swap
    ratio differs, using Eq. (3) in place of the Ising quadratic form.
    """

    sample_base = FixedCompositionIsingTarget.sample_base
    base_log_eta = FixedCompositionIsingTarget.base_log_eta
    assert_on_manifold = FixedCompositionIsingTarget.assert_on_manifold

    def __init__(self, spec, beta, target_composition, bias=0.0, device="cpu"):
        n_plus_float = target_composition * spec.n_sites
        n_plus_target = round(n_plus_float)
        if abs(n_plus_float - n_plus_target) > 1e-9:
            raise ValueError(
                f"target_composition={target_composition} * d={spec.n_sites} = "
                f"{n_plus_float} is not integral; no exact fixed-N slice exists."
            )
        super().__init__(
            spec, beta, bias=bias, device=device,
            target_composition=target_composition, composition_penalty_strength=0.0,
        )
        self.n_plus_target = n_plus_target
        self.composition_quantum = self.d
        self._log_slice_size = (
            math.lgamma(self.d + 1)
            - math.lgamma(n_plus_target + 1)
            - math.lgamma(self.d - n_plus_target + 1)
        )

    def swap_log_ratio(self, x: Tensor, t: Tensor, pairs: Tensor) -> Tensor:
        """t * (-beta) * Delta E_swap(i, j) for each listed pair, shape (B, P).

        On the slice base_log_eta is constant and bias * sum s is swap-
        invariant, so only the t-weighted energy change survives, exactly as
        in the Ising closed form.
        """
        delta = self.spec.swap_energy_change(x)
        return t[:, None] * (-self.beta) * delta[:, pairs[:, 0], pairs[:, 1]]
