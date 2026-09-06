"""Export a fitted cluster expansion as a binary spin-product expansion (JSON).

WHY. The samplers need the alloy energy as a batched torch function with
closed-form flip and swap energy changes, and neither icet nor CLEASE offers
that: both evaluate one configuration at a time through their own orbit
bookkeeping. On a fixed periodic cell with two species, ANY cluster expansion
is exactly a polynomial in spins s_i in {-1, +1},

    E(s) = J_0 + sum_terms J_term * sum_{tuples in term} prod_{i in tuple} s_i,

because every per-site basis function of a binary occupation is affine in s.
So the library is used here only as an ORACLE: the tuples are enumerated
geometrically on the cell, grouped by shape (the multiset of pairwise
distances, which is what a space-group orbit preserves), and the coefficients
are fitted by least squares to oracle energies. A correct enumeration leaves a
residual at floating-point precision; an incorrect one (a missed orbit, a
merged pair of inequivalent shapes) shows up as a residual far above it, so
the fit is its own check. This sidesteps the two libraries' incompatible orbit
representations and any basis-function sign convention.

Periodic images are enumerated explicitly (a tuple is a set of image-sites up
to a global translation), because on a small supercell a cutoff shell can sit
at exactly half the cell (fcc 4th shell at a*sqrt(2) = L/2 on the 4x4x4 cell),
where two images of the same site are both in range and both count.

Sources:
  --source clease  the MetaDNS Cu-Au expansion (CLEASE 1.1.0 format; run in
                   a venv that has clease installed)
  --source icet    the square-grid Cu/Au toy expansion in icet-ce/ (pixi env)

Spin convention in the JSON: s = +1 is species[1] (Au), s = -1 is species[0]
(Cu). The oracle's atomic numbers are mapped accordingly; the fit absorbs any
sign the library uses internally.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

CU, AU = 29, 79


def enumerate_tuples(positions, cell, cutoffs, decimals=4):
    """Site tuples on the periodic cell, grouped by pairwise-distance signature.

    cutoffs: {order: max pairwise distance} for order 2, 3, 4. A tuple is kept
    if every pairwise distance is within the cutoff for its order. Returns
    {(order, signature): [tuple of site indices, ...]}; a tuple may repeat
    when distinct periodic images of the same sites are both in range, and
    that multiplicity is meant (it is the supercell's own counting).
    """
    n = len(positions)
    max_cut = max(cutoffs.values())
    # enough periodic images that every tuple within the cutoff is reachable
    # even when the cutoff exceeds the cell (the 16-site cell is 5.37 A across
    # its short axes against a 6.0 A pair cutoff)
    widths = [
        abs(np.dot(cell[a], np.cross(cell[b], cell[c])))
        / np.linalg.norm(np.cross(cell[b], cell[c]))
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    ]
    reach = int(np.ceil(max_cut / min(widths))) + 1
    image_range = range(-reach, reach + 1)
    images = np.array(list(itertools.product(image_range, repeat=3)), dtype=float)
    shifts = images @ cell  # (n_images, 3)
    # neighbours of each site: (j, image_index, distance)
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        delta = positions[None, :, :] + shifts[:, None, :] - positions[i]  # (27, n, 3)
        dist = np.linalg.norm(delta, axis=-1)
        for img, j in zip(*np.nonzero((dist > 1e-8) & (dist <= max_cut + 1e-8))):
            neighbours[i].append((int(j), int(img), float(dist[img, j])))

    def image_site_position(site, img):
        return positions[site] + shifts[img]

    groups: dict[tuple, list] = {}
    seen: set = set()

    def register(members):
        """members: list of (site, image_index). Canonicalise up to translation."""
        order = len(members)
        pos = np.array([image_site_position(s, g) for s, g in members])
        dists = [
            np.linalg.norm(pos[a] - pos[b])
            for a, b in itertools.combinations(range(order), 2)
        ]
        if max(dists) > cutoffs[order] + 1e-8:
            return
        # canonical key: translate so the lowest-index member sits at image 0
        anchor = min(range(order), key=lambda k: (members[k][0], members[k][1]))
        anchor_shift = images[members[anchor][1]]
        key = tuple(
            sorted(
                (s, tuple((images[g] - anchor_shift).astype(int)))
                for s, g in members
            )
        )
        if key in seen:
            return
        seen.add(key)
        signature = tuple(sorted(round(dd, decimals) for dd in dists))
        # A site can meet its own periodic image inside the cutoff when the
        # cutoff exceeds the cell (the 16-site cell); s_i * s_i = 1, so reduce
        # the tuple by parity -- a site appearing an even number of times drops
        # out -- which keeps the flip/swap closed forms exact. Tuples reduced to
        # nothing are constants and are counted separately.
        counts = {}
        for s, _ in members:
            counts[int(s)] = counts.get(int(s), 0) + 1
        reduced = tuple(sorted(s for s, c in counts.items() if c % 2 == 1))
        groups.setdefault((order, signature), []).append(reduced)

    origin = int(np.flatnonzero(np.all(images == 0, axis=1))[0])
    # the point term: every per-site basis function of a binary occupation is
    # affine in the spin, so the singlet orbit is the column sum_i s_i
    groups[(1, ())] = [(i,) for i in range(n)]
    for i in range(n):
        for k in range(1, 4):
            if k + 1 not in cutoffs:
                continue
            for combo in itertools.combinations(neighbours[i], k):
                register([(i, origin)] + [(j, g) for j, g, _ in combo])
    return groups


def class_sum(spins, tuples):
    """sum over the class of prod_{i in tuple} s_i; tuples may have mixed
    lengths after parity reduction, and an empty tuple contributes 1."""
    total = np.zeros(len(spins))
    for tup in tuples:
        total += np.prod(spins[:, list(tup)], axis=-1) if tup else 1.0
    return total


def design_matrix(spins, groups):
    """(n_configs, 1 + n_terms): constant column then per-class product sums."""
    columns = [np.ones(len(spins))]
    for tuples in groups.values():
        columns.append(class_sum(spins, tuples))
    return np.stack(columns, axis=1)


def random_spins(n_configs, n_sites, rng):
    """Configurations spanning every composition, so the fit is not tied to c=1/2."""
    comps = rng.uniform(0.0, 1.0, size=n_configs)
    return np.where(rng.random((n_configs, n_sites)) < comps[:, None], 1, -1)


# ---------------------------------------------------------------- oracles ---


def clease_oracle(eci_file, structure_file, size):
    import ase.io
    from clease.calculator import attach_calculator
    from clease.settings import CEBulk, Concentration

    settings = CEBulk(
        crystalstructure="fcc", a=3.8, size=list(size),
        concentration=Concentration(basis_elements=[["Au", "Cu"]]),
        db_name="scratch_aucu.db", max_cluster_dia=[6.0, 4.5, 4.5],
    )
    eci = json.load(open(eci_file))
    atoms = attach_calculator(settings, atoms=ase.io.read(structure_file), eci=eci)

    def energy(spins):
        atoms.numbers = np.where(spins > 0, AU, CU)
        return float(atoms.get_potential_energy())

    cutoffs = {2: 6.0, 3: 4.5, 4: 4.5}
    return atoms.get_positions(), np.array(atoms.get_cell()), energy, cutoffs, {
        "source": "MetaDNS Cu-Au cluster expansion (Du et al. 2026, arXiv 2605.21722), "
                  "ECIs fitted by Damewood et al. 2022; CLEASE 1.1.0 CEBulk fcc a=3.8, "
                  "max_cluster_dia [6.0, 4.5, 4.5]",
        "eci_file": str(eci_file), "structure_file": str(structure_file),
        "energy_units": "eV, total energy of the periodic cell",
    }


def icet_oracle(ce_file, side):
    from ase import Atoms
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    ce = ClusterExpansion.read(ce_file)
    atoms = Atoms("Cu", cell=[2.5, 2.5, 20], pbc=True).repeat((side, side, 1))
    calc = ClusterExpansionCalculator(atoms, ce)

    def energy(spins):
        # calculate_total returns the cell total on the padded-cell scale the
        # fit used (eV per alloy atom times number of sites); see icet-ce/README
        return float(calc.calculate_total(occupations=np.where(spins > 0, AU, CU)))

    cutoffs = {2: 3.6, 3: 3.6}
    return atoms.get_positions(), np.array(atoms.get_cell()), energy, cutoffs, {
        "source": "icet-ce/cluster_expansion.ce (square-grid Cu/Au toy, EMT-fitted, "
                  "pairs + triplets, cutoffs 3.6/3.6 A, a = 2.5 A); evaluated here on a "
                  f"PERIODIC {side}x{side} cell with no vacancy padding",
        "ce_file": str(ce_file),
        "energy_units": "eV, ClusterExpansionCalculator.calculate_total on the periodic cell",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["clease", "icet"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eci-file", type=Path)
    parser.add_argument("--structure-file", type=Path)
    parser.add_argument("--size", type=int, nargs=3, default=(4, 4, 4))
    parser.add_argument("--ce-file", type=Path, default=Path("icet-ce/cluster_expansion.ce"))
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--n-fit", type=int, default=3000)
    parser.add_argument("--n-reference", type=int, default=64)
    args = parser.parse_args()

    if args.source == "clease":
        positions, cell, energy, cutoffs, meta = clease_oracle(
            args.eci_file, args.structure_file, args.size)
    else:
        positions, cell, energy, cutoffs, meta = icet_oracle(args.ce_file, args.side)
    n_sites = len(positions)

    groups = enumerate_tuples(positions, cell, cutoffs)
    print(f"{n_sites} sites; {len(groups)} shape classes:")
    for (order, sig), tuples in groups.items():
        print(f"  order {order} signature {sig}: {len(tuples)} tuples")

    rng = np.random.default_rng(0)
    spins = random_spins(args.n_fit, n_sites, rng)
    energies = np.array([energy(s) for s in spins])
    X = design_matrix(spins, groups)
    coef, *_ = np.linalg.lstsq(X, energies, rcond=None)
    residual = X @ coef - energies
    print(f"fit: max |residual| = {np.abs(residual).max():.3e} eV, "
          f"rms = {np.sqrt((residual ** 2).mean()):.3e} eV, "
          f"energy range {energies.min():.3f}..{energies.max():.3f} eV")
    assert np.abs(residual).max() < 1e-8, "enumeration does not span the expansion"

    reference_spins = random_spins(args.n_reference, n_sites, np.random.default_rng(1))
    reference_energies = [energy(s) for s in reference_spins]

    nn_signature = min(sig for (order, sig) in groups if order == 2)
    # one JSON term per (class, reduced length) so each tuple list is
    # rectangular; empties fold into the constant
    constant = float(coef[0])
    terms = []
    for ((order, sig), tuples), c in zip(groups.items(), coef[1:]):
        by_len = {}
        for tup in tuples:
            by_len.setdefault(len(tup), []).append(list(tup))
        constant += c * len(by_len.pop(0, []))
        for length, same in sorted(by_len.items()):
            terms.append({"order": length, "shape_order": order, "signature": list(sig),
                          "coefficient": float(c), "tuples": same})
    payload = {
        **meta,
        "n_sites": n_sites,
        "species": ["Cu", "Au"],
        "spin_convention": "+1 = Au (species[1]), -1 = Cu (species[0])",
        "cell": cell.tolist(),
        "positions": positions.tolist(),
        "cutoffs": {str(k): v for k, v in cutoffs.items()},
        "constant": constant,
        "terms": terms,
        "nearest_neighbour_pairs": [list(t) for t in groups[(2, nn_signature)] if len(t) == 2],
        "fit": {"n_configs": args.n_fit, "max_abs_residual_eV": float(np.abs(residual).max())},
        "reference": {"spins": reference_spins.tolist(), "energies": reference_energies},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
