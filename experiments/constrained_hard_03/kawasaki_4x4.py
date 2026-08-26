"""mchammer Kawasaki (CanonicalEnsemble) reference chains for the 4x4 demo.

CanonicalEnsemble's move is the non-local random unlike-pair swap -- the
same move set as the swap CTMC and the stronger classical variant
(angqvist_icet_2019). The Ising target maps onto the soft-02 campaign's
validated binary cluster expansion (ECI [0, -bias, -4*sigma], natural units
kT = 1), imported rather than re-derived.

Compute-currency accounting: mchammer counts TRIAL steps, one closed-form
Delta-E evaluation each, and its analyze_data correlation lengths are in
trial steps, NOT data entries. Snapshots land every `snapshot_interval`
trial steps; downstream N_eff(O) conversions work in trial steps throughout.

Local CPU only (no GPU, no Modal). One npz per (sigma, seed).

Example:
    pixi run -e dev python -m experiments.constrained_hard_03.kawasaki_4x4 \
        --sigmas 0.10 0.223 --n-trial-steps 200000
"""
import argparse
import time
from pathlib import Path

import numpy as np
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble

from discrete_flow_sampler.mcmc.mchammer_ising import ising_cluster_expansion


def site_index_map(supercell, D: int) -> np.ndarray:
    """Atom order -> row-major site index (row * D + col), matching the
    FixedCompositionIsingTarget flattening: the primitive cell is 1x1 in x/y,
    so integer-rounded positions are the lattice coordinates."""
    cols = np.rint(supercell.positions[:, 0]).astype(int) % D
    rows = np.rint(supercell.positions[:, 1]).astype(int) % D
    return rows * D + cols


def spins_from_symbols(symbols, index_map: np.ndarray) -> np.ndarray:
    """Au -> +1, Ag -> -1 (the soft-02 reference convention), placed at the
    mapped site indices."""
    spins = np.empty(len(symbols))
    for atom_position, site in enumerate(index_map):
        spins[site] = 1.0 if symbols[atom_position] == "Au" else -1.0
    return spins


def run_chain(D: int, sigma: float, seed: int, n_trial_steps: int,
              snapshot_interval: int):
    """One CanonicalEnsemble chain at 50/50 occupancy; returns
    (spins, mctrials, wall_seconds_setup, wall_seconds_run) with spins
    (S, D*D) in the target's site order. Wall-clock is recorded because the
    house tables' MCMC row is priced in wall-clock per effective sample (an
    MCMC sweep has no NFE analogue); setup — cluster-space and calculator
    construction — is timed separately and never folded into per-proposal
    cost, matching run_canonical_probe's convention."""
    setup_start = time.perf_counter()
    prim, _, ce = ising_cluster_expansion(sigma)
    supercell = prim.repeat((D, D, 1))
    n_sites = len(supercell)
    symbols = ["Au"] * (n_sites // 2) + ["Ag"] * (n_sites - n_sites // 2)
    np.random.default_rng(seed).shuffle(symbols)
    supercell.set_chemical_symbols(symbols)
    calculator = ClusterExpansionCalculator(supercell, ce)
    ensemble = CanonicalEnsemble(
        structure=supercell, calculator=calculator,
        temperature=1.0, boltzmann_constant=1.0, random_seed=seed,
        ensemble_data_write_interval=snapshot_interval,
        trajectory_write_interval=snapshot_interval,
        dc_filename=None,
    )
    run_start = time.perf_counter()
    ensemble.run(n_trial_steps)
    wall_seconds_run = time.perf_counter() - run_start
    mctrials, trajectory = ensemble.data_container.get("mctrial", "trajectory")
    index_map = site_index_map(supercell, D)
    spins = np.stack([
        spins_from_symbols(atoms.get_chemical_symbols(), index_map)
        for atoms in trajectory
    ])
    wall_seconds_setup = run_start - setup_start
    return spins, np.asarray(mctrials), wall_seconds_setup, wall_seconds_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.10, 0.223])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(1000, 1015)))
    parser.add_argument("--n-trial-steps", type=int, default=200_000)
    parser.add_argument("--snapshot-interval", type=int, default=20)
    parser.add_argument("--out", default="results/03_hard/demo_4x4/kawasaki")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sigma in args.sigmas:
        sigma_tag = f"s{round(sigma * 1000):03d}"
        for seed in args.seeds:
            spins, mctrials, wall_setup, wall_run = run_chain(
                args.D, sigma, seed, args.n_trial_steps, args.snapshot_interval
            )
            np.savez(
                out_dir / f"kawasaki_D{args.D}_{sigma_tag}_seed{seed}.npz",
                spins=spins, mctrial=mctrials,
                n_trial_steps=args.n_trial_steps,
                snapshot_interval=args.snapshot_interval,
                sigma=sigma, seed=seed, D=args.D,
                wall_seconds_setup=wall_setup, wall_seconds_run=wall_run,
            )
            print(f"[kawasaki] {sigma_tag} seed {seed}: {len(spins)} snapshots "
                  f"({wall_run:.1f}s run)", flush=True)


if __name__ == "__main__":
    main()
