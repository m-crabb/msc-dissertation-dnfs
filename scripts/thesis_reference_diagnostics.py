"""Recompute reference diagnostics only; never train or rewrite neural exports.

Example: python scripts/thesis_reference_diagnostics.py --out /tmp/reference-review

Reuse the hard chapter's individual-frame iid floor and whole-chain split
uncertainty. The latter is half the mean split distance, an approximate
metric-scale diagnostic rather than a calibrated standard error. For soft
cells, four chains allow only three distinct balanced partitions; the 64
seeded splits repeat these, matching the existing hard-table procedure.
All floors condition on the empirical pool and exclude its own uncertainty.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget
from experiments.constrained_hard_03.analysis.house_table_8x8 import (
    reference_standard_error, sampling_floor_from_reference)
from experiments.constrained_soft_02.analysis import composition_marginal_overlay_8x8 as soft
from experiments.dnfs_baseline_01.analysis import unconstrained_clean_demo as baseline


def fingerprint(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.relative_to(REPO_ROOT)),
            "sha256": digest}


def evaluate(chains, target, inputs, out):
    if out.exists():
        raise FileExistsError(f"Use a new output directory; preserving {out}")
    reference = torch.cat(chains)
    energies = [-target.base_log_prob(chain) / (2 * target.sigma * target.d)
                for chain in chains]
    print(f"{out.stem}: {len(chains)} chains, {len(reference)} frames; split uncertainty", flush=True)
    uncertainty = reference_standard_error(chains, target.D, chain_energies=energies)
    print(f"{out.stem}: iid floor", flush=True)
    floor = sampling_floor_from_reference(reference, target.D, 5000,
                                         reference_energy=torch.cat(energies))
    if target.D == 8:
        tv_floor = {name: soft.reference_tv_floor(
            reference, lambda x, w, fn=fn: fn(target, x, w))
            for name, fn in (("energy", soft.energy_pmf), ("composition", soft.composition_pmf))}
    else:
        # Reconstruct record-major order to retain seeded resampling identity
        # with the baseline plot's original reference input.
        reference = torch.stack(chains, 1).reshape(-1, target.d)
        def energy_pmf(x, weights):
            return torch.zeros(target.d + 1).index_add_(
                0, baseline.energy_level_index(target, x), weights)
        def magnet_pmf(x, weights):
            count = ((x + 1) / 2).sum(1).round().long()
            return torch.zeros(target.d + 1).index_add_(0, count, weights)
        tv_floor = {name: baseline.reference_tv_floor(reference, len(chains), fn)
                    for name, fn in (("energy", energy_pmf), ("magnetisation", magnet_pmf))}
    result = dict(lattice_edge=target.D, sigma=target.sigma,
                  chain_lengths=[len(chain) for chain in chains],
                  reference_uncertainty=uncertainty, iid_floor=floor,
                  marginal_tv_iid_floor=tv_floor, n_draws=5000,
                  n_replicates=200, n_splits=64, seed=0,
                  uncertainty_method="mean whole-chain split distance / 2 (approximate)",
                  floor_method="independent reference frames with replacement vs full pool",
                  inputs=[fingerprint(path) for path in inputs])
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{out.stem}: {json.dumps(dict(uncertainty=uncertainty, floor=floor, tv=tv_floor))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hard-marginals", action="store_true",
                        help="calculate only the critical 8x8 and 16x16 marginal floors")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.hard_marginals:
        from experiments.constrained_hard_03.analysis import hard_results_cell as hard
        for edge in (8, 16):
            out = args.out / f"hard_{edge}_sc_marginals.json"
            if out.exists():
                raise FileExistsError(out)
            print(f"hard {edge}: iid marginal floors", flush=True)
            chains = hard.load_reference(edge, "s220")
            if edge == 8:
                inputs = sorted((REPO_ROOT / "results/03_hard/kawasaki_w2").glob(
                    "kawasaki_D8_s220_seed*.npz"))
            else:
                directory = REPO_ROOT / "results/kawasaki_ref_d256_s220"
                inputs = [directory / "samples.pt", directory / "provenance.json"]
            result = dict(lattice_edge=edge, sigma=SIGMA_C, n_draws=5000,
                          n_replicates=200, seed=0, chain_lengths=[len(c) for c in chains],
                          marginal_tv_iid_floor={
                              "energy": hard.energy_floor(chains, edge, 5000, 200),
                              "phi": hard.phi_floor(chains, edge, 5000, 200)},
                          floor_method="independent reference frames with replacement vs full pool",
                          inputs=[fingerprint(path) for path in inputs])
            out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(result["marginal_tv_iid_floor"], flush=True)
        return
    for label, sigma in (("s010", .1), ("sc", SIGMA_C)):
        path = REPO_ROOT / f"results/01_baseline/wolff_ref_d10_sigma{sigma:g}.pt"
        pool = torch.load(path, weights_only=True)
        chains = list(pool["samples"].float().reshape(-1, pool["n_chains"], 100).unbind(1))
        evaluate(chains, IsingTarget(D=10, sigma=sigma), [path], args.out / f"baseline_{label}.json")
        for composition in (.25, .375, .5):
            paths = sorted((REPO_ROOT / "results/mchammer_vcsgc").glob(
                f"D8_s{sigma:g}_l50.0_c{composition:.3f}_seed*/spins.npy"))
            assert len(paths) == 4, paths
            target = IsingTarget(D=8, sigma=sigma)
            chains = [torch.from_numpy(np.load(path)).float() for path in paths]
            potentials = [path.with_name("potential.npy") for path in paths]
            for path, chain, potential in zip(paths, chains, potentials):
                assert torch.allclose(-target.base_log_prob(chain),
                                      torch.from_numpy(np.load(potential)).float(), atol=1e-3), path
            evaluate(chains, target, paths + potentials,
                     args.out / f"soft_{label}_c{composition:.3f}.json")


if __name__ == "__main__":
    main()
