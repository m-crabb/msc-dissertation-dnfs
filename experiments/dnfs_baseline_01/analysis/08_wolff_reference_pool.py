"""Build the Wolff reference pools that replace the Gibbs pools as the
baseline chapter's sample-level ground truth (user decision, s58 2026-08-24).

Why the swap: the s58 cross-check (07_reference_crosscheck.py) certified the
two pools AGREE, so this is not a correction — it is choosing the instrument
whose floor is honest. The Gibbs pool's dMag floor at sigma_c (0.26) is
mode-stickiness: chains stuck in one Z2 sector make pooled |m| balance a
matter of luck (chain-mean SD 0.31). A Wolff chain tunnels sectors freely
(cluster flips change M by +-2|C|), so the same chain-block bootstrap floor
collapses to the 1/sqrt(N) scale and the reference can finally certify Z2
balance at the level the neural cells sit at.

Pool structure mirrors the Gibbs pools exactly (100 independent chains x 50
records, pooled record-major, same dict keys) so the house-table filler's
chain-block bootstrap works unchanged; burn_in / thin are measured in Wolff
CLUSTERS, not sweeps, and the chains are genuinely independent (fresh seed
each), which makes the Gelman--Rubin diagnostic a real multi-start check.

Writes: results/01_baseline/wolff_ref_d10_sigma{0.1,0.22305}.pt (~a minute).
"""
from pathlib import Path

import torch

from discrete_flow_sampler.diagnostics.metrics import gelman_rubin
from discrete_flow_sampler.mcmc.wolff import wolff_sample
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "01_baseline"
L = 10
N_CHAINS = 100
N_RECORDS = 50
BURN_IN_CLUSTERS = 500
CLUSTERS_PER_RECORD = 20
BASE_SEED = 20260824


def build_pool(sigma: float) -> None:
    target = IsingTarget(D=L, sigma=sigma, bias=0.0, device="cpu")
    per_chain = [
        wolff_sample(
            target,
            n_samples=N_RECORDS,
            clusters_per_sample=CLUSTERS_PER_RECORD,
            burn_in_clusters=BURN_IN_CLUSTERS,
            seed=BASE_SEED + chain,
        )
        for chain in range(N_CHAINS)
    ]
    stacked = torch.stack(per_chain, dim=1)  # (records, chains, d)
    pooled = stacked.reshape(N_RECORDS * N_CHAINS, target.d)  # record-major
    m_per_chain = stacked.mean(dim=2).T  # (chains, records)
    rhat_m = gelman_rubin(m_per_chain.numpy())
    out = RESULTS / f"wolff_ref_d10_sigma{sigma}.pt"
    torch.save(
        {
            "samples": pooled.to(torch.int8),
            "sigma": sigma,
            "gelman_rubin_m": rhat_m,
            "n_chains": N_CHAINS,
            "burn_in": BURN_IN_CLUSTERS,
            "thin": CLUSTERS_PER_RECORD,
        },
        out,
    )
    print(f"[wolff ref sigma={sigma}] {N_CHAINS} chains x {N_RECORDS} records"
          f" -> {out.name}  R-hat(m) = {rhat_m:.4f}")


if __name__ == "__main__":
    for sigma in (0.1, 0.22305):
        build_pool(sigma)
