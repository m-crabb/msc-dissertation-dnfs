"""Build the Wolff reference pools that replace the Gibbs pools as the
baseline chapter's sample-level ground truth.

Pool structure mirrors the Gibbs pools exactly (100 independent chains x 50
records, pooled record-major, same dict keys) so the house-table filler's
chain-block bootstrap works unchanged; burn_in / thin are measured in Wolff
CLUSTERS, not sweeps, and the chains are genuinely independent (fresh seed
each), which makes the Gelman--Rubin diagnostic a real multi-start check.

Writes: results/01_baseline/wolff_ref_d10_sigma{0.1,0.220343,0.22305}.pt
(~a minute each; 0.220343 = SIGMA_C to :g precision).
"""

import json
import sys
from pathlib import Path

import torch

from discrete_flow_sampler.diagnostics.metrics import gelman_rubin
from discrete_flow_sampler.mcmc.wolff import wolff_sample
from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "01_baseline"
L = 10
N_CHAINS = 100
N_RECORDS = 50
BURN_IN_CLUSTERS = 500
CLUSTERS_PER_RECORD = 20
BASE_SEED = 20260824


def recount_flops(sigma: float) -> None:
    """FLOP-recount an EXISTING pool for the house table's cost column.

    Replays every chain with the build seeds and the cluster-size log, then
    verifies bit-identity against the stored samples -- wolff_sample is
    deterministic in its seed and the log is observation-only, so this
    prices the certified pool exactly without rebuilding it. A failed
    identity check means the sampler code drifted since the pool was built,
    and the recount must not be trusted (rebuild-and-recertify instead).
    Writes a <pool>.pt.flops.json sidecar; the pool file is never touched.
    """
    from discrete_flow_sampler.diagnostics.flops import (
        WOLFF_FLOPS_PER_CLUSTER_SITE,
        wolff_run_flops,
    )

    pool_path = RESULTS / f"wolff_ref_d10_sigma{sigma:g}.pt"
    stored = torch.load(pool_path, weights_only=True)
    target = IsingTarget(D=L, sigma=sigma, bias=0.0, device="cpu")
    cluster_sizes: list[int] = []
    per_chain = [
        wolff_sample(
            target,
            n_samples=N_RECORDS,
            clusters_per_sample=CLUSTERS_PER_RECORD,
            burn_in_clusters=BURN_IN_CLUSTERS,
            seed=BASE_SEED + chain,
            cluster_size_log=cluster_sizes,
        )
        for chain in range(N_CHAINS)
    ]
    replayed = torch.stack(per_chain, dim=1).reshape(N_RECORDS * N_CHAINS, target.d)
    assert torch.equal(replayed.to(torch.int8), stored["samples"]), (
        f"replay diverged from {pool_path.name} -- sampler drift, recount void"
    )

    total_sites = sum(cluster_sizes)
    sidecar = {
        "sigma": sigma,
        "total_clusters": len(cluster_sizes),
        "total_cluster_sites": total_sites,
        "mean_cluster_size": total_sites / len(cluster_sizes),
        "total_flops": wolff_run_flops(total_sites),
        "flops_per_cluster_site": WOLFF_FLOPS_PER_CLUSTER_SITE,
        "n_records_pooled": N_RECORDS * N_CHAINS,
        "includes_burn_in": True,
    }
    sidecar_path = Path(str(pool_path) + ".flops.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(
        f"[wolff flops sigma={sigma:g}] mean cluster "
        f"{sidecar['mean_cluster_size']:.1f} sites, total "
        f"{sidecar['total_flops']:.3g} FLOPs -> {sidecar_path.name}"
    )


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
    out = RESULTS / f"wolff_ref_d10_sigma{sigma:g}.pt"
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
    print(
        f"[wolff ref sigma={sigma}] {N_CHAINS} chains x {N_RECORDS} records"
        f" -> {out.name}  R-hat(m) = {rhat_m:.4f}"
    )


if __name__ == "__main__":
    # 0.22305 is the LEGACY pool (pairs with the archived pre-migration runs,
    # kept on disk); SIGMA_C is the pool every post-migration run evaluates
    # against. Existing pool files are not rebuilt.
    for sigma in (0.1, SIGMA_C, 0.22305):
        out = RESULTS / f"wolff_ref_d10_sigma{sigma:g}.pt"
        if "--recount-flops" in sys.argv:
            if out.exists():
                recount_flops(sigma)
            continue
        if out.exists():
            print(f"[wolff ref sigma={sigma:g}] exists, skipping {out.name}")
            continue
        build_pool(sigma)
