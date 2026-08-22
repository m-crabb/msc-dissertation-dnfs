"""Fill pass for tab:eval-unconstrained-10x10 (the house evaluation table).

Reads the frozen eval artefacts of the eight Stage-4 10x10 runs (samples.pt,
log_weights.pt, metrics.json; seeds 42-45 at sigma = 0.1 and sigma_c) and the
two cached Gibbs references, and prints the table's observable cells:

  ESS    -- frozen eval/ess_fraction, re-read not recomputed (the in-print value);
  dMag   -- MDNS Eq. 26, dCorr -- MDNS Eq. 28, EW2 -- 1-D W2 on E(x)/d (DASBS),
            each on importance-reweighted samples against the reference;
  reference row -- the sampling floor under each column: resample the 100
            Gibbs chains with replacement (chain-block bootstrap, the chain is
            the independent unit) and score the replicate against the full
            reference. A neural cell at or below this floor is indistinguishable
            from the reference at N = 5000.

FLOP/es and the matched-budget Gibbs row are NOT produced here (no FLOP counter
exists yet); those cells stay empty. Per-site energy follows the chapter's
convention E/d = -log p~(x) / (2 sigma d) (metrics.internal_energy_estimate).
"""
import json
from pathlib import Path

import torch

from discrete_flow_sampler.diagnostics.metrics import (
    correlation_profile_error, energy_wasserstein2, magnetisation_profile_error)
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "01_baseline"
L = 10
OPERATING_POINTS = {
    "sigma_0.1": dict(sigma=0.1, runs="stage_4_d10_budget_seed4*",
                      reference="gibbs_ref_d10_sigma0.1.pt"),
    "sigma_c": dict(sigma=0.22305, runs="stage_4_d10_critical_paper_curriculum_seed4*",
                    reference="gibbs_ref_d10_sigma0.22305.pt"),
}
N_BOOTSTRAP = 200


def observable_errors(x, weights, reference, target):
    energy_per_site = lambda s: -target.log_prob(s) / (2 * target.sigma * target.d)
    return {
        "dMag": magnetisation_profile_error(x, weights, reference, L),
        "dCorr": correlation_profile_error(x, weights, reference, L),
        "EW2": energy_wasserstein2(energy_per_site(x), weights, energy_per_site(reference)),
    }


def reference_floor(reference, n_chains, target, seed=0):
    """Chain-block bootstrap of the reference against itself, mean over replicates."""
    generator = torch.Generator().manual_seed(seed)
    n_records = reference.shape[0] // n_chains
    by_chain = reference.view(n_records, n_chains, -1)  # pooled order was record-major
    replicates = []
    for _ in range(N_BOOTSTRAP):
        chains = torch.randint(0, n_chains, (n_chains,), generator=generator)
        replicate = by_chain[:, chains].reshape(-1, reference.shape[1])
        uniform = torch.full((replicate.shape[0],), 1.0 / replicate.shape[0])
        replicates.append(observable_errors(replicate, uniform, reference, target))
    return {k: sum(r[k] for r in replicates) / N_BOOTSTRAP for k in replicates[0]}


def mean_sd(values):
    t = torch.tensor(values)
    return t.mean().item(), t.std().item()


def main():
    table = {}
    for point, spec in OPERATING_POINTS.items():
        ref = torch.load(RESULTS / spec["reference"], weights_only=False)
        reference, n_chains = ref["samples"].float(), ref["n_chains"]
        target = IsingTarget(D=L, sigma=spec["sigma"], bias=0.0)
        floor = reference_floor(reference, n_chains, target)

        per_seed = {}
        for run_dir in sorted(RESULTS.glob(spec["runs"])):
            x = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
            weights = torch.softmax(torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True), 0)
            metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
            per_seed[run_dir.name] = {"ESS": metrics["ess_fraction"],
                                      **observable_errors(x, weights, reference, target)}

        summary = {k: mean_sd([s[k] for s in per_seed.values()]) for k in next(iter(per_seed.values()))}
        table[point] = {"sigma": spec["sigma"], "reference_floor": floor,
                        "reference_gelman_rubin_m": ref["gelman_rubin_m"],
                        "dnfs_per_seed": per_seed, "dnfs_mean_sd": summary}

        print(f"\n== {point} (sigma={spec['sigma']}, {len(per_seed)} seeds)")
        print("  reference floor:", {k: f"{v:.2e}" for k, v in floor.items()})
        for name, s in per_seed.items():
            print(f"  {name}: " + " ".join(f"{k}={v:.4g}" for k, v in s.items()))
        print("  DNFS mean +- SD:", {k: f"{m:.4g} +- {sd:.2g}" for k, (m, sd) in summary.items()})

    out = RESULTS / "house_table_unconstrained_10x10.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
