"""Fill pass for tab:eval-unconstrained-10x10 (the house evaluation table).

Reads the frozen eval artefacts of the eight Stage-4 10x10 runs (samples.pt,
log_weights.pt, metrics.json; seeds 42-45 at sigma = 0.1 and sigma_c) and the
two cached WOLFF references, and prints the table's observable cells:

  ESS    -- frozen eval/ess_fraction, re-read not recomputed (the in-print value);
  dMag   -- MDNS Eq. 26, dCorr -- MDNS Eq. 28, EW2 -- 1-D W2 on E(x)/d (DASBS),
            each on importance-reweighted samples against the reference;
  reference row -- the sampling floor under each column: resample the 100
            WOLFF chains with replacement (chain-block bootstrap, the chain is
            the independent unit) and score the replicate against the full
            reference. A neural cell at or below this floor is indistinguishable
            from the reference at N = 5000.
  FLOP/es -- neural: measured eager forward x n_euler / ESS; reference: the
            recounted pool build from the .flops.json sidecar (08
            --recount-flops) over its effective record count.

Per-site energy follows the chapter's convention E/d = -log p~(x) / (2 sigma d)
(metrics.internal_energy_estimate).

sigma_c MIGRATION (s73, 2026-08-26): the sigma_c point now reads the Wave-1
`_sc` retrains at the ONE critical coupling SIGMA_C = ln(1+sqrt(2))/4 =
0.220343, against the matching 0.220343 Wolff pool. It previously read the
legacy 0.22305 family; those runs are archived records and are not edited.
The retrain passed its pre-registered bands 4/4 in every family (final fp32
5000-draw eval ESS: d10 0.902 +- 0.019 over floor 0.86, d8 0.962 +- 0.008
over 0.89, d4 0.986 +- 0.004 over 0.93), which is what authorises the swap.
Never mix couplings in one comparison: sigma_c runs pair with the sigma_c
pool, legacy with legacy.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from discrete_flow_sampler.diagnostics.flops import (
    chain_per_effective_sample, measured_forward_flops,
    neural_sampling_flops_per_sample, per_effective_sample, sgc_run_flops)
from discrete_flow_sampler.diagnostics.metrics import (
    correlation_profile_error, energy_wasserstein2, integrated_autocorr,
    magnetisation_profile_error)
from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
# Support experiments-package imports when invoked by file path.
import sys
sys.path.insert(0, str(REPO_ROOT))
RESULTS = REPO_ROOT / "results" / "01_baseline"
L = 10
OPERATING_POINTS = {
    "sigma_0.1": dict(sigma=0.1, runs="stage_4_d10_budget_seed4*",
                      reference="wolff_ref_d10_sigma0.1.pt"),
    "sigma_c": dict(
        sigma=SIGMA_C,
        runs="stage_4_d10_critical_paper_curriculum_sc_seed4*_20260824-wave1-sc",
        reference="wolff_ref_d10_sigma0.220343.pt"),
}
N_BOOTSTRAP = 200
# --sgc: the practitioner row, one dir per independent chain (scripts/
# mchammer_baselines.py sgc --record-spins). Dir names carry sigma at full
# repr, hence the long sigma_c glob.
SGC_RUNS = REPO_ROOT / "results" / "mchammer_sgc"
SGC_CHAINS = {"sigma_0.1": "D10_s0.1_c00.50_seed4*",
              "sigma_c": f"D10_s{SIGMA_C!r}_c00.50_seed4*"}


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


def family_flops_per_forward(run_dir: Path, target: IsingTarget) -> int:
    """Measured FLOPs of one rate-matrix forward at this run's architecture.

    Built eager from the run's own config (a compiled wrapper can hide ops
    from the dispatch-level counter; compilation changes scheduling, not
    the mathematics) at batch 1 -- the counter is batch-linear, test-pinned.
    """
    from experiments.dnfs_baseline_01.run import _construct_model, _sub_config
    from experiments.dnfs_baseline_01.configs import ModelCfg

    cfg_dict = json.loads((run_dir / "config.json").read_text())
    model_cfg = _sub_config(ModelCfg, {**cfg_dict["model"], "compile_model": False})
    model = _construct_model(SimpleNamespace(model=model_cfg), target)
    example = (target.sample_base(1, device="cpu"), torch.zeros(1))
    return measured_forward_flops(model, example)


def slowest_observable_tau_int(by_chain: torch.Tensor, target: IsingTarget) -> float:
    """tau_int of the slowest tabled observable, in record units: the larger
    of the magnetisation and energy reads, each averaged over chains.
    ``by_chain`` is (n_records, n_chains, d), record-major."""
    n_records, n_chains, _ = by_chain.shape
    series = {
        "m": by_chain.mean(dim=2).T,
        "E": (-target.log_prob(by_chain.reshape(-1, target.d))
              .view(n_records, n_chains) / (2 * target.sigma)).T,
    }
    return max(
        float(torch.tensor([integrated_autocorr(chain.numpy())
                            for chain in per_chain]).mean())
        for per_chain in series.values()
    )


def reference_flops_per_es(reference: torch.Tensor, n_chains: int,
                           target: IsingTarget, sidecar_path: Path) -> float | None:
    """Reference-row cost cell: the recounted pool build divided by its
    effective record count, tau_int taken as the larger of the energy and
    magnetisation reads (the slowest tabled observable, in record units)."""
    if not sidecar_path.exists():
        return None  # recount not run (08 --recount-flops); cell stays blank
    sidecar = json.loads(sidecar_path.read_text())
    n_records = reference.shape[0] // n_chains
    tau_int = slowest_observable_tau_int(reference.view(n_records, n_chains, -1), target)
    return chain_per_effective_sample(
        sidecar["total_flops"], sidecar["n_records_pooled"], max(tau_int, 1.0))


def score_sgc_row():
    """Practitioner row "mchammer (SGC)": eight independent single-flip chains
    per coupling (results/mchammer_sgc, Delta-mu = 0 so the chain targets the
    same p~(x) as the DNFS row), scored on their post-burn-in spin frames,
    unweighted, against the SAME Wolff pool and metric functions as the DNFS
    row. Cells are the per-chain error averaged over the 8 chains +- SD, the
    same construction as the DNFS row's per-seed average, so the two rows
    are comparable.

    FLOP/es follows the Wolff reference row exactly: sgc_run_flops =
    12 x n_trials (a free single-site flip is a bare Gibbs site update,
    burn-in trials included) over N / tau_int, with tau_int the slowest of
    the magnetisation and energy reads (slowest_observable_tau_int, chain-
    averaged, frame units -- spins.npy frame k is the state at trial
    k x data_write_interval, so N and tau_int share the unit). The bill is
    linear in tau_int, so the pooled cell equals the mean of the per-chain
    bills; the per-chain bill is also recorded so its spread is visible.
    """
    table = {}
    for point, spec in OPERATING_POINTS.items():
        reference = torch.load(RESULTS / spec["reference"], weights_only=False)["samples"].float()
        target = IsingTarget(D=L, sigma=spec["sigma"], bias=0.0)
        per_chain, frames_by_chain = {}, []
        for chain_dir in sorted(SGC_RUNS.glob(SGC_CHAINS[point])):
            frames = torch.from_numpy(np.load(chain_dir / "spins.npy")).float()
            summary = json.loads((chain_dir / "summary.json").read_text())
            uniform = torch.full((frames.shape[0],), 1.0 / frames.shape[0])
            tau_int = slowest_observable_tau_int(frames.unsqueeze(1), target)
            per_chain[chain_dir.name] = {
                **observable_errors(frames, uniform, reference, target),
                "tau_int_frames": tau_int,
                "FLOPes": chain_per_effective_sample(
                    sgc_run_flops(summary["n_steps"]), frames.shape[0], max(tau_int, 1.0)),
            }
            frames_by_chain.append(frames)

        # Pooled bill, the Wolff row's construction: all chains' trials over
        # all chains' frames at the chain-averaged slowest tau_int.
        pooled = torch.stack(frames_by_chain, dim=1)
        pooled_tau_int = slowest_observable_tau_int(pooled, target)
        n_chains, n_trials = len(per_chain), summary["n_steps"]
        pooled_flops_per_es = chain_per_effective_sample(
            n_chains * sgc_run_flops(n_trials), pooled.shape[0] * n_chains,
            max(pooled_tau_int, 1.0))

        aggregate = {k: mean_sd([c[k] for c in per_chain.values()]) for k in next(iter(per_chain.values()))}
        table[point] = {"sigma": spec["sigma"], "n_chains": n_chains,
                        "n_trials_per_chain": n_trials, "n_frames_per_chain": pooled.shape[0],
                        "sgc_per_chain": per_chain, "sgc_mean_sd": aggregate,
                        "sgc_pooled_tau_int_frames": pooled_tau_int,
                        "sgc_pooled_flops_per_es": pooled_flops_per_es}

        print(f"\n== {point} (sigma={spec['sigma']}, {n_chains} SGC chains, "
              f"{n_trials:.0e} trials, {pooled.shape[0]} frames each)")
        for name, c in per_chain.items():
            print(f"  {name}: " + " ".join(f"{k}={v:.4g}" for k, v in c.items()))
        print("  SGC mean +- SD:", {k: f"{m:.4g} +- {sd:.2g}" for k, (m, sd) in aggregate.items()})
        print(f"  pooled tau_int {pooled_tau_int:.4g} frames -> FLOP/es {pooled_flops_per_es:.2g}")

    out = RESULTS / "house_table_unconstrained_10x10_sgc.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


def main():
    table = {}
    for point, spec in OPERATING_POINTS.items():
        ref = torch.load(RESULTS / spec["reference"], weights_only=False)
        reference, n_chains = ref["samples"].float(), ref["n_chains"]
        target = IsingTarget(D=L, sigma=spec["sigma"], bias=0.0)
        floor = reference_floor(reference, n_chains, target)

        ref_flops_per_es = reference_flops_per_es(
            reference, n_chains, target,
            Path(str(RESULTS / spec["reference"]) + ".flops.json"))

        per_seed, per_forward = {}, None
        for run_dir in sorted(RESULTS.glob(spec["runs"])):
            x = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
            weights = torch.softmax(torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True), 0)
            metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
            if per_forward is None:  # one architecture per family
                per_forward = family_flops_per_forward(run_dir, target)
            n_euler = json.loads((run_dir / "config.json").read_text())["ctmc"]["n_euler_steps"]
            per_seed[run_dir.name] = {
                "ESS": metrics["ess_fraction"],
                **observable_errors(x, weights, reference, target),
                "FLOPes": per_effective_sample(
                    neural_sampling_flops_per_sample(per_forward, n_euler, target.d),
                    metrics["ess_fraction"]),
            }

        summary = {k: mean_sd([s[k] for s in per_seed.values()]) for k in next(iter(per_seed.values()))}
        table[point] = {"sigma": spec["sigma"], "reference_floor": floor,
                        "reference_gelman_rubin_m": ref["gelman_rubin_m"],
                        "reference_flops_per_es": ref_flops_per_es,
                        "dnfs_flops_per_forward": per_forward,
                        "dnfs_per_seed": per_seed, "dnfs_mean_sd": summary}

        print(f"\n== {point} (sigma={spec['sigma']}, {len(per_seed)} seeds)")
        print("  reference floor:", {k: f"{v:.2e}" for k, v in floor.items()})
        print(f"  reference FLOP/es: "
              f"{'(recount sidecar missing)' if ref_flops_per_es is None else f'{ref_flops_per_es:.2g}'}"
              f"   DNFS forward: {per_forward:.3g} FLOPs")
        for name, s in per_seed.items():
            print(f"  {name}: " + " ".join(f"{k}={v:.4g}" for k, v in s.items()))
        print("  DNFS mean +- SD:", {k: f"{m:.4g} +- {sd:.2g}" for k, (m, sd) in summary.items()})

    out = RESULTS / "house_table_unconstrained_10x10.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    score_sgc_row() if "--sgc" in sys.argv[1:] else main()
