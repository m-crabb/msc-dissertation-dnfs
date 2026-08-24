"""Fill pass for tab:eval-soft-10x10 (the soft chapter's house evaluation table).

Same cells and conventions as the unconstrained fill
(`dnfs_baseline_01/analysis/06_house_table_unconstrained.py`): frozen
eval/ess_fraction re-read, dMag (MDNS Eq. 26), dCorr (MDNS Eq. 28) and EW2 on
importance-reweighted samples against the reference, and a reference row that
prints the reference's own sampling floor under each column.

What differs for the soft target:

  reference -- the literal mchammer VC-SGC chains at kappa = lambda,
            phi = -2 c_target (results/mchammer_vcsgc/D10_*, seeds 42-45, 1M
            trials each, burn-in 1/3), re-run with `--record-spins` so the
            per-site profiles can be scored. `potential.npy` is mchammer's own
            energy at every frame and must equal the Ising energy recomputed
            from `spins.npy`; the script asserts it, which pins atom order.
  floor   -- four chains are too few units for the chain bootstrap the
            unconstrained fill uses, so the floor is a block bootstrap over
            frames pooled across chains (blocks of BLOCK frames >> tau_int of
            ~1-1.6 frames at the 100-trial write interval). Each replicate has
            N_EVAL = 5000 frames, the neural draw count, and is scored against
            the full 26.7k-frame reference: the unconstrained reference happens
            to be 5000 samples, so its full-size replicate already meant "a
            5000-draw from the reference"; here the reference is 5x larger and
            a full-size replicate would understate the floor by sqrt(5).
  energy  -- EW2 is on the *Ising* energy per site, penalty excluded: the
            penalty is the constraint, not the physics, and both sampler and
            reference draw from the same penalised law so the comparison is
            like for like. E/d = -base_log_prob(x) / (2 sigma d).
  seeds   -- every seed is scored and printed; the chapter's 0.30 ESS floor is
            applied only in the summary line, with n_pass/n_total, so the
            table can print either rule and the failed seeds are never hidden.

FLOP/es and the matched-budget VC-SGC row are NOT produced (no FLOP counter
exists yet), exactly as in the unconstrained table. sigma_c is not run in
this chapter.
"""
import json
from pathlib import Path

import numpy as np
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    correlation_profile_error, energy_wasserstein2, magnetisation_profile_error)
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
SOFT_RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
VCSGC_RESULTS = REPO_ROOT / "results" / "mchammer_vcsgc"
L, SIGMA = 10, 0.1
ESS_FLOOR = 0.30
N_BOOTSTRAP, BLOCK, N_EVAL = 200, 10, 5000

# (lambda, c_target) -> glob of the specialist run dirs that stand for that
# cell. The lambda=50 rows are the annealed production recipe the F(c)
# campaign prints; the fixed-lambda=50 c=0.50 family (tab:soft-multiseed,
# 1 seed in 4 trains) is included as a separate entry so both are on record.
CELLS = {
    (10, 0.30): "S2_d10_c03_l10_letf_ne128_seed4*",
    (10, 0.50): "S2_d10_c05_l10_letf_ne64_seed4*",
    # The four F(c) windows print from the ne128 retrain families (s62): the
    # ne64 residual vs the TI truth halved under the grid refinement in every
    # window, so ne128 is the production recipe. c=0.30 stays on ne64 until
    # its ne128 top-up (seeds 46-49) is judged -- 0/4 of the first ne128 draw
    # cleared the 0.30 ESS floor. c=0.80 prints the N11 campaign's
    # uniform-base CONTROL arm (shared tag = one launch family, one dir per
    # seed): the untagged glob would also sweep in the two replicate seed-45
    # runs kept as the FP-nondeterminism record (ESS 0.423 vs 0.899 on an
    # identical config+seed).
    (50, 0.30): "S2_d10_c030_l50_letf_ne64_anneal_seed4*",
    (50, 0.50): "S2_d10_c05_l50_letf_ne128_anneal_seed4*",
    (50, 0.55): "S2_d10_c055_l50_letf_ne128_anneal_seed4*",
    (50, 0.60): "S2_d10_c060_l50_letf_ne128_anneal_seed4*",
    (50, 0.65): "S2_d10_c065_l50_letf_ne128_anneal_seed4*",
    (50, 0.80): "S2_d10_c080_l50_letf_ne128_anneal_seed4?_20260822-N11mb",
}
FIXED_LAMBDA_CELLS = {(50, 0.50): "S2_d10_c05_l50_letf_ne64_seed4*"}

TARGET = IsingTarget(D=L, sigma=SIGMA, bias=0.0)


def energy_per_site(x):
    return -TARGET.base_log_prob(x) / (2 * SIGMA * TARGET.d)


def observable_errors(x, weights, reference):
    return {
        "dMag": magnetisation_profile_error(x, weights, reference, L),
        "dCorr": correlation_profile_error(x, weights, reference, L),
        "EW2": energy_wasserstein2(energy_per_site(x), weights, energy_per_site(reference)),
    }


def load_vcsgc_reference(penalty_strength, c_target):
    """Pooled post-burn-in spin frames over the reference seeds, order-checked."""
    frames, wall_seconds, chains = [], 0.0, 0
    for run_dir in sorted(VCSGC_RESULTS.glob(f"D{L}_s{SIGMA}_l{penalty_strength:.1f}_c{c_target:.2f}_seed*")):
        spins = torch.from_numpy(np.load(run_dir / "spins.npy")).float()
        potential = torch.from_numpy(np.load(run_dir / "potential.npy")).float()
        # mchammer's CE energy is -log p~(x) on the validated embedding; a
        # scrambled atom order would break this equality and every profile.
        assert torch.allclose(-TARGET.base_log_prob(spins), potential, atol=1e-3), run_dir
        frames.append(spins)
        wall_seconds += json.loads((run_dir / "summary.json").read_text())["wall_seconds_run"]
        chains += 1
    if not frames:
        raise FileNotFoundError(f"no VC-SGC reference with spins.npy for lambda={penalty_strength} c={c_target}")
    return torch.cat(frames), chains, wall_seconds


def reference_floor(reference, seed=0):
    """Block bootstrap: N_EVAL-frame replicates of the reference scored against all of it."""
    generator = torch.Generator().manual_seed(seed)
    n_blocks = reference.shape[0] // BLOCK
    by_block = reference[: n_blocks * BLOCK].view(n_blocks, BLOCK, -1)
    replicates = []
    for _ in range(N_BOOTSTRAP):
        blocks = torch.randint(0, n_blocks, (N_EVAL // BLOCK,), generator=generator)
        replicate = by_block[blocks].reshape(-1, reference.shape[1])
        uniform = torch.full((replicate.shape[0],), 1.0 / replicate.shape[0])
        replicates.append(observable_errors(replicate, uniform, reference))
    return {k: sum(r[k] for r in replicates) / N_BOOTSTRAP for k in replicates[0]}


def score_runs(run_glob, reference):
    per_seed = {}
    for run_dir in sorted(SOFT_RESULTS.glob(run_glob)):
        eval_dir = run_dir / "eval"
        if not (eval_dir / "samples.pt").exists():
            continue
        x = torch.load(eval_dir / "samples.pt", weights_only=True).float()
        weights = torch.softmax(torch.load(eval_dir / "log_weights.pt", weights_only=True), 0)
        metrics = json.loads((eval_dir / "metrics.json").read_text())
        per_seed[run_dir.name] = {"ESS": metrics["ess_fraction"],
                                  **observable_errors(x, weights, reference)}
    return per_seed


def mean_sd(values):
    t = torch.tensor(values)
    return t.mean().item(), (t.std().item() if len(t) > 1 else float("nan"))


def summarise(per_seed):
    keys = next(iter(per_seed.values())).keys()
    passing = {n: s for n, s in per_seed.items() if s["ESS"] >= ESS_FLOOR}
    return {
        "all": {k: mean_sd([s[k] for s in per_seed.values()]) for k in keys},
        "floor": {k: mean_sd([s[k] for s in passing.values()]) for k in keys} if passing else None,
        "n_pass": len(passing), "n_total": len(per_seed),
    }


def main():
    table = {}
    for (penalty_strength, c_target), run_glob in CELLS.items():
        reference, n_chains, wall_seconds = load_vcsgc_reference(penalty_strength, c_target)
        floor = reference_floor(reference)
        cell = {"reference_floor": floor, "reference_chains": n_chains,
                "reference_frames": reference.shape[0], "reference_wall_seconds": wall_seconds}
        families = {"specialist": run_glob, **{
            "fixed_lambda": g for (lam, c), g in FIXED_LAMBDA_CELLS.items()
            if (lam, c) == (penalty_strength, c_target)}}
        print(f"\n== lambda={penalty_strength} c={c_target} ({n_chains} reference chains, {reference.shape[0]} frames)")
        print("  reference floor:", {k: f"{v:.2e}" for k, v in floor.items()})
        for family, glob in families.items():
            per_seed = score_runs(glob, reference)
            if not per_seed:
                print(f"  [{family}] no runs match {glob}")
                continue
            summary = summarise(per_seed)
            cell[family] = {"per_seed": per_seed, **summary}
            for name, s in per_seed.items():
                print(f"  [{family}] {name}: " + " ".join(f"{k}={v:.4g}" for k, v in s.items()))
            for rule in ("all", "floor"):
                if summary[rule]:
                    print(f"  [{family}] {rule:5s} mean +- SD "
                          f"({summary['n_pass']}/{summary['n_total']} clear {ESS_FLOOR}):",
                          {k: f"{m:.4g} +- {sd:.2g}" for k, (m, sd) in summary[rule].items()})
        table[f"lambda{penalty_strength}_c{c_target:.2f}"] = cell

    out = SOFT_RESULTS / "house_table_soft_10x10.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
