"""tab:amort-4x4 fill: conditioned vs specialist per composition, one currency.

The s64 refactor of the amortisation table (user decision): every requested
composition gets a SPECIALIST comparator row (the c=0.30/0.70/0.80 twins of
the c=0.50 specialist, byte-identical recipe, Modal tag 20260825-amort-spec),
and both hardware-paired timing columns are replaced by the house FLOP/es
currency, so the table prices exactly what tab:eval-soft-10x10 prices and no
cross-device second appears anywhere.

Rows assembled from artefacts on disk:

  conditioned -- `eval/composition_sweep.json` per amortised seed
            (dnfs_baseline_01.run.composition_sweep), the row at each
            requested composition. FLOP/es = the conditioned model's own
            measured forward (bound through the CompositionConditioned
            adapter, which is how sampling actually calls it) x n_euler /
            that row's ESS fraction.
  specialist -- `eval/metrics.json` of the per-composition specialist cells.
            The c=0.50 family is the archived 2026-08-08 one; the other
            three are its composition twins (test-pinned).
  null    -- the zero-width-window conditioned cell at c=0.50, as before
            (prices the conditioning machinery; no FLOP/es of its own is
            quoted -- it exists for the ESS subtraction).
  reference -- the D=4 mchammer VC-SGC chains at kappa=lambda per
            composition: the same per-trial constant and tau_int
            construction as the 10x10 house table (19_house_table_soft),
            so "FLOP/es" means one thing across both tables.

Delivered composition and ESS print as seed mean +- SD over seeds 42-45,
every seed included (the table's declared convention: every seed trained,
not the survivors).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from discrete_flow_sampler.diagnostics.flops import (
    chain_per_effective_sample, measured_forward_flops,
    neural_sampling_flops_per_sample, per_effective_sample, vcsgc_run_flops)
from discrete_flow_sampler.models.composition_conditioned import (
    CompositionConditioned)
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
VCSGC_RESULTS = REPO_ROOT / "results" / "mchammer_vcsgc"

D_SIDE, SIGMA, PENALTY = 4, 0.1, 50.0
N_SITES = D_SIDE * D_SIDE
REQUESTED = [0.30, 0.50, 0.70, 0.80]
SEEDS = [42, 43, 44, 45]

AMORTISED_CELL = "S2_d4_camort_50k_l50_letf_anneal_offset_clip50"
NULL_CELL = "S2_d4_cnull_50k_l50_letf_anneal_offset_clip50"
SPECIALIST_CELLS = {
    0.30: "S2_d4_c03_50k_l50_letf_anneal_offset_clip50",
    0.50: "S2_d4_c05_50k_l50_letf_anneal_offset_clip50",
    0.70: "S2_d4_c07_50k_l50_letf_anneal_offset_clip50",
    0.80: "S2_d4_c08_50k_l50_letf_anneal_offset_clip50",
}

TARGET = IsingTarget(D=D_SIDE, sigma=SIGMA, bias=0.0)


def latest_run_dir(cell: str, seed: int) -> Path:
    matches = sorted(RESULTS.glob(f"{cell}_seed{seed}_*"))
    matches = [m for m in matches if (m / "eval").exists()]
    if not matches:
        raise FileNotFoundError(f"{cell} seed{seed}: no run dir with eval/")
    return matches[-1]


def forward_flops(run_dir: Path) -> int:
    """Measured eager forward at the run's architecture, batch 1.

    A conditioned model is measured through the CompositionConditioned
    adapter -- the (x, t) call path sampling actually uses -- so the
    conditioning embedding's cost is included, not assumed away.
    """
    from experiments.dnfs_baseline_01.configs import ModelCfg
    from experiments.dnfs_baseline_01.run import _construct_model, _sub_config

    cfg_model = json.loads((run_dir / "config.json").read_text())["model"]
    model_cfg = _sub_config(ModelCfg, {**cfg_model, "compile_model": False})
    model = _construct_model(SimpleNamespace(model=model_cfg), TARGET)
    if cfg_model.get("condition_on_composition"):
        model = CompositionConditioned(model, torch.tensor([0.5]))
    example = (TARGET.sample_base(1, device="cpu"), torch.zeros(1))
    return measured_forward_flops(model, example)


def flops_per_es(run_dir: Path, per_forward: int, ess_fraction: float) -> float:
    n_euler = json.loads((run_dir / "config.json").read_text())["ctmc"]["n_euler_steps"]
    return per_effective_sample(
        neural_sampling_flops_per_sample(per_forward, n_euler, N_SITES),
        ess_fraction)


def reference_flops_per_es(c_target: float) -> float:
    """Same construction as the 10x10 house table's reference cost cell."""
    total_trials, n_frames, tau_ints = 0, 0, []
    pattern = f"D{D_SIDE}_s{SIGMA}_l{PENALTY:.1f}_c{c_target:.2f}_seed*"
    run_dirs = sorted(VCSGC_RESULTS.glob(pattern))
    if not run_dirs:
        raise FileNotFoundError(f"no VC-SGC chains match {pattern}")
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        total_trials += summary["n_steps"]
        n_frames += summary["observables"]["composition"]["n_frames"]
        tau_ints.append(max(obs["tau_int_frames"]
                            for obs in summary["observables"].values()))
    tau_int = max(sum(tau_ints) / len(tau_ints), 1.0)
    return chain_per_effective_sample(
        vcsgc_run_flops(total_trials), n_frames, tau_int)


def mean_sd(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values)
    return t.mean().item(), t.std().item()


def main() -> None:
    table = {}

    # --- conditioned rows, one per requested composition -------------------
    amort_dirs = [latest_run_dir(AMORTISED_CELL, s) for s in SEEDS]
    amort_forward = forward_flops(amort_dirs[0])
    for c in REQUESTED:
        delivered, ess, cost = [], [], []
        for run_dir in amort_dirs:
            sweep = json.loads(
                (run_dir / "eval" / "composition_sweep.json").read_text())
            row = next(r for r in sweep
                       if abs(r["target_composition"] - c) < 1e-9)
            delivered.append(row["composition_mean"])
            ess.append(row["ess_fraction"])
            cost.append(flops_per_es(run_dir, amort_forward, row["ess_fraction"]))
        table[(c, "conditioned")] = {
            "delivered": mean_sd(delivered), "ess": mean_sd(ess),
            "flops_per_es": mean_sd(cost)}

    # --- specialist rows ----------------------------------------------------
    for c, cell in SPECIALIST_CELLS.items():
        delivered, ess, cost = [], [], []
        per_forward = None
        for seed in SEEDS:
            run_dir = latest_run_dir(cell, seed)
            if per_forward is None:
                per_forward = forward_flops(run_dir)
            metrics = json.loads(
                (run_dir / "eval" / "metrics.json").read_text())
            delivered.append(metrics["composition_mean"])
            ess.append(metrics["ess_fraction"])
            cost.append(flops_per_es(run_dir, per_forward,
                                     metrics["ess_fraction"]))
        table[(c, "specialist")] = {
            "delivered": mean_sd(delivered), "ess": mean_sd(ess),
            "flops_per_es": mean_sd(cost)}

    # --- null row (machinery control, c = 0.50 only) ------------------------
    delivered, ess = [], []
    for seed in SEEDS:
        metrics = json.loads(
            (latest_run_dir(NULL_CELL, seed) / "eval" / "metrics.json").read_text())
        delivered.append(metrics["composition_mean"])
        ess.append(metrics["ess_fraction"])
    table[(0.50, "null")] = {"delivered": mean_sd(delivered),
                             "ess": mean_sd(ess), "flops_per_es": None}

    # --- print in table order ----------------------------------------------
    print(f"conditioned forward: {amort_forward:.3g} FLOPs (adapter-bound)")
    for c in REQUESTED:
        ref = reference_flops_per_es(c)
        print(f"\nrequested c = {c:.2f}   VC-SGC reference FLOP/es {ref:.2g}")
        for sampler in ("conditioned", "specialist", "null"):
            row = table.get((c, sampler))
            if row is None:
                continue
            (dm, ds), (em, es) = row["delivered"], row["ess"]
            cost = row["flops_per_es"]
            cost_str = f"{cost[0]:.2g} +- {cost[1]:.1g}" if cost else "--"
            print(f"  {sampler:11s} delivered {dm:.3f} +- {ds:.3f}   "
                  f"ESS {em:.2f} +- {es:.2f}   FLOP/es {cost_str}")

    out = RESULTS / "amort_table_4x4.json"
    out.write_text(json.dumps(
        {f"c{c:.2f}_{s}": v for (c, s), v in table.items()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
