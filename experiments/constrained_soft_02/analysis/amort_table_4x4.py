"""tab:amort-4x4 fill: conditioned vs specialist per composition, one currency.

Re-pointed in the soft-chapter revamp (2026-08-30, wave 3): the offset/clip
family retired, so every row comes from the house-recipe rerun (`*_house`,
tag 20260831-softhouse-d16, fixed lambda=50 + exact field channel + EMA dual
eval) and the composition axis moves to the revamp grid. Structure unchanged:
every requested composition gets a specialist comparator row, and the
hardware-paired timing columns stay replaced by the house FLOP/es currency, so
this table prices what tab:eval-soft-8x8 prices.

The requested set is the five specialist compositions
{0.25, 0.375, 0.50, 0.625, 0.75}. Above 0.50 no specialist was trained
(0.75-as-trained would duplicate 0.25 under the Z2 mirror), so those rows are
the trained 1-c twins read through delivered(c) = 1 - delivered(1-c); ESS and
FLOP/es carry over unchanged (the mirror relabels the same distribution) and
the JSON marks `mirror_of`. The four held-out compositions are not rows here:
this table is amortised-vs-specialist at matched compute and a held-out point
has no comparator -- interpolation is the slope table's claim
(obedience_slope_table, scored against the revamp-grid exact reference
0.9950, never the archived 0.976).

Rows assembled from artefacts on disk, per --eval_dir (house dual-eval
convention: eval/ = raw final weights, eval_ema/ = the EMA shadow; the
output JSON is suffixed _ema for the shadow so neither state overwrites
the other):

  conditioned -- `composition_sweep.json` per amortised seed
            (dnfs_baseline_01.run.composition_sweep on the matching
            checkpoint), the row at each requested composition. FLOP/es =
            the conditioned model's own measured forward (bound through
            the CompositionConditioned adapter, which is how sampling
            actually calls it) x n_euler / that row's ESS fraction.
  specialist -- `metrics.json` of the per-composition house specialists
            (S2_d4_c{0250,0375,0500}_50k_l50_letf_house; >0.50 via the
            mirror as above).
  null    -- the zero-width-window conditioned house cell at c=0.50, as
            before (prices the conditioning machinery; no FLOP/es of its
            own is quoted -- it exists for the ESS subtraction).
  reference -- the D=4 mchammer VC-SGC chains at kappa=lambda per
            composition, measured directly at all five compositions
            (chains are CPU-seconds at D=4, so the reference column needs
            no mirror), 3-decimal dir naming: the same per-trial constant
            and tau_int construction as the house tables, so "FLOP/es"
            means one thing across all of them.

Delivered composition and ESS print as seed mean +- SD over seeds 42-45,
every seed included (the table's declared convention: every seed trained,
not the survivors).
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from discrete_flow_sampler.diagnostics.flops import (
    chain_per_effective_sample,
    measured_forward_flops,
    neural_sampling_flops_per_sample,
    per_effective_sample,
    vcsgc_run_flops,
)
from discrete_flow_sampler.models.composition_conditioned import CompositionConditioned
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
VCSGC_RESULTS = REPO_ROOT / "results" / "mchammer_vcsgc"

D_SIDE, SIGMA, PENALTY = 4, 0.1, 50.0
N_SITES = D_SIDE * D_SIDE
REQUESTED = [0.25, 0.375, 0.50, 0.625, 0.75]
SEEDS = [42, 43, 44, 45]

AMORTISED_CELL = "S2_d4_camort_50k_l50_letf_house"
NULL_CELL = "S2_d4_cnull_50k_l50_letf_house"
SPECIALIST_CELLS = {
    0.25: "S2_d4_c0250_50k_l50_letf_house",
    0.375: "S2_d4_c0375_50k_l50_letf_house",
    0.50: "S2_d4_c0500_50k_l50_letf_house",
}
# Requested composition -> the trained twin whose mirror stands in for it.
MIRRORED_SPECIALISTS = {0.625: 0.375, 0.75: 0.25}

TARGET = IsingTarget(D=D_SIDE, sigma=SIGMA, bias=0.0)


def latest_run_dir(cell: str, seed: int, eval_dir: str) -> Path:
    matches = sorted(RESULTS.glob(f"{cell}_seed{seed}_*"))
    matches = [m for m in matches if (m / eval_dir).exists()]
    if not matches:
        raise FileNotFoundError(f"{cell} seed{seed}: no run dir with {eval_dir}/")
    return matches[-1]


def forward_flops(run_dir: Path) -> int:
    """Measured eager forward at the run's architecture, batch 1.

    A conditioned model is measured through the CompositionConditioned
    adapter -- the (x, t) call path sampling uses -- so the conditioning
    embedding's cost is included.
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
        neural_sampling_flops_per_sample(per_forward, n_euler, N_SITES), ess_fraction
    )


def reference_flops_per_es(c_target: float) -> float:
    """Same construction as the house tables' reference cost cell."""
    total_trials, n_frames, tau_ints = 0, 0, []
    pattern = f"D{D_SIDE}_s{SIGMA}_l{PENALTY:.1f}_c{c_target:.3f}_seed*"
    run_dirs = sorted(VCSGC_RESULTS.glob(pattern))
    if not run_dirs:
        raise FileNotFoundError(f"no VC-SGC chains match {pattern}")
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        total_trials += summary["n_steps"]
        n_frames += summary["observables"]["composition"]["n_frames"]
        tau_ints.append(
            max(obs["tau_int_frames"] for obs in summary["observables"].values())
        )
    tau_int = max(sum(tau_ints) / len(tau_ints), 1.0)
    return chain_per_effective_sample(vcsgc_run_flops(total_trials), n_frames, tau_int)


def mean_sd(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values)
    return t.mean().item(), t.std().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", choices=["eval", "eval_ema"], default="eval")
    args = parser.parse_args()
    eval_dir = args.eval_dir

    table = {}

    # --- conditioned rows, one per requested composition -------------------
    amort_dirs = [latest_run_dir(AMORTISED_CELL, s, eval_dir) for s in SEEDS]
    amort_forward = forward_flops(amort_dirs[0])
    for c in REQUESTED:
        delivered, ess, cost = [], [], []
        for run_dir in amort_dirs:
            sweep = json.loads(
                (run_dir / eval_dir / "composition_sweep.json").read_text()
            )
            row = next(r for r in sweep if abs(r["composition"] - c) < 1e-9)
            delivered.append(row["composition_mean"])
            ess.append(row["ess_fraction"])
            cost.append(flops_per_es(run_dir, amort_forward, row["ess_fraction"]))
        table[(c, "conditioned")] = {
            "delivered": mean_sd(delivered),
            "ess": mean_sd(ess),
            "flops_per_es": mean_sd(cost),
        }

    # --- specialist rows (trained at c, or the Z2 mirror of the 1-c twin) --
    for c in REQUESTED:
        trained_c = MIRRORED_SPECIALISTS.get(c, c)
        cell = SPECIALIST_CELLS[trained_c]
        delivered, ess, cost = [], [], []
        per_forward = None
        for seed in SEEDS:
            run_dir = latest_run_dir(cell, seed, eval_dir)
            if per_forward is None:
                per_forward = forward_flops(run_dir)
            metrics = json.loads((run_dir / eval_dir / "metrics.json").read_text())
            raw_delivered = metrics["composition_mean"]
            delivered.append(1.0 - raw_delivered if trained_c != c else raw_delivered)
            ess.append(metrics["ess_fraction"])
            cost.append(flops_per_es(run_dir, per_forward, metrics["ess_fraction"]))
        table[(c, "specialist")] = {
            "delivered": mean_sd(delivered),
            "ess": mean_sd(ess),
            "flops_per_es": mean_sd(cost),
            **({"mirror_of": trained_c} if trained_c != c else {}),
        }

    # --- null row (machinery control, c = 0.50 only) ------------------------
    delivered, ess = [], []
    for seed in SEEDS:
        metrics = json.loads(
            (
                latest_run_dir(NULL_CELL, seed, eval_dir) / eval_dir / "metrics.json"
            ).read_text()
        )
        delivered.append(metrics["composition_mean"])
        ess.append(metrics["ess_fraction"])
    table[(0.50, "null")] = {
        "delivered": mean_sd(delivered),
        "ess": mean_sd(ess),
        "flops_per_es": None,
    }

    # --- print in table order ----------------------------------------------
    print(f"eval dir: {eval_dir}")
    print(f"conditioned forward: {amort_forward:.3g} FLOPs (adapter-bound)")
    for c in REQUESTED:
        ref = reference_flops_per_es(c)
        print(f"\nrequested c = {c:.3f}   VC-SGC reference FLOP/es {ref:.2g}")
        for sampler in ("conditioned", "specialist", "null"):
            row = table.get((c, sampler))
            if row is None:
                continue
            (dm, ds), (em, es) = row["delivered"], row["ess"]
            cost = row["flops_per_es"]
            cost_str = f"{cost[0]:.2g} +- {cost[1]:.1g}" if cost else "--"
            mirror = (
                f"   (mirror of c={row['mirror_of']:.3f})" if "mirror_of" in row else ""
            )
            print(
                f"  {sampler:11s} delivered {dm:.3f} +- {ds:.3f}   "
                f"ESS {em:.2f} +- {es:.2f}   FLOP/es {cost_str}{mirror}"
            )

    suffix = "_ema" if eval_dir == "eval_ema" else ""
    out = RESULTS / f"amort_table_4x4{suffix}.json"
    out.write_text(
        json.dumps({f"c{c:.3f}_{s}": v for (c, s), v in table.items()}, indent=2)
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
