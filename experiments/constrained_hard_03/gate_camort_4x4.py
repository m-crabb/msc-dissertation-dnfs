"""D=4 pass/fail gate for the composition-amortised head.

Validate `H2_d16_camort_s220_letf_thp_10k` on every trained slice
(n+ = 8/7/6/5) against exact enumeration before the d64 GPU run.

Mixture-eval ESS can hide a failed slice, and composition spread is expected
under the mixture. Rebuild the single-slice target at each grid composition
and score against its own enumerated conditional, as in the zero-shot
probe's null rows.

Bars:
  * energy-marginal TV <= 0.02 on every slice — the house 4x4 gate bar;
  * per-slice ESS fraction >= 0.30 on every slice — a filter, not a
    quality claim (the d16 sigma_c thp SPECIALIST reads ~0.99; the gate
    asks "is no slice dead"; quality is the d64 cells' question).
A miss on either bar on any slice fails the gate.

    pixi run -e dev python -m experiments.constrained_hard_03.gate_camort_4x4
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.constrained_hard_03.gate_4x4 import (
    _categorical_energy_bins,
    _energy,
    energy_marginal_tv,
    latest_run_dir,
    load_run,
    slice_energy_hist,
)
from discrete_flow_sampler.diagnostics.metrics import (
    conditional_pmf_at_composition,
    enumerate_states,
    ess_from_log_weights,
    exact_log_probs,
)
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

GATE_CFG = "H2_d16_camort_s220_letf_thp_10k"
TV_BAR = 0.02
ESS_BAR = 0.30


def per_slice_row(head, mixture_target, composition, n_draws, n_euler, device):
    """Score the trained head against ONE slice's enumerated conditional."""
    target = FixedCompositionIsingTarget(
        D=4, sigma=mixture_target.sigma, target_composition=composition,
        device=device,
    )
    all_states = enumerate_states(target.d)
    log_pi = exact_log_probs(target, all_states)
    slice_states, log_p_cond = conditional_pmf_at_composition(
        all_states, log_pi, target.n_plus_target
    )
    slice_states = slice_states.float()

    ts = torch.linspace(0.0, 1.0, n_euler + 1, device=device)
    with torch.no_grad():
        x_initial = target.sample_base(n_draws, device=device)
        samples, log_weights = sample_swap_ctmc(
            head, x_initial, ts, return_log_weights=True, target=target,
        )
    ess = float(ess_from_log_weights(log_weights).item()) / n_draws
    weights = torch.softmax(log_weights, dim=0)

    bins = _categorical_energy_bins(_energy(slice_states, target.A))
    sampled_hist = slice_energy_hist(samples, weights, target.A, bins)
    exact_hist = slice_energy_hist(
        slice_states, log_p_cond.exp(), target.A, bins)
    tv = energy_marginal_tv(sampled_hist, exact_hist)
    return {"composition": composition, "n_plus": target.n_plus_target,
            "ess_fraction": ess, "energy_tv": tv}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/03_hard"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-draws", type=int, default=4096)
    args = parser.parse_args(argv)

    run_dir = latest_run_dir(args.results_dir, GATE_CFG, args.seed)
    print(f"gate run: {run_dir.name}")
    head, mixture_target = load_run(run_dir, device="cpu")
    cfg = json.loads((run_dir / "config.json").read_text())
    n_euler = cfg["ctmc"]["n_euler_steps"]

    seed_everything(0)
    rows, go = [], True
    for composition in mixture_target.compositions:
        row = per_slice_row(head, mixture_target, composition,
                            args.n_draws, n_euler, device="cpu")
        row["pass"] = (row["energy_tv"] <= TV_BAR
                       and row["ess_fraction"] >= ESS_BAR)
        go &= row["pass"]
        rows.append(row)
        print(f"  c={composition:7.5f} (n+={row['n_plus']}): "
              f"ESS {row['ess_fraction']:.3f}  energy-TV {row['energy_tv']:.4f}"
              f"  {'PASS' if row['pass'] else 'FAIL'}")

    verdict = "GO" if go else "NO-GO"
    print(f"{verdict}: bars TV<={TV_BAR}, ESS>={ESS_BAR}, every slice")
    (run_dir / "gate_camort.json").write_text(json.dumps(
        {"rows": rows, "verdict": verdict, "n_draws": args.n_draws},
        indent=2))


if __name__ == "__main__":
    main()
