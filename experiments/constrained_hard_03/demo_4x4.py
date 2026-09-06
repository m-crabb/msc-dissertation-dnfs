"""4x4 demo analysis: exact-enumeration fidelity + the N_eff(O) metric for
the 10k MA/MO cells and the mchammer Kawasaki chains.

FRAMING: gate-adjacent validation and demo at the enumerable size -- NOT the
mixing probe itself (probe sizes are 8x8/16x16). It reuses the probe's
metric so the machinery transfers:

    N_eff(O) = Var_pi[O] / MSE(O_hat)

with exact E_pi[O], Var_pi[O] from the enumerated 12,870-state slice (the one
rung where ground truth has zero uncertainty) and MSE over replicate
estimates. Compute currencies stay separate: backbone fwd_stack
ROWS for the neural cells (counted by hook, so mask_one's stacked per-anchor
passes are charged honestly), TRIAL steps = closed-form energy evaluations
for Kawasaki. Never blended.

Two stages:
  gpu   -- (Modal, via modal_app::demo) fidelity via the gate's run_gate +
           neural replicate estimates; writes neural_estimates.json to the
           volume.
  local -- assembles neural_estimates.json + Kawasaki npzs + exact moments
           into the demo tables (CPU only).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.gate_4x4 import (
    _energy,
    latest_run_dir,
    load_run,
    run_gate,
)

from discrete_flow_sampler.diagnostics.metrics import (
    conditional_pmf_at_composition,
    diagonal_correlation,
    enumerate_states,
    ess_from_log_weights,
    exact_log_probs,
    half_magnetisation_order_parameter,
    integrated_autocorr,
    nn_correlation,
)
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.seeding import seed_everything

DEMO_CELLS = [
    "H2_d16_c50_s010_letf_ma_10k",
    "H2_d16_c50_s223_letf_ma_10k",
    "H2_d16_c50_s010_letf_mo_10k",
    "H2_d16_c50_s223_letf_mo_10k",
]
OBSERVABLE_NAMES = ["energy", "nn_correlation", "diagonal_correlation", "phi"]
DEMO_SIGMAS = (0.10, 0.223)


def observable_values(name, states, target):
    """Per-sample values of one probe-set observable."""
    if name == "energy":
        return _energy(states, target.A)
    if name == "nn_correlation":
        return nn_correlation(states, target.A)
    if name == "diagonal_correlation":
        return diagonal_correlation(states, target.D)
    if name == "phi":
        return half_magnetisation_order_parameter(states, target.D)
    raise ValueError(f"unknown observable {name!r}")


def exact_moments(target):
    """Exact E_pi[O], Var_pi[O] for every observable via slice enumeration."""
    all_states = enumerate_states(int(target.d))
    log_pi = exact_log_probs(target, all_states)
    slice_states, log_p_cond = conditional_pmf_at_composition(
        all_states, log_pi, target.n_plus_target
    )
    slice_states = slice_states.float()
    p_cond = log_p_cond.exp()
    moments = {}
    for name in OBSERVABLE_NAMES:
        values = observable_values(name, slice_states, target)
        mean = (p_cond * values).sum()
        var = (p_cond * (values - mean) ** 2).sum()
        moments[name] = (mean.item(), var.item())
    return moments


def n_eff_observable(estimates, exact_mean, exact_var):
    """N_eff(O) = Var_pi[O] / MSE(O_hat), MSE against the EXACT mean over R
    replicate estimates (no bias/variance split needed: truth is enumerated).
    Jackknife-over-replicates standard error, since N_eff is a nonlinear
    functional of the replicate errors."""
    estimates = np.asarray(estimates, dtype=float)
    squared_errors = (estimates - exact_mean) ** 2
    n_replicates = len(squared_errors)
    mse = squared_errors.mean()
    n_eff = exact_var / mse
    leave_one_out_mse = (squared_errors.sum() - squared_errors) / (n_replicates - 1)
    jackknife = exact_var / leave_one_out_mse
    se = math.sqrt(
        (n_replicates - 1) / n_replicates * ((jackknife - jackknife.mean()) ** 2).sum()
    )
    return float(n_eff), float(se)


class BackbonePassRowCounter:
    """Counts rows through the backbone's fwd_stack: the network-pass
    currency. Every body pass (masked, anchor-stacked, or plain) calls
    fwd_stack exactly once with the TRUE stacked row count, so the measure is
    invariant to whether a head runs one pass (masked_attention) or stacks d
    per-anchor passes into the batch dimension (mask_one)."""

    def __init__(self, backbone):
        self.rows = 0
        self._handle = backbone.fwd_stack.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.rows += args[0].shape[0]

    def close(self):
        self._handle.remove()


def neural_cell_estimates(
    results_dir, cfg_name, seeds, n_samples, n_replicates, device
):
    """Per (training seed x eval replicate): IS-weighted observable estimates,
    IS-ESS (secondary diagnostic), and the measured fwd_stack-row cost. Also
    the gate fidelity block per training seed (run_gate reused verbatim)."""
    cfg = CONFIGS[cfg_name]
    n_euler_steps = cfg.ctmc.n_euler_steps
    replicate_rows = []
    fidelity = []
    for seed in seeds:
        run_dir = latest_run_dir(results_dir, cfg_name, seed)
        print(f"[demo] {cfg_name} seed {seed}: {run_dir.name}", flush=True)
        head, target = load_run(run_dir, device)
        fidelity_metrics = run_gate(head, target, n_samples, n_euler_steps, seed)
        fidelity.append(
            {
                "seed": seed,
                "energy_tv": fidelity_metrics["energy_tv"],
                "ess_fraction": fidelity_metrics["ess_fraction"],
                "max_level_excess": fidelity_metrics["max_level_excess"],
                "antisym_violation": fidelity_metrics["antisym_violation"],
                "energy_centres": fidelity_metrics["energy_centres"],
                "hist_dnfs": fidelity_metrics["hist_dnfs"],
                "hist_exact": fidelity_metrics["hist_exact"],
            }
        )
        for replicate in range(n_replicates):
            replicate_seed = 10_000 + 100 * seed + replicate
            counter = BackbonePassRowCounter(head.backbone)
            with torch.no_grad():
                seed_everything(replicate_seed)
                x0 = target.sample_base(n_samples, device=device)
                ts = torch.linspace(0.0, 1.0, n_euler_steps + 1, device=device)
                samples, log_w = sample_swap_ctmc(
                    head, x0, ts, return_log_weights=True, target=target
                )
            counter.close()
            weights = torch.softmax(log_w, dim=0)
            estimates = {
                name: (weights * observable_values(name, samples, target)).sum().item()
                for name in OBSERVABLE_NAMES
            }
            replicate_rows.append(
                {
                    "seed": seed,
                    "replicate": replicate,
                    "estimates": estimates,
                    "backbone_rows": counter.rows,
                    "is_ess_fraction": (ess_from_log_weights(log_w).item() / n_samples),
                }
            )
    return {
        "cfg": cfg_name,
        "sigma": float(cfg.ising.sigma),
        "head_kind": cfg.head_kind,
        "n_samples": n_samples,
        "replicates": replicate_rows,
        "fidelity": fidelity,
    }


def gpu_stage(results_dir, out_path, seeds, n_samples, n_replicates, device):
    payload = [
        neural_cell_estimates(
            results_dir, cfg_name, seeds, n_samples, n_replicates, device
        )
        for cfg_name in DEMO_CELLS
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[demo] wrote {out_path}", flush=True)


def phi_support(D):
    """Exact support of phi on the 50/50 slice: sum_L = -sum_R forces
    phi = (m_L - m_R)/2 = sum_L / (D*D/2), with sum_L over D*D/2 sites of
    +/-1 taking even values in [-D*D/2, D*D/2] -- D*D/2 + 1 points."""
    half = D * D // 2
    return np.arange(-half, half + 1, 2) / half


def phi_mass_on_support(phi_values, weights, D):
    """Weighted mass of per-sample phi values on the exact support grid."""
    half = D * D // 2
    indices = np.rint((np.asarray(phi_values) * half + half) / 2).astype(int)
    return np.bincount(indices, weights=np.asarray(weights), minlength=half + 1)


def exact_phi_pmf(target):
    """Exact pmf of phi under the enumerated conditional."""
    all_states = enumerate_states(int(target.d))
    log_pi = exact_log_probs(target, all_states)
    slice_states, log_p_cond = conditional_pmf_at_composition(
        all_states, log_pi, target.n_plus_target
    )
    phi = observable_values("phi", slice_states.float(), target).numpy()
    return phi_mass_on_support(phi, log_p_cond.exp().numpy(), int(target.D))


def phi_hist_stage(results_dir, out_path, seeds, n_samples, device):
    """Light GPU stage: pooled IS-weighted phi histogram per demo cell (the
    mode-coverage exhibit -- the earlier gpu stage kept only scalar means).
    One fresh 5000-draw pass per (cell, seed), pooled across seeds."""
    payload = []
    for cfg_name in DEMO_CELLS:
        cfg = CONFIGS[cfg_name]
        n_euler_steps = cfg.ctmc.n_euler_steps
        D = int(cfg.ising.D)
        mass = np.zeros(D * D // 2 + 1)
        for seed in seeds:
            run_dir = latest_run_dir(results_dir, cfg_name, seed)
            print(f"[demo/phi] {cfg_name} seed {seed}", flush=True)
            head, target = load_run(run_dir, device)
            with torch.no_grad():
                seed_everything(20_000 + seed)
                x0 = target.sample_base(n_samples, device=device)
                ts = torch.linspace(0.0, 1.0, n_euler_steps + 1, device=device)
                samples, log_w = sample_swap_ctmc(
                    head, x0, ts, return_log_weights=True, target=target
                )
            weights = torch.softmax(log_w, dim=0).cpu().numpy()
            phi = observable_values("phi", samples, target).cpu().numpy()
            mass += phi_mass_on_support(phi, weights, D)
        mass /= len(seeds)
        payload.append(
            {
                "cfg": cfg_name,
                "sigma": float(cfg.ising.sigma),
                "head_kind": cfg.head_kind,
                "phi_support": phi_support(D).tolist(),
                "phi_mass": mass.tolist(),
            }
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[demo/phi] wrote {out_path}", flush=True)


def kawasaki_phi_mass(kawasaki_dir, sigma, target, burn_in_trials):
    """Pooled unweighted phi histogram over the post-burn-in snapshots of
    every chain at this sigma (chains equal-weighted)."""
    D = int(target.D)
    mass = np.zeros(D * D // 2 + 1)
    n_chains = 0
    for npz_path in sorted(Path(kawasaki_dir).glob("*.npz")):
        data = np.load(npz_path)
        if abs(float(data["sigma"]) - sigma) > 1e-9:
            continue
        spins = torch.tensor(data["spins"]).float()
        kept = torch.tensor(data["mctrial"] > burn_in_trials)
        phi = observable_values("phi", spins[kept], target).numpy()
        mass += phi_mass_on_support(phi, np.full(len(phi), 1.0 / len(phi)), D)
        n_chains += 1
    return mass / n_chains


def kawasaki_cell_estimates(kawasaki_dir, sigma, target, burn_in_trials):
    """Per chain: post-burn-in snapshot-mean estimates; cost = TOTAL trial
    steps (burn-in charged -- Kawasaki pays it in real use). Chains are
    selected by the sigma stored IN each npz, not by filename tag. tau_int on
    the energy series is a secondary diagnostic, converted snapshot -> trial
    units explicitly (the analyze_data trial-step gotcha applies to any
    snapshot-indexed autocorrelation, including our integrated_autocorr)."""
    rows = []
    for npz_path in sorted(Path(kawasaki_dir).glob("*.npz")):
        data = np.load(npz_path)
        if abs(float(data["sigma"]) - sigma) > 1e-9:
            continue
        spins = torch.tensor(data["spins"]).float()
        kept = torch.tensor(data["mctrial"] > burn_in_trials)
        kept_spins = spins[kept]
        estimates = {
            name: observable_values(name, kept_spins, target).mean().item()
            for name in OBSERVABLE_NAMES
        }
        energy_series = observable_values("energy", kept_spins, target).numpy()
        tau_int_snapshots = integrated_autocorr(energy_series)
        rows.append(
            {
                "seed": int(data["seed"]),
                "estimates": estimates,
                "energy_evals": int(data["n_trial_steps"]),
                "tau_int_trial_steps": (
                    tau_int_snapshots * float(data["snapshot_interval"])
                ),
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"no Kawasaki npz with sigma={sigma} under {kawasaki_dir}"
        )
    return rows


def assemble(neural_json, kawasaki_dir, out_dir, burn_in_trials=20_000):
    """Local stage: exact moments + neural estimates + Kawasaki chains ->
    N_eff(O) tables, one row per (sampler, observable), each row carrying its
    own compute currency."""
    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

    neural = json.loads(Path(neural_json).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for sigma in DEMO_SIGMAS:
        target = FixedCompositionIsingTarget(
            D=4, sigma=sigma, target_composition=0.5, bias=0.0, device="cpu"
        )
        moments = exact_moments(target)
        samplers = {
            cell["cfg"]: {
                "kind": cell["head_kind"],
                "currency": "backbone_rows",
                "rows": cell["replicates"],
            }
            for cell in neural
            if abs(cell["sigma"] - sigma) < 1e-9
        }
        samplers[f"kawasaki_sigma{sigma}"] = {
            "kind": "kawasaki_mchammer",
            "currency": "energy_evals",
            "rows": kawasaki_cell_estimates(
                kawasaki_dir, sigma, target, burn_in_trials
            ),
        }
        for sampler_name, sampler in samplers.items():
            cost_key = (
                "backbone_rows"
                if sampler["currency"] == "backbone_rows"
                else "energy_evals"
            )
            mean_cost = float(np.mean([r[cost_key] for r in sampler["rows"]]))
            for name in OBSERVABLE_NAMES:
                exact_mean, exact_var = moments[name]
                estimates = [r["estimates"][name] for r in sampler["rows"]]
                n_eff, se = n_eff_observable(estimates, exact_mean, exact_var)
                table.append(
                    {
                        "sigma": sigma,
                        "sampler": sampler_name,
                        "head_kind": sampler["kind"],
                        "observable": name,
                        "exact_mean": exact_mean,
                        "exact_var": exact_var,
                        "estimate_mean": float(np.mean(estimates)),
                        "n_replicates": len(estimates),
                        "n_eff": n_eff,
                        "n_eff_se": se,
                        "currency": sampler["currency"],
                        "mean_cost": mean_cost,
                        "n_eff_per_1e6": n_eff / (mean_cost / 1e6),
                    }
                )
    (out_dir / "neff_table.json").write_text(json.dumps(table, indent=2))
    _write_markdown_tables(table, neural, out_dir)
    print(f"[demo] wrote {out_dir}/neff_table.json + markdown tables", flush=True)


def _write_markdown_tables(table, neural, out_dir):
    lines = [
        "| sigma | sampler | observable | exact | estimate | N_eff +/- SE "
        "| currency | cost | N_eff / 1e6 units |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(
            f"| {row['sigma']} | {row['sampler']} | {row['observable']} "
            f"| {row['exact_mean']:.4f} | {row['estimate_mean']:.4f} "
            f"| {row['n_eff']:.1f} +/- {row['n_eff_se']:.1f} "
            f"| {row['currency']} | {row['mean_cost']:.4g} "
            f"| {row['n_eff_per_1e6']:.4g} |"
        )
    (out_dir / "neff_table.md").write_text("\n".join(lines) + "\n")

    fidelity_lines = [
        "| cell | seed | energy_tv | ess_frac | max_level_excess |",
        "|---|---|---|---|---|",
    ]
    for cell in neural:
        for f in cell["fidelity"]:
            fidelity_lines.append(
                f"| {cell['cfg']} | {f['seed']} | {f['energy_tv']:.4f} "
                f"| {f['ess_fraction']:.3f} | {f['max_level_excess']:.4f} |"
            )
    (out_dir / "observables_table.md").write_text("\n".join(fidelity_lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["gpu", "phi", "local"])
    parser.add_argument("--results-dir", default="results/03_hard")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--neural-json", default="results/03_hard/demo_4x4/neural_estimates.json"
    )
    parser.add_argument("--kawasaki-dir", default="results/03_hard/demo_4x4/kawasaki")
    parser.add_argument("--out", default="results/03_hard/demo_4x4")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.stage == "gpu":
        gpu_stage(
            args.results_dir,
            Path(args.out) / "neural_estimates.json",
            seeds,
            args.n_samples,
            args.n_replicates,
            args.device,
        )
    elif args.stage == "phi":
        phi_hist_stage(
            args.results_dir,
            Path(args.out) / "phi_hists.json",
            seeds,
            args.n_samples,
            args.device,
        )
    else:
        assemble(args.neural_json, args.kawasaki_dir, args.out)


if __name__ == "__main__":
    main()
