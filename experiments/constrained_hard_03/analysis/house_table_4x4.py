"""Fill pass for tab:eval-hard-4x4 (the house evaluation table, 4x4 rung).

Reads the Wave-2 4x4 matrix (tag 20260825-hard-w2: five arms x two sigma x
seeds 42/43/44, evals pulled from the Modal volume into results/03_hard/)
and prints the house columns of tab:eval-unconstrained-10x10 for each cell:

  ESS    -- frozen eval/ess_fraction, re-read not recomputed (cells in
            HELD are withheld regardless of what this fill computes for
            them);
  dMag   -- MDNS Eq. 26, dCorr -- MDNS Eq. 28, EW2 -- 1-D W2 on E(x)/d
            (DASBS), each on importance-reweighted samples;
  FLOP/es -- measured eager forward at the run's own architecture (one
            head forward per Euler step, ctmc.n_euler_steps from the saved
            config) plus the counted-but-negligible target evals, divided
            by the frozen ESS (per_effective_sample).

Two structural differences from the 10x10 fills, both consequences of the
4x4 slice being exactly enumerable (C(16,8) = 12,870 states):

  * The reference is the exactly enumerated conditional, not a certified
    chain: slice states + exact Boltzmann probabilities enter the metric
    trio through their `reference_weights` parameter, so the error columns
    read against truth and the reference row has no sampling floor -- its
    error cells are identically zero and the bootstrap floor machinery of
    the 10x10 fills has nothing to estimate.
  * The MCMC row is the materials-side engine itself (mchammer
    CanonicalEnsemble = non-local unlike-pair swap, the same move set as
    the swap CTMC), scored per chain against the exact reference with
    uniform weights over post-burn-in snapshots. Its FLOP/es is the
    analytic kawasaki_run_flops bill over all trials (burn-in is paid
    before the first usable record, mirroring the other chain bills)
    divided by the effective record count n_kept / tau_int, with tau_int
    measured per chain on the energy-per-site series -- the slowest tabled
    observable, so the effective count is the honest one.

Per-site energy follows the chapter's convention E/d = -log p~(x) /
(2 sigma d). Neural cells aggregate mean +- SD over the three seeds; the
chain row over its seed pool.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.constrained_hard_03.analysis.house_table_8x8 import flop_billing_config

from discrete_flow_sampler.diagnostics.flops import (
    chain_per_effective_sample,
    kawasaki_run_flops,
    measured_forward_flops,
    neural_sampling_flops_per_sample,
    per_effective_sample,
)
from discrete_flow_sampler.diagnostics.metrics import (
    conditional_pmf_at_composition,
    correlation_profile_error,
    energy_wasserstein2,
    enumerate_states,
    exact_log_probs,
    integrated_autocorr,
    magnetisation_profile_error,
)

L = 4
D_SITES = L * L
TAG = "20260825-hard-w2"
ARMS = {
    # The O(d^2) architecture-agnostic gate: one fully-masked pair forward
    # per candidate swap. mask_one agrees with it numerically, so its
    # fidelity cells reproduce the mask-one row within seed noise and the
    # information is in FLOP/es.
    "dh": "doubly-hollow oracle",
    "mo": "mask-one head",
    "ma": "masked-attention head",
    # Keep sweep-ladder labels identical to house_table_8x8.ARMS.
    "mamo2": "masked-attention band, two sweeps",
    "mamo2ef": "masked-attention band, two sweeps + exact field",
    "iv": "prefix-sum band, one sweep",
    "ivmo2": "prefix-sum band, two sweeps",
    "ivmo2ef": "prefix-sum band, two sweeps + exact field",
    "thp": "two-hole patch head",
    "mal": "masked-attention head, whole-lattice window",
}
# Recipe suffix and campaign tag overrides. The `mal` window probe
# (2026-08-28) changes only `attention_window="lattice"` from `ma`.
# The five ladder arms share one 30-run campaign tag across both couplings.
# Unlike house_table_8x8.ARM_PROVENANCE, this map needs only the arm key:
# the 8x8 ladder uses separate critical and floor campaign tags.
ARM_PROVENANCE = {
    # The oracle's six runs ran under their own tag; the config is a
    # wave-2 cell, so the suffix stays w2.
    "dh": ("w2", "20260829-dh-oracle-d16"),
    "mal": ("win", "20260828-win-gate"),
    **{
        arm: ("w2", "20260828-rasterord-d16")
        for arm in ("mamo2", "mamo2ef", "iv", "ivmo2", "ivmo2ef")
    },
}
SIGMA_LABELS = ("s010", "s220")
SEEDS = (42, 43, 44)
HELD = set()

# GFlowNet comparator rows: the `_par` cells — parameter parity with the ma
# head (101,378 vs 100,960 params), split lr_Z on the TB arm. These rows do
# not run through CONFIGS/build_target_and_head: the policy is the sampler
# (no Euler grid, no swap head), so the bill is the measured FLOPs of
# `policy.sample` itself — the naive prefix re-encode as implemented, d
# sequential forwards — plus one target eval per sample for the IS weight.
# Draw parity: the GFN eval stored 5000 draws against the swap cells' 512,
# so its rows use the first 512 draws and their weights, with ESS recomputed
# on that window — everywhere else in this file ESS is frozen-not-recomputed.
GFN_ARMS = {
    "gfn_tb": "GFlowNet, trajectory balance",
    "gfn_fldb": "GFlowNet, forward-looking DB",
}
GFN_TAG = "20260830-gfn-d16-par"
GFN_DRAWS = 512


def exact_reference(cfg):
    """Enumerated slice states + exact conditional probabilities, and the
    per-site energies of the slice under the chapter's E/d convention."""
    from experiments.constrained_hard_03.run import build_target_and_head

    target, head = build_target_and_head(cfg, device="cpu")
    all_states = enumerate_states(D_SITES)
    log_pi = exact_log_probs(target, all_states)
    slice_states, log_p_cond = conditional_pmf_at_composition(
        all_states, log_pi, target.n_plus_target
    )
    slice_states = slice_states.float()
    return target, head, slice_states, log_p_cond.exp()


def energy_per_site(target, states):
    return -target.log_prob(states) / (2 * target.sigma * D_SITES)


def neural_cell(run_dir, target, ref_states, ref_probs, per_forward, n_euler):
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    weights = torch.softmax(log_w, dim=0)
    ess = metrics["ess_fraction"]
    flops_raw = neural_sampling_flops_per_sample(per_forward, n_euler, D_SITES)
    return {
        "ESS": ess,
        "dMag": magnetisation_profile_error(
            samples, weights, ref_states, L, reference_weights=ref_probs
        ),
        "dCorr": correlation_profile_error(
            samples, weights, ref_states, L, reference_weights=ref_probs
        ),
        "EW2": energy_wasserstein2(
            energy_per_site(target, samples),
            weights,
            energy_per_site(target, ref_states),
            reference_weights=ref_probs,
        ),
        "FLOP/es": per_effective_sample(flops_raw, ess),
    }


def gfn_cell(run_dir, target, ref_states, ref_probs, flops_per_raw_sample):
    """One GFN seed on the first GFN_DRAWS draws of its stored eval."""
    samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()[
        :GFN_DRAWS
    ]
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)[
        :GFN_DRAWS
    ]
    weights = torch.softmax(log_w, dim=0)
    ess = (1.0 / (weights**2).sum() / GFN_DRAWS).item()
    return {
        "ESS": ess,
        "dMag": magnetisation_profile_error(
            samples, weights, ref_states, L, reference_weights=ref_probs
        ),
        "dCorr": correlation_profile_error(
            samples, weights, ref_states, L, reference_weights=ref_probs
        ),
        "EW2": energy_wasserstein2(
            energy_per_site(target, samples),
            weights,
            energy_per_site(target, ref_states),
            reference_weights=ref_probs,
        ),
        "FLOP/es": per_effective_sample(flops_per_raw_sample, ess),
    }


# Config sigma labels (sigma*100, zero-padded) vs the kawasaki_4x4 npz tags
# (round(sigma*1000)): 0.10 -> s010 vs s100; SIGMA_C -> s220 in both by the
# accident of round(220.343) = 220.
KAWASAKI_TAG = {"s010": "s100", "s220": "s220"}


def kawasaki_cell(
    kawasaki_dir, sigma_label, target, ref_states, ref_probs, burn_in_fraction
):
    """One chain = one 'seed' of the MCMC row; mean +- SD over the pool."""
    rows = []
    npz_tag = KAWASAKI_TAG[sigma_label]
    for npz_path in sorted(kawasaki_dir.glob(f"kawasaki_D{L}_{npz_tag}_seed*.npz")):
        chain = np.load(npz_path)
        spins = torch.from_numpy(chain["spins"]).float()
        burn = int(len(spins) * burn_in_fraction)
        kept = spins[burn:]
        uniform = torch.full((kept.shape[0],), 1.0 / kept.shape[0])
        energies = energy_per_site(target, kept)
        tau = max(1.0, integrated_autocorr(energies.numpy()))
        rows.append(
            {
                "dMag": magnetisation_profile_error(
                    kept, uniform, ref_states, L, reference_weights=ref_probs
                ),
                "dCorr": correlation_profile_error(
                    kept, uniform, ref_states, L, reference_weights=ref_probs
                ),
                "EW2": energy_wasserstein2(
                    energies,
                    uniform,
                    energy_per_site(target, ref_states),
                    reference_weights=ref_probs,
                ),
                "FLOP/es": chain_per_effective_sample(
                    kawasaki_run_flops(int(chain["n_trial_steps"])), kept.shape[0], tau
                ),
                "tau_int_snapshots": tau,
            }
        )
    return rows


def sampling_floor(target, ref_states, ref_probs, n_draws, n_bootstrap=200, seed=0):
    """The error a perfect sampler would still show at the neural cells'
    draw count: n_draws exact multinomial draws from the enumerated
    conditional, scored against it, averaged over bootstrap replicates.
    The reference here is exact, so the floor belongs to the sampler side --
    a neural cell at or below this row is indistinguishable from exact at
    its own N."""
    generator = torch.Generator().manual_seed(seed)
    replicates = []
    for _ in range(n_bootstrap):
        idx = torch.multinomial(
            ref_probs, n_draws, replacement=True, generator=generator
        )
        draws = ref_states[idx]
        uniform = torch.full((n_draws,), 1.0 / n_draws)
        replicates.append(
            {
                "dMag": magnetisation_profile_error(
                    draws, uniform, ref_states, L, reference_weights=ref_probs
                ),
                "dCorr": correlation_profile_error(
                    draws, uniform, ref_states, L, reference_weights=ref_probs
                ),
                "EW2": energy_wasserstein2(
                    energy_per_site(target, draws),
                    uniform,
                    energy_per_site(target, ref_states),
                    reference_weights=ref_probs,
                ),
            }
        )
    return {
        k: (sum(r[k] for r in replicates) / n_bootstrap, 0.0) for k in replicates[0]
    }


def aggregate(rows):
    keys = [k for k in rows[0] if k != "tau_int_snapshots"]
    out = {}
    for key in keys:
        values = torch.tensor([r[key] for r in rows], dtype=torch.float64)
        out[key] = (values.mean().item(), values.std().item())
    return out


def fmt(mean, sd, sci=False):
    return f"{mean:.2e}+-{sd:.1e}" if sci else f"{mean:.3f}+-{sd:.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "03_hard"
    )
    parser.add_argument(
        "--kawasaki-dir",
        type=Path,
        default=REPO_ROOT / "results" / "03_hard" / "kawasaki_w2",
    )
    parser.add_argument("--burn-in-fraction", type=float, default=0.2)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "03_hard" / "w2_4x4_house"
    )
    args = parser.parse_args(argv)

    from experiments.constrained_hard_03.configs import CONFIGS

    table = {}
    for sigma_label in SIGMA_LABELS:
        # One reference per sigma (the slice depends on sigma through the
        # Boltzmann weights, not through the state set).
        cfg_any = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_mo_10k_w2"]
        target, _, ref_states, ref_probs = exact_reference(cfg_any)
        n_draws = cfg_any.eval.n_eval_samples
        floor = sampling_floor(target, ref_states, ref_probs, n_draws)
        table[f"floor{n_draws}_{sigma_label}"] = floor

        for arm in ARMS:
            if arm in ARM_PROVENANCE:
                recipe_suffix, tag = ARM_PROVENANCE[arm]
            else:
                recipe_suffix, tag = "w2", TAG
            cfg = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_{arm}_10k_{recipe_suffix}"]
            # The head carries the FLOP forward, so it is built at the
            # billing config -- separable for a masked-attention head,
            # whatever it trained under (same function, cheaper contraction;
            # see house_table_8x8.flop_billing_config). The reference comes
            # from the target, not the head, so it is unaffected.
            _, head, _, _ = exact_reference(flop_billing_config(cfg))
            example_x = ref_states[:1]
            example_t = torch.full((1,), 0.5)
            per_forward = measured_forward_flops(head, (example_x, example_t))
            rows = []
            for seed in SEEDS:
                run_dir = args.results_dir / f"{cfg.name}_seed{seed}_{tag}"
                rows.append(
                    neural_cell(
                        run_dir,
                        target,
                        ref_states,
                        ref_probs,
                        per_forward,
                        cfg.ctmc.n_euler_steps,
                    )
                )
            cell = aggregate(rows)
            cell["held"] = (arm, sigma_label) in HELD
            # Derive the eager caption marker from the recipe's compile constraint.
            cell["eager"] = not cfg.compile_head
            cell["per_forward_flops"] = per_forward
            table[f"{arm}_{sigma_label}"] = cell

        for gfn_arm, _label in GFN_ARMS.items():
            from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS
            from experiments.constrained_hard_03.run_gfn import build_target_and_policy

            from discrete_flow_sampler.diagnostics.flops import ising_energy_eval_flops

            objective = gfn_arm.removeprefix("gfn_")
            gfn_cfg = GFN_CONFIGS[f"GFN_d16_c50_{sigma_label}_{objective}_10k_par"]
            _, policy = build_target_and_policy(gfn_cfg, "cpu")
            # Bill the sampler as implemented: FlopCounterMode around one
            # draw of policy.sample (d sequential prefix re-encodes), plus
            # the IS-weight target eval.
            flops_per_raw = measured_forward_flops(
                policy.sample, (1,)
            ) + ising_energy_eval_flops(D_SITES)
            rows = []
            for seed in SEEDS:
                run_dir = args.results_dir / f"{gfn_cfg.name}_seed{seed}_{GFN_TAG}"
                rows.append(
                    gfn_cell(run_dir, target, ref_states, ref_probs, flops_per_raw)
                )
            cell = aggregate(rows)
            cell["per_sample_flops"] = flops_per_raw
            cell["n_draws"] = GFN_DRAWS
            table[f"{gfn_arm}_{sigma_label}"] = cell

        if args.kawasaki_dir.exists():
            chain_rows = kawasaki_cell(
                args.kawasaki_dir,
                sigma_label,
                target,
                ref_states,
                ref_probs,
                args.burn_in_fraction,
            )
            if chain_rows:
                cell = aggregate(chain_rows)
                cell["n_chains"] = len(chain_rows)
                table[f"kawasaki_{sigma_label}"] = cell

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "house_table_4x4.json").write_text(json.dumps(table, indent=2))

    print(
        f"{'row':38} {'ESS':>14} {'dMag':>16} {'dCorr':>16} {'EW2':>16} {'FLOP/es':>14}"
    )
    print(
        f"{'exact enumeration (reference)':38} {'/':>14} {'0':>16} "
        f"{'0':>16} {'0':>16} {'--':>14}"
    )
    for key, cell in table.items():
        held = "  [HELD]" if cell.get("held") else ""
        ess = fmt(*cell["ESS"]) if "ESS" in cell else "/"
        flops = fmt(*cell["FLOP/es"], sci=True) if "FLOP/es" in cell else "--"
        print(
            f"{key:38} {ess:>14} {fmt(*cell['dMag']):>16} "
            f"{fmt(*cell['dCorr']):>16} {fmt(*cell['EW2']):>16} "
            f"{flops:>14}{held}"
        )


if __name__ == "__main__":
    main()
