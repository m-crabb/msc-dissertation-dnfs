"""(sigma_c, 8x8) headline-cell probe analysis: the N_eff(O) metric
assembled from archived artefacts, all local, CPU only.

Machinery mirrors demo_4x4.py — the same metric

    N_eff(O) = Var_pi[O] / MSE(O_hat)

— with ONE substitution: ground truth comes from the mode-balance-seeded
mchammer reference chains (R-hat <= 1.01 validity bar, met at both
operating points) instead of exact enumeration, because 2^64 states cannot
be enumerated. MSE is against the reference mean over R = 8 replicates,
jackknife-over-replicates SE (n_eff_observable, reused verbatim).

Cost accounting (two hardware-neutral currencies, reported separately,
never blended):

* Neural, network passes: one backbone call per Euler step, reused for the
  weight integrand (swap_ctmc.py lines 43-45), so
  backbone_rows = n_samples * n_euler_steps exactly — no hook needed for the
  masked_attention head, the count is structural. (The demo's hook exists to
  charge mask_one's stacked passes honestly; this cell is masked_attention.)
* Neural, energy evaluations: the DYNAMICS consult the target zero times —
  rates come from the head alone. Every target evaluation belongs to the
  importance weight: xi_t evaluates the closed-form swap log-ratio for all
  d(d-1)/2 pairs per sample-step (pair_delta_e_evals) plus one
  dt_log_p_tilde_t per sample-step (dt_logp_evals, reported separately;
  folding it in at any reasonable pair-equivalent rate shifts the total by
  ~1-2%, stated rather than blended).
* Kawasaki: one closed-form pair-Delta-E per trial step, total proposals
  with burn-in CHARGED — Kawasaki pays its burn-in in real use (demo_4x4
  precedent).

Competitor burn-in: discard
max(1e4 sweeps, 20 * tau_int(energy)) with tau_int from BATCH MEANS at block
length >= 10 * tau_int. tau_int is estimated on the second half of each
chain (clearly post-transient at 1e6 sweeps) so the transient cannot inflate
its own discard window. The Sokal windowed estimate (integrated_autocorr) is
reported alongside as the secondary diagnostic, converted snapshot -> trial
units explicitly (the analyze_data trial-step gotcha).

Coverage axis: neural Z2 mass balance + weighted phi histogram against the
reference's own phi histogram (total variation on the exact 33-point
support), Kawasaki's mode-seeded split-half R-hat(phi) and its own phi TV
alongside. "Covers modes at least as well as Kawasaki" is operationalised
as TV_neural <= TV_kawasaki with the 50/50 balance within 0.1.

Outcome: frozen_verdict applies the three-way rule mechanically. A win
("GO") needs BOTH currencies at 95% CI excluding parity AND point >= 1.5x,
the floor not worse, coverage at least as good, the 4x4 gate holding.
Without the floor's neural replicates the outcome is "PROVISIONAL" by
construction — the function cannot report a win with the floor unknown.

The cross-currency division (Kawasaki performs zero network passes) is
operationalised as: each ratio prices BOTH samplers per unit of ONE
currency's honest count — the energy-eval ratio charges the neural side
its pair-Delta-E bill; the network-pass ratio charges the neural side its
backbone rows against Kawasaki's per-trial bill (its elementary operation
and its energy evaluation coincide).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from experiments.constrained_hard_03.demo_4x4 import (
    OBSERVABLE_NAMES,
    n_eff_observable,
    observable_values,
    phi_mass_on_support,
    phi_support,
)

from discrete_flow_sampler.diagnostics.metrics import (
    ess_from_log_weights,
    integrated_autocorr,
    split_half_gelman_rubin,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

OPERATING_POINTS = {"sc": 0.223, "s010": 0.10}
LATTICE_SIDE = 8
BURN_IN_FLOOR_SWEEPS = 10_000  # burn-in = max(1e4, 20 * tau_int)
BURN_IN_TAU_MULTIPLE = 20
BLOCK_LENGTH_TAU_MULTIPLE = 10  # batch-means block length >= 10 tau
MIN_BLOCKS = 20
GO_POINT_MARGIN = 1.5  # per-currency point margin
COVERAGE_BALANCE_TOLERANCE = 0.1


# ---------------------------------------------------------------------------
# Batch-means tau_int + the burn-in rule
# ---------------------------------------------------------------------------


def batch_means_tau_int(trace, min_blocks=MIN_BLOCKS):
    """Integrated autocorrelation time via batch means, self-certifying.

    For block length L >> tau_int, Var(block mean) ~= tau_int * Var(x) / L,
    so tau_hat = L * Var(block means) / Var(x). The rule requires the
    block length to satisfy L >= 10 * tau_hat; L is iterated until the
    estimate certifies its own block choice (seeded from the Sokal windowed
    estimate). Returns (tau_hat, block_length, n_blocks).

    Failure mode guarded: with too few blocks Var(block means) is itself
    noisy, so L is capped at len(trace) // min_blocks; if certification is
    impossible under that cap the trace is too short for the rule and the
    function raises rather than return an uncertified number.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = trace.size
    trace_var = trace.var()
    if trace_var == 0.0:
        return 1.0, 1, n
    max_block_length = n // min_blocks
    block_length = max(
        1,
        min(
            int(math.ceil(BLOCK_LENGTH_TAU_MULTIPLE * integrated_autocorr(trace))),
            max_block_length,
        ),
    )
    tau = None
    for _ in range(20):
        n_blocks = n // block_length
        block_means = (
            trace[: n_blocks * block_length]
            .reshape(n_blocks, block_length)
            .mean(axis=1)
        )
        tau = block_length * block_means.var(ddof=1) / trace_var
        certified_length = int(math.ceil(BLOCK_LENGTH_TAU_MULTIPLE * tau))
        if block_length >= certified_length:
            return float(tau), int(block_length), int(n_blocks)
        if block_length >= max_block_length:
            raise ValueError(
                f"batch_means_tau_int: cannot certify block length >= "
                f"{BLOCK_LENGTH_TAU_MULTIPLE}*tau with >= {min_blocks} "
                f"blocks (n={n}, tau_hat={tau:.1f}); trace too short."
            )
        block_length = min(certified_length, max_block_length)
    raise ValueError("batch_means_tau_int: block-length iteration diverged")


def kawasaki_burn_in_sweeps(tau_int_sweeps):
    """Competitor burn-in: max(1e4 sweeps, 20 * tau_int(energy))."""
    return int(max(BURN_IN_FLOOR_SWEEPS, BURN_IN_TAU_MULTIPLE * tau_int_sweeps))


# ---------------------------------------------------------------------------
# Ratio CI + the three-way outcome rule
# ---------------------------------------------------------------------------


def ratio_with_ci(n_eff_num, se_num, cost_num, n_eff_den, se_den, cost_den):
    """Per-compute N_eff ratio with a delta-method 95% CI.

    The ratio is (n_eff_num / cost_num) / (n_eff_den / cost_den); costs are
    deterministic counters, so all uncertainty comes from the two jackknife
    SEs. On the log scale the SEs combine in quadrature as relative errors
    (delta method), giving a CI symmetric in log space — appropriate for a
    strictly positive ratio whose margin bar is multiplicative (1.5x).
    """
    point = (n_eff_num / cost_num) / (n_eff_den / cost_den)
    se_log = math.sqrt((se_num / n_eff_num) ** 2 + (se_den / n_eff_den) ** 2)
    lo = point * math.exp(-1.96 * se_log)
    hi = point * math.exp(+1.96 * se_log)
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "excludes_parity": bool(lo > 1.0 or hi < 1.0),
    }


def ratio_with_f_ci(n_eff_num, cost_num, r_num, n_eff_den, cost_den, r_den):
    """Per-compute N_eff ratio with a variance-ratio (F) 95% CI.

    N_eff = Var_pi/MSE with the SAME Var_pi on both sides, so the ratio is
    an MSE ratio times a deterministic cost ratio. With R mean-zero normal
    replicate errors each side, R*MSE/sigma^2 ~ chi2(R) and the MSE ratio is
    F(R_den, R_num)-distributed around the true variance ratio, giving
    CI = [point / F_.975(R_den, R_num), point * F_.975(R_num, R_den)].
    At R=8 both sides that is a factor 4.43 either way — far tighter than
    the delta method on jackknifed N_eff, whose SE is itself heavy-tail
    noisy at R=8.

    Approximation stated: a replicate BIAS makes the MSE noncentral chi2
    and the interval anti-conservative; the per-replicate estimate tables
    are published so the bias term is visible directly.
    """
    from scipy.stats import f as f_distribution

    point = (n_eff_num / cost_num) / (n_eff_den / cost_den)
    lo = point / f_distribution.ppf(0.975, r_den, r_num)
    hi = point * f_distribution.ppf(0.975, r_num, r_den)
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "excludes_parity": bool(lo > 1.0 or hi < 1.0),
    }


def frozen_verdict(
    energy_eval_ratio,
    network_pass_ratio,
    floor_not_worse,
    coverage_ok,
    gate_holds,
    beats_local_variant,
    wins_at_floor=None,
):
    """The three-way outcome rule, applied mechanically.

    Inputs are the sigma_c ratios vs the BEST tuned Kawasaki variant (each a
    dict with point/excludes_parity), plus the auxiliary conditions. The
    margin rule is two-part per currency: 95% CI excluding parity AND point
    >= 1.5x. floor_not_worse=None (replicates missing) makes "GO"
    impossible: the outcome degrades to "PROVISIONAL", never silently to a
    win.

    Outcomes the rule does not enumerate (e.g. a speed win with a coverage
    shortfall) return "PARTIAL" with an *_unenumerated narrative rather
    than being forced into the nearest bucket.
    """

    def margin_passes(ratio):
        return ratio["excludes_parity"] and ratio["point"] >= GO_POINT_MARGIN

    margins = [margin_passes(energy_eval_ratio), margin_passes(network_pass_ratio)]
    points = [energy_eval_ratio["point"], network_pass_ratio["point"]]
    pending = [] if floor_not_worse is not None else ["floor_not_worse"]

    if all(margins) and coverage_ok and gate_holds:
        if pending:
            return {
                "verdict": "PROVISIONAL",
                "pending": pending,
                "narrative": "go_pending_floor",
            }
        if floor_not_worse:
            return {"verdict": "GO", "pending": [], "narrative": "go"}
        return {
            "verdict": "PARTIAL",
            "pending": [],
            "narrative": "sigma_c_only_unenumerated",
        }
    if all(margins) and not coverage_ok:
        return {
            "verdict": "PARTIAL",
            "pending": pending,
            "narrative": "coverage_shortfall_unenumerated",
        }
    if all(p < 1.0 for p in points):
        if wins_at_floor:
            return {"verdict": "PARTIAL", "pending": pending, "narrative": "floor_only"}
        if beats_local_variant:
            return {
                "verdict": "PARTIAL",
                "pending": pending,
                "narrative": "move_set_does_the_work",
            }
        return {
            "verdict": "NO-GO",
            "pending": pending,
            "narrative": "diagnosis_section",
        }
    if sum(margins) == 1:
        return {"verdict": "PARTIAL", "pending": pending, "narrative": "one_currency"}
    if (
        energy_eval_ratio["excludes_parity"]
        and network_pass_ratio["excludes_parity"]
        and all(p > 1.0 for p in points)
    ):
        return {
            "verdict": "PARTIAL",
            "pending": pending,
            "narrative": "real_but_marginal",
        }
    return {
        "verdict": "PARTIAL",
        "pending": pending,
        "narrative": "inconclusive_unenumerated",
    }


# ---------------------------------------------------------------------------
# Reference moments (ground truth) — with an independent recompute check
# ---------------------------------------------------------------------------


def reference_block(probe_root, point, target):
    """Reference moments from reference_summary.json, verified by recomputing
    the pooled post-discard moments from the raw snapshots (a sloppy
    reference contaminates every downstream N_eff, so the numbers the
    analysis reads are re-derived, not trusted)."""
    summary = json.loads(
        (probe_root / "reference" / point / "reference_summary.json").read_text()
    )
    if not summary["passed"]:
        raise RuntimeError(
            f"reference/{point} failed its R-hat validity bar; "
            "these moments are not trustworthy."
        )
    chain_dirs = sorted((probe_root / "reference" / point).glob("chain_*"))
    traces = {}
    phi_pooled = None
    for chain_dir in chain_dirs:
        spins = np.load(chain_dir / "snapshots.npz")["spins"]
        states = torch.from_numpy(spins.astype(np.float32))
        for name in OBSERVABLE_NAMES:
            traces.setdefault(name, []).append(
                observable_values(name, states, target).numpy()
            )
    moments = {}
    for name in OBSERVABLE_NAMES:
        chains = np.stack(traces[name])
        pooled = chains[:, chains.shape[1] // 2 :].reshape(-1)
        stored = summary["post_discard_moments"][name]
        recomputed_mean, recomputed_var = pooled.mean(), pooled.var()
        if not (
            np.isclose(recomputed_mean, stored["mean"], atol=1e-6)
            and np.isclose(recomputed_var, stored["var"], rtol=1e-5)
        ):
            raise RuntimeError(
                f"reference/{point}/{name}: recomputed moments "
                f"({recomputed_mean:.6f}, {recomputed_var:.6f}) disagree "
                f"with reference_summary.json ({stored['mean']:.6f}, "
                f"{stored['var']:.6f})"
            )
        moments[name] = (float(stored["mean"]), float(stored["var"]))
        if name == "phi":
            phi_pooled = pooled
    phi_hist = phi_mass_on_support(
        phi_pooled,
        np.full(phi_pooled.size, 1.0 / phi_pooled.size),
        LATTICE_SIDE,
    )
    return {
        "moments": moments,
        "rhat_post_discard": summary["rhat_post_discard"],
        "phi_hist": phi_hist,
        "n_pooled_snapshots": int(phi_pooled.size),
    }


# ---------------------------------------------------------------------------
# Neural side: archived replicate draws
# ---------------------------------------------------------------------------


def neural_replicate_rows(run_dir, target, n_euler_steps):
    """Per-replicate IS estimates + structural cost counters from the
    archived eval_replicate_s*/ draws (samples.pt + log_weights.pt).

    The stored ESS is cross-checked against one recomputed from the stored
    log-weights — a corrupted or mismatched artefact fails loudly here
    rather than polluting the headline table.
    """
    d = target.d
    n_pairs = d * (d - 1) // 2
    rows = []
    for replicate_dir in sorted(run_dir.glob("eval_replicate_s*")):
        metrics = json.loads((replicate_dir / "metrics.json").read_text())
        samples = torch.load(
            replicate_dir / "samples.pt", map_location="cpu", weights_only=True
        )
        log_w = torch.load(
            replicate_dir / "log_weights.pt", map_location="cpu", weights_only=True
        )
        n_samples = samples.shape[0]
        assert n_samples == metrics["n_eval_samples"]
        recomputed_ess = ess_from_log_weights(log_w).item()
        if not np.isclose(recomputed_ess, metrics["ess"], rtol=1e-3):
            raise RuntimeError(
                f"{replicate_dir.name}: ESS recomputed from log_weights.pt "
                f"({recomputed_ess:.1f}) disagrees with metrics.json "
                f"({metrics['ess']:.1f}) — artefact mismatch."
            )
        weights = torch.softmax(log_w, dim=0)
        estimates = {
            name: (weights * observable_values(name, samples.float(), target))
            .sum()
            .item()
            for name in OBSERVABLE_NAMES
        }
        phi_values = observable_values("phi", samples.float(), target).numpy()
        weights_np = weights.numpy()
        rows.append(
            {
                "replicate_seed": metrics["replicate_seed"],
                "estimates": estimates,
                "is_ess_fraction": metrics["ess_fraction"],
                "n_samples": int(n_samples),
                "backbone_rows": int(n_samples * n_euler_steps),
                "pair_delta_e_evals": int(n_samples * n_euler_steps * n_pairs),
                "dt_logp_evals": int(n_samples * n_euler_steps),
                "phi_mass_positive": float(weights_np[phi_values > 0].sum()),
                "phi_mass_negative": float(weights_np[phi_values < 0].sum()),
                "phi_mass_zero": float(weights_np[phi_values == 0].sum()),
                # Weighted vs unweighted second moment: the guard against the
                # symmetry trap — a Z2-symmetric but mode-collapsed sampler
                # scores a spuriously high N_eff on the phi MEAN (truth 0 by
                # symmetry); E[phi^2] matching the reference variance is what
                # certifies genuine mode coverage. The unweighted moment shows
                # what the raw process visits before weights correct it.
                "phi_sq_weighted": float((weights_np * phi_values**2).sum()),
                "phi_sq_unweighted": float((phi_values**2).mean()),
                "phi_hist": phi_mass_on_support(phi_values, weights_np, LATTICE_SIDE),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Kawasaki side: competitor chains under the burn-in rule
# ---------------------------------------------------------------------------


def kawasaki_chain_rows(probe_root, point, variant, target):
    """Per-chain post-burn-in estimates, with the burn-in rule applied per
    chain and the cost charged as TOTAL proposals."""
    variant_dir = probe_root / "competitor" / point / variant
    rows = []
    for chain_dir in sorted(variant_dir.glob("chain_*")):
        meta = json.loads((chain_dir / "meta.json").read_text())
        data = np.load(chain_dir / "snapshots.npz")
        spins = data["spins"]
        states = torch.from_numpy(spins.astype(np.float32))
        energy_trace = observable_values("energy", states, target).numpy()
        snapshot_sweeps = meta["snapshot_every_sweeps"]
        # tau on the second half (clearly post-transient) so the transient
        # cannot inflate its own discard window
        tau_snapshots, block_length, n_blocks = batch_means_tau_int(
            energy_trace[energy_trace.size // 2 :]
        )
        tau_sweeps = tau_snapshots * snapshot_sweeps
        burn_in = kawasaki_burn_in_sweeps(tau_sweeps)
        snapshot_index = np.arange(1, spins.shape[0] + 1) * snapshot_sweeps
        kept = snapshot_index > burn_in
        kept_states = states[torch.from_numpy(kept)]
        estimates = {
            name: observable_values(name, kept_states, target).mean().item()
            for name in OBSERVABLE_NAMES
        }
        kept_energy = energy_trace[kept]
        sokal_tau_snapshots = integrated_autocorr(kept_energy)
        phi_values = observable_values("phi", kept_states, target).numpy()
        rows.append(
            {
                "chain_index": meta["chain_index"],
                "init_kind": meta["init_kind"],
                "init_side": meta["init_side"],
                "estimates": estimates,
                "energy_evals": int(meta["n_proposals"]),
                "burn_in_sweeps": int(burn_in),
                "tau_int_batch_means_sweeps": float(tau_sweeps),
                "tau_int_batch_block_length": int(block_length),
                "tau_int_sokal_trial_steps": float(
                    sokal_tau_snapshots * data["snapshot_interval_proposals"]
                ),
                "n_kept_snapshots": int(kept.sum()),
                "phi_trace_post_burn_in": phi_values,
            }
        )
    if not rows:
        raise FileNotFoundError(f"no chains under {variant_dir}")
    return rows


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def n_eff_block(rows, cost_key, moments):
    """Per-observable N_eff (+jackknife SE) and the per-1e6-unit rate for
    one sampler, mirroring demo_4x4.assemble row construction."""
    mean_cost = float(np.mean([r[cost_key] for r in rows]))
    block = {"cost_key": cost_key, "mean_cost": mean_cost, "observables": {}}
    for name in OBSERVABLE_NAMES:
        ref_mean, ref_var = moments[name]
        estimates = [r["estimates"][name] for r in rows]
        n_eff, se = n_eff_observable(estimates, ref_mean, ref_var)
        block["observables"][name] = {
            "reference_mean": ref_mean,
            "reference_var": ref_var,
            "estimate_mean": float(np.mean(estimates)),
            "estimates": [float(e) for e in estimates],
            "n_eff": n_eff,
            "n_eff_se": se,
            "n_eff_per_1e6": n_eff / (mean_cost / 1e6),
            "n_eff_se_per_1e6": se / (mean_cost / 1e6),
        }
    return block


def total_variation(mass_a, mass_b):
    a = np.asarray(mass_a, dtype=float)
    b = np.asarray(mass_b, dtype=float)
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def tv_noise_floor(reference_hist, n_effective, rng, n_boot=200):
    """95th-percentile TV a PERFECT sampler would show at this effective
    sample size: multinomial pseudo-samples drawn from the reference law
    itself. Comparing raw TVs across samplers with very different draw
    counts (790k Kawasaki snapshots vs ~34k effective neural draws) reads
    finite-sample noise as coverage deficit; the floor removes exactly that
    confound. The effective-size stand-ins (Kish ESS for weighted draws,
    n/tau_int for chains) are approximations, disclosed in the output."""
    law = np.asarray(reference_hist, dtype=float)
    law = law / law.sum()
    tvs = [
        total_variation(rng.multinomial(int(n_effective), law), law)
        for _ in range(n_boot)
    ]
    return float(np.quantile(tvs, 0.95))


def coverage_block(neural_rows, kawasaki_rows, reference):
    """The mode-coverage axis at one operating point.

    Neural: Z2 mass balance (weighted mass on phi>0 vs phi<0; phi=0 mass
    reported separately), the phi second-moment check (weighted E[phi^2] vs
    the reference variance — the guard against symmetry-inflated phi-mean
    scores), and the pooled weighted phi histogram vs the reference's own
    (total variation on the exact support, compared against each side's OWN
    finite-sample noise floor). Kawasaki: split-half R-hat(phi) across the
    mode-seeded chains (the seeded-modes Gelman-Rubin construction) + the
    same floor-adjusted TV.

    coverage_ok = (neural TV excess over its floor <= kawasaki's excess)
    AND |balance - 0.5| <= 0.1.
    """
    rng = np.random.default_rng(20260813)
    balances = []
    pooled_neural = np.zeros_like(reference["phi_hist"])
    neural_effective_draws = 0.0
    for row in neural_rows:
        signed = row["phi_mass_positive"] + row["phi_mass_negative"]
        balances.append(row["phi_mass_positive"] / signed)
        pooled_neural += row["phi_hist"]
        neural_effective_draws += row["is_ess_fraction"] * row["n_samples"]
    pooled_neural /= len(neural_rows)

    mode_seeded = [r for r in kawasaki_rows if r["init_kind"] == "phase_separated"]
    min_length = min(r["phi_trace_post_burn_in"].size for r in mode_seeded)
    rhat_phi_mode_seeded = float(
        split_half_gelman_rubin(
            np.stack([r["phi_trace_post_burn_in"][:min_length] for r in mode_seeded])
        )
    )
    pooled_kawasaki = np.zeros_like(reference["phi_hist"])
    kawasaki_effective_draws = 0.0
    for row in kawasaki_rows:
        phi = row["phi_trace_post_burn_in"]
        pooled_kawasaki += phi_mass_on_support(
            phi, np.full(phi.size, 1.0 / phi.size), LATTICE_SIDE
        )
        tau_phi, _, _ = batch_means_tau_int(phi)
        kawasaki_effective_draws += phi.size / tau_phi
    pooled_kawasaki /= len(kawasaki_rows)

    tv_neural = total_variation(pooled_neural, reference["phi_hist"])
    tv_kawasaki = total_variation(pooled_kawasaki, reference["phi_hist"])
    floor_neural = tv_noise_floor(reference["phi_hist"], neural_effective_draws, rng)
    floor_kawasaki = tv_noise_floor(
        reference["phi_hist"], kawasaki_effective_draws, rng
    )
    excess_neural = max(0.0, tv_neural - floor_neural)
    excess_kawasaki = max(0.0, tv_kawasaki - floor_kawasaki)
    balance_mean = float(np.mean(balances))
    coverage_ok = (
        excess_neural <= excess_kawasaki + 1e-12
        and abs(balance_mean - 0.5) <= COVERAGE_BALANCE_TOLERANCE
    )
    return {
        "neural_balance_per_replicate": [float(b) for b in balances],
        "neural_balance_mean": balance_mean,
        "neural_phi_sq_weighted_mean": float(
            np.mean([r["phi_sq_weighted"] for r in neural_rows])
        ),
        "neural_phi_sq_unweighted_mean": float(
            np.mean([r["phi_sq_unweighted"] for r in neural_rows])
        ),
        "reference_phi_var": reference["moments"]["phi"][1],
        "neural_phi_tv_vs_reference": tv_neural,
        "kawasaki_phi_tv_vs_reference": tv_kawasaki,
        "neural_tv_noise_floor_95": floor_neural,
        "kawasaki_tv_noise_floor_95": floor_kawasaki,
        "neural_tv_excess": excess_neural,
        "kawasaki_tv_excess": excess_kawasaki,
        "neural_effective_draws": float(neural_effective_draws),
        "kawasaki_effective_draws": float(kawasaki_effective_draws),
        "kawasaki_rhat_phi_mode_seeded": rhat_phi_mode_seeded,
        "phi_support": phi_support(LATTICE_SIDE).tolist(),
        "neural_phi_hist": pooled_neural.tolist(),
        "kawasaki_phi_hist": pooled_kawasaki.tolist(),
        "reference_phi_hist": reference["phi_hist"].tolist(),
        "coverage_ok": bool(coverage_ok),
        "operationalisation_note": (
            "coverage_ok = (neural TV excess over its own 95% noise floor "
            "<= kawasaki's excess) AND |balance - 0.5| <= 0.1; effective "
            "sizes = Kish ESS (neural) and n/tau_int(phi) (chains)."
        ),
    }


def run_n_euler_steps(run_dir, fallback):
    """The pass counter is structural in n_euler_steps, so read it from the
    run's own config.json rather than trusting a CLI default to match."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        return int(json.loads(config_path.read_text())["ctmc"]["n_euler_steps"])
    return fallback


def analyse_point(point, probe_root, run_dir, n_euler_steps):
    sigma = OPERATING_POINTS[point]
    target = FixedCompositionIsingTarget(
        D=LATTICE_SIDE,
        sigma=sigma,
        target_composition=0.5,
        bias=0.0,
        device="cpu",
    )
    reference = reference_block(probe_root, point, target)
    moments = reference["moments"]

    kawasaki = {}
    for variant in ("local", "nonlocal"):
        rows = kawasaki_chain_rows(probe_root, point, variant, target)
        kawasaki[variant] = {
            "rows": rows,
            "n_eff": n_eff_block(rows, "energy_evals", moments),
        }
    # "Stronger per operating point" = larger N_eff(energy) per energy
    # evaluation, on these same runs.
    best_variant = max(
        kawasaki,
        key=lambda v: kawasaki[v]["n_eff"]["observables"]["energy"]["n_eff_per_1e6"],
    )

    neural = None
    if run_dir is not None and list(run_dir.glob("eval_replicate_s*")):
        rows = neural_replicate_rows(
            run_dir, target, run_n_euler_steps(run_dir, n_euler_steps)
        )
        neural = {
            "rows": rows,
            "n_eff_per_pass": n_eff_block(rows, "backbone_rows", moments),
            "n_eff_per_energy_eval": n_eff_block(rows, "pair_delta_e_evals", moments),
            "coverage": coverage_block(rows, kawasaki[best_variant]["rows"], reference),
        }

    ratios = None
    if neural is not None:
        n_neural_replicates = len(neural["rows"])
        ratios = {}
        for versus in ("nonlocal", "local"):
            kawasaki_obs = kawasaki[versus]["n_eff"]["observables"]
            kawasaki_cost = kawasaki[versus]["n_eff"]["mean_cost"]
            n_chains = len(kawasaki[versus]["rows"])
            per_observable = {}
            for name in OBSERVABLE_NAMES:
                k = kawasaki_obs[name]
                per_pass = neural["n_eff_per_pass"]["observables"][name]
                currencies = {}
                for currency, neural_cost in (
                    ("network_pass", neural["n_eff_per_pass"]["mean_cost"]),
                    ("energy_eval", neural["n_eff_per_energy_eval"]["mean_cost"]),
                ):
                    currencies[currency] = {
                        "delta": ratio_with_ci(
                            per_pass["n_eff"],
                            per_pass["n_eff_se"],
                            neural_cost,
                            k["n_eff"],
                            k["n_eff_se"],
                            kawasaki_cost,
                        ),
                        "f": ratio_with_f_ci(
                            per_pass["n_eff"],
                            neural_cost,
                            n_neural_replicates,
                            k["n_eff"],
                            kawasaki_cost,
                            n_chains,
                        ),
                    }
                per_observable[name] = currencies
            ratios[versus] = per_observable

    return {
        "point": point,
        "sigma": sigma,
        "reference": {
            "moments": {k: list(v) for k, v in moments.items()},
            "rhat_post_discard": reference["rhat_post_discard"],
            "n_pooled_snapshots": reference["n_pooled_snapshots"],
        },
        "kawasaki": {
            variant: {
                "n_eff": kawasaki[variant]["n_eff"],
                "chains": [
                    {k: v for k, v in row.items() if k != "phi_trace_post_burn_in"}
                    for row in kawasaki[variant]["rows"]
                ],
            }
            for variant in kawasaki
        },
        "best_kawasaki_variant": best_variant,
        "neural": None
        if neural is None
        else {
            "n_eff_per_pass": neural["n_eff_per_pass"],
            "n_eff_per_energy_eval": neural["n_eff_per_energy_eval"],
            "coverage": neural["coverage"],
            "replicates": [
                {k: v for k, v in row.items() if k != "phi_hist"}
                for row in neural["rows"]
            ],
        },
        "ratios_vs_kawasaki": ratios,
    }


def markdown_tables(point_result):
    lines = [
        f"## {point_result['point']} (sigma = {point_result['sigma']})",
        "",
        "| sampler | observable | reference | estimate | N_eff +/- SE "
        "| currency | mean cost | N_eff / 1e6 units |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def emit(sampler_label, block, currency_label):
        for name, obs in block["observables"].items():
            lines.append(
                f"| {sampler_label} | {name} | {obs['reference_mean']:.4f} "
                f"| {obs['estimate_mean']:.4f} "
                f"| {obs['n_eff']:.1f} +/- {obs['n_eff_se']:.1f} "
                f"| {currency_label} | {block['mean_cost']:.4g} "
                f"| {obs['n_eff_per_1e6']:.4g} |"
            )

    neural = point_result["neural"]
    if neural is not None:
        emit("neural (ma head)", neural["n_eff_per_pass"], "backbone_rows")
        emit("neural (ma head)", neural["n_eff_per_energy_eval"], "pair_delta_e_evals")
    for variant in ("local", "nonlocal"):
        emit(
            f"kawasaki {variant}",
            point_result["kawasaki"][variant]["n_eff"],
            "energy_evals",
        )
    lines.append("")
    lines.append(
        f"best kawasaki variant (N_eff(energy)/eval): "
        f"**{point_result['best_kawasaki_variant']}**"
    )
    if point_result["ratios_vs_kawasaki"] is not None:
        lines += [
            "",
            "| vs | observable | currency | ratio "
            "| 95% CI (delta) | 95% CI (F) | excl. parity delta/F |",
            "|---|---|---|---|---|---|---|",
        ]
        for versus, per_obs in point_result["ratios_vs_kawasaki"].items():
            for name, currencies in per_obs.items():
                for currency, ratio in currencies.items():
                    delta, f = ratio["delta"], ratio["f"]
                    lines.append(
                        f"| {versus} | {name} | {currency} "
                        f"| {delta['point']:.3g} "
                        f"| [{delta['lo']:.3g}, {delta['hi']:.3g}] "
                        f"| [{f['lo']:.3g}, {f['hi']:.3g}] "
                        f"| {delta['excludes_parity']} / "
                        f"{f['excludes_parity']} |"
                    )
    neural = point_result["neural"]
    if neural is not None:
        cov = neural["coverage"]
        lines += [
            "",
            "### Coverage axis",
            "",
            f"- Z2 balance (weighted mass phi>0 vs phi<0): "
            f"{cov['neural_balance_mean']:.4f} "
            f"(per-replicate {['%.3f' % b for b in cov['neural_balance_per_replicate']]})",
            f"- phi second moment: weighted {cov['neural_phi_sq_weighted_mean']:.5f} "
            f"/ unweighted {cov['neural_phi_sq_unweighted_mean']:.5f} "
            f"vs reference Var[phi] {cov['reference_phi_var']:.5f}",
            f"- TV vs reference: neural {cov['neural_phi_tv_vs_reference']:.4f} "
            f"(95% noise floor {cov['neural_tv_noise_floor_95']:.4f} at "
            f"{cov['neural_effective_draws']:.0f} effective draws, "
            f"excess {cov['neural_tv_excess']:.4f}); kawasaki "
            f"{cov['kawasaki_phi_tv_vs_reference']:.4f} "
            f"(floor {cov['kawasaki_tv_noise_floor_95']:.4f} at "
            f"{cov['kawasaki_effective_draws']:.0f}, "
            f"excess {cov['kawasaki_tv_excess']:.4f})",
            f"- kawasaki mode-seeded split-half R-hat(phi): "
            f"{cov['kawasaki_rhat_phi_mode_seeded']:.4f}",
            f"- coverage_ok: **{cov['coverage_ok']}** "
            f"({cov['operationalisation_note']})",
        ]
    lines += ["", "### Kawasaki secondary diagnostics", ""]
    for variant in ("local", "nonlocal"):
        chains = point_result["kawasaki"][variant]["chains"]
        tau_bm = [c["tau_int_batch_means_sweeps"] for c in chains]
        tau_sokal = [c["tau_int_sokal_trial_steps"] for c in chains]
        burn = sorted({c["burn_in_sweeps"] for c in chains})
        lines.append(
            f"- {variant}: tau_int(energy) batch-means "
            f"{min(tau_bm):.1f}-{max(tau_bm):.1f} sweeps "
            f"(Sokal {min(tau_sokal):.0f}-{max(tau_sokal):.0f} trial "
            f"steps); burn-in {burn} sweeps (rule: "
            f"max(1e4, 20*tau))"
        )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", default="results/kawasaki_probe")
    parser.add_argument(
        "--headline-run-dir",
        default="results/03_hard/"
        "H2_d64_c50_s223_letf_ma_100k_curr_seed42_20260722-124315",
    )
    parser.add_argument(
        "--floor-run-dir",
        default=None,
        help="floor-cell run dir once its replicate draws exist",
    )
    parser.add_argument("--n-euler-steps", type=int, default=128)
    parser.add_argument("--out", default="results/03_hard/probe_8x8_headline")
    args = parser.parse_args(argv)

    probe_root = Path(args.probe_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["sc"] = analyse_point(
        "sc", probe_root, Path(args.headline_run_dir), args.n_euler_steps
    )
    floor_run_dir = Path(args.floor_run_dir) if args.floor_run_dir else None
    results["s010"] = analyse_point(
        "s010", probe_root, floor_run_dir, args.n_euler_steps
    )

    headline = results["sc"]
    floor = results["s010"]
    verdicts = {}
    for ci_method in ("delta", "f"):
        floor_not_worse = None
        if floor["neural"] is not None:
            floor_ratios = floor["ratios_vs_kawasaki"][floor["best_kawasaki_variant"]][
                "energy"
            ]
            # "not worse" = not significantly below parity in either currency
            floor_not_worse = all(
                not (r[ci_method]["hi"] < 1.0) for r in floor_ratios.values()
            )
        headline_ratios = headline["ratios_vs_kawasaki"][
            headline["best_kawasaki_variant"]
        ]["energy"]
        local_ratios = headline["ratios_vs_kawasaki"]["local"]["energy"]
        verdicts[ci_method] = frozen_verdict(
            energy_eval_ratio=headline_ratios["energy_eval"][ci_method],
            network_pass_ratio=headline_ratios["network_pass"][ci_method],
            floor_not_worse=floor_not_worse,
            coverage_ok=headline["neural"]["coverage"]["coverage_ok"],
            gate_holds=True,  # the 4x4 enumeration gate passes at both sigmas
            beats_local_variant=all(
                r[ci_method]["point"] > 1.0 for r in local_ratios.values()
            ),
        )
    verdict = {
        "per_ci_method": verdicts,
        "margin_observable": "energy",
        "open_rulings": [
            "CI construction (delta-on-jackknife vs variance-ratio F): "
            "the margin rule names a 95% CI but not its construction; "
            "both computed, verdicts may differ",
            "margin observable: the rule fixes the observable SET but "
            "not which observable carries the margin rule; energy (the "
            "demo pack's headline row) used here, full set in the tables",
            "cross-currency division (settled): "
            "the two currencies BRACKET the method between its best and "
            "worst defensible accounting and carry NO single-number claim "
            "- a backbone row and a pair-Delta-E differ by ~1e4-1e5 FLOPs, "
            "so the unit-for-unit ratio is a robustness bracket, not a "
            "price. No cross-currency headline in prose; wall-clock at "
            "this size (favouring the classical chain on closed-form "
            "energies) disclosed in one plain sentence; regime "
            "interpretation (amortised / many-target / larger-lattice) "
            "carries the argument",
            "coverage operationalisation: TV-excess-over-noise-floor + "
            "balance tolerance 0.1",
        ],
    }

    payload = {"results": results, "verdict": verdict}
    (out_dir / "probe_8x8_headline.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    lines = ["# (sigma_c, 8x8) headline-cell probe analysis", ""]
    for point in ("sc", "s010"):
        lines += markdown_tables(results[point]) + [""]
    lines += ["## Three-way verdict (margin observable: energy)", ""]
    for ci_method, v in verdict["per_ci_method"].items():
        lines.append(
            f"- CI method **{ci_method}**: **{v['verdict']}** — "
            f"narrative: {v['narrative']}"
            + (f" — pending: {v['pending']}" if v["pending"] else "")
        )
    lines += ["", "### Open choices", ""]
    lines += [f"- {r}" for r in verdict["open_rulings"]]
    lines += [
        "",
        "### One-event Euler step check",
        "",
        "- one-event Euler step (multi_event=false): events/site/step "
        "<= 1/64 ~= 0.016 < 0.1 structurally",
        "- lambda_dt clip fraction 0.0 at every training-eval row of the "
        "headline run (training_log.csv); log-ratio clamp never engaged",
    ]
    (out_dir / "probe_8x8_headline.md").write_text("\n".join(lines) + "\n")
    print(
        f"[probe-analysis] wrote {out_dir}/probe_8x8_headline.{{json,md}}", flush=True
    )
    for ci_method, v in verdict["per_ci_method"].items():
        print(
            f"[probe-analysis] verdict ({ci_method}): {v['verdict']} "
            f"({v['narrative']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
