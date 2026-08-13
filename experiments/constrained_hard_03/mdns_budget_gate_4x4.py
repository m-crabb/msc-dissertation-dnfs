"""The budget-masked MDNS 4x4 CPU/GPU gate: arms against exact enumeration.

Second instantiation of the move-restriction principle: where the swap CTMC
restricts the MOVE SET of a flip sampler, the budget-masked reference
restricts the SPECIES DRAW of a masked-diffusion sampler. All ingredients
are individually derived and exhaustively verified (see
samplers/budget_masked.py and the three tests/test_budget_* modules); this
gate tests whether they COMPOSE into a working from-scratch sampler at the
enumerable size — 4x4 torus, d = 16, c = 0.5, the C(16,8) = 12,870-state
fibre, exact conditional by enumeration.

Arms (structure frozen in the plan; A = the method, B/C = its ablations):
  A budget_tilted — budget-masked reference + preconditioner V0.
  B none          — Fig.-10 mirror: same reference, no preconditioner.
  C unconstrained — structural-failure arm: the paper's zero-imputation
                    preconditioner, blind to the budget.

Two dated passes share this driver (both prereg'd in the plan doc):
  1. First pass (2026-08-13, CPU): defaults below — 2,000 steps, seed 42,
     sigma = 0.223. Verdict recorded G0 PASS / G1 FAIL(KL) / G2 FAIL;
     stands as recorded.
  2. Amendment 01 (2026-08-13, a30 GPU): --steps 10000 --seeds 42,43,44
     --sigmas 0.10,0.223 — budget-only change, justified by the Phase-1
     finding that the G1 miss was budget-shaped (no plateau anywhere;
     KL mass in rare near-boundary/late-generation contexts). Primary
     purpose: the DNFS 4x4 side-by-side, hence the two added eval
     INSTRUMENTS (within-level excess TV and per-site free-energy bias,
     both the DNFS gate's own constructions) — instruments only, the
     eval SAMPLING protocol is unchanged and eval contexts are drawn on
     CPU RNG so the sigma_c context set is identical across passes.

Free-energy note (why the logged weights need no constant bookkeeping):
the full unnormalised trajectory weight is
    w = Q0(traj) p~(X_1) / (base(X_1) P_model(traj)),
and the reference's assignment product (the trajectory constant
N_+!(d-N_+)!/d!) is EXACTLY 1/C(d, N_+) = base(X_1), so they cancel:
log w = log p~(X_1) - sum log q_model. logmeanexp of the logged weights
therefore estimates log Z_slice directly and free_energy_lb_estimate
(paper Eq. 37 convention) applies verbatim, comparable to the DNFS gate's
on-slice numbers (same estimator, same reference).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

from experiments.constrained_hard_03.gate_4x4 import (
    _categorical_energy_bins,
    _energy,
    energy_marginal_tv,
    on_slice_free_energy_reference,
    slice_energy_hist,
    within_level_uniformity,
)
from discrete_flow_sampler.diagnostics.metrics import (
    conditional_pmf_at_composition,
    enumerate_states,
    ess_from_log_weights,
    exact_log_probs,
    free_energy_lb_estimate,
)
from discrete_flow_sampler.samplers.budget_masked import (
    GatedBudgetTiltOffset,
    MaskedConditionalNet,
    feasibility_clamped_p_plus,
    masked_count_and_budget,
    preconditioner_logit_diff,
    rollout_budget_masked,
    wdce_cross_entropy,
)
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

# First-pass frozen protocol (plan launch note, 2026-08-13); the amendment
# overrides steps/seeds/sigmas on the command line and nothing else.
LATTICE_SIDE = 4
N_SITES = 16
N_PLUS = 8
SIGMA = 0.223
TRAIN_ROLLOUTS_PER_STEP = 256
CORRUPTION_REPLICATES = 2
LEARNING_RATE = 1e-3
TRAIN_STEPS = 2000
TRAIN_SEED = 42
EVAL_ROLLOUTS = 65536
EVAL_BATCH = 8192
EVAL_CONTEXTS = 2000
EVAL_SEED = 1042
PLATEAU_ESS_FRACTION = 0.5
PLATEAU_WINDOW = 51
LATE_GENERATION_MAX_MASKED = 4

ARMS = {
    "a": "budget_tilted",
    "b": "none",
    "c": "unconstrained",
    "a2": "budget_tilted_gated",   # forensics arm A': V0 with learnable
                                   # scales on its two terms (gates init 1.0)
}

# First-pass G-cuts, computed for every (sigma, seed, arm) for continuity
# (Amendment 01: printed alongside, not re-adjudicated).
CUTS = {"energy_tv": 0.02, "conditional_kl": 0.01, "ess_fraction": 0.20,
        "plateau_ratio": 0.5, "late_error_ratio": 2.0}


def make_logit_fn(net, adjacency, sigma, mode):
    """(logit_fn, extra_trainables): the gated arm carries two learnable
    scale parameters alongside the trunk; every other arm's offset is a
    pure function."""
    if mode == "budget_tilted_gated":
        gated_offset = GatedBudgetTiltOffset(adjacency, sigma, N_PLUS)

        def logit_fn(x_masked):
            return net(x_masked) + gated_offset(x_masked)
        return logit_fn, gated_offset

    def logit_fn(x_masked):
        return net(x_masked) + preconditioner_logit_diff(
            x_masked, adjacency, sigma, N_PLUS, mode
        )
    return logit_fn, None


def draw_eval_contexts(slice_states, slice_log_p_cond, generator,
                       n_contexts=EVAL_CONTEXTS):
    """Corruption contexts from the population WDCE law with EXACT weights:
    terminal ~ exact fibre conditional, lambda ~ U(0,1), sites masked
    independently, empty masks redrawn. Drawn ONCE on CPU RNG and shared by
    every arm, seed, and pass, so all KL numbers are matched."""
    terminal_rows = torch.multinomial(
        slice_log_p_cond.exp(), n_contexts, replacement=True,
        generator=generator,
    )
    terminals = slice_states[terminal_rows].float()
    corruption_level = torch.rand(n_contexts, 1, generator=generator)
    mask = torch.rand(
        n_contexts, N_SITES, generator=generator
    ) < corruption_level
    empty = ~mask.any(dim=1)
    while empty.any():
        mask[empty] = (
            torch.rand(int(empty.sum()), N_SITES, generator=generator)
            < torch.rand(int(empty.sum()), 1, generator=generator)
        )
        empty = ~mask.any(dim=1)
    return terminals.masked_fill(mask, 0.0)


def exact_conditional_per_context(context, slice_states, slice_log_p_cond):
    """Exact Pr(X^i = +1 | unmasked part) for every masked site i of one
    context, by selecting the fibre states consistent with the unmasked
    pattern and renormalising — pure reuse of the enumerated conditional."""
    unmasked = context != 0.0
    consistent = (
        slice_states[:, unmasked].float() == context[unmasked]
    ).all(dim=1)
    log_p = slice_log_p_cond[consistent]
    posterior = torch.softmax(log_p, dim=0)
    plus_indicator = (slice_states[consistent] == 1).float()
    return posterior @ plus_indicator          # (n_sites,), valid at masked


def conditional_kl_and_late_error(logit_fn, contexts, slice_states,
                                  slice_log_p_cond, device):
    """Mean KL(exact || model) over (context, masked site) pairs — exact per
    context, no sampling floor — plus the late-generation (m <= 4) mean
    absolute error. The model conditional carries the same feasibility
    clamp generation uses: the metric scores the sampler's law."""
    with torch.no_grad():
        logits = logit_fn(contexts.to(device)).cpu()
    masked_count, budget = masked_count_and_budget(contexts, N_PLUS)
    kl_terms, late_errors = [], []
    for row, context in enumerate(contexts):
        masked_sites = (context == 0.0).nonzero(as_tuple=True)[0]
        exact_p_plus = exact_conditional_per_context(
            context, slice_states, slice_log_p_cond
        )[masked_sites]
        model_p_plus = feasibility_clamped_p_plus(
            torch.sigmoid(logits[row, masked_sites]),
            budget[row].expand(len(masked_sites)),
            masked_count[row].expand(len(masked_sites)),
        )
        model_p_plus = model_p_plus.clamp(1e-12, 1 - 1e-12)
        p, q = exact_p_plus, model_p_plus
        kl = torch.where(p > 0, p * (p.log() - q.log()), torch.zeros_like(p))
        kl = kl + torch.where(
            p < 1, (1 - p) * ((1 - p).log() - (1 - q).log()),
            torch.zeros_like(p),
        )
        kl_terms.append(kl)
        if masked_count[row] <= LATE_GENERATION_MAX_MASKED:
            late_errors.append((p - model_p_plus).abs())
    mean_kl = torch.cat(kl_terms).mean().item()
    late_error = torch.cat(late_errors).mean().item() if late_errors \
        else float("nan")
    return mean_kl, late_error


def evaluate_arm(logit_fn, target, slice_states, slice_log_p_cond,
                 exact_hist, bins, sigma, device, run_dir,
                 eval_rollouts, save_artefacts=True):
    """Eval rollouts -> ESS fraction, G0 count, energy-marginal TV, plus the
    two DNFS-shared instruments (Amendment 01): within-level excess TV and
    per-site free-energy bias. Saves terminals + log-weights so any later
    instrument can rerun off artefacts instead of GPU."""
    terminals_all, log_w_all = [], []
    generator = torch.Generator(device=device).manual_seed(EVAL_SEED)
    with torch.no_grad():
        for _ in range(max(1, eval_rollouts // EVAL_BATCH)):
            batch = min(EVAL_BATCH, eval_rollouts)
            terminals, rollout_log_prob = rollout_budget_masked(
                logit_fn, batch, N_SITES, N_PLUS, generator
            )
            terminals_all.append(terminals.cpu())
            log_w_all.append(
                (target.log_prob(terminals) - rollout_log_prob).cpu()
            )
    terminals = torch.cat(terminals_all)
    log_w = torch.cat(log_w_all)
    n_plus = ((terminals + 1) / 2).sum(dim=1)
    off_fibre = int((n_plus != N_PLUS).sum().item())
    ess = ess_from_log_weights(log_w).item()
    weights = torch.softmax(log_w, dim=0)
    adjacency_cpu = target.A.cpu()
    model_hist = slice_energy_hist(terminals, weights, adjacency_cpu, bins)

    slice_energies = _energy(slice_states.float(), adjacency_cpu)
    sample_energies = _energy(terminals, adjacency_cpu)
    levels = within_level_uniformity(
        terminals, weights, sample_energies, slice_states.float(),
        slice_energies,
    )
    free_energy_model = free_energy_lb_estimate(
        log_w, sigma, N_SITES
    ).item()
    free_energy_ref = on_slice_free_energy_reference(
        _CpuTargetView(target), slice_states
    ).item()
    if save_artefacts:
        torch.save({"terminals": terminals.to(torch.int8),
                    "log_weights": log_w}, run_dir / "eval_artefacts.pt")
    return {
        "eval_rollouts": len(terminals),
        "off_fibre_count": off_fibre,
        "ess_fraction": ess / len(terminals),
        "energy_tv": energy_marginal_tv(model_hist, exact_hist),
        "max_level_excess": max(
            (level["excess"] for level in levels), default=float("nan")
        ),
        "within_level": levels,
        "free_energy_model": free_energy_model,
        "free_energy_ref": free_energy_ref,
        "free_energy_bias": free_energy_model - free_energy_ref,
    }


class _CpuTargetView:
    """CPU view of a possibly-GPU target for the slice free-energy
    reference (the slice tensors live on CPU throughout)."""

    def __init__(self, target):
        self.sigma, self.d = target.sigma, target.d
        self._target = target

    def log_prob(self, x):
        return self._target.log_prob(x.to(self._target.device)).cpu()


def plateau_step(train_ess_series):
    """First step whose centred PLATEAU_WINDOW-median training-batch ESS
    fraction clears PLATEAU_ESS_FRACTION; None if never."""
    series = torch.tensor(train_ess_series)
    half = PLATEAU_WINDOW // 2
    for centre in range(half, len(series) - half):
        window = series[centre - half : centre + half + 1]
        if window.median().item() >= PLATEAU_ESS_FRACTION:
            return centre
    return None


def near_boundary_loss_weight(boost):
    """eta(context) = 1 + boost*1[b in {1, m-1}] — minimiser-safe (context-
    measurable; see wdce_cross_entropy's docstring) gradient reallocation
    towards the starved near-boundary contexts the Phase-1/forensics KL
    decompositions localised."""
    def eta(corrupted):
        masked_count, budget = masked_count_and_budget(corrupted, N_PLUS)
        near = (budget == 1) | (budget == masked_count - 1)
        return 1.0 + boost * near.float()
    return eta


def train_arm(arm, mode, target, sigma, seed, steps, results_root, tag,
              device, boost=0.0):
    seed_everything(seed)
    net = MaskedConditionalNet(N_SITES).to(device)
    logit_fn, gated_offset = make_logit_fn(net, target.A, sigma, mode)
    trainables = list(net.parameters())
    if gated_offset is not None:
        gated_offset.to(device)
        trainables += list(gated_offset.parameters())
    optimiser = torch.optim.Adam(trainables, lr=LEARNING_RATE)
    context_weight = near_boundary_loss_weight(boost) if boost > 0 else None
    rollout_generator = torch.Generator(device=device).manual_seed(seed)
    corruption_generator = torch.Generator(device=device).manual_seed(
        seed + 1
    )

    sigma_tag = f"s{sigma:.3f}".replace("0.", "")
    run_dir = results_root / f"arm_{arm}_{mode}_{sigma_tag}_seed{seed}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_rows, train_off_fibre = [], 0
    started = time.time()
    for step in range(steps):
        with torch.no_grad():
            terminals, rollout_log_prob = rollout_budget_masked(
                logit_fn, TRAIN_ROLLOUTS_PER_STEP, N_SITES, N_PLUS,
                rollout_generator,
            )
            log_w = target.log_prob(terminals) - rollout_log_prob
            weights = torch.softmax(log_w, dim=0)
            train_off_fibre += int(
                (((terminals + 1) / 2).sum(dim=1) != N_PLUS).sum().item()
            )
            ess_fraction = (
                ess_from_log_weights(log_w).item() / TRAIN_ROLLOUTS_PER_STEP
            )
        loss = wdce_cross_entropy(
            logit_fn, terminals, weights, CORRUPTION_REPLICATES,
            corruption_generator, context_loss_weight=context_weight,
        )
        optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.norm(torch.stack([
            parameter.grad.norm() for parameter in trainables
            if parameter.grad is not None
        ])).item()
        optimiser.step()
        log_rows.append(
            {"step": step, "loss": loss.item(),
             "train_ess_fraction": ess_fraction,
             "grad_norm": grad_norm}
        )
    wall_clock = time.time() - started

    with open(run_dir / "training_log.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    return (net, logit_fn, gated_offset, run_dir, log_rows,
            train_off_fibre, wall_clock)


def run_slate(sigma, seeds, arms, steps, results_root, tag, device,
              eval_rollouts, eval_contexts, boost=0.0):
    target = FixedCompositionIsingTarget(
        D=LATTICE_SIDE, sigma=sigma, target_composition=0.5, device=device
    )
    adjacency_cpu = target.A.cpu()
    cpu_target = FixedCompositionIsingTarget(
        D=LATTICE_SIDE, sigma=sigma, target_composition=0.5
    )
    states = enumerate_states(N_SITES)
    log_pi = exact_log_probs(cpu_target, states)
    slice_states, slice_log_p_cond = conditional_pmf_at_composition(
        states, log_pi, N_PLUS
    )
    slice_energies = _energy(slice_states.float(), adjacency_cpu)
    bins = _categorical_energy_bins(slice_energies)
    exact_hist = slice_energy_hist(
        slice_states.float(), slice_log_p_cond.exp(), adjacency_cpu, bins
    )
    # CPU RNG on purpose: the sigma_c context set must match the first pass
    contexts = draw_eval_contexts(
        slice_states, slice_log_p_cond,
        torch.Generator().manual_seed(EVAL_SEED), eval_contexts,
    )

    reports = {}
    for seed in seeds:
        for arm in arms:
            mode = ARMS[arm]
            seed_everything(seed)
            init_net = MaskedConditionalNet(N_SITES).to(device)
            init_fn, init_gated = make_logit_fn(
                init_net, target.A, sigma, mode)
            if init_gated is not None:
                init_gated.to(device)
            init_kl, init_late_error = conditional_kl_and_late_error(
                init_fn, contexts, slice_states, slice_log_p_cond, device,
            )
            print(f"[gate] sigma={sigma} seed={seed} arm {arm} ({mode}): "
                  f"training {steps} steps ...", flush=True)
            (net, logit_fn, gated_offset, run_dir, log_rows,
             train_off_fibre, wall) = \
                train_arm(arm, mode, target, sigma, seed, steps,
                          results_root, tag, device, boost)
            trained_kl, trained_late_error = conditional_kl_and_late_error(
                logit_fn, contexts, slice_states, slice_log_p_cond, device
            )
            eval_metrics = evaluate_arm(
                logit_fn, target, slice_states, slice_log_p_cond,
                exact_hist, bins, sigma, device, run_dir, eval_rollouts,
            )
            report = {
                "arm": arm, "preconditioner": mode, "sigma": sigma,
                "seed": seed, "steps": steps, "device": str(device),
                "near_boundary_boost": boost,
                "learned_gates": (
                    {"gate_budget": gated_offset.gate_budget.item(),
                     "gate_field": gated_offset.gate_field.item()}
                    if gated_offset is not None else None
                ),
                "train_off_fibre_count": train_off_fibre,
                "train_wall_clock_s": wall,
                "final_loss": log_rows[-1]["loss"],
                "plateau_step": plateau_step(
                    [row["train_ess_fraction"] for row in log_rows]
                ),
                "final_train_ess_fraction":
                    log_rows[-1]["train_ess_fraction"],
                "init_conditional_kl": init_kl,
                "trained_conditional_kl": trained_kl,
                "init_late_generation_error": init_late_error,
                "trained_late_generation_error": trained_late_error,
                **eval_metrics,
            }
            reports[f"{arm}_seed{seed}"] = report
            torch.save(net.state_dict(), run_dir / "model.pt")
            with open(run_dir / "metrics.json", "w") as handle:
                json.dump(report, handle, indent=2)
            print(f"[gate] sigma={sigma} seed={seed} arm {arm}: "
                  f"KL {trained_kl:.4f} TV {report['energy_tv']:.4f} "
                  f"ESS {report['ess_fraction']:.3f} "
                  f"FEbias {report['free_energy_bias']:+.4f} "
                  f"G0 off-fibre {report['off_fibre_count']}", flush=True)
    return reports


def first_pass_cut_table(reports, seeds):
    """The original G-cuts per (seed), for continuity (not re-adjudication)."""
    table = {}
    for seed in seeds:
        a = reports.get(f"a_seed{seed}")
        b = reports.get(f"b_seed{seed}")
        c = reports.get(f"c_seed{seed}")
        if a is None:
            continue
        row = {
            "G0_pass": all(
                r["off_fibre_count"] == 0 and r["train_off_fibre_count"] == 0
                for r in (a, b, c) if r is not None
            ),
            "G1_energy_tv": a["energy_tv"],
            "G1_conditional_kl": a["trained_conditional_kl"],
            "G1_ess_fraction": a["ess_fraction"],
            "G1_pass": (
                a["energy_tv"] <= CUTS["energy_tv"]
                and a["trained_conditional_kl"] <= CUTS["conditional_kl"]
                and a["trained_conditional_kl"] < a["init_conditional_kl"]
                and a["ess_fraction"] >= CUTS["ess_fraction"]
            ),
        }
        if b is not None:
            row["G2a_plateaus"] = (a["plateau_step"], b["plateau_step"])
            row["G2a_pass"] = a["plateau_step"] is not None and (
                b["plateau_step"] is None
                or a["plateau_step"]
                <= CUTS["plateau_ratio"] * b["plateau_step"]
            )
        if c is not None:
            ratio = c["trained_late_generation_error"] / max(
                a["trained_late_generation_error"], 1e-12
            )
            row["G2b_ratio"] = ratio
            row["G2b_pass"] = ratio >= CUTS["late_error_ratio"]
        table[f"seed{seed}"] = row
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",
                        default="results/03_hard/mdns_budget_gate_4x4")
    parser.add_argument("--tag", default="20260813-gate")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--seeds", default=str(TRAIN_SEED),
                        help="comma-separated")
    parser.add_argument("--sigmas", default=str(SIGMA),
                        help="comma-separated operating points")
    parser.add_argument("--arms", default="a,b,c")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-rollouts", type=int, default=EVAL_ROLLOUTS,
                        help="plumbing smoke only; the frozen protocol is "
                             "the default")
    parser.add_argument("--eval-contexts", type=int, default=EVAL_CONTEXTS,
                        help="plumbing smoke only")
    parser.add_argument("--near-boundary-boost", type=float, default=0.0,
                        help="eta(context) boost kappa on b in {1, m-1} "
                             "contexts; 0 = frozen-protocol loss (default)")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    results_root = Path(args.results_dir)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    arms = args.arms.split(",")
    all_reports = {}
    for sigma in (float(value) for value in args.sigmas.split(",")):
        reports = run_slate(
            sigma, seeds, arms, args.steps, results_root, args.tag,
            device, args.eval_rollouts, args.eval_contexts,
            args.near_boundary_boost,
        )
        all_reports[f"sigma_{sigma}"] = {
            "reports": reports,
            "first_pass_cuts_for_continuity":
                first_pass_cut_table(reports, seeds),
        }
    with open(results_root / f"verdict_{args.tag}.json", "w") as handle:
        json.dump(all_reports, handle, indent=2)
    print("[gate] slate complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
