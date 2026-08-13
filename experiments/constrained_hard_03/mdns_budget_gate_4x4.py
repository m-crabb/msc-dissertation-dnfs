"""The budget-masked MDNS 4x4 CPU gate: three arms against exact enumeration.

Second instantiation of the move-restriction principle: where the swap CTMC
restricts the MOVE SET of a flip sampler, the budget-masked reference
restricts the SPECIES DRAW of a masked-diffusion sampler. All ingredients
are individually derived and exhaustively verified (see
samplers/budget_masked.py and the three tests/test_budget_* modules); this
gate tests whether they COMPOSE into a working from-scratch sampler at the
enumerable size — 4x4 torus, d = 16, c = 0.5, the C(16,8) = 12,870-state
fibre, exact conditional by enumeration.

Arms (structure frozen in the plan; one seed each):
  A budget_tilted — the method: budget-masked reference + preconditioner V0.
  B none          — Fig.-10 mirror: same reference, no preconditioner.
  C unconstrained — structural-failure arm: the paper's zero-imputation
                    preconditioner, blind to the budget.

Gates (numeric cuts frozen in the plan's launch note BEFORE any arm ran,
2026-08-13): G0 off-fibre count = 0 (structural); G1 on arm A =
energy-marginal TV <= 0.02 AND conditional-KL <= 0.01 nats (trained < init
too) AND eval IS-ESS fraction >= 0.20; G2 = (a) arm A plateau step <= 0.5 x
arm B's (training-ESS 51-step centred median >= 0.5), (b) trained arm C's
late-generation (m <= 4) mean conditional error >= 2 x arm A's.

Failure localisation (why this gate is informative either way): the algebra
is verified, so a G1 failure implicates the training loop or the objective
transfer, not the derivation; a G2(a) failure with G1 passing means the
sampler works but the preconditioner claim stays at the enumerable-error
tables.

CPU-only, threads capped so the concurrent Kawasaki reference chain on this
machine is not starved.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from experiments.constrained_hard_03.gate_4x4 import (
    _categorical_energy_bins,
    _energy,
    energy_marginal_tv,
    slice_energy_hist,
)
from discrete_flow_sampler.diagnostics.metrics import (
    conditional_pmf_at_composition,
    enumerate_states,
    ess_from_log_weights,
    exact_log_probs,
)
from discrete_flow_sampler.samplers.budget_masked import (
    MaskedConditionalNet,
    feasibility_clamped_p_plus,
    masked_count_and_budget,
    preconditioner_logit_diff,
    rollout_budget_masked,
    wdce_cross_entropy,
)
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

# Frozen protocol (plan launch note, 2026-08-13). Values live here as the
# single source the run reads; the plan doc is the prereg record.
SIGMA = 0.223
LATTICE_SIDE = 4
N_SITES = 16
N_PLUS = 8
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
}


def make_logit_fn(net, adjacency, mode):
    def logit_fn(x_masked):
        return net(x_masked) + preconditioner_logit_diff(
            x_masked, adjacency, SIGMA, N_PLUS, mode
        )
    return logit_fn


def draw_eval_contexts(slice_states, slice_log_p_cond, generator):
    """Corruption contexts from the population WDCE law with EXACT weights:
    terminal ~ exact fibre conditional, lambda ~ U(0,1), sites masked
    independently, empty masks redrawn (a context must have at least one
    masked site to carry a conditional). Drawn once and shared by every
    arm and by the init/trained evals, so all KL numbers are matched."""
    terminal_rows = torch.multinomial(
        slice_log_p_cond.exp(), EVAL_CONTEXTS, replacement=True,
        generator=generator,
    )
    terminals = slice_states[terminal_rows].float()
    corruption_level = torch.rand(EVAL_CONTEXTS, 1, generator=generator)
    mask = torch.rand(
        EVAL_CONTEXTS, N_SITES, generator=generator
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
    pattern and renormalising — pure reuse of the enumerated conditional,
    no separate completion enumeration to drift from it."""
    unmasked = context != 0.0
    consistent = (
        slice_states[:, unmasked].float() == context[unmasked]
    ).all(dim=1)
    log_p = slice_log_p_cond[consistent]
    posterior = torch.softmax(log_p, dim=0)
    plus_indicator = (slice_states[consistent] == 1).float()
    return posterior @ plus_indicator          # (n_sites,), valid at masked


def conditional_kl_and_late_error(logit_fn, contexts, slice_states,
                                  slice_log_p_cond):
    """Mean KL(exact || model) over (context, masked site) pairs — the G1
    correctness metric, exact per context (no sampling floor) — plus the
    G2(b) late-generation mean absolute error over contexts with
    m <= LATE_GENERATION_MAX_MASKED. The model conditional carries the same
    feasibility clamp generation uses: the metric scores the sampler's law,
    not the raw network."""
    with torch.no_grad():
        logits = logit_fn(contexts)
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
                 exact_hist, bins):
    """Eval rollouts -> ESS fraction, G0 count, energy-marginal TV."""
    terminals_all, log_w_all = [], []
    generator = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        for _ in range(EVAL_ROLLOUTS // EVAL_BATCH):
            terminals, rollout_log_prob = rollout_budget_masked(
                logit_fn, EVAL_BATCH, N_SITES, N_PLUS, generator
            )
            terminals_all.append(terminals)
            log_w_all.append(target.log_prob(terminals) - rollout_log_prob)
    terminals = torch.cat(terminals_all)
    log_w = torch.cat(log_w_all)
    n_plus = ((terminals + 1) / 2).sum(dim=1)
    off_fibre = int((n_plus != N_PLUS).sum().item())
    ess = ess_from_log_weights(log_w).item()
    weights = torch.softmax(log_w, dim=0)
    model_hist = slice_energy_hist(terminals, weights, target.A, bins)
    return {
        "eval_rollouts": EVAL_ROLLOUTS,
        "off_fibre_count": off_fibre,
        "ess_fraction": ess / EVAL_ROLLOUTS,
        "energy_tv": energy_marginal_tv(model_hist, exact_hist),
    }


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


def train_arm(arm, mode, target, results_root, tag):
    seed_everything(TRAIN_SEED)
    net = MaskedConditionalNet(N_SITES)
    logit_fn = make_logit_fn(net, target.A, mode)
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    rollout_generator = torch.Generator().manual_seed(TRAIN_SEED)
    corruption_generator = torch.Generator().manual_seed(TRAIN_SEED + 1)

    run_dir = results_root / f"arm_{arm}_{mode}_seed{TRAIN_SEED}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_rows, train_off_fibre = [], 0
    started = time.time()
    for step in range(TRAIN_STEPS):
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
            corruption_generator,
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        log_rows.append(
            {"step": step, "loss": loss.item(),
             "train_ess_fraction": ess_fraction}
        )
    wall_clock = time.time() - started

    with open(run_dir / "training_log.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    return net, logit_fn, run_dir, log_rows, train_off_fibre, wall_clock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",
                        default="results/03_hard/mdns_budget_gate_4x4")
    parser.add_argument("--tag", default="20260813-gate")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    target = FixedCompositionIsingTarget(
        D=LATTICE_SIDE, sigma=SIGMA, target_composition=0.5
    )
    states = enumerate_states(N_SITES)
    log_pi = exact_log_probs(target, states)
    slice_states, slice_log_p_cond = conditional_pmf_at_composition(
        states, log_pi, N_PLUS
    )
    slice_energies = _energy(slice_states.float(), target.A)
    bins = _categorical_energy_bins(slice_energies)
    exact_hist = slice_energy_hist(
        slice_states.float(), slice_log_p_cond.exp(), target.A, bins
    )
    context_generator = torch.Generator().manual_seed(EVAL_SEED)
    contexts = draw_eval_contexts(
        slice_states, slice_log_p_cond, context_generator
    )

    results_root = Path(args.results_dir)
    arm_reports = {}
    for arm, mode in ARMS.items():
        print(f"[gate] arm {arm} ({mode}): init eval ...", flush=True)
        seed_everything(TRAIN_SEED)
        init_net = MaskedConditionalNet(N_SITES)     # zero final layer
        init_kl, init_late_error = conditional_kl_and_late_error(
            make_logit_fn(init_net, target.A, mode), contexts,
            slice_states, slice_log_p_cond,
        )
        print(f"[gate] arm {arm} ({mode}): training {TRAIN_STEPS} steps ...",
              flush=True)
        net, logit_fn, run_dir, log_rows, train_off_fibre, wall_clock = \
            train_arm(arm, mode, target, results_root, args.tag)
        trained_kl, trained_late_error = conditional_kl_and_late_error(
            logit_fn, contexts, slice_states, slice_log_p_cond
        )
        eval_metrics = evaluate_arm(
            logit_fn, target, slice_states, slice_log_p_cond, exact_hist,
            bins,
        )
        report = {
            "arm": arm, "preconditioner": mode,
            "train_off_fibre_count": train_off_fibre,
            "train_wall_clock_s": wall_clock,
            "final_loss": log_rows[-1]["loss"],
            "plateau_step": plateau_step(
                [row["train_ess_fraction"] for row in log_rows]
            ),
            "final_train_ess_fraction": log_rows[-1]["train_ess_fraction"],
            "init_conditional_kl": init_kl,
            "trained_conditional_kl": trained_kl,
            "init_late_generation_error": init_late_error,
            "trained_late_generation_error": trained_late_error,
            **eval_metrics,
        }
        arm_reports[arm] = report
        torch.save(net.state_dict(), run_dir / "model.pt")
        with open(run_dir / "metrics.json", "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"[gate] arm {arm}: {json.dumps(report, indent=2)}",
              flush=True)

    a, b, c = arm_reports["a"], arm_reports["b"], arm_reports["c"]
    g0_pass = all(
        r["off_fibre_count"] == 0 and r["train_off_fibre_count"] == 0
        for r in arm_reports.values()
    )
    g1 = {
        "energy_tv": a["energy_tv"], "energy_tv_cut": 0.02,
        "conditional_kl": a["trained_conditional_kl"],
        "conditional_kl_cut": 0.01,
        "kl_improved_on_init":
            a["trained_conditional_kl"] < a["init_conditional_kl"],
        "ess_fraction": a["ess_fraction"], "ess_fraction_cut": 0.20,
    }
    g1_pass = (
        g1["energy_tv"] <= 0.02
        and g1["conditional_kl"] <= 0.01
        and g1["kl_improved_on_init"]
        and g1["ess_fraction"] >= 0.20
    )
    b_never_plateaued = b["plateau_step"] is None
    g2a_pass = (
        a["plateau_step"] is not None
        and (b_never_plateaued
             or a["plateau_step"] <= 0.5 * b["plateau_step"])
    )
    late_ratio = (
        c["trained_late_generation_error"]
        / max(a["trained_late_generation_error"], 1e-12)
    )
    g2b_pass = late_ratio >= 2.0
    verdict = {
        "G0_pass": g0_pass,
        "G1": g1, "G1_pass": g1_pass,
        "G2a": {"plateau_a": a["plateau_step"],
                "plateau_b": b["plateau_step"],
                "b_never_plateaued": b_never_plateaued},
        "G2a_pass": g2a_pass,
        "G2b": {"late_error_a": a["trained_late_generation_error"],
                "late_error_c": c["trained_late_generation_error"],
                "ratio": late_ratio},
        "G2b_pass": g2b_pass,
        "arms": arm_reports,
    }
    with open(results_root / f"verdict_{args.tag}.json", "w") as handle:
        json.dump(verdict, handle, indent=2)
    print(f"[gate] VERDICT: G0={g0_pass} G1={g1_pass} "
          f"G2a={g2a_pass} G2b={g2b_pass}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
