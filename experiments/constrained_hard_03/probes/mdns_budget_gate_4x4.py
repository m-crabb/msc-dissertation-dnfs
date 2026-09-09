"""The budget-masked MDNS 4x4 CPU/GPU gate: arms against exact enumeration.

Second instantiation of the move-restriction principle: where the swap CTMC
restricts the move set of a flip sampler, the budget-masked reference
restricts the species draw of a masked-diffusion sampler. The ingredients are
derived and tested individually (samplers/budget_masked.py, the three
tests/test_budget_* modules); this gate tests whether they compose into a
working from-scratch sampler at the enumerable size — 4x4 torus, d = 16,
c = 0.5, the C(16,8) = 12,870-state fibre, exact conditional by enumeration.

Arms (a = the method, b/c = its ablations):
  a budget_tilted — budget-masked reference + preconditioner V0.
  b none          — Fig.-10 mirror: same reference, no preconditioner.
  c unconstrained — structural-failure arm: the paper's zero-imputation
                    preconditioner, blind to the budget.

Two passes share this driver:
  1. First pass (2026-08-13, CPU): defaults below — 2,000 steps, seed 42,
     sigma = 0.223. Outcome: G0 pass / G1 fail (KL) / G2 fail.
  2. Second pass (2026-08-13, a30 GPU): --steps 10000 --seeds 42,43,44
     --sigmas 0.10,0.223 — budget-only change, justified by the first
     pass's finding that the G1 miss was budget-shaped (no plateau anywhere;
     KL mass in rare near-boundary/late-generation contexts). The two added
     eval instruments (within-level excess TV and per-site free-energy bias,
     both the DNFS gate's own constructions) are instruments only: the eval
     sampling protocol is unchanged and eval contexts are drawn on CPU RNG,
     so the sigma_c context set is identical across passes.

Free energy: the full unnormalised trajectory weight is
    w = Q0(traj) p~(X_1) / (base(X_1) P_model(traj)),
and the reference's assignment product (the trajectory constant
N_+!(d-N_+)!/d!) is exactly 1/C(d, N_+) = base(X_1), so they cancel:
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

import numpy as np
import torch
from experiments.constrained_hard_03.probes.gate_4x4 import (
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
    log_variance_loss,
    masked_count_and_budget,
    preconditioner_logit_diff,
    rollout_budget_masked,
    wdce_cross_entropy,
)
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)

# First-pass protocol; the second pass overrides steps/seeds/sigmas on the
# command line and nothing else.
LATTICE_SIDE = 4
N_SITES = 16
N_PLUS = 8


def configure_lattice(lattice_side):
    """Rebind the CLI lattice globals at call time.

    Callers keep n_plus_target explicit: a definition-time N_PLUS default would
    retain the 4x4 fibre after configure_lattice(8). Half-filling sets
    N_PLUS = N_SITES // 2.
    """
    global LATTICE_SIDE, N_SITES, N_PLUS
    if lattice_side < 2 or lattice_side % 2:
        raise ValueError(
            f"lattice_side must be even and >= 2 (the half-filled fibre "
            f"needs an even site count), got {lattice_side}"
        )
    LATTICE_SIDE = lattice_side
    N_SITES = lattice_side * lattice_side
    N_PLUS = N_SITES // 2


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

# arm -> (preconditioner mode, constrained). "Constrained" = the budget-masked
# reference on the C(16,8) fibre. Arm "u" is the unconstrained control: the
# paper's own masked diffusion on the free 4x4 Ising (uniform species-1/2
# reference, no clamp, zero-imputation preconditioner = their App. D.4) against
# full 2^16 enumeration, which reproduces MDNS's published Table-4 critical
# cell on this implementation, calibrating recipe parity internally.
ARMS = {
    "a": ("budget_tilted", True),
    "b": ("none", True),
    "c": ("unconstrained", True),
    "a2": ("budget_tilted_gated", True),  # forensics arm: V0 with
    # learnable scales (init 1.0)
    "u": ("unconstrained", False),  # unconstrained control
}

# First-pass G-cuts, computed for every (sigma, seed, arm) for continuity
# (later passes print them alongside, not re-adjudicated).
CUTS = {
    "energy_tv": 0.02,
    "conditional_kl": 0.01,
    "ess_fraction": 0.20,
    "plateau_ratio": 0.5,
    "late_error_ratio": 2.0,
}


def make_logit_fn(net, adjacency, sigma, mode):
    """(logit_fn, extra_trainables): the gated arm carries two learnable
    scale parameters alongside the trunk; every other arm's offset is a pure
    function."""
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


def draw_eval_contexts(
    slice_states, slice_log_p_cond, generator, n_contexts=EVAL_CONTEXTS
):
    """Corruption contexts from the population WDCE law with exact weights:
    terminal ~ exact fibre conditional, lambda ~ U(0,1), sites masked
    independently, empty masks redrawn. Drawn once on CPU RNG and shared by
    every arm, seed and pass, so all KL numbers are matched."""
    terminal_rows = torch.multinomial(
        slice_log_p_cond.exp(),
        n_contexts,
        replacement=True,
        generator=generator,
    )
    terminals = slice_states[terminal_rows].float()
    corruption_level = torch.rand(n_contexts, 1, generator=generator)
    mask = torch.rand(n_contexts, N_SITES, generator=generator) < corruption_level
    empty = ~mask.any(dim=1)
    while empty.any():
        mask[empty] = torch.rand(
            int(empty.sum()), N_SITES, generator=generator
        ) < torch.rand(int(empty.sum()), 1, generator=generator)
        empty = ~mask.any(dim=1)
    return terminals.masked_fill(mask, 0.0)


def exact_conditional_per_context(context, slice_states, slice_log_p_cond):
    """Exact Pr(X^i = +1 | unmasked part) for every masked site i of one
    context, by selecting the fibre states consistent with the unmasked pattern
    and renormalising."""
    unmasked = context != 0.0
    consistent = (slice_states[:, unmasked].float() == context[unmasked]).all(dim=1)
    log_p = slice_log_p_cond[consistent]
    posterior = torch.softmax(log_p, dim=0)
    plus_indicator = (slice_states[consistent] == 1).float()
    return posterior @ plus_indicator  # (n_sites,), valid at masked


def conditional_kl_and_late_error(
    logit_fn, contexts, slice_states, slice_log_p_cond, device, n_plus_target
):
    """Mean KL(exact || model) over (context, masked site) pairs — exact per
    context, no sampling floor — plus the late-generation (m <= 4) mean
    absolute error. The model conditional carries the same feasibility clamp
    generation uses (none for the unconstrained arm, n_plus_target=None), so
    the metric scores the sampler's law."""
    with torch.no_grad():
        logits = logit_fn(contexts.to(device)).cpu()
    masked_count, budget = masked_count_and_budget(
        contexts, n_plus_target if n_plus_target is not None else 0
    )
    kl_terms, late_errors = [], []
    for row, context in enumerate(contexts):
        masked_sites = (context == 0.0).nonzero(as_tuple=True)[0]
        exact_p_plus = exact_conditional_per_context(
            context, slice_states, slice_log_p_cond
        )[masked_sites]
        if n_plus_target is None:
            model_p_plus = torch.sigmoid(logits[row, masked_sites])
        else:
            model_p_plus = feasibility_clamped_p_plus(
                torch.sigmoid(logits[row, masked_sites]),
                budget[row].expand(len(masked_sites)),
                masked_count[row].expand(len(masked_sites)),
            )
        model_p_plus = model_p_plus.clamp(1e-12, 1 - 1e-12)
        p, q = exact_p_plus, model_p_plus
        kl = torch.where(p > 0, p * (p.log() - q.log()), torch.zeros_like(p))
        kl = kl + torch.where(
            p < 1,
            (1 - p) * ((1 - p).log() - (1 - q).log()),
            torch.zeros_like(p),
        )
        kl_terms.append(kl)
        if masked_count[row] <= LATE_GENERATION_MAX_MASKED:
            late_errors.append((p - model_p_plus).abs())
    mean_kl = torch.cat(kl_terms).mean().item()
    late_error = torch.cat(late_errors).mean().item() if late_errors else float("nan")
    return mean_kl, late_error


def evaluate_arm(
    logit_fn,
    target,
    slice_states,
    slice_log_p_cond,
    exact_hist,
    bins,
    sigma,
    device,
    run_dir,
    eval_rollouts,
    n_plus_target,
    save_artefacts=True,
    exact_log_z=None,
    free_energy_ref=None,
):
    """Eval rollouts -> ESS fraction, G0 count, energy-marginal TV, plus the two
    DNFS-shared instruments: within-level excess TV and per-site free-energy
    bias. Saves terminals + log-weights so a later instrument can rerun off
    artefacts instead of GPU.

    `slice_states=None` is the non-enumerable regime (8x8, fibre C(64,32) ~
    1.8e18): the within-level instrument needs the enumerated slice and is
    dropped rather than approximated, and `free_energy_ref` must then be
    supplied — the slice-TI constant, the same quantity
    `on_slice_free_energy_reference` computes, agreeing to 5 decimals where
    both are computable. ESS, off-fibre count and the energy TV only need the
    reference histogram, which comes from the certified chains at 8x8."""
    terminals_all, log_w_all = [], []
    generator = torch.Generator(device=device).manual_seed(EVAL_SEED)
    with torch.no_grad():
        for _ in range(max(1, eval_rollouts // EVAL_BATCH)):
            batch = min(EVAL_BATCH, eval_rollouts)
            terminals, rollout_log_prob = rollout_budget_masked(
                logit_fn, batch, N_SITES, n_plus_target, generator
            )
            terminals_all.append(terminals.cpu())
            log_w_all.append((target.log_prob(terminals) - rollout_log_prob).cpu())
    terminals = torch.cat(terminals_all)
    log_w = torch.cat(log_w_all)
    ess = ess_from_log_weights(log_w).item()
    weights = torch.softmax(log_w, dim=0)
    adjacency_cpu = target.A.cpu()
    model_hist = slice_energy_hist(terminals, weights, adjacency_cpu, bins)
    if save_artefacts:
        torch.save(
            {"terminals": terminals.to(torch.int8), "log_weights": log_w},
            run_dir / "eval_artefacts.pt",
        )

    if n_plus_target is None:
        # Free-space (arm-u) instruments = what MDNS's own tables report: ESS,
        # marginal TV vs enumeration, and the log-Z error of the self-normalised
        # estimate. The fibre instruments are undefined off the slice.
        log_z_estimate = torch.logsumexp(log_w, dim=0) - torch.log(
            torch.tensor(float(len(log_w)))
        )
        n_plus = ((terminals + 1) / 2).sum(dim=1)
        return {
            "eval_rollouts": len(terminals),
            "composition_mean": (n_plus / N_SITES).mean().item(),
            "composition_std": (n_plus / N_SITES).std().item(),
            "ess_fraction": ess / len(terminals),
            "energy_tv": energy_marginal_tv(model_hist, exact_hist),
            "log_z_estimate": log_z_estimate.item(),
            "log_z_exact": exact_log_z,
            "log_z_abs_error": abs(log_z_estimate.item() - exact_log_z),
        }

    n_plus = ((terminals + 1) / 2).sum(dim=1)
    off_fibre = int((n_plus != n_plus_target).sum().item())
    free_energy_model = free_energy_lb_estimate(log_w, sigma, N_SITES).item()
    if free_energy_ref is None:
        free_energy_ref = on_slice_free_energy_reference(
            _CpuTargetView(target), slice_states
        ).item()
    metrics = {
        "eval_rollouts": len(terminals),
        "off_fibre_count": off_fibre,
        "ess_fraction": ess / len(terminals),
        "energy_tv": energy_marginal_tv(model_hist, exact_hist),
        "free_energy_model": free_energy_model,
        "free_energy_ref": free_energy_ref,
        "free_energy_bias": free_energy_model - free_energy_ref,
    }
    if slice_states is not None:
        # Within-level uniformity requires the enumerated slice population;
        # sampled chains cannot supply that reference.
        slice_energies = _energy(slice_states.float(), adjacency_cpu)
        sample_energies = _energy(terminals, adjacency_cpu)
        levels = within_level_uniformity(
            terminals,
            weights,
            sample_energies,
            slice_states.float(),
            slice_energies,
        )
        metrics["max_level_excess"] = max(
            (level["excess"] for level in levels), default=float("nan")
        )
        metrics["within_level"] = levels
    return metrics


class _CpuTargetView:
    """CPU view of a possibly-GPU target for the slice free-energy reference;
    the slice tensors live on CPU throughout."""

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


# Shared with the swap-CTMC trainer; retain this re-export for tests/tools.
from discrete_flow_sampler.ema import ExponentialMovingAverage  # noqa: E402


def near_boundary_loss_weight(boost):
    """eta(context) = 1 + boost*1[b in {1, m-1}]: context-measurable, hence
    minimiser-safe (see wdce_cross_entropy), reallocating gradient towards the
    near-boundary contexts the first pass's KL decompositions localised."""

    def eta(corrupted):
        masked_count, budget = masked_count_and_budget(corrupted, N_PLUS)
        near = (budget == 1) | (budget == masked_count - 1)
        return 1.0 + boost * near.float()

    return eta


def train_arm(
    arm,
    mode,
    target,
    sigma,
    seed,
    steps,
    results_root,
    tag,
    device,
    boost=0.0,
    objective="wdce",
    replicates=None,
    optimiser_kind="adam",
    ema_decay=0.0,
    ema_warmup=False,
    *,
    n_plus_target,
):
    seed_everything(seed)
    net = MaskedConditionalNet(N_SITES).to(device)
    logit_fn, gated_offset = make_logit_fn(net, target.A, sigma, mode)
    trainables = list(net.parameters())
    if gated_offset is not None:
        gated_offset.to(device)
        trainables += list(gated_offset.parameters())
    optimiser_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[
        optimiser_kind
    ]
    optimiser = optimiser_class(trainables, lr=LEARNING_RATE)
    ema = (
        ExponentialMovingAverage(trainables, ema_decay, warmup=ema_warmup)
        if ema_decay > 0
        else None
    )
    replicates = replicates or CORRUPTION_REPLICATES
    context_weight = near_boundary_loss_weight(boost) if boost > 0 else None
    rollout_generator = torch.Generator(device=device).manual_seed(seed)
    corruption_generator = torch.Generator(device=device).manual_seed(seed + 1)

    sigma_tag = f"s{sigma:.3f}".replace("0.", "")
    run_dir = results_root / f"arm_{arm}_{mode}_{sigma_tag}_seed{seed}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_rows, train_off_fibre = [], 0
    started = time.time()
    for step in range(steps):
        if objective == "lv":
            loss, terminals, log_w = log_variance_loss(
                logit_fn,
                target.log_prob,
                TRAIN_ROLLOUTS_PER_STEP,
                N_SITES,
                n_plus_target,
                rollout_generator,
            )
        else:
            with torch.no_grad():
                terminals, rollout_log_prob = rollout_budget_masked(
                    logit_fn,
                    TRAIN_ROLLOUTS_PER_STEP,
                    N_SITES,
                    n_plus_target,
                    rollout_generator,
                )
                log_w = target.log_prob(terminals) - rollout_log_prob
        with torch.no_grad():
            weights = torch.softmax(log_w, dim=0)
            if n_plus_target is not None:
                train_off_fibre += int(
                    (((terminals + 1) / 2).sum(dim=1) != n_plus_target).sum().item()
                )
            ess_fraction = ess_from_log_weights(log_w).item() / TRAIN_ROLLOUTS_PER_STEP
        if objective == "wdce":
            loss = wdce_cross_entropy(
                logit_fn,
                terminals,
                weights,
                replicates,
                corruption_generator,
                context_loss_weight=context_weight,
            )
        optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.norm(
            torch.stack(
                [
                    parameter.grad.norm()
                    for parameter in trainables
                    if parameter.grad is not None
                ]
            )
        ).item()
        optimiser.step()
        if ema is not None:
            ema.update()
        log_rows.append(
            {
                "step": step,
                "loss": loss.item(),
                "train_ess_fraction": ess_fraction,
                "grad_norm": grad_norm,
            }
        )
    wall_clock = time.time() - started

    with open(run_dir / "training_log.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    if ema is not None:
        ema.swap_in()  # evaluation sees the EMA parameters (D.2.2)
    return (
        net,
        logit_fn,
        gated_offset,
        run_dir,
        log_rows,
        train_off_fibre,
        wall_clock,
        ema,
    )


def chain_reference_energies(probe_root, point, adjacency):
    """Pooled post-burn-in energies of the certified Kawasaki reference chains,
    the 8x8 stand-in for exact enumeration.

    Two conventions must match the enumerated path or the TV is meaningless:

    * Energy is `_energy(x, A) = x^T A x`, the same function
      `slice_energy_hist` applies to the model's samples, and what the probe's
      `observable_values("energy", ...)` resolves to.
    * Burn-in discard is the second half of each chain, the probe's convention.

    The chains are equilibrium draws, so they enter the histogram with uniform
    weight; there is no importance weight to carry.
    """
    chain_dirs = sorted((Path(probe_root) / "reference" / point).glob("chain_*"))
    if not chain_dirs:
        raise FileNotFoundError(
            f"no reference chains under {probe_root}/reference/{point}; the "
            f"8x8 reference block needs the certified probe chains"
        )
    pooled = []
    for chain_dir in chain_dirs:
        spins = np.load(chain_dir / "snapshots.npz")["spins"]
        states = torch.from_numpy(spins.astype(np.float32))
        energies = _energy(states, adjacency)
        pooled.append(energies[len(energies) // 2 :])
    return torch.cat(pooled)


def build_space(
    constrained,
    sigma,
    device,
    eval_contexts,
    *,
    probe_root=None,
    probe_point=None,
    free_energy_ref=None,
):
    """Everything an arm's training/eval needs that depends only on its state
    space: the fibre (constrained arms) or the free 2^d space (arm u).

    Two reference regimes:

    * Enumeration (`probe_root=None`, the 4x4 path). Exact both ways — the free
      space is 65,536 states, still enumerable, which makes the unconstrained
      control a like-for-like reproduction of the paper's 4x4 protocol.
    * Chain-referenced (8x8). C(64,32) ~ 1.8e18 rules enumeration out, so the
      energy histogram comes from the certified 8-chain Kawasaki reference and
      the free-energy reference from the slice-TI constant. Everything needing
      the enumerated slice as a population is dropped rather than approximated:
      the exact fibre conditional, hence `contexts`, hence the conditional-KL
      and late-generation instruments, and within-level uniformity.
    """
    if probe_root is not None and not constrained:
        raise ValueError(
            "the chain reference is a fixed-composition (fibre) reference; "
            "the unconstrained control arm has no chain counterpart"
        )
    if constrained:
        target = FixedCompositionIsingTarget(
            D=LATTICE_SIDE,
            sigma=sigma,
            target_composition=0.5,
            device=device,
        )
        cpu_target = FixedCompositionIsingTarget(
            D=LATTICE_SIDE, sigma=sigma, target_composition=0.5
        )
    else:
        target = IsingTarget(D=LATTICE_SIDE, sigma=sigma, device=device)
        cpu_target = IsingTarget(D=LATTICE_SIDE, sigma=sigma)
    adjacency_cpu = target.A.cpu()
    if probe_root is not None:
        if free_energy_ref is None:
            raise ValueError(
                "the chain reference supplies no free-energy reference of "
                "its own; pass the slice-TI constant for this (sigma, size)"
            )
        reference_energies = chain_reference_energies(
            probe_root, probe_point, adjacency_cpu
        )
        bins = _categorical_energy_bins(reference_energies)
        uniform = torch.ones(len(reference_energies))
        # The chain reference is already energies, so histogram them directly
        # against the bin edges the model's samples will use.
        lower, upper = bins[:-1], bins[1:]
        membership = (reference_energies[:, None] >= lower[None, :]) & (
            reference_energies[:, None] < upper[None, :]
        )
        mass = (membership.float() * (uniform / uniform.sum())[:, None]).sum(0)
        return {
            "target": target,
            "states": None,
            "log_p": None,
            "bins": bins,
            "exact_hist": (0.5 * (lower + upper), mass),
            "contexts": None,
            "exact_log_z": None,
            "free_energy_ref": free_energy_ref,
            "n_plus_target": N_PLUS,
        }
    states = enumerate_states(N_SITES)
    log_pi = exact_log_probs(cpu_target, states)
    if constrained:
        space_states, space_log_p = conditional_pmf_at_composition(
            states, log_pi, N_PLUS
        )
        exact_log_z = None
    else:
        space_states = states
        space_log_p = log_pi  # exact_log_probs is already normalised
        # log Z must come from the unnormalised densities: exact_log_probs
        # returns log-probs with logsumexp = 0 by contract.
        exact_log_z = torch.logsumexp(cpu_target.log_prob(states.float()), dim=0).item()
    space_energies = _energy(space_states.float(), adjacency_cpu)
    bins = _categorical_energy_bins(space_energies)
    exact_hist = slice_energy_hist(
        space_states.float(), space_log_p.exp(), adjacency_cpu, bins
    )
    # CPU RNG on purpose: the sigma_c context set must match the first pass
    contexts = draw_eval_contexts(
        space_states,
        space_log_p,
        torch.Generator().manual_seed(EVAL_SEED),
        eval_contexts,
    )
    return {
        "target": target,
        "states": space_states,
        "log_p": space_log_p,
        "bins": bins,
        "exact_hist": exact_hist,
        "contexts": contexts,
        "exact_log_z": exact_log_z,
        # None = "derive it from the enumerated slice in evaluate_arm";
        # only the chain-referenced regime supplies a constant.
        "free_energy_ref": None,
        "n_plus_target": N_PLUS if constrained else None,
    }


def run_slate(
    sigma,
    seeds,
    arms,
    steps,
    results_root,
    tag,
    device,
    eval_rollouts,
    eval_contexts,
    boost=0.0,
    objective="wdce",
    replicates=None,
    optimiser_kind="adam",
    ema_decay=0.0,
    ema_warmup=False,
    probe_root=None,
    probe_point=None,
    free_energy_ref=None,
):
    spaces = {}
    for arm in arms:
        _, constrained = ARMS[arm]
        if constrained not in spaces:
            spaces[constrained] = build_space(
                constrained,
                sigma,
                device,
                eval_contexts,
                probe_root=probe_root,
                probe_point=probe_point,
                free_energy_ref=free_energy_ref,
            )

    reports = {}
    for seed in seeds:
        for arm in arms:
            mode, constrained = ARMS[arm]
            space = spaces[constrained]
            target = space["target"]
            slice_states, slice_log_p_cond = space["states"], space["log_p"]
            exact_hist, bins = space["exact_hist"], space["bins"]
            contexts = space["contexts"]
            n_plus_target = space["n_plus_target"]
            seed_everything(seed)
            init_net = MaskedConditionalNet(N_SITES).to(device)
            init_fn, init_gated = make_logit_fn(init_net, target.A, sigma, mode)
            if init_gated is not None:
                init_gated.to(device)
            # No enumerated slice -> no exact fibre conditional -> the KL and
            # late-generation instruments are undefined. NaN keeps the schema.
            init_kl, init_late_error = (
                conditional_kl_and_late_error(
                    init_fn,
                    contexts,
                    slice_states,
                    slice_log_p_cond,
                    device,
                    n_plus_target,
                )
                if contexts is not None
                else (float("nan"), float("nan"))
            )
            print(
                f"[gate] sigma={sigma} seed={seed} arm {arm} ({mode}): "
                f"training {steps} steps ...",
                flush=True,
            )
            (
                net,
                logit_fn,
                gated_offset,
                run_dir,
                log_rows,
                train_off_fibre,
                wall,
                ema,
            ) = train_arm(
                arm,
                mode,
                target,
                sigma,
                seed,
                steps,
                results_root,
                tag,
                device,
                boost,
                objective,
                replicates,
                optimiser_kind,
                ema_decay,
                ema_warmup,
                n_plus_target=n_plus_target,
            )
            trained_kl, trained_late_error = (
                conditional_kl_and_late_error(
                    logit_fn,
                    contexts,
                    slice_states,
                    slice_log_p_cond,
                    device,
                    n_plus_target,
                )
                if contexts is not None
                else (float("nan"), float("nan"))
            )
            eval_metrics = evaluate_arm(
                logit_fn,
                target,
                slice_states,
                slice_log_p_cond,
                exact_hist,
                bins,
                sigma,
                device,
                run_dir,
                eval_rollouts,
                n_plus_target=n_plus_target,
                exact_log_z=space["exact_log_z"],
                free_energy_ref=space["free_energy_ref"],
            )
            # Paper protocol: headline EMA eval plus the raw-weight eval.
            # At 2,000 steps a 0.9999 shadow still holds ~82% of init.
            raw_param_eval = None
            if ema is not None:
                ema.swap_out()
                raw_metrics = evaluate_arm(
                    logit_fn,
                    target,
                    slice_states,
                    slice_log_p_cond,
                    exact_hist,
                    bins,
                    sigma,
                    device,
                    run_dir,
                    eval_rollouts,
                    save_artefacts=False,
                    n_plus_target=n_plus_target,
                    exact_log_z=space["exact_log_z"],
                    free_energy_ref=space["free_energy_ref"],
                )
                raw_kl, raw_late = (
                    conditional_kl_and_late_error(
                        logit_fn,
                        contexts,
                        slice_states,
                        slice_log_p_cond,
                        device,
                        n_plus_target,
                    )
                    if contexts is not None
                    else (float("nan"), float("nan"))
                )
                raw_param_eval = {
                    key: value
                    for key, value in raw_metrics.items()
                    if key != "within_level"
                }
                raw_param_eval["trained_conditional_kl"] = raw_kl
                raw_param_eval["trained_late_generation_error"] = raw_late
                ema.swap_in()
            report = {
                "arm": arm,
                "preconditioner": mode,
                "sigma": sigma,
                "seed": seed,
                "steps": steps,
                "device": str(device),
                "near_boundary_boost": boost,
                "objective": objective,
                "replicates": replicates or CORRUPTION_REPLICATES,
                "optimiser": optimiser_kind,
                "ema_decay": ema_decay,
                "ema_warmup": ema_warmup,
                "learned_gates": (
                    {
                        "gate_budget": gated_offset.gate_budget.item(),
                        "gate_field": gated_offset.gate_field.item(),
                    }
                    if gated_offset is not None
                    else None
                ),
                "train_off_fibre_count": train_off_fibre,
                "train_wall_clock_s": wall,
                "final_loss": log_rows[-1]["loss"],
                "plateau_step": plateau_step(
                    [row["train_ess_fraction"] for row in log_rows]
                ),
                "final_train_ess_fraction": log_rows[-1]["train_ess_fraction"],
                "raw_param_eval": raw_param_eval,
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
            tail = (
                f"logZerr {report['log_z_abs_error']:.4f}"
                if n_plus_target is None
                else f"FEbias {report['free_energy_bias']:+.4f} "
                f"G0 off-fibre {report['off_fibre_count']}"
            )
            print(
                f"[gate] sigma={sigma} seed={seed} arm {arm}: "
                f"KL {trained_kl:.4f} TV {report['energy_tv']:.4f} "
                f"ESS {report['ess_fraction']:.3f} " + tail,
                flush=True,
            )
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
                for r in (a, b, c)
                if r is not None
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
                or a["plateau_step"] <= CUTS["plateau_ratio"] * b["plateau_step"]
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
    parser.add_argument("--results-dir", default="results/03_hard/mdns_budget_gate_4x4")
    parser.add_argument("--tag", default="20260813-gate")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--seeds", default=str(TRAIN_SEED), help="comma-separated")
    parser.add_argument(
        "--sigmas", default=str(SIGMA), help="comma-separated operating points"
    )
    parser.add_argument("--arms", default="a,b,c")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--eval-rollouts",
        type=int,
        default=EVAL_ROLLOUTS,
        help="plumbing smoke only; the protocol value is the default",
    )
    parser.add_argument(
        "--eval-contexts", type=int, default=EVAL_CONTEXTS, help="plumbing smoke only"
    )
    parser.add_argument(
        "--near-boundary-boost",
        type=float,
        default=0.0,
        help="eta(context) boost kappa on b in {1, m-1} "
        "contexts; 0 = the protocol loss (default)",
    )
    parser.add_argument(
        "--objective",
        choices=["wdce", "lv"],
        default="wdce",
        help="lv = constrained F_LV (their strongest 4x4 objective)",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="WDCE corruption replicates R (default: the "
        "protocol's 2; paper 4x4 uses 16, "
        "ablation-insensitive on 8-64)",
    )
    parser.add_argument("--optimiser", choices=["adam", "adamw"], default="adam")
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="0 = off (the protocol); paper always "
        "uses 0.9999 and evaluates the EMA weights",
    )
    parser.add_argument(
        "--ema-warmup",
        action="store_true",
        help="bias-correction warmup schedule "
        "min(decay, (1+t)/(10+t)); off = the "
        "paper-literal plain shadow",
    )
    parser.add_argument(
        "--lattice-side",
        type=int,
        default=LATTICE_SIDE,
        help="L for the LxL torus. 4 = the "
        "enumerable size; 8 requires --probe-root and "
        "--free-energy-ref, since C(64,32) rules "
        "enumeration out",
    )
    parser.add_argument(
        "--probe-root",
        default=None,
        help="certified Kawasaki reference chains, e.g. "
        "results/kawasaki_probe. Setting this switches "
        "the energy reference from enumeration to "
        "chains and DROPS the enumeration-only "
        "instruments (conditional KL, late-generation "
        "error, within-level uniformity)",
    )
    parser.add_argument(
        "--probe-point",
        default=None,
        help="operating point under the probe root: 'sc' (sigma_c) or 's010'",
    )
    parser.add_argument(
        "--free-energy-ref",
        type=float,
        default=None,
        help="slice-TI F/d for this (sigma, size), the "
        "chain regime's stand-in for the enumerated "
        "on-slice reference; 8x8 sigma_c = -1.90410",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.lattice_side != LATTICE_SIDE:
        configure_lattice(args.lattice_side)
    if (args.probe_root is None) != (args.probe_point is None):
        parser.error("--probe-root and --probe-point go together")

    results_root = Path(args.results_dir)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    arms = args.arms.split(",")
    all_reports = {}
    for sigma in (float(value) for value in args.sigmas.split(",")):
        reports = run_slate(
            sigma,
            seeds,
            arms,
            args.steps,
            results_root,
            args.tag,
            device,
            args.eval_rollouts,
            args.eval_contexts,
            args.near_boundary_boost,
            args.objective,
            args.replicates,
            args.optimiser,
            args.ema_decay,
            args.ema_warmup,
            probe_root=args.probe_root,
            probe_point=args.probe_point,
            free_energy_ref=args.free_energy_ref,
        )
        all_reports[f"sigma_{sigma}"] = {
            "reports": reports,
            "first_pass_cuts_for_continuity": first_pass_cut_table(reports, seeds),
        }
    with open(results_root / f"verdict_{args.tag}.json", "w") as handle:
        json.dump(all_reports, handle, indent=2)
    print("[gate] slate complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
