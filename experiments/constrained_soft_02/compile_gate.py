"""D=4 compile-parity gate for the soft house recipe.

compile_model=True enters the soft chapter with the house recipe, and
compile has priors on this codebase (~40% catastrophic seeds on the
factorised chassis at sigma_c), so leTF + channel + compile is gated
before any fan-out. Two parts, the GFN launch-bench pattern:

1. **Single-batch parity, eager vs compiled.** Identically-seeded models
   (the channel wrapper's zero-init gains make init bit-identical to the
   parent), one seeded batch through the real training loss
   (`kolmogorov.loss` -> residual_lenet, Eq. 10). Loss gap and every
   parameter gradient must agree to 1e-5 relative — the sharp test: a
   kernel that changes the math fails here before any optimiser
   amplification can excuse it.
2. **Short matched-seed train pair.** The full `train` entry on the gate
   config and its eager twin, n_steps cut to GATE_TRAIN_STEPS, same seed.
   Catches what single-step parity cannot: divergence entering through
   the rollout -> replay-buffer -> optimiser path, which is where the
   factorised chassis's compile failures actually lived. The 1e-5-class
   criterion applies over the PRE-AMPLIFICATION window only: this
   codebase has measured that a 2-ULP step-0 gradient difference
   decorrelates a 50k run entirely (identical config + seed gave ESS
   0.423 vs 0.899 across venues), so no same-math kernel pair can hold a
   1e-5 trace gap over hundreds of optimiser steps. In the reference D=4
   run the pair was bit-identical through step 25, showed its first
   representable gap at step ~50 (4.7e-7), and amplified to O(1e-1) by
   step 200 while both arms trained healthily — that profile IS the
   same-math signature. The late trace therefore gets a health check
   (finite everywhere, both arms' loss clearly declined), not a parity
   tolerance; a genuine compile pathology fails part 1, breaks the
   early window, or shows up as one arm not training.

Pass = both parts within tolerance. This CPU pass checks the compiled
graph's math; the venue's CUDA backend is checked separately by a single
d64 run on the venue before the family fans out (kernels differ per
backend).

Run:
    pixi run -e dev python -m experiments.constrained_soft_02.compile_gate
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.dnfs_baseline_01.run import _build_model, train
from discrete_flow_sampler.samplers.kolmogorov import loss as kolmogorov_loss
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import IsingTarget

from .configs import CONFIGS

GATE_CFG = CONFIGS["S2_d4_c05_l50_letf_house_gate"]
EAGER_CFG = CONFIGS["S2_d4_c05_l50_letf_house_gate_eager"]
RELATIVE_TOLERANCE = 1e-5
GATE_TRAIN_STEPS = 300          # 3 outer cycles: rollout + replay both exercised
GATE_OUTPUT_DIR = "results/02_constrained_soft"
GATE_TAG = "gate0e"             # fixed tag: a rerun resumes/skips, never forks

# The structurally-zero gradient (the hard compile gate's pair_mlp.2.bias
# case, re-derived here for this architecture): a key
# projection's bias adds the same vector b to every key, so for query i
# each score gains the identical constant q_i . b / sqrt(d_k), and softmax
# over keys removes any per-query constant — d loss / d b == 0 exactly,
# and only float rounding residue survives (measured 1.1e-6 on loss scale
# 1.7e3 in the reference D=4 run). A relative test on that residue flags a
# non-error, so any parameter ending in one of these suffixes is instead
# asserted SMALL on both sides.
STRUCTURAL_ZERO_SUFFIXES = ("k_proj.bias",)
STRUCTURAL_ZERO_ABSOLUTE_TOLERANCE = 1e-4

# Trace steps over which the matched-seed pair must agree to 1e-5-class:
# the window before Adam + replay feedback amplifies ULP-level rounding
# into macroscopic separation (first representable gap at step ~50 in the
# reference D=4 run; module docstring has the measured profile).
PRE_AMPLIFICATION_STEPS = 50
# Health floor for the late trace: both arms' trailing-mean loss must sit
# well below the shared step-0 loss (a factor-4 decline over 300 steps is
# far under what either healthy arm achieved — measured ~25-45x —
# while a dead arm stays at or above its start).
HEALTH_DECLINE_FACTOR = 4.0


def _gate_target(device):
    cfg = GATE_CFG.ising
    return IsingTarget(
        D=cfg.D, sigma=cfg.sigma, bias=cfg.bias, device=device,
        target_composition=cfg.target_composition,
        composition_penalty_strength=cfg.composition_penalty_strength,
        base_composition=cfg.base_composition,
        log_ratio_clamp=cfg.log_ratio_clamp,
    )


def run_single_batch_parity(device) -> tuple[bool, list[str]]:
    target = _gate_target(device)
    losses, grads = [], []
    for cfg in (EAGER_CFG, GATE_CFG):
        seed_everything(0)
        model = _build_model(cfg, target)
        torch.manual_seed(1)
        x = target.sample_base(GATE_CFG.train.batch_size, device=device)
        t = torch.rand(x.shape[0], device=device)
        dt_log_Zt = torch.zeros(x.shape[0], device=device)
        loss_value = kolmogorov_loss(x, t, dt_log_Zt, model, target)
        loss_value.backward()
        losses.append(loss_value.detach())
        grads.append(
            {name: None if p.grad is None else p.grad.detach().clone()
             for name, p in model.named_parameters()}
        )

    failures = []
    loss_gap = (losses[0] - losses[1]).abs().item()
    loss_scale = max(1.0, losses[0].abs().item())
    if loss_gap > RELATIVE_TOLERANCE * loss_scale:
        failures.append(
            f"loss gap {loss_gap:.3e} on scale {loss_scale:.3e} "
            f"(eager {losses[0].item():.8f} vs compiled {losses[1].item():.8f})"
        )
    for name, eager_grad in grads[0].items():
        compiled_grad = grads[1][name]
        if (eager_grad is None) != (compiled_grad is None):
            failures.append(
                f"{name}: grad exists on only one side "
                f"(eager={eager_grad is not None}, "
                f"compiled={compiled_grad is not None})"
            )
            continue
        if eager_grad is None:
            continue
        if name.endswith(STRUCTURAL_ZERO_SUFFIXES):
            for label, grad in (("eager", eager_grad),
                                ("compiled", compiled_grad)):
                if grad.abs().max() > STRUCTURAL_ZERO_ABSOLUTE_TOLERANCE:
                    failures.append(
                        f"{name} ({label}): structural zero violated, "
                        f"|grad|_max = {grad.abs().max().item():.3e}")
            continue
        gap = (eager_grad - compiled_grad).norm().item()
        norm = eager_grad.norm().item()
        if gap > RELATIVE_TOLERANCE * max(norm, 1e-12):
            failures.append(f"{name}: |grad gap| {gap:.3e} vs norm {norm:.3e}")
    print(f"[gate] single-batch loss gap {loss_gap:.3e} "
          f"(loss scale {loss_scale:.3e})")
    return not failures, failures


def _short_cfg(cfg):
    return replace(cfg, train=replace(cfg.train, n_steps=GATE_TRAIN_STEPS))


def run_train_pair() -> tuple[bool, str]:
    run_dirs = {}
    for cfg in (GATE_CFG, EAGER_CFG):
        run_dirs[cfg.name] = train(
            _short_cfg(cfg), seed=42, output_dir=GATE_OUTPUT_DIR,
            use_wandb=False, tag=GATE_TAG,
        )
    traces = {
        name: pd.read_csv(Path(run_dir) / "training_log.csv")
        for name, run_dir in run_dirs.items()
    }
    compiled = traces[GATE_CFG.name].set_index("step")["loss"]
    eager = traces[EAGER_CFG.name].set_index("step")["loss"]
    if not compiled.index.equals(eager.index):
        return False, "trace step grids differ between the pair"
    scale = pd.concat([compiled.abs(), eager.abs()], axis=1).max(axis=1)
    relative_gap = (compiled - eager).abs() / scale.clip(lower=1e-12)

    early_gap = relative_gap.iloc[:PRE_AMPLIFICATION_STEPS].max()
    finite = bool(np.isfinite(compiled).all() and np.isfinite(eager).all())
    declines = {
        name: trace.iloc[0] / trace.iloc[-20:].mean()
        for name, trace in (("compiled", compiled), ("eager", eager))
    }
    healthy = finite and all(
        ratio >= HEALTH_DECLINE_FACTOR for ratio in declines.values())

    summary = (
        f"pre-amplification (first {PRE_AMPLIFICATION_STEPS} steps) max "
        f"relative loss gap {early_gap:.3e}; full-trace max "
        f"{relative_gap.max():.3e} @ step {relative_gap.idxmax()} "
        f"(butterfly statistic, reported not gated); "
        f"health: finite={finite}, step0/final-20 decline "
        f"compiled {declines['compiled']:.1f}x eager {declines['eager']:.1f}x"
    )
    return early_gap <= RELATIVE_TOLERANCE and healthy, summary


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gate] device = {device}")

    parity_pass, failures = run_single_batch_parity(device)
    print(f"[gate] part 1 single-batch parity: "
          f"{'PASS' if parity_pass else 'FAIL'}")
    for failure in failures:
        print(f"[gate]   {failure}")

    pair_pass, summary = run_train_pair()
    print(f"[gate] part 2 {summary}")
    print(f"[gate] part 2 matched-seed train pair: "
          f"{'PASS' if pair_pass else 'FAIL'}")

    gate_pass = parity_pass and pair_pass
    print(f"[gate] GATE {'PASSED' if gate_pass else 'FAILED'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
