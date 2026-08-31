"""d64 GFN launch bench: compile parity gate + rollout wall-clock (s94).

Runs ONCE on the training venue's GPU before the 8x8 wave ships (the house
compile-certification pattern, compile_gate.py): inductor generates
different kernels per backend, so the d16 CPU tests certify nothing about
the venue stack. The d64 cells ship with compile_policy=True; this gate is
the tripwire that stops the launch if the venue's kernels change the math.

Part 1 — numerical parity, eager vs compiled scoring. Both objectives'
losses (forward + backward) from identically-initialised d64 policies on
identical seeded batches. Loss must agree to 1e-5; every gradient to 1e-5
relative on its norm (grads absent from an objective — e.g. log_z under
FL-DB — must be absent on both sides).

Part 2 — wall-clock, PHASE AND BATCH NAMED (the B=128 bench-ranking lesson:
a verdict is scoped to its phase). Three timings at d64:
  * rollout, B=128  — the per-step training draw (KV-cached, eager)
  * rollout, B=512  — the eval chunk
  * train step, B=128 — rollout + loss forward/backward + optimiser step
The train-step timing x 50k projects the job wall-time; the rollout
timings seed the wall-clock-per-effective-sample column that rides beside
FLOP/es in the table (the latency counterpoint: FLOP/es prices arithmetic,
not the AR policy's d sequential one-token kernels).

    pixi run -e dev python -m experiments.constrained_hard_03.gfn_launch_bench
"""
import argparse
import sys
import time

import torch

from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS
from experiments.constrained_hard_03.run_gfn import (
    _loss_and_train_diagnostics,
    build_optimiser,
    build_target_and_policy,
)

LOSS_TOLERANCE = 1e-5
GRAD_RELATIVE_TOLERANCE = 1e-5


def _grads_by_name(policy):
    return {
        name: (None if p.grad is None else p.grad.detach().clone())
        for name, p in policy.named_parameters()
    }


def parity_gate(cfg, device) -> bool:
    """Eager vs compiled loss + gradient parity on one seeded batch."""
    ok = True
    # Two identically-initialised policies; ONE shared batch sampled from
    # the eager policy (the sampler is eager in both configs, but sharing
    # the batch removes even rollout nondeterminism from the comparison).
    torch.manual_seed(0)
    _, eager_policy = build_target_and_policy(cfg, device)
    torch.manual_seed(0)
    target, compiled_policy = build_target_and_policy(cfg, device)
    compiled_policy.site_log_probs = torch.compile(compiled_policy.site_log_probs)
    if cfg.with_flow_head:
        compiled_policy.site_log_probs_and_flow_residuals = torch.compile(
            compiled_policy.site_log_probs_and_flow_residuals
        )
    with torch.no_grad():
        spins, _ = eager_policy.sample(cfg.batch_size, epsilon=cfg.epsilon)

    losses, grads = [], []
    for policy in (eager_policy, compiled_policy):
        loss, _ = _loss_and_train_diagnostics(cfg, policy, target, spins)
        policy.zero_grad(set_to_none=True)
        loss.backward()
        losses.append(float(loss.item()))
        grads.append(_grads_by_name(policy))

    loss_gap = abs(losses[0] - losses[1])
    if loss_gap > LOSS_TOLERANCE:
        print(f"  FAIL loss gap {loss_gap:.2e} > {LOSS_TOLERANCE:.0e}")
        ok = False
    for name in grads[0]:
        eager_grad, compiled_grad = grads[0][name], grads[1][name]
        if (eager_grad is None) != (compiled_grad is None):
            print(f"  FAIL grad presence differs on {name}")
            ok = False
            continue
        if eager_grad is None:
            continue
        norm = eager_grad.norm().item()
        gap = (eager_grad - compiled_grad).norm().item()
        if gap > GRAD_RELATIVE_TOLERANCE * max(norm, 1e-12):
            print(f"  FAIL grad {name}: |Δ|={gap:.2e} vs norm {norm:.2e}")
            ok = False
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {cfg.objective}: loss gap {loss_gap:.2e}")
    return ok


def _timed(fn, warmup=3, reps=10, device="cuda"):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def wall_clock_bench(cfg, device):
    torch.manual_seed(0)
    target, policy = build_target_and_policy(cfg, device)
    if cfg.compile_policy:
        policy.site_log_probs = torch.compile(policy.site_log_probs)
        if cfg.with_flow_head:
            policy.site_log_probs_and_flow_residuals = torch.compile(
                policy.site_log_probs_and_flow_residuals
            )
    optimiser = build_optimiser(cfg, policy)

    for batch, phase in ((128, "training rollout"), (512, "eval chunk")):
        seconds = _timed(
            lambda: policy.sample(batch, epsilon=cfg.epsilon), device=device
        )
        print(f"  rollout B={batch} ({phase}): {seconds*1e3:.1f} ms "
              f"= {seconds/batch*1e6:.1f} us/sample")

    def train_step():
        spins, _ = policy.sample(cfg.batch_size, epsilon=cfg.epsilon)
        loss, _ = _loss_and_train_diagnostics(cfg, policy, target, spins)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    seconds = _timed(train_step, device=device)
    projected_hours = seconds * cfg.n_steps / 3600
    print(f"  train step B={cfg.batch_size}: {seconds*1e3:.1f} ms "
          f"-> {cfg.n_steps} steps ~= {projected_hours:.2f} h "
          f"(+ eval_every rollouts on top)")


# Inductor kernels differ per SIZE as well as per backend, so each rung's
# sigma_c centres gate their own launch (the d256 wave must not ride the
# d64 certification).
_RUNG_GATE_CELLS = {
    "d64": "GFN_d64_c50_s220_{objective}_50k_par",
    "d256": "GFN_d256_c50_s220_{objective}_100k_par",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="smoke the script off-venue; the gate only "
                             "certifies the stack it runs on")
    parser.add_argument("--rung", choices=sorted(_RUNG_GATE_CELLS),
                        default="d64",
                        help="which rung's cells to gate and bench")
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("no CUDA device: run on the launch venue (or --allow-cpu)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    all_ok = True
    for objective in ("tb", "fldb"):
        cfg = GFN_CONFIGS[_RUNG_GATE_CELLS[args.rung].format(objective=objective)]
        print(f"\n=== {cfg.name} ===")
        print("compile parity gate:")
        all_ok &= parity_gate(cfg, device)
        print("wall clock:")
        wall_clock_bench(cfg, device)

    if not all_ok:
        sys.exit("PARITY GATE FAILED — do not launch compiled cells")
    print(f"\nGATE PASSED on this stack; {args.rung} wave may ship compiled.")


if __name__ == "__main__":
    main()
