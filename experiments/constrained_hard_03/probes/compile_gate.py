"""Certification gate for compiled swap heads.

Inductor kernels are backend-specific, so a pass on local CPU Inductor
certifies nothing about the training GPU stack; run this on the GPU venue
before compiled cells ship.

Two checks:

1. The two-hole-patch head test file runs with every `_head`-factory head
   compiled in place; those tests encode antisymmetry, per-pair oracles and
   Kolmogorov contracts, so a compiled kernel that changes the math fails here.
2. Gradient parity, eager vs compiled: `loss_swap` forward + backward on a
   seeded 4x4 cell from identically-initialised heads. The loss must agree to
   1e-5 and every gradient to 1e-5 relative on its norm. `pair_mlp.2.bias` is
   compared absolutely: its gradient is structurally zero — the shared
   last-layer bias cancels in the antisymmetric readout H_ij = P_ij − P_ji —
   so a relative metric on rounding residue (~1e-8) would flag a non-error.

Pass criterion: pytest exit code 0 and parity within tolerance; the gate
certifies "same math", never bit-parity.

Run locally (CPU inductor) or on the GPU venue:
    pixi run -e dev python -m experiments.constrained_hard_03.probes.compile_gate
    pixi run -e dev modal run -m \
        experiments.constrained_hard_03.modal_app::compile_gate
"""

import sys

import pytest
import torch

from discrete_flow_sampler.samplers.swap_kolmogorov import loss_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

# Structurally zero: the shared last-layer bias cancels in H_ij = P_ij - P_ji.
STRUCTURAL_ZERO_PARAMS = ("pair_mlp.2.bias",)
LOSS_TOLERANCE = 1e-5
GRAD_RELATIVE_TOLERANCE = 1e-5
GRAD_ABSOLUTE_TOLERANCE = 1e-6


class _CompileHeadsPlugin:
    """Compile each test-factory head after collection imports the test module."""

    def pytest_collection_finish(self, session):
        import tests.test_two_hole_patch_swap_head as head_tests

        original_factory = head_tests._head

        def compiled_factory(*args, **kwargs):
            head = original_factory(*args, **kwargs)
            head.compile()
            return head

        head_tests._head = compiled_factory


def run_head_tests_compiled() -> bool:
    exit_code = pytest.main(
        ["-q", "tests/test_two_hole_patch_swap_head.py", "-p", "no:cacheprovider"],
        plugins=[_CompileHeadsPlugin()],
    )
    return exit_code == 0


def _fresh_head(device):
    # Same construction as the head tests' factory (seeded), on `device`.
    import tests.test_two_hole_patch_swap_head as head_tests

    return head_tests._head().to(device).train()


def run_gradient_parity(device) -> tuple[bool, list[str]]:
    target = FixedCompositionIsingTarget(
        D=4, sigma=0.223, target_composition=0.5, device=device
    )
    torch.manual_seed(1)
    x = target.sample_base(16, device=device)
    t = torch.rand(16, device=device)
    dt_log_Zt = torch.zeros(16, device=device)

    losses, grads = [], []
    for compile_head in (False, True):
        head = _fresh_head(device)
        if compile_head:
            head.compile()
        loss = loss_swap(x, t, dt_log_Zt, head, target)
        loss.backward()
        losses.append(loss.detach())
        # Backbone parameters outside the head's forward path carry no grad;
        # kept as None so a grad existing on one side only is a parity failure.
        grads.append(
            {
                name: None if p.grad is None else p.grad.detach().clone()
                for name, p in head.named_parameters()
            }
        )

    failures = []
    if not torch.allclose(losses[0], losses[1], atol=LOSS_TOLERANCE):
        failures.append(
            f"loss mismatch: eager {losses[0].item():.8f} vs "
            f"compiled {losses[1].item():.8f}"
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
        if name in STRUCTURAL_ZERO_PARAMS:
            for label, grad in (("eager", eager_grad), ("compiled", compiled_grad)):
                if grad.abs().max() > GRAD_ABSOLUTE_TOLERANCE:
                    failures.append(
                        f"{name} ({label}): structural zero violated, "
                        f"|grad|_max = {grad.abs().max().item():.3e}"
                    )
            continue
        relative_gap = (
            (eager_grad - compiled_grad).norm() / eager_grad.norm().clamp_min(1e-30)
        ).item()
        if relative_gap > GRAD_RELATIVE_TOLERANCE:
            failures.append(f"{name}: relative grad gap {relative_gap:.3e}")
    return not failures, failures


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[compile_gate] device = {device} "
        f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})"
    )

    tests_pass = run_head_tests_compiled()
    print(
        f"[compile_gate] head tests with compiled heads: "
        f"{'PASS' if tests_pass else 'FAIL'}"
    )

    parity_pass, failures = run_gradient_parity(device)
    print(
        f"[compile_gate] gradient parity eager vs compiled: "
        f"{'PASS' if parity_pass else 'FAIL'}"
    )
    for failure in failures:
        print(f"[compile_gate]   {failure}")

    gate_pass = tests_pass and parity_pass
    print(f"[compile_gate] GATE {'PASSED' if gate_pass else 'FAILED'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
