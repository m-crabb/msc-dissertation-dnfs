"""Chunked head forward in the scout's `measure`: the unchunked call fed all
n_grid x batch trajectory states through the mask_one head at once (d anchor
copies each), which OOM'd the d=64 checkpoint scout on a 40 GB A100
(2026-07-06). Chunking the batch dim is per-sample independent (attention runs
over sites within a sample), but not bit-exact: BLAS blocking depends on the
batch shape, so fp32 reductions carry ~1e-9 residue — hence ATOL, not `==`
(same pattern as `test_swap_head_vectorised._assert_matches`)."""

import torch
from experiments.constrained_hard_03.probes.scout_euler_budget import (
    _build_head,
    measure,
)

from discrete_flow_sampler.targets.ising import SIGMA_C, FixedCompositionIsingTarget

ATOL = 1e-5


def _assert_report_matches(got, want, label):
    """Recursive ATOL comparison over measure()'s nested report dict."""
    if isinstance(want, dict):
        assert got.keys() == want.keys(), label
        for key in want:
            _assert_report_matches(got[key], want[key], f"{label}.{key}")
    elif isinstance(want, list):
        assert len(got) == len(want), label
        for index, item in enumerate(want):
            _assert_report_matches(got[index], item, f"{label}[{index}]")
    elif isinstance(want, float):
        assert abs(got - want) < ATOL, f"{label}: {got} vs {want}"
    else:
        assert got == want, f"{label}: {got} vs {want}"


def test_measure_chunked_matches_single_slice():
    """Chunk 4 over 10 states -> slices of 4/4/2; same report to ATOL as one slice."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(
        D=4, sigma=SIGMA_C, target_composition=0.5, device="cpu"
    )
    head = _build_head("mask_one", target.d, "cpu")
    states = target.sample_base(10, device="cpu")
    t_values = torch.linspace(0.0, 1.0, 10)

    chunked = measure(head, target, states, t_values, chunk_size=4)
    single_slice = measure(head, target, states, t_values, chunk_size=10)

    _assert_report_matches(chunked, single_slice, "measure report")
