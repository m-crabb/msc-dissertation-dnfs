"""Chunked head forward in the scout's `measure`: the unchunked call fed all
n_grid x batch trajectory states through the mask_one head at once (d anchor
copies each), which OOM'd the d=64 checkpoint scout on a 40 GB A100
(2026-07-06). Chunking the batch dim is bit-inert: the head is per-sample
independent (attention runs over sites within a sample)."""
import torch
from experiments.constrained_hard_03.scout_euler_budget import _build_head, measure

from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def test_measure_chunked_matches_single_slice():
    """Chunk 4 over 10 states -> slices of 4/4/2; identical report to one slice."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(
        D=4, sigma=0.22305, target_composition=0.5, device="cpu"
    )
    head = _build_head("mask_one", target.d, "cpu")
    states = target.sample_base(10, device="cpu")
    t_values = torch.linspace(0.0, 1.0, 10)

    chunked = measure(head, target, states, t_values, chunk_size=4)
    single_slice = measure(head, target, states, t_values, chunk_size=10)

    assert chunked == single_slice
