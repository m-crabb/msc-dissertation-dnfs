"""The update benchmark must not hide sampling inside its measured work."""

from types import SimpleNamespace

import pytest
import torch
from experiments.constrained_hard_03 import profile_swap, run_gfn


@pytest.mark.parametrize(
    "mode,expected_draws",
    [
        ("gfn_update", 1),
        ("gfn_train_step", 2),
    ],
)
def test_gfn_update_prepares_one_batch_outside_timing(
    monkeypatch, mode, expected_draws
):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    draws = []
    losses = []
    inside_timer = False

    def sample(batch, epsilon):
        draws.append(inside_timer)
        return torch.ones(batch, 64), None

    policy = SimpleNamespace(sample=sample)
    monkeypatch.setattr(
        run_gfn, "build_target_and_policy", lambda *args: (None, policy)
    )
    monkeypatch.setattr(
        run_gfn,
        "build_optimiser",
        lambda *args: torch.optim.AdamW([parameter], lr=0.01),
    )

    def loss(cfg, policy, target, spins):
        assert inside_timer
        assert not spins.requires_grad
        losses.append(spins)
        return parameter.square() * spins.mean(), None

    def timed(runner, repeats, device, **kwargs):
        nonlocal inside_timer
        inside_timer = True
        for _ in range(2):
            runner()
        inside_timer = False
        return [0.01, 0.01]

    monkeypatch.setattr(run_gfn, "_loss_and_train_diagnostics", loss)
    monkeypatch.setattr(profile_swap, "_timed", timed)
    monkeypatch.setattr(profile_swap, "_report", lambda *args: None)
    args = SimpleNamespace(
        d=64,
        batch=2,
        gfn_objective="tb",
        mode=mode,
        compile=False,
        repeats=2,
        profile=False,
        warmup=3,
    )
    profile_swap._run_gfn_bench(args, torch.device("cpu"))
    assert len(draws) == expected_draws
    assert draws == ([False] if mode == "gfn_update" else [True, True])
    assert parameter.item() < 1.0
    if mode == "gfn_update":
        assert losses[0] is losses[1]


def test_timing_discards_warmup_and_resets_gpu_peak(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: calls.append("reset")
    )
    times = profile_swap._timed(
        lambda: calls.append("update"), 2, torch.device("cuda"), warmup=3
    )
    assert calls == ["update"] * 3 + ["reset"] + ["update"] * 2
    assert len(times) == 2


def test_benchmark_parser_keeps_gfn_header_and_archived_headerless_rows():
    from experiments.constrained_hard_03.analysis.parse_bench_log import parse

    rows = parse(
        [
            "mode=gfn_update d=64 batch=128 compile=True",
            "gfn_update_tb_d64_B128 median 0.0040s min 0.0030s peak_mem 0.10 GB",
            "gfn_rollout_tb_d64_B128 median 0.0700s min 0.0600s peak_mem 0.06 GB",
        ]
    )
    assert len(rows) == 2
    assert rows[0]["mode"] == "gfn_update" and rows[0]["d"] == "64"
    assert rows[0]["timings"]["gfn_update_tb_d64_B128"]["median_s"] == 0.004
    assert rows[1]["mode"] == "gfn_rollout_tb_d64_B128"
