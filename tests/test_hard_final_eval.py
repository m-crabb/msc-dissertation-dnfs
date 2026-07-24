"""End-of-run eval plumbing for the hard-constraint cells: `final_eval` must
stream the draw in `eval_sample_chunk` slices (the unchunked 5000-sample eval
OOM'd all three d=64 sigma_c seeds, 2026-07-06) and `eval_only` must recover
the eval/ artefacts from a completed run dir's checkpoint."""
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from experiments.constrained_hard_03.configs import HardStageCfg
from experiments.constrained_hard_03.run import (
    build_target_and_head,
    eval_only,
    final_eval,
    final_eval_smc,
)
from experiments.dnfs_baseline_01.configs import (
    CTMCCfg,
    EvalCfg,
    IsingCfg,
    ModelCfg,
    TrainCfg,
)


def _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4):
    return HardStageCfg(
        name="tiny_hard_eval",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0, target_composition=0.5),
        train=TrainCfg(n_steps=2, batch_size=4, inner_steps_per_outer=2, seed=0),
        ctmc=CTMCCfg(n_euler_steps=8),
        eval=EvalCfg(
            eval_every=2,
            n_eval_samples=n_eval_samples,
            eval_sample_chunk=eval_sample_chunk,
        ),
        model=ModelCfg(kind="letf", hidden_dim=16, n_layers=2, n_heads=2,
                       vocab_size=2),
        estimator="control_variate",
        head_kind="mask_one",
        wandb_project="test",
    )


def test_final_eval_chunked_covers_exact_count_and_writes_artefacts(tmp_path):
    """Chunk 4 into 10 samples -> slices of 4/4/2; no drop or duplication,
    every sample stays on the fixed-composition manifold."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4)
    target, head = build_target_and_head(cfg, "cpu")

    metrics = final_eval(head, target, cfg, Path(tmp_path))

    samples = torch.load(tmp_path / "eval" / "samples.pt")
    log_weights = torch.load(tmp_path / "eval" / "log_weights.pt")
    assert samples.shape == (10, 16)
    assert log_weights.shape == (10,)
    assert torch.isfinite(log_weights).all()
    # Swap moves conserve composition exactly: every sample has 8 up-spins.
    assert ((samples == 1).float().mean(dim=1) == 0.5).all()
    assert metrics["n_eval_samples"] == 10
    assert 0.0 < metrics["ess"] <= 10.0
    assert metrics == json.loads((tmp_path / "eval" / "metrics.json").read_text())


def test_final_eval_multi_event_writes_own_dir_and_stays_on_manifold(tmp_path):
    """The --compare-multi-event probe must never clobber the one-event
    baseline: matching-step artefacts land in eval_multi_event/, and the
    simultaneous swaps still conserve composition exactly."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4)
    target, head = build_target_and_head(cfg, "cpu")

    metrics = final_eval(head, target, cfg, Path(tmp_path), multi_event=True)

    assert not (tmp_path / "eval").exists()
    samples = torch.load(tmp_path / "eval_multi_event" / "samples.pt")
    assert ((samples == 1).float().mean(dim=1) == 0.5).all()
    assert metrics["multi_event"] is True
    assert metrics["n_eval_samples"] == 10


def test_final_eval_unchunked_when_chunk_is_none(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_eval_samples=6, eval_sample_chunk=None)
    target, head = build_target_and_head(cfg, "cpu")
    metrics = final_eval(head, target, cfg, Path(tmp_path))
    assert metrics["n_eval_samples"] == 6


def test_eval_only_recovers_eval_artefacts_from_run_dir(tmp_path, monkeypatch):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    monkeypatch.setattr(
        "experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg}
    )
    run_dir = tmp_path / "tiny_hard_eval_seed7_test"
    (run_dir / "checkpoints").mkdir(parents=True)
    seeded = replace(cfg, train=replace(cfg.train, seed=7))
    (run_dir / "config.json").write_text(json.dumps(asdict(seeded)))
    _, head = build_target_and_head(cfg, "cpu")
    torch.save(head.state_dict(), run_dir / "checkpoints" / "final.pt")

    metrics = eval_only(run_dir)

    assert (run_dir / "eval" / "metrics.json").exists()
    assert metrics["n_eval_samples"] == 10


def test_eval_only_accepts_legacy_config_missing_defaulted_fields(
    tmp_path, monkeypatch
):
    """Run dirs written before a defaulted HardStageCfg field existed lack its
    key in config.json; eval_only must treat the absence as "ran with the
    then-default" instead of rejecting the dir as drifted (the strict guard
    would otherwise break the recovery path for every historical run each
    time the dataclass grows a knob)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    monkeypatch.setattr(
        "experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg}
    )
    run_dir = tmp_path / "legacy"
    (run_dir / "checkpoints").mkdir(parents=True)
    saved = asdict(cfg)
    del saved["anchor_chunk_size"]  # default None
    del saved["compile_head"]  # default False — fill must use the field default
    (run_dir / "config.json").write_text(json.dumps(saved))
    _, head = build_target_and_head(cfg, "cpu")
    torch.save(head.state_dict(), run_dir / "checkpoints" / "final.pt")

    metrics = eval_only(run_dir)

    assert metrics["n_eval_samples"] == 10


def test_final_eval_smc_writes_own_dir_and_smc_metrics(tmp_path):
    """SMC eval must land beside — never over — the plain-IS artefacts, and
    at τ=1.0 (fires whenever weights aren't exactly uniform) the resampling
    machinery is actually exercised at toy scale."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4)
    target, head = build_target_and_head(cfg, "cpu")

    metrics = final_eval_smc(head, target, cfg, Path(tmp_path), tau=1.0)

    assert not (tmp_path / "eval").exists()          # plain-IS dir untouched
    eval_dir = tmp_path / "eval_smc_tau1"
    samples = torch.load(eval_dir / "samples.pt")
    pooled_log_weights = torch.load(eval_dir / "log_weights.pt")
    assert samples.shape == (10, 16)
    assert pooled_log_weights.shape == (10,)
    assert torch.isfinite(pooled_log_weights).all()
    assert ((samples == 1).float().mean(dim=1) == 0.5).all()   # on-manifold
    assert metrics["smc_tau"] == 1.0
    assert metrics["n_resample_events"] > 0
    assert len(metrics["chunk_stats"]) == 3                    # slices 4/4/2
    assert 0.0 < metrics["ess_fraction"] <= 1.0
    assert 1 <= metrics["n_unique_samples"] <= 10
    assert metrics == json.loads((eval_dir / "metrics.json").read_text())


def test_final_eval_smc_tau_zero_is_bit_exact_plain_is(tmp_path):
    """τ=0 never fires, consumes no extra RNG, and banks nothing — so the
    pooled SMC weights must equal the plain-IS weights bit-for-bit under
    the same seed. This pins the whole eval path, not just the sampler."""
    cfg = _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4)
    target, head = build_target_and_head(cfg, "cpu")

    torch.manual_seed(3)
    plain_metrics = final_eval(head, target, cfg, Path(tmp_path))
    torch.manual_seed(3)
    smc_metrics = final_eval_smc(head, target, cfg, Path(tmp_path), tau=0.0)

    plain_log_weights = torch.load(tmp_path / "eval" / "log_weights.pt")
    pooled_log_weights = torch.load(tmp_path / "eval_smc_tau0" / "log_weights.pt")
    assert torch.equal(pooled_log_weights, plain_log_weights)
    assert smc_metrics["n_resample_events"] == 0
    assert smc_metrics["ess"] == plain_metrics["ess"]


def test_eval_only_smc_tau_runs_smc_variant_only(tmp_path, monkeypatch):
    """Backfill path: --smc-tau on a completed run dir writes the SMC
    artefacts without re-drawing the expensive plain-IS eval."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    monkeypatch.setattr(
        "experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg}
    )
    run_dir = tmp_path / "tiny_hard_eval_seed7_smc"
    (run_dir / "checkpoints").mkdir(parents=True)
    seeded = replace(cfg, train=replace(cfg.train, seed=7))
    (run_dir / "config.json").write_text(json.dumps(asdict(seeded)))
    _, head = build_target_and_head(cfg, "cpu")
    torch.save(head.state_dict(), run_dir / "checkpoints" / "final.pt")

    metrics = eval_only(run_dir, smc_tau=1.0)

    assert (run_dir / "eval_smc_tau1" / "metrics.json").exists()
    assert not (run_dir / "eval").exists()
    assert metrics["smc_tau"] == 1.0


def test_eval_only_rejects_config_drift(tmp_path, monkeypatch):
    """A run dir whose recorded config no longer matches CONFIGS must fail
    loudly rather than silently eval under the wrong settings."""
    cfg = _tiny_cfg()
    monkeypatch.setattr(
        "experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg}
    )
    run_dir = tmp_path / "drifted"
    (run_dir / "checkpoints").mkdir(parents=True)
    drifted = replace(cfg, ctmc=CTMCCfg(n_euler_steps=99))
    (run_dir / "config.json").write_text(json.dumps(asdict(drifted)))

    with pytest.raises(ValueError, match="does not match CONFIGS"):
        eval_only(run_dir)
