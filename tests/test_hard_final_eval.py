"""End-of-run eval plumbing for the hard-constraint cells: `final_eval` must
stream the draw in `eval_sample_chunk` slices (the unchunked 5000-sample eval
OOM'd all three d=64 sigma_c seeds) and `eval_only` must recover
the eval/ artefacts from a completed run dir's checkpoint."""

import json
import shutil
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
        model=ModelCfg(kind="letf", hidden_dim=16, n_layers=2, n_heads=2, vocab_size=2),
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


def test_final_eval_matching_canonical_cell_writes_plain_eval_dir(tmp_path):
    """A cell that declares the matching step canonical (the 16x16 rung's
    CTMCCfg.use_matching_step=True) gets its matching-step artefacts in plain
    eval/ — the dir train() short-circuits on and the frozen-eval comparison
    reads — with no multi_event argument: the default resolves to the config's
    step. The one-event contrast on such a cell lands in eval_one_event/ so
    neither clobbers the other."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4)
    cfg = replace(cfg, ctmc=replace(cfg.ctmc, use_matching_step=True))
    target, head = build_target_and_head(cfg, "cpu")

    metrics = final_eval(head, target, cfg, Path(tmp_path))

    assert (tmp_path / "eval" / "metrics.json").exists()
    assert not (tmp_path / "eval_multi_event").exists()
    assert metrics["multi_event"] is True
    samples = torch.load(tmp_path / "eval" / "samples.pt")
    assert ((samples == 1).float().mean(dim=1) == 0.5).all()

    contrast_metrics = final_eval(head, target, cfg, Path(tmp_path), multi_event=False)
    assert (tmp_path / "eval_one_event" / "metrics.json").exists()
    assert contrast_metrics["multi_event"] is False


def test_final_eval_unchunked_when_chunk_is_none(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_eval_samples=6, eval_sample_chunk=None)
    target, head = build_target_and_head(cfg, "cpu")
    metrics = final_eval(head, target, cfg, Path(tmp_path))
    assert metrics["n_eval_samples"] == 6


def test_eval_only_recovers_eval_artefacts_from_run_dir(tmp_path, monkeypatch):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    monkeypatch.setattr("experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg})
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
    key in config.json; eval_only treats the absence as "ran with the
    then-default" instead of rejecting the dir as drifted, which would break the
    recovery path for every historical run each time the dataclass grows a
    knob."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    monkeypatch.setattr("experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg})
    run_dir = tmp_path / "legacy"
    (run_dir / "checkpoints").mkdir(parents=True)
    saved = asdict(cfg)
    del saved["anchor_chunk_size"]  # default None
    del saved["compile_head"]  # default False — fill must use the field default
    # Nested sub-config additions must backfill too: the real d=16 run dirs
    # predate model.use_sdpa_readout and eval.eval_autocast_bf16, and a
    # top-level-only fill locked every one of them out. Both sit at their
    # defaults in _tiny_cfg; deleting a key the cfg sets non-defaultly
    # (eval_sample_chunk=4) would be real drift and must still be rejected.
    del saved["model"]["use_sdpa_readout"]  # default False
    del saved["eval"]["eval_autocast_bf16"]  # default False
    (run_dir / "config.json").write_text(json.dumps(saved))
    _, head = build_target_and_head(cfg, "cpu")
    torch.save(head.state_dict(), run_dir / "checkpoints" / "final.pt")

    metrics = eval_only(run_dir)

    assert metrics["n_eval_samples"] == 10


@pytest.mark.parametrize("suffix", ["", "_ema", "_stage6"])
def test_final_eval_smc_writes_own_dir_and_smc_metrics(tmp_path, suffix):
    """SMC eval must land beside — never over — the plain-IS artefacts, and
    at τ=1.0 (fires whenever weights aren't exactly uniform) the resampling
    machinery is actually exercised at toy scale."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4)
    target, head = build_target_and_head(cfg, "cpu")

    metrics = final_eval_smc(
        head, target, cfg, Path(tmp_path), tau=1.0, eval_dir_suffix=suffix
    )

    assert not (tmp_path / "eval").exists()  # plain-IS dir untouched
    eval_dir = tmp_path / f"eval_smc_tau1{suffix}"
    samples = torch.load(eval_dir / "samples.pt")
    pooled_log_weights = torch.load(eval_dir / "log_weights.pt")
    assert samples.shape == (10, 16)
    assert pooled_log_weights.shape == (10,)
    assert torch.isfinite(pooled_log_weights).all()
    assert ((samples == 1).float().mean(dim=1) == 0.5).all()  # on-manifold
    assert metrics["smc_tau"] == 1.0
    assert metrics["n_resample_events"] > 0
    assert len(metrics["chunk_stats"]) == 3  # slices 4/4/2
    assert 0.0 < metrics["ess_fraction"] <= 1.0
    assert 1 <= metrics["n_unique_samples"] <= 10
    assert metrics == json.loads((eval_dir / "metrics.json").read_text())


def test_final_eval_smc_tau_zero_is_bit_exact_plain_is(tmp_path):
    """τ=0 never fires, consumes no extra RNG, and banks nothing, so the pooled
    SMC weights equal the plain-IS weights bit-for-bit under the same seed —
    pinning the whole eval path, not just the sampler."""
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
    monkeypatch.setattr("experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg})
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


def test_eval_only_replicate_seed_writes_replicate_dir_and_preserves_eval(
    tmp_path, monkeypatch
):
    """Probe replicate draws (an independent sampling run with a fresh eval seed
    off the one converged checkpoint) land in eval_replicate_s<seed>/ and never
    create or touch the frozen eval/ dir the headline numbers came from."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    monkeypatch.setattr("experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg})
    run_dir = tmp_path / "tiny_hard_eval_seed7_rep"
    (run_dir / "checkpoints").mkdir(parents=True)
    seeded = replace(cfg, train=replace(cfg.train, seed=7))
    (run_dir / "config.json").write_text(json.dumps(asdict(seeded)))
    _, head = build_target_and_head(cfg, "cpu")
    torch.save(head.state_dict(), run_dir / "checkpoints" / "final.pt")

    metrics = eval_only(run_dir, replicate_seed=101)

    replicate_dir = run_dir / "eval_replicate_s101"
    assert (replicate_dir / "metrics.json").exists()
    assert not (run_dir / "eval").exists()
    assert metrics["replicate_seed"] == 101
    samples = torch.load(replicate_dir / "samples.pt")
    assert ((samples == 1).float().mean(dim=1) == 0.5).all()


def test_eval_only_replicate_seeds_differ_and_reproduce(tmp_path, monkeypatch):
    """Replicates must be independent draws (different seeds give different
    weights) yet reproducible (same seed twice gives bit-identical weights) —
    the two properties the probe's MSE-across-replicates estimate rests on."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    monkeypatch.setattr("experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg})
    run_dir = tmp_path / "tiny_hard_eval_seed7_reps"
    (run_dir / "checkpoints").mkdir(parents=True)
    seeded = replace(cfg, train=replace(cfg.train, seed=7))
    (run_dir / "config.json").write_text(json.dumps(asdict(seeded)))
    _, head = build_target_and_head(cfg, "cpu")
    torch.save(head.state_dict(), run_dir / "checkpoints" / "final.pt")

    eval_only(run_dir, replicate_seed=101)
    eval_only(run_dir, replicate_seed=102)
    first = torch.load(run_dir / "eval_replicate_s101" / "log_weights.pt")
    second = torch.load(run_dir / "eval_replicate_s102" / "log_weights.pt")
    assert not torch.equal(first, second)

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    shutil.copytree(run_dir / "checkpoints", replay_dir / "checkpoints")
    shutil.copy2(run_dir / "config.json", replay_dir / "config.json")
    eval_only(replay_dir, replicate_seed=101)
    rerun = torch.load(replay_dir / "eval_replicate_s101" / "log_weights.pt")
    assert torch.equal(rerun, first)


def test_eval_only_rejects_config_drift(tmp_path, monkeypatch):
    """A run dir whose recorded config no longer matches CONFIGS must fail
    loudly rather than silently eval under the wrong settings."""
    cfg = _tiny_cfg()
    monkeypatch.setattr("experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg})
    run_dir = tmp_path / "drifted"
    (run_dir / "checkpoints").mkdir(parents=True)
    drifted = replace(cfg, ctmc=CTMCCfg(n_euler_steps=99))
    (run_dir / "config.json").write_text(json.dumps(asdict(drifted)))

    with pytest.raises(ValueError, match="does not match CONFIGS"):
        eval_only(run_dir)


def _run_dir_with_both_checkpoints(tmp_path, cfg, seed=7):
    """A completed run dir carrying both the raw and the EMA final weights, as
    every training run writes them."""
    run_dir = tmp_path / "tiny_hard_eval_seed7_test"
    (run_dir / "checkpoints").mkdir(parents=True)
    seeded = replace(cfg, train=replace(cfg.train, seed=seed))
    (run_dir / "config.json").write_text(json.dumps(asdict(seeded)))
    _, raw_head = build_target_and_head(cfg, "cpu")
    torch.save(raw_head.state_dict(), run_dir / "checkpoints" / "final.pt")
    # A different parameter vector, so a test can tell which file was loaded.
    _, ema_head = build_target_and_head(cfg, "cpu")
    with torch.no_grad():
        for parameter in ema_head.parameters():
            parameter.add_(1.0)
    torch.save(ema_head.state_dict(), run_dir / "checkpoints" / "final_ema.pt")
    return run_dir


def test_eval_only_ema_reads_the_ema_checkpoint_into_a_suffixed_dir(
    tmp_path, monkeypatch
):
    """`use_ema` must load `final_ema.pt` — not `final.pt` — and land in
    `eval_ema_ne<k>/`. The EMA weights are the primary read for every d=256
    result (raw eval ESS at sigma_c is top-weight dominated and does not
    resolve); loading the file by name removes the hand-staged-copy workaround,
    which failed silently by producing a plausible number from the wrong
    weights.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    monkeypatch.setattr("experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg})
    run_dir = _run_dir_with_both_checkpoints(tmp_path, cfg)

    eval_only(run_dir, n_euler_override=4, use_ema=True)
    eval_only(run_dir, n_euler_override=4)

    ema_dir = run_dir / "eval_ema_ne4"
    raw_dir = run_dir / "eval_ne4"
    assert (ema_dir / "metrics.json").exists()
    assert (raw_dir / "metrics.json").exists()
    # Different weights must give different weights-under-the-target, which
    # is the only evidence that the intended checkpoint was the one loaded.
    assert not torch.equal(
        torch.load(ema_dir / "log_weights.pt"),
        torch.load(raw_dir / "log_weights.pt"),
    )


def test_eval_only_ema_without_a_grid_override_guards_only_frozen_numbers(
    tmp_path, monkeypatch
):
    """`use_ema` alone writes `eval_ema/`, the directory the training run owns.
    When a frozen EMA eval is already there, re-drawing it is refused rather
    than clobbering a number the tables read. When the trainer died between the
    raw and EMA evals (final_ema.pt on disk, eval_ema/ never written — the d256
    camort case), the same call is the recovery path and lands the canonical
    eval_ema/."""
    cfg = _tiny_cfg()
    monkeypatch.setattr("experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg})
    run_dir = _run_dir_with_both_checkpoints(tmp_path, cfg)

    # Recovery: no frozen EMA eval exists, so the draw is allowed and lands
    # in the canonical eval_ema/.
    eval_only(run_dir, use_ema=True)
    assert (run_dir / "eval_ema" / "metrics.json").exists()

    # Frozen: now that eval_ema/ holds a recorded number, the same call is
    # refused.
    with pytest.raises(ValueError, match="frozen"):
        eval_only(run_dir, use_ema=True)
