"""Protect checkpoint identity and saved run provenance before doing work."""

import json
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest
import torch
from experiments.constrained_hard_03 import run
from experiments.constrained_hard_03.configs import CONFIGS


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    cfg = next(iter(CONFIGS.values()))
    cfg = replace(cfg, train=replace(cfg.train, seed=7))
    run_dir = tmp_path / f"{cfg.name}_seed7_test"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg)))
    monkeypatch.setattr(run, "CONFIGS", {cfg.name: cfg})
    return cfg, run_dir


@pytest.mark.parametrize(
    "selection,suffix", [({"stage_best": 6}, "_stage6"), ({"use_ema": True}, "_ema")]
)
def test_smc_preserves_checkpoint_identity(saved_run, monkeypatch, selection, suffix):
    cfg, run_dir = saved_run
    head = Mock()
    monkeypatch.setattr(run, "build_target_and_head", lambda *_: (Mock(), head))
    monkeypatch.setattr(torch, "load", lambda path, **_: {"checkpoint": path.name})
    draw = Mock(return_value={})
    monkeypatch.setattr(run, "final_eval_smc", draw)
    run.eval_only(run_dir, smc_tau=0.5, **selection)
    assert draw.call_args.kwargs["eval_dir_suffix"] == suffix
    expected = (
        "best_stage6.pt" if selection.get("stage_best") is not None else "final_ema.pt"
    )
    head.load_state_dict.assert_called_once_with({"checkpoint": expected})


@pytest.mark.parametrize("smc_tau,suffix", [(None, "_ne128"), (0.5, "_stage6")])
def test_existing_eval_artifact_is_refused_before_sampling(
    saved_run, monkeypatch, smc_tau, suffix
):
    cfg, run_dir = saved_run
    name = "eval" if smc_tau is None else "eval_smc_tau0.5"
    dest = run_dir / f"{name}{suffix}"
    dest.mkdir()
    # A partial draw is evidence too, even without metrics.json.
    original = b"archived samples"
    (dest / "samples.pt").write_bytes(original)
    draw = Mock(side_effect=AssertionError("must refuse before drawing"))
    monkeypatch.setattr(run, "_chunked_eval_draw", draw)
    with pytest.raises(FileExistsError, match="existing eval artefacts"):
        if smc_tau is None:
            run.final_eval(None, None, cfg, run_dir, eval_dir_suffix=suffix)
        else:
            run.final_eval_smc(
                None, None, cfg, run_dir, tau=smc_tau, eval_dir_suffix=suffix
            )
    assert (dest / "samples.pt").read_bytes() == original
    draw.assert_not_called()


def test_resume_ignores_missing_initialisation_checkpoint(saved_run, monkeypatch):
    cfg, run_dir = saved_run
    (run_dir / "checkpoints" / "resume.pt").write_bytes(b"resume handled by trainer")
    provenance = run_dir / "init_from.txt"
    provenance.write_text("original parent\n")
    monkeypatch.setattr(run, "write_host_metadata", lambda *_: None)
    monkeypatch.setattr(run, "build_target_and_head", lambda *_: (Mock(), Mock()))
    trainer = Mock()
    monkeypatch.setattr(run, "train_swap", trainer)
    monkeypatch.setattr(run, "final_eval", lambda *_: {})
    run.train(
        cfg,
        seed=7,
        output_dir=run_dir.parent,
        use_wandb=False,
        tag="test",
        init_from=run_dir / "missing.pt",
    )
    trainer.assert_called_once()
    assert provenance.read_text() == "original parent\n"


def test_resume_rejects_config_drift_before_writing_metadata(saved_run, monkeypatch):
    cfg, run_dir = saved_run
    (run_dir / "checkpoints" / "resume.pt").write_bytes(b"resume")
    original = (run_dir / "config.json").read_bytes()
    metadata = Mock(side_effect=AssertionError("must validate first"))
    monkeypatch.setattr(run, "write_host_metadata", metadata)
    changed = replace(cfg, ctmc=replace(cfg.ctmc, n_euler_steps=999))
    with pytest.raises(ValueError, match="does not match requested config"):
        run.train(
            changed, seed=7, output_dir=run_dir.parent, use_wandb=False, tag="test"
        )
    metadata.assert_not_called()
    assert (run_dir / "config.json").read_bytes() == original


def test_resume_requires_its_original_saved_config(saved_run, monkeypatch):
    cfg, run_dir = saved_run
    (run_dir / "checkpoints" / "resume.pt").write_bytes(b"resume")
    (run_dir / "config.json").unlink()
    metadata = Mock(side_effect=AssertionError("must validate first"))
    monkeypatch.setattr(run, "write_host_metadata", metadata)
    with pytest.raises(ValueError, match="resume.*config.json"):
        run.train(cfg, seed=7, output_dir=run_dir.parent, use_wandb=False, tag="test")
    assert not (run_dir / "config.json").exists()


def test_smc_ema_can_preserve_an_existing_plain_ema_eval(saved_run, monkeypatch):
    _, run_dir = saved_run
    (run_dir / "eval_ema").mkdir()
    (run_dir / "eval_ema" / "metrics.json").write_text("frozen")
    monkeypatch.setattr(run, "build_target_and_head", lambda *_: (Mock(), Mock()))
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {})
    draw = Mock(return_value={})
    monkeypatch.setattr(run, "final_eval_smc", draw)
    run.eval_only(run_dir, smc_tau=0.5, use_ema=True)
    assert draw.call_args.kwargs["eval_dir_suffix"] == "_ema"
    assert (run_dir / "eval_ema" / "metrics.json").read_text() == "frozen"
