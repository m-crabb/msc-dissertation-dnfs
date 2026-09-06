"""What correct looks like for EMA dual-eval on the FLIP trainer, before it.

Ported from the swap trainer for the soft chapter: the soft
chassis's one missing instrument. ema_decay > 0 arms a warmup-corrected
parameter shadow (discrete_flow_sampler.ema.ExponentialMovingAverage)
updated after every optimiser step and saved as checkpoints/final_ema.pt;
the run entry then evaluates BOTH weight sets (eval/ + eval_ema/).

Contracts frozen here, each guarding a specific failure:

1. PASSIVE OBSERVER. Arming the shadow must not perturb training: the raw
   final.pt is bit-identical with and without ema_decay. The hard chapter's
   comparisons rely on "twins differing only in ema_decay train
   bit-identically"; a shadow that consumed RNG or touched grads would
   break every such comparison silently.
2. ARMED IFF WRITTEN. final_ema.pt exists exactly when armed, and differs
   from final.pt after a run whose weights moved — a shadow equal to the
   raw weights is a no-op wiring bug, and a file written when disarmed
   would change every archived cell's artefact inventory.
3. WRAPPER COVERAGE. With the exact-field channel on, the shadow tracks
   the WRAPPER's parameters: final_ema.pt carries gain_constant/gain_slope.
   The EMA must be built over the top-level module — built over an inner
   head it would silently exclude the gains, the parameters the soft
   rescue turns on.
4. RESUME. Preemption-resume carries shadow AND update counter:
   interrupted+resumed final_ema.pt is bit-identical to uninterrupted.
   Re-initialising the shadow at resume-point weights would re-create the
   init-contamination failure through the back door; resetting only the
   counter would restart the warmup schedule mid-run (ema.py docstring).
"""

from types import SimpleNamespace

import pytest
import torch

from discrete_flow_sampler.constraints.exact_field_channel import ExactFieldFlipModel
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.training import train
from discrete_flow_sampler.targets.ising import IsingTarget

N_STEPS = 12
INNER_PER_OUTER = 2


class _Preempted(Exception):
    """Stands in for the container dying with no chance to clean up."""


def _die_after(n_checkpoints):
    calls = {"n": 0}

    def hook():
        calls["n"] += 1
        if calls["n"] >= n_checkpoints:
            raise _Preempted

    return hook


def _cfgs(*, seed=0, resume_every_outer=1):
    train_cfg = SimpleNamespace(
        n_steps=N_STEPS,
        inner_steps_per_outer=INNER_PER_OUTER,
        batch_size=8,
        outer_batch_size=4,
        replay_buffer_cycles=2,
        lr=1e-3,
        seed=seed,
        grad_clip_max_norm=500.0,
        warmup_steps=0,
        resume_every_outer=resume_every_outer,
    )
    ctmc_cfg = SimpleNamespace(n_euler_steps=3)
    eval_cfg = SimpleNamespace(eval_every=4, n_eval_samples=8)
    return train_cfg, ctmc_cfg, eval_cfg


def _soft_target():
    return IsingTarget(
        D=2,
        sigma=0.1,
        target_composition=0.5,
        composition_penalty_strength=5.0,
    )


def _model(target, *, init_seed, channel=False):
    torch.manual_seed(init_seed)
    model = LeTFRateMatrix(
        d=target.d, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2
    )
    return ExactFieldFlipModel(model, target) if channel else model


def _run(run_dir, *, init_seed=0, ema_decay=0.0, channel=False, on_checkpoint=None):
    target = _soft_target()
    train_cfg, ctmc_cfg, eval_cfg = _cfgs()
    train(
        model=_model(target, init_seed=init_seed, channel=channel),
        target=target,
        train_cfg=train_cfg,
        ctmc_cfg=ctmc_cfg,
        eval_cfg=eval_cfg,
        output_dir=run_dir,
        use_wandb=False,
        estimator_mode="control_variate",
        ema_decay=ema_decay,
        on_checkpoint=on_checkpoint,
    )


def _state(path):
    return torch.load(path, weights_only=True)


def test_ema_is_a_passive_observer(tmp_path):
    _run(tmp_path / "raw", init_seed=0)
    _run(tmp_path / "armed", init_seed=0, ema_decay=0.999)
    raw = _state(tmp_path / "raw" / "checkpoints" / "final.pt")
    armed = _state(tmp_path / "armed" / "checkpoints" / "final.pt")
    assert raw.keys() == armed.keys()
    for key in raw:
        assert torch.equal(raw[key], armed[key]), key


def test_final_ema_written_iff_armed_and_differs_from_raw(tmp_path):
    _run(tmp_path / "raw", init_seed=0)
    _run(tmp_path / "armed", init_seed=0, ema_decay=0.999)
    assert not (tmp_path / "raw" / "checkpoints" / "final_ema.pt").exists()
    ema_path = tmp_path / "armed" / "checkpoints" / "final_ema.pt"
    assert ema_path.exists()
    raw = _state(tmp_path / "armed" / "checkpoints" / "final.pt")
    shadow = _state(ema_path)
    assert raw.keys() == shadow.keys()
    assert any(not torch.equal(raw[key], shadow[key]) for key in raw)


def test_shadow_covers_channel_gains(tmp_path):
    _run(tmp_path, init_seed=0, ema_decay=0.999, channel=True)
    shadow = _state(tmp_path / "checkpoints" / "final_ema.pt")
    raw = _state(tmp_path / "checkpoints" / "final.pt")
    assert shadow.keys() == raw.keys()
    assert "gain_constant" in shadow and "gain_slope" in shadow


def _stage_cfg(name, ema_decay):
    from experiments.dnfs_baseline_01.configs import (
        CTMCCfg,
        EvalCfg,
        IsingCfg,
        ModelCfg,
        StageCfg,
        TrainCfg,
    )

    return StageCfg(
        name=name,
        ising=IsingCfg(
            D=2, sigma=0.1, target_composition=0.5, composition_penalty_strength=5.0
        ),
        train=TrainCfg(
            n_steps=12,
            inner_steps_per_outer=2,
            batch_size=8,
            outer_batch_size=4,
            replay_buffer_cycles=2,
            lr=1e-3,
            seed=0,
            warmup_steps=0,
        ),
        ctmc=CTMCCfg(n_euler_steps=3),
        eval=EvalCfg(eval_every=4, n_eval_samples=8),
        model=ModelCfg(kind="let", hidden_dim=8, n_layers=1, n_heads=2),
        estimator="control_variate",
        ema_decay=ema_decay,
    )


def test_run_entry_dual_eval_and_short_circuit(tmp_path):
    """run.train writes eval/ AND eval_ema/ when armed (same metric schema,
    so the house-table ingestion reads both interchangeably), writes NO
    eval_ema/ when disarmed (archived artefact inventories unchanged), and
    the finished-run short-circuit requires BOTH dirs — a fixed-tag
    relaunch that finds eval/ but not eval_ema/ must fill the gap, not
    skip (the GFN wave's short-circuit lesson, same trap)."""
    import json
    import shutil

    from experiments.dnfs_baseline_01.run import train as run_train

    armed = _stage_cfg("ema_smoke", ema_decay=0.9)
    run_train(armed, seed=0, output_dir=tmp_path, use_wandb=False, tag="t0")
    run_dir = tmp_path / "ema_smoke_seed0_t0"
    raw_metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    ema_metrics = json.loads((run_dir / "eval_ema" / "metrics.json").read_text())
    assert set(raw_metrics) == set(ema_metrics)

    shutil.rmtree(run_dir / "eval_ema")
    run_train(armed, seed=0, output_dir=tmp_path, use_wandb=False, tag="t0")
    assert (run_dir / "eval_ema" / "metrics.json").exists()

    disarmed = _stage_cfg("ema_off_smoke", ema_decay=0.0)
    run_train(disarmed, seed=0, output_dir=tmp_path, use_wandb=False, tag="t0")
    assert not (tmp_path / "ema_off_smoke_seed0_t0" / "eval_ema").exists()


def test_resume_carries_shadow_and_counter(tmp_path):
    _run(tmp_path / "ref", init_seed=0, ema_decay=0.999)
    with pytest.raises(_Preempted):
        _run(
            tmp_path / "int", init_seed=0, ema_decay=0.999, on_checkpoint=_die_after(3)
        )
    assert not (tmp_path / "int" / "checkpoints" / "final_ema.pt").exists()
    resume_state = torch.load(
        tmp_path / "int" / "checkpoints" / "resume.pt", weights_only=True
    )
    assert resume_state.get("ema") is not None
    _run(tmp_path / "int", init_seed=999, ema_decay=0.999)
    reference = _state(tmp_path / "ref" / "checkpoints" / "final_ema.pt")
    resumed = _state(tmp_path / "int" / "checkpoints" / "final_ema.pt")
    assert reference.keys() == resumed.keys()
    for key in reference:
        assert torch.equal(reference[key], resumed[key]), key
