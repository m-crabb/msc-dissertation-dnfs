"""Per-stage best-checkpoint instrument (boundary-shock arm, 2026-08-19).

- With ``train_cfg.stage_best_checkpoints = True`` and a sigma curriculum,
  training saves ``checkpoints/best_stage<k>.pt`` for each curriculum stage
  k, chosen by the trailing median (window 3) of the periodic in-training
  eval ESS within that stage — a windowed statistic, because the archived
  16x16 record shows single-step ESS peaks are noise excursions and a
  best-by-single-step rule would checkpoint noise.
- ``checkpoints/stage_best.json`` records, per stage, the step and the
  median-ESS value the saved checkpoint was chosen at, and the step must
  lie inside that stage's boundaries.
- The instrument is pure IO: training dynamics, the CSV schema and the
  final.pt path are untouched, and with the flag off (the default and every
  archived config) no new files appear.
"""

import json
from pathlib import Path

import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _tiny_head():
    return DoublyHollowSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _tiny_cfgs(**train_overrides):
    train_kwargs = dict(
        n_steps=8,
        batch_size=8,
        outer_batch_size=8,
        inner_steps_per_outer=2,
        lr=1e-3,
        seed=0,
        replay_buffer_cycles=1,
        grad_clip_max_norm=500.0,
        warmup_steps=0,
    )
    train_kwargs.update(train_overrides)
    train_cfg = _Cfg(**train_kwargs)
    ctmc_cfg = _Cfg(n_euler_steps=8)
    eval_cfg = _Cfg(eval_every=2, n_eval_samples=16)
    return train_cfg, ctmc_cfg, eval_cfg


# The curriculum normaliser reads stages by attribute (start_step / sigma /
# optional lr), matching the config dataclasses — plain dicts are rejected.
_TWO_STAGE_CURRICULUM = [
    _Cfg(start_step=0, sigma=0.10),
    _Cfg(start_step=4, sigma=0.14),
]


def test_stage_best_checkpoints_written_per_stage_with_metadata(tmp_path):
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs(stage_best_checkpoints=True)
    train_swap(
        _tiny_head(),
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        Path(tmp_path),
        use_wandb=False,
        estimator_mode="control_variate",
        sigma_curriculum=_TWO_STAGE_CURRICULUM,
    )
    ckpt_dir = Path(tmp_path) / "checkpoints"
    assert (ckpt_dir / "best_stage0.pt").exists()
    assert (ckpt_dir / "best_stage1.pt").exists()
    meta = json.loads((ckpt_dir / "stage_best.json").read_text())
    stage0, stage1 = meta["0"], meta["1"]
    assert 0 <= stage0["step"] < 4
    assert 4 <= stage1["step"] < 8
    for record in (stage0, stage1):
        assert record["ess_trailing_median"] > 0.0
    # The saved tensors load as a plain state dict for the head.
    state = torch.load(ckpt_dir / "best_stage0.pt", weights_only=True)
    _tiny_head().load_state_dict(state)


def test_stage_best_checkpoints_off_by_default_writes_nothing(tmp_path):
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
    train_swap(
        _tiny_head(),
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        Path(tmp_path),
        use_wandb=False,
        estimator_mode="control_variate",
        sigma_curriculum=_TWO_STAGE_CURRICULUM,
    )
    ckpt_dir = Path(tmp_path) / "checkpoints"
    assert not list(ckpt_dir.glob("best_stage*.pt"))
    assert not (ckpt_dir / "stage_best.json").exists()
