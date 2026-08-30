"""Wiring gates for the GFN comparator experiment harness.

The library core is gated by test_gfn_comparator.py; these tests pin the
experiment-level contract instead: the config registry is sound, and
train_gfn writes the same artefact set run.py does (config.json,
training_log.csv, checkpoints, eval/ with samples + log-weights + metrics)
so GFN rows are drop-in comparable for the hard-chapter table scripts.
"""

import json

import torch

from experiments.constrained_hard_03.gfn_configs import (
    GFN_CONFIGS,
    GFN_OBJECTIVES,
)
from experiments.constrained_hard_03.run_gfn import train_gfn
from discrete_flow_sampler.targets.ising import SIGMA_C

from dataclasses import replace


def test_registry_keys_match_cell_names_and_objectives():
    for name, cell in GFN_CONFIGS.items():
        assert name == cell.name
        assert cell.objective in GFN_OBJECTIVES
    # Both arms present at both house couplings on the 4x4 rung.
    assert len(GFN_CONFIGS) == 4


def test_sigma_c_cells_use_exact_critical_coupling():
    # s220 must be the exact SIGMA_C = ln(1+sqrt(2))/4, never legacy 0.223
    # (sigma_c migration, s58).
    for name, cell in GFN_CONFIGS.items():
        if "_s220_" in name:
            assert cell.sigma == SIGMA_C
        if "_s010_" in name:
            assert cell.sigma == 0.10


def test_fldb_cells_carry_the_flow_head():
    for cell in GFN_CONFIGS.values():
        assert cell.with_flow_head == (cell.objective == "fldb")


def _tiny_cell(objective):
    # 2x2 lattice, seconds-scale: exercises the full train->eval->artefact
    # path, not convergence (the 2x2 convergence gate lives in
    # test_gfn_comparator.py).
    base = GFN_CONFIGS[f"GFN_d16_c50_s010_{objective}_10k"]
    return replace(
        base,
        name=f"{base.name}_wiretest",
        D=2,
        hidden_dim=16,
        n_layers=1,
        n_heads=2,
        n_steps=30,
        batch_size=32,
        n_eval_samples=64,
        eval_sample_chunk=32,
        log_every=10,
        checkpoint_every=20,
    )


def test_train_gfn_writes_house_artefact_set(tmp_path):
    for objective in GFN_OBJECTIVES:
        cfg = _tiny_cell(objective)
        run_dir = train_gfn(
            cfg, seed=42, output_dir=tmp_path, use_wandb=False, tag="wire"
        )
        assert run_dir == tmp_path / f"{cfg.name}_seed42_wire"
        assert (run_dir / "config.json").exists()
        assert (run_dir / "checkpoints" / "final.pt").exists()
        assert (run_dir / "checkpoints" / "resume.pt").exists()

        log_rows = (run_dir / "training_log.csv").read_text().strip().splitlines()
        assert log_rows[0] == "step,loss,log_z,ess_fraction_train,sigma"
        assert len(log_rows) > 3

        metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
        assert metrics["n_eval_samples"] == 64
        assert 0.0 < metrics["ess_fraction"] <= 1.0
        assert metrics["head_kind"] == f"gfn_{objective}"
        # Composition observables present -> table scripts can ingest the row.
        assert "composition_mean" in json.dumps(metrics) or any(
            "composition" in key for key in metrics
        )

        samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True)
        log_weights = torch.load(
            run_dir / "eval" / "log_weights.pt", weights_only=True
        )
        assert samples.shape == (64, 4)
        assert log_weights.shape == (64,)
        # Feasibility by construction: every eval draw on the slice.
        assert torch.all(((samples + 1) * 0.5).sum(dim=-1) == 2)


def test_completed_run_short_circuits(tmp_path):
    cfg = _tiny_cell("tb")
    run_dir = train_gfn(
        cfg, seed=42, output_dir=tmp_path, use_wandb=False, tag="wire"
    )
    metrics_before = (run_dir / "eval" / "metrics.json").read_text()
    rerun_dir = train_gfn(
        cfg, seed=42, output_dir=tmp_path, use_wandb=False, tag="wire"
    )
    assert rerun_dir == run_dir
    assert (run_dir / "eval" / "metrics.json").read_text() == metrics_before
