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
    # Both arms at both house couplings, twice: the s92 correctness wave
    # (hidden 128/3, flat lr) and the s93 parity wave (`_par`).
    assert len(GFN_CONFIGS) == 8


def test_parity_cells_match_house_d16_sizing_and_split_lr_z():
    # The s93 judging wave: the s92 cells carried a ~6x parameter advantage
    # over the wave-2 d16 heads (597.6k vs 79.5k-101k). Parity is measured
    # in PARAMETERS, not copied hyperparameters — hidden 64 / 2 layers puts
    # the policy at 101,378 params, within 0.4% of the masked-attention
    # head's 100,960 — plus the house batch 128. TB additionally splits
    # log_z into its own ~100x lr group: at a flat Adam lr of 1e-3 a scalar
    # moves at most ~lr/step, so log Z (init 0, exact slice value 10.81 at
    # sigma_c) arithmetically could not converge inside 10k steps —
    # measured tail slope +7e-4/step.
    parity = {n: c for n, c in GFN_CONFIGS.items() if n.endswith("_par")}
    assert len(parity) == 4
    for name, cell in parity.items():
        assert cell.hidden_dim == 64
        assert cell.n_layers == 2
        assert cell.n_heads == 4
        assert cell.batch_size == 128
        if cell.objective == "tb":
            assert cell.log_z_learning_rate == 0.1
        else:
            # log_z takes no gradient under FL-DB; the lever stays unset
            # where it cannot act.
            assert cell.log_z_learning_rate is None
    # The correctness wave is untouched (archived cells never retro-flip).
    for name, cell in GFN_CONFIGS.items():
        if not name.endswith("_par"):
            assert cell.hidden_dim == 128
            assert cell.log_z_learning_rate is None


def test_build_optimiser_splits_log_z_group():
    from experiments.constrained_hard_03.run_gfn import (
        build_optimiser, build_target_and_policy)

    cfg = _tiny_cell("tb")
    _, policy = build_target_and_policy(cfg, "cpu")

    flat = build_optimiser(replace(cfg, log_z_learning_rate=None), policy)
    assert len(flat.param_groups) == 1
    assert flat.param_groups[0]["lr"] == cfg.learning_rate

    split = build_optimiser(replace(cfg, log_z_learning_rate=0.1), policy)
    assert len(split.param_groups) == 2
    lrs = sorted(g["lr"] for g in split.param_groups)
    assert lrs == sorted([cfg.learning_rate, 0.1])
    log_z_group = next(g for g in split.param_groups if g["lr"] == 0.1)
    assert len(log_z_group["params"]) == 1
    assert log_z_group["params"][0] is policy.log_z
    # Every parameter is in exactly one group.
    n_split = sum(len(g["params"]) for g in split.param_groups)
    assert n_split == len(list(policy.parameters()))


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
