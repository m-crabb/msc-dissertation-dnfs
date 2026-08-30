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
    # Both arms at both house couplings, twice (s92 correctness wave at
    # hidden 128/3 flat lr; s93 parity wave `_par`), plus the s93
    # fair-tuning grid: 3x3 lr x epsilon minus the centre (= the `_par`
    # cell itself) x both arms at sigma_c only = 16 `_swp` cells; plus the
    # s94 8x8 rung: d64 `_par` centres at both couplings + the 4-arm
    # sigma_c star per objective = 12 d64 cells.
    assert len(GFN_CONFIGS) == 36


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
    parity = {n: c for n, c in GFN_CONFIGS.items()
              if n.endswith("_par") and c.D == 4}
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
        if name.endswith("_10k"):
            assert cell.hidden_dim == 128
            assert cell.log_z_learning_rate is None


def test_sweep_cells_are_par_twins_plus_declared_lr_epsilon():
    # Fair-tuning grid: each `_swp` cell must be its sigma_c `_par` arm
    # with ONLY name, learning_rate and epsilon changed — same sizing,
    # same lr_Z (the sweep axes are the NETWORK lr and the behaviour mix;
    # lr_Z stays at the frozen 0.1), same sigma, same budget. A drifted
    # field here would make the whole grid unreadable as a sweep.
    from dataclasses import fields

    sweep = {n: c for n, c in GFN_CONFIGS.items()
             if n.endswith("_swp") and c.D == 4}
    assert len(sweep) == 16
    for name, cell in sweep.items():
        parent = GFN_CONFIGS[f"GFN_d16_c50_s220_{cell.objective}_10k_par"]
        for field in fields(cell):
            if field.name in ("name", "learning_rate", "epsilon"):
                continue
            assert getattr(cell, field.name) == getattr(parent, field.name), (
                f"{name}.{field.name} drifted from its _par parent"
            )
        assert (cell.learning_rate, cell.epsilon) != (
            parent.learning_rate, parent.epsilon)


def test_compile_policy_off_at_d16_on_at_d64():
    # Archived cells never retro-flip: every d16 cell stays eager exactly
    # as it ran. The d64 cells ship compiled from launch (s94 decision),
    # still gated by the GPU numerical-parity check at the launch bench.
    for cell in GFN_CONFIGS.values():
        assert cell.compile_policy is (cell.D == 8)


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
        assert log_rows[0] == (
            "step,loss,log_z,ess_fraction_train,sigma,"
            "grad_norm,mean_log_q,log_z_is_batch,ess_frozen"
        )
        assert len(log_rows) > 3
        # The three always-on diagnostics are finite from step 0; the
        # frozen-eval column is nan when eval_every is off (these cells).
        first = dict(zip(log_rows[0].split(","), log_rows[1].split(",")))
        assert float(first["grad_norm"]) > 0
        assert float(first["mean_log_q"]) < 0  # a log-probability
        assert first["ess_frozen"] == "nan"

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


def test_d64_cells_carry_the_house_recipe_levers():
    # The s94 8x8 rung: centre = the 4x4 parity recipe at the wave-2 d64
    # budget. Sizing stays hidden 64/2/4 (104,450 params at D=8, within
    # 3.5% of the ma head's 108,256 — parity is measured params); the
    # house levers (warmup/clip/EMA/bf16 eval/in-training eval) are
    # matched field by field to H2_d64_*_w2's train/eval blocks.
    d64 = {n: c for n, c in GFN_CONFIGS.items() if c.D == 8}
    assert len(d64) == 12
    for name, cell in d64.items():
        assert cell.hidden_dim == 64 and cell.n_layers == 2
        assert cell.n_steps == 50_000 and cell.batch_size == 128
        assert cell.warmup_steps == 500
        assert cell.grad_clip_max_norm == 500.0
        assert cell.ema_decay == 0.9999
        assert cell.eval_autocast_bf16 is True
        assert cell.eval_every == 200 and cell.n_eval_samples_training == 512
        if "_s220_" in name:
            # House ladder sigmas with the final coupling repeated to give
            # it the wave-2 recipe's 20k plateau under equal step shares.
            assert len(cell.sigma_stages) == 10
            assert cell.sigma_stages[:6] == (
                0.100, 0.140, 0.170, 0.190, 0.205, 0.215)
            assert all(s == SIGMA_C for s in cell.sigma_stages[6:])
        else:
            assert cell.sigma_stages == ()  # s010 trains flat, like w2
    # The star: sigma_c only, and each arm is its centre with ONLY name,
    # learning_rate and epsilon changed (same twin rule as the d16 grid).
    from dataclasses import fields

    star = {n: c for n, c in d64.items() if n.endswith("_swp")}
    assert len(star) == 8
    for name, cell in star.items():
        assert "_s220_" in name
        parent = GFN_CONFIGS[f"GFN_d64_c50_s220_{cell.objective}_50k_par"]
        for field in fields(cell):
            if field.name in ("name", "learning_rate", "epsilon"):
                continue
            assert getattr(cell, field.name) == getattr(parent, field.name), (
                f"{name}.{field.name} drifted from its d64 _par parent"
            )


def test_warmup_ramps_network_group_but_never_log_z(tmp_path):
    from experiments.constrained_hard_03.run_gfn import (
        apply_lr_warmup, build_optimiser, build_target_and_policy)

    cfg = replace(_tiny_cell("tb"), log_z_learning_rate=0.1, warmup_steps=100)
    _, policy = build_target_and_policy(cfg, "cpu")
    optimiser = build_optimiser(cfg, policy)

    apply_lr_warmup(optimiser, cfg, step=0)
    lrs = {g["name"]: g["lr"] for g in optimiser.param_groups}
    assert lrs["network"] == cfg.learning_rate / 100
    assert lrs["log_z"] == 0.1  # exempt: the split lr exists to unstarve it

    apply_lr_warmup(optimiser, cfg, step=100)
    assert {g["name"]: g["lr"] for g in optimiser.param_groups} == {
        "network": cfg.learning_rate, "log_z": 0.1}

    # warmup off (every archived d16 cell): a no-op at any step.
    flat_cfg = replace(cfg, warmup_steps=0)
    apply_lr_warmup(optimiser, flat_cfg, step=0)
    assert optimiser.param_groups[0]["lr"] == cfg.learning_rate


def test_ema_dual_eval_and_resume_state(tmp_path):
    # ema_decay > 0 arms the house dual-eval instrument: raw weights into
    # eval/, shadow weights into eval_ema/ + checkpoints/final_ema.pt, and
    # the shadow (with its update counter) rides resume.pt so preemption
    # cannot re-create the init-contamination failure.
    cfg = replace(_tiny_cell("tb"), ema_decay=0.9999, eval_every=10)
    run_dir = train_gfn(
        cfg, seed=42, output_dir=tmp_path, use_wandb=False, tag="wire"
    )
    assert (run_dir / "checkpoints" / "final_ema.pt").exists()
    resume = torch.load(
        run_dir / "checkpoints" / "resume.pt", weights_only=True
    )
    assert resume["ema"] is not None and resume["ema"]["updates"] > 0
    for subdir in ("eval", "eval_ema"):
        metrics = json.loads((run_dir / subdir / "metrics.json").read_text())
        assert 0.0 < metrics["ess_fraction"] <= 1.0
        assert metrics["head_kind"] == "gfn_tb"
    # eval_every armed the frozen-ESS column: finite at logged eval steps.
    log_rows = (run_dir / "training_log.csv").read_text().strip().splitlines()
    header = log_rows[0].split(",")
    frozen = [dict(zip(header, r.split(",")))["ess_frozen"]
              for r in log_rows[1:]]
    assert any(value != "nan" for value in frozen)
    # EMA off: no shadow artefacts, resume carries an explicit None.
    flat_dir = train_gfn(
        replace(_tiny_cell("fldb")), seed=42,
        output_dir=tmp_path, use_wandb=False, tag="wire",
    )
    assert not (flat_dir / "eval_ema").exists()
    assert not (flat_dir / "checkpoints" / "final_ema.pt").exists()
    resume_flat = torch.load(
        flat_dir / "checkpoints" / "resume.pt", weights_only=True
    )
    assert resume_flat["ema"] is None


def test_bf16_eval_autocast_completes_with_finite_metrics(tmp_path):
    # CPU bf16 autocast exercises the same code path the GPU takes; the
    # gate is that eval under autocast still lands on the manifold (the
    # assert inside final_eval_gfn) and produces finite weights.
    cfg = replace(_tiny_cell("fldb"), eval_autocast_bf16=True)
    run_dir = train_gfn(
        cfg, seed=42, output_dir=tmp_path, use_wandb=False, tag="wire"
    )
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    assert 0.0 < metrics["ess_fraction"] <= 1.0
    log_weights = torch.load(
        run_dir / "eval" / "log_weights.pt", weights_only=True
    )
    assert torch.isfinite(log_weights).all()


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
