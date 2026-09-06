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
    # Both arms at both house couplings, twice (the correctness wave at
    # hidden 128/3 flat lr; the parity wave `_par`), plus the
    # fair-tuning grid: 3x3 lr x epsilon minus the centre (= the `_par`
    # cell itself) x both arms at sigma_c only = 16 `_swp` cells; plus the
    # 8x8 rung: d64 `_par` centres at both couplings + the 4-arm
    # sigma_c star per objective = 12 d64 cells; plus the flow-lr
    # fairness pair (fldb sigma_c centre + flow_head lr 1e-1/1e-2) = 2;
    # plus the budget-doubled fldb diagnostic = 1; plus the
    # standalone-flow arm = 1; plus the 16x16 rung: d256 `_par`
    # centres at both couplings x both arms = 4; plus the 20x20 rung:
    # d400 `_par` centres, same shape = 4.
    assert len(GFN_CONFIGS) == 48


def test_parity_cells_match_house_d16_sizing_and_split_lr_z():
    # The parity wave: the first-wave cells carried a ~6x parameter advantage
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


def test_compile_policy_off_at_d16_on_at_d64_and_above():
    # Archived cells never retro-flip: every d16 cell stays eager exactly
    # as it ran. The d64+ cells ship compiled from the start,
    # still gated by the GPU numerical-parity check at the launch bench
    # run at each cell's own size on the venue stack.
    for cell in GFN_CONFIGS.values():
        assert cell.compile_policy is (cell.D in (8, 16, 20))


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


def test_log_z_carries_no_weight_decay_and_reaches_the_d256_scale():
    """AdamW's decoupled decay caps any scalar at 1/weight_decay: under a
    constant-sign gradient Adam's normalised step saturates at magnitude 1
    and the decay term wd*theta balances it at theta = 1/wd -- 100 at the
    default wd = 0.01. That sits above every d64 slice log Z (44 / 54) and
    BELOW the d256 slice (~183 at sigma = 0.1, ~230 at sigma_c): the first
    16x16 TB wave (tag 20260831-gfn-d256) stalled with log Z pinned at
    100.0 +- 0.1 on all six seeds and a ~73-nat residual the policy cannot
    close. log Z is a normaliser, not a weight: no decay, and
    it must be able to reach the 256-site scale."""
    import torch
    from experiments.constrained_hard_03.run_gfn import (
        build_optimiser, build_target_and_policy)

    cfg = replace(_tiny_cell("tb"), log_z_learning_rate=0.1)
    _, policy = build_target_and_policy(cfg, "cpu")
    optimiser = build_optimiser(cfg, policy)
    log_z_group = next(g for g in optimiser.param_groups
                       if g["name"] == "log_z")
    assert log_z_group["weight_decay"] == 0.0

    # A constant unit gradient pushing log Z up for 3000 steps at lr 0.1:
    # undecayed it passes 250; under the default decay it stalls at 100.
    for _ in range(3000):
        optimiser.zero_grad()
        policy.log_z.grad = torch.full_like(policy.log_z, -1.0)
        optimiser.step()
    assert policy.log_z.item() > 250


def test_sigma_c_cells_use_exact_critical_coupling():
    # s220 must be the exact SIGMA_C = ln(1+sqrt(2))/4, never legacy 0.223
    # (sigma_c migration).
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
    # The 8x8 rung: centre = the 4x4 parity recipe at the wave-2 d64
    # budget. Sizing stays hidden 64/2/4 (104,450 params at D=8, within
    # 3.5% of the ma head's 108,256 — parity is measured params); the
    # house levers (warmup/clip/EMA/bf16 eval/in-training eval) are
    # matched field by field to H2_d64_*_w2's train/eval blocks.
    d64 = {n: c for n, c in GFN_CONFIGS.items() if c.D == 8}
    # 12 rung cells + 2 flr arms + 100k diagnostic + sfh arm.
    assert len(d64) == 16
    for name, cell in d64.items():
        assert cell.hidden_dim == 64 and cell.n_layers == 2
        # The one exception to the matched 50k budget is the DECLARED
        # budget-doubled diagnostic (its whole point is the moved budget;
        # its own twin test pins that nothing else moved).
        expected_steps = 100_000 if "_100k_" in name else 50_000
        assert cell.n_steps == expected_steps and cell.batch_size == 128
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


def test_build_optimiser_splits_flow_head_group():
    """The FL-DB analogue of the log_z split: the flow head's output
    must reach the tens-of-nats completion-entropy scale and Adam moves it
    ~lr per step, so at the shared lr the d64 centres were still climbing
    at 50k. The split must move EXACTLY the flow head's parameters, and
    asking for it on a policy without a flow head is a misconfiguration
    that must raise, not silently train nothing at the fast lr."""
    import pytest
    from experiments.constrained_hard_03.run_gfn import (
        build_optimiser, build_target_and_policy)

    cfg = replace(_tiny_cell("fldb"), flow_head_learning_rate=1e-2)
    _, policy = build_target_and_policy(cfg, "cpu")
    optimiser = build_optimiser(cfg, policy)
    groups = {g["name"]: g for g in optimiser.param_groups}
    assert set(groups) == {"network", "flow_head"}
    assert groups["flow_head"]["lr"] == 1e-2
    flow_params = {id(p) for p in policy.flow_head.parameters()}
    assert {id(p) for p in groups["flow_head"]["params"]} == flow_params
    network_params = {id(p) for p in groups["network"]["params"]}
    assert not (network_params & flow_params)
    # Every parameter is in exactly one group.
    assert (len(groups["network"]["params"]) + len(flow_params)
            == len(list(policy.parameters())))

    tb_cfg = replace(_tiny_cell("tb"), flow_head_learning_rate=1e-2)
    _, tb_policy = build_target_and_policy(tb_cfg, "cpu")
    with pytest.raises(ValueError, match="flow"):
        build_optimiser(tb_cfg, tb_policy)


def test_warmup_never_touches_the_flow_head_group():
    """Same exemption rationale as log_z: the split lr exists to unstarve
    the normaliser, and re-throttling it for the ramp would re-create a
    mild version of the failure at the start of every run."""
    from experiments.constrained_hard_03.run_gfn import (
        apply_lr_warmup, build_optimiser, build_target_and_policy)

    cfg = replace(_tiny_cell("fldb"), flow_head_learning_rate=1e-2,
                  warmup_steps=100)
    _, policy = build_target_and_policy(cfg, "cpu")
    optimiser = build_optimiser(cfg, policy)
    apply_lr_warmup(optimiser, cfg, step=0)
    lrs = {g["name"]: g["lr"] for g in optimiser.param_groups}
    assert lrs["network"] == cfg.learning_rate / 100
    assert lrs["flow_head"] == 1e-2


def test_flow_lr_cells_are_fldb_centre_twins_plus_one_lever():
    """The flr arms exist to test ONE diagnosis (the flow head is
    lr-starved), so they must be the fldb sigma_c parity centre with the
    flow-head lr as the only moved field -- any second difference would
    confound the reading."""
    from dataclasses import asdict

    centre = GFN_CONFIGS["GFN_d64_c50_s220_fldb_50k_par"]
    for flr_key, flow_lr in (("flr1e1", 1e-1), ("flr1e2", 1e-2)):
        cell = GFN_CONFIGS[f"GFN_d64_c50_s220_fldb_50k_{flr_key}"]
        assert cell.flow_head_learning_rate == flow_lr
        diff = {
            field: (a, b)
            for field, (a, b) in (
                (f, (asdict(centre)[f], asdict(cell)[f]))
                for f in asdict(centre)
            )
            if a != b
        }
        assert set(diff) == {"name", "flow_head_learning_rate"}, diff


def test_fldb_100k_diagnostic_is_the_centre_twin_plus_budget():
    """The budget-doubled arm settles slow-vs-broken and must move ONLY
    n_steps off the fldb sigma_c centre; any second lever would
    confound the reading. (The sigma ladder dilates WITH n_steps by the
    equal-share rule — that is the same field, not a second lever.)"""
    from dataclasses import asdict

    centre = GFN_CONFIGS["GFN_d64_c50_s220_fldb_50k_par"]
    cell = GFN_CONFIGS["GFN_d64_c50_s220_fldb_100k_par"]
    assert cell.n_steps == 100_000
    diff = {
        field: (a, b)
        for field, (a, b) in (
            (f, (asdict(centre)[f], asdict(cell)[f])) for f in asdict(centre)
        )
        if a != b
    }
    assert set(diff) == {"name", "n_steps"}, diff


def test_sfh_cell_is_the_fldb_centre_twin_plus_the_standalone_flow():
    """The standalone-flow arm answers ONE question (does the
    torchgfn-conventional parameterisation change FLDB's convergence) and
    must move only that field off the centre."""
    from dataclasses import asdict

    centre = GFN_CONFIGS["GFN_d64_c50_s220_fldb_50k_par"]
    cell = GFN_CONFIGS["GFN_d64_c50_s220_fldb_50k_sfh"]
    assert cell.standalone_flow_head is True
    diff = {
        field: (a, b)
        for field, (a, b) in (
            (f, (asdict(centre)[f], asdict(cell)[f])) for f in asdict(centre)
        )
        if a != b
    }
    assert set(diff) == {"name", "standalone_flow_head"}, diff


def test_d256_parity_cells_measured_params_within_anchor_band():
    """Parity is measured params per rung, and at d256 the policy must be
    re-sized: the d64 sizing (hidden 64) lands at 116,738 params, 12.6%
    under the chapter's thp2 head (133,632) and 15.1% under the ma cell
    (137,440) — outside the precedent band (+0.4% at d16, -3.5% at d64).
    hidden 68 restores it, and sits at parity with BOTH candidate anchors
    at once, so the anchor choice cannot be motivated. The flow head is
    excluded from the count, matching the d16/d64 parity audits (it is a
    declared delta on the fldb family, not part of the policy)."""
    from discrete_flow_sampler.models.raster_gfn_policy import RasterGFNPolicy

    cell = GFN_CONFIGS["GFN_d256_c50_s220_tb_100k_par"]
    policy = RasterGFNPolicy(
        D=cell.D,
        n_plus_target=128,
        hidden_dim=cell.hidden_dim,
        n_layers=cell.n_layers,
        n_heads=cell.n_heads,
        with_flow_head=False,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    assert n_params == 130_562
    for anchor in (133_632, 137_440):  # thp2 / ma stacks, measured
        assert abs(n_params - anchor) / anchor < 0.055


def test_d256_sigma_c_cells_mirror_the_house_100k_ladder():
    """The house d256 100k curriculum climbs 0.100 -> sigma_c over stages
    starting at 0/5k/10k/15k/20k/25k/30k then holds sigma_c for 70k. The
    GFN sigma_stages rule gives every entry an equal n_steps/len share, so
    20 stages of 5k (six climbing + fourteen at sigma_c) reproduce those
    start-steps and the plateau exactly."""
    house_climb = (0.100, 0.140, 0.170, 0.190, 0.205, 0.215)
    for objective in GFN_OBJECTIVES:
        cell = GFN_CONFIGS[f"GFN_d256_c50_s220_{objective}_100k_par"]
        assert cell.n_steps == 100_000
        assert len(cell.sigma_stages) == 20
        assert cell.sigma_stages[:6] == house_climb
        assert all(s == SIGMA_C for s in cell.sigma_stages[6:])


def test_d256_cells_are_d64_twins_plus_declared_levers():
    """The 16x16 cells are the d64 recipe with only the declared
    rung levers moved: lattice size, the parity re-size (hidden 68), and —
    on the sigma_c cells only — the house d256 budget (100k) with its
    dilated ladder. The in-training eval cadence is also a declared lever
    at this rung (house d256 regime 500/256 rather than the d64 wave's
    200/512, for within-rung parity with the swap-head rows the GFN cells
    join in tab:eval-hard-16x16). Anything else moving would break recipe
    parity with the 8x8 wave."""
    from dataclasses import asdict

    house_eval_cadence = {"eval_every", "n_eval_samples_training"}
    for objective in GFN_OBJECTIVES:
        for sigma_label, expected_extra in (
            ("s010", house_eval_cadence),
            ("s220", {"n_steps", "sigma_stages"} | house_eval_cadence),
        ):
            steps = "50k" if sigma_label == "s010" else "100k"
            d64 = GFN_CONFIGS[f"GFN_d64_c50_{sigma_label}_{objective}_50k_par"]
            d256 = GFN_CONFIGS[
                f"GFN_d256_c50_{sigma_label}_{objective}_{steps}_par"
            ]
            assert d256.D == 16
            assert d256.hidden_dim == 68
            assert d256.compile_policy is True
            assert d256.eval_every == 500
            assert d256.n_eval_samples_training == 256
            assert d256.eval_autocast_bf16 is True
            diff = {
                field
                for field in asdict(d64)
                if asdict(d64)[field] != asdict(d256)[field]
            }
            assert diff == {"name", "D", "hidden_dim"} | expected_extra, diff


def test_d400_parity_cells_measured_params_within_anchor_band():
    """Parity is measured params per rung, and at d400 the policy must be
    re-sized a second time: the 20x20 swap heads are larger than their
    16x16 siblings (thp2 158,848 / thp3 159,616 vs 133,632 at d256) and
    the policy's position embedding grows with the lattice, so carrying
    hidden 68 up lands at 140,354, 11.6% under both anchors. hidden 72
    (18 dims per head) gives 155,522, within 2.6% of both, so the anchor
    choice again cannot be motivated. Flow head excluded as at d16/d64/d256."""
    from discrete_flow_sampler.models.raster_gfn_policy import RasterGFNPolicy

    cell = GFN_CONFIGS["GFN_d400_c50_s220_tb_100k_par"]
    policy = RasterGFNPolicy(
        D=cell.D,
        n_plus_target=200,
        hidden_dim=cell.hidden_dim,
        n_layers=cell.n_layers,
        n_heads=cell.n_heads,
        with_flow_head=False,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    assert n_params == 155_522
    for anchor in (158_848, 159_616):  # thp2 / thp3 d400 stacks, measured
        assert abs(n_params - anchor) / anchor < 0.055


def test_d400_cells_are_d256_twins_plus_lattice_and_resize():
    """The 20x20 cells are the d256 recipe with exactly two levers
    moved: the lattice (D=20) and the parity re-size (hidden 72). Budgets,
    the sigma ladder (the house reuses the d256 ladder unrescaled at d400,
    absolute start-steps) and the house eval cadence all ride unchanged."""
    from dataclasses import asdict

    for objective in GFN_OBJECTIVES:
        for sigma_label, steps in (("s010", "50k"), ("s220", "100k")):
            d256 = GFN_CONFIGS[f"GFN_d256_c50_{sigma_label}_{objective}_{steps}_par"]
            d400 = GFN_CONFIGS[f"GFN_d400_c50_{sigma_label}_{objective}_{steps}_par"]
            assert d400.D == 20 and d400.hidden_dim == 72
            diff = {
                field for field in asdict(d256)
                if asdict(d256)[field] != asdict(d400)[field]
            }
            assert diff == {"name", "D", "hidden_dim"}, diff
