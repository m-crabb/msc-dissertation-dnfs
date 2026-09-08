"""Soft-constraint configs for composition-controlled Ising sampling.

Shares the baseline schema dataclasses and defines its own CONFIGS dict. Cells set
`target_composition` and `composition_penalty_strength` on `IsingCfg`; `IsingTarget`
subtracts the VCSGC-style penalty lambda * d * (c(x) - c_target)^2 from `log_prob`.

Cell-name format: `S<alphabet>_d<dim>_c<c_target_x100>_l<lambda>`.
"""

from dataclasses import replace

from experiments.dnfs_baseline_01.configs import (
    CompositionCfg,
    CompositionCurriculumStageCfg,
    CTMCCfg,
    CurriculumCfg,
    CurriculumStageCfg,
    EvalCfg,
    IsingCfg,
    LambdaCurriculumCfg,
    LambdaCurriculumStageCfg,
    ModelCfg,
    StageCfg,
    TrainCfg,
)

from discrete_flow_sampler.targets.ising import SIGMA_C

CONFIGS: dict[str, StageCfg] = {
    "S2_d4_c03_l50_letf": StageCfg(
        name="S2_d4_c03_l50_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.3,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    "S2_d4_c05_l50_letf": StageCfg(
        name="S2_d4_c05_l50_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    # λ sweep at the c=0.5 report operating point: clones
    # of the l50 cell varying only the penalty strength, so the four-seed
    # λ comparison is controlled. λ=50 is the existing cell above.
    "S2_d4_c05_l5_letf": StageCfg(
        name="S2_d4_c05_l5_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=5.0,
        ),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    "S2_d4_c05_l10_letf": StageCfg(
        name="S2_d4_c05_l10_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=10.0,
        ),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    "S2_d4_c05_l100_letf": StageCfg(
        name="S2_d4_c05_l100_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=100.0,
        ),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    # Amortised twin of S2_d4_c05_l50_letf: model conditioned on c, c drawn per
    # outer cycle on a widening window (half-width 0.05 -> 0.15 -> 0.30 at 2k/4k).
    # Levers: condition_on_composition, composition. λ=50 held fixed (no anneal)
    # because the archived D=4 λ=50 cell trained 4/4, so the comparator's recipe holds.
    "S2_d4_camort_l50_letf": StageCfg(
        name="S2_d4_camort_l50_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=2_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=4_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Narrow-window twin of the cell above: final half-width 0.15 instead of 0.30.
    # Separates coverage from intrinsic attenuation (wide run: slope 0.39 vs the
    # exact target's 0.984); the sweep's 0.30 and 0.80 rows are now extrapolation.
    "S2_d4_camort_w15_l50_letf": StageCfg(
        name="S2_d4_camort_w15_l50_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=2_000, half_width=0.10),
                CompositionCurriculumStageCfg(start_step=4_000, half_width=0.15),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Budget twin of S2_d4_camort_l50_letf: 50k steps instead of 10k, nothing else.
    # Tests whether the slope-0.39 attenuation survives a 5x budget; only stated at
    # d=16, where the exact slope (0.984) is enumerable.
    "S2_d4_camort_50k_l50_letf": StageCfg(
        name="S2_d4_camort_50k_l50_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        # Same widening shape, stretched so each width holds the same run fraction.
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # The budget twin plus a 10->25->50 lambda anneal on the window boundaries
    # (10k/20k), nothing else. The fixed-lambda twin collapsed seed 42 into one Z2
    # mode (ESS ~0); this tests whether the anneal restores survival.
    "S2_d4_camort_50k_l50_letf_anneal": StageCfg(
        name="S2_d4_camort_50k_l50_letf_anneal",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Offset anneal: the same lambda ramp, finished by 5k, before the window first
    # widens at 10k. The annealed twin collapsed 3/4 seeds at the shared boundaries
    # (median training ESS 4900 -> 11..630 at 10k), where the lambda step, replay
    # clear and window tripling land together; the fixed twin passed (3456 -> 3255).
    "S2_d4_camort_50k_l50_letf_anneal_offset": StageCfg(
        name="S2_d4_camort_50k_l50_letf_anneal_offset",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Gradient-clip probe, two strengths: the offset cell with `grad_clip_max_norm`
    # 500 -> 50 / 100. With the ramp offset, 3/4 seeds still died slowly after the
    # final widening (pre-clip norm ~30 -> 2.4e3-1.5e4, clip firing on 55-100% of
    # steps); `clip_grad_norm_` rescales, so a saturated step is ~17x a healthy one.
    "S2_d4_camort_50k_l50_letf_anneal_offset_clip50": StageCfg(
        name="S2_d4_camort_50k_l50_letf_anneal_offset_clip50",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            replay_buffer_cycles=8,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    "S2_d4_camort_50k_l50_letf_anneal_offset_clip100": StageCfg(
        name="S2_d4_camort_50k_l50_letf_anneal_offset_clip100",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            replay_buffer_cycles=8,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=100.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Null control for the amortisation machinery: conditioning on, draw window of
    # zero width, so every cycle draws c = 0.5 and the target is the
    # S2_d4_c05_l50_letf specialist. Pass: reproduces the specialist (ESS ~0.74,
    # seeds 42-45); not bit-identical, since amortised runs use a per-state c_t.
    "S2_d4_cnull_l50_letf": StageCfg(
        name="S2_d4_cnull_l50_letf",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        composition=CompositionCfg(centre=0.5, half_width=0.0),
        wandb_project="dnfs-constraints",
    ),
    # Null control at the final recipe (50k, clip 50, offset anneal): the 10k/clip500
    # cnull pair measured a 0.155 ESS-fraction machinery cost (0.801 -> 0.646), but
    # the delivered amortised family sits at 0.754 at c = 0.5, so that 0.155 is
    # recipe-confounded. Only the conditioning path differs from the `_c05_` twin.
    "S2_d4_cnull_50k_l50_letf_anneal_offset_clip50": StageCfg(
        name="S2_d4_cnull_50k_l50_letf_anneal_offset_clip50",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            replay_buffer_cycles=8,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(centre=0.5, half_width=0.0),
        wandb_project="dnfs-constraints",
    ),
    # Matched specialist ceiling for the null above: conditioning off, no draw,
    # every other field identical. The archived 10k/clip500 ceiling (0.801) would
    # rebuild the recipe confound against a 50k/clip50 null.
    "S2_d4_c05_50k_l50_letf_anneal_offset_clip50": StageCfg(
        name="S2_d4_c05_50k_l50_letf_anneal_offset_clip50",
        ising=IsingCfg(
            D=4,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            replay_buffer_cycles=8,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    "S2_d10_c03_l50_letf_ne128": StageCfg(
        name="S2_d10_c03_l50_letf_ne128",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.3,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    "S2_d10_c03_l10_letf_ne128": StageCfg(
        name="S2_d10_c03_l10_letf_ne128",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.3,
            composition_penalty_strength=10.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    # D=10 c=0.5 subcritical headline cell: the c=0.3 ne128 stability stack (warmup
    # 2000, lambda=50) with n_euler 128 -> 64 to match the baseline D=10 grid
    # (DNFS T=64). c=0.3 is dropped from the report; this is the D=10 soft witness.
    "S2_d10_c05_l50_letf_ne64": StageCfg(
        name="S2_d10_c05_l50_letf_ne64",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    # λ sweep at the c=0.5 report operating point: clones
    # of the l50 witness varying only the penalty strength, so the four-seed
    # λ comparison is controlled. λ=50 is the existing witness above.
    "S2_d10_c05_l5_letf_ne64": StageCfg(
        name="S2_d10_c05_l5_letf_ne64",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=5.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    "S2_d10_c05_l10_letf_ne64": StageCfg(
        name="S2_d10_c05_l10_letf_ne64",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=10.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    "S2_d10_c05_l100_letf_ne64": StageCfg(
        name="S2_d10_c05_l100_letf_ne64",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=100.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    # Recipe ladder rung 1: the l50 ne64 witness plus a 10->25->50 lambda anneal
    # (0/10k/20k). The lambda sweep trained 4/4 at lambda<=10 (ESS ~0.99) but 1/4 at
    # lambda=50 from scratch. Replay clears at stage boundaries (training.py).
    "S2_d10_c05_l50_letf_ne64_anneal": StageCfg(
        name="S2_d10_c05_l50_letf_ne64_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    # F(c) off-centre windows: anneal-rung clones varying target_composition only.
    # c=0.65 (typical) and c=0.80 (stress) run first; survival = the ESS 0.30 floor.
    "S2_d10_c065_l50_letf_ne64_anneal": StageCfg(
        name="S2_d10_c065_l50_letf_ne64_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.65,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    # F(c) sweep windows: ne64 anneal clones of the c=0.5 rung, target_composition
    # only. With Z_2 reflection {0.30, 0.50, 0.55, 0.60, 0.65} covers 0.30 to 0.70;
    # the tail beyond ~0.65 is left to the hard sampler.
    "S2_d10_c030_l50_letf_ne64_anneal": StageCfg(
        name="S2_d10_c030_l50_letf_ne64_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.30,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    "S2_d10_c055_l50_letf_ne64_anneal": StageCfg(
        name="S2_d10_c055_l50_letf_ne64_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.55,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    "S2_d10_c060_l50_letf_ne64_anneal": StageCfg(
        name="S2_d10_c060_l50_letf_ne64_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.60,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    "S2_d10_c080_l50_letf_ne64_anneal": StageCfg(
        name="S2_d10_c080_l50_letf_ne64_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.80,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    # Fallback rung for the c=0.80 stress window: ne64 -> ne128 only. At ne64 1 of 4
    # seeds cleared the ESS 0.30 floor; single-seed diagnostic.
    "S2_d10_c080_l50_letf_ne128_anneal": StageCfg(
        name="S2_d10_c080_l50_letf_ne128_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.80,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    # Recipe ladder fallback rung: clone of the l50 ne64 witness
    # with a finer Euler grid only, testing whether ne128 rescues d10 seed
    # survival at the tight operating point (the c=0.3 ne128 batch went 4/4).
    "S2_d10_c05_l50_letf_ne128": StageCfg(
        name="S2_d10_c05_l50_letf_ne128",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    # Matched-base twin of the c=0.80 ne128 anneal window: per-site Bernoulli(0.80)
    # base, base_composition the only lever. Measured ESS/N 0.937 on the production
    # path (the first eval's 0.171 was an x0-draw eval bug, since fixed) vs the
    # uniform twin's 0.419.
    "S2_d10_c080_l50_letf_ne128_matched_anneal": StageCfg(
        name="S2_d10_c080_l50_letf_ne128_matched_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.80,
            composition_penalty_strength=50.0,
            base_composition=0.80,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    # Matched base + fixed lambda=50 (no anneal). Redrawn on the production path:
    # 0.483, above the 0.30 floor, so the anneal is not load-bearing for survival
    # from a matched start but still buys ~1.9x (0.937 vs 0.483).
    "S2_d10_c080_l50_letf_ne128_matched_fixed50": StageCfg(
        name="S2_d10_c080_l50_letf_ne128_matched_fixed50",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.80,
            composition_penalty_strength=50.0,
            base_composition=0.80,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    # n_euler consistency control: the c=0.5 anneal witness at
    # ne128, paired against the existing ne64 anneal (same seeds) to check
    # whether n_euler shifts logZ/F at centred compositions before mixing
    # grids across the F(c) curve.
    "S2_d10_c05_l50_letf_ne128_anneal": StageCfg(
        name="S2_d10_c05_l50_letf_ne128_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        wandb_project="dnfs-constraints",
    ),
    # ---------------------------------------------------------------
    # Amortised cells: one model conditioned on the target composition, measured
    # against the six per-composition specialists above. Both clone the surviving
    # recipe (λ 10→25→50, ne128, warmup 2000, clip 500): the anneal took seed
    # survival 1/4 -> 4/4 and ne128 took the c=0.5 ESS fraction 0.699 -> 0.918.
    # ---------------------------------------------------------------
    # Continuous c on a widening window that mirrors the λ schedule step for
    # step; c ≈ 0.5 is the easy end (base and target compositions agree).
    "S2_d10_camort_l50_letf_ne128_anneal": StageCfg(
        name="S2_d10_camort_l50_letf_ne128_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # The D=10 amortised recipe with both D=4 fixes: the lambda ramp finishes at 5k
    # (before the first widening at 10k) and grad_clip_max_norm 500 -> 50 (a
    # saturated rescaled step is ~2x a healthy one instead of ~17x). Not changed:
    # replay_buffer_cycles stays 4 so a failure stays attributable; depth is next.
    "S2_d10_camort_l50_letf_ne128_anneal_offset_clip50": StageCfg(
        name="S2_d10_camort_l50_letf_ne128_anneal_offset_clip50",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Buffer depth, the lever named above: replay_buffer_cycles 4 -> 8. The clip50
    # cell ran away at the first widening (grad norm median ~800 -> ~76,000 at step
    # 10k, ESS ~40 -> 1.0); with 4 cycles the buffer's composition labels are stale
    # at a widening. Single seed.
    "S2_d10_camort_l50_letf_ne128_anneal_offset_clip50_cyc8": StageCfg(
        name="S2_d10_camort_l50_letf_ne128_anneal_offset_clip50_cyc8",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=8,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # StableAdamW arm: the cyc8 cell with optimiser="stable_adamw" and the raw
    # gradient clip disabled, so the per-tensor update-RMS threshold is the only
    # bound. Tests whether it clears eval ESS fraction >= 0.10 where the clipped
    # recipe read 0.0147 (job 269622). Single run, seed 42.
    "S2_d10_camort_offset_cyc8_stadamw": StageCfg(
        name="S2_d10_camort_offset_cyc8_stadamw",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=8,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=1e9,
            warmup_steps=2000,
            optimiser="stable_adamw",
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Control: amortise over only the six compositions that have
    # specialists. Held-out points between those atoms separate genuine
    # interpolation from memorising the training set.
    "S2_d10_cgrid_l50_letf_ne128_anneal": StageCfg(
        name="S2_d10_cgrid_l50_letf_ne128_anneal",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=10_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=20_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            values=(0.3, 0.5, 0.55, 0.6, 0.65, 0.8),
        ),
        wandb_project="dnfs-constraints",
    ),
    # ---------------------------------------------------------------------
    # Neighbour log-ratio saturation: four arms cloned from `..._anneal_offset_clip50`
    # (the control). `residual_lenet` caps log p̃_t(y)/p̃_t(x) at a ceiling (paper
    # App. E.1.1: 5). One flip moves c by 1/d, so the penalty λd(c−c_target)²
    # contributes ∓2λΔ to the ratio, Δ = c(x)−c_target; the ceiling binds once
    #
    #     Δ  >  ceiling / (2λ)         — independent of d
    #
    # At λ=50 that is Δ* = 0.05; the control sits at Δ = 0.078 with 26% of ratios
    # saturated (p99 9.9 vs 9.8 predicted). Two levers, two doses each: lower λ
    # (10, 25) keeps the paper's ceiling; raise the ceiling (20, 50) keeps
    # tightness. Read `log_ratio_clamp_frac` as the mediating variable in every arm.
    "S2_d10_camort_offset_clip50_lam10": StageCfg(
        name="S2_d10_camort_offset_clip50_lam10",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=10.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        # No λ ramp: the terminal λ is 10, so there is nothing to anneal to, and a
        # flat λ keeps "lower λ" from confounding with "fewer boundaries".
        lambda_curriculum=None,
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Staircase cell: flat λ=10, window capped at half-width 0.20 (covers [0.3, 0.7]
    # exactly) and widened in 0.05 steps with 8k dwell from 20k; checkpoint every
    # 2.5k. The λ=10 arm above was pristine through 20k and was knocked into a
    # permanent excursion cycle by the single 0.15→0.30 jump (var_dt_log_p̃ 11→224).
    "S2_d10_camort_offset_clip50_lam10_hw20": StageCfg(
        name="S2_d10_camort_offset_clip50_lam10_hw20",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=10.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
            warmup_steps=2000,
            checkpoint_every=2_500,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        # Flat λ, as in the arm above: the terminal λ is 10 and holding it
        # fixed keeps the staircase the only moving schedule.
        lambda_curriculum=None,
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.10),
                CompositionCurriculumStageCfg(start_step=28_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=36_000, half_width=0.20),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    "S2_d10_camort_offset_clip50_lam25": StageCfg(
        name="S2_d10_camort_offset_clip50_lam25",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=25.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        # One ramp step only, 10 -> 25 at 2k, finishing well before the first
        # widening at 10k so the boundaries stay disjoint.
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Ceiling 20: unbinds up to Δ = 0.20 at λ=50, comfortably past the 0.078
    # the control reaches, while exp(20)≈4.9e8 stays far inside float32.
    "S2_d10_camort_offset_clip50_clamp20": StageCfg(
        name="S2_d10_camort_offset_clip50_clamp20",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
            log_ratio_clamp=20.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Ceiling 50: unbinds to Δ = 0.50, i.e. never binds in the swept range — the
    # aggressive end of the bracket. exp(50)≈5e21, still finite in float32.
    "S2_d10_camort_offset_clip50_clamp50": StageCfg(
        name="S2_d10_camort_offset_clip50_clamp50",
        ising=IsingCfg(
            D=10,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
            log_ratio_clamp=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=50.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let",
            hidden_dim=128,
            n_layers=3,
            n_heads=4,
            vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        lambda_curriculum=LambdaCurriculumCfg(
            stages=(
                LambdaCurriculumStageCfg(
                    start_step=0, composition_penalty_strength=10.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5,
            half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    "S2_d8_c05_l10_letf_ne64": StageCfg(
        name="S2_d8_c05_l10_letf_ne64",
        ising=IsingCfg(
            D=8,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=10.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    "S2_d8_c05_l50_letf_ne64": StageCfg(
        name="S2_d8_c05_l50_letf_ne64",
        ising=IsingCfg(
            D=8,
            sigma=0.1,
            bias=0.0,
            target_composition=0.5,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
    "S2_d8_c03_l50_letf_ne128": StageCfg(
        name="S2_d8_c03_l50_letf_ne128",
        ising=IsingCfg(
            D=8,
            sigma=0.1,
            bias=0.0,
            target_composition=0.3,
            composition_penalty_strength=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
        wandb_project="dnfs-constraints",
    ),
}


_FC_NE128_BASE = CONFIGS["S2_d10_c05_l50_letf_ne128_anneal"]
for _c, _c_tag in ((0.30, "c030"), (0.55, "c055"), (0.60, "c060"), (0.65, "c065")):
    _window_name = f"S2_d10_{_c_tag}_l50_letf_ne128_anneal"
    CONFIGS[_window_name] = replace(
        _FC_NE128_BASE,
        name=_window_name,
        ising=replace(_FC_NE128_BASE.ising, target_composition=_c),
    )

_SMC_FLIP_SOFT_BASE = CONFIGS["S2_d10_c080_l50_letf_ne128_anneal"]
for _tau, _tau_tag in ((0.3, "smc03"), (0.6, "smc06")):
    _arm_name = f"{_SMC_FLIP_SOFT_BASE.name}_{_tau_tag}"
    CONFIGS[_arm_name] = replace(
        _SMC_FLIP_SOFT_BASE,
        name=_arm_name,
        train=replace(
            _SMC_FLIP_SOFT_BASE.train,
            rollout_resample_ess_fraction=_tau,
        ),
    )

_AMORT_SPECIALIST_BASE = CONFIGS["S2_d4_c05_50k_l50_letf_anneal_offset_clip50"]
for _c, _c_tag in ((0.30, "c03"), (0.70, "c07"), (0.80, "c08")):
    _twin_name = f"S2_d4_{_c_tag}_50k_l50_letf_anneal_offset_clip50"
    CONFIGS[_twin_name] = replace(
        _AMORT_SPECIALIST_BASE,
        name=_twin_name,
        ising=replace(_AMORT_SPECIALIST_BASE.ising, target_composition=_c),
    )

_FLAT_WINDOW_BASE = CONFIGS["S2_d4_camort_50k_l50_letf_anneal_offset_clip50"]
_flat_window_name = f"{_FLAT_WINDOW_BASE.name}_flatw30"
CONFIGS[_flat_window_name] = replace(
    _FLAT_WINDOW_BASE,
    name=_flat_window_name,
    composition=replace(
        _FLAT_WINDOW_BASE.composition,
        half_width=0.30,
        curriculum=None,
    ),
)


# Lambda-sweep exact-field-channel twins: tab:soft-lambda-sweep rerun with the
# closed-form flip channel (ModelCfg.exact_field_channel, gain zero-init so each
# twin is bit-identical to its parent at step 0). One declared change per twin,
# pinned by tests/test_exact_flip_channel.py. Prediction: the channel rescues the
# all-or-nothing 10x10 lambda=50 seeds (0.02/0.06/0.78/0.05).
LAMBDA_SWEEP_PARENTS = tuple(
    f"S2_d{side}_c05_l{lam}_letf{suffix}"
    for side, suffix in ((4, ""), (10, "_ne64"))
    for lam in (5, 10, 50, 100)
)
for _parent_name in LAMBDA_SWEEP_PARENTS:
    _parent = CONFIGS[_parent_name]
    CONFIGS[f"{_parent_name}_efc"] = replace(
        _parent,
        name=f"{_parent_name}_efc",
        model=replace(_parent.model, exact_field_channel=True),
    )


# ---------------------------------------------------------------------------
# The 8x8 house family. Production moves to d=64 (the hard chapter's record
# size, so the cross-route comparison is matched-size at both couplings);
# one specialist family serves the F(c) curve and the house table.


def soft_house_recipe(cell: StageCfg) -> StageCfg:
    """House recipe for new soft cells: four declared levers on an archived parent,
    exact_field_channel=True (10x10 lambda=50 rescue, 0.95/0.93/0.94/0.96 vs
    0.02/0.06/0.78/0.05), compile_model=True, ema_decay=0.9999, c_t_from_rollout=True.
    Optimiser-side values stay parent-matched; evals stay fp32 end to end."""
    return replace(
        cell,
        model=replace(cell.model, exact_field_channel=True, compile_model=True),
        train=replace(cell.train, c_t_from_rollout=True),
        ema_decay=0.9999,
    )


# Trained compositions, every c* lattice-representable at d=64 (16/24/32 sites).
# 0.625 and 0.75 are mirrored for free (F(c) = F(1-c) under a global spin flip).
# Tag = c_target x 1000, since x100 cannot write 0.375; 0.25 is the stress window.
SOFT_HOUSE_WINDOWS = ((0.25, "c0250"), (0.375, "c0375"), (0.50, "c0500"))
_D8_HOUSE_PARENT = CONFIGS["S2_d8_c03_l50_letf_ne128"]

for _c_target, _c_tag in SOFT_HOUSE_WINDOWS:
    for _sigma, _sigma_suffix in ((0.1, ""), (SIGMA_C, "_sc")):
        _house_name = f"S2_d8_{_c_tag}_l50_letf_ne128_house{_sigma_suffix}"
        CONFIGS[_house_name] = soft_house_recipe(
            replace(
                _D8_HOUSE_PARENT,
                name=_house_name,
                ising=replace(
                    _D8_HOUSE_PARENT.ising, sigma=_sigma, target_composition=_c_target
                ),
            )
        )

# Channel control: house recipe minus the channel (the only lever), critical
# coupling, centre composition only.
_HOUSE_SC_CENTRE = CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc"]
CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_nochan"] = replace(
    _HOUSE_SC_CENTRE,
    name="S2_d8_c0500_l50_letf_ne128_house_sc_nochan",
    model=replace(_HOUSE_SC_CENTRE.model, exact_field_channel=False),
)

# Sigma_c anneal arm: the nochan control plus the chapter's lambda schedule
# (10/25/50 at 0/10k/20k), one lever, test-pinned. Completes the parent / anneal /
# channel trio at 8x8 sigma_c for fig:penalty-variance.
_NOCHAN_SC = CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_nochan"]
CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_anneal"] = replace(
    _NOCHAN_SC,
    name="S2_d8_c0500_l50_letf_ne128_house_sc_anneal",
    lambda_curriculum=LambdaCurriculumCfg(
        stages=(
            LambdaCurriculumStageCfg(start_step=0, composition_penalty_strength=10.0),
            LambdaCurriculumStageCfg(
                start_step=10_000, composition_penalty_strength=25.0
            ),
            LambdaCurriculumStageCfg(
                start_step=20_000, composition_penalty_strength=50.0
            ),
        )
    ),
)

# Single-size completion at 8x8, one-lever twins of the house centre cells:
# nochan at sigma=0.1 completes the {coupling} x {channel} 2x2 (the sc nochan
# control trains 0/4 where the channel trains 4/4); lambda=10 and lambda=100
# twins at both couplings fill the lambda block of the 8x8 house table.
_HOUSE_CENTRE = CONFIGS["S2_d8_c0500_l50_letf_ne128_house"]
CONFIGS["S2_d8_c0500_l50_letf_ne128_house_nochan"] = replace(
    _HOUSE_CENTRE,
    name="S2_d8_c0500_l50_letf_ne128_house_nochan",
    model=replace(_HOUSE_CENTRE.model, exact_field_channel=False),
)
for _lam, _lam_tag in ((10.0, "l10"), (100.0, "l100")):
    for _sigma_suffix in ("", "_sc"):
        _lam_parent = CONFIGS[f"S2_d8_c0500_l50_letf_ne128_house{_sigma_suffix}"]
        _lam_name = f"S2_d8_c0500_{_lam_tag}_letf_ne128_house{_sigma_suffix}"
        CONFIGS[_lam_name] = replace(
            _lam_parent,
            name=_lam_name,
            ising=replace(_lam_parent.ising, composition_penalty_strength=_lam),
        )

# Matched-base twins: base_composition = c* at the off-centre windows, both
# couplings (the centre cells' Bernoulli(0.5) base is already matched). One
# declared lever vs the house twin, test-pinned. If the matched base rescues a
# window the failure was base reachability; if not, it is the target itself.
for _c_target, _c_tag in SOFT_HOUSE_WINDOWS:
    if _c_target == 0.50:
        continue
    for _sigma_suffix in ("", "_sc"):
        _house_twin = CONFIGS[f"S2_d8_{_c_tag}_l50_letf_ne128_house{_sigma_suffix}"]
        _mb_name = f"S2_d8_{_c_tag}_l50_letf_ne128_house_mb{_sigma_suffix}"
        CONFIGS[_mb_name] = replace(
            _house_twin,
            name=_mb_name,
            ising=replace(_house_twin.ising, base_composition=_c_target),
        )

# D=4 compile-parity check: the full house recipe and its eager twin; passes when
# the loss traces agree to compile tolerance (1e-5-class, never bit-parity).
_D4_GATE_PARENT = CONFIGS["S2_d4_c05_l50_letf"]
CONFIGS["S2_d4_c05_l50_letf_house_gate"] = soft_house_recipe(
    replace(_D4_GATE_PARENT, name="S2_d4_c05_l50_letf_house_gate")
)
_D4_GATE = CONFIGS["S2_d4_c05_l50_letf_house_gate"]
CONFIGS["S2_d4_c05_l50_letf_house_gate_eager"] = replace(
    _D4_GATE,
    name="S2_d4_c05_l50_letf_house_gate_eager",
    model=replace(_D4_GATE.model, compile_model=False),
)

# Amortised 4x4 family on the house recipe: the archived fixed-lambda 50k parent
# plus the four recipe levers, no anneal/offset/clip. The channel has no
# boundaries, so if this trains 4/4 the offset/clip confound family drops out.
CONFIGS["S2_d4_camort_50k_l50_letf_house"] = soft_house_recipe(
    replace(
        CONFIGS["S2_d4_camort_50k_l50_letf"], name="S2_d4_camort_50k_l50_letf_house"
    )
)

# Matched-base amortisation, mirroring hard camort: discrete spine draw from step
# 0 (no staircase, no lambda curriculum), base matched to the drawn c per cycle
# (the 8x8 mb twins showed the off-centre collapse was base reachability), channel
# kept on. Base is Bernoulli(c), not hard's slice-uniform mixture: single-flip
# dynamics leave a slice at the first flip, so the Eq. 4 path is -inf off-slice.
# Levers: base_matches_composition, condition_on_composition, composition (pinned).
_CAMORT_SPINE = CompositionCfg(centre=0.5, half_width=0.0, values=(0.25, 0.375, 0.5))
# The d8 cells draw uniform over every realisable composition in [0.25, 0.5]: 17
# values at 1/64 steps as an explicit tuple, so every draw is an integer site
# count. Motivated by the D=4 cell's sparse-draw cost (held-out 0.4375 at raw ESS
# 0.73 across a 0.125 gap vs 0.95 across 0.0625); D=4 keeps its 3-value spine.
_CAMORT_D8_DRAWS = CompositionCfg(
    centre=0.5, half_width=0.0, values=tuple(sites / 64 for sites in range(16, 33))
)
# The hard d64 sigma ladder, stage tuple copied verbatim (final stage at the
# exact sigma_c). Shared by the camort ladder twin below and the specialist
# ladder twin after the loop, so the two differ by amortisation alone.
_SOFT_SIGMA_LADDER_SC = CurriculumCfg(
    stages=(
        CurriculumStageCfg(start_step=0, sigma=0.100, lr=1e-3),
        CurriculumStageCfg(start_step=5_000, sigma=0.140, lr=1e-3),
        CurriculumStageCfg(start_step=10_000, sigma=0.170, lr=1e-3),
        CurriculumStageCfg(start_step=15_000, sigma=0.190, lr=1e-3),
        CurriculumStageCfg(start_step=20_000, sigma=0.205, lr=3e-4),
        CurriculumStageCfg(start_step=25_000, sigma=0.215, lr=3e-4),
        CurriculumStageCfg(start_step=30_000, sigma=SIGMA_C, lr=3e-4),
    )
)
for _sigma_suffix in ("", "_sc"):
    _camort_parent = CONFIGS[f"S2_d8_c0500_l50_letf_ne128_house{_sigma_suffix}"]
    _camort_name = f"S2_d8_camort_l50_letf_ne128_house{_sigma_suffix}"
    CONFIGS[_camort_name] = replace(
        _camort_parent,
        name=_camort_name,
        ising=replace(_camort_parent.ising, base_matches_composition=True),
        model=replace(_camort_parent.model, condition_on_composition=True),
        composition=_CAMORT_D8_DRAWS,
    )
    if _sigma_suffix == "_sc":
        # Sigma-ladder twin: the cold sigma_c camort cell does not train, while
        # hard's amortised sigma_c cell trains on a 7-stage ladder 0.1 -> sigma_c
        # over 30k steps. One lever (curriculum), stage tuple from the hard d64 recipe.
        _ladder_name = f"{_camort_name}_curr"
        CONFIGS[_ladder_name] = replace(
            CONFIGS[_camort_name],
            name=_ladder_name,
            curriculum=_SOFT_SIGMA_LADDER_SC,
        )
    # Draw-set ablation: the 17-value cell trains at sigma=0.1 but 0/4 at sigma_c
    # (centre ESS 0.001-0.007), while the motivating D=4 cell ran the 3-value spine
    # at sigma=0.1. One lever, the draw set back to the D=4 spine; both couplings.
    _spine3_name = f"S2_d8_camort_spine3_l50_letf_ne128_house{_sigma_suffix}"
    CONFIGS[_spine3_name] = replace(
        CONFIGS[_camort_name],
        name=_spine3_name,
        composition=_CAMORT_SPINE,
    )

# Specialist ladder twin: the camort ladder twin trains (centre EMA ESS 0.220-0.452
# on 4/4 seeds vs the cold cell's 0.001-0.010), so its yield ratio needs a
# specialist on the same ladder. One lever off the house sigma_c specialist.
_sc_specialist = CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc"]
_sc_specialist_ladder_name = "S2_d8_c0500_l50_letf_ne128_house_sc_curr"
CONFIGS[_sc_specialist_ladder_name] = replace(
    _sc_specialist,
    name=_sc_specialist_ladder_name,
    curriculum=_SOFT_SIGMA_LADDER_SC,
)

# Collapse-mechanism twins, sc only: spine3 sc collapses like the 17-value cell
# (ESS 0.001-0.010), and the disengaged channel gain is a casualty of the dead
# trunk, not the cause. spine1 keeps the machinery at a single value {0.5}
# (machinery-vs-mixture); rb1 kills replay staleness (staleness-vs-mixture).
_spine1_name = "S2_d8_camort_spine1_l50_letf_ne128_house_sc"
CONFIGS[_spine1_name] = replace(
    CONFIGS["S2_d8_camort_spine3_l50_letf_ne128_house_sc"],
    name=_spine1_name,
    composition=replace(_CAMORT_SPINE, values=(0.5,)),
)
_spine3_rb1_name = "S2_d8_camort_spine3_rb1_l50_letf_ne128_house_sc"
_spine3_sc = CONFIGS["S2_d8_camort_spine3_l50_letf_ne128_house_sc"]
CONFIGS[_spine3_rb1_name] = replace(
    _spine3_sc,
    name=_spine3_rb1_name,
    train=replace(_spine3_sc.train, replay_buffer_cycles=1),
)

# Composition-conditioned exact-field gain on the critical spine3 control: adds
#   (c-c0) * (composition_gain_constant + composition_gain_slope * t)
# to the archived global gain. Both scalars start at zero and use no RNG, so every
# shared tensor and draw stays paired to the dead seeds 42--45.
_spine3_cgain_name = "S2_d8_camort_spine3_cgain_l50_letf_ne128_house_sc"
CONFIGS[_spine3_cgain_name] = replace(
    _spine3_sc,
    name=_spine3_cgain_name,
    model=replace(
        _spine3_sc.model,
        exact_field_composition_gain=True,
    ),
)

# Paired-initialisation mechanism twins: the conditioner uses a private RNG stream,
# so the critical spine1 and c=.5 specialist share every initial tensor. Each arm
# adds only per-group pre-clip gradient telemetry (log_gradient_group_norms).
_pairgrad_specialist_name = "S2_d8_c0500_pairgrad_l50_letf_ne128_house_sc"
_pairgrad_specialist_parent = CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc"]
CONFIGS[_pairgrad_specialist_name] = replace(
    _pairgrad_specialist_parent,
    name=_pairgrad_specialist_name,
    train=replace(
        _pairgrad_specialist_parent.train,
        log_gradient_group_norms=True,
    ),
)

_pairgrad_spine1_name = "S2_d8_camort_spine1_pairgrad_l50_letf_ne128_house_sc"
_pairgrad_spine1_parent = CONFIGS["S2_d8_camort_spine1_l50_letf_ne128_house_sc"]
CONFIGS[_pairgrad_spine1_name] = replace(
    _pairgrad_spine1_parent,
    name=_pairgrad_spine1_name,
    train=replace(
        _pairgrad_spine1_parent.train,
        log_gradient_group_norms=True,
    ),
)

# D=4 check for the matched-base cells: the amortised 4x4 house cell
# with the staircase swapped for the spine draw and the
# base matched — every spine c is an integer site count at d=16 (4/6/8).
_D4_CAMORT_HOUSE = CONFIGS["S2_d4_camort_50k_l50_letf_house"]
CONFIGS["S2_d4_camort_mb_50k_l50_letf_house"] = replace(
    _D4_CAMORT_HOUSE,
    name="S2_d4_camort_mb_50k_l50_letf_house",
    ising=replace(_D4_CAMORT_HOUSE.ising, base_matches_composition=True),
    composition=_CAMORT_SPINE,
)

# The tab:amort-4x4 comparator rows: specialists and null re-run on the house
# recipe at the amortised 50k budget, so the machinery is priced without the
# retired clip50 recipe confound. Windows {0.25, 0.375, 0.50}, all integer site
# counts at d=16 (4/6/8). The obedience reference slope 0.976 was measured on the
# old request grid and must be re-derived by enumeration before being quoted.
_D4_SPECIALIST_HOUSE_BASE = soft_house_recipe(
    replace(
        CONFIGS["S2_d4_c05_l50_letf"],
        name="S2_d4_c0500_50k_l50_letf_house",
        train=replace(CONFIGS["S2_d4_c05_l50_letf"].train, n_steps=50_000),
    )
)
CONFIGS["S2_d4_c0500_50k_l50_letf_house"] = _D4_SPECIALIST_HOUSE_BASE
for _c_target, _c_tag in ((0.25, "c0250"), (0.375, "c0375")):
    _twin_name = f"S2_d4_{_c_tag}_50k_l50_letf_house"
    CONFIGS[_twin_name] = replace(
        _D4_SPECIALIST_HOUSE_BASE,
        name=_twin_name,
        ising=replace(_D4_SPECIALIST_HOUSE_BASE.ising, target_composition=_c_target),
    )

# The 4x4 house table family at the cross-chapter 10k budget (baseline and hard
# train every 4x4 cell for 10k); one lever off the _50k_ cell (n_steps), sigma_c
# by one more. tab:eval-soft-4x4 reads these at both couplings.
for _c_target, _c_tag in SOFT_HOUSE_WINDOWS:
    for _sigma, _sigma_suffix in ((0.1, ""), (SIGMA_C, "_sc")):
        _budget_parent = CONFIGS[f"S2_d4_{_c_tag}_50k_l50_letf_house"]
        _d4_10k_name = f"S2_d4_{_c_tag}_10k_l50_letf_house{_sigma_suffix}"
        CONFIGS[_d4_10k_name] = replace(
            _budget_parent,
            name=_d4_10k_name,
            train=replace(_budget_parent.train, n_steps=10_000),
            ising=replace(_budget_parent.ising, sigma=_sigma),
        )

# Conditioned row of tab:eval-soft-4x4: built from the 10k centre specialist as
# the 8x8 conditioned cell is from its specialist (matched base, conditioning
# flag, spine draw). Spine, not the d8 continuum: only k/16 is realisable at d=16.
for _sigma_suffix in ("", "_sc"):
    _d4_camort_parent = CONFIGS[f"S2_d4_c0500_10k_l50_letf_house{_sigma_suffix}"]
    _d4_camort_name = f"S2_d4_camort_10k_l50_letf_house{_sigma_suffix}"
    CONFIGS[_d4_camort_name] = replace(
        _d4_camort_parent,
        name=_d4_camort_name,
        ising=replace(_d4_camort_parent.ising, base_matches_composition=True),
        model=replace(_d4_camort_parent.model, condition_on_composition=True),
        composition=_CAMORT_SPINE,
    )

# Null control on the house recipe: conditioning path on, window width
# zero, so vs the c0500 specialist above the only differences are the
# machinery itself (model flag + the zero-width composition config).
CONFIGS["S2_d4_cnull_50k_l50_letf_house"] = soft_house_recipe(
    replace(
        CONFIGS["S2_d4_cnull_l50_letf"],
        name="S2_d4_cnull_50k_l50_letf_house",
        train=replace(CONFIGS["S2_d4_cnull_l50_letf"].train, n_steps=50_000),
    )
)


# ---------------------------------------------------------------------------
# Cu-Au alloy rungs: free-composition and penalised samplers on the MetaDNS/Damewood
# Cu-Au fcc expansion (data/ce/, from experiments/alloy_ce/export_binary_expansion.py).
# `sigma` is beta/2 = 1/(2 k_B T) in 1/eV; the curriculum cools 1200 K -> 500 K, where
# the 16-site exact composition marginal is bimodal (0.77 at x_Au = 0.5, L1_0; 0.15
# at 0.25, L1_2). The flip channel is off here and on in the `_efc` twins below.
K_B_EV = 8.617333262e-5


def cuau_sigma(temperature_K: float) -> float:
    """beta/2 in 1/eV at the given temperature (11.602 at 500 K)."""
    return 1.0 / (2.0 * K_B_EV * temperature_K)


_CUAU_TEMPERATURE_LADDER_K = (1200.0, 800.0, 600.0, 500.0)


def _cuau_curriculum(n_steps: int) -> CurriculumCfg:
    stage = n_steps // 4
    return CurriculumCfg(
        stages=tuple(
            CurriculumStageCfg(
                start_step=k * stage,
                sigma=cuau_sigma(T),
                lr=1e-3 if k < 2 else 3e-4,
            )
            for k, T in enumerate(_CUAU_TEMPERATURE_LADDER_K)
        )
    )


_CUAU16_PARENT = CONFIGS["S2_d4_cnull_l50_letf"]
_CUAU64_PARENT = CONFIGS["S2_d8_c03_l50_letf_ne128"]


def _cuau_flip_cell(
    name, *, sites: int, composition: float | None, penalty: float, n_steps: int
) -> StageCfg:
    parent = _CUAU16_PARENT if sites == 16 else _CUAU64_PARENT
    return replace(
        parent,
        name=name,
        ising=replace(
            parent.ising,
            sigma=cuau_sigma(_CUAU_TEMPERATURE_LADDER_K[0]),
            target_composition=composition,
            composition_penalty_strength=penalty,
            base_matches_composition=composition is not None,
            expansion_json=f"data/ce/cuau_fcc_{'2x2x4' if sites == 16 else '4x4x4'}.json",
        ),
        train=replace(parent.train, n_steps=n_steps),
        model=replace(
            parent.model, condition_on_composition=False, exact_field_channel=False
        ),
        composition=None,
        curriculum=_cuau_curriculum(n_steps),
        ema_decay=0.9999,
    )


for _sites, _steps in ((16, 10_000), (64, 50_000)):
    _free = f"A1_cuau{_sites}_T500_letf_{_steps // 1000}k_curr"
    CONFIGS[_free] = _cuau_flip_cell(
        _free, sites=_sites, composition=None, penalty=0.0, n_steps=_steps
    )
    for _c, _c_tag in ((0.25, "c25"), (0.5, "c50")):
        # lambda sets a composition SD of 1/sqrt(2 lambda d) (the penalty carries no
        # beta): 50 is the 8x8 house value; the 16-site lambda=50 cells never trained.
        _penalties = (50.0, 10.0) if _sites == 16 else (50.0,)
        for _penalty in _penalties:
            _soft = (
                f"S2_cuau{_sites}_{_c_tag}_l{int(_penalty)}_T500_letf_"
                f"{_steps // 1000}k_curr"
            )
            CONFIGS[_soft] = _cuau_flip_cell(
                _soft, sites=_sites, composition=_c, penalty=_penalty, n_steps=_steps
            )


# Soft c=0.5 at lambda=10 trains at 1200 K then collapses at the 1200 -> 800 K
# step as the hard c=0.5 cell did; lr 1e-4 from that step rescued the hard cell,
# so the soft twin gets the same schedule.
_SOFT_C50_L10 = CONFIGS["S2_cuau16_c50_l10_T500_letf_10k_curr"]
CONFIGS["S2_cuau16_c50_l10_T500_letf_10k_lowlr"] = replace(
    _SOFT_C50_L10,
    name="S2_cuau16_c50_l10_T500_letf_10k_lowlr",
    curriculum=CurriculumCfg(
        stages=tuple(
            CurriculumStageCfg(start_step=k * 2500, sigma=cuau_sigma(T), lr=lr)
            for k, (T, lr) in enumerate(
                ((1200.0, 1e-3), (800.0, 1e-4), (600.0, 1e-4), (500.0, 1e-4))
            )
        )
    ),
)


# House-strength 16-site cells, mirroring the hard rung's `*_50k_house` cells: 50k
# steps, ne128, seven-stage ladder linear in beta 1200 K -> 500 K, lr 1e-4 from the
# first step down (lr 1e-3 there collapses the hard c=0.5 cell), lambda=10.
def _cuau_house_curriculum(n_steps, n_stages=7, T_hot=1200.0, T_cold=500.0):
    beta_hot, beta_cold = 1.0 / T_hot, 1.0 / T_cold
    temps = [
        1.0 / (beta_hot + k * (beta_cold - beta_hot) / (n_stages - 1))
        for k in range(n_stages)
    ]
    return CurriculumCfg(
        stages=tuple(
            CurriculumStageCfg(
                start_step=round(k * n_steps / n_stages / 100) * 100,
                sigma=cuau_sigma(T),
                lr=1e-3 if k == 0 else 1e-4,
            )
            for k, T in enumerate(temps)
        )
    )


for _parent_name, _house_name in (
    ("A1_cuau16_T500_letf_10k_curr", "A1_cuau16_T500_letf_50k_house"),
    ("S2_cuau16_c25_l10_T500_letf_10k_curr", "S2_cuau16_c25_l10_T500_letf_50k_house"),
    ("S2_cuau16_c50_l10_T500_letf_10k_curr", "S2_cuau16_c50_l10_T500_letf_50k_house"),
):
    _parent = CONFIGS[_parent_name]
    CONFIGS[_house_name] = replace(
        _parent,
        name=_house_name,
        train=replace(_parent.train, n_steps=50_000),
        ctmc=replace(_parent.ctmc, n_euler_steps=128),
        curriculum=_cuau_house_curriculum(50_000),
    )

# 64-site cells onto the same ladder and lr cut (their definition above
# carries the four-stage ladder with lr 1e-3 at the 800 K step); the soft
# 64-site penalty stays at the 8x8 house lambda=50.
for _name in (
    "A1_cuau64_T500_letf_50k_curr",
    "S2_cuau64_c25_l50_T500_letf_50k_curr",
    "S2_cuau64_c50_l50_T500_letf_50k_curr",
):
    CONFIGS[_name] = replace(CONFIGS[_name], curriculum=_cuau_house_curriculum(50_000))

# 64-site flip-channel twins: at 16 sites the soft house cells only ran with the
# channel (channel-free soft cells died at the 1200 -> 800 K step); the free cell
# runs both as the channel's control at production size.
for _parent_name in (
    "A1_cuau64_T500_letf_50k_curr",
    "S2_cuau64_c25_l50_T500_letf_50k_curr",
    "S2_cuau64_c50_l50_T500_letf_50k_curr",
):
    _parent = CONFIGS[_parent_name]
    CONFIGS[f"{_parent_name}_efc"] = replace(
        _parent,
        name=f"{_parent_name}_efc",
        model=replace(_parent.model, exact_field_channel=True),
    )


# MetaDNS temperature-grid cells for the free ensemble: the 64-site free cell does
# not train at 500 K, so it is reported on MetaDNS's 1200 / 680 K rows with the
# ladder stopped there, the hard grid cells' schedule exactly.
_A1_64 = CONFIGS["A1_cuau64_T500_letf_50k_curr"]
CONFIGS["A1_cuau64_T1200_letf_10k"] = replace(
    _A1_64,
    name="A1_cuau64_T1200_letf_10k",
    train=replace(_A1_64.train, n_steps=10_000),
    curriculum=CurriculumCfg(
        stages=(CurriculumStageCfg(start_step=0, sigma=cuau_sigma(1200.0), lr=1e-3),)
    ),
)
CONFIGS["A1_cuau64_T680_letf_30k_l4"] = replace(
    _A1_64,
    name="A1_cuau64_T680_letf_30k_l4",
    train=replace(_A1_64.train, n_steps=30_000),
    curriculum=_cuau_house_curriculum(30_000, n_stages=4, T_cold=680.0),
)


# 16-site free cells on MetaDNS's temperature grid, ladder stopped at 1200 K (one
# stage) or 680 K (four stages linear in beta), as the 64-site grid cells; exact
# composition marginals come from enumeration, so these panels need no chain.
_A1_16_HOUSE = CONFIGS["A1_cuau16_T500_letf_50k_house"]
CONFIGS["A1_cuau16_T1200_letf_10k"] = replace(
    _A1_16_HOUSE,
    name="A1_cuau16_T1200_letf_10k",
    train=replace(_A1_16_HOUSE.train, n_steps=10_000),
    curriculum=CurriculumCfg(
        stages=(CurriculumStageCfg(start_step=0, sigma=cuau_sigma(1200.0), lr=1e-3),)
    ),
)
CONFIGS["A1_cuau16_T680_letf_30k_l4"] = replace(
    _A1_16_HOUSE,
    name="A1_cuau16_T680_letf_30k_l4",
    train=replace(_A1_16_HOUSE.train, n_steps=30_000),
    curriculum=_cuau_house_curriculum(30_000, n_stages=4, T_cold=680.0),
)


# Free-ensemble 16-site cell with the lr cut: the `_10k_curr` cell kept lr 1e-3
# through the 1200 -> 800 K step and its train ESS fell 3131 -> 96. MetaDNS reports
# NESS 0.85-0.94 on this cell, so the free rung gets every lever the slices got.
_A1_16 = CONFIGS["A1_cuau16_T500_letf_10k_curr"]
CONFIGS["A1_cuau16_T500_letf_10k_lowlr"] = replace(
    _A1_16,
    name="A1_cuau16_T500_letf_10k_lowlr",
    curriculum=CurriculumCfg(
        stages=tuple(
            CurriculumStageCfg(start_step=k * 2500, sigma=cuau_sigma(T), lr=lr)
            for k, (T, lr) in enumerate(
                ((1200.0, 1e-3), (800.0, 1e-4), (600.0, 1e-4), (500.0, 1e-4))
            )
        )
    ),
)


# Exact-field channel twins on the alloy: the channel reads the
# target's own flip log-ratio (-beta Delta E_i on an expansion), so it is
# exact on Cu-Au. One declared change from each lr-cut parent, as the
# Ising `_efc` twins are from theirs.
_S2_C25_L10 = CONFIGS["S2_cuau16_c25_l10_T500_letf_10k_curr"]
CONFIGS["S2_cuau16_c25_l10_T500_letf_10k_lowlr"] = replace(
    _S2_C25_L10,
    name="S2_cuau16_c25_l10_T500_letf_10k_lowlr",
    curriculum=CONFIGS["S2_cuau16_c50_l10_T500_letf_10k_lowlr"].curriculum,
)
for _parent_name in (
    "A1_cuau16_T500_letf_10k_lowlr",
    "S2_cuau16_c25_l10_T500_letf_10k_lowlr",
    "S2_cuau16_c50_l10_T500_letf_10k_lowlr",
):
    _parent = CONFIGS[_parent_name]
    CONFIGS[f"{_parent_name}_efc"] = replace(
        _parent,
        name=f"{_parent_name}_efc",
        model=replace(_parent.model, exact_field_channel=True),
    )

# House-recipe channel twins on the alloy: the lr-cut `_efc` twins showed the
# flip channel pays on free (+0.1-0.17 at 10k) and soft (c25 0.86, c50 0.47),
# and the 50k house recipe carried hard c=0.5 to 0.86-0.89 without one, so the
# 16-site table is completed on house + channel for the flip rungs.
for _parent_name in (
    "A1_cuau16_T500_letf_50k_house",
    "S2_cuau16_c25_l10_T500_letf_50k_house",
    "S2_cuau16_c50_l10_T500_letf_50k_house",
):
    _parent = CONFIGS[_parent_name]
    CONFIGS[f"{_parent_name}_efc"] = replace(
        _parent,
        name=f"{_parent_name}_efc",
        model=replace(_parent.model, exact_field_channel=True),
    )
