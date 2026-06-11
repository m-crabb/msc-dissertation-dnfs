"""Soft-constraint configs for composition-controlled Ising sampling.

Imports the shared schema dataclasses from the baseline experiment and
defines its own CONFIGS dict for soft-constraint cells. The cells set
`target_composition` and `composition_penalty_strength` on `IsingCfg`,
which `IsingTarget` consumes natively to add a penalty term to
`log_prob` (see docs/findings/2026-05-13-constrained-binary-alloy-probe.md
for derivation).

Cell-name format: `S<alphabet>_d<dim>_c<c_target_x100>_l<lambda>`.
"""

from experiments.dnfs_baseline_01.configs import (
    CTMCCfg,
    EvalCfg,
    IsingCfg,
    LambdaCurriculumCfg,
    LambdaCurriculumStageCfg,
    ModelCfg,
    StageCfg,
    TrainCfg,
)

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
    # λ sweep at the c=0.5 report operating point (2026-06-11 design): clones
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
    # D=10 c_target=0.5 subcritical headline cell. Follows the c=0.3 ne128 cell's
    # stability stack (warmup=2000, lambda=50) but drops n_euler 128 -> 64 to match
    # the paper-faithful baseline D=10 grid (DNFS uses T=64); the old 128 was an
    # unvalidated conservative pick. This is a controlled escalation of the
    # validated D=4 c=0.5 run up to D=10. c=0.3 is dropped from the report
    # (2026-06-09), so this is the D=10 soft witness the final report ships.
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
    # λ sweep at the c=0.5 report operating point (2026-06-11 design): clones
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
    # Recipe ladder rung 1 (2026-06-11): clone of the l50 ne64 witness plus a
    # 10->25->50 lambda anneal. The c=0.5 lambda sweep showed lambda<=10
    # trains 4/4 at ESS ~0.99 while lambda=50 from scratch goes 1/4; the
    # anneal learns the physics in the trainable window, then tightens onto
    # the operating point. Replay clears at stage boundaries (training.py).
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
    # Recipe ladder fallback rung (2026-06-11): clone of the l50 ne64 witness
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
}
