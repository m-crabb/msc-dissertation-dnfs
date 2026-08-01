"""Soft-constraint configs for composition-controlled Ising sampling.

Imports the shared schema dataclasses from the baseline experiment and
defines its own CONFIGS dict for soft-constraint cells. The cells set
`target_composition` and `composition_penalty_strength` on `IsingCfg`,
which `IsingTarget` consumes natively to subtract the extensive VCSGC-style
penalty lambda * d * (c(x) - c_target)^2 from `log_prob` (see
`IsingTarget.composition_penalty` for the form and its rationale).

Cell-name format: `S<alphabet>_d<dim>_c<c_target_x100>_l<lambda>`.
"""

from experiments.dnfs_baseline_01.configs import (
    CompositionCfg,
    CompositionCurriculumStageCfg,
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
    # Amortised validation cell at the small lattice: a controlled clone of
    # S2_d4_c05_l50_letf above, differing ONLY in that the model is
    # conditioned on c and c is drawn per outer cycle instead of being fixed
    # at 0.5. Everything else — 10k steps, ne50, λ=50 held fixed, hidden 64 —
    # is copied, so a gap against the archived four-seed c=0.5 record
    # (ess_fraction 0.58–0.85) is attributable to amortisation and nothing
    # else. It runs before the D=10 cells because it is minutes rather than
    # hours: a cheap end-to-end proof of the conditioning, the per-cycle draw,
    # the buffered baseline and the per-composition sweep.
    #
    # λ is NOT annealed here, unlike the D=10 cells. The anneal exists because
    # λ=50 from scratch trains 1 seed in 4 at D=10; at D=4 the archived λ=50
    # cell trained 4/4, and matching the comparator's recipe matters more than
    # inheriting a fix for a problem this lattice does not have.
    #
    # The c-window still widens (0.05 → 0.15 → 0.30 at the same fractions of
    # the run as the D=10 schedule, 20% and 40%): the easy end is c ≈ 0.5,
    # where base and target compositions already agree, and the final window
    # [0.20, 0.80] has to cover the archived comparators at 0.30 and 0.50.
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
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
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
    # Narrow-window twin of the cell above: identical in every respect except
    # that the window stops widening at ±0.15 instead of ±0.30.
    #
    # It exists to answer one question the wide run raised and could not
    # settle. The wide run's conditioning came out heavily attenuated —
    # realised composition tracked requested composition with slope 0.39,
    # against 0.984 for the exact enumerated target — so the model travels
    # under 40% of the distance it is asked to. Two explanations fit that
    # equally well: the range is too wide to cover with a 10k-step budget
    # (a coverage cost, which trades off and can be quantified), or the
    # conditioning is attenuated however narrow the range gets (intrinsic,
    # and no amount of budget-shuffling fixes it).
    #
    # Halving the final window separates them. If the slope rises toward 1
    # inside [0.35, 0.65], the deficit is coverage and buys a real trade-off
    # curve; if it stays near 0.4, the attenuation is intrinsic and the
    # escalation is architectural rather than a matter of scheduling.
    #
    # The sweep still evaluates the full ten compositions, so 0.30 and 0.80
    # now sit OUTSIDE the training range by construction — those rows measure
    # extrapolation, not interpolation, and must be read as such.
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
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
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
    # Budget twin of S2_d4_camort_l50_letf: 50k steps instead of 10k, and
    # NOTHING else changed — same Euler budget, same fixed λ, same capacity,
    # same widening window. One variable, so a change in conditioning fidelity
    # is attributable to training budget alone.
    #
    # This is not a proxy for the D=10 launch, and that distinction is the
    # reason it is worth the GPU time. The attenuation result — realised
    # composition tracking requested composition at slope 0.39 against the
    # exact target's 0.984 — can only be stated where the target is
    # enumerable, i.e. d = 16 here. At D=10 the lattice is d = 100 sites, so
    # there are 2^100 states, no exact slope exists to compare against, and
    # the claim cannot be made at all. If attenuation survives a 5x budget it
    # is a property of the method rather than of undertraining, and that is a
    # far stronger statement than the 10k runs can support.
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
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        # Same shape of widening, stretched over the longer run so the model
        # spends the same FRACTION of training at each width as the 10k cell.
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
    # NULL CONTROL for the amortisation machinery. Conditioning is ON, but the
    # draw window has zero width, so every outer cycle draws c = 0.5 exactly.
    # Mathematically this IS the S2_d4_c05_l50_letf specialist — same target,
    # same fixed composition — yet it reaches that target through the whole
    # amortisation path: the model adapter, the per-cycle draw, the per-state
    # composition buffer, the per-state c_t baseline, and the target binding.
    #
    # It exists because every other check on that path is partial. The exact
    # enumeration validates the target; the archived specialists validate the
    # sampler and the eval; neither touches the conditioning code, because a
    # specialist has none. This cell is the one comparison where a discrepancy
    # can only be the machinery.
    #
    # Pass condition: it reproduces the specialist — ESS fraction ~0.74 over
    # seeds 42-45 and realised composition ~0.500. Anything materially worse
    # means the attenuation measured on the wide and narrow cells is an
    # artefact and every number from them needs re-reading.
    #
    # It is NOT expected to be bit-identical, and that is the other reason it
    # is worth running: amortised training replaced the specialist's "latest
    # c_t grid applied to the whole buffer" approximation with a per-state
    # buffered c_t. That change is research-bearing, rides on every amortised
    # run, and no archived cell exercises it. This isolates it.
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
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        composition=CompositionCfg(centre=0.5, half_width=0.0),
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
    # F(c) campaign gate windows (2026-06-13): off-centre clones of the won
    # anneal rung, varying only target_composition. c=0.65 (typical) and
    # c=0.80 (stress) launch first and gate the rest of the composition sweep
    # against the pre-declared failure criteria, including the ESS 0.30 floor
    # the fallback rung below cites. The anneal
    # schedule and every other knob are held fixed so the off-centre runs are a
    # controlled test of whether the recipe generalises away from c=0.5.
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
    # F(c) sweep windows released 2026-06-17: ne64 anneal clones of the won
    # c=0.5 rung, target_composition only. The integrator stays ne64 (no
    # measured case for a finer Euler grid on the curve; the off-centre tail
    # beyond ~0.65 is left to the hard sampler). With Z_2 reflection the set
    # {0.30, 0.50, 0.55, 0.60, 0.65} covers compositions 0.30 to 0.70.
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
    # F(c) gate fallback rung (2026-06-16): clone of the c=0.80 stress window
    # with a finer Euler grid only (ne64 -> ne128), testing whether the finer
    # integration rescues seed survival at the most off-centre target. The
    # ne64 gate went 1/4 over the ESS 0.30 floor; this is the cheap single-seed
    # diagnostic before spending the per-window lambda or full-window ne128 rung.
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
    # Matched-base validation (2026-06-17): the c=0.80 ne128 anneal window with
    # a per-site Bernoulli(0.80) base, so the flow starts centred and only
    # tightens width. base_composition is the only change vs the ne128 anneal.
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
    # Matched base + fixed lambda=50 (no anneal): tests whether the matched
    # start lets the lambda curriculum be dropped entirely.
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
    # n_euler consistency control (2026-06-17): the c=0.5 anneal witness at
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
    # Amortised cells: ONE model conditioned on the target composition,
    # to be measured against the six per-composition specialists above.
    #
    # Both clone the surviving recipe verbatim — λ 10→25→50, ne128, warmup
    # 2000, grad-clip 500 — because the λ anneal is what took D=10 seed
    # survival from 1/4 to 4/4, and ne128 is what took the c=0.5 ESS
    # fraction from 0.699 (3/4 seeds) to 0.918 (4/4).
    # ---------------------------------------------------------------
    # Continuous c on a widening window, mirroring the λ schedule step for
    # step: c ≈ 0.5 is the easy end (base and target compositions already
    # agree) so the model learns there first, then generalises outward to
    # the edges where the composition gap is largest.
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
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2,
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
    # Control: amortise over ONLY the six compositions we have specialists
    # for. Held-out points between those atoms separate genuine
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
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2,
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
            centre=0.5, values=(0.3, 0.5, 0.55, 0.6, 0.65, 0.8),
        ),
        wandb_project="dnfs-constraints",
    ),
}
