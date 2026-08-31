"""Soft-constraint configs for composition-controlled Ising sampling.

Imports the shared schema dataclasses from the baseline experiment and
defines its own CONFIGS dict for soft-constraint cells. The cells set
`target_composition` and `composition_penalty_strength` on `IsingCfg`,
which `IsingTarget` consumes natively to subtract the extensive VCSGC-style
penalty lambda * d * (c(x) - c_target)^2 from `log_prob` (see
`IsingTarget.composition_penalty` for the form and its rationale).

Cell-name format: `S<alphabet>_d<dim>_c<c_target_x100>_l<lambda>`.
"""

from dataclasses import replace

from discrete_flow_sampler.targets.ising import SIGMA_C
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
    # The budget twin plus the lambda anneal, nothing else. The fixed-lambda
    # twin answered the obedience question (a surviving seed reaches the
    # exact target's slope at 5x budget) but reproduced the from-scratch
    # fragility: seed 42 collapsed into one Z2 mode with ESS ~ 0
    # (2026-08-01). This cell tests the remaining attribution: does the
    # anneal restore seed survival without giving back the obedience? The
    # lambda schedule steps on the SAME boundaries as the window widening,
    # so the penalty tightens exactly as the window opens (the D=10 cells'
    # coupling), and the final stage lands on the operating point lambda=50
    # so the reported target matches every comparator.
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
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
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
    # OFFSET ANNEAL. Same lambda ramp as the annealed twin, but finished by
    # step 5k -- before the composition window first widens at 10k.
    #
    # Why this cell exists: the annealed twin collapsed on 3 of 4 seeds, and
    # the training traces localise the damage to the exact steps where lambda
    # steps up (median training ESS 4900 -> 11..630 at step 10k on all four
    # seeds, and again at 20k), while the fixed-lambda twin passes the very
    # same window boundaries almost unscathed (3456 -> 3255). At those shared
    # boundaries the annealed cell takes three hits at once: the target moves
    # (lambda 10 -> 25), the replay buffer is cleared because the target moved
    # (`_clear_replay`), and the draw window triples. This cell separates the
    # lambda discontinuity from that pile-up by moving the ramp off the window
    # boundaries entirely, so each shock is absorbed on its own.
    #
    # Prediction if the pile-up is the cause: survival returns to the
    # fixed-lambda twin's rate, because by the time the window opens the target
    # has been at lambda=50 for 5k steps and the model has re-converged.
    # Prediction if a lambda step is intrinsically fatal here: it still dies,
    # just earlier, and the anneal is simply the wrong recipe at D=4.
    #
    # 2k/5k rather than something later: the ramp must complete far enough
    # before 10k for the model to re-converge, but the low-lambda phase must
    # stay short, because at D=4 the composition quantum is 1/16 = 0.0625 --
    # wider than the stage-1 half-width of 0.05 -- so at lambda=10 the
    # requested composition barely distinguishes the reachable states and the
    # conditioning input carries almost no gradient signal.
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
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
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
    # GRADIENT-CLIP PROBE, two strengths. Everything is the offset cell above;
    # only `grad_clip_max_norm` moves, from 500 down to 50 and 100.
    #
    # Why: with the lambda ramp moved off the window boundaries, the acute
    # collapse at the lambda steps disappears -- all four seeds cross every
    # boundary healthy and are still at training ESS ~3400 at step 20k. Three
    # of them then die *slowly*, over the following 10-30k steps, and the
    # optimiser trace says why. At the final widening the pre-clip gradient
    # norm goes from ~30 to 2.4e3-1.5e4 and the clip fires on 55-100% of every
    # subsequent step; the one surviving seed peaks at 111, clips on 22% of
    # steps for 2k steps, and returns to normal. The damage is uniform across
    # the draw window (median training ESS is as bad at c=0.5 as at the edges),
    # which rules out "the wider window asks for unreachable compositions" and
    # points at the update itself rather than the target.
    #
    # The mechanism. `clip_grad_norm_` rescales rather than skips, so once the
    # clip saturates every step has magnitude exactly max_norm -- about 17x a
    # healthy step here -- in a direction estimated from importance weights
    # that have just degenerated. Step size is then set by the clip, not by the
    # gradient, so a batch carrying almost no information produces the largest
    # update the run has ever taken. That is a runaway: worse model, higher
    # weight variance, larger gradients, another maximal step. Clipping tighter
    # bounds each such step to ~2-4x a healthy one, which is the smallest
    # intervention consistent with the diagnosis.
    #
    # Two values because the healthy phase spikes as well -- the unconditioned
    # specialist trains to ESS 0.80 with 4% of steps above 500 and a p99 norm
    # near 4900 -- so clipping at 50 may squash tail gradients that were doing
    # real work. 100 keeps more of that tail and still removes the runaway.
    # If both survive 4/4 the tighter one is preferred as the stronger claim;
    # if only 100 survives, the tail matters and that is worth reporting.
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
            n_steps=50_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3,
            seed=42, grad_clip_max_norm=50.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
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
            n_steps=50_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3,
            seed=42, grad_clip_max_norm=100.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
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
    # NULL CONTROL AT THE FINAL RECIPE. The original cnull pair above prices
    # the conditioning machinery at the pre-fix recipe (10k steps, clip 500),
    # where it measured a 0.155 ESS-fraction deficit against its matched
    # specialist (0.801 -> 0.646, seed means). But the delivered amortised
    # family (50k, clip 50, offset lambda anneal) sits at 0.754 at c = 0.5 —
    # only ~0.05 below that specialist ceiling — so the 0.155 cannot be quoted
    # as the cost of the final recipe. This pair re-prices the machinery with
    # everything else set to the final recipe. Same isolation logic as the
    # cell above: identical target, identical schedule, and the only
    # difference between this cell and its `_c05_` twin below is the
    # conditioning path itself.
    #
    # Expectations, recorded before any result (seed means at c = 0.5,
    # specialist-minus-null):
    #   - cost ~0.05: the 0.155 was recipe-confounded (short training and a
    #     saturating clip amplify the machinery's variance overhead) and the
    #     writeup quotes this pair as the machinery cost of the system as
    #     delivered.
    #   - cost ~0.155 persisting: the machinery cost is recipe-independent,
    #     and the amortised family's 0.754 beating its own null control means
    #     drawing a RANGE of compositions helps training at the centre —
    #     report both facts, do not average them.
    # Either branch is reportable; a null that fails to train at all (any
    # seed ESS < 0.1) would instead indict the zero-width path and block
    # quoting any machinery number.
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
            n_steps=50_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3,
            seed=42, grad_clip_max_norm=50.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
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
    # The matched specialist ceiling for the null control above: identical in
    # every field except that conditioning is off and no composition is drawn.
    # The archived 10k/clip500 specialist ceiling (0.801) cannot serve here —
    # reusing it against a 50k/clip50 null would rebuild exactly the recipe
    # confound this pair exists to remove. The lambda anneal is target-level
    # and independent of conditioning, so it applies cleanly to a specialist.
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
            n_steps=50_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3,
            seed=42, grad_clip_max_norm=50.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2,
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
    # VERDICT (corrected 2026-08-19): the archived eval ESS/N 0.171 was an
    # eval bug — x0 drawn inline-uniform while the path used Bernoulli(0.8),
    # fixed in de9db7c AFTER this run. Production-path A100 redraw: 0.937,
    # PASSES its ESS >= 0.30 gate and beats the uniform twin (0.419) 2.24x.
    # The uniform control reproduced its archived number, so the correction
    # is attributable to the base draw alone.
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
    # VERDICT (corrected 2026-08-19): archived 0.025 was the same eval bug as
    # the matched_anneal twin. Production-path A100 redraw: 0.483 — passes
    # the 0.30 gate, so fixed-lambda is viable from a matched start, but the
    # anneal still buys ~1.9x (0.937 vs 0.483): the curriculum is not
    # redundant, it is just not load-bearing for gate survival.
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
    # The D=10 amortised recipe carrying BOTH D=4 fixes, and nothing else.
    #
    # Two changes from the cell above, each earned separately at D=4:
    #
    #  1. The penalty ramp finishes at 5k, before the first window widening at
    #     10k. Sharing a boundary with the widening turns a lambda step into an
    #     unrecoverable collapse -- the target moves while coverage is being
    #     stretched, and the sampler has no settled regime to fall back on.
    #     Offsetting removed the acute collapse at D=4 in all four seeds.
    #  2. grad_clip_max_norm 500 -> 50. This is the larger claim.
    #     `clip_grad_norm_` RESCALES rather than skips, so once the clip
    #     saturates every step has magnitude exactly max_norm regardless of how
    #     trustworthy its direction is. At 500 that is ~17x a healthy step
    #     (median norm ~30), taken along a direction estimated from importance
    #     weights that have just degenerated -- so the least informative batch
    #     produces the largest update of the run, which worsens the model,
    #     which raises the weight variance again. At 50 a saturated step is
    #     ~2x a healthy one and the loop cannot close: at D=4 the post-widening
    #     norm falls back instead of escalating, and the three seeds that died
    #     at 500 all survive.
    #
    # Deliberately NOT changed: replay_buffer_cycles stays at 4 (D=4 uses 8).
    # Depth is the only thing that mixes compositions within a batch, so 4
    # means each batch spans ~4 compositions on a 100-site task -- a plausible
    # contributor to the earlier D=10 collapses, but untested. This run buys
    # one seed at ~15 h; adding a third simultaneous change would make a
    # failure unattributable. Buffer depth is the next lever if this collapses
    # at the widening with the same signature.
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
    # Buffer depth, the lever named above. The clip50 cell still ran away at
    # the first widening: the gradient norm stepped from a median of ~800 to
    # ~76,000 across step 10,000 and never fell back, and ESS went from ~40 to
    # 1.0 for the remaining 40,000 steps. Clipping bounds the step but cannot
    # fix a batch whose composition labels are stale: with 4 cycles the replay
    # buffer holds states drawn under the PREVIOUS half-width, so at a widening
    # the model is scored on compositions its buffer never visited. Doubling to
    # 8 cycles halves the rate at which fresh compositions enter relative to
    # optimiser steps, so the buffer tracks the widened window before the loss
    # starts charging for it. Costs ~2x the outer sampling; single seed.
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
    # StableAdamW test of the amortisation forensics' PRINTED recommendation
    # (2026-08-11; pre-registered one-run test, gates frozen before launch:
    # frozen before launch). This is a NEW dated experiment, not a reopening
    # of the closed campaign: one run, seed 42, testing whether the
    # "normalised or trust-region update" soft.tex 4.4 recommends clears the
    # G0 bar (eval ESS fraction >= 0.10) that the clipped recipe failed at
    # 0.0147 (job 269622). Identical to the cyc8 cell above except the
    # optimiser: stable_adamw with the raw-gradient clip disabled, so the
    # per-tensor update-RMS threshold is the only bounding mechanism.
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
    # ---------------------------------------------------------------------
    # Neighbour log-ratio saturation. Four arms, all cloned from
    # `..._anneal_offset_clip50` above, which is the control and is on disk.
    #
    # The mechanism. `residual_lenet` bounds log p̃_t(y)/p̃_t(x) at a ceiling
    # (paper App. E.1.1 fixes it at 5, calibrated for an unpenalised Ising
    # target). Flipping one site moves the composition by exactly 1/d, so the
    # penalty λd(c−c_target)² contributes ∓2λΔ to that ratio, Δ = c(x)−c_target.
    # The ceiling therefore starts binding once
    #
    #     Δ  >  ceiling / (2λ)         — independent of d
    #
    # At λ=50 that is Δ* = 0.05, inside the obedience error a conditioned
    # sampler achieves; the control run sits at Δ = 0.078 with 26% of ratios
    # saturated and a 99th percentile of 9.9, against the 9.8 the expression
    # predicts. Once saturated, `site_terms` is evaluated at exp(5)=148 rather
    # than the true exp(9.9)≈2e4, and because ∂_t log Z_t is the MEAN of
    # (∂_t log p̃ + site_terms) it inherits the same inflation. The residual
    # then becomes a difference of two large numbers — 717.7 − 717.7 = 27.5 in
    # the control — and squaring that difference is what wrecks the gradient.
    #
    # Two ways to stop it binding, at two doses each, so they bracket:
    #   LOWER λ    shrinks the true ratio. Keeps the paper's ceiling and every
    #              comparator (the exact 4×4 0.984, and mchammer VC-SGC via
    #              κ=λ). Costs constraint tightness.
    #   RAISE the  leaves the ratio alone and stops truncating it. Keeps
    #   ceiling    tightness, but exp(2λΔ) grows without bound if Δ drifts.
    #
    # PRE-REGISTERED, written before any arm ran. λ=10 survives (Δ*=0.25);
    # λ=25 marginal (Δ*=0.10); both ceiling arms fail by gradient runaway
    # rather than by saturation, because they trade a bounded bias for an
    # unbounded term. The ceiling arms are the ones that could refute the
    # reading above: an offline probe over trained checkpoints showed the
    # residual exploding as the ceiling rises, but those models had TRAINED at
    # 5, so the probe cannot say what training at 20 from step 0 does — a
    # model never allowed to truncate may simply never let Δ grow.
    # Read `log_ratio_clamp_frac` as the mediating variable in every arm.
    "S2_d10_camort_offset_clip50_lam10": StageCfg(
        name="S2_d10_camort_offset_clip50_lam10",
        ising=IsingCfg(
            D=10, sigma=0.1, bias=0.0, target_composition=0.5,
            composition_penalty_strength=10.0,
        ),
        train=TrainCfg(
            n_steps=50_000, batch_size=128, outer_batch_size=256,
            replay_buffer_cycles=4, lr=1e-3, seed=42,
            grad_clip_max_norm=50.0, warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        # No λ ramp: the terminal λ IS 10, so there is nothing to anneal to.
        # Holding it fixed also removes the λ-step boundaries entirely, which
        # keeps this arm from confounding "lower λ" with "fewer boundaries" —
        # the control's own offset schedule already showed the boundaries
        # survivable, so a fixed λ is the cleaner single-variable change.
        lambda_curriculum=None,
        composition=CompositionCfg(
            centre=0.5, half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # The staircase cell: flat λ=10 with coverage capped at half-width 0.20
    # and widened gradually. Rationale, from the archived λ=10 arm above:
    # that run was pristine through 20k (loss 2.76, grad 44, clamp_frac
    # 0.0000 — no saturation at Δ*=0.25) and was knocked into a permanent
    # excursion/recovery cycle by the single hw 0.15→0.30 jump, a
    # penalty-variance transient (var_dt_log_p̃ 11→224 in one step), not a
    # clamp event. Three changes follow directly:
    #   CAP hw at 0.20 — covers requested compositions [0.3, 0.7] exactly;
    #     the inherited 0.30 over-covered to [0.2, 0.8], and the extra width
    #     is what delivered the killing variance dose (λd·hw² at 0.30 is
    #     2.25× the 0.20 value).
    #   WIDEN in ≤0.05 increments with ≥8k dwell — the archived arm survived
    #     a +0.10 widening at 10k (recovery ~5–6k steps under clip 50), so
    #     +0.05 per stage is a sub-fatal dose by construction. First widening
    #     at 20k so the hw=0.05 phase can prove in-loop ESS first.
    #   CHECKPOINT every 2.5k — the λ=10 arm's final.pt landed mid-excursion
    #     (obedience slope 0.079 against in-run states at loss ~5); eval must
    #     be able to select a healthy state by a rule fixed in advance
    #     (clamp_frac == 0, grad fallen back, CV ratio < 1).
    "S2_d10_camort_offset_clip50_lam10_hw20": StageCfg(
        name="S2_d10_camort_offset_clip50_lam10_hw20",
        ising=IsingCfg(
            D=10, sigma=0.1, bias=0.0, target_composition=0.5,
            composition_penalty_strength=10.0,
        ),
        train=TrainCfg(
            n_steps=50_000, batch_size=128, outer_batch_size=256,
            replay_buffer_cycles=4, lr=1e-3, seed=42,
            grad_clip_max_norm=50.0, warmup_steps=2000,
            checkpoint_every=2_500,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2,
            condition_on_composition=True,
        ),
        estimator="control_variate",
        # Flat λ, as in the arm above: the terminal λ is 10 and holding it
        # fixed keeps the staircase the only moving schedule.
        lambda_curriculum=None,
        composition=CompositionCfg(
            centre=0.5, half_width=0.05,
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
            D=10, sigma=0.1, bias=0.0, target_composition=0.5,
            composition_penalty_strength=25.0,
        ),
        train=TrainCfg(
            n_steps=50_000, batch_size=128, outer_batch_size=256,
            replay_buffer_cycles=4, lr=1e-3, seed=42,
            grad_clip_max_norm=50.0, warmup_steps=2000,
        ),
        ctmc=CTMCCfg(n_euler_steps=128),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2,
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
            centre=0.5, half_width=0.05,
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
            D=10, sigma=0.1, bias=0.0, target_composition=0.5,
            composition_penalty_strength=50.0, log_ratio_clamp=20.0,
        ),
        train=TrainCfg(
            n_steps=50_000, batch_size=128, outer_batch_size=256,
            replay_buffer_cycles=4, lr=1e-3, seed=42,
            grad_clip_max_norm=50.0, warmup_steps=2000,
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
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5, half_width=0.05,
            curriculum=(
                CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
                CompositionCurriculumStageCfg(start_step=10_000, half_width=0.15),
                CompositionCurriculumStageCfg(start_step=20_000, half_width=0.30),
            ),
        ),
        wandb_project="dnfs-constraints",
    ),
    # Ceiling 50: unbinds to Δ = 0.50, i.e. never binds anywhere in the swept
    # range. The aggressive end of the bracket — if raising the ceiling helps
    # at all, it helps here; if the unbounded inflow term is the problem, this
    # is where it shows worst. exp(50)≈5e21, still finite in float32.
    "S2_d10_camort_offset_clip50_clamp50": StageCfg(
        name="S2_d10_camort_offset_clip50_clamp50",
        ising=IsingCfg(
            D=10, sigma=0.1, bias=0.0, target_composition=0.5,
            composition_penalty_strength=50.0, log_ratio_clamp=50.0,
        ),
        train=TrainCfg(
            n_steps=50_000, batch_size=128, outer_batch_size=256,
            replay_buffer_cycles=4, lr=1e-3, seed=42,
            grad_clip_max_norm=50.0, warmup_steps=2000,
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
                    start_step=2_000, composition_penalty_strength=25.0
                ),
                LambdaCurriculumStageCfg(
                    start_step=5_000, composition_penalty_strength=50.0
                ),
            )
        ),
        composition=CompositionCfg(
            centre=0.5, half_width=0.05,
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
        _FC_NE128_BASE, name=_window_name,
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
        _AMORT_SPECIALIST_BASE, name=_twin_name,
        ising=replace(_AMORT_SPECIALIST_BASE.ising, target_composition=_c),
    )

_FLAT_WINDOW_BASE = CONFIGS["S2_d4_camort_50k_l50_letf_anneal_offset_clip50"]
_flat_window_name = f"{_FLAT_WINDOW_BASE.name}_flatw30"
CONFIGS[_flat_window_name] = replace(
    _FLAT_WINDOW_BASE, name=_flat_window_name,
    composition=replace(
        _FLAT_WINDOW_BASE.composition, half_width=0.30, curriculum=None,
    ),
)


# The lambda-sweep exact-field-channel twins (s90, 2026-08-29): rerun
# tab:soft-lambda-sweep with the closed-form flip channel wired in
# (ModelCfg.exact_field_channel; the flip twin of the hard chapter's swap
# channel, gain zero-init so each twin is bit-identical to its parent at
# step 0). One declared change per twin, pinned by
# tests/test_exact_flip_channel.py. Built by replace() from the parents so
# recipe parity is by construction, not by copy-paste discipline. The s90
# regression motivating this measured the closed form at ~95% of every
# trained lambda=50 specialist, with the PENALTY column carrying it — the
# prediction under test is that the channel rescues the all-or-nothing
# 10x10 seeds (0.02/0.06/0.78/0.05 at lambda=50 in print).
LAMBDA_SWEEP_PARENTS = tuple(
    f"S2_d{side}_c05_l{lam}_letf{suffix}"
    for side, suffix in ((4, ""), (10, "_ne64"))
    for lam in (5, 10, 50, 100)
)
for _parent_name in LAMBDA_SWEEP_PARENTS:
    _parent = CONFIGS[_parent_name]
    CONFIGS[f"{_parent_name}_efc"] = replace(
        _parent, name=f"{_parent_name}_efc",
        model=replace(_parent.model, exact_field_channel=True),
    )


# ---------------------------------------------------------------------------
# s95 soft-chapter revamp (2026-08-30): the 8x8 house family. Production
# moves to d=64 (the hard chapter's record size, so the cross-route
# comparison is matched-size at BOTH couplings once the _sc half lands);
# one specialist family serves the F(c) curve and the house table.


def soft_house_recipe(cell: StageCfg) -> StageCfg:
    """s95 house recipe for NEW soft cells, as a recipe transform.

    Four declared instrument changes on top of an archived parent, nothing
    else: exact_field_channel=True (the closed-form penalty response — s95
    sweep verdict: full 10x10 lambda=50 rescue, 0.95/0.93/0.94/0.96 against
    the parent's 0.02/0.06/0.78/0.05), compile_model=True (hard's measured
    2.2x inner updates; certified by a d4+d64 loss-gap gate before any
    fan-out, since compile has priors on this codebase), ema_decay=0.9999
    (dual eval — hard's marginal-seed rescue, 0.750 -> 0.830 class), and
    train.c_t_from_rollout=True (bit-identical CV grid from the rollout's
    own forwards). Optimiser-side values (lr, warmup 2000, clip 500,
    batch/buffer, 50k steps) stay PARENT-matched so the before/after
    channel comparison carries no second change; evals stay fp32 end to
    end (bf16/SDPA are hard-chapter-only by standing decision).
    """
    return replace(
        cell,
        model=replace(
            cell.model, exact_field_channel=True, compile_model=True),
        train=replace(cell.train, c_t_from_rollout=True),
        ema_decay=0.9999,
    )


# The trained compositions: every c* is lattice-representable at d=64
# (16/24/32 sites). 0.625 and 0.75 are NOT trained — F(c) = F(1-c) under a
# global spin flip, so the printed curve mirrors them for free and a
# trained 0.75 would duplicate 0.25. Composition tag = c_target x 1000
# (the older x100 convention cannot write 0.375); 0.25 is the stress
# window, further from half-filling than the retired 0.30.
SOFT_HOUSE_WINDOWS = ((0.25, "c0250"), (0.375, "c0375"), (0.50, "c0500"))
_D8_HOUSE_PARENT = CONFIGS["S2_d8_c03_l50_letf_ne128"]

for _c_target, _c_tag in SOFT_HOUSE_WINDOWS:
    for _sigma, _sigma_suffix in ((0.1, ""), (SIGMA_C, "_sc")):
        _house_name = f"S2_d8_{_c_tag}_l50_letf_ne128_house{_sigma_suffix}"
        CONFIGS[_house_name] = soft_house_recipe(replace(
            _D8_HOUSE_PARENT,
            name=_house_name,
            ising=replace(
                _D8_HOUSE_PARENT.ising,
                sigma=_sigma, target_composition=_c_target),
        ))

# The wave-2 control: house recipe MINUS the channel, critical coupling,
# centre composition only. If this fails where _house_sc trains, the
# failure->rescue story gets a measured second act at matched size; if
# both train, the channel's sigma_c claim rests on the efficiency columns
# instead. Channel flag is the ONLY lever this cell gives back.
_HOUSE_SC_CENTRE = CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc"]
CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_nochan"] = replace(
    _HOUSE_SC_CENTRE,
    name="S2_d8_c0500_l50_letf_ne128_house_sc_nochan",
    model=replace(_HOUSE_SC_CENTRE.model, exact_field_channel=False),
)

# The sigma_c anneal arm (s99): completes the three-fates trio at the
# production size and coupling -- parent (= the nochan control), anneal,
# channel -- so fig:penalty-variance can be drawn at 8x8 sigma_c instead
# of 10x10 and the chapter body becomes single-size (user decision,
# s99). The nochan control PLUS the chapter's declared lambda schedule
# (10/25/50 at 0/10k/20k), one lever, test-pinned; the anneal's job is
# the deferred-shock trace: it defers the lambda^2 Var[delta_P] shock
# and repays at each boundary where the channel discharges it once.
_NOCHAN_SC = CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_nochan"]
CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_anneal"] = replace(
    _NOCHAN_SC,
    name="S2_d8_c0500_l50_letf_ne128_house_sc_anneal",
    lambda_curriculum=LambdaCurriculumCfg(stages=(
        LambdaCurriculumStageCfg(start_step=0,
                                 composition_penalty_strength=10.0),
        LambdaCurriculumStageCfg(start_step=10_000,
                                 composition_penalty_strength=25.0),
        LambdaCurriculumStageCfg(start_step=20_000,
                                 composition_penalty_strength=50.0),
    )),
)

# Matched-base twins (s99): base_composition = c* at the OFF-CENTRE
# windows, both couplings — at c* = 0.5 the house cells' Bernoulli(0.5)
# base is already matched, so the centre rows anchor both columns
# unchanged. What the pair measures: the s010 c=0.25 window is uniformly
# marginal (raw ESS 0.15-0.29) and its floor-free F(c) reads +0.024
# nats/site off truth against a bootstrap claiming +-0.001 (weight
# collapse lies to the bootstrap); at sigma_c c=0.25 the family is
# degenerate outright with training alive (in-loop ESS 47-75/batch, zero
# clamp), i.e. the eval collapse is distributional. If the matched base
# rescues a window the failure was base reachability; if not, it is the
# target itself. One declared lever vs the run house twin (test-pinned);
# the base enters only the x0 draw and the log w0 term, both on the
# corrected post-de9db7c path — the exact-field channel is pure target
# physics and does not see it.
for _c_target, _c_tag in SOFT_HOUSE_WINDOWS:
    if _c_target == 0.50:
        continue
    for _sigma_suffix in ("", "_sc"):
        _house_twin = CONFIGS[
            f"S2_d8_{_c_tag}_l50_letf_ne128_house{_sigma_suffix}"]
        _mb_name = f"S2_d8_{_c_tag}_l50_letf_ne128_house_mb{_sigma_suffix}"
        CONFIGS[_mb_name] = replace(
            _house_twin,
            name=_mb_name,
            ising=replace(_house_twin.ising, base_composition=_c_target),
        )

# D=4 gates for the compile-parity check (validate-at-D=4 rule): the full
# house recipe and its eager twin. The gate passes when their loss traces
# agree to compile tolerance (1e-5-class, never bit-parity) — the GFN
# wave's gate pattern, rerun here because compile x leTF x sigma_c is
# untested and the factorised chassis's compile history says gate first.
_D4_GATE_PARENT = CONFIGS["S2_d4_c05_l50_letf"]
CONFIGS["S2_d4_c05_l50_letf_house_gate"] = soft_house_recipe(replace(
    _D4_GATE_PARENT, name="S2_d4_c05_l50_letf_house_gate"))
_D4_GATE = CONFIGS["S2_d4_c05_l50_letf_house_gate"]
CONFIGS["S2_d4_c05_l50_letf_house_gate_eager"] = replace(
    _D4_GATE, name="S2_d4_c05_l50_letf_house_gate_eager",
    model=replace(_D4_GATE.model, compile_model=False),
)

# Wave 3: the amortised 4x4 family on the house recipe. The archived
# fixed-lambda 50k parent plus the four recipe levers, nothing else — no
# lambda anneal, no offset, no clip. Those cells existed to service
# anneal-boundary shocks (the lambda steps that took the annealed twin's
# ESS 4900 -> 11 at each shared boundary); the channel has no boundaries,
# so if this cell trains 4/4 the whole offset/clip confound family
# collapses out of the chapter (plan wave 3). The channel reads each
# row's own conditioned composition via the per-row c* path (s95,
# `4f94595`), so the amortised widening window keeps its meaning.
CONFIGS["S2_d4_camort_50k_l50_letf_house"] = soft_house_recipe(replace(
    CONFIGS["S2_d4_camort_50k_l50_letf"],
    name="S2_d4_camort_50k_l50_letf_house"))

# Matched-base amortisation (s101, plan 2026-08-31-soft-camort-matched-base).
# Design mirrors hard camort's shape where soft's flip dynamics permit it:
# discrete spine draw from step 0 (no widening staircase, no lambda
# curriculum — the D=10 campaign's G0/G1 convictions removed outright, not
# survived), base matched to the drawn c per cycle (motivated by the 8x8 mb
# twins: the off-centre specialist collapse was base reachability, one
# lever, full rescue at every window). The base is Bernoulli(c), NOT hard's
# slice-uniform mixture: single-flip dynamics leave a slice at the first
# flip and the Eq. 4 path is -inf off-slice for t<1 under a slice base —
# full support is a structural requirement, not a softening. The channel
# stays on: soft has no conservation law pinning delivered composition, so
# the request rides the conditioning scalar; zero-init means it costs
# nothing if the matched base has absorbed its job. Exactly three levers
# off the house centre cell, pinned by test_matched_base_amortisation.
_CAMORT_SPINE = CompositionCfg(
    centre=0.5, half_width=0.0, values=(0.25, 0.375, 0.5))
for _sigma_suffix in ("", "_sc"):
    _camort_parent = CONFIGS[
        f"S2_d8_c0500_l50_letf_ne128_house{_sigma_suffix}"]
    _camort_name = f"S2_d8_camort_l50_letf_ne128_house{_sigma_suffix}"
    CONFIGS[_camort_name] = replace(
        _camort_parent,
        name=_camort_name,
        ising=replace(_camort_parent.ising, base_matches_composition=True),
        model=replace(_camort_parent.model, condition_on_composition=True),
        composition=_CAMORT_SPINE,
    )

# D=4 gate for the matched-base cells (validate-at-D=4 rule): the wave-3
# camort house cell with the staircase swapped for the spine draw and the
# base matched — every spine c is an integer site count at d=16 (4/6/8).
_D4_CAMORT_HOUSE = CONFIGS["S2_d4_camort_50k_l50_letf_house"]
CONFIGS["S2_d4_camort_mb_50k_l50_letf_house"] = replace(
    _D4_CAMORT_HOUSE,
    name="S2_d4_camort_mb_50k_l50_letf_house",
    ising=replace(_D4_CAMORT_HOUSE.ising, base_matches_composition=True),
    composition=_CAMORT_SPINE,
)

# The tab:amort-4x4 comparator rows, SAME recipe (s96): pricing the
# conditioning machinery against comparators on the retiring clip50
# recipe would rebuild the recipe confound the archived cnull pair was
# built to remove — so specialists and null re-run on the house recipe
# at the amortised 50k budget. Windows follow the revamp set
# {0.25, 0.375, 0.50} (+ mirrors free): every c* is an integer site
# count at d=16 (4/6/8 sites), unlike the retired {0.30, 0.65, 0.80}
# grid (4.8/10.4/12.8). The scatter panels (app:logp-scatters, soft row)
# read the c0500 and mirror-edge cells of exactly this family. The
# obedience reference slope 0.976 was measured on the OLD request grid
# and must be re-derived by enumeration before any new slope is quoted
# against it.
_D4_SPECIALIST_HOUSE_BASE = soft_house_recipe(replace(
    CONFIGS["S2_d4_c05_l50_letf"],
    name="S2_d4_c0500_50k_l50_letf_house",
    train=replace(CONFIGS["S2_d4_c05_l50_letf"].train, n_steps=50_000),
))
CONFIGS["S2_d4_c0500_50k_l50_letf_house"] = _D4_SPECIALIST_HOUSE_BASE
for _c_target, _c_tag in ((0.25, "c0250"), (0.375, "c0375")):
    _twin_name = f"S2_d4_{_c_tag}_50k_l50_letf_house"
    CONFIGS[_twin_name] = replace(
        _D4_SPECIALIST_HOUSE_BASE, name=_twin_name,
        ising=replace(
            _D4_SPECIALIST_HOUSE_BASE.ising, target_composition=_c_target),
    )

# Null control on the house recipe: conditioning path ON, window width
# ZERO, so vs the c0500 specialist above the only differences are the
# machinery itself (model flag + the zero-width composition config).
CONFIGS["S2_d4_cnull_50k_l50_letf_house"] = soft_house_recipe(replace(
    CONFIGS["S2_d4_cnull_l50_letf"],
    name="S2_d4_cnull_50k_l50_letf_house",
    train=replace(CONFIGS["S2_d4_cnull_l50_letf"].train, n_steps=50_000),
))
