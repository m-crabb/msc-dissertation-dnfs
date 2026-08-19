"""Twin pins for the 16x16 stage-1 screen, the 12x12 bracket and the
clip-threshold continuation (2026-08-15).

Every cell here is read against a base by differencing, so its
interpretability rests on being that base with ONLY the declared fields
changed. These pins encode each declaration: rebuild the arm with its
declared fields reset to the base's values and require dataclass equality.
A pin failing means an undeclared variable rode along and the arm's read
would be unattributable — the exact failure mode that cost the first
capacity arm its verdict (hidden_dim varied with the learning rate frozen
at a value tuned for the narrow net).
"""
from dataclasses import replace

import pytest

from experiments.constrained_hard_03.configs import CONFIGS


def _reset(cell, *, model=None, train=None, ctmc=None, **top):
    """Rebuild `cell` with the given per-dataclass field overrides."""
    if model:
        cell = replace(cell, model=replace(cell.model, **model))
    if train:
        cell = replace(cell, train=replace(cell.train, **train))
    if ctmc:
        cell = replace(cell, ctmc=replace(cell.ctmc, **ctmc))
    return replace(cell, **top) if top else cell


FMO2_BASE = "H2_d256_scr5k_fmo2"

# arm name -> the reset that must reproduce the fmo2 base exactly.
FMO2_ARM_DECLARATIONS = {
    "H2_d256_scr5k_fmo2_h128": dict(model={"hidden_dim": 32}),
    "H2_d256_scr5k_fmo2_h128_lr03": dict(
        model={"hidden_dim": 32}, train={"lr": 1e-3},
    ),
    "H2_d256_scr5k_fmo2_lr03": dict(train={"lr": 1e-3}),
    "H2_d256_scr5k_fmo2_L3": dict(model={"n_layers": 2}),
    "H2_d256_scr5k_fmo2_L3_lr03": dict(
        model={"n_layers": 2}, train={"lr": 1e-3},
    ),
    "H2_d256_scr5k_fmo2_clip2000": dict(
        train={"grad_clip_max_norm": 500.0},
    ),
    "H2_d256_scr5k_fmo2_ne512_b512": dict(
        ctmc={"n_euler_steps": 128}, train={"batch_size": 128},
    ),
    "H2_d256_scr20k_fmo2": dict(train={"n_steps": 5_000}),
    # Phase 2 (2026-08-18): the batch-only decomposition arm — the
    # microbatch reset rides because loss_microbatch_size on this arm is
    # the noise-scale instrumentation, gradient-exact by parity pin, not
    # a recipe variable.
    "H2_d256_scr5k_fmo2_b512": dict(
        train={"batch_size": 128, "loss_microbatch_size": None},
    ),
    # Phase 2 (2026-08-18): the cold-CV arm — estimator the only change.
    "H2_d256_scr5k_fmo2_cv": dict(estimator="naive_mc"),
}


@pytest.mark.parametrize("arm_name", sorted(FMO2_ARM_DECLARATIONS))
def test_fmo2_screen_arm_mirrors_base_except_declared_fields(arm_name):
    base = CONFIGS[FMO2_BASE]
    arm = CONFIGS[arm_name]
    rebuilt = _reset(arm, name=base.name, **FMO2_ARM_DECLARATIONS[arm_name])
    assert rebuilt == base


def test_screen_base_recipe_is_flat_sigma010_naive():
    """The screen's fixed frame: flat sigma=0.10 (no curriculum, so no
    buffer clears, EMA resets or lr steps inside the window), naive c_t in
    every arm so the 1/128 noise floor cancels in same-screen comparisons,
    and the archived 16x16 grid/batch/clip."""
    for name in [FMO2_BASE, "H2_d256_scr5k_ma", "H2_d256_scr5k_mo"]:
        cell = CONFIGS[name]
        assert cell.ising.sigma == 0.10
        assert cell.curriculum is None
        assert cell.estimator == "naive_mc"
        assert cell.train.n_steps == 5_000
        assert cell.ising.D == 16


def test_ma_screen_base_mirrors_fmo2_base_except_head_family_fields():
    """The MA/fmo2 bases must differ only in the head family and its two
    riding conventions (EMA shadow, site orderings) plus the eval chunk
    sized to each head's measured memory. Anything else differing would
    break the cross-family read of the capacity arms."""
    fmo2 = CONFIGS[FMO2_BASE]
    ma = CONFIGS["H2_d256_scr5k_ma"]
    rebuilt = replace(
        ma,
        name=fmo2.name,
        head_kind=fmo2.head_kind,
        ema_decay=fmo2.ema_decay,
        site_orderings=fmo2.site_orderings,
        eval=replace(ma.eval, eval_sample_chunk=fmo2.eval.eval_sample_chunk),
    )
    assert rebuilt == fmo2


def test_ma_h128_bridge_arm_covaries_lr_with_width():
    """The bridge capacity arm must carry BOTH hidden 128 and lr 3e-4 —
    a frozen-lr h128 arm is positioned to reproduce the archived 8x8
    false negative, so the (h128, lr 1e-3) corner is deliberately absent
    from the MA family."""
    base = CONFIGS["H2_d256_scr5k_ma"]
    arm = CONFIGS["H2_d256_scr5k_ma_h128_lr03"]
    assert arm.model.hidden_dim == 128
    assert arm.train.lr == 3e-4
    # The declared fields plus the memory schedule: loss_microbatch_size
    # is gradient-identical to the base's single backward (parity-pinned
    # in test_loss_microbatch_parity), carried only because the h128
    # graph is measured not to fit an A100-80GB in one backward.
    assert arm.train.loss_microbatch_size == 64
    rebuilt = _reset(
        arm, name=base.name,
        model={"hidden_dim": 32},
        train={"lr": 1e-3, "loss_microbatch_size": None},
    )
    assert rebuilt == base


def test_loss_microbatch_schedule_is_confined_to_the_measured_oom_arms():
    """The backward-slicing schedule may live ONLY on cells that declare a
    reason for it: the two arms whose single-backward graph is measured to
    exceed an A100-80GB, and (2026-08-18) the batch-decomposition arm,
    where the slices ARE the gradient-noise-scale instrument (per-slice
    sqnorms + full-batch norm invert the McCandlish two-batch identity).
    It is gradient-identical everywhere (test_loss_microbatch_parity), but
    confining it keeps every other cell on the archived single-backward
    path byte-for-byte, so archived comparisons stay bit-exact."""
    expected = {
        "H2_d256_scr5k_ma_h128_lr03": 64,
        # The muP-init arm is the bridge arm's twin, so the h128 OOM
        # schedule rides unchanged (2026-08-18).
        "H2_d256_scr5k_ma_h128_lr03_mup": 64,
        "H2_d256_scr5k_mo": 16,
        "H2_d256_scr5k_fmo2_b512": 128,
        "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive": 128,
        # The buffer-invariant arm is the recipe cell's twin (cycles the
        # only change), so the noise-scale instrument rides unchanged
        # (2026-08-19).
        "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_buf2": 128,
    }
    for name, cell in CONFIGS.items():
        assert cell.train.loss_microbatch_size == expected.get(name), name


def test_ma_mup_arm_is_the_bridge_arm_with_readout_scale_the_only_change():
    """muP-init arm (2026-08-18): the bridge arm (h128 + lr03) with the
    readout score scale 32/128 the ONLY change, so the init-transient read
    is chargeable to parametrization alone. The value is the muP readout
    prescription hidden_base/hidden for the verified sqrt(h) score growth
    (the <LayerNorm'd H, omega_diff> dot has no fan-in compensation)."""
    base = CONFIGS["H2_d256_scr5k_ma_h128_lr03"]
    arm = CONFIGS["H2_d256_scr5k_ma_h128_lr03_mup"]
    assert arm.readout_score_scale == 32 / 128
    rebuilt = _reset(arm, name=base.name, readout_score_scale=1.0)
    assert rebuilt == base


def test_ma_clip60k_arm_is_the_screen_base_with_clip_the_only_change():
    """d-scaled clip arm (2026-08-18): the MA screen base with the clip
    threshold the ONLY change. 60,000 = 500 x the MEASURED early-median
    grad-norm ratio d256/d64 (91-131x from the archived logs; the printed
    pair-count heuristic's 16.2x is refuted by the same logs — an 8,095
    threshold would still bind on ~100% of early steps)."""
    base = CONFIGS["H2_d256_scr5k_ma"]
    arm = CONFIGS["H2_d256_scr5k_ma_clip60k"]
    assert arm.train.grad_clip_max_norm == 60_000.0
    rebuilt = _reset(arm, name=base.name, train={"grad_clip_max_norm": 500.0})
    assert rebuilt == base


def test_ma_screen_base_is_the_rescue_recipe_at_flat_sigma010():
    """The MA screen base must be the archived naive-rescue cell with only
    the screen's frame changed (flat sigma=0.10, 5k horizon, screen eval
    sizing). This is what licenses reading the base's expected 0.119
    stage-tail FVU off the rescue's own first 5,000 steps — same model,
    same grid, same batch, same clip, same estimator, same seed."""
    rescue = CONFIGS["H2_d256_c50_s223_letf_ma_50k_curr_naive"]
    screen = CONFIGS["H2_d256_scr5k_ma"]
    rebuilt = replace(
        screen,
        name=rescue.name,
        ising=replace(screen.ising, sigma=rescue.ising.sigma),
        curriculum=rescue.curriculum,
        train=replace(screen.train, n_steps=rescue.train.n_steps),
        eval=replace(
            screen.eval,
            n_eval_samples=rescue.eval.n_eval_samples,
            eval_sample_chunk=rescue.eval.eval_sample_chunk,
        ),
    )
    assert rebuilt == rescue


def test_d144_bracket_is_the_rescue_recipe_with_volume_the_only_mechanism_change():
    """The 12x12 bracket exists to attribute to VOLUME alone, so it must be
    the archived 16x16 naive-rescue recipe with D the only mechanism
    change; the two eval-sizing fields (chunk, in-training draw count)
    are eval-only and cannot move the trained model."""
    rescue = CONFIGS["H2_d256_c50_s223_letf_ma_50k_curr_naive"]
    bracket = CONFIGS["H2_d144_c50_s223_letf_ma_50k_curr_naive"]
    assert bracket.ising.D == 12
    rebuilt = replace(
        bracket,
        name=rescue.name,
        ising=replace(bracket.ising, D=rescue.ising.D),
        eval=replace(
            bracket.eval,
            eval_sample_chunk=rescue.eval.eval_sample_chunk,
            n_eval_samples_training=rescue.eval.n_eval_samples_training,
        ),
    )
    assert rebuilt == rescue


def test_d256_fmo2_ladder_is_the_rescue_recipe_with_head_family_the_only_mechanism_change():
    """The definitive cold fmo2 ladder at 16x16 (2026-08-18) must be the
    archived MA naive-rescue recipe with the head family — and its two
    riding conventions, EMA shadow and dual site orderings — the only
    mechanism change, plus the eval chunk sized to the factorised head's
    measured memory (eval-only, cannot move the trained model). This is
    what licenses charging any difference from the rescue's archived
    Var[log w]/site 0.0707 / ESS/N 0.0031 to the head family alone."""
    rescue = CONFIGS["H2_d256_c50_s223_letf_ma_50k_curr_naive"]
    ladder = CONFIGS["H2_d256_c50_s223_letf_fmo2_50k_curr_naive"]
    assert ladder.head_kind == "factorised"
    assert ladder.estimator == "naive_mc"
    rebuilt = replace(
        ladder,
        name=rescue.name,
        head_kind=rescue.head_kind,
        ema_decay=rescue.ema_decay,
        site_orderings=rescue.site_orderings,
        eval=replace(
            ladder.eval, eval_sample_chunk=rescue.eval.eval_sample_chunk
        ),
    )
    assert rebuilt == rescue


RECIPE_ANCHOR = "H2_d256_c50_s223_letf_fmo2_50k_curr_naive"

# recipe cell -> the reset that must reproduce the ladder anchor exactly.
# The naive recipe's microbatch reset rides for the same reason as the
# b512 screen arm's (noise-scale instrument, gradient-exact); the cv
# recipe deliberately has no microbatch (see its registry comment).
RECIPE_DECLARATIONS = {
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive": dict(
        ctmc={"n_euler_steps": 128},
        train={"batch_size": 128, "loss_microbatch_size": None},
    ),
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_cv": dict(
        ctmc={"n_euler_steps": 128},
        train={"batch_size": 128},
        estimator="naive_mc",
    ),
}


@pytest.mark.parametrize("recipe_name", sorted(RECIPE_DECLARATIONS))
def test_recipe_cell_mirrors_the_ladder_anchor_except_declared_fields(recipe_name):
    """The composed recipe cells are read against the fmo2 ladder anchor
    (and through it, the archived MA rescue), so each must be the anchor
    with ONLY its declared variance-bundle fields changed — otherwise the
    anchor-vs-recipe comparison stops isolating the bundle."""
    anchor = CONFIGS[RECIPE_ANCHOR]
    recipe = CONFIGS[recipe_name]
    rebuilt = _reset(recipe, name=anchor.name, **RECIPE_DECLARATIONS[recipe_name])
    assert rebuilt == anchor


def test_clip2000_continuation_mirrors_cv2_continuation_except_declared_fields():
    """The clip continuation reuses the archived phase-2 continuation shape
    (flat sigma_c, lr 3e-4, EMA shadow riding, --init-from the rescue
    final checkpoint at launch) with the declared differences: estimator
    stays naive so the optimiser regime continues rather than switching
    (the cv2 cell's estimator switch is exactly why it cannot serve as a
    clip control), the clip moves 500 -> 2000 as the mechanism change,
    10k not 20k, and the eval chunk rides at 128 (eval-only)."""
    cv2 = CONFIGS["H2_d256_c50_s223_letf_ma_20k_sc_cv2"]
    cont = CONFIGS["H2_d256_c50_s223_letf_ma_10k_sc_clip2000"]
    assert cont.estimator == "naive_mc"
    assert cont.train.grad_clip_max_norm == 2_000.0
    rebuilt = _reset(
        cont,
        name=cv2.name,
        estimator=cv2.estimator,
        train={
            "grad_clip_max_norm": cv2.train.grad_clip_max_norm,
            "n_steps": cv2.train.n_steps,
        },
    )
    rebuilt = replace(
        rebuilt,
        eval=replace(
            rebuilt.eval, eval_sample_chunk=cv2.eval.eval_sample_chunk
        ),
    )
    assert rebuilt == cv2
