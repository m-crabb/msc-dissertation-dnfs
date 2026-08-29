"""What correct looks like for the 4x4 house-table registry, written before it.

These are registry tests, not estimator tests: the 4x4 fill's numerical core
(exact enumeration, the sampling floor drawn from the exact pmf) is already
covered by test_house_observable_errors.py, and its reference is exact by
construction rather than estimated. What is NOT covered anywhere is the
lookup that turns an arm name into a run directory, and that lookup is where
this rung has repeatedly gone wrong.

THE FAILURE MODE IS SILENT. The fill resolves an arm to
`{cfg.name}_seed{seed}_{tag}`. Get the tag wrong and the directory is simply
absent; the cell prints `--`, which reads as "not yet run" rather than "run,
sitting on disk, looked for under the wrong name". s84 hit exactly this at
8x8, where the ladder's two couplings came from separate campaigns and an
arm-keyed provenance map sent every floor lookup to the sigma_c tag.

WHY THIS RUNG STAYS ARM-KEYED ANYWAY, which is the point of the second test.
The 8x8 repair was to key provenance by (arm, coupling). That repair is
correct THERE because the two columns were two campaigns. All 30 gate runs
at this rung came from ONE campaign under ONE tag spanning both couplings,
so an arm key names them unambiguously and a (arm, coupling) key would carry
a distinction the data does not have. Recorded as a test so the asymmetry
between the two modules reads as a decision rather than an oversight.
"""


def test_gate_arms_are_enrolled():
    """The five sweep-ladder arms must appear at the 4x4 rung too.

    The gate exists to be read one rung BELOW the 8x8 ladder, so an arm
    present at 8x8 and missing here silently narrows the comparison rather
    than failing it.
    """
    from experiments.constrained_hard_03.analysis.house_table_4x4 import ARMS

    for arm in ("mamo2", "mamo2ef", "iv", "ivmo2", "ivmo2ef"):
        assert arm in ARMS, arm


def test_gate_provenance_is_arm_keyed_because_one_campaign_ran_both():
    """One tag spans both couplings here, unlike the 8x8 ladder.

    Guards the symmetry-seeking edit: a reader who has just seen
    house_table_8x8.ARM_PROVENANCE keyed by (arm, coupling) may re-key this
    map to match. That would encode a two-campaign split that never
    happened at this rung.
    """
    from experiments.constrained_hard_03.analysis.house_table_4x4 import (
        ARM_PROVENANCE)

    for arm in ("mamo2", "mamo2ef", "iv", "ivmo2", "ivmo2ef"):
        assert ARM_PROVENANCE[arm] == ("w2", "20260828-rasterord-d16")
    assert all(isinstance(key, str) for key in ARM_PROVENANCE)


def test_every_registered_arm_names_a_real_config():
    """A tag typo or renamed arm would otherwise surface as a permanently
    blank row rather than an error.

    Mirrors test_house_table_8x8's check of the same name; the two rungs
    resolve config names by different rules (a recipe suffix here, a
    per-coupling CELL_NAME template there) and both rules can drift.
    """
    from experiments.constrained_hard_03.analysis.house_table_4x4 import (
        ARMS, ARM_PROVENANCE, SIGMA_LABELS)
    from experiments.constrained_hard_03.configs import CONFIGS

    # The eager-refill branch left with the factorised arms (2026-08-29): the
    # `_w2e` twins existed only for them, so every surviving arm resolves to
    # the plain wave-2 suffix or to its own ARM_PROVENANCE entry.
    for arm in ARMS:
        for sigma_label in SIGMA_LABELS:
            suffix = ARM_PROVENANCE[arm][0] if arm in ARM_PROVENANCE else "w2"
            name = f"H2_d16_c50_{sigma_label}_letf_{arm}_10k_{suffix}"
            assert name in CONFIGS, name


def test_oracle_is_enrolled_with_its_own_campaign_tag():
    """The oracle row fills from the 2026-08-29 relaunch, not the wave-2 tag.

    Its six runs went out as single-run DoC jobs under their own tag (the
    first launch died compiling, the second was shaped to blow the 12 h
    wall), so the arm needs an ARM_PROVENANCE entry: resolving it through
    the wave-2 default tag would look up directories that never existed
    and print a permanently blank row -- the silent failure mode this
    file's docstring describes. The recipe suffix stays `w2` because the
    CONFIG is a wave-2 cell; only the campaign tag is its own.
    """
    from experiments.constrained_hard_03.analysis.house_table_4x4 import (
        ARMS, ARM_PROVENANCE)

    assert "dh" in ARMS
    assert ARM_PROVENANCE["dh"] == ("w2", "20260829-dh-oracle-d16")


def test_oracle_exists_at_4x4_and_never_at_8x8():
    """The doubly-hollow oracle is a 4x4-only cell, and that is structural.

    `_WAVE2_ARM_KNOBS` feeds the d64 critical and floor builders as well as
    the d16 one, so an arm registered there acquires an 8x8 cell for free.
    The oracle must not: it benches at 7,036 ms per forward at d=64 against
    the masked-attention head's 6.0 ms (~1,170x), which is why
    tab:eval-hard-8x8 carries no oracle row. Registering it in
    `_D16_ONLY_ARM_KNOBS` is what keeps that true, and this test is what
    stops a later tidy-up from folding the two dicts together.
    """
    from experiments.constrained_hard_03.configs import CONFIGS

    for sigma_label in ("s010", "s220"):
        assert f"H2_d16_c50_{sigma_label}_letf_dh_10k_w2" in CONFIGS
    assert not [n for n in CONFIGS if "d64" in n and "_dh" in n]


def test_oracle_deviates_from_the_mask_one_row_in_head_and_compile_only():
    """The oracle prices the O(d^2) gate, so only FORCED knobs may differ.

    Two deviations, and the second is not a preference. `compile_head=False`
    is there because a COMPILED oracle cell cannot be run at all: measured
    2026-08-29 (DoC 280196, cancelled), `doubly_hollow` x `compile_head=True`
    spent 26 minutes without reaching step 1 on a 4x4 cell, pinned at ~95% of
    one core with 373 MiB on the GPU -- inductor still compiling a graph that
    unrolls to d^2 = 256 masked pair forwards. `c_t_from_rollout` is KEPT, the
    same knob-for-knob convention as the decision-(c) `_w2e` cells.

    The test exists to stop a THIRD deviation drifting in unnoticed, which
    would quietly make the row unreadable against the mask-one row. The legacy
    `H2_d16_c50_s223_letf_dh` cells fail this twice over -- pre-migration
    sigma 0.223 rather than SIGMA_C, and the pre-s60 recipe -- which is why
    they are not the cells the table fills from.
    """
    from dataclasses import asdict

    from experiments.constrained_hard_03.configs import CONFIGS

    for sigma_label in ("s010", "s220"):
        oracle = asdict(CONFIGS[f"H2_d16_c50_{sigma_label}_letf_dh_10k_w2"])
        mask_one = asdict(CONFIGS[f"H2_d16_c50_{sigma_label}_letf_mo_10k_w2"])
        deviated = {k for k in oracle if oracle[k] != mask_one[k]}
        assert deviated == {"name", "head_kind", "compile_head"}, deviated
        assert oracle["compile_head"] is False
        # c_t_from_rollout is exonerated and stays on, as in the _w2e cells.
        assert oracle["train"]["c_t_from_rollout"] is True
