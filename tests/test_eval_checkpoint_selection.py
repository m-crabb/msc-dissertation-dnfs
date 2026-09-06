"""Which checkpoint an eval-only pass reads, and where it writes.

The pairing is the whole point. A draw from the wrong weights returns a
plausible number and nothing in the artefacts records which file produced
it, so every checkpoint choice must also change the output directory --
otherwise a re-draw can silently overwrite a frozen number with one
computed from different weights.

`stage_best` is the instrument the rw cells are read with: whether
the sigma_c stage's best-trailing-median-ESS checkpoint beats `final.pt`.
The trainer saves those as RAW weights only (swap_training saves
`head.state_dict()`, not the EMA shadow), so the EMA pairing must be
refused rather than quietly reading `final_ema.pt`.
"""

import pytest
from experiments.constrained_hard_03.run import _eval_checkpoint_and_suffix


def test_default_reads_final_into_the_canonical_dir():
    assert _eval_checkpoint_and_suffix(False, None, None) == ("final.pt", "")


def test_ema_and_grid_override_compose():
    assert _eval_checkpoint_and_suffix(True, 512, None) == (
        "final_ema.pt",
        "_ema_ne512",
    )


def test_stage_best_names_its_checkpoint_and_its_own_dir():
    assert _eval_checkpoint_and_suffix(False, None, 6) == ("best_stage6.pt", "_stage6")


def test_stage_best_composes_with_a_grid_override():
    assert _eval_checkpoint_and_suffix(False, 128, 6) == (
        "best_stage6.pt",
        "_stage6_ne128",
    )


def test_stage_best_with_ema_is_refused():
    # No EMA twin is written per stage, so this could only ever be served by
    # silently reading a different run's-end checkpoint.
    with pytest.raises(ValueError, match="no EMA"):
        _eval_checkpoint_and_suffix(True, 512, 6)


def test_stage_index_must_be_non_negative():
    with pytest.raises(ValueError):
        _eval_checkpoint_and_suffix(False, None, -1)
