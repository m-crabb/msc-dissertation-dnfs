"""Run-dir discovery shared by fc_curve, fc_compare and fc_weighted_thermo.

One definition instead of three identical clones: discovery has real logic
(timestamped vs bare run-dir forms, eval-completeness filter) and a silent
divergence between copies would make two analyses disagree about which run
they scored.
"""

from pathlib import Path


def latest_run_dir(
    results_dir: Path, config: str, seed: int, eval_dir: str = "eval"
) -> Path | None:
    """Newest run dir for (config, seed) carrying an {eval_dir}/metrics.json.

    Matches both the timestamped `{config}_seed{seed}_<timestamp>` form that
    `batch_seeds` writes and the bare `{config}_seed{seed}` form of older
    one-off runs. Lexicographic max is chronological max because the
    timestamp suffix is zero-padded `YYYYMMDD-HHMMSS`.

    `eval_dir` selects which frozen eval qualifies a run as complete:
    "eval" (raw weights, every run) or "eval_ema" (the dual eval's
    shadow-weight draw, present only on ema_decay > 0 cells) — so an
    EMA-selected analysis can never silently score a raw draw.
    """
    matches = set(results_dir.glob(f"{config}_seed{seed}_*"))
    bare = results_dir / f"{config}_seed{seed}"
    if bare.exists():
        matches.add(bare)
    matches = sorted(m for m in matches if (m / eval_dir / "metrics.json").exists())
    return matches[-1] if matches else None


def seed_of(run_dir_name: str) -> str:
    """The seed token out of a `{config}_seed{seed}[_timestamp]` dir name."""
    return run_dir_name.split("_seed")[1].split("_")[0]


# The revamp request grid (2026-08-30): specialists {0.25, 0.375, 0.50}
# plus Z2 mirrors {0.625, 0.75} and the held-outs, every value a multiple
# of 1/16 (integer site counts at d=16). The fit depends mildly on the
# grid, so wave-3 model slopes are scored against THIS grid's exact
# reference (0.9950 at lambda=50), never the archived 0.976 — one
# definition here so the reference derivation (16) and the model slope
# fit (15) can never disagree about the grid.
REVAMP_GRID = (
    0.25,
    0.3125,
    0.375,
    0.4375,
    0.50,
    0.5625,
    0.625,
    0.6875,
    0.75,
)
