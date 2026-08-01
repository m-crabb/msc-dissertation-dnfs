"""Run-dir discovery shared by the F(c) analysis scripts (06/08/09).

One definition instead of three identical clones: discovery has real logic
(timestamped vs bare run-dir forms, eval-completeness filter) and a silent
divergence between copies would make two analyses disagree about which run
they scored.
"""

from pathlib import Path


def latest_run_dir(results_dir: Path, config: str, seed: int) -> Path | None:
    """Newest run dir for (config, seed) carrying an eval/metrics.json.

    Matches both the timestamped `{config}_seed{seed}_<timestamp>` form that
    `batch_seeds` writes and the bare `{config}_seed{seed}` form of older
    one-off runs. Lexicographic max is chronological max because the
    timestamp suffix is zero-padded `YYYYMMDD-HHMMSS`.
    """
    matches = set(results_dir.glob(f"{config}_seed{seed}_*"))
    bare = results_dir / f"{config}_seed{seed}"
    if bare.exists():
        matches.add(bare)
    matches = sorted(m for m in matches if (m / "eval" / "metrics.json").exists())
    return matches[-1] if matches else None


def seed_of(run_dir_name: str) -> str:
    """The seed token out of a `{config}_seed{seed}[_timestamp]` dir name."""
    return run_dir_name.split("_seed")[1].split("_")[0]
