"""Run-dir tagging on the shared baseline/soft `train` entry point.

A Slurm job resubmitted after preemption re-invokes `train` from scratch.
With the default timestamp suffix every invocation mints a NEW run dir, so a
retried job leaves a sibling directory, and tooling that resolves run dirs
by recency (e.g. the sweep step in the amortised sbatch) can silently pick
the wrong one. A caller-fixed `tag` — mirroring the hard experiment's
runner — gives the run a stable directory identity: the retry lands in the
SAME dir, and a run that already finished (eval/metrics.json present) is
detected and skipped rather than retrained.

Scope: this file covers the DIRECTORY-identity half of that contract only.
Mid-run continuation from `checkpoints/resume.pt` (added 2026-08-22) is a
separate contract with its own file, `tests/test_training_resume.py` — but
the two compose, and the tag is what makes the resume reachable at all: a
retry that mints a fresh dir never sees the previous attempt's checkpoint.
"""

import sys

from experiments.dnfs_baseline_01.configs import CONFIGS as BASELINE_CONFIGS
from experiments.dnfs_baseline_01.run import train


def test_fixed_tag_short_circuits_completed_run(tmp_path):
    """A completed tagged run must return immediately, before any artefact
    is (re)written — resubmission after preemption must be idempotent."""
    cfg = BASELINE_CONFIGS["stage_0_d4"]
    finished = tmp_path / "stage_0_d4_seed42_sometag" / "eval"
    finished.mkdir(parents=True)
    (finished / "metrics.json").write_text("{}")

    run_dir = train(cfg, seed=42, output_dir=tmp_path, use_wandb=False, tag="sometag")

    assert run_dir == tmp_path / "stage_0_d4_seed42_sometag"
    # The short-circuit must fire before config.json is rewritten: on a
    # resumed attempt the original file is the record of what the run was.
    assert not (run_dir / "config.json").exists()


def test_baseline_cli_passes_tag_through(monkeypatch):
    import experiments.dnfs_baseline_01.run as baseline_run

    seen = {}
    monkeypatch.setattr(
        baseline_run, "train", lambda cfg, **kwargs: seen.update(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--cfg",
            "stage_4_d8_critical_paper_curriculum",
            "--tag",
            "20260812-walkback",
            "--no-wandb",
        ],
    )
    baseline_run.main()
    assert seen["tag"] == "20260812-walkback"


def test_soft_cli_passes_tag_through(monkeypatch):
    # Using a walk-back cell name doubles as a check that the new d8 soft
    # cells are registered in the CLI's --cfg choices.
    import experiments.constrained_soft_02.run as soft_run

    seen = {}
    monkeypatch.setattr(soft_run, "train", lambda cfg, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--cfg",
            "S2_d8_c05_l50_letf_ne64",
            "--tag",
            "20260812-walkback",
            "--no-wandb",
        ],
    )
    soft_run.main()
    assert seen["tag"] == "20260812-walkback"
