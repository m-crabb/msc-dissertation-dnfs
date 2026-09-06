"""Preemption-resume plumbing shared by the flip and swap trainers.

Both trainers face the same failure: a preempted (or budget-capped) container
restarts the function with identical inputs, and without a resume checkpoint
that restart is a run from step 0. The 2026-07-23 Modal GPU recall cost the
MO 100k run its whole trajectory this way; the 2026-08-21 N11 matched-base
family lost eight runs at ~94% of a 50k budget to the same gap.

Each trainer assembles its own checkpoint payload. This module handles
atomic writes, loading, RNG state and log truncation.

Save only at outer-cycle boundaries, when the replay buffer and c_t grid
are consistent.
"""
import csv
from pathlib import Path

import torch


def save_resume_state(ckpt_dir: Path, state: dict) -> None:
    """Write `state` to `ckpt_dir/resume.pt` atomically.

    Write to a temporary file in the same directory, then atomically replace
    the checkpoint so an interrupted write leaves the previous file intact.
    """
    tmp_path = ckpt_dir / "resume.pt.tmp"
    torch.save(state, tmp_path)
    tmp_path.replace(ckpt_dir / "resume.pt")


def load_resume_state(ckpt_dir: Path, map_location) -> dict | None:
    """Load `ckpt_dir/resume.pt`, or None when there is nothing to resume.

    `map_location` selects the device on which tensors are restored.
    """
    resume_path = ckpt_dir / "resume.pt"
    if not resume_path.exists():
        return None
    return torch.load(resume_path, map_location=map_location, weights_only=True)


def capture_rng_state() -> dict:
    """Snapshot Torch CPU and all CUDA RNG streams for exact continuation."""
    return {
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict) -> None:
    """Restore the RNG streams saved by `capture_rng_state`.

    Call after model reconstruction and other setup that consumes randomness
    so the next training draw matches the uninterrupted run.
    """
    torch.set_rng_state(state["rng_cpu"].cpu())
    if state["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in state["rng_cuda"]]
        )


def truncate_log_to_step(log_path: Path, resume_step: int) -> bool:
    """Drop training-log rows at/after `resume_step`; True if the log survives.

    Rows written after the checkpoint describe steps that will be repeated
    on resume. Remove them to avoid duplicate observations in later analysis.
    Return False if the log is missing so the caller can write a fresh header.
    """
    if not log_path.exists():
        return False
    with log_path.open(newline="") as log_file:
        rows = list(csv.reader(log_file))
    header, body = rows[0], rows[1:]
    kept = [row for row in body if int(row[0]) < resume_step]
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(header)
        writer.writerows(kept)
    return True
