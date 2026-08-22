"""Preemption-resume plumbing shared by the flip and swap trainers.

Both trainers face the same failure: a preempted (or budget-capped) container
restarts the function with identical inputs, and without a resume checkpoint
that restart is a run from step 0. The 2026-07-23 Modal GPU recall cost the
MO 100k run its whole trajectory this way; the 2026-08-21 N11 matched-base
family lost eight runs at ~94% of a 50k budget to the same gap.

What lives here is the part that is genuinely identical between the two
trainers: the atomic write, the load, the RNG capture/restore pair, and the
log truncation. What does NOT live here is the payload itself — the flip
trainer retains four replay-chunk lists (states, time indices, per-state
composition, per-state c_t baseline) and both a sigma and a lambda replay
tag, while the swap trainer retains two lists plus an EMA shadow and a c_t
grid EMA. Assembling that dict is each trainer's own business; getting it to
disk safely is not.

The invariant every caller depends on: a resume checkpoint is only valid at
an OUTER-cycle boundary. Mid-cycle the replay buffer and the c_t grid are
half-rebuilt, and there is no consistent state to serialise.
"""
import csv
from pathlib import Path

import torch


def save_resume_state(ckpt_dir: Path, state: dict) -> None:
    """Write `state` to `ckpt_dir/resume.pt` atomically.

    Written to a temp file and renamed rather than in place: the whole point
    of this file is to survive a process that dies without warning, and a
    preemption landing mid-`torch.save` would otherwise leave a truncated
    resume.pt that fails to load — turning a recoverable interruption into
    the from-scratch restart the checkpoint exists to prevent. `Path.replace`
    is atomic within a filesystem, so a reader ever sees the old complete
    file or the new complete file, never a partial one.
    """
    tmp_path = ckpt_dir / "resume.pt.tmp"
    torch.save(state, tmp_path)
    tmp_path.replace(ckpt_dir / "resume.pt")


def load_resume_state(ckpt_dir: Path, map_location) -> dict | None:
    """Load `ckpt_dir/resume.pt`, or None when there is nothing to resume.

    `map_location` lets a checkpoint written on one device be restored on
    another — replay chunks are stored on CPU precisely so a run preempted
    off one GPU can continue on the next one it is scheduled onto.
    """
    resume_path = ckpt_dir / "resume.pt"
    if not resume_path.exists():
        return None
    return torch.load(resume_path, map_location=map_location, weights_only=True)


def capture_rng_state() -> dict:
    """Snapshot the torch CPU + all-device CUDA RNG streams.

    Without this the continuation diverges from the uninterrupted run at the
    first random draw even with identical weights and optimiser moments, and
    the resumed run is a *different* run wearing the same directory name.
    `get_rng_state_all` rather than `get_rng_state` so a multi-device host
    restores every generator the run might touch.
    """
    return {
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict) -> None:
    """Restore what `capture_rng_state` saved.

    Call this LAST in a restore sequence: anything that draws randomness
    after it — rebuilding a model, re-seeding, a diagnostic — consumes the
    stream the continuation is supposed to consume, and silently offsets it.
    """
    torch.set_rng_state(state["rng_cpu"].cpu())
    if state["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in state["rng_cuda"]]
        )


def truncate_log_to_step(log_path: Path, resume_step: int) -> bool:
    """Drop training-log rows at/after `resume_step`; True if the log survives.

    A dead attempt flushes a CSV row every inner step but only checkpoints
    every `resume_every_outer` cycles, so between the last checkpoint and the
    kill there are orphan rows describing steps the resumed run is about to
    redo. Left in place they would make those steps appear twice, and any
    trailing-window analysis over the log would read the duplicates as real.

    Returns False when the log is missing entirely, which tells the caller to
    open with a fresh header rather than append to nothing.
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
