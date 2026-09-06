"""Run the appendix's frozen-checkpoint redraws on up to four Modal GPUs.

Local inputs carry the exact configs and any locally available checkpoints.
The remaining checkpoints are read from the existing dnfs-results volume.
Only the new calibration directory is written; archived evaluations are untouched.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import modal
from experiments.dnfs_baseline_01.modal_app import image, volume

from scripts.configuration_calibration_4x4 import OUTPUT, ROOT

REMOTE_OUTPUT = "/results/configuration_calibration_4x4_20260906"
# Reuse a built locked environment for repeat launches; fresh scripts are mounted
# after the cached image so image reuse cannot select an obsolete evaluator.
cached_image = os.environ.get("DNFS_CALIBRATION_IMAGE")
base_image = modal.Image.from_id(cached_image) if cached_image else image
calibration_image = base_image.add_local_dir(
    ROOT / "scripts", "/repo/scripts"
).add_local_dir(OUTPUT / "inputs", "/calibration_inputs")
app = modal.App("dnfs-configuration-calibration-4x4", image=calibration_image)


def checkpoint_for(task):
    staged = Path("/calibration_inputs") / task["run"]
    checkpoint = staged / "checkpoints/final.pt"
    if checkpoint.exists():
        return checkpoint
    for run in (
        Path("/results") / task["run"],
        Path("/results") / task["group"] / task["run"],
    ):
        checkpoint = run / "checkpoints/final.pt"
        if checkpoint.exists():
            local_config = json.loads((staged / "config.json").read_text())
            volume_config = json.loads((run / "config.json").read_text())
            if local_config != volume_config:
                raise ValueError(f"local/Modal config mismatch: {task['run']}")
            return checkpoint
    raise FileNotFoundError(f"archived final.pt missing: {task['run']}")


@app.function(volumes={"/results": volume}, timeout=120)
def preflight(tasks):
    return [str(checkpoint_for(task)) for task in tasks]


@app.function(
    gpu="A100-80GB", volumes={"/results": volume}, timeout=3600, max_containers=4
)
def sample_remote(task):
    from scripts.configuration_calibration_4x4 import sample_task

    checkpoint = checkpoint_for(task)
    prefix = Path(REMOTE_OUTPUT) / task["run"]
    metadata_path = prefix.with_suffix(".json")
    if metadata_path.exists() and prefix.with_suffix(".npz").exists():
        previous = json.loads(metadata_path.read_text())
        config = Path("/calibration_inputs") / task["run"] / "config.json"
        if (
            all(previous.get(key) == value for key, value in task.items())
            and previous["checkpoint_sha256"]
            == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            and previous["config_sha256"]
            == hashlib.sha256(config.read_bytes()).hexdigest()
        ):
            print(f"Reusing completed histogram: {task['run']}", flush=True)
            return {
                "metadata": previous,
                "histogram": prefix.with_suffix(".npz").read_bytes(),
            }
    metadata = sample_task(
        task, "/calibration_inputs", REMOTE_OUTPUT, checkpoint=checkpoint
    )
    volume.commit()
    histogram = (Path(REMOTE_OUTPUT) / (task["run"] + ".npz")).read_bytes()
    return {"metadata": metadata, "histogram": histogram}


@app.function(volumes={"/results": volume}, timeout=4 * 60 * 60)
def finish_remote(tasks):
    """Complete sampling and plotting remotely after the local CLI closes."""
    from scripts.plot_configuration_calibration_4x4 import plot

    for task in tasks:
        checkpoint_for(task)
    remote = Path(REMOTE_OUTPUT)
    remote.mkdir(parents=True, exist_ok=True)
    (remote / "manifest.json").write_text(json.dumps(tasks, indent=2) + "\n")
    volume.commit()
    # Collect remotely so local disconnection cannot interrupt the final plot.
    stage = Path(tempfile.mkdtemp(prefix="calibration-report-"))
    (stage / "counts").mkdir()
    (stage / "inputs").symlink_to("/calibration_inputs", target_is_directory=True)
    (stage / "manifest.json").write_text(json.dumps(tasks, indent=2) + "\n")
    for result in sample_remote.map(tasks, order_outputs=False):
        metadata = result["metadata"]
        prefix = stage / "counts" / metadata["run"]
        prefix.with_suffix(".npz").write_bytes(result["histogram"])
        prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {metadata['panel']} seed {metadata['training_seed']}", flush=True)
    plot(stage)
    volume.reload()
    for name in (
        "configuration_calibration_4x4.pdf",
        "configuration_calibration_4x4.png",
        "summary.json",
    ):
        shutil.copy2(stage / name, remote / name)
    (remote / "COMPLETE.json").write_text(
        json.dumps(
            {
                "checkpoints": len(tasks),
                "draws": sum(t["n_samples"] for t in tasks),
                "figure": "configuration_calibration_4x4.pdf",
            },
            indent=2,
        )
        + "\n"
    )
    volume.commit()
    return str(remote)


@app.local_entrypoint()
def background():
    """Launch with modal run --detach; all remaining work executes remotely."""
    tasks = json.loads((OUTPUT / "manifest.json").read_text())
    call = finish_remote.spawn(tasks)
    record = {
        "function_call_id": call.object_id,
        "remote_output": REMOTE_OUTPUT,
        "n_checkpoints": len(tasks),
        "n_samples_per_checkpoint": tasks[0]["n_samples"],
    }
    (OUTPUT / "modal_background.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


@app.local_entrypoint()
def main(smoke: bool = False):
    tasks = json.loads((OUTPUT / "manifest.json").read_text())
    if smoke:
        tasks = [{**tasks[i], "n_samples": 10000} for i in (0, 8, 14)]
    paths = preflight.remote(tasks)
    print(f"Located all {len(paths)} frozen checkpoints", flush=True)
    destination = OUTPUT / ("smoke" if smoke else "counts")
    destination.mkdir(parents=True, exist_ok=True)
    for result in sample_remote.map(tasks, order_outputs=False):
        metadata = result["metadata"]
        prefix = destination / metadata["run"]
        prefix.with_suffix(".npz").write_bytes(result["histogram"])
        prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"Completed {metadata['panel']} seed {metadata['training_seed']}: "
            f"{metadata['n_samples']:,} draws in {metadata['seconds']:.1f}s",
            flush=True,
        )
