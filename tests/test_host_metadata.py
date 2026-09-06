"""Host-metadata record written next to each run's artefacts.

The field that matters is `device`. Wall clocks are only comparable across
runs that shared a GPU class: mars is a shared B200 that MPS-shares GPUs
between configs, so its step times cannot be read against a DoC a100's, and
`wall_clock_step_s` is uninterpretable without knowing which one produced it.
The record is appended rather than overwritten because a resumed run can be
requeued onto a different node, and then both hosts are true for different
step ranges of the same directory.
"""

import json

from experiments.dnfs_baseline_01.run import write_host_metadata


def test_write_host_metadata_records_the_device(tmp_path):
    write_host_metadata(tmp_path)

    attempts = json.loads((tmp_path / "metadata.json").read_text())
    assert len(attempts) == 1
    assert set(attempts[0]) == {"torch_version", "hostname", "platform", "device"}
    assert attempts[0]["device"]  # non-empty on CPU ("cpu") and on GPU alike


def test_write_host_metadata_appends_on_resume(tmp_path):
    write_host_metadata(tmp_path)
    write_host_metadata(tmp_path)

    attempts = json.loads((tmp_path / "metadata.json").read_text())
    assert len(attempts) == 2, "a resumed attempt must not erase the first host"
