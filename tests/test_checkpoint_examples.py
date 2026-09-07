"""Exercise the shipped inference path using the actual small checkpoint files."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.sample_checkpoint import load_bundle, main, sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family", ["baseline", "soft", "hard"])
def test_bundled_sampling(tmp_path, monkeypatch, family):
    bundle = ROOT / "checkpoints" / f"ising_{family}_4x4"
    out = tmp_path / "draw"
    monkeypatch.setattr(
        sys,
        "argv",
        ["sample_checkpoint", str(bundle), "--out", str(out), "--n-samples", "8"],
    )
    main()
    samples = torch.load(out / "samples.pt", weights_only=True)
    weights = torch.load(out / "log_weights.pt", weights_only=True)
    assert samples.shape == (8, 16)
    assert torch.isin(samples, torch.tensor([-1, 1])).all()
    assert weights.shape == (8,)
    assert torch.isfinite(weights).all()
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["n_intervals"] == (100 if family == "hard" else 49)
    if family == "hard":
        assert ((samples == 1).sum(1) == 8).all()
    original = (out / "samples.pt").read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert (out / "samples.pt").read_bytes() == original


def test_config_checksum_rejected_before_loading(tmp_path):
    source = ROOT / "checkpoints/ising_hard_4x4"
    shutil.copy(source / "manifest.json", tmp_path)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_bundle(tmp_path, "cpu")


def test_hard_registry_drift_still_rejected(tmp_path):
    source = ROOT / "checkpoints/ising_hard_4x4"
    shutil.copytree(source, tmp_path / "bundle")
    bundle = tmp_path / "bundle"
    config = json.loads((bundle / "config.json").read_text())
    config["ising"]["sigma"] = 0.2
    (bundle / "config.json").write_text(json.dumps(config))
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["sha256"]["config.json"] = sha256(bundle / "config.json")
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from registry"):
        load_bundle(bundle, "cpu")


def test_readme_recording_matches_distributed_checkpoint():
    manifest = json.loads(
        (ROOT / "checkpoints/ising_hard_24x24/manifest.json").read_text()
    )
    with np.load(ROOT / "assets/hard_rate_field_strip_24x24.npz") as data:
        metadata = json.loads(str(data["metadata"]))
        assert metadata["source_sha256"] == manifest["sha256"]
        assert metadata["checkpoint"] == "checkpoints/final_ema.pt"
        assert data["states"].shape == (129, 576)
        assert ((data["states"] == 1).sum(1) == 288).all()
        assert np.isin(data["states"], [-1, 1]).all()
        assert np.isfinite(data["rates"]).all()
        assert (data["rates"] >= 0).all()
        assert (data["rates"][:, metadata["anchor"]] == 0).all()
        # Like-spin sites cannot be partners for a composition-preserving swap.
        like_spin = data["states"] == data["states"][:, metadata["anchor"], None]
        assert np.allclose(data["rates"][like_spin], 0)
        np.testing.assert_allclose(data["times"], np.linspace(0, 1, 129))
