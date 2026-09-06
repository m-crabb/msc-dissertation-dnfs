"""Figure export regressions using small synthetic inputs."""

import importlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "module_name",
    [
        "experiments.dnfs_baseline_01.analysis.10_logp_scatter_4x4",
        "experiments.constrained_soft_02.analysis.23_logp_scatter_4x4",
    ],
)
def test_archived_scatter_command_saves_and_returns(module_name, monkeypatch, tmp_path):
    module = importlib.import_module(module_name)
    panels = [
        {
            "title": title,
            "lims": (-3.0, 0.0),
            "xlabel": "exact log probability",
            "series": [("synthetic", "tab:purple", [-2.0, -1.0], [-2.1, -0.9])],
        }
        for title in ("first panel", "second panel")
    ]
    monkeypatch.setattr(module, "panel_series", lambda results_dir: panels)
    output = tmp_path / "figures" / "scatter.png"

    with plt.rc_context():
        try:
            module.main(["--results-dir", str(tmp_path), "--out", str(output)])
            pixels = plt.imread(output)
            assert pixels.size > 0
            assert np.ptp(pixels[..., :3]) > 0
        finally:
            plt.close("all")


@pytest.mark.parametrize("layout", ["2x2", "row"])
def test_multi_head_results_cell_exports_both_couplings(
    layout, monkeypatch, tmp_path, capsys
):
    from experiments.constrained_hard_03.analysis import hard_results_cell as module

    generator = torch.Generator().manual_seed(0)
    base = torch.cat([torch.ones(32), -torch.ones(32)])
    samples = torch.stack(
        [base[torch.randperm(64, generator=generator)] for _ in range(32)]
    )
    monkeypatch.setattr(module, "load_reference", lambda *args: list(samples.chunk(4)))

    def load_cells(results_dir, lattice_edge, sigma_key, head, eval_subdir):
        return [
            {
                "name": f"{head}_{sigma_key}",
                "samples": samples,
                "weights": torch.full((len(samples),), 1.0 / len(samples)),
            }
        ]

    monkeypatch.setattr(module, "load_cells", load_cells)
    output = tmp_path / "multi_head.png"

    with plt.rc_context():
        try:
            module.build(
                tmp_path,
                8,
                ["thp", "ma"],
                "eval_ema",
                output,
                n_replicates=2,
                layout=layout,
            )
            pixels = plt.imread(output)
            assert pixels.size > 0
            assert np.ptp(pixels[..., :3]) > 0
            summary = json.loads(capsys.readouterr().out)
            for coupling in ("s010", "s220"):
                assert summary[coupling]["n_draws"] == len(samples)
                assert len(summary[coupling]["heads"]) == 2
        finally:
            plt.close("all")
