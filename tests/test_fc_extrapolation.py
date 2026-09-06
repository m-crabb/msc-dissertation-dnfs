"""Recover known continuum values from the actual linspace interval lengths."""

import importlib
import json

import numpy as np
import pytest

fc = importlib.import_module("experiments.constrained_soft_02.analysis.fc_compare")


def test_soft_extrapolation_cancels_error_proportional_to_step(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fc, "_grids_available", lambda *args: [(4, "coarse"), (9, "fine")]
    )

    def estimate(run_dir, d, n_boot, rng, eval_dir):
        # Four grid points span three intervals; nine span eight.
        step = {"coarse": 1 / 3, "fine": 1 / 8}[eval_dir]
        return 2.0 + 6.0 * step, np.array([1.0, 3.0]) + 6.0 * step

    point, boot, pair = fc._richardson_F(tmp_path, 4, 1, 2, None, bootstrap=estimate)
    assert point == pytest.approx(2.0)
    np.testing.assert_allclose(boot, [1.0, 3.0])
    assert pair == (4, 9)


def test_hard_extrapolation_cancels_step_error_and_propagates_uncertainty(tmp_path):
    paths = []
    for nodes in (4, 9):
        path = tmp_path / f"grid{nodes}.json"
        path.write_text(
            json.dumps(
                {
                    "n_euler_steps": nodes,
                    "rows": [
                        {
                            "composition": 0.5,
                            "stop_time": 1.0,
                            "free_energy_nats_per_site": 2.0 + 6.0 / (nodes - 1),
                            "var_log_w": 4.0,
                            "n_samples": 100,
                            "ess_fraction": 0.9,
                        }
                    ],
                }
            )
        )
        paths.append(path)
    (row,) = fc._hard_series(paths[:1], paths[1:], 1, lambda c: 2.0)
    assert row["F"] == pytest.approx(2.0)
    # Eliminate first-order error using coefficients -3/5 and 8/5.
    assert row["F_err"] == pytest.approx(0.2 * np.hypot(3 / 5, 8 / 5))
