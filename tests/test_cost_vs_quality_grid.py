"""Cost joining for the DNFS-vs-mchammer wall-clock grid.

This module turns two differently-derived ESS numbers into one comparable
cost, which makes it the place a wrong number enters the headline efficiency
claim. The tests pin the three ways that can happen:

  * an untimed run silently costing zero -- most archived runs predate
    `eval_draw_seconds`, and a default would fabricate the very quantity the
    grid exists to report;
  * the cost formula drifting from seconds/ESS to seconds/n_samples, which
    would flatter whichever side has the worse ESS;
  * pairing DNFS seed 42 with mchammer seed 42 as though they were the same
    experiment, when the two sides share no seed axis at all.

Fixtures are JSON only: the module reads config.json, metrics.json,
composition_sweep.json and mchammer summary.json, and never loads a model.
"""
import importlib
import json

import pytest

grid = importlib.import_module(
    "experiments.constrained_soft_02.analysis.cost_vs_quality"
)


def _write_dnfs_run(
    results_dir, name, *, seed=42, D=10, composition=0.5,
    ess_fraction=0.5, n_eval=1000, draw_seconds=None, sweep=None,
):
    run_dir = results_dir / name
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "name": name.split("_seed")[0],
        "ising": {"D": D, "target_composition": composition},
        "train": {"seed": seed},
    }))

    def _metrics(comp, ess_frac):
        payload = {
            "ess_fraction": ess_frac,
            "ess": ess_frac * n_eval,
            "n_eval_samples": n_eval,
            "eval_device": "cuda",
            "composition": comp,
        }
        if draw_seconds is not None:
            payload["eval_draw_seconds"] = draw_seconds
        return payload

    if sweep is not None:
        (run_dir / "eval" / "composition_sweep.json").write_text(
            json.dumps([_metrics(comp, frac) for comp, frac in sweep])
        )
    else:
        (run_dir / "eval" / "metrics.json").write_text(
            json.dumps(_metrics(composition, ess_fraction))
        )
    return run_dir


def _write_mchammer(baseline_dir, name, *, D=10, composition=0.5, seed=42,
                    s_per_eff=0.002, potential_s_per_eff=0.004):
    cell = baseline_dir / name
    cell.mkdir(parents=True)
    (cell / "summary.json").write_text(json.dumps({
        "D": D,
        "seed": seed,
        "target_composition": composition,
        "steps_per_second": 50_000.0,
        "hostname": "test-host",
        "observables": {
            "composition": {"seconds_per_effective_sample": s_per_eff},
            "potential": {"seconds_per_effective_sample": potential_s_per_eff},
        },
    }))


def test_untimed_runs_are_dropped_not_costed_as_zero(tmp_path):
    """A run without `eval_draw_seconds` must not reach the grid at all.

    Archived specialists predate the timing fields. Defaulting them to zero
    would make the reference look infinitely slower than a sampler that was
    never actually timed.
    """
    _write_dnfs_run(tmp_path, "timed_seed42", draw_seconds=4.0)
    _write_dnfs_run(tmp_path, "untimed_seed43", draw_seconds=None)

    collected = grid.collect_dnfs(tmp_path, D=10)

    assert list(collected["seed"]) == [42]
    assert collected["dnfs_s_per_eff"].notna().all()


def test_cost_is_seconds_per_effective_sample_not_per_drawn_sample(tmp_path):
    """seconds / ESS, so a batch worth half its size costs twice as much.

    Dividing by `n_eval_samples` instead would report identical cost for a
    sampler with ESS 0.9 and one with ESS 0.1 -- erasing the entire quality
    axis the grid is built to expose.
    """
    _write_dnfs_run(
        tmp_path, "cell_seed42",
        draw_seconds=10.0, ess_fraction=0.5, n_eval=1000,
    )

    collected = grid.collect_dnfs(tmp_path, D=10)

    # ESS = 0.5 * 1000 = 500 effective samples for 10 s of drawing.
    assert collected["dnfs_s_per_eff"].iloc[0] == pytest.approx(10.0 / 500)


def test_other_lattice_sizes_are_excluded(tmp_path):
    """D is a hard filter: a D=8 cost has no business in a D=10 grid."""
    _write_dnfs_run(tmp_path, "right_seed42", D=10, draw_seconds=4.0)
    _write_dnfs_run(tmp_path, "wrong_seed43", D=8, draw_seconds=4.0)

    assert list(grid.collect_dnfs(tmp_path, D=10)["seed"]) == [42]


def test_sides_are_seed_meaned_before_joining(tmp_path):
    """Two DNFS seeds and three mchammer seeds give ONE row, not six.

    The two sides share no seed axis, so a row-wise join would both invent a
    correspondence and multiply the row count -- silently weighting whichever
    composition happened to have more baseline seeds.
    """
    results_dir, baseline_dir = tmp_path / "runs", tmp_path / "mch"
    for seed in (42, 43):
        _write_dnfs_run(
            results_dir, f"cell_seed{seed}", seed=seed,
            draw_seconds=10.0, ess_fraction=0.5, n_eval=1000,
        )
    for seed in (42, 43, 44):
        _write_mchammer(
            baseline_dir, f"cell_seed{seed}", seed=seed, s_per_eff=0.06
        )

    table = grid.build_grid(
        grid.collect_dnfs(results_dir, D=10),
        grid.collect_mchammer(baseline_dir, D=10),
    )

    assert len(table) == 1
    row = table.iloc[0]
    assert row["n_seeds"] == 2
    # 0.06 s of MCMC per effective sample against 10/500 = 0.02 s of DNFS.
    assert row["mcmc_over_dnfs"] == pytest.approx(3.0)


def test_seed_filter_isolates_a_survivor_from_the_seed_mean(tmp_path):
    """Restricting to one seed must report THAT seed's cost, not the mean.

    The amortised cells contain both collapsed and healthy seeds, and cost per
    effective sample is 1/ESS, so a collapsed seed contributes an enormous
    number that dominates any average. The seed-mean therefore describes no
    run that exists: it neither reports what the recipe costs when it works
    nor how often it works. Both are needed, so the filter has to be able to
    pull the survivor out.
    """
    for seed, ess_fraction in ((42, 0.001), (44, 0.5)):
        _write_dnfs_run(
            tmp_path, f"cell_seed{seed}", seed=seed,
            draw_seconds=10.0, ess_fraction=ess_fraction, n_eval=1000,
        )

    everything = grid.collect_dnfs(tmp_path, D=10)
    survivor = grid.collect_dnfs(tmp_path, D=10, seeds=[44])

    assert sorted(everything["seed"]) == [42, 44]
    assert list(survivor["seed"]) == [44]
    # 10 s / 500 effective samples, not the mean of that and 10 s / 1.
    assert survivor["dnfs_s_per_eff"].iloc[0] == pytest.approx(0.02)
    assert everything["dnfs_s_per_eff"].mean() > 1.0


def test_swept_run_contributes_one_row_per_composition(tmp_path):
    """An amortised run is costed at every composition it was swept at.

    Collapsing the sweep to a single number would hide that cost per effective
    sample explodes at the window edges, which is where the amortised sampler
    actually loses to a specialist.
    """
    _write_dnfs_run(
        tmp_path, "amort_seed42", draw_seconds=10.0, n_eval=1000,
        sweep=[(0.30, 0.01), (0.50, 0.50)],
    )

    collected = grid.collect_dnfs(tmp_path, D=10).set_index("composition")

    assert sorted(collected.index) == [0.30, 0.50]
    # ESS 10 vs 500 for the same 10 s: the edge is 50x costlier.
    assert collected.loc[0.30, "dnfs_s_per_eff"] == pytest.approx(1.0)
    assert collected.loc[0.50, "dnfs_s_per_eff"] == pytest.approx(0.02)
