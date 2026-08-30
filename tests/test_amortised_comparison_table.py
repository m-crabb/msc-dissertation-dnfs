"""Comparator selection for the amortised-vs-specialist table.

This module decides which archived runs are allowed to stand as "the
specialist" for a composition, which makes it the place a wrong number enters
the report. Every filter here exists because a specific wrong comparator is
sitting in `results/02_constrained_soft` right now:

  * a `matched_anneal` cell at c = 0.80 trained against a Bernoulli(0.8) base —
    a different experiment that failed its own ESS gate;
  * an `lemlp` specialist at c = 0.30 — quoting it against a leTF amortised
    model measures architecture, not amortisation;
  * non-annealed D = 10 cells, which are not the recipe the D = 10 amortised
    cells were cloned from — while at D = 4 the annealed cells are the ones
    that don't exist, so the requirement has to follow the comparator.

Fixtures here are JSON only: the selection logic reads config.json and
metrics.json and never loads a checkpoint, so the test needs no model.
"""
import importlib
import json

import pandas as pd
import pytest

table = importlib.import_module(
    "experiments.constrained_soft_02.analysis.10_amortised_vs_specialist"
)


def _write_run(
    results_dir,
    name,
    *,
    composition=0.5,
    seed=42,
    kind="let",
    anneal=True,
    base_composition=0.5,
    D=10,
    ess_fraction=0.9,
    sweep=None,
    conditioned=False,
    cell_composition=None,
):
    run_dir = results_dir / name
    (run_dir / "eval").mkdir(parents=True)
    cfg = {
        "name": name.split("_seed")[0],
        "ising": {
            "D": D, "sigma": 0.1, "target_composition": composition,
            "composition_penalty_strength": 50.0,
            "base_composition": base_composition,
        },
        "train": {"seed": seed, "n_steps": 50_000},
        "ctmc": {"n_euler_steps": 128},
        "model": {
            "kind": kind, "hidden_dim": 128,
            "condition_on_composition": conditioned,
        },
        "lambda_curriculum": {"stages": []} if anneal else None,
        "composition": cell_composition,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "eval" / "metrics.json").write_text(
        json.dumps({"ess_fraction": ess_fraction, "composition_mean": composition})
    )
    if sweep is not None:
        (run_dir / "eval" / "composition_sweep.json").write_text(json.dumps(sweep))
    return run_dir


@pytest.fixture
def results_dir(tmp_path):
    _write_run(tmp_path, "good_c05_seed42", composition=0.5, ess_fraction=0.90)
    _write_run(tmp_path, "good_c05_seed43", composition=0.5, seed=43,
               ess_fraction=0.80)
    _write_run(tmp_path, "matched_base_c08_seed45", composition=0.8,
               base_composition=0.8, ess_fraction=0.17)
    _write_run(tmp_path, "mlp_c03_seed42", composition=0.3, kind="lemlp",
               ess_fraction=0.85)
    _write_run(tmp_path, "no_anneal_c03_seed42", composition=0.3, anneal=False,
               ess_fraction=0.66)
    return tmp_path


def test_wrong_comparators_are_excluded(results_dir):
    rows = table.collect_specialists(
        results_dir, D=10, sigma=0.1, penalty=50.0
    )

    assert sorted(rows["run"]) == ["good_c05_seed42", "good_c05_seed43"]
    # Each exclusion for its own reason, so a loosened filter fails loudly.
    assert 0.8 not in set(rows["composition"]), "matched base leaked in"
    assert "lemlp" not in set(rows["model_kind"]), "wrong architecture leaked in"


def test_repeated_run_dirs_for_one_seed_are_not_double_counted(results_dir):
    """A relaunched seed must weigh once, not twice.

    The archive really does hold two dirs for c=0.50 seed 42. Averaging over
    rows would count it twice while `n_seeds` still says 4 — a seed mean that
    is wrong in a way its own metadata denies.
    """
    _write_run(results_dir, "good_c05_seed42_relaunch", composition=0.5,
               seed=42, ess_fraction=0.10)

    aggregated = table._aggregate(
        table.collect_specialists(results_dir, D=10, sigma=0.1, penalty=50.0),
        ["composition", "n_euler_steps"],
    )
    row = aggregated.iloc[0]

    assert row["n_seeds"] == 2
    # Newest dir wins (0.10 supersedes 0.90), so the mean is over {0.10, 0.80}
    # — not the 0.60 that double-counting seed 42 would give.
    assert row["ess_fraction"] == pytest.approx(0.45)


def test_anneal_requirement_follows_the_comparator(results_dir):
    """At D=4 the archived cells have no anneal; requiring one returns nothing,
    which would read as 'no comparator exists' rather than 'wrong filter'."""
    annealed = table.collect_specialists(
        results_dir, D=10, sigma=0.1, penalty=50.0, require_anneal=True
    )
    unannealed = table.collect_specialists(
        results_dir, D=10, sigma=0.1, penalty=50.0, require_anneal=False
    )

    assert list(unannealed["run"]) == ["no_anneal_c03_seed42"]
    assert set(annealed["run"]).isdisjoint(set(unannealed["run"]))


def test_amortised_rows_join_onto_specialists_and_keep_held_out(results_dir):
    sweep = [
        {"composition": 0.25, "held_out": False, "ess_fraction": 0.5,
         "composition_mean": 0.26},
        {"composition": 0.4375, "held_out": True, "ess_fraction": 0.6,
         "composition_mean": 0.44},
        {"composition": 0.50, "held_out": False, "ess_fraction": 0.7,
         "composition_mean": 0.50},
    ]
    _write_run(results_dir, "amort_seed42", conditioned=True,
               cell_composition={"centre": 0.5}, sweep=sweep)

    amortised = table.collect_amortised(results_dir, ["amort"])
    built = table.build_table(
        amortised,
        table.collect_specialists(results_dir, D=10, sigma=0.1, penalty=50.0),
    )

    assert len(amortised) == 3
    # c=0.50 has a specialist to sit beside; c=0.4375 is held out and has
    # none, and must survive the join rather than being dropped.
    joined = built.set_index("composition")
    assert joined.loc[0.50, "ess_fraction_specialist"] == pytest.approx(0.85)
    assert joined.loc[0.50, "ess_fraction_amortised"] == pytest.approx(0.7)
    assert bool(joined.loc[0.4375, "held_out"])
    assert pd.isna(joined.loc[0.4375, "ess_fraction_specialist"])
