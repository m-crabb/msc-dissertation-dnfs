"""End-of-run eval and the per-composition sweep for the amortised cells.

Two obligations, one structural and one statistical.

**A conditioned model must survive its own final eval.** The end-of-run block
calls the model outside the training loop, and a composition-conditioned model
raises when c is missing — deliberately, so that an unconditioned eval cannot
happen silently. Unbound, an amortised cell trains to completion and then dies
at the very last step, after all the compute has been spent.

**Per-composition numbers must be measured at the composition they claim.**
Both the draw and the scoring happen inside one binding: the IS log-weights
accumulate log p̃_t at the bound c, so the free energy and internal energy read
off them must use the same c. Scoring weights drawn at c against the density at
c' mixes two targets into a single estimate and raises nothing — it just
reports a confident wrong number. `test_sweep_scores_each_row_against_its_own
_target` pins that by checking the exact enumeration moves with c.

The in-loop ESS probe is pinned to the window centre, so it tracks training
health only; every per-composition claim has to come from this sweep.
"""

import json

import pytest
import torch
from experiments.dnfs_baseline_01.configs import (
    CompositionCfg,
    CTMCCfg,
    EvalCfg,
    IsingCfg,
    ModelCfg,
    StageCfg,
    TrainCfg,
)
from experiments.dnfs_baseline_01.run import (
    HELD_OUT_COMPOSITIONS,
    SPECIALIST_COMPOSITIONS,
    SWEEP_COMPOSITIONS,
    _composition_binding,
    composition_sweep,
    ess_from_log_weights,
    eval_only,
    train,
)

from discrete_flow_sampler.targets.ising import IsingTarget


def _tiny_cfg(name, *, conditioned=True, centre=0.5, target_composition=0.5):
    """A 4-site amortised cell that trains in a couple of seconds.

    D=2 keeps `2**target.d = 16` states, so the exact-enumeration references
    in the metrics dict stay cheap while still exercising the code path that
    reads the target density under the binding.
    """
    return StageCfg(
        name=name,
        ising=IsingCfg(
            D=2,
            sigma=0.1,
            bias=0.0,
            target_composition=target_composition,
            composition_penalty_strength=5.0,
        ),
        train=TrainCfg(
            n_steps=4,
            batch_size=8,
            outer_batch_size=4,
            inner_steps_per_outer=2,
            replay_buffer_cycles=2,
            lr=1e-3,
            seed=0,
            warmup_steps=0,
        ),
        ctmc=CTMCCfg(n_euler_steps=8),
        eval=EvalCfg(eval_every=2, n_eval_samples=16),
        model=ModelCfg(
            kind="let",
            hidden_dim=8,
            n_layers=1,
            n_heads=2,
            vocab_size=2,
            condition_on_composition=conditioned,
        ),
        estimator="control_variate",
        composition=(
            CompositionCfg(centre=centre, half_width=0.1) if conditioned else None
        ),
    )


@pytest.fixture(scope="module")
def amortised_run(tmp_path_factory):
    """One trained amortised run dir, shared by the sweep tests."""
    torch.manual_seed(0)
    return train(
        _tiny_cfg("tiny_amortised"),
        seed=0,
        output_dir=tmp_path_factory.mktemp("amortised"),
        use_wandb=False,
    )


def test_binding_reaches_both_the_model_and_the_density():
    """c has to arrive in two places, and neither is inferable from the other.

    The model needs it as an input, or it predicts rates for some other
    composition; the target needs it because the penalty is the only thing that
    makes p_c differ from p. A specialist gets neither — the same objects it
    always got.
    """
    target = IsingTarget(
        D=2,
        sigma=0.1,
        target_composition=0.5,
        composition_penalty_strength=5.0,
    )
    x = target.sample_base(4, device="cpu")

    wrap, binding = _composition_binding(target, 0.8, "cpu")
    with binding:
        torch.testing.assert_close(target._row_composition(x), torch.full((4,), 0.8))
    sentinel = object()
    assert wrap(sentinel).composition.item() == pytest.approx(0.8)
    # Restored on exit: no eval can leak a binding into the next one.
    assert target._row_composition(x) == 0.5

    wrap, binding = _composition_binding(target, None, "cpu")
    with binding:
        assert wrap(sentinel) is sentinel
        assert target._row_composition(x) == 0.5


def test_conditioned_cell_completes_its_end_of_run_eval(amortised_run):
    """The launch blocker: 50k steps of training must not die at the last step."""
    metrics = json.loads((amortised_run / "eval" / "metrics.json").read_text())

    assert metrics["n_eval_samples"] == 16
    assert 0.0 < metrics["ess_fraction"] <= 1.0
    # Conditioned on the window centre, matching the in-loop ESS probe, so the
    # training curve and the final number describe the same conditional model.
    assert metrics["target_composition"] == 0.5
    assert (amortised_run / "eval" / "samples.pt").exists()


def test_sweep_covers_the_specialists_and_the_held_out_points(amortised_run):
    rows = composition_sweep(amortised_run)

    assert [row["composition"] for row in rows] == list(SWEEP_COMPOSITIONS)
    assert set(SPECIALIST_COMPOSITIONS) | set(HELD_OUT_COMPOSITIONS) == set(
        SWEEP_COMPOSITIONS
    )
    for row in rows:
        assert row["held_out"] == (row["composition"] in HELD_OUT_COMPOSITIONS)
        # The two headline quantities: does the sampler work at c, and does it
        # obey the c it was told?
        assert 0.0 < row["ess_fraction"] <= 1.0
        assert 0.0 <= row["composition_mean"] <= 1.0
        # Joinable with the archived specialists, whose metrics.json carries
        # the same key.
        assert row["target_composition"] == row["composition"]

    assert rows == json.loads(
        (amortised_run / "eval" / "composition_sweep.json").read_text()
    )


def test_sweep_on_the_ema_checkpoint_lands_in_eval_ema(tmp_path):
    """A sweep is a statement about ONE parameter state, so it must land
    beside that state's own frozen eval: `final_ema.pt` -> `eval_ema/`.

    Writing the EMA sweep into `eval/` would silently overwrite the raw
    model's sweep with EMA numbers wearing the raw path — the dual-eval
    convention keys every artefact by the weights that produced it.
    """
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = _tiny_cfg("tiny_amortised_ema")
    cfg = replace(cfg, ema_decay=0.9)
    run_dir = train(cfg, seed=0, output_dir=tmp_path, use_wandb=False)
    raw_sweep_path = run_dir / "eval" / "composition_sweep.json"
    ema_sweep_path = run_dir / "eval_ema" / "composition_sweep.json"

    raw_rows = composition_sweep(run_dir, compositions=(0.50,))
    assert raw_sweep_path.exists() and not ema_sweep_path.exists()

    ema_rows = composition_sweep(
        run_dir, compositions=(0.50,), checkpoint="final_ema.pt"
    )
    assert json.loads(ema_sweep_path.read_text()) == ema_rows
    # The raw sweep survived untouched, and the two states genuinely
    # differ — otherwise this test would pass on a broken load path.
    assert json.loads(raw_sweep_path.read_text()) == raw_rows
    assert ema_rows[0]["ess"] != raw_rows[0]["ess"]


def test_sweep_rows_carry_a_cost_axis(amortised_run):
    """Cost has to be recorded at draw time or it is gone.

    No archived run carries any timing, and the runs span several machines, so
    a cost-vs-quality grid can only be built from runs made after this exists.
    Seconds are machine-specific; `nfe_per_effective_sample` is the quotable
    one — Euler steps × samples ÷ ESS, which prices the Euler budget and is
    comparable across hardware.
    """
    rows = composition_sweep(amortised_run, compositions=(0.50,), save=False)
    row = rows[0]

    assert row["eval_draw_seconds"] > 0.0
    assert row["eval_device"] in ("cpu", "cuda")
    # 8 Euler steps × 16 samples ÷ ESS, and ESS ≤ 16, so cost ≥ 8 per
    # effective sample — i.e. never cheaper than one pass per good sample.
    assert row["nfe_per_effective_sample"] == pytest.approx(8 * 16 / row["ess"])
    assert row["nfe_per_effective_sample"] >= 8.0


def test_sweep_scores_each_row_against_its_own_target(amortised_run):
    """The binding must reach the density, not just the model.

    `free_energy_per_site_exact` is an enumeration over the target at the bound
    composition. If the binding covered only the model call, every row would
    enumerate the same c and these would all be equal — the silent failure this
    test exists to catch.
    """
    rows = composition_sweep(amortised_run, compositions=(0.30, 0.50, 0.80))
    exact = [row["free_energy_per_site_exact"] for row in rows]

    assert len(set(exact)) == 3
    # Zero-bias Ising is symmetric under (x → −x, c → 1−c), so c and 1−c are
    # the same problem: 0.30 and 0.80 must differ, 0.30 and 0.70 must not.
    mirrored = composition_sweep(amortised_run, compositions=(0.70,), save=False)
    assert exact[0] == pytest.approx(
        mirrored[0]["free_energy_per_site_exact"], rel=1e-6
    )


def test_sweep_reseeds_per_composition_for_common_random_numbers(amortised_run):
    """Every composition is drawn from the same base states and noise stream.

    Common random numbers: differences down the sweep then reflect the model's
    behaviour at c, not which base draw each row happened to get. Repeating a
    composition inside one sweep is the sharpest check — the two rows must be
    bit-identical.
    """
    rows = composition_sweep(amortised_run, compositions=(0.50, 0.30, 0.50), save=False)
    # Everything except the clock: `eval_draw_seconds` measures the machine,
    # not the draw, so it is the one field a reproducible sweep may differ on.
    statistics = [
        {k: v for k, v in row.items() if k != "eval_draw_seconds"} for row in rows
    ]

    assert statistics[0] == statistics[2]
    assert rows[0]["ess"] != rows[1]["ess"]


def test_sweep_refuses_an_unconditioned_run(tmp_path):
    """A specialist has no c input, so a 'sweep' over it would be six copies
    of one number wearing ten different labels."""
    torch.manual_seed(0)
    run_dir = train(
        _tiny_cfg("tiny_specialist", conditioned=False),
        seed=0,
        output_dir=tmp_path,
        use_wandb=False,
    )

    with pytest.raises(ValueError, match="composition conditioning"):
        composition_sweep(run_dir)


def test_specialist_end_of_run_eval_is_unchanged(tmp_path):
    """The amortisation plumbing must not perturb the archived comparators.

    Same seed twice, identical eval weights: any stray RNG draw or changed
    base-sampling route on the specialist path would break comparability with
    `results/02_constrained_soft`.
    """
    weights = []
    for index in range(2):
        torch.manual_seed(0)
        run_dir = train(
            _tiny_cfg("tiny_specialist", conditioned=False),
            seed=0,
            output_dir=tmp_path / f"run{index}",
            use_wandb=False,
        )
        weights.append(torch.load(run_dir / "eval" / "log_weights.pt"))
        metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
        assert metrics["target_composition"] == 0.5

    assert torch.equal(weights[0], weights[1])


def test_eval_only_reports_the_composition_the_samples_were_drawn_at(tmp_path):
    """`eval_only` rebuilds the target from config.json, where an amortised run
    records BOTH a fallback scalar and a window centre. The saved samples came
    from the centre, so that is the composition the recomputed metrics belong
    to; reading the fallback instead would relabel them silently."""
    torch.manual_seed(0)
    run_dir = train(
        _tiny_cfg("tiny_offcentre", centre=0.6, target_composition=0.5),
        seed=0,
        output_dir=tmp_path,
        use_wandb=False,
    )

    metrics = eval_only(run_dir)

    assert metrics["target_composition"] == 0.6
    assert 0.0 < metrics["ess_fraction"] <= 1.0


def test_eval_only_redraw_goes_through_the_target_base(tmp_path, monkeypatch):
    """`eval_only(redraw=True)` must re-DRAW from the checkpoint, not rescore
    the saved tensors — and the initial state must come from
    `target.sample_base`.

    Why this exists: the archived 2026-06-17 matched-base evals drew x0 from
    an inline uniform `torch.randint` while the base was Bernoulli(0.8),
    silently omitting a log w0 term with sd ~6.9 nats (the §D4 bug, corrected
    in de9db7c). Rescoring the saved tensors can never repair that — the
    wrong x0 is baked into the saved log-weights — so the recovery path has
    to redraw, and this test pins that it redraws through the corrected base.
    """
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = _tiny_cfg("tiny_matched", conditioned=False, target_composition=0.8)
    cfg = replace(cfg, ising=replace(cfg.ising, base_composition=0.8))
    run_dir = train(cfg, seed=0, output_dir=tmp_path, use_wandb=False)

    stale_metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    stale_samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True)

    base_draw_sizes = []
    original_sample_base = IsingTarget.sample_base

    def recording_sample_base(self, n, device):
        base_draw_sizes.append(n)
        return original_sample_base(self, n, device)

    monkeypatch.setattr(IsingTarget, "sample_base", recording_sample_base)

    metrics = eval_only(run_dir, redraw=True, redraw_seed=7)

    # The eval draw itself went through the (composition-aware) base.
    assert cfg.eval.n_eval_samples in base_draw_sizes
    # Artefacts were replaced by a genuinely fresh draw.
    fresh_samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True)
    assert not torch.equal(fresh_samples, stale_samples)
    assert metrics["redraw_seed"] == 7
    assert 0.0 < metrics["ess_fraction"] <= 1.0

    # The stale eval was archived before being overwritten...
    archived = run_dir / "eval_archived_pre_redraw"
    assert json.loads((archived / "metrics.json").read_text()) == stale_metrics
    # ...and a second redraw does not stack further archives: the FIRST
    # archive is the record of what the bug produced, later redraws are not.
    eval_only(run_dir, redraw=True, redraw_seed=8)
    assert sorted(run_dir.glob("eval_archived_*")) == [archived]


def test_eval_only_redraw_with_grid_override_leaves_eval_frozen(tmp_path):
    """`eval_only(redraw=True, n_euler_override=k)` must write its artefacts
    to `eval_ne<k>/` and leave the frozen `eval/` byte-untouched.

    Why this exists: the ne64-vs-ne128 grid-offset measurement redraws frozen
    checkpoints on BOTH grids, and the archived `eval/` dirs are the record
    the printed F(c) numbers were read from — a redraw that overwrote them
    (or even archived them, implying they were superseded) would destroy the
    very baseline the offset is measured against. The override draw is a NEW
    side measurement, so it gets a side directory and no archive step.
    """
    torch.manual_seed(0)
    run_dir = train(
        _tiny_cfg("tiny_grid_override", conditioned=False),
        seed=0,
        output_dir=tmp_path,
        use_wandb=False,
    )
    frozen_bytes = {
        name: (run_dir / "eval" / name).read_bytes()
        for name in ("samples.pt", "log_weights.pt", "metrics.json")
    }

    metrics = eval_only(run_dir, redraw=True, redraw_seed=7, n_euler_override=16)

    # All three artefacts landed in the side directory, on the override grid.
    override_dir = run_dir / "eval_ne16"
    for name in ("samples.pt", "log_weights.pt", "metrics.json"):
        assert (override_dir / name).exists()
    assert metrics["n_euler_steps"] == 16
    assert (
        json.loads((override_dir / "metrics.json").read_text())["n_euler_steps"] == 16
    )
    assert metrics["redraw_seed"] == 7

    # The frozen eval/ is byte-identical and nothing was archived.
    for name, before in frozen_bytes.items():
        assert (run_dir / "eval" / name).read_bytes() == before
    assert not list(run_dir.glob("eval_archived_*"))

    # A grid override without a fresh draw is meaningless: the saved tensors
    # were drawn on the run's own grid, so rescoring cannot move it.
    with pytest.raises(ValueError):
        eval_only(run_dir, n_euler_override=16)


def test_sweep_saves_the_frames_behind_each_row(amortised_run):
    """The house error columns (dMag, dCorr, EW2) score FRAMES against a
    reference; a row of scalars cannot be re-scored after the fact. So the
    sweep files each composition's draw and its log-weights beside the JSON,
    keyed by the requested c, and the filed weights must reproduce that row's
    own ESS -- the check that a frame set cannot land under a neighbour's c.
    """
    rows = composition_sweep(amortised_run)
    frames_root = amortised_run / "eval" / "composition_sweep"
    for row in rows:
        frame_dir = frames_root / f"c{row['composition']:.4f}"
        samples = torch.load(frame_dir / "samples.pt", weights_only=True)
        log_weights = torch.load(frame_dir / "log_weights.pt", weights_only=True)
        assert samples.shape == (row["n_eval_samples"], 4)
        assert log_weights.shape == (row["n_eval_samples"],)
        assert ess_from_log_weights(log_weights).item() == pytest.approx(
            row["ess"], rel=1e-5
        )
