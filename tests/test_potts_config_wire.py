"""Potts wiring through the hard_03 config + run plumbing.

`FixedCompositionPottsTarget` exists on its own; these tests pin what
"selecting it from a cell" must mean:

1. a Potts cell builds the Potts target with the right D / sigma / species
   counts, and a backbone whose embedding tables are sized for S -- the one
   place where a config mistake becomes an index-out-of-range deep inside
   `nn.Embedding` rather than a readable error;
2. every existing (Ising) cell is untouched -- the new fields default to the
   current behaviour, so no run dir written before them is locked out by the
   `eval_only` config-drift guard (see 58df8e0);
3. the swap CTMC conserves the *species-count vector*, not just a scalar
   n_plus -- the structural claim that makes the hard constraint generalise
   to S > 2 for free;
4. the eval path emits NO composition observables on the Potts route, because
   `diagnostics.metrics` is still two-species. Silence is the correct
   output until an S-vector version lands; a plausible-looking wrong number
   is not.
"""
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from experiments.constrained_hard_03.configs import CONFIGS, HardStageCfg
from experiments.constrained_hard_03.run import (
    build_target_and_head,
    eval_only,
    final_eval,
)
from experiments.dnfs_baseline_01.configs import (
    CTMCCfg,
    EvalCfg,
    IsingCfg,
    ModelCfg,
    TrainCfg,
)

from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget
from discrete_flow_sampler.targets.potts import FixedCompositionPottsTarget

# D=3 (d=9) with three equal species: the smallest lattice that is not the
# degenerate L=2 torus (where a site's two neighbours coincide) AND whose site
# count divides by 3, so an exact equal-composition slice exists.
THIRDS = (1 / 3, 1 / 3, 1 / 3)


def _tiny_potts_cfg(n_eval_samples=8, eval_sample_chunk=4):
    return HardStageCfg(
        name="tiny_potts",
        ising=IsingCfg(D=3, sigma=0.5025, bias=0.0, target_composition=None),
        train=TrainCfg(n_steps=2, batch_size=4, inner_steps_per_outer=2, seed=0),
        ctmc=CTMCCfg(n_euler_steps=8),
        eval=EvalCfg(
            eval_every=2,
            n_eval_samples=n_eval_samples,
            eval_sample_chunk=eval_sample_chunk,
        ),
        model=ModelCfg(kind="letf", hidden_dim=16, n_layers=2, n_heads=2,
                       vocab_size=len(THIRDS)),
        estimator="control_variate",
        head_kind="mask_one",
        target_kind="potts",
        potts_composition=THIRDS,
        wandb_project="test",
    )


def test_potts_cell_builds_potts_target_with_declared_species_counts():
    """`target_kind="potts"` must reach `FixedCompositionPottsTarget` -- not a
    subclass-of-IsingTarget that happens to import -- carrying the cell's D,
    sigma and per-species counts. sigma is the POTTS coupling: a cell
    reproducing an Ising run at s must declare 2s (potts.py module docstring).
    """
    target, _ = build_target_and_head(_tiny_potts_cfg(), "cpu")

    assert isinstance(target, FixedCompositionPottsTarget)
    assert target.D == 3 and target.d == 9
    assert target.sigma == 0.5025
    assert target.n_states == 3
    assert target.target_counts == (3, 3, 3)


def test_potts_backbone_embeddings_sized_for_species_count():
    """The footgun: a backbone built with vocab_size=2 under an S=3 target
    raises deep inside `nn.Embedding` on the first label-2 site, long after
    the config was wrong. Both tables (token_embedder for the body, omega for
    the readout) must have S rows, and a label-(S-1) state must forward."""
    cfg = _tiny_potts_cfg()
    target, head = build_target_and_head(cfg, "cpu")
    backbone = head.backbone

    assert backbone.token_embedder.weight.shape[0] == target.n_states
    assert backbone.omega.weight.shape[0] == target.n_states

    x = target.sample_base(2, device="cpu")
    assert head(x, torch.full((2,), 0.5)).shape == (2, target.d, target.d)


def test_potts_kind_requires_a_composition():
    cfg = replace(_tiny_potts_cfg(), potts_composition=None)
    with pytest.raises(ValueError, match="potts_composition"):
        build_target_and_head(cfg, "cpu")


def test_species_count_is_consistent_across_every_cell():
    """Regression + invariant. Every `H2_*` cell predates Potts and must still
    declare the binary Ising route -- a cell silently flipping kind would
    change what a re-run of a published number means. And across ALL cells the
    name's species prefix, the target kind, the composition length and the
    backbone's vocab_size must agree, which is the property `_hard_cell`
    deriving vocab_size from the composition tuple is there to guarantee."""
    for name, cfg in CONFIGS.items():
        n_species = int(name.split("_")[0][1:])
        assert cfg.model.vocab_size == n_species, name
        if n_species == 2:
            # binary route: the Ising torus, or a binary cluster
            # expansion on a real alloy cell, which fixes its own x_Au
            assert cfg.target_kind in ("ising", "cluster_expansion"), name
            assert cfg.potts_composition is None, name
            if cfg.target_kind == "cluster_expansion":
                assert cfg.ising.expansion_json is not None, name
                # the composition sweep fills every integral slice
                # between Cu3Au and CuAu; the amortised cell mixes them.
                n_sites = 16 if "cuau16" in name else 64
                assert cfg.ising.target_composition * n_sites == round(
                    cfg.ising.target_composition * n_sites), name
            else:
                assert cfg.ising.target_composition == 0.5, name
        else:
            assert cfg.target_kind == "potts", name
            assert len(cfg.potts_composition) == n_species, name
            # A scalar n_plus is meaningless for S > 2; recording 0.5 there
            # would be a false entry in the run's config.json.
            assert cfg.ising.target_composition is None, name


def test_ising_cell_still_builds_ising_target():
    cfg = CONFIGS["H2_d16_c50_s010_letf_dh"]
    target, _ = build_target_and_head(cfg, "cpu")
    assert isinstance(target, FixedCompositionIsingTarget)
    assert not isinstance(target, FixedCompositionPottsTarget)


def test_legacy_run_dir_backfills_the_new_potts_fields(tmp_path, monkeypatch):
    """Every run dir on disk predates `target_kind` / `potts_composition`, so
    their absence must read as "ran on the Ising route" rather than as drift.
    This is the constraint that forces both defaults to be the current
    behaviour; without it the whole eval_only recovery path breaks the moment
    Potts lands."""
    torch.manual_seed(0)
    cfg = replace(
        CONFIGS["H2_d16_c50_s010_letf_dh"],
        head_kind="mask_one",
        eval=EvalCfg(eval_every=2, n_eval_samples=4),
    )
    monkeypatch.setattr(
        "experiments.constrained_hard_03.run.CONFIGS", {cfg.name: cfg}
    )
    run_dir = tmp_path / "legacy_pre_potts"
    (run_dir / "checkpoints").mkdir(parents=True)
    saved = asdict(cfg)
    del saved["target_kind"]
    del saved["potts_composition"]
    (run_dir / "config.json").write_text(json.dumps(saved))
    _, head = build_target_and_head(cfg, "cpu")
    torch.save(head.state_dict(), run_dir / "checkpoints" / "final.pt")

    assert eval_only(run_dir)["n_eval_samples"] == 4


def test_potts_eval_conserves_species_counts_and_omits_binary_metrics(tmp_path):
    """The structural claim, end to end: a swap permutes two labels, so it
    preserves the label MULTISET -- Ising's scalar n_plus generalised to an
    S-vector. Every eval sample must therefore carry exactly the declared
    counts, at an untrained head (the constraint is enforced by the move set,
    not learned).

    And the deliberate gap: `composition_observables` reads
    ((x+1)/2).mean(), i.e. the mean LABEL INDEX once S > 2. It would not
    raise on Potts spins -- it would write a confident, meaningless
    `composition_mean`. The eval must emit nothing there until the S-vector
    diagnostics land."""
    torch.manual_seed(0)
    cfg = _tiny_potts_cfg(n_eval_samples=8, eval_sample_chunk=4)
    target, head = build_target_and_head(cfg, "cpu")

    metrics = final_eval(head, target, cfg, Path(tmp_path))

    samples = torch.load(tmp_path / "eval" / "samples.pt")
    assert samples.shape == (8, 9)
    target.assert_on_manifold(samples)
    assert (target.composition_counts(samples) == 3).all()

    assert metrics["n_eval_samples"] == 8
    assert not any(key.startswith("composition") for key in metrics)
    assert "magnetisation_mean" not in metrics
