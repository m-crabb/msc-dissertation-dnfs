"""What correct looks like for the 8x8 house-table fill, written before it.

The 4x4 fill (`house_table_4x4.py`) can lean on exact enumeration: its
reference IS the true conditional, so the reference row's error cells are
zero by construction and only the sampler side needs a floor. At 8x8 the
constrained slice holds C(64,32) ~ 1.8e18 states and no enumeration exists,
so the reference becomes the certified Kawasaki chain pool and three
properties that were free at 4x4 have to be earned:

  * the reference has its OWN precision, which must be reported rather than
    printed as zero (the caption at hard.tex:1150 already promises "(SE)"
    in those cells);
  * the sampling floor must be estimated against a sampled reference rather
    than drawn from an exact pmf;
  * the reference must still be composition-exact, since every error column
    compares against it on the c=0.5 slice.

The reference-SE estimator under test is the HALF-SPLIT: draw two disjoint
halves of the chain pool, measure the error metric BETWEEN them, and halve
it. Two half-size references each carry sqrt(2) times the full pool's noise
and their separation combines both, so the half-split distance is ~2x the
full pool's standard error. The alternative -- bootstrapping the reference
against itself -- estimates the wrong thing: it answers "how much does this
reference wobble under resampling", not "how far apart would two
independent references land", which is what an error column needs.
"""
import json

import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    magnetisation_profile_error,
)

L = 8
D_SITES = L * L


def _balanced_spins(n, seed, d=D_SITES):
    """n draws from the c=0.5 slice: every row exactly d/2 up, d/2 down."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.cat([torch.ones(d // 2), -torch.ones(d // 2)])
    return torch.stack([base[torch.randperm(d, generator=generator)]
                        for _ in range(n)])


# --- the reference itself -------------------------------------------------

def test_reference_is_composition_exact():
    """Every reference state sits on the slice the neural cells are scored
    on. A reference that drifted off c=0.5 would make every error column
    measure the composition gap rather than the structure."""
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    states = _balanced_spins(64, seed=0)
    assert h8.is_composition_exact(states, n_plus=D_SITES // 2)
    off_slice = states.clone()
    off_slice[3, 0] = -off_slice[3, 0]
    assert not h8.is_composition_exact(off_slice, n_plus=D_SITES // 2)


# --- reference standard error ---------------------------------------------

def test_reference_se_is_positive_and_falls_with_more_chains():
    """The reference's precision is not zero at 8x8, and it improves as the
    pool grows -- the property that distinguishes a sampled reference from
    the 4x4 enumerated one."""
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    small = [_balanced_spins(400, seed=s) for s in range(4)]
    large = [_balanced_spins(400, seed=s) for s in range(16)]

    se_small = h8.reference_standard_error(small, L, n_splits=24, seed=0)
    se_large = h8.reference_standard_error(large, L, n_splits=24, seed=0)

    assert se_small["dMag"] > 0.0
    assert se_large["dMag"] < se_small["dMag"]


def test_reference_se_tracks_root_n_scaling():
    """Quadrupling the pool should roughly halve the standard error. Pins
    the half-split estimator's CALIBRATION, not merely its direction: an
    estimator that forgot the factor of two, or that bootstrapped instead,
    would still pass the monotonicity test above."""
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    base = [_balanced_spins(500, seed=s) for s in range(4)]
    quad = [_balanced_spins(500, seed=s) for s in range(16)]

    se_base = h8.reference_standard_error(base, L, n_splits=64, seed=1)["dMag"]
    se_quad = h8.reference_standard_error(quad, L, n_splits=64, seed=1)["dMag"]

    assert 0.35 < se_quad / se_base < 0.72   # ideal 0.5, sampling slack


def test_reference_se_matches_a_directly_measured_one():
    """ABSOLUTE calibration, not just scaling: the half-split estimate must
    match the error an independent pool of the same size actually shows
    against a much larger truth.

    The factor of two is the whole content of the estimator and a naive
    check will not catch it being wrong -- comparing a single pool against a
    single truth conflates three things (the pool's noise, the truth's own
    noise, and the spread of a one-draw comparison) and reads ~0.7x even
    when the estimator is right. The controlled version averages over
    independent pools and removes the truth's contribution in quadrature.
    """
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    n_chains, per_chain, truth_n = 8, 1500, 120_000
    pool = [_balanced_spins(per_chain, seed=s) for s in range(n_chains)]
    truth = _balanced_spins(truth_n, seed=900)
    w_truth = torch.full((truth_n,), 1.0 / truth_n)

    observed = []
    for replicate in range(4):
        flat = _balanced_spins(n_chains * per_chain, seed=1000 + replicate)
        w = torch.full((flat.shape[0],), 1.0 / flat.shape[0])
        observed.append(magnetisation_profile_error(
            flat, w, truth, L, reference_weights=w_truth))
    direct = sum(observed) / len(observed)
    direct *= (1 + n_chains * per_chain / truth_n) ** -0.5

    estimated = h8.reference_standard_error(pool, L, n_splits=96, seed=11)["dMag"]
    assert 0.75 < estimated / direct < 1.35


def test_reference_se_is_far_below_the_sampling_floor():
    """The table is only readable if the reference is sharper than the
    thing it measures. With ~1e5 pooled reference draws against N=5000
    neural draws the floor must dominate the reference SE by several-fold;
    if it did not, no error-column difference between heads could be read."""
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    pool = [_balanced_spins(2000, seed=s) for s in range(15)]
    reference = torch.cat(pool)

    se = h8.reference_standard_error(pool, L, n_splits=32, seed=2)
    floor = h8.sampling_floor_from_reference(
        reference, L, n_draws=500, n_replicates=32, seed=3)

    assert floor["dMag"] > 4 * se["dMag"]


# --- the sampling floor ---------------------------------------------------

def test_sampling_floor_falls_with_more_draws():
    """The floor is the error a PERFECT sampler still shows at finite N, so
    it must decay in N. A floor that ignored n_draws would silently declare
    every neural cell 'at the floor' regardless of its draw count."""
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    reference = _balanced_spins(20_000, seed=4)
    few = h8.sampling_floor_from_reference(
        reference, L, n_draws=250, n_replicates=32, seed=5)
    many = h8.sampling_floor_from_reference(
        reference, L, n_draws=4000, n_replicates=32, seed=5)

    assert many["dMag"] < few["dMag"]
    assert many["dCorr"] < few["dCorr"]


def test_perfect_sampler_sits_at_the_floor():
    """A draw taken FROM the reference must score at the floor, within the
    floor's own spread. This is the calibration that lets the printed table
    say 'this cell is indistinguishable from exact at its own N'."""
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    reference = _balanced_spins(20_000, seed=6)
    generator = torch.Generator().manual_seed(7)
    idx = torch.randint(0, reference.shape[0], (1000,), generator=generator)
    draws = reference[idx]
    uniform = torch.full((1000,), 1.0 / 1000)

    observed = magnetisation_profile_error(draws, uniform, reference, L)
    floor = h8.sampling_floor_from_reference(
        reference, L, n_draws=1000, n_replicates=64, seed=8)

    assert observed < 2.5 * floor["dMag"]


# --- FLOP/es provenance ---------------------------------------------------

def test_flop_config_comes_from_the_run_dir_not_the_registry(tmp_path):
    """The 4x4 fill bills FLOPs off the LIVE registry (house_table_4x4.py
    :228), so a cell trained before `gather_triu_pairs` or `compile_model`
    landed is charged at today's architecture rather than its own. The 8x8
    fill must read each run's saved config.json, which is the only record of
    what actually trained."""
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    run_dir = tmp_path / "H2_d64_c50_s010_letf_thp_50k_w2_seed42_tag"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "name": "H2_d64_c50_s010_letf_thp_50k_w2",
        "gather_triu_pairs": False,
        "ctmc": {"n_euler_steps": 77},
    }))

    saved = h8.run_dir_config(run_dir)
    assert saved["gather_triu_pairs"] is False
    assert saved["ctmc"]["n_euler_steps"] == 77


def test_registry_drift_against_run_dir_is_reported(tmp_path):
    """The audit must NAME the drifted fields rather than silently pick a
    side -- a cell whose saved config disagrees with the registry is a
    provenance finding, not something for the fill to paper over."""
    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    saved = {"gather_triu_pairs": False, "compile_head": False,
             "ctmc": {"n_euler_steps": 128}}
    live = {"gather_triu_pairs": True, "compile_head": False,
            "ctmc": {"n_euler_steps": 128}}

    drift = h8.config_drift(saved, live, fields=("gather_triu_pairs",
                                                 "compile_head"))
    assert drift == {"gather_triu_pairs": (False, True)}


# --- integration on the real reference, skipped when absent ---------------

@pytest.mark.parametrize("sigma_label,npz_tag", [("s010", "s100"),
                                                 ("s220", "s220")])
def test_real_reference_certifies(sigma_label, npz_tag):
    """The chapter caption claims R-hat <= 1.01 on every observable. Pin it
    against the shipped chains so the claim cannot rot."""
    from pathlib import Path

    from experiments.constrained_hard_03.analysis import house_table_8x8 as h8

    kawasaki_dir = Path(__file__).resolve().parents[1] / "results" / "03_hard" / "kawasaki_w2"
    if not sorted(kawasaki_dir.glob(f"kawasaki_D{L}_{npz_tag}_seed*.npz")):
        pytest.skip("D8 reference chains not present")

    chains = h8.load_reference_chains(kawasaki_dir, L, npz_tag,
                                      burn_in_fraction=0.2)
    assert len(chains) == 15
    assert h8.is_composition_exact(torch.cat(chains), n_plus=D_SITES // 2)


# --- Separable billing ----------------------------------------------------
#
# The band identity is EXACT, so a masked-attention head's honest per-sample
# bill is the cheapest exact way to evaluate it -- not whichever contraction
# order happened to exist on the day it trained. Every archived MA cell ran
# dense only because the derivation landed after them. These pin the swap:
# that it happens, that it moves nothing but the bill, and that it stays off
# the families it does not apply to.

@pytest.mark.parametrize("name,expected", [
    ("H2_d64_c50_s220_letf_ma_50k_curr_w2", True),
    ("H2_d64_c50_s220_letf_mamo2ef_50k_curr_w2", True),
    ("H2_d64_c50_s220_letf_iv_50k_curr_w2", False),
    ("H2_d64_c50_s220_letf_ivmo2ef_50k_curr_w2", False),
    ("H2_d64_c50_s220_letf_thp_50k_curr_w2", False),
    ("H2_d64_c50_s220_letf_mo_50k_curr_w2", False),
    ("H2_d64_c50_s220_letf_fimo2ef_50k_curr_w2", False),
])
def test_billing_config_is_separable_for_attention_bands_only(name, expected):
    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        flop_billing_config)
    from experiments.constrained_hard_03.configs import CONFIGS

    billed = flop_billing_config(CONFIGS[name])
    assert billed.separable_band_scores is expected, name


def test_billing_config_moves_exactly_one_field():
    """A billing config that drifted on anything else would be exactly the
    misattribution `registry_config_for` exists to refuse. The swap is
    licensed by the band identity and by nothing else, so it must not carry
    a second change in with it."""
    from dataclasses import replace

    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        flop_billing_config)
    from experiments.constrained_hard_03.configs import CONFIGS

    trained = CONFIGS["H2_d64_c50_s220_letf_ma_50k_curr_w2"]
    billed = flop_billing_config(trained)
    assert billed == replace(trained, separable_band_scores=True)


def test_billed_head_computes_the_trained_head_s_function():
    """The claim the re-bill rests on: same weights, same outputs. If this
    ever failed, the table would be pricing a different model from the one
    whose ESS it prints -- the one error an exactness argument cannot
    survive."""
    import torch

    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        flop_billing_config)
    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.run import build_target_and_head

    cfg = CONFIGS["H2_d16_c50_s220_letf_ma_10k_w2"]
    torch.manual_seed(0)
    _, trained = build_target_and_head(cfg, device="cpu")
    torch.manual_seed(0)
    _, billed = build_target_and_head(flop_billing_config(cfg), device="cpu")

    d = cfg.ising.D ** 2
    half = torch.cat([torch.ones(d // 2), -torch.ones(d - d // 2)])
    x = torch.stack([half[torch.randperm(d)] for _ in range(2)])
    t = torch.full((2,), 0.5)
    with torch.no_grad():
        drift = (billed(x, t) - trained(x, t)).abs().max().item()
    assert drift < 1e-5, f"billed head moved G by {drift:.2e}"


def test_separable_billing_actually_lowers_the_attention_bill():
    """The point of the exercise. A no-op here would mean the flag never
    reached the head and the table quietly kept the dense price."""
    import torch

    from discrete_flow_sampler.diagnostics.flops import measured_forward_flops
    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        flop_billing_config)
    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.run import build_target_and_head

    cfg = CONFIGS["H2_d16_c50_s220_letf_ma_10k_w2"]
    d = cfg.ising.D ** 2
    half = torch.cat([torch.ones(d // 2), -torch.ones(d - d // 2)])
    example = (torch.stack([half[torch.randperm(d)]]), torch.full((1,), 0.5))

    def bill(c):
        torch.manual_seed(0)
        _, head = build_target_and_head(c, device="cpu")
        return measured_forward_flops(head, example)

    assert bill(flop_billing_config(cfg)) < bill(cfg)


def test_ladder_provenance_is_keyed_by_coupling_not_by_arm():
    """The ladder's two columns ran in separate campaigns under separate
    tags, so provenance must be per (arm, coupling).

    The failure this guards is SILENT: an arm-keyed map sends the floor
    lookup to the sigma_c tag, the run dirs are absent, and the fill's
    missing-condition branch prints `--` -- indistinguishable from "not yet
    run". The column would stay blank with the runs sitting on disk.
    """
    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        ARM_PROVENANCE)

    assert all(isinstance(k, tuple) and len(k) == 2 for k in ARM_PROVENANCE)
    for arm in ("mamo2", "mamo2ef", "iv", "ivmo2", "ivmo2ef"):
        assert ARM_PROVENANCE[(arm, "s220")] == "20260828-rasterord-d64"
        assert ARM_PROVENANCE[(arm, "s010")] == "20260828-rasterfloor-d64"
    assert ARM_PROVENANCE[("masep", "s010")] == "20260828-rasterfloor-d64"
    assert ("masep", "s220") not in ARM_PROVENANCE


def test_every_provenanced_cell_names_a_real_config():
    """A tag typo or a renamed arm would otherwise surface as a permanently
    blank row rather than an error."""
    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        ARM_PROVENANCE, CELL_NAME)
    from experiments.constrained_hard_03.configs import CONFIGS

    for (arm, sigma_label) in ARM_PROVENANCE:
        name = CELL_NAME[sigma_label].format(arm=arm)
        assert name in CONFIGS, name


# --- GFlowNet comparator rows ---------------------------------------------

def test_gfn_cells_name_real_configs():
    """Both GFN arms at both couplings must resolve to registered d64
    configs; a tag or name typo would otherwise print as a permanently
    blank row (same failure mode the provenance test above guards)."""
    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        GFN_ARMS, GFN_CELL_NAME, SIGMA_LABELS)
    from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS

    for gfn_arm in GFN_ARMS:
        for sigma_label in SIGMA_LABELS:
            name = GFN_CELL_NAME.format(
                sigma_label=sigma_label,
                objective=gfn_arm.removeprefix("gfn_"))
            assert name in GFN_CONFIGS, name


def test_gfn_registry_audit_catches_architecture_drift(tmp_path):
    """The GFN rows carry the same provenance promise as the swap rows: the
    bill is measured on a policy rebuilt from the registry, so the fill must
    refuse if the run's own saved config disagrees on an architecture field.
    A policy trained at hidden 32 billed at the registry's hidden 64 would
    silently overstate the row's FLOP/es."""
    from dataclasses import asdict
    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        gfn_registry_config_for)
    from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS

    name = "GFN_d64_c50_s010_tb_50k_par"
    saved = asdict(GFN_CONFIGS[name])

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "config.json").write_text(json.dumps(saved))
    assert gfn_registry_config_for(clean).name == name

    drifted_cfg = dict(saved, hidden_dim=32)
    drifted = tmp_path / "drifted"
    drifted.mkdir()
    (drifted / "config.json").write_text(json.dumps(drifted_cfg))
    with pytest.raises(ValueError, match="hidden_dim"):
        gfn_registry_config_for(drifted)


def test_gfn_row_reads_frozen_ess_and_bills_without_euler_factor(tmp_path):
    """At this rung both sides store 5000 draws, so the GFN row reads its
    frozen ess_fraction exactly like every house row (the 4x4 fill's
    truncate-and-recompute deviation does not apply). And the bill handed in
    is used per RAW SAMPLE as-is: an autoregressive rollout has no Euler
    grid, so a bill that picked up the house n_euler multiplier would
    overstate FLOP/es by two orders."""
    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        neural_cell)

    run_dir = tmp_path / "gfn_run"
    (run_dir / "eval").mkdir(parents=True)
    n = 64
    samples = _balanced_spins(n, seed=1)
    torch.save(samples, run_dir / "eval" / "samples.pt")
    torch.save(torch.zeros(n), run_dir / "eval" / "log_weights.pt")
    frozen_ess = 0.625  # deliberately NOT the value uniform weights imply
    (run_dir / "eval" / "metrics.json").write_text(
        json.dumps({"ess_fraction": frozen_ess}))

    class UniformTarget:
        sigma = 0.1

        def log_prob(self, states):
            return torch.zeros(states.shape[0])

    reference = _balanced_spins(128, seed=2)
    flops_per_raw = 1.0e6
    row = neural_cell(run_dir, UniformTarget(), reference,
                      torch.zeros(128), flops_per_raw)
    assert row["ESS"] == frozen_ess
    assert row["FLOP/es"] == pytest.approx(flops_per_raw / frozen_ess)


def test_gfn_rows_stay_outside_the_bold_comparison():
    """The GFN rows are a different sampling paradigm (decided at 4x4,
    hard.tex caption): even when a GFN cell holds the best number in a
    column, the bold must land on the best SWAP cell. A refactor that
    computed `best` over every key in the table would silently move it."""
    from experiments.constrained_hard_03.analysis.house_table_8x8 import (
        latex_table)

    def entry(ess, flops):
        return {"ESS": (ess, 0.001), "dMag": (0.05, 0.01),
                "dCorr": (0.05, 0.01), "EW2": (0.05, 0.01),
                "FLOP/es": (flops, 0.0)}

    table = {
        "mo_s010": entry(0.90, 1.0e9),
        "gfn_tb_s010": entry(0.99, 1.0e6),  # best ESS and best FLOP/es
    }
    body = latex_table(table)
    gfn_line = next(l for l in body.splitlines() if "trajectory balance" in l)
    mo_line = next(l for l in body.splitlines() if "mask-one" in l)
    assert "mathbf" not in gfn_line
    assert "mathbf" in mo_line
