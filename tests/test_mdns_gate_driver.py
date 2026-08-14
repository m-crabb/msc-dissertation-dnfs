"""Characterisation pin for the MDNS budget-masked gate driver.

Why this file exists. The driver (`mdns_budget_gate_4x4.py`) carries the
whole gate-3 result and had NO test coverage whatsoever — `build_space`,
`run_slate`, `evaluate_arm` and `conditional_kl_and_late_error` were
entirely unpinned, while the library underneath it
(`samplers/budget_masked.py`) is pinned by 35 tests. M4a has to lift the
lattice size out of the driver's module globals and make the reference
block pluggable (exact enumeration at 4x4; chains + slice-TI at 8x8,
where 2^64 states rule enumeration out), and that is precisely the kind
of refactor that moves numbers silently.

So the contract here is deliberately narrow and blunt:

1. **The 4x4 path does not move.** Golden values captured from the
   pre-refactor driver at a tiny budget, across the three arms that
   exercise the distinct branches: "a" (budget-tilted, constrained),
   "b" (untilted, constrained — the preconditioner-ablation branch) and
   "u" (unconstrained free-space control, whose instrument set is a
   DIFFERENT dict: log-Z error instead of the fibre instruments). If a
   size refactor changes any of these, it changed the 4x4 gate.
2. **Feasibility is structural, not learned.** `off_fibre_count` and
   `train_off_fibre_count` are 0 on the constrained arms at an untrained
   model and a 3-step budget alike — the G0 property that makes the
   budget-masked family worth scaling in the first place.
3. **The driver is deterministic.** Verified bit-identical over repeated
   runs before these values were frozen; asserted here so a future
   nondeterminism regression fails loudly rather than making the golden
   pin flaky.

The budget is tiny on purpose (3 steps, 512 eval rollouts, 32 training
rollouts): this pins WIRING, not convergence. The numbers below are
therefore untrained-model numbers and carry no research meaning — their
only job is to be unchanged.

Tolerance: `rel=1e-5`, the fp32 batch-blocking class used elsewhere in
this suite, not exact equality — the refactor is allowed to change
reduction ORDER (e.g. batching an eval loop differently), and is not
allowed to change the quantity.
"""
import tempfile
from pathlib import Path

import pytest
import torch

import experiments.constrained_hard_03.mdns_budget_gate_4x4 as gate

# Tiny budget. Chosen so the whole slate runs in a few seconds on CPU while
# still touching every branch of the eval path.
TRAIN_ROLLOUTS = 32
EVAL_BATCH = 256
SIGMA = 0.223
SEED = 42
STEPS = 3
EVAL_ROLLOUTS = 512
EVAL_CONTEXTS = 8

# Captured from the driver at commit 0e94c72, before any M4a size work.
GOLDEN = {
    "a_seed42": {
        "ess_fraction": 0.31994199752807617,
        "energy_tv": 0.07382223010063171,
        "off_fibre_count": 0,
        "free_energy_model": -1.4417651891708374,
        "free_energy_ref": -1.5210883617401123,
        "free_energy_bias": 0.0793231725692749,
        "max_level_excess": 0.0012499511241912398,
        "final_loss": 0.8717119693756104,
        "final_train_ess_fraction": 0.3532274067401886,
        "train_off_fibre_count": 0,
        "init_conditional_kl": 0.04470909386873245,
        "trained_conditional_kl": 0.044754497706890106,
        "init_late_generation_error": 0.08076935261487961,
        "trained_late_generation_error": 0.08063587546348572,
    },
    "b_seed42": {
        "ess_fraction": 0.03658585622906685,
        "energy_tv": 0.01104261726140976,
        "off_fibre_count": 0,
        "free_energy_model": -1.1052614450454712,
        "free_energy_ref": -1.5210883617401123,
        "free_energy_bias": 0.4158269166946411,
        "max_level_excess": 0.000819844007492021,
        "final_loss": 8.786092758178711,
        "final_train_ess_fraction": 0.044131554663181305,
        "train_off_fibre_count": 0,
        "init_conditional_kl": 0.20672012865543365,
        "trained_conditional_kl": 0.20639090240001678,
        "init_late_generation_error": 0.46342790126800537,
        "trained_late_generation_error": 0.46286383271217346,
    },
    "u_seed42": {
        "ess_fraction": 0.09306475520133972,
        "energy_tv": 0.07176536321640015,
        "final_loss": 3.068220615386963,
        "final_train_ess_fraction": 0.16672587394714355,
        "train_off_fibre_count": 0,
        "init_conditional_kl": 0.12875548005104065,
        "trained_conditional_kl": 0.12871220707893372,
        "init_late_generation_error": 0.024798929691314697,
        "trained_late_generation_error": 0.024281099438667297,
        "log_z_estimate": 15.697433471679688,
        "log_z_exact": 15.65591049194336,
        "log_z_abs_error": 0.041522979736328125,
        "composition_mean": 0.4830322265625,
    },
}

# The on-slice exact free-energy reference at (4x4, sigma_c). Cross-checked
# against the slice-TI validation stage's -1.52110 (slice_ti.py run against
# exact enumeration): the two constructions agree to 5 decimals, which is
# the evidence that swapping enumeration for TI at 8x8 substitutes the SAME
# quantity rather than a differently-normalised one.
SLICE_TI_4X4_SC = -1.52110


@pytest.fixture(scope="module")
def slate():
    """One tiny slate, shared by every assertion in the file."""
    original = (gate.TRAIN_ROLLOUTS_PER_STEP, gate.EVAL_BATCH)
    gate.TRAIN_ROLLOUTS_PER_STEP, gate.EVAL_BATCH = TRAIN_ROLLOUTS, EVAL_BATCH
    try:
        with tempfile.TemporaryDirectory() as tmp:
            yield gate.run_slate(
                sigma=SIGMA, seeds=[SEED], arms=["a", "b", "u"], steps=STEPS,
                results_root=Path(tmp), tag="pin", device="cpu",
                eval_rollouts=EVAL_ROLLOUTS, eval_contexts=EVAL_CONTEXTS,
                objective="lv",
            )
    finally:
        gate.TRAIN_ROLLOUTS_PER_STEP, gate.EVAL_BATCH = original


def _report(slate, key):
    matches = [v for k, v in slate.items() if str(k) == key]
    assert len(matches) == 1, f"expected exactly one report for {key}"
    return matches[0]


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_4x4_metrics_are_unchanged(slate, key):
    """Contract 1: the 4x4 numbers the gate-3 result rests on do not move."""
    report = _report(slate, key)
    for field, expected in GOLDEN[key].items():
        assert field in report, f"{key}: driver stopped reporting {field!r}"
        actual = report[field]
        if isinstance(expected, int) and not isinstance(expected, bool):
            assert actual == expected, f"{key}.{field}"
        else:
            assert actual == pytest.approx(expected, rel=1e-5), \
                f"{key}.{field}"


@pytest.mark.parametrize("key", ["a_seed42", "b_seed42"])
def test_constrained_arms_never_leave_the_fibre(slate, key):
    """Contract 2: feasibility is a property of the move set, so it holds
    at an untrained model and after three steps alike — no training, no
    tolerance, no clamp-dependence. This is the G0 property the whole
    budget-masked route is chosen for."""
    report = _report(slate, key)
    assert report["off_fibre_count"] == 0
    assert report["train_off_fibre_count"] == 0


def test_free_arm_reports_the_free_space_instrument_set(slate):
    """The unconstrained control lives on 2^16, not the fibre, so it must
    report log-Z error and composition spread and must NOT report the
    fibre instruments. Pinned because the size refactor touches exactly
    the branch that chooses between the two dicts."""
    report = _report(slate, "u_seed42")
    for field in ("log_z_estimate", "log_z_exact", "log_z_abs_error",
                  "composition_mean", "composition_std"):
        assert field in report
    for field in ("off_fibre_count", "free_energy_model", "within_level"):
        assert field not in report, \
            f"fibre instrument {field!r} is undefined off the slice"


def test_exact_slice_reference_agrees_with_slice_ti(slate):
    """The 8x8 plan replaces `on_slice_free_energy_reference` (enumeration)
    with the slice-TI constant. That substitution is only legitimate if the
    two agree where both are computable — they do, at 4x4."""
    reference = _report(slate, "a_seed42")["free_energy_ref"]
    assert reference == pytest.approx(SLICE_TI_4X4_SC, abs=5e-5)


def test_driver_is_deterministic():
    """Contract 3: the golden pin above is only meaningful if repeated runs
    agree. Re-runs the smallest useful slate rather than the full fixture."""
    original = (gate.TRAIN_ROLLOUTS_PER_STEP, gate.EVAL_BATCH)
    gate.TRAIN_ROLLOUTS_PER_STEP, gate.EVAL_BATCH = TRAIN_ROLLOUTS, EVAL_BATCH
    try:
        runs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                slate = gate.run_slate(
                    sigma=SIGMA, seeds=[SEED], arms=["a"], steps=STEPS,
                    results_root=Path(tmp), tag="det", device="cpu",
                    eval_rollouts=EVAL_ROLLOUTS, eval_contexts=EVAL_CONTEXTS,
                    objective="lv",
                )
            runs.append(_report(slate, "a_seed42"))
        for field in ("ess_fraction", "energy_tv", "final_loss",
                      "free_energy_model", "trained_conditional_kl"):
            assert runs[0][field] == runs[1][field], f"{field} is not stable"
    finally:
        gate.TRAIN_ROLLOUTS_PER_STEP, gate.EVAL_BATCH = original


def test_torch_default_dtype_is_untouched(slate):
    """Cheap guard: the driver runs at module import and could leave global
    torch state behind, which would silently perturb every later test in a
    parallel worker."""
    assert torch.get_default_dtype() is torch.float32


# --------------------------------------------------------------------------
# M4a: the 8x8 (non-enumerable) reference block


PROBE_ROOT = Path("results/kawasaki_probe")
SLICE_TI_8X8_SC = -1.90410      # slice_ti.py run8x8, +/- 0.00005


@pytest.fixture
def lattice_8x8():
    """configure_lattice mutates module state, so restore it — otherwise a
    parallel worker running the 4x4 pin afterwards would silently score an
    8x8 driver."""
    original = (gate.LATTICE_SIDE, gate.N_SITES, gate.N_PLUS)
    gate.configure_lattice(8)
    try:
        yield
    finally:
        gate.LATTICE_SIDE, gate.N_SITES, gate.N_PLUS = original


def test_configure_lattice_sets_a_half_filled_fibre():
    original = (gate.LATTICE_SIDE, gate.N_SITES, gate.N_PLUS)
    try:
        gate.configure_lattice(8)
        assert (gate.LATTICE_SIDE, gate.N_SITES, gate.N_PLUS) == (8, 64, 32)
        gate.configure_lattice(4)
        assert (gate.LATTICE_SIDE, gate.N_SITES, gate.N_PLUS) == (4, 16, 8)
    finally:
        gate.LATTICE_SIDE, gate.N_SITES, gate.N_PLUS = original


@pytest.mark.parametrize("bad_side", [3, 0, -2])
def test_configure_lattice_rejects_sizes_without_a_half_filled_fibre(bad_side):
    original = (gate.LATTICE_SIDE, gate.N_SITES, gate.N_PLUS)
    try:
        with pytest.raises(ValueError):
            gate.configure_lattice(bad_side)
    finally:
        gate.LATTICE_SIDE, gate.N_SITES, gate.N_PLUS = original


def test_no_stale_n_plus_default_survives_a_rebind():
    """The rebind's one real hazard: a def-time `n_plus_target=N_PLUS`
    default captured at import would keep filling the 4x4 fibre after
    configure_lattice(8), silently producing a 16-site composition on a
    64-site lattice. Assert the defaults are gone rather than trusting a
    comment."""
    import inspect
    for function in (gate.train_arm, gate.evaluate_arm,
                     gate.conditional_kl_and_late_error):
        parameter = inspect.signature(function).parameters["n_plus_target"]
        assert parameter.default is inspect.Parameter.empty, (
            f"{function.__name__} still defaults n_plus_target — it would "
            f"go stale after configure_lattice()"
        )


@pytest.mark.skipif(not PROBE_ROOT.exists(),
                    reason="certified probe chains not present locally")
def test_chain_reference_matches_the_probes_own_energy_convention(lattice_8x8):
    """The chain reference is only a valid stand-in for enumeration if it
    measures energy the way `slice_energy_hist` does. The driver's reader
    applies `_energy` directly; the probe's `reference_energies` routes via
    `observable_values("energy", ...)`. They must agree exactly — if they
    ever diverge, every 8x8 TV silently becomes meaningless."""
    from experiments.constrained_hard_03.plot_probe_8x8 import (
        reference_energies,
    )
    from discrete_flow_sampler.targets.ising import (
        FixedCompositionIsingTarget,
    )
    target = FixedCompositionIsingTarget(
        D=8, sigma=0.223, target_composition=0.5
    )
    ours = gate.chain_reference_energies(PROBE_ROOT, "sc", target.A.cpu())
    theirs = torch.from_numpy(reference_energies(PROBE_ROOT, "sc", target))
    assert ours.shape == theirs.shape
    assert torch.equal(ours, theirs)


@pytest.mark.skipif(not PROBE_ROOT.exists(),
                    reason="certified probe chains not present locally")
def test_chain_referenced_space_drops_the_enumeration_only_instruments(
        lattice_8x8):
    """The 8x8 reference block must supply a normalised energy histogram
    and the TI free-energy constant, and must report the absence of the
    enumerated slice as None — not as an empty tensor that downstream code
    would treat as a real (and empty) population."""
    space = gate.build_space(
        constrained=True, sigma=0.223, device="cpu", eval_contexts=8,
        probe_root=PROBE_ROOT, probe_point="sc",
        free_energy_ref=SLICE_TI_8X8_SC,
    )
    assert space["states"] is None
    assert space["log_p"] is None
    assert space["contexts"] is None
    assert space["exact_log_z"] is None
    assert space["n_plus_target"] == 32
    assert space["free_energy_ref"] == SLICE_TI_8X8_SC
    centres, mass = space["exact_hist"]
    assert len(centres) == len(mass) == len(space["bins"]) - 1
    assert mass.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (mass >= 0).all()


@pytest.mark.skipif(not PROBE_ROOT.exists(),
                    reason="certified probe chains not present locally")
def test_chain_reference_is_refused_for_the_unconstrained_arm(lattice_8x8):
    """The chains sample the fixed-composition fibre; there is no chain
    counterpart for the free-space control, and silently handing it a
    fibre reference would score it against the wrong population."""
    with pytest.raises(ValueError, match="fixed-composition"):
        gate.build_space(
            constrained=False, sigma=0.223, device="cpu", eval_contexts=8,
            probe_root=PROBE_ROOT, probe_point="sc",
            free_energy_ref=SLICE_TI_8X8_SC,
        )


@pytest.mark.skipif(not PROBE_ROOT.exists(),
                    reason="certified probe chains not present locally")
def test_chain_regime_requires_a_free_energy_reference(lattice_8x8):
    """Without the TI constant the free-energy bias would be computed
    against nothing; fail loudly at build time rather than reporting a
    bias against an implicit zero."""
    with pytest.raises(ValueError, match="free-energy reference"):
        gate.build_space(
            constrained=True, sigma=0.223, device="cpu", eval_contexts=8,
            probe_root=PROBE_ROOT, probe_point="sc", free_energy_ref=None,
        )


@pytest.mark.skipif(not PROBE_ROOT.exists(),
                    reason="certified probe chains not present locally")
def test_8x8_eval_runs_end_to_end_and_stays_on_the_fibre(lattice_8x8):
    """The plumbing gate the plan asks for: an untrained 8x8 model must
    evaluate against the chain reference and return the fibre instruments
    MINUS within-level, with structural feasibility intact at 64 sites
    (where `_pack_spin_keys` would have silently overflowed had the
    within-level instrument been kept)."""
    original = gate.EVAL_BATCH
    gate.EVAL_BATCH = 64
    try:
        space = gate.build_space(
            constrained=True, sigma=0.223, device="cpu", eval_contexts=8,
            probe_root=PROBE_ROOT, probe_point="sc",
            free_energy_ref=SLICE_TI_8X8_SC,
        )
        target = space["target"]
        gate.seed_everything(SEED)
        net = gate.MaskedConditionalNet(gate.N_SITES)
        logit_fn, _ = gate.make_logit_fn(net, target.A, 0.223, "budget_tilted")
        with tempfile.TemporaryDirectory() as tmp:
            metrics = gate.evaluate_arm(
                logit_fn, target, space["states"], space["log_p"],
                space["exact_hist"], space["bins"], 0.223, "cpu",
                Path(tmp), 128, n_plus_target=space["n_plus_target"],
                save_artefacts=False,
                free_energy_ref=space["free_energy_ref"],
            )
    finally:
        gate.EVAL_BATCH = original

    assert metrics["off_fibre_count"] == 0
    assert 0.0 <= metrics["ess_fraction"] <= 1.0
    assert 0.0 <= metrics["energy_tv"] <= 1.0
    assert metrics["free_energy_ref"] == SLICE_TI_8X8_SC
    assert metrics["free_energy_bias"] == pytest.approx(
        metrics["free_energy_model"] - SLICE_TI_8X8_SC
    )
    for dropped in ("within_level", "max_level_excess"):
        assert dropped not in metrics, \
            f"{dropped} needs the enumerated slice and must be omitted"
