"""Literal icet/mchammer baselines for the Ising target, with timing.

Why literal mchammer rather than our numba samplers: the thesis benchmarks
DNFS against *what materials practitioners actually run*. A fast bespoke
engine would be either an unclaimed contribution or unexplained machinery, so
the reported baselines are mchammer's own ensembles; the numba scripts
(`scripts/vcsgc_mcmc_validation.py`, `mcmc/kawasaki.py`) remain as
independent cross-checks that would catch an embedding bug here.

The embedding (validated to ~1e-9 against exact enumeration): the 2D Ising
torus is a single-layer FCC-free cell with a 10-unit vacuum gap in z and a
pair cutoff strictly between 1 and sqrt(2), so icet sees exactly one
nearest-neighbour pair orbit (4 neighbours per site) and no spurious z-image
or second-neighbour bonds. With ECI = [0, -bias, -4*sigma] the total CE
energy of a configuration equals -log p̃(x) for

    log p̃(x) = sigma * x^T A x + bias * sum_i x_i,

and working in natural units (temperature 1, k_B 1) makes the Boltzmann
weight exp(-E/kT) = exp(-E) = p̃(x) — the sampler then targets our
distribution with no further conversion. Spin convention: Au = +1, Ag = -1.

Three ensembles, one per constraint leg:

* soft  -> ``VCSGCEnsemble``. mchammer's variance-constrained penalty enters
  the acceptance exponent as -kappa * N * (c + phi/2)^2; at kappa = lambda,
  phi_Au = -2 * c_target this equals our -lambda * d * (c - c_target)^2
  identically (N = d), giving std(c) = 1/sqrt(2*lambda*d).
* hard  -> ``CanonicalEnsemble``. Swap moves preserve composition exactly:
  the hard constraint is enforced by the move set, not a penalty — the
  practitioner counterpart of Kawasaki dynamics.
* none  -> ``SemiGrandCanonicalEnsemble`` at Delta-mu = 0. Free
  single-site flips with no penalty and no conserved composition, so it
  targets p̃(x) itself: the practitioner counterpart of the
  UNCONSTRAINED leg, completing the one-package arc SGC / VC-SGC /
  Canonical across the three results chapters.

Timing discipline: ``wall_seconds_run`` times the MC loop alone (setup —
cluster-space construction, calculator build — is recorded separately and
must never be folded into per-sample cost). Statistical efficiency comes
from the Sokal integrated autocorrelation time on each observable trace, so
every runner reports seconds per *effective* sample: the only currency in
which a CPU MCMC sweep and a GPU amortised sampler can be compared honestly.
There is no NFE analogue for an mchammer sweep, so wall-clock carries the
whole comparison.
"""

import socket
import time

import numpy as np
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import (
    CanonicalEnsemble, SemiGrandCanonicalEnsemble, VCSGCEnsemble)

from discrete_flow_sampler.diagnostics.metrics import integrated_autocorr

# Natural units: exp(-E/kT) = exp(-E) = p̃(x).
NATURAL_TEMPERATURE = 1.0
NATURAL_BOLTZMANN = 1.0

# Fraction of recorded frames discarded as burn-in before any statistic is
# computed. One third is deliberately generous at these chain lengths; tau_int
# is measured on what remains, so under-discarding shows up in the numbers
# rather than silently biasing them.
BURN_IN_FRACTION = 1.0 / 3.0

_UP_SYMBOL = "Au"      # spin +1
_DOWN_SYMBOL = "Ag"    # spin -1


def ising_cluster_expansion(sigma: float, bias: float = 0.0):
    """Binary CE whose total energy equals -log p̃ of the IsingTarget.

    Single-layer cell with a 10-unit z vacuum gap and a pair cutoff in
    (1, sqrt(2)): exactly one NN pair orbit (4 neighbours/site), no spurious
    z-image or second-neighbour bonds.
    """
    primitive = Atoms(
        _UP_SYMBOL,
        positions=[(0, 0, 0)],
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 10]],
        pbc=True,
    )
    cluster_space = ClusterSpace(
        primitive, cutoffs=[1.1], chemical_symbols=[_UP_SYMBOL, _DOWN_SYMBOL]
    )
    expansion = ClusterExpansion(
        cluster_space, parameters=[0.0, -bias, -4.0 * sigma]
    )
    return primitive, cluster_space, expansion


def ising_supercell(primitive: Atoms, D: int) -> Atoms:
    """D x D torus of the single-site primitive cell."""
    return primitive.repeat((D, D, 1))


def spins_to_symbols(spins) -> list[str]:
    """Map a ±1 spin vector to chemical symbols (Au = +1, Ag = -1)."""
    return [_UP_SYMBOL if spin > 0 else _DOWN_SYMBOL for spin in np.asarray(spins)]


def atoms_to_spins(symbols) -> np.ndarray:
    """Inverse of :func:`spins_to_symbols`."""
    return np.array([1.0 if s == _UP_SYMBOL else -1.0 for s in symbols])


def vcsgc_parameters(penalty_strength: float, target_composition: float) -> dict:
    """The soft-target dictionary: kappa = lambda, phi_up = -2 * c_target.

    mchammer's VC-SGC exponent is -kappa * N * (c + phi/2)^2. Substituting
    phi = -2 * c_target gives -kappa * N * (c - c_target)^2, which is our
    penalty -lambda * d * (c - c_target)^2 exactly when kappa = lambda
    (N = d). No approximation is involved; a sign or factor error here would
    sample a wrong-but-plausible ensemble, which is why the identity is
    pinned by test over the full range of c.
    """
    return {
        "kappa": penalty_strength,
        "phis": {_UP_SYMBOL: -2.0 * target_composition},
    }


def _composition_initialised_supercell(
    primitive: Atoms, D: int, target_composition: float, seed: int
) -> tuple[Atoms, int]:
    """Supercell started AT the (quantised) target composition.

    Starting on-target shortens burn-in for VC-SGC and is mandatory for the
    canonical ensemble, where swap moves make the initial composition the
    composition forever. Quantisation to round(c * N) is the same rule the
    hard DNFS leg uses: N_up must be an integer or the slice is empty.
    """
    supercell = ising_supercell(primitive, D)
    n_sites = len(supercell)
    n_up = int(round(target_composition * n_sites))
    spins = np.array([1.0] * n_up + [-1.0] * (n_sites - n_up))
    np.random.default_rng(seed).shuffle(spins)
    supercell.set_chemical_symbols(spins_to_symbols(spins))
    return supercell, n_up


def _observable_stats(
    trace: np.ndarray, data_write_interval: int, wall_seconds_run: float
) -> dict:
    """tau_int / ESS / cost-per-effective-sample for one scalar trace.

    tau_int comes out in units of recorded frames; ``tau_int_steps`` converts
    back to raw MC trial moves through the thinning factor. ESS uses the
    ESS = n / tau_int convention (tau_int = 1 for an i.i.d. trace), matching
    the diagnostics module used everywhere else in the repo.
    """
    tau_int_frames = integrated_autocorr(trace)
    effective_samples = len(trace) / tau_int_frames
    return {
        "mean": float(np.mean(trace)),
        "std": float(np.std(trace)),
        "n_frames": int(len(trace)),
        "tau_int_frames": float(tau_int_frames),
        "tau_int_steps": float(tau_int_frames * data_write_interval),
        "ess": float(effective_samples),
        "seconds_per_effective_sample": float(wall_seconds_run / effective_samples),
    }


def _base_summary(
    ensemble_name: str,
    D: int,
    sigma: float,
    bias: float,
    n_steps: int,
    seed: int,
    data_write_interval: int,
    wall_seconds_setup: float,
    wall_seconds_run: float,
) -> dict:
    return {
        "engine": "mchammer",
        "ensemble": ensemble_name,
        "D": D,
        "d": D * D,
        "sigma": sigma,
        "bias": bias,
        "n_steps": n_steps,
        "seed": seed,
        "data_write_interval": data_write_interval,
        "burn_in_fraction": BURN_IN_FRACTION,
        "wall_seconds_setup": wall_seconds_setup,
        "wall_seconds_run": wall_seconds_run,
        "steps_per_second": n_steps / wall_seconds_run,
        "hostname": socket.gethostname(),
    }


def _post_burn_in(frame_trace: np.ndarray) -> np.ndarray:
    return frame_trace[int(len(frame_trace) * BURN_IN_FRACTION):]


def run_vcsgc(
    D: int,
    sigma: float,
    penalty_strength: float,
    target_composition: float,
    n_steps: int,
    seed: int,
    bias: float = 0.0,
    data_write_interval: int = 100,
    record_spins: bool = False,
) -> dict:
    """One VC-SGC chain at the soft-target operating point, with timing.

    ``record_spins`` additionally returns the full +-1 configuration at every
    ensemble-data row (post burn-in, int8) under ``traces["spins"]``. The
    house evaluation table scores per-site magnetisation and correlation
    profiles against the VC-SGC reference, which the scalar composition and
    potential traces cannot supply. Capture follows ``run_canonical_probe``:
    drive ``ensemble.run`` one write interval at a time and read the structure
    back between calls, so frame k is the state mchammer's row k describes
    (row-major D x D atom order, see the probe's docstring). The RNG stream is
    untouched by chunking, so the chain is identical to the one-shot run; the
    default keeps the cheaper one-shot path for callers that only need traces.
    """
    if record_spins and n_steps % data_write_interval != 0:
        raise ValueError(
            f"n_steps={n_steps} must be a multiple of "
            f"data_write_interval={data_write_interval} to align spin frames "
            "with ensemble-data rows."
        )
    setup_start = time.perf_counter()
    primitive, _, expansion = ising_cluster_expansion(sigma, bias)
    supercell, _ = _composition_initialised_supercell(
        primitive, D, target_composition, seed
    )
    calculator = ClusterExpansionCalculator(supercell, expansion)
    ensemble = VCSGCEnsemble(
        supercell,
        calculator,
        temperature=NATURAL_TEMPERATURE,
        boltzmann_constant=NATURAL_BOLTZMANN,
        **vcsgc_parameters(penalty_strength, target_composition),
        ensemble_data_write_interval=data_write_interval,
        random_seed=seed,
    )
    wall_seconds_setup = time.perf_counter() - setup_start

    n_sites = len(supercell)
    read_spins = lambda: atoms_to_spins(
        ensemble.structure.get_chemical_symbols()
    ).astype(np.int8)
    run_start = time.perf_counter()
    if record_spins:
        # Frame k = the state at trial step k * interval, matching row k of the
        # data container (written at step 0 and after every interval).
        spin_frames = np.empty((n_steps // data_write_interval + 1, n_sites), np.int8)
        for k in range(len(spin_frames) - 1):
            spin_frames[k] = read_spins()
            ensemble.run(data_write_interval)
        spin_frames[-1] = read_spins()
    else:
        ensemble.run(n_steps)
    wall_seconds_run = time.perf_counter() - run_start

    data = ensemble.data_container.data
    composition_trace = _post_burn_in(data[f"{_UP_SYMBOL}_count"].values / n_sites)
    potential_trace = _post_burn_in(data["potential"].values)

    summary = _base_summary(
        "vcsgc", D, sigma, bias, n_steps, seed, data_write_interval,
        wall_seconds_setup, wall_seconds_run,
    )
    summary["penalty_strength"] = penalty_strength
    summary["target_composition"] = target_composition
    summary["observables"] = {
        "composition": _observable_stats(
            composition_trace, data_write_interval, wall_seconds_run
        ),
        "potential": _observable_stats(
            potential_trace, data_write_interval, wall_seconds_run
        ),
    }
    # Raw post-burn-in traces ride along (numpy, not JSON-safe) so callers can
    # persist them for re-analysis; strip before serialising.
    summary["traces"] = {
        "composition": composition_trace,
        "potential": potential_trace,
    }
    if record_spins:
        summary["traces"]["spins"] = _post_burn_in(spin_frames)
    return summary


def run_sgc(
    D: int,
    sigma: float,
    initial_composition: float,
    n_steps: int,
    seed: int,
    bias: float = 0.0,
    data_write_interval: int = 100,
    record_spins: bool = False,
) -> dict:
    """One semi-grand-canonical chain at Delta-mu = 0: the unconstrained leg.

    Equal chemical potentials mean the species term drops out of the
    acceptance exponent entirely, leaving free single-site flips against the
    CE energy alone — so this chain targets p̃(x), the same distribution the
    unconstrained DNFS sampler targets, and the two are directly comparable
    in the house evaluation table. ``chemical_potentials`` is required by
    mchammer and carries no default; passing an unequal pair here would
    silently sample a field-biased Ising model.

    ``initial_composition`` is only a STARTING point, unlike the same
    argument to ``run_canonical`` (where swap moves make it the composition
    forever) or ``run_vcsgc`` (where the penalty pins it): here the
    composition floats, and 0.5 is the neutral disordered start. At sigma_c
    that start sits at the Z2 symmetric point and the chain magnetises into
    one sector or the other during burn-in, independently per seed — which
    is why the table's row pools independent seeds rather than extending one
    chain.

    ``record_spins`` follows ``run_vcsgc`` exactly: chunked driving that
    leaves the RNG stream untouched, so the chain is identical to the
    one-shot run and frame k is the state mchammer's row k describes. The
    house table's magnetisation and correlation PROFILES need configurations;
    the scalar traces cannot supply them.
    """
    if record_spins and n_steps % data_write_interval != 0:
        raise ValueError(
            f"n_steps={n_steps} must be a multiple of "
            f"data_write_interval={data_write_interval} to align spin frames "
            "with ensemble-data rows."
        )
    setup_start = time.perf_counter()
    primitive, _, expansion = ising_cluster_expansion(sigma, bias)
    supercell, _ = _composition_initialised_supercell(
        primitive, D, initial_composition, seed
    )
    calculator = ClusterExpansionCalculator(supercell, expansion)
    ensemble = SemiGrandCanonicalEnsemble(
        supercell,
        calculator,
        temperature=NATURAL_TEMPERATURE,
        boltzmann_constant=NATURAL_BOLTZMANN,
        chemical_potentials={_UP_SYMBOL: 0.0, _DOWN_SYMBOL: 0.0},
        ensemble_data_write_interval=data_write_interval,
        random_seed=seed,
    )
    wall_seconds_setup = time.perf_counter() - setup_start

    n_sites = len(supercell)
    read_spins = lambda: atoms_to_spins(
        ensemble.structure.get_chemical_symbols()
    ).astype(np.int8)
    run_start = time.perf_counter()
    if record_spins:
        spin_frames = np.empty((n_steps // data_write_interval + 1, n_sites), np.int8)
        for k in range(len(spin_frames) - 1):
            spin_frames[k] = read_spins()
            ensemble.run(data_write_interval)
        spin_frames[-1] = read_spins()
    else:
        ensemble.run(n_steps)
    wall_seconds_run = time.perf_counter() - run_start

    data = ensemble.data_container.data
    composition_trace = _post_burn_in(data[f"{_UP_SYMBOL}_count"].values / n_sites)
    potential_trace = _post_burn_in(data["potential"].values)

    summary = _base_summary(
        "sgc", D, sigma, bias, n_steps, seed, data_write_interval,
        wall_seconds_setup, wall_seconds_run,
    )
    summary["initial_composition"] = initial_composition
    summary["observables"] = {
        "composition": _observable_stats(
            composition_trace, data_write_interval, wall_seconds_run
        ),
        "potential": _observable_stats(
            potential_trace, data_write_interval, wall_seconds_run
        ),
    }
    summary["traces"] = {
        "composition": composition_trace,
        "potential": potential_trace,
    }
    if record_spins:
        summary["traces"]["spins"] = _post_burn_in(spin_frames)
    return summary


def run_canonical_probe(
    D: int,
    sigma: float,
    initial_spins: np.ndarray,
    n_proposals: int,
    snapshot_interval: int,
    seed: int,
    bias: float = 0.0,
) -> dict:
    """Canonical (non-local unlike-pair swap) chain recording full ±1 spin
    snapshots at an exact proposal interval — the mixing probe's non-local
    variant carrier. Differs from `run_canonical` in three ways:

    * explicit initial state: the supercell's symbols are set from
      ``initial_spins`` before the ensemble is built, so reference chains can
      be seeded IN a chosen phi mode (phase-separated left/right) rather than
      at a random slice point;
    * exact-interval snapshot capture: ``ensemble.run(snapshot_interval)`` is
      driven in a loop and the structure read back between calls — the
      simplest capture whose intervals are exact in trial steps (mchammer's
      trajectory observer is bypassed; its rows would also cost memory);
    * raw arrays + counters are returned instead of summary statistics: the
      probe's analysis stage owns burn-in and R̂ decisions, so nothing is
      discarded here.

    Site indexing: spin index i == atom index i of the D x D x 1 ``repeat``
    supercell. ASE's repeat enumerates the two in-plane lattice vectors
    lexicographically, so reading atom order as a row-major D x D flattening
    reproduces the torus adjacency exactly (neighbours are i±1 and i±D with
    wraparound); energy and both correlation observables are invariant to
    which in-plane axis plays "rows", and the phi half-split axis is
    self-consistent because the SAME indexing writes the initial state and
    reads every snapshot. A scrambled atom order would silently corrupt all
    spatial observables — which is why ``potential_per_snapshot`` (mchammer's
    own exactly-recomputed CE energy at each snapshot step) rides along: it
    must equal -sigma * x^T A x recomputed from the returned snapshots, and
    the probe's tests pin that equality.

    snapshots[k] is the state after k * snapshot_interval proposals
    (snapshots[0] = the initial state, matching the numba runners'
    record-at-top convention); the state after the final interval is not
    recorded. ``n_proposals`` must be a multiple of ``snapshot_interval``:
    acceptance is recovered from mchammer's per-interval ``acceptance_ratio``
    rows, and a trailing partial interval would silently undercount it.

    Returns a dict with ``snapshots`` (int8, [n_kept, d]), ``n_proposals``
    (read back from ``ensemble.step`` — the exact trial-step currency),
    ``n_accepted``, ``potential_per_snapshot``, ``composition_is_constant``,
    ``wall_seconds_setup`` and ``wall_seconds_run`` (setup — cluster-space and
    calculator construction — must never be folded into per-proposal cost).
    """
    d = D * D
    initial_spins = np.asarray(initial_spins)
    if initial_spins.shape != (d,):
        raise ValueError(
            f"initial_spins must have shape ({d},), got {initial_spins.shape}"
        )
    if n_proposals % snapshot_interval != 0:
        raise ValueError(
            f"n_proposals={n_proposals} must be a multiple of "
            f"snapshot_interval={snapshot_interval}: acceptance counting "
            "relies on full ensemble-data write intervals."
        )

    setup_start = time.perf_counter()
    primitive, _, expansion = ising_cluster_expansion(sigma, bias)
    supercell = ising_supercell(primitive, D)
    supercell.set_chemical_symbols(spins_to_symbols(initial_spins))
    calculator = ClusterExpansionCalculator(supercell, expansion)
    ensemble = CanonicalEnsemble(
        supercell,
        calculator,
        temperature=NATURAL_TEMPERATURE,
        boltzmann_constant=NATURAL_BOLTZMANN,
        ensemble_data_write_interval=snapshot_interval,
        trajectory_write_interval=np.inf,     # snapshots are captured directly
        dc_filename=None,
        random_seed=seed,
    )
    wall_seconds_setup = time.perf_counter() - setup_start

    n_snapshots = n_proposals // snapshot_interval
    snapshots = np.empty((n_snapshots, d), dtype=np.int8)
    run_start = time.perf_counter()
    for k in range(n_snapshots):
        snapshots[k] = atoms_to_spins(
            ensemble.structure.get_chemical_symbols()
        ).astype(np.int8)
        ensemble.run(snapshot_interval)
    wall_seconds_run = time.perf_counter() - run_start

    data = ensemble.data_container.data
    recorded_steps = data["mctrial"].to_numpy()
    expected_steps = np.arange(n_snapshots + 1) * snapshot_interval
    if not np.array_equal(recorded_steps, expected_steps):
        # Defensive: the acceptance/potential bookkeeping below assumes
        # mchammer writes ensemble data at exactly every write interval; a
        # version drift in that cadence must fail loudly, not skew counters.
        raise RuntimeError(
            "unexpected mchammer ensemble-data cadence: "
            f"mctrials {recorded_steps[:5]}... vs expected multiples of "
            f"{snapshot_interval}"
        )
    # Row at step k*T holds the acceptance count over ((k-1)T, kT] divided by
    # T, so the interval-weighted sum over all rows is the total accepted.
    n_accepted = int(round(
        float(data["acceptance_ratio"].to_numpy().sum()) * snapshot_interval
    ))
    # Rows 0..n_kept-1 are written at the same trial step as snapshots
    # 0..n_kept-1; the final row (post-run state) has no snapshot.
    potential_per_snapshot = data["potential"].to_numpy()[:n_snapshots]

    final_spins = atoms_to_spins(ensemble.structure.get_chemical_symbols())
    composition_is_constant = int(np.sum(final_spins > 0)) == int(
        np.sum(initial_spins > 0)
    )

    return {
        "snapshots": snapshots,
        "n_proposals": int(ensemble.step),
        "n_accepted": n_accepted,
        "potential_per_snapshot": potential_per_snapshot,
        "composition_is_constant": composition_is_constant,
        "wall_seconds_setup": wall_seconds_setup,
        "wall_seconds_run": wall_seconds_run,
    }


def run_canonical(
    D: int,
    sigma: float,
    target_composition: float,
    n_steps: int,
    seed: int,
    bias: float = 0.0,
    data_write_interval: int = 100,
) -> dict:
    """One canonical (swap-move) chain at fixed composition, with timing."""
    setup_start = time.perf_counter()
    primitive, _, expansion = ising_cluster_expansion(sigma, bias)
    supercell, n_up = _composition_initialised_supercell(
        primitive, D, target_composition, seed
    )
    n_sites = len(supercell)
    calculator = ClusterExpansionCalculator(supercell, expansion)
    ensemble = CanonicalEnsemble(
        supercell,
        calculator,
        temperature=NATURAL_TEMPERATURE,
        boltzmann_constant=NATURAL_BOLTZMANN,
        ensemble_data_write_interval=data_write_interval,
        random_seed=seed,
    )
    wall_seconds_setup = time.perf_counter() - setup_start

    run_start = time.perf_counter()
    ensemble.run(n_steps)
    wall_seconds_run = time.perf_counter() - run_start

    data = ensemble.data_container.data
    potential_trace = _post_burn_in(data["potential"].values)

    # Swap moves make composition invariant by construction; verify rather
    # than trust, because a silently-wrong move set (e.g. a flip ensemble
    # picked by mistake) would invalidate every number downstream.
    final_spins = atoms_to_spins(ensemble.structure.get_chemical_symbols())
    composition_is_constant = int(np.sum(final_spins > 0)) == n_up

    summary = _base_summary(
        "canonical", D, sigma, bias, n_steps, seed, data_write_interval,
        wall_seconds_setup, wall_seconds_run,
    )
    summary["target_composition"] = target_composition
    summary["composition_realised"] = n_up / n_sites
    summary["composition_is_constant"] = composition_is_constant
    summary["observables"] = {
        "potential": _observable_stats(
            potential_trace, data_write_interval, wall_seconds_run
        ),
    }
    summary["traces"] = {"potential": potential_trace}
    return summary
