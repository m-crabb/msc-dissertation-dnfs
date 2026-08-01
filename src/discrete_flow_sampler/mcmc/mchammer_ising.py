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

Two ensembles, matching the two constraint legs:

* soft  -> ``VCSGCEnsemble``. mchammer's variance-constrained penalty enters
  the acceptance exponent as -kappa * N * (c + phi/2)^2; at kappa = lambda,
  phi_Au = -2 * c_target this equals our -lambda * d * (c - c_target)^2
  identically (N = d), giving std(c) = 1/sqrt(2*lambda*d).
* hard  -> ``CanonicalEnsemble``. Swap moves preserve composition exactly:
  the hard constraint is enforced by the move set, not a penalty — the
  practitioner counterpart of Kawasaki dynamics.

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
from mchammer.ensembles import CanonicalEnsemble, VCSGCEnsemble

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
) -> dict:
    """One VC-SGC chain at the soft-target operating point, with timing."""
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

    run_start = time.perf_counter()
    ensemble.run(n_steps)
    wall_seconds_run = time.perf_counter() - run_start

    data = ensemble.data_container.data
    n_sites = len(supercell)
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
    return summary


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
