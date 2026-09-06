"""Certified Kawasaki reference sample set for the 16x16 fixed-composition
Ising model at the critical coupling.

Produces an output directory (default results/kawasaki_ref_d256_sc/) containing
STORED DRAWS (samples.pt), not just scalar statistics, for downstream
observable-decoupling analysis and sample-montage figures.

The coupling sigma and the output directory are command-line parameters whose
defaults reproduce the original certified reference exactly. Sigma has to be a
parameter because a reference set is a reference only for the sigma it was
generated at: every certified number below (nn-correlation, energy per site,
the Gelman-Rubin values) is a property of p(x) proportional to
exp(sigma * x^T A x) at that one sigma. Judging model samples trained at one
sigma against a reference drawn at another injects a systematic shift of size
d<nn>/dsigma * delta-sigma into every comparison, which no amount of extra
reference sampling removes. Making sigma explicit at the call site is what
stops a stale hard-coded value from silently becoming that systematic.

What makes this a CERTIFIED reference rather than just "some MCMC output":

* Matched energy convention. The chain accepts on the exact unnormalised
  target log-density log p(x) = sigma * x^T A x used by
  FixedCompositionIsingTarget.base_log_prob (torus adjacency A with each
  undirected edge counted twice, bias = 0 — the bias term is swap-invariant
  at fixed composition anyway). The script cross-checks the numba chain's
  energy against the target class on the stored samples before certifying.
* Exact hard constraint. The swap move set conserves composition by
  construction; every stored draw is asserted to have exactly 128 up-spins.
* Measured autocorrelation, then thinning. The integrated autocorrelation
  time of the nearest-neighbour-correlation observable is measured per chain
  (Sokal windowing) on a densely-recorded trace, and the stored set is thinned
  at >= 2x the worst-chain tau, so consecutive stored draws are approximately
  independent and plain stderr on the set is honest.
* Multi-start mode balance. At fixed c = 0.5 the Z2 spin-flip maps the slice
  to itself and the near-critical system phase-separates, so chains are seeded
  from BOTH ordered (phase-separated, both orientations) and random starts;
  certification includes Gelman-Rubin across all chains and an explicit
  ordered-vs-random start-condition agreement check. A chain stuck in its
  starting basin would fail these, not silently bias the set.
* External anchor. The set's nn-correlation mean must land within
  0.578756 +- 0.004, the equilibrium value measured with the independent
  mchammer engine at exact sigma_c for this lattice/composition (see the
  constant's comment below for provenance). Failure is reported loudly and
  the data kept for inspection, never deleted.

Sampler machinery is reused from discrete_flow_sampler.mcmc.kawasaki (the
non-local unlike-pair swap chain, the deliberately strong practitioner
baseline — NOT the slow-mixing local variant); observables and diagnostics
from discrete_flow_sampler.diagnostics.metrics.

Run:  pixi run -e default python -m experiments.constrained_hard_03.generate_kawasaki_reference_d256
      (add --sigma / --out-dir to generate a sigma-matched twin elsewhere)
"""
import argparse
import json
import math
import socket
import time
from pathlib import Path

import numpy as np
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    gelman_rubin,
    integrated_autocorr,
    nn_correlation,
    split_half_gelman_rubin,
)
from discrete_flow_sampler.mcmc.kawasaki import (
    init_phase_separated,
    init_random_at_composition,
    run_nonlocal_swap_chain_snapshots,
)
from discrete_flow_sampler.targets.ising import (
    SIGMA_C, FixedCompositionIsingTarget)

# `main` sets these defaults from --lattice-side before generating chains.
# build_initial_spins and run_one_chain read this shared size; imported use
# at multiple sizes would need explicit size arguments. The tested
# external_nn_anchor helper already takes its lattice side explicitly.
LATTICE_SIDE = 16
N_SITES = LATTICE_SIDE * LATTICE_SIDE          # 256
DEFAULT_SIGMA = SIGMA_C    # exact sigma_c since the s58 migration; the archived
                           # d256 reference dumps were generated at legacy 0.22305
                           # and pair ONLY with the pre-migration hard runs
TARGET_COMPOSITION = 0.5                       # 128 up / 128 down, exact
N_CHAINS = 8                                   # 2 ordered-left, 2 ordered-right, 4 random
BURN_IN_SWEEPS = 100_000                       # ~6000x the measured tau — cheap at numba speed
SAMPLING_SWEEPS = 102_400
RECORD_EVERY_SWEEPS = 2                        # dense trace for the tau measurement
SEED_BASE = 3000                               # disjoint from earlier probe seed ranges (1000/2000)
MIN_STORED_SAMPLES = 5_000
# External anchor at exact SIGMA_C, measured 2026-08-26 with the independent
# mchammer engine (icet CanonicalEnsemble, unlike-pair swaps, same CE
# embedding): 4 seeds x 25M swap trials, ~21600 effective samples,
# nn = 0.578756 +- 0.000391 (see results/kawasaki_ref_d256_s220/
# mchammer_anchor_crosscheck.json). Tolerance = max(0.004, 3 x stderr) = 0.004.
# The previous anchor 0.588 +- 0.004 was measured at LEGACY sigma 0.22305;
# holding an exact-sigma_c set to it injected the docstring's own
# d<nn>/dsigma systematic (~0.009 for delta-sigma 0.0027) and failed a set
# whose internal certification was immaculate.
CERTIFICATION_NN_TARGET = 0.578756
CERTIFICATION_NN_TOLERANCE = 0.004
THINNING_SAFETY_FACTOR = 2.0                   # thin at 2x worst-chain tau, not 1x
DEFAULT_OUT_DIR = Path("results/kawasaki_ref_d256_sc")

# Chain start conditions: mode-balanced ordered starts (phase-separated in each
# Z2/orientation basin) plus neutral random starts, so between-chain agreement
# is evidence of mixing rather than of shared initialisation.
CHAIN_START_CONDITIONS = [
    ("ordered", 0), ("ordered", 0),            # phase-separated, +domain left
    ("ordered", 1), ("ordered", 1),            # phase-separated, +domain right
    ("random", None), ("random", None), ("random", None), ("random", None),
]


def build_initial_spins(init_kind, init_side, seed):
    if init_kind == "random":
        return init_random_at_composition(
            N_SITES, TARGET_COMPOSITION, np.random.default_rng(seed)
        )
    return init_phase_separated(LATTICE_SIDE, init_side)


def run_one_chain(chain_index, sigma):
    """Burn in, then sample with dense snapshot recording. Returns
    (snapshots int8 (n_record, d), meta dict).

    sigma is threaded through rather than read from a module constant so that
    burn-in and sampling provably use the SAME coupling as the certification
    statistics computed downstream — a reference whose burn-in equilibrated at
    one sigma and whose draws were accepted at another would certify cleanly
    and still be wrong.
    """
    init_kind, init_side = CHAIN_START_CONDITIONS[chain_index]
    burn_seed = SEED_BASE + chain_index
    sampling_seed = SEED_BASE + 100 + chain_index

    spins = build_initial_spins(init_kind, init_side, burn_seed)
    burn_proposals = BURN_IN_SWEEPS * N_SITES
    t0 = time.perf_counter()
    _, spins, _ = run_nonlocal_swap_chain_snapshots(
        spins, LATTICE_SIDE, sigma, burn_proposals, burn_seed,
        thin=burn_proposals,                   # records only the (discarded) initial state
    )
    sampling_proposals = SAMPLING_SWEEPS * N_SITES
    record_every_proposals = RECORD_EVERY_SWEEPS * N_SITES
    snapshots, _, n_accepted = run_nonlocal_swap_chain_snapshots(
        spins, LATTICE_SIDE, sigma, sampling_proposals, sampling_seed,
        thin=record_every_proposals,
    )
    wall_seconds = time.perf_counter() - t0

    meta = {
        "chain_index": chain_index,
        "init_kind": init_kind,
        "init_side": init_side,
        "burn_seed": burn_seed,
        "sampling_seed": sampling_seed,
        "burn_in_sweeps": BURN_IN_SWEEPS,
        "sampling_sweeps": SAMPLING_SWEEPS,
        "record_every_sweeps": RECORD_EVERY_SWEEPS,
        "n_recorded": int(snapshots.shape[0]),
        "acceptance_rate": n_accepted / sampling_proposals,
        "wall_seconds": wall_seconds,
    }
    return snapshots, meta


def standard_error(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Generate a certified Kawasaki reference sample set for the "
                     "16x16 fixed-composition Ising model."),
    )
    parser.add_argument(
        "--sigma", type=float, default=DEFAULT_SIGMA,
        help=("Coupling the reference chain is simulated at. A reference set is "
              "a reference only for the sigma it was generated at: its "
              "nn-correlation and energy are properties of "
              "exp(sigma * x^T A x), so comparing model samples trained at "
              "sigma_model against a reference drawn at sigma_ref carries a "
              "systematic of order d<nn>/dsigma * (sigma_ref - sigma_model) "
              "that more sampling cannot average away. Set this to the sigma "
              "the model under judgement was trained at. "
              f"(default: {DEFAULT_SIGMA}, the project's sigma_c)"),
    )
    parser.add_argument(
        "--lattice-side", type=int, default=LATTICE_SIDE,
        help=("Edge D of the DxD torus; the reference is generated at d = D^2 "
              "sites and half-filling. A reference is a reference only for the "
              "LATTICE it was generated at as well as the sigma -- the "
              "nn-correlation and energy per site both depend on D -- so this "
              "must match the runs under judgement. The external nn anchor is "
              "sigma_c- AND d256-specific and is skipped automatically off "
              f"sigma_c. (default: {LATTICE_SIDE})"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=("Directory for samples.pt, provenance.json and certification.json. "
              "Give each sigma its own directory: overwriting an existing "
              "reference in place would invalidate every number already quoted "
              "from it, with nothing in the filename to reveal that it moved. "
              f"(default: {DEFAULT_OUT_DIR})"),
    )
    return parser.parse_args()


def external_nn_anchor(sigma, lattice_side):
    """The mchammer-measured 0.578756 +- 0.004 external anchor, or None off
    (sigma_c AND d256).

    The anchor is the independently measured equilibrium nn-correlation AT
    sigma_c ON the 16x16 lattice; it is a property of that one (coupling,
    size) pair. Off sigma_c it would fail spuriously through d<nn>/dsigma;
    at another lattice it would mis-certify through finite-size effects,
    which PEAK at criticality — the d400 sigma_c reference (2026-08-30) is
    the case that forced the second gate. Without the anchor the
    certification rests on the internal checks alone (Gelman-Rubin,
    start-condition agreement) and records that no external anchor exists.
    """
    if sigma == SIGMA_C and lattice_side == 16:
        return CERTIFICATION_NN_TARGET, CERTIFICATION_NN_TOLERANCE
    return None, None


def main():
    args = parse_args()
    sigma, out_dir = args.sigma, args.out_dir
    # Rebound HERE, before the target, the chains or any statistic touch them.
    # Every downstream use reads the module constant, so this is the single
    # point at which the run's lattice is fixed.
    global LATTICE_SIDE, N_SITES
    LATTICE_SIDE = args.lattice_side
    N_SITES = LATTICE_SIDE * LATTICE_SIDE
    nn_anchor, nn_tolerance = external_nn_anchor(sigma, LATTICE_SIDE)

    wall_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"D = {LATTICE_SIDE} (d = {N_SITES}), sigma = {sigma!r} -> {out_dir}/")

    target = FixedCompositionIsingTarget(
        LATTICE_SIDE, sigma, TARGET_COMPOSITION
    )
    adjacency = target.A

    dense_snapshots = []
    chain_metas = []
    for chain_index in range(N_CHAINS):
        snapshots, meta = run_one_chain(chain_index, sigma)
        dense_snapshots.append(snapshots)
        chain_metas.append(meta)
        print(f"chain {chain_index} ({meta['init_kind']}"
              f"{'' if meta['init_side'] is None else meta['init_side']}): "
              f"{meta['n_recorded']} snapshots, "
              f"accept {meta['acceptance_rate']:.3f}, "
              f"{meta['wall_seconds']:.1f}s")

    # --- autocorrelation of the certification observable, per chain ---------
    nn_traces = np.stack([
        nn_correlation(torch.from_numpy(snaps), adjacency).numpy()
        for snaps in dense_snapshots
    ])                                          # (n_chains, n_recorded)
    tau_records_per_chain = [float(integrated_autocorr(t)) for t in nn_traces]
    tau_sweeps_per_chain = [t * RECORD_EVERY_SWEEPS for t in tau_records_per_chain]
    worst_tau_sweeps = max(tau_sweeps_per_chain)

    # Thinning: >= safety factor x the worst-chain tau, expressed as a whole
    # number of recorded snapshots so stored draws are a strict subsample.
    thin_records = max(
        1, math.ceil(THINNING_SAFETY_FACTOR * worst_tau_sweeps / RECORD_EVERY_SWEEPS)
    )
    thinning_sweeps = thin_records * RECORD_EVERY_SWEEPS

    thinned_per_chain = [snaps[::thin_records] for snaps in dense_snapshots]
    thinned_nn_per_chain = [t[::thin_records] for t in nn_traces]
    samples = np.concatenate(thinned_per_chain, axis=0)
    print(f"tau per chain (sweeps): {[f'{t:.1f}' for t in tau_sweeps_per_chain]}")
    print(f"thinning: every {thinning_sweeps} sweeps -> {samples.shape[0]} stored draws")
    if samples.shape[0] < MIN_STORED_SAMPLES:
        raise RuntimeError(
            f"only {samples.shape[0]} draws after thinning at "
            f"{thinning_sweeps} sweeps; lengthen SAMPLING_SWEEPS"
        )

    # --- hard-constraint and energy-convention checks on the stored set -----
    samples_tensor = torch.from_numpy(samples)
    target.assert_on_manifold(samples_tensor.float())
    quadratic_per_site = (
        nn_correlation(samples_tensor, adjacency) * float(adjacency.sum()) / N_SITES
    )
    log_prob_from_target = target.base_log_prob(samples_tensor.float())
    convention_gap = (
        log_prob_from_target - sigma * quadratic_per_site * N_SITES
    ).abs().max().item()
    if convention_gap > 1e-3:
        raise RuntimeError(
            f"chain energy convention disagrees with "
            f"FixedCompositionIsingTarget.base_log_prob (max gap {convention_gap})"
        )

    # --- certification statistics -------------------------------------------
    nn_values = np.concatenate(thinned_nn_per_chain)
    nn_mean = float(nn_values.mean())
    nn_stderr = standard_error(nn_values)      # honest because draws are thinned past tau
    chain_nn_means = [float(t.mean()) for t in thinned_nn_per_chain]
    nn_stderr_between_chains = standard_error(chain_nn_means)

    energy_values = quadratic_per_site.numpy()
    ordered_nn = np.concatenate(
        [t for t, m in zip(thinned_nn_per_chain, chain_metas)
         if m["init_kind"] == "ordered"]
    )
    random_nn = np.concatenate(
        [t for t, m in zip(thinned_nn_per_chain, chain_metas)
         if m["init_kind"] == "random"]
    )
    start_gap = float(ordered_nn.mean() - random_nn.mean())
    start_gap_stderr = math.sqrt(
        standard_error(ordered_nn) ** 2 + standard_error(random_nn) ** 2
    )
    start_gap_z = start_gap / start_gap_stderr

    thinned_nn_stack = np.stack(thinned_nn_per_chain)
    certification = {
        "nn_correlation": {
            "mean": nn_mean,
            "stderr": nn_stderr,
            "stderr_between_chain_means": nn_stderr_between_chains,
            "n_samples": int(len(nn_values)),
            "reference_value": nn_anchor,
            "tolerance": nn_tolerance,
            "within_tolerance": (
                None if nn_anchor is None
                else bool(abs(nn_mean - nn_anchor) <= nn_tolerance)
            ),
            "no_external_anchor_at_this_sigma": nn_anchor is None,
        },
        "energy_per_site": {                   # sigma-free quadratic form x^T A x / d
            "mean": float(energy_values.mean()),
            "stderr": standard_error(energy_values),
            "log_prob_per_site_mean": float(log_prob_from_target.mean()) / N_SITES,
        },
        "multi_chain_agreement": {
            "gelman_rubin_nn_correlation": gelman_rubin(thinned_nn_stack),
            "split_half_gelman_rubin_nn_correlation":
                split_half_gelman_rubin(thinned_nn_stack),
            "per_chain_nn_means": chain_nn_means,
        },
        "start_condition_agreement": {
            "ordered_start_nn_mean": float(ordered_nn.mean()),
            "random_start_nn_mean": float(random_nn.mean()),
            "gap": start_gap,
            "gap_stderr": start_gap_stderr,
            "gap_z_score": start_gap_z,
            "agrees_within_3_sigma": bool(abs(start_gap_z) < 3.0),
        },
        "hard_constraint": {
            "composition": TARGET_COMPOSITION,
            "n_up_spins_exact": target.n_plus_target,
            "all_samples_on_manifold": True,   # assert_on_manifold passed above
        },
        "energy_convention_max_gap_vs_target_class": convention_gap,
        "certified": bool(
            (nn_anchor is None or abs(nn_mean - nn_anchor) <= nn_tolerance)
            and gelman_rubin(thinned_nn_stack) < 1.01
            and abs(start_gap_z) < 3.0
        ),
    }

    provenance = {
        "lattice_side": LATTICE_SIDE,
        "n_sites": N_SITES,
        "sigma": sigma,
        "target_composition": TARGET_COMPOSITION,
        "move_set": "nonlocal unlike-pair Kawasaki swap, Metropolis on sigma * x^T A x",
        "annealing_schedule": None,            # direct simulation at sigma_c; burn-in only
        "n_chains": N_CHAINS,
        "burn_in_sweeps": BURN_IN_SWEEPS,
        "sampling_sweeps_per_chain": SAMPLING_SWEEPS,
        "record_every_sweeps": RECORD_EVERY_SWEEPS,
        "integrated_autocorr_sweeps_per_chain": tau_sweeps_per_chain,
        "worst_chain_tau_sweeps": worst_tau_sweeps,
        "thinning_safety_factor": THINNING_SAFETY_FACTOR,
        "thinning_interval_sweeps": thinning_sweeps,
        "n_stored_samples": int(samples.shape[0]),
        "chains": chain_metas,
        "hostname": socket.gethostname(),
        "total_wall_seconds": time.perf_counter() - wall_start,
    }

    torch.save(samples_tensor, out_dir / "samples.pt")
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out_dir / "certification.json").write_text(json.dumps(certification, indent=2))

    print(json.dumps(certification, indent=2))
    if not certification["certified"]:
        print("CERTIFICATION FAILED — data kept for inspection, see certification.json")
        raise SystemExit(1)
    print(f"CERTIFIED: {samples.shape[0]} draws -> {out_dir}/")


if __name__ == "__main__":
    main()
