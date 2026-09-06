"""8x8 Kawasaki chain runner for the mixing probe: classical reference
chains + competitor chains at the two operating points.

Physical setup: 2D Ising on the D x D torus,
D = 8 (d = 64), spins ±1, fixed composition c = 0.5 (32 up), Kawasaki swap
dynamics with Metropolis acceptance on log p_tilde(x) = sigma * x^T A x.
Operating points: `sc` (sigma = 0.223, the critical coupling matching the
trained cell) and `s010` (sigma = 0.10, the subcritical floor). Observables
are the probe set reused verbatim from demo_4x4.observable_values:
energy (the sigma-free quadratic form x^T A x), nn_correlation,
diagonal_correlation, and phi (left-minus-right half magnetisation, in the
[-1, 1] half-factor convention of diagnostics.metrics).

Two stages:

* reference — mchammer CanonicalEnsemble (NON-local unlike-pair swaps) only,
  8 chains per point, mode-balanced seeding (chains 0-3 phase-separated in
  the phi > 0 mode, 4-7 in the phi < 0 mode). Validity bar: split-half R-hat
  <= 1.01 on every observable, computed after dropping the first half of each
  chain (the same discard the moment rule applies; the full-trace
  R-hat is reported alongside for transparency). On failure the chain length
  is DOUBLED and the point rerun — mechanically, no judgement — up to 3
  doublings, then the failure is reported loudly and the exit code is
  non-zero. Doubling rule: rerun-longer with the SAME seeds, not in-place
  extension. mchammer seeds Python's global RNG at ensemble construction, so
  the doubled chain reproduces the shorter run's trajectory as its prefix and
  extends it — a realization-level extension without keeping worker state
  alive across attempts (in-place extension was rejected as it would pin one
  live ensemble per chain across the whole doubling loop).

* competitor — BOTH variants: `local` (numba nearest-neighbour swap,
  kawasaki.run_local_swap_chain_snapshots) and `nonlocal` (mchammer, the same
  engine as the reference). 8 chains per variant per point: chains 0-3 seeded
  at RANDOM composition-0.5 states (competitors start neutrally), chains 4-5
  in phi mode A and 6-7 in mode B (the mode-seeded exception the coverage
  axis needs); which is which is recorded in meta.json. Everything is
  recorded from step 0 with NO burn-in discard — the analysis stage owns
  burn-in, and the crossover curve needs the full trace. R-hat here is a
  health floor (1.1), reported, never gating.

Currencies: the trial-step currency is the PROPOSAL count, recorded exactly
for both variants (numba: n_steps by construction; mchammer: ensemble.step
read back). For the local variant like-spin bond proposals are identity moves
but still cost one proposal — see run_local_swap_chain_snapshots. Wall-clock
setup (cluster-space/calculator build) is recorded separately from the MC
loop and must never be folded into per-proposal cost.

Seeds: reference = 1000 + point_offset + chain_index, competitor = 2000 +
point_offset + variant_offset + chain_index, with point offsets {sc: 0,
s010: 100} and variant offsets {local: 0, nonlocal: 50} — the ranges are
disjoint, so no two chains anywhere share a numpy/numba/mchammer seed.

Smoke mode (--smoke): D = 4, 2_000 sweeps, 2 chains per seeding mode,
2 workers, output routed to <out>/smoke/ — end-to-end plumbing test only.

Example:
    pixi run -e default python -m experiments.constrained_hard_03.probe_kawasaki_8x8 \\
        --stage reference --smoke
"""

import argparse
import json
import socket
import time
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from discrete_flow_sampler.diagnostics.metrics import split_half_gelman_rubin
from discrete_flow_sampler.mcmc.kawasaki import (
    init_phase_separated,
    init_random_at_composition,
    run_local_swap_chain_snapshots,
)
from discrete_flow_sampler.mcmc.mchammer_ising import run_canonical_probe

OPERATING_POINTS = {"sc": 0.223, "s010": 0.10}
TARGET_COMPOSITION = 0.5
FULL_LATTICE_SIDE = 8
REFERENCE_CHAINS_PER_MODE = 4  # 8 chains: 0-3 mode A, 4-7 mode B
COMPETITOR_RANDOM_CHAINS = 4  # + 2 mode A + 2 mode B = 8 chains
COMPETITOR_MODE_CHAINS = 2
REFERENCE_VALIDITY_RHAT = 1.01  # validity bar, every observable
COMPETITOR_HEALTH_RHAT = 1.1  # reported, never gating
MAX_DOUBLINGS = 3
REFERENCE_SEED_BASE = 1000
COMPETITOR_SEED_BASE = 2000
POINT_SEED_OFFSETS = {"sc": 0, "s010": 100}
VARIANT_SEED_OFFSETS = {"local": 0, "nonlocal": 50}
SMOKE_LATTICE_SIDE = 4
SMOKE_SWEEPS = 2_000
SMOKE_WORKERS = 2
SMOKE_CHAINS_PER_MODE = 2


@dataclass
class ChainSpec:
    """Everything one worker needs to run and persist a single chain."""

    stage: str  # reference | competitor
    variant: str  # local | nonlocal (reference is always nonlocal)
    point: str  # sc | s010
    sigma: float
    lattice_side: int
    chain_index: int
    seed: int
    init_kind: str  # random | phase_separated
    init_side: int | None  # 0 (phi > 0 mode) / 1 (phi < 0 mode) / None
    sweeps: int
    snapshot_every_sweeps: int
    out_dir: str


def build_initial_spins(spec: ChainSpec) -> np.ndarray:
    d = spec.lattice_side**2
    if spec.init_kind == "random":
        return init_random_at_composition(
            d, TARGET_COMPOSITION, np.random.default_rng(spec.seed)
        )
    return init_phase_separated(spec.lattice_side, spec.init_side)


def run_chain_worker(spec: ChainSpec) -> dict:
    """Run one chain, write snapshots.npz + meta.json, return the meta dict.

    Runs in a spawned subprocess: keep imports light (no torch here — the
    observable computations happen once, in the parent, over the saved
    snapshots). The local variant's wall seconds include numba JIT on the
    first call in each worker (cache=True amortises across runs); wall-clock
    is context only — the probe currency is the proposal count.
    """
    D = spec.lattice_side
    d = D * D
    n_proposals = spec.sweeps * d
    snapshot_interval = spec.snapshot_every_sweeps * d
    initial_spins = build_initial_spins(spec)

    if spec.variant == "local":
        run_start = time.perf_counter()
        snapshots, _, n_accepted = run_local_swap_chain_snapshots(
            initial_spins.copy(),
            D,
            spec.sigma,
            n_proposals,
            spec.seed,
            snapshot_interval,
        )
        wall_seconds_run = time.perf_counter() - run_start
        wall_seconds_setup = 0.0
        potential_per_snapshot = None
        engine = "numba_local_nn_swap"
    else:
        result = run_canonical_probe(
            D=D,
            sigma=spec.sigma,
            initial_spins=initial_spins,
            n_proposals=n_proposals,
            snapshot_interval=snapshot_interval,
            seed=spec.seed,
        )
        assert result["n_proposals"] == n_proposals
        assert result["composition_is_constant"]
        snapshots = result["snapshots"]
        n_accepted = result["n_accepted"]
        wall_seconds_run = result["wall_seconds_run"]
        wall_seconds_setup = result["wall_seconds_setup"]
        potential_per_snapshot = result["potential_per_snapshot"]
        engine = "mchammer_canonical_nonlocal_swap"

    chain_dir = Path(spec.out_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "spins": snapshots,
        "sigma": spec.sigma,
        "seed": spec.seed,
        "D": D,
        "snapshot_interval_proposals": snapshot_interval,
    }
    if potential_per_snapshot is not None:
        npz_payload["potential_per_snapshot"] = potential_per_snapshot
    np.savez_compressed(chain_dir / "snapshots.npz", **npz_payload)

    meta = {
        **asdict(spec),
        "d": d,
        "n_proposals": n_proposals,
        "snapshot_interval_proposals": snapshot_interval,
        "n_snapshots": int(snapshots.shape[0]),
        "n_accepted": int(n_accepted),
        "wall_seconds_run": float(wall_seconds_run),
        "wall_seconds_setup": float(wall_seconds_setup),
        "proposals_per_second": float(n_proposals / wall_seconds_run),
        "engine": engine,
        "hostname": socket.gethostname(),
    }
    (chain_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[probe] {spec.stage}/{spec.variant}/{spec.point} "
        f"chain {spec.chain_index:02d} ({spec.init_kind}"
        f"{'' if spec.init_side is None else f' side {spec.init_side}'}): "
        f"{spec.sweeps} sweeps in {wall_seconds_run:.1f}s "
        f"({meta['proposals_per_second']:.0f} proposals/s)",
        flush=True,
    )
    return meta


def run_chains_parallel(specs: list[ChainSpec], n_workers: int) -> list[dict]:
    if n_workers <= 1 or len(specs) == 1:
        return [run_chain_worker(spec) for spec in specs]
    with get_context("spawn").Pool(min(n_workers, len(specs))) as pool:
        return pool.map(run_chain_worker, specs)


# Compute traces and R-hat in the parent; lazy Torch imports keep the
# demo_4x4 -> gate_4x4 -> models dependency out of spawned chain workers.


def observable_traces(
    specs: list[ChainSpec], lattice_side: int, sigma: float
) -> dict[str, np.ndarray]:
    """Per-observable (n_chains, n_snapshots) arrays from the saved npzs,
    computed with the demo_4x4 observable set."""
    import torch
    from experiments.constrained_hard_03.demo_4x4 import (
        OBSERVABLE_NAMES,
        observable_values,
    )

    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

    target = FixedCompositionIsingTarget(
        D=lattice_side,
        sigma=sigma,
        target_composition=TARGET_COMPOSITION,
        bias=0.0,
        device="cpu",
    )
    per_chain = []
    for spec in specs:
        spins = np.load(Path(spec.out_dir) / "snapshots.npz")["spins"]
        states = torch.from_numpy(spins.astype(np.float32))
        per_chain.append(
            {
                name: observable_values(name, states, target).numpy()
                for name in OBSERVABLE_NAMES
            }
        )
    return {
        name: np.stack([chain[name] for chain in per_chain]) for name in per_chain[0]
    }


def rhat_block(
    traces: dict[str, np.ndarray], discard_first_half: bool
) -> dict[str, float]:
    out = {}
    for name, chains in traces.items():
        if discard_first_half:
            chains = chains[:, chains.shape[1] // 2 :]
        out[name] = float(split_half_gelman_rubin(chains))
    return out


def post_discard_moments(traces: dict[str, np.ndarray]) -> dict[str, dict]:
    """Frozen moment rule: drop the first half of each chain, pool the rest
    across chains, report E[O] and Var[O]."""
    moments = {}
    for name, chains in traces.items():
        pooled = chains[:, chains.shape[1] // 2 :].reshape(-1)
        moments[name] = {
            "mean": float(pooled.mean()),
            "var": float(pooled.var()),
            "n_pooled_snapshots": int(pooled.size),
        }
    return moments


# Reference and competitor stages.


def chain_dir(
    out_root: Path, spec_stage: str, point: str, variant: str, chain_index: int
) -> Path:
    if spec_stage == "reference":
        return out_root / "reference" / point / f"chain_{chain_index:02d}"
    return out_root / "competitor" / point / variant / f"chain_{chain_index:02d}"


def reference_specs(
    point: str, sweeps: int, args, out_root: Path, chains_per_mode: int
) -> list[ChainSpec]:
    inits = [("phase_separated", 0)] * chains_per_mode + [
        ("phase_separated", 1)
    ] * chains_per_mode
    return [
        ChainSpec(
            stage="reference",
            variant="nonlocal",
            point=point,
            sigma=OPERATING_POINTS[point],
            lattice_side=args.lattice_side,
            chain_index=index,
            seed=REFERENCE_SEED_BASE + POINT_SEED_OFFSETS[point] + index,
            init_kind=kind,
            init_side=side,
            sweeps=sweeps,
            snapshot_every_sweeps=args.snapshot_every_sweeps,
            out_dir=str(chain_dir(out_root, "reference", point, "nonlocal", index)),
        )
        for index, (kind, side) in enumerate(inits)
    ]


def competitor_specs(point: str, variant: str, args, out_root: Path) -> list[ChainSpec]:
    if args.smoke:
        inits = (
            [("random", None)] * SMOKE_CHAINS_PER_MODE
            + [("phase_separated", 0)] * SMOKE_CHAINS_PER_MODE
            + [("phase_separated", 1)] * SMOKE_CHAINS_PER_MODE
        )
    else:
        inits = (
            [("random", None)] * COMPETITOR_RANDOM_CHAINS
            + [("phase_separated", 0)] * COMPETITOR_MODE_CHAINS
            + [("phase_separated", 1)] * COMPETITOR_MODE_CHAINS
        )
    return [
        ChainSpec(
            stage="competitor",
            variant=variant,
            point=point,
            sigma=OPERATING_POINTS[point],
            lattice_side=args.lattice_side,
            chain_index=index,
            seed=(
                COMPETITOR_SEED_BASE
                + POINT_SEED_OFFSETS[point]
                + VARIANT_SEED_OFFSETS[variant]
                + index
            ),
            init_kind=kind,
            init_side=side,
            sweeps=args.sweeps,
            snapshot_every_sweeps=args.snapshot_every_sweeps,
            out_dir=str(chain_dir(out_root, "competitor", point, variant, index)),
        )
        for index, (kind, side) in enumerate(inits)
    ]


def run_reference_point(point: str, args, out_root: Path) -> bool:
    """Run the reference chains for one operating point, gate on the
    R-hat bar, doubling mechanically on failure. Returns pass/fail."""
    chains_per_mode = SMOKE_CHAINS_PER_MODE if args.smoke else REFERENCE_CHAINS_PER_MODE
    doubling_history = []
    passed = False
    for attempt in range(MAX_DOUBLINGS + 1):
        sweeps = args.sweeps * 2**attempt
        specs = reference_specs(point, sweeps, args, out_root, chains_per_mode)
        print(
            f"[probe] reference/{point}: attempt {attempt} — "
            f"{len(specs)} chains x {sweeps:,} sweeps "
            f"(sigma={OPERATING_POINTS[point]})",
            flush=True,
        )
        chain_metas = run_chains_parallel(specs, args.workers)
        traces = observable_traces(specs, args.lattice_side, OPERATING_POINTS[point])
        rhat_post_discard = rhat_block(traces, discard_first_half=True)
        rhat_full_trace = rhat_block(traces, discard_first_half=False)
        passed = all(
            value <= REFERENCE_VALIDITY_RHAT for value in rhat_post_discard.values()
        )
        doubling_history.append(
            {
                "attempt": attempt,
                "sweeps": sweeps,
                "rhat_post_discard": rhat_post_discard,
                "rhat_full_trace": rhat_full_trace,
                "passed": passed,
            }
        )
        print(
            f"[probe] reference/{point}: split-half R-hat (post-discard) "
            + ", ".join(f"{k}={v:.4f}" for k, v in rhat_post_discard.items())
            + f" -> {'PASS' if passed else 'FAIL'} "
            f"(bar {REFERENCE_VALIDITY_RHAT})",
            flush=True,
        )
        if passed:
            break
        if attempt < MAX_DOUBLINGS:
            print(
                f"[probe] reference/{point}: doubling chain length to "
                f"{sweeps * 2:,} sweeps (mechanical rule, rerun same seeds)",
                flush=True,
            )

    summary = {
        "stage": "reference",
        "point": point,
        "sigma": OPERATING_POINTS[point],
        "D": args.lattice_side,
        "n_chains": len(specs),
        "sweeps_final": sweeps,
        "snapshot_every_sweeps": args.snapshot_every_sweeps,
        "validity_rhat_bar": REFERENCE_VALIDITY_RHAT,
        "passed": passed,
        "doubling_rule": "rerun_same_seeds_at_double_length",
        "doubling_history": doubling_history,
        "rhat_post_discard": doubling_history[-1]["rhat_post_discard"],
        "rhat_full_trace": doubling_history[-1]["rhat_full_trace"],
        "post_discard_moments": post_discard_moments(traces),
        "chains": chain_metas,
    }
    summary_path = out_root / "reference" / point / "reference_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[probe] wrote {summary_path}", flush=True)

    if not passed:
        print(
            f"[probe] *** REFERENCE VALIDITY FAILURE at {point}: split-half "
            f"R-hat still above {REFERENCE_VALIDITY_RHAT} after "
            f"{MAX_DOUBLINGS} doublings ({sweeps:,} sweeps). The reference "
            "moments for this point are NOT trustworthy. ***",
            flush=True,
        )
    return passed


def run_competitor_point(point: str, args, out_root: Path) -> None:
    variants = ("local", "nonlocal")
    specs_by_variant = {
        variant: competitor_specs(point, variant, args, out_root)
        for variant in variants
    }
    all_specs = [spec for specs in specs_by_variant.values() for spec in specs]
    print(
        f"[probe] competitor/{point}: {len(all_specs)} chains "
        f"({len(variants)} variants) x {args.sweeps:,} sweeps "
        f"(sigma={OPERATING_POINTS[point]})",
        flush=True,
    )
    metas = run_chains_parallel(all_specs, args.workers)
    metas_by_dir = {meta["out_dir"]: meta for meta in metas}

    for variant, specs in specs_by_variant.items():
        traces = observable_traces(specs, args.lattice_side, OPERATING_POINTS[point])
        # Competitor health uses every stored record from step 0; burn-in
        # is discarded only by the analysis stage.
        rhat_full_trace = rhat_block(traces, discard_first_half=False)
        health_floor_exceeded = [
            name
            for name, value in rhat_full_trace.items()
            if value > COMPETITOR_HEALTH_RHAT
        ]
        chain_metas = [metas_by_dir[spec.out_dir] for spec in specs]
        summary = {
            "stage": "competitor",
            "point": point,
            "variant": variant,
            "sigma": OPERATING_POINTS[point],
            "D": args.lattice_side,
            "n_chains": len(specs),
            "sweeps": args.sweeps,
            "snapshot_every_sweeps": args.snapshot_every_sweeps,
            "rhat_full_trace": rhat_full_trace,
            "health_rhat_floor": COMPETITOR_HEALTH_RHAT,
            "health_floor_exceeded": health_floor_exceeded,
            "total_proposals": sum(m["n_proposals"] for m in chain_metas),
            "chains": chain_metas,
        }
        summary_path = (
            out_root / "competitor" / point / variant / "competitor_summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(
            f"[probe] competitor/{point}/{variant}: R-hat "
            + ", ".join(f"{k}={v:.3f}" for k, v in rhat_full_trace.items())
            + (
                f" (health floor {COMPETITOR_HEALTH_RHAT} exceeded on: "
                f"{health_floor_exceeded})"
                if health_floor_exceeded
                else ""
            )
            + f" -> wrote {summary_path}",
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["reference", "competitor"])
    parser.add_argument(
        "--point",
        choices=list(OPERATING_POINTS),
        default=None,
        help="default: run both points sequentially",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=1_000_000,
        help="chain length; 1 sweep = d proposals",
    )
    parser.add_argument("--snapshot-every-sweeps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="results/kawasaki_probe")
    parser.add_argument(
        "--lattice-side",
        type=int,
        default=FULL_LATTICE_SIDE,
        help="torus side D; non-default sizes (the 16x16 "
        "rescue rung's reference chain) MUST also set "
        "--out, or the 8x8 chain dirs get overwritten",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="D=4, 2000 sweeps, 2 chains/mode, 2 workers; output under <out>/smoke/",
    )
    args = parser.parse_args(argv)

    if args.lattice_side != FULL_LATTICE_SIDE and args.out == "results/kawasaki_probe":
        parser.error(
            "--lattice-side != 8 requires an explicit --out "
            "(protects the frozen 8x8 probe outputs)"
        )
    out_root = Path(args.out)
    if args.smoke:
        args.lattice_side = SMOKE_LATTICE_SIDE
        args.sweeps = SMOKE_SWEEPS
        args.workers = SMOKE_WORKERS
        out_root = out_root / "smoke"
    if args.sweeps % args.snapshot_every_sweeps != 0:
        parser.error("--sweeps must be a multiple of --snapshot-every-sweeps")

    points = [args.point] if args.point else list(OPERATING_POINTS)
    failed_points = []
    for point in points:
        if args.stage == "reference":
            if not run_reference_point(point, args, out_root):
                failed_points.append(point)
        else:
            run_competitor_point(point, args, out_root)

    if failed_points:
        print(
            f"[probe] *** FAILED reference validity at: {failed_points} ***",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
