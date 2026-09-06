"""Frozen-checkpoint state-frequency calibration at the enumerable size.

The ordinate is count(x)/N, an estimate of the *unweighted endpoint law*.
DNFS path weights are never inverted into an endpoint density. Every binary
configuration has a bin, including unvisited and (for swaps) forbidden states.
Sampling uses the production Euler implementation and the recorded grid, so
the comparison includes discretisation error as well as learned-rate error.

One histogram per training seed prevents pooling from hiding complementary
errors. A multinomial sample from the enumerated target at the same N gives
the finite-sample reference for TV and unvisited target mass. These reference
intervals describe a perfect sampler, not uncertainty in the learned model.

Usage: python -m scripts.configuration_calibration_4x4 prepare
       python -m scripts.configuration_calibration_4x4 sample --index 0
"""

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.seeding import seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/configuration_calibration_4x4"
PANELS = (
    (
        "unconstrained_s010",
        "Unconstrained",
        "01_baseline",
        "stage_4_d4_seed{seed}_20260609-*",
        (42, 43, 44, 45),
    ),
    (
        "unconstrained_sc",
        "Unconstrained",
        "01_baseline",
        "stage_4_d4_critical_sc_seed{seed}_20260824-wave1-sc",
        (42, 43, 44, 45),
    ),
    (
        "hard_s010",
        "Hard",
        "03_hard",
        "H2_d16_c50_s010_letf_thp_10k_w2_seed{seed}_20260825-hard-w2",
        (42, 43, 44),
    ),
    (
        "hard_sc",
        "Hard",
        "03_hard",
        "H2_d16_c50_s220_letf_thp_10k_w2_seed{seed}_20260825-hard-w2",
        (42, 43, 44),
    ),
    (
        "soft_s010",
        "Soft",
        "02_constrained_soft",
        "S2_d4_c0500_10k_l50_letf_house_seed{seed}_20260902-softhouse-d16-10k",
        (42, 43, 44, 45),
    ),
    (
        "soft_sc",
        "Soft",
        "02_constrained_soft",
        "S2_d4_c0500_10k_l50_letf_house_sc_seed{seed}_20260902-softhouse-d16-10k",
        (42, 43, 44, 45),
    ),
)


def count_configurations(states):
    """Site zero is the least significant bit; count every state, not visits only."""
    if states.ndim != 2 or not torch.all((states == -1) | (states == 1)):
        raise ValueError("expected a matrix of binary spins")
    powers = 2 ** torch.arange(states.shape[1], device=states.device)
    keys = ((states > 0).long() * powers).sum(dim=1)
    return torch.bincount(keys, minlength=2 ** states.shape[1]).cpu().numpy()


def exact_probabilities(target):
    """Normalised target in the same integer-key order as the count histogram."""
    states = enumerate_states(target.d).double().to(target.device)
    # Promote the frozen coupling matrix for the enumerated reference.
    coupling = target.J
    target.J = coupling.double()
    try:
        logp = target.log_prob(states)
    finally:
        target.J = coupling
    if hasattr(target, "n_plus_target"):
        logp[(states > 0).sum(dim=1) != target.n_plus_target] = -torch.inf
    powers = 2 ** torch.arange(target.d, device=states.device)
    keys = ((states > 0).long() * powers).sum(dim=1)
    probabilities = torch.zeros_like(logp)
    probabilities[keys] = logp.softmax(0)
    return probabilities.cpu().numpy()


def calibration_metrics(counts, probabilities):
    counts = np.asarray(counts)
    probabilities = np.asarray(probabilities)
    if (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("counts must be nonnegative with positive total")
    if counts.shape != probabilities.shape or not np.isclose(probabilities.sum(), 1):
        raise ValueError("counts and normalised probabilities must share their support")
    empirical = counts / counts.sum()
    return {
        "n_samples": int(counts.sum()),
        "tv": float(np.abs(empirical - probabilities).sum() / 2),
        "unvisited_target_mass": float(probabilities[counts == 0].sum()),
        "off_support_fraction": float(empirical[probabilities == 0].sum()),
        "visited_states": int(np.count_nonzero(counts)),
    }


def reference_metrics(probabilities, n_samples, replicates=200, seed=1729):
    rng = np.random.default_rng(seed)
    measurements = [
        calibration_metrics(rng.multinomial(n_samples, probabilities), probabilities)
        for _ in range(replicates)
    ]
    result = {"replicates": replicates, "seed": seed, "n_samples": n_samples}
    for key in ("tv", "unvisited_target_mass"):
        low, median, high = np.quantile(
            [m[key] for m in measurements], [0.025, 0.5, 0.975]
        )
        result.update(
            {
                f"{key}_low": float(low),
                f"{key}_median": float(median),
                f"{key}_high": float(high),
            }
        )
    return result


def prepare(output=OUTPUT, n_samples=1_000_000):
    """Freeze the exact local configs; stage available checkpoints for Modal."""
    output = Path(output)
    tasks = []
    for panel, family, group, pattern, seeds in PANELS:
        for seed in seeds:
            matches = sorted((ROOT / "results" / group).glob(pattern.format(seed=seed)))
            if len(matches) != 1:
                raise ValueError(
                    f"expected one run for {pattern}, seed {seed}: {matches}"
                )
            run = matches[0]
            staged = output / "inputs" / run.name
            staged.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / "config.json", staged / "config.json")
            checkpoint = run / "checkpoints/final.pt"
            if checkpoint.exists():
                (staged / "checkpoints").mkdir(exist_ok=True)
                shutil.copy2(checkpoint, staged / "checkpoints/final.pt")
            tasks.append(
                {
                    "panel": panel,
                    "family": family,
                    "group": group,
                    "run": run.name,
                    "training_seed": seed,
                    "redraw_seed": 20260906 + len(tasks),
                    "n_samples": n_samples,
                }
            )
    (output / "manifest.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return tasks


def load_sampler(run, family, checkpoint=None):
    """Rebuild using the production constructors and a checked frozen config."""
    from experiments.dnfs_baseline_01.run import _build_model, _rebuild_from_run_dir

    run = Path(run)
    saved = json.loads((run / "config.json").read_text())
    if family == "Hard":
        from experiments.constrained_hard_03.configs import CONFIGS, HardStageCfg
        from experiments.constrained_hard_03.run import (
            _backfill_missing_defaults,
            build_target_and_head,
        )

        _backfill_missing_defaults(saved, HardStageCfg)
        cfg = CONFIGS[saved["name"]]
        cfg = replace(
            cfg,
            head_kind=saved["head_kind"],
            train=replace(cfg.train, seed=saved["train"]["seed"]),
        )
        if json.loads(json.dumps(asdict(cfg))) != saved:
            raise ValueError(f"frozen hard config drift: {run.name}")
        cfg = replace(cfg, compile_head=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        target, model = build_target_and_head(cfg, device)
    else:
        cfg, target, device = _rebuild_from_run_dir(run)
        if cfg.condition_on_composition:
            raise ValueError("this figure selects specialists only")
        # Eager evaluation avoids a separate compile per frozen checkpoint.
        cfg.model = replace(cfg.model, compile_model=False)
        model = _build_model(cfg, target)
    if target.d != 16:
        raise ValueError("this evaluator is restricted to 4x4 Ising")
    checkpoint = Path(checkpoint or run / "checkpoints/final.pt")
    model.load_state_dict(
        torch.load(checkpoint, weights_only=True, map_location=device)
    )
    model.eval()
    return cfg, target, model, device


def sample_task(task, inputs, output, checkpoint=None, batch_size=5000):
    """Stream final states into a histogram without storing trajectories."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run = Path(inputs) / task["run"]
    checkpoint = Path(checkpoint or run / "checkpoints/final.pt")
    prefix = output / task["run"]
    cfg, target, model, device = load_sampler(run, task["family"], checkpoint)
    seed_everything(task["redraw_seed"])
    probabilities = exact_probabilities(target)
    grid = torch.linspace(0.0, 1.0, cfg.ctmc.n_euler_steps, device=device)
    counts = np.zeros(2**16, dtype=np.int64)
    start = time.perf_counter()
    with torch.no_grad():
        while counts.sum() < task["n_samples"]:
            n = min(batch_size, task["n_samples"] - int(counts.sum()))
            base = target.sample_base(n, device)
            if task["family"] == "Hard":
                samples = sample_swap_ctmc(
                    model, base, grid, multi_event=cfg.ctmc.use_matching_step
                )
                target.assert_on_manifold(samples)
            else:
                samples = sample_ctmc(model, base, grid)
            counts += count_configurations(samples)
            if counts.sum() % 100000 == 0 or counts.sum() == task["n_samples"]:
                print(
                    f"{task['run']}: {counts.sum():,}/{task['n_samples']:,}, "
                    f"{time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    elapsed = time.perf_counter() - start
    metadata = {
        **task,
        **calibration_metrics(counts, probabilities),
        "seconds": elapsed,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch_version": str(torch.__version__),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "precision": "fp32",
        "compiled": False,
        "batch_size": batch_size,
        "n_euler_steps": cfg.ctmc.n_euler_steps,
        "sigma": cfg.ising.sigma,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
        "reference": reference_metrics(probabilities, int(counts.sum())),
    }
    np.savez_compressed(
        prefix.with_suffix(".npz"), counts=counts, probabilities=probabilities
    )
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "sample"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    if args.action == "prepare":
        print(f"Prepared {len(prepare(args.output, args.n_samples))} checkpoints")
    else:
        tasks = json.loads((args.output / "manifest.json").read_text())
        task = {**tasks[args.index], "n_samples": args.n_samples}
        sample_task(
            task,
            args.output / "inputs",
            args.output / "counts",
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
