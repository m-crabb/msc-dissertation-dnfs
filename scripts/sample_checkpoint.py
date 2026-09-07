"""Draw proposals and importance weights from the bundled example checkpoints."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.seeding import seed_everything


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(bundle, device):
    """Verify the distributed files and rebuild with the production constructors."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha256(bundle / name) != expected:
            raise ValueError(f"Checksum mismatch: {bundle / name}")
    if manifest["family"] == "hard":
        from experiments.constrained_hard_03.configs import CONFIGS, HardStageCfg
        from experiments.constrained_hard_03.run import (
            _backfill_missing_defaults,
            build_target_and_head,
        )

        saved = json.loads((bundle / "config.json").read_text())
        _backfill_missing_defaults(saved, HardStageCfg)
        cfg = CONFIGS[saved["name"]]
        cfg = replace(
            cfg,
            head_kind=saved["head_kind"],
            train=replace(cfg.train, seed=saved["train"]["seed"]),
        )
        if json.loads(json.dumps(asdict(cfg))) != saved:
            raise ValueError(f"Saved hard config differs from registry: {cfg.name}")
        cfg = replace(cfg, compile_head=False)
        target, model = build_target_and_head(cfg, device)
    elif manifest["family"] in {"baseline", "soft"}:
        from experiments.dnfs_baseline_01.run import (
            _build_model,
            _construct_target,
            _rebuild_from_run_dir,
        )

        cfg, _, _ = _rebuild_from_run_dir(bundle)
        if cfg.condition_on_composition:
            raise ValueError("This demo supports specialist flip checkpoints only")
        cfg.model = replace(cfg.model, compile_model=False)
        target = _construct_target(cfg.ising, device=device)
        model = _build_model(cfg, target)
    else:
        raise ValueError(f"Unknown checkpoint family: {manifest['family']}")
    model.load_state_dict(
        torch.load(
            bundle / manifest["checkpoint"], map_location=device, weights_only=True
        )
    )
    model.eval()
    return cfg, target, model, manifest


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--n-samples", type=positive_int, default=64)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=positive_int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; choose a new output directory")
    torch.set_num_threads(args.threads)
    cfg, target, model, manifest = load_bundle(args.bundle, args.device)
    seed_everything(args.seed)
    hard = manifest["family"] == "hard"
    # Preserve each trainer's convention: hard counts intervals, flip grid points.
    grid = torch.linspace(0, 1, cfg.ctmc.n_euler_steps + int(hard), device=args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    samples, log_weights = [], []
    with torch.no_grad():
        for start in range(0, args.n_samples, args.batch_size):
            base = target.sample_base(
                min(args.batch_size, args.n_samples - start), args.device
            )
            kwargs = {"multi_event": cfg.ctmc.use_matching_step} if hard else {}
            draw, weights = (sample_swap_ctmc if hard else sample_ctmc)(
                model, base, grid, target=target, return_log_weights=True, **kwargs
            )
            samples.append(draw.cpu())
            log_weights.append(weights.cpu())
    samples, log_weights = torch.cat(samples), torch.cat(log_weights)
    if not torch.isfinite(log_weights).all():
        raise ValueError("Sampling produced nonfinite importance weights")
    torch.save(samples, args.out / "samples.pt")
    torch.save(log_weights, args.out / "log_weights.pt")
    weights = log_weights.double().softmax(0)
    metadata = {
        "bundle": args.bundle.as_posix(),
        "checkpoint": manifest["checkpoint"],
        "source_sha256": manifest["sha256"],
        "seed": args.seed,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "device": args.device,
        "threads": args.threads,
        "torch_version": torch.__version__,
        "dtype": "float32",
        "n_intervals": len(grid) - 1,
        "step_kind": "matching" if hard and cfg.ctmc.use_matching_step else "one_event",
        "draw_kind": "raw proposals with log importance weights; no resampling",
        "ess": float(1 / weights.square().sum()),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(samples)} proposals to {args.out}; ESS {metadata['ess']:.2f}")


if __name__ == "__main__":
    main()
