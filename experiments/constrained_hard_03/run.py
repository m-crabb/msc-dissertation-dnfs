"""Entry point for hard-constraint swap-CTMC runs.

Usage (local):
    pixi run -e dev python -m experiments.constrained_hard_03.run \\
        --cfg H2_d16_c50_s010_letf_dh --seed 42

Mirrors `experiments/dnfs_baseline_01/run.py` with the single-site stack
swapped for the swap-move one: `FixedCompositionIsingTarget` +
`build_swap_head` + `train_swap` + `sample_swap_ctmc`. `--smoke` shrinks the
run to a minutes-scale end-to-end check (the doubly-hollow head costs O(d^2)
masked body passes per forward, so full eval batches are slow on CPU).
"""
import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from experiments.constrained_hard_03.configs import (
    CONFIGS,
    HardStageCfg,
    build_swap_head,
)

from discrete_flow_sampler.diagnostics.metrics import (
    composition_observables,
    ess_from_log_weights,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

HEAD_KINDS = ("doubly_hollow", "mask_one", "non_antisym")


def smoke_config(cfg: HardStageCfg) -> HardStageCfg:
    """Shrink `cfg` to a minutes-scale end-to-end check (shared by the CLI
    `--smoke` flag and `modal_app.train_remote`'s `smoke` argument, so the
    two entry points can never drift apart)."""
    return replace(
        cfg,
        train=replace(cfg.train, n_steps=4, inner_steps_per_outer=2),
        ctmc=replace(cfg.ctmc, n_euler_steps=8),
        eval=replace(cfg.eval, n_eval_samples=64),
    )


def train(
    cfg: HardStageCfg,
    seed: int = 42,
    output_dir: str | Path = "results/03_hard",
    use_wandb: bool = True,
    tag: str | None = None,
):
    """Train a swap head on the fixed-composition target and save eval artefacts."""
    # Apply the per-invocation seed without mutating the frozen config.
    cfg = replace(cfg, train=replace(cfg.train, seed=seed))

    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{cfg.name}_seed{seed}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist the resolved config (incl. the effective head_kind) so the run
    # is reproducible from the directory alone.
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            group=cfg.name,
            name=f"{cfg.name}_seed{seed}_{tag}",
            config=asdict(cfg),
            tags=[
                cfg.name,
                f"D={cfg.ising.D}",
                f"sigma={cfg.ising.sigma}",
                cfg.estimator,
                cfg.head_kind,
                f"seed={seed}",
            ],
        )

    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = FixedCompositionIsingTarget(
        D=cfg.ising.D,
        sigma=cfg.ising.sigma,
        target_composition=cfg.ising.target_composition,
        bias=cfg.ising.bias,
        device=device,
    )
    backbone = LeTFRateMatrix(
        d=target.d,
        vocab_size=cfg.model.vocab_size,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
    ).to(device)
    head = build_swap_head(cfg, backbone)

    train_swap(
        head,
        target,
        cfg.train,
        cfg.ctmc,
        cfg.eval,
        run_dir,
        use_wandb=use_wandb,
        estimator_mode=cfg.estimator,
    )

    # End-of-run eval: (samples, IS log-weights) over the full t = 0 -> 1
    # trajectory. The base draw is already on the fixed-composition manifold
    # and swaps keep it there.
    with torch.no_grad():
        x_eval_initial = target.sample_base(cfg.eval.n_eval_samples, device=device)
        ts = torch.linspace(0.0, 1.0, cfg.ctmc.n_euler_steps + 1, device=device)
        eval_samples, eval_log_weights = sample_swap_ctmc(
            head, x_eval_initial, ts, return_log_weights=True, target=target,
        )
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    torch.save(eval_samples.cpu(), eval_dir / "samples.pt")
    torch.save(eval_log_weights.cpu(), eval_dir / "log_weights.pt")

    eval_metrics = {
        "n_eval_samples": int(eval_log_weights.numel()),
        "ess": float(ess_from_log_weights(eval_log_weights).item()),
        "head_kind": cfg.head_kind,
    }
    eval_metrics["ess_fraction"] = eval_metrics["ess"] / eval_metrics["n_eval_samples"]
    eval_metrics.update(
        composition_observables(
            eval_samples, target_composition=cfg.ising.target_composition,
        )
    )
    (eval_dir / "metrics.json").write_text(json.dumps(eval_metrics, indent=2))

    if use_wandb:
        wandb.log(
            {
                f"eval/{key}": value
                for key, value in eval_metrics.items()
                if isinstance(value, (int, float))
            }
        )
        wandb.finish()

    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        required=True,
        choices=list(CONFIGS.keys()),
        help="Config key from configs.py CONFIGS",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/03_hard")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--tag", default=None, help="Run-dir suffix (default: wall-clock timestamp)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shrink to a fast end-to-end check: 4 steps, 8 Euler steps, 64 "
        "eval samples (inner_steps_per_outer drops to 2 to keep whole outer "
        "cycles)",
    )
    parser.add_argument(
        "--head-kind",
        choices=HEAD_KINDS,
        default=None,
        help="Override cfg.head_kind (e.g. mask_one for speed); the effective "
        "value is recorded in config.json and eval/metrics.json",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.cfg]
    if args.head_kind is not None:
        cfg = replace(cfg, head_kind=args.head_kind)
    if args.smoke:
        cfg = smoke_config(cfg)
    train(
        cfg,
        seed=args.seed,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
