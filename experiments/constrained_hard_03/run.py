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


def build_target_and_head(
    cfg: HardStageCfg, device: str
) -> tuple[FixedCompositionIsingTarget, torch.nn.Module]:
    """Shared constructor for the train and eval-only entry points, so the
    two can never drift in how they instantiate the target/backbone/head."""
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
        use_sdpa_readout=cfg.model.use_sdpa_readout,
    ).to(device)
    return target, build_swap_head(cfg, backbone)


def final_eval(head, target, cfg: HardStageCfg, run_dir: Path) -> dict:
    """End-of-run eval: (samples, IS log-weights) over the full t = 0 -> 1
    trajectory, streamed in `eval_sample_chunk` slices. The vectorised swap
    head rides d anchor copies per sample, so an unchunked n_eval_samples
    batch OOMs at large d (all three d=64 sigma_c seeds died here,
    2026-07-06); slicing changes nothing statistically because the IS
    weights are independent per sample. Runs fp32 — the bf16 opt-in covers
    the in-training diagnostic eval only. Writes eval/ artefacts into
    `run_dir` and returns the metrics dict."""
    device = next(head.parameters()).device
    ts = torch.linspace(0.0, 1.0, cfg.ctmc.n_euler_steps + 1, device=device)
    chunk = cfg.eval.eval_sample_chunk or cfg.eval.n_eval_samples
    sample_slices, log_weight_slices = [], []
    remaining = cfg.eval.n_eval_samples
    with torch.no_grad():
        while remaining > 0:
            x_initial = target.sample_base(min(chunk, remaining), device=device)
            slice_samples, slice_log_weights = sample_swap_ctmc(
                head, x_initial, ts, return_log_weights=True, target=target,
            )
            sample_slices.append(slice_samples)
            log_weight_slices.append(slice_log_weights)
            remaining -= x_initial.shape[0]
    eval_samples = torch.cat(sample_slices)
    eval_log_weights = torch.cat(log_weight_slices)

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
    return eval_metrics


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
    target, head = build_target_and_head(cfg, device)

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

    eval_metrics = final_eval(head, target, cfg, run_dir)

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


def eval_only(run_dir: str | Path) -> dict:
    """Re-run the end-of-run eval for a completed run dir (config.json +
    checkpoints/final.pt), writing the eval/ artefacts in place. Recovery
    path for runs whose training finished but whose final eval died before
    the chunked `final_eval` landed (the 2026-07-06 d=64 OOMs)."""
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = CONFIGS[saved["name"]]
    cfg = replace(
        cfg,
        head_kind=saved["head_kind"],
        train=replace(cfg.train, seed=saved["train"]["seed"]),
    )
    # Guard against silent drift between the run's recorded config and the
    # current CONFIGS entry (json round-trip normalises tuples to lists).
    if json.loads(json.dumps(asdict(cfg))) != saved:
        raise ValueError(
            f"config.json in {run_dir} does not match CONFIGS[{saved['name']!r}]"
        )

    seed_everything(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target, head = build_target_and_head(cfg, device)
    head.load_state_dict(
        torch.load(
            run_dir / "checkpoints" / "final.pt",
            map_location=device,
            weights_only=True,
        )
    )
    eval_metrics = final_eval(head, target, cfg, run_dir)
    print(f"[eval_only] {run_dir.name}: {json.dumps(eval_metrics, indent=2)}")
    return eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default=None,
        choices=list(CONFIGS.keys()),
        help="Config key from configs.py CONFIGS",
    )
    parser.add_argument(
        "--eval-only",
        default=None,
        metavar="RUN_DIR",
        help="Skip training: re-run the end-of-run eval for this completed "
        "run dir (uses its config.json + checkpoints/final.pt)",
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

    if args.eval_only is not None:
        eval_only(args.eval_only)
        return
    if args.cfg is None:
        parser.error("--cfg is required unless --eval-only is given")

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
