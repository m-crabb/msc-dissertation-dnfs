"""Stage-agnostic entry point for DNFS Ising baseline runs.

Usage (local):
    pixi run -e dev python -m experiments.dnfs_baseline_02.run \\
        --cfg stage_1_d4 --seed 0

The same `train(cfg_name, seed, ...)` function is also imported by
`modal_app.py` for remote runs, so both paths share artefacts and metadata.
"""
import argparse
import json
import platform
import socket
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from discrete_flow_sampler.models.mlp import MLPRateMatrix
from discrete_flow_sampler.samplers import log_z_estimators
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.training import train as train_loop
from discrete_flow_sampler.targets.ising import IsingTarget
from experiments.dnfs_baseline_02.configs import CONFIGS


def _git_commit() -> str:
    """Best-effort capture of HEAD SHA for run reproducibility. Returns
    "unknown" outside a git working tree (e.g. inside a Modal sandbox)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _build_model(cfg, target):
    if cfg.model.kind == "mlp":
        return MLPRateMatrix(
            d=target.d,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
        ).to(target.device)
    raise ValueError(f"Unknown model kind: {cfg.model.kind!r}")


def _build_estimator(name: str):
    if name == "naive_mc":
        return log_z_estimators.naive_mc
    raise ValueError(f"Unknown estimator: {name!r}")


def train(
    cfg_name: str,
    seed: int = 0,
    output_dir: str | Path = "results/02_baseline",
    use_wandb: bool = True,
):
    """Top-level training entry. Importable from CLI or modal_app.

    Builds the target / model / estimator from the named config, kicks off
    `samplers.training.train`, and persists end-of-run eval samples + IS
    log-weights for downstream analysis notebooks.
    """
    base_cfg = CONFIGS[cfg_name]
    # Apply the per-invocation seed without mutating the frozen config.
    cfg = replace(base_cfg, train=replace(base_cfg.train, seed=seed))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{cfg.name}_seed{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config and host metadata next to the artefacts
    # so the run is reproducible from the directory alone.
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "git_commit": _git_commit(),
                "torch_version": torch.__version__,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
            },
            indent=2,
        )
    )

    if use_wandb:
        import wandb

        wandb.init(
            project="dnfs-baseline",
            group=cfg.name,
            name=f"{cfg.name}_seed{seed}_{timestamp}",
            config=asdict(cfg),
            tags=[
                cfg.name,
                f"D={cfg.ising.D}",
                f"sigma={cfg.ising.sigma}",
                cfg.estimator,
                cfg.model.kind,
            ],
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = IsingTarget(
        D=cfg.ising.D,
        sigma=cfg.ising.sigma,
        bias=cfg.ising.bias,
        device=device,
    )
    model = _build_model(cfg, target)
    estimator = _build_estimator(cfg.estimator)

    train_loop(
        model=model,
        target=target,
        estimator=estimator,
        train_cfg=cfg.train,
        ctmc_cfg=cfg.ctmc,
        eval_cfg=cfg.eval,
        output_dir=run_dir,
        use_wandb=use_wandb,
    )

    # End-of-run eval: a final batch of (samples, IS log-weights) over the
    # full t = 0 -> 1 trajectory. Analysis notebooks read these directly.
    with torch.no_grad():
        full_time_grid = torch.linspace(
            0.0, 1.0, cfg.ctmc.n_euler_steps, device=device
        )
        x_eval_initial = (
            torch.randint(
                0, 2, (cfg.eval.n_eval_samples, target.d), device=device
            ).float()
            * 2 - 1
        )
        eval_samples, eval_log_weights = sample_ctmc(
            model,
            x_eval_initial,
            full_time_grid,
            return_log_weights=True,
            target=target,
        )
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    torch.save(eval_samples.cpu(), eval_dir / "samples.pt")
    torch.save(eval_log_weights.cpu(), eval_dir / "log_weights.pt")

    if use_wandb:
        artifact = wandb.Artifact(
            f"eval_{cfg.name}_seed{seed}", type="evaluation"
        )
        artifact.add_file(str(eval_dir / "samples.pt"))
        artifact.add_file(str(eval_dir / "log_weights.pt"))
        wandb.log_artifact(artifact)
        wandb.finish()

    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", required=True, help="Config key from configs.py CONFIGS"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/02_baseline")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    train(
        args.cfg,
        seed=args.seed,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
