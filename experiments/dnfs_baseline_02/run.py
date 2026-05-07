"""Stage-agnostic entry point for DNFS Ising baseline runs.

Usage (local):
    pixi run -e dev python -m experiments.dnfs_baseline_02.run \\
        --cfg stage_1_d4 --seed 0

Or, to recompute eval metrics from saved samples without re-training:
    pixi run -e dev python -m experiments.dnfs_baseline_02.run \\
        --eval-only --run-dir results/02_baseline/stage_1_d4_seed0_...

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

from discrete_flow_sampler.diagnostics.metrics import (
    entropy_estimate,
    ess_from_log_weights,
    exact_free_energy,
    exact_internal_energy,
    free_energy_lb_estimate,
    internal_energy_estimate,
)
from discrete_flow_sampler.models.mlp import MLPRateMatrix
from discrete_flow_sampler.samplers import log_z_estimators
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.training import train as train_loop
from discrete_flow_sampler.targets.ising import IsingTarget
from experiments.dnfs_baseline_02.configs import CONFIGS

# State-count cutoff for exact-enumeration "Optimal Value" references at
# small D. Paper Table 2 row 1 lists analytical (Ferdinand & Fisher 1969)
# optima for D = 10×10; for D ≤ 20 we use direct enumeration, the
# small-lattice analog. Beyond D = 20 enumeration is memory-bound and
# the analytical solution is the right call.
ENUMERATION_MAX_SPINS = 20


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
    if name == "control_variate":
        return log_z_estimators.control_variate
    raise ValueError(f"Unknown estimator: {name!r}")


def _compute_eval_metrics(
    eval_samples: torch.Tensor,
    eval_log_weights: torch.Tensor,
    target,
) -> dict:
    """Aggregate end-of-run diagnostics into a JSON-friendly dict.

    Paper-faithful suite (DNFS Appendix D.1, Table 2):

    - `n_eval_samples`, `ess`, `ess_fraction`: importance-weight
      self-consistency (Eq. 42). The paper reports the normalised ESS in
      [1/K, 1]; `ess_fraction` is that value.
    - `free_energy_per_site`: F/D = -log Ẑ_lb / (2σD) from
      `free_energy_lb_estimate` (Eq. 37).
    - `internal_energy_per_site`: E/D from self-normalised IS (Eq. 38).
    - `entropy_per_site`: S/D = 2σ(E - F) / D (Table 2 caption).

    Exact-reference fields (only when target.d ≤ ENUMERATION_MAX_SPINS,
    i.e. D ≤ 20 -- the small-lattice analog of paper Table 2's "Optimal
    Value" row, which at D = 10×10 uses Ferdinand & Fisher 1969 instead):

    - `free_energy_per_site_exact`, `internal_energy_per_site_exact`,
      `entropy_per_site_exact`: enumeration-based truths.
    - `free_energy_per_site_bias`, `internal_energy_per_site_bias`,
      `entropy_per_site_bias`: estimate − exact, signed (positive ⇒
      estimate too high relative to truth).

    Args:
        eval_samples: (N, d) tensor in {-1, +1}, on `target.device`.
        eval_log_weights: (N,) IS log-weights from the same eval pass.
        target: IsingTarget (or any duck with `.d`, `.device`, `.log_prob`,
            `.sigma`).
    """
    sigma = float(target.sigma)
    D = int(target.d)

    log_p_tilde_eval = target.log_prob(eval_samples.float())
    F_hat = free_energy_lb_estimate(eval_log_weights, sigma=sigma, D=D)
    E_hat = internal_energy_estimate(
        eval_log_weights, log_p_tilde_eval, sigma=sigma, D=D
    )
    S_hat = entropy_estimate(F_hat, E_hat, sigma=sigma)

    metrics: dict = {
        "n_eval_samples": int(eval_log_weights.numel()),
        "ess": float(ess_from_log_weights(eval_log_weights).item()),
        "free_energy_per_site": float(F_hat.item()),
        "internal_energy_per_site": float(E_hat.item()),
        "entropy_per_site": float(S_hat.item()),
    }
    metrics["ess_fraction"] = metrics["ess"] / metrics["n_eval_samples"]

    if D <= ENUMERATION_MAX_SPINS:
        F_exact = exact_free_energy(target, sigma=sigma, D=D)
        E_exact = exact_internal_energy(target, sigma=sigma, D=D)
        S_exact = entropy_estimate(F_exact, E_exact, sigma=sigma)
        metrics["free_energy_per_site_exact"] = float(F_exact.item())
        metrics["internal_energy_per_site_exact"] = float(E_exact.item())
        metrics["entropy_per_site_exact"] = float(S_exact.item())
        metrics["free_energy_per_site_bias"] = (
            metrics["free_energy_per_site"]
            - metrics["free_energy_per_site_exact"]
        )
        metrics["internal_energy_per_site_bias"] = (
            metrics["internal_energy_per_site"]
            - metrics["internal_energy_per_site_exact"]
        )
        metrics["entropy_per_site_bias"] = (
            metrics["entropy_per_site"]
            - metrics["entropy_per_site_exact"]
        )

    return metrics


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

    eval_metrics = _compute_eval_metrics(eval_samples, eval_log_weights, target)
    (eval_dir / "metrics.json").write_text(json.dumps(eval_metrics, indent=2))

    if use_wandb:
        artifact = wandb.Artifact(
            f"eval_{cfg.name}_seed{seed}", type="evaluation"
        )
        artifact.add_file(str(eval_dir / "samples.pt"))
        artifact.add_file(str(eval_dir / "log_weights.pt"))
        artifact.add_file(str(eval_dir / "metrics.json"))
        wandb.log_artifact(artifact)
        # Numeric metrics only -- wandb chokes on the None / bool entries.
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
    """Recompute eval metrics from a finished run's saved samples.

    Loads `eval/samples.pt` + `eval/log_weights.pt`, reconstructs the
    target from `config.json`, and writes / overwrites `eval/metrics.json`
    in the run directory. Useful for backfilling diagnostics on older
    runs whose training pre-dated the metrics-aggregation block.
    """
    run_dir = Path(run_dir)
    cfg_dict = json.loads((run_dir / "config.json").read_text())
    eval_samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True)
    eval_log_weights = torch.load(
        run_dir / "eval" / "log_weights.pt", weights_only=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = IsingTarget(
        D=cfg_dict["ising"]["D"],
        sigma=cfg_dict["ising"]["sigma"],
        bias=cfg_dict["ising"]["bias"],
        device=device,
    )
    eval_samples = eval_samples.to(device)
    eval_log_weights = eval_log_weights.to(device)

    eval_metrics = _compute_eval_metrics(eval_samples, eval_log_weights, target)
    (run_dir / "eval" / "metrics.json").write_text(
        json.dumps(eval_metrics, indent=2)
    )
    return eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="Config key from configs.py CONFIGS (training mode)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/02_baseline")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; recompute eval/metrics.json from saved samples",
    )
    parser.add_argument(
        "--run-dir",
        help="Run directory to re-evaluate (required with --eval-only)",
    )
    args = parser.parse_args()

    if args.eval_only:
        if not args.run_dir:
            parser.error("--eval-only requires --run-dir")
        metrics = eval_only(args.run_dir)
        print(json.dumps(metrics, indent=2))
        return

    if not args.cfg:
        parser.error("--cfg is required for training mode")
    train(
        args.cfg,
        seed=args.seed,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
