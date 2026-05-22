"""Entry point for DNFS Ising baseline runs.

Usage (local):
    pixi run -e dev python -m experiments.dnfs_baseline_01.run \\
        --cfg stage_1_d4 --seed 42

Or, to recompute eval metrics from saved samples without re-training:
    pixi run -e dev python -m experiments.dnfs_baseline_01.run \\
        --eval-only --run-dir results/01_baseline/stage_1_d4_seed42_...

The same `train(cfg, seed, ...)` function is also imported by
`modal_app.py` for remote runs, so both paths share artefacts and metadata.
"""
import argparse
import json
import platform
import socket
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

import torch
from experiments.dnfs_baseline_01.configs import CONFIGS, StageCfg

from discrete_flow_sampler.diagnostics.metrics import (
    composition_observables,
    entropy_estimate,
    ess_from_log_weights,
    exact_free_energy,
    exact_internal_energy,
    free_energy_lb_estimate,
    internal_energy_estimate,
)
from discrete_flow_sampler.models.mlp import MLPRateMatrix
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.training import train as train_loop
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import IsingTarget

# State-count cutoff for exact-enumeration "Optimal Value" references at
# small D. Paper Table 2 row 1 lists analytical (Ferdinand & Fisher 1969)
# optima for D = 10×10; for D ≤ 20 we use direct enumeration, the
# small-lattice analog. Beyond D = 20 enumeration is memory-bound and
# the analytical solution is the right call.
ENUMERATION_MAX_SPINS = 20


def _build_model(cfg, target):
    if cfg.model.kind == "mlp":
        return MLPRateMatrix(
            d=target.d,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
        ).to(target.device)
    if cfg.model.kind == "lemlp":
        from discrete_flow_sampler.models.lemlp import LeMLPRateMatrix
        return LeMLPRateMatrix(
            d=target.d,
            vocab_size=cfg.model.vocab_size,
            hidden_dim=cfg.model.hidden_dim,
            n_summands=cfg.model.n_layers,   # see ModelCfg comment on n_layers
        ).to(target.device)
    if cfg.model.kind == "leconv_deep":
        from discrete_flow_sampler.models.leconv_deep import LeConvDeepRateMatrix
        return LeConvDeepRateMatrix(
            D=cfg.ising.D,
            vocab_size=cfg.model.vocab_size,
            kernel_schedule=cfg.model.kernel_schedule,
            hidden_dim=cfg.model.hidden_dim,
            use_global_context=cfg.model.hollow_global_context,
        ).to(target.device)
    if cfg.model.kind == "let":
        from discrete_flow_sampler.models.letf import LeTFRateMatrix
        return LeTFRateMatrix(
            d=target.d,
            vocab_size=cfg.model.vocab_size,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
        ).to(target.device)
    raise ValueError(f"Unknown model kind: {cfg.model.kind!r}")


def _trailing_ess_metrics(run_dir: Path, k: int = 10) -> dict:
    """Last-K training-time ESS aggregates (median/min/max).

    Addresses single-snapshot eval timing concern: the final ESS reported in
    eval/metrics.json is one trajectory draw at t = n_steps; if the trained
    model oscillates near the end, the snapshot is a lottery. The trailing-K
    window reports the recent training-time ESS distribution for a more
    honest "where did training actually land" reading. Note: training-time
    ESS is over outer_batch_size, not n_eval_samples — interpret in absolute
    counts, not as a fraction comparable to eval/ess_fraction.
    """
    recent = (
        pd.read_csv(run_dir / "training_log.csv")["ess"].dropna().tail(k)
    )
    return {
        f"ess_trailing{k}_median": float(recent.median()),
        f"ess_trailing{k}_min": float(recent.min()),
        f"ess_trailing{k}_max": float(recent.max()),
    }


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
    metrics.update(
        composition_observables(
            eval_samples,
            target_composition=getattr(target, "target_composition", None),
            composition_penalty_strength=getattr(
                target, "composition_penalty_strength", None
            ),
        )
    )

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
    cfg: StageCfg,
    seed: int = 42,
    output_dir: str | Path = "results/01_baseline",
    use_wandb: bool = True,
):
    """Top-level training entry. Importable from CLI or modal_app.

    Builds the target / model / estimator from the resolved config, kicks off
    `samplers.training.train`, and persists end-of-run eval samples + IS
    log-weights for downstream analysis notebooks.
    """
    # Apply the per-invocation seed without mutating the frozen config.
    cfg = replace(cfg, train=replace(cfg.train, seed=seed))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{cfg.name}_seed{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config and host metadata next to the artefacts
    # so the run is reproducible from the directory alone.
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
            },
            indent=2,
        )
    )

    if use_wandb:
        import wandb
        tags = [
            cfg.name,
            cfg.name.split("_d")[0],
            f"D={cfg.ising.D}",
            f"sigma={cfg.ising.sigma}",
            cfg.estimator,
            cfg.model.kind,
            f"seed={seed}",
        ]
        if cfg.ising.target_composition is not None:
            tags.extend(
                [
                    "composition-constrained",
                    f"c_target={cfg.ising.target_composition}",
                    f"lambda_c={cfg.ising.composition_penalty_strength}",
                ]
            )

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.name,
            name=f"{cfg.name}_seed{seed}_{timestamp}",
            config=asdict(cfg),
            tags=tags,
        )

    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Curriculum: target may start at an easier σ; train_loop moves it through
    # piecewise-constant stages. Fixed-σ configs start directly at cfg.ising.
    if cfg.curriculum is not None:
        target_sigma_init = cfg.curriculum.stages[0].sigma
    else:
        target_sigma_init = cfg.ising.sigma
    target = IsingTarget(
        D=cfg.ising.D,
        sigma=target_sigma_init,
        bias=cfg.ising.bias,
        device=device,
        target_composition=cfg.ising.target_composition,
        composition_penalty_strength=cfg.ising.composition_penalty_strength,
    )
    model = _build_model(cfg, target)

    train_loop(
        model=model,
        target=target,
        train_cfg=cfg.train,
        ctmc_cfg=cfg.ctmc,
        eval_cfg=cfg.eval,
        output_dir=run_dir,
        use_wandb=use_wandb,
        estimator_mode=cfg.estimator,
        sigma_curriculum=(
            cfg.curriculum.stages if cfg.curriculum is not None else None
        ),
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
    eval_metrics.update(_trailing_ess_metrics(run_dir))
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
        target_composition=cfg_dict["ising"].get("target_composition"),
        composition_penalty_strength=cfg_dict["ising"].get(
            "composition_penalty_strength", 0.0
        ),
    )
    eval_samples = eval_samples.to(device)
    eval_log_weights = eval_log_weights.to(device)

    eval_metrics = _compute_eval_metrics(eval_samples, eval_log_weights, target)
    eval_metrics.update(_trailing_ess_metrics(run_dir))
    (run_dir / "eval" / "metrics.json").write_text(
        json.dumps(eval_metrics, indent=2)
    )
    return eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        choices=list(CONFIGS.keys()),
        help="Config key from configs.py CONFIGS (training mode)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/01_baseline")
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
    cfg = CONFIGS[args.cfg]
    train(
        cfg,
        seed=args.seed,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
