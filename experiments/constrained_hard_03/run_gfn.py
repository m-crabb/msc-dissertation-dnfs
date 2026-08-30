"""Entry point for GFlowNet comparator runs (hard chapter).

Usage (local):
    pixi run -e dev python -m experiments.constrained_hard_03.run_gfn \\
        --cfg GFN_d16_c50_s220_tb_10k --seed 42

Mirrors `run.py`'s artefact contract so GFN rows are drop-in comparable:
run dir `{name}_seed{seed}_{tag}` with config.json, host metadata,
training_log.csv, checkpoints/{resume,final}.pt, and eval/ holding
samples.pt + log_weights.pt + metrics.json (ess, ess_fraction, composition
observables). The training loop itself is far simpler than the swap-CTMC
one — no time grid, no rollout: sample trajectories on-policy (epsilon-
mixed), take a TB or FL-DB gradient step, repeat.

The importance weights are exact here in a way the CTMC eval's are not:
log w = log p_tilde(x) - log q_theta(x) with q_theta an exact AR likelihood,
so ESS needs no trajectory estimator, and log-mean-exp of the weights is an
unbiased slice-partition-function estimate (E_q[p_tilde/q] = Z_slice because
the count mask puts q's support exactly on the slice) — recorded in
metrics.json as `log_z_is_estimate` next to the TB arm's learned `log_z`.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS, GFNCellCfg
from experiments.dnfs_baseline_01.run import write_host_metadata

from discrete_flow_sampler.diagnostics.metrics import (
    composition_observables,
    ess_from_log_weights,
)
from discrete_flow_sampler.models.raster_gfn_policy import RasterGFNPolicy
from discrete_flow_sampler.samplers.gfn_objectives import (
    forward_looking_db_loss,
    raster_prefix_log_reward_increments,
    trajectory_balance_loss,
)
from discrete_flow_sampler.samplers.resampling import log_mean_exp
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def build_target_and_policy(cfg: GFNCellCfg, device):
    target = FixedCompositionIsingTarget(
        D=cfg.D, sigma=cfg.sigma,
        target_composition=cfg.target_composition, device=device,
    )
    policy = RasterGFNPolicy(
        D=cfg.D,
        n_plus_target=target.n_plus_target,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        with_flow_head=cfg.with_flow_head,
    ).to(device)
    return target, policy


def _stage_sigma(cfg: GFNCellCfg, step: int) -> float:
    """Annealing ladder: equal step shares per stage, last stage = cfg.sigma.

    (gfn_configs validates sigma_stages[-1] == sigma, so the final target is
    always trained at the cell's own coupling before eval.)
    """
    if not cfg.sigma_stages:
        return cfg.sigma
    stage_length = max(1, cfg.n_steps // len(cfg.sigma_stages))
    stage = min(step // stage_length, len(cfg.sigma_stages) - 1)
    return cfg.sigma_stages[stage]


def _loss_and_train_diagnostics(cfg, policy, target, spins):
    """One objective evaluation; returns (loss, detached log q per sample)."""
    if cfg.objective == "tb":
        model_log_prob = policy.log_prob(spins)
        loss = trajectory_balance_loss(
            policy.log_z, model_log_prob, target.log_prob(spins)
        )
        return loss, model_log_prob.detach()
    site_log_probs, flow_residuals = policy.site_log_probs_and_flow_residuals(spins)
    loss = forward_looking_db_loss(
        site_log_probs,
        raster_prefix_log_reward_increments(target, spins),
        flow_residuals,
    )
    return loss, site_log_probs.detach().sum(dim=-1)


def final_eval_gfn(policy, target, cfg: GFNCellCfg, run_dir: Path) -> dict:
    """End-of-run eval, chunked like run.py's; epsilon=0 (the policy itself).

    assert_on_manifold runs on every draw — the comparator's headline claim
    is feasibility by construction, so a violated eval must crash, not
    average away.
    """
    policy.eval()
    sample_chunks, log_weight_chunks = [], []
    remaining = cfg.n_eval_samples
    while remaining > 0:
        chunk = min(cfg.eval_sample_chunk, remaining)
        spins, log_q = policy.sample(chunk)
        target.assert_on_manifold(spins)
        sample_chunks.append(spins.cpu())
        log_weight_chunks.append((target.log_prob(spins) - log_q).cpu())
        remaining -= chunk
    eval_samples = torch.cat(sample_chunks)
    eval_log_weights = torch.cat(log_weight_chunks)

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    torch.save(eval_samples, eval_dir / "samples.pt")
    torch.save(eval_log_weights, eval_dir / "log_weights.pt")

    eval_metrics = {
        "n_eval_samples": int(eval_log_weights.numel()),
        "ess": float(ess_from_log_weights(eval_log_weights).item()),
        "head_kind": f"gfn_{cfg.objective}",
        "objective": cfg.objective,
        "log_z_is_estimate": float(log_mean_exp(eval_log_weights).item()),
    }
    eval_metrics["ess_fraction"] = (
        eval_metrics["ess"] / eval_metrics["n_eval_samples"]
    )
    if cfg.objective == "tb":
        eval_metrics["log_z_learned"] = float(policy.log_z.item())
    eval_metrics.update(
        composition_observables(
            eval_samples, target_composition=cfg.target_composition
        )
    )
    (eval_dir / "metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    return eval_metrics


def train_gfn(
    cfg: GFNCellCfg,
    seed: int = 42,
    output_dir: str | Path = "results/03_hard",
    use_wandb: bool = True,
    tag: str | None = None,
    on_checkpoint=None,
):
    """Train one GFN cell and write the house artefact set.

    Preemption contract as in run.py: a caller-supplied tag lands retries in
    the SAME run dir, checkpoints/resume.pt continues training in place, and
    an existing eval/metrics.json short-circuits the whole call.
    `on_checkpoint` (Modal passes volume.commit) runs after each resume.pt
    save so resume state survives a preemption that skips the death-flush.
    """
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{cfg.name}_seed{seed}_{tag}"
    if (run_dir / "eval" / "metrics.json").exists():
        print(f"[train_gfn] {run_dir.name} already complete; nothing to do")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(
            json.dumps({**asdict(cfg), "seed": seed}, indent=2)
        )
    write_host_metadata(run_dir)

    if use_wandb:
        import wandb

        wandb_id_path = run_dir / "wandb_run_id.txt"
        stored_run_id = (
            wandb_id_path.read_text().strip() if wandb_id_path.exists() else None
        )
        wandb.init(
            project=cfg.wandb_project,
            group=cfg.name,
            name=f"{cfg.name}_seed{seed}_{tag}",
            config=asdict(cfg),
            id=stored_run_id,
            resume="allow",
            tags=[cfg.name, f"D={cfg.D}", f"sigma={cfg.sigma}",
                  "gfn", cfg.objective, f"seed={seed}"],
        )
        wandb_id_path.write_text(wandb.run.id)

    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target, policy = build_target_and_policy(cfg, device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    resume_path = checkpoint_dir / "resume.pt"
    start_step = 0
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        policy.load_state_dict(saved["policy"])
        optimiser.load_state_dict(saved["optimiser"])
        start_step = saved["step"]
        print(f"[train_gfn] resuming {run_dir.name} from step {start_step}")

    log_path = run_dir / "training_log.csv"
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if start_step == 0 and log_path.stat().st_size == 0:
        log_writer.writerow(
            ["step", "loss", "log_z", "ess_fraction_train", "sigma"]
        )

    policy.train()
    for step in range(start_step, cfg.n_steps):
        target.set_sigma(_stage_sigma(cfg, step))
        spins, _ = policy.sample(cfg.batch_size, epsilon=cfg.epsilon)
        loss, model_log_prob = _loss_and_train_diagnostics(
            cfg, policy, target, spins
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            # In-training ESS on the behaviour batch: a convergence telltale,
            # not the frozen number (epsilon-mixed draws, moving sigma).
            batch_log_weights = target.log_prob(spins) - model_log_prob
            batch_ess_fraction = float(
                ess_from_log_weights(batch_log_weights).item()
            ) / cfg.batch_size
            row = [step, float(loss.item()), float(policy.log_z.item()),
                   batch_ess_fraction, target.sigma]
            log_writer.writerow(row)
            log_file.flush()
            if use_wandb:
                import wandb

                wandb.log(dict(zip(
                    ["step", "loss", "log_z", "ess_fraction_train", "sigma"],
                    row,
                )), step=step)

        if (step + 1) % cfg.checkpoint_every == 0:
            torch.save(
                {"step": step + 1, "policy": policy.state_dict(),
                 "optimiser": optimiser.state_dict()},
                resume_path,
            )
            if on_checkpoint is not None:
                on_checkpoint()

    log_file.close()
    target.set_sigma(cfg.sigma)  # eval always at the cell's own coupling
    torch.save(policy.state_dict(), checkpoint_dir / "final.pt")
    eval_metrics = final_eval_gfn(policy, target, cfg, run_dir)
    print(f"[train_gfn] {run_dir.name}: {json.dumps(eval_metrics, indent=2)}")
    if use_wandb:
        import wandb

        wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()
                   if isinstance(v, (int, float))})
        wandb.finish()
    return run_dir


def smoke_config(cfg: GFNCellCfg) -> GFNCellCfg:
    """Minutes-scale end-to-end shrink, mirroring run.py's --smoke."""
    return replace(
        cfg,
        name=f"{cfg.name}_smoke",
        n_steps=200,
        batch_size=64,
        n_eval_samples=256,
        eval_sample_chunk=128,
        log_every=20,
        checkpoint_every=100,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, choices=sorted(GFN_CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/03_hard")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = GFN_CONFIGS[args.cfg]
    if args.smoke:
        cfg = smoke_config(cfg)
    train_gfn(
        cfg,
        seed=args.seed,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
