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
from dataclasses import MISSING, asdict, fields, is_dataclass, replace
from pathlib import Path

import torch
from experiments.constrained_hard_03.configs import (
    CONFIGS,
    HardStageCfg,
    build_swap_head,
)
from experiments.dnfs_baseline_01.configs import CurriculumCfg

from discrete_flow_sampler.diagnostics.metrics import (
    composition_observables,
    ess_from_log_weights,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.resampling import (
    ResamplingConfig,
    log_mean_exp,
)
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)
from discrete_flow_sampler.targets.potts import FixedCompositionPottsTarget

HEAD_KINDS = (
    "doubly_hollow", "mask_one", "non_antisym", "interval", "masked_attention"
)


def smoke_config(cfg: HardStageCfg) -> HardStageCfg:
    """Shrink `cfg` to a minutes-scale end-to-end check (shared by the CLI
    `--smoke` flag and `modal_app.train_remote`'s `smoke` argument, so the
    two entry points can never drift apart)."""
    # A full sigma ladder cannot fit 4 steps (_normalise_curriculum requires
    # start_step < n_steps); keep one stage-0 -> final-sigma transition at the
    # outer-cycle boundary so smoke still exercises the curriculum machinery.
    curriculum = cfg.curriculum
    if curriculum is not None:
        curriculum = CurriculumCfg(
            stages=(
                replace(curriculum.stages[0], start_step=0),
                replace(curriculum.stages[-1], start_step=2),
            )
        )
    return replace(
        cfg,
        train=replace(cfg.train, n_steps=4, inner_steps_per_outer=2),
        ctmc=replace(cfg.ctmc, n_euler_steps=8),
        eval=replace(cfg.eval, n_eval_samples=64),
        curriculum=curriculum,
    )


def build_target_and_head(
    cfg: HardStageCfg, device: str
) -> tuple[IsingTarget, torch.nn.Module]:
    """Shared constructor for the train and eval-only entry points, so the
    two can never drift in how they instantiate the target/backbone/head.

    Both targets sit on a fixed-composition manifold and expose the same
    surface to the swap stack (`swap_log_ratio`, `dt_log_p_tilde_t`,
    `sample_base`, `set_sigma`), so the branch is confined to this one place —
    nothing downstream in `train_swap` or `sample_swap_ctmc` knows which it
    got. `cfg.ising` carries the lattice for both routes; only the
    *composition* differs (scalar n_plus vs S-vector of species counts)."""
    if cfg.target_kind == "potts":
        if cfg.potts_composition is None:
            raise ValueError(
                "target_kind 'potts' requires potts_composition (the "
                "per-species fractions; its length is also S)"
            )
        target = FixedCompositionPottsTarget(
            D=cfg.ising.D,
            sigma=cfg.ising.sigma,
            composition=tuple(cfg.potts_composition),
            device=device,
        )
    else:
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
    # .to(device) on the HEAD, not just the backbone: the wrapper heads are
    # parameterless (no-op), but IntervalSwapHead owns band/position/readout
    # modules that would otherwise stay on CPU (2026-07-07 Modal crash).
    return target, build_swap_head(cfg, backbone).to(device)


def _chunked_eval_draw(
    head, target, cfg: HardStageCfg, *, multi_event: bool, smc_tau: float | None
):
    """Stream the eval draw in `eval_sample_chunk` slices; returns
    (samples, per_sample_log_weights, chunk_stats).

    Plain IS (`smc_tau=None`): weights are independent per sample, so
    slicing changes nothing statistically; chunk_stats is empty.

    SMC (`smc_tau` set): resampling couples particles WITHIN a population,
    so each chunk is an independent SMC population of size `chunk`. The
    returned per-sample log-weight is the pooled form

        ℓ_ci = (banked log-Z increments of chunk c) + (final-segment log w_ci),

    which makes the chunked run one uniform estimator again:
    logmeanexp(ℓ) equals the unbiased chunk-mean of the per-chunk SMC
    product-form Ẑ estimates, and ESS(ℓ) is the ESS of the pooled estimator
    actually used downstream (between-chunk Ẑ spread honestly included).
    Within a chunk the banked part is constant, so per-chunk final-segment
    ESS is still recoverable from the saved ℓ + chunk size.
    """
    device = next(head.parameters()).device
    ts = torch.linspace(0.0, 1.0, cfg.ctmc.n_euler_steps + 1, device=device)
    chunk = cfg.eval.eval_sample_chunk or cfg.eval.n_eval_samples
    sample_slices, log_weight_slices, chunk_stats = [], [], []
    remaining = cfg.eval.n_eval_samples
    with torch.no_grad():
        while remaining > 0:
            x_initial = target.sample_base(min(chunk, remaining), device=device)
            if smc_tau is None:
                slice_samples, slice_log_weights = sample_swap_ctmc(
                    head, x_initial, ts, return_log_weights=True, target=target,
                    multi_event=multi_event,
                )
            else:
                slice_samples, final_segment_log_weights, stats = sample_swap_ctmc(
                    head, x_initial, ts, return_log_weights=True, target=target,
                    multi_event=multi_event,
                    resampling=ResamplingConfig(ess_threshold_fraction=smc_tau),
                )
                slice_log_weights = (
                    stats.log_z_increment + final_segment_log_weights
                )
                final_segment_ess = ess_from_log_weights(final_segment_log_weights)
                chunk_stats.append({
                    "chunk_size": int(x_initial.shape[0]),
                    "n_resample_events": stats.n_events,
                    "event_steps": stats.event_steps,
                    "log_z_increment": float(stats.log_z_increment.item()),
                    "final_segment_ess_fraction": float(
                        final_segment_ess.item() / x_initial.shape[0]
                    ),
                })
            sample_slices.append(slice_samples)
            log_weight_slices.append(slice_log_weights)
            remaining -= x_initial.shape[0]
    return torch.cat(sample_slices), torch.cat(log_weight_slices), chunk_stats


def _composition_metrics(cfg: HardStageCfg, samples: torch.Tensor) -> dict:
    """Composition observables for the eval metrics dict — EMPTY on Potts.

    `diagnostics.metrics.composition_observables` is two-species throughout:
    `composition_fraction_up` computes ((x+1)/2).mean(), which is the fraction
    of +1 spins only when x is binary and is the mean LABEL INDEX once S > 2,
    and `magnetisation` averages the raw spins {-1, 1, 3, ...}. Neither
    RAISES on Potts states — they would write a confident, meaningless number
    into metrics.json, which is worse than writing nothing.

    So the Potts route reports no composition observables until the S-vector
    diagnostics land (planned next: composition_counts, S_q asymmetry,
    delta-based correlators). Nothing is lost from the constraint's point of
    view: composition here is enforced exactly by the swap move set, not
    measured, and `assert_on_manifold` still checks it.
    """
    if cfg.target_kind == "potts":
        return {}
    return composition_observables(
        samples, target_composition=cfg.ising.target_composition,
    )


def final_eval(
    head, target, cfg: HardStageCfg, run_dir: Path, multi_event: bool | None = None
) -> dict:
    """End-of-run eval: (samples, IS log-weights) over the full t = 0 -> 1
    trajectory, streamed in `eval_sample_chunk` slices. The vectorised swap
    head rides d anchor copies per sample, so an unchunked n_eval_samples
    batch OOMs at large d (all three d=64 sigma_c seeds died here,
    2026-07-06); slicing changes nothing statistically because the IS
    weights are independent per sample. Runs fp32 — the bf16 opt-in covers
    the in-training diagnostic eval only.

    `multi_event=None` (the default) resolves to the cell's own canonical
    trajectory step, `cfg.ctmc.use_matching_step` — so a matching-canonical
    cell (the 16x16 rung) lands its matching-step artefacts in plain eval/,
    the dir train() short-circuits on and frozen-eval comparisons read.
    Passing the NON-canonical step explicitly writes a contrast dir instead
    (eval_multi_event/ on a one-event cell — the --compare-multi-event
    probe — or eval_one_event/ on a matching cell), so the canonical
    baseline is never clobbered."""
    if multi_event is None:
        multi_event = cfg.ctmc.use_matching_step
    eval_samples, eval_log_weights, _ = _chunked_eval_draw(
        head, target, cfg, multi_event=multi_event, smc_tau=None
    )

    canonical_step = multi_event == cfg.ctmc.use_matching_step
    step_suffix = (
        "" if canonical_step
        else ("_multi_event" if multi_event else "_one_event")
    )
    eval_dir = run_dir / f"eval{step_suffix}"
    eval_dir.mkdir(exist_ok=True)
    torch.save(eval_samples.cpu(), eval_dir / "samples.pt")
    torch.save(eval_log_weights.cpu(), eval_dir / "log_weights.pt")

    eval_metrics = {
        "n_eval_samples": int(eval_log_weights.numel()),
        "ess": float(ess_from_log_weights(eval_log_weights).item()),
        "head_kind": cfg.head_kind,
        "multi_event": multi_event,
    }
    eval_metrics["ess_fraction"] = eval_metrics["ess"] / eval_metrics["n_eval_samples"]
    eval_metrics.update(_composition_metrics(cfg, eval_samples))
    (eval_dir / "metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    return eval_metrics


def final_eval_smc(
    head,
    target,
    cfg: HardStageCfg,
    run_dir: Path,
    tau: float = 0.5,
    multi_event: bool | None = None,
) -> dict:
    """SMC-resampled end-of-run eval, written ALONGSIDE the plain-IS eval/
    (never over it — the S7 preregistration keeps the pure-IS numbers as
    the quoted baseline).

    Same draw protocol as `final_eval` (n_eval_samples, chunking, Euler
    grid), plus adaptive systematic resampling at threshold `tau` inside
    each chunk (`samplers.resampling`). Saved log_weights.pt holds the
    pooled per-sample weights ℓ (see `_chunked_eval_draw`), so
    `logmeanexp(ℓ)` is the unbiased SMC log-Z estimate — the Eq. 37
    Jensen-LB form is NOT valid on these weights. `n_unique_samples`
    tracks ancestry collapse: resampling duplicates rows, so pooled ESS
    overstates independent-sample count when this drops well below
    n_eval_samples. Artefacts land in eval_smc_tau<τ>/ per (τ, step-kind)
    so sweeps never clobber each other."""
    if multi_event is None:
        multi_event = cfg.ctmc.use_matching_step
    eval_samples, pooled_log_weights, chunk_stats = _chunked_eval_draw(
        head, target, cfg, multi_event=multi_event, smc_tau=tau
    )

    canonical_step = multi_event == cfg.ctmc.use_matching_step
    dir_name = f"eval_smc_tau{tau:g}" + (
        "" if canonical_step
        else ("_multi_event" if multi_event else "_one_event")
    )
    eval_dir = run_dir / dir_name
    eval_dir.mkdir(exist_ok=True)
    torch.save(eval_samples.cpu(), eval_dir / "samples.pt")
    torch.save(pooled_log_weights.cpu(), eval_dir / "log_weights.pt")

    n_samples = int(pooled_log_weights.numel())
    eval_metrics = {
        "n_eval_samples": n_samples,
        "smc_tau": tau,
        "ess": float(ess_from_log_weights(pooled_log_weights).item()),
        "log_z_estimate": float(log_mean_exp(pooled_log_weights).item()),
        "n_resample_events": sum(c["n_resample_events"] for c in chunk_stats),
        "n_unique_samples": int(
            torch.unique(eval_samples, dim=0).shape[0]
        ),
        "chunk_stats": chunk_stats,
        "head_kind": cfg.head_kind,
        "multi_event": multi_event,
    }
    eval_metrics["ess_fraction"] = eval_metrics["ess"] / n_samples
    eval_metrics.update(_composition_metrics(cfg, eval_samples))
    (eval_dir / "metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    return eval_metrics


def train(
    cfg: HardStageCfg,
    seed: int = 42,
    output_dir: str | Path = "results/03_hard",
    use_wandb: bool = True,
    tag: str | None = None,
    on_checkpoint=None,
):
    """Train a swap head on the fixed-composition target and save eval artefacts.

    Preemption resume: with a caller-supplied `tag` (Modal mints one at spawn
    time so retries reuse it), a re-invocation lands in the SAME run dir; a
    `checkpoints/resume.pt` there makes `train_swap` continue instead of
    starting over, and an existing `eval/metrics.json` short-circuits the
    whole call (fully completed run being retried).
    """
    # Apply the per-invocation seed without mutating the frozen config.
    cfg = replace(cfg, train=replace(cfg.train, seed=seed))

    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{cfg.name}_seed{seed}_{tag}"
    if (run_dir / "eval" / "metrics.json").exists():
        print(f"[train] {run_dir.name} already complete; nothing to do")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist the resolved config (incl. the effective head_kind) so the run
    # is reproducible from the directory alone. Written once: on a resumed
    # attempt the original file is the record of what the run started as.
    if not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    if use_wandb:
        import wandb
        # Reuse the first attempt's wandb run on resume so the curve stays a
        # single run (steps already logged past the checkpoint are dropped by
        # wandb's monotonic-step rule -- the same rows the log truncation
        # discards locally).
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
            tags=[
                cfg.name,
                f"D={cfg.ising.D}",
                f"sigma={cfg.ising.sigma}",
                cfg.estimator,
                cfg.head_kind,
                f"seed={seed}",
            ],
        )
        wandb_id_path.write_text(wandb.run.id)

    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target, head = build_target_and_head(cfg, device)
    if cfg.curriculum is not None:
        # Start the flow at the stage-0 coupling so the step-0 stiffness
        # diagnostic fires at the sigma training actually begins from (the
        # curriculum loop in train_swap takes over from outer cycle 0). The
        # final stage restores cfg.ising.sigma before final_eval runs.
        target.set_sigma(cfg.curriculum.stages[0].sigma)

    train_swap(
        head,
        target,
        cfg.train,
        cfg.ctmc,
        cfg.eval,
        run_dir,
        use_wandb=use_wandb,
        estimator_mode=cfg.estimator,
        sigma_curriculum=(
            cfg.curriculum.stages if cfg.curriculum is not None else None
        ),
        on_checkpoint=on_checkpoint,
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


def _backfill_missing_defaults(saved: dict, cfg_class) -> None:
    """Fill defaulted config keys a run dir predates, in place, recursively.

    A run dir written before a defaulted field existed lacks its key. Treat
    that absence as "ran with the then-default" so the drift guard below does
    not lock out every older checkpoint the moment a new field is added. This
    recurses into nested config dataclasses (`model.*`, `eval.*`, ...) — a
    flat pass would leave nested additions looking like real drift.

    Only ABSENT keys are filled; keys that are present must still match
    exactly, so genuine config drift is still caught. This is sound only
    because a newly-added field's default reproduces the prior behaviour —
    check that holds before adding a non-inert default.
    """
    for cfg_field in fields(cfg_class):
        if cfg_field.name not in saved:
            if cfg_field.default is not MISSING:
                saved[cfg_field.name] = json.loads(json.dumps(cfg_field.default))
        elif is_dataclass(cfg_field.type) and isinstance(saved[cfg_field.name], dict):
            _backfill_missing_defaults(saved[cfg_field.name], cfg_field.type)


def eval_only(
    run_dir: str | Path, multi_event: bool | None = None,
    smc_tau: float | None = None,
) -> dict:
    """Re-run the end-of-run eval for a completed run dir (config.json +
    checkpoints/final.pt), writing the eval/ artefacts in place. Recovery
    path for runs whose training finished but whose final eval died before
    the chunked `final_eval` landed (the 2026-07-06 d=64 OOMs), and — with
    `multi_event=True` — the --compare-multi-event probe (same checkpoint,
    same draw protocol, matching step instead of one-event).

    With `smc_tau` set, runs ONLY the SMC-resampled eval (final_eval_smc,
    artefacts to eval_smc_tau<τ>/): the plain-IS eval/ of a completed run
    already exists, and re-drawing it costs real GPU-hours at d=64 — run
    without smc_tau first if it is genuinely missing."""
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    _backfill_missing_defaults(saved, HardStageCfg)
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
    if smc_tau is not None:
        eval_metrics = final_eval_smc(
            head, target, cfg, run_dir, tau=smc_tau, multi_event=multi_event
        )
    else:
        eval_metrics = final_eval(
            head, target, cfg, run_dir, multi_event=multi_event
        )
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
    parser.add_argument(
        "--multi-event",
        action="store_true",
        help="With --eval-only: sample with the vertex-disjoint matching "
        "step instead of the one-event step; writes eval_multi_event/",
    )
    parser.add_argument(
        "--smc-tau",
        type=float,
        default=None,
        metavar="TAU",
        help="With --eval-only: run the SMC-resampled eval (adaptive "
        "systematic resampling when interim ESS < TAU*B) instead of the "
        "plain-IS one; writes eval_smc_tau<TAU>/ alongside eval/",
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
        eval_only(
            args.eval_only,
            # None = the run's own canonical step (cfg.ctmc.use_matching_step);
            # the flag forces the matching step on a one-event cell (the
            # --compare-multi-event probe).
            multi_event=True if args.multi_event else None,
            smc_tau=args.smc_tau,
        )
        return
    if args.smc_tau is not None:
        parser.error("--smc-tau requires --eval-only (SMC is inference-time "
                     "only; run it against a completed run dir)")
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
