"""Entry point for hard-constraint swap-CTMC runs.

Usage (local):
    pixi run -e dev python -m experiments.constrained_hard_03.run \\
        --cfg H2_d16_c50_s010_letf_dh --seed 42

Mirrors `experiments/dnfs_baseline_01/run.py` with the swap-move stack:
`FixedCompositionIsingTarget` + `build_swap_head` + `train_swap` +
`sample_swap_ctmc`. `--smoke` shrinks the run to a minutes-scale check.
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
from experiments.dnfs_baseline_01.run import write_host_metadata

from discrete_flow_sampler.diagnostics.metrics import (
    composition_observables,
    ess_from_log_weights,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.models.rope_vit import RoPEViTRateMatrix
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
    MixtureCompositionIsingTarget,
)
from discrete_flow_sampler.targets.potts import FixedCompositionPottsTarget

HEAD_KINDS = (
    "doubly_hollow",
    "mask_one",
    "non_antisym",
    "interval",
    "masked_attention",
)


def smoke_config(cfg: HardStageCfg) -> HardStageCfg:
    """Shrink `cfg` to a minutes-scale end-to-end check (shared by `--smoke`
    and `modal_app.train_remote`'s `smoke` argument)."""
    # A full sigma ladder cannot fit 4 steps (start_step < n_steps); keep one
    # stage-0 -> final-sigma transition so smoke still exercises the curriculum.
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
    """Shared constructor for the train and eval-only entry points.

    Both targets expose the same surface to the swap stack (`swap_log_ratio`,
    `dt_log_p_tilde_t`, `sample_base`, `set_sigma`), so the branch lives here
    only; `cfg.ising` carries the lattice for both routes and only the
    composition differs (scalar n_plus vs S-vector of species counts)."""
    if cfg.composition_mixture is not None and cfg.target_kind == "potts":
        raise ValueError(
            "composition_mixture and the potts route both claim the target "
            "constructor; the mixture is Ising-only"
        )
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
    elif cfg.target_kind == "cluster_expansion":
        # A real alloy on the canonical rung: the exported expansion's energy
        # in place of the torus quadratic form; cfg.ising.sigma is beta/2.
        from discrete_flow_sampler.targets.cluster_expansion import (
            BinaryExpansionSpec,
            FixedCompositionClusterExpansionTarget,
        )

        if cfg.ising.expansion_json is None:
            raise ValueError(
                "target_kind 'cluster_expansion' needs ising.expansion_json"
            )
        spec = BinaryExpansionSpec.from_json(cfg.ising.expansion_json)
        if cfg.composition_mixture is not None:
            # Amortisation on the alloy: the same slice mixture as the Ising
            # route below, on the expansion's energy.
            from discrete_flow_sampler.targets.cluster_expansion import (
                MixtureCompositionClusterExpansionTarget,
            )

            target = MixtureCompositionClusterExpansionTarget(
                spec,
                beta=2.0 * cfg.ising.sigma,
                compositions=tuple(cfg.composition_mixture),
                bias=cfg.ising.bias,
                device=device,
            )
        else:
            target = FixedCompositionClusterExpansionTarget(
                spec,
                beta=2.0 * cfg.ising.sigma,
                target_composition=cfg.ising.target_composition,
                bias=cfg.ising.bias,
                device=device,
            )
    elif cfg.composition_mixture is not None:
        # Amortisation route: mixture of slices in the base, everything
        # downstream per-slice exact (swaps conserve composition row-wise).
        target = MixtureCompositionIsingTarget(
            D=cfg.ising.D,
            sigma=cfg.ising.sigma,
            compositions=tuple(cfg.composition_mixture),
            bias=cfg.ising.bias,
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
    backbone_kwargs = dict(
        d=target.d,
        vocab_size=cfg.model.vocab_size,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        use_sdpa_readout=cfg.model.use_sdpa_readout,
    )
    if cfg.model.kind == "rope_vit":
        backbone = RoPEViTRateMatrix(patch_size=cfg.model.patch_size, **backbone_kwargs)
    else:
        backbone = LeTFRateMatrix(**backbone_kwargs)
    backbone = backbone.to(device)
    # .to(device) on the head too: IntervalSwapHead owns band/position/readout
    # modules that would otherwise stay on CPU (2026-07-07 Modal crash).
    return target, build_swap_head(cfg, backbone, target).to(device)


def _chunked_eval_draw(
    head, target, cfg: HardStageCfg, *, multi_event: bool, smc_tau: float | None
):
    """Stream the eval draw in `eval_sample_chunk` slices; returns
    (samples, per_sample_log_weights, chunk_stats, transport_stats).

    `transport_stats` accumulates the sampler's additive swap counters
    (proposed / accepted / accepted_state_changing / state_steps) across all
    slices; without it an eval can post a healthy ESS while firing almost no
    state-changing swaps, i.e. while sampling the base rather than the target.

    Plain IS (`smc_tau=None`): weights are independent per sample, so slicing
    changes nothing; chunk_stats is empty. SMC (`smc_tau` set): resampling
    couples particles within a population, so each chunk is an independent
    SMC population of size `chunk` and the per-sample log-weight is pooled:

        ℓ_ci = (banked log-Z increments of chunk c) + (final-segment log w_ci),

    so logmeanexp(ℓ) is the chunk-mean of the per-chunk product-form Ẑ and
    ESS(ℓ) is the ESS of the pooled estimator used downstream. The banked
    part is constant within a chunk, so per-chunk final-segment ESS is
    recoverable from ℓ + chunk size.
    """
    device = next(head.parameters()).device
    ts = torch.linspace(0.0, 1.0, cfg.ctmc.n_euler_steps + 1, device=device)
    chunk = cfg.eval.eval_sample_chunk or cfg.eval.n_eval_samples
    sample_slices, log_weight_slices, chunk_stats = [], [], []
    transport_stats: dict = {}
    remaining = cfg.eval.n_eval_samples
    with torch.no_grad():
        while remaining > 0:
            x_initial = target.sample_base(min(chunk, remaining), device=device)
            if smc_tau is None:
                slice_samples, slice_log_weights = sample_swap_ctmc(
                    head,
                    x_initial,
                    ts,
                    return_log_weights=True,
                    target=target,
                    multi_event=multi_event,
                    matching_stats=transport_stats,
                )
            else:
                slice_samples, final_segment_log_weights, stats = sample_swap_ctmc(
                    head,
                    x_initial,
                    ts,
                    return_log_weights=True,
                    target=target,
                    multi_event=multi_event,
                    matching_stats=transport_stats,
                    resampling=ResamplingConfig(ess_threshold_fraction=smc_tau),
                )
                slice_log_weights = stats.log_z_increment + final_segment_log_weights
                final_segment_ess = ess_from_log_weights(final_segment_log_weights)
                chunk_stats.append(
                    {
                        "chunk_size": int(x_initial.shape[0]),
                        "n_resample_events": stats.n_events,
                        "event_steps": stats.event_steps,
                        "log_z_increment": float(stats.log_z_increment.item()),
                        "final_segment_ess_fraction": float(
                            final_segment_ess.item() / x_initial.shape[0]
                        ),
                    }
                )
            sample_slices.append(slice_samples)
            log_weight_slices.append(slice_log_weights)
            remaining -= x_initial.shape[0]
    return (
        torch.cat(sample_slices),
        torch.cat(log_weight_slices),
        chunk_stats,
        transport_stats,
    )


def _composition_metrics(cfg: HardStageCfg, samples: torch.Tensor) -> dict:
    """Composition observables for the eval metrics dict; empty on Potts.

    `composition_observables` is two-species: `composition_fraction_up` is
    the mean label index once S > 2 and `magnetisation` averages raw spins
    {-1, 1, 3, ...}. Neither raises on Potts states, so the Potts route writes
    nothing rather than a confident, meaningless number until S-vector
    diagnostics land. Composition is enforced by the swap move set and
    checked by `assert_on_manifold` regardless.
    """
    if cfg.target_kind == "potts":
        return {}
    return composition_observables(
        samples,
        target_composition=cfg.ising.target_composition,
    )


def _eval_output_dir(
    run_dir: Path,
    cfg: HardStageCfg,
    multi_event: bool,
    eval_dir_suffix: str = "",
    replicate_seed: int | None = None,
    smc_tau: float | None = None,
) -> Path:
    """Keep every draw's selection in its path and refuse existing artefacts.

    Partial draws count too: overwriting their tensors before writing metrics
    could leave a complete-looking directory of mixed generations.
    """
    prefix = "eval" if smc_tau is None else f"eval_smc_tau{smc_tau:g}"
    step_suffix = (
        ""
        if multi_event == cfg.ctmc.use_matching_step
        else ("_multi_event" if multi_event else "_one_event")
    )
    replicate_suffix = "" if replicate_seed is None else f"_replicate_s{replicate_seed}"
    eval_dir = run_dir / f"{prefix}{step_suffix}{replicate_suffix}{eval_dir_suffix}"
    if any(
        (eval_dir / name).exists()
        for name in ("samples.pt", "log_weights.pt", "metrics.json")
    ):
        raise FileExistsError(
            f"refusing to overwrite existing eval artefacts in {eval_dir}"
        )
    return eval_dir


def final_eval(
    head,
    target,
    cfg: HardStageCfg,
    run_dir: Path,
    multi_event: bool | None = None,
    replicate_seed: int | None = None,
    eval_dir_suffix: str = "",
) -> dict:
    """End-of-run eval: (samples, IS log-weights) over the full t = 0 -> 1
    trajectory, streamed in `eval_sample_chunk` slices (the vectorised head
    rides d anchor copies per sample; unchunked, all three d=64 sigma_c seeds
    OOMed, 2026-07-06). Runs fp32; the bf16 opt-in covers the in-training
    diagnostic eval only.

    `multi_event=None` resolves to the cell's canonical step,
    `cfg.ctmc.use_matching_step`, and writes plain eval/. Passing the
    non-canonical step explicitly writes a contrast dir (eval_multi_event/ or
    eval_one_event/), so the canonical baseline is never clobbered."""
    if multi_event is None:
        multi_event = cfg.ctmc.use_matching_step
    eval_dir = _eval_output_dir(
        run_dir, cfg, multi_event, eval_dir_suffix, replicate_seed
    )
    eval_samples, eval_log_weights, _, transport_stats = _chunked_eval_draw(
        head, target, cfg, multi_event=multi_event, smc_tau=None
    )

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
    # Jump budget integrated along the trajectory: swap events per site over
    # the full t=0->1 path. "accepted" includes same-spin no-op swaps;
    # "state_changing" is productive transport, which ESS alone cannot certify.
    per_site_trajectory_norm = cfg.ctmc.n_euler_steps / (
        float(transport_stats["state_steps"]) * target.d
    )
    eval_metrics["jumps_per_site_proposed"] = (
        float(transport_stats["proposed"]) * per_site_trajectory_norm
    )
    eval_metrics["jumps_per_site_accepted"] = (
        float(transport_stats["accepted"]) * per_site_trajectory_norm
    )
    eval_metrics["jumps_per_site_state_changing"] = (
        float(transport_stats["accepted_state_changing"]) * per_site_trajectory_norm
    )
    if replicate_seed is not None:
        eval_metrics["replicate_seed"] = replicate_seed
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
    eval_dir_suffix: str = "",
) -> dict:
    """SMC-resampled end-of-run eval, written alongside the plain-IS eval/
    (the pure-IS numbers stay the quoted baseline).

    Same draw protocol as `final_eval`, plus adaptive systematic resampling
    at threshold `tau` inside each chunk (`samplers.resampling`). Saved
    log_weights.pt holds the pooled ℓ (see `_chunked_eval_draw`):
    `exp(logmeanexp(ℓ))` is the SMC Z estimate (unbiased under valid
    importance-weight and resampling assumptions; finite Euler steps and rate
    clipping are not covered), its logarithm is Jensen-biased, and the Eq. 37
    Jensen-LB form is not valid on these weights. `n_unique_samples` tracks
    ancestry collapse: resampling duplicates rows, so pooled ESS overstates
    the independent-sample count when it drops well below n_eval_samples.
    Artefacts land in eval_smc_tau<τ>/ per (τ, step-kind)."""
    if multi_event is None:
        multi_event = cfg.ctmc.use_matching_step
    eval_dir = _eval_output_dir(run_dir, cfg, multi_event, eval_dir_suffix, smc_tau=tau)
    eval_samples, pooled_log_weights, chunk_stats, _ = _chunked_eval_draw(
        head, target, cfg, multi_event=multi_event, smc_tau=tau
    )

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
        "n_unique_samples": int(torch.unique(eval_samples, dim=0).shape[0]),
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
    init_from: str | Path | None = None,
):
    """Train a swap head on the fixed-composition target and save eval artefacts.

    Preemption resume: with a caller-supplied `tag` (Modal mints one at spawn
    so retries reuse it), a re-invocation lands in the same run dir; a
    `checkpoints/resume.pt` there makes `train_swap` continue, and an
    existing `eval/metrics.json` short-circuits the whole call.
    """
    # Apply the per-invocation seed without mutating the frozen config.
    cfg = replace(cfg, train=replace(cfg.train, seed=seed))

    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{cfg.name}_seed{seed}_{tag}"
    if (run_dir / "eval" / "metrics.json").exists():
        print(f"[train] {run_dir.name} already complete; nothing to do")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist the resolved config so the run is reproducible from the directory
    # alone; written once, so a resumed attempt keeps the original record.
    config_path = run_dir / "config.json"
    resume_path = run_dir / "checkpoints" / "resume.pt"
    if resume_path.exists() and not config_path.exists():
        raise ValueError(f"cannot resume {run_dir} without its original config.json")
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        _backfill_missing_defaults(saved, HardStageCfg)
        if json.loads(json.dumps(asdict(cfg))) != saved:
            raise ValueError(
                f"config.json in {run_dir} does not match requested config"
            )
    else:
        config_path.write_text(json.dumps(asdict(cfg), indent=2))
    # Record which GPU produced this run's wall clocks.
    write_host_metadata(run_dir)

    if use_wandb:
        import wandb

        # Reuse the first attempt's wandb run on resume so the curve stays one
        # run; wandb's monotonic-step rule drops the rows the log truncation drops.
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
    if init_from is not None and not resume_path.exists():
        # Warm-start from scripts/warm_start_swap_head.py's partial state dict;
        # strict=False because the transfer omits reinitialised keys (e.g. the
        # dead attention_readout table). The printed report records what loaded.
        transfer = torch.load(init_from, map_location=device, weights_only=True)
        missing, unexpected = head.load_state_dict(transfer, strict=False)
        print(
            f"[init_from] {init_from}: loaded {len(transfer)} keys, "
            f"missing {sorted(missing)}, unexpected {sorted(unexpected)}"
        )
        if unexpected:
            raise ValueError(f"init_from has keys the model lacks: {unexpected}")
        (run_dir / "init_from.txt").write_text(
            f"{init_from}\nmissing (kept fresh init): {sorted(missing)}\n"
        )
    if cfg.curriculum is not None:
        # Start at the stage-0 coupling so the step-0 stiffness diagnostic fires
        # at the sigma training begins from; the final stage restores cfg.ising.sigma.
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
        ema_decay=getattr(cfg, "ema_decay", 0.0),
    )

    eval_metrics = final_eval(head, target, cfg, run_dir)

    # eval/ (raw parameters, comparable to every archived cell) is written
    # first; the EMA reading lands alongside in eval_ema/.
    ema_metrics = None
    final_ema_path = run_dir / "checkpoints" / "final_ema.pt"
    if final_ema_path.exists():
        head.load_state_dict(
            torch.load(final_ema_path, map_location=target.device, weights_only=True)
        )
        ema_metrics = final_eval(head, target, cfg, run_dir, eval_dir_suffix="_ema")

    if use_wandb:
        wandb.log(
            {
                f"eval/{key}": value
                for key, value in eval_metrics.items()
                if isinstance(value, (int, float))
            }
        )
        if ema_metrics is not None:
            wandb.log(
                {
                    f"eval_ema/{key}": value
                    for key, value in ema_metrics.items()
                    if isinstance(value, (int, float))
                }
            )
        wandb.finish()

    return run_dir


def _backfill_missing_defaults(saved: dict, cfg_class) -> None:
    """Fill defaulted config keys a run dir predates, in place, recursively.

    A missing key is read as "ran with the then-default", so the drift guard
    does not lock out older checkpoints when a field is added; nested
    dataclasses (`model.*`, `eval.*`) are recursed. Present keys must still
    match exactly. Sound only while a new field's default reproduces the prior
    behaviour; check that before adding a non-inert default.
    """
    for cfg_field in fields(cfg_class):
        if cfg_field.name not in saved:
            if cfg_field.default is not MISSING:
                saved[cfg_field.name] = json.loads(json.dumps(cfg_field.default))
        elif is_dataclass(cfg_field.type) and isinstance(saved[cfg_field.name], dict):
            _backfill_missing_defaults(saved[cfg_field.name], cfg_field.type)


def _eval_checkpoint_and_suffix(
    use_ema: bool, n_euler_override: int | None, stage_best: int | None
) -> tuple[str, str]:
    """Resolve which checkpoint an eval-only pass reads and where it writes.

    Returned together because a draw from the wrong weights returns a
    plausible number and nothing in the artefacts records which file was
    read, so every checkpoint choice must also move the output directory.
    `stage_best` selects `best_stage<k>.pt`, saved by the swap trainer at the
    best trailing median-of-3 train-eval ESS within that curriculum stage
    (`stage_best_checkpoints=True`); it carries no EMA shadow, so pairing it
    with `use_ema` is refused rather than served from `final_ema.pt`.
    """
    if stage_best is not None and stage_best < 0:
        raise ValueError(f"stage index must be non-negative, got {stage_best}")
    if stage_best is not None and use_ema:
        raise ValueError(
            "stage-best checkpoints carry no EMA twin (the trainer saves raw "
            "head weights per stage); the selection read is raw-vs-raw "
            "against final.pt"
        )
    checkpoint = (
        f"best_stage{stage_best}.pt"
        if stage_best is not None
        else ("final_ema.pt" if use_ema else "final.pt")
    )
    suffix = (
        ("" if stage_best is None else f"_stage{stage_best}")
        + ("_ema" if use_ema else "")
        + ("" if n_euler_override is None else f"_ne{n_euler_override}")
    )
    return checkpoint, suffix


def eval_only(
    run_dir: str | Path,
    multi_event: bool | None = None,
    smc_tau: float | None = None,
    replicate_seed: int | None = None,
    n_euler_override: int | None = None,
    use_ema: bool = False,
    stage_best: int | None = None,
) -> dict:
    """Re-run the end-of-run eval for a completed run dir (config.json +
    checkpoints/final.pt) into an empty eval destination; existing samples,
    weights or metrics, partial draws included, are refused. Recovery path
    for runs whose final eval died after training (the 2026-07-06 d=64 OOMs)
    and, with `multi_event=True`, the --compare-multi-event probe.

    `smc_tau`: runs only the SMC eval (`final_eval_smc`, eval_smc_tau<τ>/);
    the plain-IS eval/ already exists and costs real GPU-hours at d=64.
    `replicate_seed`: an independent draw off the converged checkpoint with a
    fresh eval seed, into eval_replicate_s<seed>/. Plain IS only, since the
    probe's N_eff(O) comparison is defined on unresampled weights.
    `n_euler_override` (2026-08-18): re-draw on another time grid, into
    eval_ne<k>/, decoupling model quality from discretisation; the numbers
    are not the frozen eval/ numbers and must never be quoted as them. Plain
    IS only.
    `use_ema` (2026-08-20): load `final_ema.pt`, the primary read for every
    d=256 cell (raw ESS at sigma_c is top-weight dominated); artefacts gain
    `_ema` (eval_ema_ne<k>/). A draw into an empty eval_ema/ is allowed for
    recovery; a frozen one is refused.
    `stage_best`: read `checkpoints/best_stage<k>.pt`, write eval_stage<k>/,
    the checkpoint-selection read (stage-best vs final.pt, compared only
    against a bootstrap CI since a best-of-~25 trailing medians carries
    upward selection bias). Raw weights both sides; EMA pairing refused
    (`_eval_checkpoint_and_suffix`)."""
    checkpoint_name, eval_dir_suffix = _eval_checkpoint_and_suffix(
        use_ema, n_euler_override, stage_best
    )
    if (
        use_ema
        and n_euler_override is None
        and smc_tau is None
        and replicate_seed is None
        and multi_event is None
        and (Path(run_dir) / "eval_ema" / "metrics.json").exists()
    ):
        # Refused only when a frozen EMA eval is actually there; when the
        # trainer died between the raw and EMA evals (d256 camort, 2026-09-01:
        # final_ema.pt on disk, eval_ema/ never written) this is the recovery path.
        raise ValueError(
            "use_ema without n_euler_override would overwrite eval_ema/, the "
            "frozen EMA eval written by the training run; pass a grid "
            "override (artefacts land in eval_ema_ne<k>/) or read eval_ema/"
        )
    if smc_tau is not None and replicate_seed is not None:
        raise ValueError(
            "replicate draws are plain-IS by construction; "
            "run smc_tau and replicate_seed evals separately"
        )
    if smc_tau is not None and n_euler_override is not None:
        raise ValueError(
            "the grid-decoupling probe is plain-IS; run smc_tau and "
            "n_euler_override evals separately"
        )
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
    # Applied after the drift guard: provenance is checked against the frozen
    # config, and only this draw's sampling grid moves.
    if n_euler_override is not None:
        cfg = replace(cfg, ctmc=replace(cfg.ctmc, n_euler_steps=n_euler_override))

    seed_everything(cfg.train.seed if replicate_seed is None else replicate_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target, head = build_target_and_head(cfg, device)
    head.load_state_dict(
        torch.load(
            run_dir / "checkpoints" / checkpoint_name,
            map_location=device,
            weights_only=True,
        )
    )
    if smc_tau is not None:
        eval_metrics = final_eval_smc(
            head,
            target,
            cfg,
            run_dir,
            tau=smc_tau,
            multi_event=multi_event,
            eval_dir_suffix=eval_dir_suffix,
        )
    else:
        eval_metrics = final_eval(
            head,
            target,
            cfg,
            run_dir,
            multi_event=multi_event,
            replicate_seed=replicate_seed,
            eval_dir_suffix=eval_dir_suffix,
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
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="With --eval-only: draw a probe replicate with this fresh "
        "sampling seed; artefacts land in "
        "eval_replicate_s<SEED>/ beside the frozen eval/",
    )
    parser.add_argument(
        "--eval-ne",
        type=int,
        default=None,
        metavar="K",
        help="With --eval-only: re-draw on a K-step Euler grid instead of "
        "the cell's own; writes eval_ne<K>/ (or eval_ema_ne<K>/ with "
        "--eval-ema). Decouples model quality from discretisation.",
    )
    parser.add_argument(
        "--eval-ema",
        action="store_true",
        help="With --eval-only and --eval-ne: draw from checkpoints/"
        "final_ema.pt instead of final.pt. Requires --eval-ne so the "
        "artefacts cannot overwrite the frozen eval_ema/.",
    )
    parser.add_argument(
        "--stage-best",
        type=int,
        default=None,
        metavar="K",
        help="With --eval-only: draw from checkpoints/best_stage<K>.pt "
        "instead of final.pt; writes eval_stage<K>/. The "
        "checkpoint-selection read (sigma_c stage-best vs final).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/03_hard")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--tag", default=None, help="Run-dir suffix (default: wall-clock timestamp)"
    )
    parser.add_argument(
        "--init-from",
        default=None,
        metavar="STATE_DICT_PT",
        help="Warm-start: load this (possibly partial) state dict into the "
        "head before training (see scripts/warm_start_swap_head.py)",
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
            # None = the run's own canonical step; the flag forces the matching
            # step on a one-event cell (the --compare-multi-event probe).
            multi_event=True if args.multi_event else None,
            smc_tau=args.smc_tau,
            replicate_seed=args.eval_seed,
            n_euler_override=args.eval_ne,
            use_ema=args.eval_ema,
            stage_best=args.stage_best,
        )
        return
    if args.eval_ne is not None or args.eval_ema:
        parser.error(
            "--eval-ne/--eval-ema require --eval-only (both move "
            "only the draw, so they run against a completed run dir)"
        )
    if args.smc_tau is not None:
        parser.error(
            "--smc-tau requires --eval-only (SMC is inference-time "
            "only; run it against a completed run dir)"
        )
    if args.eval_seed is not None:
        parser.error(
            "--eval-seed requires --eval-only (replicate draws run "
            "against a completed run dir's checkpoint)"
        )
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
        init_from=args.init_from,
    )


if __name__ == "__main__":
    main()
