"""Entry point for DNFS Ising baseline runs.

Usage (local):
    pixi run -e dev python -m experiments.dnfs_baseline_01.run \\
        --cfg stage_1_d4 --seed 42

Or, to recompute eval metrics from saved samples without re-training:
    pixi run -e dev python -m experiments.dnfs_baseline_01.run \\
        --eval-only --run-dir results/01_baseline/stage_1_d4_seed42_...

Or, for an amortised (composition-conditioned) run, to re-draw the eval at
each composition in turn and write per-composition rows:
    pixi run -e dev python -m experiments.dnfs_baseline_01.run \\
        --sweep --run-dir results/02_constrained_soft/S2_d10_camort_..._seed42

The same `train(cfg, seed, ...)` function is also imported by
`modal_app.py` for remote runs, so both paths share artefacts and metadata.
"""
import argparse
import json
import platform
import shutil
import socket
import time
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import torch
from experiments.dnfs_baseline_01.configs import (
    CONFIGS,
    CTMCCfg,
    EvalCfg,
    IsingCfg,
    ModelCfg,
    StageCfg,
)

from discrete_flow_sampler.diagnostics.metrics import (
    composition_observables,
    entropy_estimate,
    ess_from_log_weights,
    exact_free_energy,
    exact_internal_energy,
    free_energy_lb_estimate,
    internal_energy_estimate,
)
from discrete_flow_sampler.models.composition_conditioned import (
    CompositionConditioned,
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

# The compositions an amortised model is measured at. The first group has a
# per-composition specialist on disk under results/02_constrained_soft, so
# those rows are a direct amortised-vs-specialist comparison at matched
# compute. The second group was never trained by anything: it sits between the
# specialists' values, so it separates a model that interpolates across the
# composition axis from one that memorised the atoms it was trained on.
SPECIALIST_COMPOSITIONS = (0.30, 0.50, 0.55, 0.60, 0.65, 0.80)
HELD_OUT_COMPOSITIONS = (0.35, 0.45, 0.575, 0.70)
SWEEP_COMPOSITIONS = tuple(
    sorted(SPECIALIST_COMPOSITIONS + HELD_OUT_COMPOSITIONS)
)


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
            condition_on_composition=cfg.model.condition_on_composition,
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
    composition: float | None = None,
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
        composition: the composition this eval was conditioned on, or None to
            read the target's own scalar. It has to be passed explicitly
            because an amortised eval's composition lives in the target's
            binding, not in `target.target_composition` — labelling the
            observables from the fallback scalar would quietly attribute a
            c = 0.80 draw to whatever the config happened to record.
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
            target_composition=(
                composition
                if composition is not None
                else getattr(target, "target_composition", None)
            ),
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


def _composition_binding(target, composition: float | None, device):
    """(model wrapper, target binding) for one composition, or the bare pair.

    `composition=None` is the specialist route: the model is called as it
    always was and nothing is bound, so an archived cell executes exactly the
    code it did before amortisation existed.

    Otherwise the composition is bound in both places it is read. The model
    needs it as an input — a conditioned model raises when c is missing rather
    than silently predicting rates for some other composition — and the target
    needs it because the composition penalty (and, on the fixed-composition
    route, the manifold itself) is what makes p_c differ from p.
    """
    if composition is None:
        return (lambda model: model), nullcontext()
    bound = torch.full((1,), float(composition), device=device)
    return (
        lambda model: CompositionConditioned(model, bound),
        target.composition_batch(bound),
    )


def _eval_at_composition(model, target, cfg, composition: float | None, device):
    """One t = 0 → 1 eval draw and its metrics, at a single composition.

    Returns `(samples, log_weights, metrics)`.

    The draw and the scoring deliberately share ONE binding. The IS
    log-weights are accumulated along the path against log p̃_t at the bound
    composition, so the free energy (Eq. 37) and internal energy (Eq. 38) read
    off them must use that same composition; scoring them under a different
    one — or unbound, where the target falls back to its scalar
    `target_composition` — mixes two densities into a single estimate and
    raises nothing at all.

    The base draw goes through `target.sample_base` rather than an inline
    `torch.randint`. At the uniform base the two are identical draws (same RNG
    consumption, so archived runs still reproduce bit-for-bit), but the base is
    a property of the target: a composition-dependent base — a Bernoulli(c)
    base, or the fixed-composition route's slice — would make an inline draw
    silently wrong.
    """
    time_grid = torch.linspace(0.0, 1.0, cfg.ctmc.n_euler_steps, device=device)
    wrap, binding = _composition_binding(target, composition, device)
    with torch.no_grad(), binding:
        draw_start = time.perf_counter()
        x_initial = target.sample_base(cfg.eval.n_eval_samples, device=device)
        samples, log_weights = sample_ctmc(
            wrap(model),
            x_initial,
            time_grid,
            return_log_weights=True,
            target=target,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        draw_seconds = time.perf_counter() - draw_start
        metrics = _compute_eval_metrics(
            samples, log_weights, target, composition=composition
        )
    # Cost of the draw, recorded because it cannot be recovered later: no
    # archived run carries any timing, and the runs are spread across several
    # machines, so a cost axis has to start accumulating from here.
    #
    # `wall_clock_step_s` in the training log is NOT this: it times only the
    # inner loss update, excluding trajectory generation and the eval draw.
    #
    # Seconds are machine-specific, so the derived quantity is the one to
    # quote across runs: `nfe_per_effective_sample` is Euler steps × samples
    # drawn, divided by ESS — a hardware-independent cost-per-good-sample that
    # prices the Euler budget honestly (ne128 costs 2× ne64 per sample and has
    # to earn it back in ESS) and is comparable to published pure-IS tables.
    metrics["eval_draw_seconds"] = draw_seconds
    metrics["eval_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    effective = metrics["ess"]
    metrics["nfe_per_effective_sample"] = (
        cfg.ctmc.n_euler_steps * cfg.eval.n_eval_samples / effective
        if effective > 0 else float("inf")
    )
    return samples, log_weights, metrics


def train(
    cfg: StageCfg,
    seed: int = 42,
    output_dir: str | Path = "results/01_baseline",
    use_wandb: bool = True,
    tag: str | None = None,
):
    """Top-level training entry. Importable from CLI or modal_app.

    Builds the target / model / estimator from the resolved config, kicks off
    `samplers.training.train`, and persists end-of-run eval samples + IS
    log-weights for downstream analysis notebooks.

    `tag` replaces the run dir's wall-clock timestamp suffix (mirroring the
    hard experiment's runner): a Slurm job resubmitted after preemption then
    lands in the SAME run dir instead of minting a sibling, and a run that
    already finished is detected and skipped. Unlike the hard runner there is
    NO mid-run checkpoint resume here — a retried run restarts from step 0,
    overwriting in place — so a fixed tag buys idempotency for completed runs
    and a stable directory identity, not warm continuation.
    """
    # Apply the per-invocation seed without mutating the frozen config.
    cfg = replace(cfg, train=replace(cfg.train, seed=seed))

    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{cfg.name}_seed{seed}_{tag}"
    # eval/metrics.json is the last artefact train writes, so its presence
    # means the run completed; the guard must fire before any file is
    # (re)written so a resubmitted finished job leaves the record untouched.
    # With the default timestamp suffix the dir is always fresh and this
    # never triggers, keeping every archived run's semantics unchanged.
    if (run_dir / "eval" / "metrics.json").exists():
        print(f"[train] {run_dir.name} already complete; nothing to do")
        return run_dir
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
            name=f"{cfg.name}_seed{seed}_{tag}",
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
    # λ anneal mirrors the σ curriculum: the target starts at the stage-0
    # penalty strength and train_loop tightens it through the stages. It is
    # plumbed here, not in constrained_soft_02, because this `train` is the
    # shared entry point that the constrained modal_app delegates to; for
    # baseline (unconstrained) configs lambda_curriculum is None and this
    # block is a no-op.
    if cfg.lambda_curriculum is not None:
        target_lambda_init = (
            cfg.lambda_curriculum.stages[0].composition_penalty_strength
        )
    else:
        target_lambda_init = cfg.ising.composition_penalty_strength
    target = IsingTarget(
        D=cfg.ising.D,
        sigma=target_sigma_init,
        bias=cfg.ising.bias,
        device=device,
        target_composition=cfg.ising.target_composition,
        composition_penalty_strength=target_lambda_init,
        base_composition=cfg.ising.base_composition,
        log_ratio_clamp=cfg.ising.log_ratio_clamp,
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
        lambda_curriculum=(
            cfg.lambda_curriculum.stages
            if cfg.lambda_curriculum is not None
            else None
        ),
        composition_centre=(
            cfg.composition.centre if cfg.composition is not None else None
        ),
        composition_half_width=(
            cfg.composition.half_width if cfg.composition is not None else 0.0
        ),
        composition_values=(
            cfg.composition.values if cfg.composition is not None else None
        ),
        composition_curriculum=(
            cfg.composition.curriculum if cfg.composition is not None else None
        ),
    )

    # End-of-run eval: a final batch of (samples, IS log-weights) over the
    # full t = 0 -> 1 trajectory. Analysis notebooks read these directly.
    #
    # An amortised model serves a whole range of compositions, so a single
    # draw has to pick one. It picks the window centre, matching the in-loop
    # ESS probe, so the training curve and this final number describe the same
    # conditional model. The per-composition picture is a separate sweep
    # (`composition_sweep`) over the trained checkpoint — this draw is not it,
    # and must not be read as it.
    eval_composition = (
        None if cfg.composition is None else float(cfg.composition.centre)
    )
    eval_samples, eval_log_weights, eval_metrics = _eval_at_composition(
        model, target, cfg, eval_composition, device
    )
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    torch.save(eval_samples.cpu(), eval_dir / "samples.pt")
    torch.save(eval_log_weights.cpu(), eval_dir / "log_weights.pt")

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


def _sub_config(cls, values: dict):
    """Rebuild one config dataclass from a run dir's `config.json`.

    Keys the dataclass no longer has are dropped, and keys it has since gained
    fall back to their defaults. A run dir is meant to stay evaluable from the
    directory alone, and a strict constructor makes every historical run
    un-evaluable the moment a config grows a knob.
    """
    known = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in known})


def _rebuild_from_run_dir(run_dir: Path):
    """(cfg, target, device) for a finished run, from `config.json` alone.

    The returned cfg carries only the sub-configs the eval paths read
    (`ising`, `model`, `ctmc`, `eval`); the curricula are training-time
    schedules and are deliberately not replayed. That means the target is
    rebuilt at the *final* σ and λ — the operating point the run ended at,
    which is what the recorded eval numbers belong to.
    """
    cfg_dict = json.loads((run_dir / "config.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ising = _sub_config(IsingCfg, cfg_dict["ising"])
    target = IsingTarget(
        D=ising.D,
        sigma=ising.sigma,
        bias=ising.bias,
        device=device,
        target_composition=ising.target_composition,
        composition_penalty_strength=ising.composition_penalty_strength,
        base_composition=ising.base_composition,
    )
    cfg = SimpleNamespace(
        ising=ising,
        model=_sub_config(ModelCfg, cfg_dict["model"]),
        ctmc=_sub_config(CTMCCfg, cfg_dict["ctmc"]),
        eval=_sub_config(EvalCfg, cfg_dict["eval"]),
        # The window centre is the composition a single eval draw is
        # conditioned on; None for a specialist run.
        composition_centre=(
            None if cfg_dict.get("composition") is None
            else float(cfg_dict["composition"]["centre"])
        ),
        condition_on_composition=cfg_dict["model"].get(
            "condition_on_composition", False
        ),
    )
    return cfg, target, device


def eval_only(
    run_dir: str | Path, redraw: bool = False, redraw_seed: int = 0,
    n_euler_override: int | None = None,
) -> dict:
    """Recompute eval metrics from a finished run's saved samples.

    Loads `eval/samples.pt` + `eval/log_weights.pt`, reconstructs the
    target from `config.json`, and writes / overwrites `eval/metrics.json`
    in the run directory. Useful for backfilling diagnostics on older
    runs whose training pre-dated the metrics-aggregation block.

    For an amortised run the saved samples were drawn at the window centre, so
    the metrics are recomputed under that binding — both to score them against
    the density they actually came from, and to label them with it. Reading
    the target's fallback scalar instead would relabel the numbers silently.

    With `redraw=True` the saved tensors are ignored: the model is rebuilt
    from `checkpoints/final.pt` and a fresh eval batch is drawn through the
    production `_eval_at_composition` path, replacing all three `eval/`
    artefacts. This exists because rescoring cannot repair a corrupted DRAW:
    evals archived before de9db7c drew x0 from an inline uniform
    `torch.randint` while a matched base was Bernoulli(0.8), omitting the
    initial-state weight term log w0 = log[p_uniform(x0)/eta(x0)] (sd ~6.9
    nats at D=10, c=0.8) from every saved log-weight. The first redraw
    copies the stale `eval/` to `eval_archived_pre_redraw/` (skipped when
    an `eval_archived_*` sibling already preserves it) — the buggy numbers
    stay on disk as the record of what the old code produced. `redraw_seed`
    seeds the fresh draw and is recorded in the metrics, since a redraw is
    a NEW measurement, never a reproduction of the archived one.

    With `n_euler_override` set (mirroring the hard experiment's convention;
    0 and None both mean "no override"), the redraw runs on that Euler time
    grid instead of the run's own and its artefacts go to `eval_ne<k>/`,
    leaving the frozen `eval/` byte-untouched and unarchived. This exists for
    the eval-grid-offset measurement: F/site read at ne64 vs ne128 differs
    (quadrature error plus finite-ESS self-normalisation bias move together
    with the grid), and separating the eval-grid component from the training
    grid needs the SAME checkpoint redrawn on both grids side by side — while
    the archived `eval/` stays the untouched record the printed numbers came
    from. The archive step is only for in-place `eval/` overwrites, so it is
    skipped here. Requires `redraw=True`: the saved tensors were drawn on the
    run's own grid, so a rescore cannot move it.
    """
    if n_euler_override == 0:
        n_euler_override = None
    if n_euler_override is not None and not redraw:
        raise ValueError(
            "n_euler_override only makes sense with redraw=True: the saved "
            "eval tensors were drawn on the run's own grid"
        )
    run_dir = Path(run_dir)
    cfg, target, device = _rebuild_from_run_dir(run_dir)
    if n_euler_override is not None:
        cfg.ctmc = replace(cfg.ctmc, n_euler_steps=n_euler_override)
        eval_dir = run_dir / f"eval_ne{n_euler_override}"
    else:
        eval_dir = run_dir / "eval"
    if redraw:
        if (
            n_euler_override is None
            and eval_dir.exists()
            and not any(run_dir.glob("eval_archived_*"))
        ):
            shutil.copytree(eval_dir, run_dir / "eval_archived_pre_redraw")
        model = _build_model(cfg, target)
        model.load_state_dict(
            torch.load(
                run_dir / "checkpoints" / "final.pt",
                map_location=device,
                weights_only=True,
            )
        )
        torch.manual_seed(redraw_seed)
        eval_samples, eval_log_weights, eval_metrics = _eval_at_composition(
            model, target, cfg, cfg.composition_centre, device
        )
        eval_metrics["redraw_seed"] = redraw_seed
        # The grid the draw ACTUALLY ran on — with an override this differs
        # from config.json, and the metrics file must be self-describing.
        eval_metrics["n_euler_steps"] = cfg.ctmc.n_euler_steps
        eval_dir.mkdir(exist_ok=True)
        torch.save(eval_samples.cpu(), eval_dir / "samples.pt")
        torch.save(eval_log_weights.cpu(), eval_dir / "log_weights.pt")
        eval_metrics.update(_trailing_ess_metrics(run_dir))
        (eval_dir / "metrics.json").write_text(
            json.dumps(eval_metrics, indent=2)
        )
        return eval_metrics
    eval_samples = torch.load(
        run_dir / "eval" / "samples.pt", weights_only=True
    ).to(device)
    eval_log_weights = torch.load(
        run_dir / "eval" / "log_weights.pt", weights_only=True
    ).to(device)

    _, binding = _composition_binding(target, cfg.composition_centre, device)
    with binding:
        eval_metrics = _compute_eval_metrics(
            eval_samples,
            eval_log_weights,
            target,
            composition=cfg.composition_centre,
        )
    eval_metrics.update(_trailing_ess_metrics(run_dir))
    (run_dir / "eval" / "metrics.json").write_text(
        json.dumps(eval_metrics, indent=2)
    )
    return eval_metrics


def composition_sweep(
    run_dir: str | Path,
    compositions: tuple[float, ...] = SWEEP_COMPOSITIONS,
    *,
    checkpoint: str = "final.pt",
    seed: int = 0,
    save: bool = True,
) -> list[dict]:
    """Re-draw a trained amortised run's eval at each composition in turn.

    This is the measurement the amortisation claim rests on: ONE model, many
    compositions, each row directly comparable to the specialist trained for
    that composition alone — the archived
    `results/02_constrained_soft/*/eval/metrics.json` carry the same
    `ess_fraction` and `composition_mean` keys. Rows at the held-out
    compositions are the interpolation test: no specialist was ever trained
    there, and the grid control never drew them.

    Rows, not one flat dict, because every quantity here is a function of c.
    A single `target_composition` scalar in a metrics dict cannot express
    "worked at 0.50, drifted at 0.80", which is exactly the failure the sweep
    is looking for.

    Each row is drawn under COMMON RANDOM NUMBERS: the sampler is reseeded to
    `seed` before every composition, so all rows start from the same base
    states and consume the same noise stream. Differences down the sweep are
    then the model's response to c rather than which draw a row happened to
    get — the same reason paired comparisons beat independent ones, and it
    costs nothing here because the draws are independent anyway.

    Args:
        run_dir: a finished run directory (config.json + checkpoints/).
        compositions: the grid to evaluate; defaults to the six specialist
            values plus the four held-out points.
        checkpoint: file under `checkpoints/`; `final.pt` is the end-of-run
            state, `latest.pt` the most recent eval-cadence snapshot.
        seed: common-random-numbers seed, shared by every row.
        save: write `eval/composition_sweep.json` (skip for exploratory runs
            that should not overwrite a recorded sweep).

    Returns:
        One dict per composition — the full eval metrics plus `composition`
        and `held_out` — in the order given.
    """
    run_dir = Path(run_dir)
    cfg, target, device = _rebuild_from_run_dir(run_dir)
    if not cfg.condition_on_composition:
        raise ValueError(
            f"{run_dir} was trained without composition conditioning, so its "
            "model has no c input: a sweep would redraw the same specialist "
            "distribution once per composition and label the copies with "
            "compositions they do not obey."
        )
    model = _build_model(cfg, target)
    model.load_state_dict(
        torch.load(
            run_dir / "checkpoints" / checkpoint,
            map_location=device,
            weights_only=True,
        )
    )

    rows = []
    for composition in compositions:
        seed_everything(seed)
        _, _, metrics = _eval_at_composition(
            model, target, cfg, float(composition), device
        )
        rows.append(
            {
                "composition": float(composition),
                "held_out": float(composition) in HELD_OUT_COMPOSITIONS,
                **metrics,
            }
        )

    if save:
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(exist_ok=True)
        (eval_dir / "composition_sweep.json").write_text(
            json.dumps(rows, indent=2)
        )
    return rows


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
        "--tag",
        default=None,
        help="Run-dir suffix (default: wall-clock timestamp). A fixed tag "
             "makes resubmission after preemption reuse the run dir and "
             "skip a completed run; it does NOT resume mid-run",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; recompute eval/metrics.json from saved samples",
    )
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="With --eval-only: ignore the saved samples and draw a fresh "
             "eval batch from checkpoints/final.pt (archives the stale "
             "eval/ first; for evals whose DRAW was wrong, e.g. pre-de9db7c "
             "matched-base runs)",
    )
    parser.add_argument(
        "--redraw-seed",
        type=int,
        default=0,
        help="Seed for the fresh --redraw draw (recorded in metrics.json)",
    )
    parser.add_argument(
        "--n-euler-override",
        type=int,
        default=None,
        help="With --redraw: draw on this Euler grid instead of the run's "
             "own; artefacts go to eval_ne<k>/ and the frozen eval/ is left "
             "untouched (grid-offset measurement)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Per-composition eval sweep of a trained amortised run "
             "(requires --run-dir)",
    )
    parser.add_argument(
        "--compositions",
        type=float,
        nargs="+",
        help="Override the sweep grid (default: the six specialist "
             "compositions plus the four held-out points)",
    )
    parser.add_argument(
        "--checkpoint",
        default="final.pt",
        help="Checkpoint under checkpoints/ to sweep (default final.pt)",
    )
    parser.add_argument(
        "--run-dir",
        help="Run directory to re-evaluate (required with --eval-only/--sweep)",
    )
    args = parser.parse_args()

    if args.sweep:
        if not args.run_dir:
            parser.error("--sweep requires --run-dir")
        rows = composition_sweep(
            args.run_dir,
            compositions=(
                tuple(args.compositions) if args.compositions
                else SWEEP_COMPOSITIONS
            ),
            checkpoint=args.checkpoint,
        )
        print(json.dumps(rows, indent=2))
        return

    if args.eval_only:
        if not args.run_dir:
            parser.error("--eval-only requires --run-dir")
        metrics = eval_only(
            args.run_dir, redraw=args.redraw, redraw_seed=args.redraw_seed,
            n_euler_override=args.n_euler_override,
        )
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
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
