"""Zero-shot transfer probe: what a trained swap sampler delivers OFF its
training target, with no retraining and no architectural change.

Motivation. The cost argument in the discussion chapter leans on amortisation
across targets, and that has been promised rather than delivered. Before
spending anything on amortised TRAINING, this probe asks how much amortisation
the existing checkpoints already carry, along two axes.

-------------------------------------------------------------------------------
Axis 1: coupling (temperature), by early stopping. No retraining, no new base.
-------------------------------------------------------------------------------
On the fixed-composition slice, `base_log_eta` is the constant -log C(d, N_A)
(`ising.py`, FixedCompositionIsingTarget.base_log_eta). Substituting into the
geometric path

    log p~_t(x) = (1 - t) * base_log_eta(x) + t * log_prob(x)

the (1 - t) term is an additive constant in x, so it cancels from every ratio —
which is exactly why `swap_log_ratio` collapses to t*sigma*Delta(x^T A x) alone.
Therefore, ON THE SLICE,

    p~_t  ∝  exp( t * sigma * x^T A x )   =   the Ising target at coupling t*sigma.

The annealing path IS a coupling anneal. A model trained to endpoint sigma
traverses the entire family {sigma' : 0 <= sigma' <= sigma} on its way there, so
stopping at t* = sigma'/sigma lands on the target at sigma' exactly.

Why the weights stay valid. `sample_swap_ctmc` accumulates
`log_weights += xi_t * dt` at the left endpoint — a running Riemann sum with no
reference to t=1 or to a terminal state. So log w(t*) = int_0^{t*} xi_s ds is by
construction the importance weight for p~_{t*}. Early stopping is not an
approximation; it reads the integral before it finishes.

-------------------------------------------------------------------------------
Axis 2: composition, by amending the target's slice. No retraining either.
-------------------------------------------------------------------------------
The swap heads take (x, t) only — they never see c except through x — so a head
transfers to another slice unchanged. Amending the target changes `sample_base`
(a different number of up-sites) and the slice constant. That constant enters
xi_t as a constant, hence contributes a constant to every log w, hence CANCELS
from self-normalised weights: ESS is untouched by it and only the absolute
log Z-hat shifts. So the composition axis is measured on a clean metric.

-------------------------------------------------------------------------------
Both axes are EXACT whatever the model does.
-------------------------------------------------------------------------------
The learned rates supply only a proposal; xi_t is evaluated against whichever
target is handed in, so the importance weights re-target by construction. A
model that transfers badly produces a low ESS, never a biased answer. That is
what makes this probe safe to run before committing to any training design: the
downside is a wasted sampling run, not a wrong number.

The failure mode it guards against is the opposite one — reading a good ESS as
proof of transfer when the weights were never re-targeted at all. The tests pin
the re-targeting against exact 4x4 enumeration precisely there.

-------------------------------------------------------------------------------
Design choice: one pass per composition, not one per stopping time.
-------------------------------------------------------------------------------
`return_cv_integrand=True` hands back xi_t at every step and
`return_all_states=True` hands back the ensemble at every step, so
cumsum(xi * dt) reconstructs the weights at EVERY stopping time from a single
sampler pass. The rejected alternative — one truncated run per t* — costs
len(stop_times) times as much for the same output, agreeing to float32
summation order (pinned by `test_running_weights_match_a_truncated_run`).

The composition axis cannot be collapsed the same way: a different slice means a
different base draw, so it genuinely needs its own pass.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    ess_from_log_weights,
    nn_correlation,
)
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

# Enumerating a slice needs all 2^d states first; d=20 is ~1M rows and already
# the documented ceiling for `enumerate_states`. The probe's exact leg is a 4x4
# gate, so refuse loudly rather than swapping to a silent approximation.
MAX_ENUMERABLE_SITES = 20


def running_log_weights(head, target, x0, ts, *, multi_event: bool = False):
    """Trajectory and importance weights at EVERY grid time, from one pass.

    Returns (trajectory, running) with trajectory (T, B, d) and running (T, B),
    aligned so that `running[k]` is the correct importance log-weight for the
    ensemble `trajectory[k]` against the target p~_{ts[k]}.

    The alignment is the part worth stating, because an off-by-one here shifts
    every reported coupling and would still look plausible: `cv_integrand[k]` is
    xi_t evaluated at (trajectory[k], ts[k]) — the LEFT endpoint of step k — so
    the weight carried by trajectory[k] is the sum of increments 0..k-1, and
    running[0] is identically zero (at t=0 the ensemble is the base and the
    weight integral is empty).
    """
    with torch.no_grad():
        trajectory, cv_integrand = sample_swap_ctmc(
            head,
            x0,
            ts,
            target=target,
            return_all_states=True,
            return_cv_integrand=True,
            multi_event=multi_event,
        )
    # Slot T-1 of cv_integrand is never written (the loop runs over the T-1
    # intervals), so it is uninitialised memory and must not enter the cumsum.
    increments = cv_integrand[:-1] * (ts[1:] - ts[:-1])[:, None]
    running = torch.cumsum(increments, dim=0)
    zero = torch.zeros_like(running[:1])
    return trajectory, torch.cat([zero, running], dim=0)


def weighted_diagnostics(log_w, states, target) -> dict:
    """Estimator health and observables for one (composition, stopping time).

    `var_log_w` is the size-comparable quantity, not ESS: ESS/N ~ exp(-Var[log w])
    holds to 1-2% on this project's healthy runs, so Var is the extensive one and
    Var-per-site the intensive one. Comparing bare ESS across lattice sizes or
    across slices of different dimension compares different-sized problems.

    `ess_fraction_predicted` is reported alongside the measured fraction as a
    self-check: a large gap means the weights are dominated by a handful of
    draws and the measured ESS has hit its own finite-N floor (a 5000-draw
    self-normalised estimator cannot resolve below 1/N), which is the artefact
    that made an earlier d256 headline unreadable.
    """
    n_draws, d = states.shape[0], states.shape[1]
    ess = float(ess_from_log_weights(log_w))
    weights = torch.softmax(log_w, dim=0)
    quadratic = (states.float() @ target.A * states.float()).sum(-1)
    var_log_w = float(log_w.var(unbiased=True))
    return {
        "ess": ess,
        "ess_fraction": ess / n_draws,
        "ess_fraction_predicted": math.exp(-var_log_w),
        "var_log_w": var_log_w,
        "var_log_w_per_site": var_log_w / d,
        "max_weight": float(weights.max()),
        "mean_quadratic": float((weights * quadratic).sum()),
        "mean_quadratic_unweighted": float(quadratic.mean()),
        "nn_correlation": float(
            (weights * nn_correlation(states.float(), target.A)).sum()
        ),
        "nn_correlation_unweighted": float(
            nn_correlation(states.float(), target.A).mean()
        ),
    }


def slice_free_energy_per_site(
    mean_log_w: float, stop_time: float, sigma: float, d: int, log_slice_size: float
) -> dict:
    """Free energy of the fixed-composition slice at coupling t*sigma, per site.

    The running weight at grid time t is the path estimator of log(Z_t / Z_0)
    for the annealing density p~_t(x) = exp[(1 - t) log eta(x) + t log p(x)].
    On the slice log eta is the constant -log C(d, n_plus), so

        Z_t = exp[-(1 - t) log C] * sum_x exp[t sigma x^T A x]
            = exp[-(1 - t) log C] * Z_slice(t sigma),       Z_0 = 1,

    and Jensen gives E_q[log w_t] <= log Z_t (paper Eq. 37: a variational LOWER
    bound on log Z, hence an UPPER bound on F). Adding the slice constant back,

        log Z_slice(t sigma) >= E_q[log w_t] + (1 - t) log C(d, n_plus),

    which at t = 1 is the bare mean log-weight, the convention of
    `free_energy_lb_estimate` and of `slice_ti.py`'s reference, and at t < 1
    prices the intermediate coupling for free from the same pass. Two units
    are returned: nats per site, -log Z_slice / d, which is what mchammer's
    thermodynamic integration reports and needs no temperature; and the
    project's reduced F/d = -log Z_slice / (2 t sigma d) (beta = 2 sigma),
    undefined at t = 0 and returned as nan there rather than raising, so a
    stop-time grid that starts at zero still produces a row.

    This is Sadigh et al. (2012) Eq. A.6, exp[-beta F_C(c)] = sum_{x on the
    slice} exp[-beta E(x)], read off the sampler's own weights: the fixed-
    composition free energy that the variable-composition ensembles reconstruct
    by integrating a chemical potential, obtained here with no path.
    """
    log_z_slice = mean_log_w + (1.0 - stop_time) * log_slice_size
    coupling = stop_time * sigma
    return {
        "mean_log_w": mean_log_w,
        "log_z_slice_estimate": log_z_slice,
        "free_energy_nats_per_site": -log_z_slice / d,
        "free_energy_per_site": (
            -log_z_slice / (2.0 * coupling * d) if coupling > 0 else math.nan
        ),
    }


def slice_states(D: int, composition: float) -> torch.Tensor:
    """Every configuration on the fixed-composition manifold, (C(d, N_A), d)."""
    d = D * D
    if d > MAX_ENUMERABLE_SITES:
        raise ValueError(
            f"slice enumeration needs 2^{d} states; the exact leg is a "
            f"{int(math.isqrt(MAX_ENUMERABLE_SITES))}x lattice gate, not a "
            "production path"
        )
    n_plus = round(composition * d)
    states = enumerate_states(d)
    return states[((states + 1) // 2).sum(-1) == n_plus]


def exact_slice_statistics(
    D: int, sigma: float, composition: float, stop_time: float
) -> dict:
    """Exact conditional statistics at coupling `stop_time * sigma`.

    On the slice the conditional is pi(x | C) ∝ exp(t*sigma*x^T A x): the slice
    constant is uniform and the bias term is swap-invariant, so nothing else
    survives. This is the ground truth the sampler is scored against at 4x4 —
    an enumeration, not another sampler.
    """
    states = slice_states(D, composition).double()
    adjacency = FixedCompositionIsingTarget(
        D=D, sigma=sigma, target_composition=composition
    ).A.double()
    quadratic = (states @ adjacency * states).sum(-1)
    weights = torch.softmax(stop_time * sigma * quadratic, dim=0)
    # nn_correlation inlined rather than called: the helper casts to float32,
    # which would discard exactly the precision this float64 reference exists
    # to provide. It is x^T A x over the ordered-pair count, and `quadratic` is
    # already that numerator.
    return {
        "n_states": states.shape[0],
        "coupling": stop_time * sigma,
        "mean_quadratic": float((weights * quadratic).sum()),
        "nn_correlation": float((weights * quadratic).sum() / adjacency.sum()),
    }


def transfer_grid(
    head,
    *,
    D: int,
    sigma: float,
    compositions,
    stop_times,
    n_samples: int,
    n_euler_steps: int,
    seed: int = 42,
    device: str = "cpu",
    sample_chunk: int | None = None,
    multi_event: bool = False,
) -> list[dict]:
    """One row per (composition, stopping time) on an unchanged head.

    `sample_chunk` streams the draw the way the production eval does: the
    (T, B, d) trajectory is the memory wall at d=256 (128 x 5000 x 256 floats is
    ~650 MB before the head's own pair activations), and only the requested
    stopping slices are retained per chunk. ESS is computed on the POOLED
    weights afterwards — computing it per chunk and averaging would report the
    chunk size as the ceiling.
    """
    ts = torch.linspace(0.0, 1.0, n_euler_steps, device=device)
    stop_indices = [int(torch.argmin((ts - t).abs())) for t in stop_times]
    rows = []
    for composition in compositions:
        target = FixedCompositionIsingTarget(
            D=D, sigma=sigma, target_composition=composition, device=device
        )
        # Diagnostics run on the pooled CPU copies of the retained slices, so
        # they need a CPU-resident adjacency; same slice, same constants.
        cpu_target = FixedCompositionIsingTarget(
            D=D, sigma=sigma, target_composition=composition
        )
        torch.manual_seed(seed)
        chunk = sample_chunk or n_samples
        per_stop_states: list[list[torch.Tensor]] = [[] for _ in stop_indices]
        per_stop_log_w: list[list[torch.Tensor]] = [[] for _ in stop_indices]
        drawn = 0
        while drawn < n_samples:
            size = min(chunk, n_samples - drawn)
            x0 = target.sample_base(size, device=device)
            trajectory, running = running_log_weights(
                head, target, x0, ts, multi_event=multi_event
            )
            for slot, index in enumerate(stop_indices):
                per_stop_states[slot].append(trajectory[index].cpu())
                per_stop_log_w[slot].append(running[index].cpu())
            drawn += size
        for slot, index in enumerate(stop_indices):
            states = torch.cat(per_stop_states[slot], dim=0)
            log_w = torch.cat(per_stop_log_w[slot], dim=0)
            target.assert_on_manifold(states.to(device))
            row = {
                "composition": composition,
                "n_plus": target.n_plus_target,
                "stop_time": float(ts[index]),
                "coupling": float(ts[index]) * sigma,
                "n_samples": states.shape[0],
            }
            row.update(weighted_diagnostics(log_w, states, cpu_target))
            row.update(slice_free_energy_per_site(
                float(log_w.mean()), row["stop_time"], sigma, D * D,
                target._log_slice_size,
            ))
            rows.append(row)
    return rows


def check_sampling_provenance(saved: dict, current: dict, run_dir=None) -> dict:
    """Assert the run's recorded config still describes what we are sampling.

    Scoped deliberately narrower than run.py's eval-only guard, which demands
    whole-config equality. That is right for a LAUNCH: the whole recipe is the
    provenance. It is wrong here, because a sampling-only probe cannot be
    reached by the training subtree — no optimiser runs, no loss is formed — and
    whole-config equality would make the probe unrunnable against any finished
    cell whose training-side config has moved since.

    These thp2 cells are exactly that case: `halt_on_cv_inversion_after` was
    cleared to None after they launched. It is a cold-CV screening halt, and
    these runs completed their full 100k horizon without it firing.

    Strict on everything OUTSIDE `train` — model shape, sigma, composition,
    head kind and n_euler_steps all change what is being sampled, and a silent
    mismatch there would be attributed to failed transfer rather than to the
    wrong model. Returns the training-side drift for the caller to log, so the
    relaxation is always visible in the run's output rather than implicit.
    """
    sampling_drift = {
        key: (saved.get(key), current.get(key))
        for key in set(saved) | set(current)
        if key != "train" and saved.get(key) != current.get(key)
    }
    if sampling_drift:
        raise ValueError(
            f"config.json in {run_dir} disagrees with CONFIGS[{saved.get('name')!r}] "
            f"on fields that change what is sampled: {sampling_drift}"
        )
    return {
        key: (value, current.get("train", {}).get(key))
        for key, value in saved.get("train", {}).items()
        if current.get("train", {}).get(key) != value
    }


def _load_head(run_dir: Path, checkpoint_name: str, device: str):
    """Rebuild the trained head from a run directory, reusing run.py's own
    constructor so the probe cannot drift from how the cell was trained."""
    from experiments.constrained_hard_03.configs import CONFIGS, HardStageCfg
    from experiments.constrained_hard_03.run import (
        _backfill_missing_defaults,
        build_target_and_head,
    )

    saved = json.loads((run_dir / "config.json").read_text())
    _backfill_missing_defaults(saved, HardStageCfg)
    cfg = CONFIGS[saved["name"]]
    cfg = replace(
        cfg,
        head_kind=saved["head_kind"],
        train=replace(cfg.train, seed=saved["train"]["seed"]),
    )
    training_drift = check_sampling_provenance(
        saved, json.loads(json.dumps(asdict(cfg))), run_dir
    )
    if training_drift:
        print(
            "[probe] training-side config drift (does not affect sampling): "
            f"{training_drift}"
        )
    _, head = build_target_and_head(cfg, device)
    head.load_state_dict(
        torch.load(
            run_dir / "checkpoints" / checkpoint_name,
            map_location=device,
            weights_only=True,
        )
    )
    head.eval()
    return head, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument(
        "--compositions",
        default="0.5",
        help="comma-separated; each must give an integral n_plus on the lattice",
    )
    parser.add_argument(
        "--stop-times",
        default="1.0",
        help="comma-separated fractions of the path; coupling = t * sigma",
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-euler-steps", type=int, default=None)
    parser.add_argument("--sample-chunk", type=int, default=None)
    # Tri-state, not a bare flag: `final_eval` resolves multi_event to
    # cfg.ctmc.use_matching_step, and this cell trains with it ON. A probe that
    # silently ran one-event would miss the t*=1 anchor against the published
    # ESS and the gap would read as failed transfer rather than a wrong step.
    parser.add_argument(
        "--multi-event", default=None, choices=("on", "off"),
        help="default: the cell's own canonical step (cfg.ctmc.use_matching_step)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head, cfg = _load_head(args.run_dir, args.checkpoint, device)
    multi_event = (
        cfg.ctmc.use_matching_step
        if args.multi_event is None
        else args.multi_event == "on"
    )
    with torch.no_grad():
        rows = transfer_grid(
            head,
            D=cfg.ising.D,
            sigma=cfg.ising.sigma,
            compositions=[float(c) for c in args.compositions.split(",")],
            stop_times=[float(t) for t in args.stop_times.split(",")],
            n_samples=args.n_samples,
            n_euler_steps=args.n_euler_steps or cfg.ctmc.n_euler_steps,
            seed=cfg.train.seed,
            device=device,
            sample_chunk=args.sample_chunk,
            multi_event=multi_event,
        )
    payload = {
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "trained_sigma": cfg.ising.sigma,
        "trained_composition": cfg.ising.target_composition,
        "multi_event": multi_event,
        "n_euler_steps": args.n_euler_steps or cfg.ctmc.n_euler_steps,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
