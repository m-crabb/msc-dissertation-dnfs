"""Offline probe: which term dominates the Kolmogorov residual, and does the
neighbour log-ratio ceiling bind?

Runs against trained checkpoints on real buffered states. No training, no GPU
required, minutes on a laptop. Its job is to decide whether a training sweep
is worth running at all, and to calibrate the ceiling if it is.

WHAT IT MEASURES

The residual (Eq. 10) decomposes as

    residual = dt_log_p̃_t(x)  -  dt_log_Z_t  +  site_terms(x)

and the training log carries only the *variance* of the first and of the
whole integrand. Variance is the right statistic for asking which term drives
the loss — the residual subtracts a per-slot mean, so a large but constant
term cancels — but it cannot answer two things:

  1. MAGNITUDE. A term can be huge and nearly constant, contributing little
     variance but dominating the residual's level if `dt_log_Z_t` fails to
     track it. `dt_log_Z_t` is estimated once per outer cycle and reused
     across a replay buffer spanning several cycles, so tracking is not
     guaranteed. This probe reports the three terms separately.

  2. CLAMP LOAD-BEARING-NESS. Whether the ceiling on log p̃_t(y)/p̃_t(x) is
     inert or is materially changing the loss, and by how much.

EXPECTED OUTCOMES

  A. At the collapsed 10x10 checkpoint the ceiling is load-bearing: raising
     it changes the residual by orders of magnitude, and `clamp_frac` at the
     paper's 5 is well above zero.
  B. At a healthy 4x4 checkpoint the ceiling is inert: raising it moves the
     residual by a negligible amount and `clamp_frac` is ~0.
  C. `site_terms` dominates the residual at the collapsed checkpoint, and
     `dt_log_p̃ - dt_log_Z_t` does not.

  If B fails — if the ceiling is equally load-bearing where training is
  healthy — then clamp binding does not separate healthy from dead runs and
  the mechanism is refuted, whatever A shows.

  If C fails and the `dt_log_p̃ - dt_log_Z_t` residue dominates by magnitude,
  the operative fix is the `dt_log_p̃` bound rather than the ceiling.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from discrete_flow_sampler.models.composition_conditioned import (
    CompositionConditioned,
)
from discrete_flow_sampler.samplers._neighbours import _log_p_tilde_at_neighbours
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.log_z_estimators import compute_c_t_grid


def _decompose(x, t, dt_log_Zt, model, target):
    """Per-state residual split into its three additive terms (Eq. 10)."""
    import torch.nn.functional as F

    G_t = model(x, t)
    G_plus, neg_G_plus = F.relu(G_t), F.relu(-G_t)

    log_p_neighbours = _log_p_tilde_at_neighbours(x, t, target, model.vocab_size)
    raw_ratio = log_p_neighbours - target.log_p_tilde_t(x, t)[:, None, None]
    ceiling = target.log_ratio_clamp
    site_terms = (G_plus - neg_G_plus * raw_ratio.clamp(max=ceiling).exp()).sum(
        dim=(-2, -1)
    )

    dt_log_p_tilde = target.dt_log_p_tilde_t(x, t)
    return {
        "dt_log_p_tilde": dt_log_p_tilde,
        "dt_log_Zt": dt_log_Zt.expand_as(dt_log_p_tilde),
        "target_residue": dt_log_p_tilde - dt_log_Zt,
        "site_terms": site_terms,
        "residual": dt_log_p_tilde - dt_log_Zt + site_terms,
        "clamp_frac": (raw_ratio > ceiling).float().mean(),
        "raw_ratio_p99": torch.quantile(raw_ratio.reshape(-1).float(), 0.99),
    }


def probe(run_dir: Path, ceilings, n_states: int, composition: float | None):
    from experiments.constrained_soft_02.configs import CONFIGS
    from experiments.dnfs_baseline_01.run import _build_model

    from discrete_flow_sampler.targets.ising import IsingTarget

    cfg = CONFIGS[json.loads((run_dir / "config.json").read_text())["name"]]
    # Terminal λ, not the curriculum's opening value: the collapse happened
    # under the final penalty strength and that is the regime being probed.
    lam = cfg.ising.composition_penalty_strength
    target = IsingTarget(
        D=cfg.ising.D,
        sigma=cfg.ising.sigma,
        bias=cfg.ising.bias,
        target_composition=cfg.ising.target_composition,
        composition_penalty_strength=lam,
        base_composition=cfg.ising.base_composition,
    )
    model = _build_model(cfg, target)
    state = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    # Both the model and the target must be bound to the requested
    # composition: the target so its penalty is scored against c, the model
    # because a `condition_on_composition` readout refuses to run without it.
    # Same pairing the training loop uses, so the probe measures the real path.
    if composition is None:
        model_bound, bind = model, nullcontext()
    else:
        c = torch.full((1,), composition)
        model_bound = CompositionConditioned(model, c)
        bind = target.composition_batch(c)

    torch.manual_seed(0)
    t_grid = torch.linspace(0.0, 1.0, cfg.ctmc.n_euler_steps + 1)
    with bind, torch.no_grad():
        x0 = target.sample_base(n_states, device=target.device)
        traj = sample_ctmc(model_bound, x0, t_grid, return_all_states=True)
        c_t_grid, _ = compute_c_t_grid(
            t_grid, traj, target, model_bound, mode=cfg.estimator
        )
        # Score at t=1, where the penalty enters log p̃_t at full strength and
        # the ceiling is therefore most likely to bind.
        x, t, dt_log_Zt = traj[-1], t_grid[-1].expand(n_states), c_t_grid[-1]

        rows = []
        for ceiling in ceilings:
            target.log_ratio_clamp = ceiling
            d = _decompose(x, t, dt_log_Zt, model_bound, target)
            rows.append(
                {
                    "ceiling": ceiling,
                    "clamp_frac": d["clamp_frac"].item(),
                    "raw_ratio_p99": d["raw_ratio_p99"].item(),
                    "dt_log_p_tilde_mean": d["dt_log_p_tilde"].mean().item(),
                    "dt_log_Zt": d["dt_log_Zt"][0].item(),
                    "target_residue_absmean": d["target_residue"].abs().mean().item(),
                    "site_terms_absmean": d["site_terms"].abs().mean().item(),
                    "residual_absmean": d["residual"].abs().mean().item(),
                    "loss": d["residual"].pow(2).mean().item(),
                }
            )
        achieved = target.composition_fraction(x).mean().item()
    return achieved, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--composition", type=float, default=None)
    ap.add_argument("--n-states", type=int, default=256)
    ap.add_argument(
        "--ceilings", type=float, nargs="+", default=[5.0, 10.0, 20.0, 50.0]
    )
    args = ap.parse_args()

    achieved, rows = probe(args.run_dir, args.ceilings, args.n_states, args.composition)
    lam_hint = (
        f"requested c={args.composition}" if args.composition else "unconditioned"
    )
    print(f"\n{args.run_dir.name}\n  {lam_hint}, achieved <c> = {achieved:.4f}")
    # dt_log_p̃ and dt_log_Z_t are printed separately, not just their
    # difference: dt_log_Z_t is the mean of (dt_log_p̃ + site_terms), so a
    # large `target_residue` can come either from a large dt_log_p̃ — which
    # would implicate the penalty's linear entry — or from a large
    # dt_log_Z_t dragged up by site_terms. Only the split distinguishes them.
    hdr = (
        "ceiling",
        "clampfrac",
        "ratio_p99",
        "dt_log_p~",
        "dt_log_Zt",
        "|dtlogp-dtlogZ|",
        "|site_terms|",
        "|residual|",
        "loss",
    )
    print("  " + "".join(f"{h:>16}" for h in hdr))
    for r in rows:
        print(
            "  "
            + "".join(
                f"{v:>16.4g}"
                for v in (
                    r["ceiling"],
                    r["clamp_frac"],
                    r["raw_ratio_p99"],
                    r["dt_log_p_tilde_mean"],
                    r["dt_log_Zt"],
                    r["target_residue_absmean"],
                    r["site_terms_absmean"],
                    r["residual_absmean"],
                    r["loss"],
                )
            )
        )


if __name__ == "__main__":
    main()
