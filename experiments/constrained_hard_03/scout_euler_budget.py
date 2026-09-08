"""Euler-budget scouting: does one-event n_euler_steps ∝ O(d) hold the
swap-CTMC clip fraction under threshold, or is the matching multi-event step
needed as d grows?

Aggregates the two frozen diagnostics for the total swap escape rate
Λ(x, t) = Σ_{i<j} [G_swap(i,j | x)]_+ and dt = 1/(n_euler_steps−1):
    - expected events per site per step = mean(Λ·dt)/d   (threshold ≤ 0.1)
    - clipped-step fraction              = P(Λ·dt > 1)    (threshold < 1%)
The one-event step fires ≤1 swap/step, so Λ·dt > 1 is where it under-fires.

Λ splits into a local part over the 2d adjacent pairs (swap ΔE touches only
bonds at i, j) and a part over the ~d²/2 non-adjacent pairs: if the
non-adjacent per-pair rate is non-negligible Λ grows like d² and the one-event
budget blows up, otherwise it grows like d and O(d) steps suffice at every
scale. Both per-pair rates are set by the intensive local energetics, so they
are ~d-invariant, which lets the converged D=4 σ_c checkpoint predict d=64 and
d=256 without retraining.

Modes:
  --checkpoint PATH   measure a trained head (fast; the D=4 anchor path)
  --n-steps N         train D=D at σ_c then measure (GPU; the empirical D≥8 path)

Usage:
  pixi run -e dev python -m experiments.constrained_hard_03.scout_euler_budget \\
      --checkpoint <run_dir>/checkpoints/final.pt --D 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from experiments.dnfs_baseline_01.configs import CTMCCfg, EvalCfg, TrainCfg

from discrete_flow_sampler.constraints.swap_readout import (
    DoublyHollowSwapHead,
    LeTFMaskOneSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.seeding import seed_everything
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

SIGMA_C = (
    0.22305  # legacy value: this scout profiled the archived (pre-migration) cells
)
CANDIDATE_STEPS = (16, 32, 64, 128, 256, 512)
EXTRAPOLATE_TO = (4, 8, 12, 16)


def _build_head(kind, d, device):
    backbone = LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=32, n_layers=2, n_heads=4
    ).to(device)
    return (DoublyHollowSwapHead if kind == "doubly_hollow" else LeTFMaskOneSwapHead)(
        backbone
    )


@torch.no_grad()
def measure(head, target, states, t_values, chunk_size=256):
    """Λ decomposition + per-candidate clip diagnostics on `states` at `t_values`.

    The head forward is chunked: mask_one stacks d anchor copies per state, so
    an unchunked pass over the flattened trajectory OOMs at d=64 (A100 40 GB).
    """
    pairs = upper_tri_pairs(target.d, states.device)  # (P, 2)
    adjacent = target.A[pairs[:, 0], pairs[:, 1]] > 0  # (P,)
    rates = F.relu(
        torch.cat(
            [
                gather_pair_scores(
                    head(states[i : i + chunk_size], t_values[i : i + chunk_size]),
                    pairs,
                )
                for i in range(0, states.shape[0], chunk_size)
            ]
        )
    )  # (S, P)

    lam = rates.sum(dim=-1)  # (S,)
    lam_adj = rates[:, adjacent].sum(dim=-1)
    lam_nonadj = rates[:, ~adjacent].sum(dim=-1)
    candidates = []
    for n in CANDIDATE_STEPS:
        lam_dt = lam / max(n - 1, 1)
        candidates.append(
            {
                "n_euler_steps": n,
                "lambda_dt_mean": lam_dt.mean().item(),
                "lambda_dt_p99": torch.quantile(lam_dt, 0.99).item(),
                "events_per_site_per_step": (lam_dt.mean() / target.d).item(),
                "clipped_step_frac": (lam_dt > 1.0).float().mean().item(),
            }
        )
    return {
        "n_pairs_adjacent": int(adjacent.sum()),
        "lambda_mean": lam.mean().item(),
        "lambda_p99": torch.quantile(lam, 0.99).item(),
        "lambda_adj_mean": lam_adj.mean().item(),
        "lambda_nonadj_mean": lam_nonadj.mean().item(),
        "rate_per_adjacent_pair": rates[:, adjacent].mean().item(),
        "rate_per_nonadjacent_pair": rates[:, ~adjacent].mean().item(),
        "candidates": candidates,
    }


def extrapolate(a, b, lam_p99_ref, d_ref):
    """Λ(D) = 2d·a + (d(d−1)/2 − 2d)·b, per-pair rates a (adj) / b (non-adj)
    assumed d-invariant. Reports Λ(D), its split, and the clip-safe one-event
    step count n_min ≈ 1 + p99(Λ) where p99(Λ) is scaled from the reference."""
    rows = []
    lam_ref = 2 * d_ref * a + (d_ref * (d_ref - 1) // 2 - 2 * d_ref) * b
    for D in EXTRAPOLATE_TO:
        d = D * D
        n_adj, n_pairs = 2 * d, d * (d - 1) // 2
        lam_adj, lam_nonadj = n_adj * a, (n_pairs - n_adj) * b
        lam = lam_adj + lam_nonadj
        lam_p99 = lam_p99_ref * lam / lam_ref
        rows.append(
            {
                "D": D,
                "d": d,
                "n_pairs": n_pairs,
                "lambda_pred": lam,
                "nonadj_fraction": lam_nonadj / lam,
                "n_min_clip_safe": int(lam_p99) + 2,  # n where p99(Λ·dt) < 1
            }
        )
    return rows


def _print_report(tag, m, extrap):
    print(f"\n=== {tag} ===")
    print(
        f"Λ mean={m['lambda_mean']:.3f}  p99={m['lambda_p99']:.3f}  "
        f"(adjacent {m['lambda_adj_mean']:.3f} + non-adjacent "
        f"{m['lambda_nonadj_mean']:.3f}; {m['n_pairs_adjacent']} adjacent pairs)"
    )
    print(
        f"per-pair rate: adjacent a={m['rate_per_adjacent_pair']:.4f}  "
        f"non-adjacent b={m['rate_per_nonadjacent_pair']:.5f}"
    )
    print(
        f"{'n_euler':>8} {'mean Λ·dt':>10} {'p99 Λ·dt':>10} "
        f"{'evt/site/step':>14} {'clip frac':>10}"
    )
    for r in m["candidates"]:
        flag = (" EVT>0.1" if r["events_per_site_per_step"] > 0.1 else "") + (
            " CLIP>1%" if r["clipped_step_frac"] > 0.01 else ""
        )
        print(
            f"{r['n_euler_steps']:>8} {r['lambda_dt_mean']:>10.4f} "
            f"{r['lambda_dt_p99']:>10.4f} {r['events_per_site_per_step']:>14.4f} "
            f"{r['clipped_step_frac']:>10.4f}{flag}"
        )
    if extrap:
        print("\nExtrapolation (per-pair rates held d-invariant):")
        print(
            f"{'D':>3} {'d':>5} {'Λ pred':>10} {'non-adj frac':>13} {'clip-safe n':>12}"
        )
        for r in extrap:
            print(
                f"{r['D']:>3} {r['d']:>5} {r['lambda_pred']:>10.2f} "
                f"{r['nonadj_fraction']:>13.2f} {r['n_min_clip_safe']:>12}"
            )


def _states_for(head, target, device, n_grid=48, batch=128):
    """On-distribution states across t: one swap-CTMC trajectory, flattened.

    Uses a modest grid/batch — this is a rate-scale estimate, not an eval.
    """
    ts = torch.linspace(0.0, 1.0, n_grid, device=device)
    x0 = target.sample_base(batch, device=device)
    traj = sample_swap_ctmc(head, x0, ts, return_all_states=True)
    return traj.reshape(-1, target.d), ts.repeat_interleave(traj.shape[1])


def from_checkpoint(ckpt_path, D, head_kind, out_dir, seed=42):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = FixedCompositionIsingTarget(
        D=D, sigma=SIGMA_C, target_composition=0.5, device=device
    )
    head = _build_head(head_kind, target.d, device)
    head.load_state_dict(torch.load(ckpt_path, map_location=device))
    states, t_values = _states_for(head, target, device)
    m = measure(head, target, states, t_values)
    extrap = extrapolate(
        m["rate_per_adjacent_pair"],
        m["rate_per_nonadjacent_pair"],
        m["lambda_p99"],
        target.d,
    )
    _print_report(f"D={D} converged checkpoint (σ_c anchor)", m, extrap)
    _save(out_dir, target.d, {"source": str(ckpt_path), **m, "extrapolation": extrap})


def from_training(D, n_steps, out_dir, seed=42):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = FixedCompositionIsingTarget(
        D=D, sigma=SIGMA_C, target_composition=0.5, device=device
    )
    head = _build_head("mask_one", target.d, device)
    train_cfg = TrainCfg(
        n_steps=n_steps,
        batch_size=128,
        outer_batch_size=128,
        replay_buffer_cycles=8,
        lr=1e-3,
        seed=seed,
        warmup_steps=500,
    )
    print(
        f"[scout] training D={D} (d={target.d}) for {n_steps} steps on {device}...",
        flush=True,
    )
    t0 = time.time()
    train_swap(
        head,
        target,
        train_cfg,
        CTMCCfg(n_euler_steps=64),
        EvalCfg(eval_every=max(n_steps // 5, 1), n_eval_samples=256),
        Path(out_dir) / "train",
        use_wandb=False,
        estimator_mode="control_variate",
    )
    print(f"[scout] trained in {time.time() - t0:.0f}s", flush=True)
    states, t_values = _states_for(head, target, device)
    m = measure(head, target, states, t_values)
    _print_report(f"D={D} trained {n_steps} steps (empirical)", m, None)
    _save(out_dir, target.d, {"n_steps": n_steps, **m})


def _save(out_dir, d, payload):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"scout_d{d}.json").write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="measure a trained head")
    parser.add_argument("--D", type=int, default=4, help="lattice side (d = D*D)")
    parser.add_argument(
        "--head-kind",
        default="mask_one",
        choices=("doubly_hollow", "mask_one"),
        help="mask_one (O(d)) is bit-exact ≡ doubly_hollow and far faster to measure",
    )
    parser.add_argument(
        "--n-steps", type=int, default=2000, help="train mode: steps (multiple of 100)"
    )
    parser.add_argument("--out", default="results/03_hard/scout")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.checkpoint:
        from_checkpoint(args.checkpoint, args.D, args.head_kind, args.out, args.seed)
    else:
        from_training(args.D, args.n_steps, args.out, args.seed)


if __name__ == "__main__":
    main()
