"""Does the IS weight variance depend on the time discretisation?

Every archived cell integrates the trajectory on a uniform t-grid whose
resolution was chosen per size and then never scaled: 100 steps at 4x4, 128
at 8x8, and still 128 at 16x16 -- 4x the sites at the same resolution.
Beskos, Crisan & Jasra (Ann. Appl. Probab. 24(4), 2014) prove that a single
importance-sampling step needs sample size growing exponentially in
dimension, while a sequence of intermediate targets keeps the effective
sample size non-degenerate at fixed N -- but only if the number of
intermediate targets grows with the dimension. Holding the grid fixed while
quadrupling the sites is exactly the regime that theorem says degenerates.

The earlier grid analysis reanalysed already-recorded 128-step rollouts
against subsampled 64/32 and warped-64 grids, so its negative result is
about allocation at a fixed budget; resolution -- more points than the run
used -- was never testable from those artefacts.

Fresh rollouts off a frozen checkpoint at several n_euler values, reporting
Var[log w] rather than ESS. The lognormal identity ESS/N = exp(-Var[log w])
reproduces every healthy archived run to within 2%, but at 16x16 the run
carries Var[log w] = 18.1, for which exp(-18.1) = 1.4e-8 -- far below the
1/N floor of a 5000-draw self-normalised estimator, so the reported 0.0031
there is the estimator hitting its floor. Var[log w] still estimates
cleanly (log w is near-Gaussian, skew 0.15).

n_euler drives both the trajectory law (Euler steps) and the weight
quadrature, so a drop in Var[log w] at finer resolution cannot be
attributed to quadrature alone. That is deliberate: the decision this
informs is whether to spend wall-clock on resolution, and either mechanism
justifies the spend equally.

Usage:
    python -m scripts.n_euler_resolution_sweep --run-dir <dir> \
        --n-euler 128,256,512 --n-draws 512
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.run import build_target_and_head

from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.seeding import seed_everything


def draw_log_weights(head, target, cfg, n_draws, n_euler, chunk, device):
    """IS log-weights for `n_draws` trajectories at a given resolution.

    Mirrors the production eval's draw protocol (chunked, fp32, canonical
    step) so the numbers are comparable with archived eval artefacts; only
    the grid resolution is varied.
    """
    multi_event = cfg.ctmc.use_matching_step
    collected = []
    drawn = 0
    while drawn < n_draws:
        batch = min(chunk, n_draws - drawn)
        t_grid = torch.linspace(0.0, 1.0, n_euler + 1, device=device)
        x_initial = target.sample_base(batch, device=device)
        with torch.no_grad():
            _, log_weights = sample_swap_ctmc(
                head,
                x_initial,
                t_grid,
                return_log_weights=True,
                target=target,
                multi_event=multi_event,
            )
        collected.append(log_weights.detach().cpu())
        drawn += batch
    return torch.cat(collected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--n-euler",
        default="128,256,512",
        help="comma-separated resolutions to compare",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=512,
        help="Var[log w] is a variance of a near-Gaussian, "
        "so it estimates well from far fewer draws "
        "than ESS needs",
    )
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = CONFIGS[saved["name"]]
    cfg = replace(cfg, head_kind=saved["head_kind"])
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    seed_everything(args.seed)
    target, head = build_target_and_head(cfg, device)
    head.load_state_dict(
        torch.load(
            run_dir / "checkpoints" / "final.pt",
            map_location=device,
            weights_only=True,
        )
    )
    head.eval()

    d = int(cfg.ising.D) ** 2
    print(
        f"run={run_dir.name} d={d} device={device} "
        f"as-run n_euler={cfg.ctmc.n_euler_steps} "
        f"multi_event={cfg.ctmc.use_matching_step}",
        flush=True,
    )
    print(
        f"{'n_euler':>8} {'Var[log w]':>11} {'Var/site':>10} "
        f"{'exp(-Var)':>11} {'SN-ESS frac':>12} {'n':>6}",
        flush=True,
    )

    rows = []
    for n_euler in (int(v) for v in args.n_euler.split(",")):
        seed_everything(args.seed)  # same base draws across arms
        log_weights = draw_log_weights(
            head, target, cfg, args.n_draws, n_euler, args.chunk, device
        )
        var = float(log_weights.var().item())
        ess_frac = float(ess_from_log_weights(log_weights).item() / len(log_weights))
        row = {
            "n_euler": n_euler,
            "var_log_w": var,
            "var_per_site": var / d,
            "exp_neg_var": float(torch.tensor(-var).exp().item()),
            "sn_ess_fraction": ess_frac,
            "n_draws": len(log_weights),
        }
        rows.append(row)
        print(
            f"{n_euler:>8} {var:>11.4f} {var / d:>10.5f} "
            f"{row['exp_neg_var']:>11.3e} {ess_frac:>12.5f} "
            f"{len(log_weights):>6}",
            flush=True,
        )

    if len(rows) > 1:
        base = rows[0]["var_log_w"]
        print("\nVar[log w] relative to the coarsest arm:")
        for row in rows:
            print(f"  n_euler {row['n_euler']:>5}: {row['var_log_w'] / base:.3f}x")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"run": run_dir.name, "d": d, "rows": rows}, indent=2)
        )


if __name__ == "__main__":
    main()
