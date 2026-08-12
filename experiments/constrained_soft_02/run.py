"""Entry point for the constrained-soft Ising experiments.

Delegates the actual training to `experiments.dnfs_baseline_01.run.train`,
which is parameterised by the cfg object and already handles
composition-penalty kwargs natively when `IsingCfg` sets them.

Usage:
    pixi run -e dev python -m experiments.constrained_soft_02.run \\
        --cfg S2_d4_c05_l50_letf --seed 42

An amortised cell's end-of-run eval is a single draw at the window centre. Its
per-composition numbers come from a separate sweep over the finished run dir,
which works for any run dir because it rebuilds from config.json:
    pixi run -e dev python -m experiments.dnfs_baseline_01.run \\
        --sweep --run-dir results/02_constrained_soft/<run_dir>
"""
import argparse

from experiments.dnfs_baseline_01.run import train

from .configs import CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", required=True, choices=list(CONFIGS.keys()),
        help="Config key from constrained_soft_02/configs.py CONFIGS",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/02_constrained_soft")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--tag",
        default=None,
        help="Run-dir suffix (default: wall-clock timestamp). A fixed tag "
             "makes resubmission after preemption reuse the run dir and "
             "skip a completed run; it does NOT resume mid-run",
    )
    args = parser.parse_args()

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
