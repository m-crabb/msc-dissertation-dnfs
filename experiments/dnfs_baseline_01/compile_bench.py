"""Same-container eager-vs-compiled bench of the flip-route trainer
(optimisation board section C, decided s60 2026-08-24).

Measures what the board asks for before Wave 1 is scheduled: the leTF
trunk is the GEMM/attention case (expect ~1.7x updates from compile, more
where runs are launch-bound), and Wave 1 is 60-70 GPU-h, so even 1.5x
returns ~25 h of cluster time.

Method — the s59 discipline in code:
  * BOTH arms run in ONE process/container, so the ratio is same-device
    by construction (the s59 "thp slower at d256" reading was a
    cross-venue artefact; the device name is printed because Modal's
    A100 class mixes PCIe/SXM4 and only same-container ratios are the
    unit).
  * Update time is the trainer's own `wall_clock_step_s` column, which
    times ONLY the inner loss update (its documented scope) — median over
    the tail so the compiled arm's first-step compilation cost is
    excluded.
  * The rollout slice is timed separately around `sample_ctmc` in
    trajectory mode (the outer step's buffer rebuild), warmup pass first.

Nothing lands on the results volume: this is a bench, and its numbers go
to the printed table (copy into the profiling review beside the s59
records).

Run on Modal (the profiling venue):
    pixi run -e dev modal run -m \
        experiments.dnfs_baseline_01.modal_app::compile_bench
"""
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from experiments.dnfs_baseline_01.configs import CONFIGS
from experiments.dnfs_baseline_01.run import _build_model, train
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.targets.ising import IsingTarget


def _short_cfg(cfg_name: str, n_steps: int):
    cfg = CONFIGS[cfg_name]
    return replace(
        cfg,
        train=replace(cfg.train, n_steps=n_steps),
        # One cheap eval at the end; the bench times updates and rollouts,
        # not the eval slice.
        eval=replace(cfg.eval, eval_every=n_steps, n_eval_samples=64),
    )


def _median_tail_step_seconds(run_root: Path, tail: int) -> float:
    log_path = next(run_root.rglob("training_log.csv"))
    step_seconds = (
        pd.read_csv(log_path)["wall_clock_step_s"].dropna().tail(tail)
    )
    return float(step_seconds.median())


def _time_rollout(model, target, n_euler_steps: int, batch_size: int,
                  device, repeats: int = 3) -> float:
    t_grid = torch.linspace(0.0, 1.0, n_euler_steps, device=device)
    timings = []
    with torch.no_grad():
        for repeat in range(repeats + 1):  # first pass = warmup/compile
            x0 = target.sample_base(batch_size, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            sample_ctmc(model, x0, t_grid, return_all_states=True,
                        target=target)
            if device == "cuda":
                torch.cuda.synchronize()
            if repeat > 0:
                timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def run_bench(cfg_name: str = "stage_4_d10", n_steps: int = 400,
              tail: int = 200) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = (
        torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    )
    print(f"[compile_bench] cfg={cfg_name} device={device_name}")

    short = _short_cfg(cfg_name, n_steps)
    arms = {
        "eager": short,
        "compiled": replace(
            short, model=replace(short.model, compile_model=True)
        ),
    }

    results: dict = {"cfg": cfg_name, "device": device_name}
    for label, arm_cfg in arms.items():
        run_root = Path(tempfile.mkdtemp(prefix=f"compile_bench_{label}_"))
        train(arm_cfg, seed=arm_cfg.train.seed, output_dir=run_root,
              use_wandb=False, tag=f"bench_{label}")
        update_s = _median_tail_step_seconds(run_root, tail)

        target = IsingTarget(
            D=arm_cfg.ising.D, sigma=arm_cfg.ising.sigma,
            bias=arm_cfg.ising.bias, device=device,
        )
        model = _build_model(arm_cfg, target)
        batch_size = (
            arm_cfg.train.outer_batch_size or arm_cfg.train.batch_size
        )
        rollout_s = _time_rollout(
            model, target, arm_cfg.ctmc.n_euler_steps, batch_size, device
        )
        results[label] = {
            "update_s_median": update_s, "rollout_s_median": rollout_s
        }
        print(f"[compile_bench] {label}: update {update_s * 1e3:.1f} ms, "
              f"rollout {rollout_s:.2f} s")

    results["update_ratio"] = (
        results["eager"]["update_s_median"]
        / results["compiled"]["update_s_median"]
    )
    results["rollout_ratio"] = (
        results["eager"]["rollout_s_median"]
        / results["compiled"]["rollout_s_median"]
    )
    print(f"[compile_bench] compile speedup: "
          f"updates {results['update_ratio']:.2f}x, "
          f"rollout {results['rollout_ratio']:.2f}x")
    return results


if __name__ == "__main__":
    run_bench()
