"""Eval-phase wall-clock bench, d64, both sampling paradigms on ONE GPU.

WHY THIS EXISTS (the latency counterpoint, decided s93 and anchored in the
tab:eval-hard-4x4 provenance block): FLOP/es prices arithmetic, not serial
depth or kernel utilisation. The KV-cached autoregressive GFN rollout is
~250x below the swap heads on FLOP/es at d64, but pays d SEQUENTIAL
one-token latency-bound kernels per sample, while the swap CTMC's 128 Euler
steps are each one large forward parallelised across lattice and batch.
Wall-clock per effective sample at matched hardware is where that
difference lands, and it must be MEASURED, not projected — the launch
gate's rollout timings showed the GFN rollout takes the same ~65 ms at
B=128 and B=512, i.e. it is latency-bound and batch is nearly free, which
no FLOP count would reveal.

WHAT IS TIMED, exactly. One frozen-eval chunk per arm, the way each side's
own frozen eval actually ran (phase and batch named, per the B=128
bench-ranking lesson: a verdict is scoped to its phase):
  * swap arms — one `sample_swap_ctmc` eval slice of `eval_sample_chunk`
    (256) draws with IS log-weights over the full 128-step Euler grid,
    fp32 under no_grad (run.py final_eval runs fp32; the bf16 opt-in
    covers the in-training diagnostic eval only). The head is built at its
    BILLING config (`flop_billing_config`): the separable contraction is an
    exact rewrite of the masked-attention band, so like the FLOP/es column
    this measures the cheapest exact evaluation of the architecture. This
    is a FRESH measurement on today's code — the archived training jobs'
    own wall clocks ("1.3 hours...") are historical fact and stay as
    quoted; nothing here restates them.
  * GFN arms — one `policy.sample` chunk of `eval_sample_chunk` (512)
    draws plus the IS-weight scoring, under the bf16 eval autocast the d64
    cells shipped with (run_gfn.final_eval_gfn's exact body).

The per-raw-sample seconds land in wallclock.json; house_table_8x8.py
divides by each seed's frozen ESS to get the Wall/es column that rides
beside FLOP/es. Per-raw cost is intensive and identical across couplings
(same architecture, same grid — asserted below), so each arm is timed once.

    pixi run -e dev python -m \\
        experiments.constrained_hard_03.bench_eval_wallclock_d64
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.constrained_hard_03.analysis.house_table_8x8 import (
    ARMS, GFN_ARMS, GFN_CELL_NAME, CELL_NAME, flop_billing_config)
from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS
from experiments.constrained_hard_03.gfn_launch_bench import _timed
from experiments.constrained_hard_03.run import build_target_and_head
from experiments.constrained_hard_03.run_gfn import (
    _eval_autocast, build_target_and_policy)
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc


# Nine arms compile in one process; the default per-function cache of 8
# would silently time later arms in eager mode against compiled siblings.
torch._dynamo.config.cache_size_limit = 64


def bench_swap_arm(arm, device, warmup, reps, chunk_override=None):
    """One fp32 eval chunk of the arm's swap CTMC, timed."""
    cfg_s010 = flop_billing_config(
        CONFIGS[CELL_NAME["s010"].format(arm=arm)])
    cfg_s220 = CONFIGS[CELL_NAME["s220"].format(arm=arm)]
    assert (cfg_s010.ctmc.n_euler_steps == cfg_s220.ctmc.n_euler_steps
            and cfg_s010.eval.eval_sample_chunk
            == cfg_s220.eval.eval_sample_chunk), \
        f"{arm}: per-raw eval cost differs across couplings; time both"
    target, head = build_target_and_head(cfg_s010, device)
    head.eval()
    ts = torch.linspace(0.0, 1.0, cfg_s010.ctmc.n_euler_steps + 1,
                        device=device)
    chunk = chunk_override or cfg_s010.eval.eval_sample_chunk

    def draw():
        with torch.no_grad():
            x_initial = target.sample_base(chunk, device=device)
            sample_swap_ctmc(head, x_initial, ts, return_log_weights=True,
                             target=target,
                             multi_event=cfg_s010.ctmc.use_matching_step)

    seconds = _timed(draw, warmup=warmup, reps=reps, device=device)
    return {
        "phase": "final-eval chunk (sample_swap_ctmc + IS weights)",
        "batch": chunk,
        "precision": "fp32",
        "n_euler_steps": cfg_s010.ctmc.n_euler_steps,
        "separable_band_scores": getattr(
            cfg_s010, "separable_band_scores", False),
        "compile_head": cfg_s010.compile_head,
        "seconds_per_chunk": seconds,
        "seconds_per_raw_sample": seconds / chunk,
    }


def bench_gfn_arm(gfn_arm, device, warmup, reps, chunk_override=None):
    """One bf16-autocast eval chunk of the cached AR rollout + IS scoring."""
    objective = gfn_arm.removeprefix("gfn_")
    cfg = GFN_CONFIGS[GFN_CELL_NAME.format(sigma_label="s010",
                                           objective=objective)]
    target, policy = build_target_and_policy(cfg, device)
    policy.eval()
    chunk = chunk_override or cfg.eval_sample_chunk

    def draw():
        with torch.no_grad(), _eval_autocast(cfg, device):
            spins, log_q = policy.sample(chunk)
            (target.log_prob(spins) - log_q).float()

    seconds = _timed(draw, warmup=warmup, reps=reps, device=device)
    return {
        "phase": "final-eval chunk (KV-cached rollout + IS weights)",
        "batch": chunk,
        "precision": "bf16-autocast" if cfg.eval_autocast_bf16 else "fp32",
        "seconds_per_chunk": seconds,
        "seconds_per_raw_sample": seconds / chunk,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="smoke the script off-venue; timings from a "
                             "CPU run certify wiring, never a column")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--smoke", action="store_true",
                        help="wiring check only: chunk 8, warmup 0, one rep, "
                             "no JSON written")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results" / "03_hard"
                        / "eval_wallclock_bench_d64" / "wallclock.json")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("no CUDA device: run on the venue (or --allow-cpu)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = (torch.cuda.get_device_name(0) if device == "cuda"
                   else "cpu")
    print(f"device: {device} ({device_name})")

    warmup, reps = (0, 1) if args.smoke else (args.warmup, args.reps)
    chunk_override = 8 if args.smoke else None
    bench = {"device": device_name, "warmup": warmup, "reps": reps,
             "arms": {}}
    for arm, fn in ({a: bench_swap_arm for a in ARMS}
                    | {g: bench_gfn_arm for g in GFN_ARMS}).items():
        row = fn(arm, device, warmup, reps, chunk_override)
        bench["arms"][arm] = row
        print(f"  {arm:9} B={row['batch']} {row['precision']}: "
              f"{row['seconds_per_chunk']*1e3:.1f} ms/chunk = "
              f"{row['seconds_per_raw_sample']*1e6:.1f} us/sample")

    if args.smoke:
        print("smoke only: no JSON written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bench, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
