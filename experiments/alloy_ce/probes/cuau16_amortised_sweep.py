"""Roll the composition-amortised 16-site Cu-Au checkpoint out on every slice.

The amortised cell trains on a mixture of five slices (n_Au = 4..8 of 16). The
swap head sees (x, t) only, so it rolls out unchanged from any slice's uniform
base and the path weights re-target by construction (probe_zero_shot_transfer,
axis 2): slices outside the mixture are a zero-shot composition-transfer read
of the same checkpoint, not an amortised one. Draws from every slice are
concatenated into one eval/ directory beside the source run, in the layout
cuau16_figure.py already splits by slice; config.json is copied so the figure
can tell mixture slices from transfer slices.

Usage: pixi run -e dev python -m experiments.alloy_ce.probes.cuau16_amortised_sweep \\
           --run-dirs "results/03_hard/H2_cuau16_camort*perslice"
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch
from experiments.constrained_hard_03.probes.probe_zero_shot_transfer import _load_head
from experiments.constrained_hard_03.run import _chunked_eval_draw

from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.targets.cluster_expansion import (
    BinaryExpansionSpec,
    FixedCompositionClusterExpansionTarget,
)

D = 16
OUT_SUFFIX = "-fullcurve"


def sweep(run_dir: Path, checkpoint: str, n_au_values, n_samples: int | None, device):
    head, cfg = _load_head(run_dir, checkpoint, device)
    if n_samples is not None:
        cfg = replace(cfg, eval=replace(cfg.eval, n_eval_samples=n_samples))
    spec = BinaryExpansionSpec.from_json(cfg.ising.expansion_json)
    # cfg.ising.sigma is the 1200 K start of the temperature ladder; the cell
    # ends, and is evaluated, at the last stage (500 K).
    sigma = (
        cfg.curriculum.stages[-1].sigma
        if cfg.curriculum is not None
        else cfg.ising.sigma
    )
    torch.manual_seed(cfg.train.seed)
    samples, log_weights, per_slice = [], [], {}
    for n_au in n_au_values:
        target = FixedCompositionClusterExpansionTarget(
            spec,
            beta=2.0 * sigma,
            target_composition=n_au / D,
            bias=cfg.ising.bias,
            device=device,
        )
        started = time.time()
        with torch.no_grad():
            slice_samples, slice_log_w, _, _ = _chunked_eval_draw(
                head, target, cfg, multi_event=cfg.ctmc.use_matching_step, smc_tau=None
            )
        samples.append(slice_samples.cpu())
        log_weights.append(slice_log_w.cpu())
        per_slice[n_au] = {
            "ess_fraction": ess_from_log_weights(slice_log_w).item()
            / slice_log_w.numel(),
            "n_samples": int(slice_log_w.numel()),
            "seconds": round(time.time() - started, 1),
        }
        print(
            f"{run_dir.name} n_Au={n_au:2d} ESS/N={per_slice[n_au]['ess_fraction']:.3f} {per_slice[n_au]['seconds']}s"
        )

    out_dir = run_dir.with_name(run_dir.name + OUT_SUFFIX)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(samples), out_dir / "eval" / "samples.pt")
    torch.save(torch.cat(log_weights), out_dir / "eval" / "log_weights.pt")
    shutil.copy(run_dir / "config.json", out_dir / "config.json")
    (out_dir / "eval" / "metrics.json").write_text(
        json.dumps(
            {
                "source_run": run_dir.name,
                "checkpoint": checkpoint,
                "composition_mixture": cfg.composition_mixture,
                "per_slice": per_slice,
            },
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dirs", nargs="+", required=True, help="glob patterns")
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument(
        "--n-au",
        default=",".join(str(n) for n in range(1, D)),
        help="comma-separated up-counts; 0 and 16 are single-state slices with no swap pairs",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="per slice; default the cell's eval draw",
    )
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dirs = sorted(
        Path(r)
        for p in args.run_dirs
        for r in glob.glob(p)
        if not r.endswith(OUT_SUFFIX)
    )
    for run_dir in run_dirs:
        sweep(
            run_dir,
            args.checkpoint,
            [int(n) for n in args.n_au.split(",")],
            args.n_samples,
            device,
        )


if __name__ == "__main__":
    main()
