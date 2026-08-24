#!/bin/bash
#SBATCH --job-name=fc-redraw-grids
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# F(c) eval-grid redraws for the Richardson extrapolation (2026-08-24, s62).
# Every passing ne128-trained window seed is redrawn at ne256 and ne512, and
# the ne64-trained c=0.30 fallback seeds at ne128 and ne256, so each
# checkpoint carries its own two-grid pair (the error is first order: the
# 21-Aug prereg measured step ratios 0.44-0.49 across ne64->128->256, and the
# s62 retrain halved the residual vs the TI truth in all four windows).
# redraw_seed=45 matches the 21-Aug convention. Artefacts go to side dirs
# eval_ne<k>/ beside the frozen eval/ (eval_only leaves eval/ untouched when
# a grid override is set). Reruns skip pairs whose metrics already exist.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python - <<'PY'
from pathlib import Path

from experiments.dnfs_baseline_01.run import eval_only

# Passing seeds per window (eval ESS fraction >= 0.30, read 2026-08-24):
# c05 4/4, c055 4/4, c060 3/4 (45 fails at 0.015), c065 3/4 (44 dead).
# The ne64 c030 fallback keeps its three passing seeds plus borderline 47.
NE128_PASSING = {
    "c05": (42, 43, 44, 45),
    "c055": (42, 43, 44, 45),
    "c060": (42, 43, 44),
    "c065": (42, 43, 45),
}
jobs = [
    (f"S2_d10_{window}_l50_letf_ne128_anneal", seed, (256, 512))
    for window, seeds in NE128_PASSING.items()
    for seed in seeds
]
jobs += [
    ("S2_d10_c030_l50_letf_ne64_anneal", seed, (128, 256))
    for seed in (43, 44, 46, 47)
]
# c=0.80 joins the plotted curve (s62): its passing pair from the N11
# uniform-base control family (ESS 0.830 / 0.423; seeds 42/43 fail the
# floor at 0.0004 / 0.059). Checkpoints synced up from the Modal volume.
jobs += [
    ("S2_d10_c080_l50_letf_ne128_anneal", seed, (256, 512))
    for seed in (44, 45)
]

results_root = Path("results/02_constrained_soft")
for cfg_name, seed, grids in jobs:
    run_dirs = [
        d for d in sorted(results_root.glob(f"{cfg_name}_seed{seed}_*"))
        if (d / "checkpoints" / "final.pt").exists()
    ]
    run_dir = run_dirs[-1]
    for n_euler in grids:
        if (run_dir / f"eval_ne{n_euler}" / "metrics.json").exists():
            print(f"skip {run_dir.name} ne{n_euler} (exists)", flush=True)
            continue
        print(f"redraw {run_dir.name} ne{n_euler}", flush=True)
        eval_only(run_dir, redraw=True, redraw_seed=45, n_euler_override=n_euler)
PY
