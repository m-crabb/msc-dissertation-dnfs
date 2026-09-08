"""Training cost of every printed sigma_c cell, priced from the loop structure.

Two cells were measured end to end by `measure_training_flops.py` (JSONs in
`results/03_hard/training_flops`); the measurement matched the accounted total
to 0.3% (8x8 patch head) and 0.05% (16x16 patch head, R=2) once the periodic
in-training eval was included. That agreement licenses pricing every other
cell by the same account with no GPU: head-forward counts from the recipe
(`training_forward_counts`) times a per-forward FLOP reading taken on the CPU.
The counter is exactly linear in batch (per-sample FLOPs identical at batch 1,
4 and 128 on the 8x8 patch head, and the batch-128 reading reproduces the
certified JSON), so the reading is taken at a small batch and scaled.

Masked-attention cells are billed under the separable contraction, as in the
house tables (`flop_billing_config`); rows that trained densely are flagged so
the caption can say their as-run cost was higher by the head's dense/separable
ratio.

GFlowNet cells have no Euler grid: one update is a sampled rollout plus the
loss pass and its backward, measured directly by wrapping one step in
`FlopCounterMode` (checked linear in batch), then scaled by the recipe's batch
and update count. Their in-training frozen-ESS diagnostic is excluded, as it is
for the swap heads (training-proper only).

    python -m experiments.constrained_hard_03.analysis.training_flops_table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from experiments.constrained_hard_03.analysis.house_table_8x8 import (
    flop_billing_config,
)
from experiments.constrained_hard_03.configs import CONFIGS
from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS
from experiments.constrained_hard_03.run import build_target_and_head
from experiments.constrained_hard_03.run_gfn import (
    _loss_and_train_diagnostics,
    build_target_and_policy,
)
from torch.utils.flop_counter import FlopCounterMode

from discrete_flow_sampler.diagnostics.flops import (
    diagnostic_eval_flops,
    measured_forward_flops,
    training_run_flops,
)

PROBE_BATCH = 4

# (size label, config key, row label, trained densely). Row order follows the
# body tables; masked-attention cells trained densely wherever the hparams
# appendix says so: every 8x8 sigma_c masked-attention cell and the 16x16
# one-sweep cell.
SWAP_ROWS = [
    ("8x8", "H2_d64_c50_s220_letf_mo_50k_curr_w2", "mask-one head", False),
    ("8x8", "H2_d64_c50_s220_letf_ma_50k_curr_w2", "masked-attention band, one sweep", True),
    ("8x8", "H2_d64_c50_s220_letf_mamo2_50k_curr_w2", "masked-attention band, two sweeps", True),
    ("8x8", "H2_d64_c50_s220_letf_mamo2ef_50k_curr_w2", "\\quad + exact field (masked attention)", True),
    ("8x8", "H2_d64_c50_s220_letf_iv_50k_curr_w2", "prefix-sum band, one sweep", False),
    ("8x8", "H2_d64_c50_s220_letf_ivmo2_50k_curr_w2", "prefix-sum band, two sweeps", False),
    ("8x8", "H2_d64_c50_s220_letf_ivmo2ef_50k_curr_w2", "\\quad + exact field (prefix sum)", False),
    ("8x8", "H2_d64_c50_s220_letf_thp_50k_curr_w2", "two-hole patch head, $R=1$", False),
    ("16x16", "H2_d256_c50_s220_letf_ma_100k_curr_b512_ne128_cv2_w3", "masked-attention band, one sweep", True),
    ("16x16", "H2_d256_c50_s220_letf_mamo2_100k_curr_b512_ne128_cv2_w3", "masked-attention band, two sweeps", False),
    ("16x16", "H2_d256_c50_s220_letf_mamo2ef_100k_curr_b512_ne128_cv2_w3", "\\quad + exact field (masked attention)", False),
    ("16x16", "H2_d256_c50_s220_letf_iv_100k_curr_b512_ne128_cv2_w3", "prefix-sum band, one sweep", False),
    ("16x16", "H2_d256_c50_s220_letf_ivmo2_100k_curr_b512_ne128_cv2_w3", "prefix-sum band, two sweeps", False),
    ("16x16", "H2_d256_c50_s220_letf_ivmo2ef_100k_curr_b512_ne128_cv2_w3", "\\quad + exact field (prefix sum)", False),
    ("16x16", "H2_d256_c50_s220_letf_thp_100k_curr_b512_ne128_cv2_w3", "two-hole patch head, $R=1$", False),
    ("16x16", "H2_d256_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w3", "two-hole patch head, $R=2$", False),
    ("20x20", "H2_d400_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w4bf16", "two-hole patch head, $R=2$", False),
    ("20x20", "H2_d400_c50_s220_letf_thp3_100k_curr_b512_ne128_cv2_w4bf16", "two-hole patch head, $R=3$", False),
    ("24x24", "H2_d576_c50_s220_letf_thp3_100k_curr_b512_ne128_cv2_w5bf16", "two-hole patch head, $R=3$", False),
    ("24x24", "H2_d576_c50_s220_letf_thp4_100k_curr_b512_ne128_cv2_w5bf16", "two-hole patch head, $R=4$", False),
]
GFN_ROWS = [
    ("8x8", "GFN_d64_c50_s220_tb_50k_par", "GFlowNet, trajectory balance"),
    ("8x8", "GFN_d64_c50_s220_fldb_50k_par", "GFlowNet, forward-looking DB"),
    ("16x16", "GFN_d256_c50_s220_tb_100k_par", "GFlowNet, trajectory balance"),
    ("16x16", "GFN_d256_c50_s220_fldb_100k_par", "GFlowNet, forward-looking DB"),
    ("20x20", "GFN_d400_c50_s220_tb_100k_par", "GFlowNet, trajectory balance"),
    ("24x24", "GFN_d576_c50_s220_tb_100k_par", "GFlowNet, trajectory balance"),
]
LABELS = {key: label for _, key, label, _ in SWAP_ROWS} | {
    key: label for _, key, label in GFN_ROWS
}
# Certified cells: measured / accounted from the harness JSONs, training-proper
# and with the in-training eval included.
MEASURED = {
    "H2_d64_c50_s220_letf_thp_50k_curr_w2": "training_flops_d64_thp_sc.json",
    "H2_d256_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w3": "training_flops_d256_thp2_sc.json",
}


def swap_cell(key: str) -> dict:
    cfg = flop_billing_config(CONFIGS[key])
    torch.manual_seed(0)
    target, head = build_target_and_head(cfg, "cpu")
    x = target.sample_base(PROBE_BATCH, device="cpu")
    with torch.no_grad():
        per_sample = measured_forward_flops(head, (x, torch.rand(PROBE_BATCH))) / PROBE_BATCH
    outer_batch = cfg.train.outer_batch_size or cfg.train.batch_size
    training = training_run_flops(
        per_sample * outer_batch,
        per_sample * cfg.train.batch_size,
        n_steps=cfg.train.n_steps,
        inner_steps_per_outer=cfg.train.inner_steps_per_outer,
        n_euler_steps=cfg.ctmc.n_euler_steps,
        c_t_from_rollout=cfg.train.c_t_from_rollout,
    )
    eval_draw_set = cfg.ctmc.n_euler_steps * per_sample * cfg.eval.n_eval_samples
    # The severable in-training frozen-ESS diagnostic, needed only to compare
    # the measured cells against the account (the counter measured both).
    diagnostic = diagnostic_eval_flops(
        per_sample * cfg.train.batch_size,
        update_batch_size=cfg.train.batch_size,
        n_euler_steps=cfg.ctmc.n_euler_steps,
        n_steps=cfg.train.n_steps,
        eval_every=getattr(cfg.eval, "eval_every", None),
        n_eval_draws=(
            getattr(cfg.eval, "n_eval_samples_training", None) or cfg.eval.n_eval_samples
        ),
    )
    return {
        "per_sample_forward_flops": per_sample,
        "n_steps": cfg.train.n_steps,
        "training_flops": training,
        "diagnostic_eval_flops": diagnostic,
        "eval_draw_set_flops": eval_draw_set,
        "training_in_eval_draw_sets": training / eval_draw_set,
    }


def gfn_step_flops(cfg, policy, target, batch: int) -> int:
    policy.zero_grad()
    counter = FlopCounterMode(display=False)
    with counter:
        spins, _ = policy.sample(batch, epsilon=cfg.epsilon)
        loss, _ = _loss_and_train_diagnostics(cfg, policy, target, spins)
        loss.backward()
    return counter.get_total_flops()


def gfn_cell(key: str) -> dict:
    cfg = GFN_CONFIGS[key]
    torch.manual_seed(0)
    target, policy = build_target_and_policy(cfg, "cpu")
    policy.train()
    per_step = {b: gfn_step_flops(cfg, policy, target, b) / b for b in (2, PROBE_BATCH)}
    per_sample_step = per_step[PROBE_BATCH]
    linearity = per_step[2] / per_sample_step
    if abs(linearity - 1.0) > 1e-3:
        raise RuntimeError(f"{key}: GFN step FLOPs not linear in batch ({linearity:.4f})")
    training = per_sample_step * cfg.batch_size * cfg.n_steps
    policy.eval()
    with torch.no_grad():
        rollout = measured_forward_flops(policy.sample, (1,))
    eval_draw_set = rollout * cfg.n_eval_samples
    return {
        "per_sample_step_flops": per_sample_step,
        "per_sample_rollout_flops": rollout,
        "n_steps": cfg.n_steps,
        "training_flops": training,
        "eval_draw_set_flops": eval_draw_set,
        "training_in_eval_draw_sets": training / eval_draw_set,
    }


def sci(value: float, suffix: str = "") -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}{suffix}$"


SIZES = ("8x8", "16x16", "20x20", "24x24")


def latex_rows(table: dict) -> str:
    """One row per head, one column per size: the eval-draw-set ratio is a
    recipe constant (43 for every 50k cell, 342 for every 100k cell), so it
    is stated in the caption rather than repeated down a column."""
    by_label: dict[str, dict[str, dict]] = {}
    for row in table["rows"]:
        by_label.setdefault(LABELS[row["cfg"]], {})[row["size"]] = row
    lines = []
    for label, cells in by_label.items():
        entries = []
        for size in SIZES:
            cell = cells.get(size)
            if cell is None:
                entries.append("--")
                continue
            flag = "{}^{\\dagger}" if cell["trained_dense"] else ""
            entries.append(sci(cell["training_flops"], flag))
        printed = label.split(" (")[0]
        lines.append(f"        {printed} & " + " & ".join(entries) + " \\\\")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results/03_hard/training_flops"))
    parser.add_argument(
        "--reuse", action="store_true", help="re-emit the LaTeX from the saved JSON"
    )
    args = parser.parse_args()
    if args.reuse:
        table = json.loads((args.out / "training_flops_table.json").read_text())
        tex = latex_rows(table)
        (args.out / "training_flops_table.tex").write_text(tex + "\n")
        print(tex)
        return

    rows = []
    for size, key, label, dense in SWAP_ROWS:
        cell = swap_cell(key)
        if key in MEASURED:
            measured = json.loads((args.out / MEASURED[key]).read_text())
            cell["measured_over_accounted"] = measured["extrapolated_training_flops"] / (
                cell["training_flops"] + cell["diagnostic_eval_flops"]
            )
        rows.append({"size": size, "cfg": key, "label": label, "trained_dense": dense, **cell})
        print(f"{size:>6} {label:<40} {cell['training_flops']:.2e}  {cell['training_in_eval_draw_sets']:6.0f} evals")
    for size, key, label in GFN_ROWS:
        cell = gfn_cell(key)
        rows.append({"size": size, "cfg": key, "label": label, "trained_dense": False, **cell})
        print(f"{size:>6} {label:<40} {cell['training_flops']:.2e}  {cell['training_in_eval_draw_sets']:6.0f} evals")

    table = {"probe_batch": PROBE_BATCH, "rows": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "training_flops_table.json").write_text(json.dumps(table, indent=2))
    tex = latex_rows(table)
    (args.out / "training_flops_table.tex").write_text(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()
