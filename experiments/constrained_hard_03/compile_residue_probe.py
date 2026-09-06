"""Forward-only compile-vs-eager residue probe (mechanism leg).

Paired re-runs localised the catastrophic-seed cell to
factorised x compile_head x exact sigma_c without isolating a mechanism.
The candidate on record: the factorised head is the one tested head whose
blindness rests on floating-point CANCELLATION (global term
sum_k psi_k - psi_i - psi_j, "rounding" row of tab:head-ladder), and
inductor reassociates reductions, so compiling may enlarge the blindness
residue exactly where the dynamic range is largest (critical coupling).

This probe tests the FORWARD half of that story on trained weights, no
training: a null here is informative too (it pushes the suspicion to the
backward/training dynamics under the fused kernels).

Instruments, all read through `head(x, t)` because `compile_head` is an
in-place nn.Module.compile that routes only forward through inductor
(direct method calls such as compute_pair_context stay eager):

  * state-swap antisymmetry violation, max_{i<j} |G(i,j|swap_ij x) + G(i,j|x)|
    (gate_4x4's certified instrument). The swapped state differs from the
    base ONLY at the two holes, so with an odd readout any violation is a
    context leak -- this IS the end-to-end blindness measure, and it works
    identically through eager and compiled forwards.
  * forward parity |G_compiled - G_eager| (max / mean / p99 over pairs and
    states), the raw size of the numerics perturbation.
  * eager-only compute_pair_context drift under hole flips (the direct
    cancellation residue), as the baseline scale the violation is compared
    against.

States: the run's own eval samples (samples.pt -- trained-distribution,
critical-scale) plus fresh uniform on-manifold draws as a control.
Weights: final.pt of the named run dirs (pass a healthy and a catastrophic
seed to read whether the leak separates them), plus a fresh-init control.
Venue: the training GPU (A100) for inductor-parity with the failed runs;
the script itself is device-agnostic so the D=4-scale smoke runs on CPU.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def antisymmetry_violation(head, states, t_value, d):
    """gate_4x4's instrument, inlined so the probe controls the time value:
    max over i<j pairs of |G(i,j|swap_ij x) + G(i,j|x)|, plus the mean."""
    from experiments.constrained_hard_03.gate_4x4 import swap2, upper_tri_pairs

    t = torch.full((states.shape[0],), t_value, device=states.device)
    base = head(states, t)
    max_violation, sum_violation, n_pairs = 0.0, 0.0, 0
    for i, j in upper_tri_pairs(d, states.device).tolist():
        swapped = head(swap2(states, i, j), t)
        pair = (swapped[:, i, j] + base[:, i, j]).abs()
        max_violation = max(max_violation, pair.max().item())
        sum_violation += pair.mean().item()
        n_pairs += 1
    return max_violation, sum_violation / n_pairs


def context_drift_eager(head, states, t_value, d, n_pairs=8):
    """Direct cancellation residue, eager numerics only: max drift of
    H_[i,j] under flips of x_i / x_j / both, over a probe pair set."""
    inner = head.head if hasattr(head, "head") else head  # unwrap ef
    if not hasattr(inner, "compute_pair_context"):
        return None
    t = torch.full((states.shape[0],), t_value, device=states.device)
    generator = torch.Generator().manual_seed(0)
    pairs = [(int(a), int(b)) for a, b in
             torch.randint(0, d, (n_pairs, 2), generator=generator)
             if a != b]
    H = inner.compute_pair_context(states, t)
    drift = 0.0
    for i, j in pairs:
        base = H[:, i, j, :]
        for flip_sites in ((i,), (j,), (i, j)):
            flipped = states.clone()
            for s in flip_sites:
                flipped[:, s] = -flipped[:, s]
            moved = inner.compute_pair_context(flipped, t)[:, i, j, :]
            drift = max(drift, (moved - base).abs().max().item())
    return drift


def probe_one(cfg, checkpoint, states, label, device, t_values=(0.5, 1.0)):
    from dataclasses import replace

    from experiments.constrained_hard_03.run import build_target_and_head

    d = states.shape[1]
    report = {"label": label}
    heads = {}
    for mode, compiled in (("eager", False), ("compiled", True)):
        _, head = build_target_and_head(
            replace(cfg, compile_head=compiled), device=device
        )
        if checkpoint is not None:
            head.load_state_dict(checkpoint)
        head.eval()
        heads[mode] = head

    with torch.no_grad():
        for t_value in t_values:
            key = f"t{t_value}"
            t = torch.full((states.shape[0],), t_value, device=device)
            g_eager = heads["eager"](states, t)
            g_compiled = heads["compiled"](states, t)
            gap = (g_compiled - g_eager).abs()
            report[key] = {
                "G_scale_mean": g_eager.abs().mean().item(),
                "parity_max": gap.max().item(),
                "parity_mean": gap.mean().item(),
                "parity_p99": gap.flatten().quantile(0.99).item(),
            }
            for mode in ("eager", "compiled"):
                vmax, vmean = antisymmetry_violation(
                    heads[mode], states, t_value, d)
                report[key][f"antisym_max_{mode}"] = vmax
                report[key][f"antisym_mean_{mode}"] = vmean
            report[key]["context_drift_eager"] = context_drift_eager(
                heads["eager"], states, t_value, d)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--run-dirs", required=True,
        help="comma-separated run dir names; final.pt + config.json + "
             "eval/samples.pt are read from each")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-states", type=int, default=256)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.run import build_target_and_head

    reports = []
    for name in [n.strip() for n in args.run_dirs.split(",") if n.strip()]:
        run_dir = args.results_dir / name
        cfg = CONFIGS[json.loads((run_dir / "config.json").read_text())["name"]]
        checkpoint = torch.load(
            run_dir / "checkpoints" / "final.pt",
            map_location=args.device, weights_only=True)
        samples = torch.load(
            run_dir / "eval" / "samples.pt",
            map_location=args.device, weights_only=True
        ).float()[: args.n_states]
        reports.append(probe_one(
            cfg, checkpoint, samples, f"{name} [trained states]", args.device))

        target, _ = build_target_and_head(cfg, device=args.device)
        base_states = target.sample_base(args.n_states, device=args.device)
        reports.append(probe_one(
            cfg, checkpoint, base_states.float(),
            f"{name} [uniform on-manifold]", args.device))

    # Fresh-init control on the last cfg: the residue scale before training.
    _, _ = build_target_and_head(cfg, device=args.device)
    reports.append(probe_one(
        cfg, None, base_states.float(), "fresh init [uniform]", args.device))

    text = json.dumps(reports, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
