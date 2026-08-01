"""d=4 constrained Ising at c_target=0.5: exact distribution vs DNFS IS-weighted.

Mirrors `01_c03_d4_exact_fidelity.py` but adds two diagnostics specific to
the c_target=0.5:

  1. **Conditional p(log p̃ | c = 0.5)** — the c-marginal is uninformative
     at c_target=0.5 (Z_2 symmetry pins ⟨c⟩ = 0.5 by construction), so the
     conditional energy distribution on the c=0.5 slice is the primary
     fidelity diagnostic. The slice has C(16,8) = 12,870 states.

  2. **Z_2 symmetry check** — the bias-zero, c_target=0.5 target is exactly
     Z_2-symmetric (π(x) = π(−x)). The IS-weighted ⟨m⟩ should be 0; the
     |mass(m>0) − mass(m<0)| asymmetry measures how badly q_θ has failed
     to learn the invariance the leTF isn't manifestly equivariant under.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    conditional_pmf_at_composition,
    enumerate_states,
    ess_from_log_weights,
    exact_log_probs,
    z2_asymmetry_from_samples,
)
from discrete_flow_sampler.targets.ising import IsingTarget
from discrete_flow_sampler.diagnostics.metrics import (
    composition_fraction_up as composition,
    marginal_tvd,
)

N_SITES = 16  # D=4 -> d = 16; 2^16 = 65,536 enumerable states


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=Path,
                   help="DNFS results dir (containing config.json + eval/)")
    p.add_argument("--n_bins_energy", type=int, default=40)
    args = p.parse_args()

    run_dir = args.run_dir
    cfg = json.loads((run_dir / "config.json").read_text())["ising"]
    if cfg["target_composition"] != 0.5:
        print(f"WARNING: this script is for c_target=0.5; run has "
              f"c_target={cfg['target_composition']}.")
    samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    stored = json.loads((run_dir / "eval" / "metrics.json").read_text())

    target = IsingTarget(
        D=cfg["D"], sigma=cfg["sigma"], bias=cfg["bias"],
        target_composition=cfg["target_composition"],
        composition_penalty_strength=cfg["composition_penalty_strength"],
    )

    # --- Exact distribution over all 2^16 states ------------------------
    states = enumerate_states(N_SITES)                  # (65536, 16) int64
    states_f = states.float()
    log_pi = exact_log_probs(target, states_f)          # normalised
    pi = log_pi.exp()

    c_states = composition(states_f)
    exact_mean_c = (pi * c_states).sum().item()
    exact_mean_m = (pi * states_f.mean(dim=-1)).sum().item()

    # Composition marginal — 17 support points k/16, k = 0..16
    support_c = torch.arange(N_SITES + 1).float() / N_SITES
    bucket = (c_states * N_SITES).round().long()
    exact_c_pmf = torch.zeros(N_SITES + 1).index_add_(0, bucket, pi)

    # Full energy marginal
    energy_states = target.log_prob(states_f)
    e_lo, e_hi = energy_states.min().item(), energy_states.max().item()
    e_edges = torch.linspace(e_lo, e_hi, args.n_bins_energy + 1)
    e_idx_states = torch.bucketize(energy_states, e_edges[1:-1], right=False)
    exact_e_pmf = torch.zeros(args.n_bins_energy).index_add_(0, e_idx_states, pi)

    # Conditional p(log p̃ | c=0.5) on the c=0.5 slice (n_plus = 8)
    n_plus_target = N_SITES // 2  # 8 for c_target=0.5, d=16
    slice_states, log_pi_cond = conditional_pmf_at_composition(
        states, log_pi, n_plus_target=n_plus_target,
    )
    pi_cond = log_pi_cond.exp()
    energy_slice = target.log_prob(slice_states.float())
    es_lo, es_hi = energy_slice.min().item(), energy_slice.max().item()
    es_edges = torch.linspace(es_lo, es_hi, args.n_bins_energy + 1)
    es_idx = torch.bucketize(energy_slice, es_edges[1:-1], right=False)
    exact_cond_pmf = torch.zeros(args.n_bins_energy).index_add_(0, es_idx, pi_cond)

    # --- IS-weighted empirical from DNFS samples -----------------------
    w = torch.softmax(log_w, dim=0)
    c_samples = composition(samples)
    weighted_mean_c = (w * c_samples).sum().item()
    weighted_mean_m = (w * samples.mean(dim=-1)).sum().item()

    bucket_s = (c_samples * N_SITES).round().long().clamp(0, N_SITES)
    weighted_c_pmf = torch.zeros(N_SITES + 1).index_add_(0, bucket_s, w)

    energy_samples = target.log_prob(samples)
    e_idx_s = torch.bucketize(energy_samples, e_edges[1:-1], right=False)
    weighted_e_pmf = torch.zeros(args.n_bins_energy).index_add_(0, e_idx_s, w)

    # IS-weighted conditional on the c=0.5 slice (samples that landed at c=0.5)
    n_plus_samples = ((samples + 1.0) * 0.5).sum(dim=-1).round().long()
    on_slice = n_plus_samples == n_plus_target
    if on_slice.sum() == 0:
        print(f"WARNING: 0 of {samples.shape[0]} samples landed on c=0.5 slice.")
        weighted_cond_pmf = torch.zeros(args.n_bins_energy)
    else:
        w_slice = w[on_slice]
        w_slice = w_slice / w_slice.sum()       # renormalise on the slice
        e_slice_samples = target.log_prob(samples[on_slice])
        es_idx_s = torch.bucketize(e_slice_samples, es_edges[1:-1], right=False)
        weighted_cond_pmf = torch.zeros(args.n_bins_energy).index_add_(
            0, es_idx_s, w_slice,
        )

    # Z_2 asymmetry on full IS-weighted samples
    z2 = z2_asymmetry_from_samples(samples, log_w)
    ess = ess_from_log_weights(log_w).item()
    ess_frac = ess / samples.shape[0]

    # --- Headline numbers ----------------------------------------------
    print(f"=== d=4 c_target=0.5 exact-fidelity vs DNFS ({run_dir.name}) ===")
    print(f"  exact   E_pi[c+]            : {exact_mean_c:.4f}    (Z_2 pins = 0.5)")
    print(f"  IS-weighted  E_hat[c+]      : {weighted_mean_c:.4f}")
    print(f"  stored unweighted mean c    : {stored['composition_mean']:.4f}")
    print(f"  exact   E_pi[m]             : {exact_mean_m:+.4f}    (Z_2 pins = 0)")
    print(f"  IS-weighted  E_hat[m]       : {weighted_mean_m:+.4f}    "
          f"(Z_2 break = {abs(weighted_mean_m):.4f})")
    print(f"  ESS / N                     : {ess_frac:.3f}  (ESS = {ess:.0f})")
    print()
    print("  Z_2 mass split (should be 0.5/0.5 at c_target=0.5, bias=0):")
    print(f"    mass(m>0) = {z2['mass_pos']:.4f}   mass(m<0) = {z2['mass_neg']:.4f}   "
          f"mass(m=0) = {z2['mass_zero']:.4f}")
    print(f"    asymmetry |pos - neg|     = {z2['asymmetry']:.4f}")
    print()
    print("  TVDs (sample-budget-limited at N=5000):")
    print(f"    composition marginal      : {marginal_tvd(weighted_c_pmf, exact_c_pmf):.4f}")
    print(f"    energy marginal (full)    : {marginal_tvd(weighted_e_pmf, exact_e_pmf):.4f}")
    print(f"    energy | c=0.5 conditional: {marginal_tvd(weighted_cond_pmf, exact_cond_pmf):.4f}")
    print(f"    (n_samples on c=0.5 slice : {int(on_slice.sum().item())} / {samples.shape[0]})")

    # --- Figure --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(support_c - 0.012, exact_c_pmf, width=0.024, label="exact π", alpha=0.7)
    ax.bar(support_c + 0.012, weighted_c_pmf, width=0.024, label="IS-weighted", alpha=0.7)
    ax.axvline(cfg["target_composition"], ls="--", c="k", lw=1, label="c_target")
    ax.set_xlabel(r"composition $c_+$")
    ax.set_ylabel("probability")
    ax.set_title("composition marginal (Z_2 → ⟨c⟩=0.5 by construction)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    es_centres = 0.5 * (es_edges[:-1] + es_edges[1:])
    ax.plot(es_centres, exact_cond_pmf, label="exact π(·|c=0.5)", lw=2)
    ax.plot(es_centres, weighted_cond_pmf, label="IS-weighted", lw=2, alpha=0.85)
    ax.set_xlabel(r"$\log \tilde p(x)$  on  c=0.5 slice")
    ax.set_ylabel("probability")
    ax.set_title(f"conditional p(log p̃ | c=0.5)   (slice size = {len(pi_cond)})")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = ["m>0", "m=0", "m<0"]
    is_mass = [z2["mass_pos"], z2["mass_zero"], z2["mass_neg"]]
    exact_mass = [
        float((pi * (states_f.mean(dim=-1) > 0).float()).sum().item()),
        float((pi * (states_f.mean(dim=-1) == 0).float()).sum().item()),
        float((pi * (states_f.mean(dim=-1) < 0).float()).sum().item()),
    ]
    xpos = torch.arange(3).float()
    width = 0.35
    ax.bar(xpos - width / 2, exact_mass, width=width, label="exact π", alpha=0.8)
    ax.bar(xpos + width / 2, is_mass, width=width, label="IS-weighted DNFS", alpha=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("IS mass")
    ax.set_title(f"Z_2 mass split   (asymmetry = {z2['asymmetry']:.4f})")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    e_centres = 0.5 * (e_edges[:-1] + e_edges[1:])
    ax.plot(e_centres, exact_e_pmf, label="exact π")
    ax.plot(e_centres, weighted_e_pmf, label="IS-weighted")
    ax.set_xlabel(r"$\log \tilde p(x)$")
    ax.set_ylabel("probability")
    ax.set_title("energy marginal (full)")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"d=4 c_target=0.5 exact fidelity  |  {run_dir.name}  |  "
        f"ESS frac = {ess_frac:.3f},  Z_2 asym = {z2['asymmetry']:.4f}"
    )
    fig.tight_layout()
    out_png = run_dir / "c05_d4_exact_fidelity.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nsaved figure to {out_png}")


if __name__ == "__main__":
    main()
