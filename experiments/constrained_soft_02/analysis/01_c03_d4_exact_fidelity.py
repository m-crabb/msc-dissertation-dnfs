"""d=4 constrained Ising: exact distribution vs DNFS IS-weighted samples.

The c=0.3 d=4 target (16 sites) is exactly enumerable (2^16 = 65,536 states),
so the constrained pmf is known exactly. The eval samples are CTMC proposal
draws with importance log-weights; the faithful estimate of any target
expectation is the SELF-NORMALISED IS estimate softmax(log_w)·f, NOT the raw
sample mean (the stored metrics.json `composition_mean` is the unweighted
proposal mean).

Reports:
  - exact E_π[c₊]  vs  IS-weighted Ê[c₊]  vs  stored unweighted mean (0.3047)
  - composition marginal (17 support points k/16) exact vs IS-weighted + TVD
  - log p̃ (energy) marginal exact vs IS-weighted (binned) + TVD
  - context for the metrics.json F/D bias (+0.18): the IS weight-tail
    signature (design §1.2)

Marginal TVDs (low-dim: 17 / ~40 bins) are informative at N=5000; the
full 2^16-state TVD is sample-floored at this budget (project_tvd_floor).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    exact_log_probs,
)
from discrete_flow_sampler.targets.ising import IsingTarget
from discrete_flow_sampler.diagnostics.metrics import (
    composition_fraction_up as composition,
    marginal_tvd,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "results" / "02_constrained_soft" / "S2_d4_c03_l50_seed42"
N_SITES = 16  # D=4 ⇒ d = 16


def main() -> None:
    cfg = json.loads((RUN_DIR / "config.json").read_text())["ising"]
    samples = torch.load(RUN_DIR / "eval" / "samples.pt", weights_only=True).float()
    log_w = torch.load(RUN_DIR / "eval" / "log_weights.pt", weights_only=True)
    stored = json.loads((RUN_DIR / "eval" / "metrics.json").read_text())

    target = IsingTarget(
        D=cfg["D"],
        sigma=cfg["sigma"],
        bias=cfg["bias"],
        target_composition=cfg["target_composition"],
        composition_penalty_strength=cfg["composition_penalty_strength"],
    )

    # --- Exact constrained distribution over all 2^16 states ---
    states = enumerate_states(N_SITES).float()          # (65536, 16)
    log_pi = exact_log_probs(target, states)             # normalised, Σ exp = 1
    pi = log_pi.exp()                                    # (65536,)

    c_states = composition(states)                       # (65536,)
    exact_mean_c = (pi * c_states).sum().item()

    # 17 composition support points: k/16, k = 0..16
    k = torch.arange(N_SITES + 1)
    support_c = k.float() / N_SITES
    bucket = (c_states * N_SITES).round().long()         # state -> support index
    exact_c_pmf = torch.zeros(N_SITES + 1)
    exact_c_pmf.index_add_(0, bucket, pi)

    energy_states = target.log_prob(states)              # log p̃ per state
    e_lo, e_hi = energy_states.min().item(), energy_states.max().item()
    e_edges = torch.linspace(e_lo, e_hi, 41)             # ~40 energy bins
    e_idx_states = torch.bucketize(energy_states, e_edges[1:-1], right=False)
    exact_e_pmf = torch.zeros(40)
    exact_e_pmf.index_add_(0, e_idx_states, pi)

    # --- IS-weighted empirical from the c=0.3 run ---
    w = torch.softmax(log_w, dim=0)                      # self-normalised IS
    c_samples = composition(samples)
    weighted_mean_c = (w * c_samples).sum().item()

    bucket_s = (c_samples * N_SITES).round().long().clamp(0, N_SITES)
    weighted_c_pmf = torch.zeros(N_SITES + 1)
    weighted_c_pmf.index_add_(0, bucket_s, w)

    energy_samples = target.log_prob(samples)
    e_idx_s = torch.bucketize(energy_samples, e_edges[1:-1], right=False)
    weighted_e_pmf = torch.zeros(40)
    weighted_e_pmf.index_add_(0, e_idx_s, w)

    # --- Headline numbers ---
    print("=== d=4 constrained exact-fidelity ===")
    print(f"  exact   E_pi[c+]            : {exact_mean_c:.4f}")
    print(f"  IS-weighted  E_hat[c+]      : {weighted_mean_c:.4f}")
    print(f"  stored unweighted mean      : {stored['composition_mean']:.4f}  "
          f"(proposal Q, not target)")
    print(f"  composition bias (exact-IS) : {exact_mean_c - weighted_mean_c:+.4f}")
    print(f"  composition marginal TVD    : {marginal_tvd(weighted_c_pmf, exact_c_pmf):.4f}")
    print(f"  energy marginal TVD         : {marginal_tvd(weighted_e_pmf, exact_e_pmf):.4f}")
    print()
    print("  context — stored scalar biases (metrics.json, already exact at d=4):")
    print(f"    F/D bias {stored['free_energy_per_site_bias']:+.4f}  "
          f"E/D bias {stored['internal_energy_per_site_bias']:+.4f}  "
          f"S/D bias {stored['entropy_per_site_bias']:+.4f}")
    print("    F/D bias is large+positive vs near-zero E/D: the IS weight-tail")
    print("    signature (Eq.37 absolute log-Ẑ bound vs Eq.38 self-normalised).")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(support_c - 0.012, exact_c_pmf, width=0.024, label="exact π", alpha=0.7)
    axes[0].bar(support_c + 0.012, weighted_c_pmf, width=0.024, label="IS-weighted", alpha=0.7)
    axes[0].axvline(cfg["target_composition"], ls="--", c="k", lw=1, label="c_target")
    axes[0].set_xlabel(r"composition $c_+$")
    axes[0].set_ylabel("probability")
    axes[0].set_title("Composition marginal: exact vs IS-weighted")
    axes[0].legend()

    e_centres = 0.5 * (e_edges[:-1] + e_edges[1:])
    axes[1].plot(e_centres, exact_e_pmf, label="exact π")
    axes[1].plot(e_centres, weighted_e_pmf, label="IS-weighted")
    axes[1].set_xlabel(r"$\log \tilde p(x)$")
    axes[1].set_ylabel("probability")
    axes[1].set_title("Energy marginal: exact vs IS-weighted")
    axes[1].legend()

    fig.tight_layout()
    out_png = RUN_DIR / "c03_d4_exact_fidelity.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nsaved figure to {out_png}")


if __name__ == "__main__":
    main()
