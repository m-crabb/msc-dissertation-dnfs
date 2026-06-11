"""d=10 constrained Ising: DNFS IS-weighted samples vs penalty-aware Gibbs.

No exact enumeration at d=10 (2^100), so the penalty-aware Gibbs chain
(notebooks/stage_2_d10_constrained_gibbs_reference.py, validated by small
exact checks + its own 3 mixing diagnostics) is the empirical reference.
Compares low-dim marginals only (composition: 101 support points; log p̃:
~40 bins) — informative at N=5000 but finite-sample floored; full 2^100 TVD
is meaningless.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
N_SITES = 100  # D=10 ⇒ d = 100


def composition(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).mean(dim=-1)


def marginal_tvd(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * (p - q).abs().sum().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="S2_d10_c03_l50_seed42",
                        help="Run dir under results/02_constrained_soft/ to evaluate.")
    args = parser.parse_args()
    run_dir = RESULTS / args.run

    cfg = json.loads((run_dir / "config.json").read_text())["ising"]
    samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    stored = json.loads((run_dir / "eval" / "metrics.json").read_text())
    ref = torch.load(RESULTS / "gibbs_chain_d10_c03.pt", weights_only=False)
    ref_samples = ref["samples"].float()

    target = IsingTarget(
        D=cfg["D"], sigma=cfg["sigma"], bias=cfg["bias"],
        target_composition=cfg["target_composition"],
        composition_penalty_strength=cfg["composition_penalty_strength"],
    )

    # Gibbs draws are unweighted samples from the exact target kernel, so their
    # empirical histogram is the finite-sample reference. DNFS samples are
    # IS-weighted.
    w = torch.softmax(log_w, dim=0)
    c_dnfs = composition(samples)
    c_ref = composition(ref_samples)
    dnfs_mean_c = (w * c_dnfs).sum().item()
    ref_mean_c = c_ref.mean().item()

    # Composition marginal: 101 support points k/100.
    k = torch.arange(N_SITES + 1)
    support_c = k.float() / N_SITES
    b_dnfs = (c_dnfs * N_SITES).round().long().clamp(0, N_SITES)
    b_ref = (c_ref * N_SITES).round().long().clamp(0, N_SITES)
    dnfs_c_pmf = torch.zeros(N_SITES + 1).index_add_(0, b_dnfs, w)
    ref_c_pmf = torch.zeros(N_SITES + 1).index_add_(
        0, b_ref, torch.full((c_ref.numel(),), 1.0 / c_ref.numel())
    )

    # log p̃ (energy) marginal on shared edges from both sample sets.
    e_dnfs = target.log_prob(samples)
    e_ref = target.log_prob(ref_samples)
    e_lo = torch.minimum(e_dnfs.min(), e_ref.min()).item()
    e_hi = torch.maximum(e_dnfs.max(), e_ref.max()).item()
    e_edges = torch.linspace(e_lo, e_hi, 41)
    ei_dnfs = torch.bucketize(e_dnfs, e_edges[1:-1], right=False)
    ei_ref = torch.bucketize(e_ref, e_edges[1:-1], right=False)
    dnfs_e_pmf = torch.zeros(40).index_add_(0, ei_dnfs, w)
    ref_e_pmf = torch.zeros(40).index_add_(
        0, ei_ref, torch.full((e_ref.numel(),), 1.0 / e_ref.numel())
    )

    # Pure-Ising base log-prob (penalty stripped). Tells us whether the
    # learned sampler matches the lattice physics independent of the soft
    # constraint pulling it toward c_target.
    eb_dnfs = target.base_log_prob(samples)
    eb_ref = target.base_log_prob(ref_samples)
    eb_lo = torch.minimum(eb_dnfs.min(), eb_ref.min()).item()
    eb_hi = torch.maximum(eb_dnfs.max(), eb_ref.max()).item()
    eb_edges = torch.linspace(eb_lo, eb_hi, 41)
    ebi_dnfs = torch.bucketize(eb_dnfs, eb_edges[1:-1], right=False)
    ebi_ref = torch.bucketize(eb_ref, eb_edges[1:-1], right=False)
    dnfs_eb_pmf = torch.zeros(40).index_add_(0, ebi_dnfs, w)
    ref_eb_pmf = torch.zeros(40).index_add_(
        0, ebi_ref, torch.full((eb_ref.numel(),), 1.0 / eb_ref.numel())
    )

    # Per-site +1 probability vector (length d=100). Translation-symmetry
    # check: under the Ising target the per-site marginals are exchangeable
    # (J is translation-invariant on the lattice), so deviations from a flat
    # band of ~c_target reveal mode collapse / spatial-bias artefacts.
    site_p_dnfs = ((samples + 1.0) * 0.5 * w.unsqueeze(-1)).sum(dim=0)
    site_p_ref = ((ref_samples + 1.0) * 0.5).mean(dim=0)

    print("=== d=10 constrained fidelity (DNFS IS-weighted vs Gibbs ref) ===")
    print(f"  run                        : {args.run}")
    print(f"  Gibbs ref   mean c+        : {ref_mean_c:.4f}")
    print(f"  DNFS IS-wtd mean c+        : {dnfs_mean_c:.4f}")
    print(f"  stored unweighted mean     : {stored['composition_mean']:.4f}  (proposal Q)")
    print(f"  composition bias (ref-DNFS): {ref_mean_c - dnfs_mean_c:+.4f}")
    print(f"  composition marginal TVD   : {marginal_tvd(dnfs_c_pmf, ref_c_pmf):.4f}")
    print(f"  log p̃ marginal TVD         : {marginal_tvd(dnfs_e_pmf, ref_e_pmf):.4f}")
    print(f"  base log-prob marginal TVD : {marginal_tvd(dnfs_eb_pmf, ref_eb_pmf):.4f}")
    print(f"  per-site p(+1) max |dev|   : {(site_p_dnfs - site_p_ref).abs().max().item():.4f}")
    print(f"  per-site p(+1) RMS dev     : {(site_p_dnfs - site_p_ref).pow(2).mean().sqrt().item():.4f}")
    print(f"  context: ess_fraction={stored['ess_fraction']:.3f} "
          f"F/D={stored['free_energy_per_site']:.3f} "
          f"(no exact ref at d=10; Gibbs is an empirical reference)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(support_c, ref_c_pmf, label="Gibbs ref")
    axes[0, 0].plot(support_c, dnfs_c_pmf, label="DNFS IS-weighted")
    axes[0, 0].axvline(cfg["target_composition"], ls="--", c="k", lw=1, label="c_target")
    axes[0, 0].set_xlabel(r"composition $c_+$"); axes[0, 0].set_ylabel("probability")
    axes[0, 0].set_xlim(0.2, 0.4)
    axes[0, 0].set_title("composition marginal"); axes[0, 0].legend()

    ec = 0.5 * (e_edges[:-1] + e_edges[1:])
    axes[0, 1].plot(ec, ref_e_pmf, label="Gibbs ref")
    axes[0, 1].plot(ec, dnfs_e_pmf, label="DNFS IS-weighted")
    axes[0, 1].set_xlabel(r"$\log \tilde p(x)$ (constrained)")
    axes[0, 1].set_ylabel("probability")
    axes[0, 1].set_title("constrained log-density marginal"); axes[0, 1].legend()

    ebc = 0.5 * (eb_edges[:-1] + eb_edges[1:])
    axes[1, 0].plot(ebc, ref_eb_pmf, label="Gibbs ref")
    axes[1, 0].plot(ebc, dnfs_eb_pmf, label="DNFS IS-weighted")
    axes[1, 0].set_xlabel(r"base $\log p_{\rm Ising}(x)$ (penalty stripped)")
    axes[1, 0].set_ylabel("probability")
    axes[1, 0].set_title("pure-Ising log-density marginal"); axes[1, 0].legend()

    site_axis = torch.arange(N_SITES)
    axes[1, 1].plot(site_axis, site_p_ref, label="Gibbs ref", lw=1)
    axes[1, 1].plot(site_axis, site_p_dnfs, label="DNFS IS-weighted", lw=1)
    axes[1, 1].axhline(cfg["target_composition"], ls="--", c="k", lw=1, label="c_target")
    axes[1, 1].set_xlabel("site index $i$")
    axes[1, 1].set_ylabel(r"$P(x_i = +1)$")
    axes[1, 1].set_title("per-site +1 probability (translation-symmetry check)")
    axes[1, 1].legend()

    fig.tight_layout()
    out_png = run_dir / "d10_c03_gibbs_fidelity.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nsaved figure to {out_png}")


if __name__ == "__main__":
    main()
