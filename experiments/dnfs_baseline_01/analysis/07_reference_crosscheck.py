"""Cross-check of the 10x10 Gibbs reference pools against Wolff + exact values.

The baseline chapter's ground truth is 100 pooled Gibbs heat-bath chains per
operating point. At sigma_c single-site dynamics has dynamic exponent z ~ 2.17
(critical slowing-down), so the reference itself is the thing under audit
here, by two independent instruments:

  1. Wolff single-cluster pools (z ~ 0.25, tunnels the Z2 sectors freely) at
     matched sample count — the sample-level check on the full energy and
     |M| marginals, not just means;
  2. the exact Kaufman / Ferdinand--Fisher internal energy at the OPERATING
     coupling (ising_exact; note DNFS Table 2's critical column is evaluated
     at exact criticality 0.220343, not at 0.22305 — the s58 finding — so
     the exact anchor here is recomputed, not quoted).

Reads: results/01_baseline/gibbs_ref_d10_sigma{0.1,0.22305}.pt.
Writes: results/01_baseline/reference_crosscheck.json + printed verdict.

Verdict rule (restart-prompt item (e)): agreement -> the Gibbs reference
stands and baseline.tex gains one sentence; disagreement -> Wolff becomes
the reference. POSTSCRIPT (s58, same day): the verdict was AGREE at both
points, and the user then chose the swap anyway -- not as a correction but
because the Wolff pool's floor is honest where the Gibbs pool's dMag floor
at sigma_c (0.26) is mode-stickiness. 08_wolff_reference_pool.py builds the
pools; this script remains the certification record. "Agreement" is read per observable: |mean difference| within
3 combined standard errors, and the energy-level total variation within the
same-size Wolff-vs-Wolff resampling scale.
"""
import json
from pathlib import Path

import numpy as np
import torch

from discrete_flow_sampler.mcmc.wolff import wolff_sample
from discrete_flow_sampler.targets.ising import IsingTarget
from discrete_flow_sampler.targets.ising_exact import ferdinand_fisher_per_site

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "01_baseline"
L = 10
OPERATING_POINTS = {
    "sigma_0.1": dict(sigma=0.1, reference="gibbs_ref_d10_sigma0.1.pt"),
    "sigma_c": dict(sigma=0.22305, reference="gibbs_ref_d10_sigma0.22305.pt"),
}
WOLFF_SEED = 20260824


def energy_per_site(samples: torch.Tensor, target: IsingTarget) -> torch.Tensor:
    return -target.log_prob(samples) / (2 * target.sigma * target.d)


def moments(values: torch.Tensor) -> tuple[float, float]:
    """(mean, naive SE). For the Gibbs pool the chain, not the sample, is the
    independent unit, so its SE is widened by the chain count downstream."""
    return values.mean().item(), values.std().item() / len(values) ** 0.5


def energy_level_tv(a: torch.Tensor, b: torch.Tensor) -> float:
    """Total variation between empirical pmfs on the exact energy levels
    (E/d is a lattice of discrete values; no binning choices needed)."""
    levels = torch.cat([a, b]).unique()
    pmf_a = torch.tensor([(a == level).float().mean() for level in levels])
    pmf_b = torch.tensor([(b == level).float().mean() for level in levels])
    return 0.5 * (pmf_a - pmf_b).abs().sum().item()


def main() -> None:
    report = {}
    for label, spec in OPERATING_POINTS.items():
        target = IsingTarget(D=L, sigma=spec["sigma"], bias=0.0, device="cpu")
        ref = torch.load(RESULTS / spec["reference"], weights_only=False)
        gibbs, n_chains = ref["samples"].float(), ref["n_chains"]
        n = gibbs.shape[0]

        wolff = wolff_sample(target, n_samples=n, seed=WOLFF_SEED)
        wolff_b = wolff_sample(target, n_samples=n, seed=WOLFF_SEED + 1)

        exact_energy = ferdinand_fisher_per_site(D=L, sigma=spec["sigma"])[
            "internal_energy"
        ]
        e_gibbs, se_g = moments(energy_per_site(gibbs, target))
        e_wolff, se_w = moments(energy_per_site(wolff, target))
        m_gibbs, sm_g = moments(gibbs.sum(dim=1).abs() / target.d)
        m_wolff, sm_w = moments(wolff.sum(dim=1).abs() / target.d)
        # Gibbs draws within a chain are correlated: scale its SE by
        # sqrt(records-per-chain) as the conservative chain-unit bound.
        records_per_chain = n // n_chains
        se_g *= records_per_chain**0.5
        sm_g *= records_per_chain**0.5

        tv_wolff_gibbs = energy_level_tv(
            energy_per_site(gibbs, target), energy_per_site(wolff, target)
        )
        tv_wolff_wolff = energy_level_tv(
            energy_per_site(wolff, target), energy_per_site(wolff_b, target)
        )

        row = {
            "exact_E": exact_energy,
            "gibbs_E": e_gibbs, "gibbs_E_se": se_g,
            "wolff_E": e_wolff, "wolff_E_se": se_w,
            "gibbs_absM": m_gibbs, "gibbs_absM_se": sm_g,
            "wolff_absM": m_wolff, "wolff_absM_se": sm_w,
            "tv_energy_wolff_vs_gibbs": tv_wolff_gibbs,
            "tv_energy_wolff_vs_wolff": tv_wolff_wolff,
            "agree_E": abs(e_gibbs - e_wolff) < 3 * (se_g**2 + se_w**2) ** 0.5,
            "agree_absM": abs(m_gibbs - m_wolff)
            < 3 * (sm_g**2 + sm_w**2) ** 0.5,
            "agree_tv": tv_wolff_gibbs < 3 * max(tv_wolff_wolff, 1e-3),
            "wolff_E_vs_exact_sigmas": abs(e_wolff - exact_energy) / se_w,
            "gibbs_E_vs_exact_sigmas": abs(e_gibbs - exact_energy) / se_g,
        }
        report[label] = row

        print(f"\n== {label} (sigma={spec['sigma']}, N={n}) ==")
        print(f"  exact E/d            {exact_energy:+.4f}")
        print(f"  Gibbs E/d            {e_gibbs:+.4f} +- {se_g:.4f}"
              f"  ({row['gibbs_E_vs_exact_sigmas']:.1f} SE from exact)")
        print(f"  Wolff E/d            {e_wolff:+.4f} +- {se_w:.4f}"
              f"  ({row['wolff_E_vs_exact_sigmas']:.1f} SE from exact)")
        print(f"  Gibbs |M|/d          {m_gibbs:.4f} +- {sm_g:.4f}")
        print(f"  Wolff |M|/d          {m_wolff:.4f} +- {sm_w:.4f}")
        print(f"  TV(E) Wolff-Gibbs    {tv_wolff_gibbs:.4f}"
              f"   [Wolff-Wolff scale {tv_wolff_wolff:.4f}]")
        verdicts = [k for k in ("agree_E", "agree_absM", "agree_tv") if not row[k]]
        print(f"  verdict: {'AGREE' if not verdicts else 'DISAGREE on ' + ', '.join(verdicts)}")

    out = RESULTS / "reference_crosscheck.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
