"""Fill pass for tab:eval-soft-4x4: the soft house recipe against exact
enumeration at the one size where the target is summable (2^16 states).

Mirrors hard's enumerable gate: every column is scored against ground
truth, never against a reference chain.

  ESS         -- frozen-eval ESS fraction (metrics.json).
  TV(c)       -- total variation between the importance-weighted
                 composition marginal and the exact one on the 17 support
                 points k/16.
  Z2 split    -- |mass(m>0) - mass(m<0)| under the weights; the target at
                 c*=0.5, bias 0 is exactly Z2-symmetric so the truth is 0.
                 Off-centre the symmetry maps c* onto 1-c*, so the column
                 is not applicable there.
  std(c)      -- weighted composition SD vs the analytic 1/sqrt(2 lambda d)
                 (0.0177 at lambda=50, 0.0125 at lambda=100, d=16). NB the
                 Gaussian width is the ENVELOPE; the exact marginal's own
                 SD is also printed, because at d=16 the discrete entropic
                 factor tilts it visibly off the envelope.
  dF/site     -- IS free-energy estimate -<log w>/(2 sigma d) minus the
                 exact -log Z/(2 sigma d), in nats per site (Eq. 37 lower
                 bound, so the sign is expected non-negative up to noise).

Families: the softhouse-d16 specialists (c* in {0.25, 0.375, 0.5},
lambda=50) and the lambda=100 fidelity row (c*=0.5, the efc-sweep family:
channel recipe pre-house, the only 4x4 lambda=100 cells with the channel).
"""
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from discrete_flow_sampler.diagnostics.metrics import (  # noqa: E402
    enumerate_states, exact_log_probs, free_energy_lb_estimate,
    marginal_tvd, z2_asymmetry_from_samples)
from discrete_flow_sampler.targets.ising import IsingTarget  # noqa: E402

SOFT_RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
N_SITES = 16
FAMILIES = {
    "c0.25_l50": "S2_d4_c0250_50k_l50_letf_house_seed4*_20260831-softhouse-d16",
    "c0.375_l50": "S2_d4_c0375_50k_l50_letf_house_seed4*_20260831-softhouse-d16",
    "c0.5_l50": "S2_d4_c0500_50k_l50_letf_house_seed4*_20260831-softhouse-d16",
    "c0.5_l100": "S2_d4_c05_l100_letf_efc_seed4*_20260829-efc-sweep",
}


def composition_pmf(c_values, weights):
    bucket = (c_values * N_SITES).round().long().clamp(0, N_SITES)
    pmf = torch.zeros(N_SITES + 1)
    pmf.index_add_(0, bucket, weights)
    return pmf


def exact_reference(target):
    states = enumerate_states(N_SITES).float()
    pi = exact_log_probs(target, states).exp()
    c_states = (states > 0).float().mean(1)
    pmf = composition_pmf(c_states, pi)
    mean_c = (pi * c_states).sum()
    std_c = ((pi * (c_states - mean_c) ** 2).sum()).sqrt().item()
    log_Z = torch.logsumexp(target.log_prob(states), 0)
    return pmf, std_c, (-log_Z / (2 * target.sigma * N_SITES)).item()


def score_run(run_dir: Path, target, exact_pmf, exact_f):
    samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    w = torch.softmax(log_w, 0)
    c = (samples > 0).float().mean(1)
    mean_c = (w * c).sum()
    return {
        "ESS": metrics["ess_fraction"],
        "TV_c": marginal_tvd(composition_pmf(c, w), exact_pmf),
        "Z2_split": z2_asymmetry_from_samples(samples, log_w)["asymmetry"],
        "std_c": ((w * (c - mean_c) ** 2).sum()).sqrt().item(),
        "dF_site": free_energy_lb_estimate(log_w, target.sigma, N_SITES).item() - exact_f,
    }


def mean_sd(values):
    t = torch.tensor(values)
    return t.mean().item(), (t.std().item() if len(t) > 1 else float("nan"))


def main():
    table = {}
    for key, glob in FAMILIES.items():
        run_dirs = sorted(SOFT_RESULTS.glob(glob))
        if not run_dirs:
            print(f"== {key}: no runs match {glob}")
            continue
        cfg = json.loads((run_dirs[0] / "config.json").read_text())["ising"]
        target = IsingTarget(
            D=cfg["D"], sigma=cfg["sigma"], bias=cfg["bias"],
            target_composition=cfg["target_composition"],
            composition_penalty_strength=cfg["composition_penalty_strength"])
        exact_pmf, exact_std, exact_f = exact_reference(target)
        analytic_std = 1 / math.sqrt(2 * cfg["composition_penalty_strength"] * N_SITES)
        per_seed = {d.name: score_run(d, target, exact_pmf, exact_f) for d in run_dirs}
        summary = {k: mean_sd([s[k] for s in per_seed.values()]) for k in next(iter(per_seed.values()))}
        table[key] = {"exact_std_c": exact_std, "analytic_std_c": analytic_std,
                      "exact_F_site": exact_f, "per_seed": per_seed, "mean_sd": summary}
        print(f"\n== {key} ({len(run_dirs)} seeds; exact std(c) {exact_std:.4f}, "
              f"analytic {analytic_std:.4f}, exact F/site {exact_f:.4f})")
        for name, s in per_seed.items():
            print("   " + name.split("_seed")[1][:2] + ": " + " ".join(f"{k}={v:.4g}" for k, v in s.items()))
        print("   mean+-SD: " + " ".join(f"{k}={m:.4g}+-{sd:.2g}" for k, (m, sd) in summary.items()))
    out = SOFT_RESULTS / "enumerable_check_4x4.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
