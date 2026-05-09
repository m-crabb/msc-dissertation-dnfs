"""Long Gibbs chain at D=10, σ=0.1 — reference samples for Stage 1's d10 verdict.

Generates 5000 independent final-state samples via single-spin-flip heat-bath,
used as the ground-truth for the energy-histogram comparison vs DNFS samples
(the headline verdict figure for Stage 1's D=10 sampler claim).

Mixing is verified two ways before the canonical samples are saved:

  1. Chain-mean log p̃(x) trace across sweeps — should plateau well before
     n_sweeps. If it doesn't, the final state still carries memory of the
     random init and the samples are biased.

  2. Initial-condition independence — re-run from all-up init, check that
     the final-state log p̃ histograms agree with the random-init run. If
     they disagree, the chain hasn't fully mixed regardless of what (1) says.

Saves:
  - results/02_baseline/gibbs_chain_d10_sigma01.pt    (canonical samples)
  - results/02_baseline/stage_1_d10_gibbs_mixing.png  (diagnostic plot)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.mcmc.gibbs import gibbs_sample
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "02_baseline"
N_CHAINS = 5000
N_SWEEPS = 1000
RECORD_EVERY = 10


def main() -> None:
    target = IsingTarget(D=10, sigma=0.1)

    # Run 1 — random init (the canonical reference run).
    torch.manual_seed(0)
    samples_random, trace_random = gibbs_sample(
        target,
        n_chains=N_CHAINS,
        n_sweeps=N_SWEEPS,
        record_energy_every=RECORD_EVERY,
    )

    # Run 2 — all-up init (initial-condition independence check).
    torch.manual_seed(1)
    x_up = torch.ones(N_CHAINS, target.d)
    samples_up, trace_up = gibbs_sample(
        target,
        n_chains=N_CHAINS,
        n_sweeps=N_SWEEPS,
        x_init=x_up,
        record_energy_every=RECORD_EVERY,
    )

    sweep_axis = torch.arange(trace_random.shape[0]) * RECORD_EVERY

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(sweep_axis, trace_random.mean(-1), label="random init")
    axes[0].plot(sweep_axis, trace_up.mean(-1), label="all-up init")
    axes[0].set_xlabel("sweep")
    axes[0].set_ylabel(r"$\langle \log \tilde p(x) \rangle$ across chains")
    axes[0].set_title("Chain-mean energy trace (mixing diagnostic 1)")
    axes[0].legend()

    final_lp_random = target.log_prob(samples_random)
    final_lp_up = target.log_prob(samples_up)
    lp_min = torch.minimum(final_lp_random.min(), final_lp_up.min()).item()
    lp_max = torch.maximum(final_lp_random.max(), final_lp_up.max()).item()
    bins = torch.linspace(lp_min, lp_max, 50).numpy()
    axes[1].hist(final_lp_random.numpy(), bins=bins, alpha=0.5, label="random init")
    axes[1].hist(final_lp_up.numpy(), bins=bins, alpha=0.5, label="all-up init")
    axes[1].set_xlabel(r"$\log \tilde p(x)$")
    axes[1].set_ylabel("count")
    axes[1].set_title("Final-state log p̃ (mixing diagnostic 2: init agreement)")
    axes[1].legend()

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / "stage_1_d10_gibbs_mixing.png"
    fig.savefig(plot_path, dpi=120)
    print(f"saved mixing diagnostic plot to {plot_path}")

    samples_path = OUT_DIR / "gibbs_chain_d10_sigma01.pt"
    torch.save(
        {
            "samples": samples_random,
            "n_chains": N_CHAINS,
            "n_sweeps": N_SWEEPS,
            "sigma": 0.1,
            "D": 10,
            "seed": 0,
            "init": "random",
        },
        samples_path,
    )
    print(f"saved {N_CHAINS} reference samples to {samples_path}")

    mean_lp_random = final_lp_random.mean().item()
    mean_lp_up = final_lp_up.mean().item()
    print()
    print("=== mixing summary ===")
    print(f"  ⟨log p̃⟩ random init:   {mean_lp_random:+.4f}")
    print(f"  ⟨log p̃⟩ all-up init:    {mean_lp_up:+.4f}")
    print(f"  abs diff:                {abs(mean_lp_random - mean_lp_up):.4f}")
    print("  trace plateau (last 20% sweep window, random init):")
    plateau_window = trace_random[-trace_random.shape[0] // 5:].mean(-1)
    print(f"    mean {plateau_window.mean().item():+.4f}, "
          f"std across window {plateau_window.std().item():.4f}")


if __name__ == "__main__":
    main()
