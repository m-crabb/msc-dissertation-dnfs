"""Annealed-start robustness check for the §5.1 Kawasaki failure demos.

The dissertation's neural sampler reaches sigma_c through a training-time
sigma-curriculum, which the writeup itself frames as "Kawasaki's tempering
paid once". Fairness therefore demands the mirror-image control: give the
Kawasaki chains the same courtesy — a simulated-annealing initialisation
that walks a sigma-ladder up to the target before anything is measured —
and ask whether either §5.1 failure demo was a cold-start artefact.

Design and expected outcomes:

  tau_int comparison at the measurement couplings, D in {10, 16, 24}: the
     annealed arm should land inside the cold arm's seed band. tau_int is a property
     of the stationary dynamics; annealing can only repair burn-in, and the
     cold protocol already spends 200k steps of burn-in. A genuine shift
     here would mean the printed slowing-down curves carry an
     initialisation artefact and must be corrected.
  R-hat(phi) and mode occupancy at D=24, sigma in {0.32, 0.40}: annealed chains start
     DISORDERED (random at the target composition) and condense into a
     phase-separated mode during the anneal. If each chain freezes into
     whichever mode it condensed into, either R-hat stays >> 1 (chains
     disagree) or — the sharper outcome — all chains condense into the
     SAME mode and R-hat looks healthy while half the target's mass is
     silently missing, exactly the coverage trap hard.tex Section 5.1
     warns that slowing-down statistics can hide. The demo is weakened
     only if annealed chains genuinely CROSS between modes: R-hat near 1
     AND both modes visited within each chain.

The anneal ladder mirrors the training curriculum's rungs (0.10 to 0.223,
the d64 recipe) and extends by steps of at most 0.04 when the target lies
beyond sigma_c. Each rung dwells for the cold protocol's full burn-in
budget (200k swap steps), so the annealed arm is strictly more generous
than the cold one. Parallel tempering is deliberately NOT used here: it is
an orthogonal wrapper that would equally accelerate the neural sampler, so
it belongs to neither side of the comparison.

Run:  pixi run python -m scripts.kawasaki_annealed_check
      pixi run python -m scripts.kawasaki_annealed_check full_curve
"""
import json
import sys
from pathlib import Path

import numpy as np

from discrete_flow_sampler.diagnostics.metrics import (
    gelman_rubin,
    integrated_autocorr,
)
from discrete_flow_sampler.mcmc.kawasaki import (
    init_random_at_composition,
    run_chain,
    run_chain_order_param,
)

OUT = Path("results/kawasaki/annealed_check")
OUT.mkdir(parents=True, exist_ok=True)

from discrete_flow_sampler.targets.ising import SIGMA_C

SIGMA_CRITICAL = SIGMA_C  # the exact critical coupling
# The training curriculum's rungs, then <=0.04 extensions past sigma_c.
CURRICULUM_RUNGS = [0.100, 0.140, 0.170, 0.190, 0.205, 0.215, SIGMA_CRITICAL]
DWELL_STEPS = 200_000  # per rung == the cold protocol's whole burn-in

# The tau_int comparison mirrors scripts/kawasaki_sweep.py failure_curves(): same step
# budget, burn, thinning and seed base, so the only difference is the init.
TAU_D = [10, 16, 24]
TAU_SIGMAS = [SIGMA_CRITICAL, 0.26]
TAU_STEPS, TAU_BURN, TAU_THIN, TAU_CHAINS, TAU_SEED = 1_500_000, 200_000, 50, 8, 100

# The R-hat / mode-occupancy check mirrors mode_coverage(): same budgets, same
# seed ensembles.
ERGO_D = 24
ERGO_SIGMAS = [0.10, 0.32, 0.40]  # 0.10 = the built-in positive control
ERGO_STEPS, ERGO_BURN, ERGO_THIN = 1_500_000, 200_000, 200
ERGO_SEEDS = [300, 400, 500, 600]


def anneal_ladder(sigma_target):
    """Curriculum rungs up to sigma_c, then steps of <=0.04 to the target."""
    ladder = [s for s in CURRICULUM_RUNGS if s < sigma_target - 1e-9]
    extension = ladder[-1] if ladder else 0.0
    while sigma_target - extension > 1e-9:
        extension = min(extension + 0.04, sigma_target)
        ladder.append(extension)
    return ladder


def annealed_init(D, sigma_target, seed, dwell_steps=DWELL_STEPS):
    """Random composition-0.5 start, walked up the ladder one dwell per rung.

    Reuses the scalar-sigma numba kernel per rung with the configuration
    carried forward; the composition assert catches any conservation bug
    the chaining could hide. `dwell_steps` is the per-rung budget in raw swap
    attempts; it is a parameter rather than the module constant because the
    failure-curve figure runs a d-scaled protocol (a fixed number of SWEEPS at
    every lattice size), and a fixed raw-step dwell would anneal a 32x32
    lattice sixteen times less thoroughly than an 8x8 one."""
    d = D * D
    rng = np.random.default_rng(seed)
    x = init_random_at_composition(d, 0.5, rng)
    n_plus0 = int((x == 1).sum())
    for rung_index, sigma in enumerate(anneal_ladder(sigma_target)):
        _, x, _ = run_chain(x, D, sigma, dwell_steps, seed * 1000 + rung_index)
        assert int((x == 1).sum()) == n_plus0, "composition not conserved!"
    return x


def _tau_row(D, sigma, init_kind):
    """One (D, sigma, init) tau_int cell: TAU_CHAINS chains, tau_int in sweeps."""
    d = D * D
    taus = []
    for chain in range(TAU_CHAINS):
        seed = TAU_SEED + chain
        if init_kind == "cold":
            x = init_random_at_composition(d, 0.5, np.random.default_rng(seed))
        else:
            x = annealed_init(D, sigma, seed)
        energy, x_final, _ = run_chain(x, D, sigma, TAU_STEPS, seed)
        assert int((x_final == 1).sum()) == d // 2
        kept = energy[TAU_BURN::TAU_THIN]
        taus.append(integrated_autocorr(kept) * TAU_THIN / d)
    taus = np.array(taus)  # in sweeps, as printed in hard.tex
    row = {
        "D": D, "sigma": sigma, "init": init_kind,
        "tau_int_sweeps_mean": float(taus.mean()),
        "tau_int_sweeps_std": float(taus.std(ddof=1)),
    }
    print(f"[tau] D={D} sigma={sigma} {init_kind}: "
          f"{taus.mean():.2f} +/- {taus.std(ddof=1):.2f} sweeps")
    return row


def tau_arm():
    """Cold vs annealed tau_int under the identical measurement."""
    return [_tau_row(D, sigma, init_kind)
            for D in TAU_D
            for sigma in TAU_SIGMAS
            for init_kind in ("cold", "annealed")]


def full_curve_arm():
    """Annealed tau_int at every sigma failure_curves() plots.

    The spot check above answers the fairness question in prose; the printed
    figure plots the full CURVE_SIGMAS grid, so for the chapter's opening
    figure to preempt the cold-start question itself it needs an annealed
    marker at every plotted sigma. The protocol is identical to tau_arm (same
    budgets, thinning and seed base), so (D, sigma) cells already measured in
    an archived summary.json are reused rather than recomputed: a recompute
    under an identical protocol could only add noise, and the archived numbers
    are the ones the prose already quotes.
    """
    prior = {}
    summary_path = OUT / "summary.json"
    if summary_path.exists():
        for row in json.loads(summary_path.read_text())["tau_arm"]:
            if row["init"] == "annealed":
                prior[(row["D"], round(row["sigma"], 5))] = row
    from scripts.kawasaki_sweep import CURVE_SIGMAS  # the figure's sigma grid
    rows = []
    for D in TAU_D:
        for sigma in CURVE_SIGMAS:
            cached = prior.get((D, round(sigma, 5)))
            if cached is not None:
                print(f"[tau] D={D} sigma={sigma} annealed: reused archived row")
                rows.append(cached)
            else:
                rows.append(_tau_row(D, sigma, "annealed"))
    payload = {
        "design": "annealed-start tau_int across the failure-curve sigma grid",
        "dwell_steps_per_rung": DWELL_STEPS,
        "rows": rows,
    }
    out_path = OUT / "tau_full_curve.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


def ergodicity_arm():
    """R-hat(phi) + per-chain mode occupancy for annealed starts."""
    rows = []
    for sigma in ERGO_SIGMAS:
        rhats, mode_signs = [], []
        for base_seed in ERGO_SEEDS:
            post_traces, signs = [], []
            for chain in range(4):
                seed = base_seed + chain
                x = annealed_init(ERGO_D, sigma, seed)
                phi, x_final, _ = run_chain_order_param(
                    x, ERGO_D, sigma, ERGO_STEPS, seed, ERGO_THIN
                )
                post = phi[ERGO_BURN // ERGO_THIN:]
                post_traces.append(post)
                # Which mode the chain occupies, and whether it ever crosses:
                # fraction of post-burn samples on the positive-phi side.
                signs.append(float((post > 0).mean()))
            rhats.append(float(gelman_rubin(np.stack(post_traces))))
            mode_signs.append(signs)
            print(f"[ergo] sigma={sigma} seed={base_seed}: "
                  f"R-hat={rhats[-1]:.2f} positive-side fractions={signs}")
        rows.append({
            "D": ERGO_D, "sigma": sigma,
            "rhat_mean": float(np.mean(rhats)),
            "rhat_std": float(np.std(rhats, ddof=1)),
            "rhat_by_seed": rhats,
            # ~0 or ~1 = chain pinned in one mode; ~0.5 = genuine crossing.
            "positive_side_fraction_by_seed_chain": mode_signs,
        })
    return rows


def main():
    if "full_curve" in sys.argv[1:]:
        full_curve_arm()
        return
    summary = {
        "design": "annealed-start robustness check for the section 5.1 demos",
        "dwell_steps_per_rung": DWELL_STEPS,
        "ladders": {str(s): anneal_ladder(s)
                    for s in set(TAU_SIGMAS) | set(ERGO_SIGMAS)},
        "tau_arm": tau_arm(),
        "ergodicity_arm": ergodicity_arm(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
