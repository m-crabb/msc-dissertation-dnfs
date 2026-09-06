"""Fill pass for tab:eval-soft-4x4: the soft house table at the enumerable
size, in the HOUSE layout (columns as tab:eval-hard-4x4 / the 10x10 house
tables; the s108 first cut with TV / Z2 / std(c) / dF columns is retired).

  reference -- exact enumeration: all 2^16 states weighted by the soft
               target's own normalised probabilities (the soft target is
               unconstrained, so no slice); the error columns read against
               truth and the reference row is exactly zero.
  floor     -- the error a PERFECT sampler shows at the neural cells' own
               draw count (5000 exact multinomial draws from the enumerated
               target, 200 replicates): a cell at or below it is
               indistinguishable from exact at its N. (hard's house_table_4x4
               convention; the 10x10 fills bootstrap a sampled reference
               instead because there the reference is itself sampled.)
  cells     -- lambda=50 house specialists at c* in {0.25, 0.375, 0.5}
               at the cross-chapter 4x4 budget of 10k steps (the _10k_
               family, tag softhouse-d16-10k; the _50k_ family is the
               amortised comparator set) and the lambda=100 centre cell (efc-sweep:
               the channel recipe pre-house, the only 4x4 lambda=100 cells
               with the channel); every seed reported, mean +- SD.
  FLOP/es   -- measured eager forward at the run's architecture x n_euler
               / frozen ESS, as the 8x8 fill (house_table_soft_8x8.py).
  couplings -- sigma=0.1 and sigma_c halves; a half with no runs on disk
               prints as skipped.
  conditioned -- the 10k conditioned twin (matched base, spine draw over
               the three windows) scored at each window from the frames its
               composition sweep filed there, so the row beside a specialist
               is the same model asked for that specialist's composition;
               skipped at a window until the sweep has run.

Extras kept in the JSON for the comments only: delivered std(c) vs the
ENUMERATED spread (not the Gaussian envelope 1/sqrt(2 lambda d), which at
d=16 is wider than the composition step and so is not the marginal), and
the IS free-energy bias vs the exact -log Z/(2 sigma d) (the Euler-grid
bias of the weight integral at n_euler=50, drive-proportional).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from discrete_flow_sampler.diagnostics.flops import (  # noqa: E402
    measured_forward_flops, neural_sampling_flops_per_sample,
    per_effective_sample)
from discrete_flow_sampler.diagnostics.metrics import (  # noqa: E402
    correlation_profile_error, energy_wasserstein2, enumerate_states,
    exact_log_probs, free_energy_lb_estimate, magnetisation_profile_error)
from discrete_flow_sampler.models.composition_conditioned import (  # noqa: E402
    CompositionConditioned)
from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget  # noqa: E402

SOFT_RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
L, D_SITES, N_DRAWS, N_BOOTSTRAP = 4, 16, 5000, 200
COUPLINGS = (("s010", 0.1), ("sc", SIGMA_C))
# family key -> (c*, lambda, run glob); "{sc}" takes "" or "_sc"
FAMILIES = {
    "c0.25_l50": (0.25, 50.0, "S2_d4_c0250_10k_l50_letf_house{sc}_seed4*"),
    "c0.375_l50": (0.375, 50.0, "S2_d4_c0375_10k_l50_letf_house{sc}_seed4*"),
    "c0.5_l50": (0.5, 50.0, "S2_d4_c0500_10k_l50_letf_house{sc}_seed4*"),
    "c0.5_l100": (0.5, 100.0, "S2_d4_c05_l100_letf_efc{sc}_seed4*"),
}
CONDITIONED = "S2_d4_camort_10k_l50_letf_house{sc}_seed4*"


def energy_per_site(target, states):
    return -target.base_log_prob(states) / (2 * target.sigma * D_SITES)


def errors(target, x, weights, ref_states, ref_probs):
    return {
        "dMag": magnetisation_profile_error(
            x, weights, ref_states, L, reference_weights=ref_probs),
        "dCorr": correlation_profile_error(
            x, weights, ref_states, L, reference_weights=ref_probs),
        "EW2": energy_wasserstein2(
            energy_per_site(target, x), weights,
            energy_per_site(target, ref_states), reference_weights=ref_probs),
    }


def sampling_floor(target, ref_states, ref_probs, seed=0):
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.full((N_DRAWS,), 1.0 / N_DRAWS)
    replicates = []
    for _ in range(N_BOOTSTRAP):
        idx = torch.multinomial(ref_probs, N_DRAWS, replacement=True,
                                generator=generator)
        replicates.append(errors(target, ref_states[idx], uniform,
                                 ref_states, ref_probs))
    return {k: sum(r[k] for r in replicates) / N_BOOTSTRAP
            for k in replicates[0]}


def flops_per_forward(run_dir, target, composition):
    from experiments.dnfs_baseline_01.configs import ModelCfg
    from experiments.dnfs_baseline_01.run import _construct_model, _sub_config
    cfg = json.loads((run_dir / "config.json").read_text())
    model_cfg = _sub_config(ModelCfg, {**cfg["model"], "compile_model": False})
    model = _construct_model(SimpleNamespace(model=model_cfg), target)
    if model_cfg.condition_on_composition:
        # The conditioned forward reads c alongside (x, t); bind it exactly
        # as the sweep does so the counted forward is the one the row paid.
        model = CompositionConditioned(model, torch.full((1,), composition))
    return measured_forward_flops(
        model, (target.sample_base(1, device="cpu"), torch.zeros(1)))


def load_frames(run_dir, composition=None):
    """The frozen eval's frames and ESS, or -- for a conditioned run -- the
    frames its sweep filed at `composition` with that row's own ESS."""
    if composition is None:
        eval_dir = run_dir / "eval"
        ess = json.loads((eval_dir / "metrics.json").read_text())["ess_fraction"]
    else:
        rows = json.loads((run_dir / "eval" / "composition_sweep.json").read_text())
        ess = next(r["ess_fraction"] for r in rows
                   if abs(r["composition"] - composition) < 1e-6)
        eval_dir = run_dir / "eval" / "composition_sweep" / f"c{composition:.4f}"
    x = torch.load(eval_dir / "samples.pt", weights_only=True).float()
    log_w = torch.load(eval_dir / "log_weights.pt", weights_only=True)
    return ess, x, log_w


def score_run(run_dir, target, ref_states, ref_probs, per_forward,
              composition=None):
    ess, x, log_w = load_frames(run_dir, composition)
    w = torch.softmax(log_w, 0)
    n_euler = json.loads(
        (run_dir / "config.json").read_text())["ctmc"]["n_euler_steps"]
    c = (x > 0).float().mean(1)
    mean_c = (w * c).sum()
    exact_f = -torch.logsumexp(target.log_prob(ref_states), 0) / (2 * target.sigma * D_SITES)
    return {
        "ESS": ess,
        **errors(target, x, w, ref_states, ref_probs),
        "FLOPes": per_effective_sample(
            neural_sampling_flops_per_sample(per_forward, n_euler, D_SITES),
            ess),
        "std_c": ((w * (c - mean_c) ** 2).sum()).sqrt().item(),
        "dF_site": (free_energy_lb_estimate(log_w, target.sigma, D_SITES) - exact_f).item(),
    }


def mean_sd(values):
    t = torch.tensor(values, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if len(t) > 1 else float("nan"))


def score_family(label, run_dirs, target, all_states, ref_probs, floor,
                 exact_std_c, composition=None):
    """Score one family's seeds (specialist eval, or the conditioned sweep
    row at `composition`), print the per-seed lines, return the table cell."""
    per_forward = flops_per_forward(run_dirs[0], target, composition)
    per_seed = {d.name: score_run(d, target, all_states, ref_probs, per_forward,
                                  composition)
                for d in run_dirs}
    summary = {k: mean_sd([s[k] for s in per_seed.values()])
               for k in next(iter(per_seed.values()))}
    print(f"\n== {label} ({len(run_dirs)} seeds) floor "
          + " ".join(f"{k}={v * 100:.1f}" for k, v in floor.items())
          + f"  exact std(c) {exact_std_c:.4f}")
    for name, s in per_seed.items():
        print("   seed" + name.split("_seed")[1][:2] + ": "
              + f"ESS={s['ESS']:.3f} " + " ".join(
                  f"{k}={s[k] * 100:.1f}" for k in ("dMag", "dCorr", "EW2"))
              + f" FLOP/es={s['FLOPes']:.2g} std_c={s['std_c']:.4f} dF={s['dF_site']:+.3f}")
    m = summary
    print("   mean+-SD: " + f"ESS {m['ESS'][0]:.3f}+-{m['ESS'][1]:.3f} "
          + " ".join(f"{k} {m[k][0] * 100:.1f}+-{m[k][1] * 100:.1f}"
                     for k in ("dMag", "dCorr", "EW2"))
          + f" FLOP/es {m['FLOPes'][0]:.2g}")
    return {"floor": floor, "exact_std_c": exact_std_c,
            "per_seed": per_seed, "mean_sd": summary}


def main():
    table = {}
    all_states = enumerate_states(D_SITES).float()
    for sigma_label, sigma in COUPLINGS:
        suffix = "_sc" if sigma_label == "sc" else ""
        for key, (c_target, lam, glob) in FAMILIES.items():
            run_dirs = sorted(SOFT_RESULTS.glob(glob.format(sc=suffix)))
            if not run_dirs:
                print(f"== {sigma_label} {key}: SKIPPED (no runs match {glob.format(sc=suffix)})")
                continue
            target = IsingTarget(D=L, sigma=sigma, bias=0.0,
                                 target_composition=c_target,
                                 composition_penalty_strength=lam)
            ref_probs = exact_log_probs(target, all_states).exp()
            c_states = (all_states > 0).float().mean(1)
            exact_mean_c = (ref_probs * c_states).sum()
            exact_std_c = ((ref_probs * (c_states - exact_mean_c) ** 2).sum()).sqrt().item()
            floor = sampling_floor(target, all_states, ref_probs)
            table[f"{sigma_label}_{key}"] = score_family(
                f"{sigma_label} {key}", run_dirs, target, all_states,
                ref_probs, floor, exact_std_c)
            if lam != 50.0:
                continue
            frame_leaf = Path("eval") / "composition_sweep" / f"c{c_target:.4f}" / "samples.pt"
            cond_dirs = [d for d in sorted(SOFT_RESULTS.glob(CONDITIONED.format(sc=suffix)))
                         if (d / frame_leaf).exists()]
            if not cond_dirs:
                print(f"== {sigma_label} {key} conditioned: SKIPPED (no sweep frames at c={c_target})")
                continue
            table[f"{sigma_label}_{key}_conditioned"] = score_family(
                f"{sigma_label} {key} conditioned", cond_dirs, target,
                all_states, ref_probs, floor, exact_std_c, composition=c_target)
    out = SOFT_RESULTS / "enumerable_check_4x4.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
