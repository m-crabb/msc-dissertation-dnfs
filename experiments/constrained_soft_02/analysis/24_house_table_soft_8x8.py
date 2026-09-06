"""Fill pass for tab:eval-soft-8x8 (the s95 revamped soft house table).

Same conventions as the 10x10 fill (house_table_soft_10x10.py), rebuilt for
the revamp (plan 2026-08-30-soft-chapter-revamp-efc):

  lattice   -- 8x8 (d=64), the hard chapter's record size, so the
               cross-route comparison is matched-size at both couplings.
  couplings -- sigma=0.1 AND SIGMA_C, the table's two halves. A missing
               reference or empty cell prints and skips rather than
               failing the fill: the table must be reviewable while cells
               are still landing (hard's house-fill behaviour).
  families  -- the house specialists (S2_d8_*_l50_letf_ne128_house{_sc})
               and, at the critical centre only, the nochan control (house
               recipe minus the channel: the one measured channel-off/on
               comparison at sigma_c, pinned one-lever by
               tests/test_soft_house_configs.py). Two s99/s100 arms join
               when their runs are on disk: the matched-base twins (`mb`,
               base_composition = c* at the off-centre windows; decides
               base-reachability vs target-itself where the uniform-base
               family is marginal or dead) and the lambda-curriculum arm
               (`anneal`, nochan + the 10/25/50 schedule at sigma_c
               centre; the third fate of the nochan/anneal/channel trio).
  dual eval -- every family is scored from eval/ AND eval_ema/ (hard's
               house convention: a second `_ema` entry per family). The
               raw entry keeps the un-averaged model on record; EMA is
               the instrument that rescued hard's marginal seeds.
  reference -- mchammer VC-SGC chains at kappa=lambda, phi=-2c*, under
               the 3-decimal composition naming (0.375 has no faithful
               2dp tag). potential.npy must equal the Ising energy
               recomputed from spins.npy under the coupling's own target
               -- the assert pins atom order, do not bypass it.
  floor     -- block bootstrap as the 10x10 fill, except the block length
               scales with the chains' own measured tau_int (>= 8x tau,
               min 10 frames): at sigma_c the 100-trial write interval no
               longer guarantees near-uncorrelated frames, and the fixed
               BLOCK=10 of the sigma=0.1 fill would understate the floor.
  energy    -- EW2 on the *Ising* energy per site, penalty excluded, as
               in the 10x10 fill: the penalty is the constraint, not the
               physics, and both sides draw from the same penalised law.

FLOP/es cells follow the 10x10 fill exactly: measured eager forward at
the run's own architecture (batch 1, batch-linear) x n_euler forwards /
frozen ESS fraction for the neural rows; analytic VC-SGC per-trial
constant x total trials (burn-in included) / (pooled frames / tau_int)
for the reference row.
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
# Run by path (numeric filenames can't be modules), so the `experiments`
# package imports below need the repo root.
sys.path.insert(0, str(REPO_ROOT))

from discrete_flow_sampler.diagnostics.flops import (  # noqa: E402
    chain_per_effective_sample, measured_forward_flops,
    neural_sampling_flops_per_sample, per_effective_sample, vcsgc_run_flops)
from discrete_flow_sampler.diagnostics.metrics import (  # noqa: E402
    correlation_profile_error, energy_wasserstein2,
    magnetisation_profile_error)
from discrete_flow_sampler.models.composition_conditioned import (  # noqa: E402
    CompositionConditioned)
from discrete_flow_sampler.targets.ising import (  # noqa: E402
    SIGMA_C, IsingTarget)
from experiments.constrained_soft_02.configs import (  # noqa: E402
    SOFT_HOUSE_WINDOWS)

SOFT_RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
VCSGC_RESULTS = REPO_ROOT / "results" / "mchammer_vcsgc"
L = 8
LAM = 50.0
COUPLINGS = (("s010", 0.1), ("sc", SIGMA_C))
ESS_FLOOR = 0.30
N_BOOTSTRAP, N_EVAL = 200, 5000
# Families scored from the frames their composition sweep filed at the
# window's c (one conditioned model, asked for each specialist's
# composition) rather than from the frozen centre eval.
CONDITIONED_FAMILIES = {"conditioned", "conditioned_ladder"}


def energy_per_site(x, target, sigma):
    return -target.base_log_prob(x) / (2 * sigma * target.d)


def observable_errors(x, weights, reference, target, sigma):
    return {
        "dMag": magnetisation_profile_error(x, weights, reference, L),
        "dCorr": correlation_profile_error(x, weights, reference, L),
        "EW2": energy_wasserstein2(
            energy_per_site(x, target, sigma), weights,
            energy_per_site(reference, target, sigma)),
    }


def load_vcsgc_reference(target, sigma, c_target, lam=LAM):
    """Pooled post-burn-in spin frames over the reference seeds, order-checked.

    Returns the pooled frames, the chains' cost record for the FLOP/es cell
    (total trials including burn-in; mchammer's own slowest per-observable
    tau_int in frame units), and that tau for the floor's block length.
    The reference is the chain at kappa = lam: a lambda twin is scored
    against ITS OWN penalised law, never against the lambda=50 chains.
    """
    frames, wall_seconds, chains = [], 0.0, 0
    total_trials, tau_ints = 0, []
    pattern = f"D{L}_s{sigma:g}_l{lam:.1f}_c{c_target:.3f}_seed*"
    for run_dir in sorted(VCSGC_RESULTS.glob(pattern)):
        spins = torch.from_numpy(np.load(run_dir / "spins.npy")).float()
        potential = torch.from_numpy(
            np.load(run_dir / "potential.npy")).float()
        # mchammer's CE energy is -log p~(x) on the validated embedding; a
        # scrambled atom order would break this equality and every profile.
        assert torch.allclose(
            -target.base_log_prob(spins), potential, atol=1e-3), run_dir
        frames.append(spins)
        summary = json.loads((run_dir / "summary.json").read_text())
        wall_seconds += summary["wall_seconds_run"]
        total_trials += summary["n_steps"]
        tau_ints.append(max(obs["tau_int_frames"]
                            for obs in summary["observables"].values()))
        chains += 1
    if not frames:
        raise FileNotFoundError(f"no VC-SGC reference matches {pattern}")
    pooled = torch.cat(frames)
    tau_int = max(sum(tau_ints) / len(tau_ints), 1.0)
    flops_per_es = chain_per_effective_sample(
        vcsgc_run_flops(total_trials), pooled.shape[0], tau_int)
    return pooled, chains, wall_seconds, flops_per_es, tau_int


def specialist_flops_per_forward(run_dir: Path, target, composition) -> int:
    """Measured FLOPs of one rate-matrix forward at this run's architecture.

    Eager build from the run's own config at batch 1 (compile stripped: a
    compiled wrapper can hide ops from the dispatch-level counter). The
    exact-field channel rides along -- its closed form is part of every
    forward the sampler pays for, so it belongs in the bill.
    """
    from experiments.dnfs_baseline_01.configs import ModelCfg
    from experiments.dnfs_baseline_01.run import _construct_model, _sub_config

    cfg_dict = json.loads((run_dir / "config.json").read_text())
    model_cfg = _sub_config(
        ModelCfg, {**cfg_dict["model"], "compile_model": False})
    model = _construct_model(SimpleNamespace(model=model_cfg), target)
    if model_cfg.condition_on_composition:
        # The conditioned forward reads c alongside (x, t); bind it exactly
        # as the sweep does so the counted forward is the one the row paid.
        model = CompositionConditioned(model, torch.full((1,), composition))
    example = (target.sample_base(1, device="cpu"), torch.zeros(1))
    return measured_forward_flops(model, example)


def reference_floor(reference, target, sigma, tau_int, seed=0):
    """Block bootstrap: N_EVAL-frame replicates scored against all frames.

    Block length >= 8 x the chains' measured tau_int (min 10 frames), so a
    critically slowed chain's correlations stay inside blocks rather than
    inflating the apparent independence of a replicate.
    """
    block = max(10, math.ceil(8 * tau_int))
    generator = torch.Generator().manual_seed(seed)
    n_blocks = reference.shape[0] // block
    by_block = reference[: n_blocks * block].view(n_blocks, block, -1)
    replicates = []
    for _ in range(N_BOOTSTRAP):
        blocks = torch.randint(
            0, n_blocks, (max(N_EVAL // block, 1),), generator=generator)
        replicate = by_block[blocks].reshape(-1, reference.shape[1])
        uniform = torch.full((replicate.shape[0],), 1.0 / replicate.shape[0])
        replicates.append(
            observable_errors(replicate, uniform, reference, target, sigma))
    return {k: sum(r[k] for r in replicates) / N_BOOTSTRAP
            for k in replicates[0]}, block


def score_runs(run_glob, reference, target, sigma, eval_subdir,
               per_forward_cache, c_target, sweep_composition=None):
    """One row per seed. A specialist is scored from its frozen eval; a
    conditioned run (`sweep_composition` set) from the frames its sweep
    filed at that composition, with that sweep row's own ESS. A conditioned
    run whose sweep predates the frames still yields its ESS (the dead
    sigma_c cells: the column reports them, the errors print as dagger).
    """
    per_seed = {}
    for run_dir in sorted(SOFT_RESULTS.glob(run_glob)):
        eval_dir = run_dir / eval_subdir
        if sweep_composition is None:
            if not (eval_dir / "samples.pt").exists():
                continue
            frame_dir = eval_dir
            ess = json.loads((eval_dir / "metrics.json").read_text())["ess_fraction"]
        else:
            sweep_path = eval_dir / "composition_sweep.json"
            if not sweep_path.exists():
                continue
            ess = next(r["ess_fraction"] for r in json.loads(sweep_path.read_text())
                       if abs(r["composition"] - sweep_composition) < 1e-6)
            frame_dir = eval_dir / "composition_sweep" / f"c{sweep_composition:.4f}"
        if (frame_dir / "samples.pt").exists():
            x = torch.load(frame_dir / "samples.pt", weights_only=True).float()
            weights = torch.softmax(
                torch.load(frame_dir / "log_weights.pt", weights_only=True), 0)
            errors = observable_errors(x, weights, reference, target, sigma)
        else:
            errors = {k: float("nan") for k in ("dMag", "dCorr", "EW2")}
        if run_glob not in per_forward_cache:  # one architecture per cell
            per_forward_cache[run_glob] = specialist_flops_per_forward(
                run_dir, target, c_target)
        n_euler = json.loads(
            (run_dir / "config.json").read_text())["ctmc"]["n_euler_steps"]
        per_seed[run_dir.name] = {
            "ESS": ess,
            **errors,
            "FLOPes": per_effective_sample(
                neural_sampling_flops_per_sample(
                    per_forward_cache[run_glob], n_euler, target.d),
                ess),
        }
    return per_seed


def mean_sd(values):
    t = torch.tensor(values)
    return t.mean().item(), (t.std().item() if len(t) > 1 else float("nan"))


def summarise(per_seed):
    keys = next(iter(per_seed.values())).keys()
    passing = {n: s for n, s in per_seed.items() if s["ESS"] >= ESS_FLOOR}
    return {
        "all": {k: mean_sd([s[k] for s in per_seed.values()]) for k in keys},
        "floor": ({k: mean_sd([s[k] for s in passing.values()])
                   for k in keys} if passing else None),
        "n_pass": len(passing), "n_total": len(per_seed),
    }


def house_cells():
    """(cell key, sigma label, sigma, c*, lambda, {family: run glob}).

    The lambda=50 windows carry the house families; the lambda twins
    (motivation section, tag 20260901-softmotiv-d64) are single-family
    cells at the centre with their own kappa=lambda reference.
    """
    for sigma_label, sigma in COUPLINGS:
        sigma_suffix = "_sc" if sigma_label == "sc" else ""
        for c_target, c_tag in SOFT_HOUSE_WINDOWS:
            families = {
                "specialist":
                    f"S2_d8_{c_tag}_l50_letf_ne128_house{sigma_suffix}"
                    f"_seed4*",
            }
            if c_target != 0.50:
                # At c* = 0.5 the house Bernoulli(0.5) base is already
                # matched, so mb twins exist only off-centre.
                families["mb"] = (
                    f"S2_d8_{c_tag}_l50_letf_ne128_house_mb{sigma_suffix}"
                    f"_seed4*")
            # The conditioned cell (one model over the window range): cold
            # at both couplings, plus the sigma-ladder twin at sigma_c, where
            # the cold cell is dead and the ladder is the delivered arm.
            families["conditioned"] = (
                f"S2_d8_camort_l50_letf_ne128_house{sigma_suffix}_seed4*")
            if sigma_label == "sc":
                families["conditioned_ladder"] = (
                    "S2_d8_camort_l50_letf_ne128_house_sc_curr_seed4*")
            if c_target == 0.50:
                # Channel-off control at both couplings (trains at
                # sigma=0.1, dead at sigma_c: the shock arrives with the
                # coupling); the anneal fate exists at sigma_c only.
                families["nochan"] = (
                    f"S2_d8_c0500_l50_letf_ne128_house{sigma_suffix}"
                    f"_nochan_seed4*")
                if sigma_label == "sc":
                    families["anneal"] = (
                        "S2_d8_c0500_l50_letf_ne128_house_sc_anneal_seed4*")
            yield (f"{sigma_label}_c{c_target:.3f}", sigma_label, sigma,
                   c_target, LAM, families)
        for lam in (10, 100):
            yield (f"{sigma_label}_c0.500_l{lam}", sigma_label, sigma,
                   0.50, float(lam), {
                       "specialist":
                           f"S2_d8_c0500_l{lam}_letf_ne128_house"
                           f"{sigma_suffix}_seed4*"})


def main():
    table = {}
    for key, sigma_label, sigma, c_target, lam, families in house_cells():
        target = IsingTarget(D=L, sigma=sigma, bias=0.0)
        try:
            reference, n_chains, wall_seconds, ref_flops_per_es, tau = \
                load_vcsgc_reference(target, sigma, c_target, lam)
        except FileNotFoundError as missing:
            print(f"\n== {key}: SKIPPED ({missing})")
            continue
        floor, block = reference_floor(reference, target, sigma, tau)
        cell = {"lambda": lam, "reference_floor": floor,
                "reference_chains": n_chains,
                "reference_frames": reference.shape[0],
                "reference_tau_int_frames": tau,
                "reference_floor_block": block,
                "reference_wall_seconds": wall_seconds,
                "reference_flops_per_es": ref_flops_per_es}
        print(f"\n== {key} ({n_chains} chains, "
              f"{reference.shape[0]} frames, tau {tau:.2f}, "
              f"block {block})")
        print("  reference floor:",
              {k: f"{v:.2e}" for k, v in floor.items()},
              f" reference FLOP/es: {ref_flops_per_es:.2g}")
        per_forward_cache = {}
        for family, glob in families.items():
            for eval_subdir in ("eval", "eval_ema"):
                per_seed = score_runs(
                    glob, reference, target, sigma, eval_subdir,
                    per_forward_cache, c_target,
                    sweep_composition=(
                        c_target if family in CONDITIONED_FAMILIES else None))
                if not per_seed:
                    print(f"  [{family}/{eval_subdir}] no runs match "
                          f"{glob}")
                    continue
                family_key = family + (
                    "_ema" if eval_subdir == "eval_ema" else "")
                summary = summarise(per_seed)
                cell[family_key] = {"per_seed": per_seed, **summary}
                for name, s in per_seed.items():
                    print(f"  [{family_key}] {name}: " + " ".join(
                        f"{k}={v:.4g}" for k, v in s.items()))
                for rule in ("all", "floor"):
                    if summary[rule]:
                        print(
                            f"  [{family_key}] {rule:5s} mean +- SD "
                            f"({summary['n_pass']}/{summary['n_total']} "
                            f"clear {ESS_FLOOR}):",
                            {k: f"{m:.4g} +- {sd:.2g}"
                             for k, (m, sd) in summary[rule].items()})
        table[key] = cell

    out = SOFT_RESULTS / "house_table_soft_8x8.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
