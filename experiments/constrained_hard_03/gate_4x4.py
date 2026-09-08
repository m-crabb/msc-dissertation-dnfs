"""The strengthened 4x4 exact-enumeration pass/fail gate + both negative controls.

`FixedCompositionIsingTarget(D=4, sigma, c=0.5)` gives d=16, N_A=8, a slice of
C(16,8)=12,870 states. On the slice Sigma(x) is constant, so the conditional is a
pure function of energy, pi(x|C) proportional to exp(sigma * x^T A x). A trained
swap head is evaluated by drawing IS-weighted samples with the swap CTMC
(`sample_swap_ctmc(return_log_weights=True, target=target)`) and comparing every
observable against the exactly enumerated conditional, across a sigma-ladder that
crosses the Ising transition.

Energy-marginal TV is the headline distance the paper reports (App D.1), but it
only constrains cross-level weights: pi(x|C) is uniform within each energy level.
Within-level uniformity closes that blind spot against a weight-matched
perfect-sampler baseline. <E> and nn_correlation are energy-tied observables;
diagonal_correlation is the independent within-level spatial one (checkerboard:
nn = -1 but diag = +1). On-slice IS-ESS is estimator health, not a mixing claim.
The eval-time antisymmetry check G(i,j | Swap2 x) + G(i,j | x) = 0 is structural
for the dh/mask_one heads and is what the non_antisym control must trip.

Run through the local CLI or Modal gate wrapper. Sampling uses torch.no_grad().
"""
import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from experiments.constrained_hard_03.configs import CONFIGS

from discrete_flow_sampler.constraints.swap_readout import swap2
from discrete_flow_sampler.diagnostics.metrics import (
    conditional_pmf_at_composition,
    diagonal_correlation,
    enumerate_states,
    ess_from_log_weights,
    exact_log_probs,
    free_energy_lb_estimate,
    nn_correlation,
)
from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.seeding import seed_everything

# The three sigma-ladder rungs (subcritical / critical / hardened) + the na cell.
RUNGS = {
    "s010": "H2_d16_c50_s010_letf_dh",
    "s223": "H2_d16_c50_s223_letf_dh",
    "s040": "H2_d16_c50_s040_letf_dh",
}
NON_ANTISYM_CELL = "H2_d16_c50_s010_letf_na"
ANTISYM_TOL = 1e-4


# --- energy-marginal helpers (unit-tested) ---


def _energy(states, adjacency):
    """Slice energy x^T A x per state (the quantity that pins pi(x|C))."""
    x = states.float()
    return torch.einsum("bi,ij,bj->b", x, adjacency, x)


def slice_energy_hist(states, weights, adjacency, bins):
    """IS-weighted histogram of the slice energy x^T A x over `bins` edges.

    Returns (bin_centres, mass): `bin_centres` are edge midpoints, `mass` the
    normalised weight per bin. `weights` are normalised here. Energy outside
    [bins[0], bins[-1]) is dropped, so the mass can sum to < 1; that lost mass is
    the off-slice leakage the product-Bernoulli control reads off.
    """
    energy = _energy(states, adjacency)
    weights = weights.float()
    weights = weights / weights.sum()
    lower, upper = bins[:-1], bins[1:]
    membership = (energy[:, None] >= lower[None, :]) & (
        energy[:, None] < upper[None, :]
    )
    mass = (membership.float() * weights[:, None]).sum(dim=0)
    return 0.5 * (lower + upper), mass


def energy_marginal_tv(dnfs_hist, exact_hist):
    """Total variation 0.5 * Sigma |mass_a - mass_b| on shared bins."""
    return 0.5 * (dnfs_hist[1] - exact_hist[1]).abs().sum().item()


def _categorical_energy_bins(slice_energies):
    """Edges that place each unique slice energy in its own bin: midpoints
    between consecutive uniques, +/-0.5 beyond the extremes."""
    unique = torch.unique(slice_energies).sort().values
    mids = 0.5 * (unique[:-1] + unique[1:])
    return torch.cat([unique[:1] - 0.5, mids, unique[-1:] + 0.5])


# --- on-slice free energy + within-level uniformity ---


def on_slice_free_energy_reference(target, slice_states):
    """On-slice free-energy reference F_ref/D, comparable to the DNFS estimate.

        F_ref/D = -logsumexp_{x in C} log p_tilde(x) / (2 sigma d)

    matching the per-site normalisation of `free_energy_lb_estimate` (metrics.py)
    so F_dnfs - F_ref is meaningful. Not `exact_free_energy`: it enumerates the
    full 2^d space, whereas the swap CTMC only visits the fixed-N slice. The
    on-manifold base constant -log C(d, N_A) enters the DNFS log-weight (via
    base_log_eta at t=0) and this reference identically, so it cancels.
    """
    sigma = float(target.sigma)
    d = int(target.d)
    log_Z_slice = torch.logsumexp(target.log_prob(slice_states.float()), dim=0)
    return -log_Z_slice / (2 * sigma * d)


def _pack_spin_keys(states):
    """Pack each row of +/-1 spins into a unique integer identity (d <= 62)."""
    x_idx = ((states + 1) // 2).long()
    powers = (2 ** torch.arange(states.shape[1], device=states.device)).long()
    return (x_idx * powers).sum(dim=-1)


def within_level_uniformity(
    sample_states,
    sample_weights,
    sample_energies,
    slice_states,
    slice_energies,
    *,
    min_count=100,
    n_ref_replicates=20,
    seed=0,
):
    """Per-energy-level within-level uniformity.

    pi(x|C) is uniform within each energy level, so energy-TV alone cannot see
    within-level structure. For each level with raw sample count n_k >= min_count,
    map every sample to its slice-state identity, form the IS-weighted
    distribution over the level's g_k states, and compute TV_k against
    uniform(1/g_k). A raw TV_k is uninterpretable on its own: a perfect sampler
    still shows TV ~ 1 - n_k/g_k when n_k << g_k (the TVD-floor trap), and skewed
    IS weights inflate TV_k further because n_eff_k = (Sigma w)^2 / Sigma w^2 < n_k.

    So a weight-matched perfect-sampler baseline is subtracted: each of the
    R = n_ref_replicates replicates keeps the level's observed normalised IS
    weight vector and assigns it to n_k uniform draws over the g_k states. Under
    the null -- sampler uniform within the level, weights independent of
    within-level identity -- that is the distribution of the statistic, so
    excess = TV_k - TV_k^ref is centred at ~0 for a faithful sampler. An
    equal-weight baseline instead models only the sample-size floor and would
    bias the excess upward, failing a low-ESS rung (sigma=0.40) spuriously.
    Returns one dict per well-populated level, n_eff_k included.
    """
    generator = torch.Generator().manual_seed(seed)
    sample_keys = _pack_spin_keys(sample_states)
    slice_keys = _pack_spin_keys(slice_states)
    results = []
    for level_energy in torch.unique(slice_energies).tolist():
        level_slice = slice_energies == level_energy
        level_keys = slice_keys[level_slice]
        g_k = int(level_keys.numel())
        level_sample = sample_energies == level_energy
        n_k = int(level_sample.sum().item())
        if n_k < min_count:
            continue

        key_to_pos = {int(k): pos for pos, k in enumerate(level_keys.tolist())}
        positions = torch.tensor(
            [key_to_pos[int(k)] for k in sample_keys[level_sample].tolist()]
        )
        weights = sample_weights[level_sample].float().cpu()  # index_add on cpu
        weights = weights / weights.sum()
        p_hat = torch.zeros(g_k).index_add_(0, positions, weights)
        tv_k = 0.5 * (p_hat - 1.0 / g_k).abs().sum().item()
        # (Sigma w)^2 / Sigma w^2 with w already normalised to Sigma w = 1.
        n_eff_k = (1.0 / weights.pow(2).sum()).item()

        tv_refs = []
        for _ in range(n_ref_replicates):
            draws = torch.randint(0, g_k, (n_k,), generator=generator)
            p_ref = torch.zeros(g_k).index_add_(0, draws, weights)
            tv_refs.append(0.5 * (p_ref - 1.0 / g_k).abs().sum().item())
        tv_ref = sum(tv_refs) / len(tv_refs)

        results.append(
            {
                "energy": float(level_energy),
                "n_k": n_k,
                "g_k": g_k,
                "n_eff_k": n_eff_k,
                "tv_k": tv_k,
                "tv_ref": tv_ref,
                "excess": tv_k - tv_ref,
            }
        )
    return results


# --- run loading ---


def latest_run_dir(results_dir, cfg_name, seed):
    """The lexicographically latest run dir for (cfg_name, seed)."""
    matches = sorted(Path(results_dir).glob(f"{cfg_name}_seed{seed}_*"))
    if not matches:
        raise FileNotFoundError(
            f"no run dir for {cfg_name} seed {seed} under {results_dir}"
        )
    return matches[-1]


def load_run(run_dir, device):
    """Rebuild (head, target) from a run dir's config.json + checkpoints/final.pt.

    Delegates to `run.build_target_and_head` so the gate cannot drift from the
    trainer in how it instantiates the backbone/head. The head wraps the backbone,
    so final.pt's keys are prefixed `backbone.` and load onto the head module
    directly. The head kind comes from config.json (the ladder trained mask_one,
    bit-exact-equal to doubly_hollow per the d=16 oracle pin).
    """
    from experiments.constrained_hard_03.run import build_target_and_head

    cfg_dict = json.loads((run_dir / "config.json").read_text())
    cfg = replace(CONFIGS[cfg_dict["name"]], head_kind=cfg_dict["head_kind"])
    target, head = build_target_and_head(cfg, device)
    state = torch.load(
        run_dir / "checkpoints" / "final.pt", map_location=device, weights_only=True
    )
    head.load_state_dict(state)
    head.eval()
    return head, target


def _finite_column(rows, column):
    values = []
    for row in rows:
        try:
            value = float(row[column])
        except (KeyError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def clamp_fractions(run_dir):
    """Final and max Lambda*dt>1 clip-fraction and log-ratio clamp-hit fraction
    from training_log.csv. Lambda*dt>1 forces one swap that step (a coarse-
    time-step signal, watch the sigma=0.40 rung); a nonzero log-ratio clamp-hit
    fraction would flag IS-weight bias (expected ~0 across the ladder).
    """
    rows = list(csv.DictReader((run_dir / "training_log.csv").open()))
    lam = _finite_column(rows, "lambda_dt_clipped_frac")
    clamp = _finite_column(rows, "log_ratio_clamp_frac")
    return {
        "lambda_dt_clipped_frac_final": lam[-1] if lam else float("nan"),
        "lambda_dt_clipped_frac_max": max(lam) if lam else float("nan"),
        "log_ratio_clamp_frac_final": clamp[-1] if clamp else float("nan"),
        "log_ratio_clamp_frac_max": max(clamp) if clamp else float("nan"),
    }


# --- the gate itself ---


def _antisymmetry_violation(head, states, d):
    """max_{i<j} |G(i,j | Swap2 x) + G(i,j | x)| over the state subsample.

    One base pass plus one swapped pass per i<j pair. Structural ~0 for the
    dh/mask_one heads; the non_antisym control must make it large (biased
    single-pass reverse rate).
    """
    t = torch.full((states.shape[0],), 0.5, device=states.device)
    base = head(states, t)
    violation = 0.0
    for pair in upper_tri_pairs(d, states.device).tolist():
        i, j = pair
        swapped = head(swap2(states, i, j), t)
        pair_violation = (swapped[:, i, j] + base[:, i, j]).abs().max().item()
        violation = max(violation, pair_violation)
    return violation


def run_gate(head, target, n_samples, n_euler_steps, seed):
    """Full positive-gate eval for one trained head on the 4x4 slice.

    Draws IS-weighted swap-CTMC samples and compares energy marginal, <E>,
    NN/diagonal correlations, within-level uniformity and F/D against the exactly
    enumerated conditional. Returns raw numbers only; the pass/fail rules live in
    `_aggregate_rung`.
    """
    device = next(head.parameters()).device
    d = int(target.d)
    adjacency = target.A

    all_states = enumerate_states(d).to(device)
    log_pi = exact_log_probs(target, all_states)
    slice_states, log_p_cond = conditional_pmf_at_composition(
        all_states, log_pi, target.n_plus_target
    )
    slice_states = slice_states.float()
    p_cond = log_p_cond.exp()
    slice_energy = _energy(slice_states, adjacency)
    bins = _categorical_energy_bins(slice_energy)

    with torch.no_grad():
        seed_everything(seed)
        x0 = target.sample_base(n_samples, device=device)
        ts = torch.linspace(0.0, 1.0, n_euler_steps + 1, device=device)
        samples, log_w = sample_swap_ctmc(
            head, x0, ts, return_log_weights=True, target=target
        )
        weights = torch.softmax(log_w, dim=0)

        exact_hist = slice_energy_hist(slice_states, p_cond, adjacency, bins)
        dnfs_hist = slice_energy_hist(samples, weights, adjacency, bins)
        energy_tv = energy_marginal_tv(dnfs_hist, exact_hist)

        sample_energy = _energy(samples, adjacency)
        e_exact = (p_cond * slice_energy).sum().item()
        e_dnfs = (weights * sample_energy).sum().item()
        nn_exact = (p_cond * nn_correlation(slice_states, adjacency)).sum().item()
        nn_dnfs = (weights * nn_correlation(samples, adjacency)).sum().item()
        diag_exact = (
            (p_cond * diagonal_correlation(slice_states, target.D)).sum().item()
        )
        diag_dnfs = (weights * diagonal_correlation(samples, target.D)).sum().item()
        ess = ess_from_log_weights(log_w).item()

        levels = within_level_uniformity(
            samples, weights, sample_energy, slice_states, slice_energy, seed=seed
        )
        excesses = [level["excess"] for level in levels]

        f_ref = on_slice_free_energy_reference(target, slice_states).item()
        f_dnfs = free_energy_lb_estimate(log_w, sigma=float(target.sigma), D=d).item()

        antisym = _antisymmetry_violation(head, samples[:64], d)

    return {
        "seed": seed,
        "n_samples": n_samples,
        "energy_tv": energy_tv,
        "e_exact": e_exact,
        "e_dnfs": e_dnfs,
        "nn_exact": nn_exact,
        "nn_dnfs": nn_dnfs,
        "diag_exact": diag_exact,
        "diag_dnfs": diag_dnfs,
        "ess": ess,
        "ess_fraction": ess / n_samples,
        "free_energy_ref": f_ref,
        "free_energy_dnfs": f_dnfs,
        "free_energy_bias": f_dnfs - f_ref,
        "antisym_violation": antisym,
        "antisym_pass": antisym < ANTISYM_TOL,
        "within_level": levels,
        "max_level_excess": max(excesses) if excesses else float("nan"),
        "mean_level_excess": (
            sum(excesses) / len(excesses) if excesses else float("nan")
        ),
        "energy_centres": exact_hist[0].tolist(),
        "hist_dnfs": dnfs_hist[1].tolist(),
        "hist_exact": exact_hist[1].tolist(),
    }


# --- aggregation + pass/fail ---


def _seed_summary(metrics):
    heavy = ("energy_centres", "hist_dnfs", "hist_exact")
    return {key: value for key, value in metrics.items() if key not in heavy}


def _aggregate_rung(per_seed):
    """Aggregate per-seed gate metrics into rung booleans (pass/fail rules)."""
    tv = torch.tensor([m["energy_tv"] for m in per_seed])

    def observable(key):
        values = torch.tensor([m[key] for m in per_seed])
        std = values.std(unbiased=True).item() if values.numel() > 1 else 0.0
        return values.mean().item(), std

    e_mean, e_std = observable("e_dnfs")
    nn_mean, nn_std = observable("nn_dnfs")
    diag_mean, diag_std = observable("diag_dnfs")
    e_exact = per_seed[0]["e_exact"]
    nn_exact = per_seed[0]["nn_exact"]
    diag_exact = per_seed[0]["diag_exact"]
    finite_excess = [
        m["max_level_excess"] for m in per_seed if math.isfinite(m["max_level_excess"])
    ]
    max_excess = max(finite_excess, default=float("nan"))
    tv_mean = tv.mean().item()

    pass_energy_tv = tv_mean <= 0.02
    pass_E = abs(e_mean - e_exact) <= 2 * e_std
    pass_nn = abs(nn_mean - nn_exact) <= 2 * nn_std
    pass_diag = abs(diag_mean - diag_exact) <= 2 * diag_std
    pass_within_level = math.isfinite(max_excess) and max_excess <= 0.05
    rung_pass = all([pass_energy_tv, pass_E, pass_nn, pass_diag, pass_within_level])
    return {
        "energy_tv_mean": tv_mean,
        "energy_tv_per_seed": tv.tolist(),
        "e_exact": e_exact,
        "e_dnfs_mean": e_mean,
        "e_dnfs_std": e_std,
        "e_abs_dev": abs(e_mean - e_exact),
        "nn_exact": nn_exact,
        "nn_dnfs_mean": nn_mean,
        "nn_dnfs_std": nn_std,
        "nn_abs_dev": abs(nn_mean - nn_exact),
        "diag_exact": diag_exact,
        "diag_dnfs_mean": diag_mean,
        "diag_dnfs_std": diag_std,
        "diag_abs_dev": abs(diag_mean - diag_exact),
        "max_level_excess": max_excess,
        "ess_fraction_mean": sum(m["ess_fraction"] for m in per_seed) / len(per_seed),
        "free_energy_bias_mean": (
            sum(m["free_energy_bias"] for m in per_seed) / len(per_seed)
        ),
        "antisym_violation_max": max(m["antisym_violation"] for m in per_seed),
        "clamp_per_seed": [m["clamp"] for m in per_seed],
        "pass_energy_tv": pass_energy_tv,
        "pass_E": pass_E,
        "pass_nn": pass_nn,
        "pass_diag": pass_diag,
        "pass_within_level": pass_within_level,
        "rung_pass": rung_pass,
        "per_seed": [_seed_summary(m) for m in per_seed],
    }


def _control_product_bernoulli(results_dir, seeds, n_samples, device):
    """Negative control (i): re-sample the trained s010 head from a product-
    Bernoulli(0.5) base (no manifold projection). Swaps conserve each sample's
    initial composition, so the composition histogram stays binomial (std ~ 0.125
    in composition units) rather than collapsing to N_A. This falsifies the
    on-manifold base, not the head.
    """
    cfg_name = RUNGS["s010"]
    n_euler = CONFIGS[cfg_name].ctmc.n_euler_steps
    per_seed = []
    for seed in seeds:
        run_dir = latest_run_dir(results_dir, cfg_name, seed)
        print(
            f"[gate] control product_bernoulli seed {seed}: {run_dir.name}", flush=True
        )
        head, target = load_run(run_dir, device)
        d = int(target.d)
        all_states = enumerate_states(d).to(device)
        log_pi = exact_log_probs(target, all_states)
        slice_states, log_p_cond = conditional_pmf_at_composition(
            all_states, log_pi, target.n_plus_target
        )
        bins = _categorical_energy_bins(_energy(slice_states.float(), target.A))
        exact_hist = slice_energy_hist(
            slice_states.float(), log_p_cond.exp(), target.A, bins
        )
        with torch.no_grad():
            seed_everything(seed)
            x0 = torch.randint(0, 2, (n_samples, d), device=device).float() * 2 - 1
            ts = torch.linspace(0.0, 1.0, n_euler + 1, device=device)
            samples = sample_swap_ctmc(head, x0, ts)  # no target -> no log-weights
            composition = ((samples + 1) * 0.5).mean(dim=-1)
            n_plus = ((samples + 1) * 0.5).sum(dim=-1)
            dnfs_hist = slice_energy_hist(
                samples, torch.ones(n_samples, device=device), target.A, bins
            )
            energy_tv = energy_marginal_tv(dnfs_hist, exact_hist)
        per_seed.append(
            {
                "seed": seed,
                "composition_mean": composition.mean().item(),
                "composition_std": composition.std(unbiased=False).item(),
                "fraction_n_plus_target": (n_plus == target.n_plus_target)
                .float()
                .mean()
                .item(),
                "energy_tv_unweighted": energy_tv,
            }
        )

    def mean(key):
        return sum(row[key] for row in per_seed) / len(per_seed)

    return {
        "cfg": cfg_name,
        "composition_mean": mean("composition_mean"),
        "composition_std": mean("composition_std"),
        "fraction_n_plus_target": mean("fraction_n_plus_target"),
        "energy_tv_unweighted": mean("energy_tv_unweighted"),
        "per_seed": per_seed,
    }


def _control_non_antisym(results_dir, seeds, n_samples, device):
    """Negative control (ii): the trained non_antisym cell put through the full
    positive-gate eval. Its single-pass reverse rates are biased, so the eval-time
    antisymmetry check must trip (violation >> tol).
    """
    cfg_name = NON_ANTISYM_CELL
    n_euler = CONFIGS[cfg_name].ctmc.n_euler_steps
    per_seed = []
    for seed in seeds:
        run_dir = latest_run_dir(results_dir, cfg_name, seed)
        print(f"[gate] control non_antisym seed {seed}: {run_dir.name}", flush=True)
        head, target = load_run(run_dir, device)
        per_seed.append(run_gate(head, target, n_samples, n_euler, seed))
    violations = [m["antisym_violation"] for m in per_seed]
    return {
        "cfg": cfg_name,
        "antisym_violation": sum(violations) / len(violations),
        "antisym_violation_max": max(violations),
        "energy_tv_mean": sum(m["energy_tv"] for m in per_seed) / len(per_seed),
        "per_seed": [
            {
                "seed": m["seed"],
                "antisym_violation": m["antisym_violation"],
                "antisym_pass": m["antisym_pass"],
                "energy_tv": m["energy_tv"],
            }
            for m in per_seed
        ],
    }


def _overall_pass(verdict, skip_controls):
    rungs_pass = all(rung["rung_pass"] for rung in verdict["rungs"].values())
    if skip_controls:
        return rungs_pass
    control_i = verdict["controls"]["product_bernoulli"]["composition_std"] > 0.05
    control_ii = (
        verdict["controls"]["non_antisym"]["antisym_violation"] > 10 * ANTISYM_TOL
    )
    return rungs_pass and control_i and control_ii


def _plot_energy_hists(hist_by_rung, out_path):
    """One panel per rung: exact conditional (black) vs IS-weighted DNFS
    histograms (seeds overlaid)."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe (Modal containers have no display)
    import matplotlib.pyplot as plt

    rungs = list(hist_by_rung.keys())
    fig, axes = plt.subplots(1, len(rungs), figsize=(5 * len(rungs), 4), squeeze=False)
    for ax, rung in zip(axes[0], rungs):
        per_seed = hist_by_rung[rung]
        centres = per_seed[0]["energy_centres"]
        ax.step(
            centres,
            per_seed[0]["hist_exact"],
            where="mid",
            color="k",
            lw=1.8,
            label=r"exact $\pi(\cdot|C)$",
        )
        for metrics in per_seed:
            ax.step(
                centres,
                metrics["hist_dnfs"],
                where="mid",
                alpha=0.6,
                lw=1.0,
                label=f"DNFS seed {metrics['seed']}",
            )
        ax.set_title(rung)
        ax.set_xlabel(r"energy $x^\top A x$")
        ax.set_ylabel("mass")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results/03_hard")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="positive-only pass (ladder rungs, no negative controls)",
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="comma-separated CONFIGS names to gate instead of "
        "the dh ladder RUNGS (e.g. the 4x4 head/backbone twins)",
    )
    args = parser.parse_args(argv)
    rungs = (
        dict(zip(args.cells.split(","), args.cells.split(","))) if args.cells else RUNGS
    )

    results_dir = Path(args.results_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out) if args.out else results_dir / "gate_4x4"
    out_dir.mkdir(parents=True, exist_ok=True)

    verdict = {"rungs": {}, "controls": {}}
    hist_by_rung = {}
    for rung, cfg_name in rungs.items():
        n_euler = CONFIGS[cfg_name].ctmc.n_euler_steps
        per_seed = []
        for seed in seeds:
            run_dir = latest_run_dir(results_dir, cfg_name, seed)
            print(f"[gate] rung {rung} seed {seed}: {run_dir.name}", flush=True)
            head, target = load_run(run_dir, args.device)
            metrics = run_gate(head, target, args.n_samples, n_euler, seed)
            metrics["run_dir"] = str(run_dir)
            metrics["clamp"] = clamp_fractions(run_dir)
            per_seed.append(metrics)
            print(
                f"[gate] rung {rung} seed {seed}: energy_tv={metrics['energy_tv']:.4f}"
                f" ess_frac={metrics['ess_fraction']:.3f}",
                flush=True,
            )
        verdict["rungs"][rung] = _aggregate_rung(per_seed)
        hist_by_rung[rung] = per_seed

    if not args.skip_controls:
        verdict["controls"]["product_bernoulli"] = _control_product_bernoulli(
            results_dir, seeds, args.n_samples, args.device
        )
        verdict["controls"]["non_antisym"] = _control_non_antisym(
            results_dir, seeds, args.n_samples, args.device
        )

    verdict["overall_pass"] = _overall_pass(verdict, args.skip_controls)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    _plot_energy_hists(hist_by_rung, out_dir / "energy_hist.png")

    summary = {"overall_pass": verdict["overall_pass"]}
    summary.update(
        {rung: verdict["rungs"][rung]["rung_pass"] for rung in verdict["rungs"]}
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
