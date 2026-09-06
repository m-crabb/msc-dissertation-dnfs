"""How much of a trained 4x4 SOFT sampler is closed-form? (flip analogue of
the hard chapter's local-field regression, and the instrument that predicts
whether an exact-field channel would pay in the soft chapter.)

The leTF emits G(i|x) = -x_i S_i(x) with S hollow at site i, so the learned
object is the blind score S. The soft target's exact flip log-ratio is
(tests/test_soft_field_regression.py pins this against brute force):

    Delta_i(x) = x_i * [ -4 sigma h_i + 2 lambda (c_null_i - c*) + lambda/d ]

with h_i = (A x)_i the local field and c_null_i the hole-excluded
composition — exactly odd in x_i with a hollow coefficient, so the
architecture's representable set contains this equilibrium log-ratio. That
does not force the trained flow score to equal it. The equilibrium blind
score is spanned by TWO
closed-form columns: the LOCAL field h_i (the hard chapter's channel) and
the GLOBAL-but-closed-form penalty offset (c_null_i - c*), which is the
soft-specific channel — a single scalar per (state, site) that any head
could be handed for the price of a running sum.

This script asks how much of each trained specialist's S is
  (a) the linear field alone,          r2_field
  (b) the penalty offset alone,        r2_penalty
  (c) their span = the exact channel,  r2_channel   <- the headline column
  (d) + local/global quadratics,       r2_quadratic
  (e) a cell-mean fit over site, 4 neighbour spins and hole-excluded up-count
      (held-out lookup, with finite fit-cell counts),
                                       r2_any_local_count
  (f) and whether its held-out residual follows the hole-excluded bond sum
      B_null (the blind global energy). This residual includes lookup
      estimation error; it does not by itself certify nonlocal structure.

Enumerates ALL 2^16 states (the soft process is unconstrained), so the linear
regressions cover the full state space, up to numerical error. The lookup
scores still depend on the fit split and unseen-cell fallback. Two
weightings are reported for the linear
designs: UNIFORM over states, mirroring the hard instrument, and
p~_t-WEIGHTED, because the soft sampler concentrates near c* and a head is
only trained where the rollout goes — a channel that looks half-useless
uniformly but complete under p~_t is still a paying channel. The lookup
tiers are uniform-only: reweighting up to 4,096 cell means by p~_t leaves many
cells with tiny effective counts and the held-out R^2 becomes an estimate
of weight noise rather than capacity.

How to read the result: the archived hard 4x4 regression put the linear
field at roughly half the variance, and the channel
paid at 8x8. If r2_channel here is well above that, the case for wiring
sigma*h and the lambda-offset into a soft head as fixed channels with
learned gains is STRONGER than the one that already paid; if the gap
(e)-(c) is large, the lookup captures dependence beyond the linear channel,
including local nonlinearity. It is not evidence of nonlocality. Archived
lookup scores used colliding site keys and centred residual variance;
recomputed lookup scores are not numerically interchangeable. CPU, minutes.
"""

import argparse
import json
from pathlib import Path

import torch

from experiments.constrained_soft_02.analysis._common import latest_run_dir
from discrete_flow_sampler.diagnostics.metrics import enumerate_states

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECIALISTS = (
    "S2_d4_c03_50k_l50_letf_anneal_offset_clip50",
    "S2_d4_c05_50k_l50_letf_anneal_offset_clip50",
    "S2_d4_c07_50k_l50_letf_anneal_offset_clip50",
    "S2_d4_c08_50k_l50_letf_anneal_offset_clip50",
)
SEEDS = (42, 43, 44, 45)


def load_run(run_dir):
    """Rebuild (model, target) from config.json + checkpoints/final.pt.

    Use the baseline trainer's builders, as the hard compile gate does.
    Specialists only: amortised models require composition at every forward
    and are out of scope here.
    """
    from experiments.dnfs_baseline_01.run import (
        _build_model, _rebuild_from_run_dir)

    cfg, target, device = _rebuild_from_run_dir(run_dir)
    if cfg.condition_on_composition:
        raise ValueError(f"{run_dir.name}: amortised run; specialists only")
    model = _build_model(cfg, target)
    state = torch.load(run_dir / "checkpoints" / "final.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, target


def r_squared(target_col, design, weights=None):
    """R^2 of `target_col` (N,) on columns of `design` (N, k), optionally
    under row weights (weighted LS, weighted variances).

    float64 with standardised columns, inherited from the hard instrument:
    an fp32 solve there returned a superset design scoring BELOW its subset
    (conditioning, not signal).
    """
    y = target_col.double()
    X = design.double()
    scale = X.std(0).clamp(min=1e-12)
    scale[X.std(0) == 0] = 1.0
    X = X / scale
    if weights is not None:
        root = weights.double().clamp(min=0).sqrt().unsqueeze(1)
        coef = torch.linalg.lstsq(X * root, (y.unsqueeze(1) * root)).solution
        residual = y - (X @ coef).squeeze(1)
        w = weights.double() / weights.double().sum()
        mean = (w * y).sum()
        var = (w * (y - mean) ** 2).sum()
        res_var = (w * (residual - (w * residual).sum()) ** 2).sum()
        return float(1.0 - res_var / var)
    coef = torch.linalg.lstsq(X, y.unsqueeze(1)).solution
    residual = y - (X @ coef).squeeze(1)
    return float(1.0 - residual.var() / y.var())


def lookup_r_squared(target_col, keys, fit_mask):
    """Held-out cell-mean fit; unseen keys predict the fit-set global mean.

    R^2 = 1 - sum(residual^2) / sum((held_y - mean(held_y))^2).
    Unlike an in-sample fit with an intercept, held-out residuals need not
    have zero mean: centring them would hide prediction bias. The result
    measures this split's predictions, not an exact local-capacity bound.
    """
    unique, inverse = torch.unique(keys, return_inverse=True)
    fit = fit_mask.float()
    sums = torch.zeros(len(unique)).index_add_(0, inverse, target_col * fit)
    counts = torch.zeros(len(unique)).index_add_(0, inverse, fit)
    means = torch.where(
        counts > 0, sums / counts.clamp(min=1), target_col[fit_mask].mean())
    held = ~fit_mask
    residual = target_col[held] - means[inverse][held]
    total = (target_col[held] - target_col[held].mean()).square().sum()
    return float(1.0 - residual.square().sum() / total), residual


def local_lookup_keys(states, adjacency):
    """Flattened integer keys for (site, neighbours), optionally plus count.

    On the 4x4 lattice a global-position neighbour mask is in [0, 2^d),
    even though each site has only 16 local patterns. Thus site * 2^d +
    mask is injective; site * 16 + mask wrongly merges distinct sites.
    Appending the hole-excluded count uses radix d because it is in [0, d).
    Integer masks avoid introducing float rounding into these identities.
    """
    d = states.shape[1]
    up = (states > 0).long()
    powers = 2 ** torch.arange(d, device=states.device)
    pattern = up @ ((adjacency > 0).long() * powers).T
    site = torch.arange(d, device=states.device)
    local = site * 2**d + pattern
    count = up.sum(1, keepdim=True) - up
    return local.reshape(-1), (local * d + count).reshape(-1)


def analyse(run_dir, t_value):
    model, target = load_run(Path(run_dir))
    d, A = target.d, target.A
    c_star = target.target_composition
    x = enumerate_states(d).float()   # ±1 spins, int64 -> float for matmuls
    t = torch.full((x.shape[0],), t_value)
    with torch.no_grad():
        G = torch.cat([model(xb, tb)
                       for xb, tb in zip(x.split(2048), t.split(2048))])
    # G is (N, d, S) with the current token's slot zeroed, so summing over
    # tokens IS the flip score for the binary vocabulary; S = -x * flip.
    S = -(x * G.sum(-1)).reshape(-1)                              # (N*d,)

    h = (x @ A)                                                   # (N, d)
    c_hollow = target.composition_fraction(x).unsqueeze(1) - (x + 1) / (2 * d)
    penalty_offset = c_hollow - c_star
    # hole-excluded bond sum: x^T A x minus site i's two-sided contribution
    bond_sum = (x @ A * x).sum(1, keepdim=True)
    B_hollow = bond_sum - 2 * x * h                               # (N, d)

    h_f, p_f, B_f = h.reshape(-1), penalty_offset.reshape(-1), B_hollow.reshape(-1)
    ones = torch.ones_like(S)
    designs = {
        "r2_field": torch.stack([ones, h_f], 1),
        "r2_penalty": torch.stack([ones, p_f], 1),
        "r2_channel": torch.stack([ones, h_f, p_f], 1),
        "r2_quadratic": torch.stack(
            [ones, h_f, p_f, h_f**2, p_f**2, h_f * p_f], 1),
    }
    # p~_t weights per STATE, repeated per site so each state's d rows share
    # its weight. exp-normalised in float64 to survive lambda*d swings.
    log_pt = ((1 - t_value) * target.base_log_eta(x)
              + t_value * target.log_prob(x)).double()
    w_state = (log_pt - log_pt.logsumexp(0)).exp()
    w = w_state.repeat_interleave(d).float()

    out = {"run": Path(run_dir).name, "t": t_value, "S_std": float(S.std()),
           "lookup_protocol": "site_bitmask_heldout_sse_v2"}
    for name, design in designs.items():
        out[name] = r_squared(S, design)
        out[name + "_wt"] = r_squared(S, design, weights=w)

    # Held-out cell means include local nonlinearity and optional count.
    keys_local, keys_count = local_lookup_keys(x, A)
    fit_mask = torch.rand(
        S.shape[0], generator=torch.Generator().manual_seed(0)) < 0.5
    out["r2_any_local"], _ = lookup_r_squared(S, keys_local, fit_mask)
    out["r2_any_local_count"], residual = lookup_r_squared(
        S, keys_count, fit_mask)
    held = ~fit_mask
    B_h, h_h = B_f[held], h_f[held]
    # Legacy JSON key retained; this is a residual association, not proof
    # that the residual is purely nonlocal.
    out["nonlocal_share_explained_by_B"] = r_squared(
        residual, torch.stack(
            [torch.ones_like(B_h), B_h, B_h**2, B_h * h_h, B_h**2 * h_h], 1))
    return out


COLUMNS = ("r2_field", "r2_penalty", "r2_channel", "r2_quadratic",
           "r2_any_local", "r2_any_local_count",
           "nonlocal_share_explained_by_B")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=REPO_ROOT / "results" / "02_constrained_soft")
    parser.add_argument("--configs", nargs="+", default=list(SPECIALISTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--t", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = []
    for config in args.configs:
        for seed in args.seeds:
            run_dir = latest_run_dir(args.results_dir, config, seed)
            if run_dir is None:
                continue
            for t_value in args.t:
                rows.append(analyse(run_dir, t_value))

    header = f"{'run':58s} {'t':>4s}" + "".join(
        f" {c.replace('r2_', '').replace('nonlocal_share_explained_by_B', 'res~B'):>12s}"
        for c in COLUMNS) + f" {'chan_wt':>8s}"
    print(header)
    for r in rows:
        print(f"{r['run'][:58]:58s} {r['t']:4.1f}" + "".join(
            f" {r[c]:12.3f}" for c in COLUMNS) + f" {r['r2_channel_wt']:8.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
