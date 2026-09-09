"""Offline characterisation of a warm (spatially ordered) base for the
fixed-composition sampler, from archived Kawasaki reference draws.

Part of the warm-base line, retained as a record but not carried forward
(see warm_base_reference.py).

The current base is uniform on the c = N_A/d slice, so log eta is constant
on the reachable set and its nearest-neighbour correlation is exactly
-1/(d-1) by exchangeability.  A warm base moves that value positive, which
shortens the transport the sampler must supply and shrinks the drive
variance it must fit.  This script measures both, for the block-occupancy
family B(b, w), against the certified Kawasaki reference sets.

The quantities:

  log rho(x) = sigma * x^T A x        the unnormalised target (Eq. 4 path end)
  log eta(x)                          the base's exact log-density
  D(x)       = log rho(x) - log eta(x)

  nn_base        = E_eta[C(x)]           C(x) = x^T A x / sum(A)
                   -> feeds the transport requirement 2 (C_target - C_base),
                      which is an exact accounting identity for how far the
                      mean bond correlation has to move.
  Var[log eta]   under p_1               the cost side of the trade
  corr(log rho, log eta) under p_1       the benefit side
  Var_{p_1}[D]   = Var[log rho] - 2 Cov[log rho, log eta] + Var[log eta]
                   -> the drive variance at t = 1; the warm base helps iff
                      Var[log eta] < 2 Cov[log rho, log eta].
  Delta c        = c_1 - c_0 = E_{p_1}[D] - E_{p_0}[D],  p_0 = eta
                   -> exactly int_0^1 Var_{p_t}[D] dt, since p_t is a
                      one-parameter exponential family in t with sufficient
                      statistic D, so c_t = d/dt log Z_t = E_{p_t}[D] and
                      d c_t/dt = Var_{p_t}[D] >= 0.  Endpoint evaluation gives
                      the integral exactly and says nothing about its shape.

All variances are population variances under the stated law (ddof = 0),
which is what the identities above refer to.

Run:  pixi run -e default python warm_base_offline_table.py [--sigma 0.223]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from warm_base_reference import (
    BlockOccupancyBase,
    UniformSliceBase,
    quadratic_form,
    torus_adjacency,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
D256_SET = REPO_ROOT / "results/kawasaki_ref_d256_sc_s223"
D64_SET = REPO_ROOT / "results/kawasaki_probe/reference/sc"

# The doc's own numbers, for the side-by-side.  Order:
# (nn_base, Var[log eta], corr, Var_p1[D], ratio, Delta c); None = "-" in the doc.
DOC_TABLE = {
    (64, "uniform"): (-0.017, 0.0, None, 25.2, 1.00, 27.2),
    (64, "B(2,w) K=1"): (None, 13.1, 0.635, 15.2, 0.61, None),
    (64, "B(2,w) K=4"): (0.176, 7.9, 0.850, 9.1, 0.36, 15.8),
    (64, "B(4,w) K=16"): (0.071, 10.4, 0.787, 10.1, 0.40, 21.2),
    (256, "uniform"): (-0.005, 0.0, None, 170.2, 1.00, 135.3),
    (256, "B(2,w) K=1"): (None, 99.0, 0.806, 60.0, 0.35, None),
    (256, "B(2,w) K=4"): (0.268, 73.3, 0.917, 38.7, 0.23, 72.6),
    (256, "B(4,w) K=16"): (0.264, 118.9, 0.877, 39.7, 0.23, 72.7),
}

N_BASE_DRAWS = 40_000
SUPPLY_EDGE_UNITS_PER_SITE = 0.93  # the adversarial panel's measured supply


def load_d256() -> np.ndarray:
    import torch

    return torch.load(D256_SET / "samples.pt").numpy().astype(np.int8)


def load_d64(thin: int = 28) -> tuple[np.ndarray, dict]:
    """Pool the 8 archived 8x8 non-local Kawasaki chains.

    Burn-in: drop the first half of every chain -- the discard rule the probe's
    own `reference_summary.json` applies to its frozen moments.  Thinning: the
    snapshots are already every 10 sweeps and the lag-1 autocorrelation of
    x^T A x on the post-discard half is ~0.02, so any thinning is a safety
    factor rather than a necessity; stride 28 is chosen to land on ~14 000
    draws, matching the size of the (lost) twin the doc used.
    """
    chains, meta = [], []
    for path in sorted(D64_SET.glob("chain_*/snapshots.npz")):
        z = np.load(path)
        spins = z["spins"]
        kept = spins[spins.shape[0] // 2 :][::thin]
        chains.append(kept)
        meta.append({"chain": path.parent.name, "kept": int(kept.shape[0])})
    return np.concatenate(chains), {"chains": meta, "thin": thin}


def build_bases(reference: np.ndarray, side: int, n_up: int):
    yield "uniform", UniformSliceBase(side, n_up)
    yield "B(2,w) K=1", BlockOccupancyBase.fit(reference, side, 2, n_offsets=1)
    yield "B(2,w) K=4", BlockOccupancyBase.fit(reference, side, 2, n_offsets=4)
    yield "B(4,w) K=16", BlockOccupancyBase.fit(reference, side, 4, n_offsets=16)


def measure(reference: np.ndarray, side: int, sigma: float, seed: int) -> dict:
    d = side * side
    A = torus_adjacency(side)
    n_up = int((reference[0] > 0).sum())
    assert ((reference > 0).sum(axis=1) == n_up).all(), "reference is off-slice"

    quad_ref = quadratic_form(reference, A)
    log_rho_ref = sigma * quad_ref
    nn_target = quad_ref.mean() / A.sum()
    nn_target_se = quad_ref.std(ddof=1) / np.sqrt(len(quad_ref)) / A.sum()

    rows = {}
    var_uniform = None
    for name, base in build_bases(reference, side, n_up):
        rng = np.random.default_rng(seed)
        draws = base.sample(N_BASE_DRAWS, rng)
        quad_base = quadratic_form(draws, A)
        nn_base = quad_base.mean() / A.sum()
        nn_base_se = quad_base.std(ddof=1) / np.sqrt(N_BASE_DRAWS) / A.sum()

        log_eta_ref = base.log_density(reference)
        var_log_eta = log_eta_ref.var()
        cov = np.cov(log_rho_ref, log_eta_ref, ddof=0)[0, 1]
        corr = (
            np.corrcoef(log_rho_ref, log_eta_ref)[0, 1] if var_log_eta > 1e-12 else None
        )
        drive_ref = log_rho_ref - log_eta_ref
        var_drive = drive_ref.var()
        if var_uniform is None:
            var_uniform = var_drive

        # c_0 = E_{p_0}[D] with p_0 = eta itself, so both terms on base draws.
        log_eta_base = base.log_density(draws)
        drive_base = sigma * quad_base - log_eta_base
        delta_c = drive_ref.mean() - drive_base.mean()
        delta_c_se = np.sqrt(
            drive_ref.var(ddof=1) / len(drive_ref)
            + drive_base.var(ddof=1) / N_BASE_DRAWS
        )
        # Decomposition of (3.3):  Delta c = [E_p1 log rho - E_p0 log rho]
        #                                  - [E_p1 log eta - E_p0 log eta].
        # The first bracket is sigma * 4d * (nn_target - nn_base) -- pure
        # energy, the same object as the transport requirement.  The second is
        # the base-entropy term, identically zero for the uniform base and
        # nonzero for every warm one.  Reported separately because the doc's
        # Delta c column tracks the first bracket alone.
        delta_c_energy = sigma * (quad_ref.mean() - quad_base.mean())
        delta_c_eta = log_eta_ref.mean() - log_eta_base.mean()

        rows[name] = dict(
            nn_base=nn_base,
            nn_base_se=nn_base_se,
            var_log_eta=var_log_eta,
            corr=corr,
            var_drive=var_drive,
            ratio=var_drive / var_uniform,
            delta_c=delta_c,
            delta_c_se=delta_c_se,
            delta_c_energy=delta_c_energy,
            delta_c_eta=delta_c_eta,
            two_cov=2 * cov,
            trade_margin=(2 * cov / var_log_eta if var_log_eta > 1e-12 else None),
        )
    return dict(
        d=d,
        side=side,
        sigma=sigma,
        n_up=n_up,
        n_reference=len(reference),
        nn_target=nn_target,
        nn_target_se=nn_target_se,
        var_log_rho=log_rho_ref.var(),
        rows=rows,
    )


def agreement(measured: float | None, doc: float | None) -> str:
    if doc is None or measured is None:
        return "n/a"
    scale = max(abs(doc), 1e-9)
    rel = abs(measured - doc) / scale
    absdiff = abs(measured - doc)
    if rel <= 0.02 or absdiff <= 0.005:
        return "AGREE"
    if rel <= 0.10:
        return "close"
    return "DISAGREE"


def print_block(res: dict) -> None:
    side, d, sigma = res["side"], res["d"], res["sigma"]
    print(
        f"\n=== {side}x{side} (d={d}), sigma={sigma}, N_ref={res['n_reference']}, "
        f"nn_target={res['nn_target']:.4f} +/- {res['nn_target_se']:.4f}, "
        f"Var[log rho]={res['var_log_rho']:.1f} ==="
    )
    head = (
        f"{'base':<12}|{'nn_base':>18}|{'Var[log eta]':>16}|{'corr':>14}|"
        f"{'Var_p1[D]':>16}|{'ratio':>12}|{'Delta c':>16}"
    )
    print(head)
    print("-" * len(head))
    for name, row in res["rows"].items():
        doc = DOC_TABLE[(d, name)]

        def cell(measured, doc_value, fmt):
            tag = agreement(measured, doc_value)
            docs = "-" if doc_value is None else format(doc_value, fmt)
            meas = "-" if measured is None else format(measured, fmt)
            mark = {"AGREE": "=", "close": "~", "DISAGREE": "X", "n/a": " "}[tag]
            return f"{meas}/{docs}{mark}"

        print(
            f"{name:<12}|"
            f"{cell(row['nn_base'], doc[0], '.3f'):>18}|"
            f"{cell(row['var_log_eta'], doc[1], '.1f'):>16}|"
            f"{cell(row['corr'], doc[2], '.3f'):>14}|"
            f"{cell(row['var_drive'], doc[3], '.1f'):>16}|"
            f"{cell(row['ratio'], doc[4], '.2f'):>12}|"
            f"{cell(row['delta_c'], doc[5], '.1f'):>16}"
        )
    print("  (cell = recomputed/doc; = agree <=2%, ~ within 10%, X differs)")
    print("  Delta c decomposition:  full (3.3) = energy term - base-entropy term")
    for name, row in res["rows"].items():
        print(
            f"    {name:<12} full {row['delta_c']:7.2f} +/- {row['delta_c_se']:.2f}"
            f" = energy {row['delta_c_energy']:7.2f} - eta {row['delta_c_eta']:6.2f}"
            f"   (doc {DOC_TABLE[(res['d'], name)][5]})"
        )
    for name, row in res["rows"].items():
        if row["corr"] is None:
            continue
        print(
            f"  variance-trade [{name}]: Var[log eta] {row['var_log_eta']:.1f} "
            f"< 2 Cov {row['two_cov']:.1f}  -> margin x{row['trade_margin']:.2f}"
        )


def print_downstream(results: dict) -> None:
    print("\n=== downstream: transport requirement 2 (nn_target - nn_base) ===")
    print(
        f"{'d':>5}|{'base':<26}|{'nn_base':>9}|{'requirement':>12}|"
        f"{'cut vs uniform':>15}|{'supply/req @0.93':>18}"
    )
    for d, res in results.items():
        exact_uniform_nn = UniformSliceBase(
            res["side"], res["n_up"]
        ).exact_nn_correlation()
        exact_req = 2 * (res["nn_target"] - exact_uniform_nn)
        print(
            f"{d:>5}|{'uniform, exact -1/(d-1)':<26}|{exact_uniform_nn:>9.4f}|"
            f"{exact_req:>12.4f}|{'-':>15}|"
            f"{SUPPLY_EDGE_UNITS_PER_SITE / exact_req:>17.1%}"
        )
        for name, row in res["rows"].items():
            if name == "uniform":
                continue
            req = 2 * (res["nn_target"] - row["nn_base"])
            print(
                f"{d:>5}|{name:<26}|{row['nn_base']:>9.4f}|{req:>12.4f}|"
                f"{1 - req / exact_req:>14.1%}|"
                f"{SUPPLY_EDGE_UNITS_PER_SITE / req:>17.1%}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sigma",
        type=float,
        default=0.223,
        help="0.223 = the sigma both reference sets were generated "
        "at and every hard cell trains at; the doc quotes "
        "0.22305 (the configs.py sigma_c comment).",
    )
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--thin-d64", type=int, default=28)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    d64_ref, d64_meta = load_d64(args.thin_d64)
    d256_ref = load_d256()
    print(
        f"d64  reference: {d64_ref.shape} from {len(d64_meta['chains'])} chains, "
        f"thin={d64_meta['thin']}, post-burn-in half"
    )
    print(f"d256 reference: {d256_ref.shape} from {D256_SET.name}")

    results = {
        64: measure(d64_ref, 8, args.sigma, args.seed),
        256: measure(d256_ref, 16, args.sigma, args.seed),
    }
    for res in results.values():
        print_block(res)
    print_downstream(results)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
