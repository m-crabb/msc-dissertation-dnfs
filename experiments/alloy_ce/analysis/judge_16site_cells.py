"""Judge the 16-site Cu-Au cells against exact enumeration (2^16 states).

Per cell and eval flavour: ESS fraction, two free-energy readings in eV/site
(F_lb = -mean log w / (beta d), paper Eq. 37 upper bound on F; F_is =
-logmeanexp log w / (beta d), the IS estimate) against the exact value, and
the composition mean. Free rung: exact = free-ensemble F and the IS-weighted
composition-marginal mass at x_Au = 0.25 / 0.5 (exact bimodal). Soft rung:
exact = the penalised target's own log Z (metrics.json). Hard rung: exact =
canonical slice F at the cell's composition (Sadigh A.6). Amortised hard
cells (`camort`): the eval draws mix five slices, each row's weight is exact
against its OWN slice (base_log_eta read off x), so the slice's rows alone are
that slice's importance sample: one row per slice, F_is = -logmeanexp over the
slice's rows, ESS the slice's own, no correction for the uniform slice choice.
"""

import glob
import itertools
import json
import math
from pathlib import Path

import torch

from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

K_B_EV = 8.617333262e-5
d = 16
spec = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
states = torch.tensor(
    list(itertools.product([-1.0, 1.0], repeat=d)), dtype=torch.float64
)
energy = spec.energy(states)
n_au = ((states + 1) / 2).sum(1)


def exact_at(T):
    """Exact free-ensemble F, per-slice F(c) and the composition marginal at temperature
    T (cached)."""
    beta = 1.0 / (K_B_EV * T)
    log_w_exact = -beta * energy
    p_exact = torch.softmax(log_w_exact, 0)
    marg = torch.zeros(d + 1, dtype=torch.float64).index_add_(0, n_au.long(), p_exact)
    F_free = -torch.logsumexp(log_w_exact, 0).item() / beta / d
    F_slice = {
        n / d: -torch.logsumexp(log_w_exact[n_au == n], 0).item() / beta / d
        for n in range(d + 1)
    }
    return beta, F_free, F_slice, marg


def cell_temperature(run):
    cfg = json.load(open(Path(run) / "config.json"))
    return round(1.0 / (2.0 * K_B_EV * cfg["curriculum"]["stages"][-1]["sigma"]))


_, F_free_exact, F_slice_exact, marg_exact = exact_at(500.0)
print(
    f"exact at 500 K: F_free {F_free_exact:.4f} eV/site; "
    f"F_slice(0.25) {F_slice_exact[0.25]:.4f}, "
    f"F_slice(0.5) {F_slice_exact[0.5]:.4f}; free marginal mass at n_Au=4,8: "
    f"{marg_exact[4]:.3f} {marg_exact[8]:.3f}"
)


def marginal(samples, log_w):
    n = ((samples + 1) / 2).sum(1).long()
    w = torch.softmax(log_w.double(), 0)
    weighted = torch.zeros(d + 1, dtype=torch.float64).index_add_(0, n, w)
    raw = torch.bincount(n, minlength=d + 1).double() / len(n)
    return weighted, raw


def free_energies(log_w, beta):
    lw = log_w.double()
    F_lb = -lw.mean().item() / beta / d
    F_is = -(torch.logsumexp(lw, 0) - math.log(len(lw))).item() / beta / d
    return F_lb, F_is


rows = []
for run in sorted(glob.glob("results/*/*cuau16*")):
    name = Path(run).name.replace("_2026", " ")
    T = cell_temperature(run)
    beta, F_free_exact, F_slice_exact, marg_exact = exact_at(T)
    for flavour in ("eval", "eval_ema"):
        mfile = Path(run) / flavour / "metrics.json"
        if not mfile.exists():
            continue
        m = json.load(open(mfile))
        lw = torch.load(Path(run) / flavour / "log_weights.pt")
        s = torch.load(Path(run) / flavour / "samples.pt")
        if "camort" in name:
            for n in sorted(set(((s + 1) / 2).sum(1).long().tolist())):
                on_slice = ((s + 1) / 2).sum(1).long() == n
                lw_slice = lw[on_slice].double()
                F_lb, F_is = free_energies(lw_slice, beta)
                ess = (torch.softmax(lw_slice, 0) ** 2).sum().reciprocal().item() / len(
                    lw_slice
                )
                rows.append(
                    dict(
                        cell=f"{name}@n{n}",
                        flavour=flavour,
                        ess=ess,
                        F_lb=F_lb,
                        F_is=F_is,
                        F_exact=F_slice_exact[n / d],
                        c_mean=n / d,
                    )
                )
            continue
        F_lb, F_is = free_energies(lw, beta)
        if name.startswith("H2"):
            F_exact = F_slice_exact[m["target_composition"]]
        else:
            F_exact = m["free_energy_per_site_exact"]
            assert abs(F_lb - m["free_energy_per_site"]) < 1e-5, (
                name,
                F_lb,
                m["free_energy_per_site"],
            )
        row = dict(
            cell=f"{name}@{T}K",
            flavour=flavour,
            ess=m["ess_fraction"],
            F_lb=F_lb,
            F_is=F_is,
            F_exact=F_exact,
            c_mean=m["composition_mean"],
        )
        if name.startswith("A1"):
            wm, raw = marginal(s, lw)
            row.update(
                m4_w=wm[4].item(),
                m8_w=wm[8].item(),
                m4_raw=raw[4].item(),
                m8_raw=raw[8].item(),
            )
        rows.append(row)

print(
    f"{'cell':62s} {'flav':8s} {'ESS':>6s} {'F_lb':>8s} {'F_is':>8s} {'F_ex':>8s} "
    f"{'dlb':>5s} {'dis':>5s} {'<c>':>6s}  m4_w m8_w | m4_raw m8_raw"
)
for r in rows:
    extra = ""
    if "m4_w" in r:
        extra = (
            f"  {r['m4_w']:.3f} {r['m8_w']:.3f} | {r['m4_raw']:.3f} {r['m8_raw']:.3f}"
        )
    print(
        f"{r['cell']:62s} {r['flavour']:8s} {r['ess']:6.3f} {r['F_lb']:8.4f} "
        f"{r['F_is']:8.4f} {r['F_exact']:8.4f} "
        f"{1e3 * (r['F_lb'] - r['F_exact']):5.1f} "
        f"{1e3 * (r['F_is'] - r['F_exact']):5.1f} {r['c_mean']:6.3f}{extra}"
    )
print("(dlb, dis = F_lb - exact, F_is - exact in meV/site)")
