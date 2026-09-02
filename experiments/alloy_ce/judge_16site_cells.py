"""Judge the 16-site Cu-Au cells against exact enumeration (2^16 states).

Per cell and eval flavour: ESS fraction, two free-energy readings in eV/site
(F_lb = -mean log w / (beta d), paper Eq. 37 upper bound on F; F_is =
-logmeanexp log w / (beta d), the IS estimate) against the exact value, and
the composition mean. Free rung: exact = free-ensemble F and the IS-weighted
composition-marginal mass at x_Au = 0.25 / 0.5 (exact bimodal). Soft rung:
exact = the penalised target's own log Z (metrics.json). Hard rung: exact =
canonical slice F at the cell's composition (Sadigh A.6).
"""
import glob, itertools, json, math
from pathlib import Path
import torch
from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

K_B_EV = 8.617333262e-5
T, d = 500.0, 16
beta = 1.0 / (K_B_EV * T)
spec = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
states = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=d)), dtype=torch.float64)
energy = spec.energy(states)
n_au = ((states + 1) / 2).sum(1)
log_w_exact = -beta * energy
p_exact = torch.softmax(log_w_exact, 0)
marg_exact = torch.zeros(d + 1, dtype=torch.float64).index_add_(0, n_au.long(), p_exact)
F_free_exact = -torch.logsumexp(log_w_exact, 0).item() / beta / d
F_slice_exact = {c: -torch.logsumexp(log_w_exact[n_au == round(c * d)], 0).item() / beta / d
                 for c in (0.25, 0.5)}
print(f"exact at {T:.0f} K: F_free {F_free_exact:.4f} eV/site; F_slice(0.25) {F_slice_exact[0.25]:.4f}, "
      f"F_slice(0.5) {F_slice_exact[0.5]:.4f}; free marginal mass at n_Au=4,8: "
      f"{marg_exact[4]:.3f} {marg_exact[8]:.3f}")

def marginal(samples, log_w):
    n = ((samples + 1) / 2).sum(1).long()
    w = torch.softmax(log_w.double(), 0)
    weighted = torch.zeros(d + 1, dtype=torch.float64).index_add_(0, n, w)
    raw = torch.bincount(n, minlength=d + 1).double() / len(n)
    return weighted, raw

def free_energies(log_w):
    lw = log_w.double()
    F_lb = -lw.mean().item() / beta / d
    F_is = -(torch.logsumexp(lw, 0) - math.log(len(lw))).item() / beta / d
    return F_lb, F_is

rows = []
for run in sorted(glob.glob("results/*/*cuau16*")):
    name = Path(run).name.split("_2026")[0]
    for flavour in ("eval", "eval_ema"):
        mfile = Path(run) / flavour / "metrics.json"
        if not mfile.exists():
            continue
        m = json.load(open(mfile))
        lw = torch.load(Path(run) / flavour / "log_weights.pt")
        s = torch.load(Path(run) / flavour / "samples.pt")
        F_lb, F_is = free_energies(lw)
        if name.startswith("H2"):
            F_exact = F_slice_exact[m["target_composition"]]
        else:
            F_exact = m["free_energy_per_site_exact"]
            assert abs(F_lb - m["free_energy_per_site"]) < 1e-5, (name, F_lb, m["free_energy_per_site"])
        row = dict(cell=name, flavour=flavour, ess=m["ess_fraction"], F_lb=F_lb, F_is=F_is,
                   F_exact=F_exact, c_mean=m["composition_mean"])
        if name.startswith("A1"):
            wm, raw = marginal(s, lw)
            row.update(m4_w=wm[4].item(), m8_w=wm[8].item(), m4_raw=raw[4].item(), m8_raw=raw[8].item())
        rows.append(row)

print(f"{'cell':46s} {'flav':8s} {'ESS':>6s} {'F_lb':>8s} {'F_is':>8s} {'F_ex':>8s} {'dlb':>5s} {'dis':>5s} {'<c>':>6s}  m4_w m8_w | m4_raw m8_raw")
for r in rows:
    extra = ""
    if "m4_w" in r:
        extra = f"  {r['m4_w']:.3f} {r['m8_w']:.3f} | {r['m4_raw']:.3f} {r['m8_raw']:.3f}"
    print(f"{r['cell']:46s} {r['flavour']:8s} {r['ess']:6.3f} {r['F_lb']:8.4f} {r['F_is']:8.4f} {r['F_exact']:8.4f} "
          f"{1e3*(r['F_lb']-r['F_exact']):5.1f} {1e3*(r['F_is']-r['F_exact']):5.1f} {r['c_mean']:6.3f}{extra}")
print("(dlb, dis = F_lb - exact, F_is - exact in meV/site)")
