"""Exact 16-site Cu-Au slice statistics at each curriculum temperature."""

import itertools

import torch

from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

K_B = 8.617333262e-5
d = 16
spec = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
states = torch.tensor(
    list(itertools.product([-1.0, 1.0], repeat=d)), dtype=torch.float64
)
E = spec.energy(states)
n_au = ((states + 1) / 2).sum(1)
sel = {
    "free": torch.ones(len(E), dtype=torch.bool),
    "c=0.25": n_au == 4,
    "c=0.5": n_au == 8,
}
temps = [1200, 800, 600, 500]


def stats(mask, T):
    beta = 1 / (K_B * T)
    lw = -beta * E[mask]
    p = torch.softmax(lw, 0)
    M = mask.sum().item()
    effN = 1 / (p**2).sum().item()
    top = p.sort(descending=True).values
    var_static = (
        (beta * E[mask]).var().item()
    )  # identity-flow log-weight variance under the uniform base
    return dict(
        M=M,
        effN=effN,
        static_ess=effN / M,
        top1=top[0].item(),
        top4=top[:4].sum().item(),
        top16=top[:16].sum().item(),
        var_static=var_static,
    )


print(
    f"{'slice':8s} {'T':>5s} {'M':>6s} {'effN':>8s} {'staticESS':>9s} {'top1':>6s} {'top4':>6s} {'top16':>6s} {'Var[bE]':>8s}"
)
prev = {}
for name, mask in sel.items():
    for T in temps:
        s = stats(mask, T)
        print(
            f"{name:8s} {T:5d} {s['M']:6d} {s['effN']:8.1f} {s['static_ess']:9.4f} {s['top1']:6.3f} {s['top4']:6.3f} {s['top16']:6.3f} {s['var_static']:8.2f}"
        )
    # KL between consecutive ladder temperatures on this support
    for Ta, Tb in zip(temps[:-1], temps[1:]):
        pa = torch.softmax(-E[mask] / (K_B * Ta), 0)
        pb = torch.softmax(-E[mask] / (K_B * Tb), 0)
        kl_ab = (pa * (pa.log() - pb.log())).sum().item()
        kl_ba = (pb * (pb.log() - pa.log())).sum().item()
        # ESS fraction if the proposal were exact p_Ta and the target p_Tb
        ess = 1 / (pa * (pb / pa) ** 2).sum().item() / 1.0
        print(
            f"   {name} {Ta}->{Tb}: KL(hot||cold) {kl_ab:.2f}  KL(cold||hot) {kl_ba:.2f}  ESS_frac(prop=hot,target=cold) {1 / ((pb**2 / pa).sum().item()):.4f}"
        )
# swap energy changes on each slice at 800 K under the target: how stiff is the landscape
for name in ("c=0.25", "c=0.5"):
    mask = sel[name]
    x = states[mask]
    beta = 1 / (K_B * 800)
    p = torch.softmax(-beta * E[mask], 0)
    idx = torch.multinomial(p, 2000, replacement=True)
    dE = spec.swap_energy_change(x[idx])  # (B, d, d)
    xi = x[idx]
    valid = xi[:, :, None] != xi[:, None, :]
    q = (beta * dE[valid]).abs()
    print(
        f"{name} 800K target-weighted |beta dE_swap| over valid swaps: median {q.median():.2f} p90 {q.quantile(0.9):.2f} max {q.max():.2f}; frac downhill {(beta * dE[valid] < 0).float().mean():.3f}"
    )
