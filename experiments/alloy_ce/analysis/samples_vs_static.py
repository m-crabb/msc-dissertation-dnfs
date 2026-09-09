"""Desk probe: sampled Cu-Au 16-site energies against the enumerated target.

The 16-site cell is enumerable (2^16 states), so <beta E> under the exact
target, under the uniform slice and under the self-normalised samples can
all be printed side by side with Var(log w) and the unique-state count.

Run: python -m experiments.alloy_ce.analysis.samples_vs_static
"""

import glob
import itertools
import re

import torch

from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec


def main() -> None:
    K_B = 8.617333262e-5
    d = 16
    beta = 1 / (K_B * 500)
    spec = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
    states = torch.tensor(
        list(itertools.product([-1.0, 1.0], repeat=d)), dtype=torch.float64
    )
    E = spec.energy(states)
    n = ((states + 1) / 2).sum(1)

    def ref(mask):
        e = beta * E[mask]
        p = torch.softmax(-e, 0)
        return e.mean().item(), (p * e).sum().item(), e.var().item()

    refs = {
        "A1": ref(torch.ones(len(E), dtype=torch.bool)),
        "c25": ref(n == 4),
        "c50": ref(n == 8),
    }
    print(
        f"{'cell':26s} {'<bE>samp':>9s} {'<bE>unif':>9s} {'<bE>targ':>9s} "
        f"{'Var(logw)':>9s} {'Var_static':>10s} {'uniq':>5s}"
    )
    for r in sorted(glob.glob("results/0*/*cuau16*")):
        s = torch.load(r + "/eval/samples.pt").double()
        lw = torch.load(r + "/eval/log_weights.pt").double()
        key = "A1" if "A1_" in r else ("c25" if "c25" in r else "c50")
        e = beta * spec.energy(s)
        u, t, v = refs[key]
        name = re.sub(r"_T500.*seed", "_s", r.split("/")[-1]).replace(
            "_20260902-cuau16", ""
        )
        print(
            f"{name:26s} {e.mean():9.2f} {u:9.2f} {t:9.2f} {lw.var():9.2f} {v:10.2f} "
            f"{len(torch.unique(s, dim=0)):5d}"
        )


if __name__ == "__main__":
    main()
