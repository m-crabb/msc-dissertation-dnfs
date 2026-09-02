"""Report the CPU desk-check cells: train-ESS trajectory, eval ESS, samples vs the static (identity-flow) reference."""
import glob, itertools, json, re, sys, torch, pandas as pd
from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec
K_B = 8.617333262e-5; d = 16
spec = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
states = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=d)), dtype=torch.float64)
E = spec.energy(states); n_au = ((states + 1) / 2).sum(1)
root = sys.argv[1] if len(sys.argv) > 1 else "results/03_hard/*cuau16-desk"
for run in sorted(glob.glob(root)):
    cfg = json.load(open(run + "/config.json")); c = cfg["ising"]["target_composition"]
    T_final = 1 / (2 * K_B * cfg["curriculum"]["stages"][-1]["sigma"]); beta = 1 / (K_B * T_final)
    mask = n_au == round(c * d); e_ref = beta * E[mask]; p = torch.softmax(-e_ref, 0)
    df = pd.read_csv(run + "/training_log.csv"); ess = df.dropna(subset=["ess"]).set_index("step")["ess"]
    m = json.load(open(run + "/eval/metrics.json"))
    s = torch.load(run + "/eval/samples.pt").double(); lw = torch.load(run + "/eval/log_weights.pt").double()
    e_s = beta * spec.energy(s)
    print(f"== {run.split('/')[-1]}  final T {T_final:.0f} K")
    print("   train ESS:", " ".join(f"{int(k)}:{int(v)}" for k, v in ess.items() if k % 500 == 0))
    print(f"   eval ESS {m['ess_fraction']:.3f}  jumps/site {m['jumps_per_site_state_changing']:.3f}  <bE> samples {e_s.mean():.2f} vs uniform {e_ref.mean():.2f} target {(p*e_ref).sum():.2f}  Var(logw) {lw.var():.2f} vs static {e_ref.var():.2f}  unique {len(torch.unique(s, dim=0))}")
