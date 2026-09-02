import pandas as pd, glob, sys, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.float_format", lambda v: f"{v:.3g}")
cells = {
 "hard_c25": "results/03_hard/H2_cuau16_c25_*seed42*",
 "hard_c50": "results/03_hard/H2_cuau16_c50_*seed42*",
 "free":     "results/02_constrained_soft/A1_cuau16_*seed42*",
 "soft_c25": "results/02_constrained_soft/S2_cuau16_c25_*seed42*",
 "soft_c50": "results/02_constrained_soft/S2_cuau16_c50_*seed42*",
}
windows = [(0,100),(400,500),(2000,2499),(2500,2520),(2520,2600),(2600,2800),(2800,3500),(4900,4999),(5000,5100),(5100,5500),(7400,7499),(7500,7600),(7600,8000),(9900,9999)]
cols_common = ["loss","ess","var_dt_log_p_tilde","var_estimator_integrand","grad_norm","log_ratio_clamp_frac","rollout_resample_events"]
extra = {"hard":["cv_var_ratio","rate_pair_mean","rate_pair_p99","lambda_dt_clipped_frac","lambda_dt_p99","proposal_drop_frac","events_per_site_per_step","c_t_offset_rms","grad_sqnorm_slice_mean"],
         "soft":["rate_site_mean","rate_site_p99","flip_prob_site_p99","flip_prob_clipped_frac","log_ratio_p99","composition_current","composition_half_width"]}
for name, pat in cells.items():
    d = sorted(glob.glob(pat))[0]
    df = pd.read_csv(d+"/training_log.csv")
    kind = "hard" if name.startswith("hard") else "soft"
    cols = cols_common + extra[kind]
    rows = []
    for a,b in windows:
        w = df[(df.step>=a)&(df.step<=b)]
        rows.append(pd.Series({c: w[c].mean() for c in cols if c in w}, name=f"{a}-{b}"))
    print("=====", name, d.split('/')[-1])
    print(pd.DataFrame(rows).T)
