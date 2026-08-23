"""Judge the exterior-vs-interior separation grid (s53 launch, 23-Aug-2026).

One row per run: raw ESS/N, EMA ESS/N (d64 only), summed inner-step time
(``wall_clock_step_s``, loss update ONLY -- never end-to-end), and the
Modal-volume wall clock (spawn tag -> final checkpoint mtime, minute
resolution).  Bands are the FROZEN ones in configs.py; the 4x4 band is a
table, not a gate.
"""
import csv, glob, json, os, re, sys
import torch
from collections import defaultdict
from datetime import datetime

ROOT = "results/03_hard"
ARMS = ["iv", "ivb", "mab", "fib", "fatt", "fimo2", "fmoatt", "fimo2ef", "mabef"]
# Exact-field channel (s54): parent arm and the FROZEN d64 bands (configs.py).
EF_BANDS = {"fimo2ef": ("fimo2", 0.78, 0.765, 0.750), "mabef": ("mab", 0.80, 0.785, 0.769)}
# Archived comparators (same recipe, earlier sessions) -- raw eval ESS/N.
D64_REF = {"MA (ma, 0707)": 0.7806, "interval (iv, 0708)": 0.6463,
           "fmo2 (0814)": 0.745, "fab8 (0813)": None}
D16_REF_DIRS = {"ma": "letf_ma_10k_seed*20260708*",
                "fab8": "letf_fab8_10k_seed*facgate",
                "fmo2": "letf_fmo2_10k_seed*facgate",
                "fbil": "letf_fbil_10k_seed*facgate"}


def ess(run, sub="eval"):
    p = os.path.join(run, sub, "metrics.json")
    return json.load(open(p))["ess_fraction"] if os.path.exists(p) else None


def inner_seconds(run):
    with open(os.path.join(run, "training_log.csv")) as f:
        return sum(float(r["wall_clock_step_s"]) for r in csv.DictReader(f)
                   if r["wall_clock_step_s"] not in ("", "nan"))


def wall_minutes(run, mtimes):
    tag = re.search(r"_(\d{8}-\d{6})$", run).group(1)
    start = datetime.strptime(tag, "%Y%m%d-%H%M%S")
    end = mtimes.get(os.path.basename(run))
    return (end - start).total_seconds() / 60 if end else None


def load_mtimes(path):
    """Optional json from `modal volume ls --json` of each run's checkpoints."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for e in json.load(open(path)):
        if e["Filename"].endswith("final.pt"):
            out[e["Filename"].split("/")[0]] = datetime.strptime(
                e["Created/Modified"][:16], "%Y-%m-%d %H:%M")
    return out


def learned_gain(run):
    """(gain_constant, gain_slope) of the exact-field channel, gain(t) = a + b t."""
    sd = torch.load(os.path.join(run, "checkpoints", "final.pt"), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    return float(sd["gain_constant"]), float(sd["gain_slope"])


def band_ef(raw, arm):
    _, strong, meaningful, null = EF_BANDS[arm]
    verdict = "STRONG" if raw >= strong else "MEANINGFUL" if raw >= meaningful else "NULL" if raw <= null else "between"
    near = min(abs(raw - e) for e in (strong, meaningful, null)) < 0.02
    return verdict + (" (within 0.02 of an edge: FP caveat)" if near else "")


def band_d16(vals):
    ok = sorted(vals, reverse=True)
    if len(ok) >= 2 and ok[1] >= 0.955: return "PARITY"
    if len(ok) >= 2 and ok[1] >= 0.94: return "MEANINGFUL"
    return "NULL" if ok[1] < 0.92 else "between"


def main():
    mtimes = load_mtimes(sys.argv[1] if len(sys.argv) > 1 else None)
    print("== 4x4 (d16), raw ESS/N by seed; band = 2/3 seeds ==")
    for sigma in ("s223", "s010"):
        print(f"-- sigma {sigma}")
        for arm in ARMS:
            runs = sorted(glob.glob(f"{ROOT}/H2_d16_c50_{sigma}_letf_{arm}_10k_seed*_20260823-*"))
            vals = [ess(r) for r in runs]
            secs = [inner_seconds(r) for r in runs]
            print(f"  {arm:7s} " + " ".join(f"{v:.3f}" for v in vals)
                  + f"  -> {band_d16(vals):10s} inner {sum(secs)/len(secs):5.0f}s/run")
        for name, pat in D16_REF_DIRS.items():
            runs = sorted(glob.glob(f"{ROOT}/H2_d16_c50_{sigma}_{pat}"))
            vals = [v for v in (ess(r) for r in runs) if v is not None]
            if vals:
                print(f"  ref {name:4s}" + " ".join(f"{v:.3f}" for v in vals))
    print("\n== d64 seed 42: raw / EMA ESS/N, inner-step total, Modal wall clock ==")
    for arm in ["ivb", "mab", "fib", "fatt", "fimo2", "fimo2ef", "mabef"]:
        run = glob.glob(f"{ROOT}/H2_d64_c50_s223_letf_{arm}_50k_curr_seed42_20260823-*")[0]
        wm = wall_minutes(run, mtimes)
        extra = ""
        if arm in EF_BANDS:
            a, b = learned_gain(run)
            extra = f"  gain a={a:.4f} b={b:.4f}  -> {band_ef(ess(run), arm)} vs {EF_BANDS[arm][0]}"
        print(f"  {arm:7s} raw {ess(run):.3f}  ema {ess(run,'eval_ema'):.3f}  "
              f"inner {inner_seconds(run)/60:5.1f} min  wall {wm if wm is None else f'{wm:.0f}'} min{extra}")
    for name, pat in {"MA 0707": "letf_ma_50k_curr_seed42_20260707*",
                      "fmo2 0814": "letf_fmo2_50k_curr_seed42_20260814-fmo2-d64",
                      "fab8 0813": "letf_fab8_50k_curr_seed42_20260813*"}.items():
        hits = glob.glob(f"{ROOT}/H2_d64_c50_s223_{pat}")
        if not hits:
            print(f"  ref {name:10s} (not archived locally)"); continue
        run = hits[0]
        e, ee = ess(run), ess(run, "eval_ema")
        print(f"  ref {name:10s} raw {e if e is None else f'{e:.3f}'}  ema {ee if ee is None else f'{ee:.3f}'}  "
              f"inner {inner_seconds(run)/60:5.1f} min")
    print("  frozen refs:", D64_REF)


if __name__ == "__main__":
    main()
