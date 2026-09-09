"""Desk probe: training-ESS every 500 steps for every Cu-Au 16-site run.

Run: python -m experiments.alloy_ce.analysis.train_ess_trajectory
"""

import glob
import re

import pandas as pd


def main() -> None:
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    rows = {}
    for d in sorted(glob.glob("results/0*/*cuau16*")):
        df = pd.read_csv(d + "/training_log.csv")
        e = df.dropna(subset=["ess"]).set_index("step")["ess"]
        name = re.sub(r"_T500.*seed", "_s", d.split("/")[-1]).replace(
            "_20260902-cuau16", ""
        )
        rows[name] = e[e.index % 500 == 0].round(0).astype(int)
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
