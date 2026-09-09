import glob
import re

import pandas as pd

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
