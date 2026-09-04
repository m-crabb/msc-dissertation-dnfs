"""Turn a `modal_app::bench` stdout log into the JSON artefact
tab:head-cost-ladder was missing (its method comment: "No JSON artefact in
the repo: harness stdout only").

Each profile_swap row prints one `mode=... head_kind=... d=... batch=...`
header followed by `<name> median <s>s min <s>s peak_mem <GB> GB` lines;
this pairs them into records keyed by the header's flags so a table cell can
be traced to the exact configuration that produced it.

    python -m experiments.constrained_hard_03.analysis.parse_bench_log \
        <log> <out.json>
"""
import json
import re
import sys

HEADER = re.compile(r"^mode=(\S+) (.*)$")
TIMING = re.compile(
    r"^(\S.*?)\s+median\s+([\d.]+)s\s+min\s+([\d.]+)s\s+peak_mem\s+([\d.]+) GB"
)


def parse(lines):
    rows, current = [], None
    for line in lines:
        header = HEADER.match(line.strip())
        if header:
            flags = dict(pair.split("=", 1) for pair in header.group(2).split())
            current = {"mode": header.group(1), **flags, "timings": {}}
            rows.append(current)
            continue
        timing = TIMING.match(line.strip())
        if timing and timing.group(1).startswith("gfn_"):
            # The GFN modes print no `mode=` header: the timing name carries
            # the configuration (gfn_<mode>_<objective>_d<sites>_B<batch>).
            current = {"mode": timing.group(1).strip(), "timings": {}}
            rows.append(current)
        if timing and current is not None:
            current["timings"][timing.group(1).strip()] = {
                "median_s": float(timing.group(2)),
                "min_s": float(timing.group(3)),
                "peak_gb": float(timing.group(4)),
            }
    return rows


if __name__ == "__main__":
    log_path, out_path = sys.argv[1:3]
    with open(log_path) as log:
        rows = parse(log)
    with open(out_path, "w") as out:
        json.dump(rows, out, indent=1)
    print(f"{len(rows)} rows -> {out_path}")
