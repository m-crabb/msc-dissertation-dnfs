"""Critical-coupling body tables projected from the two-coupling tables.

Decided 2026-09-07: the hard chapter prints sigma_c only in its main body.
The two-coupling tables (tab:eval-hard-{4x4,8x8,16x16,20x20}-full in the
appendix, emitted by house_table_<rung>.py --latex and then hand-annotated:
the 4x4 oracle dagger, the 20x20 bf16 twins omitted from the bold) are the
verified record; the body tables are a PROJECTION of them, never a second
emission, so the two can not drift. Two projections:

  --label <full-table label>       the sigma_c half of one table, six columns
  --ladder <label> <label> ...     the sigma_c halves of several rungs stacked
                                   into one table with a rung heading per block

Column convention of every full table: Method, then five sigma = 0.1
columns, then five sigma_c columns (ESS, dMag, dCorr, EW2, FLOP/es). The
24x24 table is sigma_c only already and passes through as its own six
columns. Bold is per column in the source, so dropping columns keeps the
sigma_c bold intact. Internal \\midrules of a rung become \\addlinespace in
the ladder; \\midrule separates rungs.

Run from the Overleaf tree:
  python body_tables_from_full.py --tex appendix.tex --label tab:eval-hard-8x8-full
  python body_tables_from_full.py --tex appendix.tex --ladder tab:eval-hard-16x16-full \\
      tab:eval-hard-20x20-full tab:eval-hard-24x24-full
"""

import argparse
import re
from pathlib import Path

SIGMA_C_COLUMNS = slice(6, 11)
HEADER = (
    "        Method & \\gls{ess}$\\uparrow$ & $\\Delta$Mag$\\downarrow$ & "
    "$\\Delta$Corr$\\downarrow$ & $\\mathrm{EW}_2\\downarrow$ & "
    "\\acrshort{flop}/es$\\downarrow$ \\\\"
)


def table_rows(tex, label):
    """The tabular body lines (data rows and rules) of the table carrying `label`."""
    # Anchor on the label and walk to the ENCLOSING table: a non-greedy match
    # from \begin{table} would span every earlier table in the file.
    at = tex.find(f"\\label{{{label}}}")
    assert at >= 0, f"no table labelled {label}"
    start = tex.rfind("\\begin{table}", 0, at)
    end = tex.find("\\end{table}", at)
    tabular = re.search(
        r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", tex[start:end], re.S
    )
    body = tabular.group(1)
    # Joined continuation lines: a row may wrap onto a second line before its \\.
    lines, buffer = [], ""
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        buffer = f"{buffer} {line}".strip() if buffer else line
        if line.endswith("\\\\") or "rule" in line:
            lines.append(buffer)
            buffer = ""
    # Drop the header block: everything up to and including the first \midrule.
    first_rule = next(i for i, line in enumerate(lines) if line == "\\midrule")
    return lines[first_rule + 1 :]


def sigma_c_row(row):
    """One data row cut to Method + the five sigma_c cells."""
    if "rule" in row or row.startswith("\\addlinespace"):
        return row
    cells = [c.strip() for c in row[: -len("\\\\")].split("&")]
    assert len(cells) in (6, 11), f"{len(cells)} cells in: {row[:60]}"
    kept = cells if len(cells) == 6 else [cells[0]] + cells[SIGMA_C_COLUMNS]
    return " & ".join(kept) + " \\\\"


def project(tex, label):
    rows = [sigma_c_row(r) for r in table_rows(tex, label)]
    return "\n".join(f"        {r}" for r in rows)


# Daggers whose footnote concerned the sigma = 0.1 half alone: the 16x16
# one-sweep masked-attention row is a two-seed mean at sigma = 0.1 (seed 43
# collapsed) but retains all three seeds at sigma_c, so its mark has no
# referent once that half is gone. The prefix-sum two-sweep dagger is a
# sigma_c exclusion and stays.
DROPPED_DAGGERS = {"masked-attention band, one sweep$^{\\dagger}$": "masked-attention band, one sweep"}
# The ladder has no \resizebox and the reference label set its width (23.9pt over
# the text width at \tabcolsep 4pt); the appendix tables keep the full label.
RELABELS = {"Kawasaki (thesis engine), certified reference": "Kawasaki reference (thesis engine)"}


def ladder(tex, labels, headings):
    blocks = []
    for label, heading in zip(labels, headings):
        rows = [sigma_c_row(r) for r in table_rows(tex, label) if r != "\\bottomrule"]
        for marked, plain in {**DROPPED_DAGGERS, **RELABELS}.items():
            rows = [r.replace(marked, plain) for r in rows]
        rows = ["\\addlinespace[2pt]" if r == "\\midrule" else r for r in rows]
        block = [f"\\multicolumn{{6}}{{l}}{{\\emph{{{heading}}}}} \\\\", *rows]
        blocks.append("\n".join(f"        {r}" for r in block))
    return "\n        \\midrule\n".join(blocks) + "\n        \\bottomrule"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--ladder", nargs="+")
    parser.add_argument("--headings", nargs="+", help="one per --ladder label")
    args = parser.parse_args()
    tex = args.tex.read_text()
    if args.label:
        print(project(tex, args.label))
    if args.ladder:
        headings = args.headings or [
            re.search(r"(\d+)x(\d+)", l).group(0).replace("x", r"\times") for l in args.ladder
        ]
        print(ladder(tex, args.ladder, [f"${h}$" for h in headings]))


if __name__ == "__main__":
    main()
