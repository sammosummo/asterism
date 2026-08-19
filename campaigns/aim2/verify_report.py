"""Do the numbers published in the report match the result files?

The report was written by hand from tables printed to a terminal, which is
exactly where a digit gets transposed and then survives, because nothing
downstream ever compares the two again. This compares them.

Every rate quoted in the report is recomputed from the cells and checked
against what the page says.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

RESULTS = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/*.json"
REPORT = Path("/Users/samuelmathias/.claude/jobs/f010e916/tmp/aim2_power.html")

cells = {}
for path in sorted(glob.glob(RESULTS)):
    for name, r in json.load(open(path))["cells"].items():
        if r["replicates"]:
            cells[name] = r

page = REPORT.read_text()

# (what the page claims, which cell, which test)
claims = [
    # vertical power
    ("0.055", "ABR only | vertical 0.06", "vertical"),
    ("0.070", "ABR only | vertical 0.12", "vertical"),
    ("0.230", "ABR only | vertical 0.18", "vertical"),
    ("0.455", "ABR only | vertical 0.24", "vertical"),
    ("0.080", "audiogram | vertical 0.06", "vertical"),
    ("0.160", "audiogram | vertical 0.12", "vertical"),
    ("0.335", "audiogram | vertical 0.18", "vertical"),
    ("0.545", "audiogram | vertical 0.24", "vertical"),
    # vertical level
    ("0.020", "ABR only | null: no loading", "vertical"),
    ("0.025", "ABR only | null: no path", "vertical"),
    ("0.035", "audiogram | null: no path", "vertical"),
    ("0.010", "ABR only | level: neither, no path", "vertical"),
    ("0.040", "audiogram | level: neither, no path", "vertical"),
    # horizontal level
    ("0.015", "ABR only | level: neither, no path", "horizontal"),
    ("0.030", "audiogram | level: neither, no path", "horizontal"),
    ("0.025", "ABR only | level: horizontal only", "horizontal"),
    ("0.030", "audiogram | level: horizontal only", "horizontal"),
    # direct-path grid
    ("0.115", "ABR only | direct 0.3", "horizontal"),
    ("0.300", "ABR only | direct 0.5", "horizontal"),
    ("0.500", "ABR only | direct 0.7", "horizontal"),
    ("0.155", "audiogram | direct 0.3", "horizontal"),
    ("0.330", "audiogram | direct 0.5", "horizontal"),
    ("0.545", "audiogram | direct 0.7", "horizontal"),
]

problems = []
for claimed, cell, test in claims:
    if cell not in cells:
        problems.append(f"{cell}: no such cell, but the page quotes {claimed}")
        continue
    r = cells[cell]
    actual = r[f"{test}_reject_025"] / r["replicates"]
    if abs(actual - float(claimed)) > 5e-4:
        problems.append(
            f"{cell} ({test}): page says {claimed}, cells give {actual:.3f}")
    if claimed not in page:
        problems.append(f"{cell} ({test}): {claimed} does not appear on the page")

# The two figures in the summary panel.
for figure, why in (("1,600", "people"), ("257", "cases"), ("54.5%", "best power")):
    if figure not in page:
        problems.append(f"the {why} figure {figure} is not on the page")

if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
else:
    print(f"all {len(claims)} quoted rates match the result files")
