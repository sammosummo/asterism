"""Do the numbers in the drafting pack match the cells they came from?

This page is written to be lifted into an application, so a transposed digit
here does not stay here. Everything quoted is recomputed from the result files.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

RESULTS = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results"
PAGE = Path("/Users/samuelmathias/.claude/jobs/f010e916/tmp/aim2_drafting.html")

cells = {}
for path in sorted(glob.glob(f"{RESULTS}/*.json")):
    loaded = json.load(open(path))
    for name, r in loaded.get("cells", {}).items():
        if isinstance(r, dict) and r.get("replicates"):
            cells[name] = r
page = PAGE.read_text()

claims = [
    # effect-size curve
    ("0.055", "ABR only | vertical 0.06", "vertical"),
    ("0.070", "ABR only | vertical 0.12", "vertical"),
    ("0.230", "ABR only | vertical 0.18", "vertical"),
    ("0.455", "ABR only | vertical 0.24", "vertical"),
    ("0.080", "audiogram | vertical 0.06", "vertical"),
    ("0.160", "audiogram | vertical 0.12", "vertical"),
    ("0.335", "audiogram | vertical 0.18", "vertical"),
    ("0.545", "audiogram | vertical 0.24", "vertical"),
    # design-change grid
    ("0.520", "baseline | vertical 0.24", "vertical"),
    ("0.660", "staged outcome | vertical 0.24", "vertical"),
    ("0.695", "probands x1.5 | vertical 0.24", "vertical"),
    ("0.710", "sample x1.5 | vertical 0.24", "vertical"),
    ("0.845", "probands x2 | vertical 0.24", "vertical"),
    ("0.585", "hearing precise | vertical 0.24", "vertical"),
    ("0.550", "drop acoustic units | vertical 0.24", "vertical"),
    # design-change levels
    ("0.010", "baseline | null: no path", "vertical"),
    ("0.005", "staged outcome | null: no path", "vertical"),
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
        problems.append(f"{cell}: no such cell, page quotes {claimed}")
        continue
    r = cells[cell]
    actual = r[f"{test}_reject_025"] / r["replicates"]
    if abs(actual - float(claimed)) > 5e-4:
        problems.append(f"{cell} ({test}): page {claimed}, cells {actual:.3f}")
    if claimed not in page:
        problems.append(f"{cell}: {claimed} not on the page")

# The leak medians come from their own file.
leak_path = Path(RESULTS) / "aim2_leak.json"
if leak_path.exists():
    leak = json.load(open(leak_path))["cells"]
    for key, claimed in (("0.0", "0.233"), ("0.7", "0.284")):
        actual = leak[key]["median_vertical"]
        if abs(actual - float(claimed)) > 5e-4:
            problems.append(f"leak at c'={key}: page {claimed}, file {actual:.3f}")
        if claimed not in page:
            problems.append(f"leak median {claimed} not on the page")
else:
    problems.append("the leak result file is missing")

# The level count, recomputed rather than remembered.
def nulls(name: str):
    label = name.split("|")[1].strip()
    if label in ("null: no loading", "null: no path"):
        return True, False
    if label.startswith("level: neither"):
        return True, True
    if label in ("level: horizontal only", "direct 0.0"):
        return False, True
    return False, False

total = 0
for name, r in cells.items():
    if " | " not in name:
        continue
    v_null, h_null = nulls(name)
    total += int(v_null) + int(h_null)
if "Sixteen separate level measurements" in page and total != 16:
    problems.append(f"the page says sixteen level measurements; there are {total}")

if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
else:
    print(f"all {len(claims)} quoted rates, the leak medians and the level "
          f"count ({total}) check out")
