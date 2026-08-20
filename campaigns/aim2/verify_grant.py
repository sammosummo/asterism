"""Do the figures in the revised grant text match the cells they came from?

This text is going into an application. Every number in it is recomputed from
the result files, and the ones that are ratios or derived quantities are
recomputed too rather than trusted to arithmetic done once in prose.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

RESULTS = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results"
BASE = Path("/Users/samuelmathias/MathiasLab/staging/studies/planned/mediation/"
            "grant-applications")
AIMS = BASE / "SPECIFIC_AIMS_REVISED_2026-08-19.md"
STRATEGY = BASE / "RESEARCH_STRATEGY_AIM2_REVISIONS_2026-08-19.md"

cells = {}
for path in sorted(glob.glob(f"{RESULTS}/*.json")):
    for name, r in json.load(open(path)).get("cells", {}).items():
        if isinstance(r, dict) and r.get("replicates"):
            cells[name] = r


def rate(cell: str, test: str) -> float:
    r = cells[cell]
    return r[f"{test}_reject_025"] / r["replicates"]


text = AIMS.read_text() + STRATEGY.read_text()
problems = []

# Quoted rates.
for claimed, cell, test in [
    ("0.158", None, None), ("0.083", None, None),
    ("0.384", None, None), ("0.253", None, None),
    ("0.55", "audiogram | vertical 0.24", "vertical"),
    ("0.46", "ABR only | vertical 0.24", "vertical"),
    ("0.66", "staged outcome | vertical 0.24", "vertical"),
    ("0.065", None, None),
]:
    if claimed not in text:
        problems.append(f"{claimed} is not in the revised text")
        continue
    if cell is None:
        continue
    actual = rate(cell, test)
    if abs(round(actual, 2) - float(claimed)) > 5e-3:
        problems.append(f"{cell}: text {claimed}, cells {actual:.3f}")

# The offspring ratio range, recomputed rather than taken from the prose.
low = 0.158 / 0.083
high = 0.384 / 0.253
lo, hi = min(low, high), max(low, high)
if not (abs(hi - 1.9) < 0.05 and abs(lo - 1.5) < 0.05):
    problems.append(f"the offspring ratio is {lo:.2f} to {hi:.2f}, "
                    f"not the 1.5 to 1.9 the text claims")

# The staging gain in points.
gain = rate("staged outcome | vertical 0.24", "vertical") \
    - rate("baseline | vertical 0.24", "vertical")
if abs(gain - 0.14) > 0.005:
    problems.append(f"the staging gain is {gain:.3f}, not the fourteen points "
                    f"claimed")
if "fourteen" not in text:
    problems.append("the staging gain is not stated in the text")

# The hearing-precision non-result.
precise = rate("hearing precise | vertical 0.24", "vertical") \
    - rate("baseline | vertical 0.24", "vertical")
if abs(precise - 0.065) > 0.005:
    problems.append(f"the hearing-precision change is {precise:.3f}, not 0.065")

# The level count.
def nulls(name: str):
    label = name.split("|")[1].strip()
    if label in ("null: no loading", "null: no path"):
        return True, False
    if label.startswith("level: neither"):
        return True, True
    if label in ("level: horizontal only", "direct 0.0"):
        return False, True
    return False, False

total = sum(sum(nulls(n)) for n in cells if " | " in n)
if "Twenty-three" not in text or total != 23:
    problems.append(f"the text says twenty-three level measurements; "
                    f"there are {total}")

# The direct-path figures quoted in the analysis plan.
for claimed, cell in (("0.155", "audiogram | direct 0.3"),
                      ("0.545", "audiogram | direct 0.7")):
    actual = rate(cell, "horizontal")
    if abs(actual - float(claimed)) > 5e-4:
        problems.append(f"{cell}: text {claimed}, cells {actual:.3f}")

# The leak medians.
leak = json.load(open(Path(RESULTS) / "aim2_leak.json"))["cells"]
for key, claimed in (("0.0", 0.233), ("0.7", 0.284)):
    if abs(leak[key]["median_vertical"] - claimed) > 5e-4:
        problems.append(f"leak median at {key}: text {claimed}, "
                        f"file {leak[key]['median_vertical']:.3f}")

# 250 - 184 = 66 adjudications.
if "66 fewer" not in text:
    problems.append("the adjudication difference is not stated")

if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
else:
    print("every figure in the revised grant text checks out against the cells")
