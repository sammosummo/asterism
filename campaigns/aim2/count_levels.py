"""How many level measurements are there actually, and do they all hold?

The report quotes a count. A count written from memory of which cells null
which estimand is exactly the sort of thing that comes out wrong, so it is
recounted here from the cells.

A cell contributes a level measurement per estimand that is nought in it, so
one cell can contribute two.
"""

import glob
import json
import math

cells = {}
for path in sorted(glob.glob(
        "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/*.json")):
    for name, r in json.load(open(path))["cells"].items():
        if r["replicates"]:
            cells[name] = r


def nulls(name: str) -> tuple[bool, bool]:
    """(vertical is nought, horizontal is nought) in this cell."""
    label = name.split("|")[1].strip()
    if label in ("null: no loading", "null: no path"):
        return True, False            # a b = 0, c' = 0.2
    if label.startswith("level: neither"):
        return True, True             # a b = 0 and c' = 0
    if label == "level: horizontal only":
        return False, True            # a b real, c' = 0
    if label == "direct 0.0":
        return False, True            # a b real, c' = 0
    return False, False               # a power cell


total, failures = 0, []
for name, r in sorted(cells.items()):
    v_null, h_null = nulls(name)
    for estimand, reject, is_null in (
            ("vertical", r["vertical_reject_025"], v_null),
            ("horizontal", r["horizontal_reject_025"], h_null)):
        if not is_null:
            continue
        total += 1
        rate = reject / r["replicates"]
        se = math.sqrt(max(rate * (1 - rate), 1e-9) / r["replicates"])
        if rate > 0.025 + 2 * se:
            failures.append(f"{name} ({estimand}): {rate:.3f} +/-{se:.3f}")

print(f"{total} level measurements across {len(cells)} reporting cells")
if failures:
    print(f"{len(failures)} above nominal:")
    for f in failures:
        print(f"  {f}")
else:
    print("all at or below .025 within two Monte Carlo standard errors")
