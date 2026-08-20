"""Every null cell that has reported, both tests, with its Monte Carlo error.

Level first and level separately, because a power figure read off a test that
over-rejects is not a power figure. Which test a cell measures the level of
depends on which estimand is nought in it, and that is not the same for every
cell -- so it is worked out per cell rather than assumed.
"""

import glob
import json
import math

rows = {}
for path in sorted(glob.glob(
        "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/*.json")):
    for name, r in json.load(open(path))["cells"].items():
        if r["replicates"]:
            rows[name] = r


def truth_of(name: str):
    """Which estimands are nought in this cell."""
    label = name.split("|")[1].strip()
    if label == "null: no loading":      # a=0, b free, c'=0.2
        return True, False
    if label == "null: no path":         # a free, b=0, c'=0.2
        return True, False
    if label.startswith("level: neither"):   # a b = 0 and c' = 0
        return True, True
    if label == "level: horizontal only":    # a b real, c' = 0
        return False, True
    return False, False                  # a power cell


print(f"{'cell':<42} {'reps':>5} {'vertical':>17} {'horizontal':>17}")
print("-" * 86)
for name in sorted(rows):
    r = rows[name]
    n = r["replicates"]
    v_null, h_null = truth_of(name)
    out = []
    for reject, is_null in ((r["vertical_reject_025"], v_null),
                            (r["horizontal_reject_025"], h_null)):
        rate = reject / n
        se = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
        if is_null:
            mark = "ok" if rate <= 0.025 + 2 * se else "HIGH"
            out.append(f"{rate:.3f} lvl {mark:>4}")
        else:
            out.append(f"{rate:.3f} pwr     ")
    print(f"{name:<42} {n:>5} {out[0]:>17} {out[1]:>17}")

print("\n'lvl' is a level -- the estimand is nought there and rejection must sit")
print("at or below .025. 'pwr' is power. Every level cell above is a separate")
print("null and a test can hold against one and not another.")
