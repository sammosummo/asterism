"""Which lever bought power, read against its own level cell.

Each lever changed exactly one thing from the real design and carries its own
null cell under the identical change, so a lever that lifted power by lifting
rejections generally shows up as a lever that lifted its level too.
"""

import glob
import json
import math

cells = {}
for path in sorted(glob.glob(
        "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/aim2_levers*.json")):
    for name, r in json.load(open(path))["cells"].items():
        if r["replicates"]:
            cells[name] = r

LEVERS = ["baseline", "probands x1.5", "probands x2", "hearing precise",
          "drop acoustic units", "sample x1.5"]

base = cells.get("baseline | vertical 0.24")
print(f"{'lever':<22}{'people':>7}{'cases':>7}{'level':>9}{'power':>9}"
      f"{'vs base':>9}{'refused':>9}")
print("-" * 72)
for lever in LEVERS:
    power = cells.get(f"{lever} | vertical 0.24")
    level = cells.get(f"{lever} | null: no path")
    if not power:
        print(f"{lever:<22} not reported")
        continue
    n = power["replicates"]
    rate = power["vertical_reject_025"] / n
    se = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
    level_rate = (level["vertical_reject_025"] / level["replicates"]
                  if level else float("nan"))
    level_se = (math.sqrt(max(level_rate * (1 - level_rate), 1e-9)
                          / level["replicates"]) if level else 0.0)
    flag = "" if level_rate <= 0.025 + 2 * level_se else "  LEVEL HIGH"
    change = "" if lever == "baseline" else \
        f"{rate - base['vertical_reject_025'] / base['replicates']:+.3f}"
    print(f"{lever:<22}{power['people']:>7}{power['mean_cases']:>7}"
          f"{level_rate:>9.3f}{rate:>9.3f}{change:>9}"
          f"{power['vertical_refused']:>9}{flag}")

print("\nMonte Carlo error on a power near a half is about +/-.035 at 200 "
      "replicates,\nso a change smaller than about .07 is not a change.")
