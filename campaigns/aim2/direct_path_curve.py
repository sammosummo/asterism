"""What size of direct path could this study detect?

The power campaign established that it cannot detect one of 0.2 -- rejection ran
between 0.06 and 0.12 there. This turns that into the number an application can
use: the smallest direct path reaching a usable power, read off a grid that
holds the mediated effect real and large throughout.

The level cell is in the same grid rather than borrowed, so the comparison is
against a level measured under the identical design.
"""

from __future__ import annotations

import glob
import json
import math

RESULTS = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/aim2_horizontal*.json"

rows = {}
for path in sorted(glob.glob(RESULTS)):
    for name, r in json.load(open(path))["cells"].items():
        if r["replicates"]:
            rows[name] = r

if not rows:
    raise SystemExit("no direct-path cells have landed yet")

for arm in ("ABR only", "audiogram"):
    here = {k: v for k, v in rows.items() if k.startswith(arm)}
    if not here:
        continue
    print(f"=== {arm} ===")

    # Level first: the cell where the direct path is nought.
    level_name = next((k for k in here if k.endswith("0.0")), None)
    usable = True
    if level_name:
        r = here[level_name]
        rate = r["horizontal_reject_025"] / r["replicates"]
        se = math.sqrt(max(rate * (1 - rate), 1e-9) / r["replicates"])
        verdict = "holds" if rate <= 0.025 + 2 * se else "ABOVE NOMINAL"
        print(f"  level at direct path nought: {rate:.3f} +/-{se:.3f}  {verdict}")
        usable = rate <= 0.025 + 2 * se
    else:
        print("  no level cell yet -- power below is not yet readable")
        usable = False

    print(f"  {'direct path':>12} {'power':>8} {'+/-':>7} {'refused':>8}")
    curve = []
    for name in sorted(here):
        value = float(name.split()[-1])
        if value == 0.0:
            continue
        r = here[name]
        n = r["replicates"]
        power = r["horizontal_reject_025"] / n
        se = math.sqrt(max(power * (1 - power), 1e-9) / n)
        curve.append((value, power))
        print(f"  {value:>12.1f} {power:>8.3f} {se:>7.3f} "
              f"{r['horizontal_refused']:>8}")

    if not usable:
        print("  (level did not hold, so these are not power figures)\n")
        continue

    reached = [v for v, p in curve if p >= 0.80]
    if reached:
        print(f"\n  smallest direct path detectable at 80 per cent: "
              f"{min(reached):.1f}")
    else:
        best = max(curve, key=lambda t: t[0]) if curve else None
        if best:
            print(f"\n  80 per cent is not reached at any direct path tested "
                  f"so far. The largest\n  is {best[0]:.1f}, where power is "
                  f"{best[1]:.1%}.")
            # Only the grid that was planned licenses a statement about the
            # grid. Saying "at any size" while cells are still running would be
            # a claim about the ones that have not reported.
            if best[0] < 0.7:
                print("  Cells above this are still running, so this is what "
                      "has been measured\n  rather than the whole answer.")
    print("")
