"""Turn the campaign grid into the two numbers a grant actually quotes.

A table of power at five effect sizes is not what goes in an application. What
goes in is the smallest effect the study can detect at 80 per cent, and -- when
that lies outside what the study can reach -- how much more sample would be
needed to bring it inside.

Both are read off the same curve. For a likelihood ratio test the non-centrality
grows about in proportion to the sample, so a power figure can be turned into a
required multiple of the sample by asking what non-centrality gives 80 per cent
and comparing it with the one implied by the observed power. That is an
approximation and it is stated as one: the vertical test is an
intersection-union test rather than a plain chi-square, and near the union null
its behaviour is not the chi-square this arithmetic assumes.

Nothing is reported from a cell whose level failed, and the level is checked
first rather than mentioned afterwards.
"""

from __future__ import annotations

import glob
import json
import math

from scipy.stats import chi2, ncx2
from scipy.optimize import brentq

CRITICAL = chi2.ppf(1 - 0.025, 1)
TARGET = 0.80


def noncentrality_for(power: float) -> float:
    """The non-centrality that gives this power at the one-sided .025 level."""
    if power <= 0.025:
        return 0.0
    if power >= 0.999:
        return float("inf")
    return brentq(lambda lam: ncx2.sf(CRITICAL, 1, lam) - power, 1e-9, 400.0)


rows = {}
for path in sorted(glob.glob(
        "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/*.json")):
    for name, r in json.load(open(path))["cells"].items():
        if r["replicates"]:
            rows[name] = r

needed = noncentrality_for(TARGET)
print(f"80 per cent power at the .025 level needs a non-centrality of "
      f"{needed:.2f}\n")

for arm in ("ABR only", "audiogram"):
    print(f"=== {arm} ===")

    # The level first. A power figure from a test that over-rejects is not a
    # power figure.
    levels = {k: v for k, v in rows.items()
              if k.startswith(arm) and "null" in k}
    if not levels:
        print("  no null cell has reported yet -- nothing here is quotable\n")
        continue
    worst = 0.0
    for name, r in sorted(levels.items()):
        rate = r["vertical_reject_025"] / r["replicates"]
        error = math.sqrt(max(rate * (1 - rate), 1e-9) / r["replicates"])
        worst = max(worst, rate)
        verdict = "at nominal" if rate <= 0.025 + 2 * error else "ABOVE NOMINAL"
        print(f"  level  {name.split('|')[1].strip():<18} "
              f"{rate:.3f} +/-{error:.3f}   {verdict}")
    if worst > 0.025 + 0.02:
        print("  level is too high to quote power from this arm\n")
        continue

    print(f"  {'effect':>8} {'power':>8} {'per cent of the sample needed':>32}")
    curve = []
    for name, r in sorted(rows.items()):
        if not name.startswith(arm) or "vertical" not in name:
            continue
        effect = float(name.split()[-1])
        power = r["vertical_reject_025"] / r["replicates"]
        curve.append((effect, power))
        here = noncentrality_for(power)
        share = (needed / here * 100.0) if here > 0 else float("inf")
        share_text = f"{share:,.0f}%" if math.isfinite(share) else "not reachable"
        print(f"  {effect:>8.2f} {power:>8.3f} {share_text:>32}")

    # The smallest effect reaching 80 per cent, by interpolating the curve.
    reached = [e for e, p in curve if p >= TARGET]
    if reached:
        print(f"\n  smallest effect detectable at 80 per cent: "
              f"{min(reached):.2f}")
    elif curve:
        best_effect, best_power = max(curve, key=lambda t: t[0])
        here = noncentrality_for(best_power)
        multiple = needed / here if here > 0 else float("inf")
        print(f"\n  80 per cent is not reached anywhere in the range tested.")
        print(f"  At the largest effect ({best_effect:.2f}) power is "
              f"{best_power:.1%}.")
        if math.isfinite(multiple):
            print(f"  Reaching 80 per cent there would need about "
                  f"{multiple:.1f} times the sample,")
            print(f"  which is roughly {1600 * multiple:,.0f} people against "
                  f"the 1,600 planned.")
    print("")
