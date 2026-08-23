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

from scipy.optimize import brentq
from scipy.stats import chi2, ncx2

type ResultCell = dict[str, float | int]

CRITICAL: float = float(chi2.ppf(1 - 0.025, 1))
"""Calculated the chi-squared critical value for the one-sided 0.025 level."""
TARGET: float = 0.80
"""Set the power target used for detectable-effect and sample-size summaries."""


def noncentrality_for(power: float) -> float:
    """Find the non-centrality yielding a requested power.

    Args:
        power: Target rejection probability at the one-sided 0.025 level.

    Returns:
        Non-centrality parameter producing the target power.
    """
    if power <= 0.025:
        return 0.0
    if power >= 0.999:
        return float("inf")
    return brentq(lambda lam: ncx2.sf(CRITICAL, 1, lam) - power, 1e-9, 400.0)


rows: dict[str, ResultCell] = {}
"""Initialised the combined mapping of completed campaign cells."""
for path in sorted(
    glob.glob("/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/*.json")
):
    with open(path, encoding="utf-8") as stream:
        loaded: dict[str, ResultCell] = json.load(stream)["cells"]
        """Loaded the cells recorded in one campaign result shard."""
    for name, r in loaded.items():
        if r.get("replicates"):
            rows[name] = r
            """Retained this cell because it contained completed replicates."""
"""Combined all completed result shards before deriving grant-scale summaries."""

needed: float = noncentrality_for(TARGET)
"""Calculated the non-centrality required for the target power."""
print(f"80 per cent power at the .025 level needs a non-centrality of {needed:.2f}\n")

for arm in ("ABR only", "audiogram"):
    print(f"=== {arm} ===")

    # The level first. A power figure from a test that over-rejects is not a
    # power figure.
    levels: dict[str, ResultCell] = {
        key: value
        for key, value in rows.items()
        if key.startswith(arm) and "null" in key
    }
    """Selected this arm's completed null cells for the level check."""
    if not levels:
        print("  no null cell has reported yet -- nothing here is quotable\n")
        continue
    worst: float = 0.0
    """Initialised the largest observed null rejection rate in this arm."""
    for name, r in sorted(levels.items()):
        rate: float = float(r["vertical_reject_025"]) / int(r["replicates"])
        """Calculated the rejection rate for one arm-specific null condition."""
        error: float = math.sqrt(max(rate * (1 - rate), 1e-9) / int(r["replicates"]))
        """Estimated the Monte Carlo standard error of the null rejection rate."""
        worst = max(worst, rate)
        """Updated the most liberal null rejection rate observed in the arm."""
        verdict: str = "at nominal" if rate <= 0.025 + 2 * error else "ABOVE NOMINAL"
        """Classified this level estimate against its Monte Carlo uncertainty."""
        print(
            f"  level  {name.split('|')[1].strip():<18} "
            f"{rate:.3f} +/-{error:.3f}   {verdict}"
        )
    """Checked and displayed every available null condition for this arm."""
    if worst > 0.025 + 0.02:
        print("  level is too high to quote power from this arm\n")
        continue

    print(f"  {'effect':>8} {'power':>8} {'per cent of the sample needed':>32}")
    curve: list[tuple[float, float]] = []
    """Initialised the measured effect-size and empirical-power curve."""
    for name, r in sorted(rows.items()):
        if not name.startswith(arm) or "vertical" not in name:
            continue
        effect: float = float(name.split()[-1])
        """Parsed the simulated mediated effect from the cell label."""
        power: float = float(r["vertical_reject_025"]) / int(r["replicates"])
        """Calculated empirical power for this effect-size cell."""
        curve.append((effect, power))
        """Added the completed cell to the arm's measured power curve."""
        point_ncp: float = noncentrality_for(power)
        """Converted the observed power to its approximate non-centrality."""
        share: float = needed / point_ncp * 100.0 if point_ncp > 0 else float("inf")
        """Estimated the sample percentage needed to reach the power target."""
        share_text: str = f"{share:,.0f}%" if math.isfinite(share) else "not reachable"
        """Formatted the approximate sample requirement for display."""
        print(f"  {effect:>8.2f} {power:>8.3f} {share_text:>32}")
    """Built and displayed the completed power curve for this arm."""

    # The smallest effect reaching 80 per cent, by interpolating the curve.
    reached: list[float] = [effect for effect, power in curve if power >= TARGET]
    """Selected tested effects that reached the target empirical power."""
    if reached:
        print(f"\n  smallest effect detectable at 80 per cent: {min(reached):.2f}")
    elif curve:
        best_effect, best_power = max(curve, key=lambda t: t[0])
        """Selected the largest tested effect and its measured power."""
        best_ncp: float = noncentrality_for(best_power)
        """Converted the best observed power to its approximate non-centrality."""
        multiple: float = needed / best_ncp if best_ncp > 0 else float("inf")
        """Estimated the sample-size multiple required to attain target power."""
        print("\n  80 per cent is not reached anywhere in the range tested.")
        print(f"  At the largest effect ({best_effect:.2f}) power is {best_power:.1%}.")
        if math.isfinite(multiple):
            print(
                f"  Reaching 80 per cent there would need about "
                f"{multiple:.1f} times the sample,"
            )
            print(
                f"  which is roughly {1600 * multiple:,.0f} people against "
                f"the 1,600 planned."
            )
    print("")
