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

type ResultCell = dict[str, float | int]

RESULTS: str = (
    "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/aim2_horizontal*.json"
)
"""Selected the direct-path campaign result shards."""

rows: dict[str, ResultCell] = {}
"""Initialised the combined mapping of completed direct-path cells."""
for path in sorted(glob.glob(RESULTS)):
    with open(path, encoding="utf-8") as stream:
        loaded: dict[str, ResultCell] = json.load(stream)["cells"]
        """Loaded the cells recorded in one direct-path result shard."""
    for name, r in loaded.items():
        if r["replicates"]:
            rows[name] = r
            """Retained this cell because it contained completed replicates."""
"""Combined all completed result shards before inspecting either study arm."""

if not rows:
    raise SystemExit("no direct-path cells have landed yet")

for arm in ("ABR only", "audiogram"):
    here: dict[str, ResultCell] = {
        key: value for key, value in rows.items() if key.startswith(arm)
    }
    """Selected the completed cells belonging to this study arm."""
    if not here:
        continue
    print(f"=== {arm} ===")

    # Level first: the cell where the direct path is nought.
    level_name: str | None = next((name for name in here if name.endswith("0.0")), None)
    """Located the matched cell where the direct path was nought."""
    usable: bool = True
    """Initialised whether this arm's rejection rates can be read as power."""
    if level_name:
        level_result: ResultCell = here[level_name]
        """Selected the arm-specific null result."""
        level_rate: float = float(level_result["horizontal_reject_025"]) / int(
            level_result["replicates"]
        )
        """Calculated the empirical rejection rate under the direct-path null."""
        level_error: float = math.sqrt(
            max(level_rate * (1 - level_rate), 1e-9) / int(level_result["replicates"])
        )
        """Estimated the Monte Carlo standard error of the null rejection rate."""
        verdict: str = (
            "holds" if level_rate <= 0.025 + 2 * level_error else "ABOVE NOMINAL"
        )
        """Classified whether the arm's level stayed within two standard errors."""
        print(
            f"  level at direct path nought: {level_rate:.3f} "
            f"+/-{level_error:.3f}  {verdict}"
        )
        usable = level_rate <= 0.025 + 2 * level_error
        """Allowed power interpretation only when the matched level held."""
    else:
        print("  no level cell yet -- power below is not yet readable")
        usable = False
        """Withheld power interpretation because the null cell had not reported."""

    print(f"  {'direct path':>12} {'power':>8} {'+/-':>7} {'refused':>8}")
    curve: list[tuple[float, float]] = []
    """Initialised the direct-path effect and empirical-power curve."""
    for name in sorted(here):
        value: float = float(name.split()[-1])
        """Parsed the simulated direct-path value from the cell label."""
        if value == 0.0:
            continue
        result: ResultCell = here[name]
        """Selected the completed alternative cell at this path value."""
        n: int = int(result["replicates"])
        """Recovered the Monte Carlo denominator for this cell."""
        power: float = float(result["horizontal_reject_025"]) / n
        """Calculated empirical power at this direct-path value."""
        power_error: float = math.sqrt(max(power * (1 - power), 1e-9) / n)
        """Estimated the Monte Carlo standard error of empirical power."""
        curve.append((value, power))
        """Added the measured point to the arm's power curve."""
        print(
            f"  {value:>12.1f} {power:>8.3f} {power_error:>7.3f} "
            f"{result['horizontal_refused']:>8}"
        )
    """Printed and retained every completed non-null direct-path cell."""

    if not usable:
        print("  (level did not hold, so these are not power figures)\n")
        continue

    reached: list[float] = [value for value, power in curve if power >= 0.80]
    """Selected the tested path values that attained at least 80 per cent power."""
    if reached:
        print(f"\n  smallest direct path detectable at 80 per cent: {min(reached):.1f}")
    else:
        best: tuple[float, float] | None = (
            max(curve, key=lambda point: point[0]) if curve else None
        )
        """Selected the largest tested direct path when the target was not reached."""
        if best:
            print(
                f"\n  80 per cent is not reached at any direct path tested "
                f"so far. The largest\n  is {best[0]:.1f}, where power is "
                f"{best[1]:.1%}."
            )
            # Only the grid that was planned licenses a statement about the
            # grid. Saying "at any size" while cells are still running would be
            # a claim about the ones that have not reported.
            if best[0] < 0.7:
                print(
                    "  Cells above this are still running, so this is what "
                    "has been measured\n  rather than the whole answer."
                )
    print("")
