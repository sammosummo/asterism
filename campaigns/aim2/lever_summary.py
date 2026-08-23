"""Which lever bought power, read against its own level cell.

Each lever changed exactly one thing from the real design and carries its own
null cell under the identical change, so a lever that lifted power by lifting
rejections generally shows up as a lever that lifted its level too.
"""

import glob
import json
import math

type ResultCell = dict[str, float | int]

cells: dict[str, ResultCell] = {}
"""Collected every completed lever and matching null cell."""
for path in sorted(
    glob.glob(
        "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/aim2_levers*.json"
    )
):
    with open(path, encoding="utf-8") as stream:
        loaded: dict[str, ResultCell] = json.load(stream)["cells"]
        """Loaded the cells stored in one lever-campaign result shard."""
    for name, r in loaded.items():
        if r["replicates"]:
            cells[name] = r
            """Retained this cell because it contained completed replicates."""
"""Combined the completed campaign shards into one result mapping."""

LEVERS: list[str] = [
    "baseline",
    "probands x1.5",
    "probands x2",
    "hearing precise",
    "drop acoustic units",
    "sample x1.5",
    "staged outcome",
]
"""Fixed the display order of the design changes being compared."""

base: ResultCell | None = cells.get("baseline | vertical 0.24")
"""Selected the common baseline used to express each power change."""
print(
    f"{'lever':<22}{'people':>7}{'cases':>7}{'level':>9}{'power':>9}"
    f"{'vs base':>9}{'refused':>9}"
)
print("-" * 72)
for lever in LEVERS:
    power: ResultCell | None = cells.get(f"{lever} | vertical 0.24")
    """Selected the power cell for this one-factor design change."""
    level: ResultCell | None = cells.get(f"{lever} | null: no path")
    """Selected the matched null cell used to validate its rejection rate."""
    if not power:
        print(f"{lever:<22} not reported")
        continue
    n: int = int(power["replicates"])
    """Recovered the Monte Carlo denominator for the power cell."""
    rate: float = float(power["vertical_reject_025"]) / n
    """Calculated the empirical rejection rate under the alternative."""
    se: float = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
    """Estimated the Monte Carlo standard error of power."""
    level_rate: float = (
        float(level["vertical_reject_025"]) / int(level["replicates"])
        if level
        else float("nan")
    )
    """Calculated the matched null rejection rate when that cell was available."""
    level_se: float = (
        math.sqrt(max(level_rate * (1 - level_rate), 1e-9) / int(level["replicates"]))
        if level
        else 0.0
    )
    """Estimated the Monte Carlo error of the matched level measurement."""
    flag: str = "" if level_rate <= 0.025 + 2 * level_se else "  LEVEL HIGH"
    """Flagged any design change whose matched null over-rejected."""
    change: str = (
        ""
        if lever == "baseline"
        else f"{rate - float(base['vertical_reject_025']) / int(base['replicates']):+.3f}"
    )
    """Expressed the alternative rejection-rate change from the baseline design."""
    print(
        f"{lever:<22}{power['people']:>7}{power['mean_cases']:>7}"
        f"{level_rate:>9.3f}{rate:>9.3f}{change:>9}"
        f"{power['vertical_refused']:>9}{flag}"
    )
"""Printed each design lever beside its own level and power measurements."""

print(
    "\nMonte Carlo error on a power near a half is about +/-.035 at 200 "
    "replicates,\nso a change smaller than about .07 is not a change."
)
