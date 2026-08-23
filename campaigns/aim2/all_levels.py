"""Every null cell that has reported, both tests, with its Monte Carlo error.

Level first and level separately, because a power figure read off a test that
over-rejects is not a power figure. Which test a cell measures the level of
depends on which estimand is nought in it, and that is not the same for every
cell -- so it is worked out per cell rather than assumed.
"""

from __future__ import annotations

import glob
import json
import math

type ResultCell = dict[str, float | int]

rows: dict[str, ResultCell] = {}
"""Collected every completed simulation cell across the campaign shards."""
for path in sorted(
    glob.glob("/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/*.json")
):
    with open(path, encoding="utf-8") as stream:
        cells: dict[str, ResultCell] = json.load(stream)["cells"]
        """Loaded the cells recorded in one campaign result shard."""
    for name, r in cells.items():
        if (
            r.get("replicates")
            and {
                "vertical_reject_025",
                "horizontal_reject_025",
            }
            <= r.keys()
        ):
            rows[name] = r
            """Retained this completed cell because it reported both tests."""
"""Retained only completed cells that reported both hypothesis tests."""


def truth_of(name: str) -> tuple[bool, bool]:
    """Identify which estimands are nought in a campaign cell.

    Args:
        name: Campaign cell label containing the simulated null condition.

    Returns:
        Whether the vertical and horizontal estimands are respectively nought.
    """
    label: str = name.split("|")[1].strip()
    """Isolated the scientific condition from the arm prefix."""
    if label == "null: no loading":  # a=0, b free, c'=0.2
        return True, False
    if label == "null: no path":  # a free, b=0, c'=0.2
        return True, False
    if label.startswith("level: neither"):  # a b = 0 and c' = 0
        return True, True
    if label == "level: horizontal only":  # a b real, c' = 0
        return False, True
    return False, False  # a power cell


print(f"{'cell':<42} {'reps':>5} {'vertical':>17} {'horizontal':>17}")
print("-" * 86)
for name in sorted(rows):
    result: ResultCell = rows[name]
    """Selected the completed result for the displayed cell."""
    n: int = int(result["replicates"])
    """Recovered the denominator for the cell's Monte Carlo rates."""
    v_null, h_null = truth_of(name)
    """Classified which reported tests were level measurements in this cell."""
    out: list[str] = []
    """Initialised the formatted summaries for the two estimands."""
    for reject, is_null in (
        (result["vertical_reject_025"], v_null),
        (result["horizontal_reject_025"], h_null),
    ):
        rate: float = float(reject) / n
        """Calculated the empirical rejection rate for this test."""
        se: float = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
        """Estimated the Monte Carlo standard error of the rejection rate."""
        if is_null:
            mark: str = "ok" if rate <= 0.025 + 2 * se else "HIGH"
            """Marked whether a level estimate stayed within two standard errors."""
            out.append(f"{rate:.3f} lvl {mark:>4}")
        else:
            out.append(f"{rate:.3f} pwr     ")
    """Formatted the level or power interpretation for both estimands."""
    print(f"{name:<42} {n:>5} {out[0]:>17} {out[1]:>17}")
"""Printed one interpretable row for every completed campaign cell."""

print("\n'lvl' is a level -- the estimand is nought there and rejection must sit")
print("at or below .025. 'pwr' is power. Every level cell above is a separate")
print("null and a test can hold against one and not another.")
