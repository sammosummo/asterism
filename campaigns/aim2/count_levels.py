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

type ResultCell = dict[str, float | int]

cells: dict[str, ResultCell] = {}
"""Collected completed cells from every campaign result shard."""
for path in sorted(
    glob.glob("/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/*.json")
):
    with open(path, encoding="utf-8") as stream:
        loaded: dict[str, ResultCell] = json.load(stream)["cells"]
        """Loaded the simulation cells stored in one result shard."""
    for name, r in loaded.items():
        if (
            r.get("replicates")
            and {
                "vertical_reject_025",
                "horizontal_reject_025",
            }
            <= r.keys()
        ):
            cells[name] = r
            """Retained this completed cell because it reported both tests."""
"""Combined every completed cell without counting empty placeholders."""


def nulls(name: str) -> tuple[bool, bool]:
    """Identify which estimands are nought in a campaign cell.

    Args:
        name: Campaign cell label containing the simulated condition.

    Returns:
        Whether the vertical and horizontal estimands are respectively nought.
    """
    label: str = name.split("|")[1].strip()
    """Isolated the simulated condition from the campaign arm prefix."""
    if label in ("null: no loading", "null: no path"):
        return True, False  # a b = 0, c' = 0.2
    if label.startswith("level: neither"):
        return True, True  # a b = 0 and c' = 0
    if label == "level: horizontal only":
        return False, True  # a b real, c' = 0
    if label == "direct 0.0":
        return False, True  # a b real, c' = 0
    return False, False  # a power cell


total: int = 0
"""Initialised the number of distinct level measurements."""
failures: list[str] = []
"""Initialised descriptions of measurements above their nominal level."""
for name, r in sorted(cells.items()):
    v_null, h_null = nulls(name)
    """Classified the null status of both estimands for this cell."""
    for estimand, reject, is_null in (
        ("vertical", r["vertical_reject_025"], v_null),
        ("horizontal", r["horizontal_reject_025"], h_null),
    ):
        if not is_null:
            continue
        total += 1
        """Counted this test because its estimand was nought in the cell."""
        rate: float = float(reject) / int(r["replicates"])
        """Calculated the empirical rejection rate under the null."""
        se: float = math.sqrt(max(rate * (1 - rate), 1e-9) / int(r["replicates"]))
        """Estimated the Monte Carlo standard error of the rejection rate."""
        if rate > 0.025 + 2 * se:
            failures.append(f"{name} ({estimand}): {rate:.3f} +/-{se:.3f}")
            """Recorded a level estimate more than two standard errors high."""
"""Checked every applicable test against its own simulated null condition."""

print(f"{total} level measurements across {len(cells)} reporting cells")
if failures:
    print(f"{len(failures)} above nominal:")
    for f in failures:
        print(f"  {f}")
else:
    print("all at or below .025 within two Monte Carlo standard errors")
