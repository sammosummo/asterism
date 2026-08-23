"""Do the numbers in the drafting pack match the cells they came from?

This page is written to be lifted into an application, so a transposed digit
here does not stay here. Everything quoted is recomputed from the result files.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

type ResultCell = dict[str, float | int]

RESULTS: str = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results"
"""Located the directory containing the campaign result shards."""
PAGE: Path = Path("/Users/samuelmathias/.claude/jobs/f010e916/tmp/aim2_drafting.html")
"""Located the drafting pack whose quoted figures are being verified."""

cells: dict[str, ResultCell] = {}
"""Initialised the combined mapping of completed simulation cells."""
for path in sorted(glob.glob(f"{RESULTS}/*.json")):
    with open(path, encoding="utf-8") as stream:
        loaded: dict[str, dict[str, ResultCell]] = json.load(stream)
        """Loaded one result shard with its named simulation cells."""
    for name, r in loaded.get("cells", {}).items():
        if isinstance(r, dict) and r.get("replicates"):
            cells[name] = r
            """Retained this cell because it contained completed replicates."""
"""Combined all completed cells before verifying the drafting pack."""
page: str = PAGE.read_text(encoding="utf-8")
"""Read the drafting pack text containing the quoted figures."""

claims: list[tuple[str, str, str]] = [
    # effect-size curve
    ("0.055", "ABR only | vertical 0.06", "vertical"),
    ("0.070", "ABR only | vertical 0.12", "vertical"),
    ("0.230", "ABR only | vertical 0.18", "vertical"),
    ("0.455", "ABR only | vertical 0.24", "vertical"),
    ("0.080", "audiogram | vertical 0.06", "vertical"),
    ("0.160", "audiogram | vertical 0.12", "vertical"),
    ("0.335", "audiogram | vertical 0.18", "vertical"),
    ("0.545", "audiogram | vertical 0.24", "vertical"),
    # design-change grid
    ("0.520", "baseline | vertical 0.24", "vertical"),
    ("0.660", "staged outcome | vertical 0.24", "vertical"),
    ("0.695", "probands x1.5 | vertical 0.24", "vertical"),
    ("0.710", "sample x1.5 | vertical 0.24", "vertical"),
    ("0.845", "probands x2 | vertical 0.24", "vertical"),
    ("0.585", "hearing precise | vertical 0.24", "vertical"),
    ("0.550", "drop acoustic units | vertical 0.24", "vertical"),
    # design-change levels
    ("0.010", "baseline | null: no path", "vertical"),
    ("0.005", "staged outcome | null: no path", "vertical"),
    # direct-path grid
    ("0.115", "ABR only | direct 0.3", "horizontal"),
    ("0.300", "ABR only | direct 0.5", "horizontal"),
    ("0.500", "ABR only | direct 0.7", "horizontal"),
    ("0.155", "audiogram | direct 0.3", "horizontal"),
    ("0.330", "audiogram | direct 0.5", "horizontal"),
    ("0.545", "audiogram | direct 0.7", "horizontal"),
]
"""Enumerated each quoted rate and the result cell that should reproduce it."""

problems: list[str] = []
"""Initialised descriptions of discrepancies found in the drafting pack."""
for claimed, cell, test in claims:
    if cell not in cells:
        problems.append(f"{cell}: no such cell, page quotes {claimed}")
        continue
    result: ResultCell = cells[cell]
    """Selected the completed result underlying this drafting claim."""
    actual: float = float(result[f"{test}_reject_025"]) / int(result["replicates"])
    """Recomputed the quoted rejection rate from its counts."""
    if abs(actual - float(claimed)) > 5e-4:
        problems.append(f"{cell} ({test}): page {claimed}, cells {actual:.3f}")
    if claimed not in page:
        problems.append(f"{cell}: {claimed} not on the page")
"""Checked every quoted rate against its source cell and the rendered page."""

# The leak medians come from their own file.
leak_path: Path = Path(RESULTS) / "aim2_leak.json"
"""Located the separate result file containing direct-path leak medians."""
if leak_path.exists():
    with leak_path.open(encoding="utf-8") as stream:
        leak: dict[str, ResultCell] = json.load(stream)["cells"]
        """Loaded the completed leak cells from their dedicated result file."""
    for key, claimed in (("0.0", "0.233"), ("0.7", "0.284")):
        leak_actual: float = float(leak[key]["median_vertical"])
        """Read the simulated median vertical estimand for this direct path."""
        if abs(leak_actual - float(claimed)) > 5e-4:
            problems.append(f"leak at c'={key}: page {claimed}, file {leak_actual:.3f}")
        if claimed not in page:
            problems.append(f"leak median {claimed} not on the page")
    """Checked both published leak medians against the dedicated result file."""
else:
    problems.append("the leak result file is missing")


# The level count, recomputed rather than remembered.
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
        return True, False
    if label.startswith("level: neither"):
        return True, True
    if label in ("level: horizontal only", "direct 0.0"):
        return False, True
    return False, False


total: int = 0
"""Initialised the number of distinct level measurements in the drafting pack."""
for name, _r in cells.items():
    if " | " not in name:
        continue
    v_null, h_null = nulls(name)
    """Classified both estimands under this cell's simulated condition."""
    total += int(v_null) + int(h_null)
    """Counted every estimand that was nought in the cell."""
"""Recomputed the total number of level measurements from the cell definitions."""
if "Sixteen separate level measurements" in page and total != 16:
    problems.append(f"the page says sixteen level measurements; there are {total}")

if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
else:
    print(
        f"all {len(claims)} quoted rates, the leak medians and the level "
        f"count ({total}) check out"
    )
