"""Do the numbers published in the report match the result files?

The report was written by hand from tables printed to a terminal, which is
exactly where a digit gets transposed and then survives, because nothing
downstream ever compares the two again. This compares them.

Every rate quoted in the report is recomputed from the cells and checked
against what the page says.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

type ResultCell = dict[str, float | int]

RESULTS: str = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results/*.json"
"""Selected every result shard used to draft the report."""
REPORT: Path = Path("/Users/samuelmathias/.claude/jobs/f010e916/tmp/aim2_power.html")
"""Located the rendered report whose quoted figures are being verified."""

cells: dict[str, ResultCell] = {}
"""Initialised the combined mapping of completed simulation cells."""
for path in sorted(glob.glob(RESULTS)):
    with open(path, encoding="utf-8") as stream:
        loaded: dict[str, ResultCell] = json.load(stream)["cells"]
        """Loaded the cells recorded in one campaign result shard."""
    for name, r in loaded.items():
        if r.get("replicates"):
            cells[name] = r
            """Retained this cell because it contained completed replicates."""
"""Combined every completed result shard before checking the report."""

page: str = REPORT.read_text(encoding="utf-8")
"""Read the rendered report text containing the published figures."""

# (what the page claims, which cell, which test)
claims: list[tuple[str, str, str]] = [
    # vertical power
    ("0.055", "ABR only | vertical 0.06", "vertical"),
    ("0.070", "ABR only | vertical 0.12", "vertical"),
    ("0.230", "ABR only | vertical 0.18", "vertical"),
    ("0.455", "ABR only | vertical 0.24", "vertical"),
    ("0.080", "audiogram | vertical 0.06", "vertical"),
    ("0.160", "audiogram | vertical 0.12", "vertical"),
    ("0.335", "audiogram | vertical 0.18", "vertical"),
    ("0.545", "audiogram | vertical 0.24", "vertical"),
    # vertical level
    ("0.020", "ABR only | null: no loading", "vertical"),
    ("0.025", "ABR only | null: no path", "vertical"),
    ("0.035", "audiogram | null: no path", "vertical"),
    ("0.010", "ABR only | level: neither, no path", "vertical"),
    ("0.040", "audiogram | level: neither, no path", "vertical"),
    # horizontal level
    ("0.015", "ABR only | level: neither, no path", "horizontal"),
    ("0.030", "audiogram | level: neither, no path", "horizontal"),
    ("0.025", "ABR only | level: horizontal only", "horizontal"),
    ("0.030", "audiogram | level: horizontal only", "horizontal"),
    # direct-path grid
    ("0.115", "ABR only | direct 0.3", "horizontal"),
    ("0.300", "ABR only | direct 0.5", "horizontal"),
    ("0.500", "ABR only | direct 0.7", "horizontal"),
    ("0.155", "audiogram | direct 0.3", "horizontal"),
    ("0.330", "audiogram | direct 0.5", "horizontal"),
    ("0.545", "audiogram | direct 0.7", "horizontal"),
]
"""Enumerated every report rate and the result cell that should reproduce it."""

problems: list[str] = []
"""Initialised descriptions of discrepancies found in the report."""
for claimed, cell, test in claims:
    if cell not in cells:
        problems.append(f"{cell}: no such cell, but the page quotes {claimed}")
        continue
    result: ResultCell = cells[cell]
    """Selected the completed result underlying this reported claim."""
    actual: float = float(result[f"{test}_reject_025"]) / int(result["replicates"])
    """Recomputed the quoted rejection rate from its counts."""
    if abs(actual - float(claimed)) > 5e-4:
        problems.append(
            f"{cell} ({test}): page says {claimed}, cells give {actual:.3f}"
        )
    if claimed not in page:
        problems.append(f"{cell} ({test}): {claimed} does not appear on the page")
"""Checked every quoted rate against both its source cell and the rendered page."""

# The two figures in the summary panel.
for figure, why in (("1,600", "people"), ("257", "cases"), ("54.5%", "best power")):
    if figure not in page:
        problems.append(f"the {why} figure {figure} is not on the page")

if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
else:
    print(f"all {len(claims)} quoted rates match the result files")
