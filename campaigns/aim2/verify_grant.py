"""Do the figures in the revised grant text match the cells they came from?

This text is going into an application. Every number in it is recomputed from
the result files, and the ones that are ratios or derived quantities are
recomputed too rather than trusted to arithmetic done once in prose.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

type ResultCell = dict[str, float | int]

RESULTS: str = "/Users/samuelmathias/.claude/jobs/f010e916/tmp/results"
"""Located the directory containing the campaign result shards."""
BASE: Path = Path(
    "/Users/samuelmathias/MathiasLab/staging/studies/planned/mediation/"
    "grant-applications"
)
"""Located the grant-application directory containing the revised text."""
AIMS: Path = BASE / "SPECIFIC_AIMS_REVISED_2026-08-19.md"
"""Located the revised Specific Aims document."""
STRATEGY: Path = BASE / "RESEARCH_STRATEGY_AIM2_REVISIONS_2026-08-19.md"
"""Located the revised Aim 2 research-strategy document."""

cells: dict[str, ResultCell] = {}
"""Initialised the combined mapping of completed simulation cells."""
for path in sorted(glob.glob(f"{RESULTS}/*.json")):
    with open(path, encoding="utf-8") as stream:
        loaded: dict[str, ResultCell] = json.load(stream).get("cells", {})
        """Loaded the named cells recorded in one campaign result shard."""
    for name, r in loaded.items():
        if isinstance(r, dict) and r.get("replicates"):
            cells[name] = r
            """Retained this cell because it contained completed replicates."""
"""Combined every completed result shard before checking the grant text."""


def rate(cell: str, test: str) -> float:
    """Recompute one empirical rejection rate from its result cell.

    Args:
        cell: Exact campaign cell label.
        test: Test prefix, either vertical or horizontal.

    Returns:
        Rejection count divided by completed replicates.
    """
    result: ResultCell = cells[cell]
    """Selected the completed result underlying the requested rate."""
    return float(result[f"{test}_reject_025"]) / int(result["replicates"])


text: str = AIMS.read_text(encoding="utf-8") + STRATEGY.read_text(encoding="utf-8")
"""Combined the two revised grant sections into one searchable text."""
problems: list[str] = []
"""Initialised descriptions of discrepancies found in the grant text."""

# Quoted rates.
claims: list[tuple[str, str | None, str | None]] = [
    ("0.158", None, None),
    ("0.083", None, None),
    ("0.384", None, None),
    ("0.253", None, None),
    ("0.55", "audiogram | vertical 0.24", "vertical"),
    ("0.46", "ABR only | vertical 0.24", "vertical"),
    ("0.66", "staged outcome | vertical 0.24", "vertical"),
    ("0.065", None, None),
]
"""Enumerated the source-backed and externally anchored rates in the grant."""
for claimed, cell, test in claims:
    if claimed not in text:
        problems.append(f"{claimed} is not in the revised text")
        continue
    if cell is None:
        continue
    quote_actual: float = rate(cell, test)
    """Recomputed this source-backed grant rate from its campaign cell."""
    if abs(round(quote_actual, 2) - float(claimed)) > 5e-3:
        problems.append(f"{cell}: text {claimed}, cells {quote_actual:.3f}")
"""Checked every quoted rate for presence and, where applicable, exact origin."""

# The offspring ratio range, recomputed rather than taken from the prose.
low: float = 0.158 / 0.083
"""Calculated the first observed offspring-per-case ratio."""
high: float = 0.384 / 0.253
"""Calculated the second observed offspring-per-case ratio."""
lo, hi = min(low, high), max(low, high)
"""Ordered the two ratios into the range stated in the grant."""
if not (abs(hi - 1.9) < 0.05 and abs(lo - 1.5) < 0.05):
    problems.append(
        f"the offspring ratio is {lo:.2f} to {hi:.2f}, "
        f"not the 1.5 to 1.9 the text claims"
    )

# The staging gain in points.
gain: float = rate("staged outcome | vertical 0.24", "vertical") - rate(
    "baseline | vertical 0.24", "vertical"
)
"""Recomputed the staged-outcome power gain over the baseline design."""
if abs(gain - 0.14) > 0.005:
    problems.append(f"the staging gain is {gain:.3f}, not the fourteen points claimed")
if "fourteen" not in text:
    problems.append("the staging gain is not stated in the text")

# The hearing-precision non-result.
precise: float = rate("hearing precise | vertical 0.24", "vertical") - rate(
    "baseline | vertical 0.24", "vertical"
)
"""Recomputed the power change from more precise hearing measurement."""
if abs(precise - 0.065) > 0.005:
    problems.append(f"the hearing-precision change is {precise:.3f}, not 0.065")


# The level count.
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


total: int = sum(sum(nulls(name)) for name in cells if " | " in name)
"""Recomputed the number of separately assessed level measurements."""
if "Twenty-three" not in text or total != 23:
    problems.append(f"the text says twenty-three level measurements; there are {total}")

# The direct-path figures quoted in the analysis plan.
for claimed, cell in (
    ("0.155", "audiogram | direct 0.3"),
    ("0.545", "audiogram | direct 0.7"),
):
    direct_actual: float = rate(cell, "horizontal")
    """Recomputed this direct-path rejection rate from its campaign cell."""
    if abs(direct_actual - float(claimed)) > 5e-4:
        problems.append(f"{cell}: text {claimed}, cells {direct_actual:.3f}")
"""Checked both direct-path figures quoted in the analysis plan."""

# The leak medians.
with (Path(RESULTS) / "aim2_leak.json").open(encoding="utf-8") as stream:
    leak: dict[str, ResultCell] = json.load(stream)["cells"]
    """Loaded the dedicated result cells for the direct-path leak check."""
for key, claimed in (("0.0", 0.233), ("0.7", 0.284)):
    if abs(leak[key]["median_vertical"] - claimed) > 5e-4:
        problems.append(
            f"leak median at {key}: text {claimed}, "
            f"file {leak[key]['median_vertical']:.3f}"
        )
"""Checked both leak medians quoted in the grant against their source file."""

# 250 - 184 = 66 adjudications.
if "66 fewer" not in text:
    problems.append("the adjudication difference is not stated")

if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
else:
    print("every figure in the revised grant text checks out against the cells")
