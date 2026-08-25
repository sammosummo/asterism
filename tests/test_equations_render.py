"""Equations are written so GitHub actually renders them.

The markup can be valid LaTeX and still show as raw text on GitHub, which is
where anyone reading these documents will be. These are the rules GitHub
imposes beyond LaTeX itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root holding the documentation."""

PAGES: list[Path] = sorted(
    [*(ROOT / "docs").rglob("*.md"), ROOT / "README.md"],
)
"""Collected every document a reader might open on GitHub."""


def prose(page: Path) -> str:
    """Return one document with fenced code removed.

    Args:
        page: The document to read.

    Returns:
        Everything outside a code fence, where a dollar means mathematics.
    """
    text: str = page.read_text(encoding="utf-8")
    """Read the document."""

    return "\n".join(
        part for index, part in enumerate(text.split("```")) if index % 2 == 0
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_inline_mathematics_never_wraps(page: Path) -> None:
    """Keep inline mathematics on one line, or GitHub shows it as raw text."""
    body: str = re.sub(r"(?m)^\$\$$.*?^\$\$$", "", prose(page), flags=re.S)
    """Removed display blocks, which are allowed to span lines."""

    wrapped: list[str] = [
        line.strip() for line in body.splitlines() if line.count("$") % 2
    ]
    """Found lines opening or closing a dollar without its partner."""

    assert not wrapped, (
        f"{page.name}: inline mathematics wraps to the next line, which GitHub "
        f"does not render:\n  " + "\n  ".join(wrapped[:4])
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_display_blocks_stand_alone(page: Path) -> None:
    """Require a blank line around every display block, as GitHub needs."""
    lines: list[str] = prose(page).splitlines()
    """Read the document a line at a time."""

    problems: list[str] = []
    """Collected every block GitHub would refuse to render."""

    for index, line in enumerate(lines):
        if line.strip() != "$$":
            if line.strip().startswith("$$"):
                problems.append(f"line {index + 1}: text shares the $$ line")
            continue
        opener: bool = sum(1 for x in lines[:index] if x.strip() == "$$") % 2 == 0
        """Decided whether this delimiter opens or closes a block."""

        if opener and index and lines[index - 1].strip():
            problems.append(f"line {index + 1}: no blank line before the block")
        if not opener and index + 1 < len(lines) and lines[index + 1].strip():
            problems.append(f"line {index + 1}: no blank line after the block")
    assert not problems, f"{page.name}: " + "; ".join(problems[:4])
