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


DISALLOWED: tuple[str, ...] = (
    "operatorname",
    "def",
    "newcommand",
    "renewcommand",
    "let",
    "require",
)
"""Named macros GitHub's MathJax refuses, whatever LaTeX itself permits.

GitHub answers `\\operatorname` with "The following macros are not allowed",
and shows the equation as raw text. `\\mathrm` renders identically and is
allowed.
"""


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_no_macro_github_refuses(page: Path) -> None:
    """Keep every equation to the macros GitHub will actually render."""
    body: str = prose(page)
    """Read everything outside a code fence."""

    found: list[str] = [
        macro for macro in DISALLOWED if re.search(rf"\\{macro}\b", body)
    ]
    """Collected any macro GitHub would refuse."""

    assert not found, (
        f"{page.name} uses macros GitHub refuses to render: "
        f"{', '.join('\\\\' + name for name in found)}"
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_no_macro_has_lost_its_backslash(page: Path) -> None:
    """Catch a macro written as a bare word, which renders as that word."""
    known: tuple[str, ...] = (
        "qquad",
        "quad",
        "frac",
        "tfrac",
        "sqrt",
        "mathbf",
        "boldsymbol",
        "widehat",
        "mathrm",
        "sigma",
        "rho",
        "theta",
        "alpha",
        "beta",
        "gamma",
        "lambda",
        "chi",
        "times",
        "cdot",
    )
    """Named macros common enough here that a bare one is a mistake."""

    text: str = prose(page)
    """Read everything outside a code fence."""

    spans: list[str] = [
        match.group(0) for match in re.finditer(r"(?ms)^\$\$$.*?^\$\$$", text)
    ] + re.findall(r"\$[^$\n]+\$", text)
    """Took every stretch of mathematics, display and inline."""

    problems: list[str] = [
        f"{macro!r} without its backslash"
        for span in spans
        for macro in known
        if re.search(rf"(?<![\\a-zA-Z]){macro}(?![a-zA-Z])", span)
    ]
    """Found a macro name sitting inside mathematics as a bare word."""

    assert not problems, f"{page.name}: {'; '.join(sorted(set(problems))[:4])}"
