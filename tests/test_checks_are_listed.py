"""Every script in `checks/` is named in the document that lists them.

`docs/development.md` carries the list and states why: a
script missing from the list is a script nobody runs, which is how the
association comparison sat broken and unnoticed through a change to the
interface it called.

A rule written in prose is kept by whoever remembers it. Twelve scripts had
drifted off the list before this test existed -- every check added across a
run of work on the censored, mixed bivariate and mediation models -- so
remembering was not enough.

The test asserts only that the name appears somewhere in the document. It
cannot tell whether the command beside it is right, or whether anybody has run
it lately. What it does catch is the failure that actually happened: a script
added and never listed at all.
"""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
"""Located the repository root from this test module."""

DOCUMENT: Path = ROOT / "docs" / "development.md"
"""Located the canonical inventory of scientific checks."""

CHECKS: Path = ROOT / "checks"
"""Located the maintained independent-check scripts."""


def test_every_check_is_listed_in_the_validation_document() -> None:
    """Require every maintained check script to appear in the inventory."""
    text: str = DOCUMENT.read_text()
    """Read the human-facing validation inventory."""

    unlisted: list[str] = sorted(
        script.name for script in CHECKS.glob("*.py") if script.name not in text
    )
    """Collected check scripts absent from the inventory."""

    assert not unlisted, (
        "these scripts in checks/ are not named in "
        f"docs/development.md, so nobody runs them: "
        f"{', '.join(unlisted)}"
    )


def test_the_document_does_not_name_a_check_that_has_gone() -> None:
    """The list should not promise a script that no longer exists either.

    The opposite drift to the one above, and quieter: a command copied out of
    the document fails at the shell rather than silently doing nothing, but it
    still says the suite covers something it does not.
    """
    text: str = DOCUMENT.read_text()
    """Read the human-facing validation inventory."""

    present: set[str] = {script.name for script in CHECKS.glob("*.py")}
    """Collected the check scripts that currently exist."""

    named: set[str] = {
        word.split("checks/")[1]
        for word in text.split()
        if "checks/" in word and word.endswith(".py")
    }
    """Extracted check-script names promised by the inventory."""

    gone: list[str] = sorted(named - present)
    """Collected promised check scripts that no longer exist."""

    assert not gone, (
        f"docs/development.md names scripts that are not in checks/: {', '.join(gone)}"
    )
