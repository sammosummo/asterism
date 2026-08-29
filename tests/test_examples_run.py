"""Every example runs, and the README quotes what it actually prints.

The old README examples referenced variables that were never defined, so none
of them could be run and none of them could go stale visibly. These can, and
this notices when they do. New examples are picked up automatically: drop a
file in `examples/` and it is covered.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root holding both examples and the README."""

EXAMPLES: list[Path] = sorted(
    path for path in (ROOT / "examples").glob("*.py") if path.name != "synthetic.py"
)
"""Collected every example, excluding the shared data helper beside them."""


def run(example: Path) -> str:
    """Run one example exactly as the README tells a reader to run it.

    Args:
        example: The example program to run.

    Returns:
        Everything it printed.
    """
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(example)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    """Ran it from the repository root, as a reader would."""

    assert completed.returncode == 0, f"{example.name} failed:\n{completed.stderr}"
    return completed.stdout


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_the_example_runs(example: Path) -> None:
    """Keep every example a complete program rather than a fragment."""
    assert run(example).strip(), f"{example.name} printed nothing"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_the_docs_show_what_the_example_prints(example: Path) -> None:
    """Stop the documentation quoting output an example no longer produces.

    A page may link to an example without quoting it, which is what an overview
    does. But a page that quotes any of its output must quote all of it, and
    inside a code fence. An earlier version looked only for the lines anywhere
    on the page, and so passed while one example's output sat outside its fence
    and rendered as prose.
    """
    # **Recursive on purpose.** This globbed `docs/*.md` only, so nothing under
    # `docs/models/` was ever read -- a model page could quote an example's
    # output and the quote would go unchecked, the test skipping instead of
    # comparing. That is worse than not quoting it at all, because the page
    # looks verified.
    pages: list[Path] = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
    """Looked wherever an example may legitimately be quoted."""

    printed: str = run(example)
    """Produced the output afresh rather than trusting a stored copy."""

    first: str = next(line for line in printed.splitlines() if line.strip())
    """Took one printed line as the sign that a page quotes this example."""

    quoting: list[Path] = [
        page
        for page in pages
        if example.name in (text := page.read_text(encoding="utf-8")) and first in text
    ]
    """Found the pages that quote it: they name the example **and** show a line
    of its output. Merely linking to it is not quoting it, and a printed line on
    its own is not enough either -- two examples here open with
    `people:                 1000`, so once this began reading `docs/models/`
    each was matched against the other's page and both failed on lines that were
    never theirs."""

    if not quoting:
        pytest.skip(f"{example.name} output is not quoted anywhere")

    for page in quoting:
        text: str = page.read_text(encoding="utf-8")
        """Read one page that names this example."""

        fenced: list[str] = text.split("```")[1::2]
        """Took only what lies inside a code fence."""

        inside: str = "\n".join(fenced)
        """Joined every fenced block on the page."""

        for line in printed.splitlines():
            if line.strip():
                assert line in inside, (
                    f"{page.name} no longer shows this inside a code block: {line!r}"
                )
    """Required every printed line to appear, fenced, on every page naming it."""


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_the_example_recovers_the_truth_it_claims(example: Path) -> None:
    """Require any example printing a true value to still recover it."""
    for line in run(example).splitlines():
        if "(true " not in line:
            continue
        truth: float = float(line.split("(true ")[1].split(")")[0])
        """Read the true value the example says it is recovering."""

        estimate: float = float(line.split()[-1])
        """Read what it actually recovered."""

        assert abs(estimate - truth) < 0.2, (
            f"{example.name} recovered {estimate} for a true {truth}"
        )
    """Caught an example that quietly stopped recovering its own truth."""


def test_the_readme_quickstart_runs_and_prints_what_it_shows() -> None:
    """Keep the first code a reader pastes working, and honest about its output.

    The quickstart is inline rather than a link, because someone who installed
    Asterism from a wheel has no `examples/` directory. That makes it the one
    piece of code in the documentation with nothing else guarding it.
    """
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    """Read the front page."""

    section: str = readme[readme.index("## Quickstart") :]
    """Took the quickstart and whatever follows it."""

    code: str = section.split("```python\n", 1)[1].split("```", 1)[0]
    """Took the block a reader would copy."""

    expected: str = section.split("```\n", 2)[2].split("```", 1)[0]
    """Took the output the README claims that block produces."""

    script: Path = ROOT / "examples" / ".readme_quickstart_check.py"
    """Wrote it beside the examples so relative behaviour matches."""

    script.write_text(code, encoding="utf-8")
    try:
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        """Ran exactly what the README tells a reader to paste."""

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == expected.strip(), (
            "the README quickstart no longer prints what the README shows:\n"
            f"  printed:  {completed.stdout.strip()!r}\n"
            f"  README:   {expected.strip()!r}"
        )
    finally:
        script.unlink(missing_ok=True)
    """Removed the temporary copy however the check ended."""
