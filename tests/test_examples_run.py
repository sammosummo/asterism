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
def test_the_readme_shows_what_the_example_prints(example: Path) -> None:
    """Stop the README quoting output an example no longer produces."""
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    """Read the front page that quotes them."""

    if f"examples/{example.name}" not in readme:
        pytest.skip(f"{example.name} is not quoted in the README")
    for line in run(example).splitlines():
        if line.strip():
            assert line in readme, f"the README no longer shows: {line!r}"
    """Required every printed line to appear in the README exactly."""


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
