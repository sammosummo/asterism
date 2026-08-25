"""Every example in `examples/` runs, and prints what the README says it does.

The old README examples referenced variables that were never defined, so none
of them could be run and none of them could go stale visibly. These can, and
this notices when they do.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root holding both examples and the README."""


def test_the_quickstart_runs_and_recovers_the_truth() -> None:
    """Keep the first thing a reader runs working, and honest about its answer."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    """Ran the example exactly as the README tells a reader to run it."""

    assert completed.returncode == 0, completed.stderr
    output: str = completed.stdout
    """Read what a reader would see on their own screen."""

    assert "people:        480" in output
    assert "converged:     True" in output

    heritability: float = float(
        next(line for line in output.splitlines() if line.startswith("h2")).split()[-1]
    )
    """Read the recovered heritability back out of the printed report."""

    assert 0.35 < heritability < 0.65, (
        f"the quickstart recovered {heritability}, which is no longer near its true 0.5"
    )
    """Required the example to keep recovering the truth it claims to recover."""


def test_the_readme_shows_what_the_quickstart_actually_prints() -> None:
    """Stop the README quoting output the example no longer produces."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    """Produced the output afresh rather than trusting a stored copy."""

    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    """Read the front page that quotes it."""

    for line in completed.stdout.splitlines():
        assert line in readme, f"the README no longer shows: {line!r}"
    """Required every printed line to appear in the README exactly."""
