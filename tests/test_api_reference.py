"""The generated API reference matches the package it documents.

Hand-written signatures drift. This one is generated from the docstrings, and
this test fails the moment the file on disk stops matching what the generator
produces, so a changed signature cannot ship with a stale reference.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root holding the generator and its output."""


def test_the_api_reference_is_not_stale() -> None:
    """Require docs/api-reference.md to match the built package."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_api_reference.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    """Asked the generator whether the file on disk is what it would write."""

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_public_name_appears_in_the_reference() -> None:
    """Stop a new public object shipping with no documentation at all."""
    import asterism

    reference: str = (ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
    """Read the generated reference."""

    for name in asterism.__all__:
        assert f"`{name}" in reference, f"{name} is public but not in the reference"
    """Required the reference to cover the package's own declared surface."""
