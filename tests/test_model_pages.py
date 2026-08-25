"""Every supported model has a page, and its interface section is current.

The pages are hand-written above their API marker and generated below it. This
fails when a signature changes without the page being rebuilt, and when a model
is added to the manifest without a page to describe it.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root holding the pages and the manifest."""

MANIFEST: dict[str, Any] = tomllib.loads(
    (ROOT / "release.toml").read_text(encoding="utf-8")
)
"""Read the analyses whose pages must exist."""

PAGES: Path = ROOT / "docs" / "models"
"""Located the directory of per-model pages."""


def test_the_generated_interface_sections_are_not_stale() -> None:
    """Require every page's interface to match the built package."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_model_pages.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    """Asked the generator whether the pages on disk are what it would write."""

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "analysis",
    MANIFEST["analyses"],
    ids=lambda entry: entry["id"],
)
def test_every_supported_model_has_a_page(analysis: dict[str, Any]) -> None:
    """Stop a supported model shipping with nowhere describing it."""
    slugs: dict[str, str] = import_module("tools.build_model_pages").SLUGS
    """Read the mapping the generator uses, rather than repeating it here."""

    assert analysis["id"] in slugs, f"{analysis['id']} has no page"
    page: Path = PAGES / f"{slugs[analysis['id']]}.md"
    """Located the page this analysis writes to."""

    assert page.is_file()
    text: str = page.read_text(encoding="utf-8")
    """Read it."""

    for heading in ("## The model", "## Validation", "## Interface"):
        assert heading in text, f"{page.name} has no {heading!r}"
    """Required each page to carry the equations, the evidence and the API."""

    for entry in analysis["entry_points"]:
        assert f"`{entry.removeprefix('asterism.')}" in text, (
            f"{page.name} does not document {entry}"
        )
    """Required every declared entry point to appear on its model's page."""


def test_the_index_lists_every_page() -> None:
    """Keep the index from silently omitting a model."""
    index: str = (PAGES / "README.md").read_text(encoding="utf-8")
    """Read the index."""

    for page in sorted(PAGES.glob("*.md")):
        if page.name == "README.md":
            continue
        assert page.name in index, f"the index omits {page.name}"
    """Required every page to be reachable from the index."""
