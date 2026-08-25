"""The package is ready to distribute, and cannot be distributed by accident.

Two things at once. The metadata a reader would see on PyPI is complete, so
publishing is a decision rather than a scramble. And the guard that stops
anything reaching PyPI is still there, because taking it off is a step ADR 0018
says nobody should take without meaning to.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root holding the packaging metadata."""

PROJECT: dict[str, Any] = tomllib.loads(
    (ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]
"""Read the declared package metadata once."""


def test_nothing_can_reach_pypi_by_accident() -> None:
    """Keep the classifier PyPI refuses, which is why an upload cannot happen."""
    assert "Private :: Do Not Upload" in PROJECT["classifiers"], (
        "the upload guard is gone; ADR 0018 removes it deliberately, at "
        "publication, and not before"
    )


def test_the_metadata_a_reader_would_see_is_complete() -> None:
    """Require the fields that make a package page worth reading."""
    assert PROJECT["readme"] == "README.md"
    assert PROJECT["license"] == {"file": "LICENSE"}
    assert PROJECT["authors"] == [{"name": "Samuel Mathias"}]
    assert PROJECT["keywords"]
    assert (ROOT / PROJECT["readme"]).is_file()
    assert (ROOT / PROJECT["license"]["file"]).is_file()
    """Refused metadata pointing at files that are not there."""

    urls: dict[str, str] = PROJECT["urls"]
    """Read the links a package page offers."""

    assert "Repository" in urls
    for label, target in urls.items():
        assert target.startswith("https://"), f"{label} is not an https link"
    """Required every declared link to be a real one."""


def test_the_classifiers_say_what_this_is() -> None:
    """Stop the package page describing nothing but its development status."""
    classifiers: list[str] = PROJECT["classifiers"]
    """Read the declared classifiers."""

    for prefix in (
        "License :: ",
        "Intended Audience :: ",
        "Topic :: ",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Rust",
    ):
        assert any(entry.startswith(prefix) for entry in classifiers), (
            f"no classifier begins {prefix!r}"
        )
    """Required the page to say the licence, the audience and both languages."""


def test_the_declared_licence_matches_the_licence_file() -> None:
    """Refuse metadata claiming one licence while the file grants another."""
    licence: str = (ROOT / "LICENSE").read_text(encoding="utf-8")
    """Read the licence actually granted."""

    assert "MIT License" in licence
    assert any(
        entry == "License :: OSI Approved :: MIT License"
        for entry in PROJECT["classifiers"]
    )
    """Held the classifier and the file to the same answer."""
