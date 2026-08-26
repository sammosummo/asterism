"""Public documentation contracts for Asterism's Python API."""

import inspect
import tomllib
from pathlib import Path

import asterism
import asterism.models as model_api

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root containing the versioned API reference."""


def test_every_exported_python_object_has_a_support_label() -> None:
    """Keep importability distinct from scientific support for every export."""
    reference: str = (ROOT / "docs/api-support.md").read_text(encoding="utf-8")
    """Read the public support-status inventory."""

    for name in asterism.__all__:
        assert f"| `{name}` |" in reference


def test_models_module_exports_every_public_wrapper_it_defines() -> None:
    """Keep star imports and generated API tools aligned with package exports."""
    public_definitions: set[str] = {
        name
        for name, value in vars(model_api).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__ == model_api.__name__
    }
    """Discovered public classes and functions genuinely defined by the module."""

    assert set(model_api.__all__) == public_definitions


def test_the_release_has_a_changelog_entry_and_a_release_page() -> None:
    """Require both versioned documents to cover the manifest's version.

    This used to assert the changelog said "planned" and the release page said
    "Not released", which made it a test of when it was written. It now checks
    that whatever version `release.toml` carries is described in both places,
    which is what the documents are for.
    """
    version: str = tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))[
        "version"
    ]
    """Read the version both documents must describe."""

    changelog: str = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    """Read the package-wide change history."""

    release_notes: str = (ROOT / f"docs/release-notes/{version}.md").read_text(
        encoding="utf-8"
    )
    """Read the release page source for that exact version."""

    assert f"## {version}" in changelog, f"CHANGELOG.md has no {version} entry"
    assert f"# Asterism {version}" in release_notes
