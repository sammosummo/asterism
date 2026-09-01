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


def test_the_manifest_version_has_the_right_changelog_and_release_page() -> None:
    """Require release notes for releases and Unreleased notes for development.

    A development identity must not manufacture a versioned release page. A
    fixed release must have both the matching changelog heading and page.
    """
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative version and release-state pair."""

    version: str = str(manifest["version"])
    """Selected the public version rendered into every built artifact."""

    changelog: str = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    """Read the package-wide change history."""

    if manifest.get("release") is not True:
        assert "dev" in version
        assert "## Unreleased" in changelog
        assert not (ROOT / f"docs/release-notes/{version}.md").exists()
        return
    """Kept mutable development work in the one Unreleased section."""

    release_notes: str = (ROOT / f"docs/release-notes/{version}.md").read_text(
        encoding="utf-8"
    )
    """Read the release page source for that exact version."""

    assert f"## {version}" in changelog, f"CHANGELOG.md has no {version} entry"
    assert f"# Asterism {version}" in release_notes
    assert f"- `asterism-{version}-cp313-abi3-macosx_11_0_arm64.whl`" in release_notes
    assert (
        f"- `asterism-{version}-cp313-abi3-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl`"
    ) in release_notes
