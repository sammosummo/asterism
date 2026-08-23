"""Public documentation contracts for Asterism's Python API."""

import inspect
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


def test_planned_release_has_changelog_and_release_notes() -> None:
    """Keep 0.1 interpretation versioned without claiming it has shipped."""
    changelog: str = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    """Read the package-wide change history."""

    release_notes: str = (ROOT / "docs/release-notes/0.1.0.md").read_text(
        encoding="utf-8"
    )
    """Read the private release page source for the planned first release."""

    assert "0.1.0 (planned)" in changelog
    assert "Not released" in release_notes
