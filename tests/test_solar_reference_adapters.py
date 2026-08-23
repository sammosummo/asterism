"""Portable contracts for frozen native-SOLAR comparison fixtures."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from checks.external_reference_fixture import verify

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located adapters and frozen fixtures in the current checkout."""

SOLAR_ADAPTERS: tuple[tuple[str, str], ...] = (
    ("against_solar", "against_solar.py"),
    ("bivariate_against_solar", "bivariate_against_solar.py"),
    ("liability_against_solar", "liability_against_solar.py"),
    ("mixed_bivariate_against_solar", "mixed_bivariate_against_solar.py"),
    ("spatial_against_solar", "spatial_against_solar.py"),
)
"""Named strict native-SOLAR adapters covered by the portable contract."""


@pytest.mark.parametrize(("check_id", "adapter_name"), SOLAR_ADAPTERS)
def test_solar_adapter_verifies_frozen_outputs_without_solar(
    check_id: str,
    adapter_name: str,
) -> None:
    """Recompute Asterism against a complete frozen native-SOLAR result."""
    adapter: Path = Path("checks") / adapter_name
    """Matched the repository-relative adapter identity recorded by release refresh."""

    fixture_path: Path = ROOT / "checks" / "external_fixtures" / f"{check_id}.json"
    """Selected the versioned independent-output fixture for this check."""

    result: dict[str, object] = verify(check_id, adapter, fixture_path)
    """Ran portable verification through the fail-closed public driver seam."""

    assert result["external_tool_invoked"] is False
    assert result["asterism_recomputed"] is True

    fixture: dict[str, object] = json.loads(fixture_path.read_text(encoding="utf-8"))
    """Read the envelope to check the qualified native executable identity."""

    assert fixture["tools"] == [
        {
            "name": "/usr/local/bin/solar",
            "version": "SOLAR Eclipse 9.0.0 (2022-01-01)",
            "packages": [],
        }
    ]


@pytest.mark.parametrize(("check_id", "adapter_name"), SOLAR_ADAPTERS)
def test_solar_adapter_verify_never_reaches_external_execution(
    check_id: str,
    adapter_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove portable mode does not call a native version probe or SOLAR run."""
    adapter: Path = ROOT / "checks" / adapter_name
    """Selected the frozen adapter for direct branch-level verification."""

    fixture_path: Path = ROOT / "checks" / "external_fixtures" / f"{check_id}.json"
    """Selected the existing independently refreshed native-output envelope."""

    module_name: str = f"solar_reference_test_{check_id}"
    """Chose an isolated import identity so adapters cannot share mutable state."""

    specification: ModuleSpec | None = importlib.util.spec_from_file_location(
        module_name, adapter
    )
    """Built a loader for the exact source file bound into the fixture."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the isolated adapter module without running its command entrypoint."""

    sys.modules[module_name] = module
    """Registered the module so dataclass metadata can resolve its owner."""

    try:
        specification.loader.exec_module(module)
        forbidden: Mock = Mock(
            side_effect=AssertionError("portable verification reached native SOLAR")
        )
        """Prepared a sentinel that fails on any external execution attempt."""

        monkeypatch.setattr(module, "solar_tool_identity", forbidden)
        monkeypatch.setattr(module, "run_solar", forbidden)
        reference_record: Callable[[str, Path | None], dict[str, object]] = (
            module.reference_record
        )
        """Selected the adapter's strict public two-mode seam."""

        result: dict[str, object] = reference_record("verify", fixture_path)
        """Recomputed Asterism with both external-execution paths disabled."""

        assert result["passed"] is True
        assert result["external_tool_invoked"] is False
        forbidden.assert_not_called()
    finally:
        sys.modules.pop(module_name, None)
    """Removed the isolated adapter module after the portable proof."""
