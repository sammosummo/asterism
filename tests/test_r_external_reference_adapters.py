"""Contracts for the five qualified R-backed frozen reference adapters."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from checks.external_reference_adapter import (
    ReferenceAdapterError,
    r_tool_identity,
    require_r_versions,
)

R_RULES: tuple[tuple[str, str], ...] = (
    ("against_r", "against_r.py"),
    ("bivariate_against_r", "bivariate_against_r.py"),
    ("tobit_against_censreg", "tobit_against_censreg.py"),
    ("tobit_against_mcmcglmm", "tobit_against_mcmcglmm.py"),
    ("spatial_against_spamm", "spatial_against_spamm.py"),
)
"""Named every R-backed rule and its public comparison adapter."""


def ready_r_rules() -> tuple[tuple[str, str], ...]:
    """Return only adapters whose current manifest rule claims ready evidence."""
    root: Path = Path(__file__).resolve().parents[1]
    """Located the manifest that owns each external rule's status."""

    manifest: dict[str, Any] = tomllib.loads(
        (root / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the current development release contract."""

    statuses: dict[str, str] = {
        rule["id"]: rule["status"]
        for analysis in manifest["analyses"]
        for rule in analysis.get("pass_rules", [])
    }
    """Indexed every declared pass rule by its readiness state."""

    return tuple(pair for pair in R_RULES if statuses.get(pair[0]) == "ready")


PORTABLE_R_RULES: tuple[tuple[str, str], ...] = ready_r_rules()
"""Excluded blocked fixtures until a genuine external refresh updates them."""

QUALIFIED_TOOLS: dict[str, tuple[str, dict[str, str]]] = {
    "against_r": ("4.5.2", {"regress": "1.3.22"}),
    "bivariate_against_r": ("4.5.2", {"regress": "1.3.22"}),
    "tobit_against_censreg": ("4.5.2", {"censReg": "0.5.38"}),
    "tobit_against_mcmcglmm": (
        "4.5.2",
        {"MCMCglmm": "2.36", "Matrix": "1.7.4", "coda": "0.19.4.1"},
    ),
    "spatial_against_spamm": (
        "4.5.2",
        {"spaMM": "4.6.65", "geoR": "1.9.6", "jsonlite": "2.0.0"},
    ),
}
"""Pinned the exact live R environments recorded by genuine refreshes."""


def repository_root() -> Path:
    """Return the checkout containing adapters, fixtures and the public extension."""
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("check_id", "adapter_name"), PORTABLE_R_RULES)
def test_portable_r_verification_does_not_need_r_on_path(
    check_id: str,
    adapter_name: str,
) -> None:
    """Recompute Asterism against each frozen R output with R unavailable."""
    root: Path = repository_root()
    """Located every configured command relative to the checkout root."""

    environment: dict[str, str] = dict(os.environ)
    """Preserved the built-extension environment while removing executable lookup."""

    environment["PATH"] = ""
    """Made Rscript and every other ambient executable unavailable to the adapter."""

    fixture: Path = root / "checks" / "external_fixtures" / f"{check_id}.json"
    """Selected the genuine versioned independent-reference envelope."""

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            str(root / "checks" / "external_reference_fixture.py"),
            "verify",
            "--check-id",
            check_id,
            "--adapter",
            str(Path("checks") / adapter_name),
            "--fixture",
            str(fixture.relative_to(root)),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    """Ran the exact participant-free release command without an R executable."""

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result: dict[str, Any] = json.loads(completed.stdout)
    """Parsed the strict driver's one machine-readable portable receipt."""

    assert result["passed"] is True
    assert result["result"]["external_tool_invoked"] is False
    assert result["result"]["asterism_recomputed"] is True


@pytest.mark.parametrize(("check_id", "adapter_name"), R_RULES)
def test_r_adapter_preserves_ordinary_no_argument_route(
    monkeypatch: pytest.MonkeyPatch,
    check_id: str,
    adapter_name: str,
) -> None:
    """Keep the historical human-readable live comparison as the default mode."""
    module_name: str = f"checks.{adapter_name.removesuffix('.py')}"
    """Converted the configured script path to its importable checkout module."""

    module: ModuleType = importlib.import_module(module_name)
    """Loaded the adapter without executing its command-line entry point."""

    sentinel: int = 37
    """Selected a return code distinct from both pass and failure."""

    monkeypatch.setattr(module, "main", lambda: sentinel)
    monkeypatch.setattr(sys, "argv", [adapter_name])
    assert module.entrypoint() == sentinel
    assert check_id == module.CHECK_ID


@pytest.mark.parametrize(("check_id", "adapter_name"), R_RULES)
def test_r_fixture_records_exact_qualified_versions(
    check_id: str,
    adapter_name: str,
) -> None:
    """Retain exact live R/package identity and the current adapter digest."""
    root: Path = repository_root()
    """Located the genuine versioned refresh output."""

    fixture_path: Path = root / "checks" / "external_fixtures" / f"{check_id}.json"
    """Selected one immutable independent-reference envelope."""

    fixture: dict[str, Any] = json.loads(fixture_path.read_text(encoding="utf-8"))
    """Parsed the committed deterministic fixture."""

    tool: dict[str, Any] = fixture["tools"][0]
    """Read the single Rscript tool record required by the contract."""

    # asterism-style: allow missing-following-doc -- R and package pins form one qualified environment
    expected_r, expected_packages = QUALIFIED_TOOLS[check_id]
    observed_packages: dict[str, str] = {
        package["name"]: package["version"] for package in tool["packages"]
    }
    """Normalized the ordered fixture list for an exact version comparison."""

    assert tool["name"] == "Rscript"
    assert tool["version"] == expected_r
    assert observed_packages == expected_packages
    assert fixture["adapter"]["path"] == f"checks/{adapter_name}"


def test_r_adapter_refuses_tampered_prewritten_acceptance(tmp_path: Path) -> None:
    """Reject a fixture whose numerical threshold was loosened after refresh."""
    root: Path = repository_root()
    """Located the genuine against-r fixture and strict driver."""

    source: Path = root / "checks" / "external_fixtures" / "against_r.json"
    """Selected the smallest deterministic R comparison for the refusal test."""

    fixture: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    """Parsed the genuine envelope before one intentional mutation."""

    fixture["acceptance"]["maximum_relative_estimate_difference"] = 1.0
    """Loosened a prewritten threshold while retaining a structurally valid object."""

    tampered: Path = tmp_path / "against_r.json"
    """Kept the malformed evidence outside the versioned fixture directory."""

    tampered.write_text(json.dumps(fixture), encoding="utf-8")
    """Persisted the structurally valid but scientifically altered envelope."""

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            str(root / "checks" / "external_reference_fixture.py"),
            "verify",
            "--check-id",
            "against_r",
            "--adapter",
            "checks/against_r.py",
            "--fixture",
            str(tampered),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    """Asked portable verification to accept the altered decision rule."""

    assert completed.returncode == 1
    assert "fixture acceptance was altered" in completed.stdout


def test_live_r_identity_refuses_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail a refresh host qualification when Rscript is unavailable."""
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ReferenceAdapterError, match="Rscript is not on the path"):
        r_tool_identity(("regress",))


def test_live_r_identity_refuses_unqualified_versions() -> None:
    """Reject a real-shaped tool record from an unqualified R/package release."""
    tools: list[dict[str, object]] = [
        {
            "name": "Rscript",
            "version": "4.5.1",
            "packages": [{"name": "regress", "version": "1.3.21"}],
        }
    ]
    """Constructed a complete but deliberately unqualified live environment."""

    with pytest.raises(ReferenceAdapterError, match=r"requires R 4\.5\.2"):
        require_r_versions(
            tools,
            r_version="4.5.2",
            packages={"regress": "1.3.22"},
        )
