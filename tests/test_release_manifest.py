"""Public contract tests for Asterism release metadata."""

import subprocess
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from asterism import _core

from tools.check_release import (
    conditional_quantity_errors,
    fixed_build_errors,
)
from tools.run_scientific_release import scientific_inventory_errors

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root used by release tooling."""


def test_built_extension_embeds_the_authoritative_release_manifest() -> None:
    """Require the wheel to retain the exact committed release contract."""
    manifest_text: str = (ROOT / "release.toml").read_text(encoding="utf-8")
    """Read the one editable copy of the release manifest."""

    assert _core.__release_manifest__ == manifest_text


def test_release_metadata_agrees_with_the_authoritative_manifest() -> None:
    """Require every public version surface to carry one development identity."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "tools/check_release.py", "--metadata"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    """Ran the same metadata check used by local and hosted quality gates."""

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_stable_abi_starts_at_the_supported_python_floor() -> None:
    """Do not label wheels as compatible with unsupported Python versions."""
    cargo: dict[str, object] = tomllib.loads(
        (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    )
    """Parsed the Rust dependency features which determine the ABI3 wheel tag."""

    dependencies: dict[str, object] = cast(dict[str, object], cargo["dependencies"])
    """Selected Cargo dependency metadata."""

    pyo3: dict[str, object] = cast(dict[str, object], dependencies["pyo3"])
    """Selected PyO3's stable-ABI configuration."""

    assert pyo3["features"] == ["abi3-py313"]


def test_release_manifest_owns_the_exact_rust_toolchain() -> None:
    """Require manifest, Cargo, and repository compiler selection to agree."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative compiler identity selected for release evidence."""

    cargo: dict[str, object] = tomllib.loads(
        (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    )
    """Read Cargo's minimum accepted compiler release."""

    toolchain: dict[str, object] = tomllib.loads(
        (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    )
    """Read the exact compiler selected by local and hosted Rust commands."""

    expected: object = manifest["rust_toolchain"]
    """Selected the one authoritative Rust compiler version."""

    assert expected == "1.97.1"
    assert cargo["package"]["rust-version"] == expected  # type: ignore[index]
    assert toolchain["toolchain"]["channel"] == expected  # type: ignore[index]


def test_release_manifest_owns_exact_python_build_tools() -> None:
    """Require Maturin and uv metadata to match locks and hosted execution."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read authoritative Maturin and uv versions."""

    pyproject: dict[str, object] = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    """Read exact build-system and development requirements."""

    lock: dict[str, object] = tomllib.loads(
        (ROOT / "uv.lock").read_text(encoding="utf-8")
    )
    """Read the resolved Maturin package identity."""

    maturin: object = manifest["maturin_version"]
    """Selected the one authoritative wheel-builder version."""

    uv_version: object = manifest["uv_version"]
    """Selected the one authoritative workflow package-manager version."""

    assert maturin == "1.14.1"
    assert uv_version == "0.11.21"
    assert pyproject["build-system"]["requires"] == [  # type: ignore[index]
        f"maturin=={maturin}"
    ]
    assert f"maturin=={maturin}" in pyproject["dependency-groups"]["dev"]  # type: ignore[index]
    locked_maturin: list[dict[str, object]] = [
        package
        for package in lock["package"]  # type: ignore[union-attr]
        if package["name"] == "maturin"
    ]
    """Selected the sole resolved builder package from the complete lock."""

    assert [package["version"] for package in locked_maturin] == [maturin]
    for workflow_name in ("quality.yml", "wheels.yml", "release.yml"):
        workflow: str = (ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        """Read one hosted build surface that installs uv explicitly."""

        assert f'version: "{uv_version}"' in workflow
    wheels: str = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
    """Read the action-specific manylinux Maturin pin."""

    assert f"maturin-version: v{maturin}" in wheels


def test_scientific_gate_inventory_covers_every_required_check_once() -> None:
    """Require one participant-free command or explicit blocker for every claim."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the development manifest without promoting its readiness flags."""

    required: list[tuple[str, str]] = [
        (analysis["id"], check)
        for analysis in manifest["analyses"]
        for check in analysis["required_checks"]
    ]
    """Expanded all supported scientific claims with analysis ownership."""

    configured: list[tuple[str, str]] = [
        (analysis["id"], rule["id"])
        for analysis in manifest["analyses"]
        for rule in analysis["pass_rules"]
    ]
    """Expanded the exact executable or fail-closed command inventory."""

    assert len(required) == 31
    assert configured == required
    assert len(configured) == len(set(configured))
    assert scientific_inventory_errors(manifest, ROOT) == []


def test_inventory_does_not_claim_scientific_readiness() -> None:
    """Keep executable inventory distinct from measured passing release evidence."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read every global and per-analysis readiness switch."""

    assert manifest["scientific_pass_rules_configured"] is True
    assert all(
        analysis["pass_rules_configured"] is True for analysis in manifest["analyses"]
    )
    statuses: set[str] = {
        rule["status"]
        for analysis in manifest["analyses"]
        for rule in analysis["pass_rules"]
    }
    """Confirmed that missing evidence remains visible inside the inventory."""

    assert statuses <= {"ready", "blocked", "external_fixture_pending"}
    assert "ready" in statuses
    """Accepted evidence-progress transitions without changing readiness switches."""


def test_cross_platform_agreement_is_explicitly_unmeasured_not_implied() -> None:
    """Keep two installed-wheel matrices distinct from an actual result comparison."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative development contract."""

    agreement: dict[str, object] = cast(
        dict[str, object], manifest["cross_platform_agreement"]
    )
    """Selected the separately gated Mac/Linux comparison contract."""

    assert agreement["configured"] is True
    assert agreement["probe_runner"] == "tools/cross_platform_probe.py"
    assert agreement["runner"] == "tools/compare_cross_platform.py"
    assert agreement["exact_comparisons"] == [
        "outcome_status",
        "refusal_code",
        "boundary_state",
        "field_presence",
    ]
    tolerances: list[dict[str, object]] = agreement["model_tolerances"]  # type: ignore[assignment]
    """Read complete numeric-field inventories without pretending thresholds exist."""

    assert {tolerance["analysis_id"] for tolerance in tolerances} == {
        analysis["id"]
        for analysis in manifest["analyses"]  # type: ignore[union-attr]
    }
    assert all(tolerance["measured"] is True for tolerance in tolerances)
    assert all(tolerance["numeric_fields"] for tolerance in tolerances)
    assert all(
        set(tolerance["absolute"]) == set(tolerance["numeric_fields"])
        and set(tolerance["relative"]) == set(tolerance["numeric_fields"])
        for tolerance in tolerances
    )


def test_conditional_supported_sets_partition_the_component_inventory() -> None:
    """Select one honest component-reporting mode instead of requiring both."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative analysis inventory and conditional selectors."""

    component: dict[str, Any] = next(
        analysis
        for analysis in manifest["analyses"]
        if analysis["id"] == "several_covariance_components"
    )
    """Selected the analysis with mutually exclusive reporting conventions."""

    assert conditional_quantity_errors(component) == []
    assert component["supported_quantity_sets"] == [
        {
            "when": {"component_reporting": "mean_diagonal"},
            "quantities": [
                "mean_diagonal_component_contributions",
                "mean_diagonal_proportions",
                "mean_diagonal_proportion_interval",
            ],
        },
        {
            "when": {"component_reporting": "zero_diagonal"},
            "quantities": [
                "coefficients_for_zero_diagonal_bases",
                "contrasts_for_zero_diagonal_bases",
            ],
        },
    ]

    malformed: dict[str, Any] = deepcopy(component)
    """Copied the contract so one duplicate cannot mutate the authoritative parse."""

    malformed["supported_quantity_sets"][1]["quantities"].append(
        "mean_diagonal_proportions"
    )
    """Made one field falsely required by both mutually exclusive modes."""

    assert "must partition supported_quantities" in "\n".join(
        conditional_quantity_errors(malformed)
    )


def test_fixed_build_gate_refuses_stale_or_dirty_extension_identity() -> None:
    """Require release wheels to come from the checked-out commit and a clean tree."""
    core: Any = SimpleNamespace(
        __source_commit__="b" * 40,
        __source_dirty__=True,
    )
    """Represented a wheel built from another commit with local modifications."""

    errors: list[str] = fixed_build_errors(
        core,
        repository_commit="a" * 40,
        repository_dirty=False,
    )
    """Compared immutable wheel identity with the release checkout identity."""

    assert "compiled source commit" in errors[0]
    assert "compiled source is dirty" in errors[1]


def test_fixed_build_gate_accepts_current_clean_extension_identity() -> None:
    """Permit the identity gate only for the exact clean release commit."""
    core: Any = SimpleNamespace(
        __source_commit__="a" * 40,
        __source_dirty__=False,
    )
    """Represented a clean wheel built from the checked-out release commit."""

    assert (
        fixed_build_errors(
            core,
            repository_commit="a" * 40,
            repository_dirty=False,
        )
        == []
    )


def test_fixed_build_gate_independently_refuses_a_dirty_release_checkout() -> None:
    """Do not let build-script environment overrides certify mutable source."""
    core: Any = SimpleNamespace(
        __source_commit__="a" * 40,
        __source_dirty__=False,
    )
    """Represented an apparently clean wheel matching the current commit."""

    errors: list[str] = fixed_build_errors(
        core,
        repository_commit="a" * 40,
        repository_dirty=True,
    )
    """Compared that build metadata with an independently dirty Git checkout."""

    assert errors == [
        "release checkout is dirty after excluding ignored release outputs"
    ]
