"""Cross-platform agreement contracts for installed Asterism wheels."""

from __future__ import annotations

import ast
import hashlib
import json
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tools.compare_cross_platform import (
    AgreementConfigurationError,
    compare_probe_records,
)
from tools.cross_platform_probe import NUMERIC_FIELDS_BY_ANALYSIS

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located release configuration and the participant-free probe source."""

ANALYSIS_IDS: tuple[str, ...] = tuple(f"analysis_{index}" for index in range(9))
"""Represented the complete nine-analysis release inventory in compact fixtures."""


def agreement_manifest() -> dict[str, Any]:
    """Return a configured nine-analysis agreement contract for tests.

    Returns:
        Parsed-manifest shape with measured per-analysis tolerances.
    """
    analyses: list[dict[str, str]] = [
        {"id": analysis_id} for analysis_id in ANALYSIS_IDS
    ]
    """Declared all nine supported analyses in stable order."""

    tolerances: list[dict[str, Any]] = [
        {
            "analysis_id": analysis_id,
            "measured": True,
            "numeric_fields": ["fit.estimate"],
            "absolute": {"fit.estimate": 0.01},
            "relative": {"fit.estimate": 0.02},
        }
        for analysis_id in ANALYSIS_IDS
    ]
    """Assigned an independently explicit absolute and relative tolerance to each."""

    return {
        "schema_version": 1,
        "version": "0.1.0",
        "release": True,
        "analyses": analyses,
        "cross_platform_agreement": {
            "configured": True,
            "probe_runner": "tools/cross_platform_probe.py",
            "runner": "tools/compare_cross_platform.py",
            "exact_comparisons": [
                "outcome_status",
                "refusal_code",
                "boundary_state",
                "field_presence",
            ],
            "model_tolerances": tolerances,
        },
    }


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the canonical parsed-manifest SHA-256 used by probe artifacts.

    Args:
        manifest: Parsed release manifest.

    Returns:
        Lowercase digest of canonical JSON bytes.
    """
    payload: bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    """Rendered TOML-independent canonical bytes matching the installed probe."""
    return hashlib.sha256(payload).hexdigest()


def probe_record(
    manifest: dict[str, Any],
    platform_name: str,
    architecture: str,
    python_version: str,
) -> dict[str, Any]:
    """Return one complete normalized installed-wheel probe artifact.

    Args:
        manifest: Parsed release contract shared by every target.
        platform_name: Canonical operating-system name.
        architecture: Canonical machine architecture.
        python_version: Supported major-minor interpreter version.

    Returns:
        Probe record covering every manifest analysis.
    """
    analyses: list[dict[str, Any]] = [
        {
            "analysis_id": analysis_id,
            "outcome_status": "candidate",
            "refusal_code": None,
            "boundary_state": {
                "interval.lower_limited": False,
                "interval.upper_limited": False,
            },
            "field_presence": [
                "fit",
                "fit.converged",
                "fit.estimate",
                "interval",
                "interval.lower_limited",
                "interval.upper_limited",
            ],
            "numeric_fields": {"fit.estimate": 10.0},
        }
        for analysis_id in ANALYSIS_IDS
    ]
    """Created one normalized result for every supported analysis."""

    return {
        "schema_version": 1,
        "probe_id": "asterism-cross-platform-0.1",
        "manifest_sha256": manifest_digest(manifest),
        "build": {
            "version": "0.1.0",
            "source_commit": "a" * 40,
            "source_dirty": False,
            "release": True,
            "wheel_sha256": "b" * 64,
        },
        "runtime": {
            "platform": platform_name,
            "architecture": architecture,
            "python_version": python_version,
        },
        "analyses": analyses,
    }


def complete_probe_matrix(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all four platform/interpreter probe artifacts.

    Args:
        manifest: Parsed release contract shared by every target.

    Returns:
        macOS arm64 and Linux x86-64 probes on Python 3.13 and 3.14.
    """
    return [
        probe_record(manifest, "macos", "arm64", "3.13"),
        probe_record(manifest, "linux", "x86_64", "3.13"),
        probe_record(manifest, "macos", "arm64", "3.14"),
        probe_record(manifest, "linux", "x86_64", "3.14"),
    ]


def test_all_four_artifacts_agree_at_the_declared_tolerance_edge() -> None:
    """Accept equality at absolute plus relative tolerance, inclusively."""
    manifest: dict[str, Any] = agreement_manifest()
    """Selected a complete measured agreement contract."""

    records: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created every required installed-wheel target record."""

    for record in records:
        if record["runtime"]["platform"] == "linux":
            record["analyses"][0]["numeric_fields"]["fit.estimate"] = 10.21
            """Placed one Linux value on the inclusive numerical boundary."""
    """Placed Linux exactly at 0.01 + 0.02 times the larger magnitude."""

    result: dict[str, Any] = compare_probe_records(manifest, records)
    """Compared both Python-version platform pairs through the public comparator."""

    assert result["passed"] is True
    assert result["status"] == "passed"
    assert len(result["comparisons"]) == 2
    assert all(comparison["passed"] for comparison in result["comparisons"])


def test_numeric_difference_beyond_the_written_tolerance_fails() -> None:
    """Reject a platform drift just beyond the inclusive tolerance boundary."""
    manifest: dict[str, Any] = agreement_manifest()
    """Selected a complete measured agreement contract."""

    records: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created every required installed-wheel target record."""

    records[1]["analyses"][0]["numeric_fields"]["fit.estimate"] = 10.210_001
    """Moved one Linux result beyond its absolute-plus-relative allowance."""

    result: dict[str, Any] = compare_probe_records(manifest, records)
    """Compared the deliberately drifting platform result."""

    assert result["passed"] is False
    assert "numeric mismatch" in "\n".join(result["failures"])


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    [
        ("outcome_status", "diagnostic-only", "outcome_status mismatch"),
        ("refusal_code", "MODEL_REFUSED", "refusal_code mismatch"),
        (
            "boundary_state",
            {"interval.lower_limited": True, "interval.upper_limited": False},
            "boundary_state mismatch",
        ),
        (
            "field_presence",
            ["fit", "fit.converged", "fit.estimate"],
            "field_presence mismatch",
        ),
    ],
)
def test_exact_outcome_refusal_boundary_and_presence_mismatches_fail(
    field: str, replacement: object, expected: str
) -> None:
    """Reject every exact comparison named by ADR 0012.

    Args:
        field: Normalized exact field to change on Linux.
        replacement: Deliberately disagreeing value.
        expected: Diagnostic fragment naming the failed comparison.
    """
    manifest: dict[str, Any] = agreement_manifest()
    """Selected a complete measured agreement contract."""

    records: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created every required installed-wheel target record."""

    if field == "refusal_code":
        records[0]["analyses"][0]["outcome_status"] = "refused"
        """Made the Mac comparison side a valid refused outcome."""

        records[0]["analyses"][0]["refusal_code"] = "REFERENCE_REFUSAL"
        """Gave the Mac side a stable but deliberately distinct refusal code."""

        records[0]["analyses"][0]["numeric_fields"] = {}
        """Withheld candidate-only numeric outputs on the refused Mac side."""

        records[1]["analyses"][0]["outcome_status"] = "refused"
        """Made the Linux comparison side a valid refused outcome."""

        records[1]["analyses"][0]["numeric_fields"] = {}
        """Withheld candidate-only numeric outputs on the refused Linux side."""
    """Made the refusal-code mismatch internally valid on both compared targets."""

    records[1]["analyses"][0][field] = replacement
    """Changed exactly one required structural field on Linux."""

    result: dict[str, Any] = compare_probe_records(manifest, records)
    """Compared the deliberately inconsistent normalized results."""

    assert result["passed"] is False
    assert expected in "\n".join(result["failures"])


def test_missing_and_extra_numeric_fields_fail_closed() -> None:
    """Require each probe's numeric inventory to equal its tolerance contract."""
    manifest: dict[str, Any] = agreement_manifest()
    """Selected a complete measured agreement contract."""

    missing: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created a matrix whose first Mac probe will omit the declared field."""

    missing[0]["analyses"][0]["numeric_fields"] = {}
    """Removed the only declared numeric output."""

    with pytest.raises(AgreementConfigurationError, match="numeric fields"):
        compare_probe_records(manifest, missing)

    extra: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created a second matrix whose Linux probe will add an undeclared field."""

    extra[1]["analyses"][0]["numeric_fields"]["fit.extra"] = 1.0
    """Added an output for which no acceptance tolerance was written."""

    with pytest.raises(AgreementConfigurationError, match="numeric fields"):
        compare_probe_records(manifest, extra)


def test_source_version_and_artifact_completeness_fail_closed() -> None:
    """Refuse stale builds, mixed versions, and incomplete target matrices."""
    manifest: dict[str, Any] = agreement_manifest()
    """Selected a complete measured agreement contract."""

    stale_source: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created probes before changing one embedded source identity."""

    stale_source[1]["build"]["source_commit"] = "c" * 40
    """Made one Linux wheel come from a different commit."""

    with pytest.raises(AgreementConfigurationError, match="source commit"):
        compare_probe_records(manifest, stale_source)

    wrong_version: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created probes before changing one public version."""

    wrong_version[3]["build"]["version"] = "0.1.1"
    """Made one Python 3.14 Linux result come from another version."""

    with pytest.raises(AgreementConfigurationError, match="version"):
        compare_probe_records(manifest, wrong_version)

    incomplete: list[dict[str, Any]] = complete_probe_matrix(manifest)[:-1]
    """Dropped the Python 3.14 Linux artifact entirely."""

    with pytest.raises(AgreementConfigurationError, match="artifact matrix"):
        compare_probe_records(manifest, incomplete)


def test_unmeasured_tolerances_emit_a_nonpassing_record_only_when_requested() -> None:
    """Measure development deltas without weakening release readiness."""
    manifest: dict[str, Any] = agreement_manifest()
    """Selected a complete agreement shape before clearing its measurements."""

    manifest["cross_platform_agreement"]["configured"] = False
    """Withheld the release-wide opt-in while retaining executable runners."""

    for tolerance in manifest["cross_platform_agreement"]["model_tolerances"]:
        tolerance["measured"] = False
        """Marked one model's thresholds as not yet measured across real hosts."""

        tolerance.pop("absolute")
        tolerance.pop("relative")
    """Retained each numeric-field inventory while removing unmeasured thresholds."""

    records: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created all four artifacts against the explicitly unmeasured manifest."""

    records[1]["analyses"][0]["numeric_fields"]["fit.estimate"] = 10.25
    """Introduced a visible Python 3.13 Linux delta for the measurement record."""

    result: dict[str, Any] = compare_probe_records(
        manifest, records, allow_unmeasured=True
    )
    """Produced development evidence without claiming agreement passed."""

    assert result["configured"] is False
    assert result["passed"] is False
    assert result["status"] == "unmeasured"
    assert len(result["comparisons"]) == 2
    assert all(len(comparison["exact"]) == 36 for comparison in result["comparisons"])
    measured: dict[str, Any] = result["comparisons"][0]["numeric"][0]
    """Selected the independently introduced development delta."""

    assert measured == {
        "analysis_id": "analysis_0",
        "field": "fit.estimate",
        "reference": 10.0,
        "candidate": 10.25,
        "difference": 0.25,
        "relative_difference": 0.025,
        "absolute_tolerance": None,
        "relative_tolerance": None,
        "allowed_difference": None,
        "passed": None,
    }

    with pytest.raises(AgreementConfigurationError, match="not configured"):
        compare_probe_records(manifest, records)


def test_probe_source_uses_only_the_public_asterism_interface() -> None:
    """Keep installed-wheel probes independent of private positional bindings."""
    probe_path: Path = ROOT / "tools" / "cross_platform_probe.py"
    """Selected the exact source executed inside each installed-wheel environment."""

    source: str = probe_path.read_text(encoding="utf-8")
    """Read probe source as both text and syntax for independent checks."""

    tree: ast.Module = ast.parse(source)
    """Parsed imports so aliases cannot hide a private-module dependency."""

    assert "_core" not in source
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "asterism" or node.module.startswith("asterism."))
        for node in ast.walk(tree)
    )
    imported_asterism: bool = any(
        isinstance(node, ast.Import)
        and any(alias.name == "asterism" for alias in node.names)
        for node in ast.walk(tree)
    )
    """Required the package root to be the sole Asterism dependency."""
    assert imported_asterism


def test_probe_numeric_fields_equal_the_prewritten_manifest_inventory() -> None:
    """Prevent probe edits from silently changing which numbers tolerances cover."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the exact field-selection contract embedded in every wheel."""

    agreement: dict[str, Any] = manifest["cross_platform_agreement"]
    """Selected model-specific cross-platform tolerance metadata."""

    configured: dict[str, tuple[str, ...]] = {
        str(tolerance["analysis_id"]): tuple(tolerance["numeric_fields"])
        for tolerance in agreement["model_tolerances"]
    }
    """Indexed manifest paths in their pre-written comparison order."""

    assert configured == NUMERIC_FIELDS_BY_ANALYSIS


def test_comparison_record_is_json_serialisable_and_input_independent() -> None:
    """Emit a durable record without mutating downloaded probe artifacts."""
    manifest: dict[str, Any] = agreement_manifest()
    """Selected a complete measured agreement contract."""

    records: list[dict[str, Any]] = complete_probe_matrix(manifest)
    """Created all four probe records before preserving an independent copy."""

    before: list[dict[str, Any]] = deepcopy(records)
    """Captured probe inputs to detect comparator mutation."""

    result: dict[str, Any] = compare_probe_records(manifest, records)
    """Produced the complete machine-readable agreement result."""

    assert json.loads(json.dumps(result)) == result
    assert records == before
