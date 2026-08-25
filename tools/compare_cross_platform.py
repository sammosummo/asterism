"""Compare normalized installed-wheel probes across promised 0.1 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

PROBE_ID: str = "asterism-cross-platform-0.1"
"""Named the deterministic probe protocol independently of file names."""

EXPECTED_TARGETS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("macos", "arm64", "3.13"),
        ("linux", "x86_64", "3.13"),
        ("macos", "arm64", "3.14"),
        ("linux", "x86_64", "3.14"),
    }
)
"""Required both supported interpreters on both promised wheel platforms."""

EXACT_COMPARISONS: frozenset[str] = frozenset(
    {"outcome_status", "refusal_code", "boundary_state", "field_presence"}
)
"""Fixed the structural comparisons accepted in ADR 0012."""

HEX40: re.Pattern[str] = re.compile(r"[0-9a-f]{40}")
"""Matched an embedded Git commit identity."""

HEX64: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")
"""Matched manifest and wheel SHA-256 identities."""


class AgreementConfigurationError(ValueError):
    """Report incomplete, inconsistent or stale agreement inputs."""


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Return a format-independent digest of one parsed release manifest.

    Args:
        manifest: Parsed authoritative ``release.toml`` content.

    Returns:
        Lowercase SHA-256 of canonical JSON bytes.
    """
    payload: bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    """Rendered the parsed contract identically from TOML and installed metadata."""
    return hashlib.sha256(payload).hexdigest()


def target_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """Return one probe's canonical platform/interpreter target.

    Args:
        record: Parsed normalized probe artifact.

    Returns:
        Platform, architecture and major-minor Python identity.

    Raises:
        AgreementConfigurationError: If runtime identity is incomplete.
    """
    runtime: object = record.get("runtime")
    """Read runtime identity without assuming a downloaded artifact is well formed."""

    if not isinstance(runtime, dict):
        raise AgreementConfigurationError("probe runtime must be an object")
    key: tuple[str, str, str] = (
        str(runtime.get("platform", "")),
        str(runtime.get("architecture", "")),
        str(runtime.get("python_version", "")),
    )
    """Normalised the three target coordinates used by the workflow matrix."""
    return key


def tolerance_contract(
    manifest: dict[str, Any], *, allow_unmeasured: bool
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Validate and index per-analysis numeric tolerance tables.

    Args:
        manifest: Parsed authoritative release manifest.
        allow_unmeasured: Whether an explicit development-state record may be emitted.

    Returns:
        Tolerances indexed by analysis identifier and configuration status.

    Raises:
        AgreementConfigurationError: If the comparison contract is malformed.
    """
    agreement: object = manifest.get("cross_platform_agreement")
    """Read the independently gated Mac/Linux contract."""

    if not isinstance(agreement, dict):
        raise AgreementConfigurationError(
            "cross-platform agreement configuration must be an object"
        )
    exact: object = agreement.get("exact_comparisons")
    """Read structural fields whose values may not differ by platform."""

    if not isinstance(exact, list) or {str(field) for field in exact} != set(
        EXACT_COMPARISONS
    ):
        raise AgreementConfigurationError(
            "cross-platform exact comparisons do not match ADR 0012"
        )
    analyses: object = manifest.get("analyses")
    """Read the complete supported-analysis inventory."""

    if not isinstance(analyses, list) or not all(
        isinstance(analysis, dict) and isinstance(analysis.get("id"), str)
        for analysis in analyses
    ):
        raise AgreementConfigurationError("manifest analyses are incomplete")
    analysis_ids: list[str] = [str(analysis["id"]) for analysis in analyses]
    """Preserved manifest order while validating exact tolerance coverage."""

    if not analysis_ids or len(set(analysis_ids)) != len(analysis_ids):
        raise AgreementConfigurationError("manifest analysis identifiers are invalid")
    raw_tolerances: object = agreement.get("model_tolerances")
    """Read the model-specific numeric comparison inventory."""

    if not isinstance(raw_tolerances, list) or not all(
        isinstance(tolerance, dict) for tolerance in raw_tolerances
    ):
        raise AgreementConfigurationError(
            "model-specific cross-platform tolerances must be tables"
        )
    tolerances: dict[str, dict[str, Any]] = {}
    """Indexed each validated tolerance table by supported analysis."""

    for tolerance in raw_tolerances:
        analysis_id: object = tolerance.get("analysis_id")
        """Read the supported analysis owning this numeric field inventory."""

        if not isinstance(analysis_id, str) or analysis_id in tolerances:
            raise AgreementConfigurationError(
                "model-specific tolerance analysis identifiers are invalid"
            )
        numeric_fields: object = tolerance.get("numeric_fields")
        """Read exact normalized result paths chosen for numerical comparison."""

        if (
            not isinstance(numeric_fields, list)
            or not numeric_fields
            or not all(isinstance(field, str) and field for field in numeric_fields)
            or len(set(numeric_fields)) != len(numeric_fields)
        ):
            raise AgreementConfigurationError(
                f"{analysis_id}: numeric field inventory is invalid"
            )
        measured: object = tolerance.get("measured")
        """Read whether actual cross-platform variation fixed these thresholds."""

        if not isinstance(measured, bool):
            raise AgreementConfigurationError(
                f"{analysis_id}: tolerance measured flag must be boolean"
            )
        if measured:
            absolute: object = tolerance.get("absolute")
            """Read field-specific absolute floors."""

            relative: object = tolerance.get("relative")
            """Read field-specific scale-dependent allowances."""

            expected_fields: set[str] = set(numeric_fields)
            """Required both tolerance maps to cover the selected outputs exactly."""

            for label, values in (("absolute", absolute), ("relative", relative)):
                if (
                    not isinstance(values, dict)
                    or set(values) != expected_fields
                    or not all(
                        isinstance(value, int | float)
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) >= 0.0
                        for value in values.values()
                    )
                ):
                    raise AgreementConfigurationError(
                        f"{analysis_id}: {label} tolerances are incomplete"
                    )
            """Required finite nonnegative values for every selected numeric path."""
        elif "absolute" in tolerance or "relative" in tolerance:
            raise AgreementConfigurationError(
                f"{analysis_id}: unmeasured tolerances must not contain thresholds"
            )
        tolerances[analysis_id] = tolerance
        """Retained one complete tolerance contract for probe validation."""
    """Validated every table independently before comparing inventory coverage."""

    if set(tolerances) != set(analysis_ids):
        raise AgreementConfigurationError(
            "model-specific tolerances do not cover every supported analysis exactly"
        )
    configured: bool = agreement.get("configured") is True
    """Read the deliberate opt-in that permits a passing agreement decision."""

    all_measured: bool = all(
        tolerance.get("measured") is True for tolerance in tolerances.values()
    )
    """Required every model-specific threshold to have measured provenance."""

    if configured != all_measured and (not allow_unmeasured or configured):
        missing: list[str] = sorted(
            analysis_id
            for analysis_id, tolerance in tolerances.items()
            if tolerance.get("measured") is not True
        )
        """Named exactly which supported analyses retain unmeasured tolerances."""

        raise AgreementConfigurationError(
            "cross-platform agreement is not configured; unmeasured tolerances: "
            + ", ".join(missing)
        )
    if not configured and not allow_unmeasured:
        raise AgreementConfigurationError("cross-platform agreement is not configured")
    return tolerances, configured


def validate_probe_matrix(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    tolerances: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Validate four complete probes and index them by execution target.

    Args:
        manifest: Parsed authoritative release manifest.
        records: Downloaded normalized probe artifacts.
        tolerances: Validated per-analysis numeric field contracts.

    Returns:
        Probe records indexed by platform, architecture and Python version.

    Raises:
        AgreementConfigurationError: If artifacts are missing, extra, stale or malformed.
    """
    expected_manifest_sha256: str = canonical_manifest_sha256(manifest)
    """Bound installed and checkout representations of the same parsed contract."""

    expected_version: str = str(manifest.get("version", ""))
    """Read the one public version every wheel must expose."""

    expected_release: bool = manifest.get("release") is True
    """Read whether probes are development evidence or a release decision."""

    by_target: dict[tuple[str, str, str], dict[str, Any]] = {}
    """Accumulated one and only one artifact for each promised target."""

    source_commits: set[str] = set()
    """Collected build commits for one exact-source consistency check."""

    expected_analysis_ids: set[str] = set(tolerances)
    """Required each artifact to cover precisely the supported inventory."""

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AgreementConfigurationError(
                f"probe artifact {index} is not an object"
            )
        if record.get("schema_version") != 1 or record.get("probe_id") != PROBE_ID:
            raise AgreementConfigurationError(
                f"probe artifact {index} has an unsupported protocol"
            )
        if record.get("manifest_sha256") != expected_manifest_sha256:
            raise AgreementConfigurationError(
                f"probe artifact {index} does not match the release manifest"
            )
        key: tuple[str, str, str] = target_key(record)
        """Resolved the workflow matrix coordinate for this downloaded result."""

        if key in by_target:
            raise AgreementConfigurationError(f"duplicate probe target: {key!r}")
        by_target[key] = record
        """Retained this unique artifact for later pairwise comparison."""

        build: object = record.get("build")
        """Read immutable wheel identity emitted through the public package API."""

        if not isinstance(build, dict):
            raise AgreementConfigurationError(f"probe artifact {index} has no build")
        if build.get("version") != expected_version:
            raise AgreementConfigurationError(
                f"probe artifact {index} version does not match the manifest"
            )
        if build.get("release") is not expected_release:
            raise AgreementConfigurationError(
                f"probe artifact {index} release state does not match the manifest"
            )
        source_commit: object = build.get("source_commit")
        """Read the exact source revision embedded when this platform wheel was built."""

        if not isinstance(source_commit, str) or HEX40.fullmatch(source_commit) is None:
            raise AgreementConfigurationError(
                f"probe artifact {index} has no valid source commit"
            )
        source_commits.add(source_commit)
        """Collected source identity for the all-artifact consistency check."""

        if build.get("source_dirty") is not False:
            raise AgreementConfigurationError(
                f"probe artifact {index} was built from dirty source"
            )
        wheel_sha256: object = build.get("wheel_sha256")
        """Read the saved wheel-byte identity attached to this probe."""

        if not isinstance(wheel_sha256, str) or HEX64.fullmatch(wheel_sha256) is None:
            raise AgreementConfigurationError(
                f"probe artifact {index} has no valid wheel SHA-256"
            )
        analyses: object = record.get("analyses")
        """Read normalized public-API results for all supported analyses."""

        if not isinstance(analyses, list) or not all(
            isinstance(analysis, dict) for analysis in analyses
        ):
            raise AgreementConfigurationError(
                f"probe artifact {index} analyses are malformed"
            )
        analysis_ids: list[str] = [
            str(analysis.get("analysis_id", "")) for analysis in analyses
        ]
        """Collected analysis identities while preserving duplicate detection."""

        if set(analysis_ids) != expected_analysis_ids or len(analysis_ids) != len(
            expected_analysis_ids
        ):
            raise AgreementConfigurationError(
                f"probe artifact {index} does not cover every analysis exactly"
            )
        for analysis in analyses:
            analysis_id: str = str(analysis["analysis_id"])
            """Selected the matching numeric field contract for this normalized result."""

            if analysis.get("outcome_status") not in {
                "fitted",
                "fitted-with-failures",
                "refused",
            }:
                raise AgreementConfigurationError(
                    f"{key!r}/{analysis_id}: outcome status is invalid"
                )
            refusal_code: object = analysis.get("refusal_code")
            """Read a stable refusal code only where no fit was produced."""

            if analysis.get("outcome_status") == "refused":
                if not isinstance(refusal_code, str) or not refusal_code:
                    raise AgreementConfigurationError(
                        f"{key!r}/{analysis_id}: refused outcome needs a code"
                    )
            elif refusal_code is not None:
                raise AgreementConfigurationError(
                    f"{key!r}/{analysis_id}: a fitted outcome cannot carry a refusal"
                )
            boundary_state: object = analysis.get("boundary_state")
            """Read every explicit boundary and limited-state field together."""

            if not isinstance(boundary_state, dict):
                raise AgreementConfigurationError(
                    f"{key!r}/{analysis_id}: boundary state must be an object"
                )
            field_presence: object = analysis.get("field_presence")
            """Read the complete normalized field-shape inventory."""

            if (
                not isinstance(field_presence, list)
                or not all(isinstance(field, str) and field for field in field_presence)
                or field_presence != sorted(set(field_presence))
            ):
                raise AgreementConfigurationError(
                    f"{key!r}/{analysis_id}: field presence must be sorted and unique"
                )
            numeric_fields: object = analysis.get("numeric_fields")
            """Read only numerical outputs carrying pre-written tolerances."""

            expected_numeric: set[str] = set(tolerances[analysis_id]["numeric_fields"])
            """Selected the exact allowed numeric inventory for this model."""

            wanted_numeric: set[str] = (
                set()
                if analysis.get("outcome_status") == "refused"
                else expected_numeric
            )
            """Withheld numerical outputs only where no fit exists."""

            if (
                not isinstance(numeric_fields, dict)
                or set(numeric_fields) != wanted_numeric
            ):
                raise AgreementConfigurationError(
                    f"{key!r}/{analysis_id}: numeric fields do not match the contract"
                )
            if not all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in numeric_fields.values()
            ):
                raise AgreementConfigurationError(
                    f"{key!r}/{analysis_id}: numeric fields must be finite numbers"
                )
        """Validated every analysis before admitting the target artifact."""
    """Validated all downloaded probe artifacts before checking matrix completeness."""

    if set(by_target) != set(EXPECTED_TARGETS):
        missing: list[tuple[str, str, str]] = sorted(EXPECTED_TARGETS - set(by_target))
        """Named absent targets in an actionable release diagnostic."""

        extra: list[tuple[str, str, str]] = sorted(set(by_target) - EXPECTED_TARGETS)
        """Named unexpected targets that might otherwise disguise matrix drift."""

        raise AgreementConfigurationError(
            f"probe artifact matrix is incomplete; missing={missing!r}, extra={extra!r}"
        )
    if len(source_commits) != 1:
        raise AgreementConfigurationError(
            "probe artifacts do not share one source commit"
        )
    return by_target


def analyses_by_id(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index one validated probe's analysis results.

    Args:
        record: Validated normalized probe artifact.

    Returns:
        Analysis result dictionaries keyed by manifest identifier.
    """
    return {str(analysis["analysis_id"]): analysis for analysis in record["analyses"]}


def compare_probe_records(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    allow_unmeasured: bool = False,
) -> dict[str, Any]:
    """Compare Mac and Linux probe results for both supported interpreters.

    Args:
        manifest: Parsed authoritative release manifest.
        records: Four normalized installed-wheel public-API probe artifacts.
        allow_unmeasured: Emit nonpassing development evidence when thresholds remain open.

    Returns:
        Machine-readable passing, failing or explicitly unmeasured agreement record.

    Raises:
        AgreementConfigurationError: If configuration or artifacts are incomplete.
    """
    tolerances, configured = tolerance_contract(
        manifest, allow_unmeasured=allow_unmeasured
    )
    """Validated comparison fields before trusting any downloaded result."""

    by_target: dict[tuple[str, str, str], dict[str, Any]] = validate_probe_matrix(
        manifest, records, tolerances
    )
    """Required the exact four-artifact matrix and one immutable source identity."""

    ordered_artifacts: list[dict[str, Any]] = []
    """Recorded target and wheel identities in deterministic order."""

    for key in sorted(by_target):
        record: dict[str, Any] = by_target[key]
        """Selected one validated probe in canonical target order."""

        ordered_artifacts.append(
            {
                "platform": key[0],
                "architecture": key[1],
                "python_version": key[2],
                "wheel_sha256": record["build"]["wheel_sha256"],
            }
        )
    """Retained all four artifact identities without copying model results."""

    source_commit: str = str(next(iter(by_target.values()))["build"]["source_commit"])
    """Read the already-validated common source identity."""

    base: dict[str, Any] = {
        "schema_version": 1,
        "probe_id": PROBE_ID,
        "configured": configured,
        "manifest_sha256": canonical_manifest_sha256(manifest),
        "version": str(manifest["version"]),
        "release": manifest.get("release") is True,
        "source_commit": source_commit,
        "source_dirty": False,
        "artifacts": ordered_artifacts,
    }
    """Bound every decision to its exact contract, source and saved wheel matrix."""

    comparisons: list[dict[str, Any]] = []
    """Accumulated one Mac-versus-Linux comparison for each interpreter."""

    failures: list[str] = []
    """Collected every structural and numerical disagreement for one repair cycle."""

    for python_version in ("3.13", "3.14"):
        mac: dict[str, Any] = by_target[("macos", "arm64", python_version)]
        """Selected the promised development-Mac reference result."""

        linux: dict[str, Any] = by_target[("linux", "x86_64", python_version)]
        """Selected the portable Linux-wheel candidate result."""

        mac_analyses: dict[str, dict[str, Any]] = analyses_by_id(mac)
        """Indexed Mac results by stable manifest analysis identifier."""

        linux_analyses: dict[str, dict[str, Any]] = analyses_by_id(linux)
        """Indexed Linux results through the same stable identifiers."""

        exact_results: list[dict[str, Any]] = []
        """Recorded every exact field comparison rather than only aggregate pass."""

        numeric_results: list[dict[str, Any]] = []
        """Recorded values, tolerances and decisions for every numerical field."""

        pair_failures: list[str] = []
        """Collected disagreements belonging to this interpreter pair."""

        for analysis_id in tolerances:
            left: dict[str, Any] = mac_analyses[analysis_id]
            """Selected the Mac normalized result as the reference."""

            right: dict[str, Any] = linux_analyses[analysis_id]
            """Selected the corresponding Linux normalized result."""

            for field in sorted(EXACT_COMPARISONS):
                equal: bool = left[field] == right[field]
                """Applied exact equality to status, refusal, boundary or shape."""

                exact_results.append(
                    {
                        "analysis_id": analysis_id,
                        "field": field,
                        "reference": left[field],
                        "candidate": right[field],
                        "passed": equal,
                    }
                )
                if not equal:
                    pair_failures.append(
                        f"Python {python_version} {analysis_id} {field} mismatch"
                    )
            """Compared every non-numeric contract field without tolerance."""

            tolerance: dict[str, Any] = tolerances[analysis_id]
            """Selected pre-written numeric thresholds for this model."""

            if (
                left["outcome_status"] == "refused"
                or right["outcome_status"] == "refused"
            ):
                continue
            for field in tolerance["numeric_fields"]:
                reference: float = float(left["numeric_fields"][field])
                """Read the development-Mac value used as the relative scale."""

                candidate: float = float(right["numeric_fields"][field])
                """Read the portable Linux value being checked."""

                difference: float = abs(candidate - reference)
                """Measured the observed cross-platform absolute difference."""

                relative_difference: float | None = (
                    difference / abs(reference)
                    if reference != 0.0
                    else (0.0 if difference == 0.0 else None)
                )
                """Reported scale-relative drift, leaving it undefined at a zero reference."""

                absolute_tolerance: float | None
                """Declared the measured absolute floor or its unmeasured absence."""
                relative_tolerance: float | None
                """Declared the measured relative allowance or its unmeasured absence."""
                allowed_difference: float | None
                """Declared the combined allowance or its unmeasured absence."""
                passed: bool | None
                """Declared a decision only after thresholds have measured provenance."""

                if tolerance.get("measured") is True:
                    absolute_tolerance = float(tolerance["absolute"][field])
                    """Read the measured field-specific numerical floor."""
                    relative_tolerance = float(tolerance["relative"][field])
                    """Read the measured field-specific scale-dependent allowance."""
                    allowed_difference = absolute_tolerance + (
                        relative_tolerance * abs(reference)
                    )
                    """Combined the pre-written absolute and relative tolerances."""
                    passed = difference <= allowed_difference or math.isclose(
                        difference,
                        allowed_difference,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    """Included a decimal tolerance edge despite binary64 representation."""
                else:
                    absolute_tolerance = None
                    """Withheld an acceptance floor while this field remains unmeasured."""
                    relative_tolerance = None
                    """Withheld a scale allowance while this field remains unmeasured."""
                    allowed_difference = None
                    """Prevented an observed delta from becoming its own pass threshold."""
                    passed = None
                    """Kept development measurement distinct from agreement."""

                numeric_results.append(
                    {
                        "analysis_id": analysis_id,
                        "field": field,
                        "reference": reference,
                        "candidate": candidate,
                        "difference": difference,
                        "relative_difference": relative_difference,
                        "absolute_tolerance": absolute_tolerance,
                        "relative_tolerance": relative_tolerance,
                        "allowed_difference": allowed_difference,
                        "passed": passed,
                    }
                )
                if passed is False:
                    pair_failures.append(
                        f"Python {python_version} {analysis_id} {field} numeric "
                        f"mismatch: {difference!r} > {allowed_difference!r}"
                    )
            """Compared all and only the numeric paths fixed before this release run."""
        """Compared every supported analysis for one Python-version pair."""

        comparisons.append(
            {
                "python_version": python_version,
                "reference_target": {
                    "platform": "macos",
                    "architecture": "arm64",
                },
                "candidate_target": {
                    "platform": "linux",
                    "architecture": "x86_64",
                },
                "exact": exact_results,
                "numeric": numeric_results,
                "passed": not pair_failures if configured else False,
                "failures": pair_failures,
            }
        )
        failures.extend(pair_failures)
        """Retained both the per-interpreter and release-wide failure inventories."""
    """Completed Mac/Linux comparisons independently on both supported interpreters."""

    if not configured:
        missing: list[str] = sorted(
            analysis_id
            for analysis_id, tolerance in tolerances.items()
            if tolerance.get("measured") is not True
        )
        """Named the models whose observed deltas still lack acceptance thresholds."""

        gate_failure: str = (
            "cross-platform tolerances are unmeasured for: " + ", ".join(missing)
            if missing
            else "cross-platform agreement is measured but not configured"
        )
        """Kept development evidence explicitly outside the release decision."""

        return {
            **base,
            "passed": False,
            "status": "failed" if failures else "unmeasured",
            "comparisons": comparisons,
            "failures": [*failures, gate_failure],
        }
    return {
        **base,
        "passed": not failures,
        "status": "passed" if not failures else "failed",
        "comparisons": comparisons,
        "failures": failures,
    }


def read_probe_records(input_directory: Path) -> list[dict[str, Any]]:
    """Read normalized probe JSON files from downloaded workflow artifacts.

    Args:
        input_directory: Directory containing only probe artifacts or subdirectories.

    Returns:
        Parsed probe records in deterministic path order.

    Raises:
        AgreementConfigurationError: If no JSON probe artifacts exist.
    """
    paths: list[Path] = sorted(input_directory.rglob("*.json"))
    """Discovered downloaded records regardless of artifact-directory layout."""

    if not paths:
        raise AgreementConfigurationError(
            f"no cross-platform probe artifacts found in {input_directory}"
        )
    records: list[dict[str, Any]] = []
    """Accumulated parsed JSON objects without accepting non-object roots."""

    for path in paths:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
        """Parsed one exact workflow artifact."""

        if not isinstance(parsed, dict):
            raise AgreementConfigurationError(
                f"probe artifact is not an object: {path}"
            )
        records.append(parsed)
    """Read every discovered artifact before matrix validation identifies extras."""
    return records


def main() -> int:
    """Compare downloaded probe artifacts and write one agreement record."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compare Asterism installed-wheel results across Mac and Linux."
    )
    """Defined the workflow and release command-line interface."""

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unmeasured", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()
    """Parsed contract, artifact directory and output decision path."""

    try:
        manifest: dict[str, Any] = tomllib.loads(
            arguments.manifest.read_text(encoding="utf-8")
        )
        """Parsed the authoritative checkout contract."""

        records: list[dict[str, Any]] = read_probe_records(arguments.input)
        """Read all downloaded platform/interpreter artifacts."""

        result: dict[str, Any] = compare_probe_records(
            manifest,
            records,
            allow_unmeasured=arguments.allow_unmeasured,
        )
        """Produced a passing, failing or explicitly unmeasured decision record."""
    except (AgreementConfigurationError, OSError, json.JSONDecodeError) as error:
        print(f"cross-platform agreement refused: {error}", file=sys.stderr)
        return 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    """Created the selected evidence directory only after successful validation."""

    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    """Persisted the full comparison details as durable machine-readable evidence."""

    if result["status"] == "failed":
        print("cross-platform agreement failed", file=sys.stderr)
        return 1
    if result["status"] == "unmeasured":
        print("cross-platform agreement remains explicitly unmeasured")
        return 0
    print("Cross-platform agreement passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
