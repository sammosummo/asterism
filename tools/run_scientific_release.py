"""Execute configured scientific pass rules against one fixed Asterism wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

if __package__:
    from .measure_target_resources import (
        target_resource_configuration_errors,
        target_resource_record_errors,
    )
else:
    from measure_target_resources import (
        target_resource_configuration_errors,
        target_resource_record_errors,
    )
"""Imported shared resource contracts in module and direct-script execution."""

IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"[a-z][a-z0-9_]*")
"""Restricted evidence identifiers to stable, path-safe names."""


class ReleaseConfigurationError(ValueError):
    """Report a release configuration that cannot be executed safely."""


def fixed_input_errors(
    *,
    core: ModuleType | Any,
    manifest_text: str,
    cargo_lock_text: str,
    uv_lock_text: str,
) -> list[str]:
    """Return disagreements between embedded and checkout release inputs.

    Args:
        core: Installed extension exposing exact embedded source inputs.
        manifest_text: Checkout release-manifest bytes.
        cargo_lock_text: Checkout Rust dependency-lock bytes.
        uv_lock_text: Checkout Python dependency-lock bytes.

    Returns:
        One actionable error for each stale embedded input.
    """
    errors: list[str] = []
    """Collected exact-byte disagreements before any scientific command runs."""

    for attribute, selected, label in (
        ("__release_manifest__", manifest_text, "release.toml"),
        ("__cargo_lock__", cargo_lock_text, "Cargo.lock"),
        ("__uv_lock__", uv_lock_text, "uv.lock"),
    ):
        if getattr(core, attribute, None) != selected:
            errors.append(f"installed wheel does not embed the selected {label}")
    """Compared actual bytes rather than trusting independently supplied hashes."""
    return errors


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Return the comparator's format-independent parsed-manifest digest.

    Args:
        manifest: Parsed authoritative release manifest.

    Returns:
        Lowercase SHA-256 of canonical JSON bytes.
    """
    payload: bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    """Rendered the same parsed contract representation used by platform probes."""
    return hashlib.sha256(payload).hexdigest()


def agreement_record_errors(
    *,
    record: object,
    manifest: dict[str, Any],
    source_commit: str,
    wheel_sha256s: set[str],
) -> list[str]:
    """Return structural or identity failures in one actual comparison record.

    Args:
        record: Parsed downloaded cross-platform comparison JSON.
        manifest: Parsed authoritative release manifest.
        source_commit: Commit embedded in the installed release wheel.
        wheel_sha256s: Exact saved-wheel digest set selected for release.

    Returns:
        Complete actionable comparison-record failures.
    """
    if not isinstance(record, dict):
        return ["cross-platform agreement record must be an object"]
    errors: list[str] = []
    """Collected identity, matrix and per-field decision failures together."""

    expected_identity: dict[str, object] = {
        "schema_version": 1,
        "probe_id": "asterism-cross-platform-0.1",
        "configured": True,
        "manifest_sha256": canonical_manifest_sha256(manifest),
        "version": str(manifest.get("version", "")),
        "release": True,
        "source_commit": source_commit,
        "source_dirty": False,
        "passed": True,
        "status": "passed",
    }
    """Named every release and final-decision value the actual record must carry."""

    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            errors.append(f"cross-platform agreement {field} does not match release")
    """Bound the result to this release rather than a prior passing comparison."""

    if record.get("failures") != []:
        errors.append("cross-platform agreement records failures")
    artifacts: object = record.get("artifacts")
    """Read the saved-wheel target matrix actually compared."""

    expected_targets: set[tuple[str, str, str]] = {
        ("macos", "arm64", "3.13"),
        ("linux", "x86_64", "3.13"),
        ("macos", "arm64", "3.14"),
        ("linux", "x86_64", "3.14"),
    }
    """Required all four supported installed-wheel environments exactly once."""

    actual_targets: list[tuple[str, str, str]] = []
    """Collected target coordinates with multiplicity for duplicate detection."""

    artifact_hashes: set[str] = set()
    """Collected saved-wheel hashes represented by all four runtime probes."""

    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            actual_targets.append(
                (
                    str(artifact.get("platform", "")),
                    str(artifact.get("architecture", "")),
                    str(artifact.get("python_version", "")),
                )
            )
            artifact_hashes.add(str(artifact.get("wheel_sha256", "")))
    """Read only well-formed target records before exact-set comparison."""

    if set(actual_targets) != expected_targets or len(actual_targets) != 4:
        errors.append("cross-platform agreement artifact matrix is incomplete")
    if artifact_hashes != wheel_sha256s:
        errors.append("cross-platform agreement wheel SHA-256 set does not match")
    configuration: object = manifest.get("cross_platform_agreement")
    """Read pre-written exact and numerical comparison inventories."""

    analyses: object = manifest.get("analyses")
    """Read all supported analysis identifiers required in both comparisons."""

    analysis_ids: set[str] = (
        {
            str(analysis.get("id", ""))
            for analysis in analyses
            if isinstance(analysis, dict)
        }
        if isinstance(analyses, list)
        else set()
    )
    """Collected the exact supported-analysis inventory."""

    tolerance_fields: dict[str, set[str]] = {}
    """Indexed selected numeric fields by supported analysis."""

    tolerance_values: dict[tuple[str, str], tuple[float, float]] = {}
    """Indexed exact absolute and relative thresholds for independent arithmetic."""

    exact_fields: set[str] = set()
    """Collected exact structural comparison names from the release contract."""

    if isinstance(configuration, dict):
        exact_fields = {
            str(field) for field in configuration.get("exact_comparisons", [])
        }
        """Read exact status, refusal, boundary and field-presence comparisons."""

        raw_tolerances: object = configuration.get("model_tolerances")
        """Read measured model-specific numerical field inventories."""

        if isinstance(raw_tolerances, list):
            for tolerance in raw_tolerances:
                if isinstance(tolerance, dict) and isinstance(
                    tolerance.get("analysis_id"), str
                ):
                    numeric_fields: object = tolerance.get("numeric_fields")
                    """Read selected normalized result paths for one analysis."""

                    if isinstance(numeric_fields, list):
                        analysis_id: str = str(tolerance["analysis_id"])
                        """Selected the stable analysis identifier for both indices."""

                        tolerance_fields[analysis_id] = {
                            str(field) for field in numeric_fields
                        }
                        """Retained the exact numerical decision inventory."""

                        absolute: object = tolerance.get("absolute")
                        """Read field-specific absolute tolerances from the manifest."""

                        relative: object = tolerance.get("relative")
                        """Read field-specific relative tolerances from the manifest."""

                        if isinstance(absolute, dict) and isinstance(relative, dict):
                            for field in numeric_fields:
                                if isinstance(
                                    absolute.get(field), int | float
                                ) and isinstance(relative.get(field), int | float):
                                    tolerance_values[(analysis_id, str(field))] = (
                                        float(absolute[field]),
                                        float(relative[field]),
                                    )
                                    """Indexed one field's exact absolute and relative pair."""
                            """Retained exact pre-written thresholds for every valid field."""
    """Parsed only the already configured release contract used by the comparator."""

    comparisons: object = record.get("comparisons")
    """Read both Python-version Mac/Linux comparison details."""

    if not isinstance(comparisons, list):
        errors.append("cross-platform agreement has no comparison details")
        return errors
    comparison_versions: list[str] = []
    """Collected interpreter identities with multiplicity for exact coverage."""

    expected_exact: set[tuple[str, str]] = {
        (analysis_id, field) for analysis_id in analysis_ids for field in exact_fields
    }
    """Expanded the structural decision inventory for one interpreter pair."""

    expected_numeric: set[tuple[str, str]] = {
        (analysis_id, field)
        for analysis_id, fields in tolerance_fields.items()
        for field in fields
    }
    """Expanded all pre-written numerical decisions for one interpreter pair."""

    for comparison in comparisons:
        if not isinstance(comparison, dict):
            errors.append("cross-platform agreement comparison is not an object")
            continue
        comparison_versions.append(str(comparison.get("python_version", "")))
        """Retained the interpreter coordinate for completeness validation."""

        if comparison.get("passed") is not True or comparison.get("failures") != []:
            errors.append("cross-platform agreement contains a failed comparison")
        exact: object = comparison.get("exact")
        """Read every exact structural decision made by the comparator."""

        actual_exact: list[tuple[str, str]] = []
        """Collected structural decisions with multiplicity."""

        if isinstance(exact, list):
            for decision in exact:
                if (
                    isinstance(decision, dict)
                    and decision.get("passed") is True
                    and decision.get("reference") == decision.get("candidate")
                ):
                    actual_exact.append(
                        (
                            str(decision.get("analysis_id", "")),
                            str(decision.get("field", "")),
                        )
                    )
                elif isinstance(decision, dict):
                    errors.append("cross-platform exact decision is not equal")
        """Accepted only explicit passing per-field structural decisions."""

        if set(actual_exact) != expected_exact or len(actual_exact) != len(
            expected_exact
        ):
            errors.append("cross-platform agreement exact decisions are incomplete")
        numeric: object = comparison.get("numeric")
        """Read every tolerance-backed numerical decision made by the comparator."""

        actual_numeric: list[tuple[str, str]] = []
        """Collected numerical decisions with multiplicity."""

        if isinstance(numeric, list):
            for decision in numeric:
                if not isinstance(decision, dict):
                    continue
                key: tuple[str, str] = (
                    str(decision.get("analysis_id", "")),
                    str(decision.get("field", "")),
                )
                """Selected the manifest tolerance owning this numerical decision."""

                values: tuple[object, ...] = (
                    decision.get("reference"),
                    decision.get("candidate"),
                    decision.get("difference"),
                    decision.get("absolute_tolerance"),
                    decision.get("relative_tolerance"),
                    decision.get("allowed_difference"),
                )
                """Collected every value needed to recompute the recorded decision."""

                if not all(
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values
                ):
                    errors.append("cross-platform numeric decision is non-finite")
                    continue
                reference, candidate, difference, absolute, relative, allowed = (
                    float(value) for value in values
                )
                """Narrowed validated JSON numbers for independent arithmetic."""

                expected_tolerance: tuple[float, float] | None = tolerance_values.get(
                    key
                )
                """Read the immutable pre-written tolerance pair for this result path."""

                expected_difference: float = abs(candidate - reference)
                """Recomputed the observed cross-platform difference."""

                expected_allowed: float = absolute + relative * abs(reference)
                """Recomputed the comparator's absolute-plus-relative allowance."""

                valid: bool = (
                    expected_tolerance == (absolute, relative)
                    and math.isclose(
                        difference, expected_difference, rel_tol=1e-12, abs_tol=1e-15
                    )
                    and math.isclose(
                        allowed, expected_allowed, rel_tol=1e-12, abs_tol=1e-15
                    )
                    and (
                        difference <= allowed
                        or math.isclose(
                            difference, allowed, rel_tol=1e-12, abs_tol=1e-15
                        )
                    )
                    and decision.get("passed") is True
                )
                """Verified tolerance provenance, arithmetic and the final pass together."""

                if valid:
                    actual_numeric.append(key)
                else:
                    errors.append("cross-platform numeric decision is invalid")
        """Accepted only explicit passing per-field numerical decisions."""

        if set(actual_numeric) != expected_numeric or len(actual_numeric) != len(
            expected_numeric
        ):
            errors.append("cross-platform agreement numeric decisions are incomplete")
    """Verified details rather than trusting the record's aggregate passing boolean."""

    if set(comparison_versions) != {"3.13", "3.14"} or len(comparison_versions) != 2:
        errors.append("cross-platform agreement Python comparisons are incomplete")
    return errors


def file_identity(path: Path) -> dict[str, str]:
    """Return the path and SHA-256 identity of one immutable file.

    Args:
        path: Existing regular file to identify.

    Returns:
        A JSON-serialisable path and hexadecimal digest pair.

    Raises:
        ReleaseConfigurationError: If the selected path is not a regular file.
    """
    if not path.is_file():
        raise ReleaseConfigurationError(f"required release file does not exist: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def scientific_inventory_errors(manifest: dict[str, Any], root: Path) -> list[str]:
    """Return structural failures in every required scientific pass rule.

    This validator is intentionally usable while the development release stays
    unconfigured.  It proves exact one-to-one command coverage and explicit
    blocker provenance without promoting any blocked or unrun rule to passing.

    Args:
        manifest: Parsed authoritative release manifest.
        root: Repository root containing every configured command script.

    Returns:
        All machine-readable inventory failures.
    """
    errors: list[str] = []
    """Collected all analyses so one test run exposes the complete repair list."""

    analyses: object = manifest.get("analyses")
    """Read the authoritative supported-analysis inventory."""

    if not isinstance(analyses, list) or not analyses:
        return ["at least one analysis is required"]
    for raw_analysis in analyses:
        if not isinstance(raw_analysis, dict):
            errors.append("each analysis must be a table")
            continue
        identifier: object = raw_analysis.get("id")
        """Read the analysis identifier used by blocker and evidence records."""

        if not isinstance(identifier, str):
            errors.append("each analysis needs a string id")
            continue
        required: object = raw_analysis.get("required_checks")
        """Read every scientific claim promised by this supported analysis."""

        rules: object = raw_analysis.get("pass_rules")
        """Read commands and blockers intended to discharge those claims."""

        if not isinstance(required, list) or not all(
            isinstance(check, str) for check in required
        ):
            errors.append(f"{identifier}: required_checks must be a string list")
            continue
        if not isinstance(rules, list) or not all(
            isinstance(rule, dict) for rule in rules
        ):
            errors.append(f"{identifier}: pass_rules must be a table list")
            continue
        rule_ids: list[object] = [rule.get("id") for rule in rules]
        """Retained multiplicity so duplicate commands cannot masquerade as coverage."""

        string_rule_ids: list[str] = [
            rule_id for rule_id in rule_ids if isinstance(rule_id, str)
        ]
        """Narrowed valid identities before deterministic set comparison."""

        if len(rule_ids) != len(required) or sorted(string_rule_ids) != sorted(
            required
        ):
            errors.append(f"{identifier}: pass rule ids must equal required_checks")
        if len(string_rule_ids) != len(set(string_rule_ids)):
            errors.append(f"{identifier}: pass rule ids must be unique")
        for rule in rules:
            rule_id: object = rule.get("id")
            """Selected the exact required-check identity for actionable messages."""

            label: str = f"{identifier}/{rule_id}"
            """Built the stable analysis and evidence coordinate."""

            command: object = rule.get("command")
            """Read the shell-free argument vector executed by the release runner."""

            if (
                not isinstance(command, list)
                or not command
                or not all(
                    isinstance(argument, str) and argument for argument in command
                )
            ):
                errors.append(f"{label}: command must be a nonempty string list")
            else:
                script: Path = root / command[0]
                """Resolved only the configured Python entry script."""

                if (
                    Path(command[0]).is_absolute()
                    or ".." in Path(command[0]).parts
                    or script.suffix != ".py"
                    or not script.is_file()
                ):
                    errors.append(
                        f"{label}: command must name an existing local Python script"
                    )
            if rule.get("expected_exit_code") != 0 or isinstance(
                rule.get("expected_exit_code"), bool
            ):
                errors.append(f"{label}: expected_exit_code must be 0")
            timeout: object = rule.get("timeout_seconds")
            """Read the prewritten upper wall-time bound, not a measured duration."""

            if (
                not isinstance(timeout, int)
                or isinstance(timeout, bool)
                or not 1 <= timeout <= 604_800
            ):
                errors.append(f"{label}: timeout_seconds must be between 1 and 604800")
            status: object = rule.get("status")
            """Distinguished runnable rules from explicit evidence deficits."""

            if status not in {"ready", "blocked", "external_fixture_pending"}:
                errors.append(f"{label}: status is not recognized")
            design_facts: object = rule.get("design_facts")
            """Read non-identifying facts the command must measure or report."""

            if (
                not isinstance(design_facts, dict)
                or design_facts.get("participant_free") is not True
            ):
                errors.append(
                    f"{label}: design_facts must declare participant_free=true"
                )
            elif status == "ready":
                for field in (
                    "sample_size",
                    "largest_family",
                    "trait_type",
                    "components",
                ):
                    if field not in design_facts:
                        errors.append(f"{label}: ready rule has no {field} design fact")
            blocker: object = rule.get("blocker")
            """Read the exact evidence still needed by a non-ready rule."""

            if status == "ready" and blocker is not None:
                errors.append(f"{label}: ready rule must not retain a blocker")
            if status != "ready" and (
                not isinstance(blocker, dict)
                or not isinstance(blocker.get("code"), str)
                or not blocker["code"]
                or not isinstance(blocker.get("evidence_needed"), str)
                or not blocker["evidence_needed"]
            ):
                errors.append(
                    f"{label}: non-ready rule needs a code and evidence_needed"
                )
    return errors


def configured_rules(
    manifest: dict[str, Any], root: Path
) -> list[tuple[str, dict[str, Any]]]:
    """Validate and flatten every configured scientific pass rule.

    Args:
        manifest: Parsed authoritative release manifest.
        root: Repository root containing configured check scripts.

    Returns:
        Analysis identifiers paired with executable rule dictionaries.

    Raises:
        ReleaseConfigurationError: If a claim lacks one exact executable rule.
    """
    errors: list[str] = []
    """Collected all configuration failures for one repair cycle."""

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if manifest.get("release") is not True:
        errors.append("release must be true")
    if manifest.get("scientific_pass_rules_configured") is not True:
        errors.append("scientific_pass_rules_configured must be true")
    """Required deliberate release-wide opt-ins before inspecting commands."""

    errors.extend(scientific_inventory_errors(manifest, root))
    """Required exact commands and blocker metadata even before readiness flags."""

    flattened: list[tuple[str, dict[str, Any]]] = []
    """Accumulated validated rules in deterministic manifest order."""

    analyses: object = manifest.get("analyses")
    """Read the supported scientific analysis inventory."""

    if not isinstance(analyses, list) or not analyses:
        errors.append("at least one analysis is required")
        analyses = []
        """Substituted an empty iterable after recording the invalid inventory."""
    """Refused a release with no scientific claims to validate."""

    for raw_analysis in analyses:
        if not isinstance(raw_analysis, dict):
            errors.append("each analysis must be a table")
            continue
        analysis: dict[str, Any] = raw_analysis
        """Narrowed one parsed TOML analysis to its table representation."""

        identifier: object = analysis.get("id")
        """Read the stable identifier carried into every command record."""

        if (
            not isinstance(identifier, str)
            or IDENTIFIER_PATTERN.fullmatch(identifier) is None
        ):
            errors.append(f"invalid analysis id: {identifier!r}")
            continue
        if analysis.get("pass_rules_configured") is not True:
            errors.append(f"{identifier}: pass_rules_configured must be true")
        design_range: object = analysis.get("design_range")
        """Read the measured design-range claim that bounds this evidence."""

        if (
            not isinstance(design_range, dict)
            or design_range.get("measured") is not True
        ):
            errors.append(f"{identifier}: design range must be measured")
        required_checks: object = analysis.get("required_checks")
        """Read the evidence identifiers promised by this analysis."""

        pass_rules: object = analysis.get("pass_rules")
        """Read the executable rules intended to discharge those promises."""

        if not isinstance(required_checks, list) or not all(
            isinstance(check, str) for check in required_checks
        ):
            errors.append(f"{identifier}: required_checks must be a string list")
            required_checks = []
            """Prevented malformed check metadata from entering rule matching."""
        if not isinstance(pass_rules, list) or not all(
            isinstance(rule, dict) for rule in pass_rules
        ):
            errors.append(f"{identifier}: pass_rules must be a table list")
            pass_rules = []
            """Prevented malformed pass rules from entering command validation."""
        """Required machine-readable lists before matching names to commands."""

        rule_ids: list[object] = [rule.get("id") for rule in pass_rules]
        """Collected configured rule identifiers including malformed values."""

        for check_id in required_checks:
            if rule_ids.count(check_id) != 1:
                errors.append(
                    f"{identifier}: required check {check_id!r} needs exactly one pass rule"
                )
        if sorted(
            rule_id for rule_id in rule_ids if isinstance(rule_id, str)
        ) != sorted(required_checks):
            errors.append(f"{identifier}: pass rule ids must equal required_checks")
        """Prevented missing, duplicate and unrelated commands from satisfying a claim."""

        for raw_rule in pass_rules:
            rule: dict[str, Any] = raw_rule
            """Narrowed one parsed pass-rule table for field validation."""

            rule_id: object = rule.get("id")
            """Read the check identifier carried into evidence and log paths."""

            if (
                not isinstance(rule_id, str)
                or IDENTIFIER_PATTERN.fullmatch(rule_id) is None
            ):
                errors.append(f"{identifier}: invalid pass rule id: {rule_id!r}")
                continue
            command: object = rule.get("command")
            """Read the argument vector without passing through a shell."""

            if (
                not isinstance(command, list)
                or not command
                or not all(
                    isinstance(argument, str) and argument for argument in command
                )
            ):
                errors.append(f"{identifier}/{rule_id}: command must be a string list")
                continue
            script: Path = root / command[0]
            """Resolved the configured check program inside the release checkout."""

            try:
                script.relative_to(root)
            except ValueError:
                errors.append(f"{identifier}/{rule_id}: command escapes the checkout")
            if Path(command[0]).is_absolute() or ".." in Path(command[0]).parts:
                errors.append(
                    f"{identifier}/{rule_id}: command must be repository-relative"
                )
            if script.suffix != ".py" or not script.is_file():
                errors.append(
                    f"{identifier}/{rule_id}: Python check does not exist: {command[0]}"
                )
            expected_exit_code: object = rule.get("expected_exit_code")
            """Read the only successful scientific-command exit status."""

            if expected_exit_code != 0 or isinstance(expected_exit_code, bool):
                errors.append(f"{identifier}/{rule_id}: expected_exit_code must be 0")
            timeout_seconds: object = rule.get("timeout_seconds")
            """Read the bounded wall-time allowed for this configured command."""

            if (
                not isinstance(timeout_seconds, int)
                or isinstance(timeout_seconds, bool)
                or not 1 <= timeout_seconds <= 604_800
            ):
                errors.append(
                    f"{identifier}/{rule_id}: timeout_seconds must be between 1 and 604800"
                )
            if rule.get("status") != "ready":
                blocker: object = rule.get("blocker")
                """Read the explicit evidence deficit blocking this command."""

                code: object = (
                    blocker.get("code") if isinstance(blocker, dict) else "unknown"
                )
                """Selected a stable blocker code for release diagnostics."""

                errors.append(
                    f"{identifier}/{rule_id}: scientific rule is blocked: {code}"
                )
            flattened.append((identifier, rule))
        """Validated each pass rule before allowing any command to start."""

    analysis_ids: list[str] = [
        str(analysis.get("id", ""))
        for analysis in analyses
        if isinstance(analysis, dict)
    ]
    """Retained the ordered supported inventory for end-to-end receipt coverage."""

    synthetic: object = manifest.get("synthetic_analysis_receipts")
    """Read the separately configured standard-receipt release command."""

    if not isinstance(synthetic, dict):
        errors.append("synthetic_analysis_receipts must be configured")
    else:
        if synthetic.get("configured") is not True:
            errors.append("synthetic_analysis_receipts configured must be true")
        runner: object = synthetic.get("runner")
        """Read the repository-relative public command path."""

        if not isinstance(runner, str) or not runner:
            errors.append("synthetic_analysis_receipts runner must be configured")
        else:
            runner_path: Path = root / runner
            """Resolved the configured command inside the release checkout."""

            if (
                Path(runner).is_absolute()
                or ".." in Path(runner).parts
                or runner_path.suffix != ".py"
                or not runner_path.is_file()
            ):
                errors.append("synthetic_analysis_receipts runner is invalid")
        timeout_seconds: object = synthetic.get("timeout_seconds")
        """Read the prewritten maximum wall time for all nine standard receipts."""

        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= 604_800
        ):
            errors.append("synthetic_analysis_receipts timeout_seconds is invalid")
        configured_ids: object = synthetic.get("analysis_ids")
        """Read the exact analysis inventory promised by the receipt command."""

        if configured_ids != analysis_ids:
            errors.append(
                "synthetic_analysis_receipts analysis_ids must equal supported analyses"
            )
    """Required a real installed-wheel receipt command before release execution."""

    errors.extend(
        target_resource_configuration_errors(
            manifest.get("target_resource_budgets"),
            root,
            analysis_ids,
            require_configured=True,
        )
    )
    """Required complete target profiles and prewritten budgets before execution."""

    if errors:
        raise ReleaseConfigurationError("; ".join(errors))
    return flattened


def run_scientific_release(
    *,
    manifest_path: Path,
    output_directory: Path,
    wheel_paths: list[Path],
    cross_platform_agreement_path: Path,
    core: ModuleType | Any,
) -> dict[str, Any]:
    """Execute all scientific rules and write fixed-build release evidence.

    Args:
        manifest_path: Authoritative manifest embedded in the tested wheel.
        output_directory: New directory for evidence and exact command logs.
        wheel_paths: Built wheels whose bytes the evidence must identify.
        cross_platform_agreement_path: Actual four-target result comparison record.
        core: Installed extension module exposing immutable build metadata.

    Returns:
        The complete evidence document also written as ``evidence.json``.

    Raises:
        ReleaseConfigurationError: If release inputs or pass rules are incomplete.
    """
    root: Path = manifest_path.resolve().parent
    """Located the fixed checkout holding the manifest, locks and commands."""

    manifest_text: str = manifest_path.read_text(encoding="utf-8")
    """Read the exact authoritative bytes expected inside the installed wheel."""

    manifest: dict[str, Any] = tomllib.loads(manifest_text)
    """Parsed the scientific contract only after preserving its exact identity."""

    rules: list[tuple[str, dict[str, Any]]] = configured_rules(manifest, root)
    """Validated every rule before reading fixed-build inputs or running commands."""

    cargo_lock_text: str = (root / "Cargo.lock").read_text(encoding="utf-8")
    """Read exact Rust dependency bytes expected inside the release wheel."""

    uv_lock_text: str = (root / "uv.lock").read_text(encoding="utf-8")
    """Read exact Python dependency bytes expected inside the release wheel."""

    errors: list[str] = []
    """Collected fixed-build and artifact failures before creating evidence."""

    errors.extend(
        fixed_input_errors(
            core=core,
            manifest_text=manifest_text,
            cargo_lock_text=cargo_lock_text,
            uv_lock_text=uv_lock_text,
        )
    )
    """Required exact source and dependency contracts before scientific execution."""

    if str(getattr(core, "__version__", "")) != str(manifest.get("version", "")):
        errors.append("installed wheel version does not match release.toml")
    if getattr(core, "__source_dirty__", None) is not False:
        errors.append("installed wheel was not built from a clean source tree")
    source_commit: object = getattr(core, "__source_commit__", None)
    """Read the immutable source commit recorded by the wheel build."""

    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        errors.append("installed wheel has no valid source commit")
    core_path: Path = Path(str(getattr(core, "__file__", ""))).resolve()
    """Located the extension binary imported by this exact interpreter."""

    if not core_path.is_file():
        errors.append(f"installed extension does not exist: {core_path}")
    try:
        core_path.relative_to(root)
    except ValueError:
        pass
    else:
        errors.append(
            "scientific runner imported the mutable checkout, not a saved wheel"
        )
    """Required the extension to live outside the repository source tree."""

    resolved_wheels: list[Path] = [path.resolve() for path in wheel_paths]
    """Normalised the complete artifact set before checking for duplicates."""

    if not resolved_wheels:
        errors.append("at least one built wheel is required")
    if len(resolved_wheels) != len(set(resolved_wheels)):
        errors.append("wheel paths must be unique")
    for wheel_path in resolved_wheels:
        if wheel_path.suffix != ".whl" or not wheel_path.is_file():
            errors.append(f"wheel does not exist: {wheel_path}")
    """Refused missing, repeated and non-wheel artifacts."""

    wheel_sha256s: set[str] = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in resolved_wheels
        if path.is_file()
    }
    """Collected exact saved-wheel identities for comparison-record validation."""

    agreement_bytes: bytes = b""
    """Reserved exact comparison bytes only after its path is validated."""

    agreement_record: object = None
    """Reserved the parsed comparison object copied into durable evidence."""

    if not cross_platform_agreement_path.is_file():
        errors.append(
            f"cross-platform agreement does not exist: {cross_platform_agreement_path}"
        )
    else:
        agreement_bytes = cross_platform_agreement_path.read_bytes()
        """Read the downloaded comparison record without normalizing its bytes."""

        try:
            agreement_record = json.loads(agreement_bytes)
            """Parsed details so an aggregate passing placeholder cannot satisfy release."""
        except json.JSONDecodeError as error:
            errors.append(f"cross-platform agreement is unreadable: {error}")
        else:
            errors.extend(
                agreement_record_errors(
                    record=agreement_record,
                    manifest=manifest,
                    source_commit=str(source_commit),
                    wheel_sha256s=wheel_sha256s,
                )
            )
    """Bound actual per-field Mac/Linux decisions to this fixed wheel set."""

    for lock_name in ("Cargo.lock", "uv.lock"):
        if not (root / lock_name).is_file():
            errors.append(f"required lock file does not exist: {lock_name}")
    """Required both language dependency graphs to be hashable release inputs."""

    if output_directory.exists():
        errors.append(f"evidence output already exists: {output_directory}")
    if errors:
        raise ReleaseConfigurationError("; ".join(errors))
    """Completed every fail-closed preflight before mutating the evidence directory."""

    command_directory: Path = output_directory / "commands"
    """Selected the isolated directory holding exact process streams."""

    command_directory.mkdir(parents=True)
    """Created a new evidence tree so an earlier run can never be overwritten."""

    copied_agreement_path: Path = output_directory / "cross-platform-agreement.json"
    """Selected the durable in-evidence copy of the downloaded comparison record."""

    copied_agreement_path.write_bytes(agreement_bytes)
    """Preserved exact comparison bytes beside scientific command logs."""

    commands: list[dict[str, Any]] = []
    """Accumulated exact argument, status and output identities for every rule."""

    command_environment: dict[str, str] = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ASTERISM_")
    }
    """Removed exploratory replicate, worker, and method overrides from release runs."""

    for index, (analysis_id, rule) in enumerate(rules):
        check_id: str = str(rule["id"])
        """Read the validated check identifier used by evidence paths."""

        configured_command: list[str] = list(rule["command"])
        """Copied the validated configured argument vector without a shell."""

        argv: list[str] = [sys.executable, *configured_command]
        """Bound the check to the interpreter containing the installed wheel."""

        prefix: str = f"{index:03d}-{analysis_id}-{check_id}"
        """Constructed deterministic, collision-free log names."""

        stdout_path: Path = command_directory / f"{prefix}.stdout.txt"
        """Selected the durable standard-output record for this command."""

        stderr_path: Path = command_directory / f"{prefix}.stderr.txt"
        """Selected the durable standard-error record for this command."""

        started_at: str = datetime.now(UTC).isoformat()
        """Recorded when the scientific command began in UTC."""

        try:
            completed: subprocess.CompletedProcess[str] = subprocess.run(
                argv,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=int(rule["timeout_seconds"]),
                env=command_environment,
            )
            """Executed the configured Python script without shell interpretation."""

            actual_exit_code: int | None = completed.returncode
            """Recorded the process status used by the configured pass decision."""

            stdout: str = completed.stdout
            """Captured the command's complete ordinary output."""

            stderr: str = completed.stderr
            """Captured the command's complete diagnostic output."""

            timed_out: bool = False
            """Recorded that the command completed inside its configured bound."""
        except subprocess.TimeoutExpired as error:
            actual_exit_code = None
            """Recorded the absence of an exit status for the terminated command."""

            stdout = error.stdout if isinstance(error.stdout, str) else ""
            """Preserved any ordinary output captured before the timeout."""

            stderr = error.stderr if isinstance(error.stderr, str) else ""
            """Preserved any diagnostic output captured before the timeout."""

            timed_out = True
            """Converted a bounded timeout into an auditable failed command record."""

        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        """Preserved exact process output before calculating its identity."""

        passed: bool = not timed_out and actual_exit_code == 0
        """Applied the validated zero-exit pass rule to this actual command."""

        commands.append(
            {
                "analysis_id": analysis_id,
                "check_id": check_id,
                "argv": argv,
                "expected_exit_code": 0,
                "actual_exit_code": actual_exit_code,
                "timeout_seconds": int(rule["timeout_seconds"]),
                "status": str(rule["status"]),
                "design_facts": dict(rule["design_facts"]),
                "timed_out": timed_out,
                "passed": passed,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "stdout_path": str(stdout_path.relative_to(output_directory)),
                "stdout_sha256": file_identity(stdout_path)["sha256"],
                "stderr_path": str(stderr_path.relative_to(output_directory)),
                "stderr_sha256": file_identity(stderr_path)["sha256"],
            }
        )
        """Recorded the exact command, decision and log hashes for independent review."""
    """Ran every configured scientific pass rule even when an earlier rule failed."""

    synthetic_configuration: dict[str, Any] = dict(
        manifest["synthetic_analysis_receipts"]
    )
    """Read the standard-receipt command already validated during preflight."""

    synthetic_directory: Path = output_directory / "synthetic-analysis-receipts"
    """Selected the durable directory written atomically by the receipt command."""

    synthetic_stdout_path: Path = output_directory / "synthetic-receipts.stdout.txt"
    """Selected the durable ordinary-output record for the receipt subprocess."""

    synthetic_stderr_path: Path = output_directory / "synthetic-receipts.stderr.txt"
    """Selected the durable diagnostic-output record for the receipt subprocess."""

    synthetic_argv: list[str] = [
        sys.executable,
        str(synthetic_configuration["runner"]),
        "--output",
        str(synthetic_directory),
        "--wheel",
        *[str(path) for path in resolved_wheels],
    ]
    """Bound the public command to this installed interpreter and complete wheel set."""

    synthetic_started_at: str = datetime.now(UTC).isoformat()
    """Recorded when end-to-end standard receipt production began."""

    try:
        synthetic_completed: subprocess.CompletedProcess[str] = subprocess.run(
            synthetic_argv,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(synthetic_configuration["timeout_seconds"]),
            env=command_environment,
        )
        """Executed the committed receipt command without shell interpretation."""

        synthetic_exit_code: int | None = synthetic_completed.returncode
        """Recorded the actual command status used by the release decision."""

        synthetic_stdout: str = synthetic_completed.stdout
        """Captured complete ordinary output before calculating its identity."""

        synthetic_stderr: str = synthetic_completed.stderr
        """Captured complete diagnostics before calculating their identity."""

        synthetic_timed_out: bool = False
        """Recorded that the command completed inside its prewritten bound."""
    except subprocess.TimeoutExpired as error:
        synthetic_exit_code = None
        """Recorded that a terminated command produced no exit status."""

        synthetic_stdout = error.stdout if isinstance(error.stdout, str) else ""
        """Preserved ordinary output captured before the timeout."""

        synthetic_stderr = error.stderr if isinstance(error.stderr, str) else ""
        """Preserved diagnostic output captured before the timeout."""

        synthetic_timed_out = True
        """Converted the bounded timeout into an auditable failed run."""

    synthetic_stdout_path.write_text(synthetic_stdout, encoding="utf-8")
    synthetic_stderr_path.write_text(synthetic_stderr, encoding="utf-8")
    """Preserved exact receipt-command streams regardless of its outcome."""

    synthetic_receipts: list[dict[str, Any]] = []
    """Reserved the parsed values-free index only after successful production."""

    synthetic_index_path: Path = synthetic_directory / "index.json"
    """Selected the canonical index the committed command must produce."""

    synthetic_index_sha256: str | None = None
    """Withheld a digest until a present index passed structural checks."""

    synthetic_index_valid: bool = False
    """Defaulted malformed or absent receipt inventories to release failure."""

    if (
        synthetic_exit_code == 0
        and not synthetic_timed_out
        and synthetic_index_path.is_file()
    ):
        try:
            synthetic_index: object = json.loads(
                synthetic_index_path.read_text(encoding="utf-8")
            )
            """Parsed the command's durable inventory rather than trusting stdout."""
        except json.JSONDecodeError:
            synthetic_index = None
            """Retained an invalid state for independently verifiable evidence."""

        if isinstance(synthetic_index, dict):
            raw_receipts: object = synthetic_index.get("receipts")
            """Read the ordered receipt identities from the versioned index."""

            if (
                synthetic_index.get("schema_version") == 1
                and isinstance(raw_receipts, list)
                and all(isinstance(receipt, dict) for receipt in raw_receipts)
            ):
                synthetic_receipts = [dict(receipt) for receipt in raw_receipts]
                """Copied well-formed index entries into release evidence."""

                expected_ids: list[str] = list(synthetic_configuration["analysis_ids"])
                """Read the exact configured supported-analysis order."""

                synthetic_index_valid = [
                    receipt.get("analysis_id") for receipt in synthetic_receipts
                ] == expected_ids and all(
                    receipt.get("outcome") == "reportable"
                    for receipt in synthetic_receipts
                )
                """Required complete ordered reportable coverage before release."""

                synthetic_index_sha256 = file_identity(synthetic_index_path)["sha256"]
                """Committed the exact index bytes for independent verification."""
    """Accepted neither a zero exit alone nor an aggregate passing placeholder."""

    synthetic_passed: bool = (
        synthetic_exit_code == 0 and not synthetic_timed_out and synthetic_index_valid
    )
    """Applied command status and complete-index structure to the final decision."""

    synthetic_evidence: dict[str, Any] = {
        "argv": synthetic_argv,
        "timeout_seconds": int(synthetic_configuration["timeout_seconds"]),
        "actual_exit_code": synthetic_exit_code,
        "timed_out": synthetic_timed_out,
        "passed": synthetic_passed,
        "started_at": synthetic_started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "stdout_path": str(synthetic_stdout_path.relative_to(output_directory)),
        "stdout_sha256": file_identity(synthetic_stdout_path)["sha256"],
        "stderr_path": str(synthetic_stderr_path.relative_to(output_directory)),
        "stderr_sha256": file_identity(synthetic_stderr_path)["sha256"],
        "index_path": (
            str(synthetic_index_path.relative_to(output_directory))
            if synthetic_index_path.is_file()
            else None
        ),
        "index_sha256": synthetic_index_sha256,
        "receipts": synthetic_receipts,
    }
    """Retained command, log and per-analysis identities beside scientific evidence."""

    resource_configuration: dict[str, Any] = dict(manifest["target_resource_budgets"])
    """Read the target profiles and acceptances already validated during preflight."""

    configured_measurements: list[dict[str, Any]] = list(
        resource_configuration["measurements"]
    )
    """Read the complete ordered route measurements sharing one intended host."""

    resource_output_path: Path = output_directory / "target-resource-measurements.json"
    """Selected the durable values-free measurement artifact path."""

    resource_stdout_path: Path = output_directory / "target-resources.stdout.txt"
    """Selected the durable ordinary-output record for resource orchestration."""

    resource_stderr_path: Path = output_directory / "target-resources.stderr.txt"
    """Selected the durable diagnostic-output record for resource orchestration."""

    resource_argv: list[str] = [
        sys.executable,
        str(resource_configuration["runner"]),
        "--manifest",
        str(manifest_path),
        "--output",
        str(resource_output_path),
        "--qualification",
        "release_target",
        "--host-id",
        str(configured_measurements[0]["host_id"]),
        "--wheel",
        *[str(path) for path in resolved_wheels],
    ]
    """Bound the target runner to this manifest, intended host and saved wheel set."""

    resource_started_at: str = datetime.now(UTC).isoformat()
    """Recorded when all target-route measurements began in UTC."""

    try:
        resource_completed: subprocess.CompletedProcess[str] = subprocess.run(
            resource_argv,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(resource_configuration["timeout_seconds"]),
            env=command_environment,
        )
        """Executed the committed target runner without shell interpretation."""

        resource_exit_code: int | None = resource_completed.returncode
        """Recorded the actual process status used by the resource decision."""

        resource_stdout: str = resource_completed.stdout
        """Captured complete ordinary output before hashing it."""

        resource_stderr: str = resource_completed.stderr
        """Captured complete diagnostics before hashing them."""

        resource_timed_out: bool = False
        """Recorded completion inside the outer orchestration bound."""
    except subprocess.TimeoutExpired as error:
        resource_exit_code = None
        """Recorded that a terminated resource command produced no exit status."""

        resource_stdout = error.stdout if isinstance(error.stdout, str) else ""
        """Preserved ordinary output captured before the timeout."""

        resource_stderr = error.stderr if isinstance(error.stderr, str) else ""
        """Preserved diagnostic output captured before the timeout."""

        resource_timed_out = True
        """Converted the bounded timeout into an auditable failed resource run."""

    resource_stdout_path.write_text(resource_stdout, encoding="utf-8")
    """Persisted exact target-runner ordinary output."""

    resource_stderr_path.write_text(resource_stderr, encoding="utf-8")
    """Persisted exact target-runner diagnostics."""

    resource_record: object = None
    """Withheld embedded resource details until present JSON parsed successfully."""

    resource_record_errors: list[str] = []
    """Accumulated independent identity and budget-verification failures."""

    if resource_output_path.is_file():
        try:
            resource_record = json.loads(
                resource_output_path.read_text(encoding="utf-8")
            )
            """Parsed the retained artifact independently of command output."""
        except json.JSONDecodeError as error:
            resource_record_errors.append(
                f"target resource artifact is unreadable: {error}"
            )
        """Converted malformed resource JSON into a release failure."""
    else:
        resource_record_errors.append("target resource artifact does not exist")
    """Required actual per-route observations rather than a zero exit alone."""

    expected_resource_build: dict[str, Any] = {
        "version": str(core.__version__),
        "cargo_version": str(manifest["cargo_version"]),
        "source_commit": str(source_commit),
        "source_dirty": bool(core.__source_dirty__),
        "release": bool(manifest["release"]),
        "release_manifest_sha256": hashlib.sha256(
            manifest_text.encode("utf-8")
        ).hexdigest(),
        "cargo_lock_sha256": hashlib.sha256(
            cargo_lock_text.encode("utf-8")
        ).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(uv_lock_text.encode("utf-8")).hexdigest(),
    }
    """Rendered exactly the public build identity returned by the installed package."""

    resource_record_errors.extend(
        target_resource_record_errors(
            resource_record,
            manifest=manifest,
            root=root,
            expected_build=expected_resource_build,
            wheel_sha256s=wheel_sha256s,
        )
    )
    """Recomputed every stale-input, completion and resource-budget decision."""

    resource_passed: bool = (
        resource_exit_code == 0
        and not resource_timed_out
        and not resource_record_errors
    )
    """Required both process success and independently verified per-route evidence."""

    resource_evidence: dict[str, Any] = {
        "argv": resource_argv,
        "timeout_seconds": int(resource_configuration["timeout_seconds"]),
        "actual_exit_code": resource_exit_code,
        "timed_out": resource_timed_out,
        "passed": resource_passed,
        "verification_errors": resource_record_errors,
        "started_at": resource_started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "stdout_path": str(resource_stdout_path.relative_to(output_directory)),
        "stdout_sha256": file_identity(resource_stdout_path)["sha256"],
        "stderr_path": str(resource_stderr_path.relative_to(output_directory)),
        "stderr_sha256": file_identity(resource_stderr_path)["sha256"],
        "path": (
            str(resource_output_path.relative_to(output_directory))
            if resource_output_path.is_file()
            else None
        ),
        "sha256": (
            file_identity(resource_output_path)["sha256"]
            if resource_output_path.is_file()
            else None
        ),
        "record": resource_record,
    }
    """Retained command, logs, exact artifact bytes and independent decisions."""

    manifest_identity: dict[str, str] = file_identity(manifest_path.resolve())
    """Identified the authoritative contract used for this run."""

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "sanitized_environment_prefixes": ["ASTERISM_"],
        "passed": (
            bool(commands)
            and all(command["passed"] for command in commands)
            and synthetic_passed
            and resource_passed
        ),
        "build": {
            "version": str(core.__version__),
            "cargo_version": str(manifest["cargo_version"]),
            "rust_toolchain": str(manifest["rust_toolchain"]),
            "maturin_version": str(manifest["maturin_version"]),
            "uv_version": str(manifest["uv_version"]),
            "release": bool(manifest["release"]),
            "source_commit": str(source_commit),
            "source_dirty": bool(core.__source_dirty__),
            "extension_path": str(core_path),
            "extension_sha256": file_identity(core_path)["sha256"],
            "release_manifest_sha256": hashlib.sha256(
                manifest_text.encode("utf-8")
            ).hexdigest(),
            "cargo_lock_sha256": hashlib.sha256(
                cargo_lock_text.encode("utf-8")
            ).hexdigest(),
            "uv_lock_sha256": hashlib.sha256(uv_lock_text.encode("utf-8")).hexdigest(),
        },
        "inputs": {
            "release.toml": manifest_identity,
            "Cargo.lock": file_identity(root / "Cargo.lock"),
            "uv.lock": file_identity(root / "uv.lock"),
        },
        "wheels": [file_identity(path) for path in resolved_wheels],
        "commands": commands,
        "synthetic_analysis_receipts": synthetic_evidence,
        "target_resource_budgets": resource_evidence,
        "cross_platform_agreement": {
            "path": str(copied_agreement_path.relative_to(output_directory)),
            "sha256": file_identity(copied_agreement_path)["sha256"],
            "record": agreement_record,
        },
    }
    """Bound build, contract, dependencies, artifacts and scientific outcomes together."""

    evidence_path: Path = output_directory / "evidence.json"
    """Selected the canonical machine-readable evidence path."""

    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    """Persisted the complete evidence in a stable machine-readable representation."""

    return evidence


def main() -> int:
    """Run the configured release checks through the installed-wheel interpreter."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run configured scientific release checks against a fixed wheel."
    )
    """Defined the fail-closed command used only after installed-wheel testing."""

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-platform-agreement", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, nargs="+", required=True)
    arguments: argparse.Namespace = parser.parse_args()
    """Parsed the authoritative manifest, evidence location and built artifacts."""

    core: ModuleType = importlib.import_module("asterism._core")
    """Loaded immutable build metadata from the interpreter's installed wheel."""

    try:
        evidence: dict[str, Any] = run_scientific_release(
            manifest_path=arguments.manifest,
            output_directory=arguments.output,
            wheel_paths=arguments.wheel,
            cross_platform_agreement_path=arguments.cross_platform_agreement,
            core=core,
        )
        """Executed all configured rules and wrote evidence even for scientific failures."""
    except (OSError, ReleaseConfigurationError, tomllib.TOMLDecodeError) as error:
        print(f"scientific release refused: {error}", file=sys.stderr)
        return 1
    if not evidence["passed"]:
        print(
            "scientific release checks failed; inspect release-evidence",
            file=sys.stderr,
        )
        return 1
    print("All configured scientific release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
