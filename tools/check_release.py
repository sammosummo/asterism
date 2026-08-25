"""Validate Asterism's release manifest and public version surfaces."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


def sibling(name: str) -> ModuleType:
    """Import one of this tool's heavier siblings, at the moment it is needed.

    `run_scientific_release` reaches for `asterism` and NumPy, because
    measuring a release means running it. This
    module needs neither: everything it does itself is reading and checking
    files.

    That mattered. Importing them at module scope made every entry point pay
    for the compiled package, including `--requested`, which only reads the
    version out of the manifest and answers whether a release was asked for at
    all. The release workflow runs exactly that, in a job that deliberately
    installs nothing because it decides whether anything should be built. So
    the step gating the release could not run, and the release could not start.

    Args:
        name: Sibling module name, without a package prefix.

    Returns:
        The imported sibling, in both package and direct-script execution.
    """
    if __package__:
        return importlib.import_module(f".{name}", __package__)
    return importlib.import_module(name)


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one release file.

    Args:
        path: Existing file whose exact bytes must be verified.

    Returns:
        The lowercase hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_record_reportability_errors(
    fit_record: dict[str, Any], reportable_quantities: list[str]
) -> list[str]:
    """Recompute the release-critical reporting conditions for one fit.

    Args:
        fit_record: Parsed candidate fit retained in a standard receipt.
        reportable_quantities: Fields the release state promises for this analysis branch.

    Returns:
        Human-readable failures, or an empty list for a reportable candidate.
    """
    errors: list[str] = []
    """Collected independent failures without trusting the receipt outcome label."""

    if fit_record.get("converged") is not True:
        errors.append("fit did not converge")
    for quantity in reportable_quantities:
        if fit_record.get(quantity) is None:
            errors.append(f"required quantity {quantity} is missing")
    """Required the converged free fit and every promised top-level quantity."""

    pending: list[tuple[str, Any]] = [("", fit_record)]
    """Seeded a recursive audit of profile diagnostics and numerical values."""

    while pending:
        path, value = pending.pop()
        """Selected the next nested value with its receipt-relative field path."""

        if isinstance(value, dict):
            for key, child in value.items():
                child_path: str = f"{path}.{key}" if path else str(key)
                """Extended the diagnostic path through one mapping field."""

                if key == "profile_failures" and child != 0:
                    errors.append(f"{child_path} is {child}")
                pending.append((child_path, child))
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                pending.append((f"{path}[{index}]", child))
        elif isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{path} is not finite")
    """Applied the same fail-closed conditions independently of run_analysis."""

    return errors


def fixed_build_errors(
    core: ModuleType | Any, repository_commit: str, repository_dirty: bool
) -> list[str]:
    """Return identity failures for an extension intended for release.

    Args:
        core: Installed extension exposing immutable build metadata.
        repository_commit: Commit checked out by release automation.
        repository_dirty: Whether Git sees tracked or untracked checkout changes.

    Returns:
        Human-readable fixed-build failures, or an empty list when current and clean.
    """
    errors: list[str] = []
    """Collected both stale-commit and dirty-build failures together."""

    compiled_commit: str = str(getattr(core, "__source_commit__", ""))
    """Read the source identity embedded while compiling the wheel."""

    if compiled_commit != repository_commit:
        errors.append(
            "compiled source commit does not match the release checkout: "
            f"{compiled_commit!r} != {repository_commit!r}"
        )
    if getattr(core, "__source_dirty__", None) is not False:
        errors.append("compiled source is dirty; release wheels require a clean build")
    if repository_dirty:
        errors.append(
            "release checkout is dirty after excluding ignored release outputs"
        )
    """Required both the compiled artifact and independent checkout to be exact and clean."""

    return errors


def conditional_quantity_errors(analysis: dict[str, Any]) -> list[str]:
    """Return malformed conditional-reporting errors for one analysis.

    Args:
        analysis: Parsed analysis table from the authoritative manifest.

    Returns:
        Errors when conditional sets are ambiguous or do not cover the inventory.
    """
    raw_sets: object = analysis.get("reportable_quantity_sets")
    """Read optional mutually exclusive reportable-quantity selections."""

    if raw_sets is None:
        return []
    identifier: str = str(analysis.get("id", "<missing>"))
    """Named the analysis in every actionable validation error."""

    errors: list[str] = []
    """Collected every conditional-reporting disagreement together."""

    inventory: object = analysis.get("reportable_quantities")
    """Read the complete documentation inventory to be partitioned."""

    if (
        not isinstance(inventory, list)
        or not inventory
        or not all(isinstance(quantity, str) and quantity for quantity in inventory)
    ):
        return [f"release.toml: {identifier} has an invalid reportable inventory"]
    if (
        not isinstance(raw_sets, list)
        or not raw_sets
        or not all(isinstance(quantity_set, dict) for quantity_set in raw_sets)
    ):
        return [f"release.toml: {identifier} reportable_quantity_sets must be tables"]
    """Required both the inventory and conditional selections to be explicit lists."""

    selected: list[str] = []
    """Accumulated every quantity across the mutually exclusive selections."""

    selectors: set[str] = set()
    """Collected design fields used to choose one applicable quantity set."""


    for index, quantity_set in enumerate(raw_sets):
        when: object = quantity_set.get("when")
        """Read the exact design predicate selecting this reporting mode."""

        quantities: object = quantity_set.get("quantities")
        """Read fields required only when that design predicate applies."""

        if not isinstance(when, dict) or len(when) != 1:
            errors.append(
                f"release.toml: {identifier} quantity set {index} needs one design predicate"
            )
        else:
            selector, selected_value = next(iter(when.items()))
            """Read the single explicit field/value reporting selector."""

            selectors.add(str(selector))

        if (
            not isinstance(quantities, list)
            or not quantities
            or not all(
                isinstance(quantity, str) and quantity for quantity in quantities
            )
        ):
            errors.append(
                f"release.toml: {identifier} quantity set {index} has no quantities"
            )
        else:
            selected.extend(quantities)
            """Added this mode's fields to the exact inventory coverage check."""
    """Validated every selection predicate and its required output fields."""

    if len(selectors) != 1:
        errors.append(
            f"release.toml: {identifier} quantity sets must share one design selector"
        )
    if Counter(selected) != Counter(str(quantity) for quantity in inventory):
        errors.append(
            f"release.toml: {identifier} quantity sets must partition reportable_quantities"
        )
    """Required unambiguous selection and exact, non-overlapping inventory coverage."""

    return errors



def cross_platform_configuration_errors(
    configuration: object,
    root: Path,
    analysis_ids: list[str],
) -> list[str]:
    """Return precise release blockers for the Mac/Linux agreement contract.

    Args:
        configuration: Parsed ``cross_platform_agreement`` manifest value.
        root: Release checkout containing both configured runners.
        analysis_ids: Complete supported-analysis inventory.

    Returns:
        Missing runner, malformed inventory and unmeasured-tolerance errors.
    """
    if not isinstance(configuration, dict):
        return ["release.toml: cross-platform agreement is not configured"]
    errors: list[str] = []
    """Collected independent configuration failures for one repair cycle."""

    if configuration.get("configured") is not True:
        errors.append("release.toml: cross-platform agreement is not configured")
    for field, label in (
        ("probe_runner", "probe runner"),
        ("runner", "agreement runner"),
    ):
        relative: object = configuration.get(field)
        """Read one committed repository-relative executable path."""

        if not isinstance(relative, str) or not relative:
            errors.append(f"release.toml: cross-platform {label} is missing")
            continue
        candidate: Path = root / relative
        """Resolved the configured runner inside the release checkout."""

        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append(f"release.toml: cross-platform {label} escapes the checkout")
        elif not candidate.is_file():
            errors.append(
                f"release.toml: cross-platform {label} does not exist: {relative}"
            )
    """Required both probe production and comparison programs to be committed."""

    exact: object = configuration.get("exact_comparisons")
    """Read structural outputs which must match without numerical tolerance."""

    required_exact: set[str] = {
        "outcome_status",
        "refusal_code",
        "boundary_state",
        "field_presence",
    }
    """Named the complete exact comparison contract from ADR 0012."""

    if not isinstance(exact, list) or {str(field) for field in exact} != required_exact:
        errors.append("release.toml: cross-platform exact comparisons are incomplete")
    raw_tolerances: object = configuration.get("model_tolerances")
    """Read per-analysis numeric field inventories and measured thresholds."""

    if not isinstance(raw_tolerances, list):
        errors.append(
            "release.toml: model-specific cross-platform tolerances are incomplete"
        )
        return errors
    tolerances: dict[str, dict[str, Any]] = {}
    """Indexed well-formed tables while retaining duplicate detection."""

    unmeasured: list[str] = []
    """Named models whose real cross-host variation has not fixed tolerances."""

    for index, tolerance in enumerate(raw_tolerances):
        if not isinstance(tolerance, dict):
            errors.append(
                f"release.toml: cross-platform tolerance {index} is not a table"
            )
            continue
        analysis_id: object = tolerance.get("analysis_id")
        """Read the supported analysis owning this numeric contract."""

        if not isinstance(analysis_id, str) or not analysis_id:
            errors.append(
                f"release.toml: cross-platform tolerance {index} has no analysis id"
            )
            continue
        if analysis_id in tolerances:
            errors.append(
                f"release.toml: duplicate cross-platform tolerance for {analysis_id}"
            )
            continue
        tolerances[analysis_id] = tolerance
        """Retained this unique model contract for exact inventory coverage."""

        numeric_fields: object = tolerance.get("numeric_fields")
        """Read selected normalized numerical outputs before checking thresholds."""

        if (
            not isinstance(numeric_fields, list)
            or not numeric_fields
            or not all(isinstance(field, str) and field for field in numeric_fields)
            or len(set(numeric_fields)) != len(numeric_fields)
        ):
            errors.append(
                f"release.toml: {analysis_id} numeric field inventory is incomplete"
            )
            continue
        if tolerance.get("measured") is not True:
            unmeasured.append(analysis_id)
            """Retained the precise remaining measurement gate for this model."""

            if "absolute" in tolerance or "relative" in tolerance:
                errors.append(
                    f"release.toml: {analysis_id} unmeasured tolerances contain thresholds"
                )
            continue
        expected_fields: set[str] = set(numeric_fields)
        """Required both measured tolerance maps to cover selected fields exactly."""

        for label in ("absolute", "relative"):
            values: object = tolerance.get(label)
            """Read one kind of measured numerical allowance."""

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
                errors.append(
                    f"release.toml: {analysis_id} {label} tolerances are incomplete"
                )
    """Validated every selected numeric path and its measured thresholds."""

    if set(tolerances) != set(analysis_ids):
        errors.append(
            "release.toml: model-specific cross-platform tolerances do not cover "
            "every supported analysis exactly"
        )
    if unmeasured:
        errors.append(
            "release.toml: cross-platform numeric tolerances are unmeasured for: "
            + ", ".join(sorted(unmeasured))
        )
    return errors


def release_evidence_errors(
    *,
    evidence_path: Path,
    wheel_paths: list[Path],
    root: Path,
    core: ModuleType | Any,
) -> list[str]:
    """Return failures in the produced scientific release evidence.

    Args:
        evidence_path: Machine-readable record written by the scientific runner.
        wheel_paths: Complete saved wheel set expected in that record.
        root: Checkout containing the authoritative manifest and lock files.
        core: Installed extension whose fixed build the record must describe.

    Returns:
        Human-readable evidence or artifact failures.
    """
    errors: list[str] = []
    """Collected every evidence mismatch for one actionable release report."""

    if not evidence_path.is_file():
        return [f"release evidence does not exist: {evidence_path}"]
    try:
        parsed: object = json.loads(evidence_path.read_text(encoding="utf-8"))
        """Parsed the exact record produced after the scientific commands."""
    except (OSError, json.JSONDecodeError) as error:
        return [f"release evidence is unreadable: {error}"]
    if not isinstance(parsed, dict):
        return ["release evidence must be a JSON object"]
    evidence: dict[str, Any] = parsed
    """Narrowed the parsed record to its required object representation."""

    if evidence.get("schema_version") != 1:
        errors.append("release evidence schema_version must be 1")
    if evidence.get("passed") is not True:
        errors.append("release evidence does not record an overall pass")
    """Required the supported evidence schema and its final decision."""

    build: object = evidence.get("build")
    """Read the immutable build identity recorded by the runner."""

    if not isinstance(build, dict):
        errors.append("release evidence has no build identity")
    else:
        if build.get("source_commit") != getattr(core, "__source_commit__", None):
            errors.append(
                "release evidence source commit does not match the installed wheel"
            )
        if build.get("source_dirty") is not False:
            errors.append("release evidence does not describe a clean build")
        if build.get("version") != getattr(core, "__version__", None):
            errors.append("release evidence version does not match the installed wheel")
        embedded_commitments: dict[str, str] = {
            "release_manifest_sha256": hashlib.sha256(
                str(getattr(core, "__release_manifest__", "")).encode("utf-8")
            ).hexdigest(),
            "cargo_lock_sha256": hashlib.sha256(
                str(getattr(core, "__cargo_lock__", "")).encode("utf-8")
            ).hexdigest(),
            "uv_lock_sha256": hashlib.sha256(
                str(getattr(core, "__uv_lock__", "")).encode("utf-8")
            ).hexdigest(),
        }
        """Calculated dependency and manifest commitments from installed bytes."""

        for field, digest in embedded_commitments.items():
            if build.get(field) != digest:
                errors.append(
                    f"release evidence {field} does not match the installed wheel"
                )
        """Required the evidence build identity to carry all three commitments."""

        extension_path: Path = Path(str(getattr(core, "__file__", ""))).resolve()
        """Located the extension binary whose hash identifies the tested build."""

        if not extension_path.is_file():
            errors.append(f"installed extension does not exist: {extension_path}")
        elif build.get("extension_sha256") != sha256(extension_path):
            errors.append(
                "release evidence extension SHA-256 does not match the tested build"
            )
    """Bound the evidence record to the actual imported extension binary."""

    inputs: object = evidence.get("inputs")
    """Read the manifest and language-lock identities used by the runner."""

    if not isinstance(inputs, dict):
        errors.append("release evidence has no input hashes")
    else:
        for name in ("release.toml", "Cargo.lock", "uv.lock"):
            input_record: object = inputs.get(name)
            """Read one required source or dependency identity record."""

            input_path: Path = root / name
            """Selected the authoritative file in the release checkout."""

            if not input_path.is_file():
                errors.append(f"release input does not exist: {name}")
            elif not isinstance(input_record, dict) or input_record.get(
                "sha256"
            ) != sha256(input_path):
                errors.append(f"release evidence {name} SHA-256 does not match")
    """Verified both dependency locks and the exact release contract."""

    expected_wheels: dict[str, str] = {}
    """Collected the complete actual wheel set by resolved path."""

    for wheel_path in wheel_paths:
        resolved: Path = wheel_path.resolve()
        """Normalised one saved artifact before hashing and comparison."""

        if not resolved.is_file():
            errors.append(f"release wheel does not exist: {resolved}")
            continue
        expected_wheels[str(resolved)] = sha256(resolved)
        """Recorded the exact digest of one present saved wheel."""
    if not expected_wheels:
        errors.append("release evidence verification requires at least one wheel")
    """Refused release verification without the saved artifact bytes."""

    recorded_wheels: object = evidence.get("wheels")
    """Read wheel identities emitted before any tag was created."""

    actual_wheels: dict[str, str] = {}
    """Collected well-formed evidence wheel identities for exact-set comparison."""

    if isinstance(recorded_wheels, list):
        for record in recorded_wheels:
            if isinstance(record, dict) and isinstance(record.get("path"), str):
                actual_wheels[str(Path(record["path"]).resolve())] = str(
                    record.get("sha256", "")
                )
                """Recorded one well-formed wheel identity from the evidence."""
    if actual_wheels != expected_wheels:
        errors.append("release evidence wheel SHA-256 set does not match saved wheels")
    """Verified every wheel byte-for-byte and rejected missing or extra artifacts."""

    commands: object = evidence.get("commands")
    """Read the actual command and pass records emitted by the runner."""

    try:
        release_manifest: dict[str, Any] = tomllib.loads(
            (root / "release.toml").read_text(encoding="utf-8")
        )
        """Parsed the same rule inventory already bound by the manifest digest."""
    except (OSError, tomllib.TOMLDecodeError):
        release_manifest = {}
        """Left the expected inventory empty after an earlier input-identity failure."""

    expected_rules: dict[tuple[str, str], dict[str, Any]] = {
        (str(analysis.get("id", "")), str(rule.get("id", ""))): rule
        for analysis in release_manifest.get("analyses", [])
        if isinstance(analysis, dict)
        for rule in analysis.get("pass_rules", [])
        if isinstance(rule, dict)
    }
    """Indexed exact argv, timeout, status, and design facts from the manifest."""

    recorded_rule_ids: list[tuple[str, str]] = []
    """Retained command ownership with multiplicity for completeness checking."""

    if not isinstance(commands, list) or not commands:
        errors.append("release evidence contains no scientific command records")
    else:
        for index, record in enumerate(commands):
            if not isinstance(record, dict):
                errors.append(f"release evidence command {index} is not an object")
                continue
            if (
                record.get("passed") is not True
                or record.get("expected_exit_code") != 0
                or record.get("actual_exit_code") != 0
                or record.get("timed_out") is not False
            ):
                errors.append(f"release evidence command {index} did not pass")
            key: tuple[str, str] = (
                str(record.get("analysis_id", "")),
                str(record.get("check_id", "")),
            )
            """Selected the exact configured rule this command claims to satisfy."""

            recorded_rule_ids.append(key)
            expected_rule: dict[str, Any] | None = expected_rules.get(key)
            """Read the immutable rule fields for independent comparison."""

            if expected_rule is None:
                errors.append(f"release evidence command {index} is not configured")
            else:
                argv: object = record.get("argv")
                """Read interpreter plus the shell-free configured arguments."""

                if not isinstance(argv, list) or argv[1:] != expected_rule.get(
                    "command"
                ):
                    errors.append(
                        f"release evidence command {index} argv does not match"
                    )
                if record.get("timeout_seconds") != expected_rule.get(
                    "timeout_seconds"
                ):
                    errors.append(
                        f"release evidence command {index} timeout does not match"
                    )
                if record.get("status") != "ready":
                    errors.append(f"release evidence command {index} was not ready")
                if record.get("design_facts") != expected_rule.get("design_facts"):
                    errors.append(
                        f"release evidence command {index} design facts do not match"
                    )
            for stream in ("stdout", "stderr"):
                relative_log: object = record.get(f"{stream}_path")
                """Read the evidence-relative path for one exact process stream."""

                if not isinstance(relative_log, str):
                    errors.append(
                        f"release evidence command {index} has no {stream} log"
                    )
                    continue
                log_path: Path = (evidence_path.parent / relative_log).resolve()
                """Resolved one command log beneath the evidence directory."""

                try:
                    log_path.relative_to(evidence_path.parent.resolve())
                except ValueError:
                    errors.append(
                        f"release evidence command {index} {stream} log escapes evidence"
                    )
                    continue
                if not log_path.is_file() or record.get(f"{stream}_sha256") != sha256(
                    log_path
                ):
                    errors.append(
                        f"release evidence command {index} {stream} SHA-256 does not match"
                    )
        if sorted(recorded_rule_ids) != sorted(expected_rules):
            errors.append("release evidence scientific command inventory is incomplete")
    """Verified every scientific decision and its immutable command-output identities."""

    synthetic_evidence: object = evidence.get("synthetic_analysis_receipts")
    """Read actual installed-wheel standard receipts, not a passing placeholder."""

    synthetic_configuration: object = release_manifest.get(
        "synthetic_analysis_receipts"
    )
    """Read the command and analysis inventory bound by the manifest digest."""

    if not isinstance(synthetic_evidence, dict):
        errors.append("release evidence has no synthetic receipt evidence")
    elif not isinstance(synthetic_configuration, dict):
        errors.append("release manifest has no synthetic receipt configuration")
    else:
        if (
            synthetic_evidence.get("passed") is not True
            or synthetic_evidence.get("actual_exit_code") != 0
            or synthetic_evidence.get("timed_out") is not False
        ):
            errors.append("synthetic receipt command did not pass")
        synthetic_argv: object = synthetic_evidence.get("argv")
        """Read the exact interpreter, runner, output and wheel argument vector."""

        configured_runner: object = synthetic_configuration.get("runner")
        """Read the repository-relative command committed before release."""

        if (
            not isinstance(synthetic_argv, list)
            or len(synthetic_argv) < 6
            or synthetic_argv[1] != configured_runner
            or synthetic_argv[2] != "--output"
            or synthetic_argv[4] != "--wheel"
            or {
                str(Path(argument).resolve())
                for argument in synthetic_argv[5:]
                if isinstance(argument, str)
            }
            != set(expected_wheels)
        ):
            errors.append("synthetic receipt command argv does not match release")
        if synthetic_evidence.get("timeout_seconds") != synthetic_configuration.get(
            "timeout_seconds"
        ):
            errors.append("synthetic receipt command timeout does not match release")
        for stream in ("stdout", "stderr"):
            relative_log: object = synthetic_evidence.get(f"{stream}_path")
            """Read one evidence-relative standard-receipt command stream."""

            if not isinstance(relative_log, str):
                errors.append(f"synthetic receipt evidence has no {stream} log")
                continue
            log_path: Path = (evidence_path.parent / relative_log).resolve()
            """Resolved the claimed stream beneath the evidence directory."""

            try:
                log_path.relative_to(evidence_path.parent.resolve())
            except ValueError:
                errors.append(f"synthetic receipt {stream} log escapes evidence")
                continue
            if not log_path.is_file() or synthetic_evidence.get(
                f"{stream}_sha256"
            ) != sha256(log_path):
                errors.append(f"synthetic receipt {stream} SHA-256 does not match")
        """Verified exact subprocess streams rather than trusting its zero status."""

        relative_index: object = synthetic_evidence.get("index_path")
        """Read the evidence-relative per-analysis receipt inventory path."""

        receipt_index_path: Path | None = None
        """Reserved the validated index path inside the evidence directory."""

        if not isinstance(relative_index, str):
            errors.append("synthetic receipt evidence has no index")
        else:
            receipt_index_path = (evidence_path.parent / relative_index).resolve()
            """Resolved the claimed index without trusting its path boundary."""

            try:
                receipt_index_path.relative_to(evidence_path.parent.resolve())
            except ValueError:
                errors.append("synthetic receipt index escapes evidence")
                receipt_index_path = None
                """Withheld an out-of-tree index from all subsequent reads."""

        parsed_index: object = None
        """Reserved independently parsed receipt identities after byte validation."""

        if receipt_index_path is not None:
            if not receipt_index_path.is_file():
                errors.append("synthetic receipt index does not exist")
            elif synthetic_evidence.get("index_sha256") != sha256(receipt_index_path):
                errors.append("synthetic receipt index SHA-256 does not match")
            else:
                try:
                    parsed_index = json.loads(
                        receipt_index_path.read_text(encoding="utf-8")
                    )
                    """Parsed exact index bytes after validating their identity."""
                except json.JSONDecodeError:
                    errors.append("synthetic receipt index is unreadable")

        if (
            receipt_index_path is not None
            and isinstance(synthetic_argv, list)
            and len(synthetic_argv) >= 4
            and isinstance(synthetic_argv[3], str)
        ):
            claimed_output: Path = Path(synthetic_argv[3])
            """Read the destination actually supplied to the receipt subprocess."""

            if not claimed_output.is_absolute():
                claimed_output = root / claimed_output
                """Resolved repository-relative argv exactly as the runner did."""

            if claimed_output.resolve() != receipt_index_path.parent.resolve():
                errors.append("synthetic receipt command output does not match index")
        """Bound retained receipt bytes to the command that was recorded as producing them."""

        indexed_receipts: list[dict[str, Any]] = []
        """Collected only structurally valid index entries for file verification."""

        if (
            not isinstance(parsed_index, dict)
            or parsed_index.get("schema_version") != 1
        ):
            if receipt_index_path is not None:
                errors.append("synthetic receipt index schema is invalid")
        else:
            raw_indexed_receipts: object = parsed_index.get("receipts")
            """Read the per-analysis identities from the validated index."""

            if not isinstance(raw_indexed_receipts, list) or not all(
                isinstance(receipt, dict) for receipt in raw_indexed_receipts
            ):
                errors.append("synthetic receipt index inventory is invalid")
            else:
                indexed_receipts = [dict(receipt) for receipt in raw_indexed_receipts]
                """Copied the complete ordered inventory for independent checking."""

        if synthetic_evidence.get("receipts") != indexed_receipts:
            errors.append("synthetic receipt evidence does not embed its exact index")
        expected_receipt_ids: object = synthetic_configuration.get("analysis_ids")
        """Read the complete supported inventory configured before execution."""

        actual_receipt_ids: list[object] = [
            receipt.get("analysis_id") for receipt in indexed_receipts
        ]
        """Retained order and multiplicity so duplicates cannot hide omissions."""

        if actual_receipt_ids != expected_receipt_ids:
            errors.append("synthetic receipt inventory does not cover every analysis")

        release_build: object = evidence.get("build")
        """Read the broader build record whose immutable subset receipts must echo."""

        expected_receipt_build: dict[str, Any] = (
            {
                field: release_build.get(field)
                for field in (
                    "version",
                    "cargo_version",
                    "source_commit",
                    "source_dirty",
                    "release",
                    "release_manifest_sha256",
                    "cargo_lock_sha256",
                    "uv_lock_sha256",
                )
            }
            if isinstance(release_build, dict)
            else {}
        )
        """Selected exactly the public build identity carried by fit records."""

        expected_wheel_hashes: set[str] = set(expected_wheels.values())
        """Collected saved artifact identities permitted in caller provenance."""

        for index, indexed in enumerate(indexed_receipts):
            analysis_id: object = indexed.get("analysis_id")
            """Read the stable analysis identifier expected inside this receipt."""

            if indexed.get("outcome") != "reportable":
                errors.append(f"synthetic receipt {index} is not reportable")
            relative_receipt: object = indexed.get("path")
            """Read the index-relative complete standard-receipt path."""

            receipt_path: Path | None = None
            """Reserved a validated receipt path beneath the index directory."""

            if not isinstance(relative_receipt, str) or receipt_index_path is None:
                errors.append(f"synthetic receipt {index} has no path")
            else:
                receipt_path = (receipt_index_path.parent / relative_receipt).resolve()
                """Resolved one complete receipt without trusting its path boundary."""

                try:
                    receipt_path.relative_to(receipt_index_path.parent.resolve())
                except ValueError:
                    errors.append(f"synthetic receipt {index} escapes its directory")
                    receipt_path = None
                    """Withheld an escaping path from all reads and hash checks."""

            parsed_receipt: object = None
            """Reserved one independently parsed standard receipt after byte checks."""

            receipt_text: str = ""
            """Retained exact text for a direct no-identifier assertion."""

            if receipt_path is not None:
                if not receipt_path.is_file():
                    errors.append(f"synthetic receipt {index} does not exist")
                elif indexed.get("sha256") != sha256(receipt_path):
                    errors.append(f"synthetic receipt {index} SHA-256 does not match")
                else:
                    receipt_text = receipt_path.read_text(encoding="utf-8")
                    """Read validated bytes only after their index commitment matched."""

                    try:
                        parsed_receipt = json.loads(receipt_text)
                        """Parsed the complete caller-side standard receipt."""
                    except json.JSONDecodeError:
                        errors.append(f"synthetic receipt {index} is unreadable")
            if "synthetic-" in receipt_text:
                errors.append(f"synthetic receipt {index} retained subject identifiers")
            if not isinstance(parsed_receipt, dict):
                continue
            if (
                parsed_receipt.get("schema_version") != 1
                or parsed_receipt.get("asterism_version")
                != expected_receipt_build.get("version")
                or parsed_receipt.get("analysis") != analysis_id
                or parsed_receipt.get("outcome") != "reportable"
                or parsed_receipt.get("refusal") is not None
                or parsed_receipt.get("reportability_issues") != []
            ):
                errors.append(f"synthetic receipt {index} outcome contract is invalid")
            if parsed_receipt.get("build") != expected_receipt_build:
                errors.append(
                    f"synthetic receipt {index} build identity does not match"
                )
            release_state: object = parsed_receipt.get("release_state")
            """Read the measured-range decision produced before numerical fitting."""

            fit_record: object = parsed_receipt.get("fit_record")
            """Read the complete candidate retained by the standard receipt."""

            if (
                not isinstance(release_state, dict)
                or release_state.get("release_ready") is not True
            ):
                errors.append(f"synthetic receipt {index} release state did not pass")
            if not isinstance(fit_record, dict):
                errors.append(f"synthetic receipt {index} has no fit record")
                continue
            if (
                fit_record.get("converged") is not True
                or fit_record.get("build") != expected_receipt_build
            ):
                errors.append(f"synthetic receipt {index} fit identity is invalid")
            quantities: object = (
                release_state.get("reportable_quantities")
                if isinstance(release_state, dict)
                else None
            )
            """Read fields the manifest required for this exact report branch."""

            if not isinstance(quantities, list) or not all(
                isinstance(quantity, str) and quantity in fit_record
                for quantity in quantities
            ):
                errors.append(
                    f"synthetic receipt {index} reportable quantities are incomplete"
                )
            else:
                independent_issues: list[str] = fit_record_reportability_errors(
                    fit_record, quantities
                )
                """Recomputed scientific reportability from the retained fit itself."""

                if independent_issues:
                    errors.append(
                        f"synthetic receipt {index} fit is not reportable: "
                        + "; ".join(independent_issues)
                    )
            provenance: object = parsed_receipt.get("provenance")
            """Read caller-owned wheel, dependency, source and input commitments."""

            if not isinstance(provenance, dict):
                errors.append(f"synthetic receipt {index} has no provenance")
                continue
            if provenance.get("wheel_sha256") not in expected_wheel_hashes:
                errors.append(f"synthetic receipt {index} wheel identity is invalid")
            if indexed.get("wheel_sha256") != provenance.get("wheel_sha256"):
                errors.append(f"synthetic receipt {index} index wheel does not match")
            if provenance.get("dependency_lock_sha256") != expected_receipt_build.get(
                "uv_lock_sha256"
            ):
                errors.append(
                    f"synthetic receipt {index} dependency identity is invalid"
                )
            if provenance.get("consumer_commit") != expected_receipt_build.get(
                "source_commit"
            ):
                errors.append(f"synthetic receipt {index} consumer identity is invalid")
            commitments: object = provenance.get("input_commitments")
            """Read values-free input and exact row-order identities."""

            if (
                not isinstance(commitments, dict)
                or "synthetic_fixture_sha256" not in commitments
                or "subject_order_sha256" not in commitments
                or any(
                    not isinstance(label, str)
                    or not isinstance(commitment, str)
                    or re.fullmatch(r"[0-9a-f]{64}", commitment) is None
                    for label, commitment in commitments.items()
                )
            ):
                errors.append(
                    f"synthetic receipt {index} input commitments are invalid"
                )
            elif fit_record.get("subject_order_sha256") != commitments.get(
                "subject_order_sha256"
            ):
                errors.append(f"synthetic receipt {index} subject order does not match")
        """Verified every complete receipt independently of its command's decision."""

    cross_platform_evidence: object = evidence.get("cross_platform_agreement")
    """Read the preserved actual Mac/Linux comparison and its byte identity."""

    if not isinstance(cross_platform_evidence, dict):
        errors.append("release evidence has no cross-platform agreement artifact")
    else:
        relative_record: object = cross_platform_evidence.get("path")
        """Read the evidence-relative copy retained beside scientific logs."""

        comparison_path: Path | None = None
        """Reserved the validated comparison path inside the evidence directory."""

        if not isinstance(relative_record, str):
            errors.append("release evidence cross-platform agreement has no path")
        else:
            comparison_path = (evidence_path.parent / relative_record).resolve()
            """Resolved the preserved record without trusting its path boundary."""

            try:
                comparison_path.relative_to(evidence_path.parent.resolve())
            except ValueError:
                errors.append(
                    "release evidence cross-platform agreement escapes evidence"
                )
                comparison_path = None
                """Withheld an out-of-tree path from all subsequent reads."""
        copied_record: object = None
        """Reserved the independently parsed preserved comparison details."""

        if comparison_path is not None:
            if not comparison_path.is_file():
                errors.append(
                    "release evidence cross-platform agreement does not exist"
                )
            elif cross_platform_evidence.get("sha256") != sha256(comparison_path):
                errors.append(
                    "release evidence cross-platform agreement SHA-256 does not match"
                )
            else:
                try:
                    copied_record = json.loads(
                        comparison_path.read_text(encoding="utf-8")
                    )
                    """Parsed the hashed retained bytes independently of embedded JSON."""
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(
                        f"release evidence cross-platform agreement is unreadable: {error}"
                    )
        embedded_record: object = cross_platform_evidence.get("record")
        """Read comparison details embedded for single-file evidence inspection."""

        if copied_record != embedded_record:
            errors.append(
                "release evidence cross-platform agreement record does not match its file"
            )
        try:
            manifest: dict[str, Any] = tomllib.loads(
                (root / "release.toml").read_text(encoding="utf-8")
            )
            """Parsed the authoritative contract for per-field inventory validation."""
        except (OSError, tomllib.TOMLDecodeError) as error:
            errors.append(
                f"release manifest is unreadable during evidence check: {error}"
            )
        else:
            errors.extend(
                sibling("run_scientific_release").agreement_record_errors(
                    record=embedded_record,
                    manifest=manifest,
                    source_commit=str(getattr(core, "__source_commit__", "")),
                    wheel_sha256s=set(expected_wheels.values()),
                )
            )
    """Required actual per-field decisions rather than a passing placeholder."""

    return errors


def main() -> int:
    """Check metadata consistency, and optionally require release readiness."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Validate Asterism release metadata."
    )
    """Defined the command-line interface used locally and in automation."""

    mode: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group()
    """Kept the ordinary metadata and release-readiness modes distinct."""

    mode.add_argument("--metadata", action="store_true")
    mode.add_argument("--release", action="store_true")
    mode.add_argument("--requested", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--wheel", type=Path, nargs="*")
    arguments: argparse.Namespace = parser.parse_args()
    """Parsed the requested validation mode."""

    root: Path = Path(__file__).resolve().parents[1]
    """Located the repository root containing every version surface."""

    manifest_path: Path = root / "release.toml"
    """Selected the one editable release manifest."""

    manifest_text: str = manifest_path.read_text(encoding="utf-8")
    """Read the exact bytes that must be embedded in built artifacts."""

    manifest: dict[str, Any] = tomllib.loads(manifest_text)
    """Parsed the release contract."""

    if arguments.requested:
        requested_version: str = str(manifest.get("version", ""))
        """Read the public version used to detect a partial release request."""

        requested: bool = bool(manifest.get("release")) or (
            bool(requested_version) and "dev" not in requested_version
        )
        """Detected either release opt-in so an inconsistent request fails closed."""

        print(f"requested={str(requested).lower()}")
        return 0

    core: ModuleType = importlib.import_module("asterism._core")
    """Loaded the built extension only for artifact-aware validation modes."""

    pyproject: dict[str, Any] = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    """Parsed Python package metadata."""

    cargo: dict[str, Any] = tomllib.loads(
        (root / "Cargo.toml").read_text(encoding="utf-8")
    )
    """Parsed Rust package metadata."""

    rust_toolchain: dict[str, Any] = tomllib.loads(
        (root / "rust-toolchain.toml").read_text(encoding="utf-8")
    )
    """Parsed the repository's exact local and automation compiler selection."""

    cargo_lock: dict[str, Any] = tomllib.loads(
        (root / "Cargo.lock").read_text(encoding="utf-8")
    )
    """Parsed the locked Rust package graph."""

    uv_lock: dict[str, Any] = tomllib.loads(
        (root / "uv.lock").read_text(encoding="utf-8")
    )
    """Parsed the locked Python package graph."""

    cargo_locked: dict[str, Any] | None = next(
        (
            package
            for package in cargo_lock["package"]
            if package.get("name") == "asterism"
        ),
        None,
    )
    """Located Asterism's own Cargo lock entry."""

    uv_locked: dict[str, Any] | None = next(
        (
            package
            for package in uv_lock["package"]
            if package.get("name") == "asterism"
        ),
        None,
    )
    """Located Asterism's own uv lock entry."""

    maturin_locked: dict[str, Any] | None = next(
        (package for package in uv_lock["package"] if package.get("name") == "maturin"),
        None,
    )
    """Located the exact wheel builder retained in the Python dependency lock."""

    errors: list[str] = []
    """Accumulated every disagreement for one actionable report."""

    if manifest.get("schema_version") != 1:
        errors.append("release.toml: schema_version must be 1")
    if cargo_locked is None:
        errors.append("Cargo.lock: asterism package entry is missing")
    if uv_locked is None:
        errors.append("uv.lock: asterism package entry is missing")
    if maturin_locked is None:
        errors.append("uv.lock: maturin package entry is missing")
    """Checked that the supported manifest and lockfile records exist."""

    python_version: str = str(manifest.get("version", ""))
    """Read the canonical PEP 440 package version."""

    cargo_version: str = str(manifest.get("cargo_version", ""))
    """Read the SemVer rendering required by Cargo."""

    expected: list[tuple[str, str, str]] = [
        (
            "pyproject.toml project.version",
            str(pyproject["project"]["version"]),
            python_version,
        ),
        (
            "pyproject.toml project.requires-python",
            str(pyproject["project"]["requires-python"]),
            str(manifest.get("requires_python", "")),
        ),
        ("Cargo.toml package.version", str(cargo["package"]["version"]), cargo_version),
        (
            "Cargo.lock asterism.version",
            "" if cargo_locked is None else str(cargo_locked.get("version", "")),
            cargo_version,
        ),
        (
            "uv.lock asterism.version",
            "" if uv_locked is None else str(uv_locked.get("version", "")),
            python_version,
        ),
        (
            "Cargo.toml package.rust-version",
            str(cargo["package"].get("rust-version", "")),
            str(manifest.get("rust_toolchain", "")),
        ),
        (
            "rust-toolchain.toml toolchain.channel",
            str(rust_toolchain.get("toolchain", {}).get("channel", "")),
            str(manifest.get("rust_toolchain", "")),
        ),
        (
            "uv.lock maturin.version",
            "" if maturin_locked is None else str(maturin_locked.get("version", "")),
            str(manifest.get("maturin_version", "")),
        ),
        (
            "compiled module version",
            str(core.__version__).replace("-dev.", ".dev"),
            python_version,
        ),
        ("compiled release manifest", str(core.__release_manifest__), manifest_text),
    ]
    """Collected the public version and embedded-manifest comparisons."""

    for label, actual, wanted in expected:
        if actual != wanted:
            errors.append(f"{label}: expected {wanted!r}, found {actual!r}")
    """Compared every derived surface with the authoritative manifest."""

    locked_python: str = re.sub(r"\s+", "", str(uv_lock.get("requires-python", "")))
    """Normalized uv's harmless canonical whitespace around requirement commas."""

    manifest_python: str = re.sub(r"\s+", "", str(manifest.get("requires_python", "")))
    """Normalized the authoritative requirement for semantic lock comparison."""

    if locked_python != manifest_python:
        errors.append(
            "uv.lock requires-python: expected "
            f"{manifest.get('requires_python', '')!r}, "
            f"found {uv_lock.get('requires-python', '')!r}"
        )

    maturin_requirement: str = f"maturin=={manifest.get('maturin_version', '')}"
    """Rendered the exact builder requirement owned by release metadata."""

    if pyproject.get("build-system", {}).get("requires") != [maturin_requirement]:
        errors.append(
            "pyproject.toml build-system.requires does not match maturin_version"
        )
    development_requirements: object = pyproject.get("dependency-groups", {}).get("dev")
    """Read the local development environment's explicit build-tool requirement."""

    if not isinstance(development_requirements, list) or (
        development_requirements.count(maturin_requirement) != 1
    ):
        errors.append("pyproject.toml dev group does not pin maturin_version once")
    uv_pin: str = f'version: "{manifest.get("uv_version", "")}"'
    """Rendered the exact setup-uv input required in every hosted workflow."""

    for workflow_name in ("quality.yml", "wheels.yml", "release.yml"):
        workflow_text: str = (root / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        """Read one hosted build or release workflow as immutable metadata."""

        if uv_pin not in workflow_text:
            errors.append(f"{workflow_name}: setup-uv does not match uv_version")
    wheels_workflow: str = (root / ".github" / "workflows" / "wheels.yml").read_text(
        encoding="utf-8"
    )
    """Read the action-owned manylinux builder version separately from local Maturin."""

    if (
        f"maturin-version: v{manifest.get('maturin_version', '')}"
        not in wheels_workflow
    ):
        errors.append("wheels.yml: maturin-action does not match maturin_version")
    """Bound both build tools across manifest, lock, local and hosted execution."""

    is_release: bool = bool(manifest.get("release"))
    """Read whether this manifest intentionally requests a release."""

    if is_release and ("dev" in python_version or "-" in cargo_version):
        errors.append("release.toml: a release cannot carry a development version")
    if not is_release and ("dev" not in python_version or "-dev." not in cargo_version):
        errors.append(
            "release.toml: a development build must carry development versions"
        )
    """Required development and release identities to be unambiguous."""

    analyses: list[dict[str, Any]] = list(manifest.get("analyses", []))
    """Collected the analyses whose scientific claims gate a release."""

    identifiers: list[str] = [str(analysis.get("id", "")) for analysis in analyses]
    """Collected stable analysis identifiers for uniqueness checks."""

    if not analyses:
        errors.append("release.toml: at least one supported analysis is required")
    if "" in identifiers or len(identifiers) != len(set(identifiers)):
        errors.append("release.toml: analysis ids must be non-empty and unique")
    """Checked the manifest has an unambiguous supported-analysis inventory."""

    for analysis in analyses:
        identifier: str = str(analysis.get("id", "<missing>"))
        """Named the analysis for precise validation errors."""

        if not analysis.get("entry_points"):
            errors.append(f"release.toml: {identifier} has no entry_points")
        if not analysis.get("reportable_quantities"):
            errors.append(f"release.toml: {identifier} has no reportable_quantities")
        if not analysis.get("required_checks"):
            errors.append(f"release.toml: {identifier} has no required_checks")
        errors.extend(conditional_quantity_errors(analysis))
    """Required every supported claim to name its public and evidentiary seams."""

    if arguments.release:
        if not is_release:
            errors.append("release.toml: release mode requires release = true")
        if not manifest.get("scientific_pass_rules_configured"):
            errors.append("release.toml: scientific pass rules are not configured")
        scientific_runner: str = str(manifest.get("scientific_runner", ""))
        """Read the committed command which must execute every configured rule."""

        if not scientific_runner:
            errors.append("release.toml: scientific runner is not configured")
        elif not (root / scientific_runner).is_file():
            errors.append(
                f"release.toml: scientific runner does not exist: {scientific_runner}"
            )
        """Required the global opt-in that prevents premature automatic releases."""

        errors.extend(
            cross_platform_configuration_errors(
                manifest.get("cross_platform_agreement"), root, identifiers
            )
        )
        """Kept installed-wheel testing distinct from actual result agreement."""

        medusa_smoke: object = manifest.get("medusa_smoke")
        """Read the conditional target-host portable-wheel qualification record."""

        if not isinstance(medusa_smoke, dict):
            errors.append("release.toml: Medusa smoke configuration is missing")
        elif medusa_smoke.get("required") is True:
            if medusa_smoke.get("configured") is not True:
                errors.append("release.toml: Medusa wheel smoke is unverified")
            for field in ("architecture", "glibc_version", "evidence"):
                if (
                    not isinstance(medusa_smoke.get(field), str)
                    or not medusa_smoke[field]
                ):
                    errors.append(f"release.toml: Medusa {field} is unmeasured")
            command: object = medusa_smoke.get("command")
            """Read the exact participant-free installed-wheel smoke argument vector."""

            if (
                not isinstance(command, list)
                or not command
                or not all(
                    isinstance(argument, str) and argument for argument in command
                )
            ):
                errors.append("release.toml: Medusa smoke command is not configured")
        """Required host facts and an actual saved-wheel fit, never a source install."""

        synthetic_receipts: object = manifest.get("synthetic_analysis_receipts")
        """Read end-to-end public examples using the standard analysis receipt path."""

        if not isinstance(synthetic_receipts, dict):
            errors.append("release.toml: synthetic analysis receipts are missing")
        else:
            if synthetic_receipts.get("configured") is not True:
                errors.append(
                    "release.toml: synthetic run_analysis receipts are unverified"
                )
            example_runner: object = synthetic_receipts.get("runner")
            """Read the installed-wheel runner responsible for all nine receipts."""

            if not isinstance(example_runner, str) or not example_runner:
                errors.append(
                    "release.toml: synthetic receipt runner is not configured"
                )
            elif not (root / example_runner).is_file():
                errors.append(
                    "release.toml: synthetic receipt runner does not exist: "
                    f"{example_runner}"
                )
            timeout_seconds: object = synthetic_receipts.get("timeout_seconds")
            """Read the prewritten wall-time bound for the complete receipt set."""

            if (
                not isinstance(timeout_seconds, int)
                or isinstance(timeout_seconds, bool)
                or not 1 <= timeout_seconds <= 604_800
            ):
                errors.append(
                    "release.toml: synthetic receipt timeout is not configured"
                )
            receipt_ids: object = synthetic_receipts.get("analysis_ids")
            """Read the exact supported inventory the command must fit."""

            if receipt_ids != identifiers:
                errors.append(
                    "release.toml: synthetic receipt inventory does not cover every analysis"
                )
        """Kept public fit probes distinct from reportable run_analysis receipts."""

        for analysis in analyses:
            identifier = str(analysis.get("id", "<missing>"))
            """Named the analysis for precise release-readiness errors."""


            if not analysis.get("pass_rules_configured"):
                errors.append(
                    f"release.toml: {identifier} pass rules are not configured"
                )
            if not analysis.get("pass_rules"):
                errors.append(
                    f"release.toml: {identifier} has no machine-readable pass_rules"
                )
        """Required every scientific claim to have measured limits and pass rules."""

        errors.extend(
            sibling("run_scientific_release").scientific_inventory_errors(
                manifest, root
            )
        )
        """Required exact one-to-one commands and participant-free design facts."""

        for analysis in analyses:
            for rule in analysis.get("pass_rules", []):
                if isinstance(rule, dict) and rule.get("status") != "ready":
                    blocker: object = rule.get("blocker")
                    """Read the stable evidence deficit already recorded in metadata."""

                    code: object = (
                        blocker.get("code") if isinstance(blocker, dict) else "unknown"
                    )
                    """Selected the machine blocker identity for release diagnostics."""

                    errors.append(
                        "release.toml: scientific rule is blocked: "
                        f"{analysis.get('id')}/{rule.get('id')} ({code})"
                    )
        """Prevented an inventoried absence from becoming a configured release rule."""

        style_result: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "check_python_style.py"),
                "--root",
                str(root),
                "--no-baseline",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        """Ran the live zero-debt checker independently during release readiness."""

        if style_result.returncode != 0:
            style_detail: str = (
                style_result.stderr.strip() or style_result.stdout.strip()
            )
            """Retained complete checker diagnostics for the release failure."""

            errors.append(
                "Python style gate failed"
                + (f":\n{style_detail}" if style_detail else " without diagnostics")
            )
        """Required zero maintained Python style debt before release."""

        git_result: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        """Read the exact checkout commit independently of build-time overrides."""

        git_status: subprocess.CompletedProcess[str] = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        """Checked source cleanliness independently of build-script environment values."""

        if git_result.returncode != 0 or git_status.returncode != 0:
            errors.append("git could not identify the release checkout state")
        else:
            errors.extend(
                fixed_build_errors(
                    core,
                    git_result.stdout.strip(),
                    repository_dirty=bool(git_status.stdout.strip()),
                )
            )
        """Required this clean wheel to come from the checked-out release commit."""

        if arguments.evidence is None:
            errors.append("release mode requires --evidence")
        else:
            errors.extend(
                release_evidence_errors(
                    evidence_path=arguments.evidence,
                    wheel_paths=list(arguments.wheel or []),
                    root=root,
                    core=core,
                )
            )
        """Bound release readiness to produced evidence and the exact saved wheels."""

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    """Reported all disagreements without hiding later failures behind the first."""

    print("Asterism release metadata is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
