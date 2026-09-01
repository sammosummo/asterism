"""Release-state checks and analysis receipts.

The numerical model classes remain in-memory calculations. This module reads
the release contract compiled into the extension and returns serialisable
records; the caller chooses whether and where to write them.
"""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import _core

__all__: list[str] = [
    "build_identity",
    "installed_extension_sha256",
    "release_manifest",
    "run_analysis",
    "subject_order_commitment",
]
"""Declared the supported public analysis-control interface."""

STABLE_ESTIMATOR_CODE_PREFIXES: tuple[str, ...] = (
    "BIVARIATE_",
    "COMPONENTS_",
    "DISCRETE_GXE_",
    "FIT_",
    "GXE_",
    "LIABILITY_",
    "MIXED_BIVARIATE_",
    "PREPARE_",
    "SPATIAL_",
    "TOBIT_",
)
"""Named the documented refusal vocabularies of supported 0.1 estimators."""

SUPPORT_PATTERN: re.Pattern[str] = re.compile(r"^supported_in_(\d+)_(\d+)$")
"""Parsed the release line in which an analysis first became supported."""

VERSION_PATTERN: re.Pattern[str] = re.compile(r"^(\d+)\.(\d+)(?:\.|$)")
"""Read the major and minor release line from PEP 440 development versions."""


# asterism-style: allow private-helper -- only receipt preflight interprets embedded versioned support states
def _analysis_support_is_active(support: Any, version: Any) -> bool:
    """Return whether a versioned support declaration applies to this release.

    Private because only receipt preflight interprets embedded support states;
    callers consume the resulting public release-state record.
    """
    if not isinstance(support, str) or not isinstance(version, str):
        return False
    declared: re.Match[str] | None = SUPPORT_PATTERN.fullmatch(support)
    """Read the analysis's introduction line without accepting planned states."""

    current: re.Match[str] | None = VERSION_PATTERN.match(version)
    """Read the build's current release line, including development versions."""

    if declared is None or current is None:
        return False
    introduced_major, introduced_minor = map(int, declared.groups())
    """Converted the support declaration into an ordered release pair."""

    current_major, current_minor = map(int, current.groups())
    """Converted the current package version into the same comparison pair."""

    return introduced_major == current_major and introduced_minor <= current_minor


def release_manifest() -> dict[str, Any]:
    """Return the release manifest compiled into this Asterism build.

    Returns:
        A newly parsed manifest. Mutating it cannot change the build's embedded
        contract.

    Raises:
        RuntimeError: If the extension was built without release metadata.
        ValueError: If the embedded metadata is not a valid version-1 manifest.
    """
    raw: str | None = getattr(_core, "__release_manifest__", None)
    """Read the immutable manifest text embedded by the Rust build."""
    if raw is None:
        raise RuntimeError("ASTERISM_RELEASE_MANIFEST_MISSING")

    manifest: dict[str, Any] = tomllib.loads(raw)
    """Parsed a fresh caller-owned representation of the release contract."""
    if manifest.get("schema_version") != 1:
        raise ValueError("ASTERISM_RELEASE_MANIFEST_SCHEMA_UNSUPPORTED")
    if not isinstance(manifest.get("analyses"), list):
        raise ValueError("ASTERISM_RELEASE_MANIFEST_ANALYSES_INVALID")
    return manifest


def build_identity() -> dict[str, Any]:
    """Return immutable identity compiled into this Asterism build.

    Returns:
        Public and Cargo versions, source commit, clean/dirty state, whether
        this is a fixed release build, and immutable release/dependency-lock
        commitments.

    Raises:
        RuntimeError: If the extension lacks required source identity.
    """
    manifest: dict[str, Any] = release_manifest()
    """Read version and release status from the authoritative manifest."""
    source_commit: str | None = getattr(_core, "__source_commit__", None)
    """Read the source commit embedded by the Rust build script."""
    source_dirty: bool | None = getattr(_core, "__source_dirty__", None)
    """Read whether tracked or untracked files changed the compiled source."""
    cargo_lock: str | None = getattr(_core, "__cargo_lock__", None)
    """Read the exact Rust dependency lock embedded in the extension."""
    uv_lock: str | None = getattr(_core, "__uv_lock__", None)
    """Read the exact Python dependency lock embedded in the extension."""
    raw_manifest: str | None = getattr(_core, "__release_manifest__", None)
    """Read the exact release contract whose parsed form was validated above."""
    compiled_manifest_sha256: str | None = getattr(
        _core, "__release_manifest_sha256__", None
    )
    """Read the release-contract commitment produced by the build script."""
    compiled_cargo_lock_sha256: str | None = getattr(
        _core, "__cargo_lock_sha256__", None
    )
    """Read the Rust-lock commitment produced by the build script."""
    compiled_uv_lock_sha256: str | None = getattr(_core, "__uv_lock_sha256__", None)
    """Read the Python-lock commitment produced by the build script."""
    if (
        source_commit is None
        or source_dirty is None
        or cargo_lock is None
        or uv_lock is None
        or raw_manifest is None
        or compiled_manifest_sha256 is None
        or compiled_cargo_lock_sha256 is None
        or compiled_uv_lock_sha256 is None
    ):
        raise RuntimeError("ASTERISM_BUILD_IDENTITY_MISSING")

    calculated_commitments: dict[str, str] = {
        "release_manifest_sha256": hashlib.sha256(
            raw_manifest.encode("utf-8")
        ).hexdigest(),
        "cargo_lock_sha256": hashlib.sha256(cargo_lock.encode("utf-8")).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(uv_lock.encode("utf-8")).hexdigest(),
    }
    """Independently checked every build-script digest against embedded bytes."""
    compiled_commitments: dict[str, str] = {
        "release_manifest_sha256": compiled_manifest_sha256,
        "cargo_lock_sha256": compiled_cargo_lock_sha256,
        "uv_lock_sha256": compiled_uv_lock_sha256,
    }
    """Collected the immutable digests compiled into public fit records."""
    if calculated_commitments != compiled_commitments:
        raise RuntimeError("ASTERISM_BUILD_COMMITMENT_MISMATCH")

    return {
        "version": manifest["version"],
        "cargo_version": manifest["cargo_version"],
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "release": manifest["release"],
        **compiled_commitments,
    }


def installed_extension_sha256() -> str:
    """Return the SHA-256 of the native module loaded by this process.

    This is a provenance helper rather than part of a fit record. It follows
    the imported module's own specification, so another extension-shaped file
    beside it cannot be mistaken for the binary that actually ran.

    Returns:
        Lowercase hexadecimal SHA-256 of the loaded native module.

    Raises:
        RuntimeError: If the loaded module has no readable filesystem origin.
    """
    specification: Any = getattr(_core, "__spec__", None)
    """Read the import system's identity for the module already in memory."""

    origin: Any = getattr(specification, "origin", None)
    """Selected the exact file from which that module was loaded."""

    if not isinstance(origin, str) or not origin:
        raise RuntimeError("ASTERISM_EXTENSION_ORIGIN_MISSING")
    path: Path = Path(origin).resolve()
    """Normalised the loaded binary path without searching its directory."""

    if not path.is_file():
        raise RuntimeError("ASTERISM_EXTENSION_ORIGIN_UNREADABLE")
    digest: Any = hashlib.sha256()
    """Started the binary digest without retaining its bytes in memory."""

    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError("ASTERISM_EXTENSION_ORIGIN_UNREADABLE") from error
    return digest.hexdigest()


def subject_order_commitment(subject_order: Sequence[str]) -> str:
    """Commit to exact subject order without retaining identifiers.

    Args:
        subject_order: Identifiers in the same row order as every fitted array.

    Returns:
        Lowercase hexadecimal SHA-256 of the newline-separated identifiers,
        matching Asterism's existing empirical-matrix receipt convention.

    Raises:
        ValueError: If the order is empty, duplicates an identifier, contains a
            non-string identifier or contains a newline.
    """
    identifiers: list[str] = list(subject_order)
    """Copied the caller's identifiers without retaining them beyond this call."""
    if not identifiers:
        raise ValueError("SUBJECT_ORDER_EMPTY")
    if not all(isinstance(identifier, str) for identifier in identifiers):
        raise ValueError("SUBJECT_ORDER_IDENTIFIER_NOT_STRING")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("SUBJECT_ORDER_IDENTIFIER_DUPLICATED")
    if any("\n" in identifier or "\r" in identifier for identifier in identifiers):
        raise ValueError("SUBJECT_ORDER_IDENTIFIER_CONTAINS_NEWLINE")

    encoded: bytes = "\n".join(identifiers).encode("utf-8")
    """Encoded the unambiguous identifier order used by existing receipts."""
    return hashlib.sha256(encoded).hexdigest()


# asterism-style: allow private-helper -- only run_analysis asks this, and exposing it publicly invited callers to check a design before fitting, which statistical software does not ask of anyone
def _release_state(
    analysis: str,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Report whether this build is a release with every required check ready.

    Private because nobody outside `run_analysis` needs it. A caller who wants
    a fit calls the model and gets one; a caller who wants a receipt calls
    `run_analysis`, which asks this itself.

    Args:
        analysis: Stable analysis identifier from the release manifest.
        design: Non-identifying design facts, recorded in the receipt and
            used only to select a reporting branch where an analysis has one.

    Returns:
        A serialisable release-state record. ``release_ready`` is true only
        when the pass rules are configured, every required rule is ready and
        the build is a release; ``missing_checks`` names every absence.

    Raises:
        ValueError: If the analysis is absent from the manifest or its contract
            is malformed.
    """
    manifest: dict[str, Any] = release_manifest()
    """Loaded the exact support contract compiled into this build."""

    entries: list[Any] = manifest["analyses"]
    """Selected the manifest's ordered analysis records."""
    matched: list[dict[str, Any]] = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("id") == analysis
    ]
    """Found records carrying the requested stable analysis identifier."""
    if len(matched) != 1:
        raise ValueError(f"ANALYSIS_NOT_IN_RELEASE_MANIFEST:{analysis}")

    entry: dict[str, Any] = matched[0]
    """Selected the one unambiguous support record for this analysis."""
    if not _analysis_support_is_active(entry.get("support"), manifest.get("version")):
        raise ValueError(f"ANALYSIS_OUTSIDE_RELEASE_SUPPORT:{analysis}")

    missing_checks: list[dict[str, Any]] = []
    """Accumulated whatever this build is missing before it can be released."""
    if (
        manifest.get("scientific_pass_rules_configured") is not True
        or entry.get("pass_rules_configured") is not True
    ):
        missing_checks.append(
            {
                "code": "SCIENTIFIC_PASS_RULES_NOT_CONFIGURED",
                "reason": "machine-readable pass rules are incomplete",
            }
        )

    required_checks: Any = entry.get("required_checks")
    """Read the exact scientific claims this analysis promises to discharge."""

    raw_pass_rules: Any = entry.get("pass_rules")
    """Read the machine-readable evidence state for each required claim."""

    if (
        not isinstance(required_checks, list)
        or not required_checks
        or not all(isinstance(check, str) and check for check in required_checks)
        or len(required_checks) != len(set(required_checks))
        or not isinstance(raw_pass_rules, list)
        or not raw_pass_rules
        or not all(isinstance(rule, Mapping) for rule in raw_pass_rules)
    ):
        raise ValueError(f"ANALYSIS_PASS_RULES_INVALID:{analysis}")
    pass_rules: list[Mapping[str, Any]] = list(raw_pass_rules)
    """Narrowed the validated rule-table list for exact inventory checks."""

    rule_ids: list[Any] = [rule.get("id") for rule in pass_rules]
    """Retained rule multiplicity so a duplicate cannot hide a missing check."""

    if rule_ids != required_checks:
        raise ValueError(f"ANALYSIS_PASS_RULES_INVALID:{analysis}")
    for rule in pass_rules:
        check: str = str(rule["id"])
        """Named the required check whose current evidence state is examined."""

        status: Any = rule.get("status")
        """Read whether this exact check is ready or explicitly waiting."""

        if status not in {"ready", "blocked", "external_fixture_pending"}:
            raise ValueError(f"ANALYSIS_PASS_RULES_INVALID:{analysis}")
        if status != "ready":
            blocker: Any = rule.get("blocker")
            """Read the stable reason and evidence needed for this unready rule."""

            if (
                not isinstance(blocker, Mapping)
                or not isinstance(blocker.get("code"), str)
                or not blocker.get("code")
                or not isinstance(blocker.get("evidence_needed"), str)
                or not blocker.get("evidence_needed")
            ):
                raise ValueError(f"ANALYSIS_PASS_RULES_INVALID:{analysis}")
            missing_checks.append(
                {
                    "code": "SCIENTIFIC_PASS_RULE_NOT_READY",
                    "check": check,
                    "status": status,
                    "blocker": blocker["code"],
                    "reason": blocker["evidence_needed"],
                }
            )
    """Made a blocked or pending rule part of the public fail-closed receipt."""

    if manifest.get("release") is not True:
        missing_checks.append(
            {
                "code": "BUILD_NOT_RELEASED",
                "reason": "this is a development build, not a release",
            }
        )
    """Recorded an unready rule or development build as not release ready."""

    quantities: Any = entry.get("supported_quantities")
    """Read the quantities this analysis is supported to produce."""
    if not isinstance(quantities, list) or not all(
        isinstance(quantity, str) for quantity in quantities
    ):
        raise ValueError(f"ANALYSIS_REPORTABLE_QUANTITIES_INVALID:{analysis}")

    raw_limitations: Any = entry.get("known_limitations", [])
    """Read retained negative evidence that narrows interpretation, not execution."""

    limitation_fields: tuple[str, ...] = (
        "code",
        "quantity",
        "scope",
        "consequence",
        "evidence",
        "evidence_sha256",
    )
    """Named the stable fields every receipt needs to explain one limitation."""

    if (
        not isinstance(raw_limitations, list)
        or not all(isinstance(item, Mapping) for item in raw_limitations)
        or any(
            not all(
                isinstance(item.get(field), str) and item.get(field)
                for field in limitation_fields
            )
            or item.get("quantity") not in quantities
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("evidence_sha256", "")))
            is None
            for item in raw_limitations
        )
    ):
        raise ValueError(f"ANALYSIS_KNOWN_LIMITATIONS_INVALID:{analysis}")
    known_limitations: list[dict[str, Any]] = [dict(item) for item in raw_limitations]
    """Copied limitations so callers cannot mutate the embedded manifest record."""

    applicable_quantities: list[str] = list(quantities)
    """Defaulted to the complete inventory for analyses with one report branch."""
    raw_quantity_sets: Any = entry.get("supported_quantity_sets")
    """Read optional mutually exclusive report branches selected by the design."""
    if raw_quantity_sets is not None:
        if not isinstance(raw_quantity_sets, list) or not raw_quantity_sets:
            raise ValueError(f"ANALYSIS_SUPPORTED_QUANTITY_SETS_INVALID:{analysis}")
        matched_quantity_sets: list[list[str]] = []
        """Collected report branches whose complete design predicate matched."""
        for raw_quantity_set in raw_quantity_sets:
            if not isinstance(raw_quantity_set, Mapping):
                raise ValueError(f"ANALYSIS_SUPPORTED_QUANTITY_SET_INVALID:{analysis}")
            predicate: Any = raw_quantity_set.get("when")
            """Read the exact design values selecting this report branch."""
            selected: Any = raw_quantity_set.get("quantities")
            """Read the manifest quantities required when that predicate matches."""
            if (
                not isinstance(predicate, Mapping)
                or not predicate
                or not isinstance(selected, list)
                or not selected
                or not all(isinstance(quantity, str) for quantity in selected)
                or not set(selected).issubset(quantities)
            ):
                raise ValueError(f"ANALYSIS_SUPPORTED_QUANTITY_SET_INVALID:{analysis}")
            if all(design.get(field) == value for field, value in predicate.items()):
                matched_quantity_sets.append(list(selected))
        """Matched report branches without letting a caller choose quantities directly."""

        if len(matched_quantity_sets) > 1:
            raise ValueError(f"ANALYSIS_SUPPORTED_QUANTITY_SET_AMBIGUOUS:{analysis}")
        if matched_quantity_sets:
            applicable_quantities = matched_quantity_sets[0]
            """Required only the one branch implied by this measured design."""
        else:
            applicable_quantities = []
            """Withheld every quantity when no report branch matched the design."""
            if not missing_checks:
                raise ValueError(
                    f"ANALYSIS_SUPPORTED_QUANTITY_SET_UNMATCHED:{analysis}"
                )

    return {
        "schema_version": 1,
        "asterism_version": manifest["version"],
        "analysis": analysis,
        "release_ready": not missing_checks,
        "missing_checks": missing_checks,
        "supported_quantities": applicable_quantities,
        "supported_quantity_inventory": list(quantities),
        "known_limitations": known_limitations,
        "design": dict(design),
    }


def run_analysis(
    analysis: str,
    design: Mapping[str, Any],
    fit: Callable[[], Mapping[str, Any]],
    *,
    model: Mapping[str, Any],
    provenance: Mapping[str, Any],
    subject_order: Sequence[str],
) -> dict[str, Any]:
    """Preflight one analysis and return its serialisable receipt data.

    Args:
        analysis: Stable supported-analysis identifier.
        design: Non-identifying design summary checked before fitting.
        fit: Zero-argument callback that performs the numerical fit only after
            the release-state check succeeds.
        model: Caller-owned model and estimator settings.
        provenance: Caller-owned artifact, dependency, consumer and input
            commitments.
        subject_order: Identifiers in exact fitted row order. Only their
            commitment enters the returned receipt.

    Returns:
        Version-1 receipt data. This function performs no file I/O; the caller
        writes the returned mapping beside its controlled outputs.
    """
    release_state: dict[str, Any] = _release_state(analysis, design)
    """Checked release evidence and design support before touching outcomes."""
    receipt_provenance: dict[str, Any] = dict(provenance)
    """Copied caller-owned provenance so the receipt cannot mutate its input."""
    provenance_lengths: dict[str, int] = {
        "wheel_sha256": 64,
        "dependency_lock_sha256": 64,
        "consumer_commit": 40,
    }
    """Defined the exact artifact, dependency and consumer identity fields."""
    for field, length in provenance_lengths.items():
        value: Any = receipt_provenance.get(field)
        """Read one caller-owned hexadecimal commitment."""
        if (
            not isinstance(value, str)
            or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
        ):
            raise ValueError(f"ANALYSIS_PROVENANCE_INVALID:{field}")
    """Required complete lowercase commitments before a fit could be attempted."""
    raw_commitments: Any = receipt_provenance.get("input_commitments", {})
    """Read any caller-owned values-free input commitments."""
    if not isinstance(raw_commitments, Mapping):
        raise ValueError("ANALYSIS_INPUT_COMMITMENTS_INVALID")
    input_commitments: dict[str, Any] = dict(raw_commitments)
    """Copied the nested commitment mapping before adding subject order."""
    if any(
        not isinstance(label, str)
        or not label
        or not isinstance(commitment, str)
        or re.fullmatch(r"[0-9a-f]{64}", commitment) is None
        for label, commitment in input_commitments.items()
    ):
        raise ValueError("ANALYSIS_INPUT_COMMITMENTS_INVALID")
    """Required values-free named SHA-256 commitments, not raw input descriptions."""
    order_sha256: str = subject_order_commitment(subject_order)
    """Committed to exact fitted order without retaining identifiers."""
    existing_order: Any = input_commitments.get("subject_order_sha256")
    """Read a caller-supplied commitment when one was already present."""
    if existing_order is not None and existing_order != order_sha256:
        raise ValueError("ANALYSIS_SUBJECT_ORDER_COMMITMENT_MISMATCH")
    input_commitments["subject_order_sha256"] = order_sha256
    """Added the verified fitted-row commitment to caller-owned input identity."""
    receipt_provenance["input_commitments"] = input_commitments
    """Bound the receipt provenance to the augmented input commitments."""
    build: dict[str, Any] = build_identity()
    """Identified the immutable source and release state used by this run."""
    if not release_state["release_ready"]:
        return {
            "schema_version": 1,
            "asterism_version": release_state["asterism_version"],
            "analysis": analysis,
            "outcome": "refused",
            "build": build,
            "release_state": release_state,
            "model": dict(model),
            "provenance": receipt_provenance,
            "fit_record": None,
            "refusal": {
                "code": "ANALYSIS_PREFLIGHT_FAILED",
                "message": "release or design checks required before fitting",
                "missing_checks": release_state["missing_checks"],
            },
        }

    if build["source_dirty"] is True or build["release"] is not True:
        return {
            "schema_version": 1,
            "asterism_version": release_state["asterism_version"],
            "analysis": analysis,
            "outcome": "refused",
            "build": build,
            "release_state": release_state,
            "model": dict(model),
            "provenance": receipt_provenance,
            "fit_record": None,
            "refusal": {
                "code": "ANALYSIS_BUILD_NOT_FIXED_RELEASE",
                "message": "a supported analysis needs a clean release build",
            },
        }

    try:
        raw_record: Mapping[str, Any] = fit()
        """Ran the numerical callback after release and design checks passed."""
    except ValueError as error:
        message: str = str(error)
        """Read the estimator's documented stable error code and detail."""
        code: str = message.partition(":")[0]
        """Separated the stable code from any non-identifying diagnostic detail."""
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", code) is None or not code.startswith(
            STABLE_ESTIMATOR_CODE_PREFIXES
        ):
            raise
        return {
            "schema_version": 1,
            "asterism_version": release_state["asterism_version"],
            "analysis": analysis,
            "outcome": "refused",
            "build": build,
            "release_state": release_state,
            "model": dict(model),
            "provenance": receipt_provenance,
            "fit_record": None,
            "refusal": {"code": code, "message": message},
        }
    """Normalised only Asterism's stable ValueError vocabulary as refusal."""
    if not isinstance(raw_record, Mapping):
        raise TypeError("ANALYSIS_FIT_RECORD_NOT_MAPPING")
    fit_record: dict[str, Any] = dict(raw_record)
    """Copied the fit record before adding immutable run identity."""
    existing_build: Any = fit_record.get("build")
    """Read identity already attached by an estimator returning its own record."""
    if existing_build is not None and existing_build != build:
        raise ValueError("ANALYSIS_FIT_BUILD_IDENTITY_MISMATCH")
    existing_order: Any = fit_record.get("subject_order_sha256")
    """Read an order commitment already bound when the model was prepared."""
    if existing_order is not None and existing_order != order_sha256:
        raise ValueError("ANALYSIS_FIT_SUBJECT_ORDER_MISMATCH")
    fit_record["build"] = build
    """Attached the verified immutable producer identity to the fit record."""
    fit_record["subject_order_sha256"] = order_sha256
    """Bound the returned fit record to the exact fitted order."""

    failures: list[dict[str, Any]] = []
    """Accumulated every named check the returned fit record did not meet."""
    if fit_record.get("converged") is not True:
        failures.append({"code": "FIT_NOT_CONVERGED", "field": "converged"})

    for quantity in release_state["supported_quantities"]:
        if fit_record.get(quantity) is None:
            failures.append({"code": "QUANTITY_MISSING", "field": quantity})
    """Required every quantity this supported analysis declares."""

    pending: list[tuple[str, Any]] = [("fit_record", fit_record)]
    """Seeded a recursive inspection of serialisable numerical fields."""
    while pending:
        path, value = pending.pop()
        """Selected the next nested value and its diagnostic path."""
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path: str = f"{path}.{key}"
                """Extended the diagnostic path through one mapping field."""
                if key == "profile_failures" and child != 0:
                    failures.append(
                        {"code": "PROFILE_EVALUATION_FAILED", "field": child_path}
                    )
                pending.append((child_path, child))
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                pending.append((f"{path}[{index}]", child))
        elif isinstance(value, float) and not math.isfinite(value):
            failures.append({"code": "FIT_VALUE_NOT_FINITE", "field": path})
    """Recorded non-finite values and every failed profile evaluation."""

    codes: set[str] = {failure["code"] for failure in failures}
    """Collected the distinct codes so each fact can be stated separately."""
    return {
        "schema_version": 1,
        "asterism_version": release_state["asterism_version"],
        "analysis": analysis,
        "outcome": "fitted",
        "build": build,
        "release_state": release_state,
        "model": dict(model),
        "provenance": receipt_provenance,
        "fit_record": fit_record,
        "facts": {
            "converged": "FIT_NOT_CONVERGED" not in codes,
            "all_quantities_present": "QUANTITY_MISSING" not in codes,
            "all_values_finite": "FIT_VALUE_NOT_FINITE" not in codes,
            "all_profiles_evaluated": "PROFILE_EVALUATION_FAILED" not in codes,
        },
        "failures": failures,
        "refusal": None,
    }
