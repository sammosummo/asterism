"""Measure wall time and peak RSS for every supported public analysis route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import asterism
import numpy as np

if __package__:
    from . import synthetic_analysis_receipts as route_helper
else:
    import synthetic_analysis_receipts as route_helper
"""Imported the reviewed public-route callbacks in module and script execution."""

ARTIFACT_ID: str = "asterism-target-resources-0.1"
"""Named the versioned resource-evidence protocol independently of one run."""

CALIBRATION_PLAN_ID: str = "asterism-target-resource-plan-0.1"
"""Named the values-free target-profile plan used before budgets are chosen."""

GENERATOR_ID: str = "asterism-participant-free-family-resource-v1"
"""Named the deterministic values-free target-input construction."""

SHA256_PATTERN: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")
"""Recognised lowercase complete SHA-256 commitments."""

PUBLIC_ENTRY_POINTS: dict[str, list[str]] = {
    "one_trait_gaussian_heritability": [
        "asterism.prepare",
        "asterism.PreparedModel.fit",
    ],
    "several_covariance_components": [
        "asterism.ComponentModel.fit",
        "asterism.ComponentModel.interval",
    ],
    "bivariate_genetic_correlation": [
        "asterism.BivariateModel.fit",
        "asterism.BivariateModel.interval",
        "asterism.BivariateModel.test",
    ],
    "continuous_gene_by_environment": [
        "asterism.GxeModel.fit",
        "asterism.GxeModel.interval",
        "asterism.GxeModel.test",
    ],
    "discrete_gene_by_environment": [
        "asterism.DiscreteGxeModel.fit",
        "asterism.DiscreteGxeModel.correlation_interval",
        "asterism.DiscreteGxeModel.test",
    ],
    "binary_liability_heritability": [
        "asterism.LiabilityModel.fit",
        "asterism.LiabilityModel.interval",
        "asterism.LiabilityModel.test",
    ],
    "one_trait_tobit_audiogram": [
        "asterism.tobit_fit",
        "asterism.tobit_interval",
        "asterism.tobit_test",
    ],
    "mixed_binary_censored_genetic_correlation": [
        "asterism.mixed_bivariate_fit",
        "asterism.mixed_bivariate_interval",
        "asterism.mixed_bivariate_test",
    ],
    "spatial_component_presence": [
        "asterism.SpatialModel.fit",
        "asterism.SpatialModel.bootstrap",
    ],
}
"""Fixed the exact documented Python seams exercised by each resource route."""

INPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "one_trait_gaussian_heritability": ("design", "relationship", "response"),
    "several_covariance_components": ("design", "relationship", "response"),
    "bivariate_genetic_correlation": (
        "relationship",
        "response",
        "second_response",
    ),
    "continuous_gene_by_environment": (
        "design",
        "environment",
        "relationship",
        "response",
    ),
    "discrete_gene_by_environment": (
        "design",
        "discrete_environment",
        "relationship",
        "response",
    ),
    "binary_liability_heritability": (
        "binary_status",
        "design",
        "relationship",
    ),
    "one_trait_tobit_audiogram": (
        "censored_values",
        "censoring",
        "design",
        "limits",
        "relationship",
    ),
    "mixed_binary_censored_genetic_correlation": (
        "binary_status",
        "censored_values",
        "censoring",
        "design",
        "limits",
        "relationship",
    ),
    "spatial_component_presence": (
        "design",
        "distance",
        "relationship",
        "response",
    ),
}
"""Restricted each input commitment to arrays actually supplied to that route."""


class TargetResourceError(RuntimeError):
    """A stable refusal to manufacture incomplete resource evidence."""


def canonical_json_sha256(value: object) -> str:
    """Hash one parsed commitment independently of JSON formatting.

    Args:
        value: Strict JSON-compatible value to identify.

    Returns:
        Lowercase SHA-256 of canonical compact JSON bytes.
    """
    payload: bytes = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    """Rendered a single stable representation before hashing."""

    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the exact lowercase SHA-256 of one present file.

    Args:
        path: File whose bytes form part of release evidence.

    Returns:
        Lowercase hexadecimal SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sha256(value: object) -> bool:
    """Return whether a value is a non-placeholder lowercase SHA-256.

    Args:
        value: Candidate commitment parsed from a manifest or artifact.

    Returns:
        True only for a complete digest with more than one distinct nibble.
    """
    return (
        isinstance(value, str)
        and SHA256_PATTERN.fullmatch(value) is not None
        and len(set(value)) > 1
    )


def positive_number(value: object) -> bool:
    """Return whether a JSON value is finite, numerical and strictly positive.

    Args:
        value: Candidate timing or acceptance value.

    Returns:
        True for positive finite integers or floats, excluding booleans.
    """
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def supported_platform() -> str:
    """Return the manifest spelling for this supported operating system.

    Returns:
        ``macos`` or ``linux``.

    Raises:
        TargetResourceError: If the host is outside the 0.1 platform contract.
    """
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    raise TargetResourceError("TARGET_RESOURCE_PLATFORM_UNSUPPORTED")


def peak_rss_bytes() -> int:
    """Return this process's peak resident set size in bytes.

    Returns:
        Positive peak RSS normalised across macOS and Linux.
    """
    raw_peak: int = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """Read the operating system's process-lifetime high-water mark."""

    if supported_platform() == "linux":
        return raw_peak * 1024
    return raw_peak


def target_resource_configuration_errors(
    configuration: object,
    root: Path,
    analysis_ids: list[str],
    *,
    require_configured: bool,
) -> list[str]:
    """Return failures in the target-resource manifest contract.

    Args:
        configuration: Parsed ``target_resource_budgets`` manifest table.
        root: Release checkout containing both committed programs.
        analysis_ids: Complete ordered supported-analysis inventory.
        require_configured: Whether empty pre-measurement state must block.

    Returns:
        Missing program, inventory, host, input and acceptance errors.
    """
    if not isinstance(configuration, dict):
        return ["release.toml: target resource budgets are missing"]
    errors: list[str] = []
    """Collected every independent configuration defect for one repair cycle."""

    configured: bool = configuration.get("configured") is True
    """Read the deliberate opt-in separately from the presence of runner code."""

    if require_configured and not configured:
        errors.append("release.toml: target resource budgets are unmeasured")
    for field, label in (
        ("runner", "budget runner"),
        ("route_helper", "public-route helper"),
    ):
        relative: object = configuration.get(field)
        """Read one committed repository-relative Python program."""

        if not isinstance(relative, str) or not relative:
            errors.append(f"release.toml: target resource {label} is not configured")
            continue
        candidate: Path = root / relative
        """Resolved the program beneath the release checkout."""

        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append(f"release.toml: target resource {label} escapes the checkout")
        elif candidate.suffix != ".py" or not candidate.is_file():
            errors.append(
                f"release.toml: target resource {label} does not exist: {relative}"
            )
    """Required immutable source bytes for both orchestration and public routes."""

    configured_ids: object = configuration.get("analysis_ids")
    """Read the exact ordered route inventory promised by the runner."""

    if configured_ids != analysis_ids:
        errors.append(
            "release.toml: target resource analysis_ids do not cover every analysis"
        )
    measurements: object = configuration.get("measurements")
    """Read target profiles and their prewritten wall and memory ceilings."""

    if not isinstance(measurements, list) or not all(
        isinstance(measurement, dict) for measurement in measurements
    ):
        errors.append("release.toml: target resource measurements must be tables")
        return errors
    if not configured and not require_configured:
        return errors
    measured_ids: list[object] = [
        measurement.get("analysis_id") for measurement in measurements
    ]
    """Retained ownership with multiplicity so duplicates cannot masquerade as coverage."""

    if measured_ids != analysis_ids:
        errors.append(
            "release.toml: target resource measurements do not cover every analysis"
        )
    host_contracts: set[tuple[object, object, object, object]] = set()
    """Collected intended-host coordinates represented by this aggregate artifact."""

    for index, measurement in enumerate(measurements):
        analysis_id: str = str(measurement.get("analysis_id", f"index-{index}"))
        """Named the owning route for precise configuration diagnostics."""

        host_id: object = measurement.get("host_id")
        """Read the stable human-reviewed intended-host label."""

        host_platform: object = measurement.get("platform")
        """Read the supported operating-system family for this route."""

        architecture: object = measurement.get("architecture")
        """Read the intended host's machine architecture."""

        python_version: object = measurement.get("python_version")
        """Read the intended standard-CPython minor version."""

        host_contracts.add((host_id, host_platform, architecture, python_version))
        if not isinstance(host_id, str) or not host_id.strip():
            errors.append(f"release.toml: {analysis_id} host_id is missing")
        if host_platform not in {"macos", "linux"}:
            errors.append(f"release.toml: {analysis_id} platform is unsupported")
        if not isinstance(architecture, str) or not architecture.strip():
            errors.append(f"release.toml: {analysis_id} architecture is missing")
        if python_version not in {"3.13", "3.14"}:
            errors.append(f"release.toml: {analysis_id} Python version is unsupported")
        profile_id: object = measurement.get("profile_id")
        """Read the reviewed deterministic target-profile identifier."""

        if (
            not isinstance(profile_id, str)
            or not profile_id.strip()
            or "smoke" in profile_id.lower()
        ):
            errors.append(f"release.toml: {analysis_id} profile_id is not target-sized")
        sample_size: object = measurement.get("sample_size")
        """Read the generated target roster size."""

        largest_family: object = measurement.get("largest_family")
        """Read the exact largest-family load represented by the profile."""

        seed: object = measurement.get("seed")
        """Read the deterministic participant-free generator seed."""

        if (
            not isinstance(sample_size, int)
            or isinstance(sample_size, bool)
            or sample_size < 2
        ):
            errors.append(f"release.toml: {analysis_id} sample_size is invalid")
        if (
            not isinstance(largest_family, int)
            or isinstance(largest_family, bool)
            or largest_family < 1
            or not isinstance(sample_size, int)
            or largest_family > sample_size
        ):
            errors.append(f"release.toml: {analysis_id} largest_family is invalid")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            errors.append(f"release.toml: {analysis_id} seed is invalid")
        if not valid_sha256(measurement.get("input_sha256")):
            errors.append(f"release.toml: {analysis_id} input SHA-256 is invalid")
        if not positive_number(measurement.get("max_wall_seconds")):
            errors.append(
                f"release.toml: {analysis_id} max_wall_seconds must be finite and positive"
            )
        maximum_rss: object = measurement.get("max_peak_rss_bytes")
        """Read the exact prewritten resident-memory ceiling in portable bytes."""

        if (
            not isinstance(maximum_rss, int)
            or isinstance(maximum_rss, bool)
            or maximum_rss <= 0
        ):
            errors.append(
                f"release.toml: {analysis_id} max_peak_rss_bytes must be positive"
            )
    """Validated target facts and acceptances without inferring absent values."""

    if len(host_contracts) != 1:
        errors.append(
            "release.toml: target resource measurements require one intended host per artifact"
        )
    timeout_seconds: object = configuration.get("timeout_seconds")
    """Read the outer orchestration bound separately from scientific acceptances."""

    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or not 1 <= timeout_seconds <= 604_800
    ):
        errors.append("release.toml: target resource runner timeout is invalid")
    return errors


def calibration_profile_errors(profiles: object, analysis_ids: list[str]) -> list[str]:
    """Return failures in a pre-acceptance target calibration profile set.

    Args:
        profiles: Parsed values-free generated-input profile list.
        analysis_ids: Complete ordered supported-analysis inventory.

    Returns:
        Inventory, dimension, seed and forbidden-acceptance errors.
    """
    if not isinstance(profiles, list) or not all(
        isinstance(profile, dict) for profile in profiles
    ):
        return ["target calibration profiles must be a table list"]
    errors: list[str] = []
    """Collected every independent plan defect before any target input allocation."""

    profile_ids: list[object] = [profile.get("analysis_id") for profile in profiles]
    """Retained route order and multiplicity for exact inventory comparison."""

    if profile_ids != analysis_ids:
        errors.append("target calibration profiles do not cover every analysis")
    required_fields: set[str] = {
        "analysis_id",
        "profile_id",
        "sample_size",
        "largest_family",
        "seed",
    }
    """Limited calibration plans to generated-input facts, never acceptance limits."""

    for index, profile in enumerate(profiles):
        analysis_id: str = str(profile.get("analysis_id", f"index-{index}"))
        """Named one target profile for precise plan diagnostics."""

        if set(profile) != required_fields:
            errors.append(
                f"target calibration {analysis_id} fields must contain only input facts"
            )
        profile_id: object = profile.get("profile_id")
        """Read the reviewed target-profile identifier."""

        if (
            not isinstance(profile_id, str)
            or not profile_id.strip()
            or "smoke" in profile_id.lower()
        ):
            errors.append(
                f"target calibration {analysis_id} profile_id is not target-sized"
            )
        sample_size: object = profile.get("sample_size")
        """Read the target generated roster size."""

        if (
            not isinstance(sample_size, int)
            or isinstance(sample_size, bool)
            or sample_size < 2
        ):
            errors.append(f"target calibration {analysis_id} sample_size is invalid")
        largest_family: object = profile.get("largest_family")
        """Read the exact maximum generated family size."""

        if (
            not isinstance(largest_family, int)
            or isinstance(largest_family, bool)
            or largest_family < 1
            or not isinstance(sample_size, int)
            or largest_family > sample_size
        ):
            errors.append(f"target calibration {analysis_id} largest_family is invalid")
        seed: object = profile.get("seed")
        """Read the deterministic participant-free generator seed."""

        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            errors.append(f"target calibration {analysis_id} seed is invalid")
    """Required complete target input facts without circular resource acceptances."""

    return errors


def target_resource_record_errors(
    record: object,
    *,
    manifest: dict[str, Any],
    root: Path,
    expected_build: dict[str, Any],
    wheel_sha256s: set[str],
) -> list[str]:
    """Return independent failures in one target-resource artifact.

    Args:
        record: Parsed values-free measurement artifact.
        manifest: Parsed authoritative release contract.
        root: Checkout holding the exact runner and route-helper bytes.
        expected_build: Public immutable identity of the installed wheel.
        wheel_sha256s: Exact saved-wheel digest set selected for release.

    Returns:
        Structural, stale-input, execution and budget-decision failures.
    """
    if not isinstance(record, dict):
        return ["target resource record must be an object"]
    errors: list[str] = []
    """Collected all identity and decision defects without trusting ``passed``."""

    configuration: object = manifest.get("target_resource_budgets")
    """Read the target profiles and acceptances bound by this artifact."""

    analysis_ids: list[str] = [
        str(analysis.get("id", ""))
        for analysis in manifest.get("analyses", [])
        if isinstance(analysis, dict)
    ]
    """Read the exact supported route order from the release contract."""

    errors.extend(
        target_resource_configuration_errors(
            configuration,
            root,
            analysis_ids,
            require_configured=True,
        )
    )
    if not isinstance(configuration, dict):
        return errors
    expected_identity: dict[str, object] = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "qualification": "release_target",
        "manifest_sha256": canonical_json_sha256(manifest),
        "analysis_ids": analysis_ids,
        "failures": [],
        "passed": True,
    }
    """Named every aggregate release decision that cannot be inferred from a boolean."""

    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            errors.append(f"target resource {field} does not match release")
    """Bound the artifact to this final parsed contract and complete route set."""

    for field, configuration_field in (
        ("runner_sha256", "runner"),
        ("route_helper_sha256", "route_helper"),
    ):
        relative: object = configuration.get(configuration_field)
        """Read the configured source file whose exact bytes must still match."""

        if not isinstance(relative, str):
            continue
        program_path: Path = root / relative
        """Resolved the committed program after configuration validation."""

        if program_path.is_file() and record.get(field) != file_sha256(program_path):
            label: str = "runner" if field == "runner_sha256" else "route helper"
            """Selected a concise stable diagnostic label."""

            errors.append(f"target resource {label} SHA-256 does not match")
    """Refused results produced by different orchestration or public-route code."""

    if record.get("build") != expected_build:
        errors.append("target resource build identity does not match release")
    wheel: object = record.get("wheel")
    """Read the sole runtime artifact actually imported for these measurements."""

    if not isinstance(wheel, dict):
        errors.append("target resource record has no saved runtime wheel")
    else:
        if not isinstance(wheel.get("name"), str) or not wheel.get("name"):
            errors.append("target resource runtime wheel name is missing")
        if wheel.get("sha256") not in wheel_sha256s:
            errors.append("target resource wheel SHA-256 does not match saved wheels")
    expected_measurements: object = configuration.get("measurements")
    """Read the ordered target profiles and acceptances already checked above."""

    actual_measurements: object = record.get("measurements")
    """Read the actual per-route resource observations to recompute decisions."""

    if not isinstance(expected_measurements, list) or not isinstance(
        actual_measurements, list
    ):
        errors.append("target resource measurement inventory is incomplete")
        return errors
    actual_ids: list[object] = [
        measurement.get("analysis_id")
        for measurement in actual_measurements
        if isinstance(measurement, dict)
    ]
    """Collected measurement ownership with multiplicity and preserved order."""

    if actual_ids != analysis_ids or len(actual_measurements) != len(analysis_ids):
        errors.append("target resource measurement inventory is incomplete")
    expected_by_id: dict[str, dict[str, Any]] = {
        str(measurement.get("analysis_id", "")): measurement
        for measurement in expected_measurements
        if isinstance(measurement, dict)
    }
    """Indexed already validated acceptances for field-by-field comparison."""

    for index, actual in enumerate(actual_measurements):
        if not isinstance(actual, dict):
            errors.append(f"target resource measurement {index} is not an object")
            continue
        analysis_id: str = str(actual.get("analysis_id", f"index-{index}"))
        """Named the route represented by this actual observation."""

        expected: dict[str, Any] | None = expected_by_id.get(analysis_id)
        """Selected the prewritten profile and acceptance for independent comparison."""

        if expected is None:
            errors.append(f"target resource {analysis_id} is not configured")
            continue
        for field in (
            "host_id",
            "profile_id",
            "sample_size",
            "largest_family",
            "seed",
            "max_wall_seconds",
            "max_peak_rss_bytes",
        ):
            if actual.get(field) != expected.get(field):
                errors.append(f"target resource {analysis_id} {field} does not match")
        """Prevented the measured design or acceptance from drifting at execution."""

        if actual.get("input_sha256") != expected.get("input_sha256"):
            errors.append(
                f"target resource {analysis_id} input commitment does not match"
            )
        if actual.get("public_entry_points") != PUBLIC_ENTRY_POINTS.get(analysis_id):
            errors.append(
                f"target resource {analysis_id} public entry points do not match"
            )
        if actual.get("public_api") is not True:
            errors.append(f"target resource {analysis_id} public_api must be true")
        if actual.get("target_sized") is not True:
            errors.append(f"target resource {analysis_id} target_sized must be true")
        if (
            actual.get("passed") is not True
            or actual.get("status") != "completed"
            or actual.get("timed_out") is not False
        ):
            errors.append(f"target resource {analysis_id} did not complete")
        wall_time: object = actual.get("wall_time_seconds")
        """Read the observed process wall time before comparing its ceiling."""

        if not positive_number(wall_time):
            errors.append(f"target resource {analysis_id} wall time is invalid")
        elif positive_number(expected.get("max_wall_seconds")) and float(
            wall_time
        ) > float(expected["max_wall_seconds"]):
            errors.append(f"target resource {analysis_id} exceeded max_wall_seconds")
        peak_rss: object = actual.get("peak_rss_bytes")
        """Read the normalised process high-water mark before comparing its ceiling."""

        if not isinstance(peak_rss, int) or isinstance(peak_rss, bool) or peak_rss <= 0:
            errors.append(f"target resource {analysis_id} peak RSS is invalid")
        elif isinstance(expected.get("max_peak_rss_bytes"), int) and peak_rss > int(
            expected["max_peak_rss_bytes"]
        ):
            errors.append(f"target resource {analysis_id} exceeded max_peak_rss_bytes")
    """Recomputed each completion and both resource decisions from actual values."""

    host: object = record.get("host")
    """Read the actual single host on which every route was measured."""

    first_expected: object = expected_measurements[0] if expected_measurements else None
    """Selected the common intended-host contract enforced during configuration."""

    if not isinstance(host, dict) or not isinstance(first_expected, dict):
        errors.append("target resource host identity is incomplete")
    else:
        expected_host: dict[str, object] = {
            "id": first_expected.get("host_id"),
            "platform": first_expected.get("platform"),
            "architecture": first_expected.get("architecture"),
            "python_version": first_expected.get("python_version"),
            "python_implementation": "CPython",
            "peak_rss_source": "getrusage_ru_maxrss",
            "peak_rss_bytes_normalisation": (
                "native_bytes"
                if first_expected.get("platform") == "macos"
                else "kibibytes_x_1024"
            ),
        }
        """Rendered the exact portable host and RSS-normalisation identity."""

        if host != expected_host:
            errors.append("target resource host identity does not match release")
    return errors


def family_sizes(sample_size: int, largest_family: int) -> list[int]:
    """Partition a target roster into one large family and small families.

    Args:
        sample_size: Total participant-free synthetic row count.
        largest_family: Exact largest-family stress requested by the profile.

    Returns:
        Positive family sizes summing to ``sample_size``.
    """
    if largest_family == 1:
        return [1] * sample_size
    sizes: list[int] = [largest_family]
    """Placed the reviewed largest-family stress first and exactly once."""

    remaining: int = sample_size - largest_family
    """Tracked synthetic rows still available for small family blocks."""

    while remaining >= 2:
        sizes.append(2)
        remaining -= 2
        """Removed the allocated sibling pair from the remaining row count."""
    """Filled the remainder with sibling pairs to retain repeated structure."""

    if remaining == 1:
        sizes.append(1)
    """Retained an odd final row as an unrelated singleton when required."""

    return sizes


def resource_problem(
    sample_size: int,
    largest_family: int,
    seed: int,
    *,
    include_distance: bool,
) -> dict[str, Any]:
    """Create deterministic participant-free inputs for target load measurement.

    Args:
        sample_size: Total generated roster size.
        largest_family: Exact maximum exchangeable family block size.
        seed: Fixed NumPy random seed from the manifest profile.
        include_distance: Whether to construct the spatial route's dense matrix.

    Returns:
        Synthetic matrices and traits required by the nine public routes.
    """
    sizes: list[int] = family_sizes(sample_size, largest_family)
    """Constructed the exact target family-size envelope."""

    relationship: np.ndarray = np.eye(sample_size, dtype=np.float64)
    """Started a positive-definite twice-kinship matrix at unit diagonal."""

    family_index: np.ndarray = np.empty(sample_size, dtype=np.int64)
    """Allocated non-identifying block labels for deterministic signal generation."""

    start: int = 0
    """Located the beginning of the next contiguous synthetic family block."""

    for index, size in enumerate(sizes):
        stop: int = start + size
        """Located the exclusive end of this family block."""

        relationship[start:stop, start:stop] = 0.5
        """Filled this synthetic family's covariance block with sibling relatedness."""

        np.fill_diagonal(relationship[start:stop, start:stop], 1.0)
        """Restored unit diagonal after filling the exchangeable block."""

        family_index[start:stop] = index
        """Assigned every row in the block to its generated family effect."""

        start = stop
        """Advanced to the next contiguous family block."""
    """Added exchangeable full-sibling covariance within every generated family."""

    generator: np.random.Generator = np.random.default_rng(seed)
    """Created the sole deterministic random stream for this target profile."""

    shared: np.ndarray = generator.standard_normal(len(sizes))[family_index]
    """Generated one Gaussian effect shared within each synthetic family."""

    own: np.ndarray = generator.standard_normal(sample_size)
    """Generated an independent Gaussian contribution for every synthetic row."""

    genetic: np.ndarray = np.sqrt(0.5) * shared + np.sqrt(0.5) * own
    """Generated a Gaussian signal whose covariance matches the relationship blocks."""

    residual: np.ndarray = generator.standard_normal(sample_size)
    """Generated independent environmental noise for the first trait."""

    environment: np.ndarray = np.linspace(-1.0, 1.0, sample_size, dtype=np.float64)
    """Placed the generated rows on a fixed continuous exposure grid."""

    response: np.ndarray = 0.75 * genetic + 0.55 * residual + 0.2 * environment
    """Created a continuous target trait with genetic signal and measured trend."""

    second_shared: np.ndarray = generator.standard_normal(len(sizes))[family_index]
    """Generated an independent family-level signal for the second trait."""

    second_own: np.ndarray = generator.standard_normal(sample_size)
    """Generated independent row-level signal for the second trait."""

    independent_genetic: np.ndarray = (
        np.sqrt(0.5) * second_shared + np.sqrt(0.5) * second_own
    )
    """Matched the second latent genetic signal to the same relationship matrix."""

    second_genetic: np.ndarray = 0.45 * genetic + np.sqrt(1.0 - 0.45**2) * (
        independent_genetic
    )
    """Imposed a fixed interior genetic correlation between the two traits."""

    second_response: np.ndarray = (
        0.70 * second_genetic
        + 0.30 * residual
        + 0.55 * generator.standard_normal(sample_size)
    )
    """Created a second continuous trait with fixed non-boundary genetic correlation."""

    design: np.ndarray = np.ones((sample_size, 1), dtype=np.float64)
    """Created the explicit intercept used by all one-trait target models."""

    discrete_environment: np.ndarray = (
        np.arange(sample_size, dtype=np.int64) % 2
    ).astype(np.float64)
    """Alternated rows across a balanced binary environment."""

    binary_cut: float = float(np.quantile(response, 0.65))
    """Selected a deterministic prevalence threshold from the generated latent trait."""

    binary_status: np.ndarray = (response > binary_cut).astype(np.float64)
    """Created a deterministic nondegenerate binary diagnosis from the latent trait."""

    censoring_limit: float = float(np.quantile(second_response, 0.48))
    """Selected a fixed right-censoring threshold from the second latent trait."""

    censored: np.ndarray = second_response >= censoring_limit
    """Marked generated observations at or beyond the instrument limit."""

    censoring: np.ndarray = censored.astype(np.int64)
    """Encoded right-censoring states through the public integer contract."""

    limits: np.ndarray = np.full(sample_size, censoring_limit, dtype=np.float64)
    """Applied the same generated instrument limit to every row."""

    censored_values: np.ndarray = np.where(censored, np.nan, second_response)
    """Created an approximately 52-percent right-censored continuous trait."""

    if include_distance:
        columns: int = max(2, math.ceil(math.sqrt(sample_size)))
        """Selected a compact deterministic rectangular spatial layout."""

        positions: np.ndarray = np.column_stack(
            [
                np.arange(sample_size, dtype=np.float64) % columns,
                np.arange(sample_size, dtype=np.float64) // columns,
            ]
        )
        """Placed every generated row on the deterministic rectangular grid."""

        differences: np.ndarray = positions[:, None, :] - positions[None, :, :]
        """Constructed all pairwise coordinate differences."""

        distance: np.ndarray = np.sqrt(np.sum(differences**2, axis=2))
        """Constructed the spatial route's complete Euclidean distance matrix."""
    else:
        distance = np.zeros((1, 1), dtype=np.float64)
        """Avoided allocating an unused second dense matrix for non-spatial routes."""

    return {
        "relationship": relationship,
        "response": response,
        "second_response": second_response,
        "design": design,
        "environment": environment,
        "discrete_environment": discrete_environment,
        "distance": distance,
        "binary_status": binary_status,
        "censoring": censoring,
        "limits": limits,
        "censored_values": censored_values,
    }


def resource_input_sha256(
    problem: dict[str, Any],
    analysis_id: str,
    profile: dict[str, Any],
    subject_order_sha256: str,
) -> str:
    """Commit exact generated arrays and their reviewed profile without retaining values.

    Args:
        problem: Complete participant-free generated problem.
        analysis_id: Public route selecting the arrays actually consumed.
        profile: Manifest profile that deterministically generated the arrays.
        subject_order_sha256: Commitment to the ephemeral generated row order.

    Returns:
        Canonical SHA-256 binding generator, profile, arrays and row order.
    """
    selected: dict[str, Any] = {
        name: problem[name] for name in INPUT_FIELDS[analysis_id]
    }
    """Selected exactly the arrays supplied through this public route."""

    array_sha256: str = route_helper.fixture_sha256(selected)
    """Committed NumPy types, shapes and exact contiguous bytes."""

    commitment: dict[str, object] = {
        "generator_id": GENERATOR_ID,
        "analysis_id": analysis_id,
        "profile_id": profile["profile_id"],
        "sample_size": profile["sample_size"],
        "largest_family": profile["largest_family"],
        "seed": profile["seed"],
        "array_sha256": array_sha256,
        "subject_order_sha256": subject_order_sha256,
    }
    """Bound exact values to the human-reviewable target profile facts."""

    return canonical_json_sha256(commitment)


def fit_bivariate_resource(
    problem: dict[str, Any], subject_order_sha256: str
) -> dict[str, Any]:
    """Exercise the bivariate route for an arbitrary target roster size.

    Args:
        problem: Participant-free target matrices and traits.
        subject_order_sha256: Commitment to their exact shared row order.

    Returns:
        Complete public fit, interval and test record.
    """
    sample_size: int = int(problem["relationship"].shape[0])
    """Read the configured target size from the generated relationship matrix."""

    observed: np.ndarray = np.ones((sample_size, 2), dtype=bool)
    """Marked both generated traits observed for every target row."""

    pair_design: np.ndarray = np.tile(np.eye(2), (sample_size, 1))
    """Gave each trait its own intercept in person-within-trait order."""

    response: np.ndarray = np.column_stack(
        [problem["response"], problem["second_response"]]
    ).ravel()
    """Constructed complete paired-trait inputs in person-within-trait order."""

    model: asterism.BivariateModel = asterism.BivariateModel(
        problem["relationship"],
        observed,
        pair_design,
        subject_order_sha256=subject_order_sha256,
    )
    """Built the documented public bivariate target model."""

    record: dict[str, Any] = model.fit(response)
    """Fitted the target-sized bivariate covariance model."""

    record["interval"] = model.interval(response, "rho_g")
    """Attached the public profile interval for genetic correlation."""

    record["test"] = model.test(response, "rho_g", 0.0)
    """Exercised the complete reportable correlation route without retaining results."""

    return record


def public_route(
    analysis_id: str,
    problem: dict[str, Any],
    subject_order_sha256: str,
) -> route_helper.ReceiptJob:
    """Return one reviewed public callback for an arbitrary generated profile.

    Args:
        analysis_id: Stable route selected by the manifest.
        problem: Participant-free generated matrices and traits.
        subject_order_sha256: Commitment to their exact shared row order.

    Returns:
        Lazy public analysis callback and non-identifying design metadata.

    Raises:
        TargetResourceError: If the route inventory has drifted or is unknown.
    """
    sample_size: int = int(problem["relationship"].shape[0])
    """Read the actual generated roster size for helper metadata."""

    original_pairs: int = route_helper.PAIRS
    """Preserved the small helper's module-level example size during route creation."""

    route_helper.PAIRS = sample_size // 2
    """Adjusted only lazy helper metadata to the arbitrary resource-profile size."""

    try:
        jobs: list[route_helper.ReceiptJob] = route_helper.receipt_jobs(
            problem, subject_order_sha256
        )
        """Constructed every reviewed callback lazily without fitting other routes."""
    finally:
        route_helper.PAIRS = original_pairs
        """Returned the standard receipt helper to its fixed small-example size."""
    """Restored shared module state before any numerical work began."""

    matches: list[route_helper.ReceiptJob] = [
        job for job in jobs if job.analysis_id == analysis_id
    ]
    """Selected the sole callback matching this manifest route."""

    if len(matches) != 1 or set(PUBLIC_ENTRY_POINTS) != {
        job.analysis_id for job in jobs
    }:
        raise TargetResourceError("TARGET_RESOURCE_ROUTE_INVENTORY_MISMATCH")
    job: route_helper.ReceiptJob = matches[0]
    """Read the reviewed callback after exact inventory validation."""

    if analysis_id == "bivariate_genetic_correlation":
        from functools import partial

        job = route_helper.ReceiptJob(
            analysis_id=job.analysis_id,
            design=job.design,
            model=job.model,
            fit=partial(fit_bivariate_resource, problem, subject_order_sha256),
        )
        """Removed the small example's fixed pair count for arbitrary target sizes."""

    return job


def worker_measurement(analysis_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Measure one generated public route in the current isolated process.

    Args:
        analysis_id: Supported route selected by the parent process.
        profile: Reviewed deterministic target or bounded-smoke input facts.

    Returns:
        Values-free completion, wall-time, peak-RSS and input commitments.
    """
    started: float = time.perf_counter()
    """Started wall-time measurement immediately before target input generation."""

    problem: dict[str, Any] = resource_problem(
        int(profile["sample_size"]),
        int(profile["largest_family"]),
        int(profile["seed"]),
        include_distance=analysis_id == "spatial_component_presence",
    )
    """Generated only participant-free inputs, including dense distance when consumed."""

    subject_order: list[str] = [
        f"resource-row-{index:06d}" for index in range(int(profile["sample_size"]))
    ]
    """Created ephemeral row labels solely to bind the fitted order."""

    order_sha256: str = asterism.subject_order_commitment(subject_order)
    """Discarded labels after producing their public SHA-256 commitment."""

    input_sha256: str = resource_input_sha256(
        problem, analysis_id, profile, order_sha256
    )
    """Committed every exact generated value supplied to the selected route."""

    status: str = "completed"
    """Defaulted the public-route outcome to normal completion."""

    failure_code: str | None = None
    """Withheld a failure identity until the route actually raised."""

    try:
        job: route_helper.ReceiptJob = public_route(analysis_id, problem, order_sha256)
        """Selected the same reviewed public callback used by synthetic receipts."""

        result: object = job.fit()
        """Exercised the complete fit, interval and test route through public APIs."""

        if not isinstance(result, dict):
            raise TargetResourceError("TARGET_RESOURCE_PUBLIC_RESULT_INVALID")
    except Exception as error:
        status = "failed"
        """Marked every public exception as an attempted route failure."""

        failure_code = f"PUBLIC_ROUTE_{type(error).__name__.upper()}"
        """Recorded failure without retaining generated values or exception text."""

    wall_time_seconds: float = max(
        time.perf_counter() - started, sys.float_info.epsilon
    )
    """Measured the complete input-generation and public-route elapsed time."""

    measurement: dict[str, Any] = {
        "analysis_id": analysis_id,
        "public_entry_points": PUBLIC_ENTRY_POINTS[analysis_id],
        "public_api": True,
        "profile_id": profile["profile_id"],
        "sample_size": profile["sample_size"],
        "largest_family": profile["largest_family"],
        "seed": profile["seed"],
        "input_sha256": input_sha256,
        "subject_order_sha256": order_sha256,
        "wall_time_seconds": wall_time_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "status": status,
        "timed_out": False,
        "failure_code": failure_code,
    }
    """Retained resource evidence and commitments while discarding all fit values."""

    return measurement


def json_bytes(value: object) -> bytes:
    """Serialize one artifact strictly with a stable trailing newline.

    Args:
        value: Strict JSON-compatible artifact value.

    Returns:
        Indented UTF-8 bytes rejecting non-finite numbers.
    """
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def runtime_build_identity() -> tuple[dict[str, Any], Path]:
    """Return the public build identity and installed extension path.

    Returns:
        Immutable public build fields and resolved compiled-module path.

    Raises:
        TargetResourceError: If the compiled extension cannot be located.
    """
    build: dict[str, Any] = dict(asterism.build_identity())
    """Read immutable source, manifest and dependency commitments from the wheel."""

    core: ModuleType = __import__("asterism._core", fromlist=["_core"])
    """Loaded the compiled module that actually serves public numerical calls."""

    extension_path: Path = Path(str(getattr(core, "__file__", ""))).resolve()
    """Resolved the installed extension independently of package metadata."""

    if not extension_path.is_file():
        raise TargetResourceError("TARGET_RESOURCE_EXTENSION_MISSING")
    return build, extension_path


def smoke_profiles(analysis_ids: list[str]) -> list[dict[str, Any]]:
    """Return fixed bounded profiles that can never qualify release.

    Args:
        analysis_ids: Complete supported route inventory.

    Returns:
        One deterministic sibling-pair smoke profile per route.
    """
    return [
        {
            "analysis_id": analysis_id,
            "profile_id": f"{analysis_id}-development-smoke-v1",
            "sample_size": 80,
            "largest_family": 2,
            "seed": 20_260_821 + index,
        }
        for index, analysis_id in enumerate(analysis_ids)
    ]


def write_target_resource_record(
    manifest_path: Path,
    output_path: Path,
    wheel_paths: list[Path],
    qualification: str,
    host_id: str,
    profile_plan_path: Path | None = None,
) -> dict[str, Any]:
    """Run all configured routes in fresh processes and write one artifact.

    Args:
        manifest_path: Exact release manifest embedded in the installed wheel.
        output_path: New JSON file that will hold values-free measurements.
        wheel_paths: Complete saved release artifact set.
        qualification: Release verification, target calibration or bounded smoke.
        host_id: Human-reviewed stable name for the intended runtime host.
        profile_plan_path: Versioned values-free plan for target calibration.

    Returns:
        The complete artifact also written to ``output_path``.

    Raises:
        TargetResourceError: If identity, configuration or output preflight fails.
    """
    root: Path = manifest_path.resolve().parent
    """Located the fixed checkout holding manifest, locks and runner sources."""

    manifest_text: str = manifest_path.read_text(encoding="utf-8")
    """Read the exact authoritative bytes expected inside the installed wheel."""

    manifest: dict[str, Any] = tomllib.loads(manifest_text)
    """Parsed the supported route contract without normalising its source bytes."""

    analysis_ids: list[str] = [
        str(analysis["id"])
        for analysis in manifest.get("analyses", [])
        if isinstance(analysis, dict) and analysis.get("support") == "supported_in_0_1"
    ]
    """Read the exact ordered 0.1 route inventory from the fixed contract."""

    if analysis_ids != list(PUBLIC_ENTRY_POINTS):
        raise TargetResourceError("TARGET_RESOURCE_ANALYSIS_INVENTORY_MISMATCH")
    configuration: object = manifest.get("target_resource_budgets")
    """Read target profiles only for a qualifying run."""

    if qualification == "release_target":
        configuration_errors: list[str] = target_resource_configuration_errors(
            configuration,
            root,
            analysis_ids,
            require_configured=True,
        )
        """Validated every target fact and acceptance before numerical work."""

        if configuration_errors:
            raise TargetResourceError("; ".join(configuration_errors))
        if not isinstance(configuration, dict):
            raise TargetResourceError("TARGET_RESOURCE_CONFIGURATION_MISSING")
        profiles: list[dict[str, Any]] = [
            dict(profile) for profile in configuration["measurements"]
        ]
        """Copied the complete reviewed target profiles in manifest order."""
        profile_plan_sha256: str | None = None
        """Withheld a calibration-plan hash from final release verification."""
    elif qualification == "target_calibration":
        if profile_plan_path is None or not profile_plan_path.is_file():
            raise TargetResourceError("TARGET_RESOURCE_CALIBRATION_PLAN_MISSING")
        try:
            plan: object = json.loads(profile_plan_path.read_text(encoding="utf-8"))
            """Parsed the versioned values-free target profile plan."""
        except json.JSONDecodeError as error:
            raise TargetResourceError(
                "TARGET_RESOURCE_CALIBRATION_PLAN_UNREADABLE"
            ) from error
        if (
            not isinstance(plan, dict)
            or plan.get("schema_version") != 1
            or plan.get("artifact_id") != CALIBRATION_PLAN_ID
            or plan.get("analysis_ids") != analysis_ids
        ):
            raise TargetResourceError("TARGET_RESOURCE_CALIBRATION_PLAN_INVALID")
        raw_profiles: object = plan.get("profiles")
        """Read the complete target-input inventory before choosing acceptances."""

        profile_errors: list[str] = calibration_profile_errors(
            raw_profiles, analysis_ids
        )
        """Validated exact routes and dimensions without requiring resource limits."""

        if profile_errors:
            raise TargetResourceError("; ".join(profile_errors))
        if not isinstance(raw_profiles, list):
            raise TargetResourceError("TARGET_RESOURCE_CALIBRATION_PLAN_INVALID")
        profiles = [dict(profile) for profile in raw_profiles]
        """Copied the reviewed target profiles only after complete validation."""

        profile_plan_sha256 = file_sha256(profile_plan_path)
        """Committed the artifact to the exact reviewed plan bytes."""
    elif qualification == "development_smoke":
        profiles = smoke_profiles(analysis_ids)
        """Selected bounded plumbing probes that cannot satisfy a release gate."""

        profile_plan_sha256 = None
        """Withheld target-plan identity from the fixed small smoke profiles."""
    else:
        raise TargetResourceError("TARGET_RESOURCE_QUALIFICATION_INVALID")
    if output_path.exists():
        raise TargetResourceError("TARGET_RESOURCE_OUTPUT_EXISTS")
    build, extension_path = runtime_build_identity()
    """Read both immutable build metadata and its installed extension location."""

    if canonical_json_sha256(asterism.release_manifest()) != canonical_json_sha256(
        manifest
    ):
        raise TargetResourceError("TARGET_RESOURCE_EMBEDDED_MANIFEST_MISMATCH")
    if (
        build.get("release_manifest_sha256")
        != hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    ):
        raise TargetResourceError("TARGET_RESOURCE_MANIFEST_BYTES_MISMATCH")
    try:
        extension_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise TargetResourceError("TARGET_RESOURCE_MUTABLE_CHECKOUT_IMPORT")
    """Required the installed build to embed this exact contract outside the checkout."""

    if qualification == "release_target" and (
        build.get("release") is not True or build.get("source_dirty") is not False
    ):
        raise TargetResourceError("TARGET_RESOURCE_BUILD_NOT_FIXED_RELEASE")
    runtime_wheel: Path = route_helper.runtime_wheel(wheel_paths)
    """Selected the sole saved artifact matching this interpreter platform."""

    actual_platform: str = supported_platform()
    """Normalised the actual operating-system family to the manifest vocabulary."""

    actual_architecture: str = platform.machine()
    """Read the machine architecture serving the installed public API."""

    actual_python: str = f"{sys.version_info.major}.{sys.version_info.minor}"
    """Read the standard-CPython minor version serving the installed wheel."""

    if qualification == "release_target":
        for profile in profiles:
            if (
                profile.get("host_id") != host_id
                or profile.get("platform") != actual_platform
                or profile.get("architecture") != actual_architecture
                or profile.get("python_version") != actual_python
            ):
                raise TargetResourceError("TARGET_RESOURCE_INTENDED_HOST_MISMATCH")
        """Required every target profile to name this exact intended host."""

    measurements: list[dict[str, Any]] = []
    """Accumulated one process-local observation for every supported route."""

    failures: list[str] = []
    """Accumulated stable failures without stopping later route attempts."""

    command_environment: dict[str, str] = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ASTERISM_")
    }
    """Removed exploratory model overrides from every isolated measurement process."""

    with tempfile.TemporaryDirectory(prefix="asterism-target-resources-") as temporary:
        temporary_directory: Path = Path(temporary)
        """Created a private short-lived directory for per-process result exchange."""

        for index, profile in enumerate(profiles):
            analysis_id: str = str(profile["analysis_id"])
            """Selected the next supported route without skipping prior failures."""

            worker_output: Path = temporary_directory / f"{index:02d}.json"
            """Selected the private process-local measurement result path."""

            profile_path: Path = temporary_directory / f"{index:02d}-profile.json"
            """Selected the private values-free worker profile path."""

            profile_path.write_bytes(json_bytes(profile))
            """Passed exact non-identifying profile facts without shell interpolation."""

            argv: list[str] = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--analysis-id",
                analysis_id,
                "--profile",
                str(profile_path),
                "--output",
                str(worker_output),
            ]
            """Bound each route to a fresh interpreter for a meaningful RSS high-water mark."""

            started: float = time.perf_counter()
            """Started a parent-side elapsed-time fallback before process creation."""

            completed: subprocess.CompletedProcess[str] = subprocess.run(
                argv,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                env=command_environment,
            )
            """Ran one route in a fresh interpreter without shell interpretation."""

            parent_wall: float = max(
                time.perf_counter() - started, sys.float_info.epsilon
            )
            """Retained parent-observed elapsed time even when a worker failed abruptly."""

            if worker_output.is_file():
                try:
                    raw_measurement: object = json.loads(
                        worker_output.read_text(encoding="utf-8")
                    )
                    """Parsed the worker's values-free process-local observation."""
                except json.JSONDecodeError:
                    raw_measurement = None
                    """Marked malformed worker output as a counted route failure."""
            else:
                raw_measurement = None
                """Marked absent worker output as a counted route failure."""

            if isinstance(raw_measurement, dict):
                measurement: dict[str, Any] = dict(raw_measurement)
                """Copied one well-formed route record before applying acceptances."""
            else:
                measurement = {
                    "analysis_id": analysis_id,
                    "public_entry_points": PUBLIC_ENTRY_POINTS[analysis_id],
                    "public_api": True,
                    "profile_id": profile["profile_id"],
                    "sample_size": profile["sample_size"],
                    "largest_family": profile["largest_family"],
                    "seed": profile["seed"],
                    "input_sha256": None,
                    "wall_time_seconds": parent_wall,
                    "peak_rss_bytes": None,
                    "status": "failed",
                    "timed_out": False,
                    "failure_code": "WORKER_EVIDENCE_MISSING",
                }
                """Counted the attempted route without inventing an unobserved RSS value."""

            measurement["host_id"] = host_id
            """Bound the process-local observation to the parent-verified host."""

            measurement["target_sized"] = qualification in {
                "release_target",
                "target_calibration",
            }
            """Marked whether the profile can participate in release qualification."""

            expected_input: object = profile.get("input_sha256")
            """Read the prewritten target commitment, absent only for bounded smoke."""

            input_matches: bool = (
                qualification == "development_smoke"
                or measurement.get("input_sha256") == expected_input
            )
            """Required qualifying route bytes to match their prewritten commitment."""

            if qualification == "release_target":
                measurement["max_wall_seconds"] = profile["max_wall_seconds"]
                """Copied the unchanged prewritten wall-time ceiling."""

                measurement["max_peak_rss_bytes"] = profile["max_peak_rss_bytes"]
                """Copied the unchanged prewritten peak-memory ceiling."""

                wall_within: bool = positive_number(
                    measurement.get("wall_time_seconds")
                ) and float(measurement["wall_time_seconds"]) <= float(
                    profile["max_wall_seconds"]
                )
                """Compared actual elapsed time with the unchanged target ceiling."""

                peak_value: object = measurement.get("peak_rss_bytes")
                """Read actual normalised peak RSS from the isolated worker."""

                peak_within: bool = (
                    isinstance(peak_value, int)
                    and not isinstance(peak_value, bool)
                    and peak_value > 0
                    and peak_value <= int(profile["max_peak_rss_bytes"])
                )
                """Applied both exact prewritten ceilings to actual process values."""
            else:
                wall_within = positive_number(measurement.get("wall_time_seconds"))
                """Required a real positive smoke elapsed-time observation."""

                peak_value = measurement.get("peak_rss_bytes")
                """Read the bounded smoke worker's normalised peak RSS."""

                peak_within = (
                    isinstance(peak_value, int)
                    and not isinstance(peak_value, bool)
                    and peak_value > 0
                )
                """Required real observations while withholding all release acceptances."""

            completed_route: bool = (
                completed.returncode == 0
                and measurement.get("status") == "completed"
                and measurement.get("timed_out") is False
            )
            """Required a normal zero-exit completion from the isolated public route."""

            measurement["passed"] = (
                completed_route and input_matches and wall_within and peak_within
            )
            """Combined actual completion, input identity and resource decisions."""

            if not measurement["passed"]:
                failures.append(f"{analysis_id}:RESOURCE_MEASUREMENT_FAILED")
            measurements.append(measurement)
        """Attempted all nine isolated routes regardless of earlier failures."""

    first_profile: dict[str, Any] = profiles[0]
    """Read the intended host identity shared by the complete artifact."""

    record: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "qualification": qualification,
        "manifest_sha256": canonical_json_sha256(manifest),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "route_helper_sha256": file_sha256(Path(route_helper.__file__).resolve()),
        "profile_plan_sha256": profile_plan_sha256,
        "build": build,
        "wheel": {
            "name": runtime_wheel.name,
            "sha256": file_sha256(runtime_wheel),
        },
        "host": {
            "id": host_id,
            "platform": actual_platform,
            "architecture": actual_architecture,
            "python_version": actual_python,
            "python_implementation": platform.python_implementation(),
            "peak_rss_source": "getrusage_ru_maxrss",
            "peak_rss_bytes_normalisation": (
                "native_bytes" if actual_platform == "macos" else "kibibytes_x_1024"
            ),
        },
        "analysis_ids": analysis_ids,
        "measurements": measurements,
        "failures": failures,
        "passed": not failures and len(measurements) == len(analysis_ids),
    }
    """Bound actual process evidence to build, wheel, source programs and inputs."""

    if qualification == "release_target":
        expected_host_id: object = first_profile.get("host_id")
        """Rechecked that the aggregate host retained the reviewed stable identity."""

        if expected_host_id != host_id:
            raise TargetResourceError("TARGET_RESOURCE_INTENDED_HOST_MISMATCH")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(json_bytes(record))
    """Persisted complete evidence even when one or more attempted routes failed."""

    return record


def run_worker(analysis_id: str, profile_path: Path, output_path: Path) -> int:
    """Execute one process-local measurement requested by the parent runner.

    Args:
        analysis_id: Supported public route identifier.
        profile_path: Values-free generated-input profile JSON.
        output_path: New process-local measurement JSON path.

    Returns:
        Zero only when the complete public route returned normally.
    """
    if analysis_id not in PUBLIC_ENTRY_POINTS or output_path.exists():
        return 1
    try:
        profile: object = json.loads(profile_path.read_text(encoding="utf-8"))
        """Parsed exact profile facts written by the trusted parent process."""

        if not isinstance(profile, dict) or profile.get("analysis_id") != analysis_id:
            return 1
        measurement: dict[str, Any] = worker_measurement(analysis_id, profile)
        """Measured the target route while retaining no generated input or result values."""

        output_path.write_bytes(json_bytes(measurement))
        """Wrote the process-local high-water mark for parent aggregation."""
    except (OSError, TypeError, ValueError, TargetResourceError):
        return 1
    return 0 if measurement["status"] == "completed" else 1


def main() -> int:
    """Run the saved-wheel target-resource or bounded-smoke command."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Measure all nine public Asterism routes in isolated processes; "
            "development smoke never qualifies target budgets."
        )
    )
    """Defined one explicit parent interface plus an internal process boundary."""

    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, nargs="+")
    parser.add_argument(
        "--qualification",
        choices=("release_target", "target_calibration", "development_smoke"),
        default="development_smoke",
    )
    parser.add_argument("--host-id", default="local-development-smoke")
    parser.add_argument("--profile-plan", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--analysis-id")
    parser.add_argument("--profile", type=Path)
    arguments: argparse.Namespace = parser.parse_args()
    """Parsed either aggregate release inputs or one private worker request."""

    if arguments.worker:
        if not isinstance(arguments.analysis_id, str) or arguments.profile is None:
            return 1
        return run_worker(arguments.analysis_id, arguments.profile, arguments.output)
    if arguments.manifest is None or not arguments.wheel:
        parser.error("--manifest and --wheel are required outside --worker")
    try:
        record: dict[str, Any] = write_target_resource_record(
            arguments.manifest,
            arguments.output,
            arguments.wheel,
            arguments.qualification,
            arguments.host_id,
            arguments.profile_plan,
        )
        """Executed every route and wrote evidence even for resource failures."""
    except (
        OSError,
        TargetResourceError,
        route_helper.SyntheticReceiptError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"target resource measurement refused: {error}", file=sys.stderr)
        return 1
    if record["passed"] is not True:
        print(
            "target resource measurements failed; inspect the artifact", file=sys.stderr
        )
        return 1
    if arguments.qualification == "development_smoke":
        print("All public resource routes completed (non-qualifying smoke).")
    elif arguments.qualification == "target_calibration":
        print("All target routes measured (non-qualifying calibration).")
    else:
        print("All target resource measurements passed their prewritten budgets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
