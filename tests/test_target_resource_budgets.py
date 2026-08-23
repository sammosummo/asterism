"""Contracts for saved-wheel target-resource measurements and evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.measure_target_resources import (
    ARTIFACT_ID,
    PUBLIC_ENTRY_POINTS,
    calibration_profile_errors,
    canonical_json_sha256,
    family_sizes,
    resource_problem,
    target_resource_configuration_errors,
    target_resource_record_errors,
    worker_measurement,
)

ANALYSIS_IDS: list[str] = [
    "one_trait_gaussian_heritability",
    "several_covariance_components",
    "bivariate_genetic_correlation",
    "continuous_gene_by_environment",
    "discrete_gene_by_environment",
    "binary_liability_heritability",
    "one_trait_tobit_audiogram",
    "mixed_binary_censored_genetic_correlation",
    "spatial_component_presence",
]
"""Listed the nine 0.1 routes in authoritative manifest order."""


def digest_bytes(payload: bytes) -> str:
    """Return one deterministic test-fixture digest.

    Args:
        payload: Bytes represented by the fixture commitment.

    Returns:
        Lowercase hexadecimal SHA-256.
    """
    return hashlib.sha256(payload).hexdigest()


def configured_resource_contract(root: Path) -> dict[str, Any]:
    """Create a complete nine-route target-resource contract.

    Args:
        root: Isolated checkout containing both committed runners.

    Returns:
        Parsed manifest section with prewritten target acceptances.
    """
    tool_directory: Path = root / "tools"
    """Selected the repository-relative runner directory."""

    tool_directory.mkdir(parents=True, exist_ok=True)
    """Created the isolated tool directory once for both fixtures."""

    (tool_directory / "measure_target_resources.py").write_text(
        "# fixed measurement runner\n", encoding="utf-8"
    )
    (tool_directory / "synthetic_analysis_receipts.py").write_text(
        "# fixed public-route helper\n", encoding="utf-8"
    )
    """Created the two exact program inputs committed by every record."""

    operating_system: str = "linux" if sys.platform.startswith("linux") else "macos"
    """Named the current supported operating-system family."""

    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
    """Selected the running standard-CPython version for this host contract."""

    measurements: list[dict[str, Any]] = []
    """Accumulated one independently hashed target input per public route."""

    for index, analysis_id in enumerate(ANALYSIS_IDS):
        measurements.append(
            {
                "analysis_id": analysis_id,
                "host_id": "test-intended-host",
                "platform": operating_system,
                "architecture": platform.machine(),
                "python_version": python_version,
                "profile_id": f"{analysis_id}-target-v1",
                "sample_size": 100 + 2 * index,
                "largest_family": 10,
                "seed": 20_260_821 + index,
                "input_sha256": digest_bytes(analysis_id.encode("utf-8")),
                "max_wall_seconds": 10.0 + index,
                "max_peak_rss_bytes": 1_000_000_000 + index,
            }
        )
    """Prewrote finite positive limits without sharing placeholders across routes."""

    return {
        "configured": True,
        "runner": "tools/measure_target_resources.py",
        "route_helper": "tools/synthetic_analysis_receipts.py",
        "timeout_seconds": 180,
        "analysis_ids": ANALYSIS_IDS,
        "measurements": measurements,
    }


def passing_resource_record(
    root: Path,
    manifest: dict[str, Any],
    configuration: dict[str, Any],
    build: dict[str, Any],
    wheel_name: str,
    wheel_sha256: str,
) -> dict[str, Any]:
    """Create a structurally complete passing measurement artifact.

    Args:
        root: Checkout holding the exact runner and route helper.
        manifest: Complete parsed release manifest.
        configuration: Prewritten resource contract from that manifest.
        build: Expected immutable installed-wheel build identity.
        wheel_name: Basename of the selected saved runtime wheel.
        wheel_sha256: Exact digest of that saved wheel.

    Returns:
        Nine-route record suitable for independent verification.
    """
    runner_path: Path = root / str(configuration["runner"])
    """Located the measurement program named by the contract."""

    helper_path: Path = root / str(configuration["route_helper"])
    """Located the public-route helper named by the contract."""

    measurements: list[dict[str, Any]] = []
    """Accumulated actual positive resource observations in manifest order."""

    for expected in configuration["measurements"]:
        measurements.append(
            {
                "analysis_id": expected["analysis_id"],
                "public_entry_points": PUBLIC_ENTRY_POINTS[
                    str(expected["analysis_id"])
                ],
                "public_api": True,
                "target_sized": True,
                "host_id": expected["host_id"],
                "profile_id": expected["profile_id"],
                "sample_size": expected["sample_size"],
                "largest_family": expected["largest_family"],
                "seed": expected["seed"],
                "input_sha256": expected["input_sha256"],
                "max_wall_seconds": expected["max_wall_seconds"],
                "max_peak_rss_bytes": expected["max_peak_rss_bytes"],
                "wall_time_seconds": expected["max_wall_seconds"] / 2.0,
                "peak_rss_bytes": expected["max_peak_rss_bytes"] // 2,
                "status": "completed",
                "timed_out": False,
                "passed": True,
            }
        )
    """Recorded each completion beneath its prewritten wall and memory ceiling."""

    first: dict[str, Any] = configuration["measurements"][0]
    """Read the common intended-host identity represented by this one artifact."""

    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "qualification": "release_target",
        "manifest_sha256": canonical_json_sha256(manifest),
        "runner_sha256": digest_bytes(runner_path.read_bytes()),
        "route_helper_sha256": digest_bytes(helper_path.read_bytes()),
        "build": build,
        "wheel": {"name": wheel_name, "sha256": wheel_sha256},
        "host": {
            "id": first["host_id"],
            "platform": first["platform"],
            "architecture": first["architecture"],
            "python_version": first["python_version"],
            "python_implementation": "CPython",
            "peak_rss_source": "getrusage_ru_maxrss",
            "peak_rss_bytes_normalisation": (
                "native_bytes" if first["platform"] == "macos" else "kibibytes_x_1024"
            ),
        },
        "analysis_ids": ANALYSIS_IDS,
        "measurements": measurements,
        "failures": [],
        "passed": True,
    }


def test_resource_contract_names_every_supported_public_route() -> None:
    """Keep the measurement inventory equal to the complete 0.1 route set."""
    assert list(PUBLIC_ENTRY_POINTS) == ANALYSIS_IDS
    assert all(PUBLIC_ENTRY_POINTS[analysis_id] for analysis_id in ANALYSIS_IDS)


def test_resource_configuration_requires_real_nine_route_acceptances(
    tmp_path: Path,
) -> None:
    """Reject missing routes, placeholder hashes and non-positive ceilings."""
    configuration: dict[str, Any] = configured_resource_contract(tmp_path)
    """Created the complete configured contract before isolated corruptions."""

    assert (
        target_resource_configuration_errors(
            configuration,
            tmp_path,
            ANALYSIS_IDS,
            require_configured=True,
        )
        == []
    )

    missing: dict[str, Any] = deepcopy(configuration)
    """Copied the valid contract before removing one supported route."""

    missing["measurements"] = missing["measurements"][:-1]
    """Modelled incomplete target evidence despite an unchanged identifier claim."""

    assert "do not cover every analysis" in "\n".join(
        target_resource_configuration_errors(
            missing,
            tmp_path,
            ANALYSIS_IDS,
            require_configured=True,
        )
    )

    placeholder: dict[str, Any] = deepcopy(configuration)
    """Copied the valid contract before replacing an input digest by filler."""

    placeholder["measurements"][0]["input_sha256"] = "0" * 64
    """Replaced an exact generated-input hash with obvious filler."""

    placeholder["measurements"][0]["max_wall_seconds"] = 0.0
    """Replaced the wall-time ceiling with an unmeasured zero placeholder."""

    placeholder_errors: str = "\n".join(
        target_resource_configuration_errors(
            placeholder,
            tmp_path,
            ANALYSIS_IDS,
            require_configured=True,
        )
    )
    """Collected all independent placeholder refusals for one repair cycle."""

    assert "input SHA-256 is invalid" in placeholder_errors
    assert "max_wall_seconds must be finite and positive" in placeholder_errors


def test_resource_record_binds_build_wheel_inputs_programs_and_observations(
    tmp_path: Path,
) -> None:
    """Accept only fresh complete measurements beneath prewritten limits."""
    configuration: dict[str, Any] = configured_resource_contract(tmp_path)
    """Created exact program, host, input and acceptance commitments."""

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "version": "0.1.0",
        "release": True,
        "target_resource_budgets": configuration,
        "analyses": [{"id": analysis_id} for analysis_id in ANALYSIS_IDS],
    }
    """Built the complete parsed contract whose canonical bytes identify evidence."""

    build: dict[str, Any] = {
        "version": "0.1.0",
        "cargo_version": "0.1.0",
        "source_commit": "1a2b3c4d" * 5,
        "source_dirty": False,
        "release": True,
        "release_manifest_sha256": digest_bytes(b"exact release.toml bytes"),
        "cargo_lock_sha256": digest_bytes(b"exact Cargo.lock bytes"),
        "uv_lock_sha256": digest_bytes(b"exact uv.lock bytes"),
    }
    """Represented the immutable public build identity from the installed wheel."""

    wheel_sha256: str = digest_bytes(b"saved wheel bytes")
    """Committed the artifact selected for the current runtime platform."""

    record: dict[str, Any] = passing_resource_record(
        tmp_path,
        manifest,
        configuration,
        build,
        "asterism-0.1.0-cp313-abi3-macosx.whl",
        wheel_sha256,
    )
    """Produced one complete values-free nine-route measurement artifact."""

    assert (
        target_resource_record_errors(
            record,
            manifest=manifest,
            root=tmp_path,
            expected_build=build,
            wheel_sha256s={wheel_sha256},
        )
        == []
    )

    stale_input: dict[str, Any] = deepcopy(record)
    """Copied valid evidence before detaching one measurement from its input."""

    stale_input["measurements"][0]["input_sha256"] = digest_bytes(b"other input")
    """Modelled a target run performed on uncommitted generated inputs."""

    assert "input commitment does not match" in "\n".join(
        target_resource_record_errors(
            stale_input,
            manifest=manifest,
            root=tmp_path,
            expected_build=build,
            wheel_sha256s={wheel_sha256},
        )
    )

    over_budget: dict[str, Any] = deepcopy(record)
    """Copied valid evidence before exceeding a prewritten resource ceiling."""

    over_budget["measurements"][1]["wall_time_seconds"] = 1000.0
    """Modelled a completed route whose target runtime is no longer feasible."""

    assert "exceeded max_wall_seconds" in "\n".join(
        target_resource_record_errors(
            over_budget,
            manifest=manifest,
            root=tmp_path,
            expected_build=build,
            wheel_sha256s={wheel_sha256},
        )
    )

    stale_runner: dict[str, Any] = deepcopy(record)
    """Copied valid evidence before changing the committed measurement program."""

    (tmp_path / "tools" / "measure_target_resources.py").write_text(
        "# changed measurement runner\n", encoding="utf-8"
    )
    """Changed executable bytes after the purported passing observation."""

    assert "runner SHA-256 does not match" in "\n".join(
        target_resource_record_errors(
            stale_runner,
            manifest=manifest,
            root=tmp_path,
            expected_build=build,
            wheel_sha256s={wheel_sha256},
        )
    )


def test_development_smoke_and_passing_placeholder_cannot_qualify_release(
    tmp_path: Path,
) -> None:
    """Keep bounded plumbing probes distinct from target-host qualification."""
    configuration: dict[str, Any] = configured_resource_contract(tmp_path)
    """Created a complete target contract to isolate record-level refusals."""

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "version": "0.1.0",
        "release": True,
        "target_resource_budgets": configuration,
        "analyses": [{"id": analysis_id} for analysis_id in ANALYSIS_IDS],
    }
    """Built the parsed manifest whose resource section is under verification."""

    build: dict[str, Any] = {
        "version": "0.1.0",
        "cargo_version": "0.1.0",
        "source_commit": "1a2b3c4d" * 5,
        "source_dirty": False,
        "release": True,
        "release_manifest_sha256": digest_bytes(b"release manifest"),
        "cargo_lock_sha256": digest_bytes(b"cargo lock"),
        "uv_lock_sha256": digest_bytes(b"uv lock"),
    }
    """Represented the installed release-wheel identity expected by the verifier."""

    wheel_sha256: str = digest_bytes(b"wheel")
    """Selected the sole saved runtime artifact commitment."""

    smoke: dict[str, Any] = passing_resource_record(
        tmp_path,
        manifest,
        configuration,
        build,
        "asterism-0.1.0-cp313-abi3-macosx.whl",
        wheel_sha256,
    )
    """Started from a complete record before changing only qualification status."""

    smoke["qualification"] = "development_smoke"
    """Marked the complete bounded run as deliberately non-qualifying."""

    for measurement in smoke["measurements"]:
        measurement["target_sized"] = False
        """Removed target qualification from this individual smoke observation."""
    """Marked every bounded plumbing observation honestly as non-target evidence."""

    smoke_errors: str = "\n".join(
        target_resource_record_errors(
            smoke,
            manifest=manifest,
            root=tmp_path,
            expected_build=build,
            wheel_sha256s={wheel_sha256},
        )
    )
    """Applied the same independent verifier used by release automation."""

    assert "qualification does not match release" in smoke_errors
    assert "target_sized must be true" in smoke_errors

    placeholder_errors: str = "\n".join(
        target_resource_record_errors(
            {"passed": True},
            manifest=manifest,
            root=tmp_path,
            expected_build=build,
            wheel_sha256s={wheel_sha256},
        )
    )
    """Tested the former aggregate-only shape against the strict schema."""

    assert "schema_version does not match release" in placeholder_errors
    assert "measurement inventory is incomplete" in placeholder_errors


def test_canonical_resource_digest_rejects_json_formatting_dependence() -> None:
    """Hash parsed commitments independently of whitespace and key order."""
    first: dict[str, object] = {"b": [2, 3], "a": 1}
    """Created one parsed commitment in non-canonical insertion order."""

    second: dict[str, object] = json.loads('{\n  "a": 1, "b": [2, 3]\n}')
    """Parsed the same logical commitment from differently formatted JSON."""

    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_generated_target_family_structure_is_exact_and_deterministic() -> None:
    """Preserve requested roster and largest-family stress without participant rows."""
    sizes: list[int] = family_sizes(11, 4)
    """Partitioned an odd roster around one reviewed largest family."""

    assert sizes == [4, 2, 2, 2, 1]
    assert family_sizes(5, 1) == [1, 1, 1, 1, 1]

    first: dict[str, Any] = resource_problem(11, 4, 123, include_distance=True)
    """Generated one complete spatial-capable participant-free target problem."""

    second: dict[str, Any] = resource_problem(11, 4, 123, include_distance=True)
    """Regenerated the same profile independently with the same fixed seed."""

    assert first["relationship"].shape == (11, 11)
    assert first["distance"].shape == (11, 11)
    assert np.array_equal(first["relationship"], second["relationship"])
    assert np.array_equal(first["response"], second["response"])
    assert first["relationship"][0, 3] == 0.5
    assert first["relationship"][0, 4] == 0.0


def test_failed_public_route_retains_real_attempt_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count a public refusal as failure while preserving observed wall time and RSS."""

    def refused_route(
        analysis_id: str,
        problem: dict[str, Any],
        subject_order_sha256: str,
    ) -> None:
        """Raise a stable test refusal at the public-route boundary.

        Args:
            analysis_id: Supported route selected by the worker.
            problem: Generated participant-free inputs.
            subject_order_sha256: Commitment to their exact row order.
        """
        del analysis_id, problem, subject_order_sha256
        """Made explicit that this test double never inspects generated values."""

        raise ValueError("deliberate test refusal")

    monkeypatch.setattr(
        "tools.measure_target_resources.public_route",
        refused_route,
    )
    """Replaced numerical work with a deterministic public-boundary refusal."""

    profile: dict[str, Any] = {
        "analysis_id": "one_trait_gaussian_heritability",
        "profile_id": "one-trait-development-smoke-v1",
        "sample_size": 8,
        "largest_family": 2,
        "seed": 123,
    }
    """Selected a tiny generated profile for process-resource contract testing."""

    measurement: dict[str, Any] = worker_measurement(
        "one_trait_gaussian_heritability", profile
    )
    """Measured the failed attempt through the same worker seam used in production."""

    assert measurement["status"] == "failed"
    assert measurement["failure_code"] == "PUBLIC_ROUTE_VALUEERROR"
    assert measurement["wall_time_seconds"] > 0.0
    assert measurement["peak_rss_bytes"] > 0


def test_target_resource_cli_explains_nonqualifying_smoke() -> None:
    """Expose the target-versus-smoke boundary through the public command help."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "tools/measure_target_resources.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    """Ran only argument parsing without generating or fitting any inputs."""

    assert completed.returncode == 0
    assert "development smoke never qualifies target budgets" in " ".join(
        completed.stdout.split()
    )


def test_target_calibration_plan_needs_all_routes_without_acceptance_limits() -> None:
    """Allow real target measurement before wall and RSS budgets have been chosen."""
    profiles: list[dict[str, Any]] = [
        {
            "analysis_id": analysis_id,
            "profile_id": f"{analysis_id}-target-calibration-v1",
            "sample_size": 1900,
            "largest_family": 180,
            "seed": 20_260_900 + index,
        }
        for index, analysis_id in enumerate(ANALYSIS_IDS)
    ]
    """Created reviewed target dimensions without inventing resource ceilings."""

    assert calibration_profile_errors(profiles, ANALYSIS_IDS) == []
    assert all("max_wall_seconds" not in profile for profile in profiles)
    assert all("max_peak_rss_bytes" not in profile for profile in profiles)

    incomplete: list[dict[str, Any]] = profiles[:-1]
    """Removed one route to model a misleading partial calibration campaign."""

    assert "do not cover every analysis" in "\n".join(
        calibration_profile_errors(incomplete, ANALYSIS_IDS)
    )

    smoke_named: list[dict[str, Any]] = deepcopy(profiles)
    """Copied the complete plan before mislabelling one target as a smoke."""

    smoke_named[0]["profile_id"] = "not-a-target-smoke"
    """Modelled a bounded profile that cannot justify target acceptances."""

    assert "profile_id is not target-sized" in "\n".join(
        calibration_profile_errors(smoke_named, ANALYSIS_IDS)
    )
