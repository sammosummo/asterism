"""Public contracts for the fail-closed scientific release runner."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.check_release import (
    fit_record_reportability_errors,
    release_evidence_errors,
)
from tools.compare_cross_platform import canonical_manifest_sha256
from tools.run_scientific_release import (
    ReleaseConfigurationError,
    agreement_record_errors,
    fixed_input_errors,
    run_scientific_release,
)


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one test artifact.

    Args:
        path: File whose committed bytes are being identified.

    Returns:
        The lowercase hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configured_manifest(check_name: str = "smoke") -> str:
    """Return a complete one-analysis release manifest for runner tests.

    Args:
        check_name: Stable identifier shared by the required check and pass rule.

    Returns:
        TOML text representing a deliberately configured release.
    """
    operating_system: str = "linux" if sys.platform.startswith("linux") else "macos"
    """Named the current supported operating-system family for resource evidence."""

    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
    """Selected the current standard-CPython minor version."""

    input_sha256: str = hashlib.sha256(b"fixed target resource input").hexdigest()
    """Committed the deterministic system-boundary target fixture."""

    return f'''schema_version = 1
version = "0.1.0"
cargo_version = "0.1.0"
rust_toolchain = "1.97.1"
maturin_version = "1.14.1"
uv_version = "0.11.21"
release = true
requires_python = ">=3.13,<3.15"
scientific_pass_rules_configured = true
scientific_runner = "tools/run_scientific_release.py"

[synthetic_analysis_receipts]
configured = true
runner = "tools/synthetic_analysis_receipts.py"
timeout_seconds = 30
analysis_ids = ["one_trait_gaussian_heritability"]

[cross_platform_agreement]
configured = true
probe_runner = "tools/cross_platform_probe.py"
runner = "tools/compare_cross_platform.py"
exact_comparisons = ["outcome_status", "refusal_code", "boundary_state", "field_presence"]

[[cross_platform_agreement.model_tolerances]]
analysis_id = "one_trait_gaussian_heritability"
measured = true
numeric_fields = ["fit.h2"]
absolute = {{ "fit.h2" = 1e-8 }}
relative = {{ "fit.h2" = 1e-6 }}

[[analyses]]
id = "one_trait_gaussian_heritability"
support = "supported_in_0_1"
entry_points = ["asterism.prepare"]
reportable_quantities = ["h2"]
required_checks = ["{check_name}"]
pass_rules_configured = true
pass_rules = [
  {{ id = "{check_name}", command = ["checks/{check_name}.py"], expected_exit_code = 0, timeout_seconds = 30, status = "ready", design_facts = {{ participant_free = true, sample_size = {{ min = 4, max = 100 }}, largest_family = {{ max = 20 }}, trait_type = ["continuous"], components = ["additive_relationship", "residual"] }} }},
]

[analyses.design_range]
measured = true

[analyses.design_range.sample_size]
min = 4
max = 100

[analyses.design_range.largest_family]
max = 20

[analyses.design_range.trait_type]
allowed = ["continuous"]

[analyses.design_range.components]
allowed = ["additive_relationship", "residual"]
'''


def passing_agreement(manifest_text: str) -> dict[str, Any]:
    """Return a complete passing four-target agreement record for runner tests.

    Args:
        manifest_text: Exact simulated release contract.

    Returns:
        Comparison record bound to the simulated manifest and source commit.
    """
    manifest: dict[str, Any] = tomllib.loads(manifest_text)
    """Parsed the same contract representation consumed by the comparator."""

    artifacts: list[dict[str, str]] = [
        {
            "platform": platform,
            "architecture": architecture,
            "python_version": python_version,
            "wheel_sha256": "b" * 64,
        }
        for platform, architecture, python_version in (
            ("macos", "arm64", "3.13"),
            ("linux", "x86_64", "3.13"),
            ("macos", "arm64", "3.14"),
            ("linux", "x86_64", "3.14"),
        )
    ]
    """Represented the exact platform/interpreter artifact matrix."""

    comparisons: list[dict[str, Any]] = [
        {
            "python_version": python_version,
            "passed": True,
            "failures": [],
            "exact": [
                {
                    "analysis_id": "one_trait_gaussian_heritability",
                    "field": field,
                    "reference": {
                        "outcome_status": "candidate",
                        "refusal_code": None,
                        "boundary_state": {},
                        "field_presence": ["fit", "fit.h2"],
                    }[field],
                    "candidate": {
                        "outcome_status": "candidate",
                        "refusal_code": None,
                        "boundary_state": {},
                        "field_presence": ["fit", "fit.h2"],
                    }[field],
                    "passed": True,
                }
                for field in (
                    "outcome_status",
                    "refusal_code",
                    "boundary_state",
                    "field_presence",
                )
            ],
            "numeric": [
                {
                    "analysis_id": "one_trait_gaussian_heritability",
                    "field": "fit.h2",
                    "reference": 0.5,
                    "candidate": 0.5,
                    "difference": 0.0,
                    "absolute_tolerance": 1e-8,
                    "relative_tolerance": 1e-6,
                    "allowed_difference": 5.1e-7,
                    "passed": True,
                }
            ],
        }
        for python_version in ("3.13", "3.14")
    ]
    """Recorded every exact and numeric decision on both Python versions."""

    return {
        "schema_version": 1,
        "probe_id": "asterism-cross-platform-0.1",
        "configured": True,
        "manifest_sha256": canonical_manifest_sha256(manifest),
        "version": "0.1.0",
        "release": True,
        "source_commit": "a" * 40,
        "source_dirty": False,
        "artifacts": artifacts,
        "passed": True,
        "status": "passed",
        "comparisons": comparisons,
        "failures": [],
    }


def test_receipt_verifier_recomputes_reportability_from_the_fit_record() -> None:
    """Reject internally self-consistent claims built from nonreportable fits."""
    reportable: dict[str, Any] = {
        "converged": True,
        "h2": 0.5,
        "interval": {"lower": 0.2, "upper": 0.8, "profile_failures": 0},
    }
    """Built the smallest finite converged record satisfying one promised field."""

    assert fit_record_reportability_errors(reportable, ["h2"]) == []
    assert "fit did not converge" in fit_record_reportability_errors(
        {**reportable, "converged": False}, ["h2"]
    )
    assert "required quantity h2 is missing" in fit_record_reportability_errors(
        {**reportable, "h2": None}, ["h2"]
    )
    assert "interval.profile_failures is 2" in fit_record_reportability_errors(
        {
            **reportable,
            "interval": {"lower": 0.2, "upper": 0.8, "profile_failures": 2},
        },
        ["h2"],
    )
    assert "interval.upper is not finite" in fit_record_reportability_errors(
        {
            **reportable,
            "interval": {
                "lower": 0.2,
                "upper": float("nan"),
                "profile_failures": 0,
            },
        },
        ["h2"],
    )


def test_runner_records_fixed_build_artifact_and_command_evidence(
    tmp_path: Path,
) -> None:
    """Preserve enough evidence to audit exactly what passed and what was built."""
    root: Path = tmp_path / "checkout"
    """Created an isolated release checkout."""

    (root / "checks").mkdir(parents=True)
    (root / "tools").mkdir()
    """Created the only source directories used by the configured check."""

    manifest_text: str = configured_manifest()
    """Selected a complete release contract with one executable pass rule."""

    manifest_path: Path = root / "release.toml"
    """Selected the sole editable release contract in the simulated checkout."""

    manifest_path.write_text(manifest_text, encoding="utf-8")
    """Committed the authoritative manifest bytes for the simulated build."""

    lock_texts: dict[str, str] = {
        lock_name: f"{lock_name} fixture\n" for lock_name in ("Cargo.lock", "uv.lock")
    }
    """Created exact dependency-lock bytes shared by checkout and extension."""

    for lock_name, lock_text in lock_texts.items():
        (root / lock_name).write_text(lock_text, encoding="utf-8")
    """Created both lock inputs whose digests must enter release evidence."""

    check_path: Path = root / "checks" / "smoke.py"
    """Selected the configured scientific command inside the checkout."""

    check_path.write_text(
        "import json\nprint(json.dumps({'metric': 1.0, 'passed': True}))\n",
        encoding="utf-8",
    )
    """Created a successful scientific command with observable output."""

    receipt_runner: Path = root / "tools" / "synthetic_analysis_receipts.py"
    """Selected the installed-wheel standard-receipt command from the manifest."""

    receipt_runner.write_text(
        """from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--wheel", type=Path, nargs="+", required=True)
args = parser.parse_args()
args.output.mkdir(parents=True)
wheel_sha256 = hashlib.sha256(args.wheel[0].read_bytes()).hexdigest()
manifest_sha256 = hashlib.sha256(Path("release.toml").read_bytes()).hexdigest()
cargo_sha256 = hashlib.sha256(Path("Cargo.lock").read_bytes()).hexdigest()
uv_sha256 = hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
build = {
    "version": "0.1.0",
    "cargo_version": "0.1.0",
    "source_commit": "a" * 40,
    "source_dirty": False,
    "release": True,
    "release_manifest_sha256": manifest_sha256,
    "cargo_lock_sha256": cargo_sha256,
    "uv_lock_sha256": uv_sha256,
}
order = "d" * 64
receipt = {
    "schema_version": 1,
    "asterism_version": "0.1.0",
    "analysis": "one_trait_gaussian_heritability",
    "outcome": "reportable",
    "build": build,
    "preflight": {
        "inside_supported_range": True,
        "reportable_quantities": ["h2"],
    },
    "model": {"estimator": "reml"},
    "provenance": {
        "wheel_sha256": wheel_sha256,
        "dependency_lock_sha256": uv_sha256,
        "consumer_commit": "a" * 40,
        "input_commitments": {
            "synthetic_fixture_sha256": "e" * 64,
            "subject_order_sha256": order,
        },
    },
    "fit_record": {
        "converged": True,
        "h2": 0.5,
        "build": build,
        "subject_order_sha256": order,
    },
    "reportability_issues": [],
    "refusal": None,
}
receipt_path = args.output / "one_trait_gaussian_heritability.json"
payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\\n").encode()
receipt_path.write_bytes(payload)
index = {
    "schema_version": 1,
    "receipts": [{
        "analysis_id": "one_trait_gaussian_heritability",
        "outcome": "reportable",
        "path": receipt_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "wheel_sha256": wheel_sha256,
    }],
}
(args.output / "index.json").write_text(json.dumps(index, indent=2) + "\\n")
""",
        encoding="utf-8",
    )
    """Created one deterministic system-boundary fixture emulating the real command."""


    wheel_path: Path = tmp_path / "asterism-0.1.0-cp313-abi3-manylinux.whl"
    """Selected the immutable artifact to bind into release evidence."""

    wheel_path.write_bytes(b"fixed wheel bytes")
    """Created the immutable wheel whose hash must bind the evidence."""

    agreement: dict[str, Any] = passing_agreement(manifest_text)
    """Created a real complete comparison record for the same fixed build."""

    for artifact in agreement["artifacts"]:
        artifact["wheel_sha256"] = sha256(wheel_path)
        """Bound every simulated target to the sole fixture wheel artifact."""

    agreement_path: Path = tmp_path / "cross-platform-agreement.json"
    """Selected the downloaded comparison record consumed by the release runner."""

    agreement_path.write_text(json.dumps(agreement, indent=2) + "\n", encoding="utf-8")
    """Persisted the actual comparison details before scientific evidence began."""

    core_path: Path = tmp_path / "site-packages" / "asterism" / "_core.so"
    """Selected an installed extension path outside the mutable checkout."""

    core_path.parent.mkdir(parents=True)
    core_path.write_bytes(b"fixed extension bytes")
    """Created the installed extension binary whose digest identifies the build."""

    core: Any = SimpleNamespace(
        __file__=str(core_path),
        __release_manifest__=manifest_text,
        __cargo_lock__=lock_texts["Cargo.lock"],
        __uv_lock__=lock_texts["uv.lock"],
        __source_commit__="a" * 40,
        __source_dirty__=False,
        __version__="0.1.0",
    )
    """Represented the immutable metadata exposed by an installed wheel."""

    output_directory: Path = tmp_path / "release-evidence"
    """Selected a new directory for command logs and the evidence record."""

    evidence: dict[str, Any] = run_scientific_release(
        manifest_path=manifest_path,
        output_directory=output_directory,
        wheel_paths=[wheel_path],
        cross_platform_agreement_path=agreement_path,
        core=core,
    )
    """Ran the configured rule through the same public seam as automation."""

    saved: dict[str, Any] = json.loads(
        (output_directory / "evidence.json").read_text(encoding="utf-8")
    )
    """Read back the durable machine-readable evidence rather than in-memory state."""

    assert evidence == saved
    assert saved["passed"] is True
    assert saved["sanitized_environment_prefixes"] == ["ASTERISM_"]
    assert saved["build"] == {
        "version": "0.1.0",
        "cargo_version": "0.1.0",
        "rust_toolchain": "1.97.1",
        "maturin_version": "1.14.1",
        "uv_version": "0.11.21",
        "release": True,
        "source_commit": "a" * 40,
        "source_dirty": False,
        "extension_path": str(core.__file__),
        "extension_sha256": sha256(core_path),
        "release_manifest_sha256": sha256(manifest_path),
        "cargo_lock_sha256": sha256(root / "Cargo.lock"),
        "uv_lock_sha256": sha256(root / "uv.lock"),
    }
    assert saved["inputs"]["release.toml"]["sha256"] == sha256(manifest_path)
    assert saved["inputs"]["Cargo.lock"]["sha256"] == sha256(root / "Cargo.lock")
    assert saved["inputs"]["uv.lock"]["sha256"] == sha256(root / "uv.lock")
    assert saved["wheels"] == [{"path": str(wheel_path), "sha256": sha256(wheel_path)}]
    assert saved["cross_platform_agreement"]["record"] == agreement
    copied_agreement: Path = (
        output_directory / saved["cross_platform_agreement"]["path"]
    )
    """Located the exact comparison record copied into durable release evidence."""

    assert saved["cross_platform_agreement"]["sha256"] == sha256(copied_agreement)

    tampered: dict[str, Any] = deepcopy(agreement)
    """Copied a complete record before changing one claimed numerical threshold."""

    tampered["comparisons"][0]["numeric"][0]["absolute_tolerance"] = 1.0
    """Modelled a comparator artifact that enlarged a pre-written tolerance."""

    tamper_errors: list[str] = agreement_record_errors(
        record=tampered,
        manifest=tomllib.loads(manifest_text),
        source_commit="a" * 40,
        wheel_sha256s={sha256(wheel_path)},
    )
    """Recomputed tolerance provenance rather than trusting the passing boolean."""

    assert "numeric decision is invalid" in "\n".join(tamper_errors)
    assert saved["commands"][0]["analysis_id"] == "one_trait_gaussian_heritability"
    assert saved["commands"][0]["check_id"] == "smoke"
    assert saved["commands"][0]["expected_exit_code"] == 0
    assert saved["commands"][0]["actual_exit_code"] == 0
    assert saved["commands"][0]["status"] == "ready"
    assert saved["commands"][0]["design_facts"] == {
        "participant_free": True,
        "sample_size": {"min": 4, "max": 100},
        "largest_family": {"max": 20},
        "trait_type": ["continuous"],
        "components": ["additive_relationship", "residual"],
    }
    assert saved["commands"][0]["passed"] is True
    assert saved["commands"][0]["stdout_sha256"] == sha256(
        output_directory / saved["commands"][0]["stdout_path"]
    )
    synthetic: dict[str, Any] = saved["synthetic_analysis_receipts"]
    """Read the actual standard-receipt subprocess and its durable output identity."""

    assert synthetic["passed"] is True
    assert synthetic["actual_exit_code"] == 0
    assert [receipt["analysis_id"] for receipt in synthetic["receipts"]] == [
        "one_trait_gaussian_heritability"
    ]

    assert (
        release_evidence_errors(
            evidence_path=output_directory / "evidence.json",
            wheel_paths=[wheel_path],
            root=root,
            core=core,
        )
        == []
    )
    """Required the independent verifier to accept the complete durable record."""

    wrong_output: dict[str, Any] = deepcopy(saved)
    """Copied valid evidence before detaching the command from its indexed output."""

    wrong_output["synthetic_analysis_receipts"]["argv"][3] = str(tmp_path / "elsewhere")
    """Claimed the subprocess wrote somewhere other than the retained receipt set."""

    wrong_output_path: Path = output_directory / "wrong-receipt-output-evidence.json"
    """Selected an independent evidence record for the detached-output claim."""

    wrong_output_path.write_text(
        json.dumps(wrong_output, indent=2) + "\n", encoding="utf-8"
    )
    """Persisted the self-consistent aggregate evidence around the false path."""

    wrong_output_errors: list[str] = release_evidence_errors(
        evidence_path=wrong_output_path,
        wheel_paths=[wheel_path],
        root=root,
        core=core,
    )
    """Required the verifier to bind argv output to the retained index directory."""

    assert "synthetic receipt command output does not match index" in "\n".join(
        wrong_output_errors
    )

    placeholder: dict[str, Any] = deepcopy(saved)
    """Copied valid evidence before replacing actual comparison details."""

    placeholder["cross_platform_agreement"] = {"passed": True}
    """Modelled the aggregate placeholder the release gate must never accept."""

    placeholder_path: Path = output_directory / "placeholder-evidence.json"
    """Selected an independent malformed evidence record for verification."""

    placeholder_path.write_text(
        json.dumps(placeholder, indent=2) + "\n", encoding="utf-8"
    )
    """Persisted the placeholder without changing valid evidence."""

    placeholder_errors: list[str] = release_evidence_errors(
        evidence_path=placeholder_path,
        wheel_paths=[wheel_path],
        root=root,
        core=core,
    )
    """Ran the actual release verifier against the insufficient aggregate claim."""

    assert "has no path" in "\n".join(placeholder_errors)

    synthetic_placeholder: dict[str, Any] = deepcopy(saved)
    """Copied valid evidence before discarding actual per-analysis receipts."""

    synthetic_placeholder["synthetic_analysis_receipts"] = {"passed": True}
    """Modelled an aggregate claim with no command, index or receipt bytes."""

    synthetic_placeholder_path: Path = (
        output_directory / "synthetic-placeholder-evidence.json"
    )
    """Selected an independent malformed receipt-evidence record."""

    synthetic_placeholder_path.write_text(
        json.dumps(synthetic_placeholder, indent=2) + "\n",
        encoding="utf-8",
    )
    """Persisted the aggregate placeholder without changing valid evidence."""

    synthetic_placeholder_errors: list[str] = release_evidence_errors(
        evidence_path=synthetic_placeholder_path,
        wheel_paths=[wheel_path],
        root=root,
        core=core,
    )
    """Ran the independent release verifier against the insufficient claim."""

    assert "synthetic receipt evidence has no index" in "\n".join(
        synthetic_placeholder_errors
    )

    wheel_path.write_bytes(b"different wheel bytes")
    """Changed the saved artifact after evidence production to model a stale wheel."""

    verification_errors: list[str] = release_evidence_errors(
        evidence_path=output_directory / "evidence.json",
        wheel_paths=[wheel_path],
        root=root,
        core=core,
    )
    """Rechecked the produced record against the artifact bytes selected for release."""

    assert "wheel SHA-256 set does not match" in "\n".join(verification_errors)


def test_runner_refuses_a_required_check_without_one_machine_rule(
    tmp_path: Path,
) -> None:
    """Never infer a scientific command or threshold from a human-readable name."""
    root: Path = tmp_path / "checkout"
    """Selected the isolated checkout containing the incomplete release contract."""

    root.mkdir()
    """Created an isolated checkout containing only release inputs."""

    complete_manifest: str = configured_manifest()
    """Built the valid one-rule contract before removing only its rule table."""

    rule_start: int = complete_manifest.index("pass_rules = [")
    """Located the start of the sole inline pass-rule list."""

    rule_end: int = complete_manifest.index("]\n\n[analyses.design_range]", rule_start)
    """Located the list terminator independently of its evolving rule schema."""

    manifest_text: str = (
        complete_manifest[:rule_start]
        + "pass_rules = []"
        + complete_manifest[rule_end + 1 :]
    )
    """Removed the sole executable rule while retaining its required-check name."""

    manifest_path: Path = root / "release.toml"
    """Selected the incomplete authoritative contract for this preflight."""

    manifest_path.write_text(manifest_text, encoding="utf-8")
    """Wrote the incomplete manifest for a fail-closed preflight."""

    core_path: Path = tmp_path / "site-packages" / "asterism" / "_core.so"
    """Selected an installed extension path outside the mutable checkout."""

    core_path.parent.mkdir(parents=True)
    core_path.write_bytes(b"fixed extension bytes")
    """Created the installed extension needed beyond the configuration preflight."""

    core: Any = SimpleNamespace(
        __file__=str(core_path),
        __release_manifest__=manifest_text,
        __cargo_lock__="Cargo.lock fixture\n",
        __uv_lock__="uv.lock fixture\n",
        __source_commit__="a" * 40,
        __source_dirty__=False,
        __version__="0.1.0",
    )
    """Represented a wheel built from the same incomplete manifest."""

    with pytest.raises(ReleaseConfigurationError, match="exactly one pass rule"):
        run_scientific_release(
            manifest_path=manifest_path,
            output_directory=tmp_path / "release-evidence",
            wheel_paths=[],
            cross_platform_agreement_path=tmp_path / "agreement.json",
            core=core,
        )
    """Required configuration failure before any scientific command could run."""


def test_fixed_input_preflight_refuses_stale_embedded_dependency_locks() -> None:
    """Never run scientific commands with dependencies different from the checkout."""
    core: Any = SimpleNamespace(
        __release_manifest__="manifest bytes",
        __cargo_lock__="stale Cargo lock",
        __uv_lock__="uv lock bytes",
    )
    """Represented an extension built before the Cargo dependency graph changed."""

    errors: list[str] = fixed_input_errors(
        core=core,
        manifest_text="manifest bytes",
        cargo_lock_text="current Cargo lock",
        uv_lock_text="uv lock bytes",
    )
    """Compared embedded bytes to the exact checkout inputs independently of hashes."""

    assert errors == ["installed wheel does not embed the selected Cargo.lock"]
