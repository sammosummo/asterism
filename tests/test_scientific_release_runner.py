"""Public contracts for the fail-closed scientific release runner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tools.run_scientific_release as scientific_release_runner
from tools.check_release import (
    fit_record_failures,
    release_evidence_errors,
)
from tools.compare_cross_platform import canonical_manifest_sha256
from tools.run_scientific_release import (
    ReleaseConfigurationError,
    agreement_record_errors,
    fixed_input_errors,
    medusa_smoke_record_errors,
    run_bounded_command,
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


def write_test_wheel(path: Path, extension_bytes: bytes) -> None:
    """Write a minimal wheel archive with one Asterism native module.

    Args:
        path: Wheel path to create.
        extension_bytes: Exact native-module payload to retain in the archive.
    """
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("asterism/_core.abi3.so", extension_bytes)
    """Created the one archive member needed for native-byte binding tests."""


def configured_manifest(check_name: str = "smoke") -> str:
    """Return a complete one-analysis release manifest for runner tests.

    Args:
        check_name: Stable identifier shared by the required check and pass rule.

    Returns:
        TOML text representing a deliberately configured release.
    """

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

[medusa_smoke]
required = true
configured = true
architecture = "x86_64"
glibc_version = "2.28"
command = ["tools/medusa_wheel_smoke.py"]

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
supported_quantities = ["h2"]
required_checks = ["{check_name}"]
pass_rules_configured = true
pass_rules = [
  {{ id = "{check_name}", command = ["checks/{check_name}.py"], expected_exit_code = 0, timeout_seconds = 30, status = "ready", design_facts = {{ participant_free = true, sample_size = {{ min = 4, max = 100 }}, largest_family = {{ max = 20 }}, trait_type = ["continuous"], components = ["additive_relationship", "residual"] }} }},
]

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
                        "outcome_status": "fitted",
                        "refusal_code": None,
                        "boundary_state": {},
                        "field_presence": ["fit", "fit.h2"],
                    }[field],
                    "candidate": {
                        "outcome_status": "fitted",
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


def passing_medusa_smoke(
    manifest_text: str,
    cargo_lock_text: str,
    uv_lock_text: str,
    wheel_sha256: str,
    extension_sha256: str,
) -> dict[str, Any]:
    """Return a clean Medusa result bound to one final manylinux wheel."""
    manifest: dict[str, Any] = tomllib.loads(manifest_text)
    """Parsed the exact final release identity exercised by the smoke."""

    return {
        "asterism_version": manifest["version"],
        "build_identity": {
            "version": manifest["version"],
            "cargo_version": manifest["cargo_version"],
            "source_commit": "a" * 40,
            "source_dirty": False,
            "release": manifest["release"],
            "release_manifest_sha256": hashlib.sha256(
                manifest_text.encode("utf-8")
            ).hexdigest(),
            "cargo_lock_sha256": hashlib.sha256(
                cargo_lock_text.encode("utf-8")
            ).hexdigest(),
            "uv_lock_sha256": hashlib.sha256(uv_lock_text.encode("utf-8")).hexdigest(),
        },
        "machine": "x86_64",
        "glibc_version": "2.28",
        "python_version": "3.13.14",
        "wheel_sha256": wheel_sha256,
        "extension_sha256": extension_sha256,
        "converged": True,
        "people": 360,
        "largest_family": 6,
        "loglik": -533.0,
        "mean_diagonal_proportions": [0.0, 1.0],
    }


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (("asterism_version",), "version does not match"),
        (("machine",), "architecture does not match"),
        (("glibc_version",), "glibc does not match"),
        (("converged",), "did not converge"),
        (("wheel_sha256",), "wheel SHA-256 does not match"),
        (("extension_sha256",), "extension SHA-256 does not match"),
        (("build_identity", "version"), "build identity does not match"),
        (("build_identity", "release"), "build identity does not match"),
        (("build_identity", "source_commit"), "build identity does not match"),
        (("build_identity", "source_dirty"), "build identity does not match"),
        (
            ("build_identity", "release_manifest_sha256"),
            "build identity does not match",
        ),
        (("build_identity", "cargo_lock_sha256"), "build identity does not match"),
        (("build_identity", "uv_lock_sha256"), "build identity does not match"),
    ],
)
def test_medusa_smoke_rejects_stale_runtime_fit_and_build_identity(
    change: tuple[str, ...], expected: str
) -> None:
    """Accept only a converged smoke from the selected final Linux wheel."""
    manifest_text: str = configured_manifest()
    """Rendered one release-ready manifest without a repository evidence pointer."""

    cargo_lock_text: str = "Cargo.lock fixture\n"
    """Represented the exact Rust lock text embedded in the selected wheel."""

    uv_lock_text: str = "uv.lock fixture\n"
    """Represented the exact Python lock text embedded in the selected wheel."""

    wheel_sha256: str = "b" * 64
    """Named the exact portable artifact digest accepted by the validator."""

    extension_sha256: str = "c" * 64
    """Named the exact archived native-member digest accepted by the validator."""

    record: dict[str, Any] = passing_medusa_smoke(
        manifest_text,
        cargo_lock_text,
        uv_lock_text,
        wheel_sha256,
        extension_sha256,
    )
    """Built one valid external result before changing exactly one claim."""

    target: dict[str, Any] = record
    """Started at the complete external record before locating the changed claim."""

    for field in change[:-1]:
        target = target[field]
        """Descended through the selected nested build-identity path."""

    target[change[-1]] = (
        not target[change[-1]]
        if change[-1]
        in {
            "converged",
            "release",
            "source_dirty",
        }
        else "stale"
    )
    """Made the selected categorical, boolean or digest identity stale."""

    errors: list[str] = medusa_smoke_record_errors(
        record=record,
        manifest=tomllib.loads(manifest_text),
        manifest_text=manifest_text,
        cargo_lock_text=cargo_lock_text,
        uv_lock_text=uv_lock_text,
        source_commit="a" * 40,
        linux_wheel_sha256=wheel_sha256,
        linux_extension_sha256=extension_sha256,
    )
    """Validated the external result against independently selected inputs."""

    assert expected in "\n".join(errors)


@pytest.mark.parametrize(
    ("payload", "wheel_extension_bytes", "expected"),
    [
        (None, b"linux extension bytes", "Medusa smoke does not exist"),
        ("{", b"linux extension bytes", "Medusa smoke is unreadable"),
        (
            "valid",
            None,
            "selected manylinux wheel is not a readable wheel archive",
        ),
    ],
)
def test_runner_refuses_unverifiable_external_medusa_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str | None,
    wheel_extension_bytes: bytes | None,
    expected: str,
) -> None:
    """Refuse unreadable evidence and a false manylinux wheel archive."""
    root: Path = tmp_path / "checkout"
    """Located a minimal checkout fixture for release preflight."""

    root.mkdir()
    manifest_text: str = configured_manifest()
    """Rendered the fixed release contract used by the fixture and extension."""

    (root / "release.toml").write_text(manifest_text, encoding="utf-8")
    lock_texts: dict[str, str] = {
        "Cargo.lock": "Cargo.lock fixture\n",
        "uv.lock": "uv.lock fixture\n",
    }
    """Defined the exact dependency locks embedded in the selected wheel."""

    for name, text in lock_texts.items():
        (root / name).write_text(text, encoding="utf-8")
    """Created the exact fixed inputs needed before external-evidence validation."""

    monkeypatch.setattr(
        scientific_release_runner, "configured_rules", lambda _manifest, _root: []
    )
    """Kept this test at preflight without manufacturing scientific commands."""

    wheel_path: Path = tmp_path / "asterism-0.1.0-cp313-abi3-manylinux.whl"
    """Named the sole portable artifact selected for external qualification."""

    if wheel_extension_bytes is None:
        wheel_path.write_bytes(b"arbitrary wheel bytes")
        """Modelled a false wheel selected only by its filename and suffix."""
    else:
        write_test_wheel(wheel_path, wheel_extension_bytes)
        """Created an inspectable wheel with the parameterised native payload."""

    agreement: dict[str, Any] = passing_agreement(manifest_text)
    """Built a valid cross-platform record before binding its artifact digest."""

    for artifact in agreement["artifacts"]:
        artifact["wheel_sha256"] = sha256(wheel_path)
        """Bound each platform result to the selected saved-wheel fixture."""

    agreement_path: Path = tmp_path / "agreement.json"
    """Located the external cross-platform input supplied to preflight."""

    agreement_path.write_text(json.dumps(agreement), encoding="utf-8")
    """Bound every other external input to the sole selected artifact."""

    medusa_path: Path = tmp_path / "medusa-smoke.json"
    """Located the deliberately missing or malformed Medusa input."""

    if payload == "valid":
        medusa_record: dict[str, Any] = passing_medusa_smoke(
            manifest_text,
            lock_texts["Cargo.lock"],
            lock_texts["uv.lock"],
            sha256(wheel_path),
            (
                hashlib.sha256(wheel_extension_bytes).hexdigest()
                if wheel_extension_bytes is not None
                else "0" * 64
            ),
        )
        """Bound a valid smoke to the inspectable Linux wheel member."""

        medusa_path.write_text(json.dumps(medusa_record), encoding="utf-8")
    elif payload is not None:
        medusa_path.write_text(payload, encoding="utf-8")
    core_path: Path = tmp_path / "site-packages" / "asterism" / "_core.so"
    """Located a stand-in installed extension outside the checkout fixture."""

    core_path.parent.mkdir(parents=True)
    core_path.write_bytes(b"local extension bytes")
    core: Any = SimpleNamespace(
        __file__=str(core_path),
        __release_manifest__=manifest_text,
        __cargo_lock__=lock_texts["Cargo.lock"],
        __uv_lock__=lock_texts["uv.lock"],
        __source_commit__="a" * 40,
        __source_dirty__=False,
        __version__="0.1.0",
    )
    """Represented the separately installed local wheel outside the checkout."""

    output_directory: Path = tmp_path / "release-evidence"
    """Named the directory preflight must leave absent after rejection."""

    with pytest.raises(ReleaseConfigurationError, match=expected):
        run_scientific_release(
            manifest_path=root / "release.toml",
            output_directory=output_directory,
            wheel_paths=[wheel_path],
            cross_platform_agreement_path=agreement_path,
            medusa_smoke_path=medusa_path,
            core=core,
        )
    assert not output_directory.exists()


def test_receipt_verifier_recomputes_its_conditions_from_the_fit_record() -> None:
    """Reject internally self-consistent claims built from incomplete fits."""
    complete: dict[str, Any] = {
        "converged": True,
        "h2": 0.5,
        "interval": {"lower": 0.2, "upper": 0.8, "profile_failures": 0},
    }
    """Built the smallest finite converged record satisfying one promised field."""

    assert fit_record_failures(complete, ["h2"]) == []
    assert "fit did not converge" in fit_record_failures(
        {**complete, "converged": False}, ["h2"]
    )
    assert "required quantity h2 is missing" in fit_record_failures(
        {**complete, "h2": None}, ["h2"]
    )
    assert "interval.profile_failures is 2" in fit_record_failures(
        {
            **complete,
            "interval": {"lower": 0.2, "upper": 0.8, "profile_failures": 2},
        },
        ["h2"],
    )
    assert "interval.upper is not finite" in fit_record_failures(
        {
            **complete,
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
    "outcome": "fitted",
    "build": build,
    "release_state": {
        "release_ready": True,
        "supported_quantities": ["h2"],
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
    "facts": {
        "converged": True,
        "all_quantities_present": True,
        "all_values_finite": True,
        "all_profiles_evaluated": True,
    },
    "failures": [],
    "refusal": None,
}
receipt_path = args.output / "one_trait_gaussian_heritability.json"
payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\\n").encode()
receipt_path.write_bytes(payload)
index = {
    "schema_version": 1,
    "receipts": [{
        "analysis_id": "one_trait_gaussian_heritability",
        "outcome": "fitted",
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

    linux_extension_bytes: bytes = b"linux extension bytes"
    """Fixed the native payload extracted from the selected manylinux wheel."""

    write_test_wheel(wheel_path, linux_extension_bytes)
    """Created an inspectable wheel whose exact bytes bind the evidence."""

    agreement: dict[str, Any] = passing_agreement(manifest_text)
    """Created a real complete comparison record for the same fixed build."""

    for artifact in agreement["artifacts"]:
        artifact["wheel_sha256"] = sha256(wheel_path)
        """Bound every simulated target to the sole fixture wheel artifact."""

    agreement_path: Path = tmp_path / "cross-platform-agreement.json"
    """Selected the downloaded comparison record consumed by the release runner."""

    agreement_path.write_text(json.dumps(agreement, indent=2) + "\n", encoding="utf-8")
    """Persisted the actual comparison details before scientific evidence began."""

    medusa_smoke: dict[str, Any] = passing_medusa_smoke(
        manifest_text,
        lock_texts["Cargo.lock"],
        lock_texts["uv.lock"],
        sha256(wheel_path),
        hashlib.sha256(linux_extension_bytes).hexdigest(),
    )
    """Created a converged target-host result from the same final Linux wheel."""

    medusa_smoke_path: Path = tmp_path / "medusa-smoke.json"
    """Selected the external result passed into the release runner."""

    medusa_smoke_path.write_text(
        json.dumps(medusa_smoke, indent=2) + "\n", encoding="utf-8"
    )
    """Persisted exact external bytes without placing them in the checkout."""

    core_path: Path = tmp_path / "site-packages" / "asterism" / "_core.so"
    """Selected the locally imported extension outside the checkout."""

    core_path.parent.mkdir(parents=True)
    local_extension_bytes: bytes = b"local extension bytes"
    """Fixed platform-specific bytes deliberately different from the Linux member."""

    core_path.write_bytes(local_extension_bytes)
    """Created the local extension binary whose own digest identifies the build."""

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
        medusa_smoke_path=medusa_smoke_path,
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
    assert saved["medusa_smoke"]["record"] == medusa_smoke
    copied_medusa_smoke: Path = output_directory / saved["medusa_smoke"]["path"]
    """Located the exact target-host record copied into release evidence."""

    assert saved["medusa_smoke"]["sha256"] == sha256(copied_medusa_smoke)
    assert (
        medusa_smoke["extension_sha256"]
        == hashlib.sha256(linux_extension_bytes).hexdigest()
    )
    assert medusa_smoke["extension_sha256"] != sha256(core_path)

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

    wrong_medusa_extension: dict[str, Any] = deepcopy(medusa_smoke)
    """Copied the smoke before detaching only its native-module commitment."""

    wrong_medusa_extension["extension_sha256"] = "0" * 64
    """Claimed executable bytes different from the selected wheel member."""

    wrong_medusa_extension_path: Path = output_directory / "wrong-medusa-extension.json"
    """Selected a retained record whose own outer digest remains self-consistent."""

    wrong_medusa_extension_path.write_text(
        json.dumps(wrong_medusa_extension, indent=2) + "\n", encoding="utf-8"
    )
    """Persisted the altered record so the verifier must inspect its claim."""

    wrong_medusa_extension_evidence: dict[str, Any] = deepcopy(saved)
    """Copied valid evidence before rebinding it to the altered smoke bytes."""

    wrong_medusa_extension_evidence["medusa_smoke"] = {
        "path": wrong_medusa_extension_path.name,
        "sha256": sha256(wrong_medusa_extension_path),
        "record": wrong_medusa_extension,
    }
    """Kept path, outer digest and embedded record mutually consistent."""

    wrong_medusa_extension_evidence_path: Path = (
        output_directory / "wrong-medusa-extension-evidence.json"
    )
    """Selected an independent evidence document for final verification."""

    wrong_medusa_extension_evidence_path.write_text(
        json.dumps(wrong_medusa_extension_evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    """Persisted the self-consistent evidence carrying the false native digest."""

    wrong_extension_errors: list[str] = release_evidence_errors(
        evidence_path=wrong_medusa_extension_evidence_path,
        wheel_paths=[wheel_path],
        root=root,
        core=core,
    )
    """Forced the final checker to recompute the selected wheel member itself."""

    assert "extension SHA-256 does not match selected Linux wheel" in "\n".join(
        wrong_extension_errors
    )

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

    wrong_medusa_digest: dict[str, Any] = deepcopy(saved)
    """Copied valid evidence before breaking only the retained smoke digest."""

    wrong_medusa_digest["medusa_smoke"]["sha256"] = "0" * 64
    """Detached the aggregate evidence from its exact copied target-host bytes."""

    wrong_medusa_digest_path: Path = output_directory / "wrong-medusa-digest.json"
    """Selected an independent evidence record for the stale digest claim."""

    wrong_medusa_digest_path.write_text(
        json.dumps(wrong_medusa_digest, indent=2) + "\n", encoding="utf-8"
    )

    assert "Medusa smoke SHA-256 does not match" in "\n".join(
        release_evidence_errors(
            evidence_path=wrong_medusa_digest_path,
            wheel_paths=[wheel_path],
            root=root,
            core=core,
        )
    )
    """Required the final release validator to hash the retained smoke itself."""

    failed_medusa: dict[str, Any] = deepcopy(medusa_smoke)
    """Copied the target-host record before changing its convergence result."""

    failed_medusa["converged"] = False
    """Changed only the scientific convergence result in the retained record."""

    failed_medusa_path: Path = output_directory / "failed-medusa-smoke.json"
    """Located the self-consistent but scientifically failed retained input."""

    failed_medusa_path.write_text(
        json.dumps(failed_medusa, indent=2) + "\n", encoding="utf-8"
    )
    """Created self-consistently hashed but scientifically failed smoke bytes."""

    failed_medusa_evidence: dict[str, Any] = deepcopy(saved)
    """Copied the passing evidence before replacing its Medusa commitment."""

    failed_medusa_evidence["medusa_smoke"] = {
        "path": failed_medusa_path.name,
        "sha256": sha256(failed_medusa_path),
        "record": failed_medusa,
    }
    """Rebound path, digest and embedded record to the failed smoke bytes."""

    failed_medusa_evidence_path: Path = output_directory / "failed-medusa-evidence.json"
    """Located the self-consistently hashed failed evidence fixture."""

    failed_medusa_evidence_path.write_text(
        json.dumps(failed_medusa_evidence, indent=2) + "\n", encoding="utf-8"
    )

    assert "Medusa smoke fit did not converge" in "\n".join(
        release_evidence_errors(
            evidence_path=failed_medusa_evidence_path,
            wheel_paths=[wheel_path],
            root=root,
            core=core,
        )
    )
    """Revalidated copied contents rather than trusting their embedded aggregate."""

    write_test_wheel(wheel_path, b"different linux extension bytes")
    """Replaced the artifact with a valid wheel carrying a different Linux member."""

    verification_errors: list[str] = release_evidence_errors(
        evidence_path=output_directory / "evidence.json",
        wheel_paths=[wheel_path],
        root=root,
        core=core,
    )
    """Rechecked the produced record against the artifact bytes selected for release."""

    assert "wheel SHA-256 set does not match" in "\n".join(verification_errors)
    assert "extension SHA-256 does not match selected Linux wheel" in "\n".join(
        verification_errors
    )


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

    rule_end: int = complete_manifest.index("]\n", rule_start)
    """Located the list terminator independently of its evolving rule schema."""

    manifest_text: str = (
        complete_manifest[:rule_start]
        + "pass_rules = []"
        + complete_manifest[rule_end + 1 :]
    )
    """Removed the sole executable rule while retaining its required-check name."""

    manifest_path: Path = root / "release.toml"
    """Selected the incomplete authoritative contract for this check."""

    manifest_path.write_text(manifest_text, encoding="utf-8")
    """Wrote the incomplete manifest for a fail-closed check."""

    core_path: Path = tmp_path / "site-packages" / "asterism" / "_core.so"
    """Selected an installed extension path outside the mutable checkout."""

    core_path.parent.mkdir(parents=True)
    core_path.write_bytes(b"fixed extension bytes")
    """Created the installed extension needed beyond the configuration check."""

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
            medusa_smoke_path=tmp_path / "medusa-smoke.json",
            core=core,
        )
    """Required configuration failure before any scientific command could run."""


def test_fixed_input_check_refuses_stale_embedded_dependency_locks() -> None:
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


@pytest.mark.skipif(os.name != "posix", reason="release targets are POSIX systems")
def test_bounded_command_terminates_its_spawned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminate a descendant worker as well as the timed-out command."""
    worker_path: Path = tmp_path / "worker.py"
    """Selected a descendant process with an observable termination handler."""

    worker_path.write_text(
        """from __future__ import annotations
import signal
import sys
import time
from pathlib import Path

ready_path = Path(sys.argv[1])
terminated_path = Path(sys.argv[2])

def terminate(_signal: int, _frame: object) -> None:
    terminated_path.write_text("terminated\\n", encoding="utf-8")

signal.signal(signal.SIGTERM, terminate)
ready_path.write_text("ready\\n", encoding="utf-8")
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    """Created a worker which records receipt of the process-group signal."""

    controller_path: Path = tmp_path / "controller.py"
    """Selected the direct child which owns the descendant worker."""

    controller_path.write_text(
        """from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

ready_path = Path(sys.argv[2])
subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]])
while not ready_path.is_file():
    time.sleep(0.01)
print("descendant ready", flush=True)
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    """Created a controller which would leave its worker behind if killed alone."""

    ready_path: Path = tmp_path / "ready.txt"
    """Selected the descendant-startup handshake used to avoid a timing guess."""

    terminated_path: Path = tmp_path / "terminated.txt"
    """Selected the descendant's durable process-group termination marker."""

    monkeypatch.setattr(
        scientific_release_runner,
        "TERMINATION_GRACE_SECONDS",
        0.1,
    )
    """Kept the test fast while requiring escalation past ignored SIGTERM."""

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_bounded_command(
            argv=[
                sys.executable,
                str(controller_path),
                str(worker_path),
                str(ready_path),
                str(terminated_path),
            ],
            cwd=tmp_path,
            timeout_seconds=2,
            env=dict(os.environ),
        )
    """Required timeout reporting only after the whole group was terminated."""

    assert "descendant ready" in str(caught.value.stdout)
    assert terminated_path.read_text(encoding="utf-8") == "terminated\n"
