"""Contracts for participant-free liability calibration evidence."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the standalone scientific command."""

TARGET_FIXTURE: Path = (
    ROOT / "checks" / "design_fixtures" / "liability_heritability_target_design.json"
)
"""Located the reviewed values-free liability target-design fixture."""

TARGET_FIXTURE_SHA256: str = (
    "863eedc4f18063bca27754e782b9ebb5faa341359912b034163a7ea2067c20aa"
)
"""Pinned the exact reviewed fixture bytes independently of their contents."""


def calibration_module() -> ModuleType:
    """Load the liability command without invoking its command-line entry point.

    Returns:
        Loaded standalone module registered for dataclass annotation resolution.
    """
    path: Path = ROOT / "checks" / "liability_calibration.py"
    """Located the maintained liability calibration command."""

    specification: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location("liability_calibration", path)
    )
    """Created the import specification for the standalone source file."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the module governed by the validated specification."""

    sys.modules[specification.name] = module
    """Registered the module before evaluating public dataclass annotations."""

    checks_directory: str = str(path.parent)
    """Located sibling check modules imported by the standalone command."""

    sys.path.insert(0, checks_directory)
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(checks_directory)
    """Loaded declarations with the same sibling path available to direct CLI use."""
    return module


def test_reviewed_fixture_builds_only_a_structure_matched_target() -> None:
    """Bind the approximation stress to values-free aggregate structure."""
    module: ModuleType = calibration_module()
    """Loaded the exact-byte target-fixture verifier."""

    assert module.fixture_sha256(TARGET_FIXTURE) == TARGET_FIXTURE_SHA256

    target: object = module.load_target_design(
        TARGET_FIXTURE,
        expected_sha256=TARGET_FIXTURE_SHA256,
    )
    """Built a synthetic target only after both fixture digests were verified."""

    assert target.relationship.shape == (1_909, 1_909)
    assert target.design.shape == (1_909, 6)
    assert len(target.component_sizes) == 202
    assert sum(target.component_sizes) == 1_909
    assert max(target.component_sizes) == 180
    assert target.generated_nonzero_relationship_pairs == 27_691
    assert target.observed_nonzero_relationship_pairs == 27_821
    assert target.relative_nonzero_pair_discrepancy == pytest.approx(130 / 27_821)
    assert target.fixture_sha256 == TARGET_FIXTURE_SHA256
    assert target.source_fixture_sha256 == (
        "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e"
    )


def test_target_fixture_refuses_the_wrong_exact_digest() -> None:
    """Prevent content edits from silently selecting a different target."""
    module: ModuleType = calibration_module()
    """Loaded the exact-byte target-fixture verifier."""

    with pytest.raises(ValueError, match="LIABILITY_TARGET_FIXTURE_SHA256_MISMATCH"):
        module.load_target_design(TARGET_FIXTURE, expected_sha256="0" * 64)


def test_command_requires_every_scientific_coordinate() -> None:
    """Keep release and smoke campaigns independent of ambient defaults."""
    module: ModuleType = calibration_module()
    """Loaded the explicit command-line parser."""

    with pytest.raises(SystemExit):
        module.parse_arguments([])

    arguments: object = module.parse_arguments(
        [
            "--replicates",
            "2",
            "--workers",
            "1",
            "--prevalence",
            "0.254",
            "--true-heritability",
            "0.25",
            "--levels",
            "0.01",
            "0.05",
            "0.10",
            "--nominal-coverage",
            "0.95",
            "--monte-carlo-confidence",
            "0.95",
            "--base-seed",
            "830000",
            "--scenario-seed-offset",
            "1000003",
            "--minimum-cases",
            "10",
            "--design-fixture",
            str(TARGET_FIXTURE),
            "--fixture-sha256",
            TARGET_FIXTURE_SHA256,
            "--no-write",
        ]
    )
    """Read one small but fully explicit participant-free campaign."""

    assert arguments.replicates == 2
    assert arguments.workers == 1
    assert arguments.levels == [0.01, 0.05, 0.10]
    assert arguments.no_write is True


def test_exact_monte_carlo_rules_are_one_sided_and_pre_written() -> None:
    """Fail only anti-conservative level or under-covering interval evidence."""
    module: ModuleType = calibration_module()
    """Loaded the exact binomial acceptance functions."""

    assert module.one_sided_lower_limit(20, 100, 0.95) == pytest.approx(
        0.13666132524586722
    )
    assert module.one_sided_upper_limit(90, 100, 0.95) == pytest.approx(
        0.94473676231713
    )

    assert (
        module.level_decision(
            rejections=20,
            attempted=100,
            level=0.05,
            confidence=0.95,
        )["passed"]
        is False
    )
    assert (
        module.level_decision(
            rejections=0,
            attempted=100,
            level=0.05,
            confidence=0.95,
        )["passed"]
        is True
    )
    assert (
        module.level_decision(
            rejections=1,
            attempted=1,
            level=0.05,
            confidence=0.95,
        )["passed"]
        is True
    )
    """Accepted exact boundary equality despite floating quantile round-off."""

    assert (
        module.coverage_decision(
            covered=90,
            attempted=100,
            nominal=0.95,
            confidence=0.95,
        )["passed"]
        is False
    )
    assert (
        module.coverage_decision(
            covered=95,
            attempted=100,
            nominal=0.95,
            confidence=0.95,
        )["passed"]
        is True
    )


def test_campaign_scoring_keeps_failures_in_both_denominators() -> None:
    """Count refusals and failed profiles as attempted scientific misses."""
    module: ModuleType = calibration_module()
    """Loaded the unconditional campaign scorer."""

    rows: list[object] = [
        module.ReplicateResult(
            scenario="null",
            replicate=0,
            seed=1,
            outcome="complete",
            p_value=0.04,
        ),
        module.ReplicateResult(
            scenario="null",
            replicate=1,
            seed=2,
            outcome="refused",
        ),
        module.ReplicateResult(
            scenario="heritable",
            replicate=0,
            seed=3,
            outcome="complete",
            covered=True,
        ),
        module.ReplicateResult(
            scenario="heritable",
            replicate=1,
            seed=4,
            outcome="profile_failed",
            covered=False,
            profile_failures=1,
        ),
    ]
    """Represented successful, refused, and failed-profile attempts explicitly."""

    scored: dict[str, object] = module.score_campaign(
        rows,
        levels=(0.05,),
        nominal_coverage=0.95,
        monte_carlo_confidence=0.95,
        maximum_failed_replicates=0,
    )
    """Applied the frozen rules without filtering unavailable results."""

    level: dict[str, object] = scored["level"]
    """Selected the complete null-test accounting record."""

    coverage: dict[str, object] = scored["coverage"]
    """Selected the complete heritable-interval accounting record."""

    assert level["attempted"] == 2
    assert level["computed"] == 1
    assert level["rejection"]["0.05"]["rejections"] == 1
    assert level["rejection"]["0.05"]["rate"] == 0.5
    assert coverage["attempted"] == 2
    assert coverage["computed"] == 1
    assert coverage["covered"] == 1
    assert coverage["coverage"] == 0.5
    assert scored["failed_replicates"] == 2
    assert scored["passed"] is False


def test_replicates_use_only_the_public_liability_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise fit, test, and interval through the documented model seam."""
    module: ModuleType = calibration_module()
    """Loaded the worker boundary independently of process orchestration."""

    calls: list[str] = []
    """Recorded every public model operation in invocation order."""

    class RecordingLiabilityModel:
        """Return minimal valid public records while recording method use."""

        def __init__(
            self,
            relationship: npt.NDArray[np.float64],
            status: npt.NDArray[np.float64],
            design: npt.NDArray[np.float64],
        ) -> None:
            assert relationship.shape == (4, 4)
            assert status.shape == (4,)
            assert design.shape == (4, 1)
            calls.append("initialise")
            """Confirmed the generated arrays reached the public constructor."""

        def fit(self) -> dict[str, object]:
            """Return one converged public fit record."""
            calls.append("fit")
            """Recorded explicit convergence assessment before inference."""
            return {"converged": True}

        def test(self) -> dict[str, object]:
            """Return one valid boundary-reference test record."""
            calls.append("test")
            """Recorded use of the public liability test."""
            return {
                "p_value": 0.5,
                "statistic": 0.4,
                "rule": "mixture_50_50",
            }

        def interval(self) -> dict[str, object]:
            """Return one complete public profile interval record."""
            calls.append("interval")
            """Recorded use of the public liability interval."""
            return {
                "estimate": 0.3,
                "lower": 0.1,
                "upper": 0.5,
                "lower_limited": False,
                "upper_limited": False,
                "profile_failures": 0,
                "level": 0.95,
            }

    monkeypatch.setattr(module.asterism, "LiabilityModel", RecordingLiabilityModel)
    """Replaced only the documented constructor with a recording implementation."""

    def fixed_status(
        state: object,
        job: object,
    ) -> npt.NDArray[np.float64]:
        """Return a case-balanced status independent of the random stream."""
        del state, job
        return np.array([0.0, 1.0, 0.0, 1.0])

    monkeypatch.setattr(module, "simulate_status", fixed_status)
    """Held outcome generation fixed so this test isolates the public seam."""

    state: object = module.WorkerState(
        relationship=np.eye(4),
        design=np.ones((4, 1)),
        component_sizes=(4,),
        factors={"null": (np.eye(4),), "heritable": (np.eye(4),)},
        threshold=0.0,
        true_heritability=0.25,
        nominal_coverage=0.95,
        minimum_cases=1,
    )
    """Built the smallest deterministic worker state for both scenarios."""

    module.start_worker(state)
    """Installed state through the same initializer used by process workers."""

    null_result: object = module.run_replicate(module.ReplicateJob("null", 0, 10))
    """Exercised convergence and the public boundary test."""

    heritable_result: object = module.run_replicate(
        module.ReplicateJob("heritable", 0, 20)
    )
    """Exercised convergence and the public profile interval."""

    assert null_result.outcome == "complete"
    assert null_result.p_value == 0.5
    assert heritable_result.outcome == "complete"
    assert heritable_result.covered is True
    assert calls == [
        "initialise",
        "fit",
        "test",
        "initialise",
        "fit",
        "interval",
    ]


def test_public_worker_classifies_every_unavailable_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain refusal, convergence, interval, and profile failure attempts."""
    module: ModuleType = calibration_module()
    """Loaded the public-model worker boundary."""

    class ControlledLiabilityModel:
        """Return one selected public failure mode for each worker call."""

        mode: str = "refused"
        """Selected the failure record returned by this test double."""

        def __init__(
            self,
            relationship: object,
            status: object,
            design: object,
        ) -> None:
            del relationship, status, design

        def fit(self) -> dict[str, object]:
            """Refuse, fail convergence, or permit the inference call."""
            if self.mode == "refused":
                raise ValueError("LIABILITY_INPUT_REFUSED")
            return {"converged": self.mode != "nonconverged"}

        def test(self) -> dict[str, object]:
            """Refuse the null test selected for this branch."""
            raise ValueError("LIABILITY_NULL_FIT_NOT_CONVERGED")

        def interval(self) -> dict[str, object]:
            """Refuse profiling or return a profile with one failed evaluation."""
            if self.mode == "interval_failed":
                raise ValueError("LIABILITY_INTERVAL_UNAVAILABLE")
            if self.mode == "interval_missing":
                return {}
            return {
                "estimate": 0.3,
                "lower": 0.1,
                "upper": 0.5,
                "lower_limited": False,
                "upper_limited": False,
                "profile_failures": 1,
                "level": 0.95,
            }

    monkeypatch.setattr(module.asterism, "LiabilityModel", ControlledLiabilityModel)
    """Selected controlled failures only at the documented public constructor."""

    def fixed_status(
        state: object,
        job: object,
    ) -> npt.NDArray[np.float64]:
        """Return a case-balanced status so only inference can fail."""
        del state, job
        return np.array([0.0, 1.0, 0.0, 1.0])

    monkeypatch.setattr(module, "simulate_status", fixed_status)
    """Held participant-free generation fixed across controlled failure modes."""

    state: object = module.WorkerState(
        relationship=np.eye(4),
        design=np.ones((4, 1)),
        component_sizes=(4,),
        factors={"null": (np.eye(4),), "heritable": (np.eye(4),)},
        threshold=0.0,
        true_heritability=0.25,
        nominal_coverage=0.95,
        minimum_cases=1,
    )
    """Built one deterministic worker state for all classification branches."""

    module.start_worker(state)
    """Installed the state through the production worker initializer."""

    branches: tuple[tuple[str, str, str], ...] = (
        ("null", "refused", "refused"),
        ("null", "nonconverged", "nonconverged"),
        ("null", "test_failed", "test_failed"),
        ("heritable", "interval_failed", "interval_failed"),
        ("heritable", "interval_missing", "interval_failed"),
        ("heritable", "profile_failed", "profile_failed"),
    )
    """Enumerated every unavailable public inference named by the gate."""

    outcomes: list[str] = []
    """Collected the worker classification for each retained attempt."""

    for replicate, (scenario, mode, expected) in enumerate(branches):
        ControlledLiabilityModel.mode = mode
        """Selected one estimator or inference failure without changing data."""

        result: object = module.run_replicate(
            module.ReplicateJob(scenario, replicate, replicate + 1)
        )
        """Ran one attempt through the public-model classification boundary."""

        assert result.outcome == expected
        if scenario == "heritable":
            assert result.covered is False
        outcomes.append(result.outcome)
    """Verified every failure stayed visible rather than returning no row."""

    assert outcomes == [
        "refused",
        "nonconverged",
        "test_failed",
        "interval_failed",
        "interval_failed",
        "profile_failed",
    ]


def test_release_rule_is_the_exact_frozen_executable_command() -> None:
    """Keep only this rule ready while all readiness measurements stay false."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the authoritative scientific command inventory."""

    analysis: dict[str, object] = next(
        entry
        for entry in manifest["analyses"]
        if entry["id"] == "binary_liability_heritability"
    )
    """Selected the supported liability analysis without relying on its position."""

    rule: dict[str, object] = next(
        entry
        for entry in analysis["pass_rules"]
        if entry["id"] == "liability_calibration"
    )
    """Selected the sole target calibration rule by stable identifier."""

    assert rule["status"] == "ready"
    assert "blocker" not in rule
    assert rule["timeout_seconds"] == 21_600
    assert rule["command"] == [
        "checks/liability_calibration.py",
        "--replicates",
        "200",
        "--workers",
        "6",
        "--prevalence",
        "0.254",
        "--true-heritability",
        "0.25",
        "--levels",
        "0.01",
        "0.05",
        "0.10",
        "--nominal-coverage",
        "0.95",
        "--monte-carlo-confidence",
        "0.95",
        "--base-seed",
        "830000",
        "--scenario-seed-offset",
        "1000003",
        "--minimum-cases",
        "10",
        "--design-fixture",
        "checks/design_fixtures/liability_heritability_target_design.json",
        "--fixture-sha256",
        TARGET_FIXTURE_SHA256,
        "--no-write",
    ]
    assert analysis["pass_rules_configured"] is False
    assert analysis["design_range"]["measured"] is False
