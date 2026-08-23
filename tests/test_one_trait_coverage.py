"""Contract tests for participant-free one-trait coverage evidence."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

TARGET_FIXTURE: Path = (
    Path(__file__).parents[1]
    / "checks"
    / "design_fixtures"
    / "one_trait_gaussian_heritability.json"
)
"""Located the reviewed values-free target-design aggregate."""

TARGET_FIXTURE_SHA256: str = (
    "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e"
)
"""Pinned the exact reviewed fixture bytes consumed by release evidence."""


def coverage_module() -> ModuleType:
    """Load the standalone participant-free coverage command for testing."""
    path: Path = Path(__file__).parents[1] / "checks" / "one_trait_coverage.py"
    """Located the command without turning the checks directory into a package."""

    specification: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location("one_trait_coverage", path)
    )
    """Created the import specification for the standalone source file."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the module object governed by the validated specification."""

    sys.modules[specification.name] = module
    """Registered the module so public dataclasses can resolve their annotations."""

    specification.loader.exec_module(module)
    """Executed the coverage command without invoking its command-line entry point."""
    return module


def fit_record(
    *,
    converged: bool = True,
    lower: float = 0.0,
    upper: float = 1.0,
    contains_lower_bound: bool | None = True,
    contains_upper_bound: bool | None = True,
    profile_failures: int = 0,
) -> dict[str, Any]:
    """Return the smallest public fit record needed to score coverage.

    Args:
        converged: Whether the free REML fit converged.
        lower: Reported lower profile endpoint.
        upper: Reported upper profile endpoint.
        contains_lower_bound: Mixture-calibrated lower-bound verdict.
        contains_upper_bound: Mixture-calibrated upper-bound verdict.
        profile_failures: Failed constrained evaluations in the profile.

    Returns:
        Public-record-shaped values consumed by the coverage scorer.
    """
    return {
        "converged": converged,
        "interval": {
            "lower": lower,
            "upper": upper,
            "contains_lower_bound": contains_lower_bound,
            "contains_upper_bound": contains_upper_bound,
            "profile_failures": profile_failures,
        },
    }


def test_every_replicate_and_boundary_verdict_are_scored() -> None:
    """Make convergence, profile failures, and mixture membership decisive."""
    covers: Any = coverage_module().covers
    """Selected the public record scorer independently of the simulation route."""

    assert covers(fit_record(contains_lower_bound=True), 0.0) is True
    assert covers(fit_record(contains_lower_bound=False), 0.0) is False
    assert covers(fit_record(lower=-0.1, contains_lower_bound=False), 0.0) is False
    assert covers(fit_record(contains_upper_bound=True), 1.0) is True
    assert covers(fit_record(contains_upper_bound=False), 1.0) is False
    assert covers(fit_record(upper=1.1, contains_upper_bound=False), 1.0) is False
    assert covers(fit_record(lower=0.2, upper=0.8), 0.5) is True
    assert covers(fit_record(converged=False), 0.5) is False
    assert covers(fit_record(profile_failures=1), 0.5) is False


def test_clopper_pearson_and_asymmetric_acceptance_are_pre_written() -> None:
    """Reject undercoverage while naming known near-boundary conservatism."""
    module: ModuleType = coverage_module()
    """Loaded the standalone coverage decision functions."""

    assert module.clopper_pearson(0, 10) == pytest.approx((0.0, 0.3084971078187608))
    assert module.clopper_pearson(10, 10) == pytest.approx((0.6915028921812392, 1.0))

    anti_conservative: dict[str, object] = module.coverage_decision(7_200, 8_000, 0.3)
    """Scored a cell whose exact uncertainty lies wholly below the safe band."""

    assert anti_conservative["passed"] is False
    assert anti_conservative["verdict"] == "anti_conservative"

    historically_conservative: dict[str, object] = module.coverage_decision(
        7_920, 8_000, 0.05
    )
    """Scored the documented small-heritability conservative regime."""

    assert historically_conservative["passed"] is True
    assert historically_conservative["verdict"] == "historically_conservative"

    compatible: dict[str, object] = module.coverage_decision(7_600, 8_000, 0.3)
    """Scored ordinary nominal coverage whose uncertainty overlaps the band."""

    assert compatible["passed"] is True
    assert compatible["verdict"] == "compatible"


def test_standard_envelope_ports_relationships_and_six_column_design() -> None:
    """Retain the Rust campaign's pedigree coefficients and fixed effects."""
    module: ModuleType = coverage_module()
    """Loaded the standalone synthetic-design constructor."""

    envelope: Any = module.standard_envelope(2)
    """Built two unrelated copies of the fixed 14-person extended family."""

    assert envelope.relationship.shape == (28, 28)
    assert envelope.fixed_effects.shape == (28, 6)
    assert envelope.component_sizes == (14, 14)
    assert envelope.relationship[0, 0] == 1.0
    assert envelope.relationship[0, 1] == 0.0
    assert envelope.relationship[0, 2] == 0.5
    assert envelope.relationship[2, 3] == 0.5
    assert envelope.relationship[0, 8] == 0.25
    assert envelope.relationship[3, 8] == 0.25
    assert envelope.relationship[8, 10] == 0.125
    assert not envelope.relationship[:14, 14:].any()


def test_coverage_command_requires_every_scientific_argument() -> None:
    """Prevent defaults or ambient state from selecting a release campaign."""
    module: ModuleType = coverage_module()
    """Loaded the command-line parser without starting a simulation."""

    with pytest.raises(SystemExit):
        module.parse_arguments([])

    arguments: Any = module.parse_arguments(
        [
            "--replicates",
            "2",
            "--families",
            "1",
            "--workers",
            "1",
            "--truths",
            "0",
            "0.5",
            "1",
            "--no-write",
        ]
    )
    """Read one deliberately small but fully explicit scientific command."""

    assert arguments.replicates == 2
    assert arguments.families == 1
    assert arguments.workers == 1
    assert arguments.truths == [0.0, 0.5, 1.0]
    assert arguments.no_write is True


def test_target_fixture_builds_the_reviewed_structure_matched_pedigree() -> None:
    """Bind target coverage to reviewed aggregates and a synthetic pedigree."""
    module: ModuleType = coverage_module()
    """Loaded the target-fixture verifier and deterministic generator."""

    assert module.fixture_sha256(TARGET_FIXTURE) == TARGET_FIXTURE_SHA256

    envelope: Any = module.target_envelope(
        TARGET_FIXTURE,
        expected_sha256=TARGET_FIXTURE_SHA256,
        families=202,
    )
    """Built the historical structure-matched pedigree from aggregate facts only."""

    assert envelope.relationship.shape == (1_909, 1_909)
    assert envelope.fixed_effects.shape == (1_909, 6)
    assert len(envelope.component_sizes) == 202
    assert sum(envelope.component_sizes) == 1_909
    assert max(envelope.component_sizes) == 180
    assert envelope.nonzero_relationship_pairs == 27_691
    assert envelope.observed_nonzero_relationship_pairs == 27_821
    assert envelope.relative_pair_count_discrepancy == pytest.approx(130 / 27_821)
    assert envelope.relative_pair_count_discrepancy <= 0.005


def test_real_public_fit_cell_is_deterministic_and_scores_every_attempt() -> None:
    """Run a tiny real cell and retain every estimator/profile outcome."""
    module: ModuleType = coverage_module()
    """Loaded the public-Python coverage cell runner."""

    job: Any = module.CellJob(
        truth=0.5,
        truth_index=1,
        replicates=2,
        families=1,
        fixture_path=None,
        fixture_sha256=None,
    )
    """Selected a two-replicate participant-free integration cell."""

    first: dict[str, object] = module.run_cell(job)
    """Executed both fits through the installed public Python interface."""

    second: dict[str, object] = module.run_cell(job)
    """Repeated the identical cell to test deterministic stream ownership."""

    deterministic_fields: tuple[str, ...] = (
        "seed",
        "attempted",
        "covered",
        "coverage",
        "refused",
        "nonconverged",
        "profile_failed_replicates",
        "profile_failure_evaluations",
        "complete_profiles",
        "fraction_at_zero",
        "fraction_at_one",
        "verdict",
        "passed",
    )
    """Excluded elapsed time while retaining every scientific decision field."""

    assert {field: first[field] for field in deterministic_fields} == {
        field: second[field] for field in deterministic_fields
    }
    assert first["attempted"] == 2
    assert (
        first["refused"]
        + first["nonconverged"]
        + first["profile_failed_replicates"]
        + first["complete_profiles"]
        == first["attempted"]
    )


def test_small_complete_command_emits_auditable_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose explicit coordinates and every cell decision in one record."""
    module: ModuleType = coverage_module()
    """Loaded the complete participant-free command entry point."""

    exit_code: int = module.main(
        [
            "--replicates",
            "1",
            "--families",
            "1",
            "--workers",
            "1",
            "--truths",
            "0",
            "0.5",
            "1",
            "--no-write",
        ]
    )
    """Ran a minimal boundary-and-interior smoke through real public fits."""

    evidence: dict[str, object] = json.loads(capsys.readouterr().out)
    """Parsed the sole standard-output evidence record."""

    assert exit_code == 0
    assert evidence["participant_free"] is True
    assert evidence["replicates_per_cell"] == 1
    assert evidence["families"] == 1
    assert evidence["workers"] == 1
    assert evidence["truths"] == [0.0, 0.5, 1.0]
    assert evidence["every_replicate_scored"] is True
    assert evidence["boundary_membership"] == "contains_lower_bound_or_upper_bound"
    assert len(evidence["cells"]) == 3
    assert all(cell["attempted"] == 1 for cell in evidence["cells"])
    assert evidence["passed"] is True


def test_release_manifest_pins_both_deferred_one_trait_campaigns() -> None:
    """Make both rules runnable without promoting unrun release evidence."""
    manifest: dict[str, object] = tomllib.loads(
        (Path(__file__).parents[1] / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the authoritative development release contract."""

    analysis: dict[str, object] = next(
        selected
        for selected in manifest["analyses"]
        if selected["id"] == "one_trait_gaussian_heritability"
    )
    """Selected the one-trait analysis without assuming manifest position."""

    rules: dict[str, dict[str, object]] = {
        rule["id"]: rule for rule in analysis["pass_rules"]
    }
    """Indexed the exact scientific rules by their stable identifiers."""

    truth_arguments: list[str] = [
        "0",
        "0.05",
        "0.07",
        "0.10",
        "0.20",
        "0.30",
        "0.40",
        "0.50",
        "0.60",
        "0.70",
        "0.80",
        "1",
    ]
    """Pinned the boundary, near-boundary, and interior release truth grid."""

    common: list[str] = [
        "checks/one_trait_coverage.py",
        "--replicates",
        "8000",
    ]
    """Pinned the full release denominator rather than the reduced smoke count."""

    assert rules["coverage"]["status"] == "ready"
    assert rules["coverage"]["command"] == [
        *common,
        "--families",
        "100",
        "--workers",
        "12",
        "--truths",
        *truth_arguments,
        "--no-write",
    ]
    assert rules["coverage_at_the_real_design_point"]["status"] == "ready"
    assert rules["coverage_at_the_real_design_point"]["command"] == [
        *common,
        "--families",
        "202",
        "--workers",
        "12",
        "--truths",
        *truth_arguments,
        "--design-fixture",
        "checks/design_fixtures/one_trait_gaussian_heritability.json",
        "--fixture-sha256",
        TARGET_FIXTURE_SHA256,
        "--no-write",
    ]
    assert manifest["scientific_pass_rules_configured"] is False
    assert analysis["pass_rules_configured"] is False
    assert analysis["design_range"]["measured"] is False
