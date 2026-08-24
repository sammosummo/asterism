"""Public-seam contracts for the discrete-GxE scientific release checks."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing standalone scientific commands."""

TARGET_FIXTURE: Path = (
    ROOT / "checks" / "design_fixtures" / "discrete_gxe_target_design.json"
)
"""Located the reviewed values-free discrete-GxE target aggregate."""


def check_module(name: str) -> ModuleType:
    """Load one standalone check without executing its command-line entry point.

    Args:
        name: Check filename without its ``.py`` suffix.

    Returns:
        Loaded module registered for public dataclass annotation resolution.
    """
    path: Path = ROOT / "checks" / f"{name}.py"
    """Located the named maintained check script."""

    specification: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location(name, path)
    )
    """Created the import specification for the standalone command."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the module object governed by the resolved specification."""

    sys.modules[specification.name] = module
    """Registered the module before evaluating public dataclass annotations."""

    specification.loader.exec_module(module)
    """Loaded declarations without invoking the command-line entry point."""
    return module


def test_independent_dense_fit_agrees_through_the_public_model() -> None:
    """Compare an interior negative-correlation fit without shared numerics."""
    module: ModuleType = check_module("discrete_gxe_against_independent_full_fit")
    """Loaded the participant-free independent comparison command."""

    record: dict[str, object] = module.run_comparison()
    """Ran the public Asterism fit and separately optimized dense likelihood."""

    assert record["participant_free"] is True
    assert record["public_interface"] == "asterism.DiscreteGxeModel.fit"
    assert record["independent_parameterisation"] == "genetic_cholesky_and_log_residual"
    assert record["independent_optimizer"] == "scipy_powell_derivative_free"
    assert record["truth_genetic_correlation"] < 0.0
    assert record["asterism_converged"] is True
    assert record["reference_converged"] is True
    assert record["loglik_difference"] <= record["loglik_tolerance"]
    assert max(record["absolute_differences"].values()) <= record["quantity_tolerance"]
    assert record["passed"] is True


def test_target_fixture_builds_only_a_structure_matched_synthetic_design() -> None:
    """Reproduce target aggregates without participant rows or matrices."""
    module: ModuleType = check_module("discrete_gxe_calibration")
    """Loaded the portable public-API calibration command."""

    envelope: object = module.target_envelope(
        TARGET_FIXTURE,
        expected_sha256=module.TARGET_FIXTURE_SHA256,
    )
    """Built a synthetic design only after exact fixture validation."""

    assert envelope.relationship.shape == (1_910, 1_910)
    assert envelope.design.shape == (1_910, 4)
    assert envelope.group_counts == (758, 1_152)
    assert len(envelope.component_sizes) == 203
    assert max(envelope.component_sizes) == 180
    assert envelope.nonzero_relationship_pairs == 27_691
    assert envelope.observed_nonzero_relationship_pairs == 27_821
    assert envelope.cross_group_nonzero_relationship_pairs == 13_322
    assert envelope.observed_cross_group_nonzero_relationship_pairs == 13_322
    assert envelope.mixed_group_components == 77
    assert envelope.fixture_sha256 == module.TARGET_FIXTURE_SHA256


def test_target_fixture_refuses_a_loosened_calibration_acceptance(
    tmp_path: Path,
) -> None:
    """Bind the synthetic target design to its prewritten decision contract."""
    module: ModuleType = check_module("discrete_gxe_calibration")
    """Loaded the strict aggregate fixture verifier."""

    fixture: dict[str, object] = json.loads(TARGET_FIXTURE.read_text(encoding="utf-8"))
    """Parsed the genuine reviewed fixture before one intentional mutation."""

    fixture["calibration_acceptance"]["maximum_refused_or_nonconverged"] = 1
    """Loosened the zero-unsuccessful-attempt release rule."""

    tampered: Path = tmp_path / "discrete_gxe_target_design.json"
    """Kept malformed evidence outside the versioned fixture directory."""

    tampered.write_text(json.dumps(fixture), encoding="utf-8")
    """Persisted the structurally valid but scientifically altered fixture."""

    with pytest.raises(ValueError, match="ACCEPTANCE_INVALID"):
        module.target_envelope(
            tampered,
            expected_sha256=module.fixture_sha256(tampered),
        )


def test_target_campaign_has_no_database_or_participant_row_dependency() -> None:
    """Keep release calibration portable and aggregate-only by construction."""
    source: str = (ROOT / "checks" / "discrete_gxe_calibration.py").read_text(
        encoding="utf-8"
    )
    """Read the executable command without importing scientific dependencies."""

    assert "sqlite3" not in source
    assert "SAFS.db" not in source

    fixture: dict[str, object] = json.loads(TARGET_FIXTURE.read_text(encoding="utf-8"))
    """Parsed the committed aggregate-only structural envelope."""

    assert fixture["participant_free"] is True
    assert fixture["reviewed"] is True

    def nested_keys(value: object) -> set[str]:
        """Collect fixture field names without interpreting retained values."""
        if isinstance(value, dict):
            return set(value) | {
                key for nested in value.values() for key in nested_keys(nested)
            }
        if isinstance(value, list):
            return {key for nested in value for key in nested_keys(nested)}
        return set()

    assert {
        "ids",
        "subject_ids",
        "rows",
        "ages",
        "phenotypes",
        "relationship_matrix",
    }.isdisjoint(nested_keys(fixture))
    assert max(len(value) for value in fixture.values() if isinstance(value, list)) <= 3


def test_calibration_command_requires_every_scientific_coordinate() -> None:
    """Prevent defaults or ambient state from selecting the release campaign."""
    module: ModuleType = check_module("discrete_gxe_calibration")
    """Loaded the portable calibration argument parser."""

    with pytest.raises(SystemExit):
        module.parse_arguments([])

    arguments: object = module.parse_arguments(
        [
            "--replicates",
            "2",
            "--workers",
            "1",
            "--fixture",
            str(TARGET_FIXTURE),
            "--fixture-sha256",
            module.TARGET_FIXTURE_SHA256,
            "--levels",
            "0.01",
            "0.05",
            "0.1",
            "--scenarios",
            "null",
            "noisier",
            "--tests",
            "gene_by_environment",
            "correlation",
            "--no-write",
        ]
    )
    """Parsed a deliberately small but fully explicit scientific command."""

    assert arguments.replicates == 2
    assert arguments.workers == 1
    assert arguments.fixture == TARGET_FIXTURE
    assert arguments.fixture_sha256 == module.TARGET_FIXTURE_SHA256
    assert arguments.levels == [0.01, 0.05, 0.1]
    assert arguments.scenarios == ["null", "noisier"]
    assert arguments.tests == ["gene_by_environment", "correlation"]
    assert arguments.no_write is True


def test_calibration_acceptance_scores_missing_and_anti_conservative_runs() -> None:
    """Make every attempt and exact one-sided Monte Carlo uncertainty decisive."""
    module: ModuleType = check_module("discrete_gxe_calibration")
    """Loaded the calibration scorer independently of expensive simulation."""

    compatible: dict[str, object] = module.calibration_decision(
        rejections=0,
        computed=10,
        attempted=10,
        level=0.05,
        level_check=True,
    )
    """Scored a complete level cell with no sign of anti-conservatism."""

    assert compatible["passed"] is True
    assert compatible["verdict"] == "compatible"

    anti_conservative: dict[str, object] = module.calibration_decision(
        rejections=10,
        computed=10,
        attempted=10,
        level=0.05,
        level_check=True,
    )
    """Scored a complete cell whose exact lower bound exceeds nominal level."""

    assert anti_conservative["passed"] is False
    assert anti_conservative["verdict"] == "anti_conservative"

    incomplete: dict[str, object] = module.calibration_decision(
        rejections=0,
        computed=9,
        attempted=10,
        level=0.05,
        level_check=True,
    )
    """Refused to let one missing fit disappear from the denominator."""

    assert incomplete["passed"] is False
    assert incomplete["verdict"] == "incomplete"

    power: dict[str, object] = module.calibration_decision(
        rejections=0,
        computed=10,
        attempted=10,
        level=0.05,
        level_check=False,
    )
    """Retained power descriptively without inventing a minimum threshold."""

    assert power["passed"] is True
    assert power["verdict"] == "power_not_gated"


def test_small_calibration_command_uses_public_fits_and_scores_every_attempt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run one real public fit/test without letting failures leave the receipt."""
    module: ModuleType = check_module("discrete_gxe_calibration")
    """Loaded the executable calibration command and target-envelope type."""

    families: int = 20
    """Selected enough sibling blocks for a stable but bounded integration fit."""

    people: int = 4 * families
    """Counted rows in the participant-free smoke design."""

    relationship: np.ndarray = np.zeros((people, people), dtype=float)
    """Allocated independent four-sibling relationship components."""

    for family in range(families):
        start: int = 4 * family
        """Located this synthetic family block."""

        relationship[start : start + 4, start : start + 4] = 0.5
        """Assigned ordinary full-sibling relationships."""

        relationship[np.arange(start, start + 4), np.arange(start, start + 4)] = 1.0
        """Restored unit marginal additive relationships."""

    group: np.ndarray = np.tile(np.asarray([1.0, 1.0, 2.0, 2.0]), families)
    """Placed both environment groups inside every sibling family."""

    age: np.ndarray = np.linspace(-1.0, 1.0, people)
    """Selected a deterministic participant-free continuous covariate."""

    design: np.ndarray = np.column_stack(
        [np.ones(people), age, age * age, (group == 1.0).astype(float)]
    )
    """Reproduced the target check's four-column fixed-design identity."""

    envelope: object = module.TargetEnvelope(
        relationship=relationship,
        group=group,
        design=design,
        component_sizes=(4,) * families,
        group_counts=(2 * families, 2 * families),
        nonzero_relationship_pairs=6 * families,
        observed_nonzero_relationship_pairs=6 * families,
        cross_group_nonzero_relationship_pairs=4 * families,
        observed_cross_group_nonzero_relationship_pairs=4 * families,
        mixed_group_components=families,
        fixture_sha256=module.TARGET_FIXTURE_SHA256,
    )
    """Built a small envelope with the same public scientific structure."""

    def selected_envelope(path: Path, *, expected_sha256: str) -> object:
        """Return the bounded envelope after retaining explicit fixture inputs."""
        assert path == TARGET_FIXTURE
        assert expected_sha256 == module.TARGET_FIXTURE_SHA256
        return envelope

    monkeypatch.setattr(module, "target_envelope", selected_envelope)
    exit_code: int = module.main(
        [
            "--replicates",
            "1",
            "--workers",
            "1",
            "--fixture",
            str(TARGET_FIXTURE),
            "--fixture-sha256",
            module.TARGET_FIXTURE_SHA256,
            "--levels",
            "0.05",
            "--scenarios",
            "null",
            "--tests",
            "genetic",
            "--no-write",
        ]
    )
    """Ran one complete public ``fit`` and ``test`` campaign attempt."""

    evidence: dict[str, object] = json.loads(capsys.readouterr().out)
    """Parsed the command's sole machine-readable standard output."""

    assert exit_code == 0
    assert evidence["participant_free"] is True
    assert evidence["public_interfaces"] == [
        "asterism.DiscreteGxeModel.fit",
        "asterism.DiscreteGxeModel.test",
    ]
    assert evidence["replicates_per_scenario"] == 1
    assert evidence["every_replicate_scored"] is True
    scenario: dict[str, object] = evidence["results"]["null"]
    """Selected the one executed scientific scenario."""

    assert scenario["attempted"] == 1
    assert (
        scenario["complete"]
        + scenario["refused"]
        + scenario["nonconverged"]
        + scenario["test_refused"]
        == 1
    )
    assert evidence["passed"] is True


def test_interval_command_requires_the_documented_n600_coordinates() -> None:
    """Pin the fixed 600-person, four-sibling, correlation-truth campaign."""
    module: ModuleType = check_module("discrete_gxe_correlation_interval")
    """Loaded the portable interval-campaign argument parser."""

    with pytest.raises(SystemExit):
        module.parse_arguments([])

    arguments: object = module.parse_arguments(
        [
            "--replicates",
            "2",
            "--people",
            "600",
            "--families",
            "150",
            "--sibs-per-family",
            "4",
            "--workers",
            "1",
            "--truths",
            "0.3",
            "0.6",
            "0.9",
            "--no-write",
        ]
    )
    """Parsed a reduced-denominator campaign with the exact target design."""

    assert arguments.replicates == 2
    assert arguments.people == 600
    assert arguments.families == 150
    assert arguments.sibs_per_family == 4
    assert arguments.workers == 1
    assert arguments.truths == [0.3, 0.6, 0.9]
    assert arguments.no_write is True


def test_interval_coverage_scores_every_attempt_and_exact_uncertainty() -> None:
    """Fail incomplete or credibly anti-conservative profile campaigns."""
    module: ModuleType = check_module("discrete_gxe_correlation_interval")
    """Loaded the interval scorer independently of expensive public profiles."""

    compatible: dict[str, object] = module.coverage_decision(
        covered=380,
        complete=400,
        attempted=400,
    )
    """Scored nominal complete coverage at the center of its safe band."""

    assert compatible["passed"] is True
    assert compatible["verdict"] == "compatible"

    anti_conservative: dict[str, object] = module.coverage_decision(
        covered=350,
        complete=400,
        attempted=400,
    )
    """Scored coverage whose exact upper bound remains below 0.94."""

    assert anti_conservative["passed"] is False
    assert anti_conservative["verdict"] == "anti_conservative"

    conservative: dict[str, object] = module.coverage_decision(
        covered=400,
        complete=400,
        attempted=400,
    )
    """Retained known near-boundary conservatism as explicit passing evidence."""

    assert conservative["passed"] is True
    assert conservative["verdict"] == "conservative"

    incomplete: dict[str, object] = module.coverage_decision(
        covered=399,
        complete=399,
        attempted=400,
    )
    """Prevented one failed public profile from disappearing from coverage."""

    assert incomplete["passed"] is False
    assert incomplete["verdict"] == "incomplete"


def test_small_interval_command_profiles_publicly_and_scores_every_attempt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the complete truth grid through real public fit and interval methods."""
    module: ModuleType = check_module("discrete_gxe_correlation_interval")
    """Loaded the executable participant-free interval campaign."""

    exit_code: int = module.main(
        [
            "--replicates",
            "1",
            "--people",
            "120",
            "--families",
            "30",
            "--sibs-per-family",
            "4",
            "--workers",
            "1",
            "--truths",
            "0.3",
            "0.6",
            "0.9",
            "--no-write",
        ]
    )
    """Ran one profile per documented truth on a bounded sibling roster."""

    evidence: dict[str, object] = json.loads(capsys.readouterr().out)
    """Parsed the command's sole machine-readable evidence record."""

    assert exit_code == 0
    assert evidence["participant_free"] is True
    assert evidence["public_interfaces"] == [
        "asterism.DiscreteGxeModel.fit",
        "asterism.DiscreteGxeModel.correlation_interval",
    ]
    assert evidence["people"] == 120
    assert evidence["truths"] == [0.3, 0.6, 0.9]
    assert evidence["replicates_per_truth"] == 1
    assert evidence["attempted"] == 3
    assert evidence["every_replicate_scored"] is True
    for cell in evidence["cells"]:
        assert (
            cell["complete"]
            + cell["refused"]
            + cell["nonconverged"]
            + cell["interval_refused"]
            + cell["profile_failed"]
            == 1
        )
    assert evidence["passed"] is True


def test_manifest_pins_all_three_executable_discrete_gxe_rules() -> None:
    """Replace blocker shims with exact participant-free release commands."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the authoritative development release inventory."""

    analysis: dict[str, object] = next(
        selected
        for selected in manifest["analyses"]
        if selected["id"] == "discrete_gene_by_environment"
    )
    """Selected the supported discrete-GxE analysis by stable identifier."""

    rules: dict[str, dict[str, object]] = {
        rule["id"]: rule for rule in analysis["pass_rules"]
    }
    """Indexed the exact three required checks independently of list order."""

    assert rules["discrete_gxe_against_independent_full_fit"]["command"] == [
        "checks/discrete_gxe_against_independent_full_fit.py"
    ]
    assert rules["discrete_gxe_calibration"]["command"] == [
        "checks/discrete_gxe_calibration.py",
        "--replicates",
        "500",
        "--workers",
        "12",
        "--fixture",
        "checks/design_fixtures/discrete_gxe_target_design.json",
        "--fixture-sha256",
        "2cceae6f2c5ea3f0c3de5946daeaa55cf2f35edd6d365fd1f45499f463d4c1c6",
        "--levels",
        "0.01",
        "0.05",
        "0.1",
        "--scenarios",
        "null",
        "different_genes",
        "different_scale",
        "noisier",
        "--tests",
        "gene_by_environment",
        "any_difference",
        "correlation",
        "genetic",
        "residual",
        "--no-write",
    ]
    assert rules["discrete_gxe_correlation_interval"]["command"] == [
        "checks/discrete_gxe_correlation_interval.py",
        "--replicates",
        "400",
        "--people",
        "600",
        "--families",
        "150",
        "--sibs-per-family",
        "4",
        "--workers",
        "12",
        "--truths",
        "0.3",
        "0.6",
        "0.9",
        "--no-write",
    ]
    assert all(rule["status"] == "ready" for rule in rules.values())
    assert all("blocker" not in rule for rule in rules.values())
    assert analysis["pass_rules_configured"] is True
    assert analysis["design_range"]["measured"] is True
