"""Contract tests for the participant-free continuous-GxE target gate."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

TARGET_FIXTURE: Path = (
    Path(__file__).parents[1]
    / "checks"
    / "design_fixtures"
    / "continuous_gxe_target_design.json"
)
"""Located the reviewed values-free continuous-GxE stress fixture."""

TARGET_FIXTURE_SHA256: str = (
    "f5facfdf09b9690de3db4f35961e8f9e805bc1374eccf8b1a52bbec51c99283f"
)
"""Pinned the exact reviewed fixture bytes before any target run."""


def target_module() -> ModuleType:
    """Load the standalone target-design command through its file boundary."""
    path: Path = Path(__file__).parents[1] / "checks" / "gxe_target_design.py"
    """Located the command without making the checks directory a package."""

    checks_path: str = str(path.parent)
    """Selected the sibling-check import root used by standalone commands."""

    if checks_path not in sys.path:
        sys.path.insert(0, checks_path)
    """Made the existing independent and pedigree checks importable by name."""

    specification: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location("gxe_target_design", path)
    )
    """Created the import specification for the standalone source file."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the module governed by the validated specification."""

    sys.modules[specification.name] = module
    """Registered the module so its public dataclasses resolve annotations."""

    specification.loader.exec_module(module)
    """Executed the command without invoking its command-line entry point."""
    return module


def test_target_fixture_builds_only_the_reviewed_structural_envelope() -> None:
    """Match reviewed pedigree scale while keeping age geometry synthetic."""
    module: ModuleType = target_module()
    """Loaded the target fixture verifier and deterministic constructor."""

    envelope: Any = module.gxe_target_envelope(
        TARGET_FIXTURE,
        expected_sha256=TARGET_FIXTURE_SHA256,
    )
    """Built the participant-free pedigree, environment, and fixed design."""

    assert envelope.relationship.shape == (1_909, 1_909)
    assert len(envelope.component_sizes) == 202
    assert max(envelope.component_sizes) == 180
    assert envelope.environment.shape == (1_909,)
    assert envelope.environment.min() >= -1.5
    assert envelope.environment.max() <= 1.5
    assert envelope.reporting_grid == (-1.0, 0.0, 1.0)
    assert envelope.fixed_effects.shape == (1_909, 6)
    assert np.linalg.matrix_rank(envelope.fixed_effects) == 6

    offset: int = 0
    """Tracked component slices while checking within-family age variation."""

    for size in envelope.component_sizes:
        stop: int = offset + size
        """Located the exclusive end of one synthetic pedigree component."""

        if size > 1:
            assert np.ptp(envelope.environment[offset:stop]) > 0.0
        offset = stop
        """Advanced to the next contiguous synthetic pedigree component."""
    """Required environment variation in every component able to contain it."""

    fixture_text: str = TARGET_FIXTURE.read_text(encoding="utf-8")
    """Read only fixture keys to guard the participant-free review boundary."""

    for forbidden in ("subject_id", "phenotype", "observed_age", "matrix"):
        assert forbidden not in fixture_text.lower()


def test_blockwise_likelihood_equals_the_independent_dense_definition() -> None:
    """Keep target block decomposition identical to invariant dense REML."""
    module: ModuleType = target_module()
    """Loaded the blockwise covariance and likelihood primitives."""

    reference: ModuleType = importlib.import_module("gaussian_full_fit_reference")
    """Loaded the existing independent dense Gaussian likelihood definition."""

    relationship: np.ndarray = np.eye(8)
    """Started a small two-component additive relationship."""

    for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
        relationship[first, second] = 0.5
        """Recorded one lower-directed sibling relationship entry."""

        relationship[second, first] = 0.5
        """Completed the matching symmetric sibling relationship entry."""
    """Added four sibling relationships without crossing component blocks."""

    environment: np.ndarray = np.asarray([-1.2, -0.4, 0.2, 1.0, -1.0, -0.1, 0.5, 1.3])
    """Selected environments varying inside both independent components."""

    design: np.ndarray = np.column_stack([np.ones(8), environment])
    """Built a full-rank mean design for invariant restricted likelihood."""

    response: np.ndarray = np.asarray([0.3, -0.2, 1.1, 0.7, -0.8, 0.4, 0.9, -0.1])
    """Fixed a nonconstant response independently of either likelihood route."""

    parameters: dict[str, np.ndarray] = {
        "exponential": np.asarray([np.log(0.6), 0.2, 0.35, np.log(0.5), -0.1]),
        "random_regression": np.asarray(
            [np.log(np.sqrt(0.6)), 0.15, 0.4, np.log(np.sqrt(0.5)), -0.1, 0.3]
        ),
    }
    """Selected interior natural coordinates for both supported surfaces."""

    for surface, coordinates in parameters.items():
        covariance: np.ndarray = module.surface_covariance(
            surface,
            coordinates,
            relationship,
            environment,
        )
        """Assembled the complete independent covariance for the dense definition."""

        for reml in (False, True):
            expected: float = reference.profiled_negative_loglik(
                covariance,
                design,
                response,
                reml=reml,
            )
            """Evaluated the established dense profiled likelihood."""

            observed: float = module.blockwise_profiled_negative_loglik(
                surface,
                coordinates,
                relationship,
                environment,
                design,
                response,
                (4, 4),
                reml=reml,
            )
            """Evaluated the same model by independent component summation."""

            assert observed == pytest.approx(expected, abs=1e-10)


def test_blockwise_optimizer_agrees_with_public_fits_for_both_surfaces() -> None:
    """Compare maximized likelihood and common quantities without shared fitting."""
    module: ModuleType = target_module()
    """Loaded the public-versus-independent target comparison seam."""

    sentinel: ModuleType = importlib.import_module("gxe_against_independent_full_fit")
    """Loaded the established participant-free n=80 design only as input."""

    relationship, environment, design = sentinel.structure()
    """Built forty independent sibling pairs with within-pair environments."""

    component_sizes: tuple[int, ...] = (2,) * 40
    """Declared the exact contiguous blocks used by the independent likelihood."""

    for index, surface in enumerate(("exponential", "random_regression")):
        truth: np.ndarray = module.reference_truth(surface)
        """Selected the prewritten interior surface used by target comparisons."""

        covariance: np.ndarray = module.surface_covariance(
            surface,
            truth,
            relationship,
            environment,
        )
        """Assembled a response covariance independently of the public fitter."""

        generator: np.random.Generator = np.random.default_rng(81_300 + index)
        """Created a distinct deterministic outcome stream for this surface."""

        response: np.ndarray = design @ np.asarray([1.0, 0.2]) + np.linalg.cholesky(
            covariance
        ) @ generator.standard_normal(design.shape[0])
        """Generated one complete participant-free response from the tested family."""

        result: dict[str, object] = module.compare_surface_fit(
            surface,
            relationship,
            environment,
            design,
            response,
            component_sizes,
            (-1.0, 0.0, 1.0),
            reml=True,
        )
        """Ran public Asterism and independent blockwise optimization."""

        assert result["passed"] is True, result
        assert result["asterism_converged"] is True
        assert result["reference_converged"] is True


def test_surface_response_generation_is_deterministic_and_blockwise() -> None:
    """Generate target outcomes without factoring across independent families."""
    module: ModuleType = target_module()
    """Loaded the participant-free blockwise response generator."""

    relationship: np.ndarray = np.asarray(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ]
    )
    """Fixed two unrelated sibling-pair covariance components."""

    environment: np.ndarray = np.asarray([-1.0, 0.2, -0.4, 1.0])
    """Varied the environment independently inside both components."""

    design: np.ndarray = np.column_stack([np.ones(4), environment])
    """Built the fixed mean design used by the deterministic generator."""

    first: np.ndarray = module.simulate_surface_response(
        "exponential",
        relationship,
        environment,
        design,
        (2, 2),
        seed=271,
    )
    """Generated one response from independently factored pedigree blocks."""

    second: np.ndarray = module.simulate_surface_response(
        "exponential",
        relationship,
        environment,
        design,
        (2, 2),
        seed=271,
    )
    """Repeated the identical target coordinates and random stream."""

    assert first.shape == (4,)
    assert np.array_equal(first, second)


def test_target_level_rule_is_one_sided_and_refusals_fail_closed() -> None:
    """Reject anti-conservative tests while retaining conservative references."""
    module: ModuleType = target_module()
    """Loaded the prewritten exact-binomial target decision."""

    anti_conservative: dict[str, object] = module.calibration_decision(
        rejections=20,
        attempted=100,
        refused_or_nonconverged=0,
        level=0.05,
    )
    """Scored a null rejection rate whose exact lower limit exceeds its level."""

    assert anti_conservative["passed"] is False
    assert anti_conservative["verdict"] == "anti_conservative"

    conservative: dict[str, object] = module.calibration_decision(
        rejections=0,
        attempted=100,
        refused_or_nonconverged=0,
        level=0.05,
    )
    """Scored a conservative boundary reference without demanding uniform p-values."""

    assert conservative["passed"] is True
    assert conservative["verdict"] == "compatible_or_conservative"

    refused: dict[str, object] = module.calibration_decision(
        rejections=0,
        attempted=100,
        refused_or_nonconverged=1,
        level=0.05,
    )
    """Scored one missing scientific result under the zero-refusal policy."""

    assert refused["passed"] is False
    assert refused["verdict"] == "refused_or_nonconverged"


def test_real_public_null_replicate_exercises_both_nonstandard_tests() -> None:
    """Score fit convergence and both public test records without dropping one."""
    module: ModuleType = target_module()
    """Loaded the real public null-replicate seam."""

    sentinel: ModuleType = importlib.import_module("gxe_against_independent_full_fit")
    """Loaded the established small participant-free sibling-pair structure."""

    relationship, environment, design = sentinel.structure()
    """Built n=80 arrays with informative within-family environments."""

    for surface in ("exponential", "random_regression"):
        result: dict[str, object] = module.run_null_replicate(
            surface,
            relationship,
            environment,
            design,
            (2,) * 40,
            replicate=0,
        )
        """Fitted one common flat-null response through the selected public surface."""

        assert result["fit_converged"] is True, result
        assert result["refusal"] is None
        tests: dict[str, dict[str, object]] = result["tests"]
        """Selected both reportable public null comparisons."""

        assert set(tests) == {"correlation", "interaction"}
        assert tests["correlation"]["rule"] == "mixture_50_50"
        assert tests["interaction"]["rule"] == "half_chi2_1_half_chi2_2"
        assert all(test["complete"] is True for test in tests.values())


def test_null_aggregation_keeps_every_failed_replicate_in_the_gate() -> None:
    """Make incomplete target fits a gate failure rather than missing data."""
    module: ModuleType = target_module()
    """Loaded the null-campaign aggregation seam."""

    complete_tests: dict[str, dict[str, object]] = {
        "correlation": {
            "complete": True,
            "p_value": 0.01,
            "rule": "mixture_50_50",
        },
        "interaction": {
            "complete": True,
            "p_value": 0.20,
            "rule": "half_chi2_1_half_chi2_2",
        },
    }
    """Built one complete public result with one rejection."""

    records: list[dict[str, object]] = [
        {
            "surface": "exponential",
            "replicate": 0,
            "fit_converged": True,
            "refusal": None,
            "tests": complete_tests,
        },
        {
            "surface": "exponential",
            "replicate": 1,
            "fit_converged": False,
            "refusal": "GXE_NO_START_CONVERGED",
            "tests": {},
        },
    ]
    """Added one refused replicate that must remain in both test denominators."""

    result: dict[str, object] = module.aggregate_null_records(
        "exponential",
        records,
        attempted=2,
        level=0.05,
    )
    """Aggregated both requested replicates under the zero-refusal policy."""

    assert result["passed"] is False
    tests: dict[str, dict[str, object]] = result["tests"]
    """Selected the independently scored public test cells."""

    assert tests["correlation"]["attempted"] == 2
    assert tests["correlation"]["rejections"] == 1
    assert tests["correlation"]["refused_or_nonconverged"] == 1
    assert tests["interaction"]["attempted"] == 2
    assert tests["interaction"]["refused_or_nonconverged"] == 1


def test_target_command_requires_every_scientific_coordinate() -> None:
    """Prevent defaults or ambient files from selecting a target campaign."""
    module: ModuleType = target_module()
    """Loaded the standalone target command parser."""

    with pytest.raises(SystemExit):
        module.parse_arguments([])

    arguments: Any = module.parse_arguments(
        [
            "--replicates",
            "100",
            "--workers",
            "4",
            "--design-fixture",
            str(TARGET_FIXTURE),
            "--fixture-sha256",
            TARGET_FIXTURE_SHA256,
            "--dense-sentinel",
            "checks/gxe_against_independent_full_fit.py",
            "--no-write",
        ]
    )
    """Read a reduced but fully explicit participant-free target command."""

    assert arguments.replicates == 100
    assert arguments.workers == 4
    assert arguments.design_fixture == TARGET_FIXTURE
    assert arguments.fixture_sha256 == TARGET_FIXTURE_SHA256
    assert arguments.dense_sentinel == Path(
        "checks/gxe_against_independent_full_fit.py"
    )
    assert arguments.no_write is True


def test_null_jobs_cover_each_surface_replicate_once_independent_of_workers() -> None:
    """Keep deterministic stream ownership unchanged by process scheduling."""
    module: ModuleType = target_module()
    """Loaded the target null-job partitioner."""

    jobs: list[Any] = module.null_jobs(
        replicates=10,
        workers=6,
        fixture_path=TARGET_FIXTURE,
        fixture_sha256=TARGET_FIXTURE_SHA256,
    )
    """Partitioned twenty surface-replicates into at most six worker chunks."""

    assert len(jobs) == 6
    for surface in ("exponential", "random_regression"):
        coordinates: list[int] = [
            replicate
            for job in jobs
            if job.surface == surface
            for replicate in range(job.start, job.stop)
        ]
        """Expanded this surface's deterministic coordinates across its chunks."""

        assert coordinates == list(range(10))


def test_existing_dense_full_fit_remains_a_separate_required_sentinel() -> None:
    """Retain the n=80 dense check beside, not instead of, target optimization."""
    module: ModuleType = target_module()
    """Loaded the exact-command sentinel verifier."""

    result: dict[str, object] = module.run_dense_sentinel(
        Path(__file__).parents[1] / "checks" / "gxe_against_independent_full_fit.py"
    )
    """Ran the established dense independent check in this Python environment."""

    assert result["passed"] is True, result
    surfaces: list[dict[str, object]] = result["surfaces"]
    """Selected the two supported dense-reference outcomes."""

    assert {surface["surface"] for surface in surfaces} == {
        "exponential",
        "random_regression",
    }
    assert all(surface["people"] == 80 for surface in surfaces)


def test_fixture_pre_writes_full_fit_null_and_refusal_acceptance() -> None:
    """Bind target execution to rules fixed before any target outcome is seen."""
    module: ModuleType = target_module()
    """Loaded the reviewed executable-contract verifier."""

    contract: dict[str, object] = module.gate_contract(
        TARGET_FIXTURE,
        expected_sha256=TARGET_FIXTURE_SHA256,
    )
    """Read acceptance only after binding the exact reviewed fixture bytes."""

    assert contract["release_replicates"] == 500
    assert contract["minimum_smoke_replicates"] == 100
    assert contract["levels"] == (0.05,)
    assert contract["tests"] == ("correlation", "interaction")
    assert contract["maximum_refused_or_nonconverged"] == 0
    assert contract["quantity_absolute_tolerance"] == 1e-3
    assert contract["loglik_absolute_tolerance"] == 1e-4
    assert contract["dense_sentinel"] == "checks/gxe_against_independent_full_fit.py"


def test_release_manifest_pins_the_deferred_full_target_campaign() -> None:
    """Make the runnable rule exact without promoting unrun release evidence."""
    manifest: dict[str, Any] = tomllib.loads(
        (Path(__file__).parents[1] / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the authoritative development release contract."""

    analysis: dict[str, Any] = next(
        selected
        for selected in manifest["analyses"]
        if selected["id"] == "continuous_gene_by_environment"
    )
    """Selected continuous GxE without depending on manifest position."""

    rule: dict[str, Any] = next(
        selected
        for selected in analysis["pass_rules"]
        if selected["id"] == "gxe_target_design"
    )
    """Selected only the target-design pass rule assigned to this tranche."""

    assert rule["status"] == "ready"
    assert "blocker" not in rule
    assert rule["command"] == [
        "checks/gxe_target_design.py",
        "--replicates",
        "500",
        "--workers",
        "12",
        "--design-fixture",
        "checks/design_fixtures/continuous_gxe_target_design.json",
        "--fixture-sha256",
        TARGET_FIXTURE_SHA256,
        "--dense-sentinel",
        "checks/gxe_against_independent_full_fit.py",
        "--no-write",
    ]
    assert rule["design_facts"]["sample_size"] == {"min": 1909, "max": 1909}
    assert rule["design_facts"]["largest_family"] == {"max": 180}
    assert rule["design_facts"]["family_components"] == 202
    assert rule["design_facts"]["environment_range"] == {
        "min": -1.5,
        "max": 1.5,
    }
    assert rule["design_facts"]["replicates_per_surface"] == 500
    assert manifest["scientific_pass_rules_configured"] is True
    assert analysis["pass_rules_configured"] is True
