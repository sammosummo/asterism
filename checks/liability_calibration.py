"""Calibrate liability heritability on a values-free target design.

The reviewed fixture retains only aggregate structural facts needed to stress
the Mendell--Elston approximation at about 1,910 people and a largest family of
180. Those facts select a deterministic, structure-matched synthetic pedigree;
they do not reconstruct participant rows, relationship coefficients, or
covariates. Every status is simulated.

Every attempted replicate remains in its scientific denominator. A refused or
nonconverged free fit, unavailable or malformed test, unavailable interval, or
profile with failed constrained evaluations is an explicit failure. Type-I
error fails only when its one-sided exact 95% lower limit exceeds the nominal
level. Coverage fails when its one-sided exact 95% upper limit is below the
nominal coverage target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import asterism
import numpy as np
import numpy.typing as npt
from scipy.stats import beta

try:
    from checks.one_trait_coverage import target_envelope
except ModuleNotFoundError:
    from one_trait_coverage import target_envelope
"""Imported the shared participant-free structural generator in both entry modes."""

SCENARIOS: tuple[str, str] = ("null", "heritable")
"""Named the type-I-error and interval-coverage simulation cells."""

COMPLETE_OUTCOME: str = "complete"
"""Named the sole replicate state carrying scientific output."""


@dataclass(frozen=True)
class TargetDesign:
    """Hold one reviewed, participant-free liability calibration target.

    Attributes:
        relationship: Generated block-diagonal additive relationship matrix.
        design: Generated six-column fixed-effect design.
        component_sizes: Generated independent component sizes in row order.
        fixture_sha256: Exact liability-fixture byte identity.
        source_fixture_sha256: Exact shared structural-fixture byte identity.
        observed_nonzero_relationship_pairs: Reviewed predecessor aggregate.
        generated_nonzero_relationship_pairs: Synthetic structural pair count.
        relative_nonzero_pair_discrepancy: Aggregate structure-match discrepancy.
        prevalence: Frozen generating case prevalence.
        true_heritability: Frozen generating alternative heritability.
        base_seed: First deterministic null outcome stream.
        scenario_seed_offset: Offset to disjoint alternative streams.
        minimum_cases: Minimum cases and noncases needed for an attempted fit.
        release_replicates: Frozen release replicate count per scenario.
        minimum_smoke_replicates: Smallest reviewed executable smoke.
        levels: Frozen type-I-error levels.
        nominal_coverage: Frozen profile-coverage target.
        monte_carlo_confidence: One-sided exact-binomial confidence.
        maximum_failed_replicates: Fail-closed numerical-failure allowance.
        required_test_rule: Required public boundary-reference label.
        required_interval_level: Required public profile confidence level.
    """

    relationship: npt.NDArray[np.float64]
    """Stored the complete synthetic relationship matrix."""

    design: npt.NDArray[np.float64]
    """Stored the complete synthetic fixed-effect design."""

    component_sizes: tuple[int, ...]
    """Retained generated component sizes from the reviewed histogram."""

    fixture_sha256: str
    """Bound the target to the exact liability-fixture bytes."""

    source_fixture_sha256: str
    """Bound structural generation to the exact shared-fixture bytes."""

    observed_nonzero_relationship_pairs: int
    """Retained the non-identifying predecessor pair-count aggregate."""

    generated_nonzero_relationship_pairs: int
    """Counted nonzero pairs in the generated synthetic relationship."""

    relative_nonzero_pair_discrepancy: float
    """Measured pair-count agreement without claiming reconstruction."""

    prevalence: float
    """Fixed the simulated population prevalence."""

    true_heritability: float
    """Fixed the alternative liability-scale heritability."""

    base_seed: int
    """Selected the first null-scenario random stream."""

    scenario_seed_offset: int
    """Separated null and heritable outcome streams."""

    minimum_cases: int
    """Required enough cases and noncases for a meaningful attempted fit."""

    release_replicates: int
    """Fixed the release campaign size per scientific scenario."""

    minimum_smoke_replicates: int
    """Permitted bounded executable checks without changing acceptance."""

    levels: tuple[float, ...]
    """Fixed all null rejection-rate thresholds."""

    nominal_coverage: float
    """Fixed the intended profile-interval coverage."""

    monte_carlo_confidence: float
    """Fixed the one-sided exact-binomial confidence."""

    maximum_failed_replicates: int
    """Required every requested public fit and inference result to succeed."""

    required_test_rule: str
    """Required the documented boundary-mixture reference."""

    required_interval_level: float
    """Required the documented 95% profile interval."""


@dataclass(frozen=True)
class CampaignConfiguration:
    """Hold explicit, fixture-validated campaign coordinates."""

    replicates: int
    """Retained every attempt in each scenario denominator."""

    workers: int
    """Bounded concurrent public-model evaluations."""

    prevalence: float
    """Retained the reviewed generating prevalence."""

    true_heritability: float
    """Retained the reviewed alternative truth."""

    levels: tuple[float, ...]
    """Retained the reviewed null rejection thresholds."""

    nominal_coverage: float
    """Retained the reviewed interval target."""

    monte_carlo_confidence: float
    """Retained the reviewed exact-binomial confidence."""

    base_seed: int
    """Retained deterministic null stream ownership."""

    scenario_seed_offset: int
    """Retained disjoint alternative stream ownership."""

    minimum_cases: int
    """Retained the reviewed case-count fit threshold."""

    maximum_failed_replicates: int
    """Retained the zero-failure release policy."""


@dataclass(frozen=True)
class WorkerState:
    """Hold immutable target arrays and covariance factors inside a worker."""

    relationship: npt.NDArray[np.float64]
    """Shared the generated target relationship matrix."""

    design: npt.NDArray[np.float64]
    """Shared the generated target fixed-effect design."""

    component_sizes: tuple[int, ...]
    """Located contiguous independent liability blocks."""

    factors: dict[str, tuple[npt.NDArray[np.float64], ...]]
    """Shared reusable null and alternative Cholesky factors."""

    threshold: float
    """Converted the frozen prevalence into a unit-liability threshold."""

    true_heritability: float
    """Retained the generating alternative for interval membership."""

    nominal_coverage: float
    """Retained the required public profile confidence level."""

    minimum_cases: int
    """Required both observed classes before invoking the estimator."""


WORKER_STATE: list[WorkerState] = []
"""Held the sole immutable state installed in each campaign worker."""


@dataclass(frozen=True)
class ReplicateJob:
    """Identify one independently reproducible simulation attempt."""

    scenario: str
    """Selected one of the two frozen scientific scenarios."""

    replicate: int
    """Selected one unconditional denominator position."""

    seed: int
    """Selected one schedule-independent random stream."""


@dataclass(frozen=True)
class ReplicateResult:
    """Retain one attempted replicate, including every failure state."""

    scenario: str
    """Retained scientific cell identity for unconditional aggregation."""

    replicate: int
    """Retained the attempted replicate coordinate."""

    seed: int
    """Retained deterministic outcome stream identity."""

    outcome: str
    """Classified complete inference or one explicit failure mode."""

    cases: int | None = None
    """Recorded the generated case count without retaining statuses."""

    converged: bool | None = None
    """Recorded public free-fit convergence when the fit returned."""

    p_value: float | None = None
    """Recorded a validated public p-value only for complete null tests."""

    statistic: float | None = None
    """Recorded a validated public likelihood-ratio statistic."""

    test_rule: str | None = None
    """Recorded the public null-reference rule."""

    estimate: float | None = None
    """Recorded a validated public profile estimate."""

    lower: float | None = None
    """Recorded a validated public lower endpoint."""

    upper: float | None = None
    """Recorded a validated public upper endpoint."""

    covered: bool | None = None
    """Recorded interval membership or a failed-profile miss."""

    at_bound: bool | None = None
    """Recorded whether a complete interval reached either parameter bound."""

    profile_failures: int | None = None
    """Recorded constrained profile failures without filtering the attempt."""

    error_code: str | None = None
    """Retained the refusal or malformed-record reason without raw data."""


def fixture_sha256(path: Path) -> str:
    """Hash the exact reviewed fixture bytes.

    Args:
        path: Values-free JSON fixture path.

    Returns:
        Lowercase SHA-256 digest of the exact selected bytes.
    """
    payload: bytes = path.read_bytes()
    """Read bytes without normalising their JSON representation."""
    return hashlib.sha256(payload).hexdigest()


def load_target_design(path: Path, *, expected_sha256: str) -> TargetDesign:
    """Validate both fixtures and generate the structure-matched target.

    Args:
        path: Reviewed liability-specific values-free fixture.
        expected_sha256: Exact digest pinned by the invoking command.

    Returns:
        Deterministic synthetic relationship, design, and acceptance contract.

    Raises:
        ValueError: If fixture identity, structural generation, or acceptance
            differs from the reviewed contract.
    """
    selected_sha256: str = fixture_sha256(path)
    """Bound all following interpretation to exact fixture bytes."""

    if selected_sha256 != expected_sha256:
        raise ValueError("LIABILITY_TARGET_FIXTURE_SHA256_MISMATCH")

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    """Parsed the exact-byte-verified values-free fixture."""

    if not isinstance(loaded, dict):
        raise ValueError("LIABILITY_TARGET_FIXTURE_ROOT_INVALID")
    fixture: dict[str, object] = loaded
    """Narrowed the fixture root to its required object shape."""

    identity: tuple[object, ...] = (
        fixture.get("schema_version"),
        fixture.get("fixture_id"),
        fixture.get("reviewed"),
        fixture.get("participant_free"),
        fixture.get("provenance"),
    )
    """Collected the complete review and participant-boundary identity."""

    if identity != (
        1,
        "binary_liability_heritability_target_design",
        True,
        True,
        "reviewed predecessor aggregate receipt",
    ):
        raise ValueError("LIABILITY_TARGET_FIXTURE_IDENTITY_INVALID")

    structure_value: object = fixture.get("target_structure")
    """Selected only the non-identifying structural contract."""

    if not isinstance(structure_value, dict):
        raise ValueError("LIABILITY_TARGET_STRUCTURE_INVALID")
    structure: dict[str, object] = structure_value
    """Narrowed structural aggregates to their required object shape."""

    structure_identity: tuple[object, ...] = (
        structure.get("source_fixture"),
        structure.get("source_fixture_sha256"),
        structure.get("matching_basis"),
        structure.get("reconstruction_claim"),
        structure.get("analysis_n"),
        structure.get("component_count"),
        structure.get("largest_component"),
        structure.get("observed_nonzero_relationship_pairs"),
        structure.get("generated_nonzero_relationship_pairs"),
        structure.get("maximum_relative_nonzero_pair_discrepancy"),
    )
    """Collected every quantitative and interpretive target constraint."""

    if structure_identity != (
        "one_trait_gaussian_heritability.json",
        "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e",
        "reviewed_non_identifying_aggregate_structure",
        False,
        1_909,
        202,
        180,
        27_821,
        27_691,
        0.005,
    ):
        raise ValueError("LIABILITY_TARGET_STRUCTURE_INVALID")

    if structure.get("matched_facts") != [
        "analysis_n",
        "component_count",
        "component_size_histogram",
        "largest_component",
        "nonzero_relationship_pair_count",
    ]:
        raise ValueError("LIABILITY_TARGET_MATCHING_CLAIMS_INVALID")
    if structure.get("not_retained") != [
        "participant_identifiers",
        "participant_rows",
        "phenotypes",
        "participant_covariates",
        "participant_relationship_matrix",
    ]:
        raise ValueError("LIABILITY_TARGET_PARTICIPANT_BOUNDARY_INVALID")
    if structure.get("not_claimed") != [
        "pedigree_reconstruction",
        "relationship_coefficient_identity",
        "relationship_eigenvalue_identity",
        "participant_covariate_moment_matching",
    ]:
        raise ValueError("LIABILITY_TARGET_RECONSTRUCTION_CLAIMS_INVALID")
    """Verified that matching remains explicitly distinct from reconstruction."""

    source_path: Path = path.parent / "one_trait_gaussian_heritability.json"
    """Resolved the sole reviewed structural source beside the wrapper fixture."""

    source_sha256: str = fixture_sha256(source_path)
    """Bound generation to the exact reviewed source-fixture bytes."""

    if source_sha256 != structure["source_fixture_sha256"]:
        raise ValueError("LIABILITY_SOURCE_FIXTURE_SHA256_MISMATCH")

    envelope: object = target_envelope(
        source_path,
        expected_sha256=source_sha256,
        families=202,
    )
    """Generated the structure-matched synthetic pedigree and covariates."""

    aggregate_identity: tuple[object, ...] = (
        envelope.relationship.shape,
        envelope.fixed_effects.shape,
        len(envelope.component_sizes),
        sum(envelope.component_sizes),
        max(envelope.component_sizes),
        envelope.observed_nonzero_relationship_pairs,
        envelope.nonzero_relationship_pairs,
    )
    """Recomputed every target-size and pair-count claim from generated arrays."""

    if aggregate_identity != (
        (1_909, 1_909),
        (1_909, 6),
        202,
        1_909,
        180,
        27_821,
        27_691,
    ):
        raise ValueError("LIABILITY_TARGET_GENERATED_AGGREGATES_INVALID")
    if envelope.relative_pair_count_discrepancy is None or not np.isclose(
        envelope.relative_pair_count_discrepancy,
        130 / 27_821,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("LIABILITY_TARGET_PAIR_DISCREPANCY_INVALID")
    if np.linalg.matrix_rank(envelope.fixed_effects) != 6:
        raise ValueError("LIABILITY_TARGET_DESIGN_RANK_INVALID")

    simulation_value: object = fixture.get("simulation")
    """Selected the fully frozen participant-free outcome generator."""

    if not isinstance(simulation_value, dict):
        raise ValueError("LIABILITY_TARGET_SIMULATION_INVALID")
    simulation: dict[str, object] = simulation_value
    """Narrowed simulation coordinates to their required object shape."""

    expected_simulation: dict[str, object] = {
        "scenarios": ["null", "heritable"],
        "prevalence": 0.254,
        "true_heritability": 0.25,
        "base_seed": 830_000,
        "scenario_seed_offset": 1_000_003,
        "minimum_cases": 10,
        "status_source": "deterministic_simulated_liability_only",
        "fixed_effect_source": (
            "participant_free_synthetic_design_from_source_fixture"
        ),
    }
    """Restated outcome generation independently of mutable fixture contents."""

    if simulation != expected_simulation:
        raise ValueError("LIABILITY_TARGET_SIMULATION_INVALID")

    acceptance_value: object = fixture.get("monte_carlo_acceptance")
    """Selected the acceptance rules written before campaign execution."""

    if not isinstance(acceptance_value, dict):
        raise ValueError("LIABILITY_TARGET_ACCEPTANCE_INVALID")
    acceptance: dict[str, object] = acceptance_value
    """Narrowed acceptance coordinates to their required object shape."""

    expected_acceptance: dict[str, object] = {
        "release_replicates_per_scenario": 200,
        "minimum_smoke_replicates_per_scenario": 1,
        "levels": [0.01, 0.05, 0.1],
        "nominal_coverage": 0.95,
        "confidence": 0.95,
        "level_fail_when": ("one_sided_clopper_pearson_lower_above_nominal_level"),
        "coverage_fail_when": (
            "one_sided_clopper_pearson_upper_below_nominal_coverage"
        ),
        "denominator": "every_attempted_replicate",
        "failure_outcomes": [
            "invalid_case_count",
            "refused",
            "nonconverged",
            "test_failed",
            "interval_failed",
            "profile_failed",
        ],
        "maximum_failed_replicates": 0,
        "required_test_rule": "mixture_50_50",
        "required_interval_level": 0.95,
    }
    """Restated every exact-binomial and numerical-failure decision."""

    if acceptance != expected_acceptance:
        raise ValueError("LIABILITY_TARGET_ACCEPTANCE_INVALID")

    return TargetDesign(
        relationship=np.ascontiguousarray(envelope.relationship),
        design=np.ascontiguousarray(envelope.fixed_effects),
        component_sizes=tuple(envelope.component_sizes),
        fixture_sha256=selected_sha256,
        source_fixture_sha256=source_sha256,
        observed_nonzero_relationship_pairs=27_821,
        generated_nonzero_relationship_pairs=27_691,
        relative_nonzero_pair_discrepancy=float(
            envelope.relative_pair_count_discrepancy
        ),
        prevalence=0.254,
        true_heritability=0.25,
        base_seed=830_000,
        scenario_seed_offset=1_000_003,
        minimum_cases=10,
        release_replicates=200,
        minimum_smoke_replicates=1,
        levels=(0.01, 0.05, 0.1),
        nominal_coverage=0.95,
        monte_carlo_confidence=0.95,
        maximum_failed_replicates=0,
        required_test_rule="mixture_50_50",
        required_interval_level=0.95,
    )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Read a fully explicit participant-free calibration command.

    Args:
        arguments: Argument vector excluding the executable name, or ``None``
            to read the process command line.

    Returns:
        Syntactically validated campaign and exact-fixture coordinates.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built the command contract without scientific defaults."""

    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--prevalence", required=True, type=float)
    parser.add_argument("--true-heritability", required=True, type=float)
    parser.add_argument("--levels", required=True, type=float, nargs="+")
    parser.add_argument("--nominal-coverage", required=True, type=float)
    parser.add_argument("--monte-carlo-confidence", required=True, type=float)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--scenario-seed-offset", required=True, type=int)
    parser.add_argument("--minimum-cases", required=True, type=int)
    parser.add_argument("--design-fixture", required=True, type=Path)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument(
        "--no-write",
        required=True,
        action="store_true",
        help="emit evidence to standard output without modifying the checkout",
    )
    parsed: argparse.Namespace = parser.parse_args(arguments)
    """Read every scientific coordinate directly from argv."""

    if parsed.replicates < 1 or parsed.workers < 1:
        parser.error("--replicates and --workers must be positive")
    if not 0.0 < parsed.prevalence < 1.0:
        parser.error("--prevalence must lie on the open unit interval")
    if not 0.0 <= parsed.true_heritability <= 1.0:
        parser.error("--true-heritability must lie on the closed unit interval")
    if len(set(parsed.levels)) != len(parsed.levels) or any(
        not 0.0 < level < 1.0 for level in parsed.levels
    ):
        parser.error("--levels must be unique and lie on the open unit interval")
    if not 0.0 < parsed.nominal_coverage < 1.0:
        parser.error("--nominal-coverage must lie on the open unit interval")
    if not 0.0 < parsed.monte_carlo_confidence < 1.0:
        parser.error("--monte-carlo-confidence must lie on the open unit interval")
    if parsed.base_seed < 0 or parsed.scenario_seed_offset < 1:
        parser.error("outcome seeds must select nonnegative disjoint streams")
    if parsed.minimum_cases < 1:
        parser.error("--minimum-cases must be positive")
    if len(parsed.fixture_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in parsed.fixture_sha256
    ):
        parser.error("--fixture-sha256 must be a lowercase SHA-256")
    """Rejected malformed campaign coordinates before reading fixture content."""
    return parsed


def campaign_configuration(
    arguments: argparse.Namespace,
    target: TargetDesign,
) -> CampaignConfiguration:
    """Require every command coordinate to agree with the reviewed fixture.

    Args:
        arguments: Fully explicit syntactically validated command coordinates.
        target: Exact-byte-verified reviewed target contract.

    Returns:
        Typed campaign coordinates safe to pass to process workers.

    Raises:
        ValueError: If command values depart from the reviewed release or smoke
            envelope.
    """
    if (
        not target.minimum_smoke_replicates
        <= arguments.replicates
        <= (target.release_replicates)
    ):
        raise ValueError("LIABILITY_REPLICATE_COUNT_OUTSIDE_REVIEWED_RANGE")

    command_identity: tuple[object, ...] = (
        arguments.prevalence,
        arguments.true_heritability,
        tuple(arguments.levels),
        arguments.nominal_coverage,
        arguments.monte_carlo_confidence,
        arguments.base_seed,
        arguments.scenario_seed_offset,
        arguments.minimum_cases,
        arguments.fixture_sha256,
    )
    """Collected every fixture-owned command coordinate."""

    target_identity: tuple[object, ...] = (
        target.prevalence,
        target.true_heritability,
        target.levels,
        target.nominal_coverage,
        target.monte_carlo_confidence,
        target.base_seed,
        target.scenario_seed_offset,
        target.minimum_cases,
        target.fixture_sha256,
    )
    """Collected the corresponding independently reviewed coordinates."""

    if command_identity != target_identity:
        raise ValueError("LIABILITY_COMMAND_DIFFERS_FROM_REVIEWED_FIXTURE")

    return CampaignConfiguration(
        replicates=arguments.replicates,
        workers=arguments.workers,
        prevalence=arguments.prevalence,
        true_heritability=arguments.true_heritability,
        levels=tuple(arguments.levels),
        nominal_coverage=arguments.nominal_coverage,
        monte_carlo_confidence=arguments.monte_carlo_confidence,
        base_seed=arguments.base_seed,
        scenario_seed_offset=arguments.scenario_seed_offset,
        minimum_cases=arguments.minimum_cases,
        maximum_failed_replicates=target.maximum_failed_replicates,
    )


def covariance_factors(
    target: TargetDesign,
    heritability: float,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Factor liability covariance separately inside each family.

    Args:
        target: Generated block-diagonal target relationship.
        heritability: Null or alternative generating heritability.

    Returns:
        Lower Cholesky factors in component row order.
    """
    if not 0.0 <= heritability <= 1.0:
        raise ValueError("LIABILITY_GENERATING_HERITABILITY_INVALID")

    factors: list[npt.NDArray[np.float64]] = []
    """Collected reusable covariance factors without one dense factorisation."""

    offset: int = 0
    """Tracked component placement in the target matrix."""

    for size in target.component_sizes:
        stop: int = offset + size
        """Located the current square relationship block."""

        relationship: npt.NDArray[np.float64] = target.relationship[
            offset:stop, offset:stop
        ]
        """Selected one independent synthetic pedigree component."""

        covariance: npt.NDArray[np.float64] = heritability * relationship
        """Scaled the additive covariance by the generating truth."""

        covariance = covariance + (1.0 - heritability) * np.eye(size)
        """Added independent residual variance on the unit liability scale."""

        factors.append(np.linalg.cholesky(covariance))
        offset = stop
        """Factored the component and advanced to the next independent block."""
    return tuple(factors)


def start_worker(state: WorkerState) -> None:
    """Install immutable simulation state in one process worker.

    Args:
        state: Generated target arrays and precomputed covariance factors.
    """
    WORKER_STATE.clear()
    """Removed stale state if a test or serial campaign reuses this process."""

    WORKER_STATE.append(state)
    """Installed one complete target state without participant information."""


def simulate_status(
    state: WorkerState,
    job: ReplicateJob,
) -> npt.NDArray[np.float64]:
    """Simulate one binary status vector from component covariance factors.

    Args:
        state: Immutable target arrays and per-scenario factors.
        job: Deterministic scenario and stream coordinate.

    Returns:
        Complete participant-free binary status vector.
    """
    if job.scenario not in SCENARIOS or job.replicate < 0 or job.seed < 0:
        raise ValueError("LIABILITY_REPLICATE_JOB_INVALID")

    generator: np.random.Generator = np.random.default_rng(job.seed)
    """Created the schedule-independent status stream."""

    factors: tuple[npt.NDArray[np.float64], ...] = state.factors[job.scenario]
    """Selected null or alternative covariance factors."""

    liabilities: list[npt.NDArray[np.float64]] = []
    """Collected independently simulated family liabilities in row order."""

    for size, factor in zip(state.component_sizes, factors, strict=True):
        standard_normal: npt.NDArray[np.float64] = generator.standard_normal(size)
        """Drew one component's independent standard-normal innovations."""

        liabilities.append(factor @ standard_normal)
    """Applied each block factor without materialising a dense covariance."""

    liability: npt.NDArray[np.float64] = np.concatenate(liabilities)
    """Restored the complete deterministic synthetic row order."""

    return np.ascontiguousarray((liability > state.threshold).astype(np.float64))


def finite_float(value: object) -> float | None:
    """Narrow one public record value to a finite scalar.

    Args:
        value: Untrusted value selected from a public result record.

    Returns:
        Finite float, or ``None`` when the result is absent or malformed.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    scalar: float = float(value)
    """Normalised supported Python and NumPy-compatible numeric scalars."""
    return scalar if np.isfinite(scalar) else None


def run_replicate(job: ReplicateJob) -> ReplicateResult:
    """Run and unconditionally classify one public liability-model attempt.

    Args:
        job: Deterministic scenario and outcome-stream coordinate.

    Returns:
        Complete result or one named failure retaining its denominator position.

    Raises:
        RuntimeError: If called outside the worker-initialisation contract.
    """
    if not WORKER_STATE:
        raise RuntimeError("LIABILITY_WORKER_STATE_MISSING")
    state: WorkerState = WORKER_STATE[0]
    """Read the sole immutable process-local target state."""

    status: npt.NDArray[np.float64] = simulate_status(state, job)
    """Generated one complete status vector without observed phenotypes."""

    cases: int = int(status.sum())
    """Counted cases solely to validate estimability and aggregate evidence."""

    people: int = status.size
    """Counted generated rows independently of fixture claims."""

    if cases < state.minimum_cases or cases > people - state.minimum_cases:
        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome="invalid_case_count",
            cases=cases,
            covered=False if job.scenario == "heritable" else None,
            error_code="LIABILITY_SIMULATED_CASE_COUNT_INVALID",
        )

    model: asterism.LiabilityModel = asterism.LiabilityModel(
        state.relationship,
        status,
        state.design,
    )
    """Constructed only the documented public liability model."""

    try:
        fit_value: object = model.fit()
        """Obtained an explicit public convergence verdict before inference."""
    except ValueError as refusal:
        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome="refused",
            cases=cases,
            covered=False if job.scenario == "heritable" else None,
            error_code=str(refusal),
        )

    if not isinstance(fit_value, dict):
        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome="refused",
            cases=cases,
            covered=False if job.scenario == "heritable" else None,
            error_code="LIABILITY_PUBLIC_FIT_RECORD_MISSING",
        )
    fit: dict[str, object] = fit_value
    """Narrowed the returned public fit to its required record shape."""

    if fit.get("converged") is not True:
        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome="nonconverged",
            cases=cases,
            converged=False,
            covered=False if job.scenario == "heritable" else None,
            error_code="LIABILITY_FIT_NOT_CONVERGED",
        )

    if job.scenario == "null":
        try:
            test_value: object = model.test()
            """Tested zero additive variance through the public named record."""
        except ValueError as refusal:
            return ReplicateResult(
                scenario=job.scenario,
                replicate=job.replicate,
                seed=job.seed,
                outcome="test_failed",
                cases=cases,
                converged=True,
                error_code=str(refusal),
            )

        if not isinstance(test_value, dict):
            return ReplicateResult(
                scenario=job.scenario,
                replicate=job.replicate,
                seed=job.seed,
                outcome="test_failed",
                cases=cases,
                converged=True,
                error_code="LIABILITY_PUBLIC_TEST_RECORD_MISSING",
            )
        test: dict[str, object] = test_value
        """Narrowed the returned public test to its required record shape."""

        p_value: float | None = finite_float(test.get("p_value"))
        """Validated the public boundary-test p-value."""

        statistic: float | None = finite_float(test.get("statistic"))
        """Validated the matching public likelihood-ratio statistic."""

        test_rule_value: object = test.get("rule")
        """Selected the public boundary-reference label."""

        test_rule: str | None = (
            test_rule_value if isinstance(test_rule_value, str) else None
        )
        """Narrowed the rule to a stable string when present."""

        if (
            p_value is None
            or not 0.0 <= p_value <= 1.0
            or statistic is None
            or statistic < 0.0
            or test_rule != "mixture_50_50"
        ):
            return ReplicateResult(
                scenario=job.scenario,
                replicate=job.replicate,
                seed=job.seed,
                outcome="test_failed",
                cases=cases,
                converged=True,
                p_value=p_value,
                statistic=statistic,
                test_rule=test_rule,
                error_code="LIABILITY_PUBLIC_TEST_RECORD_INVALID",
            )

        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome=COMPLETE_OUTCOME,
            cases=cases,
            converged=True,
            p_value=p_value,
            statistic=statistic,
            test_rule=test_rule,
        )

    if job.scenario != "heritable":
        raise ValueError("LIABILITY_REPLICATE_SCENARIO_INVALID")

    try:
        interval_value: object = model.interval()
        """Profiled liability heritability through the public named record."""
    except ValueError as refusal:
        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome="interval_failed",
            cases=cases,
            converged=True,
            covered=False,
            error_code=str(refusal),
        )

    if not isinstance(interval_value, dict):
        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome="interval_failed",
            cases=cases,
            converged=True,
            covered=False,
            error_code="LIABILITY_PUBLIC_INTERVAL_RECORD_MISSING",
        )
    interval: dict[str, object] = interval_value
    """Narrowed the returned public interval to its required record shape."""

    estimate: float | None = finite_float(interval.get("estimate"))
    """Validated the public profile point estimate."""

    lower: float | None = finite_float(interval.get("lower"))
    """Validated the public lower profile endpoint."""

    upper: float | None = finite_float(interval.get("upper"))
    """Validated the public upper profile endpoint."""

    interval_level: float | None = finite_float(interval.get("level"))
    """Validated the public profile confidence level."""

    profile_failures_value: object = interval.get("profile_failures")
    """Selected the public count of failed constrained evaluations."""

    profile_failures: int | None = (
        profile_failures_value
        if isinstance(profile_failures_value, int)
        and not isinstance(profile_failures_value, bool)
        else None
    )
    """Narrowed the failure count to a genuine integer."""

    lower_limited_value: object = interval.get("lower_limited")
    """Selected the required public lower-endpoint limitation state."""

    upper_limited_value: object = interval.get("upper_limited")
    """Selected the required public upper-endpoint limitation state."""

    if (
        estimate is None
        or lower is None
        or upper is None
        or not 0.0 <= lower <= estimate <= upper <= 1.0
        or interval_level is None
        or not np.isclose(
            interval_level,
            state.nominal_coverage,
            rtol=0.0,
            atol=1e-12,
        )
        or profile_failures is None
        or profile_failures < 0
        or not isinstance(lower_limited_value, bool)
        or not isinstance(upper_limited_value, bool)
    ):
        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome="interval_failed",
            cases=cases,
            converged=True,
            estimate=estimate,
            lower=lower,
            upper=upper,
            covered=False,
            profile_failures=profile_failures,
            error_code="LIABILITY_PUBLIC_INTERVAL_RECORD_INVALID",
        )

    lower_limited: bool = lower_limited_value
    """Read the public lower-endpoint limitation state."""

    upper_limited: bool = upper_limited_value
    """Read the public upper-endpoint limitation state."""

    if profile_failures > 0:
        return ReplicateResult(
            scenario=job.scenario,
            replicate=job.replicate,
            seed=job.seed,
            outcome="profile_failed",
            cases=cases,
            converged=True,
            estimate=estimate,
            lower=lower,
            upper=upper,
            covered=False,
            at_bound=lower_limited or upper_limited,
            profile_failures=profile_failures,
            error_code="LIABILITY_PROFILE_EVALUATION_FAILED",
        )

    covered: bool = lower <= state.true_heritability <= upper
    """Scored interval membership only for a complete failure-free profile."""

    return ReplicateResult(
        scenario=job.scenario,
        replicate=job.replicate,
        seed=job.seed,
        outcome=COMPLETE_OUTCOME,
        cases=cases,
        converged=True,
        estimate=estimate,
        lower=lower,
        upper=upper,
        covered=covered,
        at_bound=lower_limited or upper_limited,
        profile_failures=0,
    )


def one_sided_lower_limit(
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    """Return an exact one-sided Clopper--Pearson lower limit.

    Args:
        successes: Observed binomial successes.
        trials: Every attempted replicate in the denominator.
        confidence: Required one-sided confidence.

    Returns:
        Exact lower confidence limit.

    Raises:
        ValueError: If counts or confidence cannot define a binomial limit.
    """
    if trials < 1 or successes < 0 or successes > trials or not 0.0 < confidence < 1.0:
        raise ValueError("LIABILITY_BINOMIAL_COUNTS_INVALID")

    return (
        0.0
        if successes == 0
        else float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))
    )


def one_sided_upper_limit(
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    """Return an exact one-sided Clopper--Pearson upper limit.

    Args:
        successes: Observed binomial successes.
        trials: Every attempted replicate in the denominator.
        confidence: Required one-sided confidence.

    Returns:
        Exact upper confidence limit.

    Raises:
        ValueError: If counts or confidence cannot define a binomial limit.
    """
    if trials < 1 or successes < 0 or successes > trials or not 0.0 < confidence < 1.0:
        raise ValueError("LIABILITY_BINOMIAL_COUNTS_INVALID")

    return (
        1.0
        if successes == trials
        else float(beta.ppf(confidence, successes + 1, trials - successes))
    )


def level_decision(
    *,
    rejections: int,
    attempted: int,
    level: float,
    confidence: float,
) -> dict[str, object]:
    """Apply the prewritten validity-first null rejection decision.

    Args:
        rejections: Complete public p-values at or below the selected level.
        attempted: Every requested null replicate, including failures.
        level: Prespecified nominal type-I-error level.
        confidence: Prespecified one-sided Monte Carlo confidence.

    Returns:
        Exact lower limit and anti-conservative pass decision.
    """
    lower: float = one_sided_lower_limit(rejections, attempted, confidence)
    """Quantified whether the true rejection rate can remain at the nominal level."""

    anti_conservative: bool = lower > level and not np.isclose(
        lower,
        level,
        rtol=0.0,
        atol=1e-12,
    )
    """Preserved mathematical boundary equality across beta-quantile round-off."""

    return {
        "attempted": attempted,
        "rejections": rejections,
        "rate": rejections / attempted,
        "one_sided_clopper_pearson_lower": lower,
        "level": level,
        "confidence": confidence,
        "verdict": "anti_conservative"
        if anti_conservative
        else "compatible_or_conservative",
        "passed": not anti_conservative,
    }


def coverage_decision(
    *,
    covered: int,
    attempted: int,
    nominal: float,
    confidence: float,
) -> dict[str, object]:
    """Apply the prewritten under-coverage decision.

    Args:
        covered: Complete failure-free intervals containing the truth.
        attempted: Every requested alternative replicate, including failures.
        nominal: Prespecified profile-coverage target.
        confidence: Prespecified one-sided Monte Carlo confidence.

    Returns:
        Exact upper limit and under-coverage pass decision.
    """
    upper: float = one_sided_upper_limit(covered, attempted, confidence)
    """Quantified whether nominal coverage remains compatible with the campaign."""

    under_covering: bool = upper < nominal and not np.isclose(
        upper,
        nominal,
        rtol=0.0,
        atol=1e-12,
    )
    """Preserved mathematical boundary equality across beta-quantile round-off."""

    return {
        "attempted": attempted,
        "covered": covered,
        "coverage": covered / attempted,
        "one_sided_clopper_pearson_upper": upper,
        "nominal": nominal,
        "confidence": confidence,
        "verdict": "under_covering" if under_covering else "compatible_or_conservative",
        "passed": not under_covering,
    }


def score_campaign(
    results: Sequence[ReplicateResult],
    *,
    levels: Sequence[float],
    nominal_coverage: float,
    monte_carlo_confidence: float,
    maximum_failed_replicates: int,
) -> dict[str, object]:
    """Aggregate every attempted replicate without conditional denominators.

    Args:
        results: Complete sequence of null and heritable attempts.
        levels: Prespecified null rejection thresholds.
        nominal_coverage: Prespecified profile-coverage target.
        monte_carlo_confidence: Prespecified one-sided binomial confidence.
        maximum_failed_replicates: Allowed numerical or inference failures.

    Returns:
        Fully recomputable level, coverage, failure, and pass evidence.

    Raises:
        ValueError: If either scenario is missing or replicate coordinates repeat.
    """
    null_results: list[ReplicateResult] = [
        result for result in results if result.scenario == "null"
    ]
    """Selected every null attempt, including unavailable tests."""

    heritable_results: list[ReplicateResult] = [
        result for result in results if result.scenario == "heritable"
    ]
    """Selected every alternative attempt, including unavailable intervals."""

    if not null_results or not heritable_results:
        raise ValueError("LIABILITY_CAMPAIGN_SCENARIO_MISSING")
    if len({result.replicate for result in null_results}) != len(null_results) or len(
        {result.replicate for result in heritable_results}
    ) != len(heritable_results):
        raise ValueError("LIABILITY_CAMPAIGN_REPLICATE_DUPLICATED")

    outcome_counts: Counter[str] = Counter(result.outcome for result in results)
    """Counted every complete and failed outcome across both scenarios."""

    failed_replicates: int = sum(
        result.outcome != COMPLETE_OUTCOME for result in results
    )
    """Applied the zero-failure policy independently of level and coverage."""

    p_values: list[float] = [
        result.p_value
        for result in null_results
        if result.outcome == COMPLETE_OUTCOME and result.p_value is not None
    ]
    """Selected validated tests without changing the null denominator."""

    rejection: dict[str, dict[str, object]] = {}
    """Collected one exact-binomial decision per prespecified level."""

    for level in levels:
        rejections: int = sum(p_value <= level for p_value in p_values)
        """Counted complete public rejections at the selected threshold."""

        rejection[str(level)] = level_decision(
            rejections=rejections,
            attempted=len(null_results),
            level=level,
            confidence=monte_carlo_confidence,
        )
        """Applied the one-sided exact decision to this unconditional count."""
    """Scored each rejection count over every attempted null replicate."""

    covered: int = sum(result.covered is True for result in heritable_results)
    """Counted only complete truth-containing profiles as coverage successes."""

    coverage: dict[str, object] = coverage_decision(
        covered=covered,
        attempted=len(heritable_results),
        nominal=nominal_coverage,
        confidence=monte_carlo_confidence,
    )
    """Scored coverage over every attempted alternative replicate."""

    estimates: list[float] = [
        result.estimate
        for result in heritable_results
        if result.outcome == COMPLETE_OUTCOME and result.estimate is not None
    ]
    """Selected complete point estimates solely for descriptive evidence."""

    complete_intervals: int = sum(
        result.outcome == COMPLETE_OUTCOME for result in heritable_results
    )
    """Counted complete profiles independently of optional descriptive fields."""

    coverage["computed"] = complete_intervals
    """Recorded profiles that returned every required public field."""

    coverage["median_estimate"] = float(np.median(estimates)) if estimates else None
    """Summarised complete point estimates without imputing failed attempts."""

    coverage["at_a_bound"] = sum(
        result.at_bound is True for result in heritable_results
    ) / len(heritable_results)
    """Measured boundary incidence over the unconditional denominator."""

    level_record: dict[str, object] = {
        "attempted": len(null_results),
        "computed": len(p_values),
        "atom_at_one": sum(p_value > 0.999 for p_value in p_values) / len(null_results),
        "rejection": rejection,
    }
    """Recorded complete null accounting with failures retained in rates."""

    decisions_passed: bool = (
        all(decision["passed"] is True for decision in rejection.values())
        and coverage["passed"] is True
    )
    """Combined the independently prewritten inferential decisions."""

    return {
        "level": level_record,
        "coverage": coverage,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "failed_replicates": failed_replicates,
        "maximum_failed_replicates": maximum_failed_replicates,
        "passed": (failed_replicates <= maximum_failed_replicates and decisions_passed),
    }


def run_campaign(
    target: TargetDesign,
    configuration: CampaignConfiguration,
) -> list[ReplicateResult]:
    """Run both reviewed scenarios through public liability models.

    Args:
        target: Generated values-free target arrays.
        configuration: Explicit fixture-validated campaign coordinates.

    Returns:
        One retained result for every requested attempt.
    """
    factors: dict[str, tuple[npt.NDArray[np.float64], ...]] = {
        "null": covariance_factors(target, 0.0),
        "heritable": covariance_factors(target, configuration.true_heritability),
    }
    """Factored null and alternative covariance once before simulation."""

    state: WorkerState = WorkerState(
        relationship=target.relationship,
        design=target.design,
        component_sizes=target.component_sizes,
        factors=factors,
        threshold=NormalDist().inv_cdf(1.0 - configuration.prevalence),
        true_heritability=configuration.true_heritability,
        nominal_coverage=configuration.nominal_coverage,
        minimum_cases=configuration.minimum_cases,
    )
    """Built one immutable process-worker state from reviewed inputs."""

    jobs: list[ReplicateJob] = [
        ReplicateJob(
            scenario=scenario,
            replicate=replicate,
            seed=(
                configuration.base_seed
                + replicate
                + (0 if scenario == "null" else configuration.scenario_seed_offset)
            ),
        )
        for scenario in SCENARIOS
        for replicate in range(configuration.replicates)
    ]
    """Enumerated disjoint deterministic streams before worker scheduling."""

    if configuration.workers == 1:
        start_worker(state)
        """Installed the same immutable state in the current process."""

        results: list[ReplicateResult] = [run_replicate(job) for job in jobs]
        """Ran a deterministic single-process smoke without process overhead."""
        return results

    with ProcessPoolExecutor(
        max_workers=configuration.workers,
        initializer=start_worker,
        initargs=(state,),
    ) as pool:
        results = list(pool.map(run_replicate, jobs, chunksize=1))
        """Ran every attempt without filtering failed or unavailable records."""
    return results


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the exact-byte-bound campaign and emit one JSON evidence document.

    Args:
        arguments: Argument vector excluding the executable name, or ``None``
            to read the process command line.

    Returns:
        Zero only when every failure and Monte Carlo rule passes.
    """
    parsed: argparse.Namespace = parse_arguments(arguments)
    """Read all scientific coordinates from the explicit command."""

    target: TargetDesign = load_target_design(
        parsed.design_fixture,
        expected_sha256=parsed.fixture_sha256,
    )
    """Verified exact fixture bytes before generating any outcome."""

    configuration: CampaignConfiguration = campaign_configuration(parsed, target)
    """Required every command coordinate to match the reviewed contract."""

    started: float = time.perf_counter()
    """Started bounded campaign timing after deterministic target construction."""

    results: list[ReplicateResult] = run_campaign(target, configuration)
    """Retained one explicit record for every requested replicate."""

    score: dict[str, object] = score_campaign(
        results,
        levels=configuration.levels,
        nominal_coverage=configuration.nominal_coverage,
        monte_carlo_confidence=configuration.monte_carlo_confidence,
        maximum_failed_replicates=configuration.maximum_failed_replicates,
    )
    """Applied exact Monte Carlo rules with unconditional denominators."""

    failures: list[str] = []
    """Collected human-readable consequences of the machine decisions."""

    if score["failed_replicates"] > configuration.maximum_failed_replicates:
        failures.append("one or more attempted public fits or profiles failed")
    level_record: dict[str, object] = score["level"]
    """Selected the null-test decisions for explicit failure messages."""

    rejection: dict[str, dict[str, object]] = level_record["rejection"]
    """Selected exact rejection-rate decisions at all frozen levels."""

    for level, decision in rejection.items():
        if decision["passed"] is not True:
            failures.append(f"type-I error is anti-conservative at level {level}")
    """Reported each exact-binomial level failure without changing acceptance."""

    coverage: dict[str, object] = score["coverage"]
    """Selected the alternative profile-coverage decision."""

    if coverage["passed"] is not True:
        failures.append("profile intervals are under-covering")

    evidence: dict[str, object] = {
        "check": "liability_calibration",
        "participant_free": True,
        "target_contract": "aggregate_structure_match_not_reconstruction",
        "fixture_sha256": target.fixture_sha256,
        "source_fixture_sha256": target.source_fixture_sha256,
        "people": target.relationship.shape[0],
        "relationship_components": len(target.component_sizes),
        "largest_family": max(target.component_sizes),
        "observed_nonzero_relationship_pairs": (
            target.observed_nonzero_relationship_pairs
        ),
        "generated_nonzero_relationship_pairs": (
            target.generated_nonzero_relationship_pairs
        ),
        "relative_nonzero_pair_discrepancy": (target.relative_nonzero_pair_discrepancy),
        "public_interface": [
            "asterism.LiabilityModel.fit",
            "asterism.LiabilityModel.test",
            "asterism.LiabilityModel.interval",
        ],
        "configuration": asdict(configuration),
        "attempted_replicates": len(results),
        "elapsed_seconds": time.perf_counter() - started,
        "results": score,
        "passed": not failures,
        "failures": failures,
    }
    """Built values-free evidence sufficient to recompute every release decision."""

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
