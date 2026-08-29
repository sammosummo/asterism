"""Qualify censored-trait inference on a values-free target pedigree."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import asterism
import numpy as np
import numpy.typing as npt
from scipy.stats import beta

try:
    from checks.one_trait_coverage import target_envelope
except ModuleNotFoundError:
    from one_trait_coverage import target_envelope
"""Imported the reviewed structural generator in module and direct-command modes."""

SCENARIOS: tuple[str, str] = ("null", "heritable")
"""Separated boundary-test calibration from interior interval coverage."""

COMPLETE_OUTCOME: str = "complete"
"""Named the only replicate outcome eligible to contribute computed inference."""

FAILURE_OUTCOMES: tuple[str, ...] = (
    "invalid_censoring_count",
    "refused",
    "nonconverged",
    "test_failed",
    "interval_failed",
    "profile_failed",
)
"""Fixed every unsuccessful scientific outcome before running the campaign."""

CENSORING_SHARES: tuple[float, float] = (0.52, 0.75)
"""Fixed the two observed high-frequency audiogram censoring levels."""

TRUE_HERITABILITY: float = 0.5
"""Selected the established interior complete-trait heritability."""

TRUE_VARIANCE: float = 4.0
"""Selected the established latent complete-trait variance."""

LEVEL: float = 0.05
"""Fixed the boundary-test type-I-error level."""

NOMINAL_COVERAGE: float = 0.95
"""Fixed the public profile interval's nominal coverage."""

MONTE_CARLO_CONFIDENCE: float = 0.95
"""Fixed one-sided exact-binomial decision confidence."""

BASE_SEED: int = 920_000
"""Owned deterministic outcome streams independently of worker scheduling."""

SCENARIO_SEED_OFFSET: int = 1_000_003
"""Separated null and heritable outcome streams."""

RATE_SEED_OFFSET: int = 10_007
"""Separated the two censoring-level outcome streams."""

MAXIMUM_FAILED_REPLICATES: int = 0
"""Required every requested target-design inference attempt to complete."""

RELEASE_REPLICATES_PER_CELL: int = 200
"""Fixed the eventual campaign denominator for each scenario and censoring level."""

REQUIRED_TEST_RULE: str = "mixture_50_50"
"""Pinned the public boundary-test reference distribution."""

REQUIRED_INTERVAL_LEVEL: float = 0.95
"""Pinned the public profile-likelihood confidence level."""

MEAN_COEFFICIENTS: npt.NDArray[np.float64] = np.asarray(
    [2.0, 0.30, -0.10, 0.50, 0.05, -0.02],
    dtype=np.float64,
)
"""Reused the predeclared six-column participant-free target mean."""


@dataclass(frozen=True)
class TargetDesign:
    """Hold the participant-free relationship and fixed-effect envelope."""

    relationship: npt.NDArray[np.float64]
    """Stored the structure-matched synthetic additive relationship."""

    design: npt.NDArray[np.float64]
    """Stored the participant-free six-column fixed-effect design."""

    component_sizes: tuple[int, ...]
    """Retained only the reviewed relationship-component sizes."""

    fixture_sha256: str
    """Bound every generated array to the exact reviewed fixture bytes."""


@dataclass(frozen=True)
class ReplicateResult:
    """Retain one unconditional target-design attempt."""

    scenario: str
    """Selected null testing or heritable interval coverage."""

    censoring_share: float
    """Selected one of the two predeclared upper-censoring levels."""

    replicate: int
    """Retained the zero-based denominator position."""

    seed: int
    """Retained the schedule-independent outcome stream."""

    outcome: str
    """Classified complete inference or one exact failure bucket."""

    p_value: float | None = None
    """Recorded a valid public p-value for a complete null attempt."""

    test_rule: str | None = None
    """Recorded the exact public boundary reference for complete inference."""

    covered: bool | None = None
    """Recorded interval containment for a complete heritable attempt."""

    fit_heritability: float | None = None
    """Recorded the public complete-trait point estimate."""

    fit_total_variance: float | None = None
    """Recorded the public latent complete-trait variance estimate."""

    interval_lower: float | None = None
    """Recorded the public profile interval's lower endpoint."""

    interval_upper: float | None = None
    """Recorded the public profile interval's upper endpoint."""

    profile_failures: int | None = None
    """Recorded failed constrained evaluations for interval diagnostics."""

    error_code: str | None = None
    """Retained a stable refusal or malformed-record code without outcome data."""


@dataclass(frozen=True)
class ObservedTobit:
    """Hold one synthetic censored response submitted to the public API."""

    value: npt.NDArray[np.float64]
    """Stored measured values and NaN only where the status says censored."""

    censoring: npt.NDArray[np.int64]
    """Stored zero for measured and one for right-censored rows."""

    limit: npt.NDArray[np.float64]
    """Stored the common participant-free censoring threshold."""

    censored_count: int
    """Recorded the exact number of censored rows."""

    achieved_censoring_share: float
    """Recorded the attainable finite-sample censoring share."""


@dataclass(frozen=True)
class WorkerState:
    """Hold read-only target arrays and precomputed generating factors."""

    target: TargetDesign
    """Stored the participant-free structure and fixed design."""

    covariance_factors_by_scenario: Mapping[
        str,
        tuple[npt.NDArray[np.float64], ...],
    ]
    """Stored family-block generating factors at both scientific truths."""

    mean_coefficients: npt.NDArray[np.float64]
    """Stored the six predeclared latent-mean coefficients."""

    true_heritability: float
    """Stored the interior truth used by heritable cells."""

    required_test_rule: str
    """Pinned the public boundary-test reference."""

    required_interval_level: float
    """Pinned the public profile confidence level."""


@dataclass(frozen=True)
class ReplicateJob:
    """Describe one scheduling-independent Tobit target attempt."""

    scenario: str
    """Selected the null or interior generating truth."""

    censoring_share: float
    """Selected the intended finite-roster right-censoring share."""

    replicate: int
    """Stored the zero-based coordinate retained in every denominator."""

    seed: int
    """Stored the deterministic outcome stream independently of scheduling."""


@dataclass(frozen=True)
class CampaignConfiguration:
    """Hold every explicit scientific and scheduling coordinate."""

    replicates: int
    """Stored the requested denominator per scenario and censoring level."""

    workers: int
    """Stored the maximum number of local worker processes."""

    censoring_shares: tuple[float, ...]
    """Stored both predeclared high-frequency censoring levels."""

    true_heritability: float
    """Stored the interior complete-trait heritability truth."""

    true_variance: float
    """Stored the latent complete-trait variance."""

    level: float
    """Stored the nominal boundary-test rejection level."""

    nominal_coverage: float
    """Stored the intended profile-interval coverage."""

    monte_carlo_confidence: float
    """Stored the one-sided exact-binomial confidence."""

    base_seed: int
    """Stored the first schedule-independent outcome stream."""

    scenario_seed_offset: int
    """Stored the offset separating null and heritable streams."""

    rate_seed_offset: int
    """Stored the offset separating censoring-level streams."""

    maximum_failed_replicates: int
    """Stored the fail-closed numerical-failure allowance."""


WORKER_STATE: list[WorkerState] = []
"""Held the sole immutable state installed in each local or pool worker."""


def load_target_design(path: Path, *, expected_sha256: str) -> TargetDesign:
    """Build the Tobit target from the reviewed shared structural fixture."""
    envelope: object = target_envelope(
        path,
        expected_sha256=expected_sha256,
        families=202,
    )
    """Delegated exact-byte validation and synthetic pedigree generation."""

    return TargetDesign(
        relationship=envelope.relationship,
        design=envelope.fixed_effects,
        component_sizes=envelope.component_sizes,
        fixture_sha256=envelope.fixture_sha256,
    )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Read every scientific coordinate from an explicit command."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built the fixed target-design campaign command boundary."""

    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--censoring-shares",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument("--true-heritability", type=float, required=True)
    parser.add_argument("--true-variance", type=float, required=True)
    parser.add_argument("--level", type=float, required=True)
    parser.add_argument("--nominal-coverage", type=float, required=True)
    parser.add_argument("--monte-carlo-confidence", type=float, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--scenario-seed-offset", type=int, required=True)
    parser.add_argument("--rate-seed-offset", type=int, required=True)
    parser.add_argument("--maximum-failed-replicates", type=int, required=True)
    parser.add_argument("--design-fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--no-write", action="store_true", required=True)
    parsed: argparse.Namespace = parser.parse_args(arguments)
    """Parsed the shell-free release and bounded-smoke argument vectors."""

    fixed_coordinates: tuple[object, ...] = (
        tuple(parsed.censoring_shares),
        parsed.true_heritability,
        parsed.true_variance,
        parsed.level,
        parsed.nominal_coverage,
        parsed.monte_carlo_confidence,
        parsed.base_seed,
        parsed.scenario_seed_offset,
        parsed.rate_seed_offset,
        parsed.maximum_failed_replicates,
        parsed.fixture_sha256,
    )
    """Collected every coordinate forbidden from drifting between campaign runs."""

    expected_coordinates: tuple[object, ...] = (
        CENSORING_SHARES,
        TRUE_HERITABILITY,
        TRUE_VARIANCE,
        LEVEL,
        NOMINAL_COVERAGE,
        MONTE_CARLO_CONFIDENCE,
        BASE_SEED,
        SCENARIO_SEED_OFFSET,
        RATE_SEED_OFFSET,
        MAXIMUM_FAILED_REPLICATES,
        "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e",
    )
    """Restated the reviewed fixture and prewritten scientific decision exactly."""

    if parsed.replicates < 1 or parsed.workers < 1:
        parser.error("--replicates and --workers must be positive")
    if fixed_coordinates != expected_coordinates:
        parser.error("scientific coordinates do not match the prewritten Tobit gate")
    return parsed


def covariance_factors(
    target: TargetDesign,
    *,
    heritability: float,
    total_variance: float,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Factor the generating covariance separately inside each family."""
    if not 0.0 <= heritability <= 1.0 or total_variance <= 0.0:
        raise ValueError("TOBIT_TARGET_GENERATING_VARIANCE_INVALID")

    factors: list[npt.NDArray[np.float64]] = []
    """Collected independent family-block Cholesky factors."""

    offset: int = 0
    """Tracked contiguous component slices in the generated relationship."""

    for size in target.component_sizes:
        stop: int = offset + size
        """Located the exclusive end of this relationship component."""

        relationship: npt.NDArray[np.float64] = target.relationship[
            offset:stop, offset:stop
        ]
        """Selected one structure-matched synthetic family block."""

        covariance: npt.NDArray[np.float64] = total_variance * (
            heritability * relationship + (1.0 - heritability) * np.eye(size)
        )
        """Combined additive and residual variance at the selected truth."""

        factors.append(np.linalg.cholesky(covariance))
        offset = stop
        """Factored this family and advanced to the next independent block."""
    return tuple(factors)


def simulate_observation(
    target: TargetDesign,
    factors: tuple[npt.NDArray[np.float64], ...],
    *,
    mean_coefficients: npt.NDArray[np.float64],
    censoring_share: float,
    seed: int,
) -> ObservedTobit:
    """Generate one exact-count right-censored participant-free response."""
    people: int = target.relationship.shape[0]
    """Read the target roster size from its generated relationship matrix."""

    if (
        target.design.shape[0] != people
        or mean_coefficients.shape != (target.design.shape[1],)
        or len(factors) != len(target.component_sizes)
        or not 0.0 < censoring_share < 1.0
    ):
        raise ValueError("TOBIT_TARGET_SIMULATION_COORDINATES_INVALID")

    generator: np.random.Generator = np.random.default_rng(seed)
    """Created the sole deterministic outcome stream for this attempt."""

    complete: npt.NDArray[np.float64] = target.design @ mean_coefficients
    """Started the latent complete trait at its synthetic fixed mean."""

    offset: int = 0
    """Tracked family slices while adding correlated Gaussian draws."""

    for size, factor in zip(target.component_sizes, factors, strict=True):
        stop: int = offset + size
        """Located the response rows governed by this family factor."""

        complete[offset:stop] += factor @ generator.standard_normal(size)
        """Added one independent family draw to the latent response."""

        offset = stop
        """Advanced the response cursor to the next relationship component."""
    """Generated the complete latent trait without a dense whole-roster factor."""

    censored_count: int = round(censoring_share * people)
    """Selected the nearest attainable finite-roster censoring count."""

    if not 0 < censored_count < people:
        raise ValueError("TOBIT_TARGET_CENSORING_COUNT_INVALID")

    ordered: npt.NDArray[np.int64] = np.argsort(complete, kind="stable")
    """Ordered continuous latent values deterministically for exact censoring."""

    last_measured: float = float(complete[ordered[-censored_count - 1]])
    """Read the largest value retained as measured."""

    first_censored: float = float(complete[ordered[-censored_count]])
    """Read the smallest value assigned to the censored upper tail."""

    threshold: float = last_measured + (first_censored - last_measured) / 2.0
    """Placed the common limit strictly between measured and censored values."""

    censored: npt.NDArray[np.bool_] = complete >= threshold
    """Applied the common participant-free instrument limit."""

    if int(censored.sum()) != censored_count:
        raise ValueError("TOBIT_TARGET_CENSORING_COUNT_INVALID")

    censoring: npt.NDArray[np.int64] = censored.astype(np.int64)
    """Encoded right censoring with the documented public status code."""

    return ObservedTobit(
        value=np.where(censored, np.nan, complete),
        censoring=censoring,
        limit=np.full(people, threshold),
        censored_count=censored_count,
        achieved_censoring_share=censored_count / people,
    )


def start_worker(state: WorkerState) -> None:
    """Install immutable target state once per campaign worker."""
    WORKER_STATE.clear()
    """Removed any state retained from an earlier in-process campaign."""

    WORKER_STATE.append(state)
    """Retained one shared state instead of serialising arrays with every job."""


def finite_record_number(record: Mapping[str, object], field: str) -> float | None:
    """Read one finite non-Boolean public record number without guessing."""
    value: object = record.get(field)
    """Selected the exact documented field without accepting aliases."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number: float = float(value)
    """Normalised NumPy and Python scalar representations."""
    return number if np.isfinite(number) else None


def interval_contains(
    interval: Mapping[str, object],
    truth: float,
) -> bool:
    """Score profile containment with the public boundary-mixture verdict."""
    lower: float | None = finite_record_number(interval, "lower")
    """Read the validated lower endpoint again at the scoring seam."""

    upper: float | None = finite_record_number(interval, "upper")
    """Read the validated upper endpoint again at the scoring seam."""

    if lower is None or upper is None or truth < lower or truth > upper:
        return False
    if truth == 0.0 and lower == 0.0:
        return interval.get("contains_lower_bound") is True
    if truth == 1.0 and upper == 1.0:
        return interval.get("contains_upper_bound") is True
    return True


def run_replicate(job: ReplicateJob) -> ReplicateResult:
    """Run fit, interval, and test through the documented Python interface."""
    state: WorkerState | None = WORKER_STATE[0] if WORKER_STATE else None
    """Read the immutable state installed by local or process orchestration."""

    if state is None:
        raise RuntimeError("TOBIT_TARGET_WORKER_NOT_INITIALISED")
    if job.scenario not in SCENARIOS:
        raise ValueError("TOBIT_TARGET_SCENARIO_INVALID")

    factors: tuple[npt.NDArray[np.float64], ...] | None = (
        state.covariance_factors_by_scenario.get(job.scenario)
    )
    """Selected the generating covariance fixed for this scientific scenario."""

    if factors is None:
        raise ValueError("TOBIT_TARGET_SCENARIO_FACTORS_MISSING")

    observed: ObservedTobit = simulate_observation(
        state.target,
        factors,
        mean_coefficients=state.mean_coefficients,
        censoring_share=job.censoring_share,
        seed=job.seed,
    )
    """Generated the exact-count censored response for this attempt."""

    people: int = state.target.relationship.shape[0]
    """Read the roster size governing the intended finite-count censoring."""

    expected_censored: int = round(job.censoring_share * people)
    """Recomputed the count independently of the simulation record."""

    observed_count: int = int(np.count_nonzero(observed.censoring == 1))
    """Counted public right-censoring codes submitted to the estimator."""

    if (
        observed.censored_count != expected_censored
        or observed_count != expected_censored
        or not np.isclose(
            observed.achieved_censoring_share,
            expected_censored / people,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        return ReplicateResult(
            scenario=job.scenario,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="invalid_censoring_count",
            error_code="TOBIT_TARGET_CENSORING_COUNT_INVALID",
        )

    public_arguments: tuple[object, ...] = (
        state.target.relationship,
        observed.value,
        observed.censoring,
        observed.limit,
        state.target.design,
    )
    """Assembled the sole documented positional Tobit input boundary."""

    try:
        fit_value: object = asterism.tobit_fit(*public_arguments)
        """Fitted the complete latent trait through the public package root."""
    except ValueError as refusal:
        return ReplicateResult(
            scenario=job.scenario,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="refused",
            error_code=str(refusal),
        )

    if not isinstance(fit_value, Mapping):
        fit: Mapping[str, object] = {}
        """Represented a malformed public return as an empty record."""
    else:
        fit = fit_value
        """Narrowed the returned public fit record without changing fields."""

    fit_heritability: float | None = finite_record_number(fit, "heritability")
    """Read the point estimate only when finite."""

    fit_variance: float | None = finite_record_number(fit, "total_variance")
    """Read the latent complete-trait scale only when finite."""

    fit_censored_share: float | None = finite_record_number(fit, "censored_share")
    """Read the public input-accounting diagnostic only when finite."""

    largest_family: float | None = finite_record_number(fit, "largest_family")
    """Read the public target-structure diagnostic only when finite."""

    fit_complete: bool = (
        fit.get("converged") is True
        and fit_heritability is not None
        and 0.0 <= fit_heritability <= 1.0
        and fit_variance is not None
        and fit_variance > 0.0
        and fit_censored_share is not None
        and np.isclose(
            fit_censored_share,
            observed.achieved_censoring_share,
            rtol=0.0,
            atol=1e-12,
        )
        and largest_family is not None
        and largest_family == max(state.target.component_sizes)
    )
    """Required a converged finite fit matching the submitted target design."""

    if not fit_complete:
        return ReplicateResult(
            scenario=job.scenario,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="nonconverged",
            fit_heritability=fit_heritability,
            fit_total_variance=fit_variance,
            error_code="TOBIT_TARGET_FIT_RECORD_INCOMPLETE",
        )

    try:
        interval_value: object = asterism.tobit_interval(*public_arguments)
        """Profiled complete-trait heritability through the public package root."""
    except ValueError as refusal:
        return ReplicateResult(
            scenario=job.scenario,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="interval_failed",
            fit_heritability=fit_heritability,
            fit_total_variance=fit_variance,
            error_code=str(refusal),
        )

    if not isinstance(interval_value, Mapping):
        interval: Mapping[str, object] = {}
        """Represented a malformed public return as an empty record."""
    else:
        interval = interval_value
        """Narrowed the returned public interval without changing fields."""

    interval_estimate: float | None = finite_record_number(interval, "estimate")
    """Read the profile point only when finite."""

    interval_lower: float | None = finite_record_number(interval, "lower")
    """Read the lower profile endpoint only when finite."""

    interval_upper: float | None = finite_record_number(interval, "upper")
    """Read the upper profile endpoint only when finite."""

    interval_level: float | None = finite_record_number(interval, "level")
    """Read the actual public confidence level rather than assuming it."""

    profile_failures_value: object = interval.get("profile_failures")
    """Selected the required failed-profile diagnostic without coercion."""

    profile_failures: int | None = (
        profile_failures_value
        if isinstance(profile_failures_value, int)
        and not isinstance(profile_failures_value, bool)
        and profile_failures_value >= 0
        else None
    )
    """Accepted only a nonnegative integer profile-failure count."""

    interval_complete: bool = (
        interval_estimate is not None
        and interval_lower is not None
        and interval_upper is not None
        and 0.0 <= interval_lower <= interval_estimate <= interval_upper <= 1.0
        and interval_level is not None
        and np.isclose(
            interval_level,
            state.required_interval_level,
            rtol=0.0,
            atol=1e-12,
        )
        and profile_failures is not None
    )
    """Required the complete predeclared public interval record."""

    if not interval_complete:
        return ReplicateResult(
            scenario=job.scenario,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="interval_failed",
            fit_heritability=fit_heritability,
            fit_total_variance=fit_variance,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            profile_failures=profile_failures,
            error_code="TOBIT_TARGET_INTERVAL_RECORD_INCOMPLETE",
        )
    if profile_failures > 0:
        return ReplicateResult(
            scenario=job.scenario,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="profile_failed",
            fit_heritability=fit_heritability,
            fit_total_variance=fit_variance,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            profile_failures=profile_failures,
            error_code="TOBIT_TARGET_PROFILE_EVALUATION_FAILED",
        )

    try:
        test_value: object = asterism.tobit_test(*public_arguments)
        """Tested zero heritability through the public package root."""
    except ValueError as refusal:
        return ReplicateResult(
            scenario=job.scenario,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="test_failed",
            fit_heritability=fit_heritability,
            fit_total_variance=fit_variance,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            profile_failures=profile_failures,
            error_code=str(refusal),
        )

    if not isinstance(test_value, Mapping):
        test: Mapping[str, object] = {}
        """Represented a malformed public return as an empty record."""
    else:
        test = test_value
        """Narrowed the returned public test without changing fields."""

    statistic: float | None = finite_record_number(test, "statistic")
    """Read the likelihood-ratio statistic only when finite."""

    p_value: float | None = finite_record_number(test, "p_value")
    """Read the boundary-mixture p-value only when finite."""

    test_rule_value: object = test.get("rule")
    """Selected the public reference-distribution identity exactly."""

    test_complete: bool = (
        statistic is not None
        and statistic >= 0.0
        and p_value is not None
        and 0.0 <= p_value <= 1.0
        and test_rule_value == state.required_test_rule
    )
    """Required finite inference and the predeclared boundary reference."""

    if not test_complete:
        return ReplicateResult(
            scenario=job.scenario,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="test_failed",
            fit_heritability=fit_heritability,
            fit_total_variance=fit_variance,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            profile_failures=profile_failures,
            p_value=p_value,
            error_code="TOBIT_TARGET_TEST_RECORD_INCOMPLETE",
        )

    truth: float = 0.0 if job.scenario == "null" else state.true_heritability
    """Selected the boundary or interior coverage truth for this scenario."""

    return ReplicateResult(
        scenario=job.scenario,
        censoring_share=job.censoring_share,
        replicate=job.replicate,
        seed=job.seed,
        outcome=COMPLETE_OUTCOME,
        p_value=p_value,
        test_rule=str(test_rule_value),
        covered=interval_contains(interval, truth),
        fit_heritability=fit_heritability,
        fit_total_variance=fit_variance,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        profile_failures=profile_failures,
    )


def campaign_jobs(
    *,
    replicates: int,
    censoring_shares: tuple[float, ...],
    base_seed: int,
    scenario_seed_offset: int,
    rate_seed_offset: int,
) -> list[ReplicateJob]:
    """Enumerate every scenario, censoring level, and replicate exactly once."""
    if replicates < 1:
        raise ValueError("TOBIT_TARGET_REPLICATES_INVALID")

    jobs: list[ReplicateJob] = []
    """Collected jobs in a stable rate, scenario, replicate order."""

    for rate_index, censoring_share in enumerate(censoring_shares):
        for scenario_index, scenario in enumerate(SCENARIOS):
            for replicate in range(replicates):
                jobs.append(
                    ReplicateJob(
                        scenario=scenario,
                        censoring_share=censoring_share,
                        replicate=replicate,
                        seed=(
                            base_seed
                            + rate_index * rate_seed_offset
                            + scenario_index * scenario_seed_offset
                            + replicate
                        ),
                    )
                )
                """Bound one denominator row to its schedule-independent seed."""
    return jobs


def run_campaign(
    target: TargetDesign,
    configuration: CampaignConfiguration,
) -> list[ReplicateResult]:
    """Run all target cells through the three public Tobit calculations."""
    factors: dict[str, tuple[npt.NDArray[np.float64], ...]] = {
        "null": covariance_factors(
            target,
            heritability=0.0,
            total_variance=configuration.true_variance,
        ),
        "heritable": covariance_factors(
            target,
            heritability=configuration.true_heritability,
            total_variance=configuration.true_variance,
        ),
    }
    """Factored each generating covariance once before scheduling attempts."""

    state: WorkerState = WorkerState(
        target=target,
        covariance_factors_by_scenario=factors,
        mean_coefficients=MEAN_COEFFICIENTS,
        true_heritability=configuration.true_heritability,
        required_test_rule=REQUIRED_TEST_RULE,
        required_interval_level=REQUIRED_INTERVAL_LEVEL,
    )
    """Built one immutable public-worker state from reviewed coordinates."""

    jobs: list[ReplicateJob] = campaign_jobs(
        replicates=configuration.replicates,
        censoring_shares=configuration.censoring_shares,
        base_seed=configuration.base_seed,
        scenario_seed_offset=configuration.scenario_seed_offset,
        rate_seed_offset=configuration.rate_seed_offset,
    )
    """Enumerated every denominator row before worker scheduling."""

    if configuration.workers == 1:
        start_worker(state)
        """Installed the same immutable state in the current process."""

        results: list[ReplicateResult] = [run_replicate(job) for job in jobs]
        """Ran a bounded development smoke without process setup overhead."""
        return results

    workers_used: int = min(configuration.workers, len(jobs))
    """Capped processes at the number of independently requested attempts."""

    with ProcessPoolExecutor(
        max_workers=workers_used,
        initializer=start_worker,
        initargs=(state,),
    ) as pool:
        results = list(pool.map(run_replicate, jobs, chunksize=1))
        """Ran all attempts without filtering any failed worker record."""
    return results


def one_sided_lower_limit(
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    """Return an exact one-sided Clopper--Pearson lower limit."""
    if trials < 1 or successes < 0 or successes > trials or not 0.0 < confidence < 1.0:
        raise ValueError("TOBIT_TARGET_BINOMIAL_COUNTS_INVALID")
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
    """Return an exact one-sided Clopper--Pearson upper limit."""
    if trials < 1 or successes < 0 or successes > trials or not 0.0 < confidence < 1.0:
        raise ValueError("TOBIT_TARGET_BINOMIAL_COUNTS_INVALID")
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
    """Apply the prewritten validity-first null rejection decision."""
    lower: float = one_sided_lower_limit(rejections, attempted, confidence)
    """Quantified whether the rejection rate can remain at the nominal level."""

    anti_conservative: bool = lower > level and not np.isclose(
        lower,
        level,
        rtol=0.0,
        atol=1e-12,
    )
    """Preserved boundary equality across beta-quantile round-off."""

    return {
        "attempted": attempted,
        "rejections": rejections,
        "rate": rejections / attempted,
        "one_sided_clopper_pearson_lower": lower,
        "level": level,
        "confidence": confidence,
        "verdict": (
            "anti_conservative" if anti_conservative else "compatible_or_conservative"
        ),
        "passed": not anti_conservative,
    }


def coverage_decision(
    *,
    covered: int,
    attempted: int,
    nominal: float,
    confidence: float,
) -> dict[str, object]:
    """Apply the prewritten undercoverage decision."""
    upper: float = one_sided_upper_limit(covered, attempted, confidence)
    """Quantified whether nominal coverage remains compatible with the campaign."""

    under_covering: bool = upper < nominal and not np.isclose(
        upper,
        nominal,
        rtol=0.0,
        atol=1e-12,
    )
    """Preserved boundary equality across beta-quantile round-off."""

    return {
        "attempted": attempted,
        "covered": covered,
        "coverage": covered / attempted,
        "one_sided_clopper_pearson_upper": upper,
        "nominal": nominal,
        "confidence": confidence,
        "verdict": (
            "under_covering" if under_covering else "compatible_or_conservative"
        ),
        "passed": not under_covering,
    }


def score_campaign(
    rows: list[ReplicateResult],
    *,
    replicates_per_cell: int,
    censoring_shares: tuple[float, ...],
    level: float,
    nominal_coverage: float,
    monte_carlo_confidence: float,
    maximum_failed_replicates: int,
) -> dict[str, object]:
    """Score every requested target attempt without changing denominators."""
    if replicates_per_cell < 1:
        raise ValueError("TOBIT_TARGET_REPLICATES_INVALID")
    expected_attempts: int = (
        replicates_per_cell * len(censoring_shares) * len(SCENARIOS)
    )
    """Computed the complete requested grid before reading any outcomes."""

    if len(rows) != expected_attempts:
        raise ValueError("TOBIT_TARGET_CELL_COORDINATES_INVALID")

    allowed_outcomes: set[str] = {COMPLETE_OUTCOME, *FAILURE_OUTCOMES}
    """Restricted worker output to the predeclared exhaustive outcome vocabulary."""

    if any(row.outcome not in allowed_outcomes for row in rows):
        raise ValueError("TOBIT_TARGET_OUTCOME_INVALID")

    failure_counts: Counter[str] = Counter(
        row.outcome for row in rows if row.outcome in FAILURE_OUTCOMES
    )
    """Counted every exact failure bucket across all requested cells."""

    rates: dict[str, object] = {}
    """Accumulated unconditional evidence separately by censoring level."""

    scientific_decisions: list[bool] = []
    """Collected the fixed exact-binomial decisions for the aggregate verdict."""

    for censoring_share in censoring_shares:
        rate_record: dict[str, object] = {}
        """Accumulated the null and heritable cells at this censoring level."""

        for scenario in SCENARIOS:
            selected: list[ReplicateResult] = [
                row
                for row in rows
                if row.censoring_share == censoring_share and row.scenario == scenario
            ]
            """Selected every requested attempt without filtering failure outcomes."""

            coordinates: list[int] = [row.replicate for row in selected]
            """Retained multiplicity while validating the fixed denominator grid."""

            if len(selected) != replicates_per_cell or sorted(coordinates) != list(
                range(replicates_per_cell)
            ):
                raise ValueError("TOBIT_TARGET_CELL_COORDINATES_INVALID")
            attempted: int = len(selected)
            """Fixed the scientific denominator before reading computed inference."""

            complete: list[ReplicateResult] = [
                row for row in selected if row.outcome == COMPLETE_OUTCOME
            ]
            """Selected complete inference only for numerator calculations."""

            outcomes: Counter[str] = Counter(row.outcome for row in selected)
            """Retained complete and failed attempt multiplicities for this cell."""

            if scenario == "null":
                if any(row.p_value is None for row in complete):
                    raise ValueError("TOBIT_TARGET_COMPLETE_TEST_MISSING")
                rejections: int = sum(
                    row.p_value is not None and row.p_value <= level for row in complete
                )
                """Counted valid complete-test rejections against every attempt."""

                decision: dict[str, object] = level_decision(
                    rejections=rejections,
                    attempted=attempted,
                    level=level,
                    confidence=monte_carlo_confidence,
                )
                """Applied the prewritten anti-conservative rejection rule."""

                rate_record[scenario] = {
                    "attempted": attempted,
                    "computed": len(complete),
                    "outcomes": {
                        outcome: outcomes.get(outcome, 0)
                        for outcome in (COMPLETE_OUTCOME, *FAILURE_OUTCOMES)
                    },
                    "level": decision,
                }
                """Stored unconditional boundary-test evidence for this cell."""
            else:
                if any(row.covered is None for row in complete):
                    raise ValueError("TOBIT_TARGET_COMPLETE_INTERVAL_MISSING")
                covered: int = sum(row.covered is True for row in complete)
                """Counted complete covering intervals against every attempt."""

                decision = coverage_decision(
                    covered=covered,
                    attempted=attempted,
                    nominal=nominal_coverage,
                    confidence=monte_carlo_confidence,
                )
                """Applied the prewritten undercoverage rule."""

                rate_record[scenario] = {
                    "attempted": attempted,
                    "computed": len(complete),
                    "outcomes": {
                        outcome: outcomes.get(outcome, 0)
                        for outcome in (COMPLETE_OUTCOME, *FAILURE_OUTCOMES)
                    },
                    "coverage": decision,
                }
                """Stored unconditional interval-coverage evidence for this cell."""
            scientific_decisions.append(decision["passed"] is True)
            """Retained this cell's exact-binomial acceptance decision."""
        rates[str(censoring_share)] = rate_record
        """Stored both scientific scenarios under the canonical decimal rate."""

    failed_replicates: int = sum(failure_counts.values())
    """Counted all unsuccessful attempts independently of binomial decisions."""

    return {
        "rates": rates,
        "failure_buckets": {
            outcome: failure_counts.get(outcome, 0) for outcome in FAILURE_OUTCOMES
        },
        "failed_replicates": failed_replicates,
        "maximum_failed_replicates": maximum_failed_replicates,
        "passed": (
            all(scientific_decisions) and failed_replicates <= maximum_failed_replicates
        ),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the explicit target campaign and emit one values-free JSON record."""
    parsed: argparse.Namespace = parse_arguments(arguments)
    """Fixed every scientific and scheduling coordinate before generation."""

    target: TargetDesign = load_target_design(
        parsed.design_fixture,
        expected_sha256=parsed.fixture_sha256,
    )
    """Verified exact reviewed fixture bytes before generating any outcome."""

    configuration: CampaignConfiguration = CampaignConfiguration(
        replicates=parsed.replicates,
        workers=parsed.workers,
        censoring_shares=tuple(parsed.censoring_shares),
        true_heritability=parsed.true_heritability,
        true_variance=parsed.true_variance,
        level=parsed.level,
        nominal_coverage=parsed.nominal_coverage,
        monte_carlo_confidence=parsed.monte_carlo_confidence,
        base_seed=parsed.base_seed,
        scenario_seed_offset=parsed.scenario_seed_offset,
        rate_seed_offset=parsed.rate_seed_offset,
        maximum_failed_replicates=parsed.maximum_failed_replicates,
    )
    """Retained the complete command contract in machine-readable evidence."""

    started: float = time.perf_counter()
    """Started bounded timing after deterministic target construction."""

    attempts: list[ReplicateResult] = run_campaign(target, configuration)
    """Retained one result for every requested fit, interval, and test attempt."""

    score: dict[str, object] = score_campaign(
        attempts,
        replicates_per_cell=configuration.replicates,
        censoring_shares=configuration.censoring_shares,
        level=configuration.level,
        nominal_coverage=configuration.nominal_coverage,
        monte_carlo_confidence=configuration.monte_carlo_confidence,
        maximum_failed_replicates=configuration.maximum_failed_replicates,
    )
    """Applied exact one-sided binomial rules to unconditional denominators."""

    failures: list[str] = []
    """Collected concise consequences without changing machine decisions."""

    if score["failed_replicates"] > configuration.maximum_failed_replicates:
        failures.append("one or more required public inferences failed")

    rates_value: object = score["rates"]
    """Selected the independently scored censoring-level records."""

    if isinstance(rates_value, Mapping):
        for censoring_share, rate_value in rates_value.items():
            if not isinstance(rate_value, Mapping):
                continue
            null_value: object = rate_value.get("null")
            """Selected the boundary-test decision at this censoring level."""

            heritable_value: object = rate_value.get("heritable")
            """Selected the interior-coverage decision at this censoring level."""

            if isinstance(null_value, Mapping):
                level_value: object = null_value.get("level")
                """Read the exact-binomial level decision without assuming shape."""

                if (
                    isinstance(level_value, Mapping)
                    and level_value.get("passed") is not True
                ):
                    failures.append(
                        f"type-I error is anti-conservative at censoring {censoring_share}"
                    )
            if isinstance(heritable_value, Mapping):
                coverage_value: object = heritable_value.get("coverage")
                """Read the exact-binomial coverage decision without assuming shape."""

                if (
                    isinstance(coverage_value, Mapping)
                    and coverage_value.get("passed") is not True
                ):
                    failures.append(
                        f"profile intervals under-cover at censoring {censoring_share}"
                    )
    """Explained each failed scientific cell while preserving the score record."""

    exact_release_campaign: bool = (
        configuration.replicates == RELEASE_REPLICATES_PER_CELL
    )
    """Separated bounded development execution from eventual release evidence."""

    people: int = target.relationship.shape[0]
    """Read the target roster size governing attainable censoring fractions."""

    censoring_counts: list[int] = [
        round(censoring_share * people)
        for censoring_share in configuration.censoring_shares
    ]
    """Recorded the exact finite-roster censoring counts submitted to inference."""

    achieved_censoring_shares: list[float] = [
        count / people for count in censoring_counts
    ]
    """Distinguished exact attained fractions from their named target levels."""

    evidence: dict[str, object] = {
        "check": "tobit_target_design",
        "analysis_id": "one_trait_censored",
        "participant_free": True,
        "target_contract": "aggregate_structure_match_not_reconstruction",
        "fixture_sha256": target.fixture_sha256,
        "public_interface": [
            "asterism.tobit_fit",
            "asterism.tobit_interval",
            "asterism.tobit_test",
        ],
        "design_facts": {
            "people": people,
            "relationship_components": len(target.component_sizes),
            "largest_family": max(target.component_sizes),
            "fixed_effect_columns": target.design.shape[1],
            "censoring_shares": list(configuration.censoring_shares),
            "censoring_counts": censoring_counts,
            "achieved_censoring_shares": achieved_censoring_shares,
            "participant_structure_reconstructed": False,
        },
        "configuration": asdict(configuration),
        "release_replicates_per_scenario_per_censoring_share": (
            RELEASE_REPLICATES_PER_CELL
        ),
        "replicates_per_scenario_per_censoring_share": configuration.replicates,
        "attempted_replicates": len(attempts),
        "every_attempted_replicate_in_denominator": True,
        "failure_outcomes": list(FAILURE_OUTCOMES),
        "acceptance": {
            "level_fail_when": ("one_sided_clopper_pearson_lower_above_nominal_level"),
            "coverage_fail_when": (
                "one_sided_clopper_pearson_upper_below_nominal_coverage"
            ),
            "maximum_failed_replicates": configuration.maximum_failed_replicates,
            "required_test_rule": REQUIRED_TEST_RULE,
            "required_interval_level": REQUIRED_INTERVAL_LEVEL,
        },
        "attempts": [asdict(attempt) for attempt in attempts],
        "results": score,
        "exact_release_campaign": exact_release_campaign,
        "development_smoke_only": not exact_release_campaign,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": score["passed"] is True and not failures,
        "failures": failures,
    }
    """Built values-free evidence sufficient to recompute every decision."""

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["passed"] is True else 1


if __name__ == "__main__":
    sys.exit(main())
