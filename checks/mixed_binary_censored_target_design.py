"""Qualify the binary/right-censored model on a values-free target design.

The reviewed fixture retains only aggregate structural facts needed to stress
the sequential Mendell--Elston approximation at 1,909 people and a largest
family of 180. It selects the existing deterministic structure-matched
synthetic pedigree and fixed-effect design; it does not reconstruct participant
rows, relationships, covariates, diagnoses, or hearing thresholds.
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
from scipy.stats import beta, chi2

try:
    from checks.one_trait_coverage import target_envelope
except ModuleNotFoundError:
    from one_trait_coverage import target_envelope
"""Imported the shared participant-free structural generator in both entry modes."""

TARGET_FIXTURE_SHA256: str = (
    "aed92270dcfafb00a46a24bfaf1f3a2b52950fdea1c34deb3740e5060d6f3645"
)
"""Pinned the exact reviewed target-fixture bytes independently of their contents."""

SETTLED_DEVIANCE: float = 1e-6
"""Mirrored the public test contract's numerical point mass at nought."""


@dataclass(frozen=True)
class TargetDesign:
    """Hold the reviewed participant-free mixed-model target.

    Attributes:
        relationship: Generated block-diagonal additive relationship matrix.
        design: Generated six-column fixed-effect design.
        component_sizes: Independent component sizes in generated row order.
        fixture_sha256: Exact mixed-target fixture byte identity.
        source_fixture_sha256: Exact shared structural-fixture byte identity.
        observed_nonzero_relationship_pairs: Reviewed predecessor aggregate.
        generated_nonzero_relationship_pairs: Synthetic structural pair count.
        relative_nonzero_pair_discrepancy: Aggregate structure-match discrepancy.
        prevalence: Generating population diagnosis prevalence.
        heritabilities: Generating liability and complete-hearing heritabilities.
        genetic_correlations: Null and interior reportable-quantity truths.
        residual_correlation: Generating residual cross-trait correlation.
        hearing_variance: Complete-hearing variance before censoring.
        censoring_shares: Intended 16 and 18 kHz right-censoring shares.
        base_seed: First deterministic outcome stream.
        cell_seed_offset: Offset between scientific grid cells.
        replicate_seed_offset: Offset between replicates within a cell.
        minimum_cases: Minimum cases and noncases before model fitting.
        minimum_measured_hearing: Minimum uncensored values before model fitting.
        release_replicates: Frozen release replicate count per cell.
        minimum_smoke_replicates: Smallest reviewed executable smoke.
        nominal_coverage: Profile-interval coverage target.
        nominal_level: Interior zero-correlation test level.
        monte_carlo_confidence: One-sided exact-binomial confidence.
        point_recovery_standard_errors: Maximum point-recovery distance.
        maximum_failed_replicates: Fail-closed numerical-failure allowance.
        required_test_rule: Required public correlation-test reference.
        required_interval_level: Required public profile confidence level.
    """

    relationship: npt.NDArray[np.float64]
    """Stored the complete generated relationship matrix."""

    design: npt.NDArray[np.float64]
    """Stored the complete generated fixed-effect design."""

    component_sizes: tuple[int, ...]
    """Retained generated component sizes from the reviewed histogram."""

    fixture_sha256: str
    """Bound the target to exact mixed-fixture bytes."""

    source_fixture_sha256: str
    """Bound structural generation to exact shared-fixture bytes."""

    observed_nonzero_relationship_pairs: int
    """Retained the non-identifying predecessor pair-count aggregate."""

    generated_nonzero_relationship_pairs: int
    """Counted nonzero pairs in the generated synthetic relationship."""

    relative_nonzero_pair_discrepancy: float
    """Measured pair-count agreement without claiming reconstruction."""

    prevalence: float
    """Fixed the simulated population diagnosis prevalence."""

    heritabilities: tuple[float, float]
    """Fixed generating heritabilities on both latent scales."""

    genetic_correlations: tuple[float, float]
    """Fixed null and interior reportable-quantity truths."""

    residual_correlation: float
    """Fixed the generating residual cross-trait correlation."""

    hearing_variance: float
    """Fixed the complete-hearing variance before censoring."""

    censoring_shares: tuple[float, float]
    """Fixed the two intended high-frequency censoring shares."""

    base_seed: int
    """Selected the first schedule-independent outcome stream."""

    cell_seed_offset: int
    """Separated scientific grid-cell streams."""

    replicate_seed_offset: int
    """Separated replicates within each scientific cell."""

    minimum_cases: int
    """Required enough cases and noncases for an attempted fit."""

    minimum_measured_hearing: int
    """Required enough exact hearing values to identify continuous scale."""

    release_replicates: int
    """Fixed the release campaign size per scientific cell."""

    minimum_smoke_replicates: int
    """Permitted bounded executable checks without changing acceptance."""

    nominal_coverage: float
    """Fixed the intended profile-interval coverage."""

    nominal_level: float
    """Fixed the intended interior correlation-test level."""

    monte_carlo_confidence: float
    """Fixed one-sided exact-binomial confidence."""

    point_recovery_standard_errors: float
    """Fixed the point-recovery Monte Carlo acceptance distance."""

    maximum_failed_replicates: int
    """Required every requested public fit and inference result to succeed."""

    required_test_rule: str
    """Required the documented interior chi-square reference."""

    required_interval_level: float
    """Required the documented 95 per cent profile interval."""


@dataclass(frozen=True)
class CampaignConfiguration:
    """Hold explicit fixture-validated campaign coordinates."""

    replicates: int
    """Retained every attempt in each target cell denominator."""

    workers: int
    """Bounded concurrent public-model evaluations."""

    prevalence: float
    """Retained the reviewed generating diagnosis prevalence."""

    heritabilities: tuple[float, float]
    """Retained the reviewed latent-scale heritabilities."""

    genetic_correlations: tuple[float, float]
    """Retained the reviewed null and interior correlation truths."""

    residual_correlation: float
    """Retained the reviewed residual cross-trait correlation."""

    hearing_variance: float
    """Retained the reviewed complete-hearing variance."""

    censoring_shares: tuple[float, float]
    """Retained the intended 16 and 18 kHz censoring levels."""

    nominal_coverage: float
    """Retained the reviewed profile-coverage target."""

    nominal_level: float
    """Retained the reviewed interior correlation-test level."""

    monte_carlo_confidence: float
    """Retained the reviewed exact-binomial confidence."""

    point_recovery_standard_errors: float
    """Retained the reviewed point-recovery acceptance distance."""

    base_seed: int
    """Retained deterministic first-cell stream ownership."""

    cell_seed_offset: int
    """Retained disjoint scientific cell stream ownership."""

    replicate_seed_offset: int
    """Retained disjoint replicate stream ownership."""

    minimum_cases: int
    """Retained the reviewed minimum case and noncase count."""

    minimum_measured_hearing: int
    """Retained the reviewed minimum uncensored hearing count."""

    maximum_failed_replicates: int
    """Retained the zero-failure release policy."""


@dataclass(frozen=True)
class WorkerState:
    """Hold immutable target arrays and simulation facts inside one worker."""

    relationship: npt.NDArray[np.float64]
    """Shared the generated target relationship matrix."""

    design: npt.NDArray[np.float64]
    """Shared the generated target fixed-effect design."""

    component_sizes: tuple[int, ...]
    """Located contiguous independent relationship blocks."""

    genetic_factors: tuple[npt.NDArray[np.float64], ...]
    """Shared reusable Cholesky factors of generated relationship blocks."""

    prevalence: float
    """Fixed the generating population diagnosis prevalence."""

    heritabilities: tuple[float, float]
    """Fixed generating heritabilities on both latent scales."""

    residual_correlation: float
    """Fixed the generating residual cross-trait correlation."""

    hearing_variance: float
    """Fixed the complete-hearing variance before censoring."""

    minimum_cases: int
    """Required enough cases and noncases before model fitting."""

    minimum_measured_hearing: int
    """Required enough uncensored hearing values to identify scale."""

    nominal_coverage: float
    """Required the public profile confidence level."""

    required_test_rule: str
    """Required the public interior correlation-test reference."""

    required_interval_level: float
    """Required the public profile confidence level exactly."""


WORKER_STATE: list[WorkerState] = []
"""Held the sole immutable state installed in each campaign worker."""


@dataclass(frozen=True)
class ReplicateJob:
    """Identify one independently reproducible mixed-model attempt."""

    genetic_correlation: float
    """Selected the null or interior reportable-quantity truth."""

    censoring_share: float
    """Selected the intended 16 or 18 kHz censoring level."""

    replicate: int
    """Selected one unconditional denominator position within the cell."""

    seed: int
    """Selected one schedule-independent latent-outcome stream."""


@dataclass(frozen=True)
class ReplicateResult:
    """Retain one attempted replicate, including every failure state."""

    genetic_correlation: float
    """Retained scientific correlation-cell identity."""

    censoring_share: float
    """Retained scientific censoring-cell identity."""

    replicate: int
    """Retained the attempted replicate coordinate."""

    seed: int
    """Retained deterministic outcome-stream identity."""

    outcome: str
    """Classified complete inference or one explicit failure mode."""

    cases: int | None = None
    """Recorded generated case count without retaining statuses."""

    measured_hearing: int | None = None
    """Recorded uncensored hearing count without retaining values."""

    converged: bool | None = None
    """Recorded public free-fit convergence when the fit returned."""

    estimate: float | None = None
    """Recorded a validated public genetic-correlation estimate."""

    lower: float | None = None
    """Recorded a validated public lower profile endpoint."""

    upper: float | None = None
    """Recorded a validated public upper profile endpoint."""

    covered: bool | None = None
    """Recorded truth containment or an explicit failed-profile miss."""

    profile_failures: int | None = None
    """Recorded constrained profile failures without filtering the attempt."""

    p_value: float | None = None
    """Recorded a validated public correlation-test p-value."""

    statistic: float | None = None
    """Recorded the matching likelihood-ratio statistic."""

    test_rule: str | None = None
    """Recorded the public null-reference rule."""

    null_loglik: float | None = None
    """Recorded the validated held-null log likelihood."""

    alternative_loglik: float | None = None
    """Recorded the validated free-model log likelihood."""

    error_code: str | None = None
    """Retained a refusal or malformed-record reason without raw data."""


def fixture_sha256(path: Path) -> str:
    """Hash exact fixture bytes without normalising their JSON representation."""
    payload: bytes = path.read_bytes()
    """Read the selected values-free fixture bytes."""
    return hashlib.sha256(payload).hexdigest()


def start_worker(state: WorkerState) -> None:
    """Install immutable simulation state in one process worker."""
    WORKER_STATE.clear()
    """Removed stale state if a test or serial campaign reuses this process."""

    WORKER_STATE.append(state)
    """Installed one complete target state without participant information."""


def simulate_latent_traits(
    state: WorkerState,
    job: ReplicateJob,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate both complete latent traits under the specified covariance.

    Args:
        state: Immutable target arrays and generating covariance facts.
        job: Correlation truth and schedule-independent random stream.

    Returns:
        Unit-scale diagnosis liability and variance-scaled complete hearing.

    Raises:
        ValueError: If the job cannot identify a valid scientific cell.
    """
    if (
        job.replicate < 0
        or job.seed < 0
        or not -1.0 <= job.genetic_correlation <= 1.0
        or not 0.0 < job.censoring_share < 1.0
    ):
        raise ValueError("MIXED_TARGET_REPLICATE_JOB_INVALID")

    generator: np.random.Generator = np.random.default_rng(job.seed)
    """Created the schedule-independent latent-trait stream."""

    genetic_one_blocks: list[npt.NDArray[np.float64]] = []
    """Collected first-trait genetic values in component order."""

    genetic_two_blocks: list[npt.NDArray[np.float64]] = []
    """Collected correlated second-trait genetic values in component order."""

    orthogonal_weight: float = np.sqrt(1.0 - job.genetic_correlation**2)
    """Converted the requested genetic correlation into a stable loading."""

    for size, factor in zip(
        state.component_sizes,
        state.genetic_factors,
        strict=True,
    ):
        first_innovation: npt.NDArray[np.float64] = generator.standard_normal(size)
        """Drew one component's first genetic innovation."""

        second_innovation: npt.NDArray[np.float64] = generator.standard_normal(size)
        """Drew its independent orthogonal genetic innovation."""

        genetic_one_blocks.append(factor @ first_innovation)
        """Applied the component relationship covariance to trait one."""

        genetic_two_blocks.append(
            factor
            @ (
                job.genetic_correlation * first_innovation
                + orthogonal_weight * second_innovation
            )
        )
        """Applied the same covariance with the requested cross-trait loading."""
    """Generated correlated additive effects without a dense global factorisation."""

    genetic_one: npt.NDArray[np.float64] = np.concatenate(genetic_one_blocks)
    """Restored the complete first-trait genetic vector."""

    genetic_two: npt.NDArray[np.float64] = np.concatenate(genetic_two_blocks)
    """Restored the complete second-trait genetic vector."""

    people: int = state.relationship.shape[0]
    """Read the generated target row count independently of fixture claims."""

    residual_one: npt.NDArray[np.float64] = generator.standard_normal(people)
    """Drew independent first-trait residual innovations."""

    residual_orthogonal: npt.NDArray[np.float64] = generator.standard_normal(people)
    """Drew orthogonal innovations for second-trait residual correlation."""

    residual_two: npt.NDArray[np.float64] = (
        state.residual_correlation * residual_one
        + np.sqrt(1.0 - state.residual_correlation**2) * residual_orthogonal
    )
    """Constructed the requested residual cross-trait correlation."""

    liability: npt.NDArray[np.float64] = (
        np.sqrt(state.heritabilities[0]) * genetic_one
        + np.sqrt(1.0 - state.heritabilities[0]) * residual_one
    )
    """Generated the unit-scale latent diagnosis liability."""

    hearing: npt.NDArray[np.float64] = np.sqrt(state.hearing_variance) * (
        np.sqrt(state.heritabilities[1]) * genetic_two
        + np.sqrt(1.0 - state.heritabilities[1]) * residual_two
    )
    """Generated the complete hearing threshold before instrument censoring."""

    return liability, hearing


def simulate_traits(
    state: WorkerState,
    job: ReplicateJob,
) -> tuple[dict[str, object], dict[str, object], int, int]:
    """Apply binary and right-censoring mechanisms to simulated latent traits.

    Args:
        state: Immutable target arrays and observation-mechanism facts.
        job: Censoring share and deterministic latent-outcome stream.

    Returns:
        Two public trait dictionaries, generated cases, and measured hearing count.

    Raises:
        ValueError: If the replicate job does not identify a valid target cell.
    """
    liability, hearing = simulate_latent_traits(state, job)
    """Generated complete latent traits before applying either observation mechanism."""

    people: int = state.relationship.shape[0]
    """Read the generated target row count independently of fixture claims."""

    diagnosis_cut: float = NormalDist().inv_cdf(1.0 - state.prevalence)
    """Fixed the diagnosis threshold from population prevalence."""

    case: npt.NDArray[np.bool_] = liability > diagnosis_cut
    """Applied the prespecified diagnosis threshold to simulated liabilities."""

    hearing_cut: float = np.sqrt(state.hearing_variance) * NormalDist().inv_cdf(
        1.0 - job.censoring_share
    )
    """Fixed the audiometer limit at the prespecified population quantile."""

    censored: npt.NDArray[np.bool_] = hearing >= hearing_cut
    """Applied right censoring without substituting the limit as a measurement."""

    diagnosis: dict[str, object] = {
        "kind": "binary",
        "value": np.full(people, np.nan),
        "censoring": np.where(case, 1, 2).astype(np.int64),
        "limit": np.zeros(people),
    }
    """Encoded case and noncase regions through the documented public contract."""

    censored_hearing: dict[str, object] = {
        "kind": "censored",
        "value": np.where(censored, np.nan, hearing),
        "censoring": np.where(censored, 1, 0).astype(np.int64),
        "limit": np.full(people, hearing_cut),
    }
    """Encoded measured hearing values and upper half-line observations."""

    return diagnosis, censored_hearing, int(case.sum()), int((~censored).sum())


def finite_float(value: object) -> float | None:
    """Narrow one public record value to a finite scalar."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    scalar: float = float(value)
    """Normalised supported Python and NumPy-compatible numeric scalars."""
    return scalar if np.isfinite(scalar) else None


def one_sided_lower_limit(
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    """Return an exact one-sided Clopper--Pearson lower limit."""
    if trials < 1 or successes < 0 or successes > trials or not 0.0 < confidence < 1.0:
        raise ValueError("MIXED_TARGET_BINOMIAL_COUNTS_INVALID")
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
        raise ValueError("MIXED_TARGET_BINOMIAL_COUNTS_INVALID")
    return (
        1.0
        if successes == trials
        else float(beta.ppf(confidence, successes + 1, trials - successes))
    )


def score_campaign(
    results: Sequence[ReplicateResult],
    target: TargetDesign,
) -> dict[str, object]:
    """Score every attempted target replicate without conditional denominators.

    Args:
        results: One retained record for every attempted grid-cell replicate.
        target: Reviewed grid, denominator, and prewritten acceptance rules.

    Returns:
        Machine-readable per-cell evidence and fail-closed campaign decisions.

    Raises:
        ValueError: If cells are missing, unexpected, or duplicated.
    """
    expected_cells: tuple[tuple[float, float], ...] = tuple(
        (correlation, censoring_share)
        for correlation in target.genetic_correlations
        for censoring_share in target.censoring_shares
    )
    """Enumerated the exact fixture-owned scientific grid."""

    seen_cells: set[tuple[float, float]] = {
        (result.genetic_correlation, result.censoring_share) for result in results
    }
    """Collected all supplied cell identities before scoring."""

    if seen_cells != set(expected_cells):
        raise ValueError("MIXED_TARGET_CAMPAIGN_CELL_GRID_INVALID")

    cells: dict[str, object] = {}
    """Collected complete evidence for every scientific cell."""

    all_cells_passed: bool = True
    """Accumulated prewritten scientific decisions across the grid."""

    release_sized: bool = True
    """Required the exact release denominator independently of observed results."""

    for cell_index, (correlation, censoring_share) in enumerate(expected_cells):
        selected: list[ReplicateResult] = [
            result
            for result in results
            if result.genetic_correlation == correlation
            and result.censoring_share == censoring_share
        ]
        """Selected every attempt in this cell, including unavailable inference."""

        replicate_coordinates: set[int] = {result.replicate for result in selected}
        """Collected deterministic denominator positions for uniqueness checks."""

        if len(replicate_coordinates) != len(selected):
            raise ValueError("MIXED_TARGET_CAMPAIGN_REPLICATE_DUPLICATED")

        attempted: int = len(selected)
        """Fixed the unconditional cell denominator."""

        ordered: list[ReplicateResult] = sorted(
            selected,
            key=lambda result: result.replicate,
        )
        """Ordered attempts by their frozen within-cell stream coordinate."""

        coordinates: list[int] = [result.replicate for result in ordered]
        """Retained the exact supplied coordinate sequence for validation."""

        if coordinates != list(range(attempted)):
            raise ValueError("MIXED_TARGET_CAMPAIGN_COORDINATES_INVALID")

        expected_seeds: list[int] = [
            target.base_seed
            + cell_index * target.cell_seed_offset
            + replicate * target.replicate_seed_offset
            for replicate in range(attempted)
        ]
        """Reconstructed every schedule-independent stream from reviewed offsets."""

        if [result.seed for result in ordered] != expected_seeds:
            raise ValueError("MIXED_TARGET_CAMPAIGN_SEEDS_INVALID")

        release_sized = release_sized and attempted == target.release_replicates
        """Compared this cell with the exact prewritten release size."""

        complete: list[ReplicateResult] = [
            result for result in selected if result.outcome == "complete"
        ]
        """Selected complete records solely for fields failures could not provide."""

        failed: int = attempted - len(complete)
        """Counted every noncomplete outcome under the zero-failure rule."""

        covered: int = sum(result.covered is True for result in selected)
        """Counted failed intervals as misses over the unconditional denominator."""

        coverage_upper: float = one_sided_upper_limit(
            covered,
            attempted,
            target.monte_carlo_confidence,
        )
        """Measured whether nominal coverage remains compatible with the campaign."""

        coverage_passed: bool = coverage_upper >= target.nominal_coverage or np.isclose(
            coverage_upper,
            target.nominal_coverage,
            rtol=0.0,
            atol=1e-12,
        )
        """Failed only evidence incompatible with nominal coverage from below."""

        estimates: list[float] = [
            result.estimate for result in complete if result.estimate is not None
        ]
        """Selected complete point estimates without imputing failed attempts."""

        estimate_mean: float | None = float(np.mean(estimates)) if estimates else None
        """Summarised point recovery only when a complete estimate existed."""

        estimate_standard_error: float | None = (
            float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
            if len(estimates) > 1
            else None
        )
        """Measured Monte Carlo uncertainty only from at least two estimates."""

        if estimate_mean is not None and np.isclose(
            estimate_mean,
            correlation,
            rtol=0.0,
            atol=1e-15,
        ):
            recovery_distance: float | None = 0.0
            """Recognised exact recovery even when a one-replicate smoke has no SE."""
        elif estimate_standard_error is not None and estimate_standard_error > 0.0:
            recovery_distance = (
                abs(estimate_mean - correlation) / estimate_standard_error
                if estimate_mean is not None
                else None
            )
            """Expressed nonzero point bias on the prewritten Monte Carlo scale."""
        else:
            recovery_distance = None
            """Refused to invent precision for unavailable or degenerate estimates."""

        recovery_passed: bool = (
            recovery_distance is not None
            and recovery_distance <= target.point_recovery_standard_errors
        )
        """Applied the unchanged three-standard-error point-recovery rule."""

        people: int = target.relationship.shape[0]
        """Read the unconditional target denominator from generated arrays."""

        case_counts: list[int] = [
            result.cases
            for result in selected
            if isinstance(result.cases, int)
            and not isinstance(result.cases, bool)
            and 0 <= result.cases <= people
        ]
        """Retained every valid generated case count, including failed fits."""

        measured_counts: list[int] = [
            result.measured_hearing
            for result in selected
            if isinstance(result.measured_hearing, int)
            and not isinstance(result.measured_hearing, bool)
            and 0 <= result.measured_hearing <= people
        ]
        """Retained every valid uncensored-hearing count, including failed fits."""

        design_observations_complete: bool = (
            len(case_counts) == attempted and len(measured_counts) == attempted
        )
        """Required observation-mechanism evidence for every denominator position."""

        case_shares: list[float] = [count / people for count in case_counts]
        """Converted generated case counts to population-scale observed shares."""

        achieved_censoring_shares: list[float] = [
            1.0 - count / people for count in measured_counts
        ]
        """Converted measured counts to achieved right-censoring shares."""

        valid_tests: list[ReplicateResult] = [
            result
            for result in complete
            if result.test_rule == target.required_test_rule
            and result.p_value is not None
        ]
        """Selected complete tests carrying the required interior reference."""

        rejected: int = sum(
            result.p_value <= target.nominal_level
            for result in valid_tests
            if result.p_value is not None
        )
        """Counted nominal-level rejections among valid public tests."""

        level_lower: float | None = (
            one_sided_lower_limit(
                rejected,
                attempted,
                target.monte_carlo_confidence,
            )
            if correlation == 0.0
            else None
        )
        """Measured type-I error only in the true zero-correlation cells."""

        level_passed: bool = (
            True
            if level_lower is None
            else level_lower <= target.nominal_level
            or np.isclose(
                level_lower,
                target.nominal_level,
                rtol=0.0,
                atol=1e-12,
            )
        )
        """Failed only null evidence incompatible with the nominal level from above."""

        cell_passed: bool = (
            failed <= target.maximum_failed_replicates
            and design_observations_complete
            and len(valid_tests) == attempted
            and coverage_passed
            and recovery_passed
            and level_passed
        )
        """Combined numerical completeness with all prewritten scientific rules."""

        all_cells_passed = all_cells_passed and cell_passed
        """Accumulated the fail-closed grid verdict."""

        key: str = f"rho_g={correlation} censored={censoring_share}"
        """Named the cell in stable scalar form."""

        cells[key] = {
            "attempted": attempted,
            "complete": len(complete),
            "failed": failed,
            "covered": covered,
            "coverage": covered / attempted,
            "coverage_one_sided_clopper_pearson_upper": coverage_upper,
            "coverage_passed": coverage_passed,
            "estimate_mean": estimate_mean,
            "estimate_standard_error": estimate_standard_error,
            "recovery_distance_in_standard_errors": recovery_distance,
            "recovery_passed": recovery_passed,
            "design_observations_complete": design_observations_complete,
            "case_share_min": min(case_shares) if case_shares else None,
            "case_share_max": max(case_shares) if case_shares else None,
            "case_share_mean": (float(np.mean(case_shares)) if case_shares else None),
            "achieved_censoring_share_min": (
                min(achieved_censoring_shares) if achieved_censoring_shares else None
            ),
            "achieved_censoring_share_max": (
                max(achieved_censoring_shares) if achieved_censoring_shares else None
            ),
            "achieved_censoring_share_mean": (
                float(np.mean(achieved_censoring_shares))
                if achieved_censoring_shares
                else None
            ),
            "valid_test_count": len(valid_tests),
            "rejected": rejected,
            "rejection_rate": rejected / attempted,
            "level_one_sided_clopper_pearson_lower": level_lower,
            "level_passed": level_passed,
            "passed": cell_passed,
        }
        """Recorded enough facts to recompute every cell decision."""
    """Scored all exact target cells without dropping any attempt."""

    outcome_counts: Counter[str] = Counter(result.outcome for result in results)
    """Counted every mutually exclusive result state across the campaign."""

    failed_replicates: int = sum(result.outcome != "complete" for result in results)
    """Counted noncomplete attempts independently of per-cell decisions."""

    all_design_observations_complete: bool = all(
        isinstance(cell, dict) and cell["design_observations_complete"] is True
        for cell in cells.values()
    )
    """Required complete achieved-design evidence across every target cell."""

    execution_passed: bool = failed_replicates == 0 and all_design_observations_complete
    """Separated complete bounded execution from statistical qualification."""

    return {
        "attempted_replicates": len(results),
        "failed_replicates": failed_replicates,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "execution_passed": execution_passed,
        "release_sized": release_sized,
        "cells": cells,
        "scientific_passed": release_sized and all_cells_passed,
    }


def run_replicate(job: ReplicateJob) -> ReplicateResult:
    """Run and unconditionally classify one public mixed-model attempt.

    Args:
        job: Target cell, denominator position, and deterministic random seed.

    Returns:
        Complete public inference or one explicit nonreportable outcome record.

    Raises:
        RuntimeError: If no immutable worker state has been installed.
        ValueError: If participant-free trait generation receives an invalid job.
    """
    if not WORKER_STATE:
        raise RuntimeError("MIXED_TARGET_WORKER_STATE_MISSING")
    state: WorkerState = WORKER_STATE[0]
    """Read the sole immutable process-local target state."""

    first, second, cases, measured_hearing = simulate_traits(state, job)
    """Generated both participant-free observed-trait representations."""

    people: int = state.relationship.shape[0]
    """Read the unconditional scientific denominator's target row count."""

    if cases < state.minimum_cases or cases > people - state.minimum_cases:
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="invalid_binary_count",
            cases=cases,
            measured_hearing=measured_hearing,
            covered=False,
            error_code="MIXED_TARGET_SIMULATED_BINARY_COUNT_INVALID",
        )
    if measured_hearing < state.minimum_measured_hearing:
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="invalid_censoring_count",
            cases=cases,
            measured_hearing=measured_hearing,
            covered=False,
            error_code="MIXED_TARGET_SIMULATED_CENSORING_COUNT_INVALID",
        )

    try:
        fit_value: object = asterism.mixed_bivariate_fit(
            state.relationship,
            first,
            second,
            state.design,
        )
        """Fitted the exact binary/right-censored pair through the public API."""
    except ValueError as refusal:
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="refused",
            cases=cases,
            measured_hearing=measured_hearing,
            covered=False,
            error_code=str(refusal),
        )

    if not isinstance(fit_value, dict):
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="refused",
            cases=cases,
            measured_hearing=measured_hearing,
            covered=False,
            error_code="MIXED_TARGET_PUBLIC_FIT_RECORD_MISSING",
        )
    fit: dict[str, object] = fit_value
    """Narrowed the returned public fit to its required record shape."""

    estimate: float | None = finite_float(fit.get("genetic_correlation"))
    """Validated the reportable genetic-correlation point estimate."""

    fit_loglik: float | None = finite_float(fit.get("loglik"))
    """Validated the free likelihood needed to bind subsequent inference."""

    largest_family_value: object = fit.get("largest_family")
    """Selected the public approximation-size diagnostic."""

    largest_family: int | None = (
        largest_family_value
        if isinstance(largest_family_value, int)
        and not isinstance(largest_family_value, bool)
        else None
    )
    """Narrowed the reported family size to a genuine integer."""

    fit_identity_valid: bool = (
        fit.get("estimator") == "ml"
        and fit.get("kinds") == ["binary", "censored"]
        and largest_family == max(state.component_sizes)
    )
    """Required the returned fit to name the intended model and approximation size."""

    if (
        fit.get("converged") is not True
        or estimate is None
        or not -1.0 <= estimate <= 1.0
        or fit_loglik is None
        or not fit_identity_valid
    ):
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="nonconverged",
            cases=cases,
            measured_hearing=measured_hearing,
            converged=False,
            estimate=estimate,
            covered=False,
            error_code="MIXED_TARGET_FIT_NOT_REPORTABLE",
        )

    try:
        interval_value: object = asterism.mixed_bivariate_interval(
            state.relationship,
            first,
            second,
            state.design,
            coordinate="genetic_correlation",
        )
        """Profiled the reportable genetic correlation through the public API."""
    except ValueError as refusal:
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="interval_failed",
            cases=cases,
            measured_hearing=measured_hearing,
            converged=True,
            estimate=estimate,
            covered=False,
            error_code=str(refusal),
        )

    if not isinstance(interval_value, dict):
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="interval_failed",
            cases=cases,
            measured_hearing=measured_hearing,
            converged=True,
            estimate=estimate,
            covered=False,
            error_code="MIXED_TARGET_PUBLIC_INTERVAL_RECORD_MISSING",
        )
    interval: dict[str, object] = interval_value
    """Narrowed the public profile to its required record shape."""

    interval_estimate: float | None = finite_float(interval.get("estimate"))
    """Validated the public profile point estimate."""

    lower: float | None = finite_float(interval.get("lower"))
    """Validated the public lower profile endpoint."""

    upper: float | None = finite_float(interval.get("upper"))
    """Validated the public upper profile endpoint."""

    level: float | None = finite_float(interval.get("level"))
    """Validated the public profile confidence level."""

    profile_failures_value: object = interval.get("profile_failures")
    """Selected the public failed constrained-evaluation count."""

    profile_failures: int | None = (
        profile_failures_value
        if isinstance(profile_failures_value, int)
        and not isinstance(profile_failures_value, bool)
        else None
    )
    """Narrowed the failure count to a genuine non-Boolean integer."""

    if (
        interval_estimate is None
        or lower is None
        or upper is None
        or not -1.0 <= lower <= interval_estimate <= upper <= 1.0
        or not np.isclose(
            interval_estimate,
            estimate,
            rtol=0.0,
            atol=1e-6,
        )
        or level is None
        or not np.isclose(
            level,
            state.required_interval_level,
            rtol=0.0,
            atol=1e-12,
        )
        or profile_failures is None
        or profile_failures < 0
        or not isinstance(interval.get("lower_limited"), bool)
        or not isinstance(interval.get("upper_limited"), bool)
        or interval.get("what") != "genetic_correlation"
        or interval.get("estimator") != "ml"
    ):
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="interval_failed",
            cases=cases,
            measured_hearing=measured_hearing,
            converged=True,
            estimate=estimate,
            lower=lower,
            upper=upper,
            covered=False,
            profile_failures=profile_failures,
            error_code="MIXED_TARGET_PUBLIC_INTERVAL_RECORD_INVALID",
        )
    if profile_failures > 0:
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="profile_failed",
            cases=cases,
            measured_hearing=measured_hearing,
            converged=True,
            estimate=estimate,
            lower=lower,
            upper=upper,
            covered=False,
            profile_failures=profile_failures,
            error_code="MIXED_TARGET_PROFILE_EVALUATION_FAILED",
        )

    try:
        test_value: object = asterism.mixed_bivariate_test(
            state.relationship,
            first,
            second,
            state.design,
            coordinate="genetic_correlation",
        )
        """Tested the interior zero-correlation null through the public API."""
    except ValueError as refusal:
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="test_failed",
            cases=cases,
            measured_hearing=measured_hearing,
            converged=True,
            estimate=estimate,
            lower=lower,
            upper=upper,
            covered=False,
            profile_failures=0,
            error_code=str(refusal),
        )

    if not isinstance(test_value, dict):
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="test_failed",
            cases=cases,
            measured_hearing=measured_hearing,
            converged=True,
            estimate=estimate,
            lower=lower,
            upper=upper,
            covered=False,
            profile_failures=0,
            error_code="MIXED_TARGET_PUBLIC_TEST_RECORD_MISSING",
        )
    test: dict[str, object] = test_value
    """Narrowed the public test to its required record shape."""

    statistic: float | None = finite_float(test.get("statistic"))
    """Validated the public likelihood-ratio statistic."""

    p_value: float | None = finite_float(test.get("p_value"))
    """Validated the public interior-null p-value."""

    rule_value: object = test.get("rule")
    """Selected the public null-reference label."""

    test_rule: str | None = rule_value if isinstance(rule_value, str) else None
    """Narrowed the rule to a stable string when present."""

    null_loglik: float | None = finite_float(test.get("null_loglik"))
    """Validated the public held-null log likelihood."""

    alternative_loglik: float | None = finite_float(test.get("alternative_loglik"))
    """Validated the public free-model log likelihood."""

    likelihood_statistic: float | None = (
        max(0.0, 2.0 * (alternative_loglik - null_loglik))
        if null_loglik is not None and alternative_loglik is not None
        else None
    )
    """Recomputed the likelihood-ratio statistic from its retained likelihoods."""

    reference_p_value: float | None = (
        (1.0 if statistic < SETTLED_DEVIANCE else float(chi2.sf(statistic, 1)))
        if statistic is not None
        else None
    )
    """Recomputed the documented settled one-degree chi-square tail independently."""

    if (
        statistic is None
        or statistic < 0.0
        or p_value is None
        or not 0.0 <= p_value <= 1.0
        or test_rule != state.required_test_rule
        or test.get("what") != "genetic_correlation"
        or test.get("estimator") != "ml"
        or null_loglik is None
        or alternative_loglik is None
        or alternative_loglik < null_loglik - 1e-9
        or not np.isclose(
            alternative_loglik,
            fit_loglik,
            rtol=0.0,
            atol=1e-6,
        )
        or likelihood_statistic is None
        or not np.isclose(
            statistic,
            likelihood_statistic,
            rtol=1e-9,
            atol=1e-9,
        )
        or reference_p_value is None
        or not np.isclose(
            p_value,
            reference_p_value,
            rtol=1e-9,
            atol=1e-12,
        )
    ):
        return ReplicateResult(
            genetic_correlation=job.genetic_correlation,
            censoring_share=job.censoring_share,
            replicate=job.replicate,
            seed=job.seed,
            outcome="test_failed",
            cases=cases,
            measured_hearing=measured_hearing,
            converged=True,
            estimate=estimate,
            lower=lower,
            upper=upper,
            covered=False,
            profile_failures=0,
            p_value=p_value,
            statistic=statistic,
            test_rule=test_rule,
            null_loglik=null_loglik,
            alternative_loglik=alternative_loglik,
            error_code="MIXED_TARGET_PUBLIC_TEST_RECORD_INVALID",
        )

    return ReplicateResult(
        genetic_correlation=job.genetic_correlation,
        censoring_share=job.censoring_share,
        replicate=job.replicate,
        seed=job.seed,
        outcome="complete",
        cases=cases,
        measured_hearing=measured_hearing,
        converged=True,
        estimate=estimate,
        lower=lower,
        upper=upper,
        covered=lower <= job.genetic_correlation <= upper,
        profile_failures=0,
        p_value=p_value,
        statistic=statistic,
        test_rule=test_rule,
        null_loglik=null_loglik,
        alternative_loglik=alternative_loglik,
    )


def load_target_design(path: Path, *, expected_sha256: str) -> TargetDesign:
    """Validate reviewed fixtures and generate the structure-matched target.

    Args:
        path: Reviewed mixed-model values-free fixture.
        expected_sha256: Exact digest pinned by the invoking command.

    Returns:
        Deterministic synthetic relationship, design, and scientific contract.

    Raises:
        ValueError: If fixture identity, structural generation, or acceptance
            differs from the reviewed contract.
    """
    selected_sha256: str = fixture_sha256(path)
    """Bound all following interpretation to exact fixture bytes."""

    if selected_sha256 != expected_sha256:
        raise ValueError("MIXED_TARGET_FIXTURE_SHA256_MISMATCH")

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    """Parsed the exact-byte-verified values-free fixture."""

    if not isinstance(loaded, dict):
        raise ValueError("MIXED_TARGET_FIXTURE_ROOT_INVALID")
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
        "mixed_binary_censored_genetic_correlation_target_design",
        True,
        True,
        "reviewed predecessor aggregate receipt",
    ):
        raise ValueError("MIXED_TARGET_FIXTURE_IDENTITY_INVALID")

    structure_value: object = fixture.get("target_structure")
    """Selected only the non-identifying structural contract."""

    if not isinstance(structure_value, dict):
        raise ValueError("MIXED_TARGET_STRUCTURE_INVALID")
    structure: dict[str, object] = structure_value
    """Narrowed structural aggregates to their required object shape."""

    expected_structure: dict[str, object] = {
        "source_fixture": "one_trait_gaussian_heritability.json",
        "source_fixture_sha256": (
            "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e"
        ),
        "matching_basis": "reviewed_non_identifying_aggregate_structure",
        "reconstruction_claim": False,
        "analysis_n": 1_909,
        "component_count": 202,
        "largest_component": 180,
        "observed_nonzero_relationship_pairs": 27_821,
        "generated_nonzero_relationship_pairs": 27_691,
        "maximum_relative_nonzero_pair_discrepancy": 0.005,
        "matched_facts": [
            "analysis_n",
            "component_count",
            "component_size_histogram",
            "largest_component",
            "nonzero_relationship_pair_count",
        ],
        "not_retained": [
            "participant_identifiers",
            "participant_rows",
            "phenotypes",
            "participant_covariates",
            "participant_relationship_matrix",
        ],
        "not_claimed": [
            "pedigree_reconstruction",
            "relationship_coefficient_identity",
            "relationship_eigenvalue_identity",
            "participant_covariate_moment_matching",
        ],
    }
    """Restated every reviewed structural fact independently of fixture edits."""

    if structure != expected_structure:
        raise ValueError("MIXED_TARGET_STRUCTURE_INVALID")

    simulation_value: object = fixture.get("simulation")
    """Selected the fully frozen participant-free outcome generator."""

    expected_simulation: dict[str, object] = {
        "trait_types": ["binary", "right_censored_continuous"],
        "prevalence": 0.254,
        "heritabilities": [0.5, 0.5],
        "genetic_correlations": [0.0, 0.4],
        "residual_correlation": 0.15,
        "hearing_variance": 3.0,
        "censoring_shares": [0.52, 0.75],
        "base_seed": 1_210_000,
        "cell_seed_offset": 104_729,
        "replicate_seed_offset": 7_919,
        "minimum_cases": 10,
        "minimum_measured_hearing": 10,
        "outcome_source": "deterministic_simulated_latent_traits_only",
        "fixed_effect_source": (
            "participant_free_synthetic_design_from_source_fixture"
        ),
    }
    """Restated every latent-trait generating coordinate."""

    if simulation_value != expected_simulation:
        raise ValueError("MIXED_TARGET_SIMULATION_INVALID")

    acceptance_value: object = fixture.get("monte_carlo_acceptance")
    """Selected the acceptance rules written before campaign execution."""

    expected_acceptance: dict[str, object] = {
        "release_replicates_per_cell": 300,
        "minimum_smoke_replicates_per_cell": 1,
        "nominal_coverage": 0.95,
        "nominal_level": 0.05,
        "confidence": 0.95,
        "point_recovery_standard_errors": 3.0,
        "level_fail_when": ("one_sided_clopper_pearson_lower_above_nominal_level"),
        "coverage_fail_when": (
            "one_sided_clopper_pearson_upper_below_nominal_coverage"
        ),
        "denominator": "every_attempted_replicate",
        "failure_outcomes": [
            "invalid_binary_count",
            "invalid_censoring_count",
            "refused",
            "nonconverged",
            "test_failed",
            "interval_failed",
            "profile_failed",
        ],
        "maximum_failed_replicates": 0,
        "required_test_rule": "chi2_1",
        "required_interval_level": 0.95,
    }
    """Restated every inferential and numerical-failure decision."""

    if acceptance_value != expected_acceptance:
        raise ValueError("MIXED_TARGET_ACCEPTANCE_INVALID")

    source_path: Path = path.parent / str(structure["source_fixture"])
    """Resolved the sole reviewed structural source beside the wrapper fixture."""

    source_sha256: str = fixture_sha256(source_path)
    """Bound generation to the exact reviewed source-fixture bytes."""

    if source_sha256 != structure["source_fixture_sha256"]:
        raise ValueError("MIXED_TARGET_SOURCE_FIXTURE_SHA256_MISMATCH")

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
        raise ValueError("MIXED_TARGET_GENERATED_AGGREGATES_INVALID")
    if envelope.relative_pair_count_discrepancy is None or not np.isclose(
        envelope.relative_pair_count_discrepancy,
        130 / 27_821,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("MIXED_TARGET_PAIR_DISCREPANCY_INVALID")
    if np.linalg.matrix_rank(envelope.fixed_effects) != 6:
        raise ValueError("MIXED_TARGET_DESIGN_RANK_INVALID")

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
        heritabilities=(0.5, 0.5),
        genetic_correlations=(0.0, 0.4),
        residual_correlation=0.15,
        hearing_variance=3.0,
        censoring_shares=(0.52, 0.75),
        base_seed=1_210_000,
        cell_seed_offset=104_729,
        replicate_seed_offset=7_919,
        minimum_cases=10,
        minimum_measured_hearing=10,
        release_replicates=300,
        minimum_smoke_replicates=1,
        nominal_coverage=0.95,
        nominal_level=0.05,
        monte_carlo_confidence=0.95,
        point_recovery_standard_errors=3.0,
        maximum_failed_replicates=0,
        required_test_rule="chi2_1",
        required_interval_level=0.95,
    )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Read a fully explicit participant-free target campaign command.

    Args:
        arguments: Optional command arguments excluding the executable name.

    Returns:
        Validated explicit scientific and operational coordinates.

    Raises:
        SystemExit: If a required coordinate is missing or malformed.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Initialised the parser before adding explicit scientific coordinates."""

    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--prevalence", required=True, type=float)
    parser.add_argument("--heritabilities", required=True, type=float, nargs=2)
    parser.add_argument(
        "--genetic-correlations",
        required=True,
        type=float,
        nargs=2,
    )
    parser.add_argument("--residual-correlation", required=True, type=float)
    parser.add_argument("--hearing-variance", required=True, type=float)
    parser.add_argument("--censoring-shares", required=True, type=float, nargs=2)
    parser.add_argument("--nominal-coverage", required=True, type=float)
    parser.add_argument("--nominal-level", required=True, type=float)
    parser.add_argument("--monte-carlo-confidence", required=True, type=float)
    parser.add_argument(
        "--point-recovery-standard-errors",
        required=True,
        type=float,
    )
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--cell-seed-offset", required=True, type=int)
    parser.add_argument("--replicate-seed-offset", required=True, type=int)
    parser.add_argument("--minimum-cases", required=True, type=int)
    parser.add_argument("--minimum-measured-hearing", required=True, type=int)
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
    if any(not 0.0 <= value <= 1.0 for value in parsed.heritabilities):
        parser.error("--heritabilities must lie on the closed unit interval")
    if len(set(parsed.genetic_correlations)) != 2 or any(
        not -1.0 <= value <= 1.0 for value in parsed.genetic_correlations
    ):
        parser.error("--genetic-correlations must be two unique valid correlations")
    if not -1.0 <= parsed.residual_correlation <= 1.0:
        parser.error("--residual-correlation must lie on the closed unit interval")
    if not np.isfinite(parsed.hearing_variance) or parsed.hearing_variance <= 0.0:
        parser.error("--hearing-variance must be positive and finite")
    if len(set(parsed.censoring_shares)) != 2 or any(
        not 0.0 < value < 1.0 for value in parsed.censoring_shares
    ):
        parser.error("--censoring-shares must be two unique open-unit values")
    if not 0.0 < parsed.nominal_coverage < 1.0:
        parser.error("--nominal-coverage must lie on the open unit interval")
    if not 0.0 < parsed.nominal_level < 1.0:
        parser.error("--nominal-level must lie on the open unit interval")
    if not 0.0 < parsed.monte_carlo_confidence < 1.0:
        parser.error("--monte-carlo-confidence must lie on the open unit interval")
    if parsed.point_recovery_standard_errors <= 0.0:
        parser.error("--point-recovery-standard-errors must be positive")
    if (
        parsed.base_seed < 0
        or parsed.cell_seed_offset < 1
        or parsed.replicate_seed_offset < 1
    ):
        parser.error("outcome seeds must select nonnegative disjoint streams")
    if parsed.minimum_cases < 1 or parsed.minimum_measured_hearing < 1:
        parser.error("minimum observation counts must be positive")
    if len(parsed.fixture_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in parsed.fixture_sha256
    ):
        parser.error("--fixture-sha256 must be a lowercase SHA-256")
    """Rejected malformed coordinates before reading fixture content."""
    return parsed


def campaign_configuration(
    arguments: argparse.Namespace,
    target: TargetDesign,
) -> CampaignConfiguration:
    """Require every command coordinate to agree with the reviewed fixture.

    Args:
        arguments: Validated explicit command-line coordinates.
        target: Exact-byte reviewed target and acceptance contract.

    Returns:
        Immutable campaign coordinates safe to schedule in any worker order.

    Raises:
        ValueError: If the replicate count or any scientific coordinate differs
            from the reviewed target contract.
    """
    if not (
        target.minimum_smoke_replicates
        <= arguments.replicates
        <= target.release_replicates
    ):
        raise ValueError("MIXED_TARGET_REPLICATE_COUNT_OUTSIDE_REVIEWED_RANGE")

    command_identity: tuple[object, ...] = (
        arguments.prevalence,
        tuple(arguments.heritabilities),
        tuple(arguments.genetic_correlations),
        arguments.residual_correlation,
        arguments.hearing_variance,
        tuple(arguments.censoring_shares),
        arguments.nominal_coverage,
        arguments.nominal_level,
        arguments.monte_carlo_confidence,
        arguments.point_recovery_standard_errors,
        arguments.base_seed,
        arguments.cell_seed_offset,
        arguments.replicate_seed_offset,
        arguments.minimum_cases,
        arguments.minimum_measured_hearing,
        arguments.fixture_sha256,
    )
    """Collected every fixture-owned command coordinate."""

    target_identity: tuple[object, ...] = (
        target.prevalence,
        target.heritabilities,
        target.genetic_correlations,
        target.residual_correlation,
        target.hearing_variance,
        target.censoring_shares,
        target.nominal_coverage,
        target.nominal_level,
        target.monte_carlo_confidence,
        target.point_recovery_standard_errors,
        target.base_seed,
        target.cell_seed_offset,
        target.replicate_seed_offset,
        target.minimum_cases,
        target.minimum_measured_hearing,
        target.fixture_sha256,
    )
    """Collected corresponding independently reviewed coordinates."""

    if command_identity != target_identity:
        raise ValueError("MIXED_TARGET_COMMAND_DIFFERS_FROM_REVIEWED_FIXTURE")

    return CampaignConfiguration(
        replicates=arguments.replicates,
        workers=arguments.workers,
        prevalence=arguments.prevalence,
        heritabilities=tuple(arguments.heritabilities),
        genetic_correlations=tuple(arguments.genetic_correlations),
        residual_correlation=arguments.residual_correlation,
        hearing_variance=arguments.hearing_variance,
        censoring_shares=tuple(arguments.censoring_shares),
        nominal_coverage=arguments.nominal_coverage,
        nominal_level=arguments.nominal_level,
        monte_carlo_confidence=arguments.monte_carlo_confidence,
        point_recovery_standard_errors=arguments.point_recovery_standard_errors,
        base_seed=arguments.base_seed,
        cell_seed_offset=arguments.cell_seed_offset,
        replicate_seed_offset=arguments.replicate_seed_offset,
        minimum_cases=arguments.minimum_cases,
        minimum_measured_hearing=arguments.minimum_measured_hearing,
        maximum_failed_replicates=target.maximum_failed_replicates,
    )


def genetic_factors(
    target: TargetDesign,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Factor the additive relationship separately inside each family."""
    factors: list[npt.NDArray[np.float64]] = []
    """Collected reusable factors without a dense global factorisation."""

    offset: int = 0
    """Tracked component placement in the target matrix."""

    for size in target.component_sizes:
        stop: int = offset + size
        """Located the current square relationship block."""

        relationship: npt.NDArray[np.float64] = target.relationship[
            offset:stop,
            offset:stop,
        ]
        """Selected one independent synthetic pedigree component."""

        factors.append(np.linalg.cholesky(relationship))
        offset = stop
        """Factored the component and advanced to the next block."""
    return tuple(factors)


def run_campaign(
    target: TargetDesign,
    configuration: CampaignConfiguration,
) -> list[ReplicateResult]:
    """Run every reviewed target cell through documented public functions.

    Args:
        target: Generated participant-free arrays and reviewed structure facts.
        configuration: Exact campaign grid, seeds, and worker count.

    Returns:
        One explicit result for every scheduled replicate in stable grid order.

    Raises:
        ValueError: If target factorisation or a replicate job is invalid.
        RuntimeError: If process-worker initialisation fails.
    """
    state: WorkerState = WorkerState(
        relationship=target.relationship,
        design=target.design,
        component_sizes=target.component_sizes,
        genetic_factors=genetic_factors(target),
        prevalence=configuration.prevalence,
        heritabilities=configuration.heritabilities,
        residual_correlation=configuration.residual_correlation,
        hearing_variance=configuration.hearing_variance,
        minimum_cases=configuration.minimum_cases,
        minimum_measured_hearing=configuration.minimum_measured_hearing,
        nominal_coverage=configuration.nominal_coverage,
        required_test_rule=target.required_test_rule,
        required_interval_level=target.required_interval_level,
    )
    """Built one immutable worker state from exact-byte reviewed inputs."""

    jobs: list[ReplicateJob] = []
    """Collected every deterministic target-grid attempt before scheduling."""

    for correlation_index, correlation in enumerate(configuration.genetic_correlations):
        for censoring_index, censoring_share in enumerate(
            configuration.censoring_shares
        ):
            cell: int = (
                correlation_index * len(configuration.censoring_shares)
                + censoring_index
            )
            """Assigned one disjoint stream coordinate to the scientific cell."""

            for replicate in range(configuration.replicates):
                jobs.append(
                    ReplicateJob(
                        genetic_correlation=correlation,
                        censoring_share=censoring_share,
                        replicate=replicate,
                        seed=(
                            configuration.base_seed
                            + cell * configuration.cell_seed_offset
                            + replicate * configuration.replicate_seed_offset
                        ),
                    )
                )
                """Retained one schedule-independent unconditional attempt."""
    """Enumerated the complete four-cell campaign in stable row order."""

    if configuration.workers == 1:
        start_worker(state)
        """Installed the immutable state in the current process for bounded smoke."""

        results: list[ReplicateResult] = [run_replicate(job) for job in jobs]
        """Ran serially without changing worker behaviour or scientific ordering."""
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
    """Run the target campaign and emit one values-free JSON evidence record.

    Args:
        arguments: Optional command arguments excluding the executable name.

    Returns:
        Zero for an acceptable smoke or qualifying release campaign, otherwise one.

    Raises:
        ValueError: If fixture identity or command identity is not reviewed.
        RuntimeError: If campaign worker initialisation fails.
    """
    parsed: argparse.Namespace = parse_arguments(arguments)
    """Read every scientific coordinate from the explicit command."""

    target: TargetDesign = load_target_design(
        parsed.design_fixture,
        expected_sha256=parsed.fixture_sha256,
    )
    """Verified exact fixture bytes before generating any outcome."""

    configuration: CampaignConfiguration = campaign_configuration(parsed, target)
    """Required every command coordinate to match the reviewed fixture."""

    started: float = time.perf_counter()
    """Started bounded timing after deterministic target construction."""

    results: list[ReplicateResult] = run_campaign(target, configuration)
    """Retained one explicit record for every requested target attempt."""

    score: dict[str, object] = score_campaign(results, target)
    """Applied scientific rules over unconditional cell denominators."""

    release_sized: bool = score["release_sized"] is True
    """Distinguished exact calibration evidence from a bounded execution smoke."""

    passed: bool = (
        score["scientific_passed"] is True
        if release_sized
        else score["execution_passed"] is True
    )
    """Required full science for release-sized runs and execution only for smoke."""

    evidence: dict[str, object] = {
        "check": "mixed_binary_censored_target_design",
        "analysis_id": "mixed_binary_censored_genetic_correlation",
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
            "asterism.mixed_bivariate_fit",
            "asterism.mixed_bivariate_interval",
            "asterism.mixed_bivariate_test",
        ],
        "configuration": asdict(configuration),
        "elapsed_seconds": time.perf_counter() - started,
        "release_sized": release_sized,
        "qualifying_release_evidence": score["scientific_passed"] is True,
        "results": score,
        "passed": passed,
        "failures": []
        if passed
        else [
            "release-sized scientific decisions failed"
            if release_sized
            else "one or more bounded public-route attempts failed"
        ],
    }
    """Built evidence sufficient to recompute execution and scientific status."""

    print(json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
