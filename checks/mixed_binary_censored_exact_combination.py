"""Calibrate the exact binary-diagnosis and right-censored-hearing pairing.

The older mixed-model checks exercise binary/continuous and
censored/continuous pairs separately.  Neither evaluates the observation
mechanisms together, so neither can satisfy the 0.1 claim.  This participant-
free simulation fits, profiles, and tests the exact binary/right-censored pair
at both intended high-frequency censoring shares.

Every replicate is scored.  A refused or nonconverged fit cannot cover, and a
profile with any failed constrained evaluation cannot cover.  A coverage cell
passes only when its two-sided Clopper--Pearson interval includes 0.95.  The
zero-correlation cell additionally requires its rejection-rate interval to
include the nominal 0.05.  Point recovery is required to lie within three Monte
Carlo standard errors of the generating correlation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from statistics import NormalDist
from typing import Any

import asterism
import numpy as np
from scipy.stats import beta

PAIRS: int = 200
"""Independent sibling pairs in each participant-free replicate."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "100"))
"""Replicates per genetic-correlation and censoring-share cell."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "8"))
"""Bounded worker count for the release command."""

HERITABILITY: tuple[float, float] = (0.5, 0.5)
"""Generating liability and complete-hearing heritabilities."""

GENETIC_CORRELATIONS: tuple[float, float] = (0.0, 0.4)
"""Null and interior alternative correlations scored by the check."""

RESIDUAL_CORRELATION: float = 0.15
"""Generating residual correlation between liability and complete hearing."""

HEARING_VARIANCE: float = 3.0
"""Latent complete-hearing variance before censoring."""

PREVALENCE: float = 0.30
"""Generating diagnosis prevalence."""

CENSORING_SHARES: tuple[float, float] = (0.52, 0.75)
"""Observed high-frequency audiogram censoring shares required by 0.1."""

NOMINAL_COVERAGE: float = 0.95
"""Profile-interval coverage target."""

NOMINAL_LEVEL: float = 0.05
"""Interior zero-correlation test level."""

STANDARD_ERRORS_ALLOWED: float = 3.0
"""Pre-written Monte Carlo point-recovery acceptance band."""


def correlated(
    correlation: float, first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return two standard-normal arrays with the requested correlation."""
    return first, correlation * first + np.sqrt(1.0 - correlation**2) * second


def draw(
    correlation: float, censoring_share: float, replicate: int
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], np.ndarray]:
    """Draw and encode one binary/right-censored sibling-pair dataset."""
    cell: int = GENETIC_CORRELATIONS.index(correlation) * len(
        CENSORING_SHARES
    ) + CENSORING_SHARES.index(censoring_share)
    """Selected a disjoint deterministic stream for every scientific cell."""

    rng: np.random.Generator = np.random.default_rng(
        1_210_000 + 104_729 * cell + 7_919 * replicate
    )
    """Created the fixed participant-free random stream."""

    people: int = 2 * PAIRS
    """Counted the rows in the sibling-pair roster."""

    relationship: np.ndarray = np.eye(people)
    """Started from unit marginal additive covariance."""

    for pair in range(PAIRS):
        first: int = 2 * pair
        """Selected the first sibling's row."""

        second: int = first + 1
        """Selected the second sibling's row."""

        relationship[first, second] = 0.5
        """Recorded the first directed sibling relationship entry."""

        relationship[second, first] = 0.5
        """Completed the symmetric sibling relationship entry."""

    shared_one, shared_two = correlated(
        correlation,
        rng.standard_normal(PAIRS),
        rng.standard_normal(PAIRS),
    )
    """Drew cross-trait correlated family-level genetic effects."""

    own_one, own_two = correlated(
        correlation,
        rng.standard_normal(people),
        rng.standard_normal(people),
    )
    """Drew the remaining Mendelian genetic effects."""

    residual_one, residual_two = correlated(
        RESIDUAL_CORRELATION,
        rng.standard_normal(people),
        rng.standard_normal(people),
    )
    """Drew cross-trait correlated individual residuals."""

    genetic_one: np.ndarray = (
        np.sqrt(0.5) * np.repeat(shared_one, 2) + np.sqrt(0.5) * own_one
    )
    """Combined family and individual genetic effects for diagnosis liability."""

    genetic_two: np.ndarray = (
        np.sqrt(0.5) * np.repeat(shared_two, 2) + np.sqrt(0.5) * own_two
    )
    """Combined family and individual genetic effects for complete hearing."""

    liability: np.ndarray = (
        np.sqrt(HERITABILITY[0]) * genetic_one
        + np.sqrt(1.0 - HERITABILITY[0]) * residual_one
    )
    """Generated the unit-scale latent diagnosis liability."""

    hearing: np.ndarray = np.sqrt(HEARING_VARIANCE) * (
        np.sqrt(HERITABILITY[1]) * genetic_two
        + np.sqrt(1.0 - HERITABILITY[1]) * residual_two
    )
    """Generated the complete hearing threshold before instrument censoring."""

    diagnosis_cut: float = NormalDist().inv_cdf(1.0 - PREVALENCE)
    """Fixed the liability threshold independently of the sampled outcomes."""

    hearing_cut: float = np.sqrt(HEARING_VARIANCE) * NormalDist().inv_cdf(
        1.0 - censoring_share
    )
    """Fixed the audiometer limit at the prespecified censoring quantile."""

    hearing_censored: np.ndarray = hearing >= hearing_cut
    """Applied right censoring without replacing the latent values by the limit."""

    diagnosis: dict[str, Any] = {
        "kind": "binary",
        "value": np.full(people, np.nan),
        "censoring": np.where(liability > diagnosis_cut, 1, 2).astype(np.int64),
        "limit": np.zeros(people),
    }
    """Encoded case and noncase regions on the unit liability scale."""

    censored_hearing: dict[str, Any] = {
        "kind": "censored",
        "value": np.where(hearing_censored, np.nan, hearing),
        "censoring": np.where(hearing_censored, 1, 0).astype(np.int64),
        "limit": np.full(people, hearing_cut),
    }
    """Encoded measured values and upper half-lines for hearing."""

    design: np.ndarray = np.ones((people, 1))
    """Used one intercept per trait through the public mixed-model design."""

    return relationship, diagnosis, censored_hearing, design


def one(job: tuple[float, float, int]) -> dict[str, Any]:
    """Fit, profile, and test one exact-combination replicate."""
    correlation, censoring_share, replicate = job
    """Named the fixed cell and deterministic replicate coordinate."""

    relationship, diagnosis, hearing, design = draw(
        correlation, censoring_share, replicate
    )
    """Generated the exact binary and censored participant-free pairing."""

    record: dict[str, Any] = {
        "correlation": correlation,
        "censoring_share": censoring_share,
        "replicate": replicate,
    }
    """Identified the scientific cell before any estimator can refuse it."""

    try:
        fit: dict[str, Any] = asterism.mixed_bivariate_fit(
            relationship, diagnosis, hearing, design
        )
        """Fitted the exact binary/right-censored model through the public API."""

        interval: dict[str, Any] = asterism.mixed_bivariate_interval(
            relationship,
            diagnosis,
            hearing,
            design,
            coordinate="genetic_correlation",
        )
        """Profiled the genetic correlation."""

        test: dict[str, Any] = asterism.mixed_bivariate_test(
            relationship,
            diagnosis,
            hearing,
            design,
            coordinate="genetic_correlation",
        )
        """Tested the interior zero-correlation null."""
    except ValueError as refusal:
        record["refusal"] = str(refusal)
        """Counted a public model refusal as an explicit scientific miss."""

        return record

    record.update(
        {
            "fit_converged": fit["converged"],
            "estimate": fit["genetic_correlation"],
            "profile_failures": interval["profile_failures"],
            "covered": bool(
                fit["converged"]
                and interval["profile_failures"] == 0
                and interval["lower"] <= correlation <= interval["upper"]
            ),
            "p_value": test["p_value"],
            "test_rule": test["rule"],
        }
    )
    """Retained only fields used by the pre-written acceptance decisions."""

    return record


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    """Return the exact two-sided 95% interval for a binomial proportion."""
    lower: float = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, trials - successes + 1))
    )
    """Computed the exact lower confidence limit."""

    upper: float = (
        1.0
        if successes == trials
        else float(beta.ppf(0.975, successes + 1, trials - successes))
    )
    """Computed the exact upper confidence limit."""

    return lower, upper


def parse_arguments() -> argparse.Namespace:
    """Parse the fixed replicate and worker counts for release evidence."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built an explicit argv contract rather than relying on ambient variables."""

    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments: argparse.Namespace = parser.parse_args()
    """Read the configured counts before any process workers start."""

    if arguments.replicates < 1 or arguments.workers < 1:
        parser.error("--replicates and --workers must be positive")
    return arguments


def main() -> int:
    """Run every exact-combination cell and emit one auditable decision record."""
    arguments: argparse.Namespace = parse_arguments()
    """Bound this execution to counts recorded directly in its argument vector."""

    jobs: list[tuple[float, float, int]] = [
        (correlation, censoring_share, replicate)
        for correlation in GENETIC_CORRELATIONS
        for censoring_share in CENSORING_SHARES
        for replicate in range(arguments.replicates)
    ]
    """Enumerated disjoint cells before starting parallel work."""

    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        rows: list[dict[str, Any]] = list(pool.map(one, jobs, chunksize=1))
        """Ran every replicate without dropping refused outcomes."""

    failures: list[str] = []
    """Collected all failed pre-written acceptance rules."""

    cells: dict[str, Any] = {}
    """Stored complete aggregate evidence for each exact scientific cell."""

    for correlation in GENETIC_CORRELATIONS:
        for censoring_share in CENSORING_SHARES:
            selected: list[dict[str, Any]] = [
                row
                for row in rows
                if row["correlation"] == correlation
                and row["censoring_share"] == censoring_share
            ]
            """Selected every attempted replicate, including refusals."""

            estimates: list[float] = [
                float(row["estimate"])
                for row in selected
                if row.get("fit_converged") is True
            ]
            """Retained converged free-fit estimates for point-recovery scoring."""

            covered: int = sum(bool(row.get("covered")) for row in selected)
            """Counted coverage with refused and failed profiles as misses."""

            coverage_interval: tuple[float, float] = clopper_pearson(
                covered, len(selected)
            )
            """Quantified Monte Carlo uncertainty around interval coverage."""

            coverage_passed: bool = (
                coverage_interval[0] <= NOMINAL_COVERAGE <= coverage_interval[1]
            )
            """Applied the fixed two-sided nominal-coverage acceptance rule."""

            estimate_mean: float = (
                float(np.mean(estimates)) if estimates else float("nan")
            )
            """Computed the recovered correlation among converged free fits."""

            estimate_error: float = (
                float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
                if len(estimates) > 1
                else float("nan")
            )
            """Computed its Monte Carlo standard error without inventing one."""

            recovery_distance: float = (
                abs(estimate_mean - correlation) / estimate_error
                if np.isfinite(estimate_error) and estimate_error > 0.0
                else float("inf")
            )
            """Expressed point bias on the prespecified Monte Carlo scale."""

            recovery_passed: bool = recovery_distance <= STANDARD_ERRORS_ALLOWED
            """Applied the fixed three-standard-error recovery rule."""

            p_values: list[float] = [
                float(row["p_value"])
                for row in selected
                if row.get("test_rule") == "chi2_1"
            ]
            """Selected tests that explicitly used the required interior reference."""

            rejected: int = sum(value <= NOMINAL_LEVEL for value in p_values)
            """Counted nominal-level rejections among correctly labelled tests."""

            level_interval: tuple[float, float] | None = (
                clopper_pearson(rejected, len(selected)) if correlation == 0.0 else None
            )
            """Measured test level only in the true zero-correlation cells."""

            level_passed: bool = (
                True
                if level_interval is None
                else level_interval[0] <= NOMINAL_LEVEL <= level_interval[1]
            )
            """Applied the fixed exact-binomial level rule at the null."""

            key: str = f"rho_g={correlation} censored={censoring_share}"
            """Named the exact trait-combination cell in stable scalar form."""

            cells[key] = {
                "attempted": len(selected),
                "refused": sum("refusal" in row for row in selected),
                "converged": len(estimates),
                "estimate_mean": estimate_mean,
                "estimate_standard_error": estimate_error,
                "recovery_distance_in_standard_errors": recovery_distance,
                "recovery_passed": recovery_passed,
                "covered": covered,
                "coverage": covered / len(selected),
                "coverage_clopper_pearson": coverage_interval,
                "coverage_passed": coverage_passed,
                "test_rule_count": len(p_values),
                "rejected": rejected,
                "rejection_rate": rejected / len(selected),
                "level_clopper_pearson": level_interval,
                "level_passed": level_passed,
            }
            """Recorded enough facts to recompute every aggregate pass decision."""

            if not recovery_passed:
                failures.append(
                    f"{key}: genetic-correlation recovery is outside the band"
                )
            if not coverage_passed:
                failures.append(f"{key}: interval coverage excludes 0.95")
            if not level_passed:
                failures.append(f"{key}: zero-correlation test level excludes 0.05")
            if len(p_values) != len(selected):
                failures.append(f"{key}: not every test used the required chi2_1 rule")

    record: dict[str, Any] = {
        "check": "mixed_binary_censored_exact_combination",
        "participant_free": True,
        "people": 2 * PAIRS,
        "largest_family": 2,
        "replicates_per_cell": arguments.replicates,
        "workers": arguments.workers,
        "trait_types": ["binary", "right_censored_continuous"],
        "censoring_shares": list(CENSORING_SHARES),
        "genetic_correlations": list(GENETIC_CORRELATIONS),
        "nominal_coverage": NOMINAL_COVERAGE,
        "nominal_level": NOMINAL_LEVEL,
        "standard_errors_allowed": STANDARD_ERRORS_ALLOWED,
        "cells": cells,
        "passed": not failures,
        "failures": failures,
    }
    """Built the complete exact-combination scientific evidence record."""

    print(json.dumps(record, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
