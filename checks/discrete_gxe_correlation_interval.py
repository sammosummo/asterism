"""Measure discrete-GxE genetic-correlation interval coverage through public APIs.

The release campaign uses 600 participant-free people in 150 independent
four-sibling families and 400 fixed replicates at correlations 0.3, 0.6, and
0.9.  Every free-fit refusal, nonconvergence, interval refusal, and failed
profile evaluation remains in the prespecified denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import asterism
import numpy as np
import numpy.typing as npt
from scipy.stats import beta

TRUTHS: tuple[float, ...] = (0.3, 0.6, 0.9)
"""Pinned the documented interior genetic-correlation truth grid."""

NOMINAL: float = 0.95
"""Coverage probability claimed by the public profile interval."""

MINIMUM_COMPATIBLE: float = 0.94
"""Lower edge below which exact undercoverage evidence fails the campaign."""

MAXIMUM_COMPATIBLE: float = 0.96
"""Upper edge used only to label known conservative behavior."""

GENETIC_SD: tuple[float, float] = (0.707, 0.707)
"""Fixed equal genetic standard deviations used at every correlation truth."""

RESIDUAL_SD: tuple[float, float] = (0.707, 0.707)
"""Fixed equal residual standard deviations used at every correlation truth."""

TRUE_EFFECTS: npt.NDArray[np.float64] = np.asarray([0.5, 0.25])
"""Fixed intercept and environment mean effect for simulated responses."""

BASE_SEED: int = 2_026_081_700
"""Owned deterministic response streams independently of worker scheduling."""

STATE: dict[str, Any] = {}
"""Held immutable participant-free arrays inside campaign worker processes."""


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse every interval-campaign coordinate without scientific defaults.

    Args:
        argv: Optional explicit command arguments for tests and callers.

    Returns:
        Validated fixed design and Monte Carlo coordinates.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built one fail-closed participant-free campaign interface."""

    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--people", type=int, required=True)
    parser.add_argument("--families", type=int, required=True)
    parser.add_argument("--sibs-per-family", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--truths", type=float, nargs="+", required=True)
    parser.add_argument("--no-write", action="store_true", required=True)
    arguments: argparse.Namespace = parser.parse_args(argv)
    """Read all design, simulation, and evidence coordinates explicitly."""

    if any(
        value < 1
        for value in (
            arguments.replicates,
            arguments.people,
            arguments.families,
            arguments.sibs_per_family,
            arguments.workers,
        )
    ):
        parser.error("counts must be positive")
    if arguments.people != arguments.families * arguments.sibs_per_family:
        parser.error("--people must equal --families times --sibs-per-family")
    if arguments.sibs_per_family < 4 or arguments.sibs_per_family % 2 != 0:
        parser.error("--sibs-per-family must be even and at least four")
    if tuple(arguments.truths) != TRUTHS:
        parser.error("--truths must be exactly 0.3 0.6 0.9")
    return arguments


def coverage_decision(
    *,
    covered: int,
    complete: int,
    attempted: int,
) -> dict[str, object]:
    """Score one interval cell with exact uncertainty and a fixed denominator.

    Args:
        covered: Complete intervals containing the generating correlation.
        complete: Attempts with a converged fit and failure-free profile.
        attempted: Prespecified total replicate count.

    Returns:
        Coverage, exact one-sided limits, verdict, and pass decision.

    Raises:
        ValueError: If the supplied denominator facts are inconsistent.
    """
    if (
        attempted < 1
        or complete < 0
        or covered < 0
        or covered > complete
        or complete > attempted
    ):
        raise ValueError("DISCRETE_GXE_INTERVAL_COUNTS_INVALID")

    coverage: float | None = covered / complete if complete else None
    """Calculated coverage only across evaluable public intervals."""

    lower: float | None = (
        None
        if complete == 0
        else (
            0.0
            if covered == 0
            else float(beta.ppf(0.05, covered, complete - covered + 1))
        )
    )
    """Calculated the exact one-sided 95% lower coverage limit."""

    upper: float | None = (
        None
        if complete == 0
        else (
            1.0
            if covered == complete
            else float(beta.ppf(0.95, covered + 1, complete - covered))
        )
    )
    """Calculated the exact one-sided 95% upper coverage limit."""

    if complete != attempted:
        verdict: str = "incomplete"
        """Made any unsuccessful fit or profile a campaign failure."""

        passed: bool = False
        """Prevented unavailable intervals from leaving the denominator."""
    elif upper is not None and upper < MINIMUM_COMPATIBLE:
        verdict = "anti_conservative"
        """Found exact evidence that coverage lies below the safe band."""

        passed = False
        """Failed the prewritten one-sided undercoverage rule."""
    elif lower is not None and lower > MAXIMUM_COMPATIBLE:
        verdict = "conservative"
        """Named the documented near-boundary conservative regime."""

        passed = True
        """Allowed excess coverage while retaining its loss-of-precision label."""
    else:
        verdict = "compatible"
        """Found exact uncertainty compatible with the nominal safe band."""

        passed = True
        """Accepted one complete non-anti-conservative interval cell."""

    return {
        "attempted": attempted,
        "complete": complete,
        "covered": covered,
        "coverage": coverage,
        "one_sided_95_percent_lower": lower,
        "one_sided_95_percent_upper": upper,
        "nominal": NOMINAL,
        "minimum_compatible": MINIMUM_COMPATIBLE,
        "maximum_compatible": MAXIMUM_COMPATIBLE,
        "verdict": verdict,
        "passed": passed,
    }


def sibling_design(
    families: int,
    sibs_per_family: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Build independent sibling families with both environments in each.

    Args:
        families: Number of unrelated family blocks.
        sibs_per_family: Even number of siblings in every block.

    Returns:
        Relationship matrix, two-level environment, and fixed-effect design.
    """
    if families < 1 or sibs_per_family < 4 or sibs_per_family % 2 != 0:
        raise ValueError("DISCRETE_GXE_INTERVAL_DESIGN_INVALID")

    people: int = families * sibs_per_family
    """Counted people in the participant-free fixed family roster."""

    relationship: npt.NDArray[np.float64] = np.zeros((people, people), dtype=float)
    """Allocated independent full-sibling relationship blocks."""

    for family in range(families):
        start: int = family * sibs_per_family
        """Located the first row in this synthetic family."""

        stop: int = start + sibs_per_family
        """Located the exclusive end of this family block."""

        relationship[start:stop, start:stop] = 0.5
        """Assigned ordinary full-sibling off-diagonal relationships."""

        relationship[np.arange(start, stop), np.arange(start, stop)] = 1.0
        """Restored unit additive relationship on every diagonal."""

    first_half: npt.NDArray[np.float64] = np.ones(sibs_per_family // 2)
    """Placed half of each family in the first environment."""

    second_half: npt.NDArray[np.float64] = np.full(sibs_per_family // 2, 2.0)
    """Placed the remaining siblings in the second environment."""

    group: npt.NDArray[np.float64] = np.tile(
        np.concatenate([first_half, second_half]),
        families,
    )
    """Repeated the balanced within-family environment layout."""

    design: npt.NDArray[np.float64] = np.column_stack(
        [np.ones(people), (group == 1.0).astype(float)]
    )
    """Included an intercept and prespecified environment mean effect."""

    return relationship, group, design


def simulating_covariance(
    relationship: npt.NDArray[np.float64],
    group: npt.NDArray[np.float64],
    correlation: float,
) -> npt.NDArray[np.float64]:
    """Build a known-truth discrete-GxE covariance for one interval cell.

    Args:
        relationship: Participant-free additive relationship matrix.
        group: Two-level environment labels.
        correlation: Fixed generating genetic correlation.

    Returns:
        Dense response covariance with equal group-specific variances.
    """
    if not -1.0 < correlation < 1.0:
        raise ValueError("DISCRETE_GXE_INTERVAL_TRUTH_INVALID")

    first: npt.NDArray[np.bool_] = group == 1.0
    """Located rows in the first environment group."""

    genetic_standard_deviation: npt.NDArray[np.float64] = np.where(
        first,
        GENETIC_SD[0],
        GENETIC_SD[1],
    )
    """Assigned fixed generating genetic scales by environment."""

    across: npt.NDArray[np.float64] = np.where(
        first[:, None] == first[None, :],
        1.0,
        correlation,
    )
    """Applied the selected correlation only across environments."""

    covariance: npt.NDArray[np.float64] = (
        relationship
        * np.outer(genetic_standard_deviation, genetic_standard_deviation)
        * across
    )
    """Combined additive relationships with group genetic covariance."""

    covariance[np.diag_indices(group.size)] += (
        np.where(
            first,
            RESIDUAL_SD[0],
            RESIDUAL_SD[1],
        )
        ** 2
    )
    """Added independent residual variance to every diagonal."""

    return covariance


def start_worker(payload: dict[str, Any]) -> None:
    """Install immutable participant-free campaign arrays in one worker.

    Args:
        payload: Relationship, environment, design, and truth-specific factors.
    """
    STATE.clear()
    STATE.update(payload)


def run_replicate(job: tuple[int, int]) -> dict[str, object]:
    """Run one public fit and correlation interval, retaining every failure.

    Args:
        job: Truth-grid index and zero-based replicate index.

    Returns:
        One mutually exclusive attempt classification and interval facts.
    """
    truth_index, replicate_index = job
    """Read the deterministic scientific-cell coordinates."""

    truth: float = TRUTHS[truth_index]
    """Selected the fixed generating genetic correlation."""

    relationship: npt.NDArray[np.float64] = STATE["relationship"]
    """Selected the participant-free relationship matrix."""

    group: npt.NDArray[np.float64] = STATE["group"]
    """Selected the fixed two-level environment layout."""

    design: npt.NDArray[np.float64] = STATE["design"]
    """Selected the prespecified fixed-effect design."""

    factor: npt.NDArray[np.float64] = STATE["factors"][truth]
    """Selected the known generating covariance factor for this truth."""

    seed: int = BASE_SEED + truth_index * 1_000_003 + replicate_index
    """Owned a random stream independent of process scheduling."""

    response: npt.NDArray[np.float64] = (
        design @ TRUE_EFFECTS
        + factor @ np.random.default_rng(seed).standard_normal(group.size)
    )
    """Drew the sole participant-free response for this attempt."""

    model: asterism.DiscreteGxeModel = asterism.DiscreteGxeModel(
        relationship,
        group,
        design,
        levels=(1.0, 2.0),
    )
    """Constructed the model through the documented public Python interface."""

    try:
        fit: dict[str, Any] = model.fit(response)
        """Requested the public free REML fit before inference."""
    except ValueError as error:
        return {
            "truth": truth,
            "index": replicate_index,
            "seed": seed,
            "status": "refused",
            "failure": str(error),
        }

    if fit.get("converged") is not True:
        return {
            "truth": truth,
            "index": replicate_index,
            "seed": seed,
            "status": "nonconverged",
            "failure": "public free fit returned converged=false",
        }

    try:
        interval: dict[str, Any] = model.correlation_interval(response)
        """Profiled genetic correlation through the documented public method."""
    except ValueError as error:
        return {
            "truth": truth,
            "index": replicate_index,
            "seed": seed,
            "status": "interval_refused",
            "failure": str(error),
        }

    profile_failures: int = int(interval["profile_failures"])
    """Counted every failed constrained likelihood evaluation."""

    if profile_failures > 0:
        return {
            "truth": truth,
            "index": replicate_index,
            "seed": seed,
            "status": "profile_failed",
            "failure": f"{profile_failures} failed profile evaluations",
            "profile_failures": profile_failures,
        }
    if interval.get("rule") != "chi2_1" or interval.get("estimator") != "reml":
        return {
            "truth": truth,
            "index": replicate_index,
            "seed": seed,
            "status": "interval_refused",
            "failure": "public interval returned an unexpected rule or estimator",
        }

    lower: float = float(interval["lower"])
    """Read the public lower profile endpoint."""

    upper: float = float(interval["upper"])
    """Read the public upper profile endpoint."""

    return {
        "truth": truth,
        "index": replicate_index,
        "seed": seed,
        "status": "complete",
        "failure": None,
        "covered": lower <= truth <= upper,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "reached_bound": bool(interval["lower_limited"] or interval["upper_limited"]),
        "profile_failures": profile_failures,
    }


def score_results(
    results: list[dict[str, object]],
    *,
    replicates: int,
) -> tuple[list[dict[str, object]], list[str]]:
    """Aggregate fixed-denominator interval cells and all unsuccessful outcomes.

    Args:
        results: Classified attempt records across the complete truth grid.
        replicates: Prespecified denominator per genetic-correlation truth.

    Returns:
        Truth-indexed coverage evidence and all campaign failures.
    """
    cells: list[dict[str, object]] = []
    """Collected exact coverage decisions in documented truth order."""

    failures: list[str] = []
    """Collected every incomplete or anti-conservative cell."""

    statuses: tuple[str, ...] = (
        "complete",
        "refused",
        "nonconverged",
        "interval_refused",
        "profile_failed",
    )
    """Named the mutually exclusive attempt outcome buckets."""

    for truth in TRUTHS:
        selected: list[dict[str, object]] = [
            result for result in results if result["truth"] == truth
        ]
        """Selected the fixed denominator for one correlation truth."""

        status_counts: dict[str, int] = {
            status: sum(result["status"] == status for result in selected)
            for status in statuses
        }
        """Counted every mutually exclusive public outcome."""

        complete: list[dict[str, object]] = [
            result for result in selected if result["status"] == "complete"
        ]
        """Selected only failure-free public intervals for summaries."""

        covered: int = sum(bool(result["covered"]) for result in complete)
        """Counted complete intervals containing their generating truth."""

        decision: dict[str, object] = coverage_decision(
            covered=covered,
            complete=len(complete),
            attempted=replicates,
        )
        """Applied the prewritten exact undercoverage rule."""

        if len(selected) != replicates or sum(status_counts.values()) != replicates:
            failures.append(f"rho={truth}: not every prespecified replicate was scored")
        for status in statuses[1:]:
            if status_counts[status]:
                failures.append(
                    f"rho={truth}: {status_counts[status]} {status} attempts"
                )
        if not decision["passed"]:
            failures.append(f"rho={truth}: {decision['verdict']}")

        cells.append(
            {
                "true_correlation": truth,
                **status_counts,
                "covered": covered,
                "coverage": decision["coverage"],
                "one_sided_95_percent_lower": decision["one_sided_95_percent_lower"],
                "one_sided_95_percent_upper": decision["one_sided_95_percent_upper"],
                "verdict": decision["verdict"],
                "reached_bound": (
                    float(
                        np.mean([bool(result["reached_bound"]) for result in complete])
                    )
                    if complete
                    else None
                ),
                "median_width": (
                    float(np.median([float(result["width"]) for result in complete]))
                    if complete
                    else None
                ),
                "passed": decision["passed"],
            }
        )
    return cells, failures


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit participant-free public interval coverage campaign.

    Args:
        argv: Optional command coordinates for tests and programmatic callers.

    Returns:
        Zero only when all attempts and fixed Monte Carlo decisions pass.
    """
    arguments: argparse.Namespace = parse_arguments(argv)
    """Read every scientific coordinate before constructing the design."""

    relationship, group, design = sibling_design(
        arguments.families,
        arguments.sibs_per_family,
    )
    """Built the fixed participant-free balanced sibling roster."""

    factors: dict[float, npt.NDArray[np.float64]] = {}
    """Collected one known-covariance Cholesky factor per truth."""

    for truth in TRUTHS:
        covariance: npt.NDArray[np.float64] = simulating_covariance(
            relationship,
            group,
            truth,
        )
        """Built the known generating covariance for this interval cell."""

        factors[truth] = np.linalg.cholesky(
            covariance + 1e-10 * np.eye(arguments.people)
        )
        """Factored the generating covariance once outside the replicate loop."""

    payload: dict[str, Any] = {
        "relationship": relationship,
        "group": group,
        "design": design,
        "factors": factors,
    }
    """Prepared immutable participant-free worker state."""

    jobs: list[tuple[int, int]] = [
        (truth_index, replicate_index)
        for truth_index in range(len(TRUTHS))
        for replicate_index in range(arguments.replicates)
    ]
    """Enumerated the complete truth-by-replicate denominator."""

    started: float = time.perf_counter()
    """Started timing after deterministic covariance preparation."""

    if arguments.workers == 1:
        start_worker(payload)
        results: list[dict[str, object]] = [run_replicate(job) for job in jobs]
        """Ran the bounded serial route used by integration tests."""
    else:
        with ProcessPoolExecutor(
            arguments.workers,
            initializer=start_worker,
            initargs=(payload,),
        ) as pool:
            results = list(pool.map(run_replicate, jobs, chunksize=1))
            """Ran fixed random streams independently of worker scheduling."""

    cells, failures = score_results(results, replicates=arguments.replicates)
    """Applied fixed denominators and exact prewritten coverage decisions."""

    statuses: tuple[str, ...] = (
        "complete",
        "refused",
        "nonconverged",
        "interval_refused",
        "profile_failed",
    )
    """Restated the exhaustive public outcome buckets for receipt validation."""

    release_sized: bool = (
        arguments.replicates == 400
        and arguments.people == 600
        and arguments.families == 150
        and arguments.sibs_per_family == 4
    )
    """Distinguished bounded smoke evidence from the exact documented campaign."""

    record: dict[str, object] = {
        "check": "discrete_gxe_correlation_interval",
        "participant_free": True,
        "public_interfaces": [
            "asterism.DiscreteGxeModel.fit",
            "asterism.DiscreteGxeModel.correlation_interval",
        ],
        "estimator": "reml",
        "rule": "chi2_1",
        "people": arguments.people,
        "families": arguments.families,
        "sibs_per_family": arguments.sibs_per_family,
        "largest_family": arguments.sibs_per_family,
        "replicates_per_truth": arguments.replicates,
        "truths": list(TRUTHS),
        "workers": arguments.workers,
        "attempted": len(jobs),
        "elapsed_seconds": time.perf_counter() - started,
        "every_replicate_scored": sum(
            int(cell[status]) for cell in cells for status in statuses
        )
        == len(jobs),
        "release_sized": release_sized,
        "acceptance": {
            "nominal": NOMINAL,
            "minimum_compatible": MINIMUM_COMPATIBLE,
            "maximum_compatible": MAXIMUM_COMPATIBLE,
            "fail_when": "one_sided_95_percent_clopper_pearson_upper_below_0.94",
            "maximum_unsuccessful_attempts": 0,
        },
        "cells": cells,
        "passed": not failures,
        "failures": failures,
    }
    """Built one auditable fixed-denominator public interval receipt."""

    print(json.dumps(record, indent=2))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
