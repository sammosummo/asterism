"""Does the several-component censored model work at the design it will be used on?

The existing censored target-design rule, `tobit_target_design`, qualifies one
structured component and a residual. Every other censored rule declares the same
pair, so **nothing had been measured beyond two components at the target
design**: the shares reported for a person-level or household term at a
hundred-and-eighty-person family were an extrapolation from sibling pairs.

The structure is the reviewed, participant-free aggregate the censored rule
already uses: 1,909 analysed people, 202 relationship components with the
largest at 180, and six fixed-effect columns. Nothing here is a participant
value or a reconstructed pedigree.

Two changes make it the design this model actually meets. Each person
contributes **two records**, which is what wants a person-level component --
without one the resemblance between a person's own two ears has nowhere to go
but the heritability. And each family is cut into **households of three**,
giving a shared-environment kernel beside the genetic one. Both are the
construction `checks/sequential_against_ghk.py` climbs its component ladder on,
so that ladder's verdict on censored dimension applies to this design rather
than to a different one.

Run with:

    uv run --no-project python checks/censored_components_target_design.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asterism
from sequential_against_ghk import HOUSEHOLD_SIZE, household_kernel
from tobit_target_design import load_target_design

FIXTURE: Path = (
    Path(__file__).resolve().parent
    / "design_fixtures"
    / "one_trait_gaussian_heritability.json"
)
"""Named the reviewed aggregate this design is matched to."""

FIXTURE_SHA256: str = "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e"
"""Pinned the exact reviewed fixture bytes outside the file itself."""

RECORDS: int = 2
"""Records each person contributes, which is two ears at one frequency."""

CENSORING_SHARES: tuple[float, float] = (0.52, 0.75)
"""Fixed the two observed high-frequency audiogram censoring levels."""

HERITABLE_SHARES: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
"""Genetic, person-level, household and residual shares in the interior cell."""

NULL_SHARES: tuple[float, float, float, float] = (0.0, 0.3846, 0.3077, 0.3077)
"""The same design with no genetic variance, the other three sharing it out.

The person-level and household terms stay present and keep their ratio to one
another. A null that removed them as well would test the boundary rule on a
one-component problem, which is the case already qualified.
"""

TOTAL_VARIANCE: float = 4.0
"""Latent complete-trait variance, matching the existing censored rule."""

TESTED: int = 0
"""Component scored: the genetic one, whose share is the heritability."""

LEVEL: float = 0.05
"""Boundary-test type-I-error level."""

BOUND: float = 1e-10
"""Below this a coefficient is resting on its bound rather than near it."""

EXPECTED_BOUND_SHARE: float = 0.5
"""Share of null fits the 50:50 mixture expects to rest on the bound.

**This is the reference's own assumption, written down so it can be measured.**
The mixture that turns the statistic into a p-value is half a point mass at
nought and half a chi-squared on one degree of freedom, and the half is the
share of null fits whose estimate lands exactly on the bound. Where the measured
share is not a half the reference does not describe the statistic, whatever the
p-values happen to look like in one run. It is reported beside the level because
it is the steadier of the two: a level moves several points between seed sets at
two hundred replicates, and this does not.
"""

NOMINAL_COVERAGE: float = 0.975
"""Nominal coverage of the interval at this design, and it is not 0.95.

**The upper end of a component proportion cannot be reached when a person
contributes more than one record.** A proportion of one puts every other
component, the residual included, at nought, leaving a covariance of the
additive matrix alone -- and spread over records that matrix gives a person's
own records a correlation of exactly one, so it is singular. The profile cannot
be evaluated there, and by the rule in `src/interval.rs` an end that could not
be evaluated is covered rather than placed on evidence never gathered. So the
upper end is the bound, every time, for structural reasons and not because a
fit went wrong. Measured here at one, two and three components alike.

What is left to measure is the lower end, and a two-sided 95 per cent profile
interval puts 0.025 in each tail. Forcing the upper end to the bound hands the
upper tail back, so the quantity this campaign can score is 0.975. Scoring it
against 0.95 would ask the interval to fail a quarter of the time in a
direction the design has closed off.
"""

REPLICATES: int = 200
"""Replicates per scenario per censoring share; four cells make 800."""

WORKERS: int = 8
"""Processes used, each holding its own copy of the three components."""

BASE_SEED: int = 981_000
"""Base seed separating this rule's replicates from every other check."""

LADDER_QUALIFIED_DIMENSION: int = 600
"""Largest censored dimension `sequential_against_ghk.py` has climbed.

Its own record says 600 is the largest rung climbed and not a limit that was
found: the ladder ran out before the approximation did. This check reports the
largest censored block it actually produced against that number, so a design
that outgrew the evidence says so rather than passing quietly.
"""

STATE: dict[str, Any] = {}
"""Held each worker's copy of the design, built once rather than per replicate."""


def clopper_pearson(hits: int, trials: int) -> tuple[float, float]:
    """The exact interval on a binomial proportion.

    Args:
        hits: Successes observed.
        trials: Attempts made.

    Returns:
        The lower and upper limits of the 95 per cent exact interval.
    """
    low: float = beta.ppf(0.025, hits, trials - hits + 1) if hits else 0.0
    """Took the lower limit, or nought where nothing hit."""

    high: float = beta.ppf(0.975, hits + 1, trials - hits) if hits < trials else 1.0
    """Took the upper limit, or one where everything hit."""

    return float(low), float(high)


def target_components() -> dict[str, Any]:
    """Build the three structured components and the design over records.

    Returns:
        The components, the fixed-effect design, the family row blocks and the
        largest family.
    """
    target: Any = load_target_design(FIXTURE, expected_sha256=FIXTURE_SHA256)
    """Read the reviewed aggregate and its structure-matched pedigree."""

    people: int = target.relationship.shape[0]
    """Counted the people the aggregate carries."""

    sharing: npt.NDArray[np.float64] = np.ones((RECORDS, RECORDS))
    """Named the pattern that makes a person's own records share everything."""

    additive: npt.NDArray[np.float64] = np.kron(target.relationship, sharing)
    """Spread the additive relationship over each person's records."""

    person: npt.NDArray[np.float64] = np.kron(np.eye(people), sharing)
    """Built the person-level component: one within a person, nought between."""

    household: npt.NDArray[np.float64] = np.kron(
        household_kernel(target.relationship, HOUSEHOLD_SIZE), sharing
    )
    """Built the household component by cutting each family into homes of three."""

    design: npt.NDArray[np.float64] = np.repeat(target.design, RECORDS, axis=0)
    """Gave a person's records the same six fixed-effect columns."""

    blocks: list[npt.NDArray[np.int64]] = []
    """Collected the row ranges of each independent family."""

    offset: int = 0
    """Tracked contiguous family slices in generated row order."""

    for size in target.component_sizes:
        blocks.append(np.arange(offset, offset + size * RECORDS))
        offset += size * RECORDS
        """Took this family's rows and advanced to the next."""
    """Collected the row ranges of each independent family."""

    return {
        "components": [additive, person, household],
        "design": design,
        "blocks": blocks,
        "largest_family": max(target.component_sizes),
        "people": people,
        "fixture_sha256": target.fixture_sha256,
    }


def start_worker() -> None:
    """Build this worker's copy of the design and both generating factors."""
    STATE.update(target_components())
    """Built the components once for every replicate this worker runs."""

    rows: int = STATE["design"].shape[0]
    """Counted the rows the design carries."""

    for scenario, shares in (("null", NULL_SHARES), ("heritable", HERITABLE_SHARES)):
        covariance: npt.NDArray[np.float64] = TOTAL_VARIANCE * (
            shares[0] * STATE["components"][0]
            + shares[1] * STATE["components"][1]
            + shares[2] * STATE["components"][2]
            + shares[3] * np.eye(rows)
        )
        """Assembled the covariance this scenario's truths imply."""

        STATE[f"factor_{scenario}"] = np.linalg.cholesky(
            covariance + 1e-9 * np.eye(rows)
        )
        """Factored it once; it is the same for every replicate of the scenario."""
    """Built both generating factors once rather than per replicate."""


def one(job: tuple[str, float, int]) -> dict[str, Any]:
    """Fit and score one replicate at the target design.

    Args:
        job: The scenario, the censoring share and the replicate number.

    Returns:
        What this replicate measured, or the refusal that stopped it.
    """
    scenario, share, replicate = job
    """Read what this attempt is."""

    design: npt.NDArray[np.float64] = STATE["design"]
    """Read the fixed-effect design."""

    rows: int = design.shape[0]
    """Counted the rows once."""

    generator: np.random.Generator = np.random.default_rng(
        BASE_SEED + 7919 * replicate + 104_729 * int(share * 100) + len(scenario)
    )
    """Created the generator this replicate owns alone."""

    coefficients: npt.NDArray[np.float64] = np.array([1.0, 0.2, -0.1, 0.05, 0.3, -0.2])
    """Fixed the six latent-mean coefficients."""

    complete: npt.NDArray[np.float64] = design @ coefficients + STATE[
        f"factor_{scenario}"
    ] @ generator.standard_normal(rows)
    """Drew the complete latent trait, before any instrument stopped."""

    censored_count: int = round(share * rows)
    """Selected the exact attainable censoring count."""

    ceiling: float = float(np.sort(complete)[rows - censored_count])
    """Put the limit at the smallest value the instrument fails to measure."""

    censoring: npt.NDArray[np.int64] = (complete >= ceiling).astype(np.int64)
    """Marked 1 at or above the limit, 0 where measured.

    At or above, not above. Taking the limit as the largest measured value and
    censoring strictly above it leaves one row measured at exactly its own
    limit, which is a row the instrument both did and did not read. Every other
    check in this directory censors at or above a limit taken from the
    distribution, and this now does too.
    """

    value: npt.NDArray[np.float64] = np.where(censoring == 0, complete, np.nan)
    """Kept measured values; censored ones are never read."""

    limit: npt.NDArray[np.float64] = np.full(rows, ceiling)
    """Gave every row the same instrument limit."""

    worst_block: int = max(int(censoring[block].sum()) for block in STATE["blocks"])
    """Recorded the largest censored dimension the likelihood had to resolve."""

    model: Any = asterism.CensoredComponentModel(STATE["components"], design)
    """Prepared the model with all three structured components."""

    result: dict[str, Any] = {
        "scenario": scenario,
        "censoring_share": share,
        "replicate": replicate,
        "worst_censored_block": worst_block,
        "achieved_censoring_share": float(censoring.sum()) / rows,
    }
    """Started this replicate's record with what it was asked to do."""

    try:
        if scenario == "null":
            test: dict[str, Any] = model.test(value, censoring, limit, TESTED)
            """Tested the genetic component against having no variance at all."""

            fit: dict[str, Any] = model.fit(value, censoring, limit)
            """Fitted it freely as well, to see where the estimate landed."""

            result.update(
                p_value=float(test["p_value"]),
                test_rule=str(test["rule"]),
                rejected=bool(test["p_value"] < LEVEL),
                nuisance_at_bound=bool(test.get("nuisance_at_bound")),
                on_bound=bool(float(fit["coefficients"][TESTED]) <= BOUND),
            )
        else:
            interval: dict[str, Any] = model.interval(
                value, censoring, limit, component=TESTED
            )
            """Profiled the genetic component's mean-diagonal proportion."""

            bound_unreachable: bool = bool(
                interval["upper_limited"] and interval["contains_upper_bound"] is None
            )
            """Read whether the upper end is the bound because it could not be scored."""

            result.update(
                estimate=float(interval["estimate"]),
                lower=float(interval["lower"]),
                upper=float(interval["upper"]),
                profile_failures=int(interval["profile_failures"]),
                bound_unreachable=bound_unreachable,
                covered=bool(
                    interval["lower"] <= HERITABLE_SHARES[TESTED] <= interval["upper"]
                ),
                extra_failures=int(interval["profile_failures"])
                - (1 if bound_unreachable else 0),
            )
    except ValueError as refusal:
        result.update(refusal=str(refusal))
        """Recorded the refusal rather than dropping the replicate."""
    return result


def main() -> int:
    """Score the several-component censored model at its intended design.

    Returns:
        Zero when every cell met its rule and nothing failed.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )
    """Built the command line."""

    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--no-write", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()
    """Read the campaign size and whether this run records evidence."""

    structure: dict[str, Any] = target_components()
    """Built the design once here, only to report what it is."""

    print(
        f"{structure['people']} people, {RECORDS} records each, "
        f"largest family {structure['largest_family']}, "
        f"{arguments.replicates} replicates per cell"
    )
    print(
        f"components: additive, person-level, households of {HOUSEHOLD_SIZE}, residual"
    )

    jobs: list[tuple[str, float, int]] = [
        (scenario, share, replicate)
        for scenario in ("null", "heritable")
        for share in CENSORING_SHARES
        for replicate in range(arguments.replicates)
    ]
    """Listed every attempt this campaign makes."""

    with ProcessPoolExecutor(
        max_workers=arguments.workers, initializer=start_worker
    ) as pool:
        rows: list[dict[str, Any]] = list(pool.map(one, jobs, chunksize=1))
        """Ran every attempt, each worker holding one copy of the design."""

    failed: list[dict[str, Any]] = [
        row for row in rows if "refusal" in row or row.get("extra_failures", 0) > 0
    ]
    """Collected every attempt that did not produce the inference asked of it.

    The one profile failure at an unreachable upper bound is not counted here.
    Nothing went wrong in it: the bound is a covariance the design cannot form,
    and the interval says so through `upper_limited`. A failure anywhere else in
    the bracket is counted, because that is a fit that should have been made and
    was not.
    """

    limited: int = sum(1 for row in rows if row.get("bound_unreachable"))
    """Counted the intervals whose upper end is the bound it could not evaluate."""

    worst_block: int = max(int(row["worst_censored_block"]) for row in rows)
    """Took the largest censored dimension anywhere in the campaign."""

    cells: list[dict[str, Any]] = []
    """Collected what each cell measured."""

    failures: list[str] = []
    """Collected every rule this campaign broke."""

    print()
    print(
        f"{'scenario':>10} {'censored':>9} {'attempts':>9} {'measured':>9} "
        f"{'rate':>8} {'exact interval':>22} {'nominal':>8} {'on bound':>11}"
    )
    print("-" * 94)
    for scenario in ("null", "heritable"):
        for share in CENSORING_SHARES:
            cell: list[dict[str, Any]] = [
                row
                for row in rows
                if row["scenario"] == scenario and row["censoring_share"] == share
            ]
            """Selected this cell's attempts."""

            usable: list[dict[str, Any]] = [row for row in cell if "refusal" not in row]
            """Selected the attempts that produced a number."""

            if not usable:
                failures.append(f"{scenario} at {share:.2f}: nothing was measured")
                continue
            key: str = "rejected" if scenario == "null" else "covered"
            """Named what this scenario counts."""

            nominal: float = LEVEL if scenario == "null" else NOMINAL_COVERAGE
            """Named what that count is scored against."""

            hits: int = sum(1 for row in usable if row.get(key))
            """Counted them."""

            bound_share: float | None = (
                sum(1 for row in usable if row.get("on_bound")) / len(usable)
                if scenario == "null"
                else None
            )
            """Measured the share of null fits resting on the bound, or nothing."""

            low, high = clopper_pearson(hits, len(usable))
            """Quantified Monte Carlo uncertainty around the rate."""

            print(
                f"{scenario:>10} {share:>9.2f} {len(cell):>9d} {len(usable):>9d} "
                f"{hits / len(usable):>8.4f} "
                f"{f'[{low:.4f}, {high:.4f}]':>22} {nominal:>8.2f}"
                f"{'' if bound_share is None else f'{bound_share:>12.3f}'}"
            )
            cells.append(
                {
                    "scenario": scenario,
                    "censoring_share": share,
                    "attempted": len(cell),
                    "measured": len(usable),
                    "rate": hits / len(usable),
                    "exact_interval": [low, high],
                    "nominal": nominal,
                    "share_on_bound": bound_share,
                    "share_on_bound_expected": (
                        EXPECTED_BOUND_SHARE if scenario == "null" else None
                    ),
                }
            )
            if bound_share is not None:
                resting: int = sum(1 for row in usable if row.get("on_bound"))
                """Counted the null fits resting on the bound."""

                bound_low, bound_high = clopper_pearson(resting, len(usable))
                """Put an exact interval around that share."""

                if not bound_low <= EXPECTED_BOUND_SHARE <= bound_high:
                    failures.append(
                        f"{scenario} at {share:.2f}: {bound_share:.3f} of null fits "
                        f"rest on the bound, whose exact interval "
                        f"[{bound_low:.4f}, {bound_high:.4f}] excludes the "
                        f"{EXPECTED_BOUND_SHARE} the 50:50 mixture assumes, so the "
                        "reference does not describe the statistic here"
                    )
            if not low <= nominal <= high:
                failures.append(
                    f"{scenario} at {share:.2f}: {hits / len(usable):.4f} against a "
                    f"nominal {nominal:.2f}, whose exact interval "
                    f"[{low:.4f}, {high:.4f}] excludes it"
                )
    print()
    print(
        f"Largest censored block anywhere in the campaign: {worst_block}, against the "
        f"{LADDER_QUALIFIED_DIMENSION} the region ladder has climbed."
    )
    intervals: int = sum(1 for row in rows if "lower" in row)
    """Counted the attempts that produced an interval at all."""

    if intervals:
        print(
            f"Intervals whose upper end is the unreachable bound: {limited}/{intervals}."
            " A proportion of one leaves a person's own records perfectly"
            " correlated, so the coverage above is a lower-end statement scored"
            f" against {NOMINAL_COVERAGE}."
        )
    if worst_block > LADDER_QUALIFIED_DIMENSION:
        failures.append(
            f"the campaign produced a censored block of {worst_block}, past the "
            f"{LADDER_QUALIFIED_DIMENSION} the ladder has qualified, so its region "
            "probabilities are outside the evidence"
        )
    if failed:
        failures.append(
            f"{len(failed)} of {len(rows)} attempts failed, and none is permitted"
        )
    if not arguments.no_write:
        record: Path = (
            Path(__file__).resolve().parent.parent
            / "evidence"
            / f"censored-components-target-design-{date.today().isoformat()}.json"
        )
        """Named the dated evidence file."""

        record.write_text(
            json.dumps(
                {
                    "check": "censored_components_target_design",
                    "participant_free": True,
                    "people": structure["people"],
                    "records_per_person": RECORDS,
                    "rows": structure["design"].shape[0],
                    "largest_family": structure["largest_family"],
                    "household_size": HOUSEHOLD_SIZE,
                    "components": [
                        "additive_relationship",
                        "person_level",
                        "household",
                        "residual",
                    ],
                    "heritable_shares": list(HERITABLE_SHARES),
                    "null_shares": list(NULL_SHARES),
                    "attempted": len(rows),
                    "failed": len(failed),
                    "largest_censored_block": worst_block,
                    "intervals_limited_at_unreachable_bound": limited,
                    "ladder_qualified_dimension": LADDER_QUALIFIED_DIMENSION,
                    "fixture_sha256": structure["fixture_sha256"],
                    "cells": cells,
                    "passed": not failures,
                },
                indent=1,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten to {record}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        "\nPASSED: the boundary test held its level and the interval held its "
        "coverage at both censoring shares, with no failed attempt, at a "
        "several-component set."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
