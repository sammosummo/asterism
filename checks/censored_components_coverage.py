"""Does a several-component censored interval contain the truth as often as it claims?

The censored coverage rules that exist score one structured component and a
residual. This scores the full set the extended-high-frequency model fits:
additive, person-level, household and residual, together, with an interval taken
for **every** structured component and not only the first.

Three things about this design are not free choices, and each is a fact about
the model rather than a preference.

**Households have to cross families.** A household that is exactly a sibling
pair makes twice the kinship the identity plus the within-pair pattern, and the
household matrix the identity plus that same pattern, so the residual is their
exact combination and nothing separates additive from household. Here three
sibling pairs share a village and each household takes one person from each of
two different pairs. That identifies the three at Gram rank four while keeping
the likelihood's blocks to a village, so no censored region is larger than
twelve rows.

**The nominal is 0.975 and it is a lower-end statement.** A component
proportion of one puts every other component and the residual at nought,
leaving a covariance in which a person's two records are perfectly correlated;
it is singular, the profile cannot be evaluated there, and `src/interval.rs`
covers an end it could not evaluate rather than placing it on evidence never
gathered. So the upper end is the bound every time. A two-sided 95 per cent
interval puts 0.025 in each tail, and closing the upper one off hands that back.

**A truth sitting exactly on a bound is not scoreable here, and that is
deliberate.** `src/tobit.rs` fills the Self-Liang boundary verdict only when
there is one component, because with several of them more than one share can
rest on nought at once, which is not the case the one-component coverage rule
scored. An absent verdict says nobody has measured it. So the cell at nought is
run and reported as a **range**: an interval whose lower end is above nought
shuts a truth of nought out whatever any rule would say, and those are counted
as misses, while an interval whose end is on nought is left undecided rather
than decided by the end having landed there -- the obvious rule, which
`checks/tobit_coverage.py` records as the wrong one. The range is the most and
the least coverage the cell could have, and it fails if even the most is short
of nominal. The earlier several-component target rates used an outcome-adaptive
censoring limit and an analytic boundary reference. They are investigation
history, not qualification for this interval campaign.

A truth of exactly one is not attainable at this component set at all, for the
same reason the upper bound is not: it is a singular covariance, so there is
nothing to simulate from.

Run with:

    uv run --no-project python checks/censored_components_coverage.py

The command fixes each instrument limit from the population design before
drawing outcomes. The retained 31 August 2026 record contains 300 replicates in
each of twelve cells with no refusals. All eleven scoreable interior cells
passed their declared exact rule; the boundary cell remains a reported range.
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
from censoring_design import right_censoring_limit

VILLAGES: int = 400
"""Villages of three sibling pairs, which is 2,400 people."""

VILLAGE_PEOPLE: int = 6
"""People in each independent village block."""

RECORDS: int = 2
"""Records each person contributes, which is what wants a person-level component."""

VILLAGE_ROWS: int = VILLAGE_PEOPLE * RECORDS
"""Rows in each exact independent generating-covariance block."""

CENSORING: float = 0.52
"""Censoring share, the lighter of the two observed audiogram levels.

The heavier level is scored by `censored_components_target_design.py`, whose
interval held there. What failed at 0.75 was the boundary test, not the
interval.
"""

TOTAL_VARIANCE: float = 4.0
"""Generating complete-trait variance."""

TRUE_MEAN: float = 10.0
"""Generating complete-trait mean before censoring."""

CENSORING_LIMIT: float = right_censoring_limit(
    TRUE_MEAN,
    TOTAL_VARIANCE,
    CENSORING,
)
"""Fixed the instrument from generating facts before any outcome draw."""

TRUTHS: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.35, 0.25, 0.40),
    (0.20, 0.30, 0.20, 0.30),
    (0.35, 0.25, 0.20, 0.20),
    (0.60, 0.15, 0.10, 0.15),
)
"""Additive, person-level, household and residual shares in each cell.

The first puts the additive component exactly on its bound, which is the cell
the boundary rule would decide if there were one to decide it. The last carries
most of the variance in the additive term, which is as near one as this
component set reaches.
"""

NAMES: tuple[str, str, str] = ("additive", "person", "household")
"""Named the three structured components, in the order they are submitted."""

NOMINAL: float = 0.975
"""Coverage the interval can be held to here; see the module docstring."""

REPLICATES: int = 300
"""Replicates per cell."""

WORKERS: int = 12
"""Processes used."""

BASE_SEED: int = 993_000
"""Base seed separating this rule's replicates from every other check."""

STATE: dict[str, Any] = {}
"""Held each worker's copy of the components, built once rather than per replicate."""


FAMILY_WISE: float = 0.05
"""False-alarm rate for the campaign as a whole, not for one cell.

**This campaign makes twelve statements at once**, one for each component in
each cell, and a band drawn at 95 per cent for each of them separately is
excluded by chance in at least one cell 46 per cent of the time. A rule built
on that would fail about every other run of a model with nothing wrong with it,
which is a rule that teaches its reader to ignore it. So the band is drawn so
that the *campaign* raises a false alarm five times in a hundred, by splitting
the rate across the cells it scores.
"""


def clopper_pearson(hits: int, trials: int, cells: int) -> tuple[float, float]:
    """The exact simultaneous interval on a binomial proportion.

    Args:
        hits: Successes observed.
        trials: Attempts made.
        cells: How many cells share the campaign's false-alarm rate.

    Returns:
        The lower and upper Monte Carlo confidence limits.
    """
    alpha: float = FAMILY_WISE / cells
    """Split the campaign's rate across the statements it makes."""

    low: float = float(beta.ppf(alpha / 2, hits, trials - hits + 1)) if hits else 0.0
    """Took the lower limit, or nought where nothing hit."""

    high: float = (
        float(beta.ppf(1.0 - alpha / 2, hits + 1, trials - hits))
        if hits < trials
        else 1.0
    )
    """Took the upper limit, or one where everything hit."""

    return low, high


def components() -> dict[str, Any]:
    """Build the three structured components over records.

    Returns:
        The components, the fixed-effect design and the village size.
    """
    people: int = VILLAGES * VILLAGE_PEOPLE
    """Counted the people, six to a village."""

    additive: npt.NDArray[np.float64] = np.eye(people)
    """Started the additive relationship at unit marginal variance."""

    household: npt.NDArray[np.float64] = np.eye(people)
    """Started the household matrix at one person to a home."""

    for village in range(VILLAGES):
        base: int = village * 6
        """Located this village's first person."""

        for one, other in (
            (base, base + 1),
            (base + 2, base + 3),
            (base + 4, base + 5),
        ):
            additive[one, other] = additive[other, one] = 0.5
            """Made these two full siblings."""
        for one, other in (
            (base, base + 2),
            (base + 1, base + 4),
            (base + 3, base + 5),
        ):
            household[one, other] = household[other, one] = 1.0
            """Put these two in a home, from different families on purpose."""
    """Built one village's pedigree and homes, then the next."""

    sharing: npt.NDArray[np.float64] = np.ones((RECORDS, RECORDS))
    """Named the pattern that makes a person's own records share everything."""

    return {
        "components": [
            np.kron(additive, sharing),
            np.kron(np.eye(people), sharing),
            np.kron(household, sharing),
        ],
        "design": np.ones((people * RECORDS, 1)),
        "people": people,
    }


def village_component_blocks(
    model_components: list[npt.NDArray[np.float64]],
) -> list[npt.NDArray[np.float64]]:
    """Extract each structured component's exact 12-row diagonal blocks."""
    if len(model_components) != len(NAMES):
        raise ValueError("CENSORED_COMPONENT_COVERAGE_COMPONENT_COUNT_INVALID")
    rows: int = model_components[0].shape[0]
    """Read the common whole-design row count from the first component."""

    if rows % VILLAGE_ROWS != 0 or any(
        component.shape != (rows, rows) for component in model_components
    ):
        raise ValueError("CENSORED_COMPONENT_COVERAGE_COMPONENT_SHAPE_INVALID")
    return [
        np.stack(
            [
                component[start : start + VILLAGE_ROWS, start : start + VILLAGE_ROWS]
                for start in range(0, rows, VILLAGE_ROWS)
            ]
        )
        for component in model_components
    ]


def generating_factors(
    component_blocks: list[npt.NDArray[np.float64]],
    shares: tuple[float, float, float, float],
) -> npt.NDArray[np.float64]:
    """Factor one truth as independent 12-row village covariances."""
    blocks: int = component_blocks[0].shape[0]
    """Counted the independent villages represented by the block stack."""

    expected_shape: tuple[int, int, int] = (blocks, VILLAGE_ROWS, VILLAGE_ROWS)
    """Declared the exact common block-stack shape required for arithmetic."""

    if any(component.shape != expected_shape for component in component_blocks):
        raise ValueError("CENSORED_COMPONENT_COVERAGE_BLOCK_SHAPE_INVALID")
    covariance: npt.NDArray[np.float64] = TOTAL_VARIANCE * (
        shares[0] * component_blocks[0]
        + shares[1] * component_blocks[1]
        + shares[2] * component_blocks[2]
        + shares[3] * np.eye(VILLAGE_ROWS)
    )
    """Assembled only the exact diagonal blocks implied by this cell's truth."""

    return np.linalg.cholesky(covariance + 1e-9 * np.eye(VILLAGE_ROWS))


def apply_generating_factors(
    factors: npt.NDArray[np.float64],
    standard_normals: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply village factors to one globally ordered standard-normal draw."""
    if (
        factors.ndim != 3
        or factors.shape[1:] != (VILLAGE_ROWS, VILLAGE_ROWS)
        or standard_normals.shape != (factors.shape[0] * VILLAGE_ROWS,)
    ):
        raise ValueError("CENSORED_COMPONENT_COVERAGE_FACTOR_SHAPE_INVALID")
    generated: npt.NDArray[np.float64] = np.empty_like(standard_normals)
    """Allocated one output in the original global participant-record order."""

    for village, factor in enumerate(factors):
        start: int = village * VILLAGE_ROWS
        """Located this village in the one global normal vector."""

        stop: int = start + VILLAGE_ROWS
        """Located the first row belonging to the next village."""

        generated[start:stop] = factor @ standard_normals[start:stop]
        """Applied only this village's exact lower-triangular factor."""
    return generated


def start_worker() -> None:
    """Build this worker's copy of the components and every generating factor."""
    STATE.clear()
    """Removed state left by any earlier in-process validation call."""

    STATE.update(components())
    """Built the components once for every replicate this worker runs."""

    component_blocks: list[npt.NDArray[np.float64]] = village_component_blocks(
        STATE["components"]
    )
    """Extracted small generating blocks while retaining dense model components."""

    STATE["factors"] = [
        generating_factors(component_blocks, shares) for shares in TRUTHS
    ]
    """Built four stacks of 12-row factors rather than four dense factors."""


def one(job: tuple[int, int]) -> dict[str, Any]:
    """Fit one replicate and take an interval on every structured component.

    Args:
        job: Which cell of `TRUTHS` and which replicate.

    Returns:
        What each component's interval did, or the refusal that stopped it.
    """
    cell, replicate = job
    """Read what this attempt is."""

    shares: tuple[float, float, float, float] = TRUTHS[cell]
    """Read this cell's generating shares."""

    design: npt.NDArray[np.float64] = STATE["design"]
    """Read the fixed-effect design."""

    rows: int = design.shape[0]
    """Counted the rows once."""

    generator: np.random.Generator = np.random.default_rng(
        BASE_SEED + 7919 * replicate + 1_000_003 * cell
    )
    """Created the generator this replicate owns alone.

    The cell is in the seed. Without it every cell is fitted to the same draws,
    so two of them agreeing is one observation rather than two.
    """

    standard_normals: npt.NDArray[np.float64] = generator.standard_normal(rows)
    """Drew once in the original global row order, preserving every RNG stream."""

    complete: npt.NDArray[np.float64] = TRUE_MEAN + apply_generating_factors(
        STATE["factors"][cell],
        standard_normals,
    )
    """Drew the complete latent trait, before any instrument stopped."""

    ceiling: float = CENSORING_LIMIT
    """Applied the fixed instrument shared by every replicate."""

    censoring: npt.NDArray[np.int64] = (complete >= ceiling).astype(np.int64)
    """Marked 1 at or above the limit, 0 where measured."""

    value: npt.NDArray[np.float64] = np.where(censoring == 0, complete, np.nan)
    """Kept measured values; censored ones are never read."""

    limit: npt.NDArray[np.float64] = np.full(rows, ceiling)
    """Gave every row the same instrument limit."""

    model: Any = asterism.CensoredComponentModel(STATE["components"], design)
    """Prepared the model with all three structured components."""

    measured: list[dict[str, Any]] = []
    """Collected what each component's interval did."""

    for index, name in enumerate(NAMES):
        try:
            got: dict[str, Any] = model.interval(
                value, censoring, limit, component=index
            )
            """Profiled this component's mean-diagonal proportion."""
        except ValueError as refusal:
            measured.append({"component": name, "refusal": str(refusal)})
            continue
        truth: float = shares[index]
        """Named the generating share this interval must contain."""

        at_bound: bool = truth == 0.0
        """Read whether the truth sits exactly on the component's lower bound."""

        measured.append(
            {
                "component": name,
                "truth": truth,
                "lower": float(got["lower"]),
                "upper": float(got["upper"]),
                "lower_limited": bool(got["lower_limited"]),
                "upper_limited": bool(got["upper_limited"]),
                "verdict": got["contains_lower_bound"],
                "profile_failures": int(got["profile_failures"]),
                "at_bound": at_bound,
                "covered": None
                if at_bound
                else bool(got["lower"] <= truth <= got["upper"]),
            }
        )
    """Took an interval on every structured component of this one fit."""

    return {"cell": cell, "replicate": replicate, "components": measured}


def main() -> int:
    """Score the several-component censored interval against its nominal.

    Returns:
        Zero when every scoreable cell's exact interval covered the nominal.
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

    scored: int = len(TRUTHS) * len(NAMES)
    """Counted the statements this campaign makes, which set the band."""

    print(
        f"{VILLAGES * 6} people in villages of six, {RECORDS} records each, "
        f"{CENSORING:.2f} censored, {arguments.replicates} replicates per cell"
    )
    print("components: additive, person-level, household, residual")
    print(
        f"bands are simultaneous across {scored} cells, so the campaign raises a "
        f"false alarm {FAMILY_WISE:.2f} of the time and no single cell does"
    )

    jobs: list[tuple[int, int]] = [
        (cell, replicate)
        for cell in range(len(TRUTHS))
        for replicate in range(arguments.replicates)
    ]
    """Listed every attempt this campaign makes."""

    with ProcessPoolExecutor(
        max_workers=arguments.workers, initializer=start_worker
    ) as pool:
        rows: list[dict[str, Any]] = list(pool.map(one, jobs, chunksize=1))
        """Ran every attempt, each worker holding one copy of the components."""

    refusals: int = sum(
        1 for row in rows for item in row["components"] if "refusal" in item
    )
    """Counted every interval the model refused outright."""

    cells: list[dict[str, Any]] = []
    """Collected what each cell measured."""

    failures: list[str] = []
    """Collected every rule this campaign broke."""

    print()
    print(
        f"{'truth set':>22} {'component':>10} {'truth':>6} {'coverage':>9} "
        f"{'exact interval':>22} {'mean lower':>11}"
    )
    print("-" * 86)
    for cell, shares in enumerate(TRUTHS):
        for index, name in enumerate(NAMES):
            items: list[dict[str, Any]] = [
                row["components"][index]
                for row in rows
                if row["cell"] == cell and "refusal" not in row["components"][index]
            ]
            """Selected this component's intervals in this cell."""

            if not items:
                failures.append(f"{shares} {name}: nothing was measured")
                continue
            truth: float = shares[index]
            """Named the generating share."""

            label: str = "/".join(f"{share:.2f}" for share in shares)
            """Named the cell by its four shares."""

            mean_lower: float = float(np.mean([item["lower"] for item in items]))
            """Averaged the binding end of the interval."""

            if items[0]["at_bound"]:
                verdicts: int = sum(1 for item in items if item["verdict"] is not None)
                """Counted the intervals that carried a boundary verdict at all."""

                missed: int = sum(1 for item in items if item["lower"] > truth)
                """Counted the intervals that shut the truth out altogether.

                An end above nought excludes a truth of nought whatever the
                mixture would have said, so these are misses and not unknowns.
                """

                unknown: int = len(items) - missed
                """Counted the rest, whose end is on the bound and undecided."""

                best: float = (len(items) - missed) / len(items)
                """Took the most coverage this cell could have, every unknown a hit."""

                worst: float = (len(items) - missed - unknown) / len(items)
                """Took the least, every unknown a miss."""

                print(
                    f"{label:>22} {name:>10} {truth:>6.2f} "
                    f"{f'{worst:.2f}-{best:.2f}':>9} "
                    f"{f'{missed} shut out, {unknown} undecided':>22} "
                    f"{mean_lower:>11.4f}"
                )
                cells.append(
                    {
                        "truths": list(shares),
                        "component": name,
                        "truth": truth,
                        "scoreable": False,
                        "boundary_verdicts_present": verdicts,
                        "measured": len(items),
                        "shut_out": missed,
                        "undecided": unknown,
                        "coverage_bounds": [worst, best],
                        "mean_lower": mean_lower,
                    }
                )
                if verdicts:
                    failures.append(
                        f"{label} {name}: {verdicts} intervals carried a boundary "
                        "verdict, which this model is not supposed to fill in at "
                        "several components"
                    )
                bound_low, bound_high = clopper_pearson(
                    len(items) - missed, len(items), scored
                )
                """Put Monte Carlo uncertainty around the most it could have."""

                if bound_high < NOMINAL:
                    failures.append(
                        f"{label} {name}: {missed} of {len(items)} intervals shut the "
                        f"truth out, so coverage is at most {best:.4f} and its exact "
                        f"interval [{bound_low:.4f}, {bound_high:.4f}] is below the "
                        f"nominal {NOMINAL} however the undecided ones fall"
                    )
                continue
            hits: int = sum(1 for item in items if item["covered"])
            """Counted the intervals that contained the truth."""

            low, high = clopper_pearson(hits, len(items), scored)
            """Quantified Monte Carlo uncertainty around the coverage."""

            print(
                f"{label:>22} {name:>10} {truth:>6.2f} {hits / len(items):>9.4f} "
                f"{f'[{low:.4f}, {high:.4f}]':>22} {mean_lower:>11.4f}"
            )
            cells.append(
                {
                    "truths": list(shares),
                    "component": name,
                    "truth": truth,
                    "scoreable": True,
                    "measured": len(items),
                    "coverage": hits / len(items),
                    "exact_interval": [low, high],
                    "mean_lower": mean_lower,
                }
            )
            if not low <= NOMINAL <= high:
                failures.append(
                    f"{label} {name}: coverage {hits / len(items):.4f}, whose exact "
                    f"interval [{low:.4f}, {high:.4f}] excludes the nominal "
                    f"{NOMINAL}"
                )
    print()
    print(
        "The cell reported as a range puts a component's truth exactly on its lower\n"
        "bound. It is a range because containment splits in two there. An interval\n"
        "whose end is above nought shuts a truth of nought out whatever any rule\n"
        "would say, so those are counted as misses. An interval whose end is on\n"
        "nought is the Self-Liang question, and `src/tobit.rs` deliberately leaves\n"
        "that verdict absent at several components because nobody has measured the\n"
        "mixture there. Counting such an end as containment is the obvious rule and\n"
        "the wrong one, so they are left undecided. The pair is the most and the\n"
        "least coverage this cell could have, and the cell still fails if even the\n"
        "most is short of nominal."
    )
    if refusals:
        failures.append(f"{refusals} intervals were refused, and none is permitted")
    if not arguments.no_write:
        record: Path = (
            Path(__file__).resolve().parent.parent
            / "evidence"
            / f"censored-components-coverage-{date.today().isoformat()}.json"
        )
        """Named the dated evidence file."""

        record.write_text(
            json.dumps(
                {
                    "check": "censored_components_coverage",
                    "participant_free": True,
                    "people": VILLAGES * 6,
                    "records_per_person": RECORDS,
                    "rows": VILLAGES * 6 * RECORDS,
                    "largest_family": 2,
                    "largest_block_people": 6,
                    "expected_censoring_share": CENSORING,
                    "instrument_limit": CENSORING_LIMIT,
                    "components": [
                        "additive_relationship",
                        "person_level",
                        "household",
                        "residual",
                    ],
                    "nominal_coverage": NOMINAL,
                    "family_wise_false_alarm_rate": FAMILY_WISE,
                    "cells_sharing_the_band": scored,
                    "replicates_per_cell": arguments.replicates,
                    "refusals": refusals,
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
        f"\nPASSED: every scoreable component proportion covered its truth within "
        f"the exact interval around a nominal {NOMINAL}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
