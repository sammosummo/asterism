"""How many relatives before an additive and a person-level component separate?

Both are one within a person. What tells them apart is that additive kinship is
also non-zero between relatives and a person-level matrix is not, so the
information comes from **related pairs** rather than from records: a person's
second record sharpens their sum and says nothing about the split.

That makes the question one about the design, and this command is the retained
way to measure where a design is enough. It now fixes the censoring limit from
the population design before drawing outcomes. Results from its earlier
outcome-adaptive generator are superseded pending a fresh run.

**The question is how many, not whether.** A person-level component is only ever
added because a person contributes more than one record -- two ears, two
occasions -- so the sweep is over related pairs at two and three records each,
which is what a real design varies. (With one record a person's matrix is the
identity and the term is the residual, which is why nobody writes that model;
it is not a trap anyone falls into and is not swept here.)

components exchange variance, an over-estimate of one arrives with an
under-estimate of the other, so the correlation between their estimates across
replicates goes towards minus one while the standard deviation of their sum
stays small. Both are reported. A wide estimate uncorrelated with its neighbour
is an imprecise answer; a wide estimate correlated with it is half of one answer
reported twice, and the sum is the part that was actually measured.

**The pass rule is consistency, not unbiasedness, and the difference is the
point.** Maximum likelihood variance components are biased in finite samples --
that is what REML exists to remedy, and this model cannot use REML because a
censored observation has no residual to project onto the null space of the
design. The rule is that the bias shrinks with the design and is small at the
largest one swept. Its numerical verdict must come from the corrected run, not
from the superseded exact-count campaign.

How precisely a given pedigree separates the two is a fact about that pedigree,
and the eventual table is the answer to it rather than a verdict on it.

Run with:

    uv run --no-project python checks/component_separation.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import asterism
import numpy as np

try:
    from checks.censoring_design import right_censoring_limit
except ModuleNotFoundError:
    from censoring_design import right_censoring_limit
"""Imported the fixed-instrument solver in module and direct-command modes."""

ADDITIVE: float = 0.40
"""Generating additive share."""

PERSON: float = 0.30
"""Generating person-level share."""

RESIDUAL: float = 0.30
"""What the two leave, and what the model estimates as the remainder."""

FAMILIES: list[int] = [30, 60]
"""Sibling pairs swept, which is what carries the additive information.

The range brackets the paper's largest expanded pedigree component (about 150
rows) without turning the check into a dense-matrix stress test unrelated to
the design being qualified."""

RECORDS: list[int] = [2, 3]
"""Records per person -- two ears, or three occasions.

One record is not swept. A person-level component exists because a person was
measured more than once; with one record its matrix is the identity and the term
is the residual, so the model nobody writes is not a boundary worth spending a
third of the sweep on."""

REPLICATES: int = 200
"""Replicates per cell, which fixes how finely bias can be resolved."""

CENSORING_SHARE: float = 0.4
"""Share of records censored, this being the censored model."""

CENSORING_LIMIT: float = right_censoring_limit(
    0.0,
    ADDITIVE + PERSON + RESIDUAL,
    CENSORING_SHARE,
)
"""Fixed the instrument from the generating marginal distribution."""

BIAS_AT_LARGEST: float = 0.03
"""How far the mean estimate may sit from its truth at the largest design swept.

An estimator that is merely consistent must still have arrived by the time a
design is realistic, and this is where "arrived" is drawn. It is not a claim
that smaller designs are wrong -- they are biased, measurably and in the
direction theory predicts -- only that the bias is gone by the top of the
sweep."""

SPLIT_COST: float = 3.0
"""How many times less precisely each component may be known than their sum
before the split is the thing limiting what can be said.

**Not the correlation between the estimates, which was the first attempt and
measures the wrong thing.** That correlation goes towards minus one as a design
grows -- measured at -0.920, -0.959 and -0.973 over thirty, sixty and a hundred
and twenty sibling pairs -- because the sum becomes precisely determined and the
split is then the only uncertainty left. It describes the shape of the
uncertainty, which is always a trade-off here, and says nothing about whether a
design is adequate. What does say so is how much worse each part is known than
the whole."""

WORKERS: int = 8
"""Independent design cells that may run at once."""


@dataclass(frozen=True)
class CellJob:
    """One complete design cell owned by one process."""

    records: int
    """Records contributed by each person."""

    families: int
    """Sibling pairs carrying the additive information."""

    replicates: int
    """Deterministic response fits in this cell."""

    base_seed: int
    """Command seed before the historical coordinate increments."""

    largest_families: int
    """Largest requested design, where the existing bias rule is scored."""

    censoring_share: float = CENSORING_SHARE
    """Expected right-censoring share for this fixed-instrument cell."""


def design_matrices(families: int, records: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the additive and person-level components for one design.

    Args:
        families: Sibling pairs; each contributes two people.
        records: Records each person contributes.

    Returns:
        The additive matrix, the person-level matrix, and the row count.
    """
    people: int = families * 2
    """Counted the people, two to a sibling pair."""

    rows: int = people * records
    """Counted the rows, several to a person."""

    additive: np.ndarray = np.zeros((rows, rows))
    """Initialised the additive relationship over records."""

    for i in range(rows):
        for j in range(rows):
            first, second = i // records, j // records
            """Found which person each of the two rows belongs to."""

            if first == second:
                additive[i, j] = 1.0
                """Set a person's own records to one."""
            elif first // 2 == second // 2:
                additive[i, j] = 0.5
                """Set full siblings to a half."""
    """Set a person's own records to one and full siblings to a half."""

    person: np.ndarray = asterism.grouping_matrix(
        [f"person-{row // records}" for row in range(rows)]
    )
    """Built the person-level component by grouping the rows of one person."""
    return additive, person, rows


def one_replicate(families: int, records: int, seed: int) -> tuple[float, float] | None:
    """Fit one simulated data set and return the two estimates.

    Args:
        families: Sibling pairs in this design.
        records: Records each person contributes.
        seed: Fixed so a cell can be reproduced on its own.

    Returns:
        The additive and person-level proportions, or None where no fit
        converged.
    """
    factor, model, rows = prepare_cell(families, records)
    """Prepared the complete fixed state used by this standalone replicate."""

    return fit_prepared_replicate(factor, model, rows, seed)


def prepare_cell(
    families: int,
    records: int,
) -> tuple[np.ndarray, asterism.CensoredComponentModel, int]:
    """Build and factor one cell's dense state once.

    Args:
        families: Sibling pairs in this design.
        records: Records each person contributes.

    Returns:
        The generating factor, reusable fitted-model object, and row count.
    """
    additive, person, rows = design_matrices(families, records)
    """Built the two fixed covariance components for this cell."""

    covariance: np.ndarray = (
        ADDITIVE * additive + PERSON * person + RESIDUAL * np.eye(rows)
    )
    """Assembled the covariance the truths imply once per cell."""

    factor: np.ndarray = np.linalg.cholesky(covariance + 1e-10 * np.eye(rows))
    """Factorised the generating covariance once rather than per response."""

    people_index: np.ndarray = np.repeat(np.arange(rows // records), records)
    """Assigned the same covariates to both records contributed by a person."""

    age: np.ndarray = (people_index % 17).astype(float) - 8.0
    """Provided a centred age-like covariate without participant data."""

    sex: np.ndarray = (people_index % 2).astype(float)
    """Provided a balanced binary sex-like covariate for the design check."""

    design: np.ndarray = np.column_stack(
        [
            np.ones(rows),
            age,
            age**2,
            sex,
            age * sex,
            age**2 * sex,
        ]
    )
    """Used the paper's six fixed-effect columns in every simulated fit."""

    model: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
        [additive, person], design
    )
    """Prepared one immutable model over the cell's fixed matrices."""
    return factor, model, rows


def fit_prepared_replicate(
    factor: np.ndarray,
    model: asterism.CensoredComponentModel,
    rows: int,
    seed: int,
) -> tuple[float, float] | None:
    """Fit one response using already prepared cell state.

    Args:
        factor: Cholesky factor of the known generating covariance.
        model: Public censored-component model for this cell.
        rows: Response rows in this design.
        seed: Historical independent replicate seed.

    Returns:
        The two comparable component proportions, or ``None`` on refusal.
    """

    generator: np.random.Generator = np.random.default_rng(seed)
    """Seeded this replicate."""

    complete: np.ndarray = factor @ generator.standard_normal(rows)
    """Drew the complete trait, before any instrument stopped."""

    ceiling: float = CENSORING_LIMIT
    """Applied the pre-outcome instrument shared by every replicate."""

    censoring: np.ndarray = (complete >= ceiling).astype(np.int64)
    """Marked 1 at or above the limit, 0 where measured."""

    value: np.ndarray = np.where(censoring == 0, complete, np.nan)
    """Kept measured values; censored ones are never read."""

    try:
        fit: dict[str, Any] = model.fit(value, censoring, np.full(rows, ceiling))
        """Fitted this replicate."""
    except ValueError:
        return None
    """Refused fits are counted rather than retried."""

    if not fit["converged"]:
        return None
    shares: list[float] = fit["mean_diagonal_proportions"]
    """Took the comparable quantity, the residual's share last."""
    return float(shares[0]), float(shares[1])


def fit_prepared_replicate_at_limit(
    factor: np.ndarray,
    model: asterism.CensoredComponentModel,
    rows: int,
    seed: int,
    ceiling: float,
) -> tuple[float, float] | None:
    """Fit one replicate at a caller-selected fixed instrument limit."""
    generator: np.random.Generator = np.random.default_rng(seed)
    """Seeded this fixed-limit replicate."""

    complete: np.ndarray = factor @ generator.standard_normal(rows)
    """Drew the complete trait before applying the fixed instrument."""

    censoring: np.ndarray = (complete >= ceiling).astype(np.int64)
    """Marked observations that reached the caller-selected limit."""

    value: np.ndarray = np.where(censoring == 0, complete, np.nan)
    """Retained measured values and withheld censored latent outcomes."""

    try:
        fit: dict[str, Any] = model.fit(value, censoring, np.full(rows, ceiling))
        """Fitted this replicate under the fixed measurement instrument."""
    except ValueError:
        return None
    if not fit["converged"]:
        return None
    shares: list[float] = fit["mean_diagonal_proportions"]
    """Read the two comparable component proportions from the converged fit."""

    return float(shares[0]), float(shares[1])


# asterism-style: allow private-helper -- public worker wrapper adds cell coordinates to unexpected failures
def _run_cell(job: CellJob) -> dict[str, object]:
    """Run and summarise one complete deterministic design cell."""
    if (
        job.records < 1
        or job.families < 1
        or job.replicates < 1
        or job.largest_families < job.families
        or not 0.0 < job.censoring_share < 1.0
    ):
        raise ValueError("COMPONENT_SEPARATION_CELL_JOB_INVALID")

    factor, model, rows = prepare_cell(job.families, job.records)
    """Built the cell's dense state once inside its owning process."""

    pairs: list[tuple[float, float]] = []
    """Collected every converged estimate pair in replicate order."""

    for replicate in range(job.replicates):
        seed: int = job.base_seed + 7919 * replicate + job.families + job.records
        """Preserved the original scheduling-independent replicate stream."""

        if job.censoring_share == CENSORING_SHARE:
            got: tuple[float, float] | None = fit_prepared_replicate(
                factor, model, rows, seed
            )
            """Fitted the cell at the retained default instrument limit."""
        else:
            got: tuple[float, float] | None = fit_prepared_replicate_at_limit(
                factor,
                model,
                rows,
                seed,
                right_censoring_limit(
                    0.0,
                    ADDITIVE + PERSON + RESIDUAL,
                    job.censoring_share,
                ),
            )
            """Fitted the cell at its requested fixed instrument limit."""

        if got is not None:
            pairs.append(got)
    """Attempted every predeclared replicate and retained every completed fit."""

    failures: list[str] = []
    """Collected the existing availability and largest-design bias failures."""

    if len(pairs) < job.replicates // 2:
        failures.append(
            f"records {job.records}, families {job.families}: only "
            f"{len(pairs)} of {job.replicates} replicates fitted"
        )
        return {
            "records_per_person": job.records,
            "families": job.families,
            "censoring_share": job.censoring_share,
            "summary": None,
            "failures": failures,
        }

    additive_hat: np.ndarray = np.array([pair[0] for pair in pairs])
    """Collected the additive estimate from every completed replicate."""

    person_hat: np.ndarray = np.array([pair[1] for pair in pairs])
    """Collected the person-level estimate from every completed replicate."""

    correlation: float = (
        float(np.corrcoef(additive_hat, person_hat)[0, 1])
        if len(pairs) > 2
        else float("nan")
    )
    """Measured the existing across-replicate component trade-off."""

    additive_sd: float = float(additive_hat.std(ddof=1))
    """Computed additive uncertainty once for reporting and the split cost."""

    person_sd: float = float(person_hat.std(ddof=1))
    """Computed person-level uncertainty once for reporting and the split cost."""

    sum_sd: float = float((additive_hat + person_hat).std(ddof=1))
    """Measured how precisely the combined component was estimated."""

    summary: dict[str, object] = {
        "records_per_person": job.records,
        "families": job.families,
        "censoring_share": job.censoring_share,
        "replicates_fitted": len(pairs),
        "additive": {
            "mean": float(additive_hat.mean()),
            "sd": additive_sd,
            "truth": ADDITIVE,
        },
        "person": {
            "mean": float(person_hat.mean()),
            "sd": person_sd,
            "truth": PERSON,
        },
        "correlation_between_estimates": correlation,
        "sd_of_their_sum": sum_sd,
        "split_costs_this_many_times_the_precision": float(
            max(additive_sd, person_sd) / max(sum_sd, 1e-12)
        ),
    }
    """Retained the same fields used by existing evidence and pass rules."""

    for name, truth, values in (
        ("additive", ADDITIVE, additive_hat),
        ("person", PERSON, person_hat),
    ):
        bias: float = float(values.mean()) - truth
        """Measured how far this cell's mean sits from its truth."""

        summary[f"{name}_bias"] = bias
        """Recorded the bias at this design, scored or not."""

        resolvable: float = BIAS_AT_LARGEST + 2.0 * float(
            values.std(ddof=1) / np.sqrt(len(values))
        )
        """Widened the tolerance by twice this cell's Monte Carlo error."""

        summary[f"{name}_bias_resolvable_beyond"] = resolvable
        """Recorded the smallest bias this run could resolve beyond tolerance."""

        if job.families == job.largest_families and abs(bias) > resolvable:
            failures.append(
                f"records {job.records}, families {job.families}: {name} came "
                f"back at {values.mean():.3f} where the truth is {truth}, a bias "
                f"of {bias:+.3f} at the largest design swept -- past "
                f"{resolvable:.3f}, which is the tolerance widened by this run's "
                "own Monte Carlo error"
            )
    """Applied the unchanged bias rule only at the largest requested design."""

    return {
        "records_per_person": job.records,
        "families": job.families,
        "censoring_share": job.censoring_share,
        "summary": summary,
        "failures": failures,
    }


def run_cell(job: CellJob) -> dict[str, object]:
    """Run one cell while making unexpected worker failures coordinate-visible."""
    try:
        return _run_cell(job)
    except Exception as error:
        raise RuntimeError(
            "COMPONENT_SEPARATION_CELL_FAILED "
            f"records={job.records} families={job.families}"
        ) from error


def validate_complete_cell_results(
    results: list[dict[str, object]],
    jobs: list[CellJob],
) -> list[dict[str, object]]:
    """Require exactly one valid result for every requested design cell."""
    expected: list[tuple[int, int, float]] = [
        (job.records, job.families, job.censoring_share) for job in jobs
    ]
    """Retained the command's original deterministic cell order."""

    if len(set(expected)) != len(expected):
        raise ValueError("COMPONENT_SEPARATION_CELL_JOBS_DUPLICATE")
    order: dict[tuple[int, int, float], int] = {
        coordinate: index for index, coordinate in enumerate(expected)
    }
    """Mapped each unique coordinate back to command order."""

    seen: set[tuple[int, int, float]] = set()
    """Collected worker coordinates exactly once."""

    for result in results:
        records: object = result.get("records_per_person")
        """Read the records coordinate without coercing worker output."""

        families: object = result.get("families")
        """Read the family coordinate without coercing worker output."""

        share: object = result.get("censoring_share", CENSORING_SHARE)
        """Read the expected censoring coordinate without coercing worker output."""

        if (
            type(records) is not int
            or type(families) is not int
            or type(share) is not float
        ):
            raise ValueError("COMPONENT_SEPARATION_COORDINATES_INCOMPLETE")
        coordinate: tuple[int, int, float] = (records, families, share)
        """Narrowed the exact primitive coordinate."""

        if coordinate in seen:
            raise ValueError("COMPONENT_SEPARATION_COORDINATE_DUPLICATE")
        if coordinate not in order:
            raise ValueError("COMPONENT_SEPARATION_COORDINATES_INCOMPLETE")
        if not isinstance(result.get("failures"), list) or not (
            isinstance(result.get("summary"), dict) or result.get("summary") is None
        ):
            raise ValueError("COMPONENT_SEPARATION_CELL_RESULT_INVALID")
        seen.add(coordinate)
    """Validated every returned coordinate and its result containers."""

    if seen != set(expected):
        raise ValueError("COMPONENT_SEPARATION_COORDINATES_INCOMPLETE")
    return sorted(
        results,
        key=lambda result: order[
            (
                int(result["records_per_person"]),
                int(result["families"]),
                float(result.get("censoring_share", CENSORING_SHARE)),
            )
        ],
    )


def worker_multiprocessing_context() -> Any:
    """Use spawn so workers receive only small cell jobs, never dense parent state."""
    return multiprocessing.get_context("spawn")


def collect_cell_results(
    jobs: list[CellJob],
    workers: int,
) -> list[dict[str, object]]:
    """Run every independent cell and restore deterministic command order."""
    if workers < 1 or not jobs:
        raise ValueError("COMPONENT_SEPARATION_WORKER_CONFIGURATION_INVALID")
    workers_used: int = min(workers, len(jobs))
    """Avoided starting processes that could own no complete design cell."""

    if workers_used == 1:
        results: list[dict[str, object]] = [run_cell(job) for job in jobs]
        """Used the identical cell boundary without process overhead."""
    else:
        with ProcessPoolExecutor(
            max_workers=workers_used,
            mp_context=worker_multiprocessing_context(),
        ) as pool:
            results = list(pool.map(run_cell, jobs, chunksize=1))
            """Spawned independent cell owners without serialising dense matrices."""
        """Closed every process before validating or reporting its results."""

    return validate_complete_cell_results(results, jobs)


def main(command_arguments: list[str] | None = None) -> int:
    """Sweep the design and report where the two components separate.

    Args:
        command_arguments: Arguments excluding the executable name, or ``None``
            to read the process command line.

    Returns:
        Zero when the unchanged consistency rules pass, otherwise one.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Created the parser for the sweep's extent."""

    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--families", type=int, nargs="+", default=FAMILIES)
    parser.add_argument("--records", type=int, nargs="+", default=RECORDS)
    parser.add_argument(
        "--censoring-shares",
        type=float,
        nargs="+",
        default=[CENSORING_SHARE],
        help="fixed expected right-censoring shares to sweep",
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--no-write", action="store_true")
    arguments: argparse.Namespace = parser.parse_args(command_arguments)
    """Read how far to sweep and how hard."""

    if (
        arguments.replicates < 1
        or arguments.workers < 1
        or any(families < 1 for families in arguments.families)
        or any(records < 1 for records in arguments.records)
        or any(share <= 0.0 or share >= 1.0 for share in arguments.censoring_shares)
    ):
        parser.error("replicates, workers, families and records must be positive")
    if len(set(arguments.families)) != len(arguments.families) or len(
        set(arguments.records)
    ) != len(arguments.records):
        parser.error("families and records must not contain duplicate cells")
    if len(set(arguments.censoring_shares)) != len(arguments.censoring_shares):
        parser.error("censoring shares must not contain duplicate cells")

    largest_families: int = max(arguments.families)
    """Fixed the design at which the unchanged bias rule is scored."""

    jobs: list[CellJob] = [
        CellJob(
            records=records,
            families=families,
            replicates=arguments.replicates,
            base_seed=arguments.seed,
            largest_families=largest_families,
            censoring_share=share,
        )
        for share in arguments.censoring_shares
        for records in arguments.records
        for families in arguments.families
    ]
    """Enumerated complete cells without building dense state in the parent."""

    cell_results: list[dict[str, object]] = collect_cell_results(
        jobs,
        arguments.workers,
    )
    """Ran every complete cell and restored the command's deterministic order."""

    report: dict[str, Any] = {}
    """Collected one record per design cell."""

    failures: list[str] = []
    """Collected every cell where an identified component came back biased."""

    print(
        f"{'records':>8} {'families':>9} {'additive':>18} {'person':>18} "
        f"{'corr':>7} {'sd(sum)':>8}  split costs"
    )
    print("-" * 88)
    for cell_result in cell_results:
        failures.extend(str(failure) for failure in cell_result["failures"])
        """Made every worker-reported scientific failure visible."""

        summary_value: object = cell_result["summary"]
        """Selected the existing evidence record, absent after too few fits."""

        if not isinstance(summary_value, dict):
            continue
        summary: dict[str, Any] = summary_value
        """Narrowed the validated complete summary for reporting."""

        records: int = int(summary["records_per_person"])
        """Read the first deterministic report coordinate."""

        families: int = int(summary["families"])
        """Read the second deterministic report coordinate."""

        share: float = float(summary.get("censoring_share", CENSORING_SHARE))
        """Read the expected censoring coordinate for this report row."""

        report[f"share={share}/records={records}/families={families}"] = summary
        """Kept this cell's unchanged evidence fields in command order."""

        additive: dict[str, Any] = summary["additive"]
        """Selected the additive display values."""

        person: dict[str, Any] = summary["person"]
        """Selected the person-level display values."""

        print(
            f"{records:>8} {families:>9} "
            f"{float(additive['mean']):>8.3f} ({float(additive['sd']):.3f}) "
            f"{float(person['mean']):>8.3f} ({float(person['sd']):.3f}) "
            f"{float(summary['correlation_between_estimates']):>7.3f} "
            f"{float(summary['sd_of_their_sum']):>8.3f}  "
            f"{float(summary['split_costs_this_many_times_the_precision']):>5.1f}x",
            flush=True,
        )

    receipt: dict[str, Any] = {
        "participant_free": True,
        "what": (
            "whether an additive component and a person-level one can be told "
            "apart, swept over records per person and related pairs"
        ),
        "date": date.today().isoformat(),
        "truths": {
            "additive": ADDITIVE,
            "person": PERSON,
            "residual": RESIDUAL,
        },
        "censoring_share": (
            CENSORING_SHARE
            if len(arguments.censoring_shares) == 1
            else arguments.censoring_shares
        ),
        "instrument_limit": (
            CENSORING_LIMIT
            if len(arguments.censoring_shares) == 1
            else [
                right_censoring_limit(
                    0.0,
                    ADDITIVE + PERSON + RESIDUAL,
                    share,
                )
                for share in arguments.censoring_shares
            ]
        ),
        "replicates_per_cell": arguments.replicates,
        "split_cost_threshold": SPLIT_COST,
        "cells": report,
        "passed": not failures,
        "failures": failures,
        "note": (
            "A design that does not separate the two is reported and not "
            "failed. The pass rule is that the estimator is correct where the "
            "components are identified; where they are not, that is a fact "
            "about the design."
        ),
    }
    """Built the complete evidence record for the sweep."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / f"component-separation-{receipt['date']}.json"
    )
    """Selected the evidence path used outside release mode."""
    if not arguments.no_write:
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"\nwritten to {out}")

    print("\nWhat the split costs, as a multiple of how well the sum is known:")
    for name, cell in report.items():
        cost: float = cell["split_costs_this_many_times_the_precision"]
        """Took how much worse each part is known than the whole."""

        limiting: str = (
            "  <- the split is what limits this design" if cost > SPLIT_COST else ""
        )
        """Marked the designs where the split, not the data, is the constraint."""

        print(f"  {name:>28}: {cost:>5.1f}x{limiting}")
    print(
        "\nThe correlation between the two estimates approaches minus one as a "
        "design grows.\nThat is the sum becoming certain while the split stays "
        "open, not a design getting worse."
    )

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"\nPASSED: the bias is gone by the largest design swept, within "
        f"{BIAS_AT_LARGEST} and this run's own Monte Carlo error. It is present "
        f"at the smaller ones, which is what a maximum-likelihood variance "
        f"component does and not a fault."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
