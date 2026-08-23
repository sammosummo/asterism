"""Which design change would actually buy power?

The campaign says the design reaches 55 per cent at its most favourable effect
and needs about 1.7 times the sample for 80. "Get more people" is a true answer
and an expensive one, and it is only the right answer if people are what is
short.

There is a reason to think they are not. The mediated estimand is a product of
two halves that are informed very differently. The loading on hearing is
estimated from resemblance among relatives on a continuous trait measured in
everybody, and ought to be well determined. The path from hearing to dementia
is estimated from the genetic covariance between hearing and a *binary* outcome
at about a sixth prevalence, and in a liability model the information about that
covariance scales with the number of cases and with the relative pairs that
contain one. If the second half is the binding constraint, then adding
unaffected people buys very little and adding cases buys a great deal.

Counting the design settled which levers are worth pulling. Of about 257 cases,
200 are the adjudicated probands; the other 1,400 people contribute roughly 57
between them. So the levers follow the budget rather than the head count --
adjudication is what the money buys, and an adjudicated proband arrives as a
certain case with relatives attached.

An earlier version scaled population prevalence instead, and it is worth saying
why that was wrong: it moved only the 57, raising cases by nine per cent where
its name promised fifty. It would have been reported as "more cases barely
helps" when what it measured was "more of the minority barely helps".

Each lever changes exactly one thing from the real design, and they are
deliberately not combined -- a grid of combinations would be cheaper to justify
and impossible to attribute.

    baseline              the design as it stands
    probands x1.5, x2     more adjudicated families, which is the real spend
    hearing precise       measurement error 0.15 -> 0.05, sharpening the
                          loading; should do almost nothing if the loading is
                          not what is short, which is why it is here
    drop acoustic units   the 185 people who sit in units of Acoustic
                          participants only, removed. Not the whole Acoustic
                          contribution: most of those 600 share a unit with an
                          enrolled relative and cannot be dropped without
                          taking the relative too
    sample x1.5           half of everything again, the naive "recruit more"

The level is measured for every lever, under the identical change, because a
lever that raised power by raising the rejection rate generally would not be a
lever.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import asterism
import numpy as np
from asterism.latent_mediation import simulate

type DesignData = dict[str, np.ndarray | list[list[int]] | list[list[float]] | float]
type FamilyRecord = dict[str, object]
type Job = tuple[str, str, int]
type SimulationResult = dict[str, str | int | float]
type ReportCell = dict[str, int]

DESIGN: str = os.environ.get("DESIGN", "design.npz")
"""Selected the identifier-free design archive for the lever campaign."""
OUT: str = os.environ.get("OUT", "levers_result.json")
"""Selected the JSON path that will receive this campaign shard."""
REPLICATES: int = int(os.environ.get("REPLICATES", "200"))
"""Selected the number of simulation replicates per lever and truth cell."""
WORKERS: int = int(os.environ.get("WORKERS", "120"))
"""Selected the number of worker processes used for independent fits."""
BOOTSTRAP: int = int(os.environ.get("BOOTSTRAP", "50"))
"""Selected the reference draws used by each vertical-path test."""
QMC_POINTS: int = int(os.environ.get("QMC_POINTS", "0"))
"""Selected sequential integration or the requested quasi-Monte Carlo size."""

PROBAND: int = 0
"""Encoded the Biggs proband role in the identifier-free design."""
OFFSPRING: int = 1
"""Encoded the Biggs offspring role in the identifier-free design."""
ACOUSTIC: int = 2
"""Encoded the Acoustic participant role in the identifier-free design."""
RELATIVE: int = 3
"""Encoded the recruited-relative role in the identifier-free design."""
FIXED: dict[str, float] = {"c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}
"""Fixed nuisance paths and mediator residual variance across all cells."""
BASE_ERROR: float = 0.15
"""Set baseline measurement-error variance for directly measured audiograms."""
ABR_SENSITIVITY: float = 0.80
"""Retained provisional ABR sensitivity in the serialised campaign contract."""
ABR_SPECIFICITY: float = 0.85
"""Retained provisional ABR specificity in the serialised campaign contract."""
ARM: str = "audiogram"  # the question here is design, not measurement
"""Fixed the better measurement arm so each cell isolates a design change."""


def prevalence_for(age: float) -> float:
    """Return the design prevalence assigned to an age.

    Args:
        age: Participant age in years.

    Returns:
        Probable Alzheimer disease dementia prevalence for the age band.
    """
    if age < 60:
        return 0.001
    if age < 65:
        return 0.005
    if age < 70:
        return 0.02
    if age < 75:
        return 0.04
    if age < 80:
        return 0.08
    if age < 85:
        return 0.15
    return 0.25


TRUTHS: dict[str, dict[str, float]] = {
    "null: no path": {"a": 0.6, "b": 0.0},
    "vertical 0.24": {"a": 0.6, "b": 0.4},
}
"""Specified the matched vertical-path null and power conditions."""
LEVERS: list[str] = [
    "baseline",
    # Adjudication is what the budget buys, and it arrives as a certain case
    # with relatives attached. These two are the real "spend more" options.
    "probands x1.5",
    "probands x2",
    # Sharpens the loading on hearing. Should do almost nothing if the loading
    # is not what is short, which is the point of including it.
    "hearing precise",
    # What are the Acoustic-only units contributing? 185 people who carry
    # hearing, almost never become a case, and share a unit with nobody.
    "drop acoustic units",
    # The naive version of "recruit more": half of everything again.
    "sample x1.5",
    # The same people and the same dementia rate, graded rather than diagnosed.
    "staged outcome",
]
"""Enumerated the one-factor design changes compared with the baseline."""

# CDR 1, 2 and 3 as shares of the dementia rate, and a questionable band the
# same size as that rate. Chosen so "CDR 1 or worse" is exactly the binary rate
# the other levers use, leaving the grading as the only difference.
SEVERITY: tuple[float, float, float] = (0.56, 0.31, 0.13)
"""Allocated dementia cases across CDR 1, 2 and 3 categories."""

_D: dict[str, DesignData] = {}
"""Cached each lever-specific design after its first process-local build."""


def design(lever: str) -> DesignData:
    """Build and cache a design with exactly one lever applied.

    Args:
        lever: Named design change from the prespecified lever list.

    Returns:
        Arrays, family units, staging probabilities and measurement error.
    """
    key: str = lever
    """Used the lever label as the cache key for the derived design."""
    if key in _D:
        return _D[key]

    loaded: np.lib.npyio.NpzFile = np.load(DESIGN)
    """Loaded the arrays stored in the identifier-free baseline design."""
    units: collections.defaultdict[int, list[int]] = collections.defaultdict(list)
    """Initialised row indices grouped by simulated family unit."""
    for row, unit in enumerate(loaded["family"]):
        units[int(unit)].append(row)
    """Grouped every design row by its family-unit code."""
    ordered: list[list[int]] = [units[unit] for unit in sorted(units)]
    """Ordered the baseline family units by their stable numeric code."""
    age: np.ndarray = loaded["age"]
    """Selected the participant ages aligned to the design rows."""
    prevalence: np.ndarray = np.array([prevalence_for(value) for value in age])
    """Derived age-specific dementia prevalence for every design row."""
    error: float = BASE_ERROR
    """Initialised mediator measurement error at the baseline audiogram value."""

    if lever.startswith("probands x"):
        factor: float = float(lever.split("x")[1])
        """Parsed the requested multiplier for ascertained proband families."""
        # Duplicate the ascertained units. Each duplicate is an independent
        # family with the same structure, which is what enrolling another
        # proband of the same shape would give, and families are independent in
        # the likelihood so reusing the rows is sound.
        ascertained: list[list[int]] = [
            rows for rows in ordered if (loaded["role"][rows] == PROBAND).any()
        ]
        """Selected family units containing an adjudicated proband."""
        wanted: int = round(len(ascertained) * (factor - 1.0))
        """Calculated how many independent proband-shaped units to add."""
        ordered = ordered + ascertained[:wanted]
        """Added the requested number of ascertained family structures."""
    elif lever == "hearing precise":
        error = 0.05
        """Reduced only audiogram measurement-error variance."""
    elif lever == "drop acoustic units":
        # Units made up entirely of Acoustic participants -- no proband, no
        # enrolled relative. 185 people, not the whole Acoustic cohort: the
        # rest share a unit with an enrolled relative.
        ordered = [
            rows for rows in ordered if not (loaded["role"][rows] == ACOUSTIC).all()
        ]
        """Removed units composed entirely of unlinked Acoustic participants."""
    elif lever == "staged outcome":
        pass  # handled below, where the shares need the prevalence
    elif lever == "sample x1.5":
        # Every second unit, not the first half. The units are grouped by type,
        # so the first half holds no ascertained families at all -- taking it
        # would have added a thousand people and not one adjudicated case, and
        # then reported that recruiting more does very little.
        ordered = ordered + ordered[::2]
        """Added every second unit to preserve the baseline mixture of unit types."""

    # Stages per person, built from their own dementia rate so the grading is
    # age-indexed the same way the rate is.
    staged: list[list[float]] = []
    """Initialised optional per-person ordinal outcome probabilities."""
    if lever == "staged outcome":
        for rate in prevalence:
            questionable: float = float(rate)
            """Matched the questionable-band prevalence to dementia prevalence."""
            shares: list[float] = [
                1.0 - float(rate) - questionable,
                questionable,
            ]
            """Initialised CDR 0 and 0.5 probabilities while holding cases fixed."""
            shares.extend(float(rate) * part for part in SEVERITY)
            """Allocated dementia prevalence across CDR 1, 2 and 3."""
            staged.append(shares)
            """Stored ordinal probabilities for this participant's age-specific rate."""
    """Built ordinal probabilities only for the staged-outcome lever."""

    _D[key] = {
        "staged": staged,
        "relationship": loaded["relationship"],
        "age": age,
        "role": loaded["role"],
        "units": ordered,
        "prevalence": prevalence,
        "error": error,
    }
    """Cached the complete lever-specific design for reuse by worker fits."""
    return _D[key]


def one(job: Job) -> SimulationResult:
    """Run one deterministic lever-and-truth replicate.

    Args:
        job: Lever label, truth-cell label and replicate index.

    Returns:
        Counts, test result or refusal for the replicate.
    """
    lever, truth, replicate = job
    """Unpacked the lever, truth condition and deterministic replicate index."""
    design_data: DesignData = design(lever)
    """Loaded the one-factor design change for this worker process."""
    families: list[FamilyRecord] = []
    """Initialised the simulated family records supplied to the model."""
    for unit_index, rows in enumerate(design_data["units"]):
        size: int = len(rows)
        """Counted the people in this simulated family unit."""
        roles: np.ndarray = design_data["role"][rows]
        """Selected the role codes aligned to the family rows."""
        matrix: list[list[float]] = [
            [float(value) for value in row]
            for row in design_data["relationship"][np.ix_(rows, rows)]
        ]
        """Extracted the family's relationship submatrix as ordinary floats."""
        proband: int | None = (
            int(np.argmax(roles == PROBAND)) if (roles == PROBAND).any() else None
        )
        """Located the named proband within an ascertained unit when present."""

        families.extend(
            simulate(
                relationship=matrix,
                **TRUTHS[truth],
                **FIXED,
                families=1,
                seed=800_000 + 7_919 * replicate + 13 * unit_index,
                # None rather than an empty list: the wrapper expands a scalar or
                # None per person, and a zero-length sequence fails its shape check.
                outcome_prevalence=(
                    None
                    if design_data["staged"]
                    else [
                        float(prevalence)
                        for prevalence in design_data["prevalence"][rows]
                    ]
                ),
                outcome_category_prevalence=(
                    [design_data["staged"][row] for row in rows]
                    if design_data["staged"]
                    else None
                ),
                # CDR 1 or worse is a case, which is category two of five and the
                # same roster the binary arm conditions on.
                ascertainment_category=2 if design_data["staged"] else None,
                observe_outcome=[True] * size,
                measurement_error_variance=[design_data["error"]] * size,
                observe_mediator_proxy=[False] * size,
                sensitivity=ABR_SENSITIVITY,
                specificity=ABR_SPECIFICITY,
                ascertainment=(
                    "condition_on_named_proband_case"
                    if proband is not None
                    else "population_unconditioned"
                ),
                proband_index=proband,
            )
        )
    """Simulated every family under the selected design lever and truth."""

    out: SimulationResult = {
        "lever": lever,
        "truth": truth,
        "replicate": replicate,
    }
    """Initialised the serialisable record for this simulation replicate."""
    case_from: int = 2 if design_data["staged"] else 1
    """Selected the first outcome category counted as a dementia case."""
    out["cases"] = sum(
        1
        for f in families
        for v in f["outcome_status"]
        if v is not None and v >= case_from
    )
    """Counted cases consistently across binary and staged outcomes."""
    out["people"] = sum(len(f["outcome_status"]) for f in families)
    """Counted all simulated people without dropping refused fits."""
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        families, qmc_points=QMC_POINTS
    )
    """Prepared the latent-mediation model for the simulated families."""
    try:
        out["vertical"] = float(
            model.test_vertical(bootstrap_replicates=BOOTSTRAP)["p_value"]
        )
        """Recorded the vertical-path p-value when the test produced a result."""
    except ValueError as refusal:
        out["vertical_refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
        """Recorded the stable refusal without removing the replicate."""
    return out


def main() -> int:
    """Run the selected lever cells and write their count summary.

    Returns:
        Process exit status, zero after the result file is written.
    """
    cells: list[tuple[str, str]] = [
        (lever, truth) for lever in LEVERS for truth in TRUTHS
    ]
    """Enumerated every one-factor design change and matched truth condition."""
    chosen: str | None = os.environ.get("CELL")
    """Read the optional array-task index selecting a single campaign cell."""
    if chosen is not None:
        cells = [cells[int(chosen)]]
        """Restricted this process to the selected array-task cell."""
        print(f"cell {chosen}: {cells[0][0]} / {cells[0][1]}", flush=True)
    for lever, _ in cells:
        design_data: DesignData = design(lever)
        """Loaded this lever's derived design for its startup summary."""
        people: int = sum(len(rows) for rows in design_data["units"])
        """Counted people after applying this one design change."""
        print(
            f"{lever}: {len(design_data['units'])} units, {people} people, "
            f"expected cases "
            f"{design_data['prevalence'][np.concatenate(design_data['units'])].sum():.0f}",
            flush=True,
        )
    """Displayed the size and expected case count of each selected design."""

    jobs: list[Job] = [
        (lever, truth, replicate)
        for lever, truth in cells
        for replicate in range(REPLICATES)
    ]
    """Expanded every selected cell into independent replicate jobs."""
    print(f"{len(jobs)} fits queued\n", flush=True)

    rows: list[SimulationResult] = []
    """Initialised the completed replicate records."""
    started: float = time.time()
    """Captured the wall-clock start time for progress estimates."""
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=1):
            rows.append(got)
            """Retained each completed replicate, including scientific refusals."""
            if len(rows) % 50 == 0:
                rate: float = (time.time() - started) / len(rows)
                """Estimated mean elapsed time per completed fit."""
                print(
                    f"  {len(rows)}/{len(jobs)}  {rate:.1f}s/fit  "
                    f"eta {(len(jobs) - len(rows)) * rate / 60:.0f} min",
                    flush=True,
                )
    """Executed every selected replicate and reported periodic progress."""

    report: dict[str, ReportCell] = {}
    """Initialised the count summary for every lever and truth cell."""
    print(
        f"\n{'lever':<18}|{'truth':<15}|{'people':>7}|{'cases':>6}|"
        f"{'refused':>8}|{'reject .025':>12}"
    )
    print("-" * 72)
    for lever in LEVERS:
        for truth in TRUTHS:
            here: list[SimulationResult] = [
                result
                for result in rows
                if result["lever"] == lever and result["truth"] == truth
            ]
            """Selected all completed replicates for this exact campaign cell."""
            if not here:
                continue
            values: list[float] = [
                float(result["vertical"]) for result in here if "vertical" in result
            ]
            """Collected vertical-test p-values from non-refused replicates."""
            refused: int = len(here) - len(values)
            """Counted replicates whose vertical test produced a refusal."""
            rejected: int = sum(1 for p_value in values if p_value < 0.025)
            """Counted rejections at the prespecified 0.025 threshold."""
            cases: int = int(np.mean([float(result["cases"]) for result in here]))
            """Calculated the mean simulated case count for the campaign cell."""
            people: int = int(np.mean([float(result["people"]) for result in here]))
            """Calculated the mean simulated sample size for the campaign cell."""
            print(
                f"{lever:<18}|{truth:<15}|{people:>7}|{cases:>6}|"
                f"{refused:>4}/{len(here):<3}|{rejected:>5}/{len(here):<6}"
            )
            report[f"{lever} | {truth}"] = {
                "replicates": len(here),
                "people": people,
                "mean_cases": cases,
                "vertical_refused": refused,
                "vertical_reject_025": rejected,
            }
            """Stored the full denominator, refusal and rejection counts."""
    """Summarised every completed campaign cell without dropping refusals."""

    with open(OUT, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "design": DESIGN,
                "replicates": REPLICATES,
                "arm": ARM,
                "bootstrap": BOOTSTRAP,
                "cells": report,
            },
            stream,
            indent=2,
        )
    """Wrote the campaign coordinates and count summary as indented JSON."""
    print(f"\nwritten to {OUT}")
    print(
        "\nRead each lever against the baseline, and each power row against "
        "the level row\nabove it. A lever that lifted both has not bought "
        "power, it has bought rejections."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
