"""Level and power for Aim 2, on the design Sam settled on 18 August.

Reads `design.npz`, which carries a relationship matrix, an age and a role per
person, and a family unit per person -- and no identifiers.

- **Age sets risk.** Dementia prevalence rises steeply with age, so a person's
  rate comes from their age band rather than from one number for everybody.
  Every status is observed: a fifty-year-old with a clean CDR is a known
  non-case, and says so.
- **Family history is not a covariate.** The elevated risk of somebody with an
  affected parent is already in the relationship matrix. That is why enrolling
  the relatives matters and why no separate term is fitted for it.
- **Two measurement arms.** Probands either give a binary, misclassified ABR --
  which is what the protocol says for people too impaired for behavioural
  audiometry -- or an audiogram. Everyone else is measured properly in both.
  The pair brackets the severity question without needing to know the fraction.
- **Ascertainment is per family.** A Biggs family is drawn conditional on its
  proband being a case, which is what a clinic roster is. Everybody else is
  drawn from the population.

Level is measured under two nulls at the same replicate count as the
alternatives, and no power figure is read until it holds. Refusals are counted
and never dropped from a denominator.
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

type DesignData = dict[str, np.ndarray | list[list[int]]]
type FamilyRecord = dict[str, object]
type Job = tuple[str, str, int]
type SimulationResult = dict[str, str | int | float]
type ReportCell = dict[str, int]

DESIGN: str = os.environ.get("DESIGN", "design.npz")
"""Selected the identifier-free design archive for this campaign."""
OUT: str = os.environ.get("OUT", "campaign_result.json")
"""Selected the JSON path that will receive this campaign shard."""
REPLICATES: int = int(os.environ.get("REPLICATES", "200"))
"""Selected the number of simulation replicates per campaign cell."""
WORKERS: int = int(os.environ.get("WORKERS", "120"))
"""Selected the number of worker processes used for independent fits."""
BOOTSTRAP: int = int(os.environ.get("BOOTSTRAP", "50"))
"""Selected the reference draws used by each vertical-path test."""
# Nought selects sequential truncation, which is what SOLAR and the liability
# model use. It is an approximation -- agreeing with the accurate route to
# about 0.03 in log probability per family at pedigree correlations -- and it
# is what lets this run on the real families rather than on pairs.
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
AUDIOMETRY_ERROR: float = 0.15
"""Set measurement-error variance for a directly measured audiogram."""
# Placeholders until the ABR-audiometry bridge supplies real ones.
ABR_SENSITIVITY: float = 0.80
"""Set the provisional sensitivity of binary ABR classification."""
ABR_SPECIFICITY: float = 0.85
"""Set the provisional specificity of binary ABR classification."""


# Probable AD dementia by age. Roughly a doubling every five years past 65,
# which is the shape every prevalence study finds.
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
    "null: no loading": {"a": 0.0, "b": 0.4},
    "null: no path": {"a": 0.6, "b": 0.0},
    "vertical 0.06": {"a": 0.6, "b": 0.1},
    "vertical 0.12": {"a": 0.6, "b": 0.2},
    "vertical 0.18": {"a": 0.6, "b": 0.3},
    "vertical 0.24": {"a": 0.6, "b": 0.4},
}
"""Specified the two nulls and four mediated-effect alternatives."""
ARMS: list[str] = ["ABR only", "audiogram"]
"""Specified the two protocol-compatible mediator measurement arms."""

_D: DesignData = {}
"""Cached the identifier-free design after its first process-local load."""


def design() -> DesignData:
    """Load and cache the identifier-free campaign design.

    Returns:
        Relationship, age, role, family-unit and prevalence arrays.
    """
    if not _D:
        loaded: np.lib.npyio.NpzFile = np.load(DESIGN)
        """Loaded the arrays stored in the identifier-free design archive."""
        units: collections.defaultdict[int, list[int]] = collections.defaultdict(list)
        """Initialised row indices grouped by simulated family unit."""
        for row, unit in enumerate(loaded["family"]):
            units[int(unit)].append(row)
        """Grouped every design row by its family-unit code."""
        _D.update(
            relationship=loaded["relationship"],
            age=loaded["age"],
            role=loaded["role"],
            units=[units[k] for k in sorted(units)],
            prevalence=np.array([prevalence_for(a) for a in loaded["age"]]),
        )
        """Cached the arrays and derived age-specific prevalences used by all fits."""
    return _D


def one(job: Job) -> SimulationResult:
    """Run one deterministic simulation-and-test replicate.

    Args:
        job: Measurement arm, truth-cell label and replicate index.

    Returns:
        Counts, test results, refusals and elapsed time for the replicate.
    """
    arm, truth, replicate = job
    """Unpacked the campaign coordinates for this replicate."""
    design_data: DesignData = design()
    """Loaded the shared identifier-free design for this worker process."""
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
        """Located the named proband within a Biggs unit when one was present."""

        if proband is not None and arm == "ABR only":
            # No audiogram for the person with dementia; a fallible binary
            # result instead. Everyone else is measured properly.
            error: list[float | None] = [
                None if index == proband else AUDIOMETRY_ERROR for index in range(size)
            ]
            """Removed direct audiometry error for the proband receiving binary ABR."""
            proxy: list[bool] = [index == proband for index in range(size)]
            """Marked only the proband's mediator observation as a binary proxy."""
        else:
            error = [AUDIOMETRY_ERROR] * size
            """Applied direct audiometry error to every person in this family."""
            proxy = [False] * size
            """Marked every mediator observation as directly measured."""

        families.extend(
            simulate(
                relationship=matrix,
                **TRUTHS[truth],
                **FIXED,
                families=1,
                seed=300_000 + 7_919 * replicate + 13 * unit_index,
                outcome_prevalence=[
                    float(prevalence) for prevalence in design_data["prevalence"][rows]
                ],
                observe_outcome=[True] * size,
                measurement_error_variance=error,
                observe_mediator_proxy=proxy,
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
    """Simulated every family under the requested truth and measurement arm."""

    out: SimulationResult = {"arm": arm, "truth": truth, "replicate": replicate}
    """Initialised the serialisable record for this simulation replicate."""
    out["cases"] = sum(1 for f in families for v in f["outcome_status"] if v)
    """Counted simulated cases without dropping any refused model fit."""
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        families, qmc_points=QMC_POINTS
    )
    """Prepared the latent-mediation model for the simulated families."""
    started: float = time.time()
    """Captured the wall-clock start time for both hypothesis tests."""
    try:
        out["vertical"] = float(
            model.test_vertical(bootstrap_replicates=BOOTSTRAP)["p_value"]
        )
        """Recorded the vertical-path p-value when the test produced a result."""
    except ValueError as refusal:
        out["vertical_refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
        """Recorded the stable vertical-test refusal without changing the denominator."""
    try:
        out["horizontal"] = float(model.test_horizontal()["p_value"])
        """Recorded the horizontal-path p-value when the test produced a result."""
    except ValueError as refusal:
        out["horizontal_refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
        """Recorded the stable horizontal-test refusal without changing the denominator."""
    out["seconds"] = time.time() - started
    """Recorded elapsed time for the two hypothesis tests."""
    return out


def main() -> int:
    """Run the selected Aim 2 campaign cells and write their count summary.

    Returns:
        Process exit status, zero after the result file is written.
    """
    design_data: DesignData = design()
    """Loaded the identifier-free campaign design."""
    n: int = len(design_data["age"])
    """Counted the people included in the campaign design."""
    print(
        f"{n} people in {len(design_data['units'])} units, "
        f"{int((design_data['age'] >= 65).sum())} of them over 65"
    )
    print(
        f"{REPLICATES} replicates, {len(ARMS)} arms, {len(TRUTHS)} truths, "
        f"{WORKERS} workers, {BOOTSTRAP} reference draws",
        flush=True,
    )
    # One array task per cell, so the grid spreads across nodes rather than
    # queueing behind itself on one. CELL is the task id; unset means all.
    cells: list[tuple[str, str]] = [(arm, truth) for arm in ARMS for truth in TRUTHS]
    """Enumerated every measurement-arm and truth combination."""
    chosen: str | None = os.environ.get("CELL")
    """Read the optional array-task index selecting a single campaign cell."""
    if chosen is not None:
        cells = [cells[int(chosen)]]
        """Restricted this process to the selected array-task cell."""
        print(f"cell {chosen}: {cells[0][0]} / {cells[0][1]}", flush=True)
    jobs: list[Job] = [
        (arm, truth, replicate)
        for arm, truth in cells
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

    print(
        f"\n{'arm':<11}|{'truth':<17}|{'cases':>6}|{'refused V':>10}|"
        f"{'rej V .025':>11}|{'refused H':>10}|{'rej H .025':>11}"
    )
    print("-" * 82)
    report: dict[str, ReportCell] = {}
    """Initialised the count summary for each measurement-arm and truth cell."""
    for arm in ARMS:
        for truth in TRUTHS:
            here: list[SimulationResult] = [
                result
                for result in rows
                if result["arm"] == arm and result["truth"] == truth
            ]
            """Selected all completed replicates for this exact campaign cell."""
            vertical_values: list[float] = [
                float(result["vertical"]) for result in here if "vertical" in result
            ]
            """Collected vertical-test p-values from non-refused replicates."""
            horizontal_values: list[float] = [
                float(result["horizontal"]) for result in here if "horizontal" in result
            ]
            """Collected horizontal-test p-values from non-refused replicates."""
            vertical_refused: int = len(here) - len(vertical_values)
            """Counted replicates whose vertical test produced a refusal."""
            horizontal_refused: int = len(here) - len(horizontal_values)
            """Counted replicates whose horizontal test produced a refusal."""
            vertical_rejected: int = sum(
                1 for p_value in vertical_values if p_value < 0.025
            )
            """Counted vertical rejections at the prespecified 0.025 threshold."""
            horizontal_rejected: int = sum(
                1 for p_value in horizontal_values if p_value < 0.025
            )
            """Counted horizontal rejections at the prespecified 0.025 threshold."""
            cases: int = (
                int(np.mean([float(result["cases"]) for result in here])) if here else 0
            )
            """Calculated the mean simulated case count for the campaign cell."""
            print(
                f"{arm:<11}|{truth:<17}|{cases:>6}|"
                f"{vertical_refused:>4}/{len(here):<5}|"
                f"{vertical_rejected:>4}/{len(here):<6}|"
                f"{horizontal_refused:>4}/{len(here):<5}|"
                f"{horizontal_rejected:>4}/{len(here):<6}"
            )
            report[f"{arm} | {truth}"] = {
                "replicates": len(here),
                "mean_cases": cases,
                "vertical_refused": vertical_refused,
                "vertical_reject_025": vertical_rejected,
                "horizontal_refused": horizontal_refused,
                "horizontal_reject_025": horizontal_rejected,
            }
            """Stored the full denominator, refusals and rejection counts."""
    """Summarised every selected campaign cell without dropping refusals."""
    with open(OUT, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "design": DESIGN,
                "replicates": REPLICATES,
                "bootstrap": BOOTSTRAP,
                "cells": report,
            },
            stream,
            indent=2,
        )
    """Wrote the campaign coordinates and count summary as indented JSON."""
    print(f"\nwritten to {OUT}")
    print(
        "\nRejection under the two nulls is the LEVEL and must sit at or "
        "below .025.\nRejection under a vertical truth is the POWER."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
