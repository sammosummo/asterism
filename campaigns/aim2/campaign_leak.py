"""Does pleiotropy leak into the mediation estimate, on the real design?

The genetic covariance between hearing and dementia is `a (ab + c')` -- it sees
only the sum, so genetics alone cannot separate mediation from pleiotropy. What
separates them is the residual, within-person covariance `b sigma_m2`. So `b`,
and with it the mediation estimand, is identified from the non-genetic
association between hearing and dementia, and `c'` comes out as the difference
between a well-estimated total and a weakly-estimated product.

That raises a question the power work never asked: if the separation is weak,
does some of the direct path get read as mediation? An estimate that climbs with
pleiotropy would attack the primary claim, which matters far more than the
secondary test being underpowered.

A first attempt on 150 families of three was too noisy to answer -- the means
climbed, but no row reached two standard errors and the estimates were
heavy-tailed enough that a mean was the wrong summary. This runs it on the real
design, reports the median beside the mean, and counts how often the fit lands
somewhere absurd, because a handful of runaway fits is a different problem from
a systematic leak and the two look identical in an average.
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
type Job = tuple[float, int]
type SimulationResult = dict[str, str | int | float]
type ReportCell = dict[str, int | float]

DESIGN: str = os.environ.get("DESIGN", "design.npz")
"""Selected the identifier-free design archive for the leak campaign."""
OUT: str = os.environ.get("OUT", "leak_result.json")
"""Selected the JSON path that will receive the leak summary."""
REPLICATES: int = int(os.environ.get("REPLICATES", "200"))
"""Selected the number of replicates at each direct-path value."""
WORKERS: int = int(os.environ.get("WORKERS", "120"))
"""Selected the number of worker processes used for independent fits."""
QMC_POINTS: int = int(os.environ.get("QMC_POINTS", "0"))
"""Selected sequential integration or the requested quasi-Monte Carlo size."""

PROBAND: int = 0
"""Encoded the Biggs proband role in the identifier-free design."""
TRUE_VERTICAL: float = 0.6 * 0.4
"""Calculated the mediated effect held constant throughout the campaign."""
DIRECT: list[float] = [0.0, 0.2, 0.4, 0.7]
"""Selected direct paths spanning no pleiotropy through a large effect."""
AUDIOMETRY_ERROR: float = 0.15
"""Set measurement-error variance for every observed audiogram."""


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


_D: DesignData = {}
"""Cached the identifier-free design after its first process-local load."""


def design() -> DesignData:
    """Load and cache the identifier-free leak-campaign design.

    Returns:
        Relationship, role, family-unit and prevalence arrays.
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
            role=loaded["role"],
            units=[units[k] for k in sorted(units)],
            prevalence=np.array([prevalence_for(a) for a in loaded["age"]]),
        )
        """Cached the arrays and derived age-specific prevalences used by all fits."""
    return _D


def one(job: Job) -> SimulationResult:
    """Run one deterministic pleiotropy-leak replicate.

    Args:
        job: Direct-path value and replicate index.

    Returns:
        Fitted estimands or a stable refusal for the replicate.
    """
    c_prime, replicate = job
    """Unpacked the direct-path value and deterministic replicate index."""
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
        families.extend(
            simulate(
                relationship=matrix,
                a=0.6,
                b=0.4,
                c_prime=c_prime,
                d=0.7,
                sigma_m2=0.5,
                families=1,
                seed=2_000_000 + 7919 * replicate + 13 * unit_index,
                outcome_prevalence=[
                    float(prevalence) for prevalence in design_data["prevalence"][rows]
                ],
                observe_outcome=[True] * size,
                measurement_error_variance=[AUDIOMETRY_ERROR] * size,
                observe_mediator_proxy=[False] * size,
                ascertainment=(
                    "condition_on_named_proband_case"
                    if proband is not None
                    else "population_unconditioned"
                ),
                proband_index=proband,
            )
        )
    """Simulated every family while holding the mediated effect fixed."""
    out: SimulationResult = {"c_prime": c_prime, "replicate": replicate}
    """Initialised the serialisable record for this leak replicate."""
    try:
        fit: dict[str, object] = asterism.LatentMediationModel(
            families, qmc_points=QMC_POINTS
        ).fit()
        """Fitted the latent-mediation model to the simulated families."""
        out["vertical"] = float(fit["estimands"]["theta_vertical"])
        """Recorded the fitted mediated effect."""
        out["horizontal"] = float(fit["estimands"]["theta_horizontal"])
        """Recorded the fitted direct effect."""
    except ValueError as refusal:
        out["refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
        """Recorded the stable refusal without removing the replicate."""
    return out


def main() -> int:
    """Run the pleiotropy-leak campaign and write robust summaries.

    Returns:
        Process exit status, zero after the result file is written.
    """
    design_data: DesignData = design()
    """Loaded the identifier-free leak-campaign design."""
    print(
        f"{sum(len(rows) for rows in design_data['units'])} people in "
        f"{len(design_data['units'])} units, "
        f"{REPLICATES} replicates per value",
        flush=True,
    )
    jobs: list[Job] = [
        (c_prime, replicate) for c_prime in DIRECT for replicate in range(REPLICATES)
    ]
    """Expanded every direct-path value into deterministic replicate jobs."""
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
                print(f"  {len(rows)}/{len(jobs)}  {rate:.1f}s/fit", flush=True)
    """Executed every replicate and reported periodic progress."""

    print(f"\ntrue mediated effect {TRUE_VERTICAL:.2f} throughout\n")
    print(
        f"{'true c*':>8}{'median':>9}{'mean':>9}{'bias/se':>9}"
        f"{'|est|>1':>9}{'refused':>9}{'median c*':>11}"
    )
    print("-" * 64)
    report: dict[str, ReportCell] = {}
    """Initialised robust fit summaries for each direct-path value."""
    for c_prime in DIRECT:
        here: list[SimulationResult] = [
            result for result in rows if result["c_prime"] == c_prime
        ]
        """Selected all completed replicates at this direct-path value."""
        values: np.ndarray = np.array(
            [float(result["vertical"]) for result in here if "vertical" in result]
        )
        """Collected fitted mediated effects from non-refused replicates."""
        direct: np.ndarray = np.array(
            [float(result["horizontal"]) for result in here if "horizontal" in result]
        )
        """Collected fitted direct effects from non-refused replicates."""
        refused: int = len(here) - len(values)
        """Counted replicates whose full fit produced a refusal."""
        if len(values) < 10:
            print(f"{c_prime:>8.1f}   too few fits")
            continue
        median: float = float(np.median(values))
        """Calculated the robust centre of the fitted mediated effects."""
        mean: float = float(values.mean())
        """Calculated the mean fitted mediated effect, including heavy tails."""
        error: float = float(values.std(ddof=1) / np.sqrt(len(values)))
        """Estimated the Monte Carlo standard error of the mean."""
        runaway: int = int((np.abs(values) > 1.0).sum())
        """Counted fitted mediated effects with implausible magnitude above one."""
        print(
            f"{c_prime:>8.1f}{median:>9.4f}{mean:>9.4f}"
            f"{(mean - TRUE_VERTICAL) / error:>+9.2f}{runaway:>9}"
            f"{refused:>9}{float(np.median(direct)):>11.4f}"
        )
        report[str(c_prime)] = {
            "median_vertical": median,
            "mean_vertical": mean,
            "standard_error": error,
            "runaway": runaway,
            "refused": refused,
            "median_horizontal": float(np.median(direct)),
            "fitted": len(values),
        }
        """Stored robust centre, tail, refusal and direct-effect summaries."""
    """Summarised every direct-path value without dropping refused fits."""

    with open(OUT, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "design": DESIGN,
                "replicates": REPLICATES,
                "true_vertical": TRUE_VERTICAL,
                "cells": report,
            },
            stream,
            indent=2,
        )
    """Wrote the campaign coordinates and robust summaries as indented JSON."""
    print(f"\nwritten to {OUT}")
    print(
        "\nA median that climbs with the direct path is a leak. A mean that "
        "climbs while\nthe median does not is a few runaway fits, which is a "
        "different fault and is\nfixed differently."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
