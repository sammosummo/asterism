"""Do the kinship-class tests hold their level?

The class-weighted kinship model splits the direct parent-offspring cells of the
relationship matrix into four classes and asks whether they carry the same
variance. On the GOBS traits it says they do not for both amyloid biomarkers,
with mother-son unusually low and mother-daughter unusually high in each. That is
the most interesting thing the session found and it rests on two tests nobody
has checked.

**Two tests, and they are not the same question.** The omnibus pools every class
back together and asks whether splitting bought anything, on `k - 1` degrees of
freedom. Each contrast pools one class against everything else, on one. The
omnibus is what protects the contrasts from being read as four independent
findings, so if the omnibus does not hold its level nothing downstream does.

**This calibrates on the real pedigree and the real class split.** The classes
are badly unbalanced in GOBS -- far more measured mothers than fathers -- and a
balanced simulation would say nothing about the class that rests on the fewest
pairs and is most likely to mislead.

**The criterion is validity, not uniformity.** Every class variance is bounded
below at nought, so under the null a share of fits sit on a bound and the
statistic has an atom. What must hold is that the rejection rate does not exceed
its level.

Run with:

    ASTERISM_REPLICATES=200 uv run --no-project python checks/kinship_equality_calibration.py

It reads the GOBS pedigree, which is study material, and simulates every
response it uses. No observed phenotype is read.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import asterism
import numpy as np

DATABASE: Path = Path(
    "~/MathiasLab/staging/studies/existing/safs/data/SAFS.db"
).expanduser()
"""Resolved the read-only study database used for pedigree structure."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "200"))
"""Selected the requested replicate count from the environment."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "6"))
"""Selected the worker-process count from the environment."""

LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)
"""Fixed the nominal rejection levels checked by the calibration."""
# The share of variance the additive component carries under the null, near
# what the real traits gave.
ADDITIVE: float = 0.5
"""Set the additive variance share under the equality null."""
# Under the alternative, one class carries this much more. Mother-daughter is
# the class the amyloids put high, so it is the one moved.
LIFTED: str = "mother_daughter"
"""Selected the parent-offspring class perturbed under the alternative."""

LIFT: float = 0.5
"""Set the extra component coefficient under the powered alternative."""

STATE: dict[str, object] = {}
"""Held read-only simulation inputs initialised once in each worker."""


def start_worker(payload: dict[str, object]) -> None:
    """Initialise one worker process with shared simulation inputs.

    Args:
        payload: Relationship matrices, design, names and covariance factors.
    """
    STATE.update(payload)


def structure() -> tuple[list[np.ndarray], list[str], dict[str, int], np.ndarray]:
    """Build the GOBS pedigree design split by parent-offspring class.

    Returns:
        Class relationship matrices, class names, pair counts and fixed design.
    """
    db: sqlite3.Connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    """Opened the study database in SQLite read-only mode."""

    pedigree: list[tuple[object, object, object, object]] = list(
        db.execute("select id, fa, mo, sex from pedigree")
    )
    """Read the pedigree identifiers, parents and recorded sex values."""

    ids: list[str] = [str(p) for p, _, _, _ in pedigree]
    """Normalised pedigree identifiers to strings for the matrix builder."""

    fathers: list[str | None] = [
        None if str(f) in ("0", "", "None") else str(f) for _, f, _, _ in pedigree
    ]
    """Normalised missing and present father identifiers."""

    mothers: list[str | None] = [
        None if str(m) in ("0", "", "None") else str(m) for _, _, m, _ in pedigree
    ]
    """Normalised missing and present mother identifiers."""

    sexes: dict[str, str] = {str(p): str(s) for p, _, _, s in pedigree}
    """Indexed recorded sex values by normalised pedigree identifier."""

    ages: dict[str, float] = {}
    """Initialised the finite ages used to define the analysis roster."""

    for person, age in db.execute(
        "select subject_id, gobs_age_at_consent_years from gobs_demographics"
    ):
        try:
            value: float = float(str(age).strip())
            """Parsed one demographic age value as a floating-point number."""
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and str(person) in sexes:
            ages[str(person)] = value
            """Retained the finite age for a person represented in the pedigree."""

    keep: list[str] = sorted(ages)
    """Fixed the retained subject roster in deterministic identifier order."""

    split: dict[str, object] = asterism.kinship_classes(
        ids, fathers, mothers, [sexes.get(p) for p in ids], keep=keep
    )
    """Built relationship matrices separated by parent-offspring class."""

    order: list[str] = split["order"]
    """Read the subject order returned with the class matrices."""

    matrices: list[np.ndarray] = [np.ascontiguousarray(m) for m in split["matrices"]]
    """Made every class relationship matrix contiguous for model fitting."""

    age: np.ndarray = np.array([ages[p] for p in order])
    """Aligned age values to the relationship-matrix subject order."""

    age = (age - age.mean()) / age.std()
    """Standardised age before constructing polynomial interaction columns."""

    male: np.ndarray = np.array([1.0 if sexes[p] == "1" else 0.0 for p in order])
    """Encoded the recorded male category in subject order."""

    design: np.ndarray = np.column_stack(
        [np.ones(len(order)), age, age**2, male, age * male, age**2 * male]
    )
    """Constructed the intercept, age polynomial and sex-interaction design."""
    return matrices, split["class_names"], split["pairs"], design


def one(job: tuple[str, int]) -> dict[str, object]:
    """Fit the omnibus and class contrasts for one simulated response.

    Args:
        job: Scenario name and zero-based replicate number.

    Returns:
        Scenario identity and available p-values for every required test.
    """
    scenario, index = job
    """Separated the generating scenario from its replicate identity."""

    matrices: list[np.ndarray] = STATE["matrices"]
    """Read the class relationship matrices initialised in this worker."""

    design: np.ndarray = STATE["design"]
    """Read the fixed-effect design initialised in this worker."""

    names: list[str] = STATE["names"]
    """Read the ordered class names initialised in this worker."""

    factor: np.ndarray = STATE["factors"][scenario]
    """Selected the covariance factor for the requested scenario."""

    n: int = design.shape[0]
    """Read the subject count from the fixed-effect design."""

    y: np.ndarray = factor @ np.random.default_rng(770_000 + index).standard_normal(n)
    """Drew one deterministic response from the scenario covariance."""

    model: asterism.ComponentModel = asterism.ComponentModel(matrices, design)
    """Built the public component model for the full class decomposition."""

    out: dict[str, object] = {"scenario": scenario, "contrasts": {}}
    """Initialised the per-replicate result record."""

    try:
        omnibus: dict[str, object] | None = model.equality_test(y)
        """Tested equality across all parent-offspring class coefficients."""
    except ValueError:
        omnibus = None
        """Recorded that the omnibus fit produced no usable p-value."""

    out["omnibus"] = None if omnibus is None else omnibus["p_value"]
    """Stored the omnibus p-value when the test completed."""

    for component, name in enumerate(names, start=1):
        others: list[np.ndarray] = [
            m for index_, m in enumerate(matrices) if index_ != component
        ]
        """Selected every matrix except the class under contrast."""

        pooled: np.ndarray = np.ascontiguousarray(sum(others[1:], others[0]))
        """Pooled all comparison matrices into one contiguous component."""

        try:
            pair: asterism.ComponentModel = asterism.ComponentModel(
                [matrices[component], pooled], design
            )
            """Built the two-component model for this class contrast."""

            contrast: dict[str, object] | None = pair.equality_test(y, [0, 1])
            """Tested this class coefficient against the pooled remainder."""
        except ValueError:
            contrast = None
            """Recorded that this class contrast produced no usable p-value."""

        out["contrasts"][name] = None if contrast is None else contrast["p_value"]
        """Stored the class-contrast p-value when the test completed."""
    return out


def main() -> int:
    """Run the kinship-class calibration and print its evidence receipt."""
    matrices, names, pairs, design = structure()
    """Built the real-design matrices and covariates without reading phenotypes."""

    n: int = design.shape[0]
    """Read the measured pedigree roster size from the design."""

    whole: np.ndarray = sum(matrices[1:], matrices[0])
    """Recombined the class matrices into the ordinary additive relationship."""

    factors: dict[str, np.ndarray] = {}
    """Initialised covariance factors for the null and powered scenarios."""

    # Null: one additive variance for everything, which is the ordinary model.
    covariance: np.ndarray = ADDITIVE * whole + (1.0 - ADDITIVE) * np.eye(n)
    """Constructed the null covariance with one additive coefficient."""

    factors["null"] = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    """Factorised the null covariance for deterministic response draws."""

    # Alternative: one class carries more than the rest.
    lifted: np.ndarray = (
        ADDITIVE * whole + LIFT * matrices[1 + list(names).index(LIFTED)]
    )
    """Added the fixed class-specific variance under the powered alternative."""

    covariance = lifted + (1.0 - ADDITIVE) * np.eye(n)
    """Constructed the powered covariance including residual variance."""

    factors["lifted"] = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    """Factorised the powered covariance for deterministic response draws."""

    print(
        f"Kinship-class tests on the real GOBS pedigree, REML. {n} people.\n"
        f"Pairs per class: " + ", ".join(f"{n_} {c}" for c, n_ in pairs.items()) + ".\n"
        f"Null: one additive variance for every class. "
        f"Alternative: {LIFTED} carries {LIFT} more.\n"
        f"{REPLICATES} replicates; only the response is simulated.\n"
    )

    shared: dict[str, object] = {
        "matrices": matrices,
        "design": design,
        "names": names,
        "factors": factors,
    }
    """Assembled immutable inputs copied once into every worker process."""

    jobs: list[tuple[str, int]] = [
        (s, i) for s in ("null", "lifted") for i in range(REPLICATES)
    ]
    """Enumerated null and powered jobs for every requested replicate."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement immediately before worker launch."""

    with ProcessPoolExecutor(
        WORKERS, initializer=start_worker, initargs=(shared,)
    ) as pool:
        results: list[dict[str, object]] = list(pool.map(one, jobs, chunksize=1))
        """Ran every deterministic replicate through the worker pool."""
    print(
        f"{len(results)} of {len(jobs)} in "
        f"{(time.perf_counter() - started) / 60:.0f} minutes.\n"
    )

    failures: list[str] = []
    """Initialised the scientific pass-rule failure messages."""

    recorded: dict[str, dict[str, object]] = {}
    """Initialised the machine-readable summaries for both scenarios."""

    for scenario in ("null", "lifted"):
        got: list[dict[str, object]] = [r for r in results if r["scenario"] == scenario]
        """Selected the replicate records for the current scenario."""

        is_null: bool = scenario == "null"
        """Distinguished calibration rates from powered rejection rates."""

        print(f"{scenario}  --  {'level' if is_null else 'power'}")
        print(
            f"  {'test':<20}"
            + "".join(f"{f'p<={l:g}':>10}" for l in LEVELS)
            + f"{'of':>6}"
        )
        recorded[scenario] = {}
        """Initialised the recorded test summaries for this scenario."""

        rows: list[tuple[str, list[float | None]]] = [
            ("omnibus", [r["omnibus"] for r in got])
        ] + [(name, [r["contrasts"][name] for r in got]) for name in names]
        """Collected the omnibus and named contrast p-values for evaluation."""

        for label, values in rows:
            have: list[float] = [v for v in values if v is not None]
            """Retained only p-values produced by completed test fits."""

            if not have:
                failures.append(f"{scenario}: nothing computed for {label}")
                continue
            rates: list[float] = [
                float(np.mean([v <= level for v in have])) for level in LEVELS
            ]
            """Computed the empirical rejection rate at every nominal level."""

            recorded[scenario][label] = {
                "computed": len(have),
                "rejection": {str(l): r for l, r in zip(LEVELS, rates, strict=True)},
            }
            """Recorded availability and level-specific rejection rates."""

            marks: list[str] = []
            """Initialised the formatted calibration marks for this output row."""

            for level, rate in zip(LEVELS, rates, strict=True):
                if not is_null:
                    marks.append(f"{rate:>10.3f}")
                    continue
                error: float = float(np.sqrt(level * (1 - level) / len(have)))
                """Computed the binomial Monte Carlo standard error."""

                ceiling: float = level + 1.96 * error
                """Computed the one-sided rejection ceiling used by the pass rule."""
                marks.append(f"{rate:>9.3f}{'!' if rate > ceiling else ' '}")
                if rate > ceiling:
                    failures.append(
                        f"{label} rejects {rate:.3f} at {level:g}, above {ceiling:.3f}"
                    )
            print(f"  {label:<20}" + "".join(marks) + f"{len(have):>6}")
        print()

    print("A `!` marks a rejection rate above its level's one-sided binomial ceiling.")
    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nThe omnibus and every contrast hold their level on the real pedigree,")
    print("with its real and badly unbalanced classes.")

    print(
        json.dumps(
            {
                "estimator": "reml",
                "pedigree": "the real GOBS pedigree; only responses simulated",
                "people": n,
                "pairs": pairs,
                "additive_share": ADDITIVE,
                "lifted_class": LIFTED,
                "lift": LIFT,
                "replicates": REPLICATES,
                "levels": list(LEVELS),
                "results": recorded,
                "note": (
                    "the omnibus is what protects the contrasts; read a contrast only "
                    "where the omnibus rejects"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
