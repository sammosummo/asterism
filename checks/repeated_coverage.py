"""Does the repeated-measures model's interval contain the truth 95 times in 100?

The joint model's one interval is on a component's correlation at a named
separation. ADR 0010 records why it is that and not the floor or the rate: those
two trade off against each other almost exactly, and the curve they describe is
what the data speak to. This counts how often the interval contains the
correlation the data were simulated from.

**An interval that has never been counted is a claim rather than a result**, and
this one is newer than most: it is taken by holding a curve rather than a
parameter, so a constrained fit is a whole fit with one free parameter fewer.
Three faults were found and fixed while it was being built, each of which made
the profile sit below the likelihood and so made every interval too narrow --
which is the direction that matters. This check is what says they are gone.

The three rules are the coverage check's rules elsewhere in this package:

1. **Every replicate is scored.** An interval that could not be computed never
   covers, and the refusals are reported beside the coverage rather than under
   it.
2. **Whether a boundary point belongs to the interval is decided by the
   Self-Liang mixture**, which is ADR 0004's recipe. It almost never arises
   here: unlike a heritability, a correlation at a separation cannot sit on
   either end of what the kernel family can express -- a correlation of exactly
   one leaves a component no variance of its own, and a correlation of exactly
   nought needs an infinite rate -- so no true value in this check is on a
   bound. The verdict is read anyway, for the same reason it is read elsewhere.
3. **A cell fails when it covers less than it claims**, judged by the
   Clopper-Pearson interval on its coverage sitting entirely below the nominal
   0.95. An interval narrower than it says is the fault this whole check exists
   to catch, and nothing excuses it.

   **A cell that covers more than it claims is reported and not failed, and the
   two one-sided miss counts are printed so that nobody has to take that on
   trust.** This differs from the coverage checks for the other families, which
   fail either way, and the difference is deliberate.

   The reason is that this quantity has hard ends and a likelihood that is not
   quadratic near them, so exact two-sided coverage is not available at any
   sample size. Both are measured rather than argued:

   - At a true correlation of 0.977, half censored, **63 per cent of intervals
     had their upper end on the ceiling** the kernel family can express. A
     truncated end cannot miss, so the misses came out nought below and three
     above out of 300, and coverage was 0.990. Moving the cell away from the
     ceiling would remove the case that shows this, which is the wrong repair:
     the real audiogram's neighbouring frequencies correlate at about 0.97, so
     this is where the data live.
   - At a true correlation of 0.165, nothing censored, no end sat on a bound at
     all. On 1,200 draws coverage was 0.961 with **37 misses below and 10
     above** against 30 expected either side. The interval is genuinely wide
     upward there, because a higher correlation can be had from the floor as
     easily as from the rate and the likelihood barely distinguishes them --
     which is the same non-identifiability that put the interval on the
     correlation rather than on either parameter in the first place.

   **Reporting rather than failing is not the same as tolerating**, and the
   distinction matters because ADR 0004 records a real fault that showed up as
   over-coverage: a missing boundary mixture read as 0.977 against a nominal
   0.95. A rule that quietly passed everything above nominal would have hidden
   it. This one prints every conservative cell, with which tail is wide and how
   many ends sat on a bound, so the question ADR 0004 had to ask gets asked
   again every time the check is run.

The cells are three correlations and three censoring rates. The correlations
span what the audiogram shows: neighbouring frequencies agree closely, distant
ones hardly at all. The censoring rates stop at a half because the check costs
about seven seconds a replicate there and because the extended high-frequency
band -- where three quarters is ordinary -- has its own trouble that no interval
recipe fixes, recorded in the audiogram project.

Run with:

    ASTERISM_REPLICATES=300 uv run --no-project python checks/repeated_coverage.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import asterism
import numpy as np
from scipy.stats import beta

# Sixty nuclear families of two parents and two children, two ears each: 240
# people and 480 rows, which is the order of the sample this model was built
# for.
FAMILIES = 60
REPLICATES = 2
# Six positions, unevenly spaced, because the audiogram's are. An evenly spaced
# line would not notice a kernel that quietly used the index.
LINE = np.array([0.0, 3.0, 6.0, 10.0, 15.0, 21.0])
# Where the interval is taken. Far enough that the correlation is well below
# one and near enough that the data have pairs at about that separation.
SEPARATION = 6.0

REPLICATES_PER_CELL = int(os.environ.get("ASTERISM_REPLICATES", "300"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "12"))
NOMINAL = 0.95

# Each is a genetic floor and rate. What is checked is the correlation they
# produce at `SEPARATION`, which is what the interval is of.
SHAPES = [
    ("low", 0.0, 0.30),
    ("middling", 0.35, 0.09),
    ("high", 0.80, 0.02),
]
RATES = [0.0, 0.25, 0.50]


def kernel_at(floor: float, rate: float, separation) -> float:
    return floor + (1.0 - floor) * np.exp(-rate * np.abs(separation))


def covariance(scale: np.ndarray, floor: float, rate: float) -> np.ndarray:
    apart = np.abs(LINE[:, None] - LINE[None, :])
    return np.outer(scale, scale) * kernel_at(floor, rate, apart)


def relationship_matrix() -> np.ndarray:
    people = FAMILIES * 4
    matrix = np.eye(people)
    for family in range(FAMILIES):
        base = family * 4
        for child in (2, 3):
            for parent in (0, 1):
                matrix[base + child, base + parent] = 0.5
                matrix[base + parent, base + child] = 0.5
        matrix[base + 2, base + 3] = 0.5
        matrix[base + 3, base + 2] = 0.5
    return matrix


def draw(floor: float, rate: float, censored: float, replicate: int):
    # **The shape belongs in the seed**, or cells at the same censoring rate are
    # fitted to the same random draws and two of them agreeing is one
    # observation rather than two.
    rng = np.random.default_rng(
        830_000
        + 7919 * replicate
        + int(1000 * censored)
        + int(1_000_000 * floor)
        + int(10_000_000 * rate)
    )
    people = FAMILIES * 4
    positions = len(LINE)
    matrix = relationship_matrix()

    genetic = covariance(np.ones(positions), floor, rate)
    # The ear level decays much faster and has almost no floor, which is what
    # test-retest noise looks like beside a genetic effect.
    residual = covariance(np.full(positions, 0.9), 0.05, 0.50)

    root = np.linalg.cholesky(matrix + 1e-9 * np.eye(people))
    shared = root @ rng.normal(size=(people, positions)) @ np.linalg.cholesky(genetic).T
    rows = people * REPLICATES
    value = np.repeat(shared, REPLICATES, axis=0) + rng.normal(
        size=(rows, positions)
    ) @ np.linalg.cholesky(residual).T

    censoring = np.zeros((rows, positions), dtype=np.int64)
    limit = np.zeros((rows, positions))
    if censored > 0.0:
        for at in range(positions):
            cut = float(np.quantile(value[:, at], 1.0 - censored))
            limit[:, at] = cut
            censoring[value[:, at] >= cut, at] = 1
    design = np.ones((rows, 1))
    return matrix, value, censoring, limit, design


def one(job):
    name, floor, rate, censored, replicate = job
    matrix, value, censoring, limit, design = draw(floor, rate, censored, replicate)
    truth = float(kernel_at(floor, rate, SEPARATION))
    try:
        model = asterism.RepeatedModel(
            matrix[np.newaxis], design, REPLICATES, len(LINE), line=LINE
        )
        got = model.correlation_interval(value, censoring, limit, 0, SEPARATION)
    except ValueError as refusal:
        return {
            "shape": name,
            "censored": censored,
            "refusal": str(refusal).replace("REPEATED_", ""),
        }
    return {
        "shape": name,
        "censored": censored,
        "truth": truth,
        "covered": covers(got, truth),
        "estimate": got["estimate"],
        "width": got["upper"] - got["lower"],
        "lower_at_bound": got["lower_at_bound"],
        "upper_at_bound": got["upper_at_bound"],
        # Which side a miss fell on. A truncated end cannot miss, so the two
        # counts together say whether over-covering has a mechanism.
        "missed_low": truth < got["lower"],
        "missed_high": truth > got["upper"],
        "profile_failures": got["profile_failures"],
    }


def covers(got: dict, truth: float) -> bool:
    """Does the interval contain the truth, by ADR 0004's rule?

    Inside the two ends is containment and outside them is not. A truth exactly
    on a bound would be decided by the mixture verdict instead, and an absent
    verdict would count as not covered -- absent means the fit at the bound
    could not be made, which is unknown ground rather than ground the data ruled
    out. No cell here puts the truth on a bound, so the branch is carried and
    not exercised.
    """
    if truth < got["lower"] or truth > got["upper"]:
        return False
    if got["lower_at_bound"] and truth <= got["lower"]:
        return bool(got["contains_lower_bound"])
    if got["upper_at_bound"] and truth >= got["upper"]:
        return bool(got["contains_upper_bound"])
    return True


def clopper_pearson(hits: int, n: int) -> tuple[float, float]:
    low = beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0
    high = beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    return float(low), float(high)


def main() -> int:
    print(
        f"{FAMILIES} families of four, {REPLICATES} replicates each, "
        f"{len(LINE)} positions, {REPLICATES_PER_CELL} draws per cell, "
        f"nominal {NOMINAL}"
    )
    print(f"the interval is on the genetic correlation {SEPARATION} apart\n")
    for name, floor, rate in SHAPES:
        print(
            f"  {name:>9}: floor {floor:.2f} rate {rate:.2f} -> correlation "
            f"{kernel_at(floor, rate, SEPARATION):.3f}"
        )
    print(f"  censoring {[f'{r:.0%}' for r in RATES]}\n", flush=True)

    jobs = [
        (name, floor, rate, censored, replicate)
        for name, floor, rate in SHAPES
        for censored in RATES
        for replicate in range(REPLICATES_PER_CELL)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for row in pool.map(one, jobs, chunksize=1):
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"  {len(rows)}/{len(jobs)}", flush=True)

    print(
        f"\n{'shape':>9} | {'truth':>6} | {'censored':>8} | {'refused':>7} | "
        f"{'coverage':>8} {'95% on it':>16} | {'width':>6} | {'on bound':>8} | "
        f"{'misses':>9} | {'fails':>5}"
    )
    print("-" * 116)
    failures = []
    conservatives = []
    summary = []
    for name, floor, rate in SHAPES:
        truth = float(kernel_at(floor, rate, SEPARATION))
        for censored in RATES:
            cell = [
                r for r in rows if r["shape"] == name and r["censored"] == censored
            ]
            refused = [r for r in cell if "refusal" in r]
            hits = sum(1 for r in cell if r.get("covered"))
            low, high = clopper_pearson(hits, len(cell))
            width = np.mean([r["width"] for r in cell if "refusal" not in r]) if (
                len(cell) > len(refused)
            ) else float("nan")
            fails = sum(r.get("profile_failures", 0) for r in cell)
            on_bound = sum(
                bool(r.get("lower_at_bound")) or bool(r.get("upper_at_bound"))
                for r in cell
            )
            missed_low = sum(bool(r.get("missed_low")) for r in cell)
            missed_high = sum(bool(r.get("missed_high")) for r in cell)
            # Under-covering fails. Over-covering is reported. See rule 3
            # for why this check parts company with the others there, and for
            # why reporting is not the same as tolerating.
            passed = high >= NOMINAL
            conservative = passed and low > NOMINAL
            why = "" if passed else "covers less than it claims"
            print(
                f"{name:>9} | {truth:>6.3f} | {censored:>7.0%} | "
                f"{len(refused):>7} | {hits / len(cell):>8.3f} "
                f"[{low:.3f}, {high:.3f}] | {width:>6.3f} | "
                f"{on_bound / len(cell):>8.1%} | "
                f"{missed_low:>4}/{missed_high:<4} | {fails:>5}"
                + ("   <- FAILED" if not passed else "")
                + ("   <- covers more than it claims" if conservative else "")
            )
            summary.append(
                {
                    "shape": name,
                    "floor": floor,
                    "rate": rate,
                    "truth": truth,
                    "censored": censored,
                    "replicates": len(cell),
                    "refused": len(refused),
                    "covered": hits,
                    "coverage": hits / len(cell),
                    "lower": low,
                    "upper": high,
                    "mean_width": None if np.isnan(width) else float(width),
                    "on_bound": on_bound,
                    "missed_low": missed_low,
                    "missed_high": missed_high,
                    "profile_failures": fails,
                    "passed": passed,
                    "conservative": conservative,
                }
            )
            record = (
                f"{name} at {censored:.0%} censored: coverage "
                f"{hits / len(cell):.3f}, interval [{low:.3f}, {high:.3f}], "
                f"{missed_low} misses below and {missed_high} above, "
                f"{on_bound / len(cell):.1%} of ends on a bound"
            )
            if not passed:
                failures.append(f"{why}. {record}")
            elif conservative:
                conservatives.append(record)

    out = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / f"repeated-coverage-{date.today().isoformat()}.json"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "what": "coverage of the repeated-measures model's profile "
                "interval on a component's correlation at a separation",
                "date": date.today().isoformat(),
                "families": FAMILIES,
                "replicates_per_person": REPLICATES,
                "positions": list(LINE),
                "separation": SEPARATION,
                "draws_per_cell": REPLICATES_PER_CELL,
                "nominal": NOMINAL,
                "cells": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwritten to {out}")

    if conservatives:
        print(
            "\nCovers more than it claims, which is reported rather than failed;"
            "\nread the miss counts to see which tail is wide:"
        )
        for cell in conservatives:
            print(f"  - {cell}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: no cell covers less than it claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
