"""Does the liability model cover, and does its test hold its level?

The liability heritability of lifetime depression came back at 0.249 with an
interval and a p-value. Neither means anything until this has run.

**This calibrates on the real pedigree, not on sibling pairs.** That is the
whole point. A family's likelihood is the probability of an orthant, which is
exact at one and two people and approximate above that, and GOBS has a family of
180. Calibrating on pairs would exercise the exact path and say nothing about
the one the real analysis uses. So the relationship matrix here is the GOBS
matrix, and only the statuses are simulated.

**Two questions, and they are separate.** Whether the interval covers is about
the profile; whether the test holds its level is about the reference
distribution, which is assumed rather than earned. A heritability of nought sits
on a bound, so the even mixture of a point mass and chi-square on one degree of
freedom is the natural reference for a Gaussian variance component. Because the
liability likelihood has different boundary geometry, its rejection rate is
checked directly here rather than inferred from the Gaussian case.

**The criterion is validity, not uniformity.** The statistic sits on a bound
under its null, so a share of fits land exactly on it and return a p-value of
one. What must hold is that the rejection rate does not exceed its level.

Run with:

    ASTERISM_REPLICATES=200 uv run --no-project python checks/liability_calibration.py

It reads the GOBS pedigree, which is study material, and simulates every status
it uses. No observed phenotype is read.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from statistics import NormalDist
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import asterism
from asterism import _core

DATABASE = Path(
    "~/MathiasLab/staging/studies/existing/safs/data/SAFS.db"
).expanduser()
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "200"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "6"))
LEVELS = (0.01, 0.05, 0.10)
# The prevalence and heritability the real analysis produced, so the level and
# the coverage are measured where the answer actually sits.
PREVALENCE = 0.254
TRUE_HERITABILITY = 0.25

STATE: dict = {}


def _start(payload: dict) -> None:
    STATE.update(payload)


def structure():
    """The GOBS relationship matrix, and a design of the same shape.

    Only people with an age are kept, which is the sample the real analysis
    runs on. Nothing but the pedigree is read.
    """
    db = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    pedigree = list(db.execute("select id, fa, mo, sex from pedigree"))
    ids = [str(p) for p, _, _, _ in pedigree]
    fathers = [None if str(f) in ("0", "", "None") else str(f) for _, f, _, _ in pedigree]
    mothers = [None if str(m) in ("0", "", "None") else str(m) for _, _, m, _ in pedigree]
    sexes = {str(p): str(s) for p, _, _, s in pedigree}
    ages = {}
    for person, age in db.execute(
        "select subject_id, gobs_age_at_consent_years from gobs_demographics"
    ):
        try:
            value = float(str(age).strip())
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and str(person) in sexes:
            ages[str(person)] = value

    keep = sorted(ages)
    relationship, order = asterism.relationship_matrix(ids, fathers, mothers, keep=keep)
    assert order == keep
    age = np.array([ages[p] for p in keep])
    age = (age - age.mean()) / age.std()
    male = np.array([1.0 if sexes[p] == "1" else 0.0 for p in keep])
    design = np.column_stack(
        [np.ones(len(keep)), age, age**2, male, age * male, age**2 * male]
    )
    return np.ascontiguousarray(relationship), design


def one(job):
    scenario, index = job
    relationship = STATE["relationship"]
    design = STATE["design"]
    factor = STATE["factors"][scenario]
    threshold = STATE["threshold"]
    n = relationship.shape[0]
    liability = factor @ np.random.default_rng(830_000 + index).standard_normal(n)
    status = (liability > threshold).astype(float)
    if status.sum() < 10 or status.sum() > n - 10:
        return None

    out = {"scenario": scenario}
    try:
        statistic, p_value, _, _ = _core.liability_test(relationship, status, design)
        out["p_value"] = p_value
        out["statistic"] = statistic
    except Exception:
        out["p_value"] = None
    if scenario == "heritable":
        try:
            estimate, lower, upper, at_lower, at_upper, _ = _core.liability_interval(
                relationship, status, design
            )
            out["interval"] = {
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "at_bound": bool(at_lower or at_upper),
            }
        except Exception:
            out["interval"] = None
    return out


def main() -> int:
    relationship, design = structure()
    n = relationship.shape[0]
    # The liability value a person must cross to be a case.
    threshold = NormalDist().inv_cdf(1.0 - PREVALENCE)

    factors = {}
    for name, heritability in (("null", 0.0), ("heritable", TRUE_HERITABILITY)):
        covariance = heritability * relationship + (1.0 - heritability) * np.eye(n)
        factors[name] = np.linalg.cholesky(covariance + 1e-10 * np.eye(n))

    # The largest family is what makes this worth running: the region
    # probability is exact to two people and approximate above.
    largest = _largest_family(relationship)
    print(
        f"Liability model on the real GOBS pedigree, ML. {n} people, "
        f"largest family {largest}.\n"
        f"Prevalence {PREVALENCE}, true heritability {TRUE_HERITABILITY} where "
        f"there is one.\n{REPLICATES} replicates. Only the statuses are "
        f"simulated; the pedigree is real.\n"
    )

    shared = {
        "relationship": relationship,
        "design": design,
        "factors": factors,
        "threshold": threshold,
    }
    jobs = [(s, i) for s in ("null", "heritable") for i in range(REPLICATES)]
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS, initializer=_start, initargs=(shared,)) as pool:
        results = [r for r in pool.map(one, jobs, chunksize=1) if r]
    print(f"{len(results)} of {len(jobs)} in "
          f"{(time.perf_counter() - started) / 60:.0f} minutes.\n")

    failures: list[str] = []
    recorded: dict = {}

    null = [r["p_value"] for r in results if r["scenario"] == "null" and r["p_value"] is not None]
    print(f"{'level':>8}{'rejected':>10}{'ceiling':>10}")
    recorded["level"] = {"computed": len(null), "rejection": {}}
    for level in LEVELS:
        rate = float(np.mean([p <= level for p in null]))
        error = np.sqrt(level * (1 - level) / max(len(null), 1))
        ceiling = level + 1.96 * error
        recorded["level"]["rejection"][str(level)] = rate
        print(f"{level:>8.2f}{rate:>10.3f}{ceiling:>10.3f}"
              f"{'  OVER' if rate > ceiling else ''}")
        if rate > ceiling:
            failures.append(
                f"the test rejects {rate:.3f} at {level:g}, above {ceiling:.3f}"
            )
    atom = float(np.mean([p > 0.999 for p in null])) if null else float("nan")
    recorded["level"]["atom_at_one"] = atom
    print(f"\nthe fit sat on the bound in {atom:.1%} of null replicates\n")

    got = [r["interval"] for r in results
           if r["scenario"] == "heritable" and r.get("interval")]
    if got:
        covered = sum(1 for i in got if i["lower"] <= TRUE_HERITABILITY <= i["upper"])
        bounded = sum(1 for i in got if i["at_bound"])
        hit = covered / len(got)
        error = np.sqrt(0.95 * 0.05 / len(got))
        low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
        median = float(np.median([i["estimate"] for i in got]))
        recorded["coverage"] = {
            "coverage": hit,
            "computed": len(got),
            "median_estimate": median,
            "at_a_bound": bounded / len(got),
            "truth": TRUE_HERITABILITY,
        }
        mark = "ok" if low <= hit <= high else ("conservative" if hit > high else "UNDER")
        print(f"interval coverage {hit:.3f}  [{low:.3f}, {high:.3f}] {mark}   "
              f"of {len(got)}, median estimate {median:.3f}, "
              f"{bounded / len(got):.1%} at a bound")
        if hit < low:
            failures.append(f"the interval covers {hit:.3f}, below the band")
    else:
        failures.append("no interval was computed at all")

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nThe test holds its level and the interval covers, on the real pedigree")
    print("with its real family sizes, which is where the approximation is used.")

    print(
        json.dumps(
            {
                "estimator": "ml",
                "pedigree": "the real GOBS pedigree; only statuses simulated",
                "people": n,
                "largest_family": largest,
                "prevalence": PREVALENCE,
                "true_heritability": TRUE_HERITABILITY,
                "replicates": REPLICATES,
                "levels": list(LEVELS),
                "nominal": 0.95,
                "results": recorded,
                "note": (
                    "the reference for the test is the even mixture of a point mass "
                    "and chi-square on one degree of freedom, which is assumed from "
                    "the Gaussian case rather than derived for a liability; this is "
                    "the check that decides whether it holds"
                ),
            },
            indent=2,
        )
    )
    return 0


def _largest_family(relationship: np.ndarray) -> int:
    """The largest connected block, which is what the approximation must cover."""
    n = relationship.shape[0]
    seen = np.zeros(n, dtype=bool)
    largest = 0
    for root in range(n):
        if seen[root]:
            continue
        stack, size = [root], 0
        seen[root] = True
        while stack:
            node = stack.pop()
            size += 1
            for other in np.nonzero(relationship[node])[0]:
                if not seen[other]:
                    seen[other] = True
                    stack.append(int(other))
        largest = max(largest, size)
    return largest


if __name__ == "__main__":
    sys.exit(main())
