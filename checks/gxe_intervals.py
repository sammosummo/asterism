"""Do the genotype-by-environment intervals cover?

`checks/gxe_calibration.py` calibrates the *tests*. This calibrates the
*intervals*, which is a different question and the one that matters when a
heritability is reported at three environments with a band on each.

**Two quantities, and they are not alike.** The genetic correlation between two
environments is the interaction itself, and it is held by one coordinate of the
surface. The heritability at an environment is a ratio of two things the data
sees only in combination, and it is held by scaling the whole residual surface —
because the obvious per-coordinate solve is infeasible over much of the search,
and an infeasible point scores as no fit at all, so the profile stops at the edge
of the feasible region and reports it as a likelihood crossing. That produced
intervals that looked tight and precise and were neither: two different
heritabilities whose intervals began at the same number. The scaling route is
feasible everywhere, and this check is what says whether it is also right.

**Each surface is simulated from its own family.** A rank-one surface drawn from
one family is misspecified for the other, and an interval has no obligation to
cover the truth of a model it is not fitting. Coverage under misspecification is
a separate question from coverage, and mixing them would answer neither.

**Expect wide intervals and read the bound column.** A heritability at one
environment is weakly identified when the people are spread along a continuous
environment: an endpoint that runs to nought or one is the model saying so, not
a fault. What would be a fault is under-covering.

Run with:

    uv run --no-project python checks/gxe_intervals.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from asterism import _core

PAIRS = 250
N = 2 * PAIRS
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "300"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "6"))
LOW, HIGH = -1.0, 1.0

# Each surface gets an alternative from its own family, with a real interaction
# in it: the genetic variance changes and the correlation across the range is
# well below one.
TRUTH = {
    "exponential": {
        # sqrt(g(zi) g(zj)) exp(-lambda |zi - zj|), g(z) = exp(-0.7 + 0.6 z).
        "genetic": lambda a, b: np.exp(-0.7 + 0.3 * (a + b)) * np.exp(-0.4 * np.abs(a - b)),
        "residual": lambda z: np.exp(-0.7 + 0.2 * z),
    },
    "random_regression": {
        # q00 + q01 (zi + zj) + q11 zi zj, with q00 q11 well above q01^2.
        "genetic": lambda a, b: 0.5 + 0.05 * (a + b) + 0.45 * a * b,
        "residual": lambda z: 0.5 + 0.1 * z + 0.05 * z * z,
    },
}


def structure():
    """Sibling pairs, each person carrying an environment.

    The environment varies within a pair as well as between it, or a surface
    could not be told from a plain heritability.
    """
    relationship = np.eye(N)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5
    z = np.random.default_rng(20_260_813).uniform(-1.5, 1.5, N)
    return relationship, z, np.ones((N, 1))


RELATIONSHIP, Z, DESIGN = structure()


def truth_for(surface: str) -> dict:
    """The three quantities this check covers, computed from the surface itself."""
    setting = TRUTH[surface]
    genetic = setting["genetic"]
    residual = setting["residual"]

    def heritability(z):
        g = float(genetic(np.array(z), np.array(z)))
        return g / (g + float(residual(np.array(z))))

    g_low = float(genetic(np.array(LOW), np.array(LOW)))
    g_high = float(genetic(np.array(HIGH), np.array(HIGH)))
    g_cross = float(genetic(np.array(LOW), np.array(HIGH)))
    return {
        ("heritability", LOW, 0.0): heritability(LOW),
        ("heritability", HIGH, 0.0): heritability(HIGH),
        ("correlation", LOW, HIGH): g_cross / np.sqrt(g_low * g_high),
    }


FACTORS = {}
for name, setting in TRUTH.items():
    covariance = RELATIONSHIP * setting["genetic"](Z[:, None], Z[None, :])
    covariance[np.diag_indices(N)] += setting["residual"](Z)
    FACTORS[name] = np.linalg.cholesky(covariance + 1e-9 * np.eye(N))

TRUE_VALUES = {name: truth_for(name) for name in TRUTH}


def one(job):
    surface, index = job
    y = FACTORS[surface] @ np.random.default_rng(660_000 + index).standard_normal(N)
    out = {"surface": surface, "intervals": {}}
    for quantity, first, second in TRUE_VALUES[surface]:
        try:
            estimate, lower, upper, at_lower, at_upper = _core.gxe_interval(
                RELATIONSHIP, Z, DESIGN, y, surface, quantity, first, second, True
            )
            out["intervals"][f"{quantity}@{first}"] = {
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "at_bound": bool(at_lower or at_upper),
            }
        except Exception:
            out["intervals"][f"{quantity}@{first}"] = None
    return out


def main() -> int:
    print(
        f"Genotype-by-environment intervals, REML. {PAIRS} sibling pairs, n = {N}, "
        f"{REPLICATES} replicates.\nEach surface simulated from its own family, "
        f"environments {LOW} and {HIGH}.\n"
    )
    jobs = [(surface, index) for surface in TRUTH for index in range(REPLICATES)]
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(one, jobs, chunksize=2))
    print(f"{len(jobs)} data sets in {(time.perf_counter() - started) / 60:.1f} minutes.\n")

    failures: list[str] = []
    recorded: dict = {}
    for surface in TRUTH:
        got = [r for r in results if r["surface"] == surface]
        print(f"{surface}")
        print(
            f"  {'quantity':<20}{'truth':>8}{'median':>9}{'coverage':>10}"
            f"{'binomial 95%':>20}{'of':>6}{'at a bound':>12}"
        )
        recorded[surface] = {}
        for (quantity, first, second), truth in TRUE_VALUES[surface].items():
            key = f"{quantity}@{first}"
            have = [r["intervals"][key] for r in got if r["intervals"][key] is not None]
            if not have:
                failures.append(f"{surface}: no interval was computed for {key}")
                continue
            covered = sum(1 for i in have if i["lower"] <= truth <= i["upper"])
            bounded = sum(1 for i in have if i["at_bound"])
            hit = covered / len(have)
            error = np.sqrt(0.95 * 0.05 / len(have))
            low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
            median = float(np.median([i["estimate"] for i in have]))
            recorded[surface][key] = {
                "truth": truth,
                "median_estimate": median,
                "coverage": hit,
                "computed": len(have),
                "at_a_bound": bounded / len(have),
            }
            mark = "ok" if low <= hit <= high else ("conservative" if hit > high else "UNDER")
            print(
                f"  {key:<20}{truth:>8.3f}{median:>9.3f}{hit:>10.3f}"
                f"   [{low:.3f}, {high:.3f}] {mark:<13}{len(have):>4}{bounded / len(have):>11.1%}"
            )
            # One-sided: covering more than nominal is conservative, not wrong.
            if hit < low:
                failures.append(f"{surface} {key} covers {hit:.3f}, below the band")
        print()

    if failures:
        print("NOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("Both surfaces' intervals cover on their own family. A heritability at one")
    print("environment is weakly identified when people are spread along a continuous")
    print("environment, so an endpoint running to nought or one is the model saying so.")

    print(
        json.dumps(
            {
                "estimator": "reml",
                "pairs": PAIRS,
                "people": N,
                "replicates": REPLICATES,
                "environments": [LOW, HIGH],
                "nominal": 0.95,
                "surfaces": recorded,
                "note": (
                    "each surface is simulated from its own family; coverage under "
                    "misspecification is a separate question and is not measured here. "
                    "The heritability is held by scaling the whole residual surface "
                    "rather than by solving one coordinate, which is infeasible over "
                    "much of the search and produced a false likelihood crossing at "
                    "the edge of the feasible region"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
