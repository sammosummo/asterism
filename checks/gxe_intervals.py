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
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import asterism
import numpy as np

PAIRS: int = 250
"""Fixed the number of sibling pairs in every simulated data set."""

N: int = 2 * PAIRS
"""Computed the fixed simulated roster size."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "300"))
"""Selected the requested replicates per surface from the environment."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "6"))
"""Selected the worker-process count from the environment."""

LOW: float = -1.0
"""Fixed the lower environment at which supported quantities are profiled."""

HIGH: float = 1.0
"""Fixed the upper environment at which supported quantities are profiled."""

# Each surface gets an alternative from its own family, with a real interaction
# in it: the genetic variance changes and the correlation across the range is
# well below one.
TRUTH: dict[
    str,
    dict[
        str,
        Callable[[np.ndarray, np.ndarray], np.ndarray]
        | Callable[[np.ndarray], np.ndarray],
    ],
] = {
    "exponential": {
        # sqrt(g(zi) g(zj)) exp(-lambda |zi - zj|), g(z) = exp(-0.7 + 0.6 z).
        "genetic": lambda a, b: (
            np.exp(-0.7 + 0.3 * (a + b)) * np.exp(-0.4 * np.abs(a - b))
        ),
        "residual": lambda z: np.exp(-0.7 + 0.2 * z),
    },
    "random_regression": {
        # q00 + q01 (zi + zj) + q11 zi zj, with q00 q11 well above q01^2.
        "genetic": lambda a, b: 0.5 + 0.05 * (a + b) + 0.45 * a * b,
        "residual": lambda z: 0.5 + 0.1 * z + 0.05 * z * z,
    },
}
"""Defined one participant-free generating surface in each model family."""


def structure() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sibling pairs, each person carrying an environment.

    The environment varies within a pair as well as between it, or a surface
    could not be told from a plain heritability.

    Returns:
        Relationship matrix, continuous environments and intercept design.
    """
    relationship: np.ndarray = np.eye(N)
    """Initialised the relationship matrix with individual diagonal entries."""

    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Set the forward within-pair relationship coefficient."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Set the symmetric within-pair relationship coefficient."""

    z: np.ndarray = np.random.default_rng(20_260_813).uniform(-1.5, 1.5, N)
    """Drew fixed within- and between-family continuous environments."""
    return relationship, z, np.ones((N, 1))


RELATIONSHIP, Z, DESIGN = structure()
"""Built the deterministic design shared by every surface and replicate."""


def truth_for(surface: str) -> dict[tuple[str, float, float], float]:
    """Compute the three covered quantities directly from one surface.

    Args:
        surface: Named generating surface family.

    Returns:
        Heritabilities at both environments and their genetic correlation.
    """
    setting: dict[
        str,
        Callable[[np.ndarray, np.ndarray], np.ndarray]
        | Callable[[np.ndarray], np.ndarray],
    ] = TRUTH[surface]
    """Selected the generating genetic and residual surface functions."""

    genetic: Any = setting["genetic"]
    """Selected the two-environment genetic covariance function."""

    residual: Any = setting["residual"]
    """Selected the one-environment residual variance function."""

    def heritability(z: float) -> float:
        """Return the variance ratio at one environment.

        Args:
            z: Environment at which to evaluate the variance ratio.

        Returns:
            Genetic variance divided by total variance at that environment.
        """
        g: float = float(genetic(np.array(z), np.array(z)))
        """Evaluated genetic variance on the surface diagonal."""
        return g / (g + float(residual(np.array(z))))

    g_low: float = float(genetic(np.array(LOW), np.array(LOW)))
    """Evaluated genetic variance at the lower environment."""

    g_high: float = float(genetic(np.array(HIGH), np.array(HIGH)))
    """Evaluated genetic variance at the upper environment."""

    g_cross: float = float(genetic(np.array(LOW), np.array(HIGH)))
    """Evaluated genetic covariance across the two environments."""
    return {
        ("heritability", LOW, 0.0): heritability(LOW),
        ("heritability", HIGH, 0.0): heritability(HIGH),
        ("correlation", LOW, HIGH): g_cross / np.sqrt(g_low * g_high),
    }


FACTORS: dict[str, np.ndarray] = {}
"""Initialised covariance factors for every generating surface family."""

for name, setting in TRUTH.items():
    covariance: np.ndarray = RELATIONSHIP * setting["genetic"](Z[:, None], Z[None, :])
    """Constructed the relationship-scaled genetic covariance for this surface."""

    covariance[np.diag_indices(N)] += setting["residual"](Z)
    """Added environment-specific residual variance on the covariance diagonal."""

    FACTORS[name] = np.linalg.cholesky(covariance + 1e-9 * np.eye(N))
    """Factorised this generating covariance for deterministic response draws."""

TRUE_VALUES: dict[str, dict[tuple[str, float, float], float]] = {
    name: truth_for(name) for name in TRUTH
}
"""Computed exact truths for every generating surface family."""


def one(job: tuple[str, int]) -> dict[str, Any]:
    """Profile every supported quantity for one simulated response.

    Args:
        job: Surface name and zero-based replicate number.

    Returns:
        Named interval records or documented refusals for each quantity.
    """
    surface, index = job
    """Separated the generating surface from its replicate identity."""

    y: np.ndarray = FACTORS[surface] @ np.random.default_rng(
        660_000 + index
    ).standard_normal(N)
    """Drew one deterministic response from the selected surface covariance."""

    out: dict[str, Any] = {"surface": surface, "intervals": {}}
    """Initialised the public interval records for one simulated response."""
    model: asterism.GxeModel = asterism.GxeModel(
        RELATIONSHIP,
        Z,
        DESIGN,
        surface=surface,
    )
    """Built the documented continuous GxE model for this surface."""

    for quantity, first, second in TRUE_VALUES[surface]:
        try:
            interval: dict[str, Any] | None = model.interval(
                y,
                quantity,
                first,
                second,
                reml=True,
            )
            """Profiled the requested public GxE quantity into a named record."""
        except ValueError:
            interval = None
            """Recorded that profiling this quantity produced no interval."""

        if interval is None:
            out["intervals"][f"{quantity}@{first}"] = None
            """Recorded a documented interval refusal for this quantity."""
        else:
            out["intervals"][f"{quantity}@{first}"] = {
                "estimate": interval["estimate"],
                "lower": interval["lower"],
                "upper": interval["upper"],
                "at_bound": bool(
                    interval["lower_limited"] or interval["upper_limited"]
                ),
            }
            """Recorded named profile fields without positional unpacking."""
    return out


def main() -> int:
    """Run every interval-coverage cell and print its evidence receipt."""
    print(
        f"Genotype-by-environment intervals, REML. {PAIRS} sibling pairs, n = {N}, "
        f"{REPLICATES} replicates.\nEach surface simulated from its own family, "
        f"environments {LOW} and {HIGH}.\n"
    )
    jobs: list[tuple[str, int]] = [
        (surface, index) for surface in TRUTH for index in range(REPLICATES)
    ]
    """Enumerated every surface and replicate exactly once."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement immediately before worker launch."""

    with ProcessPoolExecutor(WORKERS) as pool:
        results: list[dict[str, Any]] = list(pool.map(one, jobs, chunksize=2))
        """Ran every deterministic interval replicate through the worker pool."""
    print(
        f"{len(jobs)} data sets in {(time.perf_counter() - started) / 60:.1f} minutes.\n"
    )

    failures: list[str] = []
    """Initialised the scientific pass-rule failure messages."""

    recorded: dict[str, dict[str, dict[str, float | int]]] = {}
    """Initialised the machine-readable summaries for every surface."""

    for surface in TRUTH:
        got: list[dict[str, Any]] = [r for r in results if r["surface"] == surface]
        """Selected every attempted replicate for the current surface."""

        print(f"{surface}")
        print(
            f"  {'quantity':<20}{'truth':>8}{'median':>9}{'coverage':>10}"
            f"{'binomial 95%':>20}{'of':>6}{'at a bound':>12}"
        )
        recorded[surface] = {}
        """Initialised quantity-specific summaries for this surface."""

        for (quantity, first, _second), truth in TRUE_VALUES[surface].items():
            key: str = f"{quantity}@{first}"
            """Constructed the stable interval identity used in result records."""

            have: list[dict[str, Any]] = [
                r["intervals"][key] for r in got if r["intervals"][key] is not None
            ]
            """Selected interval records that completed for this quantity."""

            if not have:
                failures.append(f"{surface}: no interval was computed for {key}")
                continue
            covered: int = sum(1 for i in have if i["lower"] <= truth <= i["upper"])
            """Counted intervals containing the exact generating quantity."""

            bounded: int = sum(1 for i in have if i["at_bound"])
            """Counted intervals limited by at least one parameter bound."""

            hit: float = covered / len(have)
            """Computed empirical coverage across completed intervals."""

            error: float = float(np.sqrt(0.95 * 0.05 / len(have)))
            """Computed the Monte Carlo standard error at nominal coverage."""

            low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
            """Constructed the two-sided binomial approximation band."""

            median: float = float(np.median([i["estimate"] for i in have]))
            """Computed the median point estimate as a recovery diagnostic."""

            recorded[surface][key] = {
                "truth": truth,
                "median_estimate": median,
                "coverage": hit,
                "computed": len(have),
                "at_a_bound": bounded / len(have),
            }
            """Recorded truth, recovery, coverage and boundary frequency."""

            mark: str = (
                "ok"
                if low <= hit <= high
                else ("conservative" if hit > high else "UNDER")
            )
            """Classified coverage against its Monte Carlo sampling band."""

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
    print(
        "environment, so an endpoint running to nought or one is the model saying so."
    )

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
