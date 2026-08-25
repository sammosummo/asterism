"""Do the spatial intervals cover, and does the range mean anything?

`checks/spatial_bootstrap.py` calibrates the *test* for no spatial variance.
This calibrates the *intervals*, which is a different question and the one that
matters when a coefficient proportion and a range are reported side by side.

**Two quantities, and they are not alike.** The first is the spatial raw
coefficient proportion. In this simulation every covariance basis has mean
diagonal one, so it also equals the spatial mean-diagonal proportion. That
equivalence does not hold for arbitrarily scaled fixed matrices. The decay rate
is not a variance: it enters the covariance non-linearly, and on a single
simulated data set with a true half distance of 35 km the estimate came back at
6. Its interval contained the truth that time, which is one draw and not
coverage.

**A caution about reading the lower endpoint.** An interval for the spatial raw
coefficient proportion reaching nought is not a test of whether there is a
spatial effect. Under that null the decay rate is unidentified and the deviance
has no chi-squared reference — which is the whole reason the test is
bootstrapped. The interval describes the coefficient proportion given that the
component exists; it cannot be read backwards as a hypothesis test.

The range is checked as the half distance rather than as the decay rate, because
that is the number anybody reports and the one whose coverage is meaningful in
kilometres.

Run with:

    uv run --no-project python checks/spatial_intervals.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import asterism
import numpy as np

PAIRS: int = 150
"""Full-sibling pairs in every spatial interval replicate."""

SPACING_KM: float = 2.0
"""Distance in kilometres between neighbouring pair locations."""

REPLICATES: int = 250
"""Requested simulated data sets in the interval-coverage campaign."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "6"))
"""Worker processes used for independent calibration replicates."""

# Which treatment of the decay rate is being calibrated: the supremum over it,
# which is what profiling does, or the average across it. They are different
# statistics with different nulls, so neither calibration transfers to the other
# and each has to be run.
INTEGRATED: bool = os.environ.get("ASTERISM_INTEGRATED", "") == "1"
"""Whether the decay rate is averaged out rather than profiled."""

MODE: str = "integrated" if INTEGRATED else "profile"
"""Human-readable decay treatment retained in calibration output."""

TRUTH: dict[str, float] = {
    "additive": 0.35,
    "spatial": 0.25,
    "residual": 0.40,
    "decay_per_km": 0.02,
}
"""Fixed covariance coefficients and decay rate used for every replicate."""

TRUE_HALF_KM: float = np.log(2) / TRUTH["decay_per_km"]
"""Converted the true decay rate to its half-distance."""


def structure() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build sibling pairs spread along a line, with each pair at one place.

    Returns:
        Relationship matrix, pairwise distances, intercept design and sample size.
    """
    n: int = 2 * PAIRS
    """Calculated the number of people represented by the sibling pairs."""

    relationship: np.ndarray = np.eye(n)
    """Initialised unrelated people with unit diagonal relationships."""

    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Set the forward full-sibling relationship for this family."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Set the matching reverse full-sibling relationship."""

    place: np.ndarray = np.array([(i // 2) * SPACING_KM for i in range(n)])
    """Placed both siblings in each family at their shared line location."""

    distance: np.ndarray = np.abs(place[:, None] - place[None, :])
    """Calculated every pairwise distance along the line."""

    return relationship, distance, np.ones((n, 1)), n


def one(
    index: int,
) -> dict[str, dict[str, float | bool] | None] | None:
    """Fit and interval-score one reproducible spatial replicate.

    Args:
        index: Replicate index added to the fixed random-number seed base.

    Returns:
        Raw-proportion and half-distance intervals, with ``None`` for a refused
        interval or an integrated-away half distance.
    """
    relationship, distance, design, n = structure()
    """Built the common relationship, geography and design for this replicate."""

    covariance: np.ndarray = (
        TRUTH["additive"] * relationship
        + TRUTH["spatial"] * np.exp(-TRUTH["decay_per_km"] * distance)
        + TRUTH["residual"] * np.eye(n)
    )
    """Constructed the fixed additive, spatial and residual covariance."""

    factor: np.ndarray = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    """Factored the covariance after its fixed diagonal stabiliser."""

    y: np.ndarray = factor @ np.random.default_rng(310_000 + index).standard_normal(n)
    """Drew one reproducible Gaussian outcome from the true covariance."""

    out: dict[str, dict[str, float | bool] | None] = {}
    """Initialised the two historical interval summaries for this replicate."""

    model: asterism.SpatialModel = asterism.SpatialModel(
        [relationship],
        distance,
        design,
    )
    """Built the documented spatial model for this simulated layout."""

    try:
        proportion: dict[str, float | bool] | None = model.interval(
            y,
            "1",
            reml=True,
            integrated=INTEGRATED,
        )
        """Profiled the spatial raw coefficient proportion through named fields."""
    except ValueError:
        proportion = None
        """Recorded that the raw-proportion interval could not be profiled."""

    if proportion is None:
        out["raw_coefficient_proportion"] = None
        """Recorded a documented raw-proportion interval refusal."""
    else:
        out["raw_coefficient_proportion"] = {
            "lower": proportion["lower"],
            "upper": proportion["upper"],
            "at_bound": (proportion["lower_limited"] or proportion["upper_limited"]),
        }
        """Recorded the named raw-proportion endpoints and boundary state."""
    if INTEGRATED:
        # There is no range once it has been integrated out, so there is nothing
        # to cover and nothing to check.
        out["half_distance"] = None
        """Recorded that integrating the range leaves no half-distance estimand."""
        return out
    try:
        decay: dict[str, float | bool] | None = model.interval(
            y,
            "lambda",
            reml=True,
        )
        """Profiled the spatial decay rate through its named public record."""
    except ValueError:
        decay = None
        """Recorded that the decay-rate interval could not be profiled."""
    if decay is None:
        out["half_distance"] = None
        """Recorded a documented decay-rate interval refusal."""
    else:
        # Reported as a half distance, which runs the other way: a larger decay
        # is a shorter distance.
        out["half_distance"] = {
            "lower": np.log(2) / decay["upper"],
            "upper": np.log(2) / decay["lower"],
            "at_bound": decay["lower_limited"] or decay["upper_limited"],
        }
        """Transformed the named decay endpoints into half-distance order."""
    return out


def main() -> int:
    """Run the spatial interval-coverage calibration campaign.

    Returns:
        Zero when every interval meets its coverage gate, otherwise one.
    """
    _relationship, _distance, _design, n = structure()
    """Recovered the fixed sample size while discarding already-tested matrices."""

    print(
        f"Spatial intervals, REML, {MODE}. {PAIRS} sibling pairs, n = {n}, "
        f"{SPACING_KM:.0f} km apart.\n"
        f"Truth: additive {TRUTH['additive']}, spatial {TRUTH['spatial']}, "
        f"residual {TRUTH['residual']},\n"
        f"decay {TRUTH['decay_per_km']} per km, which is a half distance of "
        f"{TRUE_HALF_KM:.0f} km.\n"
    )

    started: float = time.perf_counter()
    """Started timing the complete interval-coverage campaign."""

    with ProcessPoolExecutor(WORKERS) as pool:
        results: list[dict[str, dict[str, float | bool] | None]] = [
            result for result in pool.map(one, range(REPLICATES)) if result
        ]
        """Ran and retained successfully returned independent replicates."""

    print(
        f"{len(results)} replicates in {(time.perf_counter() - started) / 60:.0f} minutes.\n"
    )

    failures: list[str] = []
    """Collected missing intervals and empirical under-coverage failures."""

    print(
        f"{'quantity':<16}{'coverage':>10}{'binomial 95%':>22}{'of':>6}{'at a bound':>12}"
    )
    recorded: dict[str, dict[str, float | int]] = {}
    """Accumulated coverage evidence for the final JSON record."""

    quantities: list[tuple[str, float]] = [
        ("raw_coefficient_proportion", TRUTH["spatial"])
    ]
    """Started with the spatial coefficient proportion scored in both modes."""

    if not INTEGRATED:
        # There is no range to cover once it has been integrated out, so its
        # absence is the correct answer rather than a missing result. Treating
        # it as one failed the check for having worked properly.
        quantities.append(("half_distance", TRUE_HALF_KM))
    for key, truth in quantities:
        got: list[dict[str, float | bool]] = [
            result[key] for result in results if result[key] is not None
        ]
        """Retained this quantity's successfully profiled interval records."""

        if not got:
            failures.append(f"no interval was computed for {key}")
            continue
        contained: int = sum(
            1 for interval in got if interval["lower"] <= truth <= interval["upper"]
        )
        """Counted intervals containing the fixed true quantity."""

        bounded: int = sum(1 for interval in got if interval["at_bound"])
        """Counted intervals whose profile reached a parameter bound."""

        hit: float = contained / len(got)
        """Calculated empirical coverage among computed intervals."""

        error: float = np.sqrt(0.95 * 0.05 / len(got))
        """Calculated the binomial standard error under nominal 95% coverage."""

        low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
        """Constructed the fixed normal-approximation coverage band."""

        recorded[key] = {
            "coverage": hit,
            "computed": len(got),
            "at_a_bound": bounded / len(got),
        }
        """Stored coverage, availability and boundary-frequency evidence."""

        inside: bool = low <= hit <= high
        """Classified the empirical rate relative to the acceptance band."""
        print(
            f"{key:<16}{hit:>10.3f}   [{low:.3f}, {high:.3f}]  "
            f"{'ok' if inside else ('conservative' if hit > high else 'UNDER')}"
            f"{len(got):>6}{bounded / len(got):>11.1%}"
        )
        if hit < low:
            failures.append(f"{key} covers {hit:.3f}, below the band")

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(
        f"\n{'The coefficient-proportion interval covers' if INTEGRATED else 'Both intervals cover'}. Its lower "
        f"endpoint is still not a test of whether\nthere is a spatial effect: under "
        f"that null the decay rate is unidentified,\nwhich is why the test is "
        f"bootstrapped instead."
    )

    print(
        json.dumps(
            {
                "estimator": "reml",
                "decay_rate": MODE,
                "pairs": PAIRS,
                "people": n,
                "spacing_km": SPACING_KM,
                "truth": {**TRUTH, "half_distance_km": TRUE_HALF_KM},
                "replicates": REPLICATES,
                "nominal": 0.95,
                "intervals": recorded,
                "note": (
                    "an interval for the spatial raw coefficient proportion reaching nought is not a test "
                    "of whether there is a spatial effect; the decay rate is "
                    "unidentified under that null and the test is bootstrapped"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
