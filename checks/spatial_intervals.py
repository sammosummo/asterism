"""Do the spatial intervals cover, and does the range mean anything?

`checks/spatial_bootstrap.py` calibrates the *test* for no spatial variance.
This calibrates the *intervals*, which is a different question and the one that
matters when a share and a range are reported side by side.

**Two quantities, and they are not alike.** The spatial share behaves like the
other variance shares in the package. The decay rate does not: it is the one
parameter here that is not a variance, it enters the covariance non-linearly, and
on a single simulated data set with a true half distance of 35 km the estimate
came back at 6. Its interval contained the truth that time, which is one draw and
not coverage.

**A caution about reading the lower endpoint of the share.** An interval for the
spatial share reaching nought is not a test of whether there is a spatial effect.
Under that null the decay rate is unidentified and the deviance has no
chi-squared reference — which is the whole reason the test is bootstrapped. The
interval is honest about how large the share is, given that there is one; it
cannot be read backwards as a hypothesis test.

The range is checked as the half distance rather than as the decay rate, because
that is the number anybody reports and the one whose coverage is meaningful in
kilometres.

Run with:

    uv run --no-project python checks/spatial_intervals.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import os

import numpy as np

from asterism import _core

PAIRS = 150
SPACING_KM = 2.0
REPLICATES = 250
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "6"))
# Which treatment of the decay rate is being calibrated: the supremum over it,
# which is what profiling does, or the average across it. They are different
# statistics with different nulls, so neither calibration transfers to the other
# and each has to be run.
INTEGRATED = os.environ.get("ASTERISM_INTEGRATED", "") == "1"
MODE = "integrated" if INTEGRATED else "profile"

TRUTH = dict(additive=0.35, spatial=0.25, residual=0.40, decay_per_km=0.02)
TRUE_HALF_KM = np.log(2) / TRUTH["decay_per_km"]


def structure():
    """Sibling pairs, each pair at one place, spread along a line."""
    n = 2 * PAIRS
    relationship = np.eye(n)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5
    place = np.array([(i // 2) * SPACING_KM for i in range(n)])
    distance = np.abs(place[:, None] - place[None, :])
    return relationship, distance, np.ones((n, 1)), n


def one(index: int) -> dict | None:
    relationship, distance, design, n = structure()
    covariance = (
        TRUTH["additive"] * relationship
        + TRUTH["spatial"] * np.exp(-TRUTH["decay_per_km"] * distance)
        + TRUTH["residual"] * np.eye(n)
    )
    factor = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    y = factor @ np.random.default_rng(310_000 + index).standard_normal(n)

    out: dict = {}
    try:
        lower, upper, at_lower, at_upper, _ = _core.spatial_interval(
            [relationship], distance, design, y, "1", True, INTEGRATED
        )
        out["share"] = {"lower": lower, "upper": upper, "at_bound": at_lower or at_upper}
    except Exception:
        out["share"] = None
    if INTEGRATED:
        # There is no range once it has been integrated out, so there is nothing
        # to cover and nothing to check.
        out["half_distance"] = None
        return out
    try:
        lower, upper, at_lower, at_upper, _ = _core.spatial_interval(
            [relationship], distance, design, y, "lambda", True
        )
        # Reported as a half distance, which runs the other way: a larger decay
        # is a shorter distance.
        out["half_distance"] = {
            "lower": np.log(2) / upper,
            "upper": np.log(2) / lower,
            "at_bound": at_lower or at_upper,
        }
    except Exception:
        out["half_distance"] = None
    return out


def main() -> int:
    relationship, distance, design, n = structure()
    print(
        f"Spatial intervals, REML, {MODE}. {PAIRS} sibling pairs, n = {n}, "
        f"{SPACING_KM:.0f} km apart.\n"
        f"Truth: additive {TRUTH['additive']}, spatial {TRUTH['spatial']}, "
        f"residual {TRUTH['residual']},\n"
        f"decay {TRUTH['decay_per_km']} per km, which is a half distance of "
        f"{TRUE_HALF_KM:.0f} km.\n"
    )

    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        results = [r for r in pool.map(one, range(REPLICATES)) if r]
    print(f"{len(results)} replicates in {(time.perf_counter() - started) / 60:.0f} minutes.\n")

    failures = []
    print(f"{'quantity':<16}{'coverage':>10}{'binomial 95%':>22}{'of':>6}{'at a bound':>12}")
    recorded = {}
    quantities = [("share", TRUTH["spatial"])]
    if not INTEGRATED:
        # There is no range to cover once it has been integrated out, so its
        # absence is the correct answer rather than a missing result. Treating
        # it as one failed the check for having worked properly.
        quantities.append(("half_distance", TRUE_HALF_KM))
    for key, truth in quantities:
        got = [r[key] for r in results if r[key] is not None]
        if not got:
            failures.append(f"no interval was computed for {key}")
            continue
        contained = sum(1 for i in got if i["lower"] <= truth <= i["upper"])
        bounded = sum(1 for i in got if i["at_bound"])
        hit = contained / len(got)
        error = np.sqrt(0.95 * 0.05 / len(got))
        low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
        recorded[key] = {
            "coverage": hit,
            "computed": len(got),
            "at_a_bound": bounded / len(got),
        }
        inside = low <= hit <= high
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
        f"\n{'The share covers' if INTEGRATED else 'Both intervals cover'}. Its lower "
        f"endpoint is still not a test of whether\nthere is a spatial effect: under "
        f"that null the decay rate is unidentified,\nwhich is why the test is "
        f"bootstrapped instead."
    )

    Path("evidence").mkdir(exist_ok=True)
    Path(f"evidence/spatial-intervals-{MODE}-2026-08-13.json").write_text(
        json.dumps(
            {
                "what": "coverage of the spatial share and the half distance",
                "date": "2026-08-13",
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
                    "an interval for the spatial share reaching nought is not a test "
                    "of whether there is a spatial effect; the decay rate is "
                    "unidentified under that null and the test is bootstrapped"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
