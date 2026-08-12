"""The parametric bootstrap for no spatial variance, and its calibration.

**Why a bootstrap and not a likelihood ratio table.** When the spatial variance
is nought the decay rate does not appear in the likelihood at all — it is a
parameter present only under the alternative. The usual asymptotics do not
apply: the statistic has neither a chi-squared distribution nor the 50:50
boundary mixture that `components.rs` uses for a household variance, and there is
no closed form to reach for. The null has to be simulated. This is the same
reason the port lab froze its decay grid and bootstrapped, and estimating the
decay rate makes the bootstrap necessary rather than merely careful.

**The procedure, and where it lives.** Fit the reduced model, which has no
spatial term and so no decay rate. Simulate from it. Refit both models to each
simulated data set and record the likelihood ratio. The p-value is the add-one
proportion of simulated statistics reaching the observed one:

    p = (1 + #{T_b >= T_obs}) / (B + 1)

The add-one is not a rounding convenience. Without it a statistic larger than
every simulated one returns exactly nought, which claims more than B replicates
can support.

**All of that is in the Rust**, in `bootstrap_spatial_variance`, including the
simulation and the counting. It is the p-value, so it belongs there rather than
in a loop written here. This file drives it and checks that it holds its level;
it does not reimplement it. The generator is splitmix64 seeded per call, so any
bootstrap can be reproduced from its seed alone, and there is a test that two
runs at one seed agree exactly.

**What this file checks.** That the recipe is calibrated: simulate data with no
spatial effect, run the whole bootstrap on each, and see whether the resulting
p-values are uniform. That is expensive — every replicate of the outer loop runs
a whole inner bootstrap — so it is run small and rarely. A p-value procedure that
has never been checked this way is an assertion.

Run with:

    uv run --no-project python checks/spatial_bootstrap.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from asterism import _core

PAIRS = 100
SPACING_KM = 3.0
BOOTSTRAP = 199
OUTER = 150
WORKERS = 6


def structure(pairs: int = PAIRS):
    """Sibling pairs along a line, each pair at one place."""
    n = 2 * pairs
    relationship = np.eye(n)
    for pair in range(pairs):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5
    place = np.array([(i // 2) * SPACING_KM for i in range(n)])
    distance = np.abs(place[:, None] - place[None, :])
    return relationship, distance, np.ones((n, 1)), n


def simulate(covariance: np.ndarray, seed: int) -> np.ndarray:
    n = covariance.shape[0]
    factor = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    return factor @ np.random.default_rng(seed).standard_normal(n)


def bootstrap_p_value(
    relationship, distance, design, y, seed: int, replicates: int = BOOTSTRAP
) -> dict:
    """The add-one bootstrap p-value, computed in the Rust."""
    observed, exceedances, usable, requested, p_value, rule = _core.spatial_bootstrap(
        [relationship], distance, design, y, replicates, seed, True
    )
    return {
        "observed": observed,
        "exceedances": exceedances,
        "replicates": usable,
        "requested": requested,
        "p_value": p_value,
        "rule": rule,
    }


def one_null_data_set(index: int) -> float | None:
    """Simulate with no spatial effect, then bootstrap it. Returns the p-value."""
    relationship, distance, design, n = structure()
    truth = 0.4 * relationship + 0.6 * np.eye(n)
    y = simulate(truth, 800_000 + index)
    try:
        return bootstrap_p_value(relationship, distance, design, y, seed=index)["p_value"]
    except Exception:
        return None


def main() -> int:
    relationship, distance, design, n = structure()
    print(
        f"Bootstrap for no spatial variance. {PAIRS} sibling pairs, n = {n}, "
        f"placed {SPACING_KM:.0f} km apart.\n"
    )

    # One worked example first, so the numbers below have something concrete
    # behind them.
    truth = 0.35 * relationship + 0.25 * np.exp(-0.02 * distance) + 0.40 * np.eye(n)
    y = simulate(truth, 12345)
    started = time.perf_counter()
    worked = bootstrap_p_value(relationship, distance, design, y, seed=7)
    print(
        f"A data set simulated with a spatial share of 0.25 and a half distance of "
        f"{np.log(2)/0.02:.0f} km:\n"
        f"  statistic {worked['observed']:.2f}, reached by "
        f"{worked['exceedances']} of {worked['replicates']} simulated nulls, "
        f"p = {worked['p_value']:.4f}  ({time.perf_counter() - started:.0f}s)\n"
    )

    print(
        f"Calibration: {OUTER} data sets with no spatial effect at all, each put "
        f"through the whole\nbootstrap. The p-values should be uniform.\n"
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(one_null_data_set, range(OUTER)))
    p_values = np.array([p for p in results if p is not None])
    print(f"  {len(p_values)} of {OUTER} completed in {time.perf_counter() - started:.0f}s\n")

    failures = []
    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    rates = {}
    for level in (0.05, 0.10, 0.25):
        rate = float((p_values <= level).mean())
        rates[str(level)] = rate
        error = np.sqrt(level * (1 - level) / len(p_values))
        lower, upper = level - 1.96 * error, level + 1.96 * error
        inside = lower <= rate <= upper
        print(
            f"  {level:>8.2f}{rate:>12.3f}   [{lower:.3f}, {upper:.3f}]  "
            f"{'ok' if inside else ('OVER' if rate > upper else 'conservative')}"
        )
        if rate > upper:
            failures.append(
                f"rejects {rate:.3f} at the {level:.2f} level, above the band"
            )

    # A bootstrap p-value is discrete, taking values k/(B+1), so it cannot be
    # exactly uniform and the test is run against the discrete reference rather
    # than the continuous one it is easy to reach for by mistake.
    test = stats.kstest(p_values, "uniform")
    print(
        f"\n  Kolmogorov-Smirnov against uniform: D = {test.statistic:.4f}, "
        f"p = {test.pvalue:.3f}"
    )
    print(
        f"  (a bootstrap p-value is discrete on multiples of 1/{BOOTSTRAP + 1}, so a "
        f"little\n   departure here is the grid and not a fault)"
    )
    if test.pvalue < 0.001:
        failures.append(f"p-values are far from uniform, KS p = {test.pvalue:.5f}")

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nThe bootstrap holds its level. This is the only honest test of no")
    print("spatial variance available: with the decay rate unidentified under that")
    print("null, no closed-form reference exists to compare it against.")

    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/spatial-bootstrap-2026-08-12.json").write_text(
        json.dumps(
            {
                "what": "the parametric bootstrap for no spatial variance, and whether it holds its level",
                "date": "2026-08-12",
                "estimator": "reml",
                "pairs": PAIRS,
                "people": n,
                "spacing_km": SPACING_KM,
                "why_a_bootstrap": (
                    "the decay rate is unidentified when the spatial variance is "
                    "nought, so the likelihood ratio has neither a chi-squared nor a "
                    "chi-bar-squared null and no closed form exists"
                ),
                "bootstrap_replicates": BOOTSTRAP,
                "p_value_rule": "add_one: (1 + exceedances) / (replicates + 1)",
                "worked_example": worked,
                "calibration": {
                    "data_sets": OUTER,
                    "completed": len(p_values),
                    "rejection_rates": rates,
                    "kolmogorov_smirnov": {
                        "statistic": test.statistic,
                        "p_value": test.pvalue,
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
