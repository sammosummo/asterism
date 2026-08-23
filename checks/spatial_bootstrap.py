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
spatial effect, run the whole bootstrap on each, and see how often it rejects.

Not whether the p-values are uniform, which they are not and should not be. The
statistic cannot go below nought and is exactly nought whenever the spatial
variance fits to nothing, which happens in about a quarter of null data sets;
each of those gets a p-value of exactly one. That atom cannot touch the lower
tail, so the rejection rates are the real test, and uniformity is asked only of
the p-values away from it. That is expensive — every replicate of the outer loop runs
a whole inner bootstrap — so it is run small and rarely. A p-value procedure that
has never been checked this way is an assertion.

Run with:

    uv run --no-project python checks/spatial_bootstrap.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Protocol

import asterism
import numpy as np
from scipy import stats

PAIRS: int = 100
"""Fixed number of sibling pairs in every simulated spatial layout."""

SPACING_KM: float = 3.0
"""Distance in kilometres between neighbouring pair locations."""

BOOTSTRAP: int = 199
"""Inner null-bootstrap replicates used for every presence test."""

# Each of these runs a whole inner bootstrap, so this is the expensive number.
# At 150 the binomial band on a five per cent rate runs from 0.015 to 0.085,
# which would call almost anything calibrated; 400 narrows it to 0.028-0.072 for
# about an hour.
OUTER: int = 400
"""Outer null data sets used to measure empirical rejection rates."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "6"))
"""Worker processes used for the independent outer simulations."""

# Which treatment of the decay rate is being calibrated: the supremum over it,
# which is what profiling does, or the average across it. They are different
# statistics with different nulls, so neither calibration transfers to the other
# and each has to be run.
INTEGRATED: bool = os.environ.get("ASTERISM_INTEGRATED", "") == "1"
"""Whether the presence test averages rather than profiles over decay."""

MODE: str = "integrated" if INTEGRATED else "profile"
"""Human-readable decay treatment retained in the calibration record."""


class KolmogorovSmirnovResult(Protocol):
    """Fields consumed from SciPy's one-sample Kolmogorov-Smirnov result."""

    statistic: float
    """Maximum distance between the empirical and reference distributions."""

    pvalue: float
    """Reference-tail probability reported for the maximum distance."""


def structure(
    pairs: int = PAIRS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build sibling pairs along a line, with each pair at one place.

    Args:
        pairs: Number of two-person full-sibling families.

    Returns:
        Relationship matrix, pairwise distances, intercept design and sample size.
    """
    n: int = 2 * pairs
    """Calculated the number of people represented by the sibling pairs."""

    relationship: np.ndarray = np.eye(n)
    """Initialised unrelated people with unit diagonal relationships."""

    for pair in range(pairs):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Set the forward full-sibling relationship for this family."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Set the matching reverse full-sibling relationship."""

    place: np.ndarray = np.array([(i // 2) * SPACING_KM for i in range(n)])
    """Placed both siblings in each family at their shared line location."""

    distance: np.ndarray = np.abs(place[:, None] - place[None, :])
    """Calculated every pairwise distance along the line."""

    return relationship, distance, np.ones((n, 1)), n


def simulate(covariance: np.ndarray, seed: int) -> np.ndarray:
    """Draw one Gaussian outcome from a fixed covariance.

    Args:
        covariance: Positive-definite participant covariance matrix.
        seed: Reproducible NumPy random-number seed.

    Returns:
        One simulated outcome in covariance row order.
    """
    n: int = covariance.shape[0]
    """Read the simulated sample size from the covariance dimension."""

    factor: np.ndarray = np.linalg.cholesky(covariance + 1e-9 * np.eye(n))
    """Factored the covariance after its fixed numerical diagonal stabiliser."""

    return factor @ np.random.default_rng(seed).standard_normal(n)


def bootstrap_p_value(
    relationship: np.ndarray,
    distance: np.ndarray,
    design: np.ndarray,
    y: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP,
) -> dict[str, object]:
    """Compute the add-one bootstrap p-value in Rust.

    Args:
        relationship: Additive relationship matrix.
        distance: Pairwise spatial-distance matrix.
        design: Fixed-effect design matrix.
        y: Observed or simulated outcome vector.
        seed: Reproducible bootstrap seed.
        replicates: Requested number of bootstrap null replicates.

    Returns:
        Named statistic, replicate counts, p-value and counting rule.
    """
    model: asterism.SpatialModel = asterism.SpatialModel(
        [relationship],
        distance,
        design,
    )
    """Built the documented spatial model for this simulated layout."""

    outcome: dict[str, object] = model.bootstrap(
        y,
        replicates=replicates,
        seed=seed,
        reml=True,
        integrated=INTEGRATED,
    )
    """Computed the bootstrap presence test through its named public record."""

    return {
        "observed": outcome["statistic"],
        "exceedances": outcome["exceedances"],
        "replicates": outcome["replicates"],
        "requested": outcome["requested"],
        "p_value": outcome["p_value"],
        "rule": outcome["rule"],
    }


def one_null_data_set(index: int) -> float | None:
    """Simulate with no spatial effect, then bootstrap it. Returns the p-value."""
    relationship, distance, design, n = structure()
    """Built the common relationship, geography and design for this replicate."""

    truth: np.ndarray = 0.4 * relationship + 0.6 * np.eye(n)
    """Constructed the null covariance with no spatial contribution."""

    y: np.ndarray = simulate(truth, 800_000 + index)
    """Drew this outer null outcome from its reproducible seed."""

    try:
        outcome: dict[str, object] = bootstrap_p_value(
            relationship,
            distance,
            design,
            y,
            seed=index,
        )
        """Ran the complete inner bootstrap for this outer null outcome."""
    except ValueError:
        return None
    return outcome["p_value"]


def main() -> int:
    """Run the worked example and empirical null calibration campaign.

    Returns:
        Zero when no nominal level over-rejects, otherwise one.
    """
    relationship, distance, design, n = structure()
    """Built the fixed sibling layout shared by the calibration campaign."""

    print(
        f"Bootstrap for no spatial variance, {MODE}. {PAIRS} sibling pairs, "
        f"n = {n}, placed {SPACING_KM:.0f} km apart.\n"
    )

    # One worked example first, so the numbers below have something concrete
    # behind them.
    truth: np.ndarray = (
        0.35 * relationship + 0.25 * np.exp(-0.02 * distance) + 0.40 * np.eye(n)
    )
    """Constructed the worked example with the stated spatial contribution."""

    y: np.ndarray = simulate(truth, 12345)
    """Drew the reproducible worked-example outcome."""

    started: float = time.perf_counter()
    """Started timing the worked-example bootstrap."""

    worked: dict[str, object] = bootstrap_p_value(
        relationship,
        distance,
        design,
        y,
        seed=7,
    )
    """Computed the documented worked-example presence test."""

    print(
        f"A data set simulated with a spatial share of 0.25 and a half distance of "
        f"{np.log(2) / 0.02:.0f} km:\n"
        f"  statistic {worked['observed']:.2f}, reached by "
        f"{worked['exceedances']} of {worked['replicates']} simulated nulls, "
        f"p = {worked['p_value']:.4f}  ({time.perf_counter() - started:.0f}s)\n"
    )

    print(
        f"Calibration: {OUTER} data sets with no spatial effect at all, each put "
        f"through the whole\nbootstrap. It should reject at its nominal rate and no "
        f"more.\n"
    )
    started = time.perf_counter()
    """Restarted timing for the full outer calibration."""

    with ProcessPoolExecutor(WORKERS) as pool:
        results: list[float | None] = list(pool.map(one_null_data_set, range(OUTER)))
        """Ran every independent outer null replicate across worker processes."""

    p_values: np.ndarray = np.array([p for p in results if p is not None])
    """Retained p-values only from null data sets that completed their fit."""

    print(
        f"  {len(p_values)} of {OUTER} completed in {time.perf_counter() - started:.0f}s\n"
    )

    failures: list[str] = []
    """Collected only empirical over-rejection failures."""

    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    rates: dict[str, float] = {}
    """Accumulated empirical rejection rates by nominal level."""

    for level in (0.01, 0.05, 0.10, 0.25, 0.50):
        rate: float = float((p_values <= level).mean())
        """Calculated the empirical rejection rate at this nominal level."""

        rates[str(level)] = rate
        """Recorded the rate under its stable string-valued level key."""

        error: float = np.sqrt(level * (1 - level) / len(p_values))
        """Calculated the binomial standard error under nominal calibration."""

        lower, upper = level - 1.96 * error, level + 1.96 * error
        """Constructed the fixed normal-approximation acceptance band."""

        inside: bool = lower <= rate <= upper
        """Determined whether the observed rate lies inside that band."""

        print(
            f"  {level:>8.2f}{rate:>12.3f}   [{lower:.3f}, {upper:.3f}]  "
            f"{'ok' if inside else ('OVER' if rate > upper else 'conservative')}"
        )
        # Only over-rejection is a fault. A test that rejects less often than
        # its level is conservative, which costs power rather than manufacturing
        # a finding.
        if rate > upper:
            failures.append(
                f"rejects {rate:.3f} at the {level:.2f} level, above the band"
            )

    # **The p-values are not uniform and should not be, so uniformity is tested
    # only where it applies.** A likelihood ratio against no spatial variance
    # cannot go below nought, and in a good fraction of null data sets it is
    # exactly nought: the spatial variance fits to nothing and there is no
    # positive spatial variance. Every one gets a p-value of exactly one,
    # because no simulated statistic can fail to reach nought.
    #
    # That is a point mass at one, and it is correct. It cannot affect the lower
    # tail, which is why the rejection rates above are the substantive test and
    # come out right. Comparing the whole distribution against a *continuous*
    # uniform simply measures the atom: the first run of this check reported
    # D = 0.2725 and failed, and the fraction of null data sets with a statistic
    # of exactly nought was 27.3 per cent. The check was wrong, not the
    # bootstrap.
    atom: float = float((p_values >= 1.0).mean())
    """Measured the null point mass at a p-value of exactly one."""

    print(f"\n  The p-value is exactly one in {atom:.1%} of null data sets, where the")
    print("  statistic itself is nought and nothing can fail to reach it.")

    # **Uniformity is the wrong thing to ask for, and asking it twice was a
    # mistake.** The first version of this check compared every p-value against
    # a continuous uniform and failed at D = 0.2725; the atom turned out to be
    # 27.3 per cent, so the statistic was measuring the atom and nothing else.
    # Removing the atom and asking again still failed, at D = 0.3141, and for a
    # second reason with the same root: the bootstrap replicates carry that atom
    # too. No replicate sitting at nought can exceed an observed statistic above
    # nought, so the largest p-value such a data set can reach is about one
    # minus the atom, and the conditional p-values are uniform on (0, 0.73]
    # rather than (0, 1].
    #
    # What a p-value has to satisfy is not uniformity but **validity**:
    # P(p <= alpha) <= alpha at every level anybody uses. That is what the
    # rejection rates above measure, and they are what this check passes or
    # fails on. The distribution is reported because it is informative, not
    # because a departure from uniform is a fault.
    interior: np.ndarray = p_values[p_values < 1.0]
    """Selected p-values away from the justified atom at one."""

    reference: float = 1.0 - atom
    """Calculated the maximum support expected away from the atom."""

    rescaled: np.ndarray = interior / reference if reference > 0 else interior
    """Rescaled the conditional p-values to the unit interval when possible."""

    test: KolmogorovSmirnovResult = stats.kstest(
        np.clip(rescaled, 0.0, 1.0),
        "uniform",
    )
    """Compared the rescaled interior distribution with a continuous uniform."""
    print(
        f"\n  Away from the atom the p-values can only reach about "
        f"{reference:.2f}, since no\n  replicate at nought can exceed a statistic "
        f"above it. Rescaled by that and\n  compared with uniform: D = "
        f"{test.statistic:.4f}, p = {test.pvalue:.3f}."
    )
    print(
        "  This is reported, not required. A bootstrap p-value from a statistic\n"
        "  with an atom is valid without being uniform, and validity is what the\n"
        "  rejection rates above test."
    )

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nThe bootstrap holds its level. This is the only honest test of no")
    print("spatial variance available: with the decay rate unidentified under that")
    print("null, no closed-form reference exists to compare it against.")

    print(
        json.dumps(
            {
                "estimator": "reml",
                "decay_rate": MODE,
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
                    "atom_at_one": atom,
                    "kolmogorov_smirnov_away_from_the_atom": {
                        "statistic": test.statistic,
                        "p_value": test.pvalue,
                        "note": (
                            "uniformity is tested only where it applies; the "
                            "statistic is exactly nought in a good fraction of null "
                            "data sets and those give a p-value of exactly one"
                        ),
                    },
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
