"""Are the multi-component intervals and the boundary test calibrated?

`checks/bivariate_calibration.py` does this for two traits and two components.
This does it for one trait and three: additive, household and residual. Nothing
carries over — the intervals are reached by a different substitution, and the
test is a boundary test where the two-trait correlation tests were interior ones.

Three things are measured, all by simulating data whose answer is known.

**Interval coverage.** Simulate under known shares, fit, and count how often each
95 per cent interval contains the value used to generate the data.

**The boundary test under a true null.** Simulate with no household effect at all
and count how often the test says there is one. The null sits on the edge of the
parameter space, so the reference is the Self–Liang half-and-half mixture rather
than a plain chi-squared, and getting that wrong is not subtle: treating it as a
plain chi-squared would reject about twice as often as it should.

**Power, which is not calibration but decides whether any of this is worth
running.** A test that never rejects is perfectly calibrated. Simulating with a
household effect present and counting how often it is found says whether the
design can see one at all.

**What makes the two components separable is not what it first looks like.**
The obvious answer — that the households must contain unrelated people — is
wrong, and so is the framing that puts the question inside a single household.

The condition is that the three matrices A, H and I are linearly independent
across the whole design. It is a property of the design pooled together, not of
any one household. Two households, each a single pair, identify the model between
them if their pairs differ in relatedness: a sibling pair contributes a
covariance of 0.5·σ²_A + σ²_H and a spouse pair contributes σ²_H, and those two
equations plus the diagonal give three for three unknowns. Neither household
could do it alone.

So unrelated co-residents give the sharpest contrast but are not required. Full
sibs living with a half sib separate the components perfectly well, because 0.5
and 0.25 are two different numbers and one is enough variation. The one case that
genuinely fails is a design where **every** co-resident pair in the whole sample
has the **same** relatedness — siblings throughout, say — because A is then
exactly 0.5·H + 0.5·I and no amount of data recovers what has been added
together.

The real GOBS households are nowhere near that. Of 351 households holding two or
more measured people, 173 are a pair at relatedness 0.5, 81 a pair at nought, and
83 have varied relatedness within them; over the whole design 68 per cent of A is
not explained by any combination of H and I. The design used below has each
household holding two sibling pairs from different families, which is a similar
kind of contrast.

Run with:

    uv run --no-project python checks/components_calibration.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import asterism
import numpy as np

PAIRS: int = 200
"""Fixed the number of sibling pairs in every simulated data set."""

PER_HOUSEHOLD: int = 4
"""Placed two sibling pairs from different families in each household."""

INTERVAL_REPLICATES: int = 200
"""Fixed the number of replicates used to measure interval coverage."""

TEST_REPLICATES: int = 300
"""Fixed the number of replicates used for level and power measurements."""

LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)
"""Fixed the nominal rejection levels checked by the calibration."""

TRUTH: dict[str, float] = {
    "additive": 0.4,
    "household": 0.2,
    "residual": 0.4,
}
"""Set the generating component shares used by interval and power cells."""

NAMES: tuple[str, ...] = ("additive", "household", "residual")
"""Named component quantities in model order."""


def structure() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build sibling pairs in households holding two unrelated pairs.

    Returns:
        Additive and household matrices, an intercept design and roster size.
    """
    n: int = 2 * PAIRS
    """Computed the number of simulated people from the sibling-pair count."""

    relationship: np.ndarray = np.eye(n)
    """Initialised the additive relationship matrix with unit diagonals."""

    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Set the forward within-sibling-pair relationship coefficient."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Set the symmetric within-sibling-pair relationship coefficient."""

    household: np.ndarray = np.eye(n)
    """Initialised the co-residence matrix with individual diagonal entries."""

    for group in range(n // PER_HOUSEHOLD):
        block: slice = slice(group * PER_HOUSEHOLD, (group + 1) * PER_HOUSEHOLD)
        """Located the roster rows belonging to one simulated household."""

        household[block, block] = 1.0
        """Marked every pair of people within the household as co-residents."""
    return relationship, household, np.ones((n, 1)), n


def draw(n: int, shares: dict[str, float], seed: int) -> np.ndarray:
    """One data set, drawn from the model rather than mixed by hand.

    Each sibling's genetic value is correlated one half with its sib, which is
    what the relationship matrix says. Building the response any other way risks
    a sample the model cannot represent — a sibling correlation above a half
    needs a heritability above one, and the fit can only reach it by driving the
    residual to nought.

    Args:
        n: Number of people in the simulated roster.
        shares: Additive, household and residual generating shares.
        seed: Deterministic random seed for this replicate.

    Returns:
        One response drawn from the component covariance model.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic random-number generator for this replicate."""

    y: np.ndarray = np.zeros(n)
    """Initialised the simulated response before adding component effects."""

    if shares["household"] > 0:
        for group in range(n // PER_HOUSEHOLD):
            block: slice = slice(group * PER_HOUSEHOLD, (group + 1) * PER_HOUSEHOLD)
            """Located the roster rows sharing one household effect."""

            y[block] += np.sqrt(shares["household"]) * rng.standard_normal()
            """Added the shared household draw at its generating variance."""

    half: float = float(np.sqrt(0.5))
    """Computed the loading yielding half correlation between siblings."""

    for pair in range(n // 2):
        common: float = float(rng.standard_normal())
        """Drew the genetic effect shared by one sibling pair."""

        for member in range(2):
            genetic: float = half * common + half * rng.standard_normal()
            """Combined shared and individual draws into one genetic effect."""

            y[2 * pair + member] += (
                np.sqrt(shares["additive"]) * genetic
                + np.sqrt(shares["residual"]) * rng.standard_normal()
            )
            """Added genetic and residual contributions for this sibling."""
    return y


def coverage(
    relationship: np.ndarray,
    household: np.ndarray,
    design: np.ndarray,
    n: int,
) -> dict[str, Any]:
    """Measure component-proportion interval coverage.

    Args:
        relationship: Additive relationship matrix.
        household: Co-residence relationship matrix.
        design: Fixed-effect design matrix.
        n: Number of simulated people.

    Returns:
        Per-component attempt and coverage counts plus complete replicates.
    """
    model: asterism.ComponentModel = asterism.ComponentModel(
        [relationship, household],
        design,
    )
    """Built the public component model reused across coverage replicates."""

    contained: dict[str, int] = dict.fromkeys(NAMES[:2], 0)
    """Initialised interval-containment counts for the supported components."""

    attempted: dict[str, int] = dict.fromkeys(NAMES[:2], 0)
    """Initialised successful interval-attempt counts by component."""

    complete: int = 0
    """Initialised the number of replicates producing both intervals."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement before the simulation loop."""

    for replicate in range(INTERVAL_REPLICATES):
        y: np.ndarray = draw(n, TRUTH, 5000 + replicate)
        """Drew one deterministic response under the interval-generating truth."""

        whole: bool = True
        """Assumed both intervals complete until a profile fit refused."""

        for index, name in enumerate(NAMES[:2]):
            try:
                interval: dict[str, Any] = model.interval(y, index, reml=True)
                """Profiled the mean-diagonal component proportion."""
            except ValueError:
                whole = False
                """Marked the replicate incomplete after an interval refusal."""
                continue
            attempted[name] += 1
            """Counted this component's successfully computed interval."""

            if interval["lower"] <= TRUTH[name] <= interval["upper"]:
                contained[name] += 1
                """Counted an interval containing its generating component share."""
        if whole:
            complete += 1
            """Counted a replicate that produced both required intervals."""
        if (replicate + 1) % 25 == 0:
            print(
                f"  {replicate + 1} replicates, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"contained": contained, "attempted": attempted, "complete": complete}


def boundary(
    relationship: np.ndarray,
    household: np.ndarray,
    design: np.ndarray,
    n: int,
    present: bool,
) -> dict[str, Any]:
    """Measure the household test with or without an effect to find.

    Args:
        relationship: Additive relationship matrix.
        household: Co-residence relationship matrix.
        design: Fixed-effect design matrix.
        n: Number of simulated people.
        present: Whether the generating covariance includes household variance.

    Returns:
        Completed-test p-values and the number of refused tests.
    """
    model: asterism.ComponentModel = asterism.ComponentModel(
        [relationship, household],
        design,
    )
    """Built the public component model reused across boundary-test replicates."""

    shares: dict[str, float] = dict(TRUTH)
    """Started from the powered component shares with household variance present."""

    if not present:
        # No household effect at all; its share goes to the residual so the
        # total still comes to one.
        shares = {
            "additive": TRUTH["additive"],
            "household": 0.0,
            "residual": TRUTH["residual"] + TRUTH["household"],
        }
        """Moved the null household share to residual variance to preserve scale."""

    p_values: list[float] = []
    """Initialised the p-values returned by completed household tests."""

    refused: int = 0
    """Initialised the count of household tests that refused inference."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement before the simulation loop."""

    for replicate in range(TEST_REPLICATES):
        y: np.ndarray = draw(n, shares, (9000 if present else 4000) + replicate)
        """Drew one deterministic response for the selected boundary scenario."""

        try:
            outcome: dict[str, Any] = model.test(y, 1, reml=True)
            """Tested household presence through the documented named record."""
            p_values.append(outcome["p_value"])
        except ValueError:
            # The test refuses where another component has itself gone to
            # nought, because the mixture assumes only one is on the boundary.
            # Those replicates are counted out and reported, not counted as
            # failures to reject.
            refused += 1
            """Counted a refused boundary test separately from non-rejection."""
        if (replicate + 1) % 100 == 0:
            print(
                f"  {replicate + 1} replicates, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"p_values": p_values, "refused": refused}


def main() -> int:
    """Run interval, level and power campaigns and print their evidence."""
    relationship, household, design, n = structure()
    """Built the additive, co-residence and fixed-effect simulation design."""

    failures: list[str] = []
    """Initialised the scientific pass-rule failure messages."""

    print(
        f"One trait, three components, REML. {PAIRS} sibling pairs, n = {n}, "
        f"households of {PER_HOUSEHOLD}\nholding two pairs each, so half of "
        f"every household's co-resident pairs share no genes.\n"
        f"Truth: additive {TRUTH['additive']}, household {TRUTH['household']}, "
        f"residual {TRUTH['residual']}.\n"
    )

    print("Profile interval coverage, nominal 95 per cent.")
    covered: dict[str, Any] = coverage(relationship, household, design, n)
    """Measured interval coverage across the fixed simulation campaign."""

    print(
        f"\n  {covered['complete']} of {INTERVAL_REPLICATES} replicates gave "
        "every interval.\n"
    )
    print(f"  {'component':<12}{'coverage':>10}{'binomial 95%':>22}{'of':>8}")
    interval_results: dict[str, float] = {}
    """Initialised the recorded component-specific coverage rates."""

    for name in NAMES[:2]:
        counted: int = covered["attempted"][name]
        """Read the number of successfully computed intervals for this component."""

        if counted == 0:
            failures.append(f"no interval was computed for {name}")
            continue
        hit: float = covered["contained"][name] / counted
        """Computed this component's empirical interval coverage."""

        interval_results[name] = hit
        """Recorded the component's empirical interval coverage."""

        error: float = float(np.sqrt(0.95 * 0.05 / counted))
        """Computed the Monte Carlo standard error at nominal coverage."""

        low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
        """Constructed the two-sided binomial approximation band."""

        inside: bool = low <= hit <= high
        """Scored whether empirical coverage fell inside its sampling band."""

        print(
            f"  {name:<12}{hit:>10.3f}   [{low:.3f}, {high:.3f}]  "
            f"{'ok' if inside else ('conservative' if hit > high else 'UNDER')}"
            f"{counted:>8}"
        )
        if hit < low:
            failures.append(f"{name} covers {hit:.3f}, below the band")

    print("\n\nThe household test with no household effect to find.")
    null: dict[str, Any] = boundary(relationship, household, design, n, present=False)
    """Measured the household test under a true boundary null."""

    null_p: np.ndarray = np.array(null["p_values"])
    """Converted completed null-test p-values to a numerical vector."""

    tested: int = len(null_p)
    """Counted null replicates that produced a usable boundary test."""

    print(
        f"\n  {tested} of {TEST_REPLICATES} replicates tested"
        + (f", {null['refused']} refused" if null["refused"] else "")
        + ".\n"
    )
    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    rates: dict[str, float] = {}
    """Initialised empirical null rejection rates by nominal level."""

    for level in LEVELS:
        rate: float = float((null_p < level).mean())
        """Computed the empirical null rejection rate at this level."""

        rates[str(level)] = rate
        """Recorded the rejection rate using the level's stable string key."""

        error: float = float(np.sqrt(level * (1 - level) / tested))
        """Computed the binomial Monte Carlo standard error for this level."""

        lower, upper = level - 1.96 * error, level + 1.96 * error
        """Constructed the two-sided binomial approximation band."""

        # Only over-rejection fails. A boundary test that rejects too rarely
        # under-calls a household effect, which costs power rather than
        # manufacturing a finding.
        print(
            f"  {level:>8.2f}{rate:>12.3f}   [{lower:.3f}, {upper:.3f}]  "
            f"{'ok' if lower <= rate <= upper else ('conservative' if rate < lower else 'OVER')}"
        )
        if rate > upper:
            failures.append(
                f"the household test rejects {rate:.3f} at the {level:.2f} level, "
                "above the band"
            )
    atom: float = float((null_p >= 1.0).mean())
    """Measured the boundary mixture's empirical probability mass at zero."""

    print(
        f"\n  The statistic was exactly nought in {atom:.1%} of samples; the "
        "mixture expects about 50%."
    )

    print("\n\nThe same test with a household effect present. This is power, not")
    print("calibration: a test that never rejects is perfectly calibrated.")
    powered: dict[str, Any] = boundary(relationship, household, design, n, present=True)
    """Measured the same household test with its generating effect present."""

    power_p: np.ndarray = np.array(powered["p_values"])
    """Converted completed powered-test p-values to a numerical vector."""

    detected: float = float((power_p < 0.05).mean())
    """Computed power at the conventional five per cent rejection level."""
    print(
        f"\n  {len(power_p)} replicates; a true household share of "
        f"{TRUTH['household']} was found in {detected:.1%} at the five per cent level."
    )
    if detected < 0.5:
        failures.append(
            f"the test finds a household share of {TRUTH['household']} only "
            f"{detected:.0%} of the time; this design cannot see one"
        )

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nIntervals cover, the boundary test holds its level, and it can find")
    print("an effect that is there. Calibration is a different question from the")
    print("agreement with SOLAR and R, which do not fit this model at all.")

    print(
        json.dumps(
            {
                "estimator": "reml",
                "pairs": PAIRS,
                "people": n,
                "household_size": PER_HOUSEHOLD,
                "design_note": (
                    "each household holds two sibling pairs from different families, "
                    "so half of its co-resident pairs share no genes; the real GOBS "
                    "households are 28 per cent unrelated pairs"
                ),
                "truth": TRUTH,
                "intervals": {
                    "replicates_requested": INTERVAL_REPLICATES,
                    "replicates_complete": covered["complete"],
                    "nominal": 0.95,
                    "coverage": interval_results,
                },
                "household_test_under_the_null": {
                    "replicates_requested": TEST_REPLICATES,
                    "replicates_tested": tested,
                    "replicates_refused": null["refused"],
                    "rule": "self_liang_50_50_mixture",
                    "rejection_rates": rates,
                    "atom_at_zero": atom,
                },
                "power": {
                    "true_household_share": TRUTH["household"],
                    "detected_at_five_per_cent": detected,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
