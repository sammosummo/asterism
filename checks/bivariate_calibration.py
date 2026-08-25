"""Are the two-trait intervals and tests calibrated?

The comparisons against SOLAR and R ask whether Asterism computes the same
numbers. This simulation asks whether a 95 per cent interval contains the truth
at its stated rate and whether each test rejects at its stated level.

Three things are measured, all by simulating data whose answer is known.

**Interval coverage.** Simulate under known heritabilities and correlations, fit,
and count how often each 95 per cent profile interval contains the value used to
generate the data. Conservative is safe and expected at this size; anything below
the band is not.

**Test calibration.** Simulate under a *true null* — a genetic correlation of
exactly zero — and count how often the likelihood ratio test rejects. Zero is an
interior point of [-1, 1], so a plain chi-squared on one degree of freedom is the
right reference and no boundary mixture applies. That makes it the clean case: a
test wrong here is wrong everywhere. Rejection rates are checked at three levels,
and the whole p-value distribution against uniform, which is the stronger
statement — three rates can look right while the distribution is not.

**The boundary test.** The same test against a correlation of exactly one, which
takes the other branch: the null sits on a bound, so the Self–Liang 50:50 mixture
applies rather than a plain chi-squared. It is a separate code path and it is the
one the Python reference historically got wrong, so it is calibrated separately.
Here only over-rejection fails — a boundary test that rejects too rarely
under-calls two traits as genetically distinct, which costs power rather than
manufacturing a finding.

**This is slow and meant to be.** The interval half fits four profile likelihoods
per replicate and takes roughly a quarter of an hour. It is a calibration, run
when the estimator changes rather than routinely.

Run with:

    uv run --no-project python checks/bivariate_calibration.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import asterism
import numpy as np
from scipy import stats

FAMILIES: int = 30
"""Fixed the number of independent simulated families."""

PER_FAMILY: int = 6
"""Fixed the number of people within each simulated family."""

INTERVAL_REPLICATES: int = 300
"""Fixed the number of replicates used to measure interval coverage."""

TEST_REPLICATES: int = 400
"""Fixed the number of replicates used to measure test calibration."""

LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)
"""Fixed the nominal rejection levels checked by the calibration."""

TRUTH: dict[str, float] = {"h1": 0.6, "h2": 0.35, "rg": 0.55, "re": 0.25}
"""Set the generating heritabilities and component correlations."""

QUANTITIES: tuple[str, ...] = (
    "h2_first",
    "h2_second",
    "rho_g",
    "rho_e",
    "rho_p",
)
"""Named every bivariate quantity whose interval coverage is measured."""


def phenotypic(h1: float, h2: float, rg: float, re_: float) -> float:
    """Return the phenotypic correlation implied by component quantities.

    Args:
        h1: First-trait heritability.
        h2: Second-trait heritability.
        rg: Genetic correlation.
        re_: Residual correlation.

    Returns:
        The phenotypic correlation when both total variances cancel.
    """
    return rg * np.sqrt(h1 * h2) + re_ * np.sqrt((1 - h1) * (1 - h2))


def relationship() -> np.ndarray:
    """Build unrelated families whose members are all half relatives.

    Returns:
        The block-diagonal additive relationship matrix.
    """
    n: int = FAMILIES * PER_FAMILY
    """Computed the simulated roster size from family counts."""

    k: np.ndarray = np.zeros((n, n))
    """Initialised the block-diagonal additive relationship matrix."""

    for family in range(FAMILIES):
        block: slice = slice(family * PER_FAMILY, (family + 1) * PER_FAMILY)
        """Located the roster rows belonging to one simulated family."""

        k[block, block] = 0.5
        """Set all within-family off-diagonal relationships to one half."""
    np.fill_diagonal(k, 1.0)
    return k


def factor(k: np.ndarray, h1: float, h2: float, rg: float, re_: float) -> np.ndarray:
    """Factor the stacked two-trait covariance for deterministic simulation.

    Args:
        k: Additive relationship matrix.
        h1: First-trait heritability.
        h2: Second-trait heritability.
        rg: Genetic correlation.
        re_: Residual correlation.

    Returns:
        Lower Cholesky factor of the interleaved full covariance.
    """
    genetic: np.ndarray = np.array(
        [[h1, rg * np.sqrt(h1 * h2)], [rg * np.sqrt(h1 * h2), h2]]
    )
    """Constructed the two-trait additive covariance at unit total variances."""

    residual: np.ndarray = np.array(
        [
            [1 - h1, re_ * np.sqrt((1 - h1) * (1 - h2))],
            [re_ * np.sqrt((1 - h1) * (1 - h2)), 1 - h2],
        ]
    )
    """Constructed the two-trait residual covariance at unit total variances."""

    n: int = k.shape[0]
    """Read the participant count from the relationship matrix."""

    full: np.ndarray = np.kron(genetic, k) + np.kron(residual, np.eye(n))
    """Combined additive and residual terms into the stacked full covariance."""
    return np.linalg.cholesky(full + 1e-9 * np.eye(2 * n))


def draw(chol: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Draw one data set in Asterism's person-then-trait order.

    Args:
        chol: Lower factor of the stacked two-trait covariance.
        n: Number of simulated people.
        seed: Deterministic random seed for this replicate.

    Returns:
        An interleaved two-trait response vector.
    """
    z: np.ndarray = chol @ np.random.default_rng(seed).standard_normal(2 * n)
    """Drew the two stacked traits from their joint covariance."""

    values: np.ndarray = np.empty(2 * n)
    """Initialised the interleaved response vector expected by Asterism."""

    values[0::2] = z[:n]
    """Placed first-trait values in the even response positions."""

    values[1::2] = z[n:]
    """Placed second-trait values in the odd response positions."""
    return values


def coverage(k: np.ndarray) -> dict[str, Any]:
    """Measure coverage for every bivariate interval.

    Args:
        k: Additive relationship matrix for the simulation design.

    Returns:
        Per-quantity attempt and containment counts plus generating truths.
    """
    n: int = k.shape[0]
    """Read the participant count from the relationship matrix."""

    chol: np.ndarray = factor(k, TRUTH["h1"], TRUTH["h2"], TRUTH["rg"], TRUTH["re"])
    """Factorised the covariance under the interval-generating truth."""

    design: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]] * n)
    """Constructed separate intercept columns for the two traits."""

    observed: list[list[bool]] = [[True, True]] * n
    """Marked both traits observed for every simulated person."""

    model: asterism.BivariateModel = asterism.BivariateModel(k, observed, design)
    """Built the public bivariate model reused across all interval replicates."""

    true_value: dict[str, float] = dict(
        zip(
            QUANTITIES,
            (
                TRUTH["h1"],
                TRUTH["h2"],
                TRUTH["rg"],
                TRUTH["re"],
                phenotypic(TRUTH["h1"], TRUTH["h2"], TRUTH["rg"], TRUTH["re"]),
            ),
            strict=True,
        )
    )
    """Mapped every interval quantity to the exact value used to generate data."""

    contained: dict[str, int] = dict.fromkeys(QUANTITIES, 0)
    """Initialised interval-containment counts for every reported quantity."""

    # Each quantity is counted over the replicates where *its own* interval was
    # computed. Sharing one denominator across all four is wrong: a replicate
    # that fails on the third interval has already contributed to the first two,
    # so their numerators advance while the shared denominator does not, and the
    # ratio can exceed one. It did -- 1.040 -- which is how this was found.
    attempted: dict[str, int] = dict.fromkeys(QUANTITIES, 0)
    """Initialised successful interval-attempt counts by quantity."""

    complete: int = 0
    """Initialised the number of replicates producing every interval."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement before the simulation loop."""

    for replicate in range(INTERVAL_REPLICATES):
        y: np.ndarray = draw(chol, n, 20000 + replicate)
        """Drew one deterministic response under the interval-generating truth."""

        whole: bool = True
        """Assumed every interval complete until a profile fit refused."""

        for quantity in QUANTITIES:
            try:
                interval: dict[str, Any] = model.interval(y, quantity, reml=True)
                """Computed the named public interval for this quantity."""
            except ValueError:
                # An interval that will not compute does not contribute to the
                # measured coverage. It is counted out rather than as a miss.
                whole = False
                """Marked the replicate incomplete after an interval refusal."""
                continue
            attempted[quantity] += 1
            """Counted this quantity's successfully computed interval."""

            if interval["lower"] <= true_value[quantity] <= interval["upper"]:
                contained[quantity] += 1
                """Counted an interval containing its generating quantity."""
        if whole:
            complete += 1
            """Counted a replicate that produced every required interval."""
        if (replicate + 1) % 50 == 0:
            print(
                f"  {replicate + 1} replicates, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {
        "complete": complete,
        "attempted": attempted,
        "contained": contained,
        "true_value": true_value,
    }


def calibration(k: np.ndarray) -> dict[str, list[float]]:
    """Measure the genetic-correlation test under an interior null.

    Args:
        k: Additive relationship matrix for the simulation design.

    Returns:
        P-values from every completed genetic-correlation test.
    """
    n: int = k.shape[0]
    """Read the participant count from the relationship matrix."""

    # The null is true: no genetic correlation at all. The residual correlation
    # is left non-zero so that the two traits are still related, which is the
    # case a test could most easily mistake for a genetic one.
    chol: np.ndarray = factor(k, TRUTH["h1"], TRUTH["h2"], 0.0, TRUTH["re"])
    """Factorised the covariance under a true zero genetic correlation."""

    design: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]] * n)
    """Constructed separate intercept columns for the two traits."""

    observed: list[list[bool]] = [[True, True]] * n
    """Marked both traits observed for every simulated person."""

    model: asterism.BivariateModel = asterism.BivariateModel(k, observed, design)
    """Built the public bivariate model reused across null replicates."""

    p_values: list[float] = []
    """Initialised p-values returned by completed interior-null tests."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement before the simulation loop."""

    for replicate in range(TEST_REPLICATES):
        y: np.ndarray = draw(chol, n, 70000 + replicate)
        """Drew one deterministic response under the interior null."""

        try:
            outcome: dict[str, Any] = model.test(y, "rho_g", 0.0, reml=True)
            """Tested the interior genetic-correlation null through named fields."""
        except ValueError:
            continue
        p_values.append(outcome["p_value"])
        if (replicate + 1) % 100 == 0:
            print(
                f"  {replicate + 1} replicates, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"p_values": p_values}


def boundary(k: np.ndarray) -> dict[str, Any]:
    """The other branch of the test: a null sitting on a bound.

    Against zero the correlation is interior and a plain chi-squared applies.
    Against plus or minus one it sits on a bound, and the Self-Liang 50:50
    mixture applies instead. That is a different approximation and is measured
    separately here.

    Simulating with the genetic correlation exactly one makes the genetic
    covariance singular -- rank one, the two traits sharing a single genetic
    factor. That is the hard case, and it is the one the Python reference
    historically got wrong: before the constrained Rust refits existed it
    rejected six times in ten at a nominal one in a hundred, which would have
    called two traits genetically distinct in most samples where they were
    genetically identical.

    Args:
        k: Additive relationship matrix for the simulation design.

    Returns:
        Completed-test p-values and the number of zero likelihood-ratio statistics.
    """
    n: int = k.shape[0]
    """Read the participant count from the relationship matrix."""

    chol: np.ndarray = factor(k, TRUTH["h1"], TRUTH["h2"], 1.0, TRUTH["re"])
    """Factorised the singular genetic covariance at correlation one."""

    design: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]] * n)
    """Constructed separate intercept columns for the two traits."""

    observed: list[list[bool]] = [[True, True]] * n
    """Marked both traits observed for every simulated person."""

    model: asterism.BivariateModel = asterism.BivariateModel(k, observed, design)
    """Built the public bivariate model reused across boundary replicates."""

    p_values: list[float] = []
    """Initialised p-values returned by completed boundary-null tests."""

    at_zero: int = 0
    """Initialised the count of zero likelihood-ratio statistics."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement before the simulation loop."""

    for replicate in range(TEST_REPLICATES):
        y: np.ndarray = draw(chol, n, 90000 + replicate)
        """Drew one deterministic response under the correlation-one null."""

        try:
            outcome: dict[str, Any] = model.test(y, "rho_g", 1.0, reml=True)
            """Tested the boundary genetic-correlation null through named fields."""
        except ValueError:
            continue
        p_values.append(outcome["p_value"])
        if outcome["statistic"] <= 0.0:
            at_zero += 1
            """Counted a likelihood-ratio statistic in the boundary atom."""
        if (replicate + 1) % 100 == 0:
            print(
                f"  {replicate + 1} replicates, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"p_values": p_values, "at_zero": at_zero}


def phenotypic_null(k: np.ndarray) -> dict[str, Any]:
    """The test of the derived correlation against zero, under a hard null.

    A phenotypic correlation of nought could come from both components being
    nought, which would test almost nothing — the substitution would never be
    exercised. Here the genetic and residual correlations are both well away
    from zero and cancel exactly, so the traits are genetically correlated and
    residually anti-correlated and only the sum is nought. A test that quietly
    ignored the substitution would pass the easy version of this and fail here.

    Args:
        k: Additive relationship matrix for the simulation design.

    Returns:
        Completed-test p-values and the cancelling generating correlations.
    """
    n: int = k.shape[0]
    """Read the participant count from the relationship matrix."""

    rho_g: float = 0.45
    """Set the non-zero genetic correlation used by the hard derived null."""

    rho_e: float = (
        -rho_g
        * np.sqrt(TRUTH["h1"] * TRUTH["h2"])
        / np.sqrt((1 - TRUTH["h1"]) * (1 - TRUTH["h2"]))
    )
    """Solved the residual correlation that exactly cancels the genetic term."""

    assert abs(phenotypic(TRUTH["h1"], TRUTH["h2"], rho_g, rho_e)) < 1e-15
    chol: np.ndarray = factor(k, TRUTH["h1"], TRUTH["h2"], rho_g, rho_e)
    """Factorised the covariance under the exact derived-correlation null."""

    design: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]] * n)
    """Constructed separate intercept columns for the two traits."""

    observed: list[list[bool]] = [[True, True]] * n
    """Marked both traits observed for every simulated person."""

    model: asterism.BivariateModel = asterism.BivariateModel(k, observed, design)
    """Built the public bivariate model reused across derived-null replicates."""

    p_values: list[float] = []
    """Initialised p-values returned by completed derived-null tests."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement before the simulation loop."""

    for replicate in range(TEST_REPLICATES):
        y: np.ndarray = draw(chol, n, 130000 + replicate)
        """Drew one deterministic response under the derived-correlation null."""

        try:
            outcome: dict[str, Any] = model.test(y, "rho_p", 0.0, reml=True)
            """Tested the derived phenotypic-correlation null through named fields."""
        except ValueError:
            continue
        p_values.append(outcome["p_value"])
        if (replicate + 1) % 100 == 0:
            print(
                f"  {replicate + 1} replicates, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"p_values": p_values, "rho_g": rho_g, "rho_e": rho_e}


def main() -> int:
    """Run interval and test campaigns and print their evidence receipt."""
    k: np.ndarray = relationship()
    """Built the fixed block-diagonal relationship design."""

    n: int = k.shape[0]
    """Read the simulated roster size from the relationship matrix."""

    failures: list[str] = []
    """Initialised the scientific pass-rule failure messages."""

    print(
        f"Two-trait calibration. {FAMILIES} families of {PER_FAMILY}, n = {n}, REML.\n"
        f"Truth: h2 {TRUTH['h1']} and {TRUTH['h2']}, "
        f"genetic correlation {TRUTH['rg']}, residual {TRUTH['re']}, "
        f"phenotypic {phenotypic(TRUTH['h1'], TRUTH['h2'], TRUTH['rg'], TRUTH['re']):.4f} "
        f"(derived).\n"
    )

    print("Profile interval coverage, nominal 95 per cent.")
    covered: dict[str, Any] = coverage(k)
    """Measured interval coverage across the fixed simulation campaign."""

    complete: int = covered["complete"]
    """Read the number of replicates producing every required interval."""

    if complete < INTERVAL_REPLICATES * 0.9:
        failures.append(
            f"only {complete} of {INTERVAL_REPLICATES} replicates gave a whole set "
            "of intervals"
        )
    print(f"\n  {complete} of {INTERVAL_REPLICATES} replicates gave every interval.\n")
    print(f"  {'quantity':<12}{'coverage':>10}{'binomial 95%':>22}{'of':>8}")
    interval_results: dict[str, float] = {}
    """Initialised the recorded quantity-specific coverage rates."""

    for quantity in QUANTITIES:
        counted: int = covered["attempted"][quantity]
        """Read the number of computed intervals for this quantity."""

        hit: float = covered["contained"][quantity] / counted
        """Computed this quantity's empirical interval coverage."""

        interval_results[quantity] = hit
        """Recorded the quantity's empirical interval coverage."""

        standard_error: float = float(np.sqrt(0.95 * 0.05 / counted))
        """Computed the Monte Carlo standard error at nominal coverage."""

        low, high = 0.95 - 1.96 * standard_error, 0.95 + 1.96 * standard_error
        """Constructed the two-sided binomial approximation band."""

        # Under the band is the direction that matters. Over it means the
        # interval is wider than it needs to be, which costs power and misleads
        # nobody, so it is reported and not failed.
        inside: bool = low <= hit <= high
        """Scored whether empirical coverage fell inside its sampling band."""

        print(
            f"  {quantity:<12}{hit:>10.3f}   [{low:.3f}, {high:.3f}]  "
            f"{'ok' if inside else ('conservative' if hit > high else 'UNDER')}"
            f"{counted:>8}"
        )
        if hit < low:
            failures.append(f"{quantity} covers {hit:.3f}, below the band")

    print("\n\nLikelihood ratio test for the genetic correlation, null true at zero.")
    calibrated: dict[str, list[float]] = calibration(k)
    """Measured genetic-correlation testing under the true interior null."""

    p_values: np.ndarray = np.array(calibrated["p_values"])
    """Converted completed interior-null p-values to a numerical vector."""

    replicates: int = len(p_values)
    """Counted interior-null replicates that produced a usable test."""

    if replicates < TEST_REPLICATES * 0.9:
        failures.append(f"only {replicates} of {TEST_REPLICATES} replicates tested")
    print(f"\n  {replicates} of {TEST_REPLICATES} replicates tested.\n")
    if replicates == 0:
        print("  Nothing to report: no replicate produced a p-value.")
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    rates: dict[str, float] = {}
    """Initialised empirical interior-null rejection rates by nominal level."""

    for level in LEVELS:
        rate: float = float((p_values < level).mean())
        """Computed the empirical interior-null rejection rate at this level."""

        rates[str(level)] = rate
        """Recorded the rejection rate using the level's stable string key."""

        error: float = float(np.sqrt(level * (1 - level) / replicates))
        """Computed the binomial Monte Carlo standard error for this level."""

        lower, upper = level - 1.96 * error, level + 1.96 * error
        """Constructed the two-sided binomial approximation band."""

        inside = lower <= rate <= upper
        """Scored whether the rejection rate fell inside its sampling band."""

        print(
            f"  {level:>8.2f}{rate:>12.3f}   [{lower:.3f}, {upper:.3f}]  "
            f"{'ok' if inside else 'OUT'}"
        )
        if not inside:
            failures.append(
                f"rejects {rate:.3f} at the {level:.2f} level, outside the band"
            )

    # Three rates can look right while the distribution behind them is not, so
    # the whole thing is checked against uniform as well.
    test: Any = stats.kstest(p_values, "uniform")
    """Compared the complete interior-null p-value distribution with uniformity."""
    print(
        f"\n  Kolmogorov-Smirnov against uniform: D = {test.statistic:.4f}, "
        f"p = {test.pvalue:.3f}"
    )
    if test.pvalue < 0.01:
        failures.append(f"p-values are not uniform, KS p = {test.pvalue:.4f}")

    print("\n\nLikelihood ratio test at the boundary, null true at one.")
    edge: dict[str, Any] = boundary(k)
    """Measured genetic-correlation testing under the true boundary null."""

    edge_p: np.ndarray = np.array(edge["p_values"])
    """Converted completed boundary-null p-values to a numerical vector."""

    edge_n: int = len(edge_p)
    """Counted boundary-null replicates that produced a usable test."""

    print(f"\n  {edge_n} of {TEST_REPLICATES} replicates tested.\n")
    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    edge_rates: dict[str, float] = {}
    """Initialised empirical boundary-null rejection rates by nominal level."""

    for level in LEVELS:
        rate = float((edge_p < level).mean())
        """Computed the empirical boundary-null rejection rate at this level."""

        edge_rates[str(level)] = rate
        """Recorded the boundary rejection rate using the level's stable key."""

        error = np.sqrt(level * (1 - level) / edge_n)
        """Computed the binomial Monte Carlo standard error for this level."""

        lower, upper = level - 1.96 * error, level + 1.96 * error
        """Constructed the two-sided binomial approximation band."""

        # Only over-rejection fails. A boundary test that rejects too rarely
        # under-calls two traits as genetically distinct, which is the direction
        # that costs power rather than the one that manufactures a finding.
        print(
            f"  {level:>8.2f}{rate:>12.3f}   [{lower:.3f}, {upper:.3f}]  "
            f"{'ok' if lower <= rate <= upper else ('conservative' if rate < lower else 'OVER')}"
        )
        if rate > upper:
            failures.append(
                f"boundary test rejects {rate:.3f} at the {level:.2f} level, "
                "above the band"
            )
    atom: float = edge["at_zero"] / edge_n
    """Measured the boundary mixture's empirical probability mass at zero."""

    print(
        f"\n  Statistic exactly zero in {atom:.1%} of samples; the mixture "
        f"expects about 50%."
    )
    if atom < 0.35:
        failures.append(f"the atom at zero is {atom:.1%}, too small for the mixture")

    print("\n\nLikelihood ratio test for the derived correlation, null true at zero.")
    derived: dict[str, Any] = phenotypic_null(k)
    """Measured the phenotypic-correlation test under an exact derived null."""

    derived_p: np.ndarray = np.array(derived["p_values"])
    """Converted completed derived-null p-values to a numerical vector."""

    derived_n: int = len(derived_p)
    """Counted derived-null replicates that produced a usable test."""

    print(
        f"\n  Genetic correlation {derived['rho_g']:.2f} and residual "
        f"{derived['rho_e']:.4f} cancel exactly.\n"
        f"  {derived_n} of {TEST_REPLICATES} replicates tested.\n"
    )
    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    derived_rates: dict[str, float] = {}
    """Initialised empirical derived-null rejection rates by nominal level."""

    for level in LEVELS:
        rate = float((derived_p < level).mean())
        """Computed the empirical derived-null rejection rate at this level."""

        derived_rates[str(level)] = rate
        """Recorded the derived rejection rate using the level's stable key."""

        error = np.sqrt(level * (1 - level) / derived_n)
        """Computed the binomial Monte Carlo standard error for this level."""

        lower, upper = level - 1.96 * error, level + 1.96 * error
        """Constructed the two-sided binomial approximation band."""

        inside = lower <= rate <= upper
        """Scored whether the rejection rate fell inside its sampling band."""

        print(
            f"  {level:>8.2f}{rate:>12.3f}   [{lower:.3f}, {upper:.3f}]  "
            f"{'ok' if inside else 'OUT'}"
        )
        if not inside:
            failures.append(
                f"derived-correlation test rejects {rate:.3f} at the {level:.2f} "
                "level, outside the band"
            )
    derived_ks: Any = stats.kstest(derived_p, "uniform")
    """Compared the complete derived-null p-value distribution with uniformity."""
    print(
        f"\n  Kolmogorov-Smirnov against uniform: D = {derived_ks.statistic:.4f}, "
        f"p = {derived_ks.pvalue:.3f}"
    )
    if derived_ks.pvalue < 0.01:
        failures.append(
            f"derived-correlation p-values are not uniform, KS p = {derived_ks.pvalue:.4f}"
        )

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nIntervals cover and the test holds its level.")
    print("This is calibration, which is a different question from the agreement")
    print("with SOLAR and R — those say the numbers match, this says the")
    print("interval coverage and rejection rates match their stated levels.")

    print(
        json.dumps(
            {
                "families": FAMILIES,
                "people_per_family": PER_FAMILY,
                "people": n,
                "estimator": "reml",
                "truth": TRUTH,
                "intervals": {
                    "replicates_requested": INTERVAL_REPLICATES,
                    "replicates_complete": complete,
                    "replicates_by_quantity": covered["attempted"],
                    "nominal": 0.95,
                    "coverage": interval_results,
                },
                "derived_correlation_test": {
                    "replicates_requested": TEST_REPLICATES,
                    "replicates_tested": derived_n,
                    "null": {"quantity": "rho_p", "value": 0.0, "interior": True},
                    "simulated_under": {
                        "rho_g": derived["rho_g"],
                        "rho_e": derived["rho_e"],
                        "note": "non-zero components cancelling exactly",
                    },
                    "rejection_rates": derived_rates,
                    "kolmogorov_smirnov": {
                        "statistic": derived_ks.statistic,
                        "p_value": derived_ks.pvalue,
                    },
                },
                "correlation_test_at_the_boundary": {
                    "replicates_requested": TEST_REPLICATES,
                    "replicates_tested": edge_n,
                    "null": {"quantity": "rho_g", "value": 1.0, "interior": False},
                    "rule": "self_liang_50_50_mixture",
                    "rejection_rates": edge_rates,
                    "atom_at_zero": atom,
                },
                "correlation_test": {
                    "replicates_requested": TEST_REPLICATES,
                    "replicates_tested": replicates,
                    "null": {"quantity": "rho_g", "value": 0.0, "interior": True},
                    "rejection_rates": rates,
                    "kolmogorov_smirnov": {
                        "statistic": test.statistic,
                        "p_value": test.pvalue,
                    },
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
