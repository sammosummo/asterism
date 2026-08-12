"""Are the two-trait intervals and tests calibrated?

`docs/adr/0006` separates two questions that are easy to run together. The
comparisons against SOLAR and R ask whether Asterism computes the same numbers as
software that is already trusted — fidelity. This asks something they cannot:
whether the uncertainty around those numbers means what it says. Agreement with
SOLAR to seven figures says nothing about whether a 95 per cent interval contains
the truth 95 per cent of the time.

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
from pathlib import Path

import numpy as np
from scipy import stats

from asterism import _core

FAMILIES, PER_FAMILY = 30, 6
INTERVAL_REPLICATES = 300
TEST_REPLICATES = 400
LEVELS = (0.01, 0.05, 0.10)

TRUTH = dict(h1=0.6, h2=0.35, rg=0.55, re=0.25)
QUANTITIES = ("h2_first", "h2_second", "rho_g", "rho_e")


def relationship() -> np.ndarray:
    """Unrelated families of six, everybody within a family a half relative."""
    n = FAMILIES * PER_FAMILY
    k = np.zeros((n, n))
    for family in range(FAMILIES):
        block = slice(family * PER_FAMILY, (family + 1) * PER_FAMILY)
        k[block, block] = 0.5
    np.fill_diagonal(k, 1.0)
    return k


def factor(k: np.ndarray, h1: float, h2: float, rg: float, re_: float) -> np.ndarray:
    genetic = np.array([[h1, rg * np.sqrt(h1 * h2)], [rg * np.sqrt(h1 * h2), h2]])
    residual = np.array(
        [
            [1 - h1, re_ * np.sqrt((1 - h1) * (1 - h2))],
            [re_ * np.sqrt((1 - h1) * (1 - h2)), 1 - h2],
        ]
    )
    n = k.shape[0]
    full = np.kron(genetic, k) + np.kron(residual, np.eye(n))
    return np.linalg.cholesky(full + 1e-9 * np.eye(2 * n))


def draw(chol: np.ndarray, n: int, seed: int) -> np.ndarray:
    """One data set, interleaved as Asterism wants: person by person, trait
    within person."""
    z = chol @ np.random.default_rng(seed).standard_normal(2 * n)
    values = np.empty(2 * n)
    values[0::2] = z[:n]
    values[1::2] = z[n:]
    return values


def coverage(k: np.ndarray) -> dict:
    n = k.shape[0]
    chol = factor(k, TRUTH["h1"], TRUTH["h2"], TRUTH["rg"], TRUTH["re"])
    design = np.array([[1.0, 0.0], [0.0, 1.0]] * n)
    observed = [[True, True]] * n
    true_value = dict(
        zip(QUANTITIES, (TRUTH["h1"], TRUTH["h2"], TRUTH["rg"], TRUTH["re"]))
    )

    contained = {q: 0 for q in QUANTITIES}
    # Each quantity is counted over the replicates where *its own* interval was
    # computed. Sharing one denominator across all four is wrong: a replicate
    # that fails on the third interval has already contributed to the first two,
    # so their numerators advance while the shared denominator does not, and the
    # ratio can exceed one. It did -- 1.040 -- which is how this was found.
    attempted = {q: 0 for q in QUANTITIES}
    complete = 0
    started = time.perf_counter()
    for replicate in range(INTERVAL_REPLICATES):
        y = draw(chol, n, 20000 + replicate)
        whole = True
        for quantity in QUANTITIES:
            try:
                lower, upper, _, _, _ = _core.bivariate_interval(
                    k, observed, design, y, quantity, True
                )
            except Exception:
                # An interval that will not compute is not evidence about
                # coverage. It is counted out rather than counted as a miss.
                whole = False
                continue
            attempted[quantity] += 1
            if lower <= true_value[quantity] <= upper:
                contained[quantity] += 1
        if whole:
            complete += 1
        if (replicate + 1) % 50 == 0:
            print(
                f"  {replicate + 1} replicates, "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {
        "complete": complete,
        "attempted": attempted,
        "contained": contained,
        "true_value": true_value,
    }


def calibration(k: np.ndarray) -> dict:
    n = k.shape[0]
    # The null is true: no genetic correlation at all. The residual correlation
    # is left non-zero so that the two traits are still related, which is the
    # case a test could most easily mistake for a genetic one.
    chol = factor(k, TRUTH["h1"], TRUTH["h2"], 0.0, TRUTH["re"])
    design = np.array([[1.0, 0.0], [0.0, 1.0]] * n)
    observed = [[True, True]] * n

    p_values = []
    started = time.perf_counter()
    for replicate in range(TEST_REPLICATES):
        y = draw(chol, n, 70000 + replicate)
        try:
            _, p_value, _, _ = _core.bivariate_correlation_test(
                k, observed, design, y, "rho_g", 0.0, True
            )
            p_values.append(p_value)
        except Exception:
            pass
        if (replicate + 1) % 100 == 0:
            print(
                f"  {replicate + 1} replicates, "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"p_values": p_values}


def boundary(k: np.ndarray) -> dict:
    """The other branch of the test: a null sitting on a bound.

    Against zero the correlation is interior and a plain chi-squared applies.
    Against plus or minus one it sits on a bound, and the Self-Liang 50:50
    mixture applies instead (`docs/adr/0001` decision 29). That is a different
    code path and it needs its own calibration.

    Simulating with the genetic correlation exactly one makes the genetic
    covariance singular -- rank one, the two traits sharing a single genetic
    factor. That is the hard case, and it is the one the Python reference
    historically got wrong: before the constrained Rust refits existed it
    rejected six times in ten at a nominal one in a hundred, which would have
    called two traits genetically distinct in most samples where they were
    genetically identical.
    """
    n = k.shape[0]
    chol = factor(k, TRUTH["h1"], TRUTH["h2"], 1.0, TRUTH["re"])
    design = np.array([[1.0, 0.0], [0.0, 1.0]] * n)
    observed = [[True, True]] * n

    p_values, at_zero = [], 0
    started = time.perf_counter()
    for replicate in range(TEST_REPLICATES):
        y = draw(chol, n, 90000 + replicate)
        try:
            statistic, p_value, _, _ = _core.bivariate_correlation_test(
                k, observed, design, y, "rho_g", 1.0, True
            )
            p_values.append(p_value)
            if statistic <= 0.0:
                at_zero += 1
        except Exception:
            pass
        if (replicate + 1) % 100 == 0:
            print(
                f"  {replicate + 1} replicates, "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return {"p_values": p_values, "at_zero": at_zero}


def main() -> int:
    k = relationship()
    n = k.shape[0]
    failures = []

    print(
        f"Two-trait calibration. {FAMILIES} families of {PER_FAMILY}, n = {n}, REML.\n"
        f"Truth: h2 {TRUTH['h1']} and {TRUTH['h2']}, "
        f"genetic correlation {TRUTH['rg']}, residual {TRUTH['re']}.\n"
    )

    print("Profile interval coverage, nominal 95 per cent.")
    covered = coverage(k)
    complete = covered["complete"]
    if complete < INTERVAL_REPLICATES * 0.9:
        failures.append(
            f"only {complete} of {INTERVAL_REPLICATES} replicates gave a whole set "
            "of intervals"
        )
    print(
        f"\n  {complete} of {INTERVAL_REPLICATES} replicates gave all four "
        "intervals.\n"
    )
    print(f"  {'quantity':<12}{'coverage':>10}{'binomial 95%':>22}{'of':>8}")
    interval_results = {}
    for quantity in QUANTITIES:
        counted = covered["attempted"][quantity]
        hit = covered["contained"][quantity] / counted
        interval_results[quantity] = hit
        standard_error = np.sqrt(0.95 * 0.05 / counted)
        low, high = 0.95 - 1.96 * standard_error, 0.95 + 1.96 * standard_error
        # Under the band is the direction that matters. Over it means the
        # interval is wider than it needs to be, which costs power and misleads
        # nobody, so it is reported and not failed.
        inside = low <= hit <= high
        print(
            f"  {quantity:<12}{hit:>10.3f}   [{low:.3f}, {high:.3f}]  "
            f"{'ok' if inside else ('conservative' if hit > high else 'UNDER')}"
            f"{counted:>8}"
        )
        if hit < low:
            failures.append(f"{quantity} covers {hit:.3f}, below the band")

    print("\n\nLikelihood ratio test for the genetic correlation, null true at zero.")
    calibrated = calibration(k)
    p_values = np.array(calibrated["p_values"])
    replicates = len(p_values)
    if replicates < TEST_REPLICATES * 0.9:
        failures.append(f"only {replicates} of {TEST_REPLICATES} replicates tested")
    print(f"\n  {replicates} of {TEST_REPLICATES} replicates tested.\n")
    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    rates = {}
    for level in LEVELS:
        rate = float((p_values < level).mean())
        rates[str(level)] = rate
        error = np.sqrt(level * (1 - level) / replicates)
        lower, upper = level - 1.96 * error, level + 1.96 * error
        inside = lower <= rate <= upper
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
    test = stats.kstest(p_values, "uniform")
    print(
        f"\n  Kolmogorov-Smirnov against uniform: D = {test.statistic:.4f}, "
        f"p = {test.pvalue:.3f}"
    )
    if test.pvalue < 0.01:
        failures.append(f"p-values are not uniform, KS p = {test.pvalue:.4f}")

    print("\n\nLikelihood ratio test at the boundary, null true at one.")
    edge = boundary(k)
    edge_p = np.array(edge["p_values"])
    edge_n = len(edge_p)
    print(f"\n  {edge_n} of {TEST_REPLICATES} replicates tested.\n")
    print(f"  {'level':>8}{'rejected':>12}{'binomial 95%':>22}")
    edge_rates = {}
    for level in LEVELS:
        rate = float((edge_p < level).mean())
        edge_rates[str(level)] = rate
        error = np.sqrt(level * (1 - level) / edge_n)
        lower, upper = level - 1.96 * error, level + 1.96 * error
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
    atom = edge["at_zero"] / edge_n
    print(
        f"\n  Statistic exactly zero in {atom:.1%} of samples; the mixture "
        f"expects about 50%."
    )
    if atom < 0.35:
        failures.append(f"the atom at zero is {atom:.1%}, too small for the mixture")

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nIntervals cover and the test holds its level.")
    print("This is calibration, which is a different question from the agreement")
    print("with SOLAR and R — those say the numbers match, this says the")
    print("uncertainty around them means what it claims (`docs/adr/0006`).")

    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/bivariate-calibration-2026-08-12.json").write_text(
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
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
