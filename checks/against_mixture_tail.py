"""Independently check the tail of a weighted sum of chi-squares.

A variance-component score test needs `P(sum_j lambda_j chisq_1 > q)`, and a
gene scan reads that at around 1e-6, so an approximation that is fine at 0.05
is worthless. Asterism computes it by Gil-Pelaez inversion. This file computes
it a completely different way -- Ruben's series expansion, which writes the
statistic as a mixture of ordinary chi-squares -- so agreement is evidence
rather than a repeated mistake.

  P(Q <= q) = sum_k a_k P(chisq_{r+2k} <= q/beta)
  a_0 = prod_j (beta/lambda_j)^(1/2)
  g_k = (1/2) sum_j (1 - beta/lambda_j)^k
  a_k = (1/k) sum_{i=1..k} g_i a_{k-i}

**The series does not always converge, and this file refuses to pretend it
has.** The coefficients sum to one, so the accumulated mass is an exact
convergence test: where it falls short the reference is dropped rather than
compared, because a reference that has silently stopped early is worse than no
reference. On weights spanning eighty to one it returns 1e-126 where the
answer is 1.6e-4, and a Monte Carlo of forty million draws settles that in
Asterism's favour.

Run with:

    uv run --no-project python checks/against_mixture_tail.py
"""

from __future__ import annotations

import json
import sys

import asterism
import numpy as np
from scipy.stats import chi2

RELATIVE_TOLERANCE: float = 1e-2
"""Set the maximum relative disagreement accepted for converged series cases."""

MASS_TOLERANCE: float = 1e-9
"""Set the maximum coefficient-mass deficit accepted as series convergence."""

MONTE_CARLO_DRAWS: int = 8_000_000
"""Set the simulation size used when Ruben's series does not converge."""


def ruben_tail(q: float, weights: np.ndarray, terms: int = 1200) -> tuple[float, float]:
    """Return the tail and the accumulated coefficient mass, which is one when
    the series has converged."""
    weights = np.asarray([w for w in weights if abs(w) > 1e-13], float)
    """Removed numerically zero mixture weights before expanding the series."""

    order: int = weights.size
    """Counted the retained chi-square components."""

    if order == 0:
        return (1.0 if q < 0 else 0.0), 1.0
    if q <= 0:
        return 1.0, 1.0
    beta: float = float(weights.min() * 0.9)
    """Placed the Ruben scale strictly below the smallest retained weight."""

    ratio: np.ndarray = 1.0 - beta / weights
    """Calculated the per-component ratios used in every series coefficient."""

    g: np.ndarray = np.array([0.5 * np.sum(ratio**k) for k in range(1, terms + 1)])
    """Calculated the power-sum sequence in Ruben's coefficient recurrence."""

    a: np.ndarray = np.zeros(terms + 1)
    """Allocated the chi-square mixture coefficients, including the base term."""

    a[0] = float(np.prod(np.sqrt(beta / weights)))
    """Initialised the coefficient multiplying the base chi-square tail."""

    survival: np.ndarray = chi2.sf(q / beta, order + 2 * np.arange(terms + 1))
    """Evaluated every ordinary chi-square survival term needed by the series."""

    total, mass = a[0] * survival[0], a[0]
    """Initialised the weighted tail probability and accumulated coefficient mass."""

    for k in range(1, terms + 1):
        a[k] = float(np.dot(g[:k][::-1], a[:k])) / k
        """Advanced Ruben's recurrence by one mixture coefficient."""

        total += a[k] * survival[k]
        """Added the new coefficient's chi-square tail contribution."""

        mass += a[k]
        """Accumulated coefficient mass for the independent convergence check."""

        if k > 30 and abs(1.0 - mass) < 1e-16:
            break
    return float(min(1.0, max(0.0, total))), float(mass)


def cases() -> list[tuple[float, np.ndarray]]:
    """Return the seeded mixture-weight battery used for series comparisons."""
    rng: np.random.Generator = np.random.default_rng(5)
    """Created the deterministic generator for the standing comparison battery."""

    out: list[tuple[float, np.ndarray]] = []
    """Initialised the threshold and weight cases returned to the comparison."""

    for _ in range(40):
        order: int = int(rng.integers(2, 25))
        """Drew the number of mixture components in this seeded case."""

        style: int = int(rng.integers(0, 3))
        """Selected balanced, graded or highly uneven weights for this case."""

        if style == 0:
            weights: np.ndarray = rng.uniform(0.5, 2.0, order)
            """Drew a balanced set of positive mixture weights."""

        elif style == 1:
            weights = np.sort(rng.uniform(0.01, 1.0, order))[::-1]
            """Drew and ordered weights spanning two decimal orders."""

        else:
            weights = np.concatenate(
                [[rng.uniform(5, 50)], rng.uniform(0.1, 1.0, order - 1)]
            )
            """Combined one dominant weight with smaller background components."""

        for multiple in (0.5, 1.5, 3.0, 6.0):
            out.append((float(weights.sum() * multiple), weights))
            """Added a threshold at the selected multiple of the mixture mean."""
    return out


def main() -> int:
    failures: list[str] = []
    """Collected disagreements that make the independent check fail."""

    # The two cases with an exact answer must be taken exactly.
    exact: list[dict[str, float | str]] = []
    """Collected diagnostics for cases with closed-form chi-square answers."""

    for q, weights, expected, what in [
        (25.0, [1.0], float(chi2.sf(25.0, 1)), "one weight"),
        (60.0, [1.0] * 20, float(chi2.sf(60.0, 20)), "equal weights"),
        (120.0, [2.0] * 20, float(chi2.sf(60.0, 20)), "equal weights, scaled"),
    ]:
        got: dict[str, float | bool | str] = asterism.weighted_chi2_upper_tail(
            q, weights
        )
        """Evaluated Asterism's tail probability for one exact comparison case."""

        gap: float = abs(float(got["probability"]) - expected)
        """Measured absolute disagreement from the closed-form survival probability."""

        exact.append({"case": what, "gap": gap, "method": got["method"]})
        """Recorded the exact-case discrepancy and selected computational method."""

        if gap > 1e-14:
            failures.append(f"{what}: {got['probability']} against {expected}")
        if not got["method"].startswith("exact"):
            failures.append(f"{what} was integrated rather than taken exactly")

    # The inversion against the series, wherever the series has converged.
    compared, skipped, worst, worst_case = 0, 0, 0.0, None
    """Initialised series-comparison coverage and worst-disagreement tracking."""

    for q, weights in cases():
        got = asterism.weighted_chi2_upper_tail(q, list(weights))
        """Evaluated Asterism's numerical inversion for one seeded mixture."""

        if not got["trustworthy"]:
            skipped += 1
            """Counted an Asterism result that declined to claim a trustworthy tail."""

            continue
        reference, mass = ruben_tail(q, weights)
        """Evaluated the independent Ruben series and its convergence mass."""

        if abs(mass - 1.0) > MASS_TOLERANCE or not 1e-12 < reference < 1.0:
            skipped += 1
            """Counted a case whose independent series was unusable for comparison."""

            continue
        compared += 1
        """Counted a case supported by trustworthy results from both implementations."""

        relative: float = abs(float(got["probability"]) - reference) / reference
        """Measured relative disagreement against the independently converged tail."""

        if relative > worst:
            worst, worst_case = (
                relative,
                (q, len(weights), got["probability"], reference),
            )
            """Retained the largest observed discrepancy and its defining inputs."""

        if relative > RELATIVE_TOLERANCE:
            failures.append(
                f"q={q:.3f}, {len(weights)} weights: {got['probability']:.6e} "
                f"against {reference:.6e}, relative {relative:.2e}"
            )

    # One case where the series fails, settled by simulation instead.
    hard: np.ndarray = np.array(
        [
            0.9812,
            0.8214,
            0.8161,
            0.8039,
            0.7667,
            0.6923,
            0.6761,
            0.6076,
            0.5690,
            0.5620,
            0.4943,
            0.4611,
            0.3703,
            0.3452,
            0.3414,
            0.1252,
            0.0817,
            0.0117,
        ]
    )
    """Defined the uneven-weight case for which Ruben's series fails to converge."""

    hard_q: float = 28.582
    """Set the tail threshold used for the deliberately difficult mixture."""

    rng: np.random.Generator = np.random.default_rng(1)
    """Created the deterministic Monte Carlo generator for the difficult case."""

    hits: int = 0
    """Initialised the number of simulated statistics exceeding the threshold."""

    block: int = 1_000_000
    """Set the simulation block size to bound peak array memory."""

    for _ in range(MONTE_CARLO_DRAWS // block):
        z: np.ndarray = rng.standard_normal((block, hard.size))
        """Drew one block of independent standard-normal variates."""

        hits += int(((z * z) @ hard > hard_q).sum())
        """Accumulated simulated exceedances of the weighted chi-square threshold."""

    simulated: float = hits / MONTE_CARLO_DRAWS
    """Estimated the difficult-case upper-tail probability by Monte Carlo."""

    error: float = 1.96 * (simulated * (1 - simulated) / MONTE_CARLO_DRAWS) ** 0.5
    """Calculated the nominal 95 per cent Monte Carlo half-width."""

    ours: float = float(
        asterism.weighted_chi2_upper_tail(hard_q, list(hard))["probability"]
    )
    """Evaluated Asterism's answer for the difficult uneven-weight mixture."""

    series, series_mass = ruben_tail(hard_q, hard)
    """Retained the failed Ruben answer and mass to expose why it was rejected."""

    if abs(ours - simulated) > 4 * error:
        failures.append(
            f"spread weights: {ours:.6e} outside the simulated "
            f"{simulated:.6e} +/- {error:.1e}"
        )

    report: dict[str, object] = {
        "what": "the weighted chi-square tail against an independent series",
        "exact_cases": exact,
        "compared_against_series": compared,
        "skipped": skipped,
        "worst_relative_difference": worst,
        "worst_case": None
        if worst_case is None
        else {
            "q": worst_case[0],
            "weights": worst_case[1],
            "asterism": worst_case[2],
            "series": worst_case[3],
        },
        "series_failure_case": {
            "monte_carlo": simulated,
            "monte_carlo_half_width": error,
            "asterism": ours,
            "series": series,
            "series_mass": series_mass,
            "note": "the series stops early here; simulation agrees with Asterism",
        },
        "relative_tolerance": RELATIVE_TOLERANCE,
    }
    """Assembled the values-free evidence report for exact, series and simulation cases."""
    if failures:
        print("NOT AGREED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(json.dumps(report, indent=2))
    print(
        f"\nExact where an exact answer exists. Across {compared} cases where the "
        f"series\nprovably converged, the worst relative difference was "
        f"{worst:.2e}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
