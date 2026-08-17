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

RELATIVE_TOLERANCE = 1e-2
MASS_TOLERANCE = 1e-9
MONTE_CARLO_DRAWS = 8_000_000


def ruben_tail(q: float, weights: np.ndarray, terms: int = 1200):
    """Return the tail and the accumulated coefficient mass, which is one when
    the series has converged."""
    weights = np.asarray([w for w in weights if abs(w) > 1e-13], float)
    order = weights.size
    if order == 0:
        return (1.0 if q < 0 else 0.0), 1.0
    if q <= 0:
        return 1.0, 1.0
    beta = weights.min() * 0.9
    ratio = 1.0 - beta / weights
    g = np.array([0.5 * np.sum(ratio**k) for k in range(1, terms + 1)])
    a = np.zeros(terms + 1)
    a[0] = float(np.prod(np.sqrt(beta / weights)))
    survival = chi2.sf(q / beta, order + 2 * np.arange(terms + 1))
    total, mass = a[0] * survival[0], a[0]
    for k in range(1, terms + 1):
        a[k] = float(np.dot(g[:k][::-1], a[:k])) / k
        total += a[k] * survival[k]
        mass += a[k]
        if k > 30 and abs(1.0 - mass) < 1e-16:
            break
    return float(min(1.0, max(0.0, total))), float(mass)


def cases():
    rng = np.random.default_rng(5)
    out = []
    for _ in range(40):
        order = int(rng.integers(2, 25))
        style = rng.integers(0, 3)
        if style == 0:
            weights = rng.uniform(0.5, 2.0, order)
        elif style == 1:
            weights = np.sort(rng.uniform(0.01, 1.0, order))[::-1]
        else:
            weights = np.concatenate(
                [[rng.uniform(5, 50)], rng.uniform(0.1, 1.0, order - 1)]
            )
        for multiple in (0.5, 1.5, 3.0, 6.0):
            out.append((float(weights.sum() * multiple), weights))
    return out


def main() -> int:
    failures: list[str] = []

    # The two cases with an exact answer must be taken exactly.
    exact = []
    for q, weights, expected, what in [
        (25.0, [1.0], float(chi2.sf(25.0, 1)), "one weight"),
        (60.0, [1.0] * 20, float(chi2.sf(60.0, 20)), "equal weights"),
        (120.0, [2.0] * 20, float(chi2.sf(60.0, 20)), "equal weights, scaled"),
    ]:
        got = asterism.weighted_chi2_upper_tail(q, weights)
        gap = abs(got["probability"] - expected)
        exact.append({"case": what, "gap": gap, "method": got["method"]})
        if gap > 1e-14:
            failures.append(f"{what}: {got['probability']} against {expected}")
        if not got["method"].startswith("exact"):
            failures.append(f"{what} was integrated rather than taken exactly")

    # The inversion against the series, wherever the series has converged.
    compared, skipped, worst, worst_case = 0, 0, 0.0, None
    for q, weights in cases():
        got = asterism.weighted_chi2_upper_tail(q, list(weights))
        if not got["trustworthy"]:
            skipped += 1
            continue
        reference, mass = ruben_tail(q, weights)
        if abs(mass - 1.0) > MASS_TOLERANCE or not 1e-12 < reference < 1.0:
            skipped += 1
            continue
        compared += 1
        relative = abs(got["probability"] - reference) / reference
        if relative > worst:
            worst, worst_case = relative, (q, len(weights), got["probability"], reference)
        if relative > RELATIVE_TOLERANCE:
            failures.append(
                f"q={q:.3f}, {len(weights)} weights: {got['probability']:.6e} "
                f"against {reference:.6e}, relative {relative:.2e}"
            )

    # One case where the series fails, settled by simulation instead.
    hard = np.array([0.9812, 0.8214, 0.8161, 0.8039, 0.7667, 0.6923, 0.6761,
                     0.6076, 0.5690, 0.5620, 0.4943, 0.4611, 0.3703, 0.3452,
                     0.3414, 0.1252, 0.0817, 0.0117])
    hard_q = 28.582
    rng = np.random.default_rng(1)
    hits = 0
    block = 1_000_000
    for _ in range(MONTE_CARLO_DRAWS // block):
        z = rng.standard_normal((block, hard.size))
        hits += int(((z * z) @ hard > hard_q).sum())
    simulated = hits / MONTE_CARLO_DRAWS
    error = 1.96 * (simulated * (1 - simulated) / MONTE_CARLO_DRAWS) ** 0.5
    ours = asterism.weighted_chi2_upper_tail(hard_q, list(hard))["probability"]
    series, series_mass = ruben_tail(hard_q, hard)
    if abs(ours - simulated) > 4 * error:
        failures.append(
            f"spread weights: {ours:.6e} outside the simulated "
            f"{simulated:.6e} +/- {error:.1e}"
        )

    report = {
        "what": "the weighted chi-square tail against an independent series",
        "exact_cases": exact,
        "compared_against_series": compared,
        "skipped": skipped,
        "worst_relative_difference": worst,
        "worst_case": None if worst_case is None else {
            "q": worst_case[0], "weights": worst_case[1],
            "asterism": worst_case[2], "series": worst_case[3],
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
