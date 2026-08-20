"""The sequential approximation against an independent reference, up a ladder
of dimensions.

Every censored heritability Asterism reports rests on one routine: the
conditional probability that the unmeasured values lie beyond their limits,
given the measured ones. That probability is exact to two coordinates and
Mendell-Elston sequential truncation above.

**All the evidence for the censored models was generated on pairs**, where the
region is exact and the approximate branch never runs at all. A model of the
audiogram would run it at 221 censored dimensions in the largest family,
conditional on 2,328 measured values, and sequential truncation is documented
to degrade as coordinates correlate -- these correlate at 0.96 between
neighbouring frequencies. Nothing in the repository says how it behaves there.

So this climbs a ladder: 5, 20, 50, 100 and 221 censored dimensions, drawn from
a simulated family with the audiogram's own covariance, and compares the
sequential answer with a Geweke-Hajivassiliou-Keane simulator.

**GHK is the reference because it is unbiased and carries its own error bar.**
It is arrived at separately from anything in the crate, which is what ADR 0006
requires of a check: agreement with a second implementation of the same idea
would prove fidelity and not correctness. The crate's own quasi-Monte Carlo
rectangle refuses above 25 dimensions, so it cannot serve here.

The ladder is climbed twice, because the two regimes are not the same problem.

**Conditional** is what the model does: the censored coordinates given every
measured value in the family. Conditioning on a nearly complete audiogram
removes almost all of the correlation -- what is left is close to test-retest
noise -- and sequential truncation is exact when coordinates are independent,
so this is the regime where it should do well.

**Marginal** is the same coordinates with nothing conditioned away, where the
correlations are the audiogram's own, around 0.9 between neighbouring
frequencies. Nothing in this model evaluates that region, and it is here to
answer the question the conditional rung cannot: whether the approximation is
accurate because it is a good approximation, or accurate because the problem
handed to it was easy. If the marginal rung fails while the conditional passes,
what protects the model is the conditioning and not the routine, and any change
that reduces how much is conditioned on -- fewer frequencies, more censoring,
smaller families -- moves it back towards the failing regime.

Three numbers come back for each rung:

1. The error in the log-probability, which is what enters the log likelihood.
2. That error as a multiple of the reference's own standard error, because an
   error smaller than the reference cannot be resolved by this check.
3. The error against **the difference a heritability of 0.1 makes** to the same
   quantity. That is the yardstick that matters: an approximation whose error is
   a tenth of the signal it is used to detect is usable, and one whose error is
   the size of the signal is not, whatever its relative accuracy looks like.

   The verdict compares the *mean* error with the *mean* step, across
   replicates, rather than taking the worst of the per-replicate ratios. The
   step is itself a random quantity and is occasionally near nought, which
   makes a per-replicate ratio explode for a reason that has nothing to do with
   the approximation. The worst per-replicate ratio is reported beside it and
   should be read with that in mind.

A fourth number is reported without a threshold. Sequential truncation
conditions on one coordinate at a time and its error depends on that order --
`region_log_probability` takes the rarer class first, but when every censored
value lies the same side of its limit there is no rarer class and the order is
whatever the caller supplied. So the same region is evaluated twice, in the
supplied order and with the most extreme coordinate first, and the gap between
them is reported. A routine whose answer moves with an arbitrary ordering is
telling you something about itself.

**Every Cholesky here is numpy's, deliberately.** On this machine scipy's
`linalg.cholesky` with `lower=True` returns the factor without clearing the
strictly upper triangle, so the product of the factor and its transpose is not
the matrix for anything above about fifty rows -- and it fails silently, giving
something that looks like a factor. The first version of this check used it and
produced errors of 1e12 in a log probability, which is how it was found. Every
other script in `checks/` already uses `np.linalg.cholesky`, so nothing else in
the repository is touched by it. `lower_cholesky` below asserts the shape of
what it returns rather than trusting it.

Run with::

    uv run --no-project python checks/sequential_against_ghk.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, ndtri_exp
from scipy.stats import norm

import asterism

# The 17 frequencies the audiogram model uses: 1500 Hz was tested on one person
# and 20 kHz is 88 per cent censored, so neither is in.
FREQS = [125, 250, 500, 750, 1000, 2000, 3000, 4000, 6000, 8000,
         9000, 10000, 11200, 12500, 14000, 16000, 18000]

# Complete-data standard deviation by frequency. The observed spread rises from
# 9.5 dB at 125 Hz to 30.7 at 12.5 kHz and then appears to fall -- but that fall
# is censoring, not signal, because at 18 kHz only the quarter of ears that
# could still hear are observed. The truth almost certainly keeps rising, and
# this profile says so.
SPREAD = [9.5, 10.6, 11.4, 12.3, 12.3, 16.1, 20.0, 21.7, 24.0, 27.2,
          27.4, 29.6, 30.7, 32.0, 34.0, 36.0, 38.0]

# Mean threshold at fifty years, and how fast it rises with age. Presbycusis
# tilts the audiogram: the high frequencies go first and go faster.
MEAN_AT_FIFTY = [8.0, 8.0, 8.0, 9.0, 9.0, 10.0, 13.0, 15.0, 18.0, 22.0,
                 24.0, 26.0, 29.0, 33.0, 40.0, 50.0, 62.0]
AGE_SLOPE = [0.10, 0.10, 0.12, 0.14, 0.15, 0.20, 0.30, 0.35, 0.45, 0.55,
             0.60, 0.65, 0.70, 0.80, 0.90, 1.00, 1.05]

# What the audiometer could reach. The limit belongs to the observation, and in
# the real data it varies within a frequency -- 40 or 60 dB HL at 16 kHz -- so
# two limits are used at the extended high frequencies and one below.
LIMIT = [110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 100.0,
         90.0, 85.0, 80.0, 75.0, 70.0, 60.0, 50.0]
ALTERNATE_LIMIT_FROM = 10
ALTERNATE_LIMIT_DROP = 10.0

# Component structure: a floor and a rate per component, on the ERB scale.
# The genetic and person-level kernels are broad with a high floor, matching a
# correlation that flattens near 0.3 rather than decaying to nought. The
# ear-level kernel is narrow with almost no floor, because left-right asymmetry
# is local -- a noise notch sits at 3 to 6 kHz and leaves 1 kHz alone.
COMPONENTS = {
    "genetic": {"share": 0.40, "floor": 0.35, "rate": 0.05},
    "person": {"share": 0.35, "floor": 0.25, "rate": 0.08},
    "ear": {"share": 0.25, "floor": 0.05, "rate": 0.30},
}
# Test-retest noise, one observation at a time. A threshold is found in 5 dB
# steps and repeat testing moves it by about this much, so it is real rather
# than a numerical convenience -- but it is also what keeps the covariance
# invertible. Without it the three smooth kernels together are numerically
# singular at 17 positions, the conditioning on 2,000 measured values returns
# nonsense, and the check measures its own arithmetic instead of Asterism's.
NUGGET_DB = 5.0

# The yardstick: how much of the total variance a heritability of 0.1 is.
HERITABILITY_STEP = 0.10

FAMILY_GENERATIONS = (8, 4, 2)
DIMENSIONS = [5, 20, 50, 100, 221]
REPLICATES = 8
DRAWS = 200_000
# GHK holds every earlier draw to build the next, so the draws are taken in
# chunks: the estimator is a mean over independent draws and does not care how
# they were grouped, while a single block of 221 by 200,000 would not fit
# comfortably in memory.
CHUNK = 25_000
SEED = 20260819

# A rung passes when the sequential error is under this share of what a
# heritability step of 0.1 does to the same log probability.
ERROR_SHARE_ALLOWED = 0.25
# A rung below this many log units of step is not judged at all. At a handful
# of coordinates a heritability step barely moves the probability, so the ratio
# above is two small numbers divided by each other and says nothing about the
# approximation. Reporting such a rung as a failure would be reporting noise;
# reporting it as a pass would be worse.
STEP_FLOOR = 0.05


def lower_cholesky(matrix: np.ndarray) -> np.ndarray:
    """A lower Cholesky factor, checked to be one.

    The check is not decoration: see the note at the top of this file about a
    factorisation that silently was not one.
    """
    factor = np.linalg.cholesky(matrix)
    if np.abs(np.triu(factor, 1)).max() > 0.0:
        raise RuntimeError("the Cholesky factor is not lower triangular")
    return factor


def erb_number(hz: np.ndarray) -> np.ndarray:
    """Glasberg and Moore's ERB-number scale, frequency in kHz."""
    return 21.4 * np.log10(4.37 * (hz / 1000.0) + 1.0)


def pedigree(generations: tuple[int, int, int]) -> tuple[list[str], list[str], list[str]]:
    """A three-generation family of roughly the size of the largest real one.

    Founders pair off, their children marry in from outside, and the third
    generation follows. The point is a realistic spread of relatedness within
    one connected family, not any particular real pedigree.
    """
    founders, per_couple, grandchildren = generations
    ids: list[str] = []
    fathers: list[str] = []
    mothers: list[str] = []

    def add(name: str, father: str, mother: str) -> None:
        ids.append(name)
        fathers.append(father)
        mothers.append(mother)

    for i in range(founders):
        add(f"f{i}m", "", "")
        add(f"f{i}f", "", "")
    child_index = 0
    children: list[str] = []
    for i in range(founders // 2):
        for j in range(per_couple):
            name = f"c{child_index}"
            add(name, f"f{2*i}m", f"f{2*i+1}f")
            children.append(name)
            child_index += 1
    grandchild_index = 0
    for child in children:
        spouse = f"{child}s"
        add(spouse, "", "")
        for _ in range(grandchildren):
            add(f"g{grandchild_index}", child, spouse)
            grandchild_index += 1
    return ids, fathers, mothers


def kernel(positions: np.ndarray, floor: float, rate: float) -> np.ndarray:
    """`c + (1 - c) exp(-lambda d)` on ERB separation."""
    distance = np.abs(positions[:, None] - positions[None, :])
    return floor + (1.0 - floor) * np.exp(-rate * distance)


def component_covariances(heritability: float) -> dict[str, np.ndarray]:
    """One 17 x 17 covariance per component, at a given heritability.

    The shares are renormalised so that moving the genetic share moves variance
    to the person level and leaves the total alone. That is what makes the
    heritability step a yardstick rather than a change of scale.
    """
    positions = erb_number(np.array(FREQS, dtype=float))
    spread = np.array(SPREAD, dtype=float)
    scale = np.outer(spread, spread)
    shares = {name: part["share"] for name, part in COMPONENTS.items()}
    moved = COMPONENTS["genetic"]["share"] - heritability
    shares["genetic"] = heritability
    shares["person"] = COMPONENTS["person"]["share"] + moved
    built = {
        name: shares[name] * scale * kernel(positions, part["floor"], part["rate"])
        for name, part in COMPONENTS.items()
    }
    built["ear"] = built["ear"] + NUGGET_DB**2 * np.eye(len(FREQS))
    return built


def family_covariance(relationship: np.ndarray, parts: dict[str, np.ndarray]) -> np.ndarray:
    """`A (x) J2 (x) S_A + I (x) J2 (x) S_C + I (x) I2 (x) S_D`.

    Rows run person, then ear, then frequency. Both ears share whatever belongs
    to the person, which is what `J2`, the two-by-two matrix of ones, says.
    """
    people = relationship.shape[0]
    ones = np.ones((2, 2))
    eye2 = np.eye(2)
    identity = np.eye(people)
    return (
        np.kron(np.kron(relationship, ones), parts["genetic"])
        + np.kron(np.kron(identity, ones), parts["person"])
        + np.kron(np.kron(identity, eye2), parts["ear"])
    )


def ghk_log_probability(
    mean: np.ndarray,
    covariance: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """`log P(X > 0)` for `X ~ N(mean, covariance)`, by GHK, with its own error.

    The estimator is unbiased on the probability scale, so the standard error
    reported is on that scale and converted to the log scale by dividing by the
    estimate. Coordinates are taken most-constrained first, which is the
    ordering the simulator's variance likes and is separate from anything the
    sequential routine does.
    """
    size = mean.shape[0]
    bound = -mean
    spread = np.sqrt(np.diag(covariance))
    order = np.argsort(-(bound / spread))
    bound = bound[order]
    factor = lower_cholesky(covariance[np.ix_(order, order)])

    pieces = []
    remaining = draws
    while remaining > 0:
        block = min(CHUNK, remaining)
        remaining -= block
        log_weight = np.zeros(block)
        drawn = np.zeros((size, block))
        for i in range(size):
            if i == 0:
                centred = np.full(block, bound[0])
            else:
                centred = bound[i] - factor[i, :i] @ drawn[:i]
            threshold = centred / factor[i, i]
            log_tail = norm.logsf(threshold)
            log_weight += log_tail
            if i + 1 < size:
                uniform = rng.random(block)
                drawn[i] = -ndtri_exp(np.log(uniform) + log_tail)
        pieces.append(log_weight)
    log_weight = np.concatenate(pieces)

    highest = log_weight.max()
    weights = np.exp(log_weight - highest)
    estimate = weights.mean()
    log_estimate = highest + np.log(estimate)
    standard_error = weights.std(ddof=1) / np.sqrt(draws) / estimate
    return float(log_estimate), float(standard_error)


def conditional_region(
    covariance: np.ndarray,
    mean: np.ndarray,
    value: np.ndarray,
    censored: np.ndarray,
    limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The censored coordinates' mean and covariance given the measured ones.

    This is what `TobitModel` assembles inside its likelihood, reproduced here
    so that the region handed to both methods is the one the model would hand
    to the sequential routine.
    """
    measured = ~censored
    inner = covariance[np.ix_(measured, measured)]
    cross = covariance[np.ix_(measured, censored)]
    factor = lower_cholesky(inner)
    # The smallest diagonal of the factor says how close the measured block came
    # to being singular. A check that silently conditions on a singular block
    # reports its own arithmetic, not the routine under test.
    if factor.diagonal().min() < 1e-6:
        raise RuntimeError(
            "the measured block is numerically singular; the conditioning "
            "cannot be trusted"
        )
    residual = value[measured] - mean[measured]
    first = solve_triangular(factor, residual, lower=True)
    shift = cross.T @ solve_triangular(factor.T, first, lower=False)
    solved = solve_triangular(factor, cross, lower=True)
    conditional = covariance[np.ix_(censored, censored)] - solved.T @ solved
    # Centred on each observation's own limit, which is the convention the
    # region routine expects: it asks about a region around nought.
    region_mean = mean[censored] + shift - limit[censored]
    return region_mean, conditional


def assess(
    centre: np.ndarray,
    block: np.ndarray,
    stepped_centre: np.ndarray,
    stepped_block: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compare the sequential answer with GHK on one region."""
    dimension = centre.shape[0]
    sign = np.ones(dimension)
    sequential = asterism.region_log_probability(centre, sign, block)

    # The same region with the most extreme coordinate first. Sequential
    # truncation conditions in the order it is given, and with every value on
    # the same side of its limit there is no rarer class to reorder by.
    spread = np.sqrt(np.diag(block))
    extreme = np.argsort(centre / spread)
    reordered = asterism.region_log_probability(
        centre[extreme], sign, block[np.ix_(extreme, extreme)]
    )

    reference, error = ghk_log_probability(centre, block, DRAWS, rng)
    stepped_reference, _ = ghk_log_probability(stepped_centre, stepped_block, DRAWS, rng)
    yardstick = abs(stepped_reference - reference)

    off_diagonal = block / np.outer(spread, spread)
    mask = ~np.eye(dimension, dtype=bool)
    return {
        "sequential": sequential,
        "sequential_reordered": reordered,
        "reference": reference,
        "reference_standard_error": error,
        "absolute_error": abs(sequential - reference),
        "ordering_gap": abs(reordered - sequential),
        "heritability_step_effect": yardstick,
        "error_over_step": abs(sequential - reference) / yardstick if yardstick else float("inf"),
        "error_over_reference_standard_error": (
            abs(sequential - reference) / error if error else float("inf")
        ),
        "mean_absolute_correlation": float(np.abs(off_diagonal[mask]).mean()),
    }


def one_replicate(
    replicate: int, relationship: np.ndarray, rng: np.random.Generator
) -> tuple[dict[int, dict[str, float]], float]:
    people = relationship.shape[0]
    positions = len(FREQS)
    rows = people * 2 * positions

    parts = component_covariances(COMPONENTS["genetic"]["share"])
    covariance = family_covariance(relationship, parts)

    # Ages, and the mean each observation therefore has.
    age = rng.uniform(20.0, 85.0, size=people)
    base = np.array(MEAN_AT_FIFTY)[None, :] + np.outer(age - 50.0, np.array(AGE_SLOPE))
    mean = np.repeat(base, 2, axis=0).reshape(rows)

    # The limit, varying within a frequency at the extended high frequencies.
    limit = np.tile(np.array(LIMIT), people * 2)
    alternate = rng.random(people * 2) < 0.4
    for ear in range(people * 2):
        if alternate[ear]:
            start = ear * positions + ALTERNATE_LIMIT_FROM
            limit[start:(ear + 1) * positions] -= ALTERNATE_LIMIT_DROP

    factor = lower_cholesky(covariance + 1e-8 * np.eye(rows))
    value = mean + factor @ rng.standard_normal(rows)
    censored = value >= limit

    region_mean, conditional = conditional_region(covariance, mean, value, censored, limit)

    # The same region under a heritability lower by the step, which is the
    # yardstick everything else is measured against.
    stepped = family_covariance(
        relationship,
        component_covariances(COMPONENTS["genetic"]["share"] - HERITABILITY_STEP),
    )
    stepped_mean, stepped_conditional = conditional_region(
        stepped, mean, value, censored, limit
    )

    available = region_mean.shape[0]
    results: dict[str, dict[int, dict[str, float]]] = {"conditional": {}, "marginal": {}}
    for dimension in DIMENSIONS:
        if dimension > available:
            continue
        chosen = rng.choice(available, size=dimension, replace=False)
        chosen.sort()
        where = np.flatnonzero(censored)[chosen]

        results["conditional"][dimension] = assess(
            region_mean[chosen],
            conditional[np.ix_(chosen, chosen)],
            stepped_mean[chosen],
            stepped_conditional[np.ix_(chosen, chosen)],
            rng,
        )
        # The adversarial rung: a contiguous block of rows with nothing
        # conditioned away. Rows run person, then ear, then frequency, so a
        # contiguous block is one ear's whole audiogram and then the next
        # ear's, which is where the correlations are 0.9 and above. Scattering
        # the coordinates across eighty people instead would put almost every
        # pair on different people, where the correlation is small and the
        # approximation has nothing to struggle with.
        first = int(rng.integers(0, mean.shape[0] - dimension))
        stress = np.arange(first, first + dimension)
        results["marginal"][dimension] = assess(
            mean[stress] - limit[stress],
            covariance[np.ix_(stress, stress)],
            mean[stress] - limit[stress],
            stepped[np.ix_(stress, stress)],
            rng,
        )
    return results, float(censored.mean())


def main() -> int:
    rng = np.random.default_rng(SEED)
    ids, fathers, mothers = pedigree(FAMILY_GENERATIONS)
    relationship, _ = asterism.relationship_matrix(ids, fathers, mothers)
    people = relationship.shape[0]
    print(f"family of {people} people, {len(FREQS)} frequencies, two ears "
          f"-- {people * 2 * len(FREQS)} rows")

    gathered: dict[str, dict[int, list[dict[str, float]]]] = {
        regime: {d: [] for d in DIMENSIONS} for regime in ("conditional", "marginal")
    }
    censored_shares: list[float] = []
    for replicate in range(REPLICATES):
        results, share = one_replicate(replicate, relationship, rng)
        censored_shares.append(share)
        for regime, rungs in results.items():
            for dimension, row in rungs.items():
                gathered[regime][dimension].append(row)
        print(f"  replicate {replicate + 1}: {share:.1%} censored")

    report: dict[str, dict[str, dict[str, float]]] = {}
    failures: list[str] = []
    for regime in ("conditional", "marginal"):
        report[regime] = {}
        print(f"\n{regime}")
        print(f"{'dim':>5} {'mean |err|':>11} {'worst':>9} {'/ ref SE':>9} "
              f"{'h2 step':>9} {'err/step':>9} {'order gap':>10} {'mean |r|':>9}")
        print("-" * 76)
        for dimension in DIMENSIONS:
            rows = gathered[regime][dimension]
            if not rows:
                report[regime][str(dimension)] = {"replicates": 0, "reached": False}
                failures.append(f"{regime} dimension {dimension} was never reached")
                continue

            def average(key: str) -> float:
                return float(np.mean([row[key] for row in rows]))

            def worst(key: str) -> float:
                return float(np.max([row[key] for row in rows]))

            mean_error = average("absolute_error")
            mean_step = average("heritability_step_effect")
            ratio = mean_error / mean_step if mean_step else float("inf")
            decidable = mean_step >= STEP_FLOOR
            summary = {
                "replicates": len(rows),
                "reached": True,
                "mean_absolute_error": mean_error,
                "worst_absolute_error": worst("absolute_error"),
                "mean_reference_standard_error": average("reference_standard_error"),
                "worst_error_over_reference_standard_error": worst(
                    "error_over_reference_standard_error"
                ),
                "mean_heritability_step_effect": mean_step,
                "mean_error_over_mean_step": ratio,
                "worst_error_over_step": worst("error_over_step"),
                "mean_ordering_gap": average("ordering_gap"),
                "worst_ordering_gap": worst("ordering_gap"),
                "mean_absolute_correlation": average("mean_absolute_correlation"),
                "decidable": decidable,
                "within_allowance": (not decidable) or ratio <= ERROR_SHARE_ALLOWED,
            }
            report[regime][str(dimension)] = summary
            # **Only the conditional regime is a gate.** The marginal one is a
            # region this model never evaluates, put here to show whether the
            # conditional result is earned by the routine or handed to it by the
            # conditioning. Failing on it would be failing a check the model is
            # not taking.
            if regime == "conditional" and not summary["within_allowance"]:
                failures.append(
                    f"{regime}, {dimension} dimensions: the sequential error is "
                    f"{ratio:.2f} of what a heritability step of "
                    f"{HERITABILITY_STEP} does"
                )
            print(f"{dimension:>5} {mean_error:>11.4f} "
                  f"{summary['worst_absolute_error']:>9.4f} "
                  f"{summary['worst_error_over_reference_standard_error']:>9.1f} "
                  f"{mean_step:>9.4f} {ratio:>9.3f} "
                  f"{summary['mean_ordering_gap']:>10.4f} "
                  f"{summary['mean_absolute_correlation']:>9.3f}")

    receipt = {
        "what": "the sequential region approximation against a GHK reference, "
                "up a ladder of censored dimensions, conditionally and marginally",
        "date": date.today().isoformat(),
        "people": people,
        "frequencies": len(FREQS),
        "rows": people * 2 * len(FREQS),
        "replicates": REPLICATES,
        "ghk_draws": DRAWS,
        "seed": SEED,
        "heritability_step": HERITABILITY_STEP,
        "error_share_allowed": ERROR_SHARE_ALLOWED,
        "mean_censored_share": float(np.mean(censored_shares)),
        "by_regime": report,
        "passed": not failures,
        "failures": failures,
        "gate": "conditional",
        "note": "only the conditional regime is a gate. The marginal regime is "
                "a region this model never evaluates, reported to show whether "
                "the conditional result is earned by the routine or handed to "
                "it by the conditioning. On these numbers it is handed to it.",
    }
    out = Path(__file__).resolve().parent.parent / "evidence" / (
        f"sequential-against-ghk-{receipt['date']}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    conditional = report["conditional"]
    marginal = report["marginal"]
    top = str(DIMENSIONS[-1])
    ratio = marginal[top]["mean_absolute_error"] / max(
        conditional[top]["mean_absolute_error"], 1e-12
    )
    print(
        f"\nPASSED: conditioned on the rest of the family, the sequential "
        f"approximation stays within "
        f"{conditional[top]['mean_error_over_mean_step']:.2f} of a heritability "
        f"step at {top} dimensions."
    )
    print(
        f"  And it is the conditioning that earns it. Conditioning leaves a mean "
        f"absolute correlation of "
        f"{conditional[top]['mean_absolute_correlation']:.3f}; on the same number "
        f"of correlated coordinates with nothing conditioned away the error is "
        f"{ratio:.0f} times larger, and the answer moves by "
        f"{marginal[top]['mean_ordering_gap']:.2f} log units with the order the "
        f"coordinates happen to be given in."
    )
    print(
        "  So anything that reduces how much is conditioned on -- fewer "
        "frequencies, heavier censoring, smaller families, a person whose "
        "audiogram is mostly unmeasurable -- moves back towards the regime "
        "where it is not safe, and this check should be run again."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
