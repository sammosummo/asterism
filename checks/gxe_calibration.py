"""Do the two genotype-by-environment tests hold their level?

Asterism carries two cross-sectional G-by-E surfaces. They differ only in what
is put on the environment-by-environment covariance — an exponential decay in
one, a smooth quadratic in the other — and they report the same quantities, so
they are calibrated side by side and against the same nulls.

Each surface supports two tests, and they ask different questions:

- **interaction**: the genetic covariance does not involve the environment at
  all. Rejecting says something about genes and environment together, but not
  what.
- **correlation**: the genetic effects at any two environments are the same
  effects. The genetic variance may still grow or shrink with the environment;
  what is ruled out is a change in *which* genes matter.

The second is the narrower and usually the more interesting claim. A
heritability that rises with an environment can follow from a change of scale in
the measurement; a genetic correlation below one cannot.

## A null is only a null inside a family

**The two surfaces do not have the same "no reordering" null**, and this is the
main thing this check is built to measure. Under no reordering the genetic
covariance is `a(z_i) * a(z_j)` for some function `a`. The exponential surface
can hold `a(z) = exp(alpha + gamma*z)` and nothing else; the random-regression surface can
hold `a(z) = l00 + l10*z` and nothing else. Neither family contains the other.

So a rank-one surface drawn from one family is *misspecified* for the other, and
the misfit has somewhere to go: into the one parameter that is free under the
alternative and pinned under the null. A surface that cannot bend its variance
function the way the data does will bend its correlation instead, and report a
reordering that is not there.

Both rank-one scenarios are therefore run through both surfaces. Each is a
genuine null for the surface that contains it, and the check fails on those.
For the surface that does not contain it the rate is printed and not failed on —
it is not a fault in the test, it is the cost of the wrong family, and the size
of that cost is exactly what a reader needs to know before choosing a surface.

## The residual is a nuisance and must stay one

A residual variance that changes with the environment is not a
gene-by-environment interaction, but it does put environmental structure into
the covariance, and on real traits residual variances change with almost
anything. The residual surface is left free under both nulls precisely so it
cannot be mistaken for a genetic one. `residual_tilt` is where that either works
or does not.

## The criterion is validity, not uniformity

Both statistics sit on a bound under their null, so a large share of fits land
exactly on it and return a p-value of one. The p-values are uniform below a half
with an atom at one. That is correct and is not a departure. What must hold is
that the rejection rate at each level does not exceed that level.

**The two surfaces do not sit on the same kind of bound, and it shows here.**
The exponential surface's null is `lambda = 0`, a flat face of its parameter
box, which is the case the even mixture of chi-squares is derived for — and its
level comes out very close to nominal. The random-regression surface's null is
`q00*q11 - q01^2 = 0`, the *curved* boundary of the positive semidefinite cone,
where the mixture weights follow the local solid angle instead of being even.
The even mixture is therefore the wrong reference for it, erring the safe way:
its tests are valid but conservative, and its atom runs well above a half.
Restoring that power would mean bootstrapping the reference rather than looking
it up. Reparameterising will not do it — a likelihood ratio is invariant to the
coordinates it is computed in.

Run with:

    uv run --no-project python checks/gxe_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypedDict

import asterism
import numpy as np

PAIRS: int = 250
"""Fixed the number of sibling pairs in every simulated data set."""

N: int = 2 * PAIRS
"""Computed the fixed simulated roster size."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "500"))
"""Selected the requested replicates per scenario from the environment."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "6"))
"""Selected the worker-process count from the environment."""

LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)
"""Fixed the nominal rejection levels checked by the calibration."""

SURFACES: tuple[str, ...] = ("exponential", "random_regression")
"""Named the two continuous GxE surface families under comparison."""

TESTS: tuple[str, ...] = ("interaction", "correlation")
"""Named the two boundary nulls calibrated on every surface."""


class Scenario(TypedDict):
    """Describe one generating covariance and its true in-family nulls."""

    what: str
    """Explain the scientific meaning of the generating scenario."""

    genetic: Callable[[np.ndarray, np.ndarray], np.ndarray]
    """Evaluate genetic covariance at two environments."""

    residual: Callable[[np.ndarray], np.ndarray]
    """Evaluate residual variance at one or more environments."""

    true: set[str]
    """Name null hypotheses satisfied by the generating covariance."""

    family: set[str]
    """Name fitted surface families containing the generating covariance."""


def structure() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sibling pairs, each person carrying an environment.

    **The environment varies within a pair as well as between it.** With one
    environment per family a genetic surface could not be told from a plain
    heritability, and the check would be measuring nothing.

    Returns:
        Relationship matrix, continuous environments and intercept design.
    """
    relationship: np.ndarray = np.eye(N)
    """Initialised the relationship matrix with individual diagonal entries."""

    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Set the forward within-pair relationship coefficient."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Set the symmetric within-pair relationship coefficient."""

    # Drawn once and held: the design is not what is being varied.
    z: np.ndarray = np.random.default_rng(20_260_813).uniform(-1.5, 1.5, N)
    """Drew fixed within- and between-family continuous environments."""
    return relationship, z, np.ones((N, 1))


RELATIONSHIP, Z, DESIGN = structure()
"""Built the deterministic design shared by every scenario and replicate."""

# Each scenario gives the genetic covariance as a function of two environments
# and the residual variance as a function of one, then says which of the two
# nulls it makes true and which surfaces contain it. Those two together decide
# what each row means: a true null inside the surface's family is a level to be
# held, a true null outside it is the cost of the wrong family, and a false null
# is power.
BOTH: set[str] = set(SURFACES)
"""Named scenarios represented by both fitted surface families."""

SCENARIOS: dict[str, Scenario] = {
    "flat": {
        "what": "no interaction of any kind",
        "genetic": lambda a, b: np.full_like(a, 0.5),
        "residual": lambda z: np.full_like(z, 0.5),
        "true": set(TESTS),
        "family": BOTH,
    },
    "residual_tilt": {
        # The genetic covariance is flat, so both nulls hold for both surfaces.
        # The residual variance is linear, which the random-regression surface holds exactly
        # and the exponential only approximates -- deliberately, because on real
        # traits the residual will not be in either family, and residual misfit
        # leaking into a genetic test is the failure this scenario is for.
        "what": "residual variance changes with the environment, genes do not",
        "genetic": lambda a, b: np.full_like(a, 0.5),
        "residual": lambda z: 0.5 + 0.3 * z,
        "true": set(TESTS),
        "family": BOTH,
    },
    "rank_one_linear": {
        # (0.6 + 0.4 z_i)(0.6 + 0.4 z_j): exactly the random-regression surface's rank-one
        # family, and outside the exponential's, whose variance function cannot
        # reach nought.
        "what": "same genes throughout, genetic standard deviation linear in z",
        "genetic": lambda a, b: (0.6 + 0.4 * a) * (0.6 + 0.4 * b),
        "residual": lambda z: np.full_like(z, 0.5),
        "true": {"correlation"},
        "family": {"random_regression"},
    },
    "rank_one_loglinear": {
        # sqrt(g(z_i) g(z_j)) with g(z) = exp(-0.7 + 0.6 z): exactly the
        # exponential surface's rank-one family, and outside the random-regression one's.
        "what": "same genes throughout, genetic standard deviation log-linear in z",
        "genetic": lambda a, b: np.exp(-0.7 + 0.3 * (a + b)),
        "residual": lambda z: np.full_like(z, 0.5),
        "true": {"correlation"},
        "family": {"exponential"},
    },
    "reordering_quadratic": {
        # q00 q11 well above q01^2, so the correlation across the range falls
        # properly below one. In the random-regression surface's family.
        "what": "different genes at different environments, quadratic surface",
        "genetic": lambda a, b: 0.5 + 0.05 * (a + b) + 0.45 * a * b,
        "residual": lambda z: np.full_like(z, 0.5),
        "true": set(),
        "family": {"random_regression"},
    },
    "reordering_loglinear": {
        # The same log-linear variance function as rank_one_loglinear with a
        # decay switched on: at the tenth and ninetieth percentiles of z the
        # genetic correlation is exp(-0.4 * 2.4) = 0.38. In the exponential
        # surface's family. **Both surfaces need an alternative of their own**,
        # or a power comparison only says which one was handed its home ground.
        "what": "different genes at different environments, exponential surface",
        "genetic": lambda a, b: (
            np.exp(-0.7 + 0.3 * (a + b)) * np.exp(-0.4 * np.abs(a - b))
        ),
        "residual": lambda z: np.full_like(z, 0.5),
        "true": set(),
        "family": {"exponential"},
    },
}
"""Defined generating covariances, true nulls and containing model families."""


def covariance(setting: Scenario) -> np.ndarray:
    """Construct the full observation covariance for one scenario.

    Args:
        setting: Generating genetic and residual surface functions.

    Returns:
        Relationship-scaled genetic covariance plus residual diagonal.
    """
    out: np.ndarray = RELATIONSHIP * setting["genetic"](Z[:, None], Z[None, :])
    """Constructed the relationship-scaled genetic covariance."""

    out[np.diag_indices(N)] += setting["residual"](Z)
    """Added environment-specific residual variance on the diagonal."""
    return out


FACTORS: dict[str, np.ndarray] = {
    name: np.linalg.cholesky(covariance(setting) + 1e-9 * np.eye(N))
    for name, setting in SCENARIOS.items()
}
"""Factorised every generating covariance for deterministic response draws."""


def one(job: tuple[str, int]) -> dict[str, Any]:
    """Run both boundary tests on both surfaces for one response.

    Args:
        job: Scenario name and zero-based replicate number.

    Returns:
        Named public test records or documented refusals for each surface/null.
    """
    name, index = job
    """Separated the generating scenario from its replicate identity."""

    y: np.ndarray = FACTORS[name] @ np.random.default_rng(
        910_000 + 1000 * index
    ).standard_normal(N)
    """Drew one deterministic response from the scenario covariance."""

    out: dict[str, Any] = {"scenario": name}
    """Initialised the public-test records for one simulated scenario."""
    for surface in SURFACES:
        model: asterism.GxeModel = asterism.GxeModel(
            RELATIONSHIP,
            Z,
            DESIGN,
            surface=surface,
        )
        """Built one documented GxE surface for both of its null tests."""

        for null in TESTS:
            try:
                outcome: dict[str, Any] | None = model.test(y, null, reml=True)
                """Tested the requested null through the public named record."""
            except ValueError:
                outcome = None
                """Recorded that the estimator refused this surface and null."""

            if outcome is None:
                out[f"{surface}/{null}"] = None
                """Recorded a documented estimator refusal for this replicate."""
            else:
                out[f"{surface}/{null}"] = {
                    "p": outcome["p_value"],
                    "statistic": outcome["statistic"],
                    "rule": outcome["rule"],
                }
                """Recorded the named test fields consumed by the calibration."""
    return out


def main() -> int:
    """Run every GxE test-calibration cell and print its evidence receipt."""
    print(
        f"Genotype-by-environment tests, REML. {PAIRS} sibling pairs, n = {N}, "
        f"{REPLICATES} replicates.\n"
        f"Environment drawn once and held; both surfaces see the same data.\n"
    )

    jobs: list[tuple[str, int]] = [
        (name, index) for name in SCENARIOS for index in range(REPLICATES)
    ]
    """Enumerated every generating scenario and replicate exactly once."""

    started: float = time.perf_counter()
    """Started the elapsed-time measurement immediately before worker launch."""

    with ProcessPoolExecutor(WORKERS) as pool:
        results: list[dict[str, Any]] = list(pool.map(one, jobs, chunksize=4))
        """Ran every deterministic test replicate through the worker pool."""
    print(
        f"{len(jobs)} data sets in {(time.perf_counter() - started) / 60:.1f} minutes.\n"
    )

    failures: list[str] = []
    """Initialised the scientific pass-rule failure messages."""

    recorded: dict[str, dict[str, Any]] = {}
    """Initialised the machine-readable summaries for every scenario."""

    for name, setting in SCENARIOS.items():
        got: list[dict[str, Any]] = [r for r in results if r["scenario"] == name]
        """Selected every attempted replicate for the current scenario."""

        print(f"{name}  --  {setting['what']}")
        header: str = f"  {'surface':<20}{'test':<24}{'of':>5}"
        """Started the fixed-width result-table header."""

        header += "".join(f"{f'p<={level:g}':>10}" for level in LEVELS)
        """Added one rejection-rate column for every nominal level."""

        print(header + f"{'atom at 1':>12}")
        recorded[name] = {
            "what": setting["what"],
            "nulls_true": sorted(setting["true"]),
            "family": sorted(setting["family"]),
            "tests": {},
        }
        """Recorded the scenario meaning before its surface-specific tests."""

        for surface in SURFACES:
            for null in TESTS:
                key: str = f"{surface}/{null}"
                """Constructed the stable surface and null result identity."""

                p_values: np.ndarray = np.array(
                    [r[key]["p"] for r in got if r[key] is not None]
                )
                """Collected p-values from every completed test in this cell."""

                if not len(p_values):
                    failures.append(f"{name}: no result at all for {key}")
                    continue
                # A true null inside the surface's family is a level to hold; a
                # true null outside it is the cost of the wrong family; a false
                # null is power, and power on someone else's family is not a
                # comparison worth making.
                true_here: bool = null in setting["true"]
                """Determined whether the generating scenario satisfies this null."""

                in_family: bool = surface in setting["family"]
                """Determined whether the fitted surface contains this scenario."""

                is_null: bool = true_here and in_family
                """Identified true null cells eligible for calibration gating."""

                is_outside: bool = true_here and not in_family
                """Identified true nulls outside the fitted surface family."""

                rates: list[float] = [
                    float(np.mean(p_values <= level)) for level in LEVELS
                ]
                """Computed empirical rejection rates at every nominal level."""

                atom: float = float(np.mean(p_values > 0.999))
                """Measured the boundary distribution's empirical atom at one."""

                recorded[name]["tests"][key] = {
                    "computed": len(p_values),
                    "role": (
                        "null"
                        if is_null
                        else "misspecified"
                        if is_outside
                        else "power"
                        if in_family
                        else "power_wrong_family"
                    ),
                    "rejection": {
                        str(level): rate
                        for level, rate in zip(LEVELS, rates, strict=True)
                    },
                    "atom_at_one": atom,
                }
                """Recorded the cell role, availability and rejection summaries."""

                marks: list[str] = []
                """Initialised formatted calibration marks for this table row."""

                for level, rate in zip(LEVELS, rates, strict=True):
                    if not is_null:
                        marks.append(f"{rate:>10.3f}")
                        continue
                    # One-sided: over-rejection is the failure, and a rate below
                    # the level is conservative rather than wrong.
                    error: float = float(np.sqrt(level * (1 - level) / len(p_values)))
                    """Computed the binomial Monte Carlo standard error."""

                    ceiling: float = level + 1.96 * error
                    """Computed the one-sided rejection ceiling used by the rule."""

                    marks.append(f"{rate:>9.3f}{'!' if rate > ceiling else ' '}")
                    if rate > ceiling:
                        failures.append(
                            f"{name} {key} rejects {rate:.3f} at {level:g}, "
                            f"above {ceiling:.3f}"
                        )
                tag: str = (
                    "null"
                    if is_null
                    else "WRONG FAMILY"
                    if is_outside
                    else "power"
                    if in_family
                    else "power, wrong family"
                )
                """Labelled the cell as null, misspecified or powered context."""

                print(
                    f"  {surface:<20}{null + ' (' + tag + ')':<24}{len(p_values):>5}"
                    + "".join(marks)
                    + f"{atom:>11.2f}"
                )
        print()

    print(
        "A `!` marks a rejection rate above its level's one-sided binomial ceiling,\n"
        "and only null rows are marked. `WRONG FAMILY` rows are true nulls that lie\n"
        "outside that surface's reach: their rate is the cost of the wrong family,\n"
        "reported rather than failed on. Rates below a level are conservative — the\n"
        "atom at one is the boundary sitting where the null puts it, not a fault."
    )

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nBoth tests hold their level on both surfaces, on every null each")
    print("surface can represent, including where the residual variance changes")
    print("with the environment and the genes do not.")

    print(
        json.dumps(
            {
                "estimator": "reml",
                "pairs": PAIRS,
                "people": N,
                "replicates": REPLICATES,
                "levels": list(LEVELS),
                "surfaces": list(SURFACES),
                "scenarios": recorded,
                "note": (
                    "the criterion is validity and not uniformity: both statistics "
                    "sit on a bound under their null, so about half of all fits "
                    "return a p-value of one. Rows marked misspecified are true "
                    "nulls outside the fitted surface's family and are not failed on"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
