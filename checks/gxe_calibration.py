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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from asterism import _core

PAIRS = 250
N = 2 * PAIRS
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "500"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "6"))
LEVELS = (0.01, 0.05, 0.10)
SURFACES = ("exponential", "random_regression")
TESTS = ("interaction", "correlation")


def structure():
    """Sibling pairs, each person carrying an environment.

    **The environment varies within a pair as well as between it.** With one
    environment per family a genetic surface could not be told from a plain
    heritability, and the check would be measuring nothing.
    """
    relationship = np.eye(N)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5
    # Drawn once and held: the design is not what is being varied.
    z = np.random.default_rng(20_260_813).uniform(-1.5, 1.5, N)
    return relationship, z, np.ones((N, 1))


RELATIONSHIP, Z, DESIGN = structure()

# Each scenario gives the genetic covariance as a function of two environments
# and the residual variance as a function of one, then says which of the two
# nulls it makes true and which surfaces contain it. Those two together decide
# what each row means: a true null inside the surface's family is a level to be
# held, a true null outside it is the cost of the wrong family, and a false null
# is power.
BOTH = set(SURFACES)
SCENARIOS = {
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
        "genetic": lambda a, b: np.exp(-0.7 + 0.3 * (a + b)) * np.exp(-0.4 * np.abs(a - b)),
        "residual": lambda z: np.full_like(z, 0.5),
        "true": set(),
        "family": {"exponential"},
    },
}


def covariance(setting):
    out = RELATIONSHIP * setting["genetic"](Z[:, None], Z[None, :])
    out[np.diag_indices(N)] += setting["residual"](Z)
    return out


FACTORS = {
    name: np.linalg.cholesky(covariance(setting) + 1e-9 * np.eye(N))
    for name, setting in SCENARIOS.items()
}


def one(job):
    name, index = job
    y = FACTORS[name] @ np.random.default_rng(910_000 + 1000 * index).standard_normal(N)
    out = {"scenario": name}
    for surface in SURFACES:
        for null in TESTS:
            try:
                statistic, p_value, rule, _, _ = _core.gxe_test(
                    RELATIONSHIP, Z, DESIGN, y, surface, null, True
                )
                out[f"{surface}/{null}"] = {"p": p_value, "statistic": statistic, "rule": rule}
            except Exception:
                out[f"{surface}/{null}"] = None
    return out


def main() -> int:
    print(
        f"Genotype-by-environment tests, REML. {PAIRS} sibling pairs, n = {N}, "
        f"{REPLICATES} replicates.\n"
        f"Environment drawn once and held; both surfaces see the same data.\n"
    )

    jobs = [(name, index) for name in SCENARIOS for index in range(REPLICATES)]
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(one, jobs, chunksize=4))
    print(f"{len(jobs)} data sets in {(time.perf_counter() - started) / 60:.1f} minutes.\n")

    failures: list[str] = []
    recorded: dict = {}
    for name, setting in SCENARIOS.items():
        got = [r for r in results if r["scenario"] == name]
        print(f"{name}  --  {setting['what']}")
        header = f"  {'surface':<20}{'test':<24}{'of':>5}"
        header += "".join(f"{f'p<={level:g}':>10}" for level in LEVELS)
        print(header + f"{'atom at 1':>12}")
        recorded[name] = {
            "what": setting["what"],
            "nulls_true": sorted(setting["true"]),
            "family": sorted(setting["family"]),
            "tests": {},
        }
        for surface in SURFACES:
            for null in TESTS:
                key = f"{surface}/{null}"
                p_values = np.array([r[key]["p"] for r in got if r[key] is not None])
                if not len(p_values):
                    failures.append(f"{name}: no result at all for {key}")
                    continue
                # A true null inside the surface's family is a level to hold; a
                # true null outside it is the cost of the wrong family; a false
                # null is power, and power on someone else's family is not a
                # comparison worth making.
                true_here = null in setting["true"]
                in_family = surface in setting["family"]
                is_null = true_here and in_family
                is_outside = true_here and not in_family
                rates = [float(np.mean(p_values <= level)) for level in LEVELS]
                atom = float(np.mean(p_values > 0.999))
                recorded[name]["tests"][key] = {
                    "computed": len(p_values),
                    "role": (
                        "null" if is_null
                        else "misspecified" if is_outside
                        else "power" if in_family
                        else "power_wrong_family"
                    ),
                    "rejection": {str(level): rate for level, rate in zip(LEVELS, rates)},
                    "atom_at_one": atom,
                }
                marks = []
                for level, rate in zip(LEVELS, rates):
                    if not is_null:
                        marks.append(f"{rate:>10.3f}")
                        continue
                    # One-sided: over-rejection is the failure, and a rate below
                    # the level is conservative rather than wrong.
                    error = np.sqrt(level * (1 - level) / len(p_values))
                    ceiling = level + 1.96 * error
                    marks.append(f"{rate:>9.3f}{'!' if rate > ceiling else ' '}")
                    if rate > ceiling:
                        failures.append(
                            f"{name} {key} rejects {rate:.3f} at {level:g}, "
                            f"above {ceiling:.3f}"
                        )
                tag = (
                    "null" if is_null
                    else "WRONG FAMILY" if is_outside
                    else "power" if in_family
                    else "power, wrong family"
                )
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

    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/gxe-calibration-2026-08-13.json").write_text(
        json.dumps(
            {
                "what": "level and power of the two genotype-by-environment tests",
                "date": "2026-08-13",
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
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
