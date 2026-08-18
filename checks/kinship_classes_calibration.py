"""Do class coefficient-proportion intervals cover, and are weights biased?

The class-weighted kinship model asks whether a mother and her son resemble one
another by the same amount as a father and his daughter. It multiplies only the
four classes of direct parent–offspring cells inside the relationship matrix,
which is linear in the weights, so it fits as six ordinary variance components.

Earlier simulations found biased parent weights despite adequate interval
coverage. The issue is the *reporting parameterisation* rather than the model. A
weight is a ratio of two estimated coefficients, `w = v_c / s_A`, and such a
ratio is biased upward when the denominator is uncertain. The raw coefficient
proportion `v_c / sum(v)` does not carry that particular denominator fault.

Both are computed here and reported side by side. The check evaluates the
profile interval for the raw coefficient proportion and shows the weight bias.
Because the class matrices have zero diagonals, neither quantity is a share of
phenotypic variance; scientific reporting should use coefficients, the omnibus
equality test, and class contrasts.

**The classes are deliberately unbalanced**, because the real ones are: GOBS has
279 mother–daughter pairs and 102 father–son. A calibration on balanced classes
would say nothing about the class that matters least and is most likely to
mislead.

Run with:

    uv run --no-project python checks/kinship_classes_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from asterism import _core

WORKERS = int(os.environ.get("ASTERISM_WORKERS", "3"))
REPLICATES = 200
# Roughly the GOBS balance, scaled to keep the run affordable.
FAMILIES = {"mother_son": 100, "mother_daughter": 140, "father_son": 50, "father_daughter": 70}
NAMES = list(FAMILIES)

# Every weight is truly one: the null the model is built to test against. If the
# coefficient proportions are biased here, a difference between classes could be manufactured
# from nothing, which is the failure that matters.
TRUE_WEIGHT = 1.0
SIGMA_A, SIGMA_E = 0.5, 0.5


def structure():
    """A pedigree with the four classes present in unequal numbers.

    Every child has both parents recorded, because the pedigree refuses a child
    with one known parent rather than guessing the other. The imbalance
    therefore comes from which parent is *measured*, which is also where the
    real imbalance comes from: GOBS has more measured mothers than fathers.
    """
    ids, fathers, mothers, sexes, keep = [], [], [], [], []
    for name, count in FAMILIES.items():
        parent_is_mother = name.startswith("mother")
        child_is_son = name.endswith("son")
        for index in range(count):
            mother = f"{name}_m{index}"
            father = f"{name}_f{index}"
            child = f"{name}_c{index}"
            for person, sex in ((mother, "2"), (father, "1")):
                ids.append(person)
                fathers.append(None)
                mothers.append(None)
                sexes.append(sex)
            ids.append(child)
            fathers.append(father)
            mothers.append(mother)
            sexes.append("1" if child_is_son else "2")
            # Only the parent whose class this family stands for is measured.
            keep.append(mother if parent_is_mother else father)
            keep.append(child)
    return ids, fathers, mothers, sexes, keep


IDS, FATHERS, MOTHERS, SEXES, KEEP = structure()
MATRICES, ORDER, CLASS_NAMES, PAIRS = _core.kinship_classes(
    IDS, FATHERS, MOTHERS, SEXES, KEEP
)
MATRICES = [np.ascontiguousarray(m) for m in MATRICES]
N = len(ORDER)
DESIGN = np.ones((N, 1))

# The covariance the data are drawn from and its raw coefficient proportions.
WEIGHTED = MATRICES[0] + sum(TRUE_WEIGHT * m for m in MATRICES[1:])
COVARIANCE = SIGMA_A * WEIGHTED + SIGMA_E * np.eye(N)
FACTOR = np.linalg.cholesky(COVARIANCE + 1e-9 * np.eye(N))
TRUE_COEFFICIENTS = [SIGMA_A] + [SIGMA_A * TRUE_WEIGHT] * 4 + [SIGMA_E]
TRUE_PROPORTIONS = [v / sum(TRUE_COEFFICIENTS) for v in TRUE_COEFFICIENTS]


def one(index: int) -> dict | None:
    y = FACTOR @ np.random.default_rng(620_000 + index).standard_normal(N)
    # `component_fit` returns ten values, and unpacking six of them raised on
    # every replicate. A bare `except` then turned each into a dropped result,
    # so the coverage and bias figures below were computed from an empty list
    # while the check exited nought. The failure is now named and no longer
    # swallowed: a genuinely unfittable replicate is still dropped, but a
    # mistake in this file is not.
    #
    # The two leading underscores on the tail catch the same trap a second
    # time: unpacking a fixed count here means every future addition to the
    # compiled tuple silently empties this check again, and a wrong length
    # raises `ValueError` -- the same exception a real refusal to fit raises,
    # which is why it went unnoticed. `ComponentModel.fit` returns a named
    # dictionary and does not have this problem; this check predates it.
    try:
        got = _core.component_fit(MATRICES, DESIGN, y, True)
    except ValueError:
        return None
    if len(got) != 10:
        raise SystemExit(
            f"`component_fit` returned {len(got)} values, not the ten this "
            "check unpacks. Update the unpacking below rather than letting "
            "every replicate be dropped."
        )
    (
        variances,
        proportions,
        _total,
        _loglik,
        _gradient,
        _converged,
        _fixed_effects,
        _fixed_effect_errors,
        _stop_code,
        _stop_message,
    ) = got
    if variances[0] <= 0:
        return None
    out = {
        "weights": [variances[i] / variances[0] for i in range(1, 5)],
        "coefficient_proportions": [],
        "covered": [],
    }
    for component in range(1, 5):
        out["coefficient_proportions"].append(proportions[component])
        try:
            interval = _core.component_interval(MATRICES, DESIGN, y, component, True)
        except ValueError:
            interval = None
        if interval is None:
            out["covered"].append(None)
        else:
            lower, upper, _, _, _, _ = interval
            out["covered"].append(bool(lower <= TRUE_PROPORTIONS[component] <= upper))
    return out


def main() -> int:
    print(
        f"Class-weighted kinship, REML. {N} people, every true weight "
        f"{TRUE_WEIGHT}.\nPairs per class: "
        + ", ".join(f"{n} {c}" for c, n in zip(CLASS_NAMES, PAIRS, strict=True))
        + ".\n"
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        results = [r for r in pool.map(one, range(REPLICATES)) if r]
    # **A calibration computed from nothing is not a calibration.** Dropping a
    # replicate that genuinely could not be fitted is reasonable; dropping every
    # one of them and reporting coverage anyway is how this file passed for
    # weeks with an unpacking mistake in it.
    if len(results) < REPLICATES // 2:
        print(
            f"NOT CALIBRATED: only {len(results)} of {REPLICATES} replicates "
            "were fitted, which is too few to report anything from."
        )
        return 1
    print(f"{len(results)} of {REPLICATES} replicates in "
          f"{(time.perf_counter() - started) / 60:.0f} minutes.\n")

    failures = []
    print(
        f"{'class':<18}{'pairs':>7}{'coef.prop':>10}{'bias':>9}{'coverage':>11}"
        f"{'band':>18}{'weight':>10}{'bias':>9}"
    )
    recorded = {}
    for component, name in enumerate(CLASS_NAMES):
        proportions = np.array(
            [r["coefficient_proportions"][component] for r in results]
        )
        weights = np.array([r["weights"][component] for r in results])
        covered = [r["covered"][component] for r in results if r["covered"][component] is not None]
        truth = TRUE_PROPORTIONS[component + 1]
        proportion_bias = proportions.mean() - truth
        hit = float(np.mean(covered)) if covered else float("nan")
        error = np.sqrt(0.95 * 0.05 / max(len(covered), 1))
        low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
        recorded[name] = {
            "pairs": PAIRS[component],
            "true_coefficient_proportion": truth,
            "coefficient_proportion_mean": float(proportions.mean()),
            "coefficient_proportion_bias": float(proportion_bias),
            "coverage": hit,
            "intervals": len(covered),
            "weight_mean": float(weights.mean()),
            "weight_bias": float(weights.mean() - TRUE_WEIGHT),
        }
        flag = "ok" if low <= hit <= high else ("conservative" if hit > high else "UNDER")
        print(
            f"{name:<18}{PAIRS[component]:>7}{proportions.mean():>10.4f}{proportion_bias:>+9.4f}"
            f"{hit:>11.3f}   [{low:.3f}, {high:.3f}] {flag:<13}"
            f"{weights.mean():>10.3f}{weights.mean() - TRUE_WEIGHT:>+9.3f}"
        )
        if abs(proportion_bias) > 0.02:
            failures.append(
                f"{name} coefficient proportion is biased by {proportion_bias:+.4f}"
            )
        if hit < low:
            failures.append(f"{name} covers {hit:.3f}, below the band")

    worst_weight = max(abs(r["weight_bias"]) for r in recorded.values())
    print(
        f"\nThe weights are shown to keep the size of the avoided fault visible: "
        f"the worst\nis biased by {worst_weight:+.2f} on a true value of "
        f"{TRUE_WEIGHT}. Raw coefficient proportions avoid that particular ratio "
        f"bias but are scale-dependent."
    )

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nRaw coefficient proportions are unbiased and their intervals cover")
    print("for this fixed matrix coding. They are not phenotypic variance shares.")

    print(
        json.dumps(
            {
                "estimator": "reml",
                "people": N,
                "true_weight": TRUE_WEIGHT,
                "replicates": REPLICATES,
                "completed": len(results),
                "nominal": 0.95,
                "classes": recorded,
                "note": (
                    "the weights are ratios of two estimated variances and are biased "
                    "upward; raw coefficient proportions avoid that bias here but "
                    "remain scale-dependent, so report coefficients, equality tests, "
                    "and contrasts for zero-diagonal class bases"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
