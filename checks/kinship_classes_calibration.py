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

import asterism
import numpy as np

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "3"))
"""Worker processes used for independent calibration replicates."""

REPLICATES: int = 200
"""Requested simulated data sets in the class-weight calibration."""

# Roughly the GOBS balance, scaled to keep the run affordable.
FAMILIES: dict[str, int] = {
    "mother_son": 100,
    "mother_daughter": 140,
    "father_son": 50,
    "father_daughter": 70,
}
"""Fixed unequal family counts for the four measured parent-child classes."""

NAMES: list[str] = list(FAMILIES)
"""Retained the historical insertion order of the four class labels."""

# Every weight is truly one: the null the model is built to test against. If the
# coefficient proportions are biased here, a difference between classes could be manufactured
# from nothing, which is the failure that matters.
TRUE_WEIGHT: float = 1.0
"""Set every simulated parent-child class multiplier to the equality null."""

SIGMA_A, SIGMA_E = 0.5, 0.5
"""Split the simulated covariance equally between kinship and residual terms."""


def structure() -> tuple[
    list[str],
    list[str | None],
    list[str | None],
    list[str],
    list[str],
]:
    """A pedigree with the four classes present in unequal numbers.

    Every child has both parents recorded, because the pedigree refuses a child
    with one known parent rather than guessing the other. The imbalance
    therefore comes from which parent is *measured*, which is also where the
    real imbalance comes from: GOBS has more measured mothers than fathers.

    Returns:
        Pedigree identifiers, parents, sexes and measured-person order.
    """
    ids, fathers, mothers, sexes, keep = [], [], [], [], []
    """Initialised the pedigree columns and measured-person selection."""

    for name, count in FAMILIES.items():
        parent_is_mother: bool = name.startswith("mother")
        """Identified which parent contributes the measured class member."""

        child_is_son: bool = name.endswith("son")
        """Identified the child's simulated sex from the class label."""

        for index in range(count):
            mother: str = f"{name}_m{index}"
            """Created the mother's unique participant-free identifier."""

            father: str = f"{name}_f{index}"
            """Created the father's unique participant-free identifier."""

            child: str = f"{name}_c{index}"
            """Created the child's unique participant-free identifier."""

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
"""Built the fixed pedigree columns and measured-person order once."""

CLASSES: dict[str, object] = asterism.kinship_classes(
    IDS,
    FATHERS,
    MOTHERS,
    SEXES,
    KEEP,
)
"""Built the class matrices and metadata through the documented public builder."""

MATRICES: list[np.ndarray] = CLASSES["matrices"]
"""Selected the five class-basis matrices returned by the public builder."""

ORDER: list[str] = CLASSES["order"]
"""Selected the documented row order shared by every class matrix."""

CLASS_NAMES: list[str] = CLASSES["class_names"]
"""Selected the stable class names used in evidence records."""

PAIRS: list[int] = [CLASSES["pairs"][name] for name in CLASS_NAMES]
"""Restored the historical ordered pair counts from the public named mapping."""
N: int = len(ORDER)
"""Counted measured people in the class matrices' common row order."""

DESIGN: np.ndarray = np.ones((N, 1))
"""Constructed the intercept-only fixed-effect design."""

MODEL: asterism.ComponentModel = asterism.ComponentModel(MATRICES, DESIGN)
"""Built the public component model shared by the worker processes."""

# The covariance the data are drawn from and its raw coefficient proportions.
WEIGHTED: np.ndarray = MATRICES[0] + sum(
    TRUE_WEIGHT * matrix for matrix in MATRICES[1:]
)
"""Combined additive and class bases at their fixed true weights."""

COVARIANCE: np.ndarray = SIGMA_A * WEIGHTED + SIGMA_E * np.eye(N)
"""Constructed the true covariance used for every phenotype draw."""

FACTOR: np.ndarray = np.linalg.cholesky(COVARIANCE + 1e-9 * np.eye(N))
"""Factored the true covariance after its fixed diagonal stabiliser."""

TRUE_COEFFICIENTS: list[float] = [SIGMA_A] + [SIGMA_A * TRUE_WEIGHT] * 4 + [SIGMA_E]
"""Recorded the six true model coefficients in component order."""

TRUE_PROPORTIONS: list[float] = [
    value / sum(TRUE_COEFFICIENTS) for value in TRUE_COEFFICIENTS
]
"""Converted true coefficients to the diagnostic raw proportions being scored."""


def one(index: int) -> dict[str, list[float | bool | None]] | None:
    """Fit and interval-score one reproducible class-weight replicate.

    Args:
        index: Replicate index added to the fixed random-number seed base.

    Returns:
        Fitted weights, raw proportions and coverage indicators, or ``None``
        when the model or required interval cannot be fitted.
    """
    y: np.ndarray = FACTOR @ np.random.default_rng(620_000 + index).standard_normal(N)
    """Drew one outcome from the fixed class-weight covariance."""

    try:
        got: dict[str, object] = MODEL.fit(y, reml=True)
        """Fitted the class model through its named public record."""
    except ValueError:
        return None
    variances: list[float] = got["variances"]
    """Read the fitted covariance coefficients by field name."""

    proportions: list[float] = got["raw_coefficient_proportions"]
    """Read the scale-dependent diagnostic proportions by field name."""

    if variances[0] <= 0:
        return None
    out: dict[str, list[float | bool | None]] = {
        "weights": [variances[i] / variances[0] for i in range(1, 5)],
        "coefficient_proportions": [],
        "covered": [],
    }
    """Initialised the fitted quantities and interval verdicts for this replicate."""

    for component in range(1, 5):
        out["coefficient_proportions"].append(proportions[component])
        try:
            interval: dict[str, object] | None = MODEL.interval(
                y,
                component,
                reml=True,
                quantity="raw_coefficient_proportion",
            )
            """Profiled the zero-diagonal class's diagnostic raw proportion."""
        except ValueError:
            interval = None
            """Recorded an unavailable profile interval without inventing coverage."""

        if interval is None:
            out["covered"].append(None)
        else:
            out["covered"].append(
                bool(
                    interval["lower"]
                    <= TRUE_PROPORTIONS[component]
                    <= interval["upper"]
                )
            )
    return out


def main() -> int:
    """Run the class-weight bias and interval-coverage calibration.

    Returns:
        Zero when bias and coverage gates pass, otherwise one.
    """
    print(
        f"Class-weighted kinship, REML. {N} people, every true weight "
        f"{TRUE_WEIGHT}.\nPairs per class: "
        + ", ".join(f"{n} {c}" for c, n in zip(CLASS_NAMES, PAIRS, strict=True))
        + ".\n"
    )
    started: float = time.perf_counter()
    """Started timing the complete calibration campaign."""

    with ProcessPoolExecutor(WORKERS) as pool:
        results: list[dict[str, list[float | bool | None]]] = [
            result for result in pool.map(one, range(REPLICATES)) if result
        ]
        """Ran and retained successfully fitted independent replicates."""

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
    print(
        f"{len(results)} of {REPLICATES} replicates in "
        f"{(time.perf_counter() - started) / 60:.0f} minutes.\n"
    )

    failures: list[str] = []
    """Collected substantive bias and under-coverage failures."""

    print(
        f"{'class':<18}{'pairs':>7}{'coef.prop':>10}{'bias':>9}{'coverage':>11}"
        f"{'band':>18}{'weight':>10}{'bias':>9}"
    )
    recorded: dict[str, dict[str, float | int]] = {}
    """Accumulated class-specific evidence for the final JSON record."""

    for component, name in enumerate(CLASS_NAMES):
        proportions: np.ndarray = np.array(
            [r["coefficient_proportions"][component] for r in results]
        )
        """Collected fitted raw coefficient proportions for this class."""

        weights: np.ndarray = np.array([r["weights"][component] for r in results])
        """Collected fitted class-to-additive coefficient ratios."""

        covered: list[float | bool | None] = [
            r["covered"][component]
            for r in results
            if r["covered"][component] is not None
        ]
        """Retained coverage verdicts only where a profile interval was available."""

        truth: float = TRUE_PROPORTIONS[component + 1]
        """Selected this class's true raw coefficient proportion."""

        proportion_bias: float = proportions.mean() - truth
        """Calculated the mean raw-proportion bias for this class."""

        hit: float = float(np.mean(covered)) if covered else float("nan")
        """Calculated coverage among the intervals that could be profiled."""

        error: float = np.sqrt(0.95 * 0.05 / max(len(covered), 1))
        """Calculated the binomial standard error under nominal 95% coverage."""

        low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
        """Constructed the fixed normal-approximation coverage band."""

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
        """Stored the complete class-specific calibration evidence."""

        flag: str = (
            "ok" if low <= hit <= high else ("conservative" if hit > high else "UNDER")
        )
        """Classified coverage relative to the acceptance band for display."""

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

    worst_weight: float = max(
        abs(record["weight_bias"]) for record in recorded.values()
    )
    """Found the largest absolute ratio bias retained for diagnostic reporting."""
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
