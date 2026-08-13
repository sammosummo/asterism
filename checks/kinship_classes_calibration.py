"""Do the class-weighted kinship shares cover, and are they unbiased?

The class-weighted kinship model asks whether a mother and her son resemble one
another by the same amount as a father and his daughter. It multiplies only the
four classes of direct parent–offspring cells inside the relationship matrix,
which is linear in the weights, so it fits as six ordinary variance components.

**This check exists because the model has failed a calibration gate before.** In
2026 an independent qualification failed it on "multiple parent-weight bias and
coverage requirements". An implementation written fresh reproduces that failure,
so the gate was right — and what it caught is the *reporting parameterisation*
rather than the model. A weight is a ratio of two estimated variances,
`w = v_c / s_A`, and such a ratio is biased upward when the denominator is
uncertain. A share of the total variance is not a ratio of two estimates and
does not carry that fault.

So both are computed here and reported side by side. The check passes or fails on
the shares, which is what the model should report; the weights are shown so that
the size of the thing being avoided stays visible rather than becoming folklore.

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
from pathlib import Path

import numpy as np

from asterism import _core

WORKERS = int(os.environ.get("ASTERISM_WORKERS", "3"))
REPLICATES = 200
# Roughly the GOBS balance, scaled to keep the run affordable.
FAMILIES = {"mother_son": 100, "mother_daughter": 140, "father_son": 50, "father_daughter": 70}
NAMES = list(FAMILIES)

# Every weight is truly one: the null the model is built to test against. If the
# shares are biased here, a difference between classes could be manufactured
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

# The covariance the data are drawn from, and the shares it implies.
WEIGHTED = MATRICES[0] + sum(TRUE_WEIGHT * m for m in MATRICES[1:])
COVARIANCE = SIGMA_A * WEIGHTED + SIGMA_E * np.eye(N)
FACTOR = np.linalg.cholesky(COVARIANCE + 1e-9 * np.eye(N))
TRUE_VARIANCES = [SIGMA_A] + [SIGMA_A * TRUE_WEIGHT] * 4 + [SIGMA_E]
TRUE_SHARES = [v / sum(TRUE_VARIANCES) for v in TRUE_VARIANCES]


def one(index: int) -> dict | None:
    y = FACTOR @ np.random.default_rng(620_000 + index).standard_normal(N)
    try:
        variances, shares, total, loglik, gradient, converged = _core.component_fit(
            MATRICES, DESIGN, y, True
        )
    except Exception:
        return None
    if variances[0] <= 0:
        return None
    out = {"weights": [variances[i] / variances[0] for i in range(1, 5)], "shares": [], "covered": []}
    for component in range(1, 5):
        out["shares"].append(shares[component])
        try:
            lower, upper, _, _, _ = _core.component_interval(
                MATRICES, DESIGN, y, component, True
            )
            out["covered"].append(bool(lower <= TRUE_SHARES[component] <= upper))
        except Exception:
            out["covered"].append(None)
    return out


def main() -> int:
    print(
        f"Class-weighted kinship, REML. {N} people, every true weight "
        f"{TRUE_WEIGHT}.\nPairs per class: "
        + ", ".join(f"{n} {c}" for c, n in zip(CLASS_NAMES, PAIRS))
        + ".\n"
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        results = [r for r in pool.map(one, range(REPLICATES)) if r]
    print(f"{len(results)} of {REPLICATES} replicates in "
          f"{(time.perf_counter() - started) / 60:.0f} minutes.\n")

    failures = []
    print(
        f"{'class':<18}{'pairs':>7}{'share':>10}{'bias':>9}{'coverage':>11}"
        f"{'band':>18}{'weight':>10}{'bias':>9}"
    )
    recorded = {}
    for component, name in enumerate(CLASS_NAMES):
        shares = np.array([r["shares"][component] for r in results])
        weights = np.array([r["weights"][component] for r in results])
        covered = [r["covered"][component] for r in results if r["covered"][component] is not None]
        truth = TRUE_SHARES[component + 1]
        share_bias = shares.mean() - truth
        hit = float(np.mean(covered)) if covered else float("nan")
        error = np.sqrt(0.95 * 0.05 / max(len(covered), 1))
        low, high = 0.95 - 1.96 * error, 0.95 + 1.96 * error
        recorded[name] = {
            "pairs": PAIRS[component],
            "true_share": truth,
            "share_mean": float(shares.mean()),
            "share_bias": float(share_bias),
            "coverage": hit,
            "intervals": len(covered),
            "weight_mean": float(weights.mean()),
            "weight_bias": float(weights.mean() - TRUE_WEIGHT),
        }
        flag = "ok" if low <= hit <= high else ("conservative" if hit > high else "UNDER")
        print(
            f"{name:<18}{PAIRS[component]:>7}{shares.mean():>10.4f}{share_bias:>+9.4f}"
            f"{hit:>11.3f}   [{low:.3f}, {high:.3f}] {flag:<13}"
            f"{weights.mean():>10.3f}{weights.mean() - TRUE_WEIGHT:>+9.3f}"
        )
        # The share is what the model reports and what this passes or fails on.
        if abs(share_bias) > 0.02:
            failures.append(f"{name} share is biased by {share_bias:+.4f}")
        if hit < low:
            failures.append(f"{name} covers {hit:.3f}, below the band")

    worst_weight = max(abs(r["weight_bias"]) for r in recorded.values())
    print(
        f"\nThe weights are shown to keep the size of the avoided fault visible: "
        f"the worst\nis biased by {worst_weight:+.2f} on a true value of "
        f"{TRUE_WEIGHT}. That is the parameterisation\nthe 2026 gate failed, and "
        f"failing it was right."
    )

    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nThe shares are unbiased and their intervals cover. Reported as shares,")
    print("this model is usable; reported as weights it is not, on this design.")

    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/kinship-classes-2026-08-13.json").write_text(
        json.dumps(
            {
                "what": "class-weighted kinship: do the class shares cover and are they unbiased",
                "date": "2026-08-13",
                "estimator": "reml",
                "people": N,
                "true_weight": TRUE_WEIGHT,
                "replicates": REPLICATES,
                "completed": len(results),
                "nominal": 0.95,
                "classes": recorded,
                "note": (
                    "the weights are ratios of two estimated variances and are biased "
                    "upward; the shares are not and are what the model should report. "
                    "The 2026 qualification gate failed this model on parent-weight "
                    "bias and was right to."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
