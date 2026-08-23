"""How many offspring per adjudicated case?

The application has one genuinely open design question: whether to enrol
several offspring per case. Doing so cuts the number of adjudicated cases from
250 to nearer 184, because adjudication is what the budget buys, while making
each family larger. Fewer, larger families carry more information per case and
less independent information overall, and which way that trades cannot be
argued from first principles — it has to be measured.

So this simulates each candidate design from a known truth, fits it, and counts
how often the union null `a b = 0` is rejected. Two things are measured, and the
first is the one that makes the second mean anything:

- **Level**, under a truth where the null holds. `a = 0` with `b` free, and
  `b = 0` with `a` free, are different nulls and are both run: the union rule
  reports whichever part is larger, so a design could hold its level against one
  and not the other.
- **Power**, under a real mediation.

Families are drawn conditional on the proband being a case, which is what a
clinic roster is and what the model's denominator assumes.

**What this cannot say.** It measures the model as it stands, which has no
covariates, no age indexing and only single-proband conditioning. Those change
power somewhat; they do not change which design carries more of it, which is
the question here. Nothing here is a claim about the real cohort.

Run with:

    ASTERISM_REPLICATES=400 uv run --no-project python checks/mediation_power.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import asterism
import numpy as np
from asterism.latent_mediation import simulate

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "400"))
"""Set the simulation replicates per design and truth from the environment."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "6"))
"""Set the number of worker processes used for the design sweep."""
# **How many draws a simulated reference may rest on.** Where the outcome
# loading is also on a bound the even mixture is the wrong reference and one is
# simulated instead, at two more fits per draw. Two hundred is right for a
# single answer and ruinous for a campaign of four hundred, so this is set low
# here and reported with the result. It coarsens the smallest p-value the
# reference can return -- fifty puts it near 0.02 -- which matters for the 0.01
# column and is why that column is read with the refusal counts beside it.
BOOTSTRAP: int = int(os.environ.get("ASTERISM_BOOTSTRAP", "50"))
"""Set the simulated-reference draws used by boundary vertical tests."""

LEVEL: float = 0.05
"""Set the nominal level used to judge calibration and measure power."""

# The truth to draw from. `a` is the mediator's loading on the inherited factor,
# `b` the path from mediator to outcome.
TRUTHS: dict[str, dict[str, float]] = {
    "mediation": {"a": 0.6, "b": 0.4},
    "no loading": {"a": 0.0, "b": 0.4},
    "no path": {"a": 0.6, "b": 0.0},
}
"""Defined one mediated alternative and the two branches of the union null."""

FIXED: dict[str, float] = {"c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}
"""Held the direct path, outcome loading and mediator residual variance fixed."""
# A dementia threshold giving roughly a sixth affected in the population, before
# the clinic conditioning that makes every proband one.
OUTCOME_THRESHOLD: float = 1.0
"""Set the population dementia threshold used before clinic ascertainment."""

# Adjudication is what the budget buys, so the designs are named by how many
# people it can afford to adjudicate and how many offspring each brings.
#
# **The sampling axis matters more than the offspring axis and is swept first.**
# Offspring are far too young for a dementia status, so the adjudicated person
# is the only outcome in a family. If every adjudicated person is a case, every
# observed outcome is a one, and there is nothing left to estimate the outcome's
# inherited loading from -- `d` goes to its bound, the vertical test refuses,
# and the design cannot be powered because it cannot be fitted.
SAMPLING: list[tuple[str, str, int | None]] = [
    ("adjudicated, cases and not", "population_unconditioned", None),
    ("cases only", "condition_on_named_proband_case", 0),
]
"""Defined population and affected-proband sampling schemes for comparison."""

DESIGNS: list[tuple[str, int, int]] = [
    ("250 adjudicated, 1 offspring", 250, 1),
    ("184 adjudicated, 2 offspring", 184, 2),
    ("184 adjudicated, 3 offspring", 184, 3),
]
"""Defined candidate adjudication budgets and offspring counts."""


def relationship(offspring: int) -> list[list[float]]:
    """Return twice kinship for one proband and full-sibling offspring.

    Args:
        offspring: Number of full-sibling offspring enrolled with the proband.

    Returns:
        Unit-diagonal matrix with one-half relationships among all relatives.
    """
    size: int = offspring + 1
    """Counted the proband and enrolled offspring in the family."""

    matrix: np.ndarray = np.eye(size)
    """Initialised the family relationship matrix with unit diagonal."""

    matrix[0, 1:] = matrix[1:, 0] = 0.5
    """Assigned half relationships between the proband and every offspring."""

    for i in range(1, size):
        for j in range(1, size):
            if i != j:
                matrix[i, j] = 0.5
                """Assigned half relationships among distinct full-sibling offspring."""

    return [[float(value) for value in row] for row in matrix]


def one(job: tuple[str, int, int, str, str, int]) -> tuple[float | None, str | None]:
    """One replicate: draw the design, fit it, return the union-null p-value.

    A refusal comes back with its code rather than as a bare failure, because
    which refusal it is says what is wrong with the design.
    """
    _, families, offspring, sampling, truth, replicate = job
    """Unpacked the design, sampling scheme, generating truth and replicate index."""

    ascertainment, proband = next(
        (how, who) for name, how, who in SAMPLING if name == sampling
    )
    """Resolved public model ascertainment settings from the named sampling scheme."""

    drawn: list[dict[str, object]] = simulate(
        relationship=relationship(offspring),
        **TRUTHS[truth],
        **FIXED,
        families=families,
        seed=500_000 + 1_000 * replicate + 7 * offspring,
        outcome_threshold=OUTCOME_THRESHOLD,
        # The adjudicated person's dementia status is observed; the offspring
        # are enrolled for hearing and are far too young to have one.
        observe_outcome=[True] + [False] * offspring,
        # Hearing is measured on everybody who is enrolled.
        measurement_error_variance=0.15,
        ascertainment=ascertainment,
        proband_index=proband,
    )
    """Simulated one candidate-design replicate under the selected truth."""

    try:
        return (
            float(
                asterism.LatentMediationModel(drawn).test_vertical(
                    bootstrap_replicates=BOOTSTRAP
                )["p_value"]
            ),
            None,
        )
    except ValueError as refusal:
        return None, str(refusal).replace("LATENT_MEDIATION_", "")


def main() -> int:
    print(
        f"Mediation design sweep. {REPLICATES} replicates per cell, "
        f"proband-conditioned.\nTruth outside the two swept paths: "
        f"c' {FIXED['c_prime']}, d {FIXED['d']}, sigma_m2 {FIXED['sigma_m2']}; "
        f"outcome threshold {OUTCOME_THRESHOLD}.\n"
    )
    jobs: list[tuple[str, int, int, str, str, int]] = [
        (name, families, offspring, sampling, truth, replicate)
        for sampling, _, _ in SAMPLING
        for name, families, offspring in DESIGNS
        for truth in TRUTHS
        for replicate in range(REPLICATES)
    ]
    """Enumerated every sampling, design, truth and replicate combination."""

    started: float = time.perf_counter()
    """Captured the start of the complete parallel design sweep."""

    with ProcessPoolExecutor(WORKERS) as pool:
        answers: list[tuple[float | None, str | None]] = list(
            pool.map(one, jobs, chunksize=1)
        )
        """Ran every design replicate across the configured worker processes."""

    took: float = time.perf_counter() - started
    """Measured elapsed time for the complete design sweep."""

    gathered: dict[tuple[str, str, str], list[float]] = {}
    """Initialised fitted p-values indexed by sampling, design and truth."""

    refused: dict[tuple[str, str, str], Counter[str]] = {}
    """Initialised refusal-code counts indexed by sampling, design and truth."""

    for job, (answer, refusal) in zip(jobs, answers, strict=True):
        key: tuple[str, str, str] = (job[3], job[0], job[4])
        """Located the evidence cell for this replicate's sampling, design and truth."""

        gathered.setdefault(key, [])
        refused.setdefault(key, Counter())
        if answer is None:
            refused[key][refusal] += 1
            """Counted the stable refusal explaining why this replicate had no test."""

        else:
            gathered[key].append(answer)

    print(f"{len(jobs):,} fits in {took / 60:.0f} minutes.\n")
    print(
        f"  {'sampling':<28}{'design':<30}{'truth':<12}"
        f"{'rejected':>10}{'of':>7}{'refused':>9}  verdict"
    )
    failures: list[str] = []
    """Collected cells whose fit availability or null level was unacceptable."""

    recorded: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    """Initialised the nested evidence record for every design cell."""

    for sampling, _, _ in SAMPLING:
        for name, _, _ in DESIGNS:
            for truth in TRUTHS:
                values: np.ndarray = np.array(gathered[(sampling, name, truth)])
                """Collected available p-values for the current design cell."""

                gone: Counter[str] = refused[(sampling, name, truth)]
                """Collected refusal-code frequencies for the same design cell."""

                if values.size < 0.5 * REPLICATES:
                    worst: tuple[str, int] = (
                        gone.most_common(1)[0] if gone else ("nothing fitted", 0)
                    )
                    """Selected the dominant explanation for inadequate fit availability."""

                    failures.append(
                        f"{sampling} / {name} / {truth}: only {values.size} of "
                        f"{REPLICATES} fitted, mostly {worst[0]}"
                    )
                if values.size == 0:
                    print(
                        f"  {sampling:<28}{name:<30}{truth:<12}"
                        f"{'--':>10}{0:>7}{sum(gone.values()):>9}  nothing fitted"
                    )
                    continue
                rate: float = float((values < LEVEL).mean())
                """Calculated the rejection proportion among available tests."""

                error: float = float(np.sqrt(LEVEL * (1 - LEVEL) / values.size))
                """Calculated the binomial standard error under the nominal null level."""

                ceiling: float = LEVEL + 1.96 * error
                """Set the upper 95 per cent calibration bound for this fitted count."""

                is_null: bool = TRUTHS[truth]["a"] == 0.0 or TRUTHS[truth]["b"] == 0.0
                """Identified whether the generating paths satisfied the union null."""

                verdict: str = "power"
                """Initialised the alternative-cell verdict before null calibration."""

                if is_null:
                    verdict = "OVER" if rate > ceiling else "level"
                    """Judged a null cell against its binomial calibration ceiling."""

                    if rate > ceiling:
                        failures.append(
                            f"{sampling} / {name} under {truth} rejects "
                            f"{rate:.3f} where at most {ceiling:.3f} is allowed"
                        )
                recorded.setdefault(sampling, {}).setdefault(name, {})[truth] = {
                    "rejected": rate,
                    "fitted": int(values.size),
                    "refused": dict(gone),
                    "judged_as": "level" if is_null else "power",
                }
                """Recorded rejection, availability and interpretation for this cell."""
                print(
                    f"  {sampling:<28}{name:<30}{truth:<12}{rate:>10.3f}"
                    f"{values.size:>7}{sum(gone.values()):>9}  {verdict}"
                )

    print(
        "\nRead the power rows against the level rows above them. A design that "
        "rejects\nmore often under a true null has not found more mediation, it "
        "has found less\ndiscipline, and its power is not comparable."
    )
    if failures:
        print("\nNOT USABLE:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(
        json.dumps(
            {
                "what": "power and level of the union null across candidate designs",
                "date": date.today().isoformat(),
                "replicates": REPLICATES,
                "bootstrap_replicates_where_simulated": BOOTSTRAP,
                "level": LEVEL,
                "truths": TRUTHS,
                "fixed": FIXED,
                "outcome_threshold": OUTCOME_THRESHOLD,
                "designs": recorded,
                "note": (
                    "the model as it stands has no covariates, no age indexing "
                    "and only single-proband conditioning; those change power but "
                    "not which design carries more of it. nothing here is a claim "
                    "about the real cohort"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
