"""Does the test of `a b = 0` hold its level?

The vertical estimand is a product, so its null is a union: `a b = 0` holds when
the loading is nought, or the path is, or both. The test is an
intersection-union — reject only if both parts reject, so the p-value is the
larger of the two — and that construction is conservative by design. How
conservative, and whether it is ever anti-conservative, is a measurement.

Three nulls are run, because they are different nulls and a test can hold its
level against one and not another:

- **`a = 0`, path free.** The loading is non-negative, so this null sits on a
  boundary and its part of the test uses the even mixture of a point mass at
  nought and chi-square on one.
- **`b = 0`, loading free.** The path is interior and takes an ordinary
  chi-square.
- **both nought.** The corner, where the union rule takes the larger of two
  p-values that are each about right, and should be plainly conservative.

A real mediation is run alongside, so the level rows have something to be read
against.

**The design is chosen to be well behaved, on purpose.** Outcome statuses are
observed on everybody and families are sampled from the population, because the
question here is whether the test holds its level and not whether a particular
recruitment scheme identifies the model — `checks/mediation_power.py` asks that,
and finds that it often does not.

**How often the test is available is part of the answer, not a footnote.** Under
`a = 0` the model loses an identification. The inherited covariance carries the
loading only through `a` and `a(ab + c')`, so with `a` at nought the direct path
and the outcome loading enter it as `c'^2 + d^2` and nothing separates them. The
fit puts the whole of it in one and leaves the other at its bound, and the
vertical test then refuses — correctly, because the even mixture it reads the
loading against assumes the loading is the only parameter on a bound.

That refusal is not random with respect to the data, so a level measured on the
replicates that survived is a conditional one and is reported as such. Measured
over ten replicates: half are refused at 200 and 400 families and a fifth at
800, always for this reason, and under `b = 0` almost none are. A test that is
unavailable half the time under one of its own nulls is a finding about the
model, and the number belongs beside the level rather than under it.

Run with:

    ASTERISM_REPLICATES=400 uv run --no-project python checks/mediation_calibration.py
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
from scipy import stats

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "400"))
"""Set the number of simulations run under each generating truth."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "6"))
"""Set the number of worker processes used for calibration fits."""
# **How many draws a simulated reference may rest on.** Where the outcome
# loading is also on a bound the even mixture is the wrong reference and one is
# simulated instead, at two more fits per draw. Two hundred is right for a
# single answer and ruinous for a campaign of four hundred, so this is set low
# here and reported with the result. It coarsens the smallest p-value the
# reference can return -- fifty puts it near 0.02 -- which matters for the 0.01
# column and is why that column is read with the refusal counts beside it.
BOOTSTRAP: int = int(os.environ.get("ASTERISM_BOOTSTRAP", "50"))
"""Set the simulated-reference draws used for boundary vertical tests."""

FAMILIES: int = int(os.environ.get("ASTERISM_FAMILIES", "400"))
"""Set the sibling-pair count in every calibration replicate."""

LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)
"""Selected nominal rejection levels at which calibration is measured."""

TRUTHS: dict[str, dict[str, float]] = {
    "a = 0, b free": {"a": 0.0, "b": 0.4},
    "b = 0, a free": {"a": 0.6, "b": 0.0},
    "both nought": {"a": 0.0, "b": 0.0},
    "real mediation": {"a": 0.6, "b": 0.4},
}
"""Defined the two null branches, their corner and a mediated alternative."""

FIXED: dict[str, float] = {"c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}
"""Held nuisance mediation parameters fixed across generating truths."""

RELATIONSHIP: list[list[float]] = [[1.0, 0.5], [0.5, 1.0]]
"""Defined the twice-kinship matrix for each full-sibling pair."""


def one(job: tuple[str, int]) -> tuple[float | None, str | None]:
    """Return the vertical-test p-value or stable refusal for one replicate.

    Args:
        job: Named generating truth and deterministic replicate index.

    Returns:
        Available p-value paired with no refusal, or the converse.
    """
    truth, replicate = job
    """Unpacked the generating truth and deterministic replicate index."""

    drawn: list[dict[str, object]] = simulate(
        relationship=RELATIONSHIP,
        **TRUTHS[truth],
        **FIXED,
        families=FAMILIES,
        seed=900_000 + 1_000 * replicate,
        outcome_threshold=0.3,
        measurement_error_variance=0.15,
        observe_outcome=True,
        ascertainment="population_unconditioned",
    )
    """Simulated one population-sampled sibling-pair calibration replicate."""

    try:
        return float(
            asterism.LatentMediationModel(drawn).test_vertical(
                bootstrap_replicates=BOOTSTRAP
            )["p_value"]
        ), None
    except ValueError as refusal:
        return None, str(refusal).replace("LATENT_MEDIATION_", "")


def main() -> int:
    print(
        f"Level of the union null test. {REPLICATES} replicates of {FAMILIES} "
        f"sibling pairs,\npopulation sampled with both statuses observed. "
        f"Truth elsewhere: c' {FIXED['c_prime']}, d {FIXED['d']}, "
        f"sigma_m2 {FIXED['sigma_m2']}.\n"
    )
    jobs: list[tuple[str, int]] = [
        (truth, replicate) for truth in TRUTHS for replicate in range(REPLICATES)
    ]
    """Enumerated every generating truth and replicate combination."""

    started: float = time.perf_counter()
    """Captured the start of the complete calibration campaign."""

    with ProcessPoolExecutor(WORKERS) as pool:
        answers: list[tuple[float | None, str | None]] = list(
            pool.map(one, jobs, chunksize=1)
        )
        """Ran every calibration replicate across worker processes."""

    print(
        f"{len(jobs):,} fits in {(time.perf_counter() - started) / 60:.0f} minutes.\n"
    )

    gathered: dict[str, list[float]] = {truth: [] for truth in TRUTHS}
    """Initialised available p-values indexed by their generating truth."""

    refused: dict[str, Counter[str]] = {truth: Counter() for truth in TRUTHS}
    """Initialised stable refusal counts indexed by their generating truth."""

    for (truth, _), (answer, refusal) in zip(jobs, answers, strict=True):
        if answer is None:
            refused[truth][refusal] += 1
            """Counted the stable reason this replicate supplied no vertical test."""

        else:
            gathered[truth].append(answer)

    failures: list[str] = []
    """Collected inadequate fit availability and anti-conservative null cells."""

    recorded: dict[str, dict[str, object]] = {}
    """Initialised the machine-readable evidence record by generating truth."""

    header: str = "".join(f"{level:>10g}" for level in LEVELS)
    """Rendered aligned column headings for the selected nominal levels."""

    print(f"  {'truth':<18}{header}{'of':>7}{'refused':>9}  verdict")
    for truth in TRUTHS:
        values: np.ndarray = np.array(gathered[truth])
        """Collected available p-values under the current generating truth."""

        gone: Counter[str] = refused[truth]
        """Collected refusal-code frequencies under the same truth."""

        if values.size < 0.25 * REPLICATES:
            worst: str = gone.most_common(1)[0][0] if gone else "nothing fitted"
            """Selected the dominant reason for inadequate test availability."""

            failures.append(
                f"{truth}: only {values.size} of {REPLICATES} fitted, mostly {worst}"
            )
        if values.size == 0:
            print(
                f"  {truth:<18}{'--':>30}{0:>7}{sum(gone.values()):>9}  nothing fitted"
            )
            continue
        rates: list[float] = [float((values < level).mean()) for level in LEVELS]
        """Calculated conditional rejection rates at every nominal level."""

        is_null: bool = TRUTHS[truth]["a"] == 0.0 or TRUTHS[truth]["b"] == 0.0
        """Identified whether the generating paths satisfied the union null."""

        verdict: str = "power"
        """Initialised the alternative-cell interpretation before null checks."""

        if is_null:
            over: list[str] = []
            """Initialised descriptions of nominal levels exceeding calibration bounds."""

            for level, rate in zip(LEVELS, rates, strict=True):
                ceiling: float = float(
                    level + 1.96 * np.sqrt(level * (1 - level) / values.size)
                )
                """Calculated the upper 95 per cent binomial calibration bound."""

                if rate > ceiling:
                    over.append(f"{rate:.3f} at {level:g} where at most {ceiling:.3f}")
            if over:
                verdict = "OVER"
                """Marked a null truth as anti-conservative at one or more levels."""

                failures.append(f"{truth} rejects " + "; ".join(over))
            else:
                # An intersection-union test is meant to be conservative, and
                # saying which it is matters: conservative costs power, over
                # costs correctness.
                verdict = "level" if rates[1] > 0.5 * LEVELS[1] else "conservative"
                """Distinguished approximately level from deliberately conservative tests."""

        recorded[truth] = {
            "rejected": dict(zip([str(level) for level in LEVELS], rates, strict=True)),
            "fitted": int(values.size),
            "refused": dict(gone),
            "judged_as": "level" if is_null else "power",
        }
        """Recorded rejection, availability and interpretation under this truth."""

        if is_null:
            recorded[truth]["uniform_ks_p"] = float(
                stats.kstest(values, "uniform").pvalue
            )
            """Recorded a descriptive uniformity test for available null p-values."""
        print(
            f"  {truth:<18}"
            + "".join(f"{rate:>10.3f}" for rate in rates)
            + f"{values.size:>7}{sum(gone.values()):>9}  {verdict}"
        )

    print(
        "\nAn intersection-union test rejects only when both parts do, so it is "
        "expected to\nsit at or below its level rather than on it. Conservative "
        "costs power; over costs correctness."
    )
    print(
        "\nRead the refused column beside the rates. Under `a = 0` the direct "
        "path and the\noutcome loading enter the inherited covariance only as "
        "`c'^2 + d^2`, nothing\nseparates them, one of them lands on its bound "
        "and the test refuses. That refusal\nis not random with respect to the "
        "data, so a rate measured on what survived is a\nconditional one."
    )
    if failures:
        print("\nNOT CALIBRATED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(
        json.dumps(
            {
                "what": "level of the intersection-union test of a b = 0",
                "date": date.today().isoformat(),
                "replicates": REPLICATES,
                "bootstrap_replicates_where_simulated": BOOTSTRAP,
                "families": FAMILIES,
                "people_per_family": len(RELATIONSHIP),
                "fixed": FIXED,
                "truths": TRUTHS,
                "results": recorded,
                "availability_is_data_dependent": (
                    "under a = 0 the direct path and the outcome loading enter "
                    "the inherited covariance only as c'^2 + d^2 and are not "
                    "separately identified; one lands on its bound and the test "
                    "refuses, so a rate measured on what survived is conditional"
                ),
                "note": (
                    "the design is deliberately well behaved -- population "
                    "sampled, every status observed -- because this measures the "
                    "test and not whether a recruitment scheme identifies the "
                    "model, which checks/mediation_power.py asks separately"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
