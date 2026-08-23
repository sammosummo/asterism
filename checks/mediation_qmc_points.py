"""How many quasi-Monte Carlo points does the latent mediation model need?

The model integrates over a latent vector of twice the family size,
deterministically to dimension two and by quasi-Monte Carlo above it. The number of
points is a dial: more is more accurate and linearly more expensive, and at
grant scale the difference is hours.

**The dial cannot be set by taste.** A likelihood ratio is a difference of two
log likelihoods, and a chi-square on one degree of freedom has a critical value
of 3.84, so an integration error of a hundredth moves a test by nothing and an
error of one moves it by a lot. That gives the criterion used here: a setting
passes when its log likelihood sits within `TOLERANCE` of the richest setting
tried, over families of the shapes the analysis actually contains.

This asks about the integration alone, at a fixed parameter point, because a
fit would confound integration error with the optimiser's own path. If the
likelihood surface is the same then the fit is too.

Run with:

    uv run --no-project python checks/mediation_qmc_points.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import asterism
import numpy as np
from asterism.latent_mediation import simulate

POINTS: list[int] = [512, 1024, 2048, 4096, 8192, 16384]
"""Selected candidate quasi-Monte Carlo budgets below the reference setting."""

REFERENCE: int = 32768
"""Set the richest quasi-Monte Carlo budget used as the numerical reference."""
# A log likelihood this far out moves a deviance by twice as much, which is
# still two orders below the 3.84 a chi-square on one asks for.
TOLERANCE: float = 0.02
"""Set the largest accepted absolute log-likelihood difference from reference."""

# The shapes the proposed design actually holds: mostly small families, some
# up to six, with everybody's outcome observed -- which is what puts every
# person into the region rather than the density.
FAMILY_SIZES: list[int] = [2, 3, 4, 5, 6]
"""Selected the family sizes represented in the intended mediation design."""

FAMILIES_EACH: int = 40
"""Set the number of simulated families included at every selected size."""

TRUTH: dict[str, float] = {
    "a": 0.6,
    "b": 0.4,
    "c_prime": 0.2,
    "d": 0.7,
    "sigma_m2": 0.5,
}
"""Defined the mediation parameters used to generate the standing design."""
# **One parameter point cannot bound the error of a quasi-Monte Carlo rule.**
# The error is not monotone in the number of points -- different counts give
# different lattice alignments -- so a setting that happens to land close at
# one point can be well out at another. The worst case over several points is
# what a setting has to survive, and near a boundary is where it is worst,
# because that is where the region probabilities are smallest.
AT: list[dict[str, float]] = [
    {"a": 0.55, "b": 0.35, "c_prime": 0.25, "d": 0.65, "sigma_m2": 0.55},
    {"a": 0.20, "b": 0.60, "c_prime": -0.30, "d": 0.40, "sigma_m2": 0.30},
    {"a": 0.80, "b": 0.10, "c_prime": 0.05, "d": 0.90, "sigma_m2": 0.80},
    {"a": 0.05, "b": 0.45, "c_prime": 0.40, "d": 0.15, "sigma_m2": 1.20},
    {"a": 0.60, "b": 0.40, "c_prime": 0.20, "d": 0.70, "sigma_m2": 0.50},
]
"""Selected interior and near-boundary parameter points for error measurement."""


def relationship(size: int) -> list[list[float]]:
    """Return a full-sibling relationship matrix of the requested size.

    Args:
        size: Number of relatives represented by the matrix.

    Returns:
        Twice-kinship matrix with unit diagonal and half-related off-diagonal.
    """
    relationship_matrix: np.ndarray = np.eye(size)
    """Initialised an identity relationship matrix for the requested relatives."""

    for i in range(size):
        for j in range(size):
            if i != j:
                relationship_matrix[i, j] = 0.5
                """Assigned the full-sibling relationship between distinct relatives."""

    return [[float(value) for value in row] for row in relationship_matrix]


def build() -> list[dict[str, object]]:
    """Return the deterministic mixed-family-size mediation design."""
    families: list[dict[str, object]] = []
    """Initialised the family records spanning every selected size."""

    for size in FAMILY_SIZES:
        for f in range(FAMILIES_EACH):
            families.extend(
                simulate(
                    relationship=relationship(size),
                    **TRUTH,
                    families=1,
                    seed=770_000 + 131 * size + f,
                    outcome_prevalence=[0.08] * size,
                    observe_outcome=[True] * size,
                    measurement_error_variance=0.15,
                    ascertainment="population_unconditioned",
                    proband_index=None,
                )
            )
            """Added one deterministically seeded family of the current size."""
    return families


def main() -> int:
    families: list[dict[str, object]] = build()
    """Built the standing design shared by every integration setting."""

    people: int = sum(len(family["outcome_status"]) for family in families)
    """Counted all people represented by the simulated family records."""

    print(f"{len(families)} families, {people} people, sizes {FAMILY_SIZES}")
    print(
        f"reference: {REFERENCE} points; tolerance {TOLERANCE} in log likelihood\n",
        flush=True,
    )

    reference_model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        families, qmc_points=REFERENCE
    )
    """Prepared the richest integration setting used as the comparison reference."""

    started: float = time.time()
    """Captured the start of the reference likelihood evaluations."""

    reference_loglik: list[float] = [
        float(reference_model.evaluate(**at)["log_likelihood"]) for at in AT
    ]
    """Evaluated reference log likelihoods at every selected parameter point."""

    reference_seconds: float = (time.time() - started) / len(AT)
    """Calculated mean elapsed time per reference likelihood evaluation."""

    print(
        f"reference log likelihoods at {len(AT)} points, "
        f"{reference_seconds:.2f}s each\n"
    )

    print(
        f"{'points':>7} | {'worst difference':>16} | {'median':>10} | "
        f"{'seconds':>8} | {'speedup':>8} | verdict"
    )
    print("-" * 74)
    rows: list[dict[str, object]] = []
    """Initialised evidence rows for every candidate integration budget."""

    failures: list[str] = []
    """Collected conditions that prevent selecting an adequate point budget."""

    for points in POINTS:
        model: asterism.LatentMediationModel = asterism.LatentMediationModel(
            families, qmc_points=points
        )
        """Prepared the candidate integration setting on the standing design."""

        started = time.time()
        """Captured the start of the candidate likelihood evaluations."""

        differences: list[float] = [
            abs(float(model.evaluate(**at)["log_likelihood"]) - ref)
            for at, ref in zip(AT, reference_loglik, strict=True)
        ]
        """Measured candidate disagreement from reference at each parameter point."""

        took: float = (time.time() - started) / len(AT)
        """Calculated mean elapsed time per candidate likelihood evaluation."""

        worst: float = max(differences)
        """Selected the largest parameter-point disagreement for this budget."""

        ok: bool = worst <= TOLERANCE
        """Judged the candidate budget against the pre-written likelihood tolerance."""

        print(
            f"{points:>7} | {worst:>16.2e} | {np.median(differences):>10.2e} | "
            f"{took:>8.2f} | {reference_seconds / took:>7.1f}x | "
            f"{'ok' if ok else 'TOO COARSE'}",
            flush=True,
        )
        rows.append(
            {
                "points": points,
                "worst_difference": worst,
                "median_difference": float(np.median(differences)),
                "all_differences": differences,
                "seconds": took,
                "within_tolerance": ok,
            }
        )
        """Recorded accuracy and timing evidence for this integration budget."""

    usable: list[dict[str, object]] = [row for row in rows if row["within_tolerance"]]
    """Retained candidate settings satisfying the log-likelihood tolerance."""

    cheapest: dict[str, object] | None = (
        min(usable, key=lambda row: row["points"]) if usable else None
    )
    """Selected the smallest adequate point budget when one existed."""

    if cheapest is None:
        failures.append("no setting tried came within tolerance of the reference")
    else:
        print(
            f"\nCheapest setting within tolerance: {cheapest['points']} points, "
            f"{reference_seconds / cheapest['seconds']:.1f} times faster than "
            f"the reference"
        )
        if cheapest["points"] < 8192:
            print(
                f"  The package default is 8192. {cheapest['points']} is "
                f"{8192 / cheapest['points']:.0f} times cheaper and agrees to "
                f"{cheapest['worst_difference']:.1e} at worst."
            )

    receipt: dict[str, object] = {
        "what": "how many quasi-Monte Carlo points the latent mediation model needs",
        "date": date.today().isoformat(),
        "families": len(families),
        "people": people,
        "family_sizes": FAMILY_SIZES,
        "reference_points": REFERENCE,
        "reference_log_likelihood": reference_loglik,
        "parameter_points": len(AT),
        "tolerance": TOLERANCE,
        "settings": rows,
        "cheapest_within_tolerance": cheapest["points"] if cheapest else None,
        "passed": not failures,
        "failures": failures,
    }
    """Assembled the dated accuracy and runtime evidence for point selection."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / (f"mediation-qmc-points-{receipt['date']}.json")
    )
    """Selected the evidence path for the quasi-Monte Carlo point receipt."""
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
