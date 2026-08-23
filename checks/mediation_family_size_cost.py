"""What does a third observed outcome in a family cost?

The latent mediation model computes region probabilities exactly through
dimension two and by quasi-Monte Carlo above it. Every existing check on the
model observes one outcome per family, so that upper route has never been
exercised at the scale an analysis would use it, and its cost has never been
written down.

It is large. A design that observes everybody's binary outcome puts the whole
family into the region, so the dimension is the family size, and the step from
two to three is a step from a closed form to at least five hundred integrand
evaluations.

**This is not a correctness problem and the check says so.** Separately measured,
the higher-dimensional route converges to eight digits at 512 points and
factorises correctly when people are near-independent. What it lacks is
anything between an exact bivariate and a fixed quasi-Monte Carlo rule, and the
gap between those two is what this measures. Genz's algorithm is the standard
fill for dimensions three to about ten with adaptive error control; adopting it
would remove the step rather than shave it.

The number matters because it decides study designs. A family-based analysis
that wants three people per family with known status pays this, and finding
that out after submitting a campaign is expensive.

Run with:

    uv run --no-project python checks/mediation_family_size_cost.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import asterism
from asterism.latent_mediation import simulate

FAMILIES: int = 150
"""Set the number of simulated families evaluated at each family size."""

SIZES: list[int] = [2, 3, 4]
"""Selected family sizes spanning the exact and quasi-Monte Carlo routes."""

POINTS: list[int] = [512, 2048]
"""Selected two integration budgets to expose quasi-Monte Carlo scaling."""

TRUTH: dict[str, float] = {
    "a": 0.6,
    "b": 0.4,
    "c_prime": 0.2,
    "d": 0.7,
    "sigma_m2": 0.5,
}
"""Defined the mediation parameters used to generate each family."""

AT: dict[str, float] = {
    "a": 0.55,
    "b": 0.35,
    "c_prime": 0.25,
    "d": 0.65,
    "sigma_m2": 0.55,
}
"""Defined the fixed parameter point used for every timed likelihood evaluation."""
# Beyond this the step has stopped being a detail and become a design
# constraint, which is the thing worth flagging.
A_STEP_WORTH_FLAGGING: float = 10.0
"""Set the per-person slowdown that makes family size a design constraint."""


def build(size: int) -> list[dict[str, object]]:
    """Simulate the standing mediation design at one family size.

    Args:
        size: Number of observed relatives in every family.

    Returns:
        Simulated family records accepted by the latent mediation model.
    """
    relationship: list[list[float]] = [
        [1.0 if i == j else 0.5 for j in range(size)] for i in range(size)
    ]
    """Constructed the full-sibling relationship matrix at the requested size."""

    families: list[dict[str, object]] = []
    """Initialised the family records for the timed design."""

    for f in range(FAMILIES):
        families.extend(
            simulate(
                relationship=relationship,
                **TRUTH,
                families=1,
                seed=6_000 + 13 * f,
                outcome_prevalence=[0.08] * size,
                observe_outcome=[True] * size,
                measurement_error_variance=0.15,
                ascertainment="population_unconditioned",
                proband_index=None,
            )
        )
        """Added one deterministically seeded family at the current size."""
    return families


def main() -> int:
    print(f"{FAMILIES} families at each size, one likelihood evaluation each\n")
    print(
        f"{'size':>5} | {'people':>7} | "
        + " | ".join(f"{p:>6} pts" for p in POINTS)
        + " | per person"
    )
    print("-" * 62)

    rows: list[dict[str, object]] = []
    """Initialised the complete timing records returned as evidence."""

    per_person: dict[int, float] = {}
    """Initialised per-person costs indexed by observed family size."""

    for size in SIZES:
        families: list[dict[str, object]] = build(size)
        """Built the complete simulated design for this family size."""

        people: int = FAMILIES * size
        """Counted the people contributing to this timed likelihood."""

        model_times: list[float] = []
        """Initialised elapsed likelihood times for the selected point budgets."""

        for points in POINTS:
            model: asterism.LatentMediationModel = asterism.LatentMediationModel(
                families, qmc_points=points
            )
            """Prepared the mediation model at the current integration budget."""

            started: float = time.time()
            """Captured the wall-clock start of one likelihood evaluation."""

            model.evaluate(**AT)
            model_times.append(time.time() - started)
            """Recorded the elapsed evaluation time at the current point budget."""

        per_person[size] = model_times[-1] / people
        """Normalised the richer integration time by the number of people."""

        print(
            f"{size:>5} | {people:>7} | "
            + " | ".join(f"{t:9.4f}s" for t in model_times)
            + f" | {per_person[size] * 1e6:8.1f} us",
            flush=True,
        )
        rows.append(
            {
                "size": size,
                "people": people,
                "seconds": dict(zip(POINTS, model_times, strict=True)),
                "seconds_per_person": per_person[size],
            }
        )
        """Recorded the complete timing evidence for the current family size."""

    step: float = per_person[3] / per_person[2] if per_person.get(2) else float("nan")
    """Calculated the per-person cost step from exact bivariate to QMC trivariate."""

    print(
        f"\nGoing from two observed outcomes to three costs {step:.0f} times "
        f"more per person."
    )
    print(
        "Cost is linear in the number of points, so fewer points shave it and "
        "cannot\nclose it: the exact route is a closed form and the "
        "alternative is at least\nfive hundred evaluations."
    )

    receipt: dict[str, object] = {
        "what": "the cost of a third observed outcome in a family",
        "date": date.today().isoformat(),
        "families": FAMILIES,
        "sizes": SIZES,
        "points": POINTS,
        "rows": rows,
        "step_from_two_to_three": step,
        "note": "not a correctness problem; the higher-dimensional route "
        "converges and factorises. The gap is that nothing sits "
        "between an exact bivariate and a fixed quasi-Monte Carlo rule.",
        "worth_flagging_above": A_STEP_WORTH_FLAGGING,
        "a_design_constraint": step > A_STEP_WORTH_FLAGGING,
    }
    """Assembled the values-free timing receipt and design-constraint verdict."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / (f"mediation-family-size-cost-{receipt['date']}.json")
    )
    """Selected the dated evidence path for the family-size cost receipt."""
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if step > A_STEP_WORTH_FLAGGING:
        print(
            f"\nA step of {step:.0f} is a study-design constraint rather than "
            f"a performance detail.\nA design wanting three people per family "
            f"with known status has to know this."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
