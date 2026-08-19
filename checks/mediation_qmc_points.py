"""How many quasi-Monte Carlo points does the latent mediation model need?

The model integrates over a latent vector of twice the family size, exactly to
dimension two and by deterministic quasi-Monte Carlo above it. The number of
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

POINTS = [512, 1024, 2048, 4096, 8192, 16384]
REFERENCE = 32768
# A log likelihood this far out moves a deviance by twice as much, which is
# still two orders below the 3.84 a chi-square on one asks for.
TOLERANCE = 0.02

# The shapes the proposed design actually holds: mostly small families, some
# up to six, with everybody's outcome observed -- which is what puts every
# person into the region rather than the density.
FAMILY_SIZES = [2, 3, 4, 5, 6]
FAMILIES_EACH = 40
TRUTH = {"a": 0.6, "b": 0.4, "c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}
# **One parameter point cannot bound the error of a quasi-Monte Carlo rule.**
# The error is not monotone in the number of points -- different counts give
# different lattice alignments -- so a setting that happens to land close at
# one point can be well out at another. The worst case over several points is
# what a setting has to survive, and near a boundary is where it is worst,
# because that is where the region probabilities are smallest.
AT = [
    {"a": 0.55, "b": 0.35, "c_prime": 0.25, "d": 0.65, "sigma_m2": 0.55},
    {"a": 0.20, "b": 0.60, "c_prime": -0.30, "d": 0.40, "sigma_m2": 0.30},
    {"a": 0.80, "b": 0.10, "c_prime": 0.05, "d": 0.90, "sigma_m2": 0.80},
    {"a": 0.05, "b": 0.45, "c_prime": 0.40, "d": 0.15, "sigma_m2": 1.20},
    {"a": 0.60, "b": 0.40, "c_prime": 0.20, "d": 0.70, "sigma_m2": 0.50},
]


def relationship(size: int) -> list[list[float]]:
    a = np.eye(size)
    for i in range(size):
        for j in range(size):
            if i != j:
                a[i, j] = 0.5
    return [[float(v) for v in row] for row in a]


def build() -> list[dict]:
    families = []
    for size in FAMILY_SIZES:
        for f in range(FAMILIES_EACH):
            families.extend(simulate(
                relationship=relationship(size),
                **TRUTH,
                families=1,
                seed=770_000 + 131 * size + f,
                outcome_prevalence=[0.08] * size,
                observe_outcome=[True] * size,
                measurement_error_variance=0.15,
                ascertainment="population_unconditioned",
                proband_index=None,
            ))
    return families


def main() -> int:
    families = build()
    people = sum(len(f["outcome_status"]) for f in families)
    print(f"{len(families)} families, {people} people, sizes {FAMILY_SIZES}")
    print(f"reference: {REFERENCE} points; tolerance {TOLERANCE} in log "
          f"likelihood\n", flush=True)

    reference_model = asterism.LatentMediationModel(families, qmc_points=REFERENCE)
    started = time.time()
    reference_loglik = [float(reference_model.evaluate(**at)["log_likelihood"])
                        for at in AT]
    reference_seconds = (time.time() - started) / len(AT)
    print(f"reference log likelihoods at {len(AT)} points, "
          f"{reference_seconds:.2f}s each\n")

    print(f"{'points':>7} | {'worst difference':>16} | {'median':>10} | "
          f"{'seconds':>8} | {'speedup':>8} | verdict")
    print("-" * 74)
    rows, failures = [], []
    for points in POINTS:
        model = asterism.LatentMediationModel(families, qmc_points=points)
        started = time.time()
        differences = [abs(float(model.evaluate(**at)["log_likelihood"]) - ref)
                       for at, ref in zip(AT, reference_loglik)]
        took = (time.time() - started) / len(AT)
        worst = max(differences)
        ok = worst <= TOLERANCE
        print(f"{points:>7} | {worst:>16.2e} | {np.median(differences):>10.2e} | "
              f"{took:>8.2f} | {reference_seconds / took:>7.1f}x | "
              f"{'ok' if ok else 'TOO COARSE'}", flush=True)
        rows.append({"points": points, "worst_difference": worst,
                     "median_difference": float(np.median(differences)),
                     "all_differences": differences,
                     "seconds": took, "within_tolerance": ok})

    usable = [r for r in rows if r["within_tolerance"]]
    cheapest = min(usable, key=lambda r: r["points"]) if usable else None
    if cheapest is None:
        failures.append("no setting tried came within tolerance of the reference")
    else:
        print(f"\nCheapest setting within tolerance: {cheapest['points']} points, "
              f"{reference_seconds / cheapest['seconds']:.1f} times faster than "
              f"the reference")
        if cheapest["points"] < 8192:
            print(f"  The package default is 8192. {cheapest['points']} is "
                  f"{8192 / cheapest['points']:.0f} times cheaper and agrees to "
                  f"{cheapest['worst_difference']:.1e} at worst.")

    receipt = {
        "what": "how many quasi-Monte Carlo points the latent mediation model needs",
        "date": date.today().isoformat(),
        "families": len(families), "people": people,
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
    out = Path(__file__).resolve().parent.parent / "evidence" / (
        f"mediation-qmc-points-{receipt['date']}.json")
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
