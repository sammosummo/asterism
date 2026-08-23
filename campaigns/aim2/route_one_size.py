"""Both integrators at one family size, so the sizes can run beside each other.

The sizes are independent comparisons and the accurate route is single
threaded, so running them in sequence wastes every core but one. SIZE and
COUNT come from the environment.

Fewer families at the larger sizes, which has a cost worth naming: the standard
error of an estimand grows as the sample shrinks, so a disagreement expressed as
a share of it is flattered by using less data. The absolute difference is
therefore reported alongside, and it is the one to read across sizes.
"""

from __future__ import annotations

import os
import time

import asterism
from asterism.latent_mediation import simulate

TRUTH: dict[str, float] = {
    "a": 0.6,
    "b": 0.4,
    "c_prime": 0.2,
    "d": 0.7,
    "sigma_m2": 0.5,
}
"""Specified the shared latent-mediation parameters for the route comparison."""
SIZE: int = int(os.environ["SIZE"])
"""Read the simulated family size selected by the batch runner."""
COUNT: int = int(os.environ["COUNT"])
"""Read the number of independent families selected by the batch runner."""

relationship: list[list[float]] = [
    [1.0 if i == j else 0.5 for j in range(SIZE)] for i in range(SIZE)
]
"""Constructed the exchangeable within-family relationship matrix."""
families: list[dict[str, object]] = []
"""Initialised the families shared by the two integration routes."""
for unit in range(COUNT):
    families.extend(
        simulate(
            relationship=relationship,
            **TRUTH,
            families=1,
            seed=90_000 + 977 * unit + 31 * SIZE,
            outcome_prevalence=[0.08] * SIZE,
            observe_outcome=[True] * SIZE,
            measurement_error_variance=0.15,
            ascertainment="population_unconditioned",
            proband_index=None,
        )
    )
"""Simulated the requested number of deterministic independent families."""
print(f"size {SIZE}: {COUNT} families, {COUNT * SIZE} people", flush=True)

got: dict[str, dict[str, float]] = {}
"""Initialised the scientific quantities returned by each integration route."""
seconds: dict[str, float] = {}
"""Initialised the elapsed-time measurement for each integration route."""
for label, points in (("qmc", 2048), ("seq", 0)):
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        families, qmc_points=points
    )
    """Prepared the same simulated families with the selected integration route."""
    started: float = time.time()
    """Captured the route's wall-clock start time."""
    fit: dict[str, object] = model.fit()
    """Fitted the full latent-mediation model through this route."""
    vertical: dict[str, object] = model.test_vertical(bootstrap_replicates=0)
    """Evaluated the vertical-path test without a bootstrap campaign."""
    horizontal: dict[str, object] = model.test_horizontal()
    """Evaluated the horizontal direct-path test."""
    seconds[label] = time.time() - started
    """Recorded the total fit-and-test duration for this route."""
    got[label] = {
        "loglik": fit["loglik"],
        "vertical": fit["estimands"]["theta_vertical"],
        "horizontal": fit["estimands"]["theta_horizontal"],
        "vp": vertical["p_value"],
        "hp": horizontal["p_value"],
    }
    """Stored the like-for-like estimates and tests for route comparison."""
    print(f"  {label} done in {seconds[label]:.0f}s", flush=True)
"""Ran both integration routes against exactly the same simulated families."""

print(
    f"\n{'size':>5} {'quantity':>11} {'qmc 2048':>14} {'sequential':>14} {'diff':>10}",
    flush=True,
)
for quantity in got["qmc"]:
    accurate, cheap = got["qmc"][quantity], got["seq"][quantity]
    """Paired the quasi-Monte Carlo and sequential values for one quantity."""
    print(
        f"{SIZE:>5} {quantity:>11} {accurate:>14.6f} {cheap:>14.6f} "
        f"{abs(accurate - cheap):>10.2e}",
        flush=True,
    )
"""Printed the absolute route difference for every shared scientific quantity."""
print(
    f"size {SIZE} timing: qmc {seconds['qmc']:.0f}s, seq {seconds['seq']:.0f}s",
    flush=True,
)
