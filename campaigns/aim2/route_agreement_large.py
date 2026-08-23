"""Does the cheap integrator still agree on the biggest families in the design?

Agreement was established at size three. The design runs to size eight, and
sizes three and above carry 796 of its 1,600 people -- half the sample. The
sequential route conditions one dimension at a time, so whatever error it makes
has more chances to accumulate as the family grows. Checking the smallest
approximated size and assuming the rest would be assuming the thing most likely
to be false.

Fewer families at the large sizes on purpose: the accurate route is what
becomes expensive there, which is the whole reason the cheap route exists.

What counts as agreement is not a small difference. It is a difference small
against the sampling error of the same quantity, because that is the noise a
reported number already carries.
"""

from __future__ import annotations

import time

import asterism
from asterism.latent_mediation import simulate
from scipy.stats import norm

TRUTH: dict[str, float] = {
    "a": 0.6,
    "b": 0.4,
    "c_prime": 0.2,
    "d": 0.7,
    "sigma_m2": 0.5,
}
"""Specified the shared latent-mediation parameters for route comparisons."""
# Sizes 1 and 2 are exact under both routes, so they are not informative here.
PLAN: list[tuple[int, int]] = [(4, 120), (6, 90), (8, 70)]
"""Selected informative family sizes and affordable family counts."""

print("the biggest families in the design, both integrators", flush=True)
print(
    f"{'size':>5} | {'people':>6} | {'quantity':>10} | {'qmc 2048':>13} | "
    f"{'sequential':>13} | {'diff':>9} | {'qmc s':>7} | {'seq s':>6}",
    flush=True,
)
print("-" * 92, flush=True)

for size, count in PLAN:
    relationship: list[list[float]] = [
        [1.0 if i == j else 0.5 for j in range(size)] for i in range(size)
    ]
    """Constructed the exchangeable relationship matrix for this family size."""
    families: list[dict[str, object]] = []
    """Initialised the families shared by both integration routes."""
    for unit in range(count):
        families.extend(
            simulate(
                relationship=relationship,
                **TRUTH,
                families=1,
                seed=90_000 + 977 * unit + 31 * size,
                outcome_prevalence=[0.08] * size,
                observe_outcome=[True] * size,
                measurement_error_variance=0.15,
                ascertainment="population_unconditioned",
                proband_index=None,
            )
        )
    """Simulated deterministic independent families for this comparison cell."""

    got: dict[str, dict[str, float]] = {}
    """Initialised the scientific quantities returned by each integration route."""
    seconds: dict[str, float] = {}
    """Initialised the elapsed-time measurement for each integration route."""
    for label, points in (("qmc", 2048), ("seq", 0)):
        model: asterism.LatentMediationModel = asterism.LatentMediationModel(
            families, qmc_points=points
        )
        """Prepared the same families with the selected integration route."""
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
        """Stored like-for-like estimates and tests for route comparison."""
    """Ran both integration routes on the same simulated families."""

    for quantity in got["qmc"]:
        accurate, cheap = got["qmc"][quantity], got["seq"][quantity]
        """Paired the quasi-Monte Carlo and sequential values for one quantity."""
        print(
            f"{size:>5} | {count * size:>6} | {quantity:>10} | "
            f"{accurate:>13.6f} | {cheap:>13.6f} | "
            f"{abs(accurate - cheap):>9.2e} | {seconds['qmc']:>7.1f} | "
            f"{seconds['seq']:>6.1f}",
            flush=True,
        )
    """Printed the absolute route difference for every shared quantity."""

    # The comparison that decides it: the disagreement against the sampling
    # error of the same estimand, recovered from its own test.
    for estimand, p_name in (("vertical", "vp"), ("horizontal", "hp")):
        p: float = got["qmc"][p_name]
        """Selected the accurate route's p-value for this estimand."""
        z: float = abs(float(norm.ppf(max(p, 1e-12) / 2)))
        """Recovered the corresponding two-sided normal statistic."""
        if z > 1e-6:
            standard_error: float = abs(got["qmc"][estimand]) / z
            """Recovered the estimand's approximate sampling standard error."""
            share: float = (
                abs(got["qmc"][estimand] - got["seq"][estimand]) / standard_error
            )
            """Expressed route disagreement as a share of sampling error."""
            print(
                f"      -> {estimand} disagreement is {share:.1%} of one "
                f"standard error",
                flush=True,
            )
    """Scaled each estimand disagreement by its own approximate sampling error."""
    print("", flush=True)
"""Compared both routes across every planned informative family size."""

print(
    "A difference matters when it is large against sampling error, not when "
    "it is\nlarge against nought. Half the design's people sit in families of "
    "three or more.",
    flush=True,
)
