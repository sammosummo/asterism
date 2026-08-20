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
import numpy as np
from asterism.latent_mediation import simulate

TRUTH = {"a": 0.6, "b": 0.4, "c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}
# Sizes 1 and 2 are exact under both routes, so they are not informative here.
PLAN = [(4, 120), (6, 90), (8, 70)]

print("the biggest families in the design, both integrators", flush=True)
print(f"{'size':>5} | {'people':>6} | {'quantity':>10} | {'qmc 2048':>13} | "
      f"{'sequential':>13} | {'diff':>9} | {'qmc s':>7} | {'seq s':>6}",
      flush=True)
print("-" * 92, flush=True)

for size, count in PLAN:
    relationship = [[1.0 if i == j else 0.5 for j in range(size)]
                    for i in range(size)]
    families = []
    for unit in range(count):
        families.extend(simulate(
            relationship=relationship, **TRUTH, families=1,
            seed=90_000 + 977 * unit + 31 * size,
            outcome_prevalence=[0.08] * size, observe_outcome=[True] * size,
            measurement_error_variance=0.15,
            ascertainment="population_unconditioned", proband_index=None,
        ))

    got, seconds = {}, {}
    for label, points in (("qmc", 2048), ("seq", 0)):
        model = asterism.LatentMediationModel(families, qmc_points=points)
        started = time.time()
        fit = model.fit()
        vertical = model.test_vertical(bootstrap_replicates=0)
        horizontal = model.test_horizontal()
        seconds[label] = time.time() - started
        got[label] = {
            "loglik": fit["loglik"],
            "vertical": fit["estimands"]["theta_vertical"],
            "horizontal": fit["estimands"]["theta_horizontal"],
            "vp": vertical["p_value"],
            "hp": horizontal["p_value"],
        }

    for quantity in got["qmc"]:
        accurate, cheap = got["qmc"][quantity], got["seq"][quantity]
        print(f"{size:>5} | {count * size:>6} | {quantity:>10} | "
              f"{accurate:>13.6f} | {cheap:>13.6f} | "
              f"{abs(accurate - cheap):>9.2e} | {seconds['qmc']:>7.1f} | "
              f"{seconds['seq']:>6.1f}", flush=True)

    # The comparison that decides it: the disagreement against the sampling
    # error of the same estimand, recovered from its own test.
    for estimand, p_name in (("vertical", "vp"), ("horizontal", "hp")):
        from scipy.stats import norm
        p = got["qmc"][p_name]
        z = abs(norm.ppf(max(p, 1e-12) / 2))
        if z > 1e-6:
            standard_error = abs(got["qmc"][estimand]) / z
            share = abs(got["qmc"][estimand] - got["seq"][estimand]) / standard_error
            print(f"      -> {estimand} disagreement is {share:.1%} of one "
                  f"standard error", flush=True)
    print("", flush=True)

print("A difference matters when it is large against sampling error, not when "
      "it is\nlarge against nought. Half the design's people sit in families of "
      "three or more.", flush=True)
