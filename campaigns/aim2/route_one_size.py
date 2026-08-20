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

TRUTH = {"a": 0.6, "b": 0.4, "c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}
SIZE = int(os.environ["SIZE"])
COUNT = int(os.environ["COUNT"])

relationship = [[1.0 if i == j else 0.5 for j in range(SIZE)] for i in range(SIZE)]
families = []
for unit in range(COUNT):
    families.extend(simulate(
        relationship=relationship, **TRUTH, families=1,
        seed=90_000 + 977 * unit + 31 * SIZE,
        outcome_prevalence=[0.08] * SIZE, observe_outcome=[True] * SIZE,
        measurement_error_variance=0.15,
        ascertainment="population_unconditioned", proband_index=None,
    ))
print(f"size {SIZE}: {COUNT} families, {COUNT * SIZE} people", flush=True)

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
    print(f"  {label} done in {seconds[label]:.0f}s", flush=True)

print(f"\n{'size':>5} {'quantity':>11} {'qmc 2048':>14} {'sequential':>14} "
      f"{'diff':>10}", flush=True)
for quantity in got["qmc"]:
    accurate, cheap = got["qmc"][quantity], got["seq"][quantity]
    print(f"{SIZE:>5} {quantity:>11} {accurate:>14.6f} {cheap:>14.6f} "
          f"{abs(accurate - cheap):>10.2e}", flush=True)
print(f"size {SIZE} timing: qmc {seconds['qmc']:.0f}s, seq {seconds['seq']:.0f}s",
      flush=True)
