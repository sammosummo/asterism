"""Does pleiotropy leak into the mediation estimate, on the real design?

The genetic covariance between hearing and dementia is `a (ab + c')` -- it sees
only the sum, so genetics alone cannot separate mediation from pleiotropy. What
separates them is the residual, within-person covariance `b sigma_m2`. So `b`,
and with it the mediation estimand, is identified from the non-genetic
association between hearing and dementia, and `c'` comes out as the difference
between a well-estimated total and a weakly-estimated product.

That raises a question the power work never asked: if the separation is weak,
does some of the direct path get read as mediation? An estimate that climbs with
pleiotropy would attack the primary claim, which matters far more than the
secondary test being underpowered.

A first attempt on 150 families of three was too noisy to answer -- the means
climbed, but no row reached two standard errors and the estimates were
heavy-tailed enough that a mean was the wrong summary. This runs it on the real
design, reports the median beside the mean, and counts how often the fit lands
somewhere absurd, because a handful of runaway fits is a different problem from
a systematic leak and the two look identical in an average.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import asterism
import numpy as np
from asterism.latent_mediation import simulate

DESIGN = os.environ.get("DESIGN", "design.npz")
OUT = os.environ.get("OUT", "leak_result.json")
REPLICATES = int(os.environ.get("REPLICATES", "200"))
WORKERS = int(os.environ.get("WORKERS", "120"))
QMC_POINTS = int(os.environ.get("QMC_POINTS", "0"))

PROBAND = 0
TRUE_VERTICAL = 0.6 * 0.4
DIRECT = [0.0, 0.2, 0.4, 0.7]
AUDIOMETRY_ERROR = 0.15


def prevalence_for(age: float) -> float:
    if age < 60:
        return 0.001
    if age < 65:
        return 0.005
    if age < 70:
        return 0.02
    if age < 75:
        return 0.04
    if age < 80:
        return 0.08
    if age < 85:
        return 0.15
    return 0.25


_D = {}


def design():
    if not _D:
        loaded = np.load(DESIGN)
        units = collections.defaultdict(list)
        for row, unit in enumerate(loaded["family"]):
            units[int(unit)].append(row)
        _D.update(
            relationship=loaded["relationship"],
            role=loaded["role"],
            units=[units[k] for k in sorted(units)],
            prevalence=np.array([prevalence_for(a) for a in loaded["age"]]),
        )
    return _D


def one(job):
    c_prime, replicate = job
    d = design()
    families = []
    for u, rows in enumerate(d["units"]):
        size = len(rows)
        roles = d["role"][rows]
        matrix = [[float(v) for v in row]
                  for row in d["relationship"][np.ix_(rows, rows)]]
        proband = int(np.argmax(roles == PROBAND)) if (roles == PROBAND).any() else None
        families.extend(simulate(
            relationship=matrix, a=0.6, b=0.4, c_prime=c_prime, d=0.7,
            sigma_m2=0.5, families=1,
            seed=2_000_000 + 7919 * replicate + 13 * u,
            outcome_prevalence=[float(p) for p in d["prevalence"][rows]],
            observe_outcome=[True] * size,
            measurement_error_variance=[AUDIOMETRY_ERROR] * size,
            observe_mediator_proxy=[False] * size,
            ascertainment=("condition_on_named_proband_case" if proband is not None
                           else "population_unconditioned"),
            proband_index=proband))
    out = {"c_prime": c_prime, "replicate": replicate}
    try:
        fit = asterism.LatentMediationModel(families, qmc_points=QMC_POINTS).fit()
        out["vertical"] = float(fit["estimands"]["theta_vertical"])
        out["horizontal"] = float(fit["estimands"]["theta_horizontal"])
    except ValueError as refusal:
        out["refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
    return out


def main() -> int:
    d = design()
    print(f"{sum(len(r) for r in d['units'])} people in {len(d['units'])} units, "
          f"{REPLICATES} replicates per value", flush=True)
    jobs = [(c, r) for c in DIRECT for r in range(REPLICATES)]
    rows, started = [], time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=1):
            rows.append(got)
            if len(rows) % 50 == 0:
                rate = (time.time() - started) / len(rows)
                print(f"  {len(rows)}/{len(jobs)}  {rate:.1f}s/fit", flush=True)

    print(f"\ntrue mediated effect {TRUE_VERTICAL:.2f} throughout\n")
    print(f"{'true c*':>8}{'median':>9}{'mean':>9}{'bias/se':>9}"
          f"{'|est|>1':>9}{'refused':>9}{'median c*':>11}")
    print("-" * 64)
    report = {}
    for c_prime in DIRECT:
        here = [r for r in rows if r["c_prime"] == c_prime]
        values = np.array([r["vertical"] for r in here if "vertical" in r])
        direct = np.array([r["horizontal"] for r in here if "horizontal" in r])
        refused = len(here) - len(values)
        if len(values) < 10:
            print(f"{c_prime:>8.1f}   too few fits")
            continue
        median = float(np.median(values))
        mean = float(values.mean())
        error = float(values.std(ddof=1) / np.sqrt(len(values)))
        runaway = int((np.abs(values) > 1.0).sum())
        print(f"{c_prime:>8.1f}{median:>9.4f}{mean:>9.4f}"
              f"{(mean - TRUE_VERTICAL) / error:>+9.2f}{runaway:>9}"
              f"{refused:>9}{float(np.median(direct)):>11.4f}")
        report[str(c_prime)] = {
            "median_vertical": median, "mean_vertical": mean,
            "standard_error": error, "runaway": runaway, "refused": refused,
            "median_horizontal": float(np.median(direct)),
            "fitted": len(values),
        }

    json.dump({"design": DESIGN, "replicates": REPLICATES,
               "true_vertical": TRUE_VERTICAL, "cells": report},
              open(OUT, "w"), indent=2)
    print(f"\nwritten to {OUT}")
    print("\nA median that climbs with the direct path is a leak. A mean that "
          "climbs while\nthe median does not is a few runaway fits, which is a "
          "different fault and is\nfixed differently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
