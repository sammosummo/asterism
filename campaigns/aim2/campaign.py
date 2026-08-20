"""Level and power for Aim 2, on the design Sam settled on 18 August.

Reads `design.npz`, which carries a relationship matrix, an age and a role per
person, and a family unit per person -- and no identifiers.

- **Age sets risk.** Dementia prevalence rises steeply with age, so a person's
  rate comes from their age band rather than from one number for everybody.
  Every status is observed: a fifty-year-old with a clean CDR is a known
  non-case, and says so.
- **Family history is not a covariate.** The elevated risk of somebody with an
  affected parent is already in the relationship matrix. That is why enrolling
  the relatives matters and why no separate term is fitted for it.
- **Two measurement arms.** Probands either give a binary, misclassified ABR --
  which is what the protocol says for people too impaired for behavioural
  audiometry -- or an audiogram. Everyone else is measured properly in both.
  The pair brackets the severity question without needing to know the fraction.
- **Ascertainment is per family.** A Biggs family is drawn conditional on its
  proband being a case, which is what a clinic roster is. Everybody else is
  drawn from the population.

Level is measured under two nulls at the same replicate count as the
alternatives, and no power figure is read until it holds. Refusals are counted
and never dropped from a denominator.
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
OUT = os.environ.get("OUT", "campaign_result.json")
REPLICATES = int(os.environ.get("REPLICATES", "200"))
WORKERS = int(os.environ.get("WORKERS", "120"))
BOOTSTRAP = int(os.environ.get("BOOTSTRAP", "50"))
# Nought selects sequential truncation, which is what SOLAR and the liability
# model use. It is an approximation -- agreeing with the accurate route to
# about 0.03 in log probability per family at pedigree correlations -- and it
# is what lets this run on the real families rather than on pairs.
QMC_POINTS = int(os.environ.get("QMC_POINTS", "0"))

PROBAND, OFFSPRING, ACOUSTIC, RELATIVE = 0, 1, 2, 3
FIXED = {"c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}
AUDIOMETRY_ERROR = 0.15
# Placeholders until the ABR-audiometry bridge supplies real ones.
ABR_SENSITIVITY, ABR_SPECIFICITY = 0.80, 0.85

# Probable AD dementia by age. Roughly a doubling every five years past 65,
# which is the shape every prevalence study finds.
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


TRUTHS = {
    "null: no loading": {"a": 0.0, "b": 0.4},
    "null: no path":    {"a": 0.6, "b": 0.0},
    "vertical 0.06":    {"a": 0.6, "b": 0.1},
    "vertical 0.12":    {"a": 0.6, "b": 0.2},
    "vertical 0.18":    {"a": 0.6, "b": 0.3},
    "vertical 0.24":    {"a": 0.6, "b": 0.4},
}
ARMS = ["ABR only", "audiogram"]

_D = {}


def design():
    if not _D:
        loaded = np.load(DESIGN)
        units = collections.defaultdict(list)
        for row, unit in enumerate(loaded["family"]):
            units[int(unit)].append(row)
        _D.update(
            relationship=loaded["relationship"],
            age=loaded["age"],
            role=loaded["role"],
            units=[units[k] for k in sorted(units)],
            prevalence=np.array([prevalence_for(a) for a in loaded["age"]]),
        )
    return _D


def one(job):
    arm, truth, replicate = job
    d = design()
    families = []
    for u, rows in enumerate(d["units"]):
        size = len(rows)
        roles = d["role"][rows]
        matrix = [[float(v) for v in row]
                  for row in d["relationship"][np.ix_(rows, rows)]]
        proband = int(np.argmax(roles == PROBAND)) if (roles == PROBAND).any() else None

        if proband is not None and arm == "ABR only":
            # No audiogram for the person with dementia; a fallible binary
            # result instead. Everyone else is measured properly.
            error = [None if i == proband else AUDIOMETRY_ERROR for i in range(size)]
            proxy = [i == proband for i in range(size)]
        else:
            error = [AUDIOMETRY_ERROR] * size
            proxy = [False] * size

        families.extend(simulate(
            relationship=matrix,
            **TRUTHS[truth], **FIXED, families=1,
            seed=300_000 + 7_919 * replicate + 13 * u,
            outcome_prevalence=[float(p) for p in d["prevalence"][rows]],
            observe_outcome=[True] * size,
            measurement_error_variance=error,
            observe_mediator_proxy=proxy,
            sensitivity=ABR_SENSITIVITY, specificity=ABR_SPECIFICITY,
            ascertainment=("condition_on_named_proband_case" if proband is not None
                           else "population_unconditioned"),
            proband_index=proband,
        ))

    out = {"arm": arm, "truth": truth, "replicate": replicate}
    out["cases"] = sum(1 for f in families for v in f["outcome_status"] if v)
    model = asterism.LatentMediationModel(families, qmc_points=QMC_POINTS)
    started = time.time()
    try:
        out["vertical"] = float(
            model.test_vertical(bootstrap_replicates=BOOTSTRAP)["p_value"])
    except ValueError as refusal:
        out["vertical_refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
    try:
        out["horizontal"] = float(model.test_horizontal()["p_value"])
    except ValueError as refusal:
        out["horizontal_refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
    out["seconds"] = time.time() - started
    return out


def main() -> int:
    d = design()
    n = len(d["age"])
    print(f"{n} people in {len(d['units'])} units, "
          f"{int((d['age'] >= 65).sum())} of them over 65")
    print(f"{REPLICATES} replicates, {len(ARMS)} arms, {len(TRUTHS)} truths, "
          f"{WORKERS} workers, {BOOTSTRAP} reference draws", flush=True)
    # One array task per cell, so the grid spreads across nodes rather than
    # queueing behind itself on one. CELL is the task id; unset means all.
    cells = [(arm, truth) for arm in ARMS for truth in TRUTHS]
    chosen = os.environ.get("CELL")
    if chosen is not None:
        cells = [cells[int(chosen)]]
        print(f"cell {chosen}: {cells[0][0]} / {cells[0][1]}", flush=True)
    jobs = [(arm, truth, r) for arm, truth in cells
            for r in range(REPLICATES)]
    print(f"{len(jobs)} fits queued\n", flush=True)

    rows, started = [], time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=1):
            rows.append(got)
            if len(rows) % 50 == 0:
                rate = (time.time() - started) / len(rows)
                print(f"  {len(rows)}/{len(jobs)}  {rate:.1f}s/fit  "
                      f"eta {(len(jobs) - len(rows)) * rate / 60:.0f} min",
                      flush=True)

    print(f"\n{'arm':<11}|{'truth':<17}|{'cases':>6}|{'refused V':>10}|"
          f"{'rej V .025':>11}|{'refused H':>10}|{'rej H .025':>11}")
    print("-" * 82)
    report = {}
    for arm in ARMS:
        for truth in TRUTHS:
            here = [r for r in rows if r["arm"] == arm and r["truth"] == truth]
            v = [r["vertical"] for r in here if "vertical" in r]
            h = [r["horizontal"] for r in here if "horizontal" in r]
            vr, hr = len(here) - len(v), len(here) - len(h)
            v025 = sum(1 for p in v if p < 0.025)
            h025 = sum(1 for p in h if p < 0.025)
            cases = int(np.mean([r["cases"] for r in here])) if here else 0
            print(f"{arm:<11}|{truth:<17}|{cases:>6}|{vr:>4}/{len(here):<5}|"
                  f"{v025:>4}/{len(here):<6}|{hr:>4}/{len(here):<5}|"
                  f"{h025:>4}/{len(here):<6}")
            report[f"{arm} | {truth}"] = {
                "replicates": len(here), "mean_cases": cases,
                "vertical_refused": vr, "vertical_reject_025": v025,
                "horizontal_refused": hr, "horizontal_reject_025": h025,
            }
    json.dump({"design": DESIGN, "replicates": REPLICATES,
               "bootstrap": BOOTSTRAP, "cells": report},
              open(OUT, "w"), indent=2)
    print(f"\nwritten to {OUT}")
    print("\nRejection under the two nulls is the LEVEL and must sit at or "
          "below .025.\nRejection under a vertical truth is the POWER.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
