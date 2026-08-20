"""Which design change would actually buy power?

The campaign says the design reaches 55 per cent at its most favourable effect
and needs about 1.7 times the sample for 80. "Get more people" is a true answer
and an expensive one, and it is only the right answer if people are what is
short.

There is a reason to think they are not. The mediated estimand is a product of
two halves that are informed very differently. The loading on hearing is
estimated from resemblance among relatives on a continuous trait measured in
everybody, and ought to be well determined. The path from hearing to dementia
is estimated from the genetic covariance between hearing and a *binary* outcome
at about a sixth prevalence, and in a liability model the information about that
covariance scales with the number of cases and with the relative pairs that
contain one. If the second half is the binding constraint, then adding
unaffected people buys very little and adding cases buys a great deal.

Counting the design settled which levers are worth pulling. Of about 257 cases,
200 are the adjudicated probands; the other 1,400 people contribute roughly 57
between them. So the levers follow the budget rather than the head count --
adjudication is what the money buys, and an adjudicated proband arrives as a
certain case with relatives attached.

An earlier version scaled population prevalence instead, and it is worth saying
why that was wrong: it moved only the 57, raising cases by nine per cent where
its name promised fifty. It would have been reported as "more cases barely
helps" when what it measured was "more of the minority barely helps".

Each lever changes exactly one thing from the real design, and they are
deliberately not combined -- a grid of combinations would be cheaper to justify
and impossible to attribute.

    baseline              the design as it stands
    probands x1.5, x2     more adjudicated families, which is the real spend
    hearing precise       measurement error 0.15 -> 0.05, sharpening the
                          loading; should do almost nothing if the loading is
                          not what is short, which is why it is here
    drop acoustic units   the 185 people who sit in units of Acoustic
                          participants only, removed. Not the whole Acoustic
                          contribution: most of those 600 share a unit with an
                          enrolled relative and cannot be dropped without
                          taking the relative too
    sample x1.5           half of everything again, the naive "recruit more"

The level is measured for every lever, under the identical change, because a
lever that raised power by raising the rejection rate generally would not be a
lever.
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
OUT = os.environ.get("OUT", "levers_result.json")
REPLICATES = int(os.environ.get("REPLICATES", "200"))
WORKERS = int(os.environ.get("WORKERS", "120"))
BOOTSTRAP = int(os.environ.get("BOOTSTRAP", "50"))
QMC_POINTS = int(os.environ.get("QMC_POINTS", "0"))

PROBAND, OFFSPRING, ACOUSTIC, RELATIVE = 0, 1, 2, 3
FIXED = {"c_prime": 0.2, "d": 0.7, "sigma_m2": 0.5}
BASE_ERROR = 0.15
ABR_SENSITIVITY, ABR_SPECIFICITY = 0.80, 0.85
ARM = "audiogram"  # the better arm; the question here is design, not measurement


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
    "null: no path": {"a": 0.6, "b": 0.0},
    "vertical 0.24": {"a": 0.6, "b": 0.4},
}
LEVERS = [
    "baseline",
    # Adjudication is what the budget buys, and it arrives as a certain case
    # with relatives attached. These two are the real "spend more" options.
    "probands x1.5",
    "probands x2",
    # Sharpens the loading on hearing. Should do almost nothing if the loading
    # is not what is short, which is the point of including it.
    "hearing precise",
    # What are the Acoustic-only units contributing? 185 people who carry
    # hearing, almost never become a case, and share a unit with nobody.
    "drop acoustic units",
    # The naive version of "recruit more": half of everything again.
    "sample x1.5",
    # The same people and the same dementia rate, graded rather than diagnosed.
    "staged outcome",
]

# CDR 1, 2 and 3 as shares of the dementia rate, and a questionable band the
# same size as that rate. Chosen so "CDR 1 or worse" is exactly the binary rate
# the other levers use, leaving the grading as the only difference.
SEVERITY = (0.56, 0.31, 0.13)

_D = {}


def design(lever: str):
    """The design with one thing changed."""
    key = lever
    if key in _D:
        return _D[key]

    loaded = np.load(DESIGN)
    units = collections.defaultdict(list)
    for row, unit in enumerate(loaded["family"]):
        units[int(unit)].append(row)
    ordered = [units[k] for k in sorted(units)]
    age = loaded["age"]
    prevalence = np.array([prevalence_for(a) for a in age])
    error = BASE_ERROR

    if lever.startswith("probands x"):
        factor = float(lever.split("x")[1])
        # Duplicate the ascertained units. Each duplicate is an independent
        # family with the same structure, which is what enrolling another
        # proband of the same shape would give, and families are independent in
        # the likelihood so reusing the rows is sound.
        ascertained = [rows for rows in ordered
                       if (loaded["role"][rows] == PROBAND).any()]
        wanted = int(round(len(ascertained) * (factor - 1.0)))
        ordered = ordered + ascertained[:wanted]
    elif lever == "hearing precise":
        error = 0.05
    elif lever == "drop acoustic units":
        # Units made up entirely of Acoustic participants -- no proband, no
        # enrolled relative. 185 people, not the whole Acoustic cohort: the
        # rest share a unit with an enrolled relative.
        ordered = [rows for rows in ordered
                   if not (loaded["role"][rows] == ACOUSTIC).all()]
    elif lever == "staged outcome":
        pass  # handled below, where the shares need the prevalence
    elif lever == "sample x1.5":
        # Every second unit, not the first half. The units are grouped by type,
        # so the first half holds no ascertained families at all -- taking it
        # would have added a thousand people and not one adjudicated case, and
        # then reported that recruiting more does very little.
        ordered = ordered + ordered[::2]

    # Stages per person, built from their own dementia rate so the grading is
    # age-indexed the same way the rate is.
    staged = []
    if lever == "staged outcome":
        for rate in prevalence:
            questionable = float(rate)
            shares = [1.0 - float(rate) - questionable, questionable]
            shares.extend(float(rate) * part for part in SEVERITY)
            staged.append(shares)

    _D[key] = {
        "staged": staged,
        "relationship": loaded["relationship"],
        "age": age,
        "role": loaded["role"],
        "units": ordered,
        "prevalence": prevalence,
        "error": error,
    }
    return _D[key]


def one(job):
    lever, truth, replicate = job
    d = design(lever)
    families = []
    for u, rows in enumerate(d["units"]):
        size = len(rows)
        roles = d["role"][rows]
        matrix = [[float(v) for v in row]
                  for row in d["relationship"][np.ix_(rows, rows)]]
        proband = int(np.argmax(roles == PROBAND)) if (roles == PROBAND).any() else None

        families.extend(simulate(
            relationship=matrix,
            **TRUTHS[truth], **FIXED, families=1,
            seed=800_000 + 7_919 * replicate + 13 * u,
            # None rather than an empty list: the wrapper expands a scalar or
            # None per person, and a zero-length sequence fails its shape check.
            outcome_prevalence=(
                None if d["staged"]
                else [float(p) for p in d["prevalence"][rows]]),
            outcome_category_prevalence=(
                [d["staged"][r] for r in rows] if d["staged"] else None),
            # CDR 1 or worse is a case, which is category two of five and the
            # same roster the binary arm conditions on.
            ascertainment_category=2 if d["staged"] else None,
            observe_outcome=[True] * size,
            measurement_error_variance=[d["error"]] * size,
            observe_mediator_proxy=[False] * size,
            sensitivity=ABR_SENSITIVITY, specificity=ABR_SPECIFICITY,
            ascertainment=("condition_on_named_proband_case" if proband is not None
                           else "population_unconditioned"),
            proband_index=proband,
        ))

    out = {"lever": lever, "truth": truth, "replicate": replicate}
    case_from = 2 if d["staged"] else 1
    out["cases"] = sum(1 for f in families for v in f["outcome_status"]
                       if v is not None and v >= case_from)
    out["people"] = sum(len(f["outcome_status"]) for f in families)
    model = asterism.LatentMediationModel(families, qmc_points=QMC_POINTS)
    try:
        out["vertical"] = float(
            model.test_vertical(bootstrap_replicates=BOOTSTRAP)["p_value"])
    except ValueError as refusal:
        out["vertical_refusal"] = str(refusal).replace("LATENT_MEDIATION_", "")
    return out


def main() -> int:
    cells = [(lever, truth) for lever in LEVERS for truth in TRUTHS]
    chosen = os.environ.get("CELL")
    if chosen is not None:
        cells = [cells[int(chosen)]]
        print(f"cell {chosen}: {cells[0][0]} / {cells[0][1]}", flush=True)
    for lever, _ in cells:
        d = design(lever)
        people = sum(len(rows) for rows in d["units"])
        print(f"{lever}: {len(d['units'])} units, {people} people, "
              f"expected cases {d['prevalence'][np.concatenate(d['units'])].sum():.0f}",
              flush=True)

    jobs = [(lever, truth, r) for lever, truth in cells
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

    report = {}
    print(f"\n{'lever':<18}|{'truth':<15}|{'people':>7}|{'cases':>6}|"
          f"{'refused':>8}|{'reject .025':>12}")
    print("-" * 72)
    for lever in LEVERS:
        for truth in TRUTHS:
            here = [r for r in rows if r["lever"] == lever and r["truth"] == truth]
            if not here:
                continue
            values = [r["vertical"] for r in here if "vertical" in r]
            refused = len(here) - len(values)
            rejected = sum(1 for p in values if p < 0.025)
            cases = int(np.mean([r["cases"] for r in here]))
            people = int(np.mean([r["people"] for r in here]))
            print(f"{lever:<18}|{truth:<15}|{people:>7}|{cases:>6}|"
                  f"{refused:>4}/{len(here):<3}|{rejected:>5}/{len(here):<6}")
            report[f"{lever} | {truth}"] = {
                "replicates": len(here), "people": people, "mean_cases": cases,
                "vertical_refused": refused, "vertical_reject_025": rejected,
            }

    json.dump({"design": DESIGN, "replicates": REPLICATES, "arm": ARM,
               "bootstrap": BOOTSTRAP, "cells": report},
              open(OUT, "w"), indent=2)
    print(f"\nwritten to {OUT}")
    print("\nRead each lever against the baseline, and each power row against "
          "the level row\nabove it. A lever that lifted both has not bought "
          "power, it has bought rejections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
