"""Does the censored model's interval contain the truth 95 times in 100?

An interval that has never been counted is a claim rather than a result. This
simulates data sets whose heritability is known, fits the profile interval to
each, and counts how often it contains the truth -- across censoring rates,
because the whole question is how far the model can be pushed before the upper
tail stops supporting a variance.

Three rules make the number mean something, and they are the coverage check's
rules elsewhere in this package for the same reasons:

1. **Every replicate is scored.** An interval that could not be computed never
   covers. Dropping the awkward ones is exactly how a coverage check comes out
   at 95 per cent while the recipe is wrong, and the refusals are reported
   beside the coverage rather than under it.
2. **An end on a bound still counts as containment** when the truth is inside,
   because an end on nought or one is the data failing to rule that end out,
   not the interval stopping short.
3. **A cell passes when the Clopper-Pearson interval on its coverage overlaps
   the nominal 0.95**, which is a statement about this many replicates rather
   than about the number looking close by eye. **At a heritability of nought
   the requirement is one-sided instead**: nought is on a bound, a profile read
   against a plain chi-square is conservative there, and over-covering is the
   safe direction. Such a cell has to not fall below nominal; being above it is
   the recipe being careful rather than wrong.

Run with:

    ASTERISM_REPLICATES=300 uv run --no-project python checks/tobit_coverage.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import asterism
import numpy as np
from scipy.stats import beta

PAIRS = 300
TRUE_VARIANCE = 4.0
TRUE_MEAN = 10.0
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "300"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "12"))
NOMINAL = 0.95
# The cells: a heritability worth finding, one at the boundary, and censoring
# from none to the half where the real audiometry stops working.
HERITABILITIES = [0.0, 0.3, 0.5]
RATES = [0.0, 0.25, 0.50]


def draw(heritability: float, rate: float, replicate: int):
    # **The heritability belongs in the seed.** Without it, cells at the
    # same censoring rate are fitted to the same random draws, so two of
    # them agreeing is one observation rather than two -- and a first run
    # of this check read exactly that as corroboration.
    rng = np.random.default_rng(
        910_000 + 7919 * replicate + int(1000 * rate) + int(1_000_000 * heritability)
    )
    n = 2 * PAIRS
    shared = rng.normal(size=PAIRS) * np.sqrt(heritability / 2.0)
    own = rng.normal(size=n) * np.sqrt(1.0 - heritability / 2.0)
    complete = TRUE_MEAN + np.sqrt(TRUE_VARIANCE) * (np.repeat(shared, 2) + own)
    relationship = np.eye(n)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5
    if rate <= 0.0:
        return relationship, complete, np.zeros(n, dtype=bool), 0.0
    limit = float(np.quantile(complete, 1.0 - rate))
    return relationship, complete, complete >= limit, limit


def one(job):
    heritability, rate, replicate = job
    relationship, complete, censored, limit = draw(heritability, rate, replicate)
    n = complete.size
    try:
        got = asterism.tobit_interval(
            relationship,
            np.where(censored, np.nan, complete),
            np.where(censored, 1, 0).astype(np.int64),
            np.full(n, limit),
            np.ones((n, 1)),
        )
    except ValueError as refusal:
        return {"heritability": heritability, "rate": rate,
                "refusal": str(refusal).replace("TOBIT_", "")}
    return {
        "heritability": heritability, "rate": rate,
        "covered": bool(got["lower"] <= heritability <= got["upper"]),
        "width": got["upper"] - got["lower"],
        "lower_at_bound": got["lower_at_bound"],
        "upper_at_bound": got["upper_at_bound"],
        "profile_failures": got["profile_failures"],
    }


def clopper_pearson(hits: int, n: int) -> tuple[float, float]:
    low = beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0
    high = beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    return float(low), float(high)


def main() -> int:
    print(f"{PAIRS} sibling pairs, {REPLICATES} replicates per cell, "
          f"nominal {NOMINAL}")
    print(f"true variance {TRUE_VARIANCE}, heritabilities {HERITABILITIES}, "
          f"censoring {[f'{r:.0%}' for r in RATES]}\n", flush=True)

    jobs = [(h, r, k) for h in HERITABILITIES for r in RATES
            for k in range(REPLICATES)]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=2):
            rows.append(got)
            if len(rows) % 100 == 0:
                print(f"  {len(rows)}/{len(jobs)}", flush=True)

    print(f"\n{'h2':>5} | {'censored':>8} | {'refused':>9} | "
          f"{'coverage':>19} | {'width':>7} | verdict")
    print("-" * 74)
    report, failures = {}, []
    for heritability in HERITABILITIES:
        for rate in RATES:
            here = [r for r in rows
                    if r["heritability"] == heritability and r["rate"] == rate]
            refused = [r for r in here if "refusal" in r]
            # Every replicate is scored: one that could not be computed never
            # covers, so it stays in the denominator.
            hits = sum(1 for r in here if r.get("covered"))
            low, high = clopper_pearson(hits, len(here))
            width = np.mean([r["width"] for r in here if "width" in r]) \
                if len(here) > len(refused) else float("nan")
            # At the boundary the requirement is one-sided: conservative is
            # fine, under-covering is not.
            at_boundary = heritability <= 0.0
            ok = high >= NOMINAL if at_boundary else low <= NOMINAL <= high
            print(f"{heritability:>5.2f} | {rate:>7.0%} | "
                  f"{len(refused):>3}/{len(here):<5} | "
                  f"{hits / len(here):>6.3f} [{low:.3f},{high:.3f}] | "
                  f"{width:>7.3f} | {'ok' if ok else 'OFF NOMINAL'}", flush=True)
            report[f"h2={heritability} censored={rate}"] = {
                "replicates": len(here), "refused": len(refused),
                "covered": hits, "coverage": hits / len(here),
                "clopper_pearson": [low, high], "mean_width": float(width),
                "within_nominal": ok, "at_boundary": at_boundary,
            }
            if not ok:
                failures.append(
                    f"at h2 {heritability} and {rate:.0%} censored, coverage "
                    f"{hits / len(here):.3f} [{low:.3f}, {high:.3f}] falls "
                    f"below the nominal {NOMINAL}"
                    if at_boundary else
                    f"at h2 {heritability} and {rate:.0%} censored, coverage "
                    f"{hits / len(here):.3f} [{low:.3f}, {high:.3f}] excludes "
                    f"the nominal {NOMINAL}"
                )

    receipt = {
        "what": "coverage of the censored model's profile interval",
        "date": date.today().isoformat(),
        "pairs": PAIRS, "replicates": REPLICATES, "nominal": NOMINAL,
        "true_variance": TRUE_VARIANCE,
        "cells": report, "passed": not failures, "failures": failures,
    }
    out = Path(__file__).resolve().parent.parent / "evidence" / (
        f"tobit-coverage-{receipt['date']}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: every cell's coverage interval covers the nominal level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
