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
2. **Whether a boundary point belongs to the interval is decided by the
   Self-Liang mixture**, which is ADR 0004's recipe, and not by the end having
   landed on the bound. The record carries the verdict as
   ``contains_lower_bound`` and ``contains_upper_bound``, and this reads it.
   Correcting the earlier version of this check, which counted an end on nought
   as containment: that is the obvious rule and it is the wrong one. ADR 0004
   measured what it costs -- 0.977 against a nominal 0.95 at a true
   heritability of nought, where the mixture gives 0.953 -- and this check
   reported 0.980, 0.977 and 0.983 in exactly those cells while passing them.
3. **A cell passes when the Clopper-Pearson interval on its coverage overlaps
   the nominal 0.95**, which is a statement about this many replicates rather
   than about the number looking close by eye. **Every cell is two-sided,
   including the one at nought.** The earlier version allowed that cell to
   over-cover, which is what let the missing mixture go unnoticed: a rule
   written to tolerate a fault will not report it.

Run with:

    ASTERISM_REPLICATES=300 uv run --no-project python checks/tobit_coverage.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import asterism
import numpy as np
from scipy.stats import beta

PAIRS: int = 300
"""Number of independent sibling pairs in every coverage replicate."""

TRUE_VARIANCE: float = 4.0
"""Generating complete-trait variance."""

TRUE_MEAN: float = 10.0
"""Generating complete-trait mean before censoring."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "300"))
"""Default replicates retained for non-release exploratory runs."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "12"))
"""Default process count retained for non-release exploratory runs."""

NOMINAL: float = 0.95
"""Profile-interval coverage level used by the fixed acceptance rule."""

# The cells: a heritability worth finding, one at the boundary, and censoring
# from none through the two observed high-frequency audiogram censoring levels.
HERITABILITIES: list[float] = [0.0, 0.3, 0.5]
"""Boundary and interior generating heritabilities scored in every rate cell."""

RATES: list[float] = [0.0, 0.25, 0.50, 0.52, 0.75]
"""Exploratory and exact high-frequency censoring shares available to argv."""


def draw(
    heritability: float, rate: float, replicate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draw one fixed sibling-pair response and censoring cell."""
    # **The heritability belongs in the seed.** Without it, cells at the
    # same censoring rate are fitted to the same random draws, so two of
    # them agreeing is one observation rather than two -- and a first run
    # of this check read exactly that as corroboration.
    rng: np.random.Generator = np.random.default_rng(
        910_000 + 7919 * replicate + int(1000 * rate) + int(1_000_000 * heritability)
    )
    """Created a disjoint deterministic random stream for the scientific cell."""

    people: int = 2 * PAIRS
    """Counted rows in the fixed sibling-pair roster."""

    shared: np.ndarray = rng.normal(size=PAIRS) * np.sqrt(heritability / 2.0)
    """Drew shared pair-level genetic effects."""

    own: np.ndarray = rng.normal(size=people) * np.sqrt(1.0 - heritability / 2.0)
    """Drew independent remaining genetic and residual variation."""

    complete: np.ndarray = TRUE_MEAN + np.sqrt(TRUE_VARIANCE) * (
        np.repeat(shared, 2) + own
    )
    """Constructed complete observations before applying the instrument limit."""

    relationship: np.ndarray = np.eye(people)
    """Started from independent unit marginal additive covariance."""

    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Recorded the first directed sibling relationship entry."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Completed the symmetric sibling relationship entry."""

    if rate <= 0.0:
        return relationship, complete, np.zeros(people, dtype=bool), 0.0
    limit: float = float(np.quantile(complete, 1.0 - rate))
    """Selected the realized quantile yielding the configured censoring share."""

    return relationship, complete, complete >= limit, limit


def one(job: tuple[float, float, int]) -> dict[str, Any]:
    """Fit and score one interval without dropping refusals."""
    heritability, rate, replicate = job
    """Named the fixed scientific cell and deterministic replicate."""

    relationship, complete, censored, limit = draw(heritability, rate, replicate)
    """Generated one complete response and its observed censoring mask."""

    people: int = complete.size
    """Counted observations supplied to the public interval function."""

    try:
        got: dict[str, Any] = asterism.tobit_interval(
            relationship,
            np.where(censored, np.nan, complete),
            np.where(censored, 1, 0).astype(np.int64),
            np.full(people, limit),
            np.ones((people, 1)),
        )
        """Profiled the reportable interval through the documented public API."""
    except ValueError as refusal:
        return {
            "heritability": heritability,
            "rate": rate,
            "refusal": str(refusal).replace("TOBIT_", ""),
        }
    return {
        "heritability": heritability,
        "rate": rate,
        "covered": covers(got, heritability),
        "width": got["upper"] - got["lower"],
        "lower_at_bound": got["lower_at_bound"],
        "upper_at_bound": got["upper_at_bound"],
        "contains_lower_bound": got["contains_lower_bound"],
        "contains_upper_bound": got["contains_upper_bound"],
        "profile_failures": got["profile_failures"],
    }


def covers(got: dict[str, Any], truth: float) -> bool:
    """Does the interval contain the truth, by ADR 0004's rule?

    Inside the two ends is containment and outside them is not, as anywhere
    else. What differs is a truth that sits exactly on a bound: there the
    interval reaching the bound is not the question, and the mixture verdict
    is. An absent verdict counts as not covered, because absent means the fit
    at the bound could not be made -- which is unknown ground rather than
    ground the data ruled out, and every replicate is scored.
    """
    if truth < got["lower"] or truth > got["upper"]:
        return False
    if truth == 0.0 and got["lower"] == 0.0:
        return bool(got["contains_lower_bound"])
    if truth == 1.0 and got["upper"] == 1.0:
        return bool(got["contains_upper_bound"])
    return True


def clopper_pearson(hits: int, n: int) -> tuple[float, float]:
    """Return the exact two-sided 95% binomial interval."""
    low: float = float(beta.ppf(0.025, hits, n - hits + 1)) if hits else 0.0
    """Computed the exact lower Monte Carlo confidence limit."""

    high: float = float(beta.ppf(0.975, hits + 1, n - hits)) if hits < n else 1.0
    """Computed the exact upper Monte Carlo confidence limit."""

    return float(low), float(high)


def parse_arguments() -> argparse.Namespace:
    """Read exact coverage cells without changing their acceptance rule."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built the explicit cell, replicate, worker, and output command contract."""

    parser.add_argument(
        "--rate",
        action="append",
        type=float,
        choices=RATES,
        dest="rates",
        help="run one configured censoring share; repeat for multiple cells",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print evidence only, leaving the fixed release checkout unchanged",
    )
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments: argparse.Namespace = parser.parse_args()
    """Read exact release counts before any process workers start."""

    if arguments.replicates < 1 or arguments.workers < 1:
        parser.error("--replicates and --workers must be positive")
    return arguments


def main() -> int:
    arguments: argparse.Namespace = parse_arguments()
    """Selected either the complete configured grid or exact requested cells."""

    rates: list[float] = RATES if arguments.rates is None else arguments.rates
    """Retained request order so the evidence command determines its exact cells."""

    print(
        f"{PAIRS} sibling pairs, {arguments.replicates} replicates per cell, "
        f"nominal {NOMINAL}"
    )
    print(
        f"true variance {TRUE_VARIANCE}, heritabilities {HERITABILITIES}, "
        f"censoring {[f'{r:.0%}' for r in rates]}\n",
        flush=True,
    )

    jobs: list[tuple[float, float, int]] = [
        (heritability, rate, replicate)
        for heritability in HERITABILITIES
        for rate in rates
        for replicate in range(arguments.replicates)
    ]
    """Enumerated every exact cell before starting parallel execution."""

    rows: list[dict[str, Any]] = []
    """Reserved every replicate result, including explicit refusals."""

    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for got in pool.map(one, jobs, chunksize=2):
            rows.append(got)
            if len(rows) % 100 == 0:
                print(f"  {len(rows)}/{len(jobs)}", flush=True)

    print(
        f"\n{'h2':>5} | {'censored':>8} | {'refused':>9} | "
        f"{'coverage':>19} | {'width':>7} | verdict"
    )
    print("-" * 74)
    report: dict[str, Any] = {}
    """Collected per-cell quantities needed to audit the pass decision."""

    failures: list[str] = []
    """Collected every coverage cell excluding the prewritten nominal level."""

    for heritability in HERITABILITIES:
        for rate in rates:
            here: list[dict[str, Any]] = [
                r
                for r in rows
                if r["heritability"] == heritability and r["rate"] == rate
            ]
            """Selected every attempted replicate in this exact scientific cell."""

            refused: list[dict[str, Any]] = [r for r in here if "refusal" in r]
            """Retained failures in the denominator rather than silently dropping them."""

            # Every replicate is scored: one that could not be computed never
            # covers, so it stays in the denominator.
            hits: int = sum(1 for r in here if r.get("covered"))
            """Counted only explicit containment decisions as coverage."""

            low, high = clopper_pearson(hits, len(here))
            """Quantified Monte Carlo uncertainty without a visual tolerance."""

            width: float = float(
                np.mean([r["width"] for r in here if "width" in r])
                if len(here) > len(refused)
                else float("nan")
            )
            """Summarized interval precision among successfully computed profiles."""

            # Two-sided everywhere, including at nought. Over-covering is a
            # fault of the recipe as much as under-covering is, and the whole
            # point of the mixture is that the boundary cell no longer needs
            # an allowance.
            at_boundary: bool = heritability <= 0.0
            """Recorded whether the generating truth uses boundary containment rules."""

            ok: bool = low <= NOMINAL <= high
            """Applied the fixed two-sided exact-binomial acceptance criterion."""

            print(
                f"{heritability:>5.2f} | {rate:>7.0%} | "
                f"{len(refused):>3}/{len(here):<5} | "
                f"{hits / len(here):>6.3f} [{low:.3f},{high:.3f}] | "
                f"{width:>7.3f} | {'ok' if ok else 'OFF NOMINAL'}",
                flush=True,
            )
            report[f"h2={heritability} censored={rate}"] = {
                "replicates": len(here),
                "refused": len(refused),
                "covered": hits,
                "coverage": hits / len(here),
                "clopper_pearson": [low, high],
                "mean_width": float(width),
                "within_nominal": ok,
                "at_boundary": at_boundary,
            }
            """Retained enough facts to independently recompute this verdict."""

            if not ok:
                failures.append(
                    f"at h2 {heritability} and {rate:.0%} censored, coverage "
                    f"{hits / len(here):.3f} [{low:.3f}, {high:.3f}] excludes "
                    f"the nominal {NOMINAL}"
                )

    receipt: dict[str, Any] = {
        "what": "coverage of the censored model's profile interval",
        "date": date.today().isoformat(),
        "pairs": PAIRS,
        "replicates": arguments.replicates,
        "workers": arguments.workers,
        "nominal": NOMINAL,
        "true_variance": TRUE_VARIANCE,
        "censoring_shares": rates,
        "cells": report,
        "passed": not failures,
        "failures": failures,
    }
    """Built the complete exact-cell scientific evidence record."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / (f"tobit-coverage-{receipt['date']}.json")
    )
    """Selected the optional exploratory evidence path outside release mode."""
    if not arguments.no_write:
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"\nwritten to {out}")
    print(json.dumps(receipt, indent=2))
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: every cell's coverage interval covers the nominal level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
