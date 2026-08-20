"""Does the mixed bivariate model's interval contain the truth 95 times in 100?

`mixed_bivariate_calibration.py` measures whether the genetic correlation comes
back near the value it was given. That is recovery of a point estimate, and it
says nothing about the interval around it. `CONTEXT.md` is explicit that the
coverage check is what chooses an interval recipe and that nothing about
reading the code can tell you the same thing, so the interval this model
reported had never been measured at all.

This measures it, and measures the test beside it, because the two answer the
same question and one run gives both.

The rules are the coverage check's rules elsewhere in this package:

1. **Every replicate is scored.** An interval that could not be computed never
   covers, so a refusal stays in the denominator and is reported beside the
   coverage rather than under it.
2. **A cell passes when the Clopper-Pearson interval on its coverage overlaps
   the nominal 0.95**, two-sided. Over-covering is a fault of the recipe as
   much as under-covering is.
3. **No boundary rule applies here.** Nought is an interior point of a
   correlation's range, so there is no mass at the null and no Self-Liang
   mixture to read. That is what separates this from a variance against
   nought, and it is why the test reports `chi2_1`.

Three pairings, because each exercises a different part of the likelihood, and
**each draws its own data**. Correcting the pattern in the calibration receipt,
where the second trait is continuous in every pairing and drawn from the same
stream, so its three heritability means agree to six figures and the three
cells are not three independent confirmations.

Run with:

    ASTERISM_REPLICATES=300 uv run --no-project python checks/mixed_bivariate_coverage.py
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

PAIRS = 400
HERITABILITY = [0.5, 0.5]
RESIDUAL_CORRELATION = 0.15
SECOND_VARIANCE = 3.0
PREVALENCE = 0.3
CENSORED_SHARE = 0.25
TRUTHS = [0.0, 0.4, 0.7]
PAIRINGS = ["continuous", "binary", "censored"]
NOMINAL = 0.95
ALPHA = 0.05
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "300"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "10"))


def draw(pairing: str, truth: float, replicate: int):
    """Two correlated traits on sibling pairs, on a stream of their own.

    The seed carries the pairing and the truth as well as the replicate, so no
    two cells share a draw.
    """
    seed = (
        910_000
        + 7919 * replicate
        + 104_729 * PAIRINGS.index(pairing)
        + 15_485_863 * TRUTHS.index(truth)
    )
    rng = np.random.default_rng(seed)
    n = 2 * PAIRS
    relationship = np.eye(n)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5

    def correlated(rho, a, b):
        return a, rho * a + np.sqrt(1.0 - rho * rho) * b

    shared_one, shared_two = correlated(
        truth, rng.normal(size=PAIRS), rng.normal(size=PAIRS)
    )
    own_one, own_two = correlated(truth, rng.normal(size=n), rng.normal(size=n))
    residual_one, residual_two = correlated(
        RESIDUAL_CORRELATION, rng.normal(size=n), rng.normal(size=n)
    )
    genetic = [
        np.sqrt(0.5) * np.repeat(shared_one, 2) + np.sqrt(0.5) * own_one,
        np.sqrt(0.5) * np.repeat(shared_two, 2) + np.sqrt(0.5) * own_two,
    ]
    residual = [residual_one, residual_two]
    variance = [1.0, SECOND_VARIANCE]
    latent = [
        np.sqrt(HERITABILITY[t] * variance[t]) * genetic[t]
        + np.sqrt((1.0 - HERITABILITY[t]) * variance[t]) * residual[t]
        for t in range(2)
    ]
    return relationship, latent


def encode(pairing: str, latent, n: int):
    """The first trait as the pairing asks for it; the second stays continuous."""
    second = {
        "kind": "continuous",
        "value": latent[1],
        "censoring": np.zeros(n, dtype=np.int64),
        "limit": np.zeros(n),
    }
    if pairing == "continuous":
        first = {
            "kind": "continuous",
            "value": latent[0],
            "censoring": np.zeros(n, dtype=np.int64),
            "limit": np.zeros(n),
        }
    elif pairing == "binary":
        cut = float(np.quantile(latent[0], 1.0 - PREVALENCE))
        first = {
            "kind": "binary",
            "value": np.full(n, np.nan),
            "censoring": np.where(latent[0] > cut, 1, 2).astype(np.int64),
            "limit": np.zeros(n),
        }
    else:
        cut = float(np.quantile(latent[0], 1.0 - CENSORED_SHARE))
        censored = latent[0] >= cut
        first = {
            "kind": "censored",
            "value": np.where(censored, np.nan, latent[0]),
            "censoring": np.where(censored, 1, 0).astype(np.int64),
            "limit": np.full(n, cut),
        }
    return first, second


def one(job):
    pairing, truth, replicate = job
    relationship, latent = draw(pairing, truth, replicate)
    n = latent[0].size
    design = np.ones((n, 1))
    first, second = encode(pairing, latent, n)

    out = {"pairing": pairing, "truth": truth, "replicate": replicate}
    try:
        got = asterism.mixed_bivariate_interval(
            relationship, first, second, design, "genetic_correlation"
        )
        out["covered"] = bool(got["lower"] <= truth <= got["upper"])
        out["width"] = got["upper"] - got["lower"]
        out["lower_at_bound"] = got["lower_at_bound"]
        out["upper_at_bound"] = got["upper_at_bound"]
        out["profile_failures"] = got["profile_failures"]
    except ValueError as refusal:
        out["refusal"] = str(refusal).replace("MIXED_BIVARIATE_", "")
        return out

    try:
        test = asterism.mixed_bivariate_test(
            relationship, first, second, design, "genetic_correlation"
        )
        out["p_value"] = test["p_value"]
        out["rejected"] = bool(test["p_value"] < ALPHA)
    except ValueError as refusal:
        out["test_refusal"] = str(refusal).replace("MIXED_BIVARIATE_", "")
    return out


def clopper_pearson(hits: int, n: int) -> tuple[float, float]:
    low = beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0
    high = beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    return float(low), float(high)


def main() -> int:
    print(
        f"{PAIRS} sibling pairs, {REPLICATES} replicates per cell, "
        f"nominal {NOMINAL}"
    )
    print(
        f"true genetic correlations {TRUTHS}, pairings {PAIRINGS}, "
        f"heritabilities {HERITABILITY}\n",
        flush=True,
    )

    jobs = [
        (p, t, k)
        for p in PAIRINGS
        for t in TRUTHS
        for k in range(REPLICATES)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=2):
            rows.append(got)
            if len(rows) % 200 == 0:
                print(f"  {len(rows)}/{len(jobs)}", flush=True)

    print(
        f"\n{'pairing':>11} | {'rho_g':>5} | {'refused':>9} | "
        f"{'coverage':>19} | {'width':>6} | {'reject':>6} | verdict"
    )
    print("-" * 86)
    report, failures = {}, []
    for pairing in PAIRINGS:
        for truth in TRUTHS:
            here = [
                r for r in rows if r["pairing"] == pairing and r["truth"] == truth
            ]
            refused = [r for r in here if "refusal" in r]
            # Every replicate is scored: one that could not be computed never
            # covers, so it stays in the denominator.
            hits = sum(1 for r in here if r.get("covered"))
            low, high = clopper_pearson(hits, len(here))
            width = (
                float(np.mean([r["width"] for r in here if "width" in r]))
                if len(here) > len(refused)
                else float("nan")
            )
            judged = [r for r in here if "rejected" in r]
            rejected = sum(1 for r in judged if r["rejected"])
            rate = rejected / len(judged) if judged else float("nan")
            ok = low <= NOMINAL <= high
            print(
                f"{pairing:>11} | {truth:>5.2f} | {len(refused):>3}/{len(here):<5} | "
                f"{hits / len(here):>6.3f} [{low:.3f},{high:.3f}] | "
                f"{width:>6.3f} | {rate:>6.3f} | "
                f"{'ok' if ok else 'OFF NOMINAL'}",
                flush=True,
            )
            report[f"pairing={pairing} rho_g={truth}"] = {
                "replicates": len(here),
                "refused": len(refused),
                "covered": hits,
                "coverage": hits / len(here),
                "clopper_pearson": [low, high],
                "mean_width": width,
                "within_nominal": ok,
                # At a true correlation of nought this is the test's level; at
                # the others it is its power. Reported rather than gated,
                # because the pass here is about the interval.
                "rejection_rate": rate,
                "judged": len(judged),
            }
            if not ok:
                failures.append(
                    f"{pairing} at rho_g {truth}, coverage "
                    f"{hits / len(here):.3f} [{low:.3f}, {high:.3f}] excludes "
                    f"the nominal {NOMINAL}"
                )

    receipt = {
        "what": "coverage of the mixed bivariate genetic-correlation interval",
        "date": date.today().isoformat(),
        "pairs": PAIRS,
        "replicates": REPLICATES,
        "nominal": NOMINAL,
        "alpha": ALPHA,
        "heritability": HERITABILITY,
        "residual_correlation": RESIDUAL_CORRELATION,
        "second_variance": SECOND_VARIANCE,
        "prevalence": PREVALENCE,
        "censored_share": CENSORED_SHARE,
        "truths": TRUTHS,
        "cells": report,
        "passed": not failures,
    }
    out = Path(__file__).resolve().parents[1] / "evidence" / (
        f"mixed-bivariate-coverage-{date.today().isoformat()}.json"
    )
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")

    if failures:
        print("\nFAILED:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("\nPASSED: every cell's coverage interval covers the nominal level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
