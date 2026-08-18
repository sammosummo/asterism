"""Does the mixed bivariate model recover a genetic correlation it was given?

Agreement with SOLAR shows the binary-by-continuous cell computes the same
thing SOLAR computes. It says nothing about the censored cell, which SOLAR
cannot fit at all, and nothing about whether either is unbiased. Both are
measured here instead, on many data sets drawn from a known truth.

The genetic correlation is the estimand these pairs are fitted for -- a
diagnosis against hearing -- and it is the one quantity that survives a trait
whose scale is arbitrary. So it is what the pass or fail rests on. The
heritabilities are reported beside it because a correlation recovered from two
badly recovered variances would be luck.

Three pairings, because each exercises a different part:

- **continuous with continuous**, which must match what BivariateModel would do
  and is the control;
- **binary with continuous**, the diagnosis-against-hearing pair;
- **censored with continuous**, the audiometry pair, which no other engine here
  can fit.

Run with:

    ASTERISM_REPLICATES=100 uv run --no-project python checks/mixed_bivariate_calibration.py
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

PAIRS = 400
HERITABILITY = [0.5, 0.5]
GENETIC_CORRELATION = 0.4
RESIDUAL_CORRELATION = 0.15
SECOND_VARIANCE = 3.0
PREVALENCE = 0.3
CENSORED_SHARE = 0.25
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "100"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "10"))
STANDARD_ERRORS_ALLOWED = 3.0
PAIRINGS = ["continuous", "binary", "censored"]


def draw(replicate: int):
    """Two correlated traits on sibling pairs, both on their own scale."""
    rng = np.random.default_rng(880_000 + 7919 * replicate)
    n = 2 * PAIRS
    relationship = np.eye(n)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5

    def correlated(rho, a, b):
        return a, rho * a + np.sqrt(1.0 - rho * rho) * b

    shared_one, shared_two = correlated(
        GENETIC_CORRELATION, rng.normal(size=PAIRS), rng.normal(size=PAIRS)
    )
    own_one, own_two = correlated(
        GENETIC_CORRELATION, rng.normal(size=n), rng.normal(size=n)
    )
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


def one(job):
    pairing, replicate = job
    relationship, latent = draw(replicate)
    n = latent[0].size
    design = np.ones((n, 1))
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

    out = {"pairing": pairing, "replicate": replicate}
    try:
        fit = asterism.mixed_bivariate_fit(relationship, first, second, design)
        out["genetic_correlation"] = fit["genetic_correlation"]
        out["residual_correlation"] = fit["residual_correlation"]
        out["h2_first"] = fit["heritability"][0]
        out["h2_second"] = fit["heritability"][1]
    except ValueError as refusal:
        out["refusal"] = str(refusal).replace("MIXED_BIVARIATE_", "")
    return out


def summarise(values):
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values)
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def main() -> int:
    print(f"{PAIRS} sibling pairs, {REPLICATES} replicates per pairing.")
    print(f"True RhoG {GENETIC_CORRELATION}, RhoE {RESIDUAL_CORRELATION}, "
          f"h2 {HERITABILITY}.\n")
    jobs = [(pairing, r) for pairing in PAIRINGS for r in range(REPLICATES)]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=2):
            rows.append(got)

    print(f"{'first trait':<12} | {'available':>10} | {'RhoG (se)':>18} | "
          f"{'h2 first':>10} | {'h2 second':>10}")
    print("-" * 76)
    report = {}
    failures = []
    for pairing in PAIRINGS:
        here = [r for r in rows if r["pairing"] == pairing]
        fitted = [r for r in here if "genetic_correlation" in r]
        rho_mean, rho_error = summarise([r["genetic_correlation"] for r in fitted])
        first_mean, _ = summarise([r["h2_first"] for r in fitted])
        second_mean, _ = summarise([r["h2_second"] for r in fitted])
        print(f"{pairing:<12} | {len(fitted):>4}/{len(here):<5} | "
              f"{rho_mean:>10.4f} ({rho_error:.4f}) | {first_mean:>10.4f} | "
              f"{second_mean:>10.4f}")
        report[pairing] = {
            "available": len(fitted),
            "attempted": len(here),
            "genetic_correlation_mean": rho_mean,
            "genetic_correlation_standard_error": rho_error,
            "heritability_first_mean": first_mean,
            "heritability_second_mean": second_mean,
        }
        if fitted:
            away = abs(rho_mean - GENETIC_CORRELATION) / max(rho_error, 1e-12)
            if away > STANDARD_ERRORS_ALLOWED:
                failures.append(
                    f"with a {pairing} first trait the genetic correlation is "
                    f"{away:.1f} standard errors from the truth"
                )

    receipt = {
        "what": "recovery of the genetic correlation across trait kinds",
        "date": date.today().isoformat(),
        "pairs": PAIRS,
        "replicates": REPLICATES,
        "truth": {
            "genetic_correlation": GENETIC_CORRELATION,
            "residual_correlation": RESIDUAL_CORRELATION,
            "heritability": HERITABILITY,
        },
        "prevalence": PREVALENCE,
        "censored_share": CENSORED_SHARE,
        "standard_errors_allowed": STANDARD_ERRORS_ALLOWED,
        "by_pairing": report,
        "passed": not failures,
        "failures": failures,
    }
    out = Path(__file__).resolve().parent.parent / "evidence" / (
        f"mixed-bivariate-calibration-{receipt['date']}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: the genetic correlation is recovered for every pairing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
