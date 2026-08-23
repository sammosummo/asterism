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

PAIRS: int = 400
"""Fixed the number of sibling pairs in every simulated data set."""

HERITABILITY: list[float] = [0.5, 0.5]
"""Set the generating heritability for each of the two traits."""

RESIDUAL_CORRELATION: float = 0.15
"""Set the generating residual correlation shared by all cells."""

SECOND_VARIANCE: float = 3.0
"""Gave the second trait a scale distinct from the first trait."""

PREVALENCE: float = 0.3
"""Set the binary-trait prevalence imposed on the first latent trait."""

CENSORED_SHARE: float = 0.25
"""Set the share of first-trait observations censored from above."""

TRUTHS: list[float] = [0.0, 0.4, 0.7]
"""Enumerated the true genetic correlations checked for coverage."""

PAIRINGS: list[str] = ["continuous", "binary", "censored"]
"""Enumerated the first-trait kinds covered by the campaign."""

NOMINAL: float = 0.95
"""Set the coverage probability the interval claims."""

ALPHA: float = 0.05
"""Set the test-rejection threshold reported beside interval coverage."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "300"))
"""Selected the requested replicates per cell from the environment."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "10"))
"""Selected the worker-process count from the environment."""


def draw(
    pairing: str, truth: float, replicate: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Two correlated traits on sibling pairs, on a stream of their own.

    The seed carries the pairing and the truth as well as the replicate, so no
    two cells share a draw.

    Args:
        pairing: First-trait kind identifying the simulation cell.
        truth: Generating genetic correlation for the simulation cell.
        replicate: Zero-based replicate number.

    Returns:
        The sibling relationship matrix and two continuous latent traits.
    """
    seed: int = (
        910_000
        + 7919 * replicate
        + 104_729 * PAIRINGS.index(pairing)
        + 15_485_863 * TRUTHS.index(truth)
    )
    """Derived a distinct deterministic seed for this exact campaign cell."""

    rng: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic random-number generator for this replicate."""

    n: int = 2 * PAIRS
    """Computed the number of simulated people from the sibling-pair count."""

    relationship: np.ndarray = np.eye(n)
    """Initialised the relationship matrix with individual diagonal entries."""

    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Set the forward within-pair relationship coefficient."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Set the symmetric within-pair relationship coefficient."""

    def correlated(
        rho: float, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine independent draws into two vectors with correlation ``rho``.

        Args:
            rho: Desired correlation between the returned vectors.
            a: First standard-normal draw.
            b: Independent standard-normal draw.

        Returns:
            The first draw and a correlated linear combination of both draws.
        """
        return a, rho * a + np.sqrt(1.0 - rho * rho) * b

    shared_one, shared_two = correlated(
        truth, rng.normal(size=PAIRS), rng.normal(size=PAIRS)
    )
    """Drew correlated family-shared genetic effects for both traits."""

    own_one, own_two = correlated(truth, rng.normal(size=n), rng.normal(size=n))
    """Drew correlated individual genetic effects for both traits."""

    residual_one, residual_two = correlated(
        RESIDUAL_CORRELATION, rng.normal(size=n), rng.normal(size=n)
    )
    """Drew correlated individual residual effects for both traits."""

    genetic: list[np.ndarray] = [
        np.sqrt(0.5) * np.repeat(shared_one, 2) + np.sqrt(0.5) * own_one,
        np.sqrt(0.5) * np.repeat(shared_two, 2) + np.sqrt(0.5) * own_two,
    ]
    """Combined shared and individual draws into unit-variance genetic effects."""

    residual: list[np.ndarray] = [residual_one, residual_two]
    """Retained the correlated residual draws for the two traits."""

    variance: list[float] = [1.0, SECOND_VARIANCE]
    """Assigned distinct total variances to the two latent traits."""

    latent: list[np.ndarray] = [
        np.sqrt(HERITABILITY[t] * variance[t]) * genetic[t]
        + np.sqrt((1.0 - HERITABILITY[t]) * variance[t]) * residual[t]
        for t in range(2)
    ]
    """Mixed genetic and residual effects at the fixed generating variances."""
    return relationship, latent


def encode(
    pairing: str, latent: list[np.ndarray], n: int
) -> tuple[dict[str, str | np.ndarray], dict[str, str | np.ndarray]]:
    """Encode the requested first-trait kind and a continuous second trait.

    Args:
        pairing: First-trait kind to represent.
        latent: Two continuous latent traits in participant order.
        n: Number of simulated participants.

    Returns:
        Public mixed-bivariate trait mappings for the first and second traits.
    """
    second: dict[str, str | np.ndarray] = {
        "kind": "continuous",
        "value": latent[1],
        "censoring": np.zeros(n, dtype=np.int64),
        "limit": np.zeros(n),
    }
    """Represented the always-continuous second trait for the public fit."""

    if pairing == "continuous":
        first: dict[str, str | np.ndarray] = {
            "kind": "continuous",
            "value": latent[0],
            "censoring": np.zeros(n, dtype=np.int64),
            "limit": np.zeros(n),
        }
        """Represented the uncensored continuous first trait."""
    elif pairing == "binary":
        cut: float = float(np.quantile(latent[0], 1.0 - PREVALENCE))
        """Located the latent threshold giving the fixed binary prevalence."""

        first = {
            "kind": "binary",
            "value": np.full(n, np.nan),
            "censoring": np.where(latent[0] > cut, 1, 2).astype(np.int64),
            "limit": np.zeros(n),
        }
        """Encoded the first trait as binary threshold outcomes."""
    else:
        cut = float(np.quantile(latent[0], 1.0 - CENSORED_SHARE))
        """Located the upper-censoring limit giving the fixed censored share."""

        censored: np.ndarray = latent[0] >= cut
        """Marked latent observations at or above the censoring limit."""

        first = {
            "kind": "censored",
            "value": np.where(censored, np.nan, latent[0]),
            "censoring": np.where(censored, 1, 0).astype(np.int64),
            "limit": np.full(n, cut),
        }
        """Encoded observed values and right-censoring limits for the first trait."""
    return first, second


def one(job: tuple[str, float, int]) -> dict[str, object]:
    """Fit one interval and test for a deterministic simulation cell.

    Args:
        job: Pairing, generating genetic correlation and replicate number.

    Returns:
        Available interval and test fields or stable refusal suffixes.
    """
    pairing, truth, replicate = job
    """Separated the simulation-cell settings from its replicate identity."""

    relationship, latent = draw(pairing, truth, replicate)
    """Generated the fixed-design relationship matrix and latent traits."""

    n: int = latent[0].size
    """Read the simulated roster size from the first latent trait."""

    design: np.ndarray = np.ones((n, 1))
    """Constructed the intercept-only fixed-effect design."""

    first, second = encode(pairing, latent, n)
    """Encoded both latent traits for the public mixed-bivariate interface."""

    out: dict[str, object] = {
        "pairing": pairing,
        "truth": truth,
        "replicate": replicate,
    }
    """Initialised the per-replicate record with its deterministic identity."""

    try:
        got: dict[str, object] = asterism.mixed_bivariate_interval(
            relationship, first, second, design, "genetic_correlation"
        )
        """Computed the public profile interval for the genetic correlation."""

        out["covered"] = bool(got["lower"] <= truth <= got["upper"])
        """Scored whether this replicate's interval contained its generating truth."""

        out["width"] = got["upper"] - got["lower"]
        """Recorded the interval width as a precision diagnostic."""

        out["lower_limited"] = got["lower_limited"]
        """Recorded whether the lower endpoint reached the parameter bound."""

        out["upper_limited"] = got["upper_limited"]
        """Recorded whether the upper endpoint reached the parameter bound."""

        out["profile_failures"] = got["profile_failures"]
        """Recorded failed profile evaluations for fail-closed review."""
    except ValueError as refusal:
        out["refusal"] = str(refusal).replace("MIXED_BIVARIATE_", "")
        """Recorded the stable interval-refusal suffix in the scored denominator."""
        return out

    try:
        test: dict[str, object] = asterism.mixed_bivariate_test(
            relationship, first, second, design, "genetic_correlation"
        )
        """Computed the public genetic-correlation likelihood-ratio test."""

        out["p_value"] = test["p_value"]
        """Recorded the test p-value for level and power summaries."""

        out["rejected"] = bool(test["p_value"] < ALPHA)
        """Scored rejection at the fixed reporting threshold."""
    except ValueError as refusal:
        out["test_refusal"] = str(refusal).replace("MIXED_BIVARIATE_", "")
        """Recorded the stable test-refusal suffix without altering coverage."""
    return out


def clopper_pearson(hits: int, n: int) -> tuple[float, float]:
    """Return the exact two-sided 95 per cent binomial interval.

    Args:
        hits: Number of intervals that contained their generating truth.
        n: Total scored replicates, including interval refusals.

    Returns:
        Lower and upper Clopper-Pearson endpoints.
    """
    low: float = beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0
    """Computed the exact lower endpoint, including the zero-hit boundary."""

    high: float = beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    """Computed the exact upper endpoint, including the all-hit boundary."""
    return float(low), float(high)


def main() -> int:
    """Run every coverage cell and write its dated evidence receipt."""
    print(f"{PAIRS} sibling pairs, {REPLICATES} replicates per cell, nominal {NOMINAL}")
    print(
        f"true genetic correlations {TRUTHS}, pairings {PAIRINGS}, "
        f"heritabilities {HERITABILITY}\n",
        flush=True,
    )

    jobs: list[tuple[str, float, int]] = [
        (p, t, k) for p in PAIRINGS for t in TRUTHS for k in range(REPLICATES)
    ]
    """Enumerated every pairing, truth and replicate exactly once."""

    rows: list[dict[str, object]] = []
    """Initialised the collection of per-replicate fit records."""

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
    report: dict[str, dict[str, object]] = {}
    """Initialised the machine-readable summaries for every campaign cell."""

    failures: list[str] = []
    """Initialised the scientific pass-rule failure messages."""

    for pairing in PAIRINGS:
        for truth in TRUTHS:
            here: list[dict[str, object]] = [
                r for r in rows if r["pairing"] == pairing and r["truth"] == truth
            ]
            """Selected every attempted replicate for the current campaign cell."""

            refused: list[dict[str, object]] = [r for r in here if "refusal" in r]
            """Selected interval refusals while retaining them in the denominator."""

            # Every replicate is scored: one that could not be computed never
            # covers, so it stays in the denominator.
            hits: int = sum(1 for r in here if r.get("covered"))
            """Counted intervals covering truth across all attempted replicates."""

            low, high = clopper_pearson(hits, len(here))
            """Computed the exact uncertainty interval for observed coverage."""

            width: float = (
                float(np.mean([r["width"] for r in here if "width" in r]))
                if len(here) > len(refused)
                else float("nan")
            )
            """Averaged widths only across intervals that were actually produced."""

            judged: list[dict[str, object]] = [r for r in here if "rejected" in r]
            """Selected replicates whose likelihood-ratio test completed."""

            rejected: int = sum(1 for r in judged if r["rejected"])
            """Counted completed tests rejecting at the fixed alpha level."""

            rate: float = rejected / len(judged) if judged else float("nan")
            """Computed the conditional rejection rate for completed tests."""

            ok: bool = low <= NOMINAL <= high
            """Applied the pre-written overlap rule to the nominal coverage."""

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
            """Recorded coverage, precision and rejection summaries for this cell."""

            if not ok:
                failures.append(
                    f"{pairing} at rho_g {truth}, coverage "
                    f"{hits / len(here):.3f} [{low:.3f}, {high:.3f}] excludes "
                    f"the nominal {NOMINAL}"
                )

    receipt: dict[str, object] = {
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
    """Assembled the complete dated scientific evidence receipt."""

    out: Path = (
        Path(__file__).resolve().parents[1]
        / "evidence"
        / (f"mixed-bivariate-coverage-{date.today().isoformat()}.json")
    )
    """Selected the repository evidence path using the current date."""

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
