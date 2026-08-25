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
  can fit;
- **censored with censored**, two audiometry traits at once, which is what a
  genetic correlation between two extended high-frequency thresholds needs. The
  model always allowed it -- the trait kinds are independent -- but until this
  check nothing had measured it.

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

PAIRS: int = 400
"""Fixed the number of sibling pairs in every simulated data set."""

HERITABILITY: list[float] = [0.5, 0.5]
"""Set the generating heritability for each of the two traits."""

GENETIC_CORRELATION: float = 0.4
"""Set the generating genetic correlation tested by the campaign."""

RESIDUAL_CORRELATION: float = 0.15
"""Set the generating residual correlation shared by all pairings."""

SECOND_VARIANCE: float = 3.0
"""Gave the second trait a scale distinct from the first trait."""

PREVALENCE: float = 0.3
"""Set the binary-trait prevalence imposed on the first latent trait."""

CENSORED_SHARE: float = 0.25
"""Set the share of first-trait observations censored from above."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "100"))
"""Selected the requested replicates per pairing from the environment."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "10"))
"""Selected the worker-process count from the environment."""

STANDARD_ERRORS_ALLOWED: float = 3.0
"""Set the maximum accepted Monte Carlo discrepancy from the truth."""

PAIRINGS: list[str] = ["continuous", "binary", "censored", "censored_pair"]
"""Enumerated the first-trait kinds covered by the calibration."""


def draw(replicate: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Draw two correlated traits on sibling pairs on distinct scales.

    Args:
        replicate: Zero-based replicate number used to derive the random seed.

    Returns:
        The sibling relationship matrix and two continuous latent traits.
    """
    rng: np.random.Generator = np.random.default_rng(880_000 + 7919 * replicate)
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
        GENETIC_CORRELATION, rng.normal(size=PAIRS), rng.normal(size=PAIRS)
    )
    """Drew correlated family-shared genetic effects for both traits."""

    own_one, own_two = correlated(
        GENETIC_CORRELATION, rng.normal(size=n), rng.normal(size=n)
    )
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


def one(job: tuple[str, int]) -> dict[str, object]:
    """Fit one simulated data set for one first-trait kind.

    Args:
        job: Pairing name and zero-based replicate number.

    Returns:
        Fit fields or a stable model-refusal suffix for the requested replicate.
    """
    pairing, replicate = job
    """Separated the requested trait kind from its replicate identity."""

    relationship, latent = draw(replicate)
    """Generated the fixed-design relationship matrix and latent traits."""

    n: int = latent[0].size
    """Read the simulated roster size from the first latent trait."""

    design: np.ndarray = np.ones((n, 1))
    """Constructed the intercept-only fixed-effect design."""

    if pairing == "censored_pair":
        second_cut: float = float(np.quantile(latent[1], 1.0 - CENSORED_SHARE))
        """Located the second trait's own upper-censoring limit."""

        second_censored: np.ndarray = latent[1] >= second_cut
        """Marked the second trait's observations at or above its limit."""

        second: dict[str, str | np.ndarray] = {
            "kind": "censored",
            "value": np.where(second_censored, np.nan, latent[1]),
            "censoring": np.where(second_censored, 1, 0).astype(np.int64),
            "limit": np.full(n, second_cut),
        }
        """Censored the second trait too, which is the audiometry-pair case."""
    else:
        second = {
            "kind": "continuous",
            "value": latent[1],
            "censoring": np.zeros(n, dtype=np.int64),
            "limit": np.zeros(n),
        }
        """Represented the continuous second trait every other pairing uses."""

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

    out: dict[str, object] = {"pairing": pairing, "replicate": replicate}
    """Initialised the per-replicate record with its deterministic identity."""

    try:
        fit: dict[str, object] = asterism.mixed_bivariate_fit(
            relationship, first, second, design
        )
        """Fitted the public mixed-bivariate model to the simulated traits."""

        out["genetic_correlation"] = fit["genetic_correlation"]
        """Recorded the fitted genetic correlation used by the pass rule."""

        out["residual_correlation"] = fit["residual_correlation"]
        """Recorded the fitted residual correlation as a diagnostic quantity."""

        out["h2_first"] = fit["heritability"][0]
        """Recorded the fitted first-trait heritability."""

        out["h2_second"] = fit["heritability"][1]
        """Recorded the fitted second-trait heritability."""
    except ValueError as refusal:
        out["refusal"] = str(refusal).replace("MIXED_BIVARIATE_", "")
        """Recorded the stable refusal suffix instead of fabricating estimates."""
    return out


def summarise(values: list[float]) -> tuple[float, float]:
    """Return a mean and its Monte Carlo standard error.

    Args:
        values: Successfully fitted values for one reported quantity.

    Returns:
        The sample mean and standard error, or two NaNs for an empty input.
    """
    if not values:
        return float("nan"), float("nan")
    array: np.ndarray = np.asarray(values)
    """Converted successful scalar results to a numerical vector."""
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def main() -> int:
    """Run all pairing calibrations and write their dated evidence record."""
    print(f"{PAIRS} sibling pairs, {REPLICATES} replicates per pairing.")
    print(
        f"True RhoG {GENETIC_CORRELATION}, RhoE {RESIDUAL_CORRELATION}, "
        f"h2 {HERITABILITY}.\n"
    )
    jobs: list[tuple[str, int]] = [
        (pairing, r) for pairing in PAIRINGS for r in range(REPLICATES)
    ]
    """Enumerated every pairing and replicate exactly once."""

    rows: list[dict[str, object]] = []
    """Initialised the collection of per-replicate fit records."""

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=2):
            rows.append(got)

    print(
        f"{'first trait':<12} | {'available':>10} | {'RhoG (se)':>18} | "
        f"{'h2 first':>10} | {'h2 second':>10}"
    )
    print("-" * 76)
    report: dict[str, dict[str, int | float]] = {}
    """Initialised the summary retained for each first-trait pairing."""

    failures: list[str] = []
    """Initialised the scientific pass-rule failure messages."""

    for pairing in PAIRINGS:
        here: list[dict[str, object]] = [r for r in rows if r["pairing"] == pairing]
        """Selected every attempted replicate for this pairing."""

        fitted: list[dict[str, object]] = [
            r for r in here if "genetic_correlation" in r
        ]
        """Selected replicates that returned fitted scientific quantities."""

        rho_mean, rho_error = summarise([r["genetic_correlation"] for r in fitted])
        """Summarised genetic-correlation recovery and Monte Carlo uncertainty."""

        first_mean, _ = summarise([r["h2_first"] for r in fitted])
        """Summarised first-trait heritability recovery."""

        second_mean, _ = summarise([r["h2_second"] for r in fitted])
        """Summarised second-trait heritability recovery."""

        print(
            f"{pairing:<12} | {len(fitted):>4}/{len(here):<5} | "
            f"{rho_mean:>10.4f} ({rho_error:.4f}) | {first_mean:>10.4f} | "
            f"{second_mean:>10.4f}"
        )
        report[pairing] = {
            "available": len(fitted),
            "attempted": len(here),
            "genetic_correlation_mean": rho_mean,
            "genetic_correlation_standard_error": rho_error,
            "heritability_first_mean": first_mean,
            "heritability_second_mean": second_mean,
        }
        """Recorded availability and Monte Carlo summaries for this pairing."""

        if fitted:
            away: float = abs(rho_mean - GENETIC_CORRELATION) / max(rho_error, 1e-12)
            """Measured genetic-correlation bias in Monte Carlo standard errors."""

            if away > STANDARD_ERRORS_ALLOWED:
                failures.append(
                    f"with a {pairing} first trait the genetic correlation is "
                    f"{away:.1f} standard errors from the truth"
                )

    receipt: dict[str, object] = {
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
    """Assembled the complete dated scientific evidence receipt."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / (f"mixed-bivariate-calibration-{receipt['date']}.json")
    )
    """Selected the repository evidence path using the receipt date."""

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
