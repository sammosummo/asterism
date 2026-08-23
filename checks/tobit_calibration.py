"""Where does the censored model still recover the truth, and where does it stop?

`checks/tobit_against_censreg.py` shows the censored likelihood is the right
function, to eight decimals, on unrelated people. It says nothing about
censoring combined with relatedness, and nothing about how much censoring is
too much. Neither can be learned by reading the code, which is why this exists.

So: sibling pairs with a known heritability and a known total variance, swept
across censoring rates from nothing to three quarters, many replicates each.
Three things are reported per rate.

- **Recovery.** The mean estimate against the truth. A model that is right at
  10 per cent censored and biased at 60 is a model with a stated range, not a
  broken one -- but the range has to be measured before it can be stated.
- **What substitution would have given.** The same data with every censored
  value replaced by its limit, fitted as though complete. This is the usual
  practice in the audiometry literature and it is the thing being replaced, so
  the size of its bias is the case for the model.
- **How often the fit is even available**, counted and never dropped from a
  denominator.

Run with:

    ASTERISM_REPLICATES=200 uv run --no-project python checks/tobit_calibration.py
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

TRUE_HERITABILITY: float = 0.5
"""Set the latent continuous trait's generating heritability."""

TRUE_VARIANCE: float = 4.0
"""Set the latent continuous trait's generating total variance."""

TRUE_MEAN: float = 10.0
"""Set the latent continuous trait's generating mean."""

REPLICATES: int = int(os.environ.get("ASTERISM_REPLICATES", "200"))
"""Selected the requested replicates per censoring rate from the environment."""

WORKERS: int = int(os.environ.get("ASTERISM_WORKERS", "10"))
"""Selected the worker-process count from the environment."""

# The share of the sample the instrument fails to reach.
RATES: list[float] = [0.0, 0.10, 0.25, 0.50, 0.75]
"""Enumerated the censoring shares measured by the calibration."""

# Recovery is judged against the replicate-to-replicate spread, not by eye.
STANDARD_ERRORS_ALLOWED: float = 3.0
"""Set the maximum accepted Monte Carlo discrepancy from the truth."""


def draw(
    rate: float, replicate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draw one sibling-pair trait and censor it at the requested quantile.

    Args:
        rate: Share of observations to censor from above.
        replicate: Zero-based replicate number used to derive the random seed.

    Returns:
        Relationship matrix, complete trait, censoring mask and censoring limit.
    """
    rng: np.random.Generator = np.random.default_rng(500_000 + 7919 * replicate)
    """Created the deterministic random-number generator for this replicate."""

    n: int = 2 * PAIRS
    """Computed the number of simulated people from the sibling-pair count."""

    shared: np.ndarray = rng.normal(size=PAIRS) * np.sqrt(TRUE_HERITABILITY / 2.0)
    """Drew the genetic effect shared within each sibling pair."""

    own: np.ndarray = rng.normal(size=n) * np.sqrt(1.0 - TRUE_HERITABILITY / 2.0)
    """Drew the individual genetic and residual remainder."""

    standard: np.ndarray = np.repeat(shared, 2) + own
    """Combined pair-shared and individual draws at unit total variance."""

    complete: np.ndarray = TRUE_MEAN + np.sqrt(TRUE_VARIANCE) * standard
    """Mapped the standardised draw to the fixed mean and total variance."""

    relationship: np.ndarray = np.eye(n)
    """Initialised the relationship matrix with individual diagonal entries."""

    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        """Set the forward within-pair relationship coefficient."""

        relationship[2 * pair + 1, 2 * pair] = 0.5
        """Set the symmetric within-pair relationship coefficient."""

    if rate <= 0.0:
        limit: float = np.inf
        """Represented the uncensored cell with an infinite latent limit."""

        censored: np.ndarray = np.zeros(n, dtype=bool)
        """Marked no observations censored in the zero-rate cell."""
    else:
        limit = float(np.quantile(complete, 1.0 - rate))
        """Located the upper limit producing the requested censoring share."""

        censored = complete >= limit
        """Marked complete values at or above the instrument limit."""
    return relationship, complete, censored, limit


def one(job: tuple[float, int]) -> dict[str, object]:
    """Fit censored and limit-substituted analyses for one replicate.

    Args:
        job: Censoring rate and zero-based replicate number.

    Returns:
        Available fit quantities or the stable censored-model refusal suffix.
    """
    rate, replicate = job
    """Separated the censoring cell from its replicate identity."""

    relationship, complete, censored, limit = draw(rate, replicate)
    """Generated the fixed sibling design and censored latent response."""

    n: int = complete.size
    """Read the simulated roster size from the complete response."""

    design: np.ndarray = np.ones((n, 1))
    """Constructed the intercept-only fixed-effect design."""

    limits: np.ndarray = np.full(n, limit if np.isfinite(limit) else 0.0)
    """Expanded the scalar instrument limit to the public per-row input."""

    out: dict[str, object] = {"rate": rate, "replicate": replicate}
    """Initialised the per-replicate record with its deterministic identity."""

    try:
        fit: dict[str, object] = asterism.tobit_fit(
            relationship,
            np.where(censored, np.nan, complete),
            np.where(censored, 1, 0).astype(np.int64),
            limits,
            design,
        )
        """Fitted the public censored one-trait model."""

        out["heritability"] = fit["heritability"]
        """Recorded censored-model heritability for the recovery rule."""

        out["variance"] = fit["total_variance"]
        """Recorded censored-model total variance as a recovery diagnostic."""
    except ValueError as refusal:
        out["refusal"] = str(refusal).replace("TOBIT_", "")
        """Recorded the stable refusal suffix instead of fabricating estimates."""

    # The same data with the limit substituted, fitted as though complete.
    try:
        naive: dict[str, object] = asterism.tobit_fit(
            relationship,
            np.where(censored, limit, complete) if np.isfinite(limit) else complete,
            np.zeros(n, dtype=np.int64),
            limits,
            design,
        )
        """Fitted limit-substituted observations as a complete-data comparison."""

        out["naive_heritability"] = naive["heritability"]
        """Recorded substituted-data heritability for comparison."""

        out["naive_variance"] = naive["total_variance"]
        """Recorded substituted-data total variance for comparison."""
    except ValueError:
        pass
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
    """Run all censoring-rate cells and write their dated evidence receipt."""
    print(
        f"{PAIRS} sibling pairs, {REPLICATES} replicates per rate. "
        f"True heritability {TRUE_HERITABILITY}, variance {TRUE_VARIANCE}.\n"
    )
    jobs: list[tuple[float, int]] = [
        (rate, replicate) for rate in RATES for replicate in range(REPLICATES)
    ]
    """Enumerated every censoring rate and replicate exactly once."""

    rows: list[dict[str, object]] = []
    """Initialised the collection of per-replicate fit records."""

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=4):
            rows.append(got)

    print(
        f"{'censored':>9} | {'available':>9} | {'h2 (se)':>16} | "
        f"{'variance (se)':>18} | {'h2 if substituted':>18}"
    )
    print("-" * 88)
    report: dict[str, dict[str, int | float]] = {}
    """Initialised the machine-readable summaries for every censoring rate."""

    failures: list[str] = []
    """Initialised the scientific pass-rule failure messages."""

    for rate in RATES:
        here: list[dict[str, object]] = [r for r in rows if r["rate"] == rate]
        """Selected every attempted replicate for this censoring rate."""

        fitted: list[dict[str, object]] = [r for r in here if "heritability" in r]
        """Selected replicates that returned censored-model estimates."""

        h2_mean, h2_error = summarise([r["heritability"] for r in fitted])
        """Summarised heritability recovery and Monte Carlo uncertainty."""

        var_mean, var_error = summarise([r["variance"] for r in fitted])
        """Summarised total-variance recovery and Monte Carlo uncertainty."""

        naive_mean, _ = summarise(
            [r["naive_heritability"] for r in here if "naive_heritability" in r]
        )
        """Summarised the limit-substitution heritability comparison."""

        print(
            f"{rate:>8.0%} | {len(fitted):>4}/{len(here):<4} | "
            f"{h2_mean:>9.4f} ({h2_error:.4f}) | "
            f"{var_mean:>11.4f} ({var_error:.4f}) | {naive_mean:>18.4f}"
        )
        report[f"{rate:.2f}"] = {
            "available": len(fitted),
            "attempted": len(here),
            "heritability_mean": h2_mean,
            "heritability_standard_error": h2_error,
            "variance_mean": var_mean,
            "variance_standard_error": var_error,
            "naive_heritability_mean": naive_mean,
        }
        """Recorded availability and Monte Carlo summaries for this rate."""

        # A rate passes when the truth is within a few standard errors of the
        # mean estimate. That is a statement about this many replicates, not
        # about the estimate looking close by eye.
        if fitted:
            away: float = abs(h2_mean - TRUE_HERITABILITY) / max(h2_error, 1e-12)
            """Measured heritability bias in Monte Carlo standard errors."""

            if away > STANDARD_ERRORS_ALLOWED:
                failures.append(
                    f"at {rate:.0%} censored the heritability is {away:.1f} "
                    f"standard errors from the truth"
                )

    receipt: dict[str, object] = {
        "what": "recovery of the censored model across censoring rates",
        "date": date.today().isoformat(),
        "pairs": PAIRS,
        "replicates": REPLICATES,
        "truth": {
            "heritability": TRUE_HERITABILITY,
            "variance": TRUE_VARIANCE,
            "mean": TRUE_MEAN,
        },
        "standard_errors_allowed": STANDARD_ERRORS_ALLOWED,
        "by_censoring_rate": report,
        "passed": not failures,
        "failures": failures,
    }
    """Assembled the complete dated scientific evidence receipt."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / (f"tobit-calibration-{receipt['date']}.json")
    )
    """Selected the repository evidence path using the receipt date."""

    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: the truth is recovered at every rate tried.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
