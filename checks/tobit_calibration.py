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

PAIRS = 400
TRUE_HERITABILITY = 0.5
TRUE_VARIANCE = 4.0
TRUE_MEAN = 10.0
REPLICATES = int(os.environ.get("ASTERISM_REPLICATES", "200"))
WORKERS = int(os.environ.get("ASTERISM_WORKERS", "10"))
# The share of the sample the instrument fails to reach.
RATES = [0.0, 0.10, 0.25, 0.50, 0.75]
# Recovery is judged against the replicate-to-replicate spread, not by eye.
STANDARD_ERRORS_ALLOWED = 3.0


def draw(rate: float, replicate: int):
    """One data set: sibling pairs, then censored at the quantile wanted."""
    rng = np.random.default_rng(500_000 + 7919 * replicate)
    n = 2 * PAIRS
    shared = rng.normal(size=PAIRS) * np.sqrt(TRUE_HERITABILITY / 2.0)
    own = rng.normal(size=n) * np.sqrt(1.0 - TRUE_HERITABILITY / 2.0)
    standard = np.repeat(shared, 2) + own
    complete = TRUE_MEAN + np.sqrt(TRUE_VARIANCE) * standard

    relationship = np.eye(n)
    for pair in range(PAIRS):
        relationship[2 * pair, 2 * pair + 1] = 0.5
        relationship[2 * pair + 1, 2 * pair] = 0.5

    if rate <= 0.0:
        limit = np.inf
        censored = np.zeros(n, dtype=bool)
    else:
        limit = float(np.quantile(complete, 1.0 - rate))
        censored = complete >= limit
    return relationship, complete, censored, limit


def one(job):
    rate, replicate = job
    relationship, complete, censored, limit = draw(rate, replicate)
    n = complete.size
    design = np.ones((n, 1))
    limits = np.full(n, limit if np.isfinite(limit) else 0.0)

    out = {"rate": rate, "replicate": replicate}
    try:
        fit = asterism.tobit_fit(
            relationship,
            np.where(censored, np.nan, complete),
            np.where(censored, 1, 0).astype(np.int64),
            limits,
            design,
        )
        out["heritability"] = fit["heritability"]
        out["variance"] = fit["total_variance"]
    except ValueError as refusal:
        out["refusal"] = str(refusal).replace("TOBIT_", "")

    # The same data with the limit substituted, fitted as though complete.
    try:
        naive = asterism.tobit_fit(
            relationship,
            np.where(censored, limit, complete) if np.isfinite(limit) else complete,
            np.zeros(n, dtype=np.int64),
            limits,
            design,
        )
        out["naive_heritability"] = naive["heritability"]
        out["naive_variance"] = naive["total_variance"]
    except ValueError:
        pass
    return out


def summarise(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values)
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def main() -> int:
    print(f"{PAIRS} sibling pairs, {REPLICATES} replicates per rate. "
          f"True heritability {TRUE_HERITABILITY}, variance {TRUE_VARIANCE}.\n")
    jobs = [(rate, replicate) for rate in RATES for replicate in range(REPLICATES)]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(one, jobs, chunksize=4):
            rows.append(got)

    print(f"{'censored':>9} | {'available':>9} | {'h2 (se)':>16} | "
          f"{'variance (se)':>18} | {'h2 if substituted':>18}")
    print("-" * 88)
    report = {}
    failures = []
    for rate in RATES:
        here = [r for r in rows if r["rate"] == rate]
        fitted = [r for r in here if "heritability" in r]
        h2_mean, h2_error = summarise([r["heritability"] for r in fitted])
        var_mean, var_error = summarise([r["variance"] for r in fitted])
        naive_mean, _ = summarise(
            [r["naive_heritability"] for r in here if "naive_heritability" in r]
        )
        print(f"{rate:>8.0%} | {len(fitted):>4}/{len(here):<4} | "
              f"{h2_mean:>9.4f} ({h2_error:.4f}) | "
              f"{var_mean:>11.4f} ({var_error:.4f}) | {naive_mean:>18.4f}")
        report[f"{rate:.2f}"] = {
            "available": len(fitted),
            "attempted": len(here),
            "heritability_mean": h2_mean,
            "heritability_standard_error": h2_error,
            "variance_mean": var_mean,
            "variance_standard_error": var_error,
            "naive_heritability_mean": naive_mean,
        }
        # A rate passes when the truth is within a few standard errors of the
        # mean estimate. That is a statement about this many replicates, not
        # about the estimate looking close by eye.
        if fitted:
            away = abs(h2_mean - TRUE_HERITABILITY) / max(h2_error, 1e-12)
            if away > STANDARD_ERRORS_ALLOWED:
                failures.append(
                    f"at {rate:.0%} censored the heritability is {away:.1f} "
                    f"standard errors from the truth"
                )

    receipt = {
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
    out = Path(__file__).resolve().parent.parent / "evidence" / (
        f"tobit-calibration-{receipt['date']}.json")
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
