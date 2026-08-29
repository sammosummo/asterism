"""The two-person normal probability against an independent integration.

Every censored and liability heritability rests on one region probability, and
at exactly two coordinates that region probability is a bivariate normal
integral. This recomputes it here, in Python, from Owen's angular form and again
from the conditional form, and compares both against the crate.

**It exists because a hard-coded reference is not a check.** `src/normal_integrals.rs`
carries a table of 53 points said to have been computed independently. Nothing
in the repository could recompute them, so the claim rested on a sentence.
CONTEXT.md records what that costs: two comparisons once cited agreement with R
packages that were not installed, so nobody could reproduce what they asserted.

**Why two forms rather than one.** Owen's form integrates over an angle to
`asin(rho)` and gets stiff as the correlation approaches one; the conditional
form integrates over a coordinate and gets stiff where the conditioning becomes
a step. They are stiff in different places, so where they agree, neither is
being believed on its own.

**The retired quadrature is why the tolerance is where it is.** Until 29 August
2026 the crate integrated this with a fixed sixteen-point rule whose error
reached 4.5e-02 in the log probability at a correlation of 0.99 -- and only
2.3e-04 at thresholds of nought and nought, which was the one place it had been
measured. The grid below therefore moves the rectangle as well as the
correlation, because the error depends on both.

Run with:

    uv run --no-project python checks/bivariate_against_owen.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import asterism
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

THRESHOLDS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (0.5, -0.3),
    (-1.2, 0.8),
    (2.0, 1.5),
    (-2.5, -2.0),
]
"""Rectangles at the symmetric centre and away from it, including the tail."""

CORRELATIONS: list[float] = [
    -0.99,
    -0.9,
    -0.5,
    0.0,
    0.3,
    0.5,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9999,
]
"""Correlations from strongly negative to all but degenerate."""

TOLERANCE: float = 1.0e-9
"""Largest absolute error in the log probability this check will accept."""

AGREEMENT: float = 1.0e-12
"""How closely the two independent forms must agree before either is believed."""

FLOOR: float = 1.0e-12
"""Below this probability neither form resolves an answer worth comparing."""


def owen_form(first: float, second: float, correlation: float) -> float:
    """The bivariate normal distribution function by Owen's angular form.

    Args:
        first: The first coordinate's upper limit.
        second: The second coordinate's upper limit.
        correlation: The correlation between the two coordinates.

    Returns:
        The probability that both coordinates fall below their limits.
    """
    if correlation < 0.0:
        return norm.cdf(first) - owen_form(first, -second, -correlation)
    """Reflected the negative case onto the positive one, which the form covers."""

    if correlation == 0.0:
        return float(norm.cdf(first) * norm.cdf(second))

    def integrand(angle: float) -> float:
        sine: float = float(np.sin(angle))
        """Took the angle's sine, which carries the correlation into the exponent."""

        cosine: float = float(np.cos(angle))
        """Took the angle's cosine, which narrows the exponent towards the corner."""

        return float(
            np.exp(
                -(first * first - 2.0 * first * second * sine + second * second)
                / (2.0 * cosine * cosine)
            )
        )

    value, _ = quad(
        integrand, 0.0, np.arcsin(correlation), epsabs=1e-15, epsrel=1e-15, limit=800
    )
    """Integrated the angular form to the correlation's arcsine."""
    return float(norm.cdf(first) * norm.cdf(second) + value / (2.0 * np.pi))


def conditional_form(first: float, second: float, correlation: float) -> float:
    """The same probability by conditioning on the first coordinate.

    Args:
        first: The first coordinate's upper limit.
        second: The second coordinate's upper limit.
        correlation: The correlation between the two coordinates.

    Returns:
        The probability that both coordinates fall below their limits.
    """
    spread: float = float(np.sqrt((1.0 - correlation) * (1.0 + correlation)))
    """Took the conditional standard deviation, factored to avoid cancellation."""

    def integrand(value: float) -> float:
        return float(
            norm.pdf(value) * norm.cdf((second - correlation * value) / spread)
        )

    step: list[float] | None = None
    """Initialised the integrator's hint with no step to point at."""

    if correlation != 0.0 and -12.0 < second / correlation < first:
        step = [second / correlation]
        """Located the step for the integrator."""
    """Told the integrator where the conditioning turns into a step, which is the
    only place this form is difficult."""

    value, _ = quad(
        integrand, -12.0, first, epsabs=1e-15, epsrel=1e-15, limit=800, points=step
    )
    """Integrated the conditional form across the first coordinate."""
    return float(value)


def main() -> int:
    """Compare the crate against both independent forms, and report the worst."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Created the parser for the tolerance and the evidence switch."""

    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--no-write", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()
    """Read the tolerance and whether an evidence record is wanted."""

    rows: list[dict[str, Any]] = []
    """Collected one record per rectangle and correlation."""

    failures: list[str] = []
    """Collected every point the crate missed by more than the tolerance."""

    print(f"{'a':>6} {'b':>6} {'rho':>8} {'reference':>14} {'crate':>14} {'error':>10}")
    print("-" * 64)
    for first, second in THRESHOLDS:
        for correlation in CORRELATIONS:
            owen: float = owen_form(first, second, correlation)
            """Took the angular form's answer."""

            conditional: float = conditional_form(first, second, correlation)
            """Took the conditional form's answer."""

            if owen < FLOOR:
                continue
            """Skipped a probability neither form resolves, rather than comparing noise."""

            disagreement: float = abs(owen - conditional)
            """Measured how far the two independent forms sit apart."""

            if disagreement > AGREEMENT:
                failures.append(
                    f"the two independent forms disagree at ({first}, {second}) "
                    f"and correlation {correlation} by {disagreement:.2e}, so "
                    f"neither is a reference here"
                )
                continue
            """Refused to use a reference the two forms do not agree on."""

            covariance: np.ndarray = np.array([[1.0, correlation], [correlation, 1.0]])
            """Built the standardised covariance the crate is asked about."""

            crate: float = asterism.region_log_probability(
                np.array([first, second]), np.ones(2), covariance
            )
            """Asked the crate for the same region, through its only public door."""

            error: float = abs(crate - float(np.log(owen)))
            """Took the discrepancy in the log probability, which is what enters a
            likelihood."""

            print(
                f"{first:>6} {second:>6} {correlation:>8} {np.log(owen):>14.9f} "
                f"{crate:>14.9f} {error:>10.2e}"
            )
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "correlation": correlation,
                    "reference_log_probability": float(np.log(owen)),
                    "crate_log_probability": crate,
                    "absolute_error": error,
                    "forms_disagree_by": disagreement,
                }
            )
            if error > arguments.tolerance:
                failures.append(
                    f"at ({first}, {second}) and correlation {correlation} the "
                    f"crate is out by {error:.2e}"
                )

    worst: float = max((r["absolute_error"] for r in rows), default=0.0)
    """Took the largest discrepancy anywhere on the grid."""

    receipt: dict[str, Any] = {
        "what": "the two-person normal probability against two independent integrations",
        "date": date.today().isoformat(),
        "tolerance": arguments.tolerance,
        "agreement_required_between_forms": AGREEMENT,
        "points": len(rows),
        "worst_absolute_error": worst,
        "grid": rows,
        "passed": not failures,
        "failures": failures,
    }
    """Built the evidence record for the comparison."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / f"bivariate-against-owen-{receipt['date']}.json"
    )
    """Selected the evidence path used outside release mode."""
    if not arguments.no_write:
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"\nwritten to {out}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"\nPASSED: {len(rows)} points, worst absolute error {worst:.2e} in the log "
        f"probability, against a tolerance of {arguments.tolerance:.0e}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
