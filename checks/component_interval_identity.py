"""Check the reportable component interval's scale and record identity.

This is a deterministic scientific identity check, not a coverage simulation.
It fits the same covariance after multiplying its submitted relationship matrix
by ten.  The raw coefficient changes reciprocally, while the mean-diagonal
contribution, proportion, and profile interval must remain the same.  The
interval estimate must also be the exact quantity reported by ``fit``.

The command uses only Asterism's documented Python interface and exits with
status one on any failed identity.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import asterism
import numpy as np

PAIRS: int = 80
"""Number of independent sibling pairs in the deterministic design."""

SCALE: float = 10.0
"""Positive matrix rescaling whose scientific result must cancel exactly."""

RELATIVE_TOLERANCE: float = 2e-5
"""Measured optimizer and profile-bisection tolerance for equivalent fits."""

ABSOLUTE_TOLERANCE: float = 2e-7
"""Absolute allowance near a zero-valued endpoint."""


def relationship(pairs: int) -> np.ndarray:
    """Return a unit-diagonal additive matrix for independent sibling pairs."""
    matrix: np.ndarray = np.eye(2 * pairs)
    """Started from independent unit marginal genetic variances."""

    for pair in range(pairs):
        first: int = 2 * pair
        """Selected the first sibling's row."""

        second: int = first + 1
        """Selected the matching sibling's row."""

        matrix[first, second] = 0.5
        """Recorded the first directed sibling relationship entry."""

        matrix[second, first] = 0.5
        """Completed the symmetric sibling relationship entry."""
    return matrix


def main() -> int:
    """Run both equivalent fits and fail on any reportable-quantity drift."""
    matrix: np.ndarray = relationship(PAIRS)
    """Built the submitted relationship matrix on its original scale."""

    rescaled_matrix: np.ndarray = SCALE * matrix
    """Represented the identical covariance basis on a tenfold scale."""

    rng: np.random.Generator = np.random.default_rng(4_108)
    """Fixed the participant-free response stream."""

    response: np.ndarray = np.sqrt(0.7) * np.repeat(
        rng.standard_normal(PAIRS), 2
    ) + np.sqrt(0.3) * rng.standard_normal(2 * PAIRS)
    """Drew a response with additive and independent residual contributions."""

    design: np.ndarray = np.ones((2 * PAIRS, 1))
    """Used the same intercept-only mean model for both fits."""

    original_model: asterism.ComponentModel = asterism.ComponentModel([matrix], design)
    """Prepared the public component model on the original scale."""

    rescaled_model: asterism.ComponentModel = asterism.ComponentModel(
        [rescaled_matrix], design
    )
    """Prepared the equivalent model after positive matrix rescaling."""

    original_fit: dict[str, Any] = original_model.fit(response)
    """Fitted the reportable component decomposition on the original scale."""

    rescaled_fit: dict[str, Any] = rescaled_model.fit(response)
    """Fitted the same decomposition on the rescaled basis."""

    original_interval: dict[str, Any] = original_model.interval(response, 0)
    """Profiled the original mean-diagonal component proportion."""

    rescaled_interval: dict[str, Any] = rescaled_model.interval(response, 0)
    """Profiled the same reportable quantity after rescaling."""

    failures: list[str] = []
    """Collected every identity failure for one actionable command result."""

    if original_fit.get("converged") is not True:
        failures.append("original free fit did not converge")
    if rescaled_fit.get("converged") is not True:
        failures.append("rescaled free fit did not converge")
    if original_interval.get("profile_failures") != 0:
        failures.append("original interval contains failed profile evaluations")
    if rescaled_interval.get("profile_failures") != 0:
        failures.append("rescaled interval contains failed profile evaluations")

    comparisons: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "contributions": (
            np.asarray(original_fit["mean_diagonal_component_contributions"]),
            np.asarray(rescaled_fit["mean_diagonal_component_contributions"]),
        ),
        "proportions": (
            np.asarray(original_fit["mean_diagonal_proportions"]),
            np.asarray(rescaled_fit["mean_diagonal_proportions"]),
        ),
        "interval": (
            np.asarray(
                [
                    original_interval["estimate"],
                    original_interval["lower"],
                    original_interval["upper"],
                ]
            ),
            np.asarray(
                [
                    rescaled_interval["estimate"],
                    rescaled_interval["lower"],
                    rescaled_interval["upper"],
                ]
            ),
        ),
    }
    """Collected every scale-invariant reportable identity in one inventory."""

    differences: dict[str, float] = {}
    """Recorded worst absolute differences for the machine-readable output."""

    for label, (original, rescaled) in comparisons.items():
        difference: float = float(np.max(np.abs(original - rescaled)))
        """Measured the largest coordinate drift for this identity."""

        differences[label] = difference
        """Retained this identity's exact observed numerical drift."""

        if not np.allclose(
            original,
            rescaled,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        ):
            failures.append(f"{label} changed after positive matrix rescaling")

    for label, interval, fit in (
        ("original", original_interval, original_fit),
        ("rescaled", rescaled_interval, rescaled_fit),
    ):
        point_difference: float = abs(
            float(interval["estimate"]) - float(fit["mean_diagonal_proportions"][0])
        )
        """Compared the interval point with the public fit's reportable field."""

        if point_difference > ABSOLUTE_TOLERANCE:
            failures.append(f"{label} interval estimates a different quantity than fit")

    record: dict[str, Any] = {
        "check": "component_interval_identity",
        "participant_free": True,
        "people": 2 * PAIRS,
        "largest_family": 2,
        "matrix_scale": SCALE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "worst_absolute_difference": differences,
        "original_interval": original_interval,
        "rescaled_interval": rescaled_interval,
        "passed": not failures,
        "failures": failures,
    }
    """Built the complete participant-free command evidence record."""

    print(json.dumps(record, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
