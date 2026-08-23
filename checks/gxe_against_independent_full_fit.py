"""Compare both supported continuous-GxE surfaces with an independent full fit.

SciPy optimizes a separately assembled dense Gaussian likelihood in natural
covariance coordinates.  Asterism receives the same participant-free arrays
through ``GxeModel``.  The comparison covers the maximized likelihood and the
surface-independent reportable quantities at a prespecified environment grid.
It does not share Asterism's likelihood, gradients, parameterization, starts, or
optimizer.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import asterism
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaussian_full_fit_reference import DenseFit, fit_covariance

PAIRS: int = 40
"""Independent sibling pairs in each full-fit comparison."""

GRID: np.ndarray = np.asarray([-1.0, 0.0, 1.0])
"""Prespecified reporting environments compared across implementations."""

QUANTITY_TOLERANCE: float = 1e-3
"""Absolute tolerance for heritabilities, variances, and correlations."""

LOGLIK_TOLERANCE: float = 1e-4
"""Absolute tolerance for the maximized ML log likelihood."""


def structure() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a participant-free relationship, environment, and mean design."""
    people: int = 2 * PAIRS
    """Counted people in the fixed sibling-pair roster."""

    relationship: np.ndarray = np.eye(people)
    """Started from unit marginal additive covariance."""

    for pair in range(PAIRS):
        first: int = 2 * pair
        """Selected the first sibling's row."""

        second: int = first + 1
        """Selected the second sibling's row."""

        relationship[first, second] = 0.5
        """Recorded the first directed sibling relationship entry."""

        relationship[second, first] = 0.5
        """Completed the symmetric sibling relationship entry."""

    environment: np.ndarray = np.random.default_rng(20_260_813).uniform(
        -1.5, 1.5, people
    )
    """Fixed environments that vary both within and between pairs."""

    design: np.ndarray = np.column_stack(
        [np.ones(people), environment - environment.mean()]
    )
    """Included an intercept and prespecified centered environment mean effect."""

    return relationship, environment, design


def exponential_covariance(
    parameters: np.ndarray,
    relationship: np.ndarray,
    environment: np.ndarray,
) -> np.ndarray:
    """Assemble the published log-linear exponential surface independently."""
    alpha_g, gamma_g, decay, alpha_e, gamma_e = parameters
    """Named the independent log-linear genetic and residual coordinates."""

    genetic_variance: np.ndarray = np.exp(alpha_g + gamma_g * environment)
    """Evaluated the genetic log-variance function."""

    residual_variance: np.ndarray = np.exp(alpha_e + gamma_e * environment)
    """Evaluated the residual log-variance function."""

    genetic: np.ndarray = np.sqrt(
        np.outer(genetic_variance, genetic_variance)
    ) * np.exp(-decay * np.abs(environment[:, None] - environment[None, :]))
    """Built the positive-definite environmental genetic kernel."""

    covariance: np.ndarray = relationship * genetic
    """Scaled the surface by submitted additive relationships."""

    covariance[np.diag_indices_from(covariance)] += residual_variance
    """Added independent residual variation only to marginal variances."""

    return covariance


def covariance_from_loadings(parameters: np.ndarray) -> np.ndarray:
    """Return a positive-definite two-coordinate covariance from loadings."""
    loading: np.ndarray = np.asarray(
        [
            [np.exp(parameters[0]), 0.0],
            [parameters[1], np.exp(parameters[2])],
        ]
    )
    """Mapped unconstrained natural coordinates to a lower-triangular factor."""

    return loading @ loading.T


def random_regression_covariance(
    parameters: np.ndarray,
    relationship: np.ndarray,
    environment: np.ndarray,
) -> np.ndarray:
    """Assemble the two-coordinate random-regression covariance independently."""
    genetic_block: np.ndarray = covariance_from_loadings(parameters[:3])
    """Built the genetic reaction-norm coefficient covariance."""

    residual_block: np.ndarray = covariance_from_loadings(parameters[3:])
    """Built the residual variance-function coefficient covariance."""

    basis: np.ndarray = np.column_stack([np.ones(environment.size), environment])
    """Evaluated the fixed linear reaction-norm basis."""

    genetic: np.ndarray = basis @ genetic_block @ basis.T
    """Evaluated genetic covariance between every environment pair."""

    residual: np.ndarray = np.einsum("ij,jk,ik->i", basis, residual_block, basis)
    """Evaluated the positive residual variance at each environment."""

    covariance: np.ndarray = relationship * genetic
    """Scaled the genetic surface by submitted relationships."""

    covariance[np.diag_indices_from(covariance)] += residual
    """Added the environment-specific residual variance on the diagonal."""

    return covariance


def summaries(
    surface: str, parameters: np.ndarray, grid: np.ndarray
) -> dict[str, np.ndarray]:
    """Evaluate reference variances, heritabilities, and correlations on a grid."""
    if surface == "exponential":
        alpha_g, gamma_g, decay, alpha_e, gamma_e = parameters
        """Named the optimized exponential-surface coordinates."""

        genetic_variance: np.ndarray = np.exp(alpha_g + gamma_g * grid)
        """Evaluated reference genetic variances on the reporting grid."""

        residual_variance: np.ndarray = np.exp(alpha_e + gamma_e * grid)
        """Evaluated reference residual variances on the reporting grid."""

        correlation: np.ndarray = np.exp(-decay * np.abs(grid[:, None] - grid[None, :]))
        """Evaluated the reference exponential genetic correlations."""
    else:
        genetic_block: np.ndarray = covariance_from_loadings(parameters[:3])
        """Recovered the optimized genetic coefficient covariance."""

        residual_block: np.ndarray = covariance_from_loadings(parameters[3:])
        """Recovered the optimized residual coefficient covariance."""

        basis: np.ndarray = np.column_stack([np.ones(grid.size), grid])
        """Evaluated the reaction-norm basis on the reporting grid."""

        genetic_covariance: np.ndarray = basis @ genetic_block @ basis.T
        """Evaluated the full reference genetic surface."""

        genetic_variance = np.diag(genetic_covariance)
        """Extracted genetic variances from the independent covariance surface."""

        residual_variance = np.einsum("ij,jk,ik->i", basis, residual_block, basis)
        """Evaluated residual variances from their coefficient covariance."""

        denominator: np.ndarray = np.sqrt(np.outer(genetic_variance, genetic_variance))
        """Built the genetic standard-deviation products."""

        correlation = genetic_covariance / denominator
        """Converted reference covariances to the reportable correlation scale."""

    heritability: np.ndarray = genetic_variance / (genetic_variance + residual_variance)
    """Converted both reference surfaces to the common reportable estimand."""

    return {
        "genetic_variance": genetic_variance,
        "residual_variance": residual_variance,
        "heritability": heritability,
        "genetic_correlation": correlation,
    }


def fit_surface(
    surface: str,
    relationship: np.ndarray,
    environment: np.ndarray,
    design: np.ndarray,
) -> dict[str, Any]:
    """Fit one deterministic simulated surface by both independent routes."""
    if surface == "exponential":
        truth: np.ndarray = np.asarray([np.log(0.55), 0.25, 0.35, np.log(0.50), -0.15])
        """Selected an interior log-linear surface with genuine reordering."""

        covariance: Callable[[np.ndarray], np.ndarray] = lambda parameters: (
            exponential_covariance(parameters, relationship, environment)
        )
        """Bound the independent exponential covariance to this fixed design."""

        starts: list[np.ndarray] = [
            truth,
            np.asarray([np.log(0.5), 0.0, 0.1, np.log(0.5), 0.0]),
            np.asarray([np.log(0.7), -0.2, 0.7, np.log(0.4), 0.2]),
        ]
        """Selected prewritten independent starts spanning shape and scale."""

        bounds: list[tuple[float | None, float | None]] = [
            (-5.0, 3.0),
            (-3.0, 3.0),
            (0.0, 3.0),
            (-5.0, 3.0),
            (-3.0, 3.0),
        ]
        """Applied only the published natural-coordinate domain restrictions."""
    else:
        truth = np.asarray(
            [
                np.log(np.sqrt(0.60)),
                0.10 / np.sqrt(0.60),
                np.log(np.sqrt(0.30 - 0.10**2 / 0.60)),
                np.log(np.sqrt(0.50)),
                -0.05 / np.sqrt(0.50),
                np.log(np.sqrt(0.20 - 0.05**2 / 0.50)),
            ]
        )
        """Selected two interior positive-definite coefficient covariances."""

        covariance = lambda parameters: random_regression_covariance(
            parameters,
            relationship,
            environment,
        )
        """Bound the independent random-regression covariance to this design."""

        starts = [
            truth,
            np.asarray([np.log(0.7), 0.0, np.log(0.3), np.log(0.6), 0.0, np.log(0.3)]),
            np.asarray([np.log(0.5), 0.3, np.log(0.5), np.log(0.5), -0.2, np.log(0.4)]),
        ]
        """Selected prewritten positive-definite coefficient-covariance starts."""

        bounds = [(-20.0, 3.0), (-3.0, 3.0), (-20.0, 3.0)] * 2
        """Allowed either covariance block to reach its rank-one boundary."""

    rng: np.random.Generator = np.random.default_rng(
        31_401 if surface == "exponential" else 31_402
    )
    """Fixed a distinct response stream for each supported surface."""

    response: np.ndarray = design @ np.asarray([1.0, 0.2]) + np.linalg.cholesky(
        covariance(truth)
    ) @ rng.standard_normal(design.shape[0])
    """Drew one participant-free response from the exact tested surface."""

    reference: DenseFit = fit_covariance(
        covariance,
        starts,
        bounds,
        design,
        response,
        reml=False,
    )
    """Optimized the independent dense maximum likelihood."""

    model: asterism.GxeModel = asterism.GxeModel(
        relationship, environment, design, surface=surface
    )
    """Built the same supported surface through the public Python interface."""

    fitted: dict[str, Any] = model.fit(response, grid=GRID, reml=False)
    """Fitted Asterism and evaluated the prespecified reporting grid."""

    expected: dict[str, np.ndarray] = summaries(surface, reference.parameters, GRID)
    """Converted the independent natural parameters to reportable quantities."""

    differences: dict[str, float] = {
        name: float(
            np.max(np.abs(np.asarray(fitted[name], dtype=float) - expected_value))
        )
        for name, expected_value in expected.items()
    }
    """Measured worst absolute differences on each common scientific scale."""

    loglik_difference: float = abs(float(fitted["loglik"]) + reference.negative_loglik)
    """Compared maximized likelihoods including the Gaussian constant."""

    failures: list[str] = []
    """Collected convergence and numerical-agreement failures."""

    if fitted.get("converged") is not True:
        failures.append("Asterism free fit did not converge")
    if not reference.converged:
        failures.append(f"independent optimizer did not converge: {reference.message}")
    for name, difference in differences.items():
        if difference > QUANTITY_TOLERANCE:
            failures.append(f"{name} differs by {difference:.3e}")
    if loglik_difference > LOGLIK_TOLERANCE:
        failures.append(f"log likelihood differs by {loglik_difference:.3e}")

    return {
        "surface": surface,
        "people": design.shape[0],
        "largest_family": 2,
        "quantity_tolerance": QUANTITY_TOLERANCE,
        "loglik_tolerance": LOGLIK_TOLERANCE,
        "worst_absolute_differences": differences,
        "loglik_difference": loglik_difference,
        "asterism_converged": fitted.get("converged"),
        "reference_converged": reference.converged,
        "reference_message": reference.message,
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    """Run the independent full fit for both supported continuous-GxE surfaces."""
    relationship, environment, design = structure()
    """Built one fixed design shared by both surface comparisons."""

    results: list[dict[str, Any]] = [
        fit_surface(surface, relationship, environment, design)
        for surface in ("exponential", "random_regression")
    ]
    """Ran both supported surface families and no deferred powered form."""

    failures: list[str] = [
        f"{result['surface']}: {failure}"
        for result in results
        for failure in result["failures"]
    ]
    """Flattened surface-labelled failures for the command exit decision."""

    record: dict[str, Any] = {
        "check": "gxe_against_independent_full_fit",
        "participant_free": True,
        "surfaces": results,
        "passed": not failures,
        "failures": failures,
    }
    """Built the auditable full-fit comparison record."""

    print(json.dumps(record, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
