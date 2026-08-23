"""Compare discrete GxE with a separately optimized dense Gaussian fit.

The reference fit uses a genetic Cholesky factor and log residual variances,
profiles fixed effects with direct NumPy solves, and asks SciPy's derivative-free
Powell optimizer to maximize the full ML likelihood.  Asterism receives the
same participant-free arrays only through ``DiscreteGxeModel.fit``.  The two
routes therefore share neither parameterization, optimizer nor likelihood code.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import asterism
import numpy as np
from scipy.optimize import OptimizeResult, minimize

FAMILIES: int = 30
"""Number of independent four-sibling families in the comparison."""

QUANTITY_TOLERANCE: float = 2e-3
"""Maximum absolute difference on each common public quantity."""

LOGLIK_TOLERANCE: float = 2e-4
"""Maximum absolute difference in the maximized full ML log likelihood."""


@dataclass(frozen=True)
class IndependentFit:
    """Hold one independently optimized dense full-likelihood fit."""

    parameters: np.ndarray
    """Natural reference coordinates at the best derivative-free candidate."""

    negative_loglik: float
    """Profiled dense negative log likelihood at that candidate."""

    effects: np.ndarray
    """Generalized least-squares fixed-effect estimates at the optimum."""

    converged: bool
    """Whether SciPy accepted the best finite Powell search."""

    message: str
    """Raw independent optimizer conclusion retained for failures."""


def covariance_parameters(
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map independent natural coordinates to genetic and residual covariance.

    Args:
        parameters: Two log Cholesky diagonals, one free loading, and two log
            residual variances.

    Returns:
        Positive-definite two-group genetic covariance and positive residual
        variances.
    """
    loading: np.ndarray = np.asarray(
        [
            [np.exp(parameters[0]), 0.0],
            [parameters[1], np.exp(parameters[2])],
        ]
    )
    """Constructed a lower-triangular genetic factor independently."""

    genetic: np.ndarray = loading @ loading.T
    """Converted the factor to the two-group genetic covariance."""

    residual: np.ndarray = np.exp(parameters[3:5])
    """Mapped two log variances to positive residual variances."""

    return genetic, residual


def dense_covariance(
    parameters: np.ndarray,
    relationship: np.ndarray,
    group_index: np.ndarray,
) -> np.ndarray:
    """Assemble the discrete-GxE covariance without Asterism code.

    Args:
        parameters: Independent genetic-factor and residual coordinates.
        relationship: Participant-free additive relationship matrix.
        group_index: Zero-based environment index for each row.

    Returns:
        Dense positive-definite response covariance.
    """
    genetic, residual = covariance_parameters(parameters)
    """Evaluated the independent two-group covariance parameters."""

    covariance: np.ndarray = (
        relationship * genetic[group_index[:, None], group_index[None, :]]
    )
    """Selected the appropriate genetic covariance for every pair."""

    covariance[np.diag_indices_from(covariance)] += residual[group_index]
    """Added group-specific independent residual variance on the diagonal."""

    return covariance


def profiled_full_ml(
    covariance: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Evaluate the independently profiled full Gaussian ML likelihood.

    Args:
        covariance: Dense candidate response covariance.
        design: Full-rank fixed-effect design.
        response: Continuous response vector.

    Returns:
        Negative full log likelihood and profiled fixed-effect estimates.  An
        invalid covariance returns infinite objective and NaN effects.
    """
    columns: int = design.shape[1]
    """Counted fixed effects for a stable invalid-candidate result."""

    try:
        cholesky: np.ndarray = np.linalg.cholesky(covariance)
        """Factored the independently assembled dense covariance."""

        inverse_design: np.ndarray = np.linalg.solve(
            covariance,
            design,
        )
        """Applied the dense covariance inverse to every design column."""

        inverse_response: np.ndarray = np.linalg.solve(
            covariance,
            response,
        )
        """Applied the same independent inverse to the response."""

        normal: np.ndarray = design.T @ inverse_design
        """Built generalized least-squares normal equations."""

        effects: np.ndarray = np.linalg.solve(normal, design.T @ inverse_response)
        """Profiled the fixed effects separately from Asterism."""

        residual: np.ndarray = response - design @ effects
        """Computed the response residual at the profiled mean."""

        quadratic: float = float(residual @ np.linalg.solve(covariance, residual))
        """Evaluated the generalized residual sum of squares."""

        logdet: float = 2.0 * float(np.log(np.diag(cholesky)).sum())
        """Read the covariance determinant from the dense Cholesky factor."""
    except np.linalg.LinAlgError:
        return float("inf"), np.full(columns, np.nan)

    rows: int = response.size
    """Counted observations contributing Gaussian density constants."""

    negative_loglik: float = 0.5 * (rows * np.log(2.0 * np.pi) + logdet + quadratic)
    """Evaluated full maximum likelihood including its normalizing constant."""

    if not np.isfinite(negative_loglik):
        return float("inf"), np.full(columns, np.nan)
    return float(negative_loglik), effects


def fit_independent(
    relationship: np.ndarray,
    group_index: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
    starts: tuple[np.ndarray, ...],
) -> IndependentFit:
    """Optimize the dense reference with derivative-free Powell searches.

    Args:
        relationship: Participant-free additive relationship matrix.
        group_index: Zero-based environment index for each row.
        design: Full-rank fixed-effect design.
        response: Deterministically simulated response.
        starts: Prespecified natural-coordinate starting points.

    Returns:
        Best finite independent candidate across every fixed start.
    """

    def objective(parameters: np.ndarray) -> float:
        """Return the dense profiled objective at one reference candidate."""
        covariance: np.ndarray = dense_covariance(
            parameters,
            relationship,
            group_index,
        )
        """Assembled the candidate without calling Asterism."""

        value, _ = profiled_full_ml(covariance, design, response)
        """Profiled the full likelihood using independent linear algebra."""

        return value

    bounds: tuple[tuple[float, float], ...] = (
        (-5.0, 2.0),
        (-2.0, 2.0),
        (-5.0, 2.0),
        (-5.0, 2.0),
        (-5.0, 2.0),
    )
    """Kept Powell inside broad positive-variance and loading coordinates."""

    results: list[OptimizeResult] = [
        minimize(
            objective,
            start,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 4_000, "xtol": 1e-10, "ftol": 1e-12},
        )
        for start in starts
    ]
    """Ran independent derivative-free searches from every prespecified start."""

    finite: list[OptimizeResult] = [
        result for result in results if np.isfinite(float(result.fun))
    ]
    """Retained every candidate with an evaluable dense likelihood."""

    if not finite:
        return IndependentFit(
            parameters=starts[0],
            negative_loglik=float("inf"),
            effects=np.full(design.shape[1], np.nan),
            converged=False,
            message="no finite derivative-free candidate",
        )

    best: OptimizeResult = min(finite, key=lambda result: float(result.fun))
    """Selected the best full likelihood independently of success flags."""

    covariance: np.ndarray = dense_covariance(best.x, relationship, group_index)
    """Reassembled the selected optimum for fixed-effect profiling."""

    negative_loglik, effects = profiled_full_ml(covariance, design, response)
    """Recomputed the final objective and mean estimates from exact coordinates."""

    return IndependentFit(
        parameters=np.asarray(best.x, dtype=float),
        negative_loglik=negative_loglik,
        effects=effects,
        converged=bool(best.success),
        message=str(best.message),
    )


def comparison_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build one fixed participant-free negative-correlation comparison.

    Returns:
        Relationship, environment labels, zero-based group indices, fixed
        design, and deterministic response.
    """
    people: int = 4 * FAMILIES
    """Counted rows in the fixed four-sibling roster."""

    relationship: np.ndarray = np.zeros((people, people), dtype=float)
    """Allocated independent family blocks."""

    for family in range(FAMILIES):
        start: int = 4 * family
        """Located the first sibling in this family."""

        stop: int = start + 4
        """Located the exclusive family-block endpoint."""

        relationship[start:stop, start:stop] = 0.5
        """Assigned ordinary full-sibling off-diagonal relationships."""

        relationship[np.arange(start, stop), np.arange(start, stop)] = 1.0
        """Restored unit additive variance on every diagonal."""

    group_index: np.ndarray = np.tile(np.asarray([0, 0, 1, 1]), FAMILIES)
    """Placed both environment groups inside every related family."""

    environment: np.ndarray = np.where(group_index == 0, 10.0, 20.0)
    """Used noncanonical labels to exercise the public level mapping."""

    covariate: np.ndarray = np.random.default_rng(20_260_821).normal(size=people)
    """Drew one participant-free fixed covariate from a dedicated stream."""

    covariate = (covariate - covariate.mean()) / covariate.std()
    """Standardized the fixed covariate before constructing the mean design."""

    design: np.ndarray = np.column_stack(
        [np.ones(people), covariate, group_index.astype(float)]
    )
    """Included an intercept, continuous covariate, and group mean effect."""

    truth_genetic: np.ndarray = np.asarray([[0.65, -0.18], [-0.18, 0.42]])
    """Selected unequal genetic variances and a negative interior covariance."""

    truth_residual: np.ndarray = np.asarray([0.38, 0.62])
    """Selected unequal positive group-specific residual variances."""

    loading: np.ndarray = np.linalg.cholesky(truth_genetic)
    """Converted the simulation truth to reference coordinates only for drawing."""

    truth_parameters: np.ndarray = np.asarray(
        [
            np.log(loading[0, 0]),
            loading[1, 0],
            np.log(loading[1, 1]),
            np.log(truth_residual[0]),
            np.log(truth_residual[1]),
        ]
    )
    """Encoded the fixed generating covariance in independent coordinates."""

    covariance: np.ndarray = dense_covariance(
        truth_parameters,
        relationship,
        group_index,
    )
    """Built the exact covariance used to draw the response."""

    effects: np.ndarray = np.asarray([1.2, 0.35, -0.25])
    """Selected fixed nonzero mean effects for independent profiling."""

    response: np.ndarray = design @ effects + np.linalg.cholesky(
        covariance
    ) @ np.random.default_rng(31_415).standard_normal(people)
    """Drew one deterministic participant-free Gaussian response."""

    return relationship, environment, group_index, design, response


def reference_summaries(fit: IndependentFit) -> dict[str, np.ndarray | float]:
    """Convert independent coordinates to public scientific quantities.

    Args:
        fit: Independently optimized dense fit.

    Returns:
        Genetic and residual variances, heritabilities, correlation, and fixed
        effects on the same scales as the public fit record.
    """
    genetic, residual = covariance_parameters(fit.parameters)
    """Recovered the independently optimized covariance components."""

    genetic_variance: np.ndarray = np.diag(genetic)
    """Read group-specific genetic variances from the reference block."""

    heritability: np.ndarray = genetic_variance / (genetic_variance + residual)
    """Converted the independent marginal variances to heritabilities."""

    correlation: float = float(
        genetic[0, 1] / np.sqrt(genetic_variance[0] * genetic_variance[1])
    )
    """Converted the independent cross covariance to genetic correlation."""

    return {
        "genetic_variance": genetic_variance,
        "residual_variance": residual,
        "heritability": heritability,
        "genetic_correlation": correlation,
        "fixed_effects": fit.effects,
    }


def run_comparison() -> dict[str, Any]:
    """Fit one deterministic problem through both independent public routes.

    Returns:
        Machine-readable convergence, tolerance, difference, and pass evidence.
    """
    relationship, environment, group_index, design, response = comparison_problem()
    """Built the shared participant-free comparison arrays."""

    truth_genetic: np.ndarray = np.asarray([[0.65, -0.18], [-0.18, 0.42]])
    """Restated the generating block solely to build a fixed reference start."""

    truth_loading: np.ndarray = np.linalg.cholesky(truth_genetic)
    """Mapped the generating block to independent optimizer coordinates."""

    starts: tuple[np.ndarray, ...] = (
        np.asarray(
            [
                np.log(truth_loading[0, 0]),
                truth_loading[1, 0],
                np.log(truth_loading[1, 1]),
                np.log(0.38),
                np.log(0.62),
            ]
        ),
        np.asarray([np.log(0.7), 0.0, np.log(0.7), np.log(0.5), np.log(0.5)]),
        np.asarray([np.log(0.5), 0.3, np.log(0.8), np.log(0.8), np.log(0.3)]),
    )
    """Selected fixed starts spanning correlation signs and variance scales."""

    reference: IndependentFit = fit_independent(
        relationship,
        group_index,
        design,
        response,
        starts,
    )
    """Ran the separately parameterized derivative-free dense fit."""

    model: asterism.DiscreteGxeModel = asterism.DiscreteGxeModel(
        relationship,
        environment,
        design,
        levels=(10.0, 20.0),
    )
    """Constructed the corresponding model through the public Python interface."""

    fitted: dict[str, Any] = model.fit(response, reml=False)
    """Requested the public full maximum-likelihood fit."""

    expected: dict[str, np.ndarray | float] = reference_summaries(reference)
    """Mapped independent parameters to public scientific scales."""

    observed: dict[str, np.ndarray | float] = {
        "genetic_variance": np.asarray(fitted["genetic_variance"], dtype=float),
        "residual_variance": np.asarray(fitted["residual_variance"], dtype=float),
        "heritability": np.asarray(fitted["heritability"], dtype=float),
        "genetic_correlation": float(fitted["genetic_correlation"]),
        "fixed_effects": np.asarray(
            [effect["estimate"] for effect in fitted["fixed_effects"]],
            dtype=float,
        ),
    }
    """Selected only quantities independently recoverable from the dense fit."""

    differences: dict[str, float] = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(observed_value, dtype=float)
                    - np.asarray(expected[name], dtype=float)
                )
            )
        )
        for name, observed_value in observed.items()
    }
    """Measured worst absolute disagreement on each public scientific scale."""

    loglik_difference: float = abs(float(fitted["loglik"]) + reference.negative_loglik)
    """Compared maximized full likelihoods including Gaussian constants."""

    failures: list[str] = []
    """Collected every convergence and prewritten numerical failure."""

    if fitted.get("converged") is not True:
        failures.append("Asterism public full fit did not converge")
    if not reference.converged:
        failures.append(f"independent Powell fit did not converge: {reference.message}")
    for name, difference in differences.items():
        if difference > QUANTITY_TOLERANCE:
            failures.append(f"{name} differs by {difference:.3e}")
    if loglik_difference > LOGLIK_TOLERANCE:
        failures.append(f"log likelihood differs by {loglik_difference:.3e}")

    truth_correlation: float = float(
        truth_genetic[0, 1] / np.sqrt(truth_genetic[0, 0] * truth_genetic[1, 1])
    )
    """Recorded that the sentinel exercises a negative genetic correlation."""

    return {
        "check": "discrete_gxe_against_independent_full_fit",
        "participant_free": True,
        "public_interface": "asterism.DiscreteGxeModel.fit",
        "independent_parameterisation": "genetic_cholesky_and_log_residual",
        "independent_optimizer": "scipy_powell_derivative_free",
        "independent_likelihood": "dense_profiled_full_gaussian_ml",
        "people": response.size,
        "families": FAMILIES,
        "largest_family": 4,
        "truth_genetic_correlation": truth_correlation,
        "quantity_tolerance": QUANTITY_TOLERANCE,
        "loglik_tolerance": LOGLIK_TOLERANCE,
        "absolute_differences": differences,
        "loglik_difference": loglik_difference,
        "asterism_converged": fitted.get("converged"),
        "reference_converged": reference.converged,
        "reference_message": reference.message,
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    """Run the independent discrete-GxE full-fit release comparison."""
    record: dict[str, Any] = run_comparison()
    """Ran both full fits before emitting one auditable decision record."""

    print(json.dumps(record, indent=2))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
