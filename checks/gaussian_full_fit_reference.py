"""Independent dense Gaussian likelihood used by full-fit scientific checks.

This module deliberately shares no numerical likelihood or optimizer code with
Asterism.  Callers supply a covariance construction, while this module profiles
the fixed effects and evaluates ML or invariant REML using NumPy and SciPy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize

LOG_TWO_PI: float = float(np.log(2.0 * np.pi))
"""Natural logarithm of the Gaussian density constant."""


@dataclass(frozen=True)
class DenseFit:
    """Best independently optimized covariance parameters and likelihood."""

    parameters: np.ndarray
    """Natural-coordinate covariance parameters at the best candidate."""

    negative_loglik: float
    """Independently evaluated profiled negative log likelihood."""

    converged: bool
    """Whether SciPy declared its best candidate converged."""

    message: str
    """Raw optimizer conclusion retained for failed release evidence."""


def profiled_negative_loglik(
    covariance: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
    *,
    reml: bool,
) -> float:
    """Evaluate the profiled Gaussian negative log likelihood.

    Args:
        covariance: Positive-definite response covariance.
        design: Full-rank fixed-effect design.
        response: Continuous response vector.
        reml: Whether to add Asterism's invariant restricted-likelihood term.

    Returns:
        Finite profiled negative log likelihood, or infinity outside the domain.
    """
    try:
        factor: tuple[np.ndarray, bool] = cho_factor(
            covariance, lower=True, check_finite=False
        )
        """Factored the independently assembled dense covariance."""

        inverse_response: np.ndarray = cho_solve(factor, response, check_finite=False)
        """Applied the covariance inverse to the response."""

        inverse_design: np.ndarray = cho_solve(factor, design, check_finite=False)
        """Applied the same inverse to every fixed-effect column."""
    except (np.linalg.LinAlgError, ValueError):
        return float("inf")

    weighted_design: np.ndarray = design.T @ inverse_design
    """Built the generalized least-squares normal matrix."""

    weighted_response: np.ndarray = design.T @ inverse_response
    """Built the matching generalized least-squares response vector."""

    try:
        effects: np.ndarray = np.linalg.solve(weighted_design, weighted_response)
        """Profiled the fixed effects independently of Asterism."""
    except np.linalg.LinAlgError:
        return float("inf")

    residual: np.ndarray = response - design @ effects
    """Computed the profiled residual vector."""

    quadratic: float = float(residual @ cho_solve(factor, residual, check_finite=False))
    """Computed the generalized residual sum of squares."""

    cholesky: np.ndarray = np.tril(factor[0])
    """Selected the valid lower triangle returned by SciPy."""

    diagonal: np.ndarray = np.diag(cholesky)
    """Read the determinant from the Cholesky diagonal."""

    if np.any(diagonal <= 0.0) or not np.isfinite(quadratic):
        return float("inf")
    logdet: float = 2.0 * float(np.log(diagonal).sum())
    """Computed the covariance log determinant."""

    rows: int = response.size
    """Counted Gaussian observation rows."""

    value: float = 0.5 * (rows * LOG_TWO_PI + logdet + quadratic)
    """Evaluated the profiled maximum-likelihood objective."""

    if reml:
        sign_weighted, logdet_weighted = np.linalg.slogdet(weighted_design)
        """Measured the covariance-weighted design determinant."""

        sign_plain, logdet_plain = np.linalg.slogdet(design.T @ design)
        """Measured the plain determinant used for basis invariance."""

        if sign_weighted <= 0.0 or sign_plain <= 0.0:
            return float("inf")
        columns: int = design.shape[1]
        """Counted profiled fixed effects."""

        value += (
            0.5 * (float(logdet_weighted) - float(logdet_plain))
            - 0.5 * columns * LOG_TWO_PI
        )
        """Added Asterism's invariant restricted-likelihood adjustment."""

    return value if np.isfinite(value) else float("inf")


def fit_covariance(
    covariance: Callable[[np.ndarray], np.ndarray],
    starts: Sequence[np.ndarray],
    bounds: Sequence[tuple[float | None, float | None]],
    design: np.ndarray,
    response: np.ndarray,
    *,
    reml: bool,
) -> DenseFit:
    """Optimize an independently assembled covariance from several fixed starts.

    Args:
        covariance: Function mapping natural parameters to a dense covariance.
        starts: Prewritten deterministic optimizer starts.
        bounds: Natural-coordinate box constraints.
        design: Fixed-effect design.
        response: Simulated response.
        reml: Whether to optimize ML or invariant REML.

    Returns:
        Best finite SciPy fit, including its optimizer verdict.
    """

    def objective(parameters: np.ndarray) -> float:
        """Assemble and score one natural-parameter covariance."""
        try:
            matrix: np.ndarray = covariance(parameters)
            """Built the candidate covariance through the independent formula."""
        except (FloatingPointError, OverflowError, ValueError):
            return float("inf")
        return profiled_negative_loglik(matrix, design, response, reml=reml)

    results: list[OptimizeResult] = [
        minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            jac="3-point",
            bounds=bounds,
            options={
                "maxiter": 10_000,
                "maxls": 100,
                "ftol": 1e-15,
                "gtol": 1e-10,
                "finite_diff_rel_step": 1e-5,
            },
        )
        for start in starts
    ]
    """Ran SciPy's optimizer from every prespecified independent start."""

    finite: list[OptimizeResult] = [
        result for result in results if np.isfinite(float(result.fun))
    ]
    """Discarded only candidates outside the covariance domain."""

    if not finite:
        return DenseFit(
            parameters=np.asarray(starts[0], dtype=float),
            negative_loglik=float("inf"),
            converged=False,
            message="no finite independent optimizer candidate",
        )
    best: OptimizeResult = min(finite, key=lambda result: float(result.fun))
    """Selected the globally best candidate across fixed starts."""

    return DenseFit(
        parameters=np.asarray(best.x, dtype=float),
        negative_loglik=float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
    )
