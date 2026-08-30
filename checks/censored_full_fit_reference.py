"""Independent dense censored likelihood used by the several-component full fit.

This module shares no numerical likelihood, parameterisation, starting values or
optimiser with Asterism. The censored likelihood is assembled here from its
definition: a block contributes the Gaussian density of the rows an instrument
measured, times the probability that the rows it did not measure lie beyond
their limits, conditioned on the measured ones.

That probability is a multivariate normal rectangle. One censored row uses
SciPy's ``log_ndtr``, which is accurate far into the tail. Two or more use
SciPy's own Genz implementation in ``multivariate_normal.cdf``, which is a
different algorithm from the sequential truncation Asterism applies. Passing a
fixed generator makes it reproducible to the last bit, so the surface a
finite-difference optimiser walks does not move under it.

**Maximum likelihood only.** A censored row has no residual to project onto the
null space of the design, so the fixed effects are optimised here rather than
profiled out, and there is no restricted likelihood to compare against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal

LOG_TWO_PI: float = float(np.log(2.0 * np.pi))
"""Natural logarithm of the Gaussian density constant."""

MAXIMUM_POINTS: int = 200_000
"""Integration points allowed before the rectangle probability gives up."""

BARRIER: float = 1e8
"""Finite stand-in for a likelihood the covariance cannot support.

Returning infinity here is what the Gaussian full-fit reference does, and it
can afford to: it profiles the fixed effects away, so every parameter it
searches is a variance with a positive lower bound. This model searches the
fixed effects too, and its residual can be driven to nought, where two records
of one person become the same number and the covariance stops being usable.
An infinity there makes the difference of two evaluations `inf - inf`, which is
not a number, so the gradient is not a number either and L-BFGS-B stops at once
declaring success. A large finite value instead lets the line search back off.
"""

ABSOLUTE_ACCURACY: float = 1e-14
"""Absolute accuracy asked of the rectangle probability."""

RELATIVE_ACCURACY: float = 1e-11
"""Relative accuracy asked of it, which is what taking a logarithm needs.

An absolute tolerance alone is worthless in the tail: an error of 1e-10 on a
probability of 1e-8 is one part in a hundred, and its logarithm is wrong in the
second decimal. Asking for relative accuracy instead held the agreement with
Asterism at 1e-9 where an absolute rule alone left it at 2e-2.
"""


@dataclass(frozen=True)
class DenseCensoredFit:
    """Best independently optimised parameters and censored likelihood."""

    variances: np.ndarray
    """Component coefficients, the residual last."""

    effects: np.ndarray
    """Fixed effects, optimised rather than profiled."""

    negative_loglik: float
    """Independently evaluated negative log likelihood."""

    converged: bool
    """Whether SciPy declared its best candidate converged."""

    message: str
    """Raw optimiser conclusion retained for failed release evidence."""


def rectangle_log_probability(
    mean: np.ndarray,
    covariance: np.ndarray,
    limit: np.ndarray,
    *,
    seed: int,
) -> float:
    """Return the log probability that every coordinate lies at or above a limit.

    Args:
        mean: Conditional mean of the unmeasured rows.
        covariance: Conditional covariance of the unmeasured rows.
        limit: Lower limit each coordinate must exceed.
        seed: Fixed generator seed making the integration reproducible.

    Returns:
        Natural logarithm of the rectangle probability, or negative infinity
        where the probability underflowed.
    """
    if limit.size == 1:
        deviation: float = float(
            (mean[0] - limit[0]) / np.sqrt(max(float(covariance[0, 0]), 0.0))
        )
        """Standardised the single limit against its conditional spread."""

        return float(log_ndtr(deviation))
    probability: float = float(
        multivariate_normal.cdf(
            -limit,
            mean=-mean,
            cov=covariance,
            maxpts=MAXIMUM_POINTS,
            abseps=ABSOLUTE_ACCURACY,
            releps=RELATIVE_ACCURACY,
            rng=np.random.default_rng(seed),
        )
    )
    """Integrated the rectangle by negating it into a lower orthant."""

    if not np.isfinite(probability) or probability <= 0.0:
        return float("-inf")
    return float(np.log(probability))


def negative_loglik(
    covariance: np.ndarray,
    design: np.ndarray,
    value: np.ndarray,
    censoring: np.ndarray,
    limit: np.ndarray,
    effects: np.ndarray,
    blocks: Sequence[np.ndarray],
    *,
    seed: int,
) -> float:
    """Evaluate the censored negative log likelihood over independent blocks.

    The likelihood factorises over blocks of the covariance, which is the sum of
    the components and not any one of them. Callers supply that partition.

    Args:
        covariance: Positive-definite covariance over every row.
        design: Fixed-effect design, one row per record.
        value: Measured values, unread where a row is censored.
        censoring: One where a row is censored, nought where measured.
        limit: The limit each row was measured against.
        effects: Fixed effects multiplying the design.
        blocks: Row indices of each independent block.
        seed: Fixed generator seed making the integration reproducible.

    Returns:
        Finite negative log likelihood, or infinity outside the domain.
    """
    mean: np.ndarray = design @ effects
    """Applied the fixed effects once for every row."""

    total: float = 0.0
    """Accumulated the log likelihood block by block."""

    for block in blocks:
        censored: np.ndarray = block[censoring[block] == 1]
        """Selected the rows the instrument did not measure."""

        measured: np.ndarray = block[censoring[block] == 0]
        """Selected the rows it did."""

        conditional_mean: np.ndarray = mean[censored]
        """Started the unmeasured rows at their unconditional mean."""

        conditional_covariance: np.ndarray = covariance[np.ix_(censored, censored)]
        """Started them at their unconditional covariance."""

        if measured.size:
            residual: np.ndarray = value[measured] - mean[measured]
            """Took the measured rows away from what the mean predicts."""

            try:
                factor: tuple[np.ndarray, bool] = cho_factor(
                    covariance[np.ix_(measured, measured)],
                    lower=True,
                    check_finite=False,
                )
                """Factored the measured block of the candidate covariance."""
            except (LinAlgError, ValueError):
                return float("inf")
            weighted: np.ndarray = cho_solve(factor, residual, check_finite=False)
            """Solved once for the density and again for the conditioning."""

            logdet: float = 2.0 * float(np.log(np.diag(factor[0])).sum())
            """Read the determinant off the factor rather than recomputing it."""

            total -= 0.5 * (
                measured.size * LOG_TWO_PI + logdet + float(residual @ weighted)
            )
            """Added the Gaussian density of everything that was measured."""

            if censored.size:
                cross: np.ndarray = covariance[np.ix_(censored, measured)]
                """Read how the unmeasured rows covary with the measured ones."""

                conditional_mean = conditional_mean + cross @ weighted
                """Moved the unmeasured rows towards what the measured ones say."""

                conditional_covariance = conditional_covariance - cross @ cho_solve(
                    factor, cross.T, check_finite=False
                )
                """Took away the part the measured rows already explain."""
        if censored.size:
            total += rectangle_log_probability(
                conditional_mean,
                conditional_covariance,
                limit[censored],
                seed=seed,
            )
            """Added the probability the unmeasured rows lie beyond their limits."""
    return -total if np.isfinite(total) else float("inf")


def fit_censored(
    covariance: Callable[[np.ndarray], np.ndarray],
    components: int,
    starts: Sequence[np.ndarray],
    bounds: Sequence[tuple[float | None, float | None]],
    design: np.ndarray,
    value: np.ndarray,
    censoring: np.ndarray,
    limit: np.ndarray,
    blocks: Sequence[np.ndarray],
    *,
    seed: int,
) -> DenseCensoredFit:
    """Optimise an independently assembled censored likelihood from fixed starts.

    Args:
        covariance: Maps the leading variance parameters to a dense covariance.
        components: How many leading parameters are variances, the residual last.
        starts: Prewritten deterministic optimiser starts over every parameter.
        bounds: Box constraints over every parameter.
        design: Fixed-effect design.
        value: Measured values, unread where a row is censored.
        censoring: One where a row is censored, nought where measured.
        limit: The limit each row was measured against.
        blocks: Row indices of each independent block.
        seed: Fixed generator seed making the integration reproducible.

    Returns:
        Best finite SciPy fit, including its optimiser verdict.

    Note:
        SciPy chooses its own finite-difference step. A fixed relative step,
        which the Gaussian full-fit reference can afford because it profiles the
        fixed effects away, takes no step at all for a fixed effect starting at
        nought: the two points coincide, the difference is nought over nought,
        and the optimiser stops on the spot declaring success.
    """

    first: np.ndarray = covariance(np.asarray(starts[0][:components], dtype=float))
    """Built one covariance so its sparsity can be checked before any fitting."""

    membership: np.ndarray = np.empty(first.shape[0], dtype=np.int64)
    """Recorded which block each row was declared to belong to."""

    for index, block in enumerate(blocks):
        membership[block] = index
        """Stamped this block's number on every row it claims."""
    """Recorded which block each row was declared to belong to."""

    outside: np.ndarray = np.abs(first)[membership[:, None] != membership[None, :]]
    """Read every covariance entry joining two rows in different blocks."""

    if outside.size and float(outside.max()) > 1e-12 * float(np.abs(first).max()):
        raise ValueError("BLOCKS_DO_NOT_SEPARATE_THE_COVARIANCE")
    """Refused a partition the likelihood does not actually factorise over.

    Splitting the likelihood over blocks is exact only where the covariance --
    the sum of the components, not any one of them -- is nought between them.
    Get the partition wrong and every number that follows is quietly wrong, so
    this is checked once against the structure rather than trusted.
    """

    def objective(parameters: np.ndarray) -> float:
        """Assemble and score one candidate."""
        try:
            matrix: np.ndarray = covariance(parameters[:components])
            """Built the candidate covariance through the independent formula."""
        except (FloatingPointError, OverflowError, ValueError):
            return float("inf")
        scored: float = negative_loglik(
            matrix,
            design,
            value,
            censoring,
            limit,
            parameters[components:],
            blocks,
            seed=seed,
        )
        """Scored the candidate through the independent likelihood."""

        return BARRIER if not np.isfinite(scored) else scored

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
                "ftol": 1e-14,
                "gtol": 1e-9,
            },
        )
        for start in starts
    ]
    """Ran SciPy's optimiser from every prespecified independent start."""

    finite: list[OptimizeResult] = [
        result for result in results if float(result.fun) < BARRIER
    ]
    """Discarded only candidates that ended against the barrier."""

    if not finite:
        return DenseCensoredFit(
            variances=np.asarray(starts[0][:components], dtype=float),
            effects=np.asarray(starts[0][components:], dtype=float),
            negative_loglik=float("inf"),
            converged=False,
            message="no finite independent optimiser candidate",
        )
    best: OptimizeResult = min(finite, key=lambda result: float(result.fun))
    """Selected the globally best candidate across fixed starts."""

    return DenseCensoredFit(
        variances=np.asarray(best.x[:components], dtype=float),
        effects=np.asarray(best.x[components:], dtype=float),
        negative_loglik=float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
    )
