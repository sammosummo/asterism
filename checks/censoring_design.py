"""Fixed instrument limits for participant-free censoring simulations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.stats import norm


def right_censoring_limit(
    mean: npt.ArrayLike,
    marginal_variance: npt.ArrayLike,
    expected_share: float,
) -> float:
    """Solve one predeclared limit for an expected marginal censored share.

    ``mean`` and ``marginal_variance`` describe the latent Gaussian rows before
    any outcome is drawn. Correlation changes the realised count's spread but
    not its marginal expectation, so no covariance factor or response belongs
    in this calculation.
    """
    means: npt.NDArray[np.float64] = np.atleast_1d(np.asarray(mean, dtype=float))
    """Normalised scalar or row-specific means to one dimension."""

    variances: npt.NDArray[np.float64] = np.broadcast_to(
        np.asarray(marginal_variance, dtype=float), means.shape
    )
    """Put scalar and row-specific design facts on one validated shape."""

    if (
        means.ndim != 1
        or means.size == 0
        or not np.isfinite(means).all()
        or not np.isfinite(variances).all()
        or not (variances > 0.0).all()
        or not 0.0 < expected_share < 1.0
    ):
        raise ValueError("CENSORING_DESIGN_COORDINATES_INVALID")

    standard_deviations: npt.NDArray[np.float64] = np.sqrt(variances)
    """Converted marginal variances to the scale of each latent row."""

    def difference(candidate: float) -> float:
        expected: float = float(
            np.mean(norm.sf((candidate - means) / standard_deviations))
        )
        """Computed the expected marginal share above this candidate limit."""

        return expected - expected_share

    lower: float = float(np.min(means - 12.0 * standard_deviations))
    """Placed the lower bracket in the remote left tail."""

    upper: float = float(np.max(means + 12.0 * standard_deviations))
    """Bracketed normal tails far past any requested practical censoring share."""

    return float(brentq(difference, lower, upper, xtol=1e-13, rtol=1e-14))
