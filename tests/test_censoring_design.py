"""Contracts for fixed-instrument simulation designs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from checks.censoring_design import right_censoring_limit

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the scientific simulation commands."""


def test_right_censoring_limit_matches_the_pre_outcome_marginal_share() -> None:
    """Solve a heterogeneous design without drawing or sorting an outcome."""
    means: np.ndarray = np.asarray([-1.0, 0.0, 2.0])
    """Declared heterogeneous latent means before simulation."""

    variances: np.ndarray = np.asarray([1.0, 4.0, 0.25])
    """Declared heterogeneous marginal variances before simulation."""

    limit: float = right_censoring_limit(means, variances, 0.4)
    """Solved the common limit from design facts alone."""

    achieved: float = float(np.mean(norm.sf((limit - means) / np.sqrt(variances))))
    """Recomputed its expected share independently."""

    assert achieved == pytest.approx(0.4, abs=1e-12)


@pytest.mark.parametrize(
    "relative_path",
    [
        "checks/tobit_calibration.py",
        "checks/tobit_coverage.py",
        "checks/tobit_target_design.py",
        "checks/component_separation.py",
        "checks/censored_components_coverage.py",
        "checks/censored_components_target_design.py",
        "checks/censored_components_against_independent_full_fit.py",
        "checks/mixed_bivariate_calibration.py",
        "checks/mixed_bivariate_coverage.py",
        "checks/tobit_against_mcmcglmm.py",
    ],
)
def test_release_simulation_limits_do_not_depend_on_realised_outcomes(
    relative_path: str,
) -> None:
    """Keep order statistics out of every censored release generator."""
    source: str = (ROOT / relative_path).read_text(encoding="utf-8")
    """Read the complete maintained generator source."""

    assert "np.quantile(" not in source
    assert "right_censoring_limit" in source
