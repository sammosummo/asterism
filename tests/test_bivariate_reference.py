"""Test the independent bivariate calculation used for fidelity checks."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt


def reference_module() -> ModuleType:
    """Load the standalone bivariate reference without making checks a package."""
    path: Path = Path(__file__).parents[1] / "checks" / "bivariate_reference.py"
    """Located the independent reference implementation in the checks directory."""

    spec: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location("bivariate_reference", path)
    )
    """Created an import specification for that standalone source file."""

    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    """Created the module object governed by the validated specification."""

    spec.loader.exec_module(module)
    """Executed the independent reference implementation in its module object."""
    return module


def test_unbalanced_starting_scales_keep_trait_identity() -> None:
    """Keep each trait's starting scale tied to its observed response values."""
    reference: ModuleType = reference_module()
    """Loaded the independent bivariate calculation under test."""

    observed: npt.NDArray[np.bool_] = np.array(
        [[True, True], [True, False], [False, True]]
    )
    """Described deliberately unbalanced availability across the two traits."""

    response: npt.NDArray[np.float64] = np.array([1.0, 10.0, 3.0, 999.0, 999.0, 30.0])
    """Placed extreme sentinels in response positions marked unobserved."""

    assert reference.trait_scales(response, observed) == [1.0, 100.0]
