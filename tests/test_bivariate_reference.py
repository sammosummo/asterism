from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def reference_module():
    path = Path(__file__).parents[1] / "checks" / "bivariate_reference.py"
    spec = importlib.util.spec_from_file_location("bivariate_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unbalanced_starting_scales_keep_trait_identity() -> None:
    reference = reference_module()
    observed = np.array([[True, True], [True, False], [False, True]])
    response = np.array([1.0, 10.0, 3.0, 999.0, 999.0, 30.0])

    assert reference.trait_scales(response, observed) == [1.0, 100.0]
