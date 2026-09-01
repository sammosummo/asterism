"""Generating-covariance contracts for censored several-component coverage."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the standalone scientific command."""


def test_manifest_binds_the_measured_component_coverage_rule() -> None:
    """Keep the completed fixed-instrument campaign as a ready release rule."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the authoritative release contract."""

    analysis: dict[str, object] = next(
        candidate
        for candidate in manifest["analyses"]
        if candidate["id"] == "one_trait_censored_components"
    )
    """Selected the several-component censored analysis."""

    rules: list[dict[str, object]] = [
        rule
        for rule in analysis["pass_rules"]
        if rule["id"] == "censored_components_coverage"
    ]
    """Required one executable coverage rule for the analysis."""

    assert len(rules) == 1
    rule: dict[str, object] = rules[0]
    """Selected the exact rule whose measured state is under test."""

    assert rule["command"] == [
        "checks/censored_components_coverage.py",
        "--replicates",
        "300",
        "--workers",
        "12",
        "--no-write",
    ]
    assert rule["status"] == "ready"
    assert "blocker" not in rule
    assert rule["design_facts"]["measured"] is True


def coverage_module() -> ModuleType:
    """Load the standalone command without running its campaign."""
    path: Path = ROOT / "checks" / "censored_components_coverage.py"
    """Located the coverage command under test."""

    specification: ModuleSpec | None = importlib.util.spec_from_file_location(
        "censored_components_coverage", path
    )
    """Built an import specification without executing the command's main."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the temporary module named by the specification."""

    sys.modules[specification.name] = module
    """Registered the module while its annotations and globals initialise."""

    specification.loader.exec_module(module)
    return module


def dense_factor(
    module: ModuleType,
    model_components: list[npt.NDArray[np.float64]],
    shares: tuple[float, float, float, float],
) -> npt.NDArray[np.float64]:
    """Recreate the retired dense generating factor for a small fixture."""
    rows: int = model_components[0].shape[0]
    """Read the small fixture's complete row count."""

    covariance: npt.NDArray[np.float64] = module.TOTAL_VARIANCE * (
        shares[0] * model_components[0]
        + shares[1] * model_components[1]
        + shares[2] * model_components[2]
        + shares[3] * np.eye(rows)
    )
    """Assembled exactly the covariance used by the retired worker path."""

    return np.linalg.cholesky(covariance + 1e-9 * np.eye(rows))


def test_block_factors_match_the_retired_dense_diagonal_blocks_to_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace only zero off-block storage, not the generating distribution."""
    module: ModuleType = coverage_module()
    """Loaded the coverage design and block-factor helpers."""

    monkeypatch.setattr(module, "VILLAGES", 3)
    structure: dict[str, Any] = module.components()
    """Built a small three-village copy of the exact campaign structure."""

    model_components: list[npt.NDArray[np.float64]] = structure["components"]
    """Retained the same dense components that the fitted model receives."""

    component_blocks: list[npt.NDArray[np.float64]] = module.village_component_blocks(
        model_components
    )
    """Extracted only the exact diagonal blocks needed to generate outcomes."""

    for shares in module.TRUTHS:
        retired: npt.NDArray[np.float64] = dense_factor(
            module,
            model_components,
            shares,
        )
        """Factored the former whole covariance on this small fixture."""

        blockwise: npt.NDArray[np.float64] = module.generating_factors(
            component_blocks,
            shares,
        )
        """Factored the same covariance as three independent village blocks."""

        assert blockwise.shape == (3, module.VILLAGE_ROWS, module.VILLAGE_ROWS)
        for village, factor in enumerate(blockwise):
            start: int = village * module.VILLAGE_ROWS
            """Located this village's diagonal block in the retired factor."""

            stop: int = start + module.VILLAGE_ROWS
            """Located the first row belonging to the next village."""

            np.testing.assert_allclose(
                factor,
                retired[start:stop, start:stop],
                rtol=0.0,
                atol=5e-15,
            )


def test_block_application_preserves_global_rng_order_and_generated_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw one global vector and change its factor product only at round-off."""
    module: ModuleType = coverage_module()
    """Loaded the generating helpers under test."""

    monkeypatch.setattr(module, "VILLAGES", 3)
    structure: dict[str, Any] = module.components()
    """Built a small exact campaign structure for inexpensive dense comparison."""

    model_components: list[npt.NDArray[np.float64]] = structure["components"]
    """Selected the fitted model's unchanged dense covariance components."""

    shares: tuple[float, float, float, float] = module.TRUTHS[2]
    """Selected the interior target truth used by the principal campaign cell."""

    retired: npt.NDArray[np.float64] = dense_factor(
        module,
        model_components,
        shares,
    )
    """Recreated the former whole-design generating factor."""

    blockwise: npt.NDArray[np.float64] = module.generating_factors(
        module.village_component_blocks(model_components),
        shares,
    )
    """Built the replacement stack of exact village factors."""

    dense_generator: np.random.Generator = np.random.default_rng(998_341)
    """Created the reference stream used by the retired dense product."""

    block_generator: np.random.Generator = np.random.default_rng(998_341)
    """Created an identical stream used by the replacement block product."""

    rows: int = structure["design"].shape[0]
    """Read the fixture's complete global row count."""

    dense_normals: npt.NDArray[np.float64] = dense_generator.standard_normal(rows)
    """Drew the former single global standard-normal vector."""

    block_normals: npt.NDArray[np.float64] = block_generator.standard_normal(rows)
    """Drew the replacement's single global standard-normal vector."""

    assert np.array_equal(block_normals, dense_normals)
    dense_values: npt.NDArray[np.float64] = retired @ dense_normals
    """Applied the retired dense factor."""

    block_values: npt.NDArray[np.float64] = module.apply_generating_factors(
        blockwise,
        block_normals,
    )
    """Applied each exact factor to its contiguous part of the same draw."""

    np.testing.assert_allclose(block_values, dense_values, rtol=0.0, atol=5e-15)
    assert np.array_equal(
        block_values + module.TRUE_MEAN >= module.CENSORING_LIMIT,
        dense_values + module.TRUE_MEAN >= module.CENSORING_LIMIT,
    )
    assert block_generator.standard_normal() == dense_generator.standard_normal()


def test_worker_factors_only_twelve_row_block_stacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep all four 4,800-row generating factors out of every worker."""
    module: ModuleType = coverage_module()
    """Loaded worker initialisation with a small village count."""

    monkeypatch.setattr(module, "VILLAGES", 2)
    original_cholesky: Any = module.np.linalg.cholesky
    """Retained NumPy's implementation beneath a shape-recording wrapper."""

    shapes: list[tuple[int, ...]] = []
    """Collected every array shape sent to Cholesky during initialisation."""

    def record_cholesky(
        matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Record the factorisation shape, then use NumPy unchanged."""
        shapes.append(matrix.shape)
        return original_cholesky(matrix)

    monkeypatch.setattr(module.np.linalg, "cholesky", record_cholesky)
    module.start_worker()

    assert shapes == [(2, module.VILLAGE_ROWS, module.VILLAGE_ROWS)] * len(
        module.TRUTHS
    )
    assert [factor.shape for factor in module.STATE["factors"]] == shapes
    assert [component.shape for component in module.STATE["components"]] == [
        (24, 24),
        (24, 24),
        (24, 24),
    ]
