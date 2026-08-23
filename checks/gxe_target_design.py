"""Qualify both supported continuous-GxE surfaces on a values-free target design.

The target is a pedigree-scale and design-identity stress envelope. It matches
only reviewed aggregate component sizes and the already-qualified synthetic
environment range; it does not reconstruct participant ages, relationship
coefficients, trait missingness, or outcomes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import asterism
import numpy as np
import numpy.typing as npt
from gxe_against_independent_full_fit import summaries as reference_summaries
from one_trait_coverage import SimulationEnvelope, fixture_sha256, target_envelope
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import beta

LOG_TWO_PI: float = float(np.log(2.0 * np.pi))
"""Stored the Gaussian density constant for the independent likelihood."""

QUANTITY_TOLERANCE: float = 1e-3
"""Reused the prewritten independent full-fit tolerance on common quantities."""

LOGLIK_TOLERANCE: float = 1e-4
"""Reused the prewritten independent full-fit tolerance on maximized likelihood."""

MEAN_COEFFICIENTS: npt.NDArray[np.float64] = np.asarray(
    [2.0, 0.30, -0.10, 0.50, 0.05, -0.02]
)
"""Fixed generating effects for the tracked six-column mean design."""

NULL_BASE_SEED: int = 20_260_821
"""Based deterministic flat-null outcome streams independently of worker order."""

EXPECTED_TEST_RULES: dict[str, str] = {
    "correlation": "mixture_50_50",
    "interaction": "half_chi2_1_half_chi2_2",
}
"""Pinned the two nonstandard public references judged at the target design."""

MINIMUM_SMOKE_REPLICATES: int = 100
"""Required enough target null draws for a meaningful reduced qualification."""


@dataclass(frozen=True)
class GxeTargetEnvelope:
    """Hold the participant-free pedigree, environment, and fixed design.

    Attributes:
        relationship: Structure-matched synthetic additive relationship.
        environment: Bounded synthetic standardised-age coordinate.
        fixed_effects: Six-column age-by-sex mean design.
        component_sizes: Reviewed component-size histogram in row order.
        reporting_grid: Prespecified environments for common summaries.
        fixture_sha256: Exact GxE target-fixture identity.
    """

    relationship: npt.NDArray[np.float64]
    """Stored the complete block-diagonal synthetic relationship."""

    environment: npt.NDArray[np.float64]
    """Stored the bounded participant-free environment coordinate."""

    fixed_effects: npt.NDArray[np.float64]
    """Stored the full-rank tracked six-column design identity."""

    component_sizes: tuple[int, ...]
    """Retained reviewed independent component sizes in generated row order."""

    reporting_grid: tuple[float, ...]
    """Fixed the only environments at which target summaries are judged."""

    fixture_sha256: str
    """Bound target evidence to the exact reviewed GxE fixture bytes."""


@dataclass(frozen=True)
class BlockwiseFit:
    """Hold the best independently optimized blockwise Gaussian fit.

    Attributes:
        parameters: Natural covariance coordinates at the best candidate.
        negative_loglik: Independently evaluated profiled objective.
        converged: Whether SciPy accepted the best candidate.
        message: Optimizer conclusion retained for release evidence.
    """

    parameters: npt.NDArray[np.float64]
    """Stored the best independent natural covariance coordinates."""

    negative_loglik: float
    """Stored the independently maximized negative log likelihood."""

    converged: bool
    """Recorded whether the external optimizer accepted the candidate."""

    message: str
    """Retained the optimizer's exact stopping conclusion."""


@dataclass(frozen=True)
class NullChunkJob:
    """Describe one scheduling-independent target null chunk.

    Attributes:
        surface: Supported covariance family owned by the chunk.
        start: First included replicate coordinate.
        stop: Exclusive final replicate coordinate.
        fixture_path: Exact target metadata selected by the command.
        fixture_sha256: Expected reviewed fixture identity.
    """

    surface: str
    """Selected one supported surface for every replicate in this chunk."""

    start: int
    """Stored the first deterministic replicate coordinate."""

    stop: int
    """Stored the exclusive end of the deterministic replicate range."""

    fixture_path: Path
    """Located the values-free target fixture inside the checkout."""

    fixture_sha256: str
    """Bound the chunk to exact reviewed target-fixture bytes."""


def gxe_target_envelope(
    path: Path,
    *,
    expected_sha256: str,
) -> GxeTargetEnvelope:
    """Verify the reviewed metadata and build its deterministic stress design.

    Args:
        path: Reviewed values-free GxE target fixture.
        expected_sha256: Exact digest selected independently by the caller.

    Returns:
        Deterministic participant-free target envelope for both supported surfaces.

    Raises:
        ValueError: If the fixture identity or generated design violates its contract.
    """
    selected_sha256: str = fixture_sha256(path)
    """Committed the generated design to the exact selected fixture bytes."""

    if selected_sha256 != expected_sha256:
        raise ValueError("GXE_TARGET_FIXTURE_SHA256_MISMATCH")

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    """Parsed the values-free fixture only after exact-byte verification."""

    if not isinstance(loaded, dict):
        raise ValueError("GXE_TARGET_FIXTURE_ROOT_INVALID")
    fixture: dict[str, object] = loaded
    """Narrowed the reviewed JSON root to its required object shape."""

    identity: tuple[object, ...] = (
        fixture.get("schema_version"),
        fixture.get("fixture_id"),
        fixture.get("reviewed"),
        fixture.get("participant_free"),
        fixture.get("purpose"),
    )
    """Collected the complete public review and scope identity."""

    if identity != (
        1,
        "continuous_gxe_target_design",
        True,
        True,
        "pedigree-scale and design-identity stress fixture",
    ):
        raise ValueError("GXE_TARGET_FIXTURE_IDENTITY_INVALID")

    pedigree_value: object = fixture.get("pedigree")
    """Selected the separately reviewed aggregate pedigree reference."""

    environment_value: object = fixture.get("environment")
    """Selected the bounded synthetic environment settings."""

    fixed_value: object = fixture.get("fixed_effects")
    """Selected the tracked six-column design identity settings."""

    if not isinstance(pedigree_value, dict):
        raise ValueError("GXE_TARGET_PEDIGREE_SETTINGS_INVALID")
    if not isinstance(environment_value, dict):
        raise ValueError("GXE_TARGET_ENVIRONMENT_SETTINGS_INVALID")
    if not isinstance(fixed_value, dict):
        raise ValueError("GXE_TARGET_FIXED_EFFECT_SETTINGS_INVALID")

    pedigree: dict[str, object] = pedigree_value
    """Narrowed the aggregate pedigree reference to an object."""

    environment_settings: dict[str, object] = environment_value
    """Narrowed the synthetic environment settings to an object."""

    fixed_settings: dict[str, object] = fixed_value
    """Narrowed the fixed-effect settings to an object."""

    pedigree_identity: tuple[object, ...] = (
        pedigree.get("aggregate_fixture"),
        pedigree.get("aggregate_fixture_sha256"),
        pedigree.get("analysis_n"),
        pedigree.get("component_count"),
        pedigree.get("largest_component"),
        pedigree.get("generated_nonzero_relationship_pairs"),
    )
    """Collected every reviewed pedigree-scale fact claimed by this fixture."""

    if pedigree_identity != (
        "checks/design_fixtures/one_trait_gaussian_heritability.json",
        "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e",
        1_909,
        202,
        180,
        27_691,
    ):
        raise ValueError("GXE_TARGET_PEDIGREE_SETTINGS_INVALID")

    repository: Path = path.parents[2]
    """Located the checkout root from the reviewed fixture location."""

    pedigree_path: Path = repository / str(pedigree["aggregate_fixture"])
    """Located the existing reviewed aggregate fixture without an external path."""

    pedigree_envelope: SimulationEnvelope = target_envelope(
        pedigree_path,
        expected_sha256=str(pedigree["aggregate_fixture_sha256"]),
        families=int(pedigree["component_count"]),
    )
    """Generated the established structure-matched participant-free pedigree."""

    environment_identity: tuple[object, ...] = (
        environment_settings.get("name"),
        environment_settings.get("generator"),
        environment_settings.get("seed"),
        environment_settings.get("minimum"),
        environment_settings.get("maximum"),
        environment_settings.get("reporting_grid"),
        environment_settings.get(
            "require_variation_in_every_component_larger_than_one"
        ),
        environment_settings.get("matches_participant_age_moments"),
    )
    """Collected every bounded-environment claim and explicit non-claim."""

    if environment_identity != (
        "synthetic_standardised_age_coordinate",
        "uniform",
        20_260_813,
        -1.5,
        1.5,
        [-1.0, 0.0, 1.0],
        True,
        False,
    ):
        raise ValueError("GXE_TARGET_ENVIRONMENT_SETTINGS_INVALID")

    people: int = pedigree_envelope.relationship.shape[0]
    """Read the reviewed roster size from the generated pedigree."""

    environment_generator: np.random.Generator = np.random.default_rng(20_260_813)
    """Created the tracked calibration stream independently of all outcomes."""

    environment: npt.NDArray[np.float64] = environment_generator.uniform(
        -1.5, 1.5, people
    )
    """Generated bounded synthetic standardised-age coordinates."""

    fixed_identity: tuple[object, ...] = (
        fixed_settings.get("columns"),
        fixed_settings.get("sex_generator"),
        fixed_settings.get("sex_seed"),
        fixed_settings.get("require_full_rank"),
        fixed_settings.get("matches_participant_sex_counts"),
    )
    """Collected the fixed-design identity and its explicit non-claim."""

    expected_columns: list[str] = [
        "intercept",
        "synthetic_standardised_age",
        "synthetic_standardised_age_squared",
        "synthetic_sex",
        "synthetic_standardised_age_by_sex",
        "synthetic_standardised_age_squared_by_sex",
    ]
    """Restated the tracked six-column formula independently of fixture edits."""

    if fixed_identity != (expected_columns, "binary_uniform", 1, True, False):
        raise ValueError("GXE_TARGET_FIXED_EFFECT_SETTINGS_INVALID")

    sex_generator: np.random.Generator = np.random.default_rng(1)
    """Created the dedicated participant-free binary-covariate stream."""

    sex: npt.NDArray[np.float64] = sex_generator.integers(0, 2, people).astype(float)
    """Generated synthetic binary values without matching participant counts."""

    fixed_effects: npt.NDArray[np.float64] = np.column_stack(
        [
            np.ones(people),
            environment,
            environment * environment,
            sex,
            environment * sex,
            environment * environment * sex,
        ]
    )
    """Built the tracked intercept, age, age-squared, sex, and interaction design."""

    if np.linalg.matrix_rank(fixed_effects) != len(expected_columns):
        raise ValueError("GXE_TARGET_FIXED_EFFECTS_RANK_DEFICIENT")

    offset: int = 0
    """Tracked component slices while enforcing informative environments."""

    for size in pedigree_envelope.component_sizes:
        stop: int = offset + size
        """Located the exclusive end of one independent pedigree component."""

        if size > 1 and np.ptp(environment[offset:stop]) <= 0.0:
            raise ValueError("GXE_TARGET_WITHIN_COMPONENT_ENVIRONMENT_CONSTANT")
        offset = stop
        """Advanced to the next contiguous pedigree component."""
    """Required within-family variation wherever a family has multiple rows."""

    return GxeTargetEnvelope(
        relationship=pedigree_envelope.relationship,
        environment=environment,
        fixed_effects=fixed_effects,
        component_sizes=pedigree_envelope.component_sizes,
        reporting_grid=(-1.0, 0.0, 1.0),
        fixture_sha256=selected_sha256,
    )


def gate_contract(path: Path, *, expected_sha256: str) -> dict[str, object]:
    """Read the executable acceptance fixed in exact reviewed fixture bytes.

    Args:
        path: Reviewed values-free GxE target fixture.
        expected_sha256: Exact digest selected independently by the caller.

    Returns:
        Normalised full-fit, null-calibration, and dense-sentinel rules.

    Raises:
        ValueError: If any executable scientific rule differs from the review.
    """
    if fixture_sha256(path) != expected_sha256:
        raise ValueError("GXE_TARGET_FIXTURE_SHA256_MISMATCH")

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    """Parsed the acceptance contract only after exact-byte verification."""

    if not isinstance(loaded, dict):
        raise ValueError("GXE_TARGET_FIXTURE_ROOT_INVALID")

    full_value: object = loaded.get("target_full_fit_acceptance")
    """Selected the prewritten target-sized independent-agreement rule."""

    null_value: object = loaded.get("null_calibration")
    """Selected the prewritten nonstandard-reference calibration rule."""

    dense_value: object = loaded.get("independent_dense_sentinel")
    """Selected the separate established dense fidelity sentinel."""

    if not isinstance(full_value, dict):
        raise ValueError("GXE_TARGET_FULL_FIT_ACCEPTANCE_INVALID")
    if not isinstance(null_value, dict):
        raise ValueError("GXE_TARGET_NULL_ACCEPTANCE_INVALID")
    if not isinstance(dense_value, dict):
        raise ValueError("GXE_TARGET_DENSE_SENTINEL_INVALID")

    full: dict[str, object] = full_value
    """Narrowed target full-fit acceptance to an object."""

    null: dict[str, object] = null_value
    """Narrowed target null calibration to an object."""

    dense: dict[str, object] = dense_value
    """Narrowed the independent dense sentinel to an object."""

    expected_full: dict[str, object] = {
        "surfaces": ["exponential", "random_regression"],
        "estimator": "reml",
        "quantity_absolute_tolerance": QUANTITY_TOLERANCE,
        "loglik_absolute_tolerance": LOGLIK_TOLERANCE,
        "require_public_convergence": True,
        "require_independent_convergence": True,
    }
    """Restated full-fit convergence and agreement independently of JSON edits."""

    expected_null: dict[str, object] = {
        "release_replicates": 500,
        "minimum_smoke_replicates": MINIMUM_SMOKE_REPLICATES,
        "levels": [0.05],
        "tests": ["correlation", "interaction"],
        "generating_genetic_variance": 0.5,
        "generating_residual_variance": 0.5,
        "base_seed": NULL_BASE_SEED,
        "fail_when": "one_sided_95_percent_clopper_pearson_lower_above_level",
        "maximum_refused_or_nonconverged": 0,
        "require_expected_reference_rule": True,
    }
    """Restated every null, uncertainty, and refusal decision before execution."""

    expected_dense: dict[str, object] = {
        "command": "checks/gxe_against_independent_full_fit.py",
        "sample_size": 80,
        "required": True,
    }
    """Restated the existing dense sentinel separately from target optimization."""

    if full != expected_full:
        raise ValueError("GXE_TARGET_FULL_FIT_ACCEPTANCE_INVALID")
    if null != expected_null:
        raise ValueError("GXE_TARGET_NULL_ACCEPTANCE_INVALID")
    if dense != expected_dense:
        raise ValueError("GXE_TARGET_DENSE_SENTINEL_INVALID")

    return {
        "release_replicates": null["release_replicates"],
        "minimum_smoke_replicates": null["minimum_smoke_replicates"],
        "levels": tuple(null["levels"]),
        "tests": tuple(null["tests"]),
        "maximum_refused_or_nonconverged": null["maximum_refused_or_nonconverged"],
        "quantity_absolute_tolerance": full["quantity_absolute_tolerance"],
        "loglik_absolute_tolerance": full["loglik_absolute_tolerance"],
        "dense_sentinel": dense["command"],
    }


def surface_covariance(
    surface: str,
    parameters: npt.NDArray[np.float64],
    relationship: npt.NDArray[np.float64],
    environment: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Assemble either supported continuous-GxE covariance independently.

    Args:
        surface: ``exponential`` or ``random_regression``.
        parameters: Natural covariance coordinates for the selected surface.
        relationship: Additive relationship for the same rows.
        environment: Prespecified environment coordinate for each row.

    Returns:
        Dense covariance for the supplied rows.

    Raises:
        ValueError: If the surface or coordinate count is unsupported.
    """
    if surface == "exponential":
        if parameters.shape != (5,):
            raise ValueError("GXE_REFERENCE_EXPONENTIAL_PARAMETERS_INVALID")
        alpha_g, slope_g, decay, alpha_e, slope_e = parameters
        """Named the independent log-linear covariance coordinates."""

        genetic_variance: npt.NDArray[np.float64] = np.exp(
            alpha_g + slope_g * environment
        )
        """Evaluated the genetic log-variance surface on the submitted rows."""

        residual_variance: npt.NDArray[np.float64] = np.exp(
            alpha_e + slope_e * environment
        )
        """Evaluated the residual log-variance surface on the same rows."""

        genetic: npt.NDArray[np.float64] = np.sqrt(
            np.outer(genetic_variance, genetic_variance)
        ) * np.exp(-decay * np.abs(environment[:, None] - environment[None, :]))
        """Built log-linear genetic scale with exponential correlation decay."""
    elif surface == "random_regression":
        if parameters.shape != (6,):
            raise ValueError("GXE_REFERENCE_RANDOM_REGRESSION_PARAMETERS_INVALID")

        genetic_loading: npt.NDArray[np.float64] = np.asarray(
            [
                [np.exp(parameters[0]), 0.0],
                [parameters[1], np.exp(parameters[2])],
            ]
        )
        """Mapped natural genetic loadings to a two-coordinate factor."""

        residual_loading: npt.NDArray[np.float64] = np.asarray(
            [
                [np.exp(parameters[3]), 0.0],
                [parameters[4], np.exp(parameters[5])],
            ]
        )
        """Mapped natural residual loadings to a separate factor."""

        genetic_block: npt.NDArray[np.float64] = genetic_loading @ genetic_loading.T
        """Constructed the positive-semidefinite genetic coefficient covariance."""

        residual_block: npt.NDArray[np.float64] = residual_loading @ residual_loading.T
        """Constructed the positive-semidefinite residual coefficient covariance."""

        basis: npt.NDArray[np.float64] = np.column_stack(
            [np.ones(environment.size), environment]
        )
        """Evaluated the intercept-and-linear reaction-norm basis."""

        genetic = basis @ genetic_block @ basis.T
        """Evaluated genetic covariance between every environment pair."""

        residual_variance = np.einsum("ij,jk,ik->i", basis, residual_block, basis)
        """Evaluated the environment-specific residual variance."""
    else:
        raise ValueError("GXE_REFERENCE_SURFACE_UNSUPPORTED")

    covariance: npt.NDArray[np.float64] = relationship * genetic
    """Scaled the independent genetic surface by additive relationships."""

    covariance[np.diag_indices_from(covariance)] += residual_variance
    """Added independent residual variation only to marginal variances."""
    return covariance


def blockwise_profiled_negative_loglik(
    surface: str,
    parameters: npt.NDArray[np.float64],
    relationship: npt.NDArray[np.float64],
    environment: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    component_sizes: Sequence[int],
    *,
    reml: bool,
) -> float:
    """Evaluate profiled Gaussian likelihood by independent pedigree blocks.

    Args:
        surface: Supported continuous-GxE covariance family.
        parameters: Natural covariance coordinates for that family.
        relationship: Block-diagonal additive relationship.
        environment: Environment coordinate in row order.
        design: Full-rank fixed-effect design.
        response: Continuous response in the same row order.
        component_sizes: Contiguous independent block sizes.
        reml: Whether to add invariant restricted-likelihood adjustment.

    Returns:
        Finite profiled negative likelihood, or infinity outside its domain.
    """
    rows: int = response.size
    """Counted Gaussian rows in the complete target design."""

    columns: int = design.shape[1]
    """Counted fixed effects profiled globally across pedigree blocks."""

    if (
        relationship.shape != (rows, rows)
        or environment.shape != (rows,)
        or design.shape[0] != rows
        or sum(component_sizes) != rows
    ):
        raise ValueError("GXE_REFERENCE_BLOCKWISE_DIMENSIONS_INVALID")

    weighted_design: npt.NDArray[np.float64] = np.zeros((columns, columns))
    """Initialised the global generalised least-squares normal matrix."""

    weighted_response: npt.NDArray[np.float64] = np.zeros(columns)
    """Initialised the global covariance-weighted response vector."""

    response_quadratic: float = 0.0
    """Accumulated the unprofiled response quadratic across components."""

    logdet: float = 0.0
    """Accumulated the covariance log determinant across independent blocks."""

    offset: int = 0
    """Tracked contiguous component slices in reviewed row order."""

    try:
        for size in component_sizes:
            stop: int = offset + size
            """Located the exclusive end of one independent covariance block."""

            covariance: npt.NDArray[np.float64] = surface_covariance(
                surface,
                parameters,
                relationship[offset:stop, offset:stop],
                environment[offset:stop],
            )
            """Assembled this block through independent surface equations."""

            if not np.all(np.isfinite(covariance)):
                return float("inf")
            factor: tuple[npt.NDArray[np.float64], bool] = cho_factor(
                covariance,
                lower=True,
                check_finite=False,
            )
            """Factored the independently assembled component covariance."""

            block_response: npt.NDArray[np.float64] = response[offset:stop]
            """Selected this component's response rows."""

            block_design: npt.NDArray[np.float64] = design[offset:stop]
            """Selected the matching fixed-effect rows."""

            inverse_response: npt.NDArray[np.float64] = cho_solve(
                factor, block_response, check_finite=False
            )
            """Applied this covariance inverse to the response block."""

            inverse_design: npt.NDArray[np.float64] = cho_solve(
                factor, block_design, check_finite=False
            )
            """Applied the same inverse to every fixed-effect column."""

            weighted_design += block_design.T @ inverse_design
            """Accumulated the global covariance-weighted normal matrix."""

            weighted_response += block_design.T @ inverse_response
            """Accumulated the global covariance-weighted response vector."""

            response_quadratic += float(block_response @ inverse_response)
            """Accumulated the response quadratic before profiling the mean."""

            diagonal: npt.NDArray[np.float64] = np.diag(np.tril(factor[0]))
            """Read this block determinant from the valid Cholesky triangle."""

            if np.any(diagonal <= 0.0):
                return float("inf")
            logdet += 2.0 * float(np.log(diagonal).sum())
            """Accumulated this independent block's covariance log determinant."""

            offset = stop
            """Advanced to the next contiguous pedigree component."""
        """Accumulated sufficient statistics without forming a target-sized inverse."""

        effects: npt.NDArray[np.float64] = np.linalg.solve(
            weighted_design, weighted_response
        )
        """Profiled fixed effects jointly across all independent components."""
    except (np.linalg.LinAlgError, ValueError):
        return float("inf")

    quadratic: float = response_quadratic - float(weighted_response @ effects)
    """Removed the globally profiled mean contribution from the quadratic."""

    value: float = 0.5 * (rows * LOG_TWO_PI + logdet + quadratic)
    """Evaluated the profiled maximum-likelihood objective."""

    if reml:
        sign_weighted: float
        """Reserved the weighted-design determinant sign."""

        logdet_weighted: float
        """Reserved the weighted-design log determinant."""

        sign_weighted, logdet_weighted = np.linalg.slogdet(weighted_design)
        """Measured the covariance-weighted design determinant."""

        sign_plain: float
        """Reserved the ordinary design determinant sign."""

        logdet_plain: float
        """Reserved the ordinary design log determinant."""

        sign_plain, logdet_plain = np.linalg.slogdet(design.T @ design)
        """Measured the plain determinant used for basis invariance."""

        if sign_weighted <= 0.0 or sign_plain <= 0.0:
            return float("inf")
        value += (
            0.5 * (float(logdet_weighted) - float(logdet_plain))
            - 0.5 * columns * LOG_TWO_PI
        )
        """Added Asterism's invariant restricted-likelihood adjustment."""

    return value if np.isfinite(value) else float("inf")


def reference_truth(surface: str) -> npt.NDArray[np.float64]:
    """Return the prewritten interior coordinates for one supported surface.

    Args:
        surface: ``exponential`` or ``random_regression``.

    Returns:
        Natural independent-reference coordinates used to simulate one response.

    Raises:
        ValueError: If the surface is outside 0.1 support.
    """
    if surface == "exponential":
        return np.asarray([np.log(0.55), 0.25, 0.35, np.log(0.50), -0.15])
    if surface == "random_regression":
        return np.asarray(
            [
                np.log(np.sqrt(0.60)),
                0.10 / np.sqrt(0.60),
                np.log(np.sqrt(0.30 - 0.10**2 / 0.60)),
                np.log(np.sqrt(0.50)),
                -0.05 / np.sqrt(0.50),
                np.log(np.sqrt(0.20 - 0.05**2 / 0.50)),
            ]
        )
    raise ValueError("GXE_REFERENCE_SURFACE_UNSUPPORTED")


def simulate_surface_response(
    surface: str,
    relationship: npt.NDArray[np.float64],
    environment: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    component_sizes: Sequence[int],
    *,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Generate one deterministic response by independent pedigree blocks.

    Args:
        surface: Supported continuous-GxE covariance family.
        relationship: Block-diagonal synthetic additive relationship.
        environment: Bounded synthetic environment coordinate.
        design: Fixed-effect design in the same row order.
        component_sizes: Contiguous independent pedigree blocks.
        seed: Dedicated participant-free outcome stream.

    Returns:
        Complete simulated response without a target-sized covariance factorization.
    """
    rows: int = environment.size
    """Counted response rows from the fixed target environment."""

    if (
        relationship.shape != (rows, rows)
        or design.shape[0] != rows
        or design.shape[1] > MEAN_COEFFICIENTS.size
        or sum(component_sizes) != rows
    ):
        raise ValueError("GXE_TARGET_SIMULATION_DIMENSIONS_INVALID")

    parameters: npt.NDArray[np.float64] = reference_truth(surface)
    """Selected the prewritten interior covariance for this supported family."""

    response: npt.NDArray[np.float64] = design @ MEAN_COEFFICIENTS[: design.shape[1]]
    """Started the outcome at its deterministic fixed-effect mean."""

    generator: np.random.Generator = np.random.default_rng(seed)
    """Created the sole random stream for this surface response."""

    offset: int = 0
    """Tracked component slices while adding independent Gaussian variation."""

    for size in component_sizes:
        stop: int = offset + size
        """Located the exclusive end of one synthetic pedigree block."""

        covariance: npt.NDArray[np.float64] = surface_covariance(
            surface,
            parameters,
            relationship[offset:stop, offset:stop],
            environment[offset:stop],
        )
        """Assembled this component under the surface's own interior family."""

        factor: npt.NDArray[np.float64] = np.linalg.cholesky(covariance)
        """Factored only this independent component's generating covariance."""

        response[offset:stop] += factor @ generator.standard_normal(size)
        """Added the correlated Gaussian draw for every row in this component."""

        offset = stop
        """Advanced to the next contiguous pedigree component."""
    """Generated the complete outcome without crossing independent components."""
    return response


def reference_starts_and_bounds(
    surface: str,
) -> tuple[
    tuple[npt.NDArray[np.float64], ...],
    tuple[tuple[float | None, float | None], ...],
]:
    """Return fixed independent starts and natural-coordinate restrictions.

    Args:
        surface: ``exponential`` or ``random_regression``.

    Returns:
        Prespecified optimizer starts and parallel SciPy bounds.

    Raises:
        ValueError: If the surface is outside 0.1 support.
    """
    truth: npt.NDArray[np.float64] = reference_truth(surface)
    """Included the generating interior point as a diagnostic optimizer start."""

    if surface == "exponential":
        starts: tuple[npt.NDArray[np.float64], ...] = (
            truth,
            np.asarray([np.log(0.5), 0.0, 0.1, np.log(0.5), 0.0]),
            np.asarray([np.log(0.7), -0.2, 0.7, np.log(0.4), 0.2]),
        )
        """Spanned flat, moderate, and stronger exponential surface shapes."""

        bounds: tuple[tuple[float | None, float | None], ...] = (
            (-5.0, 3.0),
            (-3.0, 3.0),
            (0.0, 3.0),
            (-5.0, 3.0),
            (-3.0, 3.0),
        )
        """Applied only the published exponential natural-coordinate domain."""
        return starts, bounds

    if surface == "random_regression":
        starts = (
            truth,
            np.asarray(
                [
                    np.log(0.7),
                    0.0,
                    np.log(0.3),
                    np.log(0.6),
                    0.0,
                    np.log(0.3),
                ]
            ),
            np.asarray(
                [
                    np.log(0.5),
                    0.3,
                    np.log(0.5),
                    np.log(0.5),
                    -0.2,
                    np.log(0.4),
                ]
            ),
        )
        """Spanned distinct positive-definite coefficient covariances."""

        bounds = (
            (-20.0, 3.0),
            (-3.0, 3.0),
            (-20.0, 3.0),
            (-20.0, 3.0),
            (-3.0, 3.0),
            (-20.0, 3.0),
        )
        """Allowed either independent covariance block near its rank-one limit."""
        return starts, bounds

    raise ValueError("GXE_REFERENCE_SURFACE_UNSUPPORTED")


def fit_blockwise_surface(
    surface: str,
    relationship: npt.NDArray[np.float64],
    environment: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    component_sizes: Sequence[int],
    *,
    reml: bool,
) -> BlockwiseFit:
    """Optimize a target-sized surface through the independent block likelihood.

    Args:
        surface: Supported continuous-GxE covariance family.
        relationship: Block-diagonal synthetic additive relationship.
        environment: Bounded synthetic environment coordinate.
        design: Full-rank fixed-effect design.
        response: Participant-free simulated response.
        component_sizes: Contiguous independent pedigree blocks.
        reml: Whether to optimize invariant restricted likelihood.

    Returns:
        Best finite SciPy candidate across the prespecified starts.
    """
    starts, bounds = reference_starts_and_bounds(surface)
    """Selected fixed independent optimization coordinates before fitting."""

    def objective(parameters: npt.NDArray[np.float64]) -> float:
        """Score one natural-coordinate covariance by independent blocks."""
        try:
            return blockwise_profiled_negative_loglik(
                surface,
                parameters,
                relationship,
                environment,
                design,
                response,
                component_sizes,
                reml=reml,
            )
        except (FloatingPointError, OverflowError, ValueError):
            return float("inf")

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
    """Ran a separately implemented optimizer from every prewritten start."""

    finite: list[OptimizeResult] = [
        result for result in results if np.isfinite(float(result.fun))
    ]
    """Discarded only candidates outside the independent covariance domain."""

    if not finite:
        return BlockwiseFit(
            parameters=np.asarray(starts[0], dtype=float),
            negative_loglik=float("inf"),
            converged=False,
            message="no finite independent optimizer candidate",
        )

    best: OptimizeResult = min(finite, key=lambda result: float(result.fun))
    """Selected the globally best objective across all deterministic starts."""

    return BlockwiseFit(
        parameters=np.asarray(best.x, dtype=float),
        negative_loglik=float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
    )


def compare_surface_fit(
    surface: str,
    relationship: npt.NDArray[np.float64],
    environment: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    component_sizes: Sequence[int],
    reporting_grid: Sequence[float],
    *,
    reml: bool,
) -> dict[str, object]:
    """Compare one public fit with independent target-sized optimization.

    Args:
        surface: Supported continuous-GxE covariance family.
        relationship: Block-diagonal synthetic additive relationship.
        environment: Bounded synthetic environment coordinate.
        design: Full-rank fixed-effect design.
        response: Participant-free simulated response.
        component_sizes: Contiguous independent pedigree blocks.
        reporting_grid: Prespecified common summary environments.
        reml: Whether both routes use invariant restricted likelihood.

    Returns:
        Convergence, tolerance, difference, and pass evidence for both routes.
    """
    model: asterism.GxeModel = asterism.GxeModel(
        relationship,
        environment,
        design,
        surface=surface,
    )
    """Built the selected surface through the documented public Python class."""

    fitted: dict[str, object] = model.fit(response, grid=reporting_grid, reml=reml)
    """Fitted and summarized the target design through the public API."""

    reference: BlockwiseFit = fit_blockwise_surface(
        surface,
        relationship,
        environment,
        design,
        response,
        component_sizes,
        reml=reml,
    )
    """Optimized the same response through independent blockwise equations."""

    grid: npt.NDArray[np.float64] = np.asarray(reporting_grid, dtype=float)
    """Normalised the prewritten reporting coordinates for reference summaries."""

    expected: dict[str, npt.NDArray[np.float64]] = reference_summaries(
        surface,
        reference.parameters,
        grid,
    )
    """Converted independent coordinates to common reportable quantities."""

    quantity_names: tuple[str, ...] = (
        "genetic_variance",
        "residual_variance",
        "heritability",
        "genetic_correlation",
    )
    """Selected every common surface-independent quantity filled by both routes."""

    differences: dict[str, float] = {
        name: float(
            np.max(np.abs(np.asarray(fitted[name], dtype=float) - expected[name]))
        )
        for name in quantity_names
    }
    """Measured the worst absolute disagreement on each scientific scale."""

    loglik_difference: float = abs(float(fitted["loglik"]) + reference.negative_loglik)
    """Compared maximized likelihoods including the same Gaussian constant."""

    failures: list[str] = []
    """Collected convergence and prewritten numerical-agreement failures."""

    if fitted.get("converged") is not True:
        failures.append("Asterism free fit did not converge")
    if not reference.converged:
        failures.append(f"independent optimizer did not converge: {reference.message}")
    for name, difference in differences.items():
        if difference > QUANTITY_TOLERANCE:
            failures.append(f"{name} differs by {difference:.3e}")
    """Applied the existing full-fit quantity tolerance without tuning to target output."""

    if loglik_difference > LOGLIK_TOLERANCE:
        failures.append(f"log likelihood differs by {loglik_difference:.3e}")

    return {
        "surface": surface,
        "people": response.size,
        "largest_family": max(component_sizes),
        "estimator": "reml" if reml else "ml",
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


def calibration_decision(
    *,
    rejections: int,
    attempted: int,
    refused_or_nonconverged: int,
    level: float,
) -> dict[str, object]:
    """Apply the prewritten validity-first target null decision.

    Args:
        rejections: Public p-values at or below the prespecified level.
        attempted: Every requested null replicate in the denominator.
        refused_or_nonconverged: Replicates lacking a complete reportable test.
        level: Prespecified type-I error level.

    Returns:
        One-sided exact-binomial uncertainty and fail-closed verdict.

    Raises:
        ValueError: If the counts or level cannot define this experiment.
    """
    if (
        attempted < 1
        or rejections < 0
        or rejections > attempted
        or refused_or_nonconverged < 0
        or refused_or_nonconverged > attempted
        or not 0.0 < level < 1.0
    ):
        raise ValueError("GXE_TARGET_CALIBRATION_COUNTS_INVALID")

    lower: float = (
        0.0
        if rejections == 0
        else float(beta.ppf(0.05, rejections, attempted - rejections + 1))
    )
    """Computed the one-sided 95% exact lower limit for the null rejection rate."""

    if refused_or_nonconverged > 0:
        verdict: str = "refused_or_nonconverged"
        """Failed the target rule rather than dropping unavailable replicates."""
    elif lower > level:
        verdict = "anti_conservative"
        """Rejected a null rate whose exact uncertainty lies above its level."""
    else:
        verdict = "compatible_or_conservative"
        """Accepted valid or conservative boundary-reference behavior."""

    return {
        "attempted": attempted,
        "rejections": rejections,
        "rejection_rate": rejections / attempted,
        "one_sided_95_percent_clopper_pearson_lower": lower,
        "level": level,
        "refused_or_nonconverged": refused_or_nonconverged,
        "verdict": verdict,
        "passed": verdict == "compatible_or_conservative",
    }


def run_null_replicate(
    surface: str,
    relationship: npt.NDArray[np.float64],
    environment: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    component_sizes: Sequence[int],
    *,
    replicate: int,
) -> dict[str, object]:
    """Fit and unconditionally score one common flat-null target replicate.

    Args:
        surface: Supported continuous-GxE covariance family.
        relationship: Block-diagonal synthetic additive relationship.
        environment: Bounded synthetic environment coordinate.
        design: Full-rank fixed-effect design.
        component_sizes: Contiguous independent pedigree blocks.
        replicate: Nonnegative deterministic outcome-stream coordinate.

    Returns:
        Public free-fit and both test outcomes, including refusal evidence.
    """
    if surface not in ("exponential", "random_regression") or replicate < 0:
        raise ValueError("GXE_TARGET_NULL_JOB_INVALID")

    model: asterism.GxeModel = asterism.GxeModel(
        relationship,
        environment,
        design,
        surface=surface,
    )
    """Built the selected target surface through the public Python API."""

    response: npt.NDArray[np.float64] = design @ MEAN_COEFFICIENTS[: design.shape[1]]
    """Started the flat-null response at its prespecified fixed mean."""

    generator: np.random.Generator = np.random.default_rng(NULL_BASE_SEED + replicate)
    """Created a scheduling-independent stream shared across surface families."""

    offset: int = 0
    """Tracked pedigree components while adding common flat-null variation."""

    for size in component_sizes:
        stop: int = offset + size
        """Located the exclusive end of one independent pedigree component."""

        covariance: npt.NDArray[np.float64] = 0.5 * relationship[
            offset:stop, offset:stop
        ] + 0.5 * np.eye(size)
        """Built a unit-scale null with no environment in either covariance term."""

        response[offset:stop] += np.linalg.cholesky(covariance) @ (
            generator.standard_normal(size)
        )
        """Added this component's correlated Gaussian flat-null draw."""

        offset = stop
        """Advanced to the next independent synthetic pedigree."""
    """Generated the complete null response with every family represented."""

    try:
        fitted: dict[str, object] = model.fit(
            response,
            grid=(-1.0, 0.0, 1.0),
            reml=True,
        )
        """Fitted the free target surface before judging either constrained test."""
    except ValueError as error:
        return {
            "surface": surface,
            "replicate": replicate,
            "seed": NULL_BASE_SEED + replicate,
            "fit_converged": False,
            "refusal": str(error),
            "tests": {},
        }

    fit_converged: bool = fitted.get("converged") is True
    """Read the public free-fit convergence verdict without relabelling it."""

    tests: dict[str, dict[str, object]] = {}
    """Collected both named nonstandard null comparisons for this replicate."""

    if fit_converged:
        for null, expected_rule in EXPECTED_TEST_RULES.items():
            try:
                outcome: dict[str, object] = model.test(response, null, reml=True)
                """Ran one target null through the documented public test method."""
            except ValueError as error:
                tests[null] = {
                    "complete": False,
                    "refusal": str(error),
                    "rule": None,
                    "p_value": None,
                    "statistic": None,
                }
                """Retained the documented refusal rather than dropping the replicate."""
                continue

            p_value: float = float(outcome["p_value"])
            """Normalised the public reference-tail probability for aggregation."""

            statistic: float = float(outcome["statistic"])
            """Normalised the public likelihood-ratio statistic for validation."""

            rule: object = outcome.get("rule")
            """Read the public nonstandard reference-distribution identity."""

            alternative_loglik: float = float(outcome["alternative_loglik"])
            """Read the free likelihood repeated inside the constrained comparison."""

            fit_loglik: float = float(fitted["loglik"])
            """Read the separately requested public free-fit likelihood."""

            complete: bool = (
                np.isfinite(p_value)
                and 0.0 <= p_value <= 1.0
                and np.isfinite(statistic)
                and statistic >= 0.0
                and rule == expected_rule
                and abs(alternative_loglik - fit_loglik) <= 1e-6
            )
            """Required finite fields, the fixed rule, and identical free likelihoods."""

            tests[null] = {
                "complete": complete,
                "refusal": None,
                "rule": rule,
                "p_value": p_value,
                "statistic": statistic,
                "alternative_fit_loglik_difference": abs(
                    alternative_loglik - fit_loglik
                ),
            }
            """Stored every field needed to score or diagnose this target test."""
        """Ran both reportable test families only after a converged free fit."""

    return {
        "surface": surface,
        "replicate": replicate,
        "seed": NULL_BASE_SEED + replicate,
        "fit_converged": fit_converged,
        "refusal": None,
        "tests": tests,
    }


def aggregate_null_records(
    surface: str,
    records: Sequence[Mapping[str, object]],
    *,
    attempted: int,
    level: float,
) -> dict[str, object]:
    """Aggregate every requested target replicate into both test decisions.

    Args:
        surface: Supported surface shared by the records.
        records: Complete per-replicate public outcomes.
        attempted: Fixed requested denominator for this surface.
        level: Prespecified null rejection level.

    Returns:
        Per-test exact-binomial decisions and a fail-closed surface verdict.

    Raises:
        ValueError: If records do not exactly cover the requested surface campaign.
    """
    if surface not in ("exponential", "random_regression"):
        raise ValueError("GXE_TARGET_NULL_SURFACE_INVALID")
    if len(records) != attempted:
        raise ValueError("GXE_TARGET_NULL_RECORD_COUNT_MISMATCH")

    replicate_indices: set[int] = {
        int(record["replicate"])
        for record in records
        if isinstance(record.get("replicate"), int)
    }
    """Collected unique deterministic coordinates from every shaped record."""

    if replicate_indices != set(range(attempted)):
        raise ValueError("GXE_TARGET_NULL_REPLICATE_COORDINATES_INVALID")
    if any(record.get("surface") != surface for record in records):
        raise ValueError("GXE_TARGET_NULL_SURFACE_MISMATCH")

    decisions: dict[str, dict[str, object]] = {}
    """Collected the two prewritten nonstandard-reference decisions."""

    for null, expected_rule in EXPECTED_TEST_RULES.items():
        rejections: int = 0
        """Counted complete public p-values at or below the fixed level."""

        refused_or_nonconverged: int = 0
        """Counted unavailable or malformed results without dropping them."""

        for record in records:
            tests_value: object = record.get("tests")
            """Selected the public test collection from this replicate."""

            if record.get("fit_converged") is not True or not isinstance(
                tests_value, Mapping
            ):
                refused_or_nonconverged += 1
                """Counted the unavailable free fit in this test's denominator."""
                continue

            test_value: object = tests_value.get(null)
            """Selected the requested nonstandard null outcome."""

            if not isinstance(test_value, Mapping):
                refused_or_nonconverged += 1
                """Counted the missing named test in this test's denominator."""
                continue

            p_value: object = test_value.get("p_value")
            """Read the public reference-tail probability without assuming its type."""

            if (
                test_value.get("complete") is not True
                or test_value.get("rule") != expected_rule
                or not isinstance(p_value, int | float)
                or not np.isfinite(float(p_value))
            ):
                refused_or_nonconverged += 1
                """Counted malformed or wrong-reference test output as unavailable."""
                continue
            rejections += int(float(p_value) <= level)
            """Scored this complete public test in the unconditional denominator."""
        """Assigned every replicate to rejection, nonrejection, or failure."""

        decision: dict[str, object] = calibration_decision(
            rejections=rejections,
            attempted=attempted,
            refused_or_nonconverged=refused_or_nonconverged,
            level=level,
        )
        """Applied the one-sided exact-binomial and zero-refusal policy."""

        decision["expected_rule"] = expected_rule
        """Attached the fixed public reference identity to its decision."""

        decisions[null] = decision
        """Stored this named nonstandard test's complete target evidence."""
    """Scored correlation and interaction references independently."""

    passed: bool = all(decision["passed"] is True for decision in decisions.values())
    """Required both reportable target tests to pass without failures."""

    return {
        "surface": surface,
        "attempted": attempted,
        "level": level,
        "tests": decisions,
        "passed": passed,
    }


def null_jobs(
    *,
    replicates: int,
    workers: int,
    fixture_path: Path,
    fixture_sha256: str,
) -> list[NullChunkJob]:
    """Partition both surface campaigns without changing replicate seeds.

    Args:
        replicates: Fixed denominator for each supported surface.
        workers: Maximum process chunks across both surfaces.
        fixture_path: Reviewed values-free target fixture.
        fixture_sha256: Expected exact fixture identity.

    Returns:
        Nonempty contiguous surface chunks covering every coordinate once.

    Raises:
        ValueError: If no valid campaign can be partitioned.
    """
    if replicates < 1 or workers < 1:
        raise ValueError("GXE_TARGET_NULL_PARTITION_INVALID")

    chunks_per_surface: int = min(replicates, max(1, workers // 2))
    """Divided process ownership equally across the two supported surfaces."""

    division: tuple[int, int] = divmod(replicates, chunks_per_surface)
    """Computed balanced contiguous chunk sizes without random scheduling effects."""

    base_size, extra = division
    """Unpacked the common chunk length and number of one-row extensions."""

    jobs: list[NullChunkJob] = []
    """Collected deterministic surface chunks in a stable order."""

    for surface in ("exponential", "random_regression"):
        start: int = 0
        """Reset replicate coordinates independently for each supported surface."""

        for chunk in range(chunks_per_surface):
            stop: int = start + base_size + int(chunk < extra)
            """Assigned this chunk its balanced exclusive replicate endpoint."""

            jobs.append(
                NullChunkJob(
                    surface=surface,
                    start=start,
                    stop=stop,
                    fixture_path=fixture_path,
                    fixture_sha256=fixture_sha256,
                )
            )
            """Bound the deterministic coordinate range to exact target metadata."""

            start = stop
            """Advanced to the next nonoverlapping replicate range."""
        """Covered every replicate exactly once for this surface."""
    return jobs


def run_dense_sentinel(path: Path) -> dict[str, object]:
    """Run and validate the established n=80 independent dense comparison.

    Args:
        path: Exact standalone dense full-fit command selected by the caller.

    Returns:
        Parsed participant-free record with command-level failure evidence.

    Raises:
        ValueError: If the command does not return its documented JSON shape.
    """
    selected: Path = path.resolve()
    """Resolved the explicit sentinel without searching an ambient executable path."""

    if selected.name != "gxe_against_independent_full_fit.py" or not selected.is_file():
        raise ValueError("GXE_TARGET_DENSE_SENTINEL_INVALID")

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(selected)],
        cwd=selected.parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=1_800,
    )
    """Ran the exact existing dense sentinel in the current installed environment."""

    try:
        loaded: object = json.loads(completed.stdout)
        """Parsed the sentinel's sole machine-readable standard-output record."""
    except json.JSONDecodeError as error:
        raise ValueError("GXE_TARGET_DENSE_SENTINEL_OUTPUT_INVALID") from error

    if not isinstance(loaded, dict):
        raise ValueError("GXE_TARGET_DENSE_SENTINEL_OUTPUT_INVALID")
    record: dict[str, object] = loaded
    """Narrowed the sentinel evidence to its documented object shape."""

    surfaces_value: object = record.get("surfaces")
    """Selected the separate surface comparison outcomes."""

    valid_surfaces: bool = (
        isinstance(surfaces_value, list)
        and len(surfaces_value) == 2
        and all(isinstance(surface, dict) for surface in surfaces_value)
        and {
            surface.get("surface")
            for surface in surfaces_value
            if isinstance(surface, dict)
        }
        == {"exponential", "random_regression"}
    )
    """Required the existing check to cover both and only supported surfaces."""

    passed: bool = (
        completed.returncode == 0
        and record.get("check") == "gxe_against_independent_full_fit"
        and record.get("participant_free") is True
        and record.get("passed") is True
        and valid_surfaces
    )
    """Combined process status with the sentinel's complete scientific identity."""

    record["command"] = str(path)
    """Recorded the exact explicit sentinel path selected by the target command."""

    record["exit_code"] = completed.returncode
    """Retained command status independently of its embedded verdict."""

    record["passed"] = passed
    """Failed closed on malformed identity even if the child claimed success."""
    return record


def run_target_null_chunk(job: NullChunkJob) -> list[dict[str, object]]:
    """Run one deterministic surface chunk in an isolated worker process.

    Args:
        job: Exact surface, replicate range, and reviewed fixture identity.

    Returns:
        Complete public outcomes for every replicate in the contiguous chunk.
    """
    envelope: GxeTargetEnvelope = gxe_target_envelope(
        job.fixture_path,
        expected_sha256=job.fixture_sha256,
    )
    """Rebuilt the participant-free target design from exact reviewed metadata."""

    records: list[dict[str, object]] = [
        run_null_replicate(
            job.surface,
            envelope.relationship,
            envelope.environment,
            envelope.fixed_effects,
            envelope.component_sizes,
            replicate=replicate,
        )
        for replicate in range(job.start, job.stop)
    ]
    """Scored every owned replicate through the real public fit and test methods."""
    return records


def run_null_calibration(
    fixture_path: Path,
    fixture_sha256: str,
    *,
    replicates: int,
    workers: int,
) -> dict[str, object]:
    """Calibrate both nonstandard references on the target structural envelope.

    Args:
        fixture_path: Reviewed values-free target metadata.
        fixture_sha256: Expected exact fixture identity.
        replicates: Fixed unconditional denominator for each surface.
        workers: Maximum process count across the two surfaces.

    Returns:
        Aggregate exact-binomial, convergence, refusal, and pass evidence.
    """
    jobs: list[NullChunkJob] = null_jobs(
        replicates=replicates,
        workers=workers,
        fixture_path=fixture_path.resolve(),
        fixture_sha256=fixture_sha256,
    )
    """Partitioned deterministic replicate streams without serialising matrices."""

    workers_used: int = min(workers, len(jobs))
    """Capped processes at the number of nonempty surface chunks."""

    started: float = time.perf_counter()
    """Started wall timing around target public fits only."""

    if workers_used == 1:
        chunks: list[list[dict[str, object]]] = [
            run_target_null_chunk(job) for job in jobs
        ]
        """Ran the same worker boundary directly for a single-process command."""
    else:
        with ProcessPoolExecutor(max_workers=workers_used) as pool:
            chunks = list(pool.map(run_target_null_chunk, jobs, chunksize=1))
            """Ran independent target chunks without changing replicate seeds."""
        """Closed every worker before aggregating scientific decisions."""

    records: list[dict[str, object]] = [record for chunk in chunks for record in chunk]
    """Flattened process chunks while retaining their surface and replicate labels."""

    surfaces: list[dict[str, object]] = []
    """Collected independently judged exponential and random-regression evidence."""

    for surface in ("exponential", "random_regression"):
        selected: list[dict[str, object]] = sorted(
            (record for record in records if record.get("surface") == surface),
            key=lambda record: int(record["replicate"]),
        )
        """Restored deterministic replicate order independently of worker completion."""

        surfaces.append(
            aggregate_null_records(
                surface,
                selected,
                attempted=replicates,
                level=0.05,
            )
        )
        """Applied the fixed exact-binomial and zero-refusal target rule."""
    """Judged both supported surface families on their common flat null."""

    passed: bool = all(surface["passed"] is True for surface in surfaces)
    """Required both tests on both supported surfaces to pass."""

    return {
        "replicates_per_surface": replicates,
        "surface_replicates": 2 * replicates,
        "workers_requested": workers,
        "workers_used": workers_used,
        "base_seed": NULL_BASE_SEED,
        "generating_genetic_variance": 0.5,
        "generating_residual_variance": 0.5,
        "every_replicate_scored": True,
        "refusal_policy": "fail_when_any_refused_or_nonconverged",
        "level_rule": "one_sided_95_percent_clopper_pearson_lower_above_level",
        "surfaces": surfaces,
        "seconds": time.perf_counter() - started,
        "passed": passed,
    }


def run_target_full_fits(envelope: GxeTargetEnvelope) -> dict[str, object]:
    """Compare both target-sized public fits with independent optimization.

    Args:
        envelope: Reviewed participant-free target pedigree and design.

    Returns:
        Per-surface full-fit convergence, agreement, timing, and pass evidence.
    """
    results: list[dict[str, object]] = []
    """Collected target-sized independent comparison evidence by surface."""

    for index, surface in enumerate(("exponential", "random_regression")):
        response: npt.NDArray[np.float64] = simulate_surface_response(
            surface,
            envelope.relationship,
            envelope.environment,
            envelope.fixed_effects,
            envelope.component_sizes,
            seed=31_401 + index,
        )
        """Generated one interior response from the selected surface's own family."""

        started: float = time.perf_counter()
        """Started timing around both public and independent target fits."""

        result: dict[str, object] = compare_surface_fit(
            surface,
            envelope.relationship,
            envelope.environment,
            envelope.fixed_effects,
            response,
            envelope.component_sizes,
            envelope.reporting_grid,
            reml=True,
        )
        """Compared maximized REML and every common reportable quantity."""

        result["response_seed"] = 31_401 + index
        """Recorded the participant-free surface-specific response stream."""

        result["seconds"] = time.perf_counter() - started
        """Retained target full-fit feasibility independently by surface."""

        results.append(result)
        """Stored this supported surface's complete target agreement evidence."""
    """Exercised both supported surfaces and no powered-exponential form."""

    return {
        "surfaces": results,
        "passed": all(result["passed"] is True for result in results),
    }


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Read a fully explicit participant-free target-design command.

    Args:
        arguments: Argument vector excluding the executable, or ``None`` for argv.

    Returns:
        Validated replicate, worker, fixture, sentinel, and output selections.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built the command contract without scientific or file defaults."""

    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--design-fixture", required=True, type=Path)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--dense-sentinel", required=True, type=Path)
    parser.add_argument(
        "--no-write",
        required=True,
        action="store_true",
        help="emit one values-free JSON record without modifying the checkout",
    )
    parsed: argparse.Namespace = parser.parse_args(arguments)
    """Read every campaign coordinate directly from the supplied argument vector."""

    if parsed.replicates < MINIMUM_SMOKE_REPLICATES:
        parser.error(
            f"--replicates must be at least {MINIMUM_SMOKE_REPLICATES} for this gate"
        )
    if parsed.workers < 1:
        parser.error("--workers must be positive")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    """Run both target surfaces, null calibration, and the dense sentinel.

    Args:
        arguments: Argument vector excluding the executable, or ``None`` for argv.

    Returns:
        Zero only when every prewritten participant-free target rule passes.
    """
    parsed: argparse.Namespace = parse_arguments(arguments)
    """Fixed every scientific and file coordinate before generating outcomes."""

    contract: dict[str, object] = gate_contract(
        parsed.design_fixture,
        expected_sha256=parsed.fixture_sha256,
    )
    """Bound execution to the acceptance rules in exact reviewed fixture bytes."""

    if parsed.replicates < int(contract["minimum_smoke_replicates"]):
        raise ValueError("GXE_TARGET_REPLICATES_BELOW_REVIEWED_MINIMUM")
    if str(parsed.dense_sentinel) != contract["dense_sentinel"]:
        raise ValueError("GXE_TARGET_DENSE_SENTINEL_COMMAND_MISMATCH")

    envelope: GxeTargetEnvelope = gxe_target_envelope(
        parsed.design_fixture,
        expected_sha256=parsed.fixture_sha256,
    )
    """Built the deterministic pedigree-scale and design-identity stress envelope."""

    started: float = time.perf_counter()
    """Started complete gate timing before its three independent evidence tiers."""

    target_full_fits: dict[str, object] = run_target_full_fits(envelope)
    """Ran target-sized independent optimization for both supported surfaces."""

    dense_sentinel: dict[str, object] = run_dense_sentinel(parsed.dense_sentinel)
    """Ran the existing n=80 dense fidelity check as a separate sentinel."""

    null_calibration: dict[str, object] = run_null_calibration(
        parsed.design_fixture,
        parsed.fixture_sha256,
        replicates=parsed.replicates,
        workers=parsed.workers,
    )
    """Calibrated both nonstandard references on the reviewed target structure."""

    passed: bool = (
        target_full_fits.get("passed") is True
        and dense_sentinel.get("passed") is True
        and null_calibration.get("passed") is True
    )
    """Required target fidelity, dense fidelity, and target validity together."""

    evidence: dict[str, object] = {
        "check": "gxe_target_design",
        "analysis_id": "continuous_gene_by_environment",
        "participant_free": True,
        "public_entry_points": [
            "asterism.GxeModel.fit",
            "asterism.GxeModel.test",
        ],
        "surfaces": ["exponential", "random_regression"],
        "excluded_surface": "powered_exponential",
        "fixture_sha256": envelope.fixture_sha256,
        "purpose": "pedigree-scale and design-identity stress fixture",
        "not_claimed": [
            "reconstruction of observed GOBS age moments",
            "trait-specific missingness",
            "participant row identity",
            "relationship coefficient or eigenvalue identity",
        ],
        "design_facts": {
            "people": envelope.relationship.shape[0],
            "component_count": len(envelope.component_sizes),
            "largest_component": max(envelope.component_sizes),
            "environment_range": [
                float(envelope.environment.min()),
                float(envelope.environment.max()),
            ],
            "qualified_environment_bounds": [-1.5, 1.5],
            "reporting_grid": list(envelope.reporting_grid),
            "fixed_effect_columns": envelope.fixed_effects.shape[1],
            "fixed_effect_rank": int(np.linalg.matrix_rank(envelope.fixed_effects)),
            "within_component_environment_variation": True,
        },
        "target_full_fits": target_full_fits,
        "dense_sentinel": dense_sentinel,
        "null_calibration": null_calibration,
        "release_replicates_per_surface": contract["release_replicates"],
        "replicates_actually_run_per_surface": parsed.replicates,
        "exact_release_campaign": (
            parsed.replicates == int(contract["release_replicates"])
        ),
        "seconds": time.perf_counter() - started,
        "passed": passed,
    }
    """Built values-free evidence sufficient to recompute every gate decision."""

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
