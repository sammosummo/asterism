"""Run deterministic participant-free probes through Asterism's public API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any

import asterism
import numpy as np

PROBE_ID: str = "asterism-cross-platform-0.1"
"""Named the normalized cross-platform protocol independently of workflow files."""

PAIRS: int = 40
"""Kept synthetic fits small while retaining repeated family structure."""

SEED: int = 20_260_821
"""Fixed every generated value independently of platform and process state."""

REFUSAL_PATTERN: re.Pattern[str] = re.compile(r"([A-Z][A-Z0-9_]+)")
"""Extracted the stable leading code from a documented public ValueError."""

NUMERIC_FIELDS_BY_ANALYSIS: dict[str, tuple[str, ...]] = {
    "one_trait_gaussian_heritability": (
        "fit.h2",
        "fit.total_variance",
        "fit.loglik",
        "fit.interval.lower",
        "fit.interval.upper",
        "fit.test.statistic",
        "fit.test.p_value",
    ),
    "several_covariance_components": (
        "fit.mean_diagonal_proportions[0]",
        "fit.loglik",
        "interval.estimate",
        "interval.lower",
        "interval.upper",
        "test.statistic",
        "test.p_value",
    ),
    "bivariate_genetic_correlation": (
        "fit.rho_g",
        "fit.loglik",
        "interval.estimate",
        "interval.lower",
        "interval.upper",
        "test.statistic",
        "test.p_value",
    ),
    "continuous_gene_by_environment": (
        "fit.heritability[1]",
        "fit.genetic_correlation[0][2]",
        "fit.loglik",
        "interval.estimate",
        "interval.lower",
        "interval.upper",
        "test.statistic",
        "test.p_value",
    ),
    "discrete_gene_by_environment": (
        "fit.genetic_correlation",
        "fit.heritability[0]",
        "fit.heritability[1]",
        "fit.loglik",
        "interval.estimate",
        "interval.lower",
        "interval.upper",
        "test.statistic",
        "test.p_value",
    ),
    "binary_liability_heritability": (
        "fit.heritability",
        "fit.loglik",
        "fit.prevalence",
        "interval.estimate",
        "interval.lower",
        "interval.upper",
        "test.statistic",
        "test.p_value",
    ),
    "one_trait_censored": (
        "fit.heritability",
        "fit.total_variance",
        "fit.loglik",
        "interval.estimate",
        "interval.lower",
        "interval.upper",
        "test.statistic",
        "test.p_value",
    ),
    "mixed_binary_censored_genetic_correlation": (
        "fit.genetic_correlation",
        "fit.heritability[0]",
        "fit.heritability[1]",
        "fit.loglik",
        "interval.estimate",
        "interval.lower",
        "interval.upper",
        "test.statistic",
        "test.p_value",
    ),
}
"""Fixed selected numeric paths before observing real cross-platform differences."""


def json_value(value: Any) -> Any:
    """Convert public fit-record values to strict portable JSON values.

    Args:
        value: Scalar, array or nested public result value.

    Returns:
        Recursively normalized JSON value.

    Raises:
        ValueError: If a public numerical result is non-finite.
        TypeError: If a public result contains an unsupported value type.
    """
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer | int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.floating | float):
        number: float = float(value)
        """Narrowed NumPy and Python floating values to one JSON representation."""

        if not math.isfinite(number):
            raise ValueError("PROBE_NONFINITE_PUBLIC_RESULT")
        return number
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported public result value: {type(value).__name__}")


def field_paths(value: Any, prefix: str = "") -> list[str]:
    """Return every present field and list position in one normalized result.

    Args:
        value: Strict JSON-compatible result tree.
        prefix: Path accumulated by recursive calls.

    Returns:
        Sorted unique field-presence paths.
    """
    paths: list[str] = []
    """Accumulated container and leaf paths so missing empty values remain visible."""

    if prefix:
        paths.append(prefix)
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            child: str = f"{prefix}.{key}" if prefix else str(key)
            """Extended the normalized path by one public dictionary key."""

            paths.extend(field_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            """Retained list length and position as part of exact field presence."""

            paths.extend(field_paths(item, child))
    return sorted(set(paths))


def boundary_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    """Collect explicit boundary and interval-limit states from a result tree.

    Args:
        value: Strict JSON-compatible result tree.
        prefix: Path accumulated by recursive calls.

    Returns:
        Boundary-state values indexed by normalized public field path.
    """
    states: dict[str, Any] = {}
    """Accumulated only categorical or boolean boundary evidence."""

    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            child: str = f"{prefix}.{key}" if prefix else str(key)
            """Extended the normalized path by one public dictionary key."""

            boundary_name: bool = (
                key == "boundary"
                or key.endswith("_limited")
                or key.startswith("contains_")
            )
            """Recognised the exact boundary vocabulary used by public fit records."""

            if boundary_name:
                states[child] = item
                """Retained even ``None`` because absence of a verdict is meaningful."""
            states.update(boundary_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            """Preserved boundary state nested within positional public collections."""

            states.update(boundary_fields(item, child))
    return states


def has_failures(value: Any) -> bool:
    """Return whether any required public fit or profile failed.

    Args:
        value: Strict JSON-compatible result tree.

    Returns:
        True for nonconvergence or a positive profile-failure count.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "converged" and item is not True:
                return True
            if key == "profile_failures" and item != 0:
                return True
            if has_failures(item):
                return True
    elif isinstance(value, list):
        return any(has_failures(item) for item in value)
    return False


def normalized_analysis(
    analysis_id: str,
    result: dict[str, Any] | None,
    numeric_fields: dict[str, float] | None,
    refusal: ValueError | None = None,
) -> dict[str, Any]:
    """Build one normalized cross-platform analysis result.

    Args:
        analysis_id: Stable identifier from the release manifest.
        result: Public fit, interval and test records, where a fit was made.
        numeric_fields: Preselected numerical values named by normalized paths.
        refusal: Stable public ValueError when no fit was produced.

    Returns:
        Structural and selected numeric fields ready for exact comparison.

    Raises:
        ValueError: If a refusal has no stable uppercase code.
    """
    if refusal is not None:
        match: re.Match[str] | None = REFUSAL_PATTERN.match(str(refusal))
        """Read only the documented leading refusal code, never free-form detail."""

        if match is None:
            raise ValueError("PROBE_REFUSAL_HAS_NO_STABLE_CODE") from refusal
        return {
            "analysis_id": analysis_id,
            "outcome_status": "refused",
            "refusal_code": match.group(1),
            "boundary_state": {},
            "field_presence": [],
            "numeric_fields": {},
        }
    if result is None or numeric_fields is None:
        raise ValueError("PROBE_CANDIDATE_RESULT_MISSING")
    normalized: dict[str, Any] = json_value(result)
    """Converted arrays and scalar wrappers without dropping public fields."""

    normalized_numeric: dict[str, float] = {
        name: float(json_value(value)) for name, value in sorted(numeric_fields.items())
    }
    """Restricted numerical comparison to values with pre-written model tolerances."""

    return {
        "analysis_id": analysis_id,
        "outcome_status": (
            "fitted-with-failures" if has_failures(normalized) else "fitted"
        ),
        "refusal_code": None,
        "boundary_state": boundary_fields(normalized),
        "field_presence": field_paths(normalized),
        "numeric_fields": normalized_numeric,
    }


def synthetic_problem() -> dict[str, Any]:
    """Create one deterministic participant-free family-analysis fixture.

    Returns:
        Synthetic matrices, traits, environments and censoring records.
    """
    people: int = 2 * PAIRS
    """Created two synthetic siblings in each independent family."""

    relationship: np.ndarray = np.eye(people, dtype=np.float64)
    """Started the twice-kinship matrix with unrelated unit diagonal entries."""

    for pair in range(PAIRS):
        first: int = 2 * pair
        """Located the first synthetic sibling's row."""

        second: int = first + 1
        """Located the paired sibling's row."""

        relationship[first, second] = 0.5
        """Added the first directed entry of symmetric sibling relatedness."""

        relationship[second, first] = 0.5
        """Completed the symmetric sibling relationship block."""
    """Added textbook full-sibling relatedness within every family block."""

    rng: np.random.Generator = np.random.default_rng(SEED)
    """Created the sole deterministic random stream used by every probe."""

    shared: np.ndarray = np.repeat(rng.standard_normal(PAIRS), 2)
    """Generated one family-level genetic signal shared by each sibling pair."""

    own: np.ndarray = rng.standard_normal(people)
    """Generated the within-family genetic contribution."""

    genetic: np.ndarray = np.sqrt(0.5) * shared + np.sqrt(0.5) * own
    """Combined shared and individual terms into covariance ``relationship``."""

    residual: np.ndarray = rng.standard_normal(people)
    """Generated independent environmental noise for the first trait."""

    environment: np.ndarray = np.linspace(-1.0, 1.0, people)
    """Placed observations on a fixed continuous environmental gradient."""

    response: np.ndarray = 0.75 * genetic + 0.55 * residual + 0.20 * environment
    """Created a continuous trait with family signal and a measured trend."""

    second_genetic: np.ndarray = 0.45 * genetic + np.sqrt(1.0 - 0.45**2) * (
        np.sqrt(0.5) * np.repeat(rng.standard_normal(PAIRS), 2)
        + np.sqrt(0.5) * rng.standard_normal(people)
    )
    """Generated a second genetic signal with fixed cross-trait correlation."""

    second_response: np.ndarray = (
        0.70 * second_genetic + 0.30 * residual + 0.55 * rng.standard_normal(people)
    )
    """Created a correlated second continuous trait on its own residual scale."""

    design: np.ndarray = np.ones((people, 1), dtype=np.float64)
    """Used one explicit intercept for every one-trait public model."""

    discrete_environment: np.ndarray = np.tile(
        np.array([0.0, 1.0], dtype=np.float64), PAIRS
    )
    """Placed one sibling from every family in each discrete environment."""

    positions: np.ndarray = np.column_stack(
        [np.arange(people, dtype=np.float64) % 10.0, np.arange(people) // 10]
    )
    """Placed synthetic people on a deterministic rectangular layout."""

    difference: np.ndarray = positions[:, None, :] - positions[None, :, :]
    """Constructed all pairwise coordinate differences."""

    distance: np.ndarray = np.sqrt(np.sum(difference**2, axis=2))
    """Converted coordinate differences into a symmetric Euclidean distance matrix."""

    binary_cut: float = float(np.quantile(response, 0.65))
    """Selected a deterministic prevalence without using any real diagnosis."""

    binary_status: np.ndarray = (response > binary_cut).astype(np.float64)
    """Converted the first latent trait into a synthetic binary diagnosis."""

    censoring_limit: float = float(np.quantile(second_response, 0.48))
    """Selected approximately 52 per cent right censoring for the hearing fixture."""

    censored: np.ndarray = second_response >= censoring_limit
    """Marked synthetic thresholds beyond the instrument limit."""

    censoring: np.ndarray = censored.astype(np.int64)
    """Encoded measured and right-censored observations through the public contract."""

    limits: np.ndarray = np.full(people, censoring_limit, dtype=np.float64)
    """Applied the same synthetic instrument limit to every observation."""

    censored_values: np.ndarray = np.where(censored, np.nan, second_response)
    """Removed latent values exactly where the simulated instrument stopped."""

    return {
        "relationship": relationship,
        "response": response,
        "second_response": second_response,
        "design": design,
        "environment": environment,
        "discrete_environment": discrete_environment,
        "distance": distance,
        "binary_status": binary_status,
        "censoring": censoring,
        "limits": limits,
        "censored_values": censored_values,
    }


def probe_one_trait(problem: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise one-trait Gaussian heritability through the public prepared model.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public result tree and selected numeric comparison fields.
    """
    commitment: str = asterism.subject_order_commitment(
        [f"synthetic-{index:03d}" for index in range(2 * PAIRS)]
    )
    """Committed to synthetic order without retaining even synthetic identifiers."""

    fit: dict[str, Any] = dict(
        asterism.prepare(
            problem["design"],
            problem["relationship"],
            subject_order_sha256=commitment,
        ).fit(problem["response"])
    )
    """Ran the complete public prepared-model fit, interval and test path."""

    return {"fit": fit}, {
        "fit.h2": fit["h2"],
        "fit.total_variance": fit["total_variance"],
        "fit.loglik": fit["loglik"],
        "fit.interval.lower": fit["interval"]["lower"],
        "fit.interval.upper": fit["interval"]["upper"],
        "fit.test.statistic": fit["test"]["statistic"],
        "fit.test.p_value": fit["test"]["p_value"],
    }


def probe_components(
    problem: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise scale-invariant several-component inference through public methods.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public fit, interval and test records with selected numeric fields.
    """
    model: asterism.ComponentModel = asterism.ComponentModel(
        [problem["relationship"]], problem["design"]
    )
    """Built the public component model with structured and residual components."""

    fit: dict[str, Any] = model.fit(problem["response"])
    """Fitted scale-invariant mean-diagonal contributions."""

    interval: dict[str, Any] = model.interval(problem["response"], component=0)
    """Profiled the mean-diagonal component proportion."""

    test: dict[str, Any] = model.test(problem["response"], component=0)
    """Tested the structured component against its boundary null."""

    return {"fit": fit, "interval": interval, "test": test}, {
        "fit.mean_diagonal_proportions[0]": fit["mean_diagonal_proportions"][0],
        "fit.loglik": fit["loglik"],
        "interval.estimate": interval["estimate"],
        "interval.lower": interval["lower"],
        "interval.upper": interval["upper"],
        "test.statistic": test["statistic"],
        "test.p_value": test["p_value"],
    }


def probe_bivariate(problem: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise bivariate genetic-correlation inference through the public model.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public fit, interval and test records with selected numeric fields.
    """
    people: int = 2 * PAIRS
    """Read the fixed synthetic roster size."""

    observed: np.ndarray = np.ones((people, 2), dtype=bool)
    """Marked both synthetic traits observed for every person."""

    pair_design: np.ndarray = np.tile(np.eye(2), (people, 1))
    """Gave each trait its own intercept in person-within-trait order."""

    response: np.ndarray = np.column_stack(
        [problem["response"], problem["second_response"]]
    ).ravel()
    """Interleaved both traits in the order required by the public bivariate model."""

    model: asterism.BivariateModel = asterism.BivariateModel(
        problem["relationship"], observed, pair_design
    )
    """Built the documented unbalanced-capable public bivariate model."""

    fit: dict[str, Any] = model.fit(response)
    """Estimated the genetic correlation and descriptive quantities."""

    interval: dict[str, Any] = model.interval(response, "rho_g")
    """Profiled the genetic correlation."""

    test: dict[str, Any] = model.test(response, "rho_g", 0.0)
    """Tested the genetic correlation against the interior null."""

    return {"fit": fit, "interval": interval, "test": test}, {
        "fit.rho_g": fit["rho_g"],
        "fit.loglik": fit["loglik"],
        "interval.estimate": interval["estimate"],
        "interval.lower": interval["lower"],
        "interval.upper": interval["upper"],
        "test.statistic": test["statistic"],
        "test.p_value": test["p_value"],
    }


def probe_continuous_gxe(
    problem: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise continuous gene-by-environment inference through the public model.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public fit, interval and test records with selected numeric fields.
    """
    model: asterism.GxeModel = asterism.GxeModel(
        problem["relationship"],
        problem["environment"],
        problem["design"],
        surface="random_regression",
    )
    """Selected one of the two surfaces scientifically supported in 0.1."""

    fit: dict[str, Any] = model.fit(problem["response"])
    """Fitted heritability and genetic-correlation surfaces at a fixed grid."""

    interval: dict[str, Any] = model.interval(
        problem["response"], quantity="heritability", first=0.0
    )
    """Profiled heritability at the centre of the measured environment."""

    test: dict[str, Any] = model.test(problem["response"], "correlation")
    """Tested whether the same genetic effects act across the environment."""

    return {"fit": fit, "interval": interval, "test": test}, {
        "fit.heritability[1]": fit["heritability"][1],
        "fit.genetic_correlation[0][2]": fit["genetic_correlation"][0][2],
        "fit.loglik": fit["loglik"],
        "interval.estimate": interval["estimate"],
        "interval.lower": interval["lower"],
        "interval.upper": interval["upper"],
        "test.statistic": test["statistic"],
        "test.p_value": test["p_value"],
    }


def probe_discrete_gxe(
    problem: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise discrete gene-by-environment inference through the public model.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public fit, interval and genetic test records with selected numeric fields.
    """
    model: asterism.DiscreteGxeModel = asterism.DiscreteGxeModel(
        problem["relationship"],
        problem["discrete_environment"],
        problem["design"],
        levels=(0.0, 1.0),
    )
    """Named both synthetic exposure levels explicitly at model construction."""

    fit: dict[str, Any] = model.fit(problem["response"])
    """Fitted group-specific variances and the genetic correlation."""

    interval: dict[str, Any] = model.correlation_interval(problem["response"])
    """Profiled the genetic correlation between environments."""

    test: dict[str, Any] = model.test(problem["response"], "gene_by_environment")
    """Ran the calibrated headline genetic-difference test."""

    return {"fit": fit, "interval": interval, "test": test}, {
        "fit.genetic_correlation": fit["genetic_correlation"],
        "fit.heritability[0]": fit["heritability"][0],
        "fit.heritability[1]": fit["heritability"][1],
        "fit.loglik": fit["loglik"],
        "interval.estimate": interval["estimate"],
        "interval.lower": interval["lower"],
        "interval.upper": interval["upper"],
        "test.statistic": test["statistic"],
        "test.p_value": test["p_value"],
    }


def probe_liability(problem: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise binary liability heritability through the public model.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public fit, interval and test records with selected numeric fields.
    """
    model: asterism.LiabilityModel = asterism.LiabilityModel(
        problem["relationship"], problem["binary_status"], problem["design"]
    )
    """Built the documented binary-liability public model."""

    fit: dict[str, Any] = model.fit()
    """Fitted liability-scale heritability by maximum likelihood."""

    interval: dict[str, Any] = model.interval()
    """Profiled the liability heritability."""

    test: dict[str, Any] = model.test()
    """Tested liability heritability against its boundary null."""

    return {"fit": fit, "interval": interval, "test": test}, {
        "fit.heritability": fit["heritability"],
        "fit.loglik": fit["loglik"],
        "fit.prevalence": fit["prevalence"],
        "interval.estimate": interval["estimate"],
        "interval.lower": interval["lower"],
        "interval.upper": interval["upper"],
        "test.statistic": test["statistic"],
        "test.p_value": test["p_value"],
    }


def probe_tobit(problem: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise one-trait censored audiogram inference through public functions.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public fit, interval and test records with selected numeric fields.
    """
    fit: dict[str, Any] = asterism.tobit_fit(
        problem["relationship"],
        problem["censored_values"],
        problem["censoring"],
        problem["limits"],
        problem["design"],
    )
    """Fitted the complete latent hearing threshold under right censoring."""

    interval: dict[str, Any] = asterism.tobit_interval(
        problem["relationship"],
        problem["censored_values"],
        problem["censoring"],
        problem["limits"],
        problem["design"],
    )
    """Profiled censored-trait heritability at approximately 52 per cent censoring."""

    test: dict[str, Any] = asterism.tobit_test(
        problem["relationship"],
        problem["censored_values"],
        problem["censoring"],
        problem["limits"],
        problem["design"],
    )
    """Tested censored heritability against its boundary null."""

    return {"fit": fit, "interval": interval, "test": test}, {
        "fit.heritability": fit["heritability"],
        "fit.total_variance": fit["total_variance"],
        "fit.loglik": fit["loglik"],
        "interval.estimate": interval["estimate"],
        "interval.lower": interval["lower"],
        "interval.upper": interval["upper"],
        "test.statistic": test["statistic"],
        "test.p_value": test["p_value"],
    }


def mixed_traits(problem: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the censored-hearing and continuous trait pairing 0.1 supports.

    The binary-with-censored pairing is deferred, so this exercises the
    censored-with-continuous case the extended-high-frequency paper fits.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        First right-censored and second continuous public trait specifications.
    """
    people: int = 2 * PAIRS
    """Read the fixed synthetic roster size."""

    first: dict[str, Any] = {
        "kind": "censored",
        "value": problem["censored_values"],
        "censoring": problem["censoring"],
        "limit": problem["limits"],
    }
    """Encoded the synthetic high-frequency hearing threshold, right censored."""

    second: dict[str, Any] = {
        "kind": "continuous",
        "value": problem["response"],
        "censoring": np.zeros(people, dtype=np.int64),
        "limit": np.zeros(people, dtype=np.float64),
    }
    """Encoded the paired conventional threshold, which nothing censors."""
    return first, second


def probe_mixed_bivariate(
    problem: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise the censored-by-continuous genetic-correlation analysis.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public fit, interval and test records with selected numeric fields.
    """
    first, second = mixed_traits(problem)
    """Built the exact supported psychiatric-diagnosis and hearing pairing."""

    fit: dict[str, Any] = asterism.mixed_bivariate_fit(
        problem["relationship"], first, second, problem["design"]
    )
    """Fitted the mixed-trait public model."""

    interval: dict[str, Any] = asterism.mixed_bivariate_interval(
        problem["relationship"],
        first,
        second,
        problem["design"],
        "genetic_correlation",
    )
    """Profiled the supported genetic correlation across trait kinds."""

    test: dict[str, Any] = asterism.mixed_bivariate_test(
        problem["relationship"],
        first,
        second,
        problem["design"],
        "genetic_correlation",
    )
    """Tested the cross-trait genetic correlation against nought."""

    return {"fit": fit, "interval": interval, "test": test}, {
        "fit.genetic_correlation": fit["genetic_correlation"],
        "fit.heritability[0]": fit["heritability"][0],
        "fit.heritability[1]": fit["heritability"][1],
        "fit.loglik": fit["loglik"],
        "interval.estimate": interval["estimate"],
        "interval.lower": interval["lower"],
        "interval.upper": interval["upper"],
        "test.statistic": test["statistic"],
        "test.p_value": test["p_value"],
    }


def probe_spatial(problem: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Exercise the spatial-component presence analysis through the public model.

    Args:
        problem: Deterministic participant-free fixture.

    Returns:
        Public fit and deterministic bootstrap records with selected numeric fields.
    """
    model: asterism.SpatialModel = asterism.SpatialModel(
        [problem["relationship"]], problem["distance"], problem["design"]
    )
    """Built the documented fixed-plus-spatial covariance model."""

    fit: dict[str, Any] = model.fit(problem["response"])
    """Fitted the spatial component and descriptive range."""

    test: dict[str, Any] = model.bootstrap(problem["response"], replicates=7, seed=SEED)
    """Ran a small deterministic public bootstrap presence test."""

    return {"fit": fit, "test": test}, {
        "fit.mean_diagonal_proportions[1]": fit["mean_diagonal_proportions"][1],
        "fit.loglik": fit["loglik"],
        "test.statistic": test["statistic"],
        "test.p_value": test["p_value"],
    }


def run_all_probes() -> list[dict[str, Any]]:
    """Run every supported 0.1 analysis and normalize its public result.

    Returns:
        Nine normalized analysis results in release-manifest order.
    """
    problem: dict[str, Any] = synthetic_problem()
    """Created one shared deterministic participant-free design."""

    analyses: list[dict[str, Any]] = []
    """Accumulated every supported analysis even when a stable refusal occurs."""

    probes: tuple[
        tuple[
            str,
            Any,
        ],
        ...,
    ] = (
        ("one_trait_gaussian_heritability", probe_one_trait),
        ("several_covariance_components", probe_components),
        ("bivariate_genetic_correlation", probe_bivariate),
        ("continuous_gene_by_environment", probe_continuous_gxe),
        ("discrete_gene_by_environment", probe_discrete_gxe),
        ("binary_liability_heritability", probe_liability),
        ("one_trait_censored", probe_tobit),
        ("mixed_binary_censored_genetic_correlation", probe_mixed_bivariate),
    )
    """Mapped the exact manifest inventory to independently testable public probes."""

    for analysis_id, probe in probes:
        try:
            result, numeric_fields = probe(problem)
            """Ran one complete analysis without intercepting programming errors."""
        except ValueError as refusal:
            analyses.append(
                normalized_analysis(analysis_id, None, None, refusal=refusal)
            )
        else:
            if set(numeric_fields) != set(NUMERIC_FIELDS_BY_ANALYSIS[analysis_id]):
                raise RuntimeError(f"PROBE_NUMERIC_FIELD_INVENTORY_DRIFT:{analysis_id}")
            analyses.append(normalized_analysis(analysis_id, result, numeric_fields))
    """Converted only documented public ValueErrors into stable refusal outcomes."""
    return analyses


def runtime_identity() -> dict[str, str]:
    """Return the canonical platform, architecture and interpreter target.

    Returns:
        Runtime identity matching the four promised workflow coordinates.

    Raises:
        ValueError: If the probe runs on an unsupported platform target.
    """
    system: str = platform.system()
    """Read the operating-system family from the running wheel host."""

    platform_name: str = {"Darwin": "macos", "Linux": "linux"}.get(system, "")
    """Mapped host names to the canonical release-manifest vocabulary."""

    architecture: str = platform.machine()
    """Read the host architecture used by compiled extension code."""

    if platform_name == "macos" and architecture == "aarch64":
        architecture = "arm64"
        """Normalised the alternate Apple arm64 spelling."""
    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
    """Recorded only the supported ABI-relevant major-minor interpreter identity."""

    key: tuple[str, str, str] = (platform_name, architecture, python_version)
    """Constructed the exact coordinate expected by the comparator."""

    supported: set[tuple[str, str, str]] = {
        ("macos", "arm64", "3.13"),
        ("macos", "arm64", "3.14"),
        ("linux", "x86_64", "3.13"),
        ("linux", "x86_64", "3.14"),
    }
    """Restricted probes to the 0.1 platform and interpreter promise."""

    if key not in supported:
        raise ValueError(f"PROBE_UNSUPPORTED_RUNTIME:{key!r}")
    return {
        "platform": platform_name,
        "architecture": architecture,
        "python_version": python_version,
    }


def run_probe(wheel_path: Path) -> dict[str, Any]:
    """Run and identify the complete installed-wheel public-API probe.

    Args:
        wheel_path: Exact saved wheel installed in the current interpreter.

    Returns:
        Machine-readable probe record covering all nine supported analyses.

    Raises:
        ValueError: If the selected wheel or public build identity is invalid.
    """
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise ValueError(f"PROBE_WHEEL_MISSING:{wheel_path}")
    manifest: dict[str, Any] = asterism.release_manifest()
    """Read the structured support contract through the public Python interface."""

    analyses: object = manifest.get("analyses")
    """Read the installed wheel's supported-analysis inventory."""

    manifest_ids: list[str] = (
        [str(analysis.get("id", "")) for analysis in analyses]
        if isinstance(analyses, list)
        else []
    )
    """Collected release order without importing implementation metadata."""

    expected_ids: list[str] = [
        "one_trait_gaussian_heritability",
        "several_covariance_components",
        "bivariate_genetic_correlation",
        "continuous_gene_by_environment",
        "discrete_gene_by_environment",
        "binary_liability_heritability",
        "one_trait_censored",
        "mixed_binary_censored_genetic_correlation",
    ]
    """Fixed the probe implementation to the exact accepted 0.1 analysis inventory."""

    if manifest_ids != expected_ids:
        raise ValueError("PROBE_MANIFEST_ANALYSES_DO_NOT_MATCH_IMPLEMENTATION")
    agreement: object = manifest.get("cross_platform_agreement")
    """Read the installed wheel's pre-written numerical comparison inventory."""

    configured_fields: dict[str, tuple[str, ...]] = {}
    """Collected selected fields from the embedded release contract."""

    if isinstance(agreement, dict) and isinstance(
        agreement.get("model_tolerances"), list
    ):
        for tolerance in agreement["model_tolerances"]:
            if isinstance(tolerance, dict) and isinstance(
                tolerance.get("analysis_id"), str
            ):
                fields: object = tolerance.get("numeric_fields")
                """Read one analysis's exact selected normalized paths."""

                if isinstance(fields, list) and all(
                    isinstance(field, str) for field in fields
                ):
                    configured_fields[str(tolerance["analysis_id"])] = tuple(fields)
                    """Retained manifest order for reviewable inventory equality."""
    """Parsed only the field-selection part, never unmeasured tolerance values."""

    if configured_fields != NUMERIC_FIELDS_BY_ANALYSIS:
        raise ValueError("PROBE_MANIFEST_NUMERIC_FIELDS_DO_NOT_MATCH_IMPLEMENTATION")
    build: dict[str, Any] = asterism.build_identity()
    """Read immutable source and release state through the public Python interface."""

    wheel_sha256: str = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    """Bound normalized results to the exact installed saved-wheel bytes."""

    return {
        "schema_version": 1,
        "probe_id": PROBE_ID,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "build": {
            "version": str(asterism.__version__),
            "source_commit": str(build["source_commit"]),
            "source_dirty": bool(build["source_dirty"]),
            "release": bool(build["release"]),
            "wheel_sha256": wheel_sha256,
        },
        "runtime": runtime_identity(),
        "analyses": run_all_probes(),
    }


def main() -> int:
    """Run the installed-wheel probe and write strict normalized JSON."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run all Asterism 0.1 public analysis probes."
    )
    """Defined the installed-wheel workflow interface."""

    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments: argparse.Namespace = parser.parse_args()
    """Parsed the exact saved wheel and normalized artifact path."""

    try:
        record: dict[str, Any] = run_probe(arguments.wheel)
        """Ran all nine analyses before creating the output path."""
    except (OSError, TypeError, ValueError) as error:
        print(f"cross-platform probe refused: {error}", file=sys.stderr)
        return 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    """Created the requested workflow artifact directory after successful fits."""

    arguments.output.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    """Persisted deterministic strict JSON with no participant-level values."""

    print(f"Wrote {len(record['analyses'])} normalized public-API probes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
