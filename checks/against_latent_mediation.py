"""Independently check Asterism's latent mediation likelihood.

For low-dimensional reference cases and central derivatives, this is an
independent reference calculation. It re-derives the published
structural covariance and Gaussian rectangle probabilities using only the
Python standard library, then calls *only* the public ``asterism`` interface.
It must not import another mediation-likelihood calculation or an Asterism
implementation module. The three-dimensional QMC comparison integrates the
conditional bivariate rectangle with this file's own quadrature, so it too is
independently derived; its tolerance is set by the measured accuracy of the
compiled estimator at 8,192 points, not by repeatability.

Run after building the extension::

    uv run --no-project python checks/against_latent_mediation.py
"""

from __future__ import annotations

import ast
import math
import sys
from collections.abc import Callable
from itertools import product
from pathlib import Path

LOW_DIMENSIONAL_ABSOLUTE_TOLERANCE: float = 2.0e-8
"""Set agreement tolerance for independently evaluated low-dimensional cases."""

CENTRAL_DIFFERENCE_TOLERANCE: float = 3.0e-6
"""Set agreement tolerance for independently differentiated likelihoods."""

DETERMINISTIC_TOLERANCE: float = 0.0
"""Required exact repeatability from the deterministic QMC evaluation."""

QMC_LOG_LIKELIHOOD_TOLERANCE: float = 5.0e-4
"""Set measured agreement tolerance for the trivariate QMC log likelihood."""

UNDERFLOW_LOG_LIKELIHOOD: float = -4546.421139077653
"""Pinned the continuous-density sentinel whose ordinary-scale value underflows."""

SQRT_TWO_PI: float = math.sqrt(2.0 * math.pi)
"""Precomputed the standard-normal density normalising constant."""


def enforce_reference_independence() -> None:
    """Refuse imports that would collapse this check into a copied calculation."""
    source: str = Path(__file__).read_text(encoding="utf-8")
    """Read this reference check exactly as executed."""

    tree: ast.Module = ast.parse(source, filename=str(Path(__file__)))
    """Parsed imports so independence could be enforced structurally."""

    forbidden_prefixes: tuple[str, ...] = (
        "numpy",
        "scipy",
        "mediation",
        "combined_family_likelihood",
        "asterism.",
    )
    """Listed third-party and implementation imports forbidden to the reference."""

    for node in ast.walk(tree):
        names: list[str] = []
        """Initialised imported module names carried by the current syntax node."""

        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            """Collected modules from a direct import statement."""

        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
            """Collected the parent module from a from-import statement."""

        for name in names:
            if name.startswith(forbidden_prefixes):
                raise RuntimeError(
                    f"reference-independence guard rejected forbidden import {name!r}"
                )


enforce_reference_independence()
import asterism  # noqa: E402  # The sole permitted Asterism import.


def normal_cdf(value: float) -> float:
    """Standard-normal distribution function from the error function."""
    if value == math.inf:
        return 1.0
    if value == -math.inf:
        return 0.0
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def normal_density(value: float) -> float:
    """Standard-normal density."""
    return math.exp(-0.5 * value * value) / SQRT_TWO_PI


def simpson(function: Callable[[float], float], left: float, right: float) -> float:
    """Return Simpson's one-panel integral for a scalar callable."""
    middle: float = 0.5 * (left + right)
    """Located the midpoint required by Simpson's three ordinates."""

    return (
        (right - left)
        * (function(left) + 4.0 * function(middle) + function(right))
        / 6.0
    )


def adaptive_simpson(
    function: Callable[[float], float],
    left: float,
    right: float,
    whole: float,
    tolerance: float,
    depth: int,
) -> float:
    """Recursively refine a Simpson integral to the requested tolerance."""
    middle: float = 0.5 * (left + right)
    """Bisected the current integration interval."""

    left_part: float = simpson(function, left, middle)
    """Evaluated Simpson's rule on the left half interval."""

    right_part: float = simpson(function, middle, right)
    """Evaluated Simpson's rule on the right half interval."""

    correction: float = left_part + right_part - whole
    """Measured refinement change from the parent Simpson panel."""

    if depth == 0 or abs(correction) <= 15.0 * tolerance:
        return left_part + right_part + correction / 15.0
    return adaptive_simpson(
        function, left, middle, left_part, tolerance / 2.0, depth - 1
    ) + adaptive_simpson(
        function, middle, right, right_part, tolerance / 2.0, depth - 1
    )


def bivariate_cdf(first: float, second: float, correlation: float) -> float:
    """Evaluate a standard bivariate-normal CDF by independent 1-D quadrature."""
    if first == -math.inf or second == -math.inf:
        return 0.0
    if first == math.inf:
        return normal_cdf(second)
    if second == math.inf:
        return normal_cdf(first)
    if correlation >= 1.0 - 1.0e-14:
        return normal_cdf(min(first, second))
    if correlation <= -1.0 + 1.0e-14:
        return max(0.0, normal_cdf(first) - normal_cdf(-second))
    correlation = max(-1.0 + 1.0e-15, min(1.0 - 1.0e-15, correlation))
    """Clamped correlation away from singular endpoints after exact limit handling."""

    scale: float = math.sqrt(1.0 - correlation * correlation)
    """Calculated conditional standard deviation of the second variate."""

    upper: float = min(first, 10.0)
    """Truncated the practically negligible upper integration tail."""

    if upper <= -10.0:
        return 0.0

    def integrand(value: float) -> float:
        standardised: float = (second - correlation * value) / scale
        """Standardised the second-coordinate bound conditional on the first."""

        return normal_density(value) * normal_cdf(standardised)

    whole: float = simpson(integrand, -10.0, upper)
    """Evaluated the initial Simpson panel across the effective normal support."""

    return min(
        1.0,
        max(0.0, adaptive_simpson(integrand, -10.0, upper, whole, 2.0e-13, 30)),
    )


def rectangle_probability(
    mean: list[float],
    covariance: list[list[float]],
    lower: list[float],
    upper: list[float],
) -> float:
    """Probability of a one- or two-dimensional Gaussian rectangle."""
    dimension: int = len(mean)
    """Counted coordinates in the requested Gaussian rectangle."""

    if dimension == 0:
        return 1.0
    if dimension == 1:
        scale: float = math.sqrt(covariance[0][0])
        """Calculated the univariate Gaussian standard deviation."""

        return max(
            0.0,
            normal_cdf((upper[0] - mean[0]) / scale)
            - normal_cdf((lower[0] - mean[0]) / scale),
        )
    if dimension == 3:
        # Condition on the first coordinate and integrate its density against
        # the conditional two-dimensional rectangle, using the same adaptive
        # quadrature as the bivariate reference. Still independent of the
        # compiled implementation, which never conditions this way.
        scale = math.sqrt(covariance[0][0])
        """Calculated the conditioning coordinate's standard deviation."""

        cross: list[float] = [
            covariance[1][0] / covariance[0][0],
            covariance[2][0] / covariance[0][0],
        ]
        """Calculated regression coefficients for the two remaining coordinates."""

        conditional: list[list[float]] = [
            [
                covariance[1][1] - cross[0] * covariance[0][1],
                covariance[1][2] - cross[0] * covariance[0][2],
            ],
            [
                covariance[2][1] - cross[1] * covariance[0][1],
                covariance[2][2] - cross[1] * covariance[0][2],
            ],
        ]
        """Calculated the remaining bivariate covariance conditional on coordinate one."""

        def integrand(value: float) -> float:
            shifted_mean: list[float] = [
                mean[1] + cross[0] * (value - mean[0]),
                mean[2] + cross[1] * (value - mean[0]),
            ]
            """Shifted remaining means conditional on the integration coordinate."""

            return (
                normal_density((value - mean[0]) / scale) / scale
            ) * rectangle_probability(shifted_mean, conditional, lower[1:], upper[1:])

        low: float = max(lower[0], mean[0] - 9.0 * scale)
        """Clipped the first-coordinate lower bound to effective normal support."""

        high: float = min(upper[0], mean[0] + 9.0 * scale)
        """Clipped the first-coordinate upper bound to effective normal support."""

        if low >= high:
            return 0.0
        whole: float = simpson(integrand, low, high)
        """Evaluated the initial quadrature panel for the conditional rectangle."""

        return min(
            1.0,
            max(0.0, adaptive_simpson(integrand, low, high, whole, 1.0e-12, 28)),
        )
    if dimension > 3:
        raise ValueError("this independent exact check admits at most three dimensions")
    first_scale: float = math.sqrt(covariance[0][0])
    """Calculated the first marginal standard deviation."""

    second_scale: float = math.sqrt(covariance[1][1])
    """Calculated the second marginal standard deviation."""

    correlation: float = covariance[0][1] / (first_scale * second_scale)
    """Converted the bivariate covariance to correlation."""

    def cdf(x_value: float, y_value: float) -> float:
        return bivariate_cdf(
            (x_value - mean[0]) / first_scale,
            (y_value - mean[1]) / second_scale,
            correlation,
        )

    value: float = cdf(upper[0], upper[1]) - cdf(lower[0], upper[1])
    """Initialised inclusion-exclusion with the two upper-bound CDF values."""

    value -= cdf(upper[0], lower[1])
    """Removed probability below the second lower bound."""

    value += cdf(lower[0], lower[1])
    """Restored the doubly subtracted lower-corner probability."""
    return max(0.0, value)


def cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """Cholesky factor for the tiny positive-definite matrices used here."""
    size: int = len(matrix)
    """Counted rows in the small positive-definite matrix."""

    factor: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
    """Allocated the lower-triangular Cholesky factor."""

    for row in range(size):
        for column in range(row + 1):
            value: float = matrix[row][column] - sum(
                factor[row][index] * factor[column][index] for index in range(column)
            )
            """Removed contributions from already factorised columns."""

            if row == column:
                if value <= 0.0:
                    raise ValueError("independent covariance was not positive definite")
                factor[row][column] = math.sqrt(value)
                """Set a positive diagonal Cholesky element."""

            else:
                factor[row][column] = value / factor[column][column]
                """Set one strictly lower-triangular Cholesky element."""
    return factor


def solve_cholesky(factor: list[list[float]], right: list[float]) -> list[float]:
    """Solve ``L L' x = right`` without a third-party linear-algebra package."""
    size: int = len(factor)
    """Counted equations in the Cholesky system."""

    forward: list[float] = [0.0] * size
    """Allocated the forward-substitution solution to the lower system."""

    for row in range(size):
        forward[row] = (
            right[row]
            - sum(factor[row][column] * forward[column] for column in range(row))
        ) / factor[row][row]
        """Solved one row of the lower-triangular system."""

    answer: list[float] = [0.0] * size
    """Allocated the backward-substitution solution to the transposed system."""

    for row in range(size - 1, -1, -1):
        answer[row] = (
            forward[row]
            - sum(
                factor[column][row] * answer[column] for column in range(row + 1, size)
            )
        ) / factor[row][row]
        """Solved one row of the upper-triangular transposed system."""
    return answer


def structural_covariance(
    family: dict[str, object], parameters: dict[str, float]
) -> list[list[float]]:
    """Re-derive the latent mediation process-major covariance matrix."""
    relationship: list[list[float]] = family["relationship"]
    """Read the supplied family relationship matrix."""

    people: int = len(relationship)
    """Counted people represented by the family record."""

    a: float = parameters["a"]
    """Read the mediator loading on the inherited factor."""

    b: float = parameters["b"]
    """Read the mediator-to-outcome path coefficient."""

    c_prime: float = parameters["c_prime"]
    """Read the direct inherited path to the outcome."""

    d: float = parameters["d"]
    """Read the outcome loading on its inherited residual factor."""

    sigma_m2: float = parameters["sigma_m2"]
    """Read the mediator-specific residual variance."""

    total_inherited_mediator: float = a * b + c_prime
    """Calculated the outcome's inherited loading mediated through and around M."""

    covariance: list[list[float]] = [
        [0.0 for _ in range(2 * people)] for _ in range(2 * people)
    ]
    """Allocated the process-major mediator and outcome covariance matrix."""

    for first in range(people):
        for second in range(people):
            relation: float = float(relationship[first][second])
            """Read the inherited relationship between the current people."""

            residual: float = 1.0 if first == second else 0.0
            """Selected person-specific residual covariance only on matching people."""

            mediator_covariance: float = a * a * relation + sigma_m2 * residual
            """Calculated covariance between the two latent mediator values."""

            cross_covariance: float = (
                a * total_inherited_mediator * relation + b * sigma_m2 * residual
            )
            """Calculated mediator-to-outcome covariance for the person pair."""

            outcome_covariance: float = (
                total_inherited_mediator * total_inherited_mediator + d * d
            ) * relation + (b * b * sigma_m2 + 1.0) * residual
            """Calculated covariance between the two latent outcomes."""

            covariance[first][second] = mediator_covariance
            """Stored the mediator-mediator covariance block element."""

            covariance[first][people + second] = cross_covariance
            """Stored the mediator-outcome covariance block element."""

            covariance[people + first][second] = cross_covariance
            """Stored the symmetric outcome-mediator covariance block element."""

            covariance[people + first][people + second] = outcome_covariance
            """Stored the outcome-outcome covariance block element."""
    return covariance


def continuous_conditioning(
    family: dict[str, object], covariance: list[list[float]]
) -> tuple[list[int], list[float], list[list[float]], float]:
    """Condition latent observations on noisy mediator measurements."""
    people: int = len(family["relationship"])
    """Counted people represented by the family relationship matrix."""

    means: list[float] = [float(value) for value in family["latent_mean"]]
    """Parsed process-major latent means for mediator and outcome values."""

    observed_people: list[int] = [
        person
        for person, value in enumerate(family["mediator_measurement"])
        if value is not None
    ]
    """Located people with a continuous mediator measurement."""

    if not observed_people:
        return [], means, covariance, 0.0
    observed: list[int] = observed_people
    """Mapped measured mediators directly to their process-major coordinates."""

    values: list[float] = [
        float(family["mediator_measurement"][person]) for person in observed_people
    ]
    """Parsed noisy mediator measurements in observed-coordinate order."""

    errors: list[float] = [
        float(family["mediator_measurement_error_variance"][person])
        for person in observed_people
    ]
    """Parsed measurement-error variances in observed-coordinate order."""

    observed_covariance: list[list[float]] = [
        [
            covariance[left][right] + (errors[row] if row == column else 0.0)
            for column, right in enumerate(observed)
        ]
        for row, left in enumerate(observed)
    ]
    """Added independent measurement error to the observed mediator covariance."""

    factor: list[list[float]] = cholesky(observed_covariance)
    """Factorised the noisy observed-mediator covariance."""

    residual: list[float] = [
        values[index] - means[observed[index]] for index in range(len(observed))
    ]
    """Calculated deviations of observed mediators from their latent means."""

    solved_residual: list[float] = solve_cholesky(factor, residual)
    """Solved the observed covariance against measurement residuals."""

    log_density: float = -0.5 * (
        len(observed) * math.log(2.0 * math.pi)
        + 2.0 * sum(math.log(factor[index][index]) for index in range(len(observed)))
        + sum(
            residual[index] * solved_residual[index] for index in range(len(observed))
        )
    )
    """Evaluated the noisy continuous mediator Gaussian log density."""

    cross: list[list[float]] = [
        [covariance[row][column] for column in observed] for row in range(2 * people)
    ]
    """Selected covariance from every latent coordinate to observed mediators."""

    conditioned_mean: list[float] = [
        means[row]
        + sum(
            cross[row][column] * solved_residual[column]
            for column in range(len(observed))
        )
        for row in range(2 * people)
    ]
    """Calculated every latent coordinate's conditional mean."""

    inverse_cross: list[list[float]] = [
        solve_cholesky(factor, cross[row]) for row in range(2 * people)
    ]
    """Solved the observed covariance against every latent cross-covariance row."""

    conditioned_covariance: list[list[float]] = [
        [
            covariance[row][column]
            - sum(
                cross[row][index] * inverse_cross[column][index]
                for index in range(len(observed))
            )
            for column in range(2 * people)
        ]
        for row in range(2 * people)
    ]
    """Calculated the full latent covariance conditional on measured mediators."""
    return observed, conditioned_mean, conditioned_covariance, log_density


def independent_log_likelihood(
    family: dict[str, object], parameters: dict[str, float]
) -> float:
    """Evaluate one family from the public model equations, independently."""
    people: int = len(family["relationship"])
    """Counted people represented by the family record."""

    covariance: list[list[float]] = structural_covariance(family, parameters)
    """Constructed the independent process-major latent covariance."""

    _, mean, covariance, continuous_log_density = continuous_conditioning(
        family, covariance
    )
    """Conditioned latent variables on any noisy continuous mediator measurements."""

    proxy_people: list[int] = [
        person
        for person, value in enumerate(family["mediator_proxy_status"])
        if value is not None
    ]
    """Located people with observed binary mediator proxies."""

    outcome_people: list[int] = [
        person
        for person, value in enumerate(family["outcome_status"])
        if value is not None
    ]
    """Located people with observed binary outcomes."""

    dimensions: list[int] = proxy_people + [
        people + person for person in outcome_people
    ]
    """Combined proxy and process-offset outcome coordinates in region order."""

    if not dimensions:
        probability: float = 1.0
        """Assigned unit discrete-observation probability when no statuses exist."""

    else:
        selected_mean: list[float] = [mean[index] for index in dimensions]
        """Selected conditional means for observed discrete coordinates."""

        selected_covariance: list[list[float]] = [
            [covariance[row][column] for column in dimensions] for row in dimensions
        ]
        """Selected the matching conditional covariance submatrix."""

        probability = 0.0
        """Initialised the observed-status probability before latent-truth summation."""

        for truths in product((0, 1), repeat=len(proxy_people)):
            lower: list[float] = []
            """Initialised lower bounds for this latent proxy-truth configuration."""

            upper: list[float] = []
            """Initialised upper bounds for this latent proxy-truth configuration."""

            observation_probability: float = 1.0
            """Initialised proxy misclassification probability for this truth pattern."""

            for position, person in enumerate(proxy_people):
                truth: int = truths[position]
                """Read the current proxy's candidate latent truth state."""

                threshold: float = float(family["mediator_threshold"][person])
                """Read the mediator threshold separating latent proxy states."""

                lower.append(threshold if truth else -math.inf)
                upper.append(math.inf if truth else threshold)
                status: int = int(family["mediator_proxy_status"][person])
                """Read the observed, potentially misclassified mediator proxy."""

                sensitivity: float = float(family["mediator_proxy_sensitivity"][person])
                """Read the proxy sensitivity for this person."""

                specificity: float = float(family["mediator_proxy_specificity"][person])
                """Read the proxy specificity for this person."""

                if status == 1:
                    observation_probability *= (
                        sensitivity if truth else 1.0 - specificity
                    )
                    """Multiplied by the positive-proxy probability under this truth."""

                else:
                    observation_probability *= (
                        1.0 - sensitivity if truth else specificity
                    )
                    """Multiplied by the negative-proxy probability under this truth."""

            for person in outcome_people:
                threshold = float(family["outcome_threshold"][person])
                """Read the binary-outcome threshold for this person."""

                if int(family["outcome_status"][person]) == 1:
                    lower.append(threshold)
                    upper.append(math.inf)
                else:
                    lower.append(-math.inf)
                    upper.append(threshold)
            probability += observation_probability * rectangle_probability(
                selected_mean, selected_covariance, lower, upper
            )
            """Added this latent-truth rectangle weighted by proxy misclassification."""

    if probability <= 0.0:
        raise ValueError(
            "independent likelihood underflowed in a finite reference check"
        )
    value: float = continuous_log_density + math.log(probability)
    """Combined continuous density and discrete-region log probability."""

    if family["ascertainment"] == "condition_on_named_proband_case":
        proband: int = int(family["proband_index"])
        """Read the named proband used for conditional ascertainment."""

        outcome_index: int = people + proband
        """Located the proband outcome in process-major latent coordinates."""

        threshold = float(family["outcome_threshold"][proband])
        """Read the proband's binary-outcome threshold."""

        denominator: float = 1.0 - normal_cdf(
            (threshold - float(family["latent_mean"][outcome_index]))
            / math.sqrt(
                structural_covariance(family, parameters)[outcome_index][outcome_index]
            )
        )
        """Calculated the proband case probability defining ascertainment correction."""

        value -= math.log(denominator)
        """Subtracted the named-proband ascertainment log probability."""
    return value


def singleton(ascertainment: str = "population_unconditioned") -> dict[str, object]:
    """One mixed-observation family."""
    return {
        "relationship": [[1.0]],
        "latent_mean": [0.0, 0.0],
        "mediator_measurement": [0.18],
        "mediator_measurement_error_variance": [0.22],
        "mediator_proxy_status": [1],
        "outcome_status": [1],
        "mediator_threshold": [-0.1],
        "outcome_threshold": [0.35],
        "mediator_proxy_sensitivity": [0.8],
        "mediator_proxy_specificity": [0.85],
        "ascertainment": ascertainment,
        "proband_index": 0 if ascertainment != "population_unconditioned" else None,
    }


def related_dyad(ascertainment: str) -> dict[str, object]:
    """A two-person outcome-only family exercising inherited covariance."""
    return {
        "relationship": [[1.0, 0.5], [0.5, 1.0]],
        # Process-major: two mediator means, then two outcome means.
        "latent_mean": [0.15, -0.25, 0.35, -0.45],
        "mediator_measurement": [None, None],
        "mediator_measurement_error_variance": [None, None],
        "mediator_proxy_status": [None, None],
        "outcome_status": [1, 0],
        "mediator_threshold": [0.0, 0.0],
        "outcome_threshold": [0.35, 0.5],
        "mediator_proxy_sensitivity": [0.8, 0.8],
        "mediator_proxy_specificity": [0.85, 0.85],
        "ascertainment": ascertainment,
        "proband_index": 0 if ascertainment != "population_unconditioned" else None,
    }


def qmc_family() -> dict[str, object]:
    """Return the three-discrete-dimension numerical sentinel."""
    return {
        "relationship": [[1.0, 0.5], [0.5, 1.0]],
        "latent_mean": [0.0, 0.0, 0.0, 0.0],
        "mediator_measurement": [None, None],
        "mediator_measurement_error_variance": [None, None],
        "mediator_proxy_status": [1, None],
        "outcome_status": [1, 0],
        "mediator_threshold": [0.0, 0.0],
        "outcome_threshold": [0.4, 0.4],
        "mediator_proxy_sensitivity": [0.8, 0.8],
        "mediator_proxy_specificity": [0.85, 0.85],
        "ascertainment": "population_unconditioned",
        "proband_index": None,
    }


def public_evaluate(
    family: dict[str, object],
    parameters: dict[str, float],
    qmc_points: int = 512,
) -> dict[str, object]:
    """Call the public interface, with no internal Asterism import."""
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [family], qmc_points=qmc_points
    )
    """Prepared the public latent mediation model for one reference family."""

    return model.evaluate(**parameters)


def require_close(label: str, actual: float, expected: float, tolerance: float) -> None:
    """Raise a compact numerical comparison failure."""
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"{label}: actual={actual:.17g}, expected={expected:.17g}, "
            f"absolute_gap={abs(actual - expected):.3g}, tolerance={tolerance:.3g}"
        )


def check_continuous_underflow(parameters: dict[str, float]) -> float:
    """Pin the log-scale continuous density at a point where ordinary scale vanishes."""
    family: dict[str, object] = singleton()
    """Constructed the mixed-observation singleton family template."""

    family.update(
        {
            "mediator_measurement": [100.0],
            "mediator_measurement_error_variance": [0.15],
            "mediator_proxy_status": [None],
            "outcome_status": [None],
        }
    )
    record: dict[str, object] = public_evaluate(family, parameters)
    """Evaluated the extreme continuous measurement through the public model."""

    require_close(
        "continuous underflow public constant",
        float(record["log_likelihood"]),
        UNDERFLOW_LOG_LIKELIHOOD,
        1.0e-9,
    )
    independent: float = independent_log_likelihood(family, parameters)
    """Evaluated the same extreme measurement through independent equations."""

    require_close(
        "continuous underflow independent equation",
        independent,
        UNDERFLOW_LOG_LIKELIHOOD,
        1.0e-9,
    )
    if record["ordinary_scale_representable"] is not False:
        raise AssertionError(
            "continuous underflow was unexpectedly representable on ordinary scale"
        )
    return float(record["log_likelihood"])


def check_low_dimensional_cases(parameters: dict[str, float]) -> dict[str, float]:
    """Compare independent low-dimensional reference calculations."""
    observed: dict[str, float] = {}
    """Initialised public log likelihoods retained in the comparison report."""

    named_points: dict[str, dict[str, float]] = {
        "interior": parameters,
        "a_zero": {**parameters, "a": 0.0},
        "b_zero": {**parameters, "b": 0.0},
        "d_zero": {**parameters, "d": 0.0},
        "negative_b": {**parameters, "b": -0.3},
    }
    """Selected interior, boundary and negative-path parameter sentinels."""

    for condition in ("population_unconditioned", "condition_on_named_proband_case"):
        family: dict[str, object] = singleton(condition)
        """Constructed a singleton under the current ascertainment condition."""

        for name, point in named_points.items():
            record: dict[str, object] = public_evaluate(family, point)
            """Evaluated the current low-dimensional case through the public model."""

            expected: float = independent_log_likelihood(family, point)
            """Evaluated the same case through the independent likelihood equations."""

            label: str = f"{condition} singleton {name}"
            """Named the ascertainment and parameter point in failure diagnostics."""

            require_close(
                label,
                float(record["log_likelihood"]),
                expected,
                LOW_DIMENSIONAL_ABSOLUTE_TOLERANCE,
            )
            observed[f"singleton:{condition}:{name}"] = float(record["log_likelihood"])
            """Recorded the agreed public singleton log likelihood."""

        dyad: dict[str, object] = related_dyad(condition)
        """Constructed a related outcome dyad under the same ascertainment condition."""

        record = public_evaluate(dyad, parameters)
        """Evaluated the related dyad through the public model."""

        expected = independent_log_likelihood(dyad, parameters)
        """Evaluated the related dyad through the independent likelihood equations."""

        label = f"{condition} related outcome dyad"
        """Named the related-dyad comparison in failure diagnostics."""

        require_close(
            label,
            float(record["log_likelihood"]),
            expected,
            LOW_DIMENSIONAL_ABSOLUTE_TOLERANCE,
        )
        observed[f"dyad:{condition}"] = float(record["log_likelihood"])
        """Recorded the agreed public dyad log likelihood."""
    return observed


def check_central_differences(parameters: dict[str, float]) -> dict[str, float]:
    """Compare independent and public central derivatives for all parameters."""
    family: dict[str, object] = singleton()
    """Constructed the singleton used for deterministic derivative checks."""

    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [family], qmc_points=512
    )
    """Prepared the public model at a deterministic integration setting."""

    derivatives: dict[str, float] = {}
    """Initialised central derivatives indexed by mediation parameter."""

    for name in ("a", "b", "c_prime", "d", "sigma_m2"):
        step: float = 1.0e-5
        """Set the symmetric finite-difference perturbation for this parameter."""

        lower: dict[str, float] = dict(parameters)
        """Copied parameters for the negative central perturbation."""

        upper: dict[str, float] = dict(parameters)
        """Copied parameters for the positive central perturbation."""

        lower[name] -= step
        """Applied the negative central perturbation to the current parameter."""

        upper[name] += step
        """Applied the positive central perturbation to the current parameter."""

        public_derivative: float = (
            float(model.evaluate(**upper)["log_likelihood"])
            - float(model.evaluate(**lower)["log_likelihood"])
        ) / (2.0 * step)
        """Calculated the public model's symmetric likelihood derivative."""

        independent_derivative: float = (
            independent_log_likelihood(family, upper)
            - independent_log_likelihood(family, lower)
        ) / (2.0 * step)
        """Calculated the independent likelihood's symmetric derivative."""

        require_close(
            f"central derivative {name}",
            public_derivative,
            independent_derivative,
            CENTRAL_DIFFERENCE_TOLERANCE,
        )
        derivatives[name] = public_derivative
        """Recorded the agreed public central derivative."""
    return derivatives


def check_qmc(parameters: dict[str, float]) -> float:
    """Compare the deterministic QMC path with an independent trivariate value."""
    family: dict[str, object] = qmc_family()
    """Constructed the three-discrete-coordinate numerical sentinel."""

    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [family], qmc_points=8192
    )
    """Prepared the public model at the measured QMC comparison budget."""

    first: float = float(model.evaluate(**parameters)["log_likelihood"])
    """Evaluated the first deterministic public QMC log likelihood."""

    second: float = float(model.evaluate(**parameters)["log_likelihood"])
    """Repeated the identical public QMC evaluation."""

    require_close("QMC deterministic repeat", second, first, DETERMINISTIC_TOLERANCE)
    independent: float = independent_log_likelihood(family, parameters)
    """Evaluated the trivariate case through independent conditional quadrature."""

    require_close(
        "QMC against the independent trivariate reference",
        first,
        independent,
        QMC_LOG_LIKELIHOOD_TOLERANCE,
    )
    return first


def main() -> int:
    """Run the public-interface checks and print a numerical report."""
    parameters: dict[str, float] = {
        "a": 0.5,
        "b": 0.3,
        "c_prime": 0.2,
        "d": 0.6,
        "sigma_m2": 0.7,
    }
    """Defined the interior mediation parameter point shared by numerical checks."""

    try:
        underflow: float = check_continuous_underflow(parameters)
        """Checked log-scale stability where the ordinary continuous density vanishes."""

        low_dimensional_cases: dict[str, float] = check_low_dimensional_cases(
            parameters
        )
        """Checked independently evaluable low-dimensional likelihood cases."""

        derivatives: dict[str, float] = check_central_differences(parameters)
        """Checked public likelihood derivatives against independent equations."""

        qmc: float = check_qmc(parameters)
        """Checked deterministic QMC against independent trivariate quadrature."""

    except Exception as error:
        print(
            f"latent mediation numerical comparison failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print("Latent mediation numerical comparison")
    print(f"  continuous underflow log likelihood: {underflow}")
    print(f"  low-dimensional comparisons: {len(low_dimensional_cases)}")
    print("  central-difference parameters: " + ", ".join(sorted(derivatives)))
    print(f"  QMC-8192 log likelihood, against the independent trivariate: {qmc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
