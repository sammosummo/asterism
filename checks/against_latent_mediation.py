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
from itertools import product
from pathlib import Path
from typing import Any


LOW_DIMENSIONAL_ABSOLUTE_TOLERANCE = 2.0e-8
CENTRAL_DIFFERENCE_TOLERANCE = 3.0e-6
DETERMINISTIC_TOLERANCE = 0.0
QMC_LOG_LIKELIHOOD_TOLERANCE = 5.0e-4
UNDERFLOW_LOG_LIKELIHOOD = -4546.421139077653
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def enforce_reference_independence() -> None:
    """Refuse imports that would collapse this check into a copied calculation."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    forbidden_prefixes = (
        "numpy",
        "scipy",
        "mediation",
        "combined_family_likelihood",
        "asterism.",
    )
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            if name.startswith(forbidden_prefixes):
                raise RuntimeError(
                    "reference-independence guard rejected forbidden import "
                    f"{name!r}"
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


def _simpson(function: Any, left: float, right: float) -> float:
    middle = 0.5 * (left + right)
    return (right - left) * (
        function(left) + 4.0 * function(middle) + function(right)
    ) / 6.0


def _adaptive_simpson(
    function: Any,
    left: float,
    right: float,
    whole: float,
    tolerance: float,
    depth: int,
) -> float:
    middle = 0.5 * (left + right)
    left_part = _simpson(function, left, middle)
    right_part = _simpson(function, middle, right)
    correction = left_part + right_part - whole
    if depth == 0 or abs(correction) <= 15.0 * tolerance:
        return left_part + right_part + correction / 15.0
    return _adaptive_simpson(
        function, left, middle, left_part, tolerance / 2.0, depth - 1
    ) + _adaptive_simpson(
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
    scale = math.sqrt(1.0 - correlation * correlation)
    upper = min(first, 10.0)
    if upper <= -10.0:
        return 0.0

    def integrand(value: float) -> float:
        standardised = (second - correlation * value) / scale
        return normal_density(value) * normal_cdf(standardised)

    whole = _simpson(integrand, -10.0, upper)
    return min(
        1.0,
        max(0.0, _adaptive_simpson(integrand, -10.0, upper, whole, 2.0e-13, 30)),
    )


def rectangle_probability(
    mean: list[float],
    covariance: list[list[float]],
    lower: list[float],
    upper: list[float],
) -> float:
    """Probability of a one- or two-dimensional Gaussian rectangle."""
    dimension = len(mean)
    if dimension == 0:
        return 1.0
    if dimension == 1:
        scale = math.sqrt(covariance[0][0])
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
        cross = [
            covariance[1][0] / covariance[0][0],
            covariance[2][0] / covariance[0][0],
        ]
        conditional = [
            [
                covariance[1][1] - cross[0] * covariance[0][1],
                covariance[1][2] - cross[0] * covariance[0][2],
            ],
            [
                covariance[2][1] - cross[1] * covariance[0][1],
                covariance[2][2] - cross[1] * covariance[0][2],
            ],
        ]

        def integrand(value: float) -> float:
            shifted_mean = [
                mean[1] + cross[0] * (value - mean[0]),
                mean[2] + cross[1] * (value - mean[0]),
            ]
            return (
                normal_density((value - mean[0]) / scale) / scale
            ) * rectangle_probability(
                shifted_mean, conditional, lower[1:], upper[1:]
            )

        low = max(lower[0], mean[0] - 9.0 * scale)
        high = min(upper[0], mean[0] + 9.0 * scale)
        if low >= high:
            return 0.0
        whole = _simpson(integrand, low, high)
        return min(
            1.0,
            max(0.0, _adaptive_simpson(integrand, low, high, whole, 1.0e-12, 28)),
        )
    if dimension > 3:
        raise ValueError("this independent exact check admits at most three dimensions")
    first_scale = math.sqrt(covariance[0][0])
    second_scale = math.sqrt(covariance[1][1])
    correlation = covariance[0][1] / (first_scale * second_scale)

    def cdf(x_value: float, y_value: float) -> float:
        return bivariate_cdf(
            (x_value - mean[0]) / first_scale,
            (y_value - mean[1]) / second_scale,
            correlation,
        )

    value = cdf(upper[0], upper[1]) - cdf(lower[0], upper[1])
    value -= cdf(upper[0], lower[1])
    value += cdf(lower[0], lower[1])
    return max(0.0, value)


def cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """Cholesky factor for the tiny positive-definite matrices used here."""
    size = len(matrix)
    factor = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            value = matrix[row][column] - sum(
                factor[row][index] * factor[column][index] for index in range(column)
            )
            if row == column:
                if value <= 0.0:
                    raise ValueError("independent covariance was not positive definite")
                factor[row][column] = math.sqrt(value)
            else:
                factor[row][column] = value / factor[column][column]
    return factor


def solve_cholesky(factor: list[list[float]], right: list[float]) -> list[float]:
    """Solve ``L L' x = right`` without a third-party linear-algebra package."""
    size = len(factor)
    forward = [0.0] * size
    for row in range(size):
        forward[row] = (
            right[row]
            - sum(
                factor[row][column] * forward[column]
                for column in range(row)
            )
        ) / factor[row][row]
    answer = [0.0] * size
    for row in range(size - 1, -1, -1):
        answer[row] = (
            forward[row]
            - sum(
                factor[column][row] * answer[column]
                for column in range(row + 1, size)
            )
        ) / factor[row][row]
    return answer


def structural_covariance(
    family: dict[str, Any], parameters: dict[str, float]
) -> list[list[float]]:
    """Re-derive the latent mediation process-major covariance matrix."""
    relationship = family["relationship"]
    people = len(relationship)
    a = parameters["a"]
    b = parameters["b"]
    c_prime = parameters["c_prime"]
    d = parameters["d"]
    sigma_m2 = parameters["sigma_m2"]
    total_inherited_mediator = a * b + c_prime
    covariance = [[0.0 for _ in range(2 * people)] for _ in range(2 * people)]
    for first in range(people):
        for second in range(people):
            relation = float(relationship[first][second])
            residual = 1.0 if first == second else 0.0
            mediator_covariance = a * a * relation + sigma_m2 * residual
            cross_covariance = (
                a * total_inherited_mediator * relation + b * sigma_m2 * residual
            )
            outcome_covariance = (
                (total_inherited_mediator * total_inherited_mediator + d * d)
                * relation
                + (b * b * sigma_m2 + 1.0) * residual
            )
            covariance[first][second] = mediator_covariance
            covariance[first][people + second] = cross_covariance
            covariance[people + first][second] = cross_covariance
            covariance[people + first][people + second] = outcome_covariance
    return covariance


def continuous_conditioning(
    family: dict[str, Any], covariance: list[list[float]]
) -> tuple[list[int], list[float], list[list[float]], float]:
    """Condition latent observations on noisy mediator measurements."""
    people = len(family["relationship"])
    means = [float(value) for value in family["latent_mean"]]
    observed_people = [
        person
        for person, value in enumerate(family["mediator_measurement"])
        if value is not None
    ]
    if not observed_people:
        return [], means, covariance, 0.0
    observed = observed_people
    values = [
        float(family["mediator_measurement"][person])
        for person in observed_people
    ]
    errors = [
        float(family["mediator_measurement_error_variance"][person])
        for person in observed_people
    ]
    observed_covariance = [
        [
            covariance[left][right] + (errors[row] if row == column else 0.0)
            for column, right in enumerate(observed)
        ]
        for row, left in enumerate(observed)
    ]
    factor = cholesky(observed_covariance)
    residual = [
        values[index] - means[observed[index]]
        for index in range(len(observed))
    ]
    solved_residual = solve_cholesky(factor, residual)
    log_density = -0.5 * (
        len(observed) * math.log(2.0 * math.pi)
        + 2.0 * sum(math.log(factor[index][index]) for index in range(len(observed)))
        + sum(
            residual[index] * solved_residual[index]
            for index in range(len(observed))
        )
    )
    cross = [
        [covariance[row][column] for column in observed]
        for row in range(2 * people)
    ]
    conditioned_mean = [
        means[row]
        + sum(
            cross[row][column] * solved_residual[column]
            for column in range(len(observed))
        )
        for row in range(2 * people)
    ]
    inverse_cross = [solve_cholesky(factor, cross[row]) for row in range(2 * people)]
    conditioned_covariance = [
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
    return observed, conditioned_mean, conditioned_covariance, log_density


def independent_log_likelihood(
    family: dict[str, Any], parameters: dict[str, float]
) -> float:
    """Evaluate one family from the public model equations, independently."""
    people = len(family["relationship"])
    covariance = structural_covariance(family, parameters)
    _, mean, covariance, continuous_log_density = continuous_conditioning(
        family, covariance
    )
    proxy_people = [
        person
        for person, value in enumerate(family["mediator_proxy_status"])
        if value is not None
    ]
    outcome_people = [
        person
        for person, value in enumerate(family["outcome_status"])
        if value is not None
    ]
    dimensions = proxy_people + [people + person for person in outcome_people]
    if not dimensions:
        probability = 1.0
    else:
        selected_mean = [mean[index] for index in dimensions]
        selected_covariance = [
            [covariance[row][column] for column in dimensions] for row in dimensions
        ]
        probability = 0.0
        for truths in product((0, 1), repeat=len(proxy_people)):
            lower: list[float] = []
            upper: list[float] = []
            observation_probability = 1.0
            for position, person in enumerate(proxy_people):
                truth = truths[position]
                threshold = float(family["mediator_threshold"][person])
                lower.append(threshold if truth else -math.inf)
                upper.append(math.inf if truth else threshold)
                status = int(family["mediator_proxy_status"][person])
                sensitivity = float(
                    family["mediator_proxy_sensitivity"][person]
                )
                specificity = float(
                    family["mediator_proxy_specificity"][person]
                )
                if status == 1:
                    observation_probability *= (
                        sensitivity if truth else 1.0 - specificity
                    )
                else:
                    observation_probability *= (
                        1.0 - sensitivity if truth else specificity
                    )
            for person in outcome_people:
                threshold = float(family["outcome_threshold"][person])
                if int(family["outcome_status"][person]) == 1:
                    lower.append(threshold)
                    upper.append(math.inf)
                else:
                    lower.append(-math.inf)
                    upper.append(threshold)
            probability += observation_probability * rectangle_probability(
                selected_mean, selected_covariance, lower, upper
            )
    if probability <= 0.0:
        raise ValueError(
            "independent likelihood underflowed in a finite reference check"
        )
    value = continuous_log_density + math.log(probability)
    if family["ascertainment"] == "condition_on_named_proband_case":
        proband = int(family["proband_index"])
        outcome_index = people + proband
        threshold = float(family["outcome_threshold"][proband])
        denominator = 1.0 - normal_cdf(
            (threshold - float(family["latent_mean"][outcome_index]))
            / math.sqrt(
                structural_covariance(family, parameters)[outcome_index][
                    outcome_index
                ]
            )
        )
        value -= math.log(denominator)
    return value


def singleton(ascertainment: str = "population_unconditioned") -> dict[str, Any]:
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


def related_dyad(ascertainment: str) -> dict[str, Any]:
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


def qmc_family() -> dict[str, Any]:
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
    family: dict[str, Any],
    parameters: dict[str, float],
    qmc_points: int = 512,
) -> dict[str, Any]:
    """Call the public interface, with no internal Asterism import."""
    model = asterism.LatentMediationModel([family], qmc_points=qmc_points)
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
    family = singleton()
    family.update(
        {
            "mediator_measurement": [100.0],
            "mediator_measurement_error_variance": [0.15],
            "mediator_proxy_status": [None],
            "outcome_status": [None],
        }
    )
    record = public_evaluate(family, parameters)
    require_close(
        "continuous underflow public constant",
        float(record["log_likelihood"]),
        UNDERFLOW_LOG_LIKELIHOOD,
        1.0e-9,
    )
    independent = independent_log_likelihood(family, parameters)
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
    named_points = {
        "interior": parameters,
        "a_zero": {**parameters, "a": 0.0},
        "b_zero": {**parameters, "b": 0.0},
        "d_zero": {**parameters, "d": 0.0},
        "negative_b": {**parameters, "b": -0.3},
    }
    for condition in ("population_unconditioned", "condition_on_named_proband_case"):
        family = singleton(condition)
        for name, point in named_points.items():
            record = public_evaluate(family, point)
            expected = independent_log_likelihood(family, point)
            label = f"{condition} singleton {name}"
            require_close(
                label,
                float(record["log_likelihood"]),
                expected,
                LOW_DIMENSIONAL_ABSOLUTE_TOLERANCE,
            )
            observed[f"singleton:{condition}:{name}"] = float(record["log_likelihood"])
        dyad = related_dyad(condition)
        record = public_evaluate(dyad, parameters)
        expected = independent_log_likelihood(dyad, parameters)
        label = f"{condition} related outcome dyad"
        require_close(
            label,
            float(record["log_likelihood"]),
            expected,
            LOW_DIMENSIONAL_ABSOLUTE_TOLERANCE,
        )
        observed[f"dyad:{condition}"] = float(record["log_likelihood"])
    return observed


def check_central_differences(parameters: dict[str, float]) -> dict[str, float]:
    """Compare independent and public central derivatives for all parameters."""
    family = singleton()
    model = asterism.LatentMediationModel([family], qmc_points=512)
    derivatives: dict[str, float] = {}
    for name in ("a", "b", "c_prime", "d", "sigma_m2"):
        step = 1.0e-5
        lower = dict(parameters)
        upper = dict(parameters)
        lower[name] -= step
        upper[name] += step
        public_derivative = (
            float(model.evaluate(**upper)["log_likelihood"])
            - float(model.evaluate(**lower)["log_likelihood"])
        ) / (2.0 * step)
        independent_derivative = (
            independent_log_likelihood(family, upper)
            - independent_log_likelihood(family, lower)
        ) / (2.0 * step)
        require_close(
            f"central derivative {name}",
            public_derivative,
            independent_derivative,
            CENTRAL_DIFFERENCE_TOLERANCE,
        )
        derivatives[name] = public_derivative
    return derivatives


def check_qmc(parameters: dict[str, float]) -> float:
    """Compare the deterministic QMC path with an independent trivariate value."""
    family = qmc_family()
    model = asterism.LatentMediationModel([family], qmc_points=8192)
    first = float(model.evaluate(**parameters)["log_likelihood"])
    second = float(model.evaluate(**parameters)["log_likelihood"])
    require_close("QMC deterministic repeat", second, first, DETERMINISTIC_TOLERANCE)
    independent = independent_log_likelihood(family, parameters)
    require_close(
        "QMC against the independent trivariate reference",
        first,
        independent,
        QMC_LOG_LIKELIHOOD_TOLERANCE,
    )
    return first


def main() -> int:
    """Run the public-interface checks and print a numerical report."""
    parameters = {
        "a": 0.5,
        "b": 0.3,
        "c_prime": 0.2,
        "d": 0.6,
        "sigma_m2": 0.7,
    }
    try:
        underflow = check_continuous_underflow(parameters)
        low_dimensional_cases = check_low_dimensional_cases(parameters)
        derivatives = check_central_differences(parameters)
        qmc = check_qmc(parameters)
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
    print(
        "  central-difference parameters: "
        + ", ".join(sorted(derivatives))
    )
    print(f"  QMC-8192 log likelihood, against the independent trivariate: {qmc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
