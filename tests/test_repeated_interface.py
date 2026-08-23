"""The repeated-measures model through Python: shapes, statuses and refusals.

The mathematics is checked in Rust, against `ComponentModel`, `TobitModel` and
`MixedBivariateModel`. What is checked here is the translation: that the arrays
come back the shape they should, that the four censoring codes mean what the
docstring says, and that a caller who gets the inputs wrong is told so rather
than given a number.
"""

from __future__ import annotations

import asterism
import numpy as np
import pytest

FAMILIES = 40
REPLICATES = 2
POSITIONS = 3
LINE = np.array([0.0, 3.0, 8.0])


def relationship(families: int = FAMILIES) -> np.ndarray:
    """Nuclear families of two parents and two children."""
    people = families * 4
    matrix = np.eye(people)
    for family in range(families):
        base = family * 4
        for child in (2, 3):
            for parent in (0, 1):
                matrix[base + child, base + parent] = 0.5
                matrix[base + parent, base + child] = 0.5
        matrix[base + 2, base + 3] = 0.5
        matrix[base + 3, base + 2] = 0.5
    return matrix


def simulate(seed: int = 20_260_820) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A relationship matrix, a design and a response from the model."""
    rng = np.random.default_rng(seed)
    a = relationship()
    people = a.shape[0]
    rows = people * REPLICATES
    across = np.linalg.cholesky(a)
    effects = across @ rng.standard_normal((people, POSITIONS)) * np.sqrt(0.5)
    design = np.column_stack([np.ones(rows), rng.standard_normal(rows)])
    value = np.zeros((rows, POSITIONS))
    for person in range(people):
        for replicate in range(REPLICATES):
            row = person * REPLICATES + replicate
            value[row] = (
                10.0
                + effects[person]
                + np.sqrt(0.5) * rng.standard_normal(POSITIONS)
            )
    return a, design, value


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return simulate()


def measured(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros_like(value, dtype=np.int64), np.zeros_like(value)


def test_the_record_comes_back_the_shape_it_should(data):
    a, design, value = data
    model = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS)
    censoring, limit = measured(value)
    fit = model.fit(value, censoring, limit)

    assert len(fit["component_covariances"]) == 1
    assert fit["component_covariances"][0].shape == (POSITIONS, POSITIONS)
    assert fit["residual_covariance"].shape == (POSITIONS, POSITIONS)
    assert fit["fixed_effects"].shape == (design.shape[1], POSITIONS)
    assert fit["variance_shares"].shape == (POSITIONS, 2)
    assert fit["censored_shares"].shape == (POSITIONS,)
    # Every covariance is symmetric, which is the one property the flattening
    # across the boundary could silently break.
    for matrix in [*fit["component_covariances"], fit["residual_covariance"]]:
        assert np.allclose(matrix, matrix.T, atol=1e-12)
    assert fit["estimator"] == "ml"
    assert fit["sequential_dimension"] == 0
    assert np.allclose(fit["variance_shares"].sum(axis=1), 1.0)
    # No line was given, so there is no kernel and nothing to read from one.
    assert fit["floors"].size == 0
    with pytest.raises(ValueError, match="REPEATED_NO_KERNEL"):
        asterism.RepeatedModel.correlation(fit, 0, 1.0)


def test_a_line_gives_a_correlation_that_is_a_function(data):
    a, design, value = data
    model = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS, line=LINE)
    censoring, limit = measured(value)
    fit = model.fit(value, censoring, limit)

    assert fit["floors"].shape == (2,)
    assert fit["rates"].shape == (2,)
    at_nothing = asterism.RepeatedModel.correlation(fit, 0, 0.0)
    assert at_nothing == pytest.approx(1.0)
    # It decays, and it never falls below its own floor.
    near = asterism.RepeatedModel.correlation(fit, 0, 3.0)
    far = asterism.RepeatedModel.correlation(fit, 0, 30.0)
    assert near > far >= float(fit["floors"][0]) - 1e-12
    # The sign of the separation cannot matter.
    assert asterism.RepeatedModel.correlation(
        fit, 0, -3.0
    ) == pytest.approx(near)
    with pytest.raises(ValueError, match="REPEATED_NO_SUCH_COMPONENT"):
        asterism.RepeatedModel.correlation(fit, 5, 1.0)


def test_a_value_never_measured_is_imputed_rather_than_read(data):
    """Status 3 must ignore whatever is in ``value``.

    The value array is dense and something has to be in every cell of it. If a
    cell the status calls unmeasured were read anyway, a caller filling those
    with nought would get a different answer from one filling them with a
    thousand, and neither would be told.
    """
    a, design, value = data
    model = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS)
    censoring, limit = measured(value)
    censoring[::7, 1] = 3

    tidy = value.copy()
    tidy[censoring == 3] = 0.0
    absurd = value.copy()
    absurd[censoring == 3] = 1.0e6

    first = model.fit(tidy, censoring, limit)
    second = model.fit(absurd, censoring, limit)
    assert first["loglik"] == second["loglik"]
    assert np.allclose(
        first["component_covariances"][0], second["component_covariances"][0]
    )


def test_a_censored_value_reads_the_limit_and_not_the_value(data):
    """Status 1 must ignore ``value`` and read ``limit``, for the same reason."""
    a, design, value = data
    model = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS)
    censoring, limit = measured(value)
    censoring[::5, 2] = 1
    limit[censoring == 1] = value[censoring == 1] - 0.4

    tidy = value.copy()
    tidy[censoring == 1] = 0.0
    absurd = value.copy()
    absurd[censoring == 1] = -1.0e6

    first = model.fit(tidy, censoring, limit)
    second = model.fit(absurd, censoring, limit)
    assert first["loglik"] == second["loglik"]
    assert first["censored_shares"][2] == pytest.approx(1.0 / 5.0, abs=0.01)
    assert first["sequential_dimension"] > 0


def test_bad_inputs_are_refused(data):
    a, design, value = data
    model = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS)
    censoring, limit = measured(value)

    with pytest.raises(ValueError, match="REPEATED_CENSORING_CODE_UNKNOWN"):
        wrong = censoring.copy()
        wrong[0, 0] = 9
        model.fit(value, wrong, limit)
    with pytest.raises(ValueError, match="REPEATED_INPUTS_DIFFERENT_SHAPES"):
        model.fit(value, censoring[:-1], limit)
    with pytest.raises(ValueError, match="REPEATED_RESPONSE_WRONG_SHAPE"):
        model.fit(value[:-2], censoring[:-2], limit[:-2])
    with pytest.raises(ValueError, match="REPEATED_LINE_WRONG_LENGTH"):
        asterism.RepeatedModel(a, design, REPLICATES, POSITIONS, line=[0.0, 1.0])
    with pytest.raises(ValueError, match="REPEATED_KERNEL_POSITIONS_COINCIDE"):
        asterism.RepeatedModel(
            a, design, REPLICATES, POSITIONS, line=[0.0, 1.0, 1.0]
        )
    with pytest.raises(ValueError, match="REPEATED_NEEDS_TWO_REPLICATES"):
        asterism.RepeatedModel(a, design, 1, POSITIONS)


def test_preparing_once_and_fitting_twice_gives_the_same_answer(data):
    """The class exists so the decomposition is paid for once; fitting twice
    through the same object must not depend on which fit came first."""
    a, design, value = data
    model = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS)
    censoring, limit = measured(value)
    first = model.fit(value, censoring, limit)
    other = value + 3.0
    model.fit(other, censoring, limit)
    again = model.fit(value, censoring, limit)
    assert first["loglik"] == again["loglik"]
    assert np.allclose(
        first["component_covariances"][0], again["component_covariances"][0]
    )


def test_the_correlation_interval_comes_back_and_brackets_its_estimate(data):
    """The record's shape and the one thing an interval must always do.

    Whether it covers is `checks/repeated_coverage.py`'s question, not this
    file's. What is checked here is the translation: that the keys are there,
    that the ends bracket the estimate, and that the ends of the range are what
    the kernel family can express rather than nought and one.
    """
    a, design, value = data
    model = asterism.RepeatedModel(
        a, design, REPLICATES, POSITIONS, line=LINE
    )
    censoring, limit = measured(value)
    got = model.correlation_interval(value, censoring, limit, 0, 3.0)

    assert got["component"] == 0
    assert got["separation"] == 3.0
    assert got["level"] == 0.95
    assert got["lower"] <= got["estimate"] <= got["upper"]
    assert 0.001 <= got["lower"] and got["upper"] <= 0.999
    assert isinstance(got["profile_failures"], int)
    # The estimate is the free fit's own correlation, not a separate number.
    free = model.fit(value, censoring, limit)
    assert got["estimate"] == pytest.approx(
        asterism.RepeatedModel.correlation(free, 0, 3.0), abs=1e-9
    )
    # A bound that was not reached asks no question about itself.
    if not got["lower_at_bound"]:
        assert got["contains_lower_bound"] is None
    if not got["upper_at_bound"]:
        assert got["contains_upper_bound"] is None


def test_an_interval_without_a_kernel_is_refused(data):
    """There is no curve to hold without a line, and saying so beats holding
    something else."""
    a, design, value = data
    censoring, limit = measured(value)
    plain = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS)
    with pytest.raises(ValueError, match="REPEATED_NO_KERNEL"):
        plain.correlation_interval(value, censoring, limit, 0, 3.0)

    shaped = asterism.RepeatedModel(
        a, design, REPLICATES, POSITIONS, line=LINE
    )
    with pytest.raises(ValueError, match="REPEATED_NO_SUCH_COMPONENT"):
        shaped.correlation_interval(value, censoring, limit, 7, 3.0)
    with pytest.raises(ValueError, match="REPEATED_SEPARATION_NOT_POSITIVE"):
        shaped.correlation_interval(value, censoring, limit, 0, 0.0)


def test_the_heritability_interval_comes_back_and_brackets_its_estimate(data):
    """The record's shape, and that its ends are the ones a covariance allows.

    Whether it covers is `checks/repeated_share_coverage.py`'s question. What is
    checked here is the translation, and one thing worth stating in a test
    rather than only in prose: the lower end is 0.001 and not nought, because a
    component with no variance at a position has a singular covariance there.
    """
    a, design, value = data
    model = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS, line=LINE)
    censoring, limit = measured(value)
    got = model.heritability_interval(value, censoring, limit, 0, 1)

    assert got["component"] == 0
    assert got["position"] == 1
    assert got["level"] == 0.95
    assert got["lower"] <= got["estimate"] <= got["upper"]
    assert 0.001 <= got["lower"] and got["upper"] <= 0.999
    assert isinstance(got["profile_failures"], int)
    # The estimate is the free fit's own share, not a separate number.
    free = model.fit(value, censoring, limit)
    assert got["estimate"] == pytest.approx(
        float(np.asarray(free["variance_shares"])[1][0]), abs=1e-9
    )

    with pytest.raises(ValueError, match="REPEATED_NO_SUCH_POSITION"):
        model.heritability_interval(value, censoring, limit, 0, 99)
    plain = asterism.RepeatedModel(a, design, REPLICATES, POSITIONS)
    with pytest.raises(ValueError, match="REPEATED_NO_KERNEL"):
        plain.heritability_interval(value, censoring, limit, 0, 1)
