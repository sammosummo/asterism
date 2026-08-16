"""Scale-invariant marginal reporting for the public spatial model."""

from __future__ import annotations

import numpy as np

import asterism


def _spatial_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = 40
    people = 2 * pairs
    relationship = np.eye(people)
    for pair in range(pairs):
        first, second = 2 * pair, 2 * pair + 1
        relationship[first, second] = relationship[second, first] = 0.5

    locations = np.repeat(np.arange(pairs, dtype=float) * 7.0, 2)
    distance = np.abs(locations[:, None] - locations[None, :])
    covariance = (
        0.45 * relationship
        + 0.25 * np.exp(-0.04 * distance)
        + 0.30 * np.eye(people)
    )
    response = np.random.default_rng(5601).multivariate_normal(
        np.zeros(people), covariance
    )
    return relationship, distance, np.ones((people, 1)), response


def test_spatial_fit_names_raw_coefficients_and_reports_invariant_contributions():
    relationship, distance, design, response = _spatial_problem()

    original = asterism.SpatialModel([relationship], distance, design).fit(response)
    rescaled = asterism.SpatialModel([7.0 * relationship], distance, design).fit(response)

    assert "shares" not in original
    assert "total_variance" not in original
    assert "raw_coefficient_proportions" in original
    assert "raw_coefficient_total" in original
    assert np.isclose(sum(original["raw_coefficient_proportions"]), 1.0)
    assert original["converged"] and rescaled["converged"]

    expected = [
        original["variances"][0] * np.mean(np.diag(relationship)),
        original["variances"][1],
        original["variances"][2],
    ]
    np.testing.assert_allclose(
        original["mean_diagonal_component_contributions"], expected
    )
    assert original["mean_diagonal_total"] == sum(
        original["mean_diagonal_component_contributions"]
    )

    assert not np.isclose(
        original["raw_coefficient_proportions"][0],
        rescaled["raw_coefficient_proportions"][0],
        rtol=0.1,
    )
    original_covariance = (
        original["variances"][0] * relationship
        + original["variances"][1]
        * np.exp(-original["decay_per_km"] * distance)
        + original["variances"][2] * np.eye(len(response))
    )
    rescaled_covariance = (
        rescaled["variances"][0] * (7.0 * relationship)
        + rescaled["variances"][1]
        * np.exp(-rescaled["decay_per_km"] * distance)
        + rescaled["variances"][2] * np.eye(len(response))
    )
    np.testing.assert_allclose(
        original_covariance, rescaled_covariance, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        original["loglik"], rescaled["loglik"], rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        original["mean_diagonal_component_contributions"],
        rescaled["mean_diagonal_component_contributions"],
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        original["mean_diagonal_proportions"],
        rescaled["mean_diagonal_proportions"],
        rtol=1e-5,
        atol=1e-7,
    )


def test_spatial_intervals_name_the_quantity_being_profiled():
    relationship, distance, design, response = _spatial_problem()
    model = asterism.SpatialModel([relationship], distance, design)

    assert model.interval(response, "0")["quantity"] == "raw_coefficient_proportion"
    assert model.interval(response, "lambda")["quantity"] == "decay_per_km"
