"""Scale-invariant marginal reporting for the public spatial model."""

from __future__ import annotations

import asterism
import numpy as np
import numpy.typing as npt


# asterism-style: allow private-helper -- shared deterministic fixture for two tests
def _spatial_problem() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Construct one deterministic paired-person spatial fitting problem."""
    pairs: int = 40
    """Selected enough relationship pairs to identify the covariance components."""

    people: int = 2 * pairs
    """Counted the two people contributed by every pair."""

    relationship: npt.NDArray[np.float64] = np.eye(people)
    """Started the additive relationship matrix at unrelated identity."""

    for pair in range(pairs):
        first, second = 2 * pair, 2 * pair + 1
        """Located the two people in the current relationship pair."""

        relationship[first, second] = relationship[second, first] = 0.5
        """Assigned the expected first-degree relationship within the pair."""
    """Constructed independent pairs with within-pair relatedness."""

    locations: npt.NDArray[np.float64] = np.repeat(
        np.arange(pairs, dtype=float) * 7.0, 2
    )
    """Placed each pair together at regularly separated locations."""

    distance: npt.NDArray[np.float64] = np.abs(locations[:, None] - locations[None, :])
    """Computed the absolute pairwise distances between people."""

    covariance: npt.NDArray[np.float64] = (
        0.45 * relationship + 0.25 * np.exp(-0.04 * distance) + 0.30 * np.eye(people)
    )
    """Combined additive, spatial, and residual covariance at known scales."""

    response: npt.NDArray[np.float64] = np.random.default_rng(5601).multivariate_normal(
        np.zeros(people), covariance
    )
    """Drew a deterministic response from the known covariance model."""
    return relationship, distance, np.ones((people, 1)), response


def test_spatial_fit_names_raw_coefficients_and_reports_invariant_contributions() -> (
    None
):
    """Keep marginal contributions invariant when a fixed matrix is rescaled."""
    relationship, distance, design, response = _spatial_problem()
    """Constructed one response with additive, spatial, and residual variation."""

    original: dict[str, object] = asterism.SpatialModel(
        [relationship], distance, design
    ).fit(response)
    """Fitted the spatial model on the relationship matrix's original scale."""

    rescaled: dict[str, object] = asterism.SpatialModel(
        [7.0 * relationship], distance, design
    ).fit(response)
    """Fitted the identical covariance after rescaling the relationship matrix."""

    assert "shares" not in original
    assert "total_variance" not in original
    assert "raw_coefficient_proportions" in original
    assert "raw_coefficient_total" in original
    assert np.isclose(sum(original["raw_coefficient_proportions"]), 1.0)
    assert original["converged"] and rescaled["converged"]

    expected: list[float] = [
        original["variances"][0] * np.mean(np.diag(relationship)),
        original["variances"][1],
        original["variances"][2],
    ]
    """Computed the marginal component contributions from the original fit."""

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
    original_covariance: npt.NDArray[np.float64] = (
        original["variances"][0] * relationship
        + original["variances"][1] * np.exp(-original["decay_per_km"] * distance)
        + original["variances"][2] * np.eye(len(response))
    )
    """Reconstructed covariance from coefficients on the original matrix scale."""

    rescaled_covariance: npt.NDArray[np.float64] = (
        rescaled["variances"][0] * (7.0 * relationship)
        + rescaled["variances"][1] * np.exp(-rescaled["decay_per_km"] * distance)
        + rescaled["variances"][2] * np.eye(len(response))
    )
    """Reconstructed covariance from reciprocal coefficients after rescaling."""

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


def test_spatial_intervals_name_the_quantity_being_profiled() -> None:
    """Distinguish raw coefficient profiles from spatial decay profiles."""
    relationship, distance, design, response = _spatial_problem()
    """Constructed the shared spatial fitting problem."""

    model: asterism.SpatialModel = asterism.SpatialModel(
        [relationship], distance, design
    )
    """Prepared the public spatial model used by both profile requests."""

    assert model.interval(response, "0")["quantity"] == "raw_coefficient_proportion"
    assert model.interval(response, "lambda")["quantity"] == "decay_per_km"


def test_spatial_bootstrap_completes_its_predeclared_null_reference() -> None:
    """Require the repaired boundary fit to complete every declared replicate."""
    people: int = 4
    """Selected the smallest deterministic fixture that exposed a false failure."""

    relationship: npt.NDArray[np.float64] = np.eye(people)
    """Started two independent sibling-pair relationship blocks."""

    relationship[0, 1] = relationship[1, 0] = 0.5
    """Completed the first relationship pair symmetrically."""

    relationship[2, 3] = relationship[3, 2] = 0.5
    """Completed the second relationship pair symmetrically."""

    locations: npt.NDArray[np.float64] = np.repeat(np.arange(2, dtype=float), 2)
    """Placed each sibling pair at one distinct synthetic location."""

    distance: npt.NDArray[np.float64] = np.abs(locations[:, None] - locations[None, :])
    """Built the exact deterministic pairwise distance matrix."""

    response: npt.NDArray[np.float64] = np.random.default_rng(1).normal(size=people)
    """Reproduced the response whose twentieth refit rested on a zero boundary."""

    model: asterism.SpatialModel = asterism.SpatialModel(
        [relationship], distance, np.ones((people, 1))
    )
    """Built the public spatial presence-test model."""

    result: dict[str, object] = model.bootstrap(response, replicates=20, seed=1001)
    """Ran the deterministic public bootstrap after repairing boundary convergence."""

    assert result["replicates"] == result["requested"] == 20
    assert result["rule"] == "parametric_bootstrap_add_one"
    assert result["smallest_p_value"] == 1.0 / 21.0
    """Required the exact predeclared denominator and its corresponding p-value floor."""
