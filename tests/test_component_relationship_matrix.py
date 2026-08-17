"""Numerical matrix construction at the public ``ComponentModel`` boundary."""

from __future__ import annotations

import asterism
import numpy as np


def _paired_gene_matrix(
    pairs: int,
    *,
    multiplier: float = 1.0,
) -> np.ndarray:
    genotypes = np.zeros((2 * pairs, pairs), dtype=np.float64)
    for pair in range(pairs):
        genotypes[2 * pair : 2 * pair + 2, pair] = 1.0
    weighted = genotypes * multiplier
    return weighted @ weighted.T


def _four_person_group_matrix(subjects: int) -> np.ndarray:
    matrix = np.zeros((subjects, subjects), dtype=np.float64)
    for start in range(0, subjects, 4):
        matrix[start : start + 4, start : start + 4] = 1.0
    return matrix


def test_builder_output_is_an_ordinary_matrix_accepted_by_component_model():
    pairs = 20
    matrix = _paired_gene_matrix(pairs)
    rng = np.random.default_rng(4101)
    y = np.repeat(rng.standard_normal(pairs), 2) + rng.standard_normal(2 * pairs)

    record = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).fit(y)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2 * pairs, 2 * pairs)
    assert len(record["variances"]) == 2


def test_fit_contains_only_numerical_model_fields():
    pairs = 20
    matrix = _paired_gene_matrix(pairs)
    rng = np.random.default_rng(4102)
    y = np.repeat(rng.standard_normal(pairs), 2) + rng.standard_normal(2 * pairs)

    record = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).fit(y)

    assert set(record) == {
        "variances",
        "raw_coefficient_proportions",
        "raw_coefficient_total",
        "fixed_effects",
        "loglik",
        "scaled_gradient",
        "converged",
        "estimator",
        "mean_diagonal_component_contributions",
        "mean_diagonal_total",
        "mean_diagonal_proportions",
    }


def test_fit_uses_explicit_scale_dependent_coefficient_names():
    pairs = 20
    matrix = _paired_gene_matrix(pairs)
    rng = np.random.default_rng(4103)
    y = np.repeat(rng.standard_normal(pairs), 2) + rng.standard_normal(2 * pairs)

    record = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).fit(y)

    assert len(record["raw_coefficient_proportions"]) == 2
    assert np.isclose(sum(record["raw_coefficient_proportions"]), 1.0)
    assert "shares" not in record
    assert "total_variance" not in record


def test_component_interval_names_its_scale_dependent_quantity():
    pairs = 20
    matrix = _paired_gene_matrix(pairs)
    rng = np.random.default_rng(4104)
    y = np.repeat(rng.standard_normal(pairs), 2) + rng.standard_normal(2 * pairs)

    interval = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).interval(y, component=0)

    assert interval["quantity"] == "raw_coefficient_proportion"


def test_mean_diagonal_contributions_are_invariant_to_matrix_scaling():
    pairs = 120
    matrix = _paired_gene_matrix(pairs)
    ten_times_matrix = _paired_gene_matrix(pairs, multiplier=np.sqrt(10.0))
    rng = np.random.default_rng(4105)
    y = (
        np.sqrt(0.7) * np.repeat(rng.standard_normal(pairs), 2)
        + np.sqrt(0.3) * rng.standard_normal(2 * pairs)
    )
    design = np.ones((2 * pairs, 1))

    original = asterism.ComponentModel([matrix], design).fit(y)
    rescaled = asterism.ComponentModel([ten_times_matrix], design).fit(y)

    expected = [
        original["variances"][0] * np.mean(np.diag(matrix)),
        original["variances"][1],
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


def test_component_matrices_are_copied_before_caller_mutation():
    pairs = 40
    subjects = 2 * pairs
    gene = _paired_gene_matrix(pairs)
    group = _four_person_group_matrix(subjects)
    original_group = group.copy()
    design = np.ones((subjects, 1))
    model = asterism.ComponentModel([gene, group], design)
    control = asterism.ComponentModel([gene, original_group], design)
    rng = np.random.default_rng(4106)
    y = (
        np.repeat(rng.standard_normal(pairs), 2)
        + np.repeat(rng.standard_normal(subjects // 4), 4)
        + rng.standard_normal(subjects)
    )

    group[:] = np.nan
    copied = model.fit(y)
    expected = control.fit(y)

    np.testing.assert_allclose(copied["variances"], expected["variances"])
    np.testing.assert_allclose(
        copied["mean_diagonal_component_contributions"],
        expected["mean_diagonal_component_contributions"],
    )


def test_component_design_is_copied_before_caller_mutation():
    pairs = 40
    subjects = 2 * pairs
    matrix = _paired_gene_matrix(pairs)
    covariate = np.linspace(-1.0, 1.0, subjects)
    design = np.column_stack([np.ones(subjects), covariate])
    original_design = design.copy()
    model = asterism.ComponentModel([matrix], design)
    control = asterism.ComponentModel([matrix], original_design)
    rng = np.random.default_rng(4107)
    y = (
        0.6 * covariate
        + np.repeat(rng.standard_normal(pairs), 2)
        + rng.standard_normal(subjects)
    )

    design[:] = np.nan
    copied = model.fit(y)
    expected = control.fit(y)

    np.testing.assert_allclose(copied["variances"], expected["variances"])
    assert copied["fixed_effects"] == expected["fixed_effects"]
    assert copied["loglik"] == expected["loglik"]
