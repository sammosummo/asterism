"""Numerical matrix construction at the public ``ComponentModel`` boundary."""

from __future__ import annotations

import asterism
import numpy as np
import numpy.typing as npt
import pytest


# asterism-style: allow private-helper -- shared variant-root fixture across tests
def _paired_gene_matrix(
    pairs: int,
    *,
    multiplier: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Build a low-rank matrix connecting disjoint subject pairs."""
    genotypes: npt.NDArray[np.float64] = np.zeros((2 * pairs, pairs), dtype=np.float64)
    """Initialised one synthetic genotype column per subject pair."""

    for pair in range(pairs):
        genotypes[2 * pair : 2 * pair + 2, pair] = 1.0
        """Assigned the current genotype exclusively to its subject pair."""
    """Constructed disjoint pair-specific synthetic genotypes."""

    weighted: npt.NDArray[np.float64] = genotypes * multiplier
    """Applied the requested root-matrix scale before forming covariance."""
    return weighted @ weighted.T


# asterism-style: allow private-helper -- shared grouped-covariance fixture
def _four_person_group_matrix(subjects: int) -> npt.NDArray[np.float64]:
    """Build a block indicator matrix for consecutive groups of four."""
    matrix: npt.NDArray[np.float64] = np.zeros((subjects, subjects), dtype=np.float64)
    """Initialised an empty subject-by-subject group covariance matrix."""

    for start in range(0, subjects, 4):
        matrix[start : start + 4, start : start + 4] = 1.0
        """Filled the current four-person membership block."""
    """Constructed independent shared-environment groups of four people."""
    return matrix


def test_builder_output_is_an_ordinary_matrix_accepted_by_component_model() -> None:
    """Accept a builder's numerical matrix directly at the component boundary."""
    pairs: int = 20
    """Selected enough paired genotypes for the public smoke fit."""

    matrix: npt.NDArray[np.float64] = _paired_gene_matrix(pairs)
    """Built the structured component from paired genotype roots."""

    rng: np.random.Generator = np.random.default_rng(4101)
    """Created a deterministic response generator."""

    y: npt.NDArray[np.float64] = np.repeat(
        rng.standard_normal(pairs), 2
    ) + rng.standard_normal(2 * pairs)
    """Combined pair-shared and independent response variation."""

    record: dict[str, object] = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).fit(y)
    """Fitted the builder matrix through the public component model."""

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2 * pairs, 2 * pairs)
    assert len(record["variances"]) == 2


def test_fit_contains_only_numerical_and_producer_identity_fields() -> None:
    """Keep caller-owned file and table metadata outside the fit record."""
    pairs: int = 20
    """Selected enough paired genotypes for the fit-record contract test."""

    matrix: npt.NDArray[np.float64] = _paired_gene_matrix(pairs)
    """Built the structured component from paired genotype roots."""

    rng: np.random.Generator = np.random.default_rng(4102)
    """Created a deterministic response generator."""

    y: npt.NDArray[np.float64] = np.repeat(
        rng.standard_normal(pairs), 2
    ) + rng.standard_normal(2 * pairs)
    """Combined pair-shared and independent response variation."""

    record: dict[str, object] = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).fit(y)
    """Produced the public numerical and producer-identity record under inspection."""

    assert set(record) == {
        "variances",
        "raw_coefficient_proportions",
        "raw_coefficient_total",
        "fixed_effects",
        "loglik",
        "scaled_gradient",
        "converged",
        "stop_code",
        "stop_message",
        "polished",
        "estimator",
        "mean_diagonal_component_contributions",
        "mean_diagonal_total",
        "mean_diagonal_proportions",
        "build",
        "subject_order_sha256",
    }


def test_the_fit_says_why_the_search_stopped_as_well_as_whether_it_converged() -> None:
    """The two are different questions and the answer to one is not the other.

    The search stops on `factr`, a relative reduction in the objective.
    `converged` is decided afterwards on the recomputed projected gradient.
    Down a flat valley the first happens well before the second, so a fit can
    stop cleanly and still be reported as not converged -- which is what three
    of the eight red deer fits do while landing within 0.006 of a published
    table. Without the search's own code the two cannot be told apart, and this
    holds it in place.
    """
    pairs: int = 20
    """Selected enough paired genotypes for a well-conditioned fit."""

    matrix: npt.NDArray[np.float64] = _paired_gene_matrix(pairs)
    """Built the structured component from paired genotype roots."""

    rng: np.random.Generator = np.random.default_rng(4104)
    """Created a deterministic response generator."""

    y: npt.NDArray[np.float64] = np.repeat(
        rng.standard_normal(pairs), 2
    ) + rng.standard_normal(2 * pairs)
    """Combined pair-shared and independent response variation."""

    record: dict[str, object] = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).fit(y)
    """Captured both optimiser-stop and convergence diagnostics."""

    # Nought is a tolerance firing, one is running out of iterations, and
    # anything else is the optimiser reporting an error.
    assert record["stop_code"] in {0, 1}
    assert isinstance(record["stop_message"], str)
    assert record["stop_message"]
    # A well-conditioned fit meets the gradient test first time, so the polish
    # has nothing to do and says so.
    assert record["converged"]
    assert record["polished"] is False


def test_fit_uses_explicit_scale_dependent_coefficient_names() -> None:
    """Label raw coefficient proportions without calling them variance shares."""
    pairs: int = 20
    """Selected enough paired genotypes for the naming contract test."""

    matrix: npt.NDArray[np.float64] = _paired_gene_matrix(pairs)
    """Built the structured component from paired genotype roots."""

    rng: np.random.Generator = np.random.default_rng(4103)
    """Created a deterministic response generator."""

    y: npt.NDArray[np.float64] = np.repeat(
        rng.standard_normal(pairs), 2
    ) + rng.standard_normal(2 * pairs)
    """Combined pair-shared and independent response variation."""

    record: dict[str, object] = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).fit(y)
    """Produced the public component record whose field names are asserted."""

    assert len(record["raw_coefficient_proportions"]) == 2
    assert np.isclose(sum(record["raw_coefficient_proportions"]), 1.0)
    assert "shares" not in record
    assert "total_variance" not in record


def test_component_interval_names_its_scale_dependent_quantity() -> None:
    """Identify an explicitly requested raw-coefficient profile as diagnostic."""
    pairs: int = 20
    """Selected enough paired genotypes for a short profile fit."""

    matrix: npt.NDArray[np.float64] = _paired_gene_matrix(pairs)
    """Built the structured component from paired genotype roots."""

    rng: np.random.Generator = np.random.default_rng(4104)
    """Created a deterministic response generator."""

    y: npt.NDArray[np.float64] = np.repeat(
        rng.standard_normal(pairs), 2
    ) + rng.standard_normal(2 * pairs)
    """Combined pair-shared and independent response variation."""

    interval: dict[str, object] = asterism.ComponentModel(
        [matrix], np.ones((2 * pairs, 1))
    ).interval(y, component=0, quantity="raw_coefficient_proportion")
    """Profiled the scale-dependent coefficient proportion explicitly."""

    assert interval["quantity"] == "raw_coefficient_proportion"


def test_mean_diagonal_contributions_are_invariant_to_matrix_scaling() -> None:
    """Keep marginal contributions fixed under reciprocal coefficient scaling."""
    pairs: int = 120
    """Selected enough pairs for stable rescaling comparisons."""

    matrix: npt.NDArray[np.float64] = _paired_gene_matrix(pairs)
    """Built the component on its original numerical scale."""

    ten_times_matrix: npt.NDArray[np.float64] = _paired_gene_matrix(
        pairs, multiplier=np.sqrt(10.0)
    )
    """Built the same covariance root so its matrix is ten times larger."""

    rng: np.random.Generator = np.random.default_rng(4105)
    """Created a deterministic response generator."""

    y: npt.NDArray[np.float64] = np.sqrt(0.7) * np.repeat(
        rng.standard_normal(pairs), 2
    ) + np.sqrt(0.3) * rng.standard_normal(2 * pairs)
    """Drew a response with known shared-pair and residual contributions."""

    design: npt.NDArray[np.float64] = np.ones((2 * pairs, 1))
    """Built the intercept-only fixed-effect design."""

    original: dict[str, object] = asterism.ComponentModel([matrix], design).fit(y)
    """Fitted the model on the original component scale."""

    rescaled: dict[str, object] = asterism.ComponentModel(
        [ten_times_matrix], design
    ).fit(y)
    """Fitted the equivalent model on the tenfold component scale."""

    expected: list[float] = [
        original["variances"][0] * np.mean(np.diag(matrix)),
        original["variances"][1],
    ]
    """Computed the original fit's marginal component contributions directly."""
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


def test_mean_diagonal_interval_matches_the_reported_scale_invariant_quantity() -> None:
    """The component proportion and its interval use one scale."""
    pairs: int = 80
    """Sibling pairs in the interval rescaling fixture."""
    matrix: np.ndarray = _paired_gene_matrix(pairs)
    """Relationship component on its submitted scale."""
    ten_times_matrix: np.ndarray = _paired_gene_matrix(pairs, multiplier=np.sqrt(10.0))
    """The same component with every covariance coefficient ten times larger."""
    rng: np.random.Generator = np.random.default_rng(4108)
    """Deterministic stream for the public interval comparison."""
    y: np.ndarray = np.sqrt(0.7) * np.repeat(rng.standard_normal(pairs), 2) + np.sqrt(
        0.3
    ) * rng.standard_normal(2 * pairs)
    """Trait generated with a family component and independent residual."""
    design: np.ndarray = np.ones((2 * pairs, 1))
    """Intercept-only fixed-effect design."""

    original_model: asterism.ComponentModel = asterism.ComponentModel([matrix], design)
    """Component model using the original matrix scale."""
    rescaled_model: asterism.ComponentModel = asterism.ComponentModel(
        [ten_times_matrix], design
    )
    """Component model using the rescaled but equivalent covariance."""
    original_fit: dict[str, object] = original_model.fit(y)
    """Reported fit on the original matrix scale."""
    rescaled_fit: dict[str, object] = rescaled_model.fit(y)
    """Reported fit after matrix rescaling."""

    original: dict[str, object] = original_model.interval(
        y, component=0, quantity="mean_diagonal_proportion"
    )
    """Scale-invariant interval from the original matrix."""
    rescaled: dict[str, object] = rescaled_model.interval(
        y, component=0, quantity="mean_diagonal_proportion"
    )
    """Scale-invariant interval after rescaling the matrix."""

    assert original["quantity"] == "mean_diagonal_proportion"
    assert {
        "estimate",
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "level",
        "profile_failures",
        "contains_lower_bound",
        "contains_upper_bound",
    } <= set(original)
    assert original["profile_failures"] == 0
    assert original["estimate"] == pytest.approx(
        original_fit["mean_diagonal_proportions"][0]
    )
    assert rescaled["estimate"] == pytest.approx(
        rescaled_fit["mean_diagonal_proportions"][0]
    )
    np.testing.assert_allclose(
        [original["lower"], original["upper"]],
        [rescaled["lower"], rescaled["upper"]],
        rtol=2e-5,
        atol=2e-7,
    )


def test_component_matrices_are_copied_before_caller_mutation() -> None:
    """Prevent later caller mutation from changing a constructed model."""
    pairs: int = 40
    """Selected enough pairs to fit two distinguishable structured components."""

    subjects: int = 2 * pairs
    """Counted the subjects contributed by the disjoint pairs."""

    gene: npt.NDArray[np.float64] = _paired_gene_matrix(pairs)
    """Built the pair-specific genetic component."""

    group: npt.NDArray[np.float64] = _four_person_group_matrix(subjects)
    """Built the four-person shared-group component exposed to mutation."""

    original_group: npt.NDArray[np.float64] = group.copy()
    """Preserved the group matrix for an independent control model."""

    design: npt.NDArray[np.float64] = np.ones((subjects, 1))
    """Built the intercept-only fixed-effect design."""

    model: asterism.ComponentModel = asterism.ComponentModel([gene, group], design)
    """Constructed the model before the caller-owned group matrix changed."""

    control: asterism.ComponentModel = asterism.ComponentModel(
        [gene, original_group], design
    )
    """Constructed an equivalent control from the preserved group matrix."""

    rng: np.random.Generator = np.random.default_rng(4106)
    """Created a deterministic response generator."""

    y: npt.NDArray[np.float64] = (
        np.repeat(rng.standard_normal(pairs), 2)
        + np.repeat(rng.standard_normal(subjects // 4), 4)
        + rng.standard_normal(subjects)
    )
    """Combined pair, four-person group, and residual response variation."""

    group[:] = np.nan
    """Destroyed the caller-owned group matrix after both models were built."""

    copied: dict[str, object] = model.fit(y)
    """Fitted the model that had to rely on its construction-time copy."""

    expected: dict[str, object] = control.fit(y)
    """Fitted the control model built from the preserved matrix."""

    np.testing.assert_allclose(copied["variances"], expected["variances"])
    np.testing.assert_allclose(
        copied["mean_diagonal_component_contributions"],
        expected["mean_diagonal_component_contributions"],
    )


def test_component_design_is_copied_before_caller_mutation() -> None:
    """Prevent later caller mutation from changing the fixed-effect design."""
    pairs: int = 40
    """Selected enough pairs to estimate the covariate and shared variance."""

    subjects: int = 2 * pairs
    """Counted the subjects contributed by the disjoint pairs."""

    matrix: npt.NDArray[np.float64] = _paired_gene_matrix(pairs)
    """Built the pair-specific structured component."""

    covariate: npt.NDArray[np.float64] = np.linspace(-1.0, 1.0, subjects)
    """Constructed a nonconstant fixed-effect covariate."""

    design: npt.NDArray[np.float64] = np.column_stack([np.ones(subjects), covariate])
    """Combined the intercept and covariate into the caller-owned design."""

    original_design: npt.NDArray[np.float64] = design.copy()
    """Preserved the design for an independent control model."""

    model: asterism.ComponentModel = asterism.ComponentModel([matrix], design)
    """Constructed the model before the caller-owned design changed."""

    control: asterism.ComponentModel = asterism.ComponentModel(
        [matrix], original_design
    )
    """Constructed an equivalent control from the preserved design."""

    rng: np.random.Generator = np.random.default_rng(4107)
    """Created a deterministic response generator."""

    y: npt.NDArray[np.float64] = (
        0.6 * covariate
        + np.repeat(rng.standard_normal(pairs), 2)
        + rng.standard_normal(subjects)
    )
    """Combined a fixed covariate effect with pair and residual variation."""

    design[:] = np.nan
    """Destroyed the caller-owned design after both models were built."""

    copied: dict[str, object] = model.fit(y)
    """Fitted the model that had to rely on its construction-time design copy."""

    expected: dict[str, object] = control.fit(y)
    """Fitted the control model built from the preserved design."""

    np.testing.assert_allclose(copied["variances"], expected["variances"])
    assert copied["fixed_effects"] == expected["fixed_effects"]
    assert copied["loglik"] == expected["loglik"]


def test_component_nonfinite_inputs_are_refused_with_stable_codes() -> None:
    """NaN must be an invalid input, never a constant trait or failed search."""
    rows: int = 10
    """Rows in each small validation fixture."""
    matrix: np.ndarray = _paired_gene_matrix(rows // 2)
    """Finite component matrix used as the control input."""
    design: np.ndarray = np.ones((rows, 1))
    """Finite intercept-only control design."""
    response: np.ndarray = np.linspace(-1.0, 1.0, rows)
    """Finite nonconstant control response."""

    bad_matrix: np.ndarray = matrix.copy()
    """Component matrix into which one nonfinite value is injected."""
    bad_matrix[0, 0] = np.nan
    """Inject a nonfinite diagonal without changing its shape."""
    with pytest.raises(ValueError, match="COMPONENTS_MATRIX_NOT_FINITE"):
        asterism.ComponentModel([bad_matrix], design).fit(response)

    bad_design: np.ndarray = design.copy()
    """Fixed-effect design into which one nonfinite value is injected."""
    bad_design[0, 0] = np.nan
    """Inject a nonfinite fixed-effect value without changing its shape."""
    with pytest.raises(ValueError, match="COMPONENTS_DESIGN_NOT_FINITE"):
        asterism.ComponentModel([matrix], bad_design).fit(response)

    bad_response: np.ndarray = response.copy()
    """Response into which one nonfinite value is injected."""
    bad_response[0] = np.nan
    """Inject a nonfinite observation while retaining nonconstant finite peers."""
    with pytest.raises(ValueError, match="COMPONENTS_RESPONSE_NOT_FINITE"):
        asterism.ComponentModel([matrix], design).fit(bad_response)


def test_component_covariance_bases_must_be_identified() -> None:
    """Refuse coefficients that cannot be separated by their covariance bases."""
    rows: int = 6
    """Selected three pairs for the small construction-time validation."""

    design: np.ndarray = np.ones((rows, 1))
    """Built a full-rank intercept-only mean design."""

    relationship: np.ndarray = _paired_gene_matrix(rows // 2)
    """Built one structured basis distinct from the residual identity."""

    response: np.ndarray = np.linspace(-1.0, 1.0, rows)
    """Built a finite nonconstant response that would otherwise be fit."""

    with pytest.raises(
        ValueError,
        match=r"^COMPONENTS_COVARIANCE_BASES_RANK_DEFICIENT$",
    ):
        asterism.ComponentModel([relationship, relationship.copy()], design).fit(
            response
        )
    """Required duplicate structured matrices to refuse before fitting."""

    with pytest.raises(
        ValueError,
        match=r"^COMPONENTS_COVARIANCE_BASES_RANK_DEFICIENT$",
    ):
        asterism.ComponentModel([np.eye(rows)], design).fit(response)
    """Required a submitted identity confounded with the residual to refuse."""


def test_nonconverged_component_fit_cannot_feed_inference_or_prediction() -> None:
    """Keep a diagnostic fit available without turning it into a result."""
    matrix: np.ndarray = np.array(
        [
            [0.11326380334369826, -0.3007815031970412, -0.028517504206445846, -1.0],
            [
                -0.3007815031970412,
                0.32574213926578177,
                0.017376713372254108,
                0.3280389362405606,
            ],
            [
                -0.028517504206445846,
                0.017376713372254108,
                -0.5614754793244326,
                -0.5425739083617555,
            ],
            [-1.0, 0.3280389362405606, -0.5425739083617555, -0.6596614953578577],
        ]
    )
    """Symmetric stress matrix whose diagnostic fit has a large gradient."""
    response: np.ndarray = np.array(
        [
            -1.0217323059579484,
            -0.12056376369171515,
            -1.4247685045640195,
            -0.3973503794981748,
        ]
    )
    """Deterministic response that leaves the stress fit nonconverged."""
    design: np.ndarray = np.ones((4, 1))
    """Intercept-only design for the four-row stress model."""
    model: asterism.ComponentModel = asterism.ComponentModel([matrix], design)
    """Public component model used at every guarded seam."""
    fit: dict[str, object] = model.fit(response)
    """Diagnostic record which callers may still inspect after failure."""

    assert fit["converged"] is False
    assert fit["scaled_gradient"] > 1e-6
    with pytest.raises(ValueError, match=r"^COMPONENTS_FIT_NOT_CONVERGED$"):
        model.interval(response, component=0, quantity="raw_coefficient_proportion")
    with pytest.raises(ValueError, match=r"^COMPONENTS_FIT_NOT_CONVERGED$"):
        model.test(response, component=0)
    with pytest.raises(ValueError, match=r"^COMPONENTS_FIT_NOT_CONVERGED$"):
        model.predict(response, component=0)
