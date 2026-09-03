"""Public contracts for censored several-component boundary inference."""

from __future__ import annotations

import asterism
import numpy as np
import numpy.typing as npt
import pytest


def problem(
    seed: int = 8_302,
) -> tuple[
    asterism.CensoredComponentModel,
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Build one deterministic, identified censored several-component problem."""
    people: int = 120
    """Used enough independent families for both null nuisance terms to fit."""

    additive: npt.NDArray[np.float64] = np.eye(people)
    """Started the additive component at unrelated identity."""

    for pair in range(people // 2):
        first: int = 2 * pair
        """Located the first sibling in this adjacent pair."""

        second: int = first + 1
        """Located the second sibling in this adjacent pair."""

        additive[first, second] = 0.5
        """Set the forward full-sibling relationship."""

        additive[second, first] = 0.5
        """Mirrored the full-sibling relationship."""

    household: npt.NDArray[np.float64] = np.zeros((people, people))
    """Started an independent shared-household component."""

    for start in range(0, people, 4):
        household[start : start + 4, start : start + 4] = 1.0
        """Put two sibling pairs in each four-person household."""

    covariance: npt.NDArray[np.float64] = (
        0.6 * additive + 0.2 * household + 0.2 * np.eye(people)
    )
    """Generated a clear additive signal with both nuisance shares interior."""

    generator: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic complete-response stream."""

    complete: npt.NDArray[np.float64] = np.linalg.cholesky(
        covariance
    ) @ generator.standard_normal(people)
    """Drew the latent complete response from the declared covariance."""

    ceiling: float = 0.7
    """Fixed the instrument limit independently of the realised response."""

    censoring: npt.NDArray[np.int64] = (complete >= ceiling).astype(np.int64)
    """Encoded rows at or above the fixed ceiling as right-censored."""

    value: npt.NDArray[np.float64] = np.where(censoring == 0, complete, np.nan)
    """Retained values only where the instrument measured them."""

    limit: npt.NDArray[np.float64] = np.full(people, ceiling)
    """Retained the instrument limit for every row."""

    design: npt.NDArray[np.float64] = np.ones((people, 1))
    """Declared an intercept-only mean model."""

    model: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
        [additive, household], design
    )
    """Prepared additive, household and residual covariance components."""
    return model, value, censoring, limit, additive, design


def high_censoring_problem() -> tuple[
    asterism.CensoredComponentModel,
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Build a deterministic bootstrap whose first inner draw is unfit."""
    people: int = 30
    """Used a compact set of sibling pairs whose first null draw is too censored."""

    additive: npt.NDArray[np.float64] = np.eye(people)
    """Started the additive relationship at unrelated identity."""

    for pair in range(people // 2):
        first: int = 2 * pair
        """Located the first member of this sibling pair."""

        second: int = first + 1
        """Located the second member of this sibling pair."""

        additive[first, second] = additive[second, first] = 0.5
        """Gave the pair their symmetric full-sibling relationship."""

    design: npt.NDArray[np.float64] = np.ones((people, 1))
    """Fitted one common intercept."""

    generator: np.random.Generator = np.random.default_rng(8_300)
    """Selected the retained deterministic observed-response stream."""

    complete: npt.NDArray[np.float64] = np.linalg.cholesky(
        0.6 * additive + 0.4 * np.eye(people)
    ) @ generator.standard_normal(people)
    """Drew the complete response before applying the instrument."""

    ceiling: float = -1.5
    """Fixed a deliberately severe right-censoring limit."""

    censoring: npt.NDArray[np.int64] = (complete >= ceiling).astype(np.int64)
    """Marked the observations beyond the instrument limit."""

    value: npt.NDArray[np.float64] = np.where(censoring == 0, complete, np.nan)
    """Retained numeric values only where measurement completed."""

    limit: npt.NDArray[np.float64] = np.full(people, ceiling)
    """Recorded the fixed instrument limit for every latent row."""

    direction: npt.NDArray[np.int64] = np.ones(people, dtype=np.int64)
    """Declared right censoring for every possible generated value."""

    model: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
        [additive], design
    )
    """Prepared the one-component public model used by the bootstrap."""

    return model, value, censoring, limit, direction


def test_several_component_test_reports_an_asymptotic_reference() -> None:
    """Label the several-component analytic reference as asymptotic."""
    model, value, censoring, limit, _, _ = problem()
    """Built the identified several-component public problem."""

    result: dict[str, object] = model.test(value, censoring, limit, component=0)
    """Tested an interior several-component problem through the public method."""

    raw_statistic: float = 2.0 * (
        float(result["alternative_loglik"]) - float(result["null_loglik"])
    )
    """Reconstructed the likelihood-ratio statistic from its two fitted models."""

    expected_statistic: float = 0.0 if raw_statistic < 1e-6 else raw_statistic
    """Applied the public settlement tolerance independently of the test output."""

    assert result["rule"] == "asymptotic_mixture_50_50"
    assert result["component"] == 0
    assert result["nuisance_at_bound"] is False
    assert float(result["statistic"]) == pytest.approx(expected_statistic)

    with pytest.raises(ValueError, match="TOBIT_NO_SUCH_COMPONENT"):
        model.test(value, censoring, limit, component=2)
    """Kept a malformed component coordinate more specific than the route refusal."""


def test_several_component_test_reorders_a_nonfirst_component() -> None:
    """Attach the analytic result to the component the caller requested."""
    model, value, censoring, limit, _, _ = problem(seed=8_300)
    """Built a fixture whose first component stays interior while testing the second."""

    result: dict[str, object] = model.test(
        value,
        censoring,
        limit,
        component=1,
    )
    """Tested the second supplied covariance component rather than the first."""

    raw_statistic: float = 2.0 * (
        float(result["alternative_loglik"]) - float(result["null_loglik"])
    )
    """Reconstructed the requested component's likelihood-ratio statistic."""

    expected_statistic: float = 0.0 if raw_statistic < 1e-6 else raw_statistic
    """Applied the public settlement tolerance independently."""

    assert result["component"] == 1
    assert result["rule"] == "asymptotic_mixture_50_50"
    assert result["nuisance_at_bound"] is False
    assert float(result["statistic"]) == pytest.approx(expected_statistic)


def test_several_component_test_refuses_a_nuisance_boundary() -> None:
    """Refuse the simple asymptotic reference at a multiple boundary."""
    model, value, censoring, limit, _, _ = problem(seed=8_302)
    """Selected the deterministic fixture with a nuisance share on its bound."""

    with pytest.raises(
        ValueError,
        match=r"^TOBIT_COMPONENT_TEST_NUISANCE_AT_BOUND$",
    ):
        model.test(value, censoring, limit, component=1)


def test_several_component_covariance_bases_must_be_identified() -> None:
    """Refuse public models whose component coefficients have no unique meaning."""
    _, value, censoring, limit, additive, design = problem()
    """Reused valid censored inputs while replacing only the covariance bases."""

    duplicate: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
        [additive, additive.copy()], design
    )
    """Supplied one structured basis twice under two component positions."""

    with pytest.raises(
        ValueError,
        match=r"^TOBIT_COVARIANCE_BASES_RANK_DEFICIENT$",
    ):
        duplicate.fit(value, censoring, limit)

    residual_twice: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
        [additive, np.eye(additive.shape[0])], design
    )
    """Supplied the residual identity again as an explicit component."""

    with pytest.raises(
        ValueError,
        match=r"^TOBIT_COVARIANCE_BASES_RANK_DEFICIENT$",
    ):
        residual_twice.bootstrap(
            value,
            censoring,
            limit,
            np.ones(censoring.shape, dtype=np.int64),
            component=0,
            replicates=1,
            seed=1,
        )


def test_one_component_keeps_the_released_analytic_reference() -> None:
    """Keep every overlapping one-component result bit-identical to Tobit."""
    _, value, censoring, limit, additive, design = problem()
    """Reused the public fixture while selecting its additive matrix."""

    one_component: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
        [additive], design
    )
    """Selected the same data and design with only its additive component."""

    component_fit: dict[str, object] = one_component.fit(value, censoring, limit)
    """Fitted through the general component-shaped public interface."""

    legacy_fit: dict[str, object] = asterism.tobit_fit(
        additive,
        value,
        censoring,
        limit,
        design,
    )
    """Fitted the identical arrays through the released one-component interface."""

    assert component_fit["coefficients"] == [legacy_fit["heritability"]]
    for field in (
        "total_variance",
        "fixed_effects",
        "loglik",
        "converged",
        "scaled_gradient",
        "censored_share",
        "largest_family",
        "estimator",
        "build",
        "subject_order_sha256",
    ):
        assert component_fit[field] == legacy_fit[field]

    component_interval: dict[str, object] = one_component.interval(
        value,
        censoring,
        limit,
        component=0,
    )
    """Profiled through the general one-matrix interface."""

    legacy_interval: dict[str, object] = asterism.tobit_interval(
        additive,
        value,
        censoring,
        limit,
        design,
    )
    """Profiled the same likelihood through the released function."""

    for field in (
        "estimate",
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "level",
        "contains_lower_bound",
        "contains_upper_bound",
        "profile_failures",
        "estimator",
    ):
        assert component_interval[field] == legacy_interval[field]

    component_test: dict[str, object] = one_component.test(
        value,
        censoring,
        limit,
        component=0,
    )
    """Ran the retained analytic test through the component-shaped interface."""

    legacy_test: dict[str, object] = asterism.tobit_test(
        additive,
        value,
        censoring,
        limit,
        design,
    )
    """Ran the released one-component analytic entry point on the same arrays."""

    for field in (
        "statistic",
        "p_value",
        "rule",
        "null_loglik",
        "alternative_loglik",
        "estimator",
    ):
        assert component_test[field] == legacy_test[field]
    assert component_test["rule"] == "mixture_50_50"
    assert component_test["component"] == 0


def test_bootstrap_refits_the_complete_null_reference_reproducibly() -> None:
    """Use every requested null draw and reproduce it from the public seed."""
    model, value, censoring, limit, _, _ = problem()
    """Built an identified several-component bootstrap problem."""

    direction: npt.NDArray[np.int64] = np.ones(censoring.shape, dtype=np.int64)
    """Declared right censoring for every row, including those measured here."""

    first: dict[str, object] = model.bootstrap(
        value,
        censoring,
        limit,
        direction,
        component=0,
        replicates=5,
        seed=4_117,
    )
    """Generated the first complete seeded null reference."""

    second: dict[str, object] = model.bootstrap(
        value,
        censoring,
        limit,
        direction,
        component=0,
        replicates=5,
        seed=4_117,
    )
    """Repeated the identical public request from the same seed."""

    assert first == second
    assert first["rule"] == "parametric_bootstrap_add_one"
    assert first["replicates"] == first["requested"] == 5
    assert first["seed"] == 4_117
    assert first["component"] == 0
    assert first["nuisance_at_bound"] is False
    assert float(first["null_loglik"]) <= float(first["alternative_loglik"])
    assert first["smallest_p_value"] == 1.0 / 6.0
    assert first["p_value"] == (int(first["exceedances"]) + 1) / 6.0
    assert float(first["monte_carlo_standard_error"]) >= 0.0


def test_bootstrap_failure_preserves_inner_coordinate_and_cause() -> None:
    """A failed draw remains unknown, but no longer anonymous."""
    model, value, censoring, limit, direction = high_censoring_problem()
    """Built the public case whose first bootstrap draw is too censored."""

    with pytest.raises(
        ValueError,
        match=(
            r"^TOBIT_BOOTSTRAP_REPLICATE_FAILED:replicate=0:"
            r"cause=TOBIT_TOO_FEW_MEASURED_VALUES$"
        ),
    ):
        model.bootstrap(
            value,
            censoring,
            limit,
            direction,
            component=0,
            replicates=8,
            seed=1,
        )

    with pytest.raises(
        ValueError,
        match=(
            r"^TOBIT_BOOTSTRAP_REPLICATE_FAILED:replicate=0:"
            r"cause=TOBIT_TOO_FEW_MEASURED_VALUES$"
        ),
    ):
        model.bootstrap_replay(
            value,
            censoring,
            limit,
            direction,
            component=0,
            seed=1,
            replicate=0,
        )


def test_one_bootstrap_coordinate_can_be_replayed_exactly() -> None:
    """Expose one deterministic inner draw without turning it into a p-value."""
    model, value, censoring, limit, _, _ = problem(seed=8_300)
    """Built the identified public problem used by replay and full bootstrap."""

    direction: npt.NDArray[np.int64] = np.ones(censoring.shape, dtype=np.int64)
    """Declared right censoring for every latent row."""

    replay: dict[str, object] = model.bootstrap_replay(
        value,
        censoring,
        limit,
        direction,
        component=1,
        seed=6_119,
        replicate=0,
    )
    """Replayed only the first deterministic coordinate."""

    full: dict[str, object] = model.bootstrap(
        value,
        censoring,
        limit,
        direction,
        component=1,
        replicates=1,
        seed=6_119,
    )
    """Ran the same one-coordinate request through the complete bootstrap."""

    assert replay["replicate"] == 0
    assert replay["seed"] == 6_119
    assert replay["component"] == 1
    assert replay["observed_statistic"] == full["statistic"]
    assert int(bool(replay["exceeded"])) == full["exceedances"]
    assert "p_value" not in replay
    assert replay == model.bootstrap_replay(
        value,
        censoring,
        limit,
        direction,
        component=1,
        seed=6_119,
        replicate=0,
    )


def test_bootstrap_reorders_and_tests_a_nonfirst_component() -> None:
    """Keep the requested component attached to its own observed statistic."""
    model, value, censoring, limit, _, _ = problem(seed=8_300)
    """Built a fixture whose second component remains an interior nuisance."""

    direction: npt.NDArray[np.int64] = np.ones(censoring.shape, dtype=np.int64)
    """Declared right censoring for every latent row."""

    result: dict[str, object] = model.bootstrap(
        value,
        censoring,
        limit,
        direction,
        component=1,
        replicates=1,
        seed=6_119,
    )
    """Bootstrapped the second rather than the first supplied component."""

    raw_statistic: float = 2.0 * (
        float(result["alternative_loglik"]) - float(result["null_loglik"])
    )
    """Reconstructed the unrounded observed likelihood-ratio statistic."""

    expected_statistic: float = 0.0 if raw_statistic < 1e-6 else raw_statistic
    """Applied the public operational settlement tolerance independently."""

    assert result["component"] == 1
    assert float(result["statistic"]) == pytest.approx(expected_statistic)


def test_bootstrap_refuses_an_observed_multiple_boundary_problem() -> None:
    """Do not apply an ordinary plug-in bootstrap at a nuisance boundary."""
    model, value, censoring, limit, _, _ = problem(seed=8_302)
    """Selected the deterministic fixture with a nuisance share on its bound."""

    direction: npt.NDArray[np.int64] = np.ones(censoring.shape, dtype=np.int64)
    """Declared the complete right-censoring mechanism."""

    with pytest.raises(ValueError, match="TOBIT_BOOTSTRAP_NUISANCE_AT_BOUND"):
        model.bootstrap(
            value,
            censoring,
            limit,
            direction,
            component=1,
            replicates=1,
            seed=6_119,
        )


def test_bootstrap_requires_the_complete_censoring_design() -> None:
    """Do not invent limits or censoring directions for currently measured rows."""
    model, value, censoring, limit, _, _ = problem()
    """Built the ordinary complete right-censoring fixture."""

    direction: npt.NDArray[np.int64] = np.ones(censoring.shape, dtype=np.int64)
    """Declared the complete right-censoring mechanism."""

    with pytest.raises(ValueError, match="TOBIT_BOOTSTRAP_NO_REPLICATES"):
        model.bootstrap(
            value,
            censoring,
            limit,
            direction,
            component=0,
            replicates=0,
            seed=1,
        )

    incomplete_limit: npt.NDArray[np.float64] = limit.copy()
    """Started from an otherwise valid complete instrument record."""

    incomplete_limit[censoring == 0] = np.nan
    """Reproduced a valid fit input that cannot generate bootstrap censoring."""

    with pytest.raises(ValueError, match="TOBIT_BOOTSTRAP_LIMIT_NOT_FINITE"):
        model.bootstrap(
            value,
            censoring,
            incomplete_limit,
            direction,
            component=0,
            replicates=1,
            seed=1,
        )

    incomplete_direction: npt.NDArray[np.int64] = direction.copy()
    """Started from the complete direction record."""

    incomplete_direction[0] = 0
    """Left one row without the direction its latent draw would be censored."""

    with pytest.raises(ValueError, match="TOBIT_BOOTSTRAP_DIRECTION_REQUIRED"):
        model.bootstrap(
            value,
            censoring,
            limit,
            incomplete_direction,
            component=0,
            replicates=1,
            seed=1,
        )

    conflicting_direction: npt.NDArray[np.int64] = direction.copy()
    """Started from the observed right-censoring direction."""

    conflicting_direction[np.flatnonzero(censoring == 1)[0]] = 2
    """Contradicted the observed right-censoring status on one row."""

    with pytest.raises(ValueError, match="TOBIT_BOOTSTRAP_DIRECTION_MISMATCH"):
        model.bootstrap(
            value,
            censoring,
            limit,
            conflicting_direction,
            component=0,
            replicates=1,
            seed=1,
        )


@pytest.mark.parametrize("direction_code", [1, 2])
def test_measured_value_may_equal_its_instrument_limit(
    direction_code: int,
) -> None:
    """Let explicit status distinguish a measured value at either limit."""
    model, value, censoring, limit, _, _ = problem()
    """Built the right-censored fixture before optionally mirroring it."""

    if direction_code == 2:
        value *= -1.0
        """Mirrored each measured value around nought."""

        limit *= -1.0
        """Mirrored every row's instrument limit around nought."""

        censoring[:] = np.where(censoring == 1, 2, 0).astype(np.int64)
        """Converted observed right-censoring statuses to left-censoring."""

    measured: int = int(np.flatnonzero(censoring == 0)[0])
    """Selected one explicitly measured row."""

    value[measured] = limit[measured]
    """Put its numeric value exactly on its instrument limit."""

    direction: npt.NDArray[np.int64] = np.full(
        censoring.shape,
        direction_code,
        dtype=np.int64,
    )
    """Declared one censoring direction for every latent row."""

    result: dict[str, object] = model.bootstrap(
        value,
        censoring,
        limit,
        direction,
        component=0,
        replicates=1,
        seed=92,
    )
    """Ran one null refit to prove equality is accepted."""

    assert result["replicates"] == 1


def test_bootstrap_accepts_mixed_directions_and_row_specific_limits() -> None:
    """Carry the declared instrument row by row rather than assuming one ceiling."""
    _, value, censoring, limit, additive, design = problem(seed=8_300)
    """Selected a deterministic response and its additive covariance."""

    model: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
        [additive], design
    )
    """Used one structured component so mixed-censoring input is isolated."""
    direction: npt.NDArray[np.int64] = np.where(
        np.arange(censoring.size) % 2 == 0,
        1,
        2,
    ).astype(np.int64)
    """Alternated right and left censoring across otherwise identical rows."""

    mixed_censoring: npt.NDArray[np.int64] = censoring.copy()
    """Copied the original censoring status before changing its directions."""
    mixed_censoring[mixed_censoring != 0] = direction[mixed_censoring != 0]
    """Made every observed censored status agree with its declared direction."""

    row_limit: npt.NDArray[np.float64] = limit.copy()
    """Started the row-specific instrument from the shared ceiling fixture."""

    measured: npt.NDArray[np.bool_] = mixed_censoring == 0
    """Located rows with observed numeric values."""

    right_measured: npt.NDArray[np.bool_] = measured & (direction == 1)
    """Located measured rows governed by right censoring."""

    left_measured: npt.NDArray[np.bool_] = measured & (direction == 2)
    """Located measured rows governed by left censoring."""

    row_limit[right_measured] = value[right_measured] + 0.5
    """Placed right limits above their measured values."""

    row_limit[left_measured] = value[left_measured] - 0.5
    """Placed left limits below their measured values."""

    row_limit[(mixed_censoring != 0) & (direction == 2)] *= -1.0
    """Gave each row a finite limit consistent with its explicit observed status."""

    result: dict[str, object] = model.bootstrap(
        value,
        mixed_censoring,
        row_limit,
        direction,
        component=0,
        replicates=1,
        seed=617,
    )
    """Ran the constrained-null reference through the mixed instrument."""

    assert result["replicates"] == result["requested"] == 1
