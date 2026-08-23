"""Public-behaviour tests for the latent mediation model."""

from __future__ import annotations

import inspect
import math

import asterism
import numpy as np
import numpy.typing as npt
import pytest
from asterism import _core


def test_public_model_name_is_trait_neutral() -> None:
    """The package exposes the structural model, not one grant application."""
    assert asterism.LatentMediationModel.__name__ == "LatentMediationModel"
    assert asterism.__all__.count("LatentMediationModel") == 1


def test_compiled_model_name_is_trait_neutral() -> None:
    """The Python adapter does not translate around a trait-specific core."""
    assert hasattr(_core, "LatentMediationCore")
    obsolete_name: str = "".join(("Latent", "Primary", "Core"))
    """Reconstructed the retired trait-specific name without leaving it literal."""

    assert not hasattr(_core, obsolete_name)


def test_family_and_parameter_vocabulary_is_trait_neutral() -> None:
    """A caller describes measurements of a mediator and outcome."""
    family: dict[str, object] = {
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
        "ascertainment": "population_unconditioned",
        "proband_index": None,
    }
    """Described one exact family using mediator-and-outcome vocabulary."""

    record: dict[str, object] = asterism.LatentMediationModel([family]).evaluate(
        a=0.5,
        b=0.3,
        c_prime=0.2,
        d=0.6,
        sigma_m2=0.7,
    )
    """Evaluated the family's likelihood at one fixed structural parameter set."""

    assert math.isclose(record["log_likelihood"], -2.3527743921924436, abs_tol=1e-9)
    assert record["family_diagnostics"][0]["mediator_proxy_truth_configurations"] == 2
    assert record["fixed_parameters"] == {"sigma_y2": 1.0, "tau": 0.0}


# asterism-style: allow private-helper -- shared one-person family fixture
def _singleton(**updates: object) -> dict[str, object]:
    """Return one exact family for public-interface checks."""
    family: dict[str, object] = {
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
        "ascertainment": "population_unconditioned",
        "proband_index": None,
    }
    """Constructed the canonical one-person latent-mediation family."""

    family.update(updates)
    """Applied the test-specific family fields supplied by the caller."""
    return family


# asterism-style: allow private-helper -- shared two-person family fixture
def _dyad(**updates: object) -> dict[str, object]:
    """Return one exact two-person family."""
    family: dict[str, object] = _singleton(
        relationship=[[1.0, 0.5], [0.5, 1.0]],
        latent_mean=[0.0, 0.0, 0.0, 0.0],
        mediator_measurement=[0.18, 0.11],
        mediator_measurement_error_variance=[0.22, 0.22],
        mediator_proxy_status=[1, None],
        outcome_status=[1, 0],
        mediator_threshold=[-0.1, -0.1],
        outcome_threshold=[0.35, 0.35],
        mediator_proxy_sensitivity=[0.8, 0.8],
        mediator_proxy_specificity=[0.85, 0.85],
    )
    """Expanded the canonical fixture to a related two-person family."""

    family.update(updates)
    """Applied the test-specific dyad fields supplied by the caller."""
    return family


# asterism-style: allow private-helper -- shared optimiser-behaviour fixture
def _fit_families() -> list[dict[str, object]]:
    """Return deterministic related families for optimiser behaviour checks."""
    rng: np.random.Generator = np.random.default_rng(20260815)
    """Created the deterministic generator for every optimiser fixture family."""

    relationship: npt.NDArray[np.float64] = np.array([[1.0, 0.5], [0.5, 1.0]])
    """Defined the shared first-degree relationship for every dyad."""

    cholesky: npt.NDArray[np.float64] = np.linalg.cholesky(relationship)
    """Factorised the relationship matrix for related latent effects."""

    families: list[dict[str, object]] = []
    """Initialised the deterministic optimiser-fitting family collection."""

    for family_index in range(10):
        inherited_mediator: npt.NDArray[np.float64] = cholesky @ rng.normal(size=2)
        """Drew related genetic effects on the mediator for the current dyad."""

        inherited_outcome: npt.NDArray[np.float64] = cholesky @ rng.normal(size=2)
        """Drew related genetic effects on the outcome for the current dyad."""

        mediator: npt.NDArray[np.float64] = 0.55 * inherited_mediator + rng.normal(
            scale=math.sqrt(0.65), size=2
        )
        """Combined inherited and residual variation in the latent mediator."""

        outcome: npt.NDArray[np.float64] = (
            0.25 * mediator
            + 0.15 * inherited_mediator
            + 0.65 * inherited_outcome
            + rng.normal(size=2)
        )
        """Combined mediated, direct, inherited, and residual outcome variation."""

        measurement: npt.NDArray[np.float64] = mediator + rng.normal(
            scale=math.sqrt(0.18), size=2
        )
        """Added known continuous measurement error to the mediator."""

        families.append(
            {
                "relationship": relationship.tolist(),
                "latent_mean": [0.0, 0.0, 0.0, 0.0],
                "mediator_measurement": measurement.tolist(),
                "mediator_measurement_error_variance": [0.18, 0.18],
                "mediator_proxy_status": [None, None],
                "outcome_status": [int(outcome[family_index % 2] > 0.25), None],
                "mediator_threshold": [0.0, 0.0],
                "outcome_threshold": [0.25, 0.25],
                "mediator_proxy_sensitivity": [0.8, 0.8],
                "mediator_proxy_specificity": [0.85, 0.85],
                "ascertainment": "population_unconditioned",
                "proband_index": None,
            }
        )
    """Constructed ten related families with mixed mediator and outcome observations."""
    return families


def test_constructor_exposes_families_and_qmc_points() -> None:
    """The ordinary constructor owns the family and integration configuration."""
    assert tuple(inspect.signature(asterism.LatentMediationModel).parameters) == (
        "families",
        "qmc_points",
    )
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [_singleton()], qmc_points=256
    )
    """Constructed the public model with explicit deterministic integration work."""

    record: dict[str, object] = model.evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    """Evaluated the configured family at fixed structural parameters."""

    assert math.isfinite(record["log_likelihood"])
    with pytest.raises(ValueError, match="LATENT_MEDIATION_QMC_POINTS_INVALID"):
        asterism.LatentMediationModel([_singleton()], qmc_points=255)


def test_fixed_point_likelihood_stays_in_rust_and_on_the_log_scale() -> None:
    """Evaluate one exact family without importing a Python reference core."""
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [
            {
                **_singleton(),
                "mediator_measurement": [100.0],
                "mediator_measurement_error_variance": [0.15],
                "mediator_proxy_status": [None],
                "outcome_status": [None],
                "mediator_threshold": [0.0],
                "outcome_threshold": [0.4],
            }
        ]
    )
    """Constructed an extreme continuous-mediator family without discrete outcomes."""

    record: dict[str, object] = model.evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    """Evaluated the extreme family while retaining its log probability."""

    assert math.isclose(record["log_likelihood"], -4546.421139077653, abs_tol=1e-9)
    assert record["ordinary_scale_representable"] is False
    assert record["integration_methods"] == ["no_discrete_observation"]
    assert record["fixed_parameters"] == {"sigma_y2": 1.0, "tau": 0.0}
    assert record["approximate"] is False


def test_mixed_observations_match_reference_value() -> None:
    """Pin continuous, fallible proxy, and binary outcome observations."""
    record: dict[str, object] = asterism.LatentMediationModel([_singleton()]).evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    """Evaluated the canonical mixed-observation family at its reference point."""
    assert math.isclose(record["log_likelihood"], -2.3527743921924436, abs_tol=1e-9)
    assert record["integration_methods"] == ["bivariate_quadrature"]
    assert record["approximate"] is True
    assert record["family_diagnostics"][0]["mediator_proxy_truth_configurations"] == 2


def test_a_near_singular_bivariate_tail_is_computed_on_the_log_scale() -> None:
    """A probability far below the smallest double still has a logarithm.

    Splitting a near-perfectly correlated pair across half a standard deviation
    has a log probability near minus sixty million. This once had to fail
    closed, because the rectangle was exponentiated back to an ordinary number
    before it was used, and everything below `ln(5e-324)` became nought. The
    log scale is now carried through, so the value is returned and only the
    ordinary-scale flag says it cannot be written as a plain number.
    """
    latent_sd: float = math.sqrt(1.0e12 + 1.0)
    """Computed the outcome scale induced by the extreme direct loading."""

    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [
            _dyad(
                mediator_measurement=[None, None],
                mediator_measurement_error_variance=[None, None],
                mediator_proxy_status=[None, None],
                outcome_status=[0, 1],
                relationship=[[1.0, 0.999999999], [0.999999999, 1.0]],
                outcome_threshold=[-5.5 * latent_sd, -5.0 * latent_sd],
            )
        ]
    )
    """Constructed a near-perfectly correlated dyad split across a far tail."""

    record: dict[str, object] = model.evaluate(
        a=0.0, b=0.0, c_prime=1.0e6, d=0.0, sigma_m2=0.7
    )
    """Evaluated the tiny joint probability without returning to ordinary scale."""

    assert record["log_likelihood"] < -1e7
    assert math.isfinite(record["log_likelihood"])
    assert record["ordinary_scale_representable"] is False


def test_feasible_near_singular_bivariate_is_approximate() -> None:
    """Feasible near-singular mass is accepted but remains approximate."""
    family: dict[str, object] = _dyad(
        mediator_measurement=[None, None],
        mediator_measurement_error_variance=[None, None],
        mediator_proxy_status=[None, None],
        outcome_status=[0, 0],
        relationship=[[1.0, 0.999999999], [0.999999999, 1.0]],
        outcome_threshold=[0.0, 0.0],
    )
    """Constructed a feasible near-singular dyad with matching outcome statuses."""

    record: dict[str, object] = asterism.LatentMediationModel([family]).evaluate(
        a=0.0, b=0.0, c_prime=1.0e6, d=0.0, sigma_m2=0.7
    )
    """Evaluated the accepted near-singular probability approximation."""

    assert record["integration_methods"] == ["bivariate_quadrature"]
    assert record["approximate"] is True


def test_named_case_conditioning_matches_reference_value() -> None:
    """Check the proband denominator on a related outcome-only dyad."""
    family: dict[str, object] = _dyad(
        mediator_measurement=[None, None],
        mediator_measurement_error_variance=[None, None],
        mediator_proxy_status=[None, None],
        outcome_status=[1, 0],
        outcome_threshold=[0.35, 0.5],
        ascertainment="condition_on_named_proband_case",
        proband_index=0,
    )
    """Constructed a dyad conditioned on the named proband being a case."""

    record: dict[str, object] = asterism.LatentMediationModel([family]).evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    """Evaluated the ascertained family likelihood at its reference point."""

    assert math.isclose(record["log_likelihood"], -0.5125795909122253, abs_tol=1e-9)
    diagnostic: dict[str, object] = record["family_diagnostics"][0]
    """Selected the family-level ascertainment diagnostic."""

    assert diagnostic["ascertainment"] == "condition_on_named_proband_case"
    assert diagnostic["log_ascertainment_denominator"] < 0.0


def test_higher_dimensional_rectangle_uses_requested_qmc_work() -> None:
    """The caller selects deterministic QMC work for larger rectangles."""
    family: dict[str, object] = _dyad(
        mediator_measurement=[None, None],
        mediator_measurement_error_variance=[None, None],
        mediator_proxy_status=[1, None],
        outcome_status=[1, 0],
        mediator_threshold=[0.0, 0.0],
        outcome_threshold=[0.4, 0.4],
    )
    """Constructed a dyad requiring integration over three discrete dimensions."""

    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [family], qmc_points=256
    )
    """Selected an explicit deterministic QMC work budget."""

    record: dict[str, object] = model.evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    """Evaluated the higher-dimensional rectangle with deterministic QMC."""

    assert record["integration_methods"] == ["deterministic_genz_halton"]
    assert record["maximum_qmc_log_batch_range"] > 0.0
    assert record["approximate"] is True


def test_relationship_symmetry_is_exact() -> None:
    """Even a one-ULP directional mismatch is not symmetric."""
    asymmetric: float = float(np.nextafter(0.5, 1.0))
    """Moved one relationship entry a single binary64 step above one half."""

    with pytest.raises(ValueError, match="LATENT_MEDIATION_RELATIONSHIP_NOT_SYMMETRIC"):
        asterism.LatentMediationModel(
            [_dyad(relationship=[[1.0, 0.5], [asymmetric, 1.0]])]
        )


def test_relationship_psd_floor_is_negative_one_e_minus_nine() -> None:
    """Rounding residue at the PSD floor passes; lower eigenvalues fail."""
    accepted: float = 1.0 + 5.0e-10
    """Selected a relationship whose smallest eigenvalue stays above the floor."""

    asterism.LatentMediationModel(
        [_dyad(relationship=[[1.0, accepted], [accepted, 1.0]])]
    )
    refused: float = 1.0 + 2.0e-9
    """Selected a relationship whose smallest eigenvalue crosses the floor."""

    with pytest.raises(
        ValueError,
        match="LATENT_MEDIATION_RELATIONSHIP_NOT_POSITIVE_SEMIDEFINITE",
    ):
        asterism.LatentMediationModel(
            [_dyad(relationship=[[1.0, refused], [refused, 1.0]])]
        )


def test_binary_statuses_and_continuous_parameters_refuse_booleans() -> None:
    """Python booleans cannot silently become statuses or continuous values."""
    with pytest.raises(ValueError, match="LATENT_MEDIATION_STATUS_BOOLEAN"):
        asterism.LatentMediationModel([_singleton(mediator_proxy_status=[True])])
    model: asterism.LatentMediationModel = asterism.LatentMediationModel([_singleton()])
    """Constructed a valid model before passing a Boolean continuous parameter."""

    with pytest.raises(ValueError, match="LATENT_MEDIATION_PARAMETER_BOOLEAN"):
        model.evaluate(a=True, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7)
    with pytest.raises(ValueError, match="LATENT_MEDIATION_PROBAND_BOOLEAN"):
        asterism.LatentMediationModel(
            [
                _singleton(
                    ascertainment="condition_on_named_proband_case",
                    proband_index=True,
                )
            ]
        )


def test_public_constructor_rejects_boolean_qmc_values() -> None:
    """A Boolean is not a valid integration-work setting."""
    with pytest.raises(ValueError, match="LATENT_MEDIATION_NUMERIC_BOOLEAN"):
        asterism.LatentMediationModel([_singleton()], qmc_points=True)


def test_wrapper_accepts_in_memory_numpy_arrays() -> None:
    """Use the same array boundary as the rest of Asterism."""
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [
            _singleton(
                relationship=np.eye(1),
                latent_mean=np.zeros(2),
                mediator_measurement=np.array([0.18]),
                mediator_measurement_error_variance=np.array([0.22]),
                mediator_proxy_status=np.array([1], dtype=np.int8),
                outcome_status=np.array([1], dtype=np.int8),
            )
        ]
    )
    """Constructed the canonical family entirely from in-memory NumPy arrays."""

    record: dict[str, object] = model.evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    """Evaluated the array-backed family through the public wrapper."""

    assert math.isfinite(record["log_likelihood"])


def test_fit_reports_optimiser_and_convergence_diagnostics() -> None:
    """The fit reports its numerical method and convergence diagnostics."""
    record: dict[str, object] = asterism.LatentMediationModel(_fit_families()).fit()
    """Fitted the deterministic related-family optimiser fixture."""

    assert set(record["parameters"]) == {"a", "b", "c_prime", "d", "sigma_m2"}
    assert record["fixed_parameters"] == {"sigma_y2": 1.0, "tau": 0.0}
    assert record["route"] == "latent_mediation"
    assert "total_mediator_related_outcome_loading" in record["estimands"]
    assert math.isfinite(record["loglik"])
    assert "log_likelihood" not in record
    assert record["optimiser"] == "rcompat-lbfgsb"
    assert record["convergence_rule"] == "scaled_projected_gradient"
    assert record["deterministic_starts"] == 5
    assert record["converged"] is True
    assert record["scaled_gradient"] < record["gradient_tolerance"]
    assert isinstance(record["integration_diagnostics"]["approximate"], bool)
    assert record["integration_diagnostics"]["qmc_points"] == 8192
    assert "rectangle_methods" not in record["integration_diagnostics"]
    for family in record["integration_diagnostics"]["family_diagnostics"]:
        assert "rectangle_methods" not in family
        assert "mediator_proxy_truth_configurations" in family


def test_fit_is_deterministic_and_has_no_caller_controls() -> None:
    """Repeated fits use the same fixed recipe and return the same record."""
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        _fit_families()
    )
    """Constructed one model whose fixed fitting recipe is repeated."""

    assert model.fit() == model.fit()
    with pytest.raises(TypeError):
        model.fit(bounds={"a": (0.0, 1.0)})


def test_fit_dominates_every_fixed_start() -> None:
    """The selected optimum cannot be worse than its deterministic starts."""
    families: list[dict[str, object]] = _fit_families()
    """Built the deterministic family collection used by the optimiser."""

    model: asterism.LatentMediationModel = asterism.LatentMediationModel(families)
    """Constructed the public model whose deterministic starts are checked."""

    observed: list[tuple[float, float]] = [
        (float(value), float(error))
        for family in families
        for value, error in zip(
            family["mediator_measurement"],
            family["mediator_measurement_error_variance"],
            strict=True,
        )
        if value is not None and error is not None
    ]
    """Collected continuous mediator values with their known error variances."""

    mediator_scale2: float = float(
        np.mean([value * value + error for value, error in observed])
    )
    """Estimated the deterministic marginal scale used to transform starts."""

    mediator_scale: float = math.sqrt(mediator_scale2)
    """Converted the marginal mediator variance to its standard-deviation scale."""

    transformed_starts: list[tuple[float, float, float, float, float]] = [
        (math.sqrt(0.5), 0.0, 0.0, 0.5, math.log(0.5)),
        (math.sqrt(0.2), 0.5, 0.5, 0.5, math.log(0.8)),
        (math.sqrt(0.2), -0.5, -0.5, 0.5, math.log(0.8)),
        (math.sqrt(0.8), 0.5, -0.5, 0.2, math.log(0.2)),
        (math.sqrt(0.8), -0.5, 0.5, 0.2, math.log(0.2)),
    ]
    """Listed the five deterministic starts in optimiser coordinates."""

    start_logliks: list[float] = [
        model.evaluate(
            a=mediator_scale * alpha,
            b=beta / mediator_scale,
            c_prime=c_prime,
            d=math.sqrt(q_d),
            sigma_m2=mediator_scale2 * math.exp(eta_m),
        )["log_likelihood"]
        for alpha, beta, c_prime, q_d, eta_m in transformed_starts
    ]
    """Evaluated every fixed start on the model's reported likelihood scale."""

    assert model.fit()["loglik"] >= max(start_logliks) - 1e-10


def test_fit_requires_a_continuous_mediator_scale_anchor() -> None:
    """Binary thresholds alone do not identify the latent-mediator scale."""
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [
            _singleton(
                mediator_measurement=[None],
                mediator_measurement_error_variance=[None],
            )
        ]
    )
    """Constructed a family with no continuous mediator-scale observation."""

    with pytest.raises(
        ValueError, match="LATENT_MEDIATION_MEDIATOR_SCALE_UNIDENTIFIED"
    ):
        model.fit()


# asterism-style: allow private-helper -- shared extreme-tail dyad fixture
def _all_case_pair(outcome_threshold: float) -> dict[str, object]:
    """Two related people, both outcome cases, nothing else observed."""
    return {
        "relationship": [[1.0, 0.5], [0.5, 1.0]],
        "latent_mean": [0.0, 0.0, 0.0, 0.0],
        "mediator_measurement": [None, None],
        "mediator_measurement_error_variance": [None, None],
        "mediator_proxy_status": [None, None],
        "outcome_status": [1, 1],
        "mediator_threshold": [0.0, 0.0],
        "outcome_threshold": [outcome_threshold, outcome_threshold],
        "mediator_proxy_sensitivity": [0.8, 0.8],
        "mediator_proxy_specificity": [0.85, 0.85],
        "ascertainment": "population_unconditioned",
        "proband_index": None,
    }


def test_a_far_tail_two_person_probability_is_computed_not_refused() -> None:
    """A pair of extreme outcome cases has a tiny but well-defined likelihood.

    The corner-difference recipe cannot resolve it; the log-scale conditional
    quadrature must. The reference value was computed independently with
    high-order Gauss-Legendre quadrature on the conditional tail integral.
    """
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [_all_case_pair(10.0)]
    )
    """Constructed an all-case dyad with an extreme outcome threshold."""

    record: dict[str, object] = model.evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    """Evaluated the dyad through the dedicated bivariate tail quadrature."""

    assert record["log_likelihood"] == pytest.approx(-61.71004139194921, abs=1e-6)
    assert "bivariate_tail_quadrature" in record["integration_methods"]


def test_qmc_matches_an_independent_trivariate_value() -> None:
    """The shifted-Halton estimate lands near the independently derived truth.

    The constant is the trivariate rectangle mixture computed outside Asterism
    by conditional quadrature; the tolerance is set from the measured accuracy
    of the estimator at 8,192 points, with an order of magnitude of headroom.
    """
    model: asterism.LatentMediationModel = asterism.LatentMediationModel(
        [
            {
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
        ],
        qmc_points=8192,
    )
    """Constructed the trivariate rectangle fixture at the reference QMC budget."""

    record: dict[str, object] = model.evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    """Evaluated the deterministic shifted-Halton approximation."""

    assert record["log_likelihood"] == pytest.approx(-2.145943720364346, abs=5e-4)
