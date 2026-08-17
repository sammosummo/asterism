"""Public-behaviour tests for the latent mediation model."""

from __future__ import annotations

import inspect
import math

import asterism
import numpy as np
import pytest
from asterism import _core


def test_public_model_name_is_trait_neutral() -> None:
    """The package exposes the structural model, not one grant application."""
    assert asterism.LatentMediationModel.__name__ == "LatentMediationModel"
    assert asterism.__all__.count("LatentMediationModel") == 1


def test_compiled_model_name_is_trait_neutral() -> None:
    """The Python adapter does not translate around a trait-specific core."""
    assert hasattr(_core, "LatentMediationCore")
    obsolete_name = "".join(("Latent", "Primary", "Core"))
    assert not hasattr(_core, obsolete_name)


def test_family_and_parameter_vocabulary_is_trait_neutral() -> None:
    """A caller describes measurements of a mediator and outcome."""
    family = {
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
    record = asterism.LatentMediationModel([family]).evaluate(
        a=0.5,
        b=0.3,
        c_prime=0.2,
        d=0.6,
        sigma_m2=0.7,
    )
    assert math.isclose(record["log_likelihood"], -2.3527743921924436, abs_tol=1e-9)
    assert record["family_diagnostics"][0]["mediator_proxy_truth_configurations"] == 2
    assert record["fixed_parameters"] == {"sigma_y2": 1.0, "tau": 0.0}


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
    family.update(updates)
    return family


def _dyad(**updates: object) -> dict[str, object]:
    """Return one exact two-person family."""
    family = _singleton(
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
    family.update(updates)
    return family


def _fit_families() -> list[dict[str, object]]:
    """Return deterministic related families for optimiser behaviour checks."""
    rng = np.random.default_rng(20260815)
    relationship = np.array([[1.0, 0.5], [0.5, 1.0]])
    cholesky = np.linalg.cholesky(relationship)
    families: list[dict[str, object]] = []
    for family_index in range(10):
        inherited_mediator = cholesky @ rng.normal(size=2)
        inherited_outcome = cholesky @ rng.normal(size=2)
        mediator = 0.55 * inherited_mediator + rng.normal(
            scale=math.sqrt(0.65), size=2
        )
        outcome = (
            0.25 * mediator
            + 0.15 * inherited_mediator
            + 0.65 * inherited_outcome
            + rng.normal(size=2)
        )
        measurement = mediator + rng.normal(scale=math.sqrt(0.18), size=2)
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
    return families


def test_constructor_exposes_families_and_qmc_points() -> None:
    """The ordinary constructor owns the family and integration configuration."""
    assert tuple(inspect.signature(asterism.LatentMediationModel).parameters) == (
        "families",
        "qmc_points",
    )
    model = asterism.LatentMediationModel([_singleton()], qmc_points=256)
    record = model.evaluate(a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7)
    assert math.isfinite(record["log_likelihood"])
    with pytest.raises(ValueError, match="LATENT_MEDIATION_QMC_POINTS_INVALID"):
        asterism.LatentMediationModel([_singleton()], qmc_points=255)


def test_fixed_point_likelihood_stays_in_rust_and_on_the_log_scale() -> None:
    """Evaluate one exact family without importing a Python reference core."""
    model = asterism.LatentMediationModel(
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
    record = model.evaluate(a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7)
    assert math.isclose(record["log_likelihood"], -4546.421139077653, abs_tol=1e-9)
    assert record["ordinary_scale_representable"] is False
    assert record["integration_methods"] == ["no_discrete_observation"]
    assert record["fixed_parameters"] == {"sigma_y2": 1.0, "tau": 0.0}
    assert record["approximate"] is False


def test_mixed_observations_match_reference_value() -> None:
    """Pin continuous, fallible proxy, and binary outcome observations."""
    record = asterism.LatentMediationModel([_singleton()]).evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    assert math.isclose(record["log_likelihood"], -2.3527743921924436, abs_tol=1e-9)
    assert record["integration_methods"] == ["bivariate_quadrature"]
    assert record["approximate"] is True
    assert (
        record["family_diagnostics"][0][
            "mediator_proxy_truth_configurations"
        ]
        == 2
    )


def test_a_near_singular_bivariate_tail_is_computed_on_the_log_scale() -> None:
    """A probability far below the smallest double still has a logarithm.

    Splitting a near-perfectly correlated pair across half a standard deviation
    has a log probability near minus sixty million. This once had to fail
    closed, because the rectangle was exponentiated back to an ordinary number
    before it was used, and everything below `ln(5e-324)` became nought. The
    log scale is now carried through, so the value is returned and only the
    ordinary-scale flag says it cannot be written as a plain number.
    """
    latent_sd = math.sqrt(1.0e12 + 1.0)
    model = asterism.LatentMediationModel(
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
    record = model.evaluate(a=0.0, b=0.0, c_prime=1.0e6, d=0.0, sigma_m2=0.7)
    assert record["log_likelihood"] < -1e7
    assert math.isfinite(record["log_likelihood"])
    assert record["ordinary_scale_representable"] is False


def test_feasible_near_singular_bivariate_is_approximate() -> None:
    """Feasible near-singular mass is accepted but remains approximate."""
    family = _dyad(
        mediator_measurement=[None, None],
        mediator_measurement_error_variance=[None, None],
        mediator_proxy_status=[None, None],
        outcome_status=[0, 0],
        relationship=[[1.0, 0.999999999], [0.999999999, 1.0]],
        outcome_threshold=[0.0, 0.0],
    )
    record = asterism.LatentMediationModel([family]).evaluate(
        a=0.0, b=0.0, c_prime=1.0e6, d=0.0, sigma_m2=0.7
    )
    assert record["integration_methods"] == ["bivariate_quadrature"]
    assert record["approximate"] is True


def test_named_case_conditioning_matches_reference_value() -> None:
    """Check the proband denominator on a related outcome-only dyad."""
    family = _dyad(
        mediator_measurement=[None, None],
        mediator_measurement_error_variance=[None, None],
        mediator_proxy_status=[None, None],
        outcome_status=[1, 0],
        outcome_threshold=[0.35, 0.5],
        ascertainment="condition_on_named_proband_case",
        proband_index=0,
    )
    record = asterism.LatentMediationModel([family]).evaluate(
        a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7
    )
    assert math.isclose(record["log_likelihood"], -0.5125795909122253, abs_tol=1e-9)
    diagnostic = record["family_diagnostics"][0]
    assert diagnostic["ascertainment"] == "condition_on_named_proband_case"
    assert diagnostic["log_ascertainment_denominator"] < 0.0


def test_higher_dimensional_rectangle_uses_requested_qmc_work() -> None:
    """The caller selects deterministic QMC work for larger rectangles."""
    family = _dyad(
        mediator_measurement=[None, None],
        mediator_measurement_error_variance=[None, None],
        mediator_proxy_status=[1, None],
        outcome_status=[1, 0],
        mediator_threshold=[0.0, 0.0],
        outcome_threshold=[0.4, 0.4],
    )
    model = asterism.LatentMediationModel([family], qmc_points=256)
    record = model.evaluate(a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7)
    assert record["integration_methods"] == ["deterministic_genz_halton"]
    assert record["maximum_qmc_log_batch_range"] > 0.0
    assert record["approximate"] is True


def test_relationship_symmetry_is_exact() -> None:
    """Even a one-ULP directional mismatch is not symmetric."""
    asymmetric = np.nextafter(0.5, 1.0)
    with pytest.raises(
        ValueError, match="LATENT_MEDIATION_RELATIONSHIP_NOT_SYMMETRIC"
    ):
        asterism.LatentMediationModel(
            [_dyad(relationship=[[1.0, 0.5], [asymmetric, 1.0]])]
        )


def test_relationship_psd_floor_is_negative_one_e_minus_nine() -> None:
    """Rounding residue at the PSD floor passes; lower eigenvalues fail."""
    accepted = 1.0 + 5.0e-10
    asterism.LatentMediationModel(
        [_dyad(relationship=[[1.0, accepted], [accepted, 1.0]])]
    )
    refused = 1.0 + 2.0e-9
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
        asterism.LatentMediationModel(
            [_singleton(mediator_proxy_status=[True])]
        )
    model = asterism.LatentMediationModel([_singleton()])
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
    model = asterism.LatentMediationModel(
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
    record = model.evaluate(a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7)
    assert math.isfinite(record["log_likelihood"])


def test_fit_reports_optimiser_and_convergence_diagnostics() -> None:
    """The fit reports its numerical method and convergence diagnostics."""
    record = asterism.LatentMediationModel(_fit_families()).fit()
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
    model = asterism.LatentMediationModel(_fit_families())
    assert model.fit() == model.fit()
    with pytest.raises(TypeError):
        model.fit(bounds={"a": (0.0, 1.0)})


def test_fit_dominates_every_fixed_start() -> None:
    """The selected optimum cannot be worse than its deterministic starts."""
    families = _fit_families()
    model = asterism.LatentMediationModel(families)
    observed = [
        (float(value), float(error))
        for family in families
        for value, error in zip(
            family["mediator_measurement"],
            family["mediator_measurement_error_variance"], strict=True,
        )
        if value is not None and error is not None
    ]
    mediator_scale2 = float(
        np.mean([value * value + error for value, error in observed])
    )
    mediator_scale = math.sqrt(mediator_scale2)
    transformed_starts = [
        (math.sqrt(0.5), 0.0, 0.0, 0.5, math.log(0.5)),
        (math.sqrt(0.2), 0.5, 0.5, 0.5, math.log(0.8)),
        (math.sqrt(0.2), -0.5, -0.5, 0.5, math.log(0.8)),
        (math.sqrt(0.8), 0.5, -0.5, 0.2, math.log(0.2)),
        (math.sqrt(0.8), -0.5, 0.5, 0.2, math.log(0.2)),
    ]
    start_logliks = [
        model.evaluate(
            a=mediator_scale * alpha,
            b=beta / mediator_scale,
            c_prime=c_prime,
            d=math.sqrt(q_d),
            sigma_m2=mediator_scale2 * math.exp(eta_m),
        )["log_likelihood"]
        for alpha, beta, c_prime, q_d, eta_m in transformed_starts
    ]
    assert model.fit()["loglik"] >= max(start_logliks) - 1e-10


def test_fit_requires_a_continuous_mediator_scale_anchor() -> None:
    """Binary thresholds alone do not identify the latent-mediator scale."""
    model = asterism.LatentMediationModel(
        [
            _singleton(
                mediator_measurement=[None],
                mediator_measurement_error_variance=[None],
            )
        ]
    )
    with pytest.raises(
        ValueError, match="LATENT_MEDIATION_MEDIATOR_SCALE_UNIDENTIFIED"
    ):
        model.fit()


def _all_case_pair(outcome_threshold: float) -> dict:
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
    model = asterism.LatentMediationModel([_all_case_pair(10.0)])
    record = model.evaluate(a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7)
    assert record["log_likelihood"] == pytest.approx(-61.71004139194921, abs=1e-6)
    assert "bivariate_tail_quadrature" in record["integration_methods"]


def test_qmc_matches_an_independent_trivariate_value() -> None:
    """The shifted-Halton estimate lands near the independently derived truth.

    The constant is the trivariate rectangle mixture computed outside Asterism
    by conditional quadrature; the tolerance is set from the measured accuracy
    of the estimator at 8,192 points, with an order of magnitude of headroom.
    """
    model = asterism.LatentMediationModel(
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
    record = model.evaluate(a=0.5, b=0.3, c_prime=0.2, d=0.6, sigma_m2=0.7)
    assert record["log_likelihood"] == pytest.approx(-2.145943720364346, abs=5e-4)
