//! The Python interface to [`crate::latent_mediation`].
//!
//! Kept beside the model rather than inside it, so the model file is the
//! mathematics and this file is the translation to and from Python.

// PyO3 extracts each argument from a Python object, so a `#[pyfunction]` takes
// them by value whether or not the body consumes them. The lint cannot be
// satisfied here without breaking the macro.
#![allow(clippy::needless_pass_by_value)]
// A `#[pyfunction]`'s parameter list is the Python signature, so grouping
// arguments into a struct to shorten it would make the interface worse.
#![allow(clippy::too_many_arguments)]

use super::{
    FIT_GRADIENT_TOLERANCE, LatentMediationDesign, LatentMediationEvaluation,
    LatentMediationFamilyEvaluation, LatentMediationFamilyInput, LatentMediationFit,
    LatentMediationModel, LatentMediationParameters,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyList, PySequence, PyString};

fn code(value: &'static str) -> PyErr {
    PyValueError::new_err(value)
}

fn is_numpy_boolean(value: &Bound<'_, PyAny>) -> bool {
    let value_type = value.get_type();
    let Ok(name) = value_type.name() else {
        return false;
    };
    let Ok(module) = value_type.getattr("__module__") else {
        return false;
    };
    let Ok(module) = module.extract::<String>() else {
        return false;
    };
    module == "numpy" && (name == "bool" || name == "bool_")
}

fn reject_boolean_tree(value: &Bound<'_, PyAny>, error: &'static str) -> PyResult<()> {
    if value.is_instance_of::<PyBool>() || is_numpy_boolean(value) {
        return Err(code(error));
    }
    let value_type = value.get_type();
    let numpy_array_like = value_type
        .getattr("__module__")
        .and_then(|module| module.extract::<String>())
        .is_ok_and(|module| module == "numpy" || module.starts_with("numpy."))
        && value.hasattr("dtype")?
        && value.hasattr("tolist")?;
    if numpy_array_like {
        return reject_boolean_tree(&value.call_method0("tolist")?, error);
    }
    if value.is_instance_of::<PyString>() || value.is_instance_of::<PyBytes>() {
        return Ok(());
    }
    if let Ok(sequence) = value.cast::<PySequence>() {
        for index in 0..sequence.len()? {
            reject_boolean_tree(&sequence.get_item(index)?, error)?;
        }
    }
    Ok(())
}

/// Compiled storage and numerical evaluation for the Python wrapper.
#[pyclass(name = "LatentMediationCore")]
pub struct PyLatentMediationCore {
    model: LatentMediationModel,
}

#[pymethods]
impl PyLatentMediationCore {
    #[staticmethod]
    #[pyo3(signature = (
        relationships,
        latent_means,
        mediator_measurements,
        mediator_measurement_error_variances,
        mediator_proxy_statuses,
        outcome_statuses,
        mediator_thresholds,
        outcome_thresholds,
        mediator_proxy_sensitivities,
        mediator_proxy_specificities,
        ascertainments,
        proband_indices,
        qmc_points,
        mediator_designs,
        outcome_designs,
        outcome_prevalences
    ))]
    #[allow(clippy::too_many_arguments)]
    fn _build(
        relationships: &Bound<'_, PyAny>,
        latent_means: &Bound<'_, PyAny>,
        mediator_measurements: &Bound<'_, PyAny>,
        mediator_measurement_error_variances: &Bound<'_, PyAny>,
        mediator_proxy_statuses: &Bound<'_, PyAny>,
        outcome_statuses: &Bound<'_, PyAny>,
        mediator_thresholds: &Bound<'_, PyAny>,
        outcome_thresholds: &Bound<'_, PyAny>,
        mediator_proxy_sensitivities: &Bound<'_, PyAny>,
        mediator_proxy_specificities: &Bound<'_, PyAny>,
        ascertainments: Vec<String>,
        proband_indices: &Bound<'_, PyAny>,
        qmc_points: &Bound<'_, PyAny>,
        mediator_designs: &Bound<'_, PyAny>,
        outcome_designs: &Bound<'_, PyAny>,
        outcome_prevalences: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        for value in [
            relationships,
            latent_means,
            mediator_measurements,
            mediator_measurement_error_variances,
            mediator_thresholds,
            outcome_thresholds,
            mediator_proxy_sensitivities,
            mediator_proxy_specificities,
            qmc_points,
            mediator_designs,
            outcome_designs,
            outcome_prevalences,
        ] {
            reject_boolean_tree(value, "LATENT_MEDIATION_NUMERIC_BOOLEAN")?;
        }
        for value in [mediator_proxy_statuses, outcome_statuses] {
            reject_boolean_tree(value, "LATENT_MEDIATION_STATUS_BOOLEAN")?;
        }
        reject_boolean_tree(proband_indices, "LATENT_MEDIATION_PROBAND_BOOLEAN")?;

        let relationships: Vec<Vec<Vec<f64>>> = relationships.extract()?;
        let mediator_designs: Vec<Vec<Vec<f64>>> = mediator_designs.extract()?;
        let outcome_designs: Vec<Vec<Vec<f64>>> = outcome_designs.extract()?;
        let outcome_prevalences: Vec<Vec<Option<f64>>> = outcome_prevalences.extract()?;
        let latent_means: Vec<Vec<f64>> = latent_means.extract()?;
        let mediator_measurements: Vec<Vec<Option<f64>>> = mediator_measurements.extract()?;
        let mediator_measurement_error_variances: Vec<Vec<Option<f64>>> =
            mediator_measurement_error_variances.extract()?;
        let mediator_proxy_statuses: Vec<Vec<Option<i8>>> = mediator_proxy_statuses.extract()?;
        let outcome_statuses: Vec<Vec<Option<i8>>> = outcome_statuses.extract()?;
        let mediator_thresholds: Vec<Vec<f64>> = mediator_thresholds.extract()?;
        let outcome_thresholds: Vec<Vec<f64>> = outcome_thresholds.extract()?;
        let mediator_proxy_sensitivities: Vec<Vec<f64>> = mediator_proxy_sensitivities.extract()?;
        let mediator_proxy_specificities: Vec<Vec<f64>> = mediator_proxy_specificities.extract()?;
        let proband_indices: Vec<Option<usize>> = proband_indices.extract()?;
        let qmc_points: usize = qmc_points.extract()?;
        let family_count = relationships.len();
        let lengths = [
            latent_means.len(),
            mediator_measurements.len(),
            mediator_measurement_error_variances.len(),
            mediator_proxy_statuses.len(),
            outcome_statuses.len(),
            mediator_thresholds.len(),
            outcome_thresholds.len(),
            mediator_proxy_sensitivities.len(),
            mediator_proxy_specificities.len(),
            ascertainments.len(),
            proband_indices.len(),
        ];
        if lengths.iter().any(|length| *length != family_count) {
            return Err(code("LATENT_MEDIATION_PACKED_FAMILY_LENGTH_MISMATCH"));
        }
        let mut inputs = Vec::with_capacity(family_count);
        for index in 0..family_count {
            inputs.push(LatentMediationFamilyInput {
                relationship: relationships[index].clone(),
                mediator_design: mediator_designs[index].clone(),
                outcome_design: outcome_designs[index].clone(),
                outcome_prevalence: outcome_prevalences[index].clone(),
                latent_mean: latent_means[index].clone(),
                mediator_measurement: mediator_measurements[index].clone(),
                mediator_measurement_error_variance: mediator_measurement_error_variances[index]
                    .clone(),
                mediator_proxy_status: mediator_proxy_statuses[index].clone(),
                outcome_status: outcome_statuses[index].clone(),
                mediator_threshold: mediator_thresholds[index].clone(),
                outcome_threshold: outcome_thresholds[index].clone(),
                mediator_proxy_sensitivity: mediator_proxy_sensitivities[index].clone(),
                mediator_proxy_specificity: mediator_proxy_specificities[index].clone(),
                ascertainment: ascertainments[index].clone(),
                proband_index: proband_indices[index],
            });
        }
        Ok(Self {
            model: LatentMediationModel::build(inputs, qmc_points).map_err(code)?,
        })
    }

    #[pyo3(signature = (*, a, b, c_prime, d, sigma_m2))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c_prime: &Bound<'_, PyAny>,
        d: &Bound<'_, PyAny>,
        sigma_m2: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        for value in [a, b, c_prime, d, sigma_m2] {
            reject_boolean_tree(value, "LATENT_MEDIATION_PARAMETER_BOOLEAN")?;
        }
        let a: f64 = a.extract()?;
        let b: f64 = b.extract()?;
        let c_prime: f64 = c_prime.extract()?;
        let d: f64 = d.extract()?;
        let sigma_m2: f64 = sigma_m2.extract()?;
        let record = self
            .model
            .evaluate(LatentMediationParameters {
                a,
                b,
                c_prime,
                d,
                sigma_m2,
            })
            .map_err(code)?;
        let output = PyDict::new(py);
        output.set_item("log_likelihood", record.log_likelihood)?;
        output.set_item(
            "ordinary_scale_representable",
            record.ordinary_scale_representable,
        )?;
        output.set_item("integration_methods", record.integration_methods.clone())?;
        output.set_item(
            "maximum_qmc_log_batch_range",
            record.maximum_qmc_log_batch_range,
        )?;
        let uses_qmc = record
            .integration_methods
            .iter()
            .any(|method| method == "deterministic_genz_halton");
        let uses_bivariate = record
            .integration_methods
            .iter()
            .any(|method| method.starts_with("bivariate"));
        output.set_item("approximate", uses_qmc || uses_bivariate)?;
        let fixed = PyDict::new(py);
        fixed.set_item("sigma_y2", 1.0)?;
        fixed.set_item("tau", 0.0)?;
        output.set_item("fixed_parameters", fixed)?;

        let family_records = PyList::empty(py);
        for family in record.families {
            let item = PyDict::new(py);
            item.set_item("log_likelihood", family.log_likelihood)?;
            item.set_item("log_continuous_density", family.log_continuous_density)?;
            item.set_item("log_discrete_probability", family.log_discrete_probability)?;
            item.set_item(
                "log_ascertainment_denominator",
                family.log_ascertainment_denominator,
            )?;
            item.set_item(
                "ordinary_scale_representable",
                family.ordinary_scale_representable,
            )?;
            item.set_item("integration_methods", family.integration_methods)?;
            item.set_item(
                "maximum_qmc_log_batch_range",
                family.maximum_qmc_log_batch_range,
            )?;
            item.set_item("family_size", family.family_size)?;
            item.set_item(
                "mediator_proxy_truth_configurations",
                family.mediator_proxy_truth_configurations,
            )?;
            item.set_item("ascertainment", family.ascertainment)?;
            family_records.append(item)?;
        }
        output.set_item("family_diagnostics", family_records)?;
        Ok(output)
    }

    /// Test the vertical estimand `a b` against nought.
    #[pyo3(signature = ())]
    fn test_vertical<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let test = self.model.test_vertical().map_err(code)?;
        let out = PyDict::new(py);
        out.set_item("p_value", test.p_value)?;
        out.set_item("rule", test.rule)?;
        out.set_item("loading_statistic", test.loading_statistic)?;
        out.set_item("loading_p_value", test.loading_p_value)?;
        out.set_item("path_statistic", test.path_statistic)?;
        out.set_item("path_p_value", test.path_p_value)?;
        Ok(out)
    }

    /// Fit the five structural parameters with the fixed ML recipe.
    #[pyo3(signature = ())]
    fn fit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fit = self.model.fit().map_err(code)?;
        fit_dict(
            py,
            &fit,
            self.model.qmc_points,
            self.model.subject_count(),
            self.model.family_count(),
        )
    }
}

fn parameter_dict(
    py: Python<'_>,
    values: LatentMediationParameters,
) -> PyResult<Bound<'_, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("a", values.a)?;
    output.set_item("b", values.b)?;
    output.set_item("c_prime", values.c_prime)?;
    output.set_item("d", values.d)?;
    output.set_item("sigma_m2", values.sigma_m2)?;
    Ok(output)
}

fn family_diagnostics<'py>(
    py: Python<'py>,
    families: &[LatentMediationFamilyEvaluation],
) -> PyResult<Bound<'py, PyList>> {
    let output = PyList::empty(py);
    for (index, family) in families.iter().enumerate() {
        let item = PyDict::new(py);
        item.set_item("family_index", index)?;
        item.set_item("integration_methods", family.integration_methods.clone())?;
        item.set_item(
            "maximum_qmc_log_batch_range",
            family.maximum_qmc_log_batch_range,
        )?;
        item.set_item("ascertainment", family.ascertainment)?;
        item.set_item("family_size", family.family_size)?;
        item.set_item(
            "mediator_proxy_truth_configurations",
            family.mediator_proxy_truth_configurations,
        )?;
        output.append(item)?;
    }
    Ok(output)
}

fn integration_diagnostics<'py>(
    py: Python<'py>,
    evaluation: &LatentMediationEvaluation,
    qmc_points: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    let uses_qmc = evaluation
        .integration_methods
        .iter()
        .any(|method| method == "deterministic_genz_halton");
    let uses_bivariate = evaluation
        .integration_methods
        .iter()
        .any(|method| method.starts_with("bivariate"));
    output.set_item("qmc_points", qmc_points)?;
    output.set_item(
        "integration_methods",
        evaluation.integration_methods.clone(),
    )?;
    output.set_item(
        "maximum_qmc_log_batch_range",
        evaluation.maximum_qmc_log_batch_range,
    )?;
    output.set_item(
        "family_diagnostics",
        family_diagnostics(py, &evaluation.families)?,
    )?;
    output.set_item("uses_qmc", uses_qmc)?;
    output.set_item("uses_bivariate_quadrature", uses_bivariate)?;
    output.set_item("approximate", uses_qmc || uses_bivariate)?;
    Ok(output)
}

fn fit_dict<'py>(
    py: Python<'py>,
    fit: &LatentMediationFit,
    qmc_points: usize,
    subject_count: usize,
    family_count: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("parameters", parameter_dict(py, fit.parameters)?)?;
    output.set_item("loglik", fit.log_likelihood)?;
    output.set_item("estimator", "ml")?;
    output.set_item("route", "latent_mediation")?;
    output.set_item("n", subject_count)?;
    output.set_item("family_count", family_count)?;
    output.set_item("optimiser", "rcompat-lbfgsb")?;
    output.set_item("convergence_rule", "scaled_projected_gradient")?;
    output.set_item("deterministic_starts", 5)?;
    output.set_item("converged", true)?;
    output.set_item("scaled_gradient", fit.scaled_gradient)?;
    output.set_item("gradient_tolerance", FIT_GRADIENT_TOLERANCE)?;
    output.set_item("boundary_parameters", fit.boundary_parameters.clone())?;
    output.set_item("coefficients", fit.coefficients.clone())?;
    output.set_item("horizontal_identified", fit.horizontal_identified)?;

    let fixed = PyDict::new(py);
    fixed.set_item("sigma_y2", 1.0)?;
    fixed.set_item("tau", 0.0)?;
    output.set_item("fixed_parameters", fixed)?;

    let estimands = PyDict::new(py);
    let vertical = fit.parameters.a * fit.parameters.b;
    estimands.set_item("theta_vertical", vertical)?;
    estimands.set_item("theta_horizontal", fit.parameters.c_prime)?;
    estimands.set_item(
        "total_mediator_related_outcome_loading",
        vertical + fit.parameters.c_prime,
    )?;
    output.set_item("estimands", estimands)?;

    output.set_item(
        "integration_diagnostics",
        integration_diagnostics(py, &fit.integration, qmc_points)?,
    )?;
    let approximate = fit
        .integration
        .integration_methods
        .iter()
        .any(|method| method.starts_with("bivariate") || method == "deterministic_genz_halton");
    let mut warnings = Vec::new();
    if !fit.horizontal_identified {
        warnings.push("horizontal_decomposition_unidentified_at_a_zero");
    }
    if approximate {
        warnings.push("integration_is_approximate");
    }
    output.set_item("warnings", warnings)?;
    Ok(output)
}

/// Draw families from the model, for calibration and power work.
///
/// The design is one family's shape, drawn from as many times as asked. It is
/// returned as the same dictionaries the model takes, so a campaign is
/// `simulate` then `LatentMediationModel` with nothing in between to get wrong.
#[pyfunction]
#[pyo3(signature = (
    relationship,
    mediator_design,
    mediator_coefficients,
    outcome_design,
    outcome_coefficients,
    outcome_prevalence,
    mediator_threshold,
    outcome_threshold,
    mediator_measurement_error_variance,
    observe_mediator_proxy,
    mediator_proxy_sensitivity,
    mediator_proxy_specificity,
    observe_outcome,
    a,
    b,
    c_prime,
    d,
    sigma_m2,
    families,
    seed,
    ascertainment="population_unconditioned",
    proband_index=None,
))]
pub fn latent_mediation_simulate<'py>(
    py: Python<'py>,
    relationship: Vec<Vec<f64>>,
    mediator_design: Vec<Vec<f64>>,
    mediator_coefficients: Vec<f64>,
    outcome_design: Vec<Vec<f64>>,
    outcome_coefficients: Vec<f64>,
    outcome_prevalence: Vec<Option<f64>>,
    mediator_threshold: Vec<f64>,
    outcome_threshold: Vec<f64>,
    mediator_measurement_error_variance: Vec<Option<f64>>,
    observe_mediator_proxy: Vec<bool>,
    mediator_proxy_sensitivity: Vec<f64>,
    mediator_proxy_specificity: Vec<f64>,
    observe_outcome: Vec<bool>,
    a: f64,
    b: f64,
    c_prime: f64,
    d: f64,
    sigma_m2: f64,
    families: usize,
    seed: u64,
    ascertainment: &str,
    proband_index: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let design = LatentMediationDesign {
        relationship,
        mediator_design,
        mediator_coefficients,
        outcome_design,
        outcome_coefficients,
        outcome_prevalence,
        mediator_threshold,
        outcome_threshold,
        mediator_measurement_error_variance,
        observe_mediator_proxy,
        mediator_proxy_sensitivity,
        mediator_proxy_specificity,
        observe_outcome,
        ascertainment: ascertainment.to_owned(),
        proband_index,
    };
    let parameters = LatentMediationParameters {
        a,
        b,
        c_prime,
        d,
        sigma_m2,
    };
    let drawn =
        super::simulate(&design, parameters, families, seed).map_err(PyValueError::new_err)?;
    let output = PyList::empty(py);
    for family in drawn {
        let item = PyDict::new(py);
        item.set_item("relationship", family.relationship)?;
        item.set_item("latent_mean", family.latent_mean)?;
        item.set_item("mediator_design", family.mediator_design)?;
        item.set_item("outcome_design", family.outcome_design)?;
        item.set_item("mediator_measurement", family.mediator_measurement)?;
        item.set_item(
            "mediator_measurement_error_variance",
            family.mediator_measurement_error_variance,
        )?;
        item.set_item("mediator_proxy_status", family.mediator_proxy_status)?;
        item.set_item("outcome_status", family.outcome_status)?;
        item.set_item("mediator_threshold", family.mediator_threshold)?;
        item.set_item("outcome_threshold", family.outcome_threshold)?;
        item.set_item("outcome_prevalence", family.outcome_prevalence)?;
        item.set_item(
            "mediator_proxy_sensitivity",
            family.mediator_proxy_sensitivity,
        )?;
        item.set_item(
            "mediator_proxy_specificity",
            family.mediator_proxy_specificity,
        )?;
        item.set_item("ascertainment", family.ascertainment)?;
        item.set_item("proband_index", family.proband_index)?;
        output.append(item)?;
    }
    Ok(output)
}
