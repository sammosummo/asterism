//! The Python interface to [`crate::discrete_gxe`].
//!
//! Kept beside the model rather than inside it, so the model file is the
//! mathematics and this file is the translation to and from Python.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{DiscreteGxeModel, DiscreteGxeTest};
use nalgebra::{DMatrix, DVector};

fn build(
    relationship: &PyReadonlyArray2<'_, f64>,
    environment: &PyReadonlyArray1<'_, f64>,
    design: &PyReadonlyArray2<'_, f64>,
) -> PyResult<DiscreteGxeModel> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let environment: Vec<f64> = environment.as_array().iter().copied().collect();
    DiscreteGxeModel::build(&a, &environment, &x).map_err(PyValueError::new_err)
}

fn response(y: &PyReadonlyArray1<'_, f64>) -> DVector<f64> {
    DVector::from_iterator(y.as_array().len(), y.as_array().iter().copied())
}

fn unpack(test: &DiscreteGxeTest) -> (f64, f64, String, f64, f64) {
    (
        test.statistic,
        test.p_value,
        test.rule.to_string(),
        test.null_loglik,
        test.alternative_loglik,
    )
}

/// Fit one trait with the genes allowed to act differently in the two
/// environments.
///
/// `environment` is one label per person and must take exactly two
/// distinct finite values; the smaller label names the first group. Sex
/// coded 1 and 2 is the canonical use.
///
/// Returns the two genetic variances, the two residual variances, the two
/// heritabilities, the genetic correlation across the environments, the
/// fixed effects and their standard errors, the log likelihood, whether
/// the search converged, its scaled gradient, how many people fell in each
/// group, and the two environment labels.
#[pyfunction]
#[pyo3(signature = (relationship, environment, design, response, reml=true))]
#[allow(clippy::type_complexity)]
pub fn discrete_gxe_fit(
    relationship: PyReadonlyArray2<'_, f64>,
    environment: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    response: PyReadonlyArray1<'_, f64>,
    reml: bool,
) -> PyResult<(
    [f64; 2],
    [f64; 2],
    [f64; 2],
    f64,
    Vec<f64>,
    Vec<f64>,
    f64,
    bool,
    f64,
    [usize; 2],
    [f64; 2],
)> {
    let model = build(&relationship, &environment, &design)?;
    let fit = model
        .fit(&super::python::response(&response), reml)
        .map_err(PyValueError::new_err)?;
    Ok((
        fit.genetic_variance,
        fit.residual_variance,
        [fit.heritability(0), fit.heritability(1)],
        fit.correlation,
        fit.fixed_effects,
        fit.fixed_effect_errors,
        fit.loglik,
        fit.converged,
        fit.scaled_gradient,
        fit.counts,
        fit.levels,
    ))
}

/// One discrete gene-by-environment test.
///
/// `which` selects it:
///
/// - `"gene_by_environment"` -- a genetic difference of any kind, with the
///   two residual variances left free. **Read this one first**; it is what
///   stops the others being read as separate findings.
/// - `"any_difference"` -- anything at all differing between the
///   environments, including the residual. Not a genetic test: a trait
///   measured more noisily in one environment rejects it with nothing
///   genetic happening.
/// - `"correlation"` -- are the genes the same in both environments? This
///   is the gene-by-environment question proper.
/// - `"genetic"` -- is the genetic variance the same? A difference of scale,
///   which a difference of measurement units can produce.
/// - `"residual"` -- is the residual variance the same? Report it beside the
///   others; it is a measurement fact, not a genetic finding.
///
/// Returns the statistic, the p-value, the reference distribution it is a
/// tail of, and the two log likelihoods.
#[pyfunction]
#[pyo3(signature = (relationship, environment, design, response, which, reml=true))]
pub fn discrete_gxe_test(
    relationship: PyReadonlyArray2<'_, f64>,
    environment: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    response: PyReadonlyArray1<'_, f64>,
    which: &str,
    reml: bool,
) -> PyResult<(f64, f64, String, f64, f64)> {
    let model = build(&relationship, &environment, &design)?;
    let y = super::python::response(&response);
    let test = match which {
        "gene_by_environment" => model.gene_by_environment_test(&y, reml),
        "any_difference" => model.any_difference_test(&y, reml),
        "correlation" => model.correlation_test(&y, reml),
        "genetic" => model.genetic_equality_test(&y, reml),
        "residual" => model.residual_equality_test(&y, reml),
        _ => {
            return Err(PyValueError::new_err(
                "which must be one of gene_by_environment, any_difference, \
                 correlation, genetic, residual",
            ));
        }
    }
    .map_err(PyValueError::new_err)?;
    Ok(unpack(&test))
}

/// A 95 per cent profile interval for the genetic correlation across the two
/// environments.
///
/// Returns the estimate, the two endpoints, and whether each endpoint sat at
/// the edge of what a correlation may be rather than where the likelihood fell
/// away.
#[pyfunction]
#[pyo3(signature = (relationship, environment, design, response, reml=true))]
pub fn discrete_gxe_correlation_interval(
    relationship: PyReadonlyArray2<'_, f64>,
    environment: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    response: PyReadonlyArray1<'_, f64>,
    reml: bool,
) -> PyResult<(f64, f64, f64, bool, bool)> {
    let model = build(&relationship, &environment, &design)?;
    let y = super::python::response(&response);
    let interval = model
        .correlation_interval(&y, reml)
        .map_err(PyValueError::new_err)?;
    Ok((
        interval.estimate,
        interval.lower,
        interval.upper,
        interval.lower_limited,
        interval.upper_limited,
    ))
}
