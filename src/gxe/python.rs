//! The Python interface to [`crate::gxe`].
//!
//! Kept beside the model rather than inside it, so the model file is the
//! mathematics and this file is the translation to and from Python.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{GxeModel, Reported, Shape, Surface};
use nalgebra::{DMatrix, DVector};

fn surface_named(name: &str, shape: f64) -> PyResult<Surface> {
    match name {
        "exponential" => Ok(Surface::Exponential),
        "random_regression" => Ok(Surface::RandomRegression),
        "powered_exponential" => Shape::from_value(shape)
            .map(Surface::PoweredExponential)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "GXE_SHAPE_NOT_ON_THE_GRID: {shape}, wanted one of 0.5, 1.0, \
                     1.5, 2.0. The shape is chosen and not fitted, because a \
                     shape and a decay rate trade off against each other and a \
                     search over both wanders."
                ))
            }),
        other => Err(PyValueError::new_err(format!(
            "GXE_UNKNOWN_SURFACE: {other}, wanted exponential, random_regression \
             or powered_exponential"
        ))),
    }
}

fn build(
    relationship: &PyReadonlyArray2<'_, f64>,
    environment: &[f64],
    design: &PyReadonlyArray2<'_, f64>,
    surface: &str,
    shape: f64,
) -> PyResult<GxeModel> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    GxeModel::build(surface_named(surface, shape)?, &a, environment, &x)
        .map_err(PyValueError::new_err)
}

fn response(y: &PyReadonlyArray1<'_, f64>) -> DVector<f64> {
    let y = y.as_array();
    DVector::from_iterator(y.len(), y.iter().copied())
}

/// Fit one trait with a genotype-by-environment surface.
///
/// **The reported quantities come back evaluated on `grid`**, rather than as
/// surface coefficients for the caller to expand. The two surfaces mean the
/// same things by heritability and by genetic correlation but say them in
/// quite different coordinates, and a caller that rebuilt either from the
/// parameters would have to know which surface it was holding. The genetic
/// correlations come back as a flattened square, in the grid's own order.
#[pyfunction]
#[pyo3(signature = (relationship, environment, design, y, surface, grid, shape=1.0, reml=true))]
#[allow(clippy::type_complexity)]
pub fn gxe_fit(
    relationship: PyReadonlyArray2<'_, f64>,
    environment: Vec<f64>,
    design: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    surface: &str,
    grid: Vec<f64>,
    shape: f64,
    reml: bool,
) -> PyResult<(
    Vec<f64>,
    f64,
    bool,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    let model = build(&relationship, &environment, &design, surface, shape)?;
    let fit = model
        .fit(&response(&y), reml)
        .map_err(PyValueError::new_err)?;
    let genetic: Vec<f64> = grid.iter().map(|&z| fit.genetic_variance_at(z)).collect();
    let residual: Vec<f64> = grid.iter().map(|&z| fit.residual_variance_at(z)).collect();
    let heritability: Vec<f64> = grid.iter().map(|&z| fit.heritability_at(z)).collect();
    let mut correlations = Vec::with_capacity(grid.len() * grid.len());
    for &first in &grid {
        for &second in &grid {
            correlations.push(fit.genetic_correlation(first, second));
        }
    }
    Ok((
        fit.parameters,
        fit.loglik,
        fit.converged,
        fit.scaled_gradient,
        fit.fixed_effects,
        fit.fixed_effect_errors,
        genetic,
        residual,
        heritability,
        correlations,
    ))
}

/// Test a genotype-by-environment null.
///
/// `null` is `interaction` for a genetic covariance that does not involve
/// the environment at all, or `correlation` for one where the genetic
/// variance may change but the genetic effects at any two environments are
/// the same effects.
#[pyfunction]
#[pyo3(signature = (relationship, environment, design, y, surface, null, shape=1.0, reml=true))]
pub fn gxe_test(
    relationship: PyReadonlyArray2<'_, f64>,
    environment: Vec<f64>,
    design: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    surface: &str,
    null: &str,
    shape: f64,
    reml: bool,
) -> PyResult<(f64, f64, String, f64, f64)> {
    let model = build(&relationship, &environment, &design, surface, shape)?;
    let y = response(&y);
    let test = match null {
        "interaction" => model.interaction_test(&y, reml),
        "correlation" => model.correlation_test(&y, reml),
        "variance" => model.variance_test(&y, reml),
        other => {
            return Err(PyValueError::new_err(format!(
                "GXE_UNKNOWN_NULL: {other}, wanted interaction, correlation or variance"
            )));
        }
    }
    .map_err(PyValueError::new_err)?;
    Ok((
        test.statistic,
        test.p_value,
        test.rule.to_owned(),
        test.null_loglik,
        test.alternative_loglik,
    ))
}

/// A 95 per cent profile-likelihood interval for one reported quantity.
///
/// `quantity` is `heritability`, which uses `first` as the environment, or
/// `correlation`, which uses both.
#[pyfunction]
#[pyo3(signature = (relationship, environment, design, y, surface, quantity, first, second=0.0, shape=1.0, reml=true))]
#[allow(clippy::too_many_arguments)]
pub fn gxe_interval(
    relationship: PyReadonlyArray2<'_, f64>,
    environment: Vec<f64>,
    design: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    surface: &str,
    quantity: &str,
    first: f64,
    second: f64,
    shape: f64,
    reml: bool,
) -> PyResult<(f64, f64, f64, bool, bool)> {
    let model = build(&relationship, &environment, &design, surface, shape)?;
    let wanted = match quantity {
        "heritability" => Reported::Heritability { at: first },
        "correlation" => Reported::GeneticCorrelation { first, second },
        other => {
            return Err(PyValueError::new_err(format!(
                "GXE_UNKNOWN_QUANTITY: {other}, wanted heritability or correlation"
            )));
        }
    };
    let got = model
        .profile_interval(&response(&y), reml, wanted)
        .map_err(PyValueError::new_err)?;
    Ok((
        got.estimate,
        got.lower,
        got.upper,
        got.lower_at_bound,
        got.upper_at_bound,
    ))
}
