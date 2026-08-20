//! The Python interface to [`crate::liability`].
//!
//! Kept beside the model rather than inside it, so the model file is the
//! mathematics and this file is the translation to and from Python.

// PyO3 extracts each argument from a Python object, so a `#[pyfunction]` takes
// them by value whether or not the body consumes them. The lint cannot be
// satisfied here without breaking the macro.
#![allow(clippy::needless_pass_by_value)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::LiabilityModel;
use nalgebra::DMatrix;

fn build(
    relationship: &PyReadonlyArray2<'_, f64>,
    status: &PyReadonlyArray1<'_, f64>,
    design: &PyReadonlyArray2<'_, f64>,
) -> PyResult<LiabilityModel> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let status: Vec<f64> = status.as_array().iter().copied().collect();
    LiabilityModel::build(&a, &status, &x).map_err(PyValueError::new_err)
}

/// Fit one binary trait through a liability threshold.
///
/// Returns the liability heritability, the fixed effects, the log
/// likelihood, whether the search converged, its scaled gradient, the
/// prevalence and the largest family the region probability had to cover.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn liability_fit(
    relationship: PyReadonlyArray2<'_, f64>,
    status: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<(f64, Vec<f64>, f64, bool, bool, f64, f64, usize)> {
    let fit = build(&relationship, &status, &design)?
        .fit()
        .map_err(PyValueError::new_err)?;
    Ok((
        fit.heritability,
        fit.fixed_effects,
        fit.loglik,
        fit.converged,
        fit.polished,
        fit.scaled_gradient,
        fit.prevalence,
        fit.largest_family,
    ))
}

/// A 95 per cent profile interval for the liability heritability.
#[pyfunction]
pub fn liability_interval(
    relationship: PyReadonlyArray2<'_, f64>,
    status: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<(f64, f64, f64, bool, bool, usize)> {
    let got = build(&relationship, &status, &design)?
        .heritability_interval()
        .map_err(PyValueError::new_err)?;
    Ok((
        got.estimate,
        got.lower,
        got.upper,
        got.lower_at_bound,
        got.upper_at_bound,
        got.profile_failures,
    ))
}

/// Test the liability heritability against nought.
#[pyfunction]
pub fn liability_test(
    relationship: PyReadonlyArray2<'_, f64>,
    status: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<(f64, f64, String, f64)> {
    let test = build(&relationship, &status, &design)?
        .heritability_test()
        .map_err(PyValueError::new_err)?;
    Ok((
        test.statistic,
        test.p_value,
        test.rule.to_owned(),
        test.null_loglik,
    ))
}

/// The conditional region log-probability the censored models rest on.
///
/// **Exposed so that it can be checked.** This is the sequential truncation --
/// exact to two coordinates, Mendell-Elston above -- and every censored
/// heritability in the package rests on it. All the evidence for those models
/// was generated on pairs, where this routine's approximate branch never runs,
/// so the only way to learn how it behaves in a large family is to be able to
/// call it beside an independent reference. `checks/sequential_against_ghk.py`
/// is that reference.
///
/// `mean` is each coordinate's mean **already centred on its own limit**, and
/// `sign` is `1.0` where the value lies above that limit and `-1.0` where it
/// lies below, which is the convention `TobitModel` builds.
///
/// # Errors
///
/// Returns a stable code where the shapes disagree or the region cannot be
/// evaluated.
#[pyfunction]
pub fn region_log_probability(
    mean: PyReadonlyArray1<'_, f64>,
    sign: PyReadonlyArray1<'_, f64>,
    covariance: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let mean: Vec<f64> = mean.as_array().iter().copied().collect();
    let sign: Vec<f64> = sign.as_array().iter().copied().collect();
    let raw = covariance.as_array();
    let size = mean.len();
    if sign.len() != size || raw.shape()[0] != size || raw.shape()[1] != size {
        return Err(PyValueError::new_err("REGION_SHAPE_INVALID"));
    }
    let covariance = DMatrix::from_fn(size, size, |i, j| raw[(i, j)]);
    let normal = statrs::distribution::Normal::new(0.0, 1.0)
        .map_err(|_| PyValueError::new_err("REGION_NORMAL_INVALID"))?;
    LiabilityModel::region_log_probability(&mean, &sign, &covariance, &normal)
        .ok_or_else(|| PyValueError::new_err("REGION_NOT_EVALUABLE"))
}
