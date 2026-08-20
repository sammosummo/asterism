//! The Python interface to [`crate::tobit`].
//!
//! Kept beside the model rather than inside it, so the model file is the
//! mathematics and this file is the translation to and from Python.

#![allow(clippy::needless_pass_by_value)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{Censoring, TobitModel};
use nalgebra::DMatrix;

/// Censoring arrives as a small integer, because a string per observation is
/// a great deal of allocation for three possibilities.
pub(crate) fn censoring_from(code: i64) -> PyResult<Censoring> {
    match code {
        0 => Ok(Censoring::Measured),
        1 => Ok(Censoring::Above),
        2 => Ok(Censoring::Below),
        _ => Err(PyValueError::new_err("TOBIT_CENSORING_CODE_UNKNOWN")),
    }
}

/// Fit one trait whose measurement stops at a limit.
///
/// `censoring` is 0 where the value was measured, 1 where it lies at or above
/// its limit, and 2 where it lies at or below it.
///
/// Returns the heritability of the complete variable, its total variance, the
/// fixed effects, the log likelihood, whether the search converged, its scaled
/// gradient, the share of observations that hit a limit, and the largest
/// family the region probability had to cover.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn tobit_fit(
    relationship: PyReadonlyArray2<'_, f64>,
    value: PyReadonlyArray1<'_, f64>,
    censoring: PyReadonlyArray1<'_, i64>,
    limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<(f64, f64, Vec<f64>, f64, bool, f64, f64, usize)> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let value: Vec<f64> = value.as_array().iter().copied().collect();
    let limit: Vec<f64> = limit.as_array().iter().copied().collect();
    let censoring: Vec<Censoring> = censoring
        .as_array()
        .iter()
        .map(|c| censoring_from(*c))
        .collect::<PyResult<_>>()?;

    let fit = TobitModel::build(&a, &value, &censoring, &limit, &x)
        .map_err(PyValueError::new_err)?
        .fit()
        .map_err(PyValueError::new_err)?;
    Ok((
        fit.heritability,
        fit.total_variance,
        fit.fixed_effects,
        fit.loglik,
        fit.converged,
        fit.scaled_gradient,
        fit.censored_share,
        fit.largest_family,
    ))
}

/// A 95 per cent profile-likelihood interval for the heritability.
///
/// Returns the estimate, the two ends, whether each end sits on the
/// parameter's own bound rather than where the profile fell away, how many
/// profile fits failed, and the share of observations that hit a limit.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn tobit_interval(
    relationship: PyReadonlyArray2<'_, f64>,
    value: PyReadonlyArray1<'_, f64>,
    censoring: PyReadonlyArray1<'_, i64>,
    limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<(f64, f64, f64, bool, bool, f64, Option<bool>, Option<bool>, usize, f64)> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let value: Vec<f64> = value.as_array().iter().copied().collect();
    let limit: Vec<f64> = limit.as_array().iter().copied().collect();
    let censoring: Vec<Censoring> = censoring
        .as_array()
        .iter()
        .map(|c| censoring_from(*c))
        .collect::<PyResult<_>>()?;

    let got = TobitModel::build(&a, &value, &censoring, &limit, &x)
        .map_err(PyValueError::new_err)?
        .heritability_interval()
        .map_err(PyValueError::new_err)?;
    Ok((
        got.estimate,
        got.lower,
        got.upper,
        got.lower_at_bound,
        got.upper_at_bound,
        got.level,
        got.contains_lower_bound,
        got.contains_upper_bound,
        got.profile_failures,
        got.censored_share,
    ))
}

/// The censored heritability against nought.
///
/// Returns the statistic, the p-value, the reference rule, and the two log
/// likelihoods the statistic was made from.
#[pyfunction]
pub fn tobit_test(
    relationship: PyReadonlyArray2<'_, f64>,
    value: PyReadonlyArray1<'_, f64>,
    censoring: PyReadonlyArray1<'_, i64>,
    limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<(f64, f64, String, f64, f64)> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let value: Vec<f64> = value.as_array().iter().copied().collect();
    let limit: Vec<f64> = limit.as_array().iter().copied().collect();
    let censoring: Vec<Censoring> = censoring
        .as_array()
        .iter()
        .map(|c| censoring_from(*c))
        .collect::<PyResult<_>>()?;

    let got = TobitModel::build(&a, &value, &censoring, &limit, &x)
        .map_err(PyValueError::new_err)?
        .heritability_test()
        .map_err(PyValueError::new_err)?;
    Ok((
        got.statistic,
        got.p_value,
        got.rule.to_owned(),
        got.null_loglik,
        got.alternative_loglik,
    ))
}
