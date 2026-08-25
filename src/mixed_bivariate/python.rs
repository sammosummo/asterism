//! The Python interface to [`crate::mixed_bivariate`].

#![allow(clippy::needless_pass_by_value)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{MixedBivariateModel, TraitData, TraitKind};
use crate::tobit::python::censoring_from;
use nalgebra::DMatrix;

fn kind_from(code: i64) -> PyResult<TraitKind> {
    match code {
        0 => Ok(TraitKind::Continuous),
        1 => Ok(TraitKind::Binary),
        2 => Ok(TraitKind::Censored),
        _ => Err(PyValueError::new_err("MIXED_BIVARIATE_TRAIT_KIND_UNKNOWN")),
    }
}

fn trait_from(
    kind: i64,
    value: &PyReadonlyArray1<'_, f64>,
    censoring: &PyReadonlyArray1<'_, i64>,
    limit: &PyReadonlyArray1<'_, f64>,
) -> PyResult<TraitData> {
    Ok(TraitData {
        kind: kind_from(kind)?,
        value: value.as_array().iter().copied().collect(),
        censoring: censoring
            .as_array()
            .iter()
            .map(|c| censoring_from(*c))
            .collect::<PyResult<_>>()?,
        limit: limit.as_array().iter().copied().collect(),
    })
}

/// Fit two traits whose measurements need not be of the same kind.
///
/// `kind` is 0 for continuous, 1 for binary and 2 for censored. `censoring` is
/// 0 where measured, 1 for at or above the limit, 2 for at or below it. For a
/// binary trait, 1 is a case and the limit is nought, because the threshold is
/// carried by the intercept.
///
/// Returns both heritabilities, both total variances, the genetic and residual
/// correlations, both sets of fixed effects, the log likelihood, whether the
/// search converged, its scaled gradient, and the largest family.
#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn mixed_bivariate_fit(
    relationship: PyReadonlyArray2<'_, f64>,
    first_kind: i64,
    first_value: PyReadonlyArray1<'_, f64>,
    first_censoring: PyReadonlyArray1<'_, i64>,
    first_limit: PyReadonlyArray1<'_, f64>,
    second_kind: i64,
    second_value: PyReadonlyArray1<'_, f64>,
    second_censoring: PyReadonlyArray1<'_, i64>,
    second_limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<(
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    Vec<f64>,
    Vec<f64>,
    f64,
    bool,
    f64,
    usize,
)> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let first = trait_from(first_kind, &first_value, &first_censoring, &first_limit)?;
    let second = trait_from(second_kind, &second_value, &second_censoring, &second_limit)?;

    let fit = MixedBivariateModel::build(&a, first, second, &x)
        .map_err(PyValueError::new_err)?
        .fit()
        .map_err(PyValueError::new_err)?;
    Ok((
        fit.heritability.to_vec(),
        fit.total_variance.to_vec(),
        fit.genetic_correlation,
        fit.residual_correlation,
        fit.fixed_effects[0].clone(),
        fit.fixed_effects[1].clone(),
        fit.loglik,
        fit.converged,
        fit.scaled_gradient,
        fit.largest_family,
    ))
}

/// A 95 per cent profile-likelihood interval for one coordinate.
///
/// `coordinate` is 0 or 1 for the two heritabilities, 4 for the genetic
/// correlation and 5 for the residual one. The variances have no interval: a
/// binary trait's is fixed at one, so one would describe an assumption.
/// One correlation of the mixed bivariate model against nought.
///
/// `coordinate` is 4 for the genetic correlation and 5 for the residual one.
/// Nought is an interior point of a correlation's range, so the reference is a
/// The reference is a plain chi-square on one degree of freedom for an interior
/// null, and the Self-Liang even mixture where the null is plus or minus one.
#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn mixed_bivariate_test(
    relationship: PyReadonlyArray2<'_, f64>,
    first_kind: i64,
    first_value: PyReadonlyArray1<'_, f64>,
    first_censoring: PyReadonlyArray1<'_, i64>,
    first_limit: PyReadonlyArray1<'_, f64>,
    second_kind: i64,
    second_value: PyReadonlyArray1<'_, f64>,
    second_censoring: PyReadonlyArray1<'_, i64>,
    second_limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    coordinate: usize,
    null: f64,
) -> PyResult<(String, f64, f64, String, f64, f64)> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let first = trait_from(first_kind, &first_value, &first_censoring, &first_limit)?;
    let second = trait_from(second_kind, &second_value, &second_censoring, &second_limit)?;

    let got = MixedBivariateModel::build(&a, first, second, &x)
        .map_err(PyValueError::new_err)?
        .correlation_test(coordinate, null)
        .map_err(PyValueError::new_err)?;
    Ok((
        got.what.to_owned(),
        got.statistic,
        got.p_value,
        got.rule.to_owned(),
        got.null_loglik,
        got.alternative_loglik,
    ))
}

#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn mixed_bivariate_interval(
    relationship: PyReadonlyArray2<'_, f64>,
    first_kind: i64,
    first_value: PyReadonlyArray1<'_, f64>,
    first_censoring: PyReadonlyArray1<'_, i64>,
    first_limit: PyReadonlyArray1<'_, f64>,
    second_kind: i64,
    second_value: PyReadonlyArray1<'_, f64>,
    second_censoring: PyReadonlyArray1<'_, i64>,
    second_limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    coordinate: usize,
) -> PyResult<(
    String,
    f64,
    f64,
    f64,
    bool,
    bool,
    f64,
    usize,
    Option<bool>,
    Option<bool>,
)> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let first = trait_from(first_kind, &first_value, &first_censoring, &first_limit)?;
    let second = trait_from(second_kind, &second_value, &second_censoring, &second_limit)?;

    let got = MixedBivariateModel::build(&a, first, second, &x)
        .map_err(PyValueError::new_err)?
        .profile_interval(coordinate)
        .map_err(PyValueError::new_err)?;
    let what = match coordinate {
        super::HERITABILITY_ONE => "heritability_one",
        super::HERITABILITY_TWO => "heritability_two",
        super::GENETIC_CORRELATION => "genetic_correlation",
        super::RESIDUAL_CORRELATION => "residual_correlation",
        _ => {
            return Err(PyValueError::new_err(
                "MIXED_BIVARIATE_COORDINATE_HAS_NO_INTERVAL",
            ));
        }
    };
    Ok((
        what.to_owned(),
        got.estimate
            .ok_or_else(|| PyValueError::new_err("MIXED_BIVARIATE_PROFILE_NOT_EVALUABLE"))?,
        got.lower,
        got.upper,
        got.lower_limited,
        got.upper_limited,
        got.level,
        got.profile_failures,
        got.contains_lower_bound,
        got.contains_upper_bound,
    ))
}
