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

    let fit = TobitModel::build(std::slice::from_ref(&a), &value, &censoring, &limit, &x)
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
) -> PyResult<(
    f64,
    f64,
    f64,
    bool,
    bool,
    f64,
    Option<bool>,
    Option<bool>,
    usize,
    f64,
)> {
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

    let model = TobitModel::build(std::slice::from_ref(&a), &value, &censoring, &limit, &x)
        .map_err(PyValueError::new_err)?;
    let got = model
        .heritability_interval()
        .map_err(PyValueError::new_err)?;
    Ok((
        got.estimate
            .ok_or_else(|| PyValueError::new_err("TOBIT_PROFILE_NOT_EVALUABLE"))?,
        got.lower,
        got.upper,
        got.lower_limited,
        got.upper_limited,
        got.level,
        got.contains_lower_bound,
        got.contains_upper_bound,
        got.profile_failures,
        model.censored_share(),
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

    let got = TobitModel::build(std::slice::from_ref(&a), &value, &censoring, &limit, &x)
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

/// Turn the Python matrices into the model's own storage.
fn components_from(components: &[PyReadonlyArray2<'_, f64>]) -> Vec<DMatrix<f64>> {
    components
        .iter()
        .map(|component| {
            let view = component.as_array();
            DMatrix::from_fn(view.shape()[0], view.shape()[1], |i, j| view[(i, j)])
        })
        .collect()
}

/// The censored data every entry point below reads the same way.
type CensoredInputs = (Vec<f64>, Vec<Censoring>, Vec<f64>, DMatrix<f64>);

/// Read them once, so three entry points cannot drift on how they do it.
fn inputs(
    value: &PyReadonlyArray1<'_, f64>,
    censoring: &PyReadonlyArray1<'_, i64>,
    limit: &PyReadonlyArray1<'_, f64>,
    design: &PyReadonlyArray2<'_, f64>,
) -> PyResult<CensoredInputs> {
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let censoring: Vec<Censoring> = censoring
        .as_array()
        .iter()
        .map(|c| censoring_from(*c))
        .collect::<PyResult<_>>()?;
    Ok((
        value.as_array().iter().copied().collect(),
        censoring,
        limit.as_array().iter().copied().collect(),
        x,
    ))
}

/// Fit one censored trait with any number of variance components.
///
/// Returns the coefficients, the mean-diagonal proportions where they are
/// defined, the total variance, the fixed effects, the log likelihood, whether
/// the search converged, its scaled gradient, the censored share and the
/// largest block the region probability had to cover.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn censored_component_fit(
    components: Vec<PyReadonlyArray2<'_, f64>>,
    value: PyReadonlyArray1<'_, f64>,
    censoring: PyReadonlyArray1<'_, i64>,
    limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<(
    Vec<f64>,
    Option<Vec<f64>>,
    f64,
    Vec<f64>,
    f64,
    bool,
    f64,
    f64,
    usize,
)> {
    let (value, censoring, limit, x) = inputs(&value, &censoring, &limit, &design)?;
    let fit = TobitModel::build(
        &components_from(&components),
        &value,
        &censoring,
        &limit,
        &x,
    )
    .map_err(PyValueError::new_err)?
    .fit()
    .map_err(PyValueError::new_err)?;
    Ok((
        fit.coefficients,
        fit.mean_diagonal_proportions,
        fit.total_variance,
        fit.fixed_effects,
        fit.loglik,
        fit.converged,
        fit.scaled_gradient,
        fit.censored_share,
        fit.largest_family,
    ))
}

/// A profile-likelihood interval for one component.
///
/// With `proportion` true the interval is on the component's mean-diagonal
/// proportion, which is the comparable quantity; with it false, on the raw
/// coefficient.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn censored_component_interval(
    components: Vec<PyReadonlyArray2<'_, f64>>,
    value: PyReadonlyArray1<'_, f64>,
    censoring: PyReadonlyArray1<'_, i64>,
    limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    index: usize,
    proportion: bool,
) -> PyResult<(
    f64,
    f64,
    f64,
    bool,
    bool,
    f64,
    Option<bool>,
    Option<bool>,
    usize,
)> {
    let (value, censoring, limit, x) = inputs(&value, &censoring, &limit, &design)?;
    let model = TobitModel::build(
        &components_from(&components),
        &value,
        &censoring,
        &limit,
        &x,
    )
    .map_err(PyValueError::new_err)?;
    let got = if proportion {
        model.mean_diagonal_interval(index)
    } else {
        model.coefficient_interval(index)
    }
    .map_err(PyValueError::new_err)?;
    Ok((
        got.estimate
            .ok_or_else(|| PyValueError::new_err("TOBIT_PROFILE_NOT_EVALUABLE"))?,
        got.lower,
        got.upper,
        got.lower_limited,
        got.upper_limited,
        got.level,
        got.contains_lower_bound,
        got.contains_upper_bound,
        got.profile_failures,
    ))
}

/// A test that one component's coefficient is nought.
///
/// Returns the statistic, the p-value, the reference rule and the two log
/// likelihoods. The released one-component route retains `mixture_50_50`;
/// several-component results say `asymptotic_mixture_50_50` and refuse where a
/// nuisance component or the residual rests on a bound.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn censored_component_test(
    components: Vec<PyReadonlyArray2<'_, f64>>,
    value: PyReadonlyArray1<'_, f64>,
    censoring: PyReadonlyArray1<'_, i64>,
    limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    index: usize,
) -> PyResult<(f64, f64, String, f64, f64, bool)> {
    let (value, censoring, limit, x) = inputs(&value, &censoring, &limit, &design)?;
    let got = TobitModel::build(
        &components_from(&components),
        &value,
        &censoring,
        &limit,
        &x,
    )
    .map_err(PyValueError::new_err)?
    .coefficient_test(index)
    .map_err(PyValueError::new_err)?;
    Ok((
        got.statistic,
        got.p_value,
        got.rule.to_owned(),
        got.null_loglik,
        got.alternative_loglik,
        got.nuisance_at_bound,
    ))
}

/// A constrained-null parametric bootstrap for one component's statistic.
///
/// `direction` declares right or left censoring for every row, including rows
/// measured in the observed response. Returns the observed statistic, the
/// exceedance count, the complete requested denominator, the add-one p-value,
/// the reference rule, and the observed nuisance-boundary diagnostic.
#[pyfunction]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn censored_component_bootstrap(
    py: Python<'_>,
    components: Vec<PyReadonlyArray2<'_, f64>>,
    value: PyReadonlyArray1<'_, f64>,
    censoring: PyReadonlyArray1<'_, i64>,
    limit: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    direction: PyReadonlyArray1<'_, i64>,
    index: usize,
    replicates: usize,
    seed: u64,
) -> PyResult<(f64, usize, usize, usize, f64, String, f64, f64, bool)> {
    let (owned_value, owned_censoring, owned_limit, x) =
        inputs(&value, &censoring, &limit, &design)?;
    let directions: Vec<Censoring> = direction
        .as_array()
        .iter()
        .map(|code| censoring_from(*code))
        .collect::<PyResult<_>>()?;
    let component_matrices = components_from(&components);
    let model = TobitModel::build(
        &component_matrices,
        &owned_value,
        &owned_censoring,
        &owned_limit,
        &x,
    )
    .map_err(PyValueError::new_err)?;
    // `build` owns its copies. Release every NumPy borrow and the temporary
    // matrices while the GIL is still held, then detach the hours-long Rust
    // calculation so unrelated Python threads remain usable.
    drop(component_matrices);
    drop(components);
    drop(value);
    drop(censoring);
    drop(limit);
    drop(design);
    drop(direction);
    let got = py
        .detach(move || model.bootstrap_component_test(index, &directions, replicates, seed))
        .map_err(PyValueError::new_err)?;
    Ok((
        got.observed,
        got.exceedances,
        got.replicates,
        got.requested,
        got.p_value,
        got.rule.to_owned(),
        got.null_loglik,
        got.alternative_loglik,
        got.nuisance_at_bound,
    ))
}
