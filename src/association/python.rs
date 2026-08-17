//! The Python interface to [`crate::association`].
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

use super::{AssociationModel, Variance};
use nalgebra::{DMatrix, DVector};

fn variance_named(name: &str) -> PyResult<Variance> {
    match name {
        "held" => Ok(Variance::Held),
        "refitted" => Ok(Variance::Refitted),
        other => Err(PyValueError::new_err(format!(
            "ASSOCIATION_UNKNOWN_VARIANCE: {other}, wanted held or refitted"
        ))),
    }
}

/// Sweep many markers through a polygenic model.
///
/// Returns the null heritability, the null log likelihood, and one row per
/// marker of effect, standard error, Wald statistic, likelihood ratio and
/// p-value. A marker that could not be tested comes back with a stable code
/// in place of its numbers rather than stopping the sweep.
///
/// **Wald and the likelihood ratio are the same number under `held`.** Both
/// are returned because both were asked for; they differ only under
/// `refitted`, which refits the variance components for every marker and is
/// far slower.
#[pyfunction]
#[pyo3(signature = (relationship, design, y, markers, variance="held", refit_below=None))]
#[allow(clippy::type_complexity)]
pub fn association_sweep(
    relationship: PyReadonlyArray2<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    markers: PyReadonlyArray2<'_, f64>,
    variance: &str,
    refit_below: Option<f64>,
) -> PyResult<(
    f64,
    f64,
    Vec<(f64, f64, f64, f64, f64, bool, String)>,
    Vec<(f64, f64, f64)>,
)> {
    let a = relationship.as_array();
    let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
    let x = design.as_array();
    let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
    let response = y.as_array();
    let response = DVector::from_iterator(response.len(), response.iter().copied());
    let m = markers.as_array();
    let m = DMatrix::from_fn(m.shape()[0], m.shape()[1], |i, j| m[(i, j)]);

    let model = AssociationModel::build(&a, &x, &response).map_err(PyValueError::new_err)?;
    let results = model
        .sweep_refitting_below(&m, variance_named(variance)?, refit_below)
        .map_err(PyValueError::new_err)?;
    let covariates = model
        .covariate_effects()
        .map_err(PyValueError::new_err)?
        .into_iter()
        .map(|c| (c.estimate, c.standard_error, c.p_value))
        .collect();
    Ok((
        model.heritability(),
        model.null_loglik(),
        results
            .into_iter()
            .map(|one| match one {
                Ok(test) => (
                    test.effect,
                    test.standard_error,
                    test.wald,
                    test.likelihood_ratio,
                    test.p_value,
                    test.refitted,
                    String::new(),
                ),
                Err(code) => (
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    false,
                    code.to_owned(),
                ),
            })
            .collect(),
        covariates,
    ))
}
