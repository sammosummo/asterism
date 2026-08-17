//! The Python interface to [`crate::variant_set`].

// PyO3 extracts each argument from a Python object, so a `#[pyfunction]` takes
// them by value whether or not the body consumes them. The lint cannot be
// satisfied here without breaking the macro.
#![allow(clippy::needless_pass_by_value)]

use nalgebra::{DMatrix, DVector};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::VariantSetModel;

fn matrix(array: &PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
    let view = array.as_array();
    DMatrix::from_fn(view.shape()[0], view.shape()[1], |i, j| view[(i, j)])
}

/// Fit the null once and score every variant set against it.
///
/// `backgrounds` are the covariance bases carrying everything that is not the
/// variant set. Each entry of `roots` is `Z = G W` for one set: the dosages
/// with their column weights already applied. Returns one record per set.
#[pyfunction]
#[pyo3(name = "variant_set_scan", signature = (backgrounds, design, response, roots, reml=true))]
pub fn variant_set_scan<'py>(
    py: Python<'py>,
    backgrounds: Vec<PyReadonlyArray2<'_, f64>>,
    design: PyReadonlyArray2<'_, f64>,
    response: PyReadonlyArray1<'_, f64>,
    roots: Vec<PyReadonlyArray2<'_, f64>>,
    reml: bool,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let background: Vec<DMatrix<f64>> = backgrounds.iter().map(matrix).collect();
    let x = matrix(&design);
    let view = response.as_array();
    let y = DVector::from_iterator(view.len(), view.iter().copied());
    let model = VariantSetModel::build(&background, &x, &y, reml).map_err(PyValueError::new_err)?;

    let mut out = Vec::with_capacity(roots.len());
    for root in &roots {
        let record = PyDict::new(py);
        match model.test(&matrix(root)) {
            Ok(test) => {
                record.set_item("statistic", test.statistic)?;
                record.set_item("p_value", test.p_value)?;
                record.set_item("eigenvalues", test.eigenvalues)?;
                record.set_item("tail_method", test.tail_method)?;
                record.set_item("trustworthy", test.trustworthy)?;
                record.set_item("code", "")?;
            }
            Err(code) => {
                record.set_item("statistic", f64::NAN)?;
                record.set_item("p_value", f64::NAN)?;
                record.set_item("eigenvalues", Vec::<f64>::new())?;
                record.set_item("tail_method", "")?;
                record.set_item("trustworthy", false)?;
                record.set_item("code", code)?;
            }
        }
        out.push(record);
    }
    Ok(out)
}

/// Test each set across a family of assumed effect correlations and combine.
#[pyfunction]
#[pyo3(name = "variant_set_family_scan",
       signature = (backgrounds, design, response, roots, correlations, reml=true))]
#[allow(clippy::too_many_arguments)]
pub fn variant_set_family_scan<'py>(
    py: Python<'py>,
    backgrounds: Vec<PyReadonlyArray2<'_, f64>>,
    design: PyReadonlyArray2<'_, f64>,
    response: PyReadonlyArray1<'_, f64>,
    roots: Vec<PyReadonlyArray2<'_, f64>>,
    correlations: Vec<f64>,
    reml: bool,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let background: Vec<DMatrix<f64>> = backgrounds.iter().map(matrix).collect();
    let x = matrix(&design);
    let view = response.as_array();
    let y = DVector::from_iterator(view.len(), view.iter().copied());
    let model = VariantSetModel::build(&background, &x, &y, reml).map_err(PyValueError::new_err)?;

    let mut out = Vec::with_capacity(roots.len());
    for root in &roots {
        let record = PyDict::new(py);
        match model.test_family(&matrix(root), &correlations) {
            Ok(family) => {
                record.set_item("p_value", family.p_value)?;
                record.set_item("correlations", family.correlations.clone())?;
                record.set_item("strongest_correlation", family.strongest_correlation)?;
                record.set_item(
                    "p_values",
                    family.tests.iter().map(|t| t.p_value).collect::<Vec<_>>(),
                )?;
                record.set_item(
                    "statistics",
                    family.tests.iter().map(|t| t.statistic).collect::<Vec<_>>(),
                )?;
                record.set_item("trustworthy", family.trustworthy)?;
                record.set_item("code", "")?;
            }
            Err(code) => {
                record.set_item("p_value", f64::NAN)?;
                record.set_item("correlations", Vec::<f64>::new())?;
                record.set_item("strongest_correlation", f64::NAN)?;
                record.set_item("p_values", Vec::<f64>::new())?;
                record.set_item("statistics", Vec::<f64>::new())?;
                record.set_item("trustworthy", false)?;
                record.set_item("code", code)?;
            }
        }
        out.push(record);
    }
    Ok(out)
}
