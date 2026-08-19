//! The Python interface to [`crate::autoregressive`].
//!
//! Kept beside the model rather than inside it, so the model file is the
//! mathematics and this file is the translation to and from Python.

// PyO3 extracts each argument from a Python object, so a `#[pyfunction]` takes
// them by value whether or not the body consumes them.
#![allow(clippy::needless_pass_by_value)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::AutoregressiveModel;
use nalgebra::{DMatrix, DVector};

fn owned(array: &PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
    let view = array.as_array();
    let (rows, columns) = (view.nrows(), view.ncols());
    DMatrix::from_fn(rows, columns, |i, j| view[[i, j]])
}

fn response(array: &PyReadonlyArray1<'_, f64>) -> DVector<f64> {
    let view = array.as_array();
    DVector::from_fn(view.len(), |i, _| view[i])
}

/// Fit one trait with fixed components and a separable autoregressive grid.
///
/// `row` and `column` are the cell coordinates of each person, as whole
/// numbers of cells. The grid's spacing lives in how the caller made them, so
/// the two rates that come back are correlations *per cell* and mean nothing
/// without the cell size that produced them.
#[pyfunction]
#[pyo3(signature = (fixed, row, column, design, y, reml=true))]
#[allow(clippy::type_complexity)]
pub fn autoregressive_fit(
    fixed: Vec<PyReadonlyArray2<'_, f64>>,
    row: Vec<i64>,
    column: Vec<i64>,
    design: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    reml: bool,
) -> PyResult<(
    Vec<f64>,
    Vec<f64>,
    f64,
    // The two rates and the two half-distances travel together, as one item,
    // because PyO3 stops at twelve and this tuple is already long.
    (f64, f64, f64, f64),
    f64,
    bool,
    bool,
    f64,
    Vec<f64>,
    Vec<f64>,
)> {
    let matrices: Vec<DMatrix<f64>> = fixed.iter().map(owned).collect();
    let model = AutoregressiveModel::build(&matrices, &row, &column, &owned(&design))
        .map_err(PyValueError::new_err)?;
    let fit = model
        .fit(&response(&y), reml)
        .map_err(PyValueError::new_err)?;
    Ok((
        fit.variances,
        fit.raw_coefficient_proportions,
        fit.raw_coefficient_total,
        (
            fit.rho_row,
            fit.rho_column,
            fit.half_cells_row,
            fit.half_cells_column,
        ),
        fit.loglik,
        fit.converged,
        fit.polished,
        fit.scaled_gradient,
        fit.fixed_effects,
        fit.fixed_effect_errors,
    ))
}

/// How many distinct cells a set of coordinates holds.
///
/// Worth having on its own: a grid so fine that almost every person is alone
/// in a cell has nothing to say about neighbours, and a grid so coarse that
/// everybody shares one cell has nothing to say at all. This is how a caller
/// checks before paying for a fit.
#[pyfunction]
pub fn autoregressive_cells(row: Vec<i64>, column: Vec<i64>) -> PyResult<usize> {
    if row.len() != column.len() {
        return Err(PyValueError::new_err("AUTOREGRESSIVE_CELLS_WRONG_LENGTH"));
    }
    let mut seen: Vec<(i64, i64)> = Vec::new();
    for (r, c) in row.iter().zip(column.iter()) {
        if !seen.contains(&(*r, *c)) {
            seen.push((*r, *c));
        }
    }
    Ok(seen.len())
}
