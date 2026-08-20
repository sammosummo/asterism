//! The Python interface to [`crate::repeated`].
//!
//! Kept beside the model rather than inside it, so the model file is the
//! mathematics and this file is the translation to and from Python.
//!
//! **This one is a class where the other censored models are functions**, for
//! the reason ADR 0010 gives: preparing it does an eigendecomposition per
//! family and works out the family blocks, and none of that depends on the
//! response. A caller fitting the same roster twice should pay for it once.

#![allow(clippy::needless_pass_by_value)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{Known, RepeatedModel};
use nalgebra::DMatrix;

/// How the fit went, as against what it found: iterations, whether the
/// likelihood ever fell, whether the gradient test passed, the largest family
/// and the dimension the sequential approximation reached.
///
/// Bundled because a Python tuple built by PyO3 stops at twelve entries and
/// this record has fourteen things worth returning. The split is where it would
/// have been anyway.
type Diagnostics = (usize, bool, bool, usize, usize);

/// Everything one fit returns, flattened for the trip across.
///
/// Matrices come back row by row rather than as arrays, because the shapes are
/// known on the other side and reshaping there is one line where building numpy
/// arrays here is several.
type FitRecord = (
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    Vec<f64>,
    Diagnostics,
);

/// A prepared repeated-measures model.
#[pyclass(name = "RepeatedCore")]
pub struct PyRepeatedCore {
    model: RepeatedModel,
}

#[pymethods]
impl PyRepeatedCore {
    /// Prepare a model.
    ///
    /// `matrices` is `(components, people, people)`. `design` is
    /// `(people * replicates, covariates)`, in person-major order. `line`, if
    /// given, is where each position sits on the caller's own scale, and turns
    /// every covariance into a variance per position with a floor and a rate.
    #[new]
    #[pyo3(signature = (matrices, design, replicates, positions, line = None))]
    fn new(
        matrices: PyReadonlyArray3<'_, f64>,
        design: PyReadonlyArray2<'_, f64>,
        replicates: usize,
        positions: usize,
        line: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let raw = matrices.as_array();
        let shape = raw.shape();
        let matrices: Vec<DMatrix<f64>> = (0..shape[0])
            .map(|k| DMatrix::from_fn(shape[1], shape[2], |i, j| raw[(k, i, j)]))
            .collect();
        let x = design.as_array();
        let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);

        let model = match line {
            Some(line) => {
                let line: Vec<f64> = line.as_array().iter().copied().collect();
                if line.len() != positions {
                    return Err(PyValueError::new_err("REPEATED_LINE_WRONG_LENGTH"));
                }
                RepeatedModel::build_on_a_line(&matrices, &x, replicates, &line)
            }
            None => RepeatedModel::build(&matrices, &x, replicates, positions),
        }
        .map_err(PyValueError::new_err)?;
        Ok(Self { model })
    }

    /// Fit a response.
    ///
    /// All three arrays are `(people * replicates, positions)`. `censoring` is
    /// 0 where the value was measured, 1 where it lies at or above its limit, 2
    /// where it lies at or below it, and **3 where it was never measured at
    /// all**. `value` is read only where the status says measured, and `limit`
    /// only where it says censored.
    fn fit(
        &self,
        value: PyReadonlyArray2<'_, f64>,
        censoring: PyReadonlyArray2<'_, i64>,
        limit: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<FitRecord> {
        let value = value.as_array();
        let censoring = censoring.as_array();
        let limit = limit.as_array();
        let shape = value.shape();
        if censoring.shape() != shape || limit.shape() != shape {
            return Err(PyValueError::new_err("REPEATED_INPUTS_DIFFERENT_SHAPES"));
        }
        let mut known = Vec::with_capacity(shape[0] * shape[1]);
        for row in 0..shape[0] {
            for position in 0..shape[1] {
                known.push(match censoring[(row, position)] {
                    0 => Known::Value(value[(row, position)]),
                    1 => Known::Above(limit[(row, position)]),
                    2 => Known::Below(limit[(row, position)]),
                    3 => Known::Missing,
                    _ => return Err(PyValueError::new_err("REPEATED_CENSORING_CODE_UNKNOWN")),
                });
            }
        }

        let fit = self
            .model
            .fit_known(&known)
            .map_err(PyValueError::new_err)?;
        let flatten = |matrix: &DMatrix<f64>| -> Vec<f64> {
            let mut out = Vec::with_capacity(matrix.nrows() * matrix.ncols());
            for i in 0..matrix.nrows() {
                for j in 0..matrix.ncols() {
                    out.push(matrix[(i, j)]);
                }
            }
            out
        };
        Ok((
            fit.component_covariances.iter().map(flatten).collect(),
            flatten(&fit.residual_covariance),
            flatten(&fit.fixed_effects),
            fit.variance_shares,
            fit.floors,
            fit.rates,
            fit.loglik,
            fit.scaled_gradient,
            fit.censored_shares,
            (
                fit.iterations,
                fit.monotone,
                fit.converged,
                fit.largest_family,
                fit.sequential_dimension,
            ),
        ))
    }
}
