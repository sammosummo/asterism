//! The Python interface to [`crate::matrix_builders`].
//!
//! Kept beside the model rather than inside it, so the model file is the
//! mathematics and this file is the translation to and from Python.

use nalgebra::{DMatrix, DVector};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{
    MatrixBuildError, gene_burden_matrix, gene_linear_matrix, local_ibd_matrix,
    posterior_local_ibd_matrix,
};

fn matrix_from_array(array: &PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
    let view = array.as_array();
    DMatrix::from_row_iterator(view.nrows(), view.ncols(), view.iter().copied())
}

fn vector_from_array(array: &PyReadonlyArray1<'_, f64>) -> DVector<f64> {
    DVector::from_iterator(array.len().unwrap_or(0), array.as_array().iter().copied())
}

fn matrix_to_array(py: Python<'_>, values: &DMatrix<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let rows = values.nrows();
    let columns = values.ncols();
    let row_major: Vec<f64> = (0..rows)
        .flat_map(|row| (0..columns).map(move |column| values[(row, column)]))
        .collect();
    let array = numpy::ndarray::Array2::from_shape_vec((rows, columns), row_major)
        .map_err(|_| PyValueError::new_err("MATRIX_OUTPUT_SHAPE"))?;
    Ok(array.into_pyarray(py).unbind())
}

#[pyfunction(name = "gene_linear_matrix")]
#[pyo3(signature = (genotypes, variant_weights=None))]
pub fn build_gene_linear_matrix(
    py: Python<'_>,
    genotypes: PyReadonlyArray2<'_, f64>,
    variant_weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let genotypes = matrix_from_array(&genotypes);
    let variant_weights = variant_weights.as_ref().map(vector_from_array);
    let values = gene_linear_matrix(&genotypes, variant_weights.as_ref())
        .map_err(|error| PyValueError::new_err(error.code()))?;
    matrix_to_array(py, &values)
}

#[pyfunction(name = "gene_burden_matrix")]
#[pyo3(signature = (genotypes, variant_weights=None))]
pub fn build_gene_burden_matrix(
    py: Python<'_>,
    genotypes: PyReadonlyArray2<'_, f64>,
    variant_weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let genotypes = matrix_from_array(&genotypes);
    let variant_weights = variant_weights.as_ref().map(vector_from_array);
    let values = gene_burden_matrix(&genotypes, variant_weights.as_ref())
        .map_err(|error| PyValueError::new_err(error.code()))?;
    matrix_to_array(py, &values)
}

#[pyfunction(name = "local_ibd_matrix")]
pub fn build_local_ibd_matrix(
    py: Python<'_>,
    lineages: PyReadonlyArray2<'_, u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let view = lineages.as_array();
    if view.ncols() != 2 {
        return Err(PyValueError::new_err(
            MatrixBuildError::LineageWrongShape.code(),
        ));
    }
    let mut pairs = Vec::with_capacity(view.nrows());
    for row in view.rows() {
        pairs.push([row[0], row[1]]);
    }
    let values =
        local_ibd_matrix(&pairs).map_err(|error| PyValueError::new_err(error.code()))?;
    matrix_to_array(py, &values)
}

#[pyfunction(name = "posterior_local_ibd_matrix")]
#[pyo3(signature = (lineage_draws, draw_weights=None))]
pub fn build_posterior_local_ibd_matrix(
    py: Python<'_>,
    lineage_draws: PyReadonlyArray3<'_, u64>,
    draw_weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let view = lineage_draws.as_array();
    if view.shape()[2] != 2 {
        return Err(PyValueError::new_err(
            MatrixBuildError::LineageWrongShape.code(),
        ));
    }
    let mut draws = Vec::with_capacity(view.shape()[0]);
    for draw_index in 0..view.shape()[0] {
        let mut draw = Vec::with_capacity(view.shape()[1]);
        for row in 0..view.shape()[1] {
            let first = view[(draw_index, row, 0)];
            let second = view[(draw_index, row, 1)];
            draw.push([first, second]);
        }
        draws.push(draw);
    }
    let draw_weights = draw_weights.as_ref().map(vector_from_array);
    let values = posterior_local_ibd_matrix(&draws, draw_weights.as_ref())
        .map_err(|error| PyValueError::new_err(error.code()))?;
    matrix_to_array(py, &values)
}
