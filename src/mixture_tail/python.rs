//! The Python interface to [`crate::mixture_tail`].

// PyO3 extracts each argument from a Python object, so a `#[pyfunction]` takes
// them by value whether or not the body consumes them. The lint cannot be
// satisfied here without breaking the macro.
#![allow(clippy::needless_pass_by_value)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::weighted_chi2_upper_tail;

/// The upper tail of a weighted sum of chi-squares on one degree of freedom.
///
/// `weights` are the eigenvalues of the tested matrix after projection, so
/// they are nonnegative. Returns the probability, which recipe answered, how
/// far the series had settled, and whether cancellation has left enough digits
/// for the value to be read as a probability.
#[pyfunction]
#[pyo3(name = "weighted_chi2_upper_tail")]
pub fn py_weighted_chi2_upper_tail<'py>(
    py: Python<'py>,
    q: f64,
    weights: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let tail = weighted_chi2_upper_tail(q, &weights).map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("probability", tail.probability)?;
    out.set_item("method", tail.method)?;
    out.set_item("settled_to", tail.settled_to)?;
    out.set_item("trustworthy", tail.trustworthy)?;
    Ok(out)
}
