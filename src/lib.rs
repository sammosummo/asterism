//! Asterism: variance components models for quantitative genetics.
//!
//! What Asterism is at any moment is what `docs/adr/` describes. Today that is
//! one trait, additive and residual variance, REML and ML, an interval and a
//! likelihood ratio test. Everything else is deferred, which means not yet
//! rather than never, and a capability enters only when a planned analysis
//! needs it and the decision record has said so first.
//!
//! `CONTEXT.md` is the vocabulary. Read it before naming anything here.

mod blocks;
mod prepared;
mod relationship;

pub use prepared::{Boundary, Fit, Interval, LikelihoodRatioTest, PreparedModel};
pub use relationship::{relationship_matrix, PedigreeError, Person};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedModel>()?;
    module.add_function(pyo3::wrap_pyfunction!(relationship::relationship, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
