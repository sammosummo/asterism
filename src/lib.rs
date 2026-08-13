//! Asterism: variance components models for quantitative genetics.
//!
//! What Asterism is at any moment is what `docs/adr/` describes. Its reportable
//! surface is one trait, additive and residual variance, REML and ML, an
//! interval and a likelihood ratio test. The admitted two-trait Gaussian model
//! now has a checked fixed-state likelihood and convergent ML/REML optimisation;
//! its uncertainty, tests, coverage and JASA application remain unfinished.
//! Everything else is deferred, which means not yet rather than never.
//!
//! `CONTEXT.md` is the vocabulary. Read it before naming anything here.

mod bivariate;
mod blocks;
mod components;
mod dense;
mod kinship_classes;
mod prepared;
mod relationship;
mod spatial;

pub use bivariate::{BivariateFit, BivariateHeritabilityBoundary, BivariateModel};
pub use components::{ComponentFit, ComponentModel};
pub use kinship_classes::{CLASS_NAMES, KinshipClasses};
pub use prepared::{Boundary, Fit, Interval, LikelihoodRatioTest, PreparedModel};
pub use relationship::{relationship_matrix, PedigreeError, Person};
pub use spatial::{SpatialFit, SpatialModel};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedModel>()?;
    module.add_function(pyo3::wrap_pyfunction!(relationship::relationship, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(bivariate::bivariate_objective, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(bivariate::bivariate_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(bivariate::bivariate_interval, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(kinship_classes::kinship_classes, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(components::component_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(components::component_interval, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(components::component_test, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(components::component_blup, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_statistic, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_interval, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_bootstrap, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_distances, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        bivariate::bivariate_correlation_test,
        module
    )?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
