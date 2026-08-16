//! Asterism: variance components models for quantitative genetics.
//!
//! The crate implements Gaussian variance-component models, specialised
//! quantitative-genetic likelihoods, and numerical relationship-matrix
//! constructors. Statistical definitions and limitations are described in
//! `docs/statistical-methods.md`.

mod association;
mod bivariate;
mod blocks;
mod components;
mod dense;
mod deviance;
mod discrete_gxe;
mod gxe;
mod kinship_classes;
mod latent_mediation;
mod liability;
mod matrix_builders;
mod prepared;
mod relationship;
mod spatial;

pub use association::{AssociationModel, CovariateEffect, MarkerTest, Variance};
pub use bivariate::{BivariateFit, BivariateHeritabilityBoundary, BivariateModel};
pub use components::{ComponentFit, ComponentModel};
pub use gxe::{GxeFit, GxeModel, Surface};
pub use discrete_gxe::{DiscreteGxeFit, DiscreteGxeModel};
pub use kinship_classes::{CLASS_NAMES, KinshipClasses};
pub use latent_mediation::{
    LatentMediationEvaluation, LatentMediationFamilyEvaluation, LatentMediationFamilyInput,
    LatentMediationFit, LatentMediationModel, LatentMediationParameters,
};
pub use liability::{LiabilityFit, LiabilityModel};
pub use matrix_builders::{MatrixBuildError, gene_burden_matrix, gene_linear_matrix};
pub use prepared::{Boundary, Fit, Interval, LikelihoodRatioTest, PreparedModel};
pub use relationship::{PedigreeError, Person, relationship_matrix};
pub use spatial::{SpatialFit, SpatialModel};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedModel>()?;
    module.add_function(pyo3::wrap_pyfunction!(
        matrix_builders::python::build_gene_linear_matrix,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        matrix_builders::python::build_gene_burden_matrix,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(relationship::relationship, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        bivariate::bivariate_objective,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(bivariate::bivariate_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        bivariate::bivariate_interval,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        kinship_classes::kinship_classes,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(components::component_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        components::component_interval,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(components::component_test, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        components::component_equality_test,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        components::component_contrasts,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(components::component_blup, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(gxe::python::gxe_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(gxe::python::gxe_test, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(gxe::python::gxe_interval, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        association::python::association_sweep,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(discrete_gxe::python::discrete_gxe_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(discrete_gxe::python::discrete_gxe_test, module)?)?;
    module.add_class::<latent_mediation::python::PyLatentMediationCore>()?;
    module.add_function(pyo3::wrap_pyfunction!(
        liability::python::liability_fit,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        liability::python::liability_interval,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        liability::python::liability_test,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_statistic, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_interval, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_bootstrap, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_distances, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(spatial::spatial_blup, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        bivariate::bivariate_correlation_test,
        module
    )?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
