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
mod mixed_bivariate;
mod mixture_tail;
mod prepared;
mod relationship;
mod spatial;
mod tobit;
mod variant_set;

pub use association::{AssociationModel, CovariateEffect, MarkerTest, Variance};
pub use bivariate::{BivariateFit, BivariateHeritabilityBoundary, BivariateModel};
pub use components::{ComponentFit, ComponentModel};
pub use discrete_gxe::{DiscreteGxeFit, DiscreteGxeInterval, DiscreteGxeModel};
pub use gxe::{GxeFit, GxeModel, Surface};
pub use kinship_classes::{CLASS_NAMES, KinshipClasses};
pub use latent_mediation::{
    HorizontalSet, VerticalSet,
    HorizontalTest, LatentMediationDesign, LatentMediationEvaluation,
    LatentMediationFamilyEvaluation, LatentMediationFamilyInput, LatentMediationFit,
    LatentMediationModel, LatentMediationParameters, VerticalTest,
    simulate as simulate_latent_mediation,
};
pub use liability::{LiabilityFit, LiabilityModel};
pub use mixed_bivariate::{
    GENETIC_CORRELATION, HERITABILITY_ONE, HERITABILITY_TWO, MixedBivariateFit,
    MixedBivariateInterval, MixedBivariateModel, RESIDUAL_CORRELATION, TraitData, TraitKind,
};
pub use tobit::{Censoring, TobitFit, TobitInterval, TobitModel};
pub use mixture_tail::{MixtureTail, weighted_chi2_upper_tail};
pub use prepared::{Boundary, Fit, Interval, LikelihoodRatioTest, PreparedModel};
pub use relationship::{PedigreeError, Person, relationship_matrix};
pub use spatial::{SpatialFit, SpatialModel};
pub use variant_set::{VariantSetFamily, VariantSetModel, VariantSetTest};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedModel>()?;
    module.add_function(pyo3::wrap_pyfunction!(
        mixture_tail::python::py_weighted_chi2_upper_tail,
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
    module.add_function(pyo3::wrap_pyfunction!(
        discrete_gxe::python::discrete_gxe_fit,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        discrete_gxe::python::discrete_gxe_test,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        discrete_gxe::python::discrete_gxe_correlation_interval,
        module
    )?)?;
    module.add_class::<latent_mediation::python::PyLatentMediationCore>()?;
    module.add_function(pyo3::wrap_pyfunction!(
        latent_mediation::python::latent_mediation_simulate,
        module
    )?)?;
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
    module.add_function(pyo3::wrap_pyfunction!(
        liability::python::region_log_probability,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        variant_set::python::variant_set_scan,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        variant_set::python::variant_set_family_scan,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(tobit::python::tobit_fit, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        tobit::python::tobit_interval,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(tobit::python::tobit_test, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        mixed_bivariate::python::mixed_bivariate_test,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        mixed_bivariate::python::mixed_bivariate_fit,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        mixed_bivariate::python::mixed_bivariate_interval,
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
