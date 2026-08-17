//! Integrated fixed-point likelihood for the latent mediation model.
//!
//! This is a fixed special case.  The latent vector is always
//! process-major, with mediator entries followed by outcome entries; the
//! relationship matrix is an additive relationship with a diagonal of one;
//! the cross-process covariance is fixed at zero; the outcome innovation
//! variance is fixed at one; and there is no household covariance. Python
//! prepares in-memory values and gives the public fields generic names, while
//! every likelihood calculation in this module is Rust-owned.

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};
use statrs::distribution::{ContinuousCDF, Normal};

const LOG_TWO_PI: f64 = 1.837_877_066_409_345_3;
const LOG_HALF: f64 = -std::f64::consts::LN_2;
const TOLERANCE: f64 = 1.0e-10;
const RELATIONSHIP_PSD_FLOOR: f64 = -1.0e-9;
const MAXIMUM_OBSERVED_MEDIATOR_PROXY: usize = 12;
const MAXIMUM_DISCRETE_DIMENSION: usize = 25;
// Below this absolute scale, the corner-difference recipe cannot distinguish
// a probability from numerical zero and the log-scale conditional quadrature
// takes over.
const BIVARIATE_ABSOLUTE_TOLERANCE: f64 = 1.0e-12;
// Corner differences carry roughly sixteen epsilons of rounding, so their
// relative accuracy degrades as the probability shrinks; below this scale the
// log-scale conditional quadrature is the more accurate recipe.
const BIVARIATE_TAIL_CROSSOVER: f64 = 1.0e-9;
/// How many doubling strides the conditional integrand is followed for, when
/// climbing to its peak and again when leaving it. Each stride doubles, so two
/// hundred reaches any distance binary64 holds and the cap is only a guard
/// against a non-terminating climb.
const CLIMB_STEPS: usize = 200;
const FIT_DIMENSION: usize = 5;
const FIT_GRADIENT_STEP: f64 = 1.0e-5;
const FIT_GRADIENT_STABILITY_TOLERANCE: f64 = 1.0e-4;
const FIT_GRADIENT_TOLERANCE: f64 = 1.0e-5;
const FIT_SOLVER_PGTOL: f64 = 1.0e-6;
const FIT_BOUND_TOLERANCE: f64 = 1.0e-12;
const FIT_SELECTION_TOLERANCE: f64 = 1.0e-10;
const FIT_MAXIMUM_STENCIL_HALVINGS: usize = 8;
const PRIMES: [usize; 24] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
];
const SHIFT_PRIMES: [usize; 24] = [
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
    193, 197, 199, 211, 223,
];
const QMC_REPLICATES: usize = 8;

/// The five free structural parameters in the latent mediation model.
#[derive(Clone, Copy, Debug)]
pub struct LatentMediationParameters {
    pub a: f64,
    pub b: f64,
    pub c_prime: f64,
    pub d: f64,
    pub sigma_m2: f64,
}

/// One connected family before validation.
#[derive(Clone, Debug)]
pub struct LatentMediationFamilyInput {
    pub relationship: Vec<Vec<f64>>,
    pub latent_mean: Vec<f64>,
    /// Covariates acting on the latent mediator, one row per person. Empty for
    /// none. Every family must offer the same columns in the same order,
    /// because their coefficients are one set estimated across all of them.
    pub mediator_design: Vec<Vec<f64>>,
    /// Covariates acting on the latent outcome, one row per person. Separate
    /// from the mediator's because the two rarely want the same terms -- age
    /// belongs on dementia risk whether or not it belongs on hearing.
    pub outcome_design: Vec<Vec<f64>>,
    pub mediator_measurement: Vec<Option<f64>>,
    pub mediator_measurement_error_variance: Vec<Option<f64>>,
    pub mediator_proxy_status: Vec<Option<i8>>,
    pub outcome_status: Vec<Option<i8>>,
    pub mediator_threshold: Vec<f64>,
    pub outcome_threshold: Vec<f64>,
    pub mediator_proxy_sensitivity: Vec<f64>,
    pub mediator_proxy_specificity: Vec<f64>,
    pub ascertainment: String,
    pub proband_index: Option<usize>,
}

#[derive(Clone, Debug)]
struct LatentMediationFamily {
    relationship: DMatrix<f64>,
    latent_mean: DVector<f64>,
    /// The two designs stacked to match the latent vector: mediator covariates
    /// in the first `n` rows and their own columns, outcome covariates in the
    /// last `n` rows and theirs. One matrix, so the mean a set of coefficients
    /// implies is one multiplication.
    design: DMatrix<f64>,
    observed_mediator_measurement_indices: Vec<usize>,
    observed_mediator_measurement_values: Vec<f64>,
    observed_mediator_measurement_error_variances: Vec<f64>,
    observed_mediator_proxy: Vec<(usize, i8)>,
    observed_outcome: Vec<(usize, i8)>,
    discrete_target_indices: Vec<usize>,
    mediator_threshold: Vec<f64>,
    outcome_threshold: Vec<f64>,
    mediator_proxy_sensitivity: Vec<f64>,
    mediator_proxy_specificity: Vec<f64>,
    ascertainment: Ascertainment,
}

#[derive(Clone, Copy, Debug)]
enum Ascertainment {
    PopulationUnconditioned,
    NamedProbandCase(usize),
}

/// Numerical diagnostics for one connected family.
#[derive(Clone, Debug)]
pub struct LatentMediationFamilyEvaluation {
    pub log_likelihood: f64,
    pub log_continuous_density: f64,
    pub log_discrete_probability: f64,
    pub log_ascertainment_denominator: f64,
    pub ordinary_scale_representable: bool,
    pub integration_methods: Vec<String>,
    pub maximum_qmc_log_batch_range: f64,
    pub family_size: usize,
    pub mediator_proxy_truth_configurations: usize,
    pub ascertainment: &'static str,
}

/// The fixed-point record returned by the model.
#[derive(Clone, Debug)]
pub struct LatentMediationEvaluation {
    pub log_likelihood: f64,
    pub ordinary_scale_representable: bool,
    pub integration_methods: Vec<String>,
    pub maximum_qmc_log_batch_range: f64,
    pub families: Vec<LatentMediationFamilyEvaluation>,
}

/// A point the optimiser works in: the five structural coordinates, and then
/// one per covariate coefficient. The structural five keep their transforms
/// and their bounds; the coefficients are plain and unbounded, because a
/// covariate's effect has no sign or scale the model insists on.
type TransformedPoint = Vec<f64>;

/// A converged maximum-likelihood fit of the latent mediation model.
#[derive(Clone, Debug)]
pub struct LatentMediationFit {
    pub parameters: LatentMediationParameters,
    /// The estimated covariate coefficients: the mediator's terms first, then
    /// the outcome's, in the order the designs gave them. Empty where there
    /// are no covariates, which is the ordinary case.
    pub coefficients: Vec<f64>,
    pub log_likelihood: f64,
    pub scaled_gradient: f64,
    pub boundary_parameters: Vec<&'static str>,
    pub horizontal_identified: bool,
    pub integration: LatentMediationEvaluation,
}

#[derive(Clone, Debug)]
struct FitCandidate {
    transformed: TransformedPoint,
    objective: f64,
    scaled_gradient: f64,
    integration: LatentMediationEvaluation,
}

#[derive(Clone, Debug)]
struct InwardPenalty {
    anchor: TransformedPoint,
    base: f64,
    strength: f64,
}

impl InwardPenalty {
    fn new(anchor: TransformedPoint, base: f64) -> Self {
        Self {
            anchor,
            base,
            strength: 1.0e6 * base.abs().max(1.0),
        }
    }

    /// A finite penalty whose derivative always points away from the valid
    /// anchor, so the corresponding descent direction points back inward.
    fn value_and_gradient(&self, point: &[f64]) -> (f64, Vec<f64>) {
        let mut distance = 1.0;
        let mut gradient = Vec::with_capacity(point.len());
        for (value, anchor) in point.iter().zip(&self.anchor) {
            let delta = *value - *anchor;
            let radius = delta.hypot(1.0);
            distance += radius - 1.0;
            gradient.push(self.strength * delta / radius);
        }
        (self.base + self.strength * distance, gradient)
    }
}

/// A prepared collection of independent connected families.
#[derive(Clone, Debug)]
pub struct LatentMediationModel {
    families: Vec<LatentMediationFamily>,
    qmc_points: usize,
}

impl LatentMediationModel {
    /// Validate and fix all family data used by repeated evaluations.
    ///
    /// # Errors
    ///
    /// Returns a stable `LATENT_MEDIATION_*` code for every refused model.
    pub fn build(
        inputs: Vec<LatentMediationFamilyInput>,
        qmc_points: usize,
    ) -> Result<Self, &'static str> {
        if inputs.is_empty() {
            return Err("LATENT_MEDIATION_NO_FAMILIES");
        }
        if qmc_points < 256 || !qmc_points.is_multiple_of(8) {
            return Err("LATENT_MEDIATION_QMC_POINTS_INVALID");
        }
        let mut families = Vec::with_capacity(inputs.len());
        for input in inputs {
            let family = LatentMediationFamily::build(input)?;
            families.push(family);
        }
        Ok(Self {
            families,
            qmc_points,
        })
    }

    /// Evaluate the integrated observed-data likelihood at one parameter point.
    ///
    /// # Errors
    ///
    /// Returns a stable `LATENT_MEDIATION_*` code for an inadmissible point or a
    /// numerical failure.
    pub fn evaluate(
        &self,
        parameters: LatentMediationParameters,
    ) -> Result<LatentMediationEvaluation, &'static str> {
        // **Refused rather than defaulted where covariates exist.** Nought is a
        // real claim about a covariate's effect, not an absence of one, and a
        // likelihood evaluated at a claim nobody made is the kind of number
        // that gets quoted.
        if self.coefficient_count() != 0 {
            return Err("LATENT_MEDIATION_COEFFICIENTS_REQUIRED");
        }
        self.evaluate_at(parameters, &[])
    }

    /// The likelihood at both the structural parameters and the covariate
    /// coefficients: the mediator's terms first, then the outcome's, in the
    /// order the designs give them.
    ///
    /// # Errors
    ///
    /// Returns a stable `LATENT_MEDIATION_*` code, as [`Self::evaluate`] does,
    /// and one more where the number of coefficients is not the number of
    /// columns the designs offer.
    pub fn evaluate_at(
        &self,
        parameters: LatentMediationParameters,
        coefficients: &[f64],
    ) -> Result<LatentMediationEvaluation, &'static str> {
        validate_parameters(parameters)?;
        let mut total = 0.0;
        let mut ordinary_scale_representable = true;
        let mut methods = BTreeSet::new();
        let mut maximum_qmc_log_batch_range: f64 = 0.0;
        let mut family_records = Vec::with_capacity(self.families.len());
        for family in &self.families {
            let covariance = directional_covariance(&family.relationship, parameters)?;
            let record = family.evaluate(&covariance, self.qmc_points, coefficients)?;
            total += record.log_likelihood;
            ordinary_scale_representable &= record.ordinary_scale_representable;
            maximum_qmc_log_batch_range =
                maximum_qmc_log_batch_range.max(record.maximum_qmc_log_batch_range);
            methods.extend(record.integration_methods.iter().cloned());
            family_records.push(record);
        }
        if !total.is_finite() {
            return Err("LATENT_MEDIATION_LOG_LIKELIHOOD_NOT_FINITE");
        }
        let ordinary = total.exp();
        ordinary_scale_representable &= ordinary > 0.0 && ordinary.is_finite();
        Ok(LatentMediationEvaluation {
            log_likelihood: total,
            ordinary_scale_representable,
            integration_methods: methods.into_iter().collect(),
            maximum_qmc_log_batch_range,
            families: family_records,
        })
    }

    /// Fit all five structural parameters by maximum likelihood using the
    /// fixed, mediator-scale-invariant recipe.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start converges, or where the likelihood
    /// at the best of them cannot be evaluated.
    pub fn fit(&self) -> Result<LatentMediationFit, &'static str> {
        self.fit_holding(&[])
    }

    /// Fit with some coordinates held at nought, which is what the constrained
    /// models behind a test of `a b = 0` need.
    ///
    /// `held` names transformed coordinates: 0 is the mediator loading `a` and
    /// 1 is the mediator-to-outcome path `b`. A held coordinate is pinned by
    /// closing its bounds onto nought rather than by rewriting the objective,
    /// so the recipe, the starts and the convergence rule are the ones the
    /// unconstrained fit uses.
    ///
    /// # Errors
    ///
    /// Returns a stable `LATENT_MEDIATION_*` code, as [`Self::fit`] does.
    /// How many covariate coefficients the fit carries. Nought until a family
    /// supplies a design, at which point the optimiser's point grows by that
    /// many plain, unbounded coordinates.
    fn coefficient_count(&self) -> usize {
        self.families
            .first()
            .map_or(0, |family| family.design.ncols())
    }

    /// Fit with some structural coordinates held at nought.
    ///
    /// # Errors
    ///
    /// Returns a stable `LATENT_MEDIATION_*` code where the coordinate does not
    /// exist, no start converges, or the likelihood at the best of them cannot
    /// be evaluated.
    pub fn fit_holding(&self, held: &[usize]) -> Result<LatentMediationFit, &'static str> {
        if held.iter().any(|index| *index >= FIT_DIMENSION) {
            return Err("LATENT_MEDIATION_HELD_COORDINATE_INVALID");
        }
        let mediator_scale_squared = self.mediator_scale_squared()?;
        let coefficients = self.coefficient_count();
        let bounds = transformed_bounds_holding(held, coefficients)?;
        let mut best_converged: Option<FitCandidate> = None;
        let mut best_unresolved: Option<f64> = None;
        // The most recent underlying failure, so a fit in which nothing
        // converged reports its cause rather than only the fact.
        let mut last_error: Option<&'static str> = None;

        for mut start in deterministic_starts(coefficients) {
            for &index in held {
                start[index] = 0.0;
            }
            let Some(initial_objective) =
                self.optimisation_objective(&start, mediator_scale_squared)
            else {
                continue;
            };
            let penalty = InwardPenalty::new(start.clone(), initial_objective);
            let value_of = |candidate: &[f64]| {
                solver_value_and_gradient(
                    self,
                    candidate,
                    mediator_scale_squared,
                    &bounds,
                    &penalty,
                )
                .0
            };
            let gradient_of = |candidate: &[f64]| {
                solver_value_and_gradient(
                    self,
                    candidate,
                    mediator_scale_squared,
                    &bounds,
                    &penalty,
                )
                .1
            };
            let dimension = FIT_DIMENSION + coefficients;
            let mut control = OptimControl::default_for_dimension(dimension);
            control.maxit = 500;
            // R's own default: stop once the objective settles to about 1e-9
            // relative rather than running every start to `maxit`.
            control.factr = 1.0e3;
            control.pgtol = FIT_SOLVER_PGTOL;
            control.lmm = dimension.min(10);
            control.fnscale = initial_objective.abs().max(1.0);
            control.parscale = vec![1.0; dimension];

            let Ok(solution) = optim_lbfgsb_with_gradient(
                start.clone(),
                bounds.clone(),
                value_of,
                gradient_of,
                control,
            ) else {
                // A failed solver invocation is not turned into a stationary
                // pseudo-candidate.  The other deterministic starts remain
                // independent opportunities to resolve the fit.
                last_error = Some("LATENT_MEDIATION_SOLVER_FAILED");
                continue;
            };
            let transformed: TransformedPoint = solution.par;
            let integration = match self.evaluate_transformed(&transformed, mediator_scale_squared)
            {
                Ok(evaluation) => evaluation,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            let Some(objective) = self.optimisation_objective(&transformed, mediator_scale_squared)
            else {
                continue;
            };
            let ordinary_objective =
                |point: &[f64]| self.optimisation_objective(point, mediator_scale_squared);
            let gradient = match stable_bound_aware_gradient(
                &transformed,
                &bounds.lower,
                &bounds.upper,
                &ordinary_objective,
            ) {
                Ok(gradient) => gradient,
                Err(error) => {
                    last_error = Some(error);
                    keep_lower(&mut best_unresolved, objective);
                    continue;
                }
            };
            let scaled_gradient = scaled_projected_gradient(
                &transformed,
                &gradient,
                &bounds.lower,
                &bounds.upper,
                objective,
            );
            if !scaled_gradient.is_finite() || scaled_gradient >= FIT_GRADIENT_TOLERANCE {
                keep_lower(&mut best_unresolved, objective);
                continue;
            }
            let candidate = FitCandidate {
                transformed,
                objective,
                scaled_gradient,
                integration,
            };
            if best_converged
                .as_ref()
                .is_none_or(|seen| candidate.objective < seen.objective)
            {
                best_converged = Some(candidate);
            }
        }

        let best =
            best_converged.ok_or(last_error.unwrap_or("LATENT_MEDIATION_NO_START_CONVERGED"))?;
        if best_unresolved.is_some_and(|unresolved| {
            best.objective - unresolved > FIT_SELECTION_TOLERANCE * (1.0 + best.objective.abs())
        }) {
            return Err("LATENT_MEDIATION_BETTER_UNRESOLVED_OPTIMUM");
        }

        let parameters = transformed_parameters(&best.transformed, mediator_scale_squared)?;
        let a_at_boundary = at_lower_bound(best.transformed[0], bounds.lower[0]);
        let d_at_boundary = at_lower_bound(best.transformed[3], bounds.lower[3]);
        let mut boundary_parameters = Vec::new();
        if a_at_boundary {
            boundary_parameters.push("a");
        }
        if d_at_boundary {
            boundary_parameters.push("d");
        }
        Ok(LatentMediationFit {
            parameters,
            coefficients: best.transformed[FIT_DIMENSION..].to_vec(),
            log_likelihood: best.integration.log_likelihood,
            scaled_gradient: best.scaled_gradient,
            boundary_parameters,
            horizontal_identified: !a_at_boundary,
            integration: best.integration,
        })
    }

    fn mediator_scale_squared(&self) -> Result<f64, &'static str> {
        let mut total = 0.0;
        let mut observed = 0usize;
        for family in &self.families {
            for ((&index, &value), &error_variance) in family
                .observed_mediator_measurement_indices
                .iter()
                .zip(&family.observed_mediator_measurement_values)
                .zip(&family.observed_mediator_measurement_error_variances)
            {
                let residual = value - family.latent_mean[index];
                let contribution = residual.mul_add(residual, error_variance);
                if !contribution.is_finite() {
                    return Err("LATENT_MEDIATION_MEDIATOR_SCALE_UNIDENTIFIED");
                }
                total += contribution;
                observed += 1;
            }
        }
        if observed == 0 {
            return Err("LATENT_MEDIATION_MEDIATOR_SCALE_UNIDENTIFIED");
        }
        let scale_squared = total / observed as f64;
        if !scale_squared.is_finite() || scale_squared <= 0.0 {
            return Err("LATENT_MEDIATION_MEDIATOR_SCALE_UNIDENTIFIED");
        }
        Ok(scale_squared)
    }

    #[cfg(feature = "python")]
    fn subject_count(&self) -> usize {
        self.families
            .iter()
            .map(|family| family.relationship.nrows())
            .sum()
    }

    #[cfg(feature = "python")]
    fn family_count(&self) -> usize {
        self.families.len()
    }

    fn evaluate_transformed(
        &self,
        transformed: &[f64],
        mediator_scale_squared: f64,
    ) -> Result<LatentMediationEvaluation, &'static str> {
        self.evaluate_at(
            transformed_parameters(transformed, mediator_scale_squared)?,
            &transformed[FIT_DIMENSION..],
        )
    }

    fn transformed_objective(
        &self,
        transformed: &[f64],
        mediator_scale_squared: f64,
    ) -> Option<f64> {
        self.evaluate_transformed(transformed, mediator_scale_squared)
            .ok()
            .map(|evaluation| -evaluation.log_likelihood)
            .filter(|value| value.is_finite())
    }

    /// Negative log likelihood in a parameter-independent density coordinate.
    ///
    /// Rescaling every observed mediator quantity by `k` adds `m log(k)` to
    /// the raw negative log likelihood, where `m` is the mediator-measurement count.
    /// Subtracting the corresponding mediator-scale Jacobian keeps optimiser
    /// scaling and the independently recomputed convergence metric invariant,
    /// while `evaluate` and the fit record continue to report raw log
    /// likelihood in the caller's mediator units.
    fn optimisation_objective(
        &self,
        transformed: &[f64],
        mediator_scale_squared: f64,
    ) -> Option<f64> {
        let observed = self
            .families
            .iter()
            .map(|family| family.observed_mediator_measurement_indices.len())
            .sum::<usize>();
        let jacobian = 0.5 * observed as f64 * mediator_scale_squared.ln();
        self.transformed_objective(transformed, mediator_scale_squared)
            .map(|objective| objective - jacobian)
            .filter(|value| value.is_finite())
    }
}

fn transformed_bounds_holding(held: &[usize], coefficients: usize) -> Result<Bounds, &'static str> {
    let mut lower = vec![
        0.0,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        0.0,
        f64::NEG_INFINITY,
    ];
    lower.extend(std::iter::repeat_n(f64::NEG_INFINITY, coefficients));
    let mut upper = vec![f64::INFINITY; FIT_DIMENSION + coefficients];
    for &index in held {
        lower[index] = 0.0;
        upper[index] = 0.0;
    }
    Bounds::new(lower, upper).map_err(|_| "LATENT_MEDIATION_OPTIMISER_BOUNDS_INVALID")
}

fn deterministic_starts(coefficients: usize) -> Vec<TransformedPoint> {
    let structural = [
        [0.5_f64.sqrt(), 0.0, 0.0, 0.5, 0.5_f64.ln()],
        [0.2_f64.sqrt(), 0.5, 0.5, 0.5, 0.8_f64.ln()],
        [0.2_f64.sqrt(), -0.5, -0.5, 0.5, 0.8_f64.ln()],
        [0.8_f64.sqrt(), 0.5, -0.5, 0.2, 0.2_f64.ln()],
        [0.8_f64.sqrt(), -0.5, 0.5, 0.2, 0.2_f64.ln()],
    ];
    // Every start puts the covariate coefficients at nought, which is the only
    // value that says nothing about them. Spreading the starts over the
    // structural coordinates is what finds the several optima this likelihood
    // has; spreading them over the coefficients too would multiply the work
    // without adding a direction the optimiser cannot walk in.
    structural
        .into_iter()
        .map(|point| {
            let mut start = point.to_vec();
            start.extend(std::iter::repeat_n(0.0, coefficients));
            start
        })
        .collect()
}

fn transformed_parameters(
    transformed: &[f64],
    mediator_scale_squared: f64,
) -> Result<LatentMediationParameters, &'static str> {
    // The structural five, and however many coefficients follow them. Only the
    // five are transformed; the rest are read off where they lie.
    if transformed.len() < FIT_DIMENSION
        || transformed.iter().any(|value| !value.is_finite())
        || !mediator_scale_squared.is_finite()
        || mediator_scale_squared <= 0.0
        || transformed[0] < 0.0
        || transformed[3] < 0.0
    {
        return Err("LATENT_MEDIATION_TRANSFORMED_PARAMETER_INVALID");
    }
    let mediator_scale = mediator_scale_squared.sqrt();
    let parameters = LatentMediationParameters {
        a: mediator_scale * transformed[0],
        b: transformed[1] / mediator_scale,
        c_prime: transformed[2],
        d: transformed[3].sqrt(),
        sigma_m2: mediator_scale_squared * transformed[4].exp(),
    };
    validate_parameters(parameters)?;
    Ok(parameters)
}

fn solver_value_and_gradient(
    model: &LatentMediationModel,
    point: &[f64],
    mediator_scale_squared: f64,
    bounds: &Bounds,
    penalty: &InwardPenalty,
) -> (f64, Vec<f64>) {
    let objective =
        |candidate: &[f64]| model.optimisation_objective(candidate, mediator_scale_squared);
    let Ok(gradient) = bound_aware_gradient(
        point,
        &bounds.lower,
        &bounds.upper,
        FIT_GRADIENT_STEP,
        &objective,
    ) else {
        return penalty.value_and_gradient(point);
    };
    let Some(value) = objective(point) else {
        return penalty.value_and_gradient(point);
    };
    (value, gradient)
}

/// Bounds-aware second-order numerical gradient.  Every coordinate gets up to
/// eight step halvings before its stencil is declared unresolved.
fn bound_aware_gradient<F>(
    point: &[f64],
    lower: &[f64],
    upper: &[f64],
    step_multiplier: f64,
    objective: &F,
) -> Result<Vec<f64>, &'static str>
where
    F: Fn(&[f64]) -> Option<f64>,
{
    // The point is the structural five and however many coefficients follow;
    // the stencil walks whatever it is given rather than a fixed five.
    if point.len() < FIT_DIMENSION
        || lower.len() != point.len()
        || upper.len() != point.len()
        || !step_multiplier.is_finite()
        || step_multiplier <= 0.0
        || point.iter().any(|value| !value.is_finite())
        || (0..point.len()).any(|index| {
            point[index] < lower[index]
                || point[index] > upper[index]
                || lower[index].is_nan()
                || upper[index].is_nan()
        })
    {
        return Err("LATENT_MEDIATION_GRADIENT_STENCIL_UNRESOLVED");
    }
    let Some(base) = objective(point).filter(|value| value.is_finite()) else {
        return Err("LATENT_MEDIATION_GRADIENT_STENCIL_UNRESOLVED");
    };
    let mut gradient = Vec::with_capacity(point.len());
    for index in 0..point.len() {
        // A coordinate whose bounds have closed onto each other is held, not
        // free. It has no derivative to find -- every step off it is outside
        // the feasible set -- and asking for one would fail the stencil.
        if lower[index] == upper[index] {
            gradient.push(0.0);
            continue;
        }
        let mut step = step_multiplier * point[index].abs().max(1.0);
        let mut resolved = None;
        for _ in 0..=FIT_MAXIMUM_STENCIL_HALVINGS {
            if !step.is_finite() || step <= 0.0 {
                break;
            }
            let minus = point[index] - step;
            let plus = point[index] + step;
            let backward_feasible = minus.is_finite() && minus >= lower[index];
            let forward_feasible = plus.is_finite() && plus <= upper[index];
            let derivative = if backward_feasible && forward_feasible {
                let mut backward = point.to_vec();
                let mut forward = point.to_vec();
                backward[index] = minus;
                forward[index] = plus;
                match (objective(&backward), objective(&forward)) {
                    (Some(left), Some(right)) if left.is_finite() && right.is_finite() => {
                        Some((right - left) / (2.0 * step))
                    }
                    _ => None,
                }
            } else if forward_feasible {
                let plus_two = point[index] + 2.0 * step;
                if plus_two.is_finite() && plus_two <= upper[index] {
                    let mut forward = point.to_vec();
                    let mut forward_two = point.to_vec();
                    forward[index] = plus;
                    forward_two[index] = plus_two;
                    match (objective(&forward), objective(&forward_two)) {
                        (Some(first), Some(second)) if first.is_finite() && second.is_finite() => {
                            Some((-3.0 * base + 4.0 * first - second) / (2.0 * step))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            } else if backward_feasible {
                let minus_two = point[index] - 2.0 * step;
                if minus_two.is_finite() && minus_two >= lower[index] {
                    let mut backward = point.to_vec();
                    let mut backward_two = point.to_vec();
                    backward[index] = minus;
                    backward_two[index] = minus_two;
                    match (objective(&backward), objective(&backward_two)) {
                        (Some(first), Some(second)) if first.is_finite() && second.is_finite() => {
                            Some((3.0 * base - 4.0 * first + second) / (2.0 * step))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(value) = derivative.filter(|value| value.is_finite()) {
                resolved = Some(value);
                break;
            }
            step *= 0.5;
        }
        gradient.push(resolved.ok_or("LATENT_MEDIATION_GRADIENT_STENCIL_UNRESOLVED")?);
    }
    Ok(gradient)
}

fn stable_bound_aware_gradient<F>(
    point: &[f64],
    lower: &[f64],
    upper: &[f64],
    objective: &F,
) -> Result<Vec<f64>, &'static str>
where
    F: Fn(&[f64]) -> Option<f64>,
{
    let coarse = bound_aware_gradient(point, lower, upper, FIT_GRADIENT_STEP, objective)?;
    let fine = bound_aware_gradient(point, lower, upper, FIT_GRADIENT_STEP * 0.5, objective)?;
    if coarse.iter().zip(&fine).any(|(first, second)| {
        (first - second).abs() > FIT_GRADIENT_STABILITY_TOLERANCE * second.abs().max(1.0)
    }) {
        return Err("LATENT_MEDIATION_GRADIENT_UNSTABLE");
    }
    Ok(fine)
}

fn scaled_projected_gradient(
    point: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    objective: f64,
) -> f64 {
    if point.len() != gradient.len()
        || point.len() != lower.len()
        || point.len() != upper.len()
        || !objective.is_finite()
    {
        return f64::INFINITY;
    }
    let objective_scale = objective.abs().max(1.0);
    let mut worst = 0.0_f64;
    for index in 0..point.len() {
        let component = gradient[index];
        if !component.is_finite() {
            return f64::INFINITY;
        }
        let held = lower[index] == upper[index];
        let projected = if held
            || (at_lower_bound(point[index], lower[index]) && component > 0.0)
            || (at_upper_bound(point[index], upper[index]) && component < 0.0)
        {
            0.0
        } else {
            component
        };
        let scaled = projected.abs() * point[index].abs().max(1.0) / objective_scale;
        worst = worst.max(scaled);
    }
    worst
}

fn at_lower_bound(value: f64, lower: f64) -> bool {
    lower.is_finite()
        && value <= lower + FIT_BOUND_TOLERANCE * value.abs().max(lower.abs()).max(1.0)
}

fn at_upper_bound(value: f64, upper: f64) -> bool {
    upper.is_finite()
        && value >= upper - FIT_BOUND_TOLERANCE * value.abs().max(upper.abs()).max(1.0)
}

fn keep_lower(best: &mut Option<f64>, candidate: f64) {
    if candidate.is_finite() && best.is_none_or(|seen| candidate < seen) {
        *best = Some(candidate);
    }
}

impl LatentMediationFamily {
    fn build(input: LatentMediationFamilyInput) -> Result<Self, &'static str> {
        let size = input.relationship.len();
        if size == 0 {
            return Err("LATENT_MEDIATION_FAMILY_EMPTY");
        }
        if input.relationship.iter().any(|row| row.len() != size) {
            return Err("LATENT_MEDIATION_RELATIONSHIP_WRONG_SHAPE");
        }
        let relationship =
            DMatrix::from_fn(size, size, |row, column| input.relationship[row][column]);
        if relationship.iter().any(|value| !value.is_finite()) {
            return Err("LATENT_MEDIATION_RELATIONSHIP_NOT_FINITE");
        }
        for row in 0..size {
            if (relationship[(row, row)] - 1.0).abs() > TOLERANCE {
                return Err("LATENT_MEDIATION_RELATIONSHIP_DIAGONAL_NOT_ONE");
            }
            for column in 0..row {
                if relationship[(row, column)] != relationship[(column, row)] {
                    return Err("LATENT_MEDIATION_RELATIONSHIP_NOT_SYMMETRIC");
                }
            }
        }
        let relationship_decomposition = SymmetricEigen::new(relationship.clone());
        if relationship_decomposition.eigenvalues.min() < RELATIONSHIP_PSD_FLOOR {
            return Err("LATENT_MEDIATION_RELATIONSHIP_NOT_POSITIVE_SEMIDEFINITE");
        }
        if size > 1 && !relationship_is_connected(&relationship) {
            return Err("LATENT_MEDIATION_FAMILY_NOT_CONNECTED");
        }

        if input.latent_mean.len() != 2 * size {
            return Err("LATENT_MEDIATION_LATENT_MEAN_WRONG_LENGTH");
        }
        if input.latent_mean.iter().any(|value| !value.is_finite()) {
            return Err("LATENT_MEDIATION_LATENT_MEAN_NOT_FINITE");
        }
        for values in [
            &input.mediator_measurement,
            &input.mediator_measurement_error_variance,
        ] {
            if values.len() != size {
                return Err("LATENT_MEDIATION_CONTINUOUS_OBSERVATION_WRONG_LENGTH");
            }
        }
        for values in [&input.mediator_proxy_status, &input.outcome_status] {
            if values.len() != size {
                return Err("LATENT_MEDIATION_STATUS_WRONG_LENGTH");
            }
            if values
                .iter()
                .flatten()
                .any(|status| !matches!(status, 0 | 1))
            {
                return Err("LATENT_MEDIATION_STATUS_NOT_BINARY");
            }
        }
        for index in 0..size {
            match (
                input.mediator_measurement[index],
                input.mediator_measurement_error_variance[index],
            ) {
                (None, None) => {}
                (None, Some(_)) => {
                    return Err("LATENT_MEDIATION_MISSING_MEDIATOR_MEASUREMENT_HAS_ERROR");
                }
                (Some(_), None) => {
                    return Err("LATENT_MEDIATION_MEDIATOR_MEASUREMENT_ERROR_MISSING");
                }
                (Some(value), Some(error)) => {
                    if !value.is_finite() {
                        return Err("LATENT_MEDIATION_MEDIATOR_MEASUREMENT_NOT_FINITE");
                    }
                    if !error.is_finite() || error <= 0.0 {
                        return Err("LATENT_MEDIATION_MEDIATOR_MEASUREMENT_ERROR_NOT_POSITIVE");
                    }
                }
            }
        }
        for values in [
            &input.mediator_threshold,
            &input.outcome_threshold,
            &input.mediator_proxy_sensitivity,
            &input.mediator_proxy_specificity,
        ] {
            if values.len() != size {
                return Err("LATENT_MEDIATION_PERSON_VECTOR_WRONG_LENGTH");
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err("LATENT_MEDIATION_PERSON_VECTOR_NOT_FINITE");
            }
        }
        if input
            .mediator_proxy_sensitivity
            .iter()
            .chain(&input.mediator_proxy_specificity)
            .any(|value| *value <= 0.0 || *value >= 1.0)
        {
            return Err("LATENT_MEDIATION_MEDIATOR_PROXY_ACCURACY_OUTSIDE_OPEN_UNIT");
        }
        let observed_mediator_proxy = input.mediator_proxy_status.iter().flatten().count();
        let observed_outcome = input.outcome_status.iter().flatten().count();
        if observed_mediator_proxy > MAXIMUM_OBSERVED_MEDIATOR_PROXY {
            return Err("LATENT_MEDIATION_MEDIATOR_PROXY_DIMENSION_EXCEEDED");
        }
        if observed_mediator_proxy + observed_outcome > MAXIMUM_DISCRETE_DIMENSION {
            return Err("LATENT_MEDIATION_DISCRETE_DIMENSION_EXCEEDED");
        }

        let ascertainment = match input.ascertainment.as_str() {
            "population_unconditioned" => {
                if input.proband_index.is_some() {
                    return Err("LATENT_MEDIATION_UNCONDITIONED_PROBAND_GIVEN");
                }
                Ascertainment::PopulationUnconditioned
            }
            "condition_on_named_proband_case" => {
                let proband = input
                    .proband_index
                    .ok_or("LATENT_MEDIATION_NAMED_PROBAND_MISSING")?;
                if proband >= size {
                    return Err("LATENT_MEDIATION_PROBAND_OUTSIDE_FAMILY");
                }
                if input.outcome_status[proband] != Some(1) {
                    return Err("LATENT_MEDIATION_NAMED_PROBAND_NOT_OUTCOME_CASE");
                }
                Ascertainment::NamedProbandCase(proband)
            }
            _ => return Err("LATENT_MEDIATION_ASCERTAINMENT_UNSUPPORTED"),
        };

        let observed_mediator_measurement_indices: Vec<usize> = input
            .mediator_measurement
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.map(|_| index))
            .collect();
        let observed_mediator_measurement_values = observed_mediator_measurement_indices
            .iter()
            .map(|&index| input.mediator_measurement[index].expect("selected as observed"))
            .collect();
        let observed_mediator_measurement_error_variances = observed_mediator_measurement_indices
            .iter()
            .map(|&index| {
                input.mediator_measurement_error_variance[index]
                    .expect("validated with observation")
            })
            .collect();
        let observed_mediator_proxy: Vec<(usize, i8)> = input
            .mediator_proxy_status
            .iter()
            .enumerate()
            .filter_map(|(index, status)| status.map(|value| (index, value)))
            .collect();
        let observed_outcome: Vec<(usize, i8)> = input
            .outcome_status
            .iter()
            .enumerate()
            .filter_map(|(index, status)| status.map(|value| (index, value)))
            .collect();
        let discrete_target_indices = observed_mediator_proxy
            .iter()
            .map(|(index, _)| *index)
            .chain(observed_outcome.iter().map(|(index, _)| size + *index))
            .collect();

        let mediator_terms = design_width(&input.mediator_design, size)?;
        let outcome_terms = design_width(&input.outcome_design, size)?;
        let mut design = DMatrix::zeros(2 * size, mediator_terms + outcome_terms);
        for person in 0..size {
            for term in 0..mediator_terms {
                design[(person, term)] = input.mediator_design[person][term];
            }
            for term in 0..outcome_terms {
                design[(size + person, mediator_terms + term)] = input.outcome_design[person][term];
            }
        }

        Ok(Self {
            relationship,
            latent_mean: DVector::from_vec(input.latent_mean),
            design,
            observed_mediator_measurement_indices,
            observed_mediator_measurement_values,
            observed_mediator_measurement_error_variances,
            observed_mediator_proxy,
            observed_outcome,
            discrete_target_indices,
            mediator_threshold: input.mediator_threshold,
            outcome_threshold: input.outcome_threshold,
            mediator_proxy_sensitivity: input.mediator_proxy_sensitivity,
            mediator_proxy_specificity: input.mediator_proxy_specificity,
            ascertainment,
        })
    }

    /// The latent mean a set of coefficients implies: what was supplied,
    /// plus the covariates' contribution.
    fn mean_under(&self, coefficients: &[f64]) -> Result<DVector<f64>, &'static str> {
        if coefficients.len() != self.design.ncols() {
            return Err("LATENT_MEDIATION_COEFFICIENT_COUNT_WRONG");
        }
        if coefficients.is_empty() {
            return Ok(self.latent_mean.clone());
        }
        Ok(&self.latent_mean + &self.design * DVector::from_column_slice(coefficients))
    }

    fn evaluate(
        &self,
        covariance: &DMatrix<f64>,
        qmc_points: usize,
        coefficients: &[f64],
    ) -> Result<LatentMediationFamilyEvaluation, &'static str> {
        let latent_mean = self.mean_under(coefficients)?;
        let size = self.relationship.nrows();
        let conditional = condition_on_mediator_measurements(
            &self.discrete_target_indices,
            &self.observed_mediator_measurement_indices,
            &self.observed_mediator_measurement_values,
            &self.observed_mediator_measurement_error_variances,
            &latent_mean,
            covariance,
        )?;

        let truth_configurations = 1usize << self.observed_mediator_proxy.len();
        // **Summed on the log scale.** The configurations are added together,
        // and a family deep in the tail has every one of them below the
        // smallest double, so adding them as ordinary numbers loses the whole
        // sum to underflow however carefully each was computed.
        let mut log_terms: Vec<f64> = Vec::with_capacity(truth_configurations);
        let mut maximum_qmc_log_batch_range: f64 = 0.0;
        let mut methods = BTreeSet::new();
        for configuration in 0..truth_configurations {
            // **Summed as logs, not multiplied.** Each factor is a
            // sensitivity or a specificity, so a family with several fallible
            // proxies multiplies several numbers below one together; at
            // extreme accuracies that product underflows to nought and takes
            // the whole configuration with it, including the dominant one.
            // Every factor is strictly inside the unit interval, so its
            // logarithm is finite and the sum is exact.
            let mut log_measurement_weight = 0.0_f64;
            let mut lower = Vec::with_capacity(self.discrete_target_indices.len());
            let mut upper = Vec::with_capacity(self.discrete_target_indices.len());
            for (position, &(index, observed)) in self.observed_mediator_proxy.iter().enumerate() {
                let truth = i8::from((configuration & (1usize << position)) != 0);
                log_measurement_weight += match (truth, observed) {
                    (1, 1) => self.mediator_proxy_sensitivity[index],
                    (1, 0) => 1.0 - self.mediator_proxy_sensitivity[index],
                    (0, 1) => 1.0 - self.mediator_proxy_specificity[index],
                    (0, 0) => self.mediator_proxy_specificity[index],
                    _ => unreachable!("validated binary status"),
                }
                .ln();
                let bounds = status_bounds(truth, self.mediator_threshold[index]);
                lower.push(bounds.0);
                upper.push(bounds.1);
            }
            for &(index, status) in &self.observed_outcome {
                let bounds = status_bounds(status, self.outcome_threshold[index]);
                lower.push(bounds.0);
                upper.push(bounds.1);
            }

            let rectangle = if lower.is_empty() {
                RectangleProbability {
                    log_probability: 0.0,
                    log_batch_range: 0.0,
                    method: "no_discrete_observation",
                }
            } else {
                rectangle_probability(
                    &lower,
                    &upper,
                    &conditional.mean,
                    &conditional.covariance,
                    qmc_points,
                )?
            };
            log_terms.push(log_measurement_weight + rectangle.log_probability);
            maximum_qmc_log_batch_range =
                maximum_qmc_log_batch_range.max(rectangle.log_batch_range);
            methods.insert(rectangle.method.to_owned());
        }
        let log_discrete_probability = log_sum_exp(&log_terms);
        if !log_discrete_probability.is_finite() {
            return Err("LATENT_MEDIATION_DISCRETE_PROBABILITY_INVALID");
        }
        let log_numerator = conditional.log_continuous_density + log_discrete_probability;

        let (denominator, ascertainment_name) = match self.ascertainment {
            Ascertainment::PopulationUnconditioned => (1.0, "population_unconditioned"),
            Ascertainment::NamedProbandCase(proband) => {
                let latent_index = size + proband;
                let variance = covariance[(latent_index, latent_index)];
                if !(variance > 0.0 && variance.is_finite()) {
                    return Err("LATENT_MEDIATION_ASCERTAINMENT_VARIANCE_INVALID");
                }
                let standardised =
                    (self.outcome_threshold[proband] - latent_mean[latent_index]) / variance.sqrt();
                (normal_sf(standardised)?, "condition_on_named_proband_case")
            }
        };
        if !(denominator > 0.0 && denominator.is_finite()) {
            return Err("LATENT_MEDIATION_ASCERTAINMENT_PROBABILITY_INVALID");
        }
        let log_denominator = denominator.ln();
        let log_likelihood = log_numerator - log_denominator;
        if !log_likelihood.is_finite() {
            return Err("LATENT_MEDIATION_FAMILY_LOG_LIKELIHOOD_NOT_FINITE");
        }
        let continuous = conditional.log_continuous_density.exp();
        let numerator = log_numerator.exp();
        let likelihood = log_likelihood.exp();
        // A subnormal is greater than nought and finite while having lost
        // most of its significand, so testing for those two alone called a
        // number carrying two digits representable. The smallest normal is the
        // real boundary.
        let ordinary_scale_representable = [continuous, numerator, likelihood]
            .iter()
            .all(|value| *value >= f64::MIN_POSITIVE && value.is_finite());

        Ok(LatentMediationFamilyEvaluation {
            log_likelihood,
            log_continuous_density: conditional.log_continuous_density,
            log_discrete_probability,
            log_ascertainment_denominator: log_denominator,
            ordinary_scale_representable,
            integration_methods: methods.into_iter().collect(),
            maximum_qmc_log_batch_range,
            family_size: size,
            mediator_proxy_truth_configurations: truth_configurations,
            ascertainment: ascertainment_name,
        })
    }
}

fn relationship_is_connected(relationship: &DMatrix<f64>) -> bool {
    let size = relationship.nrows();
    let mut visited = vec![false; size];
    let mut stack = vec![0usize];
    visited[0] = true;
    while let Some(row) = stack.pop() {
        for column in 0..size {
            if row != column && relationship[(row, column)].abs() > TOLERANCE && !visited[column] {
                visited[column] = true;
                stack.push(column);
            }
        }
    }
    visited.into_iter().all(|value| value)
}

fn validate_parameters(parameters: LatentMediationParameters) -> Result<(), &'static str> {
    if [
        parameters.a,
        parameters.b,
        parameters.c_prime,
        parameters.d,
        parameters.sigma_m2,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err("LATENT_MEDIATION_PARAMETER_NOT_FINITE");
    }
    if parameters.a < 0.0 || parameters.d < 0.0 {
        return Err("LATENT_MEDIATION_LOADING_NEGATIVE");
    }
    if parameters.sigma_m2 <= 0.0 {
        return Err("LATENT_MEDIATION_SIGMA_M2_NOT_POSITIVE");
    }
    Ok(())
}

/// How many columns a design offers, refusing a ragged one.
///
/// An empty design is nought terms, which is the ordinary case: most families
/// carry no covariates and the model is what it was.
fn design_width(design: &[Vec<f64>], size: usize) -> Result<usize, &'static str> {
    if design.is_empty() {
        return Ok(0);
    }
    if design.len() != size {
        return Err("LATENT_MEDIATION_DESIGN_ROWS_WRONG");
    }
    let terms = design[0].len();
    if terms == 0 || design.iter().any(|row| row.len() != terms) {
        return Err("LATENT_MEDIATION_DESIGN_RAGGED");
    }
    if design
        .iter()
        .any(|row| row.iter().any(|value| !value.is_finite()))
    {
        return Err("LATENT_MEDIATION_DESIGN_NOT_FINITE");
    }
    Ok(terms)
}

fn directional_covariance(
    relationship: &DMatrix<f64>,
    parameters: LatentMediationParameters,
) -> Result<DMatrix<f64>, &'static str> {
    let size = relationship.nrows();
    let total = parameters.a.mul_add(parameters.b, parameters.c_prime);
    let genetic = [
        [parameters.a * parameters.a, parameters.a * total],
        [
            parameters.a * total,
            total.mul_add(total, parameters.d * parameters.d),
        ],
    ];
    // The inherited-process covariance is zero and outcome innovation variance is one.
    let residual = [
        [parameters.sigma_m2, parameters.b * parameters.sigma_m2],
        [
            parameters.b * parameters.sigma_m2,
            parameters
                .b
                .mul_add(parameters.b * parameters.sigma_m2, 1.0),
        ],
    ];
    let covariance = DMatrix::from_fn(2 * size, 2 * size, |row, column| {
        let left_process = row / size;
        let right_process = column / size;
        let left = row % size;
        let right = column % size;
        relationship[(left, right)] * genetic[left_process][right_process]
            + if left == right {
                residual[left_process][right_process]
            } else {
                0.0
            }
    });
    if covariance.clone().cholesky().is_none() {
        return Err("LATENT_MEDIATION_COVARIANCE_NOT_POSITIVE_DEFINITE");
    }
    Ok(covariance)
}

struct ConditionalDistribution {
    mean: DVector<f64>,
    covariance: DMatrix<f64>,
    log_continuous_density: f64,
}

fn condition_on_mediator_measurements(
    targets: &[usize],
    observed_indices: &[usize],
    observed_values: &[f64],
    error_variances: &[f64],
    latent_mean: &DVector<f64>,
    latent_covariance: &DMatrix<f64>,
) -> Result<ConditionalDistribution, &'static str> {
    let target_mean =
        DVector::from_iterator(targets.len(), targets.iter().map(|&i| latent_mean[i]));
    let target_covariance = DMatrix::from_fn(targets.len(), targets.len(), |row, column| {
        latent_covariance[(targets[row], targets[column])]
    });
    if observed_indices.is_empty() {
        return Ok(ConditionalDistribution {
            mean: target_mean,
            covariance: target_covariance,
            log_continuous_density: 0.0,
        });
    }
    let count = observed_indices.len();
    let observed_mean = DVector::from_iterator(
        count,
        observed_indices.iter().map(|&index| latent_mean[index]),
    );
    let observed_covariance = DMatrix::from_fn(count, count, |row, column| {
        latent_covariance[(observed_indices[row], observed_indices[column])]
            + if row == column {
                error_variances[row]
            } else {
                0.0
            }
    });
    let decomposition = observed_covariance
        .clone()
        .cholesky()
        .ok_or("LATENT_MEDIATION_OBSERVED_COVARIANCE_NOT_POSITIVE_DEFINITE")?;
    let residual = DVector::from_vec(observed_values.to_vec()) - observed_mean;
    let solved_residual = decomposition.solve(&residual);
    let quadratic = residual.dot(&solved_residual);
    let log_determinant = 2.0
        * decomposition
            .l()
            .diagonal()
            .iter()
            .map(|value| value.ln())
            .sum::<f64>();
    let log_continuous_density = -0.5 * (count as f64 * LOG_TWO_PI + log_determinant + quadratic);
    if !log_continuous_density.is_finite() {
        return Err("LATENT_MEDIATION_CONTINUOUS_LOG_DENSITY_NOT_FINITE");
    }
    if targets.is_empty() {
        return Ok(ConditionalDistribution {
            mean: target_mean,
            covariance: target_covariance,
            log_continuous_density,
        });
    }
    let cross = DMatrix::from_fn(targets.len(), count, |row, column| {
        latent_covariance[(targets[row], observed_indices[column])]
    });
    let solved_cross_transpose = decomposition.solve(&cross.transpose());
    let conditional_mean = target_mean + &cross * solved_residual;
    let conditional_covariance = target_covariance - &cross * solved_cross_transpose;
    if conditional_covariance.clone().cholesky().is_none() {
        return Err("LATENT_MEDIATION_CONDITIONAL_COVARIANCE_NOT_POSITIVE_DEFINITE");
    }
    Ok(ConditionalDistribution {
        mean: conditional_mean,
        covariance: conditional_covariance,
        log_continuous_density,
    })
}

/// `ln(sum_i exp(x_i))`, taken about the largest term so that a sum every one
/// of whose terms is below the smallest double still has a log.
fn log_sum_exp(terms: &[f64]) -> f64 {
    let largest = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !largest.is_finite() {
        return f64::NEG_INFINITY;
    }
    largest
        + terms
            .iter()
            .map(|value| (value - largest).exp())
            .sum::<f64>()
            .ln()
}

fn status_bounds(status: i8, threshold: f64) -> (f64, f64) {
    if status == 0 {
        (f64::NEG_INFINITY, threshold)
    } else {
        (threshold, f64::INFINITY)
    }
}

struct RectangleProbability {
    /// **The log of the probability, not the probability.** A family deep in
    /// the tail has a rectangle far below the smallest double, and the whole
    /// point of computing it on the log scale is lost the moment it is
    /// exponentiated: doing that capped the model at a log likelihood of
    /// -744.44, which is `ln(5e-324)` and a property of binary64 rather than of
    /// the integral.
    log_probability: f64,
    log_batch_range: f64,
    method: &'static str,
}

fn rectangle_probability(
    lower: &[f64],
    upper: &[f64],
    mean: &DVector<f64>,
    covariance: &DMatrix<f64>,
    qmc_points: usize,
) -> Result<RectangleProbability, &'static str> {
    let dimension = lower.len();
    if dimension == 0
        || upper.len() != dimension
        || mean.len() != dimension
        || covariance.nrows() != dimension
        || covariance.ncols() != dimension
    {
        return Err("LATENT_MEDIATION_RECTANGLE_SHAPE_INVALID");
    }
    if dimension > MAXIMUM_DISCRETE_DIMENSION {
        return Err("LATENT_MEDIATION_DISCRETE_DIMENSION_EXCEEDED");
    }
    if lower
        .iter()
        .zip(upper)
        .any(|(&left, &right)| left >= right || left.is_nan() || right.is_nan())
    {
        return Ok(RectangleProbability {
            log_probability: f64::NEG_INFINITY,
            log_batch_range: 0.0,
            method: "empty_rectangle",
        });
    }
    let decomposition = covariance
        .clone()
        .cholesky()
        .ok_or("LATENT_MEDIATION_RECTANGLE_COVARIANCE_NOT_POSITIVE_DEFINITE")?;
    if dimension == 1 {
        let sd = covariance[(0, 0)].sqrt();
        let log_probability =
            log_interval_probability((lower[0] - mean[0]) / sd, (upper[0] - mean[0]) / sd)?;
        return Ok(RectangleProbability {
            log_probability,
            log_batch_range: 0.0,
            method: "univariate_exact",
        });
    }
    if dimension == 2 {
        let rectangle = bivariate_rectangle(lower, upper, mean, covariance)?;
        return Ok(RectangleProbability {
            log_probability: rectangle.log_probability,
            log_batch_range: 0.0,
            method: rectangle.method,
        });
    }

    let factor = decomposition.l();
    // Eight deterministically shifted copies of the Halton sequence.  A single
    // low-discrepancy sequence carries one systematic bias, and contiguous
    // blocks of it share that bias, so their spread understates the error.
    // Shifting decorrelates the copies: their mean sheds most of the shared
    // bias and their spread is an honest stability statistic.
    // **The separation-of-variables weight is a product of one interval
    // probability per member, so it underflows long before any single one
    // does.** Accumulated on the ordinary scale, a family of four ordinary
    // members each contributing 1e-80 came back as nought, and the model that
    // works in logarithms everywhere else had a floor at 1e-308 hidden in its
    // third dimension and above. The weights are summed on the log scale
    // instead, which puts the quasi-Monte Carlo path on the same footing as
    // the one- and two-member cases.
    let mut replicate_estimates = [f64::NEG_INFINITY; QMC_REPLICATES];
    let per_replicate = qmc_points / QMC_REPLICATES;
    for (replicate, estimate) in replicate_estimates.iter_mut().enumerate() {
        let mut largest = f64::NEG_INFINITY;
        let mut rescaled_sum = 0.0;
        for sample_index in 0..per_replicate {
            let sequence_index = sample_index / 2 + 1;
            let antithetic = sample_index % 2 == 1;
            let mut latent = vec![0.0; dimension];
            let mut log_weight = 0.0;
            for row in 0..dimension {
                let preceding = (0..row)
                    .map(|column| factor[(row, column)] * latent[column])
                    .sum::<f64>();
                let standardised_lower = (lower[row] - mean[row] - preceding) / factor[(row, row)];
                let standardised_upper = (upper[row] - mean[row] - preceding) / factor[(row, row)];
                log_weight += log_interval_probability(standardised_lower, standardised_upper)?;
                if log_weight == f64::NEG_INFINITY {
                    break;
                }
                if row < dimension - 1 {
                    let mut coordinate =
                        (halton(sequence_index, PRIMES[row]) + qmc_shift(replicate, row)).fract();
                    if antithetic {
                        coordinate = 1.0 - coordinate;
                    }
                    let coordinate = coordinate.clamp(1.0e-15, 1.0 - 1.0e-15);
                    latent[row] = truncated_standard_normal(
                        standardised_lower,
                        standardised_upper,
                        coordinate,
                    )?;
                }
            }
            if log_weight == f64::NEG_INFINITY {
                continue;
            }
            if log_weight > largest {
                rescaled_sum = rescaled_sum * (largest - log_weight).exp() + 1.0;
                largest = log_weight;
            } else {
                rescaled_sum += (log_weight - largest).exp();
            }
        }
        *estimate = if largest == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            largest + rescaled_sum.ln() - (per_replicate as f64).ln()
        };
    }
    let log_estimate = log_sum_exp(&replicate_estimates) - (QMC_REPLICATES as f64).ln();
    let highest = replicate_estimates
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let lowest = replicate_estimates
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    Ok(RectangleProbability {
        log_probability: log_estimate.min(0.0),
        // A difference of logarithms, so the spread is read relative to the
        // estimate itself. Reported as an absolute ordinary-scale width it
        // said nothing at all beside a log probability: the same shakiness
        // showed as 1e-3 at one depth and 1e-200 at another.
        log_batch_range: if highest.is_finite() && lowest.is_finite() {
            highest - lowest
        } else if highest == lowest {
            0.0
        } else {
            f64::INFINITY
        },
        method: "deterministic_genz_halton",
    })
}

/// A fixed Cranley-Patterson shift for one replicate and coordinate: the
/// Kronecker sequence on the fractional part of a later prime's square root,
/// deterministic and well spread across replicates.
fn qmc_shift(replicate: usize, row: usize) -> f64 {
    ((replicate as f64 + 1.0) * (SHIFT_PRIMES[row] as f64).sqrt().fract()).fract()
}

fn interval_probability(lower: f64, upper: f64) -> Result<f64, &'static str> {
    if lower >= upper {
        return Ok(0.0);
    }
    if lower == f64::NEG_INFINITY {
        return if upper == f64::INFINITY {
            Ok(1.0)
        } else {
            normal_cdf(upper)
        };
    }
    if upper == f64::INFINITY {
        return normal_sf(lower);
    }
    if lower >= 0.0 {
        return Ok((normal_sf(lower)? - normal_sf(upper)?).max(0.0));
    }
    Ok((normal_cdf(upper)? - normal_cdf(lower)?).max(0.0))
}

fn normal() -> Result<Normal, &'static str> {
    Normal::new(0.0, 1.0).map_err(|_| "LATENT_MEDIATION_NORMAL_UNAVAILABLE")
}

fn normal_cdf(value: f64) -> Result<f64, &'static str> {
    normal_sf(-value)
}

/// Upper tail of the standard normal, as `erfc(x / sqrt 2) / 2`.
///
/// **Not `statrs`'s `Normal::cdf`, which is not accurate enough to integrate
/// against.** That routes through an `erfc` approximation carrying about 5e-11
/// of relative error -- it puts the tail at 3 deviations at 1.349898031574e-3
/// where the true value is 1.349898031630e-3 -- and the error wanders from
/// point to point rather than varying smoothly. To a quadrature asking for
/// 1e-13 that is noise, and the conditional integrand inherited it: panels
/// around 6.06 deviations kept returning an error estimate that fell only as
/// fast as the panel width, the signature of a rough integrand, so the routine
/// subdivided to its depth limit and refused rectangles as ordinary as
/// [6, 40] x [5, inf) at a correlation of 0.7.
///
/// `libm`'s is the FDLIBM routine. Against a sixty-digit evaluation it is right
/// to an ulp for ordinary arguments and never worse than 4e-14 out to 36
/// deviations -- three orders better than `statrs` at its best -- while running
/// about four times faster than routing the same quantity through the
/// regularised incomplete gamma, which is equally accurate but pays for a
/// generality this does not need. Below nought the argument is negative and the
/// result lies between one and two, so there is no cancellation to avoid.
fn normal_sf(value: f64) -> Result<f64, &'static str> {
    if !value.is_finite() {
        return Err("LATENT_MEDIATION_NORMAL_VARIATE_NOT_FINITE");
    }
    Ok(0.5 * libm::erfc(value / std::f64::consts::SQRT_2))
}

/// The point whose upper tail is `target`, for `target` at most `log 0.5`.
///
/// The forward function has no closed inverse, so this starts from the
/// asymptote `log sf(x) = -x^2/2 - log(2 pi)/2 - log x` -- which can be turned
/// round by repeated substitution -- and finishes with Newton steps on
/// `log sf`, whose slope is `-exp(log phi(x) - log sf(x))` and so costs
/// nothing extra. Working on the logarithm throughout means the routine
/// reaches wherever `log_normal_sf` reaches, rather than stopping at the
/// 1e-308 where an ordinary-scale quantile has to give up.
fn inverse_log_normal_sf(target: f64) -> Result<f64, &'static str> {
    if target.is_nan() {
        return Err("LATENT_MEDIATION_NORMAL_VARIATE_NOT_FINITE");
    }
    if target == f64::NEG_INFINITY {
        return Ok(f64::INFINITY);
    }
    if target > LOG_HALF {
        return Err("LATENT_MEDIATION_INVERSE_TAIL_OUT_OF_RANGE");
    }
    let mut point = if target > -700.0 {
        // The ordinary scale still carries the tail here, so `statrs` supplies
        // a starting point directly; the Newton steps below repair its own
        // error rather than inheriting it.
        -normal()?.inverse_cdf(target.exp())
    } else {
        let mut guess = (-2.0 * target - LOG_TWO_PI).max(1.0).sqrt();
        for _ in 0..60 {
            let next = (-2.0 * (target + 0.5 * LOG_TWO_PI + guess.ln()))
                .max(1.0)
                .sqrt();
            let settled = (next - guess).abs() <= 1.0e-15 * guess;
            guess = next;
            if settled {
                break;
            }
        }
        guess
    };
    for _ in 0..60 {
        let value = log_normal_sf(point)? - target;
        let slope = -(-0.5 * point * point - 0.5 * LOG_TWO_PI - log_normal_sf(point)?).exp();
        if !(slope < 0.0) || !slope.is_finite() {
            break;
        }
        let step = value / slope;
        if !step.is_finite() {
            break;
        }
        let next = (point - step).max(0.0);
        let settled = (next - point).abs() <= 1.0e-15 * point.max(1.0);
        point = next;
        if settled {
            break;
        }
    }
    Ok(point)
}

fn truncated_standard_normal(lower: f64, upper: f64, unit: f64) -> Result<f64, &'static str> {
    if !(0.0 < unit && unit < 1.0) {
        return Err("LATENT_MEDIATION_QMC_COORDINATE_INVALID");
    }
    if !(lower < upper) || lower.is_nan() || upper.is_nan() {
        return Err("LATENT_MEDIATION_TRUNCATION_INTERVAL_EMPTY");
    }
    // **An interval wholly in a tail is placed on the log scale.** Taking its
    // width on the ordinary scale first put a floor under the whole quasi-Monte
    // Carlo path at about 1e-308: an interval further out than that came back
    // as empty and the evaluation was refused, though nothing about it is
    // empty and the model above it works in logarithms throughout. Reflecting
    // the lower tail onto the upper one leaves a single case to write down.
    if upper <= 0.0 {
        return Ok(-truncated_standard_normal(-upper, -lower, 1.0 - unit)?);
    }
    if lower >= 0.0 {
        let at_lower = log_normal_sf(lower)?;
        let at_upper = if upper == f64::INFINITY {
            f64::NEG_INFINITY
        } else {
            log_normal_sf(upper)?
        };
        // The share of the lower tail that the interval takes up, held away
        // from cancellation at both ends.
        let share = -(at_upper - at_lower).exp_m1();
        return inverse_log_normal_sf(at_lower + (-unit * share).ln_1p());
    }
    let width = interval_probability(lower, upper)?;
    if width <= 0.0 {
        return Err("LATENT_MEDIATION_TRUNCATION_INTERVAL_EMPTY");
    }
    let distribution = normal()?;
    let lower_cumulative = if lower == f64::NEG_INFINITY {
        0.0
    } else {
        normal_cdf(lower)?
    };
    let cumulative = (lower_cumulative + unit * width).clamp(1.0e-300, 1.0 - 1.0e-16);
    Ok(distribution.inverse_cdf(cumulative))
}

fn halton(mut index: usize, base: usize) -> f64 {
    let mut fraction = 1.0;
    let mut value = 0.0;
    while index > 0 {
        fraction /= base as f64;
        value += fraction * (index % base) as f64;
        index /= base;
    }
    value
}

fn bivariate_rectangle(
    lower: &[f64],
    upper: &[f64],
    mean: &DVector<f64>,
    covariance: &DMatrix<f64>,
) -> Result<BivariateRectangle, &'static str> {
    let first_sd = covariance[(0, 0)].sqrt();
    let second_sd = covariance[(1, 1)].sqrt();
    if !first_sd.is_finite() || !second_sd.is_finite() || first_sd <= 0.0 || second_sd <= 0.0 {
        return Err("LATENT_MEDIATION_BIVARIATE_VARIANCE_INVALID");
    }
    let correlation = covariance[(0, 1)] / (first_sd * second_sd);
    if correlation.abs() >= 1.0 || !correlation.is_finite() {
        return Err("LATENT_MEDIATION_BIVARIATE_CORRELATION_INVALID");
    }
    let standardised_lower = [
        (lower[0] - mean[0]) / first_sd,
        (lower[1] - mean[1]) / second_sd,
    ];
    let standardised_upper = [
        (upper[0] - mean[0]) / first_sd,
        (upper[1] - mean[1]) / second_sd,
    ];

    // The corner-difference recipe resolves ordinary probabilities to near
    // machine accuracy, but in the tails the four corner terms cancel and the
    // difference is quadrature residue.  Everything the differences cannot
    // resolve is recomputed on the log scale by conditional quadrature, which
    // has no cancellation and needs no lower cut-off.
    let log_geometry_bound =
        log_bivariate_geometry_upper_bound(&standardised_lower, &standardised_upper, correlation)?;
    if log_geometry_bound > BIVARIATE_TAIL_CROSSOVER.ln() {
        // Ordinary scale is safe on this branch: the bound is above 1e-9.
        let geometry_bound = log_geometry_bound.exp();
        let cdf = |first: f64, second: f64| bivariate_normal_cdf(first, second, correlation);
        let corners = [
            cdf(standardised_upper[0], standardised_upper[1]),
            cdf(standardised_lower[0], standardised_upper[1]),
            cdf(standardised_upper[0], standardised_lower[1]),
            cdf(standardised_lower[0], standardised_lower[1]),
        ];
        if let [
            Ok(both_upper),
            Ok(low_first),
            Ok(low_second),
            Ok(both_lower),
        ] = corners
        {
            let corner_values = [both_upper, low_first, low_second, both_lower];
            let probability =
                corner_values[0] - corner_values[1] - corner_values[2] + corner_values[3];
            let rounding_budget =
                16.0 * f64::EPSILON * corner_values.iter().map(|value| value.abs()).sum::<f64>();
            let absolute_budget = 4.0 * BIVARIATE_ABSOLUTE_TOLERANCE + rounding_budget;
            if probability > absolute_budget {
                if probability > geometry_bound + absolute_budget {
                    return Err("LATENT_MEDIATION_BIVARIATE_PROBABILITY_OUTSIDE_BOUNDS");
                }
                return Ok(BivariateRectangle {
                    log_probability: probability.clamp(0.0, 1.0).ln(),
                    method: "bivariate_quadrature",
                });
            }
        } else if let Some(error) = corners.iter().find_map(|corner| match corner {
            Err(error) if *error != "LATENT_MEDIATION_BIVARIATE_PROBABILITY_UNRESOLVED" => {
                Some(*error)
            }
            _ => None,
        }) {
            return Err(error);
        }
    }
    let log_probability =
        log_bivariate_rectangle(&standardised_lower, &standardised_upper, correlation)?;
    // A relative slack of 1e-8 on the probability, which is an absolute slack
    // on its logarithm and so holds equally at every depth.
    if log_probability > log_geometry_bound + 1.0e-8 {
        return Err("LATENT_MEDIATION_BIVARIATE_PROBABILITY_OUTSIDE_BOUNDS");
    }
    Ok(BivariateRectangle {
        // A probability cannot exceed one, so its logarithm cannot exceed
        // nought; the ordinary-scale branch clamps for the same reason.
        log_probability: log_probability.min(0.0),
        method: "bivariate_tail_quadrature",
    })
}

struct BivariateRectangle {
    log_probability: f64,
    method: &'static str,
}

/// Log of the upper tail of the standard normal, stable arbitrarily far out.
fn log_normal_sf(value: f64) -> Result<f64, &'static str> {
    if value.is_nan() {
        return Err("LATENT_MEDIATION_NORMAL_VARIATE_NOT_FINITE");
    }
    if value == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if value == f64::INFINITY {
        return Ok(f64::NEG_INFINITY);
    }
    if value <= 6.0 {
        // The survival value is at least 1e-9 here, so its logarithm keeps
        // full relative accuracy.
        return Ok(normal_sf(value)?.ln());
    }
    // Mills-ratio continued fraction: the survival equals
    // phi(x) / (x + 1/(x + 2/(x + 3/(...)))), evaluated backward.
    let mut tail = 0.0;
    for level in (1..=40u32).rev() {
        tail = f64::from(level) / (value + tail);
    }
    Ok(-0.5 * value * value - 0.5 * LOG_TWO_PI - (value + tail).ln())
}

/// Log of a standard-normal interval probability, stable in either tail.
fn log_interval_probability(lower: f64, upper: f64) -> Result<f64, &'static str> {
    if lower >= upper {
        return Ok(f64::NEG_INFINITY);
    }
    if lower >= 0.0 {
        let log_lower_tail = log_normal_sf(lower)?;
        let log_upper_tail = log_normal_sf(upper)?;
        return Ok(log_lower_tail + (-(log_upper_tail - log_lower_tail).exp()).ln_1p());
    }
    if upper <= 0.0 {
        return log_interval_probability(-upper, -lower);
    }
    Ok(interval_probability(lower, upper)?.ln())
}

/// Log of a standardised bivariate normal rectangle by conditional
/// quadrature: the outer coordinate is integrated with the inner interval
/// probability evaluated on the log scale, so no cancellation occurs and
/// tails far beyond the corner-difference recipe stay accurate.
fn log_bivariate_rectangle(
    lower: &[f64; 2],
    upper: &[f64; 2],
    correlation: f64,
) -> Result<f64, &'static str> {
    // Integrate over the more extreme coordinate, which concentrates the
    // integrand and keeps the inner interval ordinary.
    //
    // **Extremity is how far into a tail the interval sits, not how near an
    // edge is to the origin.** Scoring it as the smaller absolute endpoint
    // called `(-inf, 40]` more extreme than `(7, inf)`, when the first holds
    // essentially all the mass and the second holds 1e-12 of it, so the outer
    // and inner coordinates were chosen the wrong way round for every rectangle
    // with a half-line in it. An interval containing the mode is not extreme at
    // all, whatever its edges; one lying wholly in a tail is as extreme as its
    // nearest edge is far out.
    let extremity = |low: f64, high: f64| {
        if low > 0.0 {
            low
        } else if high < 0.0 {
            -high
        } else {
            0.0
        }
    };
    let swap = extremity(lower[1], upper[1]) > extremity(lower[0], upper[0]);
    let (outer, inner) = if swap {
        (([lower[1], upper[1]]), ([lower[0], upper[0]]))
    } else {
        (([lower[0], upper[0]]), ([lower[1], upper[1]]))
    };
    // Factored rather than `(1 - rho^2)`: the squared form cancels as the
    // correlation approaches one, and the conditional distribution is
    // narrowest there, so the divisor most needs its digits.
    let conditional_sd = ((1.0 - correlation) * (1.0 + correlation)).sqrt();
    if !(conditional_sd > 0.0) {
        return Err("LATENT_MEDIATION_BIVARIATE_CORRELATION_INVALID");
    }
    let log_integrand = |x: f64| -> Result<f64, &'static str> {
        let shifted_lower = (inner[0] - correlation * x) / conditional_sd;
        let shifted_upper = (inner[1] - correlation * x) / conditional_sd;
        Ok(-0.5 * x * x - 0.5 * LOG_TWO_PI
            + log_interval_probability(shifted_lower, shifted_upper)?)
    };
    // The integrand is log-concave, so it has a single peak and a
    // golden-section search finds it -- but only once the search is given a
    // bracket that contains it.
    //
    // **Where the peak sits is a property of the whole rectangle, not of the
    // outer interval alone.** With correlation `rho` and an inner interval
    // beginning at `c`, the exponent `-x^2/2 + log P(inner | x)` is stationary
    // where `x = rho * m` for `m` the conditional mean of the inner interval,
    // which puts the peak near `rho * c` -- far from the origin and far from
    // the outer edges whenever the inner interval is distant. A window fixed
    // relative to the rectangle cannot find that, and it also loses the
    // ordinary case: anchored sixty either side of the rectangle's own edge,
    // a left half-line `(-inf, t]` was searched over `[t - 60, t]`, so once `t`
    // passed sixty the range began above the mode and threw nearly all the mass
    // away. The probability of a strictly growing region then strictly shrank,
    // by 1805 nats between `t = 40` and `t = 120`, and past `t = 300` the
    // quadrature failed outright.
    //
    // So rather than guess a window, climb: start from the point of the outer
    // interval nearest the origin, step out doubling the stride while the
    // integrand rises, and stop when it falls or the interval ends. Log
    // concavity makes that one climb enough, and it brackets the peak wherever
    // the peak happens to be.
    let lower_edge = outer[0];
    let upper_edge = outer[1];
    if !(lower_edge < upper_edge) {
        return Ok(f64::NEG_INFINITY);
    }
    let anchor = 0.0_f64.clamp(lower_edge, upper_edge);
    let towards = |direction: f64, from: f64, step: f64| -> f64 {
        let proposed = from + direction * step;
        if direction > 0.0 {
            proposed.min(upper_edge)
        } else {
            proposed.max(lower_edge)
        }
    };
    let mut low = anchor;
    let mut high = anchor;
    let at_anchor = log_integrand(anchor)?;
    for direction in [1.0_f64, -1.0] {
        let mut previous = at_anchor;
        let mut step = 1.0;
        let mut reached = anchor;
        for _ in 0..CLIMB_STEPS {
            let candidate = towards(direction, anchor, step);
            if candidate == reached {
                break;
            }
            let value = log_integrand(candidate)?;
            reached = candidate;
            if value <= previous {
                break;
            }
            previous = value;
            step *= 2.0;
        }
        if direction > 0.0 {
            high = reached;
        } else {
            low = reached;
        }
    }
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..120 {
        // Stop once the bracket is at the resolution of the numbers in it.
        // Running the full count regardless spent most of its evaluations
        // splitting an interval that had already collapsed.
        if high - low <= 1.0e-13 * (1.0 + high.abs().max(low.abs())) {
            break;
        }
        let first = high - golden * (high - low);
        let second = low + golden * (high - low);
        if log_integrand(first)? < log_integrand(second)? {
            low = first;
        } else {
            high = second;
        }
    }
    let peak = 0.5 * (low + high);
    let log_peak = log_integrand(peak)?;
    if log_peak == f64::NEG_INFINITY {
        return Ok(f64::NEG_INFINITY);
    }
    // Integrate out to where the integrand has fallen eighty logs below its
    // peak; the part beyond contributes a relative 1e-35 and is immaterial at
    // any tolerance used here. The same doubling climb finds that point, so the
    // integration range is set by the integrand rather than by a constant, and
    // a bisection then places it precisely.
    let drop = 80.0;
    let spent = |direction: f64| -> Result<f64, &'static str> {
        let mut inside = peak;
        let mut step = 1.0;
        let mut outside = None;
        for _ in 0..CLIMB_STEPS {
            let candidate = towards(direction, peak, step);
            if log_integrand(candidate)? < log_peak - drop {
                outside = Some(candidate);
                break;
            }
            if candidate == inside {
                break;
            }
            inside = candidate;
            step *= 2.0;
        }
        let Some(mut outside) = outside else {
            return Ok(inside);
        };
        for _ in 0..80 {
            let middle = 0.5 * (inside + outside);
            if log_integrand(middle)? >= log_peak - drop {
                inside = middle;
            } else {
                outside = middle;
            }
        }
        Ok(outside)
    };
    let left_edge = spent(-1.0)?;
    let right_edge = spent(1.0)?;
    let shifted = |x: f64| match log_integrand(x) {
        Ok(value) => (value - log_peak).exp(),
        Err(_) => f64::NAN,
    };
    let width = right_edge - left_edge;
    if !(width > 0.0) {
        return Ok(f64::NEG_INFINITY);
    }
    // **How much rounding the integrand carries depends on how large its
    // exponent is.** `shifted` is `exp(g(x) - g(peak))`, and `g` is computed to
    // a relative accuracy of an epsilon, so its absolute error is an epsilon of
    // its own size and the exponential inherits that as relative noise. At a
    // correlation of -0.999999 the exponent runs to a million and the noise is
    // 2e-10 rather than 2e-16. The request stays where it is -- an ordinary
    // rectangle should still be answered to 1e-13 -- and the noise is passed
    // in separately, so the quadrature stops at its own rounding rather than
    // subdividing to its depth limit and refusing an answer that is perfectly
    // well defined.
    let noise = f64::EPSILON * log_peak.abs().max(1.0);
    let quadrature = adaptive_simpson(
        &shifted,
        left_edge,
        right_edge,
        1.0e-13 * width.max(1.0e-6),
        noise,
        32,
    )?;
    if !(quadrature.value > 0.0) {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(log_peak + quadrature.value.ln())
}

/// An upper bound on a standardised rectangle's probability, on the log scale.
///
/// **On the log scale because the bound is used to check the tail answer, and
/// an ordinary-scale bound underflows before the answers it is meant to
/// check.** Past about 38 deviations the bound arrives as nought, the guard
/// reads `bound > 0` and steps aside, and the deep tail -- the one place the
/// conditional quadrature is the only recipe available and so the one place a
/// check is worth having -- went unchecked.
fn log_bivariate_geometry_upper_bound(
    lower: &[f64; 2],
    upper: &[f64; 2],
    correlation: f64,
) -> Result<f64, &'static str> {
    if correlation.abs() >= 1.0 || !correlation.is_finite() {
        return Err("LATENT_MEDIATION_BIVARIATE_CORRELATION_INVALID");
    }
    let marginal_bound = log_interval_probability(lower[0], upper[0])?
        .min(log_interval_probability(lower[1], upper[1])?);

    // For standardised X and Y, S=X+Y and D=X-Y are independent Gaussian
    // variables.  Every point in the rectangle lies in both induced
    // intervals, hence P(rectangle) <= P(S interval) P(D interval).
    let sum_sd = (2.0 * (1.0 + correlation)).sqrt();
    let difference_sd = (2.0 * (1.0 - correlation)).sqrt();
    let sum_probability = log_interval_probability(
        (lower[0] + lower[1]) / sum_sd,
        (upper[0] + upper[1]) / sum_sd,
    )?;
    let difference_probability = log_interval_probability(
        (lower[0] - upper[1]) / difference_sd,
        (upper[0] - lower[1]) / difference_sd,
    )?;
    Ok(marginal_bound
        .min(sum_probability + difference_probability)
        .min(0.0))
}

fn bivariate_normal_cdf(first: f64, second: f64, correlation: f64) -> Result<f64, &'static str> {
    if correlation.abs() >= 1.0 || !correlation.is_finite() {
        return Err("LATENT_MEDIATION_BIVARIATE_CORRELATION_INVALID");
    }
    if first == f64::NEG_INFINITY || second == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if first == f64::INFINITY && second == f64::INFINITY {
        return Ok(1.0);
    }
    if first == f64::INFINITY {
        return normal_cdf(second);
    }
    if second == f64::INFINITY {
        return normal_cdf(first);
    }
    if !first.is_finite() || !second.is_finite() {
        return Err("LATENT_MEDIATION_BIVARIATE_THRESHOLD_INVALID");
    }
    let geometry_bound = log_bivariate_geometry_upper_bound(
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[first, second],
        correlation,
    )?
    .exp();
    if geometry_bound <= BIVARIATE_ABSOLUTE_TOLERANCE {
        return Err("LATENT_MEDIATION_BIVARIATE_PROBABILITY_UNRESOLVED");
    }
    if correlation < 0.0 {
        return Ok(normal_cdf(first)? - bivariate_normal_cdf(first, -second, -correlation)?);
    }
    let independent = normal_cdf(first)? * normal_cdf(second)?;
    if correlation.abs() <= f64::EPSILON {
        return Ok(independent);
    }
    let integrand = |angle: f64| {
        let sine = angle.sin();
        let cosine = angle.cos();
        let exponent =
            (first - second).powi(2) / (2.0 * cosine * cosine) + first * second / (1.0 + sine);
        (-exponent).exp() / (2.0 * std::f64::consts::PI)
    };
    let quadrature = adaptive_simpson(
        &integrand,
        0.0,
        correlation.asin(),
        BIVARIATE_ABSOLUTE_TOLERANCE / 8.0,
        // An ordinary integrand on the ordinary scale, so an epsilon.
        f64::EPSILON,
        24,
    )?;
    let probability = independent + quadrature.value;
    if probability <= quadrature.estimated_error {
        return Err("LATENT_MEDIATION_BIVARIATE_PROBABILITY_UNRESOLVED");
    }
    let upper_bound = normal_cdf(first)?.min(normal_cdf(second)?);
    // **The Frechet bounds can cross by rounding, and `clamp` panics when they
    // do.** With both marginals at essentially one, the lower bound is
    // `p + q - 1` and the upper is `min(p, q)`; these meet exactly at one and
    // an ulp of arithmetic is enough to put the lower above the upper --
    // measured at 0.9999999910267894 against 0.9999999910267893, which brought
    // the whole process down through the Python boundary rather than returning
    // an error. Ordering them costs nothing and the interval they describe is
    // a point at that precision anyway.
    let lower_bound = (normal_cdf(first)? + normal_cdf(second)? - 1.0)
        .max(0.0)
        .min(upper_bound);
    if probability < lower_bound - 1.0e-10 || probability > upper_bound + 1.0e-10 {
        return Err("LATENT_MEDIATION_BIVARIATE_PROBABILITY_OUTSIDE_BOUNDS");
    }
    Ok(probability.clamp(lower_bound, upper_bound))
}

#[derive(Debug)]
struct QuadratureResult {
    value: f64,
    estimated_error: f64,
}

fn adaptive_simpson<F>(
    function: &F,
    lower: f64,
    upper: f64,
    tolerance: f64,
    // The relative rounding the integrand itself carries. An ordinary
    // integrand carries an epsilon; one built from a large exponent carries an
    // epsilon of that exponent.
    noise: f64,
    maximum_depth: usize,
) -> Result<QuadratureResult, &'static str>
where
    F: Fn(f64) -> f64,
{
    fn simpson<F>(function: &F, lower: f64, upper: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let midpoint = lower.midpoint(upper);
        (upper - lower) * (function(lower) + 4.0 * function(midpoint) + function(upper)) / 6.0
    }
    fn recurse<F>(
        function: &F,
        lower: f64,
        upper: f64,
        estimate: f64,
        tolerance: f64,
        noise: f64,
        depth: usize,
    ) -> Result<QuadratureResult, &'static str>
    where
        F: Fn(f64) -> f64,
    {
        let midpoint = lower.midpoint(upper);
        let left = simpson(function, lower, midpoint);
        let right = simpson(function, midpoint, upper);
        let error = left + right - estimate;
        if !left.is_finite() || !right.is_finite() {
            return Err("LATENT_MEDIATION_BIVARIATE_QUADRATURE_NOT_FINITE");
        }
        // The error budget is halved at every level, so it eventually asks for
        // less than the arithmetic can deliver: a panel is accepted once its
        // error reaches the level of its own rounding, because no amount of
        // further splitting can improve on that. Without this the routine
        // answers a rough integrand by subdividing to its depth limit and then
        // refusing, which is the worst of both -- it spends the most work
        // exactly where it will fail, and a likelihood that declines to be
        // evaluated stops an optimiser dead.
        let rounding = 16.0 * noise * (left.abs() + right.abs());
        if error.abs() <= (15.0 * tolerance).max(rounding) {
            Ok(QuadratureResult {
                value: left + right + error / 15.0,
                estimated_error: error.abs() / 15.0,
            })
        } else if depth == 0 {
            Err("LATENT_MEDIATION_BIVARIATE_QUADRATURE_DID_NOT_CONVERGE")
        } else {
            let left_result = recurse(
                function,
                lower,
                midpoint,
                left,
                tolerance / 2.0,
                noise,
                depth - 1,
            )?;
            let right_result = recurse(
                function,
                midpoint,
                upper,
                right,
                tolerance / 2.0,
                noise,
                depth - 1,
            )?;
            Ok(QuadratureResult {
                value: left_result.value + right_result.value,
                estimated_error: left_result.estimated_error + right_result.estimated_error,
            })
        }
    }
    if lower >= upper {
        return Ok(QuadratureResult {
            value: 0.0,
            estimated_error: 0.0,
        });
    }
    let estimate = simpson(function, lower, upper);
    if !estimate.is_finite() {
        return Err("LATENT_MEDIATION_BIVARIATE_QUADRATURE_NOT_FINITE");
    }
    recurse(
        function,
        lower,
        upper,
        estimate,
        tolerance,
        noise,
        maximum_depth,
    )
}

/// A deterministic stream, so a simulated campaign can be rerun exactly.
struct Stream(u64);

impl Stream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let first = self.uniform();
        let second = self.uniform();
        (-2.0 * first.ln()).sqrt() * (2.0 * std::f64::consts::PI * second).cos()
    }
}

/// What one family in a simulated design looks like: who is related to whom,
/// where their thresholds sit, and which of the three observations each person
/// contributes.
#[derive(Clone, Debug)]
pub struct LatentMediationDesign {
    pub relationship: Vec<Vec<f64>>,
    /// Covariates acting on the latent mediator, one row per person, and the
    /// coefficients to draw them with. Empty for none.
    pub mediator_design: Vec<Vec<f64>>,
    pub mediator_coefficients: Vec<f64>,
    /// The same for the latent outcome.
    pub outcome_design: Vec<Vec<f64>>,
    pub outcome_coefficients: Vec<f64>,
    pub mediator_threshold: Vec<f64>,
    pub outcome_threshold: Vec<f64>,
    /// The known error variance where the mediator is measured, and `None`
    /// where it is not. At least one family must measure it somewhere or the
    /// mediator scale is not identified and `fit` will say so.
    pub mediator_measurement_error_variance: Vec<Option<f64>>,
    /// Whether each person contributes a fallible binary reading of their true
    /// mediator state.
    pub observe_mediator_proxy: Vec<bool>,
    pub mediator_proxy_sensitivity: Vec<f64>,
    pub mediator_proxy_specificity: Vec<f64>,
    pub observe_outcome: Vec<bool>,
    pub ascertainment: String,
    pub proband_index: Option<usize>,
}

/// How many redraws a proband-conditioned family is allowed before the design
/// is called impossible. A threshold so far out that a case essentially never
/// occurs would otherwise spin.
const ASCERTAINMENT_ATTEMPTS: usize = 100_000;

/// Draw families from the model the likelihood integrates.
///
/// **The same covariance construction as the fit, deliberately.** A simulator
/// that built the covariance its own way would make a calibration measure the
/// agreement between two constructions rather than the behaviour of the test;
/// the construction itself is checked against an independently written
/// evaluation elsewhere, which is where that assurance belongs.
///
/// Conditioning on a proband is done by drawing and redrawing until the named
/// person is a case, which is what the model's denominator assumes and what a
/// clinic roster actually is. Drawing unconditionally and keeping the cases
/// would be a different design.
///
/// # Errors
///
/// Returns a stable code where the design does not describe a family, where a
/// proband is asked for and not named, or where the ascertainment cannot be
/// satisfied in a reasonable number of attempts.
pub fn simulate(
    design: &LatentMediationDesign,
    parameters: LatentMediationParameters,
    families: usize,
    seed: u64,
) -> Result<Vec<LatentMediationFamilyInput>, &'static str> {
    validate_parameters(parameters)?;
    let size = design.relationship.len();
    if size == 0 {
        return Err("LATENT_MEDIATION_DESIGN_EMPTY");
    }
    let lengths = [
        design.mediator_threshold.len(),
        design.outcome_threshold.len(),
        design.mediator_measurement_error_variance.len(),
        design.observe_mediator_proxy.len(),
        design.mediator_proxy_sensitivity.len(),
        design.mediator_proxy_specificity.len(),
        design.observe_outcome.len(),
    ];
    if lengths.iter().any(|length| *length != size)
        || design.relationship.iter().any(|row| row.len() != size)
    {
        return Err("LATENT_MEDIATION_DESIGN_SHAPE_INVALID");
    }
    let conditioned = match design.ascertainment.as_str() {
        "population_unconditioned" => {
            if design.proband_index.is_some() {
                return Err("LATENT_MEDIATION_PROBAND_NOT_ALLOWED");
            }
            None
        }
        "condition_on_named_proband_case" => match design.proband_index {
            Some(index) if index < size && design.observe_outcome[index] => Some(index),
            Some(_) => return Err("LATENT_MEDIATION_PROBAND_INDEX_INVALID"),
            None => return Err("LATENT_MEDIATION_PROBAND_INDEX_MISSING"),
        },
        _ => return Err("LATENT_MEDIATION_ASCERTAINMENT_UNKNOWN"),
    };

    let relationship = DMatrix::from_fn(size, size, |row, column| design.relationship[row][column]);
    let covariance = directional_covariance(&relationship, parameters)?;
    let factor = covariance
        .cholesky()
        .ok_or("LATENT_MEDIATION_COVARIANCE_NOT_POSITIVE_DEFINITE")?
        .l();

    let mediator_terms = design_width(&design.mediator_design, size)?;
    let outcome_terms = design_width(&design.outcome_design, size)?;
    if mediator_terms != design.mediator_coefficients.len()
        || outcome_terms != design.outcome_coefficients.len()
    {
        return Err("LATENT_MEDIATION_COEFFICIENT_COUNT_WRONG");
    }
    // The mean each person's covariates put them at, in the same stacked order
    // as the latent vector.
    let mut shift = vec![0.0; 2 * size];
    for person in 0..size {
        for (term, weight) in design.mediator_coefficients.iter().enumerate() {
            shift[person] += design.mediator_design[person][term] * weight;
        }
        for (term, weight) in design.outcome_coefficients.iter().enumerate() {
            shift[size + person] += design.outcome_design[person][term] * weight;
        }
    }

    let mut stream = Stream(seed);
    let mut drawn = Vec::with_capacity(families);
    for _ in 0..families {
        let mut attempts = 0usize;
        let family = loop {
            attempts += 1;
            if attempts > ASCERTAINMENT_ATTEMPTS {
                return Err("LATENT_MEDIATION_ASCERTAINMENT_UNREACHABLE");
            }
            // Process-major, mediator block first, as the covariance is built.
            let draw = DVector::from_fn(2 * size, |_, _| stream.normal());
            let latent = &factor * draw + DVector::from_column_slice(&shift);
            let case = |person: usize| latent[size + person] > design.outcome_threshold[person];
            if let Some(proband) = conditioned
                && !case(proband)
            {
                continue;
            }
            let mut measurement = vec![None; size];
            let mut proxy = vec![None; size];
            let mut outcome = vec![None; size];
            for person in 0..size {
                if let Some(variance) = design.mediator_measurement_error_variance[person] {
                    measurement[person] = Some(latent[person] + variance.sqrt() * stream.normal());
                }
                if design.observe_mediator_proxy[person] {
                    let truth = latent[person] > design.mediator_threshold[person];
                    let right = if truth {
                        design.mediator_proxy_sensitivity[person]
                    } else {
                        design.mediator_proxy_specificity[person]
                    };
                    let agrees = stream.uniform() < right;
                    proxy[person] = Some(i8::from(truth == agrees));
                }
                if design.observe_outcome[person] {
                    outcome[person] = Some(i8::from(case(person)));
                }
            }
            break LatentMediationFamilyInput {
                relationship: design.relationship.clone(),
                latent_mean: vec![0.0; 2 * size],
                mediator_design: design.mediator_design.clone(),
                outcome_design: design.outcome_design.clone(),
                mediator_measurement: measurement,
                mediator_measurement_error_variance: design
                    .mediator_measurement_error_variance
                    .clone(),
                mediator_proxy_status: proxy,
                outcome_status: outcome,
                mediator_threshold: design.mediator_threshold.clone(),
                outcome_threshold: design.outcome_threshold.clone(),
                mediator_proxy_sensitivity: design.mediator_proxy_sensitivity.clone(),
                mediator_proxy_specificity: design.mediator_proxy_specificity.clone(),
                ascertainment: design.ascertainment.clone(),
                proband_index: design.proband_index,
            };
        };
        drawn.push(family);
    }
    Ok(drawn)
}

#[cfg(feature = "python")]
pub mod python;

/// A test of the vertical estimand `a b` against nought.
#[derive(Clone, Debug)]
pub struct VerticalTest {
    /// The deviance for the mediator loading `a = 0`, and its p-value against
    /// the 50:50 reference that a boundary null needs.
    pub loading_statistic: f64,
    pub loading_p_value: f64,
    /// The deviance for the path `b = 0`, which is interior, so an ordinary
    /// chi-square on one degree of freedom.
    pub path_statistic: f64,
    pub path_p_value: f64,
    /// The p-value for `a b = 0`, which is the larger of the two above.
    pub p_value: f64,
    pub rule: &'static str,
}

impl LatentMediationModel {
    /// Test the vertical estimand `a b` against nought.
    ///
    /// **The null is a union, not a point.** `a b = 0` holds whenever the
    /// mediator carries no inherited signal (`a = 0`) *or* the mediator does
    /// not reach the outcome (`b = 0`), and those are different models. A
    /// single likelihood ratio has no reference distribution here, which is
    /// why the model reported point estimates and no p-value until now.
    ///
    /// The construction is the intersection-union test: the union null is
    /// rejected only when **both** parts are rejected, so the p-value is the
    /// larger of the two. That is exactly level `alpha` for any `alpha`, at
    /// the cost of being conservative — most so near `a = b = 0`, where both
    /// parts are true at once.
    ///
    /// Each part has its own reference. `a` is bounded below at nought, so its
    /// null sits on the boundary and takes the even mixture of a point mass
    /// with chi-square on one. `b` is signed and interior, so it takes an
    /// ordinary chi-square on one.
    ///
    /// This tests the *mediated path*, not whether mediation is the right
    /// account of the data. A trait and a mediator sharing inherited causes
    /// will reject this null without anything being mediated, and no
    /// likelihood can tell the two apart.
    ///
    /// # Errors
    ///
    /// Returns a stable `LATENT_MEDIATION_*` code where any of the three fits
    /// fails.
    pub fn test_vertical(&self) -> Result<VerticalTest, &'static str> {
        let free = self.fit()?;
        // **The even mixture below assumes `a` is the only parameter on a
        // bound.** Where `d` rests on its lower bound too -- a trait with
        // little inherited outcome variance, which is not rare -- the free fit
        // sits on the corner of the cone rather than on a face, and the
        // reference for the `a = 0` likelihood ratio is a different
        // chi-bar-square. Because the intersection-union rule reports whichever
        // part is larger, a wrong loading p-value becomes the reported one
        // whenever it binds, so this is refused rather than answered. The fit
        // already knows: it records exactly this in `boundary_parameters`.
        // `ComponentModel` guards the same case and refuses for the same reason.
        //
        // **The reference belongs to the null, so the null fit has to be
        // asked as well.** The reference for `a = 0` is set by which other
        // parameters rest on a bound where that null holds, and the free fit
        // is not that point: `d` can be comfortably positive with `a` free and
        // fall to nought once `a` is held there, which is the corner the guard
        // exists to catch and the free fit alone reports nothing about. Either
        // fit resting on `d` is enough to refuse.
        if free.boundary_parameters.contains(&"d") {
            return Err("LATENT_MEDIATION_ANOTHER_LOADING_AT_ZERO");
        }
        let without_loading = self.fit_holding(&[0])?;
        if without_loading.boundary_parameters.contains(&"d") {
            return Err("LATENT_MEDIATION_ANOTHER_LOADING_AT_ZERO");
        }
        let without_path = self.fit_holding(&[1])?;

        let loading_statistic =
            crate::deviance::deviance(free.log_likelihood, without_loading.log_likelihood);
        let path_statistic =
            crate::deviance::deviance(free.log_likelihood, without_path.log_likelihood);
        // `a` rests on its lower bound under its null, so half the mass of the
        // reference sits at nought; `b` is interior and takes the whole of it.
        let loading_p_value = crate::deviance::p_value(loading_statistic, |t| {
            0.5 * crate::deviance::chi2_upper_tail(t, 1.0)
        });
        let path_p_value =
            crate::deviance::p_value(path_statistic, |t| crate::deviance::chi2_upper_tail(t, 1.0));

        Ok(VerticalTest {
            loading_statistic,
            loading_p_value,
            path_statistic,
            path_p_value,
            p_value: loading_p_value.max(path_p_value),
            rule: "intersection_union_of_boundary_and_interior",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A covariate the model can estimate rather than one somebody had to
    /// remove beforehand. Regressing age out of hearing first treats an
    /// estimated mean as a known one, and the variance components inherit the
    /// error; fitting it jointly does not.
    #[test]
    fn a_covariate_effect_is_recovered() {
        let parameters = LatentMediationParameters {
            a: 0.6,
            b: 0.4,
            c_prime: 0.2,
            d: 0.7,
            sigma_m2: 0.5,
        };
        // One covariate on each process, taking a different value for each
        // person so it is not confounded with the family mean.
        let design = LatentMediationDesign {
            mediator_design: vec![vec![-1.0], vec![1.0]],
            mediator_coefficients: vec![0.8],
            outcome_design: vec![vec![-1.0], vec![1.0]],
            outcome_coefficients: vec![-0.5],
            relationship: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            mediator_threshold: vec![0.0, 0.0],
            outcome_threshold: vec![0.0, 0.0],
            mediator_measurement_error_variance: vec![Some(0.15), Some(0.15)],
            observe_mediator_proxy: vec![false, false],
            mediator_proxy_sensitivity: vec![0.8, 0.8],
            mediator_proxy_specificity: vec![0.85, 0.85],
            observe_outcome: vec![true, true],
            ascertainment: "population_unconditioned".to_owned(),
            proband_index: None,
        };
        let families = simulate(&design, parameters, 1_500, 20_260_817).expect("draws");
        let model = LatentMediationModel::build(families, 256).expect("model");
        let fit = model.fit().expect("fit");

        assert_eq!(fit.coefficients.len(), 2);
        let (mediator, outcome) = (fit.coefficients[0], fit.coefficients[1]);
        // The mediator's coefficient is pinned by the continuous measurement,
        // so it should come back close.
        assert!(
            (mediator - 0.8).abs() < 0.1,
            "mediator coefficient {mediator} against 0.8"
        );
        // The outcome's is read through a threshold and is looser, but it must
        // at least have the right sign and order.
        assert!(
            outcome < 0.0 && (outcome - -0.5).abs() < 0.4,
            "outcome coefficient {outcome} against -0.5"
        );
    }

    /// A likelihood evaluated at coefficients nobody supplied is a number that
    /// gets quoted, so it is refused rather than defaulted to nought.
    #[test]
    fn a_model_with_covariates_refuses_to_be_evaluated_without_them() {
        let design = LatentMediationDesign {
            mediator_design: vec![vec![1.0], vec![-1.0]],
            mediator_coefficients: vec![0.3],
            outcome_design: Vec::new(),
            outcome_coefficients: Vec::new(),
            relationship: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            mediator_threshold: vec![0.0, 0.0],
            outcome_threshold: vec![0.0, 0.0],
            mediator_measurement_error_variance: vec![Some(0.15), Some(0.15)],
            observe_mediator_proxy: vec![false, false],
            mediator_proxy_sensitivity: vec![0.8, 0.8],
            mediator_proxy_specificity: vec![0.85, 0.85],
            observe_outcome: vec![true, true],
            ascertainment: "population_unconditioned".to_owned(),
            proband_index: None,
        };
        let parameters = LatentMediationParameters {
            a: 0.6,
            b: 0.4,
            c_prime: 0.2,
            d: 0.7,
            sigma_m2: 0.5,
        };
        let families = simulate(&design, parameters, 20, 5).expect("draws");
        let model = LatentMediationModel::build(families, 256).expect("model");
        assert_eq!(
            model.evaluate(parameters).err(),
            Some("LATENT_MEDIATION_COEFFICIENTS_REQUIRED")
        );
        assert!(model.evaluate_at(parameters, &[0.3]).is_ok());
        assert_eq!(
            model.evaluate_at(parameters, &[]).err(),
            Some("LATENT_MEDIATION_COEFFICIENT_COUNT_WRONG")
        );
    }

    /// A simulator that does not draw from the model it is meant to check is
    /// worse than none: a calibration built on it would measure the gap between
    /// two constructions and call it a rejection rate. So the draws are held to
    /// the covariance the likelihood assumes, and to the case rate its
    /// thresholds imply.
    #[test]
    fn the_simulator_draws_from_the_model_the_likelihood_integrates() {
        let parameters = LatentMediationParameters {
            a: 0.6,
            b: 0.4,
            c_prime: 0.2,
            d: 0.7,
            sigma_m2: 0.5,
        };
        let relationship = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        // The mediator measured almost exactly, so the measurements stand in
        // for the latent mediator itself.
        let design = LatentMediationDesign {
            mediator_design: Vec::new(),
            mediator_coefficients: Vec::new(),
            outcome_design: Vec::new(),
            outcome_coefficients: Vec::new(),
            relationship: relationship.clone(),
            mediator_threshold: vec![0.0, 0.0],
            outcome_threshold: vec![0.3, 0.3],
            mediator_measurement_error_variance: vec![Some(1.0e-8), Some(1.0e-8)],
            observe_mediator_proxy: vec![false, false],
            mediator_proxy_sensitivity: vec![0.8, 0.8],
            mediator_proxy_specificity: vec![0.85, 0.85],
            observe_outcome: vec![true, true],
            ascertainment: "population_unconditioned".to_owned(),
            proband_index: None,
        };
        let families = simulate(&design, parameters, 200_000, 20_260_817).expect("draws");

        let matrix = DMatrix::from_fn(2, 2, |row, column| relationship[row][column]);
        let truth = directional_covariance(&matrix, parameters).expect("covariance");

        let first: Vec<f64> = families
            .iter()
            .map(|family| family.mediator_measurement[0].expect("measured"))
            .collect();
        let second: Vec<f64> = families
            .iter()
            .map(|family| family.mediator_measurement[1].expect("measured"))
            .collect();
        let count = first.len() as f64;
        let mean_first = first.iter().sum::<f64>() / count;
        let mean_second = second.iter().sum::<f64>() / count;
        let variance = first
            .iter()
            .map(|value| (value - mean_first).powi(2))
            .sum::<f64>()
            / count;
        let covariance = first
            .iter()
            .zip(&second)
            .map(|(one, two)| (one - mean_first) * (two - mean_second))
            .sum::<f64>()
            / count;
        // Four standard errors of the sample covariance at this size.
        let tolerance = 4.0 * truth[(0, 0)] / count.sqrt();
        assert!(
            (variance - truth[(0, 0)]).abs() < tolerance,
            "mediator variance {variance} against {}",
            truth[(0, 0)]
        );
        assert!(
            (covariance - truth[(0, 1)]).abs() < tolerance,
            "between-sibling mediator covariance {covariance} against {}",
            truth[(0, 1)]
        );

        // And the outcome case rate against the threshold its variance implies.
        let cases = families
            .iter()
            .filter(|family| family.outcome_status[0] == Some(1))
            .count() as f64
            / count;
        let outcome_sd = truth[(2, 2)].sqrt();
        let expected = normal_sf(0.3 / outcome_sd).expect("tail");
        assert!(
            (cases - expected).abs() < 4.0 * (expected * (1.0 - expected) / count).sqrt(),
            "case rate {cases} against {expected}"
        );
    }

    /// Conditioning on a proband means every drawn family has one, and the
    /// others are not thereby all cases.
    #[test]
    fn conditioning_on_a_proband_draws_families_that_have_one() {
        let design = LatentMediationDesign {
            mediator_design: Vec::new(),
            mediator_coefficients: Vec::new(),
            outcome_design: Vec::new(),
            outcome_coefficients: Vec::new(),
            relationship: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            mediator_threshold: vec![0.0, 0.0],
            outcome_threshold: vec![1.5, 1.5],
            mediator_measurement_error_variance: vec![Some(0.15), None],
            observe_mediator_proxy: vec![false, true],
            mediator_proxy_sensitivity: vec![0.8, 0.8],
            mediator_proxy_specificity: vec![0.85, 0.85],
            observe_outcome: vec![true, true],
            ascertainment: "condition_on_named_proband_case".to_owned(),
            proband_index: Some(0),
        };
        let parameters = LatentMediationParameters {
            a: 0.6,
            b: 0.4,
            c_prime: 0.2,
            d: 0.7,
            sigma_m2: 0.5,
        };
        let families = simulate(&design, parameters, 5_000, 7).expect("draws");
        assert!(families.iter().all(|f| f.outcome_status[0] == Some(1)));
        let relatives = families
            .iter()
            .filter(|f| f.outcome_status[1] == Some(1))
            .count();
        // The relative is enriched by the shared liability but nowhere near
        // certain; if they were all cases the conditioning would be wrong.
        assert!(
            relatives > 0 && relatives < families.len(),
            "{relatives} relatives affected"
        );
    }

    /// The normal tail against a sixty-digit reference, at the depths the
    /// conditional quadrature actually visits. `statrs`'s `Normal::cdf` is
    /// wrong by about 5e-11 here, which is noise to a quadrature asking for
    /// 1e-13 and was enough to make it refuse ordinary rectangles.
    #[test]
    fn the_normal_tail_is_accurate_where_the_quadrature_uses_it() {
        const REFERENCE: [(f64, f64); 8] = [
            (0.5, 3.085_375_387_259_869e-1),
            (3.0, 1.349_898_031_630_095_2e-3),
            (5.5, 1.898_956_246_588_772e-8),
            (6.0, 9.865_876_450_377_014e-10),
            (8.0, 6.220_960_574_271_819e-16),
            (12.0, 1.776_482_112_077_702e-33),
            (20.0, 2.753_624_118_606_331e-89),
            (26.0, 2.476_063_315_503_457e-149),
        ];
        for (point, truth) in REFERENCE {
            let value = normal_sf(point).expect("tail");
            assert!(
                ((value - truth) / truth).abs() < 1.0e-13,
                "at {point}: {value} against {truth}"
            );
        }
    }

    /// A rectangle at a correlation of nearly minus one, where both
    /// coordinates are asked to be large and they can barely both be. The
    /// exponent runs to a million, so the integrand carries a millionfold more
    /// rounding than an ordinary one; asking it for a fixed 1e-13 made the
    /// quadrature subdivide to its depth limit and refuse an answer that is
    /// perfectly well defined.
    #[test]
    fn a_nearly_opposed_rectangle_is_answered_to_the_precision_available() {
        // Checked against a sixty-digit evaluation of the same integral.
        let truth = -1_000_022.907_899_968_9;
        let value =
            log_bivariate_rectangle(&[1.0, 1.0], &[f64::INFINITY, f64::INFINITY], -0.999_999)
                .expect("rectangle");
        // An epsilon of the exponent is the best available, and that is 2e-10
        // of a million.
        assert!((value - truth).abs() < 1.0e-3, "{value} against {truth}");
        // It must also sit under the bound from `X + Y`, which is tight here.
        let bound = log_bivariate_geometry_upper_bound(
            &[1.0, 1.0],
            &[f64::INFINITY, f64::INFINITY],
            -0.999_999,
        )
        .expect("bound");
        assert!(value <= bound, "{value} is above the bound {bound}");
    }

    /// The inverse of the log upper tail must undo it at any depth, including
    /// where the ordinary scale has long since underflowed.
    #[test]
    fn the_inverse_log_tail_undoes_it_far_beyond_the_ordinary_scale() {
        for point in [0.0_f64, 1.0, 37.295_08, 316.206_656, 14_142.134_883] {
            let target = log_normal_sf(point).expect("tail");
            let recovered = inverse_log_normal_sf(target).expect("inverse");
            assert!(
                (recovered - point).abs() <= 1.0e-9 * point.max(1.0),
                "{point} came back as {recovered}"
            );
        }
    }

    /// The inverse of the log upper tail must undo it, wherever the tail is.
    #[test]
    fn the_inverse_log_tail_undoes_the_log_tail() {
        for point in [0.0_f64, 0.5, 1.0, 3.0, 6.0, 12.0, 40.0, 150.0, 800.0] {
            let target = log_normal_sf(point).expect("tail");
            let recovered = inverse_log_normal_sf(target).expect("inverse");
            assert!(
                (recovered - point).abs() <= 1.0e-12 * point.max(1.0),
                "{point} came back as {recovered} through {target}"
            );
        }
    }

    /// Placing a quantile inside an interval that sits far out in a tail: the
    /// share of the interval below the returned point must be the share asked
    /// for. Taken on the ordinary scale this had a floor at 1e-308 and the
    /// evaluation was refused instead.
    #[test]
    fn a_truncated_draw_lands_at_the_right_share_of_a_distant_interval() {
        for (lower, upper) in [
            (0.5_f64, 2.0_f64),
            (6.0, 9.0),
            (40.0, 45.0),
            (60.0, f64::INFINITY),
            (-45.0, -40.0),
        ] {
            for unit in [0.05_f64, 0.25, 0.5, 0.75, 0.95] {
                let point = truncated_standard_normal(lower, upper, unit).expect("draw");
                assert!(
                    lower <= point && point <= upper,
                    "{point} outside {lower}..{upper}"
                );
                let whole = log_interval_probability(lower, upper).expect("whole");
                let below = log_interval_probability(lower, point).expect("below");
                let share = (below - whole).exp();
                assert!(
                    (share - unit).abs() < 1.0e-9,
                    "{lower}..{upper} at {unit}: point {point} takes {share}"
                );
            }
        }
    }

    /// With independent coordinates the rectangle is the product of its
    /// marginals, so the quasi-Monte Carlo path can be held to an exact
    /// answer -- including where that answer is far below anything the
    /// ordinary scale carries.
    #[test]
    fn the_qmc_path_reaches_the_deep_tail() {
        for thresholds in [
            [1.0_f64, 1.5, 2.0, 2.5],
            [8.0, 9.0, 10.0, 11.0],
            [40.0, 45.0, 50.0, 55.0],
        ] {
            let dimension = thresholds.len();
            let lower: Vec<f64> = thresholds.to_vec();
            let upper = vec![f64::INFINITY; dimension];
            let mean = DVector::zeros(dimension);
            let covariance = DMatrix::identity(dimension, dimension);
            let truth: f64 = thresholds
                .iter()
                .map(|&threshold| log_normal_sf(threshold).expect("tail"))
                .sum();
            let estimate = rectangle_probability(&lower, &upper, &mean, &covariance, 8_192)
                .expect("rectangle");
            assert!(
                (estimate.log_probability - truth).abs() < 1.0e-3,
                "{thresholds:?}: {} against {truth}",
                estimate.log_probability
            );
        }
    }

    /// A growing region cannot hold less probability, however far out its
    /// edge is pushed. The conditional integrand used to be searched over a
    /// window fixed relative to the rectangle, so once the edge passed sixty
    /// the window sat above the mode: the answer fell by 1805 nats between
    /// `t = 40` and `t = 120` and the quadrature failed outright past 300.
    #[test]
    fn distant_rectangle_edges_do_not_lose_probability() {
        // P(M <= t, Y > 7) for independent standard normals, which is
        // Phi(t)(1 - Phi(7)) and settles at log 1.2798e-12 once t is past a
        // few deviations. Checked against a sixty-digit evaluation.
        let truth = -27.384_307_498_811_08;
        for t in [10.0, 40.0, 61.0, 120.0, 300.0, 3000.0, 10_000.0] {
            let value =
                log_bivariate_rectangle(&[f64::NEG_INFINITY, 7.0], &[t, f64::INFINITY], 0.0)
                    .expect("rectangle");
            assert!(
                (value - truth).abs() < 1.0e-12,
                "t = {t}: {value} against {truth}"
            );
        }
    }

    /// A rectangle unbounded in one coordinate is that coordinate's marginal,
    /// and stays so however deep the other one goes and whatever the
    /// correlation. The peak of the conditional integrand sits near
    /// `correlation * boundary` here, so a search window anchored on the
    /// rectangle misses it entirely.
    #[test]
    fn unbounded_coordinate_recovers_the_marginal_in_the_deep_tail() {
        // log Phi(-200), to sixty digits.
        let truth = -20_006.217_280_898_19;
        for correlation in [0.0, 0.5, -0.5, 0.9, -0.99] {
            let value = log_bivariate_rectangle(
                &[f64::NEG_INFINITY, 200.0],
                &[f64::INFINITY, f64::INFINITY],
                correlation,
            )
            .expect("rectangle");
            assert!(
                (value - truth).abs() < 1.0e-9,
                "correlation = {correlation}: {value} against {truth}"
            );
        }
    }

    /// Splitting a rectangle in two and adding the halves must return the
    /// whole, at any depth and any correlation. This holds the quadrature to
    /// account where no reference distribution reaches.
    #[test]
    fn deep_tail_rectangles_add_up() {
        for correlation in [0.0, 0.7, -0.4] {
            for (start, split, finish) in [(6.0, 9.0, 40.0), (30.0, 45.0, 90.0)] {
                let whole =
                    log_bivariate_rectangle(&[start, 5.0], &[finish, f64::INFINITY], correlation)
                        .expect("whole");
                let left =
                    log_bivariate_rectangle(&[start, 5.0], &[split, f64::INFINITY], correlation)
                        .expect("left");
                let right =
                    log_bivariate_rectangle(&[split, 5.0], &[finish, f64::INFINITY], correlation)
                        .expect("right");
                let larger = left.max(right);
                let combined = larger + ((left - larger).exp() + (right - larger).exp()).ln();
                assert!(
                    (whole - combined).abs() < 1.0e-9,
                    "correlation {correlation}, {start}..{split}..{finish}: \
                     {whole} against {combined}"
                );
            }
        }
    }

    use super::{
        FIT_GRADIENT_TOLERANCE, LatentMediationFamilyInput, LatentMediationModel,
        LatentMediationParameters, adaptive_simpson, bivariate_normal_cdf, bound_aware_gradient,
        directional_covariance, scaled_projected_gradient, stable_bound_aware_gradient,
        transformed_bounds_holding, transformed_parameters,
    };
    use nalgebra::DMatrix;

    fn singleton(mediator_measurement: Option<f64>) -> LatentMediationFamilyInput {
        LatentMediationFamilyInput {
            relationship: vec![vec![1.0]],
            latent_mean: vec![0.0, 0.0],
            mediator_design: Vec::new(),
            outcome_design: Vec::new(),
            mediator_measurement: vec![mediator_measurement],
            mediator_measurement_error_variance: vec![mediator_measurement.map(|_| 0.15)],
            mediator_proxy_status: vec![None],
            outcome_status: vec![None],
            mediator_threshold: vec![0.0],
            outcome_threshold: vec![0.4],
            mediator_proxy_sensitivity: vec![0.8],
            mediator_proxy_specificity: vec![0.85],
            ascertainment: "population_unconditioned".to_owned(),
            proband_index: None,
        }
    }

    fn dyad() -> LatentMediationFamilyInput {
        LatentMediationFamilyInput {
            relationship: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            latent_mean: vec![0.0; 4],
            mediator_design: Vec::new(),
            outcome_design: Vec::new(),
            mediator_measurement: vec![None; 2],
            mediator_measurement_error_variance: vec![None; 2],
            mediator_proxy_status: vec![None; 2],
            outcome_status: vec![None; 2],
            mediator_threshold: vec![0.0; 2],
            outcome_threshold: vec![0.4; 2],
            mediator_proxy_sensitivity: vec![0.8; 2],
            mediator_proxy_specificity: vec![0.85; 2],
            ascertainment: "population_unconditioned".to_owned(),
            proband_index: None,
        }
    }

    fn deterministic_fit_model_at_mediator_scale(unit_scale: f64) -> LatentMediationModel {
        const OBSERVATIONS: [([f64; 2], i8); 10] = [
            ([0.079_049_024_394_060_85, -0.441_168_435_356_566_97], 0),
            ([0.384_264_427_030_968_64, 0.427_824_002_685_406_66], 1),
            ([-0.288_858_813_914_036, 0.467_645_507_875_167_8], 0),
            ([1.505_519_311_654_784_5, 0.349_268_425_359_107_85], 1),
            ([0.856_146_192_031_015_6, 1.772_879_075_197_205], 0),
            ([-0.045_028_307_478_389_76, -1.420_541_068_423_922], 0),
            ([-1.561_815_005_120_057_9, -0.911_954_515_432_920_9], 0),
            ([1.089_970_373_527_701_3, -1.415_961_344_561_931_7], 1),
            ([-0.281_806_711_439_217_37, 0.311_019_458_345_349_83], 0),
            ([1.882_544_097_959_577_8, 0.582_329_545_538_015], 1),
        ];
        let families = OBSERVATIONS
            .into_iter()
            .map(
                |(mediator_measurement, status)| LatentMediationFamilyInput {
                    relationship: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
                    latent_mean: vec![0.0; 4],
                    mediator_design: Vec::new(),
                    outcome_design: Vec::new(),
                    mediator_measurement: mediator_measurement
                        .into_iter()
                        .map(|value| Some(unit_scale * value))
                        .collect(),
                    mediator_measurement_error_variance: vec![
                        Some(unit_scale * unit_scale * 0.18);
                        2
                    ],
                    mediator_proxy_status: vec![None; 2],
                    outcome_status: vec![Some(status), None],
                    mediator_threshold: vec![0.0 * unit_scale; 2],
                    outcome_threshold: vec![0.25; 2],
                    mediator_proxy_sensitivity: vec![0.8; 2],
                    mediator_proxy_specificity: vec![0.85; 2],
                    ascertainment: "population_unconditioned".to_owned(),
                    proband_index: None,
                },
            )
            .collect();
        LatentMediationModel::build(families, 256).expect("valid fit model")
    }

    /// A model whose outcome loading `d` does not rest on its bound, which the
    /// vertical test needs: the even mixture it reads the loading against
    /// assumes `a` is the only parameter on a boundary.
    ///
    /// **Sibling pairs, not lone people.** Built from unrelated singletons the
    /// model is not identified at all: a single person's relationship matrix is
    /// `[1]`, so the inherited part and the residual part land in the same
    /// entry of the same covariance and `a^2` is indistinguishable from a share
    /// of `sigma_m2`. Five parameters mapped into a 2 x 2 marginal covariance,
    /// and the three tests standing on that fixture were reading whatever the
    /// optimiser happened to settle on along a ridge. A relatedness of a half
    /// puts the inherited part in the between-person block where the residual
    /// never reaches, which is what separates them.
    ///
    /// Simulated from `a = 0.6`, `b = 0.4`, `c' = 0.2`, `d = 0.7` and
    /// `sigma_m2 = 0.5`, with a measurement error variance of 0.15 and an
    /// outcome threshold at nought. Thirty pairs is far too few to recover
    /// those, and the fit does not: what it does is rest nowhere near a bound,
    /// which is the property these tests need.
    fn interior_fit_model() -> LatentMediationModel {
        const PAIRS: [(f64, f64, i8, i8); 30] = [
            (-0.082_827_841_380_880_12, 0.268_028_928_260_397_26, 1, 0),
            (0.253_220_824_736_949_85, -0.189_648_274_707_171_13, 1, 1),
            (-0.140_721_226_026_780_67, -0.902_212_504_627_923_4, 1, 1),
            (0.331_392_185_287_071, 0.010_349_386_355_413_492, 0, 1),
            (-1.048_363_528_916_788, -0.348_536_583_261_377_9, 0, 0),
            (0.504_766_571_125_986_7, -1.247_034_810_954_168_2, 1, 0),
            (-1.106_248_747_183_598_7, -0.268_474_133_805_127_9, 0, 0),
            (-1.429_958_150_506_903_2, 1.306_075_004_603_112_4, 0, 1),
            (1.500_804_987_201_952_4, 1.225_282_364_980_799, 1, 0),
            (-0.304_884_248_038_848_95, 0.066_847_049_742_911_89, 1, 1),
            (0.168_520_136_597_886_87, 0.589_375_455_081_163_6, 1, 1),
            (1.265_156_725_120_963_9, 1.147_046_845_670_021_4, 0, 0),
            (-0.643_774_734_819_989_1, 1.380_271_208_187_542_8, 1, 1),
            (1.192_655_970_969_962_5, 1.512_584_940_887_036_2, 1, 0),
            (-1.646_448_006_768_493_2, 0.265_890_321_587_913_6, 0, 0),
            (2.786_297_905_420_202, 0.909_843_862_509_991_4, 1, 1),
            (-1.695_598_553_737_680_8, -0.386_546_812_523_611_95, 0, 0),
            (-0.123_208_817_430_467_85, 1.418_617_520_782_310_8, 0, 1),
            (-0.912_170_811_242_995_1, -0.444_806_358_186_866_3, 1, 0),
            (0.470_716_544_407_430_44, -0.551_329_088_161_151_9, 0, 0),
            (0.102_381_376_191_836_14, -0.354_060_624_891_068_2, 1, 0),
            (-1.508_547_774_847_911_5, 1.243_542_598_903_108_4, 1, 1),
            (-1.782_093_205_086_311_2, -0.595_091_914_538_564_7, 0, 0),
            (0.767_929_962_652_682_5, -0.578_653_169_129_678_2, 1, 0),
            (-0.710_523_467_746_605_2, -0.339_848_670_736_170_5, 1, 1),
            (1.405_273_728_715_498_2, 0.899_710_052_367_798_9, 0, 0),
            (0.231_919_341_295_305_03, -0.379_104_583_023_983_03, 1, 0),
            (0.164_614_119_330_150_33, 0.325_013_731_653_433_17, 1, 1),
            (-1.152_720_179_658_223_8, -0.338_332_152_453_285_4, 1, 1),
            (-1.155_030_771_797_046_5, 0.178_436_617_025_690_25, 1, 0),
        ];
        let families = PAIRS
            .into_iter()
            .map(
                |(first, second, first_status, second_status)| LatentMediationFamilyInput {
                    relationship: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
                    latent_mean: vec![0.0; 4],
                    mediator_design: Vec::new(),
                    outcome_design: Vec::new(),
                    mediator_measurement: vec![Some(first), Some(second)],
                    mediator_measurement_error_variance: vec![Some(0.15), Some(0.15)],
                    mediator_proxy_status: vec![None, None],
                    outcome_status: vec![Some(first_status), Some(second_status)],
                    mediator_threshold: vec![0.0, 0.0],
                    outcome_threshold: vec![0.0, 0.0],
                    mediator_proxy_sensitivity: vec![0.8, 0.8],
                    mediator_proxy_specificity: vec![0.85, 0.85],
                    ascertainment: "population_unconditioned".to_owned(),
                    proband_index: None,
                },
            )
            .collect();
        LatentMediationModel::build(families, 256).expect("valid fit model")
    }

    fn deterministic_fit_model() -> LatentMediationModel {
        deterministic_fit_model_at_mediator_scale(1.0)
    }

    #[test]
    fn extreme_continuous_observation_remains_finite_on_log_scale() {
        let model =
            LatentMediationModel::build(vec![singleton(Some(100.0))], 8192).expect("valid model");
        let record = model
            .evaluate(LatentMediationParameters {
                a: 0.5,
                b: 0.3,
                c_prime: 0.2,
                d: 0.6,
                sigma_m2: 0.7,
            })
            .expect("finite log likelihood");
        assert!((record.log_likelihood - (-4_546.421_139_077_653)).abs() < 1.0e-9);
        assert!(!record.ordinary_scale_representable);
        assert_eq!(record.integration_methods, ["no_discrete_observation"]);
    }

    #[test]
    fn directional_covariance_uses_process_major_order() {
        let relationship = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let covariance = directional_covariance(
            &relationship,
            LatentMediationParameters {
                a: 0.5,
                b: 0.3,
                c_prime: 0.2,
                d: 0.6,
                sigma_m2: 0.7,
            },
        )
        .expect("positive definite");
        // M_1,M_2 is the first block; Y_1,Y_2 is the second.
        assert!((covariance[(0, 1)] - 0.125).abs() < 1.0e-12);
        assert!((covariance[(0, 2)] - 0.385).abs() < 1.0e-12);
    }

    #[test]
    fn bivariate_zero_threshold_has_known_closed_form() {
        let rho: f64 = 0.5;
        let expected = 0.25 + rho.asin() / (2.0 * std::f64::consts::PI);
        let observed = bivariate_normal_cdf(0.0, 0.0, rho).expect("valid probability");
        assert!((observed - expected).abs() < 1.0e-10);
    }

    #[test]
    fn near_singular_opposed_tail_is_refused_instead_of_returning_residue() {
        let error = bivariate_normal_cdf(-5.5, 5.0, -0.999_999_998_999)
            .expect_err("unbounded cancellation must fail closed");
        assert_eq!(error, "LATENT_MEDIATION_BIVARIATE_PROBABILITY_UNRESOLVED");
        let neighboring_error = bivariate_normal_cdf(-5.5, 5.0, -0.999_999_98)
            .expect_err("the same impossible geometry remains unresolved");
        assert_eq!(
            neighboring_error,
            "LATENT_MEDIATION_BIVARIATE_PROBABILITY_UNRESOLVED"
        );

        let correlation: f64 = -0.999_999_998_999;
        let expected = 0.25 + correlation.asin() / (2.0 * std::f64::consts::PI);
        let feasible = bivariate_normal_cdf(0.0, 0.0, correlation)
            .expect("near-singular overlap remains feasible");
        assert!((feasible - expected).abs() < 1.0e-10);
    }

    #[test]
    fn adaptive_simpson_refuses_depth_exhaustion() {
        let error = adaptive_simpson(&|value| value.powi(4), 0.0, 1.0, 1.0e-16, 0.0, 0)
            .expect_err("an exhausted error budget must not be returned as success");
        assert_eq!(
            error,
            "LATENT_MEDIATION_BIVARIATE_QUADRATURE_DID_NOT_CONVERGE"
        );
    }

    #[test]
    fn fixed_point_rejects_non_diagonal_one_relationship() {
        let mut family = singleton(None);
        family.relationship[0][0] = 2.0;
        let error = LatentMediationModel::build(vec![family], 8192)
            .expect_err("wrong scale must be refused");
        assert_eq!(error, "LATENT_MEDIATION_RELATIONSHIP_DIAGONAL_NOT_ONE");
    }

    #[test]
    fn relationship_uses_exact_symmetry_and_the_documented_psd_floor() {
        let mut asymmetric = dyad();
        asymmetric.relationship[1][0] += 5.0e-11;
        assert_eq!(
            LatentMediationModel::build(vec![asymmetric], 8192)
                .expect_err("any asymmetric pair must be refused"),
            "LATENT_MEDIATION_RELATIONSHIP_NOT_SYMMETRIC"
        );

        let mut inside_floor = dyad();
        inside_floor.relationship = vec![vec![1.0, 1.000_000_000_5], vec![1.000_000_000_5, 1.0]];
        LatentMediationModel::build(vec![inside_floor], 8192)
            .expect("an eigenvalue above the documented -1e-9 floor is accepted");

        let mut below_floor = dyad();
        below_floor.relationship = vec![vec![1.0, 1.000_000_002], vec![1.000_000_002, 1.0]];
        assert_eq!(
            LatentMediationModel::build(vec![below_floor], 8192)
                .expect_err("an eigenvalue below the documented floor is refused"),
            "LATENT_MEDIATION_RELATIONSHIP_NOT_POSITIVE_SEMIDEFINITE"
        );
    }

    #[test]
    fn preparation_retains_observation_selections() {
        let mut family = dyad();
        family.mediator_measurement[0] = Some(1.25);
        family.mediator_measurement_error_variance[0] = Some(0.2);
        family.mediator_proxy_status[1] = Some(1);
        family.outcome_status[0] = Some(0);
        let model = LatentMediationModel::build(vec![family], 8192).expect("prepared dyad");
        let prepared = &model.families[0];

        assert_eq!(prepared.observed_mediator_measurement_indices, [0]);
        assert_eq!(prepared.observed_mediator_measurement_values, [1.25]);
        assert_eq!(
            prepared.observed_mediator_measurement_error_variances,
            [0.2]
        );
        assert_eq!(prepared.observed_mediator_proxy, [(1, 1)]);
        assert_eq!(prepared.observed_outcome, [(0, 0)]);
        assert_eq!(prepared.discrete_target_indices, [1, 2]);
    }

    #[test]
    fn qmc_work_is_configurable_inside_the_prepared_model() {
        let model = LatentMediationModel::build(vec![singleton(None)], 256)
            .expect("valid minimum QMC work");
        assert_eq!(model.qmc_points, 256);
        assert_eq!(
            LatentMediationModel::build(vec![singleton(None)], 255)
                .expect_err("too few QMC points must be refused"),
            "LATENT_MEDIATION_QMC_POINTS_INVALID"
        );
    }

    #[test]
    fn transformed_mapping_preserves_the_vertical_product_and_variance_rules() {
        let transformed = [0.5, 1.0, -0.2, 0.36, 0.5_f64.ln()];
        let parameters = transformed_parameters(&transformed, 4.0).expect("valid transform");
        assert!((parameters.a - 1.0).abs() < 1.0e-12);
        assert!((parameters.b - 0.5).abs() < 1.0e-12);
        assert!((parameters.c_prime + 0.2).abs() < 1.0e-12);
        assert!((parameters.d - 0.6).abs() < 1.0e-12);
        assert!((parameters.sigma_m2 - 2.0).abs() < 1.0e-12);
        assert!((parameters.a * parameters.b - transformed[0] * transformed[1]).abs() < 1.0e-12);
    }

    #[test]
    fn finite_difference_and_projected_kkt_respect_a_lower_bound() {
        let point = [0.0, 2.0, -1.0, 0.25, 0.4];
        let bounds = transformed_bounds_holding(&[], 0).expect("bounds");
        let objective = |candidate: &[f64]| {
            Some(
                (candidate[0] + 1.0).powi(2)
                    + (candidate[1] - 2.0).powi(2)
                    + (candidate[2] + 1.0).powi(2)
                    + (candidate[3] - 0.25).powi(2)
                    + (candidate[4] - 0.4).powi(2),
            )
        };
        let gradient =
            bound_aware_gradient(&point, &bounds.lower, &bounds.upper, 1.0e-5, &objective)
                .expect("one-sided stencil at lower bound");
        assert!((gradient[0] - 2.0).abs() < 1.0e-8);
        let stable = stable_bound_aware_gradient(&point, &bounds.lower, &bounds.upper, &objective)
            .expect("stable gradient");
        let norm = scaled_projected_gradient(
            &point,
            &stable,
            &bounds.lower,
            &bounds.upper,
            objective(&point).expect("objective"),
        );
        assert!(norm < FIT_GRADIENT_TOLERANCE);
    }

    #[test]
    fn optimisation_objective_and_convergence_metric_ignore_mediator_units() {
        let base = deterministic_fit_model_at_mediator_scale(1.0);
        let rescaled = deterministic_fit_model_at_mediator_scale(1.0e6);
        let point = [0.2_f64.sqrt(), 0.5, 0.5, 0.5, 0.8_f64.ln()];
        let bounds = transformed_bounds_holding(&[], 0).expect("bounds");
        let base_scale = base.mediator_scale_squared().expect("base mediator scale");
        let rescaled_scale = rescaled
            .mediator_scale_squared()
            .expect("rescaled mediator scale");
        let base_objective = |candidate: &[f64]| base.optimisation_objective(candidate, base_scale);
        let rescaled_objective =
            |candidate: &[f64]| rescaled.optimisation_objective(candidate, rescaled_scale);

        let base_value = base_objective(&point).expect("base objective");
        let rescaled_value = rescaled_objective(&point).expect("rescaled objective");
        assert!((base_value - rescaled_value).abs() < 1.0e-10);

        let base_gradient =
            stable_bound_aware_gradient(&point, &bounds.lower, &bounds.upper, &base_objective)
                .expect("base stable gradient");
        let rescaled_gradient =
            stable_bound_aware_gradient(&point, &bounds.lower, &bounds.upper, &rescaled_objective)
                .expect("rescaled stable gradient");
        let base_scaled = scaled_projected_gradient(
            &point,
            &base_gradient,
            &bounds.lower,
            &bounds.upper,
            base_value,
        );
        let rescaled_scaled = scaled_projected_gradient(
            &point,
            &rescaled_gradient,
            &bounds.lower,
            &bounds.upper,
            rescaled_value,
        );
        assert!((base_scaled - rescaled_scaled).abs() < 1.0e-8);

        let base_loglik = base
            .evaluate_transformed(&point, base_scale)
            .expect("base raw likelihood")
            .log_likelihood;
        let rescaled_loglik = rescaled
            .evaluate_transformed(&point, rescaled_scale)
            .expect("rescaled raw likelihood")
            .log_likelihood;
        assert!((rescaled_loglik - base_loglik + 20.0 * 1.0e6_f64.ln()).abs() < 1.0e-9);
    }

    /// Holding a coordinate must cost likelihood, never gain it: the held
    /// model is nested inside the free one.
    #[test]
    fn a_held_fit_cannot_beat_the_free_one() {
        let model = interior_fit_model();
        let free = model.fit().expect("free fit");
        for held in [0usize, 1] {
            let constrained = model.fit_holding(&[held]).expect("held fit");
            assert!(
                constrained.log_likelihood <= free.log_likelihood + 1e-6,
                "holding {held} beat the free fit: {} against {}",
                constrained.log_likelihood,
                free.log_likelihood
            );
        }
    }

    /// A held coordinate must actually arrive at nought, and the estimand it
    /// controls with it.
    #[test]
    fn holding_a_coordinate_puts_it_at_nought() {
        let model = interior_fit_model();
        let without_loading = model.fit_holding(&[0]).expect("held fit");
        assert!(without_loading.parameters.a.abs() < 1e-12);
        assert!((without_loading.parameters.a * without_loading.parameters.b).abs() < 1e-12);
        let without_path = model.fit_holding(&[1]).expect("held fit");
        assert!(without_path.parameters.b.abs() < 1e-12);
    }

    /// The union null is rejected only when both parts are, so the p-value is
    /// the larger of the two and each part keeps its own reference.
    /// Where the outcome loading also rests on its bound the free fit sits on
    /// the corner of the cone rather than a face, the even mixture is the wrong
    /// reference for the loading, and that wrong p-value would be the reported
    /// one whenever it binds. The standing fixture is exactly such a fit, which
    /// is the point: this is an ordinary case and not a corner.
    #[test]
    fn a_second_loading_on_its_bound_is_refused_rather_than_answered() {
        let model = deterministic_fit_model();
        assert!(
            model
                .fit()
                .expect("fits")
                .boundary_parameters
                .contains(&"d")
        );
        assert_eq!(
            model.test_vertical().err(),
            Some("LATENT_MEDIATION_ANOTHER_LOADING_AT_ZERO")
        );
    }

    #[test]
    fn the_vertical_test_takes_the_larger_part() {
        let model = interior_fit_model();
        assert!(model.fit().expect("fits").boundary_parameters.is_empty());
        let test = model.test_vertical().expect("tests");
        assert!((0.0..=1.0).contains(&test.p_value));
        assert!((test.p_value - test.loading_p_value.max(test.path_p_value)).abs() < 1e-15);
        assert!(test.p_value >= test.loading_p_value - 1e-15);
        assert!(test.p_value >= test.path_p_value - 1e-15);
        assert!(test.loading_statistic >= 0.0 && test.path_statistic >= 0.0);
    }

    /// A coordinate that does not exist cannot be held.
    #[test]
    fn an_impossible_coordinate_is_refused() {
        let model = deterministic_fit_model();
        assert_eq!(
            model.fit_holding(&[9]).err(),
            Some("LATENT_MEDIATION_HELD_COORDINATE_INVALID")
        );
    }

    #[test]
    fn latent_mediation_fit_is_deterministic_at_the_converged_solution() {
        let model = deterministic_fit_model();
        let first = model.fit().expect("first fit");
        let second = model.fit().expect("second fit");
        assert_eq!(
            first.log_likelihood.to_bits(),
            second.log_likelihood.to_bits()
        );
        assert_eq!(first.parameters.a.to_bits(), second.parameters.a.to_bits());
        assert_eq!(first.parameters.b.to_bits(), second.parameters.b.to_bits());
        assert_eq!(
            first.parameters.c_prime.to_bits(),
            second.parameters.c_prime.to_bits()
        );
        assert_eq!(first.parameters.d.to_bits(), second.parameters.d.to_bits());
        assert_eq!(
            first.parameters.sigma_m2.to_bits(),
            second.parameters.sigma_m2.to_bits()
        );
        assert!(first.scaled_gradient < FIT_GRADIENT_TOLERANCE);
    }
}
