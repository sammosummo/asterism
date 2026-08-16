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
    pub maximum_qmc_batch_range: f64,
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
    pub maximum_qmc_batch_range: f64,
    pub families: Vec<LatentMediationFamilyEvaluation>,
}

type TransformedPoint = [f64; FIT_DIMENSION];

/// A converged maximum-likelihood fit of the latent mediation model.
#[derive(Clone, Debug)]
pub struct LatentMediationFit {
    pub parameters: LatentMediationParameters,
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

#[derive(Clone, Copy, Debug)]
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
    fn value_and_gradient(self, point: &[f64]) -> (f64, Vec<f64>) {
        let mut distance = 1.0;
        let mut gradient = Vec::with_capacity(FIT_DIMENSION);
        for (value, anchor) in point.iter().zip(self.anchor) {
            let delta = *value - anchor;
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
        validate_parameters(parameters)?;
        let mut total = 0.0;
        let mut ordinary_scale_representable = true;
        let mut methods = BTreeSet::new();
        let mut maximum_qmc_batch_range: f64 = 0.0;
        let mut family_records = Vec::with_capacity(self.families.len());
        for family in &self.families {
            let covariance = directional_covariance(&family.relationship, parameters)?;
            let record = family.evaluate(&covariance, self.qmc_points)?;
            total += record.log_likelihood;
            ordinary_scale_representable &= record.ordinary_scale_representable;
            maximum_qmc_batch_range = maximum_qmc_batch_range.max(record.maximum_qmc_batch_range);
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
            maximum_qmc_batch_range,
            families: family_records,
        })
    }

    /// Fit all five structural parameters by maximum likelihood using the
    /// fixed, mediator-scale-invariant recipe.
    pub fn fit(&self) -> Result<LatentMediationFit, &'static str> {
        let mediator_scale_squared = self.mediator_scale_squared()?;
        let bounds = transformed_bounds()?;
        let mut best_converged: Option<FitCandidate> = None;
        let mut best_unresolved: Option<f64> = None;
        // The most recent underlying failure, so a fit in which nothing
        // converged reports its cause rather than only the fact.
        let mut last_error: Option<&'static str> = None;

        for start in deterministic_starts() {
            let Some(initial_objective) =
                self.optimisation_objective(&start, mediator_scale_squared)
            else {
                continue;
            };
            let penalty = InwardPenalty::new(start, initial_objective);
            let value_of = |candidate: &[f64]| {
                solver_value_and_gradient(self, candidate, mediator_scale_squared, &bounds, penalty)
                    .0
            };
            let gradient_of = |candidate: &[f64]| {
                solver_value_and_gradient(self, candidate, mediator_scale_squared, &bounds, penalty)
                    .1
            };
            let mut control = OptimControl::default_for_dimension(FIT_DIMENSION);
            control.maxit = 500;
            // R's own default: stop once the objective settles to about 1e-9
            // relative rather than running every start to `maxit`.
            control.factr = 1.0e3;
            control.pgtol = FIT_SOLVER_PGTOL;
            control.lmm = FIT_DIMENSION;
            control.fnscale = initial_objective.abs().max(1.0);
            control.parscale = vec![1.0; FIT_DIMENSION];

            let Ok(solution) = optim_lbfgsb_with_gradient(
                start.to_vec(),
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
            let Ok(transformed) = <Vec<f64> as TryInto<TransformedPoint>>::try_into(solution.par)
            else {
                continue;
            };
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
        self.evaluate(transformed_parameters(transformed, mediator_scale_squared)?)
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

fn transformed_bounds() -> Result<Bounds, &'static str> {
    Bounds::new(
        vec![
            0.0,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            0.0,
            f64::NEG_INFINITY,
        ],
        vec![f64::INFINITY; FIT_DIMENSION],
    )
    .map_err(|_| "LATENT_MEDIATION_OPTIMISER_BOUNDS_INVALID")
}

fn deterministic_starts() -> [TransformedPoint; 5] {
    [
        [0.5_f64.sqrt(), 0.0, 0.0, 0.5, 0.5_f64.ln()],
        [0.2_f64.sqrt(), 0.5, 0.5, 0.5, 0.8_f64.ln()],
        [0.2_f64.sqrt(), -0.5, -0.5, 0.5, 0.8_f64.ln()],
        [0.8_f64.sqrt(), 0.5, -0.5, 0.2, 0.2_f64.ln()],
        [0.8_f64.sqrt(), -0.5, 0.5, 0.2, 0.2_f64.ln()],
    ]
}

fn transformed_parameters(
    transformed: &[f64],
    mediator_scale_squared: f64,
) -> Result<LatentMediationParameters, &'static str> {
    if transformed.len() != FIT_DIMENSION
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
    penalty: InwardPenalty,
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
    if point.len() != FIT_DIMENSION
        || lower.len() != FIT_DIMENSION
        || upper.len() != FIT_DIMENSION
        || !step_multiplier.is_finite()
        || step_multiplier <= 0.0
        || point.iter().any(|value| !value.is_finite())
        || (0..FIT_DIMENSION).any(|index| {
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
    let mut gradient = Vec::with_capacity(FIT_DIMENSION);
    for index in 0..FIT_DIMENSION {
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
        let projected = if (at_lower_bound(point[index], lower[index]) && component > 0.0)
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

        Ok(Self {
            relationship,
            latent_mean: DVector::from_vec(input.latent_mean),
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

    fn evaluate(
        &self,
        covariance: &DMatrix<f64>,
        qmc_points: usize,
    ) -> Result<LatentMediationFamilyEvaluation, &'static str> {
        let size = self.relationship.nrows();
        let conditional = condition_on_mediator_measurements(
            &self.discrete_target_indices,
            &self.observed_mediator_measurement_indices,
            &self.observed_mediator_measurement_values,
            &self.observed_mediator_measurement_error_variances,
            &self.latent_mean,
            covariance,
        )?;

        let truth_configurations = 1usize << self.observed_mediator_proxy.len();
        let mut discrete_probability = 0.0;
        let mut maximum_qmc_batch_range: f64 = 0.0;
        let mut methods = BTreeSet::new();
        for configuration in 0..truth_configurations {
            let mut measurement_weight = 1.0;
            let mut lower = Vec::with_capacity(self.discrete_target_indices.len());
            let mut upper = Vec::with_capacity(self.discrete_target_indices.len());
            for (position, &(index, observed)) in self.observed_mediator_proxy.iter().enumerate() {
                let truth = i8::from((configuration & (1usize << position)) != 0);
                measurement_weight *= match (truth, observed) {
                    (1, 1) => self.mediator_proxy_sensitivity[index],
                    (1, 0) => 1.0 - self.mediator_proxy_sensitivity[index],
                    (0, 1) => 1.0 - self.mediator_proxy_specificity[index],
                    (0, 0) => self.mediator_proxy_specificity[index],
                    _ => unreachable!("validated binary status"),
                };
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
                    probability: 1.0,
                    batch_range: 0.0,
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
            discrete_probability += measurement_weight * rectangle.probability;
            maximum_qmc_batch_range = maximum_qmc_batch_range.max(rectangle.batch_range);
            methods.insert(rectangle.method.to_owned());
        }
        if !(discrete_probability > 0.0 && discrete_probability.is_finite()) {
            return Err("LATENT_MEDIATION_DISCRETE_PROBABILITY_INVALID");
        }
        let log_discrete_probability = discrete_probability.ln();
        let log_numerator = conditional.log_continuous_density + log_discrete_probability;

        let (denominator, ascertainment_name) = match self.ascertainment {
            Ascertainment::PopulationUnconditioned => (1.0, "population_unconditioned"),
            Ascertainment::NamedProbandCase(proband) => {
                let latent_index = size + proband;
                let variance = covariance[(latent_index, latent_index)];
                if !(variance > 0.0 && variance.is_finite()) {
                    return Err("LATENT_MEDIATION_ASCERTAINMENT_VARIANCE_INVALID");
                }
                let standardised = (self.outcome_threshold[proband]
                    - self.latent_mean[latent_index])
                    / variance.sqrt();
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
        let ordinary_scale_representable = [continuous, numerator, likelihood]
            .iter()
            .all(|value| *value > 0.0 && value.is_finite());

        Ok(LatentMediationFamilyEvaluation {
            log_likelihood,
            log_continuous_density: conditional.log_continuous_density,
            log_discrete_probability,
            log_ascertainment_denominator: log_denominator,
            ordinary_scale_representable,
            integration_methods: methods.into_iter().collect(),
            maximum_qmc_batch_range,
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

fn status_bounds(status: i8, threshold: f64) -> (f64, f64) {
    if status == 0 {
        (f64::NEG_INFINITY, threshold)
    } else {
        (threshold, f64::INFINITY)
    }
}

struct RectangleProbability {
    probability: f64,
    batch_range: f64,
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
            probability: 0.0,
            batch_range: 0.0,
            method: "empty_rectangle",
        });
    }
    let decomposition = covariance
        .clone()
        .cholesky()
        .ok_or("LATENT_MEDIATION_RECTANGLE_COVARIANCE_NOT_POSITIVE_DEFINITE")?;
    if dimension == 1 {
        let sd = covariance[(0, 0)].sqrt();
        let probability =
            interval_probability((lower[0] - mean[0]) / sd, (upper[0] - mean[0]) / sd)?;
        return Ok(RectangleProbability {
            probability,
            batch_range: 0.0,
            method: "univariate_exact",
        });
    }
    if dimension == 2 {
        let rectangle = bivariate_rectangle(lower, upper, mean, covariance)?;
        return Ok(RectangleProbability {
            probability: rectangle.probability,
            batch_range: 0.0,
            method: rectangle.method,
        });
    }

    let factor = decomposition.l();
    // Eight deterministically shifted copies of the Halton sequence.  A single
    // low-discrepancy sequence carries one systematic bias, and contiguous
    // blocks of it share that bias, so their spread understates the error.
    // Shifting decorrelates the copies: their mean sheds most of the shared
    // bias and their spread is an honest stability statistic.
    let mut replicate_estimates = [0.0; QMC_REPLICATES];
    let per_replicate = qmc_points / QMC_REPLICATES;
    for (replicate, estimate) in replicate_estimates.iter_mut().enumerate() {
        let mut sum = 0.0;
        for sample_index in 0..per_replicate {
            let sequence_index = sample_index / 2 + 1;
            let antithetic = sample_index % 2 == 1;
            let mut latent = vec![0.0; dimension];
            let mut weight = 1.0;
            for row in 0..dimension {
                let preceding = (0..row)
                    .map(|column| factor[(row, column)] * latent[column])
                    .sum::<f64>();
                let standardised_lower = (lower[row] - mean[row] - preceding) / factor[(row, row)];
                let standardised_upper = (upper[row] - mean[row] - preceding) / factor[(row, row)];
                let width = interval_probability(standardised_lower, standardised_upper)?;
                weight *= width;
                if weight == 0.0 {
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
            sum += weight;
        }
        *estimate = sum / per_replicate as f64;
    }
    let estimate = replicate_estimates.iter().sum::<f64>() / QMC_REPLICATES as f64;
    Ok(RectangleProbability {
        probability: estimate.clamp(0.0, 1.0),
        batch_range: replicate_estimates
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - replicate_estimates.iter().copied().fold(f64::INFINITY, f64::min),
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
    if !value.is_finite() {
        return Err("LATENT_MEDIATION_NORMAL_VARIATE_NOT_FINITE");
    }
    Ok(normal()?.cdf(value))
}

fn normal_sf(value: f64) -> Result<f64, &'static str> {
    if !value.is_finite() {
        return Err("LATENT_MEDIATION_NORMAL_VARIATE_NOT_FINITE");
    }
    // Symmetry avoids cancellation in the upper tail.
    Ok(normal()?.cdf(-value))
}

fn truncated_standard_normal(lower: f64, upper: f64, unit: f64) -> Result<f64, &'static str> {
    if !(0.0 < unit && unit < 1.0) {
        return Err("LATENT_MEDIATION_QMC_COORDINATE_INVALID");
    }
    let width = interval_probability(lower, upper)?;
    if width <= 0.0 {
        return Err("LATENT_MEDIATION_TRUNCATION_INTERVAL_EMPTY");
    }
    let distribution = normal()?;
    if lower >= 0.0 {
        let lower_survival = normal_sf(lower)?;
        let upper_survival = if upper == f64::INFINITY {
            0.0
        } else {
            normal_sf(upper)?
        };
        let survival = (lower_survival - unit * (lower_survival - upper_survival))
            .clamp(1.0e-300, 1.0 - 1.0e-16);
        return Ok(-distribution.inverse_cdf(survival));
    }
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
    let geometry_bound =
        bivariate_geometry_upper_bound(&standardised_lower, &standardised_upper, correlation)?;
    if geometry_bound > BIVARIATE_TAIL_CROSSOVER {
        let cdf = |first: f64, second: f64| bivariate_normal_cdf(first, second, correlation);
        let corners = [
            cdf(standardised_upper[0], standardised_upper[1]),
            cdf(standardised_lower[0], standardised_upper[1]),
            cdf(standardised_upper[0], standardised_lower[1]),
            cdf(standardised_lower[0], standardised_lower[1]),
        ];
        if let [Ok(both_upper), Ok(low_first), Ok(low_second), Ok(both_lower)] = corners {
            let corner_values = [both_upper, low_first, low_second, both_lower];
            let probability =
                corner_values[0] - corner_values[1] - corner_values[2] + corner_values[3];
            let rounding_budget = 16.0
                * f64::EPSILON
                * corner_values.iter().map(|value| value.abs()).sum::<f64>();
            let absolute_budget = 4.0 * BIVARIATE_ABSOLUTE_TOLERANCE + rounding_budget;
            if probability > absolute_budget {
                if probability > geometry_bound + absolute_budget {
                    return Err("LATENT_MEDIATION_BIVARIATE_PROBABILITY_OUTSIDE_BOUNDS");
                }
                return Ok(BivariateRectangle {
                    probability: probability.clamp(0.0, 1.0),
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
    if log_probability > (geometry_bound.ln() + 1.0e-8).max(f64::MIN) && geometry_bound > 0.0 {
        return Err("LATENT_MEDIATION_BIVARIATE_PROBABILITY_OUTSIDE_BOUNDS");
    }
    Ok(BivariateRectangle {
        probability: log_probability.exp().clamp(0.0, 1.0),
        method: "bivariate_tail_quadrature",
    })
}

struct BivariateRectangle {
    probability: f64,
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
    let extremity = |low: f64, high: f64| low.abs().min(high.abs());
    let swap = extremity(lower[1], upper[1]) > extremity(lower[0], upper[0]);
    let (outer, inner) = if swap {
        (([lower[1], upper[1]]), ([lower[0], upper[0]]))
    } else {
        (([lower[0], upper[0]]), ([lower[1], upper[1]]))
    };
    let conditional_sd = (1.0 - correlation * correlation).sqrt();
    if !(conditional_sd > 0.0) {
        return Err("LATENT_MEDIATION_BIVARIATE_CORRELATION_INVALID");
    }
    let log_integrand = |x: f64| -> Result<f64, &'static str> {
        let shifted_lower = (inner[0] - correlation * x) / conditional_sd;
        let shifted_upper = (inner[1] - correlation * x) / conditional_sd;
        Ok(-0.5 * x * x - 0.5 * LOG_TWO_PI + log_interval_probability(shifted_lower, shifted_upper)?)
    };
    // The integrand is log-concave, so a golden-section search finds its one
    // peak; the working range clips infinite limits far beyond any mass.
    let left_limit = outer[0].max(-60.0);
    let right_limit = outer[1].min(60.0);
    if left_limit >= right_limit {
        return Ok(f64::NEG_INFINITY);
    }
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut low = left_limit;
    let mut high = right_limit;
    for _ in 0..120 {
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
    // Bracket where the integrand has fallen eighty logs below its peak;
    // beyond that the contribution is immaterial at any tolerance used here.
    let drop = 80.0;
    let bracket = |mut inside: f64, mut outside: f64| -> Result<f64, &'static str> {
        if log_integrand(outside)? >= log_peak - drop {
            return Ok(outside);
        }
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
    let left_edge = bracket(peak, left_limit)?;
    let right_edge = bracket(peak, right_limit)?;
    let shifted = |x: f64| match log_integrand(x) {
        Ok(value) => (value - log_peak).exp(),
        Err(_) => f64::NAN,
    };
    let width = right_edge - left_edge;
    if !(width > 0.0) {
        return Ok(f64::NEG_INFINITY);
    }
    let quadrature = adaptive_simpson(
        &shifted,
        left_edge,
        right_edge,
        1.0e-13 * width.max(1.0e-6),
        32,
    )?;
    if !(quadrature.value > 0.0) {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(log_peak + quadrature.value.ln())
}

fn bivariate_geometry_upper_bound(
    lower: &[f64; 2],
    upper: &[f64; 2],
    correlation: f64,
) -> Result<f64, &'static str> {
    let marginal_bound =
        interval_probability(lower[0], upper[0])?.min(interval_probability(lower[1], upper[1])?);

    // For standardised X and Y, S=X+Y and D=X-Y are independent Gaussian
    // variables.  Every point in the rectangle lies in both induced
    // intervals, hence P(rectangle) <= P(S interval) P(D interval).
    let sum_sd = (2.0 * (1.0 + correlation)).sqrt();
    let difference_sd = (2.0 * (1.0 - correlation)).sqrt();
    let sum_probability = interval_probability(
        (lower[0] + lower[1]) / sum_sd,
        (upper[0] + upper[1]) / sum_sd,
    )?;
    let difference_probability = interval_probability(
        (lower[0] - upper[1]) / difference_sd,
        (upper[0] - lower[1]) / difference_sd,
    )?;
    Ok(marginal_bound
        .min(sum_probability * difference_probability)
        .clamp(0.0, 1.0))
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
    let geometry_bound = bivariate_geometry_upper_bound(
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[first, second],
        correlation,
    )?;
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
        24,
    )?;
    let probability = independent + quadrature.value;
    if probability <= quadrature.estimated_error {
        return Err("LATENT_MEDIATION_BIVARIATE_PROBABILITY_UNRESOLVED");
    }
    let upper_bound = normal_cdf(first)?.min(normal_cdf(second)?);
    let lower_bound = (normal_cdf(first)? + normal_cdf(second)? - 1.0).max(0.0);
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
        if error.abs() <= 15.0 * tolerance {
            Ok(QuadratureResult {
                value: left + right + error / 15.0,
                estimated_error: error.abs() / 15.0,
            })
        } else if depth == 0 {
            Err("LATENT_MEDIATION_BIVARIATE_QUADRATURE_DID_NOT_CONVERGE")
        } else {
            let left_result = recurse(function, lower, midpoint, left, tolerance / 2.0, depth - 1)?;
            let right_result =
                recurse(function, midpoint, upper, right, tolerance / 2.0, depth - 1)?;
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
    recurse(function, lower, upper, estimate, tolerance, maximum_depth)
}

#[cfg(feature = "python")]
pub mod python;

#[cfg(test)]
mod tests {
    use super::{
        FIT_GRADIENT_TOLERANCE, LatentMediationFamilyInput, LatentMediationModel,
        LatentMediationParameters, adaptive_simpson, bivariate_normal_cdf, bound_aware_gradient,
        directional_covariance, scaled_projected_gradient,
        stable_bound_aware_gradient, transformed_bounds, transformed_parameters,
    };
    use nalgebra::DMatrix;

    fn singleton(mediator_measurement: Option<f64>) -> LatentMediationFamilyInput {
        LatentMediationFamilyInput {
            relationship: vec![vec![1.0]],
            latent_mean: vec![0.0, 0.0],
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
        let error = adaptive_simpson(&|value| value.powi(4), 0.0, 1.0, 1.0e-16, 0)
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
        let bounds = transformed_bounds().expect("bounds");
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
        let bounds = transformed_bounds().expect("bounds");
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
