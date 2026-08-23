//! One binary trait on a pedigree, through a liability threshold.
//!
//! **This is a special case rather than part of the Gaussian general model.**
//! Nothing here is folded into `ComponentModel`, and nothing here shares its
//! machinery beyond the family blocks.
//!
//! # The model
//!
//! Every person carries an unobserved liability
//!
//! ```text
//! L_i = x_i' beta + g_i + e_i
//! ```
//!
//! and is a case when it crosses nought. The liability has no scale of its own
//! -- only its sign is ever seen -- so its variance is fixed at one and the
//! covariance is `h2 A + (1 - h2) I`. The threshold is fixed at nought and the
//! intercept carries it, which is why a fitted intercept here is a threshold in
//! disguise and not a mean.
//!
//! What comes back is a heritability **on the liability scale**. It is not
//! comparable with a heritability of an observed 0/1 variable, and it is not
//! comparable with the REML heritabilities the rest of this package reports
//! without saying which scale is meant.
//!
//! # Maximum likelihood, and why there is no choice about it
//!
//! Every other model here defaults to REML. This one cannot. REML removes the
//! fixed effects by projecting the response onto the null space of the design,
//! and there is no response to project: the likelihood is a probability of a
//! region, not a density at a point. Fixed effects cannot be profiled out by
//! least squares either. So the estimator is maximum likelihood, the fit record
//! says so, and a liability heritability must never be set beside a REML one as
//! though the two were the same quantity.
//!
//! # The region probability
//!
//! A family of `k` people needs the probability of an orthant of a `k`
//! dimensional normal. One person uses the univariate normal distribution; two
//! use fixed sixteen-point quadrature. Above that this uses the Mendell-Elston
//! sequential truncation, which conditions on each observation in turn and
//! updates the remaining means and covariances -- an approximation, and named as
//! one, but a coherent joint one rather than a pairwise shortcut that would not
//! be a likelihood at all.
//!
//! **The order matters to the approximation**, so the rarer class is taken
//! first, which is what the recovered implementation does.
//!
//! # Where the arithmetic is good, and where it is not
//!
//! The two-person case is quadrature, and its accuracy depends on how strongly
//! the two liabilities correlate. Measured: the worst error is 1.3e-09 below a
//! correlation of 0.5 and 2.3e-04 below 0.99. An additive model puts sibling
//! and parent-child liability correlations at `h2 / 2`, so everything a
//! pedigree of ordinary relatives asks for sits in the good range.
//!
//! **A relationship of one does not.** Monozygotic twins, or the same person
//! entered twice, give a liability correlation of `h2` rather than `h2 / 2`,
//! and at a high heritability that reaches where the quadrature loses digits.
//! `build` refuses a relationship matrix carrying an off-diagonal one for that
//! reason, rather than returning a number quietly worth less than it looks.

use nalgebra::DMatrix;
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::convergence::{self, TOLERANCE};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::blocks::family_blocks;
use crate::deviance::chi2_one_df_upper_tail;
use crate::interval::{self, Interval};

/// Sixteen-point Gauss-Legendre nodes and weights on [-1, 1].
const GAUSS_LEGENDRE_16: [(f64, f64); 16] = [
    (-0.989_400_934_991_649_9, 0.027_152_459_411_754_1),
    (-0.944_575_023_073_232_6, 0.062_253_523_938_647_9),
    (-0.865_631_202_387_831_7, 0.095_158_511_682_492_8),
    (-0.755_404_408_355_003, 0.124_628_971_255_533_9),
    (-0.617_876_244_402_643_7, 0.149_595_988_816_577_1),
    (-0.458_016_777_657_227_4, 0.169_156_519_395_002_5),
    (-0.281_603_550_779_258_9, 0.182_603_415_044_923_6),
    (-0.095_012_509_837_637_4, 0.189_450_610_455_068_5),
    (0.095_012_509_837_637_4, 0.189_450_610_455_068_5),
    (0.281_603_550_779_258_9, 0.182_603_415_044_923_6),
    (0.458_016_777_657_227_4, 0.169_156_519_395_002_5),
    (0.617_876_244_402_643_7, 0.149_595_988_816_577_1),
    (0.755_404_408_355_003, 0.124_628_971_255_533_9),
    (0.865_631_202_387_831_7, 0.095_158_511_682_492_8),
    (0.944_575_023_073_232_6, 0.062_253_523_938_647_9),
    (0.989_400_934_991_649_9, 0.027_152_459_411_754_1),
];

/// The smallest probability that is allowed to be taken a logarithm of.
const FLOOR: f64 = 1.0e-300;

/// A fitted liability model.
#[derive(Clone, Debug)]
pub struct LiabilityFit {
    /// The heritability **of the liability**, not of the observed status.
    pub heritability: f64,
    /// Fixed effects on the liability scale. The first is an intercept only in
    /// the sense that it carries the threshold.
    pub fixed_effects: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    /// True where the gradient test failed on the first search and a second
    /// was run from that point with the objective tolerance switched off.
    pub polished: bool,
    pub scaled_gradient: f64,
    /// The share of people who are cases, which is what the threshold reflects.
    pub prevalence: f64,
    /// Always `ml`. There is no REML here; see the module note.
    pub estimator: &'static str,
    /// How many people sat in the largest family, because the region
    /// probability is exact to two and approximate above it.
    pub largest_family: usize,
}

/// One binary trait, one relationship matrix, one threshold.
pub struct LiabilityModel {
    relationship: DMatrix<f64>,
    design: DMatrix<f64>,
    /// True where the person is a case.
    case: Vec<bool>,
    blocks: Vec<Vec<usize>>,
    rows: usize,
}

impl LiabilityModel {
    /// Validate and prepare.
    ///
    /// `status` is one value per person: anything equal to one is a case and
    /// anything equal to nought is not. **Nothing else is accepted**, because a
    /// liability model silently scoring a 2 or a missing code as a case is a
    /// fault that shows up as a plausible number rather than an error.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(
        relationship: &DMatrix<f64>,
        status: &[f64],
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let rows = status.len();
        if rows == 0 {
            return Err("LIABILITY_NO_ROWS");
        }
        if relationship.nrows() != rows || relationship.ncols() != rows {
            return Err("LIABILITY_RELATIONSHIP_WRONG_SHAPE");
        }
        if design.nrows() != rows {
            return Err("LIABILITY_DESIGN_WRONG_SHAPE");
        }
        if design.ncols() == 0 {
            return Err("LIABILITY_DESIGN_HAS_NO_COLUMNS");
        }
        if !relationship.iter().all(|v| v.is_finite()) || !design.iter().all(|v| v.is_finite()) {
            return Err("LIABILITY_NOT_FINITE");
        }
        for i in 0..rows {
            for j in 0..i {
                if (relationship[(i, j)] - relationship[(j, i)]).abs() > 1e-10 {
                    return Err("LIABILITY_RELATIONSHIP_NOT_SYMMETRIC");
                }
            }
        }
        // The quadrature behind the two-person probability loses accuracy as
        // the liability correlation approaches one, which an off-diagonal
        // relationship of one produces at a high heritability. Twins and
        // duplicate rows are refused here rather than silently scored.
        for i in 0..rows {
            for j in 0..i {
                if relationship[(i, j)].abs() > 0.9 {
                    return Err("LIABILITY_RELATIONSHIP_TOO_CLOSE");
                }
            }
        }
        let mut case = Vec::with_capacity(rows);
        for value in status {
            if !value.is_finite() {
                return Err("LIABILITY_STATUS_NOT_FINITE");
            }
            if (value - 1.0).abs() < 1e-12 {
                case.push(true);
            } else if value.abs() < 1e-12 {
                case.push(false);
            } else {
                return Err("LIABILITY_STATUS_NOT_BINARY");
            }
        }
        let cases = case.iter().filter(|c| **c).count();
        // With every person on one side of the threshold there is no threshold
        // to find, and a heritability of a constant is not a quantity.
        if cases == 0 || cases == rows {
            return Err("LIABILITY_ALL_ONE_STATUS");
        }
        Ok(Self {
            relationship: relationship.clone(),
            design: design.clone(),
            case,
            blocks: family_blocks(relationship),
            rows,
        })
    }

    /// The share of people who are cases.
    #[must_use]
    pub fn prevalence(&self) -> f64 {
        self.case.iter().filter(|c| **c).count() as f64 / self.rows as f64
    }

    /// The log likelihood at one heritability and one set of fixed effects.
    ///
    /// Returns `None` where the parameters do not describe a model, which the
    /// search reads as somewhere not to go.
    fn loglik(&self, heritability: f64, beta: &[f64]) -> Option<f64> {
        if !(0.0..=1.0).contains(&heritability) || !beta.iter().all(|b| b.is_finite()) {
            return None;
        }
        let normal = Normal::new(0.0, 1.0).ok()?;
        let mut total = 0.0;
        for block in &self.blocks {
            let size = block.len();
            // The liability mean, and the sign that turns "case" into a
            // direction: a case needs the liability above nought, so its
            // probability runs the same way as its mean.
            let mean: Vec<f64> = block
                .iter()
                .map(|&row| {
                    (0..self.design.ncols())
                        .map(|column| self.design[(row, column)] * beta[column])
                        .sum::<f64>()
                })
                .collect();
            let sign: Vec<f64> = block
                .iter()
                .map(|&row| if self.case[row] { 1.0 } else { -1.0 })
                .collect();
            let covariance = DMatrix::from_fn(size, size, |i, j| {
                let (a, b) = (block[i], block[j]);
                let same = f64::from(u8::from(a == b));
                heritability * self.relationship[(a, b)] + (1.0 - heritability) * same
            });
            total += Self::region_log_probability(&mean, &sign, &covariance, &normal)?;
        }
        total.is_finite().then_some(total)
    }

    /// The log probability that one family's liabilities fall where its
    /// statuses say they do.
    pub(crate) fn region_log_probability(
        mean: &[f64],
        sign: &[f64],
        covariance: &DMatrix<f64>,
        normal: &Normal,
    ) -> Option<f64> {
        let size = mean.len();
        let mut scaled = Vec::with_capacity(size);
        for index in 0..size {
            let variance = covariance[(index, index)];
            if !(variance > 0.0) || !variance.is_finite() {
                return None;
            }
            scaled.push(sign[index] * mean[index] / variance.sqrt());
        }
        if size == 1 {
            return Some(normal.cdf(scaled[0]).max(FLOOR).ln());
        }

        let mut correlation = DMatrix::<f64>::identity(size, size);
        for row in 0..size {
            for column in (row + 1)..size {
                let scale = (covariance[(row, row)] * covariance[(column, column)]).sqrt();
                let value = sign[row] * sign[column] * covariance[(row, column)] / scale;
                if !value.is_finite() {
                    return None;
                }
                // Never exactly one: a correlation of one makes the sequential
                // update divide by a variance of nought.
                let value = value.clamp(-0.999_999, 0.999_999);
                correlation[(row, column)] = value;
                correlation[(column, row)] = value;
            }
        }
        if size == 2 {
            return Some(
                bivariate_normal_cdf(scaled[0], scaled[1], correlation[(0, 1)], normal)
                    .max(FLOOR)
                    .ln(),
            );
        }

        // **The rarer class first.** The sequential approximation conditions on
        // each person in turn, and its error depends on that order; taking the
        // smaller group first is what the recovered implementation does.
        let cases = sign.iter().filter(|s| **s > 0.0).count();
        let cases_first = cases * 2 <= size;
        let mut order: Vec<usize> = (0..size).collect();
        order.sort_by_key(|&index| {
            let is_case = sign[index] > 0.0;
            (u8::from(is_case != cases_first), index)
        });
        let ordered: Vec<f64> = order.iter().map(|&index| scaled[index]).collect();
        let ordered_correlation =
            DMatrix::from_fn(size, size, |i, j| correlation[(order[i], order[j])]);
        mendell_elston(&ordered, &ordered_correlation, normal)
    }

    /// Fit by bounded search from several starts.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start reached a usable optimum.
    pub fn fit(&self) -> Result<LiabilityFit, &'static str> {
        self.fit_holding(None)
    }

    /// Fit with the heritability held, which is how every null and every
    /// profile endpoint here is imposed.
    ///
    /// Holding it by fixing both bounds keeps one code path for the free and
    /// the constrained fit. A null fitted by different machinery from the
    /// alternative is the classic way to get a deviance that is not one.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start reached a usable optimum.
    fn fit_holding(&self, held: Option<f64>) -> Result<LiabilityFit, &'static str> {
        let columns = self.design.ncols();
        let count = columns + 1;
        let prevalence = self.prevalence();
        // A sensible threshold before anything is fitted: the value that would
        // give this prevalence with no covariates and no family structure.
        let threshold = Normal::new(0.0, 1.0)
            .map_err(|_| "LIABILITY_NORMAL_UNAVAILABLE")?
            .inverse_cdf(prevalence.clamp(1e-6, 1.0 - 1e-6));

        let mut lower = vec![f64::NEG_INFINITY; count];
        let mut upper = vec![f64::INFINITY; count];
        lower[0] = 0.0;
        upper[0] = 1.0;
        if let Some(value) = held {
            if !(0.0..=1.0).contains(&value) {
                return Err("LIABILITY_HELD_HERITABILITY_OUT_OF_RANGE");
            }
            lower[0] = value;
            upper[0] = value;
        }

        let starts: Vec<Vec<f64>> = [0.05_f64, 0.3, 0.6]
            .iter()
            .map(|&h| {
                let mut start = vec![0.0; count];
                start[0] = held.unwrap_or(h);
                // The intercept carries the threshold, and the sign is the
                // other way round: a common trait needs a low bar.
                start[1] = threshold;
                start
            })
            .collect();

        let value_of =
            |theta: &[f64]| -> f64 { self.loglik(theta[0], &theta[1..]).map_or(1e30, |v| -v) };
        // The region probability has no closed-form derivative worth having, so
        // the gradient is a central difference of an objective that is itself
        // cheap: no factorisation is repeated, only the sequential update.
        let gradient_of = |theta: &[f64]| -> Vec<f64> {
            (0..count)
                .map(|k| {
                    let step = 1e-5 * theta[k].abs().max(1.0);
                    let mut up = theta.to_vec();
                    let mut down = theta.to_vec();
                    up[k] = (up[k] + step).min(upper[k]);
                    down[k] = (down[k] - step).max(lower[k]);
                    let width = up[k] - down[k];
                    if width <= 0.0 {
                        return 0.0;
                    }
                    (value_of(&up) - value_of(&down)) / width
                })
                .collect()
        };

        let mut best: Option<(f64, Vec<f64>)> = None;
        for start in starts {
            let Ok(bounds) = Bounds::new(lower.clone(), upper.clone()) else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(count);
            control.maxit = 300;
            control.fnscale = value_of(&start).abs().max(1.0);
            control.parscale = vec![1.0; count];
            control.factr = 1.0e3;
            control.pgtol = 1e-8;
            control.lmm = count;
            let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, &value_of, &gradient_of, control)
            else {
                continue;
            };
            let value = value_of(&solution.par);
            if value.is_finite() && best.as_ref().is_none_or(|(seen, _)| value < *seen) {
                best = Some((value, solution.par));
            }
        }
        let reading = |theta: &[f64], negative: f64| -> f64 {
            let gradient = gradient_of(theta);
            let projected = gradient
                .iter()
                .enumerate()
                .map(|(k, g)| {
                    if k == 0
                        && (held.is_some()
                            || (theta[0] <= 0.0 && *g > 0.0)
                            || (theta[0] >= 1.0 && *g < 0.0))
                    {
                        0.0
                    } else {
                        *g
                    }
                })
                .fold(0.0f64, |worst, g| worst.max(g.abs()));
            projected / negative.abs().max(1.0)
        };

        let (mut negative, mut theta) = best.ok_or("LIABILITY_NO_START_CONVERGED")?;
        let mut scaled_gradient = reading(&theta, negative);

        // One more search where the gradient test failed, and nowhere else.
        // See `convergence` for why, and for what it was measured to cost.
        let mut polished = false;
        if scaled_gradient >= TOLERANCE
            && let Some(better) = convergence::polish(
                &theta,
                negative,
                scaled_gradient,
                &lower,
                &upper,
                &value_of,
                &gradient_of,
                |candidate| {
                    let value = value_of(candidate);
                    value
                        .is_finite()
                        .then(|| (value, reading(candidate, value)))
                },
            )
        {
            polished = true;
            negative = better.negative_loglik;
            theta = better.par;
            scaled_gradient = better.scaled_gradient;
        }

        Ok(LiabilityFit {
            heritability: theta[0],
            fixed_effects: theta[1..].to_vec(),
            loglik: -negative,
            converged: scaled_gradient < TOLERANCE,
            polished,
            scaled_gradient,
            prevalence,
            estimator: "ml",
            largest_family: self.blocks.iter().map(Vec::len).max().unwrap_or(0),
        })
    }
}

/// The bivariate standard normal distribution function, by sixteen-point
/// Gauss-Legendre quadrature of the conditional.
fn bivariate_normal_cdf(a: f64, b: f64, rho: f64, normal: &Normal) -> f64 {
    if !a.is_finite() || !b.is_finite() || !rho.is_finite() {
        return f64::NAN;
    }
    if a <= -10.0 || b <= -10.0 {
        return 0.0;
    }
    if a >= 10.0 {
        return normal.cdf(b);
    }
    if b >= 10.0 {
        return normal.cdf(a);
    }
    if rho.abs() <= 1.0e-12 {
        return normal.cdf(a) * normal.cdf(b);
    }
    let rho = rho.clamp(-0.999_999, 0.999_999);
    let (lower, upper) = (-10.0, a.min(10.0));
    if upper <= lower {
        return 0.0;
    }
    let half_width = 0.5 * (upper - lower);
    let centre = 0.5 * (upper + lower);
    // Factored rather than `(1 - rho^2)`, which cancels as the correlation
    // approaches one and loses most of the significand exactly where the
    // conditional distribution is narrowest.
    let conditional_sd = ((1.0 - rho) * (1.0 + rho)).sqrt();
    GAUSS_LEGENDRE_16
        .iter()
        .map(|(node, weight)| {
            let x = centre + half_width * node;
            weight * density(x) * normal.cdf((b - rho * x) / conditional_sd)
        })
        .sum::<f64>()
        * half_width
}

/// The standard normal density.
fn density(value: f64) -> f64 {
    const INV_SQRT_TWO_PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_TWO_PI * (-0.5 * value * value).exp()
}

/// `P(Z_i <= threshold_i for every i)` for a correlated standard normal
/// vector, by the Mendell-Elston sequential truncation update.
///
/// Each observation in turn contributes its own conditional probability, and
/// the remaining means and covariances are updated for having conditioned on
/// it. It is an approximation and is named as one; what it is not is a
/// pairwise composite, which would not be a joint likelihood at all.
fn mendell_elston(thresholds: &[f64], correlation: &DMatrix<f64>, normal: &Normal) -> Option<f64> {
    let dimension = thresholds.len();
    if correlation.nrows() != dimension || correlation.ncols() != dimension {
        return None;
    }
    let mut mean = vec![0.0; dimension];
    let mut covariance = correlation.clone();
    let mut total = 0.0;
    for step in 0..dimension {
        let variance = covariance[(step, step)];
        if !(variance > 0.0) || !variance.is_finite() {
            return None;
        }
        let sd = variance.sqrt();
        let z = (thresholds[step] - mean[step]) / sd;
        let probability = normal.cdf(z).clamp(FLOOR, 1.0);
        total += probability.ln();
        let height = density(z);
        if height == 0.0 {
            continue;
        }
        let mills = height / probability;
        if !mills.is_finite() {
            return None;
        }
        let multiplier = (1.0 - z * mills - mills * mills).clamp(1.0e-12, 1.0);
        let covariance_scale = (multiplier - 1.0) / variance;
        let mean_scale = -mills / sd;
        for row in (step + 1)..dimension {
            mean[row] += covariance[(row, step)] * mean_scale;
        }
        for row in (step + 1)..dimension {
            for column in (step + 1)..=row {
                let updated = covariance[(row, column)]
                    + covariance[(row, step)] * covariance[(column, step)] * covariance_scale;
                covariance[(row, column)] = updated;
                covariance[(column, row)] = updated;
            }
        }
    }
    total.is_finite().then_some(total)
}

/// One value known only to lie one side of a limit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Truncation {
    pub limit: f64,
    /// `1.0` where the value lies at or above its limit, `-1.0` where at or
    /// below. This is the same convention `region_log_probability` takes.
    pub sign: f64,
}

/// The mean and covariance of a Gaussian vector after conditioning on some of
/// its coordinates lying beyond their limits.
///
/// # Why this exists beside [`mendell_elston`]
///
/// It is the same sequential update, arithmetic for arithmetic, and it returns
/// the same log probability -- there is a test that says so to the last bit.
/// What it also returns is the running mean and covariance that the update
/// maintains and that [`mendell_elston`] discards, because a probability was
/// all that was wanted there.
///
/// Those moments are an expectation step. A model fitted by
/// expectation-maximisation over censored values needs exactly the conditional
/// mean of each unmeasured value and the covariance of what is left uncertain,
/// and both fall out of an update that was already being computed.
///
/// # What it does differently
///
/// [`mendell_elston`] updates only the coordinates it has not yet reached,
/// which is all a probability needs. This updates every coordinate, so that one
/// already processed picks up what conditioning on a later one says about it.
/// That changes none of the probabilities: the value at step `i` reads only the
/// mean and variance of coordinate `i`, and those are updated identically
/// either way.
///
/// # Coordinates that are merely carried
///
/// A `None` in `truncation` is a coordinate with no limit -- a value that was
/// never measured at all. It is never conditioned on, because nothing is known
/// about it, but it is correlated with the ones that are and its moments move
/// when they are truncated. Dropping it instead and conditioning afterwards
/// would be the same thing done twice.
///
/// # It is an approximation, and the order is part of it
///
/// Sequential truncation is exact to two coordinates and approximate above,
/// and its error depends on the order the coordinates are taken in. The order
/// here is the one given. `checks/sequential_against_ghk.py` is what says how
/// much that costs on the design this was written for, and the answer there
/// rests on the coordinates being nearly independent once everything measured
/// has been conditioned on.
pub(crate) fn truncated_moments(
    mean: &mut [f64],
    covariance: &mut DMatrix<f64>,
    truncation: &[Option<Truncation>],
    normal: &Normal,
) -> Option<f64> {
    let dimension = mean.len();
    if covariance.nrows() != dimension
        || covariance.ncols() != dimension
        || truncation.len() != dimension
    {
        return None;
    }
    let mut total = 0.0;
    for step in 0..dimension {
        let Some(limit) = truncation[step] else {
            continue;
        };
        let variance = covariance[(step, step)];
        if !(variance > 0.0) || !variance.is_finite() {
            return None;
        }
        let sd = variance.sqrt();
        // The event is `sign * (value - limit) > 0`, whose probability is the
        // distribution function at this point however the sign falls.
        let z = limit.sign * (mean[step] - limit.limit) / sd;
        let probability = normal.cdf(z).clamp(FLOOR, 1.0);
        total += probability.ln();
        let height = density(z);
        if height == 0.0 {
            continue;
        }
        let mills = height / probability;
        if !mills.is_finite() {
            return None;
        }
        let multiplier = (1.0 - z * mills - mills * mills).clamp(1.0e-12, 1.0);
        let covariance_scale = (multiplier - 1.0) / variance;
        let mean_scale = limit.sign * mills / sd;
        let column: Vec<f64> = (0..dimension).map(|row| covariance[(row, step)]).collect();
        for row in 0..dimension {
            mean[row] += column[row] * mean_scale;
        }
        for row in 0..dimension {
            for at in 0..=row {
                let updated = covariance[(row, at)] + column[row] * column[at] * covariance_scale;
                covariance[(row, at)] = updated;
                covariance[(at, row)] = updated;
            }
        }
    }
    total.is_finite().then_some(total)
}

#[cfg(test)]
mod tests {
    use super::{
        LiabilityModel, Truncation, bivariate_normal_cdf, mendell_elston, truncated_moments,
    };
    use nalgebra::DMatrix;
    use statrs::distribution::{ContinuousCDF, Normal};

    /// A correlated pair, used by the truncation tests below.
    fn pair(variance_one: f64, variance_two: f64, correlation: f64) -> DMatrix<f64> {
        let covariance = correlation * (variance_one * variance_two).sqrt();
        let mut matrix = DMatrix::<f64>::zeros(2, 2);
        matrix[(0, 0)] = variance_one;
        matrix[(1, 1)] = variance_two;
        matrix[(0, 1)] = covariance;
        matrix[(1, 0)] = covariance;
        matrix
    }

    /// **The two updates must agree to the last bit.** They are the same
    /// arithmetic; one keeps what the other throws away. If this ever fails,
    /// the two have drifted apart and one of them is no longer the sequential
    /// truncation that `checks/sequential_against_ghk.py` measured.
    #[test]
    fn the_moments_update_returns_the_probability_update_exactly() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for size in 1..7 {
            let mut correlation = DMatrix::<f64>::identity(size, size);
            for row in 0..size {
                for column in 0..row {
                    let value = 0.9_f64.powi(i32::try_from(row - column).unwrap_or(0)) * 0.8;
                    correlation[(row, column)] = value;
                    correlation[(column, row)] = value;
                }
            }
            let thresholds: Vec<f64> = (0..size).map(|i| 0.4 * (i as f64) - 0.7).collect();
            let wanted = mendell_elston(&thresholds, &correlation, &normal)
                .expect("the probability update should run");

            // The same problem in the moments update's own terms. It works on
            // `sign * (value - limit) > 0` where the other works on
            // `Z <= threshold`, and a sign of minus one is what turns the first
            // into the second at the same limit.
            let mut mean = vec![0.0; size];
            let mut covariance = correlation.clone();
            let truncation: Vec<Option<Truncation>> = thresholds
                .iter()
                .map(|t| {
                    Some(Truncation {
                        limit: *t,
                        sign: -1.0,
                    })
                })
                .collect();
            let got = truncated_moments(&mut mean, &mut covariance, &truncation, &normal)
                .expect("the moments update should run");
            assert_eq!(got, wanted, "at {size} coordinates");
        }
    }

    /// One truncation is not an approximation at all: conditioning on a single
    /// coordinate's region is the ordinary regression, and every moment has a
    /// closed form to check against -- including the moments of a coordinate
    /// that has no limit of its own and only feels this one through their
    /// correlation.
    #[test]
    fn one_truncated_coordinate_is_exact_and_carries_the_other() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for &(limit, sign) in &[(1.5_f64, 1.0_f64), (-0.4, 1.0), (0.7, -1.0)] {
            let (variance, other_variance, correlation) = (2.0, 0.75, 0.6);
            let start = [0.3, -0.2];
            let mut mean = start;
            let mut covariance = pair(variance, other_variance, correlation);
            let truncation = [Some(Truncation { limit, sign }), None];
            let probability = truncated_moments(&mut mean, &mut covariance, &truncation, &normal)
                .expect("the moments update should run");

            let sd = variance.sqrt();
            let z = sign * (start[0] - limit) / sd;
            let expected_probability = normal.cdf(z);
            let mills = super::density(z) / expected_probability;
            let multiplier = 1.0 - z * mills - mills * mills;
            let cross = correlation * (variance * other_variance).sqrt();

            assert!((probability - expected_probability.ln()).abs() < 1e-12);
            assert!((mean[0] - (start[0] + sign * sd * mills)).abs() < 1e-12);
            assert!((covariance[(0, 0)] - variance * multiplier).abs() < 1e-12);
            // The carried coordinate moves by its regression on the truncated
            // one, and loses the same share of its shared variance.
            assert!((mean[1] - (start[1] + cross * sign * mills / sd)).abs() < 1e-12);
            assert!(
                (covariance[(1, 1)]
                    - (other_variance + cross * cross * (multiplier - 1.0) / variance))
                    .abs()
                    < 1e-12
            );
            assert!((covariance[(0, 1)] - cross * multiplier).abs() < 1e-12);
        }
    }

    /// Where it *is* an approximation, it should still be close. Two truncated
    /// coordinates at a correlation of 0.7, against a million draws.
    #[test]
    fn two_truncated_coordinates_are_close_to_the_truth() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let (variance, correlation) = (1.0, 0.7);
        let limits = [0.2_f64, -0.3];
        let start = [0.0, 0.0];
        let covariance_start = pair(variance, variance, correlation);

        let mut mean = start;
        let mut covariance = covariance_start.clone();
        let truncation = [
            Some(Truncation {
                limit: limits[0],
                sign: 1.0,
            }),
            Some(Truncation {
                limit: limits[1],
                sign: 1.0,
            }),
        ];
        truncated_moments(&mut mean, &mut covariance, &truncation, &normal)
            .expect("the moments update should run");

        let factor = covariance_start
            .clone()
            .cholesky()
            .expect("the pair should factorise")
            .l();
        let mut state = 20_260_820_u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (mut kept, mut sum, mut squares, mut cross) = (0.0, [0.0; 2], [0.0; 2], 0.0);
        for _ in 0..1_000_000 {
            let u1 = uniform();
            let u2 = uniform();
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;
            let z = [radius * angle.cos(), radius * angle.sin()];
            let first = factor[(0, 0)] * z[0];
            let second = factor[(1, 0)] * z[0] + factor[(1, 1)] * z[1];
            if first > limits[0] && second > limits[1] {
                kept += 1.0;
                sum[0] += first;
                sum[1] += second;
                squares[0] += first * first;
                squares[1] += second * second;
                cross += first * second;
            }
        }
        let truth = [sum[0] / kept, sum[1] / kept];
        // A tenth of a standard deviation, which is far tighter than the
        // difference between imputing a censored value and substituting its
        // limit, and is what this update is for.
        for index in 0..2 {
            assert!(
                (mean[index] - truth[index]).abs() < 0.1,
                "coordinate {index}: sequential {} against {} from a million draws",
                mean[index],
                truth[index]
            );
            let variance_truth = squares[index] / kept - truth[index] * truth[index];
            assert!(
                (covariance[(index, index)] - variance_truth).abs() < 0.1,
                "coordinate {index} variance: {} against {variance_truth}",
                covariance[(index, index)]
            );
        }
        let cross_truth = cross / kept - truth[0] * truth[1];
        assert!(
            (covariance[(0, 1)] - cross_truth).abs() < 0.1,
            "covariance {} against {cross_truth}",
            covariance[(0, 1)]
        );
    }

    /// The bivariate distribution function, where it can be checked by hand.
    ///
    /// At a correlation of nought it is a product; at any correlation the
    /// diagonal `P(Z1 <= 0, Z2 <= 0)` is `1/4 + asin(rho) / (2 pi)`, which is
    /// the one closed form this function has.
    #[test]
    fn the_bivariate_distribution_function_is_right_where_it_is_known() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for &(a, b) in &[(0.0, 0.0), (1.0, -0.5), (-1.3, 2.0)] {
            let product = normal.cdf(a) * normal.cdf(b);
            let got = bivariate_normal_cdf(a, b, 0.0, &normal);
            assert!(
                (got - product).abs() < 1e-9,
                "independence: {got} against {product}"
            );
        }
        // **The quadrature is excellent where it is used and poor outside
        // it**, and the two are worth separating. Measured over a grid of
        // correlations: the worst error below |rho| = 0.5 is 1.3e-09, and the
        // worst below 0.99 is 2.3e-04. An additive model puts sibling and
        // parent-child liability correlations at h2/2, so the first is the
        // range that occurs and the second is the guard rail.
        for &rho in &[-0.5_f64, -0.3, 0.0, 0.25, 0.5] {
            let wanted = 0.25 + rho.asin() / (2.0 * std::f64::consts::PI);
            let got = bivariate_normal_cdf(0.0, 0.0, rho, &normal);
            assert!(
                (got - wanted).abs() < 1e-8,
                "at rho {rho}, inside the range that occurs: {got} against the \
                 closed form {wanted}"
            );
        }
        for &rho in &[-0.95_f64, 0.9, 0.99] {
            let wanted = 0.25 + rho.asin() / (2.0 * std::f64::consts::PI);
            let got = bivariate_normal_cdf(0.0, 0.0, rho, &normal);
            assert!(
                (got - wanted).abs() < 1e-3,
                "at rho {rho}: {got} against the closed form {wanted}"
            );
        }
    }

    /// **The sequential approximation must be exact where the truth is a
    /// product**, or its updates are wrong rather than approximate.
    #[test]
    fn the_sequential_update_is_exact_when_nothing_is_correlated() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let thresholds = [0.4, -1.1, 0.9, 2.0];
        let independent = DMatrix::<f64>::identity(4, 4);
        let got = mendell_elston(&thresholds, &independent, &normal).expect("computes");
        let wanted: f64 = thresholds.iter().map(|z| normal.cdf(*z).ln()).sum();
        assert!((got - wanted).abs() < 1e-10, "{got} against {wanted}");
    }

    /// Sibling pairs with a known liability heritability, and the status the
    /// threshold implies. Returns the relationship matrix, the statuses and the
    /// design.
    fn simulate(
        heritability: f64,
        prevalence: f64,
        seed: u64,
    ) -> (DMatrix<f64>, Vec<f64>, DMatrix<f64>) {
        let families = 700;
        let n = 2 * families;
        let mut relationship = DMatrix::<f64>::identity(n, n);
        for family in 0..families {
            relationship[(2 * family, 2 * family + 1)] = 0.5;
            relationship[(2 * family + 1, 2 * family)] = 0.5;
        }
        let mut state = seed;
        let mut normal_draw = || {
            let mut total = 0.0;
            for _ in 0..12 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                total += (state >> 11) as f64 / (1u64 << 53) as f64;
            }
            total - 6.0
        };
        let threshold = Normal::new(0.0, 1.0).unwrap().inverse_cdf(1.0 - prevalence);
        let shared = (heritability / 2.0).sqrt();
        let own = (1.0 - heritability / 2.0).sqrt();
        let mut status = Vec::with_capacity(n);
        for _ in 0..families {
            // Two siblings share half their additive variance.
            let common = normal_draw() * shared;
            for _ in 0..2 {
                let liability = common + normal_draw() * own;
                status.push(f64::from(u8::from(liability > threshold)));
            }
        }
        (relationship, status, DMatrix::from_element(n, 1, 1.0))
    }

    /// A liability heritability that is there is recovered.
    #[test]
    fn a_liability_heritability_is_recovered() {
        let (relationship, status, design) = simulate(0.6, 0.25, 20_260_814);
        let model = LiabilityModel::build(&relationship, &status, &design).expect("valid");
        let fit = model.fit().expect("fits");
        assert!(
            fit.converged,
            "did not converge, |g| = {}",
            fit.scaled_gradient
        );
        assert!(
            !fit.polished,
            "the polish fired on a fit that already met the gradient test"
        );
        assert!(
            (fit.heritability - 0.6).abs() < 0.15,
            "simulated at 0.6 and recovered {}",
            fit.heritability
        );
        // The intercept carries the threshold, so it should sit near the value
        // the prevalence implies, with the opposite sign.
        let wanted = -Normal::new(0.0, 1.0).unwrap().inverse_cdf(1.0 - 0.25);
        assert!(
            (fit.fixed_effects[0] - wanted).abs() < 0.2,
            "the threshold came back as {} where {wanted} was simulated",
            fit.fixed_effects[0]
        );
        assert_eq!(fit.estimator, "ml");
    }

    /// **With no family resemblance there is none to find.** A liability model
    /// that manufactures heritability from unrelated people is worse than one
    /// that misses it.
    #[test]
    fn no_liability_heritability_is_not_invented() {
        let (relationship, status, design) = simulate(0.0, 0.3, 5150);
        let model = LiabilityModel::build(&relationship, &status, &design).expect("valid");
        let fit = model.fit().expect("fits");
        assert!(
            fit.heritability < 0.15,
            "nothing was simulated but {} came back",
            fit.heritability
        );
    }

    /// A status column that is not binary, or is all one way, is refused rather
    /// than scored. A 2 quietly read as a case is a fault that looks like a
    /// number rather than an error.
    #[test]
    fn only_a_real_binary_status_is_accepted() {
        let (relationship, status, design) = simulate(0.4, 0.25, 9);
        let n = status.len();
        let mut wrong = status.clone();
        wrong[3] = 2.0;
        assert!(LiabilityModel::build(&relationship, &wrong, &design).is_err());
        assert!(
            LiabilityModel::build(&relationship, &vec![0.0; n], &design).is_err(),
            "everyone a non-case is not a model"
        );
        assert!(LiabilityModel::build(&relationship, &vec![1.0; n], &design).is_err());
        assert!(LiabilityModel::build(&relationship, &status, &design).is_ok());
    }
}

#[cfg(test)]
mod against_the_source {
    use super::LiabilityModel;
    use nalgebra::DMatrix;

    /// Sibling pairs whose statuses come from a liability with a known
    /// heritability. **The generator is duplicated on both sides on purpose**:
    /// it is a dozen lines of arithmetic, and copying it is cheaper and safer
    /// than shipping four hundred numbers between two crates.
    fn simulate(pairs: usize, h2: f64, threshold: f64, seed: u64) -> (DMatrix<f64>, Vec<f64>) {
        let n = 2 * pairs;
        let mut a = DMatrix::<f64>::identity(n, n);
        for p in 0..pairs {
            a[(2 * p, 2 * p + 1)] = 0.5;
            a[(2 * p + 1, 2 * p)] = 0.5;
        }
        let mut state = seed;
        let mut draw = || {
            let mut total = 0.0;
            for _ in 0..12 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                total += (state >> 11) as f64 / (1u64 << 53) as f64;
            }
            total - 6.0
        };
        let shared = (h2 / 2.0).sqrt();
        let own = (1.0 - h2 / 2.0).sqrt();
        let mut status = Vec::with_capacity(n);
        for _ in 0..pairs {
            let common = draw() * shared;
            for _ in 0..2 {
                let liability = common + draw() * own;
                status.push(f64::from(u8::from(liability > threshold)));
            }
        }
        (a, status)
    }

    /// **Is this the same likelihood the SOLAR successor computes?**
    ///
    /// The region probability, the sequential approximation and the ordering
    /// were all cannibalised from that engine, so this is not an independent
    /// derivation and does not pretend to be one. What it does establish is
    /// that the transplant is faithful: two hundred sibling pairs, a grid of
    /// heritabilities and intercepts, and every value agreeing to the ten
    /// digits the reference printed.
    ///
    /// **The intercept runs the other way here.** That engine takes a case as
    /// the negative direction, so its intercept is the negative of this one's.
    /// This package's convention makes the intercept readable: `Phi(intercept)`
    /// is the prevalence a model with no covariates implies, which is 0.30
    /// against the 124 of 400 simulated here.
    #[test]
    fn the_likelihood_matches_the_solar_successor() {
        let (a, status) = simulate(200, 0.5, 0.524_400_512_708_040_9, 12_345);
        assert_eq!(status.len(), 400, "the two sides must see the same data");
        assert_eq!(
            status.iter().filter(|v| **v > 0.5).count(),
            124,
            "the two sides must see the same statuses"
        );
        let design = DMatrix::from_element(status.len(), 1, 1.0);
        let model = LiabilityModel::build(&a, &status, &design).expect("valid");

        // Printed by that engine at these inputs, with `log_sd` at nought.
        // Its intercept is the negative of ours, so ours is negated below.
        let reference: [(f64, f64, f64); 9] = [
            (0.0, -0.8, -457.832_103_073_8),
            (0.0, -0.5244, -376.524_054_617_1),
            (0.0, -0.2, -306.648_648_121_7),
            (0.1, -0.8, -450.260_386_871_9),
            (0.1, -0.5244, -371.645_210_734_8),
            (0.1, -0.2, -304.041_378_536_3),
            (0.2, -0.8, -443.305_591_065_0),
            (0.2, -0.5244, -367.181_634_833_7),
            (0.2, -0.2, -301.688_694_053_5),
        ];
        for (heritability, their_beta, wanted) in reference {
            let ours = model
                .loglik(heritability, &[-their_beta])
                .expect("evaluates");
            assert!(
                (ours - wanted).abs() < 1e-8,
                "at h2 {heritability} and their beta {their_beta}: this package \
                 gives {ours}, the successor gives {wanted}"
            );
        }
    }
}

/// One test of a heritability against nought.
#[derive(Clone, Copy, Debug)]
pub struct LiabilityTest {
    pub statistic: f64,
    pub p_value: f64,
    pub rule: &'static str,
    pub null_loglik: f64,
}

/// Compatibility name for the one shared interval record.
pub type LiabilityInterval = Interval;

impl LiabilityModel {
    /// Test the liability heritability against nought.
    ///
    /// A heritability of
    /// nought sits on a bound, so the even mixture of a point mass and
    /// chi-square on one degree of freedom is the natural reference, and it is
    /// what a Gaussian variance component gets. Its finite-sample behaviour is
    /// evaluated by `checks/liability_calibration.py`.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit fails.
    pub fn heritability_test(&self) -> Result<LiabilityTest, &'static str> {
        let free = self.fit()?;
        let null = self.fit_holding(Some(0.0))?;
        crate::convergence::require(free.converged, "LIABILITY_FIT_NOT_CONVERGED")?;
        crate::convergence::require(null.converged, "LIABILITY_NULL_FIT_NOT_CONVERGED")?;
        let statistic = (2.0 * (free.loglik - null.loglik)).max(0.0);
        let p_value = if statistic < 1e-6 {
            1.0
        } else {
            0.5 * chi2_one_df_upper_tail(statistic)
        };
        Ok(LiabilityTest {
            statistic,
            p_value: p_value.clamp(0.0, 1.0),
            rule: "mixture_50_50",
            null_loglik: null.loglik,
        })
    }

    /// A 95 per cent profile-likelihood interval for the liability
    /// heritability.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the free fit fails.
    pub fn heritability_interval(&self) -> Result<LiabilityInterval, &'static str> {
        let free = self.fit()?;
        if !free.converged {
            return Err("LIABILITY_FIT_NOT_CONVERGED");
        }
        let estimate = free.heritability;
        let got = interval::profile_interval(estimate, (0.0, 1.0), |value| {
            self.fit_holding(Some(value))
                .ok()
                .filter(|fit| fit.converged)
                .map(|fit| fit.loglik)
        });
        if got.estimate.is_none() {
            return Err("LIABILITY_PROFILE_NOT_EVALUABLE");
        }
        // Liability coverage has not yet scored the mixture endpoint; unlike
        // the one-trait Gaussian and Tobit families, the verdict stays absent.
        Ok(got)
    }
}

/// Legacy signed-profile bisection used by deferred latent-mediation confidence
/// sets. Supported bounded profiles use [`crate::interval::profile_interval`].
pub(crate) fn bisect(mut out: f64, mut inside: f64, outside: &impl Fn(f64) -> bool) -> f64 {
    for _ in 0..40 {
        let middle = 0.5 * (out + inside);
        if outside(middle) {
            out = middle;
        } else {
            inside = middle;
        }
        if (out - inside).abs() < 1e-5 {
            break;
        }
    }
    0.5 * (out + inside)
}

#[cfg(feature = "python")]
pub mod python;
