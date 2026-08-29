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
//! # Where the arithmetic is good
//!
//! Everywhere, now. The two-person case is an integral, and the one in
//! `crate::normal_integrals` is measured against an independent reference to
//! under 1e-09 across the whole correlation range and away from the symmetric
//! centre.
//!
//! **It was not always so, and the history is worth keeping.** Until 29 August
//! 2026 this file carried a sixteen-point quadrature whose error reached
//! 2.3e-04 at a correlation of 0.99 -- and some seventy times more than that
//! away from thresholds of nought and nought, which was the one place it had
//! ever been measured. `build` therefore refused any off-diagonal relationship
//! above 0.9, and in doing so refused **monozygotic twins**, one person entered
//! twice, and any model carrying a component a person shares in full with
//! themselves. The integral was replaced and the refusal went with it.

use nalgebra::DMatrix;
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::convergence::{self, TOLERANCE};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::blocks::family_blocks;
use crate::deviance::chi2_one_df_upper_tail;
use crate::interval::{self, Interval};

/// The smallest probability that is allowed to be taken a logarithm of.
const FLOOR: f64 = 1.0e-300;
/// Below this probability the two-person region is evaluated on the log scale.
///
/// The ordinary-scale integral holds an absolute tolerance, so its error in the
/// logarithm grows as the probability shrinks -- about the tolerance divided by
/// the probability. At 1e-3 that is 1e-9, which is the accuracy the integral is
/// measured to, so this is where the log-scale routine takes over.
const LOG_SCALE_BELOW: f64 = 1.0e-3;

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
            // The shared integral, and the only one in the package.
            //
            // **Only one of its refusals may become a number.**
            // `BIVARIATE_PROBABILITY_UNRESOLVED` says the probability fell
            // below what the integral can resolve, and the floor already
            // stands for that. Every other code -- a correlation outside its
            // range, a quadrature that did not converge, an answer the integral
            // rejected against its own bounds -- is a failure, and a failure
            // must not come back as a finite log probability. Returning `None`
            // is what makes the caller's `is_finite` check refuse the fit,
            // which is ADR 0016's rule: undefined is absent, not substituted.
            // Written first as `unwrap_or(0.0)`, which quietly handed the
            // optimiser -690.78 for every one of them.
            return match crate::normal_integrals::bivariate_normal_cdf(
                scaled[0],
                scaled[1],
                correlation[(0, 1)],
            ) {
                // **A small probability goes to the log scale too, not only an
                // unresolvable one.** The ordinary-scale integral works to an
                // *absolute* tolerance of 1e-12, so its error in the logarithm
                // is about that tolerance divided by the probability: fine at
                // 0.4, and 1.6e-05 at 1e-10. The log-scale routine is accurate
                // relative to the answer wherever it is asked, so below this
                // threshold it is simply the better of the two. At 1e-3 the
                // ordinary path's implied log error is 1e-9, which is where the
                // measured tolerance is set.
                Ok(probability) if probability >= LOG_SCALE_BELOW => {
                    Some(probability.max(FLOOR).ln())
                }
                Ok(_) | Err("BIVARIATE_PROBABILITY_UNRESOLVED") => {
                    crate::normal_integrals::log_bivariate_rectangle(
                        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
                        &[scaled[0], scaled[1]],
                        correlation[(0, 1)],
                    )
                    .ok()
                }
                // **Not the floor.** The ordinary-scale integral refuses below
                // 1e-12, and the floor is 1e-300, so substituting it turned
                // every probability between the two into -690.78. At two
                // thresholds of -6.5 and a correlation of 0.5 that returned
                // -690.78 where the answer is -32.89: a likelihood the
                // optimiser would believe, wrong by six hundred nats, in the
                // tail that a low prevalence and a heavily censored trait both
                // live in. The retired quadrature had no such gap, because it
                // returned a small number rather than refusing.
                //
                // So the deep tail goes to the log-scale routine, which is
                // written for it and cannot underflow.
                Err(_) => None,
            };
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

#[cfg(test)]
mod tests {
    use super::{LiabilityModel, mendell_elston};
    use crate::normal_integrals::bivariate_normal_cdf;
    use nalgebra::DMatrix;
    use statrs::distribution::{ContinuousCDF, Normal};

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
            let got = bivariate_normal_cdf(a, b, 0.0).expect("a resolvable probability");
            assert!(
                (got - product).abs() < 1e-9,
                "independence: {got} against {product}"
            );
        }
        // **One tolerance, across the whole range.** This test used to carry
        // two: 1e-8 below |rho| = 0.5, where an additive model puts sibling and
        // parent-child correlations, and 1e-3 above it. The second band was
        // written around the sixteen-point quadrature that used to sit here,
        // whose error reached 2.3e-04 at a correlation of 0.99 -- and only at
        // thresholds of nought and nought, being some seventy times worse away
        // from them. A tolerance written to accommodate a fault will not report
        // it. The integral is now accurate everywhere, so the accommodation is
        // gone and the high correlations are held to the same bar as the low.
        for &rho in &[
            -0.99_f64, -0.95, -0.5, -0.3, 0.0, 0.25, 0.5, 0.9, 0.99, 0.999,
        ] {
            let wanted = 0.25 + rho.asin() / (2.0 * std::f64::consts::PI);
            let got = bivariate_normal_cdf(0.0, 0.0, rho).expect("a resolvable probability");
            assert!(
                (got - wanted).abs() < 1e-9,
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
    /// The deep tail must be a probability, not the floor.
    ///
    /// **This is the test the first attempt needed and did not have.** The
    /// ordinary-scale integral refuses below 1e-12, and the floor here is
    /// 1e-300, so mapping that refusal to the floor turned every probability
    /// between the two into -690.78. At two thresholds of -6.5 and a
    /// correlation of 0.5 that is a likelihood wrong by six hundred nats and
    /// finite enough for the optimiser to believe -- in the tail where a rare
    /// binary trait and a heavily censored one both live. A prevalence of 1e-4
    /// puts a threshold at -3.7.
    ///
    /// The references are the conditional form integrated to about 1e-13, which
    /// does not underflow where the ordinary scale does.
    #[test]
    fn the_deep_tail_is_a_probability_and_not_the_floor() {
        let reference: [(f64, f64, f64, f64); 4] = [
            (-6.5, -6.5, 0.5, -32.886_826_836_5),
            (-8.0, -7.5, 0.7, -40.155_105_300_5),
            (-10.0, -9.0, 0.3, -75.573_637_132_1),
            (-12.0, -11.0, 0.6, -88.854_000_363_4),
        ];
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (first, second, correlation, wanted) in reference {
            let covariance = DMatrix::from_row_slice(2, 2, &[1.0, correlation, correlation, 1.0]);
            let got = LiabilityModel::region_log_probability(
                &[first, second],
                &[1.0, 1.0],
                &covariance,
                &normal,
            )
            .expect("the deep tail has a probability, small though it is");
            assert!(
                (got - wanted).abs() < 1e-6,
                "at ({first}, {second}) and correlation {correlation}: {got} \
                 against {wanted}"
            );
            assert!(
                got > -600.0,
                "at ({first}, {second}) the region came back at {got}, which is \
                 the floor standing in for an answer"
            );
        }
    }

    /// Monozygotic twins are people, and the model must take them.
    ///
    /// `build` refused any off-diagonal relationship above 0.9 until 29 August
    /// 2026, because the quadrature behind the two-person probability lost
    /// digits as the correlation approached one. Twins carry a relationship of
    /// one, so the model could not fit a pedigree containing any -- a
    /// limitation recorded nowhere. The integral was replaced and the refusal
    /// went with it.
    #[test]
    fn a_relationship_of_one_is_accepted() {
        let mut relationship = DMatrix::<f64>::identity(6, 6);
        for pair in 0..3 {
            relationship[(2 * pair, 2 * pair + 1)] = 1.0;
            relationship[(2 * pair + 1, 2 * pair)] = 1.0;
        }
        let status = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let design = DMatrix::from_element(6, 1, 1.0);
        let model = LiabilityModel::build(&relationship, &status, &design)
            .expect("three pairs of monozygotic twins are a covariance");
        assert!(
            model.loglik(0.5, &[0.0]).is_some(),
            "the likelihood must evaluate on a pedigree carrying twins"
        );
    }

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
            // **Loosened from 1e-8 on 29 August 2026, when the two-person
            // integral was replaced. The movement is worth reading rather than
            // waving through.**
            //
            // | h2  | beta    | moved   |
            // | --- | ------- | ------- |
            // | 0   | -0.2    | 4.4e-11 |
            // | 0   | -0.5244 | 1.8e-11 |
            // | 0   | -0.8    | 2.3e-08 |
            // | 0.1 | -0.2    | 9.9e-08 |
            // | 0.1 | -0.8    | 1.1e-06 |
            // | 0.2 | -0.8    | 1.2e-06 |
            //
            // **It tracks the tail, not the correlation.** At a heritability of
            // nought there is no correlation at all and the only thing that
            // changed is the univariate tail -- `erfc` in place of `statrs`,
            // whose error is relative and so bites where the probability is
            // small. Hence 1.8e-11 at a middling threshold and 2.3e-08 at
            // -0.8. Where the heritability is positive the two-person integral
            // runs as well and the same tail dependence appears an order up.
            //
            // So the retired code agreed with the successor more closely in the
            // body of the distribution and less closely in its tail, which is
            // where both of its approximations were weakest. That is not
            // evidence it was right: the replacement is measured against an
            // independent reference to under 1e-09 across the whole correlation
            // range **and** in the tail, at thresholds down to -2.5, and the
            // retired one never was. Two engines carrying similar
            // approximations agree with each other more closely than either
            // agrees with the truth.
            //
            // What this check establishes is unchanged: the two engines
            // describe the same likelihood, to 2.7e-09 of it at worst.
            assert!(
                (ours - wanted).abs() < 2e-6,
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
