//! One binary trait on a pedigree, through a liability threshold.
//!
//! **This is deliberately a special case and not part of the general model.**
//! Decision 16 of the estimator record keeps the Gaussian design Gaussian, and
//! this module is what that decision leaves room for rather than something it
//! forbids. Nothing here is folded into `ComponentModel`, and nothing here
//! shares its machinery beyond the family blocks.
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
//! dimensional normal. One and two people are exact. Above that this uses the
//! Mendell-Elston sequential truncation, which conditions on each observation in
//! turn and updates the remaining means and covariances -- an approximation, and
//! named as one, but a coherent joint one rather than a pairwise shortcut that
//! would not be a likelihood at all.
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
use statrs::distribution::{ContinuousCDF, Normal};

use crate::blocks::family_blocks;

/// Sixteen-point Gauss-Legendre nodes and weights on [-1, 1].
const GAUSS_LEGENDRE_16: [(f64, f64); 16] = [
    (-0.989_400_934_991_649_9, 0.027_152_459_411_754_1),
    (-0.944_575_023_073_232_6, 0.062_253_523_938_647_9),
    (-0.865_631_202_387_831_7, 0.095_158_511_682_492_8),
    (-0.755_404_408_355_003_0, 0.124_628_971_255_533_9),
    (-0.617_876_244_402_643_7, 0.149_595_988_816_577_1),
    (-0.458_016_777_657_227_4, 0.169_156_519_395_002_5),
    (-0.281_603_550_779_258_9, 0.182_603_415_044_923_6),
    (-0.095_012_509_837_637_4, 0.189_450_610_455_068_5),
    (0.095_012_509_837_637_4, 0.189_450_610_455_068_5),
    (0.281_603_550_779_258_9, 0.182_603_415_044_923_6),
    (0.458_016_777_657_227_4, 0.169_156_519_395_002_5),
    (0.617_876_244_402_643_7, 0.149_595_988_816_577_1),
    (0.755_404_408_355_003_0, 0.124_628_971_255_533_9),
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
            total += self.region_log_probability(&mean, &sign, &covariance, &normal)?;
        }
        total.is_finite().then_some(total)
    }

    /// The log probability that one family's liabilities fall where its
    /// statuses say they do.
    fn region_log_probability(
        &self,
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
                let scale =
                    (covariance[(row, row)] * covariance[(column, column)]).sqrt();
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

        let starts: Vec<Vec<f64>> = [0.05_f64, 0.3, 0.6]
            .iter()
            .map(|&h| {
                let mut start = vec![0.0; count];
                start[0] = h;
                // The intercept carries the threshold, and the sign is the
                // other way round: a common trait needs a low bar.
                start[1] = threshold;
                start
            })
            .collect();

        let value_of = |theta: &[f64]| -> f64 {
            self.loglik(theta[0], &theta[1..]).map_or(1e30, |v| -v)
        };
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
                optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control)
            else {
                continue;
            };
            let value = value_of(&solution.par);
            if value.is_finite() && best.as_ref().is_none_or(|(seen, _)| value < *seen) {
                best = Some((value, solution.par));
            }
        }
        let (negative, theta) = best.ok_or("LIABILITY_NO_START_CONVERGED")?;

        let gradient = gradient_of(&theta);
        let projected = gradient
            .iter()
            .enumerate()
            .map(|(k, g)| {
                if (k == 0 && theta[0] <= 0.0 && *g > 0.0)
                    || (k == 0 && theta[0] >= 1.0 && *g < 0.0)
                {
                    0.0
                } else {
                    *g
                }
            })
            .fold(0.0f64, |worst, g| worst.max(g.abs()));
        let scaled_gradient = projected / negative.abs().max(1.0);

        Ok(LiabilityFit {
            heritability: theta[0],
            fixed_effects: theta[1..].to_vec(),
            loglik: -negative,
            converged: scaled_gradient < 1e-5,
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
    let conditional_sd = (1.0 - rho * rho).sqrt();
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

#[cfg(test)]
mod tests {
    use super::{LiabilityModel, bivariate_normal_cdf, mendell_elston};
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
            let got = bivariate_normal_cdf(a, b, 0.0, &normal);
            assert!((got - product).abs() < 1e-9, "independence: {got} against {product}");
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
    fn simulate(heritability: f64, prevalence: f64, seed: u64)
        -> (DMatrix<f64>, Vec<f64>, DMatrix<f64>) {
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
        assert!(fit.converged, "did not converge, |g| = {}", fit.scaled_gradient);
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
