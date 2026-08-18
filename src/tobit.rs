//! One trait whose measurement stops at a limit -- a censored variance
//! components model.
//!
//! **This is a special case rather than part of the Gaussian general model.**
//! Nothing here is folded into `ComponentModel`, for decision 16's reasons:
//! REML has no meaning once part of the likelihood is a probability rather than
//! a density, and the fixed effects cannot be profiled out by least squares.
//!
//! # The model
//!
//! Every person carries a complete value
//!
//! ```text
//! y*_i = x_i' beta + g_i + e_i
//! ```
//!
//! with `g ~ N(0, h2 sigma^2 A)` and `e ~ N(0, (1 - h2) sigma^2 I)`. Where the
//! instrument could measure it, `y*_i` is seen exactly. Where it could not, all
//! that is known is that the value lies beyond the limit that person's
//! measurement reached.
//!
//! # What separates this from the liability model
//!
//! The liability model sees only a sign, so its variance has no scale of its
//! own and is fixed at one. **Here the scale is identified**, because the
//! uncensored values arrive on it. So `sigma^2` is estimated, and the
//! heritability that comes back is the heritability of the complete variable --
//! the number you would have had if the instrument reached far enough.
//!
//! That makes it directly comparable with a Gaussian heritability of an
//! uncensored trait, and **not** comparable with a Gaussian heritability
//! fitted to values where the censored ones were replaced by the limit. The
//! second is the usual practice and it is the thing this exists to replace.
//!
//! # The likelihood
//!
//! A family splits into the people whose value was measured and the people
//! whose was not. The measured ones contribute an ordinary multivariate normal
//! density. The unmeasured ones contribute the probability that their values
//! lie beyond their limits, **conditional on the measured ones** -- not
//! marginally, because within a family the two are correlated and that
//! correlation is the whole of the information a pedigree carries.
//!
//! The region probability is the one `LiabilityModel` already uses: exact to
//! two people, and the Mendell-Elston sequential truncation above that, with
//! the rarer side taken first. Sharing it is deliberate. A second
//! implementation of the same integral is how a package comes to hold two
//! different answers.
//!
//! # Per-observation limits
//!
//! The limit belongs to the observation, not to the trait. Extended
//! high-frequency audiometry is the case this was built for, and there the
//! recorded maximum differs between frequencies and between sessions, so a
//! censored value can carry the same number as a genuinely measured one. Only
//! the status distinguishes them, which is why the status is a separate input
//! and is never inferred from the value.
//!
//! # Maximum likelihood, and saying so
//!
//! There is no REML here. REML removes the fixed effects by projecting the
//! response onto the null space of the design, and a censored observation has
//! no response to project. The fit record says `ml`, and a censored
//! heritability must not be placed beside a REML one as though the two were
//! the same number.

use nalgebra::{Cholesky, DMatrix, DVector};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};
use statrs::distribution::Normal;

use crate::blocks::family_blocks;
use crate::liability::LiabilityModel;

const LOG_TWO_PI: f64 = 1.837_877_066_409_345_3;
/// Below this share of measured values the scale is not identified: the
/// uncensored values are what put the trait on a scale at all.
const FEWEST_MEASURED: usize = 2;

/// Which way an unmeasured value lies from its limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Censoring {
    /// The value was measured.
    Measured,
    /// The value is at or above the limit -- an instrument that ran out of
    /// output, which is what an audiometer does.
    Above,
    /// The value is at or below the limit -- an assay under its detection
    /// limit.
    Below,
}

/// A fitted censored model.
#[derive(Clone, Debug)]
pub struct TobitFit {
    /// The heritability **of the complete variable**, not of the values that
    /// happened to be measurable.
    pub heritability: f64,
    /// The total variance of the complete variable, which a liability model
    /// cannot give you.
    pub total_variance: f64,
    pub fixed_effects: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    pub scaled_gradient: f64,
    /// The share of observations that hit a limit. Read the heritability
    /// against it: at a very high share the upper tail is carried by the
    /// censoring pattern alone.
    pub censored_share: f64,
    /// Always `ml`. See the module note.
    pub estimator: &'static str,
    /// The largest family, because the region probability is exact to two
    /// people and approximate above it.
    pub largest_family: usize,
}

/// One trait, one relationship matrix, per-observation censoring.
pub struct TobitModel {
    relationship: DMatrix<f64>,
    design: DMatrix<f64>,
    value: Vec<f64>,
    censoring: Vec<Censoring>,
    limit: Vec<f64>,
    blocks: Vec<Vec<usize>>,
    rows: usize,
}

impl TobitModel {
    /// Validate and prepare.
    ///
    /// `value` holds the measured value where `censoring` is `Measured`, and is
    /// ignored otherwise. `limit` holds the limit that observation reached, and
    /// is ignored where the value was measured.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(
        relationship: &DMatrix<f64>,
        value: &[f64],
        censoring: &[Censoring],
        limit: &[f64],
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let rows = value.len();
        if rows == 0 {
            return Err("TOBIT_NO_ROWS");
        }
        if censoring.len() != rows || limit.len() != rows {
            return Err("TOBIT_INPUTS_DIFFERENT_LENGTHS");
        }
        if relationship.nrows() != rows || relationship.ncols() != rows {
            return Err("TOBIT_RELATIONSHIP_WRONG_SHAPE");
        }
        if design.nrows() != rows {
            return Err("TOBIT_DESIGN_WRONG_SHAPE");
        }
        if design.ncols() == 0 {
            return Err("TOBIT_DESIGN_HAS_NO_COLUMNS");
        }
        if !relationship.iter().all(|v| v.is_finite()) || !design.iter().all(|v| v.is_finite()) {
            return Err("TOBIT_NOT_FINITE");
        }
        for i in 0..rows {
            for j in 0..i {
                if (relationship[(i, j)] - relationship[(j, i)]).abs() > 1e-10 {
                    return Err("TOBIT_RELATIONSHIP_NOT_SYMMETRIC");
                }
            }
        }
        // The same guard the liability model carries, for the same reason: the
        // two-person quadrature loses accuracy as the correlation approaches
        // one, and a relationship of one produces that at a high heritability.
        for i in 0..rows {
            for j in 0..i {
                if relationship[(i, j)].abs() > 0.9 {
                    return Err("TOBIT_RELATIONSHIP_TOO_CLOSE");
                }
            }
        }
        for index in 0..rows {
            match censoring[index] {
                Censoring::Measured => {
                    if !value[index].is_finite() {
                        return Err("TOBIT_MEASURED_VALUE_NOT_FINITE");
                    }
                }
                Censoring::Above | Censoring::Below => {
                    if !limit[index].is_finite() {
                        return Err("TOBIT_LIMIT_NOT_FINITE");
                    }
                }
            }
        }
        let measured = censoring
            .iter()
            .filter(|c| **c == Censoring::Measured)
            .count();
        // **A trait with nothing measured has no scale.** The liability model
        // is the right model for that, and it says so by fixing the variance
        // at one; this one would return a variance the data cannot support.
        if measured < FEWEST_MEASURED {
            return Err("TOBIT_TOO_FEW_MEASURED_VALUES");
        }
        Ok(Self {
            relationship: relationship.clone(),
            design: design.clone(),
            value: value.to_vec(),
            censoring: censoring.to_vec(),
            limit: limit.to_vec(),
            blocks: family_blocks(relationship),
            rows,
        })
    }

    /// The share of observations that hit a limit.
    #[must_use]
    pub fn censored_share(&self) -> f64 {
        let censored = self
            .censoring
            .iter()
            .filter(|c| **c != Censoring::Measured)
            .count();
        censored as f64 / self.rows as f64
    }

    /// The log likelihood at one heritability, one variance and one set of
    /// fixed effects.
    ///
    /// Returns `None` where the parameters do not describe a model, which the
    /// optimiser reads as a wall rather than a failure.
    #[must_use]
    pub fn loglik(&self, heritability: f64, variance: f64, beta: &[f64]) -> Option<f64> {
        if !(0.0..=1.0).contains(&heritability) || !(variance > 0.0) || !variance.is_finite() {
            return None;
        }
        if beta.len() != self.design.ncols() || !beta.iter().all(|b| b.is_finite()) {
            return None;
        }
        let normal = Normal::new(0.0, 1.0).ok()?;
        let coefficients = DVector::from_row_slice(beta);
        let mean = &self.design * &coefficients;

        let mut total = 0.0;
        for block in &self.blocks {
            let size = block.len();
            // The family covariance, on the complete variable's own scale.
            let mut covariance = DMatrix::<f64>::zeros(size, size);
            for (row, &i) in block.iter().enumerate() {
                for (column, &j) in block.iter().enumerate() {
                    let additive = heritability * self.relationship[(i, j)];
                    let residual = if i == j { 1.0 - heritability } else { 0.0 };
                    covariance[(row, column)] = variance * (additive + residual);
                }
            }

            let measured: Vec<usize> = (0..size)
                .filter(|&r| self.censoring[block[r]] == Censoring::Measured)
                .collect();
            let censored: Vec<usize> = (0..size)
                .filter(|&r| self.censoring[block[r]] != Censoring::Measured)
                .collect();

            // The measured people contribute an ordinary density.
            let (residual_measured, conditional_mean_shift) = if measured.is_empty() {
                (None, None)
            } else {
                let sub = DMatrix::from_fn(measured.len(), measured.len(), |a, b| {
                    covariance[(measured[a], measured[b])]
                });
                let chol = Cholesky::new(sub)?;
                let deviation = DVector::from_fn(measured.len(), |a, _| {
                    self.value[block[measured[a]]] - mean[block[measured[a]]]
                });
                let solved = chol.solve(&deviation);
                let quadratic = deviation.dot(&solved);
                let log_determinant = 2.0
                    * chol
                        .l()
                        .diagonal()
                        .iter()
                        .map(|d| d.ln())
                        .sum::<f64>();
                let density = -0.5
                    * (measured.len() as f64 * LOG_TWO_PI + log_determinant + quadratic);
                if !density.is_finite() {
                    return None;
                }
                total += density;
                (Some(chol), Some(solved))
            };

            if censored.is_empty() {
                continue;
            }

            // The censored people contribute the probability that their values
            // lie beyond their limits, given what was measured.
            let mut region_mean = Vec::with_capacity(censored.len());
            let mut sign = Vec::with_capacity(censored.len());
            for &c in &censored {
                let person = block[c];
                let mut expected = mean[person];
                if let (Some(chol), Some(solved)) = (&residual_measured, &conditional_mean_shift) {
                    let _ = chol;
                    // V_CO V_OO^-1 (y_O - mu_O), assembled row by row.
                    let mut shift = 0.0;
                    for (a, &m) in measured.iter().enumerate() {
                        shift += covariance[(c, m)] * solved[a];
                    }
                    expected += shift;
                }
                // Centre on this observation's own limit, which is what makes
                // the shared region routine apply: it asks about a region
                // around nought.
                region_mean.push(expected - self.limit[person]);
                sign.push(match self.censoring[person] {
                    Censoring::Above => 1.0,
                    Censoring::Below => -1.0,
                    Censoring::Measured => unreachable!("filtered above"),
                });
            }

            let mut conditional = DMatrix::from_fn(censored.len(), censored.len(), |a, b| {
                covariance[(censored[a], censored[b])]
            });
            if let Some(chol) = &residual_measured {
                // Sigma_CC - V_CO V_OO^-1 V_OC.
                let cross = DMatrix::from_fn(measured.len(), censored.len(), |a, b| {
                    covariance[(measured[a], censored[b])]
                });
                let solved = chol.solve(&cross);
                conditional -= cross.transpose() * solved;
            }

            let probability =
                LiabilityModel::region_log_probability(&region_mean, &sign, &conditional, &normal)?;
            if !probability.is_finite() {
                return None;
            }
            total += probability;
        }
        total.is_finite().then_some(total)
    }

    /// Fit by bounded search from several starts.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start converges.
    pub fn fit(&self) -> Result<TobitFit, &'static str> {
        let columns = self.design.ncols();
        let count = columns + 2;

        // Starting values from the measured part alone. They are wrong -- that
        // is the whole point of the model -- but they are the right order of
        // magnitude, which is all a start has to be.
        let measured: Vec<f64> = (0..self.rows)
            .filter(|&i| self.censoring[i] == Censoring::Measured)
            .map(|i| self.value[i])
            .collect();
        let centre = measured.iter().sum::<f64>() / measured.len() as f64;
        let spread = measured
            .iter()
            .map(|v| (v - centre).powi(2))
            .sum::<f64>()
            .max(1e-12)
            / measured.len() as f64;

        let mut lower = vec![f64::NEG_INFINITY; count];
        let mut upper = vec![f64::INFINITY; count];
        lower[0] = 0.0;
        upper[0] = 1.0;

        let value_of = |theta: &[f64]| -> f64 {
            self.loglik(theta[0], theta[1].exp(), &theta[2..])
                .map_or(1e30, |v| -v)
        };
        // The region probability has no derivative worth writing, so the
        // gradient is a central difference, as the liability model's is.
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
        for heritability in [0.05_f64, 0.3, 0.6] {
            let mut start = vec![0.0; count];
            start[0] = heritability;
            start[1] = spread.ln();
            start[2] = centre;
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
            let objective = value_of(&solution.par);
            if objective.is_finite() && best.as_ref().is_none_or(|(seen, _)| objective < *seen) {
                best = Some((objective, solution.par));
            }
        }

        let (objective, theta) = best.ok_or("TOBIT_NO_START_CONVERGED")?;
        let gradient = gradient_of(&theta);
        let scaled_gradient = gradient
            .iter()
            .fold(0.0_f64, |worst, g| worst.max(g.abs()));
        Ok(TobitFit {
            heritability: theta[0],
            total_variance: theta[1].exp(),
            fixed_effects: theta[2..].to_vec(),
            loglik: -objective,
            converged: scaled_gradient < 1e-3,
            scaled_gradient,
            censored_share: self.censored_share(),
            estimator: "ml",
            largest_family: self.blocks.iter().map(Vec::len).max().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sibling pairs, a known heritability, and a censoring limit applied to
    /// the top share of the complete values.
    ///
    /// Returns the relationship, the values as the instrument would record
    /// them, the censoring status, the limits, and the design.
    fn simulate(
        pairs: usize,
        heritability: f64,
        variance: f64,
        mean: f64,
        censor_above: Option<f64>,
        seed: u64,
    ) -> (
        DMatrix<f64>,
        Vec<f64>,
        Vec<Censoring>,
        Vec<f64>,
        DMatrix<f64>,
    ) {
        let n = 2 * pairs;
        let mut relationship = DMatrix::<f64>::identity(n, n);
        for pair in 0..pairs {
            relationship[(2 * pair, 2 * pair + 1)] = 0.5;
            relationship[(2 * pair + 1, 2 * pair)] = 0.5;
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
        let scale = variance.sqrt();
        let shared = (heritability / 2.0).sqrt();
        let own = (1.0 - heritability / 2.0).sqrt();

        let mut value = Vec::with_capacity(n);
        let mut censoring = Vec::with_capacity(n);
        let mut limit = Vec::with_capacity(n);
        for _ in 0..pairs {
            let common = normal_draw() * shared;
            for _ in 0..2 {
                let complete = mean + scale * (common + normal_draw() * own);
                match censor_above {
                    Some(cut) if complete >= cut => {
                        value.push(f64::NAN);
                        censoring.push(Censoring::Above);
                        limit.push(cut);
                    }
                    _ => {
                        value.push(complete);
                        censoring.push(Censoring::Measured);
                        limit.push(f64::NAN);
                    }
                }
            }
        }
        (
            relationship,
            value,
            censoring,
            limit,
            DMatrix::from_element(n, 1, 1.0),
        )
    }

    /// With nothing censored the likelihood is an ordinary multivariate normal
    /// density, and it must equal one computed a different way.
    ///
    /// This is the hinge the whole model turns on. If the uncensored case does
    /// not reproduce the Gaussian likelihood then the censored case is not a
    /// generalisation of it, whatever else it may be.
    #[test]
    fn with_nothing_censored_it_is_the_gaussian_likelihood() {
        let (relationship, value, censoring, limit, design) =
            simulate(60, 0.5, 2.0, 3.0, None, 20_260_818);
        let model = TobitModel::build(&relationship, &value, &censoring, &limit, &design)
            .expect("builds");
        let (heritability, variance, mean) = (0.4, 1.7, 2.5);
        let ours = model
            .loglik(heritability, variance, &[mean])
            .expect("evaluates");

        // The same number, assembled directly, one family at a time.
        let mut theirs = 0.0;
        for pair in 0..60 {
            let block = [2 * pair, 2 * pair + 1];
            let mut covariance = DMatrix::<f64>::zeros(2, 2);
            for (r, &i) in block.iter().enumerate() {
                for (c, &j) in block.iter().enumerate() {
                    let additive = heritability * relationship[(i, j)];
                    let residual = if i == j { 1.0 - heritability } else { 0.0 };
                    covariance[(r, c)] = variance * (additive + residual);
                }
            }
            let deviation =
                DVector::from_fn(2, |r, _| value[block[r]] - mean);
            let determinant =
                covariance[(0, 0)] * covariance[(1, 1)] - covariance[(0, 1)].powi(2);
            let inverse = DMatrix::from_row_slice(
                2,
                2,
                &[
                    covariance[(1, 1)] / determinant,
                    -covariance[(0, 1)] / determinant,
                    -covariance[(1, 0)] / determinant,
                    covariance[(0, 0)] / determinant,
                ],
            );
            let quadratic = deviation.dot(&(&inverse * &deviation));
            theirs += -0.5 * (2.0 * LOG_TWO_PI + determinant.ln() + quadratic);
        }
        assert!(
            (ours - theirs).abs() < 1e-9,
            "uncensored likelihood is not the Gaussian one: {ours} against {theirs}"
        );
    }

    /// Censoring is handled, and substituting the limit is not the same thing.
    ///
    /// The comparison is the point. Replacing an unmeasurable value with the
    /// limit is what the audiometry literature does, and it compresses the
    /// upper tail: the variance falls, and with it the additive share. The
    /// censored likelihood should sit closer to the truth than the substituted
    /// fit does.
    #[test]
    fn a_censored_fit_beats_substituting_the_limit() {
        let (truth_h2, truth_variance, truth_mean) = (0.6_f64, 4.0_f64, 10.0_f64);
        // A limit about one standard deviation above the mean censors roughly a
        // sixth of the sample, which is the order extended high frequency
        // audiometry sees.
        let cut = truth_mean + truth_variance.sqrt();
        let (relationship, value, censoring, limit, design) = simulate(
            400,
            truth_h2,
            truth_variance,
            truth_mean,
            Some(cut),
            20_260_819,
        );
        let censored = censoring.iter().filter(|c| **c != Censoring::Measured).count();
        assert!(
            censored > 40,
            "the fixture censored {censored} values, too few to test with"
        );

        let model = TobitModel::build(&relationship, &value, &censoring, &limit, &design)
            .expect("builds");
        let fit = model.fit().expect("fits");
        assert_eq!(fit.estimator, "ml");

        // The same data with every censored value replaced by the limit, fitted
        // as though it were complete.
        let substituted: Vec<f64> = value
            .iter()
            .zip(&censoring)
            .map(|(v, c)| if *c == Censoring::Measured { *v } else { cut })
            .collect();
        let all_measured = vec![Censoring::Measured; substituted.len()];
        let no_limits = vec![f64::NAN; substituted.len()];
        let naive = TobitModel::build(
            &relationship,
            &substituted,
            &all_measured,
            &no_limits,
            &design,
        )
        .expect("builds")
        .fit()
        .expect("fits");

        assert!(
            naive.total_variance < truth_variance,
            "substitution did not compress the variance: {} against a true {truth_variance}",
            naive.total_variance
        );
        assert!(
            (fit.total_variance - truth_variance).abs()
                < (naive.total_variance - truth_variance).abs(),
            "the censored fit recovered the variance no better than substitution: \
             {} against {} at a true {truth_variance}",
            fit.total_variance,
            naive.total_variance
        );
    }

    /// A trait with nothing measured has no scale, and is refused rather than
    /// given one.
    #[test]
    fn a_trait_with_nothing_measured_is_refused() {
        let (relationship, _, _, _, design) = simulate(20, 0.5, 1.0, 0.0, None, 7);
        let n = 40;
        let censoring = vec![Censoring::Above; n];
        let limit = vec![0.0; n];
        let value = vec![f64::NAN; n];
        assert_eq!(
            TobitModel::build(&relationship, &value, &censoring, &limit, &design).err(),
            Some("TOBIT_TOO_FEW_MEASURED_VALUES")
        );
    }
}

#[cfg(feature = "python")]
pub mod python;
