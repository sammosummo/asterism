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

use nalgebra::{Cholesky, DMatrix, DVector, SymmetricEigen};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};
use statrs::distribution::Normal;

use crate::blocks::family_blocks;
use crate::liability::LiabilityModel;

const LOG_TWO_PI: f64 = 1.837_877_066_409_345_3;
/// Below this share of measured values the scale is not identified: the
/// uncensored values are what put the trait on a scale at all.
const FEWEST_MEASURED: usize = 2;
/// What the objective returns where the likelihood cannot be evaluated at all,
/// which happens when a family block will not factorise. It is a large finite
/// number rather than an infinity because the optimiser has to be able to work
/// with it. Because it is finite, a search that never left it would otherwise
/// be accepted as a fit: the acceptance test compares against this name rather
/// than asking whether the objective is finite, which it always is.
const INFEASIBLE: f64 = 1e30;
/// How far below nought an eigenvalue of a family's relationship block may sit
/// before the matrix is refused rather than repaired. The same floor the
/// prepared model uses, for the same reason: rounding puts a genuine nought a
/// little either side of itself, and anything past this is a matrix that is
/// not a covariance.
const EIGENVALUE_FLOOR: f64 = -1e-9;

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

/// Chi-square on one degree of freedom at 0.95.
const CHI2_ONE_95: f64 = 3.841_458_820_694_124;
/// The Self-Liang 50:50 critical value. Under a null that sits on the
/// parameter's bound, half the reference distribution's mass is at nought, so
/// the correct threshold is not the plain chi-square one above. ADR 0004
/// records what the difference costs: taking an end of nought to mean the
/// interval contains nought gives 0.977 coverage at a true heritability of
/// nought, where this gives 0.953.
const MIXTURE_CRIT: f64 = 2.705_543_454_095_404;

/// The result of testing the heritability against nought.
#[derive(Clone, Debug)]
pub struct TobitTest {
    pub statistic: f64,
    pub p_value: f64,
    /// The reference distribution the p-value was read against, as data on the
    /// record rather than as a contract term.
    pub rule: &'static str,
    pub null_loglik: f64,
    pub alternative_loglik: f64,
}

/// A profile-likelihood interval for the heritability.
#[derive(Clone, Debug)]
pub struct TobitInterval {
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
    /// True where the end sits on the parameter's own bound rather than where
    /// the profile fell away -- the data did not rule that end out.
    pub lower_at_bound: bool,
    pub upper_at_bound: bool,
    pub level: f64,
    /// Whether a boundary point belongs to the interval, decided by the
    /// Self-Liang mixture rather than by the end having landed on the bound.
    /// Present only where the corresponding end is on its bound and the fit
    /// there could be made; absent means nobody measured it, not that the
    /// question does not apply.
    pub contains_lower_bound: Option<bool>,
    pub contains_upper_bound: Option<bool>,
    /// How many profile fits failed or did not converge. Each one widened the
    /// interval rather than narrowing it, which is the safe direction, but a
    /// large count means the interval rests on fewer points than it looks.
    pub profile_failures: usize,
    /// Reported beside the interval, because how far the model can be trusted
    /// depends on it.
    pub censored_share: f64,
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
        // **A relationship matrix is refused as mathematics, not as
        // provenance.** Symmetry and the off-diagonal guard above do not make a
        // matrix a covariance: one that is not positive semi-definite reaches
        // the search, where it surfaces only as a fit that found no feasible
        // point. Checking it here says what is wrong while the caller can still
        // do something about it. Per block, because that is the only form the
        // likelihood ever factorises.
        let blocks = family_blocks(relationship);
        for block in &blocks {
            let size = block.len();
            let sub = DMatrix::from_fn(size, size, |a, b| relationship[(block[a], block[b])]);
            let smallest = SymmetricEigen::new(sub)
                .eigenvalues
                .iter()
                .fold(f64::INFINITY, |worst, value| worst.min(*value));
            if smallest < EIGENVALUE_FLOOR {
                return Err("TOBIT_RELATIONSHIP_NOT_PSD");
            }
        }

        Ok(Self {
            relationship: relationship.clone(),
            design: design.clone(),
            value: value.to_vec(),
            censoring: censoring.to_vec(),
            limit: limit.to_vec(),
            blocks,
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
        self.fit_holding(None)
    }

    /// A 95 per cent profile-likelihood interval for the heritability.
    ///
    /// **The maximum comes from the held fit at the estimate**, not from the
    /// free fit's own log likelihood, so that both ends of the comparison are
    /// computed the same way and a difference between them is the profile
    /// falling away rather than two searches disagreeing.
    ///
    /// **A fit that failed, or stopped without converging, is not a likelihood
    /// that fell away.** Counting one as outside would look to the bisection
    /// like ground the data had ruled out, and the interval would come back
    /// narrower than the data support while saying nothing about it. They are
    /// counted instead, and reported, and the interval widens over them.
    ///
    /// Read `censored_share` beside the result. The calibration in
    /// `checks/tobit_calibration.py` recovers the truth to three quarters
    /// censored on simulated data, but on real extended high-frequency
    /// thresholds the model degrades past about half.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the free fit fails.
    /// The heritability against nought.
    ///
    /// The null sits on the parameter's bound, so the reference is the
    /// Self-Liang 50:50 mixture of chi-square on nought and one degrees of
    /// freedom and not a plain chi-square. That is the same null ADR 0004
    /// calibrates the interval's boundary point against, and the same rule the
    /// liability model reports for the same reason.
    ///
    /// The statistic and the p-value both come from `deviance`, which honours
    /// the point mass at nought: two searches that land on the same likelihood
    /// give a deviance that is rounding rather than evidence, and that reads as
    /// a p-value of one rather than of a half.
    pub fn heritability_test(&self) -> Result<TobitTest, &'static str> {
        let free = self.fit()?;
        let null = self.fit_holding(Some(0.0))?;
        let statistic = crate::deviance::deviance(free.loglik, null.loglik);
        Ok(TobitTest {
            statistic,
            p_value: crate::deviance::p_value(statistic, |value| {
                0.5 * crate::deviance::chi2_one_df_upper_tail(value)
            }),
            rule: "mixture_50_50",
            null_loglik: null.loglik,
            alternative_loglik: free.loglik,
        })
    }

    pub fn heritability_interval(&self) -> Result<TobitInterval, &'static str> {
        let free = self.fit()?;
        let estimate = free.heritability;
        let at_estimate = self.fit_holding(Some(estimate))?.loglik;

        let failures = std::cell::Cell::new(0usize);
        // The deviance rather than a bare verdict, because the boundary rule
        // below needs the number and not only whether it crossed.
        let deviance_at = |value: f64| -> Option<f64> {
            match self.fit_holding(Some(value)) {
                Ok(fit) if fit.converged => Some(2.0 * (at_estimate - fit.loglik)),
                _ => {
                    failures.set(failures.get() + 1);
                    None
                }
            }
        };
        let outside = |value: f64| deviance_at(value).is_some_and(|d| d > CHI2_ONE_95);

        // Each bound is fitted once and the answer used twice: to place the
        // end, and to decide whether the bound itself belongs to the interval.
        let at_zero = deviance_at(0.0);
        let at_one = deviance_at(1.0);

        let (lower, lower_at_bound) = if at_zero.is_some_and(|d| d > CHI2_ONE_95) {
            (crate::liability::bisect(0.0, estimate, &outside), false)
        } else {
            (0.0, true)
        };
        let (upper, upper_at_bound) = if at_one.is_some_and(|d| d > CHI2_ONE_95) {
            (crate::liability::bisect(1.0, estimate, &outside), false)
        } else {
            (1.0, true)
        };
        Ok(TobitInterval {
            estimate,
            lower,
            upper,
            lower_at_bound,
            upper_at_bound,
            level: 0.95,
            contains_lower_bound: if lower == 0.0 {
                at_zero.map(|d| d <= MIXTURE_CRIT)
            } else {
                None
            },
            contains_upper_bound: if upper == 1.0 {
                at_one.map(|d| d <= MIXTURE_CRIT)
            } else {
                None
            },
            profile_failures: failures.get(),
            censored_share: self.censored_share(),
        })
    }

    /// Fit with the heritability held, or free where `held` is `None`.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start converges.
    pub fn fit_holding(&self, held: Option<f64>) -> Result<TobitFit, &'static str> {
        if let Some(value) = held {
            if !(0.0..=1.0).contains(&value) {
                return Err("TOBIT_HELD_HERITABILITY_OUT_OF_RANGE");
            }
        }
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
        let spread = (measured
            .iter()
            .map(|v| (v - centre).powi(2))
            .sum::<f64>()
            / measured.len() as f64)
            .max(1e-12);

        let mut lower = vec![f64::NEG_INFINITY; count];
        let mut upper = vec![f64::INFINITY; count];
        lower[0] = 0.0;
        upper[0] = 1.0;
        if let Some(value) = held {
            lower[0] = value;
            upper[0] = value;
        }

        let value_of = |theta: &[f64]| -> f64 {
            self.loglik(theta[0], theta[1].exp(), &theta[2..])
                .map_or(INFEASIBLE, |v| -v)
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

        // Three starts spread over the heritability, or one where it is held.
        // A held fit pinned every start to the same value and then ran the same
        // deterministic search three times over: the interval paid for that at
        // each end of every bisection step.
        let starts: &[f64] = match held {
            Some(value) => &[value],
            None => &[0.05, 0.3, 0.6],
        };
        let mut best: Option<(f64, Vec<f64>)> = None;
        for &heritability in starts {
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
            if objective < INFEASIBLE && best.as_ref().is_none_or(|(seen, _)| objective < *seen) {
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

    /// Produced by the three-start code this replaced, at the same seed and
    /// the same held value.
    const PINNED_HELD_LOGLIK: f64 = -102.736_491_872_357_05;

    /// A relationship matrix that is not a covariance is refused where the
    /// caller can still do something about it.
    ///
    /// Symmetry and the off-diagonal guard do not make a matrix positive
    /// semi-definite. A nought diagonal beside a positive off-diagonal is the
    /// simplest case: `[[0, 0.9], [0.9, 0]]` has eigenvalues of plus and minus
    /// 0.9, so it passed every check `build` used to make and then surfaced
    /// only as a fit that could find no feasible point.
    #[test]
    fn a_relationship_matrix_that_is_not_positive_semi_definite_is_refused() {
        let people = 2;
        let mut relationship = DMatrix::zeros(people, people);
        relationship[(0, 1)] = 0.9;
        relationship[(1, 0)] = 0.9;

        let value = vec![0.4, -0.2];
        let censoring = vec![Censoring::Measured, Censoring::Measured];
        let limit = vec![2.0; people];
        let design = DMatrix::from_element(people, 1, 1.0);

        match TobitModel::build(&relationship, &value, &censoring, &limit, &design) {
            Err(code) => assert_eq!(code, "TOBIT_RELATIONSHIP_NOT_PSD"),
            Ok(_) => panic!("a matrix with a negative eigenvalue was accepted as a covariance"),
        }
    }

    /// Holding the heritability changes the cost of a fit and not its answer.
    ///
    /// A held fit used to build the same start three times and run the same
    /// deterministic search over each, so the interval paid for three copies of
    /// one answer at each end of every bisection step. The value pinned above
    /// came from that three-start code; taking one start has to land on it
    /// exactly rather than merely close to it.
    #[test]
    fn holding_a_heritability_costs_one_search_and_lands_where_three_did() {
        let (relationship, value, censoring, limit, design) =
            simulate(40, 0.5, 1.0, 0.0, Some(0.75), 909);
        let model = TobitModel::build(&relationship, &value, &censoring, &limit, &design)
            .expect("the simulated matrix is a covariance");

        let fit = model
            .fit_holding(Some(0.4))
            .expect("a held fit converges here");

        assert_eq!(
            fit.loglik, PINNED_HELD_LOGLIK,
            "one start did not land where three identical starts did"
        );
    }

    /// A fit that never found a feasible point must say so rather than come
    /// back converged.
    ///
    /// Each family block here is positive semi-definite and singular -- both
    /// eigenvalues of `[[0.9, 0.9], [0.9, 0.9]]` are 1.8 and nought -- so it
    /// passes every check `build` makes, including the positive
    /// semi-definiteness one. Holding the heritability at one makes the
    /// covariance that matrix exactly, and a singular matrix has no Cholesky
    /// factor, so every start is infeasible.
    ///
    /// Before the objective's infeasible value was given a name and tested
    /// against, this returned `Ok` with `converged: true` and a log likelihood
    /// of -1e30, which `heritability_interval` then read as ground the data
    /// had ruled out.
    #[test]
    fn a_fit_with_no_feasible_start_is_refused_rather_than_reported() {
        let people = 4;
        let mut relationship = DMatrix::zeros(people, people);
        for family in 0..2 {
            let (a, b) = (2 * family, 2 * family + 1);
            relationship[(a, a)] = 0.9;
            relationship[(b, b)] = 0.9;
            relationship[(a, b)] = 0.9;
            relationship[(b, a)] = 0.9;
        }

        let value = vec![0.4, -0.2, 0.9, -0.6];
        let censoring = vec![
            Censoring::Measured,
            Censoring::Measured,
            Censoring::Measured,
            Censoring::Measured,
        ];
        let limit = vec![2.0; people];
        let design = DMatrix::from_element(people, 1, 1.0);

        let model = TobitModel::build(&relationship, &value, &censoring, &limit, &design)
            .expect("a nought diagonal is not something build refuses");

        assert!(
            model.loglik(1.0, 1.0, &[0.0]).is_none(),
            "the covariance at a held heritability of one should not factorise"
        );

        match model.fit_holding(Some(1.0)) {
            Err(code) => assert_eq!(code, "TOBIT_NO_START_CONVERGED"),
            Ok(fit) => panic!(
                "a fit with no feasible start came back converged={} at a log likelihood of {}",
                fit.converged, fit.loglik
            ),
        }
    }

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

    /// The profile interval brackets the estimate and reaches where the
    /// likelihood actually falls away.
    ///
    /// Two things are checked that a plausible-looking interval can still get
    /// wrong: the estimate must lie inside its own interval, and holding the
    /// heritability at either end must cost about half a chi-square on one --
    /// which is what the interval claims about itself and is otherwise only
    /// asserted.
    #[test]
    fn the_profile_interval_reaches_where_the_likelihood_falls_away() {
        let (relationship, value, censoring, limit, design) = simulate(
            300, 0.5, 4.0, 10.0, Some(11.0), 20_260_818);
        let model = TobitModel::build(&relationship, &value, &censoring, &limit, &design)
            .expect("builds");
        let got = model.heritability_interval().expect("intervals");

        assert!(
            got.lower <= got.estimate && got.estimate <= got.upper,
            "the estimate {} is outside its own interval [{}, {}]",
            got.estimate, got.lower, got.upper
        );
        assert!((got.level - 0.95).abs() < 1e-12);

        let peak = model.fit_holding(Some(got.estimate)).expect("held fit").loglik;
        for (name, end, at_bound) in [
            ("lower", got.lower, got.lower_at_bound),
            ("upper", got.upper, got.upper_at_bound),
        ] {
            if at_bound {
                continue;   // the data did not rule that end out
            }
            let there = model.fit_holding(Some(end)).expect("held fit").loglik;
            let cost = 2.0 * (peak - there);
            assert!(
                (cost - CHI2_ONE_95).abs() < 0.05,
                "the {name} end costs {cost} in deviance, not the {CHI2_ONE_95} \
                 an interval at this level claims"
            );
        }
    }

    /// A heritability held outside its range is refused rather than clamped.
    #[test]
    fn a_held_heritability_outside_the_range_is_refused() {
        let (relationship, value, censoring, limit, design) =
            simulate(40, 0.5, 1.0, 0.0, None, 3);
        let model = TobitModel::build(&relationship, &value, &censoring, &limit, &design)
            .expect("builds");
        assert_eq!(
            model.fit_holding(Some(1.5)).err(),
            Some("TOBIT_HELD_HERITABILITY_OUT_OF_RANGE")
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
