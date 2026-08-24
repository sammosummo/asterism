//! Two traits whose measurements need not be of the same kind.
//!
//! Either trait may be continuous, binary through a liability threshold, or
//! censored at a limit, and the pair may be any combination of the three. The
//! analysis this was built for is the genetic correlation between a psychiatric
//! diagnosis and hearing, where one trait is a diagnosis and the other is an
//! audiometric threshold that the instrument could not always reach.
//!
//! **This is one model rather than nine special cases.** The covariance never
//! changes:
//!
//! ```text
//! Σ_A = [[h₁σ₁,           ρ_G√(h₁σ₁h₂σ₂)],
//!        [ρ_G√(h₁σ₁h₂σ₂), h₂σ₂          ]]
//! Σ_E = [[(1−h₁)σ₁,                ρ_E√((1−h₁)σ₁(1−h₂)σ₂)],
//!        [ρ_E√((1−h₁)σ₁(1−h₂)σ₂),  (1−h₂)σ₂              ]]
//!
//! V = Σ_A ⊗ A + Σ_E ⊗ I
//! ```
//!
//! which is `BivariateModel`'s covariance and the saturated benchmark's. What
//! varies is only how each observation is seen: a density at a point, or the
//! probability of a region. A continuous value is the degenerate region.
//!
//! # A binary trait has no scale
//!
//! Only the sign of a liability is ever seen, so its variance carries no
//! information and is fixed at one, exactly as `LiabilityModel` fixes it. A
//! continuous or censored trait keeps a free variance, because its measured
//! values arrive on a scale. **So the kind of the trait decides its bounds**,
//! and a heritability from a binary trait is a liability heritability while one
//! from the other two is not. Placing them side by side unlabelled is the
//! mistake this note exists to prevent.
//!
//! The genetic correlation is unaffected by that difference, which is what
//! makes the mixed pair worth fitting at all: a correlation is scale free even
//! where one of its two scales is arbitrary.
//!
//! # What it costs
//!
//! Every non-continuous observation adds a dimension to a region probability,
//! and the region is evaluated once per family. Exact to two, and the
//! Mendell-Elston sequential truncation above that -- the same routine the
//! liability and censored models use, for the same reason.
//!
//! # Maximum likelihood
//!
//! No REML, for the reason it is absent from every other non-Gaussian model
//! here: a region has no response to project onto the null space of the design.

use nalgebra::{Cholesky, DMatrix, DVector};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};
use statrs::distribution::Normal;

use crate::blocks::family_blocks;
use crate::interval::{self, Interval};
use crate::liability::LiabilityModel;
use crate::tobit::Censoring;

const LOG_TWO_PI: f64 = 1.837_877_066_409_345_3;

/// What kind of measurement a trait is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraitKind {
    /// Measured on a scale, every value seen.
    Continuous,
    /// Only the side of a threshold is seen, so the variance is fixed at one.
    Binary,
    /// Measured on a scale where the instrument reached, and beyond a limit
    /// where it did not.
    Censored,
}

/// One trait's observations.
pub struct TraitData {
    pub kind: TraitKind,
    /// The measured value where `censoring` says it was measured.
    pub value: Vec<f64>,
    /// For a binary trait, `Above` is a case and `Below` is not; the limit is
    /// the threshold, which is carried by the intercept and so is nought.
    pub censoring: Vec<Censoring>,
    pub limit: Vec<f64>,
}

/// A fitted mixed bivariate model.
#[derive(Clone, Debug)]
pub struct MixedBivariateFit {
    /// Per trait. For a binary trait this is a liability heritability.
    pub heritability: [f64; 2],
    /// Per trait. Fixed at one for a binary trait, where it means nothing.
    pub total_variance: [f64; 2],
    pub genetic_correlation: f64,
    pub residual_correlation: f64,
    pub fixed_effects: [Vec<f64>; 2],
    pub loglik: f64,
    pub converged: bool,
    pub scaled_gradient: f64,
    pub kinds: [TraitKind; 2],
    /// Always `ml`.
    pub estimator: &'static str,
    pub largest_family: usize,
}

/// Which coordinate an interval is for. The variances are deliberately absent:
/// a binary trait's is fixed at one and means nothing, so an interval on it
/// would be an interval on an assumption.
pub const HERITABILITY_ONE: usize = 0;
pub const HERITABILITY_TWO: usize = 1;
pub const GENETIC_CORRELATION: usize = 4;
pub const RESIDUAL_CORRELATION: usize = 5;

/// The result of testing one correlation against nought.
#[derive(Clone, Debug)]
pub struct MixedBivariateTest {
    /// Which correlation was tested, as data on the record.
    pub what: &'static str,
    pub statistic: f64,
    pub p_value: f64,
    /// The reference distribution the p-value was read against, as data on the
    /// record rather than as a contract term.
    pub rule: &'static str,
    pub null_loglik: f64,
    pub alternative_loglik: f64,
}

/// The common profile-likelihood interval record.
pub type MixedBivariateInterval = Interval;

/// Two traits, one relationship matrix, a measurement kind for each.
pub struct MixedBivariateModel {
    relationship: DMatrix<f64>,
    design: DMatrix<f64>,
    traits: [TraitData; 2],
    blocks: Vec<Vec<usize>>,
    rows: usize,
}

impl MixedBivariateModel {
    /// Validate and prepare.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(
        relationship: &DMatrix<f64>,
        first: TraitData,
        second: TraitData,
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let rows = relationship.nrows();
        if rows == 0 {
            return Err("MIXED_BIVARIATE_NO_ROWS");
        }
        if relationship.ncols() != rows || design.nrows() != rows {
            return Err("MIXED_BIVARIATE_WRONG_SHAPE");
        }
        if design.ncols() == 0 {
            return Err("MIXED_BIVARIATE_DESIGN_HAS_NO_COLUMNS");
        }
        if !relationship.iter().all(|value| value.is_finite())
            || !design.iter().all(|value| value.is_finite())
        {
            return Err("MIXED_BIVARIATE_NOT_FINITE");
        }
        for each in [&first, &second] {
            if each.value.len() != rows || each.censoring.len() != rows || each.limit.len() != rows
            {
                return Err("MIXED_BIVARIATE_TRAIT_WRONG_LENGTH");
            }
            for index in 0..rows {
                match each.censoring[index] {
                    Censoring::Measured if !each.value[index].is_finite() => {
                        return Err("MIXED_BIVARIATE_MEASURED_VALUE_NOT_FINITE");
                    }
                    Censoring::Above | Censoring::Below if !each.limit[index].is_finite() => {
                        return Err("MIXED_BIVARIATE_LIMIT_NOT_FINITE");
                    }
                    _ => {}
                }
            }
            match each.kind {
                // A binary trait is all region and no density, which is what
                // makes its scale unidentified; a measured value there would
                // mean the trait is not binary.
                TraitKind::Binary => {
                    if each.censoring.contains(&Censoring::Measured) {
                        return Err("MIXED_BIVARIATE_BINARY_TRAIT_HAS_A_MEASURED_VALUE");
                    }
                    let cases = each
                        .censoring
                        .iter()
                        .filter(|c| **c == Censoring::Above)
                        .count();
                    if cases == 0 || cases == rows {
                        return Err("MIXED_BIVARIATE_BINARY_TRAIT_ALL_ONE_STATUS");
                    }
                }
                // A continuous trait has no limits to respect.
                TraitKind::Continuous => {
                    if each.censoring.iter().any(|c| *c != Censoring::Measured) {
                        return Err("MIXED_BIVARIATE_CONTINUOUS_TRAIT_IS_CENSORED");
                    }
                }
                TraitKind::Censored => {
                    if !each.censoring.contains(&Censoring::Measured) {
                        return Err("MIXED_BIVARIATE_CENSORED_TRAIT_HAS_NO_SCALE");
                    }
                }
            }
        }
        for i in 0..rows {
            for j in 0..i {
                if (relationship[(i, j)] - relationship[(j, i)]).abs() > 1e-10 {
                    return Err("MIXED_BIVARIATE_RELATIONSHIP_NOT_SYMMETRIC");
                }
                if relationship[(i, j)].abs() > 0.9 {
                    return Err("MIXED_BIVARIATE_RELATIONSHIP_TOO_CLOSE");
                }
            }
        }
        Ok(Self {
            relationship: relationship.clone(),
            design: design.clone(),
            traits: [first, second],
            blocks: family_blocks(relationship),
            rows,
        })
    }

    /// Whether each trait's variance is a free parameter.
    fn variance_is_free(&self) -> [bool; 2] {
        [
            self.traits[0].kind != TraitKind::Binary,
            self.traits[1].kind != TraitKind::Binary,
        ]
    }

    /// The log likelihood at one point.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn loglik(
        &self,
        heritability: [f64; 2],
        variance: [f64; 2],
        genetic_correlation: f64,
        residual_correlation: f64,
        beta: [&[f64]; 2],
    ) -> Option<f64> {
        for t in 0..2 {
            if !(0.0..=1.0).contains(&heritability[t])
                || !(variance[t] > 0.0)
                || !variance[t].is_finite()
                || beta[t].len() != self.design.ncols()
                || beta[t].iter().any(|value| !value.is_finite())
            {
                return None;
            }
        }
        if !(-1.0..=1.0).contains(&genetic_correlation)
            || !(-1.0..=1.0).contains(&residual_correlation)
        {
            return None;
        }
        let normal = Normal::new(0.0, 1.0).ok()?;

        // The two by two genetic and residual covariances.
        let additive = [heritability[0] * variance[0], heritability[1] * variance[1]];
        let residual = [
            (1.0 - heritability[0]) * variance[0],
            (1.0 - heritability[1]) * variance[1],
        ];
        let additive_cross = genetic_correlation * (additive[0] * additive[1]).sqrt();
        let residual_cross = residual_correlation * (residual[0] * residual[1]).sqrt();

        let mean: [DVector<f64>; 2] = [
            &self.design * DVector::from_row_slice(beta[0]),
            &self.design * DVector::from_row_slice(beta[1]),
        ];

        let mut total = 0.0;
        for block in &self.blocks {
            let n = block.len();
            let size = 2 * n;
            // Trait major: entry (t, i) sits at t * n + i.
            let mut covariance = DMatrix::<f64>::zeros(size, size);
            for a in 0..2 {
                for b in 0..2 {
                    let genetic = if a == b { additive[a] } else { additive_cross };
                    let environmental = if a == b { residual[a] } else { residual_cross };
                    for (row, &i) in block.iter().enumerate() {
                        for (column, &j) in block.iter().enumerate() {
                            let same_person = i == j;
                            covariance[(a * n + row, b * n + column)] = genetic
                                * self.relationship[(i, j)]
                                + if same_person { environmental } else { 0.0 };
                        }
                    }
                }
            }

            // Where each entry's value stands: measured, or a region.
            let mut exact = Vec::new();
            let mut region = Vec::new();
            for t in 0..2 {
                for (row, &person) in block.iter().enumerate() {
                    let slot = t * n + row;
                    match self.traits[t].censoring[person] {
                        Censoring::Measured => exact.push((slot, t, person)),
                        _ => region.push((slot, t, person)),
                    }
                }
            }

            let mut solved: Option<DVector<f64>> = None;
            let mut factor: Option<Cholesky<f64, nalgebra::Dyn>> = None;
            if !exact.is_empty() {
                let sub = DMatrix::from_fn(exact.len(), exact.len(), |a, b| {
                    covariance[(exact[a].0, exact[b].0)]
                });
                let chol = Cholesky::new(sub)?;
                let deviation = DVector::from_fn(exact.len(), |a, _| {
                    let (_, t, person) = exact[a];
                    self.traits[t].value[person] - mean[t][person]
                });
                let solution = chol.solve(&deviation);
                let quadratic = deviation.dot(&solution);
                let log_determinant = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
                let density =
                    -0.5 * (exact.len() as f64 * LOG_TWO_PI + log_determinant + quadratic);
                if !density.is_finite() {
                    return None;
                }
                total += density;
                solved = Some(solution);
                factor = Some(chol);
            }

            if region.is_empty() {
                continue;
            }

            let mut region_mean = Vec::with_capacity(region.len());
            let mut sign = Vec::with_capacity(region.len());
            for &(slot, t, person) in &region {
                let mut expected = mean[t][person];
                if let Some(solution) = &solved {
                    for (a, &(other, _, _)) in exact.iter().enumerate() {
                        expected += covariance[(slot, other)] * solution[a];
                    }
                }
                region_mean.push(expected - self.traits[t].limit[person]);
                sign.push(match self.traits[t].censoring[person] {
                    Censoring::Above => 1.0,
                    Censoring::Below => -1.0,
                    Censoring::Measured => unreachable!("partitioned above"),
                });
            }

            let mut conditional = DMatrix::from_fn(region.len(), region.len(), |a, b| {
                covariance[(region[a].0, region[b].0)]
            });
            if let Some(chol) = &factor {
                let cross = DMatrix::from_fn(exact.len(), region.len(), |a, b| {
                    covariance[(exact[a].0, region[b].0)]
                });
                let projected = chol.solve(&cross);
                conditional -= cross.transpose() * projected;
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
    pub fn fit(&self) -> Result<MixedBivariateFit, &'static str> {
        self.fit_holding(None)
    }

    /// One correlation against nought.
    ///
    /// **This is the question the model was built to answer.** `CONTEXT.md`
    /// names the analysis as the genetic correlation between a psychiatric
    /// diagnosis and hearing, and an estimate with an interval does not say
    /// whether the two traits share genes at all.
    ///
    /// Nought is an interior point of the correlation's range, so the reference
    /// is a plain chi-square on one degree of freedom. There is no mass at the
    /// null and no mixture to apply. That is what separates this from a
    /// variance against nought, which sits on its bound and needs the
    /// Self-Liang rule instead.
    ///
    /// The statistic and the p-value both come from `deviance`, which honours
    /// the point mass at nought: two searches that land on the same likelihood
    /// give a deviance that is rounding rather than evidence.
    ///
    /// # Errors
    ///
    /// Returns a stable code if the coordinate is not testable or either
    /// required fit did not converge.
    pub fn correlation_test(&self, coordinate: usize) -> Result<MixedBivariateTest, &'static str> {
        let what = match coordinate {
            GENETIC_CORRELATION => "genetic_correlation",
            RESIDUAL_CORRELATION => "residual_correlation",
            _ => return Err("MIXED_BIVARIATE_COORDINATE_HAS_NO_TEST"),
        };
        let free = self.fit()?;
        crate::convergence::require(free.converged, "MIXED_BIVARIATE_FIT_NOT_CONVERGED")?;
        let null = self.fit_holding(Some((coordinate, 0.0)))?;
        crate::convergence::require(null.converged, "MIXED_BIVARIATE_NULL_FIT_NOT_CONVERGED")?;
        let statistic = crate::deviance::deviance(free.loglik, null.loglik);
        Ok(MixedBivariateTest {
            what,
            statistic,
            p_value: crate::deviance::p_value(statistic, crate::deviance::chi2_one_df_upper_tail),
            rule: "chi2_1",
            null_loglik: null.loglik,
            alternative_loglik: free.loglik,
        })
    }

    /// A 95 per cent profile-likelihood interval for one coordinate.
    ///
    /// Only the two heritabilities and the two correlations are available. A
    /// binary trait's variance is fixed at one because a liability has no
    /// scale, so an interval on it would describe an assumption rather than
    /// the data.
    ///
    /// The maximum is taken from the held fit at the estimate, and a profile
    /// fit that failed counts as inside so that failures widen the interval
    /// rather than narrowing it -- both for the reasons `LiabilityModel`
    /// records.
    ///
    /// # Errors
    ///
    /// Returns a stable code for an unavailable coordinate, or where the free
    /// fit fails.
    pub fn profile_interval(
        &self,
        coordinate: usize,
    ) -> Result<MixedBivariateInterval, &'static str> {
        let (low_bound, high_bound) = match coordinate {
            HERITABILITY_ONE | HERITABILITY_TWO => (0.0, 1.0),
            GENETIC_CORRELATION | RESIDUAL_CORRELATION => (-1.0, 1.0),
            _ => return Err("MIXED_BIVARIATE_COORDINATE_HAS_NO_INTERVAL"),
        };
        let free = self.fit()?;
        crate::convergence::require(free.converged, "MIXED_BIVARIATE_FIT_NOT_CONVERGED")?;
        let estimate = match coordinate {
            HERITABILITY_ONE => free.heritability[0],
            HERITABILITY_TWO => free.heritability[1],
            GENETIC_CORRELATION => free.genetic_correlation,
            _ => free.residual_correlation,
        };
        let got = interval::profile_interval(estimate, (low_bound, high_bound), |value| {
            self.fit_holding(Some((coordinate, value)))
                .ok()
                .filter(|fit| fit.converged)
                .map(|fit| fit.loglik)
        });
        if got.estimate.is_none() {
            return Err("MIXED_BIVARIATE_PROFILE_NOT_EVALUABLE");
        }
        Ok(got)
    }

    /// Fit with one coordinate held, or everything free where `held` is `None`.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start converges.
    pub fn fit_holding(
        &self,
        held: Option<(usize, f64)>,
    ) -> Result<MixedBivariateFit, &'static str> {
        let columns = self.design.ncols();
        let free = self.variance_is_free();
        // h1, h2, log s1, log s2, rho_g, rho_e, then the two sets of effects.
        let count = 6 + 2 * columns;

        let mut lower = vec![f64::NEG_INFINITY; count];
        let mut upper = vec![f64::INFINITY; count];
        for t in 0..2 {
            lower[t] = 0.0;
            upper[t] = 1.0;
            if !free[t] {
                // A binary trait's variance is fixed at one, so its logarithm
                // is fixed at nought.
                lower[2 + t] = 0.0;
                upper[2 + t] = 0.0;
            }
        }
        lower[4] = -1.0;
        upper[4] = 1.0;
        lower[5] = -1.0;
        upper[5] = 1.0;
        if let Some((coordinate, value)) = held {
            if coordinate >= 6 {
                return Err("MIXED_BIVARIATE_HELD_COORDINATE_INVALID");
            }
            if value < lower[coordinate] || value > upper[coordinate] {
                return Err("MIXED_BIVARIATE_HELD_VALUE_OUT_OF_RANGE");
            }
            lower[coordinate] = value;
            upper[coordinate] = value;
        }

        // A starting scale from whatever was measured, per trait.
        let mut start_variance = [0.0_f64; 2];
        let mut start_mean = [0.0_f64; 2];
        for t in 0..2 {
            if !free[t] {
                start_variance[t] = 1.0;
                continue;
            }
            let measured: Vec<f64> = (0..self.rows)
                .filter(|&i| self.traits[t].censoring[i] == Censoring::Measured)
                .map(|i| self.traits[t].value[i])
                .collect();
            let centre = measured.iter().sum::<f64>() / measured.len() as f64;
            start_mean[t] = centre;
            start_variance[t] = (measured.iter().map(|v| (v - centre).powi(2)).sum::<f64>()
                / measured.len() as f64)
                .max(1e-12);
        }

        let value_of = |theta: &[f64]| -> f64 {
            let beta_one = &theta[6..6 + columns];
            let beta_two = &theta[6 + columns..];
            self.loglik(
                [theta[0], theta[1]],
                [theta[2].exp(), theta[3].exp()],
                theta[4],
                theta[5],
                [beta_one, beta_two],
            )
            .map_or(1e30, |v| -v)
        };
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
        for (h, rho) in [(0.2_f64, 0.0_f64), (0.5, 0.3), (0.5, -0.3)] {
            let mut start = vec![0.0; count];
            start[0] = h;
            start[1] = h;
            start[2] = start_variance[0].ln();
            start[3] = start_variance[1].ln();
            start[4] = rho;
            start[5] = rho;
            if let Some((coordinate, value)) = held {
                start[coordinate] = value;
            }
            start[6] = start_mean[0];
            start[6 + columns] = start_mean[1];
            let Ok(bounds) = Bounds::new(lower.clone(), upper.clone()) else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(count);
            control.maxit = 400;
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

        let (objective, theta) = best.ok_or("MIXED_BIVARIATE_NO_START_CONVERGED")?;
        // The projected, scaled gradient, as the liability model reports it.
        //
        // A coordinate resting on its bound contributes a gradient the search
        // is not free to act on, and a held coordinate contributes one it must
        // not act on. Counting either in full makes a constrained maximum
        // report that it did not converge. Dividing by the objective makes the
        // number mean the same thing at every sample size.
        //
        // This is not tidiness. A profile fit discarded as unconverged reads as
        // an end the data did not rule out, so the unprojected reading put one
        // end of every correlation interval on its own bound whatever the data
        // said -- and made the interval contradict this model's own test, which
        // rejected a correlation of nought that the interval reported as inside.
        let scaled_projected = |candidate: &[f64], objective: f64| -> f64 {
            gradient_of(candidate)
                .iter()
                .enumerate()
                .map(|(k, g)| {
                    let held_here = lower[k] == upper[k];
                    let at_lower = candidate[k] <= lower[k] && *g > 0.0;
                    let at_upper = candidate[k] >= upper[k] && *g < 0.0;
                    if held_here || at_lower || at_upper {
                        0.0
                    } else {
                        *g
                    }
                })
                .fold(0.0f64, |worst, g| worst.max(g.abs()))
                / objective.abs().max(1.0)
        };

        let (mut objective, mut theta) = (objective, theta);
        let mut scaled_gradient = scaled_projected(&theta, objective);

        // Search again where the gradient test fails, as seven other families
        // already do. This one did not, and it is the family where it matters
        // most: the constrained refits along a profile are harder than the free
        // fit, so it is those that fall short, and a discarded profile
        // evaluation is scored as a miss however well the interval covered.
        //
        // Measured on four hundred replicates of the binary-with-censored pair,
        // four per cent of them had at least one such evaluation, and that four
        // per cent is the whole of the gap between the coverage this model
        // achieves -- 0.965 by plain containment -- and the 0.925 its check
        // reported.
        if scaled_gradient >= 1e-5
            && let Some(better) = crate::convergence::polish(
                &theta,
                objective,
                scaled_gradient,
                &lower,
                &upper,
                &value_of,
                &gradient_of,
                |candidate| {
                    let negative = value_of(candidate);
                    // The sentinel `value_of` returns where the likelihood
                    // cannot be evaluated at all.
                    if !negative.is_finite() || negative >= 1e30 {
                        return None;
                    }
                    Some((negative, scaled_projected(candidate, negative)))
                },
            )
        {
            theta = better.par;
            objective = better.negative_loglik;
            scaled_gradient = better.scaled_gradient;
        }

        Ok(MixedBivariateFit {
            heritability: [theta[0], theta[1]],
            total_variance: [theta[2].exp(), theta[3].exp()],
            genetic_correlation: theta[4],
            residual_correlation: theta[5],
            fixed_effects: [
                theta[6..6 + columns].to_vec(),
                theta[6 + columns..].to_vec(),
            ],
            loglik: -objective,
            converged: scaled_gradient < 1e-5,
            scaled_gradient,
            kinds: [self.traits[0].kind, self.traits[1].kind],
            estimator: "ml",
            largest_family: self.blocks.iter().map(Vec::len).max().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sib_relationship(pairs: usize) -> DMatrix<f64> {
        let n = 2 * pairs;
        let mut a = DMatrix::<f64>::identity(n, n);
        for pair in 0..pairs {
            a[(2 * pair, 2 * pair + 1)] = 0.5;
            a[(2 * pair + 1, 2 * pair)] = 0.5;
        }
        a
    }

    /// Two correlated traits on sibling pairs, from a known truth.
    ///
    /// Returns both traits' complete values. What is done with them -- left
    /// continuous, thresholded, or censored -- is the caller's business.
    fn simulate(
        pairs: usize,
        heritability: [f64; 2],
        variance: [f64; 2],
        genetic_correlation: f64,
        residual_correlation: f64,
        seed: u64,
    ) -> [Vec<f64>; 2] {
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
        // A correlated pair from two independent draws.
        let correlated = |rho: f64, a: f64, b: f64| (a, rho * a + (1.0 - rho * rho).sqrt() * b);

        let mut out = [Vec::with_capacity(2 * pairs), Vec::with_capacity(2 * pairs)];
        for _ in 0..pairs {
            // The family's shared additive half, correlated across traits.
            let (shared_one, shared_two) = correlated(genetic_correlation, draw(), draw());
            for _ in 0..2 {
                let (own_one, own_two) = correlated(genetic_correlation, draw(), draw());
                let (residual_one, residual_two) = correlated(residual_correlation, draw(), draw());
                let genetic = [
                    (0.5_f64).sqrt() * shared_one + (0.5_f64).sqrt() * own_one,
                    (0.5_f64).sqrt() * shared_two + (0.5_f64).sqrt() * own_two,
                ];
                let residual = [residual_one, residual_two];
                for t in 0..2 {
                    let value = (heritability[t] * variance[t]).sqrt() * genetic[t]
                        + ((1.0 - heritability[t]) * variance[t]).sqrt() * residual[t];
                    out[t].push(value);
                }
            }
        }
        out
    }

    fn continuous(values: &[f64]) -> TraitData {
        TraitData {
            kind: TraitKind::Continuous,
            value: values.to_vec(),
            censoring: vec![Censoring::Measured; values.len()],
            limit: vec![f64::NAN; values.len()],
        }
    }

    /// With both traits continuous the likelihood is an ordinary multivariate
    /// normal density over the whole family block, and must equal one built a
    /// different way.
    ///
    /// This is the hinge. If the all-continuous case is not `BivariateModel`'s
    /// likelihood then nothing built on top of it generalises that model.
    #[test]
    fn with_both_traits_continuous_it_is_the_gaussian_likelihood() {
        let pairs = 40;
        let relationship = sib_relationship(pairs);
        let values = simulate(pairs, [0.5, 0.4], [2.0, 3.0], 0.4, 0.1, 20_260_820);
        let design = DMatrix::from_element(2 * pairs, 1, 1.0);
        let model = MixedBivariateModel::build(
            &relationship,
            continuous(&values[0]),
            continuous(&values[1]),
            &design,
        )
        .expect("builds");

        let heritability = [0.45_f64, 0.35_f64];
        let variance = [1.8_f64, 2.6_f64];
        let (rho_g, rho_e) = (0.3_f64, 0.15_f64);
        let mean = [0.4_f64, -0.2_f64];
        let ours = model
            .loglik(
                heritability,
                variance,
                rho_g,
                rho_e,
                [&[mean[0]], &[mean[1]]],
            )
            .expect("evaluates");

        let additive = [heritability[0] * variance[0], heritability[1] * variance[1]];
        let residual = [
            (1.0 - heritability[0]) * variance[0],
            (1.0 - heritability[1]) * variance[1],
        ];
        let additive_cross = rho_g * (additive[0] * additive[1]).sqrt();
        let residual_cross = rho_e * (residual[0] * residual[1]).sqrt();

        let mut theirs = 0.0;
        for pair in 0..pairs {
            let block = [2 * pair, 2 * pair + 1];
            let mut covariance = DMatrix::<f64>::zeros(4, 4);
            for a in 0..2 {
                for b in 0..2 {
                    let genetic = if a == b { additive[a] } else { additive_cross };
                    let environmental = if a == b { residual[a] } else { residual_cross };
                    for (row, &i) in block.iter().enumerate() {
                        for (column, &j) in block.iter().enumerate() {
                            covariance[(a * 2 + row, b * 2 + column)] = genetic
                                * relationship[(i, j)]
                                + if i == j { environmental } else { 0.0 };
                        }
                    }
                }
            }
            let deviation = DVector::from_fn(4, |slot, _| {
                let t = slot / 2;
                values[t][block[slot % 2]] - mean[t]
            });
            let chol = Cholesky::new(covariance).expect("positive definite");
            let solved = chol.solve(&deviation);
            let log_determinant = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
            theirs += -0.5 * (4.0 * LOG_TWO_PI + log_determinant + deviation.dot(&solved));
        }
        assert!(
            (ours - theirs).abs() < 1e-9,
            "the all-continuous likelihood is not the Gaussian one: {ours} against {theirs}"
        );
    }

    /// A binary trait beside a continuous one recovers the genetic correlation.
    ///
    /// This is the pair the analysis wants: a psychiatric diagnosis against
    /// hearing. The correlation is the estimand, and it is scale free, which is
    /// what makes the mixed pair worth fitting even though one of the two
    /// scales is arbitrary.
    #[test]
    fn a_binary_trait_beside_a_continuous_one_recovers_the_correlation() {
        let pairs = 500;
        let relationship = sib_relationship(pairs);
        let truth = 0.5_f64;
        // The binary trait's variance is one, because that is the only scale a
        // liability has.
        let values = simulate(pairs, [0.5, 0.5], [1.0, 2.0], truth, 0.1, 20_260_821);
        let n = 2 * pairs;
        // A third affected, so the threshold is well inside the distribution.
        let mut sorted = values[0].clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cut = sorted[2 * n / 3];

        let binary = TraitData {
            kind: TraitKind::Binary,
            value: vec![f64::NAN; n],
            censoring: values[0]
                .iter()
                .map(|v| {
                    if *v > cut {
                        Censoring::Above
                    } else {
                        Censoring::Below
                    }
                })
                .collect(),
            // The threshold is carried by the intercept, so the limit is nought.
            limit: vec![0.0; n],
        };
        let design = DMatrix::from_element(n, 1, 1.0);
        let model =
            MixedBivariateModel::build(&relationship, binary, continuous(&values[1]), &design)
                .expect("builds");
        let fit = model.fit().expect("fits");

        assert_eq!(fit.estimator, "ml");
        assert_eq!(fit.kinds, [TraitKind::Binary, TraitKind::Continuous]);
        // The binary trait's variance is not estimated, and says so.
        assert!(
            (fit.total_variance[0] - 1.0).abs() < 1e-9,
            "a binary trait's variance moved off one: {}",
            fit.total_variance[0]
        );
        assert!(
            (fit.genetic_correlation - truth).abs() < 0.30,
            "the genetic correlation came back {} against a true {truth}",
            fit.genetic_correlation
        );
    }

    /// The genetic correlation's interval reaches where the likelihood falls
    /// away, and contains the truth it was simulated from.
    ///
    /// The deviance cost at each end is checked rather than assumed. An
    /// interval can look entirely reasonable and sit somewhere the profile
    /// never crossed.
    #[test]
    fn the_genetic_correlation_interval_is_where_the_profile_falls_away() {
        let pairs = 400;
        let relationship = sib_relationship(pairs);
        let truth = 0.5_f64;
        let values = simulate(pairs, [0.5, 0.5], [1.0, 2.0], truth, 0.1, 20_260_822);
        let design = DMatrix::from_element(2 * pairs, 1, 1.0);
        let model = MixedBivariateModel::build(
            &relationship,
            continuous(&values[0]),
            continuous(&values[1]),
            &design,
        )
        .expect("builds");

        let got = model
            .profile_interval(GENETIC_CORRELATION)
            .expect("intervals");
        let estimate = got.estimate.expect("profile maximum");
        assert!(
            got.lower <= estimate && estimate <= got.upper,
            "the estimate {} is outside its own interval [{}, {}]",
            estimate,
            got.lower,
            got.upper
        );
        assert!(
            got.lower <= truth && truth <= got.upper,
            "the interval [{}, {}] misses the true {truth}",
            got.lower,
            got.upper
        );

        let peak = model
            .fit_holding(Some((GENETIC_CORRELATION, estimate)))
            .expect("held fit")
            .loglik;
        for (name, end, at_bound) in [
            ("lower", got.lower, got.lower_limited),
            ("upper", got.upper, got.upper_limited),
        ] {
            if at_bound {
                continue;
            }
            let there = model
                .fit_holding(Some((GENETIC_CORRELATION, end)))
                .expect("held fit")
                .loglik;
            let cost = 2.0 * (peak - there);
            assert!(
                (cost - 3.841_458_820_694_124).abs() < 0.05,
                "the {name} end costs {cost} in deviance, not the 95% chi-square threshold"
            );
        }
    }

    /// A coordinate with no interval says so rather than inventing one.
    #[test]
    fn a_variance_has_no_interval() {
        let pairs = 40;
        let relationship = sib_relationship(pairs);
        let values = simulate(pairs, [0.5, 0.5], [1.0, 1.0], 0.2, 0.0, 17);
        let design = DMatrix::from_element(2 * pairs, 1, 1.0);
        let model = MixedBivariateModel::build(
            &relationship,
            continuous(&values[0]),
            continuous(&values[1]),
            &design,
        )
        .expect("builds");
        // Coordinates two and three are the variances, one of which is fixed
        // at one whenever a trait is binary.
        assert_eq!(
            model.profile_interval(2).err(),
            Some("MIXED_BIVARIATE_COORDINATE_HAS_NO_INTERVAL")
        );
    }

    /// Each kind of trait refuses the data that would make it a different kind.
    #[test]
    fn a_trait_that_contradicts_its_kind_is_refused() {
        let pairs = 20;
        let n = 2 * pairs;
        let relationship = sib_relationship(pairs);
        let values = simulate(pairs, [0.5, 0.5], [1.0, 1.0], 0.2, 0.0, 11);
        let design = DMatrix::from_element(n, 1, 1.0);

        // A binary trait cannot carry a measured value.
        let contradictory = TraitData {
            kind: TraitKind::Binary,
            value: values[0].clone(),
            censoring: vec![Censoring::Measured; n],
            limit: vec![0.0; n],
        };
        assert_eq!(
            MixedBivariateModel::build(
                &relationship,
                contradictory,
                continuous(&values[1]),
                &design
            )
            .err(),
            Some("MIXED_BIVARIATE_BINARY_TRAIT_HAS_A_MEASURED_VALUE")
        );

        // A binary trait with everybody on one side has no threshold to find.
        let one_sided = TraitData {
            kind: TraitKind::Binary,
            value: vec![f64::NAN; n],
            censoring: vec![Censoring::Below; n],
            limit: vec![0.0; n],
        };
        assert_eq!(
            MixedBivariateModel::build(&relationship, one_sided, continuous(&values[1]), &design)
                .err(),
            Some("MIXED_BIVARIATE_BINARY_TRAIT_ALL_ONE_STATUS")
        );

        // A continuous trait cannot be censored; that is what censored is for.
        let mut censoring = vec![Censoring::Measured; n];
        censoring[0] = Censoring::Above;
        let mislabelled = TraitData {
            kind: TraitKind::Continuous,
            value: values[0].clone(),
            censoring,
            limit: vec![0.0; n],
        };
        assert_eq!(
            MixedBivariateModel::build(&relationship, mislabelled, continuous(&values[1]), &design)
                .err(),
            Some("MIXED_BIVARIATE_CONTINUOUS_TRAIT_IS_CENSORED")
        );
    }
}

#[cfg(feature = "python")]
pub mod python;
