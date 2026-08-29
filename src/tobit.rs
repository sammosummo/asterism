//! One trait whose measurement stops at a limit -- a censored variance
//! components model.
//!
//! **This is a special case rather than part of the Gaussian general model.**
//! Nothing here is folded into `ComponentModel`, for decision 16's reasons:
//! REML has no meaning once part of the likelihood is a probability rather than
//! a density, and the fixed effects cannot be profiled out by least squares.
//! That it now takes several components does not change this -- the two models
//! share a shape, not a likelihood.
//!
//! # The model
//!
//! Every record carries a complete value
//!
//! ```text
//! y*_i = x_i' beta + sum_c u_ci + e_i
//! ```
//!
//! with `u_c ~ N(0, p_c sigma^2 K_c)` for each supplied component `K_c`, and
//! `e ~ N(0, (1 - sum_c p_c) sigma^2 I)`. Where the instrument could measure
//! it, `y*_i` is seen exactly. Where it could not, all that is known is that
//! the value lies beyond the limit that measurement reached.
//!
//! **The residual is what the components leave.** It is never supplied and its
//! coefficient is never estimated directly, which is what puts the others on a
//! common scale rather than each needing one of its own. The search works in
//! stick-breaking coordinates, so a set summing past one is unreachable rather
//! than refused and the bounded optimiser meets no cliff -- see
//! `TobitModel::coefficients_from`.
//!
//! Those coefficients are **proportions of the total variance exactly when
//! every component carries a unit diagonal** -- true of additive kinship, of a
//! person-level matrix and of a household kernel, and not true in general. See
//! `TobitFit::coefficients`.
//!
//! **One component is the special case this began as.** With a single kinship
//! matrix the first share is the heritability, the residual is `1 - h2`, and
//! the arithmetic is what it always was -- the parameter vector has the same
//! layout, so the profile still holds the heritability by pinning one bound.
//!
//! Several components is what an audiogram needs: two records per person want
//! a person-level term beside the genetic one, or the resemblance between a
//! person's own two ears is left with nowhere to go but the heritability.
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

use crate::blocks::union_blocks;
use crate::interval::{self, Interval};
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
    /// Each component's **raw coefficient**, in the order the components were
    /// given. The residual takes `1 - sum` and is not among these.
    ///
    /// **Coefficients, not proportions, and the difference is not pedantry.**
    /// These are shares of the total variance exactly when every component
    /// carries a unit diagonal, which additive kinship, a person-level matrix
    /// and a household kernel all do. A component whose diagonal is not one --
    /// a genomic relationship matrix, or a kernel that leaves somebody
    /// unhoused at nought -- makes `total_variance` nobody's total variance
    /// and `1 - sum` not the residual's share. `ComponentModel` reports the
    /// mean-diagonal contribution for this reason; see CONTEXT.md.
    pub coefficients: Vec<f64>,
    /// The largest family, because the region probability is exact to two
    /// people and approximate above it.
    pub largest_family: usize,
}

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

/// Compatibility name for the one shared interval record.
pub type TobitInterval = Interval;

/// One trait, any number of components, per-observation censoring.
pub struct TobitModel {
    components: Vec<DMatrix<f64>>,
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
    /// `components` are the relationship matrices whose coefficients the fit
    /// estimates -- additive kinship, a person-level matrix, a household kernel,
    /// whatever the model carries. **One component is the special case this
    /// model began as**, and passing one gives exactly what it always gave.
    ///
    /// A residual is always present and is never passed: it is what is left of
    /// the variance once the components have taken their coefficients, which is
    /// what puts those coefficients on a common scale. They are proportions of
    /// the total only where every component has a unit diagonal -- see
    /// `TobitFit::coefficients`.
    ///
    /// `value` holds the measured value where `censoring` is `Measured`, and is
    /// ignored otherwise. `limit` holds the limit that observation reached, and
    /// is ignored where the value was measured.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(
        components: &[DMatrix<f64>],
        value: &[f64],
        censoring: &[Censoring],
        limit: &[f64],
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let rows = value.len();
        if rows == 0 {
            return Err("TOBIT_NO_ROWS");
        }
        if components.is_empty() {
            return Err("TOBIT_NO_COMPONENTS");
        }
        if censoring.len() != rows || limit.len() != rows {
            return Err("TOBIT_INPUTS_DIFFERENT_LENGTHS");
        }
        for component in components {
            if component.nrows() != rows || component.ncols() != rows {
                return Err("TOBIT_RELATIONSHIP_WRONG_SHAPE");
            }
            if !component.iter().all(|v| v.is_finite()) {
                return Err("TOBIT_NOT_FINITE");
            }
            for i in 0..rows {
                for j in 0..i {
                    if (component[(i, j)] - component[(j, i)]).abs() > 1e-10 {
                        return Err("TOBIT_RELATIONSHIP_NOT_SYMMETRIC");
                    }
                }
            }
        }
        if design.nrows() != rows {
            return Err("TOBIT_DESIGN_WRONG_SHAPE");
        }
        if design.ncols() == 0 {
            return Err("TOBIT_DESIGN_HAS_NO_COLUMNS");
        }
        if !design.iter().all(|v| v.is_finite()) {
            return Err("TOBIT_NOT_FINITE");
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
        // provenance.** Symmetry does not make a matrix a covariance: one that
        // is not positive semi-definite reaches the search, where it surfaces
        // only as a fit that found no feasible point. Checking it here says
        // what is wrong while the caller can still do something about it.
        //
        // **Per component, over the union's blocks.** The likelihood factorises
        // over the blocks the components make together, so that is the only
        // decomposition it ever sees; and each component has to be a covariance
        // in its own right, because the search may put all the variance on any
        // one of them.
        let blocks = union_blocks(components);
        for component in components {
            for block in &blocks {
                let size = block.len();
                let sub = DMatrix::from_fn(size, size, |a, b| component[(block[a], block[b])]);
                let smallest = SymmetricEigen::new(sub)
                    .eigenvalues
                    .iter()
                    .fold(f64::INFINITY, |worst, value| worst.min(*value));
                if smallest < EIGENVALUE_FLOOR {
                    return Err("TOBIT_RELATIONSHIP_NOT_PSD");
                }
            }
        }

        Ok(Self {
            components: components.to_vec(),
            design: design.clone(),
            value: value.to_vec(),
            censoring: censoring.to_vec(),
            limit: limit.to_vec(),
            blocks,
            rows,
        })
    }

    /// Turn a boxed search vector into coefficients that always sum within one.
    ///
    /// **The feasible set is a simplex and the optimiser only understands a
    /// box.** Written the obvious way -- each coefficient boxed in `[0, 1]` and
    /// the sum refused past one -- the search meets a cliff the central
    /// difference cannot read: the objective jumps to its infeasible value, the
    /// gradient comes back as a number the size of that jump, and the search
    /// does not move. Measured before this was fixed: at a true split of 0.45
    /// and 0.50, leaving a residual of 0.05, five seeds all returned the
    /// starting values unchanged with `converged` false. A small residual is
    /// the case this model exists for, so that is not a corner worth losing.
    ///
    /// Stick-breaking removes the cliff instead of guarding it. Each search
    /// coordinate takes a share of what the earlier ones left, so every point
    /// of the box maps to a valid set and no point of the box is infeasible.
    ///
    /// **At one component this is the identity**: `left` is one, so the first
    /// coefficient is the first coordinate and the model's arithmetic is
    /// exactly what it was.
    fn coefficients_from(search: &[f64]) -> Vec<f64> {
        let mut coefficients = Vec::with_capacity(search.len());
        let mut left = 1.0;
        for &coordinate in search {
            let taken = left * coordinate;
            coefficients.push(taken);
            left -= taken;
        }
        coefficients
    }

    /// How many components the fit estimates, the residual not among them.
    #[must_use]
    pub fn components(&self) -> usize {
        self.components.len()
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

    /// The log likelihood at one set of coefficients, one total variance and
    /// one set of fixed effects.
    ///
    /// `coefficients` carries one per component, in the order the
    /// components were given. **They must each lie in `[0, 1]` and must sum to
    /// no more than one**, because the residual takes what is left and a
    /// negative residual describes no covariance.
    ///
    /// Returns `None` where the parameters do not describe a model, which the
    /// optimiser reads as a wall rather than a failure.
    #[must_use]
    pub fn loglik(&self, coefficients: &[f64], variance: f64, beta: &[f64]) -> Option<f64> {
        if coefficients.len() != self.components.len() {
            return None;
        }
        if !coefficients.iter().all(|c| (0.0..=1.0).contains(c)) {
            return None;
        }
        // The residual takes what the components leave, so a set of shares that
        // sums past one describes no covariance and is refused rather than
        // clamped. The optimiser reads `None` as a wall and turns away from it.
        let residual_share = 1.0 - coefficients.iter().sum::<f64>();
        if residual_share < 0.0 {
            return None;
        }
        if !(variance > 0.0) || !variance.is_finite() {
            return None;
        }
        if beta.len() != self.design.ncols() || !beta.iter().all(|b| b.is_finite()) {
            return None;
        }
        let normal = Normal::new(0.0, 1.0).ok()?;
        let fixed = DVector::from_row_slice(beta);
        let mean = &self.design * &fixed;

        let mut total = 0.0;
        for block in &self.blocks {
            let size = block.len();
            // The family covariance, on the complete variable's own scale.
            let mut covariance = DMatrix::<f64>::zeros(size, size);
            for (row, &i) in block.iter().enumerate() {
                for (column, &j) in block.iter().enumerate() {
                    let mut shared = 0.0;
                    for (component, share) in self.components.iter().zip(coefficients) {
                        shared += share * component[(i, j)];
                    }
                    let residual = if i == j { residual_share } else { 0.0 };
                    covariance[(row, column)] = variance * (shared + residual);
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
                let log_determinant = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
                let density =
                    -0.5 * (measured.len() as f64 * LOG_TWO_PI + log_determinant + quadratic);
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
        // **The mixture rule was scored for one component and only one.**
        // Holding the first share at nought while the others are free to rest
        // on their own bounds is not the one-parameter-on-one-bound case the
        // fifty-fifty mixture covers, so the p-value below would be read
        // against the wrong reference. Refusing is what CONTEXT requires of an
        // unscored verdict: an absent one means nobody has measured it yet.
        // Interval coverage at several components is issue 38.
        if self.components.len() != 1 {
            return Err("TOBIT_TEST_NOT_SCORED_FOR_SEVERAL_COMPONENTS");
        }

        let free = self.fit()?;
        let null = self.fit_holding(Some(0.0))?;
        crate::convergence::require(free.converged, "TOBIT_FIT_NOT_CONVERGED")?;
        crate::convergence::require(null.converged, "TOBIT_NULL_FIT_NOT_CONVERGED")?;
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

    /// # Errors
    ///
    /// Returns a stable code if the free fit or the profile at its estimate is
    /// not evaluable as a converged likelihood.
    pub fn heritability_interval(&self) -> Result<TobitInterval, &'static str> {
        let free = self.fit()?;
        if !free.converged {
            return Err("TOBIT_FIT_NOT_CONVERGED");
        }
        let estimate = free.heritability;
        let got = interval::profile_interval(estimate, (0.0, 1.0), |value| {
            self.fit_holding(Some(value))
                .ok()
                .filter(|fit| fit.converged)
                .map(|fit| fit.loglik)
        });
        if got.estimate.is_none() {
            return Err("TOBIT_PROFILE_NOT_EVALUABLE");
        }
        // This family's coverage check scored the boundary rule through three
        // quarters censoring, **at one component**, so it may fill the mixture
        // verdict there and only there. With several components more than one
        // share can rest on nought at once, which is not the case that check
        // scored, so the interval is returned with its boundary verdict absent
        // rather than filled in from a measurement of a different model. An
        // absent verdict says nobody has measured it; a filled one says
        // somebody has.
        if self.components.len() == 1 {
            return Ok(got.scored_by_mixture());
        }
        Ok(got)
    }

    /// Fit with the heritability held, or free where `held` is `None`.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start converges.
    pub fn fit_holding(&self, held: Option<f64>) -> Result<TobitFit, &'static str> {
        if let Some(value) = held
            && !(0.0..=1.0).contains(&value)
        {
            return Err("TOBIT_HELD_HERITABILITY_OUT_OF_RANGE");
        }
        let columns = self.design.ncols();
        let parts = self.components.len();
        // The shares first, then the log total variance, then the fixed
        // effects. **The heritability stays at index nought**, which is what
        // lets the profile hold it by pinning one box bound, exactly as it did
        // when there was only ever one share. With one component this is the
        // layout this model has always had, to the slot.
        let count = parts + 1 + columns;

        // Starting values from the measured part alone. They are wrong -- that
        // is the whole point of the model -- but they are the right order of
        // magnitude, which is all a start has to be.
        let measured: Vec<f64> = (0..self.rows)
            .filter(|&i| self.censoring[i] == Censoring::Measured)
            .map(|i| self.value[i])
            .collect();
        let centre = measured.iter().sum::<f64>() / measured.len() as f64;
        let spread = (measured.iter().map(|v| (v - centre).powi(2)).sum::<f64>()
            / measured.len() as f64)
            .max(1e-12);

        let mut lower = vec![f64::NEG_INFINITY; count];
        let mut upper = vec![f64::INFINITY; count];
        for share in 0..parts {
            lower[share] = 0.0;
            upper[share] = 1.0;
        }
        // The shares are boxed individually; that they must also sum within one
        // is not a box and is refused by the likelihood instead.
        if let Some(value) = held {
            lower[0] = value;
            upper[0] = value;
        }

        let value_of = |theta: &[f64]| -> f64 {
            self.loglik(
                &Self::coefficients_from(&theta[..parts]),
                theta[parts].exp(),
                &theta[parts + 1..],
            )
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

        // Three starts spread over the first coefficient, or one where it is
        // held. A held fit pinned every start to the same value and then ran
        // the same deterministic search three times over: the interval paid for
        // that at each end of every bisection step.
        //
        // **That reasoning is exact at one component and approximate above it.**
        // Holding the first coefficient no longer pins the whole share space --
        // the others stay free -- so one start explores a simplex from a single
        // point where three would have explored it from three. It is still one
        // deterministic search per start, so three would still be three copies
        // of one answer; what is lost is spread, not repetition.
        let starts: &[f64] = match held {
            Some(value) => &[value],
            None => &[0.05, 0.3, 0.6],
        };
        let mut best: Option<(f64, Vec<f64>)> = None;
        for &heritability in starts {
            let mut start = vec![0.0; count];
            start[0] = heritability;
            // Each later coordinate takes an even split of what is left,
            // counting the residual as one more claimant, so the start sits
            // inside the simplex rather than on the face where the residual is
            // nought. These are stick-breaking coordinates, not coefficients.
            for (index, coordinate) in start.iter_mut().enumerate().take(parts).skip(1) {
                *coordinate = 1.0 / (parts - index + 1) as f64;
            }
            start[parts] = spread.ln();
            start[parts + 1] = centre;
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

        // The search is bound constrained, so what says whether it has arrived
        // is the projected gradient, not the raw one. At an optimum resting on
        // a bound the raw gradient points out of the feasible region and does
        // not go to nought, so testing it reports a correct fit as a failure.
        // A heritability of nought is the lower bound and a true nought leaves
        // about half of all samples resting there, which is why about half of
        // every null fit refused -- at three quarters censored, at half, and
        // at none at all, where this model is the one-trait model and that one
        // fits every time.
        //
        // This is the reading `components.rs` takes, in this family's
        // parameterisation: only the heritability has bounds to rest on, while
        // the log total variance and the fixed effects are free. Scaling by the
        // objective is part of that reading and was missing too, so the value
        // once called `scaled_gradient` here was neither projected nor scaled.
        let scaled_projected = |candidate: &[f64], objective: f64| -> f64 {
            gradient_of(candidate)
                .iter()
                .enumerate()
                .map(|(k, g)| {
                    if lower[k] == upper[k] {
                        // A held coordinate is not a direction the search may
                        // act on, whichever way its gradient points.
                        0.0
                    } else if crate::components::resting_on_zero(candidate[k] - lower[k]) {
                        g.min(0.0)
                    } else if crate::components::resting_on_zero(upper[k] - candidate[k]) {
                        g.max(0.0)
                    } else {
                        *g
                    }
                })
                .fold(0.0_f64, |worst, g| worst.max(g.abs()))
                / objective.abs().max(1.0)
        };

        let (mut objective, mut theta) = best.ok_or("TOBIT_NO_START_CONVERGED")?;
        let mut scaled_gradient = scaled_projected(&theta, objective);

        // Where the gradient test fails, search once more from the point
        // already found with the objective tolerance switched off. This is the
        // shared second search, and it fires only where the flag already says
        // failure, so every passing fit is left untouched to the last bit. The
        // reasons it is bounded, and why its result is checked rather than
        // trusted, are in `convergence`.
        if scaled_gradient >= crate::convergence::TOLERANCE
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
                    if !negative.is_finite() || negative >= INFEASIBLE {
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

        let coefficients = Self::coefficients_from(&theta[..parts]);
        Ok(TobitFit {
            heritability: coefficients[0],
            coefficients,
            total_variance: theta[parts].exp(),
            fixed_effects: theta[parts + 1..].to_vec(),
            loglik: -objective,
            // The shared rule, at last. This family had been testing a raw
            // unscaled gradient against a hard-coded thousandth, which is a
            // thousand times looser than every other family and was loosened
            // to accommodate the very boundary readings the projection above
            // removes. `convergence` records what the number means.
            converged: scaled_gradient < crate::convergence::TOLERANCE,
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
    // Re-pinned on 29 August 2026, when the two-person normal integral was
    // replaced by an accurate one. It moved by 2.9e-09, which is 2.8e-11 of the
    // value: the search still lands in the same place, the place is simply
    // located a little more precisely than the retired quadrature could locate
    // it. The equality below is still an equality, and still says what it says.
    const PINNED_HELD_LOGLIK: f64 = -102.736_491_869_479_9;

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

        match TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        ) {
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
        let model = TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        )
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

        let model = TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        )
        .expect("a nought diagonal is not something build refuses");

        assert!(
            model.loglik(&[1.0], 1.0, &[0.0]).is_none(),
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
    #[allow(clippy::type_complexity)]
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
        let model = TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        )
        .expect("builds");
        let (heritability, variance, mean) = (0.4, 1.7, 2.5);
        let ours = model
            .loglik(&[heritability], variance, &[mean])
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
            let deviation = DVector::from_fn(2, |r, _| value[block[r]] - mean);
            let determinant = covariance[(0, 0)] * covariance[(1, 1)] - covariance[(0, 1)].powi(2);
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
        let censored = censoring
            .iter()
            .filter(|c| **c != Censoring::Measured)
            .count();
        assert!(
            censored > 40,
            "the fixture censored {censored} values, too few to test with"
        );

        let model = TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        )
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
            std::slice::from_ref(&relationship),
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
        let (relationship, value, censoring, limit, design) =
            simulate(300, 0.5, 4.0, 10.0, Some(11.0), 20_260_818);
        let model = TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        )
        .expect("builds");
        let got = model.heritability_interval().expect("intervals");
        let estimate = got.estimate.expect("an interval has an estimate");

        assert!(
            got.lower <= estimate && estimate <= got.upper,
            "the estimate {} is outside its own interval [{}, {}]",
            estimate,
            got.lower,
            got.upper
        );
        assert!((got.level - 0.95).abs() < 1e-12);

        let peak = model.fit_holding(Some(estimate)).expect("held fit").loglik;
        for (name, end, at_bound) in [
            ("lower", got.lower, got.lower_limited),
            ("upper", got.upper, got.upper_limited),
        ] {
            if at_bound {
                continue; // the data did not rule that end out
            }
            let there = model.fit_holding(Some(end)).expect("held fit").loglik;
            let cost = 2.0 * (peak - there);
            let claimed = 3.841_458_820_694_124;
            assert!(
                (cost - claimed).abs() < 0.05,
                "the {name} end costs {cost} in deviance, not the {claimed} \
                 an interval at this level claims"
            );
        }
    }

    /// A heritability held outside its range is refused rather than clamped.
    #[test]
    fn a_held_heritability_outside_the_range_is_refused() {
        let (relationship, value, censoring, limit, design) = simulate(40, 0.5, 1.0, 0.0, None, 3);
        let model = TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        )
        .expect("builds");
        assert_eq!(
            model.fit_holding(Some(1.5)).err(),
            Some("TOBIT_HELD_HERITABILITY_OUT_OF_RANGE")
        );
    }

    /// Several components, with censoring, recovered rather than just accepted.
    ///
    /// Two records per person, a person-level component beside the additive
    /// one, and a quarter of the values censored -- so this is the only test
    /// that exercises censoring and a second component at once, which is what
    /// the model is for.
    #[test]
    fn several_components_are_recovered_from_data_that_has_them() {
        let people = 120;
        let rows = people * 2;
        let mut seed = 4242_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
        };

        // Sibling pairs, each person contributing two records.
        let mut additive = DMatrix::<f64>::zeros(rows, rows);
        let mut person = DMatrix::<f64>::zeros(rows, rows);
        for i in 0..rows {
            for j in 0..rows {
                let (a, b) = (i / 2, j / 2);
                if a == b {
                    person[(i, j)] = 1.0;
                    additive[(i, j)] = 1.0;
                } else if a / 2 == b / 2 {
                    additive[(i, j)] = 0.5;
                }
            }
        }

        let (genetic_share, person_share, total): (f64, f64, f64) = (0.4, 0.3, 4.0);
        let mut value = vec![0.0; rows];
        for family in 0..(people / 2) {
            let shared = next() + next() + next();
            for member in 0..2 {
                let who = family * 2 + member;
                let own = next() + next() + next();
                let genetic = 0.7 * shared + 0.71 * own;
                let level = next() + next() + next();
                for ear in 0..2 {
                    let noise = next() + next() + next();
                    value[who * 2 + ear] = 10.0
                        + total.sqrt()
                            * (genetic_share.sqrt() * genetic
                                + person_share.sqrt() * level
                                + (1.0 - genetic_share - person_share).sqrt() * noise);
                }
            }
        }

        let limit = {
            let mut sorted = value.clone();
            sorted.sort_by(f64::total_cmp);
            sorted[(rows * 3) / 4]
        };
        let censoring: Vec<Censoring> = value
            .iter()
            .map(|v| {
                if *v >= limit {
                    Censoring::Above
                } else {
                    Censoring::Measured
                }
            })
            .collect();
        let limits = vec![limit; rows];
        let design = DMatrix::from_element(rows, 1, 1.0);

        let model = TobitModel::build(&[additive, person], &value, &censoring, &limits, &design)
            .expect("two components over the same rows are a model");
        assert_eq!(model.components(), 2);

        let fit = model.fit().expect("a two-component fit converges here");
        assert_eq!(fit.coefficients.len(), 2, "one share per component");
        assert!(
            fit.coefficients.iter().sum::<f64>() <= 1.0,
            "the shares must leave the residual something: {:?}",
            fit.coefficients
        );
        // **On the sum, and deliberately not on each.** Across seeds this
        // design returns the person-level coefficient anywhere from 0.001 to
        // 0.6 against a truth of 0.3, while their sum stays near 0.7. Sibling
        // pairs with two records each do not tell an additive component from a
        // person-level one, which is a fact about the design and not about the
        // arithmetic -- it is what issue 35 exists to measure. An assertion on
        // the individual coefficient passed here only by the luck of the seed,
        // which is worse than no assertion at all.
        let together: f64 = fit.coefficients.iter().sum();
        assert!(
            (together - (genetic_share + person_share)).abs() < 0.2,
            "the components together came to {together:.3} where the truth is \
             {:.3}; the residual takes what is left, so their sum is what this \
             design does identify",
            genetic_share + person_share
        );
        assert_eq!(fit.estimator, "ml");
    }

    /// A small residual is a place the search must be able to reach.
    ///
    /// **The regression this exists for.** With the coefficients boxed
    /// individually and their sum refused past one, the feasible set is a
    /// simplex and the optimiser only understands a box. At a true split of
    /// 0.45 and 0.50 -- a residual of 0.05 -- the search began beside the
    /// refused region, the central difference read the infeasible value as a
    /// gradient the size of that jump, and the coefficients never moved from
    /// their starting values across every seed tried. `fit` still returned
    /// `Ok`, with `converged` false.
    ///
    /// A small residual is the audiogram case: additive, person-level and
    /// household terms together can leave little behind. Stick-breaking makes
    /// every point of the box a valid set, so there is no cliff to be stopped
    /// at.
    ///
    /// The assertion is on the **sum** of the two coefficients, not on each.
    /// Telling an additive component from a person-level one is a question
    /// about the design rather than the arithmetic, and issue 35 is the check
    /// that measures it; this test is about the search reaching the answer at
    /// all.
    #[test]
    fn the_search_reaches_a_split_that_leaves_almost_no_residual() {
        let people = 160;
        let rows = people * 2;
        let mut seed = 20_260_829_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
        };

        let mut additive = DMatrix::<f64>::zeros(rows, rows);
        let mut person = DMatrix::<f64>::zeros(rows, rows);
        for i in 0..rows {
            for j in 0..rows {
                let (a, b) = (i / 2, j / 2);
                if a == b {
                    person[(i, j)] = 1.0;
                    additive[(i, j)] = 1.0;
                } else if a / 2 == b / 2 {
                    additive[(i, j)] = 0.5;
                }
            }
        }

        let (genetic, level, total): (f64, f64, f64) = (0.45, 0.50, 4.0);
        let mut value = vec![0.0; rows];
        for family in 0..(people / 2) {
            let shared = next() + next() + next();
            for member in 0..2 {
                let who = family * 2 + member;
                let own = next() + next() + next();
                let breeding = 0.7 * shared + 0.71 * own;
                let personal = next() + next() + next();
                for ear in 0..2 {
                    let noise = next() + next() + next();
                    value[who * 2 + ear] = 10.0
                        + total.sqrt()
                            * (genetic.sqrt() * breeding
                                + level.sqrt() * personal
                                + (1.0 - genetic - level).sqrt() * noise);
                }
            }
        }

        let censoring = vec![Censoring::Measured; rows];
        let limits = vec![0.0; rows];
        let design = DMatrix::from_element(rows, 1, 1.0);
        let model = TobitModel::build(&[additive, person], &value, &censoring, &limits, &design)
            .expect("two components build");

        let fit = model.fit().expect("a fit is returned");
        let together: f64 = fit.coefficients.iter().sum();
        assert!(
            fit.converged,
            "the search stopped without converging, at {:?}",
            fit.coefficients
        );
        assert!(
            (together - (genetic + level)).abs() < 0.15,
            "the components together came to {together:.3} where the truth is \
             {:.3}; the residual is what is left, so getting their sum wrong is \
             getting the residual wrong",
            genetic + level
        );
        assert!(
            together < 1.0,
            "the coefficients must leave the residual something"
        );
    }

    /// An unscored verdict is withheld, not filled in from another model.
    ///
    /// The boundary rule and the fifty-fifty mixture were scored by a coverage
    /// simulation that ran at one component. With several, more than one
    /// coefficient can rest on nought at once, which is not that case. So the
    /// test refuses and the interval comes back with its boundary verdict
    /// absent -- an absent verdict says nobody has measured it, and a filled
    /// one would say somebody had.
    #[test]
    fn the_scored_verdicts_are_withheld_at_several_components() {
        let (relationship, value, censoring, limit, design) =
            simulate(40, 0.5, 1.0, 0.0, Some(0.5), 77);
        let person = DMatrix::<f64>::identity(relationship.nrows(), relationship.nrows());

        let one = TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        )
        .expect("one component builds");
        assert!(
            one.heritability_test().is_ok(),
            "the scored case must still be scored"
        );
        assert!(
            one.heritability_interval()
                .expect("an interval")
                .contains_lower_bound
                .is_some(),
            "at one component the boundary verdict is filled in"
        );

        let several =
            TobitModel::build(&[relationship, person], &value, &censoring, &limit, &design)
                .expect("two components build");
        assert_eq!(
            several.heritability_test().err(),
            Some("TOBIT_TEST_NOT_SCORED_FOR_SEVERAL_COMPONENTS"),
            "the mixture rule was never scored here, so no p-value is offered"
        );
        assert!(
            several
                .heritability_interval()
                .expect("an interval is still computable")
                .contains_lower_bound
                .is_none(),
            "the boundary verdict must be absent where nothing scored it"
        );
    }

    /// Every component is refused as mathematics, not just the first.
    ///
    /// The search may put all the variance on any one of them, so each has to
    /// be a covariance in its own right. Checking only the first would let a
    /// second that is not one reach the search, where it surfaces as a fit that
    /// found no feasible point rather than as the input error it is.
    #[test]
    fn a_second_component_that_is_not_a_covariance_is_refused() {
        let rows = 4;
        let good = DMatrix::<f64>::identity(rows, rows);
        let mut bad = DMatrix::<f64>::identity(rows, rows);
        // Symmetric, and not positive semi-definite.
        bad[(0, 1)] = 2.0;
        bad[(1, 0)] = 2.0;
        let value = vec![1.0, 2.0, 3.0, 4.0];
        let censoring = [Censoring::Measured; 4];
        let limits = vec![0.0; rows];
        let design = DMatrix::from_element(rows, 1, 1.0);
        assert_eq!(
            TobitModel::build(&[good, bad], &value, &censoring, &limits, &design).err(),
            Some("TOBIT_RELATIONSHIP_NOT_PSD"),
            "a second component that is not a covariance must be refused too"
        );
    }

    /// The largest block is the union's, not any one component's.
    ///
    /// The likelihood factorises over the blocks the components make together.
    /// Reporting one component's blocks would understate how many coordinates
    /// the region probability was actually asked for, which is the number a
    /// reader needs to tell whether a fit stayed inside what has been measured.
    #[test]
    fn the_largest_block_is_the_one_the_components_make_together() {
        let rows = 4;
        let first = DMatrix::<f64>::identity(rows, rows);
        let mut second = DMatrix::<f64>::identity(rows, rows);
        for i in 0..rows {
            for j in 0..rows {
                second[(i, j)] = if i == j { 1.0 } else { 0.5 };
            }
        }
        let value = vec![1.0, 2.0, 3.0, 4.0];
        let censoring = [Censoring::Measured; 4];
        let limits = vec![0.0; rows];
        let design = DMatrix::from_element(rows, 1, 1.0);

        let alone = TobitModel::build(
            std::slice::from_ref(&first),
            &value,
            &censoring,
            &limits,
            &design,
        )
        .expect("an identity is a covariance");
        let fit = alone.fit().expect("converges");
        assert_eq!(fit.largest_family, 1, "an identity leaves every row alone");

        let together = TobitModel::build(&[first, second], &value, &censoring, &limits, &design)
            .expect("both are covariances");
        let fit = together.fit().expect("converges");
        assert_eq!(
            fit.largest_family, rows,
            "the second component joins every row, so the block is all of them"
        );
    }

    /// A relationship of one is a real thing, not a mistake.
    ///
    /// Monozygotic twins have one. So do two records of the same person, which
    /// is what a person-level component is made of. `build` used to refuse any
    /// off-diagonal above 0.9, because the two-person quadrature it then used
    /// lost accuracy as the correlation approached one -- so the model could
    /// not fit a twin, and could not carry a person-level component at all.
    /// The integral was replaced on 29 August 2026 and the refusal went with
    /// it.
    #[test]
    fn a_relationship_of_one_is_accepted() {
        // Two people, two records each, a person's records carrying one with
        // each other and the ordinary coefficient with the other person's.
        let base = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let mut relationship = DMatrix::<f64>::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                relationship[(i, j)] = base[(i / 2, j / 2)];
            }
        }
        assert_eq!(relationship[(0, 1)], 1.0, "the two records of one person");

        let value = [1.0, 2.0, 3.0, 4.0];
        let censoring = [Censoring::Measured; 4];
        let limit = [0.0; 4];
        let design = DMatrix::from_element(4, 1, 1.0);
        let model = TobitModel::build(
            std::slice::from_ref(&relationship),
            &value,
            &censoring,
            &limit,
            &design,
        )
        .expect("a relationship of one is a covariance and must be accepted");
        assert!(
            model.loglik(&[0.4], 1.0, &[2.5]).is_some(),
            "the likelihood must evaluate where a person shares a record with themselves"
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
            TobitModel::build(
                std::slice::from_ref(&relationship),
                &value,
                &censoring,
                &limit,
                &design
            )
            .err(),
            Some("TOBIT_TOO_FEW_MEASURED_VALUES")
        );
    }
}

#[cfg(feature = "python")]
pub mod python;
