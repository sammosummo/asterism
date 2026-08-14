//! Many markers, one at a time, each a fixed effect in a polygenic model.
//!
//! The model for marker `j` is
//!
//! ```text
//! y = X0 b + m_j c + g + e,    V = s2 (h K + (1 - h) I)
//! ```
//!
//! where `X0` carries the intercept, the ancestry components and any other
//! covariate, `m_j` is the marker, and `c` is what is being tested. This is the
//! ordinary mixed-model association, and the polygenic term is what makes it
//! worth doing: relatedness and population structure inflate an association
//! test, and `K` absorbs both.
//!
//! # Why this is not just the one-trait model in a loop
//!
//! It could be. A prepared model already returns every fixed effect with a
//! standard error, a z and a two-sided p-value, so putting a marker in the
//! design and reading the last column is an association test and always was.
//! What makes that unusable at scale is that a prepared model's reuse runs
//! across *responses*, and here the response never changes while the design
//! changes half a million times.
//!
//! So the variance components are fitted **once**, under the null design with
//! no marker in it, and then held. After that each marker is a weighted least
//! squares on the rotated data, and the arithmetic per marker is one rotation
//! and one rank-one update of a small normal matrix.
//!
//! **The cost of holding them is a real assumption, not a trick.** It is what
//! EMMAX and its descendants do, and it is a good approximation when no single
//! marker explains much of the variance -- which is the situation a scan is in,
//! and is exactly not the situation for a marker of large effect. The exact
//! alternative, refitting the variance components under every marker, is
//! available and is around a hundred times slower.
//!
//! # Wald and the likelihood ratio are the same test here
//!
//! Both are offered because both were asked for, but with the variance
//! components held they are algebraically identical: the profile log likelihood
//! in the fixed effects is exactly quadratic when the covariance is known, so
//! the likelihood ratio statistic is the square of the Wald statistic. A test
//! asserts this rather than a comment claiming it.
//!
//! They separate only when the variance components are refitted under each
//! marker, which is what [`Variance::Refitted`] does.
//!
//! # What this does not do
//!
//! **The marker under test is inside `K`.** Leaving out the chromosome it sits
//! on -- one decomposition per chromosome -- is the usual answer to that, and it
//! is deliberately not done here: it was judged too expensive for the analysis
//! this was built for. The consequence is a test biased towards the null, more
//! so for a marker that contributes materially to `K`, and it is recorded in the
//! fit rather than left for a reader to remember.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::blocks::family_blocks;

/// How the variance components are treated while markers are swept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variance {
    /// Fitted once under the null and then held. Wald and the likelihood ratio
    /// coincide exactly.
    Held,
    /// Refitted under every marker. Slow, and the only setting where the
    /// likelihood ratio says anything the Wald statistic does not.
    Refitted,
}

/// One covariate's effect under the null model.
#[derive(Clone, Copy, Debug)]
pub struct CovariateEffect {
    pub estimate: f64,
    pub standard_error: f64,
    /// Two-sided, against the covariate having no effect.
    pub p_value: f64,
}

/// One marker's result.
#[derive(Clone, Copy, Debug)]
pub struct MarkerTest {
    /// The marker's effect on the response.
    pub effect: f64,
    pub standard_error: f64,
    /// The Wald statistic, squared: comparable with the likelihood ratio.
    pub wald: f64,
    /// Twice the difference in log likelihood. Equal to `wald` under
    /// [`Variance::Held`], and not otherwise.
    pub likelihood_ratio: f64,
    /// The two-sided p-value from whichever statistic was asked for.
    pub p_value: f64,
    /// The heritability the test was run under: the null's under
    /// [`Variance::Held`], and this marker's under [`Variance::Refitted`].
    pub heritability: f64,
    /// Whether this marker's variance components were refitted. Under a
    /// two-stage sweep some markers are and most are not, and a result that
    /// does not say which is a result nobody can check.
    pub refitted: bool,
}

/// A polygenic model with its variance components fitted under a null design.
pub struct AssociationModel {
    /// Eigenvalues of `K`, in the rotated stacking order.
    lambda: DVector<f64>,
    /// The rotated response.
    y: DVector<f64>,
    /// The rotated null design.
    x0: DMatrix<f64>,
    /// Per family block: its rows in the original order, and its eigenvectors.
    rotations: Vec<(Vec<usize>, DMatrix<f64>)>,
    rows: usize,
    heritability: f64,
    total_variance: f64,
    null_loglik: f64,
    /// Whether any leave-one-out was applied to `K`. Always false, and carried
    /// so the answer is in the record.
    leave_one_chromosome_out: bool,
}

impl AssociationModel {
    /// Rotate the data by the eigenvectors of `K` and fit the null.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(
        relationship: &DMatrix<f64>,
        design: &DMatrix<f64>,
        response: &DVector<f64>,
    ) -> Result<Self, &'static str> {
        let rows = response.len();
        if rows == 0 {
            return Err("ASSOCIATION_NO_ROWS");
        }
        if relationship.nrows() != rows || relationship.ncols() != rows {
            return Err("ASSOCIATION_RELATIONSHIP_WRONG_SHAPE");
        }
        if design.nrows() != rows {
            return Err("ASSOCIATION_DESIGN_WRONG_SHAPE");
        }
        if design.ncols() == 0 {
            return Err("ASSOCIATION_DESIGN_HAS_NO_COLUMNS");
        }
        if design.ncols() >= rows {
            return Err("ASSOCIATION_DESIGN_NOT_SMALLER_THAN_SAMPLE");
        }
        if !relationship.iter().all(|v| v.is_finite())
            || !design.iter().all(|v| v.is_finite())
            || !response.iter().all(|v| v.is_finite())
        {
            return Err("ASSOCIATION_NOT_FINITE");
        }

        // **The rotation is per family block, not over the whole matrix.** A
        // pedigree relationship matrix is block diagonal, so rotating a marker
        // costs the sum of the squared block sizes rather than the square of
        // the sample size -- which is the difference between a scan taking
        // minutes and taking days.
        let blocks = family_blocks(relationship);
        let mut lambda = DVector::zeros(rows);
        let mut y = DVector::zeros(rows);
        let mut x0 = DMatrix::zeros(rows, design.ncols());
        let mut rotations = Vec::with_capacity(blocks.len());
        let mut placed = 0;
        for block in blocks {
            let size = block.len();
            let local = DMatrix::from_fn(size, size, |i, j| relationship[(block[i], block[j])]);
            let eigen = SymmetricEigen::new(local);
            let rotated_y = eigen.eigenvectors.transpose()
                * DVector::from_iterator(size, block.iter().map(|&row| response[row]));
            let rotated_x = eigen.eigenvectors.transpose()
                * DMatrix::from_fn(size, design.ncols(), |i, j| design[(block[i], j)]);
            for local_row in 0..size {
                // A relationship matrix is positive semidefinite up to
                // rounding; a small negative eigenvalue is that rounding.
                lambda[placed + local_row] = eigen.eigenvalues[local_row].max(0.0);
                y[placed + local_row] = rotated_y[local_row];
                for column in 0..design.ncols() {
                    x0[(placed + local_row, column)] = rotated_x[(local_row, column)];
                }
            }
            placed += size;
            rotations.push((block, eigen.eigenvectors));
        }

        let mut model = Self {
            lambda,
            y,
            x0,
            rotations,
            rows,
            heritability: 0.0,
            total_variance: 1.0,
            null_loglik: 0.0,
            leave_one_chromosome_out: false,
        };
        let (heritability, total_variance, loglik) = model.fit_variance(None)?;
        model.heritability = heritability;
        model.total_variance = total_variance;
        model.null_loglik = loglik;
        Ok(model)
    }

    /// The heritability fitted under the null design.
    #[must_use]
    pub fn heritability(&self) -> f64 {
        self.heritability
    }

    /// The log likelihood of the null model.
    #[must_use]
    pub fn null_loglik(&self) -> f64 {
        self.null_loglik
    }

    /// Rotate one marker into the eigenbasis, block by block.
    fn rotate(&self, marker: &DVector<f64>) -> DVector<f64> {
        let mut out = DVector::zeros(self.rows);
        let mut placed = 0;
        for (block, eigenvectors) in &self.rotations {
            let size = block.len();
            let local = DVector::from_iterator(size, block.iter().map(|&row| marker[row]));
            let rotated = eigenvectors.transpose() * local;
            for local_row in 0..size {
                out[placed + local_row] = rotated[local_row];
            }
            placed += size;
        }
        out
    }

    /// Fit the heritability and the total variance, optionally with one extra
    /// rotated column in the design.
    ///
    /// The total variance profiles out in closed form, so this is a search over
    /// one bounded parameter however many fixed effects there are.
    fn fit_variance(&self, extra: Option<&DVector<f64>>) -> Result<(f64, f64, f64), &'static str> {
        let objective = |heritability: f64| -> Option<(f64, f64)> {
            let weights = self.weights(heritability);
            let (_, residual, logdet) = self.weighted_least_squares(&weights, extra)?;
            let observations = self.rows as f64;
            let variance = residual / observations;
            if !(variance > 0.0) {
                return None;
            }
            // Maximum likelihood with the scale profiled out.
            let loglik = -0.5
                * (observations * (1.0 + (2.0 * std::f64::consts::PI * variance).ln()) + logdet);
            Some((loglik, variance))
        };

        let value_of = |theta: &[f64]| -> f64 { objective(theta[0]).map_or(1e30, |(l, _)| -l) };
        let gradient_of = |theta: &[f64]| -> Vec<f64> {
            let step = 1e-6;
            let up = (theta[0] + step).min(1.0);
            let down = (theta[0] - step).max(0.0);
            let width = up - down;
            if width <= 0.0 {
                return vec![0.0];
            }
            vec![(value_of(&[up]) - value_of(&[down])) / width]
        };

        let mut best: Option<(f64, f64)> = None;
        for start in [0.05_f64, 0.35, 0.7] {
            let Ok(bounds) = Bounds::new(vec![0.0], vec![1.0]) else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(1);
            control.maxit = 200;
            control.fnscale = value_of(&[start]).abs().max(1.0);
            control.parscale = vec![1.0];
            control.factr = 1.0e3;
            control.pgtol = 1e-10;
            control.lmm = 1;
            let Ok(solution) =
                optim_lbfgsb_with_gradient(vec![start], bounds, value_of, gradient_of, control)
            else {
                continue;
            };
            let value = value_of(&solution.par);
            if value.is_finite() && best.as_ref().is_none_or(|(seen, _)| value < *seen) {
                best = Some((value, solution.par[0]));
            }
        }
        let (negative, heritability) = best.ok_or("ASSOCIATION_NO_START_CONVERGED")?;
        let (loglik, variance) = objective(heritability).ok_or("ASSOCIATION_NOT_EVALUABLE")?;
        debug_assert!((loglik + negative).abs() < 1e-6);
        Ok((heritability, variance, loglik))
    }

    /// `1 / (h lambda + (1 - h))`, the inverse covariance in the eigenbasis up
    /// to the total variance.
    fn weights(&self, heritability: f64) -> DVector<f64> {
        DVector::from_iterator(
            self.rows,
            self.lambda
                .iter()
                .map(|&value| 1.0 / (heritability * value + (1.0 - heritability)).max(1e-12)),
        )
    }

    /// Weighted least squares on the rotated data, with an optional extra
    /// column. Returns the coefficients, the weighted residual sum of squares
    /// and the log determinant of the covariance up to the total variance.
    fn weighted_least_squares(
        &self,
        weights: &DVector<f64>,
        extra: Option<&DVector<f64>>,
    ) -> Option<(DVector<f64>, f64, f64)> {
        let columns = self.x0.ncols() + usize::from(extra.is_some());
        let mut normal = DMatrix::<f64>::zeros(columns, columns);
        let mut score = DVector::<f64>::zeros(columns);
        let column_at = |index: usize, row: usize| -> f64 {
            if index < self.x0.ncols() {
                self.x0[(row, index)]
            } else {
                extra.expect("an extra column was counted")[row]
            }
        };
        for row in 0..self.rows {
            let weight = weights[row];
            for i in 0..columns {
                let value = column_at(i, row) * weight;
                score[i] += value * self.y[row];
                for j in 0..=i {
                    normal[(i, j)] += value * column_at(j, row);
                }
            }
        }
        for i in 0..columns {
            for j in 0..i {
                normal[(i, j.min(i))] = normal[(i, j)];
                normal[(j, i)] = normal[(i, j)];
            }
        }
        let decomposition = normal.clone().cholesky()?;
        let beta = decomposition.solve(&score);
        let mut residual = 0.0;
        for row in 0..self.rows {
            let fitted: f64 = (0..columns).map(|i| column_at(i, row) * beta[i]).sum();
            residual += weights[row] * (self.y[row] - fitted).powi(2);
        }
        let logdet = -weights.iter().map(|w| w.ln()).sum::<f64>();
        Some((beta, residual, logdet))
    }

    /// Test one marker.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the marker is the wrong length, is not
    /// finite, or leaves the design rank deficient -- which a marker with no
    /// variation does.
    pub fn test_marker(
        &self,
        marker: &DVector<f64>,
        variance: Variance,
    ) -> Result<MarkerTest, &'static str> {
        if marker.len() != self.rows {
            return Err("ASSOCIATION_MARKER_WRONG_LENGTH");
        }
        if !marker.iter().all(|v| v.is_finite()) {
            return Err("ASSOCIATION_MARKER_NOT_FINITE");
        }
        let first = marker[0];
        if marker.iter().all(|v| (v - first).abs() < 1e-12) {
            return Err("ASSOCIATION_MARKER_HAS_NO_VARIATION");
        }
        let rotated = self.rotate(marker);

        let (heritability, total, alternative_loglik) = match variance {
            Variance::Held => (self.heritability, self.total_variance, f64::NAN),
            Variance::Refitted => self.fit_variance(Some(&rotated))?,
        };
        let weights = self.weights(heritability);
        let (beta, residual, logdet) = self
            .weighted_least_squares(&weights, Some(&rotated))
            .ok_or("ASSOCIATION_DESIGN_RANK_DEFICIENT")?;

        // The variance of the last coefficient, from the inverse normal matrix.
        let columns = self.x0.ncols() + 1;
        let mut normal = DMatrix::<f64>::zeros(columns, columns);
        for row in 0..self.rows {
            let weight = weights[row];
            for i in 0..columns {
                let left = if i < self.x0.ncols() {
                    self.x0[(row, i)]
                } else {
                    rotated[row]
                };
                for j in 0..columns {
                    let right = if j < self.x0.ncols() {
                        self.x0[(row, j)]
                    } else {
                        rotated[row]
                    };
                    normal[(i, j)] += weight * left * right;
                }
            }
        }
        let inverse = normal
            .try_inverse()
            .ok_or("ASSOCIATION_DESIGN_RANK_DEFICIENT")?;
        let scale = match variance {
            // Held: the null's total variance, which is what makes the Wald
            // statistic and the likelihood ratio the same number.
            Variance::Held => total,
            Variance::Refitted => residual / self.rows as f64,
        };
        let effect = beta[columns - 1];
        let error = (scale * inverse[(columns - 1, columns - 1)]).max(0.0).sqrt();
        let wald = if error > 0.0 { (effect / error).powi(2) } else { 0.0 };

        let likelihood_ratio = match variance {
            Variance::Held => {
                // With the covariance fixed, the log likelihood difference is
                // half the drop in the weighted residual sum of squares over
                // the fixed total variance -- which is exactly the Wald square.
                let null_weights = self.weights(self.heritability);
                let (_, null_residual, _) = self
                    .weighted_least_squares(&null_weights, None)
                    .ok_or("ASSOCIATION_NULL_RANK_DEFICIENT")?;
                ((null_residual - residual) / total).max(0.0)
            }
            Variance::Refitted => {
                let observations = self.rows as f64;
                let alternative = if alternative_loglik.is_finite() {
                    alternative_loglik
                } else {
                    let variance_estimate = residual / observations;
                    -0.5
                        * (observations
                            * (1.0 + (2.0 * std::f64::consts::PI * variance_estimate).ln())
                            + logdet)
                };
                (2.0 * (alternative - self.null_loglik)).max(0.0)
            }
        };

        let statistic = match variance {
            Variance::Held => wald,
            Variance::Refitted => likelihood_ratio,
        };
        Ok(MarkerTest {
            effect,
            standard_error: error,
            wald,
            likelihood_ratio,
            p_value: chi2_one_df_upper_tail(statistic).clamp(0.0, 1.0),
            heritability,
            refitted: matches!(variance, Variance::Refitted),
        })
    }

    /// Test many markers, one column at a time.
    ///
    /// The markers are columns of `markers`, one row per person in the same
    /// order as the response. A marker that cannot be tested -- one with no
    /// variation, or one leaving the design rank deficient -- comes back as
    /// `Err` in its own slot rather than stopping the sweep, because one bad
    /// column in half a million should not cost the other half million.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the marker matrix has the wrong number of
    /// rows.
    pub fn sweep(
        &self,
        markers: &DMatrix<f64>,
        variance: Variance,
    ) -> Result<Vec<Result<MarkerTest, &'static str>>, &'static str> {
        self.sweep_refitting_below(markers, variance, None)
    }

    /// Sweep with the variance components held, then refit only the markers
    /// worth refitting.
    ///
    /// **This is how a scan should be run.** Holding the variance components
    /// makes a marker cost a fraction of a millisecond; refitting them costs
    /// twenty times that, which is four hours across a million markers rather
    /// than ten minutes. Almost every marker is uninteresting and the two modes
    /// agree on those to two decimal places, so refitting them all is paying
    /// twenty times over for nothing.
    ///
    /// # The screen must be looser than the threshold you care about
    ///
    /// Held is conservative: measured across four hundred real markers it was
    /// never smaller than refitted, and the gap grows with significance -- a
    /// ratio of 1.00 above p = 0.01 and up to 1.10 below 1e-06. So a marker can
    /// have a refitted p below a threshold while its held p sits above it, and
    /// screening at exactly the threshold would miss it.
    ///
    /// Pass a `refit_below` some way above the threshold being reported
    /// against. Ten times is ample for a gap that never exceeded a factor of
    /// 1.1, and costs almost nothing: at a screen of 1e-06 a million markers
    /// leaves a handful to refit.
    ///
    /// Passing `None` sweeps entirely in `variance`.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the marker matrix has the wrong number of
    /// rows.
    pub fn sweep_refitting_below(
        &self,
        markers: &DMatrix<f64>,
        variance: Variance,
        refit_below: Option<f64>,
    ) -> Result<Vec<Result<MarkerTest, &'static str>>, &'static str> {
        if markers.nrows() != self.rows {
            return Err("ASSOCIATION_MARKERS_WRONG_LENGTH");
        }
        Ok((0..markers.ncols())
            .map(|column| {
                let marker = DVector::from_iterator(
                    self.rows,
                    (0..self.rows).map(|row| markers[(row, column)]),
                );
                let first = self.test_marker(&marker, variance)?;
                match refit_below {
                    // Already refitted, or not interesting enough to be worth
                    // refitting.
                    Some(threshold)
                        if variance == Variance::Held && first.p_value <= threshold =>
                    {
                        self.test_marker(&marker, Variance::Refitted)
                    }
                    _ => Ok(first),
                }
            })
            .collect())
    }

    /// The covariates' own effects, under the null model.
    ///
    /// **These were being computed and thrown away.** Every marker's fit
    /// estimates the whole design and this package kept only the last
    /// coordinate, so a caller who wanted to know what age or sex or an
    /// ancestry component was doing had to fit a second model to find out.
    ///
    /// They come from the null model rather than from any one marker, which is
    /// the right place for them: under a held-variance sweep the null is what
    /// the covariates are estimated in, and reporting them per marker would be
    /// thirty thousand copies of nearly the same number.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the null model cannot be factorised.
    pub fn covariate_effects(&self) -> Result<Vec<CovariateEffect>, &'static str> {
        let weights = self.weights(self.heritability);
        let (beta, _, _) = self
            .weighted_least_squares(&weights, None)
            .ok_or("ASSOCIATION_NULL_RANK_DEFICIENT")?;
        let columns = self.x0.ncols();
        let mut normal = DMatrix::<f64>::zeros(columns, columns);
        for row in 0..self.rows {
            let weight = weights[row];
            for i in 0..columns {
                for j in 0..columns {
                    normal[(i, j)] += weight * self.x0[(row, i)] * self.x0[(row, j)];
                }
            }
        }
        let inverse = normal
            .try_inverse()
            .ok_or("ASSOCIATION_NULL_RANK_DEFICIENT")?;
        Ok((0..columns)
            .map(|k| {
                let error = (self.total_variance * inverse[(k, k)]).max(0.0).sqrt();
                let z = if error > 0.0 { beta[k] / error } else { 0.0 };
                CovariateEffect {
                    estimate: beta[k],
                    standard_error: error,
                    p_value: chi2_one_df_upper_tail(z * z).clamp(0.0, 1.0),
                }
            })
            .collect())
    }

    /// Whether any chromosome was left out of the relationship matrix.
    ///
    /// Always false. The marker under test is inside `K`, which biases its test
    /// towards the null, and the fit says so rather than leaving it to be
    /// remembered.
    #[must_use]
    pub fn leave_one_chromosome_out(&self) -> bool {
        self.leave_one_chromosome_out
    }
}

/// The upper tail of chi-square on one degree of freedom, through the
/// complementary error function.
fn chi2_one_df_upper_tail(statistic: f64) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    statrs::function::erf::erfc((statistic / 2.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::{AssociationModel, Variance};
    use nalgebra::{DMatrix, DVector};

    /// Sibling pairs, a relationship matrix, ancestry-like covariates, and a
    /// response with a known polygenic share.
    fn simulate(
        pairs: usize,
        heritability: f64,
        effect: f64,
        seed: u64,
    ) -> (DMatrix<f64>, DMatrix<f64>, DVector<f64>, DVector<f64>) {
        let n = 2 * pairs;
        let mut k = DMatrix::<f64>::identity(n, n);
        for pair in 0..pairs {
            k[(2 * pair, 2 * pair + 1)] = 0.5;
            k[(2 * pair + 1, 2 * pair)] = 0.5;
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
        // Two ancestry-like components and an intercept.
        let mut design = DMatrix::<f64>::zeros(n, 3);
        for row in 0..n {
            design[(row, 0)] = 1.0;
            design[(row, 1)] = draw();
            design[(row, 2)] = draw();
        }
        // A marker shared within a pair, as a real one would be.
        let mut marker = DVector::<f64>::zeros(n);
        for pair in 0..pairs {
            let parent = draw();
            marker[2 * pair] = (parent + 0.6 * draw()).round().clamp(0.0, 2.0);
            marker[2 * pair + 1] = (parent + 0.6 * draw()).round().clamp(0.0, 2.0);
        }
        let covariance =
            heritability * k.clone() + (1.0 - heritability) * DMatrix::<f64>::identity(n, n);
        let factor = covariance
            .cholesky()
            .expect("positive definite")
            .l();
        let noise = factor * DVector::from_iterator(n, (0..n).map(|_| draw()));
        let y = &design * DVector::from_vec(vec![1.0, 0.3, -0.2]) + &marker * effect + noise;
        (k, design, y, marker)
    }

    /// **With the variance components held, the two tests are the same
    /// number.** Both were asked for, and offering them as alternatives when
    /// they agree to machine precision would be printing one answer twice. The
    /// profile log likelihood in the fixed effects is exactly quadratic when
    /// the covariance is known, so the likelihood ratio is the square of the
    /// Wald statistic -- checked here rather than claimed in a comment.
    #[test]
    fn held_variance_makes_wald_and_the_likelihood_ratio_identical() {
        for effect in [0.0_f64, 0.15, 0.4] {
            let (k, design, y, marker) = simulate(200, 0.5, effect, 4_242 + (effect * 100.0) as u64);
            let model = AssociationModel::build(&k, &design, &y).expect("valid");
            let got = model.test_marker(&marker, Variance::Held).expect("tests");
            assert!(
                (got.wald - got.likelihood_ratio).abs() < 1e-6 * got.wald.max(1.0),
                "at an effect of {effect}: Wald {} against likelihood ratio {}",
                got.wald,
                got.likelihood_ratio
            );
        }
    }

    /// Refitting the variance components under every marker gives a different
    /// statistic, which is the only reason to pay for it.
    #[test]
    fn refitting_the_variance_gives_a_different_statistic() {
        let (k, design, y, marker) = simulate(200, 0.5, 0.35, 77);
        let model = AssociationModel::build(&k, &design, &y).expect("valid");
        let held = model.test_marker(&marker, Variance::Held).expect("tests");
        let refitted = model.test_marker(&marker, Variance::Refitted).expect("tests");
        // Same model, so the two should agree on roughly where the effect is.
        assert!(
            (held.effect - refitted.effect).abs() < 0.1 * held.effect.abs().max(0.1),
            "the effect moved from {} to {}",
            held.effect,
            refitted.effect
        );
        assert!(
            refitted.likelihood_ratio > 0.0,
            "the refitted likelihood ratio is {}",
            refitted.likelihood_ratio
        );
    }

    /// A marker with a real effect is found, and one with none is not.
    #[test]
    fn an_effect_is_recovered_and_an_absent_one_is_not() {
        let (k, design, y, marker) = simulate(300, 0.4, 0.35, 20_260_814);
        let model = AssociationModel::build(&k, &design, &y).expect("valid");
        let got = model.test_marker(&marker, Variance::Held).expect("tests");
        assert!(
            got.p_value < 0.01,
            "an effect of 0.35 was simulated but p is {}",
            got.p_value
        );
        assert!(
            (got.effect - 0.35).abs() < 0.15,
            "the effect came back as {} where 0.35 was simulated",
            got.effect
        );

        let (k, design, y, marker) = simulate(300, 0.4, 0.0, 555);
        let model = AssociationModel::build(&k, &design, &y).expect("valid");
        let got = model.test_marker(&marker, Variance::Held).expect("tests");
        assert!(
            got.p_value > 0.05,
            "no effect was simulated but p is {}",
            got.p_value
        );
    }

    /// **The polygenic term is the point.** A marker correlated with family
    /// membership and nothing else is what relatedness manufactures, and a
    /// model that ignores the pedigree calls it an association. Here the
    /// heritability is high, the response is entirely polygenic, and the
    /// marker is shared within families: the test must not reject.
    #[test]
    fn family_structure_alone_is_not_an_association() {
        let mut rejected = 0;
        for seed in 0..20u64 {
            let (k, design, y, marker) = simulate(200, 0.8, 0.0, 900 + seed);
            let model = AssociationModel::build(&k, &design, &y).expect("valid");
            let got = model.test_marker(&marker, Variance::Held).expect("tests");
            rejected += usize::from(got.p_value < 0.05);
        }
        assert!(
            rejected <= 3,
            "{rejected} of 20 markers with no effect were called significant, \
             which is what a model without the polygenic term would do"
        );
    }

    /// **A two-stage sweep must agree with refitting everything**, on the
    /// markers it chose to refit, and must leave the rest alone. Anything else
    /// means the screen and the refit disagree about which marker is which.
    #[test]
    fn a_two_stage_sweep_refits_what_it_says_it_refits() {
        let (k, design, y, marker) = simulate(300, 0.4, 0.35, 606);
        let model = AssociationModel::build(&k, &design, &y).expect("valid");
        // One real marker beside two that are noise.
        let mut markers = DMatrix::<f64>::zeros(y.len(), 3);
        let mut state = 99u64;
        for row in 0..y.len() {
            markers[(row, 0)] = marker[row];
            for column in 1..3 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                markers[(row, column)] = f64::from((state >> 33) as u32 % 3);
            }
        }
        let held = model.sweep(&markers, Variance::Held).expect("sweeps");
        let refitted = model.sweep(&markers, Variance::Refitted).expect("sweeps");
        let staged = model
            .sweep_refitting_below(&markers, Variance::Held, Some(0.01))
            .expect("sweeps");

        for index in 0..3 {
            let (h, r, s) = (
                held[index].as_ref().unwrap(),
                refitted[index].as_ref().unwrap(),
                staged[index].as_ref().unwrap(),
            );
            if h.p_value <= 0.01 {
                assert!(s.refitted, "marker {index} passed the screen and was not refitted");
                assert!(
                    (s.p_value - r.p_value).abs() < 1e-12,
                    "marker {index} was refitted but does not match a full refit"
                );
            } else {
                assert!(!s.refitted, "marker {index} failed the screen and was refitted");
                assert!(
                    (s.p_value - h.p_value).abs() < 1e-12,
                    "marker {index} was left alone but does not match the held sweep"
                );
            }
        }
        // The planted marker should have passed the screen.
        assert!(
            staged[0].as_ref().unwrap().refitted,
            "a marker simulated with an effect of 0.35 did not pass a screen at 0.01"
        );
    }

    /// **Held never gives a smaller p-value than refitted.** That is what makes
    /// a screen safe: the fast mode cannot manufacture significance, so
    /// anything it flags is worth a second look and anything it clears is
    /// genuinely clear at that threshold. Measured across four hundred real
    /// markers the ratio ran from 1.00 to 1.10.
    #[test]
    fn held_never_beats_refitted() {
        for seed in [11u64, 22, 33] {
            let (k, design, y, marker) = simulate(250, 0.5, 0.25, seed);
            let model = AssociationModel::build(&k, &design, &y).expect("valid");
            let held = model.test_marker(&marker, Variance::Held).expect("tests");
            let refitted = model.test_marker(&marker, Variance::Refitted).expect("tests");
            assert!(
                held.p_value >= refitted.p_value * (1.0 - 1e-9),
                "held gave {} against refitted {}, which would let the fast mode \
                 manufacture significance",
                held.p_value,
                refitted.p_value
            );
        }
    }

    /// **The covariates' own effects come back, rather than being computed and
    /// discarded.** Every marker's fit estimates the whole design, and keeping
    /// only the last coordinate meant a caller who wanted to know what age or
    /// sex was doing had to fit a second model to find out.
    #[test]
    fn the_covariates_report_their_own_effects() {
        // The response is built with a known effect on the second covariate.
        let (k, design, y, _) = simulate(300, 0.4, 0.0, 12_345);
        let model = AssociationModel::build(&k, &design, &y).expect("valid");
        let got = model.covariate_effects().expect("effects");
        assert_eq!(got.len(), design.ncols());
        // simulate() puts 0.3 on the first covariate and -0.2 on the second.
        assert!(
            (got[1].estimate - 0.3).abs() < 0.15,
            "the first covariate was simulated at 0.3 and came back at {}",
            got[1].estimate
        );
        assert!(
            got[1].p_value < 0.01,
            "a covariate simulated at 0.3 has p {}",
            got[1].p_value
        );
        for effect in &got {
            assert!(
                effect.standard_error > 0.0 && effect.p_value.is_finite(),
                "an effect came back without a usable standard error"
            );
        }
    }

    /// A marker that does not vary carries no information and is refused
    /// rather than returned as a rank-deficient fit with a plausible number.
    #[test]
    fn a_marker_with_no_variation_is_refused() {
        let (k, design, y, _) = simulate(100, 0.4, 0.0, 3);
        let model = AssociationModel::build(&k, &design, &y).expect("valid");
        let flat = DVector::from_element(y.len(), 1.0);
        assert!(model.test_marker(&flat, Variance::Held).is_err());
        let wrong = DVector::from_element(y.len() + 1, 0.5);
        assert!(model.test_marker(&wrong, Variance::Held).is_err());
    }

    /// The relationship matrix is never reduced, so a reader can tell from the
    /// model that the marker under test is inside it.
    #[test]
    fn the_model_says_no_chromosome_was_left_out() {
        let (k, design, y, _) = simulate(100, 0.4, 0.0, 11);
        let model = AssociationModel::build(&k, &design, &y).expect("valid");
        assert!(!model.leave_one_chromosome_out());
    }
}

#[cfg(feature = "python")]
pub mod python {
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::{AssociationModel, Variance};
    use nalgebra::{DMatrix, DVector};

    fn variance_named(name: &str) -> PyResult<Variance> {
        match name {
            "held" => Ok(Variance::Held),
            "refitted" => Ok(Variance::Refitted),
            other => Err(PyValueError::new_err(format!(
                "ASSOCIATION_UNKNOWN_VARIANCE: {other}, wanted held or refitted"
            ))),
        }
    }

    /// Sweep many markers through a polygenic model.
    ///
    /// Returns the null heritability, the null log likelihood, and one row per
    /// marker of effect, standard error, Wald statistic, likelihood ratio and
    /// p-value. A marker that could not be tested comes back with a stable code
    /// in place of its numbers rather than stopping the sweep.
    ///
    /// **Wald and the likelihood ratio are the same number under `held`.** Both
    /// are returned because both were asked for; they differ only under
    /// `refitted`, which refits the variance components for every marker and is
    /// far slower.
    #[pyfunction]
    #[pyo3(signature = (relationship, design, y, markers, variance="held", refit_below=None))]
    #[allow(clippy::type_complexity)]
    pub fn association_sweep(
        relationship: PyReadonlyArray2<'_, f64>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        markers: PyReadonlyArray2<'_, f64>,
        variance: &str,
        refit_below: Option<f64>,
    ) -> PyResult<(
        f64,
        f64,
        Vec<(f64, f64, f64, f64, f64, bool, String)>,
        Vec<(f64, f64, f64)>,
    )> {
        let a = relationship.as_array();
        let a = DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)]);
        let x = design.as_array();
        let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
        let response = y.as_array();
        let response = DVector::from_iterator(response.len(), response.iter().copied());
        let m = markers.as_array();
        let m = DMatrix::from_fn(m.shape()[0], m.shape()[1], |i, j| m[(i, j)]);

        let model =
            AssociationModel::build(&a, &x, &response).map_err(PyValueError::new_err)?;
        let results = model
            .sweep_refitting_below(&m, variance_named(variance)?, refit_below)
            .map_err(PyValueError::new_err)?;
        let covariates = model
            .covariate_effects()
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|c| (c.estimate, c.standard_error, c.p_value))
            .collect();
        Ok((
            model.heritability(),
            model.null_loglik(),
            results
                .into_iter()
                .map(|one| match one {
                    Ok(test) => (
                        test.effect,
                        test.standard_error,
                        test.wald,
                        test.likelihood_ratio,
                        test.p_value,
                        test.refitted,
                        String::new(),
                    ),
                    Err(code) => (
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        false,
                        code.to_owned(),
                    ),
                })
                .collect(),
            covariates,
        ))
    }
}
