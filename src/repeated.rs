//! Repeated measures at fixed positions on an ordered continuum.
//!
//! # What this is
//!
//! Every person is measured at the same `T` positions, `R` times over. In the
//! analysis this was built for the positions are audiometric frequencies and
//! the replicate is the ear, but nothing here knows that: the crate takes
//! positions on a line and a replicate index, and the caller decides what they
//! mean.
//!
//! ```text
//! y[p, r, t] = x[p, r]' b[., t]  +  sum_j u[j, p, t]  +  e[p, r, t]
//! ```
//!
//! The person-level effects `u[j, p, .]` are shared by every replicate of that
//! person, which is the point of the design: a genotype does not know left ear
//! from right, so the loadings are shared and left-right asymmetry gets a level
//! of its own rather than being averaged away. Each component carries a full
//! `T x T` covariance across positions,
//!
//! ```text
//! V = sum_j (K_j (x) J_R (x) S_j)  +  I (x) I_R (x) S_e
//! ```
//!
//! where `J_R` is the `R x R` matrix of ones and `(x)` is a Kronecker product.
//!
//! **This model has no name yet.** It is called what it does, because naming it
//! is deferred until it works and the name is meant to say what the biology is
//! supposed to be. The reasoning for the fitting route is in ADR 0010.
//!
//! # What is here, and what is not yet
//!
//! This is the skeleton: complete, balanced, uncensored data, and a free
//! covariance per component. Censoring comes next and the covariance kernel
//! after it, in that order, because each rests on the one before. Until the
//! kernel arrives every `S_j` is unstructured, so a fit at seventeen positions
//! estimates 153 numbers per component and nobody should read a single one of
//! them. What the skeleton is for is showing that the machinery lands on the
//! answer where the answer is already known.
//!
//! # Why the rotation
//!
//! Fitting this directly is out of reach. The audiogram design carries 125
//! parameters and every censored likelihood in this crate takes its gradient by
//! central difference, so one gradient step is 250 evaluations of a 9.4 Gflop
//! objective. ADR 0010 has the arithmetic; the conclusion is tens of hours per
//! fit.
//!
//! The complete-data problem is a different size altogether. Rotating the `R`
//! replicates of a person into their mean and `R - 1` contrasts turns `J_R`
//! into `diag(R, 0, ..., 0)`, which splits the problem in two. The contrasts
//! carry the replicate-level covariance alone and are independent of
//! everything. The mean channel is an ordinary Kronecker form, and the
//! eigenvectors of the first component's matrix diagonalise it -- provided the
//! other components are diagonal in the same basis, which
//! [`RepeatedModel::build`] checks rather than assumes.
//!
//! What is left is `P * R` blocks of `T x T`. On the audiogram design that is
//! 804 blocks of 17, about 1.3 Mflop against 9.4 Gflop, with the one `P x P`
//! eigendecomposition done in `build` rather than once per evaluation.
//!
//! **The rotation needs balance.** Everybody must have every replicate at every
//! position, which is why a value that was never measured will be imputed in
//! the E-step rather than dropped when censoring arrives. Seventeen dropped
//! observations would cost the factor of a thousand above; `src/bivariate.rs`
//! records the same lesson from the two-trait model, where the eigen-rotation
//! "survives extra traits only while everybody has every trait".
//!
//! # How it is fitted
//!
//! Expectation-maximisation, with the person-level effects as the missing data.
//! The complete-data problem separates into one closed form per component, so
//! there is no search and no gradient inside the loop. That is what will make
//! the censored version affordable: one iteration costs about one likelihood
//! evaluation instead of 250.
//!
//! EM is a route and not a definition. The observed-data likelihood is what the
//! fit reports; it is recomputed at every iteration so that a failure of
//! monotonicity would show rather than pass unnoticed; and the fit record
//! carries a central-difference gradient of the observed-data profile
//! likelihood at the point EM stopped. One number decides whether the route
//! arrived where it claimed.

use nalgebra::{DMatrix, DVector};

use crate::convergence::TOLERANCE;

/// `log(2 pi)`, which appears once per scalar observation.
const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// How far off diagonal a rotated component may be before the rotation is
/// refused, relative to the largest diagonal entry. The eigenvectors are
/// computed in double precision on matrices whose entries are small rationals,
/// so anything a genuine second structured component would produce is orders of
/// magnitude above this.
const DIAGONAL_TOLERANCE: f64 = 1e-8;

/// An eigenvalue at or below this is nought, and its direction carries no
/// information about that component. Relationship matrices are singular
/// whenever two people are genetically identical, so this is a case that
/// happens rather than one guarded against on principle.
const EIGENVALUE_FLOOR: f64 = 1e-10;

/// The most EM iterations before the fit gives up and says so.
const MAX_ITERATIONS: usize = 20_000;

/// How many times an extrapolated step may be backed off towards the plain EM
/// one before it is abandoned. Each halving takes the step length half way to
/// where it does nothing at all, so a few are enough and the point where it
/// does nothing is the ordinary EM step, which is always available.
const BACKTRACKS: usize = 12;

/// EM asks whether it has arrived when the gain it estimates is still to come
/// falls below this, relative to the size of the log-likelihood.
///
/// **The gain still to come, not the gain just had.** EM converges linearly, so
/// successive improvements fall off by a roughly constant ratio and the sum of
/// all the improvements left is the last one times `r / (1 - r)`. Where `r` is
/// near one that is very much larger than the last step, and a rule that looks
/// only at the last step stops far short. Measured here, stopping on a relative
/// improvement of `1e-12` left the gradient at `2.5e-6` against a threshold of
/// `1e-6`, and the variance it reported was wrong in the fifth decimal place
/// against [`crate::ComponentModel`] on the same data.
///
/// This is Aitken's estimate of the limit, which is the standard stopping rule
/// for EM and is doing the same job here that switching off `factr` does for
/// the searches in `src/convergence.rs`: the objective settles well before the
/// gradient does, so an objective rule alone is not enough. It is still not the
/// convergence test. The projected gradient decides that, as it does everywhere
/// else in the crate.
const REMAINING_GAIN: f64 = 1e-13;

/// Once the estimated gain has run out and the gradient test has still not
/// passed, the rule is switched off and the gradient is read again this often.
/// Reading it costs two evaluations per parameter, so it is worth doing rarely
/// and worth doing at all: it stops the search the moment it has arrived rather
/// than at whatever iteration count was guessed in advance.
const GRADIENT_CHECK_EVERY: usize = 500;

/// A direction of a component whose variance is below this is resting on the
/// boundary, and is put exactly on it.
///
/// **This is the same idea as `resting_on_zero` in `src/components.rs` and it
/// is needed for the same reason.** A search leaves a component on its bound a
/// hair above it rather than at it, so the exact test never fires and a fit
/// that is demonstrably the maximum reports itself unconverged. EM is worse
/// than a bounded search here: it approaches a boundary geometrically and never
/// arrives, so the hair is not one part in `1e-37` but one part in `1e-8`, and
/// the derivative that a boundary would make nought stays measurably above it.
///
/// The response is standardised before fitting, so a variance below this is
/// nought in every sense that matters: `src/convergence.rs` calibrates the
/// package at a gradient of `1e-6` meaning a variance share right to about
/// `1e-5`. **And it is a proposal rather than a decision** -- the likelihood is
/// evaluated at the boundary and the move is refused if it falls, which is
/// exactly the case of a direction that was small rather than absent.
const RESTING_VARIANCE: f64 = 1e-6;

/// The step for the central differences that read the gradient at the end. The
/// response is standardised before fitting, so the square roots this steps in
/// are of order one and a relative and an absolute step are the same thing.
const GRADIENT_STEP: f64 = 1e-5;

/// Everything the search carries from one step to the next: a covariance per
/// person-level component, the replicate-level covariance, and the fixed
/// effects.
type State = (Vec<DMatrix<f64>>, DMatrix<f64>, DMatrix<f64>);

/// A fitted repeated-measures model.
#[derive(Clone, Debug)]
pub struct RepeatedFit {
    /// One `T x T` covariance per person-level component, in the order the
    /// matrices were given.
    pub component_covariances: Vec<DMatrix<f64>>,
    /// The replicate-level covariance, which plays the part of the residual.
    pub residual_covariance: DMatrix<f64>,
    /// Fixed effects, `q x T`: one column per position, because the
    /// coefficients are free to differ from one position to the next.
    pub fixed_effects: DMatrix<f64>,
    /// For each position, each component's share of the total variance there,
    /// in the order the matrices were given with the residual last. Whether the
    /// first of these is a heritability depends on what was supplied, exactly
    /// as it does for [`crate::ComponentModel`].
    pub variance_shares: Vec<Vec<f64>>,
    /// The observed-data log-likelihood, in the response's own units. This is
    /// the definition of correct here, not a number EM reports in passing.
    pub loglik: f64,
    pub iterations: usize,
    /// Whether the observed-data log-likelihood rose at every iteration. EM
    /// guarantees it, so a `false` here is a fault in this code rather than a
    /// difficult data set, and it is reported rather than asserted so that a
    /// caller sees it too.
    pub monotone: bool,
    pub converged: bool,
    /// The largest absolute central-difference derivative of the observed-data
    /// profile log-likelihood at the point EM stopped, divided by the size of
    /// the log-likelihood. This is the same reading, on the same scale and
    /// against the same threshold, as every other fit in the crate.
    ///
    /// **It is taken in the symmetric square roots rather than in the
    /// covariances themselves.** Every symmetric matrix squares to something at
    /// least positive semidefinite, so the parameter space is unconstrained and
    /// a maximum has a derivative of nought in every direction, with no bound
    /// to project onto. The price is that a component resting at nought is a
    /// critical point of the parameterisation whatever the likelihood does, so
    /// a small reading there says less than it does elsewhere -- the same
    /// caveat [`crate::ComponentModel`] carries for a variance on its bound.
    pub scaled_gradient: f64,
    pub estimator: &'static str,
}

/// The data rotated into the basis where the likelihood is block diagonal.
struct Rotated {
    /// `P x T`. The replicate mean of each person, rotated by the eigenvectors.
    mean: DMatrix<f64>,
    /// `P(R - 1) x T`. The replicate contrasts, which need no second rotation
    /// because they are independent across people already.
    contrast: DMatrix<f64>,
}

/// The pieces of one mean-channel block that both the E-step and the likelihood
/// want, formed once and used by both.
struct Block {
    inverse: DMatrix<f64>,
    log_determinant: f64,
}

/// Repeated measures at fixed positions, with a replicate level inside the
/// person.
pub struct RepeatedModel {
    /// The rotated diagonal of each person-level component: `diagonals[j][k]`
    /// is what component `j` contributes in eigendirection `k`. The residual is
    /// implicit and is not stored.
    diagonals: Vec<Vec<f64>>,
    /// How many eigendirections carry information about each component, which
    /// is the rank of its matrix and the divisor in that component's update.
    ranks: Vec<usize>,
    /// The rotated design of the mean channel, `P x q`.
    design_mean: DMatrix<f64>,
    /// The rotated design of the contrast channels, `P(R - 1) x q`.
    design_contrast: DMatrix<f64>,
    /// `X' X`, which is the same before and after an orthogonal rotation of the
    /// rows and so is formed once.
    design_cross: DMatrix<f64>,
    /// The contrast channels' own `X' X`, kept separately because they share a
    /// single weight matrix and so contribute one Kronecker term to the
    /// generalised least squares rather than one per row.
    contrast_cross: DMatrix<f64>,
    /// The eigenvectors of the first component's matrix, kept because the
    /// response has to be rotated by them once per fit.
    eigenvectors: DMatrix<f64>,
    /// The Helmert matrix that separates a person's replicates into their mean
    /// and their contrasts.
    helmert: DMatrix<f64>,
    people: usize,
    replicates: usize,
    positions: usize,
    covariates: usize,
}

impl RepeatedModel {
    /// Validate and prepare. `matrices` are the person-level components, each
    /// `P x P`; the replicate-level residual is added for you. `design` is
    /// `(P * R) x q` with the rows in person-major order, so person `p`
    /// replicate `r` is row `p * R + r`.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model this can
    /// fit. `REPEATED_NOT_SIMULTANEOUSLY_DIAGONAL` is the one worth reading
    /// twice: it means a second structured component was supplied, which is a
    /// model this rotation cannot express rather than a mistake in the data.
    /// ADR 0010 expects that case to arrive with a household matrix, and it
    /// will need a dense per-family route and the cost that goes with it.
    pub fn build(
        matrices: &[DMatrix<f64>],
        design: &DMatrix<f64>,
        replicates: usize,
        positions: usize,
    ) -> Result<Self, &'static str> {
        if matrices.is_empty() {
            return Err("REPEATED_NO_COMPONENTS");
        }
        if replicates < 2 {
            return Err("REPEATED_NEEDS_TWO_REPLICATES");
        }
        if positions == 0 {
            return Err("REPEATED_NO_POSITIONS");
        }
        let people = matrices[0].nrows();
        if people == 0 {
            return Err("REPEATED_NO_PEOPLE");
        }
        for matrix in matrices {
            if matrix.nrows() != people || matrix.ncols() != people {
                return Err("REPEATED_MATRIX_WRONG_SIZE");
            }
            for i in 0..people {
                for j in 0..i {
                    if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-10 {
                        return Err("REPEATED_MATRIX_NOT_SYMMETRIC");
                    }
                }
            }
        }
        if design.nrows() != people * replicates {
            return Err("REPEATED_DESIGN_WRONG_ROWS");
        }
        let covariates = design.ncols();
        if covariates == 0 {
            return Err("REPEATED_DESIGN_NO_COLUMNS");
        }

        // The first component sets the basis and the rest have to live in it.
        let eigenvectors = matrices[0].clone().symmetric_eigen().eigenvectors;
        let transposed = eigenvectors.transpose();
        let mut diagonals = Vec::with_capacity(matrices.len());
        let mut ranks = Vec::with_capacity(matrices.len());
        for (index, matrix) in matrices.iter().enumerate() {
            let rotated = &transposed * matrix * &eigenvectors;
            let scale = (0..people)
                .map(|k| rotated[(k, k)].abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            if index > 0 {
                for i in 0..people {
                    for j in 0..people {
                        if i != j && rotated[(i, j)].abs() > DIAGONAL_TOLERANCE * scale {
                            return Err("REPEATED_NOT_SIMULTANEOUSLY_DIAGONAL");
                        }
                    }
                }
            }
            // A negative eigenvalue is a matrix that is not a covariance. A
            // rounding-sized one is nought and is treated as such; anything
            // larger is refused rather than clamped, because clamping would fit
            // a model the caller did not ask for and say nothing about it.
            let mut diagonal = Vec::with_capacity(people);
            let mut rank = 0;
            for k in 0..people {
                let value = rotated[(k, k)];
                if value < -DIAGONAL_TOLERANCE * scale {
                    return Err("REPEATED_MATRIX_NOT_POSITIVE_SEMIDEFINITE");
                }
                let value = if value > EIGENVALUE_FLOOR { value } else { 0.0 };
                if value > 0.0 {
                    rank += 1;
                }
                diagonal.push(value);
            }
            if rank == 0 {
                return Err("REPEATED_MATRIX_IS_ZERO");
            }
            diagonals.push(diagonal);
            ranks.push(rank);
        }

        let helmert = helmert_matrix(replicates);
        let (design_mean_raw, design_contrast) = rotate_replicates(design, &helmert, people);
        let design_mean = &transposed * design_mean_raw;

        // The fixed effects are solved for at every iteration and at every
        // point of the gradient, so a design that cannot support them should
        // fail here rather than there.
        let design_cross = design.transpose() * design;
        if design_cross.clone().cholesky().is_none() {
            return Err("REPEATED_DESIGN_RANK_DEFICIENT");
        }
        let contrast_cross = design_contrast.transpose() * &design_contrast;

        Ok(Self {
            diagonals,
            ranks,
            design_mean,
            design_contrast,
            design_cross,
            contrast_cross,
            eigenvectors,
            helmert,
            people,
            replicates,
            positions,
            covariates,
        })
    }

    /// How many person-level components there are, not counting the residual.
    #[must_use]
    pub fn components(&self) -> usize {
        self.diagonals.len()
    }

    /// Fit by expectation-maximisation.
    ///
    /// `y` is `(P * R) x T`, in the same row order as the design: person `p`
    /// replicate `r` is row `p * R + r`, and column `t` is position `t`.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the response does not match the model, is
    /// constant, or produces a covariance that cannot be factorised.
    pub fn fit(&self, y: &DMatrix<f64>) -> Result<RepeatedFit, &'static str> {
        if y.nrows() != self.people * self.replicates || y.ncols() != self.positions {
            return Err("REPEATED_RESPONSE_WRONG_SHAPE");
        }

        // Standardised for the same reason the one-trait model standardises:
        // it puts the gradient reading on a scale that means the same thing
        // from one data set to the next, so one threshold can serve them all.
        let count = (y.nrows() * y.ncols()) as f64;
        let mean = y.iter().sum::<f64>() / count;
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
        if !(variance > 0.0) {
            return Err("REPEATED_RESPONSE_CONSTANT");
        }
        let scale = variance.sqrt();
        let rotated = self.rotate(&(y / scale));

        let (mut sigmas, mut residual, mut fixed) = self.starting_values(&rotated)?;
        let mut loglik = self
            .loglik_at(&rotated, &sigmas, &residual, &fixed)
            .ok_or("REPEATED_START_NOT_EVALUABLE")?;
        let mut monotone = true;
        let mut iterations = 0;
        let mut previous_gain = 0.0;
        let mut relaxed = false;
        let mut scaled_gradient = f64::INFINITY;
        let mut last_reading: Option<usize> = None;

        while iterations < MAX_ITERATIONS {
            let Some(next) =
                self.accelerated_step(&rotated, &sigmas, &residual, &fixed, &mut iterations)
            else {
                return Err("REPEATED_ITERATION_NOT_EVALUABLE");
            };
            let ((next_sigmas, next_residual, next_fixed), next_loglik) = next;
            // A fall of a few units in the last place is the arithmetic and not
            // the algorithm; anything larger is this code being wrong.
            if next_loglik < loglik - 1e-8 * loglik.abs().max(1.0) {
                monotone = false;
            }
            let gain = next_loglik - loglik;
            sigmas = next_sigmas;
            residual = next_residual;
            fixed = next_fixed;
            loglik = next_loglik;
            if gain <= 0.0 {
                break;
            }

            let periodic = iterations
                >= last_reading
                    .unwrap_or(0)
                    .saturating_add(GRADIENT_CHECK_EVERY);
            let settling = !relaxed && {
                let rate = if previous_gain > 0.0 {
                    gain / previous_gain
                } else {
                    1.0
                };
                let remaining = if rate > 0.0 && rate < 1.0 {
                    gain * rate / (1.0 - rate)
                } else {
                    f64::INFINITY
                };
                remaining <= REMAINING_GAIN * loglik.abs().max(1.0)
            };
            if periodic || settling {
                if let Some(((rested_sigmas, rested_residual, rested_fixed), rested_loglik)) =
                    self.rested(&rotated, &sigmas, &residual, &fixed, loglik)
                {
                    sigmas = rested_sigmas;
                    residual = rested_residual;
                    fixed = rested_fixed;
                    loglik = rested_loglik;
                }
                scaled_gradient = self
                    .gradient_reading(&rotated, &sigmas, &residual)
                    .unwrap_or(f64::INFINITY);
                last_reading = Some(iterations);
                if scaled_gradient < TOLERANCE {
                    break;
                }
                relaxed = true;
            }
            previous_gain = gain;
        }

        if last_reading != Some(iterations) {
            if let Some(((rested_sigmas, rested_residual, rested_fixed), rested_loglik)) =
                self.rested(&rotated, &sigmas, &residual, &fixed, loglik)
            {
                sigmas = rested_sigmas;
                residual = rested_residual;
                fixed = rested_fixed;
                loglik = rested_loglik;
            }
            scaled_gradient = self
                .gradient_reading(&rotated, &sigmas, &residual)
                .unwrap_or(f64::INFINITY);
        }

        // Back to the response's own units. A covariance carries the square of
        // the scale, a fixed effect carries the scale itself, and the
        // log-likelihood carries a term per scalar observation.
        let factor = variance;
        let component_covariances: Vec<DMatrix<f64>> = sigmas.iter().map(|s| s * factor).collect();
        let residual_covariance = &residual * factor;
        let variance_shares = shares(&component_covariances, &residual_covariance);
        Ok(RepeatedFit {
            component_covariances,
            residual_covariance,
            fixed_effects: fixed * scale,
            variance_shares,
            loglik: loglik - count * scale.ln(),
            iterations,
            monotone,
            converged: scaled_gradient < TOLERANCE,
            scaled_gradient,
            estimator: "ml",
        })
    }

    /// Rotate a response into the basis the likelihood is diagonal in.
    fn rotate(&self, y: &DMatrix<f64>) -> Rotated {
        let (mean_raw, contrast) = rotate_replicates(y, &self.helmert, self.people);
        Rotated {
            mean: self.eigenvectors.transpose() * mean_raw,
            contrast,
        }
    }

    /// The covariance of the mean channel in eigendirection `k`.
    ///
    /// Both replicates carry the person's effect, so the mean of `R` of them
    /// carries `R` times its variance while the residual is averaged down to
    /// one. That asymmetry is the whole reason the rotation separates the two.
    fn mean_covariance(
        &self,
        k: usize,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let mut v = residual.clone();
        let replicates = self.replicates as f64;
        for (component, sigma) in sigmas.iter().enumerate() {
            let weight = replicates * self.diagonals[component][k];
            if weight > 0.0 {
                v += sigma * weight;
            }
        }
        v
    }

    /// Factor every mean-channel block once.
    fn blocks(&self, sigmas: &[DMatrix<f64>], residual: &DMatrix<f64>) -> Option<Vec<Block>> {
        (0..self.people)
            .map(|k| {
                let v = self.mean_covariance(k, sigmas, residual);
                let chol = v.cholesky()?;
                let log_determinant = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
                Some(Block {
                    inverse: chol.inverse(),
                    log_determinant,
                })
            })
            .collect()
    }

    /// Ordinary least squares, ignoring every component. Splitting the residual
    /// covariance evenly among the components is a start and nothing more: EM
    /// from an even split reaches the same maximum as EM from anywhere else
    /// that can be factorised, only sooner or later. The real model will start
    /// from univariate fits, which are the paper's first table anyway.
    fn starting_values(&self, rotated: &Rotated) -> Result<State, &'static str> {
        let cross = self.design_mean.transpose() * &rotated.mean
            + self.design_contrast.transpose() * &rotated.contrast;
        let chol = self
            .design_cross
            .clone()
            .cholesky()
            .ok_or("REPEATED_DESIGN_RANK_DEFICIENT")?;
        let fixed = chol.solve(&cross);
        let mean_residual = &rotated.mean - &self.design_mean * &fixed;
        let contrast_residual = &rotated.contrast - &self.design_contrast * &fixed;
        let rows = (self.people * self.replicates) as f64;
        let total = (mean_residual.transpose() * &mean_residual
            + contrast_residual.transpose() * &contrast_residual)
            / rows;
        let parts = (self.components() + 1) as f64;
        let share = symmetrised(&(total / parts));
        if share.clone().cholesky().is_none() {
            return Err("REPEATED_RESPONSE_SINGULAR");
        }
        Ok((vec![share.clone(); self.components()], share, fixed))
    }

    /// One E-step and the M-step that follows it.
    ///
    /// The complete-data log-likelihood separates: each component's covariance
    /// depends only on its own effects, and the fixed effects and the residual
    /// covariance are maximised together, the first without reference to the
    /// second. So there is no search here, only four closed forms.
    fn one_iteration(
        &self,
        rotated: &Rotated,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
    ) -> Option<State> {
        let blocks = self.blocks(sigmas, residual)?;
        let positions = self.positions;
        let replicates = self.replicates as f64;
        let root = replicates.sqrt();
        let mean_residual = &rotated.mean - &self.design_mean * fixed;

        let mut statistics = vec![DMatrix::<f64>::zeros(positions, positions); self.components()];
        let mut residual_variance = DMatrix::<f64>::zeros(positions, positions);
        let mut effects = DMatrix::<f64>::zeros(self.people, positions);

        for (k, block) in blocks.iter().enumerate() {
            let r: DVector<f64> = mean_residual.row(k).transpose();
            let weighted = &block.inverse * &r;
            let mut total = DVector::<f64>::zeros(positions);
            for (component, sigma) in sigmas.iter().enumerate() {
                let d = self.diagonals[component][k];
                if d <= 0.0 {
                    // The effect is exactly nought in this direction, so it
                    // contributes nothing to the estimate and nothing to the
                    // statistic. Skipping it is what keeps a singular
                    // relationship matrix from dividing by its own zero.
                    continue;
                }
                let projected = sigma * &weighted;
                total += &projected * (root * d);
                // The conditional second moment of the effect, divided through
                // by the eigenvalue that would otherwise appear in the inverse
                // of the component's own matrix. Written this way there is no
                // division at all, so a direction whose eigenvalue is small
                // costs accuracy nowhere.
                statistics[component] += &projected * projected.transpose() * (replicates * d)
                    + sigma
                    - sigma * &block.inverse * sigma * (replicates * d);
            }
            for t in 0..positions {
                effects[(k, t)] = total[t];
            }
            residual_variance += residual - residual * &block.inverse * residual;
        }

        // The fixed effects, given the effects just estimated. This is ordinary
        // least squares and not generalised, because the design is shared
        // across positions and the coefficients are free at every one of them,
        // so the residual covariance cancels out of the normal equations. At a
        // fixed point it therefore agrees with generalised least squares on the
        // observed data, which is a thing worth checking and is checked.
        let adjusted_mean = &rotated.mean - &effects * root;
        let cross = self.design_mean.transpose() * &adjusted_mean
            + self.design_contrast.transpose() * &rotated.contrast;
        let next_fixed = self.design_cross.clone().cholesky()?.solve(&cross);

        let mean_error = &adjusted_mean - &self.design_mean * &next_fixed;
        let contrast_error = &rotated.contrast - &self.design_contrast * &next_fixed;
        let next_residual = symmetrised(
            &((mean_error.transpose() * &mean_error
                + contrast_error.transpose() * &contrast_error
                + residual_variance)
                / (self.people * self.replicates) as f64),
        );

        let next_sigmas: Vec<DMatrix<f64>> = statistics
            .iter()
            .zip(&self.ranks)
            .map(|(statistic, rank)| symmetrised(&(statistic / *rank as f64)))
            .collect();

        Some((next_sigmas, next_residual, next_fixed))
    }

    /// Two EM steps, and an extrapolation along the line they lie on.
    ///
    /// # Why this is here and not left for later
    ///
    /// Plain EM converges linearly, and where the maximum lies on the boundary
    /// -- a component whose covariance has gone to nought, or lost a direction
    /// -- it converges slower than linearly and never arrives. Measured on
    /// twenty people with no signal in them at all, plain EM was still at 1.5e-4
    /// after twenty thousand iterations for a variance whose true maximum is
    /// exactly nought, and reported itself unconverged, correctly. A component
    /// going to nought is not an exotic case: it is what a fit says when the
    /// data do not support a component, and this model has three of them and
    /// will soon have a covariance apiece at seventeen positions.
    ///
    /// The remedy is the standard one for EM. Two steps from the same point lie
    /// on a line, and where the sequence is geometric the whole remaining path
    /// down that line can be taken at once. The step length is
    /// `-||r|| / ||v||` where `r` is the first step and `v` the change in it,
    /// which is the `S3` scheme of Varadhan and Roland's SQUAREM.
    ///
    /// **The extrapolation proposes and EM disposes.** A proposal is followed by
    /// an ordinary EM step, so what comes back is always an EM iterate; it is
    /// kept only where the observed-data likelihood is at least what two plain
    /// steps would have given; and it is projected back onto the covariances
    /// before it is tried, so an overshoot past a boundary lands on the boundary
    /// rather than outside the model. Backing the step off halves it towards
    /// the length at which it does nothing, which is the plain EM step, so there
    /// is always somewhere safe to fall back to and monotonicity is preserved
    /// whatever the extrapolation does.
    fn accelerated_step(
        &self,
        rotated: &Rotated,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
        iterations: &mut usize,
    ) -> Option<(State, f64)> {
        let (one_sigmas, one_residual, one_fixed) =
            self.one_iteration(rotated, sigmas, residual, fixed)?;
        *iterations += 1;
        let (two_sigmas, two_residual, two_fixed) =
            self.one_iteration(rotated, &one_sigmas, &one_residual, &one_fixed)?;
        *iterations += 1;
        let two_loglik = self.loglik_at(rotated, &two_sigmas, &two_residual, &two_fixed)?;

        let base = flatten(sigmas, residual, fixed);
        let one = flatten(&one_sigmas, &one_residual, &one_fixed);
        let two = flatten(&two_sigmas, &two_residual, &two_fixed);
        let mut step = Vec::with_capacity(base.len());
        let mut bend = Vec::with_capacity(base.len());
        for index in 0..base.len() {
            let moved = one[index] - base[index];
            bend.push(two[index] - one[index] - moved);
            step.push(moved);
        }
        let step_norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        let bend_norm = bend.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Written this way round so that a bend of nought, which gives an
        // infinite length, and a NaN both fall through to the plain steps.
        let mut length = -step_norm / bend_norm;
        if !(length < -1.0) || !length.is_finite() {
            return Some(((two_sigmas, two_residual, two_fixed), two_loglik));
        }

        let mut candidate = Vec::with_capacity(base.len());
        for _ in 0..BACKTRACKS {
            candidate.clear();
            for index in 0..base.len() {
                candidate
                    .push(base[index] - 2.0 * length * step[index] + length * length * bend[index]);
            }
            let (try_sigmas, try_residual, try_fixed) = unflatten(
                &candidate,
                self.components(),
                self.positions,
                self.covariates,
            );
            let attempt = self.one_iteration(rotated, &try_sigmas, &try_residual, &try_fixed);
            *iterations += 1;
            if let Some((next_sigmas, next_residual, next_fixed)) = attempt
                && let Some(next_loglik) =
                    self.loglik_at(rotated, &next_sigmas, &next_residual, &next_fixed)
                && next_loglik >= two_loglik
            {
                return Some(((next_sigmas, next_residual, next_fixed), next_loglik));
            }
            length = (length - 1.0) / 2.0;
            if length >= -1.0 - 1e-6 {
                break;
            }
        }
        Some(((two_sigmas, two_residual, two_fixed), two_loglik))
    }

    /// Put every collapsed direction exactly on the boundary, if the likelihood
    /// does not object.
    ///
    /// Returns `None` where there was nothing to move or where moving it made
    /// the fit worse, which is the case of a direction that was small rather
    /// than absent. A covariance's null space is preserved by the M-step -- it
    /// appears as a factor on both sides of every term in it -- so nought is an
    /// exact fixed point and this never has to be done twice.
    fn rested(
        &self,
        rotated: &Rotated,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
        loglik: f64,
    ) -> Option<(State, f64)> {
        let mut moved = false;
        let mut rested = Vec::with_capacity(sigmas.len());
        for sigma in sigmas {
            match rest_on_zero(sigma) {
                Some(matrix) => {
                    moved = true;
                    rested.push(matrix);
                }
                None => rested.push(sigma.clone()),
            }
        }
        let rested_residual = match rest_on_zero(residual) {
            Some(matrix) => {
                moved = true;
                matrix
            }
            None => residual.clone(),
        };
        if !moved {
            return None;
        }
        let value = self.loglik_at(rotated, &rested, &rested_residual, fixed)?;
        if value < loglik {
            return None;
        }
        // One ordinary step from where it landed, so the fixed effects are
        // those of the point being reported rather than of the point before it.
        let Some((next_sigmas, next_residual, next_fixed)) =
            self.one_iteration(rotated, &rested, &rested_residual, fixed)
        else {
            return Some(((rested, rested_residual, fixed.clone()), value));
        };
        match self.loglik_at(rotated, &next_sigmas, &next_residual, &next_fixed) {
            Some(next) if next >= value => Some(((next_sigmas, next_residual, next_fixed), next)),
            _ => Some(((rested, rested_residual, fixed.clone()), value)),
        }
    }

    /// The observed-data log-likelihood at these parameters and these fixed
    /// effects, on the standardised scale.
    fn loglik_at(
        &self,
        rotated: &Rotated,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
    ) -> Option<f64> {
        let blocks = self.blocks(sigmas, residual)?;
        let residual_chol = residual.clone().cholesky()?;
        let residual_logdet = 2.0
            * residual_chol
                .l()
                .diagonal()
                .iter()
                .map(|d| d.ln())
                .sum::<f64>();
        let residual_inverse = residual_chol.inverse();

        let mean_residual = &rotated.mean - &self.design_mean * fixed;
        let contrast_residual = &rotated.contrast - &self.design_contrast * fixed;

        let scalars = (self.people * self.replicates * self.positions) as f64;
        let mut value = -0.5 * scalars * LN_2PI;
        for (k, block) in blocks.iter().enumerate() {
            let r: DVector<f64> = mean_residual.row(k).transpose();
            let quadratic = (r.transpose() * &block.inverse * &r)[(0, 0)];
            value -= 0.5 * (block.log_determinant + quadratic);
        }
        for row in 0..contrast_residual.nrows() {
            let e: DVector<f64> = contrast_residual.row(row).transpose();
            let quadratic = (e.transpose() * &residual_inverse * &e)[(0, 0)];
            value -= 0.5 * (residual_logdet + quadratic);
        }
        value.is_finite().then_some(value)
    }

    /// The observed-data log-likelihood with the fixed effects at their own
    /// maximum, which is generalised least squares rather than ordinary because
    /// the mean channel weights every eigendirection differently.
    ///
    /// This is what the gradient is taken of, so that the reading is of the
    /// profile the crate reports everywhere else and not of a slice through a
    /// fixed effect that happens to be optimal at one point only.
    fn profile_loglik(
        &self,
        rotated: &Rotated,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
    ) -> Option<f64> {
        let fixed = self.generalised_least_squares(rotated, sigmas, residual)?;
        self.loglik_at(rotated, sigmas, residual, &fixed)
    }

    /// The fixed effects that maximise the observed-data likelihood at these
    /// variances.
    ///
    /// The unknown is a `q x T` matrix, so the normal equations are `qT` square
    /// rather than `q` square: with a different weight matrix per row they do
    /// not separate by position. On the audiogram design that is 68 by 68,
    /// which is nothing beside the blocks that built it.
    fn generalised_least_squares(
        &self,
        rotated: &Rotated,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
    ) -> Option<DMatrix<f64>> {
        let blocks = self.blocks(sigmas, residual)?;
        let residual_inverse = residual.clone().cholesky()?.inverse();
        let positions = self.positions;
        let covariates = self.covariates;
        let size = positions * covariates;

        let mut normal = DMatrix::<f64>::zeros(size, size);
        let mut right = DMatrix::<f64>::zeros(positions, covariates);

        // The contrast rows share one weight matrix, so they contribute a
        // single Kronecker term instead of one per row.
        for a in 0..covariates {
            for b in 0..covariates {
                let weight = self.contrast_cross[(a, b)];
                if weight == 0.0 {
                    continue;
                }
                for s in 0..positions {
                    for u in 0..positions {
                        normal[(a * positions + s, b * positions + u)] +=
                            weight * residual_inverse[(s, u)];
                    }
                }
            }
        }
        right += &residual_inverse * rotated.contrast.transpose() * &self.design_contrast;

        for (k, block) in blocks.iter().enumerate() {
            let x = self.design_mean.row(k);
            let y: DVector<f64> = rotated.mean.row(k).transpose();
            let weighted = &block.inverse * &y;
            for a in 0..covariates {
                if x[a] == 0.0 {
                    continue;
                }
                for b in 0..covariates {
                    let weight = x[a] * x[b];
                    if weight == 0.0 {
                        continue;
                    }
                    for s in 0..positions {
                        for u in 0..positions {
                            normal[(a * positions + s, b * positions + u)] +=
                                weight * block.inverse[(s, u)];
                        }
                    }
                }
                for s in 0..positions {
                    right[(s, a)] += x[a] * weighted[s];
                }
            }
        }

        let stacked = DVector::from_iterator(
            size,
            (0..covariates)
                .flat_map(|a| (0..positions).map(move |s| (s, a)))
                .map(|(s, a)| right[(s, a)]),
        );
        let solution = normal.cholesky()?.solve(&stacked);
        let mut fixed = DMatrix::<f64>::zeros(covariates, positions);
        for a in 0..covariates {
            for s in 0..positions {
                fixed[(a, s)] = solution[a * positions + s];
            }
        }
        Some(fixed)
    }

    /// The scaled projected gradient at the point EM stopped, by central
    /// differences in the symmetric square roots of the covariances.
    fn gradient_reading(
        &self,
        rotated: &Rotated,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
    ) -> Option<f64> {
        let mut roots: Vec<DMatrix<f64>> = sigmas.iter().map(symmetric_square_root).collect();
        roots.push(symmetric_square_root(residual));
        let at = self.profile_loglik(rotated, sigmas, residual)?;

        let value_at = |roots: &[DMatrix<f64>]| -> Option<f64> {
            let (last, rest) = roots.split_last()?;
            let sigmas: Vec<DMatrix<f64>> = rest.iter().map(|l| symmetrised(&(l * l))).collect();
            let residual = symmetrised(&(last * last));
            self.profile_loglik(rotated, &sigmas, &residual)
        };

        let mut worst = 0.0_f64;
        for which in 0..roots.len() {
            for a in 0..self.positions {
                for b in 0..=a {
                    let mut up = roots.clone();
                    let mut down = roots.clone();
                    up[which][(a, b)] += GRADIENT_STEP;
                    down[which][(a, b)] -= GRADIENT_STEP;
                    if a != b {
                        up[which][(b, a)] += GRADIENT_STEP;
                        down[which][(b, a)] -= GRADIENT_STEP;
                    }
                    let high = value_at(&up)?;
                    let low = value_at(&down)?;
                    worst = worst.max(((high - low) / (2.0 * GRADIENT_STEP)).abs());
                }
            }
        }
        Some(worst / at.abs().max(1.0))
    }
}

/// Each component's share of the total variance at each position.
fn shares(components: &[DMatrix<f64>], residual: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..residual.nrows())
        .map(|t| {
            let mut parts: Vec<f64> = components.iter().map(|c| c[(t, t)]).collect();
            parts.push(residual[(t, t)]);
            let total: f64 = parts.iter().sum();
            if total > 0.0 {
                parts.iter().map(|p| p / total).collect()
            } else {
                vec![f64::NAN; parts.len()]
            }
        })
        .collect()
}

/// Lay the whole parameter set out end to end, so that two of them can be
/// subtracted and a line drawn through them.
fn flatten(sigmas: &[DMatrix<f64>], residual: &DMatrix<f64>, fixed: &DMatrix<f64>) -> Vec<f64> {
    let mut values = Vec::new();
    for sigma in sigmas {
        values.extend(sigma.iter().copied());
    }
    values.extend(residual.iter().copied());
    values.extend(fixed.iter().copied());
    values
}

/// Read a parameter set back, **projected onto the covariances**.
///
/// An extrapolated point is a guess and can be one that is not a model: a
/// covariance with a negative direction in it. Zeroing that direction is the
/// nearest covariance to the guess in the Frobenius sense, and it is the right
/// repair rather than a convenient one, because the case that produces it is a
/// component collapsing to nought and the boundary is where it was going.
fn unflatten(
    values: &[f64],
    components: usize,
    positions: usize,
    covariates: usize,
) -> (Vec<DMatrix<f64>>, DMatrix<f64>, DMatrix<f64>) {
    let block = positions * positions;
    let mut at = 0;
    let mut sigmas = Vec::with_capacity(components);
    for _ in 0..components {
        sigmas.push(project_positive(&DMatrix::from_column_slice(
            positions,
            positions,
            &values[at..at + block],
        )));
        at += block;
    }
    let residual = project_positive(&DMatrix::from_column_slice(
        positions,
        positions,
        &values[at..at + block],
    ));
    at += block;
    let fixed = DMatrix::from_column_slice(
        covariates,
        positions,
        &values[at..at + covariates * positions],
    );
    (sigmas, residual, fixed)
}

/// The same covariance with every collapsed direction set exactly to nought,
/// or `None` where there was none to set.
fn rest_on_zero(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let eigen = symmetrised(matrix).symmetric_eigen();
    if eigen.eigenvalues.iter().all(|v| *v >= RESTING_VARIANCE) {
        return None;
    }
    let clamped = DMatrix::from_diagonal(
        &eigen
            .eigenvalues
            .map(|v| if v < RESTING_VARIANCE { 0.0 } else { v }),
    );
    Some(symmetrised(
        &(&eigen.eigenvectors * clamped * eigen.eigenvectors.transpose()),
    ))
}

/// The nearest positive semidefinite matrix, which is the same matrix with any
/// negative eigenvalue set to nought.
fn project_positive(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let eigen = symmetrised(matrix).symmetric_eigen();
    if eigen.eigenvalues.iter().all(|v| *v >= 0.0) {
        return symmetrised(matrix);
    }
    let clamped = DMatrix::from_diagonal(&eigen.eigenvalues.map(|v| v.max(0.0)));
    &eigen.eigenvectors * clamped * eigen.eigenvectors.transpose()
}

/// The average of a matrix and its transpose. Every update here is symmetric in
/// exact arithmetic and drifts by a few units in the last place in this one, so
/// this is hygiene rather than correction; without it the drift accumulates
/// over thousands of iterations and eventually a Cholesky refuses.
fn symmetrised(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    (matrix + matrix.transpose()) * 0.5
}

/// The symmetric matrix `L` with `L L' = M`, for symmetric positive
/// semidefinite `M`.
///
/// A Cholesky factor would be the usual choice and cannot be used: it does not
/// exist for a component that has gone to nought, which is exactly the case the
/// gradient has to be readable at.
fn symmetric_square_root(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let eigen = matrix.clone().symmetric_eigen();
    let roots = DMatrix::from_diagonal(&eigen.eigenvalues.map(|v| v.max(0.0).sqrt()));
    &eigen.eigenvectors * roots * eigen.eigenvectors.transpose()
}

/// The `R x R` orthogonal matrix whose first row is the replicate mean scaled
/// to unit length, and whose remaining rows are contrasts orthogonal to it.
///
/// Any orthogonal matrix with that first row would serve; this is the Helmert
/// one because it is the standard choice and because its contrasts are
/// readable -- row `m` compares replicate `m` with the mean of those before it.
fn helmert_matrix(replicates: usize) -> DMatrix<f64> {
    let mut helmert = DMatrix::zeros(replicates, replicates);
    let root = (replicates as f64).sqrt();
    for column in 0..replicates {
        helmert[(0, column)] = 1.0 / root;
    }
    for row in 1..replicates {
        let n = (row + 1) as f64;
        let scale = (n * (n - 1.0)).sqrt();
        for column in 0..row {
            helmert[(row, column)] = 1.0 / scale;
        }
        helmert[(row, row)] = -(n - 1.0) / scale;
    }
    helmert
}

/// Split a person-major matrix into its replicate mean and its replicate
/// contrasts. This works on a response and on a design alike, because the
/// rotation is on the row index and neither of them knows what the columns are.
fn rotate_replicates(
    values: &DMatrix<f64>,
    helmert: &DMatrix<f64>,
    people: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let replicates = helmert.nrows();
    let columns = values.ncols();
    let mut mean = DMatrix::zeros(people, columns);
    let mut contrast = DMatrix::zeros(people * (replicates - 1), columns);
    for person in 0..people {
        for column in 0..columns {
            for row in 0..replicates {
                let mut total = 0.0;
                for replicate in 0..replicates {
                    total += helmert[(row, replicate)]
                        * values[(person * replicates + replicate, column)];
                }
                if row == 0 {
                    mean[(person, column)] = total;
                } else {
                    contrast[(person * (replicates - 1) + row - 1, column)] = total;
                }
            }
        }
    }
    (mean, contrast)
}

#[cfg(test)]
mod tests {
    use super::{LN_2PI, RepeatedModel, helmert_matrix, rotate_replicates, symmetric_square_root};
    use nalgebra::{DMatrix, DVector};

    /// A small pedigree: `families` unrelated sets of two parents and two
    /// children, which gives a relationship matrix with real off-diagonal
    /// structure and a rank equal to its size.
    fn relationship(families: usize) -> DMatrix<f64> {
        let people = families * 4;
        let mut matrix = DMatrix::<f64>::identity(people, people);
        for family in 0..families {
            let base = family * 4;
            // Parents are unrelated to each other; each child is a half to
            // either parent and a half to its sibling.
            for child in 2..4 {
                for parent in 0..2 {
                    matrix[(base + child, base + parent)] = 0.5;
                    matrix[(base + parent, base + child)] = 0.5;
                }
            }
            matrix[(base + 2, base + 3)] = 0.5;
            matrix[(base + 3, base + 2)] = 0.5;
        }
        matrix
    }

    /// A deterministic spread of numbers, so that a test never depends on a
    /// generator's stream. Nothing here needs the values to be normal; the
    /// checks are of arithmetic identities and of the direction the likelihood
    /// moves in.
    fn spread(count: usize, offset: usize) -> Vec<f64> {
        (0..count)
            .map(|i| {
                let x = ((i + offset) as f64 * 0.618_033_988_749_895).fract();
                (x * 6.0 - 3.0) + ((i % 7) as f64 - 3.0) * 0.25
            })
            .collect()
    }

    fn design(rows: usize) -> DMatrix<f64> {
        let mut x = DMatrix::<f64>::zeros(rows, 2);
        let values = spread(rows, 11);
        for row in 0..rows {
            x[(row, 0)] = 1.0;
            x[(row, 1)] = values[row];
        }
        x
    }

    fn response(rows: usize, positions: usize) -> DMatrix<f64> {
        let values = spread(rows * positions, 3);
        DMatrix::from_iterator(positions, rows, values).transpose()
    }

    /// The likelihood written out in full, with no rotation anywhere: the
    /// covariance of every scalar observation against every other, one dense
    /// Cholesky. Slow, obvious, and the thing the rotation has to agree with.
    fn dense_loglik(
        matrices: &[DMatrix<f64>],
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        x: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
        y: &DMatrix<f64>,
    ) -> f64 {
        // Taken from the shapes rather than passed in, so that a caller cannot
        // describe the data one way here and another way to the model.
        let rows = y.nrows();
        let positions = residual.nrows();
        let replicates = rows / matrices[0].nrows();
        let size = rows * positions;
        let mut v = DMatrix::<f64>::zeros(size, size);
        for row_one in 0..rows {
            for row_two in 0..rows {
                let person_one = row_one / replicates;
                let person_two = row_two / replicates;
                for t in 0..positions {
                    for u in 0..positions {
                        let mut value = 0.0;
                        for (component, sigma) in sigmas.iter().enumerate() {
                            value += matrices[component][(person_one, person_two)] * sigma[(t, u)];
                        }
                        if row_one == row_two {
                            value += residual[(t, u)];
                        }
                        v[(row_one * positions + t, row_two * positions + u)] = value;
                    }
                }
            }
        }
        let mean = x * fixed;
        let mut error = DVector::<f64>::zeros(size);
        for row in 0..rows {
            for t in 0..positions {
                error[row * positions + t] = y[(row, t)] - mean[(row, t)];
            }
        }
        let chol = v.cholesky().expect("the dense covariance should factorise");
        let logdet = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        let quadratic = (error.transpose() * chol.inverse() * &error)[(0, 0)];
        -0.5 * (size as f64 * LN_2PI + logdet + quadratic)
    }

    #[test]
    fn the_helmert_matrix_is_orthogonal_and_averages_first() {
        for replicates in 2..6 {
            let h = helmert_matrix(replicates);
            let product = h.transpose() * &h;
            for i in 0..replicates {
                for j in 0..replicates {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (product[(i, j)] - expected).abs() < 1e-12,
                        "{replicates} replicates, entry ({i}, {j})"
                    );
                }
            }
            for column in 0..replicates {
                assert!((h[(0, column)] - 1.0 / (replicates as f64).sqrt()).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn the_rotation_moves_no_sum_of_squares() {
        let people = 5;
        let replicates = 3;
        let values = response(people * replicates, 4);
        let (mean, contrast) = rotate_replicates(&values, &helmert_matrix(replicates), people);
        let before: f64 = values.iter().map(|v| v * v).sum();
        let after: f64 =
            mean.iter().map(|v| v * v).sum::<f64>() + contrast.iter().map(|v| v * v).sum::<f64>();
        assert!((before - after).abs() < 1e-10, "{before} against {after}");
    }

    #[test]
    fn a_symmetric_square_root_squares_back() {
        let mut m = DMatrix::<f64>::zeros(3, 3);
        let values = [2.0, 0.4, -0.3, 0.4, 1.5, 0.2, -0.3, 0.2, 0.9];
        for i in 0..3 {
            for j in 0..3 {
                m[(i, j)] = values[i * 3 + j];
            }
        }
        let root = symmetric_square_root(&m);
        let back = &root * &root;
        for i in 0..3 {
            for j in 0..3 {
                assert!((back[(i, j)] - m[(i, j)]).abs() < 1e-12);
            }
        }
        // And it exists where a Cholesky factor does not.
        let singular = DMatrix::<f64>::zeros(3, 3);
        assert!(
            symmetric_square_root(&singular)
                .iter()
                .all(|v| v.abs() < 1e-15)
        );
    }

    #[test]
    fn the_rotated_likelihood_is_the_dense_one() {
        let replicates = 2;
        let positions = 3;
        let families = 3;
        let a = relationship(families);
        let people = a.nrows();
        let identity = DMatrix::<f64>::identity(people, people);
        let rows = people * replicates;
        let x = design(rows);
        let y = response(rows, positions);
        let model = RepeatedModel::build(&[a.clone(), identity.clone()], &x, replicates, positions)
            .expect("the model should build");

        // Arbitrary parameters, chosen only to be positive definite and to have
        // no accidental symmetry between the components.
        let make = |scale: f64, off: f64| {
            let mut m = DMatrix::<f64>::identity(positions, positions) * scale;
            for t in 0..positions {
                for u in 0..positions {
                    if t != u {
                        m[(t, u)] = off * (1.0 / (1.0 + (t as f64 - u as f64).abs()));
                    }
                }
            }
            m
        };
        let sigmas = vec![make(0.7, 0.25), make(0.4, -0.1)];
        let residual = make(1.1, 0.3);
        let mut fixed = DMatrix::<f64>::zeros(2, positions);
        for t in 0..positions {
            fixed[(0, t)] = 0.3 * (t as f64) - 0.4;
            fixed[(1, t)] = 0.2 - 0.05 * (t as f64);
        }

        let rotated = model.rotate(&y);
        let quick = model
            .loglik_at(&rotated, &sigmas, &residual, &fixed)
            .expect("the rotated likelihood should evaluate");
        let slow = dense_loglik(&[a, identity], &sigmas, &residual, &x, &fixed, &y);
        assert!(
            (quick - slow).abs() < 1e-9,
            "rotated {quick}, dense {slow}, difference {}",
            quick - slow
        );
    }

    #[test]
    fn three_replicates_agree_with_the_dense_likelihood_too() {
        let replicates = 3;
        let positions = 2;
        let a = relationship(2);
        let people = a.nrows();
        let identity = DMatrix::<f64>::identity(people, people);
        let rows = people * replicates;
        let x = design(rows);
        let y = response(rows, positions);
        let model = RepeatedModel::build(&[a.clone(), identity.clone()], &x, replicates, positions)
            .expect("the model should build");
        let mut sigma_one = DMatrix::<f64>::identity(positions, positions) * 0.6;
        sigma_one[(0, 1)] = 0.2;
        sigma_one[(1, 0)] = 0.2;
        let sigma_two = DMatrix::<f64>::identity(positions, positions) * 0.35;
        let residual = DMatrix::<f64>::identity(positions, positions) * 0.9;
        let fixed = DMatrix::<f64>::from_element(2, positions, 0.15);
        let rotated = model.rotate(&y);
        let quick = model
            .loglik_at(
                &rotated,
                &[sigma_one.clone(), sigma_two.clone()],
                &residual,
                &fixed,
            )
            .expect("the rotated likelihood should evaluate");
        let slow = dense_loglik(
            &[a, identity],
            &[sigma_one, sigma_two],
            &residual,
            &x,
            &fixed,
            &y,
        );
        assert!((quick - slow).abs() < 1e-9, "rotated {quick}, dense {slow}");
    }

    #[test]
    fn a_second_structured_component_is_refused_rather_than_approximated() {
        let a = relationship(2);
        let people = a.nrows();
        let mut household = DMatrix::<f64>::identity(people, people);
        household[(0, 5)] = 1.0;
        household[(5, 0)] = 1.0;
        let x = design(people * 2);
        let refused = RepeatedModel::build(&[a, household], &x, 2, 2);
        assert_eq!(refused.err(), Some("REPEATED_NOT_SIMULTANEOUSLY_DIAGONAL"));
    }

    #[test]
    fn expectation_maximisation_never_goes_downhill() {
        let replicates = 2;
        let positions = 3;
        let a = relationship(4);
        let people = a.nrows();
        let identity = DMatrix::<f64>::identity(people, people);
        let rows = people * replicates;
        let x = design(rows);
        let y = response(rows, positions);
        let model = RepeatedModel::build(&[a, identity], &x, replicates, positions)
            .expect("the model should build");

        let scale = {
            let count = (rows * positions) as f64;
            let mean = y.iter().sum::<f64>() / count;
            (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count).sqrt()
        };
        let rotated = model.rotate(&(&y / scale));
        let (mut sigmas, mut residual, mut fixed) = model
            .starting_values(&rotated)
            .expect("a start should exist");
        let mut previous = model
            .loglik_at(&rotated, &sigmas, &residual, &fixed)
            .expect("the start should evaluate");
        for step in 0..200 {
            let (next_sigmas, next_residual, next_fixed) = model
                .one_iteration(&rotated, &sigmas, &residual, &fixed)
                .expect("an iteration should evaluate");
            let next = model
                .loglik_at(&rotated, &next_sigmas, &next_residual, &next_fixed)
                .expect("an iteration should evaluate");
            assert!(
                next >= previous - 1e-9,
                "step {step} fell from {previous} to {next}"
            );
            sigmas = next_sigmas;
            residual = next_residual;
            fixed = next_fixed;
            previous = next;
        }
    }

    #[test]
    fn the_fit_stops_where_the_gradient_is_nought() {
        let replicates = 2;
        let positions = 2;
        let a = relationship(5);
        let people = a.nrows();
        let identity = DMatrix::<f64>::identity(people, people);
        let rows = people * replicates;
        let x = design(rows);
        let y = response(rows, positions);
        let model = RepeatedModel::build(&[a, identity], &x, replicates, positions)
            .expect("the model should build");
        let fit = model.fit(&y).expect("the fit should run");
        assert!(fit.monotone, "the likelihood fell during the search");
        assert!(
            fit.converged,
            "|g| = {}, after {} iterations",
            fit.scaled_gradient, fit.iterations
        );
        assert_eq!(fit.estimator, "ml");
        for position in 0..positions {
            let total: f64 = fit.variance_shares[position].iter().sum();
            assert!((total - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn the_fixed_effects_at_the_fixed_point_are_generalised_least_squares() {
        let replicates = 2;
        let positions = 2;
        let a = relationship(5);
        let people = a.nrows();
        let identity = DMatrix::<f64>::identity(people, people);
        let rows = people * replicates;
        let x = design(rows);
        let y = response(rows, positions);
        let model = RepeatedModel::build(&[a, identity], &x, replicates, positions)
            .expect("the model should build");
        let fit = model.fit(&y).expect("the fit should run");

        let count = (rows * positions) as f64;
        let mean = y.iter().sum::<f64>() / count;
        let scale = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count).sqrt();
        let rotated = model.rotate(&(&y / scale));
        let sigmas: Vec<DMatrix<f64>> = fit
            .component_covariances
            .iter()
            .map(|c| c / (scale * scale))
            .collect();
        let residual = &fit.residual_covariance / (scale * scale);
        let generalised = model
            .generalised_least_squares(&rotated, &sigmas, &residual)
            .expect("generalised least squares should solve");
        for a in 0..generalised.nrows() {
            for t in 0..positions {
                let expected = fit.fixed_effects[(a, t)] / scale;
                assert!(
                    (generalised[(a, t)] - expected).abs() < 1e-6,
                    "({a}, {t}): {} against {expected}",
                    generalised[(a, t)]
                );
            }
        }
    }
}
