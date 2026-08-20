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
//! Values that reached a limit instead of being measured, values that were
//! never measured at all, and a free covariance per component. **The covariance
//! kernel is still to come**, so until it arrives every `S_j` is unstructured:
//! a fit at seventeen positions estimates 153 numbers per component and nobody
//! should read a single one of them. What is here is the machinery, and what it
//! is for is landing on the answer where the answer is already known.
//!
//! # Censoring, and what it costs
//!
//! An observation is one of four things -- measured, above a limit, below a
//! limit, or never measured -- and [`Known`] carries which, so a value and its
//! status cannot disagree.
//!
//! What was measured contributes a density. What reached a limit contributes
//! the probability of the region it lies in, **conditional on everything that
//! family did measure**. What was never measured contributes nothing at all,
//! which is what makes the likelihood the right one for data missing at random.
//!
//! The expectation step fills all of them in anyway. A value that was never
//! measured is imputed rather than dropped, because dropping it would unbalance
//! the data and unbalanced data is what breaks the rotation. Filling in is not
//! the same as knowing: each rotated row carries the covariance of its own
//! imputation, and every statistic the maximisation forms picks that up in
//! place of an outer product it would otherwise take as certain. Leave it out
//! and the variances come back too small, which is the failure a test here is
//! written to catch.
//!
//! **The two halves are not the same size.** The maximisation is a pass over
//! blocks of `T`. The expectation step is a dense factorisation per family per
//! iteration, because conditioning on what a family measured does not survive
//! the rotation: the measured and unmeasured coordinates are scattered through
//! its rows. ADR 0010 counted that cost and accepted it -- the saving is not
//! against one likelihood evaluation but against the two hundred and fifty a
//! central-difference gradient would need.
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
use statrs::distribution::Normal;

use crate::blocks::family_blocks;
use crate::convergence::TOLERANCE;
use crate::dense::DenseFactor;
use crate::liability::{LiabilityModel, Truncation, truncated_moments};

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
    /// The share of all observations at each position that reached a limit,
    /// counting those never measured in the denominator. **Read the estimates
    /// against it**: at a very high share the tail at that
    /// position is carried by the censoring pattern rather than by anything
    /// measured.
    pub censored_shares: Vec<f64>,
    /// The largest family, in people.
    pub largest_family: usize,
    /// The largest number of censored values in one family, which is the
    /// dimension the sequential region approximation actually ran at. It is
    /// exact to two and approximate above.
    pub sequential_dimension: usize,
    pub estimator: &'static str,
}

/// What is known about one observation.
///
/// **A value and its status cannot disagree here**, because there is only one
/// of them. `src/tobit.rs` takes a value, a status and a limit as three
/// parallel arrays, which is what a numpy interface wants; this model has no
/// interface yet and can afford to make the impossible state unrepresentable
/// instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Known {
    /// Measured, and this is what it was.
    Value(f64),
    /// Not measured: known only to be at or above this limit, which is what an
    /// instrument that has run out of output leaves behind.
    Above(f64),
    /// Not measured: known only to be at or below this limit.
    Below(f64),
    /// Never measured at all, and nothing is known about where it lies. It is
    /// imputed rather than dropped, because dropping it would unbalance the
    /// data and unbalanced data is what breaks the rotation.
    Missing,
}

impl Known {
    /// The measured value, where there is one.
    fn value(self) -> Option<f64> {
        match self {
            Self::Value(value) => Some(value),
            _ => None,
        }
    }

    /// The limit and the direction, where the value is censored. A value that
    /// was never measured has neither and is carried rather than conditioned
    /// on.
    fn truncation(self) -> Option<Truncation> {
        match self {
            Self::Above(limit) => Some(Truncation { limit, sign: 1.0 }),
            Self::Below(limit) => Some(Truncation { limit, sign: -1.0 }),
            _ => None,
        }
    }
}

/// What one family leaves uncertain, in that family's own coordinates.
struct FamilySpread {
    /// The `(row, position)` each coordinate stands for.
    coordinates: Vec<(usize, usize)>,
    /// Their covariance, conditional on everything that family measured and on
    /// the region the rest of it lies in.
    covariance: DMatrix<f64>,
}

/// How uncertain each rotated row is once the unmeasured values are filled in.
///
/// A rotated row is a fixed linear combination of the raw rows, so its
/// covariance is that combination applied to the family covariances above. The
/// maximisation needs nothing else about the imputation: every statistic it
/// forms is quadratic in the data, so a mean and a covariance are between them
/// the whole expectation.
struct Spread {
    /// `P` of them, `T x T`: one per mean-channel row.
    mean: Vec<DMatrix<f64>>,
    /// `P (R - 1)` of them, one per contrast row.
    contrast: Vec<DMatrix<f64>>,
}

/// The data as the fit works on it: standardised, and rotated in advance where
/// that can be done at all.
///
/// **Rotating does not depend on the parameters, so where nothing is unmeasured
/// it is done once and never again.** Where something is unmeasured the
/// imputation changes with the parameters and so does the rotation of it, which
/// is the difference between the cheap half of this model and the expensive
/// one.
struct Prepared {
    known: Vec<Known>,
    rotated: Option<Rotated>,
}

/// The result of one expectation step.
struct Imputed {
    rotated: Rotated,
    /// `None` where nothing was unmeasured, so every rotated row is known
    /// exactly and there is nothing to correct for. The uncensored fit takes
    /// that path and pays none of this.
    spread: Option<Spread>,
    /// The observed-data log-likelihood at the parameters this was built from:
    /// a density for what was measured, and the probability of the region the
    /// rest lies in given it.
    ///
    /// **This is the definition of correct and it is not what the expectation
    /// step maximises.** The two are computed by different routes on purpose --
    /// this one through `region_log_probability`, which is exact at one and two
    /// coordinates where the sequential update is not, and which is what
    /// `TobitModel` uses and therefore what a comparison against it compares.
    loglik: f64,
}

/// The data rotated into the basis where the likelihood is block diagonal.
#[derive(Clone)]
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
    /// The person-level components as they were given. The rotation does not
    /// need them, but the expectation step over unmeasured values does: it
    /// works on one family's dense covariance, where there is no rotation to
    /// hide behind.
    matrices: Vec<DMatrix<f64>>,
    /// The families, as lists of people. Two people in different families are
    /// independent under every component at once, which is what makes the
    /// expectation step a loop over families rather than one dense solve of the
    /// whole roster.
    blocks: Vec<Vec<usize>>,
    /// The rotated diagonal of each person-level component: `diagonals[j][k]`
    /// is what component `j` contributes in eigendirection `k`. The residual is
    /// implicit and is not stored.
    diagonals: Vec<Vec<f64>>,
    /// How many eigendirections carry information about each component, which
    /// is the rank of its matrix and the divisor in that component's update.
    ranks: Vec<usize>,
    /// The design as given, `(P * R) x q`. The rotated forms below are what
    /// the maximisation uses; this is what the expectation step needs, because
    /// it works one family at a time in the response's own coordinates.
    design: DMatrix<f64>,
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

        // Blocks come from every matrix at once and not from the first, for
        // the reason `src/components.rs` gives: two people unrelated in the
        // pedigree may still share a household, and blocking on the
        // relationship alone would silently drop the covariance between them.
        let mut union = DMatrix::<f64>::zeros(people, people);
        for matrix in matrices {
            for i in 0..people {
                for j in 0..people {
                    if matrix[(i, j)] != 0.0 {
                        union[(i, j)] = 1.0;
                    }
                }
            }
        }
        let blocks = family_blocks(&union);

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
            design: design.clone(),
            matrices: matrices.to_vec(),
            blocks,
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
        let mut known = Vec::with_capacity(y.nrows() * y.ncols());
        for row in 0..y.nrows() {
            for position in 0..y.ncols() {
                known.push(Known::Value(y[(row, position)]));
            }
        }
        self.fit_known(&known)
    }

    /// Fit where some values were never measured, or reached a limit instead
    /// of being measured.
    ///
    /// `known` is `(P * R) * T` long, with observation `(row, position)` at
    /// `row * T + position` -- the same row order as the design.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the data do not match the model, leave
    /// nothing measured to set a scale by, or produce a covariance that cannot
    /// be factorised.
    pub fn fit_known(&self, known: &[Known]) -> Result<RepeatedFit, &'static str> {
        let rows = self.people * self.replicates;
        let positions = self.positions;
        if known.len() != rows * positions {
            return Err("REPEATED_RESPONSE_WRONG_SHAPE");
        }

        // Standardised for the same reason the one-trait model standardises:
        // it puts the gradient reading on a scale that means the same thing
        // from one data set to the next, so one threshold can serve them all.
        // The scale comes from what was measured, because that is the only
        // thing there is; a limit is carried by it rather than counted in it.
        let measured: Vec<f64> = known.iter().filter_map(|k| k.value()).collect();
        if measured.len() < 2 {
            return Err("REPEATED_NOTHING_MEASURED");
        }
        let count = measured.len() as f64;
        let mean = measured.iter().sum::<f64>() / count;
        let variance = measured.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
        if !(variance > 0.0) {
            return Err("REPEATED_RESPONSE_CONSTANT");
        }
        let scale = variance.sqrt();
        let scaled: Vec<Known> = known
            .iter()
            .map(|k| match *k {
                Known::Value(value) => Known::Value(value / scale),
                Known::Above(limit) => Known::Above(limit / scale),
                Known::Below(limit) => Known::Below(limit / scale),
                Known::Missing => Known::Missing,
            })
            .collect();
        let complete = scaled.iter().all(|k| k.value().is_some());
        let crude = self.rotate(&self.crude(&scaled));
        let prepared = Prepared {
            rotated: complete.then(|| crude.clone()),
            known: scaled,
        };

        let (mut sigmas, mut residual, mut fixed) = self.starting_values(&crude)?;
        let mut imputed = self
            .expectation(&prepared, &sigmas, &residual, &fixed, true)
            .ok_or("REPEATED_START_NOT_EVALUABLE")?;
        let mut loglik = imputed.loglik;
        let mut monotone = true;
        let mut iterations = 0;
        let mut previous_gain = 0.0;
        let mut relaxed = false;
        let mut scaled_gradient = f64::INFINITY;
        let mut last_reading: Option<usize> = None;

        while iterations < MAX_ITERATIONS {
            let Some(((next_sigmas, next_residual, next_fixed), next_imputed)) = self
                .accelerated_step(
                    &prepared,
                    &imputed,
                    &sigmas,
                    &residual,
                    &fixed,
                    &mut iterations,
                )
            else {
                return Err("REPEATED_ITERATION_NOT_EVALUABLE");
            };
            let next_loglik = next_imputed.loglik;
            // A fall of a few units in the last place is the arithmetic and not
            // the algorithm. Anything larger, on complete data, is this code
            // being wrong; with censoring it may instead be the expectation
            // step's own approximation, which ADR 0010 warns is not guaranteed
            // to climb the likelihood it claims to maximise. Either way it is
            // reported rather than swallowed.
            if next_loglik < loglik - 1e-8 * loglik.abs().max(1.0) {
                monotone = false;
            }
            let gain = next_loglik - loglik;
            sigmas = next_sigmas;
            residual = next_residual;
            fixed = next_fixed;
            imputed = next_imputed;
            loglik = next_loglik;
            if gain <= 0.0 {
                break;
            }

            // **The gradient is read on a schedule only where reading it is
            // cheap.** On complete data an evaluation is a pass over blocks of
            // `T`; with censoring it is a dense factorisation per family, and
            // two of those per parameter is not something to do every five
            // hundred iterations. There it is read once, at the end, which is
            // what ADR 0010 asks for.
            let periodic = complete
                && iterations
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
                if let Some(((rested_sigmas, rested_residual, rested_fixed), rested_imputed)) =
                    self.rested(&prepared, &sigmas, &residual, &fixed, loglik)
                {
                    sigmas = rested_sigmas;
                    residual = rested_residual;
                    fixed = rested_fixed;
                    loglik = rested_imputed.loglik;
                    imputed = rested_imputed;
                }
                scaled_gradient = self
                    .gradient_reading(&prepared, complete, &imputed, &sigmas, &residual, &fixed)
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
            if let Some(((rested_sigmas, rested_residual, rested_fixed), rested_imputed)) =
                self.rested(&prepared, &sigmas, &residual, &fixed, loglik)
            {
                sigmas = rested_sigmas;
                residual = rested_residual;
                fixed = rested_fixed;
                loglik = rested_imputed.loglik;
                imputed = rested_imputed;
            }
            scaled_gradient = self
                .gradient_reading(&prepared, complete, &imputed, &sigmas, &residual, &fixed)
                .unwrap_or(f64::INFINITY);
        }

        // Back to the response's own units. A covariance carries the square of
        // the scale, a fixed effect carries the scale itself, and the
        // log-likelihood carries a term per **measured** observation, because a
        // region probability is a probability and does not change with the
        // units its limits are quoted in.
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
            censored_shares: self.censored_shares(known),
            largest_family: self.blocks.iter().map(Vec::len).max().unwrap_or(0),
            sequential_dimension: self.sequential_dimension(known),
            estimator: "ml",
        })
    }

    /// The share of observations at each position that reached a limit.
    fn censored_shares(&self, known: &[Known]) -> Vec<f64> {
        let rows = self.people * self.replicates;
        (0..self.positions)
            .map(|position| {
                let hit = (0..rows)
                    .filter(|row| {
                        matches!(
                            known[row * self.positions + position],
                            Known::Above(_) | Known::Below(_)
                        )
                    })
                    .count();
                hit as f64 / rows as f64
            })
            .collect()
    }

    /// The largest number of censored values one family carries, which is the
    /// dimension the sequential approximation actually runs at.
    ///
    /// **This is in the fit record because the whole result leans on it.** The
    /// region probability is exact to two coordinates and approximate above,
    /// and a reader should not have to work out from the censoring pattern how
    /// far past two a given fit went.
    fn sequential_dimension(&self, known: &[Known]) -> usize {
        self.blocks
            .iter()
            .map(|block| {
                let mut counted = 0;
                for &person in block {
                    for replicate in 0..self.replicates {
                        let row = person * self.replicates + replicate;
                        for position in 0..self.positions {
                            if matches!(
                                known[row * self.positions + position],
                                Known::Above(_) | Known::Below(_)
                            ) {
                                counted += 1;
                            }
                        }
                    }
                }
                counted
            })
            .max()
            .unwrap_or(0)
    }

    /// A complete response to take starting values from.
    ///
    /// **Substituting the limit is the analysis this model exists to replace**,
    /// and it appears here and nowhere else. It costs nothing to be wrong at a
    /// start: expectation-maximisation from a poor one reaches the same maximum
    /// as from a good one, only later. What a start must not be is unfittable,
    /// and a substituted limit never is.
    fn crude(&self, known: &[Known]) -> DMatrix<f64> {
        let rows = self.people * self.replicates;
        let positions = self.positions;
        let mut values = DMatrix::<f64>::zeros(rows, positions);
        for position in 0..positions {
            let mut total = 0.0;
            let mut counted = 0.0;
            for row in 0..rows {
                if let Some(value) = known[row * positions + position].value() {
                    total += value;
                    counted += 1.0;
                }
            }
            let average = if counted > 0.0 { total / counted } else { 0.0 };
            for row in 0..rows {
                values[(row, position)] = match known[row * positions + position] {
                    Known::Value(value) => value,
                    Known::Above(limit) | Known::Below(limit) => limit,
                    Known::Missing => average,
                };
            }
        }
        values
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

    /// The expectation step: fill in everything that was not measured, and say
    /// how uncertain each filling is.
    ///
    /// # Why this is the expensive half
    ///
    /// The maximisation works in a basis where the likelihood is block diagonal
    /// in blocks of `T`, and that basis exists only for complete balanced data.
    /// Conditioning on what a family actually measured does not survive it:
    /// the measured and unmeasured coordinates are scattered through the
    /// family's rows, so the covariance has to be built and factorised in the
    /// response's own coordinates, densely, once per family per iteration.
    ///
    /// ADR 0010 counted the cost and accepted it. The saving that route buys is
    /// not against one likelihood evaluation but against the two hundred and
    /// fifty a central-difference gradient would need, and this is the one.
    ///
    /// # What it computes
    ///
    /// Per family, in order: the density of what was measured; the conditional
    /// distribution of what was not, given what was; the probability of the
    /// region the censored values lie in, given the same; and then the moments
    /// of that conditional distribution truncated to the region.
    ///
    /// A value that was never measured has no region of its own and is
    /// conditioned on nothing, but it is correlated with values that were
    /// censored and its moments move when they are truncated -- so it is
    /// carried through the same update rather than filled in beforehand.
    fn expectation(
        &self,
        prepared: &Prepared,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
        // Where this is false only the log-likelihood is wanted, so the
        // truncated moments are not taken and the rotation comes back empty.
        moments: bool,
    ) -> Option<Imputed> {
        // Nothing was unmeasured, so there is nothing to condition on and
        // nothing to impute. The likelihood is the ordinary one in the rotated
        // basis and the rotation was done once, before the search began.
        if let Some(rotated) = &prepared.rotated {
            let loglik = self.loglik_at(rotated, sigmas, residual, fixed)?;
            return Some(Imputed {
                rotated: rotated.clone(),
                spread: None,
                loglik,
            });
        }
        let known = &prepared.known;
        let rows = self.people * self.replicates;
        let positions = self.positions;
        let normal = Normal::new(0.0, 1.0).ok()?;
        let expected = &self.design * fixed;
        let mut values = DMatrix::<f64>::zeros(rows, positions);
        for row in 0..rows {
            for position in 0..positions {
                if let Some(value) = known[row * positions + position].value() {
                    values[(row, position)] = value;
                }
            }
        }
        let mut loglik = 0.0;
        let mut spreads: Vec<FamilySpread> = Vec::new();

        for block in &self.blocks {
            let mut coordinates: Vec<(usize, usize)> = Vec::new();
            for &person in block {
                for replicate in 0..self.replicates {
                    let row = person * self.replicates + replicate;
                    for position in 0..positions {
                        coordinates.push((row, position));
                    }
                }
            }
            let size = coordinates.len();
            let covariance = self.family_covariance(&coordinates, sigmas, residual);
            let mean: Vec<f64> = coordinates
                .iter()
                .map(|&(row, position)| expected[(row, position)])
                .collect();

            let mut measured = Vec::new();
            let mut unmeasured = Vec::new();
            for index in 0..size {
                let (row, position) = coordinates[index];
                if known[row * positions + position].value().is_some() {
                    measured.push(index);
                } else {
                    unmeasured.push(index);
                }
            }

            // What was measured, as a density, and the factor that conditions
            // what was not on it.
            let mut conditioning = None;
            if !measured.is_empty() {
                let block_covariance = DMatrix::from_fn(measured.len(), measured.len(), |i, j| {
                    covariance[(measured[i], measured[j])]
                });
                // **This is the one factorisation that decides what a fit
                // costs.** The largest family in the audiogram design has 75
                // people, two replicates and seventeen positions, so this is
                // 2,550 rows, and `src/dense.rs` measured `faer` at eighteen
                // times `nalgebra` by 1,800 of them.
                let factor = DenseFactor::new(&block_covariance)?;
                let logdet = factor.logdet();
                let deviation = DVector::from_iterator(
                    measured.len(),
                    measured.iter().map(|&index| {
                        let (row, position) = coordinates[index];
                        values[(row, position)] - mean[index]
                    }),
                );
                let solved = factor.solve_vector(&deviation);
                let quadratic = (deviation.transpose() * &solved)[(0, 0)];
                loglik -= 0.5 * (measured.len() as f64 * LN_2PI + logdet + quadratic);
                if !loglik.is_finite() {
                    return None;
                }
                conditioning = Some((factor, solved));
            }
            if unmeasured.is_empty() {
                continue;
            }

            let marginal = DMatrix::from_fn(unmeasured.len(), unmeasured.len(), |i, j| {
                covariance[(unmeasured[i], unmeasured[j])]
            });
            let (mut conditional_mean, mut conditional) = match &conditioning {
                Some((factor, solved)) => {
                    let cross = DMatrix::from_fn(unmeasured.len(), measured.len(), |i, j| {
                        covariance[(unmeasured[i], measured[j])]
                    });
                    let regression = factor.solve_matrix(&cross.transpose());
                    let shift = &cross * solved;
                    (
                        unmeasured
                            .iter()
                            .enumerate()
                            .map(|(i, &index)| mean[index] + shift[i])
                            .collect::<Vec<f64>>(),
                        marginal - &cross * &regression,
                    )
                }
                // Nobody in this family measured anything, so the conditional
                // distribution is the marginal one.
                None => (
                    unmeasured.iter().map(|&index| mean[index]).collect(),
                    marginal,
                ),
            };
            self.region_and_moments(
                known,
                &coordinates,
                &unmeasured,
                &mut conditional_mean,
                &mut conditional,
                &normal,
                moments,
                &mut loglik,
                &mut values,
                &mut spreads,
            )?;
        }

        // Where the moments were not wanted, the imputed cells were never
        // filled and rotating them would give an answer that looks right and
        // is not. An empty rotation is returned instead, so that using it is a
        // shape error rather than a silent one.
        let rotated = if moments {
            self.rotate(&values)
        } else {
            Rotated {
                mean: DMatrix::zeros(0, 0),
                contrast: DMatrix::zeros(0, 0),
            }
        };
        let spread = (moments && !spreads.is_empty()).then(|| self.collapse(&spreads));
        loglik.is_finite().then_some(Imputed {
            rotated,
            spread,
            loglik,
        })
    }

    /// The probability of the region the censored values lie in, and then the
    /// moments of the conditional distribution truncated to it.
    ///
    /// The two use different routines on purpose. The probability comes from
    /// `region_log_probability`, which is exact at one and two coordinates and
    /// is what every other censored model in the crate reports, so a comparison
    /// against one of them compares like with like. The moments come from the
    /// sequential update, which is the only thing that has them.
    #[allow(clippy::too_many_arguments)]
    fn region_and_moments(
        &self,
        known: &[Known],
        coordinates: &[(usize, usize)],
        unmeasured: &[usize],
        conditional_mean: &mut [f64],
        conditional: &mut DMatrix<f64>,
        normal: &Normal,
        moments: bool,
        loglik: &mut f64,
        values: &mut DMatrix<f64>,
        spreads: &mut Vec<FamilySpread>,
    ) -> Option<()> {
        let positions = self.positions;
        let truncation: Vec<Option<Truncation>> = unmeasured
            .iter()
            .map(|&index| {
                let (row, position) = coordinates[index];
                known[row * positions + position].truncation()
            })
            .collect();

        let censored: Vec<usize> = (0..unmeasured.len())
            .filter(|&i| truncation[i].is_some())
            .collect();
        if !censored.is_empty() {
            let region_mean: Vec<f64> = censored
                .iter()
                .map(|&i| conditional_mean[i] - truncation[i].map_or(0.0, |t| t.limit))
                .collect();
            let sign: Vec<f64> = censored
                .iter()
                .map(|&i| truncation[i].map_or(1.0, |t| t.sign))
                .collect();
            let region = DMatrix::from_fn(censored.len(), censored.len(), |a, b| {
                conditional[(censored[a], censored[b])]
            });
            *loglik +=
                LiabilityModel::region_log_probability(&region_mean, &sign, &region, normal)?;
        }
        if !moments {
            return Some(());
        }
        truncated_moments(conditional_mean, conditional, &truncation, normal)?;
        for (i, &index) in unmeasured.iter().enumerate() {
            let (row, position) = coordinates[index];
            values[(row, position)] = conditional_mean[i];
        }
        spreads.push(FamilySpread {
            coordinates: unmeasured.iter().map(|&index| coordinates[index]).collect(),
            covariance: conditional.clone(),
        });
        Some(())
    }

    /// One family's dense covariance, in the response's own coordinates.
    fn family_covariance(
        &self,
        coordinates: &[(usize, usize)],
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let size = coordinates.len();
        let mut covariance = DMatrix::<f64>::zeros(size, size);
        for a in 0..size {
            let (row_a, t) = coordinates[a];
            let person_a = row_a / self.replicates;
            for b in 0..=a {
                let (row_b, u) = coordinates[b];
                let person_b = row_b / self.replicates;
                let mut value = 0.0;
                for (component, sigma) in sigmas.iter().enumerate() {
                    let shared = self.matrices[component][(person_a, person_b)];
                    if shared != 0.0 {
                        value += shared * sigma[(t, u)];
                    }
                }
                if row_a == row_b {
                    value += residual[(t, u)];
                }
                covariance[(a, b)] = value;
                covariance[(b, a)] = value;
            }
        }
        covariance
    }

    /// Carry the family covariances through the rotation, one rotated row at a
    /// time.
    ///
    /// A rotated row is the same fixed combination of raw rows at every
    /// position, so its covariance is that combination applied twice to the
    /// family covariance and then gathered by position. Families are
    /// independent, so a row touching several of them simply sums.
    fn collapse(&self, families: &[FamilySpread]) -> Spread {
        let positions = self.positions;
        let root = (self.replicates as f64).sqrt();
        let mut mean = vec![DMatrix::<f64>::zeros(positions, positions); self.people];
        let mut contrast =
            vec![DMatrix::<f64>::zeros(positions, positions); self.people * (self.replicates - 1)];

        for family in families {
            let size = family.coordinates.len();
            let mut weights = vec![0.0; size];
            for direction in 0..self.people {
                let mut touched = false;
                for (slot, &(row, _)) in family.coordinates.iter().enumerate() {
                    let weight = self.eigenvectors[(row / self.replicates, direction)] / root;
                    weights[slot] = weight;
                    touched |= weight != 0.0;
                }
                if touched {
                    gather(&mut mean[direction], family, &weights);
                }
            }
            let people: Vec<usize> = {
                let mut seen: Vec<usize> = family
                    .coordinates
                    .iter()
                    .map(|&(row, _)| row / self.replicates)
                    .collect();
                seen.dedup();
                seen
            };
            for person in people {
                for step in 1..self.replicates {
                    for (slot, &(row, _)) in family.coordinates.iter().enumerate() {
                        weights[slot] = if row / self.replicates == person {
                            self.helmert[(step, row % self.replicates)]
                        } else {
                            0.0
                        };
                    }
                    gather(
                        &mut contrast[person * (self.replicates - 1) + step - 1],
                        family,
                        &weights,
                    );
                }
            }
        }
        Spread { mean, contrast }
    }

    /// Factor every mean-channel block once.
    fn blocks(&self, sigmas: &[DMatrix<f64>], residual: &DMatrix<f64>) -> Option<Vec<Block>> {
        (0..self.people)
            .map(|k| {
                let v = self.mean_covariance(k, sigmas, residual);
                let factor = DenseFactor::new(&v)?;
                Some(Block {
                    inverse: factor.inverse(),
                    log_determinant: factor.logdet(),
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
        imputed: &Imputed,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
    ) -> Option<State> {
        let rotated = &imputed.rotated;
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
            // **Where a rotated row was imputed rather than measured, its
            // second moment is not the square of its mean.** Every statistic
            // below is quadratic in the data, so each outer product of a
            // rotated quantity picks up the same term with the row's own
            // covariance in place of that outer product. Written through the
            // inverse once here because both the components and the residual
            // want it.
            let carried = imputed
                .spread
                .as_ref()
                .map(|spread| &block.inverse * &spread.mean[k] * &block.inverse);
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
                if let Some(carried) = &carried {
                    statistics[component] += sigma * carried * sigma * (replicates * d);
                }
            }
            for t in 0..positions {
                effects[(k, t)] = total[t];
            }
            residual_variance += residual - residual * &block.inverse * residual;
            if let Some(carried) = &carried {
                // The residual of this row is the residual covariance times the
                // inverse times the row, so its own uncertainty arrives the
                // same way.
                residual_variance += residual * carried * residual;
            }
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
        // A contrast row carries no person-level effect at all, so what is left
        // uncertain in it is left uncertain in its residual, with nothing in
        // between to change it.
        if let Some(spread) = &imputed.spread {
            for row in &spread.contrast {
                residual_variance += row;
            }
        }
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
        prepared: &Prepared,
        imputed: &Imputed,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
        iterations: &mut usize,
    ) -> Option<(State, Imputed)> {
        let (one_sigmas, one_residual, one_fixed) =
            self.one_iteration(imputed, sigmas, residual, fixed)?;
        let imputed_one =
            self.expectation(prepared, &one_sigmas, &one_residual, &one_fixed, true)?;
        *iterations += 1;
        let (two_sigmas, two_residual, two_fixed) =
            self.one_iteration(&imputed_one, &one_sigmas, &one_residual, &one_fixed)?;
        let imputed_two =
            self.expectation(prepared, &two_sigmas, &two_residual, &two_fixed, true)?;
        *iterations += 1;

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
            return Some(((two_sigmas, two_residual, two_fixed), imputed_two));
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
            let attempt = self.expectation(prepared, &try_sigmas, &try_residual, &try_fixed, true);
            *iterations += 1;
            if let Some(attempt) = attempt
                && attempt.loglik >= imputed_two.loglik
            {
                return Some(((try_sigmas, try_residual, try_fixed), attempt));
            }
            length = (length - 1.0) / 2.0;
            if length >= -1.0 - 1e-6 {
                break;
            }
        }
        Some(((two_sigmas, two_residual, two_fixed), imputed_two))
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
        prepared: &Prepared,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
        loglik: f64,
    ) -> Option<(State, Imputed)> {
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
        let imputed = self.expectation(prepared, &rested, &rested_residual, fixed, true)?;
        if imputed.loglik < loglik {
            return None;
        }
        Some(((rested, rested_residual, fixed.clone()), imputed))
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
    /// differences on the observed-data log-likelihood.
    ///
    /// # Two cases, and why they differ
    ///
    /// On complete data the fixed effects have a closed-form maximum at any
    /// variances, so they are profiled out and the reading is of the profile
    /// likelihood, which is what the rest of the crate reports.
    ///
    /// With censoring they do not: there is no generalised least squares
    /// solution to a likelihood that is part density and part region
    /// probability. So they are stepped in like everything else, and the
    /// reading covers them too. That is a stronger statement rather than a
    /// weaker one -- it says the fit is stationary in every parameter at once
    /// -- and it costs a reading over `q T` more directions.
    fn gradient_reading(
        &self,
        prepared: &Prepared,
        complete: bool,
        imputed: &Imputed,
        sigmas: &[DMatrix<f64>],
        residual: &DMatrix<f64>,
        fixed: &DMatrix<f64>,
    ) -> Option<f64> {
        let mut roots: Vec<DMatrix<f64>> = sigmas.iter().map(symmetric_square_root).collect();
        roots.push(symmetric_square_root(residual));

        let value_at = |roots: &[DMatrix<f64>], fixed: &DMatrix<f64>| -> Option<f64> {
            let (last, rest) = roots.split_last()?;
            let sigmas: Vec<DMatrix<f64>> = rest.iter().map(|l| symmetrised(&(l * l))).collect();
            let residual = symmetrised(&(last * last));
            if complete {
                self.profile_loglik(&imputed.rotated, &sigmas, &residual)
            } else {
                Some(
                    self.expectation(prepared, &sigmas, &residual, fixed, false)?
                        .loglik,
                )
            }
        };
        let at = value_at(&roots, fixed)?;

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
                    let high = value_at(&up, fixed)?;
                    let low = value_at(&down, fixed)?;
                    worst = worst.max(((high - low) / (2.0 * GRADIENT_STEP)).abs());
                }
            }
        }
        if !complete {
            for a in 0..self.covariates {
                for t in 0..self.positions {
                    let mut up = fixed.clone();
                    let mut down = fixed.clone();
                    up[(a, t)] += GRADIENT_STEP;
                    down[(a, t)] -= GRADIENT_STEP;
                    let high = value_at(&roots, &up)?;
                    let low = value_at(&roots, &down)?;
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

/// Add one rotated row's share of a family's covariance into that row's own
/// `T x T` total.
///
/// The weight sits on the raw row and so is the same at every position, which
/// is what makes this a gather by position rather than a matrix product.
fn gather(target: &mut DMatrix<f64>, family: &FamilySpread, weights: &[f64]) {
    for (a, &(_, t)) in family.coordinates.iter().enumerate() {
        if weights[a] == 0.0 {
            continue;
        }
        for (b, &(_, u)) in family.coordinates.iter().enumerate() {
            if weights[b] == 0.0 {
                continue;
            }
            target[(t, u)] += weights[a] * weights[b] * family.covariance[(a, b)];
        }
    }
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
    use super::{
        Imputed, Known, LN_2PI, Prepared, RepeatedModel, Rotated, helmert_matrix,
        rotate_replicates, symmetric_square_root,
    };
    use nalgebra::{DMatrix, DVector};
    use statrs::distribution::Normal;

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

    /// **The censored likelihood, written out by hand.** One value that
    /// reached a limit, in a family of four people with two replicates apiece
    /// at two positions, so that the censored coordinate is correlated with
    /// fifteen measured ones through three different levels at once. The answer
    /// has to be the density of what was measured times the probability that
    /// the one unmeasured value lies where it is known to lie, **given** the
    /// rest -- and the conditional part is the part that is easy to get wrong,
    /// because it is what stops the censored record being counted twice.
    #[test]
    fn the_censored_likelihood_is_the_density_and_the_region_given_it() {
        let replicates = 2;
        let positions = 2;
        let a = relationship(1);
        let people = a.nrows();
        let identity = DMatrix::<f64>::identity(people, people);
        let rows = people * replicates;
        let x = design(rows);
        let y = response(rows, positions);
        let model = RepeatedModel::build(&[a.clone(), identity.clone()], &x, replicates, positions)
            .expect("the model should build");

        let make = |scale: f64, off: f64| {
            let mut m = DMatrix::<f64>::identity(positions, positions) * scale;
            m[(0, 1)] = off;
            m[(1, 0)] = off;
            m
        };
        let sigmas = vec![make(0.7, 0.25), make(0.4, -0.1)];
        let residual = make(1.1, 0.3);
        let mut fixed = DMatrix::<f64>::zeros(2, positions);
        for t in 0..positions {
            fixed[(0, t)] = 0.3 * (t as f64) - 0.4;
            fixed[(1, t)] = 0.2 - 0.05 * (t as f64);
        }

        // One value reached a limit a little under what it would have been.
        let (censored_row, censored_position) = (5, 1);
        let limit = y[(censored_row, censored_position)] - 0.4;
        let mut known = Vec::with_capacity(rows * positions);
        for row in 0..rows {
            for position in 0..positions {
                known.push(if row == censored_row && position == censored_position {
                    Known::Above(limit)
                } else {
                    Known::Value(y[(row, position)])
                });
            }
        }
        let prepared = Prepared {
            known,
            rotated: None,
        };
        let got = model
            .expectation(&prepared, &sigmas, &residual, &fixed, true)
            .expect("the expectation step should run");

        // The same thing with no rotation and no conditioning machinery: build
        // the family covariance, take the measured block out of it, and do the
        // regression by hand.
        let size = rows * positions;
        let mut covariance = DMatrix::<f64>::zeros(size, size);
        for row_one in 0..rows {
            for row_two in 0..rows {
                for t in 0..positions {
                    for u in 0..positions {
                        let mut value = a[(row_one / replicates, row_two / replicates)]
                            * sigmas[0][(t, u)]
                            + identity[(row_one / replicates, row_two / replicates)]
                                * sigmas[1][(t, u)];
                        if row_one == row_two {
                            value += residual[(t, u)];
                        }
                        covariance[(row_one * positions + t, row_two * positions + u)] = value;
                    }
                }
            }
        }
        let expected = &x * &fixed;
        let censored = censored_row * positions + censored_position;
        let measured: Vec<usize> = (0..size).filter(|&at| at != censored).collect();

        let block = DMatrix::from_fn(measured.len(), measured.len(), |i, j| {
            covariance[(measured[i], measured[j])]
        });
        let factor = block
            .cholesky()
            .expect("the measured block should factorise");
        let logdet = 2.0 * factor.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        let deviation = DVector::from_iterator(
            measured.len(),
            measured.iter().map(|&at| {
                let (row, position) = (at / positions, at % positions);
                y[(row, position)] - expected[(row, position)]
            }),
        );
        let solved = factor.solve(&deviation);
        let quadratic = (deviation.transpose() * &solved)[(0, 0)];
        let density = -0.5 * (measured.len() as f64 * LN_2PI + logdet + quadratic);

        let cross = DMatrix::from_fn(1, measured.len(), |_, j| {
            covariance[(censored, measured[j])]
        });
        let conditional_mean = expected[(censored_row, censored_position)] + (&cross * &solved)[0];
        let conditional_variance =
            covariance[(censored, censored)] - (&cross * factor.solve(&cross.transpose()))[(0, 0)];
        let normal = Normal::new(0.0, 1.0).expect("the standard normal exists");
        let region = statrs::distribution::ContinuousCDF::cdf(
            &normal,
            (conditional_mean - limit) / conditional_variance.sqrt(),
        )
        .ln();

        assert!(
            (got.loglik - (density + region)).abs() < 1e-9,
            "{} against {} = density {density} + region {region}",
            got.loglik,
            density + region
        );
        // And the value itself is filled in above its limit, where it is known
        // to be, rather than at it.
        let filled = got.rotated.mean.nrows();
        assert!(filled > 0);
        assert!(
            got.spread.is_some(),
            "something was unmeasured, so something is uncertain"
        );
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
        // Nothing is unmeasured here, so the expectation step has nothing to
        // impute and every rotated row is known exactly.
        let complete = |rotated: &Rotated| Imputed {
            rotated: Rotated {
                mean: rotated.mean.clone(),
                contrast: rotated.contrast.clone(),
            },
            spread: None,
            loglik: 0.0,
        };
        let mut previous = model
            .loglik_at(&rotated, &sigmas, &residual, &fixed)
            .expect("the start should evaluate");
        for step in 0..200 {
            let (next_sigmas, next_residual, next_fixed) = model
                .one_iteration(&complete(&rotated), &sigmas, &residual, &fixed)
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
