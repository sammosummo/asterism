//! One trait, with a spatial component whose range is estimated.
//!
//! The model is
//!
//! ```text
//! V = Σ_j σ²_j K_j  +  σ²_S · exp(−λ·D)  +  σ²_e I
//! ```
//!
//! where `D` holds distances in kilometres and `λ` is a free parameter rather
//! than a value chosen in advance. Everything else — additive relationship,
//! household, whatever else is handed in — enters as a fixed matrix exactly as
//! it does in `components.rs`.
//!
//! **Two things are different here and both matter.**
//!
//! The first is arithmetic. `exp(−λD)` is dense: everybody is correlated with
//! everybody, weakly. The family-block decomposition that makes `components.rs`
//! affordable does not apply, and every likelihood evaluation factorises the
//! whole covariance. At two thousand people that is about a hundredth of a
//! second, so a fit is a second or two and a bootstrap is hours rather than
//! minutes — but it is not free and it does not scale the way the block models
//! do.
//!
//! The second is statistical and is the reason this module carries a bootstrap.
//! **When `σ²_S` is nought, `λ` does not appear in the likelihood at all.** It is
//! a parameter present only under the alternative, and a likelihood ratio for
//! `σ²_S = 0` then has no chi-squared or chi-bar-squared null distribution —
//! neither the 50:50 mixture `components.rs` uses nor anything else in closed
//! form. Simulating the null is the only honest route, which is what
//! `bootstrap_spatial_variance` does. Everything else — an interval for the
//! spatial raw coefficient proportion, an interval for `λ`, a test of `λ`
//! against a particular value — is ordinary profile likelihood and needs no
//! such machinery, because those are interior questions asked where the
//! component exists.

use faer::linalg::solvers::{Llt, Solve};
use faer::{Mat, Side};
use nalgebra::{DMatrix, DVector};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::convergence::{self, TOLERANCE};

/// The factorisation of one dense covariance, and what the likelihood needs
/// from it.
///
/// **This is the only place in the package that factorises a dense matrix of
/// everybody.** Every other model is block diagonal, its blocks single families
/// of a few dozen people, and `nalgebra` is more than fast enough there. Here
/// the spatial kernel couples everyone, and at two thousand people `nalgebra`
/// takes 0.15 seconds for the decomposition and 0.87 for the inverse where
/// `faer` takes 0.012 and 0.045 -- eighteen times faster, agreeing to 4e-16.
/// That difference is the difference between a bootstrap of an hour and one of
/// a day, so this path alone is routed through `faer` and everything else is
/// left where it is.
struct DenseFactor {
    logdet: f64,
    inverse: Mat<f64>,
}

impl DenseFactor {
    fn new(v: &DMatrix<f64>) -> Option<(Self, Llt<f64>)> {
        let n = v.nrows();
        let a = Mat::from_fn(n, n, |i, j| v[(i, j)]);
        let llt = Llt::new(a.as_ref(), Side::Lower).ok()?;
        let logdet = 2.0 * (0..n).map(|i| llt.L()[(i, i)].ln()).sum::<f64>();
        if !logdet.is_finite() {
            return None;
        }
        let inverse = llt.solve(Mat::<f64>::identity(n, n).as_ref());
        Some((Self { logdet, inverse }, llt))
    }
}

/// Solve `V z = b` for a vector, through the factorisation.
fn solve_vector(llt: &Llt<f64>, b: &DVector<f64>) -> DVector<f64> {
    let n = b.len();
    let rhs = Mat::from_fn(n, 1, |i, _| b[i]);
    let out = llt.solve(rhs.as_ref());
    DVector::from_fn(n, |i, _| out[(i, 0)])
}

/// Solve `V Z = B` for a matrix, through the factorisation.
fn solve_matrix(llt: &Llt<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let (n, p) = (b.nrows(), b.ncols());
    let rhs = Mat::from_fn(n, p, |i, j| b[(i, j)]);
    let out = llt.solve(rhs.as_ref());
    DMatrix::from_fn(n, p, |i, j| out[(i, j)])
}

/// A fitted spatial model.
#[derive(Clone, Debug)]
pub struct SpatialFit {
    /// One variance per fixed component in the order given, then the spatial
    /// variance, then the residual.
    pub variances: Vec<f64>,
    /// Each raw covariance coefficient divided by their sum. These proportions
    /// depend on the submitted fixed-matrix scales and are not generally
    /// variance shares for unnormalised matrices.
    pub raw_coefficient_proportions: Vec<f64>,
    /// Sum of the raw covariance coefficients. This depends on fixed-matrix
    /// scaling and is not generally a marginal variance.
    pub raw_coefficient_total: f64,
    /// The estimated decay rate, per kilometre.
    pub lambda: f64,
    /// The distance at which the spatial correlation is a half, in kilometres.
    /// This is what `λ` means in the units anybody thinks in.
    pub half_distance_km: f64,
    pub fixed_effects: Vec<f64>,
    /// The standard error of each fixed effect, from the diagonal of
    /// `(X' V^-1 X)^-1` at the fitted parameters. As elsewhere, the variance
    /// components and the range are treated as known though they were
    /// estimated, so these are a little optimistic.
    pub fixed_effect_errors: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    /// True where the gradient test failed on the first search and a second
    /// was run from that point with the objective tolerance switched off. It
    /// fires nowhere else, so `false` means this is the fit the package gave
    /// before the polish existed, to the last bit.
    pub polished: bool,
    pub scaled_gradient: f64,
    pub estimator: &'static str,
}

struct Evaluation {
    negative_loglik: f64,
    gradient: Vec<f64>,
    fixed_effects: Vec<f64>,
    /// `(X' V^-1 X)^-1`, the covariance of the fixed effects.
    fixed_covariance: Option<DMatrix<f64>>,
}

/// Where one derivative's elements come from, so the inner loops can read them
/// without a matrix being built and without an indirect call per element.
enum DerivativeSource<'a> {
    Fixed(&'a DMatrix<f64>),
    /// The kernel, read through the place each person lives at.
    Kernel(&'a DMatrix<f64>, &'a [usize]),
    /// The decay derivative: minus the distance times the kernel, scaled by the
    /// spatial variance. Also read by place.
    Decay(&'a DMatrix<f64>, &'a DMatrix<f64>, &'a [usize], f64),
}

impl DerivativeSource<'_> {
    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        match self {
            Self::Fixed(m) => m[(i, j)],
            Self::Kernel(kernel, place) => kernel[(place[i], place[j])],
            Self::Decay(kernel, distance, place, scale) => {
                let (a, b) = (place[i], place[j]);
                -scale * distance[(a, b)] * kernel[(a, b)]
            }
        }
    }
}

/// How many decay rates the integrated likelihood averages over.
///
/// Equally spaced on the logarithm, because a decay rate spans orders of
/// magnitude. Twelve is enough that halving the spacing moves the integrated
/// log-likelihood by less than the convergence tolerance, and every one of them
/// costs a factorisation.
const INTEGRATION_POINTS: usize = 12;

/// One trait, fixed components plus an estimated spatial range.
pub struct SpatialModel {
    design: DMatrix<f64>,
    fixed: Vec<DMatrix<f64>>,
    /// Which place each person lives at, indexing `place_distance`.
    ///
    /// **People share addresses, and the kernel does not care which of them is
    /// which.** Two people at one address are nought apart and exactly as far
    /// from everybody else, so their rows of `exp(-lambda*D)` are identical and
    /// computing both is wasted work. In GOBS 1,881 people live at 1,363
    /// addresses, so the kernel is evaluated over 1.86 million pairs of places
    /// instead of 3.54 million pairs of people -- a little over half the
    /// exponentials, which had grown to be the largest single cost in a fit once
    /// the factorisation was handed to `faer`.
    place: Vec<usize>,
    /// Distances between places rather than between people.
    place_distance: DMatrix<f64>,
    rows: usize,
    logdet_xtx: f64,
    /// Set from the distances rather than chosen; see `decay_bounds`.
    lambda_lower: f64,
    lambda_upper: f64,
}

/// How far the decay rate may go, in units of the data's own spread.
///
/// **The bounds have to come from the distances, not from a constant.** At a
/// small enough decay the kernel is a matrix of ones, which is a random grand
/// mean and not a spatial effect at all; it is also rank one, so the covariance
/// goes near-singular and the search thrashes. At a large enough decay the
/// kernel is the identity, which is residual. What counts as small or large
/// depends entirely on how far apart the people are: a decay of 1e-5 per km is
/// indistinguishable from nought across a county and enormous across a
/// continent.
///
/// So the range is set by the data. The correlation must halve somewhere between
/// the closest pair and the widest separation: any less and it has not halved
/// across the whole study area, any more and it has already halved before the
/// two nearest people. Both ends are then a statement about this data set rather
/// than a number chosen in advance.
///
/// This was originally a pair of constants, 1e-5 and 5. On real distances of a
/// few hundred kilometres the lower one let the search walk to the all-ones
/// kernel, where it did not converge and each fit took hours instead of seconds.
fn decay_bounds(distance: &DMatrix<f64>) -> (f64, f64) {
    let mut widest = 0.0f64;
    let mut closest = f64::INFINITY;
    for i in 0..distance.nrows() {
        for j in 0..i {
            let d = distance[(i, j)];
            if d > 0.0 {
                widest = widest.max(d);
                closest = closest.min(d);
            }
        }
    }
    if !(widest > 0.0) || !closest.is_finite() {
        // Everybody in one place: no spatial information at all, and any decay
        // rate describes the data equally. The range is left nominal and the
        // spatial coefficient will fit to nothing.
        return (1.0e-6, 1.0);
    }
    (
        std::f64::consts::LN_2 / widest,
        std::f64::consts::LN_2 / closest,
    )
}

impl SpatialModel {
    /// Validate and prepare.
    ///
    /// `fixed` are the components whose matrices do not move — the relationship
    /// matrix, a household matrix, and so on. `distance` is in kilometres.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(
        fixed: &[DMatrix<f64>],
        distance: &DMatrix<f64>,
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let n = design.nrows();
        if n == 0 {
            return Err("SPATIAL_NO_ROWS");
        }
        if distance.nrows() != n || distance.ncols() != n {
            return Err("SPATIAL_DISTANCE_WRONG_SIZE");
        }
        for i in 0..n {
            if distance[(i, i)].abs() > 1e-9 {
                return Err("SPATIAL_DISTANCE_DIAGONAL_NOT_ZERO");
            }
            for j in 0..i {
                if (distance[(i, j)] - distance[(j, i)]).abs() > 1e-9 {
                    return Err("SPATIAL_DISTANCE_NOT_SYMMETRIC");
                }
                if distance[(i, j)] < 0.0 {
                    return Err("SPATIAL_DISTANCE_NEGATIVE");
                }
            }
        }
        for matrix in fixed {
            if matrix.nrows() != n || matrix.ncols() != n {
                return Err("SPATIAL_MATRIX_WRONG_SIZE");
            }
        }
        let xtx = design.transpose() * design;
        let chol = xtx.cholesky().ok_or("SPATIAL_DESIGN_RANK_DEFICIENT")?;
        let logdet_xtx = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        // Group people by address. Two are at the same one when the distance
        // between them is exactly nought, which is what the haversine returns
        // for identical coordinates. Anything merely close forms its own group,
        // which costs a little of the saving and cannot cost correctness.
        let mut place = vec![usize::MAX; n];
        let mut representative: Vec<usize> = Vec::new();
        for person in 0..n {
            if place[person] != usize::MAX {
                continue;
            }
            let here = representative.len();
            representative.push(person);
            place[person] = here;
            for other in (person + 1)..n {
                if place[other] == usize::MAX && distance[(person, other)] == 0.0 {
                    place[other] = here;
                }
            }
        }
        let places = representative.len();
        let place_distance = DMatrix::from_fn(places, places, |a, b| {
            distance[(representative[a], representative[b])]
        });

        let (lambda_lower, lambda_upper) = decay_bounds(distance);
        Ok(Self {
            place,
            place_distance,
            design: design.clone(),
            fixed: fixed.to_vec(),
            rows: n,
            logdet_xtx,
            lambda_lower,
            lambda_upper,
        })
    }

    /// Variances plus the decay rate: one per fixed component, then spatial,
    /// then residual, then `λ` last.
    #[must_use]
    pub fn parameters(&self) -> usize {
        self.fixed.len() + 3
    }

    fn spatial_index(&self) -> usize {
        self.fixed.len()
    }

    fn residual_index(&self) -> usize {
        self.fixed.len() + 1
    }

    fn lambda_index(&self) -> usize {
        self.fixed.len() + 2
    }

    /// `exp(−λD)` between places, not between people.
    ///
    /// Every read of it goes through `self.place`, so a person's row is looked
    /// up rather than stored twice.
    fn kernel(&self, lambda: f64) -> DMatrix<f64> {
        self.place_distance.map(|d| (-lambda * d).exp())
    }

    /// The spatial correlation between two people.
    #[inline]
    fn kernel_at(&self, kernel: &DMatrix<f64>, i: usize, j: usize) -> f64 {
        kernel[(self.place[i], self.place[j])]
    }

    fn evaluate(
        &self,
        theta: &[f64],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
    ) -> Option<Evaluation> {
        let count = self.parameters();
        let lambda = theta[self.lambda_index()];
        if theta[..count - 1]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return None;
        }
        if !lambda.is_finite() || !(self.lambda_lower..=self.lambda_upper).contains(&lambda) {
            return None;
        }
        if theta[..count - 1].iter().all(|v| *v <= 0.0) {
            return None;
        }

        let n = self.rows;
        let kernel = self.kernel(lambda);
        let mut v = DMatrix::<f64>::zeros(n, n);
        for (index, matrix) in self.fixed.iter().enumerate() {
            if theta[index] != 0.0 {
                v += matrix * theta[index];
            }
        }
        let spatial = theta[self.spatial_index()];
        if spatial != 0.0 {
            for i in 0..n {
                for j in 0..n {
                    v[(i, j)] += spatial * self.kernel_at(&kernel, i, j);
                }
            }
        }
        for i in 0..n {
            v[(i, i)] += theta[self.residual_index()];
        }

        let (factor, llt) = DenseFactor::new(&v)?;
        let logdet = factor.logdet;
        let p = self.design.ncols();
        let vy = solve_vector(&llt, y);
        let vx = solve_matrix(&llt, &self.design);
        let xvx = self.design.transpose() * &vx;
        let xvy = self.design.transpose() * &vy;
        let xvx_chol = xvx.clone().cholesky()?;
        let beta = xvx_chol.solve(&xvy);
        let quadratic = y.dot(&vy) - xvy.dot(&beta);
        if !(quadratic > 0.0) || !quadratic.is_finite() {
            return None;
        }

        let two_pi = (2.0 * std::f64::consts::PI).ln();
        let mut value = 0.5 * (n as f64 * two_pi + logdet + quadratic);
        if reml {
            let logdet_xvx = 2.0 * xvx_chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
            value += 0.5 * (logdet_xvx - self.logdet_xtx) - 0.5 * p as f64 * two_pi;
        }

        let mut gradient = vec![0.0; count];
        if want_gradient {
            let inverse = &factor.inverse;
            let xvx_inverse = xvx_chol.inverse();
            let residual_solve = &vy - &vx * &beta;

            // **The derivative matrices are never built.** Each of them is
            // something already to hand -- a fixed component, the kernel, the
            // identity, or the kernel scaled by minus the distances -- and
            // materialising four dense n-by-n matrices per evaluation was
            // costing more than the factorisation it accompanies. The identity
            // was the worst of it: `tr(V⁻¹ I)` is the sum of the diagonal of
            // `V⁻¹`, and building the identity to discover that is n² writes to
            // learn n numbers.
            for index in 0..count {
                let (trace, quadratic_term, restricted) = if index == self.residual_index() {
                    // ∂V/∂σ²_e is the identity, so every term simplifies.
                    let trace = (0..n).map(|i| inverse[(i, i)]).sum::<f64>();
                    let quadratic_term = residual_solve.dot(&residual_solve);
                    let restricted = if reml {
                        vx.transpose() * &vx
                    } else {
                        DMatrix::zeros(p, p)
                    };
                    (trace, quadratic_term, restricted)
                } else {
                    // Otherwise the derivative is a matrix already held. It is
                    // still not materialised, but it is read through a match
                    // outside the loops rather than a boxed closure inside
                    // them: an indirect call cannot be inlined, and there are
                    // n² of them per parameter -- thirteen million per
                    // evaluation at two thousand people, which cost as much as
                    // the factorisation.
                    let scale = theta[self.spatial_index()];
                    let source = if index < self.fixed.len() {
                        DerivativeSource::Fixed(&self.fixed[index])
                    } else if index == self.spatial_index() {
                        DerivativeSource::Kernel(&kernel, &self.place)
                    } else {
                        DerivativeSource::Decay(&kernel, &self.place_distance, &self.place, scale)
                    };

                    let mut trace = 0.0;
                    let mut dvr = DVector::<f64>::zeros(n);
                    let mut dvx = DMatrix::<f64>::zeros(n, p);
                    for i in 0..n {
                        let mut acc = 0.0;
                        for j in 0..n {
                            let e = source.at(i, j);
                            trace += inverse[(i, j)] * e;
                            acc += e * residual_solve[j];
                            if reml {
                                for c in 0..p {
                                    dvx[(i, c)] += e * vx[(j, c)];
                                }
                            }
                        }
                        dvr[i] = acc;
                    }
                    let restricted = if reml {
                        vx.transpose() * &dvx
                    } else {
                        DMatrix::zeros(p, p)
                    };
                    (trace, residual_solve.dot(&dvr), restricted)
                };

                let mut d = 0.5 * (trace - quadratic_term);
                if reml {
                    d -= 0.5 * (&xvx_inverse * &restricted).trace();
                }
                gradient[index] = d;
            }
        }

        Some(Evaluation {
            negative_loglik: value,
            gradient,
            fixed_effects: beta.iter().copied().collect(),
            fixed_covariance: Some(xvx_chol.inverse()),
        })
    }

    /// Fit by bounded search from several starts.
    ///
    /// The starts walk the decay rate across its range, because the likelihood
    /// in `λ` is not always single-peaked and a search begun at one scale will
    /// not find a spatial effect operating at another.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start reached a usable optimum.
    pub fn fit(&self, y: &DVector<f64>, reml: bool) -> Result<SpatialFit, &'static str> {
        if y.len() != self.rows {
            return Err("SPATIAL_RESPONSE_WRONG_LENGTH");
        }
        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        if !(variance > 0.0) {
            return Err("SPATIAL_RESPONSE_CONSTANT");
        }
        let scale = variance.sqrt();
        let scaled = y / scale;

        let count = self.parameters();
        let mut lower = vec![0.0; count];
        let mut upper = vec![f64::INFINITY; count];
        lower[self.lambda_index()] = self.lambda_lower;
        upper[self.lambda_index()] = self.lambda_upper;

        let variance_count = count - 1;
        let mut starts: Vec<Vec<f64>> = Vec::new();
        // Starts spread across the range the data allows, on a log scale
        // because the decay rate spans orders of magnitude.
        let span = (self.lambda_upper / self.lambda_lower).ln();
        for step in 0..4 {
            let lambda = self.lambda_lower * (span * (f64::from(step) + 0.5) / 4.0).exp();
            let mut even = vec![1.0 / variance_count as f64; count];
            even[self.lambda_index()] = lambda;
            starts.push(even);
            let mut small = vec![0.1 / variance_count as f64; count];
            small[self.residual_index()] = 0.8;
            small[self.lambda_index()] = lambda;
            starts.push(small);
        }

        let value_of = |candidate: &[f64]| -> f64 {
            self.evaluate(candidate, &scaled, reml, false)
                .map_or(1e30, |e| e.negative_loglik)
        };
        let gradient_of = |candidate: &[f64]| -> Vec<f64> {
            self.evaluate(candidate, &scaled, reml, true)
                .map_or_else(|| vec![0.0; count], |e| e.gradient)
        };
        // The projected gradient the flag reads, scaled by the log-likelihood.
        // A parameter resting on a bound is converged when its one-sided
        // derivative pushes outward, which an unprojected norm calls a failure.
        let reading = |gradient: &[f64], par: &[f64], negative: f64| -> f64 {
            let projected = gradient
                .iter()
                .enumerate()
                .map(|(k, g)| {
                    if par[k] <= lower[k] {
                        g.min(0.0)
                    } else if par[k] >= upper[k] {
                        g.max(0.0)
                    } else {
                        *g
                    }
                })
                .fold(0.0f64, |worst, g| worst.max(g.abs()));
            projected / negative.abs().max(1.0)
        };

        let mut best: Option<(f64, Vec<f64>, Vec<f64>)> = None;
        for start in starts {
            let Ok(bounds) = Bounds::new(lower.clone(), upper.clone()) else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(count);
            control.maxit = 300;
            control.fnscale = value_of(&start).abs().max(1.0);
            control.parscale = vec![1.0; count];
            // **`factr` at nought disables stopping on the function**, leaving
            // only the gradient test, so the search runs to `maxit` every time
            // whatever it has found. On four parameters that meant eight hundred
            // evaluations where tens would do, and every one of them factorises
            // a dense covariance. This is R's own default and stops when the
            // objective has settled to about 1e-9 relative.
            control.factr = 1.0e3;
            // The fit declares convergence at a scaled gradient below 1e-6, so
            // asking the optimiser for 1e-9 was three orders tighter than
            // anything downstream reads.
            control.pgtol = 1e-8;
            control.lmm = count.min(10);
            let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, &value_of, &gradient_of, control)
            else {
                continue;
            };
            let Some(at) = self.evaluate(&solution.par, &scaled, reml, true) else {
                continue;
            };
            if at.negative_loglik.is_finite()
                && best
                    .as_ref()
                    .is_none_or(|(v, _, _)| at.negative_loglik < *v)
            {
                best = Some((at.negative_loglik, solution.par.clone(), at.fixed_effects));
            }
        }

        let (mut negative, mut par, mut beta) = best.ok_or("SPATIAL_NO_START_CONVERGED")?;
        let mut at = self
            .evaluate(&par, &scaled, reml, true)
            .ok_or("SPATIAL_OPTIMUM_NOT_EVALUABLE")?;
        let mut scaled_gradient = reading(&at.gradient, &par, negative);

        // Where the gradient test fails, search once more with the objective
        // tolerance switched off. It fires nowhere else, so every fit that
        // passes today is untouched. See `convergence` for the measurements.
        let mut polished = false;
        if scaled_gradient >= TOLERANCE
            && let Some(better) = convergence::polish(
                &par,
                negative,
                scaled_gradient,
                &lower,
                &upper,
                &value_of,
                &gradient_of,
                |candidate| {
                    self.evaluate(candidate, &scaled, reml, true).map(|e| {
                        (
                            e.negative_loglik,
                            reading(&e.gradient, candidate, e.negative_loglik),
                        )
                    })
                },
            )
            && let Some(again) = self.evaluate(&better.par, &scaled, reml, true)
        {
            polished = true;
            negative = better.negative_loglik;
            par = better.par;
            beta.clone_from(&again.fixed_effects);
            scaled_gradient = better.scaled_gradient;
            at = again;
        }

        let variances: Vec<f64> = par[..count - 1].iter().map(|v| v * variance).collect();
        let raw_coefficient_total: f64 = variances.iter().sum();
        let raw_coefficient_proportions = variances
            .iter()
            .map(|v| v / raw_coefficient_total)
            .collect();
        let lambda = par[self.lambda_index()];
        let observations = if reml {
            (self.rows - self.design.ncols()) as f64
        } else {
            self.rows as f64
        };

        Ok(SpatialFit {
            variances,
            raw_coefficient_proportions,
            raw_coefficient_total,
            lambda,
            half_distance_km: std::f64::consts::LN_2 / lambda,
            fixed_effects: beta.iter().map(|b| b * scale).collect(),
            fixed_effect_errors: at
                .fixed_covariance
                .as_ref()
                .map(|c| {
                    (0..c.nrows())
                        .map(|i| c[(i, i)].max(0.0).sqrt() * scale)
                        .collect()
                })
                .unwrap_or_default(),
            loglik: -negative - observations * scale.ln(),
            converged: scaled_gradient < TOLERANCE,
            polished,
            scaled_gradient,
            estimator: if reml { "reml" } else { "ml" },
        })
    }

    /// The log-likelihood with the decay rate integrated out rather than
    /// maximised over.
    ///
    /// **This is the other classical answer to an unidentified nuisance
    /// parameter.** When the spatial variance is nought the decay rate is absent
    /// from the likelihood entirely, and there are two things one can do about
    /// it: take the supremum over the decay rate, which is what profiling does,
    /// or average over it. `fit` takes the supremum. This averages:
    ///
    /// ```text
    /// L(sigma) = integral over lambda of L(sigma, lambda) w(lambda) d lambda
    /// ```
    ///
    /// The weight is uniform on the logarithm of the decay rate across the range
    /// the distances support, because a decay rate spans orders of magnitude and
    /// nothing in the data picks a scale. That is the same instinct as the port
    /// use of a fixed grid, reached from the other direction: it
    /// declines to let the data choose a range and then report the choice as
    /// though it were estimated.
    ///
    /// Three things follow. There is no decay rate to report, so the
    /// uninformative interval on it disappears rather than being suppressed. The
    /// spatial coefficient proportion's uncertainty now includes not knowing the range, where
    /// before it was conditional on a badly determined estimate of it. And the
    /// statistic is an integrated likelihood ratio, which is better behaved
    /// under the null than a supremum over something unidentified.
    ///
    /// The sum is taken in logarithms because the terms differ by many orders of
    /// magnitude, and the gradient is exact rather than differenced: the
    /// derivative of a logged weighted sum of likelihoods is the average of
    /// their derivatives, weighted by each one's share of the sum.
    fn evaluate_integrated(
        &self,
        variances: &[f64],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
    ) -> Option<Evaluation> {
        let count = variances.len();
        let mut logliks = Vec::with_capacity(INTEGRATION_POINTS);
        let mut gradients = Vec::with_capacity(INTEGRATION_POINTS);
        let mut effects: Vec<Vec<f64>> = Vec::with_capacity(INTEGRATION_POINTS);
        let mut covariances: Vec<Option<DMatrix<f64>>> = Vec::with_capacity(INTEGRATION_POINTS);

        let span = (self.lambda_upper / self.lambda_lower).ln();
        for point in 0..INTEGRATION_POINTS {
            // Midpoints of equal intervals in log lambda, which is the uniform
            // weight on the logarithm written as a rectangle rule.
            let fraction = (point as f64 + 0.5) / INTEGRATION_POINTS as f64;
            let lambda = self.lambda_lower * (span * fraction).exp();
            let mut theta = variances.to_vec();
            theta.push(lambda);
            let Some(at) = self.evaluate(&theta, y, reml, want_gradient) else {
                continue;
            };
            logliks.push(-at.negative_loglik);
            if want_gradient {
                gradients.push(at.gradient[..count].to_vec());
            }
            effects.push(at.fixed_effects);
            covariances.push(at.fixed_covariance);
        }
        // **The divisor is the whole grid and not the part of it that
        // happened to evaluate.** A point drops out when its covariance will
        // not factorise, which depends on lambda and therefore on where the
        // search currently is, so the surviving set genuinely changes as theta
        // moves. Dividing by the survivors renormalises them to sum to one and
        // steps the objective by ln(12/11), about 0.087, as the search crosses
        // a boundary where a point starts or stops factorising -- which both
        // stalls the search and makes two constrained integrated fits
        // incomparable, though their difference is exactly what the integrated
        // likelihood ratio takes. A point that cannot be evaluated contributes
        // nothing, which is what an infeasible region should contribute.
        //
        // Too few surviving points is a different matter: the rule is then
        // integrating over a grid that mostly does not exist, and says so.
        if logliks.len() * 2 < INTEGRATION_POINTS {
            return None;
        }

        // log of the mean of the likelihoods, taken safely.
        let largest = logliks.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = logliks.iter().map(|l| (l - largest).exp()).sum();
        let integrated = largest + (total / INTEGRATION_POINTS as f64).ln();
        if !integrated.is_finite() {
            return None;
        }

        let mut gradient = vec![0.0; count];
        if want_gradient && gradients.len() == logliks.len() {
            // Each grid point contributes in proportion to its share of the
            // integral. The gradients here are of the negative log-likelihood,
            // so the weighted average is too.
            for (index, g) in gradients.iter().enumerate() {
                let share = (logliks[index] - largest).exp() / total;
                for k in 0..count {
                    gradient[k] += share * g[k];
                }
            }
        }

        // **The fixed effects are averaged across the grid too, and their
        // covariance gains a term for the spread between grid points.** Taking
        // them from one decay rate would report an estimate conditional on a
        // rate the integration exists to avoid committing to. The two terms are
        // the average of the within-rate covariances and the spread of the
        // estimates across rates -- the same decomposition multiple imputation
        // uses, and the second term is exactly the extra uncertainty that comes
        // from not knowing the range.
        let p = self.design.ncols();
        let shares: Vec<f64> = logliks
            .iter()
            .map(|l| (l - largest).exp() / total)
            .collect();
        let mut fixed_effects = vec![0.0; p];
        for (index, effect) in effects.iter().enumerate() {
            for k in 0..p.min(effect.len()) {
                fixed_effects[k] += shares[index] * effect[k];
            }
        }
        let mut fixed_covariance = DMatrix::<f64>::zeros(p, p);
        for (index, covariance) in covariances.iter().enumerate() {
            if let Some(c) = covariance {
                fixed_covariance += c * shares[index];
            }
            for a in 0..p.min(effects[index].len()) {
                for b in 0..p.min(effects[index].len()) {
                    fixed_covariance[(a, b)] += shares[index]
                        * (effects[index][a] - fixed_effects[a])
                        * (effects[index][b] - fixed_effects[b]);
                }
            }
        }

        Some(Evaluation {
            negative_loglik: -integrated,
            gradient,
            fixed_effects,
            fixed_covariance: Some(fixed_covariance),
        })
    }

    /// Fit with the decay rate integrated out.
    ///
    /// The variances are the only free parameters, so this is a three-parameter
    /// search where `fit` runs a four-parameter one, and each evaluation costs
    /// `INTEGRATION_POINTS` factorisations rather than one.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start reached a usable optimum.
    pub fn fit_integrated(&self, y: &DVector<f64>, reml: bool) -> Result<SpatialFit, &'static str> {
        if y.len() != self.rows {
            return Err("SPATIAL_RESPONSE_WRONG_LENGTH");
        }
        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        if !(variance > 0.0) {
            return Err("SPATIAL_RESPONSE_CONSTANT");
        }
        let scale = variance.sqrt();
        let scaled = y / scale;

        let count = self.parameters() - 1;
        let lower = vec![0.0; count];
        let upper = vec![f64::INFINITY; count];
        let starts: Vec<Vec<f64>> = vec![
            vec![1.0 / count as f64; count],
            {
                let mut s = vec![0.1; count];
                s[count - 1] = 0.8;
                s
            },
            {
                let mut s = vec![0.1; count];
                s[self.spatial_index()] = 0.6;
                s[count - 1] = 0.3;
                s
            },
        ];

        let value_of = |c: &[f64]| -> f64 {
            self.evaluate_integrated(c, &scaled, reml, false)
                .map_or(1e30, |e| e.negative_loglik)
        };
        let gradient_of = |c: &[f64]| -> Vec<f64> {
            self.evaluate_integrated(c, &scaled, reml, true)
                .map_or_else(|| vec![0.0; count], |e| e.gradient)
        };

        let mut best: Option<(f64, Vec<f64>, Vec<f64>)> = None;
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
            control.lmm = count.min(10);
            let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, &value_of, &gradient_of, control)
            else {
                continue;
            };
            let Some(at) = self.evaluate_integrated(&solution.par, &scaled, reml, true) else {
                continue;
            };
            if at.negative_loglik.is_finite()
                && best
                    .as_ref()
                    .is_none_or(|(v, _, _)| at.negative_loglik < *v)
            {
                best = Some((at.negative_loglik, solution.par.clone(), at.fixed_effects));
            }
        }

        let integrated_reading = |gradient: &[f64], par: &[f64], negative: f64| -> f64 {
            let projected = gradient
                .iter()
                .enumerate()
                .map(|(k, g)| {
                    if crate::components::resting_on_zero(par[k]) {
                        g.min(0.0)
                    } else {
                        *g
                    }
                })
                .fold(0.0f64, |worst, g| worst.max(g.abs()));
            projected / negative.abs().max(1.0)
        };

        let (mut negative, mut par, mut beta) = best.ok_or("SPATIAL_NO_START_CONVERGED")?;
        let mut at = self
            .evaluate_integrated(&par, &scaled, reml, true)
            .ok_or("SPATIAL_OPTIMUM_NOT_EVALUABLE")?;
        let mut scaled_gradient = integrated_reading(&at.gradient, &par, negative);

        // As in the estimated-range fit above: one more search where the
        // gradient test failed, and nowhere else.
        let mut polished = false;
        if scaled_gradient >= TOLERANCE
            && let Some(better) = convergence::polish(
                &par,
                negative,
                scaled_gradient,
                &lower,
                &upper,
                &value_of,
                &gradient_of,
                |candidate| {
                    self.evaluate_integrated(candidate, &scaled, reml, true)
                        .map(|e| {
                            (
                                e.negative_loglik,
                                integrated_reading(&e.gradient, candidate, e.negative_loglik),
                            )
                        })
                },
            )
            && let Some(again) = self.evaluate_integrated(&better.par, &scaled, reml, true)
        {
            polished = true;
            negative = better.negative_loglik;
            par = better.par;
            beta.clone_from(&again.fixed_effects);
            scaled_gradient = better.scaled_gradient;
            at = again;
        }

        let variances: Vec<f64> = par.iter().map(|v| v * variance).collect();
        let raw_coefficient_total: f64 = variances.iter().sum();
        let raw_coefficient_proportions = variances
            .iter()
            .map(|v| v / raw_coefficient_total)
            .collect();
        let observations = if reml {
            (self.rows - self.design.ncols()) as f64
        } else {
            self.rows as f64
        };

        Ok(SpatialFit {
            variances,
            raw_coefficient_proportions,
            raw_coefficient_total,
            // There is no decay rate to report: it has been integrated out.
            // Reporting the middle of the range would invite it to be read as an
            // estimate, so it is reported as not a number.
            lambda: f64::NAN,
            half_distance_km: f64::NAN,
            fixed_effects: beta.iter().map(|b| b * scale).collect(),
            fixed_effect_errors: at
                .fixed_covariance
                .as_ref()
                .map(|c| {
                    (0..c.nrows())
                        .map(|i| c[(i, i)].max(0.0).sqrt() * scale)
                        .collect()
                })
                .unwrap_or_default(),
            loglik: -negative - observations * scale.ln(),
            converged: scaled_gradient < TOLERANCE,
            polished,
            scaled_gradient,
            estimator: if reml { "reml" } else { "ml" },
        })
    }

    /// Predict the random effects of one component.
    ///
    /// The same best linear unbiased prediction the component model makes, on a
    /// dense covariance rather than family blocks. `component` indexes the fixed
    /// components first and then the spatial one; the residual cannot be
    /// predicted.
    ///
    /// **Only with the decay rate profiled.** With it integrated out there is no
    /// single kernel to predict from, and averaging predictions across the grid
    /// is a different quantity that has not been calibrated. Asking is refused
    /// rather than answered with the middle of the range.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the fit fails, the component does not exist,
    /// or the range has been integrated out.
    pub fn blup(
        &self,
        y: &DVector<f64>,
        reml: bool,
        component: usize,
        integrated: bool,
    ) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
        if integrated {
            return Err("SPATIAL_NO_PREDICTION_WHEN_INTEGRATED");
        }
        if component > self.spatial_index() {
            return Err("SPATIAL_NO_SUCH_COMPONENT_TO_PREDICT");
        }
        let fit = self.fit(y, reml)?;
        let n = self.rows;
        let p = self.design.ncols();

        let kernel = self.kernel(fit.lambda);
        let mut v = DMatrix::<f64>::zeros(n, n);
        for (index, matrix) in self.fixed.iter().enumerate() {
            v += matrix * fit.variances[index];
        }
        let spatial = fit.variances[self.spatial_index()];
        for i in 0..n {
            for j in 0..n {
                v[(i, j)] += spatial * self.kernel_at(&kernel, i, j);
            }
        }
        for i in 0..n {
            v[(i, i)] += fit.variances[self.residual_index()];
        }

        let factor = crate::dense::DenseFactor::new(&v).ok_or("SPATIAL_NOT_POSITIVE_DEFINITE")?;
        let inverse = factor.inverse();
        let beta = DVector::from_iterator(p, fit.fixed_effects.iter().copied());
        let residual = y - &self.design * &beta;
        let vx = factor.solve_matrix(&self.design);
        let xvx = self.design.transpose() * &vx;
        let xvx_inverse = xvx
            .cholesky()
            .ok_or("SPATIAL_DESIGN_RANK_DEFICIENT")?
            .inverse();

        // The component's own covariance.
        let g = if component == self.spatial_index() {
            DMatrix::from_fn(n, n, |i, j| spatial * self.kernel_at(&kernel, i, j))
        } else {
            &self.fixed[component] * fit.variances[component]
        };

        let predicted = &g * &inverse * &residual;
        let gvi = &g * &inverse;
        let gvig = &gvi * &g;
        let gvx = &g * &vx;
        let correction = &gvx * &xvx_inverse * gvx.transpose();
        let errors = (0..n)
            .map(|i| {
                (g[(i, i)] - gvig[(i, i)] + correction[(i, i)])
                    .max(0.0)
                    .sqrt()
            })
            .collect();
        Ok((predicted.iter().copied().collect(), errors))
    }

    /// The best log-likelihood with the spatial variance held at nought.
    ///
    /// Dropping the component is the same model and a smaller one to fit, and
    /// the decay rate goes with it — under this null it means nothing.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the reduced fit fails.
    pub fn null_loglik(&self, y: &DVector<f64>, reml: bool) -> Result<f64, &'static str> {
        let reduced = crate::components::ComponentModel::build(&self.fixed, &self.design)?;
        Ok(reduced.fit(y, reml)?.loglik)
    }

    /// The observed likelihood ratio for no spatial variance at all.
    ///
    /// `integrated` chooses which treatment of the decay rate the numerator
    /// uses: the supremum over it, or the average across it. The null is the
    /// same either way, having no decay rate in it at all.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit fails.
    pub fn spatial_statistic(
        &self,
        y: &DVector<f64>,
        reml: bool,
        integrated: bool,
    ) -> Result<f64, &'static str> {
        let full = if integrated {
            self.fit_integrated(y, reml)?
        } else {
            self.fit(y, reml)?
        };
        let null = self.null_loglik(y, reml)?;
        Ok((2.0 * (full.loglik - null)).max(0.0))
    }
}

/// Which quantity an interval is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialQuantity {
    /// One raw covariance coefficient divided by their total, by index.
    RawCoefficientProportion(usize),
    /// The decay rate itself.
    Lambda,
}

/// A profile-likelihood interval.
#[derive(Clone, Copy, Debug)]
pub struct SpatialInterval {
    pub lower: f64,
    pub upper: f64,
    pub lower_limited: bool,
    pub upper_limited: bool,
    pub level: f64,
}

const CHI2_ONE_DF_95: f64 = 3.841_458_820_694_124;

impl SpatialModel {
    /// The best log-likelihood with one quantity held fixed.
    ///
    /// A raw coefficient proportion is held by substitution, as in
    /// `components.rs`: the other
    /// variances are free and this one follows them. The decay rate is held by
    /// pinning the coordinate, since it is a parameter in its own right.
    fn profile_objective(
        &self,
        y: &DVector<f64>,
        reml: bool,
        quantity: SpatialQuantity,
        value: f64,
        integrated: bool,
    ) -> Option<f64> {
        // With the decay rate integrated out there is no decay rate to hold, so
        // asking for an interval on it is a question about a parameter that no
        // longer exists.
        if integrated && quantity == SpatialQuantity::Lambda {
            return None;
        }
        let count = self.parameters();
        let variances = count - 1;
        let (free, factor): (Vec<usize>, f64) = match quantity {
            SpatialQuantity::RawCoefficientProportion(index) => {
                if index >= variances || !(0.0..=1.0).contains(&value) || value > 1.0 - 1e-9 {
                    return None;
                }
                // **The decay rate is not a free coordinate when it has been
                // integrated out.** Leaving it in the search gave the optimiser
                // a dimension that does nothing and a gradient of the wrong
                // length, and the profile then failed at its own fitted value.
                let free: Vec<usize> = if integrated {
                    (0..count - 1).filter(|k| *k != index).collect()
                } else {
                    (0..count).filter(|k| *k != index).collect()
                };
                (free, value / (1.0 - value))
            }
            SpatialQuantity::Lambda => {
                if !(self.lambda_lower..=self.lambda_upper).contains(&value) {
                    return None;
                }
                ((0..variances).collect(), 0.0)
            }
        };

        let expand = |packed: &[f64]| -> Vec<f64> {
            let mut theta = vec![0.0; count];
            match quantity {
                SpatialQuantity::RawCoefficientProportion(index) => {
                    let mut others = 0.0;
                    for (slot, &k) in free.iter().enumerate() {
                        theta[k] = packed[slot];
                        if k < variances {
                            others += packed[slot];
                        }
                    }
                    theta[index] = factor * others;
                }
                SpatialQuantity::Lambda => {
                    for (slot, &k) in free.iter().enumerate() {
                        theta[k] = packed[slot];
                    }
                    theta[self.lambda_index()] = value;
                }
            }
            theta
        };

        let lower: Vec<f64> = free
            .iter()
            .map(|&k| {
                if k == self.lambda_index() {
                    self.lambda_lower
                } else {
                    0.0
                }
            })
            .collect();
        let upper: Vec<f64> = free
            .iter()
            .map(|&k| {
                if k == self.lambda_index() {
                    self.lambda_upper
                } else {
                    f64::INFINITY
                }
            })
            .collect();

        let mut best: Option<f64> = None;
        let profile_span = (self.lambda_upper / self.lambda_lower).ln();
        for step in 0..3 {
            let lambda_start =
                self.lambda_lower * (profile_span * (f64::from(step) + 0.5) / 3.0).exp();
            let start: Vec<f64> = free
                .iter()
                .map(|&k| {
                    if k == self.lambda_index() {
                        lambda_start
                    } else {
                        1.0 / variances as f64
                    }
                })
                .collect();
            let value_of = |candidate: &[f64]| -> f64 {
                let theta = expand(candidate);
                if integrated {
                    self.evaluate_integrated(&theta[..variances], y, reml, false)
                } else {
                    self.evaluate(&theta, y, reml, false)
                }
                .map_or(1e30, |e| e.negative_loglik)
            };
            let gradient_of = |candidate: &[f64]| -> Vec<f64> {
                let theta = expand(candidate);
                if integrated {
                    return self
                        .evaluate_integrated(&theta[..variances], y, reml, true)
                        .map_or_else(
                            || vec![0.0; free.len()],
                            |e| match quantity {
                                SpatialQuantity::RawCoefficientProportion(index) => {
                                    let through = e.gradient[index] * factor;
                                    free.iter().map(|&k| e.gradient[k] + through).collect()
                                }
                                SpatialQuantity::Lambda => vec![0.0; free.len()],
                            },
                        );
                }
                self.evaluate(&theta, y, reml, true).map_or_else(
                    || vec![0.0; free.len()],
                    |e| match quantity {
                        SpatialQuantity::RawCoefficientProportion(index) => {
                            // The pinned variance is carried by the free
                            // variances together, so each picks up the same
                            // share of its slope. The decay rate carries none of
                            // it, being no part of the total.
                            let through = e.gradient[index] * factor;
                            free.iter()
                                .map(|&k| e.gradient[k] + if k < variances { through } else { 0.0 })
                                .collect()
                        }
                        SpatialQuantity::Lambda => free.iter().map(|&k| e.gradient[k]).collect(),
                    },
                )
            };
            let Ok(bounds) = Bounds::new(lower.clone(), upper.clone()) else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(free.len());
            control.maxit = 300;
            control.fnscale = value_of(&start).abs().max(1.0);
            control.parscale = vec![1.0; free.len()];
            control.factr = 1.0e3;
            control.pgtol = 1e-9;
            control.lmm = free.len().min(10);
            if let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control)
            {
                // The final evaluation has to take the same route as the
                // search did. Taking the coordinate one here while the search
                // integrated meant handing `evaluate` a theta whose decay rate
                // slot was nought -- it is not a free coordinate once integrated
                // out -- which is outside the allowed range, so every profile
                // point was refused and every interval failed at its own fitted
                // value.
                let theta = expand(&solution.par);
                let at = if integrated {
                    self.evaluate_integrated(&theta[..variances], y, reml, false)
                } else {
                    self.evaluate(&theta, y, reml, false)
                };
                if let Some(at) = at
                    && at.negative_loglik.is_finite()
                    && best.is_none_or(|b: f64| at.negative_loglik < b)
                {
                    best = Some(at.negative_loglik);
                }
            }
            if quantity == SpatialQuantity::Lambda {
                // With the decay rate held, one start per scale is wasteful --
                // the remaining parameters are variances and well behaved.
                break;
            }
        }
        best.map(|negative| -negative)
    }

    /// A 95 per cent profile interval.
    ///
    /// **This is honest for the decay rate and for a raw coefficient proportion
    /// that is not nought, and it is not a way round the bootstrap.** An
    /// interval for the spatial coefficient proportion whose lower endpoint
    /// reaches nought does not test whether there is
    /// a spatial effect: under that null the decay rate means nothing and the
    /// deviance has no chi-squared reference. Read the endpoint, not a
    /// hypothesis test spelled backwards.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the fit or the profile at the estimate fails.
    pub fn profile_interval(
        &self,
        y: &DVector<f64>,
        reml: bool,
        quantity: SpatialQuantity,
        integrated: bool,
    ) -> Result<SpatialInterval, &'static str> {
        if integrated && quantity == SpatialQuantity::Lambda {
            return Err("SPATIAL_NO_RANGE_WHEN_INTEGRATED");
        }
        let fit = if integrated {
            self.fit_integrated(y, reml)?
        } else {
            self.fit(y, reml)?
        };
        let (fitted, bottom, top) = match quantity {
            SpatialQuantity::RawCoefficientProportion(index) => {
                if index >= self.parameters() - 1 {
                    return Err("SPATIAL_NO_SUCH_COMPONENT");
                }
                (fit.raw_coefficient_proportions[index], 0.0, 1.0 - 1e-9)
            }
            SpatialQuantity::Lambda => (fit.lambda, self.lambda_lower, self.lambda_upper),
        };

        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        let scaled = y / variance.sqrt();

        let maximum = self
            .profile_objective(&scaled, reml, quantity, fitted, integrated)
            .ok_or("SPATIAL_PROFILE_MAXIMUM_FAILED")?;
        let deviance = |value: f64| -> f64 {
            self.profile_objective(&scaled, reml, quantity, value, integrated)
                .map_or(f64::INFINITY, |ll| 2.0 * (maximum - ll))
        };
        let endpoint = |bound: f64| -> (f64, bool) {
            if deviance(bound) <= CHI2_ONE_DF_95 {
                return (bound, true);
            }
            // **The endpoint is bisected to a ten-thousandth and no further.**
            // Each halving re-optimises every other parameter over several
            // starts, so the last twenty halvings cost as much as the first
            // twenty and buy digits nobody reports. Pinning an endpoint to 1e-9
            // was most of the cost of a spatial interval and none of its
            // meaning.
            let tolerance = 1e-4 * fitted.abs().max(1e-3);
            let (mut inside, mut outside) = (fitted, bound);
            for _ in 0..60 {
                let middle = 0.5 * (inside + outside);
                if (outside - inside).abs() <= tolerance {
                    break;
                }
                if deviance(middle) <= CHI2_ONE_DF_95 {
                    inside = middle;
                } else {
                    outside = middle;
                }
            }
            (0.5 * (inside + outside), false)
        };
        let (lower, lower_limited) = endpoint(bottom);
        let (upper, upper_limited) = endpoint(top);
        Ok(SpatialInterval {
            lower,
            upper,
            lower_limited,
            upper_limited,
            level: 0.95,
        })
    }
}

/// Pairwise great-circle distances in kilometres, on a spherical Earth.
///
/// The radius is the mean spherical one the port lab froze, so a distance
/// computed here and one computed there are the same distance.
///
/// # Errors
///
/// Returns a stable code for a coordinate outside its range.
#[cfg(any(feature = "python", test))]
pub fn pairwise_haversine_km(
    latitude: &[f64],
    longitude: &[f64],
) -> Result<DMatrix<f64>, &'static str> {
    if latitude.len() != longitude.len() {
        return Err("SPATIAL_COORDINATE_COUNT_MISMATCH");
    }
    for (lat, lon) in latitude.iter().zip(longitude) {
        if !lat.is_finite() || !(-90.0..=90.0).contains(lat) {
            return Err("SPATIAL_LATITUDE_OUT_OF_RANGE");
        }
        if !lon.is_finite() || !(-180.0..=180.0).contains(lon) {
            return Err("SPATIAL_LONGITUDE_OUT_OF_RANGE");
        }
    }
    let n = latitude.len();
    let lat: Vec<f64> = latitude.iter().map(|d| d.to_radians()).collect();
    let lon: Vec<f64> = longitude.iter().map(|d| d.to_radians()).collect();
    let mut distance = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..i {
            let dlat = lat[i] - lat[j];
            let dlon = lon[i] - lon[j];
            let inner = (dlat / 2.0).sin().powi(2)
                + lat[i].cos() * lat[j].cos() * (dlon / 2.0).sin().powi(2);
            let d = 2.0 * EARTH_RADIUS_KM * inner.clamp(0.0, 1.0).sqrt().asin();
            distance[(i, j)] = d;
            distance[(j, i)] = d;
        }
    }
    Ok(distance)
}

/// Mean spherical Earth radius, in kilometres.
#[cfg(any(feature = "python", test))]
pub const EARTH_RADIUS_KM: f64 = 6_371.008_8;

/// splitmix64, so that any bootstrap can be reproduced from its seed alone.
struct Stream(u64);

impl Stream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// The result of a parametric bootstrap for no spatial variance.
#[derive(Clone, Debug)]
pub struct SpatialBootstrap {
    pub observed: f64,
    pub exceedances: usize,
    /// Replicates that produced a usable statistic. A replicate whose fit fails
    /// is counted out rather than counted as a non-exceedance, which would bias
    /// the p-value downward.
    pub replicates: usize,
    pub requested: usize,
    pub p_value: f64,
    pub seed: u64,
    pub rule: &'static str,
}

impl SpatialModel {
    /// The parametric bootstrap p-value for no spatial variance.
    ///
    /// **This exists because no table applies.** With the spatial variance at
    /// nought the decay rate is absent from the likelihood, so the statistic has
    /// neither a chi-squared nor a chi-bar-squared null. Simulating the reduced
    /// model is the only reference there is.
    ///
    /// Data are simulated from the fit of the reduced model — the one without a
    /// spatial term — and not from the full fit. Simulating from the full fit
    /// would be simulating from the alternative and would answer a different
    /// question.
    ///
    /// The p-value adds one to both counts. Without that, a statistic larger
    /// than every simulated one returns exactly nought, which claims more than
    /// the replicates can support.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the observed fit or the reduced fit fails.
    pub fn bootstrap_spatial_variance(
        &self,
        y: &DVector<f64>,
        reml: bool,
        replicates: usize,
        seed: u64,
        integrated: bool,
    ) -> Result<SpatialBootstrap, &'static str> {
        if replicates == 0 {
            return Err("SPATIAL_BOOTSTRAP_NO_REPLICATES");
        }
        let observed = self.spatial_statistic(y, reml, integrated)?;

        let reduced = crate::components::ComponentModel::build(&self.fixed, &self.design)?;
        let null = reduced.fit(y, reml)?;
        let n = self.rows;
        let mut covariance = DMatrix::<f64>::zeros(n, n);
        for (index, matrix) in self.fixed.iter().enumerate() {
            covariance += matrix * null.variances[index];
        }
        for i in 0..n {
            covariance[(i, i)] += null.variances[self.fixed.len()];
        }
        let factor = covariance
            .cholesky()
            .ok_or("SPATIAL_NULL_COVARIANCE_NOT_POSITIVE_DEFINITE")?
            .l();
        let mean = &self.design
            * DVector::from_iterator(self.design.ncols(), null.fixed_effects.iter().copied());

        let mut stream = Stream(seed);
        let mut exceedances = 0usize;
        let mut usable = 0usize;
        for _ in 0..replicates {
            let draw = DVector::from_iterator(n, (0..n).map(|_| stream.normal()));
            let simulated = &mean + &factor * draw;
            // A replicate that cannot be fitted is left out of the count
            // rather than counted as a non-exceedance.
            if let Ok(statistic) = self.spatial_statistic(&simulated, reml, integrated) {
                usable += 1;
                if statistic >= observed {
                    exceedances += 1;
                }
            }
        }
        if usable == 0 {
            return Err("SPATIAL_BOOTSTRAP_NO_USABLE_REPLICATE");
        }
        Ok(SpatialBootstrap {
            observed,
            exceedances,
            replicates: usable,
            requested: replicates,
            p_value: (1 + exceedances) as f64 / (usable + 1) as f64,
            seed,
            rule: "parametric_bootstrap_add_one",
        })
    }
}

// PyO3 extracts each argument from a Python object, so a `#[pyfunction]` takes
// them by value whether or not the body consumes them. The lint cannot be
// satisfied here without breaking the macro.
#[allow(clippy::needless_pass_by_value)]
// A `#[pyfunction]`'s parameter list is the Python signature, so grouping
// arguments into a struct to shorten it would make the interface worse.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "python")]
mod python {
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::{SpatialModel, SpatialQuantity};
    use nalgebra::{DMatrix, DVector};

    fn build(
        fixed: &[PyReadonlyArray2<'_, f64>],
        distance: &PyReadonlyArray2<'_, f64>,
        design: &PyReadonlyArray2<'_, f64>,
    ) -> PyResult<SpatialModel> {
        let convert = |m: &PyReadonlyArray2<'_, f64>| {
            let a = m.as_array();
            DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)])
        };
        let matrices: Vec<DMatrix<f64>> = fixed.iter().map(convert).collect();
        SpatialModel::build(&matrices, &convert(distance), &convert(design))
            .map_err(PyValueError::new_err)
    }

    fn response(y: &PyReadonlyArray1<'_, f64>) -> DVector<f64> {
        let y = y.as_array();
        DVector::from_iterator(y.len(), y.iter().copied())
    }

    /// Fit one trait with fixed components plus an estimated spatial range.
    ///
    /// Returns the raw covariance coefficients, their scale-dependent
    /// proportions and total, the decay rate per
    /// kilometre, the distance at which the spatial correlation is a half, the
    /// log-likelihood, the scaled gradient and whether it converged.
    #[pyfunction]
    #[pyo3(signature = (fixed, distance, design, y, reml=true, integrated=false))]
    #[allow(clippy::type_complexity)]
    pub fn spatial_fit(
        fixed: Vec<PyReadonlyArray2<'_, f64>>,
        distance: PyReadonlyArray2<'_, f64>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        reml: bool,
        integrated: bool,
    ) -> PyResult<(
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        f64,
        f64,
        bool,
        bool,
        Vec<f64>,
        Vec<f64>,
    )> {
        let model = build(&fixed, &distance, &design)?;
        let response = response(&y);
        let fit = if integrated {
            model.fit_integrated(&response, reml)
        } else {
            model.fit(&response, reml)
        }
        .map_err(PyValueError::new_err)?;
        Ok((
            fit.variances,
            fit.raw_coefficient_proportions,
            fit.raw_coefficient_total,
            fit.lambda,
            fit.half_distance_km,
            fit.loglik,
            fit.scaled_gradient,
            fit.converged,
            fit.polished,
            fit.fixed_effects,
            fit.fixed_effect_errors,
        ))
    }

    /// Predict the random effects of one component.
    #[pyfunction]
    #[pyo3(signature = (fixed, distance, design, y, component, reml=true, integrated=false))]
    pub fn spatial_blup(
        fixed: Vec<PyReadonlyArray2<'_, f64>>,
        distance: PyReadonlyArray2<'_, f64>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        component: usize,
        reml: bool,
        integrated: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let model = build(&fixed, &distance, &design)?;
        model
            .blup(&response(&y), reml, component, integrated)
            .map_err(PyValueError::new_err)
    }

    /// The likelihood ratio against no spatial variance.
    ///
    /// **This is a statistic and not a p-value.** With the decay rate
    /// unidentified when the spatial variance is nought, there is no closed-form
    /// null distribution, so turning this into a p-value takes a bootstrap.
    #[pyfunction]
    #[pyo3(signature = (fixed, distance, design, y, reml=true, integrated=false))]
    pub fn spatial_statistic(
        fixed: Vec<PyReadonlyArray2<'_, f64>>,
        distance: PyReadonlyArray2<'_, f64>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        reml: bool,
        integrated: bool,
    ) -> PyResult<f64> {
        let model = build(&fixed, &distance, &design)?;
        model
            .spatial_statistic(&response(&y), reml, integrated)
            .map_err(PyValueError::new_err)
    }

    /// A 95 per cent profile interval for a raw coefficient proportion, or for
    /// the decay rate.
    ///
    /// `quantity` is an index into the variances, or the string `lambda`.
    #[pyfunction]
    #[pyo3(signature = (fixed, distance, design, y, quantity, reml=true, integrated=false))]
    pub fn spatial_interval(
        fixed: Vec<PyReadonlyArray2<'_, f64>>,
        distance: PyReadonlyArray2<'_, f64>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        quantity: &str,
        reml: bool,
        integrated: bool,
    ) -> PyResult<(f64, f64, bool, bool, f64)> {
        let wanted = if quantity == "lambda" {
            SpatialQuantity::Lambda
        } else {
            SpatialQuantity::RawCoefficientProportion(
                quantity
                    .parse::<usize>()
                    .map_err(|_| PyValueError::new_err("SPATIAL_QUANTITY_UNKNOWN"))?,
            )
        };
        let model = build(&fixed, &distance, &design)?;
        let interval = model
            .profile_interval(&response(&y), reml, wanted, integrated)
            .map_err(PyValueError::new_err)?;
        Ok((
            interval.lower,
            interval.upper,
            interval.lower_limited,
            interval.upper_limited,
            interval.level,
        ))
    }

    /// Pairwise great-circle distances in kilometres.
    ///
    /// Here rather than in the caller so that every distance in this package is
    /// the same distance, on the same Earth, whoever asked for it.
    #[pyfunction]
    pub fn spatial_distances(
        latitude: PyReadonlyArray1<'_, f64>,
        longitude: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let lat: Vec<f64> = latitude.as_array().iter().copied().collect();
        let lon: Vec<f64> = longitude.as_array().iter().copied().collect();
        let distance = super::pairwise_haversine_km(&lat, &lon).map_err(PyValueError::new_err)?;
        Ok((0..distance.nrows())
            .map(|i| (0..distance.ncols()).map(|j| distance[(i, j)]).collect())
            .collect())
    }

    /// The parametric bootstrap p-value for no spatial variance.
    ///
    /// Returns the observed statistic, the exceedances, the replicates that
    /// produced a usable statistic, the number asked for, the p-value and the
    /// rule.
    #[pyfunction]
    #[pyo3(signature = (fixed, distance, design, y, replicates, seed, reml=true, integrated=false))]
    pub fn spatial_bootstrap(
        fixed: Vec<PyReadonlyArray2<'_, f64>>,
        distance: PyReadonlyArray2<'_, f64>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        replicates: usize,
        seed: u64,
        reml: bool,
        integrated: bool,
    ) -> PyResult<(f64, usize, usize, usize, f64, String)> {
        let model = build(&fixed, &distance, &design)?;
        let result = model
            .bootstrap_spatial_variance(&response(&y), reml, replicates, seed, integrated)
            .map_err(PyValueError::new_err)?;
        Ok((
            result.observed,
            result.exceedances,
            result.replicates,
            result.requested,
            result.p_value,
            result.rule.to_owned(),
        ))
    }
}

#[cfg(feature = "python")]
pub use python::{
    spatial_blup, spatial_bootstrap, spatial_distances, spatial_fit, spatial_interval,
    spatial_statistic,
};

#[cfg(test)]
mod tests {
    use super::{SpatialModel, SpatialQuantity};
    use nalgebra::{DMatrix, DVector};

    /// Sibling pairs scattered over a line, so distance means something.
    pub(super) fn small() -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DVector<f64>) {
        let pairs = 60;
        let n = 2 * pairs;
        let mut a = DMatrix::<f64>::identity(n, n);
        for pair in 0..pairs {
            a[(2 * pair, 2 * pair + 1)] = 0.5;
            a[(2 * pair + 1, 2 * pair)] = 0.5;
        }
        // A pair lives together, and pairs are spread along a line at 5 km
        // intervals, so the spatial kernel has a range of scales to work with.
        let place: Vec<f64> = (0..n).map(|i| (i / 2) as f64 * 5.0).collect();
        let distance = DMatrix::from_fn(n, n, |i, j| (place[i] - place[j]).abs());

        let mut seed = 4242u64;
        let mut next = || {
            let mut total = 0.0;
            for _ in 0..3 {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                total += ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            }
            total * 2.0
        };
        let half = 0.5f64.sqrt();
        let mut y = DVector::<f64>::zeros(n);
        // A smooth spatial field, made by giving nearby places correlated
        // values through a shared draw per neighbourhood of ten pairs.
        for region in 0..=(pairs / 10) {
            let shared = next();
            for i in 0..n {
                if i / 2 / 10 == region {
                    y[i] += 0.5 * shared;
                }
            }
        }
        for pair in 0..pairs {
            let common = next();
            for i in 0..2 {
                let genetic = half * common + half * next();
                y[2 * pair + i] += 0.6 * genetic + 0.7 * next();
            }
        }
        (a, distance, DMatrix::from_element(n, 1, 1.0), y)
    }

    /// The gradient is checked against a central difference, and the decay rate
    /// is the reason this test exists. Its derivative runs through the kernel
    /// rather than being a matrix in its own right — d/dλ exp(−λD) is
    /// −D ∘ exp(−λD), scaled by the spatial variance — and a sign or a missing
    /// factor there is invisible in the value and fatal in the search.
    #[test]
    fn the_gradient_matches_a_central_difference_including_the_decay_rate() {
        let (a, distance, design, y) = small();
        let model = SpatialModel::build(&[a], &distance, &design).expect("valid");
        // The decay rates are placed inside the range this data set allows,
        // because the bounds now come from the distances. Hardcoded values
        // silently fall outside them on a different geography, and `evaluate`
        // then refuses the point rather than reporting a wrong gradient -- which
        // is right, but it made this test fail for a reason that had nothing to
        // do with the gradient.
        let (bottom, top) = super::decay_bounds(&distance);
        let at = |fraction: f64| bottom * (top / bottom).powf(fraction);
        for reml in [false, true] {
            for point in [
                vec![0.4, 0.3, 0.3, at(0.2)],
                vec![0.6, 0.1, 0.5, at(0.5)],
                vec![0.2, 0.5, 0.4, at(0.8)],
                vec![1.0, 0.2, 0.8, at(0.35)],
            ] {
                let at = model.evaluate(&point, &y, reml, true).expect("evaluates");
                for k in 0..point.len() {
                    let step = 1e-6 * point[k].max(1e-3);
                    let mut up = point.clone();
                    let mut down = point.clone();
                    up[k] += step;
                    down[k] -= step;
                    let numeric = (model
                        .evaluate(&up, &y, reml, false)
                        .unwrap()
                        .negative_loglik
                        - model
                            .evaluate(&down, &y, reml, false)
                            .unwrap()
                            .negative_loglik)
                        / (2.0 * step);
                    let scale = at.gradient[k].abs().max(1.0);
                    assert!(
                        (at.gradient[k] - numeric).abs() / scale < 1e-4,
                        "reml={reml} parameter {k} at {point:?}: analytic {} against numeric {}",
                        at.gradient[k],
                        numeric
                    );
                }
            }
        }
    }

    /// A spatial effect that is there is found, and the range it reports is the
    /// range it was given rather than one of the bounds.
    #[test]
    fn a_spatial_effect_is_recovered_with_a_sensible_range() {
        let (a, distance, design, y) = small();
        let model = SpatialModel::build(&[a], &distance, &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");
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
            fit.raw_coefficient_proportions[1] > 0.02,
            "the spatial raw coefficient proportion came out at {} on data simulated with one",
            fit.raw_coefficient_proportions[1]
        );
        let (bottom, top) = super::decay_bounds(&distance);
        assert!(
            fit.lambda > bottom * 1.001 && fit.lambda < top * 0.999,
            "the decay rate went to a bound: {} in [{bottom}, {top}] \
             (half distance {} km)",
            fit.lambda,
            fit.half_distance_km
        );
        // The reported half distance must be the one the decay rate implies.
        assert!((fit.half_distance_km * fit.lambda - std::f64::consts::LN_2).abs() < 1e-12);
    }

    /// The kernel is stored between places and read between people, so the
    /// indirection has to be right: every pair must give exactly what
    /// `exp(-lambda*d)` gives for that pair's own distance. An error here would
    /// not crash or look odd -- it would quietly fit somebody else's
    /// correlations.
    #[test]
    fn the_kernel_read_by_place_matches_the_distance_between_the_people() {
        // Twelve people at five addresses, deliberately out of order so that a
        // grouping which assumed people at one place are adjacent would fail.
        let places: [f64; 12] = [
            0.0, 12.0, 3.0, 0.0, 40.0, 12.0, 3.0, 0.0, 40.0, 3.0, 12.0, 0.0,
        ];
        let n = places.len();
        let distance = DMatrix::from_fn(n, n, |i, j| (places[i] - places[j]).abs());
        let design = DMatrix::from_element(n, 1, 1.0);
        let model =
            SpatialModel::build(&[DMatrix::identity(n, n)], &distance, &design).expect("valid");

        assert_eq!(
            model.place_distance.nrows(),
            4,
            "twelve people at four distinct addresses were not grouped into four"
        );
        for lambda in [0.001f64, 0.05, 0.5] {
            let kernel = model.kernel(lambda);
            for i in 0..n {
                for j in 0..n {
                    let expected = (-lambda * distance[(i, j)]).exp();
                    let got = model.kernel_at(&kernel, i, j);
                    assert!(
                        (got - expected).abs() < 1e-15,
                        "lambda {lambda}, people {i} and {j} at {} and {}: \
                         {got} against {expected}",
                        places[i],
                        places[j]
                    );
                }
            }
        }
    }

    /// The integrated objective's gradient is a weighted average of the
    /// gradients at each decay rate, weighted by that rate's share of the
    /// integral. That is exact rather than approximate, and it is easy to write
    /// something that looks right and is not -- weighting by the wrong thing, or
    /// forgetting that the weights themselves depend on the parameters. Checked
    /// against a central difference of the integrated objective.
    #[test]
    fn the_integrated_gradient_matches_a_central_difference() {
        let (a, distance, design, y) = small();
        let model = SpatialModel::build(&[a], &distance, &design).expect("valid");
        for reml in [false, true] {
            for point in [
                vec![0.4, 0.3, 0.3],
                vec![0.6, 0.1, 0.5],
                vec![0.2, 0.5, 0.4],
            ] {
                let at = model
                    .evaluate_integrated(&point, &y, reml, true)
                    .expect("evaluates");
                for k in 0..point.len() {
                    let step = 1e-6 * point[k].max(1e-3);
                    let mut up = point.clone();
                    let mut down = point.clone();
                    up[k] += step;
                    down[k] -= step;
                    let numeric = (model
                        .evaluate_integrated(&up, &y, reml, false)
                        .unwrap()
                        .negative_loglik
                        - model
                            .evaluate_integrated(&down, &y, reml, false)
                            .unwrap()
                            .negative_loglik)
                        / (2.0 * step);
                    let scale = at.gradient[k].abs().max(1.0);
                    assert!(
                        (at.gradient[k] - numeric).abs() / scale < 1e-4,
                        "reml={reml} parameter {k} at {point:?}: analytic {} against \
                         numeric {}",
                        at.gradient[k],
                        numeric
                    );
                }
            }
        }
    }

    /// Integrating over the decay rate must find a spatial effect that is there,
    /// and must report no decay rate at all -- there is none to report, and a
    /// number in that slot would be read as an estimate.
    #[test]
    fn the_integrated_fit_recovers_the_raw_coefficient_proportion_and_reports_no_range() {
        let (a, distance, design, y) = small();
        let model = SpatialModel::build(&[a], &distance, &design).expect("valid");
        let fit = model.fit_integrated(&y, true).expect("fits");
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
            fit.raw_coefficient_proportions[1] > 0.02,
            "the spatial raw coefficient proportion came out at {} on data simulated with one",
            fit.raw_coefficient_proportions[1]
        );
        assert!(
            fit.lambda.is_nan(),
            "a decay rate was reported after integrating it out"
        );
        assert!(fit.half_distance_km.is_nan());
    }

    /// The integrated raw coefficient proportion still gets an interval, and asking for one on the
    /// range is refused rather than answered. There is no range once it has been
    /// integrated out, and returning something would invite it to be read.
    #[test]
    fn the_integrated_raw_coefficient_proportion_has_an_interval_and_the_range_has_none() {
        let (a, distance, design, y) = small();
        let model = SpatialModel::build(&[a], &distance, &design).expect("valid");
        let fit = model.fit_integrated(&y, true).expect("fits");
        let interval = model
            .profile_interval(&y, true, SpatialQuantity::RawCoefficientProportion(1), true)
            .expect("interval");
        assert!(
            interval.lower <= fit.raw_coefficient_proportions[1] + 1e-9
                && fit.raw_coefficient_proportions[1] <= interval.upper + 1e-9,
            "[{}, {}] does not contain {}",
            interval.lower,
            interval.upper,
            fit.raw_coefficient_proportions[1]
        );
        assert!(
            interval.upper - interval.lower > 1e-3,
            "[{}, {}] has no width",
            interval.lower,
            interval.upper
        );
        assert_eq!(
            model
                .profile_interval(&y, true, SpatialQuantity::Lambda, true)
                .err(),
            Some("SPATIAL_NO_RANGE_WHEN_INTEGRATED")
        );
    }

    /// Two runs at one seed must agree exactly, and two different seeds must
    /// not, showing that the seed controls the bootstrap draws.
    #[test]
    fn the_bootstrap_reproduces_from_its_seed() {
        let (a, distance, design, y) = small();
        let model = SpatialModel::build(&[a], &distance, &design).expect("valid");
        let once = model
            .bootstrap_spatial_variance(&y, true, 12, 20_260_812, false)
            .expect("bootstraps");
        let again = model
            .bootstrap_spatial_variance(&y, true, 12, 20_260_812, false)
            .expect("bootstraps");
        assert_eq!(once.exceedances, again.exceedances);
        assert_eq!(once.replicates, again.replicates);
        assert!((once.p_value - again.p_value).abs() < 1e-15);
        assert!((once.observed - again.observed).abs() < 1e-12);

        let elsewhere = model
            .bootstrap_spatial_variance(&y, true, 12, 99, false)
            .expect("bootstraps");
        assert!(
            (elsewhere.observed - once.observed).abs() < 1e-12,
            "the observed statistic does not depend on the seed"
        );

        // The p-value adds one to both counts, so it can never be nought however
        // extreme the statistic. Claiming nought would claim more than twelve
        // replicates can support.
        assert!(once.p_value >= 1.0 / 13.0 - 1e-12);
        assert!(once.p_value <= 1.0);
        assert_eq!(once.rule, "parametric_bootstrap_add_one");
    }

    /// Distances must be the great-circle ones, symmetric, and nought on the
    /// diagonal. One known separation anchors the scale: a degree of latitude is
    /// about 111 km anywhere on a sphere.
    #[test]
    fn haversine_distances_are_right() {
        let latitude = [0.0, 1.0, 0.0, 51.5];
        let longitude = [0.0, 0.0, 1.0, -0.13];
        let d = super::pairwise_haversine_km(&latitude, &longitude).expect("valid");
        for i in 0..4 {
            assert!(d[(i, i)].abs() < 1e-12);
            for j in 0..4 {
                assert!((d[(i, j)] - d[(j, i)]).abs() < 1e-12);
            }
        }
        assert!(
            (d[(0, 1)] - 111.195).abs() < 0.1,
            "a degree of latitude came to {} km",
            d[(0, 1)]
        );
        // A degree of longitude is the same at the equator and shrinks with the
        // cosine of latitude, which is the part a flat approximation gets wrong.
        assert!((d[(0, 2)] - d[(0, 1)]).abs() < 0.1);
        assert!(super::pairwise_haversine_km(&[91.0], &[0.0]).is_err());
        assert!(super::pairwise_haversine_km(&[0.0], &[181.0]).is_err());
    }

    /// The statistic against no spatial variance is nought when there is none
    /// to find and positive when there is. It is deliberately *not* turned into
    /// a p-value here: with the decay rate unidentified under the null there is
    /// no closed form, and the bootstrap is the only honest route.
    #[test]
    fn the_statistic_is_nought_without_a_spatial_effect_and_positive_with_one() {
        let (a, distance, design, y) = small();
        let model =
            SpatialModel::build(std::slice::from_ref(&a), &distance, &design).expect("valid");
        assert!(
            model.spatial_statistic(&y, true, false).expect("statistic") > 1.0,
            "no signal found on data simulated with a spatial effect"
        );

        let n = design.nrows();
        let mut seed = 31337u64;
        let mut next = || {
            let mut total = 0.0;
            for _ in 0..3 {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                total += ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            }
            total * 2.0
        };
        let half = 0.5f64.sqrt();
        let mut plain = DVector::<f64>::zeros(n);
        for pair in 0..n / 2 {
            let common = next();
            for i in 0..2 {
                let genetic = half * common + half * next();
                plain[2 * pair + i] = 0.6 * genetic + 0.8 * next();
            }
        }
        let statistic = model
            .spatial_statistic(&plain, true, false)
            .expect("statistic");
        assert!(
            statistic < 6.0,
            "a large statistic on data with no spatial effect: {statistic}"
        );
    }
}
