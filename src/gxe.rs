//! Genotype by environment: one model, with a surface on the environment.
//!
//! Both cross-sectional forms have the same shape. The genetic covariance
//! between two people is their relationship times a surface evaluated at their
//! two environments, and the residual adds a variance that depends on the
//! person's own environment:
//!
//! ```text
//! V_ij = A_ij * G(E_i, E_j)  +  (i == j) * R(E_i)
//! ```
//!
//! They differ only in what surface `G` is, which is why they are one module
//! and one likelihood rather than two of each:
//!
//! - **Exponential.** `sqrt(g(E_i) g(E_j)) * exp(-lambda * |E_i - E_j|)`, with
//!   the variance log-linear in the environment. Genetic effects are the more
//!   alike the closer two environments are, and `lambda` says how fast that
//!   falls away.
//! - **Random regression.** A smooth quadratic surface,
//!   `q00 + q01*(E_i + E_j) + q11*E_i*E_j`. No decay is imposed; the surface is
//!   whatever the data supports within a shape that stays a covariance.
//!
//! **The heritability is not one number in either.** It varies with the
//! environment, and the genetic correlation between two environments is below
//! one exactly when the genes acting in them differ. That correlation is the
//! interaction, and it is what these models exist to estimate.
//!
//! **The random-regression surface has to stay a covariance, and that is not a box.** Its
//! `q01` is free to be negative while `[[q00, q01], [q01, q11]]` must remain
//! positive semidefinite. The surface is therefore held by loadings rather than
//! by its entries: with `q00 = l00^2`, `q01 = l00*l10` and `q11 = l10^2 + u`,
//! every finite `l10` and non-negative `l00`, `u` gives a covariance, because
//! the determinant is `l00^2 * u`. The awkward constraint becomes a box the
//! optimiser already handles.
//!
//! **The third coordinate is `u` and not a Cholesky `l11`.** Written this way
//! the determinant is `l00^2 * u`, so the positive-semidefinite constraint is
//! explicit. This is only a reparameterisation: both coordinates describe the
//! same covariance set and therefore give the same maximised likelihood ratio.
//!
//! What actually makes the random-regression surface's tests conservative is the shape of
//! the null, and the two surfaces differ in it. The exponential surface's null
//! is `lambda = 0`, a flat face of its parameter box, which is the case the
//! even mixture of chi-squares is derived for -- and there the level comes out
//! at 0.048 against a nominal 0.05. The random-regression surface's null is
//! `q00*q11 - q01^2 = 0`, the *curved* boundary of the positive semidefinite
//! cone. On a cone the mixture weights follow the local solid angle rather than
//! being even, so the even mixture is the wrong reference. It errs the safe
//! way: the tests are valid but conservative, sitting on the bound in 68 to 84
//! per cent of null samples where an even mixture would say 50, and rejecting
//! on 0.3 per cent at a nominal 5.
//!
//! That costs power in principle and less than expected in practice. On its own
//! family the random-regression surface still finds a reordering more often than the
//! exponential one finds a reordering on *its* own family -- 0.333 against
//! 0.189 at a nominal 0.05. Where more power is wanted the reference would have
//! to be bootstrapped rather than looked up.
//!
//!
//! **The shape is barely identified and strongly changes the answer**, which is
//! why this package treats it as fixed rather than fitted. On one
//! simulated set the four shapes spanned 0.31 in log likelihood -- a deviance of
//! 0.62, which is nothing -- while the genetic correlation they reported ran
//! from 0.92 to 0.45. A shape chosen to suit the answer would therefore be
//! invisible in the fit. Choose it for a reason outside the data, and say which
//! shape was used beside the result.
//!
//! The environment is the caller's to centre and scale. Where it is centred
//! decides what the surface's intercept means, and that is a scientific choice
//! rather than one this module should make quietly.

use nalgebra::{DMatrix, DVector};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::blocks::family_blocks;
use crate::deviance::{chi2_one_df_upper_tail, chi2_two_df_upper_tail};
use crate::dense::DenseFactor;

/// The fixed shapes the powered-exponential kernel may take.
///
/// **This is an enumeration and not a number, on purpose.** Shape and decay
/// rate are poorly separated by data: they trade off against each other and a
/// search over both wanders. Making the grid a type means an off-grid shape
/// cannot be asked for, and the shape is chosen rather than fitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A rougher kernel than the exponential, falling away faster near nought.
    Half,
    /// The exponential kernel itself.
    One,
    ThreeHalves,
    /// The Gaussian kernel, smooth at nought and much flatter near it.
    Two,
}

impl Shape {
    #[must_use]
    pub const fn value(self) -> f64 {
        match self {
            Self::Half => 0.5,
            Self::One => 1.0,
            Self::ThreeHalves => 1.5,
            Self::Two => 2.0,
        }
    }

    /// Parse one of the four fixed shapes, refusing anything else.
    #[must_use]
    pub fn from_value(value: f64) -> Option<Self> {
        [Self::Half, Self::One, Self::ThreeHalves, Self::Two]
            .into_iter()
            .find(|shape| (shape.value() - value).abs() < 1e-9)
    }
}

/// Which surface is put on the environment-by-environment covariance.
///
/// This is the only thing that differs between the two cross-sectional forms,
/// and they report the same quantities. But **neither family contains the
/// other**, and the choice has two consequences worth knowing before it is
/// made.
///
/// A crossover -- a genotype that helps in one environment and harms in another,
/// so the genetic correlation is below nought rather than merely below one -- is
/// reachable by the random-regression surface and not by the exponential one, whose kernel
/// is positive at every rate.
///
/// And a surface that cannot bend its variance function the way the data does
/// will bend its correlation instead. On a rank-one genetic surface it cannot
/// represent, with no reordering present at all, the exponential form rejected
/// the correlation null on 10.7 per cent of 2000 samples against a nominal 5. Choose before seeing the
/// answer; `checks/gxe_calibration.py` measures what choosing wrongly costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// Log-linear variances with an exponential decay in environmental
    /// distance. Five parameters: `alpha_g`, `gamma_g`, `lambda_g`, `alpha_e`,
    /// `gamma_e`.
    Exponential,
    /// A smooth quadratic surface on the covariance, held by its Cholesky
    /// factor. Six parameters: three loadings for the genetic block and three
    /// for the residual.
    RandomRegression,
    /// The exponential surface with the decay taken to a fixed power:
    /// `exp(-lambda |difference|^kappa)`. The same five free parameters, with
    /// the shape chosen rather than fitted.
    ///
    /// At `Shape::One` this is the recovered exponential model exactly, which
    /// is the source's own statement and is checked by a test.
    PoweredExponential(Shape),
}

impl Surface {
    /// The power the environmental distance is raised to inside the decay.
    ///
    /// One for the exponential, and the fixed shape otherwise. Every
    /// place the kernel is written reads it from here, so the two forms cannot
    /// drift apart.
    #[must_use]
    pub const fn decay_exponent(self) -> f64 {
        match self {
            Self::PoweredExponential(shape) => shape.value(),
            _ => 1.0,
        }
    }

    /// How many free parameters the surface carries.
    #[must_use]
    pub const fn parameters(self) -> usize {
        match self {
            Self::Exponential | Self::PoweredExponential(_) => 5,
            Self::RandomRegression => 6,
        }
    }

    /// The two coordinates that shape the genetic surface, and which are
    /// nought exactly when there is no genotype-by-environment interaction of
    /// any kind.
    ///
    /// **The two surfaces agree on where these sit**, which is convenient but
    /// not a coincidence to lean on silently. For the exponential form they are
    /// the slope of the log genetic variance and the decay rate; for random
    /// regression they are the two off-intercept loadings of the genetic
    /// block. In both, setting them to nought leaves a genetic covariance that
    /// is one number times the relationship matrix -- an ordinary heritability
    /// model with no environment in it.
    ///
    /// In both, the second of the pair is bounded below at nought and the first
    /// is free to take either sign. That asymmetry is what decides the reference
    /// distribution, so it belongs here beside the indices rather than in the
    /// test that uses them.
    #[must_use]
    pub const fn genetic_shape(self) -> [usize; 2] {
        match self {
            // gamma_g, lambda
            Self::Exponential | Self::PoweredExponential(_) => [1, 2],
            // the genetic factor's l10 and l11
            Self::RandomRegression => [1, 2],
        }
    }

    /// The box the search runs in, as parallel lower and upper vectors.
    ///
    /// Kept here so the free fit, the held fits and the profile all read the
    /// same definition rather than three copies that can drift apart.
    #[must_use]
    pub fn bounds(self) -> (Vec<f64>, Vec<f64>) {
        match self {
            // alpha_g, gamma_g, lambda >= 0, alpha_e, gamma_e
            Self::Exponential | Self::PoweredExponential(_) => (
                vec![
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                    0.0,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                vec![f64::INFINITY; 5],
            ),
            // **No bounds at all**, for each of the two blocks. The intercept
            // is carried as its logarithm and the orthogonal loading enters
            // squared, so the block is a covariance whatever the six numbers
            // are and the search never meets an edge it can sit on. This
            // follows the custom implementation this model came from; an
            // earlier version here bounded the intercept and the squared
            // loading directly, which put two walls in the parameter space
            // that the original does not have.
            Self::RandomRegression => (
                vec![
                    -15.0,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                    -15.0,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                vec![
                    15.0,
                    f64::INFINITY,
                    f64::INFINITY,
                    15.0,
                    f64::INFINITY,
                    f64::INFINITY,
                ],
            ),
        }
    }

    /// The single coordinate that is nought exactly when the genetic effects at
    /// any two environments are perfectly correlated.
    ///
    /// The genetic variance may still change with the environment; what this
    /// rules out is a change in which genes matter. For the exponential form it
    /// is the decay rate, for the random-regression form the last loading of the genetic
    /// Cholesky factor, and in both it is bounded below at nought.
    #[must_use]
    pub const fn rank_one(self) -> usize {
        self.genetic_shape()[1]
    }
}

/// The most parameters any surface uses, so one array serves both.
const PARAMETERS: usize = 6;

/// A fitted genotype-by-environment model.
///
/// **The parameters are not the answer and are not reported as one.** What the
/// two surfaces have in common is what they say about the world: how the
/// genetic variance changes across the environment, and how alike the genetic
/// effects are at two different environments. Those are asked for by
/// environment rather than read off a coefficient, and they mean the same thing
/// whichever surface produced them.
#[derive(Clone, Debug)]
pub struct GxeFit {
    pub surface: Surface,
    /// The fitted parameters, in the surface's own coordinates. Kept so a fit
    /// can be reproduced and inspected, not because they are interpretable.
    pub parameters: Vec<f64>,
    pub fixed_effects: Vec<f64>,
    pub fixed_effect_errors: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    pub scaled_gradient: f64,
    pub estimator: &'static str,
    /// The scale the response was standardised by, needed to put a variance
    /// back into the response's own units.
    variance_scale: f64,
}

impl GxeFit {
    fn theta(&self) -> [f64; PARAMETERS] {
        let mut theta = [0.0; PARAMETERS];
        theta[..self.parameters.len()].copy_from_slice(&self.parameters);
        theta
    }

    /// The genetic covariance between two environments.
    #[must_use]
    pub fn genetic_covariance(&self, first: f64, second: f64) -> f64 {
        let theta = self.theta();
        let value = match self.surface {
            Surface::Exponential | Surface::PoweredExponential(_) => {
                let gap = (first - second).abs().powf(self.surface.decay_exponent());
                (0.5 * (theta[0] + theta[1] * first)).exp()
                    * (0.5 * (theta[0] + theta[1] * second)).exp()
                    * (-theta[2] * gap).exp()
            }
            Surface::RandomRegression => {
                let g = block_from_loadings(theta[0], theta[1], theta[2]);
                g[0] + g[1] * (first + second) + g[2] * first * second
            }
        };
        value * self.variance_scale
    }

    /// The genetic variance at one environment.
    #[must_use]
    pub fn genetic_variance_at(&self, z: f64) -> f64 {
        self.genetic_covariance(z, z)
    }

    /// The residual variance at one environment.
    #[must_use]
    pub fn residual_variance_at(&self, z: f64) -> f64 {
        let theta = self.theta();
        let value = match self.surface {
            Surface::Exponential | Surface::PoweredExponential(_) => {
                (theta[3] + theta[4] * z).exp()
            }
            Surface::RandomRegression => {
                let e = block_from_loadings(theta[3], theta[4], theta[5]);
                e[0] + 2.0 * e[1] * z + e[2] * z * z
            }
        };
        value * self.variance_scale
    }

    /// The heritability at one environment.
    ///
    /// **This is the point of the model.** It says the heritability is a
    /// function of the environment, so a single number for it is not something
    /// the model has, and asking for one is asking the wrong question.
    #[must_use]
    pub fn heritability_at(&self, z: f64) -> f64 {
        let genetic = self.genetic_variance_at(z);
        let total = genetic + self.residual_variance_at(z);
        if total > 0.0 {
            genetic / total
        } else {
            f64::NAN
        }
    }

    /// The genetic correlation between two environments.
    ///
    /// **This is the interaction.** One means the same genes act in both
    /// environments and the model has found no interaction at all; below one is
    /// the interaction itself, and how far below says how much of it there is.
    #[must_use]
    pub fn genetic_correlation(&self, first: f64, second: f64) -> f64 {
        let covariance = self.genetic_covariance(first, second);
        let spread = (self.genetic_variance_at(first) * self.genetic_variance_at(second)).sqrt();
        if spread > 0.0 {
            (covariance / spread).clamp(-1.0, 1.0)
        } else {
            f64::NAN
        }
    }
}

struct Evaluation {
    negative_loglik: f64,
    gradient: [f64; PARAMETERS],
    fixed_effects: Vec<f64>,
    fixed_covariance: Option<DMatrix<f64>>,
}

/// One trait, with the genetic and residual effects regressed on an
/// environment.
pub struct GxeModel {
    surface: Surface,
    relationship: DMatrix<f64>,
    z: Vec<f64>,
    design: DMatrix<f64>,
    blocks: Vec<Vec<usize>>,
    rows: usize,
    logdet_xtx: f64,
}

/// Turn a Cholesky factor into the covariance entries it stands for.
fn block_from_loadings(log_intercept: f64, linear: f64, orthogonal: f64) -> [f64; 3] {
    let intercept = log_intercept.exp();
    [
        intercept * intercept,
        intercept * linear,
        linear * linear + orthogonal * orthogonal,
    ]
}

impl GxeModel {
    /// Validate and prepare.
    ///
    /// `z` is the environment, one value per person, already centred and scaled
    /// however the analysis wants it.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(
        surface: Surface,
        relationship: &DMatrix<f64>,
        z: &[f64],
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let n = design.nrows();
        if n == 0 {
            return Err("GXE_NO_ROWS");
        }
        if relationship.nrows() != n || relationship.ncols() != n {
            return Err("GXE_RELATIONSHIP_WRONG_SIZE");
        }
        if z.len() != n {
            return Err("GXE_ENVIRONMENT_WRONG_LENGTH");
        }
        if z.iter().any(|v| !v.is_finite()) {
            return Err("GXE_ENVIRONMENT_NOT_FINITE");
        }
        let xtx = design.transpose() * design;
        let chol = xtx.cholesky().ok_or("GXE_DESIGN_RANK_DEFICIENT")?;
        let logdet_xtx = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        Ok(Self {
            surface,
            relationship: relationship.clone(),
            z: z.to_vec(),
            design: design.clone(),
            blocks: family_blocks(relationship),
            rows: n,
            logdet_xtx,
        })
    }

    /// The genetic surface at two environments.
    fn genetic_surface(&self, theta: &[f64; PARAMETERS], zi: f64, zj: f64) -> f64 {
        match self.surface {
            Surface::Exponential | Surface::PoweredExponential(_) => {
                let (alpha, gamma, lambda) = (theta[0], theta[1], theta[2]);
                // The square root of each variance, times the decay in
                // environmental distance raised to the fixed shape.
                let gap = (zi - zj).abs().powf(self.surface.decay_exponent());
                (0.5 * (alpha + gamma * zi)).exp()
                    * (0.5 * (alpha + gamma * zj)).exp()
                    * (-lambda * gap).exp()
            }
            Surface::RandomRegression => {
                let g = block_from_loadings(theta[0], theta[1], theta[2]);
                g[0] + g[1] * (zi + zj) + g[2] * zi * zj
            }
        }
    }

    /// The residual variance at one environment.
    fn residual_surface(&self, theta: &[f64; PARAMETERS], zi: f64) -> f64 {
        match self.surface {
            Surface::Exponential | Surface::PoweredExponential(_) => {
                (theta[3] + theta[4] * zi).exp()
            }
            Surface::RandomRegression => {
                let e = block_from_loadings(theta[3], theta[4], theta[5]);
                e[0] + 2.0 * e[1] * zi + e[2] * zi * zi
            }
        }
    }

    /// The covariance of one block at these parameters.
    fn assemble(&self, block: &[usize], theta: &[f64; PARAMETERS]) -> DMatrix<f64> {
        let size = block.len();
        DMatrix::from_fn(size, size, |i, j| {
            let (a, b) = (block[i], block[j]);
            let (zi, zj) = (self.z[a], self.z[b]);
            let genetic = self.relationship[(a, b)] * self.genetic_surface(theta, zi, zj);
            if i == j {
                genetic + self.residual_surface(theta, zi)
            } else {
                genetic
            }
        })
    }

    /// The derivative of one block's covariance with respect to one parameter.
    ///
    /// For the exponential surfaces every derivative is the surface itself
    /// times something simple, because the surface is an exponential. For
    /// random regression the chain runs through the loadings, so each
    /// covariance entry moves with more than one of them.
    fn derivative(
        &self,
        block: &[usize],
        theta: &[f64; PARAMETERS],
        parameter: usize,
    ) -> DMatrix<f64> {
        let size = block.len();
        DMatrix::from_fn(size, size, |i, j| {
            let (a, b) = (block[i], block[j]);
            let (zi, zj) = (self.z[a], self.z[b]);
            match self.surface {
                Surface::Exponential | Surface::PoweredExponential(_) => {
                    let genetic = self.genetic_surface(theta, zi, zj);
                    let residual = self.residual_surface(theta, zi);
                    let gap = (zi - zj).abs().powf(self.surface.decay_exponent());
                    let d_genetic = match parameter {
                        0 => genetic,                   // alpha_g
                        1 => genetic * 0.5 * (zi + zj), // gamma_g
                        2 => -genetic * gap,            // lambda_g
                        _ => 0.0,
                    };
                    let d_residual = match parameter {
                        3 => residual,      // alpha_e
                        4 => residual * zi, // gamma_e
                        _ => 0.0,
                    };
                    let value = self.relationship[(a, b)] * d_genetic;
                    if i == j { value + d_residual } else { value }
                }
                Surface::RandomRegression => {
                    let genetic_side = parameter < 3;
                    let (log_intercept, linear, orthogonal) = if genetic_side {
                        (theta[0], theta[1], theta[2])
                    } else {
                        (theta[3], theta[4], theta[5])
                    };
                    let intercept = log_intercept.exp();
                    let d = match parameter % 3 {
                        // q00 = a^2, q01 = a * linear, and a = exp(.), so both
                        // carry a factor of a through the chain.
                        0 => [2.0 * intercept * intercept, intercept * linear, 0.0],
                        1 => [0.0, intercept, 2.0 * linear],
                        _ => [0.0, 0.0, 2.0 * orthogonal],
                    };
                    if genetic_side {
                        self.relationship[(a, b)] * (d[0] + d[1] * (zi + zj) + d[2] * zi * zj)
                    } else if i == j {
                        d[0] + 2.0 * d[1] * zi + d[2] * zi * zi
                    } else {
                        0.0
                    }
                }
            }
        })
    }

    fn evaluate(
        &self,
        theta: &[f64; PARAMETERS],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
    ) -> Option<Evaluation> {
        if theta.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // The diagonal loadings are non-negative and the off-diagonal ones are
        // free; that box is what keeps each block positive semidefinite.
        match self.surface {
            // Only the decay is constrained; the log-linear coefficients are
            // free because a variance written as an exponential is positive
            // whatever they are.
            Surface::Exponential | Surface::PoweredExponential(_) => {
                if theta[2] < 0.0 {
                    return None;
                }
            }
            // **Nothing to check.** The intercept is carried as its logarithm
            // and the orthogonal loading enters squared, so every real six
            // numbers give two covariances. That is the point of the
            // parameterisation and the reason the search has no edge to sit on.
            Surface::RandomRegression => {}
        }

        let p = self.design.ncols();
        let mut logdet = 0.0;
        let mut xvx = DMatrix::<f64>::zeros(p, p);
        let mut xvy = DVector::<f64>::zeros(p);
        let mut yvy = 0.0;
        let mut kept: Vec<(Vec<usize>, DMatrix<f64>, DVector<f64>, DMatrix<f64>)> =
            Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            let size = block.len();
            let v = self.assemble(block, theta);
            let chol = DenseFactor::new(&v)?;
            logdet += chol.logdet();
            let yb = DVector::from_iterator(size, block.iter().map(|&i| y[i]));
            let xb = DMatrix::from_fn(size, p, |r, c| self.design[(block[r], c)]);
            let vy = chol.solve_vector(&yb);
            let vx = chol.solve_matrix(&xb);
            xvx += xb.transpose() * &vx;
            xvy += xb.transpose() * &vy;
            yvy += yb.dot(&vy);
            if want_gradient {
                kept.push((block.clone(), chol.inverse(), vy, vx));
            }
        }

        let xvx_chol = xvx.clone().cholesky()?;
        let beta = xvx_chol.solve(&xvy);
        let quadratic = yvy - xvy.dot(&beta);
        if !(quadratic > 0.0) || !quadratic.is_finite() {
            return None;
        }

        let two_pi = (2.0 * std::f64::consts::PI).ln();
        let mut value = 0.5 * (self.rows as f64 * two_pi + logdet + quadratic);
        if reml {
            let logdet_xvx = 2.0 * xvx_chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
            value += 0.5 * (logdet_xvx - self.logdet_xtx) - 0.5 * p as f64 * two_pi;
        }

        let mut gradient = [0.0; PARAMETERS];
        if want_gradient {
            let xvx_inverse = xvx_chol.inverse();
            let residuals: Vec<DVector<f64>> =
                kept.iter().map(|(_, _, vy, vx)| vy - vx * &beta).collect();
            for parameter in 0..PARAMETERS {
                let mut trace = 0.0;
                let mut quadratic_term = 0.0;
                let mut restricted = DMatrix::<f64>::zeros(p, p);
                for (slot, (block, inverse, _, vx)) in kept.iter().enumerate() {
                    let dv = self.derivative(block, theta, parameter);
                    let size = block.len();
                    for i in 0..size {
                        for j in 0..size {
                            trace += inverse[(i, j)] * dv[(j, i)];
                        }
                    }
                    let r = &residuals[slot];
                    quadratic_term += r.dot(&(&dv * r));
                    if reml {
                        restricted += vx.transpose() * (&dv * vx);
                    }
                }
                let mut d = 0.5 * (trace - quadratic_term);
                if reml {
                    d -= 0.5 * (&xvx_inverse * &restricted).trace();
                }
                gradient[parameter] = d;
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
    /// # Errors
    ///
    /// Returns a stable code where no start reached a usable optimum.
    pub fn fit(&self, y: &DVector<f64>, reml: bool) -> Result<GxeFit, &'static str> {
        self.fit_holding(y, reml, &[])
    }

    /// Fit with some coordinates held at nought, which is how every null here
    /// is imposed.
    ///
    /// A held coordinate has its two bounds set equal, so the search simply does
    /// not move it. This keeps one code path for the free and the constrained
    /// fit: a null fitted by different machinery from the alternative is the
    /// classic way to get a deviance that is not a deviance.
    fn fit_holding(
        &self,
        y: &DVector<f64>,
        reml: bool,
        held: &[usize],
    ) -> Result<GxeFit, &'static str> {
        if y.len() != self.rows {
            return Err("GXE_RESPONSE_WRONG_LENGTH");
        }
        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        if !(variance > 0.0) {
            return Err("GXE_RESPONSE_CONSTANT");
        }
        let scale = variance.sqrt();
        let scaled = y / scale;

        let count = self.surface.parameters();
        let (mut lower, mut upper) = self.surface.bounds();
        for &k in held {
            if k >= count {
                return Err("GXE_HELD_COORDINATE_OUT_OF_RANGE");
            }
            lower[k] = 0.0;
            upper[k] = 0.0;
        }
        // Starts spanning no interaction (slopes at nought) and some, in both
        // directions, because a slope is signed and a search begun at nought can
        // sit there.
        let starts: Vec<Vec<f64>> = match self.surface {
            // Log variances near a half of the response's own, and decays
            // spanning "no interaction" to "a lot".
            Surface::Exponential | Surface::PoweredExponential(_) => vec![
                vec![-0.7, 0.0, 0.01, -0.7, 0.0],
                vec![-0.7, 0.3, 0.5, -0.7, 0.1],
                vec![-0.7, -0.3, 0.1, -0.7, -0.1],
                vec![-0.7, 0.0, 2.0, -0.7, 0.0],
            ],
            // Multiple fixed starts are used because this surface is
            // start-sensitive. They cover both slope directions and a flat
            // surface in the `(g_l10, g_l11)` coordinates.
            //
            // ln(0.7) is about -0.36, which puts half the response's own
            // variance in each block -- a heritability of a half.
            Surface::RandomRegression => vec![
                vec![-0.36, 0.0, 0.0, -0.36, 0.0, 0.0],
                vec![-0.36, -0.6, 0.4, -0.36, -0.1, 0.3],
                vec![-0.36, -0.2, 0.6, -0.36, -0.1, 0.3],
                vec![-0.36, 0.0, 0.6, -0.36, 0.0, 0.3],
                vec![-0.36, 0.2, 0.6, -0.36, 0.1, 0.3],
                vec![-0.36, 0.6, 0.4, -0.36, 0.1, 0.3],
                vec![-0.9, 0.0, 0.6, -0.22, 0.0, 0.45],
            ],
        };
        // Holding a coordinate collapses starts onto one another, and running
        // the same search four times only costs time.
        let mut starts: Vec<Vec<f64>> = starts
            .into_iter()
            .map(|mut start| {
                for &k in held {
                    start[k] = 0.0;
                }
                start
            })
            .collect();
        starts.dedup_by(|a, b| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-12));

        let mut best: Option<(f64, Vec<f64>, Vec<f64>, Option<DMatrix<f64>>)> = None;
        for start in starts {
            let value_of = |c: &[f64]| -> f64 {
                let mut theta = [0.0; PARAMETERS];
                theta[..c.len()].copy_from_slice(c);
                self.evaluate(&theta, &scaled, reml, false)
                    .map_or(1e30, |e| e.negative_loglik)
            };
            let gradient_of = |c: &[f64]| -> Vec<f64> {
                let mut theta = [0.0; PARAMETERS];
                theta[..c.len()].copy_from_slice(c);
                self.evaluate(&theta, &scaled, reml, true)
                    .map_or_else(|| vec![0.0; count], |e| e.gradient[..count].to_vec())
            };
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
            let mut theta = [0.0; PARAMETERS];
            theta[..solution.par.len()].copy_from_slice(&solution.par);
            let Some(at) = self.evaluate(&theta, &scaled, reml, true) else {
                continue;
            };
            if at.negative_loglik.is_finite()
                && best
                    .as_ref()
                    .is_none_or(|(value, _, _, _)| at.negative_loglik < *value)
            {
                best = Some((
                    at.negative_loglik,
                    solution.par.clone(),
                    at.fixed_effects,
                    at.fixed_covariance,
                ));
            }
        }

        let (negative, par, beta, covariance) = best.ok_or("GXE_NO_START_CONVERGED")?;
        let mut theta = [0.0; PARAMETERS];
        theta[..par.len()].copy_from_slice(&par);
        let at = self
            .evaluate(&theta, &scaled, reml, true)
            .ok_or("GXE_OPTIMUM_NOT_EVALUABLE")?;
        let projected = at.gradient[..count]
            .iter()
            .enumerate()
            .map(|(k, g)| {
                if held.contains(&k) {
                    0.0
                } else if lower[k] > f64::NEG_INFINITY && par[k] <= lower[k] {
                    g.min(0.0)
                } else {
                    *g
                }
            })
            .fold(0.0f64, |worst, g| worst.max(g.abs()));
        let scaled_gradient = projected / negative.abs().max(1.0);

        let observations = if reml {
            (self.rows - self.design.ncols()) as f64
        } else {
            self.rows as f64
        };
        Ok(GxeFit {
            surface: self.surface,
            parameters: par[..count].to_vec(),
            variance_scale: variance,
            fixed_effects: beta.iter().map(|b| b * scale).collect(),
            fixed_effect_errors: covariance
                .as_ref()
                .map(|c| {
                    (0..c.nrows())
                        .map(|i| c[(i, i)].max(0.0).sqrt() * scale)
                        .collect()
                })
                .unwrap_or_default(),
            loglik: -negative - observations * scale.ln(),
            converged: scaled_gradient < 1e-6,
            scaled_gradient,
            estimator: if reml { "reml" } else { "ml" },
        })
    }
}

/// One test of a genotype-by-environment null.
#[derive(Clone, Copy, Debug)]
pub struct GxeTest {
    /// Twice the difference in log likelihood, never below nought.
    pub statistic: f64,
    pub p_value: f64,
    /// Which reference distribution was used, so a reader can tell what the
    /// p-value is a tail of.
    pub rule: &'static str,
    pub null_loglik: f64,
    pub alternative_loglik: f64,
}



impl GxeModel {
    /// Test for any genotype-by-environment interaction at all.
    ///
    /// The null holds the genetic covariance flat: one variance times the
    /// relationship matrix, with the environment entering nowhere. **The
    /// residual surface is left free under the null**, deliberately. A
    /// residual variance that changes with the environment is not a
    /// gene-by-environment interaction, and a test that rejected because of one
    /// would be answering a different question from the one it was asked.
    ///
    /// # The reference distribution
    ///
    /// Two coordinates are held, and they are not alike: one is bounded below
    /// at nought and sits on that bound under the null, the other is free to
    /// take either sign. That is Self and Liang's mixed case, whose reference is
    /// half chi-square on one degree of freedom and half on two. Taking the
    /// plain chi-square on two would be conservative, and taking one degree of
    /// freedom would not be — but neither is derived here on trust:
    /// `checks/gxe_calibration.py` measures the rejection rate against it.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit fails.
    pub fn interaction_test(&self, y: &DVector<f64>, reml: bool) -> Result<GxeTest, &'static str> {
        let free = self.fit(y, reml)?;
        let held = self.fit_holding(y, reml, &self.surface.genetic_shape())?;
        Ok(mixture(
            free.loglik,
            held.loglik,
            "half_chi2_1_half_chi2_2",
            |statistic| {
                0.5 * chi2_one_df_upper_tail(statistic) + 0.5 * chi2_two_df_upper_tail(statistic)
            },
        ))
    }

    /// Test whether the genetic variance changes with the environment at all.
    ///
    /// This is the source-defined `gamma_G = 0` null of the recovered SOLAR
    /// model, and on the exponential surface it is exactly that: one interior
    /// coordinate held at nought, referred to chi-square on one degree of
    /// freedom with no boundary and no mixture.
    ///
    /// **On the random-regression surface it is not a separate test**, and saying so is
    /// better than inventing one. There the genetic variance is
    /// `q00 + 2 q01 z + q11 z^2`, which is constant only when `q01` and `q11`
    /// are both nought, and that is the same pair of coordinates
    /// `interaction_test` holds. So this returns that test on that surface,
    /// with its mixture reference, rather than a differently named copy.
    ///
    /// A variance that changes while the correlation stays at one is
    /// amplification rather than a reordering: the same genes throughout,
    /// acting more strongly in some environments. It can follow from a change
    /// of scale in the measurement, which is why `correlation_test` is usually
    /// the more interesting of the two.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit fails.
    pub fn variance_test(&self, y: &DVector<f64>, reml: bool) -> Result<GxeTest, &'static str> {
        match self.surface {
            Surface::RandomRegression => self.interaction_test(y, reml),
            Surface::Exponential | Surface::PoweredExponential(_) => {
                let free = self.fit(y, reml)?;
                let held = self.fit_holding(y, reml, &[self.surface.genetic_shape()[0]])?;
                Ok(mixture(
                    free.loglik,
                    held.loglik,
                    "chi2_1",
                    chi2_one_df_upper_tail,
                ))
            }
        }
    }

    /// Test whether the genetic effects at two environments are the same
    /// effects.
    ///
    /// The null is a genetic correlation of one across the whole environmental
    /// range: the genetic variance may still grow or shrink, but the ordering of
    /// people by genetic value does not change. **This is the narrower and
    /// usually the more interesting question.** A heritability that rises with
    /// an environment is a change of scale, and can follow from a change of
    /// scale in the measurement; a correlation below one cannot.
    ///
    /// One coordinate is held, on its lower bound, so the reference is the
    /// even mixture of a point mass at nought and chi-square on one degree of
    /// freedom.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit fails.
    pub fn correlation_test(&self, y: &DVector<f64>, reml: bool) -> Result<GxeTest, &'static str> {
        let free = self.fit(y, reml)?;
        let held = self.fit_holding(y, reml, &[self.surface.rank_one()])?;
        Ok(mixture(
            free.loglik,
            held.loglik,
            "mixture_50_50",
            |statistic| 0.5 * chi2_one_df_upper_tail(statistic),
        ))
    }
}

/// Assemble a test from two log likelihoods and a reference tail.
///
/// The clamped statistic and the point mass at nought both live in
/// [`crate::deviance`], which explains why each is needed.
fn mixture(alternative: f64, null: f64, rule: &'static str, tail: impl Fn(f64) -> f64) -> GxeTest {
    let statistic = crate::deviance::deviance(alternative, null);
    GxeTest {
        statistic,
        p_value: crate::deviance::p_value(statistic, tail),
        rule,
        null_loglik: null,
        alternative_loglik: alternative,
    }
}

/// A quantity the model reports and can put a profile interval on.
///
/// Both are free of the scale the response was standardised by, being ratios,
/// so the profile can run entirely on the standardised scale.
#[derive(Clone, Copy, Debug)]
pub enum Reported {
    /// The heritability at one environment.
    Heritability { at: f64 },
    /// The genetic correlation between two environments.
    GeneticCorrelation { first: f64, second: f64 },
}

/// One profile-likelihood interval.
#[derive(Clone, Copy, Debug)]
pub struct GxeInterval {
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
    /// The endpoint ran to the edge of what the quantity can be rather than to
    /// a likelihood crossing, so it is a limit of the parameter space and not a
    /// measurement. Read it beside `profile_failures`: a bound reached because
    /// the likelihood never crossed and a bound reached because the profile
    /// could not be evaluated there are both reported here, and only a non-zero
    /// failure count separates them.
    pub lower_at_bound: bool,
    pub upper_at_bound: bool,
    pub level: f64,
    /// How many profile evaluations could not be made. A failure is unknown
    /// ground, not ground the data ruled out, so the interval is widened over
    /// it rather than narrowed; a non-zero count says the endpoints rest partly
    /// on evaluations that did not come back.
    pub profile_failures: usize,
}

/// Chi-square on one degree of freedom at 0.95, the profile's threshold.
const CHI2_ONE_95: f64 = 3.841_458_820_694_124;

impl GxeModel {
    /// Move a parameter vector onto the surface where a reported quantity
    /// takes a given value.
    ///
    /// **This returns a whole repaired vector rather than one coordinate to
    /// overwrite**, because the obvious per-coordinate solve is a trap. Holding
    /// a heritability by solving the random-regression surface's `u_e` gives
    /// `u_e = (wanted - (l00_e + l10_e z)^2) / z^2`, which is negative over much
    /// of the search, and an infeasible point scores as no fit at all -- so the
    /// profile stops at the edge of the feasible region and reports it as a
    /// likelihood crossing. The interval then looks tight and precise and is
    /// neither. It showed up as two different heritabilities whose intervals
    /// began at the same number.
    ///
    /// A heritability is therefore held by *scaling the whole residual
    /// surface*, which is feasible at every point because the scale is a
    /// positive number and nothing constrains it. The search keeps all its
    /// coordinates and gains one redundant direction along which the objective
    /// is flat; that costs a little time and no correctness, since the maximum
    /// over a set does not care how the set is parameterised.
    ///
    /// Returns `None` only where the value is genuinely out of reach.
    fn repair(
        &self,
        theta: &[f64; PARAMETERS],
        quantity: Reported,
        value: f64,
    ) -> Option<[f64; PARAMETERS]> {
        let mut out = *theta;
        match quantity {
            Reported::Heritability { at } => {
                if !(value > 0.0 && value < 1.0) {
                    return None;
                }
                let genetic = self.genetic_surface(theta, at, at);
                let wanted = genetic * (1.0 - value) / value;
                let have = self.residual_surface(theta, at);
                if !(wanted > 0.0) || !(have > 0.0) || !wanted.is_finite() {
                    return None;
                }
                let scale = wanted / have;
                match self.surface {
                    // exp(alpha_e + gamma_e z) * scale
                    Surface::Exponential | Surface::PoweredExponential(_) => {
                        out[3] += scale.ln();
                    }
                    // Scaling a covariance block by s scales all three of its
                    // loadings by sqrt(s) -- and the intercept is carried as a
                    // logarithm, so its share of that is an addition.
                    Surface::RandomRegression => {
                        let root = scale.sqrt();
                        out[3] += 0.5 * scale.ln();
                        out[4] *= root;
                        out[5] *= root;
                    }
                }
                Some(out)
            }
            Reported::GeneticCorrelation { first, second } => {
                let gap = (first - second).abs();
                if gap < 1e-12 || !(-1.0..=1.0).contains(&value) {
                    // At one environment the correlation is one by
                    // construction and nothing is free to hold.
                    return None;
                }
                match self.surface {
                    // exp(-lambda gap) = value; the kernel cannot be negative
                    // however large the rate.
                    Surface::Exponential | Surface::PoweredExponential(_) => {
                        if !(value > 0.0 && value <= 1.0) {
                            return None;
                        }
                        out[2] = -value.ln() / gap.powf(self.surface.decay_exponent());
                        Some(out)
                    }
                    Surface::RandomRegression => {
                        // With A = l00 + l10 z1 and B = l00 + l10 z2, holding
                        // r^2 q11 q22 = q12^2 is a plain quadratic in u.
                        let (intercept, linear) = (theta[0].exp(), theta[1]);
                        let (a, b) = (intercept + linear * first, intercept + linear * second);
                        let (r2, z1, z2) = (value * value, first, second);
                        let qa = z1 * z1 * z2 * z2 * (r2 - 1.0);
                        let qb = r2 * (a * a * z2 * z2 + b * b * z1 * z1) - 2.0 * a * b * z1 * z2;
                        let qc = a * a * b * b * (r2 - 1.0);
                        let u = solve_quadratic(qa, qb, qc)?
                            .into_iter()
                            .filter(|u| *u >= 0.0 && u.is_finite())
                            // Squaring the constraint threw the sign away, so
                            // the root has to reproduce it as well as the size.
                            .find(|u| {
                                let q12 = a * b + u * z1 * z2;
                                (q12 >= 0.0) == (value >= 0.0)
                            })?;
                        // The coordinate is the orthogonal loading, and the
                        // quadratic was solved for its square.
                        out[2] = u.sqrt();
                        Some(out)
                    }
                }
            }
        }
    }

    /// The best log likelihood with one reported quantity held at `value`.
    ///
    /// The held coordinate is solved rather than searched, so the search runs
    /// over one coordinate fewer. Its gradient carries the chain term through
    /// the substitution: the likelihood's own derivatives are exact and only
    /// the cheap algebraic map is differenced, which costs no extra
    /// factorisation.
    fn profile_objective(
        &self,
        y: &DVector<f64>,
        reml: bool,
        quantity: Reported,
        value: f64,
        start_from: &[f64],
    ) -> Option<f64> {
        let count = self.surface.parameters();
        let repair = |free: &[f64]| self.repair(&to_theta(free), quantity, value);

        let value_of = |free: &[f64]| -> f64 {
            repair(free)
                .and_then(|theta| self.evaluate(&theta, y, reml, false))
                .map_or(1e30, |e| e.negative_loglik)
        };
        // The likelihood's own derivatives are exact; only the repair, which is
        // cheap algebra and factorises nothing, is differenced.
        let gradient_of = |free: &[f64]| -> Vec<f64> {
            let Some(theta) = repair(free) else {
                return vec![0.0; count];
            };
            let Some(at) = self.evaluate(&theta, y, reml, true) else {
                return vec![0.0; count];
            };
            (0..count)
                .map(|j| {
                    let step = 1e-6 * free[j].abs().max(1.0);
                    let mut up = free.to_vec();
                    let mut down = free.to_vec();
                    up[j] += step;
                    down[j] -= step;
                    match (repair(&up), repair(&down)) {
                        (Some(u), Some(d)) => (0..count)
                            .map(|k| at.gradient[k] * (u[k] - d[k]) / (2.0 * step))
                            .sum(),
                        _ => at.gradient[j],
                    }
                })
                .collect()
        };

        let start = start_from[..count].to_vec();
        repair(&start)?;
        let (lower, upper) = self.surface.bounds();
        let bounds = Bounds::new(lower, upper).ok()?;
        let mut control = OptimControl::default_for_dimension(count);
        control.maxit = 300;
        control.fnscale = value_of(&start).abs().max(1.0);
        control.parscale = vec![1.0; count];
        control.factr = 1.0e3;
        control.pgtol = 1e-8;
        control.lmm = count;
        let best =
            optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control)
                .map_or_else(
                    |_| value_of(&start),
                    |s| value_of(&s.par).min(value_of(&start)),
                );
        best.is_finite().then_some(-best)
    }

    /// A 95 per cent profile-likelihood interval for one reported quantity.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the free fit fails or the quantity cannot be
    /// held at its own estimate, which is how an unreachable request shows up.
    pub fn profile_interval(
        &self,
        y: &DVector<f64>,
        reml: bool,
        quantity: Reported,
    ) -> Result<GxeInterval, &'static str> {
        let fit = self.fit(y, reml)?;
        let estimate = match quantity {
            Reported::Heritability { at } => fit.heritability_at(at),
            Reported::GeneticCorrelation { first, second } => {
                fit.genetic_correlation(first, second)
            }
        };
        if !estimate.is_finite() {
            return Err("GXE_QUANTITY_NOT_FINITE");
        }

        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        let scaled = y / variance.sqrt();
        let start = fit.parameters.clone();

        // **The maximum comes from the profile objective and not from the
        // fit's own log likelihood.** They differ by a constant that depends on
        // the scaling, and taking one from the other is what made this
        // package's bivariate intervals zero-width once already.
        let at_estimate = self
            .profile_objective(&scaled, reml, quantity, estimate, &start)
            .ok_or("GXE_QUANTITY_NOT_HELD_AT_ITS_OWN_ESTIMATE")?;
        let threshold = at_estimate - 0.5 * CHI2_ONE_95;

        let (floor, ceiling) = match quantity {
            Reported::Heritability { .. } => (1e-6, 1.0 - 1e-6),
            Reported::GeneticCorrelation { .. } => match self.surface {
                // The exponential kernel is positive at every rate.
                Surface::Exponential | Surface::PoweredExponential(_) => (1e-6, 1.0),
                Surface::RandomRegression => (-1.0, 1.0),
            },
        };
        // **A profile that could not be evaluated is not a likelihood that
        // fell away.** Counted as outside, a failure looked like ground the
        // data had ruled out and the bisection stepped inward, so the interval
        // came back narrower than the data support and said nothing about it.
        // A failure is covered instead, and counted.
        let failures = std::cell::Cell::new(0usize);
        let outside = |v: f64| match self.profile_objective(&scaled, reml, quantity, v, &start) {
            None => {
                failures.set(failures.get() + 1);
                false
            }
            Some(value) => value < threshold,
        };
        let (lower, lower_at_bound) = if outside(floor) {
            (bisect(floor, estimate, &outside), false)
        } else {
            (floor, true)
        };
        let (upper, upper_at_bound) = if outside(ceiling) {
            (bisect(ceiling, estimate, &outside), false)
        } else {
            (ceiling, true)
        };
        Ok(GxeInterval {
            estimate,
            lower,
            upper,
            lower_at_bound,
            upper_at_bound,
            level: 0.95,
            profile_failures: failures.get(),
        })
    }
}

/// Real non-negative roots of `a x^2 + b x + c`, linear case included.
fn solve_quadratic(a: f64, b: f64, c: f64) -> Option<Vec<f64>> {
    if a.abs() < 1e-14 {
        return (b.abs() > 1e-14).then(|| vec![-c / b]);
    }
    let discriminant = b * b - 4.0 * a * c;
    (discriminant >= 0.0).then(|| {
        let root = discriminant.sqrt();
        vec![(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)]
    })
}

/// Bisect between a point known to be outside the interval and one inside it.
fn bisect(mut out: f64, mut inside: f64, outside: &impl Fn(f64) -> bool) -> f64 {
    for _ in 0..60 {
        let middle = 0.5 * (out + inside);
        if outside(middle) {
            out = middle;
        } else {
            inside = middle;
        }
        if (out - inside).abs() < 1e-7 {
            break;
        }
    }
    0.5 * (out + inside)
}

fn to_theta(values: &[f64]) -> [f64; PARAMETERS] {
    let mut theta = [0.0; PARAMETERS];
    theta[..values.len()].copy_from_slice(values);
    theta
}

#[cfg(test)]
mod tests {
    use super::{GxeFit, GxeModel, Reported, Shape, Surface, block_from_loadings};
    use nalgebra::{DMatrix, DVector};

    const BOTH: [Surface; 3] = [
        Surface::Exponential,
        Surface::RandomRegression,
        // A shape away from one, so the powered kernel is exercised as
        // something other than a second copy of the exponential.
        Surface::PoweredExponential(Shape::Two),
    ];

    /// Sibling pairs, each person carrying an environment.
    ///
    /// The environment varies within a pair as well as between, or a surface
    /// could never be told from a plain heritability.
    pub(super) fn small(
        genetic: [f64; 3],
        residual: [f64; 3],
        seed: u64,
    ) -> (DMatrix<f64>, Vec<f64>, DMatrix<f64>, DVector<f64>) {
        let pairs = 250;
        let n = 2 * pairs;
        let mut a = DMatrix::<f64>::identity(n, n);
        for pair in 0..pairs {
            a[(2 * pair, 2 * pair + 1)] = 0.5;
            a[(2 * pair + 1, 2 * pair)] = 0.5;
        }
        let mut state = seed;
        let mut next = || {
            let mut total = 0.0;
            for _ in 0..3 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                total += ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            }
            total * 2.0
        };
        let z: Vec<f64> = (0..n).map(|_| next()).collect();

        // Simulated from the random-regression surface, which both forms can represent
        // closely enough to be recovered from.
        let mut covariance = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let (zi, zj) = (z[i], z[j]);
                covariance[(i, j)] =
                    a[(i, j)] * (genetic[0] + genetic[1] * (zi + zj) + genetic[2] * zi * zj);
            }
            covariance[(i, i)] +=
                residual[0] + 2.0 * residual[1] * z[i] + residual[2] * z[i] * z[i];
        }
        let factor = covariance
            .clone()
            .cholesky()
            .expect("the simulating covariance is positive definite")
            .l();
        let draw = DVector::from_iterator(n, (0..n).map(|_| next()));
        (
            a,
            z.clone(),
            DMatrix::from_element(n, 1, 1.0),
            factor * draw,
        )
    }

    /// **The loadings exist to keep the random-regression surface a
    /// covariance**, and it must do so including where the off-diagonal loading
    /// is large and negative -- the case a non-negativity constraint would
    /// wrongly forbid and a careless parameterisation would wrongly allow.
    #[test]
    fn every_loading_gives_a_positive_semidefinite_block() {
        for &(l00, l10, u) in &[
            (1.0, 0.0, 0.0),
            (1.0, -3.0, 0.0),
            (0.5, 2.0, 1.5),
            (0.0, 0.0, 0.0),
            (2.0, -0.4, 0.1),
        ] {
            let [q00, q01, q11] = block_from_loadings(l00, l10, u);
            assert!(q00 >= -1e-12 && q11 >= -1e-12);
            assert!(
                q00 * q11 - q01 * q01 >= -1e-12,
                "not a covariance at ({l00}, {l10}, {u})"
            );
        }
        // A negative covariance must be reachable, which is the whole reason
        // for not using non-negative variances.
        let [_, q01, _] = block_from_loadings(1.0, -0.5, 0.2);
        assert!(q01 < 0.0, "no negative covariance is reachable");
    }

    /// Both surfaces have their gradients checked against a central difference
    /// of the objective, in ML and REML. The exponential's derivatives run
    /// through an exponential and the random-regression one's through a Cholesky, and
    /// neither is the kind of thing to take on trust.
    #[test]
    fn the_gradient_matches_a_central_difference() {
        let (a, z, design, y) = small([0.5, 0.1, 0.3], [0.5, 0.0, 0.2], 7);
        for surface in BOTH {
            let model = GxeModel::build(surface, &a, &z, &design).expect("valid");
            let points: Vec<[f64; 6]> = match surface {
                Surface::Exponential | Surface::PoweredExponential(_) => vec![
                    [-0.7, 0.2, 0.3, -0.7, 0.1, 0.0],
                    [-0.5, -0.3, 0.8, -0.9, -0.2, 0.0],
                    [-0.7, 0.0, 0.05, -0.7, 0.0, 0.0],
                ],
                Surface::RandomRegression => vec![
                    [-0.36, 0.2, 0.55, -0.36, 0.1, 0.45],
                    [-0.51, -0.3, 0.63, -0.22, -0.2, 0.32],
                    [-0.69, 0.0, 0.71, -0.69, 0.0, 0.71],
                ],
            };
            for reml in [false, true] {
                for point in &points {
                    let at = model.evaluate(point, &y, reml, true).expect("evaluates");
                    for k in 0..surface.parameters() {
                        let step = 1e-6;
                        let mut up = *point;
                        let mut down = *point;
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
                            "{surface:?} reml={reml} parameter {k}: analytic {} against \
                             numeric {}",
                            at.gradient[k],
                            numeric
                        );
                    }
                }
            }
        }
    }

    /// An interaction that is there is found by both surfaces: the heritability
    /// changes across the environment and the genetic correlation between two
    /// environments falls below one.
    #[test]
    fn an_interaction_is_recovered_by_both_surfaces() {
        let (a, z, design, y) = small([0.4, 0.25, 0.35], [0.5, 0.0, 0.05], 20_260_813);
        for surface in BOTH {
            let model = GxeModel::build(surface, &a, &z, &design).expect("valid");
            let fit = model.fit(&y, true).expect("fits");
            assert!(
                fit.converged,
                "{surface:?} did not converge, |g| = {}",
                fit.scaled_gradient
            );
            let low = fit.heritability_at(-1.0);
            let high = fit.heritability_at(1.0);
            assert!(
                low.is_finite() && high.is_finite(),
                "{surface:?}: no heritability"
            );
            assert!(
                high > low,
                "{surface:?}: the genetic variance was simulated to grow with the \
                 environment but the heritability went from {low} to {high}"
            );
            let correlation = fit.genetic_correlation(-1.0, 1.0);
            assert!(
                correlation < 0.999,
                "{surface:?}: an interaction was simulated but the genetic \
                 correlation across environments is {correlation}"
            );
        }
    }

    /// With no interaction, neither surface should systematically find one.
    ///
    /// **This reads across seeds and not one**, because the estimate is bounded:
    /// a genetic correlation across environments cannot exceed one, so under the
    /// null every departure runs downward and a single draw can sit far below one
    /// by chance. On forty null draws the exponential surface has its tenth
    /// percentile at 0.25 and its smallest at nought, where the random-regression surface
    /// falls only to 0.74. The exponential decay saturates -- once the rate is
    /// large the likelihood is nearly flat in it, and noise carries the estimate a
    /// long way.
    ///
    /// So a reported correlation below one is not on its own evidence of an
    /// interaction, and the number to check here is not any one estimate. It is
    /// that the fit sits *exactly* at no interaction a good share of the time:
    /// that atom is what `interaction_test` puts its mixture reference on.
    #[test]
    fn no_interaction_is_invented_by_either_surface() {
        for surface in BOTH {
            let mut correlations = Vec::new();
            let mut tilts = Vec::new();
            for seed in 0..16u64 {
                let (a, z, design, y) = small([0.5, 0.0, 0.0], [0.5, 0.0, 0.0], 700 + seed);
                let fit = GxeModel::build(surface, &a, &z, &design)
                    .expect("valid")
                    .fit(&y, true)
                    .expect("fits");
                correlations.push(fit.genetic_correlation(-1.0, 1.0));
                tilts.push(fit.heritability_at(1.0) - fit.heritability_at(-1.0));
            }
            let at_boundary = correlations.iter().filter(|c| **c > 0.999).count();
            assert!(
                at_boundary * 4 >= correlations.len(),
                "{surface:?}: only {at_boundary} of {} null fits sat at no \
                 interaction, so the boundary atom the test's reference rests on \
                 is not there",
                correlations.len()
            );
            tilts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let middle = tilts[tilts.len() / 2];
            assert!(
                middle.abs() < 0.1,
                "{surface:?}: no interaction was simulated but the heritability \
                 runs {middle:+.3} across the environment in the typical fit"
            );
        }
    }

    /// **A crossover is representable by one surface and not the other**, which
    /// is the sharpest difference between them and the one most likely to
    /// decide which to use. A crossover is a genotype that helps in one
    /// environment and harms in another: a genetic correlation below nought,
    /// not merely below one. The exponential surface correlates two
    /// environments as `exp(-lambda |difference|)`, which is positive whatever
    /// the rate, so it cannot reach one however strong the crossover in the
    /// data. The random-regression surface can, because its covariance is
    /// `(l00 + l10 z1)(l00 + l10 z2) + l11^2 z1 z2`, which changes sign when the
    /// two environments fall either side of `-l00 / l10`.
    #[test]
    fn only_random_regression_can_cross_over() {
        let made = |surface: Surface, parameters: Vec<f64>| GxeFit {
            surface,
            parameters,
            fixed_effects: vec![],
            fixed_effect_errors: vec![],
            loglik: 0.0,
            converged: true,
            scaled_gradient: 0.0,
            estimator: "reml",
            variance_scale: 1.0,
        };
        // intercept 0.5, linear 1.0, orthogonal 0: the sign changes at z = -0.5.
        let regression = made(
            Surface::RandomRegression,
            vec![0.5_f64.ln(), 1.0, 0.0, 0.7_f64.ln(), 0.0, 0.0],
        );
        let crossed = regression.genetic_correlation(-1.5, 1.5);
        assert!(
            crossed < -0.9,
            "the random-regression surface should reach a crossover, and got {crossed}"
        );
        // No rate, however large, takes the exponential surface below nought.
        for rate in [0.0, 0.5, 5.0, 50.0] {
            let exponential = made(Surface::Exponential, vec![-0.7, 0.6, rate, -0.7, 0.0]);
            let got = exponential.genetic_correlation(-1.5, 1.5);
            assert!(
                got >= 0.0,
                "the exponential surface reached {got} at rate {rate}, which its \
                 kernel cannot do"
            );
        }
    }

    /// The two tests answer different questions and must not answer each
    /// other's. A genetic variance that changes with the environment while the
    /// genetic correlation stays at one is a change of scale: `interaction_test`
    /// should see it and `correlation_test` should not.
    #[test]
    fn the_two_tests_separate_scale_from_reordering() {
        // A rank-one genetic surface: the variance grows with the environment,
        // but q00*q11 = q01^2, so every pair of environments correlates at one.
        let (a, z, design, y) = small([0.36, 0.30, 0.25], [0.5, 0.0, 0.0], 4242);
        for surface in BOTH {
            let model = GxeModel::build(surface, &a, &z, &design).expect("valid");
            let any = model.interaction_test(&y, true).expect("tests");
            let reordering = model.correlation_test(&y, true).expect("tests");
            assert!(
                any.statistic >= reordering.statistic - 1e-6,
                "{surface:?}: the wider null cannot fit better than the narrower \
                 one it contains ({} against {})",
                any.statistic,
                reordering.statistic
            );
            assert!(
                any.p_value < 0.05,
                "{surface:?}: the genetic variance was simulated to change with the \
                 environment but the interaction test gives p = {}",
                any.p_value
            );
            assert!(
                reordering.p_value > 0.05,
                "{surface:?}: the genetic correlation was simulated at one across \
                 environments but the correlation test gives p = {}",
                reordering.p_value
            );
        }
    }

    /// An interval has to contain its own estimate and be narrower than the
    /// whole range, or it is not saying anything. **The first is not automatic
    /// here**: the maximum is taken from the profile objective rather than from
    /// the fit's log likelihood, because those differ by a scaling constant,
    /// and taking one from the other is what made this package's bivariate
    /// intervals zero-width once already.
    #[test]
    fn an_interval_contains_its_estimate_and_says_something() {
        let (a, z, design, y) = small([0.4, 0.15, 0.30], [0.5, 0.0, 0.05], 31);
        for surface in BOTH {
            let model = GxeModel::build(surface, &a, &z, &design).expect("valid");
            for quantity in [
                Reported::Heritability { at: 0.0 },
                Reported::Heritability { at: 1.0 },
                Reported::GeneticCorrelation {
                    first: -1.0,
                    second: 1.0,
                },
            ] {
                let got = model
                    .profile_interval(&y, true, quantity)
                    .unwrap_or_else(|e| panic!("{surface:?} {quantity:?}: {e}"));
                assert!(
                    got.lower <= got.estimate + 1e-6 && got.estimate <= got.upper + 1e-6,
                    "{surface:?} {quantity:?}: [{}, {}] does not contain {}",
                    got.lower,
                    got.upper,
                    got.estimate
                );
                assert!(
                    got.upper - got.lower > 1e-4,
                    "{surface:?} {quantity:?}: the interval is {} wide, which is the \
                     zero-width fault returning",
                    got.upper - got.lower
                );
                assert!(
                    got.upper - got.lower < 1.999,
                    "{surface:?} {quantity:?}: [{}, {}] is the whole range and says \
                     nothing",
                    got.lower,
                    got.upper
                );
            }
        }
    }

    /// The two surfaces carry different numbers of parameters, and the fit says
    /// which it used rather than leaving the caller to infer it from a length.
    #[test]
    fn a_fit_says_which_surface_made_it() {
        let (a, z, design, y) = small([0.5, 0.0, 0.1], [0.5, 0.0, 0.0], 5);
        for surface in BOTH {
            let fit = GxeModel::build(surface, &a, &z, &design)
                .expect("valid")
                .fit(&y, true)
                .expect("fits");
            assert_eq!(fit.surface, surface);
            assert_eq!(fit.parameters.len(), surface.parameters());
        }
    }
}

#[cfg(feature = "python")]
pub mod python;

#[cfg(test)]
mod against_the_source {
    use super::{GxeFit, Shape, Surface};

    /// The powered-exponential surface, at all four fixed shapes, against the
    /// source-fidelity crate's own numbers. **The shape one row is the same
    /// covariance as the recovered exponential**, which is the source's
    /// statement about its own model and is checked here rather than believed.
    #[test]
    fn the_powered_surface_matches_the_equation_at_every_fixed_shape() {
        let k = [
            [1.0, 0.5, 0.25, 0.0],
            [0.5, 1.0, 0.25, 0.0],
            [0.25, 0.25, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let e = [-1.3, 0.4, 2.1, 0.9];
        // Only the off-diagonal genetic cells move with the shape; the
        // variances do not involve the kernel at all.
        let source: [(Shape, [f64; 3]); 4] = [
            (
                Shape::Half,
                [0.175_961_143_060, 0.105_193_442_188, 0.159_511_498_739],
            ),
            (
                Shape::One,
                [0.161_274_534_884, 0.074_698_567_695, 0.146_197_861_190],
            ),
            (
                Shape::ThreeHalves,
                [0.143_950_826_485, 0.039_734_390_297, 0.130_493_651_485],
            ),
            (
                Shape::Two,
                [0.124_127_355_425, 0.012_407_001_097, 0.112_523_368_251],
            ),
        ];
        for (shape, wanted) in source {
            let fit = GxeFit {
                surface: Surface::PoweredExponential(shape),
                parameters: vec![-0.6, 0.35, 0.22, -0.4, -0.15],
                fixed_effects: vec![],
                fixed_effect_errors: vec![],
                loglik: 0.0,
                converged: true,
                scaled_gradient: 0.0,
                estimator: "reml",
                variance_scale: 1.0,
            };
            for (slot, &(i, j)) in [(0usize, 1usize), (0, 2), (1, 2)].iter().enumerate() {
                let mine = k[i][j] * fit.genetic_covariance(e[i], e[j]);
                assert!(
                    (mine - wanted[slot]).abs() < 1e-11,
                    "{shape:?} ({i}, {j}): this package gives {mine}, the source \
                     gives {}",
                    wanted[slot]
                );
            }
        }
    }

    /// Cross-sectionally this surface is the one-record-per-person reduction of
    /// the longitudinal random-regression covariance
    /// `K_rs b(z_r)' Sigma_G b(z_s) + tau_P^2 J_rs + I_rs E(z_r)`, and with one
    /// record per person the same-person indicator is the identity, so that
    /// term is a constant absorbed by the residual. Setting it to nought should
    /// leave exactly this surface, and this test checks that algebra.
    ///
    /// Its two-basis block is
    /// `(intercept + linear l)(intercept + linear r) + orthogonal^2 l r`, which
    /// is this package's block with `u = orthogonal^2` -- arrived at
    /// independently here, for the unrelated reason that squaring the
    /// coordinate away keeps the derivative from vanishing at the null.
    #[test]
    fn the_random_regression_surface_reproduces_the_proposed_one() {
        let k = [
            [1.0, 0.5, 0.25, 0.0],
            [0.5, 1.0, 0.25, 0.0],
            [0.25, 0.25, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let z = [-1.3, 0.4, 2.1, 0.9];
        // Their loadings are (intercept, linear, orthogonal) and ours are the
        // same three with the intercept carried as its logarithm, which is the
        // custom implementation's own convention.
        let fit = GxeFit {
            surface: Surface::RandomRegression,
            parameters: vec![0.8_f64.ln(), -0.35, 0.5, 0.7_f64.ln(), 0.2, 0.3],
            fixed_effects: vec![],
            fixed_effect_errors: vec![],
            loglik: 0.0,
            converged: true,
            scaled_gradient: 0.0,
            estimator: "reml",
            variance_scale: 1.0,
        };
        // Printed by the source-fidelity crate with no permanent-person
        // variance and no protocol adjustment.
        let proposed = [
            [2.343_225_0, 0.349_150_0, -0.150_231_25, 0.0],
            [0.349_150_0, 1.098_400_0, 0.063_225_0, 0.0],
            [-0.150_231_25, 0.063_225_0, 2.758_025_0, 0.0],
            [0.0, 0.0, 0.0, 1.285_025_0],
        ];
        for i in 0..4 {
            for j in 0..4 {
                let mut mine = k[i][j] * fit.genetic_covariance(z[i], z[j]);
                if i == j {
                    mine += fit.residual_variance_at(z[i]);
                }
                assert!(
                    (mine - proposed[i][j]).abs() < 1e-9,
                    "({i}, {j}): this package gives {mine}, the proposed model \
                     gives {}",
                    proposed[i][j]
                );
            }
        }
    }

    /// **Is the exponential surface the recovered SOLAR model, or something
    /// like it?** The source-fidelity crate beside this one preserves the Tcl
    /// continuous-environment covariance
    /// `K_ij sqrt(VG_i VG_j) exp(-lambda_g |e_i - e_j|) + I_ij VE_i`, with the
    /// two variances log-linear in the environment. This reproduces its output
    /// on a fixed input rather than arguing from the algebra.
    #[test]
    fn the_exponential_surface_reproduces_the_recovered_solar_covariance() {
        let k = [
            [1.0, 0.5, 0.25, 0.0],
            [0.5, 1.0, 0.25, 0.0],
            [0.25, 0.25, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let e = [-1.3, 0.4, 2.1, 0.9];
        // alpha_g, gamma_g, lambda_g, alpha_e, gamma_e -- the same five.
        let fit = GxeFit {
            surface: Surface::Exponential,
            parameters: vec![-0.6, 0.35, 0.22, -0.4, -0.15],
            fixed_effects: vec![],
            fixed_effect_errors: vec![],
            loglik: 0.0,
            converged: true,
            scaled_gradient: 0.0,
            estimator: "reml",
            variance_scale: 1.0,
        };
        // Printed by the source-fidelity crate at these inputs, centre = 0.
        let source = [
            [1.162_839_743_718, 0.161_274_534_884, 0.074_698_567_695, 0.0],
            [0.161_274_534_884, 1.262_567_291_014, 0.146_197_861_190, 0.0],
            [0.074_698_567_695, 0.146_197_861_190, 1.633_728_896_148, 0.0],
            [0.0, 0.0, 0.0, 1.337_683_544_464],
        ];
        for i in 0..4 {
            for j in 0..4 {
                let mut mine = k[i][j] * fit.genetic_covariance(e[i], e[j]);
                if i == j {
                    mine += fit.residual_variance_at(e[i]);
                }
                assert!(
                    (mine - source[i][j]).abs() < 1e-11,
                    "({i}, {j}): this package gives {mine}, the recovered source \
                     gives {}",
                    source[i][j]
                );
            }
        }
    }
}
