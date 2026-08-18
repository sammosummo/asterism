//! One trait, any number of variance components.
//!
//! `prepared.rs` fits a single structured component plus residual, and owes its
//! speed to diagonalising the relationship matrix once per family block so that
//! every likelihood evaluation is a pass over a diagonal. **That saving does not
//! survive a second component.** Two structured matrices share no eigenbasis and
//! cannot be diagonalised together, so each evaluation goes back to factorising
//! a dense covariance.
//!
//! Once that is given up, the cost of two components and of five is the same
//! order: a Cholesky per block either way, with one trace term per component.
//! So this module takes an arbitrary list rather than hard-coding household,
//! which would have been the same work for less. `prepared.rs` stays where it
//! is and stays the right thing to use for one component.
//!
//! The model is
//!
//! ```text
//! y = Xβ + Σ_j u_j + e,    V = Σ_j σ²_j K_j + σ²_e I
//! ```
//!
//! with every variance at or above nought. The fixed effects are profiled out,
//! and REML differs from ML by the usual `log|X'V⁻¹X|` term and by dropping
//! `p` from the count multiplying `log 2π`.

use nalgebra::{DMatrix, DVector};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::blocks::family_blocks;
use crate::deviance::{chi2_one_df_upper_tail, chi2_upper_tail};

/// A fitted multi-component model.
#[derive(Clone, Debug)]
pub struct ComponentFit {
    /// One variance per structured component, in the order they were given,
    /// with the residual last.
    pub variances: Vec<f64>,
    /// Each raw covariance coefficient divided by their sum. These proportions
    /// depend on the submitted matrix scales and are not generally variance
    /// shares or heritabilities for unnormalised matrices.
    pub proportions: Vec<f64>,
    /// Sum of the raw covariance coefficients. This depends on matrix scaling.
    pub coefficient_total: f64,
    pub fixed_effects: Vec<f64>,
    /// The standard error of each fixed effect, the square root of the diagonal
    /// of `(X' V^-1 X)^-1` at the fitted variances.
    ///
    /// **This is the usual approximation and it is worth saying so.** It treats
    /// the variance components as known when they were estimated from the same
    /// data, so it is a little optimistic; the size of that depends on how well
    /// the components are determined, and it is the same approximation the
    /// one-trait model and SOLAR both make.
    pub fixed_effect_errors: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    pub scaled_gradient: f64,
    /// Why the search stopped, in the search's own words rather than ours.
    /// Nought means one of its tolerances fired; one means it ran out of
    /// iterations; anything else is an error inside the optimiser.
    ///
    /// **This is not the same question as `converged`.** The search stops on
    /// `factr`, a relative reduction in the objective, and `converged` is
    /// decided afterwards on the recomputed projected gradient. Down a long
    /// flat valley the objective settles well before the gradient does, so a
    /// fit can stop cleanly here and still be reported as not converged. Read
    /// the two together: a fit that ran out of iterations and a fit whose
    /// objective settled early are different faults wanting different answers,
    /// and without this they cannot be told apart.
    pub stop_code: i32,
    /// The optimiser's own message for `stop_code`, kept verbatim.
    pub stop_message: String,
    /// True where the gradient test failed on the first search and a second was
    /// run from that point with the objective tolerance switched off. It fires
    /// nowhere else, so a fit reporting `false` here is the fit the package
    /// gave before the polish existed, to the last bit.
    pub polished: bool,
    pub estimator: &'static str,
}

/// The best start so far, and what the search said when it stopped there.
struct Best {
    negative_loglik: f64,
    par: Vec<f64>,
    fixed_effects: Vec<f64>,
    converged: bool,
    scaled_gradient: f64,
    stop_code: i32,
    stop_message: String,
}

struct Evaluation {
    negative_loglik: f64,
    gradient: Vec<f64>,
    fixed_effects: Vec<f64>,
    /// `(X' V^-1 X)^-1`, which is the covariance of the fixed effects and is
    /// wanted for their standard errors. Kept only where it was formed.
    fixed_covariance: Option<DMatrix<f64>>,
}

/// Is a component resting on its lower bound of nought?
///
/// **This cannot be an exact test.** A bounded search leaves a component on its
/// bound a hair above it rather than at it -- measured at 1e-37, 1e-17 and
/// 1e-19 on ordinary problems -- so `value <= 0.0` never fires, the one-sided
/// derivative is counted in full, and a fit that is demonstrably the maximum
/// reports itself unconverged. On a unit-variance scale anything below this is
/// nought in every sense that matters.
pub(crate) fn resting_on_zero(value: f64) -> bool {
    value <= RESTING_TOLERANCE
}

const RESTING_TOLERANCE: f64 = 1e-9;

struct BlockSolve {
    rows: Vec<usize>,
    inverse: DMatrix<f64>,
    vr: DVector<f64>,
    vx: DMatrix<f64>,
}

/// One trait, several components.
pub struct ComponentModel {
    design: DMatrix<f64>,
    /// The structured components. The residual identity is implicit and always
    /// last in the parameter vector; it is not stored.
    matrices: Vec<DMatrix<f64>>,
    blocks: Vec<Vec<usize>>,
    rows: usize,
    logdet_xtx: f64,
}

impl ComponentModel {
    /// Validate and prepare. `matrices` are the structured components; the
    /// residual is added for you.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(matrices: &[DMatrix<f64>], design: &DMatrix<f64>) -> Result<Self, &'static str> {
        if matrices.is_empty() {
            return Err("COMPONENTS_NONE_GIVEN");
        }
        let n = design.nrows();
        if n == 0 {
            return Err("COMPONENTS_NO_ROWS");
        }
        for matrix in matrices {
            if matrix.nrows() != n || matrix.ncols() != n {
                return Err("COMPONENTS_MATRIX_WRONG_SIZE");
            }
            for i in 0..n {
                for j in 0..i {
                    if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-10 {
                        return Err("COMPONENTS_MATRIX_NOT_SYMMETRIC");
                    }
                }
            }
        }

        // **Blocks come from every matrix at once, not from the first.** Two
        // people unrelated in the pedigree may still share a household, and
        // blocking on the relationship alone would put them in different blocks
        // and silently drop the covariance between them. The likelihood would
        // still be a likelihood; it would be the likelihood of a different
        // model. So the pattern is the union.
        let mut union = DMatrix::<f64>::zeros(n, n);
        for matrix in matrices {
            for i in 0..n {
                for j in 0..n {
                    if matrix[(i, j)] != 0.0 {
                        union[(i, j)] = 1.0;
                    }
                }
            }
        }
        let blocks = family_blocks(&union);

        let xtx = design.transpose() * design;
        let chol = xtx.cholesky().ok_or("COMPONENTS_DESIGN_RANK_DEFICIENT")?;
        let logdet_xtx = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();

        Ok(Self {
            design: design.clone(),
            matrices: matrices.to_vec(),
            blocks,
            rows: n,
            logdet_xtx,
        })
    }

    /// How many variances there are: one per structured component plus residual.
    #[must_use]
    pub fn parameters(&self) -> usize {
        self.matrices.len() + 1
    }

    /// The covariance of one block at these variances.
    fn assemble(&self, block: &[usize], theta: &[f64]) -> DMatrix<f64> {
        let size = block.len();
        let mut v = DMatrix::<f64>::zeros(size, size);
        for (component, matrix) in self.matrices.iter().enumerate() {
            let variance = theta[component];
            if variance == 0.0 {
                continue;
            }
            for i in 0..size {
                for j in 0..size {
                    v[(i, j)] += variance * matrix[(block[i], block[j])];
                }
            }
        }
        let residual = theta[self.matrices.len()];
        for i in 0..size {
            v[(i, i)] += residual;
        }
        v
    }

    /// The derivative of a block's covariance with respect to one variance,
    /// which is just that component's own matrix.
    fn derivative(&self, block: &[usize], component: usize) -> DMatrix<f64> {
        let size = block.len();
        if component == self.matrices.len() {
            return DMatrix::identity(size, size);
        }
        let matrix = &self.matrices[component];
        DMatrix::from_fn(size, size, |i, j| matrix[(block[i], block[j])])
    }

    fn evaluate(
        &self,
        theta: &[f64],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
    ) -> Option<Evaluation> {
        self.evaluate_with_signs(theta, y, reml, want_gradient, &[])
    }

    /// Evaluate, with some coordinates permitted to be negative.
    ///
    /// **A variance below nought is not a model and is refused.** A *deviation*
    /// below nought is an ordinary thing to want, and refusing it here made the
    /// signed fit inert: the search was free to ask for a negative deviation
    /// and the likelihood answered "not a model" every time, so it could only
    /// ever move one way. That is why the first contrast results came back with
    /// forty estimates and not one negative among them.
    ///
    /// What still cannot happen is a covariance that is not one. Nothing checks
    /// that here, and nothing needs to: the factorisation below fails on it and
    /// the search reads the failure as somewhere not to go.
    fn evaluate_with_signs(
        &self,
        theta: &[f64],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
        signed: &[usize],
    ) -> Option<Evaluation> {
        if theta.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if theta
            .iter()
            .enumerate()
            .any(|(k, v)| *v < 0.0 && !signed.contains(&k))
        {
            return None;
        }
        // Everything at nought is not a covariance, and no amount of searching
        // makes it one.
        if theta.iter().all(|v| *v <= 0.0) {
            return None;
        }

        let p = self.design.ncols();
        let mut logdet = 0.0;
        let mut xvx = DMatrix::<f64>::zeros(p, p);
        let mut xvy = DVector::<f64>::zeros(p);
        let mut yvy = 0.0;
        let mut kept: Vec<BlockSolve> = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            let size = block.len();
            let v = self.assemble(block, theta);
            let factor = crate::dense::DenseFactor::new(&v)?;
            logdet += factor.logdet();

            let yb = DVector::from_iterator(size, block.iter().map(|&i| y[i]));
            let xb = DMatrix::from_fn(size, p, |r, c| self.design[(block[r], c)]);
            let vy = factor.solve_vector(&yb);
            let vx = factor.solve_matrix(&xb);
            xvx += xb.transpose() * &vx;
            xvy += xb.transpose() * &vy;
            yvy += yb.dot(&vy);

            if want_gradient {
                kept.push(BlockSolve {
                    rows: block.clone(),
                    inverse: factor.inverse(),
                    vr: vy,
                    vx,
                });
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

        let mut gradient = vec![0.0; self.parameters()];
        if want_gradient {
            let xvx_inverse = xvx_chol.inverse();
            // V⁻¹ times the residual, block by block. `vr` currently holds
            // V⁻¹y; subtracting V⁻¹Xβ turns it into V⁻¹(y − Xβ), which is what
            // every derivative below wants.
            let mut residual_solves: Vec<DVector<f64>> = Vec::with_capacity(kept.len());
            for solve in &kept {
                residual_solves.push(&solve.vr - &solve.vx * &beta);
            }

            for component in 0..self.parameters() {
                let mut trace = 0.0;
                let mut quadratic_term = 0.0;
                let mut restricted = DMatrix::<f64>::zeros(p, p);
                for (slot, solve) in kept.iter().enumerate() {
                    let dv = self.derivative(&solve.rows, component);
                    let size = solve.rows.len();
                    // tr(V⁻¹ ∂V), which needs the inverse rather than a solve.
                    for i in 0..size {
                        for j in 0..size {
                            trace += solve.inverse[(i, j)] * dv[(j, i)];
                        }
                    }
                    let r = &residual_solves[slot];
                    let dvr = &dv * r;
                    quadratic_term += r.dot(&dvr);
                    if reml {
                        restricted += solve.vx.transpose() * (&dv * &solve.vx);
                    }
                }
                let mut d = 0.5 * (trace - quadratic_term);
                if reml {
                    d -= 0.5 * (&xvx_inverse * &restricted).trace();
                }
                gradient[component] = d;
            }
        }

        Some(Evaluation {
            negative_loglik: value,
            gradient,
            fixed_effects: beta.iter().copied().collect(),
            fixed_covariance: Some(xvx_chol.inverse()),
        })
    }

    /// The objective and its gradient, for the optimiser comparison in
    /// `tests/optimiser_race.rs`.
    ///
    /// Every fit in this package goes through one bound-constrained optimiser,
    /// so which one is not a detail to be settled by a commit message. This seam
    /// exists so the two candidates can be raced on a real objective rather than
    /// on a textbook function.
    #[doc(hidden)]
    #[must_use]
    pub fn objective_for_test(
        &self,
        theta: &[f64],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
    ) -> Option<(f64, Vec<f64>)> {
        self.evaluate(theta, y, reml, want_gradient)
            .map(|e| (e.negative_loglik, e.gradient))
    }

    /// Fit by bounded search from several starts.
    ///
    /// The response is scaled to unit variance first and the variances scaled
    /// back afterwards. That keeps the search well conditioned whatever units
    /// the trait came in, and the likelihood shifts by a constant that is added
    /// back, so nothing about the answer depends on it.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start reached a usable optimum.
    pub fn fit(&self, y: &DVector<f64>, reml: bool) -> Result<ComponentFit, &'static str> {
        self.fit_with_signs(y, reml, &[])
    }

    /// Fit with some coordinates free to take either sign.
    ///
    /// **Every variance is bounded below at nought, and that is right until it
    /// is not.** A model written as a baseline plus per-class differences needs
    /// those differences signed: a class carrying *less* than the baseline is
    /// exactly as meaningful as one carrying more, and bounding the difference
    /// at nought would make one direction unreachable and the other
    /// unfalsifiable.
    ///
    /// Nothing else guards positive definiteness once a coordinate can go
    /// negative. The covariance is factorised on every evaluation anyway, so a
    /// combination that is not a covariance fails there and the search reads it
    /// as somewhere not to go -- which is the same way every other infeasible
    /// point in this package is handled.
    ///
    /// `signed` lists the coordinates that may go negative. An empty list is
    /// the ordinary fit.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the response is the wrong length or
    /// constant, or where no start reached a usable optimum.
    pub fn fit_with_signs(
        &self,
        y: &DVector<f64>,
        reml: bool,
        signed: &[usize],
    ) -> Result<ComponentFit, &'static str> {
        self.fit_general(y, reml, signed, &[])
    }

    /// Fit with some coordinates signed and some pinned at a value.
    ///
    /// Pinned values are in the response's own units, like everything a caller
    /// sees; the search runs on a standardised response, so they are converted
    /// here rather than by whoever calls this. A coordinate is pinned by making
    /// its two bounds equal, which is how every other held fit in this package
    /// works and keeps one code path for the free and the constrained case.
    fn fit_general(
        &self,
        y: &DVector<f64>,
        reml: bool,
        signed: &[usize],
        pinned: &[(usize, f64)],
    ) -> Result<ComponentFit, &'static str> {
        if y.len() != self.rows {
            return Err("COMPONENTS_RESPONSE_WRONG_LENGTH");
        }
        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        if !(variance > 0.0) {
            return Err("COMPONENTS_RESPONSE_CONSTANT");
        }
        let scale = variance.sqrt();
        let scaled = y / scale;

        let count = self.parameters();
        let mut lower = vec![0.0; count];
        let upper = vec![f64::INFINITY; count];
        let mut upper = upper;
        for &k in signed {
            if k >= count {
                return Err("COMPONENTS_SIGNED_INDEX_OUT_OF_RANGE");
            }
            // **Finite, not negative infinity.** The bounded optimiser will not
            // take an infinite bound, and silently declines every start when
            // given one, which shows up as nothing converging rather than as an
            // error. The response is scaled to unit variance before the search,
            // so a deviation of a thousand is far outside anything meaningful
            // and serves as no bound at all in practice.
            lower[k] = -1.0e3;
        }
        for &(k, value) in pinned {
            if k >= count {
                return Err("COMPONENTS_PINNED_INDEX_OUT_OF_RANGE");
            }
            // The search sees a response scaled to unit variance.
            let scaled_value = value / variance;
            lower[k] = scaled_value;
            upper[k] = scaled_value;
        }

        // Starts: everything equal, residual-heavy, and each component in turn
        // carrying most of the variance. A component that is genuinely zero is
        // found from any of them; one that is large is not always found from a
        // start that puts it near nought.
        //
        // **A signed coordinate needs its own start and will not converge from
        // these.** They put every coordinate at a positive proportion, which is
        // right for a variance and wrong for a deviation: the matrix behind a
        // signed coordinate is a difference of two others and is not positive
        // semidefinite, so a positive start pushes the covariance somewhere it
        // cannot go and the search stalls. Two fits of the same model reached
        // log likelihoods fifteen apart before this was added. The natural
        // start is every deviation at nought -- nothing differs -- with the
        // unsigned coordinates carrying the variance.
        let mut starts: Vec<Vec<f64>> = Vec::new();
        if !signed.is_empty() {
            let unsigned = count - signed.len();
            let proportion = if unsigned > 0 {
                1.0 / unsigned as f64
            } else {
                0.0
            };
            let mut start = vec![proportion; count];
            for &k in signed {
                start[k] = 0.0;
            }
            starts.push(start);
        }
        starts.extend(vec![vec![1.0 / count as f64; count], {
            let mut s = vec![0.1 / count as f64; count];
            s[count - 1] = 0.9;
            s
        }]);
        for component in 0..count - 1 {
            let mut s = vec![0.1 / count as f64; count];
            s[component] = 0.6;
            s[count - 1] = 0.3;
            starts.push(s);
        }

        let value_of = |candidate: &[f64]| -> f64 {
            self.evaluate_with_signs(candidate, &scaled, reml, false, signed)
                .map_or(1e30, |e| e.negative_loglik)
        };
        let gradient_of = |candidate: &[f64]| -> Vec<f64> {
            self.evaluate_with_signs(candidate, &scaled, reml, true, signed)
                .map_or_else(|| vec![0.0; count], |e| e.gradient)
        };
        // The convergence test is the projected gradient recomputed here, not
        // the optimiser's own word for it: a component resting on nought has a
        // one-sided derivative and is converged when that derivative pushes
        // outward, which an unprojected norm calls a failure.
        let reading = |at: &Evaluation, par: &[f64]| -> f64 {
            let projected = at
                .gradient
                .iter()
                .enumerate()
                .map(|(k, g)| {
                    if signed.contains(&k) {
                        // No bound to rest on, so no projection.
                        *g
                    } else if resting_on_zero(par[k]) {
                        g.min(0.0)
                    } else {
                        *g
                    }
                })
                .fold(0.0f64, |worst, g| worst.max(g.abs()));
            projected / at.negative_loglik.abs().max(1.0)
        };

        let mut best: Option<Best> = None;
        for start in starts {
            let Ok(bounds) = Bounds::new(lower.clone(), upper.clone()) else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(count);
            control.maxit = 500;
            control.fnscale = value_of(&start).abs().max(1.0);
            control.parscale = vec![1.0; count];
            // R's own default: stop once the objective has settled to about
            // 1e-9 relative. At nought the search ran to `maxit` every time,
            // factorising a dense covariance per evaluation for nothing.
            control.factr = 1.0e3;
            control.pgtol = 1e-9;
            control.lmm = count.min(10);
            let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, &value_of, &gradient_of, control)
            else {
                continue;
            };
            let Some(at) = self.evaluate_with_signs(&solution.par, &scaled, reml, true, signed)
            else {
                continue;
            };
            if !at.negative_loglik.is_finite() {
                continue;
            }
            let scaled_gradient = reading(&at, &solution.par);
            if best
                .as_ref()
                .is_none_or(|b| at.negative_loglik < b.negative_loglik)
            {
                best = Some(Best {
                    negative_loglik: at.negative_loglik,
                    par: solution.par.clone(),
                    fixed_effects: at.fixed_effects.clone(),
                    converged: scaled_gradient < 1e-7,
                    scaled_gradient,
                    // Kept rather than discarded. The search's own verdict is
                    // the only evidence that separates a fit which ran out of
                    // iterations from one whose objective settled early, and
                    // nothing outside the fit can recover it.
                    stop_code: solution.convergence,
                    stop_message: solution.message.clone(),
                });
            }
        }

        // Where the gradient test fails, search once more from the point
        // already found with the objective tolerance switched off.
        //
        // **This is a measurement rather than a guess.** Every one of the eight
        // red deer fits stops with `REL_REDUCTION_OF_F <= FACTR*EPSMCH` and not
        // one stops on `pgtol`, so the gradient the test reads is never a
        // criterion the search pursued; the fits that pass do so because their
        // gradient happened already to be small when the objective settled.
        // Taking `factr` out leaves `pgtol` to decide, and `pgtol` is a
        // relative-gradient test at `1e-9` -- a hundredfold tighter than the
        // `1e-7` the flag asks for -- so a search that stops on it passes
        // comfortably.
        //
        // It fires only where the flag already reports failure, so every fit
        // that passes today is untouched to the last bit and no calibration
        // moves. The extra budget is small and the result is kept only if it
        // is better on both counts, so the worst case is the cost of 200 more
        // iterations and the same answer as before.
        //
        // **The returned point is checked rather than trusted, and on this
        // problem it has to be.** All three polished deer fits end with
        // `ERROR: ABNORMAL_TERMINATION_IN_LNSRCH` -- L-BFGS-B saying its line
        // search can make no further progress in double precision -- and none
        // reaches `pgtol`. Two of the three pass the gradient test anyway and
        // the third improves threefold, so the abnormal ending is the shape of
        // the likelihood rather than a fault. Accepting only a point that is no
        // worse on the objective and strictly better on the gradient is what
        // makes taking a result from an errored search safe: it either improves
        // or it is discarded.
        let mut polished = false;
        if let Some(current) = best.as_ref()
            && !current.converged
            && let Ok(bounds) = Bounds::new(lower.clone(), upper.clone())
        {
            let mut control = OptimControl::default_for_dimension(count);
            control.maxit = 200;
            control.fnscale = value_of(&current.par).abs().max(1.0);
            control.parscale = vec![1.0; count];
            control.factr = 0.0;
            control.pgtol = 1e-9;
            control.lmm = count.min(10);
            if let Ok(again) = optim_lbfgsb_with_gradient(
                current.par.clone(),
                bounds,
                &value_of,
                &gradient_of,
                control,
            ) && let Some(at) = self.evaluate_with_signs(&again.par, &scaled, reml, true, signed)
                && at.negative_loglik.is_finite()
                // Not worse on the objective and better on the gradient. A
                // longer search that wandered is refused rather than reported.
                && at.negative_loglik <= current.negative_loglik
                && reading(&at, &again.par) < current.scaled_gradient
            {
                let scaled_gradient = reading(&at, &again.par);
                polished = true;
                best = Some(Best {
                    negative_loglik: at.negative_loglik,
                    par: again.par.clone(),
                    fixed_effects: at.fixed_effects.clone(),
                    converged: scaled_gradient < 1e-7,
                    scaled_gradient,
                    stop_code: again.convergence,
                    stop_message: again.message.clone(),
                });
            }
        }

        let Best {
            negative_loglik: negative,
            par,
            fixed_effects: beta,
            converged,
            scaled_gradient: _,
            stop_code,
            stop_message,
        } = best.ok_or("COMPONENTS_NO_START_CONVERGED")?;
        let at = self
            .evaluate_with_signs(&par, &scaled, reml, true, signed)
            .ok_or("COMPONENTS_OPTIMUM_NOT_EVALUABLE")?;
        let projected = at
            .gradient
            .iter()
            .enumerate()
            .map(|(k, g)| {
                if resting_on_zero(par[k]) {
                    g.min(0.0)
                } else {
                    *g
                }
            })
            .fold(0.0f64, |worst, g| worst.max(g.abs()));

        // Back to the response's own units. Variances carry the square of the
        // scale; the log-likelihood carries a term per observation.
        let variances: Vec<f64> = par.iter().map(|v| v * variance).collect();
        let total: f64 = variances.iter().sum();
        let proportions = variances.iter().map(|v| v / total).collect();
        let observations = if reml {
            (self.rows - self.design.ncols()) as f64
        } else {
            self.rows as f64
        };
        let loglik = -negative - observations * scale.ln();

        // The standard errors scale with the response like the effects do,
        // being square roots of a variance in the response's own units.
        let errors = at
            .fixed_covariance
            .as_ref()
            .map(|c| {
                (0..c.nrows())
                    .map(|i| (c[(i, i)].max(0.0)).sqrt() * scale)
                    .collect()
            })
            .unwrap_or_default();
        Ok(ComponentFit {
            variances,
            proportions,
            coefficient_total: total,
            fixed_effects: beta.iter().map(|b| b * scale).collect(),
            fixed_effect_errors: errors,
            loglik,
            converged,
            scaled_gradient: projected / negative.abs().max(1.0),
            stop_code,
            stop_message,
            polished,
            estimator: if reml { "reml" } else { "ml" },
        })
    }
}

/// A profile-likelihood interval for one raw coefficient proportion.
#[derive(Clone, Copy, Debug)]
pub struct ComponentInterval {
    pub lower: f64,
    pub upper: f64,
    /// True where the endpoint is the edge of the parameter space rather than a
    /// point the data ruled out. Read it beside `profile_failures`: a bound
    /// reached because the likelihood never crossed and a bound reached because
    /// the profile could not be evaluated there are both reported here, and
    /// only a non-zero failure count separates them.
    pub lower_limited: bool,
    pub upper_limited: bool,
    pub level: f64,
    /// How many profile evaluations could not be made. A failure is unknown
    /// ground, not ground the data ruled out, so the interval is widened over
    /// it rather than narrowed; a non-zero count says the endpoints rest partly
    /// on evaluations that did not come back.
    pub profile_failures: usize,
}

/// A likelihood ratio test of one component against no variance at all.
#[derive(Clone, Debug)]
pub struct ComponentTest {
    pub statistic: f64,
    pub p_value: f64,
    pub rule: &'static str,
    pub null_loglik: f64,
}

const CHI2_ONE_DF_95: f64 = 3.841_458_820_694_124;

/// One class's difference from the shared baseline.
#[derive(Clone, Copy, Debug)]
pub struct Contrast {
    /// The class's variance minus the baseline. Signed: below nought means the
    /// class carries less than the classes it is being compared with.
    pub difference: f64,
    pub lower: f64,
    pub upper: f64,
    pub lower_limited: bool,
    pub upper_limited: bool,
    /// Against this class carrying the same variance as the baseline. An
    /// interior null, so an ordinary chi-square on one degree of freedom.
    pub p_value: f64,
    pub statistic: f64,
}

impl ComponentModel {
    /// Every class's deviation from the average class, with an interval each.
    ///
    /// **This is what the omnibus and the pooled contrasts cannot give.** The
    /// omnibus says the classes are not all alike and stops. A pooled contrast
    /// compares one class against the others *combined*, so raising one class
    /// raises the pool every other class is measured against -- which is why
    /// lifting a single class in calibration made a second class reject half
    /// the time. Two findings, one effect.
    ///
    /// # The constraint, and why there has to be one
    ///
    /// Writing every class as `baseline + difference` is the obvious move and
    /// it does not work: the baseline's matrix is the sum of the class
    /// matrices, so adding a constant to the baseline and taking it off every
    /// difference leaves the covariance identical. Five numbers for four
    /// quantities, and the differences are only defined up to a common shift.
    /// A first version of this shipped that way and produced forty estimates of
    /// which not one was negative, which is what a redundant direction looks
    /// like when the search drifts along it.
    ///
    /// The differences are therefore constrained to sum to nought, by fitting
    /// on the differenced matrices `K_c - K_last`. The baseline is then the
    /// *average* class variance and each difference is a deviation from it,
    /// which is both identified and the quantity anybody actually wants: not
    /// "class c against a floating reference" but "class c against the typical
    /// class".
    ///
    /// One fit gives every class but the omitted one, so the omitted one is
    /// rotated and each class is read from a fit where it is explicit.
    ///
    /// **With only two classes this says less than it appears to.** The two
    /// deviations must sum to nought, so they are mirror images and "the first
    /// is above average" is the same statement as "the second is below" -- one
    /// degree of freedom, however it is written. Splitting into four is what
    /// makes a single class stand out from the rest.
    ///
    /// `classes` must name only things that are classes of one split. Anything
    /// not named keeps its own variance and is left alone, which is what should
    /// happen to a remainder component: pooling "everything that is not a
    /// parent-child tie" into the baseline would compare siblings with parents.
    ///
    /// # Errors
    ///
    /// Returns a stable code where fewer than two classes are named, an index
    /// is out of range, or a fit fails.
    ///
    /// # Panics
    ///
    /// If a named class is missing from the list it was just taken from, which
    /// cannot happen: the indices are validated against that list first.
    pub fn contrasts(
        &self,
        y: &DVector<f64>,
        classes: &[usize],
        reml: bool,
    ) -> Result<Vec<Contrast>, &'static str> {
        let structured = self.matrices.len();
        if classes.iter().any(|k| *k >= structured) {
            return Err("COMPONENTS_INDEX_OUT_OF_RANGE");
        }
        let mut named: Vec<usize> = classes.to_vec();
        named.sort_unstable();
        named.dedup();
        if named.len() < 2 {
            return Err("COMPONENTS_CONTRASTS_NEED_TWO");
        }

        let mut baseline = self.matrices[named[0]].clone();
        for &k in &named[1..] {
            baseline += &self.matrices[k];
        }
        let others: Vec<DMatrix<f64>> = self
            .matrices
            .iter()
            .enumerate()
            .filter(|(k, _)| !named.contains(k))
            .map(|(_, m)| m.clone())
            .collect();

        let mut out: Vec<Option<Contrast>> = vec![None; named.len()];
        for (omitted, &last) in named.iter().enumerate() {
            // Everything but the omitted class, differenced against it.
            let free: Vec<usize> = named.iter().copied().filter(|k| *k != last).collect();
            let mut matrices = vec![baseline.clone()];
            for &k in &free {
                matrices.push(&self.matrices[k] - &self.matrices[last]);
            }
            matrices.extend(others.iter().cloned());
            let model = Self::build(&matrices, &self.design)?;
            let signed: Vec<usize> = (1..=free.len()).collect();
            let fit = model.fit_with_signs(y, reml, &signed)?;

            for (slot, &k) in free.iter().enumerate() {
                let position = named.iter().position(|n| *n == k).expect("named");
                if out[position].is_some() {
                    continue;
                }
                let coordinate = slot + 1;
                let difference = fit.variances[coordinate];
                let (lower, lower_limited, upper, upper_limited) =
                    model.signed_interval(y, reml, coordinate, &signed, difference)?;
                let held = model
                    .fit_general(y, reml, &signed, &[(coordinate, 0.0)])?
                    .loglik;
                let statistic = (2.0 * (fit.loglik - held)).max(0.0);
                out[position] = Some(Contrast {
                    difference,
                    lower,
                    upper,
                    lower_limited,
                    upper_limited,
                    // An interior null: the deviation is free to take either
                    // sign, so there is no boundary and no mixture.
                    p_value: chi2_one_df_upper_tail(statistic).clamp(0.0, 1.0),
                    statistic,
                });
            }
            let _ = omitted;
            if out.iter().all(Option::is_some) {
                break;
            }
        }
        out.into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or("COMPONENTS_CONTRASTS_INCOMPLETE")
    }

    /// A profile interval for one signed coordinate, walked outward and then
    /// bisected.
    fn signed_interval(
        &self,
        y: &DVector<f64>,
        reml: bool,
        coordinate: usize,
        signed: &[usize],
        estimate: f64,
    ) -> Result<(f64, bool, f64, bool), &'static str> {
        let free = self.fit_with_signs(y, reml, signed)?.loglik;
        let threshold = free - 0.5 * 3.841_458_820_694_124;
        let outside = |value: f64| {
            self.fit_general(y, reml, signed, &[(coordinate, value)])
                .map_or(true, |fit| fit.loglik < threshold)
        };
        let spread = estimate.abs().max(1e-3);
        let walk = |direction: f64| -> (f64, bool) {
            let mut step = 0.5 * spread;
            let mut far = estimate;
            let mut found = false;
            for _ in 0..30 {
                far = estimate + direction * step;
                if outside(far) {
                    found = true;
                    break;
                }
                step *= 2.0;
            }
            if !found {
                return (far, true);
            }
            let (mut inside, mut out) = (estimate, far);
            for _ in 0..50 {
                let middle = 0.5 * (inside + out);
                if outside(middle) {
                    out = middle;
                } else {
                    inside = middle;
                }
                if (out - inside).abs() < 1e-6 * spread {
                    break;
                }
            }
            (0.5 * (inside + out), false)
        };
        let (lower, lower_limited) = walk(-1.0);
        let (upper, upper_limited) = walk(1.0);
        Ok((lower, lower_limited, upper, upper_limited))
    }
}

impl ComponentModel {
    /// The best log-likelihood with one component holding a fixed proportion of
    /// the sum of raw covariance coefficients.
    ///
    /// A coefficient proportion is not a parameter, so it cannot be pinned by
    /// fixing one. But
    /// holding it fixed is a substitution, and an easier one than it looks. From
    ///
    /// ```text
    /// v = s_j / (s_j + S)      where S is the sum of the others
    /// ```
    ///
    /// comes `s_j = S * v / (1 - v)`. So the search runs over the other
    /// variances and this one follows, and because the relation is linear in
    /// them the chain rule is a constant rather than a derivative — every free
    /// variance moves the pinned one by the same `v / (1 - v)`.
    ///
    /// A proportion of exactly one would need every other coefficient at nought and the
    /// substitution divides by zero there, so it is refused and the endpoint
    /// reports itself as limited by the parameter space.
    fn profile_objective(
        &self,
        y: &DVector<f64>,
        reml: bool,
        component: usize,
        proportion: f64,
    ) -> Option<f64> {
        if !(0.0..=1.0).contains(&proportion) || proportion > 1.0 - 1e-9 {
            return None;
        }
        let count = self.parameters();
        let free: Vec<usize> = (0..count).filter(|k| *k != component).collect();
        let factor = proportion / (1.0 - proportion);

        let expand = |packed: &[f64]| -> Vec<f64> {
            let mut theta = vec![0.0; count];
            let mut others = 0.0;
            for (slot, &k) in free.iter().enumerate() {
                theta[k] = packed[slot];
                others += packed[slot];
            }
            theta[component] = factor * others;
            theta
        };

        let mut best: Option<f64> = None;
        for start in [vec![1.0 / count as f64; free.len()], {
            let mut s = vec![0.1; free.len()];
            if let Some(last) = s.last_mut() {
                *last = 0.9;
            }
            s
        }] {
            let value_of = |candidate: &[f64]| -> f64 {
                self.evaluate(&expand(candidate), y, reml, false)
                    .map_or(1e30, |e| e.negative_loglik)
            };
            let gradient_of = |candidate: &[f64]| -> Vec<f64> {
                self.evaluate(&expand(candidate), y, reml, true)
                    .map_or_else(
                        || vec![0.0; free.len()],
                        |e| {
                            // The pinned component is carried by all the others at
                            // once, so each of them picks up the same proportion of its
                            // slope.
                            let through = e.gradient[component] * factor;
                            free.iter().map(|&k| e.gradient[k] + through).collect()
                        },
                    )
            };
            let Ok(bounds) = Bounds::new(vec![0.0; free.len()], vec![f64::INFINITY; free.len()])
            else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(free.len());
            control.maxit = 400;
            control.fnscale = value_of(&start).abs().max(1.0);
            control.parscale = vec![1.0; free.len()];
            control.factr = 1.0e3;
            control.pgtol = 1e-9;
            control.lmm = free.len().min(10);
            if let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control)
                && let Some(at) = self.evaluate(&expand(&solution.par), y, reml, false)
                && at.negative_loglik.is_finite()
                && best.is_none_or(|b: f64| at.negative_loglik < b)
            {
                best = Some(at.negative_loglik);
            }
        }
        best.map(|negative| -negative)
    }

    /// A 95 per cent profile-likelihood interval for one raw coefficient proportion.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the fit or the profile at the estimate fails.
    pub fn profile_interval(
        &self,
        y: &DVector<f64>,
        reml: bool,
        component: usize,
    ) -> Result<ComponentInterval, &'static str> {
        if component >= self.parameters() {
            return Err("COMPONENTS_NO_SUCH_COMPONENT");
        }
        let fit = self.fit(y, reml)?;
        let fitted = fit.proportions[component];

        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        let scaled = y / variance.sqrt();

        // Both ends of the difference are measured the same way, by pinning and
        // re-optimising. Taking the maximum from the free fit instead leaves a
        // constant in the deviance, which is how the two-trait intervals once
        // came to have zero width.
        let maximum = self
            .profile_objective(&scaled, reml, component, fitted)
            .ok_or("COMPONENTS_PROFILE_MAXIMUM_FAILED")?;
        let deviance = |proportion: f64| -> Option<f64> {
            self.profile_objective(&scaled, reml, component, proportion)
                .map(|ll| 2.0 * (maximum - ll))
        };

        // **A profile that could not be evaluated is not a likelihood that fell
        // away.** Read as an infinite deviance it looked like ground the data
        // had ruled out, so the bisection stepped inward and the interval came
        // back narrower than the data support -- confidently, and with nothing
        // to show it had happened. A failure is now covered rather than cut
        // away, and counted so a reader can see it.
        let mut failures = 0usize;
        let endpoint = |bound: f64, failures: &mut usize| -> (f64, bool) {
            match deviance(bound) {
                None => {
                    *failures += 1;
                    return (bound, true);
                }
                Some(value) if value <= CHI2_ONE_DF_95 => return (bound, true),
                Some(_) => {}
            }
            let (mut inside, mut outside) = (fitted, bound);
            for _ in 0..80 {
                let middle = 0.5 * (inside + outside);
                if (outside - inside).abs() <= 1e-9 {
                    break;
                }
                match deviance(middle) {
                    Some(value) if value <= CHI2_ONE_DF_95 => inside = middle,
                    Some(_) => outside = middle,
                    None => {
                        *failures += 1;
                        inside = middle;
                    }
                }
            }
            (0.5 * (inside + outside), false)
        };

        let (lower, lower_limited) = endpoint(0.0, &mut failures);
        let (upper, upper_limited) = endpoint(1.0 - 1e-9, &mut failures);
        Ok(ComponentInterval {
            lower,
            upper,
            lower_limited,
            upper_limited,
            level: 0.95,
            profile_failures: failures,
        })
    }

    /// Test one component against having no variance at all.
    ///
    /// The null sits on the edge of the parameter space, because a variance
    /// cannot be negative, so the statistic is not chi-square on one degree of
    /// freedom. It is the Self–Liang half-and-half mixture of that and a point
    /// mass at nought, which is the same rule the one-component model uses for
    /// no additive variance and the rule the port lab's inference catalogue
    /// records for a household variance under
    /// `solar_successor.household_univariate.c2_zero_boundary_lrt`.
    ///
    /// **This is the single-boundary case and only that.** It holds because the
    /// other components are away from their bounds at the optimum. Testing one
    /// component while another also rests on nought is a different question with
    /// a different null, and this returns a refusal rather than a number there.
    ///
    /// # Errors
    ///
    /// Returns a stable code where a fit fails or another component is itself at
    /// the boundary.
    pub fn component_test(
        &self,
        y: &DVector<f64>,
        reml: bool,
        component: usize,
    ) -> Result<ComponentTest, &'static str> {
        if component >= self.parameters() {
            return Err("COMPONENTS_NO_SUCH_COMPONENT");
        }
        if component == self.parameters() - 1 {
            // A model with no residual variance is not a model anybody wants
            // tested, and the mixture argument does not apply to it.
            return Err("COMPONENTS_RESIDUAL_NOT_TESTABLE");
        }
        let fit = self.fit(y, reml)?;

        // The mixture assumes one parameter on the boundary and the rest inside
        // it. If another component has also gone to nought the null is a
        // different mixture, and returning this one would be a p-value for a
        // question nobody asked.
        for (other, proportion) in fit.proportions.iter().enumerate() {
            if other != component && *proportion <= 1e-9 {
                return Err("COMPONENTS_ANOTHER_COMPONENT_AT_ZERO");
            }
        }

        // The null drops the component rather than pinning it at nought, which
        // is the same model and a smaller one to fit.
        let kept: Vec<DMatrix<f64>> = self
            .matrices
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != component)
            .map(|(_, matrix)| matrix.clone())
            .collect();
        let null_loglik = if kept.is_empty() {
            // Nothing structured left: residual only, which is still a model.
            let residual =
                ComponentModel::build(&[DMatrix::identity(self.rows, self.rows)], &self.design)?;
            residual.fit(y, reml)?.loglik
        } else {
            ComponentModel::build(&kept, &self.design)?
                .fit(y, reml)?
                .loglik
        };

        let statistic = crate::deviance::deviance(fit.loglik, null_loglik);
        // The settling tolerance matters here and an exact test for nought does
        // not: two searches never land on identically the same number, so a fit
        // resting on the boundary arrives as a statistic of about 1e-10 and was
        // being reported at p = 0.5 rather than p = 1. Under these nulls that is
        // often half the fits, and in a scan it puts the atom in the wrong place
        // for a quantile plot or an inflation factor.
        let p_value = crate::deviance::p_value(statistic, |t| 0.5 * chi2_one_df_upper_tail(t));
        Ok(ComponentTest {
            statistic,
            p_value,
            rule: "mixture_50_50",
            null_loglik,
        })
    }
}

/// Predicted random effects for one component, with how uncertain each is.
#[derive(Clone, Debug)]
pub struct Prediction {
    /// Which component was predicted.
    pub component: usize,
    /// One predicted effect per person, in the order the matrices are indexed.
    pub values: Vec<f64>,
    /// The standard error of prediction for each, the square root of the
    /// diagonal of the prediction error variance.
    ///
    /// **Not the standard deviation of the prediction.** A prediction is shrunk
    /// toward nought, so its own spread is smaller than the effect's; what is
    /// wanted is how far the prediction may be from the effect it predicts, and
    /// that is what this is.
    pub errors: Vec<f64>,
}

impl ComponentModel {
    /// Predict the random effects of one component.
    ///
    /// This is the best linear unbiased prediction: for component `k` with
    /// covariance `G = s_k K_k`,
    ///
    /// ```text
    /// u_hat = G V^-1 (y - X beta_hat)
    /// ```
    ///
    /// with the prediction error variance
    ///
    /// ```text
    /// G - G V^-1 G + G V^-1 X (X' V^-1 X)^-1 X' V^-1 G
    /// ```
    ///
    /// whose diagonal gives the standard errors. The third term is the price of
    /// having estimated the fixed effects rather than known them, and dropping
    /// it -- which is easy to do, since the first two terms look like a complete
    /// formula -- makes every prediction look more certain than it is.
    ///
    /// **The variance components are treated as known.** They were estimated
    /// from the same data, so these errors are a little optimistic. That is the
    /// usual approximation and the same one the fixed effects' standard errors
    /// make.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the fit fails or the component does not
    /// exist. The residual cannot be predicted this way and is refused: its
    /// "prediction" is just the residual itself.
    pub fn blup(
        &self,
        y: &DVector<f64>,
        reml: bool,
        component: usize,
    ) -> Result<Prediction, &'static str> {
        if component >= self.matrices.len() {
            return Err("COMPONENTS_NO_SUCH_COMPONENT_TO_PREDICT");
        }
        let fit = self.fit(y, reml)?;
        let theta: Vec<f64> = fit.variances.clone();
        let p = self.design.ncols();
        let n = self.rows;

        // The residual after the fixed effects, and the pieces every block
        // needs. `xvx` is accumulated across blocks because the fixed effects
        // are global even though the covariance is not.
        let beta = DVector::from_iterator(p, fit.fixed_effects.iter().copied());
        let residual = y - &self.design * &beta;

        let mut xvx = DMatrix::<f64>::zeros(p, p);
        let mut solves: Vec<BlockSolve> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let size = block.len();
            let v = self.assemble(block, &theta);
            let chol =
                crate::dense::DenseFactor::new(&v).ok_or("COMPONENTS_NOT_POSITIVE_DEFINITE")?;
            let rb = DVector::from_iterator(size, block.iter().map(|&i| residual[i]));
            let xb = DMatrix::from_fn(size, p, |r, c| self.design[(block[r], c)]);
            let vr = chol.solve_vector(&rb);
            let vx = chol.solve_matrix(&xb);
            xvx += xb.transpose() * &vx;
            solves.push(BlockSolve {
                rows: block.clone(),
                inverse: chol.inverse(),
                vr,
                vx,
            });
        }
        let xvx_inverse = xvx
            .cholesky()
            .ok_or("COMPONENTS_DESIGN_RANK_DEFICIENT")?
            .inverse();

        let mut values = vec![0.0; n];
        let mut errors = vec![0.0; n];
        let matrix = &self.matrices[component];
        let scale = theta[component];
        for solve in &solves {
            let (block, inverse, vr, vx) = (&solve.rows, &solve.inverse, &solve.vr, &solve.vx);
            let size = block.len();
            // G over this block, which is the component's own matrix scaled.
            let g = DMatrix::from_fn(size, size, |i, j| scale * matrix[(block[i], block[j])]);
            let predicted = &g * vr;
            // G V^-1 G, and the correction for having estimated beta.
            let gvi = &g * inverse;
            let gvig = &gvi * &g;
            // G V^-1 X, and `vx` already holds V^-1 X for this block.
            let gvx = &g * vx;
            let correction = &gvx * &xvx_inverse * gvx.transpose();
            for i in 0..size {
                values[block[i]] = predicted[i];
                let variance = g[(i, i)] - gvig[(i, i)] + correction[(i, i)];
                errors[block[i]] = variance.max(0.0).sqrt();
            }
        }
        Ok(Prediction {
            component,
            values,
            errors,
        })
    }
}

// PyO3 extracts each argument from a Python object, so a `#[pyfunction]` takes
// them by value whether or not the body consumes them. The lint cannot be
// satisfied here without breaking the macro.
#[allow(clippy::needless_pass_by_value)]
#[cfg(feature = "python")]
mod python {
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::ComponentModel;
    use nalgebra::{DMatrix, DVector};

    fn build(
        matrices: &[PyReadonlyArray2<'_, f64>],
        design: &PyReadonlyArray2<'_, f64>,
    ) -> PyResult<ComponentModel> {
        let converted: Vec<DMatrix<f64>> = matrices
            .iter()
            .map(|m| {
                let a = m.as_array();
                DMatrix::from_fn(a.shape()[0], a.shape()[1], |i, j| a[(i, j)])
            })
            .collect();
        let x = design.as_array();
        let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
        ComponentModel::build(&converted, &x).map_err(PyValueError::new_err)
    }

    fn response(y: &PyReadonlyArray1<'_, f64>) -> DVector<f64> {
        let y = y.as_array();
        DVector::from_iterator(y.len(), y.iter().copied())
    }

    /// Fit one trait with any number of variance components.
    ///
    /// `matrices` are the structured components in order; the residual is added
    /// for you and is always last in what comes back.
    #[pyfunction]
    #[pyo3(signature = (matrices, design, y, reml=true))]
    #[allow(clippy::type_complexity)]
    pub fn component_fit(
        matrices: Vec<PyReadonlyArray2<'_, f64>>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        reml: bool,
    ) -> PyResult<(
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        bool,
        Vec<f64>,
        Vec<f64>,
        i32,
        String,
        bool,
    )> {
        let model = build(&matrices, &design)?;
        let fit = model
            .fit(&response(&y), reml)
            .map_err(PyValueError::new_err)?;
        Ok((
            fit.variances,
            fit.proportions,
            fit.coefficient_total,
            fit.loglik,
            fit.scaled_gradient,
            fit.converged,
            fit.fixed_effects,
            fit.fixed_effect_errors,
            fit.stop_code,
            fit.stop_message,
            fit.polished,
        ))
    }

    /// Every class's difference from a shared baseline, with an interval each.
    ///
    /// `classes` indexes the structured matrices that are classes of one split.
    /// Returns one row per class: the difference, its interval, whether either
    /// end ran to a limit, and a test against the class matching the baseline.
    #[pyfunction]
    #[pyo3(signature = (matrices, design, y, classes, reml=true))]
    #[allow(clippy::type_complexity)]
    pub fn component_contrasts(
        matrices: Vec<PyReadonlyArray2<'_, f64>>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        classes: Vec<usize>,
        reml: bool,
    ) -> PyResult<Vec<(f64, f64, f64, bool, bool, f64, f64)>> {
        let model = build(&matrices, &design)?;
        let got = model
            .contrasts(&response(&y), &classes, reml)
            .map_err(PyValueError::new_err)?;
        Ok(got
            .into_iter()
            .map(|c| {
                (
                    c.difference,
                    c.lower,
                    c.upper,
                    c.lower_limited,
                    c.upper_limited,
                    c.p_value,
                    c.statistic,
                )
            })
            .collect())
    }

    /// Test whether several components share one variance.
    ///
    /// `components` indexes the structured matrices. Pool every one of them and
    /// the null is the ordinary single-component model.
    #[pyfunction]
    #[pyo3(signature = (matrices, design, y, components, reml=true))]
    pub fn component_equality_test(
        matrices: Vec<PyReadonlyArray2<'_, f64>>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        components: Vec<usize>,
        reml: bool,
    ) -> PyResult<(f64, f64, String, f64)> {
        let model = build(&matrices, &design)?;
        let test = model
            .equality_test(&response(&y), &components, reml)
            .map_err(PyValueError::new_err)?;
        Ok((
            test.statistic,
            test.p_value,
            test.rule.to_owned(),
            test.null_loglik,
        ))
    }

    /// Predict the random effects of one component.
    ///
    /// Returns one predicted effect per person and the standard error of each,
    /// in the order the matrices are indexed.
    #[pyfunction]
    #[pyo3(signature = (matrices, design, y, component, reml=true))]
    pub fn component_blup(
        matrices: Vec<PyReadonlyArray2<'_, f64>>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        component: usize,
        reml: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let model = build(&matrices, &design)?;
        let prediction = model
            .blup(&response(&y), reml, component)
            .map_err(PyValueError::new_err)?;
        Ok((prediction.values, prediction.errors))
    }

    /// A 95 per cent profile interval for one raw coefficient proportion.
    #[pyfunction]
    #[pyo3(signature = (matrices, design, y, component, reml=true))]
    pub fn component_interval(
        matrices: Vec<PyReadonlyArray2<'_, f64>>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        component: usize,
        reml: bool,
    ) -> PyResult<(f64, f64, bool, bool, f64, usize)> {
        let model = build(&matrices, &design)?;
        let interval = model
            .profile_interval(&response(&y), reml, component)
            .map_err(PyValueError::new_err)?;
        Ok((
            interval.lower,
            interval.upper,
            interval.lower_limited,
            interval.upper_limited,
            interval.level,
            interval.profile_failures,
        ))
    }

    /// Test one component against having no variance at all.
    #[pyfunction]
    #[pyo3(signature = (matrices, design, y, component, reml=true))]
    pub fn component_test(
        matrices: Vec<PyReadonlyArray2<'_, f64>>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        component: usize,
        reml: bool,
    ) -> PyResult<(f64, f64, String, f64)> {
        let model = build(&matrices, &design)?;
        let test = model
            .component_test(&response(&y), reml, component)
            .map_err(PyValueError::new_err)?;
        Ok((
            test.statistic,
            test.p_value,
            test.rule.to_owned(),
            test.null_loglik,
        ))
    }
}

#[cfg(feature = "python")]
pub use python::{
    component_blup, component_contrasts, component_equality_test, component_fit,
    component_interval, component_test,
};

impl ComponentModel {
    /// Test whether several components share one variance.
    ///
    /// **This is the question a split matrix is built to ask, and it is not
    /// the same as asking whether each piece is nought.** Splitting the direct
    /// parent-offspring cells of a relationship matrix by the sex of parent and
    /// child gives four classes, and every one of them carries variance if the
    /// trait is heritable at all -- so testing each against nought answers
    /// nothing, and returns a p-value near nought for anything heritable. What
    /// is wanted is whether a mother resembles her son by as much as a father
    /// resembles his daughter, which is the classes being equal to one another.
    ///
    /// The null pools the named components by adding their matrices, which is
    /// exact rather than approximate: the pieces were made by splitting a
    /// matrix, so their sum is that matrix back. Pool every class together with
    /// the remainder and the null is the ordinary additive model, and the test
    /// says whether splitting it bought anything at all.
    ///
    /// # The reference distribution
    ///
    /// Pooling `k` components removes `k - 1` free variances, and under the
    /// null the shared variance is positive rather than nought, so nothing sits
    /// on a bound and the reference is an ordinary chi-square on `k - 1`
    /// degrees of freedom. The alternative's variances are still bounded below,
    /// and where such a bound binds the deviance is stochastically smaller,
    /// which makes this conservative rather than optimistic.
    ///
    /// # Errors
    ///
    /// Returns a stable code where fewer than two distinct components are
    /// named, an index is out of range, or either fit fails.
    pub fn equality_test(
        &self,
        y: &DVector<f64>,
        components: &[usize],
        reml: bool,
    ) -> Result<ComponentTest, &'static str> {
        let structured = self.matrices.len();
        if components.iter().any(|k| *k >= structured) {
            return Err("COMPONENTS_INDEX_OUT_OF_RANGE");
        }
        let mut named: Vec<usize> = components.to_vec();
        named.sort_unstable();
        named.dedup();
        if named.len() < 2 {
            return Err("COMPONENTS_EQUALITY_NEEDS_TWO");
        }

        let mut pooled = self.matrices[named[0]].clone();
        for &k in &named[1..] {
            pooled += &self.matrices[k];
        }
        let mut reduced: Vec<DMatrix<f64>> = self
            .matrices
            .iter()
            .enumerate()
            .filter(|(k, _)| !named.contains(k))
            .map(|(_, m)| m.clone())
            .collect();
        reduced.push(pooled);

        let free = self.fit(y, reml)?;
        let null = Self::build(&reduced, &self.design)?.fit(y, reml)?;
        let statistic = (2.0 * (free.loglik - null.loglik)).max(0.0);
        let df = (named.len() - 1) as f64;
        Ok(ComponentTest {
            statistic,
            p_value: chi2_upper_tail(statistic, df).clamp(0.0, 1.0),
            rule: "chi2_k_minus_one",
            null_loglik: null.loglik,
        })
    }
}

#[cfg(test)]
mod tests {

    /// **The reason this exists.** A pooled contrast compares one class against
    /// the others combined, so lifting one class raises the pool every other
    /// class is measured against, and a second class then looks low. In
    /// calibration that made a single lifted class produce a second rejection
    /// half the time. Writing every class as a baseline plus a signed
    /// difference removes the coupling: the lifted class should come back high,
    /// and the untouched ones should come back at nought.
    #[test]
    fn a_lifted_class_does_not_drag_the_others_with_it() {
        let (matrices, design, y) = split_by_class(1.2);
        let model = ComponentModel::build(&matrices, &design).expect("valid");
        // **Only the classes go in the baseline.** Matrix 0 here is the
        // remainder -- everything that is not a parent-child tie -- and it is
        // not a class of anything. Pooling it into the baseline would compare
        // siblings with parent-child ties and call the difference a finding.
        let got = model.contrasts(&y, &[1, 2], true).expect("contrasts");
        assert_eq!(got.len(), 2);
        // The first named class is the one carrying the extra variance.
        assert!(
            got[0].difference > 0.0 && got[0].p_value < 0.05,
            "the lifted class came back at {} with p {}",
            got[0].difference,
            got[0].p_value
        );
        // **With two classes the deviations are forced to be mirror images**,
        // because they must sum to nought. "The first is above average" and
        // "the second is below average" are then the same statement, and no
        // parameterisation can separate them: with two numbers and one
        // constraint there is one degree of freedom. That is arithmetic rather
        // than a fault, and it is why the classes worth splitting are four.
        assert!(
            (got[0].difference + got[1].difference).abs() < 1e-6,
            "the deviations are {} and {} and must sum to nought",
            got[0].difference,
            got[1].difference
        );
        assert!(
            got[1].difference < 0.0,
            "the untouched class should sit below the average once the other \
             is lifted, and it is at {}",
            got[1].difference
        );
    }

    /// **The parameterisation must be identified, and this is the test that
    /// was missing.** Writing every class as a free baseline plus a free
    /// difference is rank deficient -- the baseline's matrix is the sum of the
    /// class matrices, so a constant moved from the baseline into every
    /// difference changes nothing. The first version shipped that way and
    /// passed a recovery test, because a redundant model can still put extra
    /// variance in the right place; it simply cannot say where nought is. What
    /// it produced on real data was forty estimates without one negative among
    /// them.
    ///
    /// Under the sum-to-zero constraint the deviations must sum to nought, and
    /// with no class differing they must sit around nought rather than all on
    /// one side.
    #[test]
    fn the_deviations_sum_to_nought_and_are_not_all_one_sided() {
        let (matrices, design, y) = split_by_class(0.0);
        let model = ComponentModel::build(&matrices, &design).expect("valid");
        let got = model.contrasts(&y, &[1, 2], true).expect("contrasts");
        let total: f64 = got.iter().map(|c| c.difference).sum();
        let scale = got
            .iter()
            .map(|c| c.difference.abs())
            .fold(0.0f64, f64::max)
            .max(1e-6);
        assert!(
            total.abs() < 1e-3 * scale.max(1.0),
            "the deviations sum to {total}, which they cannot do if the \
             parameterisation is identified"
        );
        assert!(
            got.iter().any(|c| c.difference < 0.0),
            "every deviation is non-negative, which is what the rank-deficient \
             version did"
        );
    }

    /// A difference can be negative, and its interval must be able to say so.
    /// Bounding it at nought would make one direction unreachable and the other
    /// unfalsifiable.
    #[test]
    fn a_difference_and_its_interval_can_be_negative() {
        let (matrices, design, y) = split_by_class(1.2);
        let model = ComponentModel::build(&matrices, &design).expect("valid");
        let got = model.contrasts(&y, &[1, 2], true).expect("contrasts");
        assert!(
            got.iter().any(|c| c.lower < 0.0),
            "no interval reaches below nought, so a class carrying less than \
             the baseline could never be reported"
        );
        for contrast in &got {
            assert!(
                contrast.lower <= contrast.difference + 1e-9
                    && contrast.difference <= contrast.upper + 1e-9,
                "[{}, {}] does not contain {}",
                contrast.lower,
                contrast.upper,
                contrast.difference
            );
        }
    }

    /// Fewer than two classes is a comparison with nothing to compare.
    #[test]
    fn contrasts_need_two_classes() {
        let (matrices, design, y) = split_by_class(0.0);
        let model = ComponentModel::build(&matrices, &design).expect("valid");
        assert!(model.contrasts(&y, &[0], true).is_err());
        assert!(model.contrasts(&y, &[1, 1], true).is_err());
        assert!(model.contrasts(&y, &[0, 99], true).is_err());
    }

    /// **Pooling every piece back together must reproduce the whole**, or the
    /// null this test uses is not the model it claims to be. A relationship
    /// matrix split by class and summed again is that matrix, so the pooled fit
    /// has to match an ordinary one-component fit.
    #[test]
    fn pooling_every_split_piece_gives_the_unsplit_model_back() {
        let (matrices, design, y) = split_by_class(0.0);
        let whole = matrices
            .iter()
            .skip(1)
            .fold(matrices[0].clone(), |sum, m| sum + m);
        let split = ComponentModel::build(&matrices, &design).expect("valid");
        let together = ComponentModel::build(&[whole], &design).expect("valid");
        let all: Vec<usize> = (0..matrices.len()).collect();
        let test = split.equality_test(&y, &all, true).expect("tests");
        let plain = together.fit(&y, true).expect("fits");
        assert!(
            (test.null_loglik - plain.loglik).abs() < 1e-8,
            "the pooled null gives {} where the unsplit model gives {}",
            test.null_loglik,
            plain.loglik
        );
        // Only that it does not reject. A single draw from a true null is
        // uniform, so p = 0.18 is an ordinary one and thresholding it any
        // harder would be testing the seed rather than the model. Whether the
        // level is held is a calibration question, not a unit test.
        assert!(
            test.p_value > 0.05,
            "no class difference was simulated but p is {}",
            test.p_value
        );
    }

    /// A class that really does differ is found.
    #[test]
    fn a_class_that_differs_is_detected() {
        let (matrices, design, y) = split_by_class(1.2);
        let split = ComponentModel::build(&matrices, &design).expect("valid");
        let all: Vec<usize> = (0..matrices.len()).collect();
        let test = split.equality_test(&y, &all, true).expect("tests");
        assert!(
            test.p_value < 0.05,
            "one class was simulated with far more variance but p is {}",
            test.p_value
        );
    }

    /// Fewer than two distinct components is a question with no content.
    #[test]
    fn equality_needs_two_distinct_components() {
        let (matrices, design, y) = split_by_class(0.0);
        let model = ComponentModel::build(&matrices, &design).expect("valid");
        assert!(model.equality_test(&y, &[0], true).is_err());
        assert!(model.equality_test(&y, &[1, 1], true).is_err());
        assert!(model.equality_test(&y, &[0, 99], true).is_err());
    }

    /// A matrix split into a remainder and two classes of parent-child tie.
    ///
    /// `extra` is added to the first class's variance, so nought simulates no
    /// class difference at all.
    fn split_by_class(extra: f64) -> (Vec<DMatrix<f64>>, DMatrix<f64>, DVector<f64>) {
        let families = 150;
        let n = 3 * families;
        let mut rest = DMatrix::<f64>::identity(n, n);
        let mut first = DMatrix::<f64>::zeros(n, n);
        let mut second = DMatrix::<f64>::zeros(n, n);
        for family in 0..families {
            let (parent, a, b) = (3 * family, 3 * family + 1, 3 * family + 2);
            rest[(a, b)] = 0.5;
            rest[(b, a)] = 0.5;
            first[(parent, a)] = 0.5;
            first[(a, parent)] = 0.5;
            second[(parent, b)] = 0.5;
            second[(b, parent)] = 0.5;
        }
        let covariance = (&rest + &first + &second) * 0.5
            + &first * extra
            + DMatrix::<f64>::identity(n, n) * 0.5;
        let factor = covariance.cholesky().expect("positive definite").l();
        let mut state = 20_260_814u64;
        let draw = DVector::from_iterator(
            n,
            (0..n).map(|_| {
                let mut total = 0.0;
                for _ in 0..12 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    total += (state >> 11) as f64 / (1u64 << 53) as f64;
                }
                total - 6.0
            }),
        );
        (
            vec![rest, first, second],
            DMatrix::from_element(n, 1, 1.0),
            factor * draw,
        )
    }
    use super::ComponentModel;
    use nalgebra::{DMatrix, DVector};

    /// Sibling pairs, in households of four so that each household holds two
    /// pairs from different families.
    ///
    /// **Drawn from the model rather than from hand-mixed coefficients.** An
    /// earlier version of this built the response by adding a shared genetic
    /// term, a shared household term and noise with coefficients chosen by eye,
    /// and produced a within-pair correlation of 0.81 against 0.25 across the
    /// household. No additive-plus-household model can reach that: siblings
    /// share half their additive variance, so it implied a heritability of 1.12.
    /// The optimiser had nowhere to go but the boundary, and the tests then
    /// refused to give a p-value because a component was resting on nought. The
    /// refusal was right and the data were wrong.
    ///
    /// Here each sibling gets a value correlated one half with its sib, which is
    /// what the relationship matrix says, and the three variances are chosen to
    /// sum to one: two fifths additive, one fifth household, two fifths
    /// residual.
    pub(super) fn small() -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DVector<f64>) {
        // **Big enough that the sample looks like the model it came from.** At
        // forty pairs the empirical within-pair correlation came out at 0.57
        // against a true 0.40, which implies a heritability above one; the fit
        // then had nowhere to go but the boundary and the tests refused to give
        // a p-value. Nothing was wrong with the estimator. Two hundred pairs
        // keeps the sample correlations close enough to the truth that the model
        // can represent them.
        let pairs = 200;
        let n = 2 * pairs;
        let mut a = DMatrix::<f64>::identity(n, n);
        for pair in 0..pairs {
            a[(2 * pair, 2 * pair + 1)] = 0.5;
            a[(2 * pair + 1, 2 * pair)] = 0.5;
        }
        // Households of four, so each holds two sibling pairs from different
        // families -- co-residents who share no genes, which is what separates a
        // household component from an additive one.
        let mut h = DMatrix::<f64>::identity(n, n);
        for household in 0..n / 4 {
            for i in 0..4 {
                for j in 0..4 {
                    h[(household * 4 + i, household * 4 + j)] = 1.0;
                }
            }
        }
        let design = DMatrix::from_element(n, 1, 1.0);

        let mut seed = 20_260_812u64;
        // Three uniforms scaled to unit variance.
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

        let (additive, household_share, residual) = (0.4f64, 0.2f64, 0.4f64);
        let half = 0.5f64.sqrt();
        let mut y = DVector::<f64>::zeros(n);
        for household in 0..n / 4 {
            let shared = next();
            for i in 0..4 {
                y[household * 4 + i] += household_share.sqrt() * shared;
            }
        }
        for pair in 0..pairs {
            // Correlated one half within the pair, unit variance each, which is
            // exactly what the relationship matrix above describes.
            let common = next();
            for i in 0..2 {
                let genetic = half * common + half * next();
                y[2 * pair + i] += additive.sqrt() * genetic + residual.sqrt() * next();
            }
        }
        (a, h, design, y)
    }

    /// The gradient is what everything downstream rests on, and an analytic one
    /// for several components is easy to get wrong in the restricted term. It is
    /// checked against a central difference of the objective at points away from
    /// the bound and at points resting on it.
    #[test]
    fn the_gradient_matches_a_central_difference() {
        let (a, h, design, y) = small();
        let model = ComponentModel::build(&[a, h], &design).expect("valid");
        for reml in [false, true] {
            for point in [
                vec![1.0, 0.5, 1.0],
                vec![0.3, 0.2, 1.4],
                vec![2.0, 0.1, 0.4],
                vec![0.7, 0.7, 0.7],
            ] {
                let at = model.evaluate(&point, &y, reml, true).expect("evaluates");
                for k in 0..point.len() {
                    let step = 1e-6 * point[k].max(1.0);
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
                        (at.gradient[k] - numeric).abs() / scale < 1e-5,
                        "reml={reml} component {k} at {point:?}: analytic {} against numeric {}",
                        at.gradient[k],
                        numeric
                    );
                }
            }
        }
    }

    /// A household component that is really there is found, and the estimate
    /// sits where the simulation put it.
    #[test]
    fn a_household_component_is_recovered() {
        let (a, h, design, y) = small();
        let model = ComponentModel::build(&[a, h], &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");
        assert!(
            fit.converged,
            "did not converge, |g| = {}",
            fit.scaled_gradient
        );
        assert_eq!(fit.variances.len(), 3);
        assert!(
            fit.proportions[1] > 0.05,
            "the household share came out at {}, but the data were simulated with one",
            fit.proportions[1]
        );
        let total: f64 = fit.proportions.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "proportions sum to {total}");
    }

    /// The fit says why the search stopped, and whether it had to search twice.
    ///
    /// **These are three different questions and the package used to answer
    /// one.** `converged` is a projected-gradient test recomputed after the
    /// search; `stop_code` is the search's own reason for stopping, which is
    /// `factr` -- the objective settling -- on every red deer fit and never
    /// `pgtol`; and `polished` says whether the second search with `factr`
    /// switched off had to run. A well-conditioned problem like this one meets
    /// the gradient test first time, so the polish must not fire: if it ever
    /// does here, it is firing on fits it was built to leave alone.
    #[test]
    fn the_fit_says_why_the_search_stopped_and_whether_it_ran_twice() {
        let (a, h, design, y) = small();
        let model = ComponentModel::build(&[a, h], &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");
        assert!(fit.converged, "|g| = {}", fit.scaled_gradient);
        assert!(
            !fit.polished,
            "the polish fired on a fit that already met the gradient test"
        );
        // Nought is a tolerance firing, one is running out of iterations.
        assert!(
            fit.stop_code == 0 || fit.stop_code == 1,
            "unexpected stop code {}: {}",
            fit.stop_code,
            fit.stop_message
        );
        assert!(!fit.stop_message.is_empty());
    }

    /// With one component this is the model `prepared.rs` fits, by a slower
    /// route. They must agree, or one of them is wrong.
    #[test]
    fn one_component_matches_the_eigen_simplified_model() {
        let (a, _, design, y) = small();
        let model = ComponentModel::build(std::slice::from_ref(&a), &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");

        let prepared = crate::prepared::PreparedModel::build(&design, &a).expect("prepared builds");
        let theirs = prepared.fit_one_trait(&y, true);
        assert!(
            (fit.proportions[0] - theirs.h2).abs() < 1e-6,
            "general {} against eigen-simplified {}",
            fit.proportions[0],
            theirs.h2
        );
        assert!(
            (fit.loglik - theirs.loglik).abs() < 1e-6,
            "log-likelihoods {} against {}",
            fit.loglik,
            theirs.loglik
        );
    }

    /// **The prediction is computed block by block and must equal the textbook
    /// formula computed densely.** The blocks are an optimisation and nothing
    /// else, so a difference between the two is a bug in the bookkeeping rather
    /// than a modelling choice -- and it would show up as predictions that were
    /// subtly wrong for people in large families and right for everybody else.
    #[test]
    fn the_prediction_matches_the_dense_formula() {
        let (a, h, design, y) = small();
        let model = ComponentModel::build(&[a.clone(), h.clone()], &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");
        let n = y.len();
        let p = design.ncols();

        for component in 0..2 {
            let got = model.blup(&y, true, component).expect("predicts");

            // The same thing, written out with no blocks at all.
            let mut v = &a * fit.variances[0] + &h * fit.variances[1];
            for i in 0..n {
                v[(i, i)] += fit.variances[2];
            }
            let inverse = v.clone().cholesky().expect("positive definite").inverse();
            let beta = DVector::from_iterator(p, fit.fixed_effects.iter().copied());
            let residual = &y - &design * &beta;
            let g = if component == 0 {
                &a * fit.variances[0]
            } else {
                &h * fit.variances[1]
            };
            let expected = &g * &inverse * &residual;

            let xvx = design.transpose() * &inverse * &design;
            let xvx_inverse = xvx.cholesky().expect("full rank").inverse();
            let gvx = &g * &inverse * &design;
            let pev = &g - &g * &inverse * &g + &gvx * &xvx_inverse * gvx.transpose();

            for i in 0..n {
                assert!(
                    (got.values[i] - expected[i]).abs() < 1e-9,
                    "component {component}, person {i}: {} against {}",
                    got.values[i],
                    expected[i]
                );
                let error = pev[(i, i)].max(0.0).sqrt();
                assert!(
                    (got.errors[i] - error).abs() < 1e-9,
                    "component {component}, person {i}: error {} against {}",
                    got.errors[i],
                    error
                );
            }
        }
    }

    /// A prediction is shrunk toward nought, and the error of prediction is
    /// smaller than the effect's own spread. Both are properties of what a BLUP
    /// is, and a formula missing its shrinkage would still look plausible.
    #[test]
    fn predictions_are_shrunk_and_carry_less_error_than_the_effect() {
        let (a, h, design, y) = small();
        let model = ComponentModel::build(&[a, h], &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");
        let predicted = model.blup(&y, true, 0).expect("predicts");

        let spread = {
            let mean = predicted.values.iter().sum::<f64>() / predicted.values.len() as f64;
            (predicted
                .values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / predicted.values.len() as f64)
                .sqrt()
        };
        let effect = fit.variances[0].sqrt();
        assert!(
            spread < effect,
            "predictions spread {spread} against an effect of {effect}: not shrunk"
        );
        for (index, error) in predicted.errors.iter().enumerate() {
            assert!(
                *error <= effect + 1e-9,
                "person {index} has a prediction error {error} above the effect's own \
                 spread {effect}"
            );
            assert!(
                *error > 0.0,
                "person {index} has no prediction error at all"
            );
        }
    }

    /// The residual cannot be predicted this way and asking is refused.
    #[test]
    fn the_residual_is_not_a_component_that_can_be_predicted() {
        let (a, h, design, y) = small();
        let model = ComponentModel::build(&[a, h], &design).expect("valid");
        assert_eq!(
            model.blup(&y, true, 2).err(),
            Some("COMPONENTS_NO_SUCH_COMPONENT_TO_PREDICT")
        );
    }

    /// The interval must contain the estimate and must have width. A
    /// degenerate interval collapsed onto its own point estimate does contain
    /// it, so containment alone is not the test -- that exact fault appeared in
    /// the two-trait intervals and was invisible to a containment check.
    #[test]
    fn the_interval_contains_the_estimate_and_has_width() {
        let (a, h, design, y) = small();
        let model = ComponentModel::build(&[a, h], &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");
        for component in 0..2 {
            let interval = model
                .profile_interval(&y, true, component)
                .expect("interval");
            let point = fit.proportions[component];
            assert!(
                interval.lower <= point + 1e-9 && point <= interval.upper + 1e-9,
                "component {component}: [{}, {}] does not contain {point}",
                interval.lower,
                interval.upper
            );
            assert!(
                interval.upper - interval.lower > 1e-3,
                "component {component}: [{}, {}] has no width",
                interval.lower,
                interval.upper
            );
        }
    }

    /// A household effect that is really there is detected; one that is not
    /// there is not invented. Both directions matter, and a test that only ever
    /// sees data with the effect present cannot tell a working test from one
    /// that always rejects.
    #[test]
    fn a_household_effect_is_detected_when_present_and_not_when_absent() {
        let (a, h, design, y) = small();
        let model = ComponentModel::build(&[a, h.clone()], &design).expect("valid");
        let present = model.component_test(&y, true, 1).expect("tests");
        assert!(
            present.p_value < 0.05,
            "a household effect was simulated but the test gave p = {}",
            present.p_value
        );

        // The same pedigree and households, but nothing shared beyond the genes.
        let n = design.nrows();
        let mut seed = 99u64;
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
        // Drawn the same way as `small`, with the household share set to
        // nought and the rest of the variance moved to the residual.
        let half = 0.5f64.sqrt();
        let (additive, residual) = (0.4f64, 0.6f64);
        let mut plain = DVector::<f64>::zeros(n);
        for pair in 0..n / 2 {
            let common = next();
            for i in 0..2 {
                let genetic = half * common + half * next();
                plain[2 * pair + i] = additive.sqrt() * genetic + residual.sqrt() * next();
            }
        }
        let absent = model.component_test(&plain, true, 1).expect("tests");
        assert!(
            absent.p_value > 0.05,
            "no household effect was simulated but the test gave p = {}",
            absent.p_value
        );
        assert_eq!(absent.rule, "mixture_50_50");
    }

    /// A household crossing two pedigree families must not be split apart. If
    /// blocks came from the relationship matrix alone these people would land in
    /// different blocks and the covariance between them would vanish, which is a
    /// different model fitted without complaint.
    #[test]
    fn a_household_spanning_two_families_stays_in_one_block() {
        let n = 4;
        let mut a = DMatrix::<f64>::identity(n, n);
        a[(0, 1)] = 0.5;
        a[(1, 0)] = 0.5;
        a[(2, 3)] = 0.5;
        a[(3, 2)] = 0.5;
        let mut h = DMatrix::<f64>::identity(n, n);
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] = 1.0;
            }
        }
        let design = DMatrix::from_element(n, 1, 1.0);
        let relationship_only = crate::blocks::family_blocks(&a);
        assert_eq!(
            relationship_only.len(),
            2,
            "the pedigree alone gives two families"
        );
        let model = ComponentModel::build(&[a, h], &design).expect("valid");
        assert_eq!(
            model.blocks.len(),
            1,
            "the household joins them, so there is one block and not two"
        );
    }
}
