//! One trait, any number of variance components.
//!
//! `prepared.rs` fits a single structured component plus residual, and owes its
//! speed to diagonalising the relationship matrix once per family block so that
//! every likelihood evaluation is a pass over a diagonal. **That saving does not
//! survive a second component.** Two structured matrices share no eigenbasis and
//! cannot be diagonalised together, so each evaluation goes back to factorising
//! a dense covariance (`docs/adr/0001`).
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

/// A fitted multi-component model.
#[derive(Clone, Debug)]
pub struct ComponentFit {
    /// One variance per structured component, in the order they were given,
    /// with the residual last.
    pub variances: Vec<f64>,
    /// Each variance as a share of their total. This is what gets reported: the
    /// first entry of a relationship-plus-residual model is the heritability.
    pub proportions: Vec<f64>,
    pub total_variance: f64,
    pub fixed_effects: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    pub scaled_gradient: f64,
    pub estimator: &'static str,
}

struct Evaluation {
    negative_loglik: f64,
    gradient: Vec<f64>,
    fixed_effects: Vec<f64>,
}

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
    pub fn build(
        matrices: &[DMatrix<f64>],
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
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
        if theta.iter().any(|v| !v.is_finite() || *v < 0.0) {
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
            let chol = v.cholesky()?;
            logdet += 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();

            let yb = DVector::from_iterator(size, block.iter().map(|&i| y[i]));
            let xb = DMatrix::from_fn(size, p, |r, c| self.design[(block[r], c)]);
            let vy = chol.solve(&yb);
            let vx = chol.solve(&xb);
            xvx += xb.transpose() * &vx;
            xvy += xb.transpose() * &vy;
            yvy += yb.dot(&vy);

            if want_gradient {
                kept.push(BlockSolve {
                    rows: block.clone(),
                    inverse: chol.inverse(),
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
        let lower = vec![0.0; count];
        let upper = vec![f64::INFINITY; count];

        // Starts: everything equal, residual-heavy, and each component in turn
        // carrying most of the variance. A component that is genuinely zero is
        // found from any of them; one that is large is not always found from a
        // start that puts it near nought.
        let mut starts: Vec<Vec<f64>> = vec![
            vec![1.0 / count as f64; count],
            {
                let mut s = vec![0.1 / count as f64; count];
                s[count - 1] = 0.9;
                s
            },
        ];
        for component in 0..count - 1 {
            let mut s = vec![0.1 / count as f64; count];
            s[component] = 0.6;
            s[count - 1] = 0.3;
            starts.push(s);
        }

        let mut best: Option<(f64, Vec<f64>, Vec<f64>, bool)> = None;
        for start in starts {
            let value_of = |candidate: &[f64]| -> f64 {
                self.evaluate(candidate, &scaled, reml, false)
                    .map_or(1e30, |e| e.negative_loglik)
            };
            let gradient_of = |candidate: &[f64]| -> Vec<f64> {
                self.evaluate(candidate, &scaled, reml, true)
                    .map_or_else(|| vec![0.0; count], |e| e.gradient)
            };
            let Ok(bounds) = Bounds::new(lower.clone(), upper.clone()) else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(count);
            control.maxit = 500;
            control.fnscale = value_of(&start).abs().max(1.0);
            control.parscale = vec![1.0; count];
            control.factr = 0.0;
            control.pgtol = 1e-9;
            control.lmm = count.min(10);
            let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control)
            else {
                continue;
            };
            let Some(at) = self.evaluate(&solution.par, &scaled, reml, true) else {
                continue;
            };
            if !at.negative_loglik.is_finite() {
                continue;
            }
            // The convergence test is the projected gradient recomputed here,
            // not the optimiser's own word for it: a component resting on nought
            // has a one-sided derivative and is converged when that derivative
            // pushes outward, which an unprojected norm calls a failure.
            let projected = at
                .gradient
                .iter()
                .enumerate()
                .map(|(k, g)| if solution.par[k] <= 0.0 { g.min(0.0) } else { *g })
                .fold(0.0f64, |worst, g| worst.max(g.abs()));
            let scaled_gradient = projected / at.negative_loglik.abs().max(1.0);
            if best
                .as_ref()
                .is_none_or(|(value, _, _, _)| at.negative_loglik < *value)
            {
                best = Some((
                    at.negative_loglik,
                    solution.par.clone(),
                    at.fixed_effects.clone(),
                    scaled_gradient < 1e-7,
                ));
            }
        }

        let (negative, par, beta, converged) = best.ok_or("COMPONENTS_NO_START_CONVERGED")?;
        let at = self
            .evaluate(&par, &scaled, reml, true)
            .ok_or("COMPONENTS_OPTIMUM_NOT_EVALUABLE")?;
        let projected = at
            .gradient
            .iter()
            .enumerate()
            .map(|(k, g)| if par[k] <= 0.0 { g.min(0.0) } else { *g })
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

        Ok(ComponentFit {
            variances,
            proportions,
            total_variance: total,
            fixed_effects: beta.iter().map(|b| b * scale).collect(),
            loglik,
            converged,
            scaled_gradient: projected / negative.abs().max(1.0),
            estimator: if reml { "reml" } else { "ml" },
        })
    }
}

/// A profile-likelihood interval for one component's share of the variance.
#[derive(Clone, Copy, Debug)]
pub struct ComponentInterval {
    pub lower: f64,
    pub upper: f64,
    /// True where the endpoint is the edge of the parameter space rather than a
    /// point the data ruled out.
    pub lower_limited: bool,
    pub upper_limited: bool,
    pub level: f64,
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

/// The upper tail of a chi-square on one degree of freedom.
fn chi2_one_df_upper_tail(statistic: f64) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    statrs::function::erf::erfc((statistic / 2.0).sqrt())
}

impl ComponentModel {
    /// The best log-likelihood with one component holding a fixed share of the
    /// total variance.
    ///
    /// A share is not a parameter, so it cannot be pinned by fixing one. But
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
    /// A share of exactly one would need every other variance at nought and the
    /// substitution divides by zero there, so it is refused and the endpoint
    /// reports itself as limited by the parameter space.
    fn profile_objective(
        &self,
        y: &DVector<f64>,
        reml: bool,
        component: usize,
        share: f64,
    ) -> Option<f64> {
        if !(0.0..=1.0).contains(&share) || share > 1.0 - 1e-9 {
            return None;
        }
        let count = self.parameters();
        let free: Vec<usize> = (0..count).filter(|k| *k != component).collect();
        let factor = share / (1.0 - share);

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
        for start in [
            vec![1.0 / count as f64; free.len()],
            {
                let mut s = vec![0.1; free.len()];
                if let Some(last) = s.last_mut() {
                    *last = 0.9;
                }
                s
            },
        ] {
            let value_of = |candidate: &[f64]| -> f64 {
                self.evaluate(&expand(candidate), y, reml, false)
                    .map_or(1e30, |e| e.negative_loglik)
            };
            let gradient_of = |candidate: &[f64]| -> Vec<f64> {
                self.evaluate(&expand(candidate), y, reml, true).map_or_else(
                    || vec![0.0; free.len()],
                    |e| {
                        // The pinned component is carried by all the others at
                        // once, so each of them picks up the same share of its
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
            control.factr = 0.0;
            control.pgtol = 1e-9;
            control.lmm = free.len().min(10);
            if let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control)
            {
                if let Some(at) = self.evaluate(&expand(&solution.par), y, reml, false) {
                    if at.negative_loglik.is_finite()
                        && best.is_none_or(|b: f64| at.negative_loglik < b)
                    {
                        best = Some(at.negative_loglik);
                    }
                }
            }
        }
        best.map(|negative| -negative)
    }

    /// A 95 per cent profile-likelihood interval for one component's share.
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
        let deviance = |share: f64| -> f64 {
            self.profile_objective(&scaled, reml, component, share)
                .map_or(f64::INFINITY, |ll| 2.0 * (maximum - ll))
        };

        let endpoint = |bound: f64| -> (f64, bool) {
            if deviance(bound) <= CHI2_ONE_DF_95 {
                return (bound, true);
            }
            let (mut inside, mut outside) = (fitted, bound);
            for _ in 0..80 {
                let middle = 0.5 * (inside + outside);
                if (outside - inside).abs() <= 1e-9 {
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

        let (lower, lower_limited) = endpoint(0.0);
        let (upper, upper_limited) = endpoint(1.0 - 1e-9);
        Ok(ComponentInterval {
            lower,
            upper,
            lower_limited,
            upper_limited,
            level: 0.95,
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
        for (other, share) in fit.proportions.iter().enumerate() {
            if other != component && *share <= 1e-9 {
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
            let residual = ComponentModel::build(&[DMatrix::identity(self.rows, self.rows)], &self.design)?;
            residual.fit(y, reml)?.loglik
        } else {
            ComponentModel::build(&kept, &self.design)?.fit(y, reml)?.loglik
        };

        let statistic = (2.0 * (fit.loglik - null_loglik)).max(0.0);
        let p_value = if statistic <= 0.0 {
            1.0
        } else {
            0.5 * chi2_one_df_upper_tail(statistic)
        };
        Ok(ComponentTest {
            statistic,
            p_value,
            rule: "mixture_50_50",
            null_loglik,
        })
    }
}

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
    pub fn component_fit(
        matrices: Vec<PyReadonlyArray2<'_, f64>>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        reml: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64, f64, f64, bool)> {
        let model = build(&matrices, &design)?;
        let fit = model
            .fit(&response(&y), reml)
            .map_err(PyValueError::new_err)?;
        Ok((
            fit.variances,
            fit.proportions,
            fit.total_variance,
            fit.loglik,
            fit.scaled_gradient,
            fit.converged,
        ))
    }

    /// A 95 per cent profile interval for one component's share of the variance.
    #[pyfunction]
    #[pyo3(signature = (matrices, design, y, component, reml=true))]
    pub fn component_interval(
        matrices: Vec<PyReadonlyArray2<'_, f64>>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        component: usize,
        reml: bool,
    ) -> PyResult<(f64, f64, bool, bool, f64)> {
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
pub use python::{component_fit, component_interval, component_test};

#[cfg(test)]
mod tests {
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

        let mut seed = 20260812u64;
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
                    let numeric = (model.evaluate(&up, &y, reml, false).unwrap().negative_loglik
                        - model.evaluate(&down, &y, reml, false).unwrap().negative_loglik)
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
        assert!(fit.converged, "did not converge, |g| = {}", fit.scaled_gradient);
        assert_eq!(fit.variances.len(), 3);
        assert!(
            fit.proportions[1] > 0.05,
            "the household share came out at {}, but the data were simulated with one",
            fit.proportions[1]
        );
        let total: f64 = fit.proportions.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "shares sum to {total}");
    }

    /// With one component this is the model `prepared.rs` fits, by a slower
    /// route. They must agree, or one of them is wrong.
    #[test]
    fn one_component_matches_the_eigen_simplified_model() {
        let (a, _, design, y) = small();
        let model = ComponentModel::build(&[a.clone()], &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");

        let prepared =
            crate::prepared::PreparedModel::build(&design, &a, None).expect("prepared builds");
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
                plain[2 * pair + i] =
                    additive.sqrt() * genetic + residual.sqrt() * next();
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
        assert_eq!(relationship_only.len(), 2, "the pedigree alone gives two families");
        let model = ComponentModel::build(&[a, h], &design).expect("valid");
        assert_eq!(
            model.blocks.len(),
            1,
            "the household joins them, so there is one block and not two"
        );
    }
}


