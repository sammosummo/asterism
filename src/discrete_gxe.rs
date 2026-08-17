//! One trait whose genes may act differently in two environments.
//!
//! This is the discrete case of gene-by-environment: the environment is a
//! binary label — sex is the canonical example — rather than a measured
//! range, so nothing is smoothed and no surface has to be chosen. The model
//! has one genetic standard deviation per environment, one residual standard
//! deviation per environment, and one genetic correlation between them.
//!
//! ```text
//! V_ij = A_ij s_i s_j c_ij + d_ij e_i^2
//! ```
//!
//! where `s_i` and `e_i` are the standard deviations for that person's
//! environment, and `c_ij` is one when two people share an environment and
//! `rho_g` when they do not.
//!
//! # What it can find, and they are different findings
//!
//! **The heritability differs between the environments.** The genetic
//! variance is larger in one than the other. That is a change of scale, and it
//! can follow from a change of scale in the measurement — men are larger, so
//! a volume measured in millimetres varies more in men whether or not the
//! genetics differ.
//!
//! **The genes differ between the environments.** The genetic correlation
//! across environments is below one, so the genes that matter in one are not
//! exactly the genes that matter in the other. This cannot be produced by a
//! change of scale, and it is usually the interesting claim.
//!
//! Unlike the exponential surface in [`crate::gxe`], the correlation here is a
//! parameter rather than a kernel, so it is free to be negative: a genotype
//! that raises a trait in one environment and lowers it in the other is
//! reachable.
//!
//! # The residual differs too, and that is not a genetic finding
//!
//! The two residual standard deviations are free, and they should be. A trait
//! that is simply noisier in one environment would otherwise push its extra
//! variance into the genetic term, and the genetic tests would reject because
//! of measurement rather than because of genes. The residual carrying its own two
//! parameters is what keeps the genetic tests answering the question they are
//! asked.

use nalgebra::{DMatrix, DVector};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::blocks::family_blocks;
use crate::deviance::chi2_upper_tail;
use crate::dense::DenseFactor;

/// The five parameters, in the order the search holds them.
const PARAMETERS: usize = 5;
const GENETIC_FIRST: usize = 0;
const GENETIC_SECOND: usize = 1;
const RESIDUAL_FIRST: usize = 2;
const RESIDUAL_SECOND: usize = 3;
const CORRELATION: usize = 4;

/// A fitted discrete gene-by-environment model.
#[derive(Clone, Debug)]
pub struct DiscreteGxeFit {
    /// The two environment labels, smaller first. The first group everywhere
    /// in this record is the people carrying the smaller label.
    pub levels: [f64; 2],
    /// Genetic variance in the first group, then the second.
    pub genetic_variance: [f64; 2],
    /// Residual variance in the first group, then the second.
    pub residual_variance: [f64; 2],
    /// The genetic correlation across the two groups. One means the same genes
    /// act in both.
    pub correlation: f64,
    pub fixed_effects: Vec<f64>,
    pub fixed_effect_errors: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    pub scaled_gradient: f64,
    pub estimator: &'static str,
    /// How many people fell in each group, because a correlation estimated
    /// across a group of thirty is not the same claim as one across a thousand.
    pub counts: [usize; 2],
}

impl DiscreteGxeFit {
    /// The heritability within one group.
    #[must_use]
    pub fn heritability(&self, group: usize) -> f64 {
        let genetic = self.genetic_variance[group];
        let total = genetic + self.residual_variance[group];
        if total > 0.0 {
            genetic / total
        } else {
            f64::NAN
        }
    }
}

/// One trait, one relationship matrix, and a binary environment.
pub struct DiscreteGxeModel {
    relationship: DMatrix<f64>,
    design: DMatrix<f64>,
    /// The two environment labels, smaller first.
    levels: [f64; 2],
    /// True where the person carries the smaller label.
    first: Vec<bool>,
    blocks: Vec<Vec<usize>>,
    rows: usize,
    logdet_xtx: f64,
}

struct Evaluation {
    negative_loglik: f64,
    gradient: [f64; PARAMETERS],
    fixed_effects: Vec<f64>,
    fixed_covariance: Option<DMatrix<f64>>,
}

impl DiscreteGxeModel {
    /// Validate and prepare.
    ///
    /// `environment` is one label per person and must take exactly two
    /// distinct finite values — sex coded 1 and 2, an exposure coded 0 and 1,
    /// or any other binary labelling. Labels are compared exactly, the people
    /// carrying the smaller label form the first group, and **a missing or
    /// unknown label must be resolved or removed before building**, because a
    /// model that quietly puts the unknowns together is estimating a
    /// correlation with a third group in it.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model.
    pub fn build(
        relationship: &DMatrix<f64>,
        environment: &[f64],
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let rows = environment.len();
        if rows == 0 {
            return Err("DISCRETE_GXE_NO_ROWS");
        }
        if relationship.nrows() != rows || relationship.ncols() != rows {
            return Err("DISCRETE_GXE_RELATIONSHIP_WRONG_SHAPE");
        }
        if design.nrows() != rows {
            return Err("DISCRETE_GXE_DESIGN_WRONG_SHAPE");
        }
        if design.ncols() == 0 {
            return Err("DISCRETE_GXE_DESIGN_HAS_NO_COLUMNS");
        }
        if !relationship.iter().all(|v| v.is_finite()) || !design.iter().all(|v| v.is_finite()) {
            return Err("DISCRETE_GXE_NOT_FINITE");
        }
        let mut levels: Vec<f64> = Vec::with_capacity(2);
        for value in environment {
            if !value.is_finite() {
                return Err("DISCRETE_GXE_ENVIRONMENT_NOT_FINITE");
            }
            if !levels.contains(value) {
                levels.push(*value);
            }
            if levels.len() > 2 {
                return Err("DISCRETE_GXE_ENVIRONMENT_NOT_TWO_LEVELS");
            }
        }
        if levels.len() != 2 {
            return Err("DISCRETE_GXE_ENVIRONMENT_NOT_TWO_LEVELS");
        }
        levels.sort_by(|left, right| left.partial_cmp(right).expect("finite labels"));
        let levels = [levels[0], levels[1]];
        let first: Vec<bool> = environment.iter().map(|value| *value == levels[0]).collect();
        let counted = first.iter().filter(|f| **f).count();
        // A group of one has no within-group pair, so its genetic standard
        // deviation rests on nothing and the correlation is unidentified.
        if counted < 2 || rows - counted < 2 {
            return Err("DISCRETE_GXE_A_GROUP_IS_TOO_SMALL");
        }
        let xtx = design.transpose() * design;
        let chol = xtx.cholesky().ok_or("DISCRETE_GXE_DESIGN_RANK_DEFICIENT")?;
        let logdet_xtx = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        Ok(Self {
            relationship: relationship.clone(),
            design: design.clone(),
            levels,
            first,
            blocks: family_blocks(relationship),
            rows,
            logdet_xtx,
        })
    }

    /// The two environment labels, smaller first.
    #[must_use]
    pub fn levels(&self) -> [f64; 2] {
        self.levels
    }

    /// How many people are in each group.
    #[must_use]
    pub fn counts(&self) -> [usize; 2] {
        let first = self.first.iter().filter(|f| **f).count();
        [first, self.rows - first]
    }

    /// One family block's covariance at these parameters.
    fn assemble(&self, block: &[usize], theta: &[f64; PARAMETERS]) -> DMatrix<f64> {
        let size = block.len();
        DMatrix::from_fn(size, size, |i, j| {
            let (a, b) = (block[i], block[j]);
            let genetic_a = if self.first[a] {
                theta[GENETIC_FIRST]
            } else {
                theta[GENETIC_SECOND]
            };
            let genetic_b = if self.first[b] {
                theta[GENETIC_FIRST]
            } else {
                theta[GENETIC_SECOND]
            };
            let correlation = if self.first[a] == self.first[b] {
                1.0
            } else {
                theta[CORRELATION]
            };
            let mut value = self.relationship[(a, b)] * genetic_a * genetic_b * correlation;
            if a == b {
                let residual = if self.first[a] {
                    theta[RESIDUAL_FIRST]
                } else {
                    theta[RESIDUAL_SECOND]
                };
                value += residual * residual;
            }
            value
        })
    }

    /// One family block's derivative with respect to one parameter.
    fn derivative(
        &self,
        block: &[usize],
        theta: &[f64; PARAMETERS],
        parameter: usize,
    ) -> DMatrix<f64> {
        let size = block.len();
        DMatrix::from_fn(size, size, |i, j| {
            let (a, b) = (block[i], block[j]);
            let (first_a, first_b) = (self.first[a], self.first[b]);
            let same = first_a == first_b;
            let correlation = if same { 1.0 } else { theta[CORRELATION] };
            let genetic_a = if first_a {
                theta[GENETIC_FIRST]
            } else {
                theta[GENETIC_SECOND]
            };
            let genetic_b = if first_b {
                theta[GENETIC_FIRST]
            } else {
                theta[GENETIC_SECOND]
            };
            let kinship = self.relationship[(a, b)];
            match parameter {
                GENETIC_FIRST | GENETIC_SECOND => {
                    let wanted = parameter == GENETIC_FIRST;
                    // The product rule: either end of the pair may belong to
                    // the group whose standard deviation is moving, and both
                    // may.
                    let mut total = 0.0;
                    if first_a == wanted {
                        total += genetic_b;
                    }
                    if first_b == wanted {
                        total += genetic_a;
                    }
                    kinship * correlation * total
                }
                RESIDUAL_FIRST | RESIDUAL_SECOND => {
                    let wanted = parameter == RESIDUAL_FIRST;
                    if a == b && first_a == wanted {
                        2.0 * theta[parameter]
                    } else {
                        0.0
                    }
                }
                // The correlation touches only cells crossing the two groups.
                _ => {
                    if same {
                        0.0
                    } else {
                        kinship * genetic_a * genetic_b
                    }
                }
            }
        })
    }

    /// Evaluate the objective, and its gradient when asked.
    fn evaluate(
        &self,
        theta: &[f64; PARAMETERS],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
    ) -> Option<Evaluation> {
        if !theta.iter().all(|v| v.is_finite())
            || theta[GENETIC_FIRST] < 0.0
            || theta[GENETIC_SECOND] < 0.0
            || theta[RESIDUAL_FIRST] < 0.0
            || theta[RESIDUAL_SECOND] < 0.0
            || !(-1.0..=1.0).contains(&theta[CORRELATION])
        {
            return None;
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
        if !value.is_finite() {
            return None;
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
                let mut total = trace - quadratic_term;
                if reml {
                    total -= (&xvx_inverse * restricted).trace();
                }
                gradient[parameter] = 0.5 * total;
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
    pub fn fit(&self, y: &DVector<f64>, reml: bool) -> Result<DiscreteGxeFit, &'static str> {
        self.fit_under(y, reml, Constraint::default())
    }

    /// Fit under a constraint, which is how every null here is imposed.
    ///
    /// The constraint reduces what the search moves rather than penalising a
    /// free search, so the null is a smaller model fitted by exactly the same
    /// code as the alternative. A null fitted by different machinery from the
    /// alternative is the classic way to get a deviance that is not one.
    fn fit_under(
        &self,
        y: &DVector<f64>,
        reml: bool,
        constraint: Constraint,
    ) -> Result<DiscreteGxeFit, &'static str> {
        if y.len() != self.rows {
            return Err("DISCRETE_GXE_RESPONSE_WRONG_LENGTH");
        }
        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        if !(variance > 0.0) {
            return Err("DISCRETE_GXE_RESPONSE_CONSTANT");
        }
        let scale = variance.sqrt();
        let scaled = y / scale;

        let width = constraint.dimension();
        let mut lower = Vec::with_capacity(width);
        let mut upper = Vec::with_capacity(width);
        for _ in 0..constraint.genetic_slots() + constraint.residual_slots() {
            lower.push(0.0);
            upper.push(f64::INFINITY);
        }
        if constraint.correlation.is_none() {
            lower.push(-1.0);
            upper.push(1.0);
        }

        // Starts spanning equal groups, unequal groups either way, and a
        // correlation both at its bound and well inside it. A correlation that
        // is genuinely one is found from any of them; one that is low is not
        // always found from a start sitting on the bound.
        let starts: Vec<Vec<f64>> = [
            [0.7, 0.7, 0.7, 0.7, 1.0],
            [0.7, 0.7, 0.7, 0.7, 0.3],
            [0.9, 0.5, 0.6, 0.8, 0.8],
            [0.5, 0.9, 0.8, 0.6, 0.8],
            [0.6, 0.6, 0.8, 0.8, -0.2],
        ]
        .into_iter()
        .map(|start| constraint.reduce(&start))
        .collect();

        let value_of = |candidate: &[f64]| -> f64 {
            let theta = constraint.expand(candidate);
            self.evaluate(&theta, &scaled, reml, false)
                .map_or(1e30, |e| e.negative_loglik)
        };
        let gradient_of = |candidate: &[f64]| -> Vec<f64> {
            let theta = constraint.expand(candidate);
            self.evaluate(&theta, &scaled, reml, true)
                .map_or_else(|| vec![0.0; width], |e| constraint.fold(&e.gradient))
        };

        let mut best: Option<(f64, Vec<f64>, Vec<f64>, Option<DMatrix<f64>>)> = None;
        for start in starts {
            let Ok(bounds) = Bounds::new(lower.clone(), upper.clone()) else {
                continue;
            };
            let mut control = OptimControl::default_for_dimension(width);
            control.maxit = 300;
            control.fnscale = value_of(&start).abs().max(1.0);
            control.parscale = vec![1.0; width];
            control.factr = 1.0e3;
            control.pgtol = 1e-9;
            control.lmm = width;
            let Ok(solution) =
                optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control)
            else {
                continue;
            };
            let theta = constraint.expand(&solution.par);
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
        let (negative, reduced, beta, covariance) = best.ok_or("DISCRETE_GXE_NO_START_CONVERGED")?;

        let par = constraint.expand(&reduced);
        let at = self
            .evaluate(&par, &scaled, reml, true)
            .ok_or("DISCRETE_GXE_OPTIMUM_NOT_EVALUABLE")?;
        // Judge convergence in the space the search actually moved in, and only
        // in the direction it was free to move: a gradient pushing outward
        // through a bound is resolved by the bound, not left unconverged.
        let folded = constraint.fold(&at.gradient);
        let projected = folded
            .iter()
            .enumerate()
            .map(|(k, g)| {
                if reduced[k] <= lower[k] {
                    g.min(0.0)
                } else if reduced[k] >= upper[k] {
                    g.max(0.0)
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
        Ok(DiscreteGxeFit {
            levels: self.levels,
            genetic_variance: [
                par[GENETIC_FIRST].powi(2) * variance,
                par[GENETIC_SECOND].powi(2) * variance,
            ],
            residual_variance: [
                par[RESIDUAL_FIRST].powi(2) * variance,
                par[RESIDUAL_SECOND].powi(2) * variance,
            ],
            correlation: par[CORRELATION],
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
            counts: self.counts(),
        })
    }

    /// Do the two environments carry the same genetic standard deviation?
    ///
    /// This is a difference of **scale**, and a scale difference is the thing
    /// most easily produced by something other than genetics: a trait measured
    /// in units that run larger in one environment varies more there whatever
    /// the genes do. Read it as a scale finding, not as a genetic one.
    ///
    /// Both standard deviations stay strictly inside their bound under this
    /// null -- nothing is being pushed to nought -- so the reference is an
    /// ordinary chi-square on one degree of freedom.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit failed.
    pub fn genetic_equality_test(
        &self,
        y: &DVector<f64>,
        reml: bool,
    ) -> Result<DiscreteGxeTest, &'static str> {
        let free = self.fit(y, reml)?;
        let null = self.fit_under(
            y,
            reml,
            Constraint {
                genetic: Tie::Together,
                ..Constraint::default()
            },
        )?;
        Ok(mixture(free.loglik, null.loglik, "chi2_1", |t| {
            chi2_upper_tail(t, 1.0)
        }))
    }

    /// Do the two environments carry the same residual standard deviation?
    ///
    /// This is here to be *reported alongside* the genetic tests rather than
    /// as a finding of its own. A trait noisier in one environment is a
    /// measurement fact.
    /// What matters is that the model held it separately while testing the
    /// genes, so the genetic tests were not answering this question by
    /// accident.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit failed.
    pub fn residual_equality_test(
        &self,
        y: &DVector<f64>,
        reml: bool,
    ) -> Result<DiscreteGxeTest, &'static str> {
        let free = self.fit(y, reml)?;
        let null = self.fit_under(
            y,
            reml,
            Constraint {
                residual: Tie::Together,
                ..Constraint::default()
            },
        )?;
        Ok(mixture(free.loglik, null.loglik, "chi2_1", |t| {
            chi2_upper_tail(t, 1.0)
        }))
    }

    /// Do the same genes act in both environments?
    ///
    /// **This is the gene-by-environment question proper.** A correlation
    /// below one says the genetic effects are not the same in the two
    /// environments, and unlike the variance tests it cannot be produced by a
    /// difference of scale or of measurement units.
    ///
    /// The null puts the correlation at one, which is the edge of what it may
    /// be, so half the null fits land exactly on it and the deviance carries a
    /// point mass at nought. The reference is the even mixture of that mass
    /// with chi-square on one degree of freedom. Using a plain chi-square here
    /// would roughly double the p-value and throw away real power.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit failed.
    pub fn correlation_test(&self, y: &DVector<f64>, reml: bool) -> Result<DiscreteGxeTest, &'static str> {
        let free = self.fit(y, reml)?;
        let null = self.fit_under(
            y,
            reml,
            Constraint {
                correlation: Some(1.0),
                ..Constraint::default()
            },
        )?;
        Ok(mixture(free.loglik, null.loglik, "mixture_50_50", |t| {
            0.5 * chi2_upper_tail(t, 1.0)
        }))
    }

    /// Is there gene-by-environment at all?
    ///
    /// **This is the headline test.** The null says the genes are the same in
    /// both environments and carry the same variance, while leaving the two
    /// residual variances free. Two constraints, of which one -- the correlation at one
    /// -- sits on a bound, so the reference is the even mixture of chi-square
    /// on one and on two degrees of freedom.
    ///
    /// Leaving the residuals free is the whole point. A trait measured more
    /// noisily in one environment is not a genetic finding, and a test that
    /// tied the residuals would reject on exactly that. See
    /// [`Self::any_difference_test`], which does tie them and is a different
    /// question.
    ///
    /// Read this before the single-parameter tests. It is what stops three
    /// tests on one trait being read as three findings.
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit failed.
    pub fn gene_by_environment_test(&self, y: &DVector<f64>, reml: bool) -> Result<DiscreteGxeTest, &'static str> {
        let free = self.fit(y, reml)?;
        let null = self.fit_under(
            y,
            reml,
            Constraint {
                genetic: Tie::Together,
                correlation: Some(1.0),
                ..Constraint::default()
            },
        )?;
        Ok(mixture(
            free.loglik,
            null.loglik,
            "mixture_chi2_1_chi2_2",
            |t| 0.5 * chi2_upper_tail(t, 1.0) + 0.5 * chi2_upper_tail(t, 2.0),
        ))
    }

    /// Does anything at all differ between the environments?
    ///
    /// The null is the ordinary polygenic model: one genetic standard
    /// deviation, one residual standard deviation, and the same genes acting
    /// in both environments. Three constraints, of which one sits on a bound, so the
    /// reference is the even mixture of chi-square on two and on three degrees
    /// of freedom. This is the null the recovered code tested.
    ///
    /// **It is not a genetic test and must not be reported as one.** It ties
    /// the two residual variances, so a trait simply measured more noisily in
    /// one environment rejects it, hard, with nothing genetic happening at
    /// all. In simulation on the GOBS pedigree, a sex difference in
    /// measurement error alone rejected this null at p = 1e-34 while every
    /// genetic test correctly reported nothing.
    ///
    /// What it is good for is a first look at whether the environments need
    /// modelling separately in any respect. For the genetic question use
    /// [`Self::gene_by_environment_test`].
    ///
    /// # Errors
    ///
    /// Returns a stable code where either fit failed.
    pub fn any_difference_test(
        &self,
        y: &DVector<f64>,
        reml: bool,
    ) -> Result<DiscreteGxeTest, &'static str> {
        let free = self.fit(y, reml)?;
        let null = self.fit_under(
            y,
            reml,
            Constraint {
                genetic: Tie::Together,
                residual: Tie::Together,
                correlation: Some(1.0),
            },
        )?;
        Ok(mixture(
            free.loglik,
            null.loglik,
            "mixture_chi2_2_chi2_3",
            |t| 0.5 * chi2_upper_tail(t, 2.0) + 0.5 * chi2_upper_tail(t, 3.0),
        ))
    }
}

/// Whether a pair of parameters is free or shared.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
enum Tie {
    #[default]
    Apart,
    Together,
}

/// Which parameters a fit is allowed to move.
#[derive(Clone, Copy, Default)]
struct Constraint {
    genetic: Tie,
    residual: Tie,
    /// Where the correlation is pinned, if it is pinned at all.
    correlation: Option<f64>,
}

impl Constraint {
    fn genetic_slots(&self) -> usize {
        if self.genetic == Tie::Together { 1 } else { 2 }
    }

    fn residual_slots(&self) -> usize {
        if self.residual == Tie::Together { 1 } else { 2 }
    }

    fn dimension(&self) -> usize {
        self.genetic_slots() + self.residual_slots() + usize::from(self.correlation.is_none())
    }

    /// Widen what the search moves into the five the model is written in.
    fn expand(&self, reduced: &[f64]) -> [f64; PARAMETERS] {
        let mut theta = [0.0; PARAMETERS];
        let mut at = 0;
        theta[GENETIC_FIRST] = reduced[at];
        at += 1;
        theta[GENETIC_SECOND] = if self.genetic == Tie::Together {
            theta[GENETIC_FIRST]
        } else {
            let value = reduced[at];
            at += 1;
            value
        };
        theta[RESIDUAL_FIRST] = reduced[at];
        at += 1;
        theta[RESIDUAL_SECOND] = if self.residual == Tie::Together {
            theta[RESIDUAL_FIRST]
        } else {
            let value = reduced[at];
            at += 1;
            value
        };
        theta[CORRELATION] = self.correlation.unwrap_or_else(|| reduced[at]);
        theta
    }

    /// Carry a gradient in the five back into what the search moves. A shared
    /// parameter collects the derivative through both places it appears, which
    /// is the chain rule and not an approximation.
    fn fold(&self, gradient: &[f64; PARAMETERS]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.dimension());
        if self.genetic == Tie::Together {
            out.push(gradient[GENETIC_FIRST] + gradient[GENETIC_SECOND]);
        } else {
            out.push(gradient[GENETIC_FIRST]);
            out.push(gradient[GENETIC_SECOND]);
        }
        if self.residual == Tie::Together {
            out.push(gradient[RESIDUAL_FIRST] + gradient[RESIDUAL_SECOND]);
        } else {
            out.push(gradient[RESIDUAL_FIRST]);
            out.push(gradient[RESIDUAL_SECOND]);
        }
        if self.correlation.is_none() {
            out.push(gradient[CORRELATION]);
        }
        out
    }

    /// Turn a start written in the five into one the search can take. A tied
    /// pair starts at the middle of what the free start proposed.
    fn reduce(&self, full: &[f64; PARAMETERS]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.dimension());
        if self.genetic == Tie::Together {
            out.push(0.5 * (full[GENETIC_FIRST] + full[GENETIC_SECOND]));
        } else {
            out.push(full[GENETIC_FIRST]);
            out.push(full[GENETIC_SECOND]);
        }
        if self.residual == Tie::Together {
            out.push(0.5 * (full[RESIDUAL_FIRST] + full[RESIDUAL_SECOND]));
        } else {
            out.push(full[RESIDUAL_FIRST]);
            out.push(full[RESIDUAL_SECOND]);
        }
        if self.correlation.is_none() {
            out.push(full[CORRELATION]);
        }
        out
    }
}

/// The result of one discrete gene-by-environment test.
#[derive(Clone, Debug)]
pub struct DiscreteGxeTest {
    /// Twice the difference in log likelihood, never below nought.
    pub statistic: f64,
    pub p_value: f64,
    /// Which reference distribution was used, so a reader can tell what the
    /// p-value is a tail of.
    pub rule: &'static str,
    pub null_loglik: f64,
    pub alternative_loglik: f64,
}

/// Assemble a test from two log likelihoods and a reference tail.
///
/// The clamped statistic and the point mass at nought both live in
/// [`crate::deviance`], which explains why each is needed.
fn mixture(
    alternative: f64,
    null: f64,
    rule: &'static str,
    tail: impl Fn(f64) -> f64,
) -> DiscreteGxeTest {
    let statistic = crate::deviance::deviance(alternative, null);
    DiscreteGxeTest {
        statistic,
        p_value: crate::deviance::p_value(statistic, tail),
        rule,
        null_loglik: null,
        alternative_loglik: alternative,
    }
}


/// A 95 per cent profile-likelihood interval for the genetic correlation.
#[derive(Clone, Copy, Debug)]
pub struct DiscreteGxeInterval {
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
    /// True where the endpoint sat at the edge of what a correlation may be
    /// rather than where the likelihood fell away. An interval that reaches a
    /// bound is coverage without precision, and saying so is the difference
    /// between a wide answer and no answer.
    pub lower_limited: bool,
    pub upper_limited: bool,
}

impl DiscreteGxeModel {
    /// A 95 per cent profile-likelihood interval for the genetic correlation.
    ///
    /// The correlation is a parameter of this model rather than a function of
    /// one, so profiling it is a matter of pinning it and refitting everything
    /// else. The endpoints solve
    ///
    /// ```text
    /// 2 [ loglik(free) - loglik(correlation held) ] = 3.8415
    /// ```
    ///
    /// **The reference is the ordinary chi-square on one degree of freedom and
    /// not the mixture the correlation *test* uses.** The test asks about a
    /// correlation of exactly one, which is the edge of the parameter space; an
    /// interval is a statement about interior values and takes the interior
    /// reference. Using the test's mixture here would give a narrower interval
    /// than the coverage it claims.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the free fit fails, or where the profile
    /// cannot be evaluated at the estimate itself.
    pub fn correlation_interval(
        &self,
        y: &DVector<f64>,
        reml: bool,
    ) -> Result<DiscreteGxeInterval, &'static str> {
        let free = self.fit(y, reml)?;
        let estimate = free.correlation;
        if !estimate.is_finite() {
            return Err("DISCRETE_GXE_CORRELATION_NOT_FINITE");
        }
        // 3.841458820694124 / 2, the drop in log likelihood that a 95 per cent
        // interval on one degree of freedom allows.
        let target = free.loglik - 1.920_729_410_347_062;

        let profile = |correlation: f64| -> Option<f64> {
            self.fit_under(
                y,
                reml,
                Constraint {
                    correlation: Some(correlation),
                    ..Constraint::default()
                },
            )
            .ok()
            .map(|fit| fit.loglik)
        };
        // The profile at the estimate must be reachable, or nothing below it is.
        profile(estimate).ok_or("DISCRETE_GXE_PROFILE_NOT_EVALUABLE")?;

        // Walk outward from the estimate to each bound, bisecting where the
        // likelihood crosses. A bound reached without crossing is reported as
        // reached rather than as an endpoint.
        let endpoint = |bound: f64| -> (f64, bool) {
            let at_bound = profile(bound);
            if at_bound.is_none_or(|value| value >= target) {
                return (bound, true);
            }
            let (mut inside, mut outside) = (estimate, bound);
            for _ in 0..80 {
                let middle = 0.5 * (inside + outside);
                match profile(middle) {
                    Some(value) if value >= target => inside = middle,
                    _ => outside = middle,
                }
            }
            (0.5 * (inside + outside), false)
        };
        let (lower, lower_limited) = endpoint(-1.0);
        let (upper, upper_limited) = endpoint(1.0);

        Ok(DiscreteGxeInterval {
            estimate,
            lower,
            upper,
            lower_limited,
            upper_limited,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Constraint, DiscreteGxeModel, Tie};
    use nalgebra::{DMatrix, DVector};

    /// A stream of standard normals. Box-Muller rather than a sum of uniforms:
    /// the tests below judge p-values, and a p-value is a statement about a tail
    /// that a bounded approximation to a normal does not have.
    struct Normals {
        state: u64,
        spare: Option<f64>,
    }

    impl Normals {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(1),
                spare: None,
            }
        }

        fn uniform(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((self.state >> 11) as f64) + 0.5) / (1u64 << 53) as f64
        }

        fn next(&mut self) -> f64 {
            if let Some(value) = self.spare.take() {
                return value;
            }
            let (u, v) = (self.uniform(), self.uniform());
            let radius = (-2.0 * u.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * v;
            self.spare = Some(radius * angle.sin());
            radius * angle.cos()
        }
    }

    /// Full sibships of four: two of the first sex, two of the second.
    ///
    /// **Both kinds of pair have to be inside a family**, or nothing identifies
    /// the model. A same-sex pair is what carries a sex's own genetic variance
    /// and an opposite-sex pair is the only place the correlation between the
    /// sexes appears at all. A design of same-sex families alone would fit and
    /// mean nothing.
    fn sibships(
        families: usize,
        genetic: [f64; 2],
        residual: [f64; 2],
        correlation: f64,
        seed: u64,
    ) -> (DMatrix<f64>, Vec<f64>, DMatrix<f64>, DVector<f64>) {
        let n = 4 * families;
        let mut a = DMatrix::<f64>::identity(n, n);
        let group: Vec<f64> = (0..n).map(|i| if i % 4 < 2 { 1.0 } else { 2.0 }).collect();
        for family in 0..families {
            for i in 0..4 {
                for j in 0..4 {
                    if i != j {
                        a[(4 * family + i, 4 * family + j)] = 0.5;
                    }
                }
            }
        }

        let mut normals = Normals::new(seed);
        let mut y = DVector::<f64>::zeros(n);
        for family in 0..families {
            // One family's covariance, drawn from directly. The whole matrix is
            // block diagonal, so there is nothing to gain from factoring it
            // entire and a great deal of time to lose.
            let block = DMatrix::from_fn(4, 4, |i, j| {
                let (p, q) = (4 * family + i, 4 * family + j);
                let first = |k: usize| group[k] == 1.0;
                let sd = |k: usize| if first(k) { genetic[0] } else { genetic[1] };
                let across = if first(p) == first(q) {
                    1.0
                } else {
                    correlation
                };
                let mut value = a[(p, q)] * sd(p) * sd(q) * across;
                if i == j {
                    let e = if first(p) { residual[0] } else { residual[1] };
                    value += e * e;
                }
                value
            });
            let factor = block
                .cholesky()
                .expect("the simulating covariance is positive definite")
                .l();
            let draw = DVector::from_iterator(4, (0..4).map(|_| normals.next()));
            let simulated = factor * draw;
            for i in 0..4 {
                y[4 * family + i] = simulated[i];
            }
        }
        (a, group, DMatrix::from_element(n, 1, 1.0), y)
    }

    /// The gradient runs through a Cholesky per family and a product rule with
    /// three ways for a parameter to appear in one cell. It is not the kind of
    /// thing to take on trust, in either estimator.
    #[test]
    fn the_gradient_matches_a_central_difference() {
        let (a, group, design, y) = sibships(60, [0.9, 0.6], [0.7, 0.8], 0.6, 11);
        let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
        let theta = [0.85, 0.55, 0.72, 0.83, 0.55];
        for reml in [false, true] {
            let at = model.evaluate(&theta, &y, reml, true).expect("evaluable");
            for parameter in 0..super::PARAMETERS {
                let step = 1e-6;
                let mut up = theta;
                let mut down = theta;
                up[parameter] += step;
                down[parameter] -= step;
                let high = model
                    .evaluate(&up, &y, reml, false)
                    .expect("evaluable")
                    .negative_loglik;
                let low = model
                    .evaluate(&down, &y, reml, false)
                    .expect("evaluable")
                    .negative_loglik;
                let numerical = (high - low) / (2.0 * step);
                assert!(
                    (at.gradient[parameter] - numerical).abs() < 1e-5 * numerical.abs().max(1.0),
                    "parameter {parameter} in {}: analytic {} against numerical {numerical}",
                    if reml { "reml" } else { "ml" },
                    at.gradient[parameter]
                );
            }
        }
    }

    /// A tied fit must give the same answer whichever way it is asked for: the
    /// folded gradient is the chain rule through a shared parameter, and a sign
    /// or a missing term there would not show up as a crash.
    #[test]
    fn a_tied_gradient_is_the_sum_of_the_two_it_ties() {
        let (a, group, design, y) = sibships(60, [0.9, 0.6], [0.7, 0.8], 0.6, 12);
        let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
        let constraint = Constraint {
            genetic: Tie::Together,
            residual: Tie::Together,
            correlation: Some(1.0),
        };
        let reduced = vec![0.8, 0.75];
        let theta = constraint.expand(&reduced);
        assert_eq!(theta, [0.8, 0.8, 0.75, 0.75, 1.0]);
        let at = model.evaluate(&theta, &y, false, true).expect("evaluable");
        let folded = constraint.fold(&at.gradient);
        assert_eq!(folded.len(), 2);
        for (parameter, expected) in folded.iter().enumerate() {
            let step = 1e-6;
            let mut up = reduced.clone();
            let mut down = reduced.clone();
            up[parameter] += step;
            down[parameter] -= step;
            let high = model
                .evaluate(&constraint.expand(&up), &y, false, false)
                .expect("evaluable")
                .negative_loglik;
            let low = model
                .evaluate(&constraint.expand(&down), &y, false, false)
                .expect("evaluable")
                .negative_loglik;
            let numerical = (high - low) / (2.0 * step);
            assert!(
                (expected - numerical).abs() < 1e-5 * numerical.abs().max(1.0),
                "tied parameter {parameter}: folded {expected} against numerical {numerical}"
            );
        }
    }

    /// A difference of genetic **scale** between the sexes is found, and named
    /// as that rather than as a difference of genes: the correlation is
    /// simulated at one and must not be reported below it.
    #[test]
    fn a_genetic_variance_difference_between_the_sexes_is_recovered() {
        let (a, group, design, y) = sibships(250, [1.1, 0.45], [0.7, 0.7], 1.0, 21);
        let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
        let fit = model.fit(&y, true).expect("the free model fits");
        assert!(fit.converged, "scaled gradient {}", fit.scaled_gradient);
        assert!(
            fit.heritability(0) > fit.heritability(1),
            "heritabilities {} and {}",
            fit.heritability(0),
            fit.heritability(1)
        );
        let scale = model
            .genetic_equality_test(&y, true)
            .expect("the test runs");
        assert!(
            scale.p_value < 0.01,
            "scale difference missed at p = {}",
            scale.p_value
        );
        assert_eq!(scale.rule, "chi2_1");
        // The genes are the same in both sexes here, and the model must not
        // say otherwise merely because the variances differ.
        let genes = model.correlation_test(&y, true).expect("the test runs");
        assert!(
            genes.p_value > 0.05,
            "a scale difference was read as different genes at p = {}",
            genes.p_value
        );
    }

    /// The gene-by-environment finding proper: the same variance in both
    /// groups, but not the same genes.
    #[test]
    fn a_correlation_below_one_is_recovered() {
        let (a, group, design, y) = sibships(250, [1.0, 1.0], [0.7, 0.7], 0.25, 31);
        let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
        let fit = model.fit(&y, true).expect("the free model fits");
        assert!(
            fit.correlation < 0.75,
            "correlation came back at {}",
            fit.correlation
        );
        let genes = model.correlation_test(&y, true).expect("the test runs");
        assert!(
            genes.p_value < 0.01,
            "different genes missed at p = {}",
            genes.p_value
        );
        assert_eq!(genes.rule, "mixture_50_50");
        // The variances are equal here and must be reported as equal.
        let scale = model
            .genetic_equality_test(&y, true)
            .expect("the test runs");
        assert!(
            scale.p_value > 0.05,
            "different genes were read as a scale difference at p = {}",
            scale.p_value
        );
        let overall = model.gene_by_environment_test(&y, true).expect("the test runs");
        assert!(
            overall.p_value < 0.01,
            "the headline test missed it at p = {}",
            overall.p_value
        );
        assert_eq!(overall.rule, "mixture_chi2_1_chi2_2");
    }

    /// Nothing is invented where there is nothing to find.
    ///
    /// Read across seeds rather than from one draw. A single null draw says
    /// almost nothing: the correlation is bounded above at one, so every
    /// departure it can show runs the same way, and a test that happened to
    /// pass once would pass for no reason worth trusting.
    #[test]
    fn nothing_is_invented_under_a_null() {
        let mut rejected = 0;
        let mut on_the_bound = 0;
        let seeds = 12;
        for seed in 0..seeds {
            let (a, group, design, y) = sibships(150, [0.9, 0.9], [0.7, 0.7], 1.0, 400 + seed);
            let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
            let fit = model.fit(&y, true).expect("the free model fits");
            if fit.correlation > 1.0 - 1e-6 {
                on_the_bound += 1;
            }
            let overall = model.gene_by_environment_test(&y, true).expect("the test runs");
            if overall.p_value <= 0.05 {
                rejected += 1;
            }
        }
        // At a true level of 0.05, four or more rejections in twelve happens
        // about twice in a thousand runs. This is a level check, not a
        // uniformity one.
        assert!(
            rejected <= 3,
            "{rejected} of {seeds} null draws rejected at 0.05"
        );
        // Under this null the correlation is at its bound, so a good share of
        // fits should sit exactly on it. None doing so would mean the search
        // never reaches the bound, and the even mixture would then be the wrong
        // reference for every test in the module.
        assert!(
            on_the_bound >= 3,
            "only {on_the_bound} of {seeds} null fits reached the correlation bound"
        );
    }

    /// **The residual standard deviations being free is what makes the genetic
    /// tests mean anything.** One sex measured twice as noisily, with identical
    /// genetics, must not come out as a genetic difference.
    /// The interval must contain the estimate, and a correlation recovered
    /// well below one must not have an interval running to the bound.
    #[test]
    fn the_correlation_interval_contains_its_estimate() {
        let (a, group, design, y) = sibships(250, [1.0, 1.0], [0.7, 0.7], 0.25, 31);
        let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
        let interval = model.correlation_interval(&y, true).expect("intervals");
        assert!(interval.lower <= interval.estimate && interval.estimate <= interval.upper);
        assert!(interval.lower >= -1.0 && interval.upper <= 1.0);
        assert!(
            interval.upper < 1.0 && !interval.upper_limited,
            "a correlation of {} ran its interval to the bound",
            interval.estimate
        );
    }

    /// Where the genes are the same in both groups the estimate sits on the
    /// bound, and the interval says it reached there rather than pretending to
    /// an endpoint it never found.
    #[test]
    fn an_interval_that_reaches_the_bound_says_so() {
        let (a, group, design, y) = sibships(120, [0.9, 0.9], [0.7, 0.7], 1.0, 77);
        let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
        let interval = model.correlation_interval(&y, true).expect("intervals");
        assert!(interval.upper_limited, "the upper endpoint left the bound");
        assert!((interval.upper - 1.0).abs() < 1e-12);
        assert!(interval.lower < interval.estimate);
    }

    #[test]
    fn a_noisier_group_is_not_read_as_a_genetic_difference() {
        let (a, group, design, y) = sibships(250, [0.9, 0.9], [0.5, 1.2], 1.0, 41);
        let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
        let noise = model
            .residual_equality_test(&y, true)
            .expect("the test runs");
        assert!(
            noise.p_value < 0.01,
            "the noise difference was missed at p = {}",
            noise.p_value
        );
        let genes = model.correlation_test(&y, true).expect("the test runs");
        assert!(
            genes.p_value > 0.05,
            "a noisier sex was read as different genes at p = {}",
            genes.p_value
        );
        let scale = model
            .genetic_equality_test(&y, true)
            .expect("the test runs");
        assert!(
            scale.p_value > 0.05,
            "a noisier sex was read as a genetic scale difference at p = {}",
            scale.p_value
        );
        let headline = model.gene_by_environment_test(&y, true).expect("the test runs");
        assert!(
            headline.p_value > 0.05,
            "a noisier group was read as gene-by-environment at p = {}",
            headline.p_value
        );
        // **And the trap this exists to avoid.** The recovered code's overall
        // null ties the residuals, so it rejects here -- correctly, on its own
        // terms, and misleadingly if read as gene-by-environment. Pinned so that nobody
        // later promotes it back to being the headline.
        let any = model.any_difference_test(&y, true).expect("the test runs");
        assert!(
            any.p_value < 1e-6,
            "the any-difference null should reject on a noisier sex, at p = {}",
            any.p_value
        );
    }

    /// **The covariance is the source's, cell for cell.**
    ///
    /// This module assembles one family block at a time, because that is what
    /// keeps a fit to a large pedigree affordable. The recovered code built the
    /// whole matrix in one pass. The two must agree exactly, and the way to
    /// know is to write the recovered construction out here and compare, rather
    /// than to assert in a comment that they match.
    #[test]
    fn the_covariance_is_the_recovered_one() {
        let (a, group, design, _) = sibships(25, [0.9, 0.6], [0.7, 0.8], 0.4, 61);
        let model = DiscreteGxeModel::build(&a, &group, &design).expect("the model builds");
        let n = group.len();
        for theta in [
            [0.9, 0.6, 0.7, 0.8, 0.4],
            [1.3, 1.3, 0.2, 0.9, 1.0],
            // A negative correlation, which the kernel forms of gene-by-
            // environment cannot represent and this one must.
            [0.5, 1.1, 0.6, 0.6, -0.8],
            // A sex with no genetic variance at all, which is on a bound.
            [0.0, 0.8, 0.5, 0.5, 0.3],
        ] {
            // The recovered construction, written out.
            let in_x: Vec<bool> = group.iter().map(|g| *g == 1.0).collect();
            let genetic_sd: Vec<f64> = in_x
                .iter()
                .map(|x| if *x { theta[0] } else { theta[1] })
                .collect();
            let residual_sd: Vec<f64> = in_x
                .iter()
                .map(|x| if *x { theta[2] } else { theta[3] })
                .collect();
            let mut expected = DMatrix::from_fn(n, n, |row, column| {
                let correlation = if in_x[row] == in_x[column] {
                    1.0
                } else {
                    theta[4]
                };
                a[(row, column)] * genetic_sd[row] * genetic_sd[column] * correlation
            });
            for index in 0..n {
                expected[(index, index)] += residual_sd[index].powi(2);
            }
            for block in &model.blocks {
                let assembled = model.assemble(block, &theta);
                for (i, &p) in block.iter().enumerate() {
                    for (j, &q) in block.iter().enumerate() {
                        assert_eq!(
                            assembled[(i, j)],
                            expected[(p, q)],
                            "cell ({p}, {q}) differs at {theta:?}"
                        );
                    }
                }
            }
        }
    }

    /// Exactly two distinct finite labels are accepted, whatever they are; a
    /// third label or a non-finite one is refused rather than swept into a
    /// group.
    #[test]
    fn only_two_groups_are_accepted() {
        let (a, group, design, y) = sibships(10, [0.9, 0.9], [0.7, 0.7], 1.0, 51);
        assert!(DiscreteGxeModel::build(&a, &group, &design).is_ok());
        for bad in [0.0, 3.0, -1.0] {
            let mut spoiled = group.clone();
            spoiled[7] = bad;
            assert_eq!(
                DiscreteGxeModel::build(&a, &spoiled, &design).err(),
                Some("DISCRETE_GXE_ENVIRONMENT_NOT_TWO_LEVELS"),
                "a third label of {bad} was accepted"
            );
        }
        let mut not_finite = group.clone();
        not_finite[7] = f64::NAN;
        assert_eq!(
            DiscreteGxeModel::build(&a, &not_finite, &design).err(),
            Some("DISCRETE_GXE_ENVIRONMENT_NOT_FINITE")
        );
        // Any two distinct labels name the two environments; the smaller label
        // is the first group. A 0/1 exposure works exactly like sex coded 1/2.
        let recoded: Vec<f64> = group.iter().map(|v| if *v == 1.0 { 0.0 } else { 1.0 }).collect();
        let relabelled = DiscreteGxeModel::build(&a, &recoded, &design).expect("binary labels build");
        assert_eq!(relabelled.levels(), [0.0, 1.0]);
        assert_eq!(
            relabelled.counts(),
            DiscreteGxeModel::build(&a, &group, &design).expect("builds").counts()
        );
        // A group of one has no within-group pair, so its genetic standard
        // deviation rests on nothing.
        let lonely: Vec<f64> = (0..group.len())
            .map(|i| if i == 0 { 1.0 } else { 2.0 })
            .collect();
        assert_eq!(
            DiscreteGxeModel::build(&a, &lonely, &design).err(),
            Some("DISCRETE_GXE_A_GROUP_IS_TOO_SMALL")
        );
        assert!(model_rejects_short_response(&a, &group, &design, &y));
    }

    fn model_rejects_short_response(
        a: &DMatrix<f64>,
        group: &[f64],
        design: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> bool {
        let model = DiscreteGxeModel::build(a, group, design).expect("the model builds");
        let short = DVector::from_iterator(y.len() - 1, y.iter().take(y.len() - 1).copied());
        model.fit(&short, true).err() == Some("DISCRETE_GXE_RESPONSE_WRONG_LENGTH")
    }
}

#[cfg(feature = "python")]
pub mod python;
