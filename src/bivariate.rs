//! Two traits, additive and residual variance, with the correlations carried as
//! free parameters.
//!
//! Admitted by `docs/adr/0001` decision 29 for the JASA reanalysis. The
//! fixed-state check this is measured against is `checks/bivariate_reference.py`,
//! written first and deliberately: written second it would have been a
//! transcription rather than a second route. The fitted REML comparison is the
//! still more independent `checks/bivariate_against_r.py`, which calls the
//! compiled Rust fit and R's `regress` (`docs/adr/0006`).
//!
//! # The model
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
//! Six parameters, and at two traits the constraint set is simply a box: the
//! variances positive, the heritabilities in [0,1], the correlations in [−1,1].
//! Nothing further is needed to keep both covariances positive semi-definite,
//! which is what makes this parameterisation worth having here even though it
//! fails at three traits (decision 5).
//!
//! # Unbalanced by construction
//!
//! A person contributes the rows for the traits they actually have, and everyone
//! with at least one measured trait is in (decision 9). This is what breaks the
//! Kronecker structure: the eigen-rotation that makes the one-trait fit fast
//! survives extra traits only while everybody has every trait. So each family
//! block is assembled and factorised directly. On the real rosters the largest
//! family is 160 people, which makes an evaluation about 0.05 Gflop against the
//! 1.2 a single dense factorisation would cost.
//!
//! # State, as of 11 August 2026
//!
//! **The fixed-state objective, the interior score and every exact-boundary
//! state's free score are checked. The bound-constrained search meets its
//! `1e-7` projected KKT criterion for ML and REML on the deterministic
//! unbalanced comparison, including exact heritability and correlation
//! boundaries.** This establishes the numerical optimiser; it does not
//! establish standard errors, tests, intervals, coverage, or the realised JASA
//! analysis. Do not report scientific results from this yet.
//!
//! What is established:
//!
//! - This objective and the reference's agree to **1.4e-14** at the same
//!   parameters on the same data, for both estimators. They are the same
//!   function.
//! - The analytic gradient matches a central difference to better than 1e-5 at
//!   six interior points and on every free coordinate of all nine exact
//!   heritability states, for both ML and REML.
//! - A literal fixed-state ML/REML golden keeps that agreement in the Rust test
//!   suite. A separate public fit regression requires both estimators to meet
//!   the projected KKT criterion across sixty deterministic unbalanced fits,
//!   with exact heritability and correlation bounds both exercised.
//!
//! # The fault that took the longest to find, in both implementations
//!
//! At h² = 0 or 1 one component variance vanishes, making the direct `(h², ρ)`
//! score non-differentiable and its correlation unidentified. At |ρ| = 1 one
//! component is rank deficient, but the full covariance can remain positive
//! definite because the other component supplies the missing direction. Exact
//! correlation bounds are therefore valid and required for decision 29's null
//! refits; zero-variance states need a lower-dimensional parameterisation.
//!
//! - **Here**: returned a large value with a **zero gradient**, which tells a
//!   quasi-Newton method it has found a stationary point. The fit settled
//!   exactly on it.
//! - **In the reference**: returned infinity, which a finite-difference gradient
//!   turns into NaN, and scipy followed it without complaint. That version
//!   stopped at points with a gradient of 2.5 while reporting success.
//!
//! The current search enumerates all nine combinations of lower, interior and
//! upper heritability states. Interior coordinates remain epsilon-open; exact
//! endpoint states remove the correlation that ceases to be an estimand. Exact
//! correlation bounds remain part of every state where that correlation exists.
//! It is worth noting how long the earlier error hid: the reference agreed with
//! SOLAR to 1e-7 on real data while carrying it, because that optimum was
//! interior.
//!
//! An earlier note here blamed the difference between the two on the optimiser —
//! scipy's L-BFGS-B against a hand-written one. That was wrong, and swapping in
//! the same algorithm proved it: the cause was in what both were told at
//! infeasible points, not in how either searched.
//!
//! # Gradients are analytic
//!
//! Decision 14, and for the reason given there: the convergence test is the
//! scaled gradient against a tolerance, and a differenced gradient cannot
//! certify convergence below its own noise floor.
//!
//! ```text
//! ∂ℓ/∂θ = −½[ tr(V⁻¹ ∂V/∂θ) − r'V⁻¹ (∂V/∂θ) V⁻¹ r ]
//! ```
//!
//! with, for REML, a further `+½ tr[(X'V⁻¹X)⁻¹ X'V⁻¹ (∂V/∂θ) V⁻¹X]`. Since
//! `V = Σ_A ⊗ A + Σ_E ⊗ I`, every `∂V/∂θ` is `(∂Σ_A/∂θ) ⊗ A + (∂Σ_E/∂θ) ⊗ I`
//! and only the two 2×2 blocks need differentiating.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::blocks::family_blocks;

/// How many free parameters: two total variances, two heritabilities, two
/// correlations.
const PARAMETERS: usize = 6;
const INVALID_OBJECTIVE_SCALE: f64 = 1.0e20;

/// Which of the two traits a row carries.
type Row = (usize, usize);

/// A validated two-trait problem: the relationship matrix, who has which trait,
/// the design, and the family blocks the likelihood is assembled over.
pub struct BivariateModel {
    relationship: DMatrix<f64>,
    blocks: Vec<Vec<Row>>,
    design: DMatrix<f64>,
    row_positions: Vec<[Option<usize>; 2]>,
    logdet_xtx: f64,
    subjects: usize,
    rows: usize,
}

/// What a two-trait fit returns.
pub struct BivariateFit {
    pub h2: [f64; 2],
    /// Whether each heritability was selected at an exact endpoint.
    pub h2_boundary: [BivariateHeritabilityBoundary; 2],
    pub total_variance: [f64; 2],
    /// Absent where either genetic variance is exactly zero.
    pub rho_g: Option<f64>,
    /// Absent where either residual variance is exactly zero.
    pub rho_e: Option<f64>,
    /// The phenotypic correlation, derived rather than estimated.
    pub rho_p: f64,
    pub loglik: f64,
    pub converged: bool,
    pub estimator: &'static str,
    /// The largest scaled gradient at the reported point. This is the
    /// convergence test decision 14 names, and it is on the record so that a
    /// claim of convergence can be checked rather than taken.
    pub scaled_gradient: f64,
}

/// The exact state selected for one trait's heritability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BivariateHeritabilityBoundary {
    /// Exactly zero additive variance for this trait.
    Lower,
    /// Strictly between zero and one.
    Interior,
    /// Exactly zero residual variance for this trait.
    Upper,
}

/// The two covariances, and their derivatives with respect to each parameter.
fn covariances(theta: &[f64; PARAMETERS]) -> ([[f64; 2]; 2], [[f64; 2]; 2]) {
    let (s1, s2, h1, h2, rg, re) = (theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]);
    let (a1, a2) = (h1 * s1, h2 * s2);
    let (e1, e2) = ((1.0 - h1) * s1, (1.0 - h2) * s2);
    let ca = rg * (a1 * a2).max(0.0).sqrt();
    let ce = re * (e1 * e2).max(0.0).sqrt();
    ([[a1, ca], [ca, a2]], [[e1, ce], [ce, e2]])
}

/// `∂Σ_A/∂θ` and `∂Σ_E/∂θ` for one parameter.
fn derivative(theta: &[f64; PARAMETERS], which: usize) -> ([[f64; 2]; 2], [[f64; 2]; 2]) {
    let (s1, s2, h1, h2, rg, re) = (theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]);
    let (a1, a2) = (h1 * s1, h2 * s2);
    let (e1, e2) = ((1.0 - h1) * s1, (1.0 - h2) * s2);

    // d/dθ of the diagonal entries.
    let (da1, da2, de1, de2) = match which {
        0 => (h1, 0.0, 1.0 - h1, 0.0), // σ₁
        1 => (0.0, h2, 0.0, 1.0 - h2), // σ₂
        2 => (s1, 0.0, -s1, 0.0),      // h₁
        3 => (0.0, s2, 0.0, -s2),      // h₂
        _ => (0.0, 0.0, 0.0, 0.0),     // the correlations
    };

    // d/dθ of the off-diagonal, ρ√(xy): the chain rule through the square root,
    // plus the direct term when θ is the correlation itself.
    let root_a = (a1 * a2).max(0.0).sqrt();
    let root_e = (e1 * e2).max(0.0).sqrt();
    let dca = if which == 4 {
        root_a
    } else if root_a > 0.0 {
        rg * (da1 * a2 + a1 * da2) / (2.0 * root_a)
    } else {
        0.0
    };
    let dce = if which == 5 {
        root_e
    } else if root_e > 0.0 {
        re * (de1 * e2 + e1 * de2) / (2.0 * root_e)
    } else {
        0.0
    };
    ([[da1, dca], [dca, da2]], [[de1, dce], [dce, de2]])
}

/// The objective and its gradient at one point, with the fixed effects profiled
/// out by generalised least squares (decision 10).
struct Evaluation {
    negative_loglik: f64,
    gradient: [f64; PARAMETERS],
}

/// Whether the six unconstrained covariance entries can be separated from the
/// rows that were actually observed.
///
/// The model is linear in `a11`, `a22`, `a12`, `e11`, `e22`, `e12`. Their six
/// masked covariance basis matrices must therefore be linearly independent.
/// The 6×6 Frobenius Gram is the small, exact way to test that rank without
/// constructing six full observation matrices.
fn covariance_basis_is_full_rank(relationship: &DMatrix<f64>, observed: &[[bool; 2]]) -> bool {
    let rows = observed
        .iter()
        .enumerate()
        .flat_map(|(person, mask)| {
            (0..2).filter_map(move |trait_index| mask[trait_index].then_some((person, trait_index)))
        })
        .collect::<Vec<_>>();
    let mut gram = DMatrix::<f64>::zeros(PARAMETERS, PARAMETERS);
    for &(first_person, first_trait) in &rows {
        for &(second_person, second_trait) in &rows {
            let covariance_entry = if first_trait == 0 && second_trait == 0 {
                0
            } else if first_trait == 1 && second_trait == 1 {
                1
            } else {
                2
            };
            let genetic = relationship[(first_person, second_person)];
            let residual = f64::from(first_person == second_person);
            let residual_entry = 3 + covariance_entry;
            gram[(covariance_entry, covariance_entry)] += genetic * genetic;
            gram[(covariance_entry, residual_entry)] += genetic * residual;
            gram[(residual_entry, covariance_entry)] += genetic * residual;
            gram[(residual_entry, residual_entry)] += residual;
        }
    }
    let eigenvalues = SymmetricEigen::new(gram).eigenvalues;
    let largest = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    largest > 0.0 && eigenvalues.iter().all(|&value| value > 1.0e-12 * largest)
}

impl BivariateModel {
    /// Build from a relationship matrix, a per-person mask saying which traits
    /// were measured, and a design over the observed rows.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the shapes disagree or nobody is measured.
    pub fn build(
        relationship: &DMatrix<f64>,
        observed: &[[bool; 2]],
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let people = relationship.nrows();
        if relationship.ncols() != people {
            return Err("BIVARIATE_RELATIONSHIP_NOT_SQUARE");
        }
        if observed.len() != people {
            return Err("BIVARIATE_OBSERVED_LENGTH_MISMATCH");
        }
        if people == 0 {
            return Err("BIVARIATE_NOTHING_MEASURED");
        }
        if relationship.iter().any(|value| !value.is_finite()) {
            return Err("BIVARIATE_RELATIONSHIP_NOT_FINITE");
        }
        for i in 0..people {
            for j in (i + 1)..people {
                if relationship[(i, j)] != relationship[(j, i)] {
                    return Err("BIVARIATE_RELATIONSHIP_ASYMMETRIC");
                }
            }
        }
        let rows: usize = observed
            .iter()
            .map(|o| usize::from(o[0]) + usize::from(o[1]))
            .sum();
        let subjects = observed.iter().filter(|mask| mask[0] || mask[1]).count();
        if rows == 0 {
            return Err("BIVARIATE_NOTHING_MEASURED");
        }
        if design.nrows() != rows {
            return Err("BIVARIATE_DESIGN_ROWS_MISMATCH");
        }
        if design.ncols() == 0 || rows <= design.ncols() {
            return Err("BIVARIATE_NO_RESIDUAL_DEGREES_OF_FREEDOM");
        }
        if design.iter().any(|value| !value.is_finite()) {
            return Err("BIVARIATE_DESIGN_NOT_FINITE");
        }

        let gram = design.transpose() * design;
        let gram_eigenvalues = SymmetricEigen::new(gram).eigenvalues;
        let largest_gram_eigenvalue = gram_eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        if gram_eigenvalues
            .iter()
            .any(|&value| value <= 1.0e-12 * largest_gram_eigenvalue)
        {
            return Err("BIVARIATE_DESIGN_RANK_DEFICIENT");
        }
        let logdet_xtx = gram_eigenvalues.iter().map(|value| value.ln()).sum();

        // Validate each exact relationship component at its own spectral scale,
        // then clip only round-off below zero. The direct block likelihood needs
        // the same projected PSD matrix that validation accepted.
        let family_people = family_blocks(relationship);
        let mut projected_relationship = relationship.clone();
        for block in &family_people {
            let size = block.len();
            let submatrix = DMatrix::from_fn(size, size, |row, column| {
                relationship[(block[row], block[column])]
            });
            let decomposition = SymmetricEigen::new(submatrix);
            if decomposition
                .eigenvalues
                .iter()
                .any(|&value| value < -1.0e-9)
            {
                return Err("BIVARIATE_RELATIONSHIP_NOT_PSD");
            }
            let clipped =
                DMatrix::from_diagonal(&decomposition.eigenvalues.map(|value| value.max(0.0)));
            let projected =
                &decomposition.eigenvectors * clipped * decomposition.eigenvectors.transpose();
            for row in 0..size {
                for column in 0..size {
                    projected_relationship[(block[row], block[column])] = projected[(row, column)];
                }
            }
        }

        if !covariance_basis_is_full_rank(&projected_relationship, observed) {
            return Err("BIVARIATE_COVARIANCE_NOT_IDENTIFIABLE");
        }

        // Rows are ordered person-major: everyone's first trait then second,
        // skipping what was not measured. The index of a row is its position in
        // that order, which is what the design and response are indexed by.
        let mut row_positions = vec![[None; 2]; people];
        let mut next = 0usize;
        for (person, mask) in observed.iter().enumerate() {
            for trait_index in 0..2 {
                if mask[trait_index] {
                    row_positions[person][trait_index] = Some(next);
                    next += 1;
                }
            }
        }

        let blocks = family_people
            .into_iter()
            .map(|block| {
                block
                    .into_iter()
                    .flat_map(|person| {
                        (0..2).filter_map(move |t| observed[person][t].then_some((person, t)))
                    })
                    .collect::<Vec<Row>>()
            })
            .filter(|block: &Vec<Row>| !block.is_empty())
            .collect();

        Ok(Self {
            relationship: projected_relationship,
            blocks,
            design: design.clone(),
            row_positions,
            logdet_xtx,
            subjects,
            rows,
        })
    }

    /// Number of measured rows, which is not twice the number of people unless
    /// everybody has both traits.
    #[must_use]
    pub fn observations(&self) -> usize {
        self.rows
    }

    /// Number of people with at least one measured trait.
    #[must_use]
    pub fn subjects(&self) -> usize {
        self.subjects
    }

    fn row_index(&self, person: usize, trait_index: usize) -> usize {
        self.row_positions[person][trait_index]
            .expect("family blocks contain only rows committed at model construction")
    }

    /// The objective, and optionally its gradient, at one point.
    ///
    /// Two passes over the blocks, because the fixed effects are profiled out
    /// and the gradient needs the residual: the first pass accumulates
    /// `X'V⁻¹X`, `X'V⁻¹y` and `y'V⁻¹y` and keeps each block's solves, and the
    /// second uses `β̂` to form `u = V⁻¹(y − Xβ̂)` per block, after which the
    /// quadratic term is simply `u'(∂V)u`.
    fn evaluate(
        &self,
        theta: &[f64; PARAMETERS],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
    ) -> Option<Evaluation> {
        self.evaluate_for_state(theta, y, reml, want_gradient, None)
    }

    fn evaluate_for_state(
        &self,
        theta: &[f64; PARAMETERS],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
        state: Option<HeritabilityState>,
    ) -> Option<Evaluation> {
        if theta.iter().any(|value| !value.is_finite())
            || theta[0] <= 0.0
            || theta[1] <= 0.0
            || !(0.0..=1.0).contains(&theta[2])
            || !(0.0..=1.0).contains(&theta[3])
            || !(-1.0..=1.0).contains(&theta[4])
            || !(-1.0..=1.0).contains(&theta[5])
        {
            return None;
        }
        if want_gradient
            && state.is_none()
            && (theta[2] <= 0.0 || theta[2] >= 1.0 || theta[3] <= 0.0 || theta[3] >= 1.0)
        {
            // In direct (h², rho) coordinates, a vanished component makes its
            // correlation unidentified and the square-root derivative is not
            // finite. A zero gradient here would invent a stationary point.
            return None;
        }
        let (sigma_a, sigma_e) = covariances(theta);
        for sigma in [&sigma_a, &sigma_e] {
            if sigma[0][0] < 0.0 || sigma[1][1] < 0.0 {
                return None;
            }
            if sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0] < -1e-12 {
                return None;
            }
        }

        let p = self.design.ncols();
        let mut logdet = 0.0;
        let mut xvx = DMatrix::<f64>::zeros(p, p);
        let mut xvy = DVector::<f64>::zeros(p);
        let mut yvy = 0.0;
        let mut kept: Vec<BlockSolve> = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            let size = block.len();
            let v = self.assemble(block, &sigma_a, &sigma_e);
            let chol = v.cholesky()?;
            logdet += 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();

            let index: Vec<usize> = block
                .iter()
                .map(|&(person, t)| self.row_index(person, t))
                .collect();
            let yb = DVector::from_iterator(size, index.iter().map(|&i| y[i]));
            let xb = DMatrix::from_fn(size, p, |r, c| self.design[(index[r], c)]);
            let vy = chol.solve(&yb);
            let vx = chol.solve(&xb);
            xvx += xb.transpose() * &vx;
            xvy += xb.transpose() * &vy;
            yvy += yb.dot(&vy);

            if want_gradient {
                kept.push(BlockSolve {
                    rows: block.clone(),
                    inverse: chol.inverse(),
                    vy,
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

        let mut gradient = [0.0; PARAMETERS];
        if want_gradient {
            let xvx_inverse = xvx_chol.inverse();
            for k in 0..PARAMETERS {
                let (da, de) = derivative(theta, k);
                let mut trace = 0.0;
                let mut residual = 0.0;
                let mut restricted = DMatrix::<f64>::zeros(p, p);
                for solve in &kept {
                    let size = solve.rows.len();
                    let dv = self.assemble(&solve.rows, &da, &de);
                    // tr(V⁻¹ ∂V), which needs the inverse rather than a solve.
                    for i in 0..size {
                        for j in 0..size {
                            trace += solve.inverse[(i, j)] * dv[(j, i)];
                        }
                    }
                    // u = V⁻¹(y − Xβ̂), so the quadratic term is u'(∂V)u.
                    let u = &solve.vy - &solve.vx * &beta;
                    residual += u.dot(&(&dv * &u));
                    if reml {
                        restricted += solve.vx.transpose() * &dv * &solve.vx;
                    }
                }
                gradient[k] = 0.5 * (trace - residual);
                if reml {
                    gradient[k] -= 0.5 * (&xvx_inverse * &restricted).trace();
                }
            }
        }

        Some(Evaluation {
            negative_loglik: value,
            gradient,
        })
    }

    /// One family block's covariance, or one block of a derivative — the shape
    /// is the same, `Σ ⊗ A` plus `Σ_E` on the diagonal.
    fn assemble(
        &self,
        block: &[Row],
        sigma_a: &[[f64; 2]; 2],
        sigma_e: &[[f64; 2]; 2],
    ) -> DMatrix<f64> {
        let size = block.len();
        DMatrix::from_fn(size, size, |i, j| {
            let (pi, ti) = block[i];
            let (pj, tj) = block[j];
            let mut value = sigma_a[ti][tj] * self.relationship[(pi, pj)];
            if pi == pj {
                value += sigma_e[ti][tj];
            }
            value
        })
    }
}

impl BivariateModel {
    /// The objective and gradient at one point, for comparison against the
    /// independent implementation. Public because a check that cannot reach the
    /// thing it checks is not a check.
    ///
    /// # Errors
    ///
    /// `None` where the covariance is not positive definite there.
    #[must_use]
    pub fn objective_at(
        &self,
        theta: &[f64; PARAMETERS],
        y: &DVector<f64>,
        reml: bool,
    ) -> Option<(f64, [f64; PARAMETERS])> {
        self.evaluate(theta, y, reml, true)
            .map(|e| (e.negative_loglik, e.gradient))
    }
}

/// One family block's solves, kept between the two passes.
struct BlockSolve {
    rows: Vec<Row>,
    inverse: DMatrix<f64>,
    vy: DVector<f64>,
    vx: DMatrix<f64>,
}

/// The box for the fully interior state: variances positive, heritabilities in
/// `(0, 1)`, and correlations in `[-1, 1]`.
///
/// The direct `(h², ρ)` coordinates are not differentiable where a component
/// variance vanishes. The interior search therefore uses the representable
/// epsilon-open interval, while eight further lower-dimensional states fit every
/// exact combination involving zero or one. They omit the correlation that no
/// longer exists rather than fabricating its derivative.
///
/// Correlations are different: `ρ = ±1` makes one component rank deficient, but
/// the sum defining the full observation covariance may remain positive
/// definite. Those exact boundaries are therefore retained, as decision 29
/// requires for the genetic-correlation null refits.
const INTERIOR_EPSILON: f64 = f64::EPSILON;
const LOWER: [f64; PARAMETERS] = [1e-8, 1e-8, INTERIOR_EPSILON, INTERIOR_EPSILON, -1.0, -1.0];
const UPPER: [f64; PARAMETERS] = [
    f64::INFINITY,
    f64::INFINITY,
    1.0 - INTERIOR_EPSILON,
    1.0 - INTERIOR_EPSILON,
    1.0,
    1.0,
];

#[derive(Clone, Copy)]
struct HeritabilityState([BivariateHeritabilityBoundary; 2]);

struct StandardisedProblem {
    model: BivariateModel,
    y: DVector<f64>,
    variance_scale: [f64; 2],
    /// Add these constants to the standardised ML and REML objectives to
    /// recover the value on the original response scale.
    objective_shift: [f64; 2],
}

struct OptimisationCandidate {
    /// The objective on the original response scale, retained for reporting.
    negative_loglik: f64,
    /// The standardised objective, used for comparisons because it has no
    /// response-unit-dependent additive constant.
    comparison_negative_loglik: f64,
    theta: [f64; PARAMETERS],
    state: HeritabilityState,
    scaled_gradient: f64,
    converged: bool,
}

fn select_best_candidate(
    converged_candidates: Vec<OptimisationCandidate>,
    failed_candidates: &[OptimisationCandidate],
) -> Result<OptimisationCandidate, &'static str> {
    let minimum = converged_candidates
        .iter()
        .map(|candidate| candidate.comparison_negative_loglik)
        .min_by(f64::total_cmp)
        .ok_or("BIVARIATE_NO_START_CONVERGED")?;
    let objective_tie_tolerance = 1.0e-10 * (1.0 + minimum.abs());
    let best = converged_candidates
        .into_iter()
        .filter(|candidate| {
            candidate.comparison_negative_loglik <= minimum + objective_tie_tolerance
        })
        .min_by(|left, right| {
            left.state
                .free_indices()
                .len()
                .cmp(&right.state.free_indices().len())
                .then_with(|| {
                    left.comparison_negative_loglik
                        .total_cmp(&right.comparison_negative_loglik)
                })
        })
        .ok_or("BIVARIATE_NO_START_CONVERGED")?;
    if failed_candidates.iter().any(|candidate| {
        candidate.comparison_negative_loglik
            < best.comparison_negative_loglik - objective_tie_tolerance
    }) {
        return Err("BIVARIATE_BETTER_UNRESOLVED_OPTIMUM");
    }
    Ok(best)
}

const HERITABILITY_STATES: [HeritabilityState; 9] = [
    HeritabilityState([
        BivariateHeritabilityBoundary::Interior,
        BivariateHeritabilityBoundary::Interior,
    ]),
    HeritabilityState([
        BivariateHeritabilityBoundary::Lower,
        BivariateHeritabilityBoundary::Interior,
    ]),
    HeritabilityState([
        BivariateHeritabilityBoundary::Upper,
        BivariateHeritabilityBoundary::Interior,
    ]),
    HeritabilityState([
        BivariateHeritabilityBoundary::Interior,
        BivariateHeritabilityBoundary::Lower,
    ]),
    HeritabilityState([
        BivariateHeritabilityBoundary::Interior,
        BivariateHeritabilityBoundary::Upper,
    ]),
    HeritabilityState([
        BivariateHeritabilityBoundary::Lower,
        BivariateHeritabilityBoundary::Lower,
    ]),
    HeritabilityState([
        BivariateHeritabilityBoundary::Lower,
        BivariateHeritabilityBoundary::Upper,
    ]),
    HeritabilityState([
        BivariateHeritabilityBoundary::Upper,
        BivariateHeritabilityBoundary::Lower,
    ]),
    HeritabilityState([
        BivariateHeritabilityBoundary::Upper,
        BivariateHeritabilityBoundary::Upper,
    ]),
];

impl HeritabilityState {
    fn genetic_correlation_present(self) -> bool {
        self.0
            .iter()
            .all(|state| *state != BivariateHeritabilityBoundary::Lower)
    }

    fn residual_correlation_present(self) -> bool {
        self.0
            .iter()
            .all(|state| *state != BivariateHeritabilityBoundary::Upper)
    }

    fn free_indices(self) -> Vec<usize> {
        let mut indices = vec![0, 1];
        for trait_index in 0..2 {
            if self.0[trait_index] == BivariateHeritabilityBoundary::Interior {
                indices.push(2 + trait_index);
            }
        }
        if self.genetic_correlation_present() {
            indices.push(4);
        }
        if self.residual_correlation_present() {
            indices.push(5);
        }
        indices
    }

    fn expand(self, free: &[f64]) -> [f64; PARAMETERS] {
        let mut theta = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        for trait_index in 0..2 {
            theta[2 + trait_index] = match self.0[trait_index] {
                BivariateHeritabilityBoundary::Lower => 0.0,
                BivariateHeritabilityBoundary::Interior => 0.5,
                BivariateHeritabilityBoundary::Upper => 1.0,
            };
        }
        for (value, index) in free.iter().zip(self.free_indices()) {
            theta[index] = *value;
        }
        theta
    }

    fn pack(self, theta: &[f64; PARAMETERS]) -> Vec<f64> {
        self.free_indices()
            .into_iter()
            .map(|index| theta[index])
            .collect()
    }

    fn pack_gradient(self, gradient: &[f64; PARAMETERS]) -> Vec<f64> {
        self.pack(gradient)
    }

    fn bounds(self) -> Result<Bounds, &'static str> {
        let indices = self.free_indices();
        Bounds::new(
            indices.iter().map(|&index| LOWER[index]).collect(),
            indices.iter().map(|&index| UPPER[index]).collect(),
        )
        .map_err(|_| "BIVARIATE_STATE_OPTIMISATION_UNRESOLVED")
    }
}

#[cfg(test)]
fn project_full(theta: &mut [f64; PARAMETERS]) {
    for index in 0..PARAMETERS {
        theta[index] = theta[index].clamp(LOWER[index], UPPER[index]);
    }
}

/// The convergence test of decision 14: the largest gradient component, scaled
/// by the parameter and by the objective so that it is a relative quantity, and
/// ignoring directions pressed against a bound the gradient points into.
///
/// This is evaluated in the reversible, trait-standardised coordinates used by
/// the optimiser. Evaluating it after mapping the variances back to response
/// units would make the convergence decision depend on whether a trait happened
/// to be measured in units, thousands, or millionths.
fn projected_gradient_norm(
    theta: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    objective: f64,
) -> f64 {
    let scale = objective.abs().max(1.0);
    let mut worst: f64 = 0.0;
    for k in 0..theta.len() {
        let at_lower = theta[k] <= lower[k] + 1e-12 && gradient[k] > 0.0;
        let at_upper = theta[k] >= upper[k] - 1e-12 && gradient[k] < 0.0;
        if at_lower || at_upper {
            continue;
        }
        worst = worst.max(gradient[k].abs() * theta[k].abs().max(1.0) / scale);
    }
    worst
}

impl BivariateModel {
    /// Fit two traits.
    ///
    /// Bound-constrained L-BFGS-B with analytic gradients, three deterministic
    /// starts and an independently recomputed projected KKT check. The retained
    /// curvature memory is six, equal to the parameter count (decision 14 as
    /// amended after measuring the available searches).
    ///
    /// # Errors
    ///
    /// Returns a stable code where the response is invalid, no exact state has a
    /// KKT-qualified optimum, or a failed search found a materially better point
    /// than every qualified candidate.
    pub fn fit(&self, y: &DVector<f64>, reml: bool) -> Result<BivariateFit, &'static str> {
        if y.len() != self.rows {
            return Err("BIVARIATE_Y_LENGTH_MISMATCH");
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err("BIVARIATE_Y_NOT_FINITE");
        }

        // One good warm start and two insurance starts in each exact
        // heritability state (decision 14). The lower/upper states remove the
        // correlation that ceases to exist when a component variance vanishes.
        let standardised = self.standardised_problem(y)?;
        let starts = [
            [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
            [1.0, 1.0, 0.3, 0.3, 0.5, 0.5],
            [1.0, 1.0, 0.7, 0.7, -0.3, 0.3],
        ];

        let mut converged_candidates = Vec::new();
        let mut failed_candidates = Vec::new();
        for state in HERITABILITY_STATES {
            for start in starts {
                if let Some(outcome) = Self::minimise(&standardised, state, start, reml)? {
                    if outcome.converged {
                        converged_candidates.push(outcome);
                    } else {
                        failed_candidates.push(outcome);
                    }
                }
            }
        }
        // When an endpoint and the epsilon-open interior are numerically tied,
        // select the lower-dimensional exact state. It says which correlation
        // is absent rather than reporting a floating-point near-boundary.
        let best = select_best_candidate(converged_candidates, &failed_candidates)?;
        let theta = best.theta;

        let (s1, s2, h1, h2, rg, re) = (theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]);
        // The phenotypic correlation is derived, never estimated: the genetic
        // and residual covariances add, and the total variances are their sums.
        let cov_p = rg * (h1 * s1 * h2 * s2).max(0.0).sqrt()
            + re * ((1.0 - h1) * s1 * (1.0 - h2) * s2).max(0.0).sqrt();
        let rho_p = cov_p / (s1 * s2).sqrt();

        Ok(BivariateFit {
            h2: [h1, h2],
            h2_boundary: best.state.0,
            total_variance: [s1, s2],
            rho_g: best.state.genetic_correlation_present().then_some(rg),
            rho_e: best.state.residual_correlation_present().then_some(re),
            rho_p,
            loglik: -best.negative_loglik,
            converged: best.converged,
            estimator: if reml { "reml" } else { "ml" },
            scaled_gradient: best.scaled_gradient,
        })
    }

    fn trait_scales(&self, y: &DVector<f64>) -> [f64; 2] {
        let mut sums = [0.0; 2];
        let mut squares = [0.0; 2];
        let mut counts = [0usize; 2];
        for positions in &self.row_positions {
            for t in 0..2 {
                if let Some(index) = positions[t] {
                    sums[t] += y[index];
                    squares[t] += y[index] * y[index];
                    counts[t] += 1;
                }
            }
        }
        let mut scales = [1.0; 2];
        for t in 0..2 {
            if counts[t] > 1 {
                let n = counts[t] as f64;
                let variance = (squares[t] - sums[t] * sums[t] / n) / (n - 1.0);
                if variance > 0.0 {
                    scales[t] = variance;
                }
            }
        }
        scales
    }

    fn standardised_problem(&self, y: &DVector<f64>) -> Result<StandardisedProblem, &'static str> {
        let variance_scale = self.trait_scales(y);
        let standard_deviation = [variance_scale[0].sqrt(), variance_scale[1].sqrt()];
        let mut standardised_y = y.clone();
        let mut standardised_design = self.design.clone();
        for positions in &self.row_positions {
            for trait_index in 0..2 {
                if let Some(row) = positions[trait_index] {
                    standardised_y[row] /= standard_deviation[trait_index];
                    standardised_design
                        .row_mut(row)
                        .scale_mut(1.0 / standard_deviation[trait_index]);
                }
            }
        }
        let standardised_gram = standardised_design.transpose() * &standardised_design;
        let eigenvalues = SymmetricEigen::new(standardised_gram).eigenvalues;
        if eigenvalues.iter().any(|value| *value <= 0.0) {
            return Err("BIVARIATE_STANDARDISED_DESIGN_RANK_DEFICIENT");
        }
        let standardised_logdet_xtx = eigenvalues.iter().map(|value| value.ln()).sum();
        let log_scale_jacobian = self
            .row_positions
            .iter()
            .flat_map(|positions| {
                positions
                    .iter()
                    .enumerate()
                    .filter_map(|(trait_index, position)| {
                        position.map(|_| standard_deviation[trait_index].ln())
                    })
            })
            .sum::<f64>();
        Ok(StandardisedProblem {
            model: Self {
                relationship: self.relationship.clone(),
                blocks: self.blocks.clone(),
                design: standardised_design,
                row_positions: self.row_positions.clone(),
                logdet_xtx: standardised_logdet_xtx,
                subjects: self.subjects,
                rows: self.rows,
            },
            y: standardised_y,
            variance_scale,
            objective_shift: [
                log_scale_jacobian,
                log_scale_jacobian + 0.5 * (standardised_logdet_xtx - self.logdet_xtx),
            ],
        })
    }

    /// Returns the objective, the point, the scaled gradient there, and whether
    /// it met the tolerance.
    ///
    /// Bound-constrained BFGS from `rcompat-lbfgsb`, a safe-Rust implementation
    /// of the algorithm used by R's `optim(method = "L-BFGS-B")`.
    ///
    /// **The dependency is kept on measurement, not on preference.** The first
    /// pure-Rust L-BFGS-B implementation tried here abandoned all three starts
    /// with line-search failures and free-coordinate scaled scores between
    /// 0.07 and 0.11. This implementation reaches the independently recomputed
    /// `1e-7` KKT criterion for both ML and REML on the same unbalanced problem,
    /// including an exact correlation-bound solution.
    ///
    /// Decision 14 preferred extending a hand-written BFGS to taking a
    /// dependency, but that reasoning rested on Astrarium already having one and
    /// it did not come across in the fresh start. A hand-written attempt is in
    /// the history: it descended and then stalled two orders of magnitude short
    /// of the tolerance, because a good bound-constrained search needs a line
    /// search that checks the slope has flattened and not merely that the value
    /// fell, and an active set that optimises over the free parameters rather
    /// than projecting a step computed as though the bounds were not there.
    /// Both are genuinely hard and both have been done properly already.
    ///
    /// The memory equals the number of free parameters in the state, at most
    /// six. This remains an L-BFGS-B representation, but retains enough recent
    /// curvature pairs to span that state's parameter space. The solver's status is not
    /// trusted on its own: the analytic score is evaluated again at the
    /// returned point and Asterism's scaled projected KKT criterion is what sets
    /// `converged`.
    fn minimise(
        standardised: &StandardisedProblem,
        state: HeritabilityState,
        start: [f64; PARAMETERS],
        reml: bool,
    ) -> Result<Option<OptimisationCandidate>, &'static str> {
        const TOLERANCE: f64 = 1e-7;
        let bounds = state.bounds()?;
        let mut x = state.pack(&start);
        for (index, value) in x.iter_mut().enumerate() {
            *value = value.clamp(bounds.lower[index], bounds.upper[index]);
        }
        let initial_theta = state.expand(&x);
        let Some(initial_evaluation) = standardised.model.evaluate_for_state(
            &initial_theta,
            &standardised.y,
            reml,
            true,
            Some(state),
        ) else {
            // This exact state/start combination is structurally infeasible.
            // It is the only condition that may be omitted silently.
            return Ok(None);
        };
        let initial_objective = initial_evaluation.negative_loglik;

        let invalid_value = |theta: &[f64; PARAMETERS]| {
            INVALID_OBJECTIVE_SCALE
                * (1.0
                    + (theta[2] - 0.5).powi(2)
                    + (theta[3] - 0.5).powi(2)
                    + theta[4].powi(2)
                    + theta[5].powi(2))
        };
        let invalid_gradient = |theta: &[f64; PARAMETERS]| {
            state.pack_gradient(&[
                0.0,
                0.0,
                2.0 * INVALID_OBJECTIVE_SCALE * (theta[2] - 0.5),
                2.0 * INVALID_OBJECTIVE_SCALE * (theta[3] - 0.5),
                2.0 * INVALID_OBJECTIVE_SCALE * theta[4],
                2.0 * INVALID_OBJECTIVE_SCALE * theta[5],
            ])
        };
        let evaluate_value = |candidate: &[f64]| -> f64 {
            let theta = state.expand(candidate);
            match standardised.model.evaluate_for_state(
                &theta,
                &standardised.y,
                reml,
                true,
                Some(state),
            ) {
                Some(e) => e.negative_loglik,
                None => invalid_value(&theta),
            }
        };
        let evaluate_gradient = |candidate: &[f64]| -> Vec<f64> {
            let theta = state.expand(candidate);
            standardised
                .model
                .evaluate_for_state(&theta, &standardised.y, reml, true, Some(state))
                .map_or_else(
                    || invalid_gradient(&theta),
                    |e| state.pack_gradient(&e.gradient),
                )
        };

        let dimension = x.len();
        let mut control = OptimControl::default_for_dimension(dimension);
        control.maxit = 500;
        control.fnscale = initial_objective.abs().max(1.0);
        control.parscale = vec![1.0; dimension];
        control.factr = 0.0;
        // Ask the dependency for a tighter raw projected score than Asterism's
        // scaled criterion. Their norms are not identical, so using the public
        // threshold here left a few deterministic stress cases just above it.
        control.pgtol = TOLERANCE * 0.1;
        control.lmm = dimension;
        let Ok(solution) = optim_lbfgsb_with_gradient(
            x,
            bounds.clone(),
            evaluate_value,
            evaluate_gradient,
            control,
        ) else {
            return Err("BIVARIATE_STATE_OPTIMISATION_UNRESOLVED");
        };

        let standardised_theta = state.expand(&solution.par);
        let standardised_at = standardised
            .model
            .evaluate_for_state(
                &standardised_theta,
                &standardised.y,
                reml,
                true,
                Some(state),
            )
            .ok_or("BIVARIATE_STATE_OPTIMISATION_UNRESOLVED")?;
        let free_gradient = state.pack_gradient(&standardised_at.gradient);
        let norm = projected_gradient_norm(
            &solution.par,
            &free_gradient,
            &bounds.lower,
            &bounds.upper,
            standardised_at.negative_loglik,
        );
        let at_interior_numerical_edge = (0..2).any(|trait_index| {
            state.0[trait_index] == BivariateHeritabilityBoundary::Interior
                && (standardised_theta[2 + trait_index] <= INTERIOR_EPSILON + 1.0e-12
                    || standardised_theta[2 + trait_index] >= 1.0 - INTERIOR_EPSILON - 1.0e-12)
        });

        let mut theta = standardised_theta;
        theta[0] *= standardised.variance_scale[0];
        theta[1] *= standardised.variance_scale[1];
        Ok(Some(OptimisationCandidate {
            // Re-evaluating a large-scale covariance can lose positive
            // definiteness to round-off. This is the exact Jacobian/invariant
            // REML transformation back to the original response scale.
            negative_loglik: standardised_at.negative_loglik
                + standardised.objective_shift[usize::from(reml)],
            comparison_negative_loglik: standardised_at.negative_loglik,
            theta,
            state,
            scaled_gradient: norm,
            converged: norm < TOLERANCE && !at_interior_numerical_edge,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BivariateHeritabilityBoundary::{Interior, Lower, Upper},
        BivariateModel, HERITABILITY_STATES, HeritabilityState, OptimisationCandidate, PARAMETERS,
        select_best_candidate,
    };
    use nalgebra::{DMatrix, DVector};

    /// Two sibling pairs' worth of relationship, small enough to reason about.
    fn small() -> (DMatrix<f64>, Vec<[bool; 2]>, DMatrix<f64>, DVector<f64>) {
        let pairs = 30;
        let n = 2 * pairs;
        let mut k = DMatrix::<f64>::identity(n, n);
        for pair in 0..pairs {
            k[(2 * pair, 2 * pair + 1)] = 0.5;
            k[(2 * pair + 1, 2 * pair)] = 0.5;
        }
        // Deliberately unbalanced: every seventh person lacks the second trait.
        let observed: Vec<[bool; 2]> = (0..n).map(|i| [true, i % 7 != 0]).collect();
        let rows: usize = observed
            .iter()
            .map(|o| usize::from(o[0]) + usize::from(o[1]))
            .sum();

        let mut design = DMatrix::<f64>::zeros(rows, 2);
        let mut y = DVector::<f64>::zeros(rows);
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        // Give the siblings a shared genetic component, correlated across the
        // two traits. Without one, h² goes to zero and the genetic correlation
        // multiplies the square root of zero — it stops affecting the likelihood
        // at all and no optimiser can converge on it. That is a real state
        // (decision 5: a correlation is absent, not zero, where the variance
        // is), and it is tested separately rather than here.
        let mut index = 0;
        let mut shared = vec![(0.0, 0.0); pairs];
        for pair in &mut shared {
            let g = next() + next() + next();
            *pair = (g, 0.6 * g + 0.4 * (next() + next() + next()));
        }
        for (person, mask) in observed.iter().enumerate() {
            let (ga, gb) = shared[person / 2];
            for t in 0..2 {
                if mask[t] {
                    design[(index, t)] = 1.0;
                    let genetic = if t == 0 { ga } else { gb };
                    y[index] = 1.4 * genetic + 0.6 * (next() + next() + next());
                    index += 1;
                }
            }
        }
        (k, observed, design, y)
    }

    /// The gradient is the part most easily got wrong and the part everything
    /// downstream rests on, so it is checked against a central difference of the
    /// objective. Analytic first derivatives, differenced only to test them.
    /// One point is not a test of a gradient. This walks a grid, including the
    /// places the search actually visits: correlations near zero, heritabilities
    /// near their bounds, unequal variances.
    #[test]
    fn the_gradient_matches_everywhere_the_search_goes() {
        let (k, observed, design, y) = small();
        let model = BivariateModel::build(&k, &observed, &design).expect("valid");
        let points: Vec<[f64; PARAMETERS]> = vec![
            [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
            [1.0, 1.0, 0.5, 0.5, 0.001, -0.001],
            [0.6, 1.9, 0.2, 0.8, 0.6, -0.4],
            [0.6, 1.9, 0.05, 0.95, -0.9, 0.9],
            [2.0, 0.4, 0.5, 0.5, 0.99, 0.99],
            [1.0, 1.0, 0.999, 0.001, 0.2, 0.2],
        ];
        let mut worst = 0.0f64;
        let mut where_worst = (0usize, 0usize, false);
        for (index, theta) in points.iter().enumerate() {
            for reml in [false, true] {
                let Some(at) = model.evaluate(theta, &y, reml, true) else {
                    continue;
                };
                for k_index in 0..PARAMETERS {
                    let step = 1e-6 * (1.0 + theta[k_index].abs());
                    let mut up = *theta;
                    let mut down = *theta;
                    up[k_index] += step;
                    down[k_index] -= step;
                    let (Some(a), Some(b)) = (
                        model.evaluate(&up, &y, reml, false),
                        model.evaluate(&down, &y, reml, false),
                    ) else {
                        continue;
                    };
                    let numeric = (a.negative_loglik - b.negative_loglik) / (2.0 * step);
                    let relative = (at.gradient[k_index] - numeric).abs() / (1.0 + numeric.abs());
                    if relative > worst {
                        worst = relative;
                        where_worst = (index, k_index, reml);
                    }
                }
            }
        }
        assert!(
            worst < 1e-5,
            "worst relative gradient error {worst:.3e} at point {} parameter {} ({})",
            where_worst.0,
            where_worst.1,
            if where_worst.2 { "reml" } else { "ml" }
        );
    }

    #[test]
    fn the_analytic_gradient_matches_a_central_difference() {
        let (k, observed, design, y) = small();
        let model = BivariateModel::build(&k, &observed, &design).expect("valid");
        let theta = [1.3, 0.8, 0.45, 0.6, 0.35, -0.2];

        for reml in [false, true] {
            let at = model.evaluate(&theta, &y, reml, true).expect("finite");
            for k_index in 0..PARAMETERS {
                let step = 1e-6 * (1.0 + theta[k_index].abs());
                let mut up = theta;
                let mut down = theta;
                up[k_index] += step;
                down[k_index] -= step;
                let a = model.evaluate(&up, &y, reml, false).expect("finite");
                let b = model.evaluate(&down, &y, reml, false).expect("finite");
                let numeric = (a.negative_loglik - b.negative_loglik) / (2.0 * step);
                let analytic = at.gradient[k_index];
                let scale = 1.0 + numeric.abs();
                assert!(
                    (analytic - numeric).abs() / scale < 1e-5,
                    "{} parameter {k_index}: analytic {analytic}, differenced {numeric}",
                    if reml { "reml" } else { "ml" }
                );
            }
        }
    }

    #[test]
    fn standardisation_preserves_the_ml_and_reml_objectives_exactly() {
        let (relationship, observed, design, y) = small();
        let model = BivariateModel::build(&relationship, &observed, &design).expect("valid");
        let standardised = model.standardised_problem(&y).expect("standardises");
        let standardised_theta = [1.2, 0.8, 0.4, 0.6, 0.25, -0.2];
        let mut original_theta = standardised_theta;
        original_theta[0] *= standardised.variance_scale[0];
        original_theta[1] *= standardised.variance_scale[1];

        for reml in [false, true] {
            let direct = model
                .evaluate(&original_theta, &y, reml, false)
                .expect("original objective")
                .negative_loglik;
            let transformed = standardised
                .model
                .evaluate(&standardised_theta, &standardised.y, reml, false)
                .expect("standardised objective")
                .negative_loglik
                + standardised.objective_shift[usize::from(reml)];
            assert!((direct - transformed).abs() < 1.0e-10);
        }
    }

    /// Prints the path. Not a check -- run with `--nocapture` when the
    /// optimiser misbehaves and delete nothing, it costs a millisecond.
    #[test]
    fn trace_the_optimiser() {
        let (k, observed, design, y) = small();
        let model = BivariateModel::build(&k, &observed, &design).expect("valid");
        let mut theta = [1.0, 1.0, 0.5, 0.5, 0.0, 0.0];
        super::project_full(&mut theta);
        for step in 0..12 {
            let e = model.evaluate(&theta, &y, true, true).expect("finite");
            let norm = super::projected_gradient_norm(
                &theta,
                &e.gradient,
                &super::LOWER,
                &super::UPPER,
                e.negative_loglik,
            );
            println!(
                "step {step:>2}  f={:>12.6}  |g|={:>10.3e}  theta={:?}",
                e.negative_loglik,
                norm,
                theta
                    .iter()
                    .map(|v| (v * 1e4).round() / 1e4)
                    .collect::<Vec<_>>()
            );
            println!(
                "            grad={:?}",
                e.gradient
                    .iter()
                    .map(|v| (v * 1e3).round() / 1e3)
                    .collect::<Vec<_>>()
            );
            // one plain gradient step, scaled, just to see the landscape
            for kk in 0..PARAMETERS {
                theta[kk] -= 1e-4 * e.gradient[kk];
            }
            super::project_full(&mut theta);
        }
    }

    #[test]
    fn the_unbalanced_fit_meets_the_public_convergence_criterion() {
        let (k, observed, design, y) = small();
        let model = BivariateModel::build(&k, &observed, &design).expect("valid");
        for reml in [false, true] {
            let fit = model.fit(&y, reml).expect("fits");
            assert!(
                fit.converged,
                "{} fit did not converge; scaled gradient was {}",
                fit.estimator, fit.scaled_gradient
            );
            assert!(fit.scaled_gradient < 1e-7);
            for h in fit.h2 {
                assert!((0.0..=1.0).contains(&h));
            }
            assert!(fit.rho_g.is_none_or(|rho| (-1.0..=1.0).contains(&rho)));
            assert!(fit.rho_e.is_none_or(|rho| (-1.0..=1.0).contains(&rho)));
            assert!((-1.0..=1.0).contains(&fit.rho_p));
        }
    }

    /// With no genetic variance the genetic correlation multiplies the square
    /// root of zero, so it has no effect on the likelihood and is not estimable.
    /// The fit must still return, and must not claim a tight gradient.
    #[test]
    fn a_correlation_with_no_variance_behind_it_is_not_estimable() {
        let pairs = 30;
        let n = 2 * pairs;
        let k = DMatrix::<f64>::identity(n, n); // nobody related to anybody
        let observed: Vec<[bool; 2]> = (0..n).map(|_| [true, true]).collect();
        let mut design = DMatrix::<f64>::zeros(2 * n, 2);
        let mut y = DVector::<f64>::zeros(2 * n);
        let mut seed = 99u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for i in 0..2 * n {
            design[(i, i % 2)] = 1.0;
            y[i] = next() + next() + next();
        }
        // With nobody related to anybody, `Σ_A ⊗ I + Σ_E ⊗ I` is `(Σ_A + Σ_E) ⊗ I`
        // and only the sum is determined: neither h² nor the genetic correlation
        // exists as an identified estimand. A flat ridge can have zero score, so
        // the prepared-model validation must refuse it before optimisation.
        assert_eq!(
            BivariateModel::build(&k, &observed, &design).err().unwrap(),
            "BIVARIATE_COVARIANCE_NOT_IDENTIFIABLE"
        );
    }

    #[test]
    fn unbalanced_is_the_ordinary_case() {
        let (k, observed, design, _y) = small();
        let model = BivariateModel::build(&k, &observed, &design).expect("valid");
        // 60 people, every seventh missing the second trait, so not 120 rows.
        assert_eq!(model.subjects(), 60);
        assert_eq!(model.observations(), 120 - 9);
    }

    #[test]
    fn genetic_and_residual_correlation_bounds_are_exact() {
        assert_eq!(super::LOWER[4], -1.0);
        assert_eq!(super::LOWER[5], -1.0);
        assert_eq!(super::UPPER[4], 1.0);
        assert_eq!(super::UPPER[5], 1.0);
    }

    #[test]
    fn all_nine_heritability_states_have_the_right_free_coordinates() {
        let states = HERITABILITY_STATES
            .iter()
            .map(|state| (state.0, state.free_indices()))
            .collect::<Vec<_>>();
        assert_eq!(states[0], ([Interior, Interior], vec![0, 1, 2, 3, 4, 5]));
        assert_eq!(states[1], ([Lower, Interior], vec![0, 1, 3, 5]));
        assert_eq!(states[2], ([Upper, Interior], vec![0, 1, 3, 4]));
        assert_eq!(states[3], ([Interior, Lower], vec![0, 1, 2, 5]));
        assert_eq!(states[4], ([Interior, Upper], vec![0, 1, 2, 4]));
        assert_eq!(states[5], ([Lower, Lower], vec![0, 1, 5]));
        assert_eq!(states[6], ([Lower, Upper], vec![0, 1]));
        assert_eq!(states[7], ([Upper, Lower], vec![0, 1]));
        assert_eq!(states[8], ([Upper, Upper], vec![0, 1, 4]));
    }

    #[test]
    fn every_exact_heritability_state_has_the_right_free_score() {
        let (relationship, observed, design, y) = small();
        let model = BivariateModel::build(&relationship, &observed, &design).expect("valid");
        let base = [1.2, 0.8, 0.4, 0.6, 0.25, -0.2];
        for state in HERITABILITY_STATES {
            let free = state.pack(&base);
            let theta = state.expand(&free);
            for reml in [false, true] {
                let at = model
                    .evaluate_for_state(&theta, &y, reml, true, Some(state))
                    .expect("finite state");
                let analytic = state.pack_gradient(&at.gradient);
                for free_index in 0..free.len() {
                    let step = 1.0e-6 * (1.0 + free[free_index].abs());
                    let mut up = free.clone();
                    let mut down = free.clone();
                    up[free_index] += step;
                    down[free_index] -= step;
                    let upper = model
                        .evaluate_for_state(&state.expand(&up), &y, reml, false, Some(state))
                        .expect("upper finite")
                        .negative_loglik;
                    let lower = model
                        .evaluate_for_state(&state.expand(&down), &y, reml, false, Some(state))
                        .expect("lower finite")
                        .negative_loglik;
                    let numeric = (upper - lower) / (2.0 * step);
                    assert!(
                        (analytic[free_index] - numeric).abs() / (1.0 + numeric.abs()) < 1.0e-5,
                        "state {:?}, {}, free coordinate {free_index}: analytic {}, numeric {}",
                        state.0,
                        if reml { "REML" } else { "ML" },
                        analytic[free_index],
                        numeric
                    );
                }
            }
        }
    }

    #[test]
    fn candidate_selection_prefers_exact_ties_and_refuses_a_better_failed_fit() {
        let candidate =
            |negative_loglik, state: HeritabilityState, converged| OptimisationCandidate {
                // Mimic an arbitrary response-unit Jacobian. State selection
                // must use the standardised comparison value, not this shift.
                negative_loglik: negative_loglik + 1.0e12,
                comparison_negative_loglik: negative_loglik,
                theta: state.expand(&state.pack(&[1.0, 1.0, 0.5, 0.5, 0.0, 0.0])),
                state,
                scaled_gradient: if converged { 1.0e-9 } else { 1.0e-4 },
                converged,
            };
        let interior = HERITABILITY_STATES[0];
        let upper_upper = HERITABILITY_STATES[8];
        let best = select_best_candidate(
            vec![
                candidate(10.0, interior, true),
                candidate(10.0 + 1.0e-12, upper_upper, true),
            ],
            &[],
        )
        .expect("selects exact tie");
        assert_eq!(best.state.0, [Upper, Upper]);

        let best = select_best_candidate(
            vec![
                candidate(10.0, interior, true),
                candidate(10.01, upper_upper, true),
            ],
            &[],
        )
        .expect("selects the genuinely better state");
        assert_eq!(best.state.0, [Interior, Interior]);

        assert_eq!(
            select_best_candidate(
                vec![candidate(10.0, interior, true)],
                &[candidate(9.0, upper_upper, false)],
            )
            .err(),
            Some("BIVARIATE_BETTER_UNRESOLVED_OPTIMUM")
        );
    }
}

#[cfg(feature = "python")]
mod python {
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::{BivariateModel, PARAMETERS};
    use nalgebra::{DMatrix, DVector};

    /// Fit two traits, for comparison against the independent implementation.
    #[pyfunction]
    #[pyo3(signature = (relationship, observed, design, y, reml=true))]
    pub fn bivariate_fit(
        relationship: PyReadonlyArray2<'_, f64>,
        observed: Vec<[bool; 2]>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        reml: bool,
    ) -> PyResult<(Vec<Option<f64>>, f64, f64, bool)> {
        let k = relationship.as_array();
        let k = DMatrix::from_fn(k.shape()[0], k.shape()[1], |i, j| k[(i, j)]);
        let x = design.as_array();
        let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
        let y = y.as_array();
        let y = DVector::from_iterator(y.len(), y.iter().copied());
        let model = BivariateModel::build(&k, &observed, &x).map_err(PyValueError::new_err)?;
        let fit = model.fit(&y, reml).map_err(PyValueError::new_err)?;
        Ok((
            vec![
                Some(fit.total_variance[0]),
                Some(fit.total_variance[1]),
                Some(fit.h2[0]),
                Some(fit.h2[1]),
                fit.rho_g,
                fit.rho_e,
            ],
            fit.loglik,
            fit.scaled_gradient,
            fit.converged,
        ))
    }

    /// Evaluate the two-trait objective and gradient at one point.
    ///
    /// Exposed so that `checks/bivariate_reference.py` can be compared against
    /// this at the same parameters on the same data — the gradient test only
    /// shows this objective agrees with its own derivative, which would pass
    /// just as well if the whole likelihood were wrong.
    ///
    /// `y` and `design` carry only the observed rows, in the order they appear
    /// reading person by person and trait within person.
    #[pyfunction]
    #[pyo3(signature = (relationship, observed, design, y, theta, reml=true))]
    pub fn bivariate_objective(
        relationship: PyReadonlyArray2<'_, f64>,
        observed: Vec<[bool; 2]>,
        design: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        theta: Vec<f64>,
        reml: bool,
    ) -> PyResult<(f64, Vec<f64>)> {
        if theta.len() != PARAMETERS {
            return Err(PyValueError::new_err("BIVARIATE_THETA_LENGTH"));
        }
        let k = relationship.as_array();
        let k = DMatrix::from_fn(k.shape()[0], k.shape()[1], |i, j| k[(i, j)]);
        let x = design.as_array();
        let x = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
        let y = y.as_array();
        let y = DVector::from_iterator(y.len(), y.iter().copied());

        let model = BivariateModel::build(&k, &observed, &x).map_err(PyValueError::new_err)?;
        let mut point = [0.0; PARAMETERS];
        point.copy_from_slice(&theta);
        model
            .objective_at(&point, &y, reml)
            .map(|(value, gradient)| (value, gradient.to_vec()))
            .ok_or_else(|| PyValueError::new_err("BIVARIATE_NOT_POSITIVE_DEFINITE"))
    }
}

#[cfg(feature = "python")]
pub use python::{bivariate_fit, bivariate_objective};
