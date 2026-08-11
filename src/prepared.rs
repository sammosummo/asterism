//! The prepared model: a validated, eigendecomposed pair of a fixed-effect
//! design and a relationship matrix, and the one-trait fit that runs against it.
//!
//! Building it validates — exact symmetry, positive semi-definiteness at a
//! -1e-9 eigenvalue floor, finiteness, shapes, design rank — and decomposes the
//! relationship matrix block by block, once. Every fit then runs against the
//! stored spectral form. There is no way to switch validation off: holding a
//! `PreparedModel` is itself the proof that it happened (`docs/adr/0002`).
//!
//! The objective is fixed by the interface. This is always the ordinary
//! Gaussian model, and the response is never inspected to choose a likelihood;
//! a response that looks binary earns a warning on the record and nothing more.
//!
//! Ported from Astrarium's `prepared.rs`, which matched native SOLAR to 8e-09
//! on the likelihood and passed a 96,000-fit coverage check. The algebra is
//! unchanged, because that is the part with numbers behind it. What is new here
//! is the likelihood ratio test.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use statrs::function::erf::erfc;

#[cfg(feature = "python")]
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use sha2::{Digest, Sha256};

use crate::blocks::family_blocks;

/// Eigenvalues of a positive semi-definite relationship matrix may dip a few
/// units in the last place below zero; anything under this floor is a genuine
/// violation rather than rounding.
const EIGENVALUE_FLOOR: f64 = -1e-9;
/// The 0.95 quantile of chi-square with one degree of freedom.
const CHI2_ONE_DF_95: f64 = 3.841_458_820_694_124;
/// The 0.95 quantile of the 50:50 mixture of chi-square with zero and one
/// degrees of freedom — the Self–Liang rule deciding whether a boundary *point*
/// belongs to the interval (`docs/adr/0004`).
const MIXTURE_CRIT: f64 = 2.705_543_454_095_404;
/// Snap width at the upper bound only. With a singular relationship matrix and
/// a response duplicated within a zero-eigenvalue direction the likelihood has
/// a pole as h² approaches one, and the search chases it. Snapping within 1e-7
/// lands such fits on exactly 1.0, where a singular covariance is refused and
/// the fit reports honestly as not converged. There is no pole at h² = 0, so
/// the lower bound keeps its exact snap.
const UPPER_SNAP_TOLERANCE: f64 = 1e-7;
/// Grid resolution for the coarse pass; both endpoints lie on the grid.
const GRID_POINTS: usize = 33;
/// A refined optimum this close to a bound is the bound.
const BOUNDARY_TOLERANCE: f64 = 1e-12;
/// Bracket tolerance for the bounded refinement of the profiled likelihood.
const XATOL: f64 = 1e-9;
/// Step for the finite-difference observed information.
const INFORMATION_STEP: f64 = 1e-4;

fn log_two_pi() -> f64 {
    (2.0 * std::f64::consts::PI).ln()
}

#[cfg(feature = "python")]
fn code(text: &str) -> PyErr {
    PyValueError::new_err(text.to_owned())
}

/// The upper tail of chi-square with one degree of freedom, computed through
/// the complementary error function rather than one minus a distribution
/// function, which loses its digits exactly where a p-value needs them.
fn chi2_one_df_upper_tail(statistic: f64) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    erfc((statistic / 2.0).sqrt())
}

/// The diagonal of the covariance over the residual variance in the rotated
/// basis, exact at both bounds.
fn variance_kernel(lam: &DVector<f64>, h2: f64) -> DVector<f64> {
    if h2 == 0.0 {
        return DVector::from_element(lam.len(), 1.0);
    }
    if h2 == 1.0 {
        return lam.clone();
    }
    lam.map(|value| 1.0 + h2 * (value - 1.0))
}

struct Profiled {
    loglik: f64,
    beta: DVector<f64>,
    sigma2: f64,
}

/// Golden-section minimisation on a closed bracket, deterministic.
fn golden_minimise(f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64, xatol: f64) -> f64 {
    const INVPHI: f64 = 0.618_033_988_749_894_8;
    let mut c = hi - INVPHI * (hi - lo);
    let mut d = lo + INVPHI * (hi - lo);
    let mut fc = f(c);
    let mut fd = f(d);
    while hi - lo > xatol {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - INVPHI * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + INVPHI * (hi - lo);
            fd = f(d);
        }
    }
    0.5 * (lo + hi)
}

/// Bisection root of `f` on a bracket carrying a sign change, deterministic.
fn bisect_root(f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    let mut flo = f(lo);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if hi - lo <= 1e-13 {
            return mid;
        }
        let fmid = f(mid);
        if (flo <= 0.0) == (fmid <= 0.0) {
            lo = mid;
            flo = fmid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// A validated, eigendecomposed design and relationship matrix; the only way in.
#[cfg_attr(feature = "python", pyclass(frozen))]
pub struct PreparedModel {
    lam: DVector<f64>,
    xt: DMatrix<f64>,
    rotations: Vec<(Vec<usize>, DMatrix<f64>)>,
    n: usize,
    p: usize,
    dfr: f64,
    logdet_xtx: f64,
    min_eigenvalue: f64,
    subject_order: Option<String>,
}

impl PreparedModel {
    fn rotate(&self, y: &DVector<f64>) -> DVector<f64> {
        let mut out = Vec::with_capacity(self.n);
        for (indices, vectors) in &self.rotations {
            let sub = DVector::from_iterator(indices.len(), indices.iter().map(|&i| y[i]));
            let rotated = vectors.transpose() * sub;
            out.extend(rotated.iter().copied());
        }
        DVector::from_vec(out)
    }

    fn profile(&self, yt: &DVector<f64>, h2: f64, reml: bool) -> Option<Profiled> {
        let d = variance_kernel(&self.lam, h2);
        if d.min() <= 0.0 || !d.iter().all(|v| v.is_finite()) {
            return None;
        }
        let n = self.n;
        let p = self.p;
        let mut xtwx = DMatrix::<f64>::zeros(p, p);
        let mut xtwy = DVector::<f64>::zeros(p);
        let mut logdet_v = 0.0;
        for i in 0..n {
            let w = 1.0 / d[i];
            logdet_v += d[i].ln();
            let yi = yt[i];
            for a in 0..p {
                let xa = self.xt[(i, a)];
                xtwy[a] += w * xa * yi;
                for b in 0..p {
                    xtwx[(a, b)] += w * xa * self.xt[(i, b)];
                }
            }
        }
        let chol = xtwx.cholesky()?;
        let beta = chol.solve(&xtwy);
        let mut quad = 0.0;
        let mut response_scale = 0.0;
        for i in 0..n {
            let mut fitted = 0.0;
            for a in 0..p {
                fitted += self.xt[(i, a)] * beta[a];
            }
            let r = yt[i] - fitted;
            quad += r * r / d[i];
            response_scale += yt[i] * yt[i] / d[i];
        }
        // A response lying in the column span of the design leaves the
        // quadratic form as rounding noise, which n·ln(sigma2) amplifies into a
        // meaningless log-likelihood. Refuse it against a scale-relative floor.
        // The negation is deliberate and must not be rewritten as `<=`: it also
        // catches a quadratic form that is not a number at all.
        #[allow(clippy::neg_cmp_op_on_partial_ord, reason = "also rejects NaN")]
        if !(quad > 1e-12 * response_scale) {
            return None;
        }
        if reml {
            let sigma2 = quad / self.dfr;
            let logdet_xwx = 2.0 * chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
            let loglik = -0.5
                * (self.dfr * log_two_pi() + logdet_v + logdet_xwx - self.logdet_xtx
                    + self.dfr * sigma2.ln()
                    + self.dfr);
            Some(Profiled { loglik, beta, sigma2 })
        } else {
            let nf = n as f64;
            let sigma2 = quad / nf;
            let loglik = -0.5 * (nf * log_two_pi() + logdet_v + nf * sigma2.ln() + nf);
            Some(Profiled { loglik, beta, sigma2 })
        }
    }

    fn loglik_at(&self, yt: &DVector<f64>, h2: f64, reml: bool) -> f64 {
        self.profile(yt, h2, reml).map_or(f64::NEG_INFINITY, |profiled| profiled.loglik)
    }

    /// Number of observations.
    #[must_use]
    pub fn observations(&self) -> usize {
        self.n
    }

    /// Number of fixed-effect columns.
    #[must_use]
    pub fn fixed_effects(&self) -> usize {
        self.p
    }

    /// The smallest eigenvalue seen across the family blocks. A relationship
    /// matrix that is singular rather than merely near it changes what the
    /// likelihood does at the upper bound, so it is worth being able to look.
    #[must_use]
    pub fn min_eigenvalue(&self) -> f64 {
        self.min_eigenvalue
    }

    /// The subject-order commitment, if one was given.
    #[must_use]
    pub fn subject_order(&self) -> Option<&str> {
        self.subject_order.as_deref()
    }

    /// Validate and decompose. Kept separate from the Python constructor so the
    /// crate is usable, and testable, without Python in the picture.
    ///
    /// # Errors
    ///
    /// Returns a stable code naming what was wrong: the relationship matrix not
    /// square, not the same size as the design, not symmetric, or not positive
    /// semi-definite; either input holding a value that is not finite; the
    /// design rank-deficient; or no residual degrees of freedom.
    #[allow(
        clippy::float_cmp,
        reason = "symmetry and the eigenvalue floor are exact tests by intent"
    )]
    pub fn build(
        x: &DMatrix<f64>,
        k: &DMatrix<f64>,
        subject_order: Option<String>,
    ) -> Result<Self, &'static str> {
        let (n, p) = (x.nrows(), x.ncols());
        if k.nrows() != k.ncols() {
            return Err("PREPARE_K_NOT_SQUARE");
        }
        if k.nrows() != n {
            return Err("PREPARE_SHAPE_MISMATCH");
        }
        if n == 0 || p == 0 || n <= p {
            return Err("PREPARE_NO_RESIDUAL_DEGREES_OF_FREEDOM");
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err("PREPARE_X_NOT_FINITE");
        }
        if k.iter().any(|value| !value.is_finite()) {
            return Err("PREPARE_K_NOT_FINITE");
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if k[(i, j)] != k[(j, i)] {
                    return Err("PREPARE_K_ASYMMETRIC");
                }
            }
        }

        let blocks = family_blocks(k);
        let mut lam = Vec::with_capacity(n);
        let mut xt_rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut rotations = Vec::with_capacity(blocks.len());
        let mut min_eigenvalue = f64::INFINITY;
        for block in blocks {
            let size = block.len();
            let sub = DMatrix::from_fn(size, size, |a, b| k[(block[a], block[b])]);
            let eigen = SymmetricEigen::new(sub);
            let x_block = DMatrix::from_fn(size, p, |a, column| x[(block[a], column)]);
            let xt_block = eigen.eigenvectors.transpose() * x_block;
            for local in 0..size {
                let mut value = eigen.eigenvalues[local];
                min_eigenvalue = min_eigenvalue.min(value);
                if value < EIGENVALUE_FLOOR {
                    return Err("PREPARE_K_NOT_PSD");
                }
                if value < 0.0 {
                    value = 0.0;
                }
                lam.push(value);
                xt_rows.push((0..p).map(|column| xt_block[(local, column)]).collect());
            }
            rotations.push((block, eigen.eigenvectors));
        }
        let lam = DVector::from_vec(lam);
        let xt = DMatrix::from_fn(n, p, |i, j| xt_rows[i][j]);

        // Rank check on the Gram spectrum. A Cholesky pivot test lets an exactly
        // duplicated column through on rounding noise, giving a pivot near
        // 1e-16 rather than zero. A relative eigenvalue floor of 1e-12
        // separates genuine deficiency, around 1e-16 of the scale, from a design
        // that is merely badly conditioned, around 1e-7.
        let gram = xt.transpose() * &xt;
        let gram_eigenvalues = SymmetricEigen::new(gram).eigenvalues;
        let max_eigenvalue = gram_eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        if gram_eigenvalues.iter().any(|&value| value <= 1e-12 * max_eigenvalue) {
            return Err("PREPARE_X_RANK_DEFICIENT");
        }
        let logdet_xtx = gram_eigenvalues.iter().map(|value| value.ln()).sum::<f64>();

        Ok(Self {
            lam,
            xt,
            rotations,
            n,
            p,
            dfr: (n - p) as f64,
            logdet_xtx,
            min_eigenvalue,
            subject_order,
        })
    }

    /// The whole of one fit, with no Python types anywhere in it.
    pub fn fit_one_trait(&self, y: &DVector<f64>, reml: bool) -> Fit {
        let mut warnings: Vec<String> = Vec::new();
        let mut sorted: Vec<f64> = y.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let distinct = 1 + sorted.windows(2).filter(|w| w[0] != w[1]).count();
        if distinct <= 2 {
            warnings.push(format!(
                "response has {distinct} distinct values; fitted as quantitative"
            ));
        }

        let yt = self.rotate(y);
        let objective = |h2: f64| -> f64 {
            let ll = self.loglik_at(&yt, h2, reml);
            if ll.is_finite() { -ll } else { f64::INFINITY }
        };

        let mut grid_best = (0usize, f64::INFINITY);
        let grid: Vec<f64> = (0..GRID_POINTS)
            .map(|i| i as f64 / (GRID_POINTS - 1) as f64)
            .collect();
        for (index, &h2) in grid.iter().enumerate() {
            let value = objective(h2);
            if value < grid_best.1 {
                grid_best = (index, value);
            }
        }
        let mut best_h2 = grid[grid_best.0];
        let mut best_negloglik = grid_best.1;
        let converged_grid = best_negloglik.is_finite();
        let lo = grid[grid_best.0.saturating_sub(1)];
        let hi = grid[(grid_best.0 + 1).min(GRID_POINTS - 1)];
        if converged_grid && hi > lo {
            let refined = golden_minimise(&objective, lo, hi, XATOL);
            let refined_value = objective(refined);
            if refined_value.is_finite() && refined_value < best_negloglik {
                best_h2 = refined;
                best_negloglik = refined_value;
            }
        }
        let _ = best_negloglik;
        if best_h2 <= BOUNDARY_TOLERANCE {
            best_h2 = 0.0;
        } else if best_h2 >= 1.0 - UPPER_SNAP_TOLERANCE {
            best_h2 = 1.0;
        }

        let final_fit = self.profile(&yt, best_h2, reml);
        let converged =
            converged_grid && final_fit.as_ref().is_some_and(|p| p.loglik.is_finite());
        let (loglik, beta, sigma2) = match final_fit {
            Some(profiled) => (profiled.loglik, profiled.beta, profiled.sigma2),
            None => (f64::NAN, DVector::from_element(self.p, f64::NAN), f64::NAN),
        };

        let boundary = if best_h2 == 0.0 {
            Boundary::Lower
        } else if best_h2 == 1.0 {
            Boundary::Upper
        } else {
            Boundary::Interior
        };

        // The interval: profile chi-square endpoints, limited flags, and the
        // mixture rule deciding whether a boundary point belongs.
        let mut interval = Interval::absent();
        if converged {
            let optimum = -loglik;
            let deviance = |h2: f64| -> f64 {
                let value = objective(h2);
                if value.is_finite() { 2.0 * (value - optimum) } else { f64::INFINITY }
            };
            let endpoint = |bound: f64| -> (f64, bool) {
                let mut outer = bound;
                if !deviance(outer).is_finite() {
                    // Bisect on feasibility toward the bound. Taking the first
                    // finite candidate stopped at the midpoint and truncated the
                    // interval whenever a crossing lay beyond it, which a
                    // singular relationship matrix made the common case.
                    let mut feasible = best_h2;
                    let mut infeasible = bound;
                    for _ in 0..200 {
                        if (infeasible - feasible).abs() <= 1e-15 {
                            break;
                        }
                        let candidate = 0.5 * (feasible + infeasible);
                        if deviance(candidate).is_finite() {
                            feasible = candidate;
                        } else {
                            infeasible = candidate;
                        }
                    }
                    outer = feasible;
                }
                if outer == best_h2 {
                    return (best_h2, true);
                }
                if deviance(outer) <= CHI2_ONE_DF_95 {
                    return (outer, true);
                }
                let crossing = |h2: f64| deviance(h2) - CHI2_ONE_DF_95;
                let (a, b) =
                    if best_h2 < outer { (best_h2, outer) } else { (outer, best_h2) };
                (bisect_root(&crossing, a, b), false)
            };
            let (lower, lower_limited) = endpoint(0.0);
            let (upper, upper_limited) = endpoint(1.0);
            interval = Interval {
                lower,
                upper,
                lower_limited,
                upper_limited,
                contains_lower_bound: (lower == 0.0).then(|| deviance(0.0) <= MIXTURE_CRIT),
                contains_upper_bound: (upper == 1.0).then(|| deviance(1.0) <= MIXTURE_CRIT),
            };
        }

        // The likelihood ratio test against no additive variance. Two fits, and
        // the second is free: the null is h² = 0 exactly, where the profiled
        // likelihood is already available. The fixed effects are the same under
        // both, which is what makes this a legitimate REML likelihood ratio.
        let test = if converged {
            let null_loglik = self.loglik_at(&yt, 0.0, reml);
            if null_loglik.is_finite() {
                let statistic = (2.0 * (loglik - null_loglik)).max(0.0);
                // Half the null's mass sits exactly at zero, because the
                // parameter is on its bound under the null: the 50:50 mixture
                // of chi-square on nought and one degrees of freedom. This is
                // the one case where that mixture is verified rather than
                // assumed (`docs/adr/0001`, decision 12).
                //
                // The atom has to be handled separately. For a positive
                // statistic the nought-degree part contributes nothing and the
                // tail is half the chi-square-on-one tail. At exactly zero the
                // statistic cannot be exceeded at all, so the p-value is one
                // rather than the half that formula would give — the estimate
                // sitting on the bound is the least surprising thing the null
                // can produce, not a coin flip. Both agree for every threshold
                // below a half, so this changes no decision; it changes what
                // the number means.
                let p_value = if statistic <= 0.0 {
                    1.0
                } else {
                    0.5 * chi2_one_df_upper_tail(statistic)
                };
                Some(LikelihoodRatioTest { statistic, null_loglik, p_value })
            } else {
                None
            }
        } else {
            None
        };

        // Observed information for the standard error, at interior points only.
        let standard_error = if converged && matches!(boundary, Boundary::Interior) {
            let value = |h2: f64| self.loglik_at(&yt, h2.clamp(0.0, 1.0), reml);
            let step = INFORMATION_STEP;
            let second = if step <= best_h2 && best_h2 <= 1.0 - step {
                (value(best_h2 + step) - 2.0 * value(best_h2) + value(best_h2 - step))
                    / (step * step)
            } else if best_h2 < 0.5 {
                (value(best_h2) - 2.0 * value(best_h2 + step) + value(best_h2 + 2.0 * step))
                    / (step * step)
            } else {
                (value(best_h2) - 2.0 * value(best_h2 - step) + value(best_h2 - 2.0 * step))
                    / (step * step)
            };
            let information = -second;
            (information.is_finite() && information > 0.0).then(|| 1.0 / information.sqrt())
        } else {
            None
        };

        Fit {
            h2: best_h2,
            total_variance: sigma2,
            beta: beta.iter().copied().collect(),
            loglik,
            converged,
            boundary,
            warnings,
            interval,
            standard_error,
            test,
        }
    }
}

/// Which bound, if any, the fit landed on. A state, not advice
/// (`docs/adr/0005`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Boundary {
    Interior,
    Lower,
    Upper,
}

impl Boundary {
    /// The state as it is written on the record.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interior => "interior",
            Self::Lower => "lower",
            Self::Upper => "upper",
        }
    }
}

/// The interval keeps one shape whatever the fit did (`docs/adr/0005`).
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
    pub lower_limited: bool,
    pub upper_limited: bool,
    pub contains_lower_bound: Option<bool>,
    pub contains_upper_bound: Option<bool>,
}

impl Interval {
    fn absent() -> Self {
        Self {
            lower: f64::NAN,
            upper: f64::NAN,
            lower_limited: false,
            upper_limited: false,
            contains_lower_bound: None,
            contains_upper_bound: None,
        }
    }
}

/// The test of no additive variance.
pub struct LikelihoodRatioTest {
    pub statistic: f64,
    pub null_loglik: f64,
    pub p_value: f64,
}

/// What a fit returns. Held in memory; nothing here writes to disk
/// (`docs/adr/0003`).
pub struct Fit {
    pub h2: f64,
    pub total_variance: f64,
    pub beta: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    pub boundary: Boundary,
    pub warnings: Vec<String>,
    pub interval: Interval,
    pub standard_error: Option<f64>,
    pub test: Option<LikelihoodRatioTest>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PreparedModel {
    #[new]
    #[pyo3(signature = (x, k, subject_ids=None, subject_order_sha256=None))]
    fn py_new(
        x: PyReadonlyArray2<'_, f64>,
        k: PyReadonlyArray2<'_, f64>,
        subject_ids: Option<Vec<String>>,
        subject_order_sha256: Option<String>,
    ) -> PyResult<Self> {
        if subject_ids.is_some() && subject_order_sha256.is_some() {
            return Err(code("PREPARE_SUBJECT_ORDER_AMBIGUOUS"));
        }
        let subject_order = match (subject_ids, subject_order_sha256) {
            (Some(ids), None) => {
                let mut joined = ids.join("\n");
                joined.push('\n');
                let mut hasher = Sha256::new();
                hasher.update(joined.as_bytes());
                Some(format!("{:x}", hasher.finalize()))
            }
            (None, Some(hex)) => {
                if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(code("PREPARE_SUBJECT_ORDER_SHA256_INVALID"));
                }
                Some(hex.to_ascii_lowercase())
            }
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!(),
        };

        let x = x.as_array();
        let k = k.as_array();
        let x_matrix = DMatrix::from_fn(x.shape()[0], x.shape()[1], |i, j| x[(i, j)]);
        let k_matrix = DMatrix::from_fn(k.shape()[0], k.shape()[1], |i, j| k[(i, j)]);
        Self::build(&x_matrix, &k_matrix, subject_order).map_err(code)
    }

    #[getter(n)]
    fn py_n(&self) -> usize {
        self.n
    }

    #[getter(p)]
    fn py_p(&self) -> usize {
        self.p
    }

    #[getter(min_eigenvalue)]
    fn py_min_eigenvalue(&self) -> f64 {
        self.min_eigenvalue
    }

    #[getter(subject_order)]
    fn py_subject_order(&self) -> Option<String> {
        self.subject_order.clone()
    }

    /// Fit the ordinary Gaussian model. The estimator is data on the record;
    /// the objective never depends on the response's values.
    #[pyo3(signature = (y, estimator="reml"))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'_, f64>,
        estimator: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let reml = match estimator {
            "reml" => true,
            "ml" => false,
            _ => return Err(code("FIT_ESTIMATOR_INVALID")),
        };
        let y = y.as_array();
        if y.len() != self.n {
            return Err(code("FIT_Y_LENGTH_MISMATCH"));
        }
        if y.iter().any(|value| !value.is_finite()) {
            return Err(code("FIT_Y_NOT_FINITE"));
        }
        let y = DVector::from_iterator(self.n, y.iter().copied());
        let fit = self.fit_one_trait(&y, reml);

        let record = PyDict::new(py);
        record.set_item("estimator", estimator)?;
        record.set_item("n", self.n)?;
        record.set_item("subject_order", self.subject_order.clone())?;
        record.set_item("h2", fit.h2)?;
        record.set_item("total_variance", fit.total_variance)?;
        record.set_item("beta", fit.beta)?;
        record.set_item("loglik", fit.loglik)?;
        record.set_item("converged", fit.converged)?;
        record.set_item("boundary", fit.boundary.as_str())?;
        record.set_item("warnings", fit.warnings)?;

        let interval = PyDict::new(py);
        interval.set_item("lower", fit.interval.lower)?;
        interval.set_item("upper", fit.interval.upper)?;
        interval.set_item("lower_limited", fit.interval.lower_limited)?;
        interval.set_item("upper_limited", fit.interval.upper_limited)?;
        interval.set_item("contains_lower_bound", fit.interval.contains_lower_bound)?;
        interval.set_item("contains_upper_bound", fit.interval.contains_upper_bound)?;
        interval.set_item("level", 0.95)?;
        interval.set_item("recipe", "profile_mixture")?;
        record.set_item("interval", interval)?;

        match fit.test {
            Some(test) => {
                let dict = PyDict::new(py);
                dict.set_item("null", "h2 = 0")?;
                dict.set_item("statistic", test.statistic)?;
                dict.set_item("null_loglik", test.null_loglik)?;
                dict.set_item("p_value", test.p_value)?;
                dict.set_item("rule", "mixture_50_50")?;
                record.set_item("test", dict)?;
            }
            None => record.set_item("test", py.None())?,
        }

        match fit.standard_error {
            Some(se) => {
                let errors = PyDict::new(py);
                errors.set_item("h2", se)?;
                record.set_item("standard_errors", errors)?;
            }
            None => record.set_item("standard_errors", py.None())?,
        }
        Ok(record)
    }
}
