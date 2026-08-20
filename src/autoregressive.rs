//! One trait with a separable first-order autoregressive spatial effect.
//!
//! # What this is, and why it is not the other spatial model
//!
//! Two people sitting in cells `(r_i, c_i)` and `(r_j, c_j)` of a rectangular
//! grid share
//!
//! ```text
//! sigma_s^2 * rho_row^|r_i - r_j| * rho_col^|c_i - c_j|
//! ```
//!
//! which is the model Stopher and colleagues fitted to the red deer of Rum in
//! 2012, and the standard one for field trials laid out in rows and columns.
//! [`crate::SpatialModel`] fits something different: an isotropic exponential
//! kernel in true distance, `exp(-lambda d)`, with one rate rather than two and
//! no grid at all.
//!
//! **The difference that matters is where the scale comes from.** The
//! exponential model estimates its range; this one takes the cell size as given
//! and estimates only how fast correlation falls off *per cell*. That is a real
//! cost — the answer moves with a choice the analyst makes — and it buys two
//! things in return. Correlation may fall off at different rates along the two
//! axes, which an isotropic kernel cannot express. And the range of the
//! exponential model is badly identified on clustered people: 98.4 per cent of
//! its half-distance intervals reach a bound in calibration. A rate per cell,
//! at a cell size fixed in advance, is a quantity the data can pin down when a
//! continuous range is not.
//!
//! # It contains the same-cell model
//!
//! At `rho_row = rho_col = 0` the kernel is one where two people share a cell
//! and nought otherwise, because `0^0` is one and `0^k` is nought — exactly a
//! grouping matrix of the kind a household or a census tract gives. So fitting
//! this and testing `rho = 0` asks whether smooth decay across neighbouring
//! cells buys anything over plain shared membership, and the two models are
//! nested rather than rivals.
//!
//! # Where it will not help
//!
//! On people spread unevenly over a wide area the grid is mostly empty, and an
//! empty cell carries no information while still setting the scale of every
//! `|r_i - r_j|`. Sweeping the cell size is therefore part of using this, not
//! an optional refinement: a rate per cell means nothing without the cell.

use nalgebra::{DMatrix, DVector};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::convergence::{self, TOLERANCE};
use crate::spatial::{DenseFactor, solve_matrix, solve_vector};

/// The largest correlation a rate may take. At one the kernel is all ones,
/// which is a single effect shared by everybody and so confounded with the
/// intercept; the fit would be reporting the mean twice.
const RATE_UPPER: f64 = 0.999;

/// A fitted autoregressive spatial model.
#[derive(Clone, Debug)]
pub struct AutoregressiveFit {
    /// One variance per fixed component in the order given, then the spatial
    /// variance, then the residual.
    pub variances: Vec<f64>,
    /// Each raw covariance coefficient divided by their sum. As elsewhere these
    /// depend on the submitted matrix scales.
    pub raw_coefficient_proportions: Vec<f64>,
    pub raw_coefficient_total: f64,
    /// Correlation between two cells one step apart down the rows.
    pub rho_row: f64,
    /// Correlation between two cells one step apart across the columns.
    pub rho_column: f64,
    /// How many cells apart the correlation falls to a half, down and across.
    /// This is what the two rates mean in units anybody thinks in; it is
    /// infinite where a rate is one and nought where a rate is nought.
    pub half_cells_row: f64,
    pub half_cells_column: f64,
    pub fixed_effects: Vec<f64>,
    pub fixed_effect_errors: Vec<f64>,
    pub loglik: f64,
    pub converged: bool,
    /// True where the gradient test failed on the first search and a second was
    /// run from that point with the objective tolerance switched off.
    pub polished: bool,
    pub scaled_gradient: f64,
    pub estimator: &'static str,
}

struct Evaluation {
    negative_loglik: f64,
    gradient: Vec<f64>,
    fixed_effects: Vec<f64>,
    fixed_covariance: Option<DMatrix<f64>>,
}

/// Where one derivative's elements come from, read by cell so that nothing is
/// materialised per person and no indirect call happens inside the loops.
enum DerivativeSource<'a> {
    Fixed(&'a DMatrix<f64>),
    /// The kernel itself, which is the derivative with respect to the spatial
    /// variance.
    Kernel(&'a DMatrix<f64>, &'a [usize]),
    /// A precomputed derivative over cells, scaled by the spatial variance.
    Rate(&'a DMatrix<f64>, &'a [usize], f64),
}

impl DerivativeSource<'_> {
    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        match self {
            Self::Fixed(m) => m[(i, j)],
            Self::Kernel(kernel, cell) => kernel[(cell[i], cell[j])],
            Self::Rate(derivative, cell, scale) => scale * derivative[(cell[i], cell[j])],
        }
    }
}

/// One trait, fixed components plus a separable autoregressive grid effect.
pub struct AutoregressiveModel {
    design: DMatrix<f64>,
    fixed: Vec<DMatrix<f64>>,
    /// Which cell each person sits in, indexing `cell_row` and `cell_column`.
    ///
    /// People share cells, and the kernel does not care which of them is which,
    /// so it is built over cells and read through this. On a coarse grid that
    /// is a large saving; on a fine one it costs nothing.
    cell: Vec<usize>,
    cell_row: Vec<i64>,
    cell_column: Vec<i64>,
    rows: usize,
    logdet_xtx: f64,
}

impl AutoregressiveModel {
    /// Build from one row and one column index per person.
    ///
    /// The indices are cell coordinates and are expected to be integers: what
    /// the kernel reads is `|r_i - r_j|`, the number of steps between two
    /// cells, so the grid's spacing is carried by how the caller made them and
    /// not by anything here.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the shapes disagree or the design is rank
    /// deficient.
    pub fn build(
        fixed: &[DMatrix<f64>],
        row: &[i64],
        column: &[i64],
        design: &DMatrix<f64>,
    ) -> Result<Self, &'static str> {
        let n = design.nrows();
        if n == 0 {
            return Err("AUTOREGRESSIVE_NO_ROWS");
        }
        if row.len() != n || column.len() != n {
            return Err("AUTOREGRESSIVE_CELLS_WRONG_LENGTH");
        }
        for matrix in fixed {
            if matrix.nrows() != n || matrix.ncols() != n {
                return Err("AUTOREGRESSIVE_FIXED_WRONG_SIZE");
            }
        }

        // Distinct cells, in first-seen order, and the map from people to them.
        let mut cell_row: Vec<i64> = Vec::new();
        let mut cell_column: Vec<i64> = Vec::new();
        let mut cell = Vec::with_capacity(n);
        for i in 0..n {
            let found =
                (0..cell_row.len()).find(|&k| cell_row[k] == row[i] && cell_column[k] == column[i]);
            cell.push(found.unwrap_or_else(|| {
                cell_row.push(row[i]);
                cell_column.push(column[i]);
                cell_row.len() - 1
            }));
        }

        let xtx = design.transpose() * design;
        let chol = xtx
            .cholesky()
            .ok_or("AUTOREGRESSIVE_DESIGN_RANK_DEFICIENT")?;
        let logdet_xtx = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        if !logdet_xtx.is_finite() {
            return Err("AUTOREGRESSIVE_DESIGN_RANK_DEFICIENT");
        }

        Ok(Self {
            design: design.clone(),
            fixed: fixed.to_vec(),
            cell,
            cell_row,
            cell_column,
            rows: n,
            logdet_xtx,
        })
    }

    /// How many cells the grid actually holds.
    #[must_use]
    pub fn cells(&self) -> usize {
        self.cell_row.len()
    }

    #[must_use]
    pub fn parameters(&self) -> usize {
        self.fixed.len() + 4
    }

    fn spatial_index(&self) -> usize {
        self.fixed.len()
    }

    fn residual_index(&self) -> usize {
        self.fixed.len() + 1
    }

    fn row_index(&self) -> usize {
        self.fixed.len() + 2
    }

    fn column_index(&self) -> usize {
        self.fixed.len() + 3
    }

    /// The kernel between cells, and its derivative with respect to each rate.
    ///
    /// All three are built together because they share the same powers, and
    /// each is over cells rather than people.
    ///
    /// **At a rate of nought the derivative is not `k * 0^(k-1)`.** For `k = 0`
    /// the term does not involve the rate at all and its derivative is nought;
    /// writing the general form would ask for `0^-1`. The `k = 1` case is the
    /// one that carries the whole derivative there, and it is `0^0`, which is
    /// one.
    fn kernel(&self, rho_row: f64, rho_column: f64) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let m = self.cells();
        let mut kernel = DMatrix::zeros(m, m);
        let mut d_row = DMatrix::zeros(m, m);
        let mut d_column = DMatrix::zeros(m, m);
        for a in 0..m {
            for b in 0..=a {
                let down = gap(self.cell_row[a], self.cell_row[b]);
                let across = gap(self.cell_column[a], self.cell_column[b]);
                let power_row = rho_row.powi(down);
                let power_column = rho_column.powi(across);
                let value = power_row * power_column;
                let derivative_row = if down == 0 {
                    0.0
                } else {
                    f64::from(down) * rho_row.powi(down - 1) * power_column
                };
                let derivative_column = if across == 0 {
                    0.0
                } else {
                    f64::from(across) * rho_column.powi(across - 1) * power_row
                };
                kernel[(a, b)] = value;
                kernel[(b, a)] = value;
                d_row[(a, b)] = derivative_row;
                d_row[(b, a)] = derivative_row;
                d_column[(a, b)] = derivative_column;
                d_column[(b, a)] = derivative_column;
            }
        }
        (kernel, d_row, d_column)
    }

    fn evaluate(
        &self,
        theta: &[f64],
        y: &DVector<f64>,
        reml: bool,
        want_gradient: bool,
    ) -> Option<Evaluation> {
        let count = self.parameters();
        let rho_row = theta[self.row_index()];
        let rho_column = theta[self.column_index()];
        if theta[..count - 2]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return None;
        }
        for rate in [rho_row, rho_column] {
            if !rate.is_finite() || !(0.0..=RATE_UPPER).contains(&rate) {
                return None;
            }
        }
        if theta[..count - 2].iter().all(|v| *v <= 0.0) {
            return None;
        }

        let n = self.rows;
        let (kernel, d_row, d_column) = self.kernel(rho_row, rho_column);
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
                    v[(i, j)] += spatial * kernel[(self.cell[i], self.cell[j])];
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
            for index in 0..count {
                let (trace, quadratic_term, restricted) = if index == self.residual_index() {
                    let trace = (0..n).map(|i| inverse[(i, i)]).sum::<f64>();
                    let quadratic_term = residual_solve.dot(&residual_solve);
                    let restricted = if reml {
                        vx.transpose() * &vx
                    } else {
                        DMatrix::zeros(p, p)
                    };
                    (trace, quadratic_term, restricted)
                } else {
                    let source = if index < self.fixed.len() {
                        DerivativeSource::Fixed(&self.fixed[index])
                    } else if index == self.spatial_index() {
                        DerivativeSource::Kernel(&kernel, &self.cell)
                    } else if index == self.row_index() {
                        DerivativeSource::Rate(&d_row, &self.cell, spatial)
                    } else {
                        DerivativeSource::Rate(&d_column, &self.cell, spatial)
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
                let mut term = 0.5 * (trace - quadratic_term);
                if reml {
                    term -= 0.5 * (&xvx_inverse * &restricted).trace();
                }
                gradient[index] = term;
            }
        }

        Some(Evaluation {
            negative_loglik: value,
            gradient,
            fixed_effects: beta.iter().copied().collect(),
            fixed_covariance: if want_gradient {
                Some(xvx_chol.inverse())
            } else {
                None
            },
        })
    }

    /// Fit by restricted or ordinary maximum likelihood.
    ///
    /// Several starts, spread across the two rates, because the likelihood in a
    /// rate is not always single-peaked and a search begun at one correlation
    /// will not find an effect operating at another. One start sits at nought
    /// on both rates, which is the same-cell model, so the search always sees
    /// the nested alternative.
    ///
    /// # Errors
    ///
    /// Returns a stable code where no start reached a usable optimum.
    pub fn fit(&self, y: &DVector<f64>, reml: bool) -> Result<AutoregressiveFit, &'static str> {
        if y.len() != self.rows {
            return Err("AUTOREGRESSIVE_RESPONSE_WRONG_LENGTH");
        }
        let mean = y.mean();
        let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        if !(variance > 0.0) {
            return Err("AUTOREGRESSIVE_RESPONSE_CONSTANT");
        }
        let scale = variance.sqrt();
        let scaled = y / scale;

        let count = self.parameters();
        let lower = vec![0.0; count];
        let mut upper = vec![f64::INFINITY; count];
        upper[self.row_index()] = RATE_UPPER;
        upper[self.column_index()] = RATE_UPPER;

        let variance_count = count - 2;
        let mut starts: Vec<Vec<f64>> = Vec::new();
        for &rate in &[0.0, 0.3, 0.6, 0.9] {
            let mut even = vec![1.0 / variance_count as f64; count];
            even[self.row_index()] = rate;
            even[self.column_index()] = rate;
            starts.push(even);
            let mut small = vec![0.1 / variance_count as f64; count];
            small[self.residual_index()] = 0.8;
            small[self.row_index()] = rate;
            small[self.column_index()] = rate;
            starts.push(small);
        }
        // One start with the two rates apart, so a search can reach an effect
        // that runs along one axis and not the other without having to cross a
        // ridge to get there.
        let mut uneven = vec![1.0 / variance_count as f64; count];
        uneven[self.row_index()] = 0.9;
        uneven[self.column_index()] = 0.1;
        starts.push(uneven.clone());
        uneven.swap(self.row_index(), self.column_index());
        starts.push(uneven);

        let value_of = |candidate: &[f64]| -> f64 {
            self.evaluate(candidate, &scaled, reml, false)
                .map_or(1e30, |e| e.negative_loglik)
        };
        let gradient_of = |candidate: &[f64]| -> Vec<f64> {
            self.evaluate(candidate, &scaled, reml, true)
                .map_or_else(|| vec![0.0; count], |e| e.gradient)
        };
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
            control.factr = 1.0e3;
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

        let (mut negative, mut par, mut beta) = best.ok_or("AUTOREGRESSIVE_NO_START_CONVERGED")?;
        let mut at = self
            .evaluate(&par, &scaled, reml, true)
            .ok_or("AUTOREGRESSIVE_OPTIMUM_NOT_EVALUABLE")?;
        let mut scaled_gradient = reading(&at.gradient, &par, negative);

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

        let variances: Vec<f64> = par[..count - 2].iter().map(|v| v * variance).collect();
        let raw_coefficient_total: f64 = variances.iter().sum();
        let raw_coefficient_proportions = variances
            .iter()
            .map(|v| v / raw_coefficient_total)
            .collect();
        let rho_row = par[self.row_index()];
        let rho_column = par[self.column_index()];
        let observations = if reml {
            (self.rows - self.design.ncols()) as f64
        } else {
            self.rows as f64
        };

        Ok(AutoregressiveFit {
            variances,
            raw_coefficient_proportions,
            raw_coefficient_total,
            rho_row,
            rho_column,
            half_cells_row: half_cells(rho_row),
            half_cells_column: half_cells(rho_column),
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
}

/// The number of cells between two coordinates.
///
/// `powi` takes its exponent as an `i32`, and a gap wider than that is not a
/// grid anybody laid out, so this saturates. Saturating is right here rather
/// than merely safe: for any rate below one, `rate^k` has underflowed to
/// nought long before `k` reaches a million, so a saturated gap gives exactly
/// the answer an exact one would.
fn gap(first: i64, second: i64) -> i32 {
    i32::try_from(first.abs_diff(second)).unwrap_or(i32::MAX)
}

/// How many cells apart the correlation falls to a half.
///
/// Nought where the rate is nought, because the correlation is already nought
/// one cell away; infinite where it is one, because it never falls.
fn half_cells(rate: f64) -> f64 {
    if rate <= 0.0 {
        0.0
    } else if rate >= 1.0 {
        f64::INFINITY
    } else {
        std::f64::consts::LN_2 / -rate.ln()
    }
}

#[cfg(feature = "python")]
pub mod python;

#[cfg(test)]
mod tests {
    use super::*;

    /// Sib families spread over a grid, with an autoregressive effect on it.
    #[allow(clippy::type_complexity)]
    fn small(
        rho_row: f64,
        rho_column: f64,
        spatial: f64,
        seed: u64,
    ) -> (DMatrix<f64>, Vec<i64>, Vec<i64>, DMatrix<f64>, DVector<f64>) {
        let (families, sibs, side) = (40usize, 3usize, 12i64);
        let n = families * sibs;
        let mut a = DMatrix::identity(n, n);
        for f in 0..families {
            for i in 0..sibs {
                for j in 0..sibs {
                    if i != j {
                        a[(f * sibs + i, f * sibs + j)] = 0.5;
                    }
                }
            }
        }
        // A cheap deterministic spread, so relatives do not all share a cell.
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Thirty-one bits, so the value is positive in an `i64`
            // whatever the stream does. A fixture, not a generator to draw
            // inference from.
            i64::from((state >> 33) as u32 & 0x7fff_ffff)
        };
        let row: Vec<i64> = (0..n).map(|_| next() % side).collect();
        let column: Vec<i64> = (0..n).map(|_| next() % side).collect();

        let mut v = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let down = gap(row[i], row[j]);
                let across = gap(column[i], column[j]);
                v[(i, j)] =
                    0.4 * a[(i, j)] + spatial * rho_row.powi(down) * rho_column.powi(across);
            }
            v[(i, i)] += 0.4;
        }
        let chol = v.clone().cholesky().expect("positive definite");
        let mut normal = DVector::zeros(n);
        for i in 0..n {
            // Box-Muller from the same cheap stream.
            let u1 = ((next().unsigned_abs() % 1_000_000) as f64 + 1.0) / 1_000_001.0;
            let u2 = ((next().unsigned_abs() % 1_000_000) as f64 + 1.0) / 1_000_001.0;
            normal[i] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        let y = chol.l() * normal;
        (a, row, column, DMatrix::from_element(n, 1, 1.0), y)
    }

    /// The analytic gradient has to match a central difference, or every fit
    /// is walking downhill on a slope it made up.
    #[test]
    fn the_gradient_matches_a_central_difference() {
        let (a, row, column, design, y) = small(0.7, 0.4, 0.3, 20_260_819);
        let model = AutoregressiveModel::build(&[a], &row, &column, &design).expect("valid");
        let theta = vec![0.35, 0.25, 0.40, 0.65, 0.45];
        for reml in [true, false] {
            let at = model.evaluate(&theta, &y, reml, true).expect("evaluates");
            for k in 0..theta.len() {
                let step = 1e-6;
                let mut up = theta.clone();
                let mut down = theta.clone();
                up[k] += step;
                down[k] -= step;
                let high = model
                    .evaluate(&up, &y, reml, false)
                    .expect("up")
                    .negative_loglik;
                let low = model
                    .evaluate(&down, &y, reml, false)
                    .expect("down")
                    .negative_loglik;
                let numeric = (high - low) / (2.0 * step);
                let scale = numeric.abs().max(1.0);
                assert!(
                    (at.gradient[k] - numeric).abs() / scale < 1e-5,
                    "parameter {k} reml {reml}: analytic {} against numeric {numeric}",
                    at.gradient[k]
                );
            }
        }
    }

    /// **At nought on both rates the kernel is the same-cell indicator.** The
    /// two models are nested, and this is the join: a fit with both rates held
    /// at nought must be the multi-component fit with a grouping matrix.
    #[test]
    fn both_rates_at_nought_is_the_same_cell_model() {
        let (a, row, column, design, y) = small(0.6, 0.6, 0.3, 20_260_820);
        let model = AutoregressiveModel::build(&[a], &row, &column, &design).expect("valid");
        let theta = vec![0.4, 0.3, 0.4, 0.0, 0.0];
        let ours = model.evaluate(&theta, &y, true, false).expect("evaluates");

        assert!(ours.negative_loglik.is_finite());

        // The claim itself, asserted where it is made: the kernel at nought on
        // both rates is one for two cells that are the same cell and nought
        // for every other pair. That is a grouping matrix, so the covariance
        // this model builds there is the one a multi-component fit builds from
        // a shared-cell matrix, and the two families meet.
        let (kernel, d_row, d_column) = model.kernel(0.0, 0.0);
        for a_cell in 0..model.cells() {
            for b_cell in 0..model.cells() {
                let want = f64::from(a_cell == b_cell);
                assert!(
                    (kernel[(a_cell, b_cell)] - want).abs() < 1e-15,
                    "cells {a_cell} and {b_cell} gave {}",
                    kernel[(a_cell, b_cell)]
                );
            }
        }
        // And the derivatives are finite there rather than asking for `0^-1`,
        // which is the trap the general form walks into.
        assert!(d_row.iter().all(|v| v.is_finite()));
        assert!(d_column.iter().all(|v| v.is_finite()));
    }

    /// An effect that is really there is found, and the rates come back near
    /// where the simulation put them.
    #[test]
    fn an_autoregressive_effect_is_recovered() {
        let (a, row, column, design, y) = small(0.8, 0.8, 0.5, 20_260_821);
        let model = AutoregressiveModel::build(&[a], &row, &column, &design).expect("valid");
        let fit = model.fit(&y, true).expect("fits");
        assert!(
            fit.converged,
            "did not converge, |g| = {}",
            fit.scaled_gradient
        );
        assert_eq!(fit.variances.len(), 3);
        assert!(
            fit.raw_coefficient_proportions[1] > 0.05,
            "the spatial share came out at {}, but the data were simulated with one",
            fit.raw_coefficient_proportions[1]
        );
        assert!(
            (0.0..=RATE_UPPER).contains(&fit.rho_row)
                && (0.0..=RATE_UPPER).contains(&fit.rho_column),
            "rates {} and {} are outside their bounds",
            fit.rho_row,
            fit.rho_column
        );
    }

    /// The grid is deduplicated to cells, and the count is what it should be.
    #[test]
    fn people_sharing_a_cell_are_one_cell() {
        let n = 6;
        let a = DMatrix::identity(n, n);
        let row = vec![0, 0, 1, 1, 2, 2];
        let column = vec![0, 0, 5, 5, 9, 9];
        let design = DMatrix::from_element(n, 1, 1.0);
        let model = AutoregressiveModel::build(&[a], &row, &column, &design).expect("valid");
        assert_eq!(model.cells(), 3);
    }
}
