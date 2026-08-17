//! Factorising one symmetric positive-definite matrix, by whichever library is
//! faster at that size.
//!
//! Two libraries do this and neither wins everywhere. Measured on a Cholesky
//! plus the inverse that a likelihood evaluation needs, single-threaded:
//!
//! ```text
//!    size    nalgebra        faer
//!       4      0.1 us      0.2 us     nalgebra
//!      10      0.9 us      0.6 us
//!      25     11.5 us      3.8 us     faer, three times
//!     100    273.5 us    103.1 us
//!     360      9.3 ms      2.7 ms     faer, three and a half times
//!    1800      1.0 s       0.06 s     faer, eighteen times
//! ```
//!
//! At four rows `faer` is half the speed: its setup costs more than the work.
//! From about ten rows it is ahead and stays ahead, by three times through the
//! sizes a family block reaches and by far more on one dense matrix of
//! everybody.
//!
//! **Both cases occur here and that is why this is a threshold rather than a
//! choice.** A pedigree is mostly small blocks — the GOBS families have a median
//! of one person and a mean of nine — but the cost is cubic in block size, so
//! the handful of large families is nearly all the work. Sending the singletons
//! through `faer` would be slower and sending the 180-person family through
//! `nalgebra` would be three times slower again.

use faer::linalg::solvers::{Llt, Solve};
use faer::{Mat, Side};
use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

/// Below this many rows `nalgebra` is faster and above it `faer` is, measured
/// rather than guessed. The crossover is near ten; twelve leaves a little room
/// rather than sitting on it.
const FAER_FROM: usize = 12;

/// One factorised covariance, and the things a likelihood asks of it.
pub(crate) enum DenseFactor {
    Small(Cholesky<f64, Dyn>),
    Large(Llt<f64>),
}

impl DenseFactor {
    /// Factorise, or `None` where the matrix is not positive definite.
    pub(crate) fn new(v: &DMatrix<f64>) -> Option<Self> {
        if v.nrows() < FAER_FROM {
            return v.clone().cholesky().map(Self::Small);
        }
        let n = v.nrows();
        let a = Mat::from_fn(n, n, |i, j| v[(i, j)]);
        Llt::new(a.as_ref(), Side::Lower).ok().map(Self::Large)
    }

    /// `log|V|`, from twice the sum of the logged diagonal of the factor.
    pub(crate) fn logdet(&self) -> f64 {
        match self {
            Self::Small(c) => 2.0 * c.l().diagonal().iter().map(|d| d.ln()).sum::<f64>(),
            Self::Large(l) => {
                let f = l.L();
                2.0 * (0..f.nrows()).map(|i| f[(i, i)].ln()).sum::<f64>()
            }
        }
    }

    /// Solve `V z = b`.
    pub(crate) fn solve_vector(&self, b: &DVector<f64>) -> DVector<f64> {
        match self {
            Self::Small(c) => c.solve(b),
            Self::Large(l) => {
                let n = b.len();
                let out = l.solve(Mat::from_fn(n, 1, |i, _| b[i]).as_ref());
                DVector::from_fn(n, |i, _| out[(i, 0)])
            }
        }
    }

    /// Solve `V Z = B`.
    pub(crate) fn solve_matrix(&self, b: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Self::Small(c) => c.solve(b),
            Self::Large(l) => {
                let (n, p) = (b.nrows(), b.ncols());
                let out = l.solve(Mat::from_fn(n, p, |i, j| b[(i, j)]).as_ref());
                DMatrix::from_fn(n, p, |i, j| out[(i, j)])
            }
        }
    }

    /// `V⁻¹`, which the traces in a gradient need and a solve cannot replace.
    pub(crate) fn inverse(&self) -> DMatrix<f64> {
        match self {
            Self::Small(c) => c.inverse(),
            Self::Large(l) => {
                let n = l.L().nrows();
                let out = l.solve(Mat::<f64>::identity(n, n).as_ref());
                DMatrix::from_fn(n, n, |i, j| out[(i, j)])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseFactor, FAER_FROM};
    use nalgebra::{DMatrix, DVector};

    /// The two routes must agree. They are chosen by size, so a difference
    /// between them would show up as an answer that depended on how many people
    /// were in a family -- which is exactly the kind of fault that looks like
    /// science.
    #[test]
    fn both_routes_give_the_same_answer() {
        for n in [4usize, 11, 12, 40, 120] {
            let v = DMatrix::from_fn(n, n, |i, j| {
                0.5f64.powi(i32::try_from(i.abs_diff(j)).unwrap_or(i32::MAX))
                    + if i == j { 1.0 } else { 0.0 }
            });
            let b = DVector::from_fn(n, |i, _| (i as f64).sin());
            let x = DMatrix::from_fn(n, 3, |i, j| ((i * 3 + j) as f64).cos());

            let chosen = DenseFactor::new(&v).expect("positive definite");
            // Whatever the threshold picked, compare against nalgebra directly.
            let reference = v.clone().cholesky().expect("positive definite");

            assert!(
                (chosen.logdet()
                    - 2.0 * reference.l().diagonal().iter().map(|d| d.ln()).sum::<f64>())
                .abs()
                    < 1e-9,
                "log determinant differs at n = {n}"
            );
            assert!(
                (chosen.solve_vector(&b) - reference.solve(&b)).amax() < 1e-9,
                "vector solve differs at n = {n}"
            );
            assert!(
                (chosen.solve_matrix(&x) - reference.solve(&x)).amax() < 1e-9,
                "matrix solve differs at n = {n}"
            );
            assert!(
                (chosen.inverse() - reference.inverse()).amax() < 1e-9,
                "inverse differs at n = {n}"
            );
        }
    }

    /// A matrix that is not positive definite is refused rather than returning
    /// something, on both routes.
    #[test]
    fn a_matrix_that_is_not_positive_definite_is_refused() {
        for n in [4usize, FAER_FROM + 5] {
            let mut v = DMatrix::<f64>::identity(n, n);
            v[(0, 0)] = -1.0;
            assert!(
                DenseFactor::new(&v).is_none(),
                "accepted a negative pivot at n = {n}"
            );
        }
    }
}
