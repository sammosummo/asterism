//! The correlation between two positions, from two numbers.
//!
//! # What it is
//!
//! ```text
//! corr(t, u) = c + (1 - c) exp(-lambda * |d_t - d_u|)
//! ```
//!
//! where `d` is where the caller has put each position on a line. **The crate
//! does not choose that line.** For the audiogram it is the ERB-number scale,
//! which predicts the observed decay slightly better than log frequency does
//! (−0.76 against −0.71), but nothing here knows that: it takes numbers and
//! treats their differences as separations.
//!
//! # Why there is a floor
//!
//! In the data this was built for, the correlation between thresholds falls
//! with separation and then flattens near 0.3 rather than decaying to nought:
//! 0.96 at one ERB, 0.83 at six, 0.56 at eleven, 0.29 at thirty-one. A bare
//! exponential cannot represent that, and a free factor model can but throws
//! away the ordering that makes the audiogram an audiogram.
//!
//! The floor is a common factor and the exponential is the local decay, and
//! **the fit chooses between them rather than the analyst**. At a large `lambda`
//! the off-diagonal is `c` everywhere, which is a one-factor model. At `c` of
//! nought it is a pure distance model. Everything between is available and the
//! likelihood picks.
//!
//! # Why it is positive definite
//!
//! `R = c J + (1 - c) K`, where `J` is the matrix of ones and `K` is the
//! exponential kernel in separation. `J` is positive semidefinite with rank
//! one; `K` is positive definite for any positive `lambda` and distinct
//! positions. A positive combination of the two is positive definite for any
//! `c` in `[0, 1)`, so the parameter space needs bounds and not repairs.
//!
//! Both ends of it are singular. At `c` of one, `R` is `J`. At `lambda` of
//! nought, `K` is `J` and `R` is `J` again. The bounds exclude both.
//!
//! # What this costs the model
//!
//! A free covariance at seventeen positions is 153 numbers per component. This
//! is nineteen: a variance at every position, a floor and a rate. That is what
//! makes the fit's own gradient affordable to read -- 125 parameters against
//! 527 -- as much as it is what makes the estimates readable.

use nalgebra::DMatrix;
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

use crate::dense::DenseFactor;

/// The largest floor. At one the correlation is one everywhere and the matrix
/// is a single common factor with no variance of its own left.
pub(crate) const FLOOR_MAX: f64 = 0.999;

/// The smallest and largest decay rates. **These are in the caller's units**,
/// because the separations are. At the bottom the correlation is nearly one
/// across the whole span and the matrix is ill conditioned rather than wrong;
/// at the top it is nearly the floor between any two distinct positions.
pub(crate) const RATE_MIN: f64 = 1.0e-4;
pub(crate) const RATE_MAX: f64 = 1.0e4;

/// How hard the conditional maximisation tries. It does not have to reach the
/// maximum: expectation-maximisation only needs the expected complete-data
/// likelihood to rise, and starting from where the last iteration left off
/// guarantees it cannot fall.
const INNER_ITERATIONS: usize = 60;

/// Separations between positions, formed once.
pub(crate) struct Kernel {
    separation: DMatrix<f64>,
    positions: usize,
}

impl Kernel {
    /// # Errors
    ///
    /// Returns a stable code where the positions do not describe a line: fewer
    /// than three of them, a value that is not finite, or two at the same
    /// place. Three because two positions have one correlation between them and
    /// two parameters to explain it with, which is a ridge rather than a fit.
    pub(crate) fn new(positions: &[f64]) -> Result<Self, &'static str> {
        let size = positions.len();
        if size < 3 {
            return Err("REPEATED_KERNEL_NEEDS_THREE_POSITIONS");
        }
        if !positions.iter().all(|v| v.is_finite()) {
            return Err("REPEATED_KERNEL_POSITION_NOT_FINITE");
        }
        for i in 0..size {
            for j in 0..i {
                if (positions[i] - positions[j]).abs() < 1e-12 {
                    return Err("REPEATED_KERNEL_POSITIONS_COINCIDE");
                }
            }
        }
        Ok(Self {
            separation: DMatrix::from_fn(size, size, |i, j| (positions[i] - positions[j]).abs()),
            positions: size,
        })
    }

    pub(crate) fn positions(&self) -> usize {
        self.positions
    }

    /// The correlation at one separation, which is what a reader should be
    /// given instead of a matrix.
    pub(crate) fn at(floor: f64, rate: f64, separation: f64) -> f64 {
        floor + (1.0 - floor) * (-rate * separation).exp()
    }

    /// The whole correlation matrix.
    pub(crate) fn correlation(&self, floor: f64, rate: f64) -> DMatrix<f64> {
        self.separation
            .map(|separation| Self::at(floor, rate, separation))
    }

    /// `dR / d floor`, which is nought on the diagonal because a separation of
    /// nought gives a correlation of one however the floor moves.
    fn floor_derivative(&self, rate: f64) -> DMatrix<f64> {
        self.separation
            .map(|separation| 1.0 - (-rate * separation).exp())
    }

    /// `dR / d rate`.
    fn rate_derivative(&self, floor: f64, rate: f64) -> DMatrix<f64> {
        self.separation
            .map(|separation| -(1.0 - floor) * separation * (-rate * separation).exp())
    }
}

/// One component's covariance, as the kernel builds it: a scale at every
/// position, and one floor and one rate shared across them.
#[derive(Clone, Debug)]
pub(crate) struct Shaped {
    /// The standard deviation at each position, so that the covariance is a
    /// correlation matrix scaled on both sides.
    pub scale: Vec<f64>,
    pub floor: f64,
    pub rate: f64,
}

impl Shaped {
    pub(crate) fn covariance(&self, kernel: &Kernel) -> DMatrix<f64> {
        let correlation = kernel.correlation(self.floor, self.rate);
        DMatrix::from_fn(self.scale.len(), self.scale.len(), |i, j| {
            self.scale[i] * self.scale[j] * correlation[(i, j)]
        })
    }

    /// The same shape with every parameter back inside its bounds.
    ///
    /// **This is the kernel family's version of projecting onto the
    /// covariances.** An extrapolated step is a guess and can be one that is
    /// not a model -- a negative scale, a floor past one, a rate at nought --
    /// and each of those is a boundary the search was going to stop at anyway.
    pub(crate) fn clamped(self) -> Self {
        Self {
            scale: self.scale.iter().map(|s| s.max(0.0)).collect(),
            floor: self.floor.clamp(0.0, FLOOR_MAX),
            rate: self.rate.clamp(RATE_MIN, RATE_MAX),
        }
    }

    /// The lower and upper bound of every packed parameter, in the same order.
    pub(crate) fn bounds(positions: usize) -> (Vec<f64>, Vec<f64>) {
        let mut lower = vec![0.0; positions];
        lower.push(0.0);
        lower.push(RATE_MIN);
        let mut upper = vec![f64::INFINITY; positions];
        upper.push(FLOOR_MAX);
        upper.push(RATE_MAX);
        (lower, upper)
    }

    pub(crate) fn packed(&self) -> Vec<f64> {
        let mut par = self.scale.clone();
        par.push(self.floor);
        par.push(self.rate);
        par
    }

    pub(crate) fn unpacked(par: &[f64]) -> Self {
        let positions = par.len() - 2;
        Self {
            scale: par[..positions].to_vec(),
            floor: par[positions],
            rate: par[positions + 1],
        }
    }
}

/// The negative of the expected complete-data log-likelihood for one component,
/// and its derivatives.
///
/// The expectation step leaves a sufficient statistic `S` and a count `weight`,
/// and what is left to do is
///
/// ```text
/// Q(theta) = -(1/2) [ weight * log|Sigma(theta)| + tr(Sigma(theta)^-1 S) ]
/// ```
///
/// which is a Wishart log-likelihood and nothing more. Without a kernel its
/// maximiser is `S / weight` in closed form; with one it is the nearest member
/// of the kernel family and has to be searched for.
fn objective(
    kernel: &Kernel,
    statistic: &DMatrix<f64>,
    weight: f64,
    par: &[f64],
) -> Option<(f64, Vec<f64>)> {
    let shaped = Shaped::unpacked(par);
    let covariance = shaped.covariance(kernel);
    let factor = DenseFactor::new(&covariance)?;
    let logdet = factor.logdet();
    let inverse = factor.inverse();
    let weighted = &inverse * statistic * &inverse;
    let trace = (0..statistic.nrows())
        .map(|i| (&inverse * statistic)[(i, i)])
        .sum::<f64>();
    let negative = 0.5 * (weight * logdet + trace);
    if !negative.is_finite() {
        return None;
    }

    // `dQ = -(1/2) tr(dSigma * G)`, so the negative's derivative is the same
    // trace with the sign the other way round.
    let gradient_matrix = &inverse * weight - weighted;
    let correlation = kernel.correlation(shaped.floor, shaped.rate);
    let size = shaped.scale.len();
    let mut gradient = Vec::with_capacity(size + 2);
    for v in 0..size {
        // A scale appears on both sides of its own row and column, which is
        // where the factor of two that cancels the half comes from.
        gradient.push(
            (0..size)
                .map(|u| shaped.scale[u] * correlation[(v, u)] * gradient_matrix[(v, u)])
                .sum::<f64>(),
        );
    }
    for derivative in [
        kernel.floor_derivative(shaped.rate),
        kernel.rate_derivative(shaped.floor, shaped.rate),
    ] {
        let mut total = 0.0;
        for t in 0..size {
            for u in 0..size {
                total += shaped.scale[t]
                    * shaped.scale[u]
                    * derivative[(t, u)]
                    * gradient_matrix[(t, u)];
            }
        }
        gradient.push(0.5 * total);
    }
    if !gradient.iter().all(|g| g.is_finite()) {
        return None;
    }
    Some((negative, gradient))
}

/// The conditional maximisation: move one component's kernel parameters uphill
/// on the expected complete-data likelihood, from where they are now.
///
/// **It does not have to arrive.** Expectation-maximisation needs the expected
/// complete-data likelihood to rise and not to be maximised, so a search that
/// runs out of iterations still leaves a valid step -- which is what makes this
/// affordable to do at every iteration for every component. Starting from the
/// current parameters is what guarantees it never falls.
pub(crate) fn maximise(
    kernel: &Kernel,
    statistic: &DMatrix<f64>,
    weight: f64,
    from: &Shaped,
) -> Option<Shaped> {
    let start = from.packed();
    let (lower, upper) = Shaped::bounds(from.scale.len());
    let bounds = Bounds::new(lower, upper).ok()?;

    let value_of = |par: &[f64]| -> f64 {
        objective(kernel, statistic, weight, par).map_or(1e30, |(value, _)| value)
    };
    let gradient_of = |par: &[f64]| -> Vec<f64> {
        objective(kernel, statistic, weight, par)
            .map_or_else(|| vec![0.0; par.len()], |(_, gradient)| gradient)
    };
    let mut control = OptimControl::default_for_dimension(start.len());
    control.maxit = INNER_ITERATIONS;
    control.fnscale = value_of(&start).abs().max(1.0);
    control.parscale = vec![1.0; start.len()];
    control.factr = 1.0e5;
    control.pgtol = 1e-9;
    control.lmm = start.len().min(10);
    let solution =
        optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control).ok()?;

    // Keep the result only where it is no worse. A search that ends badly
    // leaves the component where it was, which is a step of nothing and still a
    // step that cannot lower the likelihood.
    let before = value_of(&start);
    let after = value_of(&solution.par);
    let par = if after.is_finite() && after <= before {
        solution.par
    } else {
        start
    };
    let shaped = Shaped::unpacked(&par);
    DenseFactor::new(&shaped.covariance(kernel))?;
    Some(shaped)
}

/// A starting shape for a component, from a covariance that is not shaped.
///
/// The scales are read straight off the diagonal. The floor and the rate come
/// from a coarse sweep rather than from a guess, because a rate is the one
/// parameter here whose scale depends on units the crate knows nothing about,
/// and a search started at the wrong order of magnitude in it can sit there.
pub(crate) fn nearest(kernel: &Kernel, covariance: &DMatrix<f64>, weight: f64) -> Option<Shaped> {
    let size = covariance.nrows();
    let scale: Vec<f64> = (0..size)
        .map(|i| covariance[(i, i)].max(0.0).sqrt())
        .collect();
    if scale.iter().all(|s| *s <= 0.0) {
        return None;
    }
    let statistic = covariance * weight;
    let mut best: Option<(f64, Shaped)> = None;
    for floor in [0.0, 0.2, 0.4, 0.6] {
        for rate in [0.01, 0.05, 0.2, 1.0, 5.0] {
            let candidate = Shaped {
                scale: scale.clone(),
                floor,
                rate,
            };
            if let Some((value, _)) = objective(kernel, &statistic, weight, &candidate.packed())
                && best.as_ref().is_none_or(|(seen, _)| value < *seen)
            {
                best = Some((value, candidate));
            }
        }
    }
    best.map(|(_, shaped)| shaped)
}

#[cfg(test)]
mod tests {
    use super::{Kernel, Shaped, maximise, nearest, objective};
    use nalgebra::DMatrix;

    fn line() -> Vec<f64> {
        // Uneven on purpose: the audiogram's positions are not equally spaced
        // on any scale, and an equally spaced test would not notice a kernel
        // that quietly used the index instead of the position.
        vec![0.0, 1.4, 3.1, 6.0, 6.4, 11.0, 18.5, 31.0]
    }

    fn statistic(kernel: &Kernel, truth: &Shaped, weight: f64) -> DMatrix<f64> {
        truth.covariance(kernel) * weight
    }

    #[test]
    fn the_kernel_is_a_correlation_matrix_between_its_two_extremes() {
        let kernel = Kernel::new(&line()).expect("the line should be a line");
        for &(floor, rate) in &[(0.0, 0.05), (0.3, 0.1), (0.6, 2.0)] {
            let r = kernel.correlation(floor, rate);
            for i in 0..r.nrows() {
                assert!((r[(i, i)] - 1.0).abs() < 1e-12, "diagonal at {i}");
                for j in 0..r.ncols() {
                    assert!((r[(i, j)] - r[(j, i)]).abs() < 1e-15);
                    assert!(r[(i, j)] >= floor - 1e-12 && r[(i, j)] <= 1.0 + 1e-12);
                }
            }
            assert!(
                r.clone().cholesky().is_some(),
                "positive definite at floor {floor}, rate {rate}"
            );
        }
        // A large rate leaves the floor between any two distinct positions,
        // which is a one-factor model; a floor of nought leaves pure decay.
        let factor = kernel.correlation(0.35, 1e3);
        assert!((factor[(0, 1)] - 0.35).abs() < 1e-9);
        let distance = kernel.correlation(0.0, 0.1);
        assert!((distance[(0, 7)] - (-0.1_f64 * 31.0).exp()).abs() < 1e-12);
    }

    #[test]
    fn the_gradient_is_the_slope_of_the_objective() {
        let kernel = Kernel::new(&line()).expect("the line should be a line");
        let truth = Shaped {
            scale: vec![1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0],
            floor: 0.3,
            rate: 0.09,
        };
        let weight = 40.0;
        let statistic = statistic(&kernel, &truth, weight);

        // Somewhere away from the truth, so the gradient is not nought.
        let at = Shaped {
            scale: vec![0.8, 1.5, 1.1, 1.0, 1.4, 1.0, 0.9, 1.2],
            floor: 0.45,
            rate: 0.15,
        };
        let par = at.packed();
        let (_, gradient) =
            objective(&kernel, &statistic, weight, &par).expect("the objective should evaluate");

        for index in 0..par.len() {
            let step = 1e-6 * par[index].abs().max(1e-3);
            let mut up = par.clone();
            let mut down = par.clone();
            up[index] += step;
            down[index] -= step;
            let high = objective(&kernel, &statistic, weight, &up)
                .expect("the objective should evaluate")
                .0;
            let low = objective(&kernel, &statistic, weight, &down)
                .expect("the objective should evaluate")
                .0;
            let numerical = (high - low) / (2.0 * step);
            assert!(
                (gradient[index] - numerical).abs() < 1e-5 * numerical.abs().max(1.0),
                "parameter {index}: analytic {} against numerical {numerical}",
                gradient[index]
            );
        }
    }

    #[test]
    fn the_maximisation_finds_the_shape_its_own_statistic_came_from() {
        // The sufficient statistic of a covariance that *is* in the family, at
        // which the maximiser has nothing to trade off and must land on it.
        let kernel = Kernel::new(&line()).expect("the line should be a line");
        let truth = Shaped {
            scale: vec![1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0],
            floor: 0.3,
            rate: 0.09,
        };
        let weight = 40.0;
        let statistic = statistic(&kernel, &truth, weight);

        let mut at =
            nearest(&kernel, &(&statistic / weight), weight).expect("a start should be found");
        for _ in 0..40 {
            at = maximise(&kernel, &statistic, weight, &at).expect("the search should run");
        }
        for index in 0..truth.scale.len() {
            assert!(
                (at.scale[index] - truth.scale[index]).abs() < 1e-4,
                "scale {index}: {} against {}",
                at.scale[index],
                truth.scale[index]
            );
        }
        assert!(
            (at.floor - truth.floor).abs() < 1e-4,
            "floor {} against {}",
            at.floor,
            truth.floor
        );
        assert!(
            (at.rate - truth.rate).abs() < 1e-4,
            "rate {} against {}",
            at.rate,
            truth.rate
        );
    }

    #[test]
    fn the_maximisation_never_goes_downhill() {
        let kernel = Kernel::new(&line()).expect("the line should be a line");
        // A statistic from outside the family, so there is a real trade-off.
        let mut statistic = DMatrix::<f64>::identity(8, 8) * 2.0;
        for i in 0..8 {
            for j in 0..8 {
                if i != j {
                    statistic[(i, j)] = 0.7 / (1.0 + (i as f64 - j as f64).abs());
                }
            }
        }
        let weight = 25.0;
        let statistic = statistic * weight;
        let mut at = Shaped {
            scale: vec![1.0; 8],
            floor: 0.5,
            rate: 1.0,
        };
        let mut previous = objective(&kernel, &statistic, weight, &at.packed())
            .expect("the objective should evaluate")
            .0;
        for step in 0..25 {
            at = maximise(&kernel, &statistic, weight, &at).expect("the search should run");
            let now = objective(&kernel, &statistic, weight, &at.packed())
                .expect("the objective should evaluate")
                .0;
            assert!(
                now <= previous + 1e-9,
                "step {step} rose from {previous} to {now}"
            );
            previous = now;
        }
    }

    #[test]
    fn a_line_that_is_not_a_line_is_refused() {
        assert_eq!(
            Kernel::new(&[0.0, 1.0]).err(),
            Some("REPEATED_KERNEL_NEEDS_THREE_POSITIONS")
        );
        assert_eq!(
            Kernel::new(&[0.0, 1.0, 1.0]).err(),
            Some("REPEATED_KERNEL_POSITIONS_COINCIDE")
        );
        assert_eq!(
            Kernel::new(&[0.0, 1.0, f64::NAN]).err(),
            Some("REPEATED_KERNEL_POSITION_NOT_FINITE")
        );
    }
}
