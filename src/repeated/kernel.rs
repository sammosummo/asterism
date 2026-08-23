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
/// What the objective returns where it cannot be evaluated at all -- a shape
/// whose covariance will not factorise, or one whose value or gradient is not
/// finite.
///
/// It is a large finite number rather than an infinity because the optimiser
/// has to be able to work with it. **Being finite is also the hazard**: a
/// search that never left it would be accepted as a step by an acceptance test
/// that asked whether the objective was finite, which it always is. So the
/// acceptance below compares against this name instead. `src/tobit.rs` carries
/// the same constant for the same reason, and learnt it the same way.
const INFEASIBLE: f64 = 1.0e30;
/// The step for differencing the objective where its analytic gradient is
/// unavailable. The same step the rest of the module uses.
const GRADIENT_STEP: f64 = 1.0e-5;

/// How the correlation falls away with separation, once the floor is taken
/// off.
///
/// **Both have a floor and a rate and nothing else**, so the two are compared
/// like for like: same parameter count, different shape, and the likelihood
/// picks. A third parameter for the shape would be a third thing trading off
/// against the two that already do.
///
/// The difference is the tail. `exp(-r d)` falls by a constant factor per unit
/// of separation forever; `exp(-(r d)^2)` falls slowly at first and then far
/// faster. **A sum of components makes that worse, not better**: the total
/// correlation is a weighted sum of the component curves, and at long
/// separation the slowest of them is what is left. So a model whose components
/// all have heavy tails overstates the correlation between distant positions,
/// which is what the audiogram fits do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    /// `exp(-rate * separation)`.
    Exponential,
    /// `exp(-(rate * separation)^2)`, the squared exponential.
    Gaussian,
}

impl Shape {
    /// The decay, before the floor is put back.
    fn decay(self, rate: f64, separation: f64) -> f64 {
        let scaled = rate * separation;
        match self {
            Self::Exponential => (-scaled).exp(),
            Self::Gaussian => (-scaled * scaled).exp(),
        }
    }

    /// `d decay / d rate`.
    fn decay_slope(self, rate: f64, separation: f64) -> f64 {
        let scaled = rate * separation;
        match self {
            Self::Exponential => -separation * (-scaled).exp(),
            Self::Gaussian => -2.0 * rate * separation * separation * (-scaled * scaled).exp(),
        }
    }

    /// The rate at which the decay reaches a given value at a given
    /// separation, which is the inverse of [`Shape::decay`].
    fn rate_reaching(self, decay: f64, separation: f64) -> Option<f64> {
        if !(separation > 0.0) || !(decay > 0.0) || decay >= 1.0 {
            return None;
        }
        let logged = -decay.ln();
        Some(match self {
            Self::Exponential => logged / separation,
            Self::Gaussian => logged.sqrt() / separation,
        })
    }

    /// The name a caller uses, and reads back.
    pub(crate) fn named(name: &str) -> Option<Self> {
        match name {
            "exponential" => Some(Self::Exponential),
            "gaussian" => Some(Self::Gaussian),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Exponential => "exponential",
            Self::Gaussian => "gaussian",
        }
    }
}

/// Separations between positions, formed once.
pub(crate) struct Kernel {
    separation: DMatrix<f64>,
    positions: usize,
    shape: Shape,
}

impl Kernel {
    /// # Errors
    ///
    /// Returns a stable code where the positions do not describe a line: fewer
    /// than three of them, a value that is not finite, or two at the same
    /// place. Three because two positions have one correlation between them and
    /// two parameters to explain it with, which is a ridge rather than a fit.
    pub(crate) fn new(positions: &[f64], shape: Shape) -> Result<Self, &'static str> {
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
            shape,
        })
    }

    pub(crate) fn positions(&self) -> usize {
        self.positions
    }

    pub(crate) fn shape(&self) -> Shape {
        self.shape
    }

    /// The correlation at one separation, which is what a reader should be
    /// given instead of a matrix.
    pub(crate) fn at(shape: Shape, floor: f64, rate: f64, separation: f64) -> f64 {
        floor + (1.0 - floor) * shape.decay(rate, separation)
    }

    /// The whole correlation matrix, with the nugget in it.
    ///
    /// The diagonal stays exactly one: holding back `n` of every off-diagonal
    /// and putting it on the diagonal is `(1 - n) R + n I`, and `R`'s diagonal
    /// is one already.
    pub(crate) fn correlation(&self, floor: f64, rate: f64) -> DMatrix<f64> {
        let size = self.positions;
        DMatrix::from_fn(size, size, |i, j| {
            if i == j {
                1.0
            } else {
                (1.0 - NUGGET) * Self::at(self.shape, floor, rate, self.separation[(i, j)])
            }
        })
    }

    /// `dR / d floor`, which is nought on the diagonal because a separation of
    /// nought gives a correlation of one however the floor moves.
    fn floor_derivative(&self, rate: f64) -> DMatrix<f64> {
        self.separation
            .map(|separation| (1.0 - NUGGET) * (1.0 - self.shape.decay(rate, separation)))
    }

    /// `dR / d rate`.
    fn rate_derivative(&self, floor: f64, rate: f64) -> DMatrix<f64> {
        self.separation.map(|separation| {
            (1.0 - NUGGET) * (1.0 - floor) * self.shape.decay_slope(rate, separation)
        })
    }
}

/// How far outside its own range a floor may land before it is a mistake
/// rather than a rounding error. A part in a hundred million of a correlation is
/// far below anything a fit could tell apart.
const EDGE: f64 = 1e-8;

/// A correlation held at a value, at one separation, for one component.
///
/// **This is how a profile interval is taken on a curve rather than on a
/// parameter.** ADR 0010 records that a floor and a rate are not separately
/// estimable while the correlation they describe is, so an interval belongs on
/// the correlation at a separation that matters and not on either number.
///
/// Holding `corr(d) = v` leaves one free parameter where there were two: the
/// rate moves and the floor follows from it,
///
/// ```text
/// c = (v - u) / (1 - u),   u = decay(rate, d)
/// ```
///
/// so the conditional maximisation searches over the scales and the rate, and
/// nothing else about the fit changes. That is what keeps the constrained fit
/// as cheap as the free one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Held {
    pub separation: f64,
    pub correlation: f64,
}

impl Held {
    /// The floor the constraint implies at this rate, or `None` where the
    /// constraint cannot be met there.
    ///
    /// **The tolerance is not decoration.** At the slowest rate the constraint
    /// allows the floor is exactly nought, and that is where the search starts
    /// and often where it ends. Arithmetic puts it a few parts in ten thousand
    /// billion below nought about half the time, and without the tolerance the
    /// whole constrained fit would be refused on a rounding error.
    pub(crate) fn floor_at(self, shape: Shape, rate: f64) -> Option<f64> {
        let decay = shape.decay(rate, self.separation);
        if !(decay < 1.0) {
            return None;
        }
        let floor = (self.correlation - decay) / (1.0 - decay);
        if !floor.is_finite() || !(-EDGE..=FLOOR_MAX + EDGE).contains(&floor) {
            return None;
        }
        Some(floor.clamp(0.0, FLOOR_MAX))
    }

    /// `d floor / d rate` along the constraint.
    fn floor_slope(self, shape: Shape, rate: f64) -> f64 {
        let decay = shape.decay(rate, self.separation);
        let gap = 1.0 - decay;
        if gap.abs() < 1e-12 {
            return 0.0;
        }
        (self.correlation - 1.0) / (gap * gap) * shape.decay_slope(rate, self.separation)
    }

    /// The rate the constraint implies at a given floor, which is the other
    /// way round from [`Held::floor_at`].
    ///
    /// **The floor is the coordinate to sweep in, not the rate.** The floor runs
    /// over a bounded interval -- nought to the held correlation itself -- while
    /// the rates that go with it run to infinity, so an even grid on the floor
    /// covers the whole family and an even grid on the rate spends most of its
    /// points where nothing happens.
    fn rate_at_floor(self, shape: Shape, floor: f64) -> Option<f64> {
        let gap = 1.0 - floor;
        if gap.abs() < 1e-12 {
            return None;
        }
        shape.rate_reaching((self.correlation - floor) / gap, self.separation)
    }

    /// The largest floor the constraint allows, which is the held correlation
    /// itself, or the floor's own ceiling where that is lower.
    fn highest_floor(self) -> f64 {
        self.correlation.min(FLOOR_MAX)
    }

    /// The rates at which the constraint can be met at all.
    ///
    /// The floor has to stay between nought and its own ceiling, and both ends
    /// of that turn into an end for the rate, because the decay falls as the
    /// rate rises.
    fn rates(self, shape: Shape) -> Option<(f64, f64)> {
        let (v, d) = (self.correlation, self.separation);
        if !(0.0..=1.0).contains(&v) || !(d > 0.0) {
            return None;
        }
        // The floor is nought where the decay equals the held correlation, and
        // any faster decay would need a negative one.
        let lowest = shape
            .rate_reaching(v, d)
            .map_or(RATE_MIN, |rate| rate.clamp(RATE_MIN, RATE_MAX));
        // The floor hits its ceiling where the decay falls to this.
        let ceiling = (v - FLOOR_MAX) / (1.0 - FLOOR_MAX);
        let highest = if ceiling <= 0.0 {
            RATE_MAX
        } else {
            shape
                .rate_reaching(ceiling, d)
                .map_or(RATE_MAX, |rate| rate.clamp(RATE_MIN, RATE_MAX))
        };
        (highest > lowest).then_some((lowest, highest))
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
    maximise_fixing(kernel, statistic, weight, from, None)
}

/// The same maximisation with the scale at one position left where it is.
///
/// **This is one half of holding a variance share.** A share at a position
/// couples the scales of every component there and nothing else, so the fit
/// splits into two conditional maximisations: this one, which moves everything
/// except those scales and so cannot disturb the constraint, and
/// [`maximise_shares`], which moves those scales along it. Together they span
/// the constrained parameter space, which is what makes the pair an ECM rather
/// than a heuristic.
pub(crate) fn maximise_fixing(
    kernel: &Kernel,
    statistic: &DMatrix<f64>,
    weight: f64,
    from: &Shaped,
    fixed: Option<usize>,
) -> Option<Shaped> {
    let size = from.scale.len();
    let rate_at = size + 1;

    // **The search runs on the logarithm of the rate.** A rate is a scale
    // parameter in units the crate does not choose, so a step of a fixed size
    // in it means something different for every caller and nothing at all near
    // nought. Worse, the correlation matrix of a squared exponential is nearly
    // singular at the rates that fit best -- a condition number of 1e8 is
    // ordinary there -- so a linear step overshoots into a covariance that will
    // not factorise, the line search sees an infinite objective, and the whole
    // maximisation returns where it started.
    //
    // Measured: on a Gaussian shape whose best rate was 0.08, the search made
    // no move at all in linear rate, leaving the correlation at 31 ERB out by
    // 0.25. On the logarithm every step is multiplicative and stays inside the
    // bounds by construction.
    // **The scales go on their logarithms for the same reason the rate does.**
    // A scale's only lower bound is nought, and at nought the covariance is
    // singular and the objective is not a number, so a search on the natural
    // scale can step onto a corner of its own box where there is nothing to
    // measure. Measured on the constrained search, which meets that corner far
    // more often: the line search returned code 52 with no movement at all,
    // keeping a point 90 log units short of the answer, and the same fit run
    // with one correlation held beat the free fit by 192 log-likelihood units,
    // which cannot happen at a maximum.
    let into_search = |shaped: &Shaped| -> Vec<f64> {
        let mut par = shaped.packed();
        for scale in par.iter_mut().take(size) {
            *scale = scale.max(SCALE_MIN).ln();
        }
        par[rate_at] = par[rate_at].max(RATE_MIN).ln();
        par
    };
    let out_of_search = |par: &[f64]| -> Vec<f64> {
        let mut out = par.to_vec();
        for scale in out.iter_mut().take(size) {
            *scale = scale.exp().clamp(SCALE_MIN, SCALE_MAX);
        }
        out[rate_at] = out[rate_at].exp().clamp(RATE_MIN, RATE_MAX);
        out
    };

    let start = into_search(from);
    let (mut lower, mut upper) = Shaped::bounds(size);
    for index in 0..size {
        lower[index] = SCALE_MIN.ln();
        upper[index] = SCALE_MAX.ln();
    }
    lower[rate_at] = RATE_MIN.ln();
    upper[rate_at] = RATE_MAX.ln();
    if let Some(at) = fixed {
        // A variable whose two bounds meet is a variable the search cannot
        // move, which is what "fixed" means here. Pinning it to the start
        // rather than to the caller's number keeps this exact even after the
        // logarithm and back.
        if at >= size {
            return None;
        }
        lower[at] = start[at];
        upper[at] = start[at];
    }
    let bounds = Bounds::new(lower, upper).ok()?;

    let value_of = |par: &[f64]| -> f64 {
        objective(kernel, statistic, weight, &out_of_search(par))
            .map_or(INFEASIBLE, |(value, _)| value)
    };
    // Where the objective cannot be evaluated the gradient is differenced from
    // the value rather than reported as nought.
    //
    // **A nought gradient is what a bound-constrained search reads as a
    // stationary point.** Returning one at a shape that could not be evaluated
    // tells the search it has arrived precisely where it has not, and it then
    // stops there and reports success.
    //
    // The log reparameterisation closes the plainest route to such a shape --
    // a scale can no longer reach nought, because the lower bound is now
    // `SCALE_MIN.ln()` rather than nought itself. It does not close the
    // question. A scale at `SCALE_MIN` beside a rate at `RATE_MAX` still makes
    // a kernel the Cholesky can refuse, and any refusal anywhere arrives here
    // as the absence of a gradient.
    //
    // Differencing the sentinel instead gives a large slope pointing back
    // towards the feasible region, which is what the censored and liability
    // models get for free by differencing their objective throughout. Deep
    // inside an infeasible region both sides are the sentinel and the
    // difference degenerates to nought again -- the acceptance test below is
    // what catches that case.
    let gradient_of = |par: &[f64]| -> Vec<f64> {
        let natural = out_of_search(par);
        if let Some((_, mut gradient)) = objective(kernel, statistic, weight, &natural) {
            // The chain rule for a logarithm: multiply by the value.
            for index in 0..size {
                gradient[index] *= natural[index];
            }
            gradient[rate_at] *= natural[rate_at];
            return gradient;
        }
        // Differenced in the search's own coordinates, which are already the
        // logarithms, so no chain rule applies here -- `value_of` does the
        // mapping back itself.
        (0..par.len())
            .map(|index| {
                let step = GRADIENT_STEP * par[index].abs().max(1.0);
                let mut up = par.to_vec();
                let mut down = par.to_vec();
                up[index] += step;
                down[index] -= step;
                (value_of(&up) - value_of(&down)) / (2.0 * step)
            })
            .collect()
    };
    // An infeasible start would otherwise scale the objective by the sentinel,
    // and every genuine reduction would divide to nothing against it -- the
    // search would stop on its first iteration and call it a step.
    let before = value_of(&start);
    let mut control = OptimControl::default_for_dimension(start.len());
    control.maxit = INNER_ITERATIONS;
    control.fnscale = if before < INFEASIBLE {
        before.abs().max(1.0)
    } else {
        1.0
    };
    control.parscale = vec![1.0; start.len()];
    control.factr = 1.0e5;
    control.pgtol = 1e-9;
    control.lmm = start.len().min(10);
    let solution =
        optim_lbfgsb_with_gradient(start.clone(), bounds, value_of, gradient_of, control).ok()?;

    // Keep the result only where it is no worse. A search that ends badly
    // leaves the component where it was, which is a step of nothing and still a
    // step that cannot lower the likelihood.
    //
    // The test is against the sentinel rather than against `is_finite`, which
    // the sentinel always satisfies. Where the start was infeasible too, a
    // finiteness test would have read `INFEASIBLE <= INFEASIBLE` as an
    // improvement and taken the failed point.
    let after = value_of(&solution.par);
    let par = if after < INFEASIBLE && after <= before {
        solution.par
    } else {
        start
    };
    let shaped = Shaped::unpacked(&out_of_search(&par));
    DenseFactor::new(&shaped.covariance(kernel))?;
    Some(shaped)
}

/// How much of the correlation matrix is held back onto its own diagonal
/// before it is factorised.
///
/// **This is a numerical device and not part of the model.** A squared
/// exponential is very smooth, so at the rates that fit best its correlation
/// matrix is nearly singular -- a condition number of 6.5e8 was measured on
/// eight positions -- and the objective becomes a narrow valley: 4.75 log units
/// from the optimum its gradient still spans four orders of magnitude across
/// the parameters. L-BFGS-B's first trial step then overshoots into a
/// covariance that will not factorise, and the line search stops with
/// `ABNORMAL_TERMINATION_IN_LNSRCH` having moved nothing at all. Measured: the
/// Gaussian maximisation returned its own starting point eighty times running.
///
/// Replacing `R` by `(1 - n) R + n I` leaves the diagonal at one, changes an
/// off-diagonal correlation by one part in a hundred thousand, and bounds the
/// condition number by about `T / n`. It is what every Gaussian-process
/// implementation does, for this reason.
///
/// **It is not applied to the correlation a fit reports.** That is the model's
/// own curve; this is what its matrix is factorised as.
const NUGGET: f64 = 1.0e-5;

/// How many rates the starting sweep tries, from where, and by what factor.
///
/// **Geometric and fine, because the objective in the rate is sharp.** An
/// earlier version tried five rates a factor of five apart and it was not
/// enough: on a Gaussian shape whose best rate was 0.08, the sweep evaluated
/// 0.05 at +36,209 and 0.2 at -258 with the optimum at -677 in between, chose
/// 0.2, and the search never walked back down. The rate enters the Gaussian
/// squared, so its objective is sharper than the exponential's and a grid built
/// for one is too coarse for the other. Each point costs a `T` by `T` Cholesky,
/// so a fine grid is nearly free.
const RATE_STARTS: i32 = 22;

/// The smallest and largest scale either search will consider. They are ends of
/// the search and not of the model: a component whose standard deviation at a
/// position is outside a factor of a million either way of the standardised data
/// has nothing to do with the data.
const SCALE_MIN: f64 = 1e-6;
const SCALE_MAX: f64 = 1e6;

/// How many members of the family the constrained search tries before it
/// settles on one. The grid is even in the floor and so spans the whole family.
const SWEEP_POINTS: usize = 32;
const RATE_LOWEST: f64 = 2.0e-3;
const RATE_STEP: f64 = 1.5;

/// The constrained objective and its derivatives, in the coordinates the
/// constrained search uses: the scales, and the logarithm of the rate.
///
/// The floor is not a coordinate here. It follows from the rate through the
/// constraint, so the rate's derivative has to carry the floor's as well, by the
/// chain rule.
#[allow(clippy::too_many_arguments)]
fn value_and_gradient_of(
    par: &[f64],
    size: usize,
    shape: Shape,
    held: Held,
    kernel: &Kernel,
    statistic: &DMatrix<f64>,
    weight: f64,
) -> Option<(f64, Vec<f64>)> {
    let rate = par[size].exp();
    let floor = held.floor_at(shape, rate)?;
    let mut full: Vec<f64> = par[..size].iter().map(|s| s.exp()).collect();
    full.push(floor);
    full.push(rate);
    let (value, gradient) = objective(kernel, statistic, weight, &full)?;
    let mut out: Vec<f64> = (0..size).map(|i| gradient[i] * full[i]).collect();
    // The rate moves the floor as well as itself, so its slope carries both.
    let along = gradient[size + 1] + gradient[size] * held.floor_slope(shape, rate);
    out.push(along * rate);
    Some((value, out))
}

/// A shape that satisfies a constraint, as near the given one as the family
/// allows.
///
/// The scales are kept. The rate is moved to the nearest one the constraint
/// allows, and the floor follows from it.
pub(crate) fn started_holding(kernel: &Kernel, from: &Shaped, held: Held) -> Option<Shaped> {
    let shape = kernel.shape();
    let (lowest, highest) = held.rates(shape)?;
    let rate = from.rate.clamp(lowest, highest);
    Some(Shaped {
        scale: from.scale.clone(),
        floor: held.floor_at(shape, rate)?,
        rate,
    })
}

/// The same maximisation with one correlation held.
///
/// Searches over the scales and the rate, with the floor following from the
/// constraint. Returns `None` where the constraint cannot be met by any rate --
/// which happens when a correlation is asked for that no member of the family
/// can produce at that separation.
pub(crate) fn maximise_holding(
    kernel: &Kernel,
    statistic: &DMatrix<f64>,
    weight: f64,
    from: &Shaped,
    held: Held,
) -> Option<Shaped> {
    let shape = kernel.shape();
    let (lowest, highest) = held.rates(shape)?;
    let size = from.scale.len();

    let value_and_gradient = |par: &[f64]| -> Option<(f64, Vec<f64>)> {
        value_and_gradient_of(par, size, shape, held, kernel, statistic, weight)
    };

    // **The scales are searched on their logarithms, and so is the rate.** Both
    // are scale parameters, and on their natural scale their only lower bound is
    // nought, where the covariance is singular and the objective is not a
    // number. The search would step onto that corner, find no value there, and
    // return where it started -- code 52, `ABNORMAL_TERMINATION_IN_LNSRCH`, from
    // a start whose objective was 67.562 against a true constrained maximum at
    // -21.917. On the logarithm every step is multiplicative and no step can
    // reach nought.
    let logged_scale: Vec<f64> = from.scale.iter().map(|s| s.max(SCALE_MIN).ln()).collect();
    let at = |rate: f64| -> f64 {
        let mut candidate = logged_scale.clone();
        candidate.push(rate.ln());
        value_and_gradient_of(&candidate, size, shape, held, kernel, statistic, weight)
            .map_or(f64::INFINITY, |(value, _)| value)
    };

    // **The search cannot start on the rate's own bound either.** At the slowest
    // rate the constraint allows, the floor is nought, and nought is where the
    // constrained answer often belongs -- so the obvious start is exactly on a
    // corner of the box, where the same thing happens: code 52 again, an
    // objective of 68.190 kept against 67.787 a short step away.
    //
    // So sweep the family first and hand the search the best of it, well inside.
    // The sweep runs on the floor, because the floor is bounded and the rate is
    // not; see [`Held::rate_at_floor`].
    //
    // **The shape it was given is one of the candidates**, projected onto the
    // constraint. Without it the sweep could hand back a worse point than the one
    // it started from, and a conditional maximisation that goes downhill is not a
    // conditional maximisation -- expectation-maximisation would stop climbing
    // the likelihood, which is the one property the whole fit rests on.
    let given = from.rate.clamp(lowest, highest);
    let mut best = (at(given), given.ln());
    let highest_floor = held.highest_floor();
    for index in 0..SWEEP_POINTS {
        // Half a step in from each end, so no candidate is a corner.
        let floor = highest_floor * (index as f64 + 0.5) / SWEEP_POINTS as f64;
        let Some(rate) = held.rate_at_floor(shape, floor) else {
            continue;
        };
        if !(lowest..=highest).contains(&rate) {
            continue;
        }
        let value = at(rate);
        if value < best.0 {
            best = (value, rate.ln());
        }
    }
    let mut start = logged_scale;
    start.push(best.1);
    let mut lower = vec![SCALE_MIN.ln(); size];
    lower.push(lowest.ln());
    let mut upper = vec![SCALE_MAX.ln(); size];
    upper.push(highest.ln());
    let bounds = Bounds::new(lower, upper).ok()?;

    let value_of = |par: &[f64]| -> f64 { value_and_gradient(par).map_or(1e30, |(v, _)| v) };
    let gradient_of = |par: &[f64]| -> Vec<f64> {
        value_and_gradient(par).map_or_else(|| vec![0.0; par.len()], |(_, g)| g)
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
    let before = value_of(&start);
    let after = value_of(&solution.par);
    let par = if after.is_finite() && after <= before {
        solution.par
    } else {
        start
    };
    let rate = par[size].exp().clamp(lowest, highest);
    let shaped = Shaped {
        scale: par[..size]
            .iter()
            .map(|s| s.exp().clamp(SCALE_MIN, SCALE_MAX))
            .collect(),
        floor: held.floor_at(shape, rate)?,
        rate,
    };
    DenseFactor::new(&shaped.covariance(kernel))?;
    Some(shaped)
}

/// One position's scales, moved together along a held variance share.
///
/// **This is the other half of holding a share**, and the reason the joint
/// model's heritability needs a different machinery from its correlation.
/// A correlation belongs to one component, so holding it leaves every other
/// component's maximisation untouched. A share is a ratio *between* components:
///
/// ```text
/// share_j(t) = scale_j(t)^2 / sum_k scale_k(t)^2
/// ```
///
/// so holding it ties three maximisations that the method otherwise keeps
/// separate. What is done here is to search over the scales of the components
/// that are *not* held, at that one position, and let the held one follow:
///
/// ```text
/// scale_j(t) = sqrt( v / (1 - v) * sum_{k != j} scale_k(t)^2 )
/// ```
///
/// which satisfies the constraint by construction. Everything else about every
/// component is left where [`maximise_fixing`] put it.
///
/// The scales are searched on their logarithms for the reason given in
/// [`maximise`]: nought is where a covariance stops being one.
///
/// Returns `None` where the share cannot be met, or where any component's
/// covariance stops factorising along the way.
pub(crate) fn maximise_shares(
    kernel: &Kernel,
    statistics: &[&DMatrix<f64>],
    weights: &[f64],
    shapes: &[Shaped],
    position: usize,
    held: usize,
    share: f64,
) -> Option<Vec<Shaped>> {
    let count = shapes.len();
    if statistics.len() != count || weights.len() != count || held >= count {
        return None;
    }
    if !(0.0..1.0).contains(&share) {
        return None;
    }
    let ratio = share / (1.0 - share);
    let free: Vec<usize> = (0..count).filter(|k| *k != held).collect();
    if free.is_empty() {
        return None;
    }

    // The held scale, and how it moves with each free one.
    let derived = |logged: &[f64]| -> Option<f64> {
        let sum: f64 = logged.iter().map(|x| (2.0 * x).exp()).sum();
        let scale = (ratio * sum).sqrt();
        (scale.is_finite() && (SCALE_MIN..=SCALE_MAX).contains(&scale)).then_some(scale)
    };

    let value_and_gradient = |logged: &[f64]| -> Option<(f64, Vec<f64>)> {
        let held_scale = derived(logged)?;
        let mut total = 0.0;
        let mut own = vec![0.0; free.len()];
        for (slot, k) in free.iter().enumerate() {
            let mut packed = shapes[*k].packed();
            packed[position] = logged[slot].exp();
            let (value, gradient) = objective(kernel, statistics[*k], weights[*k], &packed)?;
            total += value;
            own[slot] = gradient[position];
        }
        let held_slope = {
            let mut packed = shapes[held].packed();
            packed[position] = held_scale;
            let (value, gradient) = objective(kernel, statistics[held], weights[held], &packed)?;
            total += value;
            gradient[position]
        };
        // The chain rule twice over: the logarithm, and the held scale that
        // every free one drags with it.
        let out = free
            .iter()
            .enumerate()
            .map(|(slot, _)| {
                let scale = logged[slot].exp();
                (own[slot] + held_slope * ratio * scale / held_scale) * scale
            })
            .collect();
        Some((total, out))
    };

    let start: Vec<f64> = free
        .iter()
        .map(|k| shapes[*k].scale[position].max(SCALE_MIN).ln())
        .collect();
    derived(&start)?;
    let bounds = Bounds::new(
        vec![SCALE_MIN.ln(); free.len()],
        vec![SCALE_MAX.ln(); free.len()],
    )
    .ok()?;

    let value_of = |par: &[f64]| -> f64 { value_and_gradient(par).map_or(1e30, |(v, _)| v) };
    let gradient_of = |par: &[f64]| -> Vec<f64> {
        value_and_gradient(par).map_or_else(|| vec![0.0; par.len()], |(_, g)| g)
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
    let before = value_of(&start);
    let after = value_of(&solution.par);
    let par = if after.is_finite() && after <= before {
        solution.par
    } else {
        start
    };

    let held_scale = derived(&par)?;
    let mut out = shapes.to_vec();
    for (slot, k) in free.iter().enumerate() {
        out[*k].scale[position] = par[slot].exp().clamp(SCALE_MIN, SCALE_MAX);
    }
    out[held].scale[position] = held_scale;
    for shaped in &out {
        DenseFactor::new(&shaped.covariance(kernel))?;
    }
    Some(out)
}

/// Shapes that satisfy a held share, as near the given ones as the constraint
/// allows: every other component is left alone and the held one follows.
pub(crate) fn started_holding_share(
    kernel: &Kernel,
    shapes: &[Shaped],
    position: usize,
    held: usize,
    share: f64,
) -> Option<Vec<Shaped>> {
    if held >= shapes.len() || !(0.0..1.0).contains(&share) {
        return None;
    }
    let sum: f64 = shapes
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != held)
        .map(|(_, shaped)| {
            let scale = *shaped.scale.get(position).unwrap_or(&0.0);
            scale * scale
        })
        .sum();
    let scale = (share / (1.0 - share) * sum).sqrt();
    if !scale.is_finite() {
        return None;
    }
    let mut out = shapes.to_vec();
    *out.get_mut(held)?.scale.get_mut(position)? = scale.clamp(SCALE_MIN, SCALE_MAX);
    DenseFactor::new(&out[held].covariance(kernel))?;
    Some(out)
}

/// A starting shape for a component, from a covariance that is not shaped.
///
/// The scales are read straight off the diagonal. The floor and the rate come
/// from a sweep rather than from a guess, because a rate is the one parameter
/// here whose scale depends on units the crate knows nothing about, and a
/// search started at the wrong order of magnitude in it can sit there.
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
    for floor in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75] {
        for step in 0..RATE_STARTS {
            let rate = RATE_LOWEST * RATE_STEP.powi(step);
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
    use super::{
        Held, Kernel, NUGGET, Shape, Shaped, maximise, maximise_holding, nearest, objective,
        started_holding,
    };
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

    /// **The constraint has to hold exactly, or the profile is a profile of
    /// something else.** A held correlation that came out near but not at the
    /// value asked for would move the whole likelihood curve, and every
    /// interval taken from it would be wrong by an amount nobody could see.
    /// **The constrained search must not depend on where it started, and it
    /// did.** With the scales searched on their natural scale, a start whose
    /// scales were already right ended on the corner of the box where a scale
    /// is nought and the covariance is singular; the line search returned code
    /// 52 with no movement, keeping an objective of 67.562 where the true
    /// constrained maximum was -21.917. Starts whose scales were half or double
    /// found the right answer. A profile made of the first kind of point would
    /// sit far below the likelihood and every interval from it would be too
    /// narrow.
    #[test]
    fn the_constrained_search_lands_in_the_same_place_from_anywhere() {
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
        let truth = Shaped {
            scale: vec![1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0],
            floor: 0.25,
            rate: 0.08,
        };
        let statistic = statistic(&kernel, &truth, 300.0);
        let held = Held {
            separation: 6.0,
            correlation: 0.5,
        };
        let mut reached = Vec::new();
        for &(factor, floor, rate) in &[
            (1.0, 0.05, 0.13),
            (2.0, 0.30, 0.02),
            (0.5, 0.00, 0.90),
            (5.0, 0.60, 0.30),
        ] {
            let from = Shaped {
                scale: truth.scale.iter().map(|s| s * factor).collect(),
                floor,
                rate,
            };
            let got = maximise_holding(&kernel, &statistic, 300.0, &from, held)
                .expect("the constraint should be reachable from anywhere");
            reached.push(
                objective(&kernel, &statistic, 300.0, &got.packed())
                    .expect("the objective should be finite")
                    .0,
            );
        }
        let best = reached.iter().copied().fold(f64::INFINITY, f64::min);
        for (index, value) in reached.iter().enumerate() {
            assert!(
                value - best < 1e-4,
                "start {index} ended at {value}, against {best} elsewhere"
            );
        }
    }

    /// A profile has one peak, at the free answer, and falls away on both
    /// sides. **If it did not, bisecting outwards from the free answer would
    /// find the wrong end**, because bisection assumes the likelihood crosses
    /// the threshold once.
    #[test]
    fn the_profile_has_one_peak_and_it_is_the_free_answer() {
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
        let truth = Shaped {
            scale: vec![1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0],
            floor: 0.25,
            rate: 0.08,
        };
        let statistic = statistic(&kernel, &truth, 300.0);
        let separation = 6.0;
        let at_truth = Kernel::at(Shape::Exponential, truth.floor, truth.rate, separation);

        let mut at = Vec::new();
        let mut grid = vec![0.2, 0.35, 0.5, 0.65];
        grid.push(at_truth);
        grid.extend([0.8, 0.9, 0.95]);
        for &value in &grid {
            let got = maximise_holding(
                &kernel,
                &statistic,
                300.0,
                &truth,
                Held {
                    separation,
                    correlation: value,
                },
            )
            .expect("every point of the grid should be reachable");
            at.push(
                objective(&kernel, &statistic, 300.0, &got.packed())
                    .expect("the objective should be finite")
                    .0,
            );
        }
        let peak = grid.iter().position(|v| *v == at_truth).expect("the truth");
        for index in 1..at.len() {
            let falling = index <= peak;
            assert_eq!(
                at[index] < at[index - 1],
                falling,
                "the profile goes the wrong way between {} and {}: {} then {}",
                grid[index - 1],
                grid[index],
                at[index - 1],
                at[index]
            );
        }
    }

    #[test]
    fn holding_a_correlation_holds_it_exactly() {
        for shape in [Shape::Exponential, Shape::Gaussian] {
            let kernel = Kernel::new(&line(), shape).expect("the line should be a line");
            let truth = Shaped {
                scale: vec![1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0],
                floor: 0.25,
                rate: 0.08,
            };
            let statistic = statistic(&kernel, &truth, 300.0);
            for &(separation, wanted) in &[(3.0, 0.4), (3.0, 0.8), (10.0, 0.2)] {
                let held = Held {
                    separation,
                    correlation: wanted,
                };
                let got = maximise_holding(&kernel, &statistic, 300.0, &truth, held)
                    .expect("the constraint should be reachable");
                let at = Kernel::at(shape, got.floor, got.rate, separation);
                // The nugget pulls every off-diagonal a part in a hundred
                // thousand towards the diagonal, so the curve is what is held
                // and the matrix follows it to within that.
                assert!(
                    (at - wanted).abs() < 1e-9,
                    "{shape:?} at {separation}: held {wanted}, got {at}"
                );
            }
        }
    }

    /// The constrained maximum cannot beat the free one, and at the free
    /// answer's own correlation it must very nearly reach it. Together those
    /// say the constrained search is searching the right smaller space.
    #[test]
    fn the_constrained_maximum_touches_the_free_one_at_its_own_correlation() {
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
        let truth = Shaped {
            scale: vec![1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0],
            floor: 0.25,
            rate: 0.08,
        };
        let statistic = statistic(&kernel, &truth, 300.0);
        // **`maximise` is a conditional maximisation and stops after sixty
        // inner iterations**, because inside the fit it is called again next
        // time round. A profile compares maxima, so the free one has to be run
        // to a fixed point before it is a fair comparison.
        let mut free =
            nearest(&kernel, &(&statistic / 300.0), 300.0).expect("a start should be findable");
        for _ in 0..40 {
            free = maximise(&kernel, &statistic, 300.0, &free).expect("the free maximisation");
        }
        let separation = 6.0;
        let at_free = Kernel::at(Shape::Exponential, free.floor, free.rate, separation);

        let objective_of = |shaped: &Shaped| {
            let mut packed = shaped.scale.clone();
            packed.push(shaped.floor);
            packed.push(shaped.rate);
            objective(&kernel, &statistic, 300.0, &packed)
                .expect("the objective should be finite")
                .0
        };
        let best = objective_of(&free);
        let held = Held {
            separation,
            correlation: at_free,
        };
        let same = maximise_holding(&kernel, &statistic, 300.0, &free, held)
            .expect("the free answer's own correlation should be reachable");
        let cost = objective_of(&same) - best;
        assert!(
            (-1e-6..1e-3).contains(&cost),
            "holding the free answer's own correlation cost {cost}"
        );

        // Away from it the constrained maximum is worse, which is what makes a
        // profile a profile.
        let elsewhere = Held {
            separation,
            correlation: (at_free - 0.3).max(0.01),
        };
        let worse = maximise_holding(&kernel, &statistic, 300.0, &free, elsewhere)
            .expect("a correlation a third away should still be reachable");
        assert!(
            objective_of(&worse) > best + 1.0,
            "moving the correlation 0.3 cost only {}",
            objective_of(&worse) - best
        );
    }

    /// A correlation the family cannot express is refused rather than
    /// approximated, because a silently nearest answer would be a profile at a
    /// value nobody asked about.
    #[test]
    fn a_starting_shape_satisfies_the_constraint_or_says_it_cannot() {
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
        let from = Shaped {
            scale: vec![1.0; 8],
            floor: 0.25,
            rate: 0.08,
        };
        let started = started_holding(
            &kernel,
            &from,
            Held {
                separation: 6.0,
                correlation: 0.5,
            },
        )
        .expect("half at six should be reachable");
        assert!(
            (Kernel::at(Shape::Exponential, started.floor, started.rate, 6.0) - 0.5).abs() < 1e-12
        );
        assert_eq!(started.scale, from.scale, "the scales should be left alone");

        // Above the floor's own ceiling there is no member of the family at all.
        assert!(
            started_holding(
                &kernel,
                &from,
                Held {
                    separation: 6.0,
                    correlation: 1.0,
                }
            )
            .is_none(),
            "a correlation of one should be refused"
        );
    }

    #[test]
    fn the_kernel_is_a_correlation_matrix_between_its_two_extremes() {
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
        for &(floor, rate) in &[(0.0, 0.05), (0.3, 0.1), (0.6, 2.0)] {
            let r = kernel.correlation(floor, rate);
            for i in 0..r.nrows() {
                assert!((r[(i, i)] - 1.0).abs() < 1e-12, "diagonal at {i}");
                for j in 0..r.ncols() {
                    assert!((r[(i, j)] - r[(j, i)]).abs() < 1e-15);
                    // The nugget holds a part in a hundred thousand of every
                    // off-diagonal back onto the diagonal, so the floor is a
                    // floor to within it and not exactly.
                    assert!(
                        r[(i, j)] >= floor * (1.0 - NUGGET) - 1e-12 && r[(i, j)] <= 1.0 + 1e-12,
                        "entry ({i}, {j}) is {} against a floor of {floor}",
                        r[(i, j)]
                    );
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
        assert!((factor[(0, 1)] - 0.35 * (1.0 - NUGGET)).abs() < 1e-9);
        let distance = kernel.correlation(0.0, 0.1);
        let wanted = (-0.1_f64 * 31.0).exp() * (1.0 - NUGGET);
        assert!((distance[(0, 7)] - wanted).abs() < 1e-12);
    }

    /// **Both shapes, because a wrong derivative in one would not show in the
    /// other.** The rate enters squared in the Gaussian, so its slope has a
    /// factor the exponential's does not.
    #[test]
    fn the_gaussian_gradient_is_the_slope_of_its_objective() {
        let kernel = Kernel::new(&line(), Shape::Gaussian).expect("the line should be a line");
        let truth = Shaped {
            scale: vec![1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0],
            floor: 0.2,
            rate: 0.06,
        };
        let weight = 40.0;
        let statistic = truth.covariance(&kernel) * weight;
        // **A well conditioned point on purpose.** The squared exponential is
        // very smooth, so at a low rate its correlation matrix is nearly
        // singular -- neighbouring positions correlate at 0.99 -- and the
        // objective and both gradients run to 1e7. There the central difference
        // is what loses digits, not the derivative, so a test at such a point
        // would measure the wrong thing.
        let at = Shaped {
            scale: vec![0.8, 1.5, 1.1, 1.0, 1.4, 1.0, 0.9, 1.2],
            floor: 0.35,
            rate: 0.25,
        };
        let par = at.packed();
        let (value, gradient) =
            objective(&kernel, &statistic, weight, &par).expect("it should evaluate");
        assert!(
            value.abs() < 1.0e4,
            "the check point should be well conditioned, and this one gives {value}"
        );
        for index in 0..par.len() {
            let step = 1e-6 * par[index].abs().max(1e-3);
            let mut up = par.clone();
            let mut down = par.clone();
            up[index] += step;
            down[index] -= step;
            let high = objective(&kernel, &statistic, weight, &up)
                .expect("it should evaluate")
                .0;
            let low = objective(&kernel, &statistic, weight, &down)
                .expect("it should evaluate")
                .0;
            let numerical = (high - low) / (2.0 * step);
            assert!(
                (gradient[index] - numerical).abs() < 1e-5 * numerical.abs().max(1.0),
                "parameter {index}: analytic {} against numerical {numerical}",
                gradient[index]
            );
        }
    }

    /// The Gaussian falls slowly at first and then far faster, which is the
    /// whole reason to have it.
    #[test]
    fn the_gaussian_has_the_lighter_tail() {
        let near = 2.0;
        let far = 25.0;
        // Rates chosen so both pass through the same point near the origin.
        let exponential = Kernel::at(Shape::Exponential, 0.0, 0.05, near);
        let rate = (-(exponential.ln())).sqrt() / near;
        let gaussian = Kernel::at(Shape::Gaussian, 0.0, rate, near);
        assert!(
            (exponential - gaussian).abs() < 1e-9,
            "same near the origin"
        );
        let far_exponential = Kernel::at(Shape::Exponential, 0.0, 0.05, far);
        let far_gaussian = Kernel::at(Shape::Gaussian, 0.0, rate, far);
        assert!(
            far_gaussian < far_exponential * 0.5,
            "at {far} apart the Gaussian is {far_gaussian} and the exponential \
             {far_exponential}; the point of it is that the first is far smaller"
        );
        for shape in [Shape::Exponential, Shape::Gaussian] {
            let kernel = Kernel::new(&line(), shape).expect("a line");
            assert!(
                kernel.correlation(0.3, 0.08).cholesky().is_some(),
                "{} should be positive definite",
                shape.name()
            );
        }
    }

    #[test]
    fn the_gradient_is_the_slope_of_the_objective() {
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
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
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
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

    /// The same for the Gaussian. **If this failed, a comparison between the
    /// two shapes on real data would be measuring the arithmetic and not the
    /// data.**
    #[test]
    fn the_gaussian_maximisation_finds_its_own_shape_too() {
        let kernel = Kernel::new(&line(), Shape::Gaussian).expect("a line");
        let truth = Shaped {
            scale: vec![1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0],
            floor: 0.25,
            rate: 0.08,
        };
        let weight = 40.0;
        let statistic = truth.covariance(&kernel) * weight;
        let mut at =
            nearest(&kernel, &(&statistic / weight), weight).expect("a start should be found");
        for _ in 0..80 {
            at = maximise(&kernel, &statistic, weight, &at).expect("the search runs");
        }
        for index in 0..truth.scale.len() {
            assert!(
                (at.scale[index] - truth.scale[index]).abs() < 1e-3,
                "scale {index}: {} against {}",
                at.scale[index],
                truth.scale[index]
            );
        }
        assert!((at.floor - truth.floor).abs() < 1e-3, "floor {}", at.floor);
        assert!((at.rate - truth.rate).abs() < 1e-3, "rate {}", at.rate);
    }

    #[test]
    fn the_maximisation_never_goes_downhill() {
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
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

    /// **The same, held.** Expectation-maximisation climbs the likelihood
    /// because every conditional maximisation does. A constrained one that went
    /// downhill would take that away, and a fit that no longer climbs is a fit
    /// that stops wherever it happens to be -- which for a profile means a point
    /// below the likelihood and an interval too narrow.
    #[test]
    fn the_constrained_maximisation_never_goes_downhill_either() {
        let kernel = Kernel::new(&line(), Shape::Exponential).expect("the line should be a line");
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
        let held = Held {
            separation: 6.0,
            correlation: 0.45,
        };
        // Started well away from the constraint on purpose, because that is the
        // case the guard is for: the sweep must not hand back worse ground than
        // it was given.
        let mut at = started_holding(
            &kernel,
            &Shaped {
                scale: vec![1.0; 8],
                floor: 0.5,
                rate: 1.0,
            },
            held,
        )
        .expect("the constraint should be reachable");
        let mut previous = objective(&kernel, &statistic, weight, &at.packed())
            .expect("the objective should evaluate")
            .0;
        for step in 0..25 {
            at = maximise_holding(&kernel, &statistic, weight, &at, held)
                .expect("the search should run");
            assert!(
                (Kernel::at(Shape::Exponential, at.floor, at.rate, held.separation)
                    - held.correlation)
                    .abs()
                    < 1e-9,
                "step {step} left the constraint"
            );
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
            Kernel::new(&[0.0, 1.0], Shape::Exponential).err(),
            Some("REPEATED_KERNEL_NEEDS_THREE_POSITIONS")
        );
        assert_eq!(
            Kernel::new(&[0.0, 1.0, 1.0], Shape::Exponential).err(),
            Some("REPEATED_KERNEL_POSITIONS_COINCIDE")
        );
        assert_eq!(
            Kernel::new(&[0.0, 1.0, f64::NAN], Shape::Exponential).err(),
            Some("REPEATED_KERNEL_POSITION_NOT_FINITE")
        );
    }
}
