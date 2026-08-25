//! One convergence rule for every model family.
//!
//! # Why this is in one place
//!
//! It was in seven. Each family tested its projected gradient against its own
//! number -- `1e-7` in the component and two-trait models, `1e-6` in the
//! spatial and the two gene-by-environment ones, `1e-5` in liability and latent
//! mediation -- and none of the seven recorded a reason. They were never
//! compared with one another because nothing ever put them side by side.
//!
//! The tidy explanation was that a family needs a looser test where quadrature
//! sits under its likelihood, and the families line up that way at a glance.
//! It is wrong. Measured on the same simulated people, liability has the
//! tightest gradients in the package and the loosest threshold by a factor of a
//! thousand. On real GOBS traits -- 63 fits, 1,300 to 1,900 people each --
//! every family reaches about `1e-08` when allowed to, the reachable floors
//! spanning `7.68e-09` to `2.69e-08`. No family is meaningfully harder to
//! converge than another, so no family needs its own number.
//!
//! # Why this number
//!
//! `1e-6` passes every fit ever measured here: all 63 GOBS fits, and all eight
//! red deer fits once [`polish`] has run. The worst case anywhere is the deer's
//! rut home range with overlap at `1.578e-07`, which cannot be improved because
//! the line search runs out of double precision, leaving a factor of six.
//!
//! `1e-7` was the alternative and it buys nothing. Starving the search on
//! purpose across five families produced 165 fits, 92 of them genuinely short
//! of the optimum, and **one** of the 165 landed between `1e-7` and `1e-6`;
//! its estimate was wrong by a millionth of a variance share. Fits land either
//! well below `1e-7` or above `1e-6` and almost never between, so the choice
//! decides about one fit in a hundred and sixty-five -- while `1e-7` fails the
//! deer fit outright.
//!
//! # What the number means
//!
//! The same starving measurement calibrated it. The worst error in a variance
//! share runs about ten times the reported gradient, and that holds across six
//! orders of magnitude. So a fit reporting `1e-6` is right to about `1e-5` in
//! any share it quotes, and one reporting `1e-3` may be wrong in the second
//! decimal place. Every starved fit above `1e-5` was caught.
//!
//! # What this does not cover
//!
//! The latent mediation model keeps its own `1e-5`, deliberately. There the
//! number is not a reported flag but a rule for accepting a start: a candidate
//! whose gradient is above it is discarded rather than reported, so tightening
//! it makes fits refuse rather than makes them honest. That is a different
//! change with a different risk, and it belongs with that model's own
//! calibration rather than here.
//!
//! The workings are in `.scratch/asterism-convergence`.

use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

/// A fit is converged when its scaled projected gradient is below this.
pub(crate) const TOLERANCE: f64 = 1e-6;

/// Require a fit that will feed an inferential or predictive result to have
/// passed the shared convergence rule.
///
/// Fit records themselves remain available when this fails: they are useful
/// diagnostics. What this gate prevents is turning such a diagnostic candidate
/// into a test, interval, or prediction. The caller supplies its family- and
/// stage-specific stable refusal code so the public API says which required fit
/// failed.
pub(crate) fn require(converged: bool, refusal: &'static str) -> Result<(), &'static str> {
    if converged { Ok(()) } else { Err(refusal) }
}

/// How much further [`polish`] may search. It never runs where the gradient
/// test already passes, so this is spent only on fits that would otherwise be
/// reported as failures.
const EXTRA_ITERATIONS: usize = 200;

/// A second search that was worth keeping.
pub(crate) struct Polished {
    pub par: Vec<f64>,
    pub negative_loglik: f64,
    pub scaled_gradient: f64,
    pub stop_code: i32,
    pub stop_message: String,
}

/// Search again from a point the gradient test rejected, with the objective
/// tolerance switched off.
///
/// # Why this exists
///
/// Two criteria were in play and nothing reconciled them. The search stops on
/// `factr`, a relative reduction in the objective; the flag is decided
/// afterwards on the projected gradient. Down a long flat valley the objective
/// settles well before the gradient does, so a fit lands on the optimum and is
/// reported as a failure. Three of the eight red deer fits did exactly that
/// while agreeing with a published table to 0.006.
///
/// Every one of those eight fits stops with
/// `REL_REDUCTION_OF_F <= FACTR*EPSMCH` and not one on `pgtol`, so the gradient
/// the flag tests is never a criterion the search pursues. Taking `factr` out
/// leaves `pgtol`, which is a relative-gradient test at `1e-9` and so far
/// tighter than the flag asks for.
///
/// # Why only here
///
/// Running every fit with `factr` off costs 98 per cent more objective
/// evaluations across the deer, each one a dense covariance factorisation, and
/// buys nothing on three of the eight -- the estimates come back identical.
/// Firing only where the flag already reports failure leaves every passing fit
/// untouched to the last bit, so no calibration moves.
///
/// # Why the result is checked rather than trusted
///
/// All three polished deer fits end with `ERROR: ABNORMAL_TERMINATION_IN_LNSRCH`
/// -- L-BFGS-B saying its line search can make no further progress in double
/// precision -- and none reaches `pgtol`. Two of the three then pass the
/// gradient test and the third improves threefold, so the abnormal ending is
/// the shape of the likelihood rather than a fault. Keeping the point only when
/// it is no worse on the objective **and** strictly better on the gradient is
/// what makes taking a result from an errored search safe: it either improves
/// or it is discarded.
///
/// `reading` is the model's own evaluation at a candidate: its negative
/// log-likelihood there and its own projected-gradient reading, or `None` where
/// the candidate cannot be evaluated. The projection differs by family --
/// which parameters have bounds to rest on, and which are free to be negative
/// -- so it stays with the model rather than being guessed at here.
// Eight, where clippy wants seven. Bundling them into a struct would move the
// same eight values one line further away from the call site and make each
// caller name them twice, which is worse to read rather than better.
#[allow(clippy::too_many_arguments)]
pub(crate) fn polish<F, G, R>(
    par: &[f64],
    negative_loglik: f64,
    scaled_gradient: f64,
    lower: &[f64],
    upper: &[f64],
    mut value_of: F,
    gradient_of: G,
    reading: R,
) -> Option<Polished>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
    R: Fn(&[f64]) -> Option<(f64, f64)>,
{
    let count = par.len();
    let bounds = Bounds::new(lower.to_vec(), upper.to_vec()).ok()?;
    let mut control = OptimControl::default_for_dimension(count);
    control.maxit = EXTRA_ITERATIONS;
    control.fnscale = value_of(par).abs().max(1.0);
    control.parscale = vec![1.0; count];
    control.factr = 0.0;
    control.pgtol = 1e-9;
    control.lmm = count.min(10);
    let again =
        optim_lbfgsb_with_gradient(par.to_vec(), bounds, value_of, gradient_of, control).ok()?;
    let (negative, gradient) = reading(&again.par)?;
    if !negative.is_finite() || negative > negative_loglik || gradient >= scaled_gradient {
        return None;
    }
    Some(Polished {
        par: again.par,
        negative_loglik: negative,
        scaled_gradient: gradient,
        stop_code: again.convergence,
        stop_message: again.message,
    })
}

#[cfg(test)]
mod tests {
    use super::require;

    /// The common gate never lets a false convergence flag through.
    #[test]
    fn required_fits_must_have_converged() {
        assert_eq!(require(true, "FAMILY_FIT_NOT_CONVERGED"), Ok(()));
        assert_eq!(
            require(false, "FAMILY_FIT_NOT_CONVERGED"),
            Err("FAMILY_FIT_NOT_CONVERGED")
        );
    }
}
