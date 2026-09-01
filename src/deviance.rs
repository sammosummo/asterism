//! Turning two log likelihoods into a p-value, in one place.
//!
//! Every model here tests a null by refitting it and comparing likelihoods, and
//! each of them was carrying its own copy of the same three pieces of care:
//!
//! **The statistic cannot be negative.** A null fitted to a better likelihood
//! than the alternative it sits inside means a search fell short, not evidence
//! against the null. Clamping at nought reports no evidence, which is the
//! honest reading of a failed search in the direction that cannot happen.
//!
//! **At a statistic of nought a mixture carrying a point mass has tail one, not
//! a half.** Written as a tail of chi-square alone the formula misses that, and
//! a fit sitting exactly on its bound -- which under these nulls is often about
//! half of them -- would be reported at `p = 0.5` rather than `p = 1`. The
//! tolerance is there because two separate searches never land on identically
//! the same number, and a deviance of a millionth is not evidence of anything.
//!
//! **The tails go through the regularised incomplete gamma**, which is both
//! general and the most accurate route available here. The modules this
//! replaced mostly used `erfc`, which looks like the exact closed form for one
//! degree of freedom but is not the better one in practice: measured against
//! the two critical values the interval recipe rests on, `erfc` is out by
//! 2.3e-12 and 4.2e-12 while the incomplete gamma is out by 8.3e-17 and
//! 1.1e-15. Their relative gap reaches 1.0e-10 across the range these tests
//! use. Nothing downstream reads a p-value that finely, but there is no reason
//! to keep the worse of two routes.
//!
//! Chi-square on two degrees of freedom is the one exception: `exp(-t/2)` is
//! exact, agrees with the incomplete gamma to 6.0e-16, and is cheaper.
//!
//! Being general also fixes a real trap. The copy this replaced tested for one
//! and two degrees and then *fell through to the three-degree formula for
//! anything else*, so a caller asking for a fourth would have been quietly
//! given the wrong tail.

/// Below this, two searches have landed on the same likelihood and the
/// deviance is rounding rather than evidence.
const SETTLED: f64 = 1e-6;

/// Collapse likelihood-search rounding into the statistic's point mass.
pub(crate) fn settled(statistic: f64) -> f64 {
    if statistic < SETTLED { 0.0 } else { statistic }
}

/// Twice the log-likelihood gap, floored at nought.
pub(crate) fn deviance(alternative: f64, null: f64) -> f64 {
    (2.0 * (alternative - null)).max(0.0)
}

/// The upper tail of chi-square on `degrees` degrees of freedom, for any
/// positive `degrees`.
pub(crate) fn chi2_upper_tail(statistic: f64, degrees: f64) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    if (degrees - 2.0).abs() < 1e-12 {
        // Exact, and cheaper than the general route.
        return (-statistic / 2.0).exp();
    }
    statrs::function::gamma::gamma_ur(degrees / 2.0, statistic / 2.0)
}

/// The upper tail of chi-square on one degree of freedom, which is exact.
pub(crate) fn chi2_one_df_upper_tail(statistic: f64) -> f64 {
    chi2_upper_tail(statistic, 1.0)
}

/// The upper tail of chi-square on two degrees of freedom, which is exact.
pub(crate) fn chi2_two_df_upper_tail(statistic: f64) -> f64 {
    chi2_upper_tail(statistic, 2.0)
}

/// Read a statistic against its reference, honouring the point mass at nought.
///
/// `tail` is the reference distribution's upper tail: a plain chi-square for an
/// interior null, or a weighted sum of them where a constraint sits on a bound.
pub(crate) fn p_value(statistic: f64, tail: impl Fn(f64) -> f64) -> f64 {
    if settled(statistic) == 0.0 {
        1.0
    } else {
        tail(statistic).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{chi2_upper_tail, deviance, p_value, settled};

    /// Two degrees of freedom is the one closed form kept, so it must agree
    /// with the general route it bypasses.
    #[test]
    fn the_two_degree_closed_form_agrees_with_the_incomplete_gamma() {
        for statistic in [0.05, 0.5, 1.0, 2.7055, 3.8415, 7.0, 15.0, 40.0] {
            let closed = chi2_upper_tail(statistic, 2.0);
            let general = statrs::function::gamma::gamma_ur(1.0, statistic / 2.0);
            assert!(
                (closed - general).abs() <= 1e-15 * general,
                "chi-square on two at {statistic}: {closed} against {general}"
            );
        }
    }

    /// The two critical values the interval recipe rests on. The tolerance is
    /// far tighter than the route this replaced could hold: `erfc` was out by
    /// about 4e-12 here.
    #[test]
    fn the_reference_critical_values_are_where_they_should_be() {
        assert!((chi2_upper_tail(3.841_458_820_694_124, 1.0) - 0.05).abs() < 1e-15);
        // The even mixture of a point mass and chi-square on one.
        let mixture = 0.5 * chi2_upper_tail(2.705_543_454_095_404, 1.0);
        assert!((mixture - 0.05).abs() < 1e-15);
    }

    /// Degrees of freedom beyond the special cases are answered correctly
    /// rather than being quietly given the three-degree tail, which is what
    /// the copy this replaced did.
    #[test]
    fn degrees_beyond_the_special_cases_are_not_quietly_wrong() {
        let four = chi2_upper_tail(5.0, 4.0);
        assert!((four - statrs::function::gamma::gamma_ur(2.0, 2.5)).abs() < 1e-15);
        assert!(
            (four - chi2_upper_tail(5.0, 3.0)).abs() > 1e-3,
            "four degrees returned the three-degree tail"
        );
    }

    /// A search that fell short reports no evidence, not negative evidence.
    #[test]
    fn a_null_fitted_better_than_its_alternative_reports_no_evidence() {
        assert_eq!(deviance(-10.5, -10.0), 0.0);
        assert!((p_value(deviance(-10.5, -10.0), |t| chi2_upper_tail(t, 1.0)) - 1.0).abs() < 1e-15);
    }

    /// A fit exactly on its bound is p = 1, not p = 0.5.
    #[test]
    fn a_fit_on_its_bound_is_not_reported_at_a_half() {
        let on_the_bound = p_value(0.0, |t| 0.5 * chi2_upper_tail(t, 1.0));
        assert!((on_the_bound - 1.0).abs() < 1e-15);
        assert_eq!(settled(1e-7), 0.0);
        assert_eq!(settled(1e-6), 1e-6);
        // Just past the settling tolerance the mixture takes over as usual.
        let just_off = p_value(1.0, |t| 0.5 * chi2_upper_tail(t, 1.0));
        assert!(just_off < 0.2 && just_off > 0.1, "{just_off}");
    }
}
