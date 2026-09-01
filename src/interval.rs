//! The profile-likelihood interval, in one place.
//!
//! # Why this is in one place
//!
//! It was in ten. `0004` chose one interval recipe by measurement -- a profile
//! likelihood, the deviance crossing a chi-square threshold, and the Self-Liang
//! mixture deciding whether a boundary point belongs -- and then nine families
//! wrote that recipe out separately, one of them twice. `components` and
//! `bivariate` differ by two comment lines. `gxe` and `liability` differ by two
//! tokens: sixty iterations against forty, and a tolerance of 1e-7 against
//! 1e-5. One paragraph of doc comment appears verbatim in four of the records.
//! The critical value stands in ten places and in three forms: eight named
//! constants under two different names, one bare literal in
//! `components::signed_interval`, and one already halved in `discrete_gxe`.
//!
//! Two things had gone wrong by the time anybody put the copies side by side,
//! and both are the reason this module exists rather than tidiness.
//!
//! **A repair reached seven of the ten.** Reading a profile evaluation that
//! failed as ground the data ruled out narrows the interval without saying so:
//! the bisection steps inward on evidence nobody gathered, and the interval
//! comes back confidently narrower than the data support. Seven now cover the
//! failure and count it. `spatial` still reads a failure as an infinite
//! deviance, and `components::signed_interval` still reads one as outside;
//! neither has a field to report it in. `prepared` does something different
//! again -- it bisects to the feasible edge, which is sound, and reports
//! nothing either. A recipe written once cannot be repaired in seven places out
//! of ten.
//!
//! **The mixture verdict exists in two families out of nine.** Whether an
//! endpoint sitting on its bound belongs to the interval is not answerable by
//! reading the code; `CONTEXT.md` is blunt about it -- the coverage check is
//! what chooses an interval recipe. `prepared` has current qualification.
//! `tobit` historically populated the verdict, but its fixed-instrument
//! coverage remeasurement is pending; the other families leave it absent.
//!
//! # What a family hands over
//!
//! The profiled log-likelihood at a held value, its fitted estimate, and its
//! bounds. Nothing else. Every family's own vocabulary -- which component,
//! which reported quantity, which surface, whether the fit is integrated -- is
//! resolved before bracketing begins, so by the time the search starts they are
//! all holding one number and asking for a likelihood.
//!
//! The maximum is taken by evaluating the same closure at the estimate rather
//! than from the free fit. Both ends of the difference are then measured the
//! same way. Taking the maximum from the free fit instead leaves a constant in
//! the deviance, which is how the two-trait intervals once came to have zero
//! width. Four of the copies being replaced carry a comment saying so, which is
//! four chances to write the warning down and none to act on it once.
//!
//! # One tolerance and one iteration cap
//!
//! 1e-9 and eighty halvings, from `0011`. The copies presently run 1e-9, 1e-7,
//! 1e-5, 1e-4 times the fitted value, and 1e-15, with caps of 200, 80, 80, 80,
//! 60, 60 and 40. None of the differences carries a comment explaining itself,
//! which is the evidence that they are accidental rather than chosen.
//!
//! What the tightest setting costs is a real question, and `0011` makes
//! adopting it conditional on measuring the answer. Half of that measurement
//! lives here: a unit bracket needs thirty halvings to reach 1e-9 and seventeen
//! to reach 1e-5, so one endpoint costs thirteen more profile evaluations than
//! the loosest copy charged. The other half is what one evaluation costs in
//! each family, which is a wall-clock measurement and belongs with the family
//! as it moves across.

/// The 95th percentile of chi-square on one degree of freedom.
///
/// The deviance threshold for a two-sided interval on an interior parameter.
const CHI2_ONE_DF_95: f64 = 3.841_458_820_694_124;

/// The 95th percentile of the even mixture of a point mass at nought and
/// chi-square on one degree of freedom.
///
/// The Self-Liang reference for a parameter sitting on its bound, and the
/// threshold the mixture verdict reads. It is smaller than the interior
/// threshold, so a bound can belong to the interval on this rule and not on the
/// other.
///
/// Unused outside the tests until a family with valid current qualification
/// moves across. `allow` rather than `expect`, because the tests here do use it,
/// so the lint fires only in a build without them.
#[allow(dead_code, reason = "no family has moved its scored bounds across yet")]
const MIXTURE_CRIT: f64 = 2.705_543_454_095_404;

/// The boundary rule must stay the easier of the two to satisfy: a bound can
/// belong to the interval on the mixture and not on the interior threshold, and
/// never the reverse. Swapping the two constants would be silent otherwise.
const _: () = assert!(MIXTURE_CRIT < CHI2_ONE_DF_95);

/// How close the bracket must come before an endpoint is called found.
const TOLERANCE: f64 = 1e-9;

/// How many halvings an endpoint gets. Thirty reach [`TOLERANCE`] on a unit
/// bracket, so this is a backstop against a bracket that will not shrink and
/// not a working limit.
const HALVINGS: usize = 80;

/// A profile-likelihood interval for one reported quantity.
///
/// One record whatever family asked for it. A field that does not apply is
/// absent rather than the record changing shape, which is `0005`'s rule: a
/// result whose type varies invites conditional handling downstream.
#[derive(Clone, Debug)]
pub struct Interval {
    /// The fitted value the interval is built around. Absent where the profile
    /// could not be evaluated there, which is the one case in which there is no
    /// interval at all.
    pub estimate: Option<f64>,
    pub lower: f64,
    pub upper: f64,
    /// True where the end sits on the parameter's own bound rather than where
    /// the profile fell away -- the data did not rule that end out.
    ///
    /// Read it beside `profile_failures`: a bound reached because the profile
    /// could not be evaluated there is a different thing from a bound the
    /// likelihood genuinely never left, and both arrive here as `true`.
    pub lower_limited: bool,
    pub upper_limited: bool,
    pub level: f64,
    /// How many profile evaluations failed. Each one widened the interval
    /// rather than narrowing it, which is the safe direction, but a large count
    /// means the interval rests on fewer points than it looks.
    pub profile_failures: usize,
    /// Whether a boundary point belongs to the interval, by the Self-Liang
    /// mixture rather than by the end having landed on the bound.
    ///
    /// **Absent means nobody has measured it here, not that the question does
    /// not apply.** A family fills this only where a coverage simulation has
    /// scored its interval at that bound, by calling
    /// [`Interval::scored_by_mixture`]. Reasoning that the geometry looks like
    /// another family's is not measuring it.
    pub contains_lower_bound: Option<bool>,
    pub contains_upper_bound: Option<bool>,
    /// The deviance at each bound, lower then upper, kept so that
    /// [`Interval::scored_by_mixture`] costs no second profile fit. Absent
    /// where the bound could not be evaluated.
    ///
    /// Private, so that the only way to build an interval is to search for one
    /// and the only way to score a bound is to have searched it.
    #[allow(
        dead_code,
        reason = "read by `scored_by_mixture`, which has no caller yet"
    )]
    at_bounds: [Option<f64>; 2],
}

impl Interval {
    /// No interval: the profile could not be evaluated at the estimate, so
    /// there is nothing to bracket from.
    ///
    /// The endpoints are not-a-number rather than the bounds, because a bound
    /// would read as an interval that ran to the edge of the parameter space,
    /// which is a claim about the data. `estimate.is_none()` is the test.
    #[must_use]
    pub(crate) fn absent() -> Self {
        Self {
            estimate: None,
            lower: f64::NAN,
            upper: f64::NAN,
            lower_limited: false,
            upper_limited: false,
            level: 0.95,
            profile_failures: 0,
            contains_lower_bound: None,
            contains_upper_bound: None,
            at_bounds: [None, None],
        }
    }

    /// Retain the published Self-Liang boundary verdict for a limited end.
    ///
    /// This remains the one-component censored compatibility record while its
    /// corrected fixed-instrument finite-sample coverage campaign is pending.
    /// Families without that compatibility surface leave these fields absent.
    ///
    /// An end that is not limited is interior, so the question does not arise
    /// and the field stays absent. An end limited because the bound could not
    /// be evaluated also stays absent: there is no deviance to score.
    #[must_use]
    #[allow(dead_code, reason = "waiting on `prepared` and `tobit` to move across")]
    pub(crate) fn scored_by_mixture(mut self) -> Self {
        if self.lower_limited {
            self.contains_lower_bound = self.at_bounds[0].map(|d| d <= MIXTURE_CRIT);
        }
        if self.upper_limited {
            self.contains_upper_bound = self.at_bounds[1].map(|d| d <= MIXTURE_CRIT);
        }
        self
    }
}

/// Bracket one endpoint, from the estimate outward to `bound`.
///
/// Returns where the end landed, whether it is limited by the bound rather than
/// by the data, and the deviance at the bound for the mixture verdict to read
/// later.
fn endpoint(
    estimate: f64,
    bound: f64,
    deviance_at: &impl Fn(f64) -> Option<f64>,
    failures: &mut usize,
) -> (f64, bool, Option<f64>) {
    let at_bound = deviance_at(bound);
    match at_bound {
        // **A profile that could not be evaluated is not a likelihood that fell
        // away.** Read as an infinite deviance it looks like ground the data
        // ruled out, and the end is placed inside the bound on evidence that
        // was never gathered. Cover it instead, and count it.
        None => {
            *failures += 1;
            return (bound, true, None);
        }
        // The threshold is never reached: the end is the bound, and the
        // interval is limited by the parameter space rather than by the data.
        Some(value) if value <= CHI2_ONE_DF_95 => return (bound, true, at_bound),
        Some(_) => {}
    }

    let (mut inside, mut outside) = (estimate, bound);
    for _ in 0..HALVINGS {
        if (outside - inside).abs() <= TOLERANCE {
            break;
        }
        let middle = 0.5 * (inside + outside);
        // The bracket has closed to neighbouring doubles and will not halve
        // again. Without this the remaining halvings all return the same point,
        // which costs profile fits and buys nothing.
        if middle == inside || middle == outside {
            break;
        }
        match deviance_at(middle) {
            Some(value) if value <= CHI2_ONE_DF_95 => inside = middle,
            Some(_) => outside = middle,
            None => {
                // Unknown, so widen rather than narrow, and say so.
                *failures += 1;
                inside = middle;
            }
        }
    }
    (0.5 * (inside + outside), false, at_bound)
}

/// A 95 per cent profile-likelihood interval.
///
/// `objective` returns the profiled log-likelihood with the quantity held at
/// the given value, or `None` where that fit could not be made. `bounds` are the
/// parameter's own limits, lower first.
///
/// A `Some` carrying a value that is not finite counts as `None`. A family need
/// not have checked, and several of them do not.
///
/// Where the objective cannot be evaluated at the estimate there is nothing to
/// bracket from and the result is [`Interval::absent`].
pub(crate) fn profile_interval(
    estimate: f64,
    bounds: (f64, f64),
    objective: impl Fn(f64) -> Option<f64>,
) -> Interval {
    // **A log-likelihood that is not finite is not a likelihood.** It is a
    // covariance that would not factorise, or a quadratic form that overflowed,
    // and it arrives wearing `Some` because the family that produced it checked
    // whether the fit ran rather than whether the number means anything.
    //
    // Left alone it reads as a deviance of infinity, which is ground the data
    // ruled out, and the end is placed inside the bound on a computation that
    // was never made. That is the fault this module exists to stop, so the
    // contract is enforced here rather than trusted: unevaluable is unevaluable,
    // whichever way a family says it. `components` never returns one -- it
    // re-evaluates and tests `is_finite` before accepting -- so this moves
    // nothing today. `prepared` does return one, and will need this when it
    // moves across.
    let evaluate = |value: f64| objective(value).filter(|ll| ll.is_finite());

    let Some(maximum) = evaluate(estimate) else {
        return Interval::absent();
    };
    let deviance_at = |value: f64| evaluate(value).map(|ll| 2.0 * (maximum - ll));

    let (bottom, top) = bounds;
    let mut failures = 0usize;
    let (lower, lower_limited, at_lower) = endpoint(estimate, bottom, &deviance_at, &mut failures);
    let (upper, upper_limited, at_upper) = endpoint(estimate, top, &deviance_at, &mut failures);

    Interval {
        estimate: Some(estimate),
        lower,
        upper,
        lower_limited,
        upper_limited,
        level: 0.95,
        profile_failures: failures,
        contains_lower_bound: None,
        contains_upper_bound: None,
        at_bounds: [at_lower, at_upper],
    }
}

#[cfg(test)]
mod tests {
    use super::{CHI2_ONE_DF_95, HALVINGS, Interval, MIXTURE_CRIT, TOLERANCE, profile_interval};
    use crate::deviance::chi2_upper_tail;
    use std::cell::Cell;

    /// A Gaussian log-likelihood in the reported quantity, peaked at `centre`.
    ///
    /// Its deviance is `((v - centre) / spread)^2` exactly, so the crossing is
    /// `centre +/- spread * sqrt(3.8415)` by arithmetic done outside this
    /// crate. That is what makes it a check rather than a restatement.
    fn quadratic(centre: f64, spread: f64) -> impl Fn(f64) -> Option<f64> {
        move |v: f64| Some(-0.5 * ((v - centre) / spread).powi(2))
    }

    /// The two critical values, against the tail they are percentiles of.
    ///
    /// Seven families type the first of these out by hand. A digit dropped in
    /// any of them would move every endpoint that family reports and nothing
    /// would say so.
    #[test]
    fn the_thresholds_are_the_percentiles_they_claim_to_be() {
        assert!((chi2_upper_tail(CHI2_ONE_DF_95, 1.0) - 0.05).abs() < 1e-15);
        let mixture = 0.5 * chi2_upper_tail(MIXTURE_CRIT, 1.0);
        assert!((mixture - 0.05).abs() < 1e-15);
    }

    /// The endpoints are where the arithmetic says, not merely where the code
    /// last put them.
    #[test]
    fn a_known_profile_gives_the_endpoints_arithmetic_predicts() {
        let (centre, spread) = (0.4, 0.1);
        let got = profile_interval(centre, (0.0, 1.0), quadratic(centre, spread));
        let half_width = spread * CHI2_ONE_DF_95.sqrt();
        assert!(
            (got.lower - (centre - half_width)).abs() < 1e-9,
            "lower at {} against {}",
            got.lower,
            centre - half_width
        );
        assert!(
            (got.upper - (centre + half_width)).abs() < 1e-9,
            "upper at {} against {}",
            got.upper,
            centre + half_width
        );
        assert!(!got.lower_limited && !got.upper_limited);
        assert_eq!(got.profile_failures, 0);
        assert_eq!(got.estimate, Some(centre));
    }

    /// A profile that never crosses reports each bound as reached, and says so
    /// through `limited` rather than by returning a suspiciously round number.
    #[test]
    fn a_flat_profile_runs_to_both_bounds_and_reports_them_as_limits() {
        let got = profile_interval(0.5, (0.0, 1.0), |_| Some(0.0));
        assert_eq!((got.lower, got.upper), (0.0, 1.0));
        assert!(got.lower_limited && got.upper_limited);
        assert_eq!(got.profile_failures, 0);
    }

    /// **The repair, stated as a test.** A profile that cannot be evaluated in a
    /// band inside the crossing must widen the interval, never narrow it.
    ///
    /// Reading the failure as an infinite deviance -- which is what `spatial`
    /// still does -- moves the outer edge inward exactly as a genuine drop does,
    /// and returns an interval narrower than the data support with nothing on
    /// the record to show it happened. Here the true crossing is at 0.204; the
    /// broken reading returns about 0.25 and the repaired one about 0.15.
    #[test]
    fn a_failed_evaluation_widens_the_interval_rather_than_narrowing_it() {
        let (centre, spread) = (0.4, 0.1);
        let true_crossing = centre - spread * CHI2_ONE_DF_95.sqrt();
        let base = quadratic(centre, spread);
        let holed = move |v: f64| {
            if (0.15..0.25).contains(&v) {
                None
            } else {
                base(v)
            }
        };
        let got = profile_interval(centre, (0.0, 1.0), holed);
        assert!(
            got.lower < true_crossing,
            "the hole should have widened the interval past {true_crossing}, not to {}",
            got.lower
        );
        assert!(
            got.profile_failures > 0,
            "a failure that moved an endpoint must be on the record"
        );
        // The upper end is clear of the hole and must be untouched by it.
        assert!((got.upper - (centre + spread * CHI2_ONE_DF_95.sqrt())).abs() < 1e-9);
    }

    /// A bound that cannot be evaluated is covered and counted, and is not
    /// confused with a bound the likelihood genuinely never left.
    #[test]
    fn a_bound_that_cannot_be_evaluated_is_covered_and_counted() {
        let base = quadratic(0.4, 0.1);
        let got = profile_interval(
            0.4,
            (0.0, 1.0),
            move |v: f64| {
                if v < 0.05 { None } else { base(v) }
            },
        );
        assert_eq!(got.lower, 0.0, "the unevaluable bound must be covered");
        assert!(got.lower_limited);
        assert_eq!(
            got.profile_failures, 1,
            "the reader has only this count to tell the two kinds of limit apart"
        );
        assert_eq!(
            got.contains_lower_bound, None,
            "there is no deviance at that bound to score"
        );
    }

    /// The verdict is absent until a family claims the measurement, and then
    /// present only where an end is on its bound.
    #[test]
    fn the_mixture_verdict_is_absent_until_a_family_asks_for_it() {
        // Flat enough that the mixture rule admits both bounds.
        let unscored = profile_interval(0.5, (0.0, 1.0), |_| Some(0.0));
        assert_eq!(unscored.contains_lower_bound, None);
        assert_eq!(unscored.contains_upper_bound, None);
        let scored = unscored.scored_by_mixture();
        assert_eq!(scored.contains_lower_bound, Some(true));
        assert_eq!(scored.contains_upper_bound, Some(true));

        // Interior ends: the question does not arise, so it stays absent even
        // for a family that has run the simulation.
        let interior = profile_interval(0.4, (0.0, 1.0), quadratic(0.4, 0.1)).scored_by_mixture();
        assert_eq!(interior.contains_lower_bound, None);
        assert_eq!(interior.contains_upper_bound, None);
    }

    /// The boundary rule is the easier of the two, so a bound can be reached and
    /// still be judged not to belong.
    ///
    /// The interval threshold is 3.8415 and the mixture's is 2.7055. A profile
    /// whose deviance at the bound sits between them limits the end there -- no
    /// crossing -- while the mixture says the bound is outside.
    #[test]
    fn a_reached_bound_can_still_fail_the_mixture_rule() {
        let between = 0.5 * (MIXTURE_CRIT + CHI2_ONE_DF_95);
        assert!(between > MIXTURE_CRIT && between < CHI2_ONE_DF_95);
        // Deviance rises to `between` at the lower bound and no further.
        let profile = move |v: f64| Some(-0.5 * between * (0.5 - v).abs() / 0.5);
        let got = profile_interval(0.5, (0.0, 1.0), profile).scored_by_mixture();
        assert!(got.lower_limited, "the interval threshold is never crossed");
        assert_eq!(
            got.contains_lower_bound,
            Some(false),
            "the mixture rule is stricter here and must be allowed to disagree"
        );
    }

    /// A log-likelihood that is not finite counts as a fit that could not be
    /// made, whichever way the family reports it.
    ///
    /// `Some(-inf)` left alone is a deviance of infinity, which is read as
    /// ground the data ruled out -- so the end would be placed inside the bound
    /// on a computation nobody made. Here the profile returns negative infinity
    /// over the same band as the `None` test above, and must produce the same
    /// widened interval and the same count.
    #[test]
    fn an_infinite_log_likelihood_counts_as_a_fit_that_could_not_be_made() {
        let (centre, spread) = (0.4, 0.1);
        let true_crossing = centre - spread * CHI2_ONE_DF_95.sqrt();
        let infinite = {
            let base = quadratic(centre, spread);
            move |v: f64| {
                if (0.15..0.25).contains(&v) {
                    Some(f64::NEG_INFINITY)
                } else {
                    base(v)
                }
            }
        };
        let got = profile_interval(centre, (0.0, 1.0), infinite);
        assert!(
            got.lower < true_crossing,
            "an infinite log-likelihood narrowed the interval to {}",
            got.lower
        );
        assert!(got.profile_failures > 0);

        // Identical to the `None` band, which is the point: the two ways of
        // saying "no answer" must not lead to different intervals.
        let holed = {
            let base = quadratic(centre, spread);
            move |v: f64| {
                if (0.15..0.25).contains(&v) {
                    None
                } else {
                    base(v)
                }
            }
        };
        let same = profile_interval(centre, (0.0, 1.0), holed);
        assert_eq!(got.lower, same.lower);
        assert_eq!(got.profile_failures, same.profile_failures);
    }

    /// A profile that is not finite at the estimate leaves no interval, the same
    /// as one that could not be evaluated there at all.
    #[test]
    fn an_infinite_maximum_leaves_no_interval() {
        let got = profile_interval(0.4, (0.0, 1.0), |_| Some(f64::NEG_INFINITY));
        assert!(got.estimate.is_none());
        let nan = profile_interval(0.4, (0.0, 1.0), |_| Some(f64::NAN));
        assert!(nan.estimate.is_none());
    }

    /// An estimate whose own profile fails leaves no interval, and says so by
    /// the absence of an estimate rather than by a bound-to-bound answer.
    #[test]
    fn an_unevaluable_estimate_leaves_no_interval() {
        let got = profile_interval(0.4, (0.0, 1.0), |_| None);
        assert!(got.estimate.is_none());
        assert!(got.lower.is_nan() && got.upper.is_nan());
        assert!(!got.lower_limited && !got.upper_limited);
    }

    /// **What the tightest tolerance costs, in profile evaluations.**
    ///
    /// A bracket of width `w` takes `ceil(log2(w / tolerance))` halvings, and
    /// each halving is a profile fit. So the cost of the choice `0011` makes is
    /// the gap between that count at 1e-9 and at the 1e-5 of the loosest copy
    /// being replaced: `log2(1e4)`, which is **thirteen extra fits per
    /// endpoint** and twenty-six per interval, whatever the bracket.
    ///
    /// That is the half of the measurement that can be made here. What thirteen
    /// fits cost in seconds differs by family and is measured as each family
    /// moves across.
    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a halving count, two digits at most"
    )]
    fn the_bracketing_costs_the_halvings_the_tolerance_demands() {
        let calls = Cell::new(0usize);
        let base = quadratic(0.4, 0.1);
        // Only the lower end crosses; the upper bound is the estimate itself,
        // so the count is one endpoint's and not two.
        let counted = |v: f64| {
            calls.set(calls.get() + 1);
            base(v)
        };
        let (estimate, bottom) = (0.4, 0.0);
        let got = profile_interval(estimate, (bottom, estimate), counted);
        assert!(!got.lower_limited);
        assert!(got.upper_limited, "the upper bound is the estimate itself");

        // One evaluation at the estimate and one at each bound, then halvings.
        let halvings = calls.get() - 3;
        let width = estimate - bottom;
        let needed = |tolerance: f64| (width / tolerance).log2().ceil() as usize;
        assert_eq!(
            halvings,
            needed(TOLERANCE),
            "a bracket {width} wide takes that many halvings to reach {TOLERANCE}"
        );
        assert_eq!(
            halvings - needed(1e-5),
            13,
            "the tightening from the loosest copy costs thirteen fits an endpoint"
        );
        assert!(
            HALVINGS > 2 * halvings,
            "the cap is a backstop, not a working limit"
        );
    }

    /// The bracket is closed to the tolerance, not merely to somewhere near it.
    #[test]
    fn the_bracket_closes_to_the_stated_tolerance() {
        let got = profile_interval(0.4, (0.0, 1.0), quadratic(0.4, 0.1));
        let exact = 0.4 - 0.1 * CHI2_ONE_DF_95.sqrt();
        assert!(
            (got.lower - exact).abs() <= TOLERANCE,
            "the lower end is {} from the crossing, past the {TOLERANCE} claimed",
            (got.lower - exact).abs()
        );
    }

    /// An absent interval carries no verdict and cannot acquire one.
    #[test]
    fn an_absent_interval_cannot_be_scored_into_having_a_verdict() {
        let got = Interval::absent().scored_by_mixture();
        assert_eq!(got.contains_lower_bound, None);
        assert_eq!(got.contains_upper_bound, None);
    }

    /// An estimate sitting on its own bound limits that end rather than
    /// bracketing an empty interval.
    #[test]
    fn an_estimate_on_its_bound_limits_that_end() {
        let got = profile_interval(0.0, (0.0, 1.0), quadratic(0.0, 0.1));
        assert_eq!(got.lower, 0.0);
        assert!(
            got.lower_limited,
            "there is nothing below the bound to search"
        );
        assert!(!got.upper_limited);
        assert!((got.upper - 0.1 * CHI2_ONE_DF_95.sqrt()).abs() < 1e-9);
    }
}
