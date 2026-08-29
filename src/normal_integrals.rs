//! The normal integrals every censored and liability model rests on.
//!
//! **One implementation, shared.** The package used to carry two of these: a
//! sixteen-point Gauss-Legendre quadrature beside the liability model, and the
//! adaptive form below beside the latent-mediation model. That is the thing
//! `crate::tobit` warns about when it explains why it shares its region
//! probability rather than writing a second one -- a second implementation of
//! the same integral is how a package comes to hold two different answers --
//! and here the two answers differed by enough to matter.
//!
//! **Measured, 29 August 2026.** The retired quadrature's error in the log
//! probability, at thresholds 0.5 and -0.3, against an independent reference:
//! 1.8e-08 at a correlation of 0.3, 1.0e-04 at 0.9, 4.5e-02 at 0.99 and 1.8e-01
//! at 0.999. Its own documentation put the worst error below 0.99 at 2.3e-04,
//! which is exactly right at thresholds of nought and nought and about seventy
//! times too small away from them: the error depends on where the rectangle is,
//! not only on how strongly the coordinates correlate.
//!
//! That error is why `build` used to refuse a relationship above 0.9, and so
//! refused monozygotic twins and any component a person shares with themselves
//! -- a person-level component being exactly one between two records of one
//! person. With the integral fixed the refusal is not needed.
//!
//! The univariate tail is `erfc`-based rather than `statrs`'s `Normal::cdf`,
//! which carries about 5e-11 of relative error that wanders from point to
//! point. A quadrature cannot integrate against a function that jitters.

const LOG_TWO_PI: f64 = 1.837_877_066_409_345_3;

// Below this absolute scale, the corner-difference recipe cannot distinguish
// a probability from numerical zero and the log-scale conditional quadrature
// takes over.
pub(crate) const BIVARIATE_ABSOLUTE_TOLERANCE: f64 = 1.0e-12;

/// The standard normal distribution function.
pub(crate) fn normal_cdf(value: f64) -> Result<f64, &'static str> {
    normal_sf(-value)
}

/// Upper tail of the standard normal, as `erfc(x / sqrt 2) / 2`.
///
/// **Not `statrs`'s `Normal::cdf`, which is not accurate enough to integrate
/// against.** That routes through an `erfc` approximation carrying about 5e-11
/// of relative error -- it puts the tail at 3 deviations at 1.349898031574e-3
/// where the true value is 1.349898031630e-3 -- and the error wanders from
/// point to point rather than varying smoothly. To a quadrature asking for
/// 1e-13 that is noise, and the conditional integrand inherited it: panels
/// around 6.06 deviations kept returning an error estimate that fell only as
/// fast as the panel width, the signature of a rough integrand, so the routine
/// subdivided to its depth limit and refused rectangles as ordinary as
/// [6, 40] x [5, inf) at a correlation of 0.7.
///
/// `libm`'s is the FDLIBM routine. Against a sixty-digit evaluation it is right
/// to an ulp for ordinary arguments and never worse than 4e-14 out to 36
/// deviations -- three orders better than `statrs` at its best -- while running
/// about four times faster than routing the same quantity through the
/// regularised incomplete gamma, which is equally accurate but pays for a
/// generality this does not need. Below nought the argument is negative and the
/// result lies between one and two, so there is no cancellation to avoid.
pub(crate) fn normal_sf(value: f64) -> Result<f64, &'static str> {
    if !value.is_finite() {
        return Err("NORMAL_VARIATE_NOT_FINITE");
    }
    Ok(0.5 * libm::erfc(value / std::f64::consts::SQRT_2))
}

/// Log of the upper tail of the standard normal, stable arbitrarily far out.
pub(crate) fn log_normal_sf(value: f64) -> Result<f64, &'static str> {
    if value.is_nan() {
        return Err("NORMAL_VARIATE_NOT_FINITE");
    }
    if value == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if value == f64::INFINITY {
        return Ok(f64::NEG_INFINITY);
    }
    if value <= 6.0 {
        // The survival value is at least 1e-9 here, so its logarithm keeps
        // full relative accuracy.
        return Ok(normal_sf(value)?.ln());
    }
    // Mills-ratio continued fraction: the survival equals
    // phi(x) / (x + 1/(x + 2/(x + 3/(...)))), evaluated backward.
    let mut tail = 0.0;
    for level in (1..=40u32).rev() {
        tail = f64::from(level) / (value + tail);
    }
    Ok(-0.5 * value * value - 0.5 * LOG_TWO_PI - (value + tail).ln())
}

/// The standard normal probability of an interval.
pub(crate) fn interval_probability(lower: f64, upper: f64) -> Result<f64, &'static str> {
    if lower >= upper {
        return Ok(0.0);
    }
    if lower == f64::NEG_INFINITY {
        return if upper == f64::INFINITY {
            Ok(1.0)
        } else {
            normal_cdf(upper)
        };
    }
    if upper == f64::INFINITY {
        return normal_sf(lower);
    }
    if lower >= 0.0 {
        return Ok((normal_sf(lower)? - normal_sf(upper)?).max(0.0));
    }
    Ok((normal_cdf(upper)? - normal_cdf(lower)?).max(0.0))
}

/// Log of a standard-normal interval probability, stable in either tail.
pub(crate) fn log_interval_probability(lower: f64, upper: f64) -> Result<f64, &'static str> {
    if lower >= upper {
        return Ok(f64::NEG_INFINITY);
    }
    if lower >= 0.0 {
        let log_lower_tail = log_normal_sf(lower)?;
        let log_upper_tail = log_normal_sf(upper)?;
        return Ok(log_lower_tail + (-(log_upper_tail - log_lower_tail).exp()).ln_1p());
    }
    if upper <= 0.0 {
        return log_interval_probability(-upper, -lower);
    }
    Ok(interval_probability(lower, upper)?.ln())
}

/// An upper bound on a standardised rectangle's probability, on the log scale.
///
/// **On the log scale because the bound is used to check the tail answer, and
/// an ordinary-scale bound underflows before the answers it is meant to
/// check.** Past about 38 deviations the bound arrives as nought, the guard
/// reads `bound > 0` and steps aside, and the deep tail -- the one place the
/// conditional quadrature is the only recipe available and so the one place a
/// check is worth having -- went unchecked.
pub(crate) fn log_bivariate_geometry_upper_bound(
    lower: &[f64; 2],
    upper: &[f64; 2],
    correlation: f64,
) -> Result<f64, &'static str> {
    if correlation.abs() >= 1.0 || !correlation.is_finite() {
        return Err("BIVARIATE_CORRELATION_INVALID");
    }
    let marginal_bound = log_interval_probability(lower[0], upper[0])?
        .min(log_interval_probability(lower[1], upper[1])?);

    // For standardised X and Y, S=X+Y and D=X-Y are independent Gaussian
    // variables.  Every point in the rectangle lies in both induced
    // intervals, hence P(rectangle) <= P(S interval) P(D interval).
    let sum_sd = (2.0 * (1.0 + correlation)).sqrt();
    let difference_sd = (2.0 * (1.0 - correlation)).sqrt();
    let sum_probability = log_interval_probability(
        (lower[0] + lower[1]) / sum_sd,
        (upper[0] + upper[1]) / sum_sd,
    )?;
    let difference_probability = log_interval_probability(
        (lower[0] - upper[1]) / difference_sd,
        (upper[0] - lower[1]) / difference_sd,
    )?;
    Ok(marginal_bound
        .min(sum_probability + difference_probability)
        .min(0.0))
}

/// The bivariate standard normal distribution function, `P(X <= a, Y <= b)`.
///
/// Owen's angular form, integrated adaptively. Accurate across the whole
/// correlation range and away from the symmetric centre, which the
/// sixteen-point quadrature it replaced was not: see this module's own
/// documentation, and the reference grid in the tests below.
pub(crate) fn bivariate_normal_cdf(
    first: f64,
    second: f64,
    correlation: f64,
) -> Result<f64, &'static str> {
    if correlation.abs() >= 1.0 || !correlation.is_finite() {
        return Err("BIVARIATE_CORRELATION_INVALID");
    }
    if first == f64::NEG_INFINITY || second == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if first == f64::INFINITY && second == f64::INFINITY {
        return Ok(1.0);
    }
    if first == f64::INFINITY {
        return normal_cdf(second);
    }
    if second == f64::INFINITY {
        return normal_cdf(first);
    }
    if !first.is_finite() || !second.is_finite() {
        return Err("BIVARIATE_THRESHOLD_INVALID");
    }
    let geometry_bound = log_bivariate_geometry_upper_bound(
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[first, second],
        correlation,
    )?
    .exp();
    if geometry_bound <= BIVARIATE_ABSOLUTE_TOLERANCE {
        return Err("BIVARIATE_PROBABILITY_UNRESOLVED");
    }
    if correlation < 0.0 {
        return Ok(normal_cdf(first)? - bivariate_normal_cdf(first, -second, -correlation)?);
    }
    let independent = normal_cdf(first)? * normal_cdf(second)?;
    if correlation.abs() <= f64::EPSILON {
        return Ok(independent);
    }
    let integrand = |angle: f64| {
        let sine = angle.sin();
        let cosine = angle.cos();
        let exponent =
            (first - second).powi(2) / (2.0 * cosine * cosine) + first * second / (1.0 + sine);
        (-exponent).exp() / (2.0 * std::f64::consts::PI)
    };
    let quadrature = adaptive_simpson(
        &integrand,
        0.0,
        correlation.asin(),
        BIVARIATE_ABSOLUTE_TOLERANCE / 8.0,
        // An ordinary integrand on the ordinary scale, so an epsilon.
        f64::EPSILON,
        24,
    )?;
    let probability = independent + quadrature.value;
    if probability <= quadrature.estimated_error {
        return Err("BIVARIATE_PROBABILITY_UNRESOLVED");
    }
    let upper_bound = normal_cdf(first)?.min(normal_cdf(second)?);
    // **The Frechet bounds can cross by rounding, and `clamp` panics when they
    // do.** With both marginals at essentially one, the lower bound is
    // `p + q - 1` and the upper is `min(p, q)`; these meet exactly at one and
    // an ulp of arithmetic is enough to put the lower above the upper --
    // measured at 0.9999999910267894 against 0.9999999910267893, which brought
    // the whole process down through the Python boundary rather than returning
    // an error. Ordering them costs nothing and the interval they describe is
    // a point at that precision anyway.
    let lower_bound = (normal_cdf(first)? + normal_cdf(second)? - 1.0)
        .max(0.0)
        .min(upper_bound);
    if probability < lower_bound - 1.0e-10 || probability > upper_bound + 1.0e-10 {
        return Err("BIVARIATE_PROBABILITY_OUTSIDE_BOUNDS");
    }
    Ok(probability.clamp(lower_bound, upper_bound))
}

#[derive(Debug)]
pub(crate) struct QuadratureResult {
    pub(crate) value: f64,
    pub(crate) estimated_error: f64,
}

/// Adaptive Simpson quadrature, to a tolerance, refusing rather than
/// returning a value it could not reach.
pub(crate) fn adaptive_simpson<F>(
    function: &F,
    lower: f64,
    upper: f64,
    tolerance: f64,
    // The relative rounding the integrand itself carries. An ordinary
    // integrand carries an epsilon; one built from a large exponent carries an
    // epsilon of that exponent.
    noise: f64,
    maximum_depth: usize,
) -> Result<QuadratureResult, &'static str>
where
    F: Fn(f64) -> f64,
{
    fn simpson<F>(function: &F, lower: f64, upper: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let midpoint = lower.midpoint(upper);
        (upper - lower) * (function(lower) + 4.0 * function(midpoint) + function(upper)) / 6.0
    }
    fn recurse<F>(
        function: &F,
        lower: f64,
        upper: f64,
        estimate: f64,
        tolerance: f64,
        noise: f64,
        depth: usize,
    ) -> Result<QuadratureResult, &'static str>
    where
        F: Fn(f64) -> f64,
    {
        let midpoint = lower.midpoint(upper);
        let left = simpson(function, lower, midpoint);
        let right = simpson(function, midpoint, upper);
        let error = left + right - estimate;
        if !left.is_finite() || !right.is_finite() {
            return Err("BIVARIATE_QUADRATURE_NOT_FINITE");
        }
        // The error budget is halved at every level, so it eventually asks for
        // less than the arithmetic can deliver: a panel is accepted once its
        // error reaches the level of its own rounding, because no amount of
        // further splitting can improve on that. Without this the routine
        // answers a rough integrand by subdividing to its depth limit and then
        // refusing, which is the worst of both -- it spends the most work
        // exactly where it will fail, and a likelihood that declines to be
        // evaluated stops an optimiser dead.
        let rounding = 16.0 * noise * (left.abs() + right.abs());
        if error.abs() <= (15.0 * tolerance).max(rounding) {
            Ok(QuadratureResult {
                value: left + right + error / 15.0,
                estimated_error: error.abs() / 15.0,
            })
        } else if depth == 0 {
            Err("BIVARIATE_QUADRATURE_DID_NOT_CONVERGE")
        } else {
            let left_result = recurse(
                function,
                lower,
                midpoint,
                left,
                tolerance / 2.0,
                noise,
                depth - 1,
            )?;
            let right_result = recurse(
                function,
                midpoint,
                upper,
                right,
                tolerance / 2.0,
                noise,
                depth - 1,
            )?;
            Ok(QuadratureResult {
                value: left_result.value + right_result.value,
                estimated_error: left_result.estimated_error + right_result.estimated_error,
            })
        }
    }
    if lower >= upper {
        return Ok(QuadratureResult {
            value: 0.0,
            estimated_error: 0.0,
        });
    }
    let estimate = simpson(function, lower, upper);
    if !estimate.is_finite() {
        return Err("BIVARIATE_QUADRATURE_NOT_FINITE");
    }
    recurse(
        function,
        lower,
        upper,
        estimate,
        tolerance,
        noise,
        maximum_depth,
    )
}

#[cfg(test)]
mod tests {
    use super::bivariate_normal_cdf;

    /// (a, b, correlation, the probability), 53 points.
    ///
    /// Computed on 29 August 2026 two independent ways -- Owen's angular form and
    /// the conditional form -- each integrated to about 1e-15 and agreeing with the
    /// other everywhere to better than 1e-12. Points whose probability falls below
    /// that are left out rather than pinned to a value nothing resolved.
    const REFERENCE: [(f64, f64, f64, f64); 53] = [
        (0.0_f64, 0.0_f64, -0.99_f64, 0.022_526_706_822_206_033_f64),
        (0.0_f64, 0.0_f64, -0.9_f64, 0.071_783_146_564_353_14_f64),
        (0.0_f64, 0.0_f64, -0.5_f64, 0.166_666_666_666_666_69_f64),
        (0.0_f64, 0.0_f64, 0.0_f64, 0.25_f64),
        (0.0_f64, 0.0_f64, 0.3_f64, 0.298_493_342_010_339_17_f64),
        (0.0_f64, 0.0_f64, 0.5_f64, 0.333_333_333_333_333_3_f64),
        (0.0_f64, 0.0_f64, 0.9_f64, 0.428_216_853_435_646_86_f64),
        (0.0_f64, 0.0_f64, 0.95_f64, 0.449_458_687_947_870_03_f64),
        (0.0_f64, 0.0_f64, 0.99_f64, 0.477_473_293_177_793_97_f64),
        (0.0_f64, 0.0_f64, 0.999_f64, 0.492_881_781_296_880_2_f64),
        (0.0_f64, 0.0_f64, 0.999_9_f64, 0.497_749_190_452_595_4_f64),
        (0.5_f64, -0.3_f64, -0.99_f64, 0.075_404_529_782_843_51_f64),
        (0.5_f64, -0.3_f64, -0.9_f64, 0.109_265_117_217_605_01_f64),
        (0.5_f64, -0.3_f64, -0.5_f64, 0.191_974_360_454_870_93_f64),
        (0.5_f64, -0.3_f64, 0.0_f64, 0.264_199_908_437_914_1_f64),
        (0.5_f64, -0.3_f64, 0.3_f64, 0.303_940_488_690_710_33_f64),
        (0.5_f64, -0.3_f64, 0.5_f64, 0.330_358_506_178_733_1_f64),
        (0.5_f64, -0.3_f64, 0.9_f64, 0.379_431_700_698_883_f64),
        (0.5_f64, -0.3_f64, 0.95_f64, 0.381_856_889_508_105_87_f64),
        (0.5_f64, -0.3_f64, 0.99_f64, 0.382_088_577_738_546_65_f64),
        (0.5_f64, -0.3_f64, 0.999_f64, 0.382_088_577_811_047_3_f64),
        (0.5_f64, -0.3_f64, 0.999_9_f64, 0.382_088_577_811_047_4_f64),
        (-1.2_f64, 0.8_f64, -0.99_f64, 2.366_517_568_415_382_7e-5_f64),
        (-1.2_f64, 0.8_f64, -0.9_f64, 0.010_976_257_490_135_394_f64),
        (-1.2_f64, 0.8_f64, -0.5_f64, 0.055_414_065_729_176_62_f64),
        (-1.2_f64, 0.8_f64, 0.0_f64, 0.090_691_539_372_028_22_f64),
        (-1.2_f64, 0.8_f64, 0.3_f64, 0.105_031_300_259_376_2_f64),
        (-1.2_f64, 0.8_f64, 0.5_f64, 0.111_380_672_271_595_7_f64),
        (-1.2_f64, 0.8_f64, 0.9_f64, 0.115_069_527_938_335_54_f64),
        (-1.2_f64, 0.8_f64, 0.95_f64, 0.115_069_670_219_308_71_f64),
        (-1.2_f64, 0.8_f64, 0.99_f64, 0.115_069_670_221_708_22_f64),
        (-1.2_f64, 0.8_f64, 0.999_f64, 0.115_069_670_221_708_23_f64),
        (-1.2_f64, 0.8_f64, 0.999_9_f64, 0.115_069_670_221_708_23_f64),
        (2.0_f64, 1.5_f64, -0.99_f64, 0.910_442_666_782_962_7_f64),
        (2.0_f64, 1.5_f64, -0.9_f64, 0.910_442_666_782_962_8_f64),
        (2.0_f64, 1.5_f64, -0.5_f64, 0.910_468_093_360_707_4_f64),
        (2.0_f64, 1.5_f64, 0.0_f64, 0.911_962_539_426_917_7_f64),
        (2.0_f64, 1.5_f64, 0.3_f64, 0.915_121_383_105_603_8_f64),
        (2.0_f64, 1.5_f64, 0.5_f64, 0.918_646_168_443_951_2_f64),
        (2.0_f64, 1.5_f64, 0.9_f64, 0.930_727_253_512_640_1_f64),
        (2.0_f64, 1.5_f64, 0.95_f64, 0.932_542_675_547_147_5_f64),
        (2.0_f64, 1.5_f64, 0.99_f64, 0.933_192_182_445_449_9_f64),
        (2.0_f64, 1.5_f64, 0.999_f64, 0.933_192_798_731_141_9_f64),
        (2.0_f64, 1.5_f64, 0.999_9_f64, 0.933_192_798_731_141_9_f64),
        (-2.5_f64, -2.0_f64, -0.5_f64, 3.033_206_837_166_693_7e-7_f64),
        (-2.5_f64, -2.0_f64, 0.0_f64, 0.000_141_270_705_515_440_2_f64),
        (-2.5_f64, -2.0_f64, 0.3_f64, 0.000_710_335_976_897_738_8_f64),
        (-2.5_f64, -2.0_f64, 0.5_f64, 0.001_559_821_950_564_552_3_f64),
        (-2.5_f64, -2.0_f64, 0.9_f64, 0.005_332_931_061_066_135_f64),
        (
            -2.5_f64,
            -2.0_f64,
            0.95_f64,
            0.005_974_929_797_870_771_5_f64,
        ),
        (
            -2.5_f64,
            -2.0_f64,
            0.99_f64,
            0.006_209_439_620_767_252_5_f64,
        ),
        (-2.5_f64, -2.0_f64, 0.999_f64, 0.006_209_665_325_776_136_f64),
        (
            -2.5_f64,
            -2.0_f64,
            0.999_9_f64,
            0.006_209_665_325_776_136_f64,
        ),
    ];

    /// The integral must be accurate everywhere, not only where the rectangle
    /// happens to be symmetric.
    ///
    /// The quadrature this replaced was measured at thresholds of nought and
    /// nought and its accuracy stated from that one place. It held there and
    /// nowhere else: at 0.5 and -0.3 with a correlation of 0.99 it was about
    /// seventy times worse than its own documentation claimed. So the grid
    /// below moves the rectangle around as well as the correlation, and the
    /// tolerance is the same everywhere on it.
    #[test]
    fn the_bivariate_integral_matches_an_independent_reference() {
        let mut worst = 0.0_f64;
        let mut worst_at = (0.0, 0.0, 0.0);
        for &(first, second, correlation, wanted) in &REFERENCE {
            let got = bivariate_normal_cdf(first, second, correlation)
                .expect("a probability above the resolvable floor must be returned");
            let error = (got - wanted).abs();
            if error > worst {
                worst = error;
                worst_at = (first, second, correlation);
            }
        }
        assert!(
            worst < 1.0e-9,
            "worst absolute error {worst:.3e} at thresholds {:?} and correlation {}, \
             against a reference good to 1e-12",
            (worst_at.0, worst_at.1),
            worst_at.2
        );
    }

    /// Only one refusal means "too small to resolve"; the rest mean "failed".
    ///
    /// `LiabilityModel` turns this integral's refusals into a log probability,
    /// and it may only substitute the floor for the one code that says the
    /// answer fell below what can be resolved. Every other code is a failure
    /// and has to reach the caller as one, or the optimiser is handed a finite
    /// number where there is no answer -- ADR 0016. This pins the code that
    /// mapping keys on, so renaming it cannot quietly turn failures into
    /// floors.
    #[test]
    fn an_invalid_correlation_is_not_reported_as_an_unresolvable_probability() {
        let error = bivariate_normal_cdf(0.5, -0.3, 1.5)
            .expect_err("a correlation outside its range is not a probability");
        assert_eq!(error, "BIVARIATE_CORRELATION_INVALID");
        assert_ne!(
            error, "BIVARIATE_PROBABILITY_UNRESOLVED",
            "a failure must not be mistakable for a probability below the floor"
        );
    }

    /// A correlation of one is what monozygotic twins give, and what a
    /// person-level component gives between two records of one person. The
    /// integral is not asked for it -- the caller clamps just below -- but it
    /// must stay accurate right up to there, because that clamp is the only
    /// thing between the model and a correlation of one.
    #[test]
    fn accuracy_holds_where_a_person_level_component_puts_it() {
        for &correlation in &[0.99_f64, 0.999, 0.9999, 0.999_999] {
            let got = bivariate_normal_cdf(0.5, -0.3, correlation)
                .expect("the near-degenerate corner must still return a probability");
            // As the correlation approaches one the rectangle tends to the
            // smaller marginal, which is Phi(-0.3).
            let limit = 0.382_088_577_811_047_4_f64;
            assert!(
                (got - limit).abs() < 1.0e-6,
                "at correlation {correlation} the integral gives {got}, \
                 where the limiting value is {limit}"
            );
        }
    }
}
