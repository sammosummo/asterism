//! The upper tail of a weighted sum of chi-squares, which is what a
//! variance-component score test needs.
//!
//! A score test for one variance component gives a statistic distributed as
//! `Q = sum_j lambda_j chi2_1`, where the weights are the eigenvalues of the
//! tested matrix after projection. There is no closed form in general, and the
//! whole difficulty of a gene-based test lives here: a scan reads p-values
//! around 1e-6, so an approximation that is fine at 0.05 is worthless.
//!
//! # Why this recipe
//!
//! Three published routes were compared against each other and against exact
//! chi-square values, and each has a regime where it fails:
//!
//! - **Davies' inversion** as published refuses to run at tight accuracy
//!   settings, and at its own default accuracy of 1e-4 it returns nonsense in
//!   the tail — nought where the answer is 5.7e-7, which is exactly the case a
//!   burden kernel produces.
//! - **Ruben's series** is exact for equal weights but collapses when the
//!   weights span a wide range: on twenty weights spanning a hundredfold it
//!   returned 1e-126 where the answer is 1.4e-6.
//! - **Gil-Pelaez inversion**, used here, agreed with an independent Ruben
//!   oracle to 1e-10 wherever that oracle converged, and agreed with Davies in
//!   the case where Ruben failed.
//!
//! # The recipe
//!
//! ```text
//! P(Q > q) = 1/2 + (1/pi) Int_0^inf sin(theta(t)) / (t rho(t)) dt
//! theta(t) = (1/2) sum_j atan(2 lambda_j t) - q t
//! rho(t)   = exp( (1/4) sum_j log(1 + 4 lambda_j^2 t^2) )
//! ```
//!
//! The integrand oscillates while its amplitude decays slowly, so integrating
//! it as one lump needs an impractical range. Splitting at the zeros of
//! `sin(theta)` turns it into an alternating series whose terms shrink, and
//! truncating there costs about one term rather than the whole remaining
//! envelope.
//!
//! Two cases have exact answers and are taken directly: a single weight, and
//! weights that are all equal. The first is what a burden kernel gives, and it
//! is the case the published inversion handles worst.
//!
//! # Accuracy
//!
//! The probability is recovered as `1/2 + integral`, so a small tail is a
//! difference of two nearly equal numbers. About six digits are lost by the
//! time the tail reaches 1e-6, which leaves enough in binary64 for a gene scan
//! and not enough for a genome-wide single-variant threshold. The returned
//! record carries the size of the last term kept, so a caller can see how far
//! the series had settled rather than trusting it.

use crate::deviance::chi2_upper_tail;

#[cfg(feature = "python")]
pub mod python;

/// Weights below this share of the largest are dropped: they are numerical
/// residue from an eigendecomposition rather than components of the statistic.
const WEIGHT_FLOOR: f64 = 1e-12;
/// Two weights within this relative distance are treated as equal, which lets
/// the exact chi-square answer be used.
const EQUAL_TOLERANCE: f64 = 1e-12;
// A p-value is never read to twelve figures, and the alternating series has to
// run a long way to settle that far. Ten is past anything downstream uses.
const RELATIVE_TOLERANCE: f64 = 1e-10;
const MAXIMUM_SEGMENTS: usize = 40_000;
/// Below this the inversion is losing too many digits to cancellation for the
/// answer to mean anything, and the caller is told so rather than shown it.
const CANCELLATION_FLOOR: f64 = 1e-11;
/// How many panels a band is split into. They double in width from the band
/// start, so this many covers a dynamic range of 2^40 -- far more than the
/// widest band a plausible statistic produces.
const BAND_PANELS: usize = 40;

const GAUSS_NODES: [f64; 20] = [
    -0.993_128_599_185_094_9,
    -0.963_971_927_277_913_8,
    -0.912_234_428_251_326,
    -0.839_116_971_822_218_8,
    -0.746_331_906_460_150_8,
    -0.636_053_680_726_515,
    -0.510_867_001_950_827_1,
    -0.373_706_088_715_419_55,
    -0.227_785_851_141_645_07,
    -0.076_526_521_133_497_34,
    0.076_526_521_133_497_34,
    0.227_785_851_141_645_07,
    0.373_706_088_715_419_55,
    0.510_867_001_950_827_1,
    0.636_053_680_726_515,
    0.746_331_906_460_150_8,
    0.839_116_971_822_218_8,
    0.912_234_428_251_326,
    0.963_971_927_277_913_8,
    0.993_128_599_185_094_9,
];
const GAUSS_WEIGHTS: [f64; 20] = [
    0.017_614_007_139_152_753,
    0.040_601_429_800_386_12,
    0.062_672_048_334_109_05,
    0.083_276_741_576_704_74,
    0.101_930_119_817_240_5,
    0.118_194_531_961_518_33,
    0.131_688_638_449_176_55,
    0.142_096_109_318_382_04,
    0.149_172_986_472_603_82,
    0.152_753_387_130_725_9,
    0.152_753_387_130_725_9,
    0.149_172_986_472_603_82,
    0.142_096_109_318_382_04,
    0.131_688_638_449_176_55,
    0.118_194_531_961_518_33,
    0.101_930_119_817_240_5,
    0.083_276_741_576_704_74,
    0.062_672_048_334_109_05,
    0.040_601_429_800_386_12,
    0.017_614_007_139_152_753,
];

/// A tail probability and how it was arrived at.
#[derive(Clone, Copy, Debug)]
pub struct MixtureTail {
    pub probability: f64,
    /// Which recipe answered: an exact chi-square where one applies, otherwise
    /// the inversion.
    pub method: &'static str,
    /// The size of the last term kept, as a share of the probability. For an
    /// exact answer this is nought.
    pub settled_to: f64,
    /// False where cancellation has eaten too many digits for the value to be
    /// read as a probability.
    pub trustworthy: bool,
}

fn phase(t: f64, weights: &[f64], q: f64) -> f64 {
    0.5 * weights.iter().map(|w| (2.0 * w * t).atan()).sum::<f64>() - q * t
}

fn log_amplitude(t: f64, weights: &[f64]) -> f64 {
    0.25 * weights
        .iter()
        .map(|w| (4.0 * w * w * t * t).ln_1p())
        .sum::<f64>()
}

fn integrand(t: f64, weights: &[f64], q: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    phase(t, weights, q).sin() * (-log_amplitude(t, weights)).exp() / t
}

/// Integrate one band with a fixed Gauss-Legendre rule.
fn gauss_panel(lower: f64, upper: f64, weights: &[f64], q: f64) -> f64 {
    let middle = 0.5 * (lower + upper);
    let half = 0.5 * (upper - lower);
    GAUSS_NODES
        .iter()
        .zip(GAUSS_WEIGHTS)
        .map(|(node, weight)| weight * integrand(middle + half * node, weights, q))
        .sum::<f64>()
        * half
}

/// Integrate one band on panels that are graded towards its start.
///
/// **A band is not automatically narrow.** Its width is set by where the phase
/// next crosses a multiple of pi, which for a small statistic is far away. The
/// integrand meanwhile is flat near the origin -- it tends to `sum(lambda) - q`
/// there rather than diverging -- and only decays once the amplitude `rho`
/// starts to grow, at around `1 / (2 max(lambda))`. So a wide band holds nearly
/// all of its mass in a small part of itself, and one fixed rule spread across
/// the whole of it puts almost no nodes where the function actually is.
///
/// The error that causes is not subtle: it made the returned survival function
/// *increase* with the statistic across the range where bands are widest, which
/// is impossible for a survival function and is what this grading fixes.
///
/// Panels double in width from the band start, so the resolution is fine where
/// the integrand lives and coarse where it has already decayed, at a fixed cost
/// rather than a recursive one.
fn graded_band(lower: f64, upper: f64, weights: &[f64], q: f64, fine: f64) -> f64 {
    let width = upper - lower;
    if !(width > 0.0) {
        return 0.0;
    }
    // Grade only as far as the band is actually wider than the scale the
    // integrand varies on. A narrow band needs one panel, and most bands are
    // narrow: their width is about `pi / q`, so grading every band to the same
    // depth would spend forty panels resolving a function that is smooth
    // across the whole of it.
    // The doublings it takes to grow the finest panel out to the whole band,
    // capped. Counted rather than taken from a logarithm, so there is no
    // rounding to reason about and no conversion that could go the wrong way
    // for a width that is not finite.
    let mut panels = 1usize;
    let mut span = fine;
    while span < width && panels < BAND_PANELS {
        span *= 2.0;
        panels += 1;
    }
    if panels == 1 {
        return gauss_panel(lower, upper, weights, q);
    }
    let mut total = 0.0;
    let mut edge = lower;
    for panel in 0..panels {
        // Widths 2^-K, 2^-K, 2^-(K-1), ... of the band, summing to the whole.
        // `panels` is at most `BAND_PANELS`, so the exponent always fits; the
        // fallback is unreachable and only spares a cast that could not be
        // read as safe on its own.
        let steps = if panel == 0 {
            panels - 1
        } else {
            panels - panel
        };
        let share = 0.5f64.powi(i32::try_from(steps).unwrap_or(i32::MAX));
        let next = if panel == panels - 1 {
            upper
        } else {
            edge + width * share
        };
        total += gauss_panel(edge, next, weights, q);
        edge = next;
    }
    total
}

/// `P(sum_j weights_j * chi2_1 > q)`.
///
/// # Errors
///
/// Refuses a non-finite input, and a negative weight, which cannot arise from
/// the projection of a positive semi-definite matrix and means the caller has
/// handed over something other than eigenvalues.
pub fn weighted_chi2_upper_tail(q: f64, weights: &[f64]) -> Result<MixtureTail, &'static str> {
    if !q.is_finite() || weights.iter().any(|w| !w.is_finite()) {
        return Err("MIXTURE_TAIL_NOT_FINITE");
    }
    if weights.iter().any(|w| *w < 0.0) {
        return Err("MIXTURE_TAIL_WEIGHT_NEGATIVE");
    }
    let largest = weights.iter().fold(0.0_f64, |a, b| a.max(*b));
    let kept: Vec<f64> = weights
        .iter()
        .copied()
        .filter(|w| *w > WEIGHT_FLOOR * largest)
        .collect();

    if kept.is_empty() {
        // The statistic is identically nought.
        return Ok(MixtureTail {
            probability: if q <= 0.0 { 1.0 } else { 0.0 },
            method: "degenerate",
            settled_to: 0.0,
            trustworthy: true,
        });
    }
    if q <= 0.0 {
        return Ok(MixtureTail {
            probability: 1.0,
            method: "below_support",
            settled_to: 0.0,
            trustworthy: true,
        });
    }
    if kept.len() == 1 {
        return Ok(MixtureTail {
            probability: chi2_upper_tail(q / kept[0], 1.0),
            method: "exact_single_weight",
            settled_to: 0.0,
            trustworthy: true,
        });
    }
    if kept
        .iter()
        .all(|w| (w - kept[0]).abs() <= EQUAL_TOLERANCE * kept[0])
    {
        return Ok(MixtureTail {
            probability: chi2_upper_tail(q / kept[0], kept.len() as f64),
            method: "exact_equal_weights",
            settled_to: 0.0,
            trustworthy: true,
        });
    }

    // March forward, cutting the integral at every crossing of a multiple of
    // pi so that successive pieces alternate in sign and shrink.
    let mut total = 0.0_f64;
    let mut lower = 0.0_f64;
    let mut band = (phase(1e-14, &kept, q) / std::f64::consts::PI).floor();
    // The phase turns over on the scale where the amplitude starts to grow, so
    // the first probe must be fine enough to see that; thereafter the phase
    // falls at rate `q`, making a band about `pi / q` wide. A probe is never
    // allowed past an eighth of that, so it cannot step over a whole band, and
    // it grows geometrically until it does -- which is what lets a small
    // statistic, whose first crossing sits at `r pi / (4 q)`, be reached at
    // all. Fixed steps could not: they left the integral unevaluated and the
    // answer at one half.
    let largest = kept.iter().fold(0.0_f64, |a, b| a.max(*b));
    let first_probe = (0.25 / largest).min(0.5);
    let widest_probe = (std::f64::consts::PI / q / 8.0).max(first_probe);
    let mut step = first_probe;
    let mut last_term = f64::INFINITY;
    let mut segments = 0usize;

    while segments < MAXIMUM_SEGMENTS {
        let mut upper = lower;
        let mut crossed = false;
        let mut probe = step;
        for _ in 0..8_000 {
            let next = upper + probe;
            if !next.is_finite() {
                break;
            }
            if (phase(next, &kept, q) / std::f64::consts::PI).floor() != band {
                let (mut low, mut high) = (upper, next);
                for _ in 0..80 {
                    let middle = 0.5 * (low + high);
                    if (phase(middle, &kept, q) / std::f64::consts::PI).floor() == band {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                upper = high;
                crossed = true;
                break;
            }
            upper = next;
            probe = (probe * 1.05).min(widest_probe);
        }
        if !crossed {
            break;
        }

        let term = graded_band(lower, upper, &kept, q, first_probe);
        total += term;
        last_term = term.abs();
        segments += 1;

        band = (phase(upper + 1e-13, &kept, q) / std::f64::consts::PI).floor();
        step = (upper - lower).max(step).min(widest_probe);
        lower = upper;

        let running = (0.5 + total / std::f64::consts::PI)
            .abs()
            .max(f64::MIN_POSITIVE);
        if segments > 4 && last_term / std::f64::consts::PI < RELATIVE_TOLERANCE * running {
            break;
        }
    }

    let probability = (0.5 + total / std::f64::consts::PI).clamp(0.0, 1.0);
    let settled_to = if probability > 0.0 {
        (last_term / std::f64::consts::PI) / probability
    } else {
        f64::INFINITY
    };
    Ok(MixtureTail {
        probability,
        method: "gil_pelaez_inversion",
        settled_to,
        trustworthy: probability > CANCELLATION_FLOOR && settled_to < 1e-3,
    })
}

#[cfg(test)]
mod tests {
    use super::weighted_chi2_upper_tail;
    use crate::deviance::chi2_upper_tail;

    /// Equal weights are an ordinary chi-square, and a single weight is one
    /// scaled. Both are taken exactly rather than integrated.
    #[test]
    fn the_two_exact_cases_are_taken_exactly() {
        let one = weighted_chi2_upper_tail(25.0, &[1.0]).expect("valid");
        assert_eq!(one.method, "exact_single_weight");
        assert!((one.probability - chi2_upper_tail(25.0, 1.0)).abs() < 1e-15);
        // The published inversion returns nought here; this must not.
        assert!(one.probability > 5.0e-7 && one.probability < 6.0e-7);

        let equal = weighted_chi2_upper_tail(60.0, &[1.0; 20]).expect("valid");
        assert_eq!(equal.method, "exact_equal_weights");
        assert!((equal.probability - chi2_upper_tail(60.0, 20.0)).abs() < 1e-18);

        // A scaled equal set is the same chi-square with the argument scaled.
        let scaled = weighted_chi2_upper_tail(120.0, &[2.0; 20]).expect("valid");
        assert!((scaled.probability - chi2_upper_tail(60.0, 20.0)).abs() < 1e-18);
    }

    /// Values from an independent Ruben series expansion, which shares no
    /// machinery with characteristic-function inversion.
    #[test]
    fn the_inversion_matches_an_independent_series() {
        for (q, weights, expected) in [
            (1.5_f64, vec![3.0_f64, 2.0, 1.0], 8.468_441e-01_f64),
            (20.0, vec![3.0, 2.0, 1.0], 2.512_656e-02),
            (60.0, vec![3.0, 2.0, 1.0], 1.770_821e-05),
            (30.0, vec![2.0, 1.0], 1.574_936e-04),
        ] {
            let got = weighted_chi2_upper_tail(q, &weights).expect("valid");
            assert_eq!(got.method, "gil_pelaez_inversion");
            let relative = (got.probability - expected).abs() / expected;
            assert!(
                relative < 1e-4,
                "q = {q}: got {} against {expected}, relative {relative:.2e}",
                got.probability
            );
            assert!(got.trustworthy, "q = {q} was not trustworthy");
        }
    }

    /// A probability falls as the statistic grows, whatever the weights.
    ///
    /// **The small end of this range is the point.** An earlier version of
    /// this test started at `q = 0.5` and passed while the tail was rising
    /// with `q` below that -- the bands are widest there, and a single
    /// quadrature rule across a wide band misses where the integrand lives.
    #[test]
    fn the_tail_decreases_and_stays_a_probability() {
        let weights = [5.0, 2.0, 1.0, 0.5];
        let mut previous = 1.0;
        for q in [
            1e-5, 1e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 2.0, 8.0, 20.0, 50.0,
        ] {
            let got = weighted_chi2_upper_tail(q, &weights).expect("valid");
            assert!((0.0..=1.0).contains(&got.probability));
            assert!(got.probability < previous, "not decreasing at q = {q}");
            previous = got.probability;
        }
    }

    /// Scaling the statistic and every weight together cannot change the
    /// answer, since `P(Q > q)` and `P(cQ > cq)` are the same statement.
    ///
    /// This needs no external reference, which is what makes it worth having:
    /// it caught the tail returning exactly one half -- the value it takes when
    /// the integral is never evaluated at all -- once the scaling pushed the
    /// first phase crossing beyond where the search reached.
    #[test]
    fn rescaling_the_whole_problem_changes_nothing() {
        let base = [3.0_f64, 2.0, 1.0];
        let reference = weighted_chi2_upper_tail(20.0, &base).expect("valid");
        for scale in [1e2, 1.0, 1e-2, 1e-4, 1e-6, 1e-8] {
            let scaled: Vec<f64> = base.iter().map(|w| w * scale).collect();
            let got = weighted_chi2_upper_tail(20.0 * scale, &scaled).expect("valid");
            assert!(
                (got.probability - reference.probability).abs() < 1e-12,
                "scale {scale}: {} against {}",
                got.probability,
                reference.probability
            );
            assert!(got.trustworthy, "scale {scale} was not trustworthy");
        }
    }

    /// A statistic far below the weights is nearly certain to be exceeded, and
    /// one far above them is not. Both ends were wrong before the bands were
    /// graded: the small end returned a half and the large end a rising curve.
    #[test]
    fn the_two_ends_of_the_range_are_the_right_way_round() {
        let weights = [5.0, 2.0, 1.0, 0.5];
        let tiny = weighted_chi2_upper_tail(1e-6, &weights).expect("valid");
        assert!(tiny.probability > 0.999, "{}", tiny.probability);
        let huge = weighted_chi2_upper_tail(400.0, &weights).expect("valid");
        assert!(huge.probability < 1e-6, "{}", huge.probability);
    }

    /// Numerical residue from an eigendecomposition is dropped, and a
    /// statistic with nothing in it is not a probability of a half.
    #[test]
    fn residual_eigenvalues_are_dropped_and_a_degenerate_statistic_is_flat() {
        let with_residue = weighted_chi2_upper_tail(20.0, &[3.0, 2.0, 1.0, 1e-18]).expect("valid");
        let without = weighted_chi2_upper_tail(20.0, &[3.0, 2.0, 1.0]).expect("valid");
        assert!((with_residue.probability - without.probability).abs() < 1e-12);

        let nothing = weighted_chi2_upper_tail(1.0, &[0.0, 0.0]).expect("valid");
        assert_eq!(nothing.method, "degenerate");
        assert_eq!(nothing.probability, 0.0);
    }

    /// Eigenvalues cannot be negative, and nothing here is finite by accident.
    #[test]
    fn malformed_weights_are_refused() {
        assert_eq!(
            weighted_chi2_upper_tail(1.0, &[1.0, -0.5]).err(),
            Some("MIXTURE_TAIL_WEIGHT_NEGATIVE")
        );
        assert_eq!(
            weighted_chi2_upper_tail(f64::NAN, &[1.0]).err(),
            Some("MIXTURE_TAIL_NOT_FINITE")
        );
    }
}
