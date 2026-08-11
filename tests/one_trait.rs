//! One-trait fits against simulated data whose heritability is known.
//!
//! These are not agreement tests. Agreement with another implementation proves
//! fidelity and never correctness (`docs/adr/0006`), so what is checked here is
//! recovery of a truth the simulation controls, plus the behaviour the decision
//! record requires at the boundary.
//!
//! The design throughout is sibling pairs, which is the least informative
//! pedigree in common use and therefore the honest one to set tolerances
//! against. With `m` pairs the phenotypic correlation between siblings is h²/2,
//! so its standard error is about (1 − ρ²)/√m and the standard error of h² is
//! roughly twice that: near 0.06 at 1,000 pairs. Tolerances below are stated in
//! multiples of that, never fitted to the answer that came out.

use asterism::{Boundary, PreparedModel};
use nalgebra::{DMatrix, DVector};

/// splitmix64: a deterministic stream, so a failure is always reproducible.
struct Stream(u64);

impl Stream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        // 53 bits of mantissa, open at both ends so the log below is safe.
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// A relationship matrix for `pairs` unrelated sibling pairs.
fn sibling_relationship(pairs: usize) -> DMatrix<f64> {
    let n = 2 * pairs;
    let mut k = DMatrix::<f64>::identity(n, n);
    for pair in 0..pairs {
        let (a, b) = (2 * pair, 2 * pair + 1);
        k[(a, b)] = 0.5;
        k[(b, a)] = 0.5;
    }
    k
}

/// One trait with a grand mean, additive variance `h2` and residual `1 - h2`.
fn simulate(pairs: usize, h2: f64, mean: f64, seed: u64) -> DVector<f64> {
    let mut stream = Stream(seed);
    let mut y = Vec::with_capacity(2 * pairs);
    let additive = h2.sqrt();
    let residual = (1.0 - h2).sqrt();
    for _ in 0..pairs {
        // Two siblings share half their additive variance.
        let shared = stream.normal();
        let first = stream.normal();
        let second = stream.normal();
        let g1 = additive * (0.5f64.sqrt() * shared + 0.5f64.sqrt() * first);
        let g2 = additive * (0.5f64.sqrt() * shared + 0.5f64.sqrt() * second);
        y.push(mean + g1 + residual * stream.normal());
        y.push(mean + g2 + residual * stream.normal());
    }
    DVector::from_vec(y)
}

fn intercept(n: usize) -> DMatrix<f64> {
    DMatrix::from_element(n, 1, 1.0)
}

fn prepare(pairs: usize) -> PreparedModel {
    let k = sibling_relationship(pairs);
    let x = intercept(2 * pairs);
    PreparedModel::build(&x, &k, None).expect("a sibling design is valid")
}

#[test]
fn the_relationship_matrix_splits_into_one_block_per_family() {
    let model = prepare(500);
    assert_eq!(model.observations(), 1000);
    assert_eq!(model.fixed_effects(), 1);
    // Every 2x2 sibling block has eigenvalues 1.5 and 0.5.
    assert!((model.min_eigenvalue() - 0.5).abs() < 1e-12);
}

#[test]
fn a_moderate_heritability_is_recovered_by_reml() {
    let model = prepare(1000);
    let y = simulate(1000, 0.5, 3.0, 20260811);
    let fit = model.fit_one_trait(&y, true);

    assert!(fit.converged);
    assert_eq!(fit.boundary, Boundary::Interior);
    // Three standard errors at this design, stated in advance.
    assert!(
        (fit.h2 - 0.5).abs() < 0.18,
        "h2 = {} is more than three standard errors from 0.5",
        fit.h2
    );
    // The grand mean is estimated far more precisely than the variance ratio.
    assert!((fit.beta[0] - 3.0).abs() < 0.1, "mean = {}", fit.beta[0]);
    assert!((fit.total_variance - 1.0).abs() < 0.15, "sigma2 = {}", fit.total_variance);
    assert!(fit.loglik.is_finite());
}

#[test]
fn ml_and_reml_both_run_and_differ_by_less_than_the_standard_error() {
    let model = prepare(1000);
    let y = simulate(1000, 0.5, 0.0, 20260812);
    let reml = model.fit_one_trait(&y, true);
    let ml = model.fit_one_trait(&y, false);

    assert!(reml.converged && ml.converged);
    // One fixed effect against two thousand observations, so the restricted and
    // unrestricted likelihoods disagree systematically but only slightly. A
    // large gap here would mean something other than the degrees of freedom.
    assert!(
        (reml.h2 - ml.h2).abs() < 0.06,
        "reml {} vs ml {}",
        reml.h2,
        ml.h2
    );
    // REML's likelihood is on n - p degrees of freedom, so the two numbers are
    // not comparable to each other and neither should be used as if they were.
    assert!(reml.loglik.is_finite() && ml.loglik.is_finite());
}

#[test]
fn the_interval_contains_the_estimate_and_keeps_its_shape() {
    let model = prepare(1000);
    let y = simulate(1000, 0.5, 0.0, 20260813);
    let fit = model.fit_one_trait(&y, true);

    assert!(fit.interval.lower <= fit.h2 && fit.h2 <= fit.interval.upper);
    assert!(fit.interval.lower >= 0.0 && fit.interval.upper <= 1.0);
    // A moderate heritability at this size should exclude zero, and the width
    // should be of the order of four standard errors.
    let width = fit.interval.upper - fit.interval.lower;
    assert!(width > 0.05 && width < 0.6, "width = {width}");
}

#[test]
fn strong_heritability_is_rejected_against_no_additive_variance() {
    let model = prepare(1000);
    let y = simulate(1000, 0.8, 0.0, 20260814);
    let fit = model.fit_one_trait(&y, true);
    let test = fit.test.expect("a converged fit carries the test");

    assert!(test.statistic > 0.0);
    assert!(
        test.p_value < 1e-6,
        "p = {} for a truth of 0.8 at 1,000 pairs",
        test.p_value
    );
    // Half the null's mass sits at zero, so the p-value can never exceed a half
    // once the statistic is positive.
    assert!(test.p_value <= 0.5);
}

#[test]
fn no_additive_variance_is_not_rejected_when_there_is_none() {
    let model = prepare(1000);
    let y = simulate(1000, 0.0, 0.0, 20260815);
    let fit = model.fit_one_trait(&y, true);
    let test = fit.test.expect("a converged fit carries the test");

    assert!(
        test.p_value > 0.05,
        "p = {} for a truth of exactly zero",
        test.p_value
    );
}

#[test]
fn a_fit_on_the_lower_bound_reports_state_and_withholds_the_standard_error() {
    // A response with no family structure at all drives the estimate to zero.
    let model = prepare(300);
    let mut stream = Stream(20260816);
    let y = DVector::from_iterator(600, (0..600).map(|_| stream.normal()));
    let fit = model.fit_one_trait(&y, true);

    if fit.boundary == Boundary::Lower {
        assert_eq!(fit.h2, 0.0);
        // Absent, never NaN and never zero (`docs/adr/0005`).
        assert!(fit.standard_error.is_none());
        // The interval keeps the shape it always has.
        assert_eq!(fit.interval.lower, 0.0);
        assert!(fit.interval.lower_limited);
        assert!(fit.interval.contains_lower_bound.is_some());
        // The test is degenerate here and must say so rather than invent a
        // statistic: the fitted model is the null.
        let test = fit.test.expect("a converged fit carries the test");
        assert!(test.statistic < 1e-9);
        assert!((test.p_value - 1.0).abs() < 1e-12);
    } else {
        // Not landing on the bound is a legitimate outcome for one draw; what
        // must not happen is landing there and reporting a standard error.
        assert_eq!(fit.boundary, Boundary::Interior);
        assert!(fit.standard_error.is_some());
    }
}

#[test]
fn an_interior_fit_carries_a_standard_error_of_a_believable_size() {
    let model = prepare(1000);
    let y = simulate(1000, 0.5, 0.0, 20260817);
    let fit = model.fit_one_trait(&y, true);

    assert_eq!(fit.boundary, Boundary::Interior);
    let se = fit.standard_error.expect("an interior fit has one");
    // The design's own standard error is near 0.06; anything outside this range
    // means the observed information is being computed wrongly.
    assert!(se > 0.01 && se < 0.25, "se = {se}");
}

#[test]
fn validation_refuses_what_it_should() {
    let n = 10;
    let good_k = sibling_relationship(5);
    let good_x = intercept(n);

    let mut asymmetric = good_k.clone();
    asymmetric[(0, 1)] = 0.4;
    assert_eq!(
        PreparedModel::build(&good_x, &asymmetric, None).err().unwrap(),
        "PREPARE_K_ASYMMETRIC"
    );

    let wrong_shape = sibling_relationship(4);
    assert_eq!(
        PreparedModel::build(&good_x, &wrong_shape, None).err().unwrap(),
        "PREPARE_SHAPE_MISMATCH"
    );

    let mut duplicated = DMatrix::<f64>::zeros(n, 2);
    for i in 0..n {
        duplicated[(i, 0)] = 1.0;
        duplicated[(i, 1)] = 1.0;
    }
    assert_eq!(
        PreparedModel::build(&duplicated, &good_k, None).err().unwrap(),
        "PREPARE_X_RANK_DEFICIENT"
    );

    let mut not_finite = good_x.clone();
    not_finite[(0, 0)] = f64::NAN;
    assert_eq!(
        PreparedModel::build(&not_finite, &good_k, None).err().unwrap(),
        "PREPARE_X_NOT_FINITE"
    );

    let mut not_psd = DMatrix::<f64>::identity(n, n);
    not_psd[(0, 1)] = 2.0;
    not_psd[(1, 0)] = 2.0;
    assert_eq!(
        PreparedModel::build(&good_x, &not_psd, None).err().unwrap(),
        "PREPARE_K_NOT_PSD"
    );
}

#[test]
fn the_objective_never_depends_on_what_the_response_looks_like() {
    // A binary-looking response is fitted as quantitative and says so. It is
    // never silently given a different likelihood.
    let model = prepare(50);
    let y = DVector::from_iterator(100, (0..100).map(|i| f64::from(i % 2)));
    let fit = model.fit_one_trait(&y, true);
    assert!(
        fit.warnings.iter().any(|w| w.contains("distinct values")),
        "warnings were {:?}",
        fit.warnings
    );
}
