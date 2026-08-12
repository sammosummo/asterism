use asterism::{BivariateHeritabilityBoundary, BivariateModel};
use nalgebra::{DMatrix, DVector};

fn problem(mut seed: u64) -> (DMatrix<f64>, Vec<[bool; 2]>, DMatrix<f64>, DVector<f64>) {
    let pairs = 30;
    let people = 2 * pairs;
    let mut relationship = DMatrix::<f64>::identity(people, people);
    for pair in 0..pairs {
        relationship[(2 * pair, 2 * pair + 1)] = 0.5;
        relationship[(2 * pair + 1, 2 * pair)] = 0.5;
    }
    let observed: Vec<[bool; 2]> = (0..people)
        .map(|person| [person % 11 != 0, person % 7 != 0])
        .collect();
    let rows = observed
        .iter()
        .map(|mask| usize::from(mask[0]) + usize::from(mask[1]))
        .sum();
    let mut design = DMatrix::<f64>::zeros(rows, 2);
    let mut response = DVector::<f64>::zeros(rows);
    let mut next = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
    };
    let mut shared = vec![(0.0, 0.0); pairs];
    for pair in &mut shared {
        let genetic = next() + next() + next();
        *pair = (genetic, 0.6 * genetic + 0.4 * (next() + next() + next()));
    }
    let mut row = 0;
    for (person, mask) in observed.iter().enumerate() {
        let (first, second) = shared[person / 2];
        for trait_index in 0..2 {
            if mask[trait_index] {
                design[(row, trait_index)] = 1.0;
                let genetic = if trait_index == 0 { first } else { second };
                response[row] = 1.4 * genetic + 0.6 * (next() + next() + next());
                row += 1;
            }
        }
    }
    (relationship, observed, design, response)
}

#[test]
fn thirty_seed_unbalanced_ml_reml_optimisation_stress() {
    let mut exact_lower_bound = 0;
    let mut exact_upper_bound = 0;
    let mut exact_h2_upper = 0;
    for seed in 1..=30 {
        let (relationship, observed, design, response) = problem(seed);
        let model = BivariateModel::build(&relationship, &observed, &design).unwrap();
        for reml in [false, true] {
            let fit = model.fit(&response, reml).unwrap_or_else(|error| {
                panic!(
                    "seed {seed} {} fit failed: {error}",
                    if reml { "REML" } else { "ML" }
                )
            });
            assert!(fit.converged);
            assert!(
                fit.scaled_gradient < 1.0e-7,
                "seed {seed} {} scaled KKT was {}",
                if reml { "REML" } else { "ML" },
                fit.scaled_gradient
            );
            for trait_index in 0..2 {
                match fit.h2_boundary[trait_index] {
                    BivariateHeritabilityBoundary::Lower => {
                        assert_eq!(fit.h2[trait_index], 0.0);
                    }
                    BivariateHeritabilityBoundary::Interior => {
                        assert!(fit.h2[trait_index] > 0.0);
                        assert!(fit.h2[trait_index] < 1.0);
                    }
                    BivariateHeritabilityBoundary::Upper => {
                        assert_eq!(fit.h2[trait_index], 1.0);
                        exact_h2_upper += 1;
                    }
                }
            }
            if seed == 2 && !reml {
                assert_eq!(
                    fit.h2_boundary,
                    [
                        BivariateHeritabilityBoundary::Upper,
                        BivariateHeritabilityBoundary::Upper,
                    ]
                );
                assert!(fit.rho_e.is_none());
            }
            if fit.rho_g == Some(-1.0) || fit.rho_e == Some(-1.0) {
                exact_lower_bound += 1;
            }
            if fit.rho_g == Some(1.0) || fit.rho_e == Some(1.0) {
                exact_upper_bound += 1;
            }
        }
    }
    assert!(exact_lower_bound > 0);
    assert!(exact_upper_bound > 0);
    assert!(exact_h2_upper > 0);
}

#[test]
fn rescaling_both_traits_rescales_only_the_variances() {
    let (relationship, observed, design, response) = problem(7);
    let model = BivariateModel::build(&relationship, &observed, &design).unwrap();
    for reml in [false, true] {
        let baseline = model.fit(&response, reml).unwrap();
        for multiplier in [1.0e-5_f64, 1.0e5_f64] {
            let scaled_response = response.map(|value| multiplier * value);
            let scaled = model.fit(&scaled_response, reml).unwrap_or_else(|error| {
                panic!(
                    "{} fit at response multiplier {multiplier} failed: {error}",
                    if reml { "REML" } else { "ML" }
                )
            });

            assert!(baseline.converged && scaled.converged);
            assert!(scaled.scaled_gradient < 1.0e-7);
            for trait_index in 0..2 {
                let expected = baseline.total_variance[trait_index] * multiplier.powi(2);
                assert!((scaled.total_variance[trait_index] / expected - 1.0).abs() < 1.0e-5);
                assert!((scaled.h2[trait_index] - baseline.h2[trait_index]).abs() < 1.0e-5);
            }
            assert_eq!(scaled.h2_boundary, baseline.h2_boundary);
            assert_eq!(scaled.rho_g.is_some(), baseline.rho_g.is_some());
            assert_eq!(scaled.rho_e.is_some(), baseline.rho_e.is_some());
            if let (Some(left), Some(right)) = (scaled.rho_g, baseline.rho_g) {
                assert!((left - right).abs() < 1.0e-5);
            }
            if let (Some(left), Some(right)) = (scaled.rho_e, baseline.rho_e) {
                assert!((left - right).abs() < 1.0e-5);
            }
            let effective_rows = response.len() - usize::from(reml) * design.ncols();
            let expected_loglik = baseline.loglik - effective_rows as f64 * multiplier.ln();
            assert!((scaled.loglik - expected_loglik).abs() < 1.0e-7);
        }
    }
}

#[test]
fn a_shared_fixed_effect_survives_internal_trait_standardisation() {
    let (relationship, observed, _trait_specific_design, mut response) = problem(13);
    let shared_design = DMatrix::<f64>::from_element(response.len(), 1, 1.0);
    let mut row = 0;
    for mask in &observed {
        for trait_index in 0..2 {
            if mask[trait_index] {
                if trait_index == 1 {
                    response[row] *= 7.0;
                }
                row += 1;
            }
        }
    }
    let model = BivariateModel::build(&relationship, &observed, &shared_design).unwrap();
    for reml in [false, true] {
        let fit = model.fit(&response, reml).unwrap_or_else(|error| {
            panic!(
                "{} shared-design fit failed: {error}",
                if reml { "REML" } else { "ML" }
            )
        });
        assert!(fit.converged);
        assert!(fit.scaled_gradient < 1.0e-7);
    }
}

#[test]
fn relationships_only_among_unmeasured_people_do_not_identify_the_model() {
    let people = 8;
    let mut relationship = DMatrix::<f64>::identity(people, people);
    relationship[(6, 7)] = 0.5;
    relationship[(7, 6)] = 0.5;
    let observed = vec![[true, true]; 6]
        .into_iter()
        .chain(vec![[false, false]; 2])
        .collect::<Vec<_>>();
    let mut design = DMatrix::<f64>::zeros(12, 2);
    for row in 0..12 {
        design[(row, row % 2)] = 1.0;
    }

    assert_eq!(
        BivariateModel::build(&relationship, &observed, &design)
            .err()
            .unwrap(),
        "BIVARIATE_COVARIANCE_NOT_IDENTIFIABLE"
    );
}

#[test]
fn each_trait_and_the_cross_trait_covariance_must_be_identifiable() {
    let people = 12;
    let mut relationship = DMatrix::<f64>::identity(people, people);
    for pair in 0..3 {
        relationship[(2 * pair, 2 * pair + 1)] = 0.5;
        relationship[(2 * pair + 1, 2 * pair)] = 0.5;
    }
    let observed = (0..people)
        .map(|person| [person < 6, person >= 6])
        .collect::<Vec<_>>();
    let mut design = DMatrix::<f64>::zeros(people, 2);
    for row in 0..people {
        design[(row, usize::from(row >= 6))] = 1.0;
    }

    assert_eq!(
        BivariateModel::build(&relationship, &observed, &design)
            .err()
            .unwrap(),
        "BIVARIATE_COVARIANCE_NOT_IDENTIFIABLE"
    );
}

#[test]
fn exact_zero_genetic_variance_removes_the_genetic_correlation() {
    let pairs = 40;
    let people = 2 * pairs;
    let mut relationship = DMatrix::<f64>::identity(people, people);
    for pair in 0..pairs {
        relationship[(2 * pair, 2 * pair + 1)] = 0.5;
        relationship[(2 * pair + 1, 2 * pair)] = 0.5;
    }
    let observed = vec![[true, true]; people];
    let mut design = DMatrix::<f64>::zeros(2 * people, 2);
    let mut response = DVector::<f64>::zeros(2 * people);
    let mut seed = 9_173_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
    };
    for pair in 0..pairs {
        let first = next() + next() + next();
        let second = 0.6 * first + 0.4 * (next() + next() + next());
        for (within_pair, sign) in [(0, 1.0), (1, -1.0)] {
            let person = 2 * pair + within_pair;
            design[(2 * person, 0)] = 1.0;
            design[(2 * person + 1, 1)] = 1.0;
            response[2 * person] = sign * first;
            response[2 * person + 1] = sign * second;
        }
    }

    let model = BivariateModel::build(&relationship, &observed, &design).unwrap();
    for reml in [false, true] {
        let fit = model.fit(&response, reml).unwrap();
        assert_eq!(
            fit.h2_boundary,
            [
                BivariateHeritabilityBoundary::Lower,
                BivariateHeritabilityBoundary::Lower,
            ]
        );
        assert_eq!(fit.h2, [0.0, 0.0]);
        assert!(fit.rho_g.is_none());
        assert!(fit.rho_e.is_some());
        assert!(fit.scaled_gradient < 1.0e-7);
    }
}
