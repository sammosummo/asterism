use asterism::BivariateModel;
use nalgebra::{DMatrix, DVector};

fn valid_problem() -> (DMatrix<f64>, Vec<[bool; 2]>, DMatrix<f64>) {
    let mut relationship = DMatrix::<f64>::identity(4, 4);
    relationship[(0, 1)] = 0.5;
    relationship[(1, 0)] = 0.5;
    relationship[(2, 3)] = 0.5;
    relationship[(3, 2)] = 0.5;
    let observed = vec![[true, true], [true, false], [false, true], [true, true]];
    let design = DMatrix::from_fn(6, 2, |row, column| {
        let trait_index = [0, 1, 0, 1, 0, 1][row];
        f64::from(trait_index == column)
    });
    (relationship, observed, design)
}

#[test]
fn bivariate_build_rejects_an_asymmetric_relationship_matrix() {
    let (mut relationship, observed, design) = valid_problem();
    relationship[(0, 1)] = 0.4;

    assert_eq!(
        BivariateModel::build(&relationship, &observed, &design)
            .err()
            .unwrap(),
        "BIVARIATE_RELATIONSHIP_ASYMMETRIC"
    );
}

#[test]
fn bivariate_build_validates_finiteness_psd_and_design_rank() {
    let (relationship, observed, design) = valid_problem();

    let mut nonfinite_relationship = relationship.clone();
    nonfinite_relationship[(0, 0)] = f64::NAN;
    assert_eq!(
        BivariateModel::build(&nonfinite_relationship, &observed, &design)
            .err()
            .unwrap(),
        "BIVARIATE_RELATIONSHIP_NOT_FINITE"
    );

    let mut indefinite_relationship = relationship.clone();
    indefinite_relationship[(0, 1)] = 1.0 + 1.0e-6;
    indefinite_relationship[(1, 0)] = 1.0 + 1.0e-6;
    assert_eq!(
        BivariateModel::build(&indefinite_relationship, &observed, &design)
            .err()
            .unwrap(),
        "BIVARIATE_RELATIONSHIP_NOT_PSD"
    );

    let mut nonfinite_design = design.clone();
    nonfinite_design[(0, 0)] = f64::INFINITY;
    assert_eq!(
        BivariateModel::build(&relationship, &observed, &nonfinite_design)
            .err()
            .unwrap(),
        "BIVARIATE_DESIGN_NOT_FINITE"
    );

    let rank_deficient_design = DMatrix::<f64>::from_element(6, 2, 1.0);
    assert_eq!(
        BivariateModel::build(&relationship, &observed, &rank_deficient_design)
            .err()
            .unwrap(),
        "BIVARIATE_DESIGN_RANK_DEFICIENT"
    );
}

#[test]
fn the_prepared_bivariate_model_owns_the_observation_pattern() {
    let (relationship, observed, design) = valid_problem();
    let model = BivariateModel::build(&relationship, &observed, &design).unwrap();
    let response = DVector::from_vec(vec![0.2, 0.1, -0.3, 0.4, 0.7, -0.2]);
    let theta = [1.0, 0.8, 0.4, 0.5, 0.2, -0.1];

    let (objective, gradient) = model.objective_at(&theta, &response, true).unwrap();
    assert!(objective.is_finite());
    assert!(gradient.iter().all(|value| value.is_finite()));
}

#[test]
fn reml_value_is_invariant_to_fixed_effect_basis_scaling() {
    let (relationship, observed, design) = valid_problem();
    let mut rescaled_design = design.clone();
    for row in 0..rescaled_design.nrows() {
        rescaled_design[(row, 0)] *= 3.0;
        rescaled_design[(row, 1)] *= 0.5;
    }
    let original = BivariateModel::build(&relationship, &observed, &design).unwrap();
    let rescaled =
        BivariateModel::build(&relationship, &observed, &rescaled_design).unwrap();
    let response = DVector::from_vec(vec![0.2, 0.1, -0.3, 0.4, 0.7, -0.2]);
    let theta = [1.0, 0.8, 0.4, 0.5, 0.2, -0.1];

    let original_value = original.objective_at(&theta, &response, true).unwrap().0;
    let rescaled_value = rescaled.objective_at(&theta, &response, true).unwrap().0;
    assert!((original_value - rescaled_value).abs() < 1.0e-12);
}

#[test]
fn fixed_state_score_refuses_nondifferentiable_zero_variance_coordinates() {
    let (relationship, observed, design) = valid_problem();
    let model = BivariateModel::build(&relationship, &observed, &design).unwrap();
    let response = DVector::from_vec(vec![0.2, 0.1, -0.3, 0.4, 0.7, -0.2]);
    let theta = [1.0, 0.8, 0.0, 0.5, 0.7, -0.1];

    assert!(model.objective_at(&theta, &response, true).is_none());
}
