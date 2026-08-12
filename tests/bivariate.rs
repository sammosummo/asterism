use asterism::BivariateModel;
use nalgebra::DMatrix;

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
