//! The repeated-measures skeleton against an answer that is already known.
//!
//! At one position the model is a three-component one-trait model and nothing
//! more: a genetic effect and a person-level effect shared by both replicates,
//! and a replicate-level residual. [`ComponentModel`] fits exactly that, by
//! L-BFGS-B on the observed-data likelihood with an analytic gradient, and it
//! has been checked against SOLAR. So the two must agree -- not approximately,
//! and not in the estimates alone, but in the log-likelihood as well, because a
//! route that disagrees with the general estimator has the bug.
//!
//! This is the check ADR 0010 asks for before censoring is allowed anywhere
//! near the code. Expectation-maximisation and a bounded quasi-Newton search
//! share no arithmetic at all, so agreement between them is worth something.

use asterism::{ComponentModel, RepeatedModel};
use nalgebra::{DMatrix, DVector};

/// splitmix64 and Box-Muller, so the data can be reproduced from the seed.
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
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Unrelated nuclear families of two parents and two children.
fn relationship(families: usize) -> DMatrix<f64> {
    let people = families * 4;
    let mut matrix = DMatrix::<f64>::identity(people, people);
    for family in 0..families {
        let base = family * 4;
        for child in 2..4 {
            for parent in 0..2 {
                matrix[(base + child, base + parent)] = 0.5;
                matrix[(base + parent, base + child)] = 0.5;
            }
        }
        matrix[(base + 2, base + 3)] = 0.5;
        matrix[(base + 3, base + 2)] = 0.5;
    }
    matrix
}

/// A person-level matrix expanded to the replicate rows: both replicates of a
/// person carry whatever belongs to the person, which is the `(x) J_R` of the
/// model written out for a package that takes one row per observation.
fn expand(matrix: &DMatrix<f64>, replicates: usize) -> DMatrix<f64> {
    let people = matrix.nrows();
    let rows = people * replicates;
    let mut expanded = DMatrix::<f64>::zeros(rows, rows);
    for row_one in 0..rows {
        for row_two in 0..rows {
            expanded[(row_one, row_two)] = matrix[(row_one / replicates, row_two / replicates)];
        }
    }
    expanded
}

struct Simulated {
    relationship: DMatrix<f64>,
    design: DMatrix<f64>,
    /// `(P * R) x T`.
    response: DMatrix<f64>,
}

/// One data set from the model itself, so that the maximum is somewhere in the
/// interior and both routes have the same place to find.
fn simulate(
    families: usize,
    replicates: usize,
    positions: usize,
    genetic: f64,
    person: f64,
    residual: f64,
    seed: u64,
) -> Simulated {
    let a = relationship(families);
    let people = a.nrows();
    let rows = people * replicates;
    let factor = a
        .clone()
        .cholesky()
        .expect("the relationship matrix should factorise")
        .l();
    let mut stream = Stream(seed);

    let mut design = DMatrix::<f64>::zeros(rows, 2);
    for row in 0..rows {
        design[(row, 0)] = 1.0;
        design[(row, 1)] = stream.normal();
    }
    let mut fixed = DMatrix::<f64>::zeros(2, positions);
    for t in 0..positions {
        fixed[(0, t)] = 0.4 * (t as f64) - 0.2;
        fixed[(1, t)] = 0.6 - 0.1 * (t as f64);
    }

    let mut response = design.clone() * fixed;
    for t in 0..positions {
        // The genetic effect is correlated across people and independent across
        // positions here; the skeleton estimates a free covariance either way,
        // and a diagonal truth is the case where the answer is easiest to say.
        let draws = DVector::from_iterator(people, (0..people).map(|_| stream.normal()));
        let effect = &factor * draws * genetic.sqrt();
        for person_index in 0..people {
            let shared = person.sqrt() * stream.normal();
            for replicate in 0..replicates {
                let row = person_index * replicates + replicate;
                response[(row, t)] +=
                    effect[person_index] + shared + residual.sqrt() * stream.normal();
            }
        }
    }
    Simulated {
        relationship: a,
        design,
        response,
    }
}

#[test]
fn one_position_reproduces_the_component_model() {
    let replicates = 2;
    let data = simulate(60, replicates, 1, 0.5, 0.2, 0.3, 20_260_820);
    let people = data.relationship.nrows();
    let identity = DMatrix::<f64>::identity(people, people);

    let model = RepeatedModel::build(
        &[data.relationship.clone(), identity.clone()],
        &data.design,
        replicates,
        1,
    )
    .expect("the model should build");
    let ours = model.fit(&data.response).expect("the fit should run");

    let general = ComponentModel::build(
        &[
            expand(&data.relationship, replicates),
            expand(&identity, replicates),
        ],
        &data.design,
    )
    .expect("the component model should build");
    let y = DVector::from_iterator(
        data.response.nrows(),
        data.response.column(0).iter().copied(),
    );
    let theirs = general
        .fit(&y, false)
        .expect("the component fit should run");

    assert!(ours.monotone, "the likelihood fell during the search");
    assert!(
        ours.converged,
        "|g| = {} after {} iterations",
        ours.scaled_gradient, ours.iterations
    );
    assert!(theirs.converged, "the comparator did not converge");
    assert_eq!(ours.estimator, theirs.estimator);

    let mine = [
        ours.component_covariances[0][(0, 0)],
        ours.component_covariances[1][(0, 0)],
        ours.residual_covariance[(0, 0)],
    ];
    for (index, (got, want)) in mine.iter().zip(&theirs.variances).enumerate() {
        assert!(
            (got - want).abs() < 1e-5 * want.abs().max(1.0),
            "component {index}: {got} against {want}"
        );
    }
    assert!(
        (ours.loglik - theirs.loglik).abs() < 1e-6,
        "log-likelihood {} against {}, difference {}",
        ours.loglik,
        theirs.loglik,
        ours.loglik - theirs.loglik
    );
    for (index, want) in theirs.fixed_effects.iter().enumerate() {
        let got = ours.fixed_effects[(index, 0)];
        assert!(
            (got - want).abs() < 1e-5 * want.abs().max(1.0),
            "fixed effect {index}: {got} against {want}"
        );
    }
}

#[test]
fn three_replicates_reproduce_the_component_model_too() {
    let replicates = 3;
    let data = simulate(40, replicates, 1, 0.4, 0.25, 0.35, 20_260_821);
    let people = data.relationship.nrows();
    let identity = DMatrix::<f64>::identity(people, people);

    let model = RepeatedModel::build(
        &[data.relationship.clone(), identity.clone()],
        &data.design,
        replicates,
        1,
    )
    .expect("the model should build");
    let ours = model.fit(&data.response).expect("the fit should run");

    let general = ComponentModel::build(
        &[
            expand(&data.relationship, replicates),
            expand(&identity, replicates),
        ],
        &data.design,
    )
    .expect("the component model should build");
    let y = DVector::from_iterator(
        data.response.nrows(),
        data.response.column(0).iter().copied(),
    );
    let theirs = general
        .fit(&y, false)
        .expect("the component fit should run");

    let mine = [
        ours.component_covariances[0][(0, 0)],
        ours.component_covariances[1][(0, 0)],
        ours.residual_covariance[(0, 0)],
    ];
    for (index, (got, want)) in mine.iter().zip(&theirs.variances).enumerate() {
        assert!(
            (got - want).abs() < 1e-5 * want.abs().max(1.0),
            "component {index}: {got} against {want}"
        );
    }
    assert!((ours.loglik - theirs.loglik).abs() < 1e-6);
}

#[test]
fn a_second_position_that_carries_nothing_new_leaves_the_first_alone() {
    // Two positions whose truth is independent. The skeleton estimates the
    // covariance between them freely and will not put it at nought, but the
    // variances at the first position should still land near where the
    // one-position fit puts them: a model that could not manage that would be
    // reading the second position into the first.
    let replicates = 2;
    let data = simulate(60, replicates, 2, 0.5, 0.2, 0.3, 20_260_822);
    let people = data.relationship.nrows();
    let identity = DMatrix::<f64>::identity(people, people);

    let model = RepeatedModel::build(
        &[data.relationship.clone(), identity.clone()],
        &data.design,
        replicates,
        2,
    )
    .expect("the model should build");
    let both = model.fit(&data.response).expect("the fit should run");
    assert!(both.monotone);
    assert!(
        both.converged,
        "|g| = {} after {} iterations",
        both.scaled_gradient, both.iterations
    );

    let first = DMatrix::from_iterator(
        data.response.nrows(),
        1,
        data.response.column(0).iter().copied(),
    );
    let one = RepeatedModel::build(&[data.relationship, identity], &data.design, replicates, 1)
        .expect("the model should build");
    let alone = one.fit(&first).expect("the fit should run");

    for component in 0..2 {
        let joint = both.component_covariances[component][(0, 0)];
        let single = alone.component_covariances[component][(0, 0)];
        assert!(
            (joint - single).abs() < 0.1,
            "component {component}: {joint} jointly against {single} alone"
        );
    }
    // The shares must still be shares.
    for position in 0..2 {
        let total: f64 = both.variance_shares[position].iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}

#[test]
fn a_singular_relationship_matrix_is_handled_rather_than_divided_by() {
    // Two genetically identical people make the relationship matrix singular,
    // and a zero eigenvalue is a direction carrying no information about the
    // genetic component. The update is written so that nothing divides by it,
    // and this is the test that says so: a pedigree with a monozygotic pair in
    // it is ordinary rather than pathological.
    let replicates = 2;
    let mut data = simulate(60, replicates, 1, 0.5, 0.2, 0.3, 20_260_823);
    let people = data.relationship.nrows();
    for family in 0..(people / 4) {
        let twin = family * 4 + 2;
        let other = family * 4 + 3;
        data.relationship[(twin, other)] = 1.0;
        data.relationship[(other, twin)] = 1.0;
    }
    let identity = DMatrix::<f64>::identity(people, people);
    let smallest = data
        .relationship
        .clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .fold(f64::INFINITY, |worst, value| worst.min(*value));
    assert!(
        smallest.abs() < 1e-9,
        "the matrix should be singular for this test to mean anything, \
         smallest eigenvalue {smallest}"
    );

    let model = RepeatedModel::build(
        &[data.relationship.clone(), identity.clone()],
        &data.design,
        replicates,
        1,
    )
    .expect("the model should build");
    let ours = model.fit(&data.response).expect("the fit should run");
    assert!(ours.monotone);
    assert!(
        ours.converged,
        "|g| = {} after {} iterations",
        ours.scaled_gradient, ours.iterations
    );

    // The comparator sees exactly the same singular matrix, so the two are
    // answering the same question and not merely both producing a number.
    let general = ComponentModel::build(
        &[
            expand(&data.relationship, replicates),
            expand(&identity, replicates),
        ],
        &data.design,
    )
    .expect("the component model should build");
    let y = DVector::from_iterator(
        data.response.nrows(),
        data.response.column(0).iter().copied(),
    );
    let theirs = general
        .fit(&y, false)
        .expect("the component fit should run");

    let mine = [
        ours.component_covariances[0][(0, 0)],
        ours.component_covariances[1][(0, 0)],
        ours.residual_covariance[(0, 0)],
    ];
    for (index, (got, want)) in mine.iter().zip(&theirs.variances).enumerate() {
        assert!(
            (got - want).abs() < 1e-4 * want.abs().max(1.0),
            "component {index}: {got} against {want}"
        );
    }
    assert!(
        (ours.loglik - theirs.loglik).abs() < 1e-5,
        "log-likelihood {} against {}",
        ours.loglik,
        theirs.loglik
    );
}
