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

use asterism::{
    Censoring, ComponentModel, Known, MixedBivariateModel, RepeatedModel, TobitModel, TraitData,
    TraitKind,
};
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

/// Keep only some rows of a square matrix.
fn subset(matrix: &DMatrix<f64>, keep: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(keep.len(), keep.len(), |i, j| matrix[(keep[i], keep[j])])
}

#[test]
fn a_value_never_measured_is_imputed_to_the_answer_dropping_it_would_give() {
    // **This is the test that says the imputation is honest.** A value that was
    // never measured contributes nothing to the likelihood, so a model that
    // imputes it must land exactly where a model given only the rest lands. If
    // the expectation step treated an imputed value as though it had been
    // measured -- which is what leaving out the covariance of the imputation
    // would do -- the variances would come back too small and this would fail.
    let replicates = 2;
    let data = simulate(50, replicates, 1, 0.5, 0.2, 0.3, 20_260_824);
    let people = data.relationship.nrows();
    let identity = DMatrix::<f64>::identity(people, people);
    let rows = people * replicates;

    // Every seventh row was never measured, which leaves some people with one
    // replicate, some with both, and a few families short of a whole person.
    let mut known: Vec<Known> = (0..rows)
        .map(|row| Known::Value(data.response[(row, 0)]))
        .collect();
    let mut kept = Vec::new();
    for row in 0..rows {
        if row % 7 == 3 {
            known[row] = Known::Missing;
        } else {
            kept.push(row);
        }
    }
    assert!(kept.len() < rows && kept.len() > rows / 2);

    let model = RepeatedModel::build(
        &[data.relationship.clone(), identity.clone()],
        &data.design,
        replicates,
        1,
    )
    .expect("the model should build");
    let ours = model.fit_known(&known).expect("the fit should run");
    assert!(ours.monotone, "the likelihood fell during the search");
    assert!(
        ours.converged,
        "|g| = {} after {} iterations",
        ours.scaled_gradient, ours.iterations
    );

    let general = ComponentModel::build(
        &[
            subset(&expand(&data.relationship, replicates), &kept),
            subset(&expand(&identity, replicates), &kept),
        ],
        &DMatrix::from_fn(kept.len(), data.design.ncols(), |i, j| {
            data.design[(kept[i], j)]
        }),
    )
    .expect("the component model should build");
    let y = DVector::from_iterator(kept.len(), kept.iter().map(|&row| data.response[(row, 0)]));
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
        "log-likelihood {} against {}, difference {}",
        ours.loglik,
        theirs.loglik,
        ours.loglik - theirs.loglik
    );
    assert_eq!(ours.sequential_dimension, 0, "nothing here is censored");
}

/// One ear tested and the other never tested, with some of the tested ones
/// having reached a limit.
///
/// # Why the comparison is set up this way
///
/// `TobitModel` refuses a relationship matrix with an off-diagonal above 0.9,
/// deliberately: its two-person quadrature loses accuracy as the correlation
/// approaches one. **A replicate design always produces exactly one there** --
/// the two ears of a person share the whole of that person's genotype -- so the
/// two models cannot be pointed at the same expanded matrix.
///
/// They can be pointed at the same *analysis*. Leave the second replicate of
/// every person unmeasured and what remains is one record per person with a
/// genetic component and a residual, which is precisely the model `TobitModel`
/// fits, on the plain relationship matrix it accepts. The repeated model still
/// has to impute the untested ear, condition the censored values on what was
/// measured, and get the region right; it simply has an independent answer to
/// be checked against while doing it.
struct OneEar {
    known: Vec<Known>,
    value: Vec<f64>,
    censoring: Vec<Censoring>,
    limit: Vec<f64>,
    design: DMatrix<f64>,
    share_of_tested: f64,
}

fn one_ear(data: &Simulated, replicates: usize, every: usize, offset: usize) -> OneEar {
    let rows = data.response.nrows();
    let people = rows / replicates;
    let mut known = Vec::with_capacity(rows);
    let mut value = Vec::with_capacity(people);
    let mut censoring = Vec::with_capacity(people);
    let mut limit = Vec::with_capacity(people);
    let mut design = DMatrix::<f64>::zeros(people, data.design.ncols());
    let mut hit = 0.0;
    for person in 0..people {
        let tested = person * replicates;
        let observed = data.response[(tested, 0)];
        // The limit sits a little below the value, so the record is genuinely
        // censored rather than censored at a limit it never reached.
        if person % every == offset {
            let at = observed - 0.35;
            known.push(Known::Above(at));
            value.push(0.0);
            censoring.push(Censoring::Above);
            limit.push(at);
            hit += 1.0;
        } else {
            known.push(Known::Value(observed));
            value.push(observed);
            censoring.push(Censoring::Measured);
            limit.push(0.0);
        }
        for replicate in 1..replicates {
            let _ = replicate;
            known.push(Known::Missing);
        }
        for column in 0..data.design.ncols() {
            design[(person, column)] = data.design[(tested, column)];
        }
    }
    OneEar {
        known,
        value,
        censoring,
        limit,
        design,
        share_of_tested: hit / people as f64,
    }
}

#[test]
fn one_censored_record_per_family_reproduces_the_tobit_model() {
    // With one censored record in a family the region is one-dimensional, where
    // the sequential update is not an approximation at all: it is the ordinary
    // truncated normal, exactly. So this is exact expectation-maximisation for
    // this likelihood, and it must land where `TobitModel` lands, by an
    // arithmetic it shares nothing of.
    //
    // A family is four people, so censoring every fourth person puts exactly
    // one in each.
    let replicates = 2;
    let data = simulate(50, replicates, 1, 0.5, 0.0, 0.4, 20_260_825);
    let cut = one_ear(&data, replicates, 4, 2);
    assert!(
        (cut.share_of_tested - 0.25).abs() < 1e-12,
        "censored share of those tested {}",
        cut.share_of_tested
    );

    // One person-level component and the residual, which is the model
    // `TobitModel` fits.
    let model = RepeatedModel::build(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        1,
    )
    .expect("the model should build");
    let ours = model.fit_known(&cut.known).expect("the fit should run");
    assert_eq!(
        ours.sequential_dimension, 1,
        "the point of this case is that no family needs more than one"
    );
    // At one dimension the expectation step is exact, so this is ordinary
    // expectation-maximisation and the likelihood cannot fall. Above one it can
    // in principle, which is why the same assertion is not made in the next
    // test: there a fall would be news rather than a fault.
    assert!(ours.monotone, "the likelihood fell during the search");

    let theirs = TobitModel::build(
        &data.relationship,
        &cut.value,
        &cut.censoring,
        &cut.limit,
        &cut.design,
    )
    .expect("the tobit model should build")
    .fit()
    .expect("the tobit fit should run");

    let genetic = ours.component_covariances[0][(0, 0)];
    let heritability = genetic / (genetic + ours.residual_covariance[(0, 0)]);
    assert!(
        (heritability - theirs.heritability).abs() < 5e-3,
        "heritability {heritability} against {}, difference {}",
        theirs.heritability,
        heritability - theirs.heritability
    );
    assert!(
        (ours.loglik - theirs.loglik).abs() < 1e-2,
        "log-likelihood {} against {}, difference {}",
        ours.loglik,
        theirs.loglik,
        ours.loglik - theirs.loglik
    );
}

#[test]
fn several_censored_records_per_family_still_track_the_tobit_model() {
    // Here two records in a family are censored, so the region is
    // two-dimensional. The likelihood is still exact there -- the region
    // probability is the bivariate distribution function and not the sequential
    // update -- but **the expectation step is not**, because the moments have
    // only the sequential update to come from. So the two need not agree
    // exactly. What they must not do is disagree by enough to matter.
    let replicates = 2;
    let data = simulate(50, replicates, 1, 0.5, 0.0, 0.4, 20_260_826);
    let cut = one_ear(&data, replicates, 2, 1);
    assert!((cut.share_of_tested - 0.5).abs() < 1e-12);

    let model = RepeatedModel::build(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        1,
    )
    .expect("the model should build");
    let ours = model.fit_known(&cut.known).expect("the fit should run");
    assert_eq!(ours.sequential_dimension, 2);

    let theirs = TobitModel::build(
        &data.relationship,
        &cut.value,
        &cut.censoring,
        &cut.limit,
        &cut.design,
    )
    .expect("the tobit model should build")
    .fit()
    .expect("the tobit fit should run");

    let genetic = ours.component_covariances[0][(0, 0)];
    let heritability = genetic / (genetic + ours.residual_covariance[(0, 0)]);
    assert!(
        (heritability - theirs.heritability).abs() < 0.02,
        "heritability {heritability} against {}, difference {}",
        theirs.heritability,
        heritability - theirs.heritability
    );
}

#[test]
fn nothing_censored_takes_the_same_road_as_the_uncensored_fit() {
    // `fit` and `fit_known` must be the same function reached two ways, and the
    // fast path that skips the dense expectation step must not change the
    // answer it skips to.
    let replicates = 2;
    let data = simulate(30, replicates, 2, 0.5, 0.2, 0.3, 20_260_827);
    let people = data.relationship.nrows();
    let identity = DMatrix::<f64>::identity(people, people);
    let model = RepeatedModel::build(&[data.relationship, identity], &data.design, replicates, 2)
        .expect("the model should build");

    let direct = model.fit(&data.response).expect("the fit should run");
    let mut known = Vec::new();
    for row in 0..data.response.nrows() {
        for position in 0..2 {
            known.push(Known::Value(data.response[(row, position)]));
        }
    }
    let through = model.fit_known(&known).expect("the fit should run");
    assert_eq!(direct.loglik, through.loglik);
    assert_eq!(direct.iterations, through.iterations);
    for position in 0..2 {
        assert_eq!(direct.censored_shares[position], 0.0);
    }
}

/// Positions on a line, unevenly spaced the way audiometric frequencies are on
/// any scale anyone would put them on.
const LINE: [f64; 6] = [0.0, 2.1, 5.0, 9.4, 14.0, 22.0];

/// The correlation the kernel puts between two positions.
fn kernel_at(floor: f64, rate: f64, separation: f64) -> f64 {
    floor + (1.0 - floor) * (-rate * separation).exp()
}

/// A covariance in the kernel family: a scale at every position, one floor and
/// one rate.
fn shaped(scale: &[f64], floor: f64, rate: f64) -> DMatrix<f64> {
    let size = scale.len();
    DMatrix::from_fn(size, size, |i, j| {
        scale[i] * scale[j] * kernel_at(floor, rate, (LINE[i] - LINE[j]).abs())
    })
}

struct OnALine {
    relationship: DMatrix<f64>,
    design: DMatrix<f64>,
    response: DMatrix<f64>,
}

/// Data from a model whose covariances really are in the kernel family, so
/// that there is a right answer for the floor and the rate to be checked
/// against.
fn simulate_on_a_line(
    families: usize,
    replicates: usize,
    genetic: &DMatrix<f64>,
    residual: &DMatrix<f64>,
    seed: u64,
) -> OnALine {
    let positions = LINE.len();
    let a = relationship(families);
    let people = a.nrows();
    let rows = people * replicates;
    let across = a
        .clone()
        .cholesky()
        .expect("the relationship matrix should factorise")
        .l();
    let along = genetic
        .clone()
        .cholesky()
        .expect("the genetic covariance should factorise")
        .l();
    let noise = residual
        .clone()
        .cholesky()
        .expect("the residual covariance should factorise")
        .l();
    let mut stream = Stream(seed);

    let mut design = DMatrix::<f64>::zeros(rows, 2);
    for row in 0..rows {
        design[(row, 0)] = 1.0;
        design[(row, 1)] = stream.normal();
    }
    let mut fixed = DMatrix::<f64>::zeros(2, positions);
    for t in 0..positions {
        fixed[(0, t)] = 0.3 * (t as f64) - 0.5;
        fixed[(1, t)] = 0.4 - 0.05 * (t as f64);
    }
    let mut response = design.clone() * fixed;

    // The genetic effects are a matrix normal: correlated down the people by
    // the relationship matrix and across the positions by the kernel.
    let draws = DMatrix::from_fn(people, positions, |_, _| stream.normal());
    let effects = &across * draws * along.transpose();
    for person in 0..people {
        for replicate in 0..replicates {
            let row = person * replicates + replicate;
            let own = DVector::from_fn(positions, |_, _| stream.normal());
            let level = &noise * own;
            for t in 0..positions {
                response[(row, t)] += effects[(person, t)] + level[t];
            }
        }
    }
    OnALine {
        relationship: a,
        design,
        response,
    }
}

/// **A floor and a rate are not separately estimable, and the correlation they
/// describe is.** This is a property of the kernel and not of any fit: over a
/// finite span, a high floor with a fast decay and no floor at all with a slow
/// one draw very nearly the same curve. Two pairs as far apart as 0.35 and
/// nought in the floor agree to within a twentieth of a correlation everywhere
/// a fit would look.
///
/// It is why the fit hands back a function rather than the two numbers behind
/// it, and why the test below checks the function.
#[test]
fn a_floor_and_a_rate_trade_off_against_each_other() {
    let mut worst: f64 = 0.0;
    let mut separation = 0.0;
    while separation <= LINE[LINE.len() - 1] {
        let with_floor = kernel_at(0.35, 0.09, separation);
        let without = kernel_at(0.0, 0.043, separation);
        worst = worst.max((with_floor - without).abs());
        separation += 0.1;
    }
    assert!(
        worst < 0.06,
        "the two curves differ by {worst}, which would make them tellable apart"
    );
}

#[test]
fn the_kernel_recovers_the_correlation_it_was_given() {
    let replicates = 2;
    let scale = [1.0, 1.1, 0.9, 1.2, 1.0, 0.8];
    let (floor, rate) = (0.35, 0.09);
    let genetic = shaped(&scale, floor, rate);
    // The replicate level decays much faster and has almost no floor, which is
    // what test-retest noise looks like beside a genetic effect.
    let residual = shaped(&[0.9, 0.9, 1.0, 1.0, 1.1, 1.1], 0.05, 0.8);
    let data = simulate_on_a_line(150, replicates, &genetic, &residual, 20_260_828);

    let model = RepeatedModel::build_on_a_line(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        &LINE,
    )
    .expect("the model should build");
    let fit = model.fit(&data.response).expect("the fit should run");

    assert!(fit.monotone, "the likelihood fell during the search");
    assert!(
        fit.converged,
        "|g| = {} after {} iterations",
        fit.scaled_gradient, fit.iterations
    );
    assert_eq!(fit.floors.len(), 2, "one component and the residual");
    assert_eq!(fit.rates.len(), 2);

    // **The curve, not the two numbers behind it.** The floor came back at
    // nought and the rate at half what it was given, and the correlation is
    // right anyway, which is the trade-off the test above pins down.
    for separation in [2.1, 5.0, 9.4, 14.0, 22.0] {
        let got = fit
            .correlation(0, separation)
            .expect("there is a kernel and a component nought");
        let wanted = kernel_at(floor, rate, separation);
        assert!(
            (got - wanted).abs() < 0.08,
            "genetic correlation at {separation}: {got} against {wanted}"
        );
    }
    // The replicate level's decay is fast enough for its own two numbers to be
    // separated, so there they can be checked directly.
    assert!(
        (fit.floors[1] - 0.05).abs() < 0.08,
        "replicate floor {} against 0.05",
        fit.floors[1]
    );
    assert!(
        (fit.rates[1] - 0.8).abs() < 0.25,
        "replicate rate {} against 0.8",
        fit.rates[1]
    );
    // And the qualitative statement the two levels are there to make: what a
    // person shares between ears reaches further across the line than what one
    // ear carries alone.
    assert!(
        fit.correlation(0, 9.4).expect("a correlation")
            > fit.correlation(1, 9.4).expect("a correlation"),
        "the genetic level should reach further: {:?} against {:?}",
        fit.correlation(0, 9.4),
        fit.correlation(1, 9.4)
    );

    // The correlation is the function the two fitted numbers describe, and
    // nothing else.
    for separation in [0.0, 2.1, 9.4, 22.0] {
        let got = fit.correlation(0, separation).expect("a correlation");
        assert!((got - kernel_at(fit.floors[0], fit.rates[0], separation)).abs() < 1e-12);
    }
    assert!((fit.correlation(0, 0.0).expect("a correlation") - 1.0).abs() < 1e-12);
    assert!(
        fit.correlation(9, 1.0).is_none(),
        "there is no component nine"
    );
}

#[test]
fn the_kernel_is_a_restriction_of_the_free_covariance() {
    // The kernel family sits inside the free one, so its best fit cannot beat
    // the free one's. If it ever did, one of the two searches would not be
    // finding what it claims to.
    let replicates = 2;
    let genetic = shaped(&[1.0, 1.1, 0.9, 1.2, 1.0, 0.8], 0.35, 0.09);
    let residual = shaped(&[0.9, 0.9, 1.0, 1.0, 1.1, 1.1], 0.05, 0.8);
    let data = simulate_on_a_line(60, replicates, &genetic, &residual, 20_260_829);

    let restricted = RepeatedModel::build_on_a_line(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        &LINE,
    )
    .expect("the model should build")
    .fit(&data.response)
    .expect("the fit should run");

    let free = RepeatedModel::build(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        LINE.len(),
    )
    .expect("the model should build")
    .fit(&data.response)
    .expect("the fit should run");

    assert!(
        restricted.loglik <= free.loglik + 1e-6,
        "the restricted fit beat the free one: {} against {}",
        restricted.loglik,
        free.loglik
    );
    // And it should not be far behind, because the truth is in the family.
    // Twelve parameters against forty-two, on data the kernel can express.
    assert!(
        free.loglik - restricted.loglik < 40.0,
        "the kernel cost {} log units, which is more than the shape it removed",
        free.loglik - restricted.loglik
    );
    assert_eq!(restricted.floors.len(), 2);
    assert!(free.floors.is_empty());

    // And the kernel should be drawing the curve the free fit found rather
    // than one of its own. The free fit has a correlation per pair of
    // positions; the kernel has two numbers; they should agree about the
    // pairs.
    let genetic = &free.component_covariances[0];
    for (a, b) in [(0usize, 1usize), (0, 3), (0, 5), (2, 4)] {
        let empirical = genetic[(a, b)] / (genetic[(a, a)] * genetic[(b, b)]).sqrt();
        let curve = restricted
            .correlation(0, LINE[b] - LINE[a])
            .expect("there is a kernel");
        assert!(
            (empirical - curve).abs() < 0.12,
            "positions {a} and {b}: free {empirical} against kernel {curve}"
        );
    }
}

#[test]
fn the_kernel_contains_independence_and_finds_it() {
    // At a floor of nought and a large rate the correlation between distinct
    // positions is nothing, so the family contains the case where the positions
    // have nothing to do with each other. A fit on data like that should say so
    // rather than inventing structure.
    let replicates = 2;
    let independent = DMatrix::<f64>::identity(LINE.len(), LINE.len());
    let data = simulate_on_a_line(120, replicates, &independent, &independent, 20_260_830);

    let fit = RepeatedModel::build_on_a_line(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        &LINE,
    )
    .expect("the model should build")
    .fit(&data.response)
    .expect("the fit should run");

    for component in 0..2 {
        let at_one = fit.correlation(component, 2.1).expect("there is a kernel");
        assert!(
            at_one < 0.25,
            "component {component} invented a correlation of {at_one} between \
             positions that are independent"
        );
    }
}

#[test]
fn a_line_the_kernel_cannot_use_is_refused() {
    let a = relationship(4);
    let people = a.nrows();
    let design = DMatrix::<f64>::from_element(people * 2, 1, 1.0);
    assert_eq!(
        RepeatedModel::build_on_a_line(std::slice::from_ref(&a), &design, 2, &[0.0, 1.0]).err(),
        Some("REPEATED_KERNEL_NEEDS_THREE_POSITIONS")
    );
    assert_eq!(
        RepeatedModel::build_on_a_line(&[a], &design, 2, &[0.0, 1.0, 1.0]).err(),
        Some("REPEATED_KERNEL_POSITIONS_COINCIDE")
    );
}

#[test]
fn the_kernel_and_censoring_work_together() {
    // The model this was all built for: a correlation that decays along a line,
    // and values at the far end of that line that reached a limit instead of
    // being measured. The censoring is put where the audiogram has it -- almost
    // none at one end, half at the other -- so the positions that matter most
    // are the ones with the least measured in them.
    let replicates = 2;
    let genetic = shaped(&[1.0, 1.1, 0.9, 1.2, 1.0, 0.8], 0.35, 0.09);
    let residual = shaped(&[0.9, 0.9, 1.0, 1.0, 1.1, 1.1], 0.05, 0.8);
    let data = simulate_on_a_line(80, replicates, &genetic, &residual, 20_260_831);

    let share = [0.0, 0.0, 0.02, 0.10, 0.30, 0.50];
    let mut known = Vec::new();
    let mut counted = 0usize;
    for row in 0..data.response.nrows() {
        for position in 0..LINE.len() {
            let value = data.response[(row, position)];
            // Deterministic, so the pattern is the same every run: the top
            // `share` of each position by value is what an instrument running
            // out of output would take.
            let cut = ((row * 7 + position * 13) % 100) as f64 / 100.0;
            if cut < share[position] {
                known.push(Known::Above(value - 0.3));
                counted += 1;
            } else {
                known.push(Known::Value(value));
            }
        }
    }
    assert!(counted > 100, "only {counted} censored");

    let fit = RepeatedModel::build_on_a_line(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        &LINE,
    )
    .expect("the model should build")
    .fit_known(&known)
    .expect("the fit should run");

    assert!(
        fit.sequential_dimension > 2,
        "the approximation should be doing some work"
    );
    assert_eq!(fit.censored_shares[0], 0.0);
    assert!(fit.censored_shares[5] > 0.4);
    // The correlation the genetic level carries should still be the one it was
    // given, at the positions where enough was measured to say.
    for separation in [2.1, 9.4] {
        let got = fit.correlation(0, separation).expect("there is a kernel");
        let wanted = kernel_at(0.35, 0.09, separation);
        assert!(
            (got - wanted).abs() < 0.15,
            "genetic correlation at {separation}: {got} against {wanted}"
        );
    }
    assert!(
        fit.correlation(0, 9.4).expect("a correlation")
            > fit.correlation(1, 9.4).expect("a correlation"),
        "the genetic level should still reach further than the replicate one"
    );
}

/// Two positions on one record per person, which is what `MixedBivariateModel`
/// fits and what this model becomes when the second replicate was never
/// measured.
struct TwoPositions {
    known: Vec<Known>,
    first: TraitData,
    second: TraitData,
    design: DMatrix<f64>,
}

/// `censor` is the share of records at the **second** position that reached a
/// limit; the first position is always measured, which is the audiogram's own
/// shape and the case where the cross-position covariance has to be recovered
/// from records that are only half there.
fn two_positions(data: &Simulated, replicates: usize, censor: usize) -> TwoPositions {
    let rows = data.response.nrows();
    let people = rows / replicates;
    let mut known = Vec::with_capacity(rows * 2);
    let mut values = [Vec::with_capacity(people), Vec::with_capacity(people)];
    let mut censoring = [Vec::with_capacity(people), Vec::with_capacity(people)];
    let mut limits = [Vec::with_capacity(people), Vec::with_capacity(people)];
    let mut design = DMatrix::<f64>::zeros(people, data.design.ncols());

    for person in 0..people {
        let tested = person * replicates;
        for position in 0..2 {
            let observed = data.response[(tested, position)];
            if position == 1 && censor > 0 && person % censor == 1 {
                let at = observed - 0.35;
                known.push(Known::Above(at));
                values[position].push(0.0);
                censoring[position].push(Censoring::Above);
                limits[position].push(at);
            } else {
                known.push(Known::Value(observed));
                values[position].push(observed);
                censoring[position].push(Censoring::Measured);
                limits[position].push(0.0);
            }
        }
        for replicate in 1..replicates {
            let _ = replicate;
            known.push(Known::Missing);
            known.push(Known::Missing);
        }
        for column in 0..data.design.ncols() {
            design[(person, column)] = data.design[(tested, column)];
        }
    }
    let kinds = [
        TraitKind::Continuous,
        if censor > 0 {
            TraitKind::Censored
        } else {
            TraitKind::Continuous
        },
    ];
    let mut each = values
        .into_iter()
        .zip(censoring)
        .zip(limits)
        .zip(kinds)
        .map(|(((value, censoring), limit), kind)| TraitData {
            kind,
            value,
            censoring,
            limit,
        });
    let first = each.next().expect("two traits");
    let second = each.next().expect("two traits");
    TwoPositions {
        known,
        first,
        second,
        design,
    }
}

#[test]
fn two_positions_reproduce_the_mixed_bivariate_model() {
    // **This is the check the `TobitModel` comparison cannot make.** That one
    // has a single position and so says nothing about the covariance *across*
    // positions, which is the whole of what this model adds. Two positions is
    // a bivariate model, and `MixedBivariateModel` fits exactly that by a
    // different search on the same likelihood.
    let replicates = 2;
    let data = simulate(60, replicates, 2, 0.5, 0.0, 0.4, 20_260_832);
    let cut = two_positions(&data, replicates, 0);

    let ours = RepeatedModel::build(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        2,
    )
    .expect("the model should build")
    .fit_known(&cut.known)
    .expect("the fit should run");
    assert!(ours.monotone, "the likelihood fell during the search");

    let theirs = MixedBivariateModel::build(&data.relationship, cut.first, cut.second, &cut.design)
        .expect("the mixed bivariate model should build")
        .fit()
        .expect("the mixed bivariate fit should run");

    let genetic = &ours.component_covariances[0];
    let residual = &ours.residual_covariance;
    for position in 0..2 {
        let heritability = genetic[(position, position)]
            / (genetic[(position, position)] + residual[(position, position)]);
        assert!(
            (heritability - theirs.heritability[position]).abs() < 5e-3,
            "heritability at {position}: {heritability} against {}",
            theirs.heritability[position]
        );
    }
    let genetic_correlation = genetic[(0, 1)] / (genetic[(0, 0)] * genetic[(1, 1)]).sqrt();
    assert!(
        (genetic_correlation - theirs.genetic_correlation).abs() < 5e-3,
        "genetic correlation {genetic_correlation} against {}",
        theirs.genetic_correlation
    );
    let residual_correlation = residual[(0, 1)] / (residual[(0, 0)] * residual[(1, 1)]).sqrt();
    assert!(
        (residual_correlation - theirs.residual_correlation).abs() < 5e-3,
        "residual correlation {residual_correlation} against {}",
        theirs.residual_correlation
    );
    assert!(
        (ours.loglik - theirs.loglik).abs() < 1e-2,
        "log-likelihood {} against {}, difference {}",
        ours.loglik,
        theirs.loglik,
        ours.loglik - theirs.loglik
    );
}

#[test]
fn a_censored_second_position_still_reproduces_the_mixed_bivariate_model() {
    // The same comparison with one of the two positions censored, which is what
    // the audiogram looks like: the low frequencies are all measured and the
    // high ones are not. Every third record reaches a limit at the second
    // position, so a family of four carries one or two of them and the region
    // the approximation runs on is small.
    let replicates = 2;
    let data = simulate(60, replicates, 2, 0.5, 0.0, 0.4, 20_260_833);
    let cut = two_positions(&data, replicates, 3);
    let censored = cut
        .second
        .censoring
        .iter()
        .filter(|c| **c == Censoring::Above)
        .count();
    assert!(censored > 60, "only {censored} censored");

    let ours = RepeatedModel::build(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        2,
    )
    .expect("the model should build")
    .fit_known(&cut.known)
    .expect("the fit should run");

    let theirs = MixedBivariateModel::build(&data.relationship, cut.first, cut.second, &cut.design)
        .expect("the mixed bivariate model should build")
        .fit()
        .expect("the mixed bivariate fit should run");

    let genetic = &ours.component_covariances[0];
    let residual = &ours.residual_covariance;
    for position in 0..2 {
        let heritability = genetic[(position, position)]
            / (genetic[(position, position)] + residual[(position, position)]);
        assert!(
            (heritability - theirs.heritability[position]).abs() < 0.03,
            "heritability at {position}: {heritability} against {}",
            theirs.heritability[position]
        );
    }
    let genetic_correlation = genetic[(0, 1)] / (genetic[(0, 0)] * genetic[(1, 1)]).sqrt();
    assert!(
        (genetic_correlation - theirs.genetic_correlation).abs() < 0.05,
        "genetic correlation {genetic_correlation} against {}",
        theirs.genetic_correlation
    );
}

/// The profile interval on the genetic correlation at a named separation.
///
/// **This is the only interval the joint model offers, and ADR 0010 says why:**
/// the floor and the rate trade off against each other almost exactly, so an
/// interval on either would be wide and would not mean what it looked like. The
/// correlation at a separation is what the data speak to.
///
/// What is checked here is what an interval has to do. It has to contain its own
/// estimate. Its ends have to cost 3.8415 in deviance and not some other number,
/// or the 95 per cent is decoration. And it has to contain the correlation the
/// data were simulated from, at least on a sample this size, or it is not an
/// interval for that quantity at all.
#[test]
fn the_interval_ends_where_the_likelihood_says_it_should() {
    let replicates = 2;
    let scale = [1.0, 1.1, 0.9, 1.2, 1.0, 0.8];
    let (floor, rate) = (0.35, 0.09);
    let genetic = shaped(&scale, floor, rate);
    let residual = shaped(&[0.9, 0.9, 1.0, 1.0, 1.1, 1.1], 0.05, 0.8);
    let data = simulate_on_a_line(150, replicates, &genetic, &residual, 20_260_828);

    let model = RepeatedModel::build_on_a_line(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        &LINE,
    )
    .expect("the model should build");

    let separation = 9.4;
    let mut known = Vec::new();
    for row in 0..data.response.nrows() {
        for position in 0..data.response.ncols() {
            known.push(Known::Value(data.response[(row, position)]));
        }
    }
    let interval = model
        .correlation_interval(&known, 0, separation)
        .expect("the interval should be takeable");

    assert_eq!(interval.component, 0);
    assert!((interval.level - 0.95).abs() < 1e-12);
    assert_eq!(
        interval.profile_failures, 0,
        "{} points of the profile could not be fitted",
        interval.profile_failures
    );
    assert!(
        interval.lower <= interval.estimate && interval.estimate <= interval.upper,
        "the interval {} to {} does not contain its own estimate {}",
        interval.lower,
        interval.upper,
        interval.estimate
    );
    assert!(
        interval.upper - interval.lower < 0.9,
        "an interval of width {} says nothing",
        interval.upper - interval.lower
    );

    // The quantity the data actually came from.
    let wanted = kernel_at(floor, rate, separation);
    assert!(
        interval.lower <= wanted && wanted <= interval.upper,
        "the interval {} to {} misses the correlation it was simulated from, {wanted}",
        interval.lower,
        interval.upper
    );

    // **What makes it a 95 per cent interval.** Each end that is not sitting on
    // a bound has to cost 3.8415 in deviance, which is the whole of ADR 0004's
    // recipe. Bisection stops at a thousandth, so the deviance is right to about
    // a hundredth.
    let free = model.fit_known(&known).expect("the free fit should run");
    let at_estimate = model
        .fit_holding(&known, Some((0, separation, interval.estimate)))
        .expect("the fit at the estimate should run")
        .loglik;
    assert!(
        (at_estimate - free.loglik).abs() < 1e-4,
        "holding the free answer's own correlation cost {} in log-likelihood",
        free.loglik - at_estimate
    );
    for (name, end, at_bound) in [
        ("lower", interval.lower, interval.lower_limited),
        ("upper", interval.upper, interval.upper_limited),
    ] {
        if at_bound {
            continue;
        }
        let held = model
            .fit_holding(&known, Some((0, separation, end)))
            .expect("the fit at the end should run")
            .loglik;
        let deviance = 2.0 * (at_estimate - held);
        assert!(
            (deviance - 3.841_458_820_694_124).abs() < 0.05,
            "the {name} end at {end} costs {deviance} in deviance"
        );
    }

    // A component that is not there, and a separation that is not a separation,
    // are refused rather than answered.
    assert_eq!(
        model
            .correlation_interval(&known, 9, separation)
            .unwrap_err(),
        "REPEATED_NO_SUCH_COMPONENT"
    );
    assert_eq!(
        model.correlation_interval(&known, 0, 0.0).unwrap_err(),
        "REPEATED_SEPARATION_NOT_POSITIVE"
    );

    // Without a kernel there is no curve to hold, and saying so is better than
    // holding something else.
    let plain = RepeatedModel::build(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        LINE.len(),
    )
    .expect("the plain model should build");
    assert_eq!(
        plain
            .correlation_interval(&known, 0, separation)
            .unwrap_err(),
        "REPEATED_NO_KERNEL"
    );
}

/// The profile interval on a component's share of the variance at one position.
///
/// **This is the heritability interval, and it is a different machinery from the
/// correlation's.** A correlation belongs to one component; a share is a ratio
/// between them, so holding it ties together maximisations that expectation
/// maximisation keeps separate. The fit is split into a step that moves
/// everything except the scales at that position and a step that moves those
/// scales along the constraint.
///
/// What is checked is what makes that split legitimate. Holding the free
/// answer's own share must reproduce the free fit, or the constrained fit is
/// searching a smaller space than it claims. Every held fit must realise the
/// share it was asked for. The ends must cost 3.8415 in deviance. And the
/// interval must contain the share the data were simulated from.
#[test]
fn the_heritability_interval_ends_where_the_likelihood_says_it_should() {
    let replicates = 2;
    let scale = [1.0, 1.1, 0.9, 1.2, 1.0, 0.8];
    let genetic = shaped(&scale, 0.35, 0.09);
    let residual = shaped(&[0.9, 0.9, 1.0, 1.0, 1.1, 1.1], 0.05, 0.8);
    let data = simulate_on_a_line(150, replicates, &genetic, &residual, 20_260_828);

    let model = RepeatedModel::build_on_a_line(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        &LINE,
    )
    .expect("the model should build");
    let mut known = Vec::new();
    for row in 0..data.response.nrows() {
        for position in 0..data.response.ncols() {
            known.push(Known::Value(data.response[(row, position)]));
        }
    }

    let at = 2usize;
    let free = model.fit_known(&known).expect("the free fit should run");
    let estimate = free.variance_shares[at][0];

    // **Holding the free answer's own share must cost nothing.** If it cost
    // something, every deviance would be measured against the wrong reference
    // and every interval would be too wide.
    let at_estimate = model
        .fit_holding_share(&known, 0, at, estimate)
        .expect("the fit at the estimate should run");
    assert!(
        (at_estimate.loglik - free.loglik).abs() < 1e-4,
        "holding the free share cost {} in log-likelihood",
        free.loglik - at_estimate.loglik
    );

    // Every held fit realises the share it was given, or it is a profile of
    // something else.
    for share in [0.1, 0.25, estimate, 0.6, 0.8] {
        let fit = model
            .fit_holding_share(&known, 0, at, share)
            .expect("the share should be reachable");
        assert!(
            (fit.variance_shares[at][0] - share).abs() < 1e-6,
            "asked for {share}, got {}",
            fit.variance_shares[at][0]
        );
    }

    let interval = model
        .heritability_interval(&known, 0, at)
        .expect("the interval should be takeable");
    assert_eq!(interval.component, 0);
    assert_eq!(interval.position, at);
    assert_eq!(interval.profile_failures, 0);
    assert!(
        interval.lower <= interval.estimate && interval.estimate <= interval.upper,
        "the interval {} to {} does not contain its own estimate {}",
        interval.lower,
        interval.upper,
        interval.estimate
    );

    // The share the data came from: the genetic variance at this position
    // against the whole of it.
    let genetic_variance = genetic[(at, at)];
    let wanted = genetic_variance / (genetic_variance + residual[(at, at)]);
    assert!(
        interval.lower <= wanted && wanted <= interval.upper,
        "the interval {} to {} misses the share it was simulated from, {wanted}",
        interval.lower,
        interval.upper
    );

    // Each end that is not on a bound costs 3.8415 in deviance, which is what
    // makes it a 95 per cent interval rather than a pair of numbers.
    for (name, end, at_bound) in [
        ("lower", interval.lower, interval.lower_limited),
        ("upper", interval.upper, interval.upper_limited),
    ] {
        if at_bound {
            continue;
        }
        let held = model
            .fit_holding_share(&known, 0, at, end)
            .expect("the fit at the end should run")
            .loglik;
        let deviance = 2.0 * (at_estimate.loglik - held);
        assert!(
            (deviance - 3.841_458_820_694_124).abs() < 0.05,
            "the {name} end at {end} costs {deviance} in deviance"
        );
    }

    // What is refused rather than answered.
    assert_eq!(
        model.heritability_interval(&known, 9, at).unwrap_err(),
        "REPEATED_NO_SUCH_COMPONENT"
    );
    assert_eq!(
        model.heritability_interval(&known, 0, 99).unwrap_err(),
        "REPEATED_NO_SUCH_POSITION"
    );
    let plain = RepeatedModel::build(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        LINE.len(),
    )
    .expect("the plain model should build");
    assert_eq!(
        plain.heritability_interval(&known, 0, at).unwrap_err(),
        "REPEATED_NO_KERNEL"
    );
}

/// **Holding a share must not stop the fit climbing.** The split into two
/// conditional maximisations is what keeps this an ECM, and if either half went
/// downhill the fit would stop wherever it happened to be and the profile would
/// sit below the likelihood.
#[test]
fn a_held_share_still_climbs_the_likelihood() {
    let replicates = 2;
    let genetic = shaped(&[1.0, 1.1, 0.9, 1.2, 1.0, 0.8], 0.35, 0.09);
    let residual = shaped(&[0.9, 0.9, 1.0, 1.0, 1.1, 1.1], 0.05, 0.8);
    let data = simulate_on_a_line(80, replicates, &genetic, &residual, 20_260_901);
    let model = RepeatedModel::build_on_a_line(
        std::slice::from_ref(&data.relationship),
        &data.design,
        replicates,
        &LINE,
    )
    .expect("the model should build");
    let mut known = Vec::new();
    for row in 0..data.response.nrows() {
        for position in 0..data.response.ncols() {
            known.push(Known::Value(data.response[(row, position)]));
        }
    }
    for share in [0.05, 0.3, 0.7] {
        let fit = model
            .fit_holding_share(&known, 0, 3, share)
            .expect("the share should be reachable");
        assert!(fit.monotone, "the likelihood fell while holding {share}");
        assert!(
            fit.converged,
            "holding {share} did not settle after {} iterations",
            fit.iterations
        );
        assert!((fit.variance_shares[3][0] - share).abs() < 1e-6);
    }
}
