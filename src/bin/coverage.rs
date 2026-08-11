//! The coverage check.
//!
//! Simulate many datasets whose heritability is known, fit every one, and count
//! how often the 95 per cent interval contains the truth. If it comes out at 95
//! per cent the interval works; if it comes out at 85 the package has been
//! printing intervals that are too narrow. There is no way to learn this by
//! reading the code, which is why it exists.
//!
//! This is not inherited. Astrarium ran a check of this shape and it is what
//! chose the interval recipe in `docs/adr/0004` — but its roster and its
//! fixtures deliberately did not come across, so the pedigree here is generated
//! and the result has to be earned again.
//!
//! Three rules that make the number mean something:
//!
//! 1. **Every replicate is scored.** A fit that does not converge never covers.
//!    Dropping the awkward ones is how a coverage check comes out at 95 per cent
//!    while the package is wrong.
//! 2. **Membership of a boundary point follows the mixture rule** of
//!    `docs/adr/0004`, not ordinary containment. At a true heritability of zero
//!    the estimate lands on the bound about half the time, and whether the
//!    interval contains the point nought is exactly the question that recipe
//!    was chosen to answer.
//! 3. **A cell passes when the Clopper–Pearson interval on its coverage
//!    overlaps [0.940, 0.960]**, which is a statement about 8,000 replicates
//!    rather than about the estimate being near 0.95 by eye.
//!
//! Run with `cargo run --release --bin coverage`.

use std::time::Instant;

use asterism::PreparedModel;
use nalgebra::{Cholesky, DMatrix, DVector};
use statrs::distribution::{Beta, ContinuousCDF};

/// The truths to check. Both bounds are in, because the bounds are where the
/// recipe is doing work, and the cluster near zero is there because that is
/// where a heritability study usually lives.
const TRUTHS: [f64; 12] = [0.0, 0.05, 0.07, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0];
/// The number for the record. A smaller count may be passed as an argument
/// while working on the check itself, but a smaller count is not the check:
/// the band is only meaningful against a Clopper–Pearson interval this tight.
const REPLICATES: usize = 8_000;

fn replicates() -> usize {
    std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(REPLICATES)
}

fn families() -> usize {
    std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(25)
}

/// Covariates are on by default, because a check run on a design nobody uses is
/// not a check. The design is an intercept, age, age squared, sex and the two
/// age-by-sex products — `age_years^1,2#sex` in SOLAR's notation, which is what
/// 1,194 of the lab's 1,246 recorded runs carry. An intercept-only design does
/// not exercise the restricted likelihood's determinant term at all, which is
/// the part most likely to be wrong.
///
/// Passing `intercept` as a third argument drops back to one column, which is
/// only useful for seeing whether a difference is caused by the covariates.
fn with_covariates() -> bool {
    std::env::args().nth(3).as_deref() != Some("intercept")
}

/// True coefficients, so that recovery of the fixed effects can be checked
/// alongside the variance.
const BETA: [f64; 6] = [2.0, 0.30, -0.10, 0.50, 0.05, -0.02];

/// A design with an intercept and, optionally, five covariates. Age is taken
/// from a person's place in the family — grandparents oldest, grandchildren
/// youngest — and standardised; sex alternates.
fn design(families: usize, block: usize, covariates: bool) -> DMatrix<f64> {
    let n = families * block;
    let columns = if covariates { 6 } else { 1 };
    let mut x = DMatrix::<f64>::zeros(n, columns);
    for row in 0..n {
        x[(row, 0)] = 1.0;
        if !covariates {
            continue;
        }
        let within = row % block;
        // Three generations: 0,1 grandparents; 2..7 parents; 8..13 children.
        let decade = match within {
            0 | 1 => 7.0,
            2..=7 => 4.5,
            _ => 2.0,
        };
        let age = (decade - 4.5) / 2.0 + ((row % 7) as f64 - 3.0) / 10.0;
        let sex = f64::from(u8::from(row % 2 == 0));
        x[(row, 1)] = age;
        x[(row, 2)] = age * age;
        x[(row, 3)] = sex;
        x[(row, 4)] = age * sex;
        x[(row, 5)] = age * age * sex;
    }
    x
}
const BAND: (f64, f64) = (0.940, 0.960);
/// Fixed, so the whole check returns next year (`docs/adr/0001`, decision 18).
const BASE_SEED: u64 = 2_026_08_11;

// ---------------------------------------------------------------- the roster

/// One person: their parents, or none if they married in or founded the family.
#[derive(Clone, Copy)]
struct Person {
    mother: Option<usize>,
    father: Option<usize>,
}

/// Three generations: a founding couple, their children, the spouses those
/// children married, and the grandchildren. Fourteen people, and between them
/// parent–offspring, full sibling, grandparental, avuncular and cousin
/// relationships — which is the point. A roster of sibling pairs would be
/// easier and would check the estimator on a pedigree nobody analyses.
fn extended_family() -> Vec<Person> {
    let founder = Person { mother: None, father: None };
    let mut family = vec![founder, founder]; // 0, 1: the founding couple
    for _ in 0..3 {
        family.push(Person { mother: Some(0), father: Some(1) }); // 2, 3, 4
    }
    for _ in 0..3 {
        family.push(founder); // 5, 6, 7: married in, unrelated to everyone
    }
    for child in 0..3 {
        for _ in 0..2 {
            family.push(Person {
                mother: Some(2 + child),
                father: Some(5 + child),
            }); // 8..13
        }
    }
    family
}

/// Kinship coefficients by the standard recursion, on a pedigree ordered so
/// that parents always precede their children.
fn kinship(people: &[Person]) -> DMatrix<f64> {
    let n = people.len();
    let mut phi = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let value = if i == j {
                match (people[i].mother, people[i].father) {
                    (Some(m), Some(f)) => 0.5 * (1.0 + phi[(m, f)]),
                    _ => 0.5,
                }
            } else {
                // i is the later-born of the pair, so recurse on its parents.
                match (people[i].mother, people[i].father) {
                    (Some(m), Some(f)) => 0.5 * (phi[(m, j)] + phi[(f, j)]),
                    _ => 0.0,
                }
            };
            phi[(i, j)] = value;
            phi[(j, i)] = value;
        }
    }
    phi
}

/// `families` copies of the extended family, unrelated to one another, giving a
/// block-diagonal relationship matrix — twice the kinship, as usual.
fn roster(families: usize) -> DMatrix<f64> {
    let family = extended_family();
    let size = family.len();
    let phi = kinship(&family);
    let n = families * size;
    let mut k = DMatrix::<f64>::zeros(n, n);
    for family_index in 0..families {
        let offset = family_index * size;
        for i in 0..size {
            for j in 0..size {
                k[(offset + i, offset + j)] = 2.0 * phi[(i, j)];
            }
        }
    }
    k
}

// ------------------------------------------------------------- the simulation

/// splitmix64, so that any cell can be reproduced from its seed alone.
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

/// Cholesky factors of `h2·K + (1 − h2)·I`, one per family, computed once for
/// the whole cell. Simulating a replicate is then one pass of matrix–vector
/// products rather than a fresh decomposition.
fn block_factors(k: &DMatrix<f64>, block: usize, h2: f64) -> Vec<DMatrix<f64>> {
    let families = k.nrows() / block;
    (0..families)
        .map(|family| {
            let offset = family * block;
            let mut covariance = DMatrix::<f64>::zeros(block, block);
            for i in 0..block {
                for j in 0..block {
                    covariance[(i, j)] = h2 * k[(offset + i, offset + j)];
                }
                covariance[(i, i)] += 1.0 - h2;
            }
            Cholesky::new(covariance)
                .expect("h2·K + (1 - h2)·I is positive definite for a non-inbred pedigree")
                .l()
        })
        .collect()
}

fn simulate(factors: &[DMatrix<f64>], block: usize, stream: &mut Stream) -> DVector<f64> {
    let n = factors.len() * block;
    let mut y = DVector::<f64>::zeros(n);
    for (family, factor) in factors.iter().enumerate() {
        let z = DVector::from_iterator(block, (0..block).map(|_| stream.normal()));
        let drawn = factor * z;
        for i in 0..block {
            y[family * block + i] = drawn[i];
        }
    }
    y
}

// ----------------------------------------------------------------- the scoring

/// Does the reported interval contain `truth`?
///
/// At an interior truth this is ordinary containment. At nought or one it is
/// the mixture rule, because the endpoint sitting on the bound says nothing by
/// itself about whether the bound is in the interval — which is the whole
/// substance of `docs/adr/0004`.
fn covers(fit: &asterism::Fit, truth: f64) -> bool {
    if !fit.converged {
        return false;
    }
    if truth == 0.0 {
        return fit.interval.lower == 0.0 && fit.interval.contains_lower_bound == Some(true);
    }
    if truth == 1.0 {
        return fit.interval.upper == 1.0 && fit.interval.contains_upper_bound == Some(true);
    }
    fit.interval.lower <= truth && truth <= fit.interval.upper
}

/// Clopper–Pearson, the exact interval for a binomial proportion. Used rather
/// than a normal approximation because coverage near 0.95 with 8,000 draws sits
/// close enough to the edge for the approximation to mislead.
fn clopper_pearson(successes: usize, trials: usize) -> (f64, f64) {
    let alpha = 0.05;
    let lower = if successes == 0 {
        0.0
    } else {
        Beta::new(successes as f64, (trials - successes + 1) as f64)
            .expect("valid beta parameters")
            .inverse_cdf(alpha / 2.0)
    };
    let upper = if successes == trials {
        1.0
    } else {
        Beta::new((successes + 1) as f64, (trials - successes) as f64)
            .expect("valid beta parameters")
            .inverse_cdf(1.0 - alpha / 2.0)
    };
    (lower, upper)
}

struct Cell {
    truth: f64,
    /// The largest average error across the fixed effects. A design that the
    /// restricted likelihood mishandles shows here before it shows in coverage.
    worst_beta_bias: f64,
    /// How far the worst fixed effect's Wald interval is from covering at 95
    /// per cent. This is what checks the standard errors rather than merely
    /// checking that they exist.
    worst_beta_coverage_error: f64,
    /// What the coverage would have been at a boundary truth if the endpoint
    /// sitting on the bound had been taken to mean the bound is in the interval
    /// — the obvious rule, and the one `docs/adr/0004` rejects. Only meaningful
    /// at a truth of nought or one; `None` elsewhere.
    naive: Option<f64>,
    seed: u64,
    coverage: f64,
    cp: (f64, f64),
    passes: bool,
    at_lower: f64,
    at_upper: f64,
    nonconverged: usize,
    median_width: f64,
    seconds: f64,
}

fn run_cell(k: &DMatrix<f64>, block: usize, truth: f64, index: usize) -> Cell {
    let started = Instant::now();
    let seed = BASE_SEED.wrapping_add(index as u64 * 1_000_003);
    let mut stream = Stream(seed);
    let n = k.nrows();
    let covariates = with_covariates();
    let x = design(n / block, block, covariates);
    let model = PreparedModel::build(&x, k, None).expect("the roster is valid");
    let fixed = if covariates {
        let beta = DVector::from_row_slice(&BETA);
        &x * beta
    } else {
        DVector::from_element(n, BETA[0])
    };
    let factors = block_factors(k, block, truth);

    let mut covered = 0usize;
    let mut at_lower = 0usize;
    let mut at_upper = 0usize;
    let mut nonconverged = 0usize;
    let mut naive_covered = 0usize;
    let mut beta_error = vec![0.0f64; model.fixed_effects()];
    let mut beta_covered = vec![0usize; model.fixed_effects()];
    let mut widths = Vec::with_capacity(replicates());

    for _ in 0..replicates() {
        let y = simulate(&factors, block, &mut stream) + &fixed;
        let fit = model.fit_one_trait(&y, true);
        if fit.converged {
            for (index, estimate) in fit.beta.iter().enumerate() {
                let truth = if covariates { BETA[index] } else { BETA[0] };
                beta_error[index] += estimate - truth;
                // The Wald interval for a fixed effect, scored the same way as
                // the interval for h2: every replicate counts. A standard error
                // that is merely present proves nothing; one whose interval
                // covers at 95 per cent is the right size.
                let effect = &fit.fixed_effects[index];
                if let (Some(lower), Some(upper)) = (effect.lower, effect.upper) {
                    if lower <= truth && truth <= upper {
                        beta_covered[index] += 1;
                    }
                }
            }
        }
        if covers(&fit, truth) {
            covered += 1;
        }
        if fit.converged {
            let naive_hit = if truth == 0.0 {
                fit.interval.lower == 0.0
            } else if truth == 1.0 {
                fit.interval.upper == 1.0
            } else {
                fit.interval.lower <= truth && truth <= fit.interval.upper
            };
            if naive_hit {
                naive_covered += 1;
            }
        }
        if !fit.converged {
            nonconverged += 1;
        } else {
            match fit.boundary {
                asterism::Boundary::Lower => at_lower += 1,
                asterism::Boundary::Upper => at_upper += 1,
                asterism::Boundary::Interior => {}
            }
            widths.push(fit.interval.upper - fit.interval.lower);
        }
    }

    widths.sort_by(f64::total_cmp);
    let median_width = if widths.is_empty() {
        f64::NAN
    } else {
        widths[widths.len() / 2]
    };
    let coverage = covered as f64 / replicates() as f64;
    let cp = clopper_pearson(covered, replicates());

    Cell {
        truth,
        worst_beta_coverage_error: beta_covered
            .iter()
            .map(|count| (*count as f64 / replicates() as f64 - 0.95).abs())
            .fold(0.0f64, f64::max),
        worst_beta_bias: beta_error
            .iter()
            .map(|total| (total / replicates() as f64).abs())
            .fold(0.0f64, f64::max),
        naive: (truth == 0.0 || truth == 1.0)
            .then(|| naive_covered as f64 / replicates() as f64),
        seed,
        coverage,
        cp,
        passes: cp.1 >= BAND.0 && cp.0 <= BAND.1,
        at_lower: at_lower as f64 / replicates() as f64,
        at_upper: at_upper as f64 / replicates() as f64,
        nonconverged,
        median_width,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn main() {
    let count = replicates();
    let families = families();
    let block = extended_family().len();
    let k = roster(families);
    let n = k.nrows();

    println!(
        "Coverage check: {families} extended families of {block}, n = {n}, \
         {count} replicates per cell, REML, {} fixed effects, base seed {BASE_SEED}.",
        if with_covariates() { "6" } else { "1" }
    );
    println!("A cell passes when its Clopper-Pearson interval overlaps [{:.3}, {:.3}].", BAND.0, BAND.1);
    println!();

    let started = Instant::now();
    let cells: Vec<Cell> = std::thread::scope(|scope| {
        let handles: Vec<_> = TRUTHS
            .iter()
            .enumerate()
            .map(|(index, &truth)| {
                let k = &k;
                scope.spawn(move || run_cell(k, block, truth, index))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("cell")).collect()
    });

    println!(
        "{:>6} {:>9} {:>18} {:>7} {:>9} {:>9} {:>7} {:>9} {:>9} {:>9}",
        "truth", "coverage", "95% CP interval", "passes", "at 0", "at 1", "failed", "width", "beta bias", "beta cov"
    );
    for cell in &cells {
        println!(
            "{:>6.2} {:>9.4} [{:>7.4}, {:>7.4}] {:>7} {:>9.4} {:>9.4} {:>7} {:>9.4} {:>9} {:>9}",
            cell.truth,
            cell.coverage,
            cell.cp.0,
            cell.cp.1,
            if cell.passes { "yes" } else { "NO" },
            cell.at_lower,
            cell.at_upper,
            cell.nonconverged,
            cell.median_width,
            format!("{:.5}", cell.worst_beta_bias),
            format!("{:.4}", 0.95 - cell.worst_beta_coverage_error),
        );
    }

    let failures: Vec<&Cell> = cells.iter().filter(|c| !c.passes).collect();
    println!();
    println!(
        "{} of {} cells pass, in {:.1} s.",
        cells.len() - failures.len(),
        cells.len(),
        started.elapsed().as_secs_f64()
    );

    // The record, in the shape the evidence file keeps.
    println!();
    println!("[");
    for (index, cell) in cells.iter().enumerate() {
        println!(
            "  {{\"true_h2\": {}, \"n\": {}, \"replicates\": {}, \"seed\": {}, \
             \"estimator\": \"reml\", \"coverage\": {}, \"cp_lower\": {}, \"cp_upper\": {}, \
             \"band\": [{}, {}], \"passes\": {}, \"fraction_at_zero\": {}, \
             \"fraction_at_one\": {}, \"nonconverged\": {}, \"median_width\": {}, \
             \"naive_boundary_coverage\": {}, \"worst_beta_bias\": {}, \"worst_beta_coverage_error\": {}, \
             \"seconds\": {:.3}}}{}",
            cell.truth,
            n,
            replicates(),
            cell.seed,
            cell.coverage,
            cell.cp.0,
            cell.cp.1,
            BAND.0,
            BAND.1,
            cell.passes,
            cell.at_lower,
            cell.at_upper,
            cell.nonconverged,
            cell.median_width,
            cell.naive.map_or_else(|| "null".to_owned(), |v| v.to_string()),
            cell.worst_beta_bias,
            cell.worst_beta_coverage_error,
            cell.seconds,
            if index + 1 == cells.len() { "" } else { "," }
        );
    }
    println!("]");

    if !failures.is_empty() {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        block_factors, covers, extended_family, kinship, roster, simulate, PreparedModel, Stream,
    };

    /// Textbook kinship coefficients. If any of these is wrong the whole check
    /// is measuring coverage against the wrong relationships.
    #[test]
    fn the_pedigree_has_the_relationships_it_claims() {
        let phi = kinship(&extended_family());
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;

        assert!(close(phi[(0, 0)], 0.5), "a non-inbred person with themselves");
        assert!(close(phi[(0, 1)], 0.0), "the founding couple are unrelated");
        assert!(close(phi[(0, 2)], 0.25), "parent and offspring");
        assert!(close(phi[(2, 3)], 0.25), "full siblings");
        assert!(close(phi[(2, 5)], 0.0), "a spouse married in");
        assert!(close(phi[(0, 8)], 0.125), "grandparent and grandchild");
        assert!(close(phi[(3, 8)], 0.125), "aunt and nephew");
        assert!(close(phi[(8, 9)], 0.25), "full siblings again");
        assert!(close(phi[(8, 10)], 0.0625), "first cousins");
        assert!(close(phi[(5, 6)], 0.0), "two spouses married in");
    }

    #[test]
    fn families_are_unrelated_to_one_another() {
        let k = roster(3);
        let size = extended_family().len();
        assert_eq!(k.nrows(), 3 * size);
        // Anyone in the first family against anyone in the second.
        for i in 0..size {
            for j in size..(2 * size) {
                assert_eq!(k[(i, j)], 0.0);
            }
        }
    }

    /// A reduced check, so that a break in the interval shows up in the ordinary
    /// test run rather than only when somebody remembers to run the binary.
    /// 1,200 replicates puts the Monte Carlo standard error near 0.006, so the
    /// band here is deliberately wider than the one the real check uses.
    #[test]
    fn coverage_holds_at_the_boundary_and_in_the_middle() {
        let block = extended_family().len();
        let k = roster(25);
        let x = nalgebra::DMatrix::from_element(k.nrows(), 1, 1.0);
        let model = PreparedModel::build(&x, &k, None).expect("valid roster");

        for (truth, seed) in [(0.0, 7u64), (0.5, 8u64)] {
            let factors = block_factors(&k, block, truth);
            let mut stream = Stream(seed);
            let mut covered = 0usize;
            let mut naive = 0usize;
            let replicates = 1_200;
            for _ in 0..replicates {
                let y = simulate(&factors, block, &mut stream);
                let fit = model.fit_one_trait(&y, true);
                if covers(&fit, truth) {
                    covered += 1;
                }
                if fit.converged && fit.interval.lower == 0.0 {
                    naive += 1;
                }
            }
            let coverage = covered as f64 / replicates as f64;
            assert!(
                (0.930..=0.970).contains(&coverage),
                "coverage {coverage} at a true heritability of {truth}"
            );
            if truth == 0.0 {
                // Counting a lower endpoint of nought as containing nought is
                // the obvious rule and it over-covers. If this ever stops being
                // true the mixture rule has stopped doing anything.
                let naive_coverage = naive as f64 / replicates as f64;
                assert!(
                    naive_coverage > coverage + 0.01,
                    "the naive rule gave {naive_coverage} against the mixture's {coverage}"
                );
            }
        }
    }

    /// The check simulates at a true heritability of one, where the covariance
    /// is the relationship matrix alone. That draw only exists if the matrix is
    /// positive definite rather than merely semi-definite.
    #[test]
    fn the_relationship_matrix_is_positive_definite() {
        let k = roster(1);
        let eigen = nalgebra::SymmetricEigen::new(k);
        let smallest = eigen.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(smallest > 1e-9, "smallest eigenvalue was {smallest}");
    }
}
