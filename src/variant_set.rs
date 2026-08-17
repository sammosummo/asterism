//! A score test for a whole set of variants at once, in the famSKAT form.
//!
//! Testing rare variants one at a time finds nothing: each has a handful of
//! carriers. The variance-component answer is to ask whether the variants in a
//! set — a gene, a pathway — carry more trait variance together than chance
//! allows, without committing to which of them matters or which direction each
//! pushes.
//!
//! ```text
//! y = X beta + Z gamma + g + e,   gamma_j ~ (0, tau),   g ~ N(0, sigma_g^2 A)
//! ```
//!
//! where `Z = G W` is the genotype matrix with its column weights already
//! applied, and the test is of `tau = 0`.
//!
//! # Why a score test rather than a likelihood ratio
//!
//! The null is on a boundary, and Asterism's usual 50:50 chi-bar-square
//! reference is right only when the tested matrix spreads across many
//! eigenvalues. A variant-set kernel does not: a burden kernel has rank one.
//! Measured on a rare-variant kernel under the null, the likelihood ratio
//! rejected 0.020 against a nominal 0.05 — badly conservative — where a score
//! test reached 0.0467.
//!
//! A score test also fits the null **once** for a whole scan rather than
//! refitting per gene, and its null distribution is computed rather than
//! assumed. With `r = P y` and `P` the residual projection at the fitted null
//! covariance,
//!
//! ```text
//! Q      = || Z' r ||^2
//! Q      ~ sum_i lambda_i chi-square(1)  under the null
//! lambda = eigenvalues of Z' P Z
//! ```
//!
//! # Why the root and not the kernel
//!
//! The interface takes `Z`, not `K = Z Z'`. Nothing is lost — they carry the
//! same information — and everything is cheaper: no `n by n` kernel is ever
//! formed, memory is one column per variant rather than one per person
//! squared, and the eigenvalues come from a matrix the size of the variant set
//! rather than the roster. For a gene of twenty variants in two thousand
//! people that is a twenty by twenty decomposition instead of two thousand by
//! two thousand.
//!
//! # What this does not do
//!
//! It treats the fitted null variances as known, which is the ordinary score
//! approximation and the one famSKAT makes. It corrects for nothing about
//! testing many sets. It chooses no weights: `Z` arrives weighted, and that
//! choice is the caller's and is not innocent — see
//! `docs/statistical-methods.md`.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::components::ComponentModel;
use crate::dense::DenseFactor;
use crate::mixture_tail::weighted_chi2_upper_tail;

#[cfg(feature = "python")]
pub mod python;

/// The result of testing one variant set.
#[derive(Clone, Debug)]
pub struct VariantSetTest {
    /// `|| Z' P y ||^2`.
    pub statistic: f64,
    pub p_value: f64,
    /// The weights of the chi-square mixture the statistic is read against,
    /// largest first. Their spread is what decides whether a 50:50 reference
    /// would have been anywhere near right.
    pub eigenvalues: Vec<f64>,
    /// How the tail was computed, and whether it is still readable as a
    /// probability rather than lost to cancellation.
    pub tail_method: &'static str,
    pub trustworthy: bool,
}

/// A null model fitted once, ready to test many variant sets against.
pub struct VariantSetModel {
    design: DMatrix<f64>,
    /// `V0^-1 X`.
    weighted_design: DMatrix<f64>,
    /// The factorised `X' V0^-1 X`.
    design_information: DenseFactor,
    /// `P y`, the projected residual.
    residual: DVector<f64>,
    covariance: DenseFactor,
    variances: Vec<f64>,
    rows: usize,
}

impl VariantSetModel {
    /// Fit the null model once, for a whole scan.
    ///
    /// `background` are the covariance bases carrying everything that is not
    /// the variant set — a relationship or genomic relationship matrix, and
    /// whatever else the design needs. `design` must include its own intercept.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the inputs do not describe a model or the
    /// null fit fails.
    pub fn build(
        background: &[DMatrix<f64>],
        design: &DMatrix<f64>,
        y: &DVector<f64>,
        reml: bool,
    ) -> Result<Self, &'static str> {
        let rows = design.nrows();
        if rows == 0 {
            return Err("VARIANT_SET_NO_ROWS");
        }
        if y.len() != rows {
            return Err("VARIANT_SET_RESPONSE_WRONG_LENGTH");
        }
        for matrix in background {
            if matrix.nrows() != rows || matrix.ncols() != rows {
                return Err("VARIANT_SET_BACKGROUND_WRONG_SHAPE");
            }
        }
        if background.iter().any(|m| m.iter().any(|v| !v.is_finite()))
            || design.iter().any(|v| !v.is_finite())
            || y.iter().any(|v| !v.is_finite())
        {
            return Err("VARIANT_SET_NOT_FINITE");
        }

        let null = ComponentModel::build(background, design)?.fit(y, reml)?;
        // **This null defines the projection for every set in the scan.** The
        // component fit returns its best start whether or not that start
        // converged, recording the fact in a flag; a null that stopped short
        // gives a wrong covariance, hence a wrong projection, hence wrong
        // eigenvalues and wrong p-values for every gene tested, with nothing in
        // any of them saying so. One fit is cheap to check and thousands of
        // results depend on it, so it is checked here rather than reported
        // afterwards.
        if !null.converged {
            return Err("VARIANT_SET_NULL_DID_NOT_CONVERGE");
        }
        // The residual variance is last, and the structured coefficients
        // precede it in the order the matrices were given.
        let mut covariance = DMatrix::<f64>::identity(rows, rows)
            * null.variances[null.variances.len() - 1];
        for (matrix, variance) in background.iter().zip(&null.variances) {
            covariance += matrix * *variance;
        }
        let factor =
            DenseFactor::new(&covariance).ok_or("VARIANT_SET_COVARIANCE_NOT_POSITIVE_DEFINITE")?;

        let weighted_design = factor.solve_matrix(design);
        let information = design.transpose() * &weighted_design;
        let information_factor =
            DenseFactor::new(&information).ok_or("VARIANT_SET_DESIGN_RANK_DEFICIENT")?;

        // r = P y = V0^-1 y - V0^-1 X (X' V0^-1 X)^-1 X' V0^-1 y
        let weighted_response = factor.solve_vector(y);
        let projected = information_factor.solve_vector(&(design.transpose() * &weighted_response));
        let residual = &weighted_response - &weighted_design * projected;

        Ok(Self {
            design: design.clone(),
            weighted_design,
            design_information: information_factor,
            residual,
            covariance: factor,
            variances: null.variances,
            rows,
        })
    }

    /// The fitted null variances, structured components first and the residual
    /// last.
    #[must_use]
    pub fn null_variances(&self) -> &[f64] {
        &self.variances
    }

    /// The projected score `s = Z' P y` and the middle matrix `M = Z' P Z`,
    /// which every value of `rho` below is built from.
    fn projected(
        &self,
        kernel_root: &DMatrix<f64>,
    ) -> Result<(DVector<f64>, DMatrix<f64>), &'static str> {
        if kernel_root.nrows() != self.rows {
            return Err("VARIANT_SET_ROOT_WRONG_SHAPE");
        }
        if kernel_root.ncols() == 0 {
            return Err("VARIANT_SET_NO_VARIANTS");
        }
        if kernel_root.iter().any(|v| !v.is_finite()) {
            return Err("VARIANT_SET_NOT_FINITE");
        }
        // A set nobody in the roster carries has nothing to test, and says so
        // rather than returning a statistic of nought as though it were news.
        if kernel_root.iter().all(|v| *v == 0.0) {
            return Err("VARIANT_SET_NO_CARRIERS");
        }
        let score = kernel_root.transpose() * &self.residual;
        let weighted_root = self.covariance.solve_matrix(kernel_root);
        let cross = self.design.transpose() * &weighted_root;
        let mut middle = kernel_root.transpose() * &weighted_root;
        middle -= cross.transpose() * self.design_information.solve_matrix(&cross);
        // Symmetry is exact in theory and drifts in arithmetic; the
        // decomposition below assumes it, so it is imposed rather than hoped for.
        let symmetric = (&middle + &middle.transpose()) * 0.5;
        Ok((score, symmetric))
    }

    /// Test one variant set, given `Z = G W`: the dosages with their column
    /// weights already applied, one row per person and one column per variant.
    ///
    /// This is the variance-component test, which assumes nothing about the
    /// direction of the variant effects. For a set where they might all push
    /// the same way, see [`Self::test_family`].
    ///
    /// # Errors
    ///
    /// Returns a stable code where the matrix is the wrong shape, is not
    /// finite, or carries nothing to test.
    pub fn test(&self, kernel_root: &DMatrix<f64>) -> Result<VariantSetTest, &'static str> {
        let (score, middle) = self.projected(kernel_root)?;
        let statistic = score.dot(&score);
        let decomposition = SymmetricEigen::new(middle);
        let mut eigenvalues: Vec<f64> = decomposition
            .eigenvalues
            .iter()
            .copied()
            .map(|value| value.max(0.0))
            .collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).expect("finite eigenvalues"));

        let tail = weighted_chi2_upper_tail(statistic, &eigenvalues)?;
        Ok(VariantSetTest {
            statistic,
            p_value: tail.probability,
            eigenvalues,
            tail_method: tail.method,
            trustworthy: tail.trustworthy,
        })
    }
}

/// One variant set tested across a family of assumptions about how its
/// variants act together, and the combination of those tests.
#[derive(Clone, Debug)]
pub struct VariantSetFamily {
    /// The assumed correlation between variant effects, in the order tested.
    /// Nought assumes nothing about direction; one assumes they all act alike.
    pub correlations: Vec<f64>,
    pub tests: Vec<VariantSetTest>,
    /// The correlation whose test came out strongest. It is a description of
    /// this data set and not an estimate of anything.
    pub strongest_correlation: f64,
    /// The combined p-value, which is what to report.
    pub p_value: f64,
    /// False where any member of the family was itself unreadable.
    pub trustworthy: bool,
}

/// Below this a p-value is transformed as `1/(p pi)`, which is the same number
/// as the tangent without the cancellation it suffers near a right angle.
const SMALL_TANGENT: f64 = 1e-6;
/// The range a p-value is held to before combining. The lower end is only
/// underflow protection; the upper end is a statement that p-values above it
/// are not told apart, which is what stops one of them annihilating the rest.
const SMALLEST_COMBINED: f64 = 1e-300;
const LARGEST_COMBINED: f64 = 1.0 - 1e-6;

/// Combine dependent p-values by the Cauchy method.
///
/// The average of `tan((1/2 - p) pi)` is Cauchy in its tail whatever the
/// dependence between the tests, which is what makes this usable here: the
/// members of a `rho` family are strongly dependent, being the same score read
/// under different assumptions, and no ordinary combination survives that.
///
/// The transformation is written as `1/(p pi)` for small `p`, which is the same
/// number without the cancellation `tan` suffers as its argument approaches a
/// right angle.
fn cauchy_combination(p_values: &[f64]) -> f64 {
    let count = p_values.len() as f64;
    let mut total = 0.0;
    for p in p_values {
        // **Both ends of the transform run away, and only one of them should.**
        // `tan((1/2 - p) pi)` goes to plus infinity as `p` falls and to minus
        // infinity as `p` rises to one, so a single member at exactly one
        // annihilates the rest: six members at 1e-8 combine to 1e-8, and the
        // same six beside one member at 1.0 combine to 0.9999999999999989 -- a
        // set with overwhelming signal reported as nothing.
        //
        // A clamp of `1 - 1e-16` does not help, because that rounds to the
        // nearest double below one and its tangent is still -2e15. The upper
        // end is therefore held at `1 - 1e-6`, which says that p-values above
        // that are not distinguished from each other. Nothing is lost: they all
        // mean the same thing, which is no evidence, and the method is meant to
        // be led by its smallest members rather than shouted down by its
        // largest. A statistic of exactly nought reaches here as a p-value of
        // exactly one, so this is an ordinary input and not a corner.
        let p = p.clamp(SMALLEST_COMBINED, LARGEST_COMBINED);
        // `tan((1/2 - p) pi)` loses its accuracy as the argument nears a
        // right angle, and `1/(p pi)` is the same number without that.
        total += if p < SMALL_TANGENT {
            1.0 / (p * std::f64::consts::PI)
        } else {
            ((0.5 - p) * std::f64::consts::PI).tan()
        };
    }
    let mean = total / count;
    // The upper tail of a standard Cauchy. Written as `1/2 - atan(x)/pi` it
    // cancels badly once the answer is small: at p = 1e-9 that loses eight of
    // the sixteen digits. The identity `atan(1/x)/pi` is the same number with
    // no subtraction in it, so it is used wherever the statistic is above one.
    if mean > 1.0 {
        (1.0 / mean).atan() / std::f64::consts::PI
    } else {
        0.5 - mean.atan() / std::f64::consts::PI
    }
}

impl VariantSetModel {
    /// Test one variant set across a family of assumptions, and combine them.
    ///
    /// Two tests bet on different truths about a set, and neither wins in
    /// general. A **burden** test assumes every variant pushes the trait the
    /// same way, adds them into one score and tests that: powerful when true,
    /// and blind when half the variants raise the trait and half lower it,
    /// because they cancel in the sum. A **variance-component** test assumes
    /// nothing about direction and asks only whether the effects are more
    /// scattered than chance allows: robust to mixed directions, weaker when
    /// they genuinely do agree.
    ///
    /// They are the two ends of one dial. `correlations` is that dial: the
    /// assumed correlation between variant effects, nought giving the
    /// variance-component test and one the burden test.
    ///
    /// **Taking the best of several tests inflates a p-value unless the
    /// looking is paid for**, and the members of this family are strongly
    /// dependent, being one score read under different assumptions. The
    /// combination is therefore the Cauchy method, whose tail is right
    /// whatever the dependence. `strongest_correlation` is reported because it
    /// says something about the set, but it is a description and not an
    /// estimate: reporting its p-value alone would be the inflation this
    /// exists to avoid.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the set cannot be tested, or where a
    /// correlation lies outside `[0, 1)`. One is excluded because the burden
    /// end is reached exactly by the rank-one root and needs no dial.
    pub fn test_family(
        &self,
        kernel_root: &DMatrix<f64>,
        correlations: &[f64],
    ) -> Result<VariantSetFamily, &'static str> {
        if correlations.is_empty() {
            return Err("VARIANT_SET_NO_CORRELATIONS");
        }
        if correlations
            .iter()
            .any(|r| !r.is_finite() || *r < 0.0 || *r >= 1.0)
        {
            return Err("VARIANT_SET_CORRELATION_OUTSIDE_UNIT");
        }
        let (score, middle) = self.projected(kernel_root)?;
        let variants = score.len();
        let ones = DVector::from_element(variants, 1.0);
        let summed = score.sum();
        let plain = score.dot(&score);

        let mut tests = Vec::with_capacity(correlations.len());
        for &rho in correlations {
            // Q_rho = (1 - rho) s's + rho (1's)^2, the same score read under a
            // different assumption about how the effects agree.
            let statistic = (1.0 - rho) * plain + rho * summed * summed;
            // The effects now have covariance (1-rho) I + rho 11', whose
            // square root is a scaled identity plus a rank-one piece.
            let scale = (1.0 - rho).sqrt();
            let corner = ((1.0 - rho + variants as f64 * rho).sqrt() - scale) / variants as f64;
            let root = DMatrix::<f64>::identity(variants, variants) * scale
                + &ones * ones.transpose() * corner;
            let transformed = &root * &middle * &root;
            let symmetric = (&transformed + &transformed.transpose()) * 0.5;
            let decomposition = SymmetricEigen::new(symmetric);
            let mut eigenvalues: Vec<f64> = decomposition
                .eigenvalues
                .iter()
                .copied()
                .map(|value| value.max(0.0))
                .collect();
            eigenvalues.sort_by(|a, b| b.partial_cmp(a).expect("finite eigenvalues"));
            let tail = weighted_chi2_upper_tail(statistic, &eigenvalues)?;
            tests.push(VariantSetTest {
                statistic,
                p_value: tail.probability,
                eigenvalues,
                tail_method: tail.method,
                trustworthy: tail.trustworthy,
            });
        }

        let p_values: Vec<f64> = tests.iter().map(|t| t.p_value).collect();
        let strongest = p_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite p-values"))
            .map(|(index, _)| correlations[index])
            .expect("nonempty family");
        Ok(VariantSetFamily {
            correlations: correlations.to_vec(),
            trustworthy: tests.iter().all(|t| t.trustworthy),
            strongest_correlation: strongest,
            p_value: cauchy_combination(&p_values),
            tests,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::VariantSetModel;
    use nalgebra::{DMatrix, DVector};

    /// Sibships, so the background matrix has something to do.
    fn roster(families: usize, sibs: usize) -> (DMatrix<f64>, DMatrix<f64>, usize) {
        let people = families * sibs;
        let relationship = DMatrix::from_fn(people, people, |i, j| {
            if i == j {
                1.0
            } else if i / sibs == j / sibs {
                0.5
            } else {
                0.0
            }
        });
        (relationship, DMatrix::from_element(people, 1, 1.0), people)
    }

    fn response(people: usize, seed: u64) -> DVector<f64> {
        // A deterministic, mean-zero response; the arithmetic below does not
        // care that it is not Gaussian.
        DVector::from_fn(people, |i, _| {
            let x = ((i as u64 * 2_654_435_761 + seed) % 1_000) as f64 / 1_000.0;
            x - 0.5
        })
    }

    /// A set nobody carries has nothing to test, and says so rather than
    /// returning a statistic of nought as though it were a finding.
    #[test]
    fn a_set_with_no_carriers_is_refused() {
        let (relationship, design, people) = roster(12, 4);
        let y = response(people, 7);
        let model = VariantSetModel::build(&[relationship], &design, &y, true).expect("fits");
        let empty = DMatrix::<f64>::zeros(people, 5);
        assert_eq!(model.test(&empty).err(), Some("VARIANT_SET_NO_CARRIERS"));
    }

    /// The statistic is a squared length and the p-value is a probability,
    /// whatever is handed in.
    #[test]
    fn the_statistic_and_its_tail_are_well_formed() {
        let (relationship, design, people) = roster(15, 4);
        let y = response(people, 11);
        let model = VariantSetModel::build(&[relationship], &design, &y, true).expect("fits");
        let root = DMatrix::from_fn(people, 6, |i, j| ((i * 7 + j * 13) % 3) as f64);
        let test = model.test(&root).expect("tests");
        assert!(test.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&test.p_value));
        assert_eq!(test.eigenvalues.len(), 6);
        assert!(test.eigenvalues.iter().all(|v| *v >= 0.0));
        // Sorted largest first, so a reader can see the spread at a glance.
        assert!(test.eigenvalues.windows(2).all(|w| w[0] >= w[1]));
    }

    /// Scaling the whole set scales the statistic and every eigenvalue by the
    /// same square, so the p-value cannot move. A test that failed this would
    /// depend on the units the weights happened to be in.
    #[test]
    fn rescaling_the_whole_set_leaves_the_p_value_alone() {
        let (relationship, design, people) = roster(15, 4);
        let y = response(people, 5);
        let model = VariantSetModel::build(&[relationship], &design, &y, true).expect("fits");
        let root = DMatrix::from_fn(people, 5, |i, j| ((i * 5 + j * 3) % 4) as f64);
        let plain = model.test(&root).expect("tests");
        let scaled = model.test(&(&root * 3.0)).expect("tests");
        assert!((scaled.statistic / plain.statistic - 9.0).abs() < 1e-8);
        assert!(
            (scaled.p_value - plain.p_value).abs() < 1e-9,
            "{} against {}",
            scaled.p_value,
            plain.p_value
        );
    }

    /// A column already in the fixed-effect design is projected out, so it
    /// cannot be found as a variant-set effect. This is what stops a test
    /// rediscovering its own covariates.
    #[test]
    fn a_column_of_the_design_carries_no_signal() {
        let (relationship, _, people) = roster(15, 4);
        let covariate = DVector::from_fn(people, |i, _| (i % 5) as f64);
        let design = DMatrix::from_fn(people, 2, |i, j| if j == 0 { 1.0 } else { covariate[i] });
        let y = response(people, 3);
        let model = VariantSetModel::build(&[relationship], &design, &y, true).expect("fits");
        let root = DMatrix::from_fn(people, 1, |i, _| covariate[i]);
        let test = model.test(&root).expect("tests");
        assert!(
            test.statistic < 1e-16,
            "a design column left a statistic of {}",
            test.statistic
        );
    }

    /// Nought must reproduce the plain variance-component test exactly: it is
    /// the same assumption written a second way.
    #[test]
    fn a_correlation_of_nought_is_the_plain_test() {
        let (relationship, design, people) = roster(15, 4);
        let y = response(people, 21);
        let model = VariantSetModel::build(&[relationship], &design, &y, true).expect("fits");
        let root = DMatrix::from_fn(people, 6, |i, j| ((i * 7 + j * 11) % 3) as f64);
        let plain = model.test(&root).expect("tests");
        let family = model.test_family(&root, &[0.0]).expect("tests");
        assert!((family.tests[0].statistic - plain.statistic).abs() < 1e-9);
        assert!((family.tests[0].p_value - plain.p_value).abs() < 1e-12);
    }

    /// The combination must not be better than the best test it combines --
    /// that would be the inflation it exists to prevent -- and must not be
    /// worse than the worst, which would make looking pointless.
    #[test]
    fn the_combination_costs_something_and_not_everything() {
        let (relationship, design, people) = roster(20, 4);
        let y = response(people, 33);
        let model = VariantSetModel::build(&[relationship], &design, &y, true).expect("fits");
        let root = DMatrix::from_fn(people, 8, |i, j| ((i * 3 + j * 5) % 4) as f64);
        let family = model
            .test_family(&root, &[0.0, 0.04, 0.25, 0.5, 0.9])
            .expect("tests");
        let best = family
            .tests
            .iter()
            .map(|t| t.p_value)
            .fold(f64::INFINITY, f64::min);
        let worst = family
            .tests
            .iter()
            .map(|t| t.p_value)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            family.p_value >= best - 1e-12,
            "combined {} beat its best member {best}",
            family.p_value
        );
        assert!(family.p_value <= worst + 1e-12);
        assert!((0.0..=1.0).contains(&family.p_value));
    }

    /// One member with nothing to report must not annihilate the rest. The
    /// transform runs to minus infinity as a p-value approaches one, so a
    /// single member at exactly one used to drag a set with overwhelming
    /// signal to a combined p-value of one.
    #[test]
    fn a_member_with_no_evidence_does_not_silence_the_others() {
        let strong = vec![1e-8; 6];
        let combined = super::cauchy_combination(&strong);
        assert!(combined < 1e-7, "six strong members combined to {combined}");

        let mut with_silent = strong.clone();
        with_silent.push(1.0);
        let combined = super::cauchy_combination(&with_silent);
        assert!(
            combined < 1e-6,
            "one member at p = 1 dragged the combination to {combined}"
        );
        // It should still cost something, since a member found nothing.
        assert!(combined > 1e-8);
    }

    /// A correlation of one is the burden end, which the rank-one root already
    /// reaches exactly, so the dial stops short of it rather than dividing by
    /// nought.
    #[test]
    fn the_dial_stops_short_of_one() {
        let (relationship, design, people) = roster(10, 4);
        let y = response(people, 9);
        let model = VariantSetModel::build(&[relationship], &design, &y, true).expect("fits");
        let root = DMatrix::from_fn(people, 4, |i, j| ((i + j) % 3) as f64);
        assert_eq!(
            model.test_family(&root, &[1.0]).err(),
            Some("VARIANT_SET_CORRELATION_OUTSIDE_UNIT")
        );
        assert_eq!(
            model.test_family(&root, &[]).err(),
            Some("VARIANT_SET_NO_CORRELATIONS")
        );
        // The burden end proper: one column, which is the summed score.
        let summed = DMatrix::from_fn(people, 1, |i, _| {
            (0..4).map(|j| ((i + j) % 3) as f64).sum::<f64>()
        });
        let burden = model.test(&summed).expect("tests");
        assert_eq!(burden.tail_method, "exact_single_weight");
    }

    /// The Cauchy combination of identical p-values is that p-value: combining
    /// a test with itself learns nothing and must cost nothing.
    #[test]
    fn combining_a_test_with_itself_changes_nothing() {
        for p in [0.5_f64, 0.05, 1e-4, 1e-9] {
            let combined = super::cauchy_combination(&[p, p, p]);
            assert!(
                (combined - p).abs() <= 1e-9 * p.max(1e-12),
                "combining {p} with itself gave {combined}"
            );
        }
    }

    /// Malformed input is refused rather than quietly reshaped.
    #[test]
    fn malformed_input_is_refused() {
        let (relationship, design, people) = roster(10, 4);
        let y = response(people, 1);
        let model = VariantSetModel::build(&[relationship.clone()], &design, &y, true).expect("fits");
        assert_eq!(
            model.test(&DMatrix::<f64>::zeros(people + 1, 3)).err(),
            Some("VARIANT_SET_ROOT_WRONG_SHAPE")
        );
        assert_eq!(
            model.test(&DMatrix::<f64>::zeros(people, 0)).err(),
            Some("VARIANT_SET_NO_VARIANTS")
        );
        let short = DVector::from_element(people - 1, 0.0);
        assert_eq!(
            VariantSetModel::build(&[relationship], &design, &short, true).err(),
            Some("VARIANT_SET_RESPONSE_WRONG_LENGTH")
        );
    }
}
