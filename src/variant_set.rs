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

    /// Test one variant set, given `Z = G W`: the dosages with their column
    /// weights already applied, one row per person and one column per variant.
    ///
    /// # Errors
    ///
    /// Returns a stable code where the matrix is the wrong shape, is not
    /// finite, or carries nothing to test.
    pub fn test(&self, kernel_root: &DMatrix<f64>) -> Result<VariantSetTest, &'static str> {
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

        // Q = || Z' P y ||^2
        let projected_root = kernel_root.transpose() * &self.residual;
        let statistic = projected_root.dot(&projected_root);

        // M = Z' P Z = Z' V0^-1 Z - (X' V0^-1 Z)' (X' V0^-1 X)^-1 (X' V0^-1 Z)
        let weighted_root = self.covariance.solve_matrix(kernel_root);
        let cross = self.design.transpose() * &weighted_root;
        let mut middle = kernel_root.transpose() * &weighted_root;
        middle -= cross.transpose() * self.design_information.solve_matrix(&cross);
        // Symmetry is exact in theory and drifts in arithmetic; the
        // decomposition below assumes it, so it is imposed rather than hoped for.
        let symmetric = (&middle + &middle.transpose()) * 0.5;

        let decomposition = SymmetricEigen::new(symmetric);
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
