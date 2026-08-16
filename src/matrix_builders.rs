//! Builders for observed-variant and local-IBD relationship matrices.
//!
//! Each builder consumes in-memory arrays and returns an ordinary dense matrix.
//! Equations and scaling are described in `docs/statistical-methods.md`.

use nalgebra::{DMatrix, DVector};

/// A submitted array could not make the relationship matrix requested.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatrixBuildError {
    NoSubjects,
    MatrixValueNotFinite,
    NoVariants,
    WeightCountMismatch,
    GenotypeNotFinite,
    GenotypeOutOfRange,
    WeightNotFinite,
    WeightNegative,
    WeightsAllZero,
    WeightedValuesAllZero,
    NoLineageDraws,
    LineageDrawSizeMismatch,
    DrawWeightCountMismatch,
    DrawWeightNotFinite,
    DrawWeightNegative,
    DrawWeightsAllZero,
    DrawWeightSumNotFinite,
    LineageWrongShape,
    LineageNegative,
}

impl MatrixBuildError {
    /// Stable code exposed through both Rust and Python.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NoSubjects => "MATRIX_NO_SUBJECTS",
            Self::MatrixValueNotFinite => "MATRIX_VALUE_NOT_FINITE",
            Self::NoVariants => "GENE_MATRIX_NO_VARIANTS",
            Self::WeightCountMismatch => "GENE_MATRIX_WEIGHT_COUNT_MISMATCH",
            Self::GenotypeNotFinite => "GENE_MATRIX_GENOTYPE_NOT_FINITE",
            Self::GenotypeOutOfRange => "GENE_MATRIX_GENOTYPE_OUT_OF_RANGE",
            Self::WeightNotFinite => "GENE_MATRIX_WEIGHT_NOT_FINITE",
            Self::WeightNegative => "GENE_MATRIX_WEIGHT_NEGATIVE",
            Self::WeightsAllZero => "GENE_MATRIX_WEIGHTS_ALL_ZERO",
            Self::WeightedValuesAllZero => "GENE_MATRIX_WEIGHTED_VALUES_ALL_ZERO",
            Self::NoLineageDraws => "LOCAL_IBD_NO_DRAWS",
            Self::LineageDrawSizeMismatch => "LOCAL_IBD_DRAW_SIZE_MISMATCH",
            Self::DrawWeightCountMismatch => "LOCAL_IBD_WEIGHT_COUNT_MISMATCH",
            Self::DrawWeightNotFinite => "LOCAL_IBD_WEIGHT_NOT_FINITE",
            Self::DrawWeightNegative => "LOCAL_IBD_WEIGHT_NEGATIVE",
            Self::DrawWeightsAllZero => "LOCAL_IBD_WEIGHTS_ALL_ZERO",
            Self::DrawWeightSumNotFinite => "LOCAL_IBD_WEIGHT_SUM_NOT_FINITE",
            Self::LineageWrongShape => "LOCAL_IBD_LINEAGE_WRONG_SHAPE",
            Self::LineageNegative => "LOCAL_IBD_LINEAGE_NEGATIVE",
        }
    }
}

fn checked_output(values: DMatrix<f64>) -> Result<DMatrix<f64>, MatrixBuildError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MatrixBuildError::MatrixValueNotFinite);
    }
    Ok(values)
}

fn checked_weights(
    variants: usize,
    weights: Option<&DVector<f64>>,
) -> Result<DVector<f64>, MatrixBuildError> {
    let weights = weights
        .cloned()
        .unwrap_or_else(|| DVector::from_element(variants, 1.0));
    if weights.len() != variants {
        return Err(MatrixBuildError::WeightCountMismatch);
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(MatrixBuildError::WeightNotFinite);
    }
    if weights.iter().any(|value| *value < 0.0) {
        return Err(MatrixBuildError::WeightNegative);
    }
    if weights.iter().all(|value| *value == 0.0) {
        return Err(MatrixBuildError::WeightsAllZero);
    }
    Ok(weights)
}

fn checked_genotypes(genotypes: &DMatrix<f64>) -> Result<(), MatrixBuildError> {
    if genotypes.nrows() == 0 {
        return Err(MatrixBuildError::NoSubjects);
    }
    if genotypes.ncols() == 0 {
        return Err(MatrixBuildError::NoVariants);
    }
    if genotypes.iter().any(|value| !value.is_finite()) {
        return Err(MatrixBuildError::GenotypeNotFinite);
    }
    if genotypes.iter().any(|value| !(0.0..=2.0).contains(value)) {
        return Err(MatrixBuildError::GenotypeOutOfRange);
    }
    Ok(())
}

/// Build `K = (G W)(G W)'` from submitted alternate-allele dosages.
///
/// The entries of `variant_weights` multiply columns directly, so their
/// squares are the variance weights. Nothing is centred, imputed or rescaled.
///
/// # Errors
///
/// Refuses no subjects, an empty variant set, missing or out-of-range dosage,
/// invalid weights, an identically nought weighted matrix, or non-finite
/// arithmetic output.
pub fn gene_linear_matrix(
    genotypes: &DMatrix<f64>,
    variant_weights: Option<&DVector<f64>>,
) -> Result<DMatrix<f64>, MatrixBuildError> {
    checked_genotypes(genotypes)?;
    let weights = checked_weights(genotypes.ncols(), variant_weights)?;
    let mut weighted = genotypes.clone();
    for column in 0..weighted.ncols() {
        weighted.column_mut(column).scale_mut(weights[column]);
    }
    if weighted.iter().all(|value| *value == 0.0) {
        return Err(MatrixBuildError::WeightedValuesAllZero);
    }
    let values = &weighted * weighted.transpose();
    checked_output(values)
}

/// Build the rank-one burden relationship matrix `K = b b'`, where
/// `b_i = sum_j G_ij w_j`.
///
/// The nonnegative weights multiply dosage columns directly. Nothing is
/// centred, imputed or rescaled.
///
/// # Errors
///
/// Refuses the same malformed inputs as [`gene_linear_matrix`].
pub fn gene_burden_matrix(
    genotypes: &DMatrix<f64>,
    variant_weights: Option<&DVector<f64>>,
) -> Result<DMatrix<f64>, MatrixBuildError> {
    checked_genotypes(genotypes)?;
    let weights = checked_weights(genotypes.ncols(), variant_weights)?;
    let burden = genotypes * weights;
    if burden.iter().all(|value| *value == 0.0) {
        return Err(MatrixBuildError::WeightedValuesAllZero);
    }
    let values = &burden * burden.transpose();
    checked_output(values)
}

fn matching_lineage_pairs(first: [u64; 2], second: [u64; 2]) -> f64 {
    let mut matches = 0_u8;
    for one in first {
        for two in second {
            if one == two {
                matches += 1;
            }
        }
    }
    f64::from(matches) / 2.0
}

/// Build the local additive relationship matrix from two founder-haplotype
/// labels per subject.
///
/// Labels are opaque equality tokens. The result is `H H' / 2`: IBD0, IBD1
/// and IBD2 are 0, 0.5 and 1 off the diagonal, while local autozygosity is
/// retained as a diagonal of 2.
///
/// # Errors
///
/// Refuses no subjects. Missing lineage labels have no representation in this
/// interface and must not be encoded as an ordinary integer.
pub fn local_ibd_matrix(lineages: &[[u64; 2]]) -> Result<DMatrix<f64>, MatrixBuildError> {
    if lineages.is_empty() {
        return Err(MatrixBuildError::NoSubjects);
    }
    let rows = lineages.len();
    let mut values = DMatrix::<f64>::zeros(rows, rows);
    for i in 0..rows {
        for j in 0..=i {
            let value = matching_lineage_pairs(lineages[i], lineages[j]);
            values[(i, j)] = value;
            values[(j, i)] = value;
        }
    }
    checked_output(values)
}

/// Average additive local-IBD matrices over posterior lineage draws.
///
/// `draw_weights` are normalised to sum to one. Omitting them assigns equal
/// weight to every draw. No malformed draw is dropped.
///
/// # Errors
///
/// Refuses no draws, no subjects, a draw with the wrong number of subjects, or
/// malformed draw weights.
pub fn posterior_local_ibd_matrix(
    lineage_draws: &[Vec<[u64; 2]>],
    draw_weights: Option<&DVector<f64>>,
) -> Result<DMatrix<f64>, MatrixBuildError> {
    let first = lineage_draws
        .first()
        .ok_or(MatrixBuildError::NoLineageDraws)?;
    if first.is_empty() {
        return Err(MatrixBuildError::NoSubjects);
    }
    if lineage_draws.iter().any(|draw| draw.len() != first.len()) {
        return Err(MatrixBuildError::LineageDrawSizeMismatch);
    }
    let weights = draw_weights
        .cloned()
        .unwrap_or_else(|| DVector::from_element(lineage_draws.len(), 1.0));
    if weights.len() != lineage_draws.len() {
        return Err(MatrixBuildError::DrawWeightCountMismatch);
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(MatrixBuildError::DrawWeightNotFinite);
    }
    if weights.iter().any(|value| *value < 0.0) {
        return Err(MatrixBuildError::DrawWeightNegative);
    }
    let weight_sum: f64 = weights.iter().sum();
    if !weight_sum.is_finite() {
        return Err(MatrixBuildError::DrawWeightSumNotFinite);
    }
    if weight_sum == 0.0 {
        return Err(MatrixBuildError::DrawWeightsAllZero);
    }

    let rows = first.len();
    let mut values = DMatrix::<f64>::zeros(rows, rows);
    for (draw, weight) in lineage_draws.iter().zip(weights.iter()) {
        let probability = *weight / weight_sum;
        if probability == 0.0 {
            continue;
        }
        for i in 0..rows {
            for j in 0..=i {
                let value = probability * matching_lineage_pairs(draw[i], draw[j]);
                values[(i, j)] += value;
                if i != j {
                    values[(j, i)] += value;
                }
            }
        }
    }
    checked_output(values)
}

#[cfg(feature = "python")]
pub mod python;

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use super::{
        gene_burden_matrix, gene_linear_matrix, local_ibd_matrix, posterior_local_ibd_matrix,
    };

    #[test]
    fn weighted_linear_gene_values_follow_the_recorded_equation() {
        let genotypes = DMatrix::from_row_slice(3, 2, &[0.0, 1.0, 1.0, 2.0, 2.0, 0.0]);
        let weights = DVector::from_vec(vec![1.0, 2.0]);

        let built = gene_linear_matrix(&genotypes, Some(&weights)).expect("valid");

        assert_eq!(
            built,
            DMatrix::from_row_slice(3, 3, &[4.0, 8.0, 0.0, 8.0, 17.0, 2.0, 0.0, 2.0, 4.0,],)
        );
    }

    #[test]
    fn burden_gene_values_are_the_outer_product_of_the_weighted_burden() {
        let genotypes = DMatrix::from_row_slice(3, 2, &[0.0, 1.0, 1.0, 2.0, 2.0, 0.0]);
        let weights = DVector::from_vec(vec![1.0, 2.0]);

        let built = gene_burden_matrix(&genotypes, Some(&weights)).expect("valid");

        assert_eq!(
            built,
            DMatrix::from_row_slice(3, 3, &[4.0, 10.0, 4.0, 10.0, 25.0, 10.0, 4.0, 10.0, 4.0,],)
        );
    }

    #[test]
    fn hard_lineages_make_additive_local_ibd_and_keep_autozygosity() {
        let lineages = [[1, 2], [1, 3], [4, 4]];

        let built = local_ibd_matrix(&lineages).expect("valid");

        assert_eq!(
            built,
            DMatrix::from_row_slice(3, 3, &[1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 2.0,],)
        );
    }

    #[test]
    fn posterior_lineages_make_the_explicitly_weighted_mean() {
        let draws = vec![vec![[1, 2], [1, 3], [4, 4]], vec![[1, 2], [3, 4], [4, 4]]];
        let weights = DVector::from_vec(vec![1.0, 3.0]);

        let built = posterior_local_ibd_matrix(&draws, Some(&weights)).expect("valid");

        assert_eq!(
            built,
            DMatrix::from_row_slice(3, 3, &[1.0, 0.125, 0.0, 0.125, 1.0, 0.75, 0.0, 0.75, 2.0,],)
        );
    }

    #[test]
    fn finite_inputs_that_overflow_are_refused() {
        let genotypes = DMatrix::from_element(2, 1, 2.0);
        let weights = DVector::from_element(1, 1e308);

        let linear = gene_linear_matrix(&genotypes, Some(&weights)).unwrap_err();
        let burden = gene_burden_matrix(&genotypes, Some(&weights)).unwrap_err();

        assert_eq!(linear.code(), "MATRIX_VALUE_NOT_FINITE");
        assert_eq!(burden.code(), "MATRIX_VALUE_NOT_FINITE");
    }
}
