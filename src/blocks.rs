//! Splitting a relationship matrix into the family blocks it is made of.

use nalgebra::DMatrix;

/// Partition an explicitly ordered relationship matrix into its exact nonzero
/// connected blocks.
///
/// A relationship matrix over unrelated families is block diagonal, and
/// diagonalising the blocks separately preserves the likelihood exactly while
/// avoiding a cubic decomposition of the whole roster. That saving is what
/// decides whether a parametric bootstrap is affordable at scale, so it is here
/// from the start rather than added later.
pub(crate) fn family_blocks(matrix: &DMatrix<f64>) -> Vec<Vec<usize>> {
    let size = matrix.nrows();
    let mut visited = vec![false; size];
    let mut blocks = Vec::new();
    for root in 0..size {
        if visited[root] {
            continue;
        }
        let mut stack = vec![root];
        visited[root] = true;
        let mut block = Vec::new();
        while let Some(index) = stack.pop() {
            block.push(index);
            for column in 0..size {
                if !visited[column] && column != index && matrix[(index, column)] != 0.0 {
                    visited[column] = true;
                    stack.push(column);
                }
            }
        }
        block.sort_unstable();
        blocks.push(block);
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::family_blocks;
    use nalgebra::DMatrix;

    #[test]
    fn an_identity_matrix_is_all_singletons() {
        let matrix = DMatrix::<f64>::identity(4, 4);
        assert_eq!(family_blocks(&matrix).len(), 4);
    }

    #[test]
    fn two_families_are_found_and_each_is_ordered() {
        let mut matrix = DMatrix::<f64>::identity(4, 4);
        matrix[(0, 2)] = 0.5;
        matrix[(2, 0)] = 0.5;
        let blocks = family_blocks(&matrix);
        assert_eq!(blocks, vec![vec![0, 2], vec![1], vec![3]]);
    }

    #[test]
    fn a_dense_matrix_is_one_block() {
        let matrix = DMatrix::<f64>::from_element(5, 5, 0.5);
        assert_eq!(family_blocks(&matrix), vec![vec![0, 1, 2, 3, 4]]);
    }
}
