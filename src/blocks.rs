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

/// Partition several components into the blocks their union makes.
///
/// The likelihood factorises over blocks of the *covariance*, and the
/// covariance is a sum of components, so two rows sit in one block when **any**
/// component connects them. Taking the blocks from one matrix and then adding a
/// second component to the covariance would factorise a likelihood that does
/// not factorise, which is a wrong answer rather than a slow one.
///
/// A component that is dense joins everything, and that is a real property of
/// the model rather than an accident: a shared-environment term written as a
/// distance kernel does exactly that, which is why this package uses a
/// household kernel, which does not.
pub(crate) fn union_blocks(components: &[DMatrix<f64>]) -> Vec<Vec<usize>> {
    let Some(first) = components.first() else {
        return Vec::new();
    };
    let size = first.nrows();
    let mut union = DMatrix::<f64>::zeros(size, size);
    for component in components {
        for row in 0..size {
            for column in 0..size {
                if component[(row, column)] != 0.0 {
                    union[(row, column)] = 1.0;
                }
            }
        }
    }
    family_blocks(&union)
}

#[cfg(test)]
mod tests {
    use super::{family_blocks, union_blocks};
    use nalgebra::DMatrix;

    #[test]
    fn a_second_component_can_join_blocks_the_first_left_apart() {
        // Two people unrelated by the first component, sharing a home in the
        // second. The likelihood does not factorise across them, so the blocks
        // must not either.
        let relationship = DMatrix::<f64>::identity(2, 2);
        let mut household = DMatrix::<f64>::identity(2, 2);
        household[(0, 1)] = 1.0;
        household[(1, 0)] = 1.0;
        assert_eq!(family_blocks(&relationship).len(), 2);
        assert_eq!(
            union_blocks(&[relationship, household]).len(),
            1,
            "the second component joins them, so there is one block"
        );
    }

    #[test]
    fn one_component_gives_what_that_component_gives() {
        let mut matrix = DMatrix::<f64>::identity(4, 4);
        matrix[(0, 2)] = 0.5;
        matrix[(2, 0)] = 0.5;
        assert_eq!(union_blocks(&[matrix.clone()]), family_blocks(&matrix));
    }

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
