//! Splitting the relationship matrix by the kind of parent–offspring tie.
//!
//! The class-weighted kinship model asks whether a mother and her son resemble
//! one another by the same amount as a father and his daughter, and so on for
//! the four direct parent–offspring classes. It multiplies **only** those cells
//! of the ordinary relationship matrix by class-specific weights and leaves
//! every other relationship alone.
//!
//! **That is linear in the weights, which is why no new likelihood is needed.**
//! Writing the weighted matrix as
//!
//! ```text
//! A(w) = A_rest + sum over classes of w_c * A_c
//! ```
//!
//! and folding the additive variance through gives
//!
//! ```text
//! V = s_A * A_rest + sum over classes of (s_A * w_c) * A_c + s_e * I
//! ```
//!
//! which is six ordinary variance components. `ComponentModel` fits it as it
//! stands, and this module's only job is to hand it the right matrices.
//!
//! The component coefficients are the fitted quantities. Ratios such as
//! `w_c = v_c / s_A` are strongly biased when the denominator is uncertain.
//! The split class matrices also have zero diagonals, so their raw coefficient
//! proportions are not shares of phenotypic variance. Equality tests and class
//! contrasts answer the scientific questions without either misinterpretation.

use std::collections::HashMap;

use nalgebra::DMatrix;

use crate::relationship::{PedigreeError, Person, relationship_matrix};

/// Which parent, and which child.
///
/// The four direct biological classes, named as the project names them.
pub const CLASS_NAMES: [&str; 4] = [
    "mother_son",
    "mother_daughter",
    "father_son",
    "father_daughter",
];

/// The relationship matrix split into the four parent–offspring classes and
/// everything else.
pub struct KinshipClasses {
    /// Every relationship that is not a direct parent–offspring tie, including
    /// the diagonal.
    pub rest: DMatrix<f64>,
    /// One matrix per class, in the order of `CLASS_NAMES`, carrying the
    /// relationship values at that class's cells and nought elsewhere.
    pub classes: [DMatrix<f64>; 4],
    /// The people, in the order the matrices are indexed.
    pub order: Vec<String>,
    /// How many pairs fell in each class. A class with few pairs is a class
    /// whose weight is poorly determined, and that is worth seeing before the
    /// answer is.
    pub pairs: [usize; 4],
}

impl KinshipClasses {
    /// Build them from a pedigree and a recorded sex for each person.
    ///
    /// `sex` maps a person to `"1"` for male and `"2"` for female, the
    /// pedigree's own coding. A person whose sex is unrecorded takes part in
    /// the relationship matrix as usual but contributes to no class, because
    /// there is no class to put them in.
    ///
    /// # Errors
    ///
    /// Returns a pedigree error where the pedigree does not make sense.
    pub fn build(
        people: &[Person],
        sex: &HashMap<String, String>,
        keep: &[String],
    ) -> Result<Self, PedigreeError> {
        let (matrix, order) = relationship_matrix(people, keep)?;
        let position: HashMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(index, id)| (id.as_str(), index))
            .collect();
        let by_id: HashMap<&str, &Person> = people.iter().map(|p| (p.id.as_str(), p)).collect();

        let n = order.len();
        let mut rest = matrix.clone();
        let mut classes = [
            DMatrix::<f64>::zeros(n, n),
            DMatrix::<f64>::zeros(n, n),
            DMatrix::<f64>::zeros(n, n),
            DMatrix::<f64>::zeros(n, n),
        ];
        let mut pairs = [0usize; 4];

        for (child_id, &child) in &position {
            let Some(record) = by_id.get(child_id) else {
                continue;
            };
            let Some(child_sex) = sex.get(*child_id) else {
                continue;
            };
            let child_is_son = match child_sex.as_str() {
                "1" => true,
                "2" => false,
                _ => continue,
            };
            for (parent_id, parent_is_mother) in [
                (record.mother.as_deref(), true),
                (record.father.as_deref(), false),
            ] {
                let Some(parent_id) = parent_id else { continue };
                let Some(&parent) = position.get(parent_id) else {
                    continue;
                };
                // The class is decided by the pedigree's link, not by the
                // matrix value: a parent and child who are also related some
                // other way still belong to their parent-offspring class, and
                // the cell carries whatever the relationship matrix says.
                let index = match (parent_is_mother, child_is_son) {
                    (true, true) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                };
                let value = matrix[(parent, child)];
                classes[index][(parent, child)] = value;
                classes[index][(child, parent)] = value;
                rest[(parent, child)] = 0.0;
                rest[(child, parent)] = 0.0;
                pairs[index] += 1;
            }
        }

        Ok(Self {
            rest,
            classes,
            order,
            pairs,
        })
    }

    /// The matrices in the order `ComponentModel` wants them: everything else
    /// first, then the four classes. The residual is added by the model.
    #[must_use]
    pub fn matrices(&self) -> Vec<DMatrix<f64>> {
        let mut out = Vec::with_capacity(5);
        out.push(self.rest.clone());
        out.extend(self.classes.iter().cloned());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{CLASS_NAMES, KinshipClasses};
    use crate::relationship::Person;
    use std::collections::HashMap;

    fn person(id: &str, father: Option<&str>, mother: Option<&str>) -> Person {
        Person {
            id: id.to_owned(),
            father: father.map(str::to_owned),
            mother: mother.map(str::to_owned),
            mz_twin: None,
        }
    }

    /// One nuclear family: a mother, a father, a son and a daughter.
    fn family() -> (Vec<Person>, HashMap<String, String>) {
        let people = vec![
            person("mum", None, None),
            person("dad", None, None),
            person("son", Some("dad"), Some("mum")),
            person("daughter", Some("dad"), Some("mum")),
        ];
        let sex: HashMap<String, String> =
            [("mum", "2"), ("dad", "1"), ("son", "1"), ("daughter", "2")]
                .iter()
                .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
                .collect();
        (people, sex)
    }

    /// **The split must lose nothing.** Everything else plus the four classes
    /// has to reconstruct the relationship matrix exactly. A cell counted twice
    /// would inflate a relationship and a cell dropped would delete one, and
    /// neither would announce itself -- the fit would simply answer a different
    /// question.
    #[test]
    fn the_parts_add_back_up_to_the_whole() {
        let (people, sex) = family();
        let split = KinshipClasses::build(&people, &sex, &[]).expect("valid");
        let (whole, order) = crate::relationship::relationship_matrix(&people, &[]).expect("valid");
        assert_eq!(order, split.order);
        let mut total = split.rest.clone();
        for class in &split.classes {
            total += class;
        }
        for i in 0..order.len() {
            for j in 0..order.len() {
                assert!(
                    (total[(i, j)] - whole[(i, j)]).abs() < 1e-12,
                    "cell ({i}, {j}) came to {} against {}",
                    total[(i, j)],
                    whole[(i, j)]
                );
            }
        }
    }

    /// Each tie lands in the class its parent and child actually make, and the
    /// four classes do not overlap.
    #[test]
    fn each_tie_lands_in_exactly_one_class() {
        let (people, sex) = family();
        let split = KinshipClasses::build(&people, &sex, &[]).expect("valid");
        assert_eq!(
            split.pairs,
            [1, 1, 1, 1],
            "one of each class was expected, got {:?} for {CLASS_NAMES:?}",
            split.pairs
        );
        for i in 0..split.order.len() {
            for j in 0..split.order.len() {
                let occupied = split.classes.iter().filter(|c| c[(i, j)] != 0.0).count();
                assert!(occupied <= 1, "cell ({i}, {j}) is in {occupied} classes");
            }
        }
    }

    /// The siblings are related and are not a parent–offspring tie, so their
    /// cell belongs to the remainder and to no class. A split that swept up
    /// every relative would be answering a different question entirely.
    #[test]
    fn siblings_stay_out_of_the_classes() {
        let (people, sex) = family();
        let split = KinshipClasses::build(&people, &sex, &[]).expect("valid");
        let at = |id: &str| split.order.iter().position(|x| x == id).expect("present");
        let (son, daughter) = (at("son"), at("daughter"));
        assert!(
            split.rest[(son, daughter)] > 0.0,
            "the siblings lost their relationship"
        );
        for class in &split.classes {
            assert_eq!(class[(son, daughter)], 0.0);
        }
        // And the parents, who are unrelated, stay unrelated.
        let (mum, dad) = (at("mum"), at("dad"));
        assert_eq!(split.rest[(mum, dad)], 0.0);
    }

    /// A person with no recorded sex cannot be put in a class, and is left in
    /// the remainder rather than guessed at.
    #[test]
    fn a_child_with_no_recorded_sex_joins_no_class() {
        let (people, mut sex) = family();
        sex.remove("son");
        let split = KinshipClasses::build(&people, &sex, &[]).expect("valid");
        assert_eq!(
            split.pairs,
            [0, 1, 0, 1],
            "the son's two ties should have fallen out of the classes"
        );
        let at = |id: &str| split.order.iter().position(|x| x == id).expect("present");
        assert!(split.rest[(at("mum"), at("son"))] > 0.0);
    }
}

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use std::collections::HashMap;

    use super::{CLASS_NAMES, KinshipClasses};
    use crate::relationship::Person;

    /// Split the relationship matrix by the kind of parent–offspring tie.
    ///
    /// Returns the five matrices the model wants — everything else, then
    /// mother–son, mother–daughter, father–son and father–daughter — with the
    /// identifiers their rows are in, the class names, and how many pairs fell
    /// in each class.
    ///
    /// **The pair counts are worth reading before the answer is.** A class's
    /// weight is determined by the pairs in it, and these classes are rarely
    /// balanced: in GOBS there are 279 mother–daughter pairs and 102
    /// father–son.
    #[pyfunction]
    #[pyo3(signature = (ids, father, mother, sex, keep=None))]
    #[allow(clippy::type_complexity)]
    pub fn kinship_classes(
        py: Python<'_>,
        ids: Vec<String>,
        father: Vec<Option<String>>,
        mother: Vec<Option<String>>,
        sex: Vec<Option<String>>,
        keep: Option<Vec<String>>,
    ) -> PyResult<(Vec<Py<PyArray2<f64>>>, Vec<String>, Vec<String>, Vec<usize>)> {
        if father.len() != ids.len() || mother.len() != ids.len() || sex.len() != ids.len() {
            return Err(PyValueError::new_err("PEDIGREE_LENGTH_MISMATCH"));
        }
        let people: Vec<Person> = (0..ids.len())
            .map(|i| Person {
                id: ids[i].clone(),
                father: father[i].clone(),
                mother: mother[i].clone(),
                mz_twin: None,
            })
            .collect();
        let coding: HashMap<String, String> = (0..ids.len())
            .filter_map(|i| sex[i].clone().map(|s| (ids[i].clone(), s)))
            .collect();
        let split = KinshipClasses::build(&people, &coding, &keep.unwrap_or_default())
            .map_err(|error| PyValueError::new_err(error.code()))?;

        let matrices = split
            .matrices()
            .iter()
            .map(|m| {
                let n = m.nrows();
                let rows: Vec<Vec<f64>> = (0..n)
                    .map(|i| (0..n).map(|j| m[(i, j)]).collect())
                    .collect();
                let flat: Vec<f64> = rows.into_iter().flatten().collect();
                numpy::ndarray::Array2::from_shape_vec((n, n), flat)
                    .expect("square by construction")
                    .into_pyarray(py)
                    .unbind()
            })
            .collect();
        Ok((
            matrices,
            split.order,
            CLASS_NAMES.iter().map(|s| (*s).to_owned()).collect(),
            split.pairs.to_vec(),
        ))
    }
}

#[cfg(feature = "python")]
pub use python::kinship_classes;
