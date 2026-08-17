//! The builder that makes an additive relationship matrix from a pedigree.
//!
//! Asterism's estimator takes a relationship matrix and does not care where it
//! came from. This builder is included because the additive matrix is the one everybody
//! needs and computing it by hand is where row-alignment mistakes live.
//!
//! What it produces is `A = 2 × kinship`, the numerator relationship matrix —
//! ones on the diagonal for a person with no inbreeding, a half between a
//! parent and a child or between full siblings, a quarter between a grandparent
//! and a grandchild.
//!
//! The pedigree arrives as parallel lists already in memory. Asterism reads no
//! files, so turning a pedigree file into these
//! lists is the caller's business and stays outside.
//!
//! Ported from Astrarium's `pedigree.rs` and `relationship.rs`, which are the
//! same recursion with a reporting layer around it that does not come across.

use std::collections::{BTreeMap, HashMap, HashSet};

use nalgebra::DMatrix;

/// One person, and their parents where both are known.
#[derive(Clone, Debug)]
pub struct Person {
    pub id: String,
    pub father: Option<String>,
    pub mother: Option<String>,
    /// People sharing a group label are treated as genetically identical.
    pub mz_twin: Option<String>,
}

/// A pedigree that failed to make sense, with a stable code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PedigreeError {
    DuplicateId(String),
    /// One parent known and the other not. Refused rather than guessed at:
    /// treating the unknown parent as an unrelated founder is a modelling
    /// choice, and a silent one would be the wrong kind.
    OneKnownParent(String),
    MissingParentRecord {
        child: String,
        parent: String,
    },
    SelfParent(String),
    Cycle(String),
    UnknownId(String),
}

impl PedigreeError {
    /// The code as it reaches the caller.
    #[must_use]
    pub fn code(&self) -> String {
        match self {
            Self::DuplicateId(id) => format!("PEDIGREE_DUPLICATE_ID: {id}"),
            Self::OneKnownParent(id) => format!("PEDIGREE_ONE_KNOWN_PARENT: {id}"),
            Self::MissingParentRecord { child, parent } => {
                format!("PEDIGREE_MISSING_PARENT_RECORD: {child} names {parent}")
            }
            Self::SelfParent(id) => format!("PEDIGREE_SELF_PARENT: {id}"),
            Self::Cycle(id) => format!("PEDIGREE_CYCLE: {id}"),
            Self::UnknownId(id) => format!("PEDIGREE_UNKNOWN_ID: {id}"),
        }
    }
}

/// Where a person has got to in the depth-first walk. A node is marked
/// `InProgress` while its ancestors are being visited, so a loop in the
/// pedigree is caught rather than looped on.
#[derive(Clone, Copy, PartialEq)]
enum Mark {
    Unvisited,
    InProgress,
    Done,
}

/// Order people so that every parent comes before every child. The relationship
/// recursion reads only rows it has already filled, so this ordering is what
/// makes one pass enough.
fn topological_order(people: &[Person]) -> Result<Vec<usize>, PedigreeError> {
    let mut position = HashMap::with_capacity(people.len());
    for (index, person) in people.iter().enumerate() {
        if position.insert(person.id.as_str(), index).is_some() {
            return Err(PedigreeError::DuplicateId(person.id.clone()));
        }
    }

    for person in people {
        match (&person.father, &person.mother) {
            (Some(father), Some(mother)) => {
                for parent in [father, mother] {
                    if parent == &person.id {
                        return Err(PedigreeError::SelfParent(person.id.clone()));
                    }
                    if !position.contains_key(parent.as_str()) {
                        return Err(PedigreeError::MissingParentRecord {
                            child: person.id.clone(),
                            parent: parent.clone(),
                        });
                    }
                }
            }
            (None, None) => {}
            _ => return Err(PedigreeError::OneKnownParent(person.id.clone())),
        }
    }

    let mut marks = vec![Mark::Unvisited; people.len()];
    let mut order = Vec::with_capacity(people.len());
    // An explicit stack rather than recursion: a deep pedigree should not be
    // able to exhaust the call stack.
    for start in 0..people.len() {
        if marks[start] != Mark::Unvisited {
            continue;
        }
        let mut stack = vec![(start, false)];
        while let Some((index, parents_done)) = stack.pop() {
            if parents_done {
                marks[index] = Mark::Done;
                order.push(index);
                continue;
            }
            match marks[index] {
                Mark::Done => continue,
                Mark::InProgress => return Err(PedigreeError::Cycle(people[index].id.clone())),
                Mark::Unvisited => {}
            }
            marks[index] = Mark::InProgress;
            stack.push((index, true));
            if let (Some(father), Some(mother)) = (&people[index].father, &people[index].mother) {
                for parent in [father, mother] {
                    let parent_index = position[parent.as_str()];
                    if marks[parent_index] == Mark::Unvisited {
                        stack.push((parent_index, false));
                    } else if marks[parent_index] == Mark::InProgress {
                        return Err(PedigreeError::Cycle(people[parent_index].id.clone()));
                    }
                }
            }
        }
    }
    Ok(order)
}

/// Everyone in `keep`, plus every ancestor any of them has. Restricting to this
/// closure before building keeps a large pedigree tractable: relationships
/// among the people being analysed depend on their ancestors and on nobody
/// else.
fn ancestor_closure(people: &[Person], keep: &[String]) -> Result<HashSet<usize>, PedigreeError> {
    let position: HashMap<&str, usize> = people
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id.as_str(), i))
        .collect();
    let mut wanted = HashSet::new();
    let mut stack = Vec::new();
    for id in keep {
        let index = *position
            .get(id.as_str())
            .ok_or_else(|| PedigreeError::UnknownId(id.clone()))?;
        if wanted.insert(index) {
            stack.push(index);
        }
    }
    while let Some(index) = stack.pop() {
        if let (Some(father), Some(mother)) = (&people[index].father, &people[index].mother) {
            for parent in [father, mother] {
                let parent_index = position[parent.as_str()];
                if wanted.insert(parent_index) {
                    stack.push(parent_index);
                }
            }
        }
    }
    Ok(wanted)
}

/// Build the additive relationship matrix.
///
/// With `keep` empty the matrix covers everybody, in an order with parents
/// before children. With `keep` given, the matrix covers exactly those people
/// **in the order they were asked for**, so that a response and a design
/// already in that order line up without further work. Ancestors outside `keep`
/// still contribute to the relationships; they simply do not get a row.
///
/// # Errors
///
/// Returns a stable code for a duplicate identifier, a person with one known
/// parent, a parent with no record of their own, somebody who is their own
/// parent, a loop in the pedigree, or an identifier in `keep` that the pedigree
/// does not contain.
pub fn relationship_matrix(
    people: &[Person],
    keep: &[String],
) -> Result<(DMatrix<f64>, Vec<String>), PedigreeError> {
    let order = topological_order(people)?;
    let included: Option<HashSet<usize>> = if keep.is_empty() {
        None
    } else {
        Some(ancestor_closure(people, keep)?)
    };
    let build: Vec<usize> = order
        .into_iter()
        .filter(|index| included.as_ref().is_none_or(|set| set.contains(index)))
        .collect();

    let dimension = build.len();
    let mut a = DMatrix::<f64>::zeros(dimension, dimension);
    let mut row_of: HashMap<&str, usize> = HashMap::with_capacity(dimension);
    let mut mz_representative: BTreeMap<&str, usize> = BTreeMap::new();

    for (row, &index) in build.iter().enumerate() {
        let person = &people[index];

        if let Some(&representative) = person
            .mz_twin
            .as_deref()
            .and_then(|group| mz_representative.get(group))
        {
            // Genetically the same person, so the same relationships to
            // everyone already placed.
            for column in 0..row {
                let value = a[(representative, column)];
                a[(row, column)] = value;
                a[(column, row)] = value;
            }
            a[(row, row)] = a[(representative, representative)];
        } else if let (Some(father), Some(mother)) = (&person.father, &person.mother) {
            let father_row = row_of[father.as_str()];
            let mother_row = row_of[mother.as_str()];
            for column in 0..row {
                // Half from each parent, which is the whole of the recursion.
                let value = 0.5 * (a[(father_row, column)] + a[(mother_row, column)]);
                a[(row, column)] = value;
                a[(column, row)] = value;
            }
            // One plus the inbreeding coefficient, which is the kinship between
            // the parents.
            a[(row, row)] = 1.0 + 0.5 * a[(father_row, mother_row)];
        } else {
            a[(row, row)] = 1.0;
        }

        if let Some(group) = person.mz_twin.as_deref() {
            mz_representative.entry(group).or_insert(row);
        }
        row_of.insert(person.id.as_str(), row);
    }

    if keep.is_empty() {
        let ids = build.iter().map(|&i| people[i].id.clone()).collect();
        return Ok((a, ids));
    }

    // Cut down to exactly the people asked for, in exactly the order asked for.
    let wanted: Vec<usize> = keep.iter().map(|id| row_of[id.as_str()]).collect();
    let size = wanted.len();
    let mut out = DMatrix::<f64>::zeros(size, size);
    for (i, &from) in wanted.iter().enumerate() {
        for (j, &to) in wanted.iter().enumerate() {
            out[(i, j)] = a[(from, to)];
        }
    }
    Ok((out, keep.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::{PedigreeError, Person, relationship_matrix};

    fn person(id: &str, father: Option<&str>, mother: Option<&str>) -> Person {
        Person {
            id: id.to_owned(),
            father: father.map(str::to_owned),
            mother: mother.map(str::to_owned),
            mz_twin: None,
        }
    }

    /// Grandparents, two children, two spouses married in, and grandchildren
    /// who are first cousins.
    fn three_generations() -> Vec<Person> {
        vec![
            person("gm", None, None),
            person("gf", None, None),
            person("a", Some("gf"), Some("gm")),
            person("b", Some("gf"), Some("gm")),
            person("a_spouse", None, None),
            person("b_spouse", None, None),
            person("a_child", Some("a_spouse"), Some("a")),
            person("b_child", Some("b_spouse"), Some("b")),
        ]
    }

    fn value(people: &[Person], first: &str, second: &str) -> f64 {
        let (a, ids) = relationship_matrix(people, &[]).expect("valid pedigree");
        let i = ids.iter().position(|id| id == first).expect("first");
        let j = ids.iter().position(|id| id == second).expect("second");
        a[(i, j)]
    }

    #[test]
    fn the_textbook_relationships_come_out() {
        let people = three_generations();
        let close = |got: f64, want: f64| assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        close(value(&people, "gm", "gm"), 1.0);
        close(value(&people, "gm", "gf"), 0.0);
        close(value(&people, "gm", "a"), 0.5);
        close(value(&people, "a", "b"), 0.5);
        close(value(&people, "a", "a_spouse"), 0.0);
        close(value(&people, "gm", "a_child"), 0.25);
        close(value(&people, "b", "a_child"), 0.25);
        close(value(&people, "a_child", "b_child"), 0.125);
    }

    #[test]
    fn inbreeding_raises_the_diagonal_above_one() {
        // Half siblings share a father; their child is inbred. The parents'
        // kinship is 1/8, so the child's diagonal is 1 + 1/8.
        let people = vec![
            person("father", None, None),
            person("mother_one", None, None),
            person("mother_two", None, None),
            person("half_one", Some("father"), Some("mother_one")),
            person("half_two", Some("father"), Some("mother_two")),
            person("child", Some("half_two"), Some("half_one")),
        ];
        assert!((value(&people, "half_one", "half_two") - 0.25).abs() < 1e-12);
        assert!((value(&people, "child", "child") - 1.125).abs() < 1e-12);
    }

    #[test]
    fn identical_twins_share_every_relationship() {
        let mut people = three_generations();
        people[2].mz_twin = Some("t".to_owned());
        people.push(Person {
            id: "a_twin".to_owned(),
            father: Some("gf".to_owned()),
            mother: Some("gm".to_owned()),
            mz_twin: Some("t".to_owned()),
        });
        assert!((value(&people, "a", "a_twin") - 1.0).abs() < 1e-12);
        assert!((value(&people, "a_twin", "a_child") - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_parent_ordered_after_a_child_is_still_handled() {
        // The input order is deliberately wrong; the builder sorts it out.
        let people = vec![
            person("child", Some("father"), Some("mother")),
            person("father", None, None),
            person("mother", None, None),
        ];
        assert!((value(&people, "child", "father") - 0.5).abs() < 1e-12);
    }

    #[test]
    fn keeping_a_subset_uses_the_ancestors_and_returns_the_asked_for_order() {
        let people = three_generations();
        // Two cousins, asked for in this order, with no grandparent kept.
        let keep = vec!["b_child".to_owned(), "a_child".to_owned()];
        let (a, ids) = relationship_matrix(&people, &keep).expect("valid");
        assert_eq!(ids, keep);
        assert_eq!(a.nrows(), 2);
        // Their relationship still runs through grandparents who have no row.
        assert!((a[(0, 1)] - 0.125).abs() < 1e-12);
        assert!((a[(0, 0)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_pedigree_that_makes_no_sense_is_refused() {
        let refuse = |people: Vec<Person>| relationship_matrix(&people, &[]).err().unwrap();

        assert_eq!(
            refuse(vec![person("a", None, None), person("a", None, None)]),
            PedigreeError::DuplicateId("a".to_owned())
        );
        assert_eq!(
            refuse(vec![person("f", None, None), person("c", Some("f"), None)]),
            PedigreeError::OneKnownParent("c".to_owned())
        );
        assert_eq!(
            refuse(vec![
                person("c", Some("f"), Some("m")),
                person("m", None, None)
            ]),
            PedigreeError::MissingParentRecord {
                child: "c".to_owned(),
                parent: "f".to_owned(),
            }
        );
        assert_eq!(
            refuse(vec![person("a", Some("a"), Some("a"))]),
            PedigreeError::SelfParent("a".to_owned())
        );
        assert_eq!(
            refuse(vec![
                person("m", None, None),
                person("a", Some("b"), Some("m")),
                person("b", Some("a"), Some("m")),
            ]),
            PedigreeError::Cycle("a".to_owned())
        );
        assert_eq!(
            relationship_matrix(&[person("a", None, None)], &["b".to_owned()])
                .err()
                .unwrap(),
            PedigreeError::UnknownId("b".to_owned())
        );
    }
}

// PyO3 extracts each argument from a Python object, so a `#[pyfunction]` takes
// them by value whether or not the body consumes them. The lint cannot be
// satisfied here without breaking the macro.
#[allow(clippy::needless_pass_by_value)]
#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::{Person, relationship_matrix};

    /// Build the additive relationship matrix from a pedigree held in memory.
    ///
    /// Returns the matrix and the identifiers its rows are in so the caller can
    /// align the response and design before passing the numerical arrays to
    /// `prepare`.
    #[pyfunction]
    #[pyo3(signature = (ids, father, mother, mz_twin=None, keep=None))]
    pub fn relationship(
        py: Python<'_>,
        ids: Vec<String>,
        father: Vec<Option<String>>,
        mother: Vec<Option<String>>,
        mz_twin: Option<Vec<Option<String>>>,
        keep: Option<Vec<String>>,
    ) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
        if father.len() != ids.len() || mother.len() != ids.len() {
            return Err(PyValueError::new_err("PEDIGREE_LENGTH_MISMATCH"));
        }
        if let Some(groups) = &mz_twin
            && groups.len() != ids.len()
        {
            return Err(PyValueError::new_err("PEDIGREE_LENGTH_MISMATCH"));
        }
        let people: Vec<Person> = (0..ids.len())
            .map(|i| Person {
                id: ids[i].clone(),
                father: father[i].clone(),
                mother: mother[i].clone(),
                mz_twin: mz_twin.as_ref().and_then(|g| g[i].clone()),
            })
            .collect();
        let keep = keep.unwrap_or_default();
        let (matrix, order) = relationship_matrix(&people, &keep)
            .map_err(|error| PyValueError::new_err(error.code()))?;
        let rows = matrix.nrows();
        let values: Vec<f64> = (0..rows)
            .flat_map(|i| (0..rows).map(move |j| (i, j)))
            .map(|(i, j)| matrix[(i, j)])
            .collect();
        let array = numpy::ndarray::Array2::from_shape_vec((rows, rows), values)
            .map_err(|_| PyValueError::new_err("PEDIGREE_MATRIX_SHAPE"))?;
        Ok((array.into_pyarray(py).unbind(), order))
    }
}

#[cfg(feature = "python")]
pub use python::relationship;
