"""Lining a relationship matrix up with per-person values, by identifier.

Asterism's numerical interface is positional, which is the one place a mistake
makes no noise: a matrix whose rows are in a different order from the response
returns a heritability, an interval and a p-value, all plausible and all for a
pedigree nobody has. These tests hold `align` to catching that.
"""

from __future__ import annotations

import asterism
import numpy as np
import numpy.typing as npt
import pytest


# asterism-style: allow private-helper -- shared pedigree fixture for alignment tests
def _pedigree(
    families: int = 60,
) -> tuple[list[str], list[str | None], list[str | None]]:
    """Build unrelated nuclear families with three children apiece."""
    ids: list[str] = []
    """Initialised identifiers in pedigree row order."""

    father: list[str | None] = []
    """Initialised paternal identifiers parallel to the pedigree rows."""

    mother: list[str | None] = []
    """Initialised maternal identifiers parallel to the pedigree rows."""

    for family in range(families):
        dad, mum = f"f{family}", f"m{family}"
        """Named the founders of the current nuclear family."""

        ids += [dad, mum]
        """Added both founders before their children."""

        father += [None, None]
        """Marked the founders' fathers unknown."""

        mother += [None, None]
        """Marked the founders' mothers unknown."""

        for child in range(3):
            ids.append(f"c{family}_{child}")
            father.append(dad)
            mother.append(mum)
        """Added three full siblings to the current family."""
    """Constructed the requested collection of unrelated nuclear families."""
    return ids, father, mother


def test_alignment_by_identifier_survives_a_shuffled_table() -> None:
    """The values come back in the matrix's order, whatever order they arrived in."""
    ids, father, mother = _pedigree(4)
    """Built a small pedigree with stable identifiers."""

    relationship, order = asterism.relationship_matrix(ids, father, mother)
    """Constructed its relationship matrix and canonical subject order."""

    rng: np.random.Generator = np.random.default_rng(11)
    """Created a deterministic generator for values and row permutation."""

    shuffled: list[str] = list(rng.permutation(order))
    """Permuted the per-person table independently of the matrix."""

    height: dict[str, float] = {name: float(rng.normal()) for name in order}
    """Assigned one deterministic height value to every subject."""

    aligned: dict[str, object] = asterism.align(
        relationship, order, shuffled, height=[height[name] for name in shuffled]
    )
    """Realigned the shuffled height column to the matrix's canonical order."""

    assert aligned["order"] == order
    assert aligned["dropped"] == []
    assert np.allclose(aligned["height"], [height[name] for name in order])
    assert np.array_equal(aligned["relationship"], relationship)


def test_a_misaligned_fit_really_does_give_a_different_answer() -> None:
    """The hazard is real, and `align` removes it.

    Fitting the same trait against the same pedigree with the response shuffled
    must not agree with fitting it aligned — otherwise there would be nothing
    here to protect against.
    """
    ids, father, mother = _pedigree()
    """Built enough nuclear families for a stable heritability contrast."""

    relationship, order = asterism.relationship_matrix(ids, father, mother)
    """Constructed their relationship matrix and canonical subject order."""

    n: int = len(order)
    """Counted the people contributing to the simulated response."""

    rng: np.random.Generator = np.random.default_rng(3)
    """Created a deterministic generator for responses and permutation."""

    factor: npt.NDArray[np.float64] = np.linalg.cholesky(
        0.6 * relationship + 0.4 * np.eye(n)
    )
    """Factorised a covariance with substantial additive heritability."""

    y: npt.NDArray[np.float64] = factor @ rng.standard_normal(n)
    """Drew one response carrying the pedigree covariance."""

    design: npt.NDArray[np.float64] = np.ones((n, 1))
    """Built the intercept-only fixed-effect design."""

    model: asterism.PreparedModel = asterism.prepare(design, relationship)
    """Prepared the correctly ordered one-trait model."""

    right: float = model.fit(y)["h2"]
    """Fitted the response against its matching relationship rows."""

    order_of: dict[str, int] = {name: i for i, name in enumerate(order)}
    """Indexed each subject's position in the canonical response."""

    shuffled_names: list[str] = list(rng.permutation(order))
    """Created a different row order for a realistic person table."""

    # The table as somebody might really hold it: same people, different order.
    table: list[float] = [float(y[order_of[name]]) for name in shuffled_names]
    """Reordered the same response values into the table's subject order."""

    wrong: float = model.fit(np.ascontiguousarray(table))["h2"]
    """Fitted the shuffled response without moving the relationship matrix."""

    assert abs(right - wrong) > 0.05, (
        "the shuffle changed nothing, so this test is not exercising the hazard"
    )

    aligned: dict[str, object] = asterism.align(
        relationship, order, shuffled_names, y=table
    )
    """Moved both the table values and matrix into one explicit subject order."""

    recovered: float = asterism.prepare(design, aligned["relationship"]).fit(
        aligned["y"]
    )["h2"]
    """Refitted the now-aligned response and relationship matrix."""

    assert recovered == pytest.approx(right, abs=1e-12)


def test_a_subset_takes_the_matrix_with_it() -> None:
    """Asking for some of the people reorders the matrix too, not just the values."""
    ids, father, mother = _pedigree(3)
    """Built a small pedigree whose rows can be subset and reordered."""

    relationship, order = asterism.relationship_matrix(ids, father, mother)
    """Constructed its relationship matrix and canonical subject order."""

    wanted: list[str] = [order[4], order[0], order[7]]
    """Selected a deliberately non-canonical subject subset."""

    values: dict[str, float] = {name: float(index) for index, name in enumerate(order)}
    """Assigned a row-identifying value to every canonical subject."""

    aligned: dict[str, object] = asterism.align(
        relationship, order, order, keep=wanted, y=[values[name] for name in order]
    )
    """Subset and reordered the values and relationship matrix together."""

    assert aligned["order"] == wanted
    assert np.allclose(aligned["y"], [values[name] for name in wanted])
    index: list[int] = [order.index(name) for name in wanted]
    """Located the expected matrix rows for the requested subset."""

    assert np.array_equal(aligned["relationship"], relationship[np.ix_(index, index)])


def test_what_it_refuses() -> None:
    """Each refusal is a way of getting it wrong that would otherwise be silent."""
    ids, father, mother = _pedigree(2)
    """Built a compact pedigree for invalid-alignment cases."""

    relationship, order = asterism.relationship_matrix(ids, father, mother)
    """Constructed its relationship matrix and canonical subject order."""

    values: list[float] = [float(index) for index in range(len(order))]
    """Built one complete value column in canonical order."""

    with pytest.raises(ValueError, match="ALIGN_VALUE_ID_DUPLICATED"):
        asterism.align(
            relationship, order, [order[0], *order[1:-1], order[0]], y=values
        )
    with pytest.raises(ValueError, match="ALIGN_COLUMN_WRONG_LENGTH"):
        asterism.align(relationship, order, order, y=values[:-1])
    with pytest.raises(ValueError, match="ALIGN_RELATIONSHIP_IDS_WRONG_LENGTH"):
        asterism.align(relationship, order[:-1], order, y=values)
    with pytest.raises(ValueError, match="ALIGN_NOT_IN_VALUES"):
        asterism.align(relationship, order, order[:-1], keep=order, y=values[:-1])
    with pytest.raises(ValueError, match="ALIGN_RELATIONSHIP_NOT_SYMMETRIC"):
        asterism.align(
            np.array([[1.0, 0.5], [0.4, 1.0]]),
            ["a", "b"],
            ["a", "b"],
            y=[1.0, 2.0],
        )
    with pytest.raises(ValueError, match="ALIGN_NOBODY_IN_COMMON"):
        asterism.align(relationship, order, ["nobody"], y=[1.0])


def test_missing_values_are_refused_unless_asked_for() -> None:
    """A gap is an error by default, and marked in `observed` when allowed."""
    ids, father, mother = _pedigree(2)
    """Built a compact pedigree for missing-value behaviour."""

    relationship, order = asterism.relationship_matrix(ids, father, mother)
    """Constructed its relationship matrix and canonical subject order."""

    values: list[float] = [float(index) for index in range(len(order))]
    """Built one initially complete value column."""

    values[3] = float("nan")
    """Inserted one explicit missing value into the column."""

    with pytest.raises(ValueError, match="ALIGN_VALUE_MISSING"):
        asterism.align(relationship, order, order, y=values)

    aligned: dict[str, object] = asterism.align(
        relationship, order, order, allow_missing=True, y=values
    )
    """Aligned the incomplete column while retaining its observation mask."""

    assert aligned["observed"].sum() == len(order) - 1
    assert not aligned["observed"][3]
