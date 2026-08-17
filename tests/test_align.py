"""Lining a relationship matrix up with per-person values, by identifier.

Asterism's numerical interface is positional, which is the one place a mistake
makes no noise: a matrix whose rows are in a different order from the response
returns a heritability, an interval and a p-value, all plausible and all for a
pedigree nobody has. These tests hold `align` to catching that.
"""

from __future__ import annotations

import asterism
import numpy as np
import pytest


def _pedigree(families: int = 60) -> tuple[list[str], list[str | None], list[str | None]]:
    ids: list[str] = []
    father: list[str | None] = []
    mother: list[str | None] = []
    for family in range(families):
        dad, mum = f"f{family}", f"m{family}"
        ids += [dad, mum]
        father += [None, None]
        mother += [None, None]
        for child in range(3):
            ids.append(f"c{family}_{child}")
            father.append(dad)
            mother.append(mum)
    return ids, father, mother


def test_alignment_by_identifier_survives_a_shuffled_table() -> None:
    """The values come back in the matrix's order, whatever order they arrived in."""
    ids, father, mother = _pedigree(4)
    relationship, order = asterism.relationship_matrix(ids, father, mother)
    rng = np.random.default_rng(11)
    shuffled = list(rng.permutation(order))
    height = {name: float(rng.normal()) for name in order}

    aligned = asterism.align(
        relationship, order, shuffled, height=[height[name] for name in shuffled]
    )
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
    relationship, order = asterism.relationship_matrix(ids, father, mother)
    n = len(order)
    rng = np.random.default_rng(3)
    factor = np.linalg.cholesky(0.6 * relationship + 0.4 * np.eye(n))
    y = factor @ rng.standard_normal(n)

    design = np.ones((n, 1))
    model = asterism.prepare(design, relationship)
    right = model.fit(y)["h2"]

    order_of = {name: i for i, name in enumerate(order)}
    shuffled_names = list(rng.permutation(order))
    # The table as somebody might really hold it: same people, different order.
    table = [y[order_of[name]] for name in shuffled_names]

    wrong = model.fit(np.ascontiguousarray(table))["h2"]
    assert abs(right - wrong) > 0.05, (
        "the shuffle changed nothing, so this test is not exercising the hazard"
    )

    aligned = asterism.align(relationship, order, shuffled_names, y=table)
    recovered = asterism.prepare(design, aligned["relationship"]).fit(aligned["y"])["h2"]
    assert recovered == pytest.approx(right, abs=1e-12)


def test_a_subset_takes_the_matrix_with_it() -> None:
    """Asking for some of the people reorders the matrix too, not just the values."""
    ids, father, mother = _pedigree(3)
    relationship, order = asterism.relationship_matrix(ids, father, mother)
    wanted = [order[4], order[0], order[7]]
    values = {name: float(i) for i, name in enumerate(order)}

    aligned = asterism.align(
        relationship, order, order, keep=wanted, y=[values[name] for name in order]
    )
    assert aligned["order"] == wanted
    assert np.allclose(aligned["y"], [values[name] for name in wanted])
    index = [order.index(name) for name in wanted]
    assert np.array_equal(aligned["relationship"], relationship[np.ix_(index, index)])


def test_what_it_refuses() -> None:
    """Each refusal is a way of getting it wrong that would otherwise be silent."""
    ids, father, mother = _pedigree(2)
    relationship, order = asterism.relationship_matrix(ids, father, mother)
    values = [float(i) for i in range(len(order))]

    with pytest.raises(ValueError, match="ALIGN_VALUE_ID_DUPLICATED"):
        asterism.align(relationship, order, [order[0], *order[1:-1], order[0]], y=values)
    with pytest.raises(ValueError, match="ALIGN_COLUMN_WRONG_LENGTH"):
        asterism.align(relationship, order, order, y=values[:-1])
    with pytest.raises(ValueError, match="ALIGN_RELATIONSHIP_IDS_WRONG_LENGTH"):
        asterism.align(relationship, order[:-1], order, y=values)
    with pytest.raises(ValueError, match="ALIGN_NOT_IN_VALUES"):
        asterism.align(relationship, order, order[:-1], keep=order, y=values[:-1])
    with pytest.raises(ValueError, match="ALIGN_RELATIONSHIP_NOT_SYMMETRIC"):
        asterism.align(np.array([[1.0, 0.5], [0.4, 1.0]]), ["a", "b"], ["a", "b"], y=[1.0, 2.0])
    with pytest.raises(ValueError, match="ALIGN_NOBODY_IN_COMMON"):
        asterism.align(relationship, order, ["nobody"], y=[1.0])


def test_missing_values_are_refused_unless_asked_for() -> None:
    """A gap is an error by default, and marked in `observed` when allowed."""
    ids, father, mother = _pedigree(2)
    relationship, order = asterism.relationship_matrix(ids, father, mother)
    values = [float(i) for i in range(len(order))]
    values[3] = float("nan")

    with pytest.raises(ValueError, match="ALIGN_VALUE_MISSING"):
        asterism.align(relationship, order, order, y=values)

    aligned = asterism.align(relationship, order, order, allow_missing=True, y=values)
    assert aligned["observed"].sum() == len(order) - 1
    assert not aligned["observed"][3]
