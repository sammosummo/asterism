"""Asterism: variance components models for quantitative genetics.

One trait with one component is two steps::

    model = asterism.prepare(x, k)
    record = model.fit(y)

Building the model validates and decomposes; the decomposition is what every
fit runs against, so many fits against one prepared model cost one preparation.
Holding a prepared model is itself the proof that validation happened, which is
why there is no way to skip it.

The other models are objects of the same shape, built from their matrices and
then fitted::

    asterism.ComponentModel([relationship, household], x).fit(y)
    asterism.BivariateModel(k, observed, design).fit(y)
    asterism.SpatialModel([relationship], distance, design).fit(y)
    asterism.GxeModel(relationship, environment, design).fit(y)

Only `prepare` diagonalises. The rest factorise a covariance on every
evaluation, which is why the one-trait model is enormously faster and why it is
still the right thing to use when one component will do.

Asterism takes arrays that are already in memory and returns a record, also in
memory. It reads no files, knows no databases, and writes nothing. Where the
record goes afterwards is the caller's business.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import PreparedModel, __version__
from ._core import relationship as _relationship
from .models import (
    BivariateModel,
    ComponentModel,
    GxeModel,
    SpatialModel,
    kinship_classes,
)

__all__ = [
    "PreparedModel",
    "prepare",
    "relationship_matrix",
    "ComponentModel",
    "BivariateModel",
    "GxeModel",
    "SpatialModel",
    "kinship_classes",
    "__version__",
]


def prepare(
    x: Any,
    k: Any,
    *,
    subject_ids: list[str] | None = None,
    subject_order_sha256: str | None = None,
) -> PreparedModel:
    """Validate and decompose a fixed-effect design and a relationship matrix.

    Parameters
    ----------
    x
        The fixed-effect design, one row per subject in fit order. It must
        include its own intercept column if one is wanted; nothing is added.
    k
        The relationship matrix, in the same subject order as ``x``. Twice the
        kinship is the usual choice, but nothing here requires it: the matrix
        is checked as mathematics rather than for its provenance, so a genomic
        relationship matrix is equally acceptable.
    subject_ids
        Identifiers in fit order. Asterism records a hash of them and echoes it
        on every fit, which is what lets a result be tied back to one exact row
        ordering. Arrays cannot carry identifiers themselves.
    subject_order_sha256
        The same commitment supplied directly, when the hash is already known.
        Giving both this and ``subject_ids`` is an error rather than a
        preference.

    Raises
    ------
    ValueError
        With a stable code naming what was wrong: the matrix not square, not
        symmetric, not positive semi-definite, shapes disagreeing, values not
        finite, the design rank-deficient, or no residual degrees of freedom.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    k = np.ascontiguousarray(k, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("PREPARE_X_NOT_TWO_DIMENSIONAL")
    if k.ndim != 2:
        raise ValueError("PREPARE_K_NOT_TWO_DIMENSIONAL")
    return PreparedModel(
        x,
        k,
        subject_ids=subject_ids,
        subject_order_sha256=subject_order_sha256,
    )


def relationship_matrix(
    ids: list[str],
    father: list[str | None],
    mother: list[str | None],
    *,
    mz_twin: list[str | None] | None = None,
    keep: list[str] | None = None,
) -> tuple[Any, list[str]]:
    """Build the additive relationship matrix from a pedigree.

    Returns the matrix and the identifiers its rows are in. Hand those
    identifiers straight to `prepare` as ``subject_ids`` and align the response
    and design to the same order — that commitment is what catches a
    misalignment, which is the way to get a confident wrong answer here.

    Parameters
    ----------
    ids, father, mother
        Parallel lists, one entry per person, in any order — parents are sorted
        before their children. A person with no parents recorded is a founder;
        use ``None`` or an empty string for both. One parent known and the other
        not is refused rather than guessed at.
    mz_twin
        An optional group label per person. People sharing a label are treated
        as genetically identical.
    keep
        The people to give rows to, in the order wanted. Their ancestors still
        contribute to the relationships without getting rows of their own, so a
        large pedigree costs only what the analysis roster needs. Omit it to
        keep everybody.

    Raises
    ------
    ValueError
        With a stable code naming the person at fault: a duplicate identifier,
        one known parent, a parent with no record, somebody who is their own
        parent, a loop in the pedigree, or an identifier in ``keep`` that the
        pedigree does not contain.
    """
    blank_to_none = lambda value: None if value in (None, "", "0") else str(value)
    return _relationship(
        [str(value) for value in ids],
        [blank_to_none(value) for value in father],
        [blank_to_none(value) for value in mother],
        None if mz_twin is None else [blank_to_none(value) for value in mz_twin],
        keep,
    )


# There is deliberately no one-shot `fit(x, k, y)` here. `docs/adr/0003` rules
# it out for the first release: one way in, because both predecessors grew their
# interface one convenience at a time. Preparing once and fitting many times is
# also the thing that makes a bootstrap affordable, and a one-shot call throws
# that away every time it is used.
