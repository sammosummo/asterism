"""Asterism: variance components models for quantitative genetics.

One trait with one component is two steps::

    model = asterism.prepare(x, k)
    record = model.fit(y)

Building the model validates and decomposes; the decomposition is what every
fit runs against, so many fits against one prepared model cost one preparation.
Validation is part of preparation and cannot be disabled.

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
from ._core import weighted_chi2_upper_tail as _weighted_chi2_upper_tail
from .latent_mediation import LatentMediationModel
from .models import (
    AssociationModel,
    BivariateModel,
    ComponentModel,
    DiscreteGxeModel,
    GxeModel,
    LiabilityModel,
    SpatialModel,
    VariantSetModel,
    kinship_classes,
)

__all__ = [
    "AssociationModel",
    "BivariateModel",
    "ComponentModel",
    "DiscreteGxeModel",
    "GxeModel",
    "LatentMediationModel",
    "LiabilityModel",
    "PreparedModel",
    "SpatialModel",
    "VariantSetModel",
    "__version__",
    "kinship_classes",
    "prepare",
    "relationship_matrix",
    "weighted_chi2_upper_tail",
]


def prepare(x: Any, k: Any) -> PreparedModel:
    """Validate and decompose a fixed-effect design and a relationship matrix.

    Parameters
    ----------
    x
        The fixed-effect design, one row per subject in fit order. It must
        include its own intercept column if one is wanted; nothing is added.
    k
        The relationship matrix, in the same subject order as ``x``. Twice the
        kinship is the usual choice, but nothing here requires it: the matrix
        is checked numerically, so a genomic relationship matrix is equally
        acceptable. Row alignment among ``x``, ``k``, and the response is
        positional and is the caller's responsibility.

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
    return PreparedModel(x, k)


def weighted_chi2_upper_tail(q: float, weights: Any) -> dict[str, Any]:
    """The upper tail of a weighted sum of chi-squares on one degree of freedom.

    ``P(sum_j weights_j * chisq_1 > q)``, which is what a variance-component
    score test reads. ``weights`` are the eigenvalues of the tested matrix
    after projection, so they are nonnegative.

    A gene scan reads this at around 1e-6, where an approximation calibrated at
    0.05 is worthless, so the result carries its own diagnostics. ``method``
    says which recipe answered: a single weight, or weights that are all equal,
    have exact chi-square answers and are taken directly rather than
    integrated. ``settled_to`` is the size of the last term kept as a share of
    the answer. ``trustworthy`` is false where the probability is small enough
    that cancellation has eaten the digits, because the tail is recovered as
    ``1/2 + integral`` and a small answer is a difference of two nearly equal
    numbers.
    """
    values = [float(v) for v in np.asarray(weights, dtype=np.float64).ravel()]
    return dict(_weighted_chi2_upper_tail(float(q), values))


def relationship_matrix(
    ids: list[str],
    father: list[str | None],
    mother: list[str | None],
    *,
    mz_twin: list[str | None] | None = None,
    keep: list[str] | None = None,
) -> tuple[Any, list[str]]:
    """Build the additive relationship matrix from a pedigree.

    Returns the matrix and the identifiers its rows are in. Align the response
    and design to that order before fitting; Asterism's numerical interface is
    positional.

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


# There is no one-shot `fit(x, k, y)`: preparing once and fitting many
# responses reuses the expensive decomposition and makes bootstrap fitting
# affordable.
