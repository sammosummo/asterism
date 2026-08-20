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
    mixed_bivariate_fit,
    mixed_bivariate_interval,
    mixed_bivariate_test,
    region_log_probability,
    tobit_fit,
    tobit_interval,
    tobit_test,
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
    "align",
    "kinship_classes",
    "mixed_bivariate_fit",
    "mixed_bivariate_interval",
    "mixed_bivariate_test",
    "prepare",
    "region_log_probability",
    "relationship_matrix",
    "tobit_fit",
    "tobit_interval",
    "tobit_test",
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


def align(
    relationship: Any,
    relationship_ids: list[str],
    ids: list[str],
    *,
    keep: list[str] | None = None,
    allow_missing: bool = False,
    **columns: Any,
) -> dict[str, Any]:
    """Line a relationship matrix up with per-person values, by identifier.

    **Asterism's numerical interface is positional, and that is the one place a
    mistake makes no noise.** A relationship matrix whose rows are in a
    different order from the response does not fail, or warn, or look wrong: it
    returns a heritability, an interval and a p-value, all of them plausible and
    all of them for a pedigree nobody has. This does the alignment by
    identifier, once, and refuses whatever it cannot line up.

    The cost of getting it wrong is not a small bias. On 300 people in 60
    families of five, simulated at a heritability of 0.6, the aligned fit
    returns 0.490 with a p-value of `1.4e-06`; the same data with the response
    in the wrong order returns 0.000, an interval of `[0.000, 0.081]`, and a
    p-value of 1. Shuffling does not perturb the answer, it destroys the
    signal, and then reports no heritability with complete confidence.

    Returns a dictionary with ``relationship`` and ``order``, one array per
    keyword column in that order, ``observed`` saying who has a complete set of
    values, and ``dropped`` naming anybody the matrix has and the values do not.
    Nothing is returned positionally, so no pair of outputs can be swapped.

    **Read ``dropped``.** A handful of names there is people without
    measurements. A great many is an identifier mismatch — one side writing
    ``001`` where the other writes ``1``, or a table that was filtered
    already — and the analysis would otherwise proceed, quietly, on whoever
    happened to survive it.

    Parameters
    ----------
    relationship
        A square, symmetric matrix — twice the kinship from
        :func:`relationship_matrix`, or a genomic relationship or estimated
        kinship computed elsewhere. It is subset and reordered, not rebuilt.
    relationship_ids
        Who each of its rows is, in its own order. For a matrix from
        :func:`relationship_matrix` this is the second value it returned; for
        one computed elsewhere it is that tool's own identifier file, which is
        exactly the pairing that goes wrong.
    ids
        Who each row of the ``columns`` is. Any order, and it may cover people
        the matrix does not.
    keep
        The people to analyse, in the order wanted. Omit it to use everybody
        the matrix and the values have in common, in the matrix's own order.
    allow_missing
        By default a person with no value for some column is refused. Set this
        to keep them, with ``nan`` where a value is missing and ``observed``
        marking who is complete — which is what the unbalanced models want.

    Raises
    ------
    ValueError
        With a stable code: a matrix that is not square or not symmetric, a
        duplicate identifier on either side, somebody in ``keep`` that one side
        does not have, or a missing value where ``allow_missing`` is not set.
    """
    matrix = np.asarray(relationship, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ALIGN_RELATIONSHIP_NOT_SQUARE")
    row_names = [str(value) for value in relationship_ids]
    if len(row_names) != matrix.shape[0]:
        raise ValueError("ALIGN_RELATIONSHIP_IDS_WRONG_LENGTH")
    if len(set(row_names)) != len(row_names):
        raise ValueError("ALIGN_RELATIONSHIP_ID_DUPLICATED")
    # Exact symmetry, as the models themselves demand: a matrix that disagrees
    # with its own transpose by one bit is a matrix somebody has edited.
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("ALIGN_RELATIONSHIP_NOT_SYMMETRIC")

    value_names = [str(value) for value in ids]
    if len(set(value_names)) != len(value_names):
        raise ValueError("ALIGN_VALUE_ID_DUPLICATED")
    for name, values in columns.items():
        if len(values) != len(value_names):
            raise ValueError(f"ALIGN_COLUMN_WRONG_LENGTH:{name}")

    row_of = {name: i for i, name in enumerate(row_names)}
    value_of = {name: i for i, name in enumerate(value_names)}
    if keep is None:
        wanted = [name for name in row_names if name in value_of]
        dropped = [name for name in row_names if name not in value_of]
    else:
        wanted = [str(value) for value in keep]
        dropped = [name for name in row_names if name not in set(wanted)]
        if len(set(wanted)) != len(wanted):
            raise ValueError("ALIGN_KEEP_ID_DUPLICATED")
        for name in wanted:
            if name not in row_of:
                raise ValueError(f"ALIGN_NOT_IN_RELATIONSHIP:{name}")
            if name not in value_of:
                raise ValueError(f"ALIGN_NOT_IN_VALUES:{name}")
    if not wanted:
        raise ValueError("ALIGN_NOBODY_IN_COMMON")

    rows = [row_of[name] for name in wanted]
    taken = [value_of[name] for name in wanted]
    aligned: dict[str, Any] = {
        "relationship": np.ascontiguousarray(matrix[np.ix_(rows, rows)]),
        "order": wanted,
        "dropped": dropped,
    }
    observed = np.ones(len(wanted), dtype=bool)
    for name, values in columns.items():
        column = np.asarray(values, dtype=np.float64)[taken]
        here = np.isfinite(column)
        if not allow_missing and not here.all():
            first = wanted[int(np.flatnonzero(~here)[0])]
            raise ValueError(f"ALIGN_VALUE_MISSING:{name}:{first}")
        observed &= here
        aligned[name] = np.ascontiguousarray(column)
    aligned["observed"] = observed
    return aligned
# There is no one-shot `fit(x, k, y)`: preparing once and fitting many
# responses reuses the expensive decomposition and makes bootstrap fitting
# affordable.
