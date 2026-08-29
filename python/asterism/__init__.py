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

from collections.abc import Callable
from typing import Any

import numpy as np

from ._core import PreparedModel, __version__
from ._core import grouping as _grouping
from ._core import relationship as _relationship
from ._core import weighted_chi2_upper_tail as _weighted_chi2_upper_tail
from .analysis import (
    build_identity,
    release_manifest,
    run_analysis,
    subject_order_commitment,
)
from .latent_mediation import LatentMediationModel
from .models import (
    AssociationModel,
    AutoregressiveModel,
    BivariateModel,
    CensoredComponentModel,
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

__all__: list[str] = [
    "AssociationModel",
    "AutoregressiveModel",
    "BivariateModel",
    "CensoredComponentModel",
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
    "build_identity",
    "grouping_matrix",
    "kinship_classes",
    "mixed_bivariate_fit",
    "mixed_bivariate_interval",
    "mixed_bivariate_test",
    "prepare",
    "region_log_probability",
    "relationship_matrix",
    "release_manifest",
    "run_analysis",
    "subject_order_commitment",
    "tobit_fit",
    "tobit_interval",
    "tobit_test",
    "weighted_chi2_upper_tail",
]
"""Declared every supported, unsupported and infrastructure-level public object."""


def prepare(
    x: Any,
    k: Any,
    *,
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
        is checked numerically, so a genomic relationship matrix is equally
        acceptable. Row alignment among ``x``, ``k``, and the response is
        positional and is the caller's responsibility.
    subject_order_sha256
        Lowercase SHA-256 from :func:`subject_order_commitment` for the exact
        row order. The prepared model echoes it on every fit record without
        retaining participant identifiers.

    Raises
    ------
    ValueError
        With a stable code naming what was wrong: the matrix not square, not
        symmetric, not positive semi-definite, shapes disagreeing, values not
        finite, the design rank-deficient, or no residual degrees of freedom.
    """
    x: np.ndarray = np.ascontiguousarray(x, dtype=np.float64)
    """Copied the design into the exact numeric layout consumed by Rust."""
    k: np.ndarray = np.ascontiguousarray(k, dtype=np.float64)
    """Copied the relationship matrix into the exact numeric layout consumed by Rust."""
    if x.ndim != 2:
        raise ValueError("PREPARE_X_NOT_TWO_DIMENSIONAL")
    if k.ndim != 2:
        raise ValueError("PREPARE_K_NOT_TWO_DIMENSIONAL")
    return PreparedModel(x, k, subject_order_sha256)


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
    values: list[float] = [
        float(v) for v in np.asarray(weights, dtype=np.float64).ravel()
    ]
    """Flattened the caller's nonnegative mixture weights for the Rust routine."""
    return dict(_weighted_chi2_upper_tail(float(q), values))


def grouping_matrix(groups: list[str | None]) -> Any:
    """Build a grouping matrix: one where two rows share a group.

    **One builder, several components.** Pass household identifiers and it is a
    household matrix. Pass the person each row belongs to and it is the
    person-level matrix — the listener kernel, where a listener contributes two
    ears — because sharing a person is the same relation as sharing a home.
    Pass a testing session and it is a session effect. The matrix does not know
    which it is, and neither does the model: what it means is what you grouped
    by.

    Parameters
    ----------
    groups
        One entry per row, in the row order the fit will use. ``None`` or an
        empty string is a group nobody knows: that row shares with nobody and
        keeps its diagonal. **The compiled builder decides that, not this
        wrapper.** Both used to, and they disagreed — Rust took an empty string
        as a real shared group while this mapped it away — which nothing could
        see, because only one side of the pair was ever tested. Mapping in one
        place is what stops that.

        A falsy value that is not a string is not treated as missing. ``"0"`` is
        a group name like any other. It is not the absence of a group — the person does
        have a home, and what is missing is which — so their effect cannot be
        told apart from their residual and they inform the component only by
        not sharing.

    Returns
    -------
    A square matrix, rows in the order given. One on the diagonal throughout, so
    a coefficient fitted against it is a proportion of the total variance.

    Notes
    -----
    It joins only rows that share a group, so it cannot enlarge a likelihood
    block beyond the groups that straddle two families. A kernel over distances
    would join every pair arithmetically, which is a different thing.
    """
    return _grouping(list(groups))


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
    blank_to_none: Callable[[str | None], str | None] = lambda value: (
        None if value in (None, "", "0") else str(value)
    )
    """Normalised the pedigree's conventional missing-parent spellings."""
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

    Asterism's numerical interface is positional, and that is the one place a
    mistake makes no noise. A relationship matrix whose rows are in a
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

    Read ``dropped``. A handful of names there is people without
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
    matrix: np.ndarray = np.asarray(relationship, dtype=np.float64)
    """Read the relationship matrix without changing its row order."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ALIGN_RELATIONSHIP_NOT_SQUARE")
    row_names: list[str] = [str(value) for value in relationship_ids]
    """Normalised the identifiers attached to relationship-matrix rows."""
    if len(row_names) != matrix.shape[0]:
        raise ValueError("ALIGN_RELATIONSHIP_IDS_WRONG_LENGTH")
    if len(set(row_names)) != len(row_names):
        raise ValueError("ALIGN_RELATIONSHIP_ID_DUPLICATED")
    # Exact symmetry, as the models themselves demand: a matrix that disagrees
    # with its own transpose by one bit is a matrix somebody has edited.
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("ALIGN_RELATIONSHIP_NOT_SYMMETRIC")

    value_names: list[str] = [str(value) for value in ids]
    """Normalised identifiers attached to the caller's value rows."""
    if len(set(value_names)) != len(value_names):
        raise ValueError("ALIGN_VALUE_ID_DUPLICATED")
    for name, values in columns.items():
        if len(values) != len(value_names):
            raise ValueError(f"ALIGN_COLUMN_WRONG_LENGTH:{name}")

    row_of: dict[str, int] = {name: i for i, name in enumerate(row_names)}
    """Indexed each identifier's relationship-matrix row."""
    value_of: dict[str, int] = {name: i for i, name in enumerate(value_names)}
    """Indexed each identifier's caller-value row."""
    wanted: list[str]
    """Declared the roster that will be aligned in one of two order policies."""
    dropped: list[str]
    """Declared the matrix rows excluded by the selected order policy."""
    if keep is None:
        wanted = [name for name in row_names if name in value_of]
        """Retained common identifiers in relationship-matrix order."""
        dropped = [name for name in row_names if name not in value_of]
        """Recorded matrix rows absent from the value table."""
    else:
        wanted = [str(value) for value in keep]
        """Retained the caller's explicit analysis order."""
        dropped = [name for name in row_names if name not in set(wanted)]
        """Recorded matrix rows outside the explicit analysis roster."""
        if len(set(wanted)) != len(wanted):
            raise ValueError("ALIGN_KEEP_ID_DUPLICATED")
        for name in wanted:
            if name not in row_of:
                raise ValueError(f"ALIGN_NOT_IN_RELATIONSHIP:{name}")
            if name not in value_of:
                raise ValueError(f"ALIGN_NOT_IN_VALUES:{name}")
    if not wanted:
        raise ValueError("ALIGN_NOBODY_IN_COMMON")

    rows: list[int] = [row_of[name] for name in wanted]
    """Selected relationship-matrix rows in the requested order."""
    taken: list[int] = [value_of[name] for name in wanted]
    """Selected value-table rows in the requested order."""
    aligned: dict[str, Any] = {
        "relationship": np.ascontiguousarray(matrix[np.ix_(rows, rows)]),
        "order": wanted,
        "dropped": dropped,
    }
    """Created the values-free aligned relationship and roster record."""
    observed: np.ndarray = np.ones(len(wanted), dtype=bool)
    """Initialised the complete-case indicator before reading value columns."""
    for name, values in columns.items():
        column: np.ndarray = np.asarray(values, dtype=np.float64)[taken]
        """Aligned one numeric column to the requested roster."""
        here: np.ndarray = np.isfinite(column)
        """Identified rows where this aligned column is observed."""
        if not allow_missing and not here.all():
            first: str = wanted[int(np.flatnonzero(~here)[0])]
            """Named the first missing value for the stable refusal detail."""
            raise ValueError(f"ALIGN_VALUE_MISSING:{name}:{first}")
        observed &= here
        """Updated the complete-case indicator with this column's status."""
        aligned[name] = np.ascontiguousarray(column)
        """Stored the aligned contiguous column under its caller-supplied name."""
    aligned["observed"] = observed
    """Stored the final complete-case indicator beside the aligned arrays."""
    return aligned


# There is no one-shot `fit(x, k, y)`: preparing once and fitting many
# responses reuses the expensive decomposition and makes bootstrap fitting
# affordable.
