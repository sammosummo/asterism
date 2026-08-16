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
from ._core import gene_burden_matrix as _gene_burden_matrix
from ._core import gene_linear_matrix as _gene_linear_matrix
from ._core import local_ibd_matrix as _local_ibd_matrix
from ._core import posterior_local_ibd_matrix as _posterior_local_ibd_matrix
from ._core import relationship as _relationship
from .latent_mediation import LatentMediationModel
from .models import (
    BivariateModel,
    ComponentModel,
    AssociationModel,
    GxeModel,
    DiscreteGxeModel,
    LiabilityModel,
    SpatialModel,
    kinship_classes,
)

__all__ = [
    "PreparedModel",
    "prepare",
    "gene_linear_matrix",
    "gene_burden_matrix",
    "local_ibd_matrix",
    "posterior_local_ibd_matrix",
    "relationship_matrix",
    "ComponentModel",
    "BivariateModel",
    "AssociationModel",
    "GxeModel",
    "DiscreteGxeModel",
    "LiabilityModel",
    "LatentMediationModel",
    "SpatialModel",
    "kinship_classes",
    "__version__",
]


def _contains_effective_mask(values: Any) -> bool:
    stack = [values]
    seen: set[int] = set()
    while stack:
        value = stack.pop()
        if np.ma.isMaskedArray(value):
            if np.any(np.ma.getmaskarray(value)):
                return True
            value = np.ma.getdata(value)
        if isinstance(value, np.ndarray):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            if value.dtype.kind == "O":
                stack.extend(value.flat)
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            stack.extend(value)
    return False


def _integer_is_exact_binary64(value: Any) -> bool:
    try:
        converted = float(value)
        return np.isfinite(converted) and int(converted) == int(value)
    except (OverflowError, ValueError):
        return False


def _preflight_binary64_values(
    values: Any, *, not_exact_code: str, not_real_code: str
) -> None:
    stack = [(values, False)]
    seen: set[int] = set()
    exact_integer_limit = 2**53
    while stack:
        value, scalar_required = stack.pop()
        if scalar_required and isinstance(value, (np.ndarray, list, tuple)):
            raise ValueError(not_real_code)
        if isinstance(value, np.ndarray):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            kind = value.dtype.kind
            if kind == "O":
                stack.extend((item, True) for item in value.flat)
            elif kind == "c" or kind not in "biuf":
                raise ValueError(not_real_code)
            elif kind in "iu":
                if kind == "u":
                    needs_exact_check = value > exact_integer_limit
                else:
                    needs_exact_check = (value > exact_integer_limit) | (
                        value < -exact_integer_limit
                    )
                if any(
                    not _integer_is_exact_binary64(item)
                    for item in value[needs_exact_check].flat
                ):
                    raise ValueError(not_exact_code)
            elif kind == "f" and value.dtype.itemsize > np.dtype(np.float64).itemsize:
                with np.errstate(over="ignore", invalid="ignore"):
                    converted = value.astype(np.float64)
                    restored = converted.astype(value.dtype)
                finite = np.isfinite(value)
                if np.any(restored[finite] != value[finite]):
                    raise ValueError(not_exact_code)
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            stack.extend((item, False) for item in value)
        elif type(value) is bool:
            continue
        elif type(value) is int:
            if not _integer_is_exact_binary64(value):
                raise ValueError(not_exact_code)
        elif type(value) is float:
            continue
        elif isinstance(value, np.generic):
            scalar_dtype = np.generic.dtype.__get__(value, np.generic)
            if type(value) is not scalar_dtype.type:
                raise ValueError(not_real_code)
            kind = scalar_dtype.kind
            if kind == "b":
                continue
            if kind in "iu":
                if not _integer_is_exact_binary64(value):
                    raise ValueError(not_exact_code)
            elif kind == "f" and (
                scalar_dtype.itemsize > np.dtype(np.float64).itemsize
                and np.isfinite(value)
                and scalar_dtype.type(float(value)) != value
            ):
                raise ValueError(not_exact_code)
            elif kind != "f":
                raise ValueError(not_real_code)
        else:
            raise ValueError(not_real_code)


def _real_float_array(
    values: Any,
    *,
    dimensions: int,
    masked_code: str,
    not_exact_code: str,
    not_real_code: str,
    wrong_shape_code: str,
) -> np.ndarray:
    if _contains_effective_mask(values):
        raise ValueError(masked_code)
    _preflight_binary64_values(
        values, not_exact_code=not_exact_code, not_real_code=not_real_code
    )
    try:
        converted_input = np.asanyarray(values)
    except ValueError as error:
        raise ValueError(wrong_shape_code) from error
    if _contains_effective_mask(converted_input):
        raise ValueError(masked_code)
    _preflight_binary64_values(
        converted_input,
        not_exact_code=not_exact_code,
        not_real_code=not_real_code,
    )
    array = np.ndarray.view(converted_input, np.ndarray).copy()
    _preflight_binary64_values(
        array, not_exact_code=not_exact_code, not_real_code=not_real_code
    )
    if array.ndim != dimensions:
        raise ValueError(wrong_shape_code)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.ascontiguousarray(array, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(not_real_code) from error


def _lineage_array(values: Any, *, dimensions: int) -> np.ndarray:
    if _contains_effective_mask(values):
        raise ValueError("LOCAL_IBD_LINEAGE_MASKED")
    try:
        converted_input = np.asanyarray(values)
    except ValueError as error:
        raise ValueError("LOCAL_IBD_LINEAGE_WRONG_SHAPE") from error
    if _contains_effective_mask(converted_input):
        raise ValueError("LOCAL_IBD_LINEAGE_MASKED")
    array = np.ndarray.view(converted_input, np.ndarray).copy()
    if array.ndim != dimensions or array.shape[-1] != 2:
        raise ValueError("LOCAL_IBD_LINEAGE_WRONG_SHAPE")
    if array.dtype.kind not in "iu":
        raise ValueError("LOCAL_IBD_LINEAGE_NOT_INTEGER")
    if array.dtype.kind == "i" and np.any(array < 0):
        raise ValueError("LOCAL_IBD_LINEAGE_NEGATIVE")
    return np.ascontiguousarray(array, dtype=np.uint64)


def gene_linear_matrix(
    genotypes: Any,
    *,
    variant_weights: Any | None = None,
) -> np.ndarray:
    """Build ``K = (G W)(G W)'`` from alternate-allele dosages.

    ``variant_weights`` multiply genotype columns directly, so their squares
    are the variance weights. Asterism does not centre, standardise, impute or
    normalise the submitted values. Every dosage must therefore be observed,
    finite and between zero and two. Masked, complex or non-exact binary64
    inputs are refused before any binary64 cast.

    The return value is an ordinary dense NumPy matrix.
    """
    genotypes = _real_float_array(
        genotypes,
        dimensions=2,
        masked_code="GENE_MATRIX_GENOTYPE_MASKED",
        not_exact_code="GENE_MATRIX_GENOTYPE_NOT_EXACT_FLOAT64",
        not_real_code="GENE_MATRIX_GENOTYPE_NOT_REAL",
        wrong_shape_code="GENE_MATRIX_GENOTYPE_WRONG_SHAPE",
    )
    weights = (
        None
        if variant_weights is None
        else _real_float_array(
            variant_weights,
            dimensions=1,
            masked_code="GENE_MATRIX_WEIGHT_MASKED",
            not_exact_code="GENE_MATRIX_WEIGHT_NOT_EXACT_FLOAT64",
            not_real_code="GENE_MATRIX_WEIGHT_NOT_REAL",
            wrong_shape_code="GENE_MATRIX_WEIGHT_WRONG_SHAPE",
        )
    )
    return _gene_linear_matrix(genotypes, weights)


def gene_burden_matrix(
    genotypes: Any,
    *,
    variant_weights: Any | None = None,
) -> np.ndarray:
    """Build the rank-one weighted-burden relationship matrix ``b b'``.

    The burden is ``b_i = sum_j G_ij w_j``. Weights are nonnegative and encode
    the prespecified same-direction alternative. As with
    :func:`gene_linear_matrix`, Asterism performs no centring, imputation or
    normalisation.
    """
    genotypes = _real_float_array(
        genotypes,
        dimensions=2,
        masked_code="GENE_MATRIX_GENOTYPE_MASKED",
        not_exact_code="GENE_MATRIX_GENOTYPE_NOT_EXACT_FLOAT64",
        not_real_code="GENE_MATRIX_GENOTYPE_NOT_REAL",
        wrong_shape_code="GENE_MATRIX_GENOTYPE_WRONG_SHAPE",
    )
    weights = (
        None
        if variant_weights is None
        else _real_float_array(
            variant_weights,
            dimensions=1,
            masked_code="GENE_MATRIX_WEIGHT_MASKED",
            not_exact_code="GENE_MATRIX_WEIGHT_NOT_EXACT_FLOAT64",
            not_real_code="GENE_MATRIX_WEIGHT_NOT_REAL",
            wrong_shape_code="GENE_MATRIX_WEIGHT_WRONG_SHAPE",
        )
    )
    return _gene_burden_matrix(genotypes, weights)


def local_ibd_matrix(lineages: Any) -> np.ndarray:
    """Build the local additive relationship matrix from two lineage labels.

    ``lineages`` has one row per subject and exactly two nonnegative integer
    founder-haplotype labels per row. Labels are opaque equality tokens. IBD0,
    IBD1 and IBD2 are 0, 0.5 and 1 off the diagonal; a locally autozygous
    subject retains a diagonal of 2. Masked labels are refused; the full
    unsigned 64-bit label domain is retained.
    """
    lineages = _lineage_array(lineages, dimensions=2)
    return _local_ibd_matrix(lineages)


def posterior_local_ibd_matrix(
    lineage_draws: Any,
    *,
    draw_weights: Any | None = None,
) -> np.ndarray:
    """Average local additive relationship matrices over lineage draws.

    ``lineage_draws`` has shape ``draws × subjects × 2``. Explicit nonnegative
    weights are normalised; omitting them assigns equal weight to every draw.
    No malformed draw is silently omitted. The result is the posterior-mean
    plug-in matrix, not propagation of lineage uncertainty through a trait
    likelihood. Masked or non-exact binary64 weights are refused.
    """
    lineage_draws = _lineage_array(lineage_draws, dimensions=3)
    weights = (
        None
        if draw_weights is None
        else _real_float_array(
            draw_weights,
            dimensions=1,
            masked_code="LOCAL_IBD_WEIGHT_MASKED",
            not_exact_code="LOCAL_IBD_WEIGHT_NOT_EXACT_FLOAT64",
            not_real_code="LOCAL_IBD_WEIGHT_NOT_REAL",
            wrong_shape_code="LOCAL_IBD_WEIGHT_WRONG_SHAPE",
        )
    )
    return _posterior_local_ibd_matrix(lineage_draws, weights)


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
