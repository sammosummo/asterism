"""Asterism: variance components models for quantitative genetics.

One way in, and it is two steps::

    model = asterism.prepare(x, k)
    record = model.fit(y)

Building the model validates and decomposes; the decomposition is what every
fit runs against, so many fits against one prepared model cost one preparation.
Holding a prepared model is itself the proof that validation happened, which is
why there is no way to skip it.

Asterism takes arrays that are already in memory and returns a record, also in
memory. It reads no files, knows no databases, and writes nothing. Where the
record goes afterwards is the caller's business.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import PreparedModel, __version__

__all__ = ["PreparedModel", "prepare", "__version__"]


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


# There is deliberately no one-shot `fit(x, k, y)` here. `docs/adr/0003` rules
# it out for the first release: one way in, because both predecessors grew their
# interface one convenience at a time. Preparing once and fitting many times is
# also the thing that makes a bootstrap affordable, and a one-shot call throws
# that away every time it is used.
