"""A trait whose measurement stops at a limit, as audiometry does.

Run it with `uv run python examples/censored.py`.

The heritability that comes back is of the *complete* trait: the number you
would have had if the instrument reached far enough. It is comparable with an
ordinary heritability, and not with one fitted to values where the censored
ones were replaced by the limit. It is maximum likelihood, never REML.
"""

from __future__ import annotations

from statistics import NormalDist

import asterism
import numpy as np

from synthetic import pedigree, trait

relationship, order = pedigree(families=200, children=2)
"""Built two hundred families of four people each."""

people: int = len(order)
"""Read the roster size from the returned order."""

complete: np.ndarray = trait(relationship, heritability=0.5, seed=11)
"""Drew the trait as it would be if nothing stopped the measurement."""

ceiling: float = NormalDist().inv_cdf(0.5)
"""Fixed the instrument at the generating population median before the draw."""

censoring: np.ndarray = (complete >= ceiling).astype(np.int64)
"""Marked 1 where the value is at or above the limit, 0 where measured."""

value: np.ndarray = np.where(censoring == 0, complete, np.nan)
"""Kept measured values; the censored ones are not read at all."""

limit: np.ndarray = np.full(people, ceiling)
"""Gave every person the same limit. Real audiometry varies by frequency."""

design: np.ndarray = np.ones((people, 1))
"""Supplied an intercept only."""

fit: dict[str, object] = asterism.tobit_fit(
    relationship, value, censoring, limit, design
)
"""Fitted the complete-trait heritability by maximum likelihood."""

print(f"people:             {people}")
print(f"censored:           {censoring.sum()} of {people}")
print(f"h2 (true 0.5):      {float(fit['heritability']):.3f}")
print(f"converged:          {fit['converged']}")

interval: dict[str, object] = asterism.tobit_interval(
    relationship, value, censoring, limit, design
)
"""Profile interval for the same complete-trait heritability."""

print(f"95% interval:       [{interval['lower']:.3f}, {interval['upper']:.3f}]")
