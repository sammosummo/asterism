"""Two traits measured differently: one censored, one not.

Run it with `uv run python examples/mixed.py`.

This is the pairing the extended high-frequency work needs: a hearing threshold
that runs into the instrument's ceiling, beside a conventional one that does
not. The genetic correlation is unaffected by the difference in how the two are
seen, which is what makes a mixed pair worth fitting at all.
"""

from __future__ import annotations

import asterism
import numpy as np
from synthetic import correlated_traits, pedigree

relationship, order = pedigree(families=200, children=2)
"""Built two hundred families of four people each."""

people: int = len(order)
"""Read the roster size from the returned order."""

high, conventional = correlated_traits(
    relationship,
    heritabilities=(0.5, 0.5),
    genetic_correlation=0.6,
    seed=13,
)
"""Drew two thresholds sharing genetic variance at a correlation of 0.6."""

ceiling: float = float(np.quantile(high, 0.6))
"""Put the ceiling where forty per cent of the high-frequency trait exceeds it."""

censoring: np.ndarray = (high >= ceiling).astype(np.int64)
"""Marked 1 at or above the limit, 0 where the threshold was measured."""

first: dict[str, object] = {
    "kind": "censored",
    "value": np.where(censoring == 0, high, np.nan),
    "censoring": censoring,
    "limit": np.full(people, ceiling),
}
"""Described the high-frequency threshold, which the instrument cuts off."""

second: dict[str, object] = {
    "kind": "continuous",
    "value": conventional,
    "censoring": np.zeros(people, dtype=np.int64),
    "limit": np.zeros(people),
}
"""Described the conventional threshold, which nothing censors."""

design: np.ndarray = np.ones((people, 1))
"""Supplied an intercept only, shared by both traits."""

fit: dict[str, object] = asterism.mixed_bivariate_fit(
    relationship, first, second, design
)
"""Fitted the genetic correlation across the two ways of seeing a value."""

print(f"people:                {people}")
print(f"censored:              {int(censoring.sum())} of {people}")
print(f"rho_g     (true 0.6):  {float(fit['genetic_correlation']):.3f}")
print(f"converged:             {fit['converged']}")

interval: dict[str, object] = asterism.mixed_bivariate_interval(
    relationship, first, second, design
)
"""Profile interval for the genetic correlation."""

print(f"95% interval:          [{interval['lower']:.3f}, {interval['upper']:.3f}]")
