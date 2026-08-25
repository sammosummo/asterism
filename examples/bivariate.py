"""Two continuous traits, and the genetic correlation between them.

Run it with `uv run python examples/bivariate.py`.
"""

from __future__ import annotations

import asterism
import numpy as np
from synthetic import correlated_traits, pedigree

relationship, order = pedigree(families=120, children=3)
"""Built a hundred and twenty families of five people each."""

people: int = len(order)
"""Read the roster size from the returned order."""

first, second = correlated_traits(
    relationship,
    heritabilities=(0.5, 0.4),
    genetic_correlation=0.6,
    seed=7,
)
"""Drew two traits sharing genetic variance at a correlation of 0.6."""

observed: np.ndarray = np.ones((people, 2), dtype=bool)
"""Said that every person has both traits. False marks a missing one."""

values: np.ndarray = np.column_stack([first, second]).ravel()
"""Stacked trait within person: person 0's pair, then person 1's, and so on."""

design: np.ndarray = np.ones((2 * people, 1))
"""Supplied an intercept over the same stacked rows the values are in."""

model: asterism.BivariateModel = asterism.BivariateModel(relationship, observed, design)
"""Built the bivariate model over the pair."""

fit: dict[str, object] = model.fit(values)
"""Fitted both heritabilities and all three correlations at once."""

print(f"people:                {people}")
print(f"h2 first  (true 0.5):  {float(fit['h2_first']):.3f}")
print(f"h2 second (true 0.4):  {float(fit['h2_second']):.3f}")
print(f"rho_g     (true 0.6):  {float(fit['rho_g']):.3f}")
print(f"converged:             {fit['converged']}")

interval: dict[str, object] = model.interval(values, "rho_g")
"""Profile interval for the genetic correlation, not a Wald interval."""

print(f"95% interval:          [{interval['lower']:.3f}, {interval['upper']:.3f}]")

test: dict[str, object] = model.test(values, "rho_g", null=0.0)
"""Tested the genetic correlation against zero."""

print(f"p (rho_g = 0):         {test['p_value']:.2e}")
