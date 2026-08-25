"""Genes acting differently in two environments, where the environment is a label.

Run it with `uv run python examples/discrete_gxe.py`.

Sex is the canonical environment; any binary label works the same way. The
model carries one genetic standard deviation per environment, one residual
standard deviation per environment, and one genetic correlation between them.
A genetic correlation below one is gene-by-environment interaction.
"""

from __future__ import annotations

import asterism
import numpy as np

from synthetic import correlated_traits, pedigree

relationship, order = pedigree(families=250, children=2)
"""Built two hundred and fifty families of four people each."""

people: int = len(order)
"""Read the roster size from the returned order."""

TRUE_CORRELATION: float = 0.6
"""Chose genes that act only partly the same in the two environments."""

in_first, in_second = correlated_traits(
    relationship,
    heritabilities=(0.5, 0.5),
    genetic_correlation=TRUE_CORRELATION,
    seed=23,
)
"""Drew what each person's trait would be in either environment."""

rng: np.random.Generator = np.random.default_rng(29)
"""Fixed the seed that assigns people to environments."""

environment: np.ndarray = rng.integers(0, 2, size=people).astype(np.float64)
"""Assigned each person to one environment or the other, at random."""

observed: np.ndarray = np.where(environment == 0, in_first, in_second)
"""Kept only the value belonging to the environment each person is in."""

design: np.ndarray = np.ones((people, 1))
"""Supplied an intercept only."""

model: asterism.DiscreteGxeModel = asterism.DiscreteGxeModel(
    relationship, environment, design
)
"""Built the discrete gene-by-environment model."""

fit: dict[str, object] = model.fit(observed)
"""Fitted a genetic and residual standard deviation for each environment."""

print(f"people:                 {people}")
print(f"in environment 1:       {int((environment == 1).sum())} of {people}")
print(f"rho_g     (true 0.6):   {float(fit['genetic_correlation']):.3f}")
print(f"converged:              {fit['converged']}")

interval: dict[str, object] = model.correlation_interval(observed)
"""Profile interval for the genetic correlation across environments."""

print(f"95% interval:           [{interval['lower']:.3f}, {interval['upper']:.3f}]")

test: dict[str, object] = model.test(observed, null="gene_by_environment")
"""Tested for interaction: is the genetic correlation one, and the scales equal?"""

print(f"p (no interaction):     {test['p_value']:.3f}")
