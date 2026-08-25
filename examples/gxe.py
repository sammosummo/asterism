"""Genes acting differently along a measured environment.

Run it with `uv run python examples/gxe.py`.

The genetic covariance between two people becomes their relationship times a
surface in their two environments. This example uses the random-regression
surface, where each person's genetic value is a slope in the environment, so
genetic variance grows away from its centre.
"""

from __future__ import annotations

import numpy as np
from synthetic import pedigree

import asterism

relationship, order = pedigree(families=500, children=2)
"""Built five hundred families of four people each: this test needs data."""

people: int = len(order)
"""Read the roster size from the returned order."""

rng: np.random.Generator = np.random.default_rng(31)
"""Fixed the seed for the environment and every deviate drawn below."""

environment: np.ndarray = rng.uniform(-1.0, 1.0, size=people)
"""Measured an environment on a range, rather than a binary label."""

chol: np.ndarray = np.linalg.cholesky(relationship + 1e-8 * np.eye(people))
"""Factorised the pedigree once, to draw genetic deviates with its covariance."""

intercept: np.ndarray = chol @ rng.normal(size=people)
"""Drew each person's genetic value at the centre of the environment."""

slope: np.ndarray = chol @ rng.normal(size=people)
"""Drew how each person's genetic value changes along the environment."""

SLOPE_SCALE: float = 0.9
"""Chose how strongly genetic effects vary with the environment."""

genetic: np.ndarray = intercept + SLOPE_SCALE * environment * slope
"""Combined them: a genetic value that depends on where the person sits."""

observed: np.ndarray = genetic + rng.normal(size=people)
"""Added residual noise of unit variance."""

design: np.ndarray = np.ones((people, 1))
"""Supplied an intercept only."""

model: asterism.GxeModel = asterism.GxeModel(
    relationship, environment, design, surface="random_regression"
)
"""Built the model on the surface this example simulated."""

fit: dict[str, object] = model.fit(observed, grid=[-1.0, 0.0, 1.0])
"""Fitted, reporting heritability at three prespecified environments."""

grid: list[float] = [float(point) for point in fit["environment"]]
"""Read back the reporting grid the fit was asked for."""

heritability: list[float] = [float(value) for value in fit["heritability"]]
"""Read the heritability at each point of that grid."""

variance: list[float] = [float(value) for value in fit["genetic_variance"]]
"""Read the genetic variance at each point, which is what the slope changes."""

correlation: float = float(fit["genetic_correlation"][0][2])
"""Read how alike genetic effects are at the two ends of the environment."""

print(f"people:                    {people}")
for point, h2, var in zip(grid, heritability, variance, strict=True):
    print(f"environment {point:+.0f}: h2 {h2:.3f}, genetic variance {var:.3f}")
print(f"rho_g between the ends:    {correlation:.3f}")
print(f"converged:                 {fit['converged']}")

interaction: dict[str, object] = model.test(observed, "interaction")
"""Tested whether genetic effects vary with the environment at all."""

print(f"p (no interaction):        {interaction['p_value']:.4f}")

print()
print("Genetic variance is smallest at the centre and grows towards either end,")
print("which is what a random-regression slope does. Genes at opposite ends of")
print("the environment are almost unrelated, and the test finds it. Detecting")
print("interaction takes far more data than estimating a heritability does.")
