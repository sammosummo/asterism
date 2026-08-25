"""A complete, runnable example: heritability of one trait in a pedigree.

Run it with `uv run python examples/quickstart.py`. It builds its own pedigree
and its own trait from a fixed seed, so it needs no data and prints the same
numbers every time.
"""

from __future__ import annotations

import asterism
import numpy as np

rng: np.random.Generator = np.random.default_rng(0)
"""Fixed the seed so this example prints the same numbers on every machine."""

FAMILIES: int = 80
"""Chose enough nuclear families to estimate a variance component."""

CHILDREN: int = 4
"""Gave each family four full siblings and two founding parents."""

parents: list[tuple[str, str]] = [
    (f"{family}-dad", f"{family}-mum") for family in range(FAMILIES)
]
"""Named the two founders of every family."""

ids: list[str] = [name for pair in parents for name in pair] + [
    f"{family}-child{child}" for family in range(FAMILIES) for child in range(CHILDREN)
]
"""Listed every founder, then every child, in one stable order."""

father: list[str | None] = [None] * (2 * FAMILIES) + [
    parents[family][0] for family in range(FAMILIES) for _ in range(CHILDREN)
]
"""Left founders unparented; gave every child its family's father."""

mother: list[str | None] = [None] * (2 * FAMILIES) + [
    parents[family][1] for family in range(FAMILIES) for _ in range(CHILDREN)
]
"""Left founders unparented; gave every child its family's mother."""

relationship, order = asterism.relationship_matrix(ids, father, mother)
"""Built the additive relationship matrix, and the row order it is in."""

people: int = len(order)
"""Counted rows from the returned order rather than assuming it."""

heritability: float = 0.5
"""Chose the true heritability this example will try to recover."""

chol: np.ndarray = np.linalg.cholesky(
    heritability * relationship + (1.0 - heritability) * np.eye(people)
)
"""Factorised the covariance implied by that heritability."""

trait: np.ndarray = chol @ rng.normal(size=people)
"""Drew one trait with exactly the covariance the model assumes."""

design: np.ndarray = np.ones((people, 1))
"""Supplied an intercept only; add columns for age, sex and so on."""

model: asterism.PreparedModel = asterism.prepare(design, relationship)
"""Prepared the one-trait model. Preparation is reused across many traits."""

fit: dict[str, object] = model.fit(trait)
"""Fitted by REML. Pass estimator='ml' for maximum likelihood instead."""

print(f"people:        {people}")
print(f"h2 (true 0.5): {fit['h2']:.3f}")
print(f"converged:     {fit['converged']}")

interval: dict[str, object] = fit["interval"]
"""Profile-likelihood interval, returned with the fit rather than recomputed."""

print(f"95% interval:  [{interval['lower']:.3f}, {interval['upper']:.3f}]")

test: dict[str, object] = fit["test"]
"""Likelihood-ratio test against zero heritability, on the correct mixture."""

print(f"p (h2 = 0):    {test['p_value']:.2e}")
