"""Two variance components: genes and a shared household.

Run it with `uv run python examples/components.py`.

Note what the reported proportions mean. `mean_diagonal_proportions` gives each
component's share of the *average person's* variance, not the variance of the
component among the people who have one. If half your sample belongs to no
household, a true household variance of 0.2 is reported as 0.1, and it is right
to: the average person really does carry half as much of it. Everyone here
belongs to a household, so the two numbers agree.
"""

from __future__ import annotations

import asterism
import numpy as np
from synthetic import pedigree

FAMILIES: int = 150
"""Chose enough families to separate two components, if only just."""

relationship, order = pedigree(families=FAMILIES, children=2)
"""Built sibling pairs with both parents: four people to a family."""

people: int = len(order)
"""Read the roster size from the returned order."""

founders: int = 2 * FAMILIES
"""Counted the founders, which the pedigree lists before the children."""

household: np.ndarray = np.zeros((people, people))
"""Reserved a second component: who shares a roof with whom."""

for family in range(FAMILIES):
    members: list[int] = [
        2 * family,
        2 * family + 1,
        founders + 2 * family,
        founders + 2 * family + 1,
    ]
    """Put both parents and both children of one family under one roof."""

    household[np.ix_(members, members)] = 1.0
    """Marked every co-resident pair in this family's household."""
"""Left nobody without a household, so the mean diagonal is one."""

covariance: np.ndarray = 0.4 * relationship + 0.2 * household + 0.4 * np.eye(people)
"""Built a truth of forty per cent genetic and twenty per cent household."""

trait: np.ndarray = np.linalg.cholesky(covariance) @ np.random.default_rng(3).normal(
    size=people
)
"""Drew one trait carrying both components."""

model: asterism.ComponentModel = asterism.ComponentModel(
    [relationship, household], np.ones((people, 1))
)
"""Built a model over both components, in the order they are listed."""

fit: dict[str, object] = model.fit(trait)
"""Fitted both components at once by REML."""

proportions: list[float] = [float(value) for value in fit["mean_diagonal_proportions"]]
"""Read each component's share of the average person's variance."""

print(f"people:               {people}")
print(f"genetic   (true 0.4): {proportions[0]:.3f}")
print(f"household (true 0.2): {proportions[1]:.3f}")
print(f"converged:            {fit['converged']}")

household_test: dict[str, object] = model.test(trait, component=1)
"""Tested the household component against zero, on the correct mixture."""

print(f"p (household = 0):    {household_test['p_value']:.3f}")

print()
print("A household that is exactly one family is nearly the pedigree itself, so")
print("the two variances are hard to tell apart: unbiased, but wide. Households")
print("holding people who share no genes are what separate them.")
