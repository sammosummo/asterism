"""A censored trait with two records per person, as an audiogram has two ears.

Run it with `uv run python examples/censored_components.py`.

**Why a second component.** Both of a person's records resemble each other for
two reasons: their genes, which their relatives share, and everything else about
that person, which nobody shares. With only a genetic component the second has
nowhere to go but the first, and the heritability takes it. A person-level
component gives it a home.

`grouping_matrix` builds that component. It gives one where two rows share a
group, so grouping by the person makes the person-level matrix and grouping by
the home would make a household one -- the relation is sharing, and a person is
one thing to share.

Compare components by their mean-diagonal proportions rather than their
coefficients: rescaling a matrix describes the same model and moves the second
while leaving the first alone.
"""

from __future__ import annotations

import asterism
import numpy as np

from synthetic import pedigree, trait

relationship, order = pedigree(families=150, children=2)
"""Built a hundred and fifty families of four people each."""

people: int = len(order)
"""Read the roster size from the returned order."""

genetic: np.ndarray = trait(relationship, heritability=1.0, seed=23)
"""Drew the purely genetic part, which both of a person's records will share."""

generator: np.random.Generator = np.random.default_rng(24)
"""Fixed the draw so this example prints the same numbers every time.

A different seed from the genetic draw on purpose: `trait` seeds its own
generator, so reusing 23 here made the person-level part the very draw the
genetic part was built from, and the two components arrived correlated."""

personal: np.ndarray = generator.normal(size=people)
"""Drew what belongs to the person and not to their genes."""

rows: int = people * 2
"""Two records per person: the two ears of one audiogram."""

expanded: np.ndarray = np.kron(relationship, np.ones((2, 2)))
"""Expanded the relationship so a person's own two records carry kinship one."""

listener: np.ndarray = asterism.grouping_matrix(
    [f"listener-{row // 2}" for row in range(rows)]
)
"""Built the person-level component by grouping the rows that share a person."""

complete: np.ndarray = (
    np.sqrt(0.4) * np.repeat(genetic, 2)
    + np.sqrt(0.3) * np.repeat(personal, 2)
    + np.sqrt(0.3) * generator.normal(size=rows)
)
"""Composed the complete trait: 0.4 genetic, 0.3 person-level, 0.3 left over."""

ceiling: float = float(np.quantile(complete, 0.6))
"""Put the instrument's ceiling where it censors two records in five."""

censoring: np.ndarray = (complete >= ceiling).astype(np.int64)
"""Marked 1 where the value is at or above the limit, 0 where measured."""

value: np.ndarray = np.where(censoring == 0, complete, np.nan)
"""Kept measured values; the censored ones are not read at all."""

limit: np.ndarray = np.full(rows, ceiling)
"""Gave every record the same limit. Real audiometry varies by frequency."""

design: np.ndarray = np.ones((rows, 1))
"""Supplied an intercept only."""

model: asterism.CensoredComponentModel = asterism.CensoredComponentModel(
    [expanded, listener], design
)
"""Built the model: additive kinship, person-level, and the residual it keeps."""

fit: dict[str, object] = model.fit(value, censoring, limit)
"""Fitted all of it by maximum likelihood, censoring and all."""

shares: list[float] = fit["mean_diagonal_proportions"]  # type: ignore[assignment]
"""Took the comparable quantity, the residual's share last."""

print(f"records:              {rows} ({people} people, two each)")
print(f"censored:             {int(censoring.sum())} of {rows}")
print(f"additive (true 0.4):  {shares[0]:.3f}")
print(f"person   (true 0.3):  {shares[1]:.3f}")
print(f"residual (true 0.3):  {shares[2]:.3f}")
print(f"converged:            {fit['converged']}")

interval: dict[str, object] = model.interval(value, censoring, limit, component=1)
"""Profiled the person-level proportion while everything else refits."""

print(f"person 95% interval:  [{interval['lower']:.3f}, {interval['upper']:.3f}]")
print(f"upper on its bound:   {interval['upper_limited']}")
print(f"profile failures:     {interval['profile_failures']}")

# **Read those two together.** An end resting on its bound is reported the same
# way whether the likelihood never fell away before it or the profile could not
# be evaluated there, and only the failure count separates them. Here it is the
# second: at a proportion of one the person-level matrix is the whole covariance
# and it is singular -- ones within a person, so half the rank of its size --
# so there is no residual left to make it invertible and no fit to be had. The
# interval widened to the bound, which is the safe direction, and the upper end
# is not a statement about the data.
