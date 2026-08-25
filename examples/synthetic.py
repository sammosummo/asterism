"""Synthetic pedigrees and traits, so every example runs without any data.

Nothing here is part of Asterism's interface. It exists so the examples beside
it are complete programs a reader can run, rather than fragments that assume
variables they never define.
"""

from __future__ import annotations

import asterism
import numpy as np


def pedigree(
    families: int = 80,
    children: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """Build unrelated nuclear families: two founders and some full siblings.

    Args:
        families: How many unrelated families to build.
        children: How many full siblings in each family.

    Returns:
        The additive relationship matrix and the row order it is in.
    """
    parents: list[tuple[str, str]] = [
        (f"{family}-dad", f"{family}-mum") for family in range(families)
    ]
    """Named the two founders of every family."""

    ids: list[str] = [name for pair in parents for name in pair] + [
        f"{family}-child{child}"
        for family in range(families)
        for child in range(children)
    ]
    """Listed every founder, then every child, in one stable order."""

    father: list[str | None] = [None] * (2 * families) + [
        parents[family][0] for family in range(families) for _ in range(children)
    ]
    """Left founders unparented; gave every child its family's father."""

    mother: list[str | None] = [None] * (2 * families) + [
        parents[family][1] for family in range(families) for _ in range(children)
    ]
    """Left founders unparented; gave every child its family's mother."""

    return asterism.relationship_matrix(ids, father, mother)


def trait(
    relationship: np.ndarray,
    heritability: float,
    seed: int,
) -> np.ndarray:
    """Draw one trait with exactly the covariance the model assumes.

    Args:
        relationship: Additive relationship matrix.
        heritability: The proportion of variance the pedigree explains.
        seed: Fixed so an example prints the same numbers every time.

    Returns:
        One trait, centred on zero with unit total variance.
    """
    people: int = relationship.shape[0]
    """Read the roster size from the matrix rather than assuming it."""

    covariance: np.ndarray = heritability * relationship + (
        1.0 - heritability
    ) * np.eye(people)
    """Built the covariance a heritability of this size implies."""

    return np.linalg.cholesky(covariance) @ np.random.default_rng(seed).normal(
        size=people
    )


def correlated_traits(
    relationship: np.ndarray,
    heritabilities: tuple[float, float],
    genetic_correlation: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two traits sharing genetic variance at a chosen correlation.

    Args:
        relationship: Additive relationship matrix.
        heritabilities: One heritability for each trait.
        genetic_correlation: The genetic correlation between them.
        seed: Fixed so an example prints the same numbers every time.

    Returns:
        The two traits, in the matrix's row order.
    """
    people: int = relationship.shape[0]
    """Read the roster size from the matrix rather than assuming it."""

    rng: np.random.Generator = np.random.default_rng(seed)
    """Drew every deviate for both traits from one seeded generator."""

    genetic: np.ndarray = np.linalg.cholesky(relationship) @ rng.normal(
        size=(people, 2)
    )
    """Drew independent genetic deviates with the pedigree's covariance."""

    second_genetic: np.ndarray = (
        genetic_correlation * genetic[:, 0]
        + np.sqrt(1.0 - genetic_correlation**2) * genetic[:, 1]
    )
    """Mixed the second trait's genetic part to the requested correlation."""

    residual: np.ndarray = rng.normal(size=(people, 2))
    """Drew independent residuals, which share nothing between traits."""

    first: np.ndarray = (
        np.sqrt(heritabilities[0]) * genetic[:, 0]
        + np.sqrt(1.0 - heritabilities[0]) * residual[:, 0]
    )
    """Combined genetic and residual parts at the first heritability."""

    second: np.ndarray = (
        np.sqrt(heritabilities[1]) * second_genetic
        + np.sqrt(1.0 - heritabilities[1]) * residual[:, 1]
    )
    """Combined genetic and residual parts at the second heritability."""

    return first, second
