"""The mixed genetic correlation can be tested against one, not only nought.

Against nought the question is whether two traits share any genes. Against one
it is whether they share all of them, and that value sits on the edge of a
correlation's range, so it needs the Self-Liang mixture rather than a plain
chi-square. This holds both the behaviour and the reference rule.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import asterism
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from synthetic import correlated_traits, pedigree

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root holding the shared example helpers."""


def mixed_pair(
    genetic_correlation: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], np.ndarray]:
    """Build one censored-with-continuous pair at a chosen correlation.

    Args:
        genetic_correlation: The truth the pair is simulated at.
        seed: Fixed so a test prints the same numbers every time.

    Returns:
        The relationship matrix, both trait specifications and the design.
    """
    relationship, order = pedigree(families=150, children=2)
    """Built a pedigree big enough to answer the question."""

    people: int = len(order)
    """Read the roster size from the returned order."""

    high, conventional = correlated_traits(
        relationship, (0.5, 0.5), genetic_correlation, seed=seed
    )
    """Drew two traits sharing genes at the requested correlation."""

    ceiling: float = float(np.quantile(high, 0.6))
    """Put the instrument's ceiling where two fifths run past it."""

    censoring: np.ndarray = (high >= ceiling).astype(np.int64)
    """Marked which of the first trait's values ran into the limit."""

    first: dict[str, Any] = {
        "kind": "censored",
        "value": np.where(censoring == 0, high, np.nan),
        "censoring": censoring,
        "limit": np.full(people, ceiling),
    }
    """Described the censored trait."""

    second: dict[str, Any] = {
        "kind": "continuous",
        "value": conventional,
        "censoring": np.zeros(people, dtype=np.int64),
        "limit": np.zeros(people),
    }
    """Described the trait nothing censors."""

    return relationship, first, second, np.ones((people, 1))


def test_the_boundary_null_uses_the_mixture_and_nought_does_not() -> None:
    """Require the reference rule to follow the null, not the caller."""
    relationship, first, second, design = mixed_pair(0.6, seed=13)
    """Built a pair that is neither unrelated nor identical."""

    interior: dict[str, Any] = asterism.mixed_bivariate_test(
        relationship, first, second, design, null=0.0
    )
    """Tested against an interior value."""

    boundary: dict[str, Any] = asterism.mixed_bivariate_test(
        relationship, first, second, design, null=1.0
    )
    """Tested against the edge of the correlation's range."""

    assert interior["rule"] == "chi2_1"
    assert boundary["rule"] == "mixture_50_50"
    """Held each null to the reference distribution it actually needs."""


def test_a_correlation_of_one_is_rejected_when_it_is_false() -> None:
    """Keep the boundary test able to find that two traits differ."""
    relationship, first, second, design = mixed_pair(0.6, seed=13)
    """Built a pair sharing some genes but not all."""

    result: dict[str, Any] = asterism.mixed_bivariate_test(
        relationship, first, second, design, null=1.0
    )
    """Asked whether the same genes govern both."""

    assert result["p_value"] < 0.01, result
    """Required the test to reject a correlation of one that is not true."""


def test_the_two_nulls_separate_a_shared_pair_from_an_identical_one() -> None:
    """Require each null to be the harder one for the truth it contradicts.

    Asserting a true null simply survives at 0.05 would be a test built to fail
    one time in twenty. This asks the relative question instead, which is what
    a reader actually wants to know: does the evidence point the right way?
    """
    relationship, first, second, design = mixed_pair(1.0, seed=13)
    """Built a pair governed by exactly the same genes."""

    against_nought: dict[str, Any] = asterism.mixed_bivariate_test(
        relationship, first, second, design, null=0.0
    )
    """Asked whether they share any genes. They share all of them."""

    against_one: dict[str, Any] = asterism.mixed_bivariate_test(
        relationship, first, second, design, null=1.0
    )
    """Asked whether they share all of them. They do."""

    assert against_nought["statistic"] > 10.0 * against_one["statistic"], (
        f"nought {against_nought['statistic']} against one {against_one['statistic']}"
    )
    assert against_one["statistic"] >= 0.0
    """Required the false null to be the one the evidence weighs against."""


@pytest.mark.parametrize("null", [-1.5, 1.5, float("nan"), float("inf")])
def test_a_null_outside_the_range_is_refused(null: float) -> None:
    """Refuse a null a correlation could never take."""
    relationship, first, second, design = mixed_pair(0.6, seed=13)
    """Built any valid pair; the refusal precedes the fitting."""

    with pytest.raises(ValueError, match="MIXED_BIVARIATE_NULL_OUTSIDE"):
        asterism.mixed_bivariate_test(relationship, first, second, design, null=null)
    """Named the refusal rather than letting the optimiser meet it."""
