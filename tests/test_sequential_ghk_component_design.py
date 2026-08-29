"""Contracts for the several-component design of the region-probability ladder.

The ladder in `checks/sequential_against_ghk.py` was built for the repeated-
measures audiogram, where a censored threshold is conditioned on a nearly
complete audiogram. The several-component censored model is a different design:
one frequency at a time, and a shared-environment kernel that is non-zero for
every pair.

These tests do not measure the approximation -- that is the check's own job, and
it takes hours. They fix the things the check's verdict depends on, which are
cheap to state and easy to break silently:

1. The heritability step moves variance between components and leaves the total
   alone. If it did not, the yardstick would be a change of scale and every
   ratio the ladder reports would be meaningless.
2. The families really are unrelated, so the genetic term is block diagonal and
   only the kernel joins them.
3. The kernel really is dense, which is the whole reason the ladder had to be
   climbed again.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the maintained ladder command."""


@pytest.fixture(scope="module")
def ladder() -> ModuleType:
    """Load the ladder command without running its command line."""
    path: Path = ROOT / "checks" / "sequential_against_ghk.py"
    """Located the exact maintained command under test."""

    specification: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location("sequential_against_ghk", path)
    )
    """Created a module specification through the command's file boundary."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the module governed by the validated specification."""

    sys.modules[specification.name] = module
    """Registered the module before evaluating its annotations."""

    specification.loader.exec_module(module)
    """Evaluated the command's definitions without invoking its entry point."""
    return module


def relationship_of(ladder: ModuleType, families: int) -> np.ndarray:
    """Build the relationship matrix over several independent families."""
    import asterism

    ids, fathers, mothers = ladder.many_families(ladder.FAMILY_GENERATIONS, families)
    """Built the roster of several unrelated families."""

    matrix, _ = asterism.relationship_matrix(ids, fathers, mothers)
    """Built the relationship matrix through Asterism's public interface."""
    return matrix


def test_the_component_shares_account_for_all_the_variance(
    ladder: ModuleType,
) -> None:
    """The four shares sum to one, so the scale means what it says."""
    assert sum(ladder.COMPONENT_SHARES.values()) == pytest.approx(1.0)


def test_several_families_are_unrelated_to_each_other(ladder: ModuleType) -> None:
    """Only the kernel may join families; the genetic term must not."""
    matrix: np.ndarray = relationship_of(ladder, 3)
    """Built three families' worth of relationship."""

    one: int = relationship_of(ladder, 1).shape[0]
    """Counted the people in a single family."""

    assert matrix.shape[0] == 3 * one
    assert np.all(matrix[:one, one:] == 0.0), (
        "people in different families are related, so the genetic term would "
        "join families and the kernel would not be the only thing that does"
    )


def test_the_household_kernel_joins_only_people_who_share_a_home(
    ladder: ModuleType,
) -> None:
    """A household kernel is categorical: one within a home, nought between homes.

    Spatial kernels were dropped from this model on 29 August 2026. The
    difference matters to the likelihood's blocks, not only to the science: a
    distance kernel is non-zero for every pair arithmetically, whereas this one
    is block diagonal and can only ever join people who actually live together.
    """
    kernel: np.ndarray = ladder.household_kernel(np.ones((9, 9)), 3)
    """Built a kernel over one family of nine, in homes of three."""

    assert kernel.shape == (9, 9)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), 1.0)
    assert set(np.unique(kernel)) <= {0.0, 1.0}, (
        "a household kernel takes only nought and one; a graded entry would "
        "make it a distance kernel again"
    )
    assert np.allclose(kernel[0:3, 0:3], 1.0)
    assert np.allclose(kernel[0:3, 3:], 0.0), (
        "the kernel joins people in different homes, so it is not categorical "
        "and could enlarge a likelihood block beyond a household"
    )


def test_a_home_that_does_not_divide_the_roster_is_still_whole(
    ladder: ModuleType,
) -> None:
    """The last home may be short, and must not be left ragged or empty."""
    kernel: np.ndarray = ladder.household_kernel(np.ones((8, 8)), 3)
    """Built a kernel over one family of eight, so its last home holds two."""

    assert np.allclose(kernel[6:8, 6:8], 1.0)
    assert np.allclose(kernel[6:8, 0:6], 0.0)


def test_the_heritability_step_leaves_the_total_variance_alone(
    ladder: ModuleType,
) -> None:
    """The yardstick must move variance between components, not change scale.

    This is the test that matters. Every ratio the ladder reports is an error
    divided by what a heritability step does. If the step also changed the total
    variance, it would be moving the probability for a second reason and the
    ratio would not mean what the check says it means.
    """
    matrix: np.ndarray = relationship_of(ladder, 1)
    """Built one family's relationship matrix."""

    kernel: np.ndarray = ladder.household_kernel(matrix)
    """Built the shared-environment kernel over that roster."""

    baseline: np.ndarray = ladder.component_covariance(
        matrix, kernel, ladder.COMPONENT_SHARES["genetic"]
    )
    """Built the covariance at the generating heritability."""

    stepped: np.ndarray = ladder.component_covariance(
        matrix, kernel, ladder.COMPONENT_SHARES["genetic"] - ladder.HERITABILITY_STEP
    )
    """Built the covariance one heritability step lower."""

    assert np.allclose(np.diag(baseline), np.diag(stepped)), (
        "the heritability step changed the total variance, so it is a change "
        "of scale and not a yardstick"
    )
    assert not np.allclose(baseline, stepped), (
        "the heritability step changed nothing at all"
    )


def test_the_total_variance_is_the_spread_and_the_noise(ladder: ModuleType) -> None:
    """The diagonal is the modelled spread plus test-retest noise, and nothing else."""
    matrix: np.ndarray = relationship_of(ladder, 1)
    """Built one family's relationship matrix."""

    kernel: np.ndarray = ladder.household_kernel(matrix)
    """Built the household kernel over that roster."""

    covariance: np.ndarray = ladder.component_covariance(
        matrix, kernel, ladder.COMPONENT_SHARES["genetic"]
    )
    """Built the covariance at the generating heritability."""

    expected: float = (
        ladder.SPREAD[ladder.COMPONENT_FREQUENCY_INDEX] ** 2 + ladder.NUGGET_DB**2
    )
    """Calculated the variance the four shares and the noise should produce."""

    assert np.allclose(np.diag(covariance), expected)


def test_both_ears_share_everything_above_the_ear(ladder: ModuleType) -> None:
    """Two ears of one person differ only by the ear term and the noise."""
    matrix: np.ndarray = relationship_of(ladder, 1)
    """Built one family's relationship matrix."""

    kernel: np.ndarray = ladder.household_kernel(matrix)
    """Built the household kernel over that roster."""

    covariance: np.ndarray = ladder.component_covariance(
        matrix, kernel, ladder.COMPONENT_SHARES["genetic"]
    )
    """Built the covariance at the generating heritability."""

    gap: float = float(covariance[0, 0] - covariance[0, 1])
    """Took the difference between one ear's variance and its covariance with the other."""

    expected: float = (
        ladder.COMPONENT_SHARES["ear"]
        * ladder.SPREAD[ladder.COMPONENT_FREQUENCY_INDEX] ** 2
        + ladder.NUGGET_DB**2
    )
    """Calculated the ear-specific variance and the noise, which is all that separates them."""

    assert gap == pytest.approx(expected)


def test_the_household_kernel_does_not_join_two_families(ladder: ModuleType) -> None:
    """A household kernel cannot merge unrelated families, and that is the point.

    The earlier spatial kernel was non-zero for every pair, so a block rule
    testing entries against nought would have put the whole roster in one block.
    A categorical household kernel cannot do that: it joins only people who
    share a home, so blocks stay the size of families unless a home straddles
    two, and the region probability is never handed more coordinates than the
    pedigree already implies.
    """
    matrix: np.ndarray = relationship_of(ladder, 2)
    """Built two unrelated families."""

    one_family: int = relationship_of(ladder, 1).shape[0]
    """Counted the people in a single family."""

    kernel: np.ndarray = ladder.household_kernel(matrix, ladder.HOUSEHOLD_SIZE)
    """Built the household kernel over both families."""

    covariance: np.ndarray = ladder.component_covariance(
        matrix, kernel, ladder.COMPONENT_SHARES["genetic"]
    )
    """Built the covariance with the household kernel present."""

    half: int = one_family * 2
    """Located the row at which the second family's records begin."""

    assert np.allclose(covariance[:half, half:], 0.0), (
        "the household kernel joins the two families, so it would enlarge a "
        "likelihood block beyond the pedigree just as a spatial kernel would"
    )


def test_a_person_with_no_relatives_lives_alone(ladder: ModuleType) -> None:
    """Homes are formed inside families, so unrelated people cannot share one.

    Passing an identity relationship matrix says nobody is related to anybody,
    which makes every person their own family, so every home holds one person.
    That is the behaviour that stops a home straddling two unrelated families,
    which is what would join them in the covariance for no modelled reason.
    """
    kernel: np.ndarray = ladder.household_kernel(np.eye(6), 3)
    """Built a kernel over six people who are related to nobody."""

    assert np.allclose(kernel, np.eye(6)), (
        "unrelated people were put in a home together, so a home can straddle "
        "two families and join them in the covariance"
    )
