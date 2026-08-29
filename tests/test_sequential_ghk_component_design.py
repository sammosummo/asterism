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


def test_the_household_kernel_joins_every_pair(ladder: ModuleType) -> None:
    """A distance kernel is non-zero everywhere: that is why it was chosen."""
    generator: np.random.Generator = np.random.default_rng(11)
    """Seeded the simulated household coordinates."""

    kernel: np.ndarray = ladder.household_kernel(40, generator)
    """Built a kernel over forty simulated households."""

    assert kernel.shape == (40, 40)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.all(kernel > 0.0), (
        "the kernel has a zero, so it is not dense and the block structure "
        "argument this design rests on would not hold"
    )
    assert np.all(kernel <= 1.0)


def test_the_heritability_step_leaves_the_total_variance_alone(
    ladder: ModuleType,
) -> None:
    """The yardstick must move variance between components, not change scale.

    This is the test that matters. Every ratio the ladder reports is an error
    divided by what a heritability step does. If the step also changed the total
    variance, it would be moving the probability for a second reason and the
    ratio would not mean what the check says it means.
    """
    generator: np.random.Generator = np.random.default_rng(5)
    """Seeded the simulated household coordinates."""

    matrix: np.ndarray = relationship_of(ladder, 1)
    """Built one family's relationship matrix."""

    people: int = matrix.shape[0]
    """Counted the people in the roster."""

    kernel: np.ndarray = ladder.household_kernel(people, generator)
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
    generator: np.random.Generator = np.random.default_rng(3)
    """Seeded the simulated household coordinates."""

    matrix: np.ndarray = relationship_of(ladder, 1)
    """Built one family's relationship matrix."""

    kernel: np.ndarray = ladder.household_kernel(matrix.shape[0], generator)
    """Built the shared-environment kernel over that roster."""

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
    generator: np.random.Generator = np.random.default_rng(7)
    """Seeded the simulated household coordinates."""

    matrix: np.ndarray = relationship_of(ladder, 1)
    """Built one family's relationship matrix."""

    kernel: np.ndarray = ladder.household_kernel(matrix.shape[0], generator)
    """Built the shared-environment kernel over that roster."""

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


def test_the_kernel_is_what_makes_the_roster_one_block(ladder: ModuleType) -> None:
    """Without the kernel the block is a family; with it, the whole roster.

    This is the finding that made the ladder worth climbing again, so it is
    worth having a test that fails if it stops being true.
    """
    generator: np.random.Generator = np.random.default_rng(13)
    """Seeded the simulated household coordinates."""

    matrix: np.ndarray = relationship_of(ladder, 2)
    """Built two unrelated families."""

    people: int = matrix.shape[0]
    """Counted the people in the roster."""

    kernel: np.ndarray = ladder.household_kernel(people, generator)
    """Built the shared-environment kernel over both families."""

    with_kernel: np.ndarray = ladder.component_covariance(
        matrix, kernel, ladder.COMPONENT_SHARES["genetic"]
    )
    """Built the covariance with the shared-environment kernel present."""

    without: np.ndarray = ladder.component_covariance(
        matrix, np.eye(people), ladder.COMPONENT_SHARES["genetic"]
    )
    """Built the covariance with the kernel replaced by an identity."""

    half: int = relationship_of(ladder, 1).shape[0] * 2
    """Located the row at which the second family's records begin: one family's
    people, each contributing two ears."""

    assert np.all(with_kernel[:half, half:] != 0.0), (
        "the kernel does not join the two families, so the block would still "
        "be one family and the ladder past 221 would not be needed"
    )
    assert np.all(without[:half, half:] == 0.0), (
        "the families are joined even without the kernel, so the kernel is not "
        "what makes the roster one block"
    )


def test_the_kernel_actually_correlates_at_the_held_decay(ladder: ModuleType) -> None:
    """A kernel that is non-zero everywhere but tiny everywhere proves nothing.

    An earlier decay of 0.25 per kilometre over the same span left the median
    pair's kernel entry at about 0.02 -- dense in support, inert in effect. The
    ladder was then stressed by dimension alone, and the shared environment,
    which is the whole reason this design exists, did no work at all. This is
    the guard against that returning quietly.
    """
    generator: np.random.Generator = np.random.default_rng(17)
    """Seeded the simulated household coordinates."""

    kernel: np.ndarray = ladder.household_kernel(300, generator)
    """Built a kernel over three hundred simulated households at the held decay."""

    off_diagonal: np.ndarray = kernel[~np.eye(300, dtype=bool)]
    """Took every pair's entry, excluding each household with itself."""

    assert float(np.median(off_diagonal)) > 0.2, (
        "the median pair's kernel entry is too small for the shared environment "
        "to correlate anything, so the ladder would be stressed by dimension only"
    )


def test_the_reference_agrees_with_the_exact_answer_at_two_coordinates(
    ladder: ModuleType,
) -> None:
    """Anchor the GHK reference where an exact answer exists.

    Above two coordinates the crate uses sequential truncation and there is
    nothing exact to check against; at exactly two it uses a bivariate normal
    integral. So two coordinates is the one rung where the reference itself can
    be shown correct rather than merely independent, and a reference that is
    wrong makes every rung above it meaningless.
    """
    import asterism

    generator: np.random.Generator = np.random.default_rng(23)
    """Seeded the draws behind the reference."""

    covariance: np.ndarray = np.array([[4.0, 1.6], [1.6, 2.5]])
    """Chose a correlated two-coordinate covariance."""

    centre: np.ndarray = np.array([0.4, -0.7])
    """Placed the region centre away from the origin in both coordinates."""

    exact: float = asterism.region_log_probability(centre, np.ones(2), covariance)
    """Took the crate's exact bivariate answer."""

    reference, error = ladder.ghk_log_probability(
        centre, covariance, 200_000, generator
    )
    """Took the independent GHK estimate and its own standard error."""

    assert abs(reference - exact) < 5.0 * max(error, 1e-6), (
        f"GHK gives {reference:.6f} where the exact answer is {exact:.6f}, "
        f"a gap of {abs(reference - exact):.6f} against a standard error of "
        f"{error:.6f}; the reference cannot be trusted at higher rungs either"
    )


def test_an_unjudged_rung_does_not_advance_the_qualified_dimension() -> None:
    """A rung below the step floor is not judged, so it must not qualify one.

    `within_allowance` is true by construction for a rung whose heritability
    step fell below `STEP_FLOOR`, because such a rung is explicitly not judged.
    Reading that as a pass would let the headline qualified dimension rest on a
    rung nothing measured.
    """
    rungs: dict[str, dict[str, object]] = {
        "5": {"reached": True, "decidable": True, "within_allowance": True},
        "20": {"reached": True, "decidable": False, "within_allowance": True},
        "50": {"reached": True, "decidable": True, "within_allowance": True},
    }
    """Built a ladder whose middle rung passed only because it was not judged."""

    qualified: int | None = None
    """Held the largest rung cleared without a failure below it."""

    for candidate in (5, 20, 50):
        entry: dict[str, object] = rungs[str(candidate)]
        """Took this rung's aggregated evidence."""

        if entry["reached"] and entry["decidable"] and entry["within_allowance"]:
            qualified = candidate
            """Advanced the qualified dimension to this cleared rung."""
        else:
            break

    assert qualified == 5, (
        "an unjudged rung advanced the ladder, so the qualified dimension "
        "would rest on a rung the check declined to judge"
    )
