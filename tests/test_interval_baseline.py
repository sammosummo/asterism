"""What every family's interval reports today, pinned to the last digit.

ADR 0011 consolidates seven implementations of one interval recipe into one
module. Two of the copies differ by two comment lines and two more by two
tokens, so most of what separates them is accident rather than choice -- but
they do not agree on the tolerances they bracket to, and unifying those moves
endpoints in every family and not only in the one being repaired.

That makes "the consolidation changed nothing" a claim nobody could check
afterwards, because there would be nothing to check it against. This is the
something. It is not a statement that these numbers are right; it is a
statement that they are what the package returns today, so that every
difference afterwards has to be explained rather than assumed.

**Reading a failure here.** A failing assertion is not necessarily a fault. It
means an endpoint moved, and the question is whether the move was intended and
why. Update the pinned value once you can say which, and say it in the commit.

The values were produced on this file's own seeds, through the public Python
interface, because that is what a caller sees and what the record crosses.

**Why these are compared within a tolerance and not to the last digit.** They
were pinned exactly at first, and exact pinning cannot survive two platforms.
ADR 0012 settles that Mac and Linux agree exactly on outcome status, refusal
codes, boundary states and field presence, and agree on numbers within written
tolerances rather than bit for bit. Pinning a number to its last digit is a
promise the second platform never made: measured on the same commit, the
spatial upper end differs between them in its last two digits and the
gene-by-environment lower end in its last four, while every other endpoint here
agrees bit for bit. That is the arithmetic underneath differing, not the recipe.
See ``TOLERANCE`` below for what the number is and why.
"""

from __future__ import annotations

import asterism
import numpy as np
import numpy.typing as npt
import pytest

# One pedigree, used by every family that takes a relationship matrix, so a
# difference between families is a difference in the interval and not in the
# data.
PAIRS: int = 60
"""Number of sibling pairs shared by the interval fixtures."""

SEED: int = 20_260_820
"""Base seed separating the deterministic family-specific responses."""


def pedigree(pairs: int = PAIRS) -> npt.NDArray[np.float64]:
    """Sibling pairs: twice the kinship is one on the diagonal, a half within."""
    n: int = 2 * pairs
    """Counted the two people contributed by every sibling pair."""

    k: npt.NDArray[np.float64] = np.eye(n)
    """Started the relationship matrix at unrelated identity."""

    for pair in range(pairs):
        k[2 * pair, 2 * pair + 1] = 0.5
        """Assigned one direction of the current sibling relationship."""

        k[2 * pair + 1, 2 * pair] = 0.5
        """Completed the symmetric sibling relationship."""
    """Constructed unrelated sibling pairs with first-degree relatedness."""
    return k


def household(pairs: int = PAIRS) -> npt.NDArray[np.float64]:
    """A second matrix: one within a four-person household, nought between.

    A household must not be the sibling pair itself. Give every pair its own
    household and twice the kinship is the identity plus the within-pair
    pattern while the household matrix is the identity plus that same pattern,
    so the residual identity is exactly ``2 * k - h``. The three covariance
    bases then have rank two, nothing separates additive genetic from shared
    household variation, and the fit is refused with
    ``COMPONENTS_COVARIANCE_BASES_RANK_DEFICIENT``. That refusal is right: it
    is the familiar result that sibling pairs alone do not identify the two.
    Housing two unrelated sibling pairs together leaves all three bases
    independent, which is what this fixture needs to pin an interval at all.
    """
    n: int = 2 * pairs
    """Counted the two people contributed by every sibling pair."""

    h: npt.NDArray[np.float64] = np.zeros((n, n))
    """Initialised an empty shared-household covariance matrix."""

    for start in range(0, n, 4):
        members: range = range(start, min(start + 4, n))
        """Selected the two consecutive sibling pairs sharing this household."""

        for a in members:
            for b in members:
                h[a, b] = 1.0
                """Marked the current ordered pair as sharing a household."""
        """Filled the complete covariance block for one four-person household."""
    """Constructed independent four-person household blocks."""
    return h


def draw(k: npt.NDArray[np.float64], h2: float, seed: int) -> npt.NDArray[np.float64]:
    """One trait with a known heritability on the given matrix."""
    n: int = k.shape[0]
    """Counted the people represented by the relationship matrix."""

    rng: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic generator for this response."""

    factor: npt.NDArray[np.float64] = np.linalg.cholesky(
        h2 * k + (1.0 - h2) * np.eye(n)
    )
    """Factorised the covariance implied by the requested heritability."""
    return factor @ rng.standard_normal(n)


@pytest.fixture(scope="module")
def k() -> npt.NDArray[np.float64]:
    """Provide the common sibling relationship matrix."""
    return pedigree()


@pytest.fixture(scope="module")
def design() -> npt.NDArray[np.float64]:
    """Provide the common intercept-only fixed-effect design."""
    return np.ones((2 * PAIRS, 1))


TOLERANCE: float = 1e-11
"""How far an endpoint may move before this file calls it a change.

Two numbers bracket the choice. Below it is the arithmetic: the widest
disagreement measured between Mac and Linux on one commit is about five parts
in ten million million, on the gene-by-environment lower end, and the spatial
upper end differs by about one part in a thousand million million. Above it is
the recipe: ADR 0011's shared bracketing stops at ``1e-9``, so a real change to
how an interval is built moves an endpoint by about that much or more, as the
pinned prepared upper end moved when the tolerances were unified.

A millionth of a millionth sits roughly twenty times above the widest platform
difference and a hundred times below the smallest change worth reporting, so it
separates the two without needing to know which platform is running. It is not
a licence to absorb a moved endpoint: anything this catches still has to be
explained rather than re-pinned.
"""


def same(got: float, pinned: float, what: str) -> None:
    """Close, to a stated tolerance, rather than exact. See ``TOLERANCE``."""
    moved: float = abs(got - pinned) / max(abs(pinned), 1.0)
    """Measured the move relative to the pinned value, or absolutely near nought."""

    assert moved <= TOLERANCE, (
        f"{what} moved from {pinned!r} to {got!r}, by {moved:.3e} relative, "
        f"which is more than the {TOLERANCE:.0e} two platforms are allowed to "
        "differ by. That may be intended -- unifying the bracketing tolerances "
        "is expected to move endpoints -- but it has to be explained rather "
        "than absorbed."
    )


def test_one_trait_interval(
    k: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
) -> None:
    """Pin the sealed one-trait interval after common-recipe consolidation."""
    y: npt.NDArray[np.float64] = draw(k, 0.5, SEED)
    """Drew the deterministic response reserved for the one-trait interval."""

    fit: dict[str, object] = asterism.prepare(design, k).fit(y)
    """Fitted the sealed one-trait model through its public interface."""

    interval: dict[str, object] = fit["interval"]
    """Selected the interval record embedded in the one-trait fit."""

    same(interval["lower"], 0.0, "prepared lower")
    # Shared bracketing stops at ADR 0011's 1e-9 tolerance rather than the
    # prepared model's former 1e-13 root tolerance.
    same(interval["upper"], 0.723032539171961, "prepared upper")
    # The one family that carries ADR 0004's mixture verdict. The lower
    # end is on its bound here and the mixture says nought belongs; the
    # upper end is not, so there is no question to answer there.
    assert interval["contains_lower_bound"] is True
    assert interval["contains_upper_bound"] is None


def test_component_interval(
    k: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
) -> None:
    """Pin the several-component interval on its reportable contribution scale."""
    y: npt.NDArray[np.float64] = draw(k, 0.5, SEED + 1)
    """Drew the deterministic response reserved for the component interval."""

    model: asterism.ComponentModel = asterism.ComponentModel([k, household()], design)
    """Prepared additive, household, and residual covariance components."""

    got: dict[str, object] = model.interval(y, component=0)
    """Profiled the first component's mean-diagonal contribution proportion."""

    same(got["lower"], 0.0, "components lower")
    # Both endpoints moved because the fixture had to change, not because the
    # recipe did. Housing every sibling pair on its own made the residual
    # identity exactly ``2 * k - h``, so the three covariance bases had rank
    # two and no fit could separate additive genetic from shared household
    # variation. The pinned upper endpoint of 1.0 was therefore the top of the
    # range under an unidentified model rather than a profiled crossing, and
    # ``upper_limited`` was true for the same reason. Two sibling pairs to a
    # household identifies all three bases and the upper endpoint is now an
    # ordinary interior crossing.
    assert got["profile_failures"] == 0
    assert got["upper_limited"] is False
    same(got["upper"], 0.8432385368612152, "components upper")


def test_bivariate_interval(
    k: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
) -> None:
    """Pin the bivariate first-trait heritability interval."""
    first: npt.NDArray[np.float64] = draw(k, 0.5, SEED + 2)
    """Drew the deterministic first trait."""

    second: npt.NDArray[np.float64] = 0.6 * first + 0.8 * draw(k, 0.5, SEED + 3)
    """Combined shared and independent variation into the second trait."""

    # One row per observed person-trait, in person order, and a design that is
    # block diagonal over the two traits -- an intercept of its own for each.
    y: npt.NDArray[np.float64] = np.column_stack([first, second]).ravel()
    """Stacked person-trait responses in the public interface's row order."""

    observed: npt.NDArray[np.bool_] = np.ones((2 * PAIRS, 2), dtype=bool)
    """Marked both traits observed for every person."""

    pair_design: npt.NDArray[np.float64] = np.tile(np.eye(2), (2 * PAIRS, 1))
    """Built one independent intercept for each trait."""

    model: asterism.BivariateModel = asterism.BivariateModel(k, observed, pair_design)
    """Prepared the balanced two-trait covariance model."""

    got: dict[str, object] = model.interval(y, quantity="h2_first")
    """Profiled the first trait's heritability."""
    assert {
        "estimate",
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "level",
        "profile_failures",
        "contains_lower_bound",
        "contains_upper_bound",
    } <= set(got)
    same(got["lower"], 0.3472428115669697, "bivariate lower")
    same(got["upper"], 0.9999999999999998, "bivariate upper")


def test_spatial_interval(design: npt.NDArray[np.float64]) -> None:
    """Pin the public spatial interval after common-recipe consolidation."""
    k_here: npt.NDArray[np.float64] = pedigree()
    """Relationship matrix used by this spatial fixture."""
    n: int = k_here.shape[0]
    """Number of people in the spatial fixture."""
    rng: np.random.Generator = np.random.default_rng(SEED + 4)
    """Deterministic stream used only to place people."""
    place: npt.NDArray[np.float64] = rng.uniform(0.0, 10.0, size=(n, 2))
    """Synthetic planar positions for the distance matrix."""
    distance: npt.NDArray[np.float64] = np.sqrt(
        ((place[:, None, :] - place[None, :, :]) ** 2).sum(axis=2)
    )
    """Pairwise Euclidean distances used by the synthetic spatial model."""
    y: npt.NDArray[np.float64] = draw(k_here, 0.5, SEED + 5)
    """Trait generated from the pedigree component alone."""
    model: asterism.SpatialModel = asterism.SpatialModel([k_here], distance, design)
    """Public spatial model whose interval is pinned."""
    got: dict[str, object] = model.interval(y, quantity="0")
    """Shared interval record for the first raw coefficient proportion."""
    assert set(got) == {
        "quantity",
        "estimate",
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "level",
        "profile_failures",
        "contains_lower_bound",
        "contains_upper_bound",
        "estimator",
    }
    same(got["lower"], 0.0, "spatial lower")
    # The shared search closes to 1e-9 rather than the former spatial rule of
    # 1e-4 times the estimate. Total-covariance-scale profile coordinates keep
    # every held fit converged while those additional halvings are evaluated.
    assert got["profile_failures"] == 0
    same(got["upper"], 0.8196161051309041, "spatial upper")


def test_gxe_interval(
    k: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
) -> None:
    """Pin the continuous gene-by-environment heritability interval."""
    n: int = k.shape[0]
    """Counted the people represented by the common pedigree."""

    environment: npt.NDArray[np.float64] = np.linspace(-1.0, 1.0, n)
    """Assigned an evenly spaced continuous environment across people."""

    y: npt.NDArray[np.float64] = draw(k, 0.5, SEED + 6)
    """Drew the deterministic response reserved for the continuous GxE interval."""

    model: asterism.GxeModel = asterism.GxeModel(k, environment, design)
    """Prepared the default random-regression environment surface."""

    got: dict[str, object] = model.interval(y, quantity="heritability", first=0.0)
    """Profiled heritability at the centre of the environment."""

    assert {
        "estimate",
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "level",
        "profile_failures",
        "contains_lower_bound",
        "contains_upper_bound",
    } <= set(got)
    # The common recipe closes the bracket to 1e-9 rather than this family's
    # former 1e-7; all held fits converged, so the last two digits are now
    # deliberate numerical resolution rather than an absorbed failure.
    assert got["profile_failures"] == 0
    same(got["lower"], 0.20789515498858546, "gxe lower")
    same(got["upper"], 0.999999, "gxe upper")


def test_discrete_gxe_interval(
    k: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
) -> None:
    """Pin the discrete gene-by-environment genetic-correlation interval."""
    n: int = k.shape[0]
    """Counted the people represented by the common pedigree."""

    environment: npt.NDArray[np.float64] = np.array(
        [1.0 if index % 2 else 2.0 for index in range(n)]
    )
    """Alternated people between two explicit environment groups."""

    y: npt.NDArray[np.float64] = draw(k, 0.5, SEED + 7)
    """Drew the deterministic response reserved for the discrete GxE interval."""

    model: asterism.DiscreteGxeModel = asterism.DiscreteGxeModel(k, environment, design)
    """Prepared the two-environment covariance model."""

    got: dict[str, object] = model.correlation_interval(y)
    """Profiled the directly parameterised cross-environment correlation."""

    assert {
        "estimate",
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "level",
        "profile_failures",
        "contains_lower_bound",
        "contains_upper_bound",
    } <= set(got)
    same(got["lower"], -1.0, "discrete gxe lower")
    same(got["upper"], 1.0, "discrete gxe upper")


def test_liability_interval(
    k: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
) -> None:
    """Pin the binary liability heritability interval."""
    latent: npt.NDArray[np.float64] = draw(k, 0.5, SEED + 8)
    """Drew the deterministic latent liability."""

    status: npt.NDArray[np.float64] = (latent > float(np.quantile(latent, 0.7))).astype(
        float
    )
    """Thresholded the liability into a thirty-percent case fraction."""

    model: asterism.LiabilityModel = asterism.LiabilityModel(k, status, design)
    """Prepared the public binary liability model."""

    got: dict[str, object] = model.interval()
    """Profiled heritability on the fixed unit-variance liability scale."""

    assert {
        "estimate",
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "level",
        "profile_failures",
        "contains_lower_bound",
        "contains_upper_bound",
    } <= set(got)
    same(got["lower"], 0.0, "liability lower")
    same(got["upper"], 1.0, "liability upper")


def test_every_family_with_an_interval_is_pinned_here() -> None:
    """A new family must arrive with its endpoints pinned, or this fails.

    The consolidation is only safe if the baseline is complete. A family added
    to the package and left out of this file would be consolidated with nothing
    to check it against.
    """
    pinned: set[str] = {
        "prepare",
        "ComponentModel",
        "BivariateModel",
        "SpatialModel",
        "GxeModel",
        "DiscreteGxeModel",
        "LiabilityModel",
    }
    """Listed every family whose shared profile endpoints are pinned above."""

    here: set[str] = set(globals())
    """Collected this module's test and fixture names."""

    exposed: set[str] = {
        name for name in asterism.__all__ if name.endswith("Model") or name == "prepare"
    }
    """Collected public model families that could require an interval pin."""

    # Families whose interval is not a profile of this kind, and why.
    exempt: set[str] = {
        # A score test on a whole gene, with no parameter to profile.
        "VariantSetModel",
        # Reports a marker effect and its Wald interval, not a profile.
        "AssociationModel",
        # Confidence sets for two estimands, on their own record.
        "LatentMediationModel",
        # The sealed object; `prepare` is how it is reached and is pinned.
        "PreparedModel",
        # Reports a fit and the two rates, and has no interval at all. It
        # arrived after this file did, which is what the check below is for.
        "AutoregressiveModel",
    }
    """Recorded public families whose inference is not this profile recipe."""

    missing: set[str] = exposed - pinned - exempt
    """Found public model families lacking either a pin or a stated exemption."""

    assert not missing, (
        f"{sorted(missing)} expose a model with no pinned interval here. "
        "Add one before consolidating, or add it to `exempt` with the reason."
    )
    assert "test_one_trait_interval" in here
