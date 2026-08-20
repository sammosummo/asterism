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
"""

from __future__ import annotations

import asterism
import numpy as np
import pytest

# One pedigree, used by every family that takes a relationship matrix, so a
# difference between families is a difference in the interval and not in the
# data.
PAIRS = 60
SEED = 20_260_820


def pedigree(pairs: int = PAIRS) -> np.ndarray:
    """Sibling pairs: twice the kinship is one on the diagonal, a half within."""
    n = 2 * pairs
    k = np.eye(n)
    for pair in range(pairs):
        k[2 * pair, 2 * pair + 1] = 0.5
        k[2 * pair + 1, 2 * pair] = 0.5
    return k


def household(pairs: int = PAIRS) -> np.ndarray:
    """A second matrix: one within a pair, nought between."""
    n = 2 * pairs
    h = np.zeros((n, n))
    for pair in range(pairs):
        for a in (2 * pair, 2 * pair + 1):
            for b in (2 * pair, 2 * pair + 1):
                h[a, b] = 1.0
    return h


def draw(k: np.ndarray, h2: float, seed: int) -> np.ndarray:
    """One trait with a known heritability on the given matrix."""
    n = k.shape[0]
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(h2 * k + (1.0 - h2) * np.eye(n))
    return factor @ rng.standard_normal(n)


@pytest.fixture(scope="module")
def k() -> np.ndarray:
    return pedigree()


@pytest.fixture(scope="module")
def design() -> np.ndarray:
    return np.ones((2 * PAIRS, 1))


def same(got: float, pinned: float, what: str) -> None:
    """Exact, not close. A tolerance here would hide the thing this catches."""
    assert got == pinned, (
        f"{what} moved from {pinned!r} to {got!r}. That may be intended -- "
        "unifying the bracketing tolerances is expected to move endpoints -- "
        "but it has to be explained rather than absorbed."
    )


def test_one_trait_interval(k, design):
    y = draw(k, 0.5, SEED)
    fit = asterism.prepare(design, k).fit(y)
    interval = fit["interval"]
    same(interval["lower"], 0.0, "prepared lower")
    same(interval["upper"], 0.7230325389053068, "prepared upper")
    # The one family that carries ADR 0004's mixture verdict. The lower
    # end is on its bound here and the mixture says nought belongs; the
    # upper end is not, so there is no question to answer there.
    assert interval["contains_lower_bound"] is True
    assert interval["contains_upper_bound"] is None


def test_component_interval(k, design):
    y = draw(k, 0.5, SEED + 1)
    model = asterism.ComponentModel([k, household()], design)
    got = model.interval(y, component=0)
    same(got["lower"], 0.0, "components lower")
    same(got["upper"], 0.9970064880382881, "components upper")


def test_bivariate_interval(k, design):
    first = draw(k, 0.5, SEED + 2)
    second = 0.6 * first + 0.8 * draw(k, 0.5, SEED + 3)
    # One row per observed person-trait, in person order, and a design that is
    # block diagonal over the two traits -- an intercept of its own for each.
    y = np.column_stack([first, second]).ravel()
    observed = np.ones((2 * PAIRS, 2), dtype=bool)
    pair_design = np.tile(np.eye(2), (2 * PAIRS, 1))
    model = asterism.BivariateModel(k, observed, pair_design)
    got = model.interval(y, quantity="h2_first")
    same(got["lower"], 0.3472428115669697, "bivariate lower")
    same(got["upper"], 0.9999999999999998, "bivariate upper")


def test_spatial_interval(design):
    k_here = pedigree()
    n = k_here.shape[0]
    rng = np.random.default_rng(SEED + 4)
    place = rng.uniform(0.0, 10.0, size=(n, 2))
    distance = np.sqrt(
        ((place[:, None, :] - place[None, :, :]) ** 2).sum(axis=2)
    )
    y = draw(k_here, 0.5, SEED + 5)
    model = asterism.SpatialModel([k_here], distance, design)
    got = model.interval(y, quantity="0")
    same(got["lower"], 0.0, "spatial lower")
    same(got["upper"], 0.8196170452907089, "spatial upper")


def test_gxe_interval(k, design):
    n = k.shape[0]
    environment = np.linspace(-1.0, 1.0, n)
    y = draw(k, 0.5, SEED + 6)
    model = asterism.GxeModel(k, environment, design)
    got = model.interval(y, quantity="heritability", first=0.0)
    same(got["lower"], 0.20789513197229442, "gxe lower")
    same(got["upper"], 0.999999, "gxe upper")


def test_discrete_gxe_interval(k, design):
    n = k.shape[0]
    environment = np.array([1.0 if i % 2 else 2.0 for i in range(n)])
    y = draw(k, 0.5, SEED + 7)
    model = asterism.DiscreteGxeModel(k, environment, design)
    got = model.correlation_interval(y)
    same(got["lower"], -1.0, "discrete gxe lower")
    same(got["upper"], 1.0, "discrete gxe upper")


def test_liability_interval(k, design):
    latent = draw(k, 0.5, SEED + 8)
    status = (latent > float(np.quantile(latent, 0.7))).astype(float)
    model = asterism.LiabilityModel(k, status, design)
    got = model.interval()
    same(got["lower"], 0.0, "liability lower")
    same(got["upper"], 1.0, "liability upper")


def test_every_family_with_an_interval_is_pinned_here():
    """A new family must arrive with its endpoints pinned, or this fails.

    The consolidation is only safe if the baseline is complete. A family added
    to the package and left out of this file would be consolidated with nothing
    to check it against.
    """
    pinned = {
        "prepare",
        "ComponentModel",
        "BivariateModel",
        "SpatialModel",
        "GxeModel",
        "DiscreteGxeModel",
        "LiabilityModel",
    }
    here = set(globals())
    exposed = {
        name
        for name in asterism.__all__
        if name.endswith("Model") or name == "prepare"
    }
    # Families whose interval is not a profile of this kind, and why.
    exempt = {
        # A score test on a whole gene, with no parameter to profile.
        "VariantSetModel",
        # Reports a marker effect and its Wald interval, not a profile.
        "AssociationModel",
        # Confidence sets for two estimands, on their own record.
        "LatentMediationModel",
        # The sealed object; `prepare` is how it is reached and is pinned.
        "PreparedModel",
        # Has no interval at all yet: it exposes `cells` and `fit` and nothing
        # else. There is nothing to pin until it grows one, and when it does it
        # belongs in `pinned` above rather than here.
        "AutoregressiveModel",
        # Has no interval yet, and what it should be an interval *of* is an
        # open question rather than an unwritten function. ADR 0010 records
        # that the kernel's floor and rate are not separately estimable while
        # the correlation they describe is, so an interval belongs on
        # `correlation(d)` at separations that matter and not on either
        # parameter. When that is settled it belongs in `pinned` above.
        "RepeatedModel",
    }
    missing = exposed - pinned - exempt
    assert not missing, (
        f"{sorted(missing)} expose a model with no pinned interval here. "
        "Add one before consolidating, or add it to `exempt` with the reason."
    )
    assert "test_one_trait_interval" in here
