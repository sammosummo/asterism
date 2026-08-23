"""The Python interface: numerical outputs and invalid-input behaviour."""

from __future__ import annotations

from collections.abc import Callable

import asterism
import numpy as np
import numpy.typing as npt
import pytest


def sibling_relationship(pairs: int) -> npt.NDArray[np.float64]:
    """Build unrelated sibling pairs on the twice-kinship scale."""
    k: npt.NDArray[np.float64] = np.eye(2 * pairs)
    """Started the relationship matrix at unrelated identity."""

    for pair in range(pairs):
        a, b = 2 * pair, 2 * pair + 1
        """Located the two people in the current sibling pair."""

        k[a, b] = k[b, a] = 0.5
        """Assigned symmetric first-degree relatedness within the pair."""
    """Constructed independent sibling-pair relationship blocks."""
    return k


def simulate(pairs: int, h2: float, seed: int) -> npt.NDArray[np.float64]:
    """Draw a trait with known heritability across unrelated sibling pairs."""
    rng: np.random.Generator = np.random.default_rng(seed)
    """Created the deterministic generator for this simulated trait."""

    shared: npt.NDArray[np.float64] = rng.standard_normal(pairs)
    """Drew one shared genetic effect per sibling pair."""

    own: npt.NDArray[np.float64] = rng.standard_normal((pairs, 2))
    """Drew the independent half of each sibling's genetic effect."""

    genetic: npt.NDArray[np.float64] = np.sqrt(h2) * (
        np.sqrt(0.5) * shared[:, None] + np.sqrt(0.5) * own
    )
    """Combined shared and individual terms at the requested genetic variance."""

    residual: npt.NDArray[np.float64] = np.sqrt(1.0 - h2) * rng.standard_normal(
        (pairs, 2)
    )
    """Drew independent residual variation at the complementary variance."""
    return (genetic + residual).reshape(-1)


@pytest.fixture(scope="module")
def model() -> asterism.PreparedModel:
    """Prepare the common large sibling-pair one-trait model."""
    pairs: int = 800
    """Selected enough sibling pairs for stable interface-level estimates."""
    return asterism.prepare(np.ones((2 * pairs, 1)), sibling_relationship(pairs))


def test_the_fit_returns_the_documented_numerical_fields(
    model: asterism.PreparedModel,
) -> None:
    """Return exactly the documented one-trait numerical fit fields."""
    record: dict[str, object] = model.fit(simulate(800, 0.5, 11))
    """Fitted the common model to a deterministic heritable trait."""

    assert set(record) == {
        "estimator",
        "n",
        "h2",
        "total_variance",
        "beta",
        "loglik",
        "converged",
        "boundary",
        "warnings",
        "build",
        "subject_order_sha256",
        "interval",
        "test",
        "standard_errors",
        "fixed_effects",
    }
    assert record["estimator"] == "reml"
    assert record["n"] == 1600
    assert record["converged"] is True
    assert record["boundary"] in {"interior", "lower", "upper"}
    assert record["build"] == asterism.build_identity()
    assert record["subject_order_sha256"] is None
    assert len(record["beta"]) == 1


def test_the_interval_keeps_one_shape_whatever_the_fit_did(
    model: asterism.PreparedModel,
) -> None:
    """Keep the interval record structurally stable across boundary states."""
    interval: dict[str, object] = model.fit(simulate(800, 0.5, 12))["interval"]
    """Selected the interval from a deterministic interior one-trait fit."""

    assert set(interval) == {
        "estimate",
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "contains_lower_bound",
        "contains_upper_bound",
        "level",
        "profile_failures",
        "recipe",
    }
    assert interval["estimate"] == pytest.approx(
        model.fit(simulate(800, 0.5, 12))["h2"]
    )
    assert interval["level"] == 0.95
    assert interval["recipe"] == "profile_mixture"
    assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0


def test_the_estimator_is_data_on_the_record_and_both_are_available(
    model: asterism.PreparedModel,
) -> None:
    """Expose both ML and REML while recording which estimator produced a fit."""
    y: npt.NDArray[np.float64] = simulate(800, 0.5, 13)
    """Drew one response shared by the two estimator comparisons."""

    assert model.fit(y, "reml")["estimator"] == "reml"
    assert model.fit(y, "ml")["estimator"] == "ml"
    with pytest.raises(ValueError, match="FIT_ESTIMATOR_INVALID"):
        model.fit(y, "restricted")


def test_a_heritable_trait_is_rejected_against_no_additive_variance(
    model: asterism.PreparedModel,
) -> None:
    """Use the boundary mixture to reject zero variance for a heritable trait."""
    test: dict[str, object] = model.fit(simulate(800, 0.7, 14))["test"]
    """Selected the zero-heritability test from a deterministic strong signal."""

    assert test["null"] == "h2 = 0"
    assert test["rule"] == "mixture_50_50"
    assert test["statistic"] > 0.0
    assert test["p_value"] < 1e-4
    # The mixture puts half the null's mass at zero, so nothing positive can
    # ever be less surprising than a half.
    assert test["p_value"] <= 0.5


def test_a_boundary_fit_withholds_the_standard_error_rather_than_faking_one() -> None:
    """Omit curvature uncertainty when the heritability estimate reaches a bound."""
    # No family structure in the response at all, so the estimate goes to zero.
    pairs: int = 200
    """Selected enough pairs for an unstructured response to favour the lower bound."""

    model: asterism.PreparedModel = asterism.prepare(
        np.ones((2 * pairs, 1)), sibling_relationship(pairs)
    )
    """Prepared the sibling model used for the lower-bound fit."""

    rng: np.random.Generator = np.random.default_rng(15)
    """Created the deterministic generator for an unrelated response."""

    record: dict[str, object] = model.fit(rng.standard_normal(2 * pairs))
    """Fitted an independent response against the sibling relationship matrix."""

    if record["boundary"] == "lower":
        assert record["h2"] == 0.0
        # The standard errors are a per-parameter mapping from which one
        # undefined at the optimum is *absent*. The mapping itself
        # stays, because the fixed effects still have theirs — nothing
        # degenerate happens to a regression coefficient when h2 pins.
        assert "h2" not in record["standard_errors"]
        assert record["standard_errors"]["beta"][0] is not None
        assert record["interval"]["lower"] == 0.0
        assert record["interval"]["lower_limited"] is True
        assert record["interval"]["contains_lower_bound"] is not None
        assert record["test"]["p_value"] == 1.0
    else:
        assert record["standard_errors"]["h2"] > 0.0


@pytest.mark.parametrize(
    ("build", "code"),
    [
        (lambda: (np.ones((10, 1)), np.eye(9)), "PREPARE_SHAPE_MISMATCH"),
        (lambda: (np.ones((10, 2)), np.eye(10)), "PREPARE_X_RANK_DEFICIENT"),
        (lambda: (np.full((10, 1), np.nan), np.eye(10)), "PREPARE_X_NOT_FINITE"),
        # One observation and one fixed effect leaves nothing to estimate a
        # variance from. Two observations and one fixed effect leaves one
        # residual degree of freedom, which is legitimate and is allowed.
        (
            lambda: (np.ones((1, 1)), np.eye(1)),
            "PREPARE_NO_RESIDUAL_DEGREES_OF_FREEDOM",
        ),
    ],
)
def test_validation_refuses_with_a_stable_code(
    build: Callable[[], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    code: str,
) -> None:
    """Expose stable refusal codes for invalid prepared-model inputs."""
    x, k = build()
    """Constructed the parametrised invalid design and relationship inputs."""

    with pytest.raises(ValueError, match=code):
        asterism.prepare(x, k)


def test_an_asymmetric_relationship_matrix_is_refused() -> None:
    """Refuse even one directional mismatch in the relationship matrix."""
    k: npt.NDArray[np.float64] = np.eye(10)
    """Started from a valid identity relationship matrix."""

    k[0, 1] = 0.5
    """Changed one off-diagonal entry without its symmetric partner."""

    with pytest.raises(ValueError, match="PREPARE_K_ASYMMETRIC"):
        asterism.prepare(np.ones((10, 1)), k)


def test_there_is_no_one_shot_fit() -> None:
    """Require explicit preparation before a one-trait fit."""
    # Preparing explicitly keeps the costly decomposition reusable.
    assert not hasattr(asterism, "fit")


def test_integer_input_widens_rather_than_being_refused(
    model: asterism.PreparedModel,
) -> None:
    """Widen exact integer designs to binary64 without refusal."""
    # Value-exact conversions happen silently; lossy ones do not happen at all.
    pairs: int = 20
    """Selected a compact sibling model for conversion behaviour."""

    k: npt.NDArray[np.float64] = sibling_relationship(pairs)
    """Built the binary64 sibling relationship matrix."""

    x: npt.NDArray[np.int64] = np.ones((2 * pairs, 1), dtype=np.int64)
    """Built an exactly representable integer intercept design."""

    prepared: asterism.PreparedModel = asterism.prepare(x, k)
    """Prepared the model after exact integer-to-binary64 widening."""

    assert prepared.n == 2 * pairs


def test_tobit_uses_the_shared_interval_record() -> None:
    """Censoring changes the objective, not the interval record."""
    pairs: int = 40
    """Selected enough sibling pairs for the censored profile smoke test."""

    y: npt.NDArray[np.float64] = simulate(pairs, 0.5, 143)
    """Drew the latent complete trait before applying an instrument limit."""

    limit_value: float = float(np.quantile(y, 0.8))
    """Selected a right-censoring limit at the trait's eightieth percentile."""

    censoring: npt.NDArray[np.int64] = (y >= limit_value).astype(np.int64)
    """Marked values at or above the instrument limit as right-censored."""

    limit: npt.NDArray[np.float64] = np.full(y.shape, limit_value)
    """Repeated the observation-specific censoring limit across subjects."""

    interval: dict[str, object] = asterism.tobit_interval(
        sibling_relationship(pairs),
        y,
        censoring,
        limit,
        np.ones((2 * pairs, 1)),
    )
    """Profiled complete-trait heritability through the public Tobit interface."""

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
    } <= set(interval)


def test_mixed_binary_censored_uses_the_shared_interval_record() -> None:
    """The supported diagnosis-hearing pair returns the common uncertainty shape."""
    pairs: int = 30
    """Selected a compact sibling design for the mixed-trait profile smoke test."""

    latent: npt.NDArray[np.float64] = simulate(pairs, 0.5, 144)
    """Drew the continuous liability underlying the binary first trait."""

    hearing: npt.NDArray[np.float64] = 0.4 * latent + simulate(pairs, 0.4, 145)
    """Constructed a genetically correlated continuous hearing trait."""

    hearing_limit: float = float(np.quantile(hearing, 0.8))
    """Selected a right-censoring limit at the hearing trait's upper tail."""

    first: dict[str, object] = {
        "kind": "binary",
        "value": np.full_like(latent, np.nan),
        "censoring": np.where(latent > np.quantile(latent, 0.7), 1, 2),
        "limit": np.zeros_like(latent),
    }
    """Encoded the first trait as a thresholded binary liability."""

    second: dict[str, object] = {
        "kind": "censored",
        "value": hearing,
        "censoring": (hearing >= hearing_limit).astype(np.int64),
        "limit": np.full(hearing.shape, hearing_limit),
    }
    """Encoded the second trait with explicit right-censoring statuses and limits."""

    interval: dict[str, object] = asterism.mixed_bivariate_interval(
        sibling_relationship(pairs),
        first,
        second,
        np.ones((2 * pairs, 1)),
    )
    """Profiled the mixed model's genetic correlation through the public interface."""

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
    } <= set(interval)


def test_every_fixed_effect_carries_wald_inference(
    model: asterism.PreparedModel,
) -> None:
    """Attach complete Wald inference to every fitted fixed effect."""
    record: dict[str, object] = model.fit(simulate(800, 0.5, 21))
    """Fitted the common intercept-only model to a deterministic trait."""

    effects: list[dict[str, object]] = record["fixed_effects"]
    """Selected the fixed-effect inference records in design-column order."""

    assert len(effects) == 1  # this fixture is intercept-only
    for effect in effects:
        assert set(effect) == {
            "estimate",
            "standard_error",
            "z",
            "p_value",
            "lower",
            "upper",
            "level",
            "rule",
        }
        assert effect["rule"] == "wald"
        assert effect["level"] == 0.95
        assert effect["lower"] < effect["estimate"] < effect["upper"]
        assert 0.0 <= effect["p_value"] <= 1.0
    assert record["standard_errors"]["beta"] == [effects[0]["standard_error"]]


def test_covariates_get_their_own_estimates_and_tests() -> None:
    """Separate a real covariate signal from an irrelevant design column."""
    # A real design: an intercept, a covariate that matters and one that does
    # not. The one that matters should be found and the one that does not
    # should not be, which is the whole use `-screen -all` is put to.
    pairs: int = 600
    """Selected enough pairs for stable fixed-effect discrimination."""

    n: int = 2 * pairs
    """Counted the people contributed by the sibling pairs."""

    rng: np.random.Generator = np.random.default_rng(31)
    """Created the deterministic fixed-effect design generator."""

    real: npt.NDArray[np.float64] = rng.standard_normal(n)
    """Drew the covariate assigned a true response effect."""

    noise: npt.NDArray[np.float64] = rng.standard_normal(n)
    """Drew the independent covariate assigned no response effect."""

    x: npt.NDArray[np.float64] = np.column_stack([np.ones(n), real, noise])
    """Combined the intercept, signal, and noise covariates into one design."""

    y: npt.NDArray[np.float64] = x @ np.array([1.0, 0.8, 0.0]) + simulate(
        pairs, 0.4, 32
    )
    """Combined known fixed effects with a heritable simulated response."""

    record: dict[str, object] = asterism.prepare(x, sibling_relationship(pairs)).fit(y)
    """Fitted the design and relationship matrix through the public interface."""

    intercept, matters, does_not = record["fixed_effects"]
    """Named the three fixed-effect records by their design-column meaning."""

    assert abs(matters["estimate"] - 0.8) < 4 * matters["standard_error"]
    assert matters["p_value"] < 1e-6
    assert does_not["p_value"] > 0.01
    assert abs(intercept["estimate"] - 1.0) < 4 * intercept["standard_error"]


def test_a_boundary_fit_still_reports_its_fixed_effects() -> None:
    """Retain fixed-effect inference when additive variance reaches a bound."""
    # h2's standard error is withheld at the bound; the fixed effects' are not.
    # Nothing degenerate happens to a regression coefficient when a variance
    # component pins.
    pairs: int = 200
    """Selected enough pairs for the unrelated response to reach the lower bound."""

    n: int = 2 * pairs
    """Counted the people contributed by the sibling pairs."""

    rng: np.random.Generator = np.random.default_rng(33)
    """Created the deterministic design and response generator."""

    x: npt.NDArray[np.float64] = np.column_stack([np.ones(n), rng.standard_normal(n)])
    """Built an intercept-plus-covariate fixed-effect design."""

    record: dict[str, object] = asterism.prepare(x, sibling_relationship(pairs)).fit(
        rng.standard_normal(n)
    )
    """Fitted an independent response against the sibling relationship matrix."""

    for effect in record["fixed_effects"]:
        assert effect["standard_error"] is not None
        assert effect["p_value"] is not None


def test_the_builder_gives_the_textbook_relationships() -> None:
    """Reproduce standard nuclear-family and first-cousin relationships."""
    ids: list[str] = [
        "gm",
        "gf",
        "a",
        "b",
        "a_spouse",
        "b_spouse",
        "a_child",
        "b_child",
    ]
    """Ordered two founders, their children, spouses, and grandchildren."""

    father: list[str | None] = [
        None,
        None,
        "gf",
        "gf",
        None,
        None,
        "a_spouse",
        "b_spouse",
    ]
    """Recorded paternal identifiers parallel to the pedigree rows."""

    mother: list[str | None] = [None, None, "gm", "gm", None, None, "a", "b"]
    """Recorded maternal identifiers parallel to the pedigree rows."""

    k, order = asterism.relationship_matrix(ids, father, mother)
    """Built the textbook twice-kinship matrix and its subject order."""

    def at(one: str, two: str) -> float:
        """Read one named relationship from the constructed matrix."""
        return float(k[order.index(one)][order.index(two)])

    assert at("gm", "gm") == 1.0  # no inbreeding
    assert at("gm", "gf") == 0.0  # the founding couple are unrelated
    assert at("gm", "a") == 0.5  # parent and child
    assert at("a", "b") == 0.5  # full siblings
    assert at("a", "a_spouse") == 0.0  # married in
    assert at("gm", "a_child") == 0.25  # grandparent and grandchild
    assert at("a_child", "b_child") == 0.125  # first cousins


def test_the_builder_keeps_the_order_asked_for_and_uses_unkept_ancestors() -> None:
    """Retain ancestor paths while returning only the requested ordered rows."""
    ids: list[str] = [
        "gm",
        "gf",
        "a",
        "b",
        "a_spouse",
        "b_spouse",
        "a_child",
        "b_child",
    ]
    """Ordered two founders, their children, spouses, and grandchildren."""

    father: list[str | None] = [
        None,
        None,
        "gf",
        "gf",
        None,
        None,
        "a_spouse",
        "b_spouse",
    ]
    """Recorded paternal identifiers parallel to the pedigree rows."""

    mother: list[str | None] = [None, None, "gm", "gm", None, None, "a", "b"]
    """Recorded maternal identifiers parallel to the pedigree rows."""

    keep: list[str] = ["b_child", "a_child"]
    """Requested the cousins in an order different from the pedigree."""

    k, order = asterism.relationship_matrix(ids, father, mother, keep=keep)
    """Built only the requested rows while traversing their omitted ancestors."""

    assert order == keep
    assert k.shape == (2, 2)
    # The cousins' relationship runs through grandparents with no row here.
    assert k[0][1] == 0.125


def test_the_builder_sorts_parents_before_children_itself() -> None:
    """Topologically order an input pedigree supplied child first."""
    k, order = asterism.relationship_matrix(
        ["child", "father", "mother"],
        ["father", None, None],
        ["mother", None, None],
    )
    """Built the matrix from a deliberately non-topological pedigree order."""

    assert k[order.index("child")][order.index("father")] == 0.5


def test_the_builder_refuses_a_pedigree_that_makes_no_sense() -> None:
    """Refuse partial parentage, duplicate identifiers, and ancestry cycles."""
    with pytest.raises(ValueError, match="PEDIGREE_ONE_KNOWN_PARENT"):
        asterism.relationship_matrix(["f", "c"], [None, "f"], [None, None])
    with pytest.raises(ValueError, match="PEDIGREE_DUPLICATE_ID"):
        asterism.relationship_matrix(["a", "a"], [None, None], [None, None])
    with pytest.raises(ValueError, match="PEDIGREE_CYCLE"):
        asterism.relationship_matrix(
            ["m", "a", "b"], [None, "b", "a"], [None, "m", "m"]
        )


def test_a_pedigree_goes_straight_into_a_fit() -> None:
    """Feed a builder matrix directly into the ordinary prepared-model fit."""
    # Pedigree in, matrix out, then an ordinary positional numerical fit.
    pairs: int = 300
    """Selected enough sibling pairs for a stable end-to-end heritability fit."""

    ids: list[str] = []
    """Initialised pedigree identifiers in family order."""

    father: list[str | None] = []
    """Initialised paternal identifiers parallel to the pedigree rows."""

    mother: list[str | None] = []
    """Initialised maternal identifiers parallel to the pedigree rows."""

    for pair in range(pairs):
        ids += [f"f{pair}_dad", f"f{pair}_mum", f"f{pair}_a", f"f{pair}_b"]
        """Added the founders and siblings of the current nuclear family."""

        father += [None, None, f"f{pair}_dad", f"f{pair}_dad"]
        """Recorded the shared father for the current siblings."""

        mother += [None, None, f"f{pair}_mum", f"f{pair}_mum"]
        """Recorded the shared mother for the current siblings."""
    """Constructed unrelated four-person nuclear families."""

    # Only the siblings are measured; the parents still carry the relationship.
    keep: list[str] = [
        identifier for identifier in ids if identifier.endswith(("_a", "_b"))
    ]
    """Selected measured siblings while retaining founders as pedigree ancestors."""

    k, order = asterism.relationship_matrix(ids, father, mother, keep=keep)
    """Built the measured-subject matrix through the complete pedigree."""

    assert order == keep
    assert k.shape == (2 * pairs, 2 * pairs)

    model: asterism.PreparedModel = asterism.prepare(np.ones((len(order), 1)), k)
    """Prepared the numerical model directly from the builder output."""

    record: dict[str, object] = model.fit(simulate(pairs, 0.5, 41))
    """Fitted a deterministic trait with known sibling heritability."""

    assert record["converged"]
    assert 0.0 <= record["h2"] <= 1.0
