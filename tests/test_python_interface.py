"""The Python interface: numerical outputs and invalid-input behaviour."""

from __future__ import annotations

import numpy as np
import pytest

import asterism


def sibling_relationship(pairs: int) -> np.ndarray:
    k = np.eye(2 * pairs)
    for pair in range(pairs):
        a, b = 2 * pair, 2 * pair + 1
        k[a, b] = k[b, a] = 0.5
    return k


def simulate(pairs: int, h2: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(pairs)
    own = rng.standard_normal((pairs, 2))
    genetic = np.sqrt(h2) * (
        np.sqrt(0.5) * shared[:, None] + np.sqrt(0.5) * own
    )
    residual = np.sqrt(1.0 - h2) * rng.standard_normal((pairs, 2))
    return (genetic + residual).reshape(-1)


@pytest.fixture(scope="module")
def model() -> asterism.PreparedModel:
    pairs = 800
    return asterism.prepare(np.ones((2 * pairs, 1)), sibling_relationship(pairs))


def test_the_fit_returns_the_documented_numerical_fields(model):
    record = model.fit(simulate(800, 0.5, 11))
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
        "interval",
        "test",
        "standard_errors",
        "fixed_effects",
    }
    assert record["estimator"] == "reml"
    assert record["n"] == 1600
    assert record["converged"] is True
    assert record["boundary"] in {"interior", "lower", "upper"}
    assert len(record["beta"]) == 1


def test_the_interval_keeps_one_shape_whatever_the_fit_did(model):
    interval = model.fit(simulate(800, 0.5, 12))["interval"]
    assert set(interval) == {
        "lower",
        "upper",
        "lower_limited",
        "upper_limited",
        "contains_lower_bound",
        "contains_upper_bound",
        "level",
        "recipe",
    }
    assert interval["level"] == 0.95
    assert interval["recipe"] == "profile_mixture"
    assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0


def test_the_estimator_is_data_on_the_record_and_both_are_available(model):
    y = simulate(800, 0.5, 13)
    assert model.fit(y, "reml")["estimator"] == "reml"
    assert model.fit(y, "ml")["estimator"] == "ml"
    with pytest.raises(ValueError, match="FIT_ESTIMATOR_INVALID"):
        model.fit(y, "restricted")


def test_a_heritable_trait_is_rejected_against_no_additive_variance(model):
    test = model.fit(simulate(800, 0.7, 14))["test"]
    assert test["null"] == "h2 = 0"
    assert test["rule"] == "mixture_50_50"
    assert test["statistic"] > 0.0
    assert test["p_value"] < 1e-4
    # The mixture puts half the null's mass at zero, so nothing positive can
    # ever be less surprising than a half.
    assert test["p_value"] <= 0.5


def test_a_boundary_fit_withholds_the_standard_error_rather_than_faking_one():
    # No family structure in the response at all, so the estimate goes to zero.
    pairs = 200
    model = asterism.prepare(np.ones((2 * pairs, 1)), sibling_relationship(pairs))
    rng = np.random.default_rng(15)
    record = model.fit(rng.standard_normal(2 * pairs))
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
        (lambda: (np.ones((1, 1)), np.eye(1)), "PREPARE_NO_RESIDUAL_DEGREES_OF_FREEDOM"),
    ],
)
def test_validation_refuses_with_a_stable_code(build, code):
    x, k = build()
    with pytest.raises(ValueError, match=code):
        asterism.prepare(x, k)


def test_an_asymmetric_relationship_matrix_is_refused():
    k = np.eye(10)
    k[0, 1] = 0.5
    with pytest.raises(ValueError, match="PREPARE_K_ASYMMETRIC"):
        asterism.prepare(np.ones((10, 1)), k)


def test_there_is_no_one_shot_fit():
    # Preparing explicitly keeps the costly decomposition reusable.
    assert not hasattr(asterism, "fit")


def test_integer_input_widens_rather_than_being_refused(model):
    # Value-exact conversions happen silently; lossy ones do not happen at all.
    pairs = 20
    k = sibling_relationship(pairs)
    x = np.ones((2 * pairs, 1), dtype=np.int64)
    prepared = asterism.prepare(x, k)
    assert prepared.n == 2 * pairs


def test_every_fixed_effect_carries_wald_inference(model):
    record = model.fit(simulate(800, 0.5, 21))
    effects = record["fixed_effects"]
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


def test_covariates_get_their_own_estimates_and_tests():
    # A real design: an intercept, a covariate that matters and one that does
    # not. The one that matters should be found and the one that does not
    # should not be, which is the whole use `-screen -all` is put to.
    pairs = 600
    n = 2 * pairs
    rng = np.random.default_rng(31)
    real = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    x = np.column_stack([np.ones(n), real, noise])
    y = x @ np.array([1.0, 0.8, 0.0]) + simulate(pairs, 0.4, 32)

    record = asterism.prepare(x, sibling_relationship(pairs)).fit(y)
    intercept, matters, does_not = record["fixed_effects"]

    assert abs(matters["estimate"] - 0.8) < 4 * matters["standard_error"]
    assert matters["p_value"] < 1e-6
    assert does_not["p_value"] > 0.01
    assert abs(intercept["estimate"] - 1.0) < 4 * intercept["standard_error"]


def test_a_boundary_fit_still_reports_its_fixed_effects():
    # h2's standard error is withheld at the bound; the fixed effects' are not.
    # Nothing degenerate happens to a regression coefficient when a variance
    # component pins.
    pairs = 200
    n = 2 * pairs
    rng = np.random.default_rng(33)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    record = asterism.prepare(x, sibling_relationship(pairs)).fit(
        rng.standard_normal(n)
    )
    for effect in record["fixed_effects"]:
        assert effect["standard_error"] is not None
        assert effect["p_value"] is not None


def test_the_builder_gives_the_textbook_relationships():
    ids = ["gm", "gf", "a", "b", "a_spouse", "b_spouse", "a_child", "b_child"]
    father = [None, None, "gf", "gf", None, None, "a_spouse", "b_spouse"]
    mother = [None, None, "gm", "gm", None, None, "a", "b"]
    k, order = asterism.relationship_matrix(ids, father, mother)
    at = lambda one, two: k[order.index(one)][order.index(two)]

    assert at("gm", "gm") == 1.0            # no inbreeding
    assert at("gm", "gf") == 0.0            # the founding couple are unrelated
    assert at("gm", "a") == 0.5             # parent and child
    assert at("a", "b") == 0.5              # full siblings
    assert at("a", "a_spouse") == 0.0       # married in
    assert at("gm", "a_child") == 0.25      # grandparent and grandchild
    assert at("a_child", "b_child") == 0.125  # first cousins


def test_the_builder_keeps_the_order_asked_for_and_uses_unkept_ancestors():
    ids = ["gm", "gf", "a", "b", "a_spouse", "b_spouse", "a_child", "b_child"]
    father = [None, None, "gf", "gf", None, None, "a_spouse", "b_spouse"]
    mother = [None, None, "gm", "gm", None, None, "a", "b"]
    keep = ["b_child", "a_child"]
    k, order = asterism.relationship_matrix(ids, father, mother, keep=keep)
    assert order == keep
    assert k.shape == (2, 2)
    # The cousins' relationship runs through grandparents with no row here.
    assert k[0][1] == 0.125


def test_the_builder_sorts_parents_before_children_itself():
    k, order = asterism.relationship_matrix(
        ["child", "father", "mother"],
        ["father", None, None],
        ["mother", None, None],
    )
    assert k[order.index("child")][order.index("father")] == 0.5


def test_the_builder_refuses_a_pedigree_that_makes_no_sense():
    with pytest.raises(ValueError, match="PEDIGREE_ONE_KNOWN_PARENT"):
        asterism.relationship_matrix(["f", "c"], [None, "f"], [None, None])
    with pytest.raises(ValueError, match="PEDIGREE_DUPLICATE_ID"):
        asterism.relationship_matrix(["a", "a"], [None, None], [None, None])
    with pytest.raises(ValueError, match="PEDIGREE_CYCLE"):
        asterism.relationship_matrix(
            ["m", "a", "b"], [None, "b", "a"], [None, "m", "m"]
        )


def test_a_pedigree_goes_straight_into_a_fit():
    # Pedigree in, matrix out, then an ordinary positional numerical fit.
    pairs = 300
    ids, father, mother = [], [], []
    for pair in range(pairs):
        ids += [f"f{pair}_dad", f"f{pair}_mum", f"f{pair}_a", f"f{pair}_b"]
        father += [None, None, f"f{pair}_dad", f"f{pair}_dad"]
        mother += [None, None, f"f{pair}_mum", f"f{pair}_mum"]

    # Only the siblings are measured; the parents still carry the relationship.
    keep = [i for i in ids if i.endswith(("_a", "_b"))]
    k, order = asterism.relationship_matrix(ids, father, mother, keep=keep)
    assert order == keep
    assert k.shape == (2 * pairs, 2 * pairs)

    model = asterism.prepare(np.ones((len(order), 1)), k)
    record = model.fit(simulate(pairs, 0.5, 41))
    assert record["converged"]
    assert 0.0 <= record["h2"] <= 1.0
