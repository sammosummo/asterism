"""The public supported-analysis preflight and receipt interface."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import asterism
import pytest
from asterism import _core

RELEASE_MANIFEST: str = """
schema_version = 1
version = "0.1.0"
cargo_version = "0.1.0"
release = true
requires_python = ">=3.13,<3.15"
scientific_pass_rules_configured = true

[[analyses]]
id = "one_trait_gaussian_heritability"
support = "supported_in_0_1"
entry_points = ["asterism.prepare", "asterism.PreparedModel.fit"]
reportable_quantities = ["h2", "interval", "test"]
required_checks = ["coverage"]
pass_rules_configured = true

[analyses.design_range]
measured = true

[analyses.design_range.sample_size]
min = 350
max = 1400

[analyses.design_range.largest_family]
max = 14

[analyses.design_range.trait_type]
allowed = ["continuous"]

[analyses.design_range.components]
allowed = ["additive_relationship"]

[[analyses]]
id = "several_covariance_components"
support = "supported_in_0_1"
entry_points = ["asterism.ComponentModel.fit"]
reportable_quantities = [
  "mean_diagonal_component_contributions",
  "mean_diagonal_proportions",
  "mean_diagonal_proportion_interval",
  "coefficients_for_zero_diagonal_bases",
  "contrasts_for_zero_diagonal_bases",
]
reportable_quantity_sets = [
  { when = { component_reporting = "mean_diagonal" }, quantities = [
    "mean_diagonal_component_contributions",
    "mean_diagonal_proportions",
    "mean_diagonal_proportion_interval",
  ] },
  { when = { component_reporting = "zero_diagonal" }, quantities = [
    "coefficients_for_zero_diagonal_bases",
    "contrasts_for_zero_diagonal_bases",
  ] },
]
required_checks = ["coverage"]
pass_rules_configured = true

[analyses.design_range]
measured = true

[analyses.design_range.sample_size]
min = 350
max = 1400

[analyses.design_range.largest_family]
max = 14

[analyses.design_range.trait_type]
allowed = ["continuous"]

[analyses.design_range.components]
allowed = ["relationship_matrices"]

[analyses.design_range.component_reporting]
allowed = ["mean_diagonal", "zero_diagonal"]
"""
"""A complete fixed release contract used only as the build-metadata adapter."""


def install_release_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install one internally consistent fixed-release metadata adapter.

    Args:
        monkeypatch: Pytest adapter used to restore compiled metadata afterwards.
    """
    manifest_sha256: str = hashlib.sha256(RELEASE_MANIFEST.encode("utf-8")).hexdigest()
    """Committed to the exact replacement manifest as the build script would."""

    monkeypatch.setattr(_core, "__release_manifest__", RELEASE_MANIFEST)
    monkeypatch.setattr(_core, "__release_manifest_sha256__", manifest_sha256)
    monkeypatch.setattr(_core, "__source_dirty__", False)
    """Replaced every coupled field instead of creating an impossible mixed build."""


def test_unmeasured_release_evidence_blocks_reportable_preflight() -> None:
    """A development build stays unreportable even inside the measured range."""
    design: dict[str, Any] = {
        "sample_size": 350,
        "largest_family": 14,
        "trait_type": "continuous",
        "components": "additive_relationship",
    }
    """Described a Gaussian pedigree design inside the measured release range."""

    preflight: dict[str, Any] = asterism.preflight_analysis(
        "one_trait_gaussian_heritability",
        design,
    )
    """Compared the proposed design with this build's release evidence."""

    assert preflight["analysis"] == "one_trait_gaussian_heritability"
    assert preflight["inside_supported_range"] is False
    assert [check["code"] for check in preflight["missing_checks"]] == [
        "SCIENTIFIC_PASS_RULES_NOT_CONFIGURED",
        "BUILD_NOT_RELEASED",
    ]
    assert preflight["reportable_quantities"] == ["h2", "interval", "test"]


def test_a_failed_preflight_refuses_before_fitting() -> None:
    """An unreleased build never evaluated a real-outcome fit callback."""
    fitted: bool = False
    """Recorded whether the numerical callback was invoked."""

    def fit() -> dict[str, Any]:
        nonlocal fitted
        # asterism-style: allow unannotated-local -- nonlocal test probe rebinding
        fitted = True
        """Recorded that the numerical callback was unexpectedly invoked."""
        return {"converged": True, "h2": 0.5}

    design: dict[str, Any] = {
        "sample_size": 350,
        "largest_family": 14,
        "trait_type": "continuous",
        "components": "additive_relationship",
    }
    """Described the proposed one-trait Gaussian analysis."""
    provenance: dict[str, Any] = {
        "wheel_sha256": "a" * 64,
        "dependency_lock_sha256": "b" * 64,
        "consumer_commit": "c" * 40,
        "input_commitments": {},
    }
    """Identified the caller-owned artifact, code and input commitments."""

    receipt: dict[str, Any] = asterism.run_analysis(
        "one_trait_gaussian_heritability",
        design,
        fit,
        model={"estimator": "reml"},
        provenance=provenance,
        subject_order=["s1", "s2"],
    )
    """Asked the standard runner to preflight and fit the proposed analysis."""

    assert fitted is False
    assert receipt["outcome"] == "refused"
    assert receipt["fit_record"] is None
    assert receipt["refusal"]["code"] == "ANALYSIS_PREFLIGHT_FAILED"
    assert (
        receipt["refusal"]["missing_checks"] == receipt["preflight"]["missing_checks"]
    )
    assert receipt["build"]["version"] == "0.1.0.dev0"
    assert receipt["build"]["cargo_version"] == "0.1.0-dev.0"
    assert receipt["build"]["release"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", receipt["build"]["source_commit"])
    assert isinstance(receipt["build"]["source_dirty"], bool)
    assert (
        receipt["provenance"]["input_commitments"]["subject_order_sha256"]
        == "1695f73d72a331875615055be1621ba63c46e9e3f556c429de2cd67f1d1f3654"
    )
    assert "s1" not in repr(receipt)
    assert "s2" not in repr(receipt)


def test_a_converged_finite_release_fit_is_reportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measured release fit crossed the public runner as reportable."""
    install_release_contract(monkeypatch)
    """Substituted the immutable build-metadata adapter with a complete release."""

    def fit() -> dict[str, Any]:
        return {
            "converged": True,
            "h2": 0.5,
            "interval": {
                "lower": 0.3,
                "upper": 0.7,
                "profile_failures": 0,
            },
            "test": {"p_value": 0.01},
        }

    receipt: dict[str, Any] = asterism.run_analysis(
        "one_trait_gaussian_heritability",
        {
            "sample_size": 350,
            "largest_family": 14,
            "trait_type": "continuous",
            "components": "additive_relationship",
        },
        fit,
        model={"estimator": "reml"},
        provenance={
            "wheel_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "consumer_commit": "c" * 40,
            "input_commitments": {},
        },
        subject_order=["s1", "s2"],
    )
    """Ran a complete fit through a measured fixed-release contract."""

    assert receipt["outcome"] == "reportable"
    assert receipt["refusal"] is None
    assert receipt["fit_record"]["h2"] == 0.5
    assert receipt["fit_record"]["build"] == receipt["build"]
    assert (
        receipt["fit_record"]["subject_order_sha256"]
        == (receipt["provenance"]["input_commitments"]["subject_order_sha256"])
    )
    assert json.loads(json.dumps(receipt, allow_nan=False))["outcome"] == "reportable"


@pytest.mark.parametrize(
    ("component_reporting", "expected_quantities"),
    [
        (
            "mean_diagonal",
            [
                "mean_diagonal_component_contributions",
                "mean_diagonal_proportions",
                "mean_diagonal_proportion_interval",
            ],
        ),
        (
            "zero_diagonal",
            [
                "coefficients_for_zero_diagonal_bases",
                "contrasts_for_zero_diagonal_bases",
            ],
        ),
    ],
)
def test_component_preflight_selects_only_the_applicable_quantity_set(
    monkeypatch: pytest.MonkeyPatch,
    component_reporting: str,
    expected_quantities: list[str],
) -> None:
    """Mutually exclusive component reporting branches never required each other."""
    install_release_contract(monkeypatch)
    """Substituted a release manifest carrying both component report branches."""

    preflight: dict[str, Any] = asterism.preflight_analysis(
        "several_covariance_components",
        {
            "sample_size": 350,
            "largest_family": 14,
            "trait_type": "continuous",
            "components": "relationship_matrices",
            "component_reporting": component_reporting,
        },
    )
    """Selected the report branch from the non-identifying component design."""

    assert preflight["inside_supported_range"] is True
    assert preflight["reportable_quantities"] == expected_quantities
    assert set(preflight["reportable_quantity_inventory"]) == {
        "mean_diagonal_component_contributions",
        "mean_diagonal_proportions",
        "mean_diagonal_proportion_interval",
        "coefficients_for_zero_diagonal_bases",
        "contrasts_for_zero_diagonal_bases",
    }


def test_component_receipt_requires_only_its_selected_report_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mean-diagonal receipt did not invent zero-diagonal contrast results."""
    install_release_contract(monkeypatch)
    """Substituted the complete clean component release contract."""

    receipt: dict[str, Any] = asterism.run_analysis(
        "several_covariance_components",
        {
            "sample_size": 350,
            "largest_family": 14,
            "trait_type": "continuous",
            "components": "relationship_matrices",
            "component_reporting": "mean_diagonal",
        },
        lambda: {
            "converged": True,
            "mean_diagonal_component_contributions": [0.4, 0.6],
            "mean_diagonal_proportions": [0.4, 0.6],
            "mean_diagonal_proportion_interval": {
                "estimate": 0.4,
                "lower": 0.2,
                "upper": 0.7,
                "profile_failures": 0,
            },
        },
        model={"estimator": "reml", "component_reporting": "mean_diagonal"},
        provenance={
            "wheel_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "consumer_commit": "c" * 40,
            "input_commitments": {},
        },
        subject_order=["s1", "s2"],
    )
    """Ran the positive-mean-diagonal branch through the standard receipt path."""

    assert receipt["outcome"] == "reportable"
    assert receipt["reportability_issues"] == []
    assert "coefficients_for_zero_diagonal_bases" not in receipt["fit_record"]
    assert "contrasts_for_zero_diagonal_bases" not in receipt["fit_record"]


@pytest.mark.parametrize(
    ("converged", "profile_failures", "expected_code"),
    [
        (False, 0, "FIT_NOT_CONVERGED"),
        (True, 1, "PROFILE_EVALUATION_FAILED"),
    ],
)
def test_an_unresolved_fit_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch,
    converged: bool,
    profile_failures: int,
    expected_code: str,
) -> None:
    """A finite candidate never became reportable by being inspectable."""
    install_release_contract(monkeypatch)
    """Substituted a complete clean release as the build-metadata adapter."""

    receipt: dict[str, Any] = asterism.run_analysis(
        "one_trait_gaussian_heritability",
        {
            "sample_size": 350,
            "largest_family": 14,
            "trait_type": "continuous",
            "components": "additive_relationship",
        },
        lambda: {
            "converged": converged,
            "h2": 0.5,
            "interval": {
                "lower": 0.3,
                "upper": 0.7,
                "profile_failures": profile_failures,
            },
            "test": {"p_value": 0.01},
        },
        model={"estimator": "reml"},
        provenance={
            "wheel_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "consumer_commit": "c" * 40,
            "input_commitments": {},
        },
        subject_order=["s1", "s2"],
    )
    """Ran the unresolved candidate through the standard analysis path."""

    assert receipt["outcome"] == "diagnostic-only"
    assert receipt["fit_record"]["h2"] == 0.5
    assert expected_code in {issue["code"] for issue in receipt["reportability_issues"]}


def test_a_stable_estimator_value_error_becomes_a_refused_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A documented estimator refusal crossed the wrapper without a traceback."""
    install_release_contract(monkeypatch)
    """Substituted a complete clean release as the build-metadata adapter."""

    def fit() -> dict[str, Any]:
        raise ValueError("PREPARE_RESPONSE_CONSTANT")

    receipt: dict[str, Any] = asterism.run_analysis(
        "one_trait_gaussian_heritability",
        {
            "sample_size": 350,
            "largest_family": 14,
            "trait_type": "continuous",
            "components": "additive_relationship",
        },
        fit,
        model={"estimator": "reml"},
        provenance={
            "wheel_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "consumer_commit": "c" * 40,
            "input_commitments": {},
        },
        subject_order=["s1", "s2"],
    )
    """Ran a stable Asterism refusal through the standard analysis path."""

    assert receipt["outcome"] == "refused"
    assert receipt["fit_record"] is None
    assert receipt["refusal"] == {
        "code": "PREPARE_RESPONSE_CONSTANT",
        "message": "PREPARE_RESPONSE_CONSTANT",
    }


def test_an_unexpected_fit_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper never mislabelled a programming failure as scientific refusal."""
    install_release_contract(monkeypatch)
    """Substituted a complete clean release as the build-metadata adapter."""

    def fit() -> dict[str, Any]:
        raise RuntimeError("unexpected implementation fault")

    with pytest.raises(RuntimeError, match="unexpected implementation fault"):
        asterism.run_analysis(
            "one_trait_gaussian_heritability",
            {
                "sample_size": 350,
                "largest_family": 14,
                "trait_type": "continuous",
                "components": "additive_relationship",
            },
            fit,
            model={"estimator": "reml"},
            provenance={
                "wheel_sha256": "a" * 64,
                "dependency_lock_sha256": "b" * 64,
                "consumer_commit": "c" * 40,
                "input_commitments": {},
            },
            subject_order=["s1", "s2"],
        )
    """Required an unexpected exception to escape the receipt wrapper."""


def test_an_unknown_uppercase_value_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uppercase user-code failure was not mistaken for an estimator refusal."""
    install_release_contract(monkeypatch)
    """Substituted a complete clean release as the build-metadata adapter."""

    def fit() -> dict[str, Any]:
        raise ValueError("CALLER_PIPELINE_BROKEN")

    with pytest.raises(ValueError, match="CALLER_PIPELINE_BROKEN"):
        asterism.run_analysis(
            "one_trait_gaussian_heritability",
            {
                "sample_size": 350,
                "largest_family": 14,
                "trait_type": "continuous",
                "components": "additive_relationship",
            },
            fit,
            model={"estimator": "reml"},
            provenance={
                "wheel_sha256": "a" * 64,
                "dependency_lock_sha256": "b" * 64,
                "consumer_commit": "c" * 40,
                "input_commitments": {},
            },
            subject_order=["s1", "s2"],
        )
    """Required an undocumented caller failure to escape the receipt wrapper."""


@pytest.mark.parametrize(
    ("identity_field", "identity_value", "expected_code"),
    [
        (
            "build",
            {
                "version": "0.0.0",
                "cargo_version": "0.0.0",
                "source_commit": "0" * 40,
                "source_dirty": False,
                "release": True,
            },
            "ANALYSIS_FIT_BUILD_IDENTITY_MISMATCH",
        ),
        (
            "subject_order_sha256",
            "0" * 64,
            "ANALYSIS_FIT_SUBJECT_ORDER_MISMATCH",
        ),
    ],
)
def test_runner_refuses_to_overwrite_conflicting_fit_identity(
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
    identity_value: Any,
    expected_code: str,
) -> None:
    """An existing fit identity was verified rather than silently replaced."""
    install_release_contract(monkeypatch)
    """Substituted a complete clean release as the build-metadata adapter."""
    fit_record: dict[str, Any] = {
        "converged": True,
        "h2": 0.5,
        "interval": {
            "lower": 0.3,
            "upper": 0.7,
            "profile_failures": 0,
        },
        "test": {"p_value": 0.01},
        identity_field: identity_value,
    }
    """Constructed a candidate carrying a conflicting immutable identity."""

    with pytest.raises(ValueError, match=expected_code):
        asterism.run_analysis(
            "one_trait_gaussian_heritability",
            {
                "sample_size": 350,
                "largest_family": 14,
                "trait_type": "continuous",
                "components": "additive_relationship",
            },
            lambda: fit_record,
            model={"estimator": "reml"},
            provenance={
                "wheel_sha256": "a" * 64,
                "dependency_lock_sha256": "b" * 64,
                "consumer_commit": "c" * 40,
                "input_commitments": {},
            },
            subject_order=["s1", "s2"],
        )
    """Required the wrapper to refuse a cross-build or cross-order record."""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wheel_sha256", None),
        ("wheel_sha256", "A" * 64),
        ("dependency_lock_sha256", "b" * 63),
        ("consumer_commit", "not-a-commit"),
    ],
)
def test_runner_requires_exact_caller_provenance_commitments(
    field: str,
    value: Any,
) -> None:
    """A receipt never claimed reproducibility from absent or malformed identity."""
    provenance: dict[str, Any] = {
        "wheel_sha256": "a" * 64,
        "dependency_lock_sha256": "b" * 64,
        "consumer_commit": "c" * 40,
        "input_commitments": {},
    }
    """Constructed the minimum caller-owned provenance contract."""
    if value is None:
        del provenance[field]
    else:
        provenance[field] = value
        """Replaced the valid commitment with a malformed value."""
    """Introduced the one malformed provenance field under test."""

    with pytest.raises(ValueError, match=f"ANALYSIS_PROVENANCE_INVALID:{field}"):
        asterism.run_analysis(
            "one_trait_gaussian_heritability",
            {
                "sample_size": 350,
                "largest_family": 14,
                "trait_type": "continuous",
                "components": "additive_relationship",
            },
            lambda: {"converged": True},
            model={"estimator": "reml"},
            provenance=provenance,
            subject_order=["s1", "s2"],
        )
    """Required provenance validation before any scientific fit could run."""


@pytest.mark.parametrize(
    "input_commitments",
    [
        {"pedigree": "not-a-sha256"},
        {"pedigree": "A" * 64},
        {"": "d" * 64},
        {1: "d" * 64},
    ],
)
def test_runner_requires_values_free_sha256_input_commitments(
    input_commitments: dict[Any, Any],
) -> None:
    """Input provenance could not carry unhashed descriptions or invalid labels."""
    provenance: dict[str, Any] = {
        "wheel_sha256": "a" * 64,
        "dependency_lock_sha256": "b" * 64,
        "consumer_commit": "c" * 40,
        "input_commitments": input_commitments,
    }
    """Constructed otherwise valid provenance with one malformed input mapping."""

    with pytest.raises(ValueError, match="ANALYSIS_INPUT_COMMITMENTS_INVALID"):
        asterism.run_analysis(
            "one_trait_gaussian_heritability",
            {
                "sample_size": 350,
                "largest_family": 14,
                "trait_type": "continuous",
                "components": "additive_relationship",
            },
            lambda: {"converged": True},
            model={"estimator": "reml"},
            provenance=provenance,
            subject_order=["s1", "s2"],
        )
    """Required every named input commitment to be a lowercase SHA-256 digest."""
