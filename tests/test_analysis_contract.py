"""The public supported-analysis receipt interface."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import asterism
import pytest
from asterism import _core
from asterism.analysis import _release_state as release_state

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
supported_quantities = ["h2", "interval", "test"]
required_checks = ["coverage"]
pass_rules_configured = true

[[analyses.pass_rules]]
id = "coverage"
status = "ready"


[[analyses]]
id = "several_covariance_components"
support = "supported_in_0_1"
entry_points = ["asterism.ComponentModel.fit"]
supported_quantities = [
  "mean_diagonal_component_contributions",
  "mean_diagonal_proportions",
  "mean_diagonal_proportion_interval",
  "coefficients_for_zero_diagonal_bases",
  "contrasts_for_zero_diagonal_bases",
]
supported_quantity_sets = [
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

[[analyses.pass_rules]]
id = "coverage"
status = "ready"

"""
"""A complete fixed release contract used only as the build-metadata adapter."""


DEVELOPMENT_MANIFEST: str = (
    RELEASE_MANIFEST.replace(
        '\ncargo_version = "0.1.0"', '\ncargo_version = "0.1.0-dev.0"'
    )
    .replace('\nversion = "0.1.0"', '\nversion = "0.1.0.dev0"')
    .replace("\nrelease = true", "\nrelease = false")
)
"""The same contract as a development build, so these tests own their own mode.

They used to read whatever the checkout happened to be, which meant the suite
could only pass while the repository was mid-development: cutting a release
turned every "a development build refuses" test red.
"""

BLOCKED_RELEASE_MANIFEST: str = RELEASE_MANIFEST.replace(
    'status = "ready"',
    'status = "blocked"\n'
    'blocker = { code = "coverage_unmeasured", '
    'evidence_needed = "Run and retain the coverage campaign." }',
    1,
)
"""A release-shaped contract whose required evidence is explicitly absent."""


def install_development_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install one internally consistent development-build metadata adapter.

    Args:
        monkeypatch: Pytest adapter used to restore compiled metadata afterwards.
    """
    manifest_sha256: str = hashlib.sha256(
        DEVELOPMENT_MANIFEST.encode("utf-8")
    ).hexdigest()
    """Committed to the exact replacement manifest as the build script would."""

    monkeypatch.setattr(_core, "__release_manifest__", DEVELOPMENT_MANIFEST)
    monkeypatch.setattr(_core, "__release_manifest_sha256__", manifest_sha256)
    monkeypatch.setattr(_core, "__source_dirty__", False)
    """Replaced every coupled field instead of creating an impossible mixed build."""


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


def install_blocked_release_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a release-shaped contract carrying one blocked required rule.

    Args:
        monkeypatch: Pytest adapter used to restore compiled metadata afterwards.
    """
    manifest_sha256: str = hashlib.sha256(
        BLOCKED_RELEASE_MANIFEST.encode("utf-8")
    ).hexdigest()
    """Committed to the exact replacement manifest as the build script would."""

    monkeypatch.setattr(_core, "__release_manifest__", BLOCKED_RELEASE_MANIFEST)
    monkeypatch.setattr(_core, "__release_manifest_sha256__", manifest_sha256)
    monkeypatch.setattr(_core, "__source_dirty__", False)
    """Created a consistent build whose scientific status remains visible."""


def test_a_blocked_required_rule_does_not_gate_runtime_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave scientific completion to release automation once rules exist."""
    install_blocked_release_contract(monkeypatch)
    """Substituted a release-shaped contract with one absent campaign."""

    fitted: bool = False
    """Recorded whether runtime readiness allowed the numerical callback."""

    def fit() -> dict[str, Any]:
        nonlocal fitted
        # asterism-style: allow unannotated-local -- nonlocal test probe rebinding
        fitted = True
        """Recorded that the configured release reached its numerical fit."""
        return {
            "converged": True,
            "h2": 0.5,
            "interval": {"lower": 0.3, "upper": 0.7, "profile_failures": 0},
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
    """Ran through the public receipt interface carrying an unready rule status."""

    assert fitted is True
    assert receipt["outcome"] == "fitted"
    assert receipt["release_state"]["release_ready"] is True
    assert receipt["release_state"]["missing_checks"] == []


@pytest.mark.parametrize(
    ("manifest", "expected_missing"),
    [
        (
            RELEASE_MANIFEST.replace(
                "scientific_pass_rules_configured = true",
                "scientific_pass_rules_configured = false",
                1,
            ),
            {
                "code": "SCIENTIFIC_PASS_RULES_NOT_CONFIGURED",
                "reason": "machine-readable pass rules are incomplete",
            },
        ),
        (
            DEVELOPMENT_MANIFEST,
            {
                "code": "BUILD_NOT_RELEASED",
                "reason": "this is a development build, not a release",
            },
        ),
    ],
)
def test_runtime_receipts_keep_the_two_approved_release_gates(
    monkeypatch: pytest.MonkeyPatch,
    manifest: str,
    expected_missing: dict[str, str],
) -> None:
    """Refuse only an unconfigured contract or a non-release build before fitting."""
    manifest_sha256: str = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    """Committed to the exact synthetic contract installed for this case."""

    monkeypatch.setattr(_core, "__release_manifest__", manifest)
    monkeypatch.setattr(_core, "__release_manifest_sha256__", manifest_sha256)
    monkeypatch.setattr(_core, "__source_dirty__", False)
    """Installed one internally consistent runtime-release contract."""

    fitted: bool = False
    """Recorded whether an approved readiness gate stopped numerical work."""

    def fit() -> dict[str, Any]:
        nonlocal fitted
        # asterism-style: allow unannotated-local -- nonlocal test probe rebinding
        fitted = True
        """Recorded an unexpected fit after an approved readiness failure."""
        return {"converged": True}

    receipt: dict[str, Any] = asterism.run_analysis(
        "one_trait_gaussian_heritability",
        {"sample_size": 350, "trait_type": "continuous"},
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
    """Asked the public receipt interface to enforce its two readiness gates."""

    assert fitted is False
    assert receipt["outcome"] == "refused"
    assert receipt["release_state"]["missing_checks"] == [expected_missing]


def test_an_analysis_introduced_in_0_2_is_active_in_a_0_2_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not hard-code the active support contract to Asterism 0.1."""
    manifest: str = (
        RELEASE_MANIFEST.replace('version = "0.1.0"', 'version = "0.2.0"')
        .replace('cargo_version = "0.1.0"', 'cargo_version = "0.2.0"')
        .replace('support = "supported_in_0_1"', 'support = "supported_in_0_2"')
    )
    """Built the same complete release contract at its next minor version."""

    monkeypatch.setattr(_core, "__release_manifest__", manifest)
    monkeypatch.setattr(
        _core,
        "__release_manifest_sha256__",
        hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
    )
    """Installed internally consistent immutable build metadata."""

    state: dict[str, Any] = release_state(
        "one_trait_gaussian_heritability",
        {"sample_size": 350, "trait_type": "continuous"},
    )
    """Evaluated support introduced on the current release line."""

    assert state["release_ready"] is True
    assert state["missing_checks"] == []


def test_known_limitations_are_carried_into_release_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Put retained negative evidence beside every supported-analysis receipt."""
    limitation: str = (
        'known_limitations = [{ code = "TARGET_LEVEL_FAILED", '
        'quantity = "test", scope = "named_target", '
        'consequence = "withhold_p_value", evidence = "evidence/target.json", '
        f'evidence_sha256 = "{"0" * 64}" }}]\n'
    )
    """Defined one complete synthetic limitation record."""
    manifest: str = RELEASE_MANIFEST.replace(
        'supported_quantities = ["h2", "interval", "test"]\n',
        'supported_quantities = ["h2", "interval", "test"]\n' + limitation,
        1,
    )
    """Added one participant-free limitation to an otherwise ready contract."""

    monkeypatch.setattr(_core, "__release_manifest__", manifest)
    monkeypatch.setattr(
        _core,
        "__release_manifest_sha256__",
        hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
    )
    """Installed the limitation-bearing manifest with matching identity."""

    state: dict[str, Any] = release_state(
        "one_trait_gaussian_heritability",
        {
            "sample_size": 350,
            "trait_type": "continuous",
            "expected_censoring_share": 0.75,
        },
    )
    """Read the limitation without turning design metadata into a refusal gate."""

    assert state["release_ready"] is True
    assert state["supported_quantities"] == ["h2", "interval", "test"]
    assert state["known_limitations"] == [
        {
            "code": "TARGET_LEVEL_FAILED",
            "quantity": "test",
            "scope": "named_target",
            "consequence": "withhold_p_value",
            "evidence": "evidence/target.json",
            "evidence_sha256": "0" * 64,
        }
    ]


def test_a_development_build_is_never_release_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a development build from passing its checks, whatever the design."""
    install_development_contract(monkeypatch)
    """Substituted a development build as the build-metadata adapter."""

    design: dict[str, Any] = {
        "sample_size": 350,
        "largest_family": 14,
        "trait_type": "continuous",
        "components": "additive_relationship",
    }
    """Described an entirely ordinary Gaussian pedigree design."""

    receipt: dict[str, Any] = asterism.run_analysis(
        "one_trait_gaussian_heritability",
        design,
        lambda: {"converged": True, "h2": 0.5, "interval": [0.1, 0.9], "test": 0.01},
        model={"estimator": "reml"},
        provenance={
            "wheel_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "consumer_commit": "c" * 40,
            "input_commitments": {},
        },
        subject_order=["s1", "s2"],
    )
    """Asked the standard runner for a receipt from this development build."""

    assert receipt["outcome"] == "refused"
    assert receipt["refusal"]["missing_checks"] == [
        {
            "code": "BUILD_NOT_RELEASED",
            "reason": "this is a development build, not a release",
        }
    ]
    """Refused for the one reason that still applies, and named it."""


def test_an_unreleased_build_refuses_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreleased build never evaluated a real-outcome fit callback."""
    install_development_contract(monkeypatch)
    """Substituted a development build as the build-metadata adapter."""

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
        receipt["refusal"]["missing_checks"]
        == receipt["release_state"]["missing_checks"]
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


def test_a_converged_finite_release_fit_meets_every_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measured release fit crossed the public runner meeting every check."""
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

    assert receipt["outcome"] == "fitted"
    assert receipt["facts"] == {
        "converged": True,
        "all_quantities_present": True,
        "all_values_finite": True,
        "all_profiles_evaluated": True,
    }
    assert receipt["refusal"] is None
    assert receipt["fit_record"]["h2"] == 0.5
    assert receipt["fit_record"]["build"] == receipt["build"]
    assert (
        receipt["fit_record"]["subject_order_sha256"]
        == (receipt["provenance"]["input_commitments"]["subject_order_sha256"])
    )
    assert json.loads(json.dumps(receipt, allow_nan=False))["outcome"] == "fitted"


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
def test_a_receipt_selects_only_the_applicable_quantity_set(
    monkeypatch: pytest.MonkeyPatch,
    component_reporting: str,
    expected_quantities: list[str],
) -> None:
    """Mutually exclusive component reporting branches never required each other."""
    install_release_contract(monkeypatch)
    """Substituted a release manifest carrying both component report branches."""

    state: dict[str, Any] = release_state(
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

    assert state["release_ready"] is True
    assert state["supported_quantities"] == expected_quantities
    assert set(state["supported_quantity_inventory"]) == {
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

    assert receipt["outcome"] == "fitted"
    assert receipt["failures"] == []
    assert all(receipt["facts"].values())
    assert "coefficients_for_zero_diagonal_bases" not in receipt["fit_record"]
    assert "contrasts_for_zero_diagonal_bases" not in receipt["fit_record"]


@pytest.mark.parametrize(
    ("converged", "profile_failures", "expected_code"),
    [
        (False, 0, "FIT_NOT_CONVERGED"),
        (True, 1, "PROFILE_EVALUATION_FAILED"),
    ],
)
def test_an_unresolved_fit_records_which_check_it_failed(
    monkeypatch: pytest.MonkeyPatch,
    converged: bool,
    profile_failures: int,
    expected_code: str,
) -> None:
    """A finite estimate arrives beside the fact that a check did not hold."""
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

    assert receipt["outcome"] == "fitted"
    assert receipt["fit_record"]["h2"] == 0.5
    assert expected_code in {failure["code"] for failure in receipt["failures"]}
    assert not all(receipt["facts"].values())


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
