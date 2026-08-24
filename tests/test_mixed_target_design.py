"""Contracts for the mixed binary/censored target-design release check."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import chi2

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the standalone scientific command."""

TARGET_FIXTURE: Path = (
    ROOT / "checks" / "design_fixtures" / "mixed_binary_censored_target_design.json"
)
"""Located the reviewed values-free mixed-model target fixture."""

TARGET_FIXTURE_SHA256: str = (
    "aed92270dcfafb00a46a24bfaf1f3a2b52950fdea1c34deb3740e5060d6f3645"
)
"""Pinned the exact reviewed fixture bytes independently of their contents."""


def target_module() -> ModuleType:
    """Load the target command without invoking its command-line entry point."""
    path: Path = ROOT / "checks" / "mixed_binary_censored_target_design.py"
    """Located the maintained standalone target command."""

    specification: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location(
            "mixed_binary_censored_target_design",
            path,
        )
    )
    """Created the import specification for the standalone command."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the module governed by the validated specification."""

    sys.modules[specification.name] = module
    """Registered the module before evaluating public dataclass annotations."""

    checks_directory: str = str(path.parent)
    """Located sibling check modules imported by the standalone command."""

    sys.path.insert(0, checks_directory)
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(checks_directory)
    """Loaded declarations with the same sibling path available to direct CLI use."""
    return module


def test_reviewed_fixture_builds_only_the_target_structural_envelope() -> None:
    """Stress intended family sizes without reconstructing participant data."""
    module: ModuleType = target_module()
    """Loaded the exact-byte target-fixture verifier."""

    assert hashlib.sha256(TARGET_FIXTURE.read_bytes()).hexdigest() == (
        TARGET_FIXTURE_SHA256
    )
    assert module.TARGET_FIXTURE_SHA256 == TARGET_FIXTURE_SHA256
    """Bound both test and command to the independently pinned fixture digest."""

    envelope: object = module.load_target_design(
        TARGET_FIXTURE,
        expected_sha256=module.TARGET_FIXTURE_SHA256,
    )
    """Built synthetic target arrays only after exact fixture verification."""

    assert envelope.relationship.shape == (1_909, 1_909)
    assert envelope.design.shape == (1_909, 6)
    assert len(envelope.component_sizes) == 202
    assert sum(envelope.component_sizes) == 1_909
    assert max(envelope.component_sizes) == 180
    assert envelope.generated_nonzero_relationship_pairs == 27_691
    assert envelope.observed_nonzero_relationship_pairs == 27_821
    assert envelope.censoring_shares == (0.52, 0.75)
    assert envelope.genetic_correlations == (0.0, 0.4)
    assert np.linalg.matrix_rank(envelope.design) == 6


def test_replicate_uses_only_the_three_public_mixed_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise fit, profile, and test through documented Python functions."""
    module: ModuleType = target_module()
    """Loaded the worker boundary independently of process orchestration."""

    calls: list[str] = []
    """Recorded every public operation in invocation order."""

    first: dict[str, object] = {
        "kind": "binary",
        "value": np.full(4, np.nan),
        "censoring": np.asarray([1, 2, 1, 2]),
        "limit": np.zeros(4),
    }
    """Encoded a balanced synthetic binary liability trait."""

    second: dict[str, object] = {
        "kind": "censored",
        "value": np.asarray([0.2, np.nan, -0.1, np.nan]),
        "censoring": np.asarray([0, 1, 0, 1]),
        "limit": np.ones(4),
    }
    """Encoded a synthetic right-censored hearing trait with measured scale."""

    def fixed_traits(
        state: object,
        job: object,
    ) -> tuple[dict[str, object], dict[str, object], int, int]:
        """Return fixed valid traits while retaining the production signature."""
        del state, job
        return first, second, 2, 2

    monkeypatch.setattr(module, "simulate_traits", fixed_traits)
    """Held outcome generation fixed so this test isolates the public seam."""

    def fit(
        relationship: npt.NDArray[np.float64],
        first_trait: dict[str, object],
        second_trait: dict[str, object],
        design: npt.NDArray[np.float64],
    ) -> dict[str, object]:
        """Return a converged minimal public fit record."""
        assert relationship.shape == (4, 4)
        assert first_trait is first
        assert second_trait is second
        assert design.shape == (4, 1)
        calls.append("fit")
        return {
            "converged": True,
            "genetic_correlation": 0.35,
            "loglik": -11.0,
            "estimator": "ml",
            "kinds": ["binary", "censored"],
            "largest_family": 4,
        }

    def interval(
        relationship: npt.NDArray[np.float64],
        first_trait: dict[str, object],
        second_trait: dict[str, object],
        design: npt.NDArray[np.float64],
        coordinate: str,
    ) -> dict[str, object]:
        """Return a complete public genetic-correlation interval."""
        del relationship, first_trait, second_trait, design
        assert coordinate == "genetic_correlation"
        calls.append("interval")
        return {
            "estimate": 0.35,
            "lower": 0.1,
            "upper": 0.6,
            "lower_limited": False,
            "upper_limited": False,
            "level": 0.95,
            "profile_failures": 0,
            "what": "genetic_correlation",
            "estimator": "ml",
        }

    def test(
        relationship: npt.NDArray[np.float64],
        first_trait: dict[str, object],
        second_trait: dict[str, object],
        design: npt.NDArray[np.float64],
        coordinate: str,
    ) -> dict[str, object]:
        """Return a complete public interior correlation test."""
        del relationship, first_trait, second_trait, design
        assert coordinate == "genetic_correlation"
        calls.append("test")
        return {
            "what": "genetic_correlation",
            "statistic": 5e-7,
            "p_value": 1.0,
            "rule": "chi2_1",
            "null_loglik": -11.00000025,
            "alternative_loglik": -11.0,
            "estimator": "ml",
        }

    monkeypatch.setattr(module.asterism, "mixed_bivariate_fit", fit)
    monkeypatch.setattr(module.asterism, "mixed_bivariate_interval", interval)
    monkeypatch.setattr(module.asterism, "mixed_bivariate_test", test)
    """Replaced only the three documented public calculations."""

    state: object = module.WorkerState(
        relationship=np.eye(4),
        design=np.ones((4, 1)),
        component_sizes=(4,),
        genetic_factors=(np.eye(4),),
        prevalence=0.25,
        heritabilities=(0.5, 0.5),
        residual_correlation=0.15,
        hearing_variance=3.0,
        minimum_cases=1,
        minimum_measured_hearing=1,
        nominal_coverage=0.95,
        required_test_rule="chi2_1",
        required_interval_level=0.95,
    )
    """Built the smallest deterministic state satisfying the public model."""

    module.start_worker(state)
    """Installed state through the same initialiser used by campaign workers."""

    result: object = module.run_replicate(
        module.ReplicateJob(
            genetic_correlation=0.4,
            censoring_share=0.52,
            replicate=0,
            seed=1,
        )
    )
    """Ran one complete attempt through all three public operations."""

    assert result.outcome == "complete"
    assert result.covered is True
    assert result.p_value == 1.0
    assert result.test_rule == "chi2_1"
    assert result.profile_failures == 0
    assert calls == ["fit", "interval", "test"]


def test_replicate_refuses_incomplete_public_inference_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require identity and likelihood fields, not only plausible numbers."""
    module: ModuleType = target_module()
    """Loaded the public-record classification boundary."""

    first: dict[str, object] = {
        "kind": "binary",
        "value": np.full(4, np.nan),
        "censoring": np.asarray([1, 2, 1, 2]),
        "limit": np.zeros(4),
    }
    """Encoded a valid balanced binary trait."""

    second: dict[str, object] = {
        "kind": "censored",
        "value": np.asarray([0.2, np.nan, -0.1, np.nan]),
        "censoring": np.asarray([0, 1, 0, 1]),
        "limit": np.ones(4),
    }
    """Encoded a valid censored trait with measured scale."""

    def fixed_traits(
        state: object,
        job: object,
    ) -> tuple[dict[str, object], dict[str, object], int, int]:
        """Return fixed valid traits through the production generator seam."""
        del state, job
        return first, second, 2, 2

    def fit(
        relationship: object,
        first_trait: object,
        second_trait: object,
        design: object,
    ) -> dict[str, object]:
        """Return a structurally complete converged public fit."""
        del relationship, first_trait, second_trait, design
        return {
            "converged": True,
            "genetic_correlation": 0.4,
            "loglik": -11.0,
            "estimator": "ml",
            "kinds": ["binary", "censored"],
            "largest_family": 4,
        }

    def interval_without_identity(
        relationship: object,
        first_trait: object,
        second_trait: object,
        design: object,
        coordinate: str,
    ) -> dict[str, object]:
        """Omit the required quantity name from an otherwise plausible profile."""
        del relationship, first_trait, second_trait, design
        assert coordinate == "genetic_correlation"
        return {
            "estimate": 0.4,
            "lower": 0.2,
            "upper": 0.6,
            "lower_limited": False,
            "upper_limited": False,
            "level": 0.95,
            "profile_failures": 0,
            "estimator": "ml",
        }

    def complete_test(
        relationship: object,
        first_trait: object,
        second_trait: object,
        design: object,
        coordinate: str,
    ) -> dict[str, object]:
        """Return a complete public test that the invalid interval must pre-empt."""
        del relationship, first_trait, second_trait, design
        assert coordinate == "genetic_correlation"
        return {
            "what": "genetic_correlation",
            "statistic": 1.0,
            "p_value": 0.3,
            "rule": "chi2_1",
            "null_loglik": -11.5,
            "alternative_loglik": -11.0,
            "estimator": "ml",
        }

    monkeypatch.setattr(module, "simulate_traits", fixed_traits)
    monkeypatch.setattr(module.asterism, "mixed_bivariate_fit", fit)
    monkeypatch.setattr(
        module.asterism,
        "mixed_bivariate_interval",
        interval_without_identity,
    )
    monkeypatch.setattr(module.asterism, "mixed_bivariate_test", complete_test)
    """Controlled only documented public responses and participant-free generation."""

    state: object = module.WorkerState(
        relationship=np.eye(4),
        design=np.ones((4, 1)),
        component_sizes=(4,),
        genetic_factors=(np.eye(4),),
        prevalence=0.25,
        heritabilities=(0.5, 0.5),
        residual_correlation=0.15,
        hearing_variance=3.0,
        minimum_cases=1,
        minimum_measured_hearing=1,
        nominal_coverage=0.95,
        required_test_rule="chi2_1",
        required_interval_level=0.95,
    )
    """Built one valid state so only public-record identity could fail."""

    module.start_worker(state)
    result: object = module.run_replicate(
        module.ReplicateJob(
            genetic_correlation=0.4,
            censoring_share=0.52,
            replicate=0,
            seed=1,
        )
    )
    """Ran the incomplete interval through fail-closed classification."""

    assert result.outcome == "interval_failed"
    assert result.covered is False
    assert result.error_code == "MIXED_TARGET_PUBLIC_INTERVAL_RECORD_INVALID"


@pytest.mark.parametrize(
    ("malformation", "expected_outcome"),
    (
        ("interval_estimate", "interval_failed"),
        ("alternative_likelihood", "test_failed"),
        ("statistic", "test_failed"),
        ("p_value", "test_failed"),
    ),
)
def test_replicate_refuses_cross_run_inference_records(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    expected_outcome: str,
) -> None:
    """Bind profiles and tests to the same public free fit and likelihood ratio."""
    module: ModuleType = target_module()
    """Loaded the public-record cross-validation boundary."""

    first: dict[str, object] = {
        "kind": "binary",
        "value": np.full(4, np.nan),
        "censoring": np.asarray([1, 2, 1, 2]),
        "limit": np.zeros(4),
    }
    """Encoded a valid balanced binary trait."""

    second: dict[str, object] = {
        "kind": "censored",
        "value": np.asarray([0.2, np.nan, -0.1, np.nan]),
        "censoring": np.asarray([0, 1, 0, 1]),
        "limit": np.ones(4),
    }
    """Encoded a valid censored trait with measured scale."""

    def fixed_traits(
        state: object,
        job: object,
    ) -> tuple[dict[str, object], dict[str, object], int, int]:
        """Return fixed valid traits through the production generator seam."""
        del state, job
        return first, second, 2, 2

    def fit(
        relationship: object,
        first_trait: object,
        second_trait: object,
        design: object,
    ) -> dict[str, object]:
        """Return the free-fit identity against which inference must agree."""
        del relationship, first_trait, second_trait, design
        return {
            "converged": True,
            "genetic_correlation": 0.4,
            "loglik": -11.0,
            "estimator": "ml",
            "kinds": ["binary", "censored"],
            "largest_family": 4,
        }

    def interval(
        relationship: object,
        first_trait: object,
        second_trait: object,
        design: object,
        coordinate: str,
    ) -> dict[str, object]:
        """Return either the matching profile or a wrong-run estimate."""
        del relationship, first_trait, second_trait, design
        assert coordinate == "genetic_correlation"
        return {
            "estimate": 0.3 if malformation == "interval_estimate" else 0.4,
            "lower": 0.2,
            "upper": 0.6,
            "lower_limited": False,
            "upper_limited": False,
            "level": 0.95,
            "profile_failures": 0,
            "what": "genetic_correlation",
            "estimator": "ml",
        }

    def test(
        relationship: object,
        first_trait: object,
        second_trait: object,
        design: object,
        coordinate: str,
    ) -> dict[str, object]:
        """Return one selectively malformed likelihood-ratio record."""
        del relationship, first_trait, second_trait, design
        assert coordinate == "genetic_correlation"
        alternative_loglik: float = (
            -10.5 if malformation == "alternative_likelihood" else -11.0
        )
        """Changed only the repeated free likelihood when requested."""

        statistic: float = 1.0 if malformation == "statistic" else 2.0
        """Changed only the likelihood-ratio statistic when requested."""

        p_value: float = 0.5 if malformation == "p_value" else float(chi2.sf(2.0, 1))
        """Changed only the chi-square tail probability when requested."""

        return {
            "what": "genetic_correlation",
            "statistic": statistic,
            "p_value": p_value,
            "rule": "chi2_1",
            "null_loglik": -12.0,
            "alternative_loglik": alternative_loglik,
            "estimator": "ml",
        }

    monkeypatch.setattr(module, "simulate_traits", fixed_traits)
    monkeypatch.setattr(module.asterism, "mixed_bivariate_fit", fit)
    monkeypatch.setattr(module.asterism, "mixed_bivariate_interval", interval)
    monkeypatch.setattr(module.asterism, "mixed_bivariate_test", test)
    """Controlled only participant-free generation and documented public records."""

    state: object = module.WorkerState(
        relationship=np.eye(4),
        design=np.ones((4, 1)),
        component_sizes=(4,),
        genetic_factors=(np.eye(4),),
        prevalence=0.25,
        heritabilities=(0.5, 0.5),
        residual_correlation=0.15,
        hearing_variance=3.0,
        minimum_cases=1,
        minimum_measured_hearing=1,
        nominal_coverage=0.95,
        required_test_rule="chi2_1",
        required_interval_level=0.95,
    )
    """Built one valid state so only cross-run inference identity could fail."""

    module.start_worker(state)
    result: object = module.run_replicate(
        module.ReplicateJob(
            genetic_correlation=0.4,
            censoring_share=0.52,
            replicate=0,
            seed=1,
        )
    )
    """Ran the malformed record through fail-closed public classification."""

    assert result.outcome == expected_outcome
    assert result.covered is False


def test_latent_generator_has_the_specified_joint_covariance() -> None:
    """Protect genetic, residual, and scale identities independently of fitting."""
    module: ModuleType = target_module()
    """Loaded the production latent-trait generator."""

    relationship: npt.NDArray[np.float64] = np.asarray([[1.0, 0.5], [0.5, 1.0]])
    """Fixed one sibling pair with unit additive diagonal."""

    state: object = module.WorkerState(
        relationship=relationship,
        design=np.ones((2, 1)),
        component_sizes=(2,),
        genetic_factors=(np.linalg.cholesky(relationship),),
        prevalence=0.25,
        heritabilities=(0.5, 0.5),
        residual_correlation=0.15,
        hearing_variance=3.0,
        minimum_cases=1,
        minimum_measured_hearing=1,
        nominal_coverage=0.95,
        required_test_rule="chi2_1",
        required_interval_level=0.95,
    )
    """Specified the model covariance independently of generated samples."""

    draws: list[npt.NDArray[np.float64]] = []
    """Collected deterministic Monte Carlo draws of all four latent variables."""

    for replicate in range(12_000):
        liability, hearing = module.simulate_latent_traits(
            state,
            module.ReplicateJob(
                genetic_correlation=0.4,
                censoring_share=0.52,
                replicate=replicate,
                seed=300_000 + replicate,
            ),
        )
        """Generated one sibling-pair draw from a distinct fixed stream."""

        draws.append(np.concatenate([liability, hearing]))
        """Stacked variables in the same block order as the expected covariance."""
    """Generated enough independent draws to distinguish missing covariance terms."""

    observed: npt.NDArray[np.float64] = np.cov(np.asarray(draws), rowvar=False)
    """Estimated the four-variable covariance from independent streams."""

    genetic: npt.NDArray[np.float64] = 0.5 * relationship
    """Computed each trait's additive covariance contribution."""

    liability_covariance: npt.NDArray[np.float64] = genetic + 0.5 * np.eye(2)
    """Added independent residual covariance on the unit liability scale."""

    hearing_covariance: npt.NDArray[np.float64] = 3.0 * liability_covariance
    """Applied the complete-hearing variance after identical heritability."""

    cross_covariance: npt.NDArray[np.float64] = np.sqrt(3.0) * (
        0.2 * relationship + 0.075 * np.eye(2)
    )
    """Combined genetic and residual cross-trait covariance independently."""

    expected: npt.NDArray[np.float64] = np.block(
        [
            [liability_covariance, cross_covariance],
            [cross_covariance, hearing_covariance],
        ]
    )
    """Assembled the scientific model covariance in tested variable order."""

    assert observed == pytest.approx(expected, abs=0.06)


def test_campaign_scoring_keeps_failures_in_every_cell_denominator() -> None:
    """Separate bounded execution from release-sized scientific evidence."""
    module: ModuleType = target_module()
    """Loaded the unconditional target-campaign scorer."""

    target: object = module.load_target_design(
        TARGET_FIXTURE,
        expected_sha256=module.TARGET_FIXTURE_SHA256,
    )
    """Loaded the exact prewritten grid and acceptance contract."""

    complete: list[object] = [
        module.ReplicateResult(
            genetic_correlation=correlation,
            censoring_share=censoring_share,
            replicate=0,
            seed=target.base_seed + cell * target.cell_seed_offset,
            outcome="complete",
            cases=954,
            measured_hearing=916 if censoring_share == 0.52 else 477,
            converged=True,
            estimate=correlation,
            lower=correlation - 0.1,
            upper=correlation + 0.1,
            covered=True,
            profile_failures=0,
            p_value=0.5,
            statistic=0.4,
            test_rule="chi2_1",
        )
        for cell, (correlation, censoring_share) in enumerate(
            (
                (0.0, 0.52),
                (0.0, 0.75),
                (0.4, 0.52),
                (0.4, 0.75),
            )
        )
    ]
    """Represented one complete operational smoke attempt in every target cell."""

    smoke: dict[str, object] = module.score_campaign(complete, target)
    """Scored the bounded run without promoting it to calibration evidence."""

    assert smoke["attempted_replicates"] == 4
    assert smoke["failed_replicates"] == 0
    assert smoke["execution_passed"] is True
    assert smoke["release_sized"] is False
    assert smoke["scientific_passed"] is False

    failed: list[object] = complete + [
        module.ReplicateResult(
            genetic_correlation=correlation,
            censoring_share=censoring_share,
            replicate=1,
            seed=(
                target.base_seed
                + cell * target.cell_seed_offset
                + target.replicate_seed_offset
            ),
            outcome="refused",
            cases=955,
            measured_hearing=917 if censoring_share == 0.52 else 478,
            covered=False,
            error_code="MIXED_TARGET_PUBLIC_REFUSAL",
        )
        for cell, (correlation, censoring_share) in enumerate(
            (
                (0.0, 0.52),
                (0.0, 0.75),
                (0.4, 0.52),
                (0.4, 0.75),
            )
        )
    ]
    """Added one explicit failed attempt to every unconditional denominator."""

    scored: dict[str, object] = module.score_campaign(failed, target)
    """Applied the same rules without dropping unavailable inference."""

    assert scored["attempted_replicates"] == 8
    assert scored["failed_replicates"] == 4
    assert scored["execution_passed"] is False
    assert scored["scientific_passed"] is False

    cells: dict[str, object] = scored["cells"]
    """Selected per-cell accounting records for denominator verification."""

    for key, cell in cells.items():
        assert cell["attempted"] == 2
        assert cell["complete"] == 1
        assert cell["covered"] == 1
        assert cell["coverage"] == 0.5
        assert cell["case_share_min"] == pytest.approx(954 / 1_909)
        assert cell["case_share_max"] == pytest.approx(955 / 1_909)
        if "censored=0.52" in key:
            assert cell["achieved_censoring_share_min"] == pytest.approx(992 / 1_909)
            assert cell["achieved_censoring_share_max"] == pytest.approx(993 / 1_909)
        else:
            assert cell["achieved_censoring_share_min"] == pytest.approx(1_431 / 1_909)
            assert cell["achieved_censoring_share_max"] == pytest.approx(1_432 / 1_909)
        assert cell["design_observations_complete"] is True

    shifted_coordinates: list[object] = [
        replace(result, replicate=result.replicate + 1) for result in complete
    ]
    """Shifted every unique coordinate away from the frozen zero-based schedule."""

    with pytest.raises(ValueError, match="MIXED_TARGET_CAMPAIGN_COORDINATES_INVALID"):
        module.score_campaign(shifted_coordinates, target)

    reused_seeds: list[object] = [
        replace(result, seed=target.base_seed) for result in complete
    ]
    """Reused one stream across cells while preserving unique coordinates."""

    with pytest.raises(ValueError, match="MIXED_TARGET_CAMPAIGN_SEEDS_INVALID"):
        module.score_campaign(reused_seeds, target)


def test_campaign_score_is_strict_json_when_recovery_precision_is_unavailable() -> None:
    """Represent unavailable Monte Carlo precision with null, never Infinity."""
    module: ModuleType = target_module()
    """Loaded the target scorer and its exact reviewed grid."""

    target: object = module.load_target_design(
        TARGET_FIXTURE,
        expected_sha256=module.TARGET_FIXTURE_SHA256,
    )
    """Loaded the exact target denominator and acceptance contract."""

    results: list[object] = [
        module.ReplicateResult(
            genetic_correlation=correlation,
            censoring_share=censoring_share,
            replicate=0,
            seed=target.base_seed + cell * target.cell_seed_offset,
            outcome="complete",
            cases=485,
            measured_hearing=916 if censoring_share == 0.52 else 477,
            converged=True,
            estimate=correlation + 0.01,
            lower=correlation - 0.1,
            upper=correlation + 0.1,
            covered=True,
            profile_failures=0,
            p_value=0.5,
            statistic=0.4,
            test_rule="chi2_1",
        )
        for cell, (correlation, censoring_share) in enumerate(
            (
                (0.0, 0.52),
                (0.0, 0.75),
                (0.4, 0.52),
                (0.4, 0.75),
            )
        )
    ]
    """Created one nonexact estimate per cell without estimable Monte Carlo SE."""

    score: dict[str, object] = module.score_campaign(results, target)
    """Scored unavailable precision without inventing a finite recovery distance."""

    for cell in score["cells"].values():
        assert cell["recovery_distance_in_standard_errors"] is None
        assert cell["recovery_passed"] is False
    assert score["scientific_passed"] is False
    json.dumps(score, allow_nan=False)
    """Proved the machine evidence is strict interoperable JSON."""


def test_command_freezes_the_target_campaign() -> None:
    """Keep the deferred target command runnable for a later release."""
    module: ModuleType = target_module()
    """Loaded the explicit target command-line contract."""

    command: list[str] = [
        "--replicates",
        "300",
        "--workers",
        "8",
        "--prevalence",
        "0.254",
        "--heritabilities",
        "0.5",
        "0.5",
        "--genetic-correlations",
        "0.0",
        "0.4",
        "--residual-correlation",
        "0.15",
        "--hearing-variance",
        "3.0",
        "--censoring-shares",
        "0.52",
        "0.75",
        "--nominal-coverage",
        "0.95",
        "--nominal-level",
        "0.05",
        "--monte-carlo-confidence",
        "0.95",
        "--point-recovery-standard-errors",
        "3.0",
        "--base-seed",
        "1210000",
        "--cell-seed-offset",
        "104729",
        "--replicate-seed-offset",
        "7919",
        "--minimum-cases",
        "10",
        "--minimum-measured-hearing",
        "10",
        "--design-fixture",
        "checks/design_fixtures/mixed_binary_censored_target_design.json",
        "--fixture-sha256",
        TARGET_FIXTURE_SHA256,
        "--no-write",
    ]
    """Restated every fixture-owned release coordinate independently."""

    parsed: object = module.parse_arguments(command)
    """Parsed the exact release rule through the real public CLI boundary."""

    assert parsed.replicates == 300
    assert parsed.workers == 8
    assert parsed.genetic_correlations == [0.0, 0.4]
    assert parsed.censoring_shares == [0.52, 0.75]
    assert parsed.no_write is True


def test_release_manifest_defers_the_binary_with_censored_pairing() -> None:
    """Stop 0.1 claiming a pairing the extended-high-frequency paper never fits."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the authoritative scientific command inventory."""

    analysis: dict[str, object] = next(
        entry
        for entry in manifest["analyses"]
        if entry["id"] == "mixed_binary_censored_genetic_correlation"
    )
    """Selected the supported mixed analysis without relying on position."""

    assert "mixed_binary_censored_target_design" not in analysis["required_checks"]
    assert "mixed_binary_censored_exact_combination" not in analysis["required_checks"]
    """Held the deferral for the binary-with-right-censored pairing."""

    assert analysis["design_range"]["trait_type"]["allowed"] == [
        "binary_and_continuous",
        "mixed_pairings_with_continuous",
    ]
    """Kept exactly the mixed pairings the paper measures."""

    assert (ROOT / "checks" / "mixed_binary_censored_target_design.py").is_file()
    assert (ROOT / "checks" / "mixed_binary_censored_exact_combination.py").is_file()
    """Kept the deferred material intact and runnable for a later release."""
