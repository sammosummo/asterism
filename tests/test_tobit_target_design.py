"""Contracts for the participant-free Tobit target-design gate."""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the standalone scientific command."""

TARGET_FIXTURE: Path = (
    ROOT / "checks" / "design_fixtures" / "one_trait_gaussian_heritability.json"
)
"""Selected the existing reviewed values-free pedigree-scale fixture."""

TARGET_FIXTURE_SHA256: str = (
    "93e74ad688784dd70f5080a8969f22a07bee4ca6d8f3fb8e2a7fea7b9e70607e"
)
"""Pinned the reviewed fixture bytes independently of the new command."""


def target_module() -> ModuleType:
    """Load the standalone Tobit target command without running its CLI."""
    path: Path = ROOT / "checks" / "tobit_target_design.py"
    """Located the exact maintained command under test."""

    specification: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location("tobit_target_design", path)
    )
    """Created a module specification through the command's file boundary."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the module governed by the validated specification."""

    sys.modules[specification.name] = module
    """Registered the module before evaluating dataclass annotations."""

    checks_directory: str = str(path.parent)
    """Selected the sibling-check import root used by direct commands."""

    sys.path.insert(0, checks_directory)
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(checks_directory)
    """Loaded declarations through the same import boundary as CLI execution."""
    return module


def test_reviewed_fixture_builds_the_intended_tobit_pedigree_scale() -> None:
    """Stress the intended roster without retaining participant material."""
    module: ModuleType = target_module()
    """Loaded the target fixture verifier and deterministic constructor."""

    target: object = module.load_target_design(
        TARGET_FIXTURE,
        expected_sha256=TARGET_FIXTURE_SHA256,
    )
    """Built the Tobit target from the already-reviewed structural fixture."""

    assert target.relationship.shape == (1_909, 1_909)
    assert target.design.shape == (1_909, 6)
    assert len(target.component_sizes) == 202
    assert sum(target.component_sizes) == 1_909
    assert max(target.component_sizes) == 180
    assert target.fixture_sha256 == TARGET_FIXTURE_SHA256


def test_clopper_pearson_acceptance_is_one_sided_and_predeclared() -> None:
    """Fail only established anti-conservatism or undercoverage."""
    module: ModuleType = target_module()
    """Loaded the exact-binomial campaign decisions."""

    assert module.one_sided_lower_limit(20, 100, 0.95) == pytest.approx(
        0.13666132524586722
    )
    assert module.one_sided_upper_limit(90, 100, 0.95) == pytest.approx(
        0.94473676231713
    )

    anti_conservative: dict[str, object] = module.level_decision(
        rejections=20,
        attempted=100,
        level=0.05,
        confidence=0.95,
    )
    """Scored a null-test cell whose exact lower bound exceeds its level."""

    compatible_level: dict[str, object] = module.level_decision(
        rejections=1,
        attempted=100,
        level=0.05,
        confidence=0.95,
    )
    """Scored a null-test cell without established anti-conservatism."""

    undercovered: dict[str, object] = module.coverage_decision(
        covered=90,
        attempted=100,
        nominal=0.95,
        confidence=0.95,
    )
    """Scored an interval cell whose exact upper bound misses nominal coverage."""

    compatible_coverage: dict[str, object] = module.coverage_decision(
        covered=95,
        attempted=100,
        nominal=0.95,
        confidence=0.95,
    )
    """Scored interval evidence still compatible with nominal coverage."""

    assert anti_conservative["passed"] is False
    assert compatible_level["passed"] is True
    assert undercovered["passed"] is False
    assert compatible_coverage["passed"] is True


def test_scoring_keeps_exact_failure_buckets_in_every_denominator() -> None:
    """Count unavailable inference as attempted level and coverage evidence."""
    module: ModuleType = target_module()
    """Loaded the unconditional target-campaign scorer."""

    rows: list[object] = [
        module.ReplicateResult(
            scenario="null",
            censoring_share=0.52,
            replicate=0,
            seed=1,
            outcome="complete",
            p_value=0.04,
        ),
        module.ReplicateResult(
            scenario="null",
            censoring_share=0.52,
            replicate=1,
            seed=2,
            outcome="refused",
        ),
        module.ReplicateResult(
            scenario="heritable",
            censoring_share=0.52,
            replicate=0,
            seed=3,
            outcome="complete",
            covered=True,
        ),
        module.ReplicateResult(
            scenario="heritable",
            censoring_share=0.52,
            replicate=1,
            seed=4,
            outcome="profile_failed",
            covered=False,
            profile_failures=1,
        ),
    ]
    """Represented complete, refused, and failed-profile scientific attempts."""

    scored: dict[str, object] = module.score_campaign(
        rows,
        replicates_per_cell=2,
        censoring_shares=(0.52,),
        level=0.05,
        nominal_coverage=0.95,
        monte_carlo_confidence=0.95,
        maximum_failed_replicates=0,
    )
    """Applied the frozen decisions without filtering unavailable records."""

    rate: dict[str, object] = scored["rates"]["0.52"]
    """Selected the complete accounting record for one censoring level."""

    null: dict[str, object] = rate["null"]
    """Selected unconditional boundary-test accounting."""

    heritable: dict[str, object] = rate["heritable"]
    """Selected unconditional interval-coverage accounting."""

    assert null["attempted"] == 2
    assert null["computed"] == 1
    assert null["level"]["rejections"] == 1
    assert null["level"]["rate"] == 0.5
    assert heritable["attempted"] == 2
    assert heritable["computed"] == 1
    assert heritable["coverage"]["covered"] == 1
    assert heritable["coverage"]["coverage"] == 0.5
    assert scored["failure_buckets"] == {
        "invalid_censoring_count": 0,
        "refused": 1,
        "nonconverged": 0,
        "test_failed": 0,
        "interval_failed": 0,
        "profile_failed": 1,
    }
    assert scored["failed_replicates"] == 2
    assert scored["passed"] is False


def test_scoring_refuses_a_missing_requested_replicate_coordinate() -> None:
    """Prevent a shortened campaign from becoming a smaller denominator."""
    module: ModuleType = target_module()
    """Loaded the exact campaign-completeness validator."""

    rows: list[object] = [
        module.ReplicateResult(
            scenario=scenario,
            censoring_share=0.52,
            replicate=0,
            seed=index,
            outcome="complete",
            p_value=0.5,
            covered=True,
        )
        for index, scenario in enumerate(("null", "heritable"))
    ]
    """Omitted replicate one from both requested scientific cells."""

    with pytest.raises(ValueError, match="TOBIT_TARGET_CELL_COORDINATES_INVALID"):
        module.score_campaign(
            rows,
            replicates_per_cell=2,
            censoring_shares=(0.52,),
            level=0.05,
            nominal_coverage=0.95,
            monte_carlo_confidence=0.95,
            maximum_failed_replicates=0,
        )


def test_command_requires_every_scientific_coordinate() -> None:
    """Keep release and smoke campaigns independent of ambient defaults."""
    module: ModuleType = target_module()
    """Loaded the standalone command-line parser."""

    with pytest.raises(SystemExit):
        module.parse_arguments([])

    arguments: object = module.parse_arguments(
        [
            "--replicates",
            "1",
            "--workers",
            "1",
            "--censoring-shares",
            "0.52",
            "0.75",
            "--true-heritability",
            "0.5",
            "--true-variance",
            "4.0",
            "--level",
            "0.05",
            "--nominal-coverage",
            "0.95",
            "--monte-carlo-confidence",
            "0.95",
            "--base-seed",
            "920000",
            "--scenario-seed-offset",
            "1000003",
            "--rate-seed-offset",
            "10007",
            "--maximum-failed-replicates",
            "0",
            "--design-fixture",
            str(TARGET_FIXTURE),
            "--fixture-sha256",
            TARGET_FIXTURE_SHA256,
            "--no-write",
        ]
    )
    """Read one bounded but fully explicit participant-free campaign."""

    assert arguments.replicates == 1
    assert arguments.workers == 1
    assert arguments.censoring_shares == [0.52, 0.75]
    assert arguments.no_write is True


def test_latent_response_uses_a_fixed_instrument_limit_before_the_draw() -> None:
    """Generate target observations without conditioning the limit on outcomes."""
    module: ModuleType = target_module()
    """Loaded the target covariance and response generator."""

    relationship: np.ndarray = np.asarray(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ]
    )
    """Fixed two unrelated synthetic sibling components."""

    target: object = module.TargetDesign(
        relationship=relationship,
        design=np.ones((4, 1)),
        component_sizes=(2, 2),
        fixture_sha256="a" * 64,
    )
    """Built the smallest two-component design accepted by the generator."""

    factors: tuple[np.ndarray, ...] = module.covariance_factors(
        target,
        heritability=0.5,
        total_variance=4.0,
    )
    """Factored each independent relationship component separately."""

    censoring_limit: float = module.right_censoring_limit(
        target.design @ np.asarray([10.0]),
        4.0,
        0.5,
    )
    """Solved the intended marginal censoring share before drawing a response."""

    first: object = module.simulate_observation(
        target,
        factors,
        mean_coefficients=np.asarray([10.0]),
        censoring_limit=censoring_limit,
        seed=44,
    )
    """Generated one participant-free right-censored response."""

    second: object = module.simulate_observation(
        target,
        factors,
        mean_coefficients=np.asarray([10.0]),
        censoring_limit=censoring_limit,
        seed=44,
    )
    """Repeated the identical stream and scientific coordinates."""

    third: object = module.simulate_observation(
        target,
        factors,
        mean_coefficients=np.asarray([10.0]),
        censoring_limit=censoring_limit,
        seed=45,
    )
    """Changed the outcome stream without recalibrating the instrument."""

    assert first.censored_count == np.count_nonzero(first.censoring == 1)
    assert first.achieved_censoring_share == first.censored_count / 4
    assert np.array_equal(first.censoring, second.censoring)
    assert np.array_equal(first.value, second.value, equal_nan=True)
    assert np.array_equal(first.limit, second.limit)
    assert np.all(first.limit == censoring_limit)
    assert np.all(third.limit == censoring_limit)
    assert np.isnan(first.value[first.censoring == 1]).all()
    assert np.isfinite(first.value[first.censoring == 0]).all()


def test_replicate_uses_only_the_three_public_tobit_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise fit, profile interval, and boundary test through public names."""
    module: ModuleType = target_module()
    """Loaded the worker independently of process orchestration."""

    calls: list[str] = []
    """Recorded the documented operations in invocation order."""

    target: object = module.TargetDesign(
        relationship=np.eye(4),
        design=np.ones((4, 1)),
        component_sizes=(4,),
        fixture_sha256="a" * 64,
    )
    """Built a minimal participant-free worker target."""

    observed: object = module.ObservedTobit(
        value=np.asarray([0.2, -0.1, np.nan, np.nan]),
        censoring=np.asarray([0, 0, 1, 1]),
        limit=np.ones(4),
        censored_count=2,
        achieved_censoring_share=0.5,
    )
    """Fixed a valid exact-count public Tobit input."""

    def fixed_observation(
        selected_target: object,
        factors: tuple[npt.NDArray[np.float64], ...],
        *,
        mean_coefficients: npt.NDArray[np.float64],
        censoring_limit: float,
        seed: int,
    ) -> object:
        """Return the fixed observation while retaining the production signature."""
        assert selected_target is target
        assert len(factors) == 1
        assert mean_coefficients.shape == (1,)
        assert censoring_limit == 1.0
        assert seed == 8
        return observed

    monkeypatch.setattr(module, "simulate_observation", fixed_observation)
    """Held simulation fixed so this test isolates the public inference seam."""

    def fit(
        relationship: npt.NDArray[np.float64],
        value: npt.NDArray[np.float64],
        censoring: npt.NDArray[np.int64],
        limit: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
    ) -> dict[str, object]:
        """Return a converged minimal public Tobit fit record."""
        assert relationship is target.relationship
        assert value is observed.value
        assert censoring is observed.censoring
        assert limit is observed.limit
        assert design is target.design
        calls.append("fit")
        return {
            "heritability": 0.45,
            "total_variance": 3.9,
            "converged": True,
            "censored_share": 0.5,
            "largest_family": 4,
        }

    def interval(
        relationship: npt.NDArray[np.float64],
        value: npt.NDArray[np.float64],
        censoring: npt.NDArray[np.int64],
        limit: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
    ) -> dict[str, object]:
        """Return a complete public heritability interval."""
        del relationship, value, censoring, limit, design
        calls.append("interval")
        return {
            "estimate": 0.45,
            "lower": 0.2,
            "upper": 0.7,
            "level": 0.95,
            "profile_failures": 0,
            "contains_lower_bound": None,
            "contains_upper_bound": None,
        }

    def test(
        relationship: npt.NDArray[np.float64],
        value: npt.NDArray[np.float64],
        censoring: npt.NDArray[np.int64],
        limit: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
    ) -> dict[str, object]:
        """Return a complete public boundary-test record."""
        del relationship, value, censoring, limit, design
        calls.append("test")
        return {
            "statistic": 3.0,
            "p_value": 0.04,
            "rule": "mixture_50_50",
        }

    monkeypatch.setattr(module.asterism, "tobit_fit", fit)
    monkeypatch.setattr(module.asterism, "tobit_interval", interval)
    monkeypatch.setattr(module.asterism, "tobit_test", test)
    """Replaced only the three documented public calculations."""

    state: object = module.WorkerState(
        target=target,
        covariance_factors_by_scenario={
            "null": (np.eye(4),),
            "heritable": (np.eye(4),),
        },
        mean_coefficients=np.asarray([10.0]),
        censoring_limits_by_share={0.5: 1.0},
        true_heritability=0.5,
        required_test_rule="mixture_50_50",
        required_interval_level=0.95,
    )
    """Built the smallest state satisfying the public model contract."""

    module.start_worker(state)
    """Installed state through the campaign's process initializer."""

    result: object = module.run_replicate(
        module.ReplicateJob(
            scenario="heritable",
            censoring_share=0.5,
            replicate=0,
            seed=8,
        )
    )
    """Ran one complete public fit, interval, and test attempt."""

    assert result.outcome == "complete"
    assert result.fit_heritability == 0.45
    assert result.fit_total_variance == 3.9
    assert result.interval_lower == 0.2
    assert result.interval_upper == 0.7
    assert result.covered is True
    assert result.p_value == 0.04
    assert result.test_rule == "mixture_50_50"
    assert result.profile_failures == 0
    assert calls == ["fit", "interval", "test"]


@pytest.mark.parametrize(
    ("mode", "expected_outcome"),
    [
        ("bad_count", "invalid_censoring_count"),
        ("fit_refusal", "refused"),
        ("nonconverged", "nonconverged"),
        ("interval_refusal", "interval_failed"),
        ("profile_failure", "profile_failed"),
        ("test_refusal", "test_failed"),
        ("wrong_test_rule", "test_failed"),
    ],
)
def test_replicate_assigns_one_exact_failure_bucket(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_outcome: str,
) -> None:
    """Classify every attempted failure without dropping its denominator row."""
    module: ModuleType = target_module()
    """Loaded the public worker and frozen failure vocabulary."""

    target: object = module.TargetDesign(
        relationship=np.eye(4),
        design=np.ones((4, 1)),
        component_sizes=(4,),
        fixture_sha256="a" * 64,
    )
    """Built a minimal participant-free target for controlled failures."""

    observed: object = module.ObservedTobit(
        value=np.asarray([0.2, -0.1, np.nan, np.nan]),
        censoring=(
            np.asarray([0, 0, 0, 1])
            if mode == "bad_count"
            else np.asarray([0, 0, 1, 1])
        ),
        limit=np.ones(4),
        censored_count=2,
        achieved_censoring_share=0.5,
    )
    """Changed only the actual status count for the invalid-count case."""

    monkeypatch.setattr(
        module,
        "simulate_observation",
        lambda *arguments, **keywords: observed,
    )
    """Held the synthetic response fixed while selecting one failure mode."""

    def fit(*arguments: object) -> dict[str, object]:
        """Return a fit refusal, nonconvergence, or valid public record."""
        del arguments
        if mode == "fit_refusal":
            raise ValueError("TOBIT_TARGET_TEST_REFUSAL")
        return {
            "heritability": 0.45,
            "total_variance": 3.9,
            "converged": mode != "nonconverged",
            "censored_share": 0.5,
            "largest_family": 4,
        }

    def interval(*arguments: object) -> dict[str, object]:
        """Return an interval refusal, failed profile, or complete record."""
        del arguments
        if mode == "interval_refusal":
            raise ValueError("TOBIT_TARGET_INTERVAL_REFUSAL")
        return {
            "estimate": 0.45,
            "lower": 0.2,
            "upper": 0.7,
            "level": 0.95,
            "profile_failures": 1 if mode == "profile_failure" else 0,
            "contains_lower_bound": None,
            "contains_upper_bound": None,
        }

    def test(*arguments: object) -> dict[str, object]:
        """Return a test refusal, wrong rule, or complete boundary record."""
        del arguments
        if mode == "test_refusal":
            raise ValueError("TOBIT_TARGET_TEST_REFUSAL")
        return {
            "statistic": 3.0,
            "p_value": 0.04,
            "rule": "chi2_1" if mode == "wrong_test_rule" else "mixture_50_50",
        }

    monkeypatch.setattr(module.asterism, "tobit_fit", fit)
    monkeypatch.setattr(module.asterism, "tobit_interval", interval)
    monkeypatch.setattr(module.asterism, "tobit_test", test)
    """Selected one controlled failure beneath the three public names."""

    module.start_worker(
        module.WorkerState(
            target=target,
            covariance_factors_by_scenario={
                "null": (np.eye(4),),
                "heritable": (np.eye(4),),
            },
            mean_coefficients=np.asarray([10.0]),
            censoring_limits_by_share={0.5: 1.0},
            true_heritability=0.5,
            required_test_rule="mixture_50_50",
            required_interval_level=0.95,
        )
    )
    """Installed the same valid target state for every selected failure."""

    result: object = module.run_replicate(
        module.ReplicateJob(
            scenario="heritable",
            censoring_share=0.5,
            replicate=0,
            seed=8,
        )
    )
    """Retained the failed attempt through the normal worker boundary."""

    assert result.outcome == expected_outcome
    assert result.outcome in module.FAILURE_OUTCOMES


def test_campaign_jobs_cover_both_scenarios_at_both_censoring_levels() -> None:
    """Give every requested scientific cell an exact deterministic denominator."""
    module: ModuleType = target_module()
    """Loaded the schedule-independent campaign enumerator."""

    jobs: list[object] = module.campaign_jobs(
        replicates=2,
        censoring_shares=(0.52, 0.75),
        base_seed=920_000,
        scenario_seed_offset=1_000_003,
        rate_seed_offset=10_007,
    )
    """Enumerated two replicates in all four scientific cells."""

    assert len(jobs) == 8
    assert [
        (job.scenario, job.censoring_share, job.replicate, job.seed) for job in jobs
    ] == [
        ("null", 0.52, 0, 920_000),
        ("null", 0.52, 1, 920_001),
        ("heritable", 0.52, 0, 1_920_003),
        ("heritable", 0.52, 1, 1_920_004),
        ("null", 0.75, 0, 930_007),
        ("null", 0.75, 1, 930_008),
        ("heritable", 0.75, 0, 1_930_010),
        ("heritable", 0.75, 1, 1_930_011),
    ]


def test_main_emits_complete_values_free_development_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Separate a bounded route smoke from the fixed release campaign."""
    module: ModuleType = target_module()
    """Loaded the command boundary and machine-readable evidence builder."""

    target: object = module.TargetDesign(
        relationship=np.eye(4),
        design=np.ones((4, 1)),
        component_sizes=(4,),
        fixture_sha256=TARGET_FIXTURE_SHA256,
    )
    """Substituted a small participant-free target to isolate orchestration."""

    monkeypatch.setattr(module, "load_target_design", lambda *args, **kwargs: target)
    """Avoided rebuilding the reviewed dense target in a command-shape test."""

    monkeypatch.setattr(module, "MEAN_COEFFICIENTS", np.asarray([10.0]))
    """Matched the compact one-column target used by this orchestration test."""

    def campaign(selected_target: object, configuration: object) -> list[object]:
        """Return one complete attempt in each requested scientific cell."""
        assert selected_target is target
        assert configuration.replicates == 1
        assert configuration.workers == 1
        return [
            module.ReplicateResult(
                scenario=scenario,
                censoring_share=rate,
                replicate=0,
                seed=index,
                outcome="complete",
                censored_count=2 if rate == 0.52 else 3,
                achieved_censoring_share=0.5 if rate == 0.52 else 0.75,
                p_value=0.5,
                test_rule="mixture_50_50",
                covered=True,
                fit_heritability=0.5,
                fit_total_variance=4.0,
                interval_lower=0.2,
                interval_upper=0.8,
                profile_failures=0,
            )
            for index, (rate, scenario) in enumerate(
                (
                    (0.52, "null"),
                    (0.52, "heritable"),
                    (0.75, "null"),
                    (0.75, "heritable"),
                )
            )
        ]

    monkeypatch.setattr(module, "run_campaign", campaign, raising=False)
    """Held numerical fitting fixed while retaining complete denominator rows."""

    exit_code: int = module.main(
        [
            "--replicates",
            "1",
            "--workers",
            "1",
            "--censoring-shares",
            "0.52",
            "0.75",
            "--true-heritability",
            "0.5",
            "--true-variance",
            "4.0",
            "--level",
            "0.05",
            "--nominal-coverage",
            "0.95",
            "--monte-carlo-confidence",
            "0.95",
            "--base-seed",
            "920000",
            "--scenario-seed-offset",
            "1000003",
            "--rate-seed-offset",
            "10007",
            "--maximum-failed-replicates",
            "0",
            "--design-fixture",
            str(TARGET_FIXTURE),
            "--fixture-sha256",
            TARGET_FIXTURE_SHA256,
            "--no-write",
        ]
    )
    """Ran the bounded command with every release coordinate still explicit."""

    evidence: dict[str, object] = json.loads(capsys.readouterr().out)
    """Parsed the command's sole standard-output evidence document."""

    assert exit_code == 0
    assert evidence["check"] == "tobit_target_design"
    assert evidence["participant_free"] is True
    assert evidence["attempted_replicates"] == 4
    assert evidence["replicates_per_scenario_per_censoring_share"] == 1
    assert evidence["every_attempted_replicate_in_denominator"] is True
    assert evidence["exact_release_campaign"] is False
    assert evidence["passed"] is True
    assert len(evidence["attempts"]) == 4
    assert evidence["design_facts"]["expected_censoring_shares"] == [0.52, 0.75]
    assert evidence["design_facts"]["fixed_censoring_limits"] == pytest.approx(
        [9.899692833070535, 8.651020499607835]
    )
    assert evidence["design_facts"]["achieved_censoring_share_range"] == {
        "min": 0.5,
        "max": 0.75,
    }


def test_command_source_uses_only_the_public_asterism_interface() -> None:
    """Keep release evidence independent of private compiled entry points."""
    path: Path = ROOT / "checks" / "tobit_target_design.py"
    """Located the exact command source governed by this contract."""

    source: str = path.read_text(encoding="utf-8")
    """Read source rather than trusting a runtime monkeypatch seam."""

    tree: ast.Module = ast.parse(source)
    """Parsed imports and package-root attribute access structurally."""

    imported_modules: set[str] = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    """Collected explicit module imports without substring heuristics."""

    public_calls: set[str] = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "asterism"
    }
    """Collected every Asterism package-root attribute read by the command."""

    assert "asterism" in imported_modules
    assert public_calls == {"tobit_fit", "tobit_interval", "tobit_test"}
    assert "_core" not in source
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "asterism"
        for node in ast.walk(tree)
    )


def test_manifest_binds_one_exact_measured_target_design_rule() -> None:
    """Keep the completed fixed-instrument campaign as the release rule."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the authoritative release contract."""

    analyses: list[dict[str, object]] = manifest["analyses"]
    """Selected the supported scientific analysis inventory."""

    analysis: dict[str, object] = next(
        candidate for candidate in analyses if candidate["id"] == "one_trait_censored"
    )
    """Selected the sole censored one-trait release claim."""

    assert analysis["required_checks"].count("tobit_target_design") == 1
    rules: list[dict[str, object]] = [
        rule for rule in analysis["pass_rules"] if rule["id"] == "tobit_target_design"
    ]
    """Required a one-to-one executable rule for the new claim."""

    assert len(rules) == 1
    rule: dict[str, object] = rules[0]
    """Selected the exact target-design command and its non-identifying facts."""

    assert rule["command"] == [
        "checks/tobit_target_design.py",
        "--replicates",
        "200",
        "--workers",
        "6",
        "--censoring-shares",
        "0.52",
        "0.75",
        "--true-heritability",
        "0.5",
        "--true-variance",
        "4.0",
        "--level",
        "0.05",
        "--nominal-coverage",
        "0.95",
        "--monte-carlo-confidence",
        "0.95",
        "--base-seed",
        "920000",
        "--scenario-seed-offset",
        "1000003",
        "--rate-seed-offset",
        "10007",
        "--maximum-failed-replicates",
        "0",
        "--design-fixture",
        "checks/design_fixtures/one_trait_gaussian_heritability.json",
        "--fixture-sha256",
        TARGET_FIXTURE_SHA256,
        "--no-write",
    ]
    assert rule["expected_exit_code"] == 0
    assert rule["status"] == "ready"
    assert "blocker" not in rule
    assert rule["design_facts"] == {
        "participant_free": True,
        "measured": True,
        "sample_size": {"min": 1_909, "max": 1_909},
        "largest_family": {"max": 180},
        "trait_type": ["right_censored_continuous"],
        "components": ["additive_relationship", "residual"],
        "relationship_components": 202,
        "fixed_effect_columns": 6,
        "expected_censoring_share": {"min": 0.52, "max": 0.75},
        "censoring_limits_fixed_before_outcomes": True,
        "scenarios": ["null", "heritable"],
        "true_heritabilities": [0.0, 0.5],
        "true_variance": 4.0,
        "replicates_per_scenario_per_censoring_share": 200,
        "attempted_replicates": 800,
        "nominal_level": 0.05,
        "nominal_coverage": 0.95,
        "monte_carlo_confidence": 0.95,
        "maximum_failed_replicates": 0,
        "required_test_rule": "mixture_50_50",
        "required_interval_level": 0.95,
        "fixture_sha256": TARGET_FIXTURE_SHA256,
        "synthetic_structure_matching": True,
        "participant_structure_reconstructed": False,
    }
    assert manifest["scientific_pass_rules_configured"] is True
    assert analysis["pass_rules_configured"] is True
