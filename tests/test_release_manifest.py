"""Public contract tests for Asterism release metadata."""

import json
import re
import subprocess
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from asterism import _core

from tools import check_release as release_checker
from tools.check_release import (
    conditional_quantity_errors,
    fixed_build_errors,
    known_limitation_errors,
    medusa_smoke_configuration_errors,
)
from tools.run_scientific_release import scientific_inventory_errors

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root used by release tooling."""


def test_built_extension_embeds_the_authoritative_release_manifest() -> None:
    """Require the wheel to retain the exact committed release contract."""
    manifest_text: str = (ROOT / "release.toml").read_text(encoding="utf-8")
    """Read the one editable copy of the release manifest."""

    assert _core.__release_manifest__ == manifest_text


def test_release_metadata_agrees_with_the_authoritative_manifest() -> None:
    """Require every public version surface to carry one release identity."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "tools/check_release.py", "--metadata"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    """Ran the same metadata check used by local and hosted quality gates."""

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_stable_abi_starts_at_the_supported_python_floor() -> None:
    """Do not label wheels as compatible with unsupported Python versions."""
    cargo: dict[str, object] = tomllib.loads(
        (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    )
    """Parsed the Rust dependency features which determine the ABI3 wheel tag."""

    dependencies: dict[str, object] = cast(dict[str, object], cargo["dependencies"])
    """Selected Cargo dependency metadata."""

    pyo3: dict[str, object] = cast(dict[str, object], dependencies["pyo3"])
    """Selected PyO3's stable-ABI configuration."""

    assert pyo3["features"] == ["abi3-py313"]


def test_release_manifest_owns_the_exact_rust_toolchain() -> None:
    """Require manifest, Cargo, and repository compiler selection to agree."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative compiler identity selected for release evidence."""

    cargo: dict[str, object] = tomllib.loads(
        (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    )
    """Read Cargo's minimum accepted compiler release."""

    toolchain: dict[str, object] = tomllib.loads(
        (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    )
    """Read the exact compiler selected by local and hosted Rust commands."""

    expected: object = manifest["rust_toolchain"]
    """Selected the one authoritative Rust compiler version."""

    assert expected == "1.97.1"
    assert cargo["package"]["rust-version"] == expected  # type: ignore[index]
    assert toolchain["toolchain"]["channel"] == expected  # type: ignore[index]


def test_release_manifest_owns_exact_python_build_tools() -> None:
    """Require Maturin and uv metadata to match locks and hosted execution."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read authoritative Maturin and uv versions."""

    pyproject: dict[str, object] = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    """Read exact build-system and development requirements."""

    lock: dict[str, object] = tomllib.loads(
        (ROOT / "uv.lock").read_text(encoding="utf-8")
    )
    """Read the resolved Maturin package identity."""

    maturin: object = manifest["maturin_version"]
    """Selected the one authoritative wheel-builder version."""

    uv_version: object = manifest["uv_version"]
    """Selected the one authoritative workflow package-manager version."""

    assert maturin == "1.14.1"
    assert uv_version == "0.11.21"
    assert pyproject["build-system"]["requires"] == [  # type: ignore[index]
        f"maturin=={maturin}"
    ]
    assert f"maturin=={maturin}" in pyproject["dependency-groups"]["dev"]  # type: ignore[index]
    locked_maturin: list[dict[str, object]] = [
        package
        for package in lock["package"]  # type: ignore[union-attr]
        if package["name"] == "maturin"
    ]
    """Selected the sole resolved builder package from the complete lock."""

    assert [package["version"] for package in locked_maturin] == [maturin]
    for workflow_name in ("quality.yml", "wheels.yml", "release.yml"):
        workflow: str = (ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        """Read one hosted build surface that installs uv explicitly."""

        assert f'version: "{uv_version}"' in workflow
    wheels: str = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
    """Read the action-specific manylinux Maturin pin."""

    assert f"maturin-version: v{maturin}" in wheels


def test_scientific_gate_inventory_covers_every_required_check_once() -> None:
    """Require one participant-free command or explicit blocker for every claim."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the release manifest without treating configuration as run evidence."""

    required: list[tuple[str, str]] = [
        (analysis["id"], check)
        for analysis in manifest["analyses"]
        for check in analysis["required_checks"]
    ]
    """Expanded all supported scientific claims with analysis ownership."""

    configured: list[tuple[str, str]] = [
        (analysis["id"], rule["id"])
        for analysis in manifest["analyses"]
        for rule in analysis["pass_rules"]
    ]
    """Expanded the exact executable or fail-closed command inventory."""

    assert len(required) == 32
    assert configured == required
    assert len(configured) == len(set(configured))
    assert scientific_inventory_errors(manifest, ROOT) == []


def test_scientific_gate_does_not_write_dated_component_evidence() -> None:
    """Keep the fixed release checkout clean until final verification."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the exact command inventory used by the scientific runner."""

    analysis: dict[str, Any] = next(
        item
        for item in manifest["analyses"]
        if item["id"] == "several_covariance_components"
    )
    """Selected the supported several-component analysis."""

    rule: dict[str, Any] = next(
        item
        for item in analysis["pass_rules"]
        if item["id"] == "components_target_design"
    )
    """Selected the one command that historically wrote dated evidence."""

    assert rule["command"] == ["checks/components_target_design.py", "--no-write"]


def test_release_inventory_is_complete_but_not_its_own_run_evidence() -> None:
    """Keep executable inventory distinct from measured passing run evidence."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read every global and per-analysis readiness switch."""

    assert manifest["scientific_pass_rules_configured"] is True
    assert manifest["release"] is True
    assert all(
        analysis["pass_rules_configured"] is True for analysis in manifest["analyses"]
    )
    statuses: set[str] = {
        rule["status"]
        for analysis in manifest["analyses"]
        for rule in analysis["pass_rules"]
    }
    """Collected command readiness without treating it as run evidence."""

    assert statuses == {"ready"}
    """Required every command to be runnable while leaving its result external."""


def test_censored_component_target_failure_is_a_narrow_known_limitation() -> None:
    """Keep one failed target visible without turning it into a global gate."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative scientific command inventory."""

    analysis: dict[str, Any] = next(
        item
        for item in manifest["analyses"]
        if item["id"] == "one_trait_censored_components"
    )
    """Selected the several-component censored analysis."""

    assert analysis["support"] == "supported_in_0_2"
    assert "asymptotic_test" in analysis["supported_quantities"]
    assert "bootstrap_test" not in analysis["supported_quantities"]
    assert {measurement["quantity"] for measurement in analysis["measured_levels"]} == {
        "asymptotic_test"
    }
    assert "censored_components_target_design" not in analysis["required_checks"]
    assert [rule["id"] for rule in analysis["pass_rules"]] == analysis[
        "required_checks"
    ]

    limitation: dict[str, Any] = analysis["known_limitations"][0]
    """Selected the exact negative result retained beside the support claim."""

    assert limitation == {
        "code": "ASYMPTOTIC_TEST_NOT_CALIBRATED_FOR_FIXED_TARGET_75",
        "quantity": "asymptotic_test",
        "scope": "fixed_instrument_1909_people_four_components_75_percent",
        "consequence": "withhold_p_value_without_design_specific_simulated_null",
        "evidence": ("evidence/censored-components-target-limitation-2026-09-01.json"),
        "evidence_sha256": (
            "14c839c82abb2e65f52988abd811bfa6a2a347efd2c7b43bc3c0c64a05d1128c"
        ),
    }

    record: dict[str, Any] = json.loads(
        (ROOT / limitation["evidence"]).read_text(encoding="utf-8")
    )
    """Read the participant-free retained failure record named by the contract."""

    assert record["outcome"] == "known_limitation"
    producer: dict[str, Any] = record["producer"]
    """Selected the exact historical build and its preserved source identity."""

    assert producer["source_commit"] == "bd3e09ed0b4985533574f5621489915d3ba2f156"
    assert producer["cargo_lock_sha256"] == (
        "20a2ce42b4f208347f0f7296c02345cf6b920149e78d48f50c7e57891e27d3c9"
    )
    assert producer["uv_lock_sha256"] == (
        "0bcb42d0252c592488b4df9c9f9c467474d1b518835c21a374e959efd28d6f37"
    )
    assert producer["source_bundle_sha256"] == (
        "237e5f007cf0ca9fc85815dc1c4ecc12bc0da01c1d66bab7686a46aab0e960fd"
    )
    assert producer["preserved_source_ref"] == (
        "refs/evidence/censored-components-target-bd3e09e"
    )
    assert producer["comparison_to_candidate"] == {
        "source_commit": "162341f12f9029d946770887bf85c0956ca0cbd3",
        "unchanged_paths": ["src/tobit.rs", "Cargo.lock", "uv.lock"],
        "target_script_difference": (
            "The reported LRT atom aggregation uses the retained in_lrt_atom "
            "field instead of exact-zero test-statistic equality."
        ),
    }
    assert record["decision"]["universal_censoring_threshold"] is None
    assert record["decision"]["fit_and_intervals_affected"] is False
    failed: dict[str, Any] = next(
        cell
        for cell in record["cells"]
        if cell["scenario"] == "null" and cell["expected_censoring_share"] == 0.75
    )
    """Selected the exact failed target cell rather than a censoring range."""
    assert failed["attempts"] == 200
    assert failed["rejections"] == 19
    assert failed["rejection_rate"] == 0.095
    assert failed["one_sided_lower_95"] == 0.06310551496161299
    assert failed["passed"] is False
    assert release_checker.measured_level_errors(analysis) == []
    assert known_limitation_errors(analysis, ROOT) == []


def test_bootstrap_limitation_counts_outer_requests_not_inner_failures() -> None:
    """Do not turn eleven failed requests into a count of failed inner fits."""
    surfaces: tuple[Path, ...] = (
        ROOT / "release.toml",
        ROOT / "python/asterism/models.py",
        ROOT / "docs/api-reference.md",
        ROOT / "docs/statistical-methods.md",
        ROOT / "docs/development.md",
    )
    """Named every public surface that explains the retained bootstrap failure."""

    inaccurate: re.Pattern[str] = re.compile(
        r"eleven\s+inner(?:-fit)?(?:\s+fits?)?\s+fail", re.IGNORECASE
    )
    """Matched claims that eleven is a known count of failed inner fits."""

    for path in surfaces:
        assert inaccurate.search(path.read_text(encoding="utf-8")) is None, path


def test_measured_levels_reject_an_undeclared_quantity() -> None:
    """Keep measurements inside the analysis's supported quantity inventory."""
    malformed: dict[str, Any] = {
        "id": "example",
        "supported_quantities": ["estimate"],
        "measured_levels": [{"quantity": "test", "measured": 0.05}],
    }
    """Named a measurement that its analysis never promises to return."""

    assert release_checker.measured_level_errors(malformed) == [
        "release.toml: example measured level 0 names an unsupported quantity"
    ]


def test_known_limitation_metadata_fails_closed() -> None:
    """Require a retained limitation to name real evidence and a known quantity."""
    malformed: dict[str, Any] = {
        "id": "example",
        "supported_quantities": ["estimate"],
        "known_limitations": [
            {
                "code": "lowercase code",
                "quantity": "missing_quantity",
                "scope": "target",
                "consequence": "withhold",
                "evidence": "evidence/absent.json",
                "evidence_sha256": "0" * 64,
            }
        ],
    }
    """Built a limitation that violates code, quantity and evidence rules."""

    errors: list[str] = known_limitation_errors(malformed, ROOT)
    """Collected every independent failure from the malformed record."""

    assert any("stable uppercase code" in error for error in errors)
    assert any("unsupported quantity" in error for error in errors)
    assert any("evidence does not exist" in error for error in errors)


def test_known_limitation_rejects_a_stale_evidence_digest() -> None:
    """Bind negative evidence as tightly as a passing scientific record."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative limitation and its current digest."""

    analysis: dict[str, Any] = next(
        item
        for item in manifest["analyses"]
        if item["id"] == "one_trait_censored_components"
    )
    """Selected the several-component censored support record."""

    stale: dict[str, Any] = deepcopy(analysis)
    """Copied the record so the authoritative manifest remains unchanged."""

    stale["known_limitations"][0]["evidence_sha256"] = "0" * 64
    """Introduced one deliberately stale immutable evidence commitment."""

    assert any(
        "evidence SHA-256 does not match" in error
        for error in known_limitation_errors(stale, ROOT)
    )


def test_component_separation_rule_is_bound_to_the_corrected_campaign() -> None:
    """Tie the measured design facts to the retained four-cell receipt."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative rule and its immutable evidence binding."""

    analysis: dict[str, Any] = next(
        item
        for item in manifest["analyses"]
        if item["id"] == "one_trait_censored_components"
    )
    """Selected the several-component censored support record."""

    rule: dict[str, Any] = next(
        item for item in analysis["pass_rules"] if item["id"] == "component_separation"
    )
    """Selected the exact corrected component-separation rule."""

    facts: dict[str, Any] = rule["design_facts"]
    """Read the predeclared design and retained campaign counts."""

    evidence_path: Path = ROOT / facts["evidence"]
    """Resolved the repository evidence path committed by the manifest."""

    assert facts["measured"] is True
    assert facts["requested_fits"] == 800
    assert facts["completed_fits"] == 799
    assert facts["evidence_sha256"] == release_checker.sha256(evidence_path)

    record: dict[str, Any] = json.loads(evidence_path.read_text(encoding="utf-8"))
    """Parsed the hashed result only after its byte identity matched."""

    assert record["passed"] is True
    assert record["failures"] == []
    assert record["replicates_per_cell"] == 200
    assert record["censoring_share"] == [0.52, 0.75]
    assert record["producer"]["source_commit"] == (
        "162341f12f9029d946770887bf85c0956ca0cbd3"
    )
    assert record["producer"]["source_dirty"] is False
    assert record["producer"]["slurm_job"] == 7347095
    assert record["producer"]["raw_receipt_sha256"] == (
        "dc78bae36bed73088067f81430ffc3ba5ef31ccf457e1a41ae6c65156364987f"
    )
    assert set(record["cells"]) == {
        "share=0.52/records=2/families=30",
        "share=0.52/records=2/families=60",
        "share=0.75/records=2/families=30",
        "share=0.75/records=2/families=60",
    }
    assert sorted(cell["replicates_fitted"] for cell in record["cells"].values()) == [
        199,
        200,
        200,
        200,
    ]
    for name, cell in record["cells"].items():
        if name.endswith("families=60"):
            assert abs(cell["additive_bias"]) <= cell["additive_bias_resolvable_beyond"]
            assert abs(cell["person_bias"]) <= cell["person_bias_resolvable_beyond"]


def test_mixed_bivariate_coverage_rule_records_completed_fixed_threshold_run() -> None:
    """Keep the corrected mixed-pair coverage result visible in the manifest."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative scientific command inventory."""

    analysis: dict[str, Any] = next(
        item
        for item in manifest["analyses"]
        if item["id"] == "mixed_binary_censored_genetic_correlation"
    )
    """Selected the already-supported mixed-pair analysis."""

    rule: dict[str, Any] = next(
        item
        for item in analysis["pass_rules"]
        if item["id"] == "mixed_bivariate_coverage"
    )
    """Selected its corrected fixed-threshold coverage gate."""

    assert rule["status"] == "ready"
    assert "blocker" not in rule
    assert rule["design_facts"]["measured"] is True
    assert rule["design_facts"]["observation_thresholds_fixed_before_outcomes"] is True


def test_cross_platform_agreement_covers_every_supported_analysis() -> None:
    """Retain measured tolerances for every supported installed-wheel probe."""
    manifest: dict[str, object] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative release contract."""

    agreement: dict[str, object] = cast(
        dict[str, object], manifest["cross_platform_agreement"]
    )
    """Selected the separately gated Mac/Linux comparison contract."""

    assert agreement["configured"] is True
    assert agreement["probe_runner"] == "tools/cross_platform_probe.py"
    assert agreement["runner"] == "tools/compare_cross_platform.py"
    assert agreement["exact_comparisons"] == [
        "outcome_status",
        "refusal_code",
        "boundary_state",
        "field_presence",
    ]
    tolerances: list[dict[str, object]] = agreement["model_tolerances"]  # type: ignore[assignment]
    """Read the complete measured numeric-field inventories."""

    assert {tolerance["analysis_id"] for tolerance in tolerances} == {
        analysis["id"]
        for analysis in manifest["analyses"]  # type: ignore[union-attr]
    }
    assert all(tolerance["measured"] is True for tolerance in tolerances)
    assert all(tolerance["numeric_fields"] for tolerance in tolerances)
    assert all(
        set(tolerance["absolute"]) == set(tolerance["numeric_fields"])
        and set(tolerance["relative"]) == set(tolerance["numeric_fields"])
        for tolerance in tolerances
    )


def test_medusa_smoke_requires_external_final_wheel_evidence() -> None:
    """Enable final-wheel smoke without pointing at stale repository evidence."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the final manifest without consulting old evidence."""

    smoke: dict[str, Any] = manifest["medusa_smoke"]
    """Selected the target-host requirement independently of its future result."""

    assert smoke == {
        "required": True,
        "configured": True,
        "architecture": "x86_64",
        "glibc_version": "2.28",
        "command": ["tools/medusa_wheel_smoke.py"],
    }


def test_medusa_release_configuration_never_points_at_repository_evidence() -> None:
    """A final release requires an external input, not a committed JSON pointer."""
    configured: dict[str, Any] = {
        "required": True,
        "configured": True,
        "architecture": "x86_64",
        "glibc_version": "2.28",
        "command": ["tools/medusa_wheel_smoke.py"],
    }
    """Represented the manifest state after exact final-wheel smoke exists externally."""

    assert medusa_smoke_configuration_errors(configured) == []
    assert "unverified" in "\n".join(
        medusa_smoke_configuration_errors({**configured, "configured": False})
    )


def test_conditional_supported_sets_partition_the_component_inventory() -> None:
    """Select one honest component-reporting mode instead of requiring both."""
    manifest: dict[str, Any] = tomllib.loads(
        (ROOT / "release.toml").read_text(encoding="utf-8")
    )
    """Read the authoritative analysis inventory and conditional selectors."""

    component: dict[str, Any] = next(
        analysis
        for analysis in manifest["analyses"]
        if analysis["id"] == "several_covariance_components"
    )
    """Selected the analysis with mutually exclusive reporting conventions."""

    assert conditional_quantity_errors(component) == []
    assert component["supported_quantity_sets"] == [
        {
            "when": {"component_reporting": "mean_diagonal"},
            "quantities": [
                "mean_diagonal_component_contributions",
                "mean_diagonal_proportions",
                "mean_diagonal_proportion_interval",
            ],
        },
        {
            "when": {"component_reporting": "zero_diagonal"},
            "quantities": [
                "coefficients_for_zero_diagonal_bases",
                "contrasts_for_zero_diagonal_bases",
            ],
        },
    ]

    malformed: dict[str, Any] = deepcopy(component)
    """Copied the contract so one duplicate cannot mutate the authoritative parse."""

    malformed["supported_quantity_sets"][1]["quantities"].append(
        "mean_diagonal_proportions"
    )
    """Made one field falsely required by both mutually exclusive modes."""

    assert "must partition supported_quantities" in "\n".join(
        conditional_quantity_errors(malformed)
    )


def test_fixed_build_gate_refuses_stale_or_dirty_extension_identity() -> None:
    """Require release wheels to come from the checked-out commit and a clean tree."""
    core: Any = SimpleNamespace(
        __source_commit__="b" * 40,
        __source_dirty__=True,
    )
    """Represented a wheel built from another commit with local modifications."""

    errors: list[str] = fixed_build_errors(
        core,
        repository_commit="a" * 40,
        repository_dirty=False,
    )
    """Compared immutable wheel identity with the release checkout identity."""

    assert "compiled source commit" in errors[0]
    assert "compiled source is dirty" in errors[1]


def test_fixed_build_gate_accepts_current_clean_extension_identity() -> None:
    """Permit the identity gate only for the exact clean release commit."""
    core: Any = SimpleNamespace(
        __source_commit__="a" * 40,
        __source_dirty__=False,
    )
    """Represented a clean wheel built from the checked-out release commit."""

    assert (
        fixed_build_errors(
            core,
            repository_commit="a" * 40,
            repository_dirty=False,
        )
        == []
    )


def test_fixed_build_gate_independently_refuses_a_dirty_release_checkout() -> None:
    """Do not let build-script environment overrides certify mutable source."""
    core: Any = SimpleNamespace(
        __source_commit__="a" * 40,
        __source_dirty__=False,
    )
    """Represented an apparently clean wheel matching the current commit."""

    errors: list[str] = fixed_build_errors(
        core,
        repository_commit="a" * 40,
        repository_dirty=True,
    )
    """Compared that build metadata with an independently dirty Git checkout."""

    assert errors == [
        "release checkout is dirty after excluding ignored release outputs"
    ]
