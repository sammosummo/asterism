"""Contracts for the several-component censored target-design command."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.stats import norm

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the standalone scientific command."""


def target_module() -> ModuleType:
    """Load the standalone command without running its campaign."""
    path: Path = ROOT / "checks" / "censored_components_target_design.py"
    """Located the standalone target command."""

    specification: ModuleSpec | None = importlib.util.spec_from_file_location(
        "censored_components_target_design", path
    )
    """Built an import specification without executing the command's main."""

    assert specification is not None and specification.loader is not None

    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the temporary module named by the specification."""

    sys.modules[specification.name] = module
    """Registered it while its dataclasses resolved their module identity."""

    checks_directory: str = str(path.parent)
    """Located direct-command imports used by the standalone script."""

    sys.path.insert(0, checks_directory)
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(checks_directory)
    return module


def valid_asymptotic_null_row(
    module: ModuleType,
    *,
    share: float = 0.52,
    replicate: int = 0,
) -> dict[str, object]:
    """Make one internally consistent persisted asymptotic null result."""
    return {
        "scenario": "null",
        "censoring_share": share,
        "replicate": replicate,
        "worst_censored_block": 100,
        "achieved_censoring_share": 0.5,
        "censoring_limit": module.expected_censoring_limits()[share],
        "p_value": 0.05,
        "test_statistic": 2.705543454095404,
        "test_rule": "asymptotic_mixture_50_50",
        "rejected": True,
        "nuisance_at_bound": False,
        "in_lrt_atom": False,
    }


def valid_bootstrap_null_row(
    module: ModuleType,
    *,
    share: float = 0.52,
    replicate: int = 0,
    bootstrap_replicates: int = 999,
) -> dict[str, object]:
    """Make one internally consistent persisted null result."""
    exceedances: int = 49
    """Put the add-one result exactly at alpha under the 999-draw rule."""

    return {
        "scenario": "null",
        "censoring_share": share,
        "replicate": replicate,
        "worst_censored_block": 100,
        "achieved_censoring_share": 0.5,
        "censoring_limit": module.expected_censoring_limits()[share],
        "p_value": (1.0 + exceedances) / (bootstrap_replicates + 1.0),
        "test_statistic": 1.0,
        "test_rule": "parametric_bootstrap_add_one",
        "rejected": True,
        "nuisance_at_bound": False,
        "in_lrt_atom": False,
        "bootstrap_exceedances": exceedances,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_requested": bootstrap_replicates,
        "bootstrap_seed": (
            module.BOOTSTRAP_BASE_SEED + 7_919 * replicate + 104_729 * int(share * 100)
        ),
    }


def valid_heritable_row(
    module: ModuleType,
    *,
    share: float = 0.52,
    replicate: int = 0,
) -> dict[str, object]:
    """Make one internally consistent persisted interval result."""
    return {
        "scenario": "heritable",
        "censoring_share": share,
        "replicate": replicate,
        "worst_censored_block": 100,
        "achieved_censoring_share": 0.5,
        "censoring_limit": module.expected_censoring_limits()[share],
        "estimate": 0.35,
        "lower": 0.10,
        "upper": 1.0,
        "upper_limited": True,
        "contains_upper_bound": None,
        "profile_failures": 1,
        "bound_unreachable": True,
        "covered": True,
        "extra_failures": 0,
    }


def valid_producer() -> dict[str, object]:
    """Make one exact non-gating checkpoint producer record."""
    commit: str = "a" * 40
    """Named the modelled clean source commit."""

    manifest: str = "b" * 64
    """Named the modelled embedded release-manifest digest."""

    return {
        "source_commit": commit,
        "release_manifest_sha256": manifest,
        "extension_sha256": "c" * 64,
        "build_identity": {
            "source_commit": commit,
            "source_dirty": False,
            "release_manifest_sha256": manifest,
        },
    }


def shard_producer(
    shard_index: int,
    shard_count: int,
    *,
    commit_character: str = "a",
) -> dict[str, object]:
    """Make one exact producer tied to an operational scheduler partition."""
    producer: dict[str, object] = valid_producer()
    """Started from the ordinary valid build provenance fixture."""

    commit: str = commit_character * 40
    """Allowed a test to model a different exact native producer."""

    producer["source_commit"] = commit
    """Made the top-level producer advertise the selected source commit."""

    build: dict[str, object] = dict(producer["build_identity"])
    """Copied the nested build identity before changing its source commit."""

    build["source_commit"] = commit
    """Kept the nested public build identity consistent with its producer."""

    producer["build_identity"] = build
    """Replaced the fixture's nested identity with its updated copy."""

    producer["execution"] = {
        "allocated_cpu_count": 128,
        "outer_workers": 1,
        "bootstrap_threads_per_outer_worker": None,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    """Recorded the operational CPU and scheduler partition allocation."""

    return producer


def valid_row_for_job(
    module: ModuleType,
    job: tuple[str, float, int],
) -> dict[str, object]:
    """Make the strict persisted row matching one outer-grid coordinate."""
    scenario, share, replicate = job
    """Read the coordinate whose strict row fixture is required."""

    return (
        valid_asymptotic_null_row(module, share=share, replicate=replicate)
        if scenario == "null"
        else valid_heritable_row(module, share=share, replicate=replicate)
    )


def write_shard_fixture(
    module: ModuleType,
    path: Path,
    signature: dict[str, object],
    jobs: list[tuple[str, float, int]],
    shard_index: int,
    shard_count: int,
    *,
    commit_character: str = "a",
    omit_last_row: bool = False,
) -> Path:
    """Write one structurally valid complete or deliberately partial shard."""
    allocation: dict[str, int] = module.shard_allocation(shard_index, shard_count)
    """Built the operational checkpoint header under test."""

    selected: list[tuple[str, float, int]] = module.select_shard_jobs(
        jobs,
        shard_index,
        shard_count,
    )
    """Recomputed the exact deterministic coordinate subset for this file."""

    if omit_last_row:
        selected = selected[:-1]
        """Made the requested fixture incomplete without inventing a coordinate."""

    rows: list[dict[str, object]] = [valid_row_for_job(module, job) for job in selected]
    """Made every retained result agree with this shard's declared coordinates."""

    history: list[dict[str, object]] = [
        {
            "producer": shard_producer(
                shard_index,
                shard_count,
                commit_character=commit_character,
            ),
            "coordinates": [list(job) for job in selected],
        }
    ]
    """Attributed every retained coordinate to this shard's exact producer."""
    module.write_checkpoint(
        path,
        signature,
        rows,
        history,
        shard=allocation,
    )
    return path


def test_instrument_limit_is_fixed_before_any_outcome_is_drawn() -> None:
    """Calibrate the intended marginal censoring share from design facts alone."""
    module: ModuleType = target_module()
    """Loaded the solver through the command under test."""

    design: np.ndarray = np.asarray([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    """Declared three pre-outcome fixed-effect rows."""

    coefficients: np.ndarray = np.asarray([2.0, 0.5])
    """Declared the mean model before any random response."""

    means: np.ndarray = design @ coefficients
    """Computed the three marginal latent means."""

    limit: float = module.right_censoring_limit(means, 4.0, 0.75)
    """Solved the common instrument limit from those design facts."""

    expected: float = float(np.mean(1.0 - norm.cdf((limit - means) / 2.0)))
    """Recomputed its expected share independently from normal tails."""

    assert expected == pytest.approx(0.75, abs=1e-12)


def test_boundary_atom_is_diagnostic_and_not_a_finite_sample_gate() -> None:
    """Keep the empirical LRT atom without promoting its asymptotic weight."""
    module: ModuleType = target_module()
    """Loaded the target rule constants and source."""

    assert module.DEFAULT_TEST_REFERENCE == "asymptotic"
    assert module.ASYMPTOTIC_TEST_RULE == "asymptotic_mixture_50_50"
    assert module.BOOTSTRAP_TEST_RULE == "parametric_bootstrap_add_one"
    assert module.BOOTSTRAP_REPLICATES == 999
    assert not hasattr(module, "EXPECTED_BOUND_SHARE")

    source: str = (ROOT / "checks" / "censored_components_target_design.py").read_text(
        encoding="utf-8"
    )
    """Read the scoring path to guard against restoring the retired gate."""

    assert "share_in_lrt_atom" in source
    assert "share_on_bound_expected" not in source
    assert 'bool(row["in_lrt_atom"])' in source
    assert 'float(row["test_statistic"]) == 0.0' not in source


def test_target_cell_requires_the_selected_reference_rule() -> None:
    """Make each selected rule identity part of checkpoint validation."""
    module: ModuleType = target_module()
    """Loaded the target rule validator."""

    module.require_test_rule(
        {"rule": "asymptotic_mixture_50_50"},
        "asymptotic",
        None,
    )
    module.require_test_rule(
        {
            "rule": "parametric_bootstrap_add_one",
            "replicates": 999,
            "requested": 999,
        },
        "bootstrap",
        999,
    )
    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_TEST_RULE_MISMATCH",
    ):
        module.require_test_rule(
            {"rule": "parametric_bootstrap_add_one"},
            "asymptotic",
            None,
        )


@pytest.mark.parametrize("field", ["replicates", "requested"])
def test_target_cell_refuses_an_incomplete_bootstrap(field: str) -> None:
    """Require both reported denominators to equal the configured inner grid."""
    module: ModuleType = target_module()
    """Loaded the target rule validator."""

    test: dict[str, object] = {
        "rule": "parametric_bootstrap_add_one",
        "replicates": 999,
        "requested": 999,
    }
    """Made a complete candidate bootstrap record."""

    test[field] = 998
    """Made one required count disagree with the configured denominator."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_BOOTSTRAP_INCOMPLETE",
    ):
        module.require_test_rule(test, "bootstrap", 999)


def test_target_uses_inclusive_rejection_and_one_sided_validity_rules() -> None:
    """Count p equal to alpha and refuse only measured anti-conservatism."""
    source: str = (ROOT / "checks" / "censored_components_target_design.py").read_text(
        encoding="utf-8"
    )
    """Read the executable scoring contract."""

    assert 'test["p_value"] <= LEVEL' in source
    assert "level_decision(" in source
    assert "coverage_decision(" in source


def test_shard_partition_covers_the_grid_once_without_changing_signature() -> None:
    """Partition the declared order deterministically, completely and disjointly."""
    module: ModuleType = target_module()
    """Loaded the operational partition helpers and scientific signature."""

    jobs: list[tuple[str, float, int]] = [
        (scenario, share, replicate)
        for scenario in ("null", "heritable")
        for share in module.CENSORING_SHARES
        for replicate in range(7)
    ]
    """Made an uneven grid so shard sizes need not happen to match."""

    partitions: list[list[tuple[str, float, int]]] = [
        module.select_shard_jobs(jobs, shard_index, 10) for shard_index in range(10)
    ]
    """Selected all ten independent scheduler partitions."""

    flattened: list[tuple[str, float, int]] = [
        job for partition in partitions for job in partition
    ]
    """Collected every selected coordinate across the full array."""

    assert len(flattened) == len(jobs)
    assert len(set(flattened)) == len(jobs)
    assert set(flattened) == set(jobs)
    assert partitions == [
        [job for position, job in enumerate(jobs) if position % 10 == shard_index]
        for shard_index in range(10)
    ]

    signature: dict[str, object] = module.reusable_checkpoint_signature(
        ROOT,
        7,
        "asymptotic",
    )
    """Built the one common scientific identity used by all ten partitions."""

    assert "shard_index" not in signature
    assert "shard_count" not in signature
    paths: list[Path] = [
        module.shard_checkpoint_path(Path("target.json"), index, 10)
        for index in range(10)
    ]
    """Derived distinct checkpoint names from one common array-task base."""

    assert len(set(paths)) == 10
    assert paths[0] == Path("target.shard-000-of-010.json")
    assert paths[-1] == Path("target.shard-009-of-010.json")


def test_checkpoint_rows_require_unique_complete_coordinates_for_their_mode() -> None:
    """Reject duplicates, missing cells and rows from another reference mode."""
    module: ModuleType = target_module()
    """Loaded the resumable campaign validators."""

    jobs: list[tuple[str, float, int]] = [
        ("null", 0.52, 0),
        ("heritable", 0.52, 0),
    ]
    """Declared a minimal two-cell outer grid."""

    null: dict[str, object] = valid_asymptotic_null_row(module)
    """Made one complete ordinary analytic null result."""

    heritable: dict[str, object] = valid_heritable_row(module)
    """Made the matching interval coordinate."""

    module.validate_campaign_rows(
        [null, heritable],
        jobs,
        "asymptotic",
        None,
        require_complete=True,
    )

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKPOINT_COORDINATE_DUPLICATE",
    ):
        module.validate_campaign_rows(
            [null, null, heritable],
            jobs,
            "asymptotic",
            None,
            require_complete=True,
        )

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKPOINT_COORDINATES_MISSING",
    ):
        module.validate_campaign_rows(
            [null],
            jobs,
            "asymptotic",
            None,
            require_complete=True,
        )

    bootstrap: dict[str, object] = valid_bootstrap_null_row(module)
    """Made one complete optional-bootstrap row."""

    shortened: dict[str, object] = dict(bootstrap)
    """Copied the valid row before shortening one reported count."""

    shortened["bootstrap_replicates"] = 998
    """Made the completed count disagree with the predeclared inner grid."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_BOOTSTRAP_INCOMPLETE",
    ):
        module.validate_campaign_rows(
            [shortened],
            [("null", 0.52, 0)],
            "bootstrap",
            999,
            require_complete=True,
        )

    with pytest.raises(ValueError):
        module.validate_campaign_rows(
            [bootstrap],
            [("null", 0.52, 0)],
            "asymptotic",
            None,
            require_complete=True,
        )


def test_checkpoint_round_trip_is_atomic_and_rejects_stale_signatures(
    tmp_path: Path,
) -> None:
    """Resume exact rows only under the design and implementation that made them."""
    module: ModuleType = target_module()
    """Loaded the checkpoint reader and writer."""

    checkpoint: Path = tmp_path / "campaign.json"
    """Placed the operational checkpoint outside repository evidence."""

    signature: dict[str, object] = {
        "campaign": "test",
        "test_reference": "asymptotic",
    }
    """Represented one exact campaign identity."""

    jobs: list[tuple[str, float, int]] = [("heritable", 0.52, 0)]
    """Declared one outer coordinate for the round trip."""

    rows: list[dict[str, object]] = [valid_heritable_row(module)]
    """Made one completed interval row."""

    history: list[dict[str, object]] = [
        {
            "producer": valid_producer(),
            "coordinates": [["heritable", 0.52, 0]],
        }
    ]
    """Attributed that row to one exact, non-gating producer build."""

    module.write_checkpoint(checkpoint, signature, rows, history)
    assert module.read_checkpoint(
        checkpoint,
        signature,
        jobs,
        "asymptotic",
        None,
    ) == (
        rows,
        history,
    )
    assert not list(tmp_path.glob(f".{checkpoint.name}.*.tmp"))

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKPOINT_STALE",
    ):
        module.read_checkpoint(
            checkpoint,
            {"campaign": "different"},
            jobs,
            "asymptotic",
            None,
        )

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKPOINT_STALE",
    ):
        module.read_checkpoint(
            checkpoint,
            signature,
            jobs,
            "bootstrap",
            999,
        )


def test_complete_shard_merge_recovers_the_full_grid_and_provenance(
    tmp_path: Path,
) -> None:
    """Merge all complete partitions into the ordinary full scoring inputs."""
    module: ModuleType = target_module()
    """Loaded the checkpoint partition and merge implementation."""

    jobs: list[tuple[str, float, int]] = [
        (scenario, 0.52, replicate)
        for scenario in ("null", "heritable")
        for replicate in range(2)
    ]
    """Declared a small complete grid split evenly across two partitions."""

    signature: dict[str, object] = {
        "campaign": "test",
        "test_reference": "asymptotic",
    }
    """Declared one common scientific campaign identity for both shards."""

    paths: list[Path] = [
        write_shard_fixture(
            module,
            tmp_path / f"shard-{index}.json",
            signature,
            jobs,
            index,
            2,
        )
        for index in range(2)
    ]
    """Wrote the two complete compatible operational checkpoints."""

    rows, history, shard_count = module.merge_shard_checkpoints(
        paths,
        signature,
        jobs,
        "asymptotic",
        None,
    )
    """Merged the validated inputs into ordinary full-campaign scoring values."""

    coordinates: set[tuple[str, float, int]] = {
        (row["scenario"], row["censoring_share"], row["replicate"]) for row in rows
    }
    """Recovered every scientific coordinate from the merged result rows."""

    assert coordinates == set(jobs)
    assert shard_count == 2
    assert len(history) == 2
    assert {
        invocation["producer"]["execution"]["shard_index"] for invocation in history
    } == {0, 1}


def test_shard_merge_refuses_duplicate_missing_and_partial_partitions(
    tmp_path: Path,
) -> None:
    """Refuse duplicate indices, an absent shard, or an incomplete shard grid."""
    module: ModuleType = target_module()
    """Loaded strict partition-set and row-completeness validation."""

    jobs: list[tuple[str, float, int]] = [
        (scenario, 0.52, replicate)
        for scenario in ("null", "heritable")
        for replicate in range(3)
    ]
    """Declared a grid with two rows in each of three partitions."""

    signature: dict[str, object] = {
        "campaign": "test",
        "test_reference": "asymptotic",
    }
    """Declared the common scientific identity shared by valid partitions."""

    paths: list[Path] = [
        write_shard_fixture(
            module,
            tmp_path / f"complete-{index}.json",
            signature,
            jobs,
            index,
            3,
        )
        for index in range(3)
    ]
    """Made one valid three-way partition set before corrupting its selection."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_MERGE_SHARD_DUPLICATE",
    ):
        module.merge_shard_checkpoints(
            [paths[0], paths[0]],
            signature,
            jobs,
            "asymptotic",
            None,
        )

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_MERGE_SHARDS_MISSING",
    ):
        module.merge_shard_checkpoints(
            [paths[0], paths[2]],
            signature,
            jobs,
            "asymptotic",
            None,
        )

    partial: Path = write_shard_fixture(
        module,
        tmp_path / "partial-1.json",
        signature,
        jobs,
        1,
        3,
        omit_last_row=True,
    )
    """Removed one required coordinate from the middle partition only."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKPOINT_COORDINATES_MISSING",
    ):
        module.merge_shard_checkpoints(
            [paths[0], partial, paths[2]],
            signature,
            jobs,
            "asymptotic",
            None,
        )


def test_shard_merge_refuses_stale_signature_and_incompatible_provenance(
    tmp_path: Path,
) -> None:
    """Do not combine rows from another campaign or another native producer."""
    module: ModuleType = target_module()
    """Loaded scientific identity and exact producer checks."""

    jobs: list[tuple[str, float, int]] = [
        ("null", 0.52, 0),
        ("heritable", 0.52, 0),
    ]
    """Declared one coordinate for each of two partitions."""

    signature: dict[str, object] = {
        "campaign": "test",
        "test_reference": "asymptotic",
    }
    """Declared the scientific identity the final merge expects."""

    first: Path = write_shard_fixture(
        module,
        tmp_path / "first.json",
        signature,
        jobs,
        0,
        2,
    )
    """Wrote the valid first partition used in both refusal cases."""

    stale: Path = write_shard_fixture(
        module,
        tmp_path / "stale.json",
        {"campaign": "different", "test_reference": "asymptotic"},
        jobs,
        1,
        2,
    )
    """Wrote the second partition under a different scientific signature."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKPOINT_STALE",
    ):
        module.merge_shard_checkpoints(
            [first, stale],
            signature,
            jobs,
            "asymptotic",
            None,
        )

    incompatible: Path = write_shard_fixture(
        module,
        tmp_path / "incompatible.json",
        signature,
        jobs,
        1,
        2,
        commit_character="d",
    )
    """Wrote a same-signature shard made by another exact native producer."""
    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_MERGE_PROVENANCE_INCOMPATIBLE",
    ):
        module.merge_shard_checkpoints(
            [first, incompatible],
            signature,
            jobs,
            "asymptotic",
            None,
        )


def test_campaign_signature_covers_scientific_sources_and_runtime() -> None:
    """Bind reusable rows to scientific code, locks, toolchain and runtime."""
    module: ModuleType = target_module()
    """Loaded the signature implementation."""

    relatives: set[str] = {
        path.relative_to(ROOT).as_posix() for path in module.campaign_source_paths(ROOT)
    }
    """Collected the exact transitive source paths included in its code digest."""

    assert {
        "Cargo.lock",
        "Cargo.toml",
        "pyproject.toml",
        "rust-toolchain.toml",
        "uv.lock",
        "src/tobit.rs",
        "src/tobit/python.rs",
        "python/asterism/models.py",
        "checks/censored_components_target_design.py",
        "checks/censoring_design.py",
        "checks/one_trait_coverage.py",
        "checks/sequential_against_ghk.py",
        "checks/tobit_target_design.py",
        "checks/design_fixtures/one_trait_gaussian_heritability.json",
    } <= relatives
    assert "release.toml" not in relatives
    assert "checks/component_separation.py" not in relatives
    assert "checks/external_fixtures/tobit_against_mcmcglmm.json" not in relatives

    signature: dict[str, object] = module.reusable_checkpoint_signature(
        ROOT,
        200,
        "asymptotic",
    )
    """Made the scientifically reusable local campaign identity."""

    assert "loaded_extension" not in signature
    assert "build" not in signature
    assert signature["test_reference"] == "asymptotic"
    assert "bootstrap_replicates" not in signature
    assert "bootstrap_threads" not in signature
    encoded: str = module.json.dumps(signature)
    """Flattened the reusable record to check forbidden build identities."""

    assert "source_commit" not in encoded
    assert "source_dirty" not in encoded
    assert "release.toml" not in encoded
    runtime: dict[str, str] = signature["runtime"]
    """Selected the persisted numerical runtime identity."""

    assert runtime["python"]
    assert runtime["python_implementation"]
    assert runtime["system"]
    assert runtime["machine"]
    assert runtime["platform"]
    assert runtime["numpy"]
    assert runtime["scipy"]


def make_fingerprint_root(root: Path) -> Path:
    """Create a minimal checkout-shaped scientific fingerprint fixture."""
    contents: dict[str, str] = {
        "Cargo.lock": "cargo lock\n",
        "Cargo.toml": "cargo manifest\n",
        "build.rs": "build input\n",
        "pyproject.toml": "python manifest\n",
        "release.toml": 'status = "blocked"\nblocker = "pending"\n',
        "rust-toolchain.toml": "toolchain\n",
        "uv.lock": "uv lock\n",
        "src/tobit.rs": "numerical rust\n",
        "python/asterism/models.py": "numerical python\n",
        "checks/censored_components_target_design.py": "target campaign\n",
        "checks/censoring_design.py": "scientific check\n",
        "checks/one_trait_coverage.py": "target envelope\n",
        "checks/sequential_against_ghk.py": "household construction\n",
        "checks/tobit_target_design.py": "target decisions\n",
        "checks/component_separation.py": "unrelated campaign\n",
        "checks/external_fixtures/tobit_against_mcmcglmm.json": "{}\n",
        "checks/design_fixtures/one_trait_gaussian_heritability.json": "{}\n",
    }
    """Declared every selected path plus deliberately excluded release metadata."""

    for relative, content in contents.items():
        path: Path = root / relative
        """Located this fixture file beneath the temporary checkout root."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root / "checks" / "design_fixtures" / "one_trait_gaussian_heritability.json"


def test_reusable_signature_ignores_release_metadata_and_unrelated_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep non-scientific edits from discarding multiweek target work."""
    module: ModuleType = target_module()
    """Loaded the reusable signature implementation."""

    fixture: Path = make_fingerprint_root(tmp_path)
    """Built the minimal checkout-shaped signature fixture."""

    monkeypatch.setattr(module, "FIXTURE_SHA256", module.file_sha256(fixture))
    runtime: dict[str, str] = {"python": "test", "platform": "test"}
    """Fixed the runtime so this test changes release metadata alone."""

    before: dict[str, object] = module.reusable_checkpoint_signature(
        tmp_path,
        200,
        "asymptotic",
        runtime=runtime,
    )
    """Captured the blocked campaign's reusable scientific identity."""

    (tmp_path / "release.toml").write_text(
        'status = "ready"\nblocker = "removed"\n',
        encoding="utf-8",
    )
    after: dict[str, object] = module.reusable_checkpoint_signature(
        tmp_path,
        200,
        "asymptotic",
        runtime=runtime,
    )
    """Recomputed it after only status and blocker metadata changed."""

    assert after == before

    for relative in (
        "checks/component_separation.py",
        "checks/external_fixtures/tobit_against_mcmcglmm.json",
    ):
        path: Path = tmp_path / relative
        """Located one unrelated check input deliberately outside this closure."""

        path.write_text("changed unrelated evidence\n", encoding="utf-8")
        assert (
            module.reusable_checkpoint_signature(
                tmp_path,
                200,
                "asymptotic",
                runtime=runtime,
            )
            == before
        )
    """Kept unrelated campaigns from discarding valid target-design rows."""


def test_reusable_signature_changes_with_science_locks_toolchain_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject any numerical, dependency, toolchain, runtime or parameter change."""
    module: ModuleType = target_module()
    """Loaded the reusable signature implementation."""

    fixture: Path = make_fingerprint_root(tmp_path)
    """Built the minimal checkout-shaped signature fixture."""

    monkeypatch.setattr(module, "FIXTURE_SHA256", module.file_sha256(fixture))
    runtime: dict[str, str] = {"python": "test", "platform": "test"}
    """Fixed the baseline numerical runtime identity."""

    baseline: dict[str, object] = module.reusable_checkpoint_signature(
        tmp_path,
        200,
        "asymptotic",
        runtime=runtime,
    )
    """Captured the initial scientific identity."""

    candidates: list[tuple[str, str]] = [
        ("src/tobit.rs", "changed numerical rust\n"),
        ("Cargo.lock", "changed cargo lock\n"),
        ("rust-toolchain.toml", "changed toolchain\n"),
        ("checks/censored_components_target_design.py", "changed target campaign\n"),
        ("checks/tobit_target_design.py", "changed target decisions\n"),
    ]
    """Named independent source, dependency and toolchain mutations."""

    for relative, changed in candidates:
        path: Path = tmp_path / relative
        """Located the one selected scientific input."""

        original: str = path.read_text(encoding="utf-8")
        """Retained its baseline bytes for the next independent mutation."""

        path.write_text(changed, encoding="utf-8")
        assert (
            module.reusable_checkpoint_signature(
                tmp_path,
                200,
                "asymptotic",
                runtime=runtime,
            )
            != baseline
        )
        path.write_text(original, encoding="utf-8")

    changed_runtime: dict[str, str] = {**runtime, "python": "different"}
    """Changed the runtime without touching checkout bytes."""

    assert (
        module.reusable_checkpoint_signature(
            tmp_path,
            200,
            "asymptotic",
            runtime=changed_runtime,
        )
        != baseline
    )
    bootstrap: dict[str, object] = module.reusable_checkpoint_signature(
        tmp_path,
        200,
        "bootstrap",
        999,
        runtime=runtime,
    )
    """Selected the optional finite-null mode with its complete inner grid."""

    assert bootstrap != baseline
    assert bootstrap["test_reference"] == "bootstrap"
    assert bootstrap["bootstrap_replicates"] == 999
    assert "bootstrap_threads" not in bootstrap
    assert (
        module.reusable_checkpoint_signature(
            tmp_path,
            200,
            "bootstrap",
            998,
            runtime=runtime,
        )
        != bootstrap
    )
    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_BOOTSTRAP_REPLICATES_NOT_APPLICABLE",
    ):
        module.reusable_checkpoint_signature(
            tmp_path,
            200,
            "asymptotic",
            999,
            runtime=runtime,
        )


def test_loaded_extension_must_match_the_current_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Refuse a stale binary even though binary identity is not persisted."""
    module: ModuleType = target_module()
    """Loaded the invocation-time extension validator."""

    cargo_lock: str = "cargo lock\n"
    """Declared the current checkout's Rust dependency lock."""

    manifest: str = 'status = "ready"\n'
    """Declared the current checkout's release metadata."""

    uv_lock: str = "uv lock\n"
    """Declared the current checkout's Python dependency lock."""

    (tmp_path / "Cargo.lock").write_text(cargo_lock, encoding="utf-8")
    (tmp_path / "release.toml").write_text(manifest, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(uv_lock, encoding="utf-8")
    commit: str = "a" * 40
    """Declared the clean checkout commit the extension must carry."""

    build: dict[str, object] = {
        "source_commit": commit,
        "source_dirty": False,
        "cargo_lock_sha256": module.file_sha256(tmp_path / "Cargo.lock"),
        "release_manifest_sha256": module.file_sha256(tmp_path / "release.toml"),
        "uv_lock_sha256": module.file_sha256(tmp_path / "uv.lock"),
    }
    """Modelled the public identity of an extension built from that checkout."""

    monkeypatch.setattr(module.asterism, "build_identity", lambda: dict(build))
    dirty: list[bool] = [False]
    """Allowed this test to turn the modelled checkout dirty later."""

    def fake_git_text(root: Path, arguments: list[str]) -> str:
        """Return the modelled checkout status or commit."""
        return (
            " M src/tobit.rs"
            if "status" in arguments and dirty[0]
            else ("" if "status" in arguments else commit)
        )

    monkeypatch.setattr(module, "git_text", fake_git_text)
    module.validate_loaded_extension(tmp_path)

    build["source_commit"] = "b" * 40
    """Made the loaded extension advertise a different source commit."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_EXTENSION_CHECKOUT_MISMATCH",
    ):
        module.validate_loaded_extension(tmp_path)

    build["source_commit"] = commit
    """Restored the matching source commit before testing another mismatch."""

    build["release_manifest_sha256"] = module.hashlib.sha256(
        b'status = "blocked"\n'
    ).hexdigest()
    """Modelled a binary not rebuilt after a metadata-only manifest transition."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_EXTENSION_CHECKOUT_MISMATCH",
    ):
        module.validate_loaded_extension(tmp_path)

    build["release_manifest_sha256"] = module.file_sha256(tmp_path / "release.toml")
    """Restored the manifest commitment exposed by the loaded extension."""

    dirty[0] = True
    """Made the current checkout dirty independently of the extension build."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKOUT_NOT_CLEAN",
    ):
        module.validate_loaded_extension(tmp_path)


def test_producer_provenance_uses_the_loaded_extension_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record the public digest of the binary that actually ran."""
    module: ModuleType = target_module()
    """Loaded the target command's invocation-time provenance path."""

    monkeypatch.setattr(module, "validate_loaded_extension", lambda root: None)
    build: dict[str, object] = {
        "source_commit": "a" * 40,
        "release_manifest_sha256": "b" * 64,
    }
    """Modelled the already validated public build identity."""

    monkeypatch.setattr(module.asterism, "build_identity", lambda: dict(build))
    monkeypatch.setattr(
        module.asterism,
        "installed_extension_sha256",
        lambda: "c" * 64,
    )
    """Supplied the digest from the imported-module helper, with no directory scan."""

    assert module.producer_provenance(tmp_path) == {
        "source_commit": "a" * 40,
        "release_manifest_sha256": "b" * 64,
        "extension_sha256": "c" * 64,
        "build_identity": build,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rejected", False),
        ("p_value", 0.049),
        ("in_lrt_atom", True),
        ("nuisance_at_bound", True),
        ("replicate", True),
        ("test_statistic", float("nan")),
    ],
)
def test_checkpoint_rejects_corrupt_asymptotic_null_primitives(
    field: str,
    value: object,
) -> None:
    """Do not trust edited null decisions, algebra, streams or primitive types."""
    module: ModuleType = target_module()
    """Loaded the strict resumed-row validator."""

    row: dict[str, object] = valid_asymptotic_null_row(module)
    """Built an otherwise valid null row to corrupt."""

    row[field] = value
    """Corrupted one otherwise complete persisted null result."""

    with pytest.raises(ValueError):
        module.validate_campaign_rows(
            [row],
            [("null", 0.52, 0)],
            "asymptotic",
            None,
            require_complete=True,
        )


def test_asymptotic_checkpoint_accepts_the_public_operational_atom() -> None:
    """Treat likelihood-search rounding exactly as the public test does."""
    module: ModuleType = target_module()
    """Loaded the strict resumed-row validator."""

    row: dict[str, object] = valid_asymptotic_null_row(module)
    """Built the row shape rejected by the first analytic campaign."""

    row.update(
        p_value=1.0,
        test_statistic=8.549250196665525e-11,
        rejected=False,
        in_lrt_atom=True,
    )
    """Reproduced a public result settled into the point mass at nought."""

    module.validate_campaign_rows(
        [row],
        [("null", 0.52, 0)],
        "asymptotic",
        None,
        require_complete=True,
    )


def test_asymptotic_checkpoint_rejects_fake_bootstrap_completion() -> None:
    """Do not decorate an analytic result with bootstrap work that never ran."""
    module: ModuleType = target_module()
    """Loaded the strict mode-specific row validator."""

    row: dict[str, object] = valid_asymptotic_null_row(module)
    """Built one valid ordinary analytic row."""

    row.update(
        bootstrap_exceedances=49,
        bootstrap_replicates=999,
        bootstrap_requested=999,
        bootstrap_seed=1,
    )
    """Added the counters that only a completed bootstrap may report."""

    with pytest.raises(ValueError):
        module.validate_campaign_rows(
            [row],
            [("null", 0.52, 0)],
            "asymptotic",
            None,
            require_complete=True,
        )


def test_checkpoint_rejects_an_impossible_zero_statistic_bootstrap() -> None:
    """A zero observed statistic must be exceeded by every non-negative null draw."""
    module: ModuleType = target_module()
    """Loaded the strict resumed-row validator."""

    row: dict[str, object] = valid_bootstrap_null_row(module)
    """Built an otherwise valid null row to make physically impossible."""

    row["test_statistic"] = 0.0
    """Placed the observed likelihood-ratio statistic at its lower bound."""

    row["in_lrt_atom"] = True
    """Made the atom membership coherent but left an impossible exceedance count."""

    with pytest.raises(ValueError):
        module.validate_campaign_rows(
            [row],
            [("null", 0.52, 0)],
            "bootstrap",
            999,
            require_complete=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("covered", False),
        ("bound_unreachable", False),
        ("extra_failures", 1),
        ("lower", 0.40),
        ("upper", 0.90),
        ("upper_limited", False),
        ("contains_upper_bound", True),
        ("contains_upper_bound", "yes"),
        ("profile_failures", True),
    ],
)
def test_checkpoint_rejects_corrupt_interval_primitives(
    field: str,
    value: object,
) -> None:
    """Do not trust edited coverage, profile status, endpoints or field types."""
    module: ModuleType = target_module()
    """Loaded the strict resumed-row validator."""

    row: dict[str, object] = valid_heritable_row(module)
    """Built an otherwise valid interval row to corrupt."""

    row[field] = value
    """Corrupted one otherwise complete persisted interval result."""

    with pytest.raises(ValueError):
        module.validate_campaign_rows(
            [row],
            [("heritable", 0.52, 0)],
            "asymptotic",
            None,
            require_complete=True,
        )


def test_checkpoint_rejects_changed_limit_and_malformed_refusal() -> None:
    """Keep one fixed instrument limit per share and a strict refusal schema."""
    module: ModuleType = target_module()
    """Loaded the strict resumed-row validator."""

    first: dict[str, object] = valid_heritable_row(module, replicate=0)
    """Built the first valid interval row at the shared censoring design."""

    second: dict[str, object] = valid_heritable_row(module, replicate=1)
    """Built a second valid interval row at the same censoring design."""

    second["censoring_limit"] = 1.5
    """Changed the instrument between two outcomes at the same intended share."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKPOINT_LIMIT_MISMATCH",
    ):
        module.validate_campaign_rows(
            [first, second],
            [("heritable", 0.52, 0), ("heritable", 0.52, 1)],
            "asymptotic",
            None,
            require_complete=True,
        )

    first["censoring_limit"] = 1.5
    """Corrupted every retained limit uniformly, defeating consistency alone."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_CHECKPOINT_LIMIT_MISMATCH",
    ):
        module.validate_campaign_rows(
            [first, second],
            [("heritable", 0.52, 0), ("heritable", 0.52, 1)],
            "asymptotic",
            None,
            require_complete=True,
        )

    refusal: dict[str, object] = {
        key: value
        for key, value in valid_heritable_row(module).items()
        if key
        in {
            "scenario",
            "censoring_share",
            "replicate",
            "worst_censored_block",
            "achieved_censoring_share",
            "censoring_limit",
        }
    }
    """Retained only the common fields permitted on an explicit refusal."""

    refusal["refusal"] = ""
    """Made an empty refusal marker that could conceal a lost result."""

    with pytest.raises(ValueError):
        module.validate_campaign_rows(
            [refusal],
            [("heritable", 0.52, 0)],
            "asymptotic",
            None,
            require_complete=True,
        )

    refusal["refusal"] = "TOBIT_INTERVAL_REFUSED"
    """Made the same row a valid explicit refusal for downstream scoring."""

    module.validate_campaign_rows(
        [refusal],
        [("heritable", 0.52, 0)],
        "asymptotic",
        None,
        require_complete=True,
    )
    assert module.row_failed(refusal) is True
    assert module.row_bound_unreachable(refusal) is False


def test_main_reports_a_heritable_refusal_without_dereferencing_interval_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Score a complete refusal campaign as failed instead of crashing on its row."""
    module: ModuleType = target_module()
    """Loaded the command's complete reporting path."""

    rows: list[dict[str, object]] = [
        valid_asymptotic_null_row(module, share=share)
        for share in module.CENSORING_SHARES
    ]
    """Provided one measured null attempt for each configured censoring share."""

    measured_interval: dict[str, object] = valid_heritable_row(
        module,
        share=module.CENSORING_SHARES[1],
    )
    """Provided one measured interval so ordinary interval reporting still runs."""

    refusal: dict[str, object] = {
        key: value
        for key, value in valid_heritable_row(
            module,
            share=module.CENSORING_SHARES[0],
        ).items()
        if key
        in {
            "scenario",
            "censoring_share",
            "replicate",
            "worst_censored_block",
            "achieved_censoring_share",
            "censoring_limit",
        }
    }
    """Built the strict common-field-only form of a refused interval attempt."""

    refusal["refusal"] = "TOBIT_INTERVAL_REFUSED"
    """Marked the first heritable attempt as an explicit scientific refusal."""

    rows.extend([refusal, measured_interval])
    """Completed the four-cell reporting grid with the refusal retained."""

    signature: dict[str, object] = {"campaign": "test"}
    """Used one stable signature for both the pre-run and post-run checks."""

    monkeypatch.setattr(sys, "argv", ["censored-components-target", "--no-write"])
    monkeypatch.setattr(
        module,
        "target_metadata",
        lambda: {"people": 1_909, "largest_family": 180},
    )
    monkeypatch.setattr(module, "checkpoint_signature", lambda *arguments: signature)
    monkeypatch.setattr(module, "producer_provenance", lambda root: valid_producer())
    monkeypatch.setattr(
        module,
        "collect_campaign_rows",
        lambda *arguments: (rows, [{"producer": valid_producer(), "coordinates": []}]),
    )
    """Replaced only campaign production while retaining end-to-end scoring."""

    status: int = module.main()
    """Ran argument parsing, cell scoring, failure collection and terminal reporting."""

    output: str = capsys.readouterr().out
    """Captured the command record to verify the refusal reached failed reporting."""

    assert status == 1
    assert "heritable at 0.52: nothing was measured" in output
    assert "1 of 4 attempts failed" in output


@pytest.mark.parametrize(
    ("option", "message"),
    [
        (
            ["--bootstrap-replicates", "99"],
            "--bootstrap-replicates requires --test-reference bootstrap",
        ),
        (
            ["--bootstrap-threads", "2"],
            "--bootstrap-threads requires --test-reference bootstrap",
        ),
    ],
)
def test_main_defaults_to_asymptotic_and_rejects_bootstrap_options_there(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: list[str],
    message: str,
) -> None:
    """Make the ordinary command analytic and keep inner options mode-specific."""
    module: ModuleType = target_module()
    """Loaded the public command parser and orchestration seam."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["censored-components-target", *option],
    )
    with pytest.raises(SystemExit, match="2"):
        module.main()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--shard-index", "0"],
            "--shard-index and --shard-count must be supplied together",
        ),
        (
            ["--shard-index", "0", "--shard-count", "10"],
            "shard production requires --checkpoint",
        ),
        (
            [
                "--shard-index",
                "10",
                "--shard-count",
                "10",
                "--checkpoint",
                "campaign.json",
            ],
            "CENSORED_COMPONENT_TARGET_SHARD_ALLOCATION_INVALID",
        ),
        (
            [
                "--merge-checkpoints",
                "shard-0.json",
                "shard-1.json",
                "--checkpoint",
                "campaign.json",
            ],
            "--merge-checkpoints is incompatible with --checkpoint",
        ),
        (
            [
                "--merge-checkpoints",
                "shard-0.json",
                "shard-1.json",
                "--shard-index",
                "0",
                "--shard-count",
                "2",
                "--checkpoint",
                "campaign.json",
            ],
            "shard production and --merge-checkpoints are incompatible",
        ),
        (
            ["--merge-checkpoints", "shard-0.json"],
            "--merge-checkpoints requires at least two shard checkpoints",
        ),
    ],
)
def test_main_rejects_incompatible_or_incomplete_shard_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    """Fail before fitting when shard production or merge is underspecified."""
    module: ModuleType = target_module()
    """Loaded the public command parser."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["censored-components-target", *arguments],
    )
    with pytest.raises(SystemExit, match="2"):
        module.main()
    assert message in capsys.readouterr().err


def test_main_bounds_automatic_bootstrap_threads_and_requires_positive_override(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the automatic outer-by-inner allocation within visible CPUs."""
    module: ModuleType = target_module()
    """Loaded the public resource-allocation checks."""

    monkeypatch.setattr(module, "allocated_cpu_count", lambda: 4)
    monkeypatch.setattr(
        sys,
        "argv",
        ["censored-components-target", "--test-reference", "bootstrap"],
    )
    with pytest.raises(SystemExit, match="2"):
        module.main()
    assert (
        "outer workers times bootstrap threads exceed the visible CPU allocation"
        in capsys.readouterr().err
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "censored-components-target",
            "--test-reference",
            "bootstrap",
            "--bootstrap-threads",
            "0",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        module.main()
    assert "bootstrap threads must be positive" in capsys.readouterr().err


def test_main_rejects_explicit_outer_by_inner_bootstrap_oversubscription(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep an explicit inner allocation within the scheduler-visible CPU total."""
    module: ModuleType = target_module()
    """Loaded the public bootstrap allocation seam."""

    monkeypatch.setattr(module, "allocated_cpu_count", lambda: 128)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "censored-components-target",
            "--test-reference",
            "bootstrap",
            "--workers",
            "8",
            "--bootstrap-threads",
            "128",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        module.main()
    assert (
        "outer workers times bootstrap threads exceed the visible CPU allocation"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("arguments", "reference", "bootstrap_replicates", "bootstrap_threads"),
    [
        (["--test-reference", "asymptotic"], "asymptotic", None, None),
        (
            [
                "--test-reference",
                "bootstrap",
                "--bootstrap-replicates",
                "99",
            ],
            "bootstrap",
            99,
            4,
        ),
        (
            [
                "--test-reference",
                "bootstrap",
                "--bootstrap-replicates",
                "99",
                "--bootstrap-threads",
                "3",
            ],
            "bootstrap",
            99,
            3,
        ),
    ],
)
def test_main_binds_the_selected_reference_through_the_campaign(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    reference: str,
    bootstrap_replicates: int | None,
    bootstrap_threads: int | None,
) -> None:
    """Pass the selected mode through signatures, workers and final scoring."""
    module: ModuleType = target_module()
    """Loaded the command without running its real target fits."""

    rows: list[dict[str, object]] = []
    """Collected one complete measured row per target cell for reporting."""

    for share in module.CENSORING_SHARES:
        rows.append(
            valid_asymptotic_null_row(module, share=share)
            if reference == "asymptotic"
            else valid_bootstrap_null_row(
                module,
                share=share,
                bootstrap_replicates=99,
            )
        )
        rows.append(valid_heritable_row(module, share=share))
    signature_calls: list[tuple[int, str, int | None]] = []
    """Recorded both pre-run and post-run campaign identities."""

    def signature(
        replicates: int,
        selected: str,
        inner: int | None,
    ) -> dict[str, object]:
        """Return one stable signature while recording its selected mode."""
        signature_calls.append((replicates, selected, inner))
        return {"campaign": "test", "test_reference": selected}

    collection_calls: list[tuple[str, int | None, int | None]] = []
    """Recorded the mode handed to worker orchestration."""

    def collect(
        jobs: list[tuple[str, float, int]],
        workers: int,
        selected: str,
        inner: int | None,
        threads: int | None,
        checkpoint: Path | None,
        campaign_signature: dict[str, object],
        producer: dict[str, object],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Return the complete fixture grid through the production call shape."""
        assert len(jobs) == 4
        assert workers == module.WORKERS
        assert checkpoint is None
        assert campaign_signature["test_reference"] == selected
        assert producer == valid_producer()
        collection_calls.append((selected, inner, threads))
        return rows, [{"producer": valid_producer(), "coordinates": []}]

    monkeypatch.setattr(
        sys,
        "argv",
        ["censored-components-target", "--replicates", "1", "--no-write", *arguments],
    )
    monkeypatch.setattr(
        module,
        "target_metadata",
        lambda: {"people": 1_909, "largest_family": 180},
    )
    monkeypatch.setattr(module, "checkpoint_signature", signature)
    monkeypatch.setattr(module, "producer_provenance", lambda root: valid_producer())
    monkeypatch.setattr(module, "collect_campaign_rows", collect)
    monkeypatch.setattr(module, "allocated_cpu_count", lambda: 32)

    module.main()

    assert signature_calls == [
        (1, reference, bootstrap_replicates),
        (1, reference, bootstrap_replicates),
    ]
    assert collection_calls == [(reference, bootstrap_replicates, bootstrap_threads)]


def test_checkpoint_lock_refuses_a_second_writer(tmp_path: Path) -> None:
    """Prevent two campaigns from overwriting one resumable result stream."""
    module: ModuleType = target_module()
    """Loaded the advisory writer lock."""

    checkpoint: Path = tmp_path / "campaign.json"
    """Located the checkpoint whose single-writer lock is under test."""

    with (
        module.checkpoint_writer(checkpoint),
        pytest.raises(
            ValueError,
            match="CENSORED_COMPONENT_TARGET_CHECKPOINT_LOCKED",
        ),
        module.checkpoint_writer(checkpoint),
    ):
        pytest.fail("the second writer unexpectedly acquired the lock")


def test_worker_pool_uses_spawn_to_avoid_inheriting_the_checkpoint_lock() -> None:
    """Keep spawned workers from extending the parent's advisory-lock lifetime."""
    module: ModuleType = target_module()
    """Loaded the worker-process context selection."""

    context: object = module.worker_multiprocessing_context()
    """Selected the actual multiprocessing context passed to the executor."""

    assert context.get_start_method() == "spawn"


def test_exceptional_pool_shutdown_terminates_and_reaps_workers() -> None:
    """Do not wait for hours of active bootstrap work after a worker fails."""
    module: ModuleType = target_module()
    """Loaded the Python 3.13 fallback used when public termination is absent."""

    class FakeProcess:
        """Record abrupt process operations while modelling a stubborn worker."""

        def __init__(self) -> None:
            self.alive: bool = True
            """Tracked whether the fake process still needs stopping."""

            self.operations: list[str] = []
            """Recorded the ordered termination and reaping operations."""

        def is_alive(self) -> bool:
            """Report whether the fake still needs escalation."""
            return self.alive

        def terminate(self) -> None:
            """Record a graceful abrupt-stop request without exiting yet."""
            self.operations.append("terminate")

        def kill(self) -> None:
            """Record forced termination and make the process stop."""
            self.operations.append("kill")
            self.alive = False
            """Made subsequent liveness checks report the forced stop."""

        def join(self, timeout: float | None = None) -> None:
            """Record reaping without sleeping in the unit test."""
            self.operations.append("join")

    class FakePool:
        """Expose the worker handles available on supported Python 3.13."""

        def __init__(self, process: FakeProcess) -> None:
            self._processes: dict[int, FakeProcess] = {1: process}
            """Exposed the private worker registry used by Python 3.13."""

            self.shutdown_arguments: list[tuple[bool, bool]] = []
            """Recorded non-blocking cancellation and final resource cleanup."""

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            """Record cancellation of work that has not reached a process."""
            self.shutdown_arguments.append((wait, cancel_futures))

    process: FakeProcess = FakeProcess()
    """Created one stubborn worker that requires escalation to a kill."""

    pool: FakePool = FakePool(process)
    """Wrapped the worker in the executor interface used by the abort helper."""

    module.abort_process_pool(pool)

    assert pool.shutdown_arguments == [(False, True), (True, True)]
    assert process.operations == ["terminate", "join", "kill", "join"]
    assert process.alive is False


def test_failed_future_saves_completed_rows_and_does_not_fill_the_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Checkpoint simultaneous successes before aborting a bounded worker queue."""
    module: ModuleType = target_module()
    """Loaded the resumable queue orchestration."""

    good: dict[str, object] = valid_asymptotic_null_row(module)
    """Prepared the completed result that must survive a peer failure."""

    jobs: list[tuple[str, float, int]] = [
        ("null", 0.52, 0),
        ("heritable", 0.52, 0),
        ("null", 0.52, 1),
    ]
    """Declared two initial attempts and one that must remain unsubmitted."""

    class FakeFuture:
        """Return one row or one worker exception immediately."""

        def __init__(self, value: object) -> None:
            self.value: object = value
            """Stored either the completed row or the worker exception."""

            self.cancelled: bool = False
            """Tracked whether orchestration attempted to cancel the future."""

        def result(self) -> dict[str, object]:
            """Return the successful row or reproduce the worker failure."""
            if isinstance(self.value, BaseException):
                raise self.value
            return self.value

        def done(self) -> bool:
            """Make both initial futures available in the same checkpoint batch."""
            return True

        def cancel(self) -> None:
            """Record cancellation if orchestration leaves this future active."""
            self.cancelled = True
            """Marked this fake future as cancelled."""

    class FakePool:
        """Record submissions without starting a child process."""

        def __init__(self) -> None:
            self.submitted: list[tuple[str, float, int]] = []
            """Recorded jobs submitted before collection encountered failure."""

        def submit(
            self,
            function: object,
            job: tuple[str, float, int],
        ) -> FakeFuture:
            """Return a success for the first job and a failure for the second."""
            self.submitted.append(job)
            return FakeFuture(good if job[0] == "null" else RuntimeError("failed"))

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            """Ordinary shutdown is not expected on this exceptional path."""
            pytest.fail("exceptional collection waited for ordinary pool shutdown")

    pool: FakePool = FakePool()
    """Created the executor double used by the bounded queue."""

    aborted: list[FakePool] = []
    """Recorded whether exceptional orchestration aborted the fake executor."""

    monkeypatch.setattr(module, "ProcessPoolExecutor", lambda **arguments: pool)
    monkeypatch.setattr(
        module,
        "wait",
        lambda active, return_when: (set(active), set()),
    )
    monkeypatch.setattr(module, "abort_process_pool", aborted.append)
    monkeypatch.setattr(module, "allocated_cpu_count", lambda: 8)
    """Replaced process mechanics while retaining the production queue algorithm."""

    checkpoint: Path = tmp_path / "campaign.json"
    """Located the checkpoint that must retain the simultaneous success."""

    with pytest.raises(
        RuntimeError,
        match="CENSORED_COMPONENT_TARGET_WORKER_FAILED",
    ):
        module.collect_campaign_rows(
            jobs,
            workers=2,
            test_reference="asymptotic",
            bootstrap_replicates=None,
            bootstrap_threads=None,
            checkpoint=checkpoint,
            signature={"campaign": "test", "test_reference": "asymptotic"},
            producer=valid_producer(),
        )

    payload: dict[str, object] = module.json.loads(checkpoint.read_text())
    """Reloaded the operational checkpoint written before the pool abort."""

    assert payload["rows"] == [good]
    assert payload["producer_history"] == [
        {
            "producer": {
                **valid_producer(),
                "execution": {
                    "allocated_cpu_count": 8,
                    "outer_workers": 2,
                    "bootstrap_threads_per_outer_worker": None,
                },
            },
            "coordinates": [["null", 0.52, 0]],
        }
    ]
    assert pool.submitted == jobs[:2]
    assert aborted == [pool]


def test_worker_factors_the_generating_covariance_by_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not retain two additional whole-target dense factors per worker."""
    module: ModuleType = target_module()
    """Loaded worker initialisation without building the real target."""

    blocks: list[np.ndarray] = [np.asarray([0, 1]), np.asarray([2, 3])]
    """Made two tiny independent families for the memory-shape contract."""

    structure: dict[str, object] = {
        "components": [np.eye(4), np.eye(4), np.eye(4)],
        "design": np.ones((4, 6)),
        "blocks": blocks,
        "largest_family": 1,
        "people": 2,
        "fixture_sha256": "a" * 64,
    }
    """Substituted small matrices while retaining the target's component shape."""

    monkeypatch.setattr(module, "target_components", lambda: structure)
    validated: list[Path] = []
    """Recorded the worker's independent extension validation."""

    monkeypatch.setattr(module, "validate_loaded_extension", validated.append)
    monkeypatch.setattr(
        module.asterism,
        "installed_extension_sha256",
        lambda: "d" * 64,
    )
    """Bound the worker to the same modelled binary as its parent."""

    monkeypatch.setenv("RAYON_NUM_THREADS", "64")
    """Modelled an inherited allocation that the bootstrap must override."""

    module.start_worker(
        test_reference="bootstrap",
        bootstrap_replicates=1,
        bootstrap_threads=3,
        expected_extension_sha256="d" * 64,
    )

    assert validated == [ROOT]
    assert "factor_null" not in module.STATE
    assert "factor_heritable" not in module.STATE
    assert [factor.shape for factor in module.STATE["factors_null"]] == [
        (2, 2),
        (2, 2),
    ]
    assert [factor.shape for factor in module.STATE["factors_heritable"]] == [
        (2, 2),
        (2, 2),
    ]
    assert "model" in module.STATE
    assert "components" not in module.STATE
    assert module.STATE["test_reference"] == "bootstrap"
    assert module.STATE["bootstrap_replicates"] == 1
    assert module.os.environ["RAYON_NUM_THREADS"] == "3"


def test_worker_refuses_a_different_loaded_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop before allocating target state when a child imports other bytes."""
    module: ModuleType = target_module()
    """Loaded worker initialisation without launching a process pool."""

    monkeypatch.setattr(module, "validate_loaded_extension", lambda root: None)
    monkeypatch.setattr(
        module.asterism,
        "installed_extension_sha256",
        lambda: "e" * 64,
    )
    """Modelled a child that resolves a different binary than its parent."""

    with pytest.raises(
        ValueError,
        match="CENSORED_COMPONENT_TARGET_WORKER_EXTENSION_MISMATCH",
    ):
        module.start_worker(
            test_reference="asymptotic",
            bootstrap_replicates=None,
            bootstrap_threads=None,
            expected_extension_sha256="f" * 64,
        )


def test_null_worker_path_uses_the_ordinary_asymptotic_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call the public analytic test without inventing bootstrap completion."""
    module: ModuleType = target_module()
    """Loaded the target worker entry point."""

    class FakeModel:
        """Expose only the public analytic method selected by this mode."""

        def test(
            self,
            value: np.ndarray,
            censoring: np.ndarray,
            limit: np.ndarray,
            *,
            component: int,
        ) -> dict[str, object]:
            """Check what `one` sends through the public analytic seam."""
            assert np.isfinite(value).all()
            assert not censoring.any()
            assert np.array_equal(limit, np.full(4, 2.0))
            assert component == 0
            return {
                "statistic": 2.705543454095404,
                "p_value": 0.05,
                "rule": "asymptotic_mixture_50_50",
                "nuisance_at_bound": False,
            }

        def bootstrap(self, *arguments: object, **keywords: object) -> None:
            """Fail if the ordinary release gate enters the optional mode."""
            pytest.fail("ordinary target campaign called the bootstrap")

    state: dict[str, object] = {
        "design": np.ones((4, 6)),
        "blocks": [np.asarray([0, 1]), np.asarray([2, 3])],
        "mean_coefficients": np.asarray([1.0, 0.2, -0.1, 0.05, 0.3, -0.2]),
        "limits": {0.52: 2.0},
        "factors_null": [np.zeros((2, 2)), np.zeros((2, 2))],
        "model": FakeModel(),
        "test_reference": "asymptotic",
    }
    """Substituted a complete small ordinary-mode worker state."""

    monkeypatch.setattr(module, "STATE", state)
    result: dict[str, object] = module.one(("null", 0.52, 3))
    """Ran one null attempt through the ordinary public analytic method."""

    assert result["test_rule"] == "asymptotic_mixture_50_50"
    assert result["nuisance_at_bound"] is False
    assert result["p_value"] == 0.05
    assert result["rejected"] is True
    assert result["in_lrt_atom"] is False
    assert not any(key.startswith("bootstrap_") for key in result)


def test_null_worker_records_the_public_operational_atom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a rounded likelihood gap beside the point mass its p-value used."""
    module: ModuleType = target_module()
    """Loaded the target worker entry point."""

    class FakeModel:
        """Return the exact public result that stopped the first campaign."""

        def test(
            self,
            value: np.ndarray,
            censoring: np.ndarray,
            limit: np.ndarray,
            *,
            component: int,
        ) -> dict[str, object]:
            """Expose a raw rounding gap whose p-value is already settled."""
            assert component == 0
            return {
                "statistic": 8.549250196665525e-11,
                "p_value": 1.0,
                "rule": "asymptotic_mixture_50_50",
                "nuisance_at_bound": False,
            }

    monkeypatch.setattr(
        module,
        "STATE",
        {
            "design": np.ones((4, 6)),
            "blocks": [np.asarray([0, 1]), np.asarray([2, 3])],
            "mean_coefficients": np.asarray([1.0, 0.2, -0.1, 0.05, 0.3, -0.2]),
            "limits": {0.52: 2.0},
            "factors_null": [np.zeros((2, 2)), np.zeros((2, 2))],
            "model": FakeModel(),
            "test_reference": "asymptotic",
        },
    )
    """Substituted a complete small worker state."""

    result: dict[str, object] = module.one(("null", 0.52, 3))
    """Ran the failed result shape through the production worker seam."""

    assert result["test_statistic"] == 8.549250196665525e-11
    assert result["p_value"] == 1.0
    assert result["in_lrt_atom"] is True
    module.validate_null_row(result, "asymptotic", None)


def test_null_worker_path_pins_the_complete_bootstrap_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production outer path with a tiny deterministic fake model."""
    module: ModuleType = target_module()
    """Loaded the target worker entry point."""

    class FakeModel:
        """Return one exact complete bootstrap record after checking its inputs."""

        def bootstrap(
            self,
            value: np.ndarray,
            censoring: np.ndarray,
            limit: np.ndarray,
            direction: np.ndarray,
            *,
            component: int,
            replicates: int,
            seed: int,
        ) -> dict[str, object]:
            """Check what `one` sends through the real public method seam."""
            assert np.isfinite(value).all()
            assert not censoring.any()
            assert np.array_equal(limit, np.full(4, 2.0))
            assert np.array_equal(direction, np.ones(4, dtype=np.int64))
            assert component == 0
            assert replicates == 999
            assert seed == module.BOOTSTRAP_BASE_SEED + 7_919 * 3 + 104_729 * 52
            return {
                "statistic": 1.0,
                "p_value": 0.05,
                "rule": "parametric_bootstrap_add_one",
                "nuisance_at_bound": False,
                "exceedances": 49,
                "replicates": 999,
                "requested": 999,
            }

    blocks: list[np.ndarray] = [np.asarray([0, 1]), np.asarray([2, 3])]
    """Declared two independent blocks for the tiny generator."""

    state: dict[str, object] = {
        "design": np.ones((4, 6)),
        "blocks": blocks,
        "mean_coefficients": np.asarray([1.0, 0.2, -0.1, 0.05, 0.3, -0.2]),
        "limits": {0.52: 2.0},
        "factors_null": [np.zeros((2, 2)), np.zeros((2, 2))],
        "model": FakeModel(),
        "test_reference": "bootstrap",
        "bootstrap_replicates": 999,
    }
    """Substituted a complete small worker state without dense target matrices."""

    monkeypatch.setattr(module, "STATE", state)
    result: dict[str, object] = module.one(("null", 0.52, 3))
    """Ran the same outer worker function used by the scientific command."""

    assert result["test_rule"] == "parametric_bootstrap_add_one"
    assert result["bootstrap_replicates"] == 999
    assert result["bootstrap_requested"] == 999
    assert result["bootstrap_exceedances"] == 49
    assert result["test_statistic"] == 1.0
    assert result["rejected"] is True
