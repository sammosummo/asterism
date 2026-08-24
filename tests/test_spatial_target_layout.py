"""Contract tests for the participant-free spatial target-layout gate."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import tomllib
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

TARGET_FIXTURE: Path = (
    Path(__file__).parents[1]
    / "checks"
    / "design_fixtures"
    / "spatial_component_presence_target_layout.json"
)
"""Located the reviewed values-free target-layout aggregate."""

TARGET_FIXTURE_SHA256: str = (
    "7d0d7bb2afde1a5f7566e2fa04e47292d148d978d832b2c85fb6c4b76fe93d6b"
)
"""Pinned the exact reviewed fixture bytes before any target run."""


def target_module() -> ModuleType:
    """Load the standalone target-layout command through its file boundary."""
    path: Path = Path(__file__).parents[1] / "checks" / "spatial_target_layout.py"
    """Located the command without turning the checks directory into a package."""

    checks_path: str = str(path.parent)
    """Selected the sibling-check import root used by standalone commands."""

    if checks_path not in sys.path:
        sys.path.insert(0, checks_path)
    """Made the established pedigree helper importable by its standalone name."""

    specification: importlib.machinery.ModuleSpec | None = (
        importlib.util.spec_from_file_location("spatial_target_layout", path)
    )
    """Created the import specification for the standalone source file."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the module governed by the validated specification."""

    sys.modules[specification.name] = module
    """Registered the module so public dataclasses can resolve annotations."""

    specification.loader.exec_module(module)
    """Executed the command without invoking its command-line entry point."""
    return module


def test_target_fixture_recreates_only_reviewed_aggregate_structure() -> None:
    """Match target scale and geometry without retaining observed locations."""
    module: ModuleType = target_module()
    """Loaded the strict fixture verifier and synthetic-layout constructor."""

    assert module.fixture_sha256(TARGET_FIXTURE) == TARGET_FIXTURE_SHA256

    envelope: Any = module.spatial_target_envelope(
        TARGET_FIXTURE,
        expected_sha256=TARGET_FIXTURE_SHA256,
    )
    """Built the synthetic pedigree, household, distances, and fixed design."""

    assert envelope.eligible_people == 1_883
    assert envelope.missing_people == 91
    assert envelope.relationship.shape == (1_792, 1_792)
    assert envelope.household.shape == (1_792, 1_792)
    assert envelope.distance.shape == (1_792, 1_792)
    assert envelope.fixed_effects.shape == (1_792, 6)
    assert np.linalg.matrix_rank(envelope.fixed_effects) == 6
    assert len(envelope.component_sizes) == 190
    assert max(envelope.component_sizes) == 165
    assert len(envelope.location_sizes) == 1_140
    assert Counter(envelope.location_sizes) == {
        1: 733,
        2: 245,
        3: 106,
        4: 35,
        5: 16,
        6: 4,
        7: 1,
    }
    assert int(np.count_nonzero(np.triu(envelope.household, k=1))) == 1_014
    assert np.array_equal(envelope.distance, envelope.distance.T)
    assert np.allclose(np.diag(envelope.distance), 0.0)
    assert max(envelope.relative_quantile_errors.values()) <= 0.10

    fixture: dict[str, Any] = module.load_fixture(TARGET_FIXTURE)
    """Read the reviewed aggregate record to inspect only its field names."""

    keys: set[str] = module.nested_keys(fixture)
    """Collected nested keys without inspecting or exposing any participant row."""

    assert "subject_id" not in keys
    assert "latitude" not in keys
    assert "longitude" not in keys
    assert "coordinates" not in keys
    assert "distance_matrix" not in keys


def test_layout_generation_is_exactly_reproducible() -> None:
    """Repeat the synthetic layout without consulting external participant data."""
    module: ModuleType = target_module()
    """Loaded the deterministic target constructor twice."""

    first: Any = module.spatial_target_envelope(
        TARGET_FIXTURE,
        expected_sha256=TARGET_FIXTURE_SHA256,
    )
    """Generated one complete participant-free target envelope."""

    second: Any = module.spatial_target_envelope(
        TARGET_FIXTURE,
        expected_sha256=TARGET_FIXTURE_SHA256,
    )
    """Repeated the identical reviewed fixture and deterministic streams."""

    assert np.array_equal(first.relationship, second.relationship)
    assert np.array_equal(first.household, second.household)
    assert np.array_equal(first.distance, second.distance)
    assert np.array_equal(first.fixed_effects, second.fixed_effects)


def test_target_decision_refuses_incomplete_or_nonreportable_inference() -> None:
    """Make every fit and bootstrap failure decisive before target evidence."""
    module: ModuleType = target_module()
    """Loaded the prewritten target acceptance decision."""

    fit: dict[str, object] = {
        "converged": True,
        "loglik": -100.0,
        "scaled_gradient": 1e-9,
    }
    """Built the smallest converged public spatial fit record."""

    bootstrap: dict[str, object] = {
        "statistic": 10.0,
        "exceedances": 0,
        "replicates": 199,
        "requested": 199,
        "p_value": 0.005,
        "rule": "parametric_bootstrap_add_one",
        "smallest_reportable": 0.005,
    }
    """Built a complete target detection with the exact add-one reference."""

    passed: dict[str, object] = module.target_decision(
        fit,
        bootstrap,
        requested=199,
        alpha=0.05,
        refusal=None,
    )
    """Scored the complete reportable result under the prewritten policy."""

    assert passed == {"passed": True, "reason": "target_presence_detected"}

    for changed_fit, changed_bootstrap, refusal, reason in (
        (
            {**fit, "converged": False},
            bootstrap,
            None,
            "free_fit_not_converged",
        ),
        (
            fit,
            {**bootstrap, "replicates": 198},
            None,
            "bootstrap_incomplete",
        ),
        (
            fit,
            {**bootstrap, "p_value": 0.10},
            None,
            "target_presence_not_detected",
        ),
        (fit, bootstrap, "SPATIAL_BOOTSTRAP_REPLICATE_FAILED", "estimator_refused"),
    ):
        decision: dict[str, object] = module.target_decision(
            changed_fit,
            changed_bootstrap,
            requested=199,
            alpha=0.05,
            refusal=refusal,
        )
        """Scored one deliberately nonreportable target outcome."""

        assert decision == {"passed": False, "reason": reason}


def test_small_real_public_bootstrap_uses_every_requested_replicate() -> None:
    """Exercise the same public fit and bootstrap seam on a small real problem."""
    module: ModuleType = target_module()
    """Loaded the target command's public model runner."""

    problem: Any = module.small_spatial_problem()
    """Built a compact participant-free sibling and shared-location layout."""

    result: dict[str, object] = module.run_spatial_problem(
        problem,
        bootstrap_replicates=3,
        bootstrap_seed=4_177,
    )
    """Ran the compiled public fit and null bootstrap without mocked internals."""

    assert result["fit"]["converged"] is True
    assert len(result["fit"]["subject_order_sha256"]) == 64
    assert result["bootstrap"]["requested"] == 3
    assert result["bootstrap"]["replicates"] == 3
    assert result["bootstrap"]["smallest_reportable"] == 0.25
    assert result["decision"] == {
        "passed": False,
        "reason": "target_presence_not_detected",
    }
    """Kept the three-replicate smoke distinct from five-per-cent release evidence."""


def test_release_manifest_defers_the_spatial_presence_analysis() -> None:
    """Keep the target gate executable after 0.1 stopped claiming spatial support."""
    manifest: dict[str, Any] = tomllib.loads(
        (Path(__file__).parents[1] / "release.toml").read_text(encoding="utf-8")
    )
    """Parsed the authoritative development release contract."""

    analysis_ids: list[str] = [str(entry["id"]) for entry in manifest["analyses"]]
    """Collected every analysis 0.1 makes a scientific claim about."""

    assert "spatial_component_presence" not in analysis_ids
    """Held the deferral, so no release evidence waits on the spatial target run."""

    assert TARGET_FIXTURE.is_file()
    assert (
        Path(__file__).parents[1] / "checks" / "spatial_target_layout.py"
    ).is_file()
    """Kept the deferred material intact and runnable for a later release."""


def test_target_command_requires_every_scientific_argument() -> None:
    """Prevent defaults or ambient state from selecting release evidence."""
    module: ModuleType = target_module()
    """Loaded the standalone command-line parser."""

    with pytest.raises(SystemExit):
        module.parse_arguments([])
