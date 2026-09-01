"""Contracts for the parallel component-separation campaign."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the standalone scientific command."""


def separation_module() -> ModuleType:
    """Load the standalone command without running its campaign."""
    path: Path = ROOT / "checks" / "component_separation.py"
    """Located the participant-free scientific command."""

    specification: ModuleSpec | None = importlib.util.spec_from_file_location(
        "component_separation", path
    )
    """Built an import specification whose name is pickle-resolvable."""

    assert specification is not None and specification.loader is not None
    module: ModuleType = importlib.util.module_from_spec(specification)
    """Created the command module for focused contract tests."""

    sys.modules[specification.name] = module
    """Made the generated dataclass resolvable by process-pool pickling."""

    specification.loader.exec_module(module)
    return module


def test_design_uses_the_paper_fixed_effect_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct the six fixed-effect columns used by the paper's model."""
    module: ModuleType = separation_module()
    """Loaded the participant-free separation command."""

    captured: list[object] = []
    """Retained the matrices and design supplied to the model constructor."""

    class FakeModel:
        """Capture the design without entering the numerical estimator."""

        def __init__(self, matrices: object, design: object) -> None:
            captured.extend([matrices, design])

    monkeypatch.setattr(module.asterism, "CensoredComponentModel", FakeModel)
    module.prepare_cell(30, 2)

    assert len(captured) == 2
    assert captured[1].shape == (120, 6)
    assert captured[1][:, 0].tolist() == [1.0] * 120


def test_cell_grid_carries_each_requested_censoring_share() -> None:
    """Keep 52% and 75% fixed instruments as distinct scientific cells."""
    module: ModuleType = separation_module()
    """Loaded the command whose cell grid is under test."""

    jobs: list[Any] = [
        module.CellJob(2, 30, 2, 100, 60, share) for share in (0.52, 0.75)
    ]
    """Declared one otherwise identical cell at each target censoring share."""

    results: list[dict[str, object]] = [
        {
            "records_per_person": job.records,
            "families": job.families,
            "censoring_share": job.censoring_share,
            "summary": {},
            "failures": [],
        }
        for job in jobs
    ]
    """Made one complete worker result for every requested cell coordinate."""

    assert module.validate_complete_cell_results(results, jobs) == results


def fake_cell(job: Any) -> dict[str, object]:
    """Return a complete deterministic cell without numerical fitting."""
    return {
        "records_per_person": job.records,
        "families": job.families,
        "summary": {"coordinate": [job.records, job.families]},
        "failures": [],
    }


def complete_cell(job: Any) -> dict[str, object]:
    """Return a complete report-shaped cell for command orchestration tests."""
    return {
        "records_per_person": job.records,
        "families": job.families,
        "summary": {
            "records_per_person": job.records,
            "families": job.families,
            "replicates_fitted": job.replicates,
            "additive": {"mean": 0.40, "sd": 0.10, "truth": 0.40},
            "person": {"mean": 0.30, "sd": 0.08, "truth": 0.30},
            "correlation_between_estimates": -0.90,
            "sd_of_their_sum": 0.05,
            "split_costs_this_many_times_the_precision": 2.0,
            "additive_bias": 0.0,
            "additive_bias_resolvable_beyond": 0.05,
            "person_bias": 0.0,
            "person_bias_resolvable_beyond": 0.05,
        },
        "failures": [],
    }


def test_cell_runner_preserves_seeds_and_prepares_dense_state_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep original streams while reusing one cell's matrices and factor."""
    module: ModuleType = separation_module()
    """Loaded the cell worker boundary."""

    prepared: list[tuple[int, int]] = []
    """Recorded how often the dense design was prepared."""

    def prepare(families: int, records: int) -> tuple[str, str, int]:
        """Return a tiny prepared-state marker for the worker test."""
        prepared.append((families, records))
        return "factor", "model", 4

    seeds: list[int] = []
    """Recorded the exact streams handed to each replicate."""

    def fit(factor: str, model: str, rows: int, seed: int) -> tuple[float, float]:
        """Return varying valid estimates without entering the numerical core."""
        assert (factor, model, rows) == ("factor", "model", 4)
        seeds.append(seed)
        offset: float = 0.001 * len(seeds)
        """Varied both estimates while leaving their sum unchanged."""

        return 0.40 + offset, 0.30 - offset

    monkeypatch.setattr(module, "prepare_cell", prepare)
    monkeypatch.setattr(module, "fit_prepared_replicate", fit)

    job: Any = module.CellJob(
        records=2,
        families=30,
        replicates=3,
        base_seed=100,
        largest_families=60,
    )
    """Selected a non-gating cell with three original replicate coordinates."""

    result: dict[str, object] = module.run_cell(job)
    """Ran all cell replicates through one prepared dense state."""

    assert prepared == [(30, 2)]
    assert seeds == [132, 8_051, 15_970]
    assert result["records_per_person"] == 2
    assert result["families"] == 30
    assert result["summary"]["replicates_fitted"] == 3
    assert result["failures"] == []


def test_collection_is_worker_count_invariant_and_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the same ordered cells from direct and multi-process schedules."""
    module: ModuleType = separation_module()
    """Loaded campaign orchestration without real fits."""

    jobs: list[Any] = [
        module.CellJob(
            records=records,
            families=families,
            replicates=2,
            base_seed=100,
            largest_families=60,
        )
        for records in (2, 3)
        for families in (30, 60)
    ]
    """Declared four uniquely ordered scientific cells."""

    monkeypatch.setattr(module, "run_cell", fake_cell)
    direct: list[dict[str, object]] = module.collect_cell_results(jobs, workers=1)
    """Collected the direct execution reference."""

    constructor: list[tuple[int, str]] = []
    """Recorded the process count and multiprocessing start method."""

    class FakePool:
        """Model ProcessPoolExecutor while running jobs in-process."""

        def __init__(self, *, max_workers: int, mp_context: Any) -> None:
            constructor.append((max_workers, mp_context.get_start_method()))

        def __enter__(self) -> FakePool:
            """Enter the modelled executor context."""
            return self

        def __exit__(self, *arguments: object) -> None:
            """Close the modelled executor without suppressing exceptions."""

        def map(
            self,
            function: Any,
            submitted: list[Any],
            *,
            chunksize: int,
        ) -> list[dict[str, object]]:
            """Apply the worker boundary in the executor's stable map order."""
            assert chunksize == 1
            return [function(job) for job in submitted]

    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    parallel: list[dict[str, object]] = module.collect_cell_results(jobs, workers=4)
    """Collected the same cells through the multi-process branch."""

    assert parallel == direct
    assert constructor == [(4, "spawn")]


def test_collection_requires_every_unique_cell_coordinate() -> None:
    """Reject missing, duplicate, or unexpected worker output coordinates."""
    module: ModuleType = separation_module()
    """Loaded the complete-grid validator."""

    jobs: list[Any] = [
        module.CellJob(2, 30, 2, 100, 60),
        module.CellJob(2, 60, 2, 100, 60),
    ]
    """Declared a two-cell campaign grid."""

    complete: list[dict[str, object]] = [fake_cell(job) for job in jobs]
    """Made one result for every requested coordinate."""

    assert module.validate_complete_cell_results(complete, jobs) == complete

    with pytest.raises(ValueError, match="COMPONENT_SEPARATION_COORDINATES_INCOMPLETE"):
        module.validate_complete_cell_results(complete[:1], jobs)
    with pytest.raises(ValueError, match="COMPONENT_SEPARATION_COORDINATE_DUPLICATE"):
        module.validate_complete_cell_results([complete[0], complete[0]], jobs)


def test_worker_exceptions_name_the_failed_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make an unexpected numerical worker failure visible with its coordinate."""
    module: ModuleType = separation_module()
    """Loaded the cell worker boundary."""

    def fail(families: int, records: int) -> None:
        """Model an unexpected failure while preparing dense state."""
        raise RuntimeError("numerical failure")

    monkeypatch.setattr(module, "prepare_cell", fail)
    job: Any = module.CellJob(3, 120, 2, 100, 120)
    """Selected the exact coordinate that should reach the error."""

    with pytest.raises(
        RuntimeError,
        match="COMPONENT_SEPARATION_CELL_FAILED records=3 families=120",
    ):
        module.run_cell(job)


def test_command_output_and_scientific_jobs_are_worker_count_invariant(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep seeds, complete coordinates, decisions, and report independent of workers."""
    module: ModuleType = separation_module()
    """Loaded the complete command interface."""

    calls: list[tuple[int, list[Any]]] = []
    """Recorded both requested schedules and their scientific jobs."""

    def collect(jobs: list[Any], workers: int) -> list[dict[str, object]]:
        """Return deterministic complete cells without numerical fitting."""
        calls.append((workers, jobs))
        return [complete_cell(job) for job in jobs]

    monkeypatch.setattr(module, "collect_cell_results", collect)
    common: list[str] = [
        "--replicates",
        "2",
        "--families",
        "30",
        "60",
        "--records",
        "2",
        "3",
        "--seed",
        "100",
        "--no-write",
    ]
    """Fixed every scientific coordinate while varying workers only."""

    first_status: int = module.main([*common, "--workers", "1"])
    """Ran the command through the direct scheduler."""

    first_output: str = capsys.readouterr().out
    """Captured the direct execution report."""

    second_status: int = module.main([*common, "--workers", "4"])
    """Ran the command through the parallel scheduler."""

    second_output: str = capsys.readouterr().out
    """Captured the parallel execution report."""

    assert first_status == second_status == 0
    assert first_output == second_output
    assert [call[0] for call in calls] == [1, 4]
    first_jobs: list[Any] = calls[0][1]
    """Selected the direct run's complete scientific coordinates."""

    second_jobs: list[Any] = calls[1][1]
    """Selected the parallel run's complete scientific coordinates."""

    assert first_jobs == second_jobs
    assert [(job.records, job.families) for job in first_jobs] == [
        (2, 30),
        (2, 60),
        (3, 30),
        (3, 60),
    ]
    assert all(job.base_seed == 100 for job in first_jobs)
    assert all(job.largest_families == 60 for job in first_jobs)


def test_command_rejects_nonpositive_workers() -> None:
    """Refuse a process count that cannot own any design cell."""
    module: ModuleType = separation_module()
    """Loaded argument validation before any numerical work starts."""

    with pytest.raises(SystemExit, match="2"):
        module.main(["--workers", "0", "--no-write"])
