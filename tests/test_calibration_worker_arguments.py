"""Command-line allocation contracts for the fixed-threshold campaigns."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from checks import (
    mixed_bivariate_calibration,
    mixed_bivariate_coverage,
    tobit_calibration,
)

CAMPAIGNS: tuple[ModuleType, ...] = (
    tobit_calibration,
    mixed_bivariate_calibration,
    mixed_bivariate_coverage,
)
"""Named the three environment-compatible commands that now accept argv."""


def successful_rows(campaign: ModuleType, jobs: list[Any]) -> list[dict[str, object]]:
    """Return complete deterministic records for one command's public report."""
    if campaign is tobit_calibration:
        return [
            {
                "rate": rate,
                "replicate": replicate,
                "instrument_limit": 0.0,
                "achieved_censoring_share": rate,
                "heritability": campaign.TRUE_HERITABILITY,
                "variance": campaign.TRUE_VARIANCE,
                "naive_heritability": campaign.TRUE_HERITABILITY,
            }
            for rate, replicate in jobs
        ]
    if campaign is mixed_bivariate_calibration:
        return [
            {
                "pairing": pairing,
                "replicate": replicate,
                "genetic_correlation": campaign.GENETIC_CORRELATION,
                "residual_correlation": campaign.RESIDUAL_CORRELATION,
                "h2_first": campaign.HERITABILITY[0],
                "h2_second": campaign.HERITABILITY[1],
            }
            for pairing, replicate in jobs
        ]
    return [
        {
            "pairing": pairing,
            "truth": truth,
            "replicate": replicate,
            "covered": True,
            "width": 0.2,
            "rejected": False,
        }
        for pairing, truth, replicate in jobs
    ]


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_no_arguments_preserve_historical_defaults(campaign: ModuleType) -> None:
    """Keep existing interactive and environment-selected invocations unchanged."""
    arguments: object = campaign.parse_arguments([])  # type: ignore[attr-defined]
    """Parsed no explicit allocation through this standalone command."""

    assert arguments.replicates == campaign.REPLICATES  # type: ignore[attr-defined]
    assert arguments.workers == campaign.WORKERS  # type: ignore[attr-defined]


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_explicit_allocation_overrides_defaults(campaign: ModuleType) -> None:
    """Let a release command pin scalable work without ambient variables."""
    arguments: object = campaign.parse_arguments(  # type: ignore[attr-defined]
        ["--replicates", "17", "--workers", "23"]
    )
    """Selected deliberately non-default values visible to campaign orchestration."""

    assert arguments.replicates == 17
    assert arguments.workers == 23


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_explicit_worker_count_reaches_process_pool(
    campaign: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the parsed allocation to the process-pool constructor unchanged."""
    allocations: list[int] = []
    """Recorded each process count requested by the campaign helper."""

    class RecordingExecutor:
        """Stand in for a process pool without executing scientific work."""

        def __init__(self, *, max_workers: int) -> None:
            allocations.append(max_workers)

        def __enter__(self) -> RecordingExecutor:
            return self

        def __exit__(self, *_arguments: object) -> None:
            return None

        def map(
            self,
            function: Callable[[Any], Any],
            jobs: Iterable[Any],
            *,
            chunksize: int,
        ) -> list[Any]:
            del function, jobs, chunksize
            return []

    monkeypatch.setattr(campaign, "ProcessPoolExecutor", RecordingExecutor)
    """Replaced only process creation, leaving the campaign helper itself under test."""

    assert campaign.run_jobs([], 23) == []  # type: ignore[attr-defined]
    assert allocations == [23]


@pytest.mark.parametrize("campaign", CAMPAIGNS)
@pytest.mark.parametrize(
    "arguments",
    (
        ["--replicates", "0"],
        ["--replicates", "-1"],
        ["--workers", "0"],
        ["--workers", "-1"],
    ),
)
def test_allocation_counts_must_be_positive(
    campaign: ModuleType,
    arguments: list[str],
) -> None:
    """Reject empty worker pools and empty scientific denominators before work."""
    with pytest.raises(SystemExit):
        campaign.parse_arguments(arguments)  # type: ignore[attr-defined]


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_no_write_suppresses_only_the_dated_evidence_file(
    campaign: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the complete scientific report while omitting its dated JSON write."""

    def run_jobs(jobs: list[Any], workers: int) -> list[dict[str, object]]:
        """Supply complete fitted rows at the command's public orchestration seam."""
        assert workers == 1
        return successful_rows(campaign, jobs)

    monkeypatch.setattr(campaign, "run_jobs", run_jobs)
    writes: list[tuple[Path, str]] = []
    """Recorded evidence writes without touching the checkout."""

    def write_text(destination: Path, content: str, **_options: object) -> int:
        """Capture the dated evidence payload in memory."""
        writes.append((destination, content))
        return len(content)

    monkeypatch.setattr(Path, "write_text", write_text)
    common: list[str] = ["--replicates", "2", "--workers", "1"]
    """Selected a small complete report independent of process allocation."""

    ordinary_status: int = campaign.main(common)  # type: ignore[attr-defined]
    """Ran the existing evidence-writing command."""

    ordinary_output: str = capsys.readouterr().out
    """Captured its complete scientific report and write confirmation."""

    assert ordinary_status == 0
    assert len(writes) == 1
    evidence_path, evidence_text = writes.pop()
    """Selected the sole dated JSON write made by the ordinary command."""

    assert json.loads(evidence_text)["replicates"] == 2
    written_line: str = f"\nwritten to {evidence_path}\n"
    """Named the only stdout fragment coupled to the suppressed write."""

    assert written_line in ordinary_output

    dry_status: int = campaign.main([*common, "--no-write"])  # type: ignore[attr-defined]
    """Ran the same command with filesystem output disabled."""

    dry_output: str = capsys.readouterr().out
    """Captured the dry run's complete scientific report."""

    assert dry_status == ordinary_status
    assert writes == []
    assert dry_output == ordinary_output.replace(written_line, "")
