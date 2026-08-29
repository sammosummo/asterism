"""The installed-wheel standard-receipt release command."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import asterism
import pytest

ANALYSIS_IDS: list[str] = [
    "one_trait_gaussian_heritability",
    "several_covariance_components",
    "bivariate_genetic_correlation",
    "continuous_gene_by_environment",
    "discrete_gene_by_environment",
    "binary_liability_heritability",
    "one_trait_censored",
    "mixed_binary_censored_genetic_correlation",
]
"""Fixed the complete 0.1 supported-analysis inventory independently of the tool."""


def test_direct_command_exposes_its_release_interface() -> None:
    """The manifest's repository-relative runner remained directly executable."""
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "tools/synthetic_analysis_receipts.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    """Invoked the exact direct-script form committed in release.toml."""

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--output" in completed.stdout
    assert "--wheel" in completed.stdout


def test_development_build_cannot_write_synthetic_release_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A development checkout refused before creating apparent release evidence.

    The development build is substituted here rather than assumed from the
    checkout. Reading the live build meant this test could only run while the
    repository was unreleased, which is the opposite of when it is wanted.
    """
    from asterism import _core

    from tools.synthetic_analysis_receipts import (
        SyntheticReceiptError,
        write_synthetic_analysis_receipts,
    )

    development: str = (
        str(_core.__release_manifest__)
        .replace("\nrelease = true", "\nrelease = false")
        .replace('\ncargo_version = "0.1.0"', '\ncargo_version = "0.1.0-dev.0"')
        .replace('\nversion = "0.1.0"', '\nversion = "0.1.0.dev0"')
    )
    """Took the compiled contract and made it a development one."""

    monkeypatch.setattr(_core, "__release_manifest__", development)
    monkeypatch.setattr(
        _core,
        "__release_manifest_sha256__",
        hashlib.sha256(development.encode("utf-8")).hexdigest(),
    )
    monkeypatch.setattr(_core, "__source_dirty__", False)
    """Replaced every coupled field rather than making an impossible mixed build."""

    wheel: Path = tmp_path / "asterism-0.1.0.dev0-cp313-abi3-macosx.whl"
    """Created a named stand-in artifact whose bytes can be committed safely."""

    wheel.write_bytes(b"development wheel stand-in")
    """Gave the command a present immutable file rather than a missing-input error."""

    output: Path = tmp_path / "synthetic-analysis-receipts"
    """Selected an absent destination so any write is observable."""

    with pytest.raises(
        SyntheticReceiptError,
        match="SYNTHETIC_RECEIPT_BUILD_NOT_RELEASED",
    ):
        write_synthetic_analysis_receipts(output, [wheel])
    """Ran the public release command against this development build."""

    assert not output.exists()


def test_all_nine_lazy_jobs_return_converged_supported_fields() -> None:
    """Every receipt callback exercised its complete compiled public inference path."""
    from tools.cross_platform_probe import synthetic_problem
    from tools.synthetic_analysis_receipts import (
        ReceiptJob,
        receipt_jobs,
        subject_order,
    )

    problem: dict[str, Any] = synthetic_problem()
    """Created the same participant-free arrays the release command will commit."""

    commitment: str = asterism.subject_order_commitment(subject_order())
    """Bound every public fit to the shared deterministic row order."""

    jobs: list[ReceiptJob] = receipt_jobs(problem, commitment)
    """Built numerical callbacks without running public preflight prematurely."""

    required: dict[str, set[str]] = {
        "one_trait_gaussian_heritability": {"h2", "interval", "test"},
        "several_covariance_components": {
            "mean_diagonal_component_contributions",
            "mean_diagonal_proportions",
            "mean_diagonal_proportion_interval",
        },
        "bivariate_genetic_correlation": {"rho_g", "interval", "test"},
        "continuous_gene_by_environment": {
            "heritability",
            "genetic_correlation",
            "heritability_interval",
            "genetic_correlation_interval",
            "correlation_test",
            "interaction_test",
        },
        "discrete_gene_by_environment": {
            "genetic_correlation",
            "correlation_interval",
            "gene_by_environment_test",
            "correlation_test",
            "genetic_variance_test",
        },
        "binary_liability_heritability": {"heritability", "interval", "test"},
        "one_trait_censored": {"heritability", "interval", "test"},
        "mixed_binary_censored_genetic_correlation": {
            "genetic_correlation",
            "interval",
            "test",
        },
    }
    """Copied the supported-field contract from accepted ADR 0012 explicitly."""

    assert [job.analysis_id for job in jobs] == ANALYSIS_IDS
    for job in jobs:
        record: dict[str, Any] = job.fit()
        """Ran one real compiled fit, profile and test through documented entry points."""

        assert record["converged"] is True
        assert required[job.analysis_id].issubset(record)
