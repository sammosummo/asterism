"""Public contract tests for statistical methods and bibliography integrity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.check_statistical_references import validate_references

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root containing the public documentation checker."""


def run_reference_checker(
    methods_path: Path,
    bibliography_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the checker against explicit methods and bibliography paths.

    Args:
        methods_path: Markdown specification to validate.
        bibliography_path: BibTeX file against which citation keys resolve.

    Returns:
        Completed checker process with captured diagnostics.
    """
    return subprocess.run(
        [
            sys.executable,
            "tools/check_statistical_references.py",
            "--methods",
            str(methods_path),
            "--bibliography",
            str(bibliography_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def copy_statistical_documentation(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the canonical documentation and its local targets into a fixture.

    Args:
        tmp_path: Isolated test directory.

    Returns:
        Paths to the copied methods document and bibliography.
    """
    fixture_docs: Path = tmp_path / "docs"
    """Selected the isolated documentation root used by a negative test."""

    fixture_docs.mkdir()

    relative_targets: tuple[Path, ...] = (
        Path("statistical-methods.md"),
        Path("references.bib"),
        Path("adr/0012-small-analysis-ready-releases.md"),
        Path("adr/0013-scientific-support-does-not-remove-public-models.md"),
        Path("research/asterism-0.1-statistical-method-sources.md"),
    )
    """Named every local file whose existence the copied methods document expects."""

    for relative_target in relative_targets:
        source_path: Path = ROOT / "docs" / relative_target
        """Located one canonical source file required by the fixture."""

        fixture_path: Path = fixture_docs / relative_target
        """Selected the corresponding repository-relative fixture path."""

        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(
            source_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    """Copied the methods document and each link target without participant data."""

    return fixture_docs / "statistical-methods.md", fixture_docs / "references.bib"


def test_canonical_statistical_references_pass() -> None:
    """Require all canonical citation keys, metadata, equations, and links to pass."""
    completed: subprocess.CompletedProcess[str] = run_reference_checker(
        ROOT / "docs" / "statistical-methods.md",
        ROOT / "docs" / "references.bib",
    )
    """Ran the public integrity command against the canonical documents."""

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_checker_includes_analyses_introduced_in_the_current_release(
    tmp_path: Path,
) -> None:
    """Do not omit a 0.2 analysis from a 0.2 methods-coverage check."""
    methods_path: Path = tmp_path / "statistical-methods.md"
    """Selected the minimal canonical-document fixture."""

    bibliography_path: Path = tmp_path / "references.bib"
    """Selected an empty bibliography because the fixture has no citations."""

    manifest_path: Path = tmp_path / "release.toml"
    """Selected the versioned support fixture."""
    methods_path.write_text(
        "# Statistical methods\n\n"
        "## What a release requires\n\n"
        "## Numerical implementation\n",
        encoding="utf-8",
    )
    bibliography_path.write_text("", encoding="utf-8")
    manifest_path.write_text(
        'version = "0.2.0.dev0"\n'
        "[[analyses]]\n"
        'id = "new_0_2_analysis"\n'
        'support = "supported_in_0_2"\n'
        'supported_quantities = ["new_quantity"]\n',
        encoding="utf-8",
    )

    problems: list[str] = validate_references(
        methods_path,
        bibliography_path,
        manifest_path,
    )
    """Validated coverage against an analysis introduced in 0.2."""

    assert "supported analysis is absent from methods: new_0_2_analysis" in problems
    assert (
        "supported quantity is absent from methods: new_0_2_analysis.new_quantity"
        in problems
    )


def test_checker_rejects_a_missing_local_markdown_target(tmp_path: Path) -> None:
    """Catch stale ADR and research-note slugs before documentation release."""
    copied_paths: tuple[Path, Path] = copy_statistical_documentation(tmp_path)
    """Created a complete isolated copy before introducing one broken link."""

    methods_path: Path = copied_paths[0]
    """Selected the copied methods specification for the broken-link fixture."""

    bibliography_path: Path = copied_paths[1]
    """Selected the copied bibliography kept valid in the broken-link fixture."""

    methods_path.write_text(
        methods_path.read_text(encoding="utf-8")
        + "\n[Missing local target](adr/does-not-exist.md)\n",
        encoding="utf-8",
    )

    completed: subprocess.CompletedProcess[str] = run_reference_checker(
        methods_path,
        bibliography_path,
    )
    """Validated the fixture containing one deliberately stale Markdown link."""

    assert completed.returncode == 1
    assert "local Markdown target does not exist: adr/does-not-exist.md" in (
        completed.stderr
    )


def test_checker_rejects_an_unresolved_citation_key(tmp_path: Path) -> None:
    """Require every rendered author-year citation to resolve in BibTeX."""
    copied_paths: tuple[Path, Path] = copy_statistical_documentation(tmp_path)
    """Created a complete isolated copy before corrupting one citation key."""

    methods_path: Path = copied_paths[0]
    """Selected the copied methods specification for the citation fixture."""

    bibliography_path: Path = copied_paths[1]
    """Selected the copied bibliography kept valid in the citation fixture."""

    methods_text: str = methods_path.read_text(encoding="utf-8")
    """Read the copied methods document for a controlled one-key mutation."""

    methods_text = methods_text.replace("#wilks1938", "#missingCitation", 1)
    """Replaced one canonical key while preserving its author-year label."""

    methods_path.write_text(methods_text, encoding="utf-8")

    completed: subprocess.CompletedProcess[str] = run_reference_checker(
        methods_path,
        bibliography_path,
    )
    """Validated the fixture containing one deliberately unresolved citation."""

    assert completed.returncode == 1
    assert "citation key does not resolve: missingCitation" in completed.stderr
