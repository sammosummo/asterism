"""Public command tests for Asterism's Python style checker."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root containing the public checker command."""


def run_style_checker(path: Path) -> subprocess.CompletedProcess[str]:
    """Run the public style checker against one isolated source file.

    Args:
        path: Python source file to check.

    Returns:
        The completed checker process, including its diagnostic output.
    """
    return subprocess.run(
        [sys.executable, "tools/check_python_style.py", str(path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_style_checker_rejects_an_unannotated_local(tmp_path: Path) -> None:
    """Require the checker to enforce first-assignment annotations."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        "def calculate(value: int) -> int:\n"
        "    result = value + 1\n"
        "    return result\n",
        encoding="utf-8",
    )
    """Wrote a function whose first local assignment lacks its annotation."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 1
    assert "unannotated-local" in completed.stderr
    assert ":2:" in completed.stderr


def test_style_checker_requires_complete_function_annotations(tmp_path: Path) -> None:
    """Require annotations on callable parameters and return values."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text("def calculate(value):\n    return value + 1\n", encoding="utf-8")
    """Wrote a function with neither parameter nor return annotations."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 1
    assert "missing-parameter-annotation" in completed.stderr
    assert "missing-return-annotation" in completed.stderr


def test_style_checker_requires_documentation_after_an_assignment(
    tmp_path: Path,
) -> None:
    """Require a following string to explain a non-trivial assignment."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        "def calculate(value: int) -> int:\n"
        "    result: int = value + 1\n"
        "    return result\n",
        encoding="utf-8",
    )
    """Wrote an annotated transformation without its following documentation."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 1
    assert "missing-following-doc" in completed.stderr
    assert ":2:" in completed.stderr


def test_style_checker_accepts_a_narrow_reasoned_exception(tmp_path: Path) -> None:
    """Allow one named rule only when the adjacent exception explains why."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        "def calculate(value: int) -> int:\n"
        "    # asterism-style: allow unannotated-local -- assigned by a protocol\n"
        "    result = value + 1\n"
        '    """Calculated the protocol result."""\n'
        "    return result\n",
        encoding="utf-8",
    )
    """Wrote one unannotated local with the guide's narrow exception syntax."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 0, completed.stderr


def test_style_checker_rejects_an_unexplained_private_helper(tmp_path: Path) -> None:
    """Require private helpers to identify their concrete reason for existing."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        "def _increment(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    """Wrote a private helper without a narrow exception explaining its purpose."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 1
    assert "private-helper" in completed.stderr


def test_style_checker_rejects_a_single_use_nested_helper(tmp_path: Path) -> None:
    """Keep a one-off calculation local unless extraction has a stated reason."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        "def calculate(value: int) -> int:\n"
        '    """Calculate the adjusted value."""\n'
        "    def increment(item: int) -> int:\n"
        '        """Increment one value."""\n'
        "        return item + 1\n"
        "    return increment(value)\n",
        encoding="utf-8",
    )
    """Wrote a nested helper extracted for only one call."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 1
    assert "single-use-helper" in completed.stderr


def test_style_checker_rejects_an_unknown_exception_rule(tmp_path: Path) -> None:
    """Keep the exception vocabulary narrow and machine-reviewable."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        "value: int = 1\n"
        '"""Selected the example value."""\n'
        "# asterism-style: allow everything -- this is intentionally too broad\n",
        encoding="utf-8",
    )
    """Wrote a broad exception outside the guide's four-rule vocabulary."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 1
    assert "invalid-style-exception" in completed.stderr


def test_style_checker_discovers_maintained_sources(tmp_path: Path) -> None:
    """Make the default gate cover maintained source areas automatically."""
    source_directory: Path = tmp_path / "python" / "asterism"
    """Selected one of the maintained source roots named by the style guide."""

    source_directory.mkdir(parents=True)
    """Created the isolated package-source tree."""

    source: Path = source_directory / "example.py"
    """Selected a package module for automatic discovery."""

    source.write_text("def calculate(value):\n    return value + 1\n", encoding="utf-8")
    """Wrote a module carrying detectable annotation failures."""

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "tools/check_python_style.py", "--root", str(tmp_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    """Ran automatic maintained-source discovery through the public command."""

    assert completed.returncode == 1
    assert str(source) in completed.stderr
    assert "missing-parameter-annotation" in completed.stderr


def test_style_checker_handles_conditional_expressions(tmp_path: Path) -> None:
    """Do not mistake an expression's alternative for a statement suite."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        "value: int = 1 if True else 2\n"
        '"""Selected the value through a conditional expression."""\n',
        encoding="utf-8",
    )
    """Wrote a documented conditional expression with no style violation."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 0, completed.stderr


def test_style_checker_ignores_exception_text_inside_a_string(tmp_path: Path) -> None:
    """Treat only comments as directives, never source or documentation text."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        'TEXT: str = "# asterism-style: allow everything -- quoted text"\n'
        '"""Stored text which happens to describe an exception marker."""\n',
        encoding="utf-8",
    )
    """Wrote an exception-shaped string which is not a source directive."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 0, completed.stderr


def test_style_checker_allows_rebinding_after_the_first_annotation(
    tmp_path: Path,
) -> None:
    """Require an annotation once per scope rather than on every rebinding."""
    source: Path = tmp_path / "example.py"
    """Selected an isolated maintained-source fixture."""

    source.write_text(
        "value: int = 1\n"
        '"""Initialised the example value."""\n'
        "value = value + 1\n"
        '"""Incremented the already annotated value."""\n',
        encoding="utf-8",
    )
    """Wrote a documented rebinding after an annotated first assignment."""

    completed: subprocess.CompletedProcess[str] = run_style_checker(source)
    """Checked the fixture through the public command."""

    assert completed.returncode == 0, completed.stderr


def run_discovered_style_checker(
    root: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    """Run maintained-tree discovery against an isolated repository root.

    Args:
        root: Isolated repository root carrying maintained Python files.
        options: Additional public checker options placed before discovery.

    Returns:
        The completed checker command including zero-debt diagnostics.
    """
    return subprocess.run(
        [
            sys.executable,
            "tools/check_python_style.py",
            "--root",
            str(root),
            *options,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_default_and_explicit_zero_debt_modes_report_every_violation(
    tmp_path: Path,
) -> None:
    """Keep the default and compatibility spellings equally fail-closed."""
    source_directory: Path = tmp_path / "python" / "asterism"
    """Selected the maintained package area used by whole-tree discovery."""

    source_directory.mkdir(parents=True)
    """Created the isolated maintained package area."""

    source: Path = source_directory / "example.py"
    """Selected one maintained source carrying two annotation violations."""

    source.write_text("def example(value):\n    return value\n", encoding="utf-8")
    """Created debt that neither public command spelling may accept."""

    default: subprocess.CompletedProcess[str] = run_discovered_style_checker(tmp_path)
    """Ran the default whole-tree zero-debt gate."""

    explicit: subprocess.CompletedProcess[str] = run_discovered_style_checker(
        tmp_path, "--no-baseline"
    )
    """Ran the retained explicit spelling of the same zero-debt gate."""

    assert default.returncode == 1
    assert explicit.returncode == 1
    assert default.stderr == explicit.stderr
    assert str(source) in default.stderr
    assert "missing-parameter-annotation" in default.stderr
    assert "missing-return-annotation" in default.stderr


def test_style_checker_has_no_command_that_accepts_debt(tmp_path: Path) -> None:
    """Reject the retired baseline-writing option without creating an inventory."""
    completed: subprocess.CompletedProcess[str] = run_discovered_style_checker(
        tmp_path, "--write-baseline"
    )
    """Requested the removed debt-acceptance command through the public CLI."""

    assert completed.returncode == 2
    assert "unrecognized arguments: --write-baseline" in completed.stderr
    assert not list(tmp_path.glob("*.json"))
