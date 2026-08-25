"""Repository-level contracts for ordinary and release automation."""

import ast
import tomllib
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root containing hosted automation."""


def test_quality_workflow_runs_every_ordinary_gate() -> None:
    """Keep package-wide engineering checks visible in hosted automation."""
    workflow: str = (ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    """Read the ordinary codebase-quality workflow as its public configuration."""

    required_commands: tuple[str, ...] = (
        "cargo fmt --check",
        "cargo clippy --locked --all-features --all-targets -- -D warnings",
        "cargo test --release --locked --all-features",
        "uv lock --check",
        "ruff format --check",
        "ruff check",
        "tools/check_python_style.py",
        "tools/check_release.py --metadata",
    )
    """Named every ordinary gate fixed by the accepted release contract."""

    for command in required_commands:
        assert command in workflow


def test_wheel_workflow_builds_and_tests_the_promised_artifacts() -> None:
    """Build each platform wheel once and install it on both Python versions."""
    workflow: str = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
    """Read the installed-wheel workflow as its public configuration."""

    required_fragments: tuple[str, ...] = (
        "macos-15",
        "ubuntu-24.04",
        "x86_64-unknown-linux-gnu",
        'manylinux: "2_17"',
        'manylinux: "off"',
        "rust-toolchain: 1.97.1",
        'test "$(uname -m)"',
        'python: ["3.13", "3.14"]',
        "PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b",
        "maturin-version: v1.14.1",
        "tools/check_release.py --metadata",
        "pytest -q tests",
        "tools/cross_platform_probe.py",
        "tools/compare_cross_platform.py",
        "asterism-cross-platform-probe-",
        "asterism-cross-platform-agreement",
    )
    """Named the architecture, interpreter and installed-wheel guarantees."""

    for fragment in required_fragments:
        assert fragment in workflow


def test_every_python_build_uses_the_locked_maturin() -> None:
    """Keep build isolation, development and hosted wheels on one Maturin."""
    project: dict[str, object] = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    """Read the build-system and development dependency declarations."""
    wheels: str = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
    """Read the Maturin version selected for fixed hosted wheels."""
    locked_version: str = "1.14.1"
    """Named the version resolved and hashed in the checked-in uv lockfile."""

    assert project["build-system"]["requires"] == [f"maturin=={locked_version}"]
    assert f"maturin=={locked_version}" in project["dependency-groups"]["dev"]
    assert f"maturin-version: v{locked_version}" in wheels


def test_every_hosted_python_job_uses_one_uv_release() -> None:
    """Keep lock resolution and wheel installation on one exact uv build."""
    expected: str = 'version: "0.11.21"'
    """Selected the verified uv release used to refresh the committed lockfile."""

    for workflow_name in ("quality.yml", "wheels.yml", "release.yml"):
        workflow: str = (ROOT / ".github/workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        """Read one hosted workflow which invokes uv for release work."""

        assert "astral-sh/setup-uv@v6" in workflow
        assert expected in workflow
    """Required the exact uv release wherever hosted automation installs uv."""


def test_every_rust_build_uses_the_repository_toolchain() -> None:
    """Keep local, quality-CI and fixed-wheel compilers on one corrective release."""
    toolchain: dict[str, object] = tomllib.loads(
        (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    )
    """Read the exact compiler selected by repository configuration."""
    cargo: dict[str, object] = tomllib.loads(
        (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    )
    """Read the minimum compiler accepted by Cargo metadata."""
    quality: str = (ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    """Read the ordinary Rust gate's explicit compiler installation."""
    wheels: str = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
    """Read the compiler requested inside the manylinux wheel action."""
    version: object = toolchain["toolchain"]["channel"]
    """Selected the one exact repository compiler release."""

    assert version == "1.97.1"
    assert cargo["package"]["rust-version"] == version
    assert f"RUSTUP_TOOLCHAIN: {version}" in quality
    assert f"rust-toolchain: {version}" in wheels


def test_release_workflow_fails_closed_and_publishes_checksums() -> None:
    """Prevent tagging until configured pass rules permit an automatic release."""
    workflow: str = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    """Read the private-release workflow as its public configuration."""

    assert "tools/check_release.py --requested" in workflow
    assert "tools/check_release.py --release" in workflow
    assert ".release-venv/bin/python tools/run_scientific_release.py" in workflow
    assert "uv export --frozen --all-groups --no-emit-project" in workflow
    assert "--evidence release-evidence/evidence.json" in workflow
    assert (
        "--cross-platform-agreement cross-platform-agreement/agreement.json" in workflow
    )
    assert "--wheel dist/*.whl" in workflow
    assert "sha256sum" in workflow
    assert "gh release create" in workflow


def test_the_release_detect_step_needs_nothing_it_is_not_given() -> None:
    """The step that decides whether to release must run where nothing is built.

    `release.yml` decides whether a release was asked for before it builds
    anything, so its `detect` job sets up a bare interpreter and installs no
    dependencies at all. That is deliberate: the question is only what version
    the manifest names.

    `check_release.py` answered it by importing its heavier siblings at module
    scope, and those import `asterism` and NumPy because measuring a release
    means running it. So the only step that can start a release could not run,
    and a release could never have started. It failed on `main` exactly that way.

    Such siblings are imported where they are used instead. This checks the
    import graph rather than the behaviour, because reproducing the job means
    reproducing an interpreter with nothing installed.
    """
    source: str = (ROOT / "tools/check_release.py").read_text(encoding="utf-8")
    """Read the script the detect job runs, as automation sees it."""

    tree: ast.Module = ast.parse(source)
    """Parsed it without importing it, which would need what it must not need."""

    forbidden: frozenset[str] = frozenset(
        {"asterism", "numpy", "run_scientific_release"}
    )
    """Named what the detect job does not have and must not be asked for."""

    at_module_scope: list[str] = []
    """Collected every name imported before any function body runs."""

    for node in tree.body:
        if isinstance(node, ast.Import):
            at_module_scope.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            at_module_scope.append(node.module.lstrip(".").split(".")[0])
    """Walked only the top level, which is what an import of this module runs."""

    assert forbidden.isdisjoint(at_module_scope), (
        "tools/check_release.py imports "
        f"{sorted(forbidden.intersection(at_module_scope))} at module scope, which "
        "the release workflow's detect job does not install"
    )

    workflow: str = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    """Read the release workflow to confirm the detect step is still that script."""

    assert "check_release.py --requested" in workflow


def test_ordinary_automation_runs_once_for_a_commit_and_not_twice() -> None:
    """An unfiltered `push:` doubles the bill for no extra coverage.

    `push:` with no branch filter fires on every branch, and `pull_request:`
    fires again on the same commit the moment a pull request is open, so every
    commit under review ran both matrices twice over. Hosted minutes are finite
    on a private repository and the Mac runners bill at ten times the Linux
    ones, so the duplicate half was the larger share of what a branch cost --
    and it is what exhausted the month.

    Restricting `push` to `main` leaves branches covered by `pull_request`
    alone and `main` covered by `push` alone. `release.yml` is deliberately not
    checked here: it triggers on `main` only already, and nothing about it may
    be cancelled part-way.

    This reads the file as text, as the checks above it do, because the one
    thing worth avoiding here is making the test suite depend on a YAML parser
    the package does not otherwise need.
    """
    for name in ("quality.yml", "wheels.yml"):
        workflow: str = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
        """Read one ordinary workflow as its public configuration."""

        assert "  push:\n    branches: [main]\n" in workflow, (
            f"{name} runs on every push to every branch, which duplicates every "
            "pull request's matrix"
        )
        assert "  pull_request:\n" in workflow
        assert (
            "cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}" in workflow
        ), f"{name} must supersede superseded branch runs, and never cancel main"
