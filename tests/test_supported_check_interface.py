"""Public-interface contract for the scientific checks supporting Asterism 0.1."""

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the checkout containing the scientific check scripts."""

DEFERRED_PRIVATE_CHECKS: frozenset[str] = frozenset(
    {
        "association_against_solar.py",
        "association_tail.py",
    }
)
"""Named private checks whose analysis families are explicitly outside 0.1."""


def test_supported_scientific_checks_use_only_the_public_python_interface() -> None:
    """Prevent a supported check from depending on positional compiled bindings."""
    check_paths: list[Path] = sorted((ROOT / "checks").glob("*.py"))
    """Collected every maintained scientific check for an exhaustive boundary gate."""

    private_checks: list[str] = []
    """Collected scripts that still name Asterism's private compiled module."""

    for check_path in check_paths:
        source: str = check_path.read_text(encoding="utf-8")
        """Read the check source without importing or executing scientific workloads."""

        if "_core" in source:
            private_checks.append(check_path.name)
    """Identified every direct dependency on the private positional interface."""

    assert private_checks == sorted(DEFERRED_PRIVATE_CHECKS)
