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


RETIRED_INTERVAL_FIELDS: frozenset[str] = frozenset(
    {
        "lower_at_bound",
        "upper_at_bound",
    }
)
"""Named the interval fields ADR 0011 retired in favour of the limited pair."""


def test_supported_scientific_checks_read_no_retired_interval_field() -> None:
    """Stop a check from reading an interval field the shared module stopped publishing.

    ADR 0011 decision 4 retired ``lower_at_bound`` for ``lower_limited`` when
    seven copies of the interval recipe became one module. Two coverage checks
    kept reading the retired names, and because a missing key raises
    ``KeyError`` rather than the ``ValueError`` a refusal raises, both crashed
    before scoring a single replicate. A scientific check that cannot run is
    worse than one that fails: the release runner saw a crash where it should
    have seen a measured coverage, and the censored and mixed families went
    unmeasured while looking merely awkward. The statistical methods document
    was already guarded against these names; the checks were not.
    """
    offenders: list[str] = []
    """Collected each check still reading a field the interval module retired."""

    for check_path in sorted((ROOT / "checks").glob("*.py")):
        source: str = check_path.read_text(encoding="utf-8")
        """Read the check source without importing or executing scientific workloads."""

        for retired in sorted(RETIRED_INTERVAL_FIELDS):
            if retired in source:
                offenders.append(f"{check_path.name}: {retired}")
    """Identified every remaining use of the retired interval vocabulary."""

    assert offenders == []
