"""No check factorises with scipy's Cholesky.

On the machine this was written on -- numpy 2.5.2, scipy 1.18.0, macOS on ARM
-- `scipy.linalg.cholesky(a, lower=True)` returns the factor without clearing
the strictly upper triangle. The result looks like a factor, is the right shape
and the right dtype, and `np.tril` of it is exactly correct, but the product of
it and its transpose is not the matrix for anything above about fifty rows.

It fails silently, which is the whole problem. The first version of
`checks/sequential_against_ghk.py` used it and produced errors of 1e12 in a log
probability, and the first three explanations reached for were all about the
statistics.

Every other script in `checks/` already used `np.linalg.cholesky`, so nothing
in the repository was ever wrong because of this. That is luck rather than
policy, and this test is the policy.

The rule is not "scipy is bad". It is that a factorisation should come from one
place, and `np.linalg.cholesky` is that place here: it returns the lower factor
and nothing else, with no triangle argument to get wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
"""Located the repository root from this test module."""

CHECKS: Path = ROOT / "checks"
"""Located the independent-check scripts governed by this policy."""

SCIPY_CHOLESKY: re.Pattern[str] = re.compile(
    r"(?<!\w)(scipy\.linalg\.)?cholesky\s*\(", re.MULTILINE
)
"""Matched unqualified and SciPy-qualified Cholesky calls."""


def test_no_check_uses_scipy_cholesky() -> None:
    """Keep every independent check on NumPy's lower-factor convention."""
    offenders: list[str] = []
    """Collected scripts that imported or called SciPy's Cholesky routine."""

    for script in sorted(CHECKS.glob("*.py")):
        text: str = script.read_text()
        """Read one independent-check script for source-policy inspection."""

        if (
            "from scipy.linalg import" in text
            and "cholesky" in text.split("from scipy.linalg import")[1].split("\n")[0]
        ):
            offenders.append(f"{script.name} imports cholesky from scipy.linalg")
        for line in text.splitlines():
            stripped: str = line.strip()
            """Removed indentation before classifying the source line."""

            if stripped.startswith("#") or "np.linalg.cholesky" in line:
                continue
            if "scipy.linalg.cholesky" in line and "`" not in line:
                offenders.append(f"{script.name}: {stripped}")
    """Inspected every check for the factorisation routine known to be unsafe here."""

    assert not offenders, (
        "these use scipy's Cholesky, which on this platform returns a factor "
        "with the strictly upper triangle left uncleared; use "
        "np.linalg.cholesky: " + "; ".join(offenders)
    )


def test_numpys_cholesky_is_actually_lower_triangular() -> None:
    """The property the checks rely on, asserted rather than assumed.

    If this ever fails, every simulated draw in `checks/` is wrong and the
    failure will not announce itself anywhere else.
    """
    import numpy as np

    rng: np.random.Generator = np.random.default_rng(0)
    """Created a deterministic generator for positive-definite test matrices."""

    for size in (5, 50, 500):
        root: np.ndarray = rng.standard_normal((size, size))
        """Drew a dense square root candidate at the current matrix size."""

        matrix: np.ndarray = root @ root.T + size * np.eye(size)
        """Constructed a strictly positive-definite covariance matrix."""

        factor: np.ndarray = np.linalg.cholesky(matrix)
        """Computed NumPy's documented lower-triangular Cholesky factor."""

        assert np.abs(np.triu(factor, 1)).max() == 0.0
        assert np.abs(factor @ factor.T - matrix).max() < 1e-8 * size
    """Verified both triangularity and reconstruction across representative sizes."""
