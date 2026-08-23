"""Does Asterism import the final2555 empirical relationship matrix unchanged?

The SAFS WGS project released a PC-Relate additive relationship matrix over
2,555 subjects, and its own status note says what is missing: the matrix is
approved for a named R target, and **another engine must check order, scale,
symmetry and exact coefficient preservation before use.** This is that check
for Asterism.

It is deliberately not a re-derivation. The matrix is somebody else's result and
this asks one question about it: does the number in the file arrive in Asterism
as the same number, against the identifier it belongs to. A relationship matrix
that is subtly reordered fits without complaint and answers a different
question, which is the failure this exists to make impossible.

**The evidence it writes is values free.** Identifiers are hashed rather than
recorded, so the receipt can live in the repository while the matrix stays in
controlled storage. Nothing here copies participant-level material anywhere.

Run it where the matrix lives:

    ASTERISM_KINSHIP_MATRIX=<path to the float64 col-major .bin> \\
    ASTERISM_KINSHIP_IDS=<path to matrix_ids.tsv> \\
    uv run --no-project python checks/empirical_kinship_import.py

It fails rather than skips when the matrix is missing, because a check that
quietly passes when it did not run is worse than no check.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Protocol

import asterism
import numpy as np

# What the released contract says. A disagreement with any of these is the
# finding, not a detail to be tidied away.
CONTRACT: dict[str, int | float | str] = {
    "subjects": 2555,
    "off_diagonal_scale": "2 * PC-Relate kinship",
    "diagonal_scale": "1 + PC-Relate F",
    "minimum_eigenvalue": 0.000377305213515519,
    "repair": "none",
}
"""Pinned identity, scale and definiteness facts for the released matrix."""


class Sha256Digest(Protocol):
    """Operations used from an incrementally updated SHA-256 digest."""

    def update(self, block: bytes) -> None:
        """Add one binary block to the digest state."""

    def hexdigest(self) -> str:
        """Return the completed digest as lower-case hexadecimal text."""


def sha256(path: Path) -> str:
    """Hash a file incrementally without loading participant data into memory.

    Args:
        path: File whose exact bytes should be committed by the receipt.

    Returns:
        Lower-case SHA-256 hexadecimal digest.
    """
    digest: Sha256Digest = hashlib.sha256()
    """Initialised the SHA-256 state used across binary input blocks."""

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


SUBJECT_COLUMNS: tuple[str, ...] = (
    "iid",
    "id",
    "subject_id",
    "sample.id",
    "sample_id",
)
"""Accepted explicit identifier headers in preference order."""


def read_ids(path: Path) -> list[str]:
    """The matrix's own identifier order, taken from the named column.

    **The column has to be named, not positional.** This file leads with
    `matrix_index`, so taking the first column gives the row number rather than
    the subject, and every coefficient would then be attributed to the wrong
    person while every structural check still passed. That is the exact failure
    this whole script exists to prevent, and it is available in its own input.

    Args:
        path: Tab-delimited identifier file paired with the relationship matrix.

    Returns:
        Subject identifiers in the matrix's exact row and column order.

    Raises:
        SystemExit: If the file is empty or has no accepted subject column.
    """
    lines: list[str] = [
        line.rstrip("\n") for line in path.read_text().splitlines() if line.strip()
    ]
    """Read non-empty tab-delimited rows while preserving their order."""

    if not lines:
        raise SystemExit(f"{path} is empty")
    header: list[str] = [heading.strip().lower() for heading in lines[0].split("\t")]
    """Normalised the header solely for explicit column-name matching."""

    column: int | None = next(
        (header.index(name) for name in SUBJECT_COLUMNS if name in header), None
    )
    """Located the first recognised subject column without positional guessing."""

    if column is None:
        raise SystemExit(
            f"{path} has no subject column. Its header is {header}; one of "
            f"{SUBJECT_COLUMNS} is needed, because taking a column by position "
            "is how a matrix gets silently reordered."
        )
    return [line.split("\t")[column].strip() for line in lines[1:]]


def main() -> int:
    """Validate and receipt one controlled empirical relationship matrix.

    Returns:
        Zero when identity, structure and exact preservation checks pass,
        otherwise one.

    Raises:
        SystemExit: If required paths or structural input requirements are absent.
    """
    matrix_path: str | None = os.environ.get("ASTERISM_KINSHIP_MATRIX")
    """Read the controlled matrix path without copying its contents."""

    ids_path: str | None = os.environ.get("ASTERISM_KINSHIP_IDS")
    """Read the controlled identifier-order path paired with the matrix."""

    if not matrix_path or not ids_path:
        raise SystemExit(
            "Set ASTERISM_KINSHIP_MATRIX and ASTERISM_KINSHIP_IDS. This check "
            "reads the matrix where it lives and never copies it."
        )
    matrix_file, ids_file = Path(matrix_path), Path(ids_path)
    """Converted both validated environment paths to path objects."""

    for path in (matrix_file, ids_file):
        if not path.exists():
            raise SystemExit(f"missing: {path}")

    # A fixture run exercises the machinery on a synthetic matrix. It relaxes
    # only the two checks that identify *this* release -- the subject count and
    # the exact minimum eigenvalue -- because a synthetic matrix is honestly a
    # different matrix. Everything structural still has to hold.
    fixture: bool = bool(os.environ.get("ASTERISM_KINSHIP_FIXTURE"))
    """Selected the existing synthetic-fixture reporting mode when requested."""

    if fixture:
        print("FIXTURE RUN: identity checks relaxed, structure still enforced.\n")

    ids: list[str] = read_ids(ids_file)
    """Read subject identifiers in the matrix's claimed order."""

    n: int = len(ids)
    """Counted matrix subjects from the identifier contract."""

    failures: list[str] = []
    """Collected every identity, structure and preservation failure."""

    print(f"identifiers: {n}")
    if n != CONTRACT["subjects"] and not fixture:
        failures.append(f"expected {CONTRACT['subjects']} subjects, found {n}")
    if len(set(ids)) != n:
        failures.append("the identifier file repeats a subject")

    expected_bytes: int = n * n * 8
    """Calculated exact bytes for a square float64 matrix of this order."""

    actual_bytes: int = matrix_file.stat().st_size
    """Read the controlled matrix's byte count without opening its values."""

    print(f"matrix bytes: {actual_bytes} (expected {expected_bytes})")
    if actual_bytes != expected_bytes:
        failures.append(
            f"the matrix is {actual_bytes} bytes, not {expected_bytes}; it is "
            "not a {n} by {n} float64 matrix"
        )
        raise SystemExit("\n".join(failures))

    # Column-major, as the contract names it. Reading it row-major would
    # transpose it, which a symmetric matrix hides -- so the shape is checked
    # against the identifier order rather than trusted.
    matrix: np.ndarray = np.fromfile(matrix_file, dtype=np.float64).reshape(
        (n, n),
        order="F",
    )
    """Loaded the matrix in its contracted column-major float64 representation."""

    finite: bool = bool(np.isfinite(matrix).all())
    """Checked that every supplied relationship coefficient is finite."""

    print(f"all finite: {finite}")
    if not finite:
        failures.append("the matrix holds a non-finite value")

    asymmetry: float = float(np.abs(matrix - matrix.T).max())
    """Measured the maximum exact symmetry discrepancy."""

    print(f"max |A - A'|: {asymmetry:g}")
    if asymmetry != 0.0:
        failures.append(f"the matrix is not exactly symmetric: {asymmetry:g}")

    diagonal: np.ndarray = np.diag(matrix)
    """Selected self-relationships for scale validation and evidence."""

    off: np.ndarray = matrix[~np.eye(n, dtype=bool)]
    """Selected all off-diagonal relationships for scale evidence."""

    print(
        f"diagonal   : min {diagonal.min():.6f} max {diagonal.max():.6f} "
        f"mean {diagonal.mean():.6f}   [{CONTRACT['diagonal_scale']}]"
    )
    print(
        f"off-diagonal: min {off.min():.6f} max {off.max():.6f}   "
        f"[{CONTRACT['off_diagonal_scale']}]"
    )
    # A diagonal of one plus the inbreeding coefficient sits at or just above
    # one. A diagonal of exactly one everywhere would mean the scale is the
    # other convention and every inbreeding coefficient has been discarded.
    if np.allclose(diagonal, 1.0, atol=1e-12):
        failures.append(
            "every diagonal entry is one, so this is not the 1 + F scale the "
            "contract names"
        )

    smallest: float = float(np.linalg.eigvalsh(matrix).min())
    """Calculated the minimum symmetric eigenvalue for definiteness and identity."""

    print(
        f"minimum eigenvalue: {smallest:.6e} "
        f"(contract {CONTRACT['minimum_eigenvalue']:.6e})"
    )
    if smallest <= 0.0:
        failures.append(f"the matrix is not positive definite: {smallest:.6e}")
    if abs(smallest - CONTRACT["minimum_eigenvalue"]) > 1e-9:
        failures.append(
            f"the minimum eigenvalue is {smallest:.6e}, not the contracted "
            f"{CONTRACT['minimum_eigenvalue']:.6e} -- this is a different matrix"
        )

    # **The part that matters.** Asterism subsets and reorders a supplied
    # matrix; it does not rebuild it. So the coefficient that arrives must be
    # the coefficient in the file, for the pair it belongs to.
    values: dict[str, np.ndarray] = {"trait": np.zeros(n)}
    """Created a disposable trait needed to exercise the public alignment boundary."""

    kept_same: dict[str, np.ndarray | list[str]] = asterism.align(
        relationship=matrix, relationship_ids=ids, ids=ids, keep=ids, **values
    )
    """Aligned the matrix while retaining its original identifier order."""

    identity_error: float = float(
        np.abs(np.asarray(kept_same["relationship"]) - matrix).max()
    )
    """Measured exact coefficient preservation in the original order."""

    print(f"exact preservation, same order   : max difference {identity_error:g}")
    if identity_error != 0.0:
        failures.append(
            f"importing in the matrix's own order changed a coefficient by "
            f"{identity_error:g}"
        )

    # Reordering is where an identifier mismatch shows itself. Reversing the
    # order must permute the matrix and nothing else.
    reversed_ids: list[str] = list(reversed(ids))
    """Constructed a deterministic alternative subject order."""

    permuted: dict[str, np.ndarray | list[str]] = asterism.align(
        relationship=matrix, relationship_ids=ids, ids=ids, keep=reversed_ids, **values
    )
    """Asked the public alignment boundary to reverse rows and columns together."""

    expected: np.ndarray = matrix[::-1, ::-1]
    """Constructed the exact matrix expected under reversed identifier order."""

    permute_error: float = float(
        np.abs(np.asarray(permuted["relationship"]) - expected).max()
    )
    """Measured exact coefficient preservation after reordering."""

    print(f"exact preservation, reversed order: max difference {permute_error:g}")
    if permute_error != 0.0:
        failures.append(
            f"reordering changed a coefficient by {permute_error:g}, so the "
            "matrix and its identifiers are not being kept together"
        )

    receipt: dict[
        str,
        str | int | float | bool | dict[str, int | float | str] | list[str],
    ] = {
        "what": "Asterism import receipt for the final2555 empirical "
        "relationship matrix",
        "date": date.today().isoformat(),
        "subjects": n,
        "matrix_sha256": sha256(matrix_file),
        # Values free: the identifiers themselves never leave controlled
        # storage, but their exact order is pinned by this hash.
        "identifier_order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "max_asymmetry": asymmetry,
        "diagonal": {
            "min": float(diagonal.min()),
            "max": float(diagonal.max()),
            "mean": float(diagonal.mean()),
        },
        "off_diagonal": {"min": float(off.min()), "max": float(off.max())},
        "minimum_eigenvalue": smallest,
        "exact_preservation_same_order": identity_error == 0.0,
        "exact_preservation_reordered": permute_error == 0.0,
        "contract": CONTRACT,
        "passed": not failures,
        "failures": failures,
    }
    """Assembled values-free identity, structure and preservation evidence."""

    out: Path = (
        Path(__file__).resolve().parent.parent
        / "evidence"
        / (f"empirical-kinship-import-{receipt['date']}.json")
    )
    """Selected the dated repository evidence path for the import receipt."""

    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwritten to {out}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: the matrix imports into Asterism unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
