"""Shared mechanics for strict frozen external-reference adapters.

Adapters keep their scientific inputs, public Asterism calls and acceptance
rules in their own modules.  This file only provides deterministic hashing,
fixture loading, exact R/package identity and the optional command-line mode.
Its own SHA-256 is included in every adapter's input identities so changing
these mechanics makes every existing fixture stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION: int = 1
"""Matched the strict envelope contract in external_reference_fixture.py."""

HASH_DECIMALS: int = 12
"""Removed irrelevant last-bit BLAS drift from generated floating-point inputs."""


class ReferenceAdapterError(ValueError):
    """Report an unusable fixture, missing R dependency or malformed output."""


def parse_reference_arguments(description: str) -> argparse.Namespace:
    """Parse optional strict adapter mode while preserving no-argument execution.

    Args:
        description: Module documentation displayed by ``--help``.

    Returns:
        Arguments with either both reference fields or neither.

    Raises:
        SystemExit: If only one strict-reference argument is supplied.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=description)
    """Extended each historical no-argument check with two explicit options."""

    parser.add_argument("--reference-mode", choices=("refresh", "verify"))
    parser.add_argument("--reference-fixture", type=Path)
    arguments: argparse.Namespace = parser.parse_args()
    """Parsed no positional arguments so historical invocation remains unchanged."""

    if (arguments.reference_mode is None) != (arguments.reference_fixture is None):
        parser.error(
            "--reference-mode and --reference-fixture must be supplied together"
        )
    return arguments


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic finite JSON bytes for a participant-free input.

    Args:
        value: JSON-compatible input description.

    Returns:
        UTF-8 bytes with stable key ordering and no insignificant whitespace.
    """
    try:
        rendered: str = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        """Rendered a platform-independent canonical scalar or structured input."""
    except (TypeError, ValueError) as error:
        raise ReferenceAdapterError(
            f"input is not finite canonical JSON: {error}"
        ) from error
    return rendered.encode("utf-8")


def json_identity(name: str, value: object) -> dict[str, str]:
    """Identify one canonical structured input by SHA-256.

    Args:
        name: Stable human-readable input name.
        value: JSON-compatible participant-free input.

    Returns:
        Strict fixture identity record.
    """
    digest: str = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    """Hashed the exact canonical representation rather than Python object identity."""

    return {"name": name, "sha256": digest}


def array_identity(name: str, value: np.ndarray) -> dict[str, str]:
    """Identify one generated numeric array in a platform-stable representation.

    Args:
        name: Stable human-readable input name.
        value: Participant-free generated array.

    Returns:
        Strict fixture identity record including dtype, shape and canonical values.
    """
    array: np.ndarray = np.asarray(value)
    """Accepted any NumPy view without trusting its memory order or byte order."""

    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ReferenceAdapterError(
                f"generated input {name!r} contains non-finite data"
            )
        normalized: np.ndarray = np.round(array.astype("<f8"), HASH_DECIMALS)
        """Quantised harmless last-bit platform drift before exact byte hashing."""
    elif np.issubdtype(array.dtype, np.integer):
        normalized = array.astype("<i8")
        """Normalized integer width and byte order across Mac and Linux."""
    elif np.issubdtype(array.dtype, np.bool_):
        normalized = array.astype("|b1")
        """Represented Boolean masks as one byte per exact status value."""
    else:
        raise ReferenceAdapterError(
            f"generated input {name!r} has unsupported dtype {array.dtype}"
        )
    header: bytes = canonical_json_bytes(
        {"dtype": normalized.dtype.str, "shape": list(normalized.shape)}
    )
    """Bound otherwise identical bytes to their semantic shape and scalar type."""

    digest: Any = hashlib.sha256()
    """Built one streaming digest without materialising a large JSON matrix."""

    digest.update(header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(normalized).tobytes(order="C"))
    return {"name": name, "sha256": digest.hexdigest()}


def support_source_identity() -> dict[str, str]:
    """Identify this shared adapter implementation for stale-fixture detection.

    Returns:
        Strict identity record for this exact source file.
    """
    source: Path = Path(__file__).resolve()
    """Resolved the imported support module rather than trusting the working directory."""

    digest: str = hashlib.sha256(source.read_bytes()).hexdigest()
    """Bound every fixture to the canonicalisation and R-version mechanics used."""

    return {"name": "source:checks/external_reference_adapter.py", "sha256": digest}


def load_fixture(path: Path, check_id: str) -> dict[str, Any]:
    """Load the frozen envelope used by a portable adapter verification.

    Args:
        path: Versioned fixture written by the strict driver.
        check_id: Adapter's fixed manifest check identifier.

    Returns:
        Parsed fixture object.

    Raises:
        ReferenceAdapterError: If the path or minimum envelope identity is invalid.
    """
    if not path.is_file():
        raise ReferenceAdapterError(f"reference fixture is not a file: {path}")
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
        """Parsed only committed JSON and never evaluated fixture-controlled code."""
    except json.JSONDecodeError as error:
        raise ReferenceAdapterError(
            f"reference fixture is invalid JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise ReferenceAdapterError("reference fixture must be a JSON object")
    if parsed.get("adapter_contract_version") != CONTRACT_VERSION:
        raise ReferenceAdapterError(
            f"reference fixture contract must be {CONTRACT_VERSION}"
        )
    if parsed.get("check_id") != check_id:
        raise ReferenceAdapterError("reference fixture check_id does not match adapter")
    return parsed


def r_tool_identity(packages: tuple[str, ...]) -> list[dict[str, object]]:
    """Return exact live R and package versions after loading every package.

    Args:
        packages: R packages whose installed code is part of the comparison.

    Returns:
        One strict Rscript tool record with ordered package versions.

    Raises:
        ReferenceAdapterError: If R or any requested package cannot load.
    """
    executable: str | None = shutil.which("Rscript")
    """Required a real executable in refresh mode rather than recording a placeholder."""

    if executable is None:
        raise ReferenceAdapterError("Rscript is not on the path")
    package_vector: str = ",".join(json.dumps(package) for package in packages)
    """Rendered package names as R string literals without shell interpolation."""

    expression: str = (
        f"ps <- c({package_vector}); "
        "for (p in ps) library(p, character.only=TRUE); "
        'cat("R\\t", as.character(getRversion()), "\\n", sep=""); '
        'for (p in ps) cat(p, "\\t", as.character(packageVersion(p)), "\\n", sep="")'
    )
    """Loaded every dependency before reporting exact executable and package versions."""

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [executable, "--vanilla", "-e", expression],
        capture_output=True,
        check=False,
        text=True,
    )
    """Invoked live R without a shell so paths and names cannot alter the command."""

    if completed.returncode != 0:
        raise ReferenceAdapterError(
            "R package/version probe failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    versions: dict[str, str] = {}
    """Collected the one R version and every requested package version."""

    for line in completed.stdout.splitlines():
        fields: list[str] = line.strip().split("\t")
        """Parsed the explicit tab-delimited probe rather than ambient R banners."""

        if len(fields) == 2:
            versions[fields[0]] = fields[1]
            """Stored one exact identity from the qualified live R process."""
    expected_names: tuple[str, ...] = ("R", *packages)
    """Named the exact identities required for a complete live refresh."""

    if any(not versions.get(name) for name in expected_names):
        raise ReferenceAdapterError(
            f"R version probe omitted identities: expected {expected_names!r}, "
            f"found {versions!r}"
        )
    package_records: list[dict[str, str]] = [
        {"name": package, "version": versions[package]} for package in packages
    ]
    """Preserved caller order so identical refreshes serialize byte-for-byte."""

    return [
        {
            "name": "Rscript",
            "version": versions["R"],
            "packages": package_records,
        }
    ]


def require_r_versions(
    tools: list[dict[str, object]],
    *,
    r_version: str,
    packages: dict[str, str],
) -> None:
    """Require the qualified live R environment named by a release refresh.

    Args:
        tools: Result returned by :func:`r_tool_identity`.
        r_version: Exact qualified R release.
        packages: Exact qualified package versions by name.

    Raises:
        ReferenceAdapterError: If any executable or package identity differs.
    """
    if len(tools) != 1 or tools[0].get("version") != r_version:
        raise ReferenceAdapterError(f"refresh requires R {r_version}, found {tools!r}")
    observed_packages: object = tools[0].get("packages")
    """Read the package identities only after confirming the R tool record."""

    if not isinstance(observed_packages, list):
        raise ReferenceAdapterError("R tool record has no package list")
    observed: dict[str, str] = {
        str(package.get("name")): str(package.get("version"))
        for package in observed_packages
        if isinstance(package, dict)
    }
    """Normalized the live response for one exact qualified-version comparison."""

    if observed != packages:
        raise ReferenceAdapterError(
            f"refresh requires R packages {packages!r}, found {observed!r}"
        )


def emit_reference_record(record: dict[str, Any]) -> int:
    """Print one strict adapter response and return its scientific exit status.

    Args:
        record: Response consumed by the fail-closed fixture driver.

    Returns:
        Zero only when the adapter's prewritten acceptance passed.
    """
    print(json.dumps(record, allow_nan=False, sort_keys=True))
    return 0 if record.get("passed") is True else 1
