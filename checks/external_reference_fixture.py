"""Verify or refresh one versioned independent-reference fixture.

The release workflow cannot assume that native SOLAR or specialist R packages
are installed on Ubuntu.  An external-reference adapter therefore has two
explicit modes.  ``refresh`` runs the live external program on a qualified
host and returns its outputs and identities.  ``verify`` recomputes Asterism
against those frozen outputs without invoking the external program.

This driver owns the fail-closed fixture envelope.  Existing comparison scripts
must implement the adapter contract before their release rules can become
ready; a historical JSON result is deliberately insufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION: int = 1
"""Version of the deterministic independent-reference fixture envelope."""

CONTRACT_VERSION: int = 1
"""Version required in every adapter refresh and verification response."""


class FixtureError(ValueError):
    """Report an incomplete or stale external-reference fixture."""


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one regular file.

    Args:
        path: File whose exact bytes identify a script or generated input.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        FixtureError: If ``path`` is not a regular file.
    """
    if not path.is_file():
        raise FixtureError(f"required fixture input is not a file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_record(
    adapter: Path,
    fixture: Path,
    mode: str,
) -> dict[str, Any]:
    """Run one strict adapter mode and parse its sole JSON response.

    Args:
        adapter: Repository-relative comparison script implementing the contract.
        fixture: Fixture path supplied to verification or refresh.
        mode: Either ``refresh`` for live external execution or ``verify``.

    Returns:
        Parsed adapter response.

    Raises:
        FixtureError: If the adapter fails or does not return one JSON object.
    """
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            str(adapter),
            "--reference-mode",
            mode,
            "--reference-fixture",
            str(fixture),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    """Ran the adapter without a shell or an implicit missing-tool fallback."""

    if completed.returncode != 0:
        detail: str = completed.stderr.strip() or completed.stdout.strip()
        """Preserved the adapter's actionable failure without treating it as skip."""

        raise FixtureError(
            f"external-reference adapter exited {completed.returncode}: {detail}"
        )
    try:
        payload: object = json.loads(completed.stdout)
        """Required the adapter to emit one unambiguous machine record."""
    except json.JSONDecodeError as error:
        raise FixtureError(f"adapter output is not one JSON object: {error}") from error
    if not isinstance(payload, dict):
        raise FixtureError("adapter output must be a JSON object")
    return payload


def identity_errors(identities: object) -> list[str]:
    """Return errors in generated-input identity records.

    Args:
        identities: Candidate list of stable names and SHA-256 digests.

    Returns:
        All structural identity failures.
    """
    if not isinstance(identities, list) or not identities:
        return ["input_identities must be a nonempty list"]
    errors: list[str] = []
    """Collected duplicate, missing, and malformed input identities."""

    names: list[str] = []
    """Retained names with multiplicity for duplicate detection."""

    for identity in identities:
        if not isinstance(identity, dict):
            errors.append("each input identity must be an object")
            continue
        name: object = identity.get("name")
        """Read the stable description of one deterministic generated input."""

        digest: object = identity.get("sha256")
        """Read the canonical byte or array digest supplied by the adapter."""

        if not isinstance(name, str) or not name:
            errors.append("each input identity needs a nonempty name")
        else:
            names.append(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            errors.append("each input identity needs a lowercase SHA-256")
    if len(names) != len(set(names)):
        errors.append("input identity names must be unique")
    return errors


def tool_errors(tools: object) -> list[str]:
    """Return errors in live external-tool and package versions.

    Args:
        tools: Candidate tool records returned by a live adapter.

    Returns:
        All missing or malformed version records.
    """
    if not isinstance(tools, list) or not tools:
        return ["tools must be a nonempty list"]
    errors: list[str] = []
    """Collected incomplete executable and package identities."""

    for tool in tools:
        if not isinstance(tool, dict):
            errors.append("each tool identity must be an object")
            continue
        for field in ("name", "version"):
            if not isinstance(tool.get(field), str) or not tool[field]:
                errors.append(f"each tool identity needs a nonempty {field}")
        packages: object = tool.get("packages", [])
        """Read optional independently versioned R packages."""

        if not isinstance(packages, list):
            errors.append("tool packages must be a list")
            continue
        for package in packages:
            if not isinstance(package, dict) or not all(
                isinstance(package.get(field), str) and package[field]
                for field in ("name", "version")
            ):
                errors.append("each package needs a nonempty name and version")
    return errors


def fixture_errors(
    fixture: object,
    *,
    check_id: str,
    adapter: Path,
) -> list[str]:
    """Return envelope and exact-source failures for one frozen fixture.

    Args:
        fixture: Parsed candidate fixture.
        check_id: Required manifest check identifier.
        adapter: Current comparison adapter whose bytes must match.

    Returns:
        All reasons the fixture cannot be trusted by a portable check.
    """
    if not isinstance(fixture, dict):
        return ["fixture must be a JSON object"]
    errors: list[str] = []
    """Collected schema, adapter, input, tool, and external-output failures."""

    if fixture.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"fixture schema_version must be {SCHEMA_VERSION}")
    if fixture.get("adapter_contract_version") != CONTRACT_VERSION:
        errors.append(f"adapter_contract_version must be {CONTRACT_VERSION}")
    if fixture.get("check_id") != check_id:
        errors.append("fixture check_id does not match the configured rule")
    adapter_identity: object = fixture.get("adapter")
    """Read the exact comparison script bound into the fixture."""

    if not isinstance(adapter_identity, dict):
        errors.append("fixture adapter identity must be an object")
    else:
        if adapter_identity.get("path") != adapter.as_posix():
            errors.append("fixture adapter path does not match")
        try:
            current_digest: str = sha256(adapter)
            """Identified the current adapter without trusting fixture metadata."""
        except FixtureError as error:
            errors.append(str(error))
        else:
            if adapter_identity.get("sha256") != current_digest:
                errors.append("fixture adapter SHA-256 is stale")
    errors.extend(identity_errors(fixture.get("input_identities")))
    errors.extend(tool_errors(fixture.get("tools")))
    external_outputs: object = fixture.get("external_outputs")
    """Read frozen independent values without accepting an empty placeholder."""

    if not isinstance(external_outputs, dict) or not external_outputs:
        errors.append("external_outputs must be a nonempty object")
    acceptance: object = fixture.get("acceptance")
    """Read the prewritten decision rule without accepting an empty placeholder."""

    if not isinstance(acceptance, dict) or not acceptance:
        errors.append("acceptance must be a nonempty object")
    return errors


def refresh(check_id: str, adapter: Path, fixture_path: Path) -> dict[str, Any]:
    """Run a live adapter and return the deterministic fixture to persist.

    Args:
        check_id: Manifest check identifier being refreshed.
        adapter: Live R or SOLAR comparison adapter.
        fixture_path: Destination also communicated to the adapter.

    Returns:
        Validated deterministic fixture envelope.

    Raises:
        FixtureError: If the live comparison or any identity is incomplete.
    """
    response: dict[str, Any] = adapter_record(adapter, fixture_path, "refresh")
    """Ran the external dependency only in the explicitly requested live mode."""

    expected: dict[str, object] = {
        "reference_fixture_contract": CONTRACT_VERSION,
        "check_id": check_id,
        "passed": True,
        "external_tool_invoked": True,
        "asterism_recomputed": True,
    }
    """Named every decision a genuine live refresh must prove."""

    for field, value in expected.items():
        if response.get(field) != value:
            raise FixtureError(f"live adapter must return {field}={value!r}")
    fixture: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "adapter_contract_version": CONTRACT_VERSION,
        "check_id": check_id,
        "adapter": {"path": adapter.as_posix(), "sha256": sha256(adapter)},
        "input_identities": response.get("input_identities"),
        "tools": response.get("tools"),
        "external_outputs": response.get("external_outputs"),
        "acceptance": response.get("acceptance"),
    }
    """Excluded timestamps and host paths so identical live results are stable."""

    errors: list[str] = fixture_errors(fixture, check_id=check_id, adapter=adapter)
    """Applied the same strict envelope validation used by portable verification."""

    if errors:
        raise FixtureError("; ".join(errors))
    return fixture


def verify(check_id: str, adapter: Path, fixture_path: Path) -> dict[str, Any]:
    """Recompute Asterism against one frozen independent output fixture.

    Args:
        check_id: Manifest check identifier being verified.
        adapter: Portable comparison adapter.
        fixture_path: Existing versioned independent-reference fixture.

    Returns:
        Adapter verification record.

    Raises:
        FixtureError: If fixture identity or numerical comparison fails.
    """
    if not fixture_path.is_file():
        raise FixtureError(
            "external-reference fixture is absent; run explicit live refresh first"
        )
    try:
        fixture: object = json.loads(fixture_path.read_text(encoding="utf-8"))
        """Parsed the versioned fixture without executing any external dependency."""
    except json.JSONDecodeError as error:
        raise FixtureError(f"fixture is not valid JSON: {error}") from error
    errors: list[str] = fixture_errors(fixture, check_id=check_id, adapter=adapter)
    """Rejected stale scripts and incomplete independent provenance first."""

    if errors:
        raise FixtureError("; ".join(errors))
    response: dict[str, Any] = adapter_record(adapter, fixture_path, "verify")
    """Asked the adapter to recompute Asterism using only frozen external outputs."""

    expected: dict[str, object] = {
        "reference_fixture_contract": CONTRACT_VERSION,
        "check_id": check_id,
        "passed": True,
        "external_tool_invoked": False,
        "asterism_recomputed": True,
    }
    """Required an actual portable comparison rather than envelope validation alone."""

    for field, value in expected.items():
        if response.get(field) != value:
            raise FixtureError(f"verify adapter must return {field}={value!r}")
    if response.get("input_identities") != fixture["input_identities"]:  # type: ignore[index]
        raise FixtureError("verify adapter recomputed different input identities")
    return response


def parse_arguments() -> argparse.Namespace:
    """Parse the explicit portable-verification or live-refresh command."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built one shell-free interface shared by all external gates."""

    parser.add_argument("mode", choices=("verify", "refresh"))
    parser.add_argument("--check-id", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the selected fail-closed external-reference operation."""
    arguments: argparse.Namespace = parse_arguments()
    """Read exact paths and mode without consulting ambient tool availability."""

    adapter: Path = arguments.adapter
    """Selected the repository comparison script named by the pass rule."""

    fixture_path: Path = arguments.fixture
    """Selected the versioned independent-reference envelope."""

    try:
        if arguments.mode == "refresh":
            result: dict[str, Any] = refresh(arguments.check_id, adapter, fixture_path)
            """Ran live independent software and built a validated frozen record."""

            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            """Persisted only a successful deterministic refresh."""
        else:
            result = verify(arguments.check_id, adapter, fixture_path)
            """Recomputed Asterism without invoking R or SOLAR."""
    except FixtureError as error:
        record: dict[str, Any] = {
            "check": arguments.check_id,
            "mode": arguments.mode,
            "passed": False,
            "failure": str(error),
        }
        """Turned missing fixtures and adapter contracts into explicit gate failures."""

        print(json.dumps(record, indent=2))
        return 1
    record = {
        "check": arguments.check_id,
        "mode": arguments.mode,
        "passed": True,
        "result": result,
    }
    """Recorded the verified portable result or completed live refresh."""

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
