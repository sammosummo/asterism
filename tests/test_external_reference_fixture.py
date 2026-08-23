"""Contracts for portable frozen independent-reference evidence."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from checks.external_reference_fixture import (
    FixtureError,
    fixture_errors,
    refresh,
    verify,
)


def adapter_source() -> str:
    """Return a tiny adapter implementing both strict fixture modes."""
    return textwrap.dedent(
        """
        import argparse
        import json

        parser = argparse.ArgumentParser()
        parser.add_argument("--reference-mode", required=True)
        parser.add_argument("--reference-fixture", required=True)
        arguments = parser.parse_args()
        identities = [{"name": "canonical synthetic arrays", "sha256": "a" * 64}]
        record = {
            "reference_fixture_contract": 1,
            "check_id": "independent_smoke",
            "passed": True,
            "external_tool_invoked": arguments.reference_mode == "refresh",
            "asterism_recomputed": True,
            "input_identities": identities,
        }
        if arguments.reference_mode == "refresh":
            record.update({
                "tools": [{
                    "name": "Rscript",
                    "version": "fixture-1.0",
                    "packages": [{"name": "regress", "version": "fixture-1.0"}],
                }],
                "external_outputs": {"estimate": 0.5},
                "acceptance": {"absolute_tolerance": 1e-6},
            })
        print(json.dumps(record))
        """
    )


def test_live_refresh_then_portable_verify_preserves_exact_identities(
    tmp_path: Path,
) -> None:
    """Require live and portable modes to agree on scripts and generated inputs."""
    adapter: Path = tmp_path / "adapter.py"
    """Selected a stand-in for one R or native-SOLAR comparison script."""

    adapter.write_text(adapter_source(), encoding="utf-8")
    """Created a deterministic two-mode adapter without installing external tools."""

    fixture_path: Path = tmp_path / "fixture.json"
    """Selected the versioned envelope produced only by live refresh."""

    fixture: dict[str, Any] = refresh("independent_smoke", adapter, fixture_path)
    """Built a fixture from a response proving live external execution."""

    assert (
        fixture_errors(
            fixture,
            check_id="independent_smoke",
            adapter=adapter,
        )
        == []
    )
    fixture_path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    """Persisted the same deterministic form committed by the command-line driver."""

    result: dict[str, Any] = verify("independent_smoke", adapter, fixture_path)
    """Recomputed the candidate without invoking the stand-in external tool."""

    assert result["external_tool_invoked"] is False
    assert result["asterism_recomputed"] is True


def test_portable_verify_refuses_missing_or_stale_evidence(tmp_path: Path) -> None:
    """Fail closed before an absent fixture or changed adapter can count."""
    adapter: Path = tmp_path / "adapter.py"
    """Created the current comparison script identity."""

    adapter.write_text(adapter_source(), encoding="utf-8")
    fixture_path: Path = tmp_path / "fixture.json"
    """Selected a path with no claimed independent result yet."""

    with pytest.raises(FixtureError, match="explicit live refresh"):
        verify("independent_smoke", adapter, fixture_path)

    fixture: dict[str, Any] = refresh("independent_smoke", adapter, fixture_path)
    """Created otherwise valid evidence before changing the calculation source."""

    adapter.write_text(adapter_source() + "\n# changed\n", encoding="utf-8")
    """Simulated a scientific-script change after the independent live result."""

    assert "fixture adapter SHA-256 is stale" in fixture_errors(
        fixture,
        check_id="independent_smoke",
        adapter=adapter,
    )


def test_portable_verify_refuses_malformed_fixture_content(tmp_path: Path) -> None:
    """Reject invalid JSON and empty scientific evidence before adapter execution."""
    adapter: Path = tmp_path / "adapter.py"
    """Created one otherwise conforming strict adapter."""

    adapter.write_text(adapter_source(), encoding="utf-8")
    fixture_path: Path = tmp_path / "fixture.json"
    """Selected a local path for malformed fixture variants."""

    fixture_path.write_text("{not-json", encoding="utf-8")
    """Wrote syntax that cannot represent a versioned evidence envelope."""

    with pytest.raises(FixtureError, match="not valid JSON"):
        verify("independent_smoke", adapter, fixture_path)

    fixture: dict[str, Any] = refresh("independent_smoke", adapter, fixture_path)
    """Built a valid baseline envelope before removing required scientific content."""

    fixture["external_outputs"] = {}
    """Replaced independent outputs with a passing-shaped empty placeholder."""

    fixture["acceptance"] = {}
    """Replaced the prewritten decision rule with an empty placeholder."""

    problems: list[str] = fixture_errors(
        fixture,
        check_id="independent_smoke",
        adapter=adapter,
    )
    """Collected every malformed scientific-evidence diagnostic."""

    assert "external_outputs must be a nonempty object" in problems
    assert "acceptance must be a nonempty object" in problems


def test_portable_verify_refuses_stale_generated_input_hash(tmp_path: Path) -> None:
    """Require the adapter to regenerate the exact inputs named by live refresh."""
    adapter: Path = tmp_path / "adapter.py"
    """Created one stable strict adapter for both fixture modes."""

    adapter.write_text(adapter_source(), encoding="utf-8")
    fixture_path: Path = tmp_path / "fixture.json"
    """Selected the envelope whose generated-input identity will be altered."""

    fixture: dict[str, Any] = refresh("independent_smoke", adapter, fixture_path)
    """Built genuine passing-shaped evidence before simulating stale input."""

    fixture["input_identities"][0]["sha256"] = "b" * 64
    """Changed only the recorded generated input while retaining valid syntax."""

    fixture_path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    """Persisted the structurally valid but stale input identity."""

    with pytest.raises(FixtureError, match="recomputed different input identities"):
        verify("independent_smoke", adapter, fixture_path)


def test_fixture_driver_refuses_false_live_or_nonportable_claims(
    tmp_path: Path,
) -> None:
    """Require refresh to invoke external code and verify to deny doing so."""
    fixture_path: Path = tmp_path / "fixture.json"
    """Selected one envelope path shared by the two refusal variants."""

    no_live_adapter: Path = tmp_path / "no_live.py"
    """Selected an adapter that falsely calls a refresh external-free."""

    no_live_adapter.write_text(
        adapter_source().replace(
            'arguments.reference_mode == "refresh"',
            "False",
        ),
        encoding="utf-8",
    )
    """Forced the adapter to admit that no external tool was invoked."""

    with pytest.raises(FixtureError, match="external_tool_invoked=True"):
        refresh("independent_smoke", no_live_adapter, fixture_path)

    always_live_adapter: Path = tmp_path / "always_live.py"
    """Selected an adapter that claims external execution even during verify."""

    always_live_adapter.write_text(
        adapter_source().replace(
            'arguments.reference_mode == "refresh"',
            "True",
        ),
        encoding="utf-8",
    )
    """Forced both strict modes to claim live external execution."""

    fixture: dict[str, Any] = refresh(
        "independent_smoke", always_live_adapter, fixture_path
    )
    """Built the fixture through the genuinely marked live branch."""

    fixture_path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    """Persisted the valid refresh envelope before portable refusal."""

    with pytest.raises(FixtureError, match="external_tool_invoked=False"):
        verify("independent_smoke", always_live_adapter, fixture_path)
