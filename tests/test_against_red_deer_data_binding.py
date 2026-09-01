"""The red-deer comparison uses an explicit external-data binding."""

from __future__ import annotations

from pathlib import Path

import pytest

from checks.against_red_deer import data_directory


def test_red_deer_data_binding_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not revive a workspace-relative fallback when the binding is absent."""
    monkeypatch.delenv("RED_DEER_DATA", raising=False)

    with pytest.raises(SystemExit, match="Set RED_DEER_DATA"):
        data_directory()


def test_red_deer_data_binding_must_be_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject a misspelt or unavailable external-data path immediately."""
    monkeypatch.setenv("RED_DEER_DATA", str(tmp_path / "missing"))

    with pytest.raises(SystemExit, match="must name an existing directory"):
        data_directory()


def test_red_deer_data_binding_returns_the_explicit_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use the named deposit exactly when its directory is available."""
    deposit: Path = tmp_path / "red-deer"
    """Temporary directory standing in for the bound external deposit."""
    deposit.mkdir()
    monkeypatch.setenv("RED_DEER_DATA", str(deposit))

    assert data_directory() == deposit.resolve()
