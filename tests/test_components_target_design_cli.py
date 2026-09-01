"""Command-line contracts for the several-component target-design check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from checks import components_target_design as campaign


def test_no_write_keeps_the_checkout_clean(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Let the release runner retain stdout without creating dated evidence."""
    monkeypatch.setattr(
        campaign,
        "target_matrices",
        lambda: (np.eye(2), np.eye(2), 2),
    )
    monkeypatch.setattr(campaign, "REPLICATES", 2)
    monkeypatch.setattr(
        campaign,
        "one",
        lambda *_arguments: {
            "estimate": campaign.TRUE_HOUSEHOLD,
            "covered": True,
        },
    )
    writes: list[Path] = []
    """Recorded every attempted evidence write without touching the checkout."""

    def record_write(path: Path, _content: str, **_options: object) -> int:
        writes.append(path)
        return 0

    monkeypatch.setattr(Path, "write_text", record_write)

    assert campaign.main([]) == 0
    ordinary_output: str = capsys.readouterr().out
    """Captured the historical standalone report including its saved path."""

    assert len(writes) == 1
    assert "written to" in ordinary_output
    writes.clear()

    assert campaign.main(["--no-write"]) == 0
    assert writes == []
    assert "written to" not in capsys.readouterr().out
