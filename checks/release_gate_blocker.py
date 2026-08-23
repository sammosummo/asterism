"""Emit one named scientific-release blocker and fail.

This command makes a missing participant-free or target-design check
executable and auditable without pretending that absence is evidence.  A
blocked pass rule must be replaced by a real command before release; this
program always exits one.
"""

from __future__ import annotations

import argparse
import json
import sys


def parse_arguments() -> argparse.Namespace:
    """Parse the exact analysis, check, and blocker identity."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    """Built a stable machine-readable blocker interface."""

    parser.add_argument("--analysis", required=True)
    parser.add_argument("--check", required=True)
    parser.add_argument("--code", required=True)
    return parser.parse_args()


def main() -> int:
    """Print the configured blocker and return the release-failing status."""
    arguments: argparse.Namespace = parse_arguments()
    """Read the manifest-owned blocker identity."""

    record: dict[str, object] = {
        "analysis_id": arguments.analysis,
        "check_id": arguments.check,
        "blocker_code": arguments.code,
        "participant_free": True,
        "passed": False,
    }
    """Made the absent evidence explicit without reading scientific data."""

    print(json.dumps(record, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
