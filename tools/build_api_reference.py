"""Generate `docs/api-reference.md` from the package's own docstrings.

Run it with `uv run python tools/build_api_reference.py`. A test requires the
file on disk to match what this produces, so the reference cannot drift from
the code the way hand-written signatures do.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import asterism

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root the reference is written into."""

OUTPUT: Path = ROOT / "docs" / "api-reference.md"
"""Named the one generated reference."""

HEADER: str = """# API reference

Generated from the package's own docstrings by
`tools/build_api_reference.py`. Do not edit it by hand: a test requires this
file to match what that script produces.

Support status for each object is in [api-support.md](api-support.md). Runnable
programs using them are in [`examples/`](../examples/).

"""
"""Fixed the preamble, which says where the file comes from."""


def indented(text: str, spaces: int = 0) -> str:
    """Re-indent a docstring so Markdown does not read it as code.

    Args:
        text: A raw docstring.
        spaces: How far to indent every line.

    Returns:
        The cleaned text.
    """
    cleaned: str = inspect.cleandoc(text or "")
    """Removed the uniform leading whitespace Python docstrings carry."""

    pad: str = " " * spaces
    """Built the indent once."""

    return "\n".join(pad + line if line else "" for line in cleaned.splitlines())


def signature_of(obj: Any) -> str:
    """Render one callable's signature, or an empty string if it has none.

    Args:
        obj: A function, method or class.

    Returns:
        The signature in parentheses, or "".
    """
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def render() -> str:
    """Build the whole reference from the public package surface.

    Returns:
        The complete Markdown document.
    """
    parts: list[str] = [HEADER]
    """Collected the document a section at a time."""

    names: list[str] = sorted(asterism.__all__)
    """Took the public surface from the package, not from a list kept here."""

    classes: list[str] = [
        name for name in names if inspect.isclass(getattr(asterism, name, None))
    ]
    """Separated classes, which carry methods, from plain functions."""

    others: list[str] = [name for name in names if name not in classes]
    """Kept everything else in one alphabetical run."""

    parts.append("## Models\n")
    for name in classes:
        cls: Any = getattr(asterism, name)
        """Read one public class."""

        parts.append(f"### `{name}`\n")
        parts.append(indented(cls.__doc__ or "No documentation.") + "\n")
        methods: list[str] = sorted(
            attribute
            for attribute in dir(cls)
            if not attribute.startswith("_") and callable(getattr(cls, attribute, None))
        )
        """Listed the public methods this class offers."""

        for method_name in methods:
            method: Any = getattr(cls, method_name)
            """Read one public method."""

            parts.append(f"#### `{name}.{method_name}{signature_of(method)}`\n")
            parts.append(indented(method.__doc__ or "No documentation.") + "\n")

    parts.append("## Functions and values\n")
    for name in others:
        obj: Any = getattr(asterism, name, None)
        """Read one public function or value."""

        if callable(obj):
            parts.append(f"### `{name}{signature_of(obj)}`\n")
            parts.append(indented(obj.__doc__ or "No documentation.") + "\n")
        else:
            parts.append(f"### `{name}`\n")
            parts.append(f"`{type(obj).__name__}`. Its value belongs to the build.\n")

    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    """Write the reference, or check the one on disk matches.

    Returns:
        Zero when the file is correct or was written, one when it is stale.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )
    """Built the one-flag command line."""

    parser.add_argument("--check", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()
    """Read whether this run writes or only checks."""

    rendered: str = render()
    """Built the document once."""

    if arguments.check:
        current: str = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        """Read whatever is on disk now."""

        if current != rendered:
            print("docs/api-reference.md is stale; run tools/build_api_reference.py")
            return 1
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
