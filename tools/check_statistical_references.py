"""Validate the canonical statistical methods document and bibliography."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

ROOT: Path = Path(__file__).resolve().parents[1]
"""Located the repository root containing the canonical documentation."""

CITATION_PATTERN: re.Pattern[str] = re.compile(
    r"\[(?P<label>[^\]]+?)\]\((?:\.\./)*references\.bib#(?P<key>[A-Za-z][A-Za-z0-9]*)\)",
    re.DOTALL,
)
"""Matched author-year links carrying a canonical BibTeX key.

The leading `../` is optional because the per-model pages sit one
directory below the bibliography they cite.
"""

ENTRY_PATTERN: re.Pattern[str] = re.compile(
    r"^@\w+\{(?P<key>[A-Za-z][A-Za-z0-9]*),\n(?P<body>.*?)^\}\s*$",
    re.DOTALL | re.MULTILINE,
)
"""Matched complete top-level BibTeX entries and their bodies."""

LOCAL_LINK_PATTERN: re.Pattern[str] = re.compile(
    r"\]\((?P<target>(?!https?://|mailto:|#)[^\s)#]+)(?:#[^)]*)?\)"
)
"""Matched local Markdown targets while excluding URLs and bare fragments."""

YEAR_PATTERN: re.Pattern[str] = re.compile(
    r"(?m)^\s*year\s*=\s*\{(?P<year>\d{4})\},?\s*$"
)
"""Read the four-digit year used by an author-year citation."""

REQUIRED_FIELDS: tuple[str, ...] = ("author", "title", "year", "doi", "url")
"""Required stable metadata on every canonical reference."""

REQUIRED_HEADINGS: tuple[str, ...] = (
    "## What a release requires",
    "## Numerical implementation",
)
"""Named the method sections that must remain in the shared document.

Each supported model's own section left this list when the equations, the
evidence and the interface for each were gathered onto a page of its own, and
the unsupported models' sections left with them, for outside-support.md.
That requirement did not go away, it moved: `tests/test_model_pages.py`
requires every analysis in the manifest to have a page carrying "## The
model", and takes the list from the manifest rather than repeating it here,
so a new model cannot ship without one.
"""


def validate_references(
    methods_path: Path,
    bibliography_path: Path,
    manifest_path: Path,
) -> list[str]:
    """Return integrity failures across methods, citations, and release scope.

    Args:
        methods_path: Canonical Markdown methods specification.
        bibliography_path: Canonical BibTeX bibliography.
        manifest_path: Release manifest defining supported analyses.

    Returns:
        Stable human-readable failures; an empty list means the contract passed.
    """
    pages: list[Path] = [
        page
        for page in (
            *sorted((methods_path.parent / "models").glob("*.md")),
            methods_path.parent / "outside-support.md",
        )
        if page.is_file()
    ]
    """Found the per-model pages, which now carry most of the equations.

    The methods document was one file until the equations, the evidence and the
    interface for each model were gathered onto a page of its own. Absent
    companions are skipped, so a caller may point this at a lone document --
    which is what the tests that feed it a deliberately broken one do. Reading both
    keeps this check exactly as strong as it was: a citation or a supported
    quantity satisfies it wherever it is written, and nowhere else.
    """

    methods_text: str = "\n".join(
        [methods_path.read_text(encoding="utf-8")]
        + [page.read_text(encoding="utf-8") for page in pages]
    )
    """Read the methods specification whose equations and citations are checked."""

    bibliography_text: str = bibliography_path.read_text(encoding="utf-8")
    """Read the sole bibliography against which citation keys resolve."""

    manifest: dict[str, Any] = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    """Parsed the public release scope used to check methods coverage."""

    problems: list[str] = []
    """Collected every problem so one command can guide a complete repair."""

    citations: list[tuple[str, str]] = [
        (match.group("label"), match.group("key"))
        for match in CITATION_PATTERN.finditer(methods_text)
    ]
    """Collected each rendered author-year label with its BibTeX key."""

    entry_matches: list[re.Match[str]] = list(ENTRY_PATTERN.finditer(bibliography_text))
    """Parsed top-level entries without treating nested braces as entry endings."""

    entry_bodies: dict[str, str] = {}
    """Indexed each unique canonical key to the fields in its BibTeX entry."""

    entry_years: dict[str, str] = {}
    """Indexed citation keys to the years their author-year labels must carry."""

    for entry_match in entry_matches:
        key: str = entry_match.group("key")
        """Read the candidate canonical key from one complete entry."""

        body: str = entry_match.group("body")
        """Selected the field text belonging to the candidate entry."""

        if key in entry_bodies:
            problems.append(f"duplicate bibliography key: {key}")
            continue
        entry_bodies[key] = body
        """Registered the unique entry body for citation resolution."""

        year_match: re.Match[str] | None = YEAR_PATTERN.search(body)
        """Located the publication year needed by the rendered citation label."""

        if year_match is not None:
            entry_years[key] = year_match.group("year")
            """Registered the publication year for author-year validation."""
    """Built the canonical key and year indexes while detecting duplicates."""

    for key, body in entry_bodies.items():
        for field in REQUIRED_FIELDS:
            if re.search(rf"(?m)^\s*{field}\s*=\s*\{{.+", body) is None:
                problems.append(f"bibliography entry {key} is missing {field}")
        """Required complete, source-resolving metadata on the canonical entry."""

        if re.search(r"(?<!\\)%", body) is not None:
            problems.append(f"bibliography entry {key} contains an unescaped percent")
    """Validated metadata and TeX-safe percent signs in every BibTeX entry."""

    citation_keys: set[str] = {key for _, key in citations}
    """Collected the bibliography keys used by the methods specification."""

    bibliography_keys: set[str] = set(entry_bodies)
    """Collected every key defined by the canonical bibliography."""

    for key in sorted(citation_keys - bibliography_keys):
        problems.append(f"citation key does not resolve: {key}")
    """Rejected method citations absent from the canonical bibliography."""

    for key in sorted(bibliography_keys - citation_keys):
        problems.append(f"uncited canonical bibliography entry: {key}")
    """Prevented unused entries from accumulating in the canonical bibliography."""

    for label, key in citations:
        year: str | None = entry_years.get(key)
        """Read the expected publication year for this rendered citation."""

        if year is not None and year not in label:
            compact_label: str = " ".join(label.split())
            """Collapsed wrapped Markdown labels into a readable diagnostic."""

            problems.append(
                f"citation {key} has label {compact_label!r} without year {year}"
            )
    """Required rendered labels to agree with their BibTeX publication years."""

    if methods_text.count("$$") % 2 != 0:
        problems.append("unbalanced display-math delimiters in statistical methods")

    if methods_text.count(r"\begin{aligned}") != methods_text.count(r"\end{aligned}"):
        problems.append("unbalanced aligned equation environment")

    if methods_text.count(r"\begin{cases}") != methods_text.count(r"\end{cases}"):
        problems.append("unbalanced cases equation environment")

    if methods_text.count(r"\begin{pmatrix}") != methods_text.count(r"\end{pmatrix}"):
        problems.append("unbalanced matrix equation environment")
    """Checked the display and structured equation delimiters used in the document."""

    control_characters: list[int] = [
        index
        for index, character in enumerate(methods_text)
        if ord(character) < 32 and character not in {"\n", "\t"}
    ]
    """Located control characters that can silently corrupt LaTeX commands."""

    if control_characters:
        problems.append(
            "statistical methods contains control characters at offsets "
            + ", ".join(str(index) for index in control_characters)
        )

    for legacy_field in ("lower_at_bound", "upper_at_bound"):
        if legacy_field in methods_text:
            problems.append(f"statistical methods uses retired field {legacy_field}")
    """Kept the canonical methods document on the shared public interval schema."""

    for source in [methods_path, *pages]:
        text: str = source.read_text(encoding="utf-8")
        """Read one document, so its links resolve against its own directory."""

        for match in sorted(
            {m.group("target") for m in LOCAL_LINK_PATTERN.finditer(text)}
        ):
            target_path: Path = source.parent / match
            """Resolved the target relative to the page that links to it."""

            if not target_path.exists():
                problems.append(
                    f"local Markdown target does not exist: {match} "
                    f"(linked from {source.name})"
                )
    """Rejected stale links, each judged from where it is actually written."""

    for heading in REQUIRED_HEADINGS:
        if heading not in methods_text:
            problems.append(f"statistical methods is missing heading: {heading}")
    """Required all supported methodological sections and numerical provenance."""

    analyses: list[dict[str, Any]] = cast(
        list[dict[str, Any]], manifest.get("analyses", [])
    )
    """Selected analysis records from the authoritative release manifest."""

    supported_ids: list[str] = [
        cast(str, analysis["id"])
        for analysis in analyses
        if analysis.get("support") == "supported_in_0_1"
    ]
    """Collected every analysis ID whose method must be documented."""

    for analysis_id in supported_ids:
        if f"`{analysis_id}`" not in methods_text:
            problems.append(f"supported analysis is absent from methods: {analysis_id}")
    """Tied the documented support table directly to the release manifest."""

    for analysis in analyses:
        if analysis.get("support") != "supported_in_0_1":
            continue

        analysis_id: str = cast(str, analysis["id"])
        """Read the supported analysis named in a missing-quantity diagnostic."""

        supported_quantities: list[str] = cast(
            list[str], analysis.get("supported_quantities", [])
        )
        """Selected the exact manifest vocabulary requiring public-field mappings."""

        for supported_quantity in supported_quantities:
            if f"`{supported_quantity}`" not in methods_text:
                problems.append(
                    f"supported quantity is absent from methods: "
                    f"{analysis_id}.{supported_quantity}"
                )
    """Required every manifest quantity to appear as an exact documented token."""

    return problems


def main() -> int:
    """Run the statistical documentation integrity check.

    Returns:
        Zero when citations, equations, and release coverage pass; one otherwise.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Validate statistical methods citations and equations."
    )
    """Created the public command-line interface used by tests and release checks."""

    parser.add_argument(
        "--methods",
        type=Path,
        default=ROOT / "docs" / "statistical-methods.md",
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=ROOT / "docs" / "references.bib",
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "release.toml")

    arguments: argparse.Namespace = parser.parse_args()
    """Parsed optional paths while retaining repository defaults."""

    problems: list[str] = validate_references(
        arguments.methods,
        arguments.bibliography,
        arguments.manifest,
    )
    """Validated the complete statistical documentation contract."""

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    print("statistical methods references: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
