"""Enforce Asterism's project-specific Python style rules."""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

EXCEPTION_PATTERN: re.Pattern[str] = re.compile(
    r"# asterism-style: allow ([a-z-]+) -- (\S.*)$"
)
"""Matched the only accepted project-style exception form."""

EXCEPTION_RULES: frozenset[str] = frozenset(
    {
        "missing-following-doc",
        "private-helper",
        "single-use-helper",
        "unannotated-local",
    }
)
"""Listed the four narrow exception names defined by the style guide."""

MAINTAINED_ROOTS: tuple[str, ...] = ("python", "tests", "checks", "campaigns", "tools")
"""Named source areas covered by Asterism's Python style contract."""

EXCLUDED_PARTS: frozenset[str] = frozenset(
    {".claude", ".venv", "__pycache__", "results", "target"}
)
"""Excluded generated results, environments, caches and third-party worktrees."""


@dataclass(frozen=True)
class Problem:
    """Represent one exact project-style violation."""

    path: Path
    """Held the current source path used in line-addressed diagnostics."""
    line: int
    """Held the current one-based source line for human review."""
    rule: str
    """Named the violated project-local rule."""
    message: str
    """Explained the smallest repair required by the rule."""

    def diagnostic(self) -> str:
        """Return the familiar line-addressed command diagnostic."""
        return f"{self.path}:{self.line}: {self.rule}: {self.message}"


def make_problem(
    *,
    path: Path,
    line: int,
    rule: str,
    message: str,
) -> Problem:
    """Create one line-addressed project-style violation.

    Args:
        path: Python source carrying the violation.
        line: Current one-based diagnostic line.
        rule: Project-local rule identifier.
        message: Human-readable repair instruction.

    Returns:
        Structured violation suitable for complete diagnostics.
    """
    return Problem(path, line, rule, message)


def allows_rule(comments: dict[int, str], node: ast.AST, rule: str) -> bool:
    """Return whether an adjacent, reasoned exception permits one rule.

    Args:
        comments: Comment tokens indexed by their one-based source line.
        node: Smallest syntax construct affected by the exception.
        rule: Stable rule name to match.

    Returns:
        True only for the named rule on the same or immediately preceding line.
    """
    line_numbers: tuple[int, int] = (max(node.lineno - 1, 1), node.lineno)
    """Selected the immediately preceding and starting source lines."""

    for line_number in line_numbers:
        match: re.Match[str] | None = EXCEPTION_PATTERN.search(
            comments.get(line_number, "")
        )
        """Looked for the exact exception form on an adjacent line."""

        if match is not None and match.group(1) == rule:
            return True
    return False


def containing_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    """Find the lexical module, class or callable owning one syntax node.

    Args:
        node: Syntax node whose scope is required.
        parents: Direct parent lookup for the parsed module.

    Returns:
        The nearest lexical scope node.
    """
    current: ast.AST = node
    """Started at the syntax construct whose owner is required."""

    while current in parents:
        current = parents[current]
        """Moved one syntax level towards the module root."""

        if isinstance(
            current,
            ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ):
            return current
    return current


def find_problems(path: Path) -> list[Problem]:
    """Return structured project-style violations in one Python source file.

    Args:
        path: Source file to inspect.

    Returns:
        Structured violations carrying complete line-addressed diagnostics.
    """
    source: str = path.read_text(encoding="utf-8")
    """Read the maintained source exactly as committed."""

    comments: dict[int, str] = {
        token.start[0]: token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    }
    """Collected real comment tokens so strings cannot act as directives."""

    try:
        tree: ast.Module = ast.parse(source, filename=str(path))
        """Parsed the source through Python's public syntax tree."""
    except SyntaxError as error:
        line: int = error.lineno or 1
        """Located the syntax error when the parser supplied no line."""

        return [
            make_problem(
                path=path,
                line=line,
                rule="syntax-error",
                message=error.msg,
            )
        ]
    problems: list[Problem] = []
    """Collected all style failures so one run can guide a complete repair."""

    parents: dict[ast.AST, ast.AST] = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    """Indexed lexical parents for per-scope first-assignment checks."""

    for line_number, comment in comments.items():
        if "asterism-style:" not in comment:
            continue
        match: re.Match[str] | None = EXCEPTION_PATTERN.search(comment)
        """Parsed each declared exception instead of ignoring malformed markers."""

        if match is None or match.group(1) not in EXCEPTION_RULES:
            problems.append(
                make_problem(
                    path=path,
                    line=line_number,
                    rule="invalid-style-exception",
                    message="use a named rule and a specific reason",
                )
            )
    """Rejected broad, unknown and unexplained exception directives."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        parameters: list[ast.arg] = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        """Collected parameters whose annotations live on ordinary argument nodes."""

        if node.args.vararg is not None:
            parameters.append(node.args.vararg)
        if node.args.kwarg is not None:
            parameters.append(node.args.kwarg)
        """Included variadic parameters when the callable declares them."""

        for parameter in parameters:
            if parameter.arg in {"self", "cls"}:
                continue
            if parameter.annotation is None:
                problems.append(
                    make_problem(
                        path=path,
                        line=parameter.lineno,
                        rule="missing-parameter-annotation",
                        message=f"annotate {parameter.arg!r}",
                    )
                )
        if node.returns is None:
            problems.append(
                make_problem(
                    path=path,
                    line=node.lineno,
                    rule="missing-return-annotation",
                    message=f"annotate {node.name!r}",
                )
            )
        private_name: bool = node.name.startswith("_") and not (
            node.name.startswith("__") and node.name.endswith("__")
        )
        """Distinguished private helpers from Python protocol methods."""

        if private_name and not allows_rule(comments, node, "private-helper"):
            problems.append(
                make_problem(
                    path=path,
                    line=node.lineno,
                    rule="private-helper",
                    message=f"explain why {node.name!r} must be private",
                )
            )
    """Applied the complete callable-annotation rule."""

    for container in ast.walk(tree):
        if not isinstance(container, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        nested_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
            statement
            for statement in container.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        """Collected helpers defined directly inside one callable."""

        for nested in nested_functions:
            calls: int = sum(
                1
                for candidate in ast.walk(container)
                if isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Name)
                and candidate.func.id == nested.name
            )
            """Counted direct calls to the local helper within its owner."""

            if calls == 1 and not allows_rule(comments, nested, "single-use-helper"):
                problems.append(
                    make_problem(
                        path=path,
                        line=nested.lineno,
                        rule="single-use-helper",
                        message=(
                            f"keep {nested.name!r} inside its only caller or explain "
                            "the boundary"
                        ),
                    )
                )
    """Applied the objectively detectable nested single-use-helper rule."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.type_comment is not None:
            continue
        name: str = node.targets[0].id
        """Identified a first-assignment form which can carry an annotation."""

        if name == "_":
            continue
        scope: ast.AST = containing_scope(node, parents)
        """Located the lexical scope in which the name is assigned."""

        annotated_before: bool = any(
            isinstance(candidate, ast.AnnAssign)
            and isinstance(candidate.target, ast.Name)
            and candidate.target.id == name
            and candidate.lineno < node.lineno
            and containing_scope(candidate, parents) is scope
            for candidate in ast.walk(scope)
        )
        """Checked for the required first annotation earlier in this scope."""

        if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
            parameter_names: set[str] = {
                parameter.arg
                for parameter in (
                    *scope.args.posonlyargs,
                    *scope.args.args,
                    *scope.args.kwonlyargs,
                )
            }
            """Collected already annotated or exempt callable parameters."""

            annotated_before = annotated_before or name in parameter_names
            """Treated parameter rebinding as rebinding an existing name."""

        if annotated_before:
            continue
        if allows_rule(comments, node, "unannotated-local"):
            continue
        problems.append(
            make_problem(
                path=path,
                line=node.lineno,
                rule="unannotated-local",
                message=f"annotate the first assignment to {name!r}",
            )
        )
    """Applied the comprehensive first-assignment annotation rule."""

    for parent in ast.walk(tree):
        for field_name in ("body", "orelse", "finalbody"):
            candidate_suite: object = getattr(parent, field_name, [])
            """Read a possible statement-suite field without assuming its shape."""

            if not isinstance(candidate_suite, list) or not all(
                isinstance(statement, ast.stmt) for statement in candidate_suite
            ):
                continue
            statements: list[ast.stmt] = candidate_suite
            """Read one ordered statement suite from the syntax tree."""

            for index, statement in enumerate(statements):
                if not isinstance(
                    statement, ast.Assign | ast.AnnAssign | ast.AugAssign
                ):
                    continue
                following: ast.stmt | None = (
                    statements[index + 1] if index + 1 < len(statements) else None
                )
                """Located the statement immediately after the transformation."""

                documented: bool = (
                    isinstance(following, ast.Expr)
                    and isinstance(following.value, ast.Constant)
                    and isinstance(following.value.value, str)
                )
                """Recognised an intentional standalone documentation string."""

                if not documented and not allows_rule(
                    comments, statement, "missing-following-doc"
                ):
                    problems.append(
                        make_problem(
                            path=path,
                            line=statement.lineno,
                            rule="missing-following-doc",
                            message=(
                                "add a standalone string immediately after this assignment"
                            ),
                        )
                    )
    """Applied the following-documentation rule to ordered statement suites."""

    return problems


def check_path(path: Path) -> list[str]:
    """Return stable line-addressed diagnostics for one explicit source path.

    Args:
        path: Source file to inspect directly.

    Returns:
        Human-readable diagnostics for every current violation.
    """
    return [problem.diagnostic() for problem in find_problems(path)]


def maintained_paths(root: Path) -> list[Path]:
    """Discover maintained Python files under the guide's named source areas.

    Args:
        root: Repository root to search.

    Returns:
        Sorted Python paths excluding generated and external material.
    """
    paths: list[Path] = []
    """Accumulated maintained Python sources from each named area."""

    for relative in MAINTAINED_ROOTS:
        directory: Path = root / relative
        """Resolved one maintained source root."""

        if not directory.is_dir():
            continue
        paths.extend(
            path
            for path in directory.rglob("*.py")
            if not EXCLUDED_PARTS.intersection(path.parts)
        )
    """Discovered Python sources while excluding generated and external trees."""

    return sorted(paths)


def main() -> int:
    """Check named files or the whole maintained tree with no accepted debt."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Check Asterism's project-specific Python style."
    )
    """Defined the command interface used by tests and quality automation."""

    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="compatibility spelling; every invocation now enforces zero debt",
    )
    parser.add_argument("paths", nargs="*", type=Path)
    arguments: argparse.Namespace = parser.parse_args()
    """Parsed the maintained source files selected for inspection."""

    paths: list[Path] = arguments.paths or maintained_paths(arguments.root)
    """Used explicit fixtures when supplied and repository discovery otherwise."""

    if not paths:
        print(
            "no maintained Python found under "
            f"{arguments.root}: this gate checked nothing, which is not the "
            "same as passing",
            file=sys.stderr,
        )
        return 1
    """Refused an empty run.

    `EXCLUDED_PARTS` holds `.claude`, and a git worktree made by this project's
    own tooling lives under `.claude/worktrees/`, so every path inside one is
    excluded and discovery returns nothing. The checker then printed nothing and
    exited nought, which reads exactly like success. It is the fault this
    package keeps finding in its own models -- a calculation that could not be
    made, read as one that was -- and it belongs to the gates too.
    """

    problems: list[Problem] = []
    """Accumulated failures across every selected file."""

    for path in paths:
        problems.extend(find_problems(path))
    """Checked every path without stopping after the first file."""

    diagnostics: list[str] = [problem.diagnostic() for problem in problems]
    """Exposed every violation because maintained debt is no longer accepted."""

    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    """Reported all current violations in one repair cycle."""

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
