#!/usr/bin/env python3
"""Count code lines in Python files, excluding docstrings, comments, and blank lines."""

import ast
import sys
from pathlib import Path

WARN_DEFAULT = 300
FAIL_DEFAULT = 400
# The accounting package and its CLI renderers are cohesive subsystems; they get a larger limit
# so that the repository-wide rule stays unchanged everywhere else.
WARN_ACCOUNTING = 600
FAIL_ACCOUNTING = 800
ACCOUNTING_CLI_PREFIXES = ("collect", "export", "metrics", "savings")


def limits_for(relative: Path) -> tuple[int, int]:
    """Return (warn, fail) code-line limits for a path relative to the repository root."""
    parts = relative.parts
    if parts[:3] == ("src", "ssgrep", "accounting"):
        return WARN_ACCOUNTING, FAIL_ACCOUNTING
    if parts[:3] == ("src", "ssgrep", "cli") and relative.name.startswith(ACCOUNTING_CLI_PREFIXES):
        return WARN_ACCOUNTING, FAIL_ACCOUNTING
    return WARN_DEFAULT, FAIL_DEFAULT


def get_docstring_spans(tree: ast.AST) -> set[tuple[int, int]]:
    """
    Extract line number spans for all docstrings.

    Returns a set of (lineno, end_lineno) tuples representing the full extent of each
    docstring. Spans are found by checking if the first statement in a
    Module/ClassDef/FunctionDef/AsyncFunctionDef is an Expr node containing a
    Constant with a string value.
    """
    spans = set()

    def check_docstring(node):
        """Check if the first statement of a node is a docstring."""
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr):
            expr = body[0]
            if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                # This is a docstring; record its full span
                spans.add((expr.lineno, expr.end_lineno or expr.lineno))

    # Check module-level docstring
    check_docstring(tree)

    # Check class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            check_docstring(node)

    return spans


def count_code_lines(filepath: str | Path) -> int | None:
    """
    Count non-comment, non-blank code lines in a Python file, excluding docstrings.

    Excludes:
    - Blank lines
    - Lines whose stripped form starts with '#'
    - Docstrings (module, class, function, async function)

    Returns:
        The count of code lines, or None if the file cannot be parsed.
    """
    filepath = Path(filepath)

    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return None

    # Get all docstring line spans
    docstring_spans = get_docstring_spans(tree)

    # Helper to check if a line number is within any docstring span
    def is_in_docstring(lineno: int) -> bool:
        for start, end in docstring_spans:
            if start <= lineno <= end:
                return True
        return False

    lines = source.splitlines()
    code_lines = 0

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            continue

        # Skip comment lines
        if stripped.startswith("#"):
            continue

        # Skip lines in docstring spans
        if is_in_docstring(lineno):
            continue

        code_lines += 1

    return code_lines


def main():
    """CLI: walk src/**/*.py; warn above each file's warn limit, fail above its fail limit."""
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"

    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    oversized = []
    warnings = []
    for pyfile in sorted(src_dir.rglob("*.py")):
        count = count_code_lines(pyfile)
        if count is None:
            # Could not parse; skip
            continue
        rel_path = pyfile.relative_to(repo_root)
        warn, fail = limits_for(rel_path)
        if count > fail:
            oversized.append((str(rel_path), count, fail))
        elif count > warn:
            warnings.append((str(rel_path), count, warn))

    for filepath, count, warn in warnings:
        print(f"warning: {filepath} has {count} code lines (warn above {warn})", file=sys.stderr)

    if oversized:
        print("Files exceeding their code-line limit:", file=sys.stderr)
        print("", file=sys.stderr)
        for filepath, count, fail in oversized:
            print(f"  {filepath:<50} {count:>4} lines (fail above {fail})", file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
