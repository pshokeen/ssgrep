#!/usr/bin/env python3
"""Count code lines in Python files, excluding docstrings, comments, and blank lines."""

import ast
import sys
from pathlib import Path


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
    """CLI: walk src/**/*.py and fail if any file exceeds 400 code lines."""
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"

    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    oversized = []
    for pyfile in sorted(src_dir.rglob("*.py")):
        count = count_code_lines(pyfile)
        if count is None:
            # Could not parse; skip
            continue
        if count > 400:
            rel_path = pyfile.relative_to(repo_root)
            oversized.append((str(rel_path), count))

    if oversized:
        # Print table on failure
        print("Files exceeding 400 code lines:", file=sys.stderr)
        print("", file=sys.stderr)
        for filepath, count in oversized:
            print(f"  {filepath:<50} {count:>4} lines", file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
