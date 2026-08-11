#!/usr/bin/env python3
"""PostToolUse hook to warn about oversized Python source files."""

import json
import os
import sys
from pathlib import Path

# Get the project root from environment or derive from script location
project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
if project_dir:
    repo_root = Path(project_dir)
else:
    # Fallback: script is at .claude/hooks/check-file-size.py, so root is 3 levels up
    repo_root = Path(__file__).parent.parent.parent

scripts_dir = repo_root / "scripts"
if scripts_dir not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import the counter function
try:
    from check_file_size import count_code_lines
except ImportError:
    # If we can't import, silently exit (don't crash the hook)
    sys.exit(0)


def main():
    """Read hook JSON from stdin and warn if edited file is oversized."""
    try:
        hook_payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        # Malformed JSON; silently ignore
        sys.exit(0)

    # Extract the edited file path from the PostToolUse payload
    # Payload has 'tool_name' and 'tool_input' at top level
    try:
        tool_input = hook_payload.get("tool_input", {})

        # Both Write and Edit tools use 'file_path' parameter
        file_path = tool_input.get("file_path", "")
    except (KeyError, AttributeError, TypeError):
        # Could not extract path; silently ignore
        sys.exit(0)

    if not file_path:
        sys.exit(0)

    filepath = Path(file_path)

    # Only apply to *.py under src/
    try:
        rel_path = filepath.relative_to(repo_root)
    except ValueError:
        # Path is outside the repo
        sys.exit(0)

    # Check if it's under src/ and is a Python file
    if not (filepath.suffix == ".py" and rel_path.parts[0] == "src"):
        sys.exit(0)

    # Count code lines (will return None if file can't be parsed)
    count = count_code_lines(filepath)
    if count is None:
        # Could not parse; silently ignore
        sys.exit(0)

    # Warn based on thresholds using JSON output format
    if count > 400:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"⚠️ {rel_path} is {count} lines (exceeds ceiling of 400)"
            }
        }
        print(json.dumps(output))
    elif count > 300:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"⚠️ {rel_path} is {count} lines"
            }
        }
        print(json.dumps(output))

    # Always exit with success (never block)
    sys.exit(0)


if __name__ == "__main__":
    main()
