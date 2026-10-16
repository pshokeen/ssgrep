"""CLI commands for ssgrep."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

_CORE_ROOTS = frozenset(
    {"index", "init", "mcp", "note", "prune", "rules", "search", "show", "status", "version"}
)
_DISCOVERY_OPTIONS = frozenset({"--help", "-h", "--interactive", "-i"})

SAVINGS_EXPORT_GROUP_HELP = (
    "Organization export: preview, check, enable, disable, send data, or run its service"
)


def accounting_command_visible() -> bool:
    """Avoid building unrelated optional parsers for a plain legacy invocation.

    Root help, completion, interactive mode, accounting commands, unknown roots and
    ambiguous global-option invocations retain full registration. New core commands
    conservatively take that path too; this allowlist is not a dispatch authority.
    """
    root_index = 1
    # This documented flag has no value; do not guess how any other option parses.
    while root_index < len(sys.argv) and sys.argv[root_index] == "--json":
        root_index += 1
    if (
        root_index >= len(sys.argv)
        or os.path.basename(sys.argv[0]) != "ssgrep"
        or sys.argv[root_index] not in _CORE_ROOTS
        or os.environ.get("_SSGREP_COMPLETE")
    ):
        return True
    return any(argument in _DISCOVERY_OPTIONS for argument in sys.argv)


def resolve_project_dir(project_dir: str) -> Path:
    """Resolve the project path recorded in an authored note."""
    return Path(project_dir).expanduser().absolute()


def to_jsonable(data: Any) -> dict[str, Any]:
    """Convert a frozen contract dataclass into JSON-serializable plain data.

    usecli's JSON mode serializes the value returned from ``handle()`` with a
    strict ``json.dumps`` (no default hook), so datetimes and enums must be
    reduced to plain types first. Datetimes become ISO 8601 strings and enums
    become their values.
    """
    if not is_dataclass(data) or isinstance(data, type):
        raise TypeError(f"to_jsonable expects a dataclass instance, got {type(data)!r}")
    result = json.loads(json.dumps(asdict(data), default=_json_default))
    if not isinstance(result, dict):
        raise TypeError("dataclass did not serialize to a JSON object")
    return result


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def savings_export_group(main_app: Any, *, help_text: str = SAVINGS_EXPORT_GROUP_HELP):
    """Return the one `savings export` Typer group, creating it through usecli's group registry."""
    import typer
    from usecli.cli.core.base_command import NestedCommandRegistry

    savings = NestedCommandRegistry().get_or_create_group(main_app, "savings")
    export = getattr(savings, "_ssgrep_export_group", None)
    if export is None:
        export = typer.Typer(
            help=help_text,
            no_args_is_help=True,
            add_completion=False,
        )
        savings.add_typer(export, name="export")
        setattr(savings, "_ssgrep_export_group", export)  # noqa: B010
    return export
