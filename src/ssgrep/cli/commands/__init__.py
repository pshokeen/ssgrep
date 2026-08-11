"""CLI commands for ssgrep."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ssgrep import paths


def resolve_project_dir(project_dir: str) -> Path:
    """Resolve project_dir to an absolute path.

    Delegates to ssgrep.paths.resolve_live so that the path used to LOCATE
    the index directory comes from the same module as the canonicalization
    used to MATCH scope. In particular the tilde is expanded here, before
    resolve(): a quoted ``--project-dir "~/code/x"`` used to become
    ``<cwd>/~/code/x`` and match nothing.

    Args:
        project_dir: A path string (relative, absolute, or home-relative).

    Returns:
        Resolved absolute Path.

    Raises:
        RuntimeError: If the path doesn't resolve to an absolute path.
    """
    path = paths.resolve_live(project_dir)
    if not path.is_absolute():
        # Should not happen with resolve(), but be defensive
        msg = f"Could not resolve {project_dir} to an absolute path"
        raise RuntimeError(msg)
    return path


def to_jsonable(data: Any) -> Any:
    """Convert a frozen contract dataclass into JSON-serializable plain data.

    usecli's JSON mode serializes the value returned from ``handle()`` with a
    strict ``json.dumps`` (no default hook), so datetimes and enums must be
    reduced to plain types first. Datetimes become ISO 8601 strings and enums
    become their values.
    """
    if not is_dataclass(data) or isinstance(data, type):
        raise TypeError(f"to_jsonable expects a dataclass instance, got {type(data)!r}")
    return json.loads(json.dumps(asdict(data), default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)
