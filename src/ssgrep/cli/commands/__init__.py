"""CLI commands for ssgrep."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


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
