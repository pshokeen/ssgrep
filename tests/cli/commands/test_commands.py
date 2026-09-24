"""Tests for shared CLI command serialization and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import pytest

from ssgrep.cli import commands
from ssgrep.utilities.types import ContentType, ResultCard


class ExampleEnum(Enum):
    VALUE = "value"


@dataclass(frozen=True)
class JsonContract:
    when: datetime
    kind: ExampleEnum
    path: Path


def test_resolve_project_dir_expands_user_and_makes_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = commands.resolve_project_dir("~/project/../project")

    assert resolved == tmp_path / "project" / ".." / "project"
    assert resolved.is_absolute()


def test_to_jsonable_recursively_reduces_contract_values() -> None:
    when = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    contract = JsonContract(when=when, kind=ExampleEnum.VALUE, path=Path("somewhere"))

    assert commands.to_jsonable(contract) == {
        "when": when.isoformat(),
        "kind": "value",
        "path": "somewhere",
    }


def test_to_jsonable_handles_nested_real_contract(sample_card: ResultCard) -> None:
    payload = commands.to_jsonable(sample_card)

    assert payload["timestamp"] == "2025-01-02T03:04:00+00:00"
    assert payload["content_type"] == ContentType.RESPONSE.value
    assert payload["files_touched"] == ["tests/test_example.py"]


@pytest.mark.parametrize("invalid", [object(), JsonContract])
def test_to_jsonable_rejects_non_instances(invalid: object) -> None:
    with pytest.raises(TypeError, match="expects a dataclass instance"):
        commands.to_jsonable(invalid)


def test_to_jsonable_rejects_non_object_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = JsonContract(datetime.now(UTC), ExampleEnum.VALUE, Path("x"))
    monkeypatch.setattr(commands.json, "loads", lambda _serialized: [])

    with pytest.raises(TypeError, match="did not serialize to a JSON object"):
        commands.to_jsonable(contract)


def test_json_default_fallback_stringifies_unknown_values() -> None:
    value = object()

    assert commands._json_default(value) == str(value)
