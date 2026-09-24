"""Unit tests for opt-in external transcript roots."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from ssgrep.sessions import discovery_roots


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_native_schema_sniff_accepts_a_type_after_noise(tmp_path: Path):
    shard = _write_lines(
        tmp_path / "mixed.jsonl",
        ["", "truncated {", json.dumps(["not", "a", "dict"]), json.dumps({"type": "user"})],
    )
    assert discovery_roots._looks_like_native_shard(shard)


def test_native_schema_sniff_rejects_nontranscripts_and_stops_at_five(tmp_path: Path):
    invalid = _write_lines(
        tmp_path / "data.jsonl",
        [json.dumps({"value": i}) for i in range(5)] + [json.dumps({"type": "too-late"})],
    )
    assert not discovery_roots._looks_like_native_shard(invalid)
    assert not discovery_roots._looks_like_native_shard(
        _write_lines(tmp_path / "blank.jsonl", [""])
    )


def test_native_schema_sniff_fails_open_on_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    original_open = builtins.open

    def fail_for_target(path, *args, **kwargs):
        if Path(path) == tmp_path / "unreadable.jsonl":
            raise OSError("denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_for_target)
    assert discovery_roots._looks_like_native_shard(tmp_path / "unreadable.jsonl")


def test_discover_native_recurses_sorts_filters_and_hashes_paths(tmp_path: Path, capsys):
    root = tmp_path / "external"
    valid_b = _write_lines(root / "z" / "same.jsonl", [json.dumps({"type": "assistant"})])
    valid_a = _write_lines(root / "a" / "same.jsonl", [json.dumps({"type": "user"})])
    invalid = _write_lines(root / "dump.jsonl", [json.dumps({"rows": []})])

    sessions = discovery_roots._discover_native(root)
    assert [s.path for s in sessions] == [valid_a, valid_b]
    assert all(s.is_main and s.parent_session_id is None for s in sessions)
    expected = [
        f"same~{hashlib.sha1(str(path.resolve()).encode('utf-8')).hexdigest()[:8]}"
        for path in (valid_a, valid_b)
    ]
    assert [s.session_id for s in sessions] == expected
    error = capsys.readouterr().err
    assert str(invalid) in error
    assert "not a transcript file" in error


def test_parse_roots_ignores_empty_entries_and_expands_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    pairs = discovery_roots.parse_roots(" : ~/notes :: native= ./exports : /absolute ")
    assert pairs == [
        ("native", tmp_path / "notes"),
        ("native", Path("exports")),
        ("native", Path("/absolute")),
    ]


def test_parse_roots_treats_pathlike_equals_as_a_bare_native_path():
    assert discovery_roots.parse_roots("./dir=name") == [("native", Path("dir=name"))]
    assert discovery_roots.parse_roots("/dir=name") == [("native", Path("/dir=name"))]


def test_parse_roots_rejects_unknown_adapter_and_empty_tagged_path():
    with pytest.raises(discovery_roots.UnknownTranscriptFormatError) as unknown:
        discovery_roots.parse_roots("codex=/exports")
    assert "codex" in str(unknown.value)
    assert "native" in str(unknown.value)
    assert unknown.value.command == ""

    with pytest.raises(discovery_roots.UnknownTranscriptFormatError) as empty:
        discovery_roots.parse_roots("native=  ")
    assert "empty path" in str(empty.value)


def test_discover_external_without_configuration_is_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(discovery_roots.ENV_VAR, raising=False)
    assert discovery_roots.discover_external() == []
    assert discovery_roots.discover_external({}) == []


def test_discover_external_warns_for_missing_root(tmp_path: Path, capsys):
    missing = tmp_path / "not-mounted"
    assert discovery_roots.discover_external({discovery_roots.ENV_VAR: str(missing)}) == []
    error = capsys.readouterr().err
    assert str(missing) in error
    assert "vanished" in error


def test_discover_external_dispatches_each_configured_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    calls: list[Path] = []

    def adapter(path: Path):
        calls.append(path)
        return [path.name]

    monkeypatch.setitem(discovery_roots.ADAPTERS, "native", adapter)
    result = discovery_roots.discover_external(
        {discovery_roots.ENV_VAR: f"native={first}:{second}"}
    )
    assert result == ["one", "two"]
    assert calls == [first, second]
