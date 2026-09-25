"""Exhaustive unit tests for the offline omp adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.sessions import records
from ssgrep.sessions.adapters import pi as pi_module
from ssgrep.sessions.adapters.base import ReadResult
from ssgrep.sessions.adapters.omp import OmpAdapter
from ssgrep.sessions.adapters.pi import (
    _IGNORED_ENTRY_TYPES,
    _read_complete_records,
)


def _json_line(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode() + b"\n"


def _write(path: Path, *values: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_line(value) for value in values))
    return path


def _header(
    session_id: str = "shared-id",
    cwd: str = "/work/project",
    **extra: object,
) -> dict:
    return {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": "2026-01-01T00:00:00Z",
        "cwd": cwd,
        **extra,
    }


def _entry(entry_type: str, entry_id: str, **extra: object) -> dict:
    return {
        "type": entry_type,
        "id": entry_id,
        "parentId": None,
        "timestamp": "2026-01-01T00:00:01Z",
        **extra,
    }


def _message(role: str, content: object, entry_id: str, **extra: object) -> dict:
    return _entry(
        "message",
        entry_id,
        message={"role": role, "content": content, **extra},
    )


def test_discover_namespaces_identity_and_scopes_by_header_cwd(tmp_path):
    root = tmp_path / "sessions"
    path = _write(
        root / "-ghq-github.com-pshokeen-ssgrep" / "session.jsonl",
        _header(git={"branch": "main"}, model="header-model"),
        _entry("model_change", "model", modelId="selected-model", provider="provider"),
        _message(
            "assistant",
            [{"type": "text", "text": "answer"}],
            "a",
            model="used-model",
            provider="provider",
        ),
    )
    # A valid transcript outside the requested cwd is not discovered.
    _write(root / "other" / "other.jsonl", _header("other", "/elsewhere/project"))

    sources = OmpAdapter(root).discover(scope="/work")

    assert len(sources) == 1
    source = sources[0]
    assert source.adapter == "omp"
    assert source.key == f"omp:{path.absolute()}"
    assert source.session.path == path
    assert source.session.session_id == "omp:shared-id"
    assert source.session.runtime == "omp"
    assert source.session.project_paths == ("/work/project",)
    assert source.session.source_project == "project"
    assert source.session.agent_model == "used-model"
    assert source.session.is_main
    assert source.fingerprint.size == path.stat().st_size
    assert [s.session.session_id for s in OmpAdapter(root).discover(scope="/elsewhere")] == [
        "omp:other"
    ]


def test_discover_child_sessions_and_no_subagents(tmp_path):
    root = tmp_path / "sessions"
    _write(root / "child.jsonl", _header(parentSessionId="parent-id", rlmDepth=2))
    _write(root / "main.jsonl", _header("main", rlmDepth=0))

    sources = OmpAdapter(root).discover()
    assert [source.session.is_main for source in sources] == [False, True]
    child = sources[0]
    assert child.session.session_id == "omp:shared-id"
    assert child.session.parent_session_id == "omp:parent-id"
    assert [
        source.session.session_id for source in OmpAdapter(root).discover(no_subagents=True)
    ] == ["omp:main"]


def test_discover_fallback_metadata_and_scope_exclusion(tmp_path):
    root = tmp_path / "sessions"
    fallback = _write(
        root / "nested" / "fallback.jsonl",
        _header("fallback-id", "", rlmDepth=True),
        _entry("model_change", "m", model="fallback-model"),
    )
    source = OmpAdapter(root).discover()[0]
    assert source.session.session_id == "omp:fallback-id"
    assert source.session.project_paths == ()
    assert source.session.source_project == "nested"
    assert source.session.agent_model == "fallback-model"
    assert source.session.is_main
    assert source.session.path == fallback
    assert OmpAdapter(root).discover(scope="/work") == []


def test_invalid_or_empty_session_headers_are_not_discovered(tmp_path):
    root = tmp_path / "sessions"
    _write(root / "empty-id.jsonl", _header(""))
    _write(root / "wrong-first.jsonl", {"type": "custom"}, _header("later"))
    (root / "empty.jsonl").write_bytes(b"")
    (root / "lock.jsonl").write_bytes(b"")
    assert OmpAdapter(root).discover() == []


def test_leading_title_record_before_session_header_is_discovered(tmp_path):
    """omp writes a `title` record before the `session` header; Pi never does."""
    root = tmp_path / "sessions"
    _write(
        root / "title-first.jsonl",
        {"type": "title", "v": 1, "title": "", "updatedAt": "2026-01-01T00:00:00Z", "pad": ""},
        _header("omp-session", "/work/project"),
        _message("user", [{"type": "text", "text": "question"}], "u"),
    )
    sources = OmpAdapter(root).discover()
    assert len(sources) == 1
    source = sources[0]
    assert source.session.session_id == "omp:omp-session"
    result = OmpAdapter(root).read(source)
    # The leading `title` record has an empty auto-title: skipped as unsignal.
    assert result.skipped_records == 1
    assert [record["type"] for record in result.records] == ["user"]


def test_discover_handles_absent_roots_directory_and_file_races(tmp_path, monkeypatch):
    adapter = OmpAdapter(tmp_path / "absent")
    assert adapter.discover() == []

    root = tmp_path / "sessions"
    path = _write(root / "one.jsonl", _header())
    original_rglob = Path.rglob

    def broken_rglob(self, pattern):
        if self == root:
            raise OSError("directory disappeared")
        return original_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", broken_rglob)
    assert OmpAdapter(root).discover() == []
    monkeypatch.setattr(Path, "rglob", original_rglob)

    original_read = _read_complete_records
    monkeypatch.setattr(
        pi_module,
        "_read_complete_records",
        lambda candidate: (_ for _ in ()).throw(OSError("file disappeared")),
    )
    assert OmpAdapter(root).discover() == []
    monkeypatch.setattr(pi_module, "_read_complete_records", original_read)

    monkeypatch.setattr(pi_module, "file_fingerprint", lambda candidate: None)
    assert OmpAdapter(root).discover() == []
    assert path.exists()


def test_explicit_native_and_ssgrep_environment_root_overrides(tmp_path, monkeypatch):
    explicit = OmpAdapter("~/explicit-sessions")
    assert explicit.root == Path.home() / "explicit-sessions"

    monkeypatch.setenv("OMP_SESSIONS_DIR", str(tmp_path / "omp-native"))
    assert OmpAdapter().root == tmp_path / "omp-native"
    monkeypatch.setenv("SSGREP_OMP_SESSIONS_DIR", str(tmp_path / "omp-ssgrep"))
    assert OmpAdapter().root == tmp_path / "omp-ssgrep"
    monkeypatch.delenv("SSGREP_OMP_SESSIONS_DIR")
    monkeypatch.delenv("OMP_SESSIONS_DIR")
    monkeypatch.setenv("OMP_AGENT_DIR", str(tmp_path / "omp-agent"))
    assert OmpAdapter().root == tmp_path / "omp-agent" / "sessions"


def test_default_roots_follow_home_override(tmp_path, monkeypatch):
    for name in (
        "OMP_AGENT_DIR",
        "OMP_SESSIONS_DIR",
        "SSGREP_OMP_SESSIONS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert OmpAdapter().root == tmp_path / ".omp" / "agent" / "sessions"


def test_read_normalizes_signal_titles_models_and_tools_without_noise(tmp_path, capsys):
    secret = "MUST-NOT-LEAK"
    path = _write(
        tmp_path / "sessions" / "session.jsonl",
        _header(git={"branch": "initial"}),
        _entry("model_change", "m", modelId="chosen", provider="model-provider"),
        _entry("title_change", "title", title="A useful title", source="auto"),
        _message(
            "user",
            [
                {"type": "text", "text": "question"},
                {"type": "thinking", "thinking": secret},
                {"type": "image", "data": secret},
                {"type": "toolCall", "name": "not-a-user-tool", "arguments": {}},
                "not-a-block",
            ],
            "u",
        ),
        _message(
            "assistant",
            [
                {"type": "thinking", "thinking": secret},
                {"type": "text", "text": "answer"},
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "Read",
                    "arguments": {"file_path": "src/app.py"},
                },
                {"type": "toolCall", "name": "Odd", "arguments": ["unusual", "but", "preserved"]},
                {"type": "toolCall", "arguments": {}},
                {"type": "future", "payload": secret},
                42,
            ],
            "a",
            model="actual-model",
            provider="actual-provider",
        ),
        _message(
            "toolResult",
            [{"type": "text", "text": secret}],
            "tool-result",
            toolName="Read",
        ),
        _entry("custom", "ignored", customType="tool_execution_start", data={"toolName": secret}),
        _entry("custom_message", "ignored-2", customType="lsp-late-diagnostic", content=secret),
        _entry("thinking_level_change", "ignored-3", thinkingLevel="high"),
        _entry("service_tier_change", "ignored-4", serviceTier=None),
        _entry("title", "generic", title="Generic"),
        _message("user", "plain string prompt", "u2"),
    )
    adapter = OmpAdapter(path.parent)
    source = adapter.discover()[0]
    result = adapter.read(source)

    assert result.malformed_records == result.skipped_records == 0
    assert [record["type"] for record in result.records] == [
        "custom-title",
        "user",
        "assistant",
        "custom-title",
        "user",
    ]
    title, user, assistant, generic, plain = result.records
    assert title == {
        "type": "custom-title",
        "custom-title": "A useful title",
        "sessionId": "omp:shared-id",
        "uuid": "omp:title",
        "timestamp": "2026-01-01T00:00:01Z",
        "cwd": "/work/project",
        "gitBranch": "initial",
    }
    assert user["message"] == {
        "role": "user",
        "content": [{"type": "text", "text": "question"}],
    }
    assert user["uuid"] == "omp:u"
    assert assistant["message"] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "answer"},
            {
                "type": "tool_use",
                "input": {"file_path": "src/app.py"},
                "id": "call-1",
                "name": "Read",
            },
            {"type": "tool_use", "name": "Odd", "input": ["unusual", "but", "preserved"]},
        ],
        "model": "actual-model",
        "provider": "actual-provider",
    }
    assert generic["custom-title"] == "Generic"
    assert plain["message"]["content"] == [{"type": "text", "text": "plain string prompt"}]
    assert secret not in json.dumps(result.records)
    assert capsys.readouterr().out == capsys.readouterr().err == ""


_OMP_QUIET_TYPES = _IGNORED_ENTRY_TYPES | {
    "custom",
    "custom_message",
    "thinking_level_change",
    "service_tier_change",
}


@pytest.mark.parametrize(
    "ignored_type",
    sorted(_OMP_QUIET_TYPES),
)
def test_read_silently_ignores_known_non_signal_entries(tmp_path, ignored_type):
    path = _write(
        tmp_path / ignored_type / "session.jsonl",
        _header(),
        _entry(ignored_type, "ignored", summary="not signal"),
    )
    source = OmpAdapter(path.parent).discover()[0]
    assert OmpAdapter(path.parent).read(source).skipped_records == 0


def test_read_reports_malformed_oversized_nonobjects_unknown_and_truncated_records(tmp_path):
    path = tmp_path / "sessions" / "degraded.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        _json_line(_header())
        + b"\n"
        + b'{"type": broken}\n'
        + b'{"type":"message","message":{"role":"user","content":"bad-utf8-\xff"}}\n'
        + _json_line(["not", "an", "object"])
        + (b"x" * (records.MAX_LINE_BYTES + 1))
        + b"\n"
        + _json_line({"type": "future-entry"})
        + _json_line({"type": "message", "message": "not-an-object"})
        + _json_line(_message("future-role", [], "bad-role"))
        + _json_line(_message("user", {"not": "valid content"}, "bad-content"))
        + b'{"type":"message"'
    )
    adapter = OmpAdapter(path.parent)
    source = adapter.discover()[0]
    result = adapter.read(source)
    assert result.records == ()
    assert result.malformed_records == 2
    assert result.skipped_records == 7


def test_complete_blank_and_incomplete_blank_lines_are_ignored(tmp_path):
    complete = tmp_path / "complete.jsonl"
    complete.write_bytes(b"  \r\n")
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_bytes(b"   ")
    assert _read_complete_records(complete) == ReadResult(())
    assert _read_complete_records(incomplete) == ReadResult(())


def test_semantically_invalid_titles_and_empty_message_blocks_are_handled(tmp_path):
    path = _write(
        tmp_path / "sessions" / "invalid.jsonl",
        _header(modelId="header-model", branch="header-branch"),
        _entry("model_change", "m", model="changed", provider="provider"),
        _entry("git", "g", branch="branch-from-record"),
        _entry("title_change", "bad-title", title=123),
        _message("assistant", [], "a"),
        _message("user", [], "u"),
    )
    adapter = OmpAdapter(path.parent)
    source = adapter.discover()[0]
    result = adapter.read(source)
    assert result.skipped_records == 1
    assert [record["message"] for record in result.records] == [
        {
            "role": "assistant",
            "content": [],
            "model": "changed",
            "provider": "provider",
        },
        {"role": "user", "content": []},
    ]
    assert all(record["gitBranch"] == "branch-from-record" for record in result.records)
