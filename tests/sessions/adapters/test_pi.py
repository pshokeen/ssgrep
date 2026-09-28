"""Exhaustive unit tests for the offline Pi/Prime Agent adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.sessions import records
from ssgrep.sessions.adapters import pi as pi_module
from ssgrep.sessions.adapters.pi import PiAdapter
from ssgrep.sessions.adapters.prime_agent import PrimeAgentAdapter


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


def test_discover_identifies_runtimes_namespaces_identity_and_scopes_by_header_cwd(tmp_path):
    pi_root = tmp_path / "pi"
    prime_root = tmp_path / "prime"
    pi_path = _write(
        pi_root / "encoded-project" / "pi-file.jsonl",
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
    prime_path = _write(
        prime_root / "prime-file.jsonl",
        _header(parentSessionId="parent-id", rlmDepth=2),
    )
    # A valid transcript outside the requested cwd is not discovered.
    _write(pi_root / "other" / "other.jsonl", _header("other", "/elsewhere"))

    pi_sources = PiAdapter(pi_root).discover(scope="/work")
    prime_sources = PrimeAgentAdapter(prime_root).discover()

    assert len(pi_sources) == len(prime_sources) == 1
    pi_source = pi_sources[0]
    prime_source = prime_sources[0]
    assert pi_source.adapter == "pi"
    assert pi_source.key == f"pi:{pi_path.absolute()}"
    assert pi_source.session.path == pi_path
    assert pi_source.session.session_id == "pi:shared-id"
    assert pi_source.session.runtime == "pi"
    assert pi_source.session.project_paths == ("/work/project",)
    assert pi_source.session.source_project == "project"
    assert pi_source.session.agent_model == "used-model"
    assert pi_source.session.is_main
    assert pi_source.fingerprint.size == pi_path.stat().st_size

    assert prime_source.adapter == "prime-agent"
    assert prime_source.key == f"prime-agent:{prime_path.absolute()}"
    assert prime_source.session.session_id == "prime-agent:shared-id"
    assert prime_source.session.runtime == "prime-agent"
    assert not prime_source.session.is_main
    assert prime_source.session.parent_session_id == "prime-agent:parent-id"
    assert PrimeAgentAdapter(prime_root).discover(no_subagents=True) == []


def test_discover_fallback_metadata_and_scope_exclusion(tmp_path):
    root = tmp_path / "sessions"
    fallback = _write(
        root / "nested" / "fallback.jsonl",
        _header("fallback-id", "", rlmDepth=True),
        _entry("model_change", "m", model="fallback-model"),
    )
    source = PiAdapter(root).discover()[0]
    assert source.session.session_id == "pi:fallback-id"
    assert source.session.project_paths == ()
    assert source.session.source_project == "nested"
    assert source.session.agent_model == "fallback-model"
    assert source.session.is_main
    assert source.session.path == fallback
    assert PiAdapter(root).discover(scope="/work") == []


def test_prime_discovers_artifact_children_and_rejects_non_transcript_jsonl(tmp_path):
    root = tmp_path / "agent" / "sessions"
    _write(root / "main.jsonl", _header("main", "/work/main", rlmDepth=0))
    artifacts = root.parent / "session-artifacts" / "main"
    _write(
        artifacts / "rlm-subagents.jsonl",
        {"type": "rlm_subagent", "childId": "child", "prompt": "not a transcript"},
    )
    child = _write(
        artifacts / "sub-deadbeef" / "child.jsonl",
        _header("child", "/work/main", rlmDepth=1),
    )
    sources = PrimeAgentAdapter(root).discover()
    assert [source.session.session_id for source in sources] == [
        "prime-agent:child",
        "prime-agent:main",
    ]
    child_source = next(source for source in sources if source.session.path == child)
    assert not child_source.session.is_main
    assert [
        source.session.session_id for source in PrimeAgentAdapter(root).discover(no_subagents=True)
    ] == ["prime-agent:main"]


def test_invalid_or_empty_session_headers_are_not_discovered(tmp_path):
    root = tmp_path / "sessions"
    _write(root / "empty-id.jsonl", _header(""))
    _write(root / "wrong-first.jsonl", {"type": "custom"}, _header("later"))
    (root / "empty.jsonl").write_bytes(b"")
    assert PiAdapter(root).discover() == []


def test_discover_handles_absent_roots_directory_and_file_races(tmp_path, monkeypatch):
    adapter = PiAdapter(tmp_path / "absent")
    assert adapter.discover() == []

    root = tmp_path / "sessions"
    path = _write(root / "one.jsonl", _header())
    original_rglob = Path.rglob

    def broken_rglob(self, pattern):
        if self == root:
            raise OSError("directory disappeared")
        return original_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", broken_rglob)
    assert PiAdapter(root).discover() == []
    monkeypatch.setattr(Path, "rglob", original_rglob)

    original_read = pi_module._read_complete_records
    monkeypatch.setattr(
        pi_module,
        "_read_complete_records",
        lambda candidate: (_ for _ in ()).throw(OSError("file disappeared")),
    )
    assert PiAdapter(root).discover() == []
    monkeypatch.setattr(pi_module, "_read_complete_records", original_read)

    monkeypatch.setattr(pi_module, "file_fingerprint", lambda candidate: None)
    assert PiAdapter(root).discover() == []
    assert path.exists()


def test_explicit_native_and_ssgrep_environment_root_overrides(tmp_path, monkeypatch):
    explicit = PiAdapter("~/explicit-sessions")
    assert explicit.root == Path.home() / "explicit-sessions"

    monkeypatch.setenv("PI_SESSION_DIR", str(tmp_path / "pi-native"))
    assert PiAdapter().root == tmp_path / "pi-native"
    monkeypatch.setenv("SSGREP_PI_SESSIONS_DIR", str(tmp_path / "pi-ssgrep"))
    assert PiAdapter().root == tmp_path / "pi-ssgrep"
    monkeypatch.delenv("SSGREP_PI_SESSIONS_DIR")
    monkeypatch.delenv("PI_SESSION_DIR")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    assert PiAdapter().root == tmp_path / "pi-agent" / "sessions"

    monkeypatch.delenv("PRIME_AGENT_SESSION_DIR")
    monkeypatch.setenv("PRIME_AGENT_CODING_AGENT_SESSION_DIR", str(tmp_path / "legacy"))
    assert PrimeAgentAdapter().root == tmp_path / "legacy"
    monkeypatch.setenv("PRIME_AGENT_SESSION_DIR", str(tmp_path / "modern"))
    assert PrimeAgentAdapter().root == tmp_path / "modern"
    monkeypatch.setenv("SSGREP_PRIME_AGENT_SESSIONS_DIR", str(tmp_path / "ssgrep"))
    assert PrimeAgentAdapter().root == tmp_path / "ssgrep"


def test_default_roots_follow_home_override(tmp_path, monkeypatch):
    for name in (
        "PI_CODING_AGENT_DIR",
        "PI_CODING_AGENT_SESSION_DIR",
        "PI_SESSION_DIR",
        "PRIME_AGENT_CODING_AGENT_DIR",
        "PRIME_AGENT_CODING_AGENT_SESSION_DIR",
        "PRIME_AGENT_SESSION_DIR",
        "SSGREP_PI_SESSIONS_DIR",
        "SSGREP_PRIME_AGENT_SESSIONS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert PiAdapter().root == tmp_path / ".pi" / "agent" / "sessions"
    assert PrimeAgentAdapter().root == tmp_path / ".prime" / "agent" / "sessions"


def test_read_normalizes_signal_titles_git_models_and_tools_without_noise(tmp_path, capsys):
    secret = "MUST-NOT-LEAK"
    path = _write(
        tmp_path / "sessions" / "session.jsonl",
        _header(git={"branch": "initial"}),
        _entry("model_change", "m", modelId="chosen", provider="model-provider"),
        _entry("session_info", "title", name="A useful title"),
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
        _entry("git_state", "git", git={"branch": "feature"}),
        _entry("custom-title", "custom", **{"custom-title": "Custom"}),
        _entry("title", "generic", title="Generic"),
        _entry("session_title", "session-title", name="Session title"),
        _message("user", "plain string prompt", "u2"),
    )
    adapter = PrimeAgentAdapter(path.parent)
    source = adapter.discover()[0]
    result = adapter.read(source)

    assert result.malformed_records == result.skipped_records == 0
    assert [record["type"] for record in result.records] == [
        "custom-title",
        "user",
        "assistant",
        "custom-title",
        "custom-title",
        "custom-title",
        "user",
    ]
    title, user, assistant, custom, generic, session_title, plain = result.records
    assert title == {
        "type": "custom-title",
        "custom-title": "A useful title",
        "sessionId": "prime-agent:shared-id",
        "uuid": "prime-agent:title",
        "timestamp": "2026-01-01T00:00:01Z",
        "cwd": "/work/project",
        "gitBranch": "initial",
    }
    assert user["message"] == {
        "role": "user",
        "content": [{"type": "text", "text": "question"}],
    }
    assert user["uuid"] == "prime-agent:u"
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
    assert custom["custom-title"] == "Custom"
    assert generic["custom-title"] == "Generic"
    assert session_title["custom-title"] == "Session title"
    assert plain["message"]["content"] == [{"type": "text", "text": "plain string prompt"}]
    assert plain["gitBranch"] == "feature"
    assert secret not in json.dumps(result.records)
    assert capsys.readouterr().out == capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "ignored_type",
    sorted(pi_module._IGNORED_ENTRY_TYPES),
)
def test_read_silently_ignores_known_non_signal_entries(tmp_path, ignored_type):
    path = _write(
        tmp_path / ignored_type / "session.jsonl",
        _header(),
        _entry(ignored_type, "ignored", summary="not signal"),
    )
    source = PiAdapter(path.parent).discover()[0]
    assert PiAdapter(path.parent).read(source).skipped_records == 0


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
    adapter = PiAdapter(path.parent)
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
    assert pi_module._read_complete_records(complete) == pi_module.ReadResult(())
    assert pi_module._read_complete_records(incomplete) == pi_module.ReadResult(())


def test_semantically_invalid_titles_and_empty_message_blocks_are_handled(tmp_path):
    path = _write(
        tmp_path / "sessions" / "invalid.jsonl",
        _header(modelId="header-model", branch="header-branch"),
        _entry("model_change", "m", model="changed", provider="provider"),
        _entry("git", "g", branch="branch-from-record"),
        _entry("session_info", "bad-title", name=123),
        _message("assistant", [], "a"),
        _message("user", [], "u"),
    )
    adapter = PiAdapter(path.parent)
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


def test_present_reflects_transcript_file_existence(tmp_path):
    path = _write(tmp_path / "sessions" / "session.jsonl", _header())
    source = PiAdapter(path.parent).discover()[0]
    assert PiAdapter(path.parent).present(source) is True

    path.unlink()
    assert PiAdapter(path.parent).present(source) is False
