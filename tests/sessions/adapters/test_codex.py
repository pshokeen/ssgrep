"""Unit tests for the offline Codex rollout adapter."""

from __future__ import annotations

import json
from pathlib import Path

import ssgrep.sessions.adapters.codex as codex_module
from ssgrep.sessions.adapters.codex import CodexAdapter


def _json_line(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode() + b"\n"


def _write(path: Path, *values: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_line(value) for value in values))
    return path


def _envelope(record_type: str, payload: object, timestamp: str = "2026-01-01T00:00:00Z") -> dict:
    return {"timestamp": timestamp, "type": record_type, "payload": payload}


def _session_meta(session_id: str = "abc123", cwd: str = "/work/project") -> dict:
    return _envelope(
        "session_meta",
        {
            "id": session_id,
            "timestamp": "2026-01-01T00:00:00Z",
            "cwd": cwd,
            "originator": "codex_exec",
            "model_provider": "openai",
            "base_instructions": {"text": "system"},
        },
    )


def _turn_context(model: str = "gpt-5.5") -> dict:
    return _envelope(
        "turn_context",
        {"turn_id": "turn-1", "cwd": "/work/project", "model": model},
    )


def _message(role: str, *blocks: dict, timestamp: str = "2026-01-01T00:00:01Z") -> dict:
    return _envelope(
        "response_item",
        {"type": "message", "role": role, "content": list(blocks)},
        timestamp=timestamp,
    )


def _text_block(text: str, block_type: str = "input_text") -> dict:
    return {"type": block_type, "text": text}


def _function_call(name: str, arguments: object, call_id: str = "call_1") -> dict:
    payload: dict = {"type": "function_call", "name": name, "call_id": call_id}
    if isinstance(arguments, str):
        payload["arguments"] = arguments
    else:
        payload["arguments"] = json.dumps(arguments, separators=(",", ":"))
    return _envelope("response_item", payload, timestamp="2026-01-01T00:00:02Z")


def _custom_tool_call(name: str, input_value: object, call_id: str = "call_2") -> dict:
    return _envelope(
        "response_item",
        {"type": "custom_tool_call", "name": name, "call_id": call_id, "input": input_value},
        timestamp="2026-01-01T00:00:03Z",
    )


def _reasoning() -> dict:
    return _envelope(
        "response_item",
        {"type": "reasoning", "summary": [], "content": None},
        timestamp="2026-01-01T00:00:04Z",
    )


def test_discover_identifies_runtime_namespaces_identity_and_cwd(tmp_path):
    path = _write(
        tmp_path / "sessions" / "2026" / "01" / "01" / "rollout.jsonl",
        _session_meta("abc123", "/work/project"),
        _turn_context("gpt-5.5"),
        _message("user", _text_block("hello")),
    )
    sources = CodexAdapter(tmp_path / "sessions").discover()

    assert len(sources) == 1
    source = sources[0]
    assert source.adapter == "codex"
    assert source.key == f"codex:{path.absolute()}"
    assert source.session.path == path
    assert source.session.session_id == "codex:abc123"
    assert source.session.runtime == "codex"
    assert source.session.project_paths == ("/work/project",)
    assert source.session.source_project == "project"
    assert source.session.agent_model == "gpt-5.5"
    assert source.session.is_main
    assert source.session.parent_session_id is None
    assert source.fingerprint.size == path.stat().st_size


def test_discover_fallback_source_project_without_cwd(tmp_path):
    _write(
        tmp_path / "sessions" / "nested" / "rollout.jsonl",
        _session_meta("noid-cwd", ""),
    )
    source = CodexAdapter(tmp_path / "sessions").discover()[0]
    assert source.session.project_paths == ()
    assert source.session.source_project == "nested"
    assert source.session.agent_model is None
    assert CodexAdapter(tmp_path / "sessions").discover(scope="/work") == []


def test_discover_ignores_non_session_and_missing_root(tmp_path):
    empty_root = tmp_path / "sessions"
    assert CodexAdapter(empty_root).discover() == []
    _write(
        empty_root / "not-a-session.jsonl",
        _message("user", _text_block("no meta header")),
    )
    assert CodexAdapter(empty_root).discover() == []


def test_read_normalizes_messages_and_drops_noise(tmp_path):
    _write(
        tmp_path / "rollout.jsonl",
        _session_meta("abc", "/work/project"),
        _turn_context("gpt-5.5"),
        _message("developer", _text_block("system prompt")),
        _message("user", _text_block("do the thing")),
        _reasoning(),
        _message("assistant", _text_block("on it", "output_text")),
        _function_call("exec_command", {"cmd": "pwd"}),
        _envelope(
            "response_item",
            {"type": "function_call_output", "call_id": "call_1", "output": "x"},
        ),
        _envelope("response_item", {"type": "token_count", "tokens": 5}),
    )
    source = CodexAdapter(tmp_path).discover()[0]
    result = CodexAdapter(tmp_path).read(source)

    # developer, reasoning, function_call_output, and token_count are excluded;
    # user + assistant (+ attached tool) survive.
    assert result.malformed_records == 0
    assert result.skipped_records == 0
    types = [record["type"] for record in result.records]
    assert types == ["user", "assistant"]

    user = result.records[0]
    assert user["message"]["role"] == "user"
    assert user["message"]["content"] == [{"type": "text", "text": "do the thing"}]
    assert user["cwd"] == "/work/project"
    assert user["sessionId"] == "codex:abc"

    assistant = result.records[1]
    assert assistant["message"]["role"] == "assistant"
    assert assistant["message"]["model"] == "gpt-5.5"
    blocks = assistant["message"]["content"]
    assert blocks[0] == {"type": "text", "text": "on it"}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["name"] == "exec_command"
    assert blocks[1]["input"] == {"cmd": "pwd"}
    assert blocks[1]["id"] == "call_1"


def test_read_attaches_multiple_tools_to_one_assistant_turn(tmp_path):
    _write(
        tmp_path / "rollout.jsonl",
        _session_meta("abc"),
        _message("assistant", _text_block("reading", "output_text")),
        _function_call("read", {"file_path": "/work/a.py"}, "call_1"),
        _custom_tool_call("apply_patch", "*** Begin Patch\n*** Update File: b.py", "call_2"),
        _message("user", _text_block("next")),
        _message("assistant", _text_block("done", "output_text")),
    )
    source = CodexAdapter(tmp_path).discover()[0]
    result = CodexAdapter(tmp_path).read(source)

    assistant = result.records[0]
    blocks = assistant["message"]["content"]
    assert [block["type"] for block in blocks] == ["text", "tool_use", "tool_use"]
    assert blocks[1]["name"] == "read"
    assert blocks[1]["input"] == {"file_path": "/work/a.py"}
    assert blocks[2]["name"] == "apply_patch"
    assert blocks[2]["input"] == {"input": "*** Begin Patch\n*** Update File: b.py"}
    # The second user turn closes the first assistant turn.
    assert result.records[1]["type"] == "user"
    assert result.records[2]["message"]["content"] == [{"type": "text", "text": "done"}]


def test_read_tool_only_assistant_turn(tmp_path):
    _write(
        tmp_path / "rollout.jsonl",
        _session_meta("abc"),
        _message("user", _text_block("go")),
        _function_call("exec_command", {"cmd": "ls"}),
    )
    source = CodexAdapter(tmp_path).discover()[0]
    result = CodexAdapter(tmp_path).read(source)
    assert [record["type"] for record in result.records] == ["user", "assistant"]
    tool = result.records[1]["message"]["content"][0]
    assert tool["type"] == "tool_use"
    assert tool["name"] == "exec_command"


def test_read_tolerates_incomplete_tail(tmp_path):
    path = tmp_path / "rollout.jsonl"
    complete = _json_line(_session_meta("abc")) + _json_line(_message("user", _text_block("hi")))
    path.write_bytes(complete + b'{"timestamp": "x", "type": "response_item", "payload": {}')
    source = CodexAdapter(tmp_path).discover()[0]
    result = CodexAdapter(tmp_path).read(source)
    assert [record["type"] for record in result.records] == ["user"]
    assert result.skipped_records == 1


def _string_message(role: str, content: object) -> dict:
    return _envelope(
        "response_item",
        {"type": "message", "role": role, "content": content},
        timestamp="2026-01-01T00:00:01Z",
    )


def test_read_complete_records_counts_degraded_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    oversized = b'{"type": "response_item", "payload": {"pad": "' + b"x" * 600_000 + b'"}}\n'
    path.write_bytes(
        b"".join(
            [
                _json_line(_session_meta("abc")),
                oversized,
                b"\n",
                b"this is not json\n",
                b'["not", "a", "dict"]\n',
                _json_line(_message("user", _text_block("ok"))),
            ]
        )
    )

    result = codex_module._read_complete_records(path)

    assert result.malformed_records == 1
    assert result.skipped_records == 2
    assert [record["type"] for record in result.records] == ["session_meta", "response_item"]


def test_inspect_skips_records_with_non_dict_payload():
    info = codex_module._inspect(
        (
            _session_meta("abc", "/work/project"),
            {"timestamp": "t", "type": "event_msg", "payload": "not-a-dict"},
        ),
        "codex",
    )
    assert info is not None
    assert info.session_id == "codex:abc"
    assert info.cwd == "/work/project"


def test_text_blocks_string_and_invalid_content():
    assert codex_module._text_blocks("hi") == [{"type": "text", "text": "hi"}]
    assert codex_module._text_blocks("") == []
    assert codex_module._text_blocks(123) is None
    assert codex_module._text_blocks(
        [
            {"type": "text", "text": "kept"},
            "junk",
            {"type": "refusal", "refusal": "no"},
        ]
    ) == [{"type": "text", "text": "kept"}]


def test_tool_block_edge_cases():
    assert codex_module._tool_block({"type": "function_call", "arguments": "{}"}) is None
    bad = codex_module._tool_block({"type": "function_call", "name": "x", "arguments": "not json"})
    assert bad is not None
    assert bad["input"] == {"arguments": "not json"}
    passed = codex_module._tool_block({"type": "function_call", "name": "x", "arguments": {"a": 1}})
    assert passed is not None
    assert passed["input"] == {"a": 1}
    assert codex_module._tool_block({"type": "custom_tool_call", "input": {}}) is None
    assert codex_module._tool_block({"type": "unknown"}) is None


def test_normalize_drops_world_state_and_invalid_signal(tmp_path):
    _write(
        tmp_path / "rollout.jsonl",
        _session_meta("abc"),
        _envelope("world_state", {"snapshot": []}),
        _string_message("user", 123),
        _string_message("assistant", 123),
        _envelope(
            "response_item",
            {"type": "function_call", "arguments": "{}"},
        ),
    )
    source = CodexAdapter(tmp_path).discover()[0]
    result = CodexAdapter(tmp_path).read(source)
    assert result.records == ()


def test_normalize_tool_only_assistant_carries_model(tmp_path):
    _write(
        tmp_path / "rollout.jsonl",
        _session_meta("abc"),
        _turn_context("gpt-5.5"),
        _message("user", _text_block("go")),
        _function_call("exec_command", {"cmd": "ls"}),
    )
    source = CodexAdapter(tmp_path).discover()[0]
    result = CodexAdapter(tmp_path).read(source)
    assistant = result.records[1]
    assert assistant["type"] == "assistant"
    assert assistant["message"]["model"] == "gpt-5.5"
    assert assistant["message"]["content"][0]["name"] == "exec_command"


def test_root_resolves_from_env_var(tmp_path, monkeypatch):
    sessions = tmp_path / "env-sessions"
    monkeypatch.setenv("SSGREP_CODEX_SESSIONS_DIR", str(sessions))
    adapter = CodexAdapter()
    assert adapter.root == sessions


def test_discover_returns_empty_on_rglob_error(tmp_path, monkeypatch):
    class _RaisingRoot:
        def is_dir(self):
            return True

        def rglob(self, pattern):
            raise OSError("boom")

    monkeypatch.setattr(CodexAdapter, "root", property(lambda self: _RaisingRoot()))
    assert CodexAdapter().discover() == []


def test_discover_skips_unreadable_and_unfingerprintable_sources(tmp_path, monkeypatch):
    _write(tmp_path / "good.jsonl", _session_meta("good", "/work/project"))
    _write(tmp_path / "bad.jsonl", _session_meta("bad", "/work/project"))

    real_read = codex_module._read_complete_records

    def fake_read(path):
        if path.name == "bad.jsonl":
            raise OSError("boom")
        return real_read(path)

    monkeypatch.setattr(codex_module, "_read_complete_records", fake_read)
    sources = CodexAdapter(tmp_path).discover()
    assert [source.session.session_id for source in sources] == ["codex:good"]

    monkeypatch.setattr(codex_module, "file_fingerprint", lambda _path: None)
    assert CodexAdapter(tmp_path).discover() == []


def test_present_reflects_transcript_file_existence(tmp_path):
    path = _write(tmp_path / "rollout.jsonl", _session_meta("abc", "/work/project"))
    source = CodexAdapter(tmp_path).discover()[0]
    assert source.session.path == path
    assert CodexAdapter(tmp_path).present(source) is True

    path.unlink()
    assert CodexAdapter(tmp_path).present(source) is False
