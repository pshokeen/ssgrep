"""Complete tests for the framework-free portions of the MCP service."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError

import pytest

from ssgrep.services import mcp_server
from ssgrep.utilities.types import (
    ContentType,
    EmptyQueryError,
    IndexNotFoundError,
    IndexNotReadyError,
    InvalidPredicateError,
)


def test_reconcile_on_startup_calls_global_index(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(mcp_server.api, "index", lambda: calls.append("index"))

    mcp_server._reconcile_on_startup()

    assert calls == ["index"]
    assert capsys.readouterr().err == ""


def test_reconcile_on_startup_is_best_effort_and_logs_to_stderr(monkeypatch, capsys):
    def broken_index():
        raise RuntimeError("disk is read-only")

    monkeypatch.setattr(mcp_server.api, "index", broken_index)

    mcp_server._reconcile_on_startup()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "MCP startup reconciliation failed: disk is read-only\n"


def test_start_reconcile_background_launches_daemon_thread(monkeypatch):
    from unittest.mock import Mock

    started = []
    thread = Mock()
    thread.start.side_effect = lambda: started.append("started")

    def fake_thread(**kwargs):
        started.append(kwargs)
        return thread

    monkeypatch.setattr(mcp_server.threading, "Thread", fake_thread)

    mcp_server._start_reconcile_background()

    assert started[0]["target"] is mcp_server._reconcile_on_startup
    assert started[0]["daemon"] is True
    assert started[0]["name"] == "ssgrep-mcp-reconcile"
    assert started == [started[0], "started"]


def test_plain_recursively_serializes_contract_values(sample_card):
    plain_card = mcp_server._plain(sample_card)

    assert plain_card["timestamp"] == "2025-01-02T03:04:00+00:00"
    assert plain_card["content_type"] == "response"
    assert plain_card["files_touched"] == ["tests/test_example.py"]

    moment = datetime(2025, 1, 1, tzinfo=UTC)
    assert mcp_server._plain(
        {"when": moment, "values": (ContentType.PROMPT, [ContentType.RESPONSE])}
    ) == {
        "when": moment.isoformat(),
        "values": ["prompt", ["response"]],
    }
    marker = object()
    assert mcp_server._plain(marker) is marker


def test_fail_returns_structured_error_and_never_uses_stdout(capsys):
    result = mcp_server._fail("some_tool", ValueError("bad value"))

    captured = capsys.readouterr()
    assert result == {"error": "some_tool failed: bad value"}
    assert captured.out == ""
    assert captured.err == "some_tool failed: bad value\n"


def test_search_sessions_returns_bounded_plain_payload(monkeypatch, sample_response):
    response = replace(
        sample_response,
        total_matches=12,
        omitted_count=8,
        excerpts_truncated=True,
        clamped=True,
        index_empty=False,
    )
    calls = []

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return response

    monkeypatch.setattr(mcp_server.api, "search", fake_search)

    result = mcp_server.search_sessions("pytest fixtures", limit=4, where="project = 'demo'")

    assert calls == [("pytest fixtures", {"limit": 4, "where": "project = 'demo'"})]
    assert result == {
        "results": [mcp_server._plain(response.results[0])],
        "total_matches": 12,
        "omitted_count": 8,
        "excerpts_truncated": True,
        "clamped": True,
        "index_empty": False,
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (EmptyQueryError("ignored internal wording"), "Query must not be empty."),
        (InvalidPredicateError("Invalid where predicate"), "Invalid where predicate"),
        (IndexNotFoundError("Run `ssgrep index` first."), "Run `ssgrep index` first."),
        (IndexNotReadyError("Index is incomplete"), "Index is incomplete"),
    ],
)
def test_search_sessions_maps_expected_failures(monkeypatch, error, expected):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(mcp_server.api, "search", fail)

    assert mcp_server.search_sessions("query") == {"error": expected}


def test_search_sessions_maps_unexpected_failure_and_logs(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr(mcp_server.api, "search", fail)

    assert mcp_server.search_sessions("query") == {"error": "search_sessions failed: unreadable"}
    assert capsys.readouterr().err == "search_sessions failed: unreadable\n"


@pytest.mark.parametrize(
    "error",
    [IndexNotFoundError("No index"), IndexNotReadyError("Index rebuilding")],
)
def test_show_session_maps_index_failures(monkeypatch, error):
    def fail(ref):
        raise error

    monkeypatch.setattr(mcp_server.api, "show", fail)

    assert mcp_server.show_session("ref") == {"error": str(error)}


def test_show_session_maps_unexpected_failure_and_logs(monkeypatch, capsys):
    def fail(ref):
        raise RuntimeError("corrupt row")

    monkeypatch.setattr(mcp_server.api, "show", fail)

    assert mcp_server.show_session("ref") == {"error": "show_session failed: corrupt row"}
    assert capsys.readouterr().err == "show_session failed: corrupt row\n"


def test_show_session_returns_actionable_missing_episode(monkeypatch):
    monkeypatch.setattr(mcp_server.api, "show", lambda ref: None)

    assert mcp_server.show_session("missing-ref") == {
        "error": ("Episode not found: missing-ref. Run search_sessions to find valid refs.")
    }


def test_show_session_truncates_prompt_first(monkeypatch, sample_detail):
    detail = replace(sample_detail, prompt_text="abcdefgh", response_text="response")
    monkeypatch.setattr(mcp_server.api, "show", lambda ref: detail)

    result = mcp_server.show_session("ref", max_chars=3)

    assert result["prompt_text"] == "abc"
    assert result["response_text"] == ""
    assert result["truncated"] is True
    assert result["timestamp"] == "2025-01-02T03:04:00+00:00"
    assert result["files_touched"] == ["tests/test_example.py"]


def test_show_session_spends_remaining_budget_on_response(monkeypatch, sample_detail):
    detail = replace(sample_detail, prompt_text="abc", response_text="123456")
    monkeypatch.setattr(mcp_server.api, "show", lambda ref: detail)

    result = mcp_server.show_session("ref", max_chars=7)

    assert result["prompt_text"] == "abc"
    assert result["response_text"] == "1234"
    assert result["truncated"] is True


def test_show_session_clamps_negative_budget_to_zero(monkeypatch, sample_detail):
    monkeypatch.setattr(mcp_server.api, "show", lambda ref: sample_detail)

    result = mcp_server.show_session("ref", max_chars=-100)

    assert result["prompt_text"] == ""
    assert result["response_text"] == ""
    assert result["truncated"] is True


def test_show_session_folds_stored_truncation_flags(monkeypatch, sample_detail):
    stored_truncated = replace(sample_detail, response_truncated=True)
    monkeypatch.setattr(mcp_server.api, "show", lambda ref: stored_truncated)
    result = mcp_server.show_session("ref", max_chars=1_000)
    assert result["truncated"] is True

    monkeypatch.setattr(mcp_server.api, "show", lambda ref: sample_detail)
    clean = mcp_server.show_session("ref", max_chars=1_000)
    assert clean["truncated"] is False


def test_index_status_serializes_existing_index(monkeypatch, sample_stats):
    monkeypatch.setattr(mcp_server.api, "status", lambda: sample_stats)

    result = mcp_server.index_status()

    assert result == mcp_server._plain(sample_stats)
    assert result["last_index_time"] == "2025-01-02T03:04:00+00:00"
    assert "message" not in result


def test_index_status_adds_recovery_message_for_missing_index(monkeypatch, sample_stats):
    stats = replace(sample_stats, index_exists=False)
    monkeypatch.setattr(mcp_server.api, "status", lambda: stats)

    result = mcp_server.index_status()

    assert result["index_exists"] is False
    assert result["message"] == "No index found. Run `ssgrep index` to build one."


def test_index_status_maps_unexpected_failure_and_logs(monkeypatch, capsys):
    def fail():
        raise RuntimeError("status unavailable")

    monkeypatch.setattr(mcp_server.api, "status", fail)

    assert mcp_server.index_status() == {"error": "index_status failed: status unavailable"}
    assert capsys.readouterr().err == "index_status failed: status unavailable\n"


def test_ssgrep_version_returns_installed_distribution_version(monkeypatch):
    monkeypatch.setattr(
        "importlib.metadata.version", lambda name: "9.8.7" if name == "ssgrep" else None
    )

    assert mcp_server._ssgrep_version() == "9.8.7"


def test_ssgrep_version_degrades_when_distribution_metadata_is_missing(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing)

    assert mcp_server._ssgrep_version() == "unknown"


def test_get_mcp_server_configures_offline_sdk_registers_tools_and_caches(
    monkeypatch, fake_fastmcp_module
):
    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp_module)
    monkeypatch.setattr(mcp_server, "_ssgrep_version", lambda: "2.3.4")
    reconciliations = []
    monkeypatch.setattr(
        mcp_server,
        "_start_reconcile_background",
        lambda: reconciliations.append("done"),
    )
    monkeypatch.setattr(mcp_server, "_mcp", None)

    first = mcp_server.get_mcp_server()
    second = mcp_server.get_mcp_server()

    assert first is second
    assert first.name == "ssgrep"
    assert first.kwargs == {"instructions": mcp_server._INSTRUCTIONS, "version": "2.3.4"}
    assert fake_fastmcp_module.settings.show_server_banner is False
    assert fake_fastmcp_module.settings.check_for_updates == "off"
    assert reconciliations == ["done"]
    assert [(function.__name__, options) for function, options in first.registrations] == [
        (
            "search_sessions",
            {"annotations": {"readOnlyHint": True, "openWorldHint": False}},
        ),
        ("show_session", {"annotations": {"readOnlyHint": True}}),
        ("index_status", {"annotations": {"readOnlyHint": True}}),
    ]
    assert len(fake_fastmcp_module.FastMCP.instances) == 1
