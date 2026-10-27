"""Complete tests for the framework-free portions of the MCP service."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import threading
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
    InvalidLimitError,
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
    monkeypatch.setattr(mcp_server, "_reconcile_thread", None)

    mcp_server._start_reconcile_background()

    assert started[0]["target"] is mcp_server._reconcile_on_startup
    assert started[0]["daemon"] is True
    assert started[0]["name"] == "ssgrep-mcp-reconcile"
    assert started == [started[0], "started"]
    assert mcp_server._reconcile_thread is thread


@pytest.fixture
def blocked_reconcile(monkeypatch):
    """A live stand-in for a background reconcile stuck in native code."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    monkeypatch.setattr(mcp_server, "_reconcile_thread", thread)
    monkeypatch.setattr(mcp_server, "_RECONCILE_GRACE_SECONDS", 0.05)
    yield thread
    release.set()
    thread.join()


def test_tools_report_update_in_progress_without_entering_the_pipeline_during_reconcile(
    monkeypatch, blocked_reconcile
):
    """The pipeline marks the index incomplete for the whole reconcile and shares
    native state with it, so tools answer "retry" instead of calling it."""

    def must_not_run(*args, **kwargs):
        raise AssertionError("the pipeline must not be entered during the reconcile")

    monkeypatch.setattr(mcp_server.api, "search", must_not_run)
    monkeypatch.setattr(mcp_server.api, "show", must_not_run)

    for result in (mcp_server.search_sessions("query"), mcp_server.show_session("ref")):
        assert result == {
            "error": (
                "The global index is being updated by this server's background "
                "reconcile; retry in a moment."
            )
        }
        assert "ssgrep index" not in result["error"]


def test_tools_use_the_api_again_once_the_reconcile_has_finished(monkeypatch, sample_response):
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()
    monkeypatch.setattr(mcp_server, "_reconcile_thread", finished)
    monkeypatch.setattr(mcp_server.api, "search", lambda *args, **kwargs: sample_response)

    assert "error" not in mcp_server.search_sessions("query")


def test_settle_reconcile_reports_finished_and_abandoned_threads(monkeypatch, blocked_reconcile):
    assert mcp_server._settle_reconcile() is False

    monkeypatch.setattr(mcp_server, "_reconcile_thread", None)
    assert mcp_server._settle_reconcile() is True

    done = threading.Thread(target=lambda: None)
    done.start()
    monkeypatch.setattr(mcp_server, "_reconcile_thread", done)
    assert mcp_server._settle_reconcile() is True


def test_flush_standard_streams_ignores_closed_streams(monkeypatch):
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stdout", closed)
    monkeypatch.setattr(sys, "stderr", closed)

    mcp_server._flush_standard_streams()


def test_stop_on_signal_settles_then_dies_by_the_signal(monkeypatch):
    events = []
    monkeypatch.setattr(mcp_server, "_settle_reconcile", lambda: events.append("settle"))
    monkeypatch.setattr(mcp_server, "_flush_standard_streams", lambda: events.append("flush"))
    monkeypatch.setattr(
        mcp_server.signal,
        "signal",
        lambda signum, handler: events.append(("reset", signum, handler)),
    )
    monkeypatch.setattr(
        mcp_server.os, "kill", lambda pid, signum: events.append(("kill", pid, signum))
    )

    mcp_server._stop_on_signal(signal.SIGINT, None)

    assert events == [
        "settle",
        "flush",
        ("reset", signal.SIGINT, signal.SIG_DFL),
        ("kill", os.getpid(), signal.SIGINT),
    ]


def test_stop_on_signal_dies_by_the_signal_when_it_interrupts_a_stdout_flush(monkeypatch):
    """CPython runs the handler inside the main thread's interrupted flush; the
    handler's own flush of that buffer is reentrant and must not stop the kill."""
    events = []
    monkeypatch.setattr(mcp_server, "_settle_reconcile", lambda: events.append("settle"))
    monkeypatch.setattr(mcp_server.signal, "signal", lambda signum, handler: None)
    monkeypatch.setattr(
        mcp_server.os, "kill", lambda pid, signum: events.append(("kill", pid, signum))
    )

    class InterruptedRaw(io.RawIOBase):
        interrupted = False

        def writable(self):
            return True

        def write(self, data):
            if not self.interrupted:
                self.interrupted = True
                mcp_server._stop_on_signal(signal.SIGTERM, None)
            return len(data)

    stdout = io.TextIOWrapper(io.BufferedWriter(InterruptedRaw()))
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    stdout.write("partial line")
    stdout.flush()

    assert events == ["settle", ("kill", os.getpid(), signal.SIGTERM)]


class _FakeServer:
    def __init__(self, events):
        self.events = events

    def run(self):
        self.events.append("run")


def _stub_serve(monkeypatch, events, *, settled):
    monkeypatch.setattr(
        mcp_server.signal,
        "signal",
        lambda signum, handler: events.append(("handler", signum, handler)),
    )
    monkeypatch.setattr(mcp_server, "get_mcp_server", lambda: _FakeServer(events))
    monkeypatch.setattr(mcp_server, "_settle_reconcile", lambda: events.append("settle") or settled)
    monkeypatch.setattr(mcp_server, "_flush_standard_streams", lambda: events.append("flush"))
    monkeypatch.setattr(mcp_server.os, "_exit", lambda code: events.append(("exit", code)))


def test_serve_installs_handlers_first_and_returns_normally_when_reconcile_finished(
    monkeypatch,
):
    events = []
    _stub_serve(monkeypatch, events, settled=True)

    mcp_server.serve()

    assert events == [
        ("handler", signal.SIGINT, mcp_server._stop_on_signal),
        ("handler", signal.SIGTERM, mcp_server._stop_on_signal),
        "run",
        "settle",
    ]


def test_serve_abandons_a_stuck_reconcile_without_interpreter_finalization(monkeypatch):
    events = []
    _stub_serve(monkeypatch, events, settled=False)

    mcp_server.serve()

    assert events[-4:] == ["run", "settle", "flush", ("exit", 0)]


_SIGNAL_CHILD = """
import signal, threading, time
from ssgrep.services import mcp_server

mcp_server._RECONCILE_GRACE_SECONDS = 0.2
release = threading.Event()
stuck = threading.Thread(target=release.wait, daemon=True)
stuck.start()
mcp_server._reconcile_thread = stuck

class Server:
    def run(self):
        print("ready", flush=True)
        # Bounded naps, not one long sleep: a signal that lands just before a long
        # sleep's syscall sets CPython's flag without interrupting it, so the handler
        # would not run until the sleep ends. Here it runs within one 50 ms nap.
        for _ in range(1200):
            time.sleep(0.05)

mcp_server.get_mcp_server = lambda: Server()
mcp_server.serve()
"""


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_serve_stops_on_signal_with_stuck_reconcile_and_no_traceback(signum):
    """A real process stops promptly by the signal, even with the reconcile hung."""
    child = subprocess.Popen(
        [sys.executable, "-c", _SIGNAL_CHILD],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signum)
        _, stderr = child.communicate(timeout=30)
    finally:
        child.kill()

    assert child.returncode == -signum
    assert "Traceback" not in stderr


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
        (InvalidLimitError("Limit must be at least 1."), "Limit must be at least 1."),
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
    assert len(fake_fastmcp_module.FastMCP.instances) == 1


def test_save_note_persists_original_investment_and_enriches_without_duplicate_text(
    tmp_path, monkeypatch
):
    import json

    from ssgrep.services import knowledge
    from ssgrep.sessions import notes

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(knowledge, "_index_saved_note", lambda: None)
    metadata = {
        "work_id": "join-investigation",
        "provider": "anthropic",
        "model": "test-model",
        "elapsed_seconds": 840,
        "basis": "agent_estimated",
    }
    saved = mcp_server.save_note(
        title="Why did the join duplicate bookings?",
        body="Join on booking ID and flight ID.",
        project_dir=str(tmp_path),
        work=metadata,
    )
    assert saved["ok"] and saved["indexed"]
    assert saved["work"]["work"]["elapsed_seconds"] == 840
    rows = [
        json.loads(line)
        for line in next(notes.notes_dir().glob("*.jsonl")).read_text().splitlines()
    ]
    assert [row["type"] for row in rows] == ["user", "assistant"]
    assert rows[1]["message"]["content"][0]["text"] == "Join on booking ID and flight ID."


def test_invalid_save_work_is_rejected_before_creating_a_note(tmp_path, monkeypatch):
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    result = mcp_server.save_note(
        title="question", body="answer", work={"work_id": "bad", "elapsed_seconds": True}
    )
    assert result["ok"] is False
    assert result["condition"] == "invalid_work"
    assert not (tmp_path / "data").exists()


def test_enrich_existing_reference_preserves_text_and_retains_revisions(
    tmp_path, monkeypatch, sample_detail
):
    from ssgrep.search import detail
    from ssgrep.services.knowledge import work_for
    from ssgrep.sessions.work import work_history

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(detail, "show", lambda ref: sample_detail)
    metadata = {"work_id": "old-investigation", "model": "known-model", "elapsed_seconds": 12}
    first = mcp_server.save_note(ref=sample_detail.episode_id, work=metadata)
    assert first["enriched"] and first["work"]["revision"] == 1
    again = mcp_server.save_note(ref=sample_detail.episode_id, work=metadata)
    assert again["work"]["revision"] == 1
    revised = mcp_server.save_note(
        ref=sample_detail.episode_id, work={**metadata, "elapsed_seconds": 15}
    )
    assert revised["work"]["revision"] == 2
    assert work_for(sample_detail) == revised["work"]
    assert [item["work"]["elapsed_seconds"] for item in work_history("old-investigation")] == [
        12,
        15,
    ]
    assert not (tmp_path / "data" / "notes").exists()


def test_explicit_historical_inference_is_optional_and_does_not_rewrite_sources(
    tmp_path, monkeypatch, sample_detail
):
    from pathlib import Path

    from ssgrep.search import detail
    from ssgrep.services.knowledge import work_for
    from tests.services.test_api import _claude_evidence, _evidence_detail

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    episode = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    source = Path(episode.source_path)
    before = source.read_bytes()
    monkeypatch.setattr(detail, "show", lambda ref: episode)
    assert work_for(episode) is None
    assert not (tmp_path / "data").exists()
    enriched = mcp_server.save_note(ref=episode.episode_id, infer_work=True)
    assert enriched["ok"]
    work = enriched["work"]["work"]
    assert work["basis"] == "transcript_inferred"
    assert work["model"] == "legacy-model" and "provider" not in work
    assert work["elapsed_seconds"] == 9
    assert work["usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert source.read_bytes() == before
    assert work_for(episode) == enriched["work"]


def test_work_metadata_does_not_use_the_show_character_budget_or_change_legacy_output(
    tmp_path, monkeypatch, sample_detail
):
    from ssgrep.sessions.work import attach_work, subject_digest

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(mcp_server.api, "show", lambda ref: sample_detail)
    old = mcp_server.show_session(sample_detail.episode_id, max_chars=5)
    assert "work" not in old and old["prompt_text"] == sample_detail.prompt_text[:5]
    attach_work(
        sample_detail.episode_id,
        subject_digest(sample_detail),
        {"work_id": "source-cost", "provider": "provider-a", "model": "model-a"},
    )
    enriched = mcp_server.show_session(sample_detail.episode_id, max_chars=5)
    assert enriched.pop("work")["work"]["model"] == "model-a"
    assert enriched == old


def test_index_failure_after_save_keeps_note_and_work_and_names_existing_ref(tmp_path, monkeypatch):
    from pathlib import Path

    from ssgrep.services import knowledge
    from ssgrep.sessions.work import work_history

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))

    def fail():
        raise OSError("isolated indexing failure")

    monkeypatch.setattr(knowledge, "_index_saved_note", fail)
    result = mcp_server.save_note(
        title="durable",
        body="preserve this",
        work={"work_id": "durable-cost", "elapsed_seconds": 9},
    )
    assert not result["ok"] and result["saved"] and not result["indexed"]
    assert result["condition"] == "index_failed" and result["ref"].endswith(":ep:0")
    assert "preserve this" in Path(result["note_file"]).read_text()
    assert work_history("durable-cost")[0]["work"]["elapsed_seconds"] == 9
    assert "not another save_note" in result["message"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"title": "x", "body": "y", "infer_work": True},
        {"ref": "some:ep:0"},
        {"ref": "some:ep:0", "work": {"work_id": "x"}, "infer_work": True},
        {"ref": "some:ep:0", "title": "replacement", "work": {"work_id": "x"}},
    ],
)
def test_invalid_save_modes_leave_knowledge_untouched(tmp_path, monkeypatch, arguments):
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    result = mcp_server.save_note(**arguments)
    assert result["condition"] == "invalid_work" and not result["ok"]
    assert not (tmp_path / "data").exists()


def test_saved_work_binds_to_the_indexed_note_text_not_markup_or_whitespace(
    tmp_path, monkeypatch, sample_detail
):
    from pathlib import Path

    from ssgrep.pipeline.episodes import extract_episode_text
    from ssgrep.services import knowledge
    from ssgrep.sessions.adapters.base import read_jsonl
    from ssgrep.sessions.episodes import segment_episode_groups

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(knowledge, "_index_saved_note", lambda: None)
    saved = knowledge.save_note(
        title="  <command-name>/note</command-name> Real question \n",
        body="  Answer with a trailing newline.\n",
        work={"work_id": "whitespace", "elapsed_seconds": 45},
    )
    shard = Path(saved["note_file"])
    [(episode, group)] = segment_episode_groups(list(read_jsonl(shard).records), shard.stem)
    prompt, response = extract_episode_text(group)
    indexed = replace(
        sample_detail,
        episode_id=episode.episode_id,
        runtime="ssgrep",
        prompt_text=prompt or episode.prompt_text,
        response_text=response or episode.response_text,
    )
    assert knowledge.work_for(indexed) == saved["work"]
    assert saved["work"]["work"]["elapsed_seconds"] == 45


def test_save_waits_for_startup_and_missing_index_never_creates_work(
    tmp_path, monkeypatch, blocked_reconcile
):
    from ssgrep.search import detail

    waiting = mcp_server.save_note(title="question", body="answer")
    assert waiting["condition"] == "index_updating" and not waiting["ok"]
    assert not (tmp_path / "data").exists()
    monkeypatch.setattr(mcp_server, "_reconcile_thread", None)

    def missing(ref):
        raise IndexNotFoundError("No index.")

    monkeypatch.setattr(detail, "show", missing)
    absent = mcp_server.save_note(ref="old:ep:0", work={"work_id": "old"})
    assert absent["condition"] == "missing_index" and not absent["ok"]
    assert not (tmp_path / "data").exists()


def test_save_refuses_unsafe_note_directory_then_recovers(tmp_path, monkeypatch):
    from pathlib import Path

    from ssgrep.services import knowledge

    root, other = tmp_path / "data", tmp_path / "unrelated"
    root.mkdir()
    other.mkdir()
    (root / "notes").symlink_to(other, target_is_directory=True)
    failed = mcp_server.save_note(title="question", body="answer")
    assert failed["condition"] == "storage_unavailable" and not failed["ok"]
    assert list(other.iterdir()) == []
    (root / "notes").unlink()
    monkeypatch.setattr(knowledge, "_index_saved_note", lambda: None)
    restored = mcp_server.save_note(title="question", body="answer")
    assert restored["saved"] and restored["indexed"]
    assert "answer" in Path(restored["note_file"]).read_text()


def test_inference_without_recorded_evidence_does_not_fabricate_work(
    tmp_path, monkeypatch, sample_detail
):
    from ssgrep.search import detail

    monkeypatch.setattr(detail, "show", lambda ref: replace(sample_detail, source_path=None))
    result = mcp_server.save_note(ref=sample_detail.episode_id, infer_work=True)
    assert result["condition"] == "work_evidence_unavailable" and not result["ok"]
    assert not (tmp_path / "data" / "work-metadata").exists()


def test_mcp_missing_note_fields_is_a_validation_error(tmp_path):
    result = mcp_server.save_note(title="missing body")
    assert result["condition"] == "invalid_work" and not result["ok"]
    assert not (tmp_path / "data").exists()
