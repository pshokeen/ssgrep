"""In-process capture behavior (tasks 3.4-3.6): policy, fail-open, MCP/note boundaries."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting import capture, capture_policy, fingerprint, journal, journal_reader, policy
from ssgrep.accounting.contracts import Operation, Outcome, Phase, Transport
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.services import api, mcp_server
from ssgrep.utilities.types import (
    EmptyQueryError,
    EpisodeDetail,
    IndexNotFoundError,
    ResultCard,
    SearchResponse,
)

PROFILE = "personal-local"


@pytest.fixture(autouse=True)
def _reset_capture_state():
    journal.close_all()
    fingerprint.forget_keys()
    capture._unsafe.clear()
    capture_policy._cache = ("", None, capture_policy._DISABLED)
    yield
    journal.close_all()
    capture._unsafe.clear()


def _card(index: int = 0, **overrides: object) -> ResultCard:
    fields: dict = {
        "ref": f"claude:s{index}:ep:{index}",
        "title": "t",
        "timestamp": None,
        "score": 0.5,
        "excerpt": "x",
        "files_touched": (),
        "is_subagent": False,
    }
    fields.update(overrides)
    return ResultCard(**fields)


def _detail(**overrides: object) -> EpisodeDetail:
    fields: dict = {
        "episode_id": "e1",
        "session_id": "s1",
        "title": "t",
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "git_branch": None,
        "cwd": "/w",
        "prompt_text": "p" * 50,
        "response_text": "r" * 50,
        "files_touched": (),
        "tool_names": (),
        "is_subagent": False,
    }
    fields.update(overrides)
    return EpisodeDetail(**fields)


def _read():
    return journal_reader.read_journal(PROFILE, quarantine=False)


def _sequence(read) -> list[tuple[str, str | None]]:
    return [(str(e.phase), None if e.outcome is None else str(e.outcome)) for e in read.events]


def test_disabled_by_default_writes_nothing_and_returns_results_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = SearchResponse(results=[_card()], total_matches=1)
    monkeypatch.setattr(api.search_module, "search", lambda q, **k: response)
    assert api.search("q") is response
    assert not (tmp_path / "data" / "accounting").exists()


def test_enabled_search_records_started_completed_with_original_object_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    response = SearchResponse(
        results=[_card(0, score=0.91), _card(1, score=0.5)], total_matches=2, omitted_count=1
    )
    monkeypatch.setattr(api.search_module, "search", lambda q, **k: response)

    assert api.search("q", limit=2, token_budget=700) is response

    read = _read()
    started, completed = read.events
    assert (started.phase, completed.phase) == (Phase.STARTED, Phase.COMPLETED)
    assert completed.outcome == Outcome.OK and completed.refs == (
        "claude:s0:ep:0",
        "claude:s1:ep:1",
    )
    assert completed.ranks_scores == ((1, 910), (2, 500))
    assert (completed.limit, completed.token_budget, completed.omitted_count) == (2, 700, 1)
    assert started.execution_id == completed.execution_id
    assert started.outcome is None and started.refs == ()  # nothing known at entry
    assert str(started.transport) == "api"  # no transport declared by a direct api caller


def test_service_error_is_recorded_and_the_original_exception_object_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    original = EmptyQueryError("Query must not be empty.")

    def boom(q: str, **k: object) -> SearchResponse:
        raise original

    monkeypatch.setattr(api.search_module, "search", boom)
    with pytest.raises(EmptyQueryError) as caught:
        api.search("  ")
    assert caught.value is original
    assert _sequence(_read()) == [("started", None), ("completed", "error")]


def test_when_search_and_append_both_fail_caller_sees_only_the_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    original = IndexNotFoundError("No global index found.")

    def boom(q: str, **k: object) -> SearchResponse:
        raise original

    def broken_append(*_a: object, **_k: object) -> bool:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(api.search_module, "search", boom)
    monkeypatch.setattr(journal, "append", broken_append)
    with pytest.raises(IndexNotFoundError) as caught:
        api.search("q")
    assert caught.value is original and caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_append_failure_never_changes_a_successful_result(monkeypatch: pytest.MonkeyPatch) -> None:
    policy.enable_collection(PROFILE)
    response = SearchResponse(results=[_card()], total_matches=1)
    monkeypatch.setattr(api.search_module, "search", lambda q, **k: response)
    monkeypatch.setattr(journal, "append", lambda *a, **k: (_ for _ in ()).throw(OSError(5, "io")))
    assert api.search("q") is response


def test_keyboard_interrupt_is_recorded_interrupted_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)

    def interrupted(q: str, **k: object) -> SearchResponse:
        raise KeyboardInterrupt

    monkeypatch.setattr(api.search_module, "search", interrupted)
    with pytest.raises(KeyboardInterrupt):
        api.search("q")
    assert _sequence(_read()) == [("started", None), ("completed", "interrupted")]


def test_show_outcomes_ok_empty_error(monkeypatch: pytest.MonkeyPatch) -> None:
    policy.enable_collection(PROFILE)
    monkeypatch.setattr(
        api.detail, "show", lambda ref: _detail(response_truncated=True) if ref == "hit" else None
    )
    api.show("hit")
    assert api.show("nope") is None
    monkeypatch.setattr(api.detail, "show", lambda ref: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        api.show("hit")

    completed = [e for e in _read().events if e.phase is Phase.COMPLETED]
    by_outcome = {str(e.outcome): e for e in completed}
    assert set(by_outcome) == {"ok", "empty", "error"}
    assert by_outcome["ok"].refs == ("hit",) and by_outcome["ok"].truncated is True
    assert str(by_outcome["empty"].operation) == "show"


def test_empty_search_is_outcome_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    policy.enable_collection(PROFILE)
    monkeypatch.setattr(
        api.search_module, "search", lambda q, **k: SearchResponse(results=[], total_matches=0)
    )
    api.search("q")
    assert _sequence(_read()) == [("started", None), ("completed", "empty")]


def test_excess_refs_are_bounded_with_an_explicit_marker_and_intact_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    cards = [_card(i, ref="claude:" + "s" * 700 + f":ep:{i}") for i in range(200)]
    monkeypatch.setattr(
        api.search_module,
        "search",
        lambda q, **k: SearchResponse(results=cards, total_matches=200),
    )
    api.search("q", limit=50)
    segment = next(
        (Path(os.environ["SSGREP_DATA_DIR"]) / "accounting" / PROFILE / "journal").iterdir()
    )
    assert max(len(line) + 1 for line in segment.read_bytes().splitlines()) <= 64 * 1024
    completed = [e for e in _read().events if e.phase is Phase.COMPLETED][0]
    assert completed.refs_omitted is True and 0 < len(completed.refs) < 200
    assert completed.outcome == Outcome.OK and completed.omitted_count == 0


def test_mcp_search_records_rendered_with_unavailable_bytes_and_unchanged_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    response = SearchResponse(
        results=[_card()], total_matches=1, excerpts_truncated=True, omitted_count=3
    )
    monkeypatch.setattr(api.search_module, "search", lambda q, **k: response)

    payload = mcp_server.search_sessions("q")

    assert payload["excerpts_truncated"] is True and payload["omitted_count"] == 3
    read = _read()
    rendered = [e for e in read.events if e.phase is Phase.RENDERED]
    assert len(rendered) == 1
    (event,) = rendered
    assert str(event.transport) == "mcp" and event.truncated is True and event.omitted_count == 3
    assert event.rendered_bytes is not None and event.rendered_bytes.value is None
    assert event.rendered_bytes.reason == "mcp_framework_owns_serialization"


def test_mcp_show_render_describes_the_wrapper_truncation_not_the_api_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    monkeypatch.setattr(api.detail, "show", lambda ref: _detail(prompt_text="p" * 100))

    result = mcp_server.show_session("ref", max_chars=10)

    assert result["truncated"] is True and len(result["prompt_text"]) == 10
    read = _read()
    completed = next(e for e in read.events if e.phase is Phase.COMPLETED)
    rendered = next(e for e in read.events if e.phase is Phase.RENDERED)
    assert completed.truncated is False  # the API result itself was not truncated
    assert rendered.truncated is True  # the MCP wrapper truncated the response afterwards


def test_mcp_error_return_records_error_completion_and_no_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    monkeypatch.setattr(
        api.search_module, "search", lambda q, **k: (_ for _ in ()).throw(EmptyQueryError("e"))
    )
    assert mcp_server.search_sessions("") == {"error": "Query must not be empty."}
    assert _sequence(_read()) == [("started", None), ("completed", "error")]


def test_transport_label_is_consumed_by_exactly_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    policy.enable_collection(PROFILE)
    monkeypatch.setattr(
        api.search_module,
        "search",
        lambda q, **k: SearchResponse(results=[_card()], total_matches=1),
    )
    mcp_server.search_sessions("q")
    api.search("direct")  # a later non-transport caller must not inherit "mcp"
    transports = [str(e.transport) for e in _read().events if e.phase is Phase.STARTED]
    assert sorted(transports) == ["api", "mcp"]


def test_each_event_uses_its_own_cwd_for_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ws" / "team"
    (root / "repo").mkdir(parents=True)
    other = tmp_path / "ws" / "team-other"
    other.mkdir()
    policy.enable_collection(PROFILE)
    policy.save_profile(
        ProfilePolicy(profile_id=PROFILE, roots=(str(root),), collection_enabled=True)
    )
    monkeypatch.setattr(
        api.search_module, "search", lambda q, **k: SearchResponse(results=[], total_matches=0)
    )

    monkeypatch.chdir(root / "repo")
    api.search("inside")
    monkeypatch.chdir(other)
    api.search("outside: a lexical-prefix sibling")
    monkeypatch.chdir(root)
    api.search("inside again")  # same process, scope changed back: captured again

    started = [e for e in _read().events if e.phase is Phase.STARTED]
    assert [e.cwd for e in started] == [os.path.realpath(root / "repo"), os.path.realpath(root)]


def test_policy_changes_apply_to_the_next_event_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api.search_module, "search", lambda q, **k: SearchResponse(results=[], total_matches=0)
    )
    api.search("before enabling")  # no policy: nothing
    policy.enable_collection(PROFILE)
    api.search("enabled")
    policy.pause(PROFILE)
    api.search("paused")
    policy.resume(PROFILE)
    api.search("resumed")
    assert len([e for e in _read().events if e.phase is Phase.STARTED]) == 2


def test_corrupt_or_unsafe_policy_means_capture_off_never_permissive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy.enable_collection(PROFILE)
    path = policy.policy_path()
    monkeypatch.setattr(
        api.search_module, "search", lambda q, **k: SearchResponse(results=[], total_matches=0)
    )
    path.write_text("{not json")
    api.search("corrupt")
    assert not (tmp_path / "data" / "accounting" / PROFILE / "journal").exists()


def test_org_profile_workspace_never_falls_back_to_personal_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    org_root = tmp_path / "org-repo"
    org_root.mkdir()
    policy.enable_collection(PROFILE)
    policy.save_profile(ProfilePolicy(profile_id="org-a", roots=(str(org_root),)))
    monkeypatch.setattr(
        api.search_module, "search", lambda q, **k: SearchResponse(results=[], total_matches=0)
    )
    monkeypatch.chdir(org_root)
    api.search("inside an org workspace whose profile has not enabled collection")
    assert not (tmp_path / "data" / "accounting" / PROFILE / "journal").exists()
    assert not (tmp_path / "data" / "accounting" / "org-a" / "journal").exists()


def test_unsafe_journal_directory_disables_capture_for_the_process_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy.enable_collection(PROFILE)
    victim = tmp_path / "victim"
    victim.mkdir()
    profile_dir = tmp_path / "data" / "accounting" / PROFILE
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "journal").symlink_to(victim)
    response = SearchResponse(results=[_card()], total_matches=1)
    monkeypatch.setattr(api.search_module, "search", lambda q, **k: response)

    assert api.search("q1") is response and api.search("q2") is response
    assert list(victim.iterdir()) == []
    assert PROFILE in capture._unsafe


def test_correlation_id_is_recorded_only_when_well_formed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    monkeypatch.setattr(
        api.search_module, "search", lambda q, **k: SearchResponse(results=[], total_matches=0)
    )
    monkeypatch.setenv(capture.CORRELATION_ENV, "run-42:call.7")
    api.search("a")
    monkeypatch.setenv(capture.CORRELATION_ENV, "bad value; rm -rf /")
    api.search("b")
    ids = [e.correlation_id for e in _read().events if e.phase is Phase.STARTED]
    assert sorted(ids, key=str) == [None, "run-42:call.7"]


def test_note_boundary_records_one_started_completed_pair_without_duplicate_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.cli.commands import note_command
    from ssgrep.sessions import notes

    policy.enable_collection(PROFILE)
    monkeypatch.setattr(note_command.api, "index", lambda: replace_stats())
    project = tmp_path / "proj"
    project.mkdir()
    note_command_instance = object.__new__(note_command.NoteCommand)
    monkeypatch.setattr(note_command, "is_json_mode", lambda: True)

    note_command_instance.handle("How did we fix X?", "the body", str(project))

    read = _read()
    assert [(str(e.phase), str(e.operation), str(e.transport)) for e in read.events] == [
        ("started", "note", "cli"),
        ("completed", "note", "cli"),
    ]
    assert read.events[1].outcome == Outcome.OK and read.events[0].cwd == os.path.realpath(project)
    shard = next(notes.notes_dir().iterdir())
    assert shard.read_text().count("the body") == 1  # the note was written exactly once


def replace_stats():
    from ssgrep.utilities.types import IndexStats

    return IndexStats(
        session_count=1,
        episode_count=1,
        chunk_count=1,
        index_size_bytes=1,
        last_index_time=None,
        model_id="m",
        vector_dimension=1,
        skipped_records=0,
        malformed_records=0,
        schema_version=1,
    )


def test_note_write_failure_is_recorded_as_error_and_reraised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.cli.commands import note_command

    policy.enable_collection(PROFILE)
    boom = OSError(30, "Read-only file system")
    monkeypatch.setattr(
        note_command.notes, "write_note", lambda *a, **k: (_ for _ in ()).throw(boom)
    )
    instance = object.__new__(note_command.NoteCommand)
    with pytest.raises(OSError) as caught:
        instance.handle("title", "body", str(tmp_path))
    assert caught.value is boom
    assert _sequence(_read()) == [("started", None), ("completed", "error")]


def test_note_is_not_captured_as_a_lookup_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.cli.commands import note_command

    policy.enable_collection(PROFILE)
    monkeypatch.setattr(note_command.api, "index", lambda: replace_stats())
    monkeypatch.setattr(note_command, "is_json_mode", lambda: True)
    object.__new__(note_command.NoteCommand).handle("t", "b", str(tmp_path))
    assert {str(e.operation) for e in _read().events} == {"note"}
    assert Operation.NOTE.value == "note" and Transport.CLI.value == "cli"
    assert replace(ProfilePolicy(), paused=True).paused is True
