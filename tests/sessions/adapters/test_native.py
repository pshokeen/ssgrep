"""Tests for the native JSONL adapter that federates existing roots."""

from __future__ import annotations

from pathlib import Path

from ssgrep.sessions import discovery as discovery_module
from ssgrep.sessions.adapters.base import ReadResult
from ssgrep.sessions.adapters.native import NativeAdapter
from ssgrep.utilities.types import SessionFile


def _session(tmp_path: Path, name: str, runtime: str) -> SessionFile:
    path = tmp_path / f"{name}.jsonl"
    path.write_bytes(b'{"type":"user"}\n{"type":"assistant"}\n')
    return SessionFile(path=path, session_id=name, is_main=True, runtime=runtime)


def test_discover_combines_corpus_notes_and_external(tmp_path: Path, monkeypatch) -> None:
    claude = _session(tmp_path, "claude", "claude")
    ssgrep = _session(tmp_path, "note", "ssgrep")
    native = _session(tmp_path, "external", "native")
    monkeypatch.setattr(
        discovery_module,
        "discover_sessions",
        lambda *, scope, no_subagents: [claude] if scope is None else [],
    )
    from ssgrep.sessions import notes as notes_module

    monkeypatch.setattr(notes_module, "discover_notes", lambda: [ssgrep])
    from ssgrep.sessions import discovery_roots as roots_module

    monkeypatch.setattr(roots_module, "discover_external", lambda: [native])

    sources = NativeAdapter().discover()

    by_runtime = {source.session.runtime: source for source in sources}
    assert set(by_runtime) == {"claude", "ssgrep", "native"}
    assert by_runtime["claude"].cache_cwds is True
    assert by_runtime["ssgrep"].cache_cwds is False


def test_discover_filters_scope_and_drops_unreadable(tmp_path: Path) -> None:
    missing = SessionFile(
        path=tmp_path / "missing.jsonl",
        session_id="missing",
        is_main=True,
    )
    from ssgrep.sessions import discovery as discovery_module

    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(
        discovery_module,
        "discover_sessions",
        lambda *, scope, no_subagents: [missing] if scope is None else [],
    )
    try:
        sources = NativeAdapter().discover(scope="/elsewhere")
        assert sources == []
    finally:
        monkeypatch.undo()


def test_read_delegates_to_jsonl_parser(tmp_path: Path) -> None:
    from ssgrep.sessions.adapters.base import jsonl_source

    session = _session(tmp_path, "claude", "claude")
    source = jsonl_source(session, adapter="native", cache_cwds=True)
    assert source is not None
    result = NativeAdapter().read(source)
    assert isinstance(result, ReadResult)
    assert len(result.records) == 2
