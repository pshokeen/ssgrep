"""Tests for the shared adapter contracts and JSONL helpers."""

from __future__ import annotations

from pathlib import Path

from ssgrep.sessions.adapters.base import (
    ReadResult,
    SourceFingerprint,
    file_fingerprint,
    jsonl_source,
    read_jsonl,
)
from ssgrep.utilities.types import SessionFile


def test_file_fingerprint_reads_first_line_and_stat(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"first line\nsecond line\n")
    fingerprint = file_fingerprint(path)
    assert fingerprint is not None
    assert fingerprint.size == path.stat().st_size
    assert fingerprint.mtime == path.stat().st_mtime
    assert fingerprint.digest == __import__("hashlib").sha256(b"first line\n").hexdigest()


def test_file_fingerprint_returns_none_for_unreadable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    assert file_fingerprint(missing) is None


def test_jsonl_source_builds_descriptor_and_skips_missing(tmp_path: Path) -> None:
    session = SessionFile(
        path=tmp_path / "s.jsonl",
        session_id="s",
        is_main=True,
        project_paths=(str(tmp_path),),
    )
    assert jsonl_source(session, adapter="native", cache_cwds=True) is None
    session.path.write_bytes(b'{"type":"user"}\n')
    source = jsonl_source(session, adapter="native", cache_cwds=True)
    assert source is not None
    assert source.adapter == "native"
    assert source.key == str(session.path.absolute())
    assert source.session is session
    assert source.cache_cwds is True
    assert isinstance(source.fingerprint, SourceFingerprint)


def test_read_jsonl_counts_lines_and_tolerates_tail(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    lines = [
        b"\n",
        b'{"type":"user"}\n',
        b"{bad}\n",
        b"[]\n",
        b'{"type":"future"}\n',
        b"x" * 61 + b"\n",
        b'{"type":"assistant"}',
    ]
    path.write_bytes(b"".join(lines))
    monkeypatch = __import__("pytest").MonkeyPatch()
    from ssgrep.sessions import records as records_module

    monkeypatch.setattr(records_module, "MAX_LINE_BYTES", 60)
    try:
        result = read_jsonl(path)
    finally:
        monkeypatch.undo()
    assert result.records == ({"type": "user"}, {"type": "future"})
    assert result.malformed_records == 1
    assert result.skipped_records == 3
    assert isinstance(result, ReadResult)
