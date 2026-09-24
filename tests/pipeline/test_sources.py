"""Tests for source descriptors, the registry, and OpenCode normalization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from ssgrep.pipeline.sources import (
    SourceDescriptor,
    delete_registry,
    from_row,
    to_descriptor,
    to_row,
    to_transcript_source,
    union_descriptors,
)
from ssgrep.sessions.adapters.base import SourceFingerprint, TranscriptSource
from ssgrep.store import SOURCES_TABLE
from ssgrep.utilities.types import SessionFile


def make_source(
    tmp_path: Path,
    *,
    adapter: str = "claude",
    key: str = "k",
    runtime: str = "claude",
    path: Path | None = None,
    project_paths: tuple[str, ...] = ("/work/app",),
) -> TranscriptSource:
    path = path or tmp_path / "session.jsonl"
    session = SessionFile(
        path=path,
        session_id="session-1",
        is_main=True,
        project_paths=project_paths,
        source_project="project-1",
        runtime=runtime,
    )
    return TranscriptSource(
        adapter=adapter,
        key=key,
        session=session,
        fingerprint=SourceFingerprint(size=10, mtime=1.5, digest="abc"),
        cache_cwds=adapter == "claude",
    )


def test_descriptor_round_trip_via_registry_rows(tmp_path) -> None:
    source = make_source(tmp_path, key="a=1", project_paths=("/x", "/y"))
    descriptor = to_descriptor(source)
    restored = from_row(to_row(descriptor))
    assert restored == descriptor
    assert restored.project_paths == ("/x", "/y")
    assert restored.cache_cwds is True
    assert restored.digest == "abc"


def test_descriptor_to_transcript_source_round_trip(tmp_path) -> None:
    descriptor = to_descriptor(make_source(tmp_path, key="b", runtime="pi"))
    source = to_transcript_source(descriptor)
    assert source.adapter == descriptor.adapter
    assert source.key == descriptor.key
    assert source.fingerprint == SourceFingerprint(
        size=descriptor.size, mtime=descriptor.mtime, digest=descriptor.digest
    )
    assert source.session.path == Path(descriptor.path)
    assert source.session.runtime == "pi"


def test_union_descriptors_fresh_wins_and_registry_fills(tmp_path) -> None:
    registry = {
        "gone": SourceDescriptor(
            adapter="claude", key="gone", path="/g", size=1, mtime=1.0, digest="d"
        ),
        "shared": SourceDescriptor(
            adapter="claude", key="shared", path="/old", size=1, mtime=1.0, digest="d"
        ),
    }
    fresh = [make_source(tmp_path, key="shared", path=tmp_path / "new.jsonl")]
    merged = union_descriptors(fresh, registry)
    assert set(merged) == {"gone", "shared"}
    assert merged["gone"].path == "/g"  # preserved frozen
    assert merged["shared"].path == str(tmp_path / "new.jsonl")  # fresh wins


def test_opencode_sources_keep_per_session_fingerprints(tmp_path) -> None:
    database = tmp_path / "opencode.db"
    database.write_bytes(b"first-line\nrest")
    sources = [
        make_source(tmp_path, adapter="opencode", key="o1", runtime="opencode", path=database),
        make_source(tmp_path, adapter="opencode", key="o2", runtime="opencode", path=database),
        make_source(tmp_path, adapter="claude", key="c1"),
    ]
    for source in sources:
        descriptor = to_descriptor(source)
        assert descriptor.digest == source.fingerprint.digest
        assert descriptor.mtime == source.fingerprint.mtime
        assert descriptor.size == source.fingerprint.size
    # The shared database file's own fingerprint must not leak in: sessions
    # carry their per-session fingerprints so unchanged sessions memo-hit.
    assert to_descriptor(sources[0]).digest == "abc"


def test_delete_registry_removes_each_key() -> None:
    repository = Mock()
    delete_registry(repository, ["a", "b"])
    assert repository.delete.call_count == 2
    repository.delete.assert_any_call(SOURCES_TABLE, "key = 'a'")
    repository.delete.assert_any_call(SOURCES_TABLE, "key = 'b'")
