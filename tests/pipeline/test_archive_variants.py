"""Historical archive shapes measured in the actual recovery corpus."""

import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest

from ssgrep.pipeline import app, rows
from ssgrep.pipeline.archive import retained_rows
from ssgrep.pipeline.sources import (
    SourceDescriptor,
    read_registry,
    to_row,
    to_transcript_source,
)
from ssgrep.services import api
from ssgrep.store import LanceStore
from tests.pipeline.test_app import fake_models as fake_models, write_claude_session


def seed():
    path = write_claude_session("retained")
    api.index()
    repo = LanceStore()
    descriptor = read_registry(repo)[str(path)]
    path.unlink()
    return repo, descriptor


def test_session_only_archive_is_preserved(tmp_path):
    repo = LanceStore().initialize()
    path = str(tmp_path / "empty.jsonl")
    descriptor = SourceDescriptor(
        adapter="native",
        key=path,
        path=path,
        size=0,
        mtime=0,
        digest="",
        session_id="empty",
    )
    session = rows.build_session_row(session_id="empty", path=path, runtime="claude")
    repo.upsert("sources", to_row(descriptor))
    repo.upsert("sessions", session.model_dump())
    actual = retained_rows(to_transcript_source(descriptor))
    assert actual == [[session], [], []]
    repo.close()


def test_obsolete_alias_does_not_redeclare_canonical_rows():
    repo, descriptor = seed()
    obsolete = replace(descriptor, key=descriptor.key + ".old", path=descriptor.path + ".old")
    repo.upsert("sources", to_row(obsolete))
    assert retained_rows(to_transcript_source(obsolete)) == [[], [], []]
    canonical = retained_rows(to_transcript_source(descriptor))
    assert len(canonical[0]) == 1 and canonical[1] and canonical[2]
    repo.close()


def test_shared_session_partitions_legacy_chunks_without_rewriting():
    repo, descriptor = seed()
    alias = replace(descriptor, key=descriptor.key + ".old", path=descriptor.path + ".old")
    repo.upsert("sources", to_row(alias))
    # A historical source can retain additional chunk IDs beyond the current
    # episode's segmentation. Its text/vector must not be regenerated or lost.
    original = repo.rows("chunks", limit=1)[0]
    historical = {
        **original,
        "chunk_id": original["chunk_id"] + ":legacy",
        "source_path": alias.path,
        "text": "Historical retained text",
        "search_text": "Historical retained text",
    }
    repo.upsert("chunks", historical)
    archived = retained_rows(to_transcript_source(alias))
    assert archived[0] == [] and archived[1] == [] and len(archived[2]) == 1
    actual = archived[2][0]
    assert actual.chunk_id == historical["chunk_id"]
    assert actual.text == historical["text"]
    assert actual.vector.tolist() == historical["vector"]
    canonical = retained_rows(to_transcript_source(descriptor))
    assert all(row.source_path == descriptor.path for row in canonical[2])
    assert sum(map(len, [canonical[2], archived[2]])) == repo.count("chunks")
    repo.close()


def test_unregistered_archive_source_is_rejected():
    repo, descriptor = seed()
    alien = replace(descriptor, key="/unknown", path="/unknown")
    with pytest.raises(ValueError, match="incomplete archive"):
        retained_rows(to_transcript_source(alien))
    repo.close()


def test_unavailable_final_statistics_fail_closed():
    class Handle:
        async def result(self):
            pass

        def stats(self):
            return None

    handle = Handle()
    engine = Mock(update=Mock(return_value=handle))
    with pytest.raises(RuntimeError, match="statistics unavailable"):
        asyncio.run(app._drive_update(engine, total=0, full_reprocess=False, quiet=True))
    engine.update.assert_called_once_with(full_reprocess=False)
