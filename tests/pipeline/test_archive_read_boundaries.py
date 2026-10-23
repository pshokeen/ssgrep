"""An inconsistent retained read must not redeclare or rewrite missing-source history."""

from __future__ import annotations

import pytest

from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline.archive import retained_rows
from ssgrep.pipeline.sources import to_transcript_source
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE, SOURCES_TABLE, LanceStore
from tests.pipeline.test_app import fake_models as fake_models
from tests.pipeline.test_archive_variants import seed

TABLES = (SESSIONS_TABLE, EPISODES_TABLE, CHUNKS_TABLE, SOURCES_TABLE)


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("truncated", "truncated rows"),
        ("episode-sequence", "episode sequence is incomplete"),
        ("foreign-source", "source identity mismatch"),
        ("episode-link", "chunk content or episode link mismatch"),
        ("content-type", "chunk content or episode link mismatch"),
        ("content", "chunk content or episode link mismatch"),
        ("empty-vector", "invalid retained vector"),
        ("vector-width", "invalid retained vector"),
        ("nonfinite-vector", "invalid retained vector"),
        ("proxy-width", "invalid retained proxy vector"),
        ("nonfinite-proxy", "invalid retained proxy vector"),
        ("row-schema", "stored row schema mismatch"),
    ],
)
def test_inconsistent_archive_read_fails_closed_and_keeps_retained_data(
    monkeypatch, damage: str, reason: str
) -> None:
    repo, descriptor = seed()
    source = to_transcript_source(descriptor)
    original = LanceStore.rows
    try:
        before = {table: original(repo, table, limit=repo.count(table)) for table in TABLES}
        valid = retained_rows(source)
        assert [row.episode_id for row in valid[1]] == ["retained:ep:0"]
        assert {row.chunk_id for row in valid[2]} == {
            row["chunk_id"] for row in before[CHUNKS_TABLE]
        }

        def damaged_read(store, table, *args, **kwargs):
            result = [dict(row) for row in original(store, table, *args, **kwargs)]
            if table == SESSIONS_TABLE:
                if damage == "truncated":
                    return result[:-1]
                if damage == "row-schema":
                    result[0]["obsolete_column"] = "must not be silently dropped"
            elif table == EPISODES_TABLE and damage == "episode-sequence":
                result[0]["episode_id"] = "retained:ep:99"
            elif table == CHUNKS_TABLE:
                row = result[0]
                if damage == "foreign-source":
                    row["source_path"] = "/unregistered-history.jsonl"
                elif damage == "episode-link":
                    row["episode_id"] = "retained:ep:99"
                elif damage == "content-type":
                    row["content_type"] = "unknown-role"
                elif damage == "content":
                    row["text"] = "Not the indexed episode text"
                elif damage == "empty-vector":
                    row["vector"] = []
                elif damage == "vector-width":
                    row["vector"] = [[0.0] * (DIMENSION - 1)]
                elif damage == "nonfinite-vector":
                    row["vector"] = [[float("nan")] * DIMENSION]
                elif damage == "proxy-width":
                    row["proxy_vector"] = [0.0] * (DIMENSION - 1)
                elif damage == "nonfinite-proxy":
                    row["proxy_vector"] = [float("inf")] * DIMENSION
            return result

        with monkeypatch.context() as fault:
            fault.setattr(LanceStore, "rows", damaged_read)
            with pytest.raises(ValueError, match=f"incomplete archive.*{reason}"):
                retained_rows(source)

        # The real tables remain authoritative; a transient invalid read cannot repair them.
        assert {table: original(repo, table, limit=repo.count(table)) for table in TABLES} == before
        recovered = retained_rows(source)
        assert [row.episode_id for row in recovered[1]] == ["retained:ep:0"]
        assert {row.chunk_id: row.vector.tolist() for row in recovered[2]} == {
            row["chunk_id"]: row["vector"] for row in before[CHUNKS_TABLE]
        }
    finally:
        repo.close()
