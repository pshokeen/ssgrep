"""Tests for index observability (status command).

Key invariants:
- status() succeeds even when no index exists
- status() does NOT load the embedding model (184ms cost)
- Monkeypatching embed._get_model to raise proves the guard
- Mutation test: make status touch the encoder and prove it goes red
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from ssgrep import embed, observability, staleness, store
from ssgrep.types import IndexStats


class TestStatusNoIndex:
    """Status when no index has been built yet."""

    def test_status_returns_index_exists_false_when_missing(self, tmp_path):
        """status() succeeds and reports index_exists=False when .ssgrep/ missing."""
        result = observability.status(tmp_path)

        assert isinstance(result, IndexStats)
        assert result.index_exists is False
        assert result.session_count == 0
        assert result.episode_count == 0
        assert result.chunk_count == 0
        assert result.index_size_bytes == 0
        assert result.last_index_time is None
        assert result.tombstoned_source_count == 0
        assert result.tombstoned_chunk_count == 0

    def test_status_uses_default_model_when_no_index(self, tmp_path):
        """When no index exists, status returns the default model_id and dimension."""
        result = observability.status(tmp_path)

        assert result.model_id == embed.MODEL_ID
        assert result.vector_dimension == embed.DIMENSION
        assert result.schema_version == store.SCHEMA_VERSION

    def test_status_no_model_load_without_index(self, tmp_path):
        """status() does not load the embedding model when index is absent.

        Monkeypatch _get_model to raise; if status() called it, this would fail.
        """

        def raise_on_model_load(*args, **kwargs):
            raise RuntimeError("Model loader was called but status should not call it")

        with patch("ssgrep.embed._get_model", side_effect=raise_on_model_load):
            # This must not raise
            result = observability.status(tmp_path)
            assert result.index_exists is False


class TestStatusWithIndex:
    """Status when index exists."""

    def test_status_reports_session_episode_chunk_counts(self, tmp_path):
        """status() reports accurate session, episode, and chunk counts."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        # Build a minimal index
        conn = store.init_db(db_path)
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "session-1",
                "/path/to/session-1.jsonl",
                1,
                None,
                None,
                None,
                None,
                None,
                None,
                "available",
            ),
        )
        conn.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ep-1",
                "session-1",
                "Test Episode",
                None,
                None,
                None,
                "",
                "",
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                "available",
            ),
        )
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("chunk-1", "ep-1", "session-1", "Test text", "prompt", 0, "available"),
        )
        conn.commit()
        conn.close()

        result = observability.status(tmp_path)

        assert result.index_exists is True
        assert result.session_count == 1
        assert result.episode_count == 1
        assert result.chunk_count == 1

    def test_status_reports_index_size(self, tmp_path):
        """status() reports the on-disk size of index.db and vectors.f32."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        # Create an index
        conn = store.init_db(db_path)
        conn.close()

        # Create a vectors file with some content
        vec_path = index_dir / "vectors.f32"
        vec_path.write_bytes(b"\x00" * 1024)  # 1 KB of vectors

        result = observability.status(tmp_path)

        # Index size should include both files
        assert result.index_size_bytes > 0
        # Verify it includes the db and vectors file sizes
        db_size = db_path.stat().st_size
        vec_size = vec_path.stat().st_size
        assert result.index_size_bytes == db_size + vec_size

    def test_status_reports_last_index_time(self, tmp_path):
        """status() reports the timestamp of the last successful index run."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        now = datetime.now(UTC)
        store.set_meta(conn, "last_index_time", now.isoformat())
        conn.commit()
        conn.close()

        result = observability.status(tmp_path)

        assert result.last_index_time is not None
        # Compare without microseconds due to potential precision loss
        assert abs((result.last_index_time - now).total_seconds()) < 1

    def test_status_reports_model_binding(self, tmp_path):
        """status() reports the embedded model_id and dimension."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        store.set_meta(conn, "model_id", "minishlab/potion-base-8M")
        store.set_meta(conn, "vector_dimension", "256")
        conn.commit()
        conn.close()

        result = observability.status(tmp_path)

        assert result.model_id == "minishlab/potion-base-8M"
        assert result.vector_dimension == 256

    def test_status_reports_parsing_degradation(self, tmp_path):
        """status() reports counts of skipped and malformed records."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        store.set_meta(conn, "skipped_records", "10")
        store.set_meta(conn, "malformed_records", "3")
        conn.commit()
        conn.close()

        result = observability.status(tmp_path)

        assert result.skipped_records == 10
        assert result.malformed_records == 3

    def test_status_reports_tombstone_counts(self, tmp_path):
        """status() reports counts of tombstoned sources and chunks."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)

        # Insert a session and mark it as absent
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "session-1",
                "/path/to/session-1.jsonl",
                1,
                None,
                None,
                None,
                None,
                None,
                None,
                "absent",
            ),
        )
        conn.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ep-1",
                "session-1",
                "Test",
                None,
                None,
                None,
                "",
                "",
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                "absent",
            ),
        )
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("chunk-1", "ep-1", "session-1", "text", "prompt", 0, "absent"),
        )
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("chunk-2", "ep-1", "session-1", "text2", "response", 1, "absent"),
        )

        conn.commit()
        conn.close()

        result = observability.status(tmp_path)

        assert result.tombstoned_source_count == 1
        assert result.tombstoned_chunk_count == 2

    def test_status_excludes_tombstoned_from_available_counts(self, tmp_path):
        """status() excludes tombstoned sources from session/episode/chunk counts."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)

        # Insert available content
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "session-1",
                "/path/1.jsonl",
                1,
                None,
                None,
                None,
                None,
                None,
                None,
                "available",
            ),
        )
        conn.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ep-1",
                "session-1",
                "Test",
                None,
                None,
                None,
                "",
                "",
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                "available",
            ),
        )
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("chunk-1", "ep-1", "session-1", "text", "prompt", 0, "available"),
        )

        # Insert tombstoned content
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("session-2", "/path/2.jsonl", 1, None, None, None, None, None, None, "absent"),
        )
        conn.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ep-2",
                "session-2",
                "Test2",
                None,
                None,
                None,
                "",
                "",
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                "absent",
            ),
        )
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("chunk-2", "ep-2", "session-2", "text2", "response", 1, "absent"),
        )

        conn.commit()
        conn.close()

        result = observability.status(tmp_path)

        # Only available counts should be reported
        assert result.session_count == 1
        assert result.episode_count == 1
        assert result.chunk_count == 1
        # Tombstoned counts are separate
        assert result.tombstoned_source_count == 1
        assert result.tombstoned_chunk_count == 1


class TestStatusNoModelLoad:
    """Core requirement: status() never loads the embedding model."""

    def test_status_does_not_load_model_with_existing_index(self, tmp_path):
        """status() with an existing index does not load the embedding model.

        This is the critical guard: 184ms cost avoided on every status call.
        """
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        # Create a real index
        conn = store.init_db(db_path)
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "session-1",
                "/path/1.jsonl",
                1,
                None,
                None,
                None,
                None,
                None,
                None,
                "available",
            ),
        )
        conn.commit()
        conn.close()

        def raise_on_model_load(*args, **kwargs):
            raise RuntimeError("Model loader was called but status should not call it")

        with patch("ssgrep.embed._get_model", side_effect=raise_on_model_load):
            # Must not raise
            result = observability.status(tmp_path)
            assert result.index_exists is True

    def test_status_does_not_call_encode(self, tmp_path):
        """status() does not call embed.encode() at all."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        conn.close()

        call_count = {"encode": 0}

        original_encode = embed.encode

        def counting_encode(*args, **kwargs):
            call_count["encode"] += 1
            return original_encode(*args, **kwargs)

        with patch("ssgrep.embed.encode", side_effect=counting_encode):
            result = observability.status(tmp_path)
            # Positive control: the call genuinely ran and produced a real
            # status for the index built above. Without this, a bypassed
            # patch (or a status() that crashed before ever reaching the
            # encode path) would make the == 0 spy pass vacuously.
            assert result.index_exists is True
            assert result.schema_version == store.SCHEMA_VERSION
            # encode should never be called
            assert call_count["encode"] == 0


class TestStatusSchemaAndDrift:
    """Drift detection: model id or schema version mismatch."""

    def test_status_reports_schema_version(self, tmp_path):
        """status() reports the schema version from the index."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        # Schema version is set by init_db
        conn.close()

        result = observability.status(tmp_path)

        assert result.schema_version == store.SCHEMA_VERSION

    def test_status_reports_stored_schema_version(self, tmp_path):
        """status() returns the schema_version stored in meta, not the code version."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        # Override the schema version in meta
        store.set_meta(conn, "schema_version", "2")
        conn.commit()
        conn.close()

        result = observability.status(tmp_path)

        assert result.schema_version == 2


class TestStatusMutationGuard:
    """Mutation testing: prove the guard against model loading works."""

    def test_mutation_touching_encoder_fails_guard(self, tmp_path, monkeypatch):
        """If status() is mutated to call encode(), this test catches it.

        This is the mutation test for the main guard: we prove that if
        observability.py is changed to call embed.encode(), the test fails.

        To validate: temporarily add a call to embed.encode() in observability.status()
        and verify this test goes red.
        """
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        conn.close()

        # Monkeypatch _get_model to raise immediately
        def raise_on_model_load(*args, **kwargs):
            raise RuntimeError("status() called embed._get_model, which loads the model")

        monkeypatch.setattr("ssgrep.embed._get_model", raise_on_model_load)

        # If status() or anything it calls touches the model encoder,
        # this will raise
        result = observability.status(tmp_path)

        # Verify we got a result, not an exception
        assert result.index_exists is True

    def test_mutation_guard_catches_encode_call(self, tmp_path, monkeypatch):
        """If status() is mutated to call encode(), this test catches it."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        conn.close()

        # Make encode() raise, so any call is caught
        def raise_on_encode(*args, **kwargs):
            raise RuntimeError("status() called embed.encode(), which loads the model")

        monkeypatch.setattr("ssgrep.embed.encode", raise_on_encode)

        # Must not raise
        result = observability.status(tmp_path)
        assert result.index_exists is True


class TestStatusEdgeCases:
    """Edge cases and error handling."""

    def test_status_handles_missing_metadata_gracefully(self, tmp_path):
        """status() returns defaults for missing metadata fields."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        # Don't set any extra metadata beyond schema_version
        conn.close()

        result = observability.status(tmp_path)

        # Should use defaults
        assert result.model_id == embed.MODEL_ID
        assert result.vector_dimension == embed.DIMENSION
        assert result.skipped_records == 0
        assert result.malformed_records == 0

    def test_status_handles_invalid_datetime(self, tmp_path):
        """status() handles invalid last_index_time format gracefully."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        store.set_meta(conn, "last_index_time", "not a valid datetime")
        conn.commit()
        conn.close()

        result = observability.status(tmp_path)

        # Should fall back to None
        assert result.last_index_time is None

    def test_status_empty_index(self, tmp_path):
        """status() reports correct counts for an empty but valid index."""
        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        # Create an empty index (no sessions, episodes, or chunks)
        conn = store.init_db(db_path)
        conn.close()

        result = observability.status(tmp_path)

        assert result.index_exists is True
        assert result.session_count == 0
        assert result.episode_count == 0
        assert result.chunk_count == 0


class TestStatusStalenessWiring:
    """Verify staleness is detected and reported in IndexStats.

    Tests construct distinguishable worlds (fresh vs. stale) and assert
    exact staleness values. Mutation tests prove the wiring is load-bearing.
    """

    def test_status_fresh_index_returns_stale_false_count_zero(self, tmp_path, monkeypatch):
        """Fresh index: stale=False, stale_count=0.

        All discovered files match their cursors.
        """
        from ssgrep import discovery

        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        conn.close()

        def mock_discover(project_dir, scope=None):
            return []

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        result = observability.status(tmp_path)

        assert result.stale is False
        assert result.stale_count == 0

    def test_status_stale_index_returns_exact_count(self, tmp_path, monkeypatch):
        """Stale index: stale=True with stale_count == exact number of changed files.

        Build index, track 3 files, report them as appended. Assert count is 3.
        """
        from ssgrep import discovery

        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        file_a = tmp_path / "session-a.jsonl"
        file_b = tmp_path / "session-b.jsonl"
        file_c = tmp_path / "session-c.jsonl"
        file_a.write_text("x" * 100)
        file_b.write_text("y" * 100)
        file_c.write_text("z" * 100)

        for path in [file_a, file_b, file_c]:
            cursor = store.FileCursor(
                path=path, size=100, mtime=1000.0, byte_offset=0, first_line_hash="old"
            )
            store.upsert_session_file(conn, cursor)
        conn.commit()
        conn.close()

        def mock_discover(project_dir, scope=None):
            return [
                discovery.SessionFile(
                    path=file_a, session_id="s-a", is_main=True, size=200, mtime=2000.0
                ),
                discovery.SessionFile(
                    path=file_b, session_id="s-b", is_main=True, size=200, mtime=2000.0
                ),
                discovery.SessionFile(
                    path=file_c, session_id="s-c", is_main=True, size=200, mtime=2000.0
                ),
            ]

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        result = observability.status(tmp_path)

        assert result.stale is True
        assert result.stale_count == 3

    def test_status_does_not_load_model_during_staleness_check(self, tmp_path, monkeypatch):
        """status() never loads embedding model, even when checking staleness.

        Staleness detection is discovery + file stat, not model-based.
        """
        from ssgrep import discovery

        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        file_a = tmp_path / "session-a.jsonl"
        file_a.write_text("x" * 100)

        cursor = store.FileCursor(
            path=file_a, size=100, mtime=1000.0, byte_offset=0, first_line_hash="old"
        )
        store.upsert_session_file(conn, cursor)
        conn.commit()
        conn.close()

        # Report file as appended so staleness path runs
        def mock_discover(project_dir, scope=None):
            return [
                discovery.SessionFile(
                    path=file_a, session_id="s-a", is_main=True, size=200, mtime=2000.0
                ),
            ]

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        def raise_on_model_load(*args, **kwargs):
            raise RuntimeError("Model loader was called but status should not call it")

        with patch("ssgrep.embed._get_model", side_effect=raise_on_model_load):
            # Must not raise even though staleness is detected
            result = observability.status(tmp_path)
            assert result.stale is True
            assert result.stale_count == 1

    def test_status_mutation_staleness_call_removed_produces_stale_false_count_zero(
        self, tmp_path, monkeypatch
    ):
        """Mutation test: removing staleness call leaves stale=False, count=0.

        Proves the wiring is load-bearing by showing before/after differ.
        """
        from ssgrep import discovery, search

        index_dir = tmp_path / ".ssgrep"
        db_path = index_dir / "index.db"

        conn = store.init_db(db_path)
        file_a = tmp_path / "session-a.jsonl"
        file_a.write_text("x" * 100)

        cursor = store.FileCursor(
            path=file_a, size=100, mtime=1000.0, byte_offset=0, first_line_hash="old"
        )
        store.upsert_session_file(conn, cursor)
        conn.commit()
        conn.close()

        def mock_discover(project_dir, scope=None):
            return [
                discovery.SessionFile(
                    path=file_a, session_id="s-a", is_main=True, size=200, mtime=2000.0
                ),
            ]

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        # Baseline: staleness detected
        result_before = observability.status(tmp_path)
        assert result_before.stale is True
        assert result_before.stale_count == 1

        # Simulate mutation: replace staleness_summary with one that always returns unchanged
        def mock_staleness_summary_unchanged(project_dir, discovered_files=None):
            return staleness.StalenessReport(
                unchanged_count=1,
                appended_count=0,
                rewritten_count=0,
                vanished_count=0,
                stale_files=[],
            )

        monkeypatch.setattr(search, "staleness_summary", mock_staleness_summary_unchanged)

        # Mutated: staleness not detected
        result_after = observability.status(tmp_path)
        assert result_after.stale is False
        assert result_after.stale_count == 0

        # Verify before/after are different (proves the call matters)
        assert result_before.stale != result_after.stale
        assert result_before.stale_count != result_after.stale_count


class TestCwdCacheDegradation:
    """Tests for cwd-cache health detection in status()."""

    def test_status_cwd_cache_degraded_false_on_clean_run(self, tmp_path_project_indexed):
        """On a healthy run with warm cache, cwd_cache_degraded is False."""
        # Warm run: cache is populated and all files are in it
        result = observability.status(tmp_path_project_indexed)

        assert result.cwd_cache_degraded is False
        assert result.cwd_cache_fallback_scans == 0

    def test_status_cwd_cache_degraded_true_on_broken_cache(
        self, tmp_path_project_indexed, monkeypatch
    ):
        """When cache fallback scans occur, cwd_cache_degraded is True."""
        from ssgrep import discovery, search

        # Simulate fallback scans by incrementing the counter directly.
        # In a real scenario, this would happen when _file_matches_scope finds
        # a file not in the cache and must scan it inline.
        def trigger_fallback_scans():
            discovery._cwd_index_stats["fallback_scans"] += 5  # Simulate 5 fallback scans

        # Patch staleness_summary to trigger fallback scans as a side effect
        original_staleness_summary = search.staleness_summary

        def patched_staleness_summary(*args, **kwargs):
            trigger_fallback_scans()
            return original_staleness_summary(*args, **kwargs)

        monkeypatch.setattr(search, "staleness_summary", patched_staleness_summary)
        discovery._cwd_index_stats["fallback_scans"] = 0

        result = observability.status(tmp_path_project_indexed)

        assert result.cwd_cache_degraded is True
        # The 5 fallback scans are triggered by the patched staleness_summary side effect,
        # which simulates the real scenario where discover_sessions finds files not in cache.
        assert result.cwd_cache_fallback_scans == 5

    def test_status_cwd_cache_fields_default_when_no_index(self, tmp_path):
        """cwd_cache_* fields default to healthy values when no index exists."""
        result = observability.status(tmp_path)

        # Real assertions: verify default healthy state (no proxy assertions)
        assert result.cwd_cache_degraded is False
        assert result.cwd_cache_fallback_scans == 0


class TestStatusCommandRendering:
    """Tests that verify status command actually renders cache health to output."""

    def test_status_renders_cwd_cache_line_when_degraded(self, monkeypatch, capsys):
        """PROPERTY TEST: StatusCommand renders 'Cwd cache:' line when degraded=True.

        Instantiates StatusCommand, monkeypatches api.status to return degraded stats,
        calls handle(), and captures stdout. Asserts the degradation line is actually
        printed with the fallback scan count. Making this code unreachable (return None
        before it) causes this test to fail.
        """
        from ssgrep.cli.commands import status as status_cmd
        from ssgrep.cli.commands.status import StatusCommand
        from tests.conftest import build_index_stats

        stats = build_index_stats(
            index_exists=True,
            cwd_cache_degraded=True,
            cwd_cache_fallback_scans=7,
        )
        monkeypatch.setattr(status_cmd.api, "status", lambda project: stats)
        monkeypatch.setattr(status_cmd, "is_json_mode", lambda: False)

        from unittest.mock import MagicMock

        cmd = StatusCommand(MagicMock())
        cmd.handle(project_dir=".")

        out = capsys.readouterr().out

        # PROPERTY: Degraded status must render the line with the count
        assert (
            "Cwd cache:" in out
        ), f"StatusCommand.handle() should print 'Cwd cache:' when degraded. output: {out}"
        assert (
            "7" in out
        ), f"StatusCommand.handle() should include fallback count (7). output: {out}"

    def test_status_omits_cwd_cache_line_when_healthy(self, monkeypatch, capsys):
        """PROPERTY TEST: StatusCommand omits cache line when degraded=False.

        Verifies the inverse: healthy status does not print the cache line,
        keeping normal output clean.
        """
        from ssgrep.cli.commands import status as status_cmd
        from ssgrep.cli.commands.status import StatusCommand
        from tests.conftest import build_index_stats

        stats = build_index_stats(
            index_exists=True,
            cwd_cache_degraded=False,
            cwd_cache_fallback_scans=0,
        )
        monkeypatch.setattr(status_cmd.api, "status", lambda project: stats)
        monkeypatch.setattr(status_cmd, "is_json_mode", lambda: False)

        from unittest.mock import MagicMock

        cmd = StatusCommand(MagicMock())
        cmd.handle(project_dir=".")

        out = capsys.readouterr().out

        # PROPERTY: Healthy status must NOT render the cache line
        assert (
            "Cwd cache:" not in out
        ), f"StatusCommand.handle() should NOT print cache line when healthy. output: {out}"
        # Sanity check: normal output is still there
        assert (
            "Sessions:" in out
        ), f"StatusCommand.handle() should still print normal lines when healthy. output: {out}"
