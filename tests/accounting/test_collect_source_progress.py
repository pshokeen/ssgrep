"""Stat-stable source reads and unique-source worker progress."""

from __future__ import annotations

import os

from ssgrep.accounting import collector_live
from ssgrep.accounting.collector_progress import source_progress
from ssgrep.accounting.collector_store import SourceRecord
from tests.accounting import collector_scene as scene
from tests.accounting.readers_fixtures import write_records
from tests.accounting.test_collect_read import Cycle, claude_dir, claude_records


def test_stable_native_source_is_not_parsed_again_with_positive_ingest(tmp_path, monkeypatch):
    write_records(claude_dir(tmp_path) / "stable.jsonl", claude_records("stable", 2))
    cycle = Cycle(tmp_path)
    first = cycle.run()
    assert first.evidence and first.bytes_read > 0
    calls = []
    original = cycle.readers["claude"].read

    def observed(*args, **kwargs):
        calls.append(args[0].path)
        return original(*args, **kwargs)

    monkeypatch.setattr(cycle.readers["claude"], "read", observed)
    idle = cycle.run()
    assert idle.bytes_read == 0 and idle.evidence == []
    assert all(record.complete for record in idle.sources.values())
    assert calls == []


def test_same_size_native_mutation_is_not_hidden_by_fast_skip(tmp_path, monkeypatch):
    path = claude_dir(tmp_path) / "stable.jsonl"
    write_records(path, claude_records("stable", 2))
    cycle = Cycle(tmp_path)
    cycle.run()
    original_stamp = path.stat().st_mtime_ns
    body = path.read_bytes().replace(b"echo 1", b"echo 9")
    path.write_bytes(body)
    os.utime(path, ns=(original_stamp, original_stamp))
    changed = cycle.run()
    assert changed.bytes_read > 0 and changed.resets > 0
    assert changed.dropped.get("conflict", 0) > 0


def test_unvisited_database_pending_count_is_unknown_not_zero():
    counts = source_progress([SourceRecord("opencode", "opencode:/fixture/db")])
    assert counts["sources_unvisited"] == 1
    assert counts["sources_complete"] == 0
    assert counts["pending_records"] is None
    assert not counts["pending_count_complete"]


def test_read_stage_counts_unique_discovered_sources_not_quantum_callbacks(tmp_path, monkeypatch):
    scene.install_key()
    scene.journal_events(scene.start_marker())
    scene.write_transcript(
        tmp_path, "left", scene.search_records("left", "call", "q", (10, 0, 0)), scene.at()
    )
    original = collector_live.read_sources

    def repeated_callback(**kwargs):
        outcome = original(**kwargs)
        for key, record in outcome.sources.items():
            kwargs["on_source"](key, record)
            kwargs["on_source"](key, record)
        return outcome

    monkeypatch.setattr(collector_live, "read_sources", repeated_callback)
    scene.collect(scene.at())
    # Inspect actual worker output, not the count of callback invocations.
    from ssgrep.accounting.collector_progress import read_progress

    progress = read_progress("personal-local")
    assert progress is not None
    assert progress["sources_done"] <= progress["sources_total"] == 1
    assert progress["sources_complete"] == 1
