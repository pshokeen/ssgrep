"""Independent row-count acceptance for bounded, fair SQLite/WAL collection."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from ssgrep.accounting.collector_inputs import ClassifyingLedger
from ssgrep.accounting.collector_read import read_sources
from ssgrep.accounting.readers.base import EvidenceLedger, SourceState
from ssgrep.accounting.readers.opencode import OpenCodeReader
from tests.accounting.readers_fixtures import (
    T0,
    opencode_add_message,
    opencode_add_part,
    opencode_add_session,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
)


def store(path: Path, parts: int = 1, pending: int = 0):
    writer = opencode_connect(path)
    writer.execute("BEGIN")
    opencode_add_session(writer, "owner")
    opencode_add_message(writer, "message", "owner", opencode_assistant_data())
    for index in range(parts):
        # Deterministic realistic skew: mostly small, occasional large output.
        output = "x" * (1048576 if index % 37 == 0 else 512)
        data = opencode_tool_part(
            f"call-{index}",
            "bash",
            {"command": "ssgrep search alpha"},
            status="running" if index < pending else "completed",
            output=output,
        )
        opencode_add_part(writer, f"part-{index:05}", "message", "owner", data, at=T0 + index)
    writer.execute("COMMIT")
    return writer


def collect(discovered, known, ledger, budget=1, baseline=False):
    return read_sources(
        discovered=discovered,
        known=known,
        ledger=ledger,
        admit=lambda _: None,
        byte_budget=budget,
        time_budget=100,
        clock=lambda: 0,
        baseline_new=baseline,
    )


def test_large_wal_store_eventually_visits_independently_counted_rows(tmp_path):
    path = tmp_path / "large.db"
    # Expected counts are construction inputs, not inferred from reader output.
    parts, pending = 1203, 503
    writer = store(path, parts, pending)
    reader = OpenCodeReader()
    source = reader.source_for(path)
    state, calls, results, pending_ids = None, set(), set(), set()
    try:
        assert Path(f"{path}-wal").stat().st_size > 0
        for _ in range(parts * 8 + 20):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = SourceState.from_dict(json.loads(json.dumps(step.state.to_dict())))
            calls.update(e.record_id for e in step.evidence if e.kind == "tool_call")
            results.update(e.record_id for e in step.evidence if e.kind == "tool_result")
            pending_ids.update(e.payload["id"] for e in step.evidence if e.kind == "pending_tool")
            if not step.more:
                break
        else:
            raise AssertionError("fresh rows starved behind unfinished rows")
        progress = state.extra["opencode"]
        assert all(
            progress["rows_visited"][table] >= n
            for table, n in {"session": 1, "message": 1, "part": parts}.items()
        )
        assert calls == {f"call:call-{i}" for i in range(pending, parts)}
        assert len(results) == parts - pending
        assert pending_ids == {f"part-{i:05}" for i in range(pending)}
        assert 1 <= step.pending_records <= 32
        assert len(state.extra["pending"]["part"]) <= 32
        assert progress["pending_records"] is None
        assert not progress["pending_count_complete"]
        assert len(json.dumps(state.to_dict())) < 65536
        assert progress["status"] == "pending"
        assert progress["bytes_read"] > 0
        # Complete the LAST formerly capped pending row without advancing its time.
        done = opencode_tool_part("call-502", "bash", {}, output="late completion")
        writer.execute("UPDATE part SET data=? WHERE id='part-00502'", (json.dumps(done),))
        found = False
        for _ in range(parts * 8 + 20):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            if "result:call-502" in step.evidence_ids():
                found = True
                break
        assert found, "bounded rechecks must reach every retained pending id"
        assert step.pending_records <= 32
    finally:
        writer.close()


def test_source_fairness_and_zero_budget_never_completes_unvisited_database(tmp_path):
    paths = [tmp_path / "first.db", tmp_path / "second.db"]
    writers = [store(p, parts=20) for p in paths]
    reader = OpenCodeReader()
    discovered = {str(i): (reader, reader.source_for(p)) for i, p in enumerate(paths)}
    ledger = ClassifyingLedger()
    try:
        zero = collect(discovered, {}, ledger, budget=0, baseline=True)
        assert all(not r.complete for r in zero.sources.values())
        assert all(
            r.state["extra"]["opencode"]["status"] == "unvisited" for r in zero.sources.values()
        )
        first = collect(discovered, zero.sources, ledger)
        second = collect(discovered, first.sources, ledger)
        assert all(r.state["extra"]["opencode"]["rows_visited"] for r in second.sources.values())
        assert all(not r.complete for r in second.sources.values())
    finally:
        for writer in writers:
            writer.close()


def test_failure_details_survive_a_successful_source(tmp_path):
    good, bad = tmp_path / "good.db", tmp_path / "bad.db"
    writer = store(good)
    bad.write_bytes(b"not SQLite")
    reader = OpenCodeReader()
    discovered = {
        "bad": (reader, reader.source_for(bad)),
        "good": (reader, reader.source_for(good)),
    }
    try:
        outcome = collect(discovered, {}, ClassifyingLedger(), budget=16 * 1024 * 1024)
        assert outcome.sources["good"].complete
        assert not outcome.sources["bad"].complete
        assert outcome.sources["bad"].gaps["database_unavailable"] == 1
        assert outcome.gaps["database_unavailable"] == 1
        assert outcome.sources["bad"].state["extra"]["opencode"]["status"] == "unavailable"
    finally:
        writer.close()


def test_deleted_pending_and_unknown_terminal_status_keep_provenance_without_success(tmp_path):
    path = tmp_path / "pending.db"
    writer = store(path, parts=3, pending=1)
    reader = OpenCodeReader()
    source = reader.source_for(path)
    try:
        future = opencode_tool_part("future", "bash", {}, status="cancelled")
        opencode_add_part(writer, "future", "message", "owner", future, at=T0 + 10)
        initial = reader.read(source)
        assert "result:future" not in initial.evidence_ids()
        assert initial.state.extra["unsupported"]["part:future"] == "unsupported_tool_status"
        writer.execute("DELETE FROM part WHERE id='part-00000'")
        after = reader.read(source, initial.state)
        abandoned = after.state.extra["abandoned"]["part:part-00000"]
        assert abandoned["call_id"] == "call-0"
        assert abandoned["session_id"] == "owner"
        assert abandoned["reason"] == "pending_row_missing"
        assert after.state.extra["opencode"]["abandoned_records"] == 1
        assert not any(e.record_id == "result:call-0" for e in after.evidence)
        assert after.state.extra["opencode"]["status"] != "complete"
    finally:
        writer.close()


def test_concurrent_append_is_deferred_until_next_committed_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "concurrent.db"
    writer = store(path)
    reader = OpenCodeReader()
    snapshot = reader.snapshot
    source = reader.source_for(path)

    @contextmanager
    def append_after_pin(database, *, descriptor=None, tracker=None):
        with snapshot(database, descriptor=descriptor, tracker=tracker) as connection:
            opencode_add_part(
                writer,
                "later",
                "message",
                "owner",
                opencode_tool_part("later", "bash", {}),
                at=T0 + 20,
            )
            yield connection

    try:
        with monkeypatch.context() as patch:
            patch.setattr(reader, "snapshot", append_after_pin)
            before = reader.read(source)
        assert "call:later" not in before.evidence_ids()
        assert before.state.extra["opencode"]["rows_visited"]["part"] == 1
        after = reader.read(source, before.state)
        assert "call:later" in after.evidence_ids()
        assert after.state.extra["opencode"]["rows_visited"]["part"] == 2
    finally:
        writer.close()


def test_backdated_committed_insert_behind_timestamp_keyset_is_eventually_visited(tmp_path):
    path = tmp_path / "backdated.db"
    writer = store(path)
    reader = OpenCodeReader()
    source = reader.source_for(path)
    try:
        initial = reader.read(source)
        opencode_add_part(
            writer,
            "earlier-id",
            "message",
            "owner",
            opencode_tool_part("backdated", "bash", {}),
            at=T0 - 100,
        )
        state = initial.state
        seen = set()
        for _ in range(40):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            seen.update(step.evidence_ids())
            if "result:backdated" in seen:
                break
        assert "call:backdated" in seen and "result:backdated" in seen
        assert state.extra["opencode"]["rows_visited"]["part"] >= 2
    finally:
        writer.close()


def test_terminal_observations_clear_pending_without_native_timestamp_changes(tmp_path):
    path = tmp_path / "lifecycle.db"
    writer = store(path, parts=35, pending=35)
    reader = OpenCodeReader()
    source = reader.source_for(path)
    ledger = EvidenceLedger()
    try:
        initial = reader.read(source, maintenance=True)
        for evidence in initial.evidence:
            ledger.admit(evidence)
        for index in range(35):
            writer.execute(
                "UPDATE part SET data=? WHERE id=?",
                (
                    json.dumps(opencode_tool_part(f"call-{index}", "bash", {}, output="done")),
                    f"part-{index:05}",
                ),
            )
        state, results = initial.state, set()
        for _ in range(500):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            for evidence in step.evidence:
                if ledger.admit(evidence) and evidence.kind == "tool_result":
                    results.add(evidence.record_id)
            if state.extra["opencode"]["status"] == "complete":
                break
        assert results == {f"result:call-{i}" for i in range(35)}
        assert state.extra["opencode"]["pending_records"] == 0
        assert ledger.conflicts() == []
    finally:
        writer.close()


def test_native_bytes_include_ignored_payload_not_database_size_or_emitted_evidence(tmp_path):
    path = tmp_path / "bytes.db"
    writer = store(path, parts=0)
    ignored = {"type": "text", "text": "private" * 65536}
    expected_payload_bytes = len(json.dumps(ignored).encode())
    opencode_add_part(writer, "ignored", "message", "owner", ignored, at=T0 + 1)
    reader = OpenCodeReader()
    try:
        step = reader.read(reader.source_for(path))
        assert not any(e.kind in {"tool_call", "tool_result"} for e in step.evidence)
        assert step.bytes_read >= expected_payload_bytes
        assert step.state.extra["opencode"]["bytes_read"] == step.bytes_read
        assert step.state.extra["opencode"]["rows_visited"]["part"] == 1
        assert step.state.offset == 0  # No fabricated SQLite file-byte cursor.
    finally:
        writer.close()


def test_database_baseline_is_unvisited_not_a_skipped_complete_high_watermark(tmp_path):
    from ssgrep.accounting.collector_baseline import baseline_record

    database = tmp_path / "baseline.db"
    writer = store(database)
    reader = OpenCodeReader()
    source = reader.source_for(database)
    try:
        baseline = baseline_record(reader, source)
        assert not baseline.complete
        assert baseline.state["extra"]["opencode"]["status"] == "unvisited"
        assert baseline.state["extra"]["opencode"]["rows_visited"] == {}
        assert "wm" not in baseline.state["extra"]
        assert baseline.offset == baseline.skipped_bytes == 0
        outcome = collect(
            {"native": (reader, source)},
            {"native": baseline},
            ClassifyingLedger(),
            budget=16 * 1024 * 1024,
        )
        assert outcome.sources["native"].complete
        (result,) = [item for item in outcome.evidence if item.kind == "tool_result"]
        assert result.payload["call_id"] == "call-0"
        assert result.payload["output"] == "x" * 1048576
    finally:
        writer.close()
