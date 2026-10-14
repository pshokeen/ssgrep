"""Additive persisted shapes the UX layer reads: document schema stamps and session facts.

* ``receipts/<id>.json`` and every ``history/periods.json`` row carry ``schema``.
* The stored reduction document (``collector/state/reduction-*.json``) carries
  ``session_parents`` and ``session_cwd`` as plain facts; an older blob without them still
  loads (fields default empty). None of this is ledger content: the snapshot never mentions it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ssgrep.accounting import collector_publish as publish, collector_store as store, service
from ssgrep.accounting.collector_state import ReductionDoc
from ssgrep.accounting.contracts import SCHEMA_VERSION
from tests.accounting import collector_scene as scene
from tests.accounting.readers_fixtures import write_records
from tests.accounting.reduce_builders import (
    PROFILE,
    omp_header,
    omp_launch,
    omp_search,
    omp_user,
    results_json,
    ts,
)
from tests.accounting.snapshot_builders import enable_policy
from tests.accounting.test_collect_publish import CHARS, ledger

CWD = "/work/proj"


def test_receipt_file_stamps_the_receipts_own_schema_not_a_constant() -> None:
    record = ledger().rows().receipts[0]
    assert publish.receipt_document(record)["schema"] == record.receipt.schema == SCHEMA_VERSION
    future = replace(record, receipt=replace(record.receipt, schema=7))
    assert publish.receipt_document(future)["schema"] == 7


def test_stored_receipt_stays_readable_by_the_facade_with_the_stamp(tmp_path: Path) -> None:
    rows = ledger().rows()
    publish.write_receipts(PROFILE, rows)
    record = rows.receipts[0]
    stored = json.loads(
        (
            tmp_path / "data" / "accounting" / PROFILE / "receipts" / f"{record.receipt_id}.json"
        ).read_bytes()
    )
    assert stored["schema"] == SCHEMA_VERSION
    view = service.read_receipt(PROFILE, record.receipt_id)
    assert view is not None and view.receipt_id == record.receipt_id


def test_every_history_row_carries_the_schema_version() -> None:
    document = publish.history_document(ledger().rows(), CHARS, "2026-11-21")
    assert len(document) == 3
    assert [row["schema"] for row in document] == [SCHEMA_VERSION] * 3


def test_older_reduction_blob_without_the_new_fields_still_loads_empty() -> None:
    doc = ReductionDoc(
        coverage={"recorded_lookups": 2},
        session_roots=(("omp", "w", "omp:root"),),
        session_last=(("omp", "w", "2026-10-12T10:00:00.000Z"),),
    )
    older = doc.to_dict()
    del older["session_parents"], older["session_cwd"]  # what a pre-change collector wrote
    loaded = ReductionDoc.from_dict(older)
    assert loaded.session_parents == () and loaded.session_cwd == ()
    assert loaded.session_roots == doc.session_roots and loaded.coverage == doc.coverage


def test_new_fields_round_trip_exactly_including_a_null_parent() -> None:
    doc = ReductionDoc(
        session_parents=(("omp", "child", "root"), ("omp", "root", None)),
        session_cwd=(("omp", "child", CWD),),
    )
    again = ReductionDoc.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert again.session_parents == doc.session_parents
    assert again.session_cwd == doc.session_cwd


def omp_tree(root: Path) -> None:
    base = root / "omp-sessions" / "-work"
    parent = base / "root.jsonl"
    when = scene.at(hour=10)
    write_records(
        parent,
        [
            omp_header("root-1", ts(12, 8)),
            omp_user("e-a", "investigate the failing deploy", ts(12, 9)),
            *omp_launch("e-l1", "c-launch", "ChildAlpha", ts(12, 9, 10), ts(12, 9, 11)),
        ],
    )
    write_records(
        base / "root" / "ChildAlpha.jsonl",
        [
            omp_header("child-a", ts(12, 9, 12), parent=parent),
            *omp_search("c1", "call-1", "deploy", results_json(), (12, 10, 0, 0)),
        ],
    )
    import os

    for path in (parent, base / "root" / "ChildAlpha.jsonl"):
        os.utime(path, (when.timestamp(), when.timestamp()))


def reduction_doc() -> ReductionDoc:
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    raw = store.read_blob(PROFILE, checkpoint.blobs["reduction"])
    return ReductionDoc.from_dict(raw)


def test_a_real_cycle_persists_session_parents_and_cwd_and_keeps_them_out_of_the_snapshot(
    tmp_path: Path,
) -> None:
    enable_policy()
    scene.install_key()
    scene.journal_events(scene.start_marker())
    omp_tree(tmp_path)
    result = scene.collect(scene.at(hour=11))
    assert result.state == "published"
    doc = reduction_doc()
    assert dict(((rt, sid), parent) for rt, sid, parent in doc.session_parents) == {
        ("omp", "root-1"): None,
        ("omp", "child-a"): "root-1",
    }
    assert doc.session_cwd == (("omp", "child-a", CWD),)
    assert dict(((rt, sid), root) for rt, sid, root in doc.session_roots)[("omp", "child-a")] == (
        "omp:root-1"
    )
    raw = (scene.accounting_dir() / "snapshots" / "current.json").read_text()
    assert "session_parents" not in raw and "session_cwd" not in raw and "child-a" not in raw
    # a clock-only cycle republishes from the stored document and keeps the facts
    scene.collect(scene.at(hour=12))
    again = reduction_doc()
    assert again.session_parents == doc.session_parents and again.session_cwd == doc.session_cwd
    assert replace(again, session_last=()).session_parents == doc.session_parents


def test_older_blob_without_invocation_sessions_loads_empty_and_the_field_round_trips() -> None:
    older = ReductionDoc().to_dict()
    assert "invocation_sessions" in older
    del older["invocation_sessions"]  # what a pre-change collector wrote
    assert ReductionDoc.from_dict(older).invocation_sessions == ()
    doc = ReductionDoc(invocation_sessions=(("inv-a", "omp", "child-a"), ("inv-b", "claude", "s")))
    again = ReductionDoc.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert again.invocation_sessions == doc.invocation_sessions


def test_a_real_cycle_maps_every_invocation_id_to_its_session_sorted(tmp_path: Path) -> None:
    enable_policy()
    scene.install_key()
    scene.journal_events(scene.start_marker())
    omp_tree(tmp_path)
    assert scene.collect(scene.at(hour=11)).state == "published"
    doc = reduction_doc()
    assert doc.invocation_sessions == tuple(sorted(doc.invocation_sessions))
    assert doc.invocation_sessions and len(doc.invocation_sessions) == len(
        {inv for inv, _, _ in doc.invocation_sessions}
    )
    by_invocation = {inv: (rt, sid) for inv, rt, sid in doc.invocation_sessions}
    assert set(by_invocation.values()) == {("omp", "child-a")}
    ids = {e.invocation_id for e in doc.executions if e.invocation_id}
    assert ids <= set(by_invocation)  # every execution's invocation resolves to a session
    raw = (scene.accounting_dir() / "snapshots" / "current.json").read_text()
    assert "invocation_sessions" not in raw
    scene.collect(scene.at(hour=12))  # clock-only cycle keeps the facts
    assert reduction_doc().invocation_sessions == doc.invocation_sessions
