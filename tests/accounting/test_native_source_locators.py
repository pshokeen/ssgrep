"""Existing index metadata and scoped OpenCode sources need no prior digest pin."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ssgrep.accounting.native_source import (
    NativeSourceAdapter,
    NativeSourceReference,
    NativeSourceStatus,
)
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.scope_events import Scoper
from tests.accounting.readers_fixtures import (
    opencode_add_message,
    opencode_add_part,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
)
from tests.accounting.reduce_builders import policy
from tests.accounting.test_native_source import NOW, REF, old_source


def digest_tree(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_existing_index_native_locator_resolves_old_source_read_only(tmp_path: Path) -> None:
    from ssgrep.store import LanceStore

    path, _, adapter = old_source(tmp_path)
    repo = LanceStore()
    repo.table("episodes", create=True)
    repo.table("metadata", create=True)
    repo.upsert(
        "episodes",
        {
            "episode_id": REF,
            "session_id": "src1",
            "project": "project",
            "title": "old",
            "source_path": str(path),
            "cwd": "/work/proj",
            "runtime": "claude",
        },
    )
    repo.set_meta("index_state", "ready")
    repo.set_meta("last_index_time", "frozen")
    before = digest_tree(repo.path)
    adapter.references = {}
    adapter.repository = repo
    result = adapter.resolve(REF)
    assert result.status is NativeSourceStatus.OK
    assert result.line.key == ("claude", "src1")
    assert adapter.observed_references[REF].record_digests
    assert digest_tree(repo.path) == before
    repo.close()


def test_opencode_reads_only_referenced_session_and_preserves_database(tmp_path: Path) -> None:
    database = tmp_path / "opencode.db"
    connection = opencode_connect(database)
    for session in ("old", "unrelated"):
        connection.execute(
            "INSERT INTO session VALUES (?,?,?,?,?,?,?)",
            (session, "p", None, "/work/proj", session, 1, 1),
        )
        opencode_add_message(
            connection, "message-" + session, session, opencode_assistant_data(), at=1
        )
        opencode_add_part(
            connection,
            "part-" + session,
            "message-" + session,
            session,
            opencode_tool_part(
                "witness-" + session,
                "bash",
                {"command": "cat log"},
                output="old witness",
                start=1,
                end=2,
            ),
            at=2,
        )
    connection.close()
    before = digest_tree(tmp_path)
    ref = "opencode:old:ep:0"
    adapter = NativeSourceAdapter(
        {ref: NativeSourceReference(ref, "opencode", "old", database)},
        {"opencode": OpenCodeReader()},
        (tmp_path,),
        Scoper(policy()),
        NOW,
    )
    result = adapter.resolve(ref)
    assert result.status is NativeSourceStatus.OK
    assert result.line is not None
    assert result.line.key == ("opencode", "old")
    assert all(record.ev.session_id == "old" for record in result.line.records)
    assert digest_tree(tmp_path) == before
    assert adapter.bytes_read <= adapter.byte_budget


def test_index_namespaced_session_maps_back_to_native_identity(tmp_path: Path) -> None:
    from ssgrep.accounting.native_source_index import index_reference
    from ssgrep.store import LanceStore

    repo = LanceStore()
    repo.table("episodes", create=True)
    repo.table("metadata", create=True)
    repo.set_meta("index_state", "ready")
    repo.set_meta("last_index_time", "frozen")
    repo.upsert(
        "episodes",
        {
            "episode_id": "omp:native-id:ep:0",
            "session_id": "omp:native-id",
            "project": "p",
            "title": "source",
            "source_path": str(tmp_path / "old.jsonl"),
            "runtime": "omp",
            "cwd": "/work/proj",
        },
    )
    ref, status = index_reference("omp:native-id:ep:0", repo)
    assert status is NativeSourceStatus.OK and ref.session_id == "native-id"
    repo.close()


def test_claude_index_filename_alias_is_not_falsely_asserted_native_session(tmp_path: Path) -> None:
    from ssgrep.store import LanceStore

    path, _, adapter = old_source(tmp_path)
    alias = path.with_name("resumed-copy.jsonl")
    alias.write_bytes(path.read_bytes())
    repo = LanceStore()
    repo.table("episodes", create=True)
    repo.table("metadata", create=True)
    ref = "resumed-copy:ep:0"
    repo.upsert(
        "episodes",
        {
            "episode_id": ref,
            "session_id": "resumed-copy",
            "project": "p",
            "title": "old",
            "source_path": str(alias),
            "cwd": "/work/proj",
            "runtime": "claude",
        },
    )
    repo.set_meta("index_state", "ready")
    repo.set_meta("last_index_time", "frozen")
    adapter.references = {}
    adapter.repository = repo
    result = adapter.resolve(ref)
    assert result.status is NativeSourceStatus.OK and result.line.key == ("claude", "src1")
    observed = adapter.observed_references[ref]
    assert observed.session_id == "src1" and observed.indexed_session_id == "resumed-copy"
    repo.close()


def test_verified_opencode_row_set_survives_append_and_rejects_clock_drift(tmp_path: Path) -> None:
    import json
    import sqlite3

    database = tmp_path / "opencode.db"
    connection = opencode_connect(database)
    connection.execute(
        "INSERT INTO session VALUES (?,?,?,?,?,?,?)", ("old", "p", None, "/work/proj", "old", 1, 1)
    )
    opencode_add_message(connection, "old-message", "old", opencode_assistant_data(), at=1)
    opencode_add_part(
        connection,
        "old-part",
        "old-message",
        "old",
        opencode_tool_part(
            "old-call", "bash", {"command": "cat log"}, output="old witness", start=1, end=2
        ),
        at=2,
    )
    connection.close()
    ref = "opencode:old:ep:0"
    adapter = NativeSourceAdapter(
        {ref: NativeSourceReference(ref, "opencode", "old", database)},
        {"opencode": OpenCodeReader()},
        (tmp_path,),
        Scoper(policy()),
        NOW,
    )
    original = adapter.resolve(ref)
    assert original.status is NativeSourceStatus.OK
    with sqlite3.connect(database) as connection:
        opencode_add_message(connection, "new-message", "old", opencode_assistant_data(), at=NOW)
        opencode_add_part(
            connection,
            "new-part",
            "new-message",
            "old",
            opencode_tool_part(
                "new-call",
                "bash",
                {"command": "ssgrep search future"},
                output="future",
                start=NOW,
                end=NOW + 1,
            ),
            at=NOW,
        )
        connection.execute("UPDATE session SET time_updated = ? WHERE id = 'old'", (NOW,))
    appended = adapter.resolve(ref)
    assert appended.status is NativeSourceStatus.OK
    assert appended.source_digest == original.source_digest
    assert appended.line is not None and original.line is not None
    assert [r.identity for r in appended.line.records] == [
        r.identity for r in original.line.records
    ]
    with sqlite3.connect(database) as connection:
        data = json.loads(
            connection.execute("SELECT data FROM part WHERE id = 'old-part'").fetchone()[0]
        )
        data["state"]["time"]["end"] = 999
        connection.execute("UPDATE part SET data = ? WHERE id = 'old-part'", (json.dumps(data),))
    assert adapter.resolve(ref).status is NativeSourceStatus.CHANGED


def test_native_opencode_user_fact_is_not_discovery_credit(tmp_path: Path) -> None:
    from ssgrep.accounting.evidence_tokens import Token
    from ssgrep.accounting.sources import Consumer, Resolver
    from ssgrep.accounting.timeline import build_timelines
    from tests.accounting.test_reduce_evidence import FACT, PATH

    database = tmp_path / "opencode.db"
    connection = opencode_connect(database)
    connection.execute(
        "INSERT INTO session VALUES (?,?,?,?,?,?,?)", ("old", "p", None, "/work/proj", "old", 1, 1)
    )
    opencode_add_message(
        connection, "user-message", "old", {"role": "user", "time": {"created": 5}}, at=5
    )
    opencode_add_part(
        connection, "user-part", "user-message", "old", {"type": "text", "text": FACT}, at=5
    )
    opencode_add_message(connection, "assistant-message", "old", opencode_assistant_data(), at=10)
    opencode_add_part(
        connection,
        "witness-part",
        "assistant-message",
        "old",
        opencode_tool_part(
            "witness", "bash", {"command": "cat log"}, output=FACT, start=10, end=20
        ),
        at=20,
    )
    connection.close()
    ref = "opencode:old:ep:0"
    adapter = NativeSourceAdapter(
        {ref: NativeSourceReference(ref, "opencode", "old", database)},
        {"opencode": OpenCodeReader()},
        (tmp_path,),
        Scoper(policy()),
        NOW,
    )
    resolved = Resolver(build_timelines([]), adapter.scoper, native_sources=adapter).resolve(
        Token("path", PATH),
        (ref,),
        Consumer("claude:consumer", ("claude", "consumer"), 1, NOW),
    )
    assert resolved.source_user_supplied
    assert not resolved.immediate
    assert "source_user_supplied" in {reason for reason, _ in resolved.reasons}


@pytest.fixture
def owned_locator(tmp_path, monkeypatch):
    """A real isolated index pointing at an owned readable native transcript."""
    from dataclasses import replace

    from ssgrep.store import LanceStore

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "index"))
    path, _, adapter = old_source(tmp_path / "native")
    repo = LanceStore()
    repo.table("episodes", create=True)
    repo.table("metadata", create=True)
    repo.upsert(
        "episodes",
        {
            "episode_id": REF,
            "session_id": "src1",
            "project": "project",
            "title": "owned native witness",
            "source_path": str(path),
            "cwd": "/work/proj",
            "runtime": "claude",
        },
    )
    repo.set_meta("index_state", "ready")
    repo.set_meta("last_index_time", "owned-generation")
    adapter.references = {}
    adapter.repository = repo
    control = replace(adapter, observed_references={}, _snapshots={})
    resolved = control.resolve(REF)
    assert resolved.status is NativeSourceStatus.OK
    assert resolved.line is not None and resolved.line.key == ("claude", "src1")
    try:
        yield repo, path, adapter
    finally:
        repo.close()


@pytest.mark.parametrize(
    "case,status",
    [
        ("not_ready", NativeSourceStatus.UNRESOLVED),
        ("duplicate", NativeSourceStatus.UNRESOLVED),
        ("oversized", NativeSourceStatus.BOUNDED),
        ("expired", NativeSourceStatus.EXPIRED),
        ("purged", NativeSourceStatus.EXPIRED),
        ("absent", NativeSourceStatus.MISSING),
        ("missing", NativeSourceStatus.MISSING),
        ("quarantined", NativeSourceStatus.UNAPPROVED),
    ],
)
def test_index_authority_rejects_without_reading_native_bytes(owned_locator, case, status):
    from ssgrep.accounting.native_source_index import MAX_LOCATOR_BYTES

    repo, path, adapter = owned_locator
    if case == "not_ready":
        repo.set_meta("index_state", "building")
    elif case == "duplicate":
        repo.table("episodes").add(repo.rows("episodes"))
        assert repo.count("episodes") == 2
    elif case == "oversized":
        repo.update("episodes", f"episode_id = '{REF}'", {"cwd": "/" + "x" * MAX_LOCATOR_BYTES})
    else:
        repo.update("episodes", f"episode_id = '{REF}'", {"source_status": case})
    native_before = path.read_bytes()
    index_before = digest_tree(repo.path)
    result = adapter.resolve(REF)
    assert result.status is status
    assert result.line is None and result.source_digest == "" and not result.record_digests
    assert result.bytes_read == adapter.bytes_read == 0
    assert not adapter.observed_references
    assert path.read_bytes() == native_before
    assert digest_tree(repo.path) == index_before
    assert str(path) not in result.reason and "investigate" not in result.reason


def test_absent_owned_index_is_unknown_and_does_not_create_storage(tmp_path, monkeypatch):
    from ssgrep.store import LanceStore

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "absent-index"))
    path, _, adapter = old_source(tmp_path / "native")
    adapter.references = {}
    adapter.repository = LanceStore()
    before = digest_tree(tmp_path)
    try:
        result = adapter.resolve(REF)
        assert result.status is NativeSourceStatus.UNRESOLVED
        assert result.line is None and result.bytes_read == adapter.bytes_read == 0
        assert not adapter.repository.exists()
        assert path.is_file() and digest_tree(tmp_path) == before
    finally:
        adapter.repository.close()


def test_index_generation_flip_discards_real_locator_before_native_read(owned_locator, monkeypatch):
    repo, path, adapter = owned_locator
    original_rows = repo.rows
    selected = []

    def advance_after_selection(name, **kwargs):
        rows = original_rows(name, **kwargs)
        if name == "episodes":
            selected.extend(rows)
            repo.set_meta("last_index_time", "next-owned-generation")
        return rows

    monkeypatch.setattr(repo, "rows", advance_after_selection)
    native_before = path.read_bytes()
    result = adapter.resolve(REF)
    assert len(selected) == 1 and selected[0]["source_path"] == str(path)
    assert result.status is NativeSourceStatus.CHANGED
    assert result.line is None and not adapter.observed_references
    assert result.bytes_read == adapter.bytes_read == 0
    assert path.read_bytes() == native_before


@pytest.fixture
def owned_opencode_record(tmp_path):
    from tests.accounting.test_native_source_review_edges import database_scene

    database, originals, adapter = database_scene(tmp_path)
    target = next(item for item in originals if item.kind == "tool_result")
    reference = NativeSourceReference("opencode:s:ep:0", "opencode", "s", database)
    writer = opencode_connect(database)
    # Keep a real committed WAL live throughout every descriptor-backed read.
    writer.execute("UPDATE session SET time_updated = time_updated + 1 WHERE id = 's'")
    owner_bytes = writer.execute(
        "SELECT length(CAST(id AS BLOB)) + length(CAST(project_id AS BLOB)) + "
        "length(CAST(directory AS BLOB)) FROM session WHERE id = 's'"
    ).fetchone()[0]
    part_bytes = writer.execute(
        "SELECT length(CAST(data AS BLOB)) FROM part WHERE id = 'p'"
    ).fetchone()[0]
    control = adapter.read_record(
        reference, target.native_identity, target.digest, locator=target.locator
    )
    assert control.status is NativeSourceStatus.OK and control.evidence == target
    assert control.bytes_read == owner_bytes + part_bytes
    assert Path(str(database) + "-wal").is_file()
    try:
        yield database, target, reference, adapter, writer, owner_bytes, part_bytes
    finally:
        writer.close()


@pytest.mark.parametrize("locator", ["", "part:", "unknown:p", "part:" + "x" * 513])
def test_opencode_invalid_locators_never_query_or_charge_rows(owned_opencode_record, locator):
    database, target, reference, adapter, _, _, _ = owned_opencode_record
    before = digest_tree(database.parent)
    charged = adapter.bytes_read
    result = adapter.read_record(reference, target.native_identity, target.digest, locator=locator)
    assert result.status is NativeSourceStatus.UNRESOLVED and result.evidence is None
    assert result.bytes_read == 0 and adapter.bytes_read == charged
    assert digest_tree(database.parent) == before


@pytest.mark.parametrize(
    "case,status",
    [
        ("foreign_session", NativeSourceStatus.IDENTITY_MISMATCH),
        ("missing_row", NativeSourceStatus.MISSING),
        ("wrong_identity", NativeSourceStatus.IDENTITY_MISMATCH),
        ("wrong_digest", NativeSourceStatus.CHANGED),
    ],
)
def test_opencode_direct_row_authority_and_missingness_are_distinct(
    owned_opencode_record, case, status
):
    database, target, reference, adapter, _, owner_bytes, part_bytes = owned_opencode_record
    locator, identity, digest = target.locator, target.native_identity, target.digest
    expected_bytes = owner_bytes + part_bytes
    if case == "foreign_session":
        locator, expected_bytes = "session:foreign", owner_bytes
    elif case == "missing_row":
        locator, expected_bytes = "part:missing", owner_bytes
    elif case == "wrong_identity":
        identity += "-different"
    else:
        digest = "0" * 64
    before = digest_tree(database.parent)
    charged = adapter.bytes_read
    result = adapter.read_record(reference, identity, digest, locator=locator)
    assert result.status is status and result.evidence is None
    assert result.bytes_read == expected_bytes
    assert adapter.bytes_read - charged == expected_bytes
    assert digest_tree(database.parent) == before
    assert str(database) not in result.reason and "answer" not in result.reason


def test_opencode_direct_row_budget_stops_before_materializing_payload(owned_opencode_record):
    from dataclasses import replace

    database, target, reference, adapter, _, owner_bytes, part_bytes = owned_opencode_record
    adapter = replace(adapter, bytes_read=0, byte_budget=owner_bytes + part_bytes - 1)
    before = digest_tree(database.parent)
    result = adapter.read_record(
        reference, target.native_identity, target.digest, locator=target.locator
    )
    assert result.status is NativeSourceStatus.BOUNDED and result.evidence is None
    assert result.bytes_read == adapter.bytes_read == owner_bytes
    assert adapter.bytes_read < adapter.byte_budget
    assert digest_tree(database.parent) == before


@pytest.mark.parametrize("case", ["row_count", "locator_id", "locator_count"])
def test_opencode_session_row_limits_preserve_unknown_and_owned_database(
    owned_opencode_record, case
):
    from dataclasses import replace

    from ssgrep.accounting.native_source import MAX_SOURCE_RECORDS

    database, _, reference, adapter, _, owner_bytes, _ = owned_opencode_record
    if case == "row_count":
        adapter.max_records = 1
    elif case == "locator_id":
        reference = replace(reference, row_locators=("part:" + "x" * 513,))
    else:
        reference = replace(reference, row_locators=("part:p",) * (MAX_SOURCE_RECORDS + 1))
    adapter.references = {reference.ref: reference}
    before = digest_tree(database.parent)
    charged = adapter.bytes_read
    result = adapter.resolve(reference.ref)
    assert result.status is NativeSourceStatus.BOUNDED and result.line is None
    assert result.bytes_read >= owner_bytes
    assert adapter.bytes_read - charged == result.bytes_read
    assert adapter.bytes_read <= adapter.byte_budget
    assert not adapter.observed_references
    assert digest_tree(database.parent) == before


def test_opencode_source_replacement_during_real_snapshot_is_not_accepted(
    owned_opencode_record, tmp_path, monkeypatch
):
    from tests.accounting.test_native_source_review_edges import database_scene

    database, target, reference, adapter, _, _, _ = owned_opencode_record
    replacement, _, _ = database_scene(tmp_path / "replacement")
    reader = adapter.readers["opencode"]
    original_row = reader._row
    replaced = False
    original_inode = database.stat().st_ino

    def replace_after_decode(source, table, row, held, result):
        nonlocal replaced
        original_row(source, table, row, held, result)
        if table == "part" and not replaced:
            replaced = True
            replacement.replace(database)

    monkeypatch.setattr(reader, "_row", replace_after_decode)
    charged = adapter.bytes_read
    result = adapter.read_record(
        reference, target.native_identity, target.digest, locator=target.locator
    )
    assert replaced and database.stat().st_ino != original_inode
    assert result.status is NativeSourceStatus.CHANGED and result.evidence is None
    assert result.bytes_read > 0 and adapter.bytes_read - charged == result.bytes_read
    assert str(database) not in result.reason and "answer" not in result.reason


def test_malformed_opencode_row_is_incomplete_without_private_parser_error(owned_opencode_record):
    database, target, reference, adapter, writer, owner_bytes, _ = owned_opencode_record
    private_payload = '{"private_customer_token":"not-for-error-output"'
    writer.execute("UPDATE part SET data = ? WHERE id = 'p'", (private_payload,))
    before = digest_tree(database.parent)
    charged = adapter.bytes_read
    result = adapter.read_record(
        reference, target.native_identity, target.digest, locator=target.locator
    )
    assert result.status is NativeSourceStatus.INCOMPLETE and result.evidence is None
    assert result.bytes_read == owner_bytes + len(private_payload.encode())
    assert adapter.bytes_read - charged == result.bytes_read
    assert "not-for-error-output" not in repr(result) and str(database) not in repr(result)
    assert digest_tree(database.parent) == before
