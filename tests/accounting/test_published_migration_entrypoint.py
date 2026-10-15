"""Strict current decoding must not precede the one-way published-format gate."""

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CollectError


def test_published_method_header_is_recognized_without_current_revaluation():
    from ssgrep.accounting.contracts import digest_of
    from ssgrep.accounting.published_legacy import validate_checkpoint

    method = {
        "label": "production-identifier-candidates-v1",
        "units": [],
        "thresholds": [["conservative", 8], ["central", 6], ["optimistic", 4]],
        "frequency_view": "freq-old",
        "tokenizer": "",
        "pricing": "",
        "calibrated": False,
    }
    namespace = digest_of(method)[:32]
    blob = store.write_blob(
        "personal-local",
        "ledger-" + namespace[:12],
        {
            "state": {"methodology": method, "candidates": [], "debits": [], "frozen": []},
            "superseded": {},
            "collection_scope": {},
        },
    )
    cp = store.Checkpoint(
        "personal-local", methodology=namespace, blobs={"ledger-" + namespace[:12]: blob}
    )
    assert validate_checkpoint("personal-local", cp)
    cp.methodology = "f" * 32
    cp.blobs = {"ledger-" + cp.methodology[:12]: blob}
    before = dict(cp.blobs)
    try:
        validate_checkpoint("personal-local", cp)
    except CollectError as error:
        assert error.condition == "state_corrupt"
    else:
        raise AssertionError("A foreign-method checkpoint cannot authorize migration or orphan GC")
    assert cp.blobs == before


def test_original_scope_aliases_move_into_the_same_opaque_archive(monkeypatch):
    from datetime import UTC, datetime

    from ssgrep.accounting import published_capture
    from ssgrep.accounting.identity_migration import ARCHIVE, archive_current

    profile = "personal-local"
    original = {"original_format": "opaque-original-quantities"}
    reference = store.write_blob(profile, "ledger-old", original)
    reduction = store.write_blob(
        profile,
        "reduction",
        {
            "session_roots": [["claude", "first", "claude:first"]],
            "session_cwd": [["claude", "first", "/workspace/alpha"]],
        },
    )
    pointers = {
        "ledger-old": reference,
        "ledger-old-scope-previous": reference,
        "reduction": reduction,
        "reduction-scope-previous": reduction,
    }
    checkpoint = store.Checkpoint(profile, methodology="old", blobs=dict(pointers))
    # This test isolates reference routing, not legacy format admission or
    # native proof. The private two-phase driver exercises those with real407.
    monkeypatch.setattr(
        published_capture,
        "captured_documents",
        lambda *_: {"format": "published-session-v1", "history_blob": None, "receipt_blobs": {}},
    )
    previous = archive_current(profile, checkpoint, datetime(2026, 10, 14, tzinfo=UTC), [])
    assert previous == pointers
    assert set(checkpoint.blobs) == {ARCHIVE}
    manifest = store.read_blob(profile, checkpoint.blobs[ARCHIVE])
    assert manifest is not None
    assert manifest["blobs"] == pointers
    assert manifest["root_mapping_status"] == "unavailable"
    assert store.read_blob(profile, reference) == original


def test_original_publication_capture_validates_before_copying_private_content(monkeypatch):
    from ssgrep.accounting.published_capture import _envelope

    original = b" [ ] \n"
    reference = _envelope("personal-local", original, "history", 3, "2026-10-14T12:00:00Z")
    archived = store.read_blob("personal-local", reference)
    assert archived is not None
    assert archived["original_utf8"].encode() == original
    writes = []
    monkeypatch.setattr(store, "write_blob", lambda *args: writes.append(args))
    try:
        _envelope(
            "personal-local",
            b'[{"prompt":"private source text"}]',
            "history",
            3,
            "2026-10-14T12:00:00Z",
        )
    except CollectError as error:
        assert error.condition == "migration_history_unavailable"
    else:
        raise AssertionError("Private content must not enter a new original-publication envelope")
    assert writes == []


def test_publication_capture_precedes_native_resolution_and_survives_retry(monkeypatch):
    from datetime import UTC, datetime

    from ssgrep.accounting import collector_read, published_capture
    from ssgrep.accounting.identity_migration import prepare
    from tests.accounting.reduce_builders import policy

    checkpoint = store.Checkpoint(
        "personal-local",
        generation=3,
        identity_migration={
            "target": "native_identity_v2",
            "status": "unavailable",
            "published_capture": "original-capture",
        },
    )
    calls = []

    def capture(profile, current, *, limit, byte_budget):
        calls.append((current.identity_migration.get("published_capture"), limit, byte_budget))
        return False, 17

    monkeypatch.setattr(published_capture, "stage", capture)
    monkeypatch.setattr(collector_read, "make_readers", lambda _: {})
    result = prepare(
        "personal-local",
        checkpoint,
        datetime(2026, 10, 14, tzinfo=UTC),
        policy=policy(),
        byte_budget=64,
        max_records=999,
    )
    assert calls == [("original-capture", 32, 64)]
    assert result.changed and not result.complete and result.bytes_read == 17
    assert result.warnings == ("identity_migration_pending",)
    assert checkpoint.identity_migration["published_capture"] == "original-capture"
    assert checkpoint.identity_migration["segment"] == 0
    assert checkpoint.identity_method != "native_identity_v2"


def test_pending_privacy_compaction_returns_partial_without_looping_into_native_work(monkeypatch):
    from datetime import UTC, datetime

    from ssgrep.accounting import evidence_migration
    from ssgrep.accounting.collector_cycle import CycleContext
    from ssgrep.accounting.collector_progress import Progress
    from ssgrep.accounting.collector_setup import prepare_evidence
    from ssgrep.accounting.collector_types import CollectOptions, CycleBudget

    checkpoint = store.Checkpoint("personal-local", generation=3)
    calls = []

    def quantum(profile, current, *, byte_budget):
        calls.append(byte_budget)
        current.retention["evidence_migration_pending"] = True
        return current

    monkeypatch.setattr(evidence_migration, "migrate_evidence", quantum)
    context = CycleContext(
        profile="personal-local",
        options=CollectOptions(budget=CycleBudget(seconds=1, bytes=17)),
        now=datetime(2026, 10, 14, tzinfo=UTC),
        progress=Progress("personal-local", "collect"),
    )
    saved, blocked = prepare_evidence(context, checkpoint)
    assert saved is checkpoint and calls == [17]
    assert blocked is not None
    assert blocked.result.state == "partial" and not blocked.report.complete
    assert blocked.report.stopped_by == "evidence_migration"
    assert blocked.result.warnings == ("evidence_migration_pending",)


def test_published_generation_three_recovers_staged_four_without_current_decode_or_orphan_gc(
    monkeypatch,
):
    import os

    from ssgrep.accounting import collector_steps, snapshots, storage
    from ssgrep.accounting.contracts import digest_of

    profile = "personal-local"
    method = {
        "label": "production-identifier-candidates-v1",
        "units": [],
        "thresholds": [["conservative", 8], ["central", 6], ["optimistic", 4]],
        "frequency_view": "freq-old",
        "tokenizer": "",
        "pricing": "",
        "calibrated": False,
    }
    namespace = digest_of(method)[:32]
    reference = store.write_blob(
        profile,
        "ledger-" + namespace[:12],
        {
            "state": {"methodology": method, "candidates": [], "debits": [], "frozen": []},
            "superseded": {},
            "collection_scope": {},
        },
    )
    checkpoint = store.Checkpoint(
        profile, generation=3, methodology=namespace, blobs={"ledger-" + namespace[:12]: reference}
    )

    def envelope(generation, methodology):
        return snapshots.build_envelope(
            generation=generation,
            counter_epoch=1,
            methodology_id=methodology,
            policy_version=1,
            collected_at="2026-10-14T12:00:00Z",
            coverage_through="2026-10-14T12:00:00Z",
            source_manifest_digest="a" * 64,
            windows={},
            coverage={},
            observations={},
            estimates={},
        )

    old = envelope(3, namespace)
    snapshots.stage_generation(profile, old)
    snapshots.make_current(profile, 3)
    snapshots.stage_generation(profile, envelope(4, namespace))
    staged = snapshots.read_generation(profile, 4)
    assert staged is not None
    assert staged.methodology_id == namespace
    directory = storage.open_profile_dir(profile, "snapshots")
    try:
        storage.atomic_write(directory, ".legacy-crash.tmp", b"partial")
    finally:
        os.close(directory)

    def forbidden(*args):
        raise AssertionError(
            "Legacy snapshot repair must not decode old work or sweep accepted blob children"
        )

    monkeypatch.setattr(collector_steps, "load_state", forbidden)
    monkeypatch.setattr(store, "sweep_orphans", forbidden)
    assert collector_steps.recover(profile, checkpoint) is None
    assert snapshots.read_generation(profile, 4) is None
    recovered = snapshots.read_generation(profile, 3)
    assert recovered is not None
    assert snapshots.encode(recovered) == snapshots.encode(old)
    directory = storage.open_profile_dir(profile, "snapshots")
    try:
        assert ".legacy-crash.tmp" not in storage.list_names(directory)
        assert storage.read_file(
            directory, "current.json", snapshots.MAX_SNAPSHOT_BYTES
        ) == snapshots.encode(old)
    finally:
        os.close(directory)
    # The corrected writer can now stage its genuine next generation rather
    # than colliding forever with the old writer's uncommitted file.
    snapshots.stage_generation(profile, envelope(4, "f" * 32))
    staged = snapshots.read_generation(profile, 4)
    assert staged is not None
    assert staged.methodology_id == "f" * 32


def test_snapshot_recovery_retains_pin_authority_but_not_uncommitted_duplicate_generations(
    monkeypatch,
):
    from dataclasses import replace

    from ssgrep.accounting import snapshots

    profile = "personal-local"
    committed = snapshots.build_envelope(
        generation=3,
        counter_epoch=1,
        methodology_id="a" * 32,
        policy_version=1,
        collected_at="2026-10-14T12:00:00Z",
        coverage_through="2026-10-14T12:00:00Z",
        source_manifest_digest="b" * 64,
        windows={},
        coverage={},
        observations={},
        estimates={},
    )
    snapshots.publish(profile, committed)
    snapshots.stage_generation(profile, replace(committed, generation=4))
    pins = {committed.snapshot_id}
    monkeypatch.setattr(snapshots, "pinned_snapshot_ids", lambda _: pins)
    assert snapshots.discard_uncommitted(profile, 3) == [4]
    assert snapshots.read_generation(profile, 4) is None
    recovered = snapshots.read_generation(profile, 3)
    assert recovered is not None
    assert recovered.snapshot_id in pins
    unique = snapshots.build_envelope(
        generation=4,
        counter_epoch=2,
        methodology_id="c" * 32,
        policy_version=1,
        collected_at="2026-10-14T12:01:00Z",
        coverage_through="2026-10-14T12:01:00Z",
        source_manifest_digest="d" * 64,
        windows={},
        coverage={},
        observations={},
        estimates={},
    )
    snapshots.stage_generation(profile, unique)
    pins.clear()
    pins.add(unique.snapshot_id)
    try:
        snapshots.discard_uncommitted(profile, 3)
    except ValueError as error:
        assert str(error) == "uncommitted_pinned_generation"
    else:
        raise AssertionError(
            "A uniquely pinned uncommitted snapshot must block instead of losing its only bytes"
        )
    assert snapshots.read_generation(profile, 4) == unique
    assert snapshots.current_generation(profile) == 3
