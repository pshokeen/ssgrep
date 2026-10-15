"""Original-publication maintenance must expose backlog, never replay all children."""

import pytest

from ssgrep.accounting import collector_published_retention as maintenance, collector_store as store

PROFILE = "personal-local"
ARCHIVE = "identity-archive-native_identity_v2"


def prepare(
    checkpoint, budget=1024 * 1024, evidence_before_day="2026-11-01", before_day="2026-10-01"
):
    return maintenance.prepare(
        PROFILE,
        checkpoint,
        rows=None,
        before_day=before_day,
        evidence_before_day=evidence_before_day,
        clock=lambda: 0,
        deadline=1,
        byte_budget=budget,
    )


def test_oversized_original_manifest_is_pending_without_decoding_it(monkeypatch):
    reference = store.write_blob(PROFILE, ARCHIVE, {"payload": "x" * 256})
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})

    def forbidden(*args):
        raise AssertionError(
            "An oversized child must not be read outside its explicit maintenance budget"
        )

    monkeypatch.setattr(store, "read_blob", forbidden)
    work = prepare(checkpoint, budget=1)
    assert work.pending and not work.changed and not work.delete_receipts
    assert checkpoint.retention[maintenance.KEY]["blocked_child_bytes"] > 1


def test_unchanged_accepted_archive_proof_skips_all_owned_child_reads_after_restart(monkeypatch):
    reference = store.write_blob(PROFILE, ARCHIVE, {"schema": 1, "blobs": {}})
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
    assert not prepare(checkpoint).pending
    restarted = store.Checkpoint.from_dict(checkpoint.to_dict())

    def forbidden(*args):
        raise AssertionError("An unchanged completed proof cannot hide work in a replay hydrator")

    monkeypatch.setattr(store, "read_blob", forbidden)
    assert not prepare(restarted).pending


def test_maintenance_quantum_selects_only_one_original_data_child(monkeypatch):
    from ssgrep.accounting import published_retention

    a = store.write_blob(PROFILE, "original-a", {"retained": "a"})
    b = store.write_blob(PROFILE, "original-b", {"retained": "b"})
    reference = store.write_blob(
        PROFILE,
        ARCHIVE,
        {
            "schema": 1,
            "blobs": {"a": a, "b": b},
            "segments": [],
            "published_documents": {
                "format": "published-session-v1",
                "history_blob": None,
                "receipt_blobs": {},
            },
        },
    )
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
    selections = []

    def selected(*args, **kwargs):
        selections.append(kwargs["blob_names"])
        return published_retention.ChildResult(False, (), kwargs["manifest"])

    monkeypatch.setattr(published_retention, "stage_children", selected)
    first = prepare(checkpoint)
    assert first.pending and selections == [("a",)]
    restarted = store.Checkpoint.from_dict(checkpoint.to_dict())
    assert not prepare(restarted).pending
    assert selections == [("a",), ("b",)]


def test_public_receipt_cleanup_is_a_bounded_postcommit_obligation(monkeypatch):
    import os

    from ssgrep.accounting import purge_files, storage

    reference = store.write_blob(PROFILE, ARCHIVE, {"schema": 1, "blobs": {}})
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
    assert not prepare(checkpoint).pending
    names = [f"r{i:03}" for i in range(32)]
    directory = storage.open_profile_dir(PROFILE, "receipts")
    try:
        for name in names:
            storage.atomic_write(directory, name + ".json", b"{}\\n")
    finally:
        os.close(directory)
    checkpoint.retention[maintenance.KEY]["cleanup_ids"] = names

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A <=32-ID cleanup intent must not enumerate all receipt IDs or read archive bodies"
        )

    monkeypatch.setattr(purge_files, "receipt_ids", forbidden)
    monkeypatch.setattr(store, "read_blob", forbidden)
    work = prepare(checkpoint)
    assert len(work.delete_receipts) == 32
    store.commit_checkpoint(PROFILE, checkpoint)  # Durable intent precedes any unlink.
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    assert prepare(restarted).delete_receipts == work.delete_receipts
    assert maintenance.finish(PROFILE, work) == 32
    assert not prepare(restarted).delete_receipts
    assert restarted.retention[maintenance.KEY]["cleanup_ids"] == []


def test_counter_linkage_uses_evidence_floor_without_aging_publications_at_that_floor(monkeypatch):
    from ssgrep.accounting import published_retention

    counter = store.write_blob(
        PROFILE, "original-counter", {"counted": [["e|completed", "2026-10-20"]]}
    )
    ledger = store.write_blob(PROFILE, "original-ledger", {"retained": "unchanged quantities"})
    calls = []

    def selected(*args, **kwargs):
        calls.append((kwargs["blob_names"], kwargs["before_day"]))
        return published_retention.ChildResult(False, (), kwargs["manifest"])

    monkeypatch.setattr(published_retention, "stage_children", selected)
    for counter_name in ("counters", "counters-scope-previous"):
        reference = store.write_blob(
            PROFILE,
            ARCHIVE,
            {
                "schema": 1,
                "blobs": {counter_name: counter, "ledger-old": ledger},
                "segments": [],
                "published_documents": {
                    "format": "published-session-v1",
                    "history_blob": None,
                    "receipt_blobs": {},
                },
            },
        )
        checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
        assert prepare(checkpoint).pending
        assert calls[-1] == ((counter_name,), "2026-11-01")
        assert not prepare(checkpoint).pending
        assert calls[-1] == (("ledger-old",), "2026-10-01")
        # A changed evidence horizon invalidates the completed counter proof even
        # when receipt retention and every archive pointer are unchanged.
        assert prepare(checkpoint, evidence_before_day="2026-11-02").pending
        assert calls[-1] == ((counter_name,), "2026-11-02")


def test_future_counter_is_indexed_once_across_days_then_only_expired_linkage_changes(monkeypatch):
    counter = {
        "epoch": 7,
        "epoch_started_at": "2026-09-01T00:00:00Z",
        "counted": [["future|invoked", "2026-12-01"]],
        "executions": [],
        "rendered_bytes": [],
        "durations": [],
    }
    reference = store.write_blob(PROFILE, "counters", counter)
    manifest = {
        "schema": 1,
        "blobs": {"counters": reference},
        "segments": [],
        "published_documents": {
            "format": "published-session-v1",
            "history_blob": None,
            "receipt_blobs": {},
        },
    }
    checkpoint = store.Checkpoint(
        PROFILE, blobs={ARCHIVE: store.write_blob(PROFILE, ARCHIVE, manifest)}
    )
    original_read, reads = store.read_blob, []

    def observed(profile, filename):
        if filename == reference:
            reads.append(filename)
        return original_read(profile, filename)

    monkeypatch.setattr(store, "read_blob", observed)
    assert not prepare(checkpoint).pending  # Empty phases finish in this same quantum.
    assert reads == [reference]
    assert not prepare(
        checkpoint, evidence_before_day="2026-11-02", before_day="2026-10-02"
    ).pending
    assert not prepare(
        checkpoint, evidence_before_day="2026-12-01", before_day="2026-10-03"
    ).pending
    assert reads == [reference]  # Equality is retained, so the minimum day still proves clean.
    assert not prepare(
        checkpoint, evidence_before_day="2026-12-02", before_day="2026-10-04"
    ).pending
    assert reads == [reference, reference]
    after = original_read(PROFILE, checkpoint.blobs[ARCHIVE])
    assert after is not None
    expired = original_read(PROFILE, after["blobs"]["counters"])
    assert expired == {**counter, "counted": []}

    def forbidden(profile, filename):
        if filename == after["blobs"]["counters"]:
            raise AssertionError("A known-empty linkage child must not be reread on a later day")
        return original_read(profile, filename)

    monkeypatch.setattr(store, "read_blob", forbidden)
    assert not prepare(
        checkpoint, evidence_before_day="2026-12-03", before_day="2026-10-05"
    ).pending


def test_exact_changed_child_reference_invalidates_its_cached_empty_date_proof(monkeypatch):
    original = {"epoch": 2, "counted": []}
    reference = store.write_blob(PROFILE, "counters", original)
    manifest = {
        "schema": 1,
        "blobs": {"counters": reference},
        "segments": [],
        "published_documents": {
            "format": "published-session-v1",
            "history_blob": None,
            "receipt_blobs": {},
        },
    }
    checkpoint = store.Checkpoint(
        PROFILE, blobs={ARCHIVE: store.write_blob(PROFILE, ARCHIVE, manifest)}
    )
    assert not prepare(checkpoint).pending
    indexed = store.read_blob(PROFILE, checkpoint.blobs[ARCHIVE])
    assert indexed is not None
    replacement = store.write_blob(
        PROFILE, "counters", {**original, "counted": [["late|rendered", "2026-10-01"]]}
    )
    indexed["blobs"]["counters"] = (
        replacement  # Keep the stale index deliberately; only its source ref differs.
    )
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, indexed)
    original_read, reads = store.read_blob, []

    def observed(profile, filename):
        if filename == replacement:
            reads.append(filename)
        return original_read(profile, filename)

    monkeypatch.setattr(store, "read_blob", observed)
    assert not prepare(checkpoint).pending
    assert reads == [replacement]
    after = original_read(PROFILE, checkpoint.blobs[ARCHIVE])
    assert after is not None
    assert original_read(PROFILE, after["blobs"]["counters"]) == original


def test_ledger_reduction_segment_history_and_receipt_indexes_skip_clean_future_days(monkeypatch):
    from collections import Counter

    from tests.accounting.conftest import build_evidence
    from tests.accounting.test_published_archive_purge import seed

    checkpoint, _, manifest = seed()
    reduction = store.write_blob(
        PROFILE,
        "reduction-original",
        {
            "executions": [
                {
                    "execution_id": "old",
                    "day": "2026-10-13",
                    "runtime": "claude",
                    "session_id": "alpha",
                }
            ],
            "invocations": [],
            "session_roots": [["claude", "alpha", "claude:alpha"]],
            "session_last": [["claude", "alpha", "2026-10-13T10:00:00Z"]],
        },
    )
    manifest["blobs"]["reduction"] = reduction
    segments = store.write_segments(PROFILE, [build_evidence(ts_utc="2026-10-13T10:00:00Z")])
    manifest["segments"] = [ref.to_dict() for ref in segments]
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    data_refs = {
        manifest["blobs"]["ledger-original"],
        reduction,
        manifest["published_documents"]["history_blob"],
        *manifest["published_documents"]["receipt_blobs"].values(),
    }
    original_blob, original_segment = store.read_blob, store.read_segment
    bodies, compact_reads = [], []

    def blob(profile, filename):
        if filename in data_refs:
            bodies.append(filename)
        return original_blob(profile, filename)

    def segment(profile, ref):
        compact_reads.append(ref.name)
        yield from original_segment(profile, ref)

    monkeypatch.setattr(store, "read_blob", blob)
    monkeypatch.setattr(store, "read_segment", segment)
    for _ in range(8):
        if not prepare(checkpoint).pending:
            break
    else:
        raise AssertionError("Finite selected-child indexing backlog did not drain")
    counts = Counter(bodies)
    assert all(counts[ref] == 1 for ref in data_refs)
    assert compact_reads == [segments[0].name]
    before = (list(bodies), list(compact_reads))
    assert not prepare(
        checkpoint, before_day="2026-10-02", evidence_before_day="2026-11-02"
    ).pending
    assert not prepare(
        checkpoint, before_day="2026-10-13", evidence_before_day="2026-11-03"
    ).pending
    assert (
        bodies,
        compact_reads,
    ) == before  # Inclusive minimum is still future under exclusive expiry.
    # Only one newly eligible child is processed; other expired children remain
    # explicit backlog rather than being reread/reduced together.
    assert prepare(checkpoint, before_day="2026-10-14", evidence_before_day="2026-11-04").pending
    assert bodies[-1] == manifest["blobs"]["ledger-original"]
    assert len(bodies) == len(before[0]) + 1 and compact_reads == before[1]


def test_empty_archive_metadata_edges_are_bounded_and_resumable_without_fake_completion():
    references = {}
    for number in range(70):
        name = ARCHIVE + f"-{number:03}"
        references[name] = store.write_blob(PROFILE, name, {"schema": 1, "blobs": {}})
    checkpoint = store.Checkpoint(PROFILE, blobs=references)
    first = prepare(checkpoint)
    assert first.pending
    job = checkpoint.retention[maintenance.KEY]
    assert job["metadata_edges"] == 64 and job["archive_after"] == ARCHIVE + "-063"
    restarted = store.Checkpoint.from_dict(checkpoint.to_dict())
    assert not prepare(restarted).pending
    assert restarted.retention[maintenance.KEY]["metadata_edges"] == 6


def test_legacy_segment_index_budget_defers_then_publishes_queryable_identity_ranges(
    tmp_path, monkeypatch
):
    import os
    from dataclasses import replace

    from ssgrep.accounting import collector_maintenance, storage
    from ssgrep.accounting.native_identity import with_origin
    from tests.accounting.test_minimal_evidence import CANARY, evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    items = [
        with_origin(
            replace(evidence(), session_id=f"session-{number}", record_id=f"call:{number}"),
            f"claude:session-{number}",
            f"call:{number}",
            "native",
        )
        for number in range(2)
    ]
    body = b"".join(store.evidence_line(item) + b"\n" for item in items)
    directory = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        storage.write_once(directory, "seg-legacy-index.jsonl", body)
    finally:
        os.close(directory)
    legacy = store.SegmentRef("seg-legacy-index.jsonl", len(items), len(body), "backfill")
    checkpoint = store.Checkpoint(PROFILE, generation=11, segments=[legacy])
    store.commit_checkpoint(PROFILE, checkpoint)
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    checkpoint_bytes = (root / store.CHECKPOINT).read_bytes()
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    kwargs = {"clock": lambda: 0, "deadline": 1, "should_stop": lambda: False}
    used, pending = collector_maintenance.ensure_indexes(
        PROFILE, checkpoint, byte_budget=0, **kwargs
    )
    assert used == 0 and pending
    assert checkpoint.segments == [legacy]
    assert checkpoint.retention["evidence_index_pending"] == [legacy.name]
    assert {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    used, pending = collector_maintenance.ensure_indexes(
        PROFILE, checkpoint, byte_budget=len(body), **kwargs
    )
    assert used == len(body) and not pending
    assert "evidence_index_pending" not in checkpoint.retention
    (indexed,) = checkpoint.segments
    assert indexed.index is not None and indexed.name == legacy.name
    assert indexed.origin == "backfill"
    assert (root / "evidence" / legacy.name).read_bytes() == body
    assert (root / store.CHECKPOINT).read_bytes() == checkpoint_bytes
    selected = list(store.iter_evidence(PROFILE, [indexed], native_keys={items[0].native_key}))
    assert [(item.native_key, item.digest) for item in selected] == [
        (items[0].native_key, items[0].digest)
    ]
    selected = list(store.iter_evidence(PROFILE, [indexed], sessions={("claude", "session-1")}))
    assert [(item.native_key, item.digest) for item in selected] == [
        (items[1].native_key, items[1].digest)
    ]
    assert CANARY.encode() not in b"".join(
        path.read_bytes() for path in root.rglob("*") if path.is_file()
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    assert restarted.segments == [indexed]
    assert (
        list(store.iter_evidence(PROFILE, restarted.segments))[0].native_key == items[0].native_key
    )


@pytest.mark.parametrize("damage", ["missing", "truncated", "oversized", "unterminated"])
def test_damaged_legacy_index_input_is_typed_corruption_without_checkpoint_publication(
    tmp_path, monkeypatch, damage
):
    import os

    import pytest

    from ssgrep.accounting import collector_maintenance, storage
    from ssgrep.accounting.collector_types import CollectError
    from tests.accounting.test_minimal_evidence import evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    body = store.evidence_line(evidence()) + b"\n"
    directory = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        if damage != "missing":
            damaged = {
                "truncated": body[:-2],
                "oversized": body + b"  ",
                "unterminated": body[:-1],
            }[damage]
            storage.write_once(directory, "seg-damaged.jsonl", damaged)
    finally:
        os.close(directory)
    checkpoint = store.Checkpoint(
        PROFILE,
        generation=12,
        segments=[
            store.SegmentRef(
                "seg-damaged.jsonl",
                1,
                len(body) - 1 if damage == "unterminated" else len(body),
            )
        ],
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    with pytest.raises(CollectError) as failed:
        collector_maintenance.ensure_indexes(
            PROFILE,
            checkpoint,
            clock=lambda: 0,
            deadline=1,
            byte_budget=len(body),
            should_stop=lambda: False,
        )
    assert failed.value.condition == "state_corrupt"
    assert {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    assert store.load_checkpoint(PROFILE) == checkpoint
    assert checkpoint.segments[0].index is None


def test_incremental_compaction_stops_before_second_merge_and_resumes_without_record_loss(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from ssgrep.accounting import collector_maintenance, storage
    from ssgrep.accounting.native_identity import with_origin
    from tests.accounting.test_minimal_evidence import evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    items = [
        with_origin(
            replace(evidence(), record_id=f"call:{number}"),
            "claude:compaction",
            f"call:{number}",
            "native",
        )
        for number in range(4)
    ]
    refs = [store.write_segments(PROFILE, [item], origin="backfill")[0] for item in items]
    monkeypatch.setattr(store, "MAX_SEGMENTS", 2)
    monkeypatch.setattr(store, "SEGMENT_TARGET_BYTES", sum(sorted(ref.size for ref in refs)[-2:]))
    checkpoint = store.Checkpoint(PROFILE, generation=13, segments=refs)
    store.commit_checkpoint(PROFILE, checkpoint)
    directory = storage.data_root() / "accounting" / PROFILE / "collector"
    durable_bytes = (directory / store.CHECKPOINT).read_bytes()
    originals = {ref.name: (directory / "evidence" / ref.name).read_bytes() for ref in refs}
    first_pair = collector_maintenance.compact_pair(refs)
    assert first_pair is not None
    first_cost = sum(ref.size for ref in first_pair)
    kwargs = {
        "expired_sessions": set(),
        "clock": lambda: 0,
        "deadline": 1,
        "should_stop": lambda: False,
    }
    partial, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE, refs, previous=None, byte_budget=first_cost + 1, **kwargs
    )
    assert used == first_cost and dropped == 0 and len(partial) == 3
    assert job["compaction_pending"] and not job["pending"]
    assert all(ref.origin == "backfill" for ref in partial)
    expected = sorted((item.native_key, item.digest) for item in items)
    assert (
        sorted((item.native_key, item.digest) for item in store.iter_evidence(PROFILE, partial))
        == expected
    )
    assert {name: (directory / "evidence" / name).read_bytes() for name in originals} == originals
    assert (directory / store.CHECKPOINT).read_bytes() == durable_bytes
    final, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE, partial, previous=job, byte_budget=sum(ref.size for ref in partial), **kwargs
    )
    assert used > 0 and dropped == 0 and len(final) == 2
    assert not job["compaction_pending"] and not job["pending"]
    assert (
        sorted((item.native_key, item.digest) for item in store.iter_evidence(PROFILE, final))
        == expected
    )
    assert (directory / store.CHECKPOINT).read_bytes() == durable_bytes
    checkpoint.segments = final
    store.commit_checkpoint(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    assert (
        sorted(
            (item.native_key, item.digest)
            for item in store.iter_evidence(PROFILE, restarted.segments)
        )
        == expected
    )


def test_unmergeable_segments_do_not_claim_unfinishable_compaction_backlog(tmp_path, monkeypatch):
    from dataclasses import replace

    from ssgrep.accounting import collector_maintenance, storage
    from tests.accounting.test_minimal_evidence import evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    refs = [
        store.write_segments(
            PROFILE, [replace(evidence(), record_id=f"call:{number}", origin_id=f"call:{number}")]
        )[0]
        for number in range(3)
    ]
    monkeypatch.setattr(store, "MAX_SEGMENTS", 2)
    monkeypatch.setattr(store, "SEGMENT_TARGET_BYTES", min(ref.size for ref in refs) - 1)
    directory = storage.data_root() / "accounting" / PROFILE / "collector"
    before = {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert collector_maintenance.compact_pair(refs) is None
    retained, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE,
        refs,
        expired_sessions=set(),
        previous=None,
        clock=lambda: 0,
        deadline=1,
        byte_budget=sum(ref.size for ref in refs),
        should_stop=lambda: False,
    )
    assert retained == refs and used == dropped == 0
    assert not job["compaction_pending"] and not job["pending"]
    assert {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    } == before
    assert [item.record_id for item in store.iter_evidence(PROFILE, retained)] == [
        "call:0",
        "call:1",
        "call:2",
    ]


def test_legacy_index_backlog_advances_one_segment_per_budget_across_restart(tmp_path, monkeypatch):
    from dataclasses import replace

    from ssgrep.accounting import collector_maintenance, storage
    from tests.accounting.test_minimal_evidence import CANARY, evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    items = [
        replace(
            evidence(),
            session_id=f"owner-{number}",
            record_id=f"call:{number}",
            origin_domain=f"claude:owner-{number}",
            origin_id=f"call:{number}",
            digest=f"digest-{number}",
        )
        for number in range(3)
    ]
    refs = [replace(store.write_segments(PROFILE, [item])[0], index=None) for item in items]
    checkpoint = store.Checkpoint(PROFILE, generation=14, segments=refs)
    store.commit_checkpoint(PROFILE, checkpoint)
    kwargs = {"clock": lambda: 0, "deadline": 1, "should_stop": lambda: False}
    expected = [(item.native_key, item.digest, item.session_id) for item in items]
    for number, legacy in enumerate(refs):
        checkpoint = store.load_checkpoint(PROFILE)
        assert checkpoint is not None
        used, pending = collector_maintenance.ensure_indexes(
            PROFILE, checkpoint, byte_budget=legacy.size, **kwargs
        )
        assert used == legacy.size
        assert pending == (number < len(refs) - 1)
        assert [ref.index is not None for ref in checkpoint.segments] == [
            index <= number for index in range(len(refs))
        ]
        remaining = [ref.name for ref in refs[number + 1 :]]
        assert checkpoint.retention.get("evidence_index_pending", []) == remaining
        for owner, item in enumerate(items[: number + 1]):
            selected = list(
                store.iter_evidence(
                    PROFILE,
                    checkpoint.segments,
                    sessions={("claude", f"owner-{owner}")},
                    native_keys={item.native_key},
                )
            )
            assert [(row.native_key, row.digest, row.session_id) for row in selected] == [
                expected[owner]
            ]
        store.commit_checkpoint(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    assert "evidence_index_pending" not in restarted.retention
    assert [
        (item.native_key, item.digest, item.session_id)
        for item in store.iter_evidence(PROFILE, restarted.segments)
    ] == expected
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    assert CANARY.encode() not in b"".join(
        path.read_bytes() for path in root.rglob("*") if path.is_file()
    )


def test_bounded_expiry_resumes_then_compacts_without_mixing_live_and_backfill_owners(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from ssgrep.accounting import collector_maintenance, storage
    from tests.accounting.test_minimal_evidence import CANARY, evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    active = [
        replace(
            evidence(),
            session_id=f"active-{number}",
            record_id=f"call:{number}",
            origin_domain=f"claude:active-{number}",
            origin_id=f"call:{number}",
            digest=f"active-digest-{number}",
        )
        for number in range(4)
    ]
    expired = [
        replace(
            evidence(),
            session_id="expired",
            record_id=f"expired:{number}",
            origin_domain="claude:expired",
            origin_id=f"expired:{number}",
            digest=f"expired-digest-{number}",
        )
        for number in range(2)
    ]
    refs = [
        store.write_segments(PROFILE, [expired[0], active[0]], origin="live")[0],
        store.write_segments(PROFILE, [active[1]], origin="live")[0],
        store.write_segments(PROFILE, [expired[1], active[2]], origin="backfill")[0],
        store.write_segments(PROFILE, [active[3]], origin="backfill")[0],
    ]
    monkeypatch.setattr(store, "MAX_SEGMENTS", 2)
    monkeypatch.setattr(store, "SEGMENT_TARGET_BYTES", sum(ref.size for ref in refs))
    checkpoint = store.Checkpoint(PROFILE, generation=15, segments=refs)
    store.commit_checkpoint(PROFILE, checkpoint)
    kwargs = {
        "expired_sessions": {("claude", "expired")},
        "clock": lambda: 0,
        "deadline": 1,
        "should_stop": lambda: False,
    }
    partial, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE, refs, previous=None, byte_budget=refs[0].size, **kwargs
    )
    assert used == refs[0].size and dropped == 1
    assert job["pending"] == [ref.name for ref in refs[1:]]
    assert job["compaction_pending"]
    assert {item.native_key for item in store.iter_evidence(PROFILE, partial)} == {
        item.native_key for item in [*active, expired[1]]
    }
    checkpoint.segments = partial
    checkpoint.retention["evidence_maintenance"] = job
    store.commit_checkpoint(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    final, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE,
        restarted.segments,
        previous=restarted.retention["evidence_maintenance"],
        byte_budget=10 * sum(ref.size for ref in refs),
        **kwargs,
    )
    assert used > 0 and dropped == 1
    assert not job["pending"] and not job["compaction_pending"]
    assert len(final) == 2
    expected = {
        "live": {(item.native_key, item.digest, item.session_id) for item in active[:2]},
        "backfill": {(item.native_key, item.digest, item.session_id) for item in active[2:]},
    }
    assert {
        ref.origin: {
            (item.native_key, item.digest, item.session_id)
            for item in store.iter_evidence(PROFILE, [ref])
        }
        for ref in final
    } == expected
    restarted.segments = final
    restarted.retention["evidence_maintenance"] = job
    store.commit_checkpoint(PROFILE, restarted)
    store.sweep_orphans(PROFILE, restarted)
    durable = store.load_checkpoint(PROFILE)
    assert durable is not None
    assert (
        list(store.iter_evidence(PROFILE, durable.segments, sessions={("claude", "expired")})) == []
    )
    for item in active:
        retained = list(
            store.iter_evidence(PROFILE, durable.segments, native_keys={item.native_key})
        )
        assert [(row.native_key, row.digest, row.session_id) for row in retained] == [
            (item.native_key, item.digest, item.session_id)
        ]
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    persisted = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    assert CANARY.encode() not in persisted
    assert all(item.digest.encode() not in persisted for item in expired)
