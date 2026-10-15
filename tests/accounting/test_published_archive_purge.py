"""Original published JSON is erased by its original dates, never revalued."""

import hashlib
import json

import pytest

from ssgrep.accounting import collector_store as store, purge, storage
from ssgrep.accounting.published_documents import merge_history, read_document
from ssgrep.accounting.purge_model import PurgeError
from ssgrep.accounting.purge_select import Selection
from tests.accounting.test_published_documents import ARCHIVE, OLD_NAMESPACE, PROFILE


def seed(*, mixed=False, mapped=True):
    roots = ("claude:alpha", "claude:beta")
    periods = ("2026-10-14", "2026-10-14") if mixed else ("2026-10-13", "2026-10-14")
    rows = [
        {
            "schema": 1,
            "period": period,
            "revision": 1,
            "status": "closed",
            "supersedes": None,
            "net": {
                "value": -78 if mixed else -39,
                "unit": "context_tokens",
                "basis": "estimated",
                "method": "chars_div4_v1",
                "reason": "",
            },
            "kind": "original",
            "recomputable": True,
        }
        for period in sorted(set(periods))
    ]

    def envelope(document, kind):
        text = json.dumps(document, indent=2) + "\n"
        return store.write_blob(
            PROFILE,
            "published-original",
            {
                "kind": kind,
                "original_utf8": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_generation": 3,
                "original_collected_at": "2026-10-14T12:00:00Z",
            },
        )

    history = envelope(rows, "history")
    receipts = {}
    for index, (root, period) in enumerate(zip(roots, periods, strict=True)):
        receipt_id = f"original-{index}"
        receipts[receipt_id] = envelope(
            {
                "schema": 1,
                "receipt_id": receipt_id,
                "revision": 1,
                "root": root,
                "task_turn": None,
                "candidate_type": "search",
                "atoms": [],
                "allocations": [],
                "exclusions": [],
                "costs": [],
                "methodology": OLD_NAMESPACE,
                "evidence_available": False,
                "execution_ids": [],
                "unit": "context_tokens",
                "period": period,
                "missing_costs": [],
                "atom_digests": [],
                "credit_raw": [],
                "identities": {"frequency": "", "tokenizer": "", "pricing": ""},
            },
            "receipt",
        )
    ledger = {
        "state": {
            "methodology": {"name": "original-published-input"},
            "candidates": [],
            "debits": [
                {
                    "debit_id": f"old-{index}",
                    "task": {"root": root, "turn": None, "branch": "main"},
                    "day": period,
                    "amount": {"value": 39, "unit": "context_tokens", "basis": "estimated"},
                }
                for index, (root, period) in enumerate(zip(roots, periods, strict=True))
            ],
            "frozen": [],
        },
        "superseded": {},
    }
    original_blobs = {
        "ledger-original": store.write_blob(PROFILE, "ledger-original", ledger),
        "published-history": history,
        **{f"published-{key}": value for key, value in receipts.items()},
    }
    period_roots = {
        period: [root for root, value in zip(roots, periods, strict=True) if value == period]
        for period in set(periods)
    }
    manifest = {
        "schema": 1,
        "methodology": OLD_NAMESPACE,
        "collection_scope": {},
        "identity_method": "published-session-v1",
        "replacement_method": "native_identity_v2",
        "superseded_at": "2026-12-01T00:00:00Z",
        "segments": [],
        "blobs": original_blobs,
        "session_roots": [["claude", "alpha", roots[0]], ["claude", "beta", roots[1]]],
        "session_cwd": [["claude", "alpha", "/alpha"], ["claude", "beta", "/beta"]],
        "root_mapping_status": "available" if mapped else "unavailable",
        "published_documents": {
            "format": "published-session-v1",
            "history_blob": history,
            "receipt_blobs": receipts,
            "roots_by_period": period_roots,
            "source_namespace": OLD_NAMESPACE,
            "source_generation": 3,
            "original_collected_at": "2026-10-14T12:00:00Z",
        },
    }
    checkpoint = store.Checkpoint(
        PROFILE,
        identity_method="native_identity_v2",
        blobs={ARCHIVE: store.write_blob(PROFILE, ARCHIVE, manifest)},
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    return checkpoint, rows, manifest


def test_date_scoped_purge_keeps_other_workspace_original_values_and_bytes():
    checkpoint, rows, original = seed()
    kept_ref = original["published_documents"]["receipt_blobs"]["original-1"]
    kept_envelope = store.read_blob(PROFILE, kept_ref)
    purge.execute(Selection(PROFILE, "2026-10-14T00:00:00Z", "/alpha"))
    current = store.load_checkpoint(PROFILE)
    assert current is not None
    manifest = store.read_blob(PROFILE, current.blobs[ARCHIVE])
    assert manifest is not None
    documents = manifest["published_documents"]
    envelope, kept = read_document(PROFILE, documents["history_blob"], "history")
    assert kept == rows[1:]
    assert envelope["source_generation"] == 3
    assert envelope["original_collected_at"] == "2026-10-14T12:00:00Z"
    assert documents["receipt_blobs"] == {"original-1": kept_ref}
    assert store.read_blob(PROFILE, kept_ref) == kept_envelope
    ledger = store.read_blob(PROFILE, manifest["blobs"]["ledger-original"])
    assert ledger is not None
    assert [row["debit_id"] for row in ledger["state"]["debits"]] == ["old-1"]
    merged, warnings = merge_history(PROFILE, [], current.blobs)
    assert warnings == () and merged[0]["net"]["value"] == -39
    public = storage.data_root() / "accounting" / PROFILE / "history" / "periods.json"
    assert json.loads(public.read_bytes()) == merged


@pytest.mark.parametrize("mixed,mapped", [(True, True), (False, False)])
def test_unprovable_original_scope_blocks_before_any_checkpoint_write(mixed, mapped):
    checkpoint, _, _ = seed(mixed=mixed, mapped=mapped)
    with pytest.raises(PurgeError) as failed:
        purge.plan(Selection(PROFILE, "2026-10-15T00:00:00Z", "/alpha"))
    assert failed.value.condition == "archive_scope_unresolved"
    assert store.load_checkpoint(PROFILE) == checkpoint


def test_whole_profile_uses_original_periods_not_migration_timestamp():
    _, _, _ = seed(mapped=False)
    purge.execute(Selection(PROFILE, "2026-10-15T00:00:00Z", None))
    committed = store.load_checkpoint(PROFILE)
    assert committed is not None
    assert ARCHIVE not in committed.blobs


def test_bounded_history_child_reads_only_selected_original_document(monkeypatch):
    from ssgrep.accounting.published_retention import stage_children

    checkpoint, rows, original = seed()
    history = original["published_documents"]["history_blob"]
    read = store.read_blob
    observed = []

    def selected_only(profile, filename):
        observed.append(filename)
        assert filename == history
        return read(profile, filename)

    with monkeypatch.context() as instrument:
        instrument.setattr(store, "read_blob", selected_only)
        result = stage_children(
            PROFILE,
            checkpoint,
            name=ARCHIVE,
            manifest=original,
            before_day="2026-10-14",
            history=True,
        )
        assert result.changed and result.expired_receipts == ()
    assert observed == [history]
    after = read(PROFILE, checkpoint.blobs[ARCHIVE])
    assert after is not None
    assert after["blobs"]["ledger-original"] == original["blobs"]["ledger-original"]
    assert (
        after["published_documents"]["receipt_blobs"]
        == original["published_documents"]["receipt_blobs"]
    )
    envelope, retained = read_document(
        PROFILE, after["published_documents"]["history_blob"], "history"
    )
    assert retained == rows[1:]
    assert envelope["original_collected_at"] == "2026-10-14T12:00:00Z"


def test_bounded_indexed_receipt_child_erases_ref_without_reading_other_bodies(monkeypatch):
    from ssgrep.accounting.published_retention import stage_children

    checkpoint, _, original = seed()
    documents = original["published_documents"]
    documents["receipt_periods"] = {"original-0": "2026-10-13", "original-1": "2026-10-14"}
    documents["receipt_roots"] = {"original-0": "claude:alpha", "original-1": "claude:beta"}
    original_history = documents["history_blob"]
    kept = documents["receipt_blobs"]["original-1"]

    def no_body_read(*args):
        raise AssertionError("indexed receipt pruning must not read document bodies")

    with monkeypatch.context() as instrument:
        instrument.setattr(store, "read_blob", no_body_read)
        result = stage_children(
            PROFILE,
            checkpoint,
            name=ARCHIVE,
            manifest=original,
            before_day="2026-10-14",
            receipt_ids=("original-0",),
        )
        assert result.changed and result.expired_receipts == ("original-0",)
    after = store.read_blob(PROFILE, checkpoint.blobs[ARCHIVE])
    assert after is not None
    assert after["published_documents"]["receipt_blobs"] == {"original-1": kept}
    assert after["published_documents"]["history_blob"] == original_history
    assert after["published_documents"]["receipt_min_period"] == "2026-10-14"
    assert after["blobs"]["ledger-original"] == original["blobs"]["ledger-original"]


def test_counter_child_expires_only_dated_linkage_preserving_all_numeric_aggregates(monkeypatch):
    from ssgrep.accounting.published_retention import stage_children

    checkpoint, _, manifest = seed()
    counter = {
        "epoch": 7,
        "epoch_started_at": "2026-09-01T00:00:00Z",
        "executions": [
            {
                "operation": "search",
                "transport": "cli",
                "outcome": "success",
                "runtime": "unknown",
                "count": 99,
            }
        ],
        "rendered_bytes": [{"operation": "search", "transport": "cli", "bytes": 12345}],
        "rendered_unknown": [],
        "duration_unknown": [],
        "skipped_expired": 2,
        "durations": [
            {
                "operation": "search",
                "transport": "cli",
                "histogram": {"bounds_us": [1000], "buckets": [98, 1], "sum_us": 234567},
            }
        ],
        "counted": [
            ["old|invoked", "2026-10-01"],
            ["boundary|rendered", "2026-10-13"],
            ["new|invoked", "2026-10-14"],
        ],
    }
    ref = store.write_blob(PROFILE, "counters", counter)
    manifest["blobs"]["counters"] = ref
    read, observed = store.read_blob, []

    def only_counter(profile, filename):
        observed.append(filename)
        assert filename == ref
        return read(profile, filename)

    with monkeypatch.context() as instrument:
        instrument.setattr(store, "read_blob", only_counter)
        result = stage_children(
            PROFILE,
            checkpoint,
            name=ARCHIVE,
            manifest=manifest,
            before_day="2026-10-13",
            blob_names=("counters",),
        )
    assert result.changed and result.expired_receipts == () and observed == [ref]
    archived = read(PROFILE, checkpoint.blobs[ARCHIVE])
    assert archived is not None
    pruned = read(PROFILE, archived["blobs"]["counters"])
    assert pruned == {**counter, "counted": counter["counted"][1:]}
    assert archived["published_documents"]["source_generation"] == 3


@pytest.mark.parametrize("crash_at", ["before_cleanup", "history_write"])
@pytest.mark.parametrize(
    "workspace,cutoff,remaining",
    [
        ("/alpha", "2026-10-14T00:00:00Z", ["2026-10-14"]),
        (None, "2026-10-15T00:00:00Z", []),
    ],
)
def test_archive_only_committed_purge_recovers_publications_without_an_active_generation(
    monkeypatch,
    tmp_path,
    workspace,
    cutoff,
    remaining,
    crash_at,
):
    from ssgrep.accounting import purge_cleanup, purge_files, purge_recovery
    from ssgrep.accounting.collector_types import CollectError
    from ssgrep.accounting.published_documents import merge_stored_history, restore_receipts

    checkpoint, _, original = seed()
    native = tmp_path / "original-native.jsonl"
    native.write_bytes(b'{"query":"PRIVATE_NATIVE_STAYS_ORIGINAL"}\n')
    native_bytes = native.read_bytes()
    cached, _ = merge_history(PROFILE, [], checkpoint.blobs)
    purge_cleanup.write_history(PROFILE, cached)
    restore_receipts(PROFILE, checkpoint.blobs, set())
    public = storage.data_root() / "accounting" / PROFILE / "history" / "periods.json"
    receipts = storage.data_root() / "accounting" / PROFILE / "receipts"
    kept_receipt = (receipts / "original-1.json").read_bytes()
    selection = Selection(PROFILE, cutoff, workspace)

    def crash_before_publication_cleanup(*_args):
        raise CollectError("state_corrupt", "private postcommit diagnostic")

    with monkeypatch.context() as crash:
        target, name = (
            (purge, "_finish") if crash_at == "before_cleanup" else (purge_cleanup, "write_history")
        )
        crash.setattr(target, name, crash_before_publication_cleanup)
        with pytest.raises(PurgeError) as failed:
            purge.execute(selection)
    assert "Nothing changed" not in str(failed.value)
    assert "resume" in str(failed.value)
    assert "private postcommit diagnostic" not in str(failed.value)
    committed = store.load_checkpoint(PROFILE)
    assert committed is not None
    assert (
        committed.blobs != checkpoint.blobs
    )  # erasure really committed before the injected failure
    assert committed.generation == checkpoint.generation == 0
    assert purge_files.read_marker(PROFILE) is None
    assert purge_recovery.has_pending_erasure(selection)
    planned = purge.plan(selection).state
    assert planned is not None
    assert not planned.changed
    assert json.loads(public.read_bytes()) == cached  # physically stale after the crash
    # Even before recovery, a cached wrapper must not override the committed archive.
    visible, warnings = merge_stored_history(PROFILE, json.loads(public.read_bytes()))
    assert warnings == ()
    assert [row["period"] for row in visible] == remaining
    purge.execute(selection)
    assert not purge_recovery.has_pending_erasure(selection)
    final = json.loads(public.read_bytes())
    assert [row["period"] for row in final] == remaining
    assert not (receipts / "original-0.json").exists()
    recovered = store.load_checkpoint(PROFILE)
    assert recovered is not None
    if remaining:
        assert (receipts / "original-1.json").read_bytes() == kept_receipt
        assert final[0]["net"]["value"] == -39
        kept = store.read_blob(PROFILE, recovered.blobs[ARCHIVE])
        assert kept is not None
        assert (
            kept["published_documents"]["source_generation"]
            == original["published_documents"]["source_generation"]
        )
    else:
        assert not (receipts / "original-1.json").exists()
        assert ARCHIVE not in recovered.blobs
    assert native.read_bytes() == native_bytes


def test_scoped_purge_keeps_active_root_children_when_original_archive_has_same_logical_role():
    from dataclasses import replace

    from ssgrep.accounting import collector_reduction_store
    from ssgrep.accounting.collector_state import REDUCTION, ReductionDoc, load_state
    from ssgrep.accounting.evidence_references import referenced_files
    from tests.accounting.test_collect_counters import invocation

    checkpoint, _, original = seed()
    mine = replace(invocation(1, "2026-10-01T00:00:00Z"), cwd="/alpha", session_id="alpha")
    other = replace(invocation(2, "2026-10-01T00:00:00Z"), cwd="/beta", session_id="beta")
    active = ReductionDoc(
        invocations=(mine, other),
        lookup_schema="lookups_v1",
        session_roots=(("claude", "alpha", "claude:alpha"), ("claude", "beta", "claude:beta")),
        session_cwd=(("claude", "alpha", "/alpha"), ("claude", "beta", "/beta")),
        invocation_sessions=(
            (mine.invocation_id, "claude", "alpha"),
            (other.invocation_id, "claude", "beta"),
        ),
    )
    checkpoint.blobs[REDUCTION] = collector_reduction_store.write_document(
        PROFILE, REDUCTION, active.to_dict()
    )
    root_manifest = store.read_blob(PROFILE, checkpoint.blobs[REDUCTION])
    assert root_manifest is not None
    surviving_child = root_manifest["roots"]["claude:beta"]
    surviving_rows = store.read_blob(PROFILE, surviving_child)
    # Actual407's old reduction was unsharded; it must not shadow the active
    # same-role root manifest while orphan collection follows both ancestries.
    old_reduction = {
        "executions": [],
        "invocations": [],
        "session_roots": [["claude", "beta", "claude:beta"]],
        "session_cwd": [["claude", "beta", "/beta"]],
        "session_parents": [],
        "invocation_sessions": [],
    }
    old_ref = store.write_blob(PROFILE, REDUCTION, old_reduction)
    original["blobs"][REDUCTION] = old_ref
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, original)
    checkpoint.identity_migration = {
        "status": "complete",
        "superseded_blobs": dict(original["blobs"]),
    }
    store.commit_checkpoint(PROFILE, checkpoint)
    kept_publication = store.read_blob(
        PROFILE, original["published_documents"]["receipt_blobs"]["original-1"]
    )
    purge.execute(Selection(PROFILE, "2026-10-14T00:00:00Z", "/alpha"))
    committed = store.load_checkpoint(PROFILE)
    assert committed is not None
    restored = load_state(PROFILE, committed, committed.methodology)
    assert restored.reduction is not None
    assert restored.reduction.invocations == (other,)
    _, protected = referenced_files(PROFILE, committed)
    assert surviving_child in protected and old_ref in protected
    assert store.read_blob(PROFILE, surviving_child) == surviving_rows
    assert store.read_blob(PROFILE, old_ref) == old_reduction
    assert committed.identity_migration["superseded_blobs"][REDUCTION] != committed.blobs[REDUCTION]
    store.sweep_orphans(PROFILE, committed)
    recovered = store.load_checkpoint(PROFILE)
    assert recovered is not None
    restarted = load_state(PROFILE, recovered, committed.methodology)
    assert restarted.reduction == restored.reduction
    archived = store.read_blob(PROFILE, committed.blobs[ARCHIVE])
    assert archived is not None
    kept_ref = archived["published_documents"]["receipt_blobs"]["original-1"]
    assert store.read_blob(PROFILE, kept_ref) == kept_publication


@pytest.mark.parametrize("untimed", [False, True])
def test_archived_segment_expiry_commits_before_retiring_original_and_preserves_measurements(
    tmp_path, monkeypatch, untimed
):
    from dataclasses import replace

    from ssgrep.accounting.evidence_references import referenced_files
    from ssgrep.accounting.native_identity import with_origin
    from ssgrep.accounting.published_retention import stage_children
    from tests.accounting.test_minimal_evidence import CANARY, evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkpoint, _, manifest = seed()

    def usage(identity, stamp):
        return with_origin(
            replace(
                evidence(),
                record_id=identity,
                ts_utc=stamp,
                kind="usage",
                payload={
                    "cwd": "/alpha",
                    "tokens": {"input_tokens": 17, "output_tokens": 9},
                    "cost_nano_usd": 611,
                    "streaming_updates": 3,
                    "output": CANARY,
                },
            ),
            "claude:alpha",
            identity,
            "native",
        )

    old = usage("usage:old", "2026-10-01T00:00:00Z")
    newer = usage("usage:new", "2026-10-14T00:00:00Z")
    survivors = [newer]
    if untimed:
        survivors.append(usage("usage:untimed", None))
    (original,) = store.write_segments(PROFILE, [old, *survivors], origin="backfill")
    manifest["segments"] = [original.to_dict()]
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    store.commit_checkpoint(PROFILE, checkpoint)
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "evidence"
    original_bytes = (directory / original.name).read_bytes()
    assert CANARY.encode() not in original_bytes

    result = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=manifest,
        before_day="2026-10-14",
        segment_names=(original.name,),
    )
    assert result.changed and result.manifest is not None
    replacements = [store.SegmentRef.from_dict(raw) for raw in result.manifest["segments"]]
    (replacement,) = replacements
    assert replacement.origin == "backfill"
    assert (directory / replacement.name).read_bytes() == b"".join(
        store.evidence_line(item) + b"\n" for item in survivors
    )
    restored = list(store.iter_evidence(PROFILE, replacements))
    assert [(item.native_key, item.digest) for item in restored] == [
        (item.native_key, item.digest) for item in survivors
    ]
    assert all(
        item.payload["tokens"] == {"input_tokens": 17, "output_tokens": 9} for item in restored
    )
    assert all(item.payload["cost_nano_usd"] == 611 for item in restored)
    assert all(item.payload["streaming_updates"] == 3 for item in restored)
    dates = result.manifest["published_documents"]["retention_child_dates"]
    assert dates["segment:" + replacement.name]["minimum"] == "2026-10-14"
    assert dates["segment:" + replacement.name]["complete"] is not untimed
    durable = store.load_checkpoint(PROFILE)
    assert durable is not None
    assert original.name in referenced_files(PROFILE, durable)[0]
    assert original.name not in referenced_files(PROFILE, checkpoint)[0]
    assert (directory / original.name).read_bytes() == original_bytes

    store.commit_checkpoint(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    assert not (directory / original.name).exists()
    retained_bytes = (directory / replacement.name).read_bytes()
    second = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=result.manifest,
        before_day="2026-10-14",
        segment_names=(replacement.name,),
    )
    assert not second.changed
    assert (directory / replacement.name).read_bytes() == retained_bytes
    assert store.load_checkpoint(PROFILE) == checkpoint


def test_completed_retention_refreshes_superseded_refs_without_rewriting_other_ancestries(
    tmp_path, monkeypatch
):
    from ssgrep.accounting.published_retention import stage_children

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkpoint, _, manifest = seed()
    counters = {"epoch": 4, "counted": [["expired|invoked", "2026-10-01"]], "executions": []}
    prior = store.write_blob(PROFILE, "counters", counters)
    active = store.write_blob(PROFILE, "counters", {**counters, "epoch": 9, "counted": []})
    manifest["blobs"]["counters"] = prior
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    other_name = ARCHIVE + "-other"
    other = store.write_blob(
        PROFILE, other_name, {"schema": 1, "blobs": {"counters": prior}, "segments": []}
    )
    checkpoint.blobs.update({other_name: other, "counters": active})
    checkpoint.identity_migration = {
        "status": "complete",
        "superseded_blobs": {**manifest["blobs"], "other-ancestry": active},
    }
    store.commit_checkpoint(PROFILE, checkpoint)
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "state"
    protected_bytes = {ref: (directory / ref).read_bytes() for ref in (other, prior, active)}
    result = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=manifest,
        before_day="2026-10-14",
        blob_names=("counters",),
    )
    assert result.changed and result.manifest is not None
    replacement = result.manifest["blobs"]["counters"]
    assert replacement != prior
    assert store.read_blob(PROFILE, replacement) == {**counters, "counted": []}
    assert checkpoint.identity_migration["superseded_blobs"] == {
        **manifest["blobs"],
        "counters": replacement,
        "other-ancestry": active,
    }
    assert checkpoint.blobs["counters"] == active and checkpoint.blobs[other_name] == other
    store.commit_checkpoint(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    assert {ref: (directory / ref).read_bytes() for ref in protected_bytes} == protected_bytes
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    other_manifest = store.read_blob(PROFILE, restarted.blobs[other_name])
    active_counters = store.read_blob(PROFILE, restarted.blobs["counters"])
    assert other_manifest is not None and active_counters is not None
    assert other_manifest["blobs"]["counters"] == prior
    assert active_counters["epoch"] == 9


@pytest.mark.parametrize(
    "selection",
    [
        {"before_day": "not-a-date", "history": True},
        {"name": "not-an-archive", "history": True},
        {"receipt_ids": tuple(f"r{number}" for number in range(33))},
        {"history": True, "receipt_ids": ("original-0",)},
        {"blob_names": ("first", "second")},
        {"segment_names": ("first", "second")},
    ],
)
def test_invalid_original_retention_quantum_rejects_before_reading_or_writing(
    tmp_path, monkeypatch, selection
):
    from ssgrep.accounting.published_retention import stage_children

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkpoint, rows, manifest = seed()
    root = storage.data_root() / "accounting"
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid expiry selection reached persisted bodies")

    with monkeypatch.context() as guarded:
        guarded.setattr(store, "read_blob", forbidden)
        guarded.setattr(store, "read_segment", forbidden)
        guarded.setattr(store, "write_blob", forbidden)
        guarded.setattr(store, "write_segments", forbidden)
        with pytest.raises(ValueError):
            stage_children(
                PROFILE,
                checkpoint,
                manifest=manifest,
                **{"name": ARCHIVE, "before_day": "2026-10-14", **selection},
            )
    assert {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    accepted = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=manifest,
        before_day="2026-10-14",
        history=True,
    )
    assert accepted.changed and accepted.manifest is not None
    _, retained = read_document(
        PROFILE, accepted.manifest["published_documents"]["history_blob"], "history"
    )
    assert retained == rows[1:]
    receipts = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=accepted.manifest,
        before_day="2026-10-14",
        receipt_ids=("original-0",),
    )
    assert receipts.changed and receipts.expired_receipts == ("original-0",)
    assert receipts.manifest is not None
    assert receipts.manifest["published_documents"]["receipt_blobs"] == {
        "original-1": manifest["published_documents"]["receipt_blobs"]["original-1"]
    }
