"""Publication capture budgets use real owned files and measured descriptor reads.

These isolate the capture quantum, not legacy admission/native-proof migration;
the private authentic407 two-phase driver exercises those end to end.
"""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from ssgrep.accounting.collect import store
from ssgrep.accounting.ledger.contracts import digest_of
from ssgrep.accounting.privacy import published_capture as capture, storage

PROFILE = "personal-local"
STAMP = "2026-10-14T12:00:00Z"


def setup(monkeypatch, receipts=0, foreign_first=False):
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
    ledger = store.write_blob(
        PROFILE,
        "ledger-" + namespace[:12],
        {
            "state": {"methodology": method, "candidates": [], "debits": [], "frozen": []},
            "superseded": {},
            "collection_scope": {},
        },
    )
    checkpoint = store.Checkpoint(
        PROFILE, generation=3, methodology=namespace, blobs={"ledger-" + namespace[:12]: ledger}
    )
    monkeypatch.setattr(
        capture.snapshots, "read_generation", lambda *_: SimpleNamespace(collected_at=STAMP)
    )
    bodies = {"history": b" [ ] \n"}
    paths = {}
    for sub in ("history", "receipts"):
        directory = storage.open_profile_dir(PROFILE, sub)
        try:
            if sub == "history":
                storage.atomic_write(directory, "periods.json", bodies["history"])
                paths["history"] = (
                    storage.data_root() / "accounting" / PROFILE / sub / "periods.json"
                )
            else:
                for number in range(receipts):
                    identity = f"r{number:03}"
                    row = {
                        "schema": 1,
                        "receipt_id": identity,
                        "revision": 1,
                        "root": "claude:first",
                        "task_turn": "turn",
                        "candidate_type": "identifier",
                        "atoms": [],
                        "allocations": [],
                        "exclusions": [],
                        "costs": [],
                        "methodology": "foreign-method"
                        if number == 0 and foreign_first
                        else namespace,
                        "evidence_available": True,
                        "execution_ids": [],
                        "unit": "context_tokens:chars_div4_v1",
                        "period": "2026-10-12",
                        "missing_costs": [],
                        "atom_digests": [],
                        "credit_raw": [["conservative", 0], ["central", 0], ["optimistic", 0]],
                        "identities": {"frequency": "", "tokenizer": "", "pricing": ""},
                    }
                    bodies[identity] = (json.dumps(row, sort_keys=True) + "\n").encode()
                    storage.atomic_write(directory, identity + ".json", bodies[identity])
                    paths[identity] = (
                        storage.data_root() / "accounting" / PROFILE / sub / (identity + ".json")
                    )
        finally:
            os.close(directory)
    inodes = {(path.stat().st_dev, path.stat().st_ino): name for name, path in paths.items()}
    original_read = os.read
    reads = []

    def observed(descriptor, size):
        value = original_read(descriptor, size)
        info = os.fstat(descriptor)
        name = inodes.get((info.st_dev, info.st_ino))
        if name is not None:
            reads.append((name, len(value)))
        return value

    monkeypatch.setattr(capture.os, "read", observed)
    return checkpoint, bodies, reads, paths


def document(checkpoint):
    captured = store.read_blob(PROFILE, checkpoint.identity_migration["published_capture"])
    assert captured is not None
    return captured


def test_history_budget_blocks_before_body_read_and_resumes_at_exact_boundary(monkeypatch):
    checkpoint, bodies, reads, _ = setup(monkeypatch)
    original_envelope = capture._envelope

    def forbidden(*args, **kwargs):
        raise AssertionError("A blocked history must not write an original-document envelope")

    monkeypatch.setattr(capture, "_envelope", forbidden)
    complete, consumed = capture.stage(PROFILE, checkpoint, byte_budget=len(bodies["history"]) - 1)
    assert not complete and consumed == 0 and reads == []
    saved = document(checkpoint)
    assert saved["history_blob"] is None and saved["after"] == ""
    assert saved["blocked_child_bytes"] == len(bodies["history"])
    restarted = store.Checkpoint.from_dict(checkpoint.to_dict())
    monkeypatch.setattr(capture, "_envelope", original_envelope)
    complete, consumed = capture.stage(PROFILE, restarted, byte_budget=len(bodies["history"]))
    assert complete and consumed == len(bodies["history"])
    assert reads == [("history", len(bodies["history"]))]
    captured = store.read_blob(PROFILE, document(restarted)["history_blob"])
    assert captured is not None
    assert captured["original_utf8"].encode() == bodies["history"]
    assert captured["sha256"] == hashlib.sha256(bodies["history"]).hexdigest()
    assert captured["source_generation"] == 3 and captured["original_collected_at"] == STAMP


def test_history_and_first_receipt_share_one_byte_budget_without_cursor_advance(monkeypatch):
    checkpoint, bodies, reads, _ = setup(monkeypatch, receipts=1)
    envelopes = []
    original_envelope = capture._envelope

    def observed(profile, raw, kind, generation, stamp):
        envelopes.append(kind)
        return original_envelope(profile, raw, kind, generation, stamp)

    monkeypatch.setattr(capture, "_envelope", observed)
    quota = len(bodies["history"]) + len(bodies["r000"]) - 1
    complete, consumed = capture.stage(PROFILE, checkpoint, byte_budget=quota)
    assert not complete and consumed == len(bodies["history"])
    assert reads == [("history", len(bodies["history"]))]
    assert document(checkpoint)["after"] == "" and document(checkpoint)["receipt_blobs"] == {}
    assert envelopes == ["history"]
    complete, consumed = capture.stage(PROFILE, checkpoint, byte_budget=len(bodies["r000"]))
    assert complete and consumed == len(bodies["r000"])
    assert reads[-1] == ("r000", len(bodies["r000"]))
    original = store.read_blob(PROFILE, document(checkpoint)["receipt_blobs"]["r000"])
    assert original is not None
    assert original["original_utf8"].encode() == bodies["r000"]
    assert envelopes == ["history", "receipt"]


def test_combined_history_and_receipt_exact_boundary_captures_both(monkeypatch):
    checkpoint, bodies, reads, _ = setup(monkeypatch, receipts=1)
    quota = len(bodies["history"]) + len(bodies["r000"])
    complete, consumed = capture.stage(PROFILE, checkpoint, byte_budget=quota)
    assert complete and consumed == quota
    assert reads == [("history", len(bodies["history"])), ("r000", len(bodies["r000"]))]
    assert document(checkpoint)["after"] == "r000.json"
    assert set(document(checkpoint)["receipt_blobs"]) == {"r000"}


def test_history_counts_toward_the_32_document_quantum(monkeypatch):
    checkpoint, bodies, reads, _ = setup(monkeypatch, receipts=32)
    complete, consumed = capture.stage(PROFILE, checkpoint, limit=32)
    assert not complete and len(document(checkpoint)["receipt_blobs"]) == 31
    assert len(reads) == 32 and document(checkpoint)["after"] == "r030.json"
    assert consumed == sum(len(bodies[name]) for name, _ in reads)
    complete, consumed = capture.stage(PROFILE, checkpoint, limit=32)
    assert complete and consumed == len(bodies["r031"])


def test_foreign_receipt_still_requires_budget_and_charges_its_bounded_read(monkeypatch):
    checkpoint, bodies, reads, paths = setup(monkeypatch, receipts=2, foreign_first=True)
    quota = len(bodies["history"]) + len(bodies["r000"]) - 1
    complete, consumed = capture.stage(PROFILE, checkpoint, byte_budget=quota)
    assert not complete and consumed == len(bodies["history"])
    assert document(checkpoint)["after"] == "" and all(name != "r000" for name, _ in reads)
    complete, consumed = capture.stage(PROFILE, checkpoint, byte_budget=len(bodies["r000"]))
    assert not complete and consumed == len(bodies["r000"])
    assert (
        document(checkpoint)["after"] == "r000.json" and document(checkpoint)["receipt_blobs"] == {}
    )
    complete, consumed = capture.stage(PROFILE, checkpoint, byte_budget=len(bodies["r001"]))
    assert complete and consumed == len(bodies["r001"])
    assert set(document(checkpoint)["receipt_blobs"]) == {"r001"}
    assert paths["r000"].read_bytes() == bodies["r000"]


def test_malformed_unknown_receipt_is_typed_and_never_skipped_or_enveloped(monkeypatch):
    from ssgrep.accounting.collect.store import CollectError

    checkpoint, _, _, paths = setup(monkeypatch, receipts=1)
    assert not capture.stage(PROFILE, checkpoint, limit=1)[
        0
    ]  # Preserve actual captured history first.
    reference = checkpoint.identity_migration["published_capture"]
    paths["r000"].write_bytes(b"not-json\\n")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "An undecodable potentially owned receipt must not produce an envelope"
        )

    monkeypatch.setattr(capture, "_envelope", forbidden)
    try:
        capture.stage(PROFILE, checkpoint)
    except CollectError as error:
        assert error.condition == "migration_history_unavailable"
    else:
        raise AssertionError("Unknown malformed receipts cannot be skipped into fake completion")
    assert checkpoint.identity_migration["published_capture"] == reference
    assert document(checkpoint)["after"] == "" and not document(checkpoint)["complete"]


@pytest.mark.parametrize("mutation", ["truncate", "grow"])
def test_capture_rejects_concurrent_original_mutation_and_retry_preserves_accepted_values(
    monkeypatch, mutation
):
    from ssgrep.accounting.collect.store import CollectError
    from ssgrep.accounting.privacy.published import read_document
    from tests.accounting.privacy.test_published_schema import GENUINE_HISTORY

    checkpoint, _, _, paths = setup(monkeypatch)
    history = paths["history"]
    history.write_bytes(GENUINE_HISTORY)
    before = store.dumps(checkpoint.to_dict())
    inode = history.stat().st_ino
    real_read = os.read
    mutated = False

    def racing_read(descriptor, size):
        nonlocal mutated
        if os.fstat(descriptor).st_ino != inode or mutated:
            return real_read(descriptor, size)
        mutated = True
        if mutation == "truncate":
            history.write_bytes(b"")
            return real_read(descriptor, size)
        body = real_read(descriptor, size)
        history.write_bytes(GENUINE_HISTORY + b" ")
        return body

    with monkeypatch.context() as race:
        race.setattr(capture.os, "read", racing_read)
        with pytest.raises(CollectError) as failed:
            capture.stage(PROFILE, checkpoint)
    assert failed.value.condition == "migration_history_unavailable"
    assert store.dumps(checkpoint.to_dict()) == before
    state_dir = storage.data_root() / "accounting" / PROFILE / "collector" / "state"
    assert not list(state_dir.glob("published-original-*.json"))

    history.write_bytes(GENUINE_HISTORY)
    complete, consumed = capture.stage(PROFILE, checkpoint)
    assert complete and consumed == len(GENUINE_HISTORY)
    captured, rows = read_document(PROFILE, document(checkpoint)["history_blob"], "history")
    assert rows[0]["net"]["value"] == -78 and rows[0]["period"] == "2026-10-14"
    assert captured["original_utf8"].encode() == GENUINE_HISTORY


def test_capture_refuses_symlinked_original_without_copying_target_bytes(monkeypatch, tmp_path):
    from ssgrep.accounting.collect.store import CollectError
    from tests.accounting.privacy.test_published_schema import GENUINE_HISTORY

    checkpoint, _, _, paths = setup(monkeypatch)
    history = paths["history"]
    private = tmp_path / "not-owned-history.json"
    private.write_bytes(GENUINE_HISTORY)
    history.unlink()
    history.symlink_to(private)
    before = store.dumps(checkpoint.to_dict())
    with pytest.raises(CollectError) as failed:
        capture.stage(PROFILE, checkpoint)
    assert failed.value.condition == "migration_history_unavailable"
    assert store.dumps(checkpoint.to_dict()) == before and private.read_bytes() == GENUINE_HISTORY
    state_dir = storage.data_root() / "accounting" / PROFILE / "collector" / "state"
    assert not list(state_dir.glob("published-original-*.json"))
    history.unlink()
    history.write_bytes(GENUINE_HISTORY)
    history.chmod(0o600)
    assert capture.stage(PROFILE, checkpoint) == (True, len(GENUINE_HISTORY))
    captured = store.read_blob(PROFILE, document(checkpoint)["history_blob"])
    assert captured is not None and captured["original_utf8"].encode() == GENUINE_HISTORY


def test_missing_generation_refuses_capture_without_advancing_checkpoint(monkeypatch):
    from ssgrep.accounting.collect.store import CollectError

    checkpoint, _, _, paths = setup(monkeypatch)
    before = store.dumps(checkpoint.to_dict())
    monkeypatch.setattr(capture.snapshots, "read_generation", lambda *_: None)

    with pytest.raises(CollectError) as caught:
        capture.stage(PROFILE, checkpoint)

    assert caught.value.condition == "migration_history_unavailable"
    assert store.dumps(checkpoint.to_dict()) == before
    assert paths["history"].read_bytes() == b" [ ] \n"


def test_nonreceipt_files_advance_cursor_but_owned_receipt_is_captured(monkeypatch):
    checkpoint, bodies, _, paths = setup(monkeypatch, receipts=1)
    assert capture.stage(PROFILE, checkpoint, limit=1)[0] is False
    receipts = storage.open_profile_dir(PROFILE, "receipts")
    try:
        storage.atomic_write(receipts, "operator-note.txt", b"not a receipt")
    finally:
        os.close(receipts)

    complete, consumed = capture.stage(PROFILE, checkpoint, limit=1)

    assert not complete and consumed == 0
    assert document(checkpoint)["after"] == "operator-note.txt"
    assert paths["r000"].read_bytes() == bodies["r000"]
    complete, consumed = capture.stage(PROFILE, checkpoint, limit=1)
    assert complete and consumed == len(bodies["r000"])
    assert set(document(checkpoint)["receipt_blobs"]) == {"r000"}


def test_receipt_identity_mismatch_keeps_capture_checkpoint_and_public_bytes(monkeypatch):
    from ssgrep.accounting.collect.store import CollectError

    checkpoint, _, _, paths = setup(monkeypatch, receipts=1)
    assert capture.stage(PROFILE, checkpoint, limit=1)[0] is False
    original = json.loads(paths["r000"].read_bytes())
    original["receipt_id"] = "different-id"
    paths["r000"].write_text(json.dumps(original) + "\n")
    before = store.dumps(checkpoint.to_dict())
    malformed = paths["r000"].read_bytes()

    with pytest.raises(CollectError) as caught:
        capture.stage(PROFILE, checkpoint)

    assert caught.value.condition == "migration_history_unavailable"
    assert store.dumps(checkpoint.to_dict()) == before
    assert paths["r000"].read_bytes() == malformed


def test_capture_refuses_oversized_history_before_read_or_checkpoint_advance(monkeypatch):
    from ssgrep.accounting.collect.store import CollectError

    checkpoint, _, reads, paths = setup(monkeypatch)
    before = store.dumps(checkpoint.to_dict())
    monkeypatch.setattr(capture, "MAX_SNAPSHOT_BYTES", 4)

    with pytest.raises(CollectError) as caught:
        capture.stage(PROFILE, checkpoint)

    assert caught.value.condition == "migration_history_unavailable"
    assert reads == []
    assert store.dumps(checkpoint.to_dict()) == before
    assert paths["history"].read_bytes() == b" [ ] \n"


def test_capture_rejects_oversized_receipt_inventory_before_cursor_commit(monkeypatch):
    from ssgrep.accounting.collect.store import CollectError

    checkpoint, _, _, _ = setup(monkeypatch)
    before = store.dumps(checkpoint.to_dict())
    monkeypatch.setattr(capture.storage, "list_names", lambda _fd: ["r000.json"] * 250_001)

    with pytest.raises(CollectError) as caught:
        capture.stage(PROFILE, checkpoint)

    assert caught.value.condition == "migration_history_unavailable"
    assert store.dumps(checkpoint.to_dict()) == before


def test_capture_root_index_requires_day_and_root_provenance() -> None:
    valid = {
        "candidates": [{"day": "2026-10-14", "task": {"root": "claude:alpha"}}],
        "debits": [],
        "frozen": [
            {"root": "claude:beta", "leaves": [{"day": "2026-10-15"}]},
        ],
    }
    assert capture._roots_by_period(valid) == {
        "2026-10-14": ["claude:alpha"],
        "2026-10-15": ["claude:beta"],
    }
    assert (
        capture._roots_by_period({"candidates": [{"day": None, "task": {"root": "claude:alpha"}}]})
        is None
    )
    assert capture._roots_by_period({"frozen": [{"leaves": [{"day": "2026-10-14"}]}]}) is None


def test_captured_documents_requires_complete_migration_and_strips_cursor_metadata():
    from ssgrep.accounting.collect.store import CollectError

    incomplete = store.write_blob(
        PROFILE,
        "published-capture",
        {
            "format": "published-session-v1",
            "history_blob": None,
            "receipt_blobs": {},
            "complete": False,
            "after": "r000.json",
        },
    )
    checkpoint = store.Checkpoint(PROFILE, identity_migration={"published_capture": incomplete})
    with pytest.raises(CollectError) as caught:
        capture.captured_documents(PROFILE, checkpoint)
    assert caught.value.condition == "migration_history_unavailable"

    complete = store.write_blob(
        PROFILE,
        "published-capture",
        {
            "format": "published-session-v1",
            "history_blob": None,
            "receipt_blobs": {},
            "complete": True,
            "after": "r000.json",
        },
    )
    checkpoint.identity_migration["published_capture"] = complete
    assert capture.captured_documents(PROFILE, checkpoint) == {
        "format": "published-session-v1",
        "history_blob": None,
        "receipt_blobs": {},
    }


def test_complete_capture_stage_is_idempotent_without_rereading_sources(monkeypatch):
    checkpoint, _, reads, _ = setup(monkeypatch)
    assert capture.stage(PROFILE, checkpoint) == (True, len(b" [ ] \n"))
    first_reads = list(reads)

    assert capture.stage(PROFILE, checkpoint) == (True, 0)

    assert reads == first_reads


def test_capture_rejects_nonobject_receipt_without_committing_progress(monkeypatch):
    from ssgrep.accounting.collect.store import CollectError

    checkpoint, _, _, paths = setup(monkeypatch, receipts=1)
    assert capture.stage(PROFILE, checkpoint, limit=1)[0] is False
    paths["r000"].write_bytes(b"[]\n")
    before = store.dumps(checkpoint.to_dict())

    with pytest.raises(CollectError) as caught:
        capture.stage(PROFILE, checkpoint)

    assert caught.value.condition == "migration_history_unavailable"
    assert store.dumps(checkpoint.to_dict()) == before
    assert paths["r000"].read_bytes() == b"[]\n"
