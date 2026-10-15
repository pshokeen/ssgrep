"""Compact references replay exact originals, never changed text or zeroed accounting."""

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.evidence_replay import EvidenceReplay
from tests.accounting.test_minimal_evidence import PROFILE
from tests.accounting.test_native_source import old_source


def target(adapter):
    line = adapter.resolve("claude:src1:ep:0").line
    return next(record.ev for record in line.records if record.kind == "tool_result")


def test_compact_replay_reads_exact_authorized_original_and_keeps_call_link(tmp_path):
    path, _, adapter = old_source(tmp_path)
    item = target(adapter)
    compact = store.evidence_from_line(store.evidence_line(item))
    before = path.read_bytes()
    adapter.bytes_read = 0
    replay = EvidenceReplay(PROFILE, adapter=adapter)
    restored = replay.hydrate(compact)
    assert restored.payload["output"] == item.payload["output"]
    assert restored.payload.get("call_record_id") == item.payload.get("call_record_id")
    assert restored.identity == item.identity
    assert replay.unavailable == {}
    assert path.read_bytes() == before


def test_deleted_original_preserves_compact_checkpoint_and_reports_replay_limit(tmp_path):
    path, _, adapter = old_source(tmp_path)
    item = target(adapter)
    refs = store.write_segments(PROFILE, [item])
    checkpoint = store.Checkpoint(
        PROFILE,
        generation=8,
        segments=refs,
        blobs={"accepted": store.write_blob(PROFILE, "accepted", {"tokens": 729})},
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    path.unlink()
    replay = EvidenceReplay(PROFILE, adapter=adapter)
    restored = replay.hydrate(next(store.iter_evidence(PROFILE, refs)))
    assert restored.payload["_replay_unavailable"] == "source_missing"
    assert replay.unavailable[(item.runtime, item.session_id)] == "source_missing"
    committed = store.load_checkpoint(PROFILE)
    assert committed is not None
    assert committed.generation == 8
    assert store.read_blob(PROFILE, committed.blobs["accepted"]) == {"tokens": 729}


def test_changed_original_never_substitutes_text_under_old_digest(tmp_path):
    path, _, adapter = old_source(tmp_path)
    item = target(adapter)
    compact = store.evidence_from_line(store.evidence_line(item))
    path.write_bytes(
        path.read_bytes().replace(item.payload["output"].encode(), b"changed-private-source")
    )
    replay = EvidenceReplay(PROFILE, adapter=adapter)
    restored = replay.hydrate(compact)
    assert restored.digest == item.digest
    assert "output" not in restored.payload
    assert restored.payload["_replay_unavailable"] == "source_digest_drift"


def test_actual_cycle_keeps_accepted_rows_after_original_deletion(tmp_path):
    from pathlib import Path

    from ssgrep.accounting.collector_state import load_state
    from tests.accounting import collector_scene as scene, verify_scenes as vs
    from tests.accounting.test_collect_read import NOW, seeded

    seeded(tmp_path)
    checkpoint = scene.committed()
    before = load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert before.ledger is not None and before.ledger.debits
    for source in checkpoint.sources.values():
        path = Path(source.source_id.split(":", 1)[1])
        assert path.is_relative_to(tmp_path)
        path.unlink()
    vs.filler(vs.projects(tmp_path), "new-input", NOW, 1)
    scene.collect(NOW)
    after_checkpoint = scene.committed()
    after = load_state(PROFILE, after_checkpoint, after_checkpoint.methodology)
    assert after.ledger is not None
    assert after.reduction is not None
    assert after.ledger.candidates == before.ledger.candidates
    assert after.ledger.debits == before.ledger.debits
    assert any("replay" in key or "source_missing" in key for key in after.reduction.coverage)


def test_optional_cache_verifies_content_not_only_stored_digest_field(tmp_path):
    import json

    from ssgrep.accounting import policy, storage
    from ssgrep.accounting.evidence_cache import cache_collected_evidence, cached_evidence

    _, _, adapter = old_source(tmp_path)
    item = target(adapter)
    policy.enable_collection(PROFILE, evidence_cache=(1, 1024 * 1024))
    cache_collected_evidence(PROFILE, [item])
    refs = store.write_segments(PROFILE, [item])
    compact = next(store.iter_evidence(PROFILE, refs))
    cached = cached_evidence(PROFILE, compact)
    assert cached is not None
    assert cached.payload["output"] == item.payload["output"]
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    path = next(directory.iterdir())
    raw = json.loads(path.read_bytes())
    raw["payload"]["output"] = "changed-cache-private-canary"
    path.write_text(json.dumps(raw))
    assert cached_evidence(PROFILE, compact) is None
    replay = EvidenceReplay(PROFILE, adapter=adapter)
    assert replay.hydrate(compact).payload["output"] == item.payload["output"]
