"""Canonical root evaluations survive restarts without rerating unrelated roots."""

from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_root_cache import restore_cache, save_cache
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting.valuation_builders import METHOD, candidate, debit


def test_restored_canonical_root_rows_are_identical_without_revaluation(monkeypatch):
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": EvidenceBundle((candidate("a"),), (debit("d", 40),))}, today="2026-11-02"
    )
    expected = ledger.rows().canonical_bytes()
    blobs = {}
    save_cache("personal-local", ledger, blobs)
    restored = VersionedLedger.restore_versioned(ledger.state(), ledger.superseded_state())
    restore_cache("personal-local", restored, blobs)
    import ssgrep.accounting.replay as replay

    actual = replay.value_evidence
    calls = []

    def observed(*args, **kwargs):
        calls.append(args)
        return actual(*args, **kwargs)

    monkeypatch.setattr(replay, "value_evidence", observed)
    assert restored.rows().canonical_bytes() == expected
    assert calls == []


def test_root_cache_keeps_named_archives_and_is_invalidated_by_privacy_purge():
    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": EvidenceBundle((candidate("a"),), ())}, today="2026-11-02")
    ledger.archive_current("original")
    original = ledger.archives_state()
    ledger.rows()
    assert ledger._eval_cache
    ledger.purge(roots=("root-1",))
    assert not ledger._eval_cache
    assert not ledger.rows().receipts
    assert not ledger.archives_state()
    assert original["original"].receipts
