"""Receipt/history writers and snapshot coverage blocks, read back through the facade."""

from __future__ import annotations

import os
import stat
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import collector_publish as publish, service, snapshots, storage
from ssgrep.accounting.collector_store import SourceRecord, source_key
from ssgrep.accounting.collector_types import RetentionCoverage
from ssgrep.accounting.contracts import (
    Attribution,
    Basis,
    Exposure,
    Outcome,
    Quantity,
    Reuse,
    TaskId,
    canonical_bytes,
)
from ssgrep.accounting.counters import empty_counters
from ssgrep.accounting.reduce_types import ExecutionRecord, Reduction
from ssgrep.accounting.replay import Ledger, LedgerRows
from ssgrep.accounting.service_types import Scope
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from ssgrep.accounting.windows import build_windows
from tests.accounting.snapshot_builders import enable_policy
from tests.accounting.valuation_builders import CHARS, METHOD, atom, candidate, debit

PROFILE = "personal-local"
COLLECTED = "2026-11-07T12:00:00Z"
NOW = datetime(2026, 11, 7, 12, 0, 0, tzinfo=UTC)
CONTENT_CANARY = "SECRET-CONTENT-CANARY-9d41"
LOCATOR_CANARY = "witness:/home/canary-user/private/witness.jsonl#L7"
T2 = TaskId("root-2", "turn-1", "main")


def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def profile_dir(tmp_path: Path, *parts: str) -> Path:
    return data_dir(tmp_path) / "accounting" / PROFILE / Path(*parts)


def ledger() -> Ledger:
    """Periods 11-01 (two revisions: late evidence) and 11-05 (negative net)."""
    ledger = Ledger(METHOD)
    ledger.ingest(
        EvidenceBundle(
            (
                candidate("c1", day="2026-11-01", immediate=(atom("s1", 400, 0, "ev-1"),)),
                candidate(
                    "c2",
                    task=T2,
                    day="2026-11-05",
                    exposure_ids=("e2",),
                    immediate=(atom("s3", 400, 0, "ev-3"),),
                ),
            ),
            (debit("x2", 4000, task=T2, day="2026-11-05", exposure_id="e2"),),
        )
    )
    ledger.ingest(
        EvidenceBundle(
            (
                candidate(
                    "c1b",
                    day="2026-11-01",
                    accepted_day="2026-11-20",
                    immediate=(atom("s2", 800, 0, "ev-2"),),
                ),
            ),
            (),
        )
    )
    return ledger


def source(source_id: str, **fields: Any) -> SourceRecord:
    offset = fields.pop("offset", 0)
    return SourceRecord(
        runtime=fields.pop("runtime", "claude"),
        source_id=source_id,
        state={"offset": offset},
        **fields,
    )


def reduction(**coverage: int) -> Reduction:
    return Reduction(profile=PROFILE, coverage=dict(coverage))


def coverage_of(sources: dict[str, SourceRecord], **overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "reduction": reduction(),
        "sources": sources,
        "journal": {"segments": 2, "incomplete_tails": 0, "quarantined": 0, "conflicts": 0},
        "read_gaps": {},
        "dropped": {},
        "stopped_by": "complete",
        "retention": RetentionCoverage(),
        "index": "available",
        "lag_bytes": 0,
    }
    args.update(overrides)
    return publish.snapshot_coverage(**args)


def keyed(*records: SourceRecord) -> dict[str, SourceRecord]:
    return {r.key: r for r in records}


# ---- receipts ------------------------------------------------------------------------------


def test_receipt_round_trips_through_the_read_facade_without_content_or_locators(
    tmp_path: Path,
) -> None:
    rows = ledger().rows()
    record = next(r for r in rows.receipts if r.receipt.root == "root-2")
    private = replace(
        record,
        content=CONTENT_CANARY,
        witness_locators=(LOCATOR_CANARY,),
        action_locators=(LOCATOR_CANARY,),
    )
    rows = replace(rows, receipts=tuple(private if r is record else r for r in rows.receipts))

    assert publish.write_receipts(PROFILE, rows) == 3  # c1, c1b, c2: one file each

    stored = (profile_dir(tmp_path, "receipts") / f"{record.receipt_id}.json").read_bytes()
    assert CONTENT_CANARY.encode() not in stored
    assert b"canary-user" not in stored and b"locator" not in stored
    assert b'"content"' not in stored
    view = service.read_receipt(PROFILE, record.receipt_id, include_content=True)
    assert view is not None
    assert view.content is None  # the writer never stores content, even when asked for it
    assert (view.revision, view.root, view.task_turn) == (1, "root-2", "turn-1")
    assert view.candidate_type == "path"
    assert view.atoms == record.receipt.atoms and view.atoms
    assert view.allocations == tuple((s, tuple(a)) for s, a in record.receipt.allocations)
    assert dict(view.costs)["exposure:x2"].value == 1000
    assert dict(view.costs)["exposure:x2"].basis is Basis.ESTIMATED
    assert view.methodology == METHOD.hash
    assert view.evidence_available is True and view.note is None
    document = publish.receipt_document(record)
    assert document["credit_raw"] == [["conservative", 400], ["central", 400], ["optimistic", 400]]
    assert document["unit"] == "context_tokens:chars_div4_v1"
    assert document["period"] == "2026-11-05"
    assert set(document["identities"]) == {"frequency", "tokenizer", "pricing"}


def test_only_the_latest_receipt_revision_is_written(tmp_path: Path) -> None:
    led = Ledger(METHOD)
    led.ingest(
        EvidenceBundle(
            (candidate("c1", day="2026-11-01", exposure_ids=("e1",)),),
            (debit("x1", None, exposure_id="e1"),),
        )
    )
    led.ingest(
        EvidenceBundle((), (debit("x2", 4000, exposure_id="e1", accepted_day="2026-11-20"),))
    )
    rows = led.rows()
    assert [r.receipt.revision for r in rows.receipts] == [1, 2]  # precondition: two revisions
    receipt_id = rows.receipts[0].receipt_id

    assert publish.write_receipts(PROFILE, rows) == 1

    view = service.read_receipt(PROFILE, receipt_id)
    assert view is not None and view.revision == 2
    costs = dict(view.costs)
    assert costs["exposure:x2"].value == 1000
    assert costs["exposure:x1"].value is None and costs["exposure:x1"].basis is Basis.UNAVAILABLE
    stored = (profile_dir(tmp_path, "receipts") / f"{receipt_id}.json").read_bytes()
    assert b"downstream_exposure_unobserved" in stored  # missing_costs reason is kept
    assert [p.name for p in profile_dir(tmp_path, "receipts").iterdir()] == [f"{receipt_id}.json"]


def test_receipt_writes_are_idempotent_and_rewrite_only_changed_receipts(tmp_path: Path) -> None:
    rows = ledger().rows()
    assert publish.write_receipts(PROFILE, rows) == 3
    path = profile_dir(tmp_path, "receipts")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}

    assert publish.write_receipts(PROFILE, rows) == 0
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()} == before

    record = next(r for r in rows.receipts if r.receipt.root == "root-2")
    bumped = replace(record, receipt=replace(record.receipt, revision=record.receipt.revision + 1))
    updated = replace(rows, receipts=(*rows.receipts, bumped))  # exactly one receipt changes
    assert publish.write_receipts(PROFILE, updated) == 1
    view = service.read_receipt(PROFILE, record.receipt_id)
    assert view is not None and view.revision == bumped.receipt.revision
    assert publish.write_receipts(PROFILE, updated) == 0


def test_hostile_receipt_ids_are_skipped_and_counted_never_written(tmp_path: Path) -> None:
    rows = ledger().rows()
    good = rows.receipts[0]
    hostile = [
        replace(good, receipt=replace(good.receipt, receipt_id=bad))
        for bad in ("../../evil", "a/b", "", ".hidden", "x" * 80)
    ]
    only_hostile = replace(rows, receipts=tuple(hostile))
    assert publish.unwritable_receipts(only_hostile) == 5
    assert publish.write_receipts(PROFILE, only_hostile) == 0
    assert not profile_dir(tmp_path, "receipts").exists()  # nothing created for nothing written

    mixed = replace(rows, receipts=(good, *hostile))
    assert publish.unwritable_receipts(mixed) == 5
    assert publish.write_receipts(PROFILE, mixed) == 1
    assert sorted(p.name for p in profile_dir(tmp_path, "receipts").iterdir()) == [
        f"{good.receipt_id}.json"
    ]
    assert not (data_dir(tmp_path) / "evil.json").exists()


def test_receipt_files_are_private(tmp_path: Path) -> None:
    publish.write_receipts(PROFILE, ledger().rows())
    receipts = profile_dir(tmp_path, "receipts")
    assert stat.S_IMODE(receipts.stat().st_mode) == 0o700
    files = list(receipts.iterdir())
    assert files and all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in files)


def test_frozen_root_receipt_says_evidence_expired_and_never_returns_content(
    tmp_path: Path,
) -> None:
    led = ledger()
    assert led.expire_roots(["root-1"]) == ["root-1"]
    rows = led.rows()
    frozen = [r for r in rows.receipts if r.receipt.root == "root-1"]
    assert frozen and all(not r.receipt.evidence_available for r in frozen)

    publish.write_receipts(PROFILE, rows)

    view = service.read_receipt(PROFILE, frozen[0].receipt_id, include_content=True)
    assert view is not None
    assert view.evidence_available is False
    assert view.content is None
    assert view.note is not None and view.note.startswith("evidence_expired")
    stored = (profile_dir(tmp_path, "receipts") / f"{frozen[0].receipt_id}.json").read_bytes()
    assert b'"missing_costs":[["recomputation","evidence_expired"]]' in stored
    live = next(r for r in rows.receipts if r.receipt.root == "root-2")
    live_view = service.read_receipt(PROFILE, live.receipt_id)
    assert live_view is not None and live_view.evidence_available is True


# ---- representative receipt ------------------------------------------------------------------


def synthetic_rows(*spec: tuple[str, int, dict[str, int], bool]) -> LedgerRows:
    """(id, revision, {scenario: credit_raw}, has_costs) records over real receipt shapes."""
    base = ledger().rows().receipts
    paying = next(r for r in base if r.receipt.costs)
    bare = next(r for r in base if not r.receipt.costs)
    records = []
    for receipt_id, revision, credit, has_costs in spec:
        proto = paying if has_costs else bare
        records.append(
            replace(
                proto,
                receipt=replace(proto.receipt, receipt_id=receipt_id, revision=revision),
                credit_raw=tuple(credit.items()),
            )
        )
    return LedgerRows("ns", (), tuple(records))


def test_representative_receipt_prefers_largest_positive_central_credit() -> None:
    rows = synthetic_rows(
        ("a-small", 1, {"central": 50}, True),
        ("b-big", 1, {"central": 900}, False),
        ("c-big-tie", 1, {"central": 900}, True),
        ("d-zero", 1, {"central": 0}, True),
    )
    assert publish.representative_receipt(rows) == "b-big"  # ties break by id


def test_representative_receipt_uses_only_the_latest_revision() -> None:
    rows = synthetic_rows(
        ("r1", 1, {"central": 5000}, True),  # superseded: must not count
        ("r1", 2, {"central": 10}, True),
        ("r2", 1, {"central": 300}, True),
    )
    assert publish.representative_receipt(rows) == "r2"


def test_representative_receipt_falls_back_to_first_costed_id_then_none() -> None:
    no_credit = synthetic_rows(
        ("zz", 1, {"central": 0}, True),
        ("mm", 1, {"central": 0}, True),
        ("aa", 1, {"central": 0}, False),
    )
    assert publish.representative_receipt(no_credit) == "mm"
    assert publish.representative_receipt(synthetic_rows(("aa", 1, {"central": 0}, False))) is None
    assert publish.representative_receipt(LedgerRows("ns", (), ())) is None


# ---- history ---------------------------------------------------------------------------------


def test_history_shows_both_revisions_and_the_total_counts_only_the_newest(
    tmp_path: Path,
) -> None:
    rows = ledger().rows()
    document = publish.history_document(rows, CHARS, "2026-11-21")
    assert [(d["period"], d["revision"], d["status"]) for d in document] == [
        ("2026-11-01", 1, "superseded"),
        ("2026-11-01", 2, "sealed"),
        ("2026-11-05", 1, "sealed"),
    ]
    assert document[1]["supersedes"] == 1 and document[1]["kind"] == "adjustment"
    assert document[0]["supersedes"] is None and document[0]["kind"] == "original"
    assert all(d["recomputable"] is True for d in document)
    assert publish.write_history(PROFILE, document) is True

    view = service.read_history(PROFILE)

    nets = {(e.period, e.revision): (e.net.value, e.current) for e in view.entries}
    assert nets == {
        ("2026-11-01", 1): (100, False),  # one 400-char atom -> 100 tokens
        ("2026-11-01", 2): (300, True),  # late evidence: 100 + 200
        ("2026-11-05", 1): (-900, True),  # negative net survives: 100 credit, 1000 debit
    }
    assert view.current_total.value == 300 - 900  # superseded 100 is never added
    assert view.current_total.basis is Basis.ESTIMATED
    assert view.reason is None


def test_history_status_is_provisional_before_the_seal_day() -> None:
    rows = ledger().rows()
    early = publish.history_document(rows, CHARS, "2026-11-06")
    by_key = {(d["period"], d["revision"]): d["status"] for d in early}
    assert by_key[("2026-11-05", 1)] == "provisional"
    assert by_key[("2026-11-01", 1)] == "superseded"  # superseded wins over the clock


def test_history_writes_are_idempotent_and_replace_on_change(tmp_path: Path) -> None:
    document = publish.history_document(ledger().rows(), CHARS, "2026-11-21")
    assert publish.write_history(PROFILE, document) is True
    path = profile_dir(tmp_path, "history", "periods.json")
    first = path.read_bytes()
    assert publish.write_history(PROFILE, document) is False
    assert path.read_bytes() == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    shorter = document[:1]
    assert publish.write_history(PROFILE, shorter) is True
    assert path.read_bytes() != first
    assert len(service.read_history(PROFILE).entries) == 1


def test_history_marks_frozen_periods_as_not_recomputable() -> None:
    led = ledger()
    led.expire_roots(["root-1"])
    document = publish.history_document(led.rows(), CHARS, "2026-11-21")
    flags = {d["period"]: d["recomputable"] for d in document}
    assert flags == {"2026-11-01": False, "2026-11-05": True}
    frozen = next(d for d in document if d["period"] == "2026-11-01")
    assert frozen["status"] in {"sealed", "provisional", "superseded"}  # field carries the flag
    assert "recomputable" not in frozen["status"]


def test_history_of_a_unit_the_period_never_had_is_unavailable_not_zero() -> None:
    from ssgrep.accounting.units import WORK_SECONDS, UnitSpec

    document = publish.history_document(
        ledger().rows(), UnitSpec(WORK_SECONDS, "monotonic_v1"), "2026-11-21"
    )
    assert document and all(d["net"]["value"] is None for d in document)
    assert all(d["net"]["basis"] == "unavailable" and d["net"]["reason"] for d in document)


# ---- snapshot coverage -----------------------------------------------------------------------


def test_coverage_numbers_for_a_hand_built_source_set() -> None:
    done = source("claude:/h/a.jsonl", size=1000, offset=1000, complete=True)
    more = source("claude:/h/b.jsonl", size=5000, offset=1000, more=True, runtime="claude")
    torn = source("codex:/h/c.jsonl", size=300, offset=250, pending_tail=50, runtime="codex")
    gone = source("pi:/h/d.jsonl", size=200, offset=200, status="missing", runtime="pi")
    skipped = source(
        "codex:/h/e.jsonl",
        size=700,
        offset=700,
        baseline="skipped",
        skipped_bytes=700,
        complete=True,
        runtime="codex",
        resets=2,
    )
    cov = coverage_of(keyed(done, more, torn, gone, skipped), lag_bytes=4050)

    assert cov["complete"] is False and cov["partial"] is True
    assert cov["capture_completeness"] == "unknown"
    assert cov["sources"]["total"] == 5
    assert cov["sources"]["complete"] == 2  # done + baseline-skipped complete
    assert cov["sources"]["lag_bytes"] == 4000 + 50  # unread suffix incl. the torn tail
    assert cov["sources"]["pending_tail_bytes"] == 50
    assert cov["sources"]["missing"] == 1
    assert cov["sources"]["baseline_skipped_bytes"] == 700
    assert cov["lag_bytes"] == 4050 and cov["resets"] == 2
    assert cov["sources"]["by_runtime"] == {
        "claude": {
            "sources": 2,
            "processed_bytes": 2000,
            "total_bytes": 6000,
            "lag_bytes": 4000,
            "missing": 0,
        },
        "codex": {
            "sources": 2,
            "processed_bytes": 250 + 700,
            "total_bytes": 1000,
            "lag_bytes": 50,
            "missing": 0,
        },
        "pi": {
            "sources": 1,
            "processed_bytes": 200,
            "total_bytes": 200,
            "lag_bytes": 0,
            "missing": 1,
        },
    }
    marks = {m["source"]: m for m in cov["watermarks"]}
    assert [m["source"] for m in cov["watermarks"]] == sorted(marks)
    assert marks[more.key] == {
        "source": more.key,
        "runtime": "claude",
        "offset": 1000,
        "size": 5000,
        "lag_bytes": 4000,
        "pending_tail_bytes": 0,
        "status": "ok",
    }
    assert marks[torn.key]["pending_tail_bytes"] == 50 and marks[torn.key]["lag_bytes"] == 50
    assert marks[gone.key]["status"] == "missing"
    assert marks[skipped.key]["lag_bytes"] == 0 and marks[skipped.key]["offset"] == 700


def test_a_fully_processed_set_is_complete_but_still_never_claims_real_world_capture() -> None:
    cov = coverage_of(
        keyed(
            source("claude:/h/a.jsonl", size=10, offset=10, complete=True),
            source("omp:/h/b.jsonl", size=0, offset=0, complete=True, runtime="omp"),
        ),
        reduction=reduction(recorded_lookups=3, attributed_lookups=3),
    )
    assert cov["complete"] is True and cov["partial"] is False
    assert cov["capture_completeness"] == "unknown"
    assert cov["sources"]["lag_bytes"] == 0 and cov["stopped_by"] == "complete"


@pytest.mark.parametrize(
    "mutate",
    [
        {"stopped_by": "byte_budget"},
        {"stopped_by": "time_budget"},
        {"journal": {"segments": 1, "incomplete_tails": 1, "quarantined": 0, "conflicts": 0}},
    ],
)
def test_budget_stop_or_journal_tail_makes_the_snapshot_incomplete(mutate: dict[str, Any]) -> None:
    done = keyed(source("claude:/h/a.jsonl", size=10, offset=10, complete=True))
    assert coverage_of(done)["complete"] is True
    cov = coverage_of(done, **mutate)
    assert cov["complete"] is False and cov["partial"] is True


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "missing"},
        {"status": "unreadable"},
        {"more": True},
        {"pending_tail": 4},
        {"complete": False},
    ],
)
def test_any_unfinished_source_makes_the_snapshot_incomplete(bad: dict[str, Any]) -> None:
    fields: dict[str, Any] = {"size": 10, "offset": 10, "complete": True} | bad
    cov = coverage_of(keyed(source("claude:/h/a.jsonl", **fields)))
    assert cov["complete"] is False and cov["partial"] is True
    assert cov["sources"]["complete"] == 0


def test_partial_is_driven_by_reducer_coverage_even_when_complete() -> None:
    done = keyed(source("claude:/h/a.jsonl", size=10, offset=10, complete=True))
    base = {"recorded_lookups": 10, "attributed_lookups": 10}
    assert coverage_of(done, reduction=reduction(**base))["partial"] is False
    for extra in (
        {"unavailable_sources": 1},
        {"ambiguous_joins": 2},
        {"attributed_lookups": 9},
    ):
        cov = coverage_of(done, reduction=reduction(**{**base, **extra}))
        assert cov["complete"] is True and cov["partial"] is True, extra
    zero = coverage_of(done, reduction=reduction(**base, unavailable_sources=0, ambiguous_joins=0))
    assert (
        zero["partial"] is False
        and zero["unavailable_sources"] == 0
        and zero["ambiguous_joins"] == 0
    )


def test_lookup_counters_and_exclusions_come_from_the_reducer_by_name_and_missing_is_omitted() -> (
    None
):
    cov = coverage_of(
        {},
        reduction=reduction(
            recorded_lookups=112,
            attributed_lookups=91,
            candidate_receipts=13,
            source_costs_available=8,
            ambiguous_joins=3,
            unavailable_sources=2,
            explicit=70,  # a reducer denominator claim that is not a service counter
            **{"exclusion.prior_known": 4, "exclusion.user_supplied": 2, "exclusion.x": 0},
        ),
    )
    for name, value in (
        ("recorded_lookups", 112),
        ("attributed_lookups", 91),
        ("candidate_receipts", 13),
        ("source_costs_available", 8),
        ("ambiguous_joins", 3),
        ("unavailable_sources", 2),
    ):
        assert cov[name] == value
    assert cov["exclusions"] == {"prior_known": 4, "user_supplied": 2, "x": 0}
    assert "explicit" not in cov
    sparse = coverage_of({}, reduction=reduction(recorded_lookups=5))
    assert sparse["recorded_lookups"] == 5
    for absent in (
        "attributed_lookups",
        "candidate_receipts",
        "source_costs_available",
        "ambiguous_joins",
        "unavailable_sources",
    ):
        assert absent not in sparse  # missing key is omitted, never invented as 0
    assert sparse["exclusions"] == {}


def test_gaps_merge_read_gaps_journal_findings_and_prefixed_drops() -> None:
    cov = coverage_of(
        {},
        read_gaps={"oversized_record": 2, "malformed_json": 0, "reset_truncated": 1},
        journal={"segments": 3, "incomplete_tails": 1, "quarantined": 4, "conflicts": 0},
        dropped={"duplicate": 7, "conflict": 1, "kind_not_consumed": 0},
    )
    assert cov["gaps"] == {
        "dropped_conflict": 1,
        "dropped_duplicate": 7,
        "journal_incomplete_tail": 1,
        "journal_quarantined": 4,
        "oversized_record": 2,
        "reset_truncated": 1,
    }
    assert cov["journal"] == {
        "segments": 3,
        "incomplete_tails": 1,
        "quarantined": 4,
        "conflicts": 0,
    }
    clean = coverage_of({})
    assert clean["gaps"] == {}


def test_retention_and_index_are_reported_verbatim() -> None:
    cov = coverage_of(
        {},
        retention=RetentionCoverage("2026-10-01", "2025-10-01", 3),
        index="unavailable",
    )
    assert cov["retention"] == {
        "evidence_expired_before": "2026-10-01",
        "aggregates_expired_before": "2025-10-01",
        "frozen_roots": 3,
    }
    assert cov["index"] == "unavailable"
    assert coverage_of({})["retention"]["evidence_expired_before"] is None


def test_coverage_is_deterministic_and_float_free() -> None:
    records = [
        source(f"claude:/h/{n}.jsonl", size=100 * n, offset=10 * n, more=n % 2 == 0)
        for n in range(1, 8)
    ]
    forward = canonical_bytes(coverage_of(keyed(*records)))  # raises on any float
    backward = canonical_bytes(coverage_of(keyed(*reversed(records))))
    assert forward == backward


def test_hostile_source_ids_never_appear_in_any_output(monkeypatch: pytest.MonkeyPatch) -> None:
    canary = "/home/victim-canary/.claude/projects/secret-project/session.jsonl"
    hostile = source(f"claude:{canary}", size=500, offset=100, more=True)
    other = source("codex:/Users/other-canary/sessions/s.jsonl", size=9, offset=9, complete=True)
    sources = keyed(hostile, other)
    seen: list[bytes] = []
    real = publish.digest_of
    monkeypatch.setattr(
        publish, "digest_of", lambda value: seen.append(canonical_bytes(value)) or real(value)
    )

    cov = coverage_of(sources)
    digest = publish.source_manifest_digest(sources)
    counters = publish.snapshot_observations(
        counters=empty_counters(1, "2026-11-01T00:00:00Z"), backfills=[]
    )

    outputs = [canonical_bytes(cov), digest.encode(), canonical_bytes(counters), *seen]
    assert seen, "the digest must have hashed something"
    for blob in outputs:
        for forbidden in (
            b"victim-canary",
            b"other-canary",
            b"/home/",
            b"/Users/",
            b".jsonl",
            b"secret-project",
        ):
            assert forbidden not in blob
    assert hostile.key.encode() in canonical_bytes(cov)  # only the opaque hash identifies it
    assert hostile.key == source_key(hostile.source_id)


# ---- manifest digest and coverage_through -----------------------------------------------------


def test_manifest_digest_tracks_source_state_but_not_clocks_or_paths() -> None:
    base = source(
        "claude:/h/a.jsonl", size=100, offset=40, mtime_ns=1, last_ts="2026-11-01T00:00:00Z"
    )
    digest = publish.source_manifest_digest(keyed(base))
    assert digest == publish.source_manifest_digest(keyed(base))
    clocks = replace(base, mtime_ns=999, last_ts="2026-12-24T00:00:00Z", resets=3, gaps={"x": 1})
    assert publish.source_manifest_digest(keyed(clocks)) == digest
    for changed in (
        source("claude:/h/a.jsonl", size=101, offset=40),
        source("claude:/h/a.jsonl", size=100, offset=41),
        source("claude:/h/a.jsonl", size=100, offset=40, status="missing"),
        source("claude:/h/a.jsonl", size=100, offset=40, baseline="skipped"),
        source("claude:/h/a.jsonl", size=100, offset=40, runtime="codex"),
        source("claude:/h/other.jsonl", size=100, offset=40),
    ):
        assert publish.source_manifest_digest(keyed(changed)) != digest
    two = keyed(base, source("pi:/h/b.jsonl", size=1, offset=1, runtime="pi"))
    assert publish.source_manifest_digest(two) == publish.source_manifest_digest(
        dict(reversed(list(two.items())))
    )
    assert publish.source_manifest_digest({}) == publish.source_manifest_digest({})


def test_coverage_through_is_the_latest_evidence_instant_capped_at_now() -> None:
    fallback = "2026-11-07T12:00:00Z"
    sources = keyed(
        source("a", last_ts="2026-11-06T10:00:00Z"),
        source("b", last_ts="2026-11-06T23:59:59Z"),
        source("c", last_ts=None),
    )
    assert publish.coverage_through(sources, None, fallback) == "2026-11-06T23:59:59Z"
    assert (
        publish.coverage_through(sources, "2026-11-07T01:00:00Z", fallback)
        == "2026-11-07T01:00:00Z"
    )
    assert (
        publish.coverage_through(sources, "2026-11-07T01:00:00+02:00", fallback)
        == "2026-11-06T23:59:59Z"
    )
    assert (
        publish.coverage_through(keyed(source("a", last_ts="2027-01-01T00:00:00Z")), None, fallback)
        == fallback
    )
    assert publish.coverage_through({}, None, fallback) == fallback
    assert (
        publish.coverage_through(keyed(source("a", last_ts="garbage")), "also garbage", fallback)
        == fallback
    )
    assert publish.coverage_through({}, "2026-11-02T00:00:00Z", fallback) == "2026-11-02T00:00:00Z"


# ---- observations ----------------------------------------------------------------------------


def test_observations_carry_counters_epoch_and_backfills() -> None:
    counters = empty_counters(4, "2026-11-01T00:00:00Z")
    series = {"since": "2026-10-01T00:00:00Z", "until": "2026-10-02T00:00:00Z", "counts": []}
    out = publish.snapshot_observations(counters=counters, backfills=[series])
    assert out["epoch"] == {"counter_epoch": 4, "started_at": "2026-11-01T00:00:00Z"}
    assert out["counters"]["epoch"] == 4
    assert out["backfill"] == [series] and out["backfill"][0] is not series
    bare = publish.snapshot_observations(counters=None, backfills=[])
    assert bare == {"backfill": []}
    canonical_bytes(out)  # float-free


# ---- end to end through the read facade --------------------------------------------------------


def execution(hours_ago: int, operation: str, outcome: Outcome = Outcome.OK) -> ExecutionRecord:
    ts = datetime.fromtimestamp(NOW.timestamp() - hours_ago * 3600, UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return ExecutionRecord(
        execution_id=f"x-{hours_ago}-{operation}",
        ts_utc=ts,
        day=ts[:10],
        operation=operation,
        transport="cli",
        outcome=outcome,
        outcome_basis="journal",
        attribution=Attribution.UNATTRIBUTED,
        source="journal",
        runtime="claude",
        session_id=None,
        invocation_id=None,
        actor="agent",
        task=None,
        rendered_bytes=Quantity.observed(100, "bytes"),
        visible_bytes=Quantity.observed(100, "bytes"),
        exposure=Exposure.OBSERVED,
        reuse=Reuse.NONE_OBSERVED,
        timing_known=True,
        in_scope=True,
    )


def test_published_snapshot_is_read_back_exactly_by_the_summary_and_explain_paths(
    tmp_path: Path,
) -> None:
    enable_policy()
    led = ledger()
    led.expire_roots(["root-1"])  # one frozen root: explain must say the evidence expired
    rows = led.rows()
    publish.write_receipts(PROFILE, rows)
    publish.write_history(PROFILE, publish.history_document(rows, CHARS, "2026-11-21"))
    representative = publish.representative_receipt(rows)
    assert representative is not None

    hostile = source(
        "claude:/home/victim-canary/p/s.jsonl",
        size=900,
        offset=300,
        more=True,
        last_ts="2026-11-07T11:00:00Z",
    )
    sources = keyed(hostile)
    windows = build_windows(
        executions=[
            execution(1, "search"),
            execution(2, "search"),
            execution(3, "show", Outcome.ERROR),
        ],
        sealed={},
        rows=rows,
        retention=RetentionCoverage(frozen_roots=1),
        now=NOW,
        collection_since_day="2026-10-01",
        evidence_horizon_days=30,
    )
    coverage = publish.snapshot_coverage(
        reduction=reduction(
            recorded_lookups=112,
            attributed_lookups=91,
            candidate_receipts=13,
            source_costs_available=8,
            ambiguous_joins=3,
            unavailable_sources=2,
            **{"exclusion.prior_known": 4, "exclusion.user_supplied": 2},
        ),
        sources=sources,
        journal={"segments": 1, "incomplete_tails": 0, "quarantined": 0, "conflicts": 0},
        read_gaps={},
        dropped={},
        stopped_by="byte_budget",
        retention=RetentionCoverage(frozen_roots=1),
        index="available",
        lag_bytes=600,
    )
    envelope = snapshots.build_envelope(
        generation=1,
        counter_epoch=1,
        methodology_id=METHOD.hash,
        policy_version=1,
        collected_at=COLLECTED,
        coverage_through=publish.coverage_through(sources, None, COLLECTED),
        source_manifest_digest=publish.source_manifest_digest(sources),
        windows=windows,
        coverage=coverage,
        observations=publish.snapshot_observations(
            counters=empty_counters(1, "2026-11-01T00:00:00Z"), backfills=[]
        ),
        estimates={"representative_receipt": representative},
    )
    snapshots.publish(PROFILE, envelope)

    raw = profile_dir(tmp_path, "snapshots", "current.json").read_bytes()
    assert b"victim-canary" not in raw and b"/home/" not in raw

    view = service.read_summary(PROFILE, Scope(window="7d"), now=NOW)
    assert view.coverage["recorded_lookups"] == 112
    assert view.coverage["attributed_lookups"] == 91
    assert view.coverage["candidate_receipts"] == 13
    assert view.coverage["source_costs_available"] == 8
    assert view.coverage["ambiguous_joins"] == 3
    assert view.coverage["unavailable_sources"] == 2
    assert envelope.coverage["partial"] is True  # collector verdict kept in the snapshot
    life = service.read_summary(PROFILE, Scope(window="lifetime"), now=NOW)
    assert life.partial is True  # frozen root => lifetime window itself is partial
    assert life.coverage == view.coverage
    assert view.exclusions == (("prior_known", 4), ("user_supplied", 2))
    assert view.index == "available"
    assert view.coverage_through == "2026-11-07T11:00:00Z"
    measured = dict(view.measured)
    assert measured["searches"].value == 2 and measured["shows"].value == 1
    assert measured["failed"].value == 1
    assert view.receipt_id == representative
    stored = coverage
    assert stored["lag_bytes"] == 600 and stored["sources"]["lag_bytes"] == 600
    assert stored["stopped_by"] == "byte_budget" and stored["complete"] is False

    explained = service.read_receipt(PROFILE, view.receipt_id or "", include_content=True)
    assert explained is not None and explained.receipt_id == representative
    assert explained.content is None
    live = next(r for r in rows.receipts if r.receipt_id == representative)
    assert explained.evidence_available is live.receipt.evidence_available
    assert explained.revision == live.receipt.revision

    frozen = next(r for r in rows.receipts if not r.receipt.evidence_available)
    frozen_view = service.read_receipt(PROFILE, frozen.receipt_id, include_content=True)
    assert frozen_view is not None
    assert frozen_view.evidence_available is False and frozen_view.content is None
    assert frozen_view.note is not None and "evidence_expired" in frozen_view.note

    history = service.read_history(PROFILE)
    assert {(e.period, e.revision) for e in history.entries} == {
        ("2026-11-01", 1),
        ("2026-11-05", 1),
    }  # frozen root-1 collapsed its revisions into one frozen period row


def test_storage_dirs_are_opened_through_the_private_storage_layer(tmp_path: Path) -> None:
    publish.write_receipts(PROFILE, ledger().rows())
    publish.write_history(PROFILE, [])
    for parts in (("receipts",), ("history",)):
        directory = storage.open_profile_dir(PROFILE, *parts, create=False)
        try:
            assert stat.S_IMODE(os.fstat(directory).st_mode) == 0o700
        finally:
            os.close(directory)
    assert (profile_dir(tmp_path, "history") / "periods.json").read_bytes() == b"[]\n"


def test_baseline_skipped_history_is_never_lag_but_a_later_unfinished_source_is() -> None:
    # OpenCode-style baseline: positioned by watermark, byte offset 0 of a 5000-byte database
    db = source("opencode:/h/db", size=5000, offset=0, baseline="skipped", complete=True)
    db.skipped_bytes = 5000
    db.runtime = "opencode"
    cov = coverage_of(keyed(db))
    assert cov["sources"]["lag_bytes"] == 0
    assert cov["watermarks"][0]["lag_bytes"] == 0
    assert cov["sources"]["by_runtime"]["opencode"]["lag_bytes"] == 0
    assert cov["sources"]["by_runtime"]["opencode"]["processed_bytes"] == 5000
    assert cov["complete"] is True

    grown = replace(db, complete=False, more=True)  # new data arrived after the baseline
    cov = coverage_of(keyed(grown))
    assert cov["sources"]["lag_bytes"] == 5000 and cov["complete"] is False
