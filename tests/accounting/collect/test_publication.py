# ---- Survivor source: test_publish.py ----
"""Receipt/history writers and snapshot coverage blocks, read back through the facade."""

import collections
import os
import stat
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import ssgrep.accounting.views.snapshots as snapshots
from ssgrep.accounting.collect import (
    cycle_publish as publish,
    steps as collector_steps,
    store as store,
)
from ssgrep.accounting.collect.collector import RetentionCoverage
from ssgrep.accounting.collect.replay import Ledger, LedgerRows
from ssgrep.accounting.collect.state import ReductionDoc, VersionedLedger, load_state
from ssgrep.accounting.collect.store import SourceRecord, source_key
from ssgrep.accounting.ledger.contracts import (
    SCHEMA_VERSION,
    Attribution,
    Basis,
    Exposure,
    Outcome,
    Quantity,
    Reuse,
    TaskId,
    canonical_bytes,
)
from ssgrep.accounting.ledger.ledger_contracts import empty_counters
from ssgrep.accounting.ledger.reduce_records import ExecutionRecord, Reduction
from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.views import service
from ssgrep.accounting.views.history import build_windows
from ssgrep.accounting.views.types import Scope
from tests.accounting.collect import collector_scene as scene, verify_scenes as vs
from tests.accounting.collect.test_collect_suites import invocation
from tests.accounting.ledger.reduce_builders import (
    omp_header,
    omp_launch,
    omp_search,
    omp_user,
    results_json,
    ts,
)
from tests.accounting.ledger.valuation_builders import CHARS, METHOD, atom, candidate, debit
from tests.accounting.readers.readers_fixtures import write_records
from tests.accounting.views.snapshot_builders import enable_policy

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
    assert document["schema"] == record.receipt.schema == SCHEMA_VERSION  # the receipt's own
    future = replace(record, receipt=replace(record.receipt, schema=7))
    assert publish.receipt_document(future)["schema"] == 7


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


def test_representative_receipt_never_selects_an_unwritable_hostile_id() -> None:
    rows = synthetic_rows(
        ("../escape", 1, {"central": 10000}, True), ("safe", 1, {"central": 10}, True)
    )
    assert publish.representative_receipt(rows) == "safe"
    assert publish.representative_receipt(replace(rows, receipts=rows.receipts[:1])) is None


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
    assert [d["schema"] for d in document] == [SCHEMA_VERSION] * 3  # every row is stamped
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
    from ssgrep.accounting.ledger.units import WORK_SECONDS, UnitSpec

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
    real = collector_steps.digest_of
    monkeypatch.setattr(
        collector_steps,
        "digest_of",
        lambda value: seen.append(canonical_bytes(value)) or real(value),
    )

    cov = coverage_of(sources)
    digest = collector_steps.source_manifest_digest(sources)
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
    digest = collector_steps.source_manifest_digest(keyed(base))
    assert digest == collector_steps.source_manifest_digest(keyed(base))
    clocks = replace(base, mtime_ns=999, last_ts="2026-12-24T00:00:00Z", resets=3, gaps={"x": 1})
    assert collector_steps.source_manifest_digest(keyed(clocks)) == digest
    for changed in (
        source("claude:/h/a.jsonl", size=101, offset=40),
        source("claude:/h/a.jsonl", size=100, offset=41),
        source("claude:/h/a.jsonl", size=100, offset=40, status="missing"),
        source("claude:/h/a.jsonl", size=100, offset=40, baseline="skipped"),
        source("claude:/h/a.jsonl", size=100, offset=40, runtime="codex"),
        source("claude:/h/other.jsonl", size=100, offset=40),
    ):
        assert collector_steps.source_manifest_digest(keyed(changed)) != digest
    two = keyed(base, source("pi:/h/b.jsonl", size=1, offset=1, runtime="pi"))
    assert collector_steps.source_manifest_digest(two) == collector_steps.source_manifest_digest(
        dict(reversed(list(two.items())))
    )
    assert collector_steps.source_manifest_digest({}) == collector_steps.source_manifest_digest({})


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
    naive = keyed(source("naive", last_ts="2027-01-01T12:00:00"))  # no offset: never a watermark
    assert (
        publish.coverage_through(naive, "2026-10-12T11:00:00Z", "2026-10-12T12:00:00Z")
        == "2026-10-12T11:00:00Z"
    )


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
        source_manifest_digest=collector_steps.source_manifest_digest(sources),
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


# ---- session facts persisted beside the ledger -----------------------------------------------


def test_missing_invocation_cwd_does_not_override_the_earliest_proven_directory() -> None:
    missing = invocation(1, "2026-10-12T09:00:00Z")
    known = replace(invocation(2, "2026-10-12T10:00:00Z"), cwd="/work/first")
    later = replace(invocation(3, "2026-10-12T11:00:00Z"), cwd="/work/later")
    reduction = Reduction(PROFILE, invocations=(missing, later, known))
    assert collector_steps.session_cwds(reduction) == (("claude", "s", "/work/first"),)


SESSION_CWD = "/work/proj"


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
    for path in (parent, base / "root" / "ChildAlpha.jsonl"):
        os.utime(path, (when.timestamp(), when.timestamp()))


def reduction_doc() -> ReductionDoc:
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    doc = load_state(PROFILE, checkpoint, checkpoint.methodology).reduction
    assert doc is not None
    return doc


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
    assert doc.session_cwd == (("omp", "child-a", SESSION_CWD),)
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


# ---- receipt files follow the ledger when a late parent retracts a child's root ---------------

LINEAGE_NOW = scene.at(hour=20)
STAMP = scene.at(hour=10).timestamp()


def sessions(root: Path) -> Path:
    return root / "omp-sessions" / "-work"


def write_child(root: Path) -> None:
    parent = sessions(root) / "root.jsonl"
    child = sessions(root) / "root" / "ChildAlpha.jsonl"
    write_records(
        child,
        [
            omp_header("child-a", ts(12, 9, 12), parent=parent),
            *omp_search("c1", "call-1", "deploy", results_json(), (12, 10, 0, 0)),
            *omp_search("c2", "call-2", "billing", results_json(), (12, 10, 1, 0)),
        ],
    )
    os.utime(child, (STAMP, STAMP))


def write_parent(root: Path) -> None:
    parent = sessions(root) / "root.jsonl"
    write_records(
        parent,
        [
            omp_header("root-1", ts(12, 8)),
            omp_user("e-a", "investigate the failing deploy", ts(12, 9)),
            *omp_launch("e-l1", "c-launch", "ChildAlpha", ts(12, 9, 10), ts(12, 9, 11)),
        ],
    )
    os.utime(parent, (STAMP, STAMP))


def committed_ledger() -> VersionedLedger:
    committed = vs.checkpoint()
    assert committed is not None
    state = load_state(PROFILE, committed, committed.methodology)
    assert state.ledger is not None
    return VersionedLedger.restore_versioned(state.ledger, state.superseded)


def live_debit_roots() -> dict[str, set[str]]:
    by_id: dict[str, set[str]] = collections.defaultdict(set)
    for item in committed_ledger()._debits.values():
        by_id[item.debit_id].add(item.task.root)
    return dict(by_id)


def publish_run_late_parent(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture(day=12)
    write_child(root)
    first = scene.collect(LINEAGE_NOW)
    assert first.state == "published"
    assert {r for roots in live_debit_roots().values() for r in roots} == {"omp:child-a"}
    write_parent(root)
    assert scene.collect(LINEAGE_NOW).state == "published"


def test_receipt_files_follow_the_ledger_when_a_root_disappears(tmp_path: Path) -> None:
    publish_run_late_parent(tmp_path)
    receipts = tmp_path / "data" / "accounting" / PROFILE / "receipts"
    on_disk = {p.stem for p in receipts.glob("*.json")} if receipts.exists() else set()
    in_ledger = {r.receipt_id for r in committed_ledger().rows().receipts}
    assert on_disk == in_ledger  # no ghost receipt of a retracted root stays explainable
    assert store.load_checkpoint(PROFILE) is not None


# ---- Coverage: stop reasons, retention maintenance, retired methodology, partition edges ----


def prepare_retention(checkpoint: store.Checkpoint, *, budget: int = 1 << 20) -> Any:
    from ssgrep.accounting.collect import collector

    return collector.prepare(
        PROFILE,
        checkpoint,
        rows=None,
        before_day="2026-10-01",
        evidence_before_day="2026-11-01",
        clock=lambda: 0,
        deadline=1,
        byte_budget=budget,
    )


def delete_native_sources() -> None:
    for source in scene.committed().sources.values():
        Path(source.source_id.split(":", 1)[1]).unlink()


def test_journal_expiry_keeps_a_minimal_start_for_an_execution_crossing_the_horizon() -> None:
    from ssgrep.accounting.capture.journal_codec import minimal_started
    from ssgrep.accounting.collect.collector import expire_journal_events
    from ssgrep.accounting.ledger.contracts import Phase
    from tests.accounting.conftest import build_event

    def started(execution_id: str) -> Any:
        return build_event(
            execution_id=execution_id,
            phase=Phase.STARTED,
            outcome=None,
            mono_ns=None,
            rendered_bytes=None,
            ts_utc="2025-12-31T23:00:00Z",
        )

    crossing, orphan = started("crossing"), started("orphan")
    finished = build_event(
        execution_id="crossing", phase=Phase.COMPLETED, ts_utc="2026-01-01T01:00:00Z"
    )
    kept, dropped = expire_journal_events([crossing, orphan, finished], "2026-01-01")
    assert kept == [minimal_started(crossing), finished]
    assert kept[0] != crossing and kept[0].cwd is None  # evidence is stripped, identity stays
    assert dropped == 1  # the start of an execution that never reached the horizon is dropped


def test_retention_child_sizes_and_cleanup_intents_reject_invalid_references() -> None:
    from ssgrep.accounting.collect import collector

    reference = store.write_blob(PROFILE, "retention-child", {"k": "v"})
    on_disk = profile_dir_for_state() / reference
    assert collector._size(PROFILE, reference) == on_disk.stat().st_size
    for invalid_name in ("", ".", "..", "a/b", "a\0b", 7):
        with pytest.raises(store.CollectError) as invalid:
            collector._size(PROFILE, invalid_name)
        assert invalid.value.condition == "state_corrupt"

    limit = tuple(f"cleanup-{number:02}" for number in range(32))
    assert collector._remaining_public(PROFILE, None, limit) == ()  # no receipt directory yet
    for oversized in (limit + ("cleanup-32",), "cleanup-00"):
        with pytest.raises(store.CollectError) as invalid:
            collector._remaining_public(PROFILE, None, oversized)
        assert invalid.value.condition == "state_corrupt"


def profile_dir_for_state() -> Path:
    return scene.accounting_dir() / "collector" / store.STATE_DIR


def test_original_publication_manifest_must_be_an_object() -> None:
    from ssgrep.accounting.collect import collector
    from ssgrep.accounting.readers.identity_migration import ARCHIVE

    valid = store.Checkpoint(
        PROFILE, blobs={ARCHIVE: store.write_blob(PROFILE, ARCHIVE, {"schema": 1, "blobs": {}})}
    )
    assert not prepare_retention(valid).pending
    damaged = store.Checkpoint(
        PROFILE, blobs={ARCHIVE: store.write_blob(PROFILE, ARCHIVE, ["not", "a", "manifest"])}
    )
    with pytest.raises(store.CollectError) as invalid:
        prepare_retention(damaged)
    assert invalid.value.condition == "state_corrupt"
    assert collector.KEY not in damaged.retention  # nothing claims progress over damage


def test_receipt_retention_waits_for_budget_before_selecting_a_receipt_child(monkeypatch) -> None:
    from ssgrep.accounting.collect import collector
    from ssgrep.accounting.privacy import retention as published_retention
    from ssgrep.accounting.readers.identity_migration import ARCHIVE

    receipt = store.write_blob(PROFILE, "original-receipt", {"receipt": "body"})
    manifest = {
        "schema": 1,
        "blobs": {},
        "segments": [],
        "published_documents": {
            "format": "published-session-v1",
            "history_blob": None,
            "receipt_blobs": {"receipt-1": receipt},
        },
    }
    reference = store.write_blob(PROFILE, ARCHIVE, manifest)
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
    child = collector._size(PROFILE, receipt)
    needed = collector._size(PROFILE, reference) + child
    selections: list[tuple[str, ...]] = []

    def selected(*args: Any, **kwargs: Any) -> Any:
        selections.append(kwargs["receipt_ids"])
        return published_retention.ChildResult(False, (), kwargs["manifest"])

    monkeypatch.setattr(published_retention, "stage_children", selected)
    blocked = prepare_retention(checkpoint, budget=needed - 1)
    assert blocked.pending and not blocked.changed and not blocked.delete_receipts
    assert checkpoint.retention[collector.KEY]["blocked_child_bytes"] == child
    assert selections == []
    resumed = prepare_retention(checkpoint, budget=needed)
    assert not resumed.pending
    assert selections == [("receipt-1",)]


def test_budget_expiring_after_staging_publishes_a_partial_cycle(tmp_path: Path) -> None:
    from ssgrep.accounting.collect.collector import CollectOptions, CycleBudget

    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    ticks = [0.0]

    def overrun(step: str) -> None:
        if step == "state_written":
            ticks[0] += 3600

    options = CollectOptions(budget=CycleBudget(seconds=60))
    result = scene.collect(scene.at(hour=9), options, clock=lambda: ticks[0], on_step=overrun)
    report = scene.report(result)
    assert result.state == "partial" and report.status == "published" and not report.complete
    assert report.stopped_by == "cycle_budget_exceeded"
    assert scene.coverage(scene.current())["stopped_by"] == "cycle_budget_exceeded"
    steady = scene.collect(scene.at(hour=9, minute=1), options, clock=lambda: ticks[0])
    assert steady.state == "published" and scene.report(steady).stopped_by == "complete"


def test_unfinished_original_publication_retention_is_reported_until_it_fits(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect import collector
    from ssgrep.accounting.collect.collector import CollectOptions, CycleBudget
    from ssgrep.accounting.readers.identity_migration import ARCHIVE

    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    assert scene.collect(scene.at(hour=9)).state == "published"
    checkpoint = scene.committed()
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, {"schema": 1, "blobs": {}})
    store.commit_checkpoint(PROFILE, checkpoint)
    blocked = scene.collect(scene.at(hour=9, minute=5), CollectOptions(budget=CycleBudget(bytes=1)))
    assert blocked.state == "partial"
    assert scene.report(blocked).stopped_by == "published_retention_budget"
    job = scene.committed().retention[collector.KEY]
    assert job["pending"] and job["blocked_child_bytes"] > 1
    resumed = scene.collect(scene.at(hour=9, minute=6))
    assert resumed.state == "published" and scene.report(resumed).stopped_by == "complete"
    assert not scene.committed().retention[collector.KEY]["pending"]


def test_pending_evidence_maintenance_is_partial_until_the_queue_drains(tmp_path: Path) -> None:
    from datetime import timedelta

    from ssgrep.accounting.collect.collector import CollectOptions, CycleBudget
    from tests.accounting.collect.test_collect_suites import seed

    paths = seed(tmp_path)
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    assert scene.collect(scene.at(hour=11)).state == "published"
    assert len(scene.committed().segments) == 2
    expired = scene.at() + timedelta(days=31)
    blocked = scene.collect(expired, CollectOptions(budget=CycleBudget(bytes=1)))
    assert blocked.state == "partial"
    assert scene.report(blocked).stopped_by == "evidence_maintenance_budget"
    assert len(scene.committed().retention["evidence_maintenance"]["pending"]) == 1
    resumed = scene.collect(expired + timedelta(minutes=1))
    assert resumed.state == "published" and scene.report(resumed).stopped_by == "complete"
    assert scene.committed().retention["evidence_maintenance"]["pending"] == []
    assert list(store.iter_evidence(PROFILE, scene.committed().segments)) == []


def test_reduction_deferred_by_the_budget_is_reported_and_then_resumed(tmp_path: Path) -> None:
    from ssgrep.accounting.collect.collector import CollectOptions, CycleBudget
    from ssgrep.accounting.collect.partitions import load_partitions
    from tests.accounting.collect.test_collect_suites import current_ledger, seed

    paths = seed(tmp_path)
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    ticks = [0.0]

    def overrun(step: str) -> None:
        if step == "evidence_written":  # reading is done; reduction then has no budget left
            ticks[0] += 3600

    options = CollectOptions(budget=CycleBudget(seconds=60))
    partial = scene.collect(scene.at(hour=11), options, clock=lambda: ticks[0], on_step=overrun)
    assert partial.state == "partial" and scene.report(partial).new_evidence == 2
    assert scene.report(partial).stopped_by == "reduction_budget"
    assert load_partitions(PROFILE, scene.committed()).pending == ["claude:left"]
    assert len(current_ledger()[1].reduction.invocations) == 2  # accepted rows, not guessed
    resumed = scene.collect(scene.at(hour=11, minute=1), options, clock=lambda: ticks[0])
    assert resumed.state == "published" and scene.report(resumed).stopped_by == "complete"
    assert len(current_ledger()[1].reduction.invocations) == 3


def test_root_state_conversion_beyond_its_quantum_is_reported_until_finished(
    tmp_path: Path, monkeypatch
) -> None:
    import functools

    from ssgrep.accounting.collect import state as collector_state
    from ssgrep.accounting.ledger.ledger_codec import ledger_state_to_dict

    vs.enable(pin=True)
    vs.start_capture()
    for number in range(3):
        vs.pair(vs.projects(tmp_path), number, scene.at(hour=9))
    assert scene.collect(scene.at(hour=9)).state == "published"
    checkpoint = scene.committed()
    state = collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.ledger is not None and not state.superseded and not state.archives
    name = collector_state.ledger_blob_name(checkpoint.methodology)
    checkpoint.blobs[name] = store.write_blob(  # the pre-root-format single-document ledger
        PROFILE,
        name,
        {
            "state": ledger_state_to_dict(state.ledger),
            "superseded": {},
            "archives": {},
            "collection_scope": state.collection_scope,
        },
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    monkeypatch.setattr(
        collector_state,
        "save_ledger_incremental",
        functools.partial(collector_state.save_ledger_incremental, migration_limit=1),
    )
    scene.append_transcript(
        vs.projects(tmp_path) / "-work" / "cons0.jsonl",
        scene.search_records("cons0", "t0-next", "different query", (13, 0, 0)),
        scene.at(hour=13),
    )
    partial = scene.collect(scene.at(hour=13))
    assert partial.state == "partial"
    assert scene.report(partial).stopped_by == "root_state_migration_budget"
    assert scene.committed().retention["root_state_migration"] == {"pending_roots": 1}
    converted = vs.ledger_digest()
    finished = scene.collect(scene.at(hour=13, minute=1))
    assert finished.state == "published" and scene.report(finished).stopped_by == "complete"
    assert scene.committed().retention["root_state_migration"] == {"pending_roots": 0}
    assert vs.ledger_digest() == converted  # conversion never changes an accepted row


def test_unverifiable_legacy_originals_stop_the_cycle_at_identity_migration(tmp_path: Path) -> None:
    from tests.accounting.conftest import build_evidence

    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    assert scene.collect(scene.at(hour=9)).state == "published"
    published = scene.current()
    checkpoint = scene.committed()
    original = build_evidence(runtime="claude", source_id="claude:/missing/orig.jsonl")
    checkpoint.identity_method, checkpoint.identity_migration = "", {}
    checkpoint.segments = [*checkpoint.segments, *store.write_segments(PROFILE, [original])]
    store.commit_checkpoint(PROFILE, checkpoint)
    blocked = scene.collect(scene.at(hour=9, minute=5))
    report = scene.report(blocked)
    assert blocked.state == "partial" and blocked.warnings == ("identity_migration_unavailable",)
    assert (report.status, report.stopped_by) == ("failed", "identity_migration")
    assert report.error == "identity_migration_unavailable" and not report.complete
    migration = scene.committed().identity_migration
    assert migration["status"] == "unavailable"
    assert migration["unavailable"] == [["claude", original.session_id, "source_unapproved"]]
    assert scene.committed().identity_method != "native_identity_v2"
    assert scene.current() == published  # no new generation claims a coherent identity


def test_pending_privacy_compaction_stops_the_cycle_before_any_native_work(tmp_path: Path) -> None:
    from ssgrep.accounting.collect.collector import CollectOptions, CycleBudget
    from ssgrep.accounting.collect.replay import evidence_line
    from tests.accounting.conftest import build_evidence

    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    assert scene.collect(scene.at(hour=9)).state == "published"
    published = scene.current()
    checkpoint = scene.committed()
    legacy = build_evidence(source_id="claude:/missing/legacy.jsonl")
    body = evidence_line(legacy, plaintext=True) + b"\n"
    directory = storage.open_profile_dir(PROFILE, "collector", store.EVIDENCE_DIR)
    try:
        storage.write_once(directory, "seg-legacy-plain.jsonl", body)
    finally:
        os.close(directory)
    checkpoint.segments = [
        *checkpoint.segments,
        store.SegmentRef("seg-legacy-plain.jsonl", 1, len(body)),
    ]
    checkpoint.retention.pop("evidence_format")  # the plaintext segment predates compaction
    store.commit_checkpoint(PROFILE, checkpoint)
    blocked = scene.collect(
        scene.at(hour=9, minute=5), CollectOptions(budget=CycleBudget(bytes=len(body) - 1))
    )
    report = scene.report(blocked)
    assert blocked.state == "partial" and blocked.warnings == ("evidence_migration_pending",)
    assert (report.status, report.stopped_by) == ("failed", "evidence_migration")
    assert scene.committed().retention["evidence_migration_pending"]
    assert scene.current() == published
    for minute in range(6, 12):
        resumed = scene.collect(scene.at(hour=9, minute=minute))
        if resumed.state == "published":
            break
    assert resumed.state == "published"
    assert not scene.committed().retention.get("evidence_migration_pending")
    assert all(ref.name != "seg-legacy-plain.jsonl" for ref in scene.committed().segments)


def test_ledger_of_another_methodology_is_never_accepted_as_current_state(tmp_path: Path) -> None:
    from ssgrep.accounting.collect import state as collector_state
    from ssgrep.accounting.ledger.ledger_codec import ledger_state_to_dict
    from tests.accounting.collect.test_collect_suites import seeded

    seeded(tmp_path)
    checkpoint = scene.committed()
    original = checkpoint.to_dict()
    published = scene.current()
    foreign = store.write_blob(
        PROFILE,
        "ledger",
        {
            "state": ledger_state_to_dict(VersionedLedger(METHOD).state()),
            "superseded": {},
            "collection_scope": checkpoint.collection_scope,
        },
    )
    # The checkpoint names an older, coherent ledger while the current methodology's
    # slot holds a ledger that really belongs to another methodology.
    current = checkpoint.methodology
    checkpoint.methodology = METHOD.hash
    checkpoint.blobs[collector_state.ledger_blob_name(METHOD.hash)] = foreign
    checkpoint.blobs[collector_state.ledger_blob_name(current)] = foreign
    store.commit_checkpoint(PROFILE, checkpoint)
    with pytest.raises(store.CollectError) as mismatch:
        scene.collect(scene.at(hour=20))
    assert mismatch.value.condition == "state_corrupt"
    assert "does not match state" in str(mismatch.value)
    assert scene.current() == published
    store.commit_checkpoint(PROFILE, store.Checkpoint.from_dict(original))
    assert scene.report(scene.collect(scene.at(hour=20))).generation == 2


def test_uniquely_pinned_uncommitted_snapshot_blocks_collection_until_unpinned(
    tmp_path: Path,
) -> None:
    from tests.accounting.collect.test_collect_suites import NOW, seeded
    from tests.accounting.export.snapshots_driver import make_envelope

    seeded(tmp_path)
    published = scene.current()
    stranded = make_envelope(2)
    snapshots.stage_generation(PROFILE, stranded)
    storage.write_private(
        tmp_path / "data",
        "accounting",
        PROFILE,
        "delivery",
        "ops",
        "metrics.json",
        data=('{"last_acked_snapshot_id": "' + stranded.snapshot_id + '"}').encode(),
    )
    with pytest.raises(store.CollectError) as pinned:
        scene.collect(NOW)
    assert pinned.value.condition == "state_corrupt"
    assert "uniquely pinned" in str(pinned.value)
    assert snapshots.read_generation(PROFILE, 2) == stranded  # its only bytes were kept
    assert scene.current() == published
    pin_file = tmp_path / "data" / "accounting" / PROFILE / "delivery" / "ops" / "metrics.json"
    assert pin_file.is_file()
    pin_file.unlink()
    assert scene.report(scene.collect(NOW)).generation == 2


def test_unindexed_legacy_segments_are_indexed_one_quantum_per_cycle_before_reading(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.collector import CollectOptions, CycleBudget
    from tests.accounting.collect.test_collect_suites import seed

    paths = seed(tmp_path)
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    assert scene.collect(scene.at(hour=11)).state == "published"
    checkpoint = scene.committed()
    assert len(checkpoint.segments) == 2
    accepted = vs.ledger_digest()
    checkpoint.segments = [replace(ref, index=None) for ref in checkpoint.segments]
    checkpoint.blobs.pop("partitions")  # bootstrap enumerates sessions from the segments
    store.commit_checkpoint(PROFILE, checkpoint)
    blocked = scene.collect(scene.at(hour=12), CollectOptions(budget=CycleBudget(bytes=1)))
    assert blocked.state == "partial"
    assert scene.report(blocked).stopped_by == "evidence_index_budget"
    committed = scene.committed()
    assert [ref.index is None for ref in committed.segments].count(True) == 1
    assert len(committed.retention["evidence_index_pending"]) == 1
    resumed = scene.collect(scene.at(hour=12, minute=1))
    assert resumed.state == "published" and scene.report(resumed).stopped_by == "complete"
    assert all(ref.index is not None for ref in scene.committed().segments)
    assert "evidence_index_pending" not in scene.committed().retention
    assert vs.ledger_digest() == accepted  # indexing never changes an accepted row


def test_indexed_retained_line_that_is_not_evidence_is_state_corruption() -> None:
    from ssgrep.accounting.collect.cycle import IndexWriter, iter_selected
    from ssgrep.accounting.collect.replay import evidence_line
    from tests.accounting.conftest import build_evidence

    item = build_evidence()

    def indexed(name: str, line: bytes) -> store.SegmentRef:
        directory = storage.open_profile_dir(PROFILE, "collector", store.EVIDENCE_DIR)
        try:
            storage.write_once(directory, name, line)
        finally:
            os.close(directory)
        writer = IndexWriter()
        writer.add(item, 0, line)
        return store.SegmentRef(name, 1, len(line), "live", writer.write(PROFILE))

    sessions = {(item.runtime, item.session_id)}
    good = indexed("seg-indexed-good.jsonl", evidence_line(item) + b"\n")
    assert [row.digest for row in iter_selected(PROFILE, good, sessions=sessions)] == [item.digest]
    damaged = indexed("seg-indexed-damaged.jsonl", b'{"not": "evidence"}\n')
    with pytest.raises(store.CollectError) as invalid:
        list(iter_selected(PROFILE, damaged, sessions=sessions))
    assert invalid.value.condition == "state_corrupt"


@pytest.mark.parametrize("variant", ["recompute", "partitions_lost", "methodology_replay"])
def test_deleted_native_sources_keep_accepted_rows_and_report_replay_unavailable(
    tmp_path: Path, variant: str
) -> None:
    from ssgrep.accounting.collect import partitions as collector_partitions
    from ssgrep.accounting.collect.collector import CollectOptions
    from tests.accounting.collect.test_collect_suites import NOW, seeded

    seeded(tmp_path)
    accepted = vs.ledger_digest()
    checkpoint = scene.committed()
    prior = collector_partitions.load_partitions(PROFILE, checkpoint)
    assert prior.unavailable == {} and prior.metas

    def sources_of(ref: str) -> set:
        blob = store.read_blob(PROFILE, ref)
        assert blob is not None
        return set(blob["sources"])

    prior_sources = {name: sources_of(ref) for name, ref in prior.metas.items()}
    options = scene.LIVE
    if variant == "recompute":
        options = CollectOptions(recompute=True)
    else:
        if variant == "partitions_lost":  # a checkpoint that predates the partition manifest
            checkpoint.blobs.pop("partitions")
        else:  # partitions persisted by an older reducer methodology
            prior.methodology = "older-methodology"
            collector_partitions.save_partitions(PROFILE, prior, checkpoint.blobs)
        store.commit_checkpoint(PROFILE, checkpoint)
    delete_native_sources()
    result = scene.collect(NOW, options)
    assert result.state == "partial" and scene.report(result).stopped_by == "replay_unavailable"
    assert "historical_replay_unavailable" in result.warnings
    assert vs.ledger_digest() == accepted  # missing originals never erase accepted usage
    after = collector_partitions.load_partitions(PROFILE, scene.committed())
    assert after.unavailable == {"claude:cons0": "source_missing", "claude:src0": "source_missing"}
    for name, sources in prior_sources.items():
        assert sources <= sources_of(after.metas[name])


def test_new_usage_in_a_session_with_unreplayable_history_is_marked_unavailable(
    tmp_path: Path,
) -> None:
    from tests.accounting.collect.test_collect_suites import current_ledger, seeded
    from tests.accounting.ledger.reduce_builders import (
        MCP,
        card,
        claude_tool_result,
        claude_tool_use,
    )

    seeded(tmp_path)
    prior, _ = current_ledger()
    (original,) = prior.state().candidates
    (vs.projects(tmp_path) / "-work" / "cons0.jsonl").unlink()
    body = results_json(card("claude:src0:ep:0", vs.fact(0)))
    continued = [
        claude_tool_use(
            "cons0",
            "t0c",
            MCP,
            {"query": vs.query(0) + " again"},
            ts=ts(12, 14, 0, 10),
            message_id="m0c",
        ),
        claude_tool_result("cons0", "t0c", body, ts=ts(12, 14, 0, 12)),
        claude_tool_use(
            "cons0",
            "t0d",
            "Bash",
            {"command": f"vim {vs.path(0)}"},
            ts=ts(12, 14, 0, 20),
            message_id="m0d",
        ),
        claude_tool_result("cons0", "t0d", "ok", ts=ts(12, 14, 0, 22)),
    ]
    scene.write_transcript(tmp_path, "cons0-continued", continued, scene.at(hour=13))
    result = scene.collect(scene.at(hour=13))
    assert scene.report(result).new_evidence == 4 and result.state == "partial"
    current, _ = current_ledger()
    flagged = [
        row
        for row in current.state().candidates
        if "prior_context_replay_unavailable" in row.eligibility_reasons
    ]
    assert len(current.state().candidates) == 2 and len(flagged) == 1
    assert flagged[0].eligibility.value == "unavailable"
    assert original in current.state().candidates  # the accepted candidate is untouched
    assert "prior_context_replay_unavailable" not in original.eligibility_reasons


def test_pending_projection_of_a_changed_session_takes_its_normal_reduction(tmp_path: Path) -> None:
    from ssgrep.accounting.collect import partitions as collector_partitions
    from tests.accounting.collect.test_collect_suites import current_ledger, seed

    paths = seed(tmp_path)
    checkpoint = scene.committed()
    partitions = collector_partitions.load_partitions(PROFILE, checkpoint)
    blobs = {name: part["blob"] for name, part in partitions.parts.items()}
    for part in partitions.parts.values():
        part.pop("candidate_identity_method")  # the pre-cutover manifest shape
    collector_partitions.save_partitions(PROFILE, partitions, checkpoint.blobs)
    store.commit_checkpoint(PROFILE, checkpoint)
    assert len(collector_partitions.load_partitions(PROFILE, checkpoint).pending_projections) == 3
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    result = scene.collect(scene.at(hour=11))
    assert result.state == "published" and scene.report(result).stopped_by == "complete"
    after = collector_partitions.load_partitions(PROFILE, scene.committed())
    assert not after.pending_projections
    unchanged = {name for name, part in after.parts.items() if part["blob"] == blobs.get(name)}
    assert unchanged == {
        name for name, part in after.parts.items() if "claude:left" not in part["members"]
    }
    assert len(current_ledger()[1].reduction.invocations) == 3


def test_damaged_partition_metadata_fails_closed_before_publication(tmp_path: Path) -> None:
    from ssgrep.accounting.collect import partitions as collector_partitions
    from tests.accounting.collect.test_collect_suites import seed

    paths = seed(tmp_path)
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    checkpoint = scene.committed()
    published = scene.current()
    partitions = collector_partitions.load_partitions(PROFILE, checkpoint)
    name = sorted(partitions.metas)[0]
    healthy_meta = store.read_blob(PROFILE, partitions.metas[name])
    partitions.metas[name] = store.write_blob(PROFILE, "partition-meta", ["not", "a", "meta"])
    collector_partitions.save_partitions(PROFILE, partitions, checkpoint.blobs)
    store.commit_checkpoint(PROFILE, checkpoint)
    with pytest.raises(store.CollectError) as damaged:
        scene.collect(scene.at(hour=11))
    assert damaged.value.condition == "state_corrupt"
    assert scene.current() == published
    repaired = scene.committed()
    partitions = collector_partitions.load_partitions(PROFILE, repaired)
    partitions.metas[name] = store.write_blob(PROFILE, "partition-meta", healthy_meta)
    collector_partitions.save_partitions(PROFILE, partitions, repaired.blobs)
    store.commit_checkpoint(PROFILE, repaired)
    assert scene.collect(scene.at(hour=11)).state == "published"


def test_forked_session_sharing_native_record_ids_joins_its_origin_partition(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.partitions import load_partitions
    from tests.accounting.collect.test_collect_suites import current_ledger

    vs.enable(pin=True)
    vs.start_capture()
    scene.collect(scene.at(hour=8, minute=5))
    uuids = ("3f2b8c1e-6d0a-4a57-9a43-5f6f0d9d2a11", "7c9e6679-7425-40de-944b-e07fc1f90ae7")
    records = [
        {**row, "uuid": uuids[index]}
        for index, row in enumerate(scene.search_records("forkme", "fm-1", "query", (10, 0, 0)))
    ]
    scene.write_transcript(tmp_path, "forkme", records, scene.at())
    assert scene.collect(scene.at()).state == "published"
    accepted = [r.invocation_id for r in current_ledger()[1].reduction.invocations]
    forked = [{**row, "sessionId": "forkchild"} for row in records]
    scene.write_transcript(tmp_path, "forkchild", forked, scene.at())
    assert scene.collect(scene.at(minute=1)).state == "published"
    kinds = {
        (e.session_id, e.kind) for e in store.iter_evidence(PROFILE, scene.committed().segments)
    }
    assert ("forkchild", "ownership_claim") in kinds and ("forkchild", "tool_call") not in kinds
    parts = load_partitions(PROFILE, scene.committed()).parts
    assert [p["members"] for p in parts.values() if "claude:forkme" in p["members"]] == [
        ["claude:forkchild", "claude:forkme"]
    ]
    assert [r.invocation_id for r in current_ledger()[1].reduction.invocations] == accepted


def test_unchanged_reduction_roots_are_reused_from_the_prior_manifest() -> None:
    from ssgrep.accounting.collect.partitions import (
        REDUCTION_FORMAT,
        expand_reduction_document,
        write_reduction_document,
    )

    def execution(session: str, when: str) -> dict[str, str]:
        return {
            "runtime": "claude",
            "session_id": session,
            "execution_id": f"e-{session}",
            "ts_utc": when,
        }

    previous = {
        "schema": 1,
        "executions": [
            execution("a", "2026-11-01T00:00:00Z"),
            execution("b", "2026-11-01T00:00:00Z"),
        ],
    }
    document = {
        "schema": 1,
        "executions": [
            execution("a", "2026-11-01T00:00:00Z"),
            execution("b", "2026-11-02T00:00:00Z"),
        ],
    }
    prior = {"format": REDUCTION_FORMAT, "roots": {"claude:a": "prior-root-a.json"}}
    reused = store.read_blob(
        PROFILE,
        write_reduction_document(
            PROFILE, "reduction", document, previous=previous, prior_manifest=prior
        ),
    )
    assert reused is not None
    assert reused["roots"]["claude:a"] == "prior-root-a.json"  # unchanged: not rewritten
    assert reused["roots"]["claude:b"] != "prior-root-b.json"
    reused_b = store.read_blob(PROFILE, reused["roots"]["claude:b"])
    assert reused_b is not None
    assert reused_b["executions"] == [execution("b", "2026-11-02T00:00:00Z")]
    rewritten = store.read_blob(
        PROFILE, write_reduction_document(PROFILE, "reduction", document, prior_manifest=prior)
    )
    assert rewritten is not None
    assert rewritten["roots"]["claude:a"] != "prior-root-a.json"  # nothing proves it unchanged
    expanded = expand_reduction_document(PROFILE, rewritten)
    assert [row["execution_id"] for row in expanded["executions"]] == ["e-a", "e-b"]


def test_evidence_maintenance_rejects_a_malformed_queue_and_skips_vanished_segments() -> None:
    from ssgrep.accounting.collect.steps import maintain_evidence
    from ssgrep.accounting.ledger.contracts import digest_of
    from tests.accounting.conftest import build_evidence

    (ref,) = store.write_segments(PROFILE, [build_evidence()])
    digest = digest_of([])
    common: dict[str, Any] = {
        "expired_sessions": frozenset(),
        "clock": lambda: 0,
        "deadline": 1,
        "byte_budget": 1 << 20,
        "should_stop": lambda: False,
    }
    with pytest.raises(store.CollectError) as malformed:
        maintain_evidence(
            PROFILE, [ref], previous={"expired_digest": digest, "pending": "seg"}, **common
        )
    assert malformed.value.condition == "state_corrupt"
    refs, job, used, dropped = maintain_evidence(
        PROFILE,
        [ref],
        previous={"expired_digest": digest, "pending": ["seg-vanished.jsonl"]},
        **common,
    )
    assert refs == [ref] and (used, dropped) == (0, 0)
    assert job == {"expired_digest": digest, "pending": [], "compaction_pending": False}


def test_compaction_leaves_segments_alone_when_merging_cannot_reduce_their_count(
    monkeypatch,
) -> None:
    import json

    from ssgrep.accounting.collect.replay import evidence_line
    from ssgrep.accounting.collect.steps import maintain_evidence
    from tests.accounting.conftest import build_evidence

    def legacy_segment(name: str, items: list[Any]) -> store.SegmentRef:
        body = b""
        for item in items:  # older writers omitted the origin fields: smaller lines
            raw = json.loads(evidence_line(item))
            for field in ("evidence_format", "origin_domain", "origin_id", "origin_status"):
                raw.pop(field)
            body += (
                json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                + b"\n"
            )
        directory = storage.open_profile_dir(PROFILE, "collector", store.EVIDENCE_DIR)
        try:
            storage.write_once(directory, name, body)
        finally:
            os.close(directory)
        return store.SegmentRef(name, len(items), len(body))

    refs = [
        legacy_segment(
            f"seg-legacy-{number}.jsonl",
            [build_evidence(record_id=f"r{number}-{i}", locator=f"byte:{i}") for i in range(6)],
        )
        for number in range(2)
    ]
    monkeypatch.setattr(store, "MAX_SEGMENTS", 1)
    total = sum(ref.size for ref in refs)
    monkeypatch.setattr(store, "SEGMENT_TARGET_BYTES", total)
    common: dict[str, Any] = {
        "expired_sessions": frozenset(),
        "previous": None,
        "clock": lambda: 0,
        "deadline": 1,
        "byte_budget": 1 << 30,
        "should_stop": lambda: False,
    }
    # Re-serialised lines are larger than the stored ones, so the pair would split again.
    unchanged, job, used, dropped = maintain_evidence(PROFILE, refs, **common)
    assert unchanged == refs and (used, dropped) == (0, 0)
    assert job["compaction_pending"] is True  # still a candidate; nothing was charged
    monkeypatch.setattr(store, "SEGMENT_TARGET_BYTES", 10 * total)
    merged, job, used, dropped = maintain_evidence(PROFILE, refs, **common)
    assert [ref.records for ref in merged] == [12] and used == total and dropped == 0
    assert job["compaction_pending"] is False


def test_retired_rows_are_attached_once_and_other_value_errors_are_not_masked() -> None:
    from tests.accounting.ledger.test_named_ledger_revisions import accepted_v3

    proof = accepted_v3()
    ledger = VersionedLedger.restore_versioned(proof.expired_state, proof.superseded)
    collector_steps.archive_retired_rows(PROFILE, ledger, {})
    name = "methodology-retired:" + ledger.namespace
    archive = ledger.archives[name]
    collector_steps.archive_retired_rows(PROFILE, ledger, {})
    assert ledger.archives[name] is archive  # an attached archive is never rebuilt

    frozen = proof.expired_state.frozen[0]
    leaf = frozen.leaves[0]
    conflicting = replace(leaf, debit_raw=leaf.debit_raw + 1)  # same key, different content
    forged = replace(
        proof.expired_state,
        frozen=(*proof.expired_state.frozen, replace(frozen, root="forged", leaves=(conflicting,))),
    )
    with pytest.raises(ValueError, match="conflicting leaves"):
        collector_steps.archive_retired_rows(
            PROFILE, VersionedLedger.restore_versioned(forged, proof.superseded), {}
        )


def test_methodology_retirement_requires_a_coherent_previous_ledger() -> None:
    from ssgrep.accounting.collect import state as collector_state
    from ssgrep.accounting.ledger.ledger_codec import ledger_state_to_dict

    checkpoint = store.Checkpoint(PROFILE, methodology="previous-methodology")
    collector_steps.retire_methodology(PROFILE, checkpoint, METHOD.hash)  # nothing to retire
    assert checkpoint.blobs == {}

    name = collector_state.ledger_blob_name("previous-methodology")
    checkpoint.blobs[name] = store.write_blob(
        PROFILE,
        name,
        {
            "state": ledger_state_to_dict(VersionedLedger(METHOD).state()),
            "superseded": {},
            "collection_scope": {},
        },
    )
    with pytest.raises(store.CollectError) as foreign:
        collector_steps.retire_methodology(PROFILE, checkpoint, METHOD.hash)
    assert foreign.value.condition == "state_corrupt"

    coherent = store.Checkpoint(PROFILE, methodology=METHOD.hash)
    coherent.blobs[collector_state.ledger_blob_name(METHOD.hash)] = checkpoint.blobs[name]
    before = dict(coherent.blobs)
    collector_steps.retire_methodology(PROFILE, coherent, "a-newer-methodology")
    assert coherent.blobs == before  # a current-valuation ledger has nothing to archive


def test_public_receipt_cleanup_never_lists_a_receipt_the_ledger_still_publishes() -> None:
    from ssgrep.accounting.collect import collector

    rows = ledger().rows()
    published = rows.receipts[0].receipt_id
    stale = "stale-receipt"
    directory = storage.open_profile_dir(PROFILE, "receipts")
    try:
        for name in (published, stale):
            storage.atomic_write(directory, name + ".json", b"{}\n")
    finally:
        os.close(directory)
    assert collector._remaining_public(PROFILE, rows, (published, stale)) == (stale,)
    # Without a ledger to protect it, the same file is an ordinary cleanup candidate.
    assert collector._remaining_public(PROFILE, None, (published, stale)) == (published, stale)


def test_reduction_root_documents_must_be_objects() -> None:
    from ssgrep.accounting.collect.partitions import REDUCTION_FORMAT, expand_reduction_document

    manifest = {
        "format": REDUCTION_FORMAT,
        "summary": {"schema": 1},
        "roots": {"root": store.write_blob(PROFILE, "reduction-root", ["not", "an", "object"])},
    }
    with pytest.raises(store.CollectError) as invalid:
        expand_reduction_document(PROFILE, manifest)
    assert invalid.value.condition == "state_corrupt"
