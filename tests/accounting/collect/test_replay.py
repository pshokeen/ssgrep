# ---- Survivor source: test_replay.py ----
"""Full replay vs restarted incremental batches in any order: identical bytes."""

import json
import os
import random
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.capture.fingerprint import key_id
from ssgrep.accounting.capture.journal_reader import read_journal
from ssgrep.accounting.collect import frequency as frequency, store as store
from ssgrep.accounting.collect.collector import CycleBudget
from ssgrep.accounting.collect.frequency import FrequencyAssignments
from ssgrep.accounting.collect.read import Retained
from ssgrep.accounting.collect.replay import Ledger, full_replay, select_namespace
from ssgrep.accounting.collect.state import VersionedLedger, load_state
from ssgrep.accounting.ledger.contracts import TaskId
from ssgrep.accounting.ledger.reduce import methodology_for, native_source_adapter, reduce_profile
from ssgrep.accounting.ledger.valuation_inputs import Candidate, Debit, EvidenceBundle
from ssgrep.accounting.privacy import policy
from tests.accounting.collect import collector_scene as scene, verify_scenes as vs
from tests.accounting.ledger.reduce_builders import (
    KEY,
    MCP,
    PROFILE,
    claude_tool_result,
    claude_tool_use,
    results_json,
    ts,
)
from tests.accounting.ledger.test_lineage import run_late_parent
from tests.accounting.ledger.valuation_builders import METHOD, atom, candidate, debit
from tests.accounting.readers.readers_fixtures import SyntheticHome, append_bytes, build_home, jsonl


def corpus() -> EvidenceBundle:
    """Several tasks, days, overlapping atoms, late evidence, missing and partial debits."""
    t1 = TaskId("root-1", "turn-1", "main")
    t2 = TaskId("root-1", "turn-2", "main")
    unknown = TaskId("root-1", None, "main")
    t3 = TaskId("root-2", "turn-1", "main")
    cands = [
        candidate("a", task=t1, accepted_day="2026-11-02"),
        candidate("b", task=t1, candidate_type="identifier", accepted_day="2026-11-03"),
        candidate(
            "c", task=t2, immediate=(atom("c1", 600, 100, "ev-1"),), accepted_day="2026-11-03"
        ),
        candidate("u", task=unknown, immediate=(atom("u1", 200, 900, "ev-1"),)),
        candidate("d", task=t3, day="2026-11-03", immediate=(atom("d1", 80, 0, "ev-3"),)),
        candidate(
            "late",
            task=t3,
            day="2026-11-03",
            accepted_day="2026-11-30",
            immediate=(atom("l1", 800, 5000, "ev-4"),),
        ),
    ]
    debits = [
        debit("x1", 640, task=t1, accepted_day="2026-11-02"),
        debit("x2", None, task=t1, accepted_day="2026-11-02"),
        debit("x3", 200, task=t2, partial=True),
        debit("x4", 400, task=t3, day="2026-11-03", accepted_day="2026-11-30"),
    ]
    return EvidenceBundle(tuple(cands), tuple(debits))


def batches(bundle: EvidenceBundle, rng: random.Random) -> list[EvidenceBundle]:
    items: list[tuple[str, Candidate | Debit]] = [("c", c) for c in bundle.candidates]
    items += [("d", d) for d in bundle.debits]
    rng.shuffle(items)
    cuts = sorted(rng.sample(range(1, len(items)), 3))
    out = []
    for lo, hi in zip([0, *cuts], [*cuts, len(items)], strict=True):
        part = items[lo:hi]
        cands = tuple(x for k, x in part if isinstance(x, Candidate))
        debs = tuple(x for k, x in part if isinstance(x, Debit))
        out.append(EvidenceBundle(cands, debs))
    return out


def test_incremental_batches_in_different_orders_equal_full_replay_bytes_and_digest():
    reference = full_replay(corpus(), METHOD)
    assert reference.periods and reference.receipts
    rng = random.Random(1012)
    for _ in range(25):
        ledger = Ledger(METHOD)
        for part in batches(corpus(), rng):
            ledger.ingest(part)
            ledger.rows()  # intermediate reductions must not poison later ones
        rows = ledger.rows()
        assert rows.canonical_bytes() == reference.canonical_bytes()
        assert rows.digest == reference.digest


def test_restart_from_persisted_state_matches_a_continuous_run():
    reference = full_replay(corpus(), METHOD)
    parts = batches(corpus(), random.Random(3))
    first = Ledger(METHOD)
    first.ingest(parts[0])
    first.ingest(parts[1])
    resumed = Ledger.restore(first.state())  # process restart
    for part in parts[2:]:
        resumed.ingest(part)
    assert resumed.rows().canonical_bytes() == reference.canonical_bytes()


def test_reingesting_everything_is_idempotent():
    ledger = Ledger(METHOD)
    first = ledger.ingest(corpus())
    again = ledger.ingest(corpus())
    assert first.accepted == 10 and again.accepted == 0 and again.duplicates == 10
    assert ledger.rows().canonical_bytes() == full_replay(corpus(), METHOD).canonical_bytes()


def test_canonical_rows_contain_no_clock_values_and_no_floats():
    rows = full_replay(corpus(), METHOD).canonical_bytes().decode()
    assert "collected_at" not in rows and "mono_ns" not in rows
    full_replay(corpus(), METHOD).canonical_bytes()  # raises TypeError on any float


def test_late_evidence_replays_to_the_same_revisions_as_it_arrived_incrementally():
    ledger = Ledger(METHOD)
    early = corpus().accepted_through("2026-11-05")
    late = EvidenceBundle(
        tuple(c for c in corpus().candidates if c not in early.candidates),
        tuple(d for d in corpus().debits if d not in early.debits),
    )
    ledger.ingest(early)
    before = ledger.rows()
    ledger.ingest(late)
    after = ledger.rows()
    assert after.canonical_bytes() == full_replay(corpus(), METHOD).canonical_bytes()
    kept = {(p.period, p.revision): p.digest for p in before.periods}
    now = {(p.period, p.revision): p.digest for p in after.periods}
    assert all(now[k] == v for k, v in kept.items())  # append-only: old rows unchanged
    assert len(now) > len(kept)  # and the late evidence added a revision


def test_receipt_revisions_are_append_only_and_deterministic():
    ledger = Ledger(METHOD)
    ledger.ingest(corpus().accepted_through("2026-11-05"))
    first = {(r.receipt_id, r.receipt.revision): r.digest for r in ledger.rows().receipts}
    ledger.ingest(corpus())
    second = {(r.receipt_id, r.receipt.revision): r.digest for r in ledger.rows().receipts}
    assert all(second[k] == v for k, v in first.items())


def test_methodology_change_is_a_new_namespace_selected_not_summed():
    other = replace(METHOD, label="v2", frequency_view="freq-2")
    old = full_replay(corpus(), METHOD)
    new = full_replay(corpus(), other)
    assert old.namespace != new.namespace and old.digest != new.digest
    assert select_namespace([old, new], other) is new
    assert select_namespace([old, new], METHOD) is old
    with pytest.raises(KeyError):
        select_namespace([old], other)
    assert all(p.namespace == old.namespace for p in old.periods)
    assert all(r.receipt.methodology == METHOD.hash for r in old.receipts)


def test_clock_and_input_order_never_change_rows():
    import time

    ledger_a, ledger_b = Ledger(METHOD), Ledger(METHOD)
    bundle = corpus()
    ledger_a.ingest(bundle)
    time.sleep(0.01)
    ledger_b.ingest(
        EvidenceBundle(tuple(reversed(bundle.candidates)), tuple(reversed(bundle.debits)))
    )
    assert ledger_a.rows().canonical_bytes() == ledger_b.rows().canonical_bytes()


def test_expired_roots_keep_replayed_totals_and_survive_restart():
    ledger = Ledger(METHOD)
    ledger.ingest(corpus())
    before = {(p.period, p.revision): p.digest for p in ledger.rows().periods if p.revision == 1}
    ledger.expire_roots(["root-1"])
    restored = Ledger.restore(ledger.state())
    rows = restored.rows()
    assert rows.canonical_bytes() == ledger.rows().canonical_bytes()
    flags = {r.receipt.root: r.receipt.evidence_available for r in rows.receipts}
    assert flags == {"root-1": False, "root-2": True}
    assert before  # the pre-expiry rows existed; totals were kept, not recomputed away


# ---- the real collector: incremental runs converge on the single-shot ledger -------------------
# Witness/action locators inside the private ledger state name the transcript files, so runs are
# compared over the SAME paths (each run wipes its predecessor first).

NOW = scene.at(hour=20)
CLOCK = scene.at(hour=9)
TINY = 2_500  # bytes per cycle: forces many cycles over the corpus
ROUNDS = (1, 2)
RUNTIMES = {"claude", "codex", "omp", "pi", "prime-agent", "opencode"}


def seeded(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, scene.at(hour=9))
    assert scene.report(scene.collect(NOW)).generation == 1
    assert scene.counter_total(scene.current()) == 2


def test_compaction_preserves_the_same_evidence_and_ledger_as_uncompacted_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    first = set(vs.evidence_identities())
    vs.pair(vs.projects(tmp_path), 1, scene.at(hour=10))
    monkeypatch.setattr(store, "MAX_SEGMENTS", 1)
    scene.collect(NOW)
    actual = vs.facts()
    assert len(scene.committed().segments) == 1
    assert first < set(actual.evidence)
    assert len(actual.evidence) == len(set(actual.evidence))
    reference_env = vs.env_for(tmp_path)
    reference_env["SSGREP_DATA_DIR"] = str(tmp_path / "reference" / "data")
    # Witness/action locators are ledger identity: only the accounting store may differ.
    with vs.isolated(reference_env):
        vs.enable(pin=True)
        vs.start_capture()
        vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
        vs.pair(vs.projects(tmp_path), 1, scene.at(hour=10))
        scene.collect(NOW)
        expected = vs.facts()
    assert actual.evidence == expected.evidence
    assert actual.ledger == expected.ledger
    assert actual.counters == expected.counters


def test_incremental_arrival_of_the_parent_equals_the_single_shot_ledger(tmp_path: Path) -> None:
    incremental = run_late_parent(tmp_path, parent_later=True)
    vs.wipe(tmp_path)
    shutil.rmtree(tmp_path / "omp-sessions", ignore_errors=True)
    single = run_late_parent(tmp_path, parent_later=False)
    assert incremental.evidence == single.evidence
    assert incremental.ledger == single.ledger  # canonical rows identical to full replay
    assert incremental.snapshot_id == single.snapshot_id


def environment(root: Path, home: SyntheticHome) -> dict[str, str]:
    return {**home.env, "SSGREP_DATA_DIR": str(root / "data")}


def prepare(home: SyntheticHome) -> None:
    """Policy, key, pinned view, capture start, and creditable sessions in the claude store."""
    vs.enable(pin=True)
    vs.start_capture(day=1)  # capture began Oct 1 08:00: every fixture record is later
    for n in (0, 1, 2):
        vs.pair(home.paths["claude"], n, CLOCK)


def append_live(home: SyntheticHome, round_: int) -> None:
    """New records land in an already-known transcript between cycles (an active session)."""
    target = home.paths["claude"] / "-work" / "c1.jsonl"
    call = f"tapp{round_}"
    records = [
        claude_tool_use(
            "c1",
            call,
            MCP,
            {"query": f"late query {round_}"},
            ts=ts(12, 15, round_, 0),
            message_id=f"mapp{round_}",
        ),
        claude_tool_result("c1", call, results_json(), ts=ts(12, 15, round_, 3)),
    ]
    append_bytes(target, jsonl(records))
    os.utime(target, (CLOCK.timestamp(), CLOCK.timestamp()))


def single_shot(root: Path) -> vs.Facts:
    home = build_home(root / "hm")
    with vs.isolated(environment(root, home)):
        prepare(home)
        for round_ in ROUNDS:
            append_live(home, round_)
        result = scene.collect(NOW)
        assert result.state == "published" and scene.report(result).complete
        # the corpus really spans all six runtimes and the ledger has real receipts to compare
        coverage = scene.coverage(scene.current())
        assert set(coverage["sources"]["by_runtime"]) == RUNTIMES
        assert coverage["candidate_receipts"] >= 3
        out = vs.facts()
    vs.wipe(root)
    shutil.rmtree(root, ignore_errors=True)  # the synthetic home too: it is rebuilt from scratch
    return out


def run_incremental(home: SyntheticHome) -> int:
    """Many tiny cycles with new data arriving mid-backlog; returns the number of cycles."""
    cycles = 0
    for round_ in ROUNDS:
        for _ in range(2):  # new data lands while the backlog is still unfinished
            scene.collect(NOW, budget=CycleBudget(bytes=TINY))
            cycles += 1
        append_live(home, round_)
    while True:
        result = scene.collect(NOW, budget=CycleBudget(bytes=TINY))
        cycles += 1
        if scene.report(result).stopped_by == "complete":
            return cycles
        assert cycles < 400, "backlog never finished"


def test_incremental_with_restarts_and_live_appends_equals_single_shot(tmp_path: Path) -> None:
    root = tmp_path / "run"
    expected = single_shot(root)
    home = build_home(root / "hm")  # the same paths again, rebuilt from scratch
    with vs.isolated(environment(root, home)):
        prepare(home)
        cycles = run_incremental(home)
        got = vs.facts()

    assert expected.ledger is not None and expected.evidence
    assert {e[0] for e in expected.evidence} == RUNTIMES
    assert any(owner[-1] for owner in expected.ownership)  # Real target conflicts survive.
    assert (
        len(
            {owner[4][0] for owner in expected.ownership if owner[0] == ("claude", "c1", "call:t5")}
        )
        == 3
    )  # All native copy owners, regardless of which was primary.
    assert cycles >= 6  # a real incremental history, not a one-cycle run
    for facts in (expected, got):
        conflict_rows = [row for row in facts.ownership if row[0] == ("claude", "c1", "call:t3")]
        assert len({row[2] for row in conflict_rows}) == 2
        assert all(row[-1] for row in conflict_rows)
        assert not any(row[0] == ("claude", "c1", "call:t3") for row in facts.native_evidence)
    assert got.native_evidence == expected.native_evidence
    assert got.ownership == expected.ownership
    assert len(set(got.evidence)) == len(got.evidence)
    assert got.ledger == expected.ledger  # full canonical ledger identity
    assert got.counters == expected.counters

    expected_payload = json.loads(expected.current_bytes)
    actual_payload = json.loads(got.current_bytes)
    for field in ("generation", "collected_at", "snapshot_id"):
        expected_payload.pop(field)
        actual_payload.pop(field)
    assert expected_payload.keys() == actual_payload.keys()
    for field in expected_payload.keys() - {"coverage"}:
        assert expected_payload[field] == actual_payload[field], field
    expected_coverage, actual_coverage = expected_payload["coverage"], actual_payload["coverage"]
    assert expected_coverage.keys() == actual_coverage.keys()
    for field in expected_coverage.keys() - {"gaps"}:
        assert expected_coverage[field] == actual_coverage[field], field
    expected_gaps, actual_gaps = expected_coverage["gaps"], actual_coverage["gaps"]
    assert expected_gaps.keys() == actual_gaps.keys()
    admission_diagnostics = {"dropped_before_capture", "dropped_conflict"}
    for reason in expected_gaps.keys() - admission_diagnostics:
        assert expected_gaps[reason] == actual_gaps[reason], reason
    # Fresh/census revisits and which conflicting body was first admitted affect
    # diagnostic admission observations, never unique executions or ledger rows.
    for gaps in (expected_gaps, actual_gaps):
        for reason in admission_diagnostics:
            count = gaps[reason]
            assert isinstance(count, int) and not isinstance(count, bool) and count > 0
    # Each snapshot still addresses its entire actual coverage, never a
    # normalized view. Equal ledger rows do not promise equal operational history.
    assert (got.snapshot_id == expected.snapshot_id) == (actual_payload == expected_payload)


def test_a_from_scratch_replay_of_the_retained_evidence_equals_the_incremental_ledger(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    home = build_home(root / "hm")
    with vs.isolated(environment(root, home)):
        prepare(home)
        run_incremental(home)
        committed = vs.checkpoint()
        assert committed is not None
        accepted = load_state(PROFILE, committed, committed.methodology)
        assert accepted.ledger is not None
        incremental_rows = VersionedLedger.restore_versioned(
            accepted.ledger, accepted.superseded, accepted.archives
        ).rows()
        retained = Retained(PROFILE, committed.segments)
        evidence = retained.items()
        assert retained.replay_available
        events = read_journal(PROFILE).events
        # A manual fixture pin is not the accepted lookup input when the actual
        # index was absent. Replay the exact frozen probe/assignments, not today's
        # rarity or a different scanner methodology.
        view = accepted.frequency_probe or frequency.load_pinned(PROFILE)
        assert view is not None and methodology_for(view) == accepted.ledger.methodology
        assignments = FrequencyAssignments(
            view,
            assignments=accepted.frequency_assignments,
            loader=lambda view_id: frequency.load_view(PROFILE, view_id),
            fingerprint_key_id=key_id(KEY),
        )
        ownership = policy.read_policy_file()
        profile_policy = ownership.profiles[PROFILE]
        reduction = reduce_profile(
            PROFILE,
            evidence,
            events,
            policy=profile_policy,
            frequency=view,
            fingerprint_key=KEY,
            ownership=ownership,
            frequency_assignments=assignments,
            native_sources=native_source_adapter(
                profile_policy,
                now_ms=int(NOW.timestamp() * 1000),
                ownership=ownership,
            ),
        )
        assert not assignments.unavailable
        assert assignments.records() == accepted.frequency_assignments
    replayed = VersionedLedger(methodology_for(view))
    replayed.reconcile(
        dict(zip(reduction.bundle_roots, reduction.bundles, strict=True)), today="2026-10-12"
    )
    assert reduction.bundles and replayed.rows().receipts  # a real ledger, not two empty ones
    assert replayed.rows().canonical_bytes() == incremental_rows.canonical_bytes()
    assert replayed.rows().digest == incremental_rows.digest


@pytest.mark.parametrize("changed_first", [False, True])
def test_a_conflicted_lookup_variant_has_no_financial_authority_after_compact_restart(
    tmp_path: Path, changed_first: bool
) -> None:
    from copy import deepcopy

    from ssgrep.accounting.collect.replay import EvidenceReplay, evidence_from_line, evidence_line
    from ssgrep.accounting.ledger.valuation_inputs import Eligibility
    from ssgrep.accounting.readers.base import EvidenceLedger
    from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader
    from ssgrep.accounting.readers.timeline import build_timelines
    from tests.accounting.ledger.reduce_builders import policy as fixture_policy, source_session
    from tests.accounting.ledger.test_reduce_evidence import FACT, VIEW, consumer
    from tests.accounting.readers.readers_fixtures import write_records

    vs.enable()
    records = consumer()
    changed_record = deepcopy(records[1])
    changed_record["message"]["content"][0]["input"]["query"] = "different supported search"
    files = {
        "witness": source_session("src1", FACT),
        "consumer": records,
        "changed": [changed_record],
    }
    reader = ClaudeReader()
    decoded = {}
    for name, rows in files.items():
        path = write_records(scene.transcripts(tmp_path) / f"{name}.jsonl", rows)
        decoded[name] = reader.read(reader.source_for(path)).evidence
    original = [*decoded["witness"], *decoded["consumer"]]
    baseline = reduce_profile(
        PROFILE, original, [], policy=fixture_policy(), frequency=VIEW, fingerprint_key=KEY
    )
    assert any(row.candidate.eligibility is Eligibility.SUPPORTED for row in baseline.candidates)
    assert any(
        debit.chars is not None and debit.chars > 0
        for bundle in baseline.bundles
        for debit in bundle.debits
    )

    ordered = (
        [*decoded["changed"], *original] if changed_first else [*original, *decoded["changed"]]
    )
    ledger = EvidenceLedger()
    accepted = []
    for item in ordered:
        if ledger.admit(item):
            accepted.append(item)
        elif (claim := ledger.ownership_claim(item)) is not None:
            accepted.append(claim)
    compact = [evidence_from_line(evidence_line(item)) for item in accepted]
    replay = EvidenceReplay(PROFILE)
    restored = replay.hydrate_many(compact)
    assert not replay.unavailable
    timelines = build_timelines(restored)
    [changed] = decoded["changed"]
    assert changed.native_identity in timelines.conflicted and timelines.conflicts == 1
    reduced = reduce_profile(
        PROFILE, restored, [], policy=fixture_policy(), frequency=VIEW, fingerprint_key=KEY
    )
    assert not reduced.invocations
    assert not reduced.candidates
    assert not any(bundle.candidates or bundle.debits for bundle in reduced.bundles)


def test_hydration_preserves_plain_evidence_and_marks_unresolvable_compact_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting.collect.replay import EvidenceReplay
    from tests.accounting.conftest import build_evidence

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)

    plain = build_evidence()
    compact = replace(
        plain,
        source_id="different-runtime:/unavailable/source",
        payload={**plain.payload, "_evidence_format": "compact-v1"},
    )
    replay = EvidenceReplay(PROFILE)

    assert replay.hydrate(plain) is plain
    unavailable = replay.hydrate(compact)
    assert unavailable.identity == compact.identity
    assert unavailable.payload["_replay_unavailable"] == "source_unresolved"
    assert replay.unavailable == {(compact.runtime, compact.session_id): "source_unresolved"}
    assert len(replay.failures) == 1
    assert replay.adapter is not None


def test_bulk_hydration_without_collection_preserves_plain_and_rejects_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting.collect.replay import EvidenceReplay
    from tests.accounting.conftest import build_evidence

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))

    plain = build_evidence()
    compact = replace(plain, payload={**plain.payload, "_evidence_format": "compact-v1"})
    replay = EvidenceReplay(PROFILE)

    hydrated = replay.hydrate_many([plain, compact])

    assert hydrated[0] is plain
    assert hydrated[1].identity == compact.identity
    assert hydrated[1].payload["_replay_unavailable"] == "source_unapproved"
    assert replay.unavailable == {(compact.runtime, compact.session_id): "source_unapproved"}
    assert len(replay.failures) == 1
    assert replay.adapter is None


@pytest.mark.parametrize(
    ("drift", "reason"),
    [
        ({"ts_utc": "2026-11-01T07:00:02Z"}, "source_timestamp_drift"),
        ({"branch_id": "other-branch"}, "source_ownership_drift"),
        (
            {"payload": {"name": "bash", "arguments": {"command": "ls"}, "cwd": "/other"}},
            "source_ownership_drift",
        ),
    ],
)
def test_compact_replay_rejects_verified_identity_with_drifted_authority(
    drift: dict[str, object], reason: str
) -> None:
    from types import SimpleNamespace

    from ssgrep.accounting.collect.replay import EvidenceReplay
    from tests.accounting.conftest import build_evidence

    source_record = build_evidence(
        payload={"name": "bash", "arguments": {"command": "ls"}, "cwd": "/expected"}
    )
    compact = replace(
        source_record,
        source_id="claude:/replayed/source.jsonl",
        payload={**source_record.payload, "_evidence_format": "compact-v1"},
    )
    actual = replace(source_record, **drift)

    class MatchingAdapter:
        def read_record(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(evidence=source_record, status=SimpleNamespace(value="ok"))

    matching = EvidenceReplay(PROFILE, adapter=MatchingAdapter())
    assert matching.hydrate(compact) == source_record
    assert matching.unavailable == {} and matching.failures == []

    class Adapter:
        def read_record(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(evidence=actual, status=SimpleNamespace(value="ok"))

    replay = EvidenceReplay(PROFILE, adapter=Adapter())
    unavailable = replay.hydrate(compact)

    assert unavailable.identity == compact.identity
    assert unavailable.payload["_replay_unavailable"] == reason
    assert replay.unavailable == {(compact.runtime, compact.session_id): reason}


def cached_compact(*, origin: bool = True):
    from ssgrep.accounting.collect.replay_evidence import cache_collected_evidence
    from ssgrep.accounting.readers.base import evidence_digest
    from ssgrep.accounting.readers.native_identity import with_origin
    from tests.accounting.ledger.test_minimal_evidence import evidence

    base = evidence()
    if origin:
        item = with_origin(base, "claude:cache-hydration", "call:a", "native")
    else:
        item = replace(
            base,
            origin_domain=None,
            origin_id=None,
            origin_status="legacy_session_scoped",
            digest=evidence_digest(base.kind, base.payload),
        )
    cache_collected_evidence(PROFILE, [item])
    refs = store.write_segments(PROFILE, [item])
    (compact,) = store.iter_evidence(PROFILE, refs)
    return item, compact


@pytest.mark.parametrize("origin", [True, False])
def test_compact_hydration_serves_the_authorized_plaintext_cache_without_reading_the_source(
    origin: bool,
) -> None:
    from ssgrep.accounting.collect.replay import EvidenceReplay
    from tests.accounting.ledger.test_minimal_evidence import CANARY

    policy.enable_collection(PROFILE, evidence_cache=(1, 16384))
    item, compact = cached_compact(origin=origin)
    assert compact.payload["_evidence_format"] == "compact-v1"
    assert CANARY not in json.dumps(compact.payload)
    assert not Path(item.source_id.removeprefix("claude:")).exists()

    single = EvidenceReplay(PROFILE)
    restored = single.hydrate(compact)
    bulk = EvidenceReplay(PROFILE)
    [restored_many] = bulk.hydrate_many([compact])
    for replay, row in ((single, restored), (bulk, restored_many)):
        assert row.payload == item.payload and row.identity == item.identity
        assert replay.adapter is None and replay.unavailable == {} and replay.failures == []

    # Control: without the cache the same compact row cannot leave the unapproved source.
    current = policy.load_profile_policy(PROFILE)
    policy.save_profile(
        replace(
            current,
            evidence_cache_enabled=False,
            evidence_cache_retention_days=0,
            evidence_cache_max_bytes=0,
        )
    )
    uncached = EvidenceReplay(PROFILE)
    assert uncached.hydrate(compact).payload["_replay_unavailable"] == "source_unapproved"
    assert uncached.unavailable == {(compact.runtime, compact.session_id): "source_unapproved"}


def test_compact_hydration_without_active_collection_is_unapproved_for_each_state(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.replay import EvidenceReplay
    from tests.accounting.conftest import build_evidence

    plain = build_evidence()
    compact = replace(
        plain,
        source_id=f"claude:{scene.transcripts(tmp_path) / 'gone.jsonl'}",
        payload={**plain.payload, "_evidence_format": "compact-v1"},
    )
    never_enabled = EvidenceReplay(PROFILE)
    assert never_enabled.hydrate(compact).payload["_replay_unavailable"] == "source_unapproved"
    assert never_enabled.adapter is None

    policy.enable_collection(PROFILE)
    policy.pause(PROFILE)
    paused = EvidenceReplay(PROFILE)
    assert paused.hydrate(compact).payload["_replay_unavailable"] == "source_unapproved"
    assert paused.unavailable == {(compact.runtime, compact.session_id): "source_unapproved"}

    policy.resume(PROFILE)
    resumed = EvidenceReplay(PROFILE)
    assert resumed.hydrate(compact).payload["_replay_unavailable"] == "source_missing"


def test_evidence_cache_keeps_out_of_scope_records_out_and_drops_unreadable_entries() -> None:
    from ssgrep.accounting.collect.replay_evidence import (
        CACHE_DIR,
        cache_collected_evidence,
        maintain_cache,
    )
    from ssgrep.accounting.privacy import storage
    from tests.accounting.ledger.test_minimal_evidence import evidence

    approved = "/approved"
    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 16384))
    policy.save_profile(replace(enabled, roots=(approved,)))
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / CACHE_DIR
    inside = replace(evidence(), payload={**evidence().payload, "cwd": approved})
    outside = replace(
        evidence(), record_id="call:out", payload={**evidence().payload, "cwd": "/elsewhere"}
    )
    cache_collected_evidence(PROFILE, [outside])
    assert list(directory.iterdir()) == []
    cache_collected_evidence(PROFILE, [inside])
    (kept,) = directory.iterdir()

    junk = directory / ("0" * 64 + ".json")
    junk.write_bytes(b'{"payload": "not an evidence row"')
    junk.chmod(0o600)
    unsupported = directory / ("1" * 64 + ".json")
    unsupported.write_bytes(
        json.dumps({"evidence_format": "future-v9", "payload": {}}, separators=(",", ":")).encode()
    )
    unsupported.chmod(0o600)
    maintain_cache(PROFILE)
    assert [path.name for path in directory.iterdir()] == [kept.name]


def test_cache_writer_releases_the_consent_lock_when_the_cache_directory_is_unsafe(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.replay_evidence import CACHE_DIR, CacheWriter, maintain_cache
    from ssgrep.accounting.privacy import storage

    policy.enable_collection(PROFILE, evidence_cache=(1, 16384))
    collector = storage.data_root() / "accounting" / PROFILE / "collector"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.close(storage.open_profile_dir(PROFILE, "collector"))
    (collector / CACHE_DIR).symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(OSError):
        CacheWriter(PROFILE, create=False)
    (collector / CACHE_DIR).unlink()
    # The failed constructor must not leave the accounting consent lock held.
    assert maintain_cache(PROFILE).evidence_cache_enabled is True


def test_nothing_unavailable_leaves_the_reduction_and_its_bundles_untouched(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.replay import preserve_unavailable
    from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader
    from tests.accounting.ledger.reduce_builders import policy as fixture_policy, source_session
    from tests.accounting.ledger.test_reduce_evidence import FACT, VIEW, consumer
    from tests.accounting.readers.readers_fixtures import write_records

    vs.enable()
    reader = ClaudeReader()
    decoded = []
    for name, rows in {"witness": source_session("src1", FACT), "consumer": consumer()}.items():
        path = write_records(scene.transcripts(tmp_path) / f"{name}.jsonl", rows)
        decoded.extend(reader.read(reader.source_for(path)).evidence)
    reduction = reduce_profile(
        PROFILE, decoded, [], policy=fixture_policy(), frequency=VIEW, fingerprint_key=KEY
    )
    assert reduction.invocations and reduction.bundles
    ledger = VersionedLedger(methodology_for(VIEW))

    same, desired = preserve_unavailable(ledger, reduction, None, {}, {})
    assert same is reduction
    assert desired == dict(zip(reduction.bundle_roots, reduction.bundles, strict=True))
    assert "historical_replay_unavailable" not in same.warnings

    blocked = reduction.invocations[0].invocation_id
    held, _ = preserve_unavailable(ledger, reduction, None, {}, {blocked: "frequency_unavailable"})
    assert blocked not in {row.invocation_id for row in held.invocations}
    assert held.coverage["frequency_revision_unavailable"] == 1
    assert "historical_replay_unavailable" in held.warnings


def resolved_witness(resolver, ref, consumer_at):
    from ssgrep.accounting.collect.replay_evidence import Token
    from tests.accounting.ledger.test_reduce_evidence import PATH

    return resolver.resolve(Token("path", PATH), (ref,), consumer_at)


def test_a_native_source_is_read_once_and_reused_for_the_same_reference(tmp_path: Path) -> None:
    from ssgrep.accounting.collect.replay_evidence import Consumer, Resolver
    from ssgrep.accounting.readers.timeline import build_timelines
    from tests.accounting.readers.test_native_source import NOW, REF, old_source

    _, _, adapter = old_source(tmp_path)
    resolver = Resolver(build_timelines([]), adapter.scoper, native_sources=adapter)
    consumer_at = Consumer("claude:cons", ("claude", "cons"), 1, NOW)

    first = resolved_witness(resolver, REF, consumer_at)
    spent = adapter.bytes_read
    assert spent > 0 and first.immediate and all(row.usable for row in first.immediate)
    again = resolved_witness(resolver, REF, consumer_at)
    assert adapter.bytes_read == spent
    assert again == first


def source_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_native_origin_claimed_by_another_session_has_no_valuation_authority(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.replay_evidence import Consumer, Resolver
    from ssgrep.accounting.readers.timeline import build_timelines
    from tests.accounting.ledger.reduce_builders import claude_evidence
    from tests.accounting.readers.test_native_source import NOW, REF, old_source

    path, _, adapter = old_source(tmp_path / "native")
    retained = claude_evidence(tmp_path / "retained", {"-work/src1.jsonl": source_rows(path)})
    consumer_at = Consumer("claude:cons", ("claude", "cons"), 1, NOW)
    owned = build_timelines(retained)
    assert not owned.ownership_unknown
    clean = resolved_witness(Resolver(owned, adapter.scoper), REF, consumer_at)
    assert clean.immediate and all(row.usable for row in clean.immediate)

    forks = [
        replace(
            row,
            session_id="src1-fork",
            source_id=row.source_id.replace("src1.jsonl", "src1x.jsonl"),
        )
        for row in retained
    ]
    assert [row.native_identity for row in forks] == [row.native_identity for row in retained]
    contested = build_timelines([*retained, *forks])
    assert contested.ownership_unknown
    result = resolved_witness(Resolver(contested, adapter.scoper), REF, consumer_at)
    assert ("source_ownership_unknown", "claude:src1") in result.reasons
    assert result.immediate and not any(row.usable for row in result.immediate)


def alias_adapter(tmp_path: Path):
    from tests.accounting.readers.test_native_source import old_source

    path, reference, adapter = old_source(tmp_path)
    alias = "claude:indexed-filename:ep:0"
    adapter.references = {
        alias: replace(reference, ref=alias, indexed_session_id="indexed-filename")
    }
    return path, alias, adapter


def test_a_native_witness_whose_origin_was_retained_with_other_content_is_conflicted(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.replay_evidence import Consumer, Resolver
    from ssgrep.accounting.readers.timeline import build_timelines
    from tests.accounting.ledger.reduce_builders import claude_evidence
    from tests.accounting.readers.test_native_source import NOW

    path, alias, adapter = alias_adapter(tmp_path / "native")
    consumer_at = Consumer("claude:cons", ("claude", "cons"), 1, NOW)
    control = resolved_witness(
        Resolver(build_timelines([]), adapter.scoper, native_sources=adapter), alias, consumer_at
    )
    assert control.immediate and all(row.usable for row in control.immediate)

    rows = source_rows(path)
    for row in rows:
        content = row.get("message", {}).get("content")
        if isinstance(content, list) and content and content[0].get("type") == "tool_result":
            content[0]["content"] = "a different result than the native source holds"
    retained = claude_evidence(tmp_path / "retained", {"-work/src1.jsonl": rows})
    assert {row.native_identity for row in retained} >= {"claude:src1:result:w1"}
    result = resolved_witness(
        Resolver(build_timelines(retained), adapter.scoper, native_sources=adapter),
        alias,
        consumer_at,
    )
    assert ("conflicting_source_variants", "claude:src1") in result.reasons
    assert result.immediate and not any(row.usable for row in result.immediate)


def test_a_native_witness_copied_into_a_foreign_session_has_no_authority_to_the_consumer(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.replay_evidence import Consumer, Resolver
    from ssgrep.accounting.readers.timeline import build_timelines
    from tests.accounting.ledger.reduce_builders import claude_evidence
    from tests.accounting.readers.test_native_source import NOW

    path, alias, adapter = alias_adapter(tmp_path / "native")
    consumer_at = Consumer("claude:cons", ("claude", "cons"), 1, NOW)
    retained = claude_evidence(tmp_path / "retained", {"-work/src1.jsonl": source_rows(path)})
    copies = [
        replace(
            row,
            session_id="src1-copy",
            source_id=row.source_id.replace("src1.jsonl", "src1-copy.jsonl"),
        )
        for row in retained
    ]
    assert [row.native_identity for row in copies] == [row.native_identity for row in retained]
    result = resolved_witness(
        Resolver(build_timelines(copies), adapter.scoper, native_sources=adapter),
        alias,
        consumer_at,
    )
    assert ("source_ownership_unknown", "claude:src1") in result.reasons
    assert result.immediate and not any(row.usable for row in result.immediate)


@pytest.mark.parametrize("runtime", ["pi", "omp"])
@pytest.mark.parametrize("retained", ["before_setup", "after_setup"])
def test_header_only_source_matches_fully_retained_credit(tmp_path, runtime, retained):
    from ssgrep.accounting.collect.replay_evidence import Consumer, Resolver, Token
    from ssgrep.accounting.readers.timeline import build_timelines
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    path, reader, _, _, reference, adapter = json_scene(tmp_path, runtime)
    path.write_bytes(path.read_bytes().replace(b"answer", b"answer_value"))
    originals = reader.read(reader.source_for(path)).evidence
    adapter.references = {reference.ref: reference}
    consumer_at = Consumer("claude:consumer", ("claude", "consumer"), 1, adapter.now_ms)
    token = Token("identifier", "answer_value")
    complete = Resolver(build_timelines(originals), adapter.scoper)
    expected = complete.resolve(token, (reference.ref,), consumer_at)
    assert expected.immediate and all(atom.usable for atom in expected.immediate)
    kept = (
        originals
        if retained == "after_setup"
        else [item for item in originals if item.kind == "session_header"]
    )
    assert kept
    resolver = Resolver(build_timelines(kept), adapter.scoper, native_sources=adapter)
    actual = resolver.resolve(token, (reference.ref,), consumer_at)
    assert actual == expected
    assert {atom.actor for atom in actual.immediate} == {f"{runtime}:s"}
    spent = adapter.bytes_read
    assert resolver.resolve(token, (reference.ref,), consumer_at) == actual
    assert adapter.bytes_read == spent


@pytest.mark.parametrize(
    "case", ["expired", "identity", "scope", "conflict", "header_drift", "unavailable"]
)
def test_header_only_fallback_preserves_source_authority_gates(tmp_path, case):
    from ssgrep.accounting.collect.replay_evidence import Consumer, Resolver, Token
    from ssgrep.accounting.readers.timeline import build_timelines
    from ssgrep.accounting.views.filter import Scoper
    from tests.accounting.ledger.reduce_builders import policy
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    path, reader, _, _, reference, adapter = json_scene(tmp_path, "pi")
    path.write_bytes(path.read_bytes().replace(b"answer", b"answer_value"))
    originals = reader.read(reader.source_for(path)).evidence
    kept = [item for item in originals if item.kind == "session_header"]
    assert kept
    if case == "expired":
        reference = replace(reference, expires_at_ms=adapter.now_ms)
    elif case == "identity":
        reference = replace(reference, session_id="other")
    elif case == "scope":
        adapter.scoper = Scoper(policy("/elsewhere", profile="org"))
    elif case == "conflict":
        kept.append(replace(kept[0], digest="0" * 64))
    elif case == "header_drift":
        kept = [replace(kept[0], digest="0" * 64)]
    else:
        kept = [
            replace(kept[0], payload={**kept[0].payload, "_replay_unavailable": "source_expired"})
        ]
    adapter.references = {reference.ref: reference}
    resolver = Resolver(build_timelines(kept), adapter.scoper, native_sources=adapter)
    result = resolver.resolve(
        Token("identifier", "answer_value"),
        (reference.ref,),
        Consumer("claude:consumer", ("claude", "consumer"), 1, adapter.now_ms),
    )
    assert result.immediate and not any(atom.usable for atom in result.immediate)
