"""9.x collector_read: discovery, baseline, bounded resumable reading (real files)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import collector_modes, collector_read, storage
from ssgrep.accounting.collector_inputs import AdmitRules, ClassifyingLedger
from ssgrep.accounting.collector_read import (
    ReadOutcome,
    baseline_record,
    discover,
    make_readers,
    read_sources,
)
from ssgrep.accounting.collector_store import (
    CONSUMED_KINDS,
    SourceRecord,
    source_key,
)
from ssgrep.accounting.collector_types import MIB, CollectOptions, CycleBudget
from ssgrep.accounting.contracts import RUNTIMES, RawEvidence
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.conftest import build_evidence
from tests.accounting.readers_fixtures import (
    T0,
    append_bytes,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    jsonl,
    opencode_add_message,
    opencode_add_part,
    opencode_add_session,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    write_records,
)
from tests.accounting.reduce_builders import MCP, PROFILE, ts

BIG = 10**9


def admit_all(_: RawEvidence) -> str | None:
    return None


def env_for(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "CODEX_HOME": str(tmp_path / "codex"),
        "OMP_SESSIONS_DIR": str(tmp_path / "omp"),
        "PI_SESSION_DIR": str(tmp_path / "pi"),
        "PRIME_AGENT_SESSION_DIR": str(tmp_path / "prime" / "sessions"),
        "SSGREP_OPENCODE_DB": str(tmp_path / "oc" / "opencode.db"),
    }


def claude_dir(tmp_path: Path) -> Path:
    return tmp_path / "claude" / "projects" / "-work-demo"


def claude_records(session: str, calls: int, pad: str = "") -> list[dict]:
    records = [claude_prompt(session, "go " + pad)]
    for n in range(calls):
        call = f"{session}-t{n}"
        records.append(claude_tool_use(session, call, "Bash", {"command": f"echo {n} {pad}"}))
        records.append(claude_tool_result(session, call, "out"))
    return records


class Cycle:
    """Drives read_sources the way a collector does: known records carry between calls."""

    def __init__(self, tmp_path: Path) -> None:
        self.env = env_for(tmp_path)
        self.readers = make_readers(self.env)
        self.ledger = ClassifyingLedger()
        self.known: dict[str, SourceRecord] = {}
        self.evidence: list[RawEvidence] = []

    def run(self, **overrides: Any) -> ReadOutcome:
        args: dict[str, Any] = {
            "discovered": discover(self.readers),
            "known": self.known,
            "admit": admit_all,
            "ledger": self.ledger,
            "byte_budget": BIG,
            "time_budget": 3600.0,
            "clock": lambda: 0.0,
        }
        args.update(overrides)
        outcome = read_sources(**args)
        self.known = dict(outcome.sources)
        self.evidence.extend(outcome.evidence)
        return outcome

    def drain(self, *, limit: int = 200, **overrides: Any) -> list[ReadOutcome]:
        outcomes = []
        for _ in range(limit):
            outcomes.append(self.run(**overrides))
            if outcomes[-1].stopped_by == "complete" and all(
                record.complete for record in outcomes[-1].sources.values()
            ):
                return outcomes
        raise AssertionError("did not converge")


def ids(items: list[RawEvidence]) -> list[tuple[str, str, str]]:
    return [(e.runtime, e.session_id, e.record_id) for e in items]


# ---------------------------------------------------------------- discovery


def test_make_readers_covers_every_runtime_and_discover_keys_by_source_key(tmp_path: Path) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 1))
    write_records(tmp_path / "pi" / "p1.jsonl", [pi_header("p1")])
    readers = make_readers(env_for(tmp_path))
    assert set(readers) == set(RUNTIMES)
    found = discover(readers)
    assert list(found) == sorted(found)
    assert len(found) == 2
    for key, (reader, source) in found.items():
        assert key == source_key(source.source_id)
        assert reader is readers[source.runtime]


def test_discover_survives_one_reader_raising_oserror(tmp_path: Path) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 1))
    readers = make_readers(env_for(tmp_path))

    def boom() -> list:
        raise PermissionError("no")

    readers["codex"].discover = boom
    found = discover(readers)
    assert [s.runtime for _, s in found.values()] == ["claude"]


# ------------------------------------------------------------------ budgets


def test_byte_budget_resumes_without_duplicates_and_equals_unbudgeted_union(
    tmp_path: Path,
) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 12, "x" * 40))
    write_records(claude_dir(tmp_path) / "s2.jsonl", claude_records("s2", 7, "y" * 40))
    write_records(
        tmp_path / "pi" / "p1.jsonl",
        [
            pi_header("p1"),
            *[
                r
                for n in range(6)
                for r in (
                    pi_tool_call(f"e{n}", f"c{n}", "bash", {"command": "ls"}),
                    pi_tool_result(f"r{n}", f"c{n}", "ok"),
                )
            ],
        ],
    )
    whole = Cycle(tmp_path)
    one = whole.run()
    assert one.stopped_by == "complete" and one.evidence

    chunked = Cycle(tmp_path)
    outcomes = chunked.drain(byte_budget=300)
    assert len(outcomes) > 3
    assert {o.stopped_by for o in outcomes[:-1]} == {"byte_budget"}
    assert sorted(ids(chunked.evidence)) == sorted(ids(one.evidence))
    assert len(ids(chunked.evidence)) == len(set(ids(chunked.evidence)))
    assert all(r.complete and not r.more and r.pending_tail == 0 for r in chunked.known.values())

    # Reader checkpoints agree; persistent fairness counters describe different schedules.
    def reader_state(record: SourceRecord) -> dict:
        state = dict(record.state)
        state["extra"] = {k: v for k, v in state["extra"].items() if not k.startswith("collector_")}
        return state

    assert {k: reader_state(r) for k, r in chunked.known.items()} == {
        k: reader_state(r) for k, r in whole.known.items()
    }


def test_record_larger_than_remaining_budget_is_read_whole_then_call_stops(
    tmp_path: Path,
) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    records = claude_records("s1", 4, "z" * 600)
    write_records(path, records)
    lines = path.read_bytes().splitlines(keepends=True)
    cycle = Cycle(tmp_path)
    first = cycle.run(byte_budget=10)  # far smaller than the first record
    assert first.stopped_by == "byte_budget"
    assert first.bytes_read == len(lines[0])  # exactly one whole record, not a fragment
    [record] = first.sources.values()
    assert record.offset == len(lines[0]) and record.more and not record.complete
    assert ids(first.evidence) == [("claude", "s1", "turn:u-1")]


def test_time_budget_with_fake_clock_stops_between_sources(tmp_path: Path) -> None:
    for n in range(3):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 2))
    ticks = iter(range(0, 1000))
    outcome = Cycle(tmp_path).run(clock=lambda: float(next(ticks)), time_budget=2.5)
    assert outcome.stopped_by == "time_budget"
    done = [r for r in outcome.sources.values() if r.complete]
    untouched = [r for r in outcome.sources.values() if r.offset == 0]
    assert done and untouched and len(done) + len(untouched) == 3
    assert all(not r.complete for r in untouched)


def test_stop_requested_before_any_source_reads_nothing(tmp_path: Path) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 2))
    cycle = Cycle(tmp_path)
    outcome = cycle.run(should_stop=lambda: True)
    assert outcome.stopped_by == "stop_requested"
    assert outcome.evidence == [] and outcome.bytes_read == 0
    [record] = outcome.sources.values()
    assert record.offset == 0 and not record.complete
    # resuming reads everything
    assert cycle.run().stopped_by == "complete" and len(cycle.evidence) == 5


def test_stop_requested_midway_finishes_current_read_and_skips_the_rest(tmp_path: Path) -> None:
    for n in range(3):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 2))
    flag = {"stop": False}
    cycle = Cycle(tmp_path)
    outcome = cycle.run(
        should_stop=lambda: flag["stop"],
        on_source=lambda key, record: flag.update(stop=True),
    )
    assert outcome.stopped_by == "stop_requested"
    assert sum(r.complete for r in outcome.sources.values()) == 1


def test_unreached_source_stat_refresh_shows_current_lag(tmp_path: Path) -> None:
    first = claude_dir(tmp_path) / "a.jsonl"
    second = claude_dir(tmp_path) / "b.jsonl"
    write_records(first, claude_records("a", 1))
    write_records(second, claude_records("b", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    key_b = source_key(f"claude:{second}")
    assert cycle.known[key_b].complete
    old_size = cycle.known[key_b].size
    append_bytes(second, jsonl(claude_records("b2", 1)))
    append_bytes(first, jsonl(claude_records("a2", 1)))
    outcome = cycle.run(should_stop=lambda: True)  # reaches nothing
    record = outcome.sources[key_b]
    assert record.size == second.stat().st_size > old_size
    assert record.mtime_ns == second.stat().st_mtime_ns
    assert not record.complete and record.offset == old_size  # lag = size - offset > 0
    assert outcome.bytes_read == 0


def test_unreached_source_that_vanished_is_unreadable_not_an_exception(tmp_path: Path) -> None:
    path = claude_dir(tmp_path) / "a.jsonl"
    write_records(path, claude_records("a", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    discovered = discover(cycle.readers)
    path.unlink()
    outcome = cycle.run(discovered=discovered, should_stop=lambda: True)
    [record] = outcome.sources.values()
    assert record.status == "unreadable" and not record.complete


# ----------------------------------------------------------------- baseline


def test_baseline_skips_old_bytes_but_reads_later_appended_records(tmp_path: Path) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 3))
    size = path.stat().st_size
    cycle = Cycle(tmp_path)
    outcome = cycle.run(baseline_new=True)
    [record] = outcome.sources.values()
    assert outcome.evidence == [] and outcome.bytes_read == 0
    assert record.baseline == "skipped" and record.complete
    assert record.skipped_bytes == record.offset == size == record.size
    append_bytes(path, jsonl([claude_tool_use("s1", "late", "Bash", {"command": "ls"})]))
    later = cycle.run(baseline_new=True)
    assert ids(later.evidence) == [("claude", "s1", "call:late")]
    assert later.sources[record.key].baseline == "skipped"  # provenance kept
    assert later.sources[record.key].complete


def test_baseline_stops_at_last_newline_so_a_torn_tail_is_read_once_when_finished(
    tmp_path: Path,
) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 1))
    whole = path.stat().st_size
    pending = jsonl([claude_tool_use("s1", "late", "Bash", {"command": "ls"})])
    append_bytes(path, pending[:-1])
    reader = make_readers(env_for(tmp_path))["claude"]
    [source] = reader.discover()
    record = baseline_record(reader, source)
    assert record.offset == whole and record.skipped_bytes == whole
    assert record.pending_tail == len(pending) - 1 and not record.complete
    cycle = Cycle(tmp_path)
    cycle.known = {record.key: record}
    first = cycle.run()
    assert first.evidence == [] and first.sources[record.key].pending_tail == len(pending) - 1
    append_bytes(path, pending[-1:])
    second = cycle.run()
    assert ids(second.evidence) == [("claude", "s1", "call:late")]
    assert second.sources[record.key].complete
    assert cycle.run().evidence == []


def test_baseline_finds_last_newline_across_chunks_of_a_huge_unterminated_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pi" / "p1.jsonl"
    write_records(path, [pi_header("p1")])
    head = path.stat().st_size
    append_bytes(path, b"x" * (3 * 1024 * 1024))  # no newline anywhere in the tail
    reader = make_readers(env_for(tmp_path))["pi"]
    [source] = reader.discover()
    record = baseline_record(reader, source)
    assert record.offset == head and record.pending_tail == 3 * 1024 * 1024
    assert baseline_record(reader, source).state == record.state


def test_baseline_of_empty_and_newline_free_files(tmp_path: Path) -> None:
    empty = tmp_path / "pi" / "e.jsonl"
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")
    nonl = tmp_path / "pi" / "n.jsonl"
    nonl.write_bytes(b'{"type":"session"')
    reader = make_readers(env_for(tmp_path))["pi"]
    records = {s.path.name: baseline_record(reader, s) for s in reader.discover()}
    assert records["e.jsonl"].offset == 0 and records["e.jsonl"].complete
    assert records["n.jsonl"].offset == 0 and records["n.jsonl"].pending_tail == 17


def test_baseline_of_missing_source_is_unreadable_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "pi" / "p1.jsonl"
    write_records(path, [pi_header("p1")])
    reader = make_readers(env_for(tmp_path))["pi"]
    [source] = reader.discover()
    path.unlink()
    assert baseline_record(reader, source).status == "unreadable"


# ----------------------------------------------------------- torn tail / reset


def test_torn_tail_never_accepted_then_read_exactly_once(tmp_path: Path) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 1))
    full = jsonl([claude_tool_use("s1", "tail", "Bash", {"command": "ls"})])
    append_bytes(path, full[:-1])
    cycle = Cycle(tmp_path)
    first = cycle.run()
    [record] = first.sources.values()
    assert record.pending_tail == len(full) - 1 and not record.complete and not record.more
    assert ("claude", "s1", "call:tail") not in ids(first.evidence)
    again = cycle.run()
    assert again.evidence == [] and again.sources[record.key].pending_tail == len(full) - 1
    append_bytes(path, b"\n")
    done = cycle.run()
    assert ids(done.evidence) == [("claude", "s1", "call:tail")]
    assert done.sources[record.key].complete and done.sources[record.key].pending_tail == 0
    assert ids(cycle.evidence).count(("claude", "s1", "call:tail")) == 1


def test_truncation_and_rotation_produce_reset_gap_and_reread_without_crash(
    tmp_path: Path,
) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 4))
    cycle = Cycle(tmp_path)
    cycle.run()
    write_records(path, claude_records("s9", 1))  # truncated, different content
    outcome = cycle.run()
    [record] = outcome.sources.values()
    assert outcome.resets == 1 and outcome.gaps["reset_truncated"] == 1
    assert record.resets == 1 and record.gaps["reset_truncated"] == 1
    assert {e.session_id for e in outcome.evidence} == {"s9"} and len(outcome.evidence) == 3
    # rotation: same name, new inode, at least as large
    write_records(path, claude_records("s9", 1))
    rotated = tmp_path / "rot.jsonl"
    rotated.write_bytes(path.read_bytes() + b"")
    os.replace(rotated, path)
    again = cycle.run()
    assert again.resets == 1 and again.gaps["reset_rotated"] == 1
    assert again.evidence == []  # same events: ledger dedupes the re-read copy
    assert again.dropped["duplicate"] == 3


def test_missing_source_keeps_its_record_with_missing_status(tmp_path: Path) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    [before] = cycle.known.values()
    path.unlink()
    outcome = cycle.run()
    [after] = outcome.sources.values()
    assert after.status == "missing" and not after.complete
    assert after.state == before.state and after.source_id == before.source_id
    assert outcome.evidence == []


# --------------------------------------------------- dedupe / conflict / admit


def test_duplicate_copy_of_history_file_dedupes_through_ledger(tmp_path: Path) -> None:
    records = claude_records("s1", 3)
    write_records(claude_dir(tmp_path) / "s1.jsonl", records)
    write_records(claude_dir(tmp_path) / "copy-of-s1.jsonl", records)
    outcome = Cycle(tmp_path).run()
    native = [e for e in outcome.evidence if e.kind != "ownership_claim"]
    claims = [e for e in outcome.evidence if e.kind == "ownership_claim"]
    assert len(native) == len({e.native_key for e in native}) == 7
    assert {tuple(e.payload["target_native_key"]) for e in claims} == {e.native_key for e in native}
    assert outcome.dropped == {"duplicate": 7}
    assert len(outcome.sources) == 2 and all(r.complete for r in outcome.sources.values())


def test_conflicting_variant_is_counted_as_conflict_not_duplicate(tmp_path: Path) -> None:
    write_records(
        claude_dir(tmp_path) / "a.jsonl",
        [
            claude_prompt("s1", "hi"),
            claude_tool_use("s1", "t1", "Bash", {"command": "ls"}),
        ],
    )
    write_records(
        claude_dir(tmp_path) / "b.jsonl",
        [
            claude_prompt("s1", "hi"),
            claude_tool_use("s1", "t1", "Bash", {"command": "rm -rf /tmp/other"}),
        ],
    )
    cycle = Cycle(tmp_path)
    outcome = cycle.run()
    assert outcome.dropped == {"duplicate": 1, "conflict": 1}
    native = [e for e in outcome.evidence if e.kind != "ownership_claim"]
    assert len(native) == len({e.native_key for e in native}) == 2
    assert {e.kind for e in native} == {"user_turn", "tool_call"}
    assert cycle.ledger.conflict_count == 1


def test_admit_reasons_are_counted_and_those_records_not_retained(tmp_path: Path) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 3))
    cycle = Cycle(tmp_path)

    def admit(item: RawEvidence) -> str | None:
        return "before_horizon" if item.kind == "tool_result" else None

    outcome = cycle.run(admit=admit)
    assert outcome.dropped == {"before_horizon": 3}
    assert {e.kind for e in outcome.evidence} == {"user_turn", "tool_call"}
    # dropped records never reached the ledger: admitting them later still succeeds
    later = read_again_all(cycle)
    assert [e.kind for e in later] == ["tool_result"] * 3


def read_again_all(cycle: Cycle) -> list[RawEvidence]:
    cycle.known = {}
    outcome = cycle.run()
    return outcome.evidence


def test_usage_and_other_kinds_outside_consumed_set_are_never_retained(tmp_path: Path) -> None:
    usage = {"input_tokens": 5, "output_tokens": 7}
    write_records(
        claude_dir(tmp_path) / "s1.jsonl",
        [
            claude_prompt("s1", "hi"),
            claude_tool_use("s1", "t1", "Bash", {"command": "ls"}, usage=usage),
        ],
    )
    cycle = Cycle(tmp_path)
    outcome = cycle.run()
    assert outcome.dropped == {"kind_not_consumed": 1}
    assert {e.kind for e in outcome.evidence} == {"user_turn", "tool_call"}
    assert all(e.kind in CONSUMED_KINDS for e in outcome.evidence)
    assert list(cycle.ledger.final_usage()) == []  # usage never even reached the ledger


# --------------------------------------------------------------- unreadable


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions (chmod 000)")
def test_unreadable_file_marks_status_and_keeps_state(tmp_path: Path) -> None:
    good = claude_dir(tmp_path) / "good.jsonl"
    bad = claude_dir(tmp_path) / "bad.jsonl"
    write_records(good, claude_records("g", 1))
    write_records(bad, claude_records("b", 1))
    bad.chmod(0)
    try:
        cycle = Cycle(tmp_path)
        outcome = cycle.run()
    finally:
        bad.chmod(0o600)
    assert outcome.unreadable == 1
    by_name = {Path(r.source_id.split(":", 1)[1]).name: r for r in outcome.sources.values()}
    assert by_name["bad.jsonl"].status == "unreadable" and not by_name["bad.jsonl"].complete
    assert by_name["bad.jsonl"].offset == 0
    assert not by_name["bad.jsonl"].state.get("head_digest")
    assert not by_name["bad.jsonl"].state.get("tail_digest")
    assert by_name["good.jsonl"].status == "ok" and by_name["good.jsonl"].complete
    assert {e.session_id for e in outcome.evidence} == {"g"}
    # once readable, the very same record resumes from offset 0 and reads it all
    again = cycle.run()
    assert {e.session_id for e in again.evidence} == {"b"}
    assert again.sources[by_name["bad.jsonl"].key].status == "ok"


# ----------------------------------------------------------------- opencode


def oc_db(tmp_path: Path) -> Path:
    return tmp_path / "oc" / "opencode.db"


def oc_add_call(connection, n: int, at: int) -> None:
    opencode_add_message(connection, f"msg_{n}", "ses_1", opencode_assistant_data(cost=1), at=at)
    opencode_add_part(
        connection,
        f"prt_{n}",
        f"msg_{n}",
        "ses_1",
        opencode_tool_part(f"call_{n}", "bash", {"command": "ls"}, start=at + 1, end=at + 2),
        at=at,
    )


def db_bytes(path: Path) -> dict[str, str]:
    def digest(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "absent"

    return {"db": digest(path), "wal": digest(Path(f"{path}-wal"))}


def test_opencode_unvisited_baseline_reads_history_and_later_wal_once_without_writes(
    tmp_path: Path,
) -> None:
    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    oc_add_call(connection, 1, T0 + 100)
    oc_add_call(connection, 2, T0 + 200)
    cycle = Cycle(tmp_path)
    reader = cycle.readers["opencode"]
    [source] = reader.discover()

    before = db_bytes(db)
    record = baseline_record(reader, source)
    assert db_bytes(db) == before
    assert record.baseline == "skipped" and not record.complete and record.status == "ok"
    assert not record.state["extra"].get("wm")
    assert record.state["extra"]["opencode"]["status"] == "unvisited"
    assert record.state["extra"]["identity"]["inode"] == db.stat().st_ino
    assert record.size == db.stat().st_size and record.skipped_bytes == 0

    # A database baseline has not visited any row; it must not claim complete or
    # invent maxima. An unrestricted read admits history, unlike JSONL byte skipping.
    outcome = cycle.run(baseline_new=True)
    assert sorted(e.record_id for e in outcome.evidence if e.kind == "tool_call") == [
        "call:call_1",
        "call:call_2",
    ]
    assert all(r.complete for r in outcome.sources.values()) and db_bytes(db) == before
    # rows committed after the baseline (still only in the WAL) are read; older ones are not
    wal_before = Path(f"{db}-wal").stat().st_size
    oc_add_call(connection, 3, T0 + 300)
    assert Path(f"{db}-wal").stat().st_size > wal_before  # really in the WAL
    later = cycle.run(baseline_new=True)
    got = sorted(ids(later.evidence))
    assert got == [("opencode", "ses_1", "call:call_3"), ("opencode", "ses_1", "result:call_3")]
    assert later.bytes_read > 0
    [visited] = later.sources.values()
    assert visited.state["extra"]["wm"]["part"] == [T0 + 300, "prt_3"]
    assert all(r.complete for r in later.sources.values())
    # reading never writes the database or its WAL
    after_read = db_bytes(db)
    assert cycle.run().evidence == []
    assert db_bytes(db) == after_read
    connection.close()


def test_opencode_new_source_without_baseline_reads_everything(tmp_path: Path) -> None:
    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    oc_add_call(connection, 1, T0 + 100)
    cycle = Cycle(tmp_path)
    outcome = cycle.run()
    kinds = sorted(e.kind for e in outcome.evidence)
    assert kinds == ["session_header", "tool_call", "tool_result"]
    assert outcome.dropped["kind_not_consumed"] >= 1  # usage, including maintenance visits
    assert all(e.kind in CONSUMED_KINDS for e in outcome.evidence)
    assert all(r.complete for r in outcome.sources.values())
    connection.close()


def test_opencode_corrupt_database_is_unreadable_not_an_exception(tmp_path: Path) -> None:
    db = oc_db(tmp_path)
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not a sqlite database" * 50)
    cycle = Cycle(tmp_path)
    outcome = cycle.run()
    [record] = outcome.sources.values()
    assert record.status == "unreadable" and not record.complete
    assert outcome.unreadable == 1 and outcome.evidence == []
    reader = cycle.readers["opencode"]
    [source] = reader.discover()
    baseline = baseline_record(reader, source)
    assert not baseline.complete and baseline.state["extra"]["opencode"]["status"] == "unvisited"


def test_opencode_pages_resume_across_budgeted_calls_without_duplicates(tmp_path: Path) -> None:
    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    connection.execute("BEGIN")
    for n in range(1, 1101):  # more than the reader's 1000-row page
        oc_add_call(connection, n, T0 + 100 + n)
    connection.execute("COMMIT")
    whole = Cycle(tmp_path)
    whole.drain()
    chunked = Cycle(tmp_path)
    # One native row is the cooperative quantum, not the old 1000-row page.
    outcomes = chunked.drain(byte_budget=1, limit=1100 * 8 + 20)
    assert len(outcomes) >= 2 and outcomes[0].stopped_by == "byte_budget"
    assert outcomes[0].bytes_read > 0 and outcomes[0].evidence
    [first_record] = outcomes[0].sources.values()
    assert sum(first_record.state["extra"]["opencode"]["rows_visited"].values()) == 1
    assert sorted(ids(chunked.evidence)) == sorted(ids(whole.evidence))
    assert len(chunked.evidence) == len(set(ids(chunked.evidence))) == 1 + 2 * 1100
    assert all(r.complete and not r.more for r in chunked.known.values())
    [record] = chunked.known.values()
    assert record.state["extra"]["wm"] == next(iter(whole.known.values())).state["extra"]["wm"]
    assert record.state["extra"]["wm"] == {
        "session": [T0, "ses_1"],
        "message": [T0 + 1200, "msg_1100"],
        "part": [T0 + 1200, "prt_1100"],
    }
    assert all(
        record.state["extra"]["opencode"]["rows_visited"][table] >= count
        for table, count in {"session": 1, "message": 1100, "part": 1100}.items()
    )
    connection.close()


def test_on_source_called_per_source_in_sorted_key_order_when_all_known(tmp_path: Path) -> None:
    for n in range(3):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    seen: list[str] = []
    cycle.run(on_source=lambda key, record: seen.append(key))
    assert seen == sorted(cycle.known) and len(seen) == 3


def test_backlogged_sources_are_read_before_finished_ones(tmp_path: Path) -> None:
    for n in range(3):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    keys = sorted(cycle.known)
    # grow the LAST key's file so it has unread data; it must be read first
    last = cycle.known[keys[-1]]
    path = Path(last.source_id.split(":", 1)[1])
    append_bytes(path, jsonl([claude_prompt("zz", "more", uuid="u-zz")]))
    seen: list[str] = []
    cycle.run(on_source=lambda key, record: seen.append(key))
    assert seen[0] == keys[-1] and seen[1:] == keys[:-1]


def test_same_size_rewrite_is_not_starved_by_an_always_growing_backlog(tmp_path: Path) -> None:
    for n in range(2):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    backlog, rewritten = [cycle.known[key] for key in sorted(cycle.known)]
    backlog_path = Path(backlog.source_id.split(":", 1)[1])
    rewritten_path = Path(rewritten.source_id.split(":", 1)[1])
    session = rewritten_path.stem
    replacement = "n" + session[1:]
    write_records(rewritten_path, claude_records(replacement, 1))
    assert rewritten_path.stat().st_size == rewritten.size
    for n in range(4):
        append_bytes(
            backlog_path,
            jsonl([claude_prompt("growing", "more", uuid=f"u-growing-{n}")]),
        )
        cycle.run(byte_budget=1)
        if cycle.known[rewritten.key].resets:
            break
    assert cycle.known[rewritten.key].resets == 1
    assert any(e.session_id == replacement for e in cycle.evidence)


# ---- admission rules -------------------------------------------------------------------------


def test_expired_conversation_and_retention_rejections_do_not_discard_needed_structure() -> None:
    rules = AdmitRules(
        expired_sessions=frozenset({("claude", "expired")}),
        horizon_day="2026-10-10",
        capture_start="2026-10-11T00:00:00Z",
    )
    old = build_evidence(ts_utc="2026-10-09T23:59:59Z")
    assert rules(old) == "beyond_retention"
    structural = replace(old, kind="session_header")
    assert rules(structural) is None
    assert rules(replace(structural, session_id="expired")) == "expired_conversation"
    assert rules(replace(structural, runtime="codex", session_id="expired")) is None
    assert rules(replace(old, ts_utc="2026-10-12T00:00:00Z")) is None


def test_backfill_retains_undated_structure_but_not_undated_prompt_text() -> None:
    rules = AdmitRules(window=("2026-10-11T00:00:00Z", "2026-10-12T00:00:00Z"))
    undated = build_evidence(ts_utc=None)
    assert rules(undated) == "outside_interval"
    assert rules(replace(undated, kind="lineage")) is None
    assert rules(replace(undated, ts_utc="2026-10-11T00:00:00Z")) is None
    assert rules(replace(undated, ts_utc="2026-10-12T00:00:00Z")) == "outside_interval"


@pytest.mark.parametrize("invalid", [True, "not-an-instant"])
def test_invalid_capture_timestamp_is_unknown_not_an_epoch_origin(invalid: object) -> None:
    assert collector_modes.parse_stamp(invalid) is None
    assert collector_modes.parse_stamp("2026-10-12T01:00:00+02:00") == datetime(
        2026, 10, 11, 23, tzinfo=UTC
    )


# ---- the real collector over damaged, vanishing and budgeted sources -----------------------------

NOW = scene.at(hour=20)
CLOCK = scene.at(hour=9)
DAY = 12
SMALL = 1_100  # about one transcript per cycle: every cycle makes progress, none finishes


def seeded(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, scene.at(hour=9))
    assert scene.report(scene.collect(NOW)).generation == 1
    assert scene.counter_total(scene.current()) == 2


def test_complete_malformed_source_line_advances_watermark_and_counts_gap_once(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    path = vs.projects(tmp_path) / "-work" / "cons0.jsonl"
    with path.open("ab") as handle:
        handle.write(b"not-json\n")
    scene.collect(NOW)
    checkpoint = scene.committed()
    record = next(row for row in checkpoint.sources.values() if row.source_id.endswith(str(path)))
    assert record.offset == path.stat().st_size and record.gaps["malformed_json"] == 1
    assert scene.coverage(scene.current())["gaps"]["malformed_json"] == 1
    assert scene.counter_total(scene.current()) == 2
    scene.collect(NOW)
    again = scene.committed().sources[record.key]
    assert again.offset == record.offset and again.gaps == record.gaps


def test_disappearing_source_during_prebaseline_does_not_hide_other_sources(tmp_path: Path) -> None:
    vs.enable()
    vs.filler(vs.projects(tmp_path), "gone", scene.at(hour=9), 1)
    vs.filler(vs.projects(tmp_path), "present", scene.at(hour=9), 1)
    discovered = collector_read.discover(collector_read.make_readers())
    vanished = vs.projects(tmp_path) / "-work" / "gone.jsonl"
    vanished.unlink()
    known = {}
    assert collector_modes.pre_baseline(discovered, known, NOW) == 1
    assert len(known) == 1
    record = next(iter(known.values()))
    assert record.source_id.endswith("present.jsonl") and record.baseline == "skipped"
    assert record.offset > 0


def test_inaccessible_index_is_reported_unknown_without_losing_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    index = storage.data_root() / "lancedb"
    index.mkdir(mode=0o700)
    original = Path.stat

    def denied(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == index:
            raise PermissionError("index stat denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", denied)
        scene.collect(NOW)
        assert scene.coverage(scene.current())["index"] == "unknown"
        assert scene.counter_total(scene.current()) == 2
    scene.collect(NOW)
    assert scene.coverage(scene.current())["index"] == "available"


def test_partial_backfill_resumes_its_own_offsets_after_restart(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture(day=12)
    vs.pair(vs.projects(tmp_path), 0, scene.at(day=11, hour=9))
    scene.collect(NOW)
    live_offsets = {key: row.offset for key, row in scene.committed().sources.items()}
    before = scene.counter_total(scene.current())
    options = CollectOptions(
        backfill=True,
        since="2026-10-11T00:00:00Z",
        until="2026-10-13T00:00:00Z",
        budget=CycleBudget(bytes=1),
    )
    first = scene.collect(NOW, options)
    assert first.state == "partial"
    job = scene.committed().backfill_job
    assert job["status"] == "running"
    assert any(row["state"]["offset"] > 0 for row in job["sources"].values())
    for _ in range(30):
        result = scene.collect(NOW, options)
        assert scene.counter_total(scene.current()) == before
        assert {key: row.offset for key, row in scene.committed().sources.items()} == live_offsets
        if result.state == "published":
            break
    else:
        pytest.fail("backfill did not finish its bounded resumable scan")
    checkpoint = scene.committed()
    assert not checkpoint.backfill_job
    assert len(checkpoint.backfills) == 1
    assert (
        sum(
            count
            for key, count in checkpoint.backfills[0]["counts"].items()
            if key.startswith("search:")
        )
        == 1
    )
    identities = vs.evidence_identities()
    assert identities and len(identities) == len(set(identities))
    scene.collect(NOW, CollectOptions(backfill=True, since=options.since, until=options.until))
    assert scene.committed().backfills == checkpoint.backfills
    assert vs.evidence_identities() == identities


# ---- cooperative cycle budgets: a backlog resumes across cycles and converges to single-shot ----
# Convergence is judged on retained evidence identities, the canonical ledger digest, and the
# live counters of an unbudgeted single-shot run of the same scene.


def backlog_scene(root: Path, sessions: int = 5) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in range(sessions):
        vs.pair(vs.projects(root), n, CLOCK)


def cycles_until_done(budget: CycleBudget, limit: int = 300) -> list:
    results = []
    for _ in range(limit):
        result = scene.collect(NOW, budget=budget)
        results.append(result)
        if scene.report(result).stopped_by == "complete":
            return results
    raise AssertionError("never completed")


def test_a_small_budget_backlog_resumes_to_exactly_the_single_shot_ledger(tmp_path: Path) -> None:
    backlog_scene(tmp_path)
    single = scene.collect(NOW)
    assert scene.report(single).complete and scene.report(single).stopped_by == "complete"
    reference = vs.facts()
    vs.wipe(tmp_path)

    backlog_scene(tmp_path)
    results = cycles_until_done(CycleBudget(bytes=SMALL))
    reports = [scene.report(r) for r in results]
    assert len(reports) >= 4  # the backlog genuinely needed several cycles
    assert [r.stopped_by for r in reports[:-1]] == ["byte_budget"] * (len(reports) - 1)
    assert all(r.status == "published" and not r.complete for r in reports[:-1])
    assert [r.generation for r in reports] == list(range(1, len(reports) + 1))
    assert reports[-1].complete and reports[-1].lag_bytes == 0

    final = vs.facts()
    assert final.evidence == reference.evidence  # no duplicate and no missing record
    assert len(set(final.evidence)) == len(final.evidence)
    assert final.ledger == reference.ledger  # final ledger == single-shot
    assert final.counters == reference.counters
    assert final.snapshot_id == reference.snapshot_id  # even the snapshot identity agrees
    # the budget bound each cycle (one bounded record may overshoot, never unbounded further work)
    largest = max(
        len(json.dumps(r).encode()) + 1
        for path in vs.projects(tmp_path).rglob("*.jsonl")
        for r in map(json.loads, path.read_text().splitlines())
    )
    assert all(r.bytes_read <= SMALL + largest for r in reports)
    assert sum(r.bytes_read for r in reports) >= scene.report(single).bytes_read  # all consumed


def test_every_partial_generation_states_its_prefix_lag_and_never_claims_completeness(
    tmp_path: Path,
) -> None:
    backlog_scene(tmp_path)
    total = sum(p.stat().st_size for p in vs.projects(tmp_path).rglob("*.jsonl"))
    lags: list[int] = []
    for _ in range(300):
        result = scene.collect(NOW, budget=CycleBudget(bytes=SMALL))
        envelope = scene.current()
        coverage = scene.coverage(envelope)
        watermarks = coverage["watermarks"]
        report = scene.report(result)
        assert coverage["lag_bytes"] == report.lag_bytes == sum(w["lag_bytes"] for w in watermarks)
        assert all(0 <= w["offset"] <= w["size"] for w in watermarks)
        assert coverage["stopped_by"] == report.stopped_by
        if report.stopped_by == "complete":
            assert coverage["complete"] is True and coverage["lag_bytes"] == 0
            break
        assert coverage["complete"] is False  # a lagging generation is never complete
        # A budget can stop before rechecking an already-consumed source. Zero known
        # byte lag does not prove that this discovery/read pass completed.
        assert any("collection_incomplete" in w for w in envelope.warnings)
        assert all(
            w["partial"] for w in [scene.window(envelope, name)["coverage"] for name in ("24h",)]
        )
        assert result.state == "partial"
        lags.append(report.lag_bytes)
    assert lags == sorted(lags, reverse=True) and lags[0] <= total  # lag only ever shrinks
    assert lags[-1] < lags[0]
    assert scene.coverage(scene.current())["complete"] is True


def test_an_incomplete_tail_never_completes_until_its_newline_arrives(tmp_path: Path) -> None:
    backlog_scene(tmp_path, sessions=2)
    target = vs.projects(tmp_path) / "-work" / "cons1.jsonl"
    whole = target.read_bytes()
    last = whole.rstrip(b"\n").rfind(b"\n") + 1
    torn = whole[: last + 40]  # half a record, no newline
    target.write_bytes(torn)
    before = scene.collect(NOW)
    assert scene.report(before).complete is False  # a torn tail is not a complete source
    envelope = scene.current()
    watermark = next(
        w for w in scene.coverage(envelope)["watermarks"] if w["pending_tail_bytes"] > 0
    )
    assert watermark["offset"] + watermark["pending_tail_bytes"] == watermark["size"]
    assert watermark["offset"] == last  # the watermark stops at the last newline
    assert (
        scene.coverage(envelope)["complete"] is False
        and scene.coverage(envelope)["sources"]["complete"]
        < (scene.coverage(envelope)["sources"]["total"])
    )
    for _ in range(3):  # repeating changes nothing: still pending, still not complete
        again = scene.collect(NOW)
        assert not scene.report(again).complete and scene.report(again).new_evidence == 0
    held = {i for i in vs.evidence_identities() if i[1] == "cons1"}
    names = {record_id for _, _, record_id, _ in held}
    assert "call:t1b" in names and "result:t1b" not in names  # only the torn last line is absent

    target.write_bytes(whole)  # the writer finishes the line
    after = scene.collect(NOW)
    assert scene.report(after).complete and scene.report(after).new_evidence > 0
    done = vs.evidence_identities()
    assert len(set(done)) == len(done)
    # the completed tail now yields exactly what an unbudgeted run of the finished file yields
    reference_root = tmp_path / "ref"
    with vs.isolated(vs.env_for(reference_root)):
        backlog_scene(reference_root, sessions=2)
        scene.collect(NOW)
        assert vs.evidence_identities() == done


def test_a_record_larger_than_the_remaining_budget_completes_then_stops(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    big = "x" * 40_000
    records = [
        claude_prompt("big", big, uuid="u-big", ts=ts(12, 9)),
        claude_prompt("big", "second", uuid="u-two", ts=ts(12, 9, 1)),
        claude_prompt("big", "third", uuid="u-three", ts=ts(12, 9, 2)),
    ]
    vs.put(vs.projects(tmp_path), "big", records, CLOCK)
    first = scene.collect(NOW, budget=CycleBudget(bytes=1_000))
    report = scene.report(first)
    assert report.stopped_by == "byte_budget" and report.new_evidence >= 1
    assert report.bytes_read >= 40_000  # the started record finished (cooperative, not hard)
    assert report.lag_bytes > 0  # but nothing further was started: later records still unread
    rest = cycles_until_done(CycleBudget(bytes=1_000))
    assert scene.report(rest[-1]).complete
    identities = [i[2] for i in vs.evidence_identities()]
    assert len(identities) == len(set(identities)) == 3


def test_time_budget_stops_between_sources_and_the_next_cycle_finishes_the_rest(
    tmp_path: Path,
) -> None:
    backlog_scene(tmp_path)
    ticks = iter(range(0, 10_000, 11))  # every clock read advances 11 s: the 10 s target passes
    first = scene.collect(NOW, clock=lambda: float(next(ticks)))
    report = scene.report(first)
    assert report.stopped_by == "time_budget" and not report.complete
    assert report.lag_bytes > 0 and first.state == "partial"
    rest = cycles_until_done(CycleBudget())
    assert scene.report(rest[-1]).complete
    identities = vs.evidence_identities()
    assert len(identities) == len(set(identities))
    reference = tmp_path / "ref"
    with vs.isolated(vs.env_for(reference)):
        backlog_scene(reference)
        scene.collect(NOW)
        expected = vs.facts()
    assert expected.evidence  # the unbudgeted reference really read everything
    assert tuple(identities) == expected.evidence


def test_a_real_backlog_over_32_mib_resumes_under_the_default_budget(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    pad = "p" * 900
    records = [
        claude_prompt("huge", f"{pad}{i}", uuid=f"u{i}", ts=ts(12, 9, i // 3600 % 60, i % 60))
        for i in range(40_000)
    ]
    target = vs.put(vs.projects(tmp_path), "huge", records, CLOCK)
    assert target.stat().st_size > 32 * MIB
    first = scene.collect(NOW)  # default 32 MiB budget
    one = scene.report(first)
    assert one.stopped_by == "byte_budget" and not one.complete and one.lag_bytes > 0
    assert 30 * MIB <= one.bytes_read <= 33 * MIB  # bounded by the default, not by the file
    seen = len(vs.evidence_identities())
    second = scene.collect(NOW)
    two = scene.report(second)
    assert two.complete and two.lag_bytes == 0
    identities = [i[2] for i in vs.evidence_identities()]
    assert len(identities) == len(set(identities)) == 40_000 and seen < 40_000  # all of it, once
    assert one.bytes_read + two.bytes_read >= target.stat().st_size


# ---- conflicting copies of one native event are counted once per dropped record, on any schedule


def variant(command: str) -> list[dict]:
    return [
        claude_prompt("s1", "hello", uuid="u1", ts="2026-10-12T09:00:00.000Z"),
        claude_tool_use(
            "s1", "t1", MCP, {"query": command}, ts="2026-10-12T09:00:05.000Z", message_id="m1"
        ),
    ]


def conflict_scene(root: Path) -> None:
    vs.enable()
    vs.start_capture()
    vs.put(vs.projects(root), "one", variant("alpha"), CLOCK)
    vs.put(vs.projects(root), "two", variant("beta"), CLOCK)
    vs.put(vs.projects(root), "three", variant("beta"), CLOCK)


def finish(budget: CycleBudget | None) -> tuple[dict, str, int]:
    cycles = 0
    while True:
        result = scene.collect(NOW, budget=budget)
        cycles += 1
        if scene.report(result).stopped_by == "complete":
            break
        assert cycles < 200
    envelope = scene.current()
    return scene.coverage(envelope)["gaps"], envelope.snapshot_id, cycles


@pytest.mark.parametrize("name", ["a", "eeeee", "g", "hh", "ffffff", "bb"])
def test_dropped_conflicts_do_not_depend_on_how_the_backlog_was_cut(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / name
    with vs.isolated(vs.env_for(root)):
        conflict_scene(root)
        one_gaps, one_id, one_cycles = finish(None)
        evidence = vs.evidence_identities()
    vs.wipe(root)
    with vs.isolated(vs.env_for(root)):
        conflict_scene(root)
        cut_gaps, cut_id, cut_cycles = finish(CycleBudget(bytes=1))  # one record per cycle
        assert vs.evidence_identities() == evidence  # same retained variant either way
    assert one_cycles == 1 and cut_cycles >= 3  # the schedules really differ
    assert cut_gaps == one_gaps
    assert cut_id == one_id
    dropped = one_gaps.get("dropped_conflict", 0)
    assert dropped >= 1  # the conflict is reported, never silently swallowed
    assert dropped + one_gaps.get("dropped_duplicate", 0) <= 2


# ---- collection control: validation first, recompute from retained evidence only -------------


def credit_scene(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)


def test_the_library_also_refuses_a_backfill_without_an_interval_before_any_io(
    tmp_path: Path,
) -> None:
    vs.enable()
    vs.start_capture()
    before = vs.tree(tmp_path / "data")
    for options in (
        CollectOptions(backfill=True),
        CollectOptions(backfill=True, since="2026-10-01T00:00:00Z"),
        CollectOptions(backfill=True, until="2026-10-02T00:00:00Z"),
    ):
        with pytest.raises(Exception, match="requires_interval|needs both"):
            scene.collect(scene.at(hour=10), options)
    after = vs.tree(tmp_path / "data")
    assert {k: v for k, v in after.items() if "worker/" not in k} == {
        k: v for k, v in before.items() if "worker/" not in k
    }
    assert vs.checkpoint() is None  # no checkpoint, no snapshot: no counters changed
    assert not (vs.accounting_dir() / "snapshots").exists()


def test_recompute_does_not_import_live_appends_and_preserves_canonical_measurements(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collector_state import load_state
    from ssgrep.accounting.contracts import canonical_bytes

    def measurements() -> dict:
        checkpoint = scene.committed()
        state = load_state(PROFILE, checkpoint, checkpoint.methodology)
        assert state.ledger is not None and state.ledger.debits

        def normalized(rows):
            return sorted(
                (
                    replace(
                        row,
                        provenance=tuple(
                            pair for pair in row.provenance if pair[0] != "frequency_revision"
                        ),
                    )
                    for row in rows
                ),
                key=canonical_bytes,
            )

        return {
            "methodology": state.ledger.methodology,
            "candidates": normalized(state.ledger.candidates),
            "debits": sorted(state.ledger.debits, key=canonical_bytes),
            "frozen": state.ledger.frozen,
        }

    credit_scene(tmp_path)
    scene.collect(scene.at(hour=10))
    base = vs.facts()
    before_measurements = measurements()
    vs.pair(vs.projects(tmp_path), 2, CLOCK)  # a NEW live record the recompute must not import
    result = scene.collect(
        scene.at(hour=11),
        CollectOptions(recompute=True, since="2026-10-12T00:00:00Z", until="2026-10-13T00:00:00Z"),
    )
    assert result.mode == "recompute" and result.unavailable_periods == ()  # nothing expired
    after = vs.facts()
    assert after.evidence == base.evidence  # the new transcript was not read
    # An explicit recompute changes only its named frequency revision and archives
    # old rows; every current quantity, footprint, identity and missingness is stable.
    assert measurements() == before_measurements
    assert scene.report(result).new_evidence == 0 and scene.report(result).bytes_read == 0
    follow = scene.collect(scene.at(hour=12))  # a normal collection imports it afterwards
    assert scene.report(follow).new_evidence > 0 and vs.facts().evidence != base.evidence


def test_missing_index_does_not_block_collection_and_is_reported_not_hidden(tmp_path: Path) -> None:
    credit_scene(tmp_path)
    assert not (tmp_path / "data" / "lancedb").exists()  # there is no search index at all
    result = scene.collect(scene.at(hour=10))
    assert result.state == "published" and scene.report(result).new_evidence > 0
    envelope = scene.current()
    assert scene.coverage(envelope)["index"] == "unavailable"  # said out loud ...
    assert (
        scene.window(envelope, "lifetime")["observations"]["searches"]["value"] == 3
    )  # ... usage counted
    assert not (tmp_path / "data" / "lancedb").exists()  # and collection never created one


def test_present_index_directory_is_never_opened_or_modified_by_collection(
    tmp_path: Path,
) -> None:
    credit_scene(tmp_path)
    index = tmp_path / "data" / "lancedb"
    index.mkdir(parents=True)
    (index / "table.lance").write_bytes(b"opaque index bytes")
    stamps = {p: p.stat().st_mtime_ns for p in index.rglob("*")}
    scene.collect(scene.at(hour=10))
    scene.collect(scene.at(hour=11), CollectOptions(recompute=True))
    assert scene.coverage(scene.current())["index"] == "available"
    assert {p: p.stat().st_mtime_ns for p in index.rglob("*")} == stamps
    assert (index / "table.lance").read_bytes() == b"opaque index bytes"


def test_a_corrupt_index_directory_does_not_stop_usage_collection(tmp_path: Path) -> None:
    credit_scene(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "lancedb").write_text("not a directory at all")  # unusable index
    result = scene.collect(scene.at(hour=10))
    assert result.state == "published" and scene.report(result).complete
    assert scene.coverage(scene.current())["index"] == "unavailable"
    assert scene.window(scene.current(), "lifetime")["observations"]["searches"]["value"] == 3


def test_old_database_drop_diagnostics_count_admission_observations_not_usage(tmp_path: Path):
    from ssgrep.accounting import collector_store

    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    oc_add_call(connection, 1, T0 + 100)
    oc_add_call(connection, 2, T0 + 200)
    native_env = env_for(tmp_path)
    assert datetime.fromtimestamp(T0 / 1000, UTC) < scene.at(day=1, hour=8)

    def drain(budget: CycleBudget):
        for _ in range(100):
            result = scene.collect(NOW, budget=budget, env=native_env)
            assert scene.report(result).status == "published"
            if scene.report(result).complete:
                return scene.current()
        raise AssertionError("old-record census did not converge")

    def part_visits() -> int:
        [record] = [row for row in scene.committed().sources.values() if row.runtime == "opencode"]
        visits = record.state["extra"]["opencode"]["rows_visited"]["part"]
        assert isinstance(visits, int) and not isinstance(visits, bool)
        return visits

    try:
        reference_env = {
            **vs.env_for(tmp_path),
            "SSGREP_DATA_DIR": str(tmp_path / "reference-data"),
        }
        with vs.isolated(reference_env):
            vs.enable()
            vs.start_capture(day=1)
            reference = drain(CycleBudget(bytes=BIG))
            reference_drops = scene.coverage(reference)["gaps"]["dropped_before_capture"]
            assert part_visits() >= 2
            # Each completed native part emits one call and one result. Fresh
            # and census visits are distinct admission observations, not usage.
            assert reference_drops == 2 * part_visits()

        vs.enable()
        vs.start_capture(day=1)
        resumed = drain(CycleBudget(bytes=1))  # Every cycle reloads persisted collector state.
        resumed_drops = scene.coverage(resumed)["gaps"]["dropped_before_capture"]
        assert resumed_drops == 2 * part_visits()
        assert resumed.observations == reference.observations
        assert resumed.windows == reference.windows
        assert resumed.estimates == reference.estimates
        assert resumed.source_manifest_digest == reference.source_manifest_digest
        if resumed_drops != reference_drops:
            assert resumed.snapshot_id != reference.snapshot_id  # Coverage is content, too.
        checkpoint = scene.committed()
        retained = list(collector_store.iter_evidence(PROFILE, checkpoint.segments))
        assert retained and all(item.kind == "session_header" for item in retained)
        idle = drain(CycleBudget(bytes=1))
        assert idle.snapshot_id == resumed.snapshot_id

        # A genuinely additional old tool remains excluded from imported usage,
        # while every actual fresh/census rejection remains visible diagnostically.
        oc_add_call(connection, 3, T0 + 300)
        appended = drain(CycleBudget(bytes=1))
        appended_drops = scene.coverage(appended)["gaps"]["dropped_before_capture"]
        assert appended_drops == 2 * part_visits()
        assert appended_drops >= resumed_drops + 2
        assert scene.counter_total(appended) == scene.counter_total(resumed)
    finally:
        connection.close()
