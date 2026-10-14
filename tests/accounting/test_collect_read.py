"""9.x collector_read: discovery, baseline, bounded resumable reading (real files)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.collector_inputs import ClassifyingLedger
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
    evidence_line,
    source_key,
)
from ssgrep.accounting.contracts import RUNTIMES, RawEvidence
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

    def drain(self, **overrides: Any) -> list[ReadOutcome]:
        outcomes = []
        for _ in range(200):
            outcomes.append(self.run(**overrides))
            if outcomes[-1].stopped_by == "complete":
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
    # the final state of every source equals the unbudgeted one
    assert {k: r.state for k, r in chunked.known.items()} == {
        k: r.state for k, r in whole.known.items()
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
    assert len(outcome.evidence) == 7  # 1 turn + 3 calls + 3 results, once
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
    assert len(outcome.evidence) == 2
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
    assert by_name["bad.jsonl"].state == {}
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


def test_opencode_baseline_then_only_later_rows_including_wal_and_file_untouched(
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
    assert record.baseline == "skipped" and record.complete and record.status == "ok"
    assert record.state["extra"]["wm"] == {
        "session": [T0, "ses_1"],
        "message": [T0 + 200, "msg_2"],
        "part": [T0 + 200, "prt_2"],
    }
    assert record.state["extra"]["identity"]["inode"] == db.stat().st_ino
    assert record.size == db.stat().st_size and record.skipped_bytes == 0

    # pre-existing rows are skipped by the cycle that baselines the new source
    outcome = cycle.run(baseline_new=True)
    assert outcome.evidence == [] and db_bytes(db) == before
    # rows committed after the baseline (still only in the WAL) are read; older ones are not
    wal_before = Path(f"{db}-wal").stat().st_size
    oc_add_call(connection, 3, T0 + 300)
    assert Path(f"{db}-wal").stat().st_size > wal_before  # really in the WAL
    later = cycle.run(baseline_new=True)
    got = sorted(ids(later.evidence))
    assert got == [("opencode", "ses_1", "call:call_3"), ("opencode", "ses_1", "result:call_3")]
    assert later.bytes_read >= sum(len(evidence_line(e)) for e in later.evidence)
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
    assert outcome.dropped == {"kind_not_consumed": 1}  # the assistant usage record
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
    assert baseline_record(reader, source).status == "unreadable"


def test_opencode_pages_resume_across_budgeted_calls_without_duplicates(tmp_path: Path) -> None:
    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    connection.execute("BEGIN")
    for n in range(1, 1101):  # more than the reader's 1000-row page
        oc_add_call(connection, n, T0 + 100 + n)
    connection.execute("COMMIT")
    whole = Cycle(tmp_path).run()
    chunked = Cycle(tmp_path)
    outcomes = chunked.drain(byte_budget=1)
    assert len(outcomes) >= 2 and outcomes[0].stopped_by == "byte_budget"
    assert outcomes[0].bytes_read >= sum(len(evidence_line(e)) for e in outcomes[0].evidence) > 0
    assert sorted(ids(chunked.evidence)) == sorted(ids(whole.evidence))
    assert len(chunked.evidence) == len(set(ids(chunked.evidence))) == 1 + 2 * 1100
    assert all(r.complete and not r.more for r in chunked.known.values())
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
