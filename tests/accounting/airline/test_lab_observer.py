"""Observer: hash-chained trace, real subprocess recording, shims, spools and native joins.

Every subprocess here is real; the only fakes are tiny ``/bin/sh`` stand-ins for ``ssgrep`` and
``psql``. Nothing needs docker, a network, or a live agent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from eval.accounting_airline import observer as obs
from eval.accounting_airline.observer import (
    GENESIS,
    NativeRecord,
    NativeSession,
    ObservedCall,
    Observer,
    ObserverError,
    Trace,
    install_shims,
    link_native,
    parse_native_omp,
    parse_psql_table,
    read_trace,
    result_record,
    sha,
    write_pgpass,
)
from eval.accounting_airline.standin_agent import LABEL, NativeWriter
from tests.accounting.readers_fixtures import (
    OMP_USAGE,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    pi_user,
    write_records,
)

REPO = Path(__file__).resolve().parents[3]
BASE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def _phases(path: Path, count: int) -> Trace:
    trace = Trace(path, "run-1")
    for index in range(count):
        trace.append({"kind": "phase", "name": f"p{index}"})
    return trace


def _observer(tmp_path: Path) -> Observer:
    return Observer(tmp_path / "verifier", "run-1")


def _run(observer: Observer, argv: list[str], *, timeout: float = 20, actor: str = "verifier"):
    return observer.run(
        actor=actor, task="T1", tool="t", argv=argv, env=BASE_ENV, cwd=observer.dir, timeout=timeout
    )


# ------------------------------------------------------------------------- hash chain


def test_chain_reads_back_in_order_and_links_prev_hashes(tmp_path: Path) -> None:
    trace = _phases(tmp_path / "t.ndjson", 5)
    records = read_trace(trace.path)
    assert [r["name"] for r in records] == [f"p{i}" for i in range(5)]
    assert [r["seq"] for r in records] == list(range(5))
    assert records[0]["prev"] == GENESIS
    assert all(b["prev"] == a["hash"] for a, b in zip(records, records[1:], strict=False))
    assert read_trace(tmp_path / "missing.ndjson") == []


def test_tampered_byte_is_detected_with_positive_control(tmp_path: Path) -> None:
    trace = _phases(tmp_path / "t.ndjson", 4)
    original = trace.path.read_bytes()
    assert len(read_trace(trace.path)) == 4  # untouched file is fine
    assert b'"p2"' in original
    trace.path.write_bytes(original.replace(b'"p2"', b'"q2"'))
    with pytest.raises(ObserverError, match="chain broken"):
        read_trace(trace.path)


def test_deleted_line_is_detected_in_middle_and_head(tmp_path: Path) -> None:
    trace = _phases(tmp_path / "t.ndjson", 4)
    lines = trace.path.read_bytes().splitlines(keepends=True)
    for drop in (0, 1, 2):
        trace.path.write_bytes(b"".join(ln for i, ln in enumerate(lines) if i != drop))
        with pytest.raises(ObserverError, match="chain broken"):
            read_trace(trace.path)
    trace.path.write_bytes(b"".join(lines))
    assert len(read_trace(trace.path)) == 4


def test_reordered_lines_are_detected(tmp_path: Path) -> None:
    trace = _phases(tmp_path / "t.ndjson", 4)
    lines = trace.path.read_bytes().splitlines(keepends=True)
    lines[1], lines[2] = lines[2], lines[1]
    trace.path.write_bytes(b"".join(lines))
    with pytest.raises(ObserverError, match="chain broken"):
        read_trace(trace.path)


def test_garbage_and_hashless_lines_are_refused(tmp_path: Path) -> None:
    trace = _phases(tmp_path / "t.ndjson", 2)
    with trace.path.open("ab") as handle:
        handle.write(b"not json\n")
    with pytest.raises(ObserverError, match="not a valid record"):
        read_trace(trace.path)
    trace.path.write_bytes(b'{"seq":0,"prev":"x"}\n')
    with pytest.raises(ObserverError, match="not a valid record"):
        read_trace(trace.path)


def test_result_record_refuses_an_unknown_state() -> None:
    with pytest.raises(ObserverError, match="unknown state"):
        result_record("c", "done", 0, "", "", None, lambda text: sha(text))


# ----------------------------------------------------------------------- Observer.run


def test_run_records_dispatch_and_result_with_ids_argv_and_output(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    call, proc = _run(observer, ["/bin/sh", "-c", "printf 'out\\n'; exit 3"])
    assert proc is not None and proc.returncode == 3
    assert call.state == "verified"  # the observer saw a completed process, even a failing one
    assert (call.exit_code, call.stdout, call.stdout_bytes) == (3, "out\n", 4)
    assert (call.actor, call.task, call.tool) == ("verifier", "T1", "t")
    assert call.argv == ("/bin/sh", "-c", "printf 'out\\n'; exit 3")
    assert call.call_id.startswith("verifier-")
    events = read_trace(observer.trace.path)
    assert [e["kind"] for e in events] == ["dispatch", "result"]
    dispatch, result = events
    assert dispatch["run_id"] == "run-1" and dispatch["state"] == "attempted"
    assert isinstance(dispatch["argv"], list) and result["call_id"] == dispatch["call_id"]
    assert result["stdout_digest"] == sha("out\n")


def test_run_never_interprets_arguments_as_a_shell_string(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    marker = tmp_path / "pwned"
    payload = f"hello; touch {marker}"
    call, _ = _run(observer, ["/bin/echo", payload])
    assert call.stdout == payload + "\n" and not marker.exists()
    assert call.argv == ("/bin/echo", payload)
    # A bare string is not a command line either: iterating it names a nonexistent program.
    bad, proc = _run(observer, "echo hi")  # ty: ignore[invalid-argument-type]
    assert proc is None and bad.state == "unknown"


def test_run_shown_argv_is_what_the_trace_records(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    call, _ = observer.run(
        actor="control", task=None, tool="ssgrep", argv=["/bin/echo", "x"],
        env=BASE_ENV, cwd=observer.dir, timeout=20, shown=["ssgrep", "search", "x"],
    )  # fmt: skip
    assert call.argv == ("ssgrep", "search", "x") and call.stdout == "x\n"


def test_timeout_is_interrupted_with_reason_and_no_exit_code(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    call, proc = _run(observer, ["/bin/sleep", "30"], timeout=0.3)
    assert proc is None and call.state == "interrupted" and call.exit_code is None
    result = read_trace(observer.trace.path)[-1]
    assert result["kind"] == "result" and result["reason"] == "timeout"


def test_killed_process_is_interrupted_not_verified(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    call, proc = _run(observer, ["/bin/sh", "-c", "kill -9 $$"])
    assert proc is not None and proc.returncode < 0
    assert call.state == "interrupted"


def test_missing_binary_is_unknown_with_the_os_error_named(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    call, proc = _run(observer, [str(tmp_path / "no-such-binary")])
    assert proc is None and call.state == "unknown" and call.exit_code is None
    assert read_trace(observer.trace.path)[-1]["reason"] == "FileNotFoundError"


def test_unknown_actor_is_refused_before_anything_is_recorded(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    with pytest.raises(ObserverError, match="unknown actor"):
        _run(observer, ["/bin/echo"], actor="solver-c")
    assert read_trace(observer.trace.path) == []


def test_dispatch_without_result_is_attempted_then_finalized_unknown(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    observer.trace.append(obs.dispatch_record("solver-b-orphan", "solver-b", None, "ssgrep",
                                              ["ssgrep", "search", "q"], "/w"))  # fmt: skip
    (call,) = observer.freeze().calls
    assert call.state == "attempted" and call.exit_code is None  # never inferred done
    assert observer.finalize() == 1
    (call,) = observer.freeze().calls
    assert call.state == "unknown" and call.exit_code is None
    assert read_trace(observer.trace.path)[-1]["reason"] == "no_completion_evidence"
    assert observer.finalize() == 0  # nothing left pending; no duplicate results


# ---------------------------------------------------------------------------- shims


def _fake(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


class Rig:
    def __init__(self, tmp_path: Path, ssgrep_body: str = 'echo "fake ssgrep $*"\n') -> None:
        self.shims = install_shims(tmp_path / "bin", python=sys.executable, repo_root=REPO)
        self.spool = tmp_path / "spool" / "solver-b.ndjson"
        self.spool.parent.mkdir()
        self.task_file = tmp_path / "task"
        self.task_file.write_text("L2\n")
        self.psql = _fake(
            tmp_path / "psql",
            'printf "args:%s\\n" "$*" >&2\nsql=$(cat)\ncase "$sql" in\n'
            '*FAIL*) echo "ERROR:  23505: duplicate key" >&2; exit 3;;\nesac\n'
            'printf "a\\tb\\n1\\t2\\n"\n',
        )
        pgpass = tmp_path / ".pgpass"
        write_pgpass(pgpass, host="h", port=5432, passwords={"writer": "pw"})
        self.pgpass = pgpass
        self.env = {
            **BASE_ENV,
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            obs.ENV_SPOOL: str(self.spool),
            obs.ENV_ACTOR: "solver-b",
            obs.ENV_RUN: "run-1",
            obs.ENV_TASK_FILE: str(self.task_file),
            obs.ENV_REAL + "SSGREP": str(_fake(tmp_path / "real-ssgrep", ssgrep_body)),
            obs.ENV_REAL + "PSQL": str(self.psql),
            obs.ENV_SQL_HOST: "h",
            obs.ENV_SQL_PORT: "5432",
            obs.ENV_SQL_PGPASS: str(self.pgpass),
        }

    def run(self, script: str, stdin: str = "", **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/sh", "-c", script], input=stdin, capture_output=True, text=True,
            env={**self.env, **env}, timeout=120, check=False,
        )  # fmt: skip

    def events(self) -> list[dict]:
        return read_trace(self.spool)


def test_install_shims_writes_executable_wrappers_and_pgpass_is_private(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert set(rig.shims) == {"ssgrep", "sql"}
    for tool, path in rig.shims.items():
        assert os.access(path, os.X_OK) and f"observer shim {tool}" in path.read_text()
    assert rig.pgpass.stat().st_mode & 0o777 == 0o600
    assert rig.pgpass.read_text() == "h:5432:*:lab_writer:pw\n"


def test_ssgrep_shim_in_a_loop_spools_two_dispatches_with_one_native_call(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    done = rig.run("for q in alpha beta; do ssgrep search $q; done", **{obs.ENV_NATIVE_CALL: "c-1"})
    assert done.returncode == 0
    assert done.stdout == "fake ssgrep search alpha\nfake ssgrep search beta\n"
    events = rig.events()
    dispatches = [e for e in events if e["kind"] == "dispatch"]
    results = [e for e in events if e["kind"] == "result"]
    assert [d["argv"] for d in dispatches] == [
        ["ssgrep", "search", "alpha"],
        ["ssgrep", "search", "beta"],
    ]
    assert {d["native_call_id"] for d in dispatches} == {"c-1"}
    assert len({d["call_id"] for d in dispatches}) == 2
    assert all(d["actor"] == "solver-b" and d["task"] == "L2" for d in dispatches)
    assert [(r["state"], r["exit_code"]) for r in results] == [("verified", 0)] * 2


def test_shim_passes_through_streams_and_exit_code(tmp_path: Path) -> None:
    rig = Rig(tmp_path, "echo out; echo err >&2; exit 7\n")
    done = rig.run("ssgrep show r1")
    assert (done.returncode, done.stdout, done.stderr) == (7, "out\n", "err\n")
    (result,) = (e for e in rig.events() if e["kind"] == "result")
    assert result["state"] == "verified" and result["exit_code"] == 7


def test_shim_child_environment_drops_lab_variables(tmp_path: Path) -> None:
    rig = Rig(tmp_path, "/usr/bin/env\n")
    done = rig.run("ssgrep search q", LAB_SECRET="leak")
    assert done.returncode == 0
    names = {line.split("=", 1)[0] for line in done.stdout.splitlines() if "=" in line}
    assert "PATH" in names and not {n for n in names if n.startswith("LAB_")}
    assert "leak" not in done.stdout


def test_shim_with_missing_real_binary_records_unknown_and_exits_127(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    done = rig.run("ssgrep search q", **{obs.ENV_REAL + "SSGREP": str(tmp_path / "absent")})
    assert done.returncode == 127 and "cannot run" in done.stderr
    (result,) = (e for e in rig.events() if e["kind"] == "result")
    assert result["state"] == "unknown" and result["reason"] == "FileNotFoundError"


def test_sql_shim_records_statement_role_database_and_committed_outcome(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    statement = "BEGIN; UPDATE t SET x = 1; COMMIT;"
    done = rig.run("sql --role writer --database airline", stdin=statement)
    assert done.returncode == 0 and done.stdout == "a\tb\n1\t2\n"
    assert "-U lab_writer" in done.stderr and "-d airline" in done.stderr
    start, finish = (e for e in rig.events() if e["kind"] == "sql")
    assert (start["role"], start["database"]) == ("writer", "airline")
    assert start["statement_digest"] == sha(statement) and start["statement_chars"] == len(
        statement
    )
    assert (obs._spool_blob_dir(rig.spool) / start["statement_blob"]).read_text() == statement
    assert (finish["txn"], finish["sqlstate"], finish["exit_code"]) == ("committed", None, 0)
    assert parse_psql_table(done.stdout) == [{"a": "1", "b": "2"}]


@pytest.mark.parametrize(
    ("statement", "txn", "sqlstate", "code"),
    [
        ("SELECT 1", "autocommit", None, 0),
        ("BEGIN; DELETE FROM t; ROLLBACK;", "rolled_back", None, 0),
        ("BEGIN; INSERT FAIL; COMMIT;", "rolled_back", "23505", 3),
        ("INSERT FAIL", "error", "23505", 3),
    ],
)
def test_sql_shim_txn_outcome_matrix(
    tmp_path: Path, statement: str, txn: str, sqlstate: str | None, code: int
) -> None:
    rig = Rig(tmp_path)
    done = rig.run("sql --role writer --database airline", stdin=statement)
    assert done.returncode == code
    finish = [e for e in rig.events() if e["kind"] == "sql"][-1]
    assert (finish["txn"], finish["sqlstate"]) == (txn, sqlstate)


def test_sql_outcome_unit_cases() -> None:
    assert obs._sql_outcome("begin; select 1;", 0, "") == ("unknown", None)  # no COMMIT seen
    assert obs._sql_outcome("select 1", 0, "") == ("autocommit", None)
    assert obs._sql_outcome("x", 1, "ERROR:  42P01: no table") == ("error", "42P01")


def test_sql_shim_refuses_unknown_role_without_running_psql(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    done = rig.run("sql --role root --database airline", stdin="SELECT 1")
    assert done.returncode != 0 and "usage: sql" in done.stderr
    done = rig.run("sql --role reader --database 'a;b'", stdin="SELECT 1")
    assert done.returncode != 0 and "usage: sql" in done.stderr
    assert not any(e["kind"] == "sql" for e in rig.events()) if rig.spool.exists() else True


# ----------------------------------------------------------- spool ingest + separation


def test_ingest_spool_is_idempotent_and_verified_first(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.run("ssgrep search q").returncode == 0
    assert rig.run("sql --role reader --database airline", stdin="SELECT 1").returncode == 0
    observer = _observer(tmp_path)
    blobs = obs._spool_blob_dir(rig.spool)
    # Mutation proof: a tampered spool is refused before ANY record is copied.
    original = rig.spool.read_bytes()
    rig.spool.write_bytes(original.replace(b'"ssgrep"', b'"ssgrex"', 1))
    with pytest.raises(ObserverError):
        observer.ingest_spool(rig.spool, blobs, expected_actor="solver-b", expected_task="L2")
    assert read_trace(observer.trace.path) == []
    rig.spool.write_bytes(original)
    copied = observer.ingest_spool(rig.spool, blobs, expected_actor="solver-b", expected_task="L2")
    assert copied == len(read_trace(rig.spool)) > 0
    assert (
        observer.ingest_spool(rig.spool, blobs, expected_actor="solver-b", expected_task="L2") == 0
    )
    assert len(read_trace(observer.trace.path)) == copied
    ingested = read_trace(observer.trace.path)
    assert all(e["via"] == "spool" for e in ingested)
    assert [e["spool_seq"] for e in ingested] == list(range(copied))
    # a later spool append is picked up exactly once
    assert rig.run("ssgrep show r").returncode == 0
    assert (
        observer.ingest_spool(rig.spool, blobs, expected_actor="solver-b", expected_task="L2") == 2
    )
    assert (
        observer.ingest_spool(rig.spool, blobs, expected_actor="solver-b", expected_task="L2") == 0
    )
    frozen = observer.freeze()
    assert [c.tool for c in frozen.calls] == ["ssgrep", "sql", "ssgrep"]
    assert frozen.calls[0].stdout == "fake ssgrep search q\n"  # blob copied across
    assert all(c.state == "unknown" for c in frozen.calls)  # no independent native evidence
    (sql_call,) = (c for c in frozen.calls if c.tool == "sql")
    assert [e["phase"] for e in sql_call.sql] == ["start", "finish"]


def test_verifier_lookups_never_appear_among_solver_b_calls(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.run("ssgrep search solver-q").returncode == 0
    observer = _observer(tmp_path)
    _run(observer, ["/bin/echo", "verifier-q"], actor="verifier")
    _run(observer, ["/bin/echo", "control-q"], actor="control")
    observer.ingest_spool(
        rig.spool, obs._spool_blob_dir(rig.spool), expected_actor="solver-b", expected_task="L2"
    )
    frozen = observer.freeze()
    solver_b = [c for c in frozen.calls if c.actor == "solver-b"]
    assert [c.argv for c in solver_b] == [("ssgrep", "search", "solver-q")]
    assert {c.actor for c in frozen.calls} == {"solver-b", "verifier", "control"}
    assert all("verifier-q" not in c.argv for c in solver_b)
    assert [c.argv[-1] for c in frozen.calls if c.actor == "verifier"] == ["verifier-q"]


# --------------------------------------------------------------------- parse_native_omp


def _standin_session(tmp_path: Path) -> NativeWriter:
    writer = NativeWriter(tmp_path / "sessions", "sess-1", "/w")
    writer.user("do the thing")
    writer.assistant(calls=[("c1", "bash", {"command": "echo hi"})], scripted_input=12)
    writer.result("c1", "bash", "hi\n", False)
    writer.assistant(text="done", scripted_input=20)
    return writer


def test_parse_native_counts_usage_calls_and_label_from_a_writer_file(tmp_path: Path) -> None:
    writer = _standin_session(tmp_path)
    session = parse_native_omp(writer.path)
    assert (session.session_id, session.cwd, session.runtime_label) == ("sess-1", "/w", LABEL)
    assert session.model == "scripted-standin-v1" and session.malformed_lines == 0
    kinds = [r.kind for r in session.records]
    assert kinds == ["user_turn", "usage", "tool_call", "tool_result", "usage"]
    usage = [r for r in session.records if r.kind == "usage"]
    assert [u.usage["input"] for u in usage] == [12, 20] and all(
        u.cost_nano_usd is None for u in usage
    )
    call = next(r for r in session.records if r.kind == "tool_call")
    assert (call.call_id, call.name, call.text) == ("c1", "bash", "echo hi")
    result = next(r for r in session.records if r.kind == "tool_result")
    assert (result.call_id, result.text, result.is_error) == ("c1", "hi\n", False)
    assert session.partial_tail_bytes == 0 and session.complete_bytes == session.file_bytes


def test_parse_native_reads_costs_as_integer_nano_usd_from_fixture_records(tmp_path: Path) -> None:
    path = write_records(
        tmp_path / "s.jsonl",
        [
            pi_header("s9"),
            pi_user("e0", "go"),
            pi_tool_call("e1", "b1", "bash", {"command": "ls"}, usage=OMP_USAGE),
            pi_tool_result("e2", "b1", "boom", is_error=True),
        ],
    )
    session = parse_native_omp(path)
    usage = next(r for r in session.records if r.kind == "usage")
    assert usage.usage == {"input": 4, "output": 6, "total": 10}
    assert usage.cost_nano_usd == 500_000_000 and isinstance(usage.cost_nano_usd, int)
    assert next(r for r in session.records if r.kind == "tool_result").is_error is True
    assert session.runtime_label is None  # a non-stand-in header carries no lab label


def test_partial_tail_and_malformed_lines_are_counted_not_parsed(tmp_path: Path) -> None:
    path = write_records(tmp_path / "s.jsonl", [pi_header("s1"), pi_user("e0", "hello")])
    complete = path.read_bytes()
    tail = b'{"type":"message","id":"e1","mess'
    with path.open("ab") as handle:
        handle.write(b"{broken\n[1, 2]\n" + tail)
    session = parse_native_omp(path)
    prefix = complete + b"{broken\n[1, 2]\n"
    assert session.partial_tail_bytes == len(tail)
    assert session.complete_bytes == len(prefix) and session.file_bytes == len(prefix) + len(tail)
    assert session.file_digest == sha(prefix) != sha(path.read_bytes())
    assert session.malformed_lines == 2
    assert [r.kind for r in session.records] == ["user_turn"]
    # completing the tail line moves it into the digest and the records
    with path.open("ab") as handle:
        handle.write(b'ssage":{"role":"user","content":"x"}}\n')
    grown = parse_native_omp(path)
    assert grown.partial_tail_bytes == 0 and grown.file_digest != session.file_digest


def test_record_native_and_freeze_keep_only_the_latest_snapshot(tmp_path: Path) -> None:
    writer = _standin_session(tmp_path)
    observer = _observer(tmp_path)
    first = observer.record_native("solver-a", "L1", writer.path, source_root=tmp_path / "sessions")
    writer.assistant(calls=[("c2", "bash", {"command": "echo two"})])
    writer.result("c2", "bash", "two\n", False)
    second = observer.record_native(
        "solver-a", "L1", writer.path, source_root=tmp_path / "sessions"
    )
    assert len(second.records) == len(first.records) + 3  # usage + call + result
    events = read_trace(observer.trace.path)
    assert sum(e["kind"] == "native_session" for e in events) == 2  # both snapshots are kept
    frozen = observer.freeze()
    actor, task, session = frozen.sessions["sess-1"]
    assert (actor, task) == ("solver-a", "L1")
    assert len(session.records) == len(second.records)
    assert session.file_digest == second.file_digest and session.runtime_label == LABEL
    calls = [r for r in session.records if r.kind == "tool_call"]
    assert [(r.call_id, r.text) for r in calls] == [("c1", "echo hi"), ("c2", "echo two")]
    assert frozen.trace_head == events[-1]["hash"] and frozen.run_id == "run-1"
    # Mutation proof: editing the recorded trace makes freeze refuse the evidence.
    observer.trace.path.write_bytes(observer.trace.path.read_bytes().replace(b"bash", b"bssh", 1))
    with pytest.raises(ObserverError):
        observer.freeze()


# ------------------------------------------------------------------------- link_native


def _call(call_id: str, actor: str, wall_ms: int, native: str | None = None) -> ObservedCall:
    return ObservedCall(call_id, actor, None, "ssgrep", ("ssgrep",), "verified", 0, "", 0, 0, 1,
                        1, (), wall_ms * 1_000_000, native)  # fmt: skip


def _window(call_id: str, start: int, stop: int) -> list[NativeRecord]:
    return [
        NativeRecord(0, "tool_call", f"call:{call_id}", "x", call_id, "bash", start),
        NativeRecord(1, "tool_result", f"result:{call_id}", "y", call_id, "bash", stop),
    ]


def _session(*records: NativeRecord) -> NativeSession:
    return NativeSession("s", "/p", None, None, None, records, 0, 0, 0, "d")


def test_link_native_joins_inside_window_and_leaves_outside_unlinked() -> None:
    sessions = {"s": ("solver-b", None, _session(*_window("n1", 10_000, 10_400)))}
    inside = _call("a", "solver-b", 10_200)
    slack = _call("b", "solver-b", 10_400 + obs.LINK_SLACK_MS)
    outside = _call("c", "solver-b", 10_400 + obs.LINK_SLACK_MS + 1)
    other_actor = _call("d", "verifier", 10_200)
    linked = link_native([inside, slack, outside, other_actor], sessions)
    assert [c.native_call_id for c in linked] == ["n1", "n1", None, None]


def test_link_native_refuses_to_guess_between_overlapping_windows() -> None:
    records = [*_window("n1", 10_000, 10_400), *_window("n2", 10_100, 10_500)]
    sessions = {"s": ("solver-b", None, _session(*records))}
    linked = link_native([_call("a", "solver-b", 10_200), _call("b", "solver-b", 9_800)], sessions)
    assert [c.native_call_id for c in linked] == [None, "n1"]


def test_link_native_keeps_an_exact_env_link_over_the_window() -> None:
    sessions = {"s": ("solver-b", None, _session(*_window("n1", 10_000, 10_400)))}
    (call,) = link_native([_call("a", "solver-b", 10_200, native="exact")], sessions)
    assert call.native_call_id == "exact"


# ---------------------------------------------------------------------- parse_psql_table


def test_parse_psql_table_shapes() -> None:
    assert parse_psql_table("a\tb\n1\t2\n3\t4\n") == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]
    assert parse_psql_table("n\n7\n") == [{"n": "7"}]
    assert parse_psql_table("a\tb\n") == []  # header only: zero rows
    assert parse_psql_table("a\tb\n\n\t2\n") == [
        {"a": "", "b": "2"}
    ]  # blank line skipped, NULL cell
    assert parse_psql_table("") is None and parse_psql_table("\n\n") is None


def test_parse_psql_table_width_mismatch_is_none_never_a_guess() -> None:
    assert parse_psql_table("a\tb\n1\t2\nINSERT 0 1\n") is None  # command tag after a row
    assert parse_psql_table("a\tb\n1\t2\t3\n") is None
    assert parse_psql_table("a\tb\n1\n") is None
