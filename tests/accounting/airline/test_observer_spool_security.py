"""Solver-authored hash chains are not authority to read files or attest results."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eval.accounting_airline import observer as obs
from eval.accounting_airline.standin_agent import NativeWriter


def spool(tmp_path: Path, *, actor: str = "solver-b", task: str = "B/L3", run: str = "run"):
    observer = obs.Observer(tmp_path / "verifier", "run")
    trace = obs.Trace(tmp_path / "spool.ndjson", run)
    store = obs._spool_blob(trace.path)
    dispatch = obs.dispatch_record(
        "solver-b-call",
        actor,
        task,
        "ssgrep",
        ["ssgrep", "search", "q"],
        str(tmp_path),
        "native-call",
    )
    trace.append(dispatch)
    result = obs.result_record("solver-b-call", "verified", 0, "visible\n", "", 10, store)
    return observer, trace, result


def ingest(observer, trace, association=None):
    return observer.ingest_spool(
        trace.path,
        obs._spool_blob_dir(trace.path),
        expected_actor="solver-b",
        expected_task="B/L3",
        native_association=association,
    )


def native(observer, tmp_path, text="visible\n", actor="solver-b", task="B/L3"):
    writer = NativeWriter(tmp_path / "native", "session", str(tmp_path))
    writer.user("search q")
    writer.assistant(calls=[("native-call", "bash", {"command": "ssgrep search q"})])
    writer.result("native-call", "bash", text, False)
    return observer.record_native(actor, task, writer.path, source_root=tmp_path / "native")


@pytest.mark.parametrize("locator", ["absolute", "../canary", "nested/digest"])
def test_spool_rejects_arbitrary_blob_locators(tmp_path, locator):
    observer, trace, result = spool(tmp_path)
    canary = tmp_path / "canary"
    canary.write_text("planted secret")
    result["blob"] = str(canary) if locator == "absolute" else locator
    trace.append(result)
    with pytest.raises(obs.ObserverError):
        ingest(observer, trace)
    assert obs.read_trace(observer.trace.path) == []
    assert not (observer.blobs / obs.sha("planted secret")).exists()


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "digest", "oversize", "fifo"])
def test_spool_rejects_redirected_or_invalid_blob(tmp_path, attack):
    observer, trace, result = spool(tmp_path)
    target = obs._spool_blob_dir(trace.path) / result["blob"]
    canary = tmp_path / "canary"
    canary.write_text("visible\n")
    target.unlink()
    if attack == "symlink":
        target.symlink_to(canary)
    elif attack == "hardlink":
        os.link(canary, target)
    elif attack == "fifo":
        os.mkfifo(target)
    else:
        target.write_bytes(b"wrong" if attack == "digest" else b"x" * (obs.MAX_BLOB_BYTES + 1))
    trace.append(result)
    with pytest.raises(obs.ObserverError):
        ingest(observer, trace)
    assert obs.read_trace(observer.trace.path) == []


@pytest.mark.parametrize(
    "change", [{"actor": "verifier"}, {"actor": "solver-a"}, {"task": "A/L1"}, {"run": "other-run"}]
)
def test_spool_rejects_wrong_bindings(tmp_path, change):
    observer, trace, result = spool(tmp_path, **change)
    trace.append(result)
    with pytest.raises(obs.ObserverError):
        ingest(observer, trace)
    assert obs.read_trace(observer.trace.path) == []


@pytest.mark.parametrize("kind", ["native_session", "native_record", "native_snapshot", "phase"])
def test_spool_cannot_forge_native_or_verifier_records(tmp_path, kind):
    observer, trace, _ = spool(tmp_path)
    canary = tmp_path / "canary"
    canary.write_text("planted secret")
    trace.append({"kind": kind, "actor": "verifier", "path": str(canary), "blob": str(canary)})
    with pytest.raises(obs.ObserverError):
        ingest(observer, trace)
    assert obs.read_trace(observer.trace.path) == []


def test_valid_spool_requires_matching_native_result(tmp_path):
    observer, trace, result = spool(tmp_path)
    trace.append(result)
    association = native(observer, tmp_path)
    assert ingest(observer, trace, association) == 2
    assert ingest(observer, trace, association) == 0
    (call,) = observer.freeze().calls
    assert (call.actor, call.state, call.stdout, call.native_call_id) == (
        "solver-b",
        "verified",
        "visible\n",
        "native-call",
    )


@pytest.mark.parametrize("association_kind", ["missing", "wrong-result", "wrong-actor"])
def test_self_hashed_result_is_not_verified_without_native_evidence(tmp_path, association_kind):
    observer, trace, result = spool(tmp_path)
    trace.append(result)
    association = None
    if association_kind != "missing":
        association = native(
            observer,
            tmp_path,
            text="different\n",
            actor="solver-a" if association_kind == "wrong-actor" else "solver-b",
        )
    if association_kind == "wrong-actor":
        with pytest.raises(obs.ObserverError):
            ingest(observer, trace, association)
    else:
        ingest(observer, trace, association)
        (call,) = observer.freeze().calls
        assert call.state == "unknown" and call.exit_code is None


@pytest.mark.parametrize("kind", ["result", "native_record", "sql"])
def test_freeze_revalidates_all_blob_locators(tmp_path, kind):
    observer = obs.Observer(tmp_path / "verifier", "run")
    canary = tmp_path / "canary"
    canary.write_text("planted secret")
    observer.trace.append(obs.dispatch_record("c", "solver-b", "B/L3", "sql", ["sql"], "/w"))
    if kind == "result":
        record = obs.result_record("c", "verified", 0, "visible", "", 1, observer.blob)
        record["blob"] = str(canary)
    elif kind == "sql":
        record = {"kind": "sql", "call_id": "c", "statement_blob": str(canary)}
    else:
        native(observer, tmp_path)
        record = {"kind": "native_record", "session_id": "session", "blob": str(canary)}
    observer.trace.append(record)
    with pytest.raises(obs.ObserverError):
        observer.freeze()


@pytest.mark.parametrize("attack", ["outside", "symlink", "hardlink"])
def test_native_source_requires_owned_root_and_regular_unlinked_file(tmp_path, attack):
    observer = obs.Observer(tmp_path / "verifier", "run")
    root = tmp_path / "native"
    root.mkdir()
    canary = tmp_path / "canary.jsonl"
    canary.write_text('{"type":"session","id":"canary"}\n')
    path = canary
    if attack != "outside":
        path = root / "session.jsonl"
        if attack == "symlink":
            path.symlink_to(canary)
        else:
            os.link(canary, path)
    with pytest.raises(obs.ObserverError):
        observer.record_native("solver-b", "B/L3", path, source_root=root)
    assert obs.read_trace(observer.trace.path) == []


def test_real_shim_spool_native_linked_same_flow(tmp_path):
    from tests.accounting.airline.test_lab_observer import Rig

    rig = Rig(tmp_path)
    writer = NativeWriter(tmp_path / "native", "real-session", str(tmp_path))
    writer.user("Search q")
    writer.assistant(calls=[("real-call", "bash", {"command": "ssgrep search q"})])
    done = rig.run("ssgrep search q", **{obs.ENV_NATIVE_CALL: "real-call"})
    assert done.returncode == 0
    writer.result("real-call", "bash", done.stdout, False)
    observer = obs.Observer(tmp_path / "verifier", "run-1")
    association = observer.record_native(
        "solver-b", "L2", writer.path, source_root=tmp_path / "native"
    )
    observer.ingest_spool(
        rig.spool,
        obs._spool_blob_dir(rig.spool),
        expected_actor="solver-b",
        expected_task="L2",
        native_association=association,
    )
    (call,) = observer.freeze().calls
    assert (call.state, call.stdout, call.native_call_id) == ("verified", done.stdout, "real-call")


def test_previously_accepted_spool_prefix_cannot_be_replaced(tmp_path):
    observer, trace, result = spool(tmp_path)
    trace.append(result)
    ingest(observer, trace)
    trace.path.write_text("")
    trace.append(
        obs.dispatch_record(
            "other", "solver-b", "B/L3", "ssgrep", ["ssgrep", "search", "q"], str(tmp_path)
        )
    )
    with pytest.raises(obs.ObserverError, match="prefix changed"):
        ingest(observer, trace)


def test_shared_spool_preserves_two_registered_native_associations(tmp_path):
    observer = obs.Observer(tmp_path / "verifier", "run")
    trace = obs.Trace(tmp_path / "spool.ndjson", "run")
    store = obs._spool_blob(trace.path)
    associations: dict[str | None, obs.NativeSession] = {}
    for label in ("parent", "child"):
        writer = NativeWriter(tmp_path / "native", label, str(tmp_path))
        writer.user("Search")
        writer.assistant(calls=[(label, "bash", {"command": "ssgrep search q"})])
        writer.result(label, "bash", label + "\n", False)
        associations[label] = observer.record_native(
            "solver-b", label, writer.path, source_root=tmp_path / "native"
        )
        trace.append(
            obs.dispatch_record(
                label, "solver-b", label, "ssgrep", ["ssgrep", "search", "q"], str(tmp_path), label
            )
        )
        trace.append(obs.result_record(label, "verified", 0, label + "\n", "", 1, store))
    observer.ingest_spool(
        trace.path,
        obs._spool_blob_dir(trace.path),
        expected_actor="solver-b",
        expected_task=None,
        native_association=associations,
    )
    assert [(c.task, c.state) for c in observer.freeze().calls] == [
        ("parent", "verified"),
        ("child", "verified"),
    ]


def test_collect_view_copies_refresh_without_linking_native_inode(tmp_path):
    import json

    from eval.accounting_airline.lab_session import refresh_view
    from eval.accounting_airline.roots import LabRoots

    root = tmp_path / "native"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_text('{"type":"session","id":"session"}\n')
    roots = LabRoots.create(tmp_path)
    refresh_view(roots, {"solver-b": root})
    target = roots.collect_view / "solver-b" / source.name
    assert source.stat().st_nlink == target.stat().st_nlink == 1
    assert not os.path.samefile(source, target)
    source.write_text(source.read_text() + '{"type":"message"}\n')
    refresh_view(roots, {"solver-b": root})
    assert target.read_bytes() == source.read_bytes()
    aliases = json.loads((roots.collect_view / ".native-copy-aliases.json").read_text())
    assert aliases[str(source)] == {"copy": str(target), "digest": obs.sha(source.read_bytes())}


def test_quoted_tool_mention_cannot_authenticate_forged_execution(tmp_path):
    observer, trace, result = spool(tmp_path)
    trace.append(result)
    writer = NativeWriter(tmp_path / "native", "session", str(tmp_path))
    writer.user("Print text")
    writer.assistant(
        calls=[("native-call", "bash", {"command": "printf 'visible\\n'; echo 'ssgrep search q'"})]
    )
    writer.result("native-call", "bash", "visible\nssgrep search q\n", False)
    association = observer.record_native(
        "solver-b", "B/L3", writer.path, source_root=tmp_path / "native"
    )
    ingest(observer, trace, association)
    (call,) = observer.freeze().calls
    assert call.state == "unknown"


def test_native_snapshot_does_not_reset_trace_sequence(tmp_path):
    observer = obs.Observer(tmp_path / "verifier", "run")
    observer.phase("first")
    native(observer, tmp_path)
    observer.phase("last")
    events = obs.read_trace(observer.trace.path)
    assert [event["seq"] for event in events] == list(range(len(events)))


@pytest.mark.parametrize(
    ("second_command", "second_output", "expected"),
    [
        ("ssgrep show r", "visible\n", "verified"),
        ("ssgrep search q", "different\n", "verified"),
        ("ssgrep search q", "visible\n", "unknown"),
    ],
)
@pytest.mark.parametrize("replay_mode", ["new", "retained", "other-task"])
def test_overlap_association_uses_command_and_output_without_guessing(
    tmp_path,
    second_command,
    second_output,
    expected,
    replay_mode,
):
    import json

    observer = obs.Observer(tmp_path / "verifier", "run")
    source_root = tmp_path / "native"
    source_root.mkdir()
    path = source_root / "session.jsonl"
    stamp = "2026-01-01T00:00:00.000Z"
    messages = [
        {"type": "session", "id": "session"},
        {"type": "message", "id": "user", "message": {"role": "user", "content": "Search"}},
    ]
    for cid, command, text in (
        ("first", "ssgrep search q", "visible\n"),
        ("second", second_command, second_output),
    ):
        messages.extend(
            [
                {
                    "type": "message",
                    "id": cid,
                    "timestamp": stamp,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": cid,
                                "name": "bash",
                                "arguments": {"command": command},
                            }
                        ],
                    },
                },
                {
                    "type": "message",
                    "id": cid + "-result",
                    "timestamp": stamp,
                    "message": {
                        "role": "toolResult",
                        "toolCallId": cid,
                        "toolName": "bash",
                        "content": text,
                        "isError": False,
                    },
                },
            ]
        )
    path.write_text("".join(json.dumps(message) + "\n" for message in messages))
    association = observer.record_native("solver-b", "B/L3", path, source_root=source_root)
    start = next(record.ts_ms for record in association.records if record.kind == "tool_call")
    assert start is not None
    dispatch = obs.dispatch_record(
        "d", "solver-b", "B/L3", "ssgrep", ["ssgrep", "search", "q"], str(tmp_path)
    )
    dispatch["wall_ns"] = start * 1_000_000
    if replay_mode != "new":
        if replay_mode == "other-task":
            dispatch["task"] = "B/L4"
            expected = "unknown"
        # The older verifier retained the correct actor/task/session, but lost the
        # association because both native windows overlapped its coarse slack.
        observer.trace.append(
            {**dispatch, "via": "spool", "spool_seq": 0, "native_session_id": "session"}
        )
        result = obs.result_record("d", "verified", 0, "visible\n", "", 10, observer.blob)
        observer.trace.append({**result, "via": "spool", "spool_seq": 1})
    else:
        trace = obs.Trace(tmp_path / "spool.ndjson", "run")
        trace.append(dispatch)
        trace.append(
            obs.result_record("d", "verified", 0, "visible\n", "", 10, obs._spool_blob(trace.path))
        )
        ingest(observer, trace, association)
    before = observer.trace.path.read_bytes()
    (call,) = observer.freeze().calls
    assert call.state == expected
    assert call.native_call_id == ("first" if expected == "verified" else None)
    assert observer.trace.path.read_bytes() == before


def test_supervisor_attestation_rejects_solver_spool_and_changed_completion(tmp_path):
    observer, trace, result = spool(tmp_path)
    trace.append(result)
    ingest(observer, trace)
    with pytest.raises(obs.ObserverError):
        observer.attest_supervisor_spool(
            trace.path,
            obs._spool_blob_dir(trace.path),
            expected_actor="solver-b",
            expected_tasks={"B/L3"},
        )
    protected = observer.dir / "solver-b" / "spool.ndjson"
    protected.parent.mkdir()
    protected.write_bytes(trace.path.read_bytes())
    protected_blobs = obs._spool_blob_dir(protected)
    protected_blobs.mkdir()
    (protected_blobs / result["blob"]).write_text("visible\n")
    assert (
        observer.attest_supervisor_spool(
            protected,
            protected_blobs,
            expected_actor="solver-b",
            expected_tasks={"B/L3"},
        )
        == 1
    )
    assert (
        observer.attest_supervisor_spool(
            protected,
            protected_blobs,
            expected_actor="solver-b",
            expected_tasks={"B/L3"},
        )
        == 0
    )
    protected.write_text("")
    forged = obs.Trace(protected, "run")
    dispatch = obs.read_trace(trace.path)[0]
    forged.append({k: v for k, v in dispatch.items() if k not in ("seq", "prev", "hash")})
    forged.append(
        obs.result_record(
            "solver-b-call", "verified", 0, "forged\n", "", 10, obs._spool_blob(protected)
        )
    )
    with pytest.raises(obs.ObserverError):
        observer.attest_supervisor_spool(
            protected,
            protected_blobs,
            expected_actor="solver-b",
            expected_tasks={"B/L3"},
        )


def test_solver_cannot_append_supervisor_completion_receipt(tmp_path):
    observer, trace, result = spool(tmp_path)
    trace.append(result)
    trace.append(
        {
            "kind": "supervisor_completion",
            "call_id": "solver-b-call",
            "actor": "solver-b",
            "task": "B/L3",
            "state": "verified",
            "exit_code": 0,
            "stdout_digest": result["stdout_digest"],
        }
    )
    with pytest.raises(obs.ObserverError):
        ingest(observer, trace)


@pytest.mark.parametrize("projection", ["omp-truncation", "head-prefix"])
def test_attested_execution_with_omp_truncation_debits_only_rendered_text(tmp_path, projection):
    import json
    from dataclasses import replace

    from eval.accounting_airline.accounting_oracle import arithmetic_profile, evaluate
    from tests.accounting.airline.oracle_fixtures import TOKEN, source_a

    raw = (
        json.dumps(
            {
                "ok": True,
                "data": {
                    "results": [
                        {"ref": "omp:sa:ep:1", "response_text": "context " * 300 + TOKEN},
                    ]
                },
            }
        )
        + "\n"
    )
    visible = raw[:765] + (
        "…\n\n\nWall time: 0.12 seconds\n\n"
        "[Some lines truncated to 768 bytes. Use artifact://7:raw:1-1 to read line 1 whole]"
    )
    command = "ssgrep search q --json"
    if projection == "head-prefix":
        visible = raw[:120] + "\n\nWall time: 0.12 seconds"
        command += " | head -c 120"
    assert TOKEN not in visible
    observer = obs.Observer(tmp_path / "verifier", "run")
    writer = NativeWriter(tmp_path / "native", "consumer", str(tmp_path))
    writer.user("Look up a rule and act on it")
    writer.assistant(calls=[("lookup", "bash", {"command": command})])
    writer.result("lookup", "bash", visible, False)
    writer.assistant(calls=[("act", "bash", {"command": f"psql -c select '{TOKEN}'"})])
    writer.result("act", "bash", "ok\n", False)
    association = observer.record_native(
        "solver-b", "B/L3", writer.path, source_root=tmp_path / "native"
    )
    trace = obs.Trace(observer.dir / "solver-b" / "spool.ndjson", "run")
    dispatch = obs.dispatch_record(
        "broker-call",
        "solver-b",
        "B/L3",
        "ssgrep",
        ["ssgrep", "search", "q", "--json"],
        str(tmp_path),
        "lookup",
    )
    stamp = next(
        r.ts_ms for r in association.records if r.kind == "tool_call" and r.call_id == "lookup"
    )
    assert stamp is not None
    dispatch["wall_ns"] = stamp * 1_000_000
    trace.append(dispatch)
    trace.append(
        obs.result_record("broker-call", "verified", 0, raw, "", 10, obs._spool_blob(trace.path))
    )
    ingest(observer, trace, association)
    assert observer.freeze().calls[0].state == "unknown"  # a hash chain grants no authority
    original_native = writer.path.read_bytes()
    observer.attest_supervisor_spool(
        trace.path,
        obs._spool_blob_dir(trace.path),
        expected_actor="solver-b",
        expected_tasks={"B/L3"},
    )
    frozen = observer.freeze()
    (call,) = frozen.calls
    assert (call.state, call.native_call_id, call.stdout) == ("verified", "lookup", raw)
    source = source_a()
    frozen = replace(frozen, sessions={**frozen.sessions, source.sid: source.entry()})
    result = evaluate(frozen, arithmetic_profile([("identifier", TOKEN)]))
    assert result.counts["journaled_search"] == 1
    assert result.debits[0].chars == len(visible)
    assert result.estimates["central"].debit == len(visible) // 4
    assert result.estimates["central"].credit == 0
    assert result.verdicts == ()
    assert writer.path.read_bytes() == original_native


@pytest.mark.parametrize(
    "suffix",
    [
        "\n\n\nWall time: 0.12 seconds\n\n"
        "[Some lines truncated to 768 bytes. "
        "Use artifact:///planted-canary:raw:1-1 to read line 1 whole]",
        "\n\n\nWall time: 0.12 seconds\n\n"
        "[Some lines truncated to 4096 bytes. Use artifact://7:raw:1-1 to read line 1 whole]",
        "\n[arbitrary missing output]",
    ],
)
def test_unrecognized_partial_projection_cannot_claim_visible_link(suffix):
    from eval.accounting_airline.observer_blob import native_output_matches

    raw = "x" * 2000
    assert not native_output_matches(raw, raw[:765] + "…" + suffix, supervisor_completion=True)


@pytest.mark.parametrize("query", ["minimum connection", "airport rules"])
def test_literal_loop_argv_expansion_is_matched_without_execution(query):
    from eval.accounting_airline.observer_blob import native_invokes

    command = 'for q in "minimum connection" "airport rules"; do ssgrep search "$q" --json; done'
    assert native_invokes(command, ["ssgrep", "search", query, "--json"])
    assert not native_invokes(command, ["ssgrep", "search", "not in loop", "--json"])
    assert not native_invokes('ssgrep search "$q" --json', ["ssgrep", "search", query, "--json"])
    assert not native_invokes(
        'for q in "$(cat planted-canary)"; do ssgrep search "$q" --json; done',
        ["ssgrep", "search", query, "--json"],
    )


@pytest.mark.parametrize("matching_stderr", [True, False])
def test_stderr_only_failure_is_corroborated_by_native_error_output(tmp_path, matching_stderr):
    observer = obs.Observer(tmp_path / "verifier", "run")
    writer = NativeWriter(tmp_path / "native", "session", str(tmp_path))
    argv = ["sql", "--role", "reader", "--database", "airline"]
    writer.user("Execute the query")
    writer.assistant(calls=[("failed", "bash", {"command": " ".join(argv)})])
    stderr = "ERROR:  42P01: relation does not exist\n"
    rendered_stderr = stderr if matching_stderr else "different native failure\n"
    writer.result(
        "failed", "bash", rendered_stderr + "\n[exit code: 3]\n\nWall time: 0.12 seconds", True
    )
    association = observer.record_native(
        "solver-b", "B/L3", writer.path, source_root=tmp_path / "native"
    )
    trace = obs.Trace(tmp_path / "spool.ndjson", "run")
    trace.append(
        obs.dispatch_record("failed", "solver-b", "B/L3", "sql", argv, str(tmp_path), "failed")
    )
    trace.append(
        obs.result_record("failed", "verified", 3, "", stderr, 10, obs._spool_blob(trace.path))
    )
    ingest(observer, trace, association)
    (call,) = observer.freeze().calls
    expected = ("verified", 3, "", "failed") if matching_stderr else ("unknown", None, "", None)
    assert (call.state, call.exit_code, call.stdout, call.native_call_id) == expected
