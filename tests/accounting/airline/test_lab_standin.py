"""Scripted stand-in agent: really executes shell steps, writes REAL omp-format JSONL, and is
always labelled so it can never pass for a live run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from eval.accounting_airline import observer as obs, standin_agent as agent
from eval.accounting_airline.manifest import STANDIN_LABEL
from eval.accounting_airline.observer import Observer, install_shims, parse_native_omp, read_trace
from eval.accounting_airline.standin_agent import Bash, Ctx, DocRead, Say, run_turn
from ssgrep.accounting.readers.omp import OmpReader
from tests.accounting.reduce_builders import read_runtime

REPO = Path(__file__).resolve().parents[3]
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def _turn(tmp_path: Path, steps: list, *, env: dict[str, str] | None = None, **kw: Any) -> dict:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return run_turn(
        session_dir=tmp_path / "omp" / "-work", session_id=kw.get("session_id", "sess-1"),
        cwd=work, prompt="please do it", steps=steps, env=env or ENV, params={},
        task_file=kw.get("task_file"), task=kw.get("task"),
    )  # fmt: skip


def _records(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _product_evidence(tmp_path: Path):
    return read_runtime(OmpReader({"OMP_SESSIONS_DIR": str(tmp_path / "omp")}))


# ------------------------------------------------------------ (a) real execution + real format


def test_bash_steps_really_run_and_the_native_result_holds_their_output(tmp_path: Path) -> None:
    marker = tmp_path / "work" / "made-by-step"
    summary = _turn(
        tmp_path,
        [
            Bash("printf 'alpha\\n'"),
            Bash("echo beta; echo err >&2"),
            Bash(f"touch {marker.name}; exit 4"),
            Say("finished"),
        ],
    )
    assert marker.exists()  # the command truly executed (side effect on disk)
    assert summary["tool_calls"] == 3
    session = parse_native_omp(Path(summary["file"]))
    results = [r for r in session.records if r.kind == "tool_result"]
    assert results[0].text.startswith("alpha\n") and results[0].is_error is False
    assert results[1].text.startswith("beta\nerr\n")  # stdout + stderr, as omp shows both
    assert "[exit code: 4]" in results[2].text and results[2].is_error is True
    assert all("Wall time:" in r.text for r in results)  # faithful omp bash trailer
    calls = [r for r in session.records if r.kind == "tool_call"]
    assert [c.text for c in calls][:2] == ["printf 'alpha\\n'", "echo beta; echo err >&2"]
    assert [c.name for c in calls] == ["bash"] * 3
    assert [r.call_id for r in results] == [c.call_id for c in calls]
    assert session.records[0].kind == "user_turn" and session.records[0].text == "please do it"
    assert session.records[-1].kind == "usage"  # the closing Say


def test_trailer_can_be_disabled_by_environment(tmp_path: Path) -> None:
    summary = _turn(tmp_path, [Bash("printf x")], env={**ENV, agent.ENV_TRAILER: "none"})
    (result,) = (
        r for r in parse_native_omp(Path(summary["file"])).records if r.kind == "tool_result"
    )
    assert result.text == "x"


def test_header_and_summary_say_scripted_standin_and_product_reader_agrees(tmp_path: Path) -> None:
    summary = _turn(tmp_path, [Bash("printf 'one'"), Bash("echo two"), Say("ok")])
    path = Path(summary["file"])
    header, second = _records(summary["file"])[:2]
    assert header["type"] == "session" and header["id"] == "sess-1"
    assert header["labRuntime"] == header["provider"] == STANDIN_LABEL
    assert second["type"] == "model_change" and "standin" in second["model"]
    native = parse_native_omp(path)
    assert native.runtime_label == STANDIN_LABEL and native.session_id == "sess-1"
    native_calls = {r.call_id: r for r in native.records if r.kind == "tool_call"}
    assert len(native_calls) == 2
    # Positive twin: the PRODUCT reader decodes the same file as real omp evidence.
    evidence = _product_evidence(tmp_path)
    assert {e.runtime for e in evidence} == {"omp"}
    product_calls = {e.payload["call_id"]: e for e in evidence if e.kind == "tool_call"}
    assert set(product_calls) == set(native_calls)
    for cid, record in native_calls.items():
        payload = product_calls[cid].payload
        assert payload["name"] == "bash" and payload["arguments"] == {"command": record.text}
    results = {e.payload["call_id"]: e for e in evidence if e.kind == "tool_result"}
    assert set(results) == set(native_calls)
    assert any(e.kind == "user_turn" for e in evidence)
    assert any(e.kind == "session_header" for e in evidence)


def test_product_reader_rejects_a_mutated_non_omp_shape(tmp_path: Path) -> None:
    """Mutation proof: renaming the call blocks (so they are not omp tool calls) drops evidence."""
    summary = _turn(tmp_path, [Bash("echo hi")])
    assert any(e.kind == "tool_call" for e in _product_evidence(tmp_path))
    path = Path(summary["file"])
    path.write_text(path.read_text().replace('"toolCall"', '"tool_use"'))
    assert not any(e.kind == "tool_call" for e in _product_evidence(tmp_path))


def test_a_second_turn_reopens_the_same_session_file_and_parent_chain(tmp_path: Path) -> None:
    first = _turn(tmp_path, [Bash("echo a")])
    second = _turn(tmp_path, [Bash("echo b")])
    assert first["file"] == second["file"] and not first["continued"] and second["continued"]
    messages = [r for r in _records(first["file"]) if r["type"] == "message"]
    ids = {m["id"] for m in messages}
    assert all(m["parentId"] in ids | {None} for m in messages)
    assert sum(1 for r in _records(first["file"]) if r["type"] == "session") == 1
    calls = [r for r in parse_native_omp(Path(first["file"])).records if r.kind == "tool_call"]
    assert len({c.call_id for c in calls}) == 2


def test_task_file_is_written_before_any_step_runs(tmp_path: Path) -> None:
    task_file = tmp_path / "task"
    _turn(tmp_path, [Bash(f"cat {task_file}")], task_file=task_file, task="L3")
    (result,) = (
        r for r in parse_native_omp(Path(_only_session(tmp_path))).records
        if r.kind == "tool_result"
    )  # fmt: skip
    assert result.text.startswith("L3")


def _only_session(tmp_path: Path) -> str:
    (path,) = (tmp_path / "omp" / "-work").glob("*.jsonl")
    return str(path)


# --------------------------------------------------------------- (b) DocRead vs a quoted mention


def test_docread_is_a_read_call_that_runs_nothing_and_a_quoted_mention_only_echoes(
    tmp_path: Path,
) -> None:
    shims = install_shims(tmp_path / "bin", python=sys.executable, repo_root=REPO)
    spool = tmp_path / "spool.ndjson"
    real = tmp_path / "real-ssgrep"
    real.write_text("#!/bin/sh\necho REAL-SSGREP-RAN\n")
    real.chmod(0o755)
    env = {
        **ENV, "PATH": f"{shims['ssgrep'].parent}:/usr/bin:/bin",
        obs.ENV_SPOOL: str(spool), obs.ENV_ACTOR: "solver-b", obs.ENV_RUN: "run-1",
        obs.ENV_REAL + "SSGREP": str(real),
    }  # fmt: skip
    doc = "xd://mcp__ssgrep_search_sessions"
    summary = _turn(
        tmp_path,
        [DocRead(doc, "docs: ssgrep search <query>"), Bash('echo "ssgrep search x"')],
        env=env,
    )
    native = parse_native_omp(Path(summary["file"]))
    read_call = next(r for r in native.records if r.kind == "tool_call" and r.name == "read")
    assert read_call.text == doc and doc.startswith("xd://")
    read_result = next(
        r for r in native.records if r.call_id == read_call.call_id and r.kind == "tool_result"
    )
    assert read_result.text == "docs: ssgrep search <query>" and read_result.is_error is False
    bash = next(r for r in native.records if r.kind == "tool_call" and r.name == "bash")
    assert bash.text == 'echo "ssgrep search x"'
    bash_result = next(
        r for r in native.records if r.call_id == bash.call_id and r.kind == "tool_result"
    )
    assert bash_result.text.startswith("ssgrep search x\n")  # echo printed the words only
    assert "REAL-SSGREP-RAN" not in bash_result.text
    assert not spool.exists() or read_trace(spool) == []  # no dispatch: nothing executed ssgrep
    # Positive twin: an actual `ssgrep` invocation in the same environment IS dispatched.
    _turn(tmp_path, [Bash("ssgrep search x")], env=env, session_id="sess-2")
    dispatches = [e for e in read_trace(spool) if e["kind"] == "dispatch"]
    assert [d["argv"] for d in dispatches] == [["ssgrep", "search", "x"]]
    call_ids = {
        r.call_id for r in parse_native_omp(_session_file(tmp_path, "sess-2")).records
        if r.kind == "tool_call"
    }  # fmt: skip
    assert dispatches[0]["native_call_id"] in call_ids  # exact link via LAB_NATIVE_CALL_ID


def _session_file(tmp_path: Path, session_id: str) -> Path:
    (path,) = (tmp_path / "omp" / "-work").glob(f"*_{session_id}.jsonl")
    return path


def test_docread_evidence_reaches_the_product_reader_as_a_read_call(tmp_path: Path) -> None:
    _turn(tmp_path, [DocRead("xd://some_tool", "doc text")])
    calls = [e for e in _product_evidence(tmp_path) if e.kind == "tool_call"]
    assert [(c.payload["name"], c.payload["arguments"]) for c in calls] == [
        ("read", {"path": "xd://some_tool"})
    ]


# ------------------------------------------------------------------------ (c) dynamic steps


def test_dynamic_step_reads_the_earlier_outcome_and_none_is_skipped(tmp_path: Path) -> None:
    seen: list[Any] = []

    def follow_up(ctx: Ctx) -> str | None:
        assert ctx.last is not None
        seen.append((ctx.last.command, ctx.last.returncode, len(ctx.outcomes)))
        return f"echo got-{ctx.last.text.split()[0]}"

    def skipped(ctx: Ctx) -> str | None:
        seen.append("skipped-called")
        return None

    def closing(ctx: Ctx) -> str:
        return f"outcomes={len(ctx.outcomes)}"

    summary = _turn(
        tmp_path,
        [Bash("echo first"), Bash(follow_up), Bash(skipped), Say(closing)],
    )
    assert seen == [("echo first", 0, 1), "skipped-called"]
    assert summary["tool_calls"] == 2  # the None step produced no call and no outcome
    session = parse_native_omp(Path(summary["file"]))
    calls = [r.text for r in session.records if r.kind == "tool_call"]
    assert calls == ["echo first", "echo got-first"]
    results = [r.text for r in session.records if r.kind == "tool_result"]
    assert results[1].startswith("got-first\n")


def test_dynamic_say_is_written_as_the_closing_assistant_text(tmp_path: Path) -> None:
    summary = _turn(tmp_path, [Bash("echo x"), Say(lambda ctx: f"saw {len(ctx.outcomes)} call")])
    last = _records(summary["file"])[-1]["message"]
    assert last["role"] == "assistant" and last["stopReason"] == "stop"
    assert last["content"] == [{"type": "text", "text": "saw 1 call"}]
    assert last["provider"] == STANDIN_LABEL


def test_timeout_step_records_timed_out_error_with_no_exit_code(tmp_path: Path) -> None:
    summary = _turn(tmp_path, [Bash("sleep 30", timeout=0.3), Bash("echo after")])
    session = parse_native_omp(Path(summary["file"]))
    first, second = (r for r in session.records if r.kind == "tool_result")
    assert "[timed out]" in first.text and first.is_error is True
    assert second.is_error is False


# ------------------------------------------------------------------- (d) never mistaken for live


def test_every_artifact_carries_the_standin_label(tmp_path: Path) -> None:
    summary = _turn(tmp_path, [Bash("echo hi"), DocRead("xd://t", "d"), Say("bye")])
    assert summary["label"] == STANDIN_LABEL
    records = _records(summary["file"])
    assert records[0]["labRuntime"] == STANDIN_LABEL
    assert all(
        r["message"]["provider"] == STANDIN_LABEL
        for r in records
        if r["type"] == "message" and r["message"]["role"] == "assistant"
    )
    observer = Observer(tmp_path / "verifier", "run-1")
    session = observer.record_native("solver-a", "L1", Path(summary["file"]), source_root=tmp_path)
    assert session.runtime_label == STANDIN_LABEL
    event = next(e for e in read_trace(observer.trace.path) if e["kind"] == "native_session")
    assert event["runtime_label"] == STANDIN_LABEL
    frozen = observer.freeze()
    assert frozen.sessions["sess-1"][2].runtime_label == STANDIN_LABEL


def test_cli_entry_writes_the_summary_with_the_label(tmp_path: Path) -> None:
    params = tmp_path / "params.json"
    params.write_text('{"database": "airline"}')
    done = subprocess.run(
        [sys.executable, "-m", "eval.accounting_airline.standin_agent", "--script", "nope",
         "--params", str(params), "--session-dir", str(tmp_path / "s"), "--session-id", "x",
         "--cwd", str(tmp_path)],  # fmt: skip
        capture_output=True, text=True, cwd=REPO, env={**ENV, "PYTHONPATH": str(REPO)},
        check=False, timeout=120,
    )  # fmt: skip
    assert done.returncode != 0  # an unknown script is refused, nothing is written
    assert not (tmp_path / "s").exists() or not list((tmp_path / "s").glob("*.jsonl"))


def test_docs_script_is_a_single_read_and_never_a_bash_call() -> None:
    from eval.accounting_airline import standin_scripts

    prompt, steps = standin_scripts.build("DOCS", {"database": "airline"})
    assert prompt and [type(s) for s in steps] == [DocRead, Say]
    assert isinstance(steps[0], DocRead) and steps[0].path.startswith("xd://")
    with pytest.raises(ValueError, match="unknown scripted turn"):
        standin_scripts.build("L9", {"database": "airline"})
