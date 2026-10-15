"""Running scripted-standin sessions and grading what they REALLY produced.

A stand-in turn is a real subprocess (``standin_agent``) whose shell tools go through the observer
shims to real PostgreSQL roles and the real ``ssgrep`` binary. Grading never re-derives expected
values from the agent's SQL: it parses the OBSERVED tool output (psql tables captured by the
observer) and hands those actual rows to the lab's independent ``domain_oracle`` verifiers.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import domain_oracle as oracle, l4_reference as l4, l4_verify, queries
from .lab_product import BINARY, REPO_ROOT
from .observer import (
    ENV_ACTOR,
    ENV_REAL,
    ENV_RUN,
    ENV_SPOOL,
    ENV_SQL_HOST,
    ENV_SQL_PGPASS,
    ENV_SQL_PORT,
    ENV_TASK_FILE,
    Frozen,
    Observer,
    install_shims,
    parse_psql_table,
    write_pgpass,
)
from .observer_blob import read_blob
from .roots import LabRoots, runtime_env
from .sandbox import SandboxProfile, run_sandboxed

STANDIN = "scripted-standin"
TURN_TIMEOUT = 900.0


@dataclass(frozen=True)
class Turn:
    """One user turn handed to a stand-in process."""

    script: str
    task: str  # e.g. "A/L1"
    params: Mapping[str, Any]


@dataclass
class TurnRun:
    turn: Turn
    actor: str
    session_id: str
    session_file: Path
    returncode: int | None
    pid: int | None
    elapsed_ms: int
    summary: dict[str, Any]
    spool: Path
    stderr_chars: int
    source_root: Path


@dataclass
class ActorKit:
    """Everything one solver actor needs: env, shims, spool, credential references."""

    actor: str
    env: dict[str, str]
    spool: Path
    task_file: Path
    sessions_dir: Path


def make_kit(
    roots: LabRoots,
    actor: str,
    *,
    run_id: str,
    sessions_dir: Path,
    model_cache: Path,
    sql_port: int,
    passwords: Mapping[str, str],
    shim_dir: Path,
    spool_dir: Path,
) -> ActorKit:
    """Explicit environment (no ambient copy) with ``ssgrep``/``sql`` shims first on PATH."""
    env = runtime_env(roots, actor, runtime_sessions=sessions_dir, model_cache=model_cache)
    spool_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pgpass = roots.actor_home(actor) / ".pgpass-lab"  # per actor: B never sees A's file
    write_pgpass(pgpass, host="127.0.0.1", port=sql_port, passwords=passwords)
    psql = shutil.which("psql")
    if psql is None:
        raise RuntimeError("psql is required for the restricted SQL tool")
    spool = spool_dir / f"{actor}.ndjson"
    task_file = spool_dir / f"{actor}.task"
    task_file.write_text("")
    env["PATH"] = os.pathsep.join([str(shim_dir), env["PATH"]])
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.update(
        {
            ENV_SPOOL: str(spool),
            ENV_ACTOR: actor,
            ENV_RUN: run_id,
            ENV_TASK_FILE: str(task_file),
            ENV_REAL + "SSGREP": str(BINARY),
            ENV_REAL + "PSQL": psql,
            ENV_SQL_HOST: "127.0.0.1",
            ENV_SQL_PORT: str(sql_port),
            ENV_SQL_PGPASS: str(pgpass),
        }
    )
    return ActorKit(actor, env, spool, task_file, sessions_dir)


def install_tools(roots: LabRoots) -> Path:
    directory = roots.tools / "solver"
    install_shims(directory, python=sys.executable, repo_root=REPO_ROOT)
    return directory


def run_turn(
    kit: ActorKit,
    turn: Turn,
    *,
    session_id: str,
    cwd: Path,
    params_dir: Path,
    profile: SandboxProfile | None,
) -> TurnRun:
    """One stand-in process; sandboxed when ``profile`` is given. Exactly once, never retried."""
    params_dir.mkdir(parents=True, exist_ok=True)
    params = params_dir / f"{kit.actor}-{turn.script}.params.json"
    summary = params_dir / f"{kit.actor}-{turn.script}.summary.json"
    params.write_text(json.dumps(dict(turn.params)))
    argv = [
        sys.executable, "-m", "eval.accounting_airline.standin_agent",
        "--script", turn.script, "--params", str(params), "--session-dir", str(kit.sessions_dir),
        "--session-id", session_id, "--cwd", str(cwd), "--task-file", str(kit.task_file),
        "--task", turn.task, "--summary", str(summary),
    ]  # fmt: skip
    started = time.monotonic()
    import subprocess

    if profile is not None:
        done = run_sandboxed(profile, argv, env=kit.env, cwd=cwd, timeout=TURN_TIMEOUT)
    else:
        done = subprocess.run(
            argv, env=kit.env, cwd=cwd, capture_output=True, text=True, timeout=TURN_TIMEOUT,
            check=False,
        )  # fmt: skip
    data = json.loads(summary.read_text()) if summary.exists() else {}
    if done.returncode:
        data["stderr_tail"] = done.stderr[-1500:]
    path = Path(data["file"]) if data.get("file") else next(kit.sessions_dir.glob("*.jsonl"))
    return TurnRun(
        turn,
        kit.actor,
        session_id,
        path,
        done.returncode,
        None,
        int((time.monotonic() - started) * 1000),
        data,
        kit.spool,
        len(done.stderr),
        kit.sessions_dir,
    )


def absorb(observer: Observer, run: TurnRun) -> None:
    """Fold a finished turn into the verifier-owned trace (native records + shim spool)."""
    spool_blobs = run.spool.parent / (run.spool.stem + "-blobs")
    association = observer.record_native(
        run.actor, run.turn.task, run.session_file, source_root=run.source_root
    )
    observer.ingest_spool(
        run.spool,
        spool_blobs,
        expected_actor=run.actor,
        expected_task=run.turn.task,
        native_association=association,
    )


# ----------------------------------------------------------------------- tagged results


def statement_of(observer: Observer, call: Any) -> str:
    for event in call.sql:
        blob = event.get("statement_blob")
        if blob:
            from .observer import MAX_BLOB_BYTES

            return read_blob(observer.blobs, blob, limit=MAX_BLOB_BYTES)
    return ""


def tagged_tables(
    frozen: Frozen, observer: Observer, actor: str, *, task: str | None = None
) -> dict[str, list[dict[str, str]]]:
    """``tag -> rows`` of the actor's OBSERVED successful SQL results (``-- step:<tag>``)."""
    out: dict[str, list[dict[str, str]]] = {}
    for call in frozen.calls:
        if call.tool != "sql" or call.actor != actor or call.state != "verified":
            continue
        if call.exit_code != 0 or (task is not None and call.task != task):
            continue
        text = statement_of(observer, call)
        marker = "-- step:"
        if marker not in text:
            continue
        tag = text.split(marker, 1)[1].splitlines()[0].strip()
        rows = parse_psql_table(call.stdout)
        if rows is not None:
            out[tag] = rows
    return out


def sql_attempts(frozen: Frozen, observer: Observer, actor: str) -> list[dict[str, Any]]:
    """Every observed SQL attempt (including failures/repairs) with its transaction outcome."""
    out = []
    for call in frozen.calls:
        if call.tool != "sql" or call.actor != actor:
            continue
        finish = next((e for e in call.sql if e.get("phase") == "finish"), {})
        text = statement_of(observer, call)
        tag = text.split("-- step:", 1)[1].splitlines()[0].strip() if "-- step:" in text else ""
        out.append(
            {
                "call": call.call_id,
                "tag": tag,
                "state": call.state,
                "exit": call.exit_code,
                "txn": finish.get("txn"),
                "sqlstate": finish.get("sqlstate"),
                "task": call.task,
                "role": next((e.get("role") for e in call.sql if e.get("phase") == "start"), None),
            }  # fmt: skip
        )
    return out


# ---------------------------------------------------------------- type conversion for graders


def _bool(value: str) -> bool:
    return value in ("t", "true", "True", "1")


def cash_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [{**r, "settled_minor": int(r["settled_minor"])} for r in rows]


def connection_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {**r, "gap_minutes": int(r["gap_minutes"]), "minimum_minutes": int(r["minimum_minutes"]),
         "valid": _bool(r["valid"])}
        for r in rows
    ]  # fmt: skip


def clock_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [{**r, "utc_offset_hours": int(r["utc_offset_hours"])} for r in rows]


def route_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [{**r, "legs": [x for x in r["legs"].strip("{}").split(",") if x]} for r in rows]


# ----------------------------------------------------------------------------- graders


def grade_l1(
    tables: Mapping[str, list[dict[str, str]]], state: oracle.State
) -> dict[str, list[str]]:
    return {
        "cash": oracle.verify_cash(cash_rows(tables.get("cash", [])), state),
        "seats": oracle.verify_seats(tables.get("seats", []), state),
        "naive_cash_detected": oracle.verify_cash(cash_rows(tables.get("cash_naive", [])), state),
    }


def grade_l2(
    tables: Mapping[str, list[dict[str, str]]], state: oracle.State
) -> dict[str, list[str]]:
    return {
        "affected": oracle.verify_affected(tables.get("affected", []), state),
        "connections": oracle.verify_connections(
            connection_rows(tables.get("connections", [])), state
        ),
        "local_clock": oracle.verify_local_clock(clock_rows(tables.get("clocks", [])), state),
        "instants": oracle.verify_flight_instants(tables.get("instants", []), state),
    }


def grade_l3(
    tables: Mapping[str, list[dict[str, str]]],
    plan: Mapping[str, Sequence[str]],
    state: oracle.State,
) -> dict[str, list[str]]:
    policy_expected: list[str] = []
    multi = [tuple(v) for v in plan.values() if len(v) > 1]
    if multi:
        policy_expected = oracle.policy_ids(state, multi[0])
    policy = [r["rule_id"] for r in tables.get("policy", [])]
    return {
        "routes": oracle.verify_routes(route_rows(tables.get("routes", [])), state),
        "joint_plan": oracle.plan_violations(state, {b: tuple(v) for b, v in plan.items()}),
        "policy": []
        if policy == policy_expected
        else [f"policy ids {policy} != {policy_expected}"],
    }


def grade_l4(
    before: oracle.State, after: oracle.State, plan: Mapping[str, Sequence[str]]
) -> list[str]:
    requests = [
        l4.request_for(cast(Any, before), f"l4:lab:{b}", b, tuple(legs))
        for b, legs in sorted(plan.items())
    ]
    return l4_verify.verify_joint_apply(before, after, requests)


def policy_ids_in(text: str) -> list[str]:
    import re

    return sorted(set(re.findall(r"\bMCT-[A-Z]{3}-\d{3}\b", text)))


__all__ = ["queries"]  # re-export marker for lint (queries is used by callers via this module)
