"""Four bounded native turns, never scripted answer execution or retries."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Protocol

from .domain_lane import write_private
from .gates import FAIL
from .lab_context import Run
from .lab_product import REPO_ROOT
from .lab_session import ActorKit
from .live_bridge import Bridge
from .live_prompts import prompt_for
from .live_security import live_profile
from .live_sql import library_path
from .observer import ENV_REAL, ENV_SPOOL, ENV_SQL_PGPASS, ENV_TASK_FILE
from .report import GateEntry, Report
from .roots import runtime_env
from .runtime_omp import TurnResult, TurnSpec, run_turn

READER_ENV = {"COLUMNS": "120", "NO_COLOR": "1"}


class LiveBlocked(RuntimeError):
    """A precise failed prerequisite, retained without fabricating native evidence."""


def seal_kit(run: Run, kit: ActorKit) -> None:
    kit.env.update(READER_ENV)
    private = run.roots.verifier / kit.actor
    private.mkdir(mode=0o700, exist_ok=True)
    old = Path(kit.env[ENV_SQL_PGPASS])
    password_file = private / "pgpass"
    old.rename(password_file)
    kit.env[ENV_SQL_PGPASS] = str(password_file)
    kit.spool = private / "spool.ndjson"
    kit.task_file = private / "task"
    kit.task_file.write_text("")
    kit.env[ENV_SPOOL], kit.env[ENV_TASK_FILE] = str(kit.spool), str(kit.task_file)
    kit.env["AIRLINE_LIBPQ_PATH"] = str(library_path(kit.env[ENV_REAL + "PSQL"]))
    backend = run.roots.tools / "admin" / f"sql-protocol-{kit.actor}"
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "from eval.accounting_airline.live_sql import main; raise SystemExit(main())"
    )
    backend.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -c {shlex.quote(code)} "$@"\n'
    )
    backend.chmod(0o700)
    kit.env[ENV_REAL + "PSQL"] = str(backend)


class _CaptureReport(Protocol):
    @property
    def report(self) -> Report: ...


class _CaptureStatus(Protocol):
    @property
    def capture_status(self) -> str: ...


def capture_failure(
    run: _CaptureReport, task: str, result: _CaptureStatus, attempts: list[dict]
) -> str | None:
    """Name every supervisor failure before ingestion; retained native billing is independent."""
    details = []
    native_status = result.capture_status or "unavailable"
    if native_status != "completed":
        detail = f"{task}: native capture_status={native_status}; no retries"
        run.report.add_gate(
            "live_session_gate", GateEntry(f"native.{task}.{native_status}", FAIL, detail)
        )
        details.append(detail)
    for status in sorted(
        {a.get("capture_status") or "unavailable" for a in attempts} - {"completed"}
    ):
        count = sum((a.get("capture_status") or "unavailable") == status for a in attempts)
        detail = f"{task}: bridge capture_status={status}; attempts={count}; no retries"
        run.report.add_gate(
            "live_session_gate", GateEntry(f"native.{task}.bridge.{status}", FAIL, detail)
        )
        details.append(detail)
    return "; ".join(details) or None


def native_turn(run: Run, bridge: Bridge, task: str, model: str) -> TurnResult:
    actor = "solver-a" if task in ("L1", "L2") else "solver-b"
    kit = run.kit_a if actor == "solver-a" else run.kit_b
    label = "a" if actor == "solver-a" else "b"
    database = run.a_db if label == "a" else run.b_db
    first = task in ("L1", "L3")
    kit.task_file.write_text(f"{label.upper()}/{task}")
    bridge.writer = task == "L4"
    clients = run.roots.tools / actor
    bridge.install_clients(clients)
    env = runtime_env(run.roots, actor, runtime_sessions=kit.sessions_dir, model_cache=run.cache)
    env.update(READER_ENV)
    run.notes["auth_gateway"].configure(actor, env)
    env["PATH"] = str(clients) + os.pathsep + env["PATH"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    scratch = run.roots.actor_home(actor) / "tmp"
    scratch.mkdir(mode=0o700, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    spec = TurnSpec(
        prompt_for(task, database),
        kit.sessions_dir,
        run.roots.project,
        ("bash",),
        model=model,
        continue_session=not first,
        max_time="10m",
        thinking=run.manifest.sampling.get("thinking", "low"),
        extra_flags=("--auto-approve", "--no-lsp", "--no-pty"),
    )
    output = run.roots.artifacts / "native-attempts" / task
    (run.roots.artifacts / "prompts").mkdir(mode=0o700, exist_ok=True)
    write_private(
        run.roots.artifacts / "prompts" / f"{task}.json", {"task": task, "prompt": spec.prompt}
    )
    first_bridge_attempt = len(bridge.attempts)
    result = run_turn(
        spec,
        env=env,
        profile=live_profile(run.roots, actor),
        timeout=620,
        authorized=True,
        output_dir=output,
    )
    attempts = bridge.attempts[first_bridge_attempt:]
    payload = {**result.to_dict(), "bridge_attempts": attempts}
    write_private(output / "result.json", payload)
    run.notes.setdefault("turns", {})[task] = payload
    capture_error = capture_failure(run, task, result, attempts)
    blobs = kit.spool.parent / (kit.spool.stem + "-blobs")
    if len(result.native.session_files) != 1 or not result.native.session_id:
        if capture_error:
            raise LiveBlocked(capture_error)
        raise LiveBlocked(
            f"{task}: native runtime produced {len(result.native.session_files)} "
            f"sessions (expected one); exit={result.returncode}; "
            f"diagnostics={output / 'stderr.txt'}"
        )
    native = result.native.session_files[0]
    sid = result.native.session_id
    old_sid = run.a_sid if label == "a" else run.b_sid
    if not first and sid != old_sid:
        raise LiveBlocked(f"{task}: --continue changed native identity {old_sid} -> {sid}")
    if label == "a":
        run.a_sid = sid
    else:
        run.b_sid = sid
        if sid == run.a_sid:
            raise LiveBlocked("B inherited A's native identity")
    run.native[label] = native
    association = run.observer.record_native(
        actor, f"{label.upper()}/{task}", native, source_root=spec.session_dir
    )
    run.observer.ingest_spool(
        kit.spool,
        blobs,
        expected_actor=actor,
        expected_task=f"{label.upper()}/{task}",
        native_association=association,
    )
    actor_tasks = ("L1", "L2") if label == "a" else ("L3", "L4")
    run.observer.attest_supervisor_spool(
        kit.spool,
        blobs,
        expected_actor=actor,
        expected_tasks={
            f"{label.upper()}/{name}" for name in actor_tasks if name in run.notes["turns"]
        },
    )
    if capture_error:
        raise LiveBlocked(capture_error)
    if result.native.usage_records == 0:
        raise LiveBlocked(
            f"{task}: native session {sid} contains no provider usage records; "
            f"exit={result.returncode}; diagnostics={output / 'stderr.txt'}"
        )
    return result
