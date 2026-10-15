"""Independent observer: an append-only, verifier-owned trace of what REALLY happened.

Nothing here reads the product journal, reducer or collector. Evidence comes from three
places the product does not control:

* ``Observer.run`` executes an argv list (never a shell string) for the verifier/control
  actors and records dispatch, raw output, exit status and duration;
* the ``shim`` entry point (``python -m eval.accounting_airline.observer shim ...``) stands in
  front of ``ssgrep`` and the restricted SQL tool for the solver actors and spools the same
  facts from INSIDE the solver's own process tree, so a shell loop with two real searches
  yields two dispatches;
* ``parse_native_omp`` reads the runtime's native session file with its own small parser.

Every record carries run/actor/task ids and is hash chained, so a broken or edited trace is
detected (``read_trace`` raises). A call's state is one of ``attempted`` (dispatch seen, no
completion evidence yet), ``verified`` (completion observed), ``interrupted`` (killed or timed
out) or ``unknown`` (the observer could not tell). A SQL ``txn`` outcome is a separate fact
from the tool call that carried it.

Limitation (named, recorded in manifests): the solver-side spool is writable by the solver's
own process, so it is tamper-EVIDENT only through the cross-check against native records.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .observer_blob import (
    MAX_SOURCE_BYTES,
    ObserverError,
    native_invokes,
    native_output_matches,
    read_blob,
    read_owned,
    read_source,
    record_blobs,
    supervisor_receipts,
    validate_spool,
)

ACTORS = ("solver-a", "solver-b", "verifier", "control")
STATES = ("attempted", "verified", "interrupted", "unknown")
TRACE_NAME = "observer-trace.ndjson"  # not *.jsonl: transcript discovery must never see it
BLOB_DIR = "observer-blobs"
MAX_BLOB_BYTES = 1_000_000
MAX_ARGV_CHARS = 4000
GENESIS = "0" * 64
ENV_SPOOL = "LAB_OBSERVER_SPOOL"
ENV_ACTOR = "LAB_OBSERVER_ACTOR"
ENV_TASK_FILE = "LAB_OBSERVER_TASK_FILE"
ENV_RUN = "LAB_RUN_ID"
ENV_NATIVE_CALL = "LAB_NATIVE_CALL_ID"  # stand-in only: exact link to its native tool call
ENV_REAL = "LAB_OBSERVER_REAL_"  # + upper-case tool name -> absolute path of the real binary
ENV_SQL_HOST, ENV_SQL_PORT, ENV_SQL_PGPASS = "LAB_SQL_HOST", "LAB_SQL_PORT", "LAB_SQL_PGPASS"
SQL_USERS = {"reader": "lab_reader", "writer": "lab_writer"}
SQLSTATE = re.compile(r"ERROR:\s+([0-9A-Z]{5}):")
KNOWN_LIMITATION = "solver_spool_writable_by_solver_cross_checked_against_native_records"


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ----------------------------------------------------------------------------- trace


class Trace:
    """Append-only JSONL hash chain; safe for concurrent appenders (flock)."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path, self.run_id = path, run_id
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        os.close(fd)

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            prev, seq = _tail(fd)
            body = {
                "seq": seq,
                "prev": prev,
                "ts_utc": utc_now(),
                "wall_ns": time.time_ns(),
                "run_id": self.run_id,
                **record,
            }
            body["hash"] = sha(_canon(body))
            os.write(fd, _canon(body) + b"\n")
            return body
        finally:
            os.close(fd)


def _tail(fd: int) -> tuple[str, int]:
    size = os.fstat(fd).st_size
    if size == 0:
        return GENESIS, 0
    with os.fdopen(os.dup(fd), "rb") as handle:
        handle.seek(max(0, size - 65536))
        lines = [ln for ln in handle.read().split(b"\n") if ln.strip()]
    last = json.loads(lines[-1])
    return last["hash"], last["seq"] + 1


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Every record, verifying the chain; a broken link raises ``ObserverError``."""
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    return _read_trace_bytes(path.read_bytes())


def _read_trace_bytes(data: bytes) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prev = GENESIS
    for number, line in enumerate(data.split(b"\n")):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            claimed = record.pop("hash")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ObserverError(f"trace line {number} is not a valid record") from exc
        if record.get("prev") != prev or sha(_canon({**record})) != claimed:
            raise ObserverError(f"trace chain broken at seq {record.get('seq')}")
        record["hash"] = claimed
        prev = claimed
        out.append(record)
    return out


# ------------------------------------------------------------------ native session parse


@dataclass(frozen=True)
class NativeRecord:
    seq: int
    kind: str  # user_turn | tool_call | tool_result | usage
    record_id: str
    text: str = ""
    call_id: str | None = None
    name: str | None = None
    ts_ms: int | None = None
    is_error: bool | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    cost_nano_usd: int | None = None
    origin_id: str | None = (
        None  # Native message/block identity, independent of reusable toolCallId.
    )

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def digest(self) -> str:
        return sha(_canon([self.kind, self.record_id, self.text, self.call_id]))


@dataclass(frozen=True)
class NativeSession:
    session_id: str
    path: str
    cwd: str | None
    runtime_label: str | None
    model: str | None
    records: tuple[NativeRecord, ...]
    file_bytes: int
    complete_bytes: int  # newline-complete prefix
    partial_tail_bytes: int
    file_digest: str
    malformed_lines: int = 0
    parent_session: str | None = None
    source_aliases: tuple[str, ...] = ()


def _ms(stamp: Any) -> int | None:
    if not isinstance(stamp, str):
        return None
    try:
        return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return ""


def call_text(arguments: Any) -> str:
    """The model-written text of a tool call: a shell command verbatim, else string leaves."""
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        for key in ("command", "cmd", "script"):
            if isinstance(arguments.get(key), str):
                return arguments[key]
    leaves: list[str] = []

    def walk(value: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, str):
            leaves.append(value)
        elif isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(arguments)
    return "\n".join(leaves)


def _usage_of(raw: Any) -> tuple[dict[str, int], int | None]:
    names = {
        "input": "input",
        "output": "output",
        "cacheRead": "cache_read",
        "cacheWrite": "cache_write",
        "totalTokens": "total",
        "reasoningTokens": "reasoning",
    }
    tokens = {n: raw[k] for k, n in names.items() if isinstance(raw.get(k), int) and raw[k] >= 0}
    cost = raw.get("cost")
    nano = None
    if isinstance(cost, dict) and isinstance(cost.get("total"), int | float):
        nano = round(float(cost["total"]) * 1_000_000_000)  # float only to read provider JSON
    return tokens, nano


def parse_native_omp(path: Path) -> NativeSession:
    """Parse an omp/pi-format JSONL on the newline-complete prefix; partial tails are counted."""
    data = path.read_bytes()
    return _parse_native_bytes(path, data)


def _parse_native_bytes(path: Path, data: bytes) -> NativeSession:
    parent_session = None
    cut = data.rfind(b"\n") + 1
    complete, tail = data[:cut], data[cut:]
    session_id, cwd, label, model = path.stem, None, None, None
    records: list[NativeRecord] = []
    malformed = 0
    for line in complete.split(b"\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if not isinstance(obj, dict):
            malformed += 1
            continue
        kind, entry, ts = obj.get("type"), str(obj.get("id") or ""), _ms(obj.get("timestamp"))
        if kind == "session":
            session_id = str(obj.get("id") or session_id)
            cwd = obj.get("cwd") if isinstance(obj.get("cwd"), str) else None
            label = obj.get("labRuntime") if isinstance(obj.get("labRuntime"), str) else None
            parent_session = (
                obj.get("parentSession") if isinstance(obj.get("parentSession"), str) else None
            )
            continue
        message = obj.get("message")
        if kind != "message" or not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            text = _text_blocks(message.get("content"))
            if text:
                records.append(
                    NativeRecord(
                        len(records),
                        "user_turn",
                        f"turn:{entry}",
                        text,
                        ts_ms=ts,
                        origin_id=f"message:{entry}:user" if entry else None,
                    )
                )
        elif role == "assistant":
            model = message.get("model") or model
            if isinstance(message.get("usage"), dict):
                tokens, nano = _usage_of(message["usage"])
                records.append(
                    NativeRecord(
                        len(records),
                        "usage",
                        f"usage:{entry}",
                        ts_ms=ts,
                        usage=tokens,
                        cost_nano_usd=nano,
                        origin_id=f"message:{entry}:usage" if entry else None,
                    )  # fmt: skip
                )
            for ordinal, block in enumerate(message.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    records.append(
                        NativeRecord(
                            len(records),
                            "tool_call",
                            f"call:{block.get('id')}",
                            call_text(block.get("arguments")),
                            str(block.get("id")),
                            str(block.get("name")),
                            ts,
                            origin_id=f"message:{entry}:call:{ordinal}" if entry else None,
                        )  # fmt: skip
                    )
        elif role == "toolResult":
            flag = message.get("isError")
            records.append(
                NativeRecord(
                    len(records),
                    "tool_result",
                    f"result:{message.get('toolCallId')}",
                    _text_blocks(message.get("content")),
                    str(message.get("toolCallId")),
                    str(message.get("toolName")),
                    ts,
                    flag if isinstance(flag, bool) else None,
                    origin_id=f"message:{entry}:result" if entry else None,
                )  # fmt: skip
            )
    return NativeSession(
        session_id,
        str(path),
        cwd,
        label,
        model if isinstance(model, str) else None,
        tuple(records),
        len(data),
        len(complete),
        len(tail),
        sha(complete),
        malformed,
        parent_session,
    )


# ------------------------------------------------------------------------ frozen evidence


@dataclass(frozen=True)
class ObservedCall:
    call_id: str
    actor: str
    task: str | None
    tool: str
    argv: tuple[str, ...]
    state: str
    exit_code: int | None
    stdout: str
    stdout_chars: int
    stdout_bytes: int
    duration_ns: int | None
    dispatch_seq: int
    sql: tuple[Mapping[str, Any], ...] = ()
    wall_ns: int = 0
    native_call_id: str | None = None  # set by ``link_native`` (time-window join), else None
    via_spool: bool = False
    native_session_id: str | None = None


@dataclass(frozen=True)
class Frozen:
    """Everything the independent oracle may consume; no product-derived value is in here."""

    run_id: str
    calls: tuple[ObservedCall, ...]
    sessions: Mapping[str, tuple[str, str | None, NativeSession]]  # key -> (actor, task, session)
    phases: tuple[Mapping[str, Any], ...]
    trace_head: str


class Observer:
    """The verifier-side recorder. One instance per lab run; trace lives in ``verifier/``."""

    def __init__(self, verifier_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = verifier_dir
        self.blobs = verifier_dir / BLOB_DIR
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(verifier_dir / TRACE_NAME, run_id)
        self._ingested: dict[str, tuple[int, str]] = {}
        self._native: dict[tuple[str, str | None], NativeSession] = {}
        self._native_calls: dict[tuple[str, str | None], set[str]] = {}

    # -- blobs ------------------------------------------------------------------------
    def blob(self, text: str) -> str:
        if len(text.encode()) > MAX_BLOB_BYTES:
            raise ObserverError("observer output exceeds blob bound")
        digest = sha(text)
        target = self.blobs / digest
        if not target.exists():
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(text.encode()[:MAX_BLOB_BYTES])
        return digest

    def phase(self, name: str, **detail: Any) -> dict[str, Any]:
        return self.trace.append({"kind": "phase", "name": name, **detail})

    # -- verifier/control commands ------------------------------------------------------
    def run(
        self,
        *,
        actor: str,
        task: str | None,
        tool: str,
        argv: Sequence[str],
        env: Mapping[str, str],
        cwd: Path,
        timeout: float,
        stdin: str | None = None,
        shown: Sequence[str] | None = None,
    ) -> tuple[ObservedCall, subprocess.CompletedProcess[str] | None]:
        """Execute ``argv`` (a list) and record dispatch + raw result. Never a shell string.

        ``shown`` is the normalized argv written to the trace (``['ssgrep', 'search', ...]``)
        when the executed argv starts with an absolute binary path.
        """
        from .bounded_capture import capture_process, decode_text, text_encoding

        if actor not in ACTORS:
            raise ObserverError(f"unknown actor {actor!r}")
        call_id = f"{actor}-{secrets.token_hex(5)}"
        self.trace.append(dispatch_record(call_id, actor, task, tool, shown or argv, str(cwd)))
        started = time.monotonic_ns()
        proc: subprocess.CompletedProcess[str] | None = None
        try:
            captured = capture_process(
                list(argv),
                input_bytes=stdin.encode(text_encoding()) if stdin is not None else None,
                env=env,
                cwd=cwd,
                timeout=timeout,
                stdout_limit=MAX_BLOB_BYTES,
                stderr_limit=MAX_BLOB_BYTES,
                output_dir=self.dir / "control-attempts" / call_id,
            )
            assert captured.returncode is not None  # The owned child has been reaped.
            if captured.status != "completed":
                self.trace.append(
                    result_record(
                        call_id,
                        "interrupted",
                        None if captured.timed_out else captured.returncode,
                        "",
                        "",
                        time.monotonic_ns() - started,
                        self.blob,
                        reason=captured.status,
                        capture=captured.metadata(),
                    )
                )
                if captured.status == "interrupted":
                    proc = subprocess.CompletedProcess(list(argv), captured.returncode, "", "")
                if captured.output_limit:
                    raise ObserverError("verifier/control capture output_limit; attempt retained")
            else:
                try:
                    stdout = decode_text(captured.stdout)
                    stderr = decode_text(captured.stderr)
                except UnicodeDecodeError:
                    metadata = {
                        **captured.metadata(),
                        "capture_status": "invalid_output_encoding",
                        "termination_cause": "invalid_output_encoding",
                        "capture_complete": False,
                    }
                    self.trace.append(
                        result_record(
                            call_id,
                            "interrupted",
                            captured.returncode,
                            "",
                            "",
                            time.monotonic_ns() - started,
                            self.blob,
                            reason="invalid_output_encoding",
                            capture=metadata,
                        )
                    )
                    raise ObserverError(
                        "verifier/control invalid_output_encoding; attempt retained"
                    ) from None
                proc = subprocess.CompletedProcess(list(argv), captured.returncode, stdout, stderr)
                state = "verified" if proc.returncode >= 0 else "interrupted"
                self.trace.append(
                    result_record(
                        call_id,
                        state,
                        proc.returncode,
                        stdout,
                        stderr,
                        time.monotonic_ns() - started,
                        self.blob,
                        capture=captured.metadata(),
                    )
                )
        except OSError as exc:
            self.trace.append(
                result_record(
                    call_id,
                    "unknown",
                    None,
                    "",
                    "",
                    time.monotonic_ns() - started,
                    self.blob,
                    reason=type(exc).__name__,
                )  # fmt: skip
            )
        return self._call(call_id), proc

    def _call(self, call_id: str) -> ObservedCall:
        return next(
            c for c in freeze_calls(read_trace(self.trace.path), self.blobs) if c.call_id == call_id
        )

    # -- native sessions & spools -----------------------------------------------------------
    def record_native(
        self,
        actor: str,
        task: str | None,
        path: Path,
        *,
        source_root: Path,
        source_aliases: Mapping[str, str] | None = None,
    ) -> NativeSession:
        """Snapshot a supervisor-selected native source, never a spool-provided path."""
        if actor not in ACTORS:
            raise ObserverError("invalid native actor")
        from dataclasses import replace

        data = read_source(path, source_root)
        aliases = source_aliases or {}
        if any(
            not isinstance(alias, str) or digest != sha(data) for alias, digest in aliases.items()
        ):
            raise ObserverError("native copy alias does not match the owned source digest")
        session = replace(_parse_native_bytes(path, data), source_aliases=tuple(aliases))
        if any(
            s.session_id == session.session_id and owner != actor
            for (owner, _), s in self._native.items()
        ):
            raise ObserverError("native session identity belongs to another actor")
        previous = {
            r.call_id
            for (owner, label), s in self._native.items()
            if owner == actor and label != task and s.session_id == session.session_id
            for r in s.records
            if r.kind == "tool_call"
        }
        self._native_calls[actor, task] = {
            r.call_id
            for r in session.records
            if r.kind == "tool_call" and r.call_id is not None and r.call_id not in previous
        }
        self._native[actor, task] = session
        self.trace.append(
            {
                "kind": "native_session",
                "actor": actor,
                "task": task,
                "session_id": session.session_id,
                "path": str(path),
                "file_digest": session.file_digest,
                "file_bytes": session.file_bytes,
                "complete_bytes": session.complete_bytes,
                "partial_tail_bytes": session.partial_tail_bytes,
                "runtime_label": session.runtime_label,
                "model": session.model,
                "records": len(session.records),
                "parent_session": session.parent_session,
                "source_aliases": session.source_aliases,
            }  # fmt: skip
        )
        for record in session.records:
            self.trace.append(
                {
                    "kind": "native_record",
                    "actor": actor,
                    "session_id": session.session_id,
                    "native_seq": record.seq,
                    "rkind": record.kind,
                    "record_id": record.record_id,
                    "origin_id": record.origin_id,
                    "call_id": record.call_id,
                    "name": record.name,
                    "ts_ms": record.ts_ms,
                    "is_error": record.is_error,
                    "chars": record.chars,
                    "digest": record.digest,
                    "usage": dict(record.usage),
                    "cost_nano_usd": record.cost_nano_usd,
                    "blob": self.blob(record.text) if record.text else None,
                }  # fmt: skip
            )
        return session

    def ingest_spool(
        self,
        spool: Path,
        spool_blobs: Path,
        *,
        expected_actor: str,
        expected_task: str | None,
        native_association: NativeSession | Mapping[str | None, NativeSession] | None = None,
    ) -> int:
        """Validate an entire untrusted append-only spool before copying its new prefix.

        Associations must be snapshots registered by this supervisor for the supplied
        actor/task. Missing independent evidence is permitted only as unknown evidence.
        """
        if expected_actor not in ("solver-a", "solver-b"):
            raise ObserverError("spool actor must be a solver")
        associations = (
            dict(native_association)
            if isinstance(native_association, Mapping)
            else {expected_task: native_association}
        )
        for task, session in associations.items():
            if session is not None and self._native.get((expected_actor, task)) is not session:
                raise ObserverError("native association is not registered for this actor/task")
        if not spool.exists() and not spool.is_symlink():
            return 0
        records = _read_trace_bytes(read_owned(spool.parent, spool.name, limit=MAX_SOURCE_BYTES))
        done, head = self._ingested.get(str(spool), (-1, GENESIS))
        if done >= len(records) or (done >= 0 and records[done]["hash"] != head):
            raise ObserverError("previously accepted spool prefix changed")
        validate_spool(
            records, run_id=self.run_id, actor=expected_actor, tasks=set(associations), done=done
        )
        new = records[done + 1 :]
        existing = {e["call_id"] for e in read_trace(self.trace.path) if e["kind"] == "dispatch"}
        if any(r["kind"] == "dispatch" and r["call_id"] in existing for r in new):
            raise ObserverError("solver call id collides with accepted evidence")
        contents = [record_blobs(record, spool_blobs, MAX_BLOB_BYTES) for record in new]
        completions = {
            record["call_id"]: (record, blobs.get("blob", ""))
            for record, blobs in zip(new, contents, strict=True)
            if record["kind"] == "result"
        }
        for record, blobs in zip(new, contents, strict=True):
            payload = {k: v for k, v in record.items() if k not in ("seq", "prev", "hash")}
            payload.update(via="spool", spool_seq=record["seq"])
            if record["kind"] == "dispatch":
                native = associations[record["task"]]
                payload["native_session_id"] = native.session_id if native is not None else None
                allowed = self._native_calls.get((expected_actor, record["task"]), set())
                result, stdout = completions.get(record["call_id"], (None, ""))
                payload["native_call_id"] = _associated_call(
                    record, native, allowed, result, stdout
                )
            for text in blobs.values():
                self.blob(text)
            self.trace.append(payload)
        if records:
            self._ingested[str(spool)] = (len(records) - 1, records[-1]["hash"])
        return len(new)

    def attest_supervisor_spool(
        self,
        spool: Path,
        spool_blobs: Path,
        *,
        expected_actor: str,
        expected_tasks: set[str | None],
    ) -> int:
        """Record broker completion independently of the runtime's rendered exposure.

        The reserved subtree must be outside any untrusted solver's access; only a
        supervisor-controlled producer may write it. Solver-side records cannot opt in.
        Receipts append evidence; original native files and earlier trace records stay intact.
        """
        protected = self.dir / expected_actor / "spool.ndjson"
        if (
            expected_actor not in ("solver-a", "solver-b")
            or not expected_tasks
            or spool.absolute() != protected.absolute()
            or spool_blobs.absolute() != _spool_blob_dir(protected).absolute()
        ):
            raise ObserverError("completion attestation requires the protected actor spool")
        records = _read_trace_bytes(read_owned(spool.parent, spool.name, limit=MAX_SOURCE_BYTES))
        validate_spool(
            records, run_id=self.run_id, actor=expected_actor, tasks=expected_tasks, done=-1
        )
        for record in records:
            record_blobs(record, spool_blobs, MAX_BLOB_BYTES)
        receipts = supervisor_receipts(records, read_trace(self.trace.path))
        for receipt in receipts:
            self.trace.append(receipt)
        return len(receipts)

    def finalize(self) -> int:
        """Dispatches without any completion evidence become ``unknown`` (never inferred done)."""
        events = read_trace(self.trace.path)
        done = {e["call_id"] for e in events if e["kind"] == "result"}
        pending = [e for e in events if e["kind"] == "dispatch" and e["call_id"] not in done]
        for event in pending:
            self.trace.append(
                {
                    "kind": "result",
                    "call_id": event["call_id"],
                    "state": "unknown",
                    "exit_code": None,
                    "reason": "no_completion_evidence",
                    "stdout_chars": 0,
                    "stdout_bytes": 0,
                    "stdout_digest": None,
                    "blob": None,
                    "duration_ns": None,
                }  # fmt: skip
            )
        return len(pending)

    def freeze(self) -> Frozen:
        return freeze(self.trace.path, self.blobs)


# ------------------------------------------------------------------------------- records


def dispatch_record(
    call_id: str,
    actor: str,
    task: str | None,
    tool: str,
    argv: Sequence[str],
    cwd: str,
    native_call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "dispatch", "call_id": call_id, "actor": actor, "task": task, "tool": tool,
        "argv": [a[:MAX_ARGV_CHARS] for a in argv], "cwd": cwd, "state": "attempted",
        "native_call_id": native_call_id,
    }  # fmt: skip


def result_record(
    call_id: str,
    state: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    duration_ns: int | None,
    store_blob,  # noqa: ANN001 - callable(text) -> digest
    **extra: Any,
) -> dict[str, Any]:
    if state not in STATES:
        raise ObserverError(f"unknown state {state!r}")
    return {
        "kind": "result", "call_id": call_id, "state": state, "exit_code": exit_code,
        "stdout_chars": len(stdout), "stdout_bytes": len(stdout.encode()),
        "stdout_digest": sha(stdout), "blob": store_blob(stdout) if stdout else None,
        "stderr_chars": len(stderr), "stderr_digest": sha(stderr) if stderr else None,
        "duration_ns": duration_ns, **extra,
    }  # fmt: skip


def freeze_calls(events: Sequence[Mapping[str, Any]], blobs: Path) -> list[ObservedCall]:
    # Freeze is a second trust boundary: validate every locator, including SQL and native
    # records that may not be consumed by the particular call projection below.
    for event in events:
        record_blobs(event, blobs, MAX_BLOB_BYTES)
    results = {e["call_id"]: e for e in events if e["kind"] == "result"}
    sql: dict[str, list[Mapping[str, Any]]] = {}
    for e in events:
        if e["kind"] == "sql":
            sql.setdefault(e["call_id"], []).append(e)
    calls: list[ObservedCall] = []
    for e in events:
        if e["kind"] != "dispatch":
            continue
        res = results.get(e["call_id"])
        text = ""
        if res and res.get("blob"):
            text = read_blob(blobs, res["blob"], limit=MAX_BLOB_BYTES)
        calls.append(
            ObservedCall(
                e["call_id"],
                e["actor"],
                e.get("task"),
                e["tool"],
                tuple(e["argv"]),
                (
                    "unknown"
                    if e.get("via") == "spool" and res
                    else res["state"]
                    if res
                    else "attempted"
                ),
                res.get("exit_code") if res and e.get("via") != "spool" else None,
                text,
                res["stdout_chars"] if res else 0,
                res["stdout_bytes"] if res else 0,
                res.get("duration_ns") if res else None,
                e["seq"],
                tuple(sql.get(e["call_id"], ())),
                e.get("wall_ns", 0),
                e.get("native_call_id"),
                e.get("via") == "spool",
                e.get("native_session_id"),
            )  # fmt: skip
        )
    return calls


LINK_SLACK_MS = 250


def link_native(
    calls: Sequence[ObservedCall],
    sessions: Mapping[str, tuple[str, str | None, NativeSession]],
) -> tuple[ObservedCall, ...]:
    """Join each solver dispatch to the native tool_call whose execution window holds it.

    A window is ``[call ts, result ts]`` (with slack for coarse native timestamps) in the
    actor's own session. A dispatch inside no window (or inside several) stays unlinked: the
    oracle then reports its exposure as unobserved instead of guessing.
    """
    from dataclasses import replace

    windows: dict[str, list[tuple[int, int, str]]] = {}
    for session_id, (actor, _task, native) in sessions.items():
        by_id = {r.call_id: r for r in native.records if r.kind == "tool_result"}
        for r in native.records:
            if r.kind != "tool_call" or r.ts_ms is None or r.call_id is None:
                continue
            end = by_id.get(r.call_id)
            stop = end.ts_ms if end is not None and end.ts_ms is not None else r.ts_ms
            windows.setdefault(actor, []).append(
                (r.ts_ms - LINK_SLACK_MS, stop + LINK_SLACK_MS, r.call_id)
            )
            _ = session_id
    out = []
    for call in calls:
        found = [
            cid
            for lo, hi, cid in windows.get(call.actor, ())
            if lo <= call.wall_ns // 1_000_000 <= hi
        ]
        if not call.via_spool and call.native_call_id is None and len(found) == 1:
            call = replace(call, native_call_id=found[0])
        out.append(call)
    return tuple(out)


def freeze(trace_path: Path, blobs: Path) -> Frozen:
    """Frozen evidence; the LATEST native snapshot of each session wins (re-recording is safe)."""
    events = read_trace(trace_path)
    heads: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    current: str | None = None
    for e in events:
        if e["kind"] == "native_session":
            current = e["session_id"]
            heads[current] = e
            rows[current] = []
        elif e["kind"] == "native_record" and current is not None and current == e["session_id"]:
            rows[current].append(e)
        else:
            current = None
    sessions: dict[str, tuple[str, str | None, NativeSession]] = {}
    for session_id, e in heads.items():
        records = []
        for r in rows[session_id]:
            text = read_blob(blobs, r["blob"], limit=MAX_BLOB_BYTES) if r.get("blob") else ""
            records.append(
                NativeRecord(
                    r["native_seq"],
                    r["rkind"],
                    r["record_id"],
                    text,
                    r.get("call_id"),
                    r.get("name"),
                    r.get("ts_ms"),
                    r.get("is_error"),
                    r.get("usage") or {},
                    r.get("cost_nano_usd"),
                    r.get("origin_id"),
                )  # fmt: skip
            )
        sessions[session_id] = (
            e["actor"], e.get("task"),
            NativeSession(session_id, e["path"], None, e.get("runtime_label"), e.get("model"),
                          tuple(records), e["file_bytes"], e["complete_bytes"],
                          e["partial_tail_bytes"], e["file_digest"],
                          parent_session=e.get("parent_session"),
                          source_aliases=tuple(e.get("source_aliases", ()))),
        )  # fmt: skip
    return Frozen(
        events[0]["run_id"] if events else "",
        _verify_spool_calls(link_native(freeze_calls(events, blobs), sessions), sessions, events),
        sessions,
        tuple(e for e in events if e["kind"] == "phase"),
        events[-1]["hash"] if events else GENESIS,
    )


def _associated_call(
    dispatch: Mapping[str, Any],
    session: NativeSession | None,
    allowed: set[str],
    result: Mapping[str, Any] | None,
    stdout: str,
    *,
    supervisor_completion: bool = False,
) -> str | None:
    if session is None or result is None:
        return None
    explicit = dispatch.get("native_call_id")
    ends: dict[str | None, list[NativeRecord]] = {}
    for record in session.records:
        if record.kind == "tool_result":
            ends.setdefault(record.call_id, []).append(record)
    stamp = dispatch["wall_ns"] // 1_000_000
    found = []
    for record in session.records:
        if (
            record.kind != "tool_call"
            or record.call_id not in allowed
            or (explicit is not None and explicit != record.call_id)
        ):
            continue
        completions = ends.get(record.call_id, [])
        if len(completions) != 1:
            continue
        end = completions[0]
        if (
            not native_invokes(record.text, dispatch["argv"])
            or not native_output_matches(
                stdout,
                end.text,
                supervisor_completion=supervisor_completion,
                stderr_chars=result.get("stderr_chars", 0),
                stderr_digest=result.get("stderr_digest"),
            )
            or end.seq <= record.seq
            or end.is_error is None
            or (result.get("exit_code") == 0) != (not end.is_error)
        ):
            continue
        if explicit is None and (
            record.ts_ms is None
            or end.ts_ms is None
            or not record.ts_ms - LINK_SLACK_MS <= stamp <= end.ts_ms + LINK_SLACK_MS
        ):
            continue
        found.append(record.call_id)
    return found[0] if len(found) == 1 else None


def _native_snapshot_calls(
    events: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str | None, str], set[str]]:
    """Recover supervisor task boundaries for read-only replay of retained traces."""
    snapshots: dict[tuple[str, str | None, str], set[str]] = {}
    allowed: dict[tuple[str, str | None, str], set[str]] = {}
    active: tuple[str, str | None, str] | None = None
    previous: set[str] = set()
    for event in events:
        if event["kind"] == "native_session":
            active = (event["actor"], event.get("task"), event["session_id"])
            previous = {
                cid
                for (actor, task, sid), ids in snapshots.items()
                if actor == active[0] and sid == active[2] and task != active[1]
                for cid in ids
            }
            snapshots[active] = set()
            allowed[active] = set()
        elif event["kind"] == "native_record" and active is not None:
            if event.get("session_id") != active[2] or event.get("actor") != active[0]:
                active = None
            elif event.get("rkind") == "tool_call" and event.get("call_id") is not None:
                cid = event["call_id"]
                snapshots[active].add(cid)
                if cid not in previous:
                    allowed[active].add(cid)
        else:
            active = None
    return allowed


def _verify_spool_calls(
    calls: Sequence[ObservedCall],
    sessions: Mapping[str, tuple[str, str | None, NativeSession]],
    events: Sequence[Mapping[str, Any]],
) -> tuple[ObservedCall, ...]:
    """Separate independently attested execution from the native visible projection."""
    from dataclasses import replace

    results = {e["call_id"]: e for e in events if e["kind"] == "result"}
    attestations = {e["call_id"]: e for e in events if e["kind"] == "supervisor_completion"}
    allowed = _native_snapshot_calls(events)
    out = []
    consumed: dict[tuple[str, str], int] = {}
    for call in calls:
        if not call.via_spool:
            out.append(call)
            continue
        result = results.get(call.call_id)
        receipt = attestations.get(call.call_id)
        trusted = (
            result is not None
            and receipt is not None
            and receipt.get("actor") == call.actor
            and receipt.get("task") == call.task
            and receipt.get("run_id") == result.get("run_id")
            and receipt.get("stdout_digest") == sha(call.stdout) == result.get("stdout_digest")
            and receipt.get("state") == result.get("state")
            and receipt.get("exit_code") == result.get("exit_code")
        )
        entry = sessions.get(call.native_session_id or "")
        if entry is not None and entry[0] == call.actor:
            dispatch = {
                "native_call_id": call.native_call_id,
                "wall_ns": call.wall_ns,
                "argv": call.argv,
            }
            linked = _associated_call(
                dispatch,
                entry[2],
                allowed.get((call.actor, call.task, call.native_session_id or ""), set()),
                result,
                call.stdout,
                supervisor_completion=trusted,
            )
            call = replace(call, native_call_id=linked)
        else:
            call = replace(call, native_call_id=None)
        if trusted and result is not None:
            # Raw bytes remain execution evidence only. The oracle reads native result.text
            # for exposure; an unlinked or truncated native result never exposes hidden bytes.
            out.append(replace(call, state=result["state"], exit_code=result.get("exit_code")))
            continue
        native_call = native_result = None
        if entry is not None and entry[0] == call.actor:
            candidates = [
                r
                for r in entry[2].records
                if r.kind == "tool_call" and r.call_id == call.native_call_id
            ]
            completions = [
                r
                for r in entry[2].records
                if r.kind == "tool_result" and r.call_id == call.native_call_id
            ]
            if len(candidates) == len(completions) == 1:
                native_call, native_result = candidates[0], completions[0]
        key = (call.native_session_id or "", call.native_call_id or "")
        cursor = consumed.get(key, 0)
        proof = call.stdout
        if not proof and native_result is not None and result is not None:
            stderr = native_result.text[: result.get("stderr_chars", 0)]
            if stderr and sha(stderr) == result.get("stderr_digest"):
                proof = stderr
        match = -1 if native_result is None else native_result.text.find(proof, cursor)
        # Empty stdout is not absence of evidence: an exact stderr digest corroborates
        # a native failure. Neither unobserved stream content nor command mentions suffice.
        if (
            native_call is not None
            and native_result is not None
            and result is not None
            and proof
            and match >= 0
            and native_invokes(native_call.text, call.argv)
            and native_result.seq > native_call.seq
            and native_result.is_error is not None
            and (result.get("exit_code") == 0) == (not native_result.is_error)
        ):
            consumed[key] = match + len(proof)
            call = replace(call, state=result["state"], exit_code=result.get("exit_code"))
        else:
            uncertain_sql = tuple(
                {**row, "txn": "unknown", "exit_code": None}
                if row.get("phase") == "finish"
                else row
                for row in call.sql
            )
            call = replace(call, native_call_id=None, sql=uncertain_sql)
        out.append(call)
    return tuple(out)


# ------------------------------------------------------------------------------ the shim


def _task_of(env: Mapping[str, str]) -> str | None:
    path = env.get(ENV_TASK_FILE)
    try:
        return Path(path).read_text().strip() or None if path else None
    except OSError:
        return None


def _spool_blob_dir(spool: Path) -> Path:
    return spool.parent / (spool.stem + "-blobs")


def _spool_blob(spool: Path):  # noqa: ANN202
    directory = _spool_blob_dir(spool)

    def store(text: str) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        if len(text.encode()) > MAX_BLOB_BYTES:
            raise ObserverError("observer output exceeds blob bound")
        digest = sha(text)
        target = directory / digest
        if not target.exists():
            target.write_bytes(text.encode()[:MAX_BLOB_BYTES])
        return digest

    return store


def _sql_outcome(script: str, returncode: int, stderr: str) -> tuple[str, str | None]:
    """(txn, sqlstate): committed / rolled_back / autocommit / error / unknown."""
    found = SQLSTATE.search(stderr)
    state = found.group(1) if found else None
    text = script.upper()
    explicit = bool(re.search(r"\bBEGIN\b", text))
    if returncode == 0:
        if explicit and re.search(r"\bROLLBACK\b", text):
            return "rolled_back", state
        return ("committed" if explicit and "COMMIT" in text else "autocommit"
                if not explicit else "unknown"), state  # fmt: skip
    if explicit:
        return "rolled_back", state  # psql ON_ERROR_STOP ends the session: open txn aborts
    return "error", state


def shim_main(
    tool: str,
    args: Sequence[str],
    env: Mapping[str, str] | None = None,
    *,
    group_owner: bool = False,
    timeout: float = 180,
    cancel_event=None,
) -> int:
    """Bounded real tool capture; bridge descendants stay in its supervisor-owned session."""
    from .bounded_capture import capture_process, decode_text, text_encoding

    if group_owner and (os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid()):
        raise ObserverError("bridge shim is not the owned session leader")
    env = dict(os.environ if env is None else env)
    spool = Path(env[ENV_SPOOL])
    actor, run_id = env[ENV_ACTOR], env.get(ENV_RUN, "")
    trace = Trace(spool, run_id)
    store = _spool_blob(spool)
    call_id = f"{actor}-{secrets.token_hex(5)}"
    task = _task_of(env)
    stdin_text: str | None = None
    if tool == "sql":
        argv, stdin_text = _sql_argv(args, env), sys.stdin.read(128_001)
        if len(stdin_text.encode()) > 128_000:
            raise ObserverError("tool input exceeds request bound")
    else:
        argv = [env[ENV_REAL + tool.upper()], *args]
    shown = [tool, *args]
    trace.append(
        dispatch_record(call_id, actor, task, tool, shown, os.getcwd(), env.get(ENV_NATIVE_CALL))
    )
    if stdin_text is not None:
        trace.append(
            {
                "kind": "sql",
                "call_id": call_id,
                "phase": "start",
                "role": _flag(args, "--role"),
                "database": _flag(args, "--database"),
                "statement_digest": sha(stdin_text),
                "statement_blob": store(stdin_text),
                "statement_chars": len(stdin_text),
            }  # fmt: skip
        )
    started = time.monotonic_ns()
    capture_dir = (
        Path(env["LAB_CAPTURE_DIR"]) if group_owner else spool.parent / "shim-attempts" / call_id
    )

    failure_recorded = False

    def record_failure(captured, reason=None, *, observation="final") -> None:
        nonlocal failure_recorded
        reason = reason or captured.status
        metadata = {
            **captured.metadata(),
            "capture_status": reason,
            "termination_cause": reason,
            "capture_complete": False,
            "capture_observation": observation,
        }
        trace.append(
            result_record(
                call_id,
                "interrupted",
                captured.returncode,
                "",
                f"restricted tool bridge: {reason}",
                time.monotonic_ns() - started,
                store,
                reason=reason,
                capture=metadata,
            )
        )
        if stdin_text is not None:
            trace.append(
                {
                    "kind": "sql",
                    "call_id": call_id,
                    "phase": "finish",
                    "txn": "unknown",
                    "sqlstate": None,
                    "exit_code": captured.returncode,
                }
            )
        if group_owner:
            fd = os.open(
                capture_dir / "tool-capture.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "w") as handle:
                json.dump(metadata, handle)
            marker = capture_dir / "tool-capture-failed"
            pending = marker.with_suffix(".pending")
            fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(reason.encode("ascii"))
            os.replace(pending, marker)
        failure_recorded = True

    def interrupted(signum: int, _frame: Any) -> None:
        trace.append(
            result_record(
                call_id,
                "interrupted",
                None,
                "",
                "",
                time.monotonic_ns() - started,
                store,
                reason=f"signal:{signum}",
            )
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        proc = capture_process(
            argv,
            input_bytes=stdin_text.encode(text_encoding()) if stdin_text is not None else None,
            env=_child_env(tool, env),
            cwd=Path.cwd(),
            timeout=timeout,
            cancel_event=cancel_event,
            stdout_limit=MAX_BLOB_BYTES,
            stderr_limit=MAX_BLOB_BYTES,
            output_dir=capture_dir / "tool",
            start_new_session=not group_owner,
            on_output_limit=lambda captured: record_failure(
                captured, observation="termination_snapshot"
            ),
        )
    except OSError as exc:
        trace.append(result_record(call_id, "unknown", None, "", str(exc), None, store,
                                   reason=type(exc).__name__))  # fmt: skip
        print(f"{tool}: cannot run: {exc}", file=sys.stderr)
        return 127
    if proc.status != "completed":
        if not failure_recorded:
            record_failure(proc)
        print(f"restricted tool bridge: {proc.status}", file=sys.stderr)
        return proc.returncode or 1
    assert proc.returncode is not None  # Completed captures have a reaped child.
    try:
        stdout = decode_text(proc.stdout)
        stderr = decode_text(proc.stderr)
    except UnicodeDecodeError:
        record_failure(proc, "invalid_output_encoding")
        print("restricted tool bridge: invalid_output_encoding", file=sys.stderr)
        return proc.returncode or 1
    state = "verified" if proc.returncode >= 0 else "interrupted"
    trace.append(
        result_record(
            call_id,
            state,
            proc.returncode,
            stdout,
            stderr,
            time.monotonic_ns() - started,
            store,
            capture=proc.metadata(),
        )  # fmt: skip
    )
    if stdin_text is not None:
        txn, sqlstate = _sql_outcome(stdin_text, proc.returncode, stderr)
        trace.append({"kind": "sql", "call_id": call_id, "phase": "finish", "txn": txn,
                      "sqlstate": sqlstate, "exit_code": proc.returncode})  # fmt: skip
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    sys.stdout.flush()
    return proc.returncode


def _flag(args: Sequence[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


def _sql_argv(args: Sequence[str], env: Mapping[str, str]) -> list[str]:
    role, database = _flag(args, "--role"), _flag(args, "--database")
    if role not in SQL_USERS or not database or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", database):
        raise SystemExit("usage: sql --role reader|writer --database <name>  (SQL on stdin)")
    return [
        env[ENV_REAL + "PSQL"], "-X", "-q", "-A", "-F", "\t",
        "-P", "footer=off", "-P", "pager=off", "-v", "ON_ERROR_STOP=1",
        "-v", "VERBOSITY=verbose",
        "-h", env[ENV_SQL_HOST], "-p", env[ENV_SQL_PORT], "-U", SQL_USERS[role], "-d", database,
        "-f", "-",
    ]  # fmt: skip


def _child_env(tool: str, env: Mapping[str, str]) -> dict[str, str]:
    child = {k: v for k, v in env.items() if not k.startswith("LAB_")}
    if tool == "sql":
        child["PGPASSFILE"] = env[ENV_SQL_PGPASS]
        child["PGCONNECT_TIMEOUT"] = "10"
    return child


SHIM_SCRIPT = """#!/bin/sh
# Lab observer shim ({tool}): record dispatch/result, then pass through.
PYTHONPATH={repo}${{PYTHONPATH:+:$PYTHONPATH}} exec {python} \\
  -m eval.accounting_airline.observer shim {tool} "$@"
"""


def install_shims(directory: Path, *, python: str, repo_root: Path) -> dict[str, Path]:
    """Executable ``ssgrep`` and ``sql`` wrappers that route a solver's calls through the shim."""
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for tool in ("ssgrep", "sql"):
        target = directory / tool
        target.write_text(SHIM_SCRIPT.format(tool=tool, repo=str(repo_root), python=python))
        target.chmod(0o755)
        out[tool] = target
    return out


def write_pgpass(path: Path, *, host: str, port: int, passwords: Mapping[str, str]) -> None:
    """A psql password file by reference (0600); only the named lab roles appear in it."""
    lines = [f"{host}:{port}:*:{SQL_USERS[k]}:{v}\n" for k, v in sorted(passwords.items())]
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.writelines(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2 and args[0] == "shim":
        return shim_main(args[1], args[2:])
    print(
        "usage: python -m eval.accounting_airline.observer shim <ssgrep|sql> [args...]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())


def parse_psql_table(text: str) -> list[dict[str, str]] | None:
    """Rows of ONE unaligned psql result (header line + tab-separated rows); else None.

    The ``sql`` tool prints exactly this shape. Values stay strings: callers convert the
    columns they know are integers. A width mismatch (command tags, several statements,
    error text) returns None rather than a guess.
    """
    lines = [ln for ln in text.splitlines() if ln != ""]
    if not lines:
        return None
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(header):
            return None
        rows.append(dict(zip(header, cells, strict=True)))
    return rows
