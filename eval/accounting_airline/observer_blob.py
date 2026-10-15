"""Bounded, descriptor-relative reads at the solver-to-verifier boundary.

A digest authenticates bytes, not who produced them. Caller-supplied actor/task/native
bindings are supervisor policy; no field in a solver record can grant that authority.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .observer_capture import valid_capture

MAX_SOURCE_BYTES = 32_000_000
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ObserverError(RuntimeError):
    """Untrusted observer input failed its boundary checks."""


def _directory(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ObserverError("observer directory has unsafe type, owner or permissions")


def read_owned(root: Path, relative: str, *, limit: int) -> bytes:
    """Open beneath a supervisor-selected root; never follow record-selected links."""
    parts = relative.split("/")
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ObserverError("observer locator is not a safe relative path")
    directory = None
    fd = None
    try:
        absolute = root.absolute()
        if ".." in absolute.parts:
            raise ObserverError("observer root is not canonical")
        directory = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in absolute.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        _directory(directory)
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
            _directory(directory)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_mode & 0o022
            or before.st_size > limit
        ):
            raise ObserverError("observer file has unsafe type, owner, links, permissions or size")
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if not remaining or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ObserverError("observer file changed or exceeded its read bound")
        return b"".join(chunks)
    except OSError as exc:
        raise ObserverError("observer file could not be safely opened") from exc
    finally:
        if fd is not None:
            os.close(fd)
        if directory is not None:
            os.close(directory)


def read_blob(root: Path, digest: Any, *, limit: int) -> str:
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ObserverError("observer blob must be an exact lowercase SHA256 basename")
    data = read_owned(root, digest, limit=limit)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ObserverError("observer blob digest mismatch")
    try:
        return data.decode("utf-8")
    except UnicodeError as exc:
        raise ObserverError("observer blob is not UTF-8") from exc


def read_source(path: Path, source_root: Path) -> bytes:
    # Do not resolve the untrusted path: that would erase evidence of a redirect.
    try:
        relative = path.absolute().relative_to(source_root.absolute()).as_posix()
    except ValueError as exc:
        raise ObserverError("native source is outside its supervisor-owned root") from exc
    return read_owned(source_root, relative, limit=MAX_SOURCE_BYTES)


COMMON = {"kind", "seq", "prev", "hash", "ts_utc", "wall_ns", "run_id"}
FIELDS = {
    "dispatch": {"call_id", "actor", "task", "tool", "argv", "cwd", "state", "native_call_id"},
    "result": {
        "call_id",
        "state",
        "exit_code",
        "stdout_chars",
        "stdout_bytes",
        "stdout_digest",
        "blob",
        "stderr_chars",
        "stderr_digest",
        "duration_ns",
        "reason",
        "capture",
    },
    "sql": {
        "call_id",
        "phase",
        "role",
        "database",
        "statement_digest",
        "statement_blob",
        "statement_chars",
        "txn",
        "sqlstate",
        "exit_code",
    },
}


def _integer(value: Any, *, nullable: bool = False, signed: bool = False) -> bool:
    return (nullable and value is None) or (type(value) is int and (signed or value >= 0))


def _string(value: Any, *, nullable: bool = False) -> bool:
    return (nullable and value is None) or (isinstance(value, str) and len(value) <= 4000)


def validate_spool(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    actor: str,
    tasks: set[str | None],
    done: int,
) -> None:
    """Validate the complete chain's grammar before copying any record or blob."""
    dispatches: dict[str, Mapping[str, Any]] = {}
    completed: set[str] = set()
    sql_phases: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        kind = record.get("kind")
        if not isinstance(kind, str) or kind not in FIELDS or set(record) - COMMON - FIELDS[kind]:
            raise ObserverError("unsupported solver spool record schema")
        if (
            type(record.get("seq")) is not int
            or record["seq"] != index
            or record.get("run_id") != run_id
            or not _integer(record.get("wall_ns"))
            or not _string(record.get("ts_utc"))
        ):
            raise ObserverError("solver spool run or sequence binding mismatch")
        cid = record.get("call_id")
        if not _string(cid) or not cid:
            raise ObserverError("invalid solver call id")
        if kind == "dispatch":
            if not FIELDS["dispatch"] <= set(record):
                raise ObserverError("solver dispatch is missing required fields")
            argv = record.get("argv")
            if (
                cid in dispatches
                or record.get("actor") != actor
                or not _string(record.get("task"), nullable=True)
                or (index > done and record.get("task") not in tasks)
                or record.get("tool") not in ("ssgrep", "sql")
                or record.get("state") != "attempted"
                or not _string(record.get("native_call_id"), nullable=True)
                or not _string(record.get("cwd"))
                or not isinstance(argv, list)
                or not argv
                or len(argv) > 1024
                or not all(_string(a) for a in argv)
                or argv[0] != record["tool"]
            ):
                raise ObserverError("solver dispatch actor/task/schema mismatch")
            dispatches[cid] = record
        elif cid not in dispatches:
            raise ObserverError("solver result has no dispatch in this spool")
        elif kind == "result":
            # Native-only observations and start/signal failures may lack process measurements.
            if not (FIELDS["result"] - {"reason", "capture"}) <= set(record):
                raise ObserverError("solver result is missing required fields")
            stderr_digest = record["stderr_digest"]
            if stderr_digest is not None and (
                not isinstance(stderr_digest, str) or not DIGEST.fullmatch(stderr_digest)
            ):
                raise ObserverError("invalid solver stderr digest")
            if (
                cid in completed
                or record.get("state") not in ("verified", "interrupted", "unknown")
                or not _integer(record.get("exit_code"), nullable=True, signed=True)
                or not _integer(record.get("duration_ns"), nullable=True)
                or not all(
                    _integer(record.get(key))
                    for key in ("stdout_chars", "stdout_bytes", "stderr_chars")
                )
                or not _string(record.get("reason"), nullable=True)
                or not isinstance(record.get("stdout_digest"), str)
                or not DIGEST.fullmatch(record["stdout_digest"])
                or (record["state"] == "verified" and record.get("exit_code") is None)
            ):
                raise ObserverError("invalid solver result schema")
            if "capture" in record and not valid_capture(record["capture"], record):
                raise ObserverError("invalid solver capture metadata")
            completed.add(cid)
        else:
            phase = record.get("phase")
            if (
                dispatches[cid]["tool"] != "sql"
                or phase not in ("start", "finish")
                or (cid, phase) in sql_phases
            ):
                raise ObserverError("invalid solver SQL association")
            sql_phases.add((cid, phase))
            if phase == "start":
                if set(record) - COMMON != {
                    "call_id",
                    "phase",
                    "role",
                    "database",
                    "statement_digest",
                    "statement_blob",
                    "statement_chars",
                }:
                    raise ObserverError("invalid solver SQL start fields")
                if (
                    record.get("role") not in ("reader", "writer")
                    or not _string(record.get("database"))
                    or not _integer(record.get("statement_chars"))
                ):
                    raise ObserverError("invalid solver SQL start schema")
            else:
                if set(record) - COMMON != {"call_id", "phase", "txn", "sqlstate", "exit_code"}:
                    raise ObserverError("invalid solver SQL completion fields")
                if (
                    record.get("txn")
                    not in ("committed", "rolled_back", "autocommit", "error", "unknown")
                    or not _integer(record.get("exit_code"), signed=True)
                    or not _string(record.get("sqlstate"), nullable=True)
                ):
                    raise ObserverError("invalid solver SQL completion schema")


def record_blobs(record: Mapping[str, Any], root: Path, limit: int) -> dict[str, str]:
    """Validate every supported blob locator, including native/SQL snapshot content."""
    contents = {}
    for key in ("blob", "statement_blob"):
        digest = record.get(key)
        if digest is not None:
            contents[key] = read_blob(root, digest, limit=limit)
    if record.get("kind") == "result":
        text = contents.get("blob", "")
        if (
            record.get("stdout_digest") not in (None, hashlib.sha256(text.encode()).hexdigest())
            or record.get("stdout_chars") != len(text)
            or record.get("stdout_bytes") != len(text.encode())
        ):
            raise ObserverError("solver result content does not match declared output")
    if "statement_blob" in contents:
        text = contents["statement_blob"]
        if record.get("statement_digest") != hashlib.sha256(
            text.encode()
        ).hexdigest() or record.get("statement_chars") != len(text):
            raise ObserverError("solver SQL content does not match statement digest")
    return contents


def native_invokes(command: str, argv: Sequence[str]) -> bool:
    """Recognize executable positions, not tool names quoted as output or arguments.

    Shell indirection/definitions are intentionally unverifiable here. This is not a shell
    interpreter and never executes the command to decide what it meant.
    """
    try:
        lexer = shlex.shlex(command, posix=False, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    if any(t in ("function", "alias", "eval", "source", "{", "}") for t in tokens):
        return False
    if tokens[:1] == ["for"]:
        return _literal_loop_invokes(tokens, argv)
    executable = True
    for index, token in enumerate(tokens):
        if token in (";", "&&", "||", "|", "("):
            executable = True
        elif executable:
            if token in ("do", "then", "else"):
                continue
            # Literal environment assignments precede the executable; quoted
            # text/expansions are not independently verifiable assignment values.
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9._:/-]*", token):
                continue
            if argv and Path(token).name == argv[0]:
                arguments = tokens[index + 1 : index + len(argv)]
                try:
                    decoded = [shlex.split(argument)[0] for argument in arguments]
                except (ValueError, IndexError):
                    return False
                if decoded == list(argv[1:]):
                    return True
            executable = False
    return False


def _literal_loop_invokes(tokens: list[str], argv: Sequence[str]) -> bool:
    """Recognize only a finite literal for-list with one unmodified command body."""
    if (
        len(tokens) < 9
        or not argv
        or tokens[2] != "in"
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tokens[1])
        or tokens[-2:] != [";", "done"]
    ):
        return False
    try:
        split = tokens.index(";")
        if split < 4 or tokens[split + 1] != "do":
            return False
        body = tokens[split + 2 : -2]
        if len(body) != len(argv) or Path(body[0]).name != argv[0]:
            return False
        values = [shlex.split(token)[0] for token in tokens[3:split]]
        if any(any(char in value for char in "$`") for value in values):
            return False
        arguments = [shlex.split(token)[0] for token in body[1:]]
    except (ValueError, IndexError):
        return False
    variables = {f"${tokens[1]}", "${" + tokens[1] + "}"}
    for value in values:
        expanded = [
            value if arg in variables and not token.startswith("'") else arg
            for token, arg in zip(body[1:], arguments, strict=True)
        ]
        if expanded == list(argv[1:]):
            return True
    return False


def supervisor_receipts(
    records: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attest only protected completions identical to the already retained observations."""
    observed_dispatches = {e["call_id"]: e for e in events if e["kind"] == "dispatch"}
    observed_results = {e["call_id"]: e for e in events if e["kind"] == "result"}
    existing = {e["call_id"]: e for e in events if e["kind"] == "supervisor_completion"}
    results = {r["call_id"]: r for r in records if r["kind"] == "result"}
    receipts = []
    metadata = {"run_id", "wall_ns", "ts_utc"}
    dispatch_fields = (FIELDS["dispatch"] - {"native_call_id"}) | metadata
    result_fields = FIELDS["result"] | metadata
    for dispatch in records:
        if dispatch["kind"] != "dispatch":
            continue
        cid = dispatch["call_id"]
        observed = observed_dispatches.get(cid)
        if (
            observed is None
            or observed.get("via") != "spool"
            or any(dispatch.get(key) != observed.get(key) for key in dispatch_fields)
        ):
            raise ObserverError("protected dispatch differs from retained observation")
        result = results.get(cid)
        if result is None:
            continue
        observed_result = observed_results.get(cid)
        if (
            observed_result is None
            or observed_result.get("via") != "spool"
            or any(result.get(key) != observed_result.get(key) for key in result_fields)
        ):
            raise ObserverError("protected completion differs from retained observation")
        receipt = {
            "kind": "supervisor_completion",
            "call_id": cid,
            "actor": dispatch["actor"],
            "task": dispatch["task"],
            "state": result["state"],
            "exit_code": result["exit_code"],
            "stdout_digest": result["stdout_digest"],
            "run_id": dispatch["run_id"],
        }
        old = existing.get(cid)
        if old is not None:
            if any(old.get(key) != value for key, value in receipt.items()):
                raise ObserverError("protected completion conflicts with prior attestation")
        else:
            receipts.append({**receipt, "source_spool_head": records[-1]["hash"]})
    return receipts


OMP_TRUNCATED_LINE = re.compile(
    r"(?P<prefix>[^\n]+)…\n\n\nWall time: \d+(?:\.\d+)? seconds\n\n"
    r"\[Some lines truncated to 768 bytes\. Use artifact://[0-9]+:raw:1-1 "
    r"to read line 1 whole\]\n?"
)


def native_output_matches(
    stdout: str,
    rendered: str,
    *,
    supervisor_completion: bool,
    stderr_chars: int = 0,
    stderr_digest: str | None = None,
) -> bool:
    """Match actual visible bytes; a renderer artifact locator is never opened.

    Only independently attested broker completions may use OMP's partial-line projection
    for native linkage. This does not expose the omitted remainder to the oracle.
    """
    if stdout and stdout in rendered:
        return True
    if (
        not stdout
        and 0 < stderr_chars <= len(rendered)
        and hashlib.sha256(rendered[:stderr_chars].encode()).hexdigest() == stderr_digest
    ):
        return True
    if not supervisor_completion:
        return False
    if not stdout:
        return True
    truncated = OMP_TRUNCATED_LINE.fullmatch(rendered)
    if truncated is not None:
        prefix = truncated["prefix"]
        return (
            0 < len(prefix.encode()) <= 768
            and len(prefix) < len(stdout)
            and stdout.startswith(prefix)
        )
    # A supervised command may deliberately pipe through head. Strip only the known
    # visible wall-time footer; all remaining characters must be an exact raw prefix.
    prefix = re.sub(r"\n\nWall time: \d+(?:\.\d+)? seconds\Z", "", rendered)
    return bool(prefix) and len(prefix) < len(stdout) and stdout.startswith(prefix)
