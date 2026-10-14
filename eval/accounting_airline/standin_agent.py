"""SCRIPTED STAND-IN agent process (runtime label ``scripted-standin``).

This is NOT an LLM and NEVER counts toward the live gate. It is a deterministic process that
does exactly what a solver would have to do mechanically, with REAL effects:

* every tool call is a real shell command run through ``/bin/sh -c`` with the observer shims in
  front of ``ssgrep`` and ``sql`` (real PostgreSQL through the restricted lab roles, real
  ``.venv/bin/ssgrep`` search/show against the real index);
* its native session is a REAL omp-format JSONL (the record shapes the repo's readers decode),
  written record by record as events happen, with real timestamps and the real tool output;
* usage in the native records is SCRIPTED integers (``provider=scripted-standin``, no cost): it
  is a labelled stand-in for provider usage, never a measurement of a model.

Every artifact it writes carries ``labRuntime: scripted-standin`` in the session header.
The command text of a step is lab-authored (never transcript-derived), which is the same trust
the omp bash tool places in its own model.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LABEL = "scripted-standin"
MODEL = "scripted-standin-v1"
DEFAULT_TOOL_TIMEOUT = 240.0
ENV_TRAILER = "LAB_STANDIN_TRAILER"  # "omp" (default, faithful) | "none"


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _entry_id() -> str:
    return secrets.token_hex(4)


# ------------------------------------------------------------------------ native writer


class NativeWriter:
    """Append omp-format records to one session file (a later turn reopens the same file)."""

    def __init__(self, session_dir: Path, session_id: str, cwd: str, *, label: str = LABEL) -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(session_dir.glob(f"*_{session_id}.jsonl"))
        self.session_id = session_id
        self.parent: str | None = None
        self.calls = 0
        if existing:
            self.path = existing[0]
            self.continued = True
            self.parent = self._last_id()
        else:
            stamp = now_iso().replace(":", "-").replace(".", "-")
            self.path = session_dir / f"{stamp}_{session_id}.jsonl"
            self.continued = False
            self._append(
                {
                    "type": "session",
                    "version": 3,
                    "id": session_id,
                    "timestamp": now_iso(),
                    "cwd": cwd,
                    "labRuntime": label,
                    "provider": label,
                }  # fmt: skip
            )
            self._append(
                {
                    "type": "model_change",
                    "id": _entry_id(),
                    "parentId": None,
                    "timestamp": now_iso(),
                    "model": MODEL,
                }  # fmt: skip
            )

    def _last_id(self) -> str | None:
        last = None
        for line in self.path.read_text().splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "message":
                last = obj.get("id")
        return last

    def _append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("ab") as handle:
            handle.write(json.dumps(record).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _message(self, message: Mapping[str, Any]) -> str:
        entry = _entry_id()
        self._append(
            {
                "type": "message",
                "id": entry,
                "parentId": self.parent,
                "timestamp": now_iso(),
                "message": dict(message),
            }  # fmt: skip
        )
        self.parent = entry
        return entry

    def user(self, text: str) -> None:
        self._message({"role": "user", "content": [{"type": "text", "text": text}]})

    def assistant(
        self,
        *,
        text: str | None = None,
        calls: Sequence[tuple[str, str, Mapping[str, Any]]] = (),
        scripted_input: int = 0,
    ) -> None:
        """One assistant message: optional text plus tool calls ``(id, name, arguments)``."""
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for call_id, name, arguments in calls:
            content.append(
                {"type": "toolCall", "id": call_id, "name": name, "arguments": dict(arguments)}
            )
        out = sum(len(str(b.get("text") or b.get("arguments") or "")) for b in content) // 4
        usage = {"input": scripted_input, "output": out, "cacheRead": 0, "cacheWrite": 0,
                 "totalTokens": scripted_input + out}  # fmt: skip
        self._message(
            {
                "role": "assistant",
                "provider": LABEL,
                "model": MODEL,
                "content": content,
                "usage": usage,
                "stopReason": "toolUse" if calls else "stop",
            }  # fmt: skip
        )

    def result(self, call_id: str, name: str, text: str, is_error: bool) -> None:
        self._message(
            {
                "role": "toolResult",
                "toolCallId": call_id,
                "toolName": name,
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            }  # fmt: skip
        )

    def next_call_id(self) -> str:
        self.calls += 1
        return f"standin_{self.session_id[:8]}_{int(time.time() * 1000) % 10**8}_{self.calls}"


# ------------------------------------------------------------------------------ steps


@dataclass
class Outcome:
    """What one executed tool step really produced (the same text the native result holds)."""

    command: str
    text: str
    returncode: int | None
    is_error: bool
    call_id: str


@dataclass
class Ctx:
    """Per-turn working memory for dynamic steps (an agent remembers what it just saw)."""

    params: Mapping[str, Any]
    outcomes: list[Outcome] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)

    @property
    def last(self) -> Outcome | None:
        return self.outcomes[-1] if self.outcomes else None


Text = str | Callable[[Ctx], str | None]


@dataclass(frozen=True)
class Bash:
    """A real shell command. ``command`` may be computed from earlier outcomes; None skips."""

    command: Text
    timeout: float = DEFAULT_TOOL_TIMEOUT


@dataclass(frozen=True)
class DocRead:
    """A non-executing documentation/file read (``read`` tool): nothing is dispatched."""

    path: str
    output: str


@dataclass(frozen=True)
class Say:
    """The assistant's closing message of the turn (no tool call)."""

    text: Text


Step = Bash | DocRead | Say


def _resolve(value: Text, ctx: Ctx) -> str | None:
    if isinstance(value, str):
        return value
    return value(ctx)


def _context_chars(ctx: Ctx) -> int:
    return sum(len(o.text) + len(o.command) for o in ctx.outcomes)


def run_step(step: Step, ctx: Ctx, writer: NativeWriter, env: Mapping[str, str], cwd: Path) -> None:
    if isinstance(step, Say):
        text = _resolve(step.text, ctx) or ""
        writer.assistant(text=text, scripted_input=_context_chars(ctx) // 4)
        return
    call_id = writer.next_call_id()
    if isinstance(step, DocRead):
        writer.assistant(
            calls=[(call_id, "read", {"path": step.path})], scripted_input=_context_chars(ctx) // 4
        )
        writer.result(call_id, "read", step.output, False)
        ctx.outcomes.append(Outcome(f"read {step.path}", step.output, 0, False, call_id))
        return
    command = _resolve(step.command, ctx)
    if command is None:
        return
    writer.assistant(
        calls=[(call_id, "bash", {"command": command})], scripted_input=_context_chars(ctx) // 4
    )
    started = time.monotonic()
    child = dict(env)
    child["LAB_NATIVE_CALL_ID"] = call_id
    try:
        proc = subprocess.run(
            ["/bin/sh", "-c", command], capture_output=True, text=True, cwd=cwd, check=False,
            env=child, timeout=step.timeout, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        text = proc.stdout + proc.stderr
        code: int | None = proc.returncode
        if code:
            text += f"\n[exit code: {code}]"
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        text, code = out + "\n[timed out]", None
    if env.get(ENV_TRAILER, "omp") == "omp":  # the real omp bash tool appends this trailer
        text += f"\n\nWall time: {time.monotonic() - started:.2f} seconds"
    is_error = code != 0
    writer.result(call_id, "bash", text, is_error)
    ctx.outcomes.append(Outcome(command, text, code, is_error, call_id))


# --------------------------------------------------------------------------- CLI entry


def run_turn(
    *,
    session_dir: Path,
    session_id: str,
    cwd: Path,
    prompt: str,
    steps: Sequence[Step],
    env: Mapping[str, str],
    params: Mapping[str, Any],
    task_file: Path | None,
    task: str | None,
) -> dict[str, Any]:
    """One user turn of the scripted conversation; returns a summary (no transcript text)."""
    if task_file is not None and task is not None:
        task_file.write_text(task)
    writer = NativeWriter(session_dir, session_id, str(cwd))
    writer.user(prompt)
    ctx = Ctx(params)
    for step in steps:
        run_step(step, ctx, writer, env, cwd)
    return {
        "label": LABEL, "session_id": session_id, "file": str(writer.path),
        "continued": writer.continued, "tool_calls": len(ctx.outcomes), "memory": ctx.memory,
    }  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m eval.accounting_airline.standin_agent --script L1 --params p.json ...``"""
    # run as ``-m`` this file is ``__main__``: use the package module so the step classes the
    # scripts build are the very classes ``run_step`` dispatches on
    from eval.accounting_airline import standin_agent as package, standin_scripts

    parser = argparse.ArgumentParser(prog="standin_agent", description="SCRIPTED STAND-IN (no LLM)")
    parser.add_argument("--script", required=True, help="named scripted turn")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--summary", type=Path, help="write the turn summary JSON here")
    args = parser.parse_args(argv)
    params = json.loads(args.params.read_text())
    prompt, steps = standin_scripts.build(args.script, params)
    summary = package.run_turn(
        session_dir=args.session_dir, session_id=args.session_id, cwd=args.cwd, prompt=prompt,
        steps=steps, env=os.environ, params=params, task_file=args.task_file, task=args.task,
    )  # fmt: skip
    if args.summary:
        args.summary.write_text(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
