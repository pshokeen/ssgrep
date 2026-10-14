"""Hand-built frozen evidence for the lab accounting oracle tests.

Every text has an EXACT, caller-chosen length so the tests can derive credit/debit by hand.
Nothing here uses product code: sessions, dispatches and results are literal data.
"""

from __future__ import annotations

import json
from dataclasses import replace

from eval.accounting_airline.observer import (
    Frozen,
    NativeRecord,
    NativeSession,
    ObservedCall,
)

TOKEN = "MCT-DEN-045"
OTHER = "MCT-JFK-060"
REF_A1 = "omp:sa:ep:1"
SEARCH_CMD = 'ssgrep search "connection rules" --json'
SHOW_CMD = f"ssgrep show {REF_A1} --json"
CONSUMER_T0, SOURCE_T0 = 5000, 1000


def text(n: int, tag: str = "") -> str:
    """Exactly ``n`` characters; ``tag`` first, then dots (a dot is never a token character)."""
    assert len(tag) <= n, (tag, n)
    return tag + "." * (n - len(tag))


def command(cmd: str, n: int) -> str:
    return cmd.ljust(n)


def _pad_json(build, n: int) -> str:
    base = build("")
    assert len(base) <= n, (len(base), n)
    out = build("x" * (n - len(base)))
    assert len(out) == n
    return out


def search_json(excerpt: str, n: int, ref: str = REF_A1) -> str:
    """A search document of exactly ``n`` chars: one card, padding in its own ``pad`` field."""

    def build(pad: str) -> str:
        card = {"ref": ref, "title": "rules", "excerpt": excerpt, "pad": pad}
        return json.dumps({"ok": True, "data": {"results": [card]}}, separators=(",", ":"))

    return _pad_json(build, n)


def show_json(body: str, n: int, ref: str = REF_A1) -> str:
    def build(pad: str) -> str:
        doc = {"episode_id": ref, "prompt_text": body, "response_text": pad}
        return json.dumps({"ok": True, "data": doc}, separators=(",", ":"))

    return _pad_json(build, n)


class SessionBuilder:
    """Chronological native records with deterministic ids and strictly increasing times."""

    def __init__(self, sid: str, actor: str, label: str | None = None, t0: int = 1000) -> None:
        self.sid, self.actor, self.label, self.now = sid, actor, label, t0
        self.records: list[NativeRecord] = []

    def _add(self, kind: str, body: str, call_id: str | None, error: bool | None) -> str:
        record_id = f"{self.sid}-r{len(self.records)}"
        self.records.append(
            NativeRecord(len(self.records), kind, record_id, body, call_id, None, self.now, error)
        )
        self.now += 10
        return record_id

    def user(self, body: str) -> str:
        return self._add("user_turn", body, None, None)

    def call(self, call_id: str, body: str) -> str:
        return self._add("tool_call", body, call_id, None)

    def result(self, call_id: str, body: str, *, error: bool | None = None) -> str:
        return self._add("tool_result", body, call_id, error)

    def entry(self) -> tuple[str, str | None, NativeSession]:
        native = NativeSession(self.sid, f"/lab/{self.sid}.jsonl", "/lab", "omp", "m",
                               tuple(self.records), 0, 0, 0, "0" * 64)  # fmt: skip
        return self.actor, self.label, native


def observed(
    call_id: str,
    actor: str,
    argv: tuple[str, ...],
    *,
    native: str | None = None,
    tool: str = "ssgrep",
    state: str = "verified",
    exit_code: int | None = 0,
    stdout: str = "",
    chars: int | None = None,
    seq: int = 0,
) -> ObservedCall:
    return ObservedCall(
        call_id, actor, None, tool, argv, state, exit_code, stdout,
        len(stdout) if chars is None else chars, len(stdout.encode()), 1, seq,
        (), 0, native,
    )  # fmt: skip


def frozen(sessions: list[SessionBuilder], calls: list[ObservedCall]) -> Frozen:
    ordered = tuple(replace(c, dispatch_seq=i + 1) for i, c in enumerate(calls))
    return Frozen("run", ordered, {s.sid: s.entry() for s in sessions}, (), "head")


# ------------------------------------------------------------------------------ the world


def source_a(
    *, call_len: int = 40, witness_tail: str = "", t0: int = SOURCE_T0, l2_user: str | None = None,
    call_tag: str = "", first_result_tag: str = "",
) -> SessionBuilder:  # fmt: skip
    """Session ``sa`` (solver-a): L1 then L2; the witness is record ``sa-r5``.

    chars: r0 20, r1 24, r2 30 | r3 16, r4 ``call_len``, r5 100 (+TOKEN), r6 60, r7 80.
    """
    sa = SessionBuilder("sa", "solver-a", "L2", t0)
    sa.user(text(20))
    sa.call("c0", text(24))
    sa.result("c0", text(30, first_result_tag))
    sa.user(l2_user if l2_user is not None else text(16))
    sa.call("c1", text(call_len, call_tag))
    sa.result("c1", text(100, f"{TOKEN} {witness_tail}".strip()))
    sa.call("c2", text(60))
    sa.result("c2", text(80))
    return sa


def lookup(
    sb: SessionBuilder,
    call_id: str,
    visible: str,
    *,
    cmd: str = SEARCH_CMD,
    cmd_len: int = 50,
    error: bool | None = None,
) -> None:
    sb.call(call_id, command(cmd, cmd_len))
    sb.result(call_id, visible, error=error)


def consumer_b(
    *,
    sid: str = "sb",
    actor: str = "solver-b",
    brief: str | None = None,
    excerpt: str | None = None,
    visible: str | None = None,
    action: str | None = "use",
    t0: int = CONSUMER_T0,
    cmd: str = SEARCH_CMD,
    cid: str = "cs",
) -> SessionBuilder:
    """Consumer ``sb``: brief 30, search call 50, result 200 (card carries TOKEN), action 60/20.

    ``action``: None (no later call), "use" (call text carries TOKEN, result ok), "error_flag"
    (native is_error), "error_text" (result contains ``Traceback``).
    """
    sb = SessionBuilder(sid, actor, "L3", t0)
    sb.user(brief if brief is not None else text(30))
    shown = visible if visible is not None else search_json(excerpt or f"{TOKEN} min 45", 200)
    lookup(sb, f"{cid}1", shown, cmd=cmd)
    if action:
        sb.call(f"{cid}2", command(f"psql -c select '{TOKEN}'", 60))
        flag = True if action == "error_flag" else None
        body = "Traceback: boom" if action == "error_text" else "ok"
        sb.result(f"{cid}2", body.ljust(20, "."), error=flag)
    return sb


def search_call(sid: str = "sb", actor: str = "solver-b", **kw) -> ObservedCall:
    """The dispatch for ``cs1`` of ``consumer_b`` (stdout is the 200-char card document)."""
    stdout = kw.pop("stdout", None)
    if stdout is None:
        stdout = search_json(f"{TOKEN} min 45", 200)
    argv = kw.pop("argv", ("ssgrep", "search", "connection rules", "--json"))
    return observed(kw.pop("call_id", f"{sid}-o1"), actor, argv,
                    native=kw.pop("native", "cs1"), stdout=stdout, **kw)  # fmt: skip


def base_world() -> tuple[Frozen, SessionBuilder, SessionBuilder]:
    sa, sb = source_a(), consumer_b()
    return frozen([sa, sb], [search_call()]), sa, sb
