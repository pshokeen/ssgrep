"""Scenes and observers for the collector guarantee tests (``test_verify_*``).

Every scene is real: native Claude transcripts under the isolated ``CLAUDE_CONFIG_DIR``, journal
events through ``journal.append`` with the real keyed fingerprint, the pinned candidate-frequency
view the collector itself loads, and cycles through ``collector.collect_once`` with the real
reducer. Nothing below the collector is faked. Instants are fixed UTC so a scene is reproducible
byte for byte across processes: a reference run (same scene, another data root, or another
process) is what a crash/restart/incremental run must converge to.

``isolated(env)`` points THIS process at another set of stores for the duration of a block, so
a reference run needs no second test process; ``env_for(root)`` rebuilds the layout the autouse
fixture of ``tests/conftest.py`` creates, so a subprocess sees what the in-process test sees.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import ssgrep.accounting.views.snapshots as snapshots
from ssgrep.accounting.capture import fingerprint, journal
from ssgrep.accounting.collect import frequency as frequency, store as store
from ssgrep.accounting.collect.frequency import FrequencyView
from ssgrep.accounting.collect.state import VersionedLedger, load_state
from ssgrep.accounting.privacy import policy, storage
from tests.accounting.collect import collector_scene as scene
from tests.accounting.ledger.reduce_builders import (
    MCP,
    PROFILE,
    card,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    rare_view,
    results_json,
    source_session,
    ts,
)
from tests.accounting.readers.readers_fixtures import jsonl

NAMES = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
)
CAPTURE_DAY = 11  # capture begins on day 11; fact sources are dated then, consumers on day 12
LAYOUT = {
    "SSGREP_DATA_DIR": "data",
    "CLAUDE_CONFIG_DIR": "claude",
    "OPENCODE_CONFIG_DIR": "opencode",
    "CODEX_HOME": "codex",
    "PI_CODING_AGENT_DIR": "pi-agent",
    "PRIME_AGENT_CODING_AGENT_DIR": "prime-agent",
    "PI_SESSION_DIR": "pi-sessions",
    "PRIME_AGENT_SESSION_DIR": "prime-agent-sessions",
    "OMP_SESSIONS_DIR": "omp-sessions",
    "OMP_AGENT_DIR": "omp-agent",
    "XDG_DATA_HOME": "xdg-data",
}
UNSET = (
    "SSGREP_PI_SESSIONS_DIR",
    "SSGREP_PRIME_AGENT_SESSIONS_DIR",
    "SSGREP_OMP_SESSIONS_DIR",
    "SSGREP_CODEX_SESSIONS_DIR",
    "SSGREP_OPENCODE_DB",
    "SSGREP_TRANSCRIPT_DIRS",
    "SSGREP_CAPTURE_PROFILE",
    "SSGREP_MCP_CALLER_WORKSPACE",
)
MANAGED = tuple(LAYOUT) + UNSET + ("HOME",)


def env_for(root: Path) -> dict[str, str]:
    """The isolated environment of ``tests/conftest.py`` rebuilt for any ``root``."""
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update({name: str(root / part) for name, part in LAYOUT.items()})
    env["HOME"] = str(root / "home")
    return env


@contextmanager
def isolated(env: Mapping[str, str]) -> Iterator[None]:
    """Run a block against the stores named by ``env`` (a fresh worker's view of the world)."""
    saved = {name: os.environ.get(name) for name in MANAGED}
    journal.close_all()
    try:
        for name in MANAGED:
            os.environ.pop(name, None)
        os.environ.update({k: v for k, v in env.items() if k in MANAGED})
        fingerprint.forget_keys()
        yield
    finally:
        journal.close_all()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        fingerprint.forget_keys()


def code(n: int) -> str:
    return f"ERR_QUOTA_{NAMES[n].upper()}"


def path(n: int) -> str:
    return f"services/{NAMES[n]}/limits.py"


def fact(n: int) -> str:
    return f"deploy failed: {code(n)} raised in {path(n)}"


def query(n: int) -> str:
    return f"why did the {NAMES[n]} deploy fail"


def pinned_view(count: int = len(NAMES)) -> FrequencyView:
    """A frozen view in which every scene fact is rare (credit possible in every scenario)."""
    rare: list[str] = []
    for n in range(count):
        rare += [code(n), path(n)]
    return rare_view(*rare)


def enable(*, pin: FrequencyView | bool = False) -> None:
    """Enable measurement, install the fixed key, optionally pin a view (True: the scene's)."""
    policy.enable_collection(PROFILE)
    scene.install_key()
    if pin:
        frequency.pin(PROFILE, pinned_view() if pin is True else pin)


def start_capture(day: int = CAPTURE_DAY) -> None:
    scene.journal_events(
        scene.execution("n-start", ts(day, 8), text="capture begins", transport="cli")
    )


def projects(root: Path) -> Path:
    return root / "claude" / "projects"


def put(base: Path, name: str, records: list[dict[str, Any]], clock: datetime) -> Path:
    """A native Claude transcript ``<base>/-work/<name>.jsonl``; mtime = the scene clock."""
    target = base / "-work" / f"{name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(jsonl(records))
    os.utime(target, (clock.timestamp(), clock.timestamp()))
    return target


def pair(base: Path, n: int, clock: datetime) -> None:
    """Fact source ``src<n>`` (day 11) and a consumer that searches, reads and acts on it."""
    put(base, f"src{n}", source_session(f"src{n}", fact(n), when_day=CAPTURE_DAY), clock)
    hour, minute = 9 + n, 10 * n
    session = f"cons{n}"
    body = results_json(card(f"claude:src{n}:ep:0", fact(n)))
    records = [
        claude_prompt(session, "fix the deploy", uuid=f"u{n}", ts=ts(12, hour, minute)),
        claude_tool_use(
            session,
            f"t{n}a",
            MCP,
            {"query": query(n)},
            ts=ts(12, hour, minute, 10),
            message_id=f"m{n}a",
        ),
        claude_tool_result(session, f"t{n}a", body, ts=ts(12, hour, minute, 12)),
        claude_tool_use(
            session,
            f"t{n}b",
            "Bash",
            {"command": f"vim {path(n)}"},
            ts=ts(12, hour, minute, 20),
            message_id=f"m{n}b",
        ),
        claude_tool_result(session, f"t{n}b", "ok", ts=ts(12, hour, minute, 22)),
    ]
    put(base, session, records, clock)
    scene.journal_events(
        scene.execution(f"n-{n}", ts(12, hour, minute, 11), text=query(n), transport="mcp")
    )


def filler(base: Path, name: str, clock: datetime, count: int, *, first: int = 0) -> None:
    """``count`` ordinary prompts (one document each): a corpus to pin a frequency view from."""
    records = [
        claude_prompt(
            name,
            f"{name} prompt number {i}",
            uuid=f"{name}{i}",
            ts=ts(CAPTURE_DAY, 9, i // 60, i % 60),
        )
        for i in range(first, first + count)
    ]
    put(base, name, records, clock)


def timed_event(
    execution_id: str,
    when: str,
    microseconds: int | None,
    text: str = "q",
    transport: str = "mcp",
) -> list:
    """A journal execution whose duration is exactly ``microseconds`` (None: unknown)."""
    events = scene.execution(execution_id, when, text=text, transport=transport)
    mono = None if microseconds is None else microseconds * 1000
    return [replace(e, mono_ns=mono) if e.mono_ns is not None else e for e in events]


# ---- observers: what a reader/operator sees on disk -------------------------------------------


def accounting_dir() -> Path:
    return storage.data_root() / "accounting" / PROFILE


def checkpoint() -> store.Checkpoint | None:
    return store.load_checkpoint(PROFILE)


def ledger_digest() -> str | None:
    """Digest of the canonical ledger rows of the committed state (the replay-equality value)."""
    committed = checkpoint()
    if committed is None:
        return None
    state = load_state(PROFILE, committed, committed.methodology)
    if state.ledger is None:
        return None
    return VersionedLedger.restore_versioned(state.ledger, state.superseded).rows().digest


def evidence_identities() -> list[tuple[str, str, str, str]]:
    """Every retained evidence record as (runtime, session, record id, digest), sorted."""
    committed = checkpoint()
    if committed is None:
        return []
    items = store.iter_evidence(PROFILE, committed.segments)
    return sorted((e.runtime, e.session_id, e.record_id, e.digest) for e in items)


def evidence_semantics() -> tuple[tuple, tuple]:
    """Unconflicted measurements plus every variant's ownership and conflict facts.

    Primary bodies of conflicted targets have no financial authority. Their full
    owner/digest/link/time evidence remains in the complete variant union; only
    unconflicted native measurements require a common physical representative.
    Admission-time conflict bits become target-level conflict authority.
    """
    from ssgrep.accounting.ledger.contracts import target_key
    from ssgrep.accounting.readers.native_identity import CLAIM, claimed_owner, owner

    committed = checkpoint()
    assert committed is not None
    native, ownership = set(), set()
    variants: dict[tuple[str, str, str], set[str]] = {}
    flagged: set[tuple[str, str, str]] = set()
    for item in store.iter_evidence(PROFILE, committed.segments):
        payload = item.payload
        if item.kind == CLAIM:
            target = target_key(item)
            assert target is not None
            record_id, digest = payload["target_record_id"], payload["target_digest"]
            kind, observed_owner = payload["original_kind"], claimed_owner(item)
        else:
            target, record_id, digest, kind = (
                item.native_key,
                item.record_id,
                item.digest,
                item.kind,
            )
            observed_owner = owner(item)
            measurements = {
                key: value
                for key, value in payload.items()
                if key not in {"cwd", "parent_session", "call_record_id", "call_link_status"}
            }
            native.add(
                (
                    target,
                    record_id,
                    digest,
                    kind,
                    item.ts_utc,
                    item.schema,
                    item.origin_domain,
                    item.origin_id,
                    item.origin_status,
                    "call_record_id" in payload,
                    "call_link_status" in payload,
                    json.dumps(measurements, sort_keys=True),
                )
            )
        variants.setdefault(target, set()).add(digest)
        if payload.get("origin_conflict") is True:
            flagged.add(target)
        ownership.add(
            (
                target,
                record_id,
                digest,
                kind,
                observed_owner,
                payload.get("call_record_id"),
                payload.get("call_link_status"),
                item.locator,
                item.ts_utc,
            )
        )
    conflicted = flagged | {target for target, digests in variants.items() if len(digests) > 1}
    native = {row for row in native if row[0] not in conflicted}
    ownership = {(*row, row[0] in conflicted) for row in ownership}
    return tuple(sorted(native, key=repr)), tuple(sorted(ownership, key=repr))


def tree(directory: Path, *, skip: Iterable[str] = ()) -> dict[str, bytes]:
    """Relative path -> bytes for every file below ``directory`` (``skip`` = name substrings)."""
    if not directory.is_dir():
        return {}
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and not any(s in str(p) for s in skip)
    }


def snapshot_files() -> dict[str, bytes]:
    return tree(accounting_dir() / "snapshots")


def run_to_idle(now: datetime, options: Any | None = None, limit: int = 400, **extra: Any):
    """Collect until a cycle ends ``complete`` (a restart between every cycle); the results."""
    results = []
    for _ in range(limit):
        result = scene.collect(now, options or scene.LIVE, **extra)
        results.append(result)
        if result.report is not None and result.report.stopped_by == "complete":
            return results
    raise AssertionError(f"still not complete after {limit} cycles")


@dataclass(frozen=True)
class Facts:
    """Everything a reader or operator can compare between two runs of the same scene."""

    ledger: str | None
    evidence: tuple
    snapshot_id: str
    counters: str
    generation: int
    current_bytes: bytes
    native_evidence: tuple
    ownership: tuple


def facts() -> Facts:
    raw = (accounting_dir() / "snapshots" / "current.json").read_bytes()
    envelope = snapshots.decode(raw)
    native_evidence, ownership = evidence_semantics()
    return Facts(
        ledger_digest(),
        tuple(evidence_identities()),
        envelope.snapshot_id,
        json.dumps(envelope.observations.get("counters"), sort_keys=True),
        envelope.generation,
        raw,
        native_evidence,
        ownership,
    )


def wipe(root: Path) -> None:
    """Forget every trace of a run: accounting data, journal handles, transcripts, key cache."""
    journal.close_all()
    for part in ("data", "claude"):
        shutil.rmtree(root / part, ignore_errors=True)
    fingerprint.forget_keys()
