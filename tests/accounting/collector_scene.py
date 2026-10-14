"""Scenes for collector tests: real journal events, real native transcripts, real reducer.

Nothing is hand-built below the collector: events go through ``journal.append`` with the real
keyed fingerprint, transcripts are native Claude JSONL under the isolated ``CLAUDE_CONFIG_DIR``
the shared fixture sets, and cycles run ``collector.collect_once`` with the default reducer.
Times are fixed UTC instants so every run of a scene is reproducible; file modification times
are pinned to the scene clock so pre-baselining never mistakes a fixture for old history.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ssgrep.accounting import fingerprint, journal, plainfile, snapshots, storage
from ssgrep.accounting.collector import collect_once
from ssgrep.accounting.collector_store import Checkpoint, dumps, load_checkpoint
from ssgrep.accounting.collector_types import (
    CollectOptions,
    CollectResult,
    CycleBudget,
    CycleReport,
)
from ssgrep.accounting.contracts import ExecutionEvent
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.ledger_contracts import SnapshotEnvelope
from ssgrep.accounting.plain_codec import to_plain
from tests.accounting.readers_fixtures import jsonl
from tests.accounting.reduce_builders import (
    KEY,
    MCP,
    PROFILE,
    claude_tool_result,
    claude_tool_use,
    event,
    results_json,
    ts,
)

DAY = 12  # 2026-10-12: the scene's "today"
LIVE = CollectOptions()


def at(day: int = DAY, hour: int = 10, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, second, tzinfo=UTC)


def clock_after(start: datetime, **delta: int) -> datetime:
    return start + timedelta(**delta)


def install_key(profile: str = PROFILE) -> None:
    """The profile's fingerprint key: fixed so scene events and the collector agree."""
    directory = storage.open_profile_dir(profile)
    try:
        storage.write_once(directory, fingerprint.KEY_FILE, KEY)
    finally:
        os.close(directory)
    fingerprint.forget_keys()


def pin_frequency(view: FrequencyView, profile: str = PROFILE) -> None:
    """Pin the candidate-frequency view the collector loads (it never derives one itself)."""
    directory = storage.open_profile_dir(profile, "collector")
    try:
        storage.write_once(directory, "frequency.json", dumps(to_plain(view)))
    finally:
        os.close(directory)


def transcripts(root: Path) -> Path:
    return root / "claude" / "projects" / "-work"


def write_transcript(root: Path, name: str, records: list[dict[str, Any]], clock: datetime) -> Path:
    """A native Claude transcript whose mtime is the scene clock (never 'old history')."""
    target = transcripts(root) / f"{name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(jsonl(records))
    stamp = clock.timestamp()
    os.utime(target, (stamp, stamp))
    return target


def append_transcript(path: Path, records: Iterable[dict[str, Any]], clock: datetime) -> None:
    with path.open("ab") as handle:
        handle.write(jsonl(list(records)))
    stamp = clock.timestamp()
    os.utime(path, (stamp, stamp))


def journal_events(events: Iterable[ExecutionEvent], profile: str = PROFILE) -> None:
    for item in events:
        journal.append(profile, item)
    journal.close_all()  # one file per scene step: segments are immutable once closed


def execution(execution_id: str, when: str, **fields: Any) -> list[ExecutionEvent]:
    return event(execution_id, when, **fields)


def search_records(
    session: str, call_id: str, query: str, when: tuple[int, int, int]
) -> list[dict[str, Any]]:
    """One MCP search call + its (empty) result, dated ``DAY`` at ``when`` (h, m, s)."""
    hour, minute, second = when
    return [
        claude_tool_use(session, call_id, MCP, {"query": query}, ts=ts(DAY, hour, minute, second)),
        claude_tool_result(session, call_id, results_json(), ts=ts(DAY, hour, minute, second + 3)),
    ]


def start_marker() -> list[ExecutionEvent]:
    """An early event so capture 'began' before the scene's transcripts (baseline = 07:55)."""
    return execution("n-start", ts(DAY, 8), text="capture begins", transport="cli")


def collect(
    now: datetime,
    options: CollectOptions = LIVE,
    *,
    budget: CycleBudget | None = None,
    **extra: Any,
) -> CollectResult:
    chosen = (
        options
        if budget is None
        else CollectOptions(
            since=options.since,
            until=options.until,
            backfill=options.backfill,
            recompute=options.recompute,
            new_epoch=options.new_epoch,
            budget=budget,
        )
    )
    return collect_once(PROFILE, chosen, now=now, **extra)


def report(result: CollectResult) -> CycleReport:
    assert result.report is not None, result
    return result.report


def accounting_dir() -> Path:
    return storage.data_root() / "accounting" / PROFILE


def current() -> SnapshotEnvelope:
    raw = plainfile.read_private(
        storage.data_root(), "accounting", PROFILE, "snapshots", "current.json", max_bytes=1 << 24
    )
    assert raw is not None, "no current snapshot"
    return snapshots.decode(raw)


def mapping(value: object) -> dict[str, Any]:
    """A snapshot section narrowed to the dict the envelope really holds (fails loudly if not)."""
    assert isinstance(value, dict), type(value)
    return value


def coverage(envelope: SnapshotEnvelope) -> dict[str, Any]:
    return mapping(envelope.coverage)


def window(envelope: SnapshotEnvelope, name: str) -> dict[str, Any]:
    return mapping(envelope.windows[name])


def observations(envelope: SnapshotEnvelope) -> dict[str, Any]:
    return mapping(envelope.observations)


def committed() -> Checkpoint:
    """The committed checkpoint; a missing one fails the test with a clear message."""
    loaded = load_checkpoint(PROFILE)
    assert loaded is not None, "no committed checkpoint"
    return loaded


def quantity(envelope: SnapshotEnvelope, window_name: str, name: str) -> int | None:
    entry = mapping(window(envelope, window_name)["observations"])[name]
    return entry["value"]


def counter_total(envelope: SnapshotEnvelope, *, runtime: str | None = None) -> int:
    """Sum of the live execution counters (optionally one capture-time runtime label)."""
    rows = observations(envelope)["counters"]["executions"]
    return sum(r["count"] for r in rows if runtime is None or r["runtime"] == runtime)


def file_digests(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


Hook = Callable[[str], None]
