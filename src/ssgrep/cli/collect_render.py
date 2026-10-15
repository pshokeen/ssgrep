"""Human renderers for ``metrics collect`` and ``metrics serve`` (width-aware, plain text).

They render command data and startup details; every untrusted string passes through ``safe``.
The startup screen exposes the endpoint token-file path, never its value.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text

from ssgrep.accounting.service_types import NextAction
from ssgrep.cli.collect_json import SHORT_ID
from ssgrep.cli.savings_text import Doc, fmt_bytes, fmt_int, safe

_STOPPED = {"byte_budget": "byte budget", "time_budget": "time budget"}


def _count(value: object) -> str:
    return "unknown" if not isinstance(value, int) else fmt_int(value)


def _size(value: object) -> str:
    return "unknown" if not isinstance(value, int) else fmt_bytes(value)


def _next(doc: Doc, actions: tuple[NextAction, ...]) -> None:
    doc.blank()
    for number, action in enumerate(actions):
        doc.command("Next:" if number == 0 else "Or:", action.command)
        doc.line(safe(action.why), indent=6)


def _backfill(doc: Doc, data: dict[str, Any]) -> None:
    doc.heading(
        "HISTORICAL BACKFILL (not live executions)",
        "Imported history is kept as its own series; it never counts as recent live activity.",
    )
    interval = data["interval"]
    doc.pair("Interval", f"{safe(interval['since'])} to {safe(interval['until'])} (UTC)")
    series = data.get("backfill")
    if not isinstance(series, dict):
        doc.line("No backfill series was recorded by this run.", indent=2)
        return
    if "complete" in series:
        doc.pair("Complete", "yes" if series["complete"] is True else "no, run again to continue")
    counts = series.get("counts")
    if isinstance(counts, list):
        for pair in counts:
            if isinstance(pair, list) and len(pair) == 2 and isinstance(pair[1], int):
                doc.pair(safe(pair[0]), fmt_int(pair[1]), indent=4)


def _recompute(doc: Doc, data: dict[str, Any]) -> None:
    periods = data["unavailable_periods"]
    doc.heading("RECOMPUTE", "Uses retained frozen evidence only; no index is rebuilt.")
    if not periods:
        doc.line("Every requested period could be recomputed.", indent=2)
        return
    doc.line(f"{len(periods)} period(s) unavailable; their saved totals are kept:", indent=2)
    for period in periods:
        reason = safe(period["reason"]).replace("_", " ")
        doc.bullet(f"{safe(period['period'])}: {reason}", indent=4)


def render_collect(
    data: dict[str, Any], warnings: tuple[str, ...], actions: tuple[NextAction, ...], width: int
) -> Text:
    doc = Doc(width)
    state = data["state"]
    title = {
        "published": "Collection complete.",
        "partial": "Collection stopped early; progress is saved.",
    }.get(state, "Collection state")
    doc.line(title, style="bold")
    doc.pair("Profile", safe(data["profile"]))
    doc.pair("Mode", safe(data["mode"]))
    doc.pair("Counter epoch", _count(data["epoch"]))
    doc.pair("Generation", _count(data["generation"]))
    snapshot = data["snapshot_id"]
    doc.pair("Snapshot", safe(snapshot)[:SHORT_ID] if snapshot else "none published")
    doc.pair("Bytes read", _size(data["bytes_read"]))
    doc.pair("Lag remaining", _size(data["lag_bytes"]))
    if state == "partial":
        reason = _STOPPED.get(str(data["stopped_by"]), safe(data["stopped_by"]))
        doc.blank()
        doc.line(f"Stopped by {reason} -- run again to resume.", style="bold")
    if data["new_epoch"]:
        doc.blank()
        doc.line(
            f"New counter epoch {_count(data['epoch'])} declared: live counters restart at zero "
            "for future events; ledger and history totals are kept."
        )
    if data["mode"] == "backfill":
        _backfill(doc, data)
    if data["mode"] == "recompute":
        _recompute(doc, data)
    for warning in warnings:
        doc.line(f"Note: {safe(warning)}")
    _next(doc, actions)
    return doc.text()


def render_running(data: dict[str, Any], actions: tuple[NextAction, ...], width: int) -> Text:
    doc = Doc(width)
    owner = data["owner"]
    doc.line("Another worker is already collecting; nothing was started.", style="bold")
    doc.line("This is normal: its snapshot is not touched and no competing generation was made.")
    doc.heading("RUNNING WORKER")
    pid = owner["pid"]
    doc.pair("Process", f"pid {pid}" if pid is not None else "unknown pid")
    doc.pair("Kind", safe(owner["kind"]) if owner["kind"] else "unknown")
    doc.pair("Phase", safe(owner["phase"]) if owner["phase"] else "unknown (no progress file)")
    done, total = owner["sources_done"], owner["sources_total"]
    sources = "unknown" if done is None or total is None else f"{fmt_int(done)} of {fmt_int(total)}"
    doc.pair("Sources", sources)
    doc.pair("Bytes read", _size(owner["bytes_read"]))
    doc.pair("Lag remaining", _size(owner["lag_bytes"]))
    if owner["updated_at"]:
        doc.pair("Updated", safe(owner["updated_at"]))
    _next(doc, actions)
    return doc.text()


def render_serve_start(
    profile: str,
    url: str,
    port: int,
    token_file: str,
    collect_seconds: int,
    export_seconds: int,
    pid: int,
    width: int,
) -> Text:
    doc = Doc(width)
    doc.line(f"ssgrep metrics serve (pid {pid})", style="bold")
    doc.line(f"URL: {safe(url)}")
    doc.line(f"Port: {port}")
    doc.line(f"Profile: {safe(profile)}")
    doc.line(f"Authentication token file: {safe(token_file)}")
    doc.line(f"Collecting every {collect_seconds} s, export check every {export_seconds} s.")
    doc.line("Foreground only: stop with Ctrl-C or SIGTERM.")
    return doc.text()


def render_serve_final(data: dict[str, Any], width: int) -> Text:
    doc = Doc(width)
    doc.line("Worker stopped.", style="bold")
    doc.line(f"Loopback server stopped at {safe(data['url'])}.")
    doc.line(f"Port: {_count(data['port'])}")
    doc.line(f"Authentication token file: {safe(data['token_file'])}")
    doc.pair("Profile", safe(data["profile"]))
    doc.pair("State", safe(data["phase"]))
    doc.pair("Cycles run", _count(data["cycles"]))
    report = data["last_report"]
    if report:
        generation = report.get("generation")
        doc.pair("Last cycle", f"{safe(report.get('status'))}, generation {_count(generation)}")
    if data["last_error"]:
        doc.pair("Last error", safe(data["last_error"]))
    doc.blank()
    doc.line("The worker lease is released.")
    return doc.text()
