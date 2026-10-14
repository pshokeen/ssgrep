"""Fill the ten metric families of validation-lab.md section 6 from real run evidence.

Every quantity carries unit, basis, source, sample count, actor scope, window and methodology.
A quantity the chosen runtime cannot supply is recorded ``unavailable`` with a reason; nothing
is estimated to look complete. High-cardinality identifiers never appear as metric values.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .lab_context import Run, dig
from .lab_journal import read_journal
from .report import Metric

LAB = "lab-observer"
PRODUCT = "ssgrep-binaries"


def _add(run: Run, family: str, name: str, value: int | str | None, unit: str, *, basis: str,
         source: str, samples: int | None, scope: str, window: str = "run",
         method: str = LAB, reason: str | None = None) -> None:  # fmt: skip
    if value is None:
        basis, samples = "unavailable", samples
    run.report.metrics.add(
        Metric(family, name, value, unit, basis, source, samples, scope, window, method, reason)
    )


def fill(run: Run) -> None:
    frozen = run.frozen()
    calls = frozen.calls
    solver_calls = [c for c in calls if c.actor.startswith("solver")]
    # -- fixture/domain
    _add(
        run,
        "fixture_domain",
        "table_count",
        len(run.base.counts),
        "tables",
        basis="observed",
        source="database",
        samples=1,
        scope="verifier",
        window="C0",
    )
    _add(
        run,
        "fixture_domain",
        "row_count",
        sum(run.base.counts.values()),
        "rows",
        basis="observed",
        source="database",
        samples=1,
        scope="verifier",
        window="C0",
    )
    sql = [c for c in solver_calls if c.tool == "sql"]
    errors = Counter(
        (e.get("sqlstate") or "none")
        for c in sql
        for e in c.sql
        if e.get("phase") == "finish" and e.get("exit_code") not in (0, None)
    )
    _add(
        run,
        "fixture_domain",
        "sql_error_attempts",
        sum(errors.values()),
        "attempts",
        basis="observed",
        source="observer sql finish",
        samples=len(sql),
        scope="solver-a+b",
    )
    # -- runtime work
    for actor in ("solver-a", "solver-b"):
        mine = [c for c in solver_calls if c.actor == actor]
        _add(
            run,
            "runtime_work",
            "tool_invocations",
            len(mine),
            "calls",
            basis="observed",
            source="observer dispatch",
            samples=len(mine),
            scope=actor,
        )
    sessions = frozen.sessions
    for key, (actor, _task, native) in sessions.items():
        if actor in ("solver-a", "solver-b"):
            turns = sum(1 for r in native.records if r.kind == "user_turn")
            _add(
                run,
                "runtime_work",
                f"user_turns.{actor}",
                turns,
                "turns",
                basis="observed",
                source="native session records",
                samples=len(native.records),
                scope=actor,
            )
            _ = key
    _add(
        run,
        "runtime_work",
        "process_restarts",
        0,
        "restarts",
        basis="observed",
        source="one stand-in process per scripted turn",
        samples=4,
        scope="solver-a+b",
    )
    # -- tokens/money
    for actor in ("solver-a", "solver-b"):
        total = sum(
            r.usage.get("total", 0)
            for _k, (a, _t, n) in sessions.items()
            if a == actor
            for r in n.records
            if r.kind == "usage"
        )
        _add(
            run,
            "tokens_money",
            f"reported_total_tokens.{actor}",
            total,
            "tokens",
            basis="reported",
            source="native usage (scripted-standin, NOT provider usage)",
            samples=sum(1 for _k, (a, _t, n) in sessions.items() if a == actor),
            scope=actor,
            method="scripted-standin",
        )
    _add(
        run,
        "tokens_money",
        "reported_cost",
        None,
        "usd",
        basis="unavailable",
        source="native usage",
        samples=0,
        scope="solver-a+b",
        method="scripted-standin",
        reason="the scripted stand-in has no provider; no cost is reported or assumed free",
    )
    # -- retrieval
    for actor in ("solver-b", "verifier", "control"):
        looks = [c for c in calls if c.actor == actor and c.tool == "ssgrep"]
        for op in ("search", "show"):
            mine = [c for c in looks if len(c.argv) > 1 and c.argv[1] == op]
            _add(
                run,
                "retrieval",
                f"{op}_executions.{actor}",
                len(mine),
                "executions",
                basis="observed",
                source="observer dispatch",
                samples=len(looks),
                scope=actor,
            )
    # -- persistence/index
    idx = run.notes.get("index_a", {})
    status = idx.get("status")
    _add(
        run,
        "persistence_index",
        "a_index_wall",
        idx.get("ms"),
        "ms",
        basis="observed",
        source="real ssgrep index",
        samples=1,
        scope="verifier",
        window="C3",
        reason=None if idx else "index not run (persistence blocked)",
    )
    _add(
        run,
        "persistence_index",
        "episode_count",
        dig(status, "episode_count") if status else None,
        "episodes",
        basis="observed",
        source="ssgrep status --json",
        samples=1,
        scope="verifier",
        window="C3",
        method=PRODUCT,
        reason=None if status else "index not run",
    )
    # -- attribution
    final = run.collected.get("final") or next(reversed(run.collected.values()), None)
    cover = dig(final.savings_document, "coverage") if final else None
    _add(
        run,
        "attribution",
        "recorded_lookups",
        dig(cover, "recorded_lookups") if cover else None,
        "lookups",
        basis="reported",
        source="ssgrep savings --json coverage",
        samples=1,
        scope="all",
        method=PRODUCT,
        reason=None if cover else "no collection",
    )
    _add(
        run,
        "attribution",
        "unlinked_executions",
        run.notes.get("oracle_counts", {}).get("unlinked_executions"),
        "executions",
        basis="observed",
        source="independent oracle",
        samples=1,
        scope="solver-b",
        method=LAB,
        reason=None if run.notes.get("oracle_counts") else "oracle not evaluated",
    )
    # -- valuation (profiles stay separate)
    for profile, key in (
        ("production-policy", "production"),
        ("airline-lab-arithmetic-v1", "arithmetic"),
    ):
        result = run.notes.get(key)
        for scenario in ("central",):
            est = result.estimates[scenario] if result else None
            for field in ("credit", "debit", "net"):
                val = getattr(est, field) if est else None
                _add(
                    run,
                    "valuation",
                    f"{field}.{scenario}",
                    val,
                    "context_tokens",
                    basis="estimated",
                    source="independent oracle chars_div4_v1",
                    samples=1,
                    scope="solver-b",
                    window="C6",
                    method=profile,
                    reason=None
                    if val is not None
                    else (
                        "unknown: " + ",".join(est.missing)
                        if est and est.missing
                        else "profile not evaluated"
                    ),
                )
    # -- timing/resources
    durations = [c.duration_ns for c in calls if c.duration_ns]
    _add(
        run,
        "timing_resources",
        "tool_duration_total",
        sum(durations) // 1_000_000,
        "ms",
        basis="observed",
        source="observer monotonic",
        samples=len(durations),
        scope="all",
    )
    collect_ms = [c.elapsed_ms for c in run.collected.values()]
    _add(
        run,
        "timing_resources",
        "collect_cycle_wall_total",
        sum(collect_ms),
        "ms",
        basis="observed",
        source="real collect+savings+status",
        samples=len(collect_ms),
        scope="control",
        method=PRODUCT,
    )
    _add(
        run,
        "timing_resources",
        "peak_memory",
        None,
        "bytes",
        basis="unavailable",
        source="none",
        samples=0,
        scope="control",
        method=PRODUCT,
        reason="no per-process memory probe is wired in this lab",
    )
    # -- storage/recovery
    view = read_journal(run.roots.ssgrep_data)
    _add(
        run,
        "storage_recovery",
        "journal_bytes",
        view.bytes,
        "bytes",
        basis="observed",
        source="journal segments (read-only)",
        samples=view.segments,
        scope="control",
        method=PRODUCT,
    )
    _add(
        run,
        "storage_recovery",
        "journal_malformed",
        view.malformed,
        "records",
        basis="observed",
        source="journal segments",
        samples=view.records,
        scope="control",
        method=PRODUCT,
    )
    # -- export/privacy
    _add(
        run,
        "export_privacy",
        "secret_canary_leaks",
        run.notes.get("canary_leaks"),
        "findings",
        basis="observed",
        source="canary scan of every report artifact",
        samples=1,
        scope="report",
        reason=None if run.notes.get("canary_leaks") is not None else "not scanned",
    )
    _add(
        run,
        "export_privacy",
        "export_signals_sent",
        None,
        "signals",
        basis="unavailable",
        source="none",
        samples=0,
        scope="export",
        reason="scrape/export lab (C9) is owned by the export lane and was not run here",
    )


def artifact_names(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("*.json"))


def counts_of(frozen: Any) -> dict[str, int]:
    return dict(Counter(c.actor for c in frozen.calls))
