"""Live-only metric provenance: native provider usage is never stand-in usage or savings."""

from __future__ import annotations

from .lab_context import Run
from .lab_journal import read_journal
from .lab_metrics import _add


def fill(run: Run) -> None:
    calls = run.frozen().calls
    for name, value, unit in (
        ("table_count", len(run.base.counts), "tables"),
        ("row_count", sum(run.base.counts.values()), "rows"),
    ):
        _add(
            run,
            "fixture_domain",
            name,
            value,
            unit,
            basis="observed",
            source="PostgreSQL seed",
            samples=1,
            scope="verifier",
            window="C0",
        )
    for actor, turns in (("solver-a", ("L2", "L1")), ("solver-b", ("L4", "L3"))):
        latest = next(
            (run.notes.get("turns", {}).get(t) for t in turns if t in run.notes.get("turns", {})),
            None,
        )
        native = latest["native"] if latest else {}
        usage = native.get("usage_total", {})
        samples = native.get("usage_records", 0)
        for category in ("input", "output", "cache_read", "cache_write", "total", "reasoning"):
            value = usage.get(category)
            _add(
                run,
                "tokens_money",
                f"provider_tokens.{category}",
                value,
                "tokens",
                basis="reported",
                source="native OMP provider usage records (latest full conversation)",
                samples=samples,
                scope=actor,
                method="native-provider-reported",
                reason="provider did not report this category" if value is None else None,
            )
        cost = native.get("cost_nano_usd")
        _add(
            run,
            "tokens_money",
            "runtime_reported_cost",
            cost,
            "nano_usd",
            basis="reported",
            source="native OMP usage.cost; Codex catalog estimate, not provider invoice; "
            "no ssgrep repricing",
            samples=samples,
            scope=actor,
            method="native-runtime-cost",
            reason="no native runtime cost record available" if cost is None else None,
        )
        _add(
            run,
            "tokens_money",
            "provider_billed_cost",
            None,
            "nano_usd",
            basis="unavailable",
            source="native OMP record",
            samples=samples,
            scope=actor,
            method="provider-billed-cost",
            reason="selected Codex runtime reports catalog estimates, not billed cost",
        )
        _add(
            run,
            "runtime_work",
            "user_turns",
            native.get("user_turns"),
            "turns",
            basis="observed",
            source="native OMP session",
            samples=1 if latest else 0,
            scope=actor,
            reason="no native session" if not latest else None,
        )
    for actor in ("solver-a", "solver-b", "verifier", "control"):
        for op in ("search", "show"):
            observed = [
                c
                for c in calls
                if c.actor == actor and c.tool == "ssgrep" and len(c.argv) > 1 and c.argv[1] == op
            ]
            _add(
                run,
                "retrieval",
                f"{op}_executions",
                len(observed),
                "calls",
                basis="observed",
                source="independent observer",
                samples=len(observed),
                scope=actor,
            )
    indexed = run.notes.get("index_a", {})
    _add(
        run,
        "persistence_index",
        "a_index_wall",
        indexed.get("ms"),
        "ms",
        basis="observed",
        source="real offline ssgrep index",
        samples=1 if indexed else 0,
        scope="verifier",
        reason="A persistence barrier not reached" if not indexed else None,
    )
    candidates = run.notes.get("reuse_candidates")
    _add(
        run,
        "attribution",
        "supported_reuse_candidates",
        len(candidates) if candidates is not None else None,
        "candidates",
        basis="observed",
        source="independent evidence oracle",
        samples=1,
        scope="solver-b",
        method="airline-lab-arithmetic-v1",
        reason="reuse oracle not evaluated" if candidates is None else None,
    )
    for name in ("production", "arithmetic"):
        result = run.notes.get(name)
        for field in ("credit", "debit", "net"):
            estimate = result.estimates["central"] if result else None
            value = getattr(estimate, field) if estimate else None
            _add(
                run,
                "valuation",
                f"{field}.central",
                value,
                "context_tokens",
                basis="estimated",
                source="independent oracle; not a causal A-vs-B comparison",
                samples=1,
                scope="solver-b",
                method=result.profile if result else name,
                reason="profile unavailable or evidence incomplete" if value is None else None,
            )
    _add(
        run,
        "timing_resources",
        "tool_duration",
        sum(c.duration_ns or 0 for c in calls) // 1_000_000,
        "ms",
        basis="observed",
        source="observer monotonic durations",
        samples=len(calls),
        scope="all",
    )
    journal = read_journal(run.roots.ssgrep_data)
    _add(
        run,
        "storage_recovery",
        "journal_bytes",
        journal.bytes,
        "bytes",
        basis="observed",
        source="private journal segments",
        samples=journal.segments,
        scope="control",
    )
    _add(
        run,
        "export_privacy",
        "scrape_gate",
        run.report.gate_status("scrape"),
        "status",
        basis="observed",
        source="C9 actual HTTP/OTLP receiver gate",
        samples=1,
        scope="control",
    )
