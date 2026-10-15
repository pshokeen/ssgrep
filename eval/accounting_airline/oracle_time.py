"""Independent raw journal clocks reconciled against supervisor dispatch envelopes.

Journal rows are subjects, not observer evidence: each timing must uniquely fit an
actual observed dispatch. Transcript wall gaps are a separate 300-second model.
"""

from __future__ import annotations

import hmac
import json
import os
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from .oracle_estimates import bounded_estimate, source_work


def lookup_arguments(argv):
    """Read positional query/ref and explicit where without trusting shell text."""
    positional, where = [], None
    args = iter(argv[2:])
    for arg in args:
        if arg == "--":
            positional.extend(args)
            break
        name, equal, value = arg.partition("=")
        if name in ("--where", "-w", "--limit", "-l", "--token-budget"):
            value = value if equal else next(args, None)
            if value is None:
                return None
            if name in ("--where", "-w"):
                where = value
        elif arg in ("--json", "--quiet", "--no-color"):
            continue
        elif arg.startswith("-"):
            return None
        else:
            positional.append(arg)
    return (positional[0], where) if len(positional) == 1 else None


def read_timing_events(data: Path, calls=()) -> tuple[dict, ...]:
    fingerprints = {}
    key_path = data / "accounting" / "personal-local" / "fingerprint.key"
    try:
        fd = os.open(key_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            key = os.read(fd, 33)
        finally:
            os.close(fd)
        if len(key) == 32:
            for call in calls:
                if (
                    call.tool != "ssgrep"
                    or len(call.argv) < 3
                    or call.argv[1] not in ("search", "show")
                ):
                    continue

                def normalized(text):
                    return unicodedata.normalize("NFC", text).strip() or None

                args = call.argv
                parsed = lookup_arguments(args)
                if parsed is None:
                    continue
                query, where = parsed
                message = json.dumps(
                    {
                        "v": "fp1",
                        "op": args[1],
                        "q": normalized(query),
                        "w": normalized(where) if where is not None else None,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
                fingerprints[call.call_id] = hmac.digest(key, message, "sha256").hex()
    except OSError:
        pass  # A missing key cannot authorize a join, but raw unknown clocks stay visible.
    events = []
    for path in sorted((data / "accounting" / "personal-local" / "journal").glob("seg-*.jsonl")):
        with path.open("rb") as handle:
            while line := handle.readline(1000001):
                if len(line) > 1000000:
                    raise ValueError("oversized journal timing record")
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("operation") in ("search", "show"):
                    event["_observer_call_ids"] = (
                        [
                            identity
                            for identity, fingerprint in fingerprints.items()
                            if fingerprint == event.get("fingerprint")
                        ]
                        if calls
                        else None
                    )
                    events.append(event)
    return tuple(events)


def utc_nanoseconds(stamp) -> tuple[int | None, str]:
    """Parse explicit UTC/offset evidence with integer arithmetic, never local timezone."""
    try:
        value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if value.tzinfo is None or value.utcoffset() is None:
            return None, "journal_utc_timezone_missing"
        delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
        return ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 1000, ""
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None, "journal_utc_clock_invalid"


def reconcile_timing(scan, events):
    """Return unique raw service/model nanoseconds; no intermediate time quantization."""
    calls = {d.call_id: d for inv in scan.invocations if inv.is_lookup for d in inv.dispatches}
    calls.update(
        {d.call_id: d for d in scan.unlinked if len(d.argv) > 1 and d.argv[1] in ("search", "show")}
    )
    grouped = {}
    for event in events:
        linked = event.get("_observer_call_ids")
        if linked is not None and not set(linked).intersection(calls):
            continue
        grouped.setdefault(event.get("execution_id"), {}).setdefault(event.get("phase"), []).append(
            dict(event)
        )
    assigned, reasons, service = {}, [], 0
    for identity, phases in grouped.items():
        done = {json.dumps(e, sort_keys=True): e for e in phases.get("completed", [])}
        identities = {
            (e.get("operation"), e.get("transport"), e.get("profile"), e.get("fingerprint"))
            for group in phases.values()
            for e in group
        }
        if (
            len(identities) != 1
            or len(done) != 1
            or any(
                len({json.dumps(e, sort_keys=True) for e in rows}) != 1 for rows in phases.values()
            )
        ):
            reasons.append("conflicting_journal_timing")
            continue
        event = next(iter(done.values()))
        stamp, clock_reason = utc_nanoseconds(event.get("ts_utc"))
        duration = event.get("mono_ns")
        if clock_reason:
            reasons.append(clock_reason)
            continue
        assert stamp is not None  # utc_nanoseconds succeeded without a clock reason.
        if type(duration) is not int or duration < 0:
            reasons.append("journal_monotonic_duration_missing")
            continue
        # Journal ts_utc is quantized to whole seconds. Intersect its uncertainty
        # interval with the independently observed dispatch/finish envelope.
        hits = [
            call
            for call in calls.values()
            if call.argv[1] == event.get("operation")
            and (
                event.get("_observer_call_ids") is None
                or call.call_id in event["_observer_call_ids"]
            )
            and type(call.duration_ns) is int
            and call.duration_ns >= 0
            and max(stamp, call.wall_ns + duration)
            <= min(stamp + 999999999, call.wall_ns + call.duration_ns)
        ]
        if len(hits) != 1:
            continue  # verifier/control clocks do not enter solver valuation
        assigned.setdefault(hits[0].call_id, []).append((identity, duration))
    work = 0
    for call in calls.values():
        matches = assigned.get(call.call_id, [])
        if len(matches) != 1:
            reasons.append("no_unique_observed_journal_timing")
        else:
            service += matches[0][1]
            work += min(matches[0][1], 300000000000)
    return service, work, tuple(sorted(set(reasons)))


def time_estimates(scan, verdicts, events):
    service_ns, debit_ns, timing_missing = reconcile_timing(scan, events)
    rows = {}
    for band, scenario in enumerate(("conservative", "central", "optimistic")):
        by_task = {}
        missing = [f"debit:{reason}" for reason in timing_missing]
        for verdict in verdicts:
            if (
                (verdict.eligible or verdict.reason == "rarity_unknown")
                and verdict.reuse == "action_reference"
                and not verdict.proven_bands[band]
            ):
                missing.append("credit:rarity_bound_unproved")
            if verdict.reuse == "action_reference" and verdict.reason in (
                "source_atom_unavailable",
                "source_unresolved",
                "source_time_unknown",
                "source_evidence_conflict",
                "prior_context_conflict",
            ):
                missing.append(f"credit:{verdict.reason}")
            if verdict.proven_bands[band]:
                by_task.setdefault(verdict.task, []).extend(verdict.atoms[scenario])
        credit_ms, source_missing = source_work(scan, by_task)
        if source_missing:
            missing.append("credit:source_seconds_unavailable")
        if not scan.invocations and not scan.unlinked:
            missing.extend(("credit:no_valuation_evidence", "debit:no_valuation_evidence"))
        missing = tuple(sorted(set(missing)))
        rows[scenario] = bounded_estimate(
            scenario,
            credit_ms * 1000000,
            any(x.startswith("credit:") for x in missing),
            debit_ns,
            debit_ns,
            None if any(x.startswith("debit:") for x in missing) else debit_ns,
            1000000000,
            missing,
        )
    return rows, {
        "measured_service_ns": None if timing_missing else service_ns,
        "known_service_ns": service_ns,
        "active_proxy_ns": debit_ns,
        "measured_service_ms": None if timing_missing else service_ns // 1000000,
        "known_service_ms": service_ns // 1000000,
        "complete": not timing_missing,
        "mapping": "journal_active_proxy_ns_v2",
        "raw_unit": "work_ns",
        "source_precision": "native_milliseconds",
        "debit_precision": "journal_monotonic_nanoseconds",
        "source_method": "capped_gap_v1",
        "missing": list(timing_missing),
    }
