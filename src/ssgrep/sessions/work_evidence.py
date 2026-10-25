"""Explicit, bounded, read-only evidence recovery for one indexed episode.

Claude JSONL and Codex/Pi/omp native JSONL are supported. Codex token-count
bookkeeping is intentionally not attributed: session totals and repeated last-
request counts do not prove an episode's usage. Missing evidence stays absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from ssgrep.sessions.adapters import codex, omp, pi
from ssgrep.sessions.adapters.base import SourceFingerprint, TranscriptSource
from ssgrep.sessions.episodes import segment_episode_groups
from ssgrep.utilities.types import EpisodeDetail, SessionFile

_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_LINE_BYTES = 500_000
_USAGE = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_input_tokens": "cache_read_tokens",
    "cache_creation_input_tokens": "cache_write_tokens",
    "reasoning_tokens": "reasoning_tokens",
}
_PI_USAGE = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cacheRead": "cache_read_tokens",
    "cacheWrite": "cache_write_tokens",
}


def _read(path: Path) -> tuple[dict, ...] | None:
    # Nonblocking open prevents FIFOs from hanging before fstat; NOFOLLOW also
    # closes the final-component symlink race. No discovery or filesystem writes.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_SOURCE_BYTES:
            return None
        data = stream.read(_MAX_SOURCE_BYTES + 1)
        after = os.fstat(stream.fileno())
    current = os.stat(path, follow_symlinks=False)
    if len(data) > _MAX_SOURCE_BYTES or any(
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        for item in (after, current)
    ):
        return None
    parsed = []
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            return None  # incomplete source cannot prove final boundaries
        if not line.strip():
            continue
        if len(line) > _MAX_LINE_BYTES:
            return None
        value = json.loads(line)
        if not isinstance(value, dict):
            return None
        parsed.append(value)
    return tuple(parsed)


def _normalize(raw: tuple[dict, ...], detail: EpisodeDetail, source_path: str) -> list[dict] | None:
    runtime = detail.runtime
    if runtime == "claude":
        expected = detail.parent_session_id if detail.is_subagent else detail.session_id
        identities = {r.get("sessionId") for r in raw if r.get("sessionId")}
        if identities != {expected}:
            return None
        return list(raw)
    header_type = "session_meta" if runtime == "codex" else "session"
    header_ids = {
        (record.get("payload", {}) if runtime == "codex" else record).get("id")
        for record in raw
        if record.get("type") == header_type
    }
    if len(header_ids) != 1 or any(not isinstance(value, str) or not value for value in header_ids):
        return None
    inspector = {"codex": codex._inspect, "pi": pi._inspect, "omp": omp._inspect}[runtime]
    info = inspector(raw, runtime)
    if info is None or info.session_id != detail.session_id:
        return None
    # Never seed normalization from the discovery model: that is the session's
    # last model, which can leak future turns into an older selected episode.
    source = TranscriptSource(
        runtime,
        source_path,
        SessionFile(Path(source_path), detail.session_id, not detail.is_subagent, runtime=runtime),
        SourceFingerprint(0, 0, ""),
    )
    if runtime == "codex":
        return list(codex._normalize(raw, source))
    normalized, _ = pi._normalize(raw, source)
    originals: dict[str, list[dict]] = {}
    provider = None
    for record in raw:
        if record.get("type") == "model_change":
            provider = record.get("provider")
        if record.get("type") != "message" or not isinstance(record.get("message"), dict):
            continue
        message = record["message"]
        key = f"{runtime}:{record.get('id')}"
        original = dict(message)
        original["provider"] = message.get("provider", provider)
        originals.setdefault(key, []).append(original)
    for record in normalized:
        matches = originals.get(record.get("uuid"), [])
        if not matches or record.get("type") != "assistant":
            continue
        original = matches.pop(0)
        message = record["message"]
        message.pop("provider", None)
        if original.get("provider"):
            message["provider"] = original["provider"]
        for field in ("usage", "stopReason"):
            if field in original:
                message[field] = original[field]
    return list(normalized)


def _usage(message: dict, runtime: str) -> dict | None:
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    fields = _PI_USAGE if runtime in ("pi", "omp") else _USAGE
    result = {}
    for source, target in fields.items():
        if source not in usage:
            continue
        value = usage[source]
        if type(value) is not int or value < 0:
            return None
        result[target] = value
    return result or None


def _contributions(group: list[dict], runtime: str) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    identities: dict[str, tuple] = {}
    conflicting_ids = set()
    for record in group:
        if record.get("type") != "assistant":
            continue
        message = record.get("message", {})
        model, provider = message.get("model"), message.get("provider")
        if model == "<synthetic>":
            continue
        model = model if isinstance(model, str) and model.strip() else None
        provider = provider if isinstance(provider, str) and provider.strip() else None
        buckets.setdefault((model, provider), []).append(record)
        identity = message.get("id") or record.get("uuid")
        if isinstance(identity, str) and identity:
            key = (model, provider)
            if identity in identities and identities[identity] != key:
                conflicting_ids.add(identity)
            identities[identity] = key
    result = []
    for (model, provider), records in buckets.items():
        contribution = {"basis": "transcript_inferred"}
        if model:
            contribution["model"] = model
        if provider:
            contribution["provider"] = provider
        messages: dict[str, dict] = {}
        known = runtime != "codex"
        for record in records:
            message = record.get("message", {})
            identity = message.get("id") or record.get("uuid")
            usage = _usage(message, runtime)
            if (
                not isinstance(identity, str)
                or not identity
                or usage is None
                or identity in conflicting_ids
            ):
                known = False
                continue
            previous = messages.get(identity)
            # Repeated message IDs are cumulative snapshots, not extra calls.
            # Conflicting/decreasing counters are ambiguous, so omit the total.
            if previous is not None and (
                previous.keys() != usage.keys()
                or any(usage[key] < value for key, value in previous.items())
            ):
                known = False
            messages[identity] = usage
        if known and messages:
            common = set.intersection(*(set(value) for value in messages.values()))
            if common:
                contribution["usage"] = {
                    key: sum(value[key] for value in messages.values()) for key in sorted(common)
                }
        if len(contribution) > 1:
            result.append(contribution)
    return result


def _elapsed(group: list[dict], runtime: str) -> int | None:
    messages = [r for r in group if r.get("type") in ("user", "assistant")]
    if runtime == "codex" or not messages or messages[0].get("type") != "user":
        return None
    last = messages[-1]
    message = last.get("message", {})
    if message.get("model") == "<synthetic>":
        return None
    if last.get("type") != "assistant" or not (
        message.get("stop_reason") == "end_turn" or message.get("stopReason") == "stop"
    ):
        return None
    try:
        timestamps = [
            datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in messages
        ]
        if any(t.tzinfo is None for t in timestamps):
            return None
        if any(b < a for a, b in pairwise(timestamps)):
            return None
        seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        return int(seconds)  # Whole elapsed seconds, rounded down.
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return None


def infer_work(detail: EpisodeDetail) -> dict | None:
    """Recover only evidence bound to this exact, untruncated indexed episode.

    None means unsupported, unsafe/unreadable, mismatched or evidence-free.
    Individual unknown measurements are omitted, never replaced with zero.
    """
    if (
        detail.runtime not in ("claude", "codex", "pi", "omp")
        or not detail.source_path
        or detail.prompt_truncated
        or detail.response_truncated
    ):
        return None
    try:
        raw = _read(Path(detail.source_path))
        if raw is None:
            return None
        normalized = _normalize(raw, detail, detail.source_path)
        if normalized is None:
            return None
        from ssgrep.pipeline.episodes import extract_episode_text
        from ssgrep.sessions.work import validate_work

        for episode, group in segment_episode_groups(normalized, detail.session_id):
            if episode.episode_id != detail.episode_id:
                continue
            prompt, response = extract_episode_text(group)
            if (prompt or episode.prompt_text, response or episode.response_text) != (
                detail.prompt_text,
                detail.response_text,
            ):
                return None
            contributions = _contributions(group, detail.runtime)
            if not contributions:
                return None
            identity = json.dumps([detail.runtime, detail.episode_id], separators=(",", ":"))
            work = {
                "work_id": "episode:" + hashlib.sha256(identity.encode()).hexdigest(),
                "basis": "transcript_inferred",
                "source_ref": detail.episode_id,
            }
            if len(contributions) == 1:
                work.update(contributions[0])
            else:
                work["contributions"] = contributions
            elapsed = _elapsed(group, detail.runtime)
            if elapsed is not None:
                work["elapsed_seconds"] = elapsed
            return validate_work(work)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return None
    return None
