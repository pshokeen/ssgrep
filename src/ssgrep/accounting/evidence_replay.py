"""Hydrate one compact reference from separately authorized originals or explicit cache."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.evidence_codec import COMPACT

TEXT_KINDS = frozenset({"user_turn", "tool_call", "tool_result"})


@dataclass(frozen=True)
class ReplayUnavailable:
    runtime: str
    session_id: str
    source_id: str
    reason: str
    native_identity: str


class EvidenceReplay:
    def __init__(self, profile: str, *, adapter: Any = None, env: Any = None) -> None:
        self.profile = profile
        self.adapter = adapter
        self.env = env
        self.unavailable: dict[tuple[str, str], str] = {}
        self.failures: list[ReplayUnavailable] = []

    def hydrate(self, item: RawEvidence) -> RawEvidence:
        if item.payload.get("_evidence_format") != COMPACT or item.kind not in TEXT_KINDS:
            return item
        from ssgrep.accounting.evidence_cache import cached_evidence

        cached = cached_evidence(self.profile, item)
        if cached is not None:
            return replace(
                cached, payload={k: v for k, v in cached.payload.items() if k != "_evidence_format"}
            )
        if self.adapter is None:
            from ssgrep.accounting.policy import load_profile_policy
            from ssgrep.accounting.reduce_io import native_source_adapter

            policy = load_profile_policy(self.profile)
            if not policy.collection_enabled or policy.paused:
                return self._missing(item, "source_unapproved")
            self.adapter = native_source_adapter(
                policy, self.env, now_ms=time.time_ns() // 1_000_000
            )
        from ssgrep.accounting.native_source import NativeSourceReference

        prefix = item.runtime + ":"
        if not item.source_id.startswith(prefix):
            return self._missing(item, "source_unresolved")
        reference = NativeSourceReference(
            item.source_id,
            item.runtime,
            item.session_id,
            Path(item.source_id[len(prefix) :]),
            record_digests=((item.native_identity, item.digest),),
            cwd=item.payload.get("cwd"),
        )
        result = self.adapter.read_record(
            reference, item.native_identity, item.digest, locator=item.locator
        )
        return self._accepted(item, result)

    def _accepted(self, item, result):
        if result.evidence is not None and result.evidence.identity == item.identity:
            if result.evidence.ts_utc != item.ts_utc:
                return self._missing(item, "source_timestamp_drift")
            if (result.evidence.runtime, result.evidence.session_id, result.evidence.branch_id) != (
                item.runtime,
                item.session_id,
                item.branch_id,
            ):
                return self._missing(item, "source_ownership_drift")
            for field in ("cwd", "parent_session"):
                if (
                    field in item.payload
                    and result.evidence.payload.get(field) != item.payload[field]
                ):
                    return self._missing(item, "source_ownership_drift")
            payload = dict(result.evidence.payload)
            for name in ("call_record_id", "call_link_status", "occurrence"):
                if name in item.payload:
                    payload[name] = item.payload[name]
            return replace(result.evidence, payload=payload, ts_utc=item.ts_utc)
        reason = getattr(result.status, "value", str(result.status))
        if reason == "ok":
            reason = "source_identity_mismatch"
        return self._missing(item, reason)

    def hydrate_many(self, items) -> list[RawEvidence]:
        """Batch only selected affected sessions; default storage still has no text."""
        from ssgrep.accounting.evidence_cache import cached_evidence
        from ssgrep.accounting.evidence_replay_batch import read_batch
        from ssgrep.accounting.native_source import NativeSourceReference
        from ssgrep.accounting.policy import load_profile_policy
        from ssgrep.accounting.reduce_io import native_source_adapter

        items = list(items)
        result = list(items)
        policy = load_profile_policy(self.profile)
        if self.adapter is None and (not policy.collection_enabled or policy.paused):
            return [
                self._missing(item, "source_unapproved")
                if item.payload.get("_evidence_format") == COMPACT and item.kind in TEXT_KINDS
                else item
                for item in items
            ]
        groups = defaultdict(list)
        for index, item in enumerate(items):
            if item.payload.get("_evidence_format") != COMPACT or item.kind not in TEXT_KINDS:
                continue
            cached = cached_evidence(self.profile, item) if policy.evidence_cache_enabled else None
            if cached is not None:
                result[index] = replace(
                    cached,
                    payload={k: v for k, v in cached.payload.items() if k != "_evidence_format"},
                )
                continue
            if not item.source_id.startswith(item.runtime + ":"):
                result[index] = self._missing(item, "source_unresolved")
                continue
            groups[(item.runtime, item.session_id, item.source_id)].append((index, item))
        if not groups:
            return result
        if self.adapter is None:
            self.adapter = native_source_adapter(
                policy, self.env, now_ms=time.time_ns() // 1_000_000
            )
        for (runtime, session, source), rows in sorted(groups.items()):
            selected = [item for _, item in rows]
            reference = NativeSourceReference(
                source,
                runtime,
                session,
                Path(source[len(runtime) + 1 :]),
                cwd=selected[0].payload.get("cwd"),
            )
            hydrated = read_batch(self.adapter, reference, selected)
            for (index, item), native in zip(rows, hydrated, strict=True):
                result[index] = self._accepted(item, native)
        return result

    def _missing(self, item: RawEvidence, reason: str) -> RawEvidence:
        self.unavailable[(item.runtime, item.session_id)] = reason
        self.failures.append(
            ReplayUnavailable(
                item.runtime, item.session_id, item.source_id, reason, item.native_identity
            )
        )
        return replace(item, payload={**item.payload, "_replay_unavailable": reason})
