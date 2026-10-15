"""Compact duplicate-origin ownership; metadata never becomes another exposure."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from ssgrep.accounting.contracts import PERSONAL_PROFILE, RawEvidence, digest_of

CLAIM = "ownership_claim"


def owner(item: RawEvidence) -> tuple[str, str, str, str | None, str | None]:
    cwd, parent = item.payload.get("cwd"), item.payload.get("parent_session")
    return (
        item.source_id,
        item.session_id,
        item.branch_id,
        cwd if isinstance(cwd, str) else None,
        parent if isinstance(parent, str) else None,
    )


def claimed_owner(item: RawEvidence) -> tuple[str, str, str, str | None, str | None]:
    p = item.payload
    return (
        p.get("claimant_source_id", item.source_id),
        p.get("claimant_session_id", item.session_id),
        p.get("claimant_branch_id", item.branch_id),
        p.get("claimant_cwd"),
        p.get("claimant_parent_session"),
    )


def target_key(item: RawEvidence) -> tuple[str, str, str] | None:
    value = item.payload.get("target_native_key")
    return (
        tuple(value)
        if isinstance(value, list) and len(value) == 3 and all(isinstance(v, str) for v in value)
        else None
    )


class OwnershipClaims:
    def __init__(self) -> None:
        self.first: dict[bytes, str] = {}
        self.extra: dict[bytes, set[str]] = {}
        self.conflicts: dict[bytes, dict[str, RawEvidence]] = {}

    @staticmethod
    def signature(item: RawEvidence, *, claimed: bool = False) -> str:
        value = claimed_owner(item) if claimed else owner(item)
        linked = item.payload.get("call_record_id")
        if isinstance(linked, str) and item.payload.get("call_link_status") not in {
            "ambiguous",
            "missing_context",
        }:
            return digest_of([value, linked])
        return digest_of(value)

    def note(self, item: RawEvidence, key: bytes) -> None:
        if item.kind == CLAIM:
            target = target_key(item)
            if target is None:
                return
            key = hashlib.blake2b("\0".join(target).encode(), digest_size=16).digest()
            signature = self.signature(item, claimed=True)
            if item.payload.get("origin_conflict") is True:
                signature = digest_of([signature, item.payload.get("target_digest")])
                conflict = self.conflict_evidence(item)
                if conflict is not None:
                    self.conflicts.setdefault(key, {})[conflict.digest] = conflict
            self.extra.setdefault(key, set()).add(signature)
        else:
            self.first.setdefault(key, self.signature(item))

    def take(self, item: RawEvidence, key: bytes, *, conflict: bool = False) -> RawEvidence | None:
        signature = self.signature(item)
        if conflict:
            signature = digest_of([signature, item.digest])
        if self.first.get(key) == signature or signature in self.extra.get(key, ()):
            return None
        self.extra.setdefault(key, set()).add(signature)
        source, session, branch, cwd, parent = owner(item)
        payload = {
            "target_native_key": list(item.native_key),
            "original_kind": item.kind,
            "target_record_id": item.record_id,
            "target_digest": item.digest,
            "call_record_id": item.payload.get("call_record_id"),
            "call_link_status": item.payload.get("call_link_status"),
            "claimant_source_id": source,
            "claimant_session_id": session,
            "claimant_branch_id": branch,
            "claimant_cwd": cwd,
            "claimant_parent_session": parent,
        }
        if conflict:
            payload.update(target_digest=item.digest, origin_conflict=True)
        record_id = "ownership:" + digest_of(payload)
        return replace(
            item,
            kind=CLAIM,
            record_id=record_id,
            digest=digest_of(payload),
            payload=payload,
            origin_domain="native_ownership_claim_v1",
            origin_id=record_id,
            origin_status="metadata_claim",
        )

    @staticmethod
    def conflict_evidence(item: RawEvidence) -> RawEvidence | None:
        if item.kind != CLAIM or item.payload.get("origin_conflict") is not True:
            return None
        target, digest = target_key(item), item.payload.get("target_digest")
        if target is None or not isinstance(digest, str):
            return None
        return replace(
            item,
            runtime=target[0],
            session_id=target[1],
            record_id=target[2],
            origin_domain=target[1],
            origin_id=target[2],
            digest=digest,
            kind=item.payload.get("original_kind", "unknown"),
            payload={},
        )


def native_in_scope(record: Any, timelines: Any, scoper: Any) -> bool:
    """Unknown origin ownership is local-observable, never organization-authoritative."""
    personal = scoper.policy.profile_id == PERSONAL_PROFILE and not scoper.policy.roots
    if not personal and (
        record.identity in timelines.ownership_unknown
        or record.identity in timelines.order_unknown
        or record.ev.origin_status
        in {"missing_session_id", "missing_entry_id", "legacy_unverified"}
    ):
        return False
    claims = timelines.ownership_claims.get(record.identity, ())
    if claims and not personal:
        return all(scoper.in_scope(claim[3]) for claim in claims)
    cwd = record.ev.payload.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        cwd = timelines.sessions[(record.ev.runtime, record.ev.session_id)].cwd_at(record.seq)
    return scoper.in_scope(cwd)
