"""Separately consented, expiring byte-bounded plaintext; never accounting authority."""

from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import replace

from ssgrep.accounting import storage
from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.evidence_codec import evidence_from_line, evidence_line
from ssgrep.accounting.ledger_contracts import ProfilePolicy

CACHE_DIR = "cache"
MAX_CACHE_BYTES = 32 * 1024 * 1024


def _name(item: RawEvidence) -> str:
    return (
        hashlib.sha256((item.native_identity + "\0" + item.digest).encode()).hexdigest() + ".json"
    )


def _entries(fd: int, policy: ProfilePolicy, now: int) -> list[tuple[int, str, int]]:
    """One inventory: expire/trim by metadata before reading retained content once."""
    from ssgrep.accounting.scope_events import Scoper

    candidates = []
    for name in storage.list_names(fd):
        try:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        expired = info.st_mtime + policy.evidence_cache_retention_days * 86400 <= now
        if not policy.evidence_cache_enabled or expired or name.endswith(".tmp"):
            storage.delete_file(fd, name)
        else:
            candidates.append((info.st_mtime_ns, name, info.st_size))
    ordered = deque(sorted(candidates))
    size = sum(row[2] for row in ordered)
    while ordered and size > policy.evidence_cache_max_bytes:
        _, name, count = ordered.popleft()
        storage.delete_file(fd, name)
        size -= count
    scoper, entries = Scoper(policy), []
    for row in ordered:
        _, name, _ = row
        try:
            raw = storage.read_file(fd, name, policy.evidence_cache_max_bytes)
            item = evidence_from_line(raw) if raw is not None else None
            authorized = item is not None and scoper.in_scope(item.payload.get("cwd"))
        except (ValueError, KeyError, TypeError):
            authorized = False
        if authorized:
            entries.append(row)
        else:
            storage.delete_file(fd, name)
    return entries


def _bounded_policy(policy: ProfilePolicy) -> ProfilePolicy:
    if (
        not policy.collection_enabled
        or not policy.evidence_cache_enabled
        or not 1 <= policy.evidence_cache_retention_days <= 30
        or not 1 <= policy.evidence_cache_max_bytes <= MAX_CACHE_BYTES
    ):
        return replace(
            policy,
            evidence_cache_enabled=False,
            evidence_cache_retention_days=0,
            evidence_cache_max_bytes=0,
        )
    return policy


def maintain_cache(
    profile: str, policy: ProfilePolicy | None = None, *, now: int | None = None
) -> ProfilePolicy:
    with CacheWriter(profile, policy, now=now, create=False) as writer:
        return writer.policy


class CacheWriter:
    """One bounded metadata inventory per fresh collection batch, not per record."""

    def __init__(
        self,
        profile: str,
        policy: ProfilePolicy | None = None,
        *,
        now: int | None = None,
        create: bool = True,
    ) -> None:
        from ssgrep.accounting.policy import load_profile_policy, policy_path

        self.fd = None
        self._consent_lock = None
        self.policy = _bounded_policy(policy or load_profile_policy(profile))
        try:
            try:
                self.fd = storage.open_profile_dir(profile, "collector", CACHE_DIR, create=False)
            except FileNotFoundError:
                pass
            if self.fd is not None or self.policy.evidence_cache_enabled:
                from ssgrep.accounting import plainfile

                consent_lock = plainfile.exclusive_lock(policy_path().parent.parent, "accounting")
                consent_lock.__enter__()
                self._consent_lock = consent_lock
                self.policy = _bounded_policy(load_profile_policy(profile))
                if self.fd is None and create and self.policy.evidence_cache_enabled:
                    self.fd = storage.open_profile_dir(profile, "collector", CACHE_DIR)
            from ssgrep.accounting.scope_events import Scoper

            self.scoper = Scoper(self.policy)
            self.entries = (
                deque(_entries(self.fd, self.policy, int(time.time()) if now is None else now))
                if self.fd is not None
                else deque()
            )
            self.names = {row[1] for row in self.entries}
            self.size = sum(row[2] for row in self.entries)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> CacheWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def add(self, item: RawEvidence) -> None:
        if (
            self.fd is None
            or not self.policy.evidence_cache_enabled
            or self.policy.paused
            or "_evidence_format" in item.payload
        ):
            return
        if not self.scoper.in_scope(item.payload.get("cwd")):
            return
        expired_before = (
            time.time_ns() - self.policy.evidence_cache_retention_days * 86400 * 1_000_000_000
        )
        while self.entries and self.entries[0][0] <= expired_before:
            _, old, count = self.entries.popleft()
            storage.delete_file(self.fd, old)
            self.names.discard(old)
            self.size -= count
        body = evidence_line(item, plaintext=True) + b"\n"
        name = _name(item)
        if len(body) > self.policy.evidence_cache_max_bytes or name in self.names:
            return
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            while self.entries and self.size + len(body) > self.policy.evidence_cache_max_bytes:
                _, old, count = self.entries.popleft()
                storage.delete_file(self.fd, old)
                self.names.discard(old)
                self.size -= count
        if not storage.write_once(self.fd, name, body):
            self.entries = deque(_entries(self.fd, self.policy, int(time.time())))
            self.names = {row[1] for row in self.entries}
            self.size = sum(row[2] for row in self.entries)
            return
        info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        self.entries.append((info.st_mtime_ns, name, len(body)))
        self.names.add(name)
        self.size += len(body)

    def close(self) -> None:
        try:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
        finally:
            if self._consent_lock is not None:
                self._consent_lock.__exit__(None, None, None)
                self._consent_lock = None


def cache_collected_evidence(profile: str, items: Sequence[RawEvidence]) -> None:
    """Admit only newly accepted native records at the live/backfill read boundary."""
    if not items:
        return
    with CacheWriter(profile) as writer:
        for item in items:
            writer.add(item)


def cached_evidence(profile: str, item: RawEvidence) -> RawEvidence | None:
    from ssgrep.accounting.policy import load_profile_policy

    policy = load_profile_policy(profile)
    if (
        not policy.collection_enabled
        or policy.paused
        or not policy.evidence_cache_enabled
        or not 1 <= policy.evidence_cache_retention_days <= 30
        or not 1 <= policy.evidence_cache_max_bytes <= MAX_CACHE_BYTES
    ):
        return None
    from ssgrep.accounting.scope_events import Scoper

    if not Scoper(policy).in_scope(item.payload.get("cwd")):
        return None
    try:
        fd = storage.open_profile_dir(profile, "collector", CACHE_DIR, create=False)
    except FileNotFoundError:
        return None
    try:
        try:
            info = os.stat(_name(item), dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if info.st_mtime + policy.evidence_cache_retention_days * 86400 <= time.time():
            storage.delete_file(fd, _name(item))
            return None
        try:
            raw = storage.read_file(fd, _name(item), policy.evidence_cache_max_bytes)
            if raw is None:
                return None
            candidate = evidence_from_line(raw)
            candidate = replace(
                candidate,
                payload={k: v for k, v in candidate.payload.items() if k != "_evidence_format"},
            )
            from ssgrep.accounting.native_identity import with_origin
            from ssgrep.accounting.readers.values import evidence_digest

            if candidate.origin_domain is not None and candidate.origin_id is not None:
                digest = with_origin(
                    candidate, candidate.origin_domain, candidate.origin_id, candidate.origin_status
                ).digest
            else:
                digest = evidence_digest(candidate.kind, candidate.payload)
            if (
                candidate.identity != item.identity
                or digest != item.digest
                or candidate.ts_utc != item.ts_utc
            ):
                storage.delete_file(fd, _name(item))
                return None
        except (ValueError, KeyError, TypeError):
            storage.delete_file(fd, _name(item))
            return None
    finally:
        os.close(fd)
    return candidate
