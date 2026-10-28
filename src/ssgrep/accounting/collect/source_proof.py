"""Idle proofs for native sources: a cycle skips a source only when it provably did not change.

A JSONL-family source is skipped while its stat signature equals the one stored with its
last complete read. A signature stamped within one timestamp granule of its capture
(``racy``) cannot prove that alone, so that read also stores the SHA-256 of the bytes and
later cycles compare bytes too. A cycle hashes each source at most once, charges every
hashed byte to its read budget, and drops the digest as soon as the stored stamps are
settled relative to the cycle's own clock reading: from then on the signature alone is
proof. A database source is skipped only on its accepted census generation.
"""

from __future__ import annotations

import os
import stat
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.racy import racy
from ssgrep.accounting.readers.base import Source, source_digest
from ssgrep.accounting.readers.opencode import OpenCodeReader

if TYPE_CHECKING:
    from ssgrep.accounting.collect.read import ReadOutcome
    from ssgrep.accounting.collect.store import SourceRecord

SIGNATURE = "collector_source_signature"
DIGEST = "collector_source_digest"


def source_signature(reader: Any, source: Source) -> list | None:
    """Content-free descriptor metadata; DB writes can live only in its WAL."""
    paths = [source.path]
    if isinstance(reader, OpenCodeReader):
        paths.append(source.path.with_name(source.path.name + "-wal"))
    signature = []
    for path in paths:
        try:
            info = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            signature.append(None)
            continue
        except OSError:
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        signature.append(
            [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        )
    return signature if signature[0] is not None else None


def accepted_database_generation(extra: dict) -> dict | None:
    """Only the reader's accepted v3 clean census can authorize an idle skip."""
    stats = extra.get("opencode", {})
    generation = stats.get("source_generation")
    epoch = extra.get("census_epoch", {})
    census = extra.get("census", {})
    if (
        stats.get("cursor_version") != 3
        or stats.get("status") != "complete"
        or stats.get("snapshot_checked") is not True
        or stats.get("discovery_pending")
        or stats.get("known_pending_lower_bound", 0)
        or not isinstance(generation, dict)
        or epoch.get("dirty") is not False
        or epoch.get("generation") != generation
        or not all(
            census.get(table, {}).get("done") is True for table in ("session", "message", "part")
        )
    ):
        return None
    return generation


def _stamps(signature: list) -> list[int]:
    return [n for row in signature if row for n in row[3:]]


class SourceProofs:
    """One cycle's byte proofs: each source hashed at most once, every hashed byte charged."""

    def __init__(self, byte_budget: int, out: ReadOutcome) -> None:
        self.byte_budget = byte_budget
        self.out = out
        self.hashed: dict[Path, tuple[list, str | None]] = {}

    def digest(self, path: Path, signature: list) -> str | None:
        """SHA-256 of ``path``, hashed after the stat that produced ``signature``.

        None when unreadable, when hashing would leave no budget to read with, or when this
        cycle already hashed the source under another signature (it moved mid-cycle).
        """
        if path in self.hashed:
            seen, digest = self.hashed[path]
            return digest if seen == signature else None
        if signature[0][2] >= self.byte_budget - self.out.bytes_read:
            return None
        digest, hashed = source_digest(path)
        self.out.bytes_read += hashed
        self.hashed[path] = (signature, digest)
        return digest

    def observe(self, reader: Any, source: Source) -> tuple[list | None, str | None]:
        """Signature taken before a read plus the digest it must be stored with.

        The digest is "" when the signature is settled (proof on its own) and None when a
        racy signature cannot be paired with bytes (store no signature at all).
        """
        observed = time_ns()
        signature = source_signature(reader, source)
        if signature is None or not racy(observed, *_stamps(signature)):
            return signature, ""
        return signature, self.digest(source.path, signature)

    def idle(self, reader: Any, source: Source, record: SourceRecord | None) -> SourceRecord | None:
        """``record`` when the source provably did not change since it; None otherwise.

        A stored digest that still matches is dropped once the stored stamps are settled
        relative to this cycle's clock (read before the stat), so later cycles only stat.
        """
        if (
            record is None
            or record.status != "ok"
            or not record.complete
            or record.more
            or record.pending_tail
        ):
            return None
        extra = record.state.get("extra", {})
        if isinstance(reader, OpenCodeReader):
            from ssgrep.accounting.readers.opencode_storage import SnapshotError, source_generation

            accepted = accepted_database_generation(extra)
            if accepted is None or accepted != extra.get("collector_source_generation"):
                return None
            try:
                return record if source_generation(source.path) == accepted else None
            except (OSError, SnapshotError):
                return None
        observed = time_ns()
        signature = source_signature(reader, source)
        if signature is None or signature != extra.get(SIGNATURE):
            return None
        stored = extra.get(DIGEST)
        if stored is None:
            return record
        if self.digest(source.path, signature) != stored:
            return None
        if racy(observed, *_stamps(signature)):
            return record
        state = deepcopy(record.state)
        del state["extra"][DIGEST]
        return replace(record, state=state)
