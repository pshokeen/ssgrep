"""Frozen raw index inputs for the independent lab oracle; no accounting outputs imported.

Only the public Lance repository read seam is used. The caller supplies the lab's
canonical project authority, not authority asserted by the index itself. No index,
settings, key, or corpus is created or rebuilt by this adapter.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .observer import ObservedCall


@dataclass(frozen=True)
class IndexStats:
    generation: str = ""
    documents: int = 0
    df: tuple[tuple[str, int], ...] = ()
    corpus_digest: str = ""
    normalization: str = "lexical_ascii_lower_v1"
    document_unit: str = "indexed_chunk"
    coverage: str = "complete_indexed_chunks"
    profile_scope: str = "personal-local"
    authority_id: str = ""
    key_id: str = ""
    unavailable_reason: str = ""
    mapping: str = "lexical_superset_lower_bound_v2"

    def provenance(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in (
                "generation",
                "corpus_digest",
                "normalization",
                "document_unit",
                "coverage",
                "profile_scope",
                "authority_id",
                "key_id",
                "unavailable_reason",
                "mapping",
            )
        }


def rarity_upper_df(stats: IndexStats | None, kind: str, token: str) -> int | None:
    """Bound whole-candidate df independently of production frequency outputs.

    Whole-candidate occurrence implies occurrence of every guaranteed ASCII component.
    Minimum PRESENT component df bounds whole-candidate df above. Missing components
    never imply df=0; failing a lower-bound test is inconclusive, not rejection.
    """
    if (
        stats is None
        or stats.unavailable_reason
        or type(stats.documents) is not int
        or stats.documents <= 0
        or stats.normalization != "lexical_ascii_lower_v1"
        or stats.document_unit != "indexed_chunk"
        or stats.coverage != "complete_indexed_chunks"
        or stats.mapping != "lexical_superset_lower_bound_v2"
        or not all(
            (
                stats.generation,
                stats.corpus_digest,
                stats.profile_scope,
                stats.authority_id,
                stats.key_id,
            )
        )
        or kind not in ("identifier", "path", "flag", "error_code")
        or not token.isascii()
        or not 1 <= len(token) <= 160
    ):
        return None
    counts = dict(stats.df)
    present = [counts[c] for c in set(re.findall(r"[a-z0-9_]{2,}", token.lower())) if c in counts]
    if not present or any(type(n) is not int or not 1 <= n <= stats.documents for n in present):
        return None
    return min(present)


def rarity_bands(stats: IndexStats | None, kind: str, token: str) -> tuple[bool, ...]:
    """Prove unchanged integer 8/6/4-bit thresholds without rounded-log approximation."""
    upper = rarity_upper_df(stats, kind, token)
    if upper is None or stats is None:
        return (False, False, False)
    return tuple(stats.documents + 1 >= (upper + 1) * (2**bits) for bits in (8, 6, 4))


def freeze_lookups(
    previous: Mapping[str, IndexStats], calls: Sequence[ObservedCall], current: IndexStats
) -> dict[str, IndexStats]:
    frozen = dict(previous)
    for call in calls:
        if call.tool == "ssgrep" and len(call.argv) > 1 and call.argv[1] in ("search", "show"):
            frozen.setdefault(call.call_id, current)
    return frozen


def accept_lookup_inputs(
    previous: Mapping[str, IndexStats],
    calls: Sequence[ObservedCall],
    before: IndexStats,
    after: IndexStats,
    *,
    calls_after: Sequence[ObservedCall] | None = None,
) -> dict[str, IndexStats]:
    """Bind first-collected inputs, not lookup wall-time or later report-time statistics.

    Production accepts journal/native lookups during collection. The independently
    observed collector boundary brackets its raw index inputs; a changed generation,
    scope, key or statistics cannot prove which view was consumed and stays unknown.
    Existing assignments are immutable, including late native joins to observed calls.
    Native turns are sequential in the authorized lane. A newly observed late call
    does not establish collector acceptance: never attach earlier positive stats to
    it merely because the index was stable. Preserve that uncertainty explicitly.
    """
    current = (
        before
        if before == after
        else IndexStats(unavailable_reason="frequency_inputs_changed_during_collection")
    )
    accepted = freeze_lookups(previous, calls, current)
    if calls_after is not None:
        known = set(previous) | {call.call_id for call in calls}
        late = [call for call in calls_after if call.call_id not in known]
        reason = (
            "lookup_acceptance_during_collection_unproven"
            if before == after
            else "lookup_raced_collection"
        )
        accepted = freeze_lookups(accepted, late, IndexStats(unavailable_reason=reason))
    return accepted


def _generation(repo) -> str:
    stamp = repo.get_meta("last_index_time")
    if not stamp:
        raise ValueError("generation_unknown")
    versions = []
    for name in ("metadata", "chunks"):
        table = repo.table(name)
        version = table.version if table is not None else None
        if type(version) is not int or version < 0:
            raise ValueError("generation_unknown")
        versions.append(f"{name}:{version}")
    return f"{stamp}@{','.join(versions)}"


def read_index_inputs(data: Path, project: Path, *, repository=None) -> IndexStats:
    """Read actual lab-owned index metadata once; retain typed absence/damage, not a fallback."""
    repo = repository
    owned = repo is None
    try:
        if repo is None:
            from ssgrep.store import LanceStore

            repo = LanceStore()
            repo.path = data / "lancedb"
        if not repo.exists() or repo.get_meta("index_state") != "ready":
            raise ValueError("index_not_ready")
        generation = _generation(repo)
        raw = repo.get_meta("lexical_stats")
        if not isinstance(raw, str) or len(raw.encode()) > 16 * 1024 * 1024:
            raise ValueError("stats_absent_or_oversized")
        doc = json.loads(raw)
        n, df, avg = doc.get("n_docs"), doc.get("df"), doc.get("avgdl")
        if (
            type(n) is not int
            or not 0 < n <= 100000
            or not isinstance(df, dict)
            or len(df) > 100000
            or type(avg) not in (int, float)
            or not math.isfinite(avg)
            or avg < 0
        ):
            raise ValueError("stats_invalid")
        if any(
            not isinstance(k, str)
            or not re.fullmatch(r"[a-z0-9_]{2,}", k)
            or type(v) is not int
            or not 1 <= v <= n
            for k, v in df.items()
        ):
            raise ValueError("stats_invalid")
        if repo.count("chunks") != n:
            raise ValueError("document_unit_incompatible")
        offsets = list(range(n))
        chunks = repo.table("chunks")
        if chunks is None:
            raise ValueError("chunk_table_unavailable")
        projected = (
            chunks.take_offsets(offsets)
            .select(["chunk_id", "cwd", "_rowoffset"])
            .to_arrow()
            .sort_by("_rowoffset")
        )
        if projected["_rowoffset"].to_pylist() != offsets:
            raise ValueError("chunk_offsets_incompatible")
        rows = projected.select(["chunk_id", "cwd"]).to_pylist()
        project = project.resolve(strict=True)
        ids = set()
        scopes = []
        for row in rows:
            identity, cwd = row.get("chunk_id"), row.get("cwd")
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError("chunk_identity_invalid")
            if not isinstance(cwd, str) or not Path(cwd).resolve().is_relative_to(project):
                raise ValueError("scope_unapproved")
            ids.add(identity)
            scopes.append((identity, cwd))
        key_path = data / "accounting" / "personal-local" / "fingerprint.key"
        fd = os.open(key_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            key = os.read(fd, 33)
        finally:
            os.close(fd)
        if len(key) != 32:
            raise ValueError("key_unavailable")
        key_id = hmac.digest(key, b"ssgrep-frequency-key-id-v1", "sha256").hex()
        if (
            _generation(repo) != generation
            or repo.get_meta("lexical_stats") != raw
            or repo.get_meta("index_state") != "ready"
        ):
            raise ValueError("generation_changed")

        def lexical_digest(text: str) -> str:
            return hmac.digest(
                key, b"ssgrep-frequency-lexical-v1\0" + text.encode("utf-8"), "sha256"
            ).hex()

        def digest(value) -> str:
            return hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                    "utf-8"
                )
            ).hexdigest()

        # Independently reproduce the declared raw-authority wire digest. The
        # exact lexical metadata bytes and immutable logical chunk order matter;
        # neither parsed/re-dumped statistics nor sorted chunk identities suffice.
        scope_digest = digest([generation, n])
        for identity, cwd in scopes:
            scope_digest = digest(
                [
                    scope_digest,
                    lexical_digest("chunk-id\0" + identity),
                    lexical_digest("cwd\0" + str(cwd)),
                ]
            )
        result = IndexStats(
            generation,
            n,
            tuple(sorted(df.items())),
            digest([lexical_digest(raw), scope_digest]),
            authority_id=hashlib.sha256(str(project).encode()).hexdigest(),
            key_id=key_id,
        )
        if owned:
            repo.close()
            repo = None
        return result
    except Exception as exc:  # Third-party repository failures are unavailable, never a rebuild.
        return IndexStats(
            unavailable_reason=f"frequency_input_unavailable:{type(exc).__name__}:{exc}"
        )
    finally:
        if owned and repo is not None:
            try:
                repo.close()
            except Exception:
                pass  # The primary read/close failure is already retained as unavailable.
