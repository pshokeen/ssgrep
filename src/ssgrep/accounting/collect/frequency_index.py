"""Bounded construction and immutable reads of indexed lexical frequency views."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import islice
from typing import Any, NotRequired, Protocol, TypedDict

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.frequency import (
    BASIS_INDEX,
    LEXICAL_MAPPING,
    LEXICAL_NORMALIZATION,
    FrequencyView,
    frequency_key_id,
    lexical_key,
)
from ssgrep.accounting.collect.store import CollectError, dumps, loads
from ssgrep.accounting.ledger.contracts import digest_of
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.racy import racy
from ssgrep.store import LanceStore

# --- index_frequency.py ---

Authority = Callable[[str | None], bool]


class Repository(Protocol):
    """Read-only metadata and immutable direct-page snapshot seam."""

    def exists(self) -> bool: ...
    def get_meta(self, key: str) -> str | None: ...
    def count(self, name: str) -> int: ...
    def snapshot(self, generation: str) -> Repository: ...
    def rows_page(self, *, start: int, limit: int) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def _generation(repository: Repository) -> str | None:
    """Include immutable Lance table versions; detect damage without reading full df."""
    generation = repository.get_meta("last_index_time")
    if not generation:
        return None
    table = getattr(repository, "table", None)
    if table is None:
        return generation  # Other read interfaces supply their own immutable generation.
    versions = []
    for name in ("metadata", "chunks"):
        value = table(name)
        version = value.version if value is not None else None
        if type(version) is not int or version < 0:
            return None
        versions.append(f"{name}:{version}")
    return f"{generation}@{','.join(versions)}"


def _authority(profile: str) -> tuple[Authority, str]:
    from ssgrep.accounting.privacy.policy import read_policy_file
    from ssgrep.accounting.views.filter import Scoper

    ownership = read_policy_file()
    policy = ownership.profiles.get(profile)
    if policy is None or not policy.collection_enabled or policy.paused:
        raise ValueError("no active corpus authority")
    scoper = Scoper(policy, ownership)
    return scoper.in_scope, digest_of([policy, ownership])


def _read(
    profile: str, repository: Repository, authority: Authority, authority_id: str
) -> FrequencyView:
    from ssgrep.accounting.capture.fingerprint import load_key

    advance = globals()["advance"]

    if not repository.exists():
        return FrequencyView.unknown("frequency_index_absent", profile=profile)
    try:
        key = load_key(profile, create=False)
    except (OSError, ValueError):
        key = None
    if key is None:
        return FrequencyView.unknown("frequency_key_unavailable", profile=profile)
    return advance(profile, repository, authority, authority_id, key, _generation)


def read_index_view(
    profile: str,
    *,
    repository: Repository | None = None,
    authority: Authority | None = None,
    authority_id: str | None = None,
    cached: FrequencyView | None = None,
) -> FrequencyView:
    """Read compatible current statistics, advancing private bounded build work if needed.

    Explicit authority is an integration seam for an already approved ownership
    snapshot, not authority carried by index data. Supply its stable identity too.
    Default authority uses current local profile policy, never organization enrollment
    alone. Any unapproved chunk invalidates the global df; filtered df is not invented.
    """
    owned = repository is None
    try:
        if authority is None:
            try:
                authority, authority_id = _authority(profile)
            except (OSError, ValueError):
                return FrequencyView.unknown("frequency_scope_unapproved", profile=profile)
        if not authority_id:
            return FrequencyView.unknown("frequency_scope_unapproved", profile=profile)
        if repository is None:
            repository = IndexRepository()
        from ssgrep.accounting.capture.fingerprint import load_key

        key = load_key(profile, create=False)
        reuse = (
            cached is not None
            and bool(cached.generation)
            and cached.profile_scope == profile
            and cached.authority_id == authority_id
            and key is not None
            and cached.key_id == frequency_key_id(key)
            and cached.unavailable_reason
            not in (
                "frequency_generation_changed",
                "frequency_index_unreadable",
                "frequency_build_pending",
                "frequency_snapshot_unavailable",
            )
            and repository.exists()
            and repository.get_meta("index_state") == "ready"
            and _generation(repository) == cached.generation
        )
        # A ready index generation is immutable. Unchanged generation, authority and
        # key require no lexical_stats read, df copy or chunk authorization projection.
        result = cached if reuse else _read(profile, repository, authority, authority_id)
    except Exception:
        # An unreadable local index/key must not trigger rebuild or enable rarity.
        result = FrequencyView.unknown("frequency_index_unreadable", profile=profile)
    finally:
        if owned and repository is not None:
            try:
                repository.close()
            except Exception:
                result = FrequencyView.unknown("frequency_index_unreadable", profile=profile)
    return result


# --- index_frequency_build.py ---

MAX_STATS_BYTES = 16 * 1024 * 1024

MAX_BUILD_BYTES = 64 * 1024 * 1024

HEAD_BYTES = 1024 * 1024

PAGE_BYTES = 1024 * 1024

VISITS = 4096

SECONDS = 0.25

ORPHAN_ENTRIES = 4096

ORPHAN_DELETIONS = 64

CHECKPOINT = "frequency-build-head.json"

_TOKEN = re.compile(r"[a-z0-9_]{2,}")

_PAGE = re.compile(r"frequency-build-(terms|seen)-[0-9]{8}\.json")


class PageRef(TypedDict):
    name: str
    digest: str
    records: int
    bytes: int


class BuildJob(TypedDict):
    schema: int
    profile: str
    generation: str
    authority_id: str
    key_id: str
    stats_digest: str
    documents: int
    term_cursor: int
    chunk_cursor: int
    terms: list[PageRef]
    seen: list[PageRef]
    bytes: int
    scope_digest: str
    failure: NotRequired[str]
    complete: NotRequired[bool]


def _load(profile) -> BuildJob | None:
    try:
        directory = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError:
        return None
    try:
        raw = storage.read_file(directory, CHECKPOINT, HEAD_BYTES)
        if raw is None:
            return None
        document = loads(raw)
        job: BuildJob = document["job"]
        if digest_of(job) != document["digest"]:
            raise ValueError("frequency build digest mismatch")
        return job
    finally:
        os.close(directory)


def _save(profile, job):
    body = dumps({"job": job, "digest": digest_of(job)})
    if len(body) > HEAD_BYTES:
        raise ValueError("frequency build head budget exceeded")
    directory = storage.open_profile_dir(profile, "collector")
    try:
        storage.atomic_write(directory, CHECKPOINT, body)
    finally:
        os.close(directory)


def _page_name(kind, number):
    return f"frequency-build-{kind}-{number:08d}.json"


def _discard(profile, job):
    directory = storage.open_profile_dir(profile, "collector")
    try:
        for kind in ("terms", "seen"):
            refs = job[kind]
            for number in range(len(refs) + 1):
                # Include the next uncommitted page after a page-write/head-write crash.
                storage.delete_file(directory, _page_name(kind, number))
        storage.delete_file(directory, CHECKPOINT)
        storage.fsync_dir(directory)
    finally:
        os.close(directory)


def _head_present(directory):
    try:
        os.stat(CHECKPOINT, dir_fd=directory, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _clear_orphans(profile):
    """Absent head only: frozen accepted views embed DF and never reference these pages."""
    try:
        directory = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError:
        return None
    try:
        pages = []
        with os.scandir(directory) as entries:
            for visited, entry in enumerate(islice(entries, ORPHAN_ENTRIES + 1), start=1):
                if visited > ORPHAN_ENTRIES:
                    return "frequency_build_orphan_inventory_budget_exceeded"
                if _PAGE.fullmatch(entry.name) is not None:
                    pages.append(entry.name)
        if _head_present(directory):
            return "frequency_build_orphan_cleanup_pending"
        removed = False
        for name in pages[:ORPHAN_DELETIONS]:
            if _head_present(directory):
                return "frequency_build_orphan_cleanup_pending"
            removed |= storage.delete_file(directory, name)
        if removed:
            storage.fsync_dir(directory)
        return "frequency_build_orphan_cleanup_pending" if len(pages) > ORPHAN_DELETIONS else None
    finally:
        os.close(directory)


def _page(profile, reference):
    if _PAGE.fullmatch(reference["name"]) is None:
        raise ValueError("invalid frequency page name")
    directory = storage.open_profile_dir(profile, "collector", create=False)
    try:
        raw = storage.read_file(directory, reference["name"], PAGE_BYTES)
    finally:
        os.close(directory)
    if raw is None or hashlib.sha256(raw).hexdigest() != reference["digest"]:
        raise ValueError("frequency page missing or damaged")
    document = loads(raw)
    if len(document) != reference["records"] or len(raw) != reference["bytes"]:
        raise ValueError("frequency page size mismatch")
    return document


def _append(profile, job, kind, document):
    if not document:
        return
    body = dumps(document)
    if len(body) > PAGE_BYTES or job["bytes"] + len(body) > MAX_BUILD_BYTES:
        job["failure"] = "frequency_build_budget_exceeded"
        return
    name = _page_name(kind, len(job[kind]))
    directory = storage.open_profile_dir(profile, "collector")
    try:
        # Referenced pages are immutable. Only the next unreferenced cursor slot may
        # be replaced on retry; the atomic head is the sole acceptance point.
        storage.atomic_write(directory, name, body)
    finally:
        os.close(directory)
    job[kind].append(
        {
            "name": name,
            "digest": hashlib.sha256(body).hexdigest(),
            "records": len(document),
            "bytes": len(body),
        }
    )
    job["bytes"] += len(body)


def _frozen_view(profile, job):
    df = {}
    for reference in job["terms"]:
        page = _page(profile, reference)
        if not isinstance(page, dict) or df.keys() & page.keys():
            raise ValueError("invalid frequency term page")
        df.update(page)
    view = FrequencyView(
        version_id="",
        basis=BASIS_INDEX,
        normalization=LEXICAL_NORMALIZATION,
        doc_count=job["documents"],
        df=df,
        generation=job["generation"],
        corpus_digest=digest_of([job["stats_digest"], job["scope_digest"]]),
        document_unit="indexed_chunk",
        coverage="complete_indexed_chunks",
        profile_scope=profile,
        authority_id=job["authority_id"],
        mapping=LEXICAL_MAPPING,
        key_id=job["key_id"],
    )
    return replace(view, version_id=view.identity())


def advance(profile, repository, authority, authority_id, key, generation_of):
    """Advance one immutable generation; pending does not accept a frequency pin."""
    key_id = frequency_key_id(key)
    generation = ""

    def unknown(reason):
        return replace(
            FrequencyView.unknown(reason, profile=profile),
            generation=generation,
            authority_id=authority_id,
            key_id=key_id,
            normalization=LEXICAL_NORMALIZATION,
            mapping=LEXICAL_MAPPING,
            document_unit="indexed_chunk",
        )

    try:
        job: BuildJob | None = _load(profile)
        if job is not None:
            if not isinstance(job, dict) or job.get("schema") != 1:
                return unknown("frequency_build_damaged")
            changed = (job["profile"], job["authority_id"], job["key_id"]) != (
                profile,
                authority_id,
                key_id,
            )
            advanced = (job.get("failure") or job.get("complete")) and (
                generation_of(repository) != job["generation"]
            )
            if changed or advanced:
                _discard(profile, job)
                job = None
    except (ValueError, KeyError, TypeError):
        return unknown("frequency_build_damaged")
    if job is None:
        cleanup = _clear_orphans(profile)
        if cleanup is not None:
            return unknown(cleanup)
        if repository.get_meta("index_state") != "ready":
            return unknown("frequency_index_not_ready")
        generation = generation_of(repository)
        if not generation:
            return unknown("frequency_generation_unknown")
    else:
        generation = job["generation"]
    try:
        frozen = repository.snapshot(generation)
    except Exception:
        if job is not None:
            _discard(profile, job)
        return unknown("frequency_snapshot_unavailable")
    try:
        if generation_of(frozen) != generation or frozen.get_meta("index_state") != "ready":
            return unknown("frequency_generation_changed")
        raw = frozen.get_meta("lexical_stats")
        if not raw:
            return unknown("frequency_stats_absent")
        if len(raw.encode("utf-8")) > MAX_STATS_BYTES:
            return unknown("frequency_stats_budget_exceeded")
        try:
            stats = json.loads(raw)
        except (ValueError, TypeError):
            return unknown("frequency_stats_damaged")
        if not isinstance(stats, dict):
            return unknown("frequency_stats_damaged")
        documents, df, average = stats.get("n_docs"), stats.get("df"), stats.get("avgdl")
        if (
            type(documents) is not int
            or documents < 0
            or not isinstance(df, dict)
            or type(average) not in (int, float)
            or not math.isfinite(average)
            or average < 0
        ):
            return unknown("frequency_stats_damaged")
        if documents == 0:
            return unknown("frequency_empty_corpus")
        if frozen.count("chunks") != documents:
            return unknown("frequency_stats_incompatible")
        stats_digest = lexical_key(key, raw)
        if job is None:
            if (
                repository.get_meta("index_state") != "ready"
                or generation_of(repository) != generation
            ):
                return unknown("frequency_generation_changed")
            job = {
                "schema": 1,
                "profile": profile,
                "generation": generation,
                "authority_id": authority_id,
                "key_id": key_id,
                "stats_digest": stats_digest,
                "documents": documents,
                "term_cursor": 0,
                "chunk_cursor": 0,
                "terms": [],
                "seen": [],
                "bytes": 0,
                "scope_digest": digest_of([generation, documents]),
            }
        if job["stats_digest"] != stats_digest or job["documents"] != documents:
            return unknown("frequency_generation_changed")
        if job.get("failure"):
            return unknown(job["failure"])
        if not (
            type(job["term_cursor"]) is int
            and 0 <= job["term_cursor"] <= len(df)
            and type(job["chunk_cursor"]) is int
            and 0 <= job["chunk_cursor"] <= documents
            and sum(page["records"] for page in job["terms"]) == job["term_cursor"]
            and sum(page["records"] for page in job["seen"]) == job["chunk_cursor"]
            and sum(page["bytes"] for page in job["terms"] + job["seen"]) == job["bytes"]
            and 0 <= job["bytes"] <= MAX_BUILD_BYTES
        ):
            return unknown("frequency_build_damaged")
        if job.get("complete"):
            return _frozen_view(profile, job)
        terms = {}
        deadline = time.monotonic() + SECONDS
        for term, count in islice(df.items(), job["term_cursor"], job["term_cursor"] + VISITS):
            if (
                not isinstance(term, str)
                or _TOKEN.fullmatch(term) is None
                or type(count) is not int
                or not 1 <= count <= documents
            ):
                job["failure"] = "frequency_stats_damaged"
                break
            terms[lexical_key(key, term)] = count
            job["term_cursor"] += 1
            if time.monotonic() >= deadline:
                break
        _append(profile, job, "terms", terms)
        if job["term_cursor"] == len(df) and not job.get("failure"):
            seen = set()
            for reference in job["seen"]:
                seen.update(_page(profile, reference))
            if len(seen) != job["chunk_cursor"]:
                return unknown("frequency_build_damaged")
            start = job["chunk_cursor"]
            limit = min(VISITS, documents - start)
            chunks = frozen.rows_page(start=start, limit=limit) if limit else []
            opaque_page = []
            if len(chunks) != limit:
                job["failure"] = "frequency_stats_incompatible"
            elif len(dumps(chunks)) > PAGE_BYTES:
                job["failure"] = "frequency_scope_page_budget_exceeded"
            else:
                deadline = time.monotonic() + SECONDS
                for chunk in chunks:
                    identity, cwd = chunk.get("chunk_id"), chunk.get("cwd")
                    if (
                        not isinstance(identity, str)
                        or not identity
                        or cwd is not None
                        and not isinstance(cwd, str)
                    ):
                        job["failure"] = "frequency_stats_incompatible"
                        break
                    opaque = lexical_key(key, "chunk-id\0" + identity)
                    if opaque in seen:
                        job["failure"] = "frequency_stats_incompatible"
                        break
                    if authority(cwd) is not True:
                        job["failure"] = "frequency_scope_unapproved"
                        break
                    seen.add(opaque)
                    opaque_page.append(opaque)
                    job["scope_digest"] = digest_of(
                        [job["scope_digest"], opaque, lexical_key(key, "cwd\0" + str(cwd))]
                    )
                    job["chunk_cursor"] += 1
                    if time.monotonic() >= deadline:
                        break
            _append(profile, job, "seen", opaque_page)
        if generation_of(frozen) != generation or frozen.get_meta("lexical_stats") != raw:
            return unknown("frequency_generation_changed")
        job["complete"] = job["term_cursor"] == len(df) and job["chunk_cursor"] == documents
        _save(profile, job)
        if job.get("failure") or not job["complete"]:
            return unknown(job.get("failure", "frequency_build_pending"))
        return _frozen_view(profile, job)
    finally:
        if frozen is not repository:
            frozen.close()


# --- index_frequency_repository.py ---

_VERSION = re.compile(r".+@metadata:([0-9]+),chunks:([0-9]+)$")


class IndexRepository:
    """Private accounting adapter; checkout changes handles, never index contents.

    A page cursor is a logical row offset in ONE immutable chunks table version.
    take_offsets requests only that range, unlike a query OFFSET prefix scan. Results
    carry _rowoffset and are sorted within the bounded page, not over the corpus.
    """

    def __init__(self) -> None:
        self.store = LanceStore()
        self.tables: dict[str, Any] = {}

    def exists(self) -> bool:
        return self.store.exists()

    def table(self, name: str):
        return self.tables[name] if name in self.tables else self.store.table(name)

    def get_meta(self, key: str) -> str | None:
        if not self.tables:
            return self.store.get_meta(key)
        from ssgrep.store import quote

        rows = self.tables["metadata"].search().where(f"key = {quote(key)}").select(["value"])
        values = rows.limit(1).to_list()
        return str(values[0]["value"]) if values else None

    def count(self, name: str) -> int:
        table = self.table(name)
        return int(table.count_rows()) if table is not None else 0

    def rows_page(self, *, start: int, limit: int) -> list[dict[str, Any]]:
        table = self.table("chunks")
        rows = (
            table.take_offsets(list(range(start, start + limit)))
            .select(["chunk_id", "cwd", "_rowoffset"])
            .to_arrow()
            .sort_by("_rowoffset")
        )
        if rows["_rowoffset"].to_pylist() != list(range(start, start + limit)):
            raise ValueError("frequency page does not cover its immutable offset range")
        return rows.select(["chunk_id", "cwd"]).to_pylist()

    def snapshot(self, generation: str) -> IndexRepository:
        match = _VERSION.fullmatch(generation)
        if match is None:
            raise ValueError("frequency snapshot needs immutable table versions")
        frozen = IndexRepository()
        try:
            for name, version in zip(("metadata", "chunks"), match.groups(), strict=True):
                table = frozen.store.table(name)
                if table is None:
                    raise ValueError("frequency snapshot table is absent")
                table.checkout(int(version))
                frozen.tables[name] = table
            return frozen
        except Exception:
            frozen.close()
            raise

    def close(self) -> None:
        self.store.close()


# --- collector_frequency_gc.py ---

BLOB = "frequency-maintenance"

VIEW = re.compile(r"frequency-(freq-[0-9a-f]{24})\.json$")

MAX_REFERENCE_BYTES = 4 * 1024 * 1024

REFERENCE_BYTE_BUDGET = 16 * 1024 * 1024

MAX_INVENTORY_ENTRIES = 250_000


@dataclass(frozen=True)
class FrequencyGCOutcome:
    pending: bool = False
    removed: int = 0
    warnings: tuple[str, ...] = ()


def _read_reference(profile, filename, sub=store.STATE_DIR):
    directory = (
        storage.open_profile_dir(profile, "collector", sub, create=False)
        if sub == store.STATE_DIR
        else storage.open_profile_dir(profile, sub, create=False)
    )
    try:
        descriptor = storage.open_regular(directory, filename)
        try:
            size = os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)
        if size > MAX_REFERENCE_BYTES:
            return None, size
        body = storage.read_file(directory, filename, MAX_REFERENCE_BYTES)
        return store.loads(body) if body is not None else None, size
    finally:
        os.close(directory)


def _receipt_inventory(profile, *, pin=False):
    try:
        directory = storage.open_profile_dir(profile, "receipts", create=False)
    except FileNotFoundError:
        return [], None
    try:
        observed = time.time_ns()
        info = os.fstat(directory)
        with os.scandir(directory) as entries:
            listing = sorted((entry.name, entry.inode()) for entry in entries)
        stamp: list[int | str] = [info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns]
        if pin or racy(observed, info.st_mtime_ns, info.st_ctime_ns):
            # Racy-clean: a fresh stamp is trusted only with the (name, inode) listing it covers.
            stamp.append(digest_of(listing))
        return [name for name, _ in listing if name.endswith(".json")], stamp
    finally:
        os.close(directory)


def _receipts_unchanged(profile, stamp):
    if stamp is None or len(stamp) == 4:
        return stamp == _receipt_stamp(profile)
    return stamp == _receipt_inventory(profile, pin=True)[1]


def _reference_sources(blobs):
    return sorted(
        (name, filename)
        for name, filename in blobs.items()
        if name.startswith("ledger-")
        or name.startswith("identity-archive-")
        or name.split("-scope-", 1)[0] in {"frequency-assignments", "reduction", "partitions"}
    )


def _receipt_stamp(profile):
    try:
        directory = storage.open_profile_dir(profile, "receipts", create=False)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(directory)
        return [info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns]
    finally:
        os.close(directory)


def stage(profile, checkpoint, blobs, *, current_refs, before_day, today):
    """Stage queues/reference handles before the acceptance checkpoint, never erase."""
    raw = store.read_blob(profile, checkpoint.blobs.get(BLOB))
    if raw is not None and (not isinstance(raw, dict) or raw.get("schema") != 1):
        raise CollectError("state_corrupt", "Frequency maintenance job is invalid.")
    sources = _reference_sources(blobs)
    # A completed epoch does no directory/body inventory again on the same day.
    source_version = digest_of(sources)
    if raw and raw["today"] == today and raw["phase"] == "complete":
        blobs[BLOB] = checkpoint.blobs[BLOB]
        return False
    if (
        raw
        and raw["phase"] not in {"complete", "blocked"}
        and raw["source_version"] == source_version
        and _receipts_unchanged(profile, raw.get("receipt_stamp"))
    ):
        protected = sorted(set(raw["protected"]) | set(current_refs))
        if protected != raw["protected"]:
            raw["protected"] = protected
            blobs[BLOB] = store.write_blob(profile, BLOB, raw)
        else:
            blobs[BLOB] = checkpoint.blobs[BLOB]
        return bool(raw["phase"] != "complete")
    directory = storage.open_profile_dir(profile, "collector")
    try:
        candidates = []
        for name in storage.list_names(directory):
            if VIEW.fullmatch(name) is None:
                continue
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if datetime.fromtimestamp(info.st_mtime, UTC).date().isoformat() < before_day:
                candidates.append(name)
    finally:
        os.close(directory)
    receipts, receipt_stamp = _receipt_inventory(profile) if candidates else ([], None)
    bounded = len(candidates) + len(receipts) + len(sources) <= MAX_INVENTORY_ENTRIES
    if raw and raw["phase"] not in {"complete", "blocked"} and bounded:
        known = {filename for _, filename in raw["sources"]}
        raw["sources"].extend(pair for pair in sources if pair[1] not in known)
        known_receipts = set(raw["receipts"])
        raw["receipts"].extend(name for name in receipts if name not in known_receipts)
        known_candidates = set(raw["candidates"])
        raw["candidates"].extend(
            name for name in sorted(candidates) if name not in known_candidates
        )
        raw["protected"] = sorted(set(raw["protected"]) | set(current_refs))
        raw["source_version"] = source_version
        raw["receipt_stamp"] = receipt_stamp
        raw["phase"] = "references"
        blobs[BLOB] = store.write_blob(profile, BLOB, raw)
        return True
    job = {
        "schema": 1,
        "today": today,
        "before_day": before_day,
        "source_version": source_version,
        "sources": sources,
        "source_cursor": 0,
        "receipt_cursor": 0,
        "receipts": receipts,
        "receipt_stamp": receipt_stamp,
        "candidates": sorted(candidates),
        "candidate_cursor": 0,
        "protected": sorted(set(current_refs)),
        "phase": "references"
        if candidates and bounded
        else "blocked"
        if not bounded
        else "complete",
        "warnings": [] if bounded else ["frequency_inventory_bound"],
    }
    blobs[BLOB] = store.write_blob(profile, BLOB, job)
    return job["phase"] != "complete"


def _collect_refs(name, raw):
    if not isinstance(raw, dict):
        return None, []
    if name.startswith("identity-archive-"):
        if raw.get("schema") != 1:
            return None, []
        return set(), _reference_sources(raw.get("blobs", {}))
    if raw.get("reference_index_complete") is False:
        return None, []
    refs = raw.get("frequency_refs")
    if isinstance(refs, list) and all(isinstance(view, str) for view in refs):
        return set(refs), []
    # No full legacy ledger/reduction/assignment traversal during collection.
    return None, []


def finish(profile, *, generation, limit=32):
    """Resume only accepted work, then durably commit progress at the same generation."""
    from ssgrep.accounting.collect.frequency import prune_views

    checkpoint = store.load_checkpoint(profile)
    if checkpoint is None or checkpoint.generation != generation or BLOB not in checkpoint.blobs:
        return FrequencyGCOutcome()
    job = store.read_blob(profile, checkpoint.blobs[BLOB])
    if not isinstance(job, dict) or job.get("schema") != 1:
        raise CollectError("state_corrupt", "Frequency maintenance job is invalid.")
    if job["phase"] == "complete":
        return FrequencyGCOutcome()
    if job["phase"] == "blocked":
        return FrequencyGCOutcome(True, warnings=tuple(job["warnings"]))
    protected = set(job["protected"])
    charged = visited = 0
    while job["phase"] == "references" and visited < limit:
        if job["source_cursor"] < len(job["sources"]):
            name, filename = job["sources"][job["source_cursor"]]
            raw, size = _read_reference(profile, filename)
            if charged and charged + size > REFERENCE_BYTE_BUDGET:
                break
            refs, children = _collect_refs(name, raw)
            if refs is None:
                job["phase"] = "blocked"
                job["warnings"] = ["frequency_references_unknown"]
                break
            protected.update(refs)
            known = {filename for _, filename in job["sources"]}
            job["sources"].extend(pair for pair in children if pair[1] not in known)
            job["source_cursor"] += 1
        elif job["receipt_cursor"] < len(job["receipts"]):
            filename = job["receipts"][job["receipt_cursor"]]
            try:
                raw, size = _read_reference(profile, filename, "receipts")
            except FileNotFoundError:
                raw, size = {}, 0  # Physical committed retention already removed this receipt.
            if charged and charged + size > REFERENCE_BYTE_BUDGET:
                break
            if raw is None:
                job["phase"] = "blocked"
                job["warnings"] = ["frequency_references_unknown"]
                break
            from ssgrep.accounting.collect.state import frequency_refs

            protected.update(frequency_refs(raw))
            job["receipt_cursor"] += 1
        else:
            job["phase"] = "pruning"
            break
        charged += size
        visited += 1
    removed = 0
    if job["phase"] == "pruning":
        # A changed receipt inventory needs a new accepted reference-proof queue.
        if not _receipts_unchanged(profile, job.get("receipt_stamp")):
            return FrequencyGCOutcome(True, warnings=("frequency_reference_inventory_changed",))
        start = job["candidate_cursor"]
        names = job["candidates"][start : start + limit]
        result = prune_views(
            profile,
            candidates=names,
            referenced=protected,
            before_day=job["before_day"],
            limit=limit,
        )
        removed = result.removed
        job["candidate_cursor"] += result.scanned
        if job["candidate_cursor"] >= len(job["candidates"]):
            job["phase"] = "complete"
            job["sources"] = []  # Completed proof no longer pins obsolete namespace children.
            job["receipts"] = []
            job["candidates"] = []
    job["protected"] = sorted(protected)
    checkpoint.blobs[BLOB] = store.write_blob(profile, BLOB, job)
    store.commit_checkpoint(profile, checkpoint)
    return FrequencyGCOutcome(job["phase"] != "complete", removed, tuple(job["warnings"]))
