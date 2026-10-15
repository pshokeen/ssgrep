"""Bounded frequency construction using private keyed pages and a small atomic head.

Each call parses <=16MiB raw metadata and <=1MiB head, visits <=4096 terms and
<=4096 direct chunk projections (<=1MiB serialized). Duplicate proof reads prior
keyed seen pages; all retained build pages together are capped at 64MiB. Only new
<=1MiB pages and the head are written per step, not a growing corpus checkpoint.
A 250ms cooperative deadline applies to term/scope visits, NOT bounded metadata,
uniqueness-proof, final materialization, or atomic IO. This is not a hard deadline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import replace
from itertools import islice
from typing import NotRequired, TypedDict

from ssgrep.accounting import storage
from ssgrep.accounting.collector_store import dumps, loads
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.frequency import (
    BASIS_INDEX,
    LEXICAL_MAPPING,
    LEXICAL_NORMALIZATION,
    FrequencyView,
    frequency_key_id,
    lexical_key,
)

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
    if len(df) != job["term_cursor"]:
        raise ValueError("frequency term coverage mismatch")
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
