"""Pipeline hub: discovery -> records -> episodes -> signal -> metadata -> chunker
-> embed -> store/vectors.

Design notes that are not obvious from the code alone:

Incremental reads. records.read_records() always starts at byte 0, so it
cannot serve the append-offset cursor; _read_from_offset() below is a
from-scratch line reader that mirrors its size cap, malformed-line, and
unknown-type handling exactly (reusing records.classify_record/RecordType so
the two readers cannot silently drift), but seeks to an arbitrary byte offset
first. A final line with no trailing newline is left unconsumed so a file
being written concurrently is picked up whole on the next run.

Episode identity across incremental runs. Each incremental parse only ever
sees the newly appended records, never the whole file, so episode ordinals
are renumbered to continue after the highest episode index already stored
for that session_id (queried from the open connection) rather than trusting
episodes.segment_episodes()'s own zero-based numbering. This guarantees new
episode_id / chunk_id values never collide with previously-embedded rows.
The corollary: an episode that was still "open" (mid-response, no closing
user record yet) at the end of one run and gets more assistant text appended
before the next user turn is *not* reopened and merged on the next run --
doing so would require re-deriving already-embedded chunk_ids with different
text, and vectors.f32 has no update/delete, so that would orphan a vector
row. The appended continuation becomes its own (headless) episode instead:
strictly more, never fewer or lost, indexed rows.

Signal re-derivation. episodes.segment_episodes() does its own light inline
text extraction and does not apply signal.strip_markup() or surface
system/away_summary prose. _extract_episode_text() re-derives clean
prompt/response text per episode via signal.classify_signal() over the same
group of records episodes.segment_episode_groups() paired with that
episode, which is what the design's signal/noise rules (D7) actually
require, without modifying episodes.py's own extraction.

Episode/group alignment is structural, not reconciled. _build_episodes()
used to re-derive each episode's records with a second, separately
maintained grouping function and fall back to treating the WHOLE batch of
new_records as every episode's group whenever its count disagreed with
segment_episodes()'s own -- which happened whenever a `user` turn's
extracted text was empty (e.g. a pure tool_result echo) and segment_episodes
folded it into the next episode while the second function split on it
regardless. Measured impact on this repo's own index before the fix: every
one of one session's 733 episodes held all 912 of the session's distinct
chunks (733x duplication), because each got the same whole-batch group.
episodes.segment_episode_groups() now returns (episode, its own records) so
there is nothing to reconcile and no fallback exists; see its docstring.

Rewrite/truncation deletes. store.delete_session_chunks() is documented as
prune-only. The Rewrite/Truncation Guard requirement is a different,
legitimate case (indexing itself must purge one file's stale rows before a
full re-parse), so _delete_rows_for_session() runs the equivalent statements
directly against the connection rather than repurposing that helper. It also
deletes the FTS rows *before* the chunk rows they are looked up from --
store.delete_session_chunks() does it in the other order, which leaves
orphaned chunks_fts rows behind (its DELETE FROM chunks runs first, so the
subsequent "chunk_id IN (SELECT ... FROM chunks WHERE session_id = ?)"
subquery finds nothing).

Generations. GenerationalStore's stage/commit-generation dance copies whole
files, which is right for the rare rebuild path (schema/model mismatch,
detected corruption, or an explicit rebuild request) but far too expensive
to run on every routine append. Routine incremental runs operate directly on
the current generation's live files; crash-safety for that path comes from
always appending vectors before inserting the chunk row that references
them, plus SQLite's own transactional commit -- so the likely residue of a
crash is an orphaned (harmless, unreferenced) trailing vector row. A rebuild
builds a full fresh generation and only swaps the manifest pointer at the
end, so a crash mid-rebuild leaves the prior generation fully intact.

Concurrent rebuild vs. incremental run. A rebuild racing an incremental run
that is still using the generation the rebuild is about to supersede is a
real scenario, not a hypothetical one -- hooks enqueue indexing work
asynchronously, so a hook-triggered incremental index() can be mid-flight
when a user runs `ssgrep index --rebuild`. The entire non-rebuild body below
runs inside `with gen_store.hold_generation(...)`, a SHARED advisory lock on
the generation whose live files this call is about to open
(store.GenerationalStore.hold_generation()'s docstring has the full
mechanism); commit_generation()'s cleanup step takes that same lock
EXCLUSIVELY before deleting a superseded generation's files and skips it
(deferring to a later cleanup) if this run is still holding it. That closes
the corruption window, but not a second, quieter problem: this call's own
`gen_store` was constructed before the rebuild committed, so by the time it
reaches checkpoint() its cached notion of "current generation" is stale.
checkpoint() itself re-checks the on-disk manifest and silently declines to
write when it no longer matches, rather than clobbering the manifest back
to a generation the rebuild has already superseded. See
tests/test_concurrent_index.py for the empirically-traced failure mode both
of these close.

Crash recovery is wired, not merely available. Write ordering alone bounds
what a crash can leave behind; it does not clean it up, and it cannot rule
out the reverse case (a chunk row that landed while the vector bytes behind
it did not, e.g. on power loss, since vectors.append() does not fsync). So
index() also drives GenerationalStore's cheap journal on the incremental
path: recover() runs once at the top, before any of those files are opened
or validated, and checkpoint() records the durable vector-row watermark
immediately after each SQLite commit. Both call sites are load-bearing and
covered by mutation-tested tests -- removing either one is a silent
regression, since a store can look fine on the happy path while the crash
window it is supposed to close stays wide open.

Tombstoning. A source is "vanished" only when its recorded path no longer
exists on disk, checked directly against the sessions table -- never
inferred from a discovery run being scoped with no_subagents=True, which
would otherwise be indistinguishable from real disappearance for subagent
sessions.

Forced rebuild loses tombstones by design. The Forced Rebuild requirement
defines a rebuild as end-state-equivalent to deleting .ssgrep/ and
re-indexing from scratch, which by definition cannot recover sessions that
are absent from disk at rebuild time. Routine incremental indexing never
does this; only rebuild does.

A rebuild may not silently shrink the index. That end-state equivalence is
also what makes the rebuild path the only one that can destroy a buyer's
history: the commit swaps in whatever this run discovered and unlinks what
it replaces. A scope mismatch makes discovery return nothing, so the commit
used to install an EMPTY generation over a healthy one and report success.
rebuild_guard.commit_or_refuse() gates the commit below on old-vs-new session
AND chunk counts and is deliberately placed after the staged generation is
complete but before commit_generation() touches the manifest, so a refusal
leaves the live generation both current and untouched; the staged one is
discarded. The incremental path writes into the live generation in place and
cannot clobber it, so it is not gated. Note this is not a --rebuild-only
concern: force_rebuild below is `rebuild or needs_rebuild(...)`, so any
release that bumps SCHEMA_VERSION or the embedding model sends a plain
`ssgrep index` down the same path.

The gate's two count triples are read with matching filters (see
rebuild_guard.read_counts): tombstoned rows are excluded from both, because a
rebuild structurally cannot reproduce them, and no_subagents restricts both to
main sessions, because `--no-subagents` drops whole sessions by request. Left
unmatched, either one is a cliff rather than attrition and hard-blocks a buyer
who did nothing wrong.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shlex
import sqlite3
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ssgrep import (
    chunker,
    discovery,
    discovery_roots,
    embed,
    episodes,
    indexer_support,
    metadata,
    notes,
    paths,
    rebuild_guard,
    records,
    store,
    vectors,
)
from ssgrep import signal as signal_mod
from ssgrep.types import (
    Chunk,
    ContentType,
    Episode,
    FileCursor,
    IndexNotFoundError,
    IndexStats,
    SessionFile,
)


@dataclass
class _RunStats:
    files_appended: int = 0
    files_full_parsed: int = 0
    files_missing_at_open: int = 0
    files_vanished: int = 0
    files_reappeared: int = 0
    malformed_records: int = 0
    skipped_records: int = 0
    episodes_created: int = 0
    chunks_created: int = 0
    #: Queue hints dropped by the drain's scope filter (indexer_support).
    queue_items_out_of_scope: int = 0


def index(
    project_dir: Path,
    *,
    rebuild: bool = False,
    no_subagents: bool = False,
    quiet: bool = False,
    allow_shrink: bool = False,
    index_dir: Path | None = None,
    scope: str | None = None,
) -> IndexStats:
    """Build or update the local index for project_dir's session transcripts.

    Mirrors ssgrep.api.index()'s contract; index_dir is an extra keyword-only
    override (defaulting to project_dir/".ssgrep" per D2) that lets callers
    keep index state separate from the scanned project, mainly for tests.

    `scope` decouples WHICH transcripts are discovered from WHERE the index
    lives (see docs: an org rename or repo move strands history recorded
    under the old path; `--scope /old/path` indexes it into the current
    project's index). Defaults to project_dir -- fully backward compatible.
    The scope is persisted in the index's meta table so the search-time
    staleness machinery compares against the scope the index was actually
    built with (without that, a foreign-scope index looks 100% vanished at
    search time and staleness advises a destructive rebuild). Changing an
    existing index's scope forces a full rebuild, exactly like a model
    change, so an index never silently mixes two scopes' discovery sets.

    Per D13, drains the work queue as part of reconciliation, processing hints
    enqueued by SessionEnd hooks.

    On the rebuild path only, refuses (RebuildWouldShrinkError) to commit a
    generation that would drop the indexed session count to zero or below half
    the live one, unless allow_shrink is set -- see rebuild_guard for why an
    unguarded commit here was the worst failure this tool had.
    """
    # Detect missing transcript root early and surface it distinguishably
    # from an empty scope (both are valid but require different error messages)
    # Raise IndexNotFoundError (a SearchException) to let the CLI/MCP layer handle it,
    # rather than exiting directly here (layering violation).
    transcript_root = paths.resolve_claude_dir() / "projects"
    if not transcript_root.exists():
        raise IndexNotFoundError(
            f"Cannot build index — no Claude Code transcripts at {transcript_root}. "
            "Set CLAUDE_CONFIG_DIR if Claude Code's config lives elsewhere.",
            condition="missing_index",
            # NOT IndexNotFoundError's default of "ssgrep index": the command
            # that just failed is an infinite loop for any agent or script
            # that follows the machine-readable remedy field. `mkdir -p` is
            # not the answer either -- manufacturing an empty root silences
            # the only signal a mis-configured user ever gets.
            command="export CLAUDE_CONFIG_DIR=<path to your .claude directory>",
        )

    index_dir = index_dir or (project_dir / ".ssgrep")
    gen_store = store.GenerationalStore(index_dir)
    # Repair any half-finished two-file commit left by a previous crash
    # BEFORE anything reads or opens those files. Ordering is load-bearing
    # three ways: recover() truncates vectors.f32 in place, so it must run
    # before open_vectors() mmaps it; it deletes chunk rows, so it must run
    # before init_db() holds a connection; and it must run before
    # rebuild_guard.needs_rebuild()'s validate_alignment(), or a crash's trailing
    # unreferenced vector rows read as corruption and escalate a cheap
    # in-place truncation into a full re-embed of the whole corpus.
    # Safe on a fresh run: with no index.db (or no index_dir at all) this
    # returns (0, 0) and creates nothing.
    #
    # Everything through checkpoint() below runs under a SHARED hold on the
    # generation recover() just repaired (gen_store.current_generation, as
    # loaded from the manifest above) -- guarding against a concurrent
    # rebuild's commit_generation() unlinking those exact live files while
    # this call still has them open (see hold_generation()'s and
    # _cleanup_old_generations()'s docstrings, and this module's
    # "Concurrent rebuild vs. incremental run" note). If this call turns
    # out to be a rebuild itself, the guard is released immediately below
    # -- a rebuild writes to its own brand-new next_gen, never to the
    # generation being guarded, so holding the lock any longer would only
    # block that generation's eventual cleanup for no reason.
    guard = ExitStack()
    guard.enter_context(gen_store.hold_generation(gen_store.current_generation))
    try:
        gen_store.recover()
        live_db, live_vec = gen_store.get_index_path(), gen_store.get_vector_path()

        model_id, dimension = embed.get_model_info()
        # Scope is an INDEX property, not a call property: an omitted scope
        # means "keep the scope this index was built with", falling back to
        # project_dir only when nothing is persisted (fresh or legacy index).
        # The first shipped version defaulted omitted-scope to project_dir --
        # a unanimous blind-review blocker: every routine caller that omits
        # the flag (plain `ssgrep index`, MCP startup reconciliation,
        # `ssgrep init`) was treated as an explicit scope change BACK to
        # project_dir, either locking scoped indexes out via the shrink
        # guard or silently discarding scoped history, and dequeuing
        # SessionEnd work items into a doomed staged generation. Only an
        # EXPLICIT, different --scope changes scope.
        persisted_scope = indexer_support.read_persisted_scope(live_db)
        effective_scope = indexer_support.effective_scope_from(scope, persisted_scope, project_dir)
        if (
            scope is None
            and persisted_scope is not None
            and persisted_scope != str(project_dir)
            and not quiet
        ):
            # The moved-project visibility line: before scope persistence,
            # this situation surfaced LOUDLY (empty discovery -> shrink
            # refusal + census). Persistence keeps the scoped history safe
            # and incremental -- but sessions recorded under the CURRENT
            # directory are outside scope and silently untracked, so say so
            # every run rather than letting that become invisible.
            print(
                f"ssgrep: note: this index tracks scope {persisted_scope} "
                f"(persisted). Sessions recorded under {project_dir} are not "
                f"included; pass --scope {shlex.quote(str(project_dir))} to "
                f"retarget (full rebuild)."
            )
        scope_changed = persisted_scope is not None and persisted_scope != effective_scope
        if scope_changed and not quiet:
            print(
                f"ssgrep: index scope changing from {persisted_scope} to "
                f"{effective_scope}; a full rebuild is required and will run now."
            )
        force_rebuild = (
            rebuild
            or scope_changed
            or rebuild_guard.needs_rebuild(live_db, live_vec, model_id, dimension)
        )

        next_gen = None
        retiring_counts = (0, 0, 0)
        if force_rebuild:
            # Read what this rebuild would retire BEFORE staging over it, so
            # the pre-commit shrink gate below has something to compare against.
            retiring_counts = rebuild_guard.read_counts(live_db, main_only=no_subagents)
            guard.close()  # not touching the guarded (old) generation's files
            next_gen = gen_store.current_generation + 1
            db_path, vec_path = gen_store.stage_generation(next_gen)
        else:
            db_path, vec_path = live_db, live_vec

        conn = store.init_db(db_path)
        vec_store = vectors.open_vectors(vec_path, dimension=dimension)
        store.set_meta(conn, "model_id", model_id)
        store.set_meta(conn, "vector_dimension", str(dimension))
        store.set_meta(conn, "scope", effective_scope)

        stats = _RunStats()
        corpus_sessions = list(
            discovery.discover_sessions(
                project_dir, scope=effective_scope, no_subagents=no_subagents
            )
        )
        # Notes are index-owned authored content (notes.py): always included,
        # independent of scope -- they were written FOR this index.
        # External roots (discovery_roots.py) are explicit user configuration:
        # likewise always included, bypassing cwd/scope matching.
        discovered = (
            corpus_sessions + notes.discover_notes(index_dir) + discovery_roots.discover_external()
        )
        corpus_session_count = len(corpus_sessions)

        # Drain the work queue: process hints left by SessionEnd hooks.
        # Items are reconciliation hints, not source of truth; work queue is optional.
        # A corrupt or unreadable queue degrades to no-op; a failed drain does not
        # prevent normal indexing from proceeding.
        indexer_support.drain_workqueue(
            conn, index_dir, project_dir, vec_store, stats, quiet, scope=effective_scope
        )

        if not quiet:
            print(f"ssgrep: indexing {len(discovered)} discovered transcript(s)")

        for session in sorted(discovered, key=lambda s: str(s.path)):
            _index_file(conn, vec_store, session, stats=stats)

        if not force_rebuild:
            _reconcile_vanished_and_reappeared(conn, stats)

        now = datetime.now(UTC)
        store.set_meta(conn, "last_index_time", now.isoformat())
        store.set_meta(conn, "skipped_records", str(stats.skipped_records))
        store.set_meta(conn, "queue_items_out_of_scope", str(stats.queue_items_out_of_scope))
        store.set_meta(conn, "corpus_session_count", str(corpus_session_count))
        store.set_meta(conn, "malformed_records", str(stats.malformed_records))
        conn.commit()

        if not force_rebuild:
            # The SQLite commit above is the point every chunk row became
            # durable, and every vector row they reference was appended before
            # it -- so journal that row count now, as the new known-good
            # watermark for the next recover(). Only on the incremental path:
            # a rebuild's manifest write is commit_generation()'s job below,
            # and checkpointing here would pin the OLD generation number to the
            # NEW generation's row count, which is precisely the torn state
            # recover() would then act on. (If a concurrent rebuild has since
            # superseded this generation, checkpoint() itself detects that
            # against the on-disk manifest and silently no-ops rather than
            # clobbering it -- see checkpoint()'s docstring.)
            gen_store.checkpoint(vec_store.row_count)

        session_count, episode_count, chunk_count = _read_counts(conn)
        tombstoned_sources, tombstoned_chunks = store.get_tombstone_stats(conn)

        conn.close()
        vectors.close(vec_store)
    finally:
        guard.close()

    if force_rebuild:
        assert next_gen is not None
        rebuild_guard.commit_or_refuse(
            gen_store,
            next_gen,
            retiring_counts,
            allow_shrink=allow_shrink,
            no_subagents=no_subagents,
        )

    # Add .ssgrep/ to .gitignore to prevent accidental commits of cached data
    indexer_support.add_to_gitignore(project_dir)

    return IndexStats(
        session_count=session_count,
        episode_count=episode_count,
        chunk_count=chunk_count,
        index_size_bytes=_dir_size(index_dir),
        last_index_time=now,
        model_id=model_id,
        vector_dimension=dimension,
        skipped_records=stats.skipped_records,
        malformed_records=stats.malformed_records,
        schema_version=store.SCHEMA_VERSION,
        tombstoned_source_count=tombstoned_sources,
        tombstoned_chunk_count=tombstoned_chunks,
        index_exists=True,
        queue_items_out_of_scope=stats.queue_items_out_of_scope,
        corpus_session_count=corpus_session_count,
    )


def _index_file(
    conn: sqlite3.Connection,
    vec_store: vectors.VectorStore,
    session: SessionFile,
    *,
    stats: _RunStats,
) -> None:
    """Incrementally (or fully) index one transcript file.

    The corpus mutates underneath every long-running scan, so every touch of
    session.path below is wrapped for FileNotFoundError/OSError: a file that
    vanished or became unreadable since discovery is a normal, counted skip,
    never a crash. Genuine disappearance is detected and tombstoned
    separately in _reconcile_vanished_and_reappeared(), keyed off the
    sessions table rather than this function's control flow, so a transient
    read failure here never marks a still-present file absent.
    """
    try:
        disk_stat = session.path.stat()
        current_hash = _first_line_hash(session.path)
    except (FileNotFoundError, OSError):
        stats.files_missing_at_open += 1
        return

    disk_size, disk_mtime = disk_stat.st_size, disk_stat.st_mtime
    existing_cursor = store.get_session_file(conn, session.path)
    need_full = _needs_full_reparse(existing_cursor, disk_size, current_hash)

    if need_full and existing_cursor is not None:
        _delete_rows_for_session(conn, session.session_id)

    start_offset = 0 if need_full else existing_cursor.byte_offset  # type: ignore[union-attr]
    if not need_full and start_offset >= disk_size:
        return  # nothing appended since last run: no read, no encode(), no writes

    try:
        new_records, end_offset, rstats = _read_from_offset(
            session.path, start_offset, records.MAX_LINE_BYTES
        )
    except (FileNotFoundError, OSError):
        stats.files_missing_at_open += 1
        return

    stats.malformed_records += rstats.malformed_lines
    stats.skipped_records += rstats.skipped_oversized
    stats.files_full_parsed += int(need_full)
    stats.files_appended += int(not need_full)

    agent_meta = None
    if not session.is_main:
        agent_meta = metadata.load_agent_meta(session.path.with_suffix(".meta.json"))

    enriched_session = session
    if agent_meta is not None:
        enriched_session = dataclasses.replace(
            session,
            agent_type=agent_meta.agent_type or session.agent_type,
            agent_name=agent_meta.name or session.agent_name,
            agent_description=agent_meta.description or session.agent_description,
            agent_model=agent_meta.model or session.agent_model,
        )
    store.insert_session(conn, enriched_session)

    if new_records:
        start_index = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE session_id = ?", (session.session_id,)
        ).fetchone()[0]
        new_episodes = _build_episodes(new_records, enriched_session, agent_meta, start_index)
        new_chunks: list[Chunk] = []
        for ep in new_episodes:
            store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)
            new_chunks.extend(chunker.chunk_episode(ep))
        stats.episodes_created += len(new_episodes)

        if new_chunks:
            vecs = embed.encode([c.text for c in new_chunks])
            vec_rows = vectors.append(vec_store, vecs)  # durable before any chunk references it
            for chunk, vec_row in zip(new_chunks, vec_rows, strict=True):
                store.insert_chunk(conn, chunk, vec_row)
            stats.chunks_created += len(new_chunks)

    store.upsert_session_file(
        conn,
        FileCursor(
            path=session.path,
            size=disk_size,
            mtime=disk_mtime,
            byte_offset=end_offset,
            first_line_hash=current_hash,
        ),
    )


def _build_episodes(
    new_records: list[dict],
    session: SessionFile,
    agent_meta: metadata.AgentMeta | None,
    start_index: int,
) -> list[Episode]:
    """Segment, renumber, and enrich the episodes found in new_records.

    episodes.segment_episode_groups() is the single segmentation pass: it
    returns each episode paired with the exact records that produced it, so
    there is no second, separately maintained grouping computation that
    could disagree with it and no whole-batch fallback to disagree into.
    (There used to be one -- see that function's docstring for what it cost:
    a group-count mismatch caused every episode to be chunked from the
    ENTIRE batch of new_records, multiplying the index by the episode
    count.) Each `group` below is therefore always exactly episode `ep`'s
    own records, never any other episode's.
    """
    grouped = episodes.segment_episode_groups(new_records, session.session_id)

    built: list[Episode] = []
    for i, (ep, group) in enumerate(grouped):
        # harvest_metadata is now the sole title authority; pass episode_index for ordinal fallback
        epmeta = metadata.harvest_metadata(
            group, session.session_id, agent_meta, episode_index=start_index + i
        )
        clean_prompt, clean_response = _extract_episode_text(group)
        title = epmeta.title

        built.append(
            dataclasses.replace(
                ep,
                episode_id=f"{session.session_id}:ep:{start_index + i}",
                prompt_text=clean_prompt or ep.prompt_text,
                response_text=clean_response or ep.response_text,
                title=title,
                timestamp=epmeta.timestamp,
                git_branch=epmeta.git_branch,
                cwd=epmeta.cwd,
                files_touched=epmeta.files_touched,
                tool_names=epmeta.tool_names,
                is_subagent=not session.is_main,
                agent_type=session.agent_type,
                agent_name=session.agent_name,
                agent_description=session.agent_description,
                parent_session_id=session.parent_session_id,
            )
        )
    return built


def _extract_episode_text(group: list[dict]) -> tuple[str, str]:
    """Derive clean prompt/response text for one episode via signal.py.

    Reuses signal.classify_signal() (markup stripping, away_summary,
    tool-result/thinking exclusion) instead of episodes.py's own inline
    extraction, keeping the two content_type streams separate.
    """
    prompt_parts: list[str] = []
    response_parts: list[str] = []
    for record in group:
        rec_type = record.get("type", "")
        if rec_type in ("user", "assistant"):
            content = record.get("message", {}).get("content", [])
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                result = signal_mod.classify_signal(record, block)
                if not (result.is_signal and result.text):
                    continue
                if result.content_type == ContentType.PROMPT:
                    prompt_parts.append(result.text)
                elif result.content_type == ContentType.RESPONSE:
                    response_parts.append(result.text)
        elif rec_type == "system":
            result = signal_mod.classify_signal(record)
            if result.is_signal and result.text and result.content_type == ContentType.RESPONSE:
                response_parts.append(result.text)
    return "\n".join(prompt_parts), "\n".join(response_parts)


def _needs_full_reparse(cursor: FileCursor | None, disk_size: int, current_hash: str) -> bool:
    """The Rewrite/Truncation Guard: resume only if size grew-or-held AND the
    first-line hash still matches; otherwise the file must be re-parsed whole.
    """
    if cursor is None:
        return True
    if disk_size < cursor.size:
        return True
    return current_hash != cursor.first_line_hash


def _read_from_offset(
    path: Path, start_offset: int, max_line_bytes: int
) -> tuple[list[dict], int, records.Stats]:
    """Parse only the bytes from start_offset to EOF. See module docstring."""
    new_records: list[dict] = []
    stats = records.Stats()
    pos = start_offset
    with open(path, "rb") as f:
        f.seek(start_offset)
        for raw_line in f:
            if not raw_line.endswith(b"\n"):
                break  # incomplete final line: wait for it to be finished on a later run
            pos += len(raw_line)
            stats.total_lines += 1
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if len(line.encode("utf-8")) > max_line_bytes:
                stats.skipped_oversized += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.malformed_lines += 1
                continue
            stats.parsed_records += 1
            if records.classify_record(record) == records.RecordType.UNKNOWN:
                stats.unknown_types += 1
            new_records.append(record)
    return new_records, pos, stats


def _first_line_hash(path: Path) -> str:
    with open(path, "rb") as f:
        first_line = f.readline()
    return hashlib.sha256(first_line).hexdigest()


def _delete_rows_for_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Hard-delete one session's rows ahead of a full re-parse. See module
    docstring for why this doesn't call store.delete_session_chunks().
    """
    conn.execute(
        "DELETE FROM chunks_fts WHERE chunk_id IN "
        "(SELECT chunk_id FROM chunks WHERE session_id = ?)",
        (session_id,),
    )
    conn.execute(
        "DELETE FROM chunks_fts_tri WHERE chunk_id IN "
        "(SELECT chunk_id FROM chunks WHERE session_id = ?)",
        (session_id,),
    )
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM episodes WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def _reconcile_vanished_and_reappeared(conn: sqlite3.Connection, stats: _RunStats) -> None:
    """Tombstone sources whose recorded path no longer exists, and clear the
    tombstone on any that have reappeared. Existence is re-checked directly
    against disk here (D12/D13): a discovery run scoped with no_subagents
    never counts as evidence a subagent source has vanished.
    """
    rows = conn.execute("SELECT session_id, path, source_status FROM sessions").fetchall()
    for session_id, path_str, status in rows:
        exists = Path(path_str).exists()
        if not exists and status != "absent":
            store.tombstone_session_chunks(conn, session_id)
            stats.files_vanished += 1
        elif exists and status == "absent":
            store.mark_source_available(conn, session_id)
            stats.files_reappeared += 1


def _read_counts(conn: sqlite3.Connection) -> tuple[int, int, int]:
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    ep_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return sessions, ep_count, chunk_count


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
