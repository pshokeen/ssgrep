"""Rebuild decisions: when a full rebuild is required, and when one must be refused.

Two separate questions live here, both about the same rare, expensive, and --
until this module existed -- unconditionally destructive path.

Is a rebuild needed? needs_rebuild() answers that from the stored
schema_version / model_id / vector_dimension plus a vectors-vs-chunks
alignment check. It is deliberately read through read_existing_meta() rather
than store.init_db(), which unconditionally overwrites schema_version to the
current value on open -- a mismatch has to be observed before that call.

Would committing this rebuild destroy the buyer's index? That is the reason
this module exists. indexer.index() used to commit the rebuilt generation
unconditionally: there was no comparison of any kind between the generation
being retired and the one replacing it. A scope mismatch (running ssgrep from
a directory whose recorded cwd no longer matches, or after the project moved)
makes discovery return zero transcripts, and the resulting empty generation
was swapped over a healthy one -- printing "Indexed 0 sessions, 0 episodes, 0
chunks." and exiting 0 while hundreds of indexed sessions became
unrecoverable. The prior generation's files are unlinked by
commit_generation()'s cleanup, and nothing anywhere warned.

This is not a --rebuild-only hazard, which is what makes it a release-blocking
one: indexer.index() computes `rebuild or needs_rebuild(...)`, so a plain
`ssgrep index` immediately after any shipped upgrade that bumps SCHEMA_VERSION
or changes the embedding model takes the identical path with no flag at all.
Any such release would have been a mass-loss event for every mis-scoped buyer.

check_shrink() is therefore a pre-commit gate on the rebuild path only: it
refuses to retire a populated generation in favour of a drastically smaller
one, names old-vs-new counts and scope mismatch in the message, and offers an
explicit opt-in. The incremental path never clobbers -- it writes into the
live generation in place -- so it needs no such gate and does not get one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ssgrep import store, vectors
from ssgrep.store.generations import GenerationalStore
from ssgrep.types import RebuildWouldShrinkError

SHRINK_FLOOR = 0.5
"""A rebuild may not retire a populated generation for one holding fewer than
this fraction of its sessions, or of its chunks.

A judgement call, not a derived constant. The floor has to sit below any
plausible legitimate shrink and above any plausible accidental one. A buyer
who prunes old history will see the count fall gradually; the accidental case
is not gradual, because a scope mismatch does not shrink the corpus, it
empties it. 0.5 leaves ordinary attrition alone while catching the failure
that actually loses data, and --allow-shrink exists precisely because no
threshold can tell a deliberate halving from an accident.

Two legitimate shrinks are NOT left to this threshold, because both are
cliffs rather than attrition and both would hard-block a buyer who did
nothing wrong. They are excluded from the comparison instead, by
read_counts(), so the floor never sees them:

- Deleted transcripts. A forced rebuild cannot reproduce rows whose source
  file is gone (see indexer.py's "Forced rebuild loses tombstones by
  design"), so counting them on the retiring side compares a number that
  includes them against one that structurally cannot. read_counts()
  therefore counts only rows whose source file is still on disk RIGHT NOW,
  checked directly -- see _gone_session_ids() for why the stored tombstone
  flag is not trusted to answer that.
- --no-subagents. That flag drops whole sessions on purpose
  (discovery.py's `if no_subagents and not is_main: continue`), and subagents
  are the majority of a heavy Task-tool user's corpus. read_counts(
  main_only=True) restricts BOTH sides to main sessions so the deliberate
  exclusion cancels out and a genuine scope mismatch is still caught.
"""

_PRESENT = "source_status != 'absent'"
"""Tombstoned rows are excluded from every count. See SHRINK_FLOOR."""

_GONE_TABLE = "ssgrep_rebuild_guard_gone"
"""Temp table holding session ids whose transcript is no longer on disk."""


def read_existing_meta(db_path: Path) -> tuple[str | None, str | None, str | None]:
    """Read schema_version/model_id/vector_dimension without going through
    store.init_db(), which unconditionally overwrites schema_version to the
    current value on open -- a mismatch must be observed before that call.
    """
    if not db_path.exists():
        return None, None, None
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        finally:
            conn.close()
    except sqlite3.Error:
        return None, None, None
    return rows.get("schema_version"), rows.get("model_id"), rows.get("vector_dimension")


def needs_rebuild(db_path: Path, vec_path: Path, model_id: str, dimension: int) -> bool:
    """True when the on-disk index cannot be appended to and must be rebuilt."""
    stored_schema, stored_model, stored_dim = read_existing_meta(db_path)
    if stored_schema is not None and stored_schema != str(store.SCHEMA_VERSION):
        return True
    if stored_model is not None and stored_model != model_id:
        return True
    if stored_dim is not None and stored_dim != str(dimension):
        return True
    valid, _ = vectors.validate_alignment(db_path, vec_path)
    return not valid


def _gone_session_ids(conn: sqlite3.Connection) -> None:
    """Stage, in a temp table, every session whose transcript is not on disk.

    Existence is checked against the filesystem here rather than read off
    ``source_status``, and that distinction is the whole point. Only ONE code
    path ever writes ``source_status = 'absent'``:
    indexer._reconcile_vanished_and_reappeared(), which runs exclusively
    under ``if not force_rebuild:``. So the flag records whether an
    incremental run happened to observe the deletion -- not whether the file
    is actually gone.

    Trusting it wedged the index permanently. A buyer archives most of a
    year-old corpus off to cold storage, then takes a routine upgrade that
    bumps SCHEMA_VERSION. From that moment needs_rebuild() is true forever,
    so indexer.py's `force_rebuild = rebuild or needs_rebuild(...)` sends
    EVERY `ssgrep index` down the gated path -- which means the incremental
    path, the only writer of the tombstone flag, is unreachable. The
    exclusion SHRINK_FLOOR's docstring names as the reason deleted
    transcripts "never see the floor" then filters zero rows: old counts
    include the archived sessions, the rebuild structurally cannot reproduce
    them, and the refusal fires. It fires again on every subsequent run,
    with or without --rebuild, because a refusal commits nothing and the
    live schema_version never advances. The index stops updating AND stops
    being readable (search exits 4 telling the buyer to run the rebuild that
    exits 2), and the only escape is the --allow-shrink flag the same message
    calls irreversible -- against the sole surviving copy of that history.

    Checking disk makes the invariant SHRINK_FLOOR asserts true by
    construction instead of true only if an incremental run happened to go
    first.
    """
    conn.execute(f"CREATE TEMP TABLE {_GONE_TABLE} (session_id TEXT PRIMARY KEY)")
    rows = conn.execute("SELECT session_id, path FROM sessions").fetchall()
    conn.executemany(
        f"INSERT OR IGNORE INTO {_GONE_TABLE} VALUES (?)",
        [(session_id,) for session_id, path in rows if not path or not Path(path).exists()],
    )


def read_counts(db_path: Path, *, main_only: bool = False) -> tuple[int, int, int]:
    """Return (sessions, episodes, chunks) for an index that may not exist.

    Counts only rows a rebuild could actually reproduce, so the two sides of
    check_shrink()'s comparison measure the same population (see SHRINK_FLOOR):

    - Rows whose transcript is no longer on disk are always excluded, whether
      or not anything has tombstoned them yet. A rebuild cannot bring those
      back, so counting them on the retiring side would refuse a rebuild for
      a buyer whose only "loss" was deleting files months ago. See
      _gone_session_ids() -- reading this off ``source_status`` alone is what
      made the refusal permanent.
    - ``main_only`` additionally excludes subagent sessions and everything
      hanging off them, for callers running with --no-subagents.

    Returns (0, 0, 0) for a missing or unreadable database rather than
    raising: this runs against the generation a rebuild is about to retire,
    which by definition may be absent (first ever run) or damaged (a rebuild
    triggered precisely because it failed validation). "Unknown" and "empty"
    both mean the same thing to check_shrink() -- there is nothing measurable
    to lose, so it must not block the rebuild that is trying to fix things.
    """
    if not db_path.exists():
        return 0, 0, 0
    here = f"{_PRESENT} AND session_id NOT IN (SELECT session_id FROM {_GONE_TABLE})"
    main_sessions = f"SELECT session_id FROM sessions WHERE is_main = 1 AND {here}"
    session_filter = " AND is_main = 1" if main_only else ""
    episode_filter = " AND COALESCE(is_subagent, 0) = 0" if main_only else ""
    chunk_filter = f" AND session_id IN ({main_sessions})" if main_only else ""
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            _gone_session_ids(conn)
            counts = tuple(
                conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {here}{extra}").fetchone()[0]
                for table, extra in (
                    ("sessions", session_filter),
                    ("episodes", episode_filter),
                    ("chunks", chunk_filter),
                )
            )
        finally:
            conn.close()
    except sqlite3.Error:
        return 0, 0, 0
    return counts[0], counts[1], counts[2]


def check_shrink(
    old_counts: tuple[int, int, int],
    new_counts: tuple[int, int, int],
    *,
    allow_shrink: bool = False,
    no_subagents: bool = False,
) -> None:
    """Raise RebuildWouldShrinkError if committing new_counts over old_counts
    would drop the indexed session count OR the indexed chunk count to zero,
    or below SHRINK_FLOOR of it.

    Chunks are gated as well as sessions because a session row is not a proxy
    for indexed content. indexer._index_file() calls store.insert_session()
    unconditionally for every discovered transcript, before and independently
    of the `if new_records:` block that builds episodes and chunks -- so a
    regression anywhere downstream of discovery (an upstream transcript-format
    change, an episode-builder or chunker bug, a records.MAX_LINE_BYTES
    change) yields the identical session count with an empty corpus. Gating on
    sessions alone let that commit, unlink the previous generation, and exit 0
    with "Indexed N sessions, 0 episodes, 0 chunks" -- the same end state this
    module exists to prevent, differing only in which printed number is
    nonzero. It is invisible to the compensating diagnostic too, since
    cli/commands/index.py gates that on `session_count == 0`. Chunks are what
    search actually reads, and check_shrink() was already being handed them.

    ``old_chunks == 0`` skips the chunk arm: an index with sessions but no
    chunks has no searchable content to lose, and blocking there would wedge
    the rebuild that is trying to fix it.

    Call this AFTER the staged generation is fully built but BEFORE
    commit_generation() touches the manifest, so a refusal leaves the live
    generation both current and intact.
    """
    old_sessions, new_sessions = old_counts[0], new_counts[0]
    old_chunks, new_chunks = old_counts[2], new_counts[2]
    if allow_shrink or old_sessions == 0:
        return
    sessions_ok = new_sessions > 0 and new_sessions >= old_sessions * SHRINK_FLOOR
    chunks_ok = old_chunks == 0 or (new_chunks > 0 and new_chunks >= old_chunks * SHRINK_FLOOR)
    if sessions_ok and chunks_ok:
        return
    subagent_note = (
        "Subagent sessions were excluded from BOTH counts above, so "
        "--no-subagents alone does not explain this.\n"
        if no_subagents
        else ""
    )
    raise RebuildWouldShrinkError(
        "Refusing to replace the existing index with a much smaller one.\n"
        f"  existing: {old_counts[0]} sessions, {old_counts[1]} episodes, "
        f"{old_counts[2]} chunks\n"
        f"  rebuilt:  {new_counts[0]} sessions, {new_counts[1]} episodes, "
        f"{new_counts[2]} chunks\n"
        "Counts alone cannot tell these causes apart, so nothing was committed:\n"
        "  - a scope mismatch: discovery found few or no transcripts for this\n"
        "    project directory, e.g. ssgrep was run from a different directory\n"
        "    than your sessions were recorded in, or the project was moved or\n"
        "    renamed. The census below lists the working directories your\n"
        "    transcripts actually record.\n"
        "  - a regression in parsing or chunking, which would yield the same\n"
        "    transcripts with far less indexed content.\n"
        "Transcripts you deleted from disk are NOT a cause: both counts above\n"
        "already exclude every session whose transcript is gone, so deleting\n"
        "old history can never trigger this refusal.\n"
        f"{subagent_note}"
        "Committing this rebuild would discard the sessions above with no way "
        "to get them back.\n"
        "Your existing index has NOT been modified. If the smaller index is "
        "genuinely what you want, re-run with --allow-shrink.",
        old_counts=old_counts,
        new_counts=new_counts,
    )


def commit_or_refuse(
    gen_store: GenerationalStore,
    next_gen: int,
    retiring_counts: tuple[int, int, int],
    *,
    allow_shrink: bool = False,
    no_subagents: bool = False,
) -> None:
    """Gate the staged generation, then commit it -- or discard it and raise.

    The last point at which the live generation is still both intact and
    current: commit_generation() writes the manifest and only then unlinks
    what it supersedes. A refusal must therefore also drop the staged
    generation, or the buyer pays disk for a rebuild they were told did not
    happen.

    The staged counts are re-read from the staged database with the SAME
    filters used for ``retiring_counts``, rather than taken from whatever the
    caller happened to tally. Symmetry is the whole property being checked;
    comparing a filtered number against an unfiltered one is how the tombstone
    and --no-subagents cliffs got in.
    """
    new_counts = read_counts(gen_store.get_index_path(next_gen), main_only=no_subagents)
    try:
        check_shrink(
            retiring_counts, new_counts, allow_shrink=allow_shrink, no_subagents=no_subagents
        )
    except RebuildWouldShrinkError:
        gen_store.discard_generation(next_gen)
        raise
    gen_store.commit_generation(next_gen)
