"""Genuine concurrency tests for indexer.index(): two real index() calls
racing against the same index directory via real OS threads, real file
I/O, and real SQLite locking -- not a simulated interleaving.

local-index spec's Concurrency and Crash Safety requirement: "Concurrent
runs of index() against the same store SHALL NOT corrupt it (via SQLite's
own locking, or an equivalent mechanism)." Before this file, the only tests
in the suite with "concurrent" in their name were in test_workqueue.py, and
those never launch a second thread or process at all -- they call two
WorkQueue instances back-to-back in a specific order to simulate what
concurrency might produce. No test anywhere raced two real, simultaneously
running index() calls. This file does: threading.Thread workers, started
together via a threading.Barrier so the OS scheduler -- not test code --
decides how the two runs actually interleave.

embed.encode() is monkeypatched to a fast, deterministic, content-derived
stand-in (same rationale as test_indexer.py: these tests exercise
concurrency/locking, not embedding quality). Deterministic-by-content is
load-bearing here, not just a speed shortcut: it lets
_assert_every_chunk_points_at_its_own_vector() recompute "what vector
should this exact chunk_id's row hold" after the fact and compare it
against what is actually sitting at that row in vectors.f32 -- a strictly
stronger check than vectors.validate_alignment(), which only confirms every
vec_row is in-bounds and unorphaned. A chunk silently pointing at the WRONG
in-bounds, singly-referenced row (a "row swap") would pass
validate_alignment() cleanly while still returning wrong search results;
only re-deriving the expected vector from content catches that.

Two scenarios, with two different outcomes:

1. test_two_concurrent_plain_index_calls_stay_consistent -- two ordinary
   (rebuild=False) index() calls against an overlapping corpus, the
   realistic case of e.g. a SessionEnd hook auto-indexing while a user runs
   `ssgrep index` by hand. This PASSES. It is not safe by any explicit
   coordination in indexer.py for this specific race; it is safe as a side
   effect of store.set_meta()'s write acquiring SQLite's WAL writer lock
   right at the top of index() (before discovery or the per-file loop even
   run), combined with index() never calling conn.commit() until the very
   end. Whichever of the two calls wins that lock runs its entire pass
   uncontended; the other blocks on its own first write for the winner's
   whole duration, and by the time it unblocks, every file the winner
   touched already has an up-to-date cursor, so the loser's per-file loop
   finds nothing left to do and finishes having written nothing further.
   Confirmed by instrumented tracing before this test was written, not just
   theorized -- see the docstring on that test for the observed trace.

2. test_concurrent_rebuild_cannot_destroy_a_racing_incremental_run -- a
   rebuild (rebuild=True, e.g. a user running `ssgrep index --rebuild`, or
   an automatic rebuild triggered by a schema/model version bump) racing an
   in-flight ordinary index() call. This is now safe, but not for free.
   GenerationalStore.commit_generation()'s _cleanup_old_generations() used
   to unconditionally unlink the previous generation's index.db/vectors.f32
   the moment the rebuild committed, with no awareness that a concurrently
   running incremental index() call still had those exact files open and
   was mid-write -- a real, previously-confirmed defect, not a hypothetical
   one. Two independent mechanisms now prevent it, and a fix covering only
   one of them would still lose data: a generation-scoped advisory lock
   that defers cleanup of a generation for as long as any index() call
   still holds it, and a checkpoint() that refuses to write the manifest
   once the on-disk generation has moved past what it believes is current.
   See that test's docstring for the full mechanism, what the defect used
   to produce, and why both parts are load-bearing.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssgrep import api, embed, indexer, store, vectors

DIM = 256


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_indexer.py's fake_home / _write_jsonl /
# _user_record / _assistant_record patterns; kept local to this file rather
# than imported so this file has no cross-file coupling with other tests
# under concurrent edit elsewhere in the suite).
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import ssgrep.discovery as discovery_mod

    discovery_mod._cwd_index_cache.clear()
    return home


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Deterministic, content-derived: the same text always maps to the
    same vector, regardless of which thread or in what order it is
    computed. This is what lets the corruption check below recompute the
    "correct" vector for any chunk after the fact.
    """
    import hashlib

    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        out[i] = rng.standard_normal(DIM).astype(np.float32)
    return out


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "encode", _fake_encode)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _user_record(uid: str, text: str, cwd: str, session_id: str) -> dict[str, Any]:
    return {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": text},
        "uuid": uid,
        "timestamp": "2026-07-01T10:00:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


def _assistant_record(
    uid: str, text: str, cwd: str, session_id: str, parent_uuid: str
) -> dict[str, Any]:
    return {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "uuid": uid,
        "timestamp": "2026-07-01T10:01:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


N_TURNS = 4
CHUNKS_PER_FILE = N_TURNS * 2  # one prompt + one response chunk per episode


def _make_session(path: Path, session_id: str, marker: str, cwd: str) -> None:
    """A synthetic session file with N_TURNS user/assistant turns, each
    carrying `marker` plus enough unique per-turn text that every chunk in
    the whole test gets distinct, individually identifiable content.
    """
    records = []
    for i in range(N_TURNS):
        uid, aid = f"{session_id}-u{i}", f"{session_id}-a{i}"
        records.append(
            _user_record(uid, f"{marker} unique prompt turn {i} {session_id}", cwd, session_id)
        )
        records.append(
            _assistant_record(
                aid, f"{marker} unique response turn {i} {session_id}", cwd, session_id, uid
            )
        )
    _write_jsonl(path, records)


def _fetch_all_chunks(db_path: Path) -> list[tuple[str, str, int | None]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT chunk_id, text, vec_row FROM chunks").fetchall()
    finally:
        conn.close()


def _assert_every_chunk_points_at_its_own_vector(db_path: Path, vec_path: Path) -> None:
    """The strong corruption check described in the module docstring: for
    every chunk row, recompute the vector its own text SHOULD produce (via
    the same deterministic fake_encode the indexing run used) and compare
    it against whatever vector is actually sitting at its stored vec_row.

    Also checks for vec_row aliasing (two chunks claiming the same row) --
    the direct, textbook signature of two racing appenders each computing
    row numbers from a stale in-memory row_count.
    """
    rows = _fetch_all_chunks(db_path)
    vstore = vectors.open_vectors(vec_path, dimension=DIM)

    out_of_bounds = [(cid, vr) for cid, _t, vr in rows if vr is None or vr >= vstore.row_count]
    assert not out_of_bounds, f"chunks with out-of-range/missing vec_row: {out_of_bounds}"

    by_row: dict[int, list[str]] = {}
    for chunk_id, _text, vec_row in rows:
        by_row.setdefault(vec_row, []).append(chunk_id)
    aliased = {row: ids for row, ids in by_row.items() if len(ids) > 1}
    assert not aliased, (
        f"multiple chunks alias the same vec_row -- a hallmark of two "
        f"concurrent vectors.append() calls computing row numbers from a "
        f"stale row_count: {aliased}"
    )

    mismatched = []
    for chunk_id, text, vec_row in rows:
        expected = _fake_encode([text])[0]
        actual = vstore.array[vec_row]
        if not np.allclose(expected, actual, atol=1e-4):
            mismatched.append((chunk_id, vec_row))
    assert not mismatched, (
        f"chunks whose vec_row holds a DIFFERENT chunk's vector (row-swap "
        f"corruption, invisible to vectors.validate_alignment()): {mismatched[:10]}"
    )


# ---------------------------------------------------------------------------
# Scenario 1: two ordinary index() calls. This test PASSES.
# ---------------------------------------------------------------------------


def test_two_concurrent_plain_index_calls_stay_consistent(fake_home: Path, tmp_path: Path) -> None:
    """Two real, simultaneously started index() calls (rebuild=False on
    both) against the same project/index_dir, racing over a corpus that
    includes both already-indexed baseline files and brand-new files both
    calls will independently discover.

    Empirically observed trace of what actually happens (instrumented
    outside this test, then reproduced here as plain assertions): both
    threads reach store.set_meta()'s write of "model_id" at nearly the same
    instant -- the first real SQLite write of index()'s single long-lived
    transaction, which happens before discover_sessions() even runs.
    Whichever thread's connection wins that write acquires SQLite's WAL
    writer lock and holds it, uncontended, all the way through discovery,
    every file in its per-file loop, and its own conn.commit() at the very
    end. The other thread blocks inside that same early set_meta() call
    (busy_timeout=5000 retries under the hood) for the winner's entire run.
    By the time it unblocks, the winner has already committed cursors for
    every file that needed indexing, so the loser's own per-file loop --
    now reading a fresh, post-commit snapshot -- finds every file already
    up to date and returns having written nothing further. Net effect: the
    two runs behave like one clean, uncontended pass, not two overlapping
    ones, even though both were genuinely launched together and neither
    was told to wait for the other.

    This test does not assume that outcome -- it only asserts the
    properties that must hold regardless of which thread happens to win:
    no exception, an aligned index, no vec_row aliasing or row-swap
    corruption, exactly one clean pass worth of chunks (never double, never
    partial), and a real search() call finding the right, singular content
    afterward.
    """
    home = fake_home
    project_dir = home.parent / "project"
    project_dir.mkdir()
    index_dir = project_dir / ".ssgrep"
    cwd = str(project_dir)
    sessions_dir = home / ".claude" / "projects" / "proj"

    # Baseline: already indexed once, single-threaded, before the race --
    # the realistic "index already exists" starting point.
    baseline_files = [f"baseline-{i}" for i in range(4)]
    for name in baseline_files:
        _make_session(sessions_dir / f"{name}.jsonl", name, f"BASE-{name}", cwd)
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    # New content that neither run has seen yet; both threads' discovery
    # scans will independently find every one of these.
    new_files = [f"new-{i}" for i in range(10)]
    for name in new_files:
        _make_session(sessions_dir / f"{name}.jsonl", name, f"NEW-{name}", cwd)

    all_files = baseline_files + new_files
    expected_chunk_count = len(all_files) * CHUNKS_PER_FILE

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            indexer.index(project_dir, index_dir=index_dir, quiet=True)
        except BaseException as exc:  # noqa: BLE001 - captured for the main thread to report
            errors.append(exc)

    t1 = threading.Thread(target=worker, name="idx-1")
    t2 = threading.Thread(target=worker, name="idx-2")
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert (
        not t1.is_alive() and not t2.is_alive()
    ), "a concurrent index() call hung past its timeout"
    assert not errors, f"concurrent index() raised: {errors}"

    gen_store = store.GenerationalStore(index_dir)
    db_path, vec_path = gen_store.get_index_path(), gen_store.get_vector_path()

    # No crash residue: a genuinely torn manifest or an orphaned/dangling
    # vector row from the race would show up here as non-(0, 0).
    truncated, dropped = gen_store.recover()
    assert (truncated, dropped) == (0, 0), (
        f"recover() found residue after the race (truncated={truncated}, dropped={dropped}) "
        f"-- the manifest and the on-disk files disagree about how many vector rows are valid"
    )

    is_valid, reason = vectors.validate_alignment(db_path, vec_path)
    assert is_valid, f"index left unaligned by the race: {reason}"

    rows = _fetch_all_chunks(db_path)
    assert len(rows) == expected_chunk_count, (
        f"expected exactly {expected_chunk_count} chunks (one clean pass over "
        f"{len(all_files)} files), got {len(rows)} -- the race produced either "
        f"duplicated or lost content"
    )

    texts = [t for _cid, t, _vr in rows]
    assert len(texts) == len(
        set(texts)
    ), "duplicate chunk text under different chunk_ids after the race"

    _assert_every_chunk_points_at_its_own_vector(db_path, vec_path)

    # And finally: does a real search() call still return the right thing?
    # "turn 0" is included so the BM25 leg's implicit-AND-of-all-tokens
    # uniquely singles out episode :ep:0 among new-7's 4 near-identical
    # turns (they differ only by turn number), rather than relying on
    # score-tie-break luck across four otherwise-similar candidates.
    response = api.search(project_dir, "NEW-new-7 unique prompt turn 0")
    assert response.results, "search() found nothing for content indexed during the race"
    assert (
        response.results[0].ref == "new-7:ep:0"
    ), f"search() returned the wrong episode after the race: {response.results[0].ref!r}"
    assert "NEW-new-7" in response.results[0].excerpt


# ---------------------------------------------------------------------------
# Scenario 2: a rebuild racing an in-flight incremental run. This test
# PASSES -- it guards against a defect that used to make this race
# destructive; see module docstring and the test's own docstring for the
# two-part fix.
# ---------------------------------------------------------------------------


def test_concurrent_rebuild_cannot_destroy_a_racing_incremental_run(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild (rebuild=True) racing an ordinary, still-running index()
    call against the same index_dir.

    This guards against a real, previously-confirmed defect in
    store.GenerationalStore.commit_generation(): its
    _cleanup_old_generations() step used to unconditionally unlink the
    superseded generation's index.db and vectors.f32 the instant the
    rebuild committed, with no coordination with any other index() call
    that might still be writing to those exact files. Two independent
    mechanisms now close this, and a fix covering only one of them would
    still lose data -- just less catastrophically:

    1. GenerationalStore.hold_generation() takes a SHARED fcntl.flock on a
       small per-generation lock file (created on first use, never
       deleted -- deleting it would let some future opener acquire a new
       inode's lock that no longer excludes the original holder).
       indexer.index() holds this lock, for gen_store.current_generation,
       across its entire live-file span -- from recover() through the
       final checkpoint()/close() -- releasing it early on the rebuild
       branch (which never touches the old generation) or in a finally on
       the incremental branch. _cleanup_old_generations() now takes that
       same lock EXCLUSIVELY and non-blocking before unlinking a
       generation's files; if an incremental run still holds it SHARED,
       the attempt fails immediately and that generation is left in
       place, picked up by a later commit_generation() call once nothing
       holds the lock (cleanup re-scans every generation older than
       current on each call, so nothing is permanently skipped).
    2. checkpoint() now re-reads the on-disk manifest immediately before
       writing and silently no-ops if the current on-disk generation
       disagrees with this GenerationalStore object's own, possibly
       stale, idea of the current generation. Without this, mechanism 1
       alone would still leave a bug: it stops the incremental run's
       *files* from being destroyed, but the incremental run's in-memory
       generation number was captured before the race, so its own final
       checkpoint() would still silently revert the manifest from the
       rebuild's newly-committed generation back to the old one --
       discarding a completed rebuild without corrupting anything, which
       is just as real a data-loss bug.

    To make an otherwise timing-dependent race deterministic and
    repeatable in CI (a bare threading.Barrier start, as in the test
    above, would only very rarely land the rebuild's commit inside the
    incremental run's write window), vectors.open_vectors() is
    monkeypatched to pause the INCREMENTAL thread, via a real
    threading.Event, at the exact point a real slow incremental run (many
    files, a loaded disk, lock contention of its own) would still
    realistically be: right after it has opened generation 0's live
    vectors.f32, before its per-file loop has done any real work. The
    rebuild thread only starts once that pause is confirmed, and signals
    the incremental thread to resume only after its OWN commit_generation()
    call has returned. Every operation either thread performs from that
    point on is real, unmodified indexer.index()/store.py/vectors.py code;
    only the TIMING of one checkpoint is pinned, which is what turns an
    already-possible bad interleaving into a reliable regression test
    instead of a flaky one.

    Before the fix, this is the end state the race used to produce
    (confirmed by instrumented tracing when this test was first written,
    and reproduced again by reverting the fix under a mutation test):

    - generation 0's vectors.f32 no longer existed (unlinked by the
      rebuild's cleanup). The incremental thread's own vec_store object
      still believed it held N rows (read at open time, before the
      unlink); its next vectors.append() call reopened the path in "ab"
      mode, which *creates a brand-new, empty file* at that name (append
      mode creates-if-missing) and wrote into it starting at physical
      row 0 -- while still returning row numbers computed from the stale,
      pre-unlink row count. The chunks it inserted referenced rows the
      new, tiny file did not have.
    - generation 0's index.db no longer existed either. The incremental
      thread's sqlite3 connection, opened before the unlink, kept working
      against the now-nameless inode via its already-open file descriptor
      -- but every row it committed through that connection was written
      to an inode with no directory entry, so the instant that connection
      closed at the end of index(), all of it -- the untouched baseline
      content this run never even needed to re-touch, and the new content
      it was actually there to add -- was gone. Nothing referenced it
      anymore.
    - the incremental run's own final checkpoint() then overwrote the
      manifest back to generation 0 (its own GenerationalStore object's
      view of "current generation" was loaded before the rebuild ran and
      was never refreshed), even though generation 1 -- the rebuild's own
      fully valid, complete output -- was what was actually sitting on
      disk and correct.
    - net result: the manifest pointed at generation 0, whose index.db did
      not exist at all. The index was not just missing some data -- it
      was unreadable outright. Meanwhile generation 1's files, which were
      completely valid and DID contain everything (baseline and new
      content alike), sat on disk unreferenced by anything, due to be
      silently deleted the next time any commit_generation() ran.

    With both mechanisms above in place, the actual sequence is instead:
    cleanup of generation 0 is deferred while the incremental thread
    safely finishes its write; the rebuild's manifest write to
    generation 1 stands; the incremental thread's own checkpoint() then
    finds the on-disk generation (1) disagrees with its stale local copy
    (0) and no-ops instead of reverting it. The final state is
    generation 1 -- the rebuild's own complete, independently-discovered
    output, which happens to already include the concurrently-added file
    too, since it existed on disk before the rebuild's own discovery scan
    ran -- durable, valid, and correctly referenced by the manifest.
    Generation 0's now-superseded-but-never-corrupted files are picked up
    by cleanup on some future commit_generation() call, once nothing
    holds its lock.

    This test asserts what a correct implementation must guarantee: the
    index is readable and aligned afterward, and contains both the
    rebuild's content and the concurrently-added new content.
    """
    home = fake_home
    project_dir = home.parent / "project"
    project_dir.mkdir()
    index_dir = project_dir / ".ssgrep"
    cwd = str(project_dir)
    sessions_dir = home / ".claude" / "projects" / "proj"

    baseline_files = [f"base-{i}" for i in range(3)]
    for name in baseline_files:
        _make_session(sessions_dir / f"{name}.jsonl", name, f"BASE-{name}", cwd)
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    _make_session(sessions_dir / "newcontent.jsonl", "newcontent", "ZZNEWMARKER", cwd)
    all_files = [*baseline_files, "newcontent"]
    expected_chunk_count = len(all_files) * CHUNKS_PER_FILE

    incremental_paused = threading.Event()
    let_incremental_resume = threading.Event()
    real_open_vectors = vectors.open_vectors

    def gated_open_vectors(path: Path, dimension: int = 256):
        result = real_open_vectors(path, dimension=dimension)
        if threading.current_thread().name == "incremental":
            incremental_paused.set()
            if not let_incremental_resume.wait(timeout=10):
                raise TimeoutError("rebuild thread never signaled completion")
        return result

    monkeypatch.setattr(vectors, "open_vectors", gated_open_vectors)

    errors: list[tuple[str, BaseException]] = []

    def incremental_worker() -> None:
        try:
            indexer.index(project_dir, index_dir=index_dir, quiet=True, rebuild=False)
        except BaseException as exc:  # noqa: BLE001
            errors.append(("incremental", exc))

    def rebuild_worker() -> None:
        try:
            if not incremental_paused.wait(timeout=10):
                raise TimeoutError("incremental thread never reached its pause point")
            indexer.index(project_dir, index_dir=index_dir, quiet=True, rebuild=True)
        except BaseException as exc:  # noqa: BLE001
            errors.append(("rebuild", exc))
        finally:
            let_incremental_resume.set()

    t_inc = threading.Thread(target=incremental_worker, name="incremental")
    t_reb = threading.Thread(target=rebuild_worker, name="rebuild")
    t_inc.start()
    t_reb.start()
    t_inc.join(timeout=30)
    t_reb.join(timeout=30)

    assert (
        not t_inc.is_alive() and not t_reb.is_alive()
    ), "a concurrent index() call hung past its timeout"
    assert not errors, f"concurrent index() raised: {errors}"

    gen_store = store.GenerationalStore(index_dir)
    db_path, vec_path = gen_store.get_index_path(), gen_store.get_vector_path()

    try:
        rows = _fetch_all_chunks(db_path)
    except sqlite3.Error as exc:
        pytest.fail(
            f"the index the manifest points to (generation {gen_store.current_generation}, "
            f"{db_path}) is not even readable after the race: {exc!r}. This is the concurrent "
            f"rebuild-vs-incremental defect described in this test's docstring: the rebuild's "
            f"_cleanup_old_generations() unlinked this generation's files while the incremental "
            f"run still had them open, and its final checkpoint() then pointed the manifest back "
            f"at this now-empty/nonexistent generation."
        )

    is_valid, reason = vectors.validate_alignment(db_path, vec_path)
    assert is_valid, f"index left unaligned by the concurrent rebuild: {reason}"

    assert len(rows) == expected_chunk_count, (
        f"expected {expected_chunk_count} chunks (baseline + new content, all present exactly "
        f"once) in the surviving generation, got {len(rows)} -- content from one or both "
        f"concurrent runs was lost"
    )

    texts = [t for _cid, t, _vr in rows]
    assert any("ZZNEWMARKER" in t for t in texts), (
        "the incremental run's new content is entirely absent from the final index -- lost to "
        "the concurrent rebuild's cleanup of the generation it was still writing to"
    )
    assert any(
        "BASE-base-0" in t for t in texts
    ), "the rebuild's own baseline content is absent from the final index"

    _assert_every_chunk_points_at_its_own_vector(db_path, vec_path)

    response = api.search(project_dir, "ZZNEWMARKER unique prompt")
    assert response.results, "search() cannot find the incrementally-added content after the race"
