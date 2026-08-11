"""Direct tests for GenerationalStore's generation-scoped advisory locking:
hold_generation() and _cleanup_old_generations()'s cooperating fcntl.flock
calls, plus the small helpers that back them (_generation_lock_path,
_open_generation_lock_fd, _load_manifest, _live_vector_row_count).

Why this file exists. hold_generation()/_cleanup_old_generations() close a
real, previously-confirmed data-loss bug: a concurrent --rebuild's
commit_generation() used to unconditionally unlink a superseded
generation's index.db/vectors.f32 the instant it committed, with no
awareness that an ordinary incremental index() call might still have those
exact files open and be mid-write. Before this file, the ENTIRE safety net
for that fix was one threaded, real-race test in
tests/test_concurrent_index.py
(test_concurrent_rebuild_cannot_destroy_a_racing_incremental_run) -- which
proves the mechanism survives a genuine race, but never isolates the
locking primitives themselves: nothing called hold_generation() directly,
asserted that a held generation's files survive cleanup while an unheld
one is reclaimed, or checked that the lock file itself is never among the
files cleanup deletes (store.py's own docstring on
_generation_lock_path() calls out unlinking a flock'd path as "a classic
race in itself"). This file closes that gap with synchronous, single-
process tests: no threads are needed because fcntl.flock() locks are
scoped to the OPEN FILE DESCRIPTION, not the thread or process, so two
independent os.open() calls against the same path from the very same
thread contend exactly as two real processes would -- see each test's
docstring for why that is safe to rely on here.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

from ssgrep.store import GenerationalStore, init_db
from ssgrep.vectors import open_vectors

DIM = 256


def _make_empty_generation(gen_store: GenerationalStore, gen: int) -> None:
    """Create generation `gen`'s files with zero chunks and zero vectors.

    Vacuously valid per validate_generations()'s own docstring ("An empty
    chunks table is vacuously valid"), which is all these tests need: they
    exercise lock *coordination*, not data integrity -- that is already
    covered by tests/test_atomicity.py.
    """
    index_path, vec_path = gen_store.stage_generation(gen)
    init_db(index_path).close()
    open_vectors(vec_path, dimension=DIM)


# ---------------------------------------------------------------------------
# 1. hold_generation() and _cleanup_old_generations() must contend for the
#    SAME lock file, and a held generation must be genuinely, verifiably
#    locked -- not just locked "in spirit".
# ---------------------------------------------------------------------------


def test_hold_generation_and_cleanup_probe_share_the_same_lock_path(tmp_path: Path) -> None:
    """hold_generation()'s SHARED lock and _cleanup_old_generations()'s
    EXCLUSIVE lock only coordinate anything if they flock() the exact same
    file. Proven directly, using the identical mechanism cleanup itself
    uses (a fresh, independent fd opened via os.open() on
    _generation_lock_path(gen), then a non-blocking EXCLUSIVE flock): while
    hold_generation(0) is held open by this test, that probe must fail
    immediately (LOCK_NB) rather than block or silently succeed. Once
    hold_generation(0) is released, the identical probe must succeed.

    Non-blocking probes throughout (LOCK_NB) so a regression that makes
    this coordination a no-op fails fast with a clear assertion instead of
    hanging the suite.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)
    _make_empty_generation(gen_store, 0)

    with gen_store.hold_generation(0):
        lock_path = gen_store._generation_lock_path(0)
        assert lock_path.exists(), "hold_generation() must create the lock file it locks"

        probe_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with pytest.raises(OSError):
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe_fd)

    # Released: the identical probe against the identical path now succeeds.
    probe_fd = os.open(gen_store._generation_lock_path(0), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise
        fcntl.flock(probe_fd, fcntl.LOCK_UN)
    finally:
        os.close(probe_fd)


def test_hold_generation_allows_a_second_concurrent_shared_holder(tmp_path: Path) -> None:
    """hold_generation()'s docstring guarantees multiple simultaneous
    holders are expected and safe -- e.g. two concurrent plain index()
    calls both operating on the same live generation
    (test_two_concurrent_plain_index_calls_stay_consistent in
    test_concurrent_index.py exercises this at the full-index level; this
    isolates the locking primitive itself). Proven via a non-blocking
    SHARED probe rather than a second real hold_generation() call so a
    regression to an EXCLUSIVE lock fails with a clean assertion instead
    of deadlocking this thread against itself.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)
    _make_empty_generation(gen_store, 0)

    with gen_store.hold_generation(0):
        probe_fd = os.open(gen_store._generation_lock_path(0), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(probe_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)  # must not raise
            fcntl.flock(probe_fd, fcntl.LOCK_UN)
        finally:
            os.close(probe_fd)


# ---------------------------------------------------------------------------
# 2. _cleanup_old_generations() must SKIP a held generation (files survive)
#    and must RECLAIM an unheld, superseded one. Both halves in one flow,
#    via the real public API (commit_generation()/hold_generation()) --
#    the same entry points indexer.py actually calls.
# ---------------------------------------------------------------------------


def test_cleanup_skips_a_held_generation_but_reclaims_an_unheld_one(tmp_path: Path) -> None:
    """Generation 0 is held for this entire test (simulating a concurrent
    incremental index() call still reading it). While held:

    - committing generation 1 supersedes 0 -- 0's cleanup must be SKIPPED,
      leaving its files in place.
    - committing generation 2 (0 still held, but 1 is NOT) must SKIP 0
      again but RECLAIM 1. This is the two-old-generations case that
      actually distinguishes "skip" from "everything happens to survive
      because nothing old exists yet".

    After generation 0's hold is released, a later commit (generation 3)
    must finally reclaim it -- proving the earlier skips were deferral,
    never a permanent leak (cleanup re-scans every generation older than
    current on every call, per _cleanup_old_generations()'s docstring).
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)
    _make_empty_generation(gen_store, 0)

    with gen_store.hold_generation(0):
        _make_empty_generation(gen_store, 1)
        gen_store.commit_generation(1)

        assert gen_store.get_index_path(0).exists(), (
            "generation 0 is held via hold_generation() -- its files must survive "
            "commit_generation(1)'s cleanup pass"
        )
        assert gen_store.get_vector_path(0).exists()

        _make_empty_generation(gen_store, 2)
        gen_store.commit_generation(2)

        assert gen_store.get_index_path(
            0
        ).exists(), "generation 0 must still survive -- it is still held at this point"
        assert gen_store.get_vector_path(0).exists()
        assert not gen_store.get_index_path(1).exists(), (
            "generation 1 was never held by anything -- commit_generation(2)'s "
            "cleanup must have reclaimed it"
        )
        assert not gen_store.get_vector_path(1).exists()

    # Hold released. A later commit re-scans every generation < current and
    # picks up what an earlier commit had to skip.
    _make_empty_generation(gen_store, 3)
    gen_store.commit_generation(3)

    assert not gen_store.get_index_path(0).exists(), (
        "once nothing holds generation 0, a later commit must reclaim it -- the "
        "earlier skip was deferral, not a permanent leak"
    )
    assert not gen_store.get_vector_path(0).exists()


# ---------------------------------------------------------------------------
# 3. The generation lock file itself must never be among the files
#    _cleanup_old_generations() unlinks.
# ---------------------------------------------------------------------------


def test_cleanup_never_unlinks_the_generation_lock_file(tmp_path: Path) -> None:
    """_generation_lock_path()'s own docstring: unlinking a flock'd path
    while another opener still holds a lock on the original inode is
    "a classic race in itself" -- a later open() of the same path would
    create a NEW inode whose lock excludes nobody the original holder
    thought it excluded. This proves _cleanup_old_generations() -- which
    unlinks generation 0's index.db and vectors.f32 once nothing holds its
    lock -- leaves the lock file itself untouched.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)
    _make_empty_generation(gen_store, 0)

    lock_path_0 = gen_store._generation_lock_path(0)
    with gen_store.hold_generation(0):
        pass  # momentary hold, just to bring the lock file into existence
    assert lock_path_0.exists(), "sanity: hold_generation() must have created it"

    _make_empty_generation(gen_store, 1)
    gen_store.commit_generation(1)  # nothing holds gen 0 now -- its data files are reclaimed

    assert not gen_store.get_index_path(0).exists(), "sanity: gen 0's data files were reclaimed"
    assert not gen_store.get_vector_path(0).exists()
    assert lock_path_0.exists(), (
        "the generation's advisory lock file must never be unlinked by cleanup -- "
        "doing so would break lock identity for any other still-open holder, per "
        "_generation_lock_path()'s own docstring"
    )


# ---------------------------------------------------------------------------
# 4. Small helpers backing the above, each with zero prior direct coverage.
# ---------------------------------------------------------------------------


def test_generation_lock_path_is_deterministic_and_unique_per_generation(tmp_path: Path) -> None:
    """hold_generation() and _cleanup_old_generations() each open their OWN
    fd independently and must still land on the same file, so this must be
    a pure function of (index_dir, gen): stable across repeated calls and
    across fresh GenerationalStore instances pointed at the same
    index_dir, and never colliding across different generations (a
    collision would make holding generation 0 block cleanup of some
    unrelated generation N).
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)

    assert gen_store._generation_lock_path(0) == gen_store._generation_lock_path(0)
    assert gen_store._generation_lock_path(0) == GenerationalStore(index_dir)._generation_lock_path(
        0
    )

    paths = {gen_store._generation_lock_path(g) for g in range(5)}
    assert len(paths) == 5, "every generation must map to a distinct lock file"
    for p in paths:
        assert p.parent == index_dir


def test_open_generation_lock_fd_creates_index_dir_and_a_usable_lock_file(
    tmp_path: Path,
) -> None:
    """_open_generation_lock_fd() may be the very first thing to touch
    index_dir (e.g. hold_generation() called on a store whose __init__ is
    documented to never create index_dir). It must create that directory
    on demand, leave a real 0-byte file at exactly _generation_lock_path(gen),
    apply the same 0o700 posture as the rest of this module (the directory
    may hold data revealing session content), and hand back an fd that is
    actually flock()-able.
    """
    index_dir = tmp_path / "fresh" / ".ssgrep"
    gen_store = GenerationalStore(index_dir)
    assert not index_dir.exists(), "sanity: GenerationalStore.__init__ must not create it"

    lock_path = gen_store._generation_lock_path(7)
    fd = gen_store._open_generation_lock_fd(7)
    try:
        assert index_dir.is_dir()
        assert (index_dir.stat().st_mode & 0o777) == 0o700
        assert lock_path.exists()
        assert lock_path.stat().st_size == 0

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the fd is genuinely lockable
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_load_manifest_falls_back_to_zero_zero_on_missing_or_corrupt_manifest(
    tmp_path: Path,
) -> None:
    """__init__'s only source for current_generation/committed_vector_rows.
    Must degrade to (0, 0) -- fresh store, generation 0 -- rather than
    raise, both when no manifest was ever written and when one exists but
    fails to parse as JSON (a foreign or hand-edited file; a genuine crash
    mid-write can't produce this given _write_manifest()'s
    temp-file-then-rename protocol, but this method has no way to tell the
    two apart and must not special-case it). A real, valid manifest is
    also checked, as the positive companion to the two fallback cases.
    """
    index_dir = tmp_path / ".ssgrep"

    assert GenerationalStore(index_dir)._load_manifest() == (0, 0), "no manifest written yet"

    real_store = GenerationalStore(index_dir)
    real_store.checkpoint(3)
    assert GenerationalStore(index_dir)._load_manifest() == (
        0,
        3,
    ), "a real, validly-written manifest must round-trip"

    (index_dir / ".manifest").write_text("{not valid json")
    assert GenerationalStore(index_dir)._load_manifest() == (
        0,
        0,
    ), "a corrupt manifest must degrade to (0, 0), not raise"


def test_live_vector_row_count_reads_file_size_and_zero_when_absent(tmp_path: Path) -> None:
    """The single source of truth commit_generation() and
    validate_generations() both rely on for "how many vector rows does
    this file actually, physically hold" -- derived from the file's real
    size on disk (never from the manifest/journal, which is exactly the
    value recover()/checkpoint() cross-check THIS against), and 0 for a
    path that does not exist yet rather than raising. Also checks floor
    division on a partial trailing row: a half-written last row must never
    be counted as a whole one.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)
    missing = index_dir / "vectors.f32"

    assert gen_store._live_vector_row_count(missing) == 0

    index_dir.mkdir(parents=True)
    vec_path = index_dir / "vectors.f32"

    vec_path.write_bytes(b"\x00" * (3 * GenerationalStore.VECTOR_ROW_BYTES))
    assert gen_store._live_vector_row_count(vec_path) == 3

    vec_path.write_bytes(b"\x00" * (3 * GenerationalStore.VECTOR_ROW_BYTES + 17))
    assert (
        gen_store._live_vector_row_count(vec_path) == 3
    ), "17 trailing bytes are not a full row and must not round up to 4"
