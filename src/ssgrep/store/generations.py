"""Atomic manifest-based generational store for crash-safe multi-file commits."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from ssgrep import embed


class GenerationalStore:
    """Atomic manifest-based store for crash-safe multi-file commits.

    index.db and vectors.f32 are two separate files and cannot be committed
    as one atomic unit. Without help, a crash between writing one and the
    other leaves either an orphaned vector row (harmless: nothing points at
    it) or, far worse, a chunk row that references a vector row that was
    never durably written. This class closes that window two ways, matched
    to how expensive each is to run:

    - commit_generation(): a rare, full staged-generation swap (index.db.N,
      vectors.f32.N) behind an atomically-written manifest, for rebuilds.
      The staged generation is validated for internal consistency BEFORE
      the manifest is ever written -- a crash, or a caller bug, that leaves
      the staged generation broken is refused rather than swapped in, which
      is what protects the live generation being replaced: cleanup of that
      live generation's files only ever runs after the swap to the new one
      is confirmed both valid and durable. (Previously this deleted the old
      generation unconditionally, which meant an incomplete or invalid
      staged write would destroy the still-good generation it was supposed
      to replace, with nothing left to recover from.)
    - checkpoint() / recover(): a cheap alternative for routine appends,
      which are far too frequent to pay a full-copy cost for (D12: ~430MB
      of index per year). checkpoint() durably journals how many vector
      rows were valid as of the last known-good SQLite commit. recover() --
      an explicit step a caller takes when opening the store to index,
      never run implicitly by __init__ or by a read-only caller such as
      search's index-path resolution, which is documented (D13) to detect
      and refuse rather than repair -- cross-checks that journal against
      what the chunks table actually references (never trusting the
      journal alone to certify correctness, only to skip work when it
      already agrees with the file on disk) and reconciles both
      directions: unreferenced trailing vector rows are truncated away,
      and any chunk that durably references a vector row which was never
      durably written is dropped, since the vector bytes behind it cannot
      be reconstructed.

    Manifest writes (both paths) go to a temp file in index_dir, are
    fsynced, then os.replace()'d into place, with the containing directory
    fsynced too -- so a crash mid-write can only ever leave the previous
    manifest (fully written, or absent) behind, never a torn one.

    Concurrent rebuild vs. incremental run. The two mechanisms above assume
    only ONE caller is ever touching a given generation's live files at a
    time. That is false: a rebuild (commit_generation(), building a brand
    new generation) can run concurrently with an ordinary incremental run
    that opened the OLD generation's files before the rebuild committed and
    is still writing to them when _cleanup_old_generations() would
    otherwise unlink those exact files out from under it -- corrupting or
    destroying both the incremental run's new writes and the untouched
    baseline data that was already durably there. hold_generation() and
    _cleanup_old_generations()'s cooperating flock() calls close that
    window (see both docstrings). That alone still leaves a subtler
    problem: the incremental run's own GenerationalStore instance was
    constructed before the rebuild committed, so its cached
    current_generation is stale by the time it finishes and calls
    checkpoint() -- checkpoint() re-reads the on-disk manifest and refuses
    to write when it disagrees, rather than silently clobbering the
    manifest back to the now-superseded generation and discarding the
    rebuild's completed work. See tests/test_concurrent_index.py for the
    full empirical trace this closes.
    """

    # dimension * sizeof(float32); the single source of truth for the
    # dimension is embed.DIMENSION (embed.py is a pure leaf module).
    VECTOR_ROW_BYTES = embed.DIMENSION * 4

    def __init__(self, index_dir: Path):
        """Initialize the generational store at the given directory.

        Read-only: only reads an existing manifest if present, and never
        creates the index directory or any file. Search's index-path
        resolution relies on exactly this to stay side-effect free on a
        read-only query path.
        """
        self.index_dir = index_dir
        self.manifest_path = index_dir / ".manifest"
        self.current_generation, self.committed_vector_rows = self._load_manifest()

    def _load_manifest(self) -> tuple[int, int]:
        """Load (generation, committed_vector_row_count), or (0, 0) if the
        manifest is missing or unreadable.

        (0, 0) is the right READ-side default -- an uncommitted staged
        generation must never be mistaken for the live one -- but it is not
        safe to make a DESTRUCTIVE decision on. See
        _refuse_if_not_newer_than_live(), which asks _manifest_is_readable()
        instead of trusting this.
        """
        readable, generation, rows = self._read_manifest()
        return (generation, rows) if readable else (0, 0)

    def _read_manifest(self) -> tuple[bool, int, int]:
        """(readable, generation, vector_row_count) straight from disk.

        ``readable`` is False both when the manifest is absent and when it is
        present but unparseable, because the two are indistinguishable to
        every caller that matters and both mean "the live generation is
        unknown".
        """
        data: object = None
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                data = None
        if isinstance(data, dict) and isinstance(data.get("generation"), int):
            return True, data["generation"], data.get("vector_row_count", 0)
        return False, 0, 0

    def _manifest_is_readable(self) -> bool:
        """True when the manifest can be read and believed right now."""
        return self._read_manifest()[0]

    def _generation_files_exist(self, gen: int) -> bool:
        """True if either of generation `gen`'s two files is on disk."""
        return self.get_index_path(gen).exists() or self.get_vector_path(gen).exists()

    def get_index_path(self, gen: int | None = None) -> Path:
        """Get the path for the index.db at a given generation."""
        if gen is None:
            gen = self.current_generation
        if gen == 0:
            return self.index_dir / "index.db"
        return self.index_dir / f"index.db.{gen}"

    def get_vector_path(self, gen: int | None = None) -> Path:
        """Get the path for the vectors.f32 at a given generation."""
        if gen is None:
            gen = self.current_generation
        if gen == 0:
            return self.index_dir / "vectors.f32"
        return self.index_dir / f"vectors.f32.{gen}"

    def stage_generation(self, next_gen: int) -> tuple[Path, Path]:
        """Return paths for staging the next generation, cleared of any
        residue left by an earlier, abandoned attempt at that same generation.

        These files should be written during indexing. Only after both are
        complete should commit_generation() be called to atomically swap.

        Staging is idempotent, and has to be. A rebuild that dies partway --
        Ctrl-C, OOM, an exception mid-embed -- leaves index.db.N, SQLite's
        -wal/-shm sidecars, and vectors.f32.N behind. Nothing points at them
        (the manifest was never written, so generation N was never current),
        but the next rebuild computes the same next_gen and, before this
        cleanup existed, opened those exact files and kept appending. The
        staged vectors.f32 then held the abandoned attempt's rows AND the new
        one's, while the staged chunks table referenced only the new one's --
        leaving the abandoned rows orphaned, which validate_generations()
        rejects. commit_generation() therefore raised "refusing to commit
        generation N: ... not mutually consistent", and so did every rebuild
        after it, permanently: the only escape was deleting .ssgrep/ by hand,
        which nothing told the user to do while ssgrep's own error text
        advised re-running the command that kept failing. Unlinking first is
        what makes a rebuild what indexer.py's module docstring already claims
        it is -- end-state-equivalent to deleting .ssgrep/ and starting over.

        The live generation is never touched -- see discard_generation() for
        how that is enforced against a manifest another process may have moved
        on since this store was constructed. next_gen == current_generation
        happens only on a fresh, never-committed store (both 0), where there
        is nothing to unlink anyway.
        """
        if next_gen > self.current_generation:
            self.discard_generation(next_gen)
        return (self.get_index_path(next_gen), self.get_vector_path(next_gen))

    def discard_generation(self, gen: int) -> None:
        """Unlink generation `gen`'s artifacts, SQLite sidecars included.

        Used to clear residue before staging, and to drop a fully-built staged
        generation whose commit a caller has decided to refuse (see
        rebuild_guard.check_shrink()).

        This is the only destructive operation outside commit_generation(), so
        it runs the same two-part protocol the rest of this module is built on:

        1. A non-blocking EXCLUSIVE flock on generation `gen`'s own lock file,
           which commit_generation() takes for the whole of its manifest write.
           Whichever of the two gets the lock first wins outright: a commit
           that lands first makes the check below refuse, and a discard that
           lands first makes that commit's validate_generations() gate refuse.
           Neither can interleave into "manifest points at gen N, gen N's files
           are gone".
        2. The refusal itself is decided against the manifest RE-READ FROM DISK
           under that lock, never against the self.current_generation cached by
           __init__. indexer.index() constructs its store at the top of a run
           and reaches staging much later; if another run committed generation
           N in between, the cached value is stale and `gen > cached` is still
           true for the generation the manifest now points at -- so trusting it
           would unlink the live index. checkpoint() re-reads the manifest for
           exactly this class of staleness.

        Generations at or below the live one are either the live index itself
        or an older one a concurrent reader may still hold open under
        hold_generation(); unlinking either is precisely the data loss the
        generational scheme exists to prevent, and reclaiming superseded
        generations is _cleanup_old_generations()'s job, which takes the locks
        that make it safe.
        """
        lock_fd = self._open_generation_lock_fd(gen)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError(
                    f"refusing to discard generation {gen}: another run holds it open, "
                    "so it may be committing it right now"
                ) from error
            try:
                self._refuse_if_not_newer_than_live(gen)
                index_path = self.get_index_path(gen)
                for path in (
                    index_path,
                    index_path.with_name(index_path.name + "-wal"),
                    index_path.with_name(index_path.name + "-shm"),
                    index_path.with_name(index_path.name + "-journal"),
                    self.get_vector_path(gen),
                ):
                    path.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def _refuse_if_not_newer_than_live(self, gen: int) -> None:
        """Raise unless `gen` is provably NOT the live generation.

        Two refusals, because "which generation is live" has two failure
        modes and only one of them is a stale number.

        1. `gen` is not strictly newer than every generation that might still
           be live: the manifest on disk right now AND this store's cached
           one, whichever is higher.

        2. The manifest cannot be read at all, and `gen`'s files exist. Then
           nothing on disk establishes which generation is live, and unlinking
           is a guess. The read-side default of generation 0 (see
           _load_manifest) is deliberately fail-SAFE for reading and
           catastrophic for deleting, and this is the code that must not
           inherit it.

           That inheritance was a silent total-loss bug. `.ssgrep/.manifest`
           is a ~100-byte JSON file beside a multi-hundred-megabyte index; a
           backup or sync that skips dotfiles, a zero-length restore, a user
           clearing what looks like a stale state file, or a transient read
           error on a network home loses it while every index.db.N and
           vectors.f32.N stays perfectly intact. A store whose live generation
           was N >= 1 then reported 0, so stage_generation(1) computed
           `1 > 0`, this check re-read the SAME fail-open value and agreed,
           and the buyer's live index.db.1 / vectors.f32.1 were unlinked
           before a single transcript had been read. rebuild_guard was blinded
           by the identical read -- it measures the retiring generation at
           get_index_path(), which had become the nonexistent index.db -- so
           the shrink gate saw (0, 0, 0) and waved the empty rebuild through
           at exit 0, printing "Indexed 0 sessions". One unreadable dotfile,
           and the entire index was gone with nothing left to recover from.

           A readable manifest is what makes residue provably residue: `gen`
           above the committed generation is an abandoned attempt and is
           cleared as before. Without one there is no proof either way, and
           for a tool whose product IS the buyer's indexed history, refusing
           loudly with the files intact is the only acceptable side to err on.
        """
        on_disk_generation, _ = self._load_manifest()
        live = max(on_disk_generation, self.current_generation)
        if gen <= live:
            raise ValueError(
                f"refusing to discard generation {gen}: it is not newer than the "
                f"live generation ({live})"
            )
        if not self._manifest_is_readable() and self._generation_files_exist(gen):
            raise ValueError(
                f"refusing to discard generation {gen}: {self.manifest_path} is missing "
                "or unreadable, so which generation is live cannot be established, and "
                f"generation {gen}'s files are present. Nothing has been deleted. "
                "Restore the manifest from a backup, or recreate it as "
                '{"generation": N, "vector_row_count": 0} naming the newest '
                "index.db.N you want to keep."
            )

    def commit_generation(self, next_gen: int, vector_row_count: int | None = None) -> None:
        """Atomically commit the staged generation by updating the manifest.

        Refuses (raises ValueError) rather than swap to a generation whose
        files are not mutually consistent -- see the class docstring for
        why that gate has to run before the manifest is touched.
        vector_row_count defaults to the staged vectors.f32's actual row
        count (from its file size) when not given explicitly.

        Runs under a blocking EXCLUSIVE flock on generation next_gen's own
        lock file, held across the validate-then-write-manifest sequence. That
        is the other half of discard_generation()'s protocol: without it, a
        concurrent run's discard could unlink next_gen's files in the window
        between validate_generations() passing and _write_manifest() landing,
        leaving the manifest pointing at a generation whose files are gone.
        Blocking rather than LOCK_NB because two rebuilds racing on the same
        generation must serialize, not fail -- the loser's validate gate then
        refuses on its own merits, which is the intended, non-destructive
        outcome. Nothing ever holds this lock and then waits on the committer,
        so it cannot deadlock: indexer.index() releases its hold_generation()
        guard on the OLD generation before staging a new one.
        """
        lock_fd = self._open_generation_lock_fd(next_gen)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                self._commit_generation_locked(next_gen, vector_row_count)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def _commit_generation_locked(self, next_gen: int, vector_row_count: int | None) -> None:
        """commit_generation()'s body, run while holding next_gen's lock."""
        staged_index = self.get_index_path(next_gen)
        staged_vec = self.get_vector_path(next_gen)

        if vector_row_count is None:
            vector_row_count = self._live_vector_row_count(staged_vec)

        if not self.validate_generations(staged_index, staged_vec):
            raise ValueError(
                f"refusing to commit generation {next_gen}: {staged_index} and "
                f"{staged_vec} are not mutually consistent (some chunk's vec_row "
                "falls outside the rows vectors.f32 actually holds)"
            )

        self._write_manifest(next_gen, vector_row_count)
        self.current_generation = next_gen
        self.committed_vector_rows = vector_row_count
        self._cleanup_old_generations()

    def checkpoint(self, vector_row_count: int) -> None:
        """Durably journal that vector_row_count rows of the CURRENT
        generation's vectors.f32 are known-good: as of the SQLite commit
        that made this true, every chunk referencing a vec_row below this
        count is durable and every one of those rows is durably written.
        Does not change current_generation and deletes nothing -- this is
        the cheap per-append alternative to commit_generation() that D12
        requires instead of copying the whole store on every append.

        Refuses (silently -- this is a normal, expected outcome of losing
        a race, not an error) to write when the on-disk manifest's
        generation no longer matches self.current_generation. That
        mismatch means some OTHER commit_generation() call -- e.g. a
        rebuild racing this run -- has already swapped the manifest to a
        newer generation since this GenerationalStore was constructed.
        This run's own data is safe regardless (hold_generation() is what
        guarantees the files it was writing to were never deleted out from
        under it), but its watermark for a generation nothing points at
        anymore is meaningless, and writing it would silently revert the
        manifest to that stale, superseded generation, discarding whatever
        the concurrent commit_generation() call just committed.
        """
        on_disk_generation, _ = self._load_manifest()
        if on_disk_generation != self.current_generation:
            return
        self._write_manifest(self.current_generation, vector_row_count)
        self.committed_vector_rows = vector_row_count

    def recover(self) -> tuple[int, int]:
        """Reconcile the current generation's files. Call this once,
        explicitly, when opening the store to index (see the class
        docstring for why this is not run from __init__).

        Returns (truncated_vector_rows, dropped_chunk_count).

        Two independent crash windows are closed, matching either order a
        two-file commit could be attempted in:

        - vectors were appended but the SQLite commit that would reference
          them either never landed or landed without a matching
          checkpoint() call yet: those trailing rows are unreferenced by
          any chunk actually in the table (checked directly, never assumed
          from the journal alone) and are truncated off vectors.f32.
        - a chunk durably references a vec_row that vectors.f32 does not
          actually have (the far worse case: the SQLite commit landed, the
          vector bytes behind it did not). Nothing can reconstruct those
          bytes, so the only consistent outcome is to drop that chunk (and
          its FTS shadow) rather than leave a reference a search could
          return and then have nothing to compare against.
        """
        db_path = self.get_index_path()
        vec_path = self.get_vector_path()
        if not db_path.exists():
            return 0, 0

        live_rows = self._live_vector_row_count(vec_path)
        if live_rows == self.committed_vector_rows:
            # Journal agrees with the file on disk: nothing has been
            # appended (or lost) since the last known-good checkpoint, so
            # the full chunks-table cross-check below would find nothing.
            # This is the common, un-crashed case, and it is what keeps
            # recover() cheap enough to run on every open (D12) -- the
            # journal is trusted only to skip work, never to certify
            # correctness on its own; any disagreement falls through to
            # the real cross-check below.
            return 0, 0

        conn = sqlite3.connect(str(db_path))
        try:
            dangling = [
                row[0]
                for row in conn.execute(
                    "SELECT chunk_id FROM chunks WHERE vec_row IS NOT NULL AND vec_row >= ?",
                    (live_rows,),
                ).fetchall()
            ]
            if dangling:
                placeholders = ",".join("?" for _ in dangling)
                conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", dangling)
                conn.execute(
                    f"DELETE FROM chunks_fts_tri WHERE chunk_id IN ({placeholders})", dangling
                )
                conn.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", dangling)

            max_row = conn.execute(
                "SELECT MAX(vec_row) FROM chunks WHERE vec_row IS NOT NULL"
            ).fetchone()[0]
            conn.commit()
        finally:
            conn.close()

        needed_rows = 0 if max_row is None else max_row + 1
        truncated = 0
        if live_rows > needed_rows:
            truncated = live_rows - needed_rows
            with open(vec_path, "r+b") as f:
                f.truncate(needed_rows * self.VECTOR_ROW_BYTES)
                f.flush()
                os.fsync(f.fileno())

        if dangling or truncated:
            self.checkpoint(needed_rows)

        return truncated, len(dangling)

    def _live_vector_row_count(self, vec_path: Path) -> int:
        if not vec_path.exists():
            return 0
        return vec_path.stat().st_size // self.VECTOR_ROW_BYTES

    def _write_manifest(self, generation: int, vector_row_count: int) -> None:
        """Write the manifest atomically: temp file in index_dir, fsynced,
        then os.replace()'d into place, with index_dir itself fsynced
        afterward so the rename survives a crash immediately after it. A
        crash during this method can therefore only ever leave the
        previous manifest (fully written, or absent) in place -- never a
        torn one.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.index_dir, 0o700)
        manifest = {
            "generation": generation,
            "vector_row_count": vector_row_count,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        fd, tmp_name = tempfile.mkstemp(dir=self.index_dir, prefix=".manifest.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(manifest))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.manifest_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        dir_fd = os.open(self.index_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _cleanup_old_generations(self) -> None:
        """Delete every generation strictly older than the current one --
        UNLESS some other caller is still actively using it.

        Only ever called after _write_manifest() has already landed the
        new generation as current (durably, via fsync + rename), so every
        file this deletes is always already-superseded: the manifest, not
        this loop, is what makes a generation "old". A crash partway
        through just leaves a stale file or two for a later commit to
        sweep (unlink(missing_ok=True) makes that idempotent); it can
        never delete the generation that is now current, because that
        generation's own number is never inside range(self.current_generation).

        "Already-superseded" is not the same as "unused": an ordinary
        incremental index() call can open an old generation's live files
        (via hold_generation(), a SHARED flock on that generation's lock
        file) before a concurrent rebuild commits and supersedes it, and
        can still be mid-write when this method runs. Deleting those files
        out from under an open sqlite3 connection / mmap'd vectors.f32
        does not raise -- it silently orphans every byte written through
        them from that point on (see tests/test_concurrent_index.py for
        the full mechanism). So before unlinking generation `gen`'s files,
        this takes a non-blocking EXCLUSIVE flock on that same lock file;
        if some other caller's hold_generation() is holding it (shared),
        the exclusive attempt fails immediately (LOCK_NB) rather than
        blocking a rebuild's commit on an unrelated run's progress, and
        this generation is left in place. Nothing is lost by skipping it:
        the loop re-scans every gen < current_generation, not just the
        newly superseded one, so a busy generation is simply picked up by
        whichever LATER commit_generation() call finds it free.
        """
        for gen in range(self.current_generation):
            lock_fd = self._open_generation_lock_fd(gen)
            try:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    continue  # in active use -- retried by a future commit
                try:
                    self.get_index_path(gen).unlink(missing_ok=True)
                    self.get_vector_path(gen).unlink(missing_ok=True)
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    def _generation_lock_path(self, gen: int) -> Path:
        """Path to generation `gen`'s advisory lock file.

        Deliberately never unlinked (by hold_generation() or by the
        cleanup above): unlinking a flock()'d path while another process
        still holds a lock on the open file descriptor is a classic race
        in itself -- a subsequent open() of the same path would create a
        NEW inode that the original holder's lock does nothing to
        exclude. Generation numbers only ever increase and are never
        reused, so leaving these tiny (0-byte) files behind forever costs
        nothing worth reclaiming, unlike index.db/vectors.f32 (D12).
        """
        return self.index_dir / f".gen.{gen}.lock"

    def _open_generation_lock_fd(self, gen: int) -> int:
        """Open (creating if needed) generation `gen`'s lock file, without
        acquiring any lock on it yet. Callers are responsible for
        flock()ing and for os.close()ing the returned fd.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.index_dir, 0o700)
        return os.open(self._generation_lock_path(gen), os.O_CREAT | os.O_RDWR, 0o600)

    @contextlib.contextmanager
    def hold_generation(self, gen: int | None = None) -> Iterator[None]:
        """Hold a SHARED advisory lock on generation `gen` (default:
        current_generation) for the duration of the with-block.

        Callers that are about to open and use a generation's live
        index.db/vectors.f32 -- i.e. an incremental index() run, between
        deciding it will operate on the CURRENT generation in place and
        finishing all reads/writes against those files -- should wrap
        that entire span in this. It signals to any concurrent
        _cleanup_old_generations() that this generation is in active use
        and must not be unlinked yet (see that method's docstring for the
        corruption this prevents). Multiple holders are expected and safe
        -- e.g. two concurrent plain index() calls both operating on the
        same live generation (test_two_concurrent_plain_index_calls_stay_
        consistent) -- since flock()'s shared mode allows any number of
        simultaneous holders; only cleanup's EXCLUSIVE attempt is ever
        blocked by this.

        Uses fcntl.flock, a POSIX advisory lock scoped to the OS (works
        across both threads and separate processes, which matters since
        ordinary CLI usage is one-process-per-invocation). CI runs macOS
        and Linux only; this module makes no attempt to support Windows.
        """
        if gen is None:
            gen = self.current_generation
        lock_fd = self._open_generation_lock_fd(gen)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def validate_generations(self, db_path: Path, vec_path: Path) -> bool:
        """Check that every chunk's vec_row is backed by an actual vector
        row: 0 <= vec_row < (live vector row count) AND every vector row in
        vectors.f32 is referenced by at least one chunk (no orphaned rows).
        Returns False if db_path can't be opened, if vec_path is missing
        outright while chunks reference rows, if any chunk's vec_row falls
        outside the rows vectors.f32 actually holds, or if there are orphaned
        vector rows. An empty chunks table is vacuously valid. This is
        commit_generation()'s pre-commit gate -- a generation that fails this
        can never be swapped in.
        """
        if not db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                chunk_rows = {
                    row[0]
                    for row in conn.execute(
                        "SELECT vec_row FROM chunks WHERE vec_row IS NOT NULL"
                    ).fetchall()
                }
            finally:
                conn.close()
        except sqlite3.Error:
            return False

        vec_row_count = self._live_vector_row_count(vec_path)

        # Check all chunk rows are within bounds
        if not all(0 <= row < vec_row_count for row in chunk_rows):
            return False

        # Check there are no orphaned vector rows (every row is referenced)
        referenced_rows = set(chunk_rows)
        all_valid_rows = set(range(vec_row_count))
        orphaned_rows = all_valid_rows - referenced_rows
        return len(orphaned_rows) == 0
