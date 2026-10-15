"""Per-process journal segments with collision-resistant execution identities.

Each process appends only to its own exclusive, rotating segment
(``accounting/<profile>/journal/seg-v2-<nonce>-<rotation>.jsonl``). A per-segment
nonblocking lock coordinates compaction, never a global append lock. ``execution_id``
is ``<process nonce>-<seq>``:
the nonce is random per process (re-drawn in a forked child, which also drops
the inherited descriptors), the sequence is monotonic under a process-wide
lock, so two live executions can never share an identity. One execution yields
up to three immutable events (``started``, ``completed``, ``rendered``),
identified by ``(execution_id, phase)``.

Append is one pre-serialized ``write(2)`` of a bounded record, no ``fsync``.
If a write is cut short (ENOSPC, EFBIG, signal), the tail is a fragment with
no terminator: the next append begins with a newline so the fragment can
never fuse with the following record, and the reader never admits it. Power
loss before the OS flushes the page cache can still lose events; coverage then
stays unknown rather than claimed complete.

Every append revalidates the owned 0600 descriptor and its original name in
the current private namespace, using read-only no-follow directory traversal.
Two segment candidates bound rotation under compactor contention/reclamation;
a dropped event raises the storage error without waiting on the collector.
Configured-path or root-inode changes close old descriptors and start a fresh
nonce/sequence. Retired managed identities cannot carry in-flight phases into
a new root; unsafe-profile latches are keyed by their original root inode.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ssgrep.accounting import datadir, storage
from ssgrep.accounting.contracts import ExecutionEvent
from ssgrep.accounting.journal_codec import serialize_event

SEGMENT_PREFIX = "seg-"
SEGMENT_SUFFIX = ".jsonl"
_NONCE_BYTES = 8
SEGMENT_TARGET_BYTES = 8 * 1024 * 1024


@dataclass
class _Segment:
    fd: int
    day: int
    name: str
    directory_identity: tuple[int, int]
    size: int = 0
    dirty: bool = False


class _State:
    """Process-local identity/descriptors, bound lazily to one configured data root."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.nonce = os.urandom(_NONCE_BYTES).hex()
        self.seq = 0
        self.segment_seq = 0
        self.lock = threading.Lock()
        self.segments: dict[str, _Segment] = {}
        self.unsafe: set[str] = set()
        self.root: Path | None = None
        self.authority: datadir.SafeRootSnapshot | None = None
        self.unsafe_roots: dict[datadir.SafeRootSnapshot, set[str]] = {}
        self.retired_nonces: set[str] = set()


_state = _State()


def _reinit_after_fork() -> None:
    """Child of a fork: fresh nonce/sequence/lock; never share the parent's segment."""
    global _state
    previous = _state
    for segment in _state.segments.values():
        with contextlib.suppress(OSError):
            os.close(segment.fd)
    _state = _State()
    _state.retired_nonces = previous.retired_nonces | {previous.nonce}
    _state.unsafe_roots = {root: set(profiles) for root, profiles in previous.unsafe_roots.items()}


os.register_at_fork(after_in_child=_reinit_after_fork)


def _process_state() -> _State:
    """Fork fallback without reading or mutating any configured namespace."""
    if _state.pid != os.getpid():
        _reinit_after_fork()
    return _state


def _current() -> _State:
    """Expose state only after a noncreating observation of the configured root."""
    state = _process_state()
    with state.lock:
        _bind_root(state)
    return state


def _bind_root(
    state: _State,
    *,
    root: datadir.SafeRootSnapshot | None = None,
    path: Path | None = None,
) -> None:
    """Bind metadata under the state lock; namespace observation never creates."""
    if path is None:
        path = storage.data_root()
        root = datadir.snapshot(path)
    if state.root == path and state.authority == root:
        return
    # An initially absent root becoming real keeps its already-issued nonce.
    # Every move away from an existing namespace or configured path retires it.
    if state.root is not None and (state.root != path or state.authority is not None):
        for segment in state.segments.values():
            with contextlib.suppress(OSError):
                os.close(segment.fd)
        state.segments.clear()
        if state.authority is not None:
            state.unsafe_roots[state.authority] = state.unsafe
        state.retired_nonces.add(state.nonce)
        state.nonce = os.urandom(_NONCE_BYTES).hex()
        state.seq = 0
        state.segment_seq = 0
    state.root = path
    state.authority = root
    state.unsafe = state.unsafe_roots.setdefault(root, set()) if root is not None else set()


def _require_root(state: _State) -> None:
    """Configuration/namespace changes are transient, never foreign-root latches."""
    if state.authority is None:
        raise datadir.RootChanged(errno.ESTALE, "journal root is absent")
    datadir.require_current(state.authority)


def next_execution_id(*, root: datadir.SafeRootSnapshot | None = None) -> str:
    """A unique identity; an explicit capture root cannot inherit a foreign nonce."""
    state = _process_state()
    with state.lock:
        _bind_root(state)
        if root is not None:
            if state.authority != root:
                raise datadir.RootChanged(errno.ESTALE, "execution root changed before identity")
            datadir.require_current(root)
        state.seq += 1
        return f"{state.nonce}-{state.seq}"


def segment_name(nonce: str, rotation: int = 0) -> str:
    return f"{SEGMENT_PREFIX}v2-{nonce}-{rotation}{SEGMENT_SUFFIX}"


def _segment(
    state: _State,
    profile: str,
    day: int,
    previous: _Segment | None = None,
) -> _Segment:
    segment = state.segments.get(profile)
    if segment is not None:
        return segment
    root = state.authority
    if root is None:
        raise datadir.RootChanged(errno.ESTALE, "journal root is absent")
    directory = (
        _current_directory(profile, previous, root)
        if previous is not None
        else storage.open_profile_dir(profile, "journal", root=root)
    )
    try:
        info = os.fstat(directory)
        directory_identity = (info.st_dev, info.st_ino)
        for _ in range(2):
            state.segment_seq += 1
            name = segment_name(state.nonce, state.segment_seq)
            try:
                fd = storage.create_exclusive(directory, name)
                break
            except FileExistsError:
                continue  # never reuse a closed file, even after orderly shutdown
        else:
            raise FileExistsError(errno.EEXIST, "journal segment allocation collided twice")
    finally:
        os.close(directory)
    segment = state.segments[profile] = _Segment(fd, day, name, directory_identity)
    return segment


def _rotate(state: _State, profile: str) -> None:
    segment = state.segments.pop(profile, None)
    if segment is not None:
        os.close(segment.fd)


def _current_directory(
    profile: str,
    segment: _Segment,
    root: datadir.SafeRootSnapshot,
) -> int:
    """Read-only authority check; never repair or recreate a warm namespace."""
    try:
        directory = storage.open_profile_dir(profile, "journal", create=False, root=root)
    except FileNotFoundError as error:
        raise storage.UnsafeStorageError(
            error.errno, "active journal namespace disappeared"
        ) from error
    try:
        info = os.fstat(directory)
        if (info.st_dev, info.st_ino) != segment.directory_identity:
            raise storage.UnsafeStorageError(errno.ESTALE, "active journal namespace changed")
    except BaseException:
        os.close(directory)
        raise
    return directory


def _validate_segment(segment: _Segment, directory: int) -> bool:
    """False only for a cooperatively reclaimed inode; other changes are unsafe."""
    info = os.fstat(segment.fd)
    if not stat.S_ISREG(info.st_mode):
        raise storage.UnsafeStorageError(errno.EINVAL, "active journal is not a regular file")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise storage.UnsafeStorageError(
            errno.EPERM, "active journal is not owned and private 0600"
        )
    if info.st_nlink not in (0, 1):
        raise storage.UnsafeStorageError(errno.EPERM, "active journal has external links")
    try:
        current = os.stat(segment.name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError as error:
        if info.st_nlink == 0:
            return False
        raise storage.UnsafeStorageError(
            error.errno, "active journal was moved outside its original name"
        ) from error
    if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
        raise storage.UnsafeStorageError(errno.ESTALE, "active journal name was replaced")
    if (
        current.st_uid != os.geteuid()
        or stat.S_IMODE(current.st_mode) != 0o600
        or current.st_nlink != 1
    ):
        raise storage.UnsafeStorageError(errno.EPERM, "active journal authority changed")
    return True


def _write(segment: _Segment, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    try:
        while view:
            count = os.write(segment.fd, view)
            if count <= 0:
                raise OSError(errno.EIO, "journal write made no progress")
            written += count
            segment.size += count
            view = view[count:]
    except BaseException:
        if written:
            segment.dirty = True  # an unterminated fragment now sits at the tail
        raise


def append(
    profile: str,
    event: ExecutionEvent,
    *,
    root: datadir.SafeRootSnapshot | None = None,
) -> bool:
    """Append one event to this process's segment; raises ``OSError`` on failure.

    ``UnsafeStorageError`` marks the profile unsafe for the rest of the process
    in its data root; other failures are retried next event. Root-switched
    in-flight managed execution identities return False without writing.
    """
    data = serialize_event(event)
    state = _process_state()
    with state.lock:
        _bind_root(state)
        if event.execution_id.partition("-")[0] in state.retired_nonces:
            return False
        if root is not None and state.authority != root:
            return False
        if state.authority is None:
            assert state.root is not None
            created = datadir.snapshot(state.root, create=True)
            if created is None:
                raise datadir.RootChanged(errno.ESTALE, "journal root disappeared")
            _bind_root(state, root=created, path=created.path)
        if profile in state.unsafe:
            return False
        day = int(time.time() // 86400)
        try:
            last_error: OSError | None = None
            previous: _Segment | None = None
            for _ in range(2):
                segment = _segment(state, profile, day, previous)
                previous = segment
                assert state.authority is not None
                directory = _current_directory(profile, segment, state.authority)
                try:
                    try:
                        fcntl.flock(segment.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        last_error = error
                        _rotate(state, profile)  # collector busy: fresh file, never wait
                        continue
                    try:
                        if not _validate_segment(segment, directory):
                            last_error = FileNotFoundError(
                                errno.ENOENT, "journal segment was reclaimed"
                            )
                            _rotate(state, profile)
                            continue
                        if segment.day != day or segment.size >= SEGMENT_TARGET_BYTES:
                            _rotate(state, profile)
                            continue
                        record = b"\n" + data if segment.dirty else data
                        _require_root(state)
                        _write(segment, record)
                        segment.dirty = False
                        break
                    finally:
                        if state.segments.get(profile) is segment:
                            fcntl.flock(segment.fd, fcntl.LOCK_UN)
                finally:
                    os.close(directory)
            else:
                if last_error is not None:
                    raise last_error
                raise BlockingIOError(errno.EAGAIN, "journal rotation did not settle")
        except storage.UnsafeStorageError:
            _require_root(state)
            state.unsafe.add(profile)
            raise
    return True


def close_all() -> None:
    """Release this process's segments (tests and orderly shutdown)."""
    if _state.pid != os.getpid():
        _reinit_after_fork()
    state = _state
    with state.lock:
        for segment in state.segments.values():
            with contextlib.suppress(OSError):
                os.close(segment.fd)
        state.segments.clear()
        state.unsafe.clear()
        state.unsafe_roots.clear()
