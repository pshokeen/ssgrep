"""Per-process journal segments with collision-resistant execution identities.

Each process appends only to its own segment
(``accounting/<profile>/journal/seg-<nonce>.jsonl``), so concurrent processes
never contend on an append lock. ``execution_id`` is ``<process nonce>-<seq>``:
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
"""

from __future__ import annotations

import contextlib
import errno
import os
import threading
from dataclasses import dataclass

from ssgrep.accounting import storage
from ssgrep.accounting.contracts import ExecutionEvent
from ssgrep.accounting.journal_codec import serialize_event

SEGMENT_PREFIX = "seg-"
SEGMENT_SUFFIX = ".jsonl"
_NONCE_BYTES = 8


@dataclass
class _Segment:
    fd: int
    dirty: bool = False


class _State:
    """Process-local identity and descriptors; replaced wholesale after fork."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.nonce = os.urandom(_NONCE_BYTES).hex()
        self.seq = 0
        self.lock = threading.Lock()
        self.segments: dict[str, _Segment] = {}
        self.unsafe: set[str] = set()


_state = _State()


def _reinit_after_fork() -> None:
    """Child of a fork: fresh nonce/sequence/lock; never share the parent's segment."""
    global _state
    for segment in _state.segments.values():
        with contextlib.suppress(OSError):
            os.close(segment.fd)
    _state = _State()


os.register_at_fork(after_in_child=_reinit_after_fork)


def _current() -> _State:
    """The state valid for this process (covers fork paths that skip the hook)."""
    if _state.pid != os.getpid():
        _reinit_after_fork()
    return _state


def next_execution_id() -> str:
    """A new identity unique across processes, threads, and forks."""
    state = _current()
    with state.lock:
        state.seq += 1
        return f"{state.nonce}-{state.seq}"


def segment_name(nonce: str) -> str:
    return f"{SEGMENT_PREFIX}{nonce}{SEGMENT_SUFFIX}"


def _segment(state: _State, profile: str) -> _Segment:
    segment = state.segments.get(profile)
    if segment is not None:
        return segment
    directory = storage.open_profile_dir(profile, "journal")
    try:
        try:
            fd = storage.create_exclusive(directory, segment_name(state.nonce))
        except FileExistsError:  # a nonce collision: never share another process's file
            state.nonce = os.urandom(_NONCE_BYTES).hex()
            fd = storage.create_exclusive(directory, segment_name(state.nonce))
    finally:
        os.close(directory)
    segment = state.segments[profile] = _Segment(fd)
    return segment


def _write(segment: _Segment, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    try:
        while view:
            count = os.write(segment.fd, view)
            if count <= 0:
                raise OSError(errno.EIO, "journal write made no progress")
            written += count
            view = view[count:]
    except BaseException:
        if written:
            segment.dirty = True  # an unterminated fragment now sits at the tail
        raise


def append(profile: str, event: ExecutionEvent) -> bool:
    """Append one event to this process's segment; raises ``OSError`` on failure.

    ``UnsafeStorageError`` marks the profile unsafe for the rest of the process
    (capture is then disabled for it); other failures are retried next event.
    """
    data = serialize_event(event)
    state = _current()
    with state.lock:
        if profile in state.unsafe:
            return False
        try:
            segment = _segment(state, profile)
        except storage.UnsafeStorageError:
            state.unsafe.add(profile)
            raise
        record = b"\n" + data if segment.dirty else data
        _write(segment, record)
        segment.dirty = False
    return True


def close_all() -> None:
    """Release this process's segments (tests and orderly shutdown)."""
    state = _current()
    with state.lock:
        for segment in state.segments.values():
            with contextlib.suppress(OSError):
                os.close(segment.fd)
        state.segments.clear()
        state.unsafe.clear()
