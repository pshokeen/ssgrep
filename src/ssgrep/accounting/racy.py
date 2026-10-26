"""Git's racy-clean rule for stat-signature caches.

A cache that skips re-reading a file because (dev, ino, size, mtime_ns, ctime_ns)
is unchanged is only sound when a later change must move a timestamp. Linux
stamps inodes from a coarse clock that advances once per scheduler tick (1/HZ;
HZ=100 gives 10 ms), so a same-size rewrite in the tick that produced the cached
timestamps keeps every field equal. A signature is therefore trusted on its own
only when its newest timestamp is at least one granule older than
``time.time_ns()`` read *before* the stat that produced it; otherwise callers
re-read the file or keep the signature only together with a byte comparison.

Filesystems that truncate timestamps to whole seconds (ext4 with 128-byte inodes,
HFS+, FAT's two-second mtime) stamp ``mtime_ns % 10**9 == 0``; such signatures
need a two-second granule instead.

The rule assumes file stamps and ``time.time_ns()`` come from the same clock. A
filesystem stamped by another host's clock (for example NFS with more than one
granule of skew) defeats it, as it does git's index.

Not every before/after comparison uses this rule: the TOCTOU checks in
``sessions/work_evidence.py`` and ``readers/opencode_storage.py`` compare stat
fields taken around a single read and stay blind to a same-size rewrite within
one tick.
"""

from __future__ import annotations

GRANULE_NS = 10_000_000
_SECOND_NS = 1_000_000_000
_WHOLE_SECOND_GRANULE_NS = 2 * _SECOND_NS


def racy(observed_ns: int, *stamps_ns: int) -> bool:
    """True when ``stamps_ns`` (mtime/ctime ns) cannot prove stability after ``observed_ns``.

    ``observed_ns`` is ``time.time_ns()`` read before the stat that produced the stamps;
    no stamps (nothing on disk) is never racy.
    """
    if not stamps_ns:
        return False
    granule = (
        _WHOLE_SECOND_GRANULE_NS
        if all(stamp % _SECOND_NS == 0 for stamp in stamps_ns)
        else GRANULE_NS
    )
    return max(stamps_ns) + granule > observed_ns
