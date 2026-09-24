"""Process-level file-descriptor headroom for the embedded Lance engine.

Lance's scanner opens one file descriptor per concurrently-read fragment
(plus index and manifest files) and has no open-fd-count throttle of its own;
a single multivector search measured more than 512 simultaneous fds against a
mid-size corpus. macOS defaults the soft limit to 256, so every search fails
there with ``Too many open files (os error 24)``. The hard limit is far
higher on supported platforms (``unlimited`` on macOS), so ssgrep raises the
soft limit once per process immediately before the first Lance connection —
on both the read path (``LanceStore._connect``) and the ingestion path
(``pipeline.app._build_environment``).
"""

from __future__ import annotations

import logging
import resource

logger = logging.getLogger(__name__)

#: Soft-limit target. The measured single-search peak stays below 1024 fds;
#: 10240 is the classic macOS soft cap and leaves >10x headroom. The target
#: is clamped to the hard limit, so a low-hard-limit system keeps its legal
#: maximum instead of failing the raise.
TARGET_SOFT_NOFILE = 10240


def raise_nofile_limit(target: int = TARGET_SOFT_NOFILE) -> bool:
    """Raise the soft ``RLIMIT_NOFILE`` to ``target``; best-effort, idempotent.

    Returns whether the soft limit was raised. Never raises: the raise is a
    resilience measure for the engine's scanner, and a platform that refuses
    it must not break the command about to run. The request is clamped to the
    process hard limit, and a target at or below the current soft limit is a
    no-op (repeated calls after a successful raise report ``False``).
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except OSError:
        logger.debug("RLIMIT_NOFILE query failed; keeping the current limit")
        return False
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    if target <= soft:
        return False
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError):
        logger.debug("RLIMIT_NOFILE raise to %s refused; keeping %s", target, soft)
        return False
    return True
