"""Matched source snapshots and owned synthetic-credential lifecycle."""

from __future__ import annotations

import hashlib
import os
import stat

from .live_native import LiveBlocked
from .observer_blob import MAX_SOURCE_BYTES, read_source


def freeze_source(path, root, expected_digest):
    """One bounded no-follow read; immutable bytes, never a later path reopen."""
    data = read_source(path, root)
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise LiveBlocked("matched.source_integrity: source differs from retained A digest")
    return data


def install_source(roots, data, expected_digest):
    """Write the frozen snapshot, then attest the actual indexed bytes."""
    if not isinstance(data, bytes) or len(data) > MAX_SOURCE_BYTES:
        raise LiveBlocked("matched.source_integrity: invalid bounded immutable snapshot")
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise LiveBlocked("matched.source_integrity: snapshot differs from declaration")
    path = roots.collect_view / "verified-a.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fchmod(handle.fileno(), 0o400)
    freeze_source(path, roots.collect_view, expected_digest)
    return path


def _owned_directory(fd):
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise LiveBlocked("matched.credential_cleanup: unsafe parent ownership or permissions")


def _unlink_credential(root_fd, parts):
    """Directory descriptors pin every parent; no solver-controlled link is followed."""
    directory = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
            _owned_directory(directory)
        os.unlink(parts[-1], dir_fd=directory)
    finally:
        os.close(directory)


def remove_credentials(roots):
    """Remove both pre-seal and sealed files, including partially prepared actors."""
    roots.verify_ownership()
    directory = os.open(roots.root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    errors = []
    try:
        # Do not let a renamed ancestor redirect the initial root open either.
        for part in roots.root.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        _owned_directory(directory)
        for actor, runtime in (("solver-a", "runtime-a"), ("solver-b", "runtime-b")):
            for parts in ((runtime, "home", ".pgpass-lab"), ("verifier", actor, "pgpass")):
                try:
                    _unlink_credential(directory, parts)
                except FileNotFoundError:
                    pass
                except (OSError, LiveBlocked) as exc:
                    errors.append(exc)
    finally:
        os.close(directory)
    if errors:
        raise LiveBlocked(
            "matched.credential_cleanup: refused unsafe parent or unlink"
        ) from errors[0]


def close_run(run):
    """Credentials are removed even if observer finalization or DB teardown fails."""
    try:
        try:
            run.observer.finalize()
        finally:
            if not run.instance.removed:
                run.instance.teardown()
    finally:
        remove_credentials(run.roots)
