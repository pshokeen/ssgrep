"""One canonicalization for every path identity in ssgrep.

Two identities have to agree or a buyer loses their index:

1. The identity that LOCATES the index directory (``<project>/.ssgrep``).
2. The identity that MATCHES scope (does a transcript's recorded ``cwd``
   fall at or beneath the requested project directory?).

Before this module they came from different code. The scope side went
through ``Path.resolve()``; the ``cwd`` side went through nothing. On a
case-insensitive filesystem that divergence is silent data loss:
``--project-dir ~/code/app`` and ``--project-dir ~/Code/app`` resolve to
the *same physical* ``.ssgrep`` directory, but only one of them matched a
``cwd`` recorded as ``~/Code/app``. Indexing under the other spelling
replaced that index with an empty one. A quoted tilde
(``--project-dir "~/code/x"``) had the same shape of failure by a
different route: ``resolve()`` turns it into ``<cwd>/~/code/x``.

So both identities are produced here, by the same steps:

    expanduser -> lexical normpath -> case-fold IFF the filesystem is
    case-insensitive

``resolve()`` is applied by :func:`resolve_live` only, and only to paths
that exist right now. A recorded ``cwd`` is history: the directory may be
long gone, and ``resolve()`` on a nonexistent relative-looking path
silently anchors it to the *current* working directory, inventing a path
that never existed. :func:`canonical` therefore never touches the
filesystem, and every comparison canonicalizes BOTH sides.

``resolve()`` is deliberately *kept* on the live side. It is load-bearing:
a user who types a symlinked ``--project-dir`` must land on the same index
as one who types the real path. The fix for the mismatch is to canonicalize
the history side to match, never to de-normalize the live side.

Case sensitivity is PROBED at runtime, never assumed from the platform.
macOS is case-insensitive by default but ships case-sensitive APFS as a
supported option, and Linux happily mounts case-insensitive volumes; both
assumptions are wrong on real buyer machines. Note also that
``os.path.normcase`` is a no-op on POSIX -- it folds case only on Windows --
so the fold here is explicit.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

_probe_lock = threading.Lock()
_probe_result: bool | None = None


def _run_probe(directory: str) -> bool:
    """Create a file with an upper-case marker, then stat the lower-case name.

    Returns True if the two names refer to the same file, i.e. the
    filesystem under ``directory`` is case-insensitive.
    """
    fd, created = tempfile.mkstemp(prefix="ssgrep-CASEPROBE-", dir=directory)
    os.close(fd)
    try:
        head, name = os.path.split(created)
        other = os.path.join(head, name.replace("CASEPROBE", "caseprobe"))
        if other == created:  # pragma: no cover - marker is fixed and upper-case
            raise RuntimeError("case probe produced an identical name")
        return os.path.exists(other)
    finally:
        os.unlink(created)


def filesystem_is_case_insensitive() -> bool:
    """Whether path lookups fold case on this machine. Probed, then cached.

    The probe writes into the system temp directory, which on every default
    macOS/Linux/Windows configuration shares its case behaviour with the
    volume holding the user's projects. If the probe cannot run at all
    (read-only temp, sandbox) we fall back to the platform default, which
    is a guess and is labelled as one.
    """
    global _probe_result
    with _probe_lock:
        if _probe_result is None:
            try:
                _probe_result = _run_probe(tempfile.gettempdir())
            except (OSError, RuntimeError):
                _probe_result = sys.platform in ("darwin", "win32")
        return _probe_result


def reset_case_probe() -> None:
    """Drop the cached probe result so the next call re-probes.

    Exists for tests that need to exercise both filesystem behaviours in
    one process.
    """
    global _probe_result
    with _probe_lock:
        _probe_result = None


def fold_case(text: str) -> str:
    """Apply the case rule of the local filesystem to a path string.

    ``os.path.normcase`` normalizes separators (and folds case on Windows);
    the explicit ``lower()`` is what actually folds on POSIX, where
    ``normcase`` returns its argument unchanged.
    """
    if not filesystem_is_case_insensitive():
        return os.path.normcase(text)
    return os.path.normcase(text).lower()


def canonical(path: str | Path) -> Path:
    """The canonical identity of a path, without touching the filesystem.

    Safe for historical paths -- a ``cwd`` read out of a transcript whose
    directory no longer exists canonicalizes exactly as it did when it was
    recorded. Idempotent: ``canonical(canonical(p)) == canonical(p)``.
    """
    text = os.path.expanduser(os.fspath(path))
    text = os.path.normpath(text)
    return Path(fold_case(text))


def resolve_live(path: str | Path) -> Path:
    """A usable, absolute path for a directory that exists right now.

    ``expanduser`` -> ``resolve()``. ``expanduser`` runs *before*
    ``resolve()``, which is what stops a quoted ``"~/code/x"`` from being
    anchored under the current working directory. ``resolve()`` is kept
    because it is load-bearing for a symlinked ``--project-dir``.

    This is deliberately NOT case-folded, and is therefore NOT the identity:
    call :func:`canonical` on the result for that. Two reasons, both about
    what happens when the probe is wrong:

    - A folded path is only guaranteed to open the same files if the
      filesystem really is case-insensitive. The probe reads the system
      temp directory, so a project on a case-sensitive volume alongside a
      case-insensitive ``/tmp`` would get a path that does not exist --
      turning a matching bug into a total loss of the index. Folding only
      at comparison time degrades to "slightly too lenient matching"
      instead.
    - The path is shown to the user and written into ``.gitignore`` and
      Claude Code's settings; the on-disk spelling is the useful one.

    The identities still cannot disagree: the scope string handed to
    matching *is* this same path, and both sides of every comparison go
    through :func:`canonical`.
    """
    expanded = Path(os.path.expanduser(os.fspath(path)))
    return expanded.resolve()


def resolve_claude_dir() -> Path:
    """Claude Code's config directory: ``$CLAUDE_CONFIG_DIR`` or ``~/.claude``.

    Claude Code lets a user relocate its entire config tree with the
    ``CLAUDE_CONFIG_DIR`` environment variable. Everything ssgrep reads and
    writes on Claude Code's side lives under that one root -- the transcripts
    it indexes (``<root>/projects``) and the ``settings.json`` the SessionEnd
    hook is installed into. Hardcoding ``~/.claude`` fails a relocated user
    two ways, and the second is silent: indexing finds no transcripts, and
    ``hooks install`` writes a hook into a settings file Claude Code never
    reads, so the hook never runs and produces no output to notice.

    The environment is read on every call, never captured at import time. A
    user's shell can export the variable after ssgrep is installed, a test
    can set it per-case, and a long-lived MCP server can be restarted under a
    new value; a module-level constant would freeze whichever value happened
    to exist when the first import ran.

    An unset -- or set-but-empty -- variable means "not relocated". Exporting
    an empty string is how shells leave a cleared variable behind, and
    treating that as a root would point ssgrep at the filesystem root.

    The fallback deliberately goes through ``Path.home()`` rather than this
    module's own ``expanduser``: it is the same expression the hardcoded
    sites used, so an unrelocated user's paths are unchanged to the byte
    *before* the resolution below is applied.

    The result is passed through :func:`resolve_live` -- the same live-side
    resolution ``--project-dir`` gets (see identity #1 in this module's own
    docstring) -- so a relocated user who reaches the same physical directory
    through two different spellings (a symlink, e.g. macOS's ``/tmp`` ->
    ``/private/tmp``; or a home directory mounted through a symlink) gets
    back the identical path either way. This used to be unresolved, on the
    theory that "comparisons still go through :func:`canonical`" -- true for
    the scope-vs-cwd identity that theory was written for, but false for the
    transcript-path identity discovery builds *from this function's return
    value*: ``SessionFile.path``, and therefore the ``session_files`` cursor
    primary key, were built directly off whichever spelling the environment
    variable carried, with no canonicalizing step of any kind in between.
    Two spellings of one root indexed the same transcript twice, doubling
    its chunks, and left a cursor for one spelling that could never match a
    stat recorded under the other -- so staleness never cleared (issue #2).
    ``canonical()`` alone cannot fix this: it never touches the filesystem
    and so never resolves a symlink; only ``resolve()`` does, which is why
    this now matches ``resolve_live``'s treatment rather than merely relying
    on downstream ``canonical()`` calls. The returned path is still the live,
    on-disk spelling (post symlink resolution) -- not case-folded -- so it
    remains correct to *open* files and to print in remedies; only
    :func:`canonical` produces the lexically-reduced comparison key.
    """
    raw = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    base = Path.home() / ".claude" if not raw else Path(os.path.expanduser(raw))
    return resolve_live(base)


def is_at_or_beneath(child: str | Path, ancestor: str | Path) -> bool:
    """Whether ``child`` is ``ancestor`` or lives underneath it.

    Both operands go through :func:`canonical` -- the whole point of this
    module is that no comparison may canonicalize only one side.
    """
    try:
        canonical(child).relative_to(canonical(ancestor))
    except ValueError:
        return False
    return True
