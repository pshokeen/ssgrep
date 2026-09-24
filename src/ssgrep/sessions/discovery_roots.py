"""External transcript roots: opt-in federation beyond the Claude Code corpus.

`SSGREP_TRANSCRIPT_DIRS` names extra directories of transcripts to index
alongside the auto-discovered corpus (field-report item #7). Entry syntax is
colon-separated paths, each optionally tagged with a format adapter:

    SSGREP_TRANSCRIPT_DIRS="/team/notes:native=/exports/claude"

Only the ``native`` adapter (Claude Code record-pair JSONL, the same schema
`ssgrep note` writes) exists today; the registry is the seam future runtime
adapters (prime-agent, codex, ...) plug into. An UNKNOWN format tag raises
UnknownTranscriptFormatError -- loudly, per root, never silently -- because a
typo that silently skipped a root would look exactly like data loss at search
time. A missing DIRECTORY, by contrast, warns on stderr and is skipped:
external roots live on network mounts and laptops; a temporarily absent mount
must not brick indexing (the shrink guard still refuses the resulting
disappearance downstream, refuse-and-explain, so nothing is lost silently).

Scope rule (mirrors notes.py): external roots are EXPLICIT user configuration,
so their files are always in scope -- they bypass cwd matching entirely. The
--scope flag governs which auto-discovered corpus transcripts are included,
not content the user deliberately pointed ssgrep at.

Identity: every external file indexes as a main session whose document id is
its stem plus a short path hash -- deterministic across runs (stable
incremental indexing) and collision-free across roots with same-named files.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from ssgrep.utilities.types import SessionFile

ENV_VAR = "SSGREP_TRANSCRIPT_DIRS"


class UnknownTranscriptFormatError(ValueError):
    """An SSGREP_TRANSCRIPT_DIRS entry names a format with no adapter.

    The `command` attribute is set to ``None`` so the generic exception
    handler's ``carries_remedy`` gate (index.py) suppresses the misleading
    "Try --rebuild" fallback advice: rebuilding cannot fix a config typo and
    could nudge a user toward the destructive --allow-shrink path.
    """

    #: The carries_remedy sentinel (index.py/status.py/init.py all compute
    #: `carries_remedy = getattr(error, "command", None) is not None`).
    #: A non-None value (even an empty string) means "this error already
    #: gives the user the actionable fix; suppress the generic --rebuild advice."
    #: Using "" rather than None because the gate is `is not None`.
    command: str = ""


def _looks_like_native_shard(path: Path) -> bool:
    """Best-effort schema sniff: at least one of the first N non-empty lines
    must parse as a dict with a ``type`` key (the Claude Code record format's
    mandatory field). Checking multiple lines rather than only the first makes
    the filter tolerant of a single corrupt/truncated first line — the scenario
    the brief's own Q.B flags (concurrent writer truncating a line mid-append).
    Fail-open on read errors so the main indexer's malformed-record counter
    can count truly unreadable files. Returns False only when every sampled
    line parses as valid JSON but NONE has a ``type`` key — this is the
    clear-non-transcript signal (data dumps, log files, exports).
    """
    import json as _json

    SAMPLE_LINES = 5  # check first 5 non-empty lines before giving up
    seen = 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seen += 1
                try:
                    obj = _json.loads(line)
                    if isinstance(obj, dict) and "type" in obj:
                        return True
                except ValueError:
                    # Unparseable line (e.g. truncated by concurrent write):
                    # treat as neutral -- don't penalise the whole file for it.
                    pass
                if seen >= SAMPLE_LINES:
                    break
    except OSError:
        return True  # unreadable: pass through to the real parser, which counts it
    return False  # all sampled non-empty lines parsed but none had a `type` key


def _discover_native(root: Path) -> list[SessionFile]:
    """Native-format adapter: every *.jsonl under root, recursively.

    Files that contain no recognisable native record (no line with a ``type``
    key on the first non-empty line) are silently skipped so non-transcript
    .jsonl files (data dumps, logs, exports) in a shared directory do not
    inflate ``session_count`` with no diagnostic signal.
    """
    out: list[SessionFile] = []
    root = root.resolve()
    for shard in sorted(root.rglob("*.jsonl")):
        if not _looks_like_native_shard(shard):
            print(
                f"ssgrep: {ENV_VAR}: {shard} has no recognisable native record "
                f"(no line with a 'type' key); skipping (not a transcript file)",
                file=sys.stderr,
            )
            continue
        digest = hashlib.sha1(str(shard.resolve()).encode("utf-8")).hexdigest()[:8]
        out.append(
            SessionFile(
                path=shard,
                session_id=f"{shard.stem}~{digest}",
                is_main=True,
            )
        )
    return out


#: Format tag -> adapter. The v2 design's seam: future runtime adapters
#: (prime-agent, codex, ...) register here as read-side variants.
ADAPTERS = {"native": _discover_native}


def parse_roots(raw: str) -> list[tuple[str, Path]]:
    """Parse the env value into (format, root) pairs, failing loudly on
    unknown format tags. Empty entries (leading/trailing/double colons) are
    ignored. A tag is only recognized before ``=``; bare entries are native.
    """
    pairs: list[tuple[str, Path]] = []
    for entry in raw.split(":"):
        entry = entry.strip()
        if not entry:
            continue
        fmt, sep, path_part = entry.partition("=")
        if sep and not fmt.startswith(("/", "~", ".")):
            fmt_tag = fmt.strip()
            if fmt_tag not in ADAPTERS:
                raise UnknownTranscriptFormatError(
                    f"{ENV_VAR} entry {entry!r} names unknown format "
                    f"{fmt_tag!r}; known formats: {sorted(ADAPTERS)}"
                )
            path_part = path_part.strip()
            if not path_part:
                raise UnknownTranscriptFormatError(
                    f"{ENV_VAR} entry {entry!r} has an empty path after the "
                    f"format tag '{fmt_tag}=' -- likely an unset shell variable "
                    f"(e.g. native=$UNSET_VAR). Set a real path or remove the entry."
                )
            pairs.append((fmt_tag, Path(path_part).expanduser()))
        else:
            pairs.append(("native", Path(entry).expanduser()))
    return pairs


def discover_external(environ: dict[str, str] | None = None) -> list[SessionFile]:
    """SessionFiles from every configured external root.

    Unknown format tags raise (see module docstring); missing directories
    warn to stderr and are skipped. No env var -> [] (zero behavior change
    for every existing installation).
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV_VAR, "")
    if not raw:
        return []
    out: list[SessionFile] = []
    for fmt, root in parse_roots(raw):
        if not root.is_dir():
            print(
                f"ssgrep: {ENV_VAR} root {root} is not a directory; skipping "
                f"(previously indexed sessions from it will read as vanished; "
                f"the rebuild guard refuses destructive shrinks)",
                file=sys.stderr,
            )
            continue
        out.extend(ADAPTERS[fmt](root))
    return out
