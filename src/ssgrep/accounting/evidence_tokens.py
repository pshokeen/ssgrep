"""Deterministic candidate tokens: identifiers, paths, flags, and error codes.

One scanner serves BOTH extraction and the frequency view, so the two always agree
on normalization (design D5). Matching is whole-token only: a candidate is an
extracted, normalized token and later references are compared as normalized
tokens, never as arbitrary substrings.

Excluded at scan time (with a reason, never silently): refs, dates, URLs, home
roots and usernames, and generic tokens. Query/user-supplied/prior-known facts
are excluded by the caller, which knows the context.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from dataclasses import dataclass, field

NORMALIZATION = "cand1"
MAX_SCAN_CHARS = 256 * 1024
MIN_LEN = 5
MAX_LEN = 160
TYPES = ("identifier", "path", "flag", "error_code")

REF_RE = re.compile(
    r"[^\s\"'`,\[\]{}()<>]+:ep:\d+"
    r"|\b(?:claude|codex|omp|pi|prime-agent|opencode):[\w.\-/]+"
)
URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s\"'`<>)\]]+")
SEG = r"[\w.@+\-]+"
PATH_RE = re.compile(
    rf"(?<![\w/.~@+\-])((?:~|\.{{1,2}})?(?:/{SEG})+|[\w@+\-][\w.@+\-]*(?:/{SEG})+)(?![\w@+\-])"
)
FLAG_RE = re.compile(r"(?<![\w\-])(--[a-z][a-z0-9]*(?:-[a-z0-9]+)*)(?![\w\-])")
WORD_RE = re.compile(r"(?<![\w\-./])[A-Za-z_][\w]*(?:[.\-][A-Za-z0-9_]+)*(?![\w\-])")
ERROR_RE = re.compile(
    r"^(?:E[A-Z]{3,12}|ERR_[A-Z0-9_]+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_(?:ERROR|ERR|FAILED|FAILURE"
    r"|INVALID|NOT_FOUND|TIMEOUT|DENIED|EXCEEDED|CONFLICT)|[A-Z]{1,4}\d{3,5})$"
)
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
HOME_RE = re.compile(r"^(?:~|/(?:Users|home)/[^/]+)(?:/|$)")
HOME_USER_RE = re.compile(r"/(?:Users|home)/([^/\s\"']+)")
EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")
FILE_EXTENSIONS = frozenset(
    "py ts tsx js jsx json yaml yml toml md txt sql sh rs go java c h cpp hpp rb lock cfg ini "
    "html css csv jsonl lance".split()
)
GENERIC = frozenset(
    """total_matches omitted_count excerpts_truncated index_empty files_touched source_path
    source_project parent_session_id git_branch agent_model agent_name content_type
    is_subagent source_absent prompt_text response_text is_error tool_use tool_result
    tool_call call_id session_id none_observed __init__ __main__ __name__ __file__
    e.g i.e etc. node_modules""".split()
)
GENERIC_FLAGS = frozenset(
    """--help --version --json --quiet --verbose --force --yes --all --limit --where
    --token-budget --dry-run --no-color --debug --silent --output --input --config""".split()
)


@dataclass(frozen=True, order=True)
class Token:
    """A normalized candidate; ``kind`` is one of :data:`TYPES`."""

    kind: str
    text: str

    @property
    def key(self) -> str:
        return f"{self.kind}\0{self.text}"


@dataclass
class Scan:
    kept: set[Token] = field(default_factory=set)
    excluded: dict[Token, str] = field(default_factory=dict)  # token -> reason (first wins)


def normalize(kind: str, text: str) -> str:
    """The single normalization shared by extraction, matching, and frequency."""
    value = unicodedata.normalize("NFC", text)
    if kind == "path":
        value = posixpath.normpath(value) if value not in ("/", "") else value
        value = value[2:] if value.startswith("./") else value
    elif kind == "flag":
        value = value.lower()
    return value


def _mask(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _path_ok(text: str) -> str | None:
    """None when ``text`` is an acceptable path candidate, else the exclusion reason."""
    segments = [part for part in text.split("/") if part and part not in ("~", ".", "..")]
    if len(segments) < 1 or all(re.fullmatch(r"[\d\-_.]+", part) for part in segments):
        return "date_or_number"
    if any(DATE_RE.fullmatch(part) for part in segments):
        return "date_or_number"
    absolute = text.startswith(("/", "~", "./", "../"))
    last = segments[-1]
    ext = EXTENSION_RE.search(last)
    has_ext = bool(ext) and ext.group(0)[1:].lower() in FILE_EXTENSIONS
    if absolute and len(segments) < 2 and not has_ext:
        return "generic"
    if not absolute and len(segments) < 2:
        return "generic"
    if len(segments) < 3 and not has_ext and not absolute:
        return "generic"  # e.g. "and/or", "read/write"
    return None


def _ident_kind(word: str) -> str | None:
    """``identifier``/``error_code`` when a bare word has identifier structure."""
    if ERROR_RE.fullmatch(word):
        return "error_code"
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", word):
        return "identifier"
    if re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", word):
        return "identifier"
    if re.fullmatch(r"[A-Za-z_]*[a-z][A-Z][A-Za-z0-9]*", word):
        return "identifier"
    if re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){1,}", word) and (
        any(ch.isdigit() for ch in word) or word.count("-") >= 2
    ):
        return "identifier"
    parts = word.split(".")
    if len(parts) >= 3 and all(len(p) >= 2 and re.fullmatch(r"[A-Za-z_]\w*", p) for p in parts):
        return "identifier"
    if len(parts) == 2 and all(re.fullmatch(r"[A-Za-z_]\w*", p) for p in parts):
        if parts[1].lower() in FILE_EXTENSIONS and len(parts[0]) >= 2:
            return "path"  # a bare file name
        if "_" in word or re.search(r"[a-z][A-Z]", word):
            return "identifier"
    return None


def scan_text(text: str, *, home: bool = False) -> Scan:
    """Extract and classify candidates from observed result content.

    ``home=True`` keeps home-rooted paths: used only to MATCH later actions, never to
    propose a candidate.
    """
    out = Scan()
    if not text:
        return out
    if len(text) > MAX_SCAN_CHARS:
        text = text[:MAX_SCAN_CHARS]
    users = set(HOME_USER_RE.findall(text))
    text = _mask(_mask(text, URL_RE), REF_RE)

    def keep(kind: str, raw: str) -> None:
        value = normalize(kind, raw)
        token = Token(kind, value)
        if not MIN_LEN <= len(value) <= MAX_LEN:
            return
        if value in GENERIC or value.rstrip(".") in GENERIC or value.startswith("__"):
            out.excluded.setdefault(token, "generic")
        elif kind == "flag" and value in GENERIC_FLAGS:
            out.excluded.setdefault(token, "generic")
        elif value in users and not home:
            out.excluded.setdefault(token, "username")
        elif DATE_RE.search(value) and kind != "path":
            out.excluded.setdefault(token, "date")
        else:
            out.kept.add(token)

    for match in PATH_RE.finditer(text):
        raw = match.group(1).rstrip(".")
        if HOME_RE.match(raw) and not home:
            out.excluded.setdefault(Token("path", normalize("path", raw)), "home_path")
            continue
        reason = _path_ok(raw)
        if reason:
            continue
        keep("path", raw)
    text = _mask(text, PATH_RE)
    for match in FLAG_RE.finditer(text):
        keep("flag", match.group(1))
    text = _mask(text, FLAG_RE)
    for match in WORD_RE.finditer(text):
        raw = match.group(0).rstrip(".-")
        kind = _ident_kind(raw)
        if kind:
            keep(kind, raw)
    return out


def tokens_of(text: str, *, home: bool = False) -> set[Token]:
    """Kept tokens only (action/relay matching with ``home=True``; frequency documents)."""
    return scan_text(text, home=home).kept
