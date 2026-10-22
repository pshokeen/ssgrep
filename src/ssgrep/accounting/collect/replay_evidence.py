"""Cache, normalize, and resolve compact evidence used by deterministic replay."""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import time
import unicodedata
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ssgrep.accounting.collect.read import MAX_ATOMS, MAX_REFS, resolve_ref
from ssgrep.accounting.collect.replay import evidence_from_line, evidence_line
from ssgrep.accounting.ledger.contracts import RawEvidence
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy, SourceAtom
from ssgrep.accounting.ledger.reduce_join import Interpretation
from ssgrep.accounting.ledger.units import GAP_CAP_MS
from ssgrep.accounting.ledger.valuation_inputs import AtomStatus, ValuationAtom
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.readers.native_source import NativeSourceAdapter, NativeSourceStatus
from ssgrep.accounting.readers.timeline import Record, SessionKey, Timeline, Timelines, record_text

if TYPE_CHECKING:
    from ssgrep.accounting.views.filter import Scoper

CACHE_DIR = "cache"
MAX_CACHE_BYTES = 32 * 1024 * 1024


def _name(item: RawEvidence) -> str:
    return (
        hashlib.sha256((item.native_identity + "\0" + item.digest).encode()).hexdigest() + ".json"
    )


def _entries(fd: int, policy: ProfilePolicy, now: int) -> list[tuple[int, str, int]]:
    """One inventory: expire/trim by metadata before reading retained content once."""
    from ssgrep.accounting.views.filter import Scoper

    candidates = []
    for name in storage.list_names(fd):
        try:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:  # pragma: no cover - entry removed between listing and stat
            continue
        expired = info.st_mtime + policy.evidence_cache_retention_days * 86400 <= now
        if not policy.evidence_cache_enabled or expired or name.endswith(".tmp"):
            storage.delete_file(fd, name)
        else:
            candidates.append((info.st_mtime_ns, name, info.st_size))
    ordered = deque(sorted(candidates))
    size = sum(row[2] for row in ordered)
    while ordered and size > policy.evidence_cache_max_bytes:
        _, name, count = ordered.popleft()
        storage.delete_file(fd, name)
        size -= count
    scoper, entries = Scoper(policy), []
    for row in ordered:
        _, name, _ = row
        try:
            raw = storage.read_file(fd, name, policy.evidence_cache_max_bytes)
            item = evidence_from_line(raw) if raw is not None else None
            authorized = item is not None and scoper.in_scope(item.payload.get("cwd"))
        except (ValueError, KeyError, TypeError):
            authorized = False
        if authorized:
            entries.append(row)
        else:
            storage.delete_file(fd, name)
    return entries


def _bounded_policy(policy: ProfilePolicy) -> ProfilePolicy:
    if (
        not policy.collection_enabled
        or not policy.evidence_cache_enabled
        or not 1 <= policy.evidence_cache_retention_days <= 30
        or not 1 <= policy.evidence_cache_max_bytes <= MAX_CACHE_BYTES
    ):
        return replace(
            policy,
            evidence_cache_enabled=False,
            evidence_cache_retention_days=0,
            evidence_cache_max_bytes=0,
        )
    return policy


def maintain_cache(
    profile: str, policy: ProfilePolicy | None = None, *, now: int | None = None
) -> ProfilePolicy:
    with CacheWriter(profile, policy, now=now, create=False) as writer:
        return writer.policy


class CacheWriter:
    """One bounded metadata inventory per fresh collection batch, not per record."""

    def __init__(
        self,
        profile: str,
        policy: ProfilePolicy | None = None,
        *,
        now: int | None = None,
        create: bool = True,
    ) -> None:
        from ssgrep.accounting.privacy.policy import load_profile_policy, policy_path

        self.fd = None
        self._consent_lock = None
        self.policy = _bounded_policy(policy or load_profile_policy(profile))
        try:
            try:
                self.fd = storage.open_profile_dir(profile, "collector", CACHE_DIR, create=False)
            except FileNotFoundError:
                pass
            if self.fd is not None or self.policy.evidence_cache_enabled:
                from ssgrep.accounting.privacy import storage as plainfile

                consent_lock = plainfile.exclusive_lock(policy_path().parent.parent, "accounting")
                consent_lock.__enter__()
                self._consent_lock = consent_lock
                self.policy = _bounded_policy(load_profile_policy(profile))
                if self.fd is None and create and self.policy.evidence_cache_enabled:
                    self.fd = storage.open_profile_dir(profile, "collector", CACHE_DIR)
            from ssgrep.accounting.views.filter import Scoper

            self.scoper = Scoper(self.policy)
            self.entries = (
                deque(_entries(self.fd, self.policy, int(time.time()) if now is None else now))
                if self.fd is not None
                else deque()
            )
            self.names = {row[1] for row in self.entries}
            self.size = sum(row[2] for row in self.entries)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> CacheWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def add(self, item: RawEvidence) -> None:
        if (
            self.fd is None
            or not self.policy.evidence_cache_enabled
            or self.policy.paused
            or "_evidence_format" in item.payload
        ):
            return
        if not self.scoper.in_scope(item.payload.get("cwd")):
            return
        expired_before = (
            time.time_ns() - self.policy.evidence_cache_retention_days * 86400 * 1_000_000_000
        )
        while self.entries and self.entries[0][0] <= expired_before:
            _, old, count = self.entries.popleft()
            storage.delete_file(self.fd, old)
            self.names.discard(old)
            self.size -= count
        body = evidence_line(item, plaintext=True) + b"\n"
        name = _name(item)
        if len(body) > self.policy.evidence_cache_max_bytes or name in self.names:
            return
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            while self.entries and self.size + len(body) > self.policy.evidence_cache_max_bytes:
                _, old, count = self.entries.popleft()
                storage.delete_file(self.fd, old)
                self.names.discard(old)
                self.size -= count
        if not storage.write_once(self.fd, name, body):
            self.entries = deque(_entries(self.fd, self.policy, int(time.time())))
            self.names = {row[1] for row in self.entries}
            self.size = sum(row[2] for row in self.entries)
            return
        info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        self.entries.append((info.st_mtime_ns, name, len(body)))
        self.names.add(name)
        self.size += len(body)

    def close(self) -> None:
        try:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
        finally:
            if self._consent_lock is not None:
                self._consent_lock.__exit__(None, None, None)
                self._consent_lock = None


def cache_collected_evidence(profile: str, items: Sequence[RawEvidence]) -> None:
    """Admit only newly accepted native records at the live/backfill read boundary."""
    if not items:
        return
    with CacheWriter(profile) as writer:
        for item in items:
            writer.add(item)


def cached_evidence(profile: str, item: RawEvidence) -> RawEvidence | None:
    from ssgrep.accounting.privacy.policy import load_profile_policy

    policy = load_profile_policy(profile)
    if (
        not policy.collection_enabled
        or policy.paused
        or not policy.evidence_cache_enabled
        or not 1 <= policy.evidence_cache_retention_days <= 30
        or not 1 <= policy.evidence_cache_max_bytes <= MAX_CACHE_BYTES
    ):
        return None
    from ssgrep.accounting.views.filter import Scoper

    if not Scoper(policy).in_scope(item.payload.get("cwd")):
        return None
    try:
        fd = storage.open_profile_dir(profile, "collector", CACHE_DIR, create=False)
    except FileNotFoundError:
        return None
    try:
        try:
            info = os.stat(_name(item), dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if info.st_mtime + policy.evidence_cache_retention_days * 86400 <= time.time():
            storage.delete_file(fd, _name(item))
            return None
        try:
            raw = storage.read_file(fd, _name(item), policy.evidence_cache_max_bytes)
            if raw is None:  # pragma: no cover - entry removed between stat and read
                return None
            candidate = evidence_from_line(raw)
            candidate = replace(
                candidate,
                payload={k: v for k, v in candidate.payload.items() if k != "_evidence_format"},
            )
            from ssgrep.accounting.readers.base import evidence_digest
            from ssgrep.accounting.readers.native_identity import with_origin

            if candidate.origin_domain is not None and candidate.origin_id is not None:
                digest = with_origin(
                    candidate, candidate.origin_domain, candidate.origin_id, candidate.origin_status
                ).digest
            else:
                digest = evidence_digest(candidate.kind, candidate.payload)
            if (
                candidate.identity != item.identity
                or digest != item.digest
                or candidate.ts_utc != item.ts_utc
            ):
                storage.delete_file(fd, _name(item))
                return None
        except (ValueError, KeyError, TypeError):
            storage.delete_file(fd, _name(item))
            return None
    finally:
        os.close(fd)
    return candidate


NORMALIZATION = "cand2"
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


MIN_PATH_TAIL = 2


def match_keys(token: Token) -> frozenset[str]:
    """Keys under which ``token`` is compared: itself plus whole-segment path tails."""
    keys = {token.key}
    if token.kind == "path":
        parts = [p for p in token.text.split("/") if p not in ("", "~", ".", "..")]
        for start in range(len(parts) - MIN_PATH_TAIL + 1):
            keys.add("path\0" + "/".join(parts[start:]))
    return frozenset(keys)


class TokenKeys:
    """Memoized match keys of record text; every index shares one scan per distinct text."""

    def __init__(self) -> None:
        self._cache: dict[bytes, frozenset[str]] = {}

    def of_text(self, text: str) -> frozenset[str]:
        if not text:
            return frozenset()
        digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
        keys = self._cache.get(digest)
        if keys is None:
            found: set[str] = set()
            for token in tokens_of(text, home=True):
                found |= match_keys(token)
            keys = self._cache[digest] = frozenset(found)
        return keys

    def of(self, record: Record) -> frozenset[str]:
        if record.kind not in ("user_turn", "tool_call", "tool_result"):
            return frozenset()
        return self.of_text(record_text(record))


@dataclass(frozen=True)
class Exposed:
    """A candidate token visible in one lookup result, with the cards that show it."""

    token: Token
    refs: tuple[str, ...]


@dataclass
class Extraction:
    exposed: list[Exposed] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)  # scan-level reason -> count


def extract(interp: Interpretation) -> Extraction:
    """Candidates from the text the model could actually see (never hidden/truncated text)."""
    out = Extraction()
    text_refs: tuple[str, ...] = ()
    if interp.cards:
        sources = [(card.ref, card.text) for card in interp.cards]
    else:
        sources = [("", interp.text)]
        found = {m.group(0) for m in REF_RE.finditer(interp.text) if ":ep:" in m.group(0)}
        text_refs = tuple(sorted(found)[:5])
    refs_by_token: dict[Token, set[str]] = {}
    for ref, text in sources:
        scan = scan_text(text)
        for token in scan.kept:
            refs_by_token.setdefault(token, set()).add(ref)
        for token, reason in scan.excluded.items():
            if token not in refs_by_token:
                out.excluded[reason] = out.excluded.get(reason, 0) + 1
    for token in sorted(refs_by_token):
        refs = tuple(sorted(r for r in refs_by_token[token] if r))
        out.exposed.append(Exposed(token, refs or text_refs))
    return out


def atom_identity(record: Record) -> tuple[str, str, str]:
    origin = record.identity
    return origin, origin, record.ev.digest


class _SourceIndex:
    """Earliest occurrence of each candidate key in one source session, and its first result."""

    def __init__(self, line: Timeline, tk: TokenKeys) -> None:
        self.first: dict[str, tuple[int, str]] = {}
        self.result: dict[str, int] = {}
        for record in line.records:
            for key in tk.of(record):
                if key not in self.first:
                    self.first[key] = (record.seq, record.kind)
                if record.kind == "tool_result" and key not in self.result:
                    self.result[key] = record.seq

    def witness(self, token: Token) -> tuple[int, str, int | None] | None:
        """``(first seq, first kind, first result seq)`` of the whole token, else None."""
        first = self.first.get(token.key)
        return None if first is None else (first[0], first[1], self.result.get(token.key))


@dataclass(frozen=True)
class Consumer:
    """Where the consuming exposure happened (for self-later-work checks)."""

    root: str
    session: SessionKey
    seq: int
    ts: int | None


@dataclass(frozen=True)
class Resolution:
    """Atoms per tier for one candidate, plus why anything was not creditable."""

    immediate: tuple[ValuationAtom, ...] = ()
    discovery: tuple[ValuationAtom, ...] = ()
    episode: tuple[ValuationAtom, ...] = ()
    reasons: tuple[tuple[str, str], ...] = ()
    source_roots: tuple[str, ...] = ()
    witness_ts: int | None = None
    source_user_supplied: bool = False
    locators: tuple[str, ...] = ()


@dataclass
class Resolver:
    timelines: Timelines
    scoper: Scoper
    known_atoms: Mapping[str, str] = field(default_factory=dict)  # origin event -> digest
    root_of: Mapping[SessionKey, SessionKey] = field(default_factory=dict)
    tk: TokenKeys = field(default_factory=TokenKeys)
    native_sources: NativeSourceAdapter | None = None
    _native_lines: dict[SessionKey, Timeline] = field(default_factory=dict)
    _native_refs: dict[str, SessionKey] = field(default_factory=dict)
    _native_failures: dict[str, str] = field(default_factory=dict)
    _indexes: dict[int, tuple[Timeline, _SourceIndex]] = field(default_factory=dict)
    _observed_origins: dict[str, tuple[SessionKey, Record]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._observed_origins = {
            record.identity: (key, record)
            for key, line in self.timelines.sessions.items()
            for record in line.records
        }

    def _index(self, line: Timeline) -> _SourceIndex:
        cached = self._indexes.get(id(line))
        if cached is None:
            index = _SourceIndex(line, self.tk)
            self._indexes[id(line)] = (line, index)
            return index
        return cached[1]

    def _status(self, line: Timeline, record: Record, consumer: Consumer) -> tuple[AtomStatus, str]:
        scope, why = self.scoper.source_status(line.cwd_at(record.seq))
        if scope is not AtomStatus.OK:
            return scope, why
        _, origin, digest = atom_identity(record)
        if origin in self.timelines.conflicted:
            return AtomStatus.DIGEST_CONFLICT, "conflicting_source_variants"
        if origin in self.timelines.ownership_unknown:
            return AtomStatus.UNAVAILABLE, "source_ownership_unknown"
        observed = self._observed_origins.get(origin)
        if observed is not None and observed[1].ev.digest != digest:
            return AtomStatus.DIGEST_CONFLICT, "conflicting_source_variants"
        if observed is not None and observed[0] != line.key:
            owner, original = observed
            root = self.root_of.get(owner, owner)
            if f"{root[0]}:{root[1]}" == consumer.root:
                comparable = consumer.ts is not None and original.ts is not None
                later = (
                    original.seq >= consumer.seq if owner == consumer.session else not comparable
                )
                if later or (comparable and original.ts >= consumer.ts):
                    return AtomStatus.SELF_LATER_WORK, "consumer_own_later_work"
            # A copied global origin does not prove a different containing owner.
            return AtomStatus.UNAVAILABLE, "source_ownership_unknown"
        same_root = self.root_of.get(line.key, line.key)
        if f"{same_root[0]}:{same_root[1]}" == consumer.root:
            same = line.key == consumer.session
            comparable = consumer.ts is not None and record.ts is not None
            later = record.seq >= consumer.seq if same else not comparable
            if later or (comparable and record.ts >= consumer.ts):
                return AtomStatus.SELF_LATER_WORK, "consumer_own_later_work"
        elif consumer.ts is not None and record.ts is not None and record.ts > consumer.ts:
            return AtomStatus.UNAVAILABLE, "source_postdates_exposure"
        return AtomStatus.OK, ""

    def _atom(
        self, line: Timeline, record: Record, consumer: Consumer, reasons: set
    ) -> ValuationAtom | None:
        text = record_text(record)
        if not text:
            return None
        _, origin, digest = atom_identity(record)
        atom_id = origin
        status, why = self._status(line, record, consumer)
        drift = self._drift(origin, digest)
        if status is AtomStatus.OK and drift:
            status, why = AtomStatus.DIGEST_CONFLICT, "source_digest_drift"
        if status is not AtomStatus.OK:
            reasons.add((why, line.key[0] + ":" + line.key[1]))
        start_ms, end_ms, timing_reason = self._gap(line, record, consumer)
        t_start = start_ms // 1000 if start_ms is not None else None
        t_end = end_ms // 1000 if end_ms is not None else None
        source = SourceAtom(atom_id, origin, digest, len(text))
        return ValuationAtom(
            source,
            0,
            len(text),
            t_start,
            t_end,
            None,
            status,
            work_method="capped_gap_v1",
            cost_reason=why or timing_reason,
            actor=f"{line.key[0]}:{line.key[1]}",
            t_start_ms=start_ms,
            t_end_ms=end_ms,
        )

    @staticmethod
    def _gap(
        line: Timeline, record: Record, consumer: Consumer
    ) -> tuple[int | None, int | None, str]:
        """Cap each positive adjacent source gap BEFORE any scenario/actor union."""
        following = line.records[record.seq + 1] if record.seq + 1 < len(line.records) else None
        if following is None:
            return None, None, "source_terminal_gap"
        if following.kind == "user_turn":
            return None, None, "source_task_boundary_gap"
        if following.kind == "tool_call":
            from ssgrep.accounting.readers.classify import classify_evidence

            invocation = classify_evidence(following.ev).invocation
            if invocation is not None and any(
                op in {"search", "show"} for op in invocation.operations
            ):
                return None, None, "source_lookup_boundary_gap"
        if record.ts is None or following.ts is None:
            return None, None, "source_missing_clock"
        if following.ts < record.ts:
            return None, None, "source_clock_reversed"
        if consumer.ts is not None and following.ts > consumer.ts:
            return None, None, "source_gap_after_exposure"
        gap_ms = min(following.ts - record.ts, GAP_CAP_MS)
        return record.ts, record.ts + gap_ms, ""

    def _drift(self, origin: str, digest: str) -> bool:
        """The digest recorded for this origin event earlier differs from what resolves now."""
        held = self.known_atoms.get(origin)
        return held is not None and held != digest

    def resolve(self, token: Token, refs: tuple[str, ...], consumer: Consumer) -> Resolution:
        reasons: set[tuple[str, str]] = set()
        immediate: dict[str, ValuationAtom] = {}
        discovery: dict[str, ValuationAtom] = {}
        episode: dict[str, ValuationAtom] = {}
        roots: set[str] = set()
        locators: set[str] = set()
        witness_ts: int | None = None
        user_supplied = False
        for ref in sorted(set(refs))[:MAX_REFS]:
            key = resolve_ref(ref, self.timelines.sessions)
            line = self.timelines.sessions.get(key) if key is not None else None
            retained_header = line
            if (
                line is not None
                and line.records
                and all(
                    record.kind == "session_header"
                    and not record.ev.payload.get("_replay_unavailable")
                    and self._status(line, record, consumer)[0] is AtomStatus.OK
                    and not self._drift(record.identity, record.ev.digest)
                    for record in line.records
                )
            ):
                # Headers can survive when bodies were outside consumer capture (before
                # setup or during a pause). Authorized source recall is independent of
                # consumer observation; a header alone cannot disprove a native witness.
                line = None
            else:
                retained_header = None
            if line is None and ref in self._native_refs:
                key = self._native_refs[ref]
                line = self._native_lines[key]
            if line is None and ref in self._native_failures:
                reasons.add((self._native_failures[ref], ref))
                continue
            if line is None and self.native_sources is not None:
                native = self.native_sources.resolve(ref)
                if native.line is not None and native.status is NativeSourceStatus.OK:
                    if retained_header is not None and (
                        native.line.key != retained_header.key
                        or any(
                            not any(
                                (record.identity, record.ev.digest, record.ts)
                                == (held.identity, held.ev.digest, held.ts)
                                for record in native.line.records
                            )
                            for held in retained_header.records
                        )
                    ):
                        reasons.add(("source_digest_drift", ref))
                        self._native_failures[ref] = "source_digest_drift"
                        continue
                    line = native.line
                    key = line.key
                    held_line = self._native_lines.get(key)
                    if held_line is not None and (
                        len(held_line.records) != len(line.records)
                        or any(
                            (a.identity, a.ev.digest, a.ts) != (b.identity, b.ev.digest, b.ts)
                            for a, b in zip(held_line.records, line.records, strict=True)
                        )
                    ):
                        reasons.add(("source_digest_drift", ref))
                        self._native_failures[ref] = "source_digest_drift"
                        continue
                    self._native_lines[key] = line
                    self._native_refs[ref] = key
                    for source_record in line.records:
                        self._observed_origins.setdefault(
                            source_record.identity, (key, source_record)
                        )
                else:
                    self._native_failures[ref] = native.reason
                    reasons.add((native.reason, ref))
                    continue
            if line is None or key is None:
                reasons.add(("source_unresolved", "ref"))
                continue
            hit = self._index(line).witness(token)
            if hit is None:
                reasons.add(("witness_not_found", key[0] + ":" + key[1]))
                continue
            first_seq, kind, seq_or_none = hit
            label = key[0] + ":" + key[1]
            if seq_or_none is None:
                reasons.add(("source_not_observed", label))
                continue
            seq = seq_or_none
            if kind == "user_turn" and first_seq < seq:
                user_supplied = True
                reasons.add(("source_user_supplied", label))
                continue
            if kind == "tool_call" and first_seq < seq:
                reasons.add(("source_known_before_witness", label))
                continue
            witness = line.records[seq]
            root_key = self.root_of.get(key, key)
            roots.add(f"{root_key[0]}:{root_key[1]}")
            locators.add(f"{witness.ev.source_id}@{witness.ev.locator}")
            if witness.ts is not None:
                witness_ts = witness.ts if witness_ts is None else min(witness_ts, witness.ts)
            turn = max((t.seq for t in line.turns if t.seq <= seq), default=0)
            later_turns = [t.seq for t in line.turns if t.seq > seq]
            end = (later_turns[0] if later_turns else len(line.records)) - 1
            call = line.call_for(witness)
            invalid = self._atom(line, witness, consumer, set())
            if invalid is not None and invalid.status is not AtomStatus.OK:
                # the witness itself is not creditable, so nothing around it is either
                self._put(immediate, line, witness, consumer, reasons)
                status = invalid.status
            else:
                status = AtomStatus.OK
            for record in ([call] if call is not None else []) + [witness]:
                self._put(immediate, line, record, consumer, reasons, status)
            for record in line.records[turn : seq + 1][:MAX_ATOMS]:
                self._put(discovery, line, record, consumer, reasons, status)
            for record in line.records[turn : end + 1][:MAX_ATOMS]:
                self._put(episode, line, record, consumer, reasons, status)
        if not (immediate or discovery or episode) and not reasons:
            reasons.add(("source_unresolved", "ref"))
        placeholder = ()
        if not (immediate or discovery or episode) and reasons and not user_supplied:
            placeholder = (self._placeholder(token, sorted(reasons)[0][0]),)
        order = lambda items: tuple(items[k] for k in sorted(items))  # noqa: E731
        return Resolution(
            order(immediate) or placeholder,
            order(discovery),
            order(episode),
            tuple(sorted(reasons)),
            tuple(sorted(roots)),
            witness_ts,
            user_supplied,
            tuple(sorted(locators)),
        )

    def _put(self, bucket, line, record, consumer, reasons, forced=AtomStatus.OK) -> None:
        atom = self._atom(line, record, consumer, reasons)
        if atom is not None:
            if forced is not AtomStatus.OK and atom.status is AtomStatus.OK:
                atom = replace(atom, status=forced)
            bucket[atom.atom_id] = atom

    @staticmethod
    def _placeholder(token: Token, reason: str) -> ValuationAtom:
        identity = f"unresolved:{reason}:{token.kind}"
        atom = SourceAtom(identity, identity, "", 0)
        status = (
            AtomStatus.DIGEST_CONFLICT
            if reason == "source_digest_drift"
            else AtomStatus.UNAVAILABLE
        )
        if reason in {"source_out_of_scope", "source_scope_unknown", "source_unapproved"}:
            status = AtomStatus.OUT_OF_SCOPE
        return ValuationAtom(
            atom,
            0,
            0,
            None,
            None,
            None,
            status,
            work_method="capped_gap_v1",
            cost_reason=reason,
        )
