"""Pure rule helpers for the lab's independent accounting oracle.

Standard library plus the lab observer only: nothing here may import any ``ssgrep``
valuation/replay/reduce/collector code (the oracle must not be its own subject).
"""

from __future__ import annotations

import json
import posixpath
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from .observer import NativeRecord

SCENARIOS = ("conservative", "central", "optimistic")
REF_RE = re.compile(r"omp:([\w-]+):ep:(\d+)")
DEFAULT_IDENTIFIER = r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+"
ERROR_MARKERS = ("ERROR:", "Traceback")
VIEW_FILTERS = frozenset({"ssgrep", "head", "tail", "cat"})
SKIP_KEYS = frozenset({"ref", "refs", "episode_id", "score", "rank"})
CONTENT_KINDS = ("user_turn", "tool_call", "tool_result")

_QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")
_REDIRECT = re.compile(r"\d*>&\d+|\d*>\s*/dev/null")
_SEPARATORS = re.compile(r"\|\||&&|[|;&\n]")
_ASSIGN = re.compile(r"\w+=\S*")
_PURE_LOOP = re.compile(r"^\s*for\s+\w+\s+in\b.*?;\s*do\s+(.*?)\s*;\s*done\s*$", re.S)


# ------------------------------------------------------------------------------- tokens


@lru_cache(maxsize=512)
def _pattern(token: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])")


def has_token(text: str, token: str) -> bool:
    """Whole-token occurrence: neighbours are never ``[A-Za-z0-9_-]``."""
    return bool(token) and _pattern(token).search(text) is not None


def extract_tokens(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Distinct whole-token matches of ``pattern`` in text order."""
    seen: dict[str, None] = {}
    for found in pattern.finditer(text):
        token = found.group(0)
        if token and has_token(text, token):
            seen.setdefault(token)
    return list(seen)


def candidate_tokens(text: str) -> list[tuple[str, str]]:
    """Independent candidate-vocabulary contract, not a production scanner import."""
    users = set(re.findall(r"/(?:Users|home)/([^/\s\"']+)", text))
    masked = re.sub(r"(?:[a-zA-Z][a-zA-Z0-9+.-]*://\S+|[^\s\"']+:ep:\d+)", " ", text[:262144])
    generic = set(
        (
            "total_matches omitted_count excerpts_truncated index_empty files_touched "
            "source_path source_project parent_session_id git_branch agent_model agent_name "
            "content_type is_subagent source_absent prompt_text response_text is_error "
            "tool_use tool_result tool_call call_id session_id none_observed node_modules"
        ).split()
    )
    generic_flags = set(
        (
            "--help --version --json --quiet --verbose --force --yes --all --limit "
            "--where --token-budget --dry-run --no-color --debug --silent "
            "--output --input --config"
        ).split()
    )
    extensions = set(
        (
            "py ts tsx js jsx json yaml yml toml md txt sql sh rs go java c h cpp hpp "
            "rb lock cfg ini html css csv jsonl lance"
        ).split()
    )
    out: dict[tuple[str, str], None] = {}

    def keep(kind: str, raw: str) -> None:
        token = unicodedata.normalize("NFC", raw)
        if kind == "path":
            token = posixpath.normpath(token)
        if (
            5 <= len(token) <= 160
            and token not in generic
            and token not in users
            and not token.startswith("__")
            and not re.search(r"\d{4}-\d{2}-\d{2}", token)
            and not (kind == "flag" and token in generic_flags)
        ):
            out.setdefault((kind, token), None)

    path = re.compile(
        r"(?<![\w/.~@+-])(?:~|\.{1,2})?(?:/[\w.@+-]+)+|(?<![\w/.~@+-])[\w@+-][\w.@+-]*(?:/[\w.@+-]+)+"
    )
    for match in path.finditer(masked):
        raw = match.group().rstrip(".")
        if re.match(r"^(?:~|/(?:Users|home)/[^/]+)(?:/|$)", raw):
            continue
        parts = [p for p in raw.split("/") if p not in ("", ".", "..")]
        extension = parts[-1].rsplit(".", 1)[-1].lower() if parts else ""
        absolute = raw.startswith(("/", "~", "./", "../"))
        if parts and not all(re.fullmatch(r"[\d_.-]+", p) for p in parts):
            if extension in extensions or (absolute and len(parts) >= 2) or len(parts) >= 3:
                keep("path", raw)
    masked = path.sub(" ", masked)
    flag = re.compile(r"(?<![\w-])--[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?![\w-])")
    for match in flag.finditer(masked):
        keep("flag", match.group())
    masked = flag.sub(" ", masked)
    for match in re.finditer(r"(?<![\w./-])[A-Za-z_]\w*(?:[.-][A-Za-z0-9_]+)*(?![\w-])", masked):
        word = match.group().rstrip(".-")
        error = re.fullmatch(
            r"E[A-Z]{3,12}|ERR_[A-Z0-9_]+|[A-Z]{1,4}\d{3,5}|[A-Z][A-Z0-9_]*_(?:ERROR|ERR|FAILED|FAILURE|INVALID|NOT_FOUND|TIMEOUT|DENIED|EXCEEDED|CONFLICT)",
            word,
        )
        structured = (
            re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", word)
            or re.fullmatch(r"[A-Za-z_]*[a-z][A-Z][A-Za-z0-9]*", word)
            or (
                re.fullmatch(DEFAULT_IDENTIFIER, word)
                and (any(c.isdigit() for c in word) or word.count("-") >= 2)
            )
        )
        parts = word.split(".")
        dotted = len(parts) >= 2 and all(re.fullmatch(r"[A-Za-z_]\w*", p) for p in parts)
        if error:
            keep("error_code", word)
        elif structured or (
            dotted and (len(parts) >= 3 or "_" in word or re.search(r"[a-z][A-Z]", word))
        ):
            keep("identifier", word)
        elif dotted and parts[-1].lower() in extensions:
            keep("path", word)
    return list(out)


def is_mixed(command: str, dispatches: int) -> bool:
    """True when lookup output shares an invocation with unrelated shell output."""
    plain = _REDIRECT.sub(" ", _QUOTED.sub("''", command.replace("\\\n", " ")))
    loop = _PURE_LOOP.fullmatch(plain)
    if loop and dispatches > 1:
        segments = _SEPARATORS.split(loop.group(1))
        for segment in segments:
            words = segment.split()
            while words and _ASSIGN.fullmatch(words[0]):
                words.pop(0)
            if words and words[0] not in VIEW_FILTERS:
                return True
        return False
    for segment in _SEPARATORS.split(plain):
        words = segment.split()
        while words and _ASSIGN.fullmatch(words[0]):
            words.pop(0)
        if words and words[0] not in VIEW_FILTERS:
            return True
    return False


def is_error_text(text: str) -> bool:
    return any(marker in text for marker in ERROR_MARKERS)


# -------------------------------------------------------------------------------- cards


@dataclass(frozen=True)
class Card:
    ref: str | None
    text: str


@dataclass(frozen=True)
class Parsed:
    status: str  # json | fallback | none | error | empty_exit
    cards: tuple[Card, ...]
    ok: bool | None
    empty: bool
    refs: tuple[str, ...]  # distinct visible refs; the only ref evidence in fallback status
    text: str


def _leaves(value: object, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in SKIP_KEYS:
                _leaves(item, out)
    elif isinstance(value, list):
        for item in value:
            _leaves(item, out)


def _ref_of(card: dict) -> str | None:
    """A card's ref: the first ``ref``/``episode_id`` that looks like an episode ref, else any."""
    named = [card[k] for k in ("ref", "episode_id") if isinstance(card.get(k), str)]
    return next((n for n in named if REF_RE.fullmatch(n)), named[0] if named else None)


def _pick(ref: str | None, hint: str | None) -> str | None:
    if ref and REF_RE.fullmatch(ref):
        return ref
    return hint or ref


def _from_json(obj: dict, hint_ref: str | None, text: str) -> Parsed:
    ok = obj["ok"] if isinstance(obj.get("ok"), bool) else None
    if ok is False:
        return Parsed("json", (), False, False, (), text)
    payload = obj
    while isinstance(payload.get("data"), dict) or isinstance(payload.get("episode"), dict):
        payload = payload["data"] if isinstance(payload.get("data"), dict) else payload["episode"]
    results = payload.get("results")
    if isinstance(results, list):
        cards = []
        for item in results:
            if isinstance(item, dict):
                leaves: list[str] = []
                _leaves(item, leaves)
                cards.append(Card(_ref_of(item), "\n".join(leaves)))
        return Parsed("json", tuple(cards), ok, not results, (), text)
    if "prompt_text" in payload or "response_text" in payload:
        body = "\n".join(str(payload.get(k) or "") for k in ("prompt_text", "response_text"))
        return Parsed("json", (Card(_pick(_ref_of(payload), hint_ref), body),), ok, False, (), text)
    return Parsed("json", (), ok, False, (), text)


def parse_visible(text: str, *, strict: bool, hint_ref: str | None = None) -> Parsed:
    """Parse what the model saw. ``strict`` (mixed shell output) demands a whole JSON document."""
    body, obj = text.strip(), None
    try:
        obj = json.loads(body)
    except ValueError:
        start = body.find("{")
        if not strict and start >= 0:
            try:
                obj = json.JSONDecoder().raw_decode(body[start:])[0]
            except ValueError:
                obj = None
    if isinstance(obj, dict):
        return _from_json(obj, hint_ref, text)
    if strict:
        return Parsed("none", (), None, False, (), text)
    refs = tuple(dict.fromkeys(found.group(0) for found in REF_RE.finditer(text)))
    if hint_ref and not refs:
        refs = (hint_ref,)
    return Parsed("fallback", (Card(None, text),), None, False, refs, text)


def refs_for_token(parsed: Parsed, token: str) -> list[str | None]:
    """Refs a token is visible under: its carrying cards, or every visible ref when unparsed.

    Unparsed (truncated) text cannot say which card carries the token; the source session
    must itself contain the token as a tool result, so a wrong guess cannot earn credit.
    """
    if parsed.status == "json":
        return list(dict.fromkeys(c.ref for c in parsed.cards if has_token(c.text, token)))
    if parsed.status != "fallback" or not has_token(parsed.text, token):
        return []
    return list(parsed.refs) or [None]


def card_texts(parsed: Parsed) -> list[str]:
    return [card.text for card in parsed.cards]


# ------------------------------------------------------------------------------- source


@dataclass(frozen=True)
class Source:
    reason: str | None
    witness: int | None


def resolve_source(token: str, records: Sequence[NativeRecord], exposure_ts: int | None) -> Source:
    first = next(
        (i for i, r in enumerate(records) if r.kind in CONTENT_KINDS and has_token(r.text, token)),
        None,
    )
    if first is None:
        return Source("source_not_observed", None)
    if records[first].kind == "user_turn":
        return Source("source_user_supplied", None)
    witness = next(
        (
            i
            for i in range(first, len(records))
            if records[i].kind == "tool_result" and has_token(records[i].text, token)
        ),
        None,
    )
    if witness is None:
        return Source("source_not_observed", None)
    if records[first].kind == "tool_call":
        return Source("source_known_before_witness", witness)
    stamp = records[witness].ts_ms
    if stamp is None or exposure_ts is None:
        return Source("source_time_unknown", witness)
    if stamp >= exposure_ts:
        return Source("source_self_or_later", witness)
    return Source(None, witness)


def task_bounds(records: Sequence[NativeRecord], index: int) -> tuple[int, int]:
    """[start, end) of the task holding ``records[index]`` (start 0 before any user turn)."""
    start = next((i for i in range(index, -1, -1) if records[i].kind == "user_turn"), 0)
    end = next(
        (i for i in range(index + 1, len(records)) if records[i].kind == "user_turn"),
        len(records),
    )
    return start, end


def source_atoms(
    session_id: str, records: Sequence[NativeRecord], witness: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    """(immediate, discovery, episode) atom ids; None when the producing call is missing."""
    result = records[witness]
    call = next(
        (
            i
            for i in range(witness - 1, -1, -1)
            if records[i].kind == "tool_call"
            and result.call_id is not None
            and records[i].call_id == result.call_id
        ),
        None,
    )
    if call is None:
        return None
    start, end = task_bounds(records, witness)

    def ident(i: int) -> str:
        return f"{session_id}:{records[i].record_id}"

    immediate = tuple(sorted({ident(call), ident(witness)}))
    discovery = tuple(sorted({ident(i) for i in range(start, witness + 1) if records[i].text}))
    episode = tuple(sorted({ident(i) for i in range(start, end) if records[i].text}))
    return immediate, discovery, episode
