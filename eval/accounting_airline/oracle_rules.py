"""Pure rule helpers for the lab's independent accounting oracle.

Standard library plus the lab observer only: nothing here may import any ``ssgrep``
valuation/replay/reduce/collector code (the oracle must not be its own subject).
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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


# ------------------------------------------------------------------------ import boundary

_DOTTED = re.compile(r"ssgrep\.accounting\.\w+(?:\.\w+)*")
_FORBIDDEN_LEAVES = ("valuation", "replay", "reduce", "collector")


def imported_modules(path: Path) -> list[str]:
    """Every dotted module the source imports (``from a import b`` yields ``a`` and ``a.b``)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append(node.module)
            found.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _DOTTED.fullmatch(node.value):  # importlib-style string imports
                found.append(node.value)
    return found


def import_violations(path: Path) -> list[str]:
    """Imports of production valuation/replay/reduce/collector code in ``path`` (direct only)."""
    out = []
    for module in imported_modules(path):
        top = module.split(".")
        if top[0] != "ssgrep":
            continue
        leaf = top[2] if len(top) > 2 and top[1] == "accounting" else ""
        if any(leaf.startswith(name) for name in _FORBIDDEN_LEAVES):
            out.append(f"{Path(path).name}: forbidden import {module}")
    return list(dict.fromkeys(out))
