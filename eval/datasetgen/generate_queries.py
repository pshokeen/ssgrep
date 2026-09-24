"""Query generation stage (Task 12 of the retrieval-eval overhaul).

Consumes ``eval/datasetgen/sessions.parquet`` (T6 output: 400 fictional coding
sessions across the 5 runtimes and 7 scenario classes) and authors ~600+
labelled queries in the 7 D5 classes, each with generator-grounded targets,
confusable hard negatives, and a frozen 80/20 holdout split assigned at
generation time (seeded RNG, component-based so no episode is a target of both
a train and a holdout query).

Output: ``eval/datasetgen/queries.jsonl`` -- one JSON object per line. The
legacy query-label field names are preserved (``id``, ``query``,
``class``, ``anchors``, ``anchor_mode``, ``target_episode_ids``,
``target_session_ids``, ``subagent_only``, ``notes``) plus the new fields
``split`` (``train`` | ``holdout``), ``runtime`` (the stratification cell),
``hard_negative_episode_ids`` (confusable same-project episodes expected to
grade 0), and ``grounding`` (bounded per-target prompt/response excerpts for
the T14 judge).

Episode id derivation (documented schemes from the T7-T11 emitter learnings;
T14 reconciles against the actual ingested index):

- claude:      ``<stem>~<8-hex-sha1-of-absolute-path>`` where the stem is the
               raw ``claude-<nnnn>`` id and the hash is computed over the
               canonical relative path ``transcripts/claude/<raw>.jsonl``
               (the T13/T15 dataset layout; the absolute prefix is unknown at
               generation time, so T14 must reconcile the hash).
- codex:       ``codex:codex-<nnnn>``
- pi:          ``pi:pi-<nnnn>``
- prime-agent: ``prime-agent:<parquet session_id>``
- opencode:    ``opencode:<parquet session_id>`` (fallback ``sess-<nnnn>``)

Raw ids for claude/codex/pi are content-derived: rows are sorted by
``_row_key`` = (project, title, episodes-json) exactly as the emitters do, so
``<runtime>-<nnnn>`` matches the emitted file names. Episodes are
``<session_id>:ep:<k>`` with ``k`` the text-bearing episode index (emitters
drop text-less prompts, and every parquet episode has a non-empty prompt).

Anti-leakage discipline: paraphrase / decision-rationale /
tool-failure-recovery / multi-hop queries are authored from the target
episode's summary with every anchor substring removed, then verified by
``_leakage_safe`` so no query text contains any of its anchors.
exact-identifier and error-string queries are literal by construction;
cross-runtime/project-scoped queries name their project plus a content topic
shared with both target episodes (the project slug alone appears in almost no
target text, so the topic carries the retrieval signal). The split is assigned
over connected components of the query-target episode graph, so an episode can
never be graded by both a train and a holdout query.

Determinism: fixed seed, sorted iteration everywhere, no wall-clock values;
the same parquet produces a byte-identical ``queries.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

#: Fixed seed for the holdout split RNG (never wall-clock).
SEED = 0x12C0FFEE
HOLDOUT_FRACTION = 0.2
TRAIN_FRACTION = 1.0 - HOLDOUT_FRACTION
MIN_PER_CLASS = 60
TARGET_PER_CLASS = 86
#: Per-runtime query budget within each class (sums to TARGET_PER_CLASS).
#: Every cell is >= 6 so the 80/20 split leaves >= 5 in train per cell.
PER_CLASS_RUNTIME_TARGETS: dict[str, int] = {
    "opencode": 34,
    "codex": 14,
    "prime-agent": 14,
    "claude": 12,
    "pi": 12,
}
MIN_PER_CELL = 6

SCENARIO_CLASSES: tuple[str, ...] = (
    "exact-identifier",
    "error-string",
    "paraphrase",
    "multi-hop",
    "decision-rationale",
    "cross-runtime/project-scoped",
    "tool-failure-recovery",
)

RUNTIMES: tuple[str, ...] = ("claude", "codex", "opencode", "pi", "prime-agent")

CLASS_PREFIX: dict[str, str] = {
    "exact-identifier": "exact",
    "error-string": "error",
    "paraphrase": "para",
    "multi-hop": "multi",
    "decision-rationale": "decis",
    "cross-runtime/project-scoped": "cross",
    "tool-failure-recovery": "tool",
}

#: Canonical relative path prefix for claude external-root files (the T13/T15
#: dataset layout); the sha1 in the indexed session id is over this path.
_CLAUDE_TRANSCRIPTS_REL = "transcripts/claude"

#: Fictional persona names from the T6 workflow; stripped from summaries so
#: they never become query topic words.
_PERSONA_NAMES: frozenset[str] = frozenset(
    "Ada Mercer David Okoro Elena Vasquez Finn Calloway Grace Lindqvist Imani Coleman "
    "Jules Ferreira Kai Nakamura Marek Novak Priya Anand Theo Marcotte Yuki Tanabe".split()
)

_STOPWORDS: frozenset[str] = frozenset(
    "a an and are as at be been being but by for from has have had how i in is it its of on "
    "or that the this to was we were what when where which who will with you your our my "
    "there here then now also with from into onto about for and but or of in on at to by as "
    "can could would should may might must do does did not no yes so if then than very just "
    "more most some any all each every both few such only own same too".split()
)

#: Generic coding verbs that carry no retrieval signal; stripped from topics.
_GENERIC_VERBS: frozenset[str] = frozenset(
    "add added adding address addresses addressing build built change changed changing "
    "check checking create created creating debug debugging do does doing ensure ensure "
    "fix fixed fixing get getting handle handled handling help helped implement implemented "
    "implementing improve improved improving include included including integrate integrated "
    "integrating make makes making need needed needs optimize optimized optimizing provide "
    "provided providing refactor refactored refactoring remove removed removing run running "
    "set setting setup show showing support supported supporting test tested testing try "
    "tried trying update updated updating use used using want wanted wants work worked working "
    "write wrote writing can could should would will may might must".split()
)

#: Session-summary boilerplate prefixes stripped before topic extraction.
_SUMMARY_PREFIXES: tuple[str, ...] = (
    "in this coding session",
    "in this session",
    "during this session",
    "during a routine update",
    "during the development",
    "this session",
)

_PARAPHRASE_TEMPLATES: tuple[str, ...] = (
    "how did we handle the {topic}",
    "what is the approach for the {topic}",
    "how should the {topic} be done",
)

_DECISION_TEMPLATES: tuple[str, ...] = (
    "why did we decide on the {topic}",
    "what was the reasoning behind the {topic}",
    "why is the {topic} the chosen approach",
)

_TOOL_FAILURE_TEMPLATES: tuple[str, ...] = (
    "how did we recover when the {topic} failed",
    "what went wrong with the {topic} and how was it fixed",
    "how was the {topic} failure resolved",
)

_MULTI_TEMPLATES: tuple[str, ...] = (
    "how does the {topic} relate to the {topic2}",
    "what connects the {topic} with the {topic2}",
    "why did the {topic} lead to the {topic2}",
)

#: Generic fallback topics when every template leaks an anchor.
_FALLBACK_TOPICS: tuple[str, str] = ("the recent work", "the follow-up")

_IDENT_RE = re.compile(
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b|\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b"
)

_ERROR_RE = re.compile(
    r"\b[A-Z][A-Za-z]*(?:Error|Exception)\b\s*:[^\n]{0,100}"
    r"|\b[A-Z][A-Za-z]*(?:Error|Exception)\b"
    r"|\b(?:failed to|unable to|could not|no such|not found|invalid|cannot|refusing to|"
    r"must be|missing|unsupported|unknown)\b[^\n]{0,80}"
    r"|\berror\s*:[^\n]{0,80}"
)

_NUMBER_RE = re.compile(r"\b\d{4,}(?:[.,]\d+)*\b")

_DECISION_RE = re.compile(
    r"\b(?:we chose|the reason|because|prefer|decided|recommend|best practice|better to|"
    r"instead of|rather than|trade-?off|rationale|why)\b",
    re.IGNORECASE,
)

_FAILURE_RE = re.compile(
    r"\b(?:fail(?:ed|ure|ing)?|error|crash|broken|exception|retry|recover|fixed|resolved|"
    r"workaround|rollback)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Index:
    """Episode/session lookup tables derived once from the parquet."""

    episodes: dict[str, dict[str, Any]]
    by_runtime: dict[str, list[str]] = field(default_factory=dict)
    by_project: dict[str, list[str]] = field(default_factory=dict)
    by_session: dict[str, list[str]] = field(default_factory=dict)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Content-derived sort key (identical to the T7-T11 emitters)."""
    return (
        str(row.get("project", "")),
        str(row.get("title", "")),
        json.dumps(row.get("episodes", []), sort_keys=True),
    )


def _episode_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    """Normalize the ``episodes`` column to ``(prompt, response)`` pairs.

    Text-less prompts are skipped exactly as the emitters skip them
    (episodes.py:81-87), so episode numbering matches the ingested index.
    """
    pairs: list[tuple[str, str]] = []
    episodes = row.get("episodes")
    if not isinstance(episodes, list):
        return pairs
    for item in episodes:
        if isinstance(item, dict):
            prompt = item.get("prompt")
            response = item.get("response")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            prompt, response = item[0], item[1]
        else:
            continue
        if prompt is None or str(prompt) == "":
            continue
        pairs.append((str(prompt), str(response or "")))
    return pairs


def _session_id(runtime: str, raw_id: str) -> str:
    """Indexed session id per the documented emitter scheme."""
    if runtime == "claude":
        digest = hashlib.sha1(f"{_CLAUDE_TRANSCRIPTS_REL}/{raw_id}.jsonl".encode()).hexdigest()[:8]
        return f"{raw_id}~{digest}"
    if runtime == "codex":
        return f"codex:{raw_id}"
    if runtime == "pi":
        return f"pi:{raw_id}"
    if runtime == "prime-agent":
        return f"prime-agent:{raw_id}"
    return f"opencode:{raw_id}"


def _build_index(rows: list[dict[str, Any]]) -> _Index:
    """Derive episode ids and lookup tables from the parquet rows.

    Raw ids for claude/codex/pi are content-derived (``_row_key`` sort, like
    the emitters); prime-agent/opencode use the parquet ``session_id`` column.
    """
    sorted_rows = sorted(rows, key=_row_key)
    counters: dict[str, int] = {}
    raw_ids: dict[int, str] = {}
    for pos, row in enumerate(sorted_rows):
        runtime = row["runtime"]
        if runtime in ("prime-agent", "opencode"):
            raw = str(row.get("session_id") or f"sess-{pos:04d}")
        else:
            n = counters.get(runtime, 0)
            counters[runtime] = n + 1
            raw = f"{runtime}-{n:04d}"
        raw_ids[pos] = raw

    episodes: dict[str, dict[str, Any]] = {}
    by_runtime: dict[str, list[str]] = {runtime: [] for runtime in RUNTIMES}
    by_project: dict[str, list[str]] = {}
    by_session: dict[str, list[str]] = {}
    sessions: dict[str, dict[str, Any]] = {}

    for pos, row in enumerate(sorted_rows):
        runtime = row["runtime"]
        raw = raw_ids[pos]
        session_id = _session_id(runtime, raw)
        project = str(row.get("project") or "")
        summary = str(row.get("summary") or "")
        sessions[session_id] = {
            "runtime": runtime,
            "project": project,
            "title": str(row.get("title") or ""),
            "summary": summary,
            "raw_id": raw,
        }
        pairs = _episode_pairs(row)
        episode_ids = [f"{session_id}:ep:{k}" for k in range(len(pairs))]
        by_session[session_id] = episode_ids
        for k, (prompt, response) in enumerate(pairs):
            episode_id = f"{session_id}:ep:{k}"
            episodes[episode_id] = {
                "runtime": runtime,
                "project": project,
                "session_id": session_id,
                "prompt": prompt,
                "response": response,
                "summary": summary,
            }
            by_runtime[runtime].append(episode_id)
            by_project.setdefault(project, []).append(episode_id)

    for values in by_runtime.values():
        values.sort()
    for values in by_project.values():
        values.sort()
    return _Index(
        episodes=episodes,
        by_runtime=by_runtime,
        by_project=by_project,
        by_session=by_session,
        sessions=sessions,
    )


def _identifiers(text: str) -> list[str]:
    """snake_case (>= 2 parts) and camelCase tokens, deduped, sorted."""
    return sorted({match.group(0) for match in _IDENT_RE.finditer(text)})


def _error_substrings(text: str) -> list[str]:
    """Distinct error-message substrings, quality-filtered, sorted."""
    subs: set[str] = set()
    for match in _ERROR_RE.finditer(text):
        sub = match.group(0).strip().rstrip(".,;:!?")
        if len(sub) < 6:
            continue
        if sub.lower() in ("error", "error:"):
            continue
        subs.add(sub)
    return sorted(subs)


def _distinctive_anchor(text: str) -> str:
    """The most memorable term in an episode: identifier > error > number."""
    candidates = [item for item in _identifiers(text) if len(item) >= 4]
    candidates += [item for item in _error_substrings(text) if len(item) >= 6]
    candidates += [match.group(0) for match in _NUMBER_RE.finditer(text)]
    if not candidates:
        candidates = [
            word
            for word in re.split(r"[^a-zA-Z0-9]+", text.lower())
            if len(word) >= 4 and word not in _STOPWORDS
        ]
    candidates.sort(key=lambda item: (-len(item), item))
    return candidates[0]


def _topic_words(text: str, anchors: list[str], max_words: int = 6) -> str:
    """Content words from an episode prompt/summary with anchors removed."""
    for anchor in anchors:
        text = text.replace(anchor, " ")
    words = [re.sub(r"[^a-zA-Z0-9]+", "", word).lower() for word in re.split(r"\s+", text) if word]
    lowered = " ".join(words)
    for prefix in _SUMMARY_PREFIXES:
        if lowered.startswith(prefix):
            words = words[len(prefix.split()) :]
            break
    if (
        len(words) >= 2
        and words[0].capitalize() in _PERSONA_NAMES
        and words[1].capitalize() in _PERSONA_NAMES
    ):
        words = words[2:]
    kept: list[str] = []
    for word in words:
        if word and word not in _STOPWORDS and word not in _GENERIC_VERBS and word not in kept:
            kept.append(word)
        if len(kept) >= max_words:
            break
    return " ".join(kept)


def _leaks(query: str, anchors: list[str]) -> bool:
    """True when any anchor substring appears in the query (case-insensitive)."""
    lowered = query.lower()
    return any(anchor and len(anchor) >= 4 and anchor.lower() in lowered for anchor in anchors)


def _leakage_safe(
    templates: tuple[str, ...],
    topics: list[str],
    anchors: list[str],
) -> str:
    """First template whose rendered query contains no anchor substring.

    Falls back to generic topics when every template leaks; raises when even
    the generic fallback leaks (should never happen with real anchors).
    """
    for template in templates:
        query = template.format(topic=topics[0], topic2=topics[1] if len(topics) > 1 else topics[0])
        if not _leaks(query, anchors):
            return query
    for template in templates:
        query = template.format(topic=_FALLBACK_TOPICS[0], topic2=_FALLBACK_TOPICS[1])
        if not _leaks(query, anchors):
            return query
    raise ValueError(f"could not author a leakage-safe query for anchors {anchors!r}")


def _shared_content_words(query: str, text: str) -> int:
    """Count query content words that also appear in ``text``."""
    query_words = {
        word
        for word in re.split(r"[^a-zA-Z0-9]+", query.lower())
        if word and word not in _STOPWORDS
    }
    text_words = set(re.split(r"[^a-zA-Z0-9]+", text.lower()))
    return len(query_words & text_words)


def _hard_negatives(
    index: _Index,
    query: str,
    target_ids: set[str],
    runtime: str,
    project: str,
) -> list[str]:
    """Confusable same-project episodes expected to grade 0 (never targets).

    Candidates are same-project, same-runtime episodes ranked by shared
    content words with the query; the top two are recorded as expected
    grade-0 hard negatives for the T14 judge to verify.
    """
    candidates: list[tuple[int, str]] = []
    for episode_id in index.by_project.get(project, []):
        if episode_id in target_ids:
            continue
        episode = index.episodes[episode_id]
        if episode["runtime"] != runtime:
            continue
        score = _shared_content_words(query, episode["prompt"] + " " + episode["summary"])
        candidates.append((score, episode_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [episode_id for _, episode_id in candidates[:2]]


def _query_record(
    index: _Index,
    class_name: str,
    query: str,
    anchors: list[str],
    target_ids: list[str],
    primary: dict[str, Any],
) -> dict[str, Any]:
    """One query record in the legacy-plus-new schema (fixed key order)."""
    target_set = sorted(set(target_ids))
    return {
        "id": "",
        "query": query,
        "class": class_name,
        "anchors": anchors,
        "anchor_mode": "all",
        "target_episode_ids": target_set,
        "target_session_ids": sorted({index.episodes[ep]["session_id"] for ep in target_set}),
        "subagent_only": False,
        "split": "",
        "runtime": primary["runtime"],
        "hard_negative_episode_ids": _hard_negatives(
            index, query, set(target_set), primary["runtime"], primary["project"]
        ),
        "grounding": [
            {
                "episode_id": episode_id,
                "project": index.episodes[episode_id]["project"],
                "runtime": index.episodes[episode_id]["runtime"],
                "prompt": index.episodes[episode_id]["prompt"][:300],
                "response": index.episodes[episode_id]["response"][:300],
            }
            for episode_id in target_set
        ],
        "notes": "",
    }


def _gen_exact_identifier(index: _Index, runtime: str, n: int) -> list[dict[str, Any]]:
    """Identifiers appearing verbatim only in their target episodes."""
    identifier_sets: dict[str, list[str]] = {}
    for episode_id, episode in index.episodes.items():
        for identifier in _identifiers(episode["prompt"] + " " + episode["response"]):
            identifier_sets.setdefault(identifier, []).append(episode_id)
    for identifier in identifier_sets:
        identifier_sets[identifier] = sorted(set(identifier_sets[identifier]))
    candidates = sorted(
        identifier_sets.items(), key=lambda item: (len(item[1]), -len(item[0]), item[0])
    )
    out: list[dict[str, Any]] = []
    for identifier, targets in candidates:
        if len(out) >= n:
            break
        runtime_targets = [ep for ep in targets if index.episodes[ep]["runtime"] == runtime]
        if not runtime_targets:
            continue
        primary = index.episodes[runtime_targets[0]]
        context = _context_words(primary["prompt"], identifier)
        query = " ".join([identifier] + context)
        out.append(_query_record(index, "exact-identifier", query, [identifier], targets, primary))
    if len(out) < n:
        raise ValueError(f"exact-identifier: only {len(out)} identifiers for {runtime}, need {n}")
    return out


def _context_words(text: str, exclude: str) -> list[str]:
    """Up to two most frequent content words from ``text`` besides ``exclude``."""
    counts: dict[str, int] = {}
    for word in re.split(r"[^a-zA-Z0-9]+", text.lower()):
        if word and word not in _STOPWORDS and word != exclude.lower():
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:2]]


def _gen_error_string(index: _Index, runtime: str, n: int) -> list[dict[str, Any]]:
    """Error-message substrings from target response text."""
    error_sets: dict[str, list[str]] = {}
    for episode_id, episode in index.episodes.items():
        for substring in _error_substrings(episode["response"]):
            error_sets.setdefault(substring, []).append(episode_id)
    for substring in error_sets:
        error_sets[substring] = sorted(set(error_sets[substring]))
    candidates = sorted(error_sets.items(), key=lambda item: (len(item[1]), -len(item[0]), item[0]))
    out: list[dict[str, Any]] = []
    for substring, targets in candidates:
        if len(out) >= n:
            break
        runtime_targets = [ep for ep in targets if index.episodes[ep]["runtime"] == runtime]
        if not runtime_targets:
            continue
        primary = index.episodes[runtime_targets[0]]
        out.append(_query_record(index, "error-string", substring, [substring], targets, primary))
    if len(out) < n:
        raise ValueError(f"error-string: only {len(out)} substrings for {runtime}, need {n}")
    return out


def _gen_conceptual(
    index: _Index,
    runtime: str,
    n: int,
    class_name: str,
    templates: tuple[str, ...],
    *,
    prefer: re.Pattern[str] | None,
) -> list[dict[str, Any]]:
    """Shared authoring for paraphrase / decision-rationale / tool-failure."""
    episodes = index.by_runtime[runtime]
    if prefer is not None:
        episodes = sorted(
            episodes,
            key=lambda episode_id: (
                0 if prefer.search(index.episodes[episode_id]["response"]) else 1,
                episode_id,
            ),
        )
    out: list[dict[str, Any]] = []
    for episode_id in episodes:
        if len(out) >= n:
            break
        episode = index.episodes[episode_id]
        anchor = _distinctive_anchor(episode["prompt"] + " " + episode["response"])
        topic = _topic_words(episode["prompt"], [anchor])
        query = _leakage_safe(templates, [topic], [anchor])
        out.append(_query_record(index, class_name, query, [anchor], [episode_id], episode))
    if len(out) < n:
        raise ValueError(f"{class_name}: only {len(out)} episodes for {runtime}, need {n}")
    return out


def _gen_multi_hop(index: _Index, runtime: str, n: int) -> list[dict[str, Any]]:
    """Queries grounded in >= 2 episodes spanning a narrative arc.

    Same-session pairs first (adjacent then non-adjacent), then same-project
    cross-session pairs, so the arc is strongest where available.
    """
    pairs: list[tuple[str, str]] = []
    for session_id in sorted(index.by_session):
        episode_ids = index.by_session[session_id]
        if index.episodes[episode_ids[0]]["runtime"] != runtime:
            continue
        for i in range(len(episode_ids)):
            for j in range(i + 1, len(episode_ids)):
                pairs.append((episode_ids[i], episode_ids[j]))
    for project in sorted(index.by_project):
        episode_ids = index.by_project[project]
        same_runtime = [ep for ep in episode_ids if index.episodes[ep]["runtime"] == runtime]
        other_runtime = [ep for ep in episode_ids if index.episodes[ep]["runtime"] != runtime]
        for first in same_runtime:
            for second in other_runtime:
                pairs.append((first, second))
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for first, second in pairs:
        key = (first, second) if first <= second else (second, first)
        if key not in seen:
            seen.add(key)
            ordered.append((first, second))

    out: list[dict[str, Any]] = []
    for first_id, second_id in ordered:
        if len(out) >= n:
            break
        first = index.episodes[first_id]
        second = index.episodes[second_id]
        anchors = [
            _distinctive_anchor(first["prompt"] + " " + first["response"]),
            _distinctive_anchor(second["prompt"] + " " + second["response"]),
        ]
        topic_a = _topic_words(first["prompt"], anchors)
        topic_b = _topic_words(second["prompt"], anchors)
        query = _leakage_safe(_MULTI_TEMPLATES, [topic_a, topic_b], anchors)
        out.append(_query_record(index, "multi-hop", query, anchors, [first_id, second_id], first))
    if len(out) < n:
        raise ValueError(f"multi-hop: only {len(out)} pairs for {runtime}, need {n}")
    return out


def _cross_topic(primary_text: str, other_text: str, project: str) -> str:
    """A topic string that shares real content words with BOTH target texts.

    Starts with the union topic (project slug words stripped via the anchor);
    if that topic does not actually appear in every target (the project slug
    itself occurs in almost no target text), falls back to the per-target
    distinctive anchors, then to the most frequent content words of both
    targets. Never returns an empty string: an empty topic would leave the
    query with no content anchor besides the project name, which retriever
    grading cannot answer.
    """

    def shares_both(candidate: str) -> bool:
        return bool(candidate) and (
            _shared_content_words(candidate, primary_text) >= 1
            and _shared_content_words(candidate, other_text) >= 1
        )

    union = primary_text + " " + other_text
    topic = _topic_words(union, anchors=[project], max_words=4)
    if shares_both(topic):
        return topic
    anchors = [_distinctive_anchor(primary_text), _distinctive_anchor(other_text)]
    topic = " ".join(anchor for anchor in anchors if anchor)
    if shares_both(topic):
        return topic
    words = _context_words(primary_text, exclude=project)
    words += _context_words(other_text, exclude=project)
    topic = " ".join(dict.fromkeys(words))
    if not topic:
        raise ValueError(f"could not author a content-bearing cross-runtime topic for {project}")
    return topic


def _single_topic(text: str, project: str) -> str:
    """Non-empty content topic for a single project-scoped target episode."""
    topic = _topic_words(text, anchors=[project], max_words=4)
    if topic:
        return topic
    anchor = _distinctive_anchor(text)
    if anchor:
        return anchor
    words = _context_words(text, exclude=project)
    topic = " ".join(words)
    if not topic:
        raise ValueError(f"could not author a content-bearing topic for {project}")
    return topic


def _gen_cross_runtime(index: _Index, runtime: str, n: int) -> list[dict[str, Any]]:
    """Cross-runtime pairs (same project, two runtimes) then project-scoped.

    Every primary-branch query names the project plus a topic computed from
    the union of both target episodes' text, so the query shares true content
    words with each target (not just the project slug). Query texts are
    deduplicated per runtime; the single-target fallback branch applies the
    same topic logic.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for episode_id in index.by_runtime[runtime]:
        if len(out) >= n:
            break
        episode = index.episodes[episode_id]
        others = [
            other
            for other in index.by_project.get(episode["project"], [])
            if index.episodes[other]["runtime"] != runtime
        ]
        if not others:
            continue
        other = others[0]
        other_episode = index.episodes[other]
        topic = _cross_topic(
            episode["prompt"] + " " + episode["response"],
            other_episode["prompt"] + " " + other_episode["response"],
            episode["project"],
        )
        query = (
            f"how was the {episode['project']} {topic} work handled across "
            f"{runtime} and {other_episode['runtime']}"
        )
        if query in seen:
            continue
        seen.add(query)
        out.append(
            _query_record(
                index,
                "cross-runtime/project-scoped",
                query,
                [episode["project"]],
                [episode_id, other],
                episode,
            )
        )
    if len(out) < n:
        for episode_id in index.by_runtime[runtime]:
            if len(out) >= n:
                break
            episode = index.episodes[episode_id]
            topic = _single_topic(episode["prompt"] + " " + episode["response"], episode["project"])
            query = f"what is the approach for the {episode['project']} {topic}"
            if query in seen:
                continue
            seen.add(query)
            out.append(
                _query_record(
                    index,
                    "cross-runtime/project-scoped",
                    query,
                    [episode["project"]],
                    [episode_id],
                    episode,
                )
            )
    if len(out) < n:
        raise ValueError(f"cross-runtime/project-scoped: only {len(out)} for {runtime}, need {n}")
    return out


_GENERATORS: dict[str, Any] = {
    "exact-identifier": _gen_exact_identifier,
    "error-string": _gen_error_string,
    "paraphrase": lambda index, runtime, n: _gen_conceptual(
        index, runtime, n, "paraphrase", _PARAPHRASE_TEMPLATES, prefer=None
    ),
    "decision-rationale": lambda index, runtime, n: _gen_conceptual(
        index, runtime, n, "decision-rationale", _DECISION_TEMPLATES, prefer=_DECISION_RE
    ),
    "tool-failure-recovery": lambda index, runtime, n: _gen_conceptual(
        index, runtime, n, "tool-failure-recovery", _TOOL_FAILURE_TEMPLATES, prefer=_FAILURE_RE
    ),
    "multi-hop": _gen_multi_hop,
    "cross-runtime/project-scoped": _gen_cross_runtime,
}


class _UnionFind:
    """Minimal path-compressing union-find over query-target episode ids."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b


def assign_splits(queries: list[dict[str, Any]], *, seed: int = SEED) -> list[dict[str, Any]]:
    """Assign the frozen 80/20 split, component-based and leakage-free.

    The split is part of the artifact: it is assigned once at generation time
    with a seeded RNG and never reshuffled afterwards. Queries are grouped
    into connected components of the query->target-episode graph (episodes
    co-targeted by a query are unioned), so every query targeting the same
    episode necessarily lands in the same split -- no episode can be a target
    of both a train and a holdout query.

    Components are elected to holdout greedily in ``(-query_count, root)``
    order with ``rand < HOLDOUT_FRACTION`` while keeping every affected
    class x runtime cell at >= MIN_PER_CELL-1 train queries (and within a
    per-cell holdout cap so the overall fraction stays near the 0.2 target);
    the remaining empty cells are force-filled where a feasible component
    exists. Deterministic under the same parquet + seed.
    """
    # ---- union-find over co-targeted episodes ----
    uf = _UnionFind()
    for query in queries:
        targets = query["target_episode_ids"]
        for episode_id in targets[1:]:
            uf.union(targets[0], episode_id)

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        root = uf.find(query["target_episode_ids"][0])
        components[root].append(query)

    # No query spans more than one component (checked defensively).
    for query in queries:
        targets = query["target_episode_ids"]
        root = uf.find(targets[0])
        for episode_id in targets[1:]:
            if uf.find(episode_id) != root:
                raise ValueError(f"query {query['id']} spans multiple components")

    comp_episodes = {
        root: {ep for q in qs for ep in q["target_episode_ids"]} for root, qs in components.items()
    }
    canonical = {root: min(eps) for root, eps in comp_episodes.items()}
    comp_query_count = {root: len(qs) for root, qs in components.items()}
    comp_cells = {
        root: Counter((q["class"], q["runtime"]) for q in qs) for root, qs in components.items()
    }
    cell_size = Counter((q["class"], q["runtime"]) for q in queries)
    train_floor = MIN_PER_CELL - 1  # every cell keeps at least this many train queries

    # ---- deterministic greedy election ----
    rng = random.Random(seed)
    holdout_cell: Counter[tuple[str, str]] = Counter()
    split_of: dict[str, str] = {}
    ordered = sorted(components, key=lambda root: (-comp_query_count[root], canonical[root]))
    for root in ordered:
        split = "train"
        if rng.random() < HOLDOUT_FRACTION:
            feasible = True
            for cell, count in comp_cells[root].items():
                if cell_size[cell] - (holdout_cell[cell] + count) < train_floor:
                    feasible = False
                    break
                cap = math.ceil(HOLDOUT_FRACTION * cell_size[cell])
                if holdout_cell[cell] + count > cap:
                    feasible = False
                    break
            if feasible:
                split = "holdout"
                for cell, count in comp_cells[root].items():
                    holdout_cell[cell] += count
        split_of[root] = split

    # ---- force at least one holdout per empty feasible cell ----
    for cell in sorted(cell_size):
        if holdout_cell[cell] > 0 or cell_size[cell] < MIN_PER_CELL:
            continue
        candidates = [
            (
                0 if all(c == cell for c in comp_cells[root]) else 1,
                comp_query_count[root],
                canonical[root],
                root,
            )
            for root in components
            if comp_cells[root].get(cell, 0) > 0
        ]
        candidates.sort()
        for _, _, _, root in candidates:
            feasible = all(
                cell_size[c] - (holdout_cell[c] + count) >= train_floor
                for c, count in comp_cells[root].items()
            )
            if not feasible:
                continue
            split_of[root] = "holdout"
            for c, count in comp_cells[root].items():
                holdout_cell[c] += count
            break

    # ---- apply ----
    for root, qs in components.items():
        split = split_of[root]
        for query in qs:
            query["split"] = split

    # ---- leakage invariant: no episode is a target of both splits ----
    train_episodes: set[str] = set()
    holdout_episodes: set[str] = set()
    for root, qs in components.items():
        bucket = holdout_episodes if split_of[root] == "holdout" else train_episodes
        for query in qs:
            bucket.update(query["target_episode_ids"])
    overlap = train_episodes & holdout_episodes
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"episodes targeted by both train and holdout queries: {sample}")
    return queries


def validate_queries(queries: list[dict[str, Any]], index: _Index) -> None:
    """Fail loudly when the authored query set is unanswerable or duplicated.

    Three checks guard the cross-runtime defect fixes: (1) every cross-runtime
    query text is pairwise distinct within its class (the degeneration this
    class previously produced); (2) every cross-runtime query shares at least
    one content word (after the project slug is subtracted) with each of its
    target episodes -- the project name alone appears in almost no target
    text, so a project-only query cannot be graded; (3) no ``(query,
    target_set)`` pair appears twice among cross-runtime queries.
    """
    cross_queries = [q for q in queries if q["class"] == "cross-runtime/project-scoped"]

    cross_texts: dict[str, list[str]] = defaultdict(list)
    for query in cross_queries:
        cross_texts[query["query"]].append(query["id"])
    duplicated = {text: ids for text, ids in cross_texts.items() if len(ids) > 1}
    if duplicated:
        raise ValueError(f"duplicate cross-runtime query text: {duplicated}")

    for query in cross_queries:
        project = query["anchors"][0] if query["anchors"] else ""
        query_without_project = query["query"].replace(project, "")
        for episode_id in query["target_episode_ids"]:
            episode = index.episodes[episode_id]
            text = episode["prompt"] + " " + episode["response"]
            if _shared_content_words(query_without_project, text) < 1:
                raise ValueError(
                    f"cross-runtime query {query['id']} shares no content with target {episode_id}"
                )

    seen_pairs: set[tuple[str, tuple[str, ...]]] = set()
    for query in cross_queries:
        pair = (query["query"], tuple(query["target_episode_ids"]))
        if pair in seen_pairs:
            raise ValueError(f"duplicate cross-runtime (query, target_set) pair: {query['id']}")
        seen_pairs.add(pair)


def generate_queries(parquet_path: str | Path, *, seed: int = SEED) -> list[dict[str, Any]]:
    """Author the full labelled query set from ``sessions.parquet``.

    Deterministic: the same parquet and seed always produce the same query
    list (and therefore the same ``queries.jsonl`` bytes).
    """
    source = Path(parquet_path)
    table = pq.read_table(source)
    if "runtime" not in table.column_names:
        raise ValueError(f"{source}: missing required 'runtime' column")
    rows = table.to_pylist()
    index = _build_index(rows)

    queries: list[dict[str, Any]] = []
    for class_name in SCENARIO_CLASSES:
        for runtime in RUNTIMES:
            target = PER_CLASS_RUNTIME_TARGETS[runtime]
            queries.extend(_GENERATORS[class_name](index, runtime, target))

    counters: dict[str, int] = {}
    for query in queries:
        prefix = CLASS_PREFIX[query["class"]]
        counters[prefix] = counters.get(prefix, 0) + 1
        query["id"] = f"{prefix}-{counters[prefix]:04d}"
    assign_splits(queries, seed=seed)
    validate_queries(queries, index)
    return queries


def write_queries(queries: list[dict[str, Any]], out_path: str | Path) -> None:
    """Write ``queries.jsonl`` with one JSON object per line (deterministic)."""
    lines = [json.dumps(query, ensure_ascii=False) for query in queries]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(__file__).with_name("sessions.parquet"),
        help="T6 sessions.parquet input",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("queries.jsonl"),
        help="queries.jsonl output path",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="holdout split seed")
    args = parser.parse_args(argv)

    queries = generate_queries(args.parquet, seed=args.seed)
    write_queries(queries, args.out)

    per_class: dict[str, int] = {}
    per_split: dict[str, int] = {}
    for query in queries:
        per_class[query["class"]] = per_class.get(query["class"], 0) + 1
        per_split[query["split"]] = per_split.get(query["split"], 0) + 1
    print(f"wrote {len(queries)} queries -> {args.out}")
    print(f"  splits: {dict(sorted(per_split.items()))}")
    for class_name in SCENARIO_CLASSES:
        print(f"  {class_name}: {per_class.get(class_name, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
