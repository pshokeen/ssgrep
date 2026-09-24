"""Live-corpus distribution profiler for the synthetic benchmark.

Reads — never writes — a copy of the live ssgrep data root and emits
``profile.json`` containing only *aggregates*: counts, histograms, length
distributions, and frequency tables. By design no field in the output is
transcript text: only lengths, counts, and tokens survive the privacy
boundary (plan decision D2). The profiled shape is the stratification target
for the NeMo synthetic generation workflow (plan T6).

The profiler connects with lancedb's disconnected snapshot semantics
(``read_consistency_interval=0``, the same read-only pattern as
``LanceStore._connect``) and uses ``LanceStore.rows`` for every read. It never
calls ``indexer.index()`` and never writes into the data root.

Usage::

    uv run python -m eval.datasetgen.profile_live --data-root <copy> \\
        --out eval/datasetgen/profile.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE, LanceStore

#: Every runtime ssgrep indexes, in a stable order. The synthetic corpus is
#: stratified to all five, so the profile always carries all five keys.
RUNTIMES = ("claude", "codex", "opencode", "pi", "prime-agent")

#: Reference census measured by the retrieval-eval-overhaul plan on 2026-08-22
#: (plan: 1161 / 2913 / 24900 / ~449MB / schema v6; opencode=950,
#: prime-agent=98, codex=83, pi=26, claude=4). Embedded so a later profile can
#: detect drift between the live corpus and the census the benchmark froze.
SOURCE_CENSUS: dict[str, Any] = {
    "sessions": 1161,
    "episodes": 2913,
    "chunks": 24900,
    "index_size_approx_mb": 449,
    "schema_version": 6,
    "runtimes": {
        "opencode": 950,
        "prime-agent": 98,
        "codex": 83,
        "pi": 26,
        "claude": 4,
    },
}

#: Top-N cutoff for frequency tables so profile.json stays small and bounded.
TOP_N = 25


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile, matching ``eval/harness`` semantics."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _length_stats(values: Iterable[Any]) -> dict[str, float]:
    """Character-length distribution over an iterable of sized values."""
    lengths = [len(value) for value in values]
    return _numeric_stats(lengths)


def _numeric_stats(values: Iterable[int | float]) -> dict[str, float]:
    """p10/p50/p90/max over a numeric sample (counts, durations, ...)."""
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "p10": _percentile(numbers, 0.10),
        "p50": _percentile(numbers, 0.50),
        "p90": _percentile(numbers, 0.90),
        "max": max(numbers) if numbers else 0.0,
    }


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _frequency_table(items: Iterable[str]) -> dict[str, int]:
    """Collapse newline-separated names into a bounded frequency table.

    ``files_touched``/``tool_names`` are joined with ``"\\n"`` (see
    ``src/ssgrep/pipeline/rows.py``), so tokenizing on newline is the exact
    inverse of what the pipeline wrote.
    """
    counts: Counter[str] = Counter()
    for item in items:
        for token in item.split("\n"):
            if token:
                counts[token] += 1
    return dict(counts.most_common(TOP_N))


def _iso(value: Any) -> str | None:
    """Normalize a stored datetime to a UTC ISO timestamp."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


TITLE_PATTERNS = (
    "contains_backtick",
    "ends_with_question",
    "ends_with_exclamation",
    "all_uppercase",
    "starts_with_symbol",
    "subagent_marker",
)


def _title_pattern_counts(titles: Iterable[str]) -> dict[str, int]:
    counts = {pattern: 0 for pattern in TITLE_PATTERNS}
    for title in titles:
        if "`" in title:
            counts["contains_backtick"] += 1
        if title.endswith("?"):
            counts["ends_with_question"] += 1
        if title.endswith("!"):
            counts["ends_with_exclamation"] += 1
        if title and title.isupper():
            counts["all_uppercase"] += 1
        if title and not title[0].isalnum():
            counts["starts_with_symbol"] += 1
        if "subagent" in title.lower():
            counts["subagent_marker"] += 1
    return counts


def build_profile(store: LanceStore) -> dict[str, Any]:
    """Read every data table once and summarize the live distribution."""
    session_rows = store.rows(SESSIONS_TABLE, columns=["session_id", "runtime"])
    episode_rows = store.rows(
        EPISODES_TABLE,
        columns=[
            "session_id",
            "runtime",
            "is_subagent",
            "project",
            "title",
            "timestamp",
            "files_touched",
            "tool_names",
        ],
    )
    chunk_rows = store.rows(
        CHUNKS_TABLE,
        columns=["episode_id", "runtime", "content_type", "is_subagent", "text"],
    )

    sessions_per_runtime: Counter[str] = Counter(row["runtime"] for row in session_rows)
    episodes_per_runtime: Counter[str] = Counter(row["runtime"] for row in episode_rows)
    chunks_per_runtime: Counter[str] = Counter(row["runtime"] for row in chunk_rows)

    episodes_per_session: Counter[str] = Counter(row["session_id"] for row in episode_rows)
    chunks_per_episode: Counter[str] = Counter(row["episode_id"] for row in chunk_rows)
    content_type: Counter[str] = Counter(row["content_type"] for row in chunk_rows)

    subagent_episodes = sum(bool(row["is_subagent"]) for row in episode_rows)
    subagent_chunks = sum(bool(row["is_subagent"]) for row in chunk_rows)

    episodes_per_project: Counter[str] = Counter(row["project"] for row in episode_rows)

    titles = [str(row["title"]) for row in episode_rows]

    timestamps = [
        timestamp for timestamp in (_iso(row["timestamp"]) for row in episode_rows) if timestamp
    ]

    return {
        "runtimes": {
            runtime: {
                "sessions": sessions_per_runtime.get(runtime, 0),
                "episodes": episodes_per_runtime.get(runtime, 0),
                "chunks": chunks_per_runtime.get(runtime, 0),
            }
            for runtime in RUNTIMES
        },
        "census": {
            "sessions": len(session_rows),
            "episodes": len(episode_rows),
            "chunks": len(chunk_rows),
        },
        "episodes_per_session": {
            "histogram": _histogram(count for count in episodes_per_session.values()),
            "stats": _numeric_stats(episodes_per_session.values()),
        },
        "chunks_per_episode": {
            "histogram": _histogram(count for count in chunks_per_episode.values()),
            "stats": _numeric_stats(chunks_per_episode.values()),
        },
        "content_type": dict(sorted(content_type.items())),
        "text_length_chars": {
            "prompt": _length_stats(
                row["text"] for row in chunk_rows if row["content_type"] == "prompt"
            ),
            "response": _length_stats(
                row["text"] for row in chunk_rows if row["content_type"] == "response"
            ),
        },
        "subagent": {
            "episode_share": (subagent_episodes / len(episode_rows) if episode_rows else 0.0),
            "chunk_share": subagent_chunks / len(chunk_rows) if chunk_rows else 0.0,
        },
        "projects": {
            "count": len(episodes_per_project),
            "episodes_per_project": {
                "histogram": _histogram(count for count in episodes_per_project.values()),
                "stats": _numeric_stats(episodes_per_project.values()),
            },
        },
        "title": {
            "length_chars": _length_stats(titles),
            "pattern_counts": _title_pattern_counts(titles),
        },
        "files_touched": _frequency_table(row["files_touched"] for row in episode_rows),
        "tool_names": _frequency_table(row["tool_names"] for row in episode_rows),
        "timestamp_span": {
            "earliest": min(timestamps) if timestamps else None,
            "latest": max(timestamps) if timestamps else None,
        },
    }


@contextmanager
def _point_data_dir(path: Path) -> Any:
    """Temporarily point ``SSGREP_DATA_DIR`` at ``path`` (restored after)."""
    previous = os.environ.get("SSGREP_DATA_DIR")
    os.environ["SSGREP_DATA_DIR"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SSGREP_DATA_DIR", None)
        else:
            os.environ["SSGREP_DATA_DIR"] = previous


def _default_data_root() -> Path:
    from ssgrep.store.paths import data_dir

    return data_dir()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile live transcript distributions for the synthetic benchmark"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root to analyze (use a COPY); default: the live data root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/datasetgen/profile.json"),
        help="Output JSON path",
    )
    args = parser.parse_args(argv)

    root = args.data_root if args.data_root is not None else _default_data_root()
    if not (root / "lancedb").is_dir():
        print(f"error: no lancedb/ under {root}", file=sys.stderr)
        return 2

    with _point_data_dir(root):
        store = LanceStore()
        payload = build_profile(store)
    payload["source_census"] = SOURCE_CENSUS
    payload["profile_schema_version"] = 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
