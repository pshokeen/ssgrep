"""Standardized evaluation runner: named arms, manifest preflight, versioned payloads.

The single sanctioned way to measure retrieval quality against a frozen
dataset. It composes the T1 metric engine (:mod:`eval.metrics`) and the T2
ranking instrumentation (:mod:`eval.ranking`) behind one deterministic CLI::

    python -m eval.run_eval --dataset v1 --arm default [--quick] [--json-out PATH]

Each run writes a timestamped payload to ``eval/results/current_<UTC-ts>.json``
by default; pass ``--json-out PATH`` to write to a fixed path instead.

Dataset contract (frozen by ``eval/datasetgen/freeze.py``; the runner refuses
anything that deviates)::

    <dataset-dir>/
        manifest.json        # {"version": str, "files": {path: sha256-hex}}
        queries.jsonl        # id, query, class, target_episode_ids,
                             #   subagent_only, split (train|holdout)
        qrels.tsv            # "query-id<TAB>corpus-id<TAB>grade", grades 0-3
        transcripts/         # native .jsonl, ingested via the real adapter

Preflight (fail loudly, never silently fall back): every manifest-listed file
must exist with a matching sha256, no unlisted file may exist in the dataset,
and every graded qrel episode must survive ingestion before a single query is
scored.

Arms come from :mod:`eval.arms`. ``brute_force_reference`` is the computed
reference arm (per-query exhaustive MaxSim) whose numbers land in the
payload's ``reference_arms`` block; ``--with-reference`` adds it to any
ordinary arm.

Payload::

    {result_schema_version, dataset_version, arm,
     summary:{overall, train, holdout, class:*}, per_query[],
     metric_definitions, gates{}, reference_arms{}, provenance{},
     index_size_bytes, latency_p50_ms, latency_p95_ms, build_seconds,
     token_vector_count | token_vector_count_note}

Everything is a deterministic function of its inputs except the provenance
date and the wall-clock latency readings; the determinism test freezes the
clock and asserts byte-identical payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

from eval import arms, harness, metrics, provenance, ranking
from eval.datasetgen.ingest import (
    CLAUDE_DIR,
    CODEX_DIR,
    OPENCODE_DB_NAME,
    PI_DIR,
    PRIME_AGENT_DIR,
    PRIME_AGENT_SESSIONS_DIR,
    TRANSCRIPTS_DIR,
)
from eval.metrics import METRIC_DEFINITIONS, RESULT_SCHEMA_VERSION
from ssgrep.indexing import indexer
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE, LanceStore
from ssgrep.store.paths import database_dir
from ssgrep.utilities.types import IndexStats

EVAL_DIR = Path(__file__).resolve().parent
DATASET_ROOT = EVAL_DIR / "dataset"

FINAL_LIMIT = 10
EPISODE_DEPTH = 100
CHUNK_DEPTH = 100
POOL_DEPTH = 800
QUICK_LIMIT = 5
#: Default depth of the metric run handed to ir_measures. The summary
#: r@50/r@100 are computed over this extended run (production top-10 page
#: plus the deep-prefetch pool), so with the default they are real deep
#: recall numbers rather than equal to r@10 by construction. ``--limit 10``
#: reproduces the legacy truncated behavior.
METRIC_LIMIT_DEFAULT = EPISODE_DEPTH
MANIFEST_NAME = "manifest.json"
QUERIES_NAME = "queries.jsonl"
QRELS_NAME = "qrels.tsv"

#: The five per-runtime env overrides the rebuild path points at the
#: dataset's emitter output (same names as eval/datasetgen/ingest.py).
_RUNTIME_ENV_OVERRIDES = (
    "SSGREP_TRANSCRIPT_DIRS",
    "SSGREP_CODEX_SESSIONS_DIR",
    "SSGREP_PI_SESSIONS_DIR",
    "SSGREP_PRIME_AGENT_SESSIONS_DIR",
    "SSGREP_OPENCODE_DB",
)

# Every wall-clock reading flows through this binding so tests can freeze
# time by rebinding it (and ``time.perf_counter``) without touching callers.
_perf_counter = time.perf_counter


#: One group's aggregate metrics inside a result payload. ``total=False``
#: because payloads carry extra keys (``n``, ...) and baselines may predate
#: a field; the gate logic reads via ``.get()``.
_SummarySlice = TypedDict(
    "_SummarySlice",
    {"ndcg@10": float | None, "rr@10": float | None, "r@50": float | None},
    total=False,
)


class _ResultPayload(TypedDict, total=False):
    """The gate-relevant subset of a result payload."""

    summary: dict[str, _SummarySlice]
    latency_p95_ms: float | None
    index_size_bytes: int | None


#: One query's reference-arm numbers (see ``_reference_record``).
_ReferenceRecord = TypedDict(
    "_ReferenceRecord",
    {
        "query_id": str,
        "episode_r@50": float | None,
        "episode_r@100": float | None,
        "chunk_r@50": float | None,
        "chunk_r@100": float | None,
        "engine_episode_r@100": float | None,
        "latency_ms": float | None,
    },
)


class PreflightError(RuntimeError):
    """A dataset or environment precondition failed; nothing was scored."""


class ManifestError(PreflightError):
    """The dataset manifest is missing, inconsistent, or tampered."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round3(value: float) -> float:
    return round(float(value), 3)


def dataset_files(dataset_dir: Path) -> list[Path]:
    """Every regular file under the dataset except the manifest, sorted."""
    if not dataset_dir.is_dir():
        raise ManifestError(f"dataset directory does not exist: {dataset_dir}")
    return sorted(
        path for path in dataset_dir.rglob("*") if path.is_file() and path.name != MANIFEST_NAME
    )


def preflight_manifest(dataset_dir: Path) -> dict:
    """Verify every manifest-listed file byte-for-byte; refuse loudly on mismatch."""
    manifest_path = dataset_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ManifestError(f"no {MANIFEST_NAME} in {dataset_dir}; cannot preflight")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ManifestError(f"{manifest_path} is not valid JSON ({error})") from None
    listed = manifest.get("files")
    if not isinstance(listed, dict) or not listed:
        raise ManifestError(
            f"{manifest_path} must declare a non-empty 'files' {{path: sha256}} map"
        )
    on_disk = {
        path.relative_to(dataset_dir).as_posix(): path for path in dataset_files(dataset_dir)
    }
    absent = [name for name in listed if name not in on_disk]
    if absent:
        raise ManifestError(
            f"manifest lists {len(absent)} file(s) missing from {dataset_dir}: "
            + ", ".join(sorted(absent))
        )
    extra = sorted(set(on_disk) - set(listed))
    if extra:
        raise ManifestError(
            f"{len(extra)} file(s) in {dataset_dir} are not in the manifest "
            f"(tampered artifact?): " + ", ".join(extra)
        )
    for name, path in on_disk.items():
        entry = listed[name]
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        if not isinstance(expected, str):
            raise ManifestError(f"{manifest_path}: entry {name!r} is not a sha256 hex string")
        digest = _sha256(path)
        if digest != expected:
            raise ManifestError(f"sha256 mismatch for {name}: expected {expected}, got {digest}")
    return dict(manifest)


def load_queries(dataset_dir: Path) -> list[dict]:
    """Load labeled queries, validating every runner-required field."""
    path = dataset_dir / QUERIES_NAME
    if not path.is_file():
        raise PreflightError(f"dataset has no {QUERIES_NAME}: {path}")
    rows: list[dict] = []
    required = {"id", "query", "class", "target_episode_ids", "split"}
    for line_number, line in enumerate(path.open(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PreflightError(f"{path}:{line_number} is not JSON ({error})") from None
        missing = required - set(row)
        if missing:
            raise PreflightError(f"{path}:{line_number} missing fields {sorted(missing)}")
        if row["split"] not in {"train", "holdout"}:
            raise PreflightError(
                f"{path}:{line_number} split={row['split']!r} (expected train|holdout)"
            )
        rows.append(row)
    if not rows:
        raise PreflightError(f"dataset has zero queries: {path}")
    return rows


def load_qrels(dataset_dir: Path) -> dict[str, dict[str, int]]:
    """Load graded qrels; 3-column and trec_eval 4-column (Q0) forms accepted."""
    path = dataset_dir / QRELS_NAME
    if not path.is_file():
        raise PreflightError(f"dataset has no {QRELS_NAME}: {path}")
    qrels: dict[str, dict[str, int]] = {}
    for line_number, line in enumerate(path.open(), start=1):
        fields = line.split()
        if not fields or fields[0].lower() in {"query-id", "qid"}:
            continue
        if len(fields) == 4 and fields[1] == "Q0":
            query_id, _, episode_id, grade = fields
        elif len(fields) == 3:
            query_id, episode_id, grade = fields
        else:
            raise PreflightError(
                f"{path}:{line_number}: expected 'query-id Q0 corpus-id grade' or "
                f"'query-id corpus-id grade' (got {len(fields)} columns)"
            )
        try:
            grade_value = int(grade)
        except ValueError:
            raise PreflightError(f"{path}:{line_number} grade {grade!r} is not an int") from None
        qrels.setdefault(query_id, {})[episode_id] = grade_value
    if not qrels:
        raise PreflightError(f"dataset has zero qrels: {path}")
    return qrels


def validate_qrel_targets(index_dir: Path, qrels: Mapping[str, Mapping[str, int]]) -> None:
    """Refuse to score when any graded episode did not survive ingestion."""
    with harness._private_data_dir(index_dir):
        repository = LanceStore()
        available = {
            str(row["episode_id"])
            for row in repository.rows(
                EPISODES_TABLE,
                columns=["episode_id"],
                limit=10_000_000,
            )
        }
    targets = {episode for docs in qrels.values() for episode, grade in docs.items() if grade > 0}
    missing = sorted(targets - available)
    if missing:
        raise PreflightError(
            f"{len(missing)}/{len(targets)} qrel target episode(s) are not in the index: "
            + ", ".join(missing[:10])
        )


def reconcile_qrel_targets(
    index_dir: Path, qrels: Mapping[str, Mapping[str, int]]
) -> tuple[dict[str, Mapping[str, int]], int]:
    """Map frozen-canonical episode ids onto the ingested index id space.

    The frozen artifact pins claude external-root ids using a sha1 of the
    *relative* transcript path (``transcripts/claude/<raw>.jsonl``), while the
    real adapter hashes the *absolute* path (T12/T14 known gap). Ids that are
    absent from the index are remaped 1:1 to the unique indexed episode with
    the same ``(stem-without-hash, :ep:N)``; ids that already exist, and all
    non-claude runtimes (their ids are path-independent), pass through
    unchanged. Ambiguous or unresolvable ids remain unmapped so the fail-loud
    ``validate_qrel_targets`` still flags them. Returns (qrels', n_remapped).
    """
    with harness._private_data_dir(index_dir):
        rows = LanceStore().rows(EPISODES_TABLE, columns=["episode_id"], limit=10_000_000)
    indexed = {str(row["episode_id"]) for row in rows}
    candidates: dict[tuple[str, str], list[str]] = {}
    for episode_id in indexed:
        prefix, _, n = episode_id.rpartition(":")
        candidates.setdefault((prefix.split("~")[0], n), []).append(episode_id)

    remap: dict[str, str] = {}
    for docs in qrels.values():
        for canonical in docs:
            if canonical in indexed:
                continue
            prefix, _, n = canonical.rpartition(":")
            choices = candidates.get((prefix.split("~")[0], n))
            if choices and len(choices) == 1:
                remap[canonical] = choices[0]
    if not remap:
        return dict(qrels), 0
    return (
        {
            qid: {remap.get(episode_id, episode_id): grade for episode_id, grade in docs.items()}
            for qid, docs in qrels.items()
        },
        len(remap),
    )


def _reconcile_and_validate_qrel_targets(
    index_dir: Path, qrels: Mapping[str, Mapping[str, int]]
) -> tuple[dict[str, Mapping[str, int]], int]:
    """Combined reconcile and validate: one scan of the episodes table.

    This avoids two separate scans of the episodes table by combining
    ``reconcile_qrel_targets`` and ``validate_qrel_targets`` into a single
    function.
    """
    with harness._private_data_dir(index_dir):
        rows = LanceStore().rows(EPISODES_TABLE, columns=["episode_id"], limit=10_000_000)
    indexed = {str(row["episode_id"]) for row in rows}

    # Reconcile: map frozen-canonical episode ids onto the ingested index id space
    candidates: dict[tuple[str, str], list[str]] = {}
    for episode_id in indexed:
        prefix, _, n = episode_id.rpartition(":")
        candidates.setdefault((prefix.split("~")[0], n), []).append(episode_id)

    remap: dict[str, str] = {}
    for docs in qrels.values():
        for canonical in docs:
            if canonical in indexed:
                continue
            prefix, _, n = canonical.rpartition(":")
            choices = candidates.get((prefix.split("~")[0], n))
            if choices and len(choices) == 1:
                remap[canonical] = choices[0]

    # Apply remap
    if remap:
        qrels = {
            qid: {remap.get(episode_id, episode_id): grade for episode_id, grade in docs.items()}
            for qid, docs in qrels.items()
        }

    # Validate: refuse to score when any graded episode did not survive ingestion
    targets = {episode for docs in qrels.values() for episode, grade in docs.items() if grade > 0}
    missing = sorted(targets - indexed)
    if missing:
        raise PreflightError(
            f"{len(missing)}/{len(targets)} qrel target episode(s) are not in the index: "
            + ", ".join(missing[:10])
        )

    return dict(qrels), len(remap)


def build_dataset_index(index_dir: Path, dataset_dir: Path, *, rebuild: bool) -> IndexStats:
    """Build (or adopt) the private index; returns its observability stats.

    The rebuild path points the FIVE per-runtime env overrides at the
    dataset's emitter output (the same layout ``eval/datasetgen/ingest.py``
    freezes under ``eval/dataset/v1/``)::

        transcripts/claude/            -> SSGREP_TRANSCRIPT_DIRS=native=<dir>
        transcripts/codex/             -> SSGREP_CODEX_SESSIONS_DIR
        transcripts/pi/                -> SSGREP_PI_SESSIONS_DIR
        transcripts/prime-agent/sessions/ -> SSGREP_PRIME_AGENT_SESSIONS_DIR
        <dataset>/opencode.db          -> SSGREP_OPENCODE_DB

    Pointing ``SSGREP_TRANSCRIPT_DIRS`` at the whole ``transcripts/`` tree
    would make the native adapter recursively ingest the codex/pi/prime-agent
    files as runtime ``native`` with wrong episode ids, so the frozen qrels
    (which reference ``codex:codex-0000:ep:0`` etc.) could never match. The
    native (claude) root is required; the other four overrides are set only
    for paths the dataset actually carries (a temp dataset may be native-only,
    while the frozen v1 carries all five). All overrides are restored after.

    The CocoIndex runtime allows exactly one open environment per process, so
    a second ``indexer.index`` pass in the same invocation (the old two-pass
    pattern) is impossible: the vector index is deferred while the chunks
    table is empty at ``initialize()`` time, so a rebuild explicitly creates
    it once the pass has finished (indices below the 256-token-vector
    training minimum are skipped the same way production defers them).

    With ``rebuild=False`` the index is trusted as-is: no ingestion, no
    compaction, no writes. This is what makes two runs over one caller-owned
    index byte-deterministic (lancedb compaction/index training are not) and
    is the workflow T16's baseline step uses (build once with T13's ingestion
    helper, then evaluate with ``--no-rebuild``).
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in _RUNTIME_ENV_OVERRIDES}
    try:
        with harness._private_data_dir(index_dir):
            repository = LanceStore()
            if not rebuild:
                if not repository.exists():
                    raise PreflightError(
                        f"no index at {index_dir}; build one first (drop --no-rebuild)"
                    )
                return _stats_from_repository(repository)
            transcripts = dataset_dir / TRANSCRIPTS_DIR
            if not transcripts.is_dir():
                raise PreflightError(f"dataset has no transcripts/ directory: {transcripts}")
            native_dir = transcripts / CLAUDE_DIR
            if not native_dir.is_dir():
                raise PreflightError(f"dataset has no transcripts/claude/ directory: {native_dir}")
            os.environ["SSGREP_TRANSCRIPT_DIRS"] = f"native={native_dir}"
            for env_name, path in (
                ("SSGREP_CODEX_SESSIONS_DIR", transcripts / CODEX_DIR),
                ("SSGREP_PI_SESSIONS_DIR", transcripts / PI_DIR),
                (
                    "SSGREP_PRIME_AGENT_SESSIONS_DIR",
                    transcripts / PRIME_AGENT_DIR / PRIME_AGENT_SESSIONS_DIR,
                ),
                ("SSGREP_OPENCODE_DB", dataset_dir / OPENCODE_DB_NAME),
            ):
                if path.exists():
                    os.environ[env_name] = str(path)
            stats = indexer.index(rebuild=True, allow_shrink=True, quiet=True)
            LanceStore().ensure_vector_index()
            return stats
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _stats_from_repository(repository: LanceStore) -> IndexStats:
    """Observability stats read from an existing index (no writes)."""
    return IndexStats(
        session_count=repository.count(SESSIONS_TABLE),
        episode_count=repository.count(EPISODES_TABLE),
        chunk_count=repository.count(CHUNKS_TABLE),
        index_size_bytes=harness._dir_size(database_dir()),
        last_index_time=None,
        model_id=repository.get_meta("model_id") or "",
        vector_dimension=int(repository.get_meta("vector_dimension") or 0),
        skipped_records=int(repository.get_meta("skipped_records") or 0),
        malformed_records=int(repository.get_meta("malformed_records") or 0),
        schema_version=repository.schema_version,
    )


def _chunk_episode_map(index_dir: Path) -> dict[str, str]:
    """One columnar scan: chunk_id -> episode_id across the whole index."""
    with harness._private_data_dir(index_dir):
        rows = LanceStore().rows(
            CHUNKS_TABLE,
            columns=["chunk_id", "episode_id"],
            limit=10_000_000,
        )
    return {str(row["chunk_id"]): str(row["episode_id"]) for row in rows}


def _relevant_episodes(qrels_query: Mapping[str, int]) -> set[str]:
    return {episode for episode, grade in qrels_query.items() if grade > 0}


def _episode_recall(
    ranked: Sequence[tuple[str, float]],
    qrels_query: Mapping[str, int],
    *,
    depth: int,
) -> float | None:
    """Fraction of relevant episodes in the top-``depth`` ranks (None if none relevant)."""
    relevant = _relevant_episodes(qrels_query)
    if not relevant:
        return None
    retrieved = {episode for episode, _score in ranked[:depth]}
    return len(relevant & retrieved) / len(relevant)


def _chunk_recall(
    ranked_chunk_ids: Sequence[str],
    chunk_episode: Mapping[str, str],
    qrels_query: Mapping[str, int],
    *,
    depth: int,
) -> float | None:
    """Chunk-level analog: a chunk is relevant iff its episode is grade>0."""
    relevant = _relevant_episodes(qrels_query)
    if not relevant:
        return None
    total = sum(1 for episode in chunk_episode.values() if episode in relevant)
    if not total:
        return None
    recalled = sum(
        1 for chunk_id in ranked_chunk_ids[:depth] if chunk_episode.get(chunk_id) in relevant
    )
    return recalled / total


def _brute_force_episodes(
    rows: Sequence[tuple[str, str, float]],
) -> list[tuple[str, float]]:
    """Episode ranking from a brute-force scan: best chunk MaxSim per episode.

    Deterministic (score desc, episode_id asc on ties). Production rollup
    bonuses are deliberately not applied here -- what they buy is precisely
    what the reference arm exists to measure.
    """
    best: dict[str, float] = {}
    for _chunk_id, episode_id, score in rows:
        previous = best.get(episode_id)
        if previous is None or score > previous:
            best[episode_id] = score
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def _extended_metric_run(
    final_episodes: Sequence[tuple[str, float]],
    prefetch_episodes: Sequence[tuple[str, float]],
    *,
    limit: int,
) -> list[tuple[str, float]]:
    """Extend the production top-``limit`` page with the deep-prefetch pool.

    The summary r@50/r@100 are computed over this extended run so they are
    real deep-recall numbers rather than equal to r@10 by construction. The
    production page keeps its order at the front; prefetch episodes not
    already present are appended in prefetch order (deduped by episode id)
    until ``limit`` entries or the pool is exhausted. Scores are reassigned
    order-preserving (``len(extended) - index``) so ir_measures sorts exactly
    by this sequence.
    """
    extended: list[tuple[str, float]] = list(final_episodes)
    seen = {episode_id for episode_id, _score in extended}
    for episode_id, _score in prefetch_episodes:
        if len(extended) >= limit:
            break
        if episode_id not in seen:
            seen.add(episode_id)
            extended.append((episode_id, 0.0))
    return [
        (episode_id, float(len(extended) - index))
        for index, (episode_id, _score) in enumerate(extended)
    ]


def _positive_mean(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None and value > 0]
    return sum(finite) / len(finite) if finite else None


def _mean_excluding_none(values: Sequence[float | None]) -> float | None:
    """Mean over non-None values; 0.0 is a legitimate value and is kept.

    Unlike ``_positive_mean`` (which filters ``value > 0``), recall means
    must include zeros: a query that retrieved nothing relevant is a real
    0.0, not a missing observation. Only ``None`` (no relevant episodes at
    all, the engine's convention) is excluded.
    """
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _groups_for(queries: Sequence[Mapping[str, object]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"overall": [], "train": [], "holdout": []}
    for row in queries:
        query_id = str(row["id"])
        groups["overall"].append(query_id)
        groups.setdefault(f"class:{row['class']}", []).append(query_id)
        groups[str(row["split"])].append(query_id)
    return groups


def _prefetch_recall_aggregates(
    per_query: Sequence[Mapping], groups: Mapping[str, Sequence[str]]
) -> dict[str, Mapping[str, float | None]]:
    """Group means of the per-query prefetch recall numbers.

    Each per-query record carries ``prefetch_episode`` (``r@10``/``r@50``/
    ``r@100`` over the 800-chunk pool rolled up to episodes) and
    ``prefetch_chunk`` (``chunk_r@100`` at depth 100). These are the raw
    deep-pool recall diagnostics; the summary's plain ``r@50``/``r@100``
    are computed over the metric run extended to ``metric_limit`` (default
    100), so with the default they are real deep-recall numbers rather than
    equal to ``r@10`` by construction.

    Queries whose value is ``None`` (no relevant episodes) are excluded from
    the mean, matching the engine's None convention; ``0.0`` is a legitimate
    value and is kept (see ``_mean_excluding_none``).
    """
    by_query = {str(record["query_id"]): record for record in per_query}
    keys = ("prefetch_r@10", "prefetch_r@50", "prefetch_r@100", "prefetch_chunk_r@100")
    aggregates: dict[str, Mapping[str, float | None]] = {}
    for group_name, query_ids in groups.items():
        values: dict[str, list[float | None]] = {key: [] for key in keys}
        for query_id in query_ids:
            record = by_query.get(str(query_id))
            if record is None:
                continue
            episode = record.get("prefetch_episode") or {}
            chunk = record.get("prefetch_chunk") or {}
            values["prefetch_r@10"].append(episode.get("r@10"))
            values["prefetch_r@50"].append(episode.get("r@50"))
            values["prefetch_r@100"].append(episode.get("r@100"))
            values["prefetch_chunk_r@100"].append(chunk.get("chunk_r@100"))
        aggregates[group_name] = {key: _mean_excluding_none(values[key]) for key in keys}
    return aggregates


def derive_gates(baseline: Mapping[str, object]) -> dict[str, float | None]:
    """The five gate constants derived from a baseline payload (T5 spec).

    Baseline payloads are JSON documents; the gates only read the documented
    ``_ResultPayload`` subset (summary / latency / size keys).
    """
    payload = cast(_ResultPayload, baseline)
    overall = payload["summary"]["overall"]
    floors = {
        "ndcg@10_floor": overall.get("ndcg@10"),
        "mrr_floor": overall.get("rr@10"),
        "r@50_floor": overall.get("r@50"),
    }
    gates: dict[str, float | None] = {
        name: None if value is None else float(value) - 0.01 for name, value in floors.items()
    }
    p95 = payload.get("latency_p95_ms")
    size = payload.get("index_size_bytes")
    gates["p95_ceiling"] = None if p95 is None else 1.2 * float(p95)
    gates["size_ceiling"] = None if size is None else 0.5 * float(size)
    return gates


def _check_pass(value: float | None, boundary: float | None, *, floor: bool) -> bool | None:
    """None == not comparable; recorded explicitly rather than guessed."""
    if value is None or boundary is None:
        return None
    if floor:
        return float(value) >= float(boundary)
    return float(value) <= float(boundary)


def check_gates(gates: Mapping[str, float | None], payload: Mapping[str, object]) -> dict:
    """Compare one run against gate constants; every comparison recorded."""
    result = cast(_ResultPayload, payload)
    overall = result["summary"]["overall"]
    checks = {
        "ndcg@10": {
            "value": overall.get("ndcg@10"),
            "floor": gates.get("ndcg@10_floor"),
            "pass": _check_pass(overall.get("ndcg@10"), gates.get("ndcg@10_floor"), floor=True),
        },
        "mrr": {
            "value": overall.get("rr@10"),
            "floor": gates.get("mrr_floor"),
            "pass": _check_pass(overall.get("rr@10"), gates.get("mrr_floor"), floor=True),
        },
        "r@50": {
            "value": overall.get("r@50"),
            "floor": gates.get("r@50_floor"),
            "pass": _check_pass(overall.get("r@50"), gates.get("r@50_floor"), floor=True),
        },
        "latency_p95_ms": {
            "value": result.get("latency_p95_ms"),
            "ceiling": gates.get("p95_ceiling"),
            "pass": _check_pass(
                result.get("latency_p95_ms"), gates.get("p95_ceiling"), floor=False
            ),
        },
        "size_bytes": {
            "value": result.get("index_size_bytes"),
            "ceiling": gates.get("size_ceiling"),
            "pass": _check_pass(
                result.get("index_size_bytes"), gates.get("size_ceiling"), floor=False
            ),
        },
    }
    return {
        "checks": checks,
        "all_pass": all(check["pass"] is not False for check in checks.values()),
    }


def resolve_dataset_dir(dataset_name: str | None, dataset_dir: Path | None) -> Path:
    """``--dataset NAME`` -> DATASET_ROOT/NAME; explicit ``--dataset-dir`` wins."""
    if dataset_dir is not None:
        return dataset_dir
    if dataset_name is None:
        raise PreflightError("pass either --dataset NAME or --dataset-dir PATH")
    return DATASET_ROOT / dataset_name


def _reference_record(
    *,
    query_id: str,
    brute_episodes: Sequence[tuple[str, float]],
    brute_rows: Sequence[tuple[str, str, float]],
    chunk_episode: Mapping[str, str],
    qrels_query: Mapping[str, int],
    brute_ms: float,
    engine_episode_r100: float | None,
) -> _ReferenceRecord:
    """One query's reference-arm numbers (episode + chunk recall ceilings)."""
    return {
        "query_id": query_id,
        "episode_r@50": _episode_recall(brute_episodes, qrels_query, depth=50),
        "episode_r@100": _episode_recall(brute_episodes, qrels_query, depth=EPISODE_DEPTH),
        "chunk_r@50": _chunk_recall(
            [chunk_id for chunk_id, _episode, _score in brute_rows[:50]],
            chunk_episode,
            qrels_query,
            depth=50,
        ),
        "chunk_r@100": _chunk_recall(
            [chunk_id for chunk_id, _episode, _score in brute_rows[:CHUNK_DEPTH]],
            chunk_episode,
            qrels_query,
            depth=CHUNK_DEPTH,
        ),
        "engine_episode_r@100": engine_episode_r100,
        "latency_ms": _round3(brute_ms),
    }


def _reference_block(reference_rows: Sequence[_ReferenceRecord]) -> dict:
    """Aggregate per-query reference records into the payload's block."""
    ratios = [
        float(row["engine_episode_r@100"]) / float(row["episode_r@100"])
        for row in reference_rows
        if row["engine_episode_r@100"] is not None
        and row["episode_r@100"] is not None
        and float(row["episode_r@100"]) > 0
    ]
    return {
        "arm": "brute_force_reference",
        "episode_r@50": _positive_mean([row["episode_r@50"] for row in reference_rows]),
        "episode_r@100": _positive_mean([row["episode_r@100"] for row in reference_rows]),
        "chunk_r@50": _positive_mean([row["chunk_r@50"] for row in reference_rows]),
        "chunk_r@100": _positive_mean([row["chunk_r@100"] for row in reference_rows]),
        "ann_recall100_ratio": _positive_mean(ratios),
        "per_query": [{**row} for row in reference_rows],
    }


def _process_single_query(
    row: dict,
    *,
    index_dir: Path,
    qrels: Mapping[str, Mapping[str, int]],
    chunk_episode: Mapping[str, str],
    arm: arms.Arm,
    with_reference: bool,
    limit: int,
) -> dict:
    """Process a single query: ranking, metrics, and reference arm if needed.

    Returns a dict with keys: run_entry, per_query_entry, reference_row,
    final_latency. Designed to be called in parallel via ThreadPoolExecutor.
    """
    from ssgrep import search as search_module

    query_id = str(row["id"])
    query_text = str(row["query"])
    qrels_query = qrels.get(query_id, {})

    brute_rows: list[tuple[str, str, float]] | None = None
    brute_episodes: list[tuple[str, float]] = []
    brute_ms = 0.0
    if with_reference:
        brute_rows, brute_ms = ranking.brute_force_ranking(index_dir, query_text)
        brute_episodes = _brute_force_episodes(brute_rows)

    if arm.computed and brute_rows is not None:
        final_episodes = brute_episodes[:FINAL_LIMIT]
        prefetch_episodes = brute_episodes[:EPISODE_DEPTH]
        chunk_ids = [chunk_id for chunk_id, _episode, _score in brute_rows[:CHUNK_DEPTH]]
        final_ms = prefetch_ms = chunk_ms = brute_ms
    else:
        # Pre-compute query matrix once to avoid duplicate embedding
        query_matrix = search_module._query_matrix(query_text)
        # Combined final + prefetch with shared context, separate pools
        final_episodes, prefetch_episodes, chunk_ranks, final_ms, prefetch_ms = (
            ranking.combined_final_and_prefetch(
                index_dir,
                query_text,
                query_matrix,
                final_limit=FINAL_LIMIT,
                pool_depth=POOL_DEPTH,
                chunk_depth=CHUNK_DEPTH,
            )
        )
        chunk_ids = [chunk_id for chunk_id, _episode, _score in chunk_ranks]
        chunk_ms = prefetch_ms  # Same search, same timing

    if arm.computed:
        run_entry = {episode_id: float(score) for episode_id, score in brute_episodes[:limit]}
    else:
        metric_run_episodes = _extended_metric_run(final_episodes, prefetch_episodes, limit=limit)
        run_entry = {episode_id: float(score) for episode_id, score in metric_run_episodes}

    per_query_entry = {
        "query_id": query_id,
        "query": query_text,
        "class": row.get("class"),
        "split": row.get("split"),
        "final": {
            "ranking": [
                {"episode_id": episode_id, "score": score} for episode_id, score in final_episodes
            ],
            "latency_ms": _round3(final_ms),
        },
        "prefetch_episode": {
            "depth": EPISODE_DEPTH,
            "pool_depth": POOL_DEPTH,
            "ranking": [episode_id for episode_id, _score in prefetch_episodes],
            "r@10": _episode_recall(prefetch_episodes, qrels_query, depth=10),
            "r@50": _episode_recall(prefetch_episodes, qrels_query, depth=50),
            "r@100": _episode_recall(prefetch_episodes, qrels_query, depth=EPISODE_DEPTH),
            "latency_ms": _round3(prefetch_ms),
        },
        "prefetch_chunk": {
            "depth": CHUNK_DEPTH,
            "ranking": chunk_ids,
            "chunk_r@10": _chunk_recall(chunk_ids, chunk_episode, qrels_query, depth=10),
            "chunk_r@50": _chunk_recall(chunk_ids, chunk_episode, qrels_query, depth=50),
            "chunk_r@100": _chunk_recall(chunk_ids, chunk_episode, qrels_query, depth=CHUNK_DEPTH),
            "latency_ms": _round3(chunk_ms),
        },
    }

    reference_row = None
    if with_reference:
        reference_row = _reference_record(
            query_id=query_id,
            brute_episodes=brute_episodes,
            brute_rows=brute_rows or [],
            chunk_episode=chunk_episode,
            qrels_query=qrels_query,
            brute_ms=brute_ms,
            engine_episode_r100=(
                _episode_recall(prefetch_episodes, qrels_query, depth=EPISODE_DEPTH)
                if not arm.computed
                else _episode_recall(brute_episodes, qrels_query, depth=EPISODE_DEPTH)
            ),
        )

    return {
        "query_id": query_id,
        "run_entry": run_entry,
        "per_query_entry": per_query_entry,
        "reference_row": reference_row,
        "final_latency": final_ms,
    }


def _eval_worker_count(n_queries: int) -> int:
    """Size the per-query worker pool.

    ``SSGREP_EVAL_WORKERS`` caps it explicitly -- CI's emulated Linux
    container has ~8 GB and every worker imports torch, so a 16-way pool
    OOM-kills a worker there (``BrokenProcessPool``) while the same suite
    passes on the macOS legs. Without the override the pool is bounded by
    the host CPU count, and never exceeds the number of queries or drops
    below one.
    """
    raw = os.environ.get("SSGREP_EVAL_WORKERS", "").strip()
    cap = int(raw) if raw.isdigit() and int(raw) > 0 else min(16, os.cpu_count() or 1)
    return max(1, min(cap, n_queries))


def run_eval(
    dataset_dir: Path,
    *,
    arm_name: str = "default",
    index_dir: Path | None = None,
    rebuild: bool = True,
    quick: bool = False,
    with_reference: bool = False,
    baseline_path: Path | None = None,
    date: str | None = None,
    limit: int = METRIC_LIMIT_DEFAULT,
    parallel: bool = True,
) -> dict:
    """Run one full evaluation; returns the versioned payload dict.

    Ingestion, ranking, measurement, and provenance all run inside the
    private-data context with the arm's knob overrides applied, so the
    provenance tuning constants record exactly what the measured code saw.
    """
    limit = max(1, min(int(limit), POOL_DEPTH))
    manifest = preflight_manifest(dataset_dir)
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir)
    evaluated = queries[:QUICK_LIMIT] if quick else queries
    if not evaluated:
        raise PreflightError(f"{dataset_dir} has no queries to evaluate")
    arm = arms.resolve(arm_name)
    with_reference = with_reference or arm.computed
    index_dir = index_dir or Path(tempfile.mkdtemp(prefix="ssgrep-eval-"))

    build_started = _perf_counter()
    with harness._private_data_dir(index_dir), arms.arm_env(arm):
        stats = build_dataset_index(index_dir, dataset_dir, rebuild=rebuild)
        build_seconds = _perf_counter() - build_started
        # The frozen artifact's canonical id space (relative-path claude
        # hashes) differs from the ingested index space (absolute-path
        # hashes); bridge the gap before the fail-loud target check.
        # Combined reconcile and validate: one scan of the episodes table.
        qrels, n_remapped = _reconcile_and_validate_qrel_targets(index_dir, qrels)
        if n_remapped:
            print(
                f"ssgrep-eval: reconciled {n_remapped} canonical episode id(s) "
                "onto the ingested index id space",
                file=sys.stderr,
            )
        chunk_episode = _chunk_episode_map(index_dir)

        # One discarded warm-up: model load / index open land outside the
        # timed samples (harness precedent).
        if evaluated:
            ranking.final_ranking(index_dir, evaluated[0]["query"], limit=FINAL_LIMIT)

        # Set SSGREP_DATA_DIR once for all threads to use
        os.environ["SSGREP_DATA_DIR"] = str(index_dir)

        run: dict[str, dict[str, float]] = {}
        reference_rows: list[_ReferenceRecord] = []
        final_latencies: list[float] = []
        per_query: list[dict] = []

        if parallel:
            # Process queries in parallel using ProcessPoolExecutor
            # Each process has its own environment, avoiding GIL and contention
            max_workers = _eval_worker_count(len(evaluated))
            results_by_id: dict[str, dict] = {}

            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_single_query,
                        row,
                        index_dir=index_dir,
                        qrels=qrels,
                        chunk_episode=chunk_episode,
                        arm=arm,
                        with_reference=with_reference,
                        limit=limit,
                    ): row
                    for row in evaluated
                }
                for future in as_completed(futures):
                    result = future.result()
                    results_by_id[result["query_id"]] = result

            # Collect results in the original query order
            for row in evaluated:
                query_id = str(row["id"])
                result = results_by_id[query_id]
                run[query_id] = result["run_entry"]
                final_latencies.append(result["final_latency"])
                per_query.append(result["per_query_entry"])
                if result["reference_row"] is not None:
                    reference_rows.append(result["reference_row"])
        else:
            # Sequential processing (for determinism tests)
            for row in evaluated:
                result = _process_single_query(
                    row,
                    index_dir=index_dir,
                    qrels=qrels,
                    chunk_episode=chunk_episode,
                    arm=arm,
                    with_reference=with_reference,
                    limit=limit,
                )
                run[result["query_id"]] = result["run_entry"]
                final_latencies.append(result["final_latency"])
                per_query.append(result["per_query_entry"])
                if result["reference_row"] is not None:
                    reference_rows.append(result["reference_row"])

        report = metrics.evaluate_run(qrels, run)
        metrics_by_query: dict[str, dict[str, object]] = {}
        for metric_row in report.per_query:
            metrics_by_query.setdefault(str(metric_row["query_id"]), {})[
                str(metric_row["metric"])
            ] = metric_row["value"]
        for record in per_query:
            record["metrics"] = metrics_by_query.get(str(record["query_id"]), {})

        groups = _groups_for(evaluated)
        summary = metrics.group_summary(qrels, run, groups)
        for group_name, aggregate in summary.items():
            aggregate["n"] = len(groups[group_name])

        # The summary r@50/r@100 above are computed over the metric run
        # extended to ``limit`` (production top-10 page plus the deep-pool
        # episodes), so with the default they are real deep-recall numbers.
        # The prefetch_* keys surface the raw deep-pool recall at the
        # 100/800 seam as additive diagnostics (existing keys untouched).
        prefetch_agg = _prefetch_recall_aggregates(per_query, groups)
        for group_name, aggregate in summary.items():
            aggregate.update(prefetch_agg[group_name])

        with harness._private_data_dir(index_dir):
            db_path = database_dir()
        ops = {
            "index_size_bytes": harness._dir_size(db_path),
            "latency_p50_ms": _round3(harness._percentile(final_latencies, 0.50)),
            "latency_p95_ms": _round3(harness._percentile(final_latencies, 0.95)),
            "build_seconds": round(build_seconds, 3),
        }
        ops.update(harness._token_vector_count(db_path))

        provenance_block = provenance.build_provenance(
            date=date or datetime.now(UTC).isoformat(timespec="seconds"),
            task=("run_eval reference arm" if arm.computed else f"run_eval arm={arm.name}"),
            label=f"dataset {dataset_dir.name} / arm {arm.name}",
            stats=stats,
            db_path=db_path,
        )

    payload: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dataset_version": str(manifest.get("version", dataset_dir.name)),
        "arm": arm.name,
        "metric_limit": limit,
        "summary": summary,
        "per_query": per_query,
        "metric_definitions": {
            **METRIC_DEFINITIONS,
            "prefetch": {
                "formulation": (
                    "mean per-query prefetch recall over the group; "
                    "prefetch_r@k is the episode rollup of the 800-chunk "
                    "pool at depths 10/50/100, prefetch_chunk_r@100 is "
                    "chunk-level recall at depth 100"
                ),
                "cutoffs": [10, 50, 100],
                "pool_depth": POOL_DEPTH,
                "episode_depth": EPISODE_DEPTH,
                "chunk_depth": CHUNK_DEPTH,
                "notes": (
                    "The summary r@50/r@100 are computed over the metric run "
                    "extended to the configured depth (metric_limit, default "
                    "100): the production top-10 page plus the remaining "
                    "deep-pool episodes (deduped, prefetch order). With the "
                    "default they are real deep-recall numbers, not equal to "
                    "r@10 by construction; passing --limit 10 reproduces the "
                    "legacy truncated behavior. The prefetch_* keys carry "
                    "the raw deep-pool recall at the 100/800 seam. Queries "
                    "with no relevant episodes (None) are excluded from the "
                    "mean; 0.0 is kept. Per-query detail lives under "
                    "per_query[].prefetch_episode / per_query[].prefetch_chunk."
                ),
            },
        },
        "gates": {},
        "reference_arms": {},
        "provenance": provenance_block,
        **ops,
    }
    if baseline_path is not None:
        try:
            baseline = json.loads(baseline_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise PreflightError(f"cannot read baseline {baseline_path}: {error}") from error
        derived = derive_gates(baseline)
        payload["gates"] = {
            "from_baseline": str(baseline_path),
            **derived,
            "check": check_gates(derived, payload),
        }
    if reference_rows:
        payload["reference_arms"] = {"brute_force": _reference_block(reference_rows)}
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (0 on success)."""
    parser = argparse.ArgumentParser(
        prog="eval.run_eval",
        description="Run the standardized retrieval evaluation against a frozen dataset.",
    )
    parser.add_argument("--dataset", help="dataset name under eval/dataset (e.g. v1)")
    parser.add_argument("--dataset-dir", type=Path, help="explicit dataset directory")
    parser.add_argument("--arm", default="default", help="named arm (eval/arms.py)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"evaluate only the first {QUICK_LIMIT} queries",
    )
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="also run the brute-force reference arm",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="write the versioned payload here (default: eval/results/current_<UTC-ts>.json)",
    )
    parser.add_argument("--baseline", type=Path, help="baseline payload to derive gates from")
    parser.add_argument("--index-dir", type=Path, help="private index directory (default: temp)")
    parser.add_argument("--no-rebuild", action="store_true", help="reuse an existing --index-dir")
    parser.add_argument(
        "--limit",
        type=int,
        default=METRIC_LIMIT_DEFAULT,
        help=(
            "metric run depth (default 100; r@50/r@100 are computed at this "
            "depth; --limit 10 reproduces the legacy truncated behavior)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        dataset_dir = resolve_dataset_dir(args.dataset, args.dataset_dir)
        payload = run_eval(
            dataset_dir,
            arm_name=args.arm,
            index_dir=args.index_dir,
            rebuild=not args.no_rebuild,
            quick=args.quick,
            with_reference=args.with_reference,
            baseline_path=args.baseline,
            limit=args.limit,
        )
    except ValueError as error:
        print(f"usage error: {error}", file=sys.stderr)
        return 2
    except PreflightError as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1

    overall = payload["summary"]["overall"]
    print(f"arm={payload['arm']} dataset={payload['dataset_version']} queries={overall.get('n')}")
    print(
        f"  ndcg@10={overall.get('ndcg@10')}  rr@10={overall.get('rr@10')}  "
        f"r@50={overall.get('r@50')}"
    )
    print(
        f"  index_size_bytes={payload.get('index_size_bytes')}  "
        f"latency_p95_ms={payload.get('latency_p95_ms')}  "
        f"build_seconds={payload.get('build_seconds')}"
    )
    if payload["gates"]:
        print(f"  gates: all_pass={payload['gates']['check']['all_pass']}")
    if payload["reference_arms"]:
        reference = payload["reference_arms"]["brute_force"]
        print(
            f"  reference: episode_r@100={reference.get('episode_r@100')}  "
            f"ann_recall100_ratio={reference.get('ann_recall100_ratio')}"
        )
    json_out = args.json_out or (
        Path("eval/results") / f"current_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
