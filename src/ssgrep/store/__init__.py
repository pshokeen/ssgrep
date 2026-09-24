"""Lazy access to ssgrep's single local LanceDB database."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import lancedb
import numpy as np

from ssgrep.indexing.embed import DIMENSION
from ssgrep.store.paths import database_dir, ensure_data_dir
from ssgrep.store.schema import (
    COMPAT_TABLES,
    EMBEDDING_CONFIG,
    PRIMARY_KEYS,
    SCHEMA_VERSION,
    TABLE_SCHEMAS,
)
from ssgrep.utilities.limits import raise_nofile_limit

CHUNKS_TABLE = "chunks"
EPISODES_TABLE = "episodes"
SESSIONS_TABLE = "sessions"
CURSORS_TABLE = "cursors"
META_TABLE = "metadata"
CWD_CACHE_TABLE = "cwd_cache"
SOURCES_TABLE = "sources"

#: Single-vector prefilter column (mean of each chunk's token vectors). Its
#: IVF-PQ index powers stage 1 of ``SSGREP_TWO_STAGE`` search.
PROXY_COLUMN = "proxy_vector"
#: Candidate floor for two-stage search: mean-vector proxies need deeper
#: candidate lists than exact scoring would, so stage 1 never selects fewer.
TWO_STAGE_MIN_CANDIDATES = 200

#: Every table compacted once after a successful full reconciliation. Lance's
#: ``optimize()`` runs compaction + cleanup (+ index refresh); superseded row
#: versions and their data files are pruned immediately (see
#: ``POST_COMPACT_CLEANUP_RETENTION``), so a post-reconcile ``status`` reports
#: the settled on-disk size instead of retaining a second copy of every
#: fragment for the library's default 7-day retention.
#:
#: The bookkeeping tables (``metadata``, ``cursors``, ``cwd_cache``,
#: ``sources``) are included too: each ``set_meta``/``upsert`` appends a
#: fragment and nothing else ever rewrites them, so they fragment without
#: bound — a live ``metadata`` table reached 329 fragments (full-scanned twice
#: per search), which bloats disk and multiplies the per-scan open file
#: descriptors that exhaust processes running under a low ``RLIMIT_NOFILE``.
OPTIMIZE_TABLES = (
    CHUNKS_TABLE,
    EPISODES_TABLE,
    SESSIONS_TABLE,
    META_TABLE,
    CURSORS_TABLE,
    CWD_CACHE_TABLE,
    SOURCES_TABLE,
)

#: Cleanup cutoff passed to ``optimize()`` after reconciliation. LanceDB's
#: default cleanup retention is 7 days, which leaves superseded fragments and
#: per-commit version/transaction files on disk after every run — measured at
#: ~495 MB extra (~2.3x the settled index) on a 385 MB database. ssgrep is a
#: single-writer CLI with no time-travel reads: the latest version is always
#: preserved, so a 1-second cutoff keeps only the final compacted state.
POST_COMPACT_CLEANUP_RETENTION = timedelta(seconds=1)

logger = logging.getLogger(__name__)

#: IVF-PQ tuning. ``NUM_SUB_VECTORS`` must divide the token dimension
#: (96 / 8 = 12); partitions follow LanceDB's rows // 4096 rule capped at 64.
NUM_SUB_VECTORS = DIMENSION // 8
assert DIMENSION % NUM_SUB_VECTORS == 0, "num_sub_vectors must divide the embedding dimension"
MAX_NUM_PARTITIONS = 64
#: Probes a quarter of the partition ceiling so large indexes still visit
#: ~25% of IVF clusters; the engine clamps it below the ceiling.
NPROBES = max(1, MAX_NUM_PARTITIONS // 4)
#: refine_factor >= 1 rescores ANN hits on raw vectors, putting every
#: production ``_distance`` on the exact ``1 - MaxSim`` scale (see T2).
REFINE_FACTOR = 2

#: Inverse of search's ``DISTANCE_TO_MAXSIM_OFFSET``: two-stage stage 2
#: computes MaxSim directly and reports ``_distance = 1 - MaxSim`` so both
#: paths feed the caller one identical scale.
MAXSIM_TO_DISTANCE_OFFSET = 1.0

#: PQ code width in bits. LanceDB's docs claim 8-only, but 0.37.1 accepts 4
#: at ``create_index``; Task 7 measured recall on a sandbox corpus before
#: adopting. 8 stays the shipped default unless the measured gate says 4 is
#: safe (recall@10/MRR within 0.01 of the 8-bit run AND index bytes <= 0.8x).
DEFAULT_PQ_BITS = 8
ALLOWED_PQ_BITS = (4, 8)


def _validate_pq_bits(num_bits: int) -> int:
    """Reject any PQ width outside the measured set; never fall back silently."""
    if num_bits not in ALLOWED_PQ_BITS:
        allowed = ", ".join(str(bits) for bits in ALLOWED_PQ_BITS)
        raise ValueError(
            f"num_bits must be one of ({allowed},) for the IVF-PQ index; "
            f"got {num_bits!r} (check SSGREP_PQ_BITS)"
        )
    return num_bits


def _pq_bits_from_env() -> int:
    """Resolve SSGREP_PQ_BITS; unset keeps DEFAULT_PQ_BITS, invalid raises."""
    raw = os.environ.get("SSGREP_PQ_BITS", "").strip()
    if not raw:
        return DEFAULT_PQ_BITS
    try:
        bits = int(raw)
    except ValueError:
        bits = -1
    return _validate_pq_bits(bits)


def _index_num_partitions(row_count: int) -> int:
    """LanceDB rule of thumb: one IVF partition per ~4096 rows, capped at 64."""
    return max(1, min(MAX_NUM_PARTITIONS, row_count // 4096))


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    """Clamped integer env override; unparsable values fall back to default."""
    try:
        return max(lo, min(hi, int(os.environ.get(name, "").strip())))
    except ValueError:
        return default


def quote(value: str) -> str:
    """Quote one string literal for a Lance SQL predicate."""
    return "'" + value.replace("'", "''") + "'"


#: Lance's Rust layer reports empty IVF clusters during index training as
#: advisory WARN lines straight to stderr, bypassing Python's logging module
#: entirely. The message and its ``Help:`` continuation are not actionable:
#: they describe the expected shape of late-interaction token vectors, and the
#: index still builds correctly. They are dropped at the fd level instead.
_KMEANS_WARNING = b"KMeans:"
_KMEANS_HELP = b"Help: this could mean"


def _is_index_noise(line: bytes) -> bool:
    return _KMEANS_WARNING in line or _KMEANS_HELP in line


@contextmanager
def _filtered_stderr(drop: Callable[[bytes], bool]) -> Iterator[None]:
    """Temporarily route fd 2 through a filter, replaying kept lines after.

    Lance's Rust layer writes index-training warnings directly to the process
    stderr file descriptor, so neither ``logging`` filters nor replacing
    ``sys.stderr`` can intercept them (verified empirically). Redirecting fd 2
    for the duration of the wrapped call is the only reliable suppression; a
    background thread drains the pipe so the writer can never block on a full
    buffer, and non-dropped lines are replayed to the real stderr afterwards.
    """
    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    collected: list[bytes] = []

    def _pump() -> None:
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            collected.append(chunk)

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        thread.join()
        os.close(read_fd)
    for line in b"".join(collected).split(b"\n"):
        if line and not drop(line):
            os.write(2, line + b"\n")


def _maxsim(query: Any, matrix: Any) -> float:
    """Exact MaxSim between a query matrix and one chunk's token matrix.

    Rows are re-normalized before the dot products to mirror the engine's
    cosine kernel: f16 storage quantization drifts unit norms by ~1e-4
    (measured T9), and skipping renormalization shifts summed MaxSim by up to
    ~0.03 against ``multivector_search``'s refined scores (measured T10).
    """
    q = np.atleast_2d(np.asarray(query, dtype=np.float32))
    d = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return float((q @ d.T).max(axis=1).sum())


class LanceStore:
    """Repository over the global database; reads never create storage."""

    def __init__(self) -> None:
        self.path = database_dir()
        self._db: Any | None = None

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    def _connect(self, *, create: bool) -> Any | None:
        if self._db is not None:
            return self._db
        if not create and not self.path.is_dir():
            return None
        if create:
            ensure_data_dir()
        # The scanner opens one fd per concurrently-read fragment (plus index
        # and manifest files) with no open-fd-count throttle; measured peak
        # per search exceeds 512 while macOS defaults the soft limit to 256.
        raise_nofile_limit()
        self._db = lancedb.connect(str(self.path), read_consistency_interval=timedelta(seconds=0))
        if create:
            os.chmod(self.path, 0o700)
        return self._db

    def exists(self) -> bool:
        return self.path.is_dir()

    def schema_matches(self) -> bool:
        """Return whether every searchable data table matches its model.

        Only ``COMPAT_TABLES`` participate: the ``sources`` registry is an
        indexer-internal table and must not force a rebuild when it appears.
        """
        try:
            return all(
                (table := self.table(name)) is not None
                and set(table.schema.names) == set(schema.model_fields)
                for name in COMPAT_TABLES
                for schema in (TABLE_SCHEMAS[name],)
            )
        except Exception:
            return False

    def table(self, name: str, *, create: bool = False) -> Any | None:
        db = self._connect(create=create)
        if db is None:
            return None
        if name in db.list_tables().tables:
            return db.open_table(name)
        if not create:
            return None
        try:
            schema = TABLE_SCHEMAS[name]
        except KeyError as exc:
            raise KeyError(f"unknown ssgrep table: {name}") from exc
        kwargs: dict[str, Any] = {}
        if name == CHUNKS_TABLE:
            kwargs["embedding_functions"] = [EMBEDDING_CONFIG]
        return db.create_table(name, schema=schema, **kwargs)

    def initialize(self) -> LanceStore:
        for name in TABLE_SCHEMAS:
            self.table(name, create=True)
        self.ensure_vector_index()
        return self

    def reset(self) -> LanceStore:
        db = self._connect(create=False)
        if db is not None:
            # ``drop_all_tables`` replaced the deprecated ``drop_database`` as
            # of lancedb 0.17; the directory removal is the authoritative wipe.
            if hasattr(db, "drop_all_tables"):
                db.drop_all_tables()
            if hasattr(db, "close"):
                db.close()
        self._db = None
        shutil.rmtree(self.path, ignore_errors=True)
        return self.initialize()

    def count(self, name: str, where: str | None = None) -> int:
        table = self.table(name)
        return int(table.count_rows(where)) if table is not None else 0

    def rows(
        self,
        name: str,
        *,
        where: str | None = None,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        table = self.table(name)
        if table is None:
            return []
        query = table.search()
        if where:
            query = query.where(where, prefilter=True)
        if columns is not None:
            query = query.select(list(columns))
        if limit is not None:
            query = query.limit(limit)
        return query.to_list()

    @staticmethod
    def _records(
        name: str, rows: Mapping[str, Any] | Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if isinstance(rows, Mapping):
            row_items = (cast(Mapping[str, Any], rows),)
        else:
            row_items = rows
        records = [dict(row) for row in row_items]
        schema = TABLE_SCHEMAS[name]
        for record in records:
            for field_name, field in schema.model_fields.items():
                if field_name not in record and not field.is_required():
                    record[field_name] = field.get_default(call_default_factory=True)
        return records

    def upsert(
        self,
        name: str,
        rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    ) -> Any | None:
        records = self._records(name, rows)
        if not records:
            return None
        table = self.table(name, create=True)
        assert table is not None
        return (
            table.merge_insert(PRIMARY_KEYS[name])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(records)
        )

    def delete(self, name: str, where: str) -> Any | None:
        table = self.table(name)
        return table.delete(where) if table is not None else None

    def update(self, name: str, where: str, values: Mapping[str, Any]) -> Any | None:
        table = self.table(name)
        return table.update(where=where, values=dict(values)) if table is not None else None

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        rows = self.rows(META_TABLE, where=f"key = {quote(key)}", columns=["value"], limit=1)
        return str(rows[0]["value"]) if rows else default

    def set_meta(self, key: str, value: str) -> Any | None:
        return self.upsert(
            META_TABLE,
            {"key": key, "value": value},
        )

    def ensure_vector_index(self, *, replace: bool = False, num_bits: int | None = None) -> None:
        """Create the cosine IVF-PQ indexes on the chunks ``vector`` columns.

        Late-interaction multivectors only support cosine distance (the Rust
        layer rejects l2/dot), so both indexes are always built with
        ``metric="cosine"``: one on the multivector ``vector`` column, one on
        the single-vector ``proxy_vector`` prefilter column (stage 1 of
        ``SSGREP_TWO_STAGE`` search). Idempotent per column: a finished cosine
        index is left untouched unless ``replace`` is set. ``num_bits`` selects
        the PQ code width explicitly; ``None`` resolves ``SSGREP_PQ_BITS``
        (default 8). Invalid widths raise instead of falling back silently,
        and validation runs before every early return so a misconfigured
        environment can never be quietly ignored.

        The build is deferred (not failed) when there is not yet enough data to
        train a meaningful index: an empty chunks table, a corpus whose total
        token count is below IVF-PQ's 256-token-vector training minimum, or —
        for the proxy column only — a corpus written before the proxy fill
        whose values are all null. A later reconcile builds deferred indexes
        once the data is there. Any other failure propagates so the caller
        never silently falls back to a full scan.
        """
        bits = _validate_pq_bits(_pq_bits_from_env() if num_bits is None else num_bits)
        table = self.table(CHUNKS_TABLE, create=True)
        assert table is not None
        row_count = table.count_rows()
        for column in ("vector", PROXY_COLUMN):
            if column == PROXY_COLUMN and (
                row_count == 0 or table.count_rows(f"{PROXY_COLUMN} IS NOT NULL") == 0
            ):
                continue
            exists = any(
                index.index_type == "IvfPq" and column in index.columns
                for index in table.list_indices()
            )
            if exists and not replace:
                continue
            if row_count == 0:
                continue
            try:
                with _filtered_stderr(_is_index_noise):
                    table.create_index(
                        metric="cosine",
                        vector_column_name=column,
                        replace=replace,
                        num_sub_vectors=NUM_SUB_VECTORS,
                        num_partitions=_index_num_partitions(row_count),
                        num_bits=bits,
                    )
            except RuntimeError as exc:
                if "Not enough rows to train PQ" in str(exc):
                    continue
                raise

    def optimize_tables(self) -> None:
        """Compact the data tables after reconciliation; never raises.

        Each table gets one ``optimize()`` call (compaction + cleanup under
        ``POST_COMPACT_CLEANUP_RETENTION`` + index refresh). A failing table is
        logged as a warning and the remaining tables are still optimized:
        compaction is a size optimization and must never fail an index run.
        """
        for name in OPTIMIZE_TABLES:
            table = self.table(name)
            if table is None:
                continue
            try:
                table.optimize(
                    cleanup_older_than=POST_COMPACT_CLEANUP_RETENTION,
                )
            except Exception as exc:  # noqa: BLE001 - per-table tolerance by contract
                logger.warning("Compaction failed for %s table: %s", name, exc)

    def multivector_search(
        self,
        query_matrix: Any,
        *,
        limit: int,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        """Native LanceDB MaxSim search over the multivector ``vector`` column.

        The query is a ``(num_tokens, DIMENSION)`` float32 matrix produced by the
        query-time embedder (``SsgrepEmbedding.compute_query_embeddings``).
        LanceDB accepts the raw matrix as-is and computes MaxSim against every
        chunk's per-token vectors, returning the engine's ``_distance`` column
        (``refine_factor`` rescores every hit onto the exact
        ``_distance = 1 - MaxSim`` scale). ``where`` filters rows before
        scoring. ``limit`` goes straight to the engine: candidate-pool sizing
        (oversampling) is the caller's decision, and ``search`` multiplies its
        user-facing limit by ``OVERSAMPLE_FACTOR`` before calling.
        """
        table = self.table(CHUNKS_TABLE)
        if table is None or table.count_rows() == 0:
            return []
        search = table.search(query_matrix, vector_column_name="vector")
        # refine_factor forces exact rescoring: unrefined ANN distances sit on
        # a lossy sum-of-min-cosine-distance scale, refined ones on 1 - MaxSim.
        search = search.refine_factor(_env_int("SSGREP_REFINE_FACTOR", REFINE_FACTOR, 1, 20))
        search = search.nprobes(_env_int("SSGREP_NPROBES", NPROBES, 1, 512))
        if where:
            search = search.where(where, prefilter=True)
        return search.limit(limit).to_list()

    def two_stage_search(
        self,
        query_matrix: Any,
        *,
        query_proxy: Any,
        limit: int,
        where: str | None = None,
        pool_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Prefiltered two-stage search behind ``SSGREP_TWO_STAGE``.

        Stage 1 runs a single-vector ANN over ``proxy_vector`` (each chunk's
        L2-normalized mean token vector) for ``limit`` chunk ids, applying the
        same SQL ``where`` prefilter as ``multivector_search``. Stage 2 scores
        ONLY those chunks' full token matrices client-side (numpy MaxSim) and
        reports them on the production scale ``_distance = 1 - MaxSim``, so
        the caller's conversion matches the single-stage path exactly.
        ``pool_size`` slices the scored survivors to the caller's rollup pool
        AFTER scoring, mirroring ``multivector_search``'s ``limit`` semantics
        (candidate selection depth and rollup pool size are independent).

        Client-side rescoring was chosen over a ``.bypass_vector_index()``
        engine query (round-trip verified): bypass rows arrive on the flat
        scan scale ``T + 1 - 2*MaxSim`` (measured T2), not the refined
        ``1 - MaxSim`` every other production score uses, so engine-side
        rescoring would need an empirically inferred conversion to stay
        score-consistent; computing MaxSim directly removes that ambiguity.

        Stage-1 proxy cosine scores never reach the caller: they only SELECT
        candidates. Final ranking uses exclusively stage-2 MaxSim scores.
        Chunks written before the proxy fill carry null proxies and are
        invisible to stage 1; rebuild (or re-reconcile changed sources) to
        fill them before enabling the flag.
        """
        table = self.table(CHUNKS_TABLE)
        if table is None or table.count_rows() == 0:
            return []
        search = table.search(query_proxy, vector_column_name=PROXY_COLUMN)
        search = search.refine_factor(_env_int("SSGREP_REFINE_FACTOR", REFINE_FACTOR, 1, 20))
        search = search.nprobes(_env_int("SSGREP_NPROBES", NPROBES, 1, 512))
        if where:
            search = search.where(where, prefilter=True)
        hits = search.select(["chunk_id", "_distance"]).limit(limit).to_list()
        ids = [str(hit["chunk_id"]) for hit in hits]
        if not ids:
            return []
        # Candidates already satisfy ``where`` (stage-1 prefilter), so the
        # fetch narrows on ids alone.
        rows = self.rows(CHUNKS_TABLE, where=f"chunk_id IN ({', '.join(quote(i) for i in ids)})")
        scored: list[dict[str, Any]] = []
        for row in rows:
            row["_distance"] = MAXSIM_TO_DISTANCE_OFFSET - _maxsim(query_matrix, row["vector"])
            scored.append(row)
        scored.sort(key=lambda item: (item["_distance"], str(item["chunk_id"])))
        if pool_size is not None:
            scored = scored[:pool_size]
        return scored

    def close(self) -> None:
        if self._db is not None and hasattr(self._db, "close"):
            self._db.close()
        self._db = None


__all__ = [
    "CHUNKS_TABLE",
    "CURSORS_TABLE",
    "CWD_CACHE_TABLE",
    "EPISODES_TABLE",
    "META_TABLE",
    "SESSIONS_TABLE",
    "SOURCES_TABLE",
    "LanceStore",
    "ensure_data_dir",
    "quote",
]
