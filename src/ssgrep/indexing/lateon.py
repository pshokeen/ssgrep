"""Late-interaction (pylate) index-time embedder for CocoIndex.

CocoIndex's built-in sentence-transformers embedder is hard-wired to a single
1-D ``VectorSchema`` and cannot be pointed at a pylate ``ColBERT`` (which
produces one ``(num_tokens, DIMENSION)`` matrix per text). This module provides
the replacement provider: a thin wrapper around the PyLate ``models.ColBERT``
loaded through ``ssgrep.indexing.embed.load_embedder`` whose ``embed`` returns
a per-token ``(num_tokens, DIMENSION)`` float32 matrix. The pipeline persists
those matrices as a multivector column via an explicit ``LanceType`` (see
``ssgrep.pipeline.rows``), so the whole CocoIndex engine ingests, persists,
indexes, and searches multivectors without being bypassed.

Index-time batching
-------------------
CocoIndex runs every ``process_source`` coroutine on a single event loop. If
each source called ``ColBERT.encode`` directly it would block that loop with a
small, source-local batch (a few chunks), serializing the "concurrent" sources
on the GPU and leaving the device under-utilized. Instead
:meth:`ColBERTEmbedder.encode_many_async` hands each source's texts to a
shared asyncio accumulator; one background consumer task drains them into large
``SSGREP_EMBED_BATCH_SIZE`` batches, runs ``ColBERT.encode`` off the loop (in an
executor), deduplicates byte-identical texts within a batch, and resolves each
caller's future with its per-text matrices. One consumer means exactly one model
call is in flight at a time, so the device (MPS/CPU) is never contended by
parallel encode calls. ``pylate``'s own ``encode`` length-sorts before batching,
so coalescing across sources also yields more uniform internal batches.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np
from scipy.cluster import hierarchy

from ssgrep.indexing.embed import (
    MODEL_ID,
    MODEL_REVISION,
    embed_device,
    load_embedder,
    resolve_pool_factor,
)

#: Env var overriding the target number of texts coalesced into one model batch.
EMBED_BATCH_ENV = "SSGREP_EMBED_BATCH_SIZE"

#: Default target batch size (texts per ``ColBERT.encode`` call).
DEFAULT_BATCH_SIZE = 64

#: Env var overriding how long the consumer idles for more texts when the
#: accumulator is momentarily empty (milliseconds).
EMBED_FLUSH_MS_ENV = "SSGREP_EMBED_FLUSH_MS"

#: Default idle window the batcher waits for additional texts (milliseconds).
DEFAULT_FLUSH_MS = 50


def _env_int(name: str, default: int) -> int:
    """A positive int from ``name``, or ``default`` when unset/blank/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _pool_token_matrix(
    matrix: np.ndarray,
    pool_factor: int,
    protected_tokens: int = 1,
) -> np.ndarray:
    """Pool one document's token matrix via Ward-linkage cluster means.

    Replicates PyLate's ``pool_embeddings_hierarchical`` for a single
    document: the first ``protected_tokens`` rows (the query/CLS-like token)
    are kept verbatim, and the remaining token vectors are merged into
    ``max(len // pool_factor, 1)`` clusters whose means replace them.

    The one deliberate difference from PyLate is numerical: the model emits
    L2-normalized float32 token vectors, and ``1 - cosine`` can round to a
    tiny negative value (-1e-7) when float32 dot products land just above
    1.0. scipy's Ward linkage propagates that noise into negative linkage
    distances, and ``fcluster`` then raises ``ValueError: Linkage 'Z'
    contains negative distances`` (observed on real transcripts). The
    distances are therefore clamped to ``>= 0`` before linkage, so pooling
    degrades to a zero-distance merge instead of crashing the index run.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    protected = matrix[:protected_tokens]
    to_pool = matrix[protected_tokens:]
    num_embeddings = to_pool.shape[0]
    num_clusters = max(num_embeddings // pool_factor, 1)

    # Skip pooling if it wouldn't reduce anything (also covers the empty and
    # single-token cases without special-casing them).
    if num_clusters >= num_embeddings:
        return matrix

    # Cosine similarity -> cosine distance, with float32 rounding noise
    # above cos=1.0 clamped away (see docstring).
    distances = np.clip(1.0 - to_pool @ to_pool.T, 0.0, None)
    condensed = distances[np.triu_indices(num_embeddings, k=1)]

    linkage_matrix = hierarchy.linkage(condensed, method="ward")
    labels = hierarchy.fcluster(linkage_matrix, t=num_clusters, criterion="maxclust")

    # Cluster means, mirroring PyLate's scatter_add_ / count-mask bookkeeping.
    # ``fcluster`` labels are 1-indexed and may use fewer than ``num_clusters``
    # distinct values, so zero-count rows are filtered out below.
    sums = np.zeros((num_clusters, to_pool.shape[1]), dtype=np.float32)
    np.add.at(sums, labels - 1, to_pool)
    counts = np.bincount(labels - 1, minlength=num_clusters)
    nonzero = counts > 0
    pooled = (sums[nonzero] / counts[nonzero, None]).astype(np.float32)
    return np.concatenate([protected, pooled], axis=0)


class ColBERTEmbedder:
    """CocoIndex context provider producing per-token ``(T, DIMENSION)`` matrices.

    ``embed`` / ``encode_many`` are the synchronous, single-shot paths
    (used by tests and single-call callers). ``encode_many_async`` is the
    batched path the pipeline's ``process_source`` awaits: it coalesces the
    chunk texts of all concurrently-running sources into large model batches on
    a shared consumer, so the device stays busy and the event loop stays free.
    ``__coco_memo_key__`` ties ``detect_change`` memoization to the model
    identity, so swapping the model (or device) invalidates memos and re-embeds.
    """

    def __init__(
        self,
        model_name_or_path: str = MODEL_ID,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        flush_seconds: float | None = None,
        pool_factor: int | None = None,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._device = device or embed_device()
        self._batch_size = batch_size or _env_int(EMBED_BATCH_ENV, DEFAULT_BATCH_SIZE)
        self._flush_seconds = (
            flush_seconds
            if flush_seconds is not None
            else _env_int(EMBED_FLUSH_MS_ENV, DEFAULT_FLUSH_MS) / 1000.0
        )
        # Document-side token pooling (SSGREP_POOL_FACTOR). Queries are never
        # pooled: the encode paths force pool_factor=1 when is_query=True.
        self._pool_factor = resolve_pool_factor() if pool_factor is None else pool_factor
        # Batching state is created lazily on the first async call so it binds
        # to the exact event loop ``process_source`` runs on. A per-environment
        # embedder is never shared across loops, so this stays on one loop.
        self._queue: (
            asyncio.Queue[tuple[list[str], bool, asyncio.Future[list[np.ndarray]]]] | None
        ) = None
        self._consumer: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _model(self) -> Any:
        # ``load_embedder`` lazily loads the PyLate ColBERT for ``MODEL_ID``
        # (revision + device resolved there), cached process-wide under a lock.
        return load_embedder()

    def _encode(
        self,
        texts: list[str],
        *,
        is_query: bool,
    ) -> list[np.ndarray]:
        """One blocking ``ColBERT.encode`` call, with document pooling applied.

        The model is always asked for ``pool_factor=1`` (no pooling): for
        documents ssgrep applies :func:`_pool_token_matrix` itself because
        PyLate's in-model pooling can crash scipy's ``fcluster`` on float32
        cosine rounding (see that function). Queries are never pooled.
        """
        model = self._model()
        encoded = model.encode(
            list(texts),
            is_query=is_query,
            normalize_embeddings=True,
            device=self._device,
            batch_size=self._batch_size,
            pool_factor=1,
        )
        matrices = [np.asarray(matrix, dtype=np.float32) for matrix in encoded]
        if not is_query and self._pool_factor > 1:
            matrices = [_pool_token_matrix(matrix, self._pool_factor) for matrix in matrices]
        return matrices

    def encode_many(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
    ) -> list[np.ndarray]:
        """Embed a list of texts into per-token ``(T, DIMENSION)`` float32 matrices.

        ``is_query`` selects the query prefix / padding semantics of the ColBERT
        model (queries are padded, documents are not). Returns one matrix per
        input text, each of shape ``(num_tokens, DIMENSION)``. This is the
        synchronous single-shot path; the index pipeline should prefer
        :meth:`encode_many_async` so batches span multiple sources.
        """
        return self._encode(texts, is_query=is_query)

    def embed(self, text: str) -> np.ndarray:
        """Embed one text into a ``(num_tokens, DIMENSION)`` float32 matrix."""
        return self.encode_many([text], is_query=False)[0]

    async def encode_many_async(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
    ) -> list[np.ndarray]:
        """Coalesced batch embedding for the concurrent index pipeline.

        Submits ``texts`` to a shared accumulator and awaits the consumer's
        result. Every concurrently-running ``process_source`` coroutine does the
        same, so the consumer drains many sources' chunks into one large
        ``ColBERT.encode`` call instead of one tiny call per source. Awaited, so
        the event loop stays free for other sources to parse/chunk/submit while
        the device is busy.
        """
        loop = asyncio.get_running_loop()
        queue = self._ensure_consumer(loop)
        future: asyncio.Future[list[np.ndarray]] = loop.create_future()
        await queue.put((list(texts), is_query, future))
        return await future

    def _ensure_consumer(
        self, loop: asyncio.AbstractEventLoop
    ) -> asyncio.Queue[tuple[list[str], bool, asyncio.Future[list[np.ndarray]]]]:
        """Create the shared queue + consumer task on ``loop`` (idempotent)."""
        if self._consumer is None or self._loop is not loop:
            self._loop = loop
            queue: asyncio.Queue[tuple[list[str], bool, asyncio.Future[list[np.ndarray]]]] = (
                asyncio.Queue()
            )
            self._queue = queue
            self._consumer = loop.create_task(self._consume())
            return queue
        assert self._queue is not None
        return self._queue

    async def _consume(self) -> None:
        """Drain the accumulator into batches and resolve each caller's future.

        Drains immediately-available items (coalescing happens naturally while a
        batch is encoding in the executor); only when the accumulator is empty
        does it idle up to ``flush_seconds`` waiting for the next burst. Each
        batch is encoded off the loop so the event loop stays responsive.
        """
        loop = asyncio.get_running_loop()
        queue = self._ensure_consumer(loop)
        while True:
            try:
                first = queue.get_nowait()
            except asyncio.QueueEmpty:
                try:
                    first = await asyncio.wait_for(queue.get(), timeout=self._flush_seconds)
                except TimeoutError:
                    continue
            batch = [first]
            while len(batch) < self._batch_size:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            await self._encode_batch(loop, batch)

    async def _encode_batch(
        self,
        loop: asyncio.AbstractEventLoop,
        batch: list[tuple[list[str], bool, asyncio.Future[list[np.ndarray]]]],
    ) -> None:
        """Embed one coalesced batch and resolve its futures in submission order.

        Identical ``(text, is_query)`` entries within the batch are embedded once
        and their vectors copied (fresh arrays, never aliased) to each duplicate
        position, so byte-identical chunks (repeated errors, ``git status``
        blocks, boilerplate) cost one model row instead of many. Any encode
        failure is propagated to every future in the batch, never swallowed —
        otherwise the awaiting ``process_source`` coroutines would hang forever.
        """
        # Flatten texts with their provenance so per-caller ordering survives.
        flat: list[tuple[str, bool, int, int]] = [
            (text, is_query, item_index, position)
            for item_index, (texts, is_query, _future) in enumerate(batch)
            for position, text in enumerate(texts)
        ]
        # One model row per distinct (text, is_query); map duplicates onto it.
        first_index: dict[tuple[str, bool], int] = {}
        for index, (text, is_query, _item, _pos) in enumerate(flat):
            first_index.setdefault((text, is_query), index)
        unique_by_flag: dict[bool, list[str]] = {}
        for index, (text, is_query, _item, _pos) in enumerate(flat):
            if first_index[(text, is_query)] == index:
                unique_by_flag.setdefault(is_query, []).append(text)

        try:
            vector_by_key: dict[tuple[str, bool], np.ndarray] = {}
            for is_query, texts in unique_by_flag.items():
                encoded = await loop.run_in_executor(None, self._encode_unique, texts, is_query)
                for text, matrix in zip(texts, encoded, strict=True):
                    vector_by_key[(text, is_query)] = matrix

            per_item: list[list[np.ndarray]] = [[] for _ in batch]
            for index, (text, is_query, item_index, _position) in enumerate(flat):
                # Duplicate positions get a fresh copy so chunk rows never share
                # a mutable numpy array (np.asarray would keep the same object).
                if first_index[(text, is_query)] == index:
                    per_item[item_index].append(vector_by_key[(text, is_query)])
                else:
                    per_item[item_index].append(vector_by_key[(text, is_query)].copy())
            for (_texts, _is_query, future), result in zip(batch, per_item, strict=True):
                if not future.done():
                    future.set_result(result)
        except BaseException as exc:  # noqa: BLE001 - every caller must unblock
            for _texts, _is_query, future in batch:
                if not future.done():
                    future.set_exception(exc)

    def _encode_unique(self, texts: list[str], is_query: bool) -> list[np.ndarray]:
        """One blocking encode call for a deduplicated group.

        Runs in the consumer's executor thread so the event loop is never
        blocked by the model forward pass. ``pylate`` length-sorts internally, so
        the coalesced batch yields uniform sub-batches of ``batch_size``.
        """
        return self._encode(texts, is_query=is_query)

    def __coco_memo_key__(self) -> tuple[str, str, str]:
        """Stable identity for ``detect_change`` memo invalidation."""
        return (self._model_name_or_path, MODEL_REVISION, self._device)


__all__ = ["ColBERTEmbedder"]
