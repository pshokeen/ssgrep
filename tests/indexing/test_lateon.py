"""Offline unit tests for :mod:`ssgrep.indexing.lateon`.

The real ``ColBERTEmbedder`` is a thin wrapper around a lazily loaded
PyLate ``models.ColBERT``. These tests monkeypatch ``load_embedder`` (and
``embed_device``) so the wrapper's wiring — constructor, ``_model``, the
vectorized ``encode_many`` path, the single-text ``embed`` path, and the
``__coco_memo_key__`` identity — is exercised without touching a real model.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ssgrep.indexing import embed, lateon


class _FakeColBERT:
    """Minimal stand-in for PyLate ``models.ColBERT``.

    ``encode`` returns one ``(num_tokens, 2)`` float32 matrix per input text so
    the wrapper's ``np.asarray(..., dtype=np.float32)`` conversion is exercised
    with a small, cheap shape. Call records accumulate on ``self.calls``.
    """

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> list[np.ndarray]:
        self.calls.append({"texts": texts, **kwargs})
        return [np.array([[1.0, 2.0]], dtype=np.float32) for _ in texts]


class _FakeMultiTokenColBERT:
    """A fake returning a fixed multi-token matrix, for document pooling tests.

    The matrix is L2-normalized float32 with more rows than a pool factor of
    2 would keep, so the wrapper's own pooling is observable (row reduction).
    """

    def __init__(self, rows: int = 8, cols: int = 2, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        matrix = rng.normal(size=(rows, cols)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> list[np.ndarray]:
        self.calls.append({"texts": texts, **kwargs})
        return [self.matrix.copy() for _ in texts]


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeColBERT:
    model = _FakeColBERT()
    monkeypatch.setattr(lateon, "load_embedder", lambda: model)
    return model


def test_constructor_defaults_to_default_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """model_name_or_path defaults to the pinned MODEL_ID, device to auto."""
    monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
    embedder = lateon.ColBERTEmbedder()
    assert embedder._model_name_or_path == embed.MODEL_ID
    assert embedder._device == "cpu"


def test_constructor_accepts_explicit_model_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit model name and device are honored; auto-device is not called."""
    calls: list[str] = []
    monkeypatch.setattr(lateon, "embed_device", lambda: calls.append("called") or "cpu")

    embedder = lateon.ColBERTEmbedder("some/model", device="mps")
    assert embedder._model_name_or_path == "some/model"
    assert embedder._device == "mps"
    assert calls == []  # explicit device short-circuits auto-detection


def test_constructor_falls_back_to_auto_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No explicit device → embed_device() is consulted."""
    monkeypatch.setattr(lateon, "embed_device", lambda: "cuda")
    embedder = lateon.ColBERTEmbedder()
    assert embedder._device == "cuda"


def test_model_returns_lazy_loaded_embedder(
    fake_model: _FakeColBERT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_model() delegates to the (cached) module-level load_embedder."""
    monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
    embedder = lateon.ColBERTEmbedder()
    assert embedder._model() is fake_model


def test_env_int_parses_valid_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid env int is parsed and returned."""
    monkeypatch.setenv(lateon.EMBED_BATCH_ENV, "128")
    assert lateon._env_int(lateon.EMBED_BATCH_ENV, 64) == 128


def test_env_int_clamps_negative_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-positive env int clamps up to 1 (max(1, ...))."""
    monkeypatch.setenv(lateon.EMBED_BATCH_ENV, "-5")
    assert lateon._env_int(lateon.EMBED_BATCH_ENV, 64) == 1


def test_env_int_returns_default_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric env value falls back to the default."""
    monkeypatch.setenv(lateon.EMBED_BATCH_ENV, "not-a-number")
    assert lateon._env_int(lateon.EMBED_BATCH_ENV, 64) == 64


def test_env_int_returns_default_on_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset/blank env value falls back to the default."""
    monkeypatch.delenv(lateon.EMBED_BATCH_ENV, raising=False)
    assert lateon._env_int(lateon.EMBED_BATCH_ENV, 64) == 64


def test_encode_many_returns_float32_matrices_per_text(
    fake_model: _FakeColBERT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """encode_many maps one matrix per text and records the encode kwargs."""
    monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
    monkeypatch.delenv(embed.POOL_FACTOR_ENV, raising=False)

    embedder = lateon.ColBERTEmbedder()
    out = embedder.encode_many(["first", "second"], is_query=True)

    assert len(out) == 2
    assert all(v.dtype == np.float32 for v in out)
    assert fake_model.calls == [
        {
            "texts": ["first", "second"],
            "is_query": True,
            "normalize_embeddings": True,
            "device": "cpu",
            "batch_size": lateon.DEFAULT_BATCH_SIZE,
            # Queries are NEVER pooled, regardless of SSGREP_POOL_FACTOR.
            "pool_factor": 1,
        }
    ]


def test_encode_many_defaults_is_query_false(
    fake_model: _FakeColBERT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """encode_many passes is_query=False by default."""
    monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")

    lateon.ColBERTEmbedder().encode_many(["text"])
    assert fake_model.calls[0]["is_query"] is False


def test_embed_returns_first_token_matrix(
    fake_model: _FakeColBERT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embed() is the single-text convenience over encode_many(..., is_query=False)."""
    monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
    embedder = lateon.ColBERTEmbedder()
    out = embedder.embed("hello")
    assert out.shape == (1, 2)
    assert out.dtype == np.float32


def test_coco_memo_key_combines_model_revision_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """__coco_memo_key__ ties detect_change memoization to model identity."""
    monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
    embedder = lateon.ColBERTEmbedder("some/model", device="mps")
    assert embedder.__coco_memo_key__() == ("some/model", embed.MODEL_REVISION, "mps")


def _run(coro):
    """Run a coroutine on a fresh event loop (the consumer binds per loop)."""
    return asyncio.run(coro)


class TestEncodeManyAsync:
    """The batched ``encode_many_async`` path coalesces across callers."""

    def test_coalesces_cross_call_into_one_model_call(
        self,
        fake_model: _FakeColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two submissions made before the consumer runs → one encode call."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        embedder = lateon.ColBERTEmbedder(batch_size=64, flush_seconds=5.0)

        async def go():
            first = asyncio.ensure_future(embedder.encode_many_async(["a", "b"]))
            second = asyncio.ensure_future(embedder.encode_many_async(["c"]))
            r1, r2 = await asyncio.gather(first, second)
            return r1, r2

        r1, r2 = _run(go())
        assert len(fake_model.calls) == 1
        assert fake_model.calls[0]["texts"] == ["a", "b", "c"]
        assert fake_model.calls[0]["is_query"] is False
        assert len(r1) == 2
        assert len(r2) == 1
        assert all(v.shape == (1, 2) for v in r1 + r2)

    def test_dedups_identical_texts_and_copies_vectors(
        self,
        fake_model: _FakeColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Byte-identical texts embed once; duplicates get fresh equal copies."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        embedder = lateon.ColBERTEmbedder(batch_size=64)

        async def go():
            return await embedder.encode_many_async(["dup", "other", "dup"])

        result = _run(go())
        # The model saw only the unique texts.
        assert fake_model.calls[0]["texts"] == ["dup", "other"]
        assert len(result) == 3
        # Duplicate positions carry equal vectors, as fresh arrays (no aliasing).
        np.testing.assert_array_equal(result[0], result[2])
        assert result[0] is not result[2]
        assert result[1].shape == (1, 2)

    def test_dedup_matches_undeduped_output(
        self,
        fake_model: _FakeColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dedup must not change values: same as a run with no duplicates."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        embedder = lateon.ColBERTEmbedder(batch_size=64)

        async def go():
            deduped = await embedder.encode_many_async(["x", "x"])
            return deduped

        deduped = _run(go())
        # Without dedup the model would have returned two independent rows for
        # the repeated text; both carry the same value either way.
        np.testing.assert_array_equal(deduped[0], deduped[1])

    def test_lone_text_still_returns(
        self,
        fake_model: _FakeColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A single text is flushed without waiting for more (no deadlock)."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        embedder = lateon.ColBERTEmbedder(batch_size=64, flush_seconds=0.001)

        result = _run(embedder.encode_many_async(["solo"]))
        assert len(result) == 1
        assert result[0].shape == (1, 2)

    def test_sequential_calls_when_consumer_idle(
        self,
        fake_model: _FakeColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A call after the consumer idled (waiting on the empty queue) works."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        embedder = lateon.ColBERTEmbedder(batch_size=64, flush_seconds=0.001)

        async def go():
            first = await embedder.encode_many_async(["x"])
            second = await embedder.encode_many_async(["y"])
            return first, second

        first, second = _run(go())
        assert len(first) == 1
        assert len(second) == 1

    def test_idles_when_queue_empty_between_batches(
        self,
        fake_model: _FakeColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The consumer times out waiting on an empty queue, then resumes."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        embedder = lateon.ColBERTEmbedder(batch_size=64, flush_seconds=0.01)

        async def go():
            # First batch empties the queue; the consumer then idles in
            # wait_for(queue.get(), flush_seconds) until it times out.
            await embedder.encode_many_async(["first"])
            await asyncio.sleep(embedder._flush_seconds * 5)
            later = await embedder.encode_many_async(["later"])
            return later

        later = _run(go())
        assert len(later) == 1
        assert later[0].shape == (1, 2)
        # Every text was embedded (both batches hit the model).
        assert len(fake_model.calls) == 2

    def test_propagates_encode_errors_to_callers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A model failure surfaces to the awaiting caller, not a hang."""

        class _RaisingModel:
            def encode(self, texts: list[str], **kwargs: object) -> object:
                raise RuntimeError("boom")

        monkeypatch.setattr(lateon, "load_embedder", lambda: _RaisingModel())
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        embedder = lateon.ColBERTEmbedder(batch_size=64)

        with pytest.raises(RuntimeError, match="boom"):
            _run(embedder.encode_many_async(["x"]))


class TestTokenPooling:
    """Document-side token pooling via SSGREP_POOL_FACTOR (schema v6 lever).

    Documents pool (default 2) — in ssgrep, after the model call, because
    PyLate's in-model Ward pooling can crash on float32 cosine rounding (see
    ``_pool_token_matrix``). The model always receives ``pool_factor=1``.
    Queries NEVER pool regardless of the env.
    """

    @pytest.fixture
    def multi_model(self, monkeypatch: pytest.MonkeyPatch) -> _FakeMultiTokenColBERT:
        model = _FakeMultiTokenColBERT(rows=8, cols=2, seed=7)
        monkeypatch.setattr(lateon, "load_embedder", lambda: model)
        return model

    def test_document_encode_asks_model_for_pool_factor_one(
        self,
        multi_model: _FakeMultiTokenColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Document pooling runs in ssgrep, so the model call never pools."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        monkeypatch.delenv(embed.POOL_FACTOR_ENV, raising=False)

        lateon.ColBERTEmbedder().encode_many(["doc"])

        assert multi_model.calls[0]["pool_factor"] == 1
        assert multi_model.calls[0]["is_query"] is False

    def test_document_encode_applies_default_pool_factor(
        self,
        multi_model: _FakeMultiTokenColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        monkeypatch.delenv(embed.POOL_FACTOR_ENV, raising=False)

        out = lateon.ColBERTEmbedder().encode_many(["doc"])

        assert embed.DEFAULT_POOL_FACTOR == 2
        # 8 token rows -> 1 protected + ~3-4 cluster means: strictly fewer rows.
        assert len(out[0]) < 8
        # The wrapper pools exactly as _pool_token_matrix does.
        expected = lateon._pool_token_matrix(multi_model.matrix, 2)
        np.testing.assert_array_equal(out[0], expected)

    def test_document_encode_honors_env_override(
        self,
        multi_model: _FakeMultiTokenColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "1")

        out = lateon.ColBERTEmbedder().encode_many(["doc"])

        assert multi_model.calls[0]["pool_factor"] == 1
        np.testing.assert_array_equal(out[0], multi_model.matrix)

    def test_query_encode_never_pools_even_with_env_set(
        self,
        multi_model: _FakeMultiTokenColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "3")

        out = lateon.ColBERTEmbedder().encode_many(["query"], is_query=True)

        assert multi_model.calls[0]["pool_factor"] == 1
        np.testing.assert_array_equal(out[0], multi_model.matrix)

    def test_async_document_path_pools(
        self,
        multi_model: _FakeMultiTokenColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The pipeline's batched path pools documents after the model call."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "2")
        embedder = lateon.ColBERTEmbedder(batch_size=64)

        result = _run(embedder.encode_many_async(["a"]))

        assert multi_model.calls[0]["pool_factor"] == 1
        assert len(result) == 1
        assert len(result[0]) < 8
        np.testing.assert_array_equal(result[0], lateon._pool_token_matrix(multi_model.matrix, 2))

    def test_async_query_path_never_pools(
        self,
        multi_model: _FakeMultiTokenColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "3")
        embedder = lateon.ColBERTEmbedder(batch_size=64)

        result = _run(embedder.encode_many_async(["q"], is_query=True))

        assert multi_model.calls[0]["pool_factor"] == 1
        assert len(result[0]) == 8
        np.testing.assert_array_equal(result[0], multi_model.matrix)

    def test_constructor_accepts_explicit_pool_factor(
        self,
        multi_model: _FakeMultiTokenColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "2")

        embedder = lateon.ColBERTEmbedder(pool_factor=1)
        out = embedder.encode_many(["doc"])

        assert multi_model.calls[0]["pool_factor"] == 1
        np.testing.assert_array_equal(out[0], multi_model.matrix)

    def test_async_dedup_copies_pooled_vectors(
        self,
        multi_model: _FakeMultiTokenColBERT,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pooled duplicates still come back as fresh, equal matrices."""
        monkeypatch.setattr(lateon, "embed_device", lambda: "cpu")
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "2")
        embedder = lateon.ColBERTEmbedder(batch_size=64)

        result = _run(embedder.encode_many_async(["dup", "dup"]))

        np.testing.assert_array_equal(result[0], result[1])
        assert result[0] is not result[1]
        assert result[0].shape == lateon._pool_token_matrix(multi_model.matrix, 2).shape


class TestPoolTokenMatrix:
    """Ward-linkage document pooling (:func:`lateon._pool_token_matrix`)."""

    def _matrix(self, rows: list[list[float]]) -> np.ndarray:
        return np.asarray(rows, dtype=np.float32)

    def test_merges_tokens_into_cluster_means(self) -> None:
        matrix = self._matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        out = lateon._pool_token_matrix(matrix, 2)
        # 3 poolable rows -> 1 cluster mean; protected row survives verbatim.
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(out[0], [1.0, 0.0])
        np.testing.assert_allclose(out[1], [1.0 / 3.0, 2.0 / 3.0])

    def test_clamps_negative_cosine_distances(self) -> None:
        # Rows with norm > 1 make 1 - cosine < 0 in exact float32 arithmetic.
        # PyLate's pooling passes those to scipy and crashes with "Linkage 'Z'
        # contains negative distances"; ssgrep clamps them away. Deterministic
        # on every platform/BLAS because the dot products round well below 1.
        matrix = self._matrix([[3.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
        out = lateon._pool_token_matrix(matrix, 2)
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(out[0], [3.0, 0.0])
        # All three non-protected rows land in one cluster.
        np.testing.assert_allclose(out[1], [0.0, 2.0])

    def test_filters_unused_cluster_slots(self) -> None:
        # Identical rows merge to a single cluster even when more are allowed,
        # leaving zero-count slots that must be dropped, never emitted as zeros.
        matrix = self._matrix([[1.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
        out = lateon._pool_token_matrix(matrix, 2)
        assert out.shape == (2, 2)
        assert out[0, 0] == 1.0
        assert out[1, 0] == 5.0
        assert out[1, 1] == 0.0

    def test_returns_input_when_pooling_would_not_reduce(self) -> None:
        matrix = self._matrix([[1.0, 0.0], [0.0, 1.0]])
        out = lateon._pool_token_matrix(matrix, 2)
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(out, matrix)

    def test_single_token_is_unchanged(self) -> None:
        matrix = self._matrix([[1.0, 2.0]])
        out = lateon._pool_token_matrix(matrix, 2)
        assert out.shape == (1, 2)
        np.testing.assert_array_equal(out, matrix)

    def test_output_is_float32(self) -> None:
        matrix = self._matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        out = lateon._pool_token_matrix(matrix, 2)
        assert out.dtype == np.float32
        assert matrix.dtype == np.float32
