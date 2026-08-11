"""Tests for embedding encoder."""

import numpy as np
import pytest

from ssgrep import embed
from ssgrep.embed import DIMENSION, MODEL_ID, encode, get_model_info

# Check if model2vec is available
try:
    import model2vec  # noqa: F401

    MODEL2VEC_AVAILABLE = True
except ImportError:
    MODEL2VEC_AVAILABLE = False


@pytest.mark.skipif(not MODEL2VEC_AVAILABLE, reason="Requires model2vec runtime")
def test_shape():
    result = encode(["hello", "world"])
    assert result.shape == (2, DIMENSION)


@pytest.mark.skipif(not MODEL2VEC_AVAILABLE, reason="Requires model2vec runtime")
def test_dtype():
    result = encode(["test"])
    assert result.dtype == np.float32


@pytest.mark.skipif(not MODEL2VEC_AVAILABLE, reason="Requires model2vec runtime")
def test_empty_input():
    result = encode([])
    assert result.shape == (0, DIMENSION)


def test_model_info():
    model_id, dim = get_model_info()
    assert model_id == MODEL_ID
    assert dim == 256


class _FakeModel:
    """Stand-in for model2vec.StaticModel.

    encode()'s normalization contract (unit-length output, safe on
    zero-magnitude input) lives entirely in encode() itself, after the
    model.encode(texts) call -- it doesn't depend on what the real model
    computes. Substituting a fake model with pre-set, hand-picked output lets
    these tests exercise that normalization code directly and
    deterministically, without a network call or the real model2vec runtime
    (the existing test_shape/test_dtype/test_empty_input tests above are
    skipped without model2vec; these are not).
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        assert len(texts) == len(self._embeddings), (
            "fixture mismatch: fake model given a different number of texts "
            "than pre-set embeddings"
        )
        return self._embeddings


def test_encode_normalizes_to_unit_length(monkeypatch):
    """encode()'s output vectors must be unit-length (L2 norm == 1.0, within
    float tolerance) -- this is what actually gets written to vectors.f32.

    This must be asserted on encode()'s own output, not on search rankings:
    vectors.cosine_top_k() independently re-normalizes both the query and
    every stored vector on every call, so cosine-similarity *ranking* is
    mathematically invariant to whether encode() itself normalizes. A
    ranking-based test (e.g. "search still returns the right episode") would
    keep passing even if this normalization step were deleted entirely --
    the stored vectors.f32 bytes would just quietly be wrong.
    """
    raw = np.array(
        [
            [3.0, 4.0, 0.0, 0.0],  # magnitude 5
            [1.0, 1.0, 1.0, 1.0],  # magnitude 2
            [10.0, 0.0, 0.0, 0.0],  # magnitude 10, already axis-aligned
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(embed, "_get_model", lambda: _FakeModel(raw))

    result = embed.encode(["a", "b", "c"])

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(
        norms, [1.0, 1.0, 1.0], atol=1e-5, err_msg="encode() output is not unit-length"
    )
    # Direction must be preserved, not just magnitude coincidentally fixed.
    np.testing.assert_allclose(result[0], raw[0] / 5.0, atol=1e-5)
    np.testing.assert_allclose(result[1], raw[1] / 2.0, atol=1e-5)
    np.testing.assert_allclose(result[2], raw[2] / 10.0, atol=1e-5)


def test_encode_zero_vector_guard_avoids_nan(monkeypatch):
    """A zero-magnitude raw embedding must not be divided by zero.

    encode() guards this with `norms = np.where(norms > 0, norms, 1)` before
    dividing. If that guard were removed or broken, a zero vector would
    produce 0/0 = NaN, which would poison vectors.f32 -- and every downstream
    consumer (cosine_top_k's matrix-wide dot product, mmap reads of other
    rows via the same file, etc.) with no exception raised anywhere.
    """
    raw = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # zero vector: guard must kick in
            [2.0, 0.0, 0.0, 0.0],  # ordinary vector, for contrast
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(embed, "_get_model", lambda: _FakeModel(raw))

    result = embed.encode(["empty", "ordinary"])

    assert not np.isnan(result).any(), "zero-vector guard failed to prevent NaN"
    assert not np.isinf(result).any(), "zero-vector guard failed to prevent Inf"
    # Guarded divisor is 1, so the zero row stays exactly zero rather than NaN.
    np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0, 0.0], atol=1e-6)
    # The ordinary row is unaffected by the guard and still normalizes normally.
    np.testing.assert_allclose(result[1], [1.0, 0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.skipif(not MODEL2VEC_AVAILABLE, reason="Requires model2vec runtime")
def test_cold_load_timeout_is_enforced_and_actionable(monkeypatch):
    """A hung first-run model download must fail fast with an actionable
    message, not hang for the full production timeout -- let alone the
    ~25s the underlying HTTP stack can silently take when a connection is
    accepted but never responds -- a condition reproduced end-to-end against
    a real black-holed connection both before and after this fix.

    Shrinks the deadline via the SSGREP_MODEL_LOAD_TIMEOUT override so this
    test itself stays fast while
    still exercising the real timeout-enforcement code path in
    embed._load_model() (the queue.Empty branch), not just asserting on a
    mocked-out outcome. StaticModel.from_pretrained is replaced with a
    function that blocks forever, simulating a black-holed network read
    that never returns; the real background thread genuinely hangs (it is a
    daemon, so it cannot block process/test-session exit) while the main
    thread's queue.get(timeout=...) is what bounds how long we wait for it.
    """
    import time

    def hangs_forever(*args, **kwargs):
        time.sleep(999)

    # Through the real env override, not by patching a module constant: the
    # override IS part of the fix (there was no way to raise the deadline at
    # all, which is what made a too-short default a dead end for buyers), so
    # it should be exercised rather than bypassed.
    monkeypatch.setenv(embed.MODEL_LOAD_TIMEOUT_ENV, "1")
    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.setattr(model2vec.StaticModel, "from_pretrained", hangs_forever)

    start = time.perf_counter()
    with pytest.raises(embed.ModelDownloadError, match="Timed out"):
        embed._get_model()
    elapsed = time.perf_counter() - start

    # 3x the 1.0s budget -- loose enough for slow/noisy CI, tight enough to
    # prove the deadline is actually enforced rather than silently ignored.
    assert elapsed < 3.0, (
        f"Timeout enforcement took {elapsed:.2f}s wall-clock against a 1.0s "
        f"budget -- the deadline is not being enforced."
    )


@pytest.mark.skipif(not MODEL2VEC_AVAILABLE, reason="Requires model2vec runtime")
def test_first_run_proxy_error_is_actionable_not_a_raw_exception(monkeypatch):
    """A first-run download that fails with httpx.ProxyError must surface as
    a clear, actionable ModelDownloadError, not the raw httpx exception (or
    a bare stack trace at the CLI).

    httpx.ProxyError is the one failure mode huggingface_hub's
    snapshot_download explicitly re-raises rather than quietly falling back
    to any local cache (`except httpx.ProxyError: raise` in
    huggingface_hub/_snapshot_download.py) -- so it is the case most likely
    to reach ssgrep's caller unwrapped if this error handling regresses.
    Empirically reproduced end-to-end against a real, persistent listening
    proxy that always responds 502: the exception
    huggingface_hub actually surfaces in that exact scenario is
    httpx.ProxyError("502 Bad Gateway"). This test reproduces that outcome
    deterministically, without a real socket.
    """
    import httpx

    def raises_proxy_error(*args, **kwargs):
        raise httpx.ProxyError("502 Bad Gateway")

    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.setattr(model2vec.StaticModel, "from_pretrained", raises_proxy_error)

    with pytest.raises(embed.ModelDownloadError) as exc_info:
        embed._get_model()

    message = str(exc_info.value)
    assert not isinstance(
        exc_info.value, httpx.ProxyError
    ), "the raw httpx exception must not reach the caller directly"
    assert (
        "502 Bad Gateway" in message
    ), f"the underlying cause should be named in the message, not swallowed: {message!r}"
    assert "network" in message.lower() or "proxy" in message.lower(), (
        f"message must point at network/proxy as the likely cause, not just "
        f"repeat the raw error: {message!r}"
    )


def test_vector_row_bytes_matches_embed_dimension():
    """The store's vector row stride is derived from the one true dimension.

    Guard against the dimension being centralized in embed.DIMENSION but a
    consumer regressing to a hardcoded copy that silently diverges.
    """
    from ssgrep import embed
    from ssgrep.store.generations import GenerationalStore

    assert GenerationalStore.VECTOR_ROW_BYTES == embed.DIMENSION * 4


def test_model_download_error_message_reports_restricted_size():
    """The timeout message's download size matches the restricted download (~30 MB).

    String-consistency guard: README and the error message must agree on the
    size a buyer actually downloads (ONNX excluded), not the full repo size.
    """
    import inspect

    from ssgrep import embed

    source = inspect.getsource(embed)
    assert "~30 MB download" in source
    assert "~61 MB download" not in source
