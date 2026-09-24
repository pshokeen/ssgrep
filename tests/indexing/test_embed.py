"""Offline unit tests for :mod:`ssgrep.indexing.embed`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ssgrep.indexing import embed


def test_default_model_identity_and_dimension() -> None:
    assert embed.DEFAULT_MODEL_ID == "lightonai/answerai-colbert-small-v1"
    assert embed.MODEL_ID == embed.DEFAULT_MODEL_ID
    assert embed.DIMENSION == 96
    # The default is pinned: upstream main has force-pushed the weights away
    # before, so an unpinned load can resolve to a config/tokenizer-only revision.
    assert embed.DEFAULT_MODEL_REVISION == "e507cd12947a2b4b52201d150967df3c19a90590"
    assert embed.MODEL_REVISION == embed.DEFAULT_MODEL_REVISION


def test_rerank_provider_is_removed() -> None:
    """Re-ranking is gone: no default reranker id, and the env var errors."""
    assert not hasattr(embed, "DEFAULT_RERANK_MODEL_ID")
    assert not hasattr(embed, "DEFAULT_RERANK_MODEL_REVISION")
    assert not hasattr(embed, "RERANK_MODEL_REVISION")
    # Kept only so the store module can still import the name.
    assert embed.RERANK_MODEL_ID == ""


def test_rerank_env_var_is_an_explicit_error() -> None:
    """SSGREP_RERANK_MODEL set → import fails loudly, not a silent no-op."""
    import subprocess
    import sys

    probe = "import ssgrep.indexing.embed"
    env = {**__import__("os").environ, "SSGREP_RERANK_MODEL": "cross-encoder/ms-marco-TinyBERT-L-6"}
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "no longer supported" in result.stderr
    assert "SSGREP_RERANK_MODEL" in result.stderr


def test_rerank_model_env_raise_is_covered_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSGREP_RERANK_MODEL set → the guard raises instead of a silent no-op."""
    monkeypatch.setenv(embed.RERANK_MODEL_ENV, "cross-encoder/ms-marco-TinyBERT-L-6")
    with pytest.raises(RuntimeError, match="msg"):
        embed._reject_deprecated_env(embed.RERANK_MODEL_ENV, "msg")


def test_rerank_blend_env_raise_is_covered_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSGREP_RERANK_BLEND set → the guard raises instead of a silent no-op."""
    monkeypatch.setenv(embed.RERANK_BLEND_ENV, "0.3")
    with pytest.raises(RuntimeError, match="msg"):
        embed._reject_deprecated_env(embed.RERANK_BLEND_ENV, "msg")


def test_env_overrides_swap_model_identity_and_clear_revision() -> None:
    """Module-level identity is env-driven at import; probe a fresh process."""
    import subprocess
    import sys

    probe = (
        "import ssgrep.indexing.embed as e;"
        "print(e.MODEL_ID);print(e.MODEL_REVISION);print(e.embed_device())"
    )
    env = {
        **__import__("os").environ,
        "SSGREP_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "SSGREP_EMBED_DEVICE": "cpu",
    }
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    ).stdout.splitlines()
    assert out == [
        "sentence-transformers/all-MiniLM-L6-v2",
        "",  # unpinned for non-default models
        "cpu",
    ]

    default_env = dict(__import__("os").environ)
    default_env.pop("SSGREP_EMBED_MODEL", None)
    default_env.pop("SSGREP_EMBED_DEVICE", None)
    default_env.pop("SSGREP_NUM_THREADS", None)
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=default_env, check=True
    ).stdout.splitlines()
    saved_device = os.environ.pop(embed.EMBED_DEVICE_ENV, None)
    try:
        expected_device = embed.embed_device()
    finally:
        if saved_device is not None:
            os.environ[embed.EMBED_DEVICE_ENV] = saved_device
    assert out == [
        embed.DEFAULT_MODEL_ID,
        embed.DEFAULT_MODEL_REVISION,
        # Auto-detected device in a fresh process must match this process's
        # auto-detection (same machine): fastest available (mps/cuda/cpu).
        expected_device,
    ]


def test_embed_device_auto_detects_and_ignores_blank_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    original = embed.embed_device()
    monkeypatch.delenv(embed.EMBED_DEVICE_ENV, raising=False)
    auto = embed.embed_device()
    assert auto == original  # deterministic within a process
    expected = "cpu"
    if torch.cuda.is_available():
        expected = "cuda"
    elif torch.backends.mps.is_available():
        expected = "mps"
    assert auto == expected
    assert auto in {"cpu", "mps", "cuda"}
    monkeypatch.setenv(embed.EMBED_DEVICE_ENV, "   ")
    assert embed.embed_device() == auto
    monkeypatch.setenv(embed.EMBED_DEVICE_ENV, "cpu")
    assert embed.embed_device() == "cpu"


class TestLoadEmbedder:
    """load_embedder returns a PyLate ColBERT-compatible object, lazily cached."""

    def test_returns_pylate_colbert_with_encode_signature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The returned object exposes .encode(texts, is_query, normalize_embeddings)."""
        calls: list[dict[str, object]] = []

        class _FakeColBERT:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def encode(self, texts: object, **kwargs: object) -> object:
                calls.append({"texts": texts, **kwargs})
                return "embeddings"

        monkeypatch.setattr(embed, "_MODEL", None)
        monkeypatch.setattr("pylate.models.ColBERT", _FakeColBERT)
        monkeypatch.setattr(embed, "embed_device", lambda: "cpu")

        model = embed.load_embedder()
        assert isinstance(model, _FakeColBERT)
        assert model.kwargs["model_name_or_path"] == embed.MODEL_ID
        assert model.kwargs["device"] == "cpu"

        out = model.encode(["hello"], is_query=True, normalize_embeddings=True)
        assert out == "embeddings"
        assert calls == [{"texts": ["hello"], "is_query": True, "normalize_embeddings": True}]

    def test_lazy_cache_returns_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The model is loaded once and shared across callers."""
        construct_count = 0

        class _FakeColBERT:
            def __init__(self, **kwargs: object) -> None:
                nonlocal construct_count
                construct_count += 1

        monkeypatch.setattr(embed, "_MODEL", None)
        monkeypatch.setattr("pylate.models.ColBERT", _FakeColBERT)
        monkeypatch.setattr(embed, "embed_device", lambda: "cpu")

        first = embed.load_embedder()
        second = embed.load_embedder()
        assert first is second
        assert construct_count == 1
        assert embed._MODEL is first

    def test_lock_recheck_returns_model_set_by_other_thread(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The locked re-check returns a model another thread cached meanwhile."""

        class _FakeLock:
            def __enter__(self) -> _FakeLock:
                embed._MODEL = object()  # racer wins while we wait for the lock
                return self

            def __exit__(self, *args: object) -> bool:
                return False

        monkeypatch.setattr(embed, "_MODEL", None)
        monkeypatch.setattr(embed, "_lock", _FakeLock())
        monkeypatch.setattr("pylate.models.ColBERT", object)

        model = embed.load_embedder()
        assert embed._MODEL is model


def test_auto_device_cuda_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA branch of _auto_device is exercised when MPS is unavailable."""
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv(embed.EMBED_DEVICE_ENV, raising=False)
    assert embed.embed_device() == "cuda"


def test_auto_device_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU fallback of _auto_device when neither MPS nor CUDA is available."""
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(embed.EMBED_DEVICE_ENV, raising=False)
    assert embed.embed_device() == "cpu"


def test_model_load_error_is_a_runtime_error() -> None:
    assert issubclass(embed.ModelDownloadError, RuntimeError)


def _gated_error() -> BaseException:
    import httpx
    from huggingface_hub.errors import GatedRepoError

    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/x"))
    return GatedRepoError("401 Client Error. Cannot access gated repo", response=response)


def _missing_error() -> BaseException:
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/x"))
    return RepositoryNotFoundError("404 Client Error. Repository Not Found", response=response)


def test_model_load_message_handles_gated_missing_and_network_failures() -> None:
    gated = embed.model_load_message(_gated_error(), "google/embeddinggemma-300m")
    assert gated is not None
    assert "gated" in gated.lower()
    assert "huggingface-cli login" in gated
    assert "HF_TOKEN" in gated
    assert "google/embeddinggemma-300m" in gated

    missing = embed.model_load_message(_missing_error(), "nope/nope")
    assert missing is not None
    assert "Could not find model nope/nope" in missing

    network = embed.model_load_message(OSError("Connection reset by peer"))
    assert network is not None
    assert "Could not load model" in network

    # A genuine programming error must NOT be mislabeled as a model problem.
    assert embed.model_load_message(ValueError("bad shape")) is None
    assert embed.model_load_message(RuntimeError("database closed")) is None


def test_raise_model_error_only_raises_for_model_failures() -> None:
    with pytest.raises(embed.ModelDownloadError):
        embed.raise_model_error(_gated_error())

    # Non-model errors pass through untouched.
    embed.raise_model_error(ValueError("bad shape"))  # no raise


def test_model_is_cached_probe_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    # Returns a plain bool without raising, for a normal and a broken cache.
    assert isinstance(embed._model_is_cached("some/model"), bool)
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", None)
    assert embed._model_is_cached("some/model") is False


def _fake_cache(monkeypatch: pytest.MonkeyPatch, tmp_path, model_id: str) -> Path:
    """Point HF_HUB_CACHE at a scratch dir and return the repo's storage folder."""
    import huggingface_hub.constants

    repo_dir = tmp_path / f"models--{model_id.replace('/', '--')}"
    (repo_dir / "snapshots").mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    return repo_dir


def test_model_is_cached_false_when_never_downloaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """No snapshot dir at all → not cached."""
    _fake_cache(monkeypatch, tmp_path, "org/model")
    assert embed._model_is_cached("org/model") is False


def test_model_is_cached_false_for_partial_snapshot_without_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Interrupted download (config/tokenizer only, no weights) → not cached."""
    repo_dir = _fake_cache(monkeypatch, tmp_path, "org/model")
    snapshot = repo_dir / "snapshots" / "abcdef1234567890"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "vocab.txt").write_text("x")
    assert embed._model_is_cached("org/model") is False


def test_model_is_cached_false_when_tree_cache_lists_missing_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Tree cache (authoritative list) shows files missing from the snapshot."""
    repo_dir = _fake_cache(monkeypatch, tmp_path, "org/model")
    commit = "abcdef1234567890"
    snapshot = repo_dir / "snapshots" / commit
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    trees = repo_dir / "trees"
    trees.mkdir()
    (trees / f"{commit}.json").write_text(
        '{"format_version": 1, "files": {'
        '"config.json": {"size": 2, "blob_id": "x"}, '
        '"pytorch_model.bin": {"size": 3, "blob_id": "y"}}}'
    )
    assert embed._model_is_cached("org/model") is False


def test_model_is_cached_true_for_complete_tree_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Every tree-cache file present → cached."""
    repo_dir = _fake_cache(monkeypatch, tmp_path, "org/model")
    commit = "abcdef1234567890"
    snapshot = repo_dir / "snapshots" / commit
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "pytorch_model.bin").write_bytes(b"\x00")
    trees = repo_dir / "trees"
    trees.mkdir()
    (trees / f"{commit}.json").write_text(
        '{"format_version": 1, "files": {'
        '"config.json": {"size": 2, "blob_id": "x"}, '
        '"pytorch_model.bin": {"size": 3, "blob_id": "y"}}}'
    )
    assert embed._model_is_cached("org/model") is True


def test_model_is_cached_true_for_weights_without_tree_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """No tree cache, but the weights are present → cached (fallback path)."""
    repo_dir = _fake_cache(monkeypatch, tmp_path, "org/model")
    snapshot = repo_dir / "snapshots" / "abcdef1234567890"
    snapshot.mkdir()
    (snapshot / "pytorch_model.bin").write_bytes(b"\x00")
    assert embed._model_is_cached("org/model") is True


def test_model_is_cached_false_for_broken_weight_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A dangling weight symlink does not count as cached."""
    repo_dir = _fake_cache(monkeypatch, tmp_path, "org/model")
    snapshot = repo_dir / "snapshots" / "abcdef1234567890"
    snapshot.mkdir()
    target = repo_dir / "blobs" / "deadbeef"
    (snapshot / "pytorch_model.bin").symlink_to(target)  # dangling
    assert embed._model_is_cached("org/model") is False


def test_configure_model_loading_sets_offline_when_all_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached models load fully offline: no network call, no warning."""
    monkeypatch.setattr(embed, "_model_is_cached", lambda model_id: True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    embed.configure_model_loading("a/model", "b/model")

    import huggingface_hub.constants

    assert huggingface_hub.constants.HF_HUB_OFFLINE is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_configure_model_loading_stays_online_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing model keeps the normal one-time download path available."""
    monkeypatch.setattr(embed, "_model_is_cached", lambda model_id: model_id == "a/model")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    embed.configure_model_loading("a/model", "b/model")

    import huggingface_hub.constants

    assert huggingface_hub.constants.HF_HUB_OFFLINE is False
    assert "HF_HUB_OFFLINE" not in os.environ


def test_configure_model_loading_respects_user_offline_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-set offline flag is never cleared, even on a cache miss."""
    monkeypatch.setattr(embed, "_model_is_cached", lambda model_id: False)
    monkeypatch.setattr(embed, "_USER_OFFLINE_ENV", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    embed.configure_model_loading("a/model")

    import huggingface_hub.constants

    assert huggingface_hub.constants.HF_HUB_OFFLINE is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_configure_model_loading_disables_progress_and_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress bars are disabled and model advisories are filtered, always."""
    monkeypatch.setattr(embed, "_model_is_cached", lambda model_id: True)
    embed.configure_model_loading("a/model")

    from transformers.utils.logging import is_progress_bar_enabled

    assert is_progress_bar_enabled() is False
    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"

    # The unauthenticated-request advisory and the sentence-transformers
    # default-prompt note are dropped; other logs pass through.
    import io
    import logging

    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    for logger_name in ("huggingface_hub.utils._http", "sentence_transformers.base.model"):
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
    try:
        logging.getLogger("huggingface_hub.utils._http").warning(
            "You are sending unauthenticated requests to the HF Hub."
        )
        logging.getLogger("sentence_transformers.base.model").warning(
            "Default prompt name is set to 'query'."
        )
        logging.getLogger("huggingface_hub.utils._http").warning("some other warning")
    finally:
        for logger_name in ("huggingface_hub.utils._http", "sentence_transformers.base.model"):
            logging.getLogger(logger_name).removeHandler(handler)
    assert "unauthenticated requests" not in captured.getvalue()
    assert "Default prompt name" not in captured.getvalue()
    assert "some other warning" in captured.getvalue()


# ---------------------------------------------------------------------------
# ensure_model_downloaded tests
# ---------------------------------------------------------------------------


class TestEnsureModelDownloaded:
    """Prompt-driven download with a usecli progress bar shim."""

    def test_noop_when_model_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cached model → no network call to snapshot_download."""
        monkeypatch.setattr(embed, "_model_is_cached", lambda model_id: True)
        snapshot_called = False

        def _never_call(*args: object, **kwargs: object) -> object:
            nonlocal snapshot_called
            snapshot_called = True
            pytest.fail("snapshot_download must not be called for a cached model")
            return None

        monkeypatch.setattr("huggingface_hub.snapshot_download", _never_call)
        embed.ensure_model_downloaded("some/model", revision="abc123")
        assert not snapshot_called, "snapshot_download was called for cached model"

    def test_calls_snapshot_download_on_cache_miss(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Uncached model → delegates to snapshot_download with correct args."""
        called: dict[str, object] = {}
        monkeypatch.setattr(embed, "_model_is_cached", lambda model_id: False)

        def fake_snapshot(
            repo_id: str,
            revision: str | None = None,
            tqdm_class: object = None,
            local_files_only: bool = False,
            **kwargs: object,
        ) -> str:
            called.update(
                repo_id=repo_id,
                revision=revision,
                tqdm_class=tqdm_class,
                local_files_only=local_files_only,
            )
            return "/fake/snapshot"

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        embed.ensure_model_downloaded("test/model", revision="abc")
        assert called.get("repo_id") == "test/model"
        assert called.get("revision") == "abc"
        assert called.get("local_files_only") is False
        assert called.get("tqdm_class") is embed._UsecliTqdm

    def test_sets_quiet_on_shim_class(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The quiet flag is propagated to _UsecliTqdm._quiet before download."""
        monkeypatch.setattr(embed, "_model_is_cached", lambda model_id: False)
        monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: "")
        embed._UsecliTqdm._quiet = False
        embed.ensure_model_downloaded("m", quiet=True)
        assert embed._UsecliTqdm._quiet is True

    def test_propagates_exceptions_from_snapshot_download(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Errors from snapshot_download are not swallowed."""
        monkeypatch.setattr(embed, "_model_is_cached", lambda model_id: False)

        def _fail(*args: object, **kwargs: object) -> str:
            raise ConnectionError("Connection refused")

        monkeypatch.setattr("huggingface_hub.snapshot_download", _fail)
        with pytest.raises(ConnectionError, match="Connection refused"):
            embed.ensure_model_downloaded("bad/model")

    def test_usecli_tqdm_transfer_bar_creates_progress_bar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_UsecliTqdm creates a usecli ProgressBar when desc is 'Downloading bytes'."""
        from ssgrep.indexing.embed import _UsecliTqdm

        monkeypatch.setattr(_UsecliTqdm, "_quiet", False)
        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B", unit_scale=True)
        assert shim._bar is not None
        assert shim._bar.total == 100
        shim.close()

    def test_usecli_tqdm_non_transfer_bar_stays_silent(self) -> None:
        """_UsecliTqdm does not create a progress bar for non-transfer instances."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Reconstructing (incomplete total...)", total=100, unit="B")
        assert shim._bar is None


class TestUsecliTqdmShim:
    """Unit tests for the _UsecliTqdm shim class internals."""

    def test_getattr_delegates_to_tqdm(self) -> None:
        """__getattr__ forwards attribute access to the underlying tqdm."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Fetching 5 files", total=5)
        assert shim.total == 5
        assert shim.n == 0

    def test_set_postfix_str_is_noop(self) -> None:
        """set_postfix_str does nothing (rate info not rendered on usecli bar)."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        shim.set_postfix_str("10 MB/s")  # must not raise
        shim.close()

    def test_update_on_non_transfer_bar_does_nothing(self) -> None:
        """update on a non-transfer bar does not create a bar."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Fetching 5 files", total=5)
        shim.update(1)
        assert shim._bar is None
        assert shim.n == 1

    def test_update_transfer_bar_advances_progress(self) -> None:
        """update on a transfer bar advances the usecli bar and self.n."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        assert shim._bar is not None
        assert shim._bar.completed == 0

        shim.update(50)
        assert shim.n == 50
        assert shim._bar.completed == 50

        shim.update(50)
        assert shim.n == 100
        assert shim._bar is None  # completed → closed

    def test_update_syncs_growing_total(self) -> None:
        """update syncs the usecli bar's total when the underlying total grows."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        assert shim._bar is not None
        shim.total = 200  # snapshot adds more file sizes
        shim.update(50)
        assert shim._bar.total == 200

    def test_refresh_on_non_transfer_bar_is_noop(self) -> None:
        """refresh on a non-transfer bar returns immediately."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Reconstructing ...", total=100)
        shim.refresh()  # must not raise

    def test_refresh_on_stale_bar_returns_fast(self) -> None:
        """refresh returns early when the usecli bar is already exited."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        assert shim._bar is not None
        shim._bar.__exit__(None, None, None)
        shim.refresh()  # must not raise — _progress is None
        assert shim._bar is not None  # only __exit__'d, not nulled by our code yet

    def test_refresh_syncs_total_when_changed(self) -> None:
        """refresh syncs the usecli bar's total when the hard total changed."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        assert shim._bar is not None
        shim.total = 250
        shim.refresh()
        assert shim._bar.total == 250

    def test_refresh_closes_bar_when_complete(self) -> None:
        """refresh closes the usecli bar when n >= total."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        shim.n = 100
        shim.total = 100
        shim.refresh()
        assert shim._bar is None  # bar closed

    def test_set_description_updates_usecli_bar(self) -> None:
        """set_description updates the usecli bar description."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        assert shim._bar is not None
        shim.set_description("Downloading model")
        assert shim._bar.description == "Downloading model: "

    def test_set_description_complete_closes_bar(self) -> None:
        """set_description with 'complete' in the desc closes the bar."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        shim.set_description("Download complete")
        assert shim._bar is None

    def test_close_cleans_up_usecli_bar(self) -> None:
        """close exits the usecli bar and clears the reference."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Downloading bytes", total=100, unit="B")
        shim.close()
        assert shim._bar is None

    def test_close_on_non_transfer_bar_is_safe(self) -> None:
        """close on a bar-less shim does not raise."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Fetching 5 files", total=5)
        shim.close()  # must not raise

    def test_close_cleanup_when_disable_reset(self) -> None:
        """close still runs registry cleanup when disable was reset externally."""
        from ssgrep.indexing.embed import _UsecliTqdm

        shim = _UsecliTqdm(desc="Fetching 5 files", total=5)
        shim.disable = False
        shim.close()
        assert shim.disable is True  # set by close's cleanup path
        shim.close()  # idempotent


class TestResolvePoolFactor:
    """SSGREP_POOL_FACTOR parsing: default 2, allowed {1, 2, 3}."""

    def test_unset_env_defaults_to_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(embed.POOL_FACTOR_ENV, raising=False)
        assert embed.resolve_pool_factor() == 2

    def test_blank_env_defaults_to_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "   ")
        assert embed.resolve_pool_factor() == 2

    @pytest.mark.parametrize("value", ["1", "2", "3"])
    def test_allowed_values_pass_through(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, value)
        assert embed.resolve_pool_factor() == int(value)

    def test_one_disables_pooling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "1")
        assert embed.resolve_pool_factor() == 1

    def test_unparsable_value_falls_back_to_default_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, "not-a-number")
        with caplog.at_level("WARNING", logger="ssgrep.indexing.embed"):
            assert embed.resolve_pool_factor() == embed.DEFAULT_POOL_FACTOR
        assert any("SSGREP_POOL_FACTOR" in record.message for record in caplog.records)

    @pytest.mark.parametrize(
        ("raw", "clamped"),
        [("7", "3"), ("100", "3"), ("0", "1"), ("-4", "1")],
    )
    def test_out_of_range_values_clamp_to_nearest_allowed_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        raw: str,
        clamped: str,
    ) -> None:
        monkeypatch.setenv(embed.POOL_FACTOR_ENV, raw)
        with caplog.at_level("WARNING", logger="ssgrep.indexing.embed"):
            assert embed.resolve_pool_factor() == int(clamped)
        assert any("clamp" in record.message.lower() for record in caplog.records)
