"""Tests for embedding model download optimization and revision pinning.

Verifies that restricting the model download to exclude ONNX files
produces identical embeddings and successfully loads the model, and that
the isolated cache each test downloads into is the cache the model is
actually loaded from (each test loads via the snapshot path returned by
its own snapshot_download call — never the developer's global cache).

NOTE: All tests in this module are marked @pytest.mark.network and are
excluded from default test runs via pytest.ini_options addopts. Run them
with: `uv run pytest -m network`

These tests are critical: they are the only guard against a silent embedding
model change. The embedding vectors produced by the restricted model download
must be bit-identical to the full download, as MAIN_SESSION_BOOST and the
committed eval numbers in test_published_numbers.py are pinned to this exact
model version. Deleting or disabling these tests would silently break search
quality if the model version or download filter ever changes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

#: The exact allow_patterns used by src/ssgrep/embed.py's cold-load download.
RESTRICTED_PATTERNS = [
    "*.json",
    "*.txt",
    "*.md",
    "model.safetensors",
    ".gitattributes",
]


def _download(cache_dir: str, restricted: bool) -> str:
    """Download the pinned model revision into ``cache_dir``.

    Returns the local snapshot directory the files landed in, so callers can
    load THAT snapshot rather than whatever the global cache resolves to.
    """
    from huggingface_hub import snapshot_download

    from ssgrep.embed import MODEL_ID, MODEL_REVISION

    kwargs: dict = {
        "cache_dir": cache_dir,
        "revision": MODEL_REVISION,
        "force_download": True,
    }
    if restricted:
        kwargs["allow_patterns"] = RESTRICTED_PATTERNS
    return snapshot_download(MODEL_ID, **kwargs)


class TestEmbeddingDownloadOptimization:
    """Verify download optimization (excluding ONNX) doesn't change embeddings."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_restricted_download_produces_identical_embeddings(self):
        """Download with allow_patterns exclusion should produce bit-identical embeddings.

        This is the critical gate: if embeddings change even slightly,
        the optimization breaks retrieval quality pinned in test_published_numbers.py.
        Each model is loaded from the snapshot path its own download produced,
        so the comparison is genuinely restricted-download vs full-download.
        """
        from model2vec import StaticModel

        test_texts = [
            "hello world",
            "how did I solve that before",
            "embedding model",
            "semantic search",
        ]

        with tempfile.TemporaryDirectory() as tmpdir_restricted:
            snapshot_path = _download(tmpdir_restricted, restricted=True)
            model_restricted = StaticModel.from_pretrained(snapshot_path, force_download=False)
            embeddings_restricted = np.array(model_restricted.encode(test_texts), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir_full:
            snapshot_path = _download(tmpdir_full, restricted=False)
            model_full = StaticModel.from_pretrained(snapshot_path, force_download=False)
            embeddings_full = np.array(model_full.encode(test_texts), dtype=np.float32)

        assert embeddings_restricted.shape == embeddings_full.shape, (
            f"Shape mismatch: restricted {embeddings_restricted.shape} vs "
            f"full {embeddings_full.shape}"
        )

        max_diff = np.abs(embeddings_restricted - embeddings_full).max()
        mean_diff = np.abs(embeddings_restricted - embeddings_full).mean()

        # Allow tiny float tolerance (1e-6) but not corpus sensitivity (0.012)
        tolerance = 1e-6
        assert max_diff < tolerance, (
            f"Embeddings differ more than float tolerance: max_diff={max_diff} "
            f"(tolerance={tolerance}). This breaks retrieval quality. "
            f"Restricted: {embeddings_restricted[0, :5]}\n"
            f"Full:       {embeddings_full[0, :5]}"
        )
        assert (
            mean_diff < tolerance
        ), f"Mean embedding diff is {mean_diff}, exceeds tolerance {tolerance}"

    @pytest.mark.slow
    @pytest.mark.network
    def test_model_loads_from_cold_cache_with_restricted_download(self):
        """Model must load from the exact cache the restricted download populated."""
        from model2vec import StaticModel

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = _download(tmpdir, restricted=True)

            # Load from the snapshot the download above produced — NOT the
            # global cache, which may hold an unrelated (full) snapshot.
            model = StaticModel.from_pretrained(snapshot_path, force_download=False)

            embeddings = model.encode(["test"])
            assert embeddings.shape == (1, 256), f"Expected shape (1, 256), got {embeddings.shape}"

    @pytest.mark.slow
    @pytest.mark.network
    def test_verify_onnx_not_downloaded_with_restriction(self):
        """Verify that allow_patterns truly excludes ONNX files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _download(tmpdir, restricted=True)

            onnx_found = []
            safetensors_found = []
            for root, _dirs, files in os.walk(tmpdir):
                for f in files:
                    if "onnx" in f or f.endswith(".onnx"):
                        onnx_found.append(os.path.join(root, f))
                    if f.endswith(".safetensors"):
                        safetensors_found.append(os.path.join(root, f))

            assert len(onnx_found) == 0, (
                f"ONNX files should not be downloaded with allow_patterns restriction, "
                f"but found: {onnx_found}"
            )
            assert (
                len(safetensors_found) > 0
            ), "model.safetensors should be downloaded but was not found"

    @pytest.mark.slow
    @pytest.mark.network
    def test_offline_reload_from_isolated_cache(self):
        """A restricted download must be reloadable fully offline.

        Populates an isolated cache with the pinned restricted snapshot, then
        runs ssgrep's own embed path in a subprocess with HF_HUB_OFFLINE=1 and
        HF_HUB_CACHE pointed at that cache. This proves the shipped download
        filter leaves a cache that ssgrep can cold-start from with zero
        network access — the exact situation of a buyer's second run.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _download(tmpdir, restricted=True)

            script = (
                "from ssgrep import embed\n"
                "vecs = embed.encode(['offline reload test'])\n"
                "assert vecs.shape == (1, embed.DIMENSION), vecs.shape\n"
                "print('OFFLINE-RELOAD-OK')\n"
            )
            env = dict(os.environ)
            env["HF_HUB_OFFLINE"] = "1"
            env["HF_HUB_CACHE"] = tmpdir
            # Belt and braces: make sure nothing falls back to the real cache.
            env.pop("HF_HOME", None)

            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            assert result.returncode == 0, (
                f"offline reload failed (exit {result.returncode}).\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "OFFLINE-RELOAD-OK" in result.stdout
