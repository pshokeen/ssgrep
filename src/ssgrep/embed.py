"""Embedding encoder using model2vec.

Offline-first model loading: once the model is cached locally (the normal
case after the first run), loading it makes no network calls at all —
model2vec's own cache-first resolution (``force_download=False``) is what
guarantees this, not a network call that happens to fail over to the cache.
A cold load only reaches the network when no local snapshot exists yet (a
genuine first run), and that download is bounded by a hard wall-clock
timeout and wrapped so failures are actionable instead of a raw library
stack trace or an unbounded hang. See ``ModelDownloadError`` and
``_load_model()`` below, and the semantic-embedding spec's "Offline
Operation After First-Run Download" requirement.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

import numpy as np

MODEL_ID = "minishlab/potion-base-8M"
DIMENSION = 256

#: Pinned Hugging Face commit for MODEL_ID. Captured from the verified local
#: snapshot (refs/main) so every install resolves byte-identical model files;
#: an upstream force-push or re-upload cannot silently change embeddings and
#: invalidate published retrieval numbers or existing indexes.
MODEL_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"

#: Env var overriding the model-load deadline, in seconds. There was no
#: override at all, which is what turned a too-short default into a dead end.
MODEL_LOAD_TIMEOUT_ENV = "SSGREP_MODEL_LOAD_TIMEOUT"

# Wall-clock budget for loading the model, in seconds. Two budgets, because
# the two situations differ by three orders of magnitude and one number
# cannot serve both.
#
# huggingface_hub's own timeout knobs (etag_timeout / HF_HUB_ETAG_TIMEOUT,
# default 10s) are not a reliable bound on this call: the httpx client
# huggingface_hub==1.24.0 builds for its default transport is constructed
# with timeout=None (huggingface_hub/utils/_http.py::default_client_factory),
# so a black-holed connection can hang far longer than those knobs suggest —
# empirically ~25s before huggingface_hub itself gives up. Enforcing our own
# deadline on a background thread is the only way to bound this reliably.
#
# WARM: the model is already cached, the load is pure local disk I/O
# (measured 0.195-0.378s), and nothing touches the network. A tight bound is
# right here.
_WARM_LOAD_TIMEOUT_SECONDS = 30.0

# COLD: a genuine first-run download — ~30 MB with the restricted
# allow_patterns below (61 MB unrestricted) — not the "~8 MB" the
# README once claimed. A single 10s budget covered both cases and was therefore
# an implicit ~120 Mbit/s sustained-throughput requirement: measured against
# real huggingface.co, 25 / 50 / 100 Mbit/s all failed, and only 125 Mbit/s
# passed. So the very first command a paying buyer ran exited 1 and told
# them their network was unreachable, on a link that was fine. With no env
# var, flag, or config to raise it, there was nothing they could do.
_COLD_LOAD_TIMEOUT_SECONDS = 600.0


def _model_is_cached() -> bool:
    """True when a local snapshot exists, so no download is needed.

    Best-effort and never raises: an unavailable or restructured hub cache
    just means we assume a cold load and allow the longer budget, which is
    the safe direction — the tight budget must never be applied to a real
    download.
    """
    try:
        return _pinned_snapshot_dir() is not None
    except Exception:
        return False


def _pinned_snapshot_dir() -> Path | None:
    """The local snapshot directory for MODEL_REVISION, or None if absent.

    Best-effort and never raises (same rationale as _model_is_cached): a
    missing or restructured hub cache just means the pinned revision must be
    downloaded, which snapshot_download() then does explicitly.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        snapshot_dir = (
            Path(HF_HUB_CACHE)
            / f"models--{MODEL_ID.replace('/', '--')}"
            / "snapshots"
            / MODEL_REVISION
        )
        if (snapshot_dir / "config.json").exists() and (
            snapshot_dir / "model.safetensors"
        ).exists():
            return snapshot_dir
        return None
    except Exception:
        return None


def _load_timeout_seconds() -> float:
    """The deadline for this load: the env override, else warm/cold default."""
    override = os.environ.get(MODEL_LOAD_TIMEOUT_ENV, "").strip()
    if override:
        try:
            parsed = float(override)
        except ValueError:
            parsed = 0.0
        if parsed > 0:
            return parsed
    return _WARM_LOAD_TIMEOUT_SECONDS if _model_is_cached() else _COLD_LOAD_TIMEOUT_SECONDS


_model = None


class ModelDownloadError(RuntimeError):
    """The embedding model could not be loaded.

    Raised when there is no usable local cache and the first-run download
    fails or times out, instead of letting the underlying huggingface_hub /
    httpx exception (or an unbounded hang) reach the caller directly — per
    the semantic-embedding spec's "a failed download SHALL produce an
    actionable error... rather than a raw stack trace."
    """


def _load_model():
    """Load the StaticModel, offline-first, with a hard timeout and an
    actionable error on failure.

    ``force_download=False`` is what makes a warm-cache load genuinely
    offline: model2vec's own folder resolution
    (``model2vec.persistence.persistence._resolve_folder``) checks the local
    Hugging Face cache first (``maybe_get_cached_model_path``) and returns it
    directly with no network call at all when a snapshot is already present,
    only falling through to ``huggingface_hub.snapshot_download`` (a real
    network call) when nothing is cached. The default, ``force_download=True``
    (what ``StaticModel.from_pretrained`` used unconditionally before this
    fix), skips that local check entirely and always attempts
    ``snapshot_download`` first — that was the defect: every cold load (i.e.
    every fresh CLI process, since the model is a module-level singleton
    that does not survive across processes) made a genuine outbound HTTPS
    attempt, even long after the model was cached.

    The load runs on a background daemon thread so a hard timeout can be
    enforced regardless of what the underlying HTTP stack does internally
    (see ``_load_timeout_seconds()``). The thread is never joined on
    timeout: as a daemon it cannot block process exit, and it is simply
    abandoned if it never completes.
    """
    result: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            from huggingface_hub import snapshot_download
            from model2vec import StaticModel

            # On cold loads (first run), restrict download to exclude ONNX which is
            # ~28.8 MB and never used by StaticModel. This cuts the first-run download
            # from ~61 MB to ~30 MB. On warm loads (model already cached), we skip
            # this call entirely to stay offline (no network access on warm loads).
            #
            # The download is pinned to MODEL_REVISION and the model is loaded
            # from that revision's snapshot directory (StaticModel.from_pretrained
            # accepts a local path but no revision argument), so upstream changes
            # to the repo's main ref can never alter the embedding bytes.
            snapshot_dir = _pinned_snapshot_dir()
            if snapshot_dir is None:
                snapshot_dir = Path(
                    snapshot_download(
                        MODEL_ID,
                        revision=MODEL_REVISION,
                        force_download=False,
                        allow_patterns=[
                            "*.json",
                            "*.txt",
                            "*.md",
                            "model.safetensors",
                            ".gitattributes",
                        ],
                    )
                )

            model = StaticModel.from_pretrained(snapshot_dir, force_download=False)
        except Exception as exc:  # relayed to the waiting thread below, not raised here
            result.put(("error", exc))
        else:
            result.put(("ok", model))

    threading.Thread(target=worker, daemon=True, name="ssgrep-model-load").start()

    timeout = _load_timeout_seconds()
    try:
        status, payload = result.get(timeout=timeout)
    except queue.Empty:
        # Deliberately does NOT assert the network is unreachable. It usually
        # is not: huggingface_hub's download workers are non-daemon and are
        # joined at interpreter shutdown, so the transfer keeps running (and
        # normally completes) after this error is printed — which is also why
        # the process appears to hang silently for a while afterwards. Naming
        # a broken connection sent buyers to debug a network that was fine.
        raise ModelDownloadError(
            f"Timed out after {timeout:.0f}s loading the embedding model "
            f"({MODEL_ID}). ssgrep needs a one-time ~30 MB download from Hugging "
            f"Face on first run; if it is still in progress it will finish in the "
            f"background, and re-running this command will use it. If it does not, "
            f"check your connection — and HTTPS_PROXY/HTTP_PROXY, if you're behind "
            f"a proxy. To allow longer, set {MODEL_LOAD_TIMEOUT_ENV} to a number of "
            f"seconds."
        ) from None

    if status == "error":
        raise ModelDownloadError(
            f"Could not load the embedding model ({MODEL_ID}): {payload}. If this is the "
            f"first run, ssgrep needs a one-time download from Hugging Face — check your "
            f"network connection and proxy settings, then try again. If the model was "
            f"downloaded before, the local cache may be corrupt; removing "
            f"~/.cache/huggingface/hub/models--minishlab--potion-base-8M and retrying "
            f"will force a fresh download."
        ) from payload

    return payload


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIMENSION), dtype=np.float32)
    model = _get_model()
    embeddings = model.encode(texts)
    result = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    result = result / norms
    return result


def encode_sequence(texts: list[str]) -> list[np.ndarray]:
    """Per-token embeddings for each text, unpooled, unnormalized.

    Unlike encode() (one mean-pooled, L2-normalized vector per text -- the
    contract the index format and every existing vector-leg query depend
    on), this returns model2vec's raw per-token static embeddings: one
    (n_tokens, DIMENSION) array per input text, n_tokens varying per text
    (empty array for a text with no tokens). This is the building block for
    late-interaction (ColBERT-style MaxSim) scoring over a short candidate
    list -- see search/rerank.py -- never for the corpus-wide vector index,
    which stays exactly as before.

    Same underlying singleton model as encode(); no separate load, no extra
    network or disk I/O.
    """
    if not texts:
        return []
    model = _get_model()
    return model.encode_as_sequence(texts)


def get_model_info() -> tuple[str, int]:
    return MODEL_ID, DIMENSION
