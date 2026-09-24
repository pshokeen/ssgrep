"""Pinned embedding model identity and PyLate loader for ssgrep.

ssgrep embeds transcripts with a late-interaction ColBERT model loaded through
PyLate: ``lightonai/answerai-colbert-small-v1``, a 96-dim per-token encoder.
``load_embedder()`` returns a PyLate ``models.ColBERT`` (a
``SentenceTransformer`` subclass) whose ``.encode(texts, is_query=...,
normalize_embeddings=True)`` produces the multi-vector embeddings used at both
index time and query time. This module owns the model identity, the environment
overrides, the actionable error surface for first-run downloads, and a ``tqdm``
shim that renders download progress through a usecli ``ProgressBar`` instead of
the default tqdm bar.

Re-ranking is no longer supported: the old cross-encoder provider has been
removed, and setting ``SSGREP_RERANK_MODEL`` is an explicit error at import
time rather than a silent no-op.

``embed_device()`` auto-detects MPS (Apple Silicon) or CUDA (NVIDIA) before
falling back to CPU, so the embedder uses the fastest available device;
``configure_model_loading()`` also sizes torch's thread pool for the host
machine.

Offline operation after first-run download is inherited from
sentence-transformers / huggingface_hub: once the snapshot is cached, loading
makes no network call at all (``snapshot_download`` checks the local cache
first and only falls through to the network when a file is genuinely
missing). The default embedding download requires no Hugging Face account.
``SSGREP_EMBED_MODEL`` can still point at a *gated* repository, whose first
download additionally requires accepting its terms on the Hub and logging in
(``huggingface-cli login`` or ``HF_TOKEN``); a missing grant surfaces as
``ModelDownloadError`` with those exact steps instead of a raw
``GatedRepoError``.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm as _tqdm_base

if TYPE_CHECKING:
    from usecli import ProgressBar

logger = logging.getLogger(__name__)

#: Default embedding model (late-interaction ColBERT, 96-d per token). Not
#: gated on the Hub, so the first download needs no Hugging Face account.
#: Smaller than LateOn (the previous default) at the same 299-token window,
#: so it embeds faster.
DEFAULT_MODEL_ID = "lightonai/answerai-colbert-small-v1"

#: Pinned Hugging Face commit for DEFAULT_MODEL_ID. Upstream ``main`` is not
#: stable (repos in this family have force-pushed the weights between commits
#: before), so an unpinned load can resolve to a revision that carries only
#: tokenizer/config files. This commit was verified to resolve the full,
#: loadable snapshot (model.safetensors + Dense projection, 96-d ColBERT).
DEFAULT_MODEL_REVISION = "e507cd12947a2b4b52201d150967df3c19a90590"

#: Env var overriding the embedding model id (for testing and swap-in models).
EMBED_MODEL_ENV = "SSGREP_EMBED_MODEL"

#: Deprecated env var for the removed re-ranker. Re-ranking is no longer
#: supported; setting this is an explicit error at import time (see below).
RERANK_MODEL_ENV = "SSGREP_RERANK_MODEL"

#: Deprecated env var for the removed hybrid blend weight. Hybrid retrieval
#: (and its RRF/blend weighting) is no longer supported; setting this is an
#: explicit error at import time (see below).
RERANK_BLEND_ENV = "SSGREP_RERANK_BLEND"

#: Env var overriding the torch device for embedding (e.g. ``mps`` on Apple
#: Silicon, ``cuda``, ``cpu``). When unset, ssgrep auto-detects the fastest
#: available device (MPS on Apple Silicon, CUDA on NVIDIA, else CPU).
EMBED_DEVICE_ENV = "SSGREP_EMBED_DEVICE"

#: Env var overriding the torch CPU thread count for embedding. When unset,
#: ssgrep sizes the thread pool from the host core count (bounded so
#: interactive latency stays responsive).
NUM_THREADS_ENV = "SSGREP_NUM_THREADS"

#: Env var overriding the document-side token pooling factor. Pooling merges
#: each document's token embeddings into ~1/N cluster means (ssgrep's
#: Ward-linkage pooling, applied after the model call), cutting stored token
#: vectors ~N-fold at published ~100% (N=2) / ~99% (N=3) retrieval-quality
#: retention. DOCUMENT embeddings only: queries are never pooled regardless
#: of this setting.
POOL_FACTOR_ENV = "SSGREP_POOL_FACTOR"

#: Default pooling factor (2 = PyLate's recommended setting; 1 disables).
DEFAULT_POOL_FACTOR = 2

#: Allowed pooling factors. Unparsable values fall back to
#: ``DEFAULT_POOL_FACTOR`` with a warning; other integers clamp to the
#: nearest allowed value with a warning.
ALLOWED_POOL_FACTORS = (1, 2, 3)


def resolve_pool_factor() -> int:
    """The pooling factor from ``SSGREP_POOL_FACTOR`` (default 2).

    Invalid values never abort an index run: an unparsable value warns and
    uses the default, and an out-of-range integer clamps to the nearest of
    ``(1, 2, 3)`` so a typo degrades to a documented setting instead of a
    failed rebuild.
    """
    raw = os.environ.get(POOL_FACTOR_ENV, "").strip()
    if not raw:
        return DEFAULT_POOL_FACTOR
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using default %d.",
            POOL_FACTOR_ENV,
            raw,
            DEFAULT_POOL_FACTOR,
        )
        return DEFAULT_POOL_FACTOR
    if value in ALLOWED_POOL_FACTORS:
        return value
    clamped = min(ALLOWED_POOL_FACTORS, key=lambda allowed: abs(allowed - value))
    logger.warning(
        "%s=%d is outside {%s}; clamping to %d.",
        POOL_FACTOR_ENV,
        value,
        ", ".join(str(allowed) for allowed in ALLOWED_POOL_FACTORS),
        clamped,
    )
    return clamped


def _reject_deprecated_env(var: str, message: str) -> None:
    """Raise when a deprecated env var is set (removed feature, no silent no-op)."""
    if os.environ.get(var, "").strip():
        raise RuntimeError(message)


#: Re-ranking was removed from ssgrep. A user who still sets
#: ``SSGREP_RERANK_MODEL`` gets an explicit error instead of a silent no-op.
_reject_deprecated_env(
    RERANK_MODEL_ENV,
    f"{RERANK_MODEL_ENV} is set but re-ranking is no longer supported by "
    f"ssgrep. Unset {RERANK_MODEL_ENV} to continue.",
)

#: Hybrid retrieval (and its RRF/blend weighting) was removed from ssgrep. A
#: user who still sets ``SSGREP_RERANK_BLEND`` gets an explicit error instead
#: of a silent no-op.
_reject_deprecated_env(
    RERANK_BLEND_ENV,
    f"{RERANK_BLEND_ENV} is set but hybrid retrieval is no longer supported "
    f"by ssgrep. Unset {RERANK_BLEND_ENV} to continue.",
)


def unit_mean_vector(matrix: Any) -> np.ndarray:
    """L2-normalized mean of a ``(num_tokens, DIMENSION)`` token matrix.

    The single-vector proxy behind two-stage search (``SSGREP_TWO_STAGE``):
    cosine distance between the chunk's and the query's mean vectors
    approximates MaxSim cheaply for candidate selection. Computed from
    already-in-memory encode output — never an extra model call.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    mean = values.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean if norm == 0.0 else (mean / norm).astype(np.float32)


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


#: Env var overriding the embedding vector dimension. Needed when
#: ``SSGREP_EMBED_MODEL`` points at a model whose output dimension differs
#: from the default (e.g. ``Qwen/Qwen3-Embedding-0.6B`` is 1024-d): the Lance
#: vector column is declared from this constant, so model and dimension must
#: agree.
EMBED_DIMENSION_ENV = "SSGREP_EMBED_DIMENSION"

try:
    DIMENSION = int(_env_or_default(EMBED_DIMENSION_ENV, "96"))
except ValueError:  # pragma: no cover - defensive
    DIMENSION = 96

MODEL_ID = _env_or_default(EMBED_MODEL_ENV, DEFAULT_MODEL_ID)

#: Deprecated: re-ranking is no longer supported. Kept only so the store
#: module can still import the name; setting ``SSGREP_RERANK_MODEL`` raises at
#: import time above.
RERANK_MODEL_ID = ""

#: Resolved revision for the active embedding model: the pinned commit for the
#: default model, otherwise ``""`` (unpinned, upstream ``main``).
MODEL_REVISION = DEFAULT_MODEL_REVISION if MODEL_ID == DEFAULT_MODEL_ID else ""


def _auto_device() -> str:
    """The fastest torch device available on this machine (mps > cuda > cpu)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - defensive
        pass
    return "cpu"


def embed_device() -> str:
    """Torch device for embedding and re-ranking: env override, else best available.

    ``SSGREP_EMBED_DEVICE`` wins when set; otherwise MPS on Apple Silicon,
    CUDA on NVIDIA, else CPU. Auto-detection is what makes re-ranking use the
    GPU by default instead of the old hardcoded CPU fallback.
    """
    return _env_or_default(EMBED_DEVICE_ENV, _auto_device())


def _default_thread_count() -> int:
    """A performant torch CPU thread count: bounded by cores, capped at 8."""
    cores = os.cpu_count() or 4
    return min(cores, 8)


def configure_threads() -> None:
    """Size torch's intra-op thread pool for the host machine.

    Respects ``SSGREP_NUM_THREADS``; otherwise uses up to 8 threads so CPU
    inference (embedding and re-ranking) is parallelized without saturating
    every core. Sets the OMP/MKL env vars too, because BLAS backends read
    them at import time. Idempotent.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - defensive
        return
    raw = _env_or_default(NUM_THREADS_ENV, str(_default_thread_count()))
    try:
        count = int(raw.strip())
    except ValueError:  # pragma: no cover - defensive
        count = _default_thread_count()
    count = max(1, min(count, 64))
    os.environ.setdefault("OMP_NUM_THREADS", str(count))
    os.environ.setdefault("MKL_NUM_THREADS", str(count))
    torch.set_num_threads(count)


#: Lazily-loaded PyLate ColBERT embedder singleton (see ``load_embedder``).
_MODEL: Any | None = None

#: Guards ``_MODEL`` so concurrent callers share one loaded model.
_lock = threading.Lock()


def load_embedder() -> Any:
    """Load (once) and return the PyLate ColBERT embedder for ``MODEL_ID``.

    Returns a PyLate ``models.ColBERT`` (a ``SentenceTransformer`` subclass)
    exposing ``.encode(texts, is_query=..., normalize_embeddings=True)`` for
    both index-time and query-time use. Lazy-cached in a module-level
    singleton guarded by a lock so concurrent callers share one loaded model.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _lock:
        if _MODEL is not None:
            return _MODEL
        from pylate import models

        _MODEL = models.ColBERT(
            model_name_or_path=MODEL_ID,
            revision=MODEL_REVISION or None,
            device=embed_device(),
        )
    return _MODEL


class ModelDownloadError(RuntimeError):
    """A local embedding or re-ranker model could not be loaded.

    Raised instead of letting the underlying huggingface_hub / transformers
    exception reach the caller directly when the failure is a first-run
    download problem (gated repo, missing auth, network, or a corrupt local
    cache) — per the semantic-embedding contract's "a failed download SHALL
    produce an actionable error... rather than a raw stack trace".
    """


def _huggingface_errors() -> tuple[type[BaseException], ...] | None:
    """Known huggingface_hub failure types, or None if the package is absent."""
    try:
        import huggingface_hub.errors as errors
    except Exception:  # pragma: no cover - defensive
        return None
    return (
        errors.GatedRepoError,
        errors.RepositoryNotFoundError,
        errors.LocalEntryNotFoundError,
        errors.EntryNotFoundError,
    )


def model_load_message(exc: BaseException, model_id: str | None = None) -> str | None:
    """An actionable message when ``exc`` is a local model-load failure.

    Returns ``None`` when the exception is not a recognizable load/download
    problem (the caller should re-raise it as-is instead of mislabeling a
    genuine bug as a model problem).
    """
    model_id = model_id or MODEL_ID
    hf = _huggingface_errors()
    if hf is not None and isinstance(exc, hf[0]):  # GatedRepoError
        return (
            f"Model {model_id} is gated on Hugging Face. Open "
            f"https://huggingface.co/{model_id} in a browser, accept the model's "
            f"terms of use, then log in locally with `huggingface-cli login` (or set "
            f"the HF_TOKEN environment variable) and run the command again. ssgrep "
            f"needs this one-time download so the model is cached and later runs "
            f"are fully offline."
        )
    if hf is not None and isinstance(exc, hf[1:]):  # missing/offline repo or file
        return (
            f"Could not find model {model_id} in the local Hugging Face cache or on "
            f"the Hub: {exc}. If you set SSGREP_EMBED_MODEL, "
            f"verify the id — otherwise check that "
            f"https://huggingface.co/{model_id} exists and that your network can "
            f"reach huggingface.co (HTTPS_PROXY/HTTP_PROXY are honored)."
        )
    if isinstance(exc, (OSError, ConnectionError)) or "offline" in str(exc).lower():
        return (
            f"Could not load model {model_id}: {exc}. The model must be downloaded "
            f"at least once (a local cache miss or a corrupt local snapshot). Check "
            f"your network connection and proxy settings, or delete the cached "
            f"snapshot under ~/.cache/huggingface/hub/models--{model_id.replace('/', '--')} "
            f"and retry to force a fresh download."
        )
    return None


def raise_model_error(exc: BaseException, model_id: str | None = None) -> None:
    """Raise ``ModelDownloadError`` when ``exc`` is a model load failure."""
    message = model_load_message(exc, model_id)
    if message is not None:
        raise ModelDownloadError(message) from exc


def _snapshot_is_complete(snapshot: Path, repo_dir: Path) -> bool:
    """Whether ``snapshot`` (a ``snapshots/<commit>`` dir) holds every file.

    The per-commit tree cache (``trees/<commit>.json``) written by
    ``snapshot_download`` is the authoritative expected-file list; when it is
    absent (e.g. a snapshot copied in manually), fall back to requiring the
    model weights — the files a download completes last, so their presence is
    a safe completeness proxy. Broken symlinks count as missing.
    """
    try:
        from huggingface_hub._tree_cache import read_tree_cache

        tree = read_tree_cache(str(repo_dir), snapshot.name)
        if tree is not None:
            return all((snapshot / name).exists() for name in tree)
    except Exception:  # pragma: no cover - defensive
        pass
    return (
        any(p.is_file() for p in snapshot.glob("*.safetensors"))
        or any(p.is_file() for p in snapshot.glob("*.bin"))
        or any(p.is_file() for p in snapshot.glob("*.onnx"))
    )


def _model_is_cached(model_id: str) -> bool:
    """Whether a complete, loadable snapshot for ``model_id`` exists locally.

    A snapshot counts as cached only when every file listed in the commit's
    tree cache is present (or, without a tree cache, when the model weights
    are). An interrupted download that left a partial snapshot therefore
    returns ``False``, so callers re-download the missing files instead of
    going offline and failing to load. Never raises.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_dir = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            return False
        return any(
            entry.is_dir() and _snapshot_is_complete(entry, repo_dir)
            for entry in snapshots.iterdir()
        )
    except Exception:
        return False


#: Server-sent advisory huggingface_hub logs on every anonymous Hub request.
#: It is emitted through the module logger (not ``warnings.warn``), so the
#: only reliable suppression is a logging filter on that exact logger.
_HF_UNAUTHENTICATED_WARNING = "unauthenticated requests to the HF Hub"

#: One-time sentence-transformers advisory for models whose config pins a
#: default prompt (Qwen rerankers). Irrelevant to ssgrep's usage.
_ST_DEFAULT_PROMPT_WARNING = "Default prompt name is set to"

#: Advisory loggers ssgrep silences (module loggers; parent filters do not
#: apply to child records, so each emitter must be filtered directly).
_QUIET_LOGGERS = (
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "sentence_transformers.base.model",
)


class _ModelNoiseFilter(logging.Filter):
    """Drop model-loading advisories from Hugging Face / sentence-transformers.

    The unauthenticated-request message is a server-driven ``X-HF-Warning``
    header huggingface_hub echoes at WARNING level; the default-prompt note is
    a one-time config advisory. Both are noise for ssgrep, which downloads
    public models without an account and is deliberately offline once cached.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _HF_UNAUTHENTICATED_WARNING in message:
            return False
        return _ST_DEFAULT_PROMPT_WARNING not in message


def _suppress_model_noise() -> None:
    """Idempotently attach the advisory filter to model-loading loggers."""
    for logger_name in _QUIET_LOGGERS:
        logger = logging.getLogger(logger_name)
        if not any(isinstance(f, _ModelNoiseFilter) for f in logger.filters):
            logger.addFilter(_ModelNoiseFilter())


#: Whether the user explicitly set an offline env var before ssgrep managed
#: it. A genuinely offline machine must stay offline even when the cache probe
#: disagrees, so ``_set_hf_offline`` never clears a user-set flag.
_USER_OFFLINE_ENV = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))


def _set_hf_offline(offline: bool) -> None:
    """Force huggingface_hub offline mode (env var + runtime constant).

    ``HF_HUB_OFFLINE`` is read into ``huggingface_hub.constants`` at import
    time, so setting the env var alone is insufficient once the library is
    loaded; patching the module attribute makes ``is_offline_mode()`` honor
    the flag immediately. ``TRANSFORMERS_OFFLINE`` is read by transformers'
    own hub helpers, which consult it before calling into huggingface_hub.
    """
    effective = offline or _USER_OFFLINE_ENV
    if effective:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    elif not _USER_OFFLINE_ENV:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = effective
    except Exception:  # pragma: no cover - defensive
        pass


def configure_model_loading(*model_ids: str) -> None:
    """Prefer the local cache and silence Hugging Face load/download output.

    Call once per process before the first model load, with the ids that
    operation will use (``MODEL_ID`` for indexing). Idempotent and safe to
    call from multiple entry points.

    - Disables transformers' "Loading weights" tqdm bar and huggingface_hub's
      download bars.
    - Suppresses the "You are sending unauthenticated requests to the HF Hub"
      advisory huggingface_hub echoes from the server.
    - When every given model is already cached, forces offline mode so loading
      makes no network call at all (no HEAD/update checks, no download bars,
      no warning). On a cache miss the normal one-time download proceeds, with
      the bars and that warning still suppressed.
    """
    # 0. Size torch's thread pool for this host before any model loads.
    configure_threads()

    # 1. Hide download/load progress (covers both transformers and HF Hub).
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()  # also disables huggingface_hub's bars
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # pragma: no cover - defensive
        pass

    # 2. Drop the model-loading advisories (HF + sentence-transformers).
    _suppress_model_noise()

    # 3. Cache-first: offline when everything needed is already local.
    offline = bool(model_ids) and all(_model_is_cached(model_id) for model_id in model_ids)
    _set_hf_offline(offline)


# ---------------------------------------------------------------------------
# tqdm → usecli progress bar shim
# ---------------------------------------------------------------------------


class _UsecliTqdm(_tqdm_base):
    """tqdm subclass that renders download progress through usecli.

    Pass as ``tqdm_class`` to ``huggingface_hub.snapshot_download`` so the
    byte-transfer progress bar is shown with a usecli ``ProgressBar`` instead
    of the native tqdm bar.

    Only instances whose ``desc`` contains ``"Downloading bytes"`` (the
    snapshot-level transfer bar) create a visible usecli bar; all other
    instances (reconstruct bar, file-count bar from ``thread_map``, and
    per-file ``_AggregatedTqdm`` instances) are silent.

    Not a subclass of ``huggingface_hub.utils.tqdm`` so ``_create_progress_bar``
    does *not* inject ``disable=True`` — the usecli bar renders even when
    HF's own progress bars are programmatically suppressed.
    """

    _quiet: bool = False  # overridden per download in ensure_model_downloaded

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from usecli import ProgressBar

        desc_value = kwargs.get("desc", "") or ""
        super().__init__(*args, disable=True, **kwargs)
        if desc_value:
            self.desc = desc_value
        self.gui = False
        self.nrows = None  # not set in tqdm's disable path; needed by _decr_instances
        self._bar: ProgressBar | None = None
        if "Downloading bytes" in desc_value:
            self._bar = ProgressBar(
                total=max(self.total or 1, 1),
                description="Downloading embedding model",
                quiet=_UsecliTqdm._quiet,
            )
            self._bar.__enter__()

    def refresh(self, *args: Any, **kwargs: Any) -> None:
        if self._bar is None:
            return
        if self._bar._progress is None or self._bar._task_id is None:
            return
        tqdm_total = self.total or 0
        tqdm_n = self.n
        if tqdm_total != self._bar.total:
            self._bar._progress.update(self._bar._task_id, total=tqdm_total)
            self._bar.total = tqdm_total
        if tqdm_total > 0 and tqdm_n >= tqdm_total:
            self._bar.__exit__(None, None, None)
            self._bar = None

    def update(self, n: int = 1) -> None:
        self.n += n
        if self._bar is None:
            return
        tqdm_total = self.total or 0
        if tqdm_total and self._bar._progress is not None and self._bar._task_id is not None:
            if tqdm_total != self._bar.total:
                self._bar._progress.update(self._bar._task_id, total=tqdm_total)
                self._bar.total = tqdm_total
        self._bar.update(completed=self.n)
        if tqdm_total > 0 and self.n >= tqdm_total:
            self._bar.__exit__(None, None, None)
            self._bar = None

    def close(self) -> None:
        if self._bar is not None:
            self._bar.__exit__(None, None, None)
            self._bar = None
        if self.disable:
            return
        self.disable = True
        self._decr_instances(self)

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        self.desc = (desc + ": ") if desc else ""
        if self._bar is not None:
            self._bar.update(description=self.desc)
            if desc and "complete" in desc.lower():
                self._bar.__exit__(None, None, None)
                self._bar = None

    def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
        pass


def ensure_model_downloaded(
    model_id: str,
    revision: str | None = None,
    *,
    quiet: bool = False,
    description: str = "Downloading model",
) -> None:
    """Download the snapshot for ``model_id`` if not already cached.

    Shows a usecli ``ProgressBar`` during the download (via ``_UsecliTqdm``).
    When the model is already in the local HF cache the call is a no-op and no
    progress is shown.  Progress is suppressed when ``quiet=True``, which is
    auto-detected by usecli (JSON mode / non-TTY) even when ``quiet=False``.
    """
    if _model_is_cached(model_id):
        return

    from huggingface_hub import snapshot_download

    _UsecliTqdm._quiet = quiet

    snapshot_download(  # type: ignore
        repo_id=model_id,
        revision=revision,
        tqdm_class=_UsecliTqdm,
        local_files_only=False,
    )


__all__ = [
    "ALLOWED_POOL_FACTORS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_POOL_FACTOR",
    "DIMENSION",
    "EMBED_DEVICE_ENV",
    "EMBED_DIMENSION_ENV",
    "EMBED_MODEL_ENV",
    "MODEL_ID",
    "MODEL_REVISION",
    "ModelDownloadError",
    "NUM_THREADS_ENV",
    "POOL_FACTOR_ENV",
    "RERANK_MODEL_ENV",
    "RERANK_MODEL_ID",
    "configure_model_loading",
    "configure_threads",
    "embed_device",
    "ensure_model_downloaded",
    "load_embedder",
    "model_load_message",
    "raise_model_error",
    "resolve_pool_factor",
    "unit_mean_vector",
]
