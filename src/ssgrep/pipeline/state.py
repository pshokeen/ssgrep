"""Stable identity and location constants for the CocoIndex pipeline.

The ``ContextKey`` name strings are load-bearing: CocoIndex treats them as the
stable identity of providers and managed targets across runs (renaming one
silently breaks memoization and managed-table tracking). They are versioned so
a deliberate pipeline redesign can namespace fresh state.
"""

from __future__ import annotations

import os
from pathlib import Path

import cocoindex as coco
from cocoindex.connectors import lancedb

from ssgrep.indexing.lateon import ColBERTEmbedder
from ssgrep.store.paths import data_dir

#: Name of the single CocoIndex App that owns transcript ingestion.
APP_NAME = "SessionIndexV1"

#: LMDB path override; mirrors SSGREP_DATA_DIR / COCOINDEX_DB conventions.
LMDB_PATH_ENV = "SSGREP_COCOINDEX_DB"

#: Shared LanceDB connection (the same database LanceStore reads).
LANCE_DB = coco.ContextKey[lancedb.LanceAsyncConnection]("ssgrep_lance_v1")

#: The pylate-backed late-interaction embedder, bound to the pinned encoding
#: model. ``detect_change=True`` ties memoization to the provider's
#: ``__coco_memo_key__`` (model id, revision, device), so swapping the
#: embedding model invalidates memos and re-embeds (the required ``--rebuild``
#: re-embeds regardless).
EMBEDDER = coco.ContextKey[ColBERTEmbedder]("ssgrep_embedder_v1", detect_change=True)


def lmdb_path() -> Path:
    """The CocoIndex LMDB state file, inside the private application root."""
    override = os.environ.get(LMDB_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser().absolute()
    return data_dir() / "cocoindex" / "state.db"


__all__ = ["APP_NAME", "EMBEDDER", "LANCE_DB", "LMDB_PATH_ENV", "lmdb_path"]
