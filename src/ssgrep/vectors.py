"""Memory-mapped vector store."""

from __future__ import annotations

import io
import mmap
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ssgrep import embed


@dataclass
class VectorStore:
    path: Path
    dimension: int
    file: mmap.mmap | None = None
    array: np.ndarray | None = None
    row_count: int = 0
    _file_handle: io.BufferedRandom | None = None  # Keep file handle open
    _closed: bool = False


def open_vectors(path: Path, dimension: int = embed.DIMENSION) -> VectorStore:
    store = VectorStore(path=path, dimension=dimension)
    if path.exists() and path.stat().st_size > 0:
        store.row_count = path.stat().st_size // (dimension * 4)
        f = open(path, "r+b")
        store._file_handle = f
        store.file = mmap.mmap(f.fileno(), 0)
        store.array = np.frombuffer(store.file, dtype=np.float32).reshape(-1, dimension)
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        path.touch()
    return store


def append(store: VectorStore, vectors: np.ndarray) -> list[int]:
    if store._closed:
        raise ValueError("Cannot append to closed VectorStore")

    n = vectors.shape[0]
    start_row = store.row_count

    with open(store.path, "ab") as f:
        f.write(vectors.astype(np.float32).tobytes())

    store.row_count += n

    # Release numpy array export before closing mmap
    store.array = None
    if store.file:
        store.file.close()
    # Close old file handle before opening new one
    if store._file_handle:
        store._file_handle.close()

    file_handle = open(store.path, "r+b")
    store._file_handle = file_handle
    store.file = mmap.mmap(file_handle.fileno(), 0)
    store.array = np.frombuffer(store.file, dtype=np.float32).reshape(-1, store.dimension)

    return list(range(start_row, start_row + n))


def close(store: VectorStore) -> None:
    """Release all resources held by the store."""
    store.array = None
    if store.file:
        store.file.close()
        store.file = None
    if store._file_handle:
        store._file_handle.close()
        store._file_handle = None
    store._closed = True


def read(store: VectorStore, rows: list[int]) -> np.ndarray:
    if store.array is None:
        raise ValueError("Store not initialized")
    return store.array[rows]


def cosine_top_k(
    store: VectorStore,
    query: np.ndarray,
    k: int = 20,
    candidate_rows: list[int] | None = None,
) -> list[tuple[int, float]]:
    if store.array is None or store.row_count == 0:
        return []

    if candidate_rows is not None:
        matrix = store.array[candidate_rows]
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        matrix_norm = matrix / norms
        scores = matrix_norm @ query_norm
        top_indices = np.argsort(scores)[::-1][:k]
        return [(candidate_rows[i], float(scores[i])) for i in top_indices]
    else:
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(store.array, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        matrix_norm = store.array / norms
        scores = matrix_norm @ query_norm
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices]


def detect_orphans(store: VectorStore, valid_rows: set[int]) -> list[int]:
    """Find vector rows that no chunk references.

    Returns list of orphaned row indices (vector rows with no owning chunk).
    """
    return [i for i in range(store.row_count) if i not in valid_rows]


def validate_alignment(db_path: Path, vec_path: Path) -> tuple[bool, str]:
    """Validate that chunks and vectors are mutually aligned.

    Checks:
    1. Every chunk has a valid vec_row (within bounds of vectors.f32)
    2. Every vector row in vectors.f32 is referenced by some chunk

    Returns: (is_valid, error_message)
    """
    import sqlite3

    try:
        if not db_path.exists():
            return True, ""

        conn = sqlite3.connect(str(db_path))

        # Check if vectors file exists
        if not vec_path.exists():
            # If index.db has chunks but no vectors file, that's corruption
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.close()
            if chunk_count > 0:
                return False, "index.db has chunks but vectors.f32 is missing"
            return True, ""

        # Calculate expected vector rows
        vec_size = vec_path.stat().st_size
        # embed.DIMENSION dimensions * 4 bytes per float32
        vec_row_count = vec_size // (embed.DIMENSION * 4)

        # Check for chunks with missing or invalid vec_row
        invalid_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE vec_row IS NULL OR vec_row >= ?",
            (vec_row_count,),
        ).fetchone()[0]

        if invalid_chunks > 0:
            conn.close()
            return False, f"index.db has {invalid_chunks} chunks with missing or invalid vec_row"

        # Check for orphaned vector rows
        chunk_rows = set(
            conn.execute("SELECT DISTINCT vec_row FROM chunks WHERE vec_row IS NOT NULL").fetchall()
        )
        chunk_rows = {r[0] for r in chunk_rows}
        valid_rows = set(range(vec_row_count))
        orphaned = valid_rows - chunk_rows

        conn.close()

        if orphaned:
            return False, f"vectors.f32 has {len(orphaned)} orphaned rows with no owning chunk"

        return True, ""

    except Exception as e:
        return False, f"validation error: {e}"
