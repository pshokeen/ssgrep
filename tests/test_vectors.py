"""Tests for memory-mapped vector store."""

import numpy as np
import pytest

from ssgrep.vectors import append, close, cosine_top_k, detect_orphans, open_vectors, read


def test_append_and_read(tmp_path):
    store = open_vectors(tmp_path / "vectors.f32", dimension=4)
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    rows = append(store, vectors)
    assert rows == [0, 1]
    result = read(store, [0, 1])
    np.testing.assert_array_almost_equal(result, vectors)


def test_cosine_top_k(tmp_path):
    store = open_vectors(tmp_path / "vectors.f32", dimension=4)
    vectors = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0.707, 0.707, 0, 0],
        ],
        dtype=np.float32,
    )
    append(store, vectors)
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    results = cosine_top_k(store, query, k=2)
    assert len(results) == 2
    assert results[0][0] == 0


def test_cosine_with_candidates(tmp_path):
    """candidate_rows must restrict BOTH which rows can be returned AND which
    rows the cosine scores are computed over.

    A weaker version of this test (checking only `results[0][0] in [1, 2]`
    with two candidates tied at score 0) passes even if the candidate
    restriction is silently dropped entirely (`matrix = store.array` instead
    of `store.array[candidate_rows]`): scores get computed over the whole
    store, but results are still labeled via `candidate_rows[i]`, so a
    returned row ID can be silently paired with a score that belongs to a
    completely different, non-candidate row. That's worse than "returns too
    many results" -- it's a wrong row/score pairing with no error raised.

    To catch that, row 0 here is the true global best match (score 1.0) but
    is deliberately excluded from the candidate set, and rows 1/2 are given
    distinct, non-tied scores so there's exactly one correct answer to pin.
    """
    store = open_vectors(tmp_path / "vectors.f32", dimension=4)
    vectors = np.array(
        [
            [1, 0, 0, 0],  # row 0: identical to query (score 1.0) -- NOT a candidate
            [0, 1, 0, 0],  # row 1: candidate, orthogonal to query (score 0.0)
            [0.6, 0.8, 0, 0],  # row 2: candidate, partial match to query (score 0.6)
        ],
        dtype=np.float32,
    )
    append(store, vectors)
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    results = cosine_top_k(store, query, k=1, candidate_rows=[1, 2])

    assert len(results) == 1
    row, score = results[0]
    # Row 0 is the unrestricted global best match; it must never leak through
    # a candidate-restricted query, and the winner among the actual
    # candidates is unambiguous (0.6 beats 0.0).
    assert row == 2, f"expected the best-scoring candidate (row 2), got row {row}"
    assert score == pytest.approx(0.6, abs=1e-4), (
        f"row {row} was returned with score {score}, which does not match its "
        f"actual cosine similarity to the query -- scores are being computed "
        f"over the wrong rows"
    )


def test_detect_orphans(tmp_path):
    store = open_vectors(tmp_path / "vectors.f32", dimension=4)
    append(store, np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    orphans = detect_orphans(store, {0})
    assert orphans == [1]


def test_reopen_and_append(tmp_path):
    """Test the defect: append() crashes on reopened non-empty store.
    This reproduces the exact bug described in the issue.
    """
    p = tmp_path / "vectors.f32"
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    # First append to a fresh store
    s1 = open_vectors(p, dimension=4)
    rows1 = append(s1, vectors)
    assert rows1 == [0, 1]
    assert s1.row_count == 2
    del s1

    # Reopen the non-empty store and append again
    s2 = open_vectors(p, dimension=4)
    assert s2.row_count == 2
    rows2 = append(s2, vectors)
    assert rows2 == [2, 3]
    assert s2.row_count == 4


def test_sequential_appends_bit_identity(tmp_path):
    """Three sequential appends across separate open_vectors() calls.
    Assert row_count and bit-identity of all previously written rows.
    """
    p = tmp_path / "vectors.f32"
    dimension = 8
    vectors1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    vectors2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.float32)
    vectors3 = np.array(
        [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]],
        dtype=np.float32,
    )

    # First append
    s1 = open_vectors(p, dimension=dimension)
    append(s1, vectors1)
    assert s1.row_count == 2
    del s1

    # Second append
    s2 = open_vectors(p, dimension=dimension)
    assert s2.row_count == 2
    np.testing.assert_array_equal(read(s2, [0, 1]), vectors1)
    append(s2, vectors2)
    assert s2.row_count == 3
    del s2

    # Third append
    s3 = open_vectors(p, dimension=dimension)
    assert s3.row_count == 3
    np.testing.assert_array_equal(read(s3, [0, 1]), vectors1)
    np.testing.assert_array_equal(read(s3, [2]), vectors2)
    append(s3, vectors3)
    assert s3.row_count == 5
    np.testing.assert_array_equal(read(s3, [0, 1]), vectors1)
    np.testing.assert_array_equal(read(s3, [2]), vectors2)
    np.testing.assert_array_equal(read(s3, [3, 4]), vectors3)
    del s3


def test_file_handle_closed_on_append(tmp_path):
    """Assert that old file handles are properly closed when append() replaces them."""
    p = tmp_path / "vectors.f32"
    vectors = np.array([[1, 2, 3, 4]], dtype=np.float32)

    store = open_vectors(p, dimension=4)
    append(store, vectors)

    # Hold a reference to the file handle that append() just created
    old_handle = store._file_handle

    # Second append will replace store._file_handle with a new one
    append(store, vectors)

    # The old handle should now be closed
    assert old_handle.closed, "Old file handle not closed after append()"

    # The new handle should still be open
    assert not store._file_handle.closed, "New file handle should be open"

    # After close(), both handles should be closed and attributes nullified
    handle_before_close = store._file_handle
    mmap_before_close = store.file
    close(store)

    # Verify the mmap was actually closed, not just set to None
    assert mmap_before_close.closed, "Mmap not closed in close()"
    # Verify the file handle was actually closed, not just set to None
    assert handle_before_close.closed, "File handle not closed in close()"
    # Verify attributes were nullified
    assert store._file_handle is None, "File handle attribute not nullified"
    assert store.file is None, "Mmap attribute not nullified"


def test_append_after_close_raises_error(tmp_path):
    """Assert append after close() raises a clear error."""
    p = tmp_path / "vectors.f32"
    vectors = np.array([[1, 0, 0, 0]], dtype=np.float32)

    store = open_vectors(p, dimension=4)
    append(store, vectors)
    close(store)

    # Attempting to append should raise ValueError with clear message
    with_error = False
    try:
        append(store, vectors)
    except ValueError as e:
        with_error = True
        assert "closed" in str(e).lower(), f"Error message unclear: {e}"

    assert with_error, "Expected ValueError when appending to closed store"


def test_open_vectors_partial_trailing_write_fails_loud_not_silent(tmp_path):
    """A torn/partial trailing write (e.g. a crash mid-append) must never be
    silently misread as a smaller-but-valid store.

    open_vectors()'s row_count computation (`size // (dimension * 4)`) looks
    like it could silently floor away a partial trailing row with no signal.
    It can't, today: `store.array` is built by reshaping the *entire*
    file's bytes to (-1, dimension) a few lines later in the same function,
    which requires the byte count to already be an exact multiple of the
    row size. Any genuine partial row (file size not a multiple of
    dimension * 4) makes that reshape raise before open_vectors() can
    return -- so no caller ever observes a VectorStore whose row_count
    silently disagrees with its own array.

    This is reachable in practice: search.py, repair.py, and revectorize.py
    all call open_vectors() without a preceding GenerationalStore.recover()
    (only indexer.index() calls recover() first), so a torn write left by a
    previous crash can genuinely reach this function. This test pins the
    safety net that makes that reachability harmless: it is incidental to
    numpy's strictness, not deliberately coded, so a future change (e.g.
    wrapping the reshape in a try/except that falls back to the floored
    row_count) could silently reintroduce exactly the corruption this
    guards against.
    """
    p = tmp_path / "vectors.f32"
    dimension = 4
    row_bytes = dimension * 4  # 16 bytes/row

    # Two complete rows (32 bytes) plus 5 stray bytes: a genuine torn
    # write, not an exact multiple of row_bytes.
    p.write_bytes(b"\x00" * (row_bytes * 2 + 5))

    with pytest.raises(ValueError):
        open_vectors(p, dimension=dimension)
