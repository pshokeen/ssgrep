"""Tests for text chunking."""

import signal
import tempfile
from pathlib import Path

from ssgrep import store
from ssgrep.chunker import CHUNK_OVERLAP, CHUNK_TARGET_SIZE, chunk_episode, chunk_text
from ssgrep.types import ContentType
from tests.conftest import build_episode


def test_basic_chunking():
    text = "word " * 400
    chunks = chunk_text(text, "ep-1", "sess-1", ContentType.PROMPT)
    assert len(chunks) > 1


def test_no_oversized_chunks():
    text = "x" * 5000
    chunks = chunk_text(text, "ep-1", "sess-1", ContentType.RESPONSE)
    for c in chunks:
        assert len(c.text) <= CHUNK_TARGET_SIZE + 100


def test_empty_text():
    chunks = chunk_text("", "ep-1", "sess-1", ContentType.PROMPT)
    assert chunks == []


def test_short_text():
    text = "Short text"
    chunks = chunk_text(text, "ep-1", "sess-1", ContentType.PROMPT)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_deterministic():
    text = "word " * 400
    c1 = chunk_text(text, "ep-1", "sess-1", ContentType.PROMPT)
    c2 = chunk_text(text, "ep-1", "sess-1", ContentType.PROMPT)
    assert [c.text for c in c1] == [c.text for c in c2]


def test_chunk_ids_stable():
    text = "word " * 400
    chunks = chunk_text(text, "ep-1", "sess-1", ContentType.PROMPT)
    ids = [c.chunk_id for c in chunks]
    chunks2 = chunk_text(text, "ep-1", "sess-1", ContentType.PROMPT)
    ids2 = [c.chunk_id for c in chunks2]
    assert ids == ids2


# ---------------------------------------------------------------------------
# Content-derived chunk ids (types.Chunk's own contract: "Chunk
# ids are content-derived and therefore stable across runs"). A content
# hash caps any future duplicate-chunking bug's blast radius at 1x via
# INSERT OR REPLACE idempotency, instead of appending a distinct row per
# duplicate insertion -- which is what let the whole-batch fallback bug
# multiply this repo's own index by 733x before it was removed (see
# episodes.segment_episode_groups's docstring and indexer._build_episodes).
# ---------------------------------------------------------------------------


def test_chunk_id_reflects_text_not_just_position():
    """Two texts that chunk into the same NUMBER of pieces must not produce
    the same id sequence just because the pieces land at the same ordinal
    position -- that would be the old positional scheme (`...:index`) in
    disguise. Construct two same-length, same-episode, same-content-type
    texts with different content at every position and assert their id
    sets are fully disjoint.
    """
    text_a = "alpha " * 400
    text_b = "gamma " * 400
    assert len(text_a) == len(text_b)

    chunks_a = chunk_text(text_a, "ep-1", "sess-1", ContentType.PROMPT)
    chunks_b = chunk_text(text_b, "ep-1", "sess-1", ContentType.PROMPT)
    assert len(chunks_a) == len(chunks_b) > 1, "fixture must produce >1 comparable chunks"

    ids_a = {c.chunk_id for c in chunks_a}
    ids_b = {c.chunk_id for c in chunks_b}
    assert ids_a.isdisjoint(ids_b), "different content at the same position must not share an id"


def test_chunk_id_differs_across_episodes_for_identical_text():
    """Two different episodes producing byte-identical chunk text (a common
    short reply, a boilerplate disclaimer) must NOT collide on chunk_id --
    doing so would let INSERT OR REPLACE silently drop one episode's chunk
    row in favor of the other's, losing that episode from search entirely.
    Guards the id's episode_id-scoped prefix.
    """
    text = "please continue"
    c1 = chunk_text(text, "ep-A", "sess-1", ContentType.PROMPT)
    c2 = chunk_text(text, "ep-B", "sess-1", ContentType.PROMPT)
    assert len(c1) == len(c2) == 1
    assert c1[0].text == c2[0].text
    assert c1[0].chunk_id != c2[0].chunk_id


def test_repeated_windows_within_one_episode_get_distinct_ids():
    """Highly repetitive input can legitimately make chunk_text emit two
    byte-identical windows at different positions (e.g. consecutive windows
    over one long run of a repeated character). Their ids must still
    differ: detail.show() reconstructs an episode's full text by
    concatenating every one of its chunk rows in insertion order (the
    episodes table stores no prompt/response text of its own), so two
    positionally distinct chunks colliding on id would make
    INSERT OR REPLACE silently drop one -- shortening the reconstructed
    text below what was actually indexed.
    """
    text = "Z" * 60_000
    chunks = chunk_text(text, "ep-1", "sess-1", ContentType.PROMPT)
    assert len(chunks) > 2, "fixture must actually produce repeated identical windows"

    texts = [c.text for c in chunks]
    assert len(set(texts)) < len(texts), "fixture must contain at least one repeated window"

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "every chunk, even with duplicate text, needs a unique id"


def test_reinserting_identical_chunks_is_idempotent_not_additive():
    """The idempotency content hashing is meant to buy: re-chunking and
    re-inserting the SAME episode text must leave the stored row count
    exactly where it started, via INSERT OR REPLACE naturally overwriting
    same-id rows -- not append duplicates the way a fresh positional index
    could if re-processing ever started from a different counter value.
    """
    episode = build_episode(
        episode_id="ep-1",
        session_id="sess-1",
        prompt_text="alpha " * 400,
        response_text="beta " * 400,
    )
    chunks = chunk_episode(episode)
    assert len(chunks) > 2

    # Use the real on-disk init path (store.init_db) so the schema (incl.
    # chunks_fts) matches production exactly.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "index.db"
        conn = store.init_db(db_path)

        for chunk in chunks:
            store.insert_chunk(conn, chunk, vec_row=None)
        conn.commit()
        count_after_first = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count_after_first == len(chunks)

        # Re-chunk (simulating a retry/reprocess) and re-insert.
        chunks_again = chunk_episode(episode)
        for chunk in chunks_again:
            store.insert_chunk(conn, chunk, vec_row=None)
        conn.commit()
        count_after_second = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        assert (
            count_after_second == count_after_first
        ), "re-inserting identical chunks must not grow the table"
        conn.close()


# ---------------------------------------------------------------------------
# Overlap-continuation stride: `start = end - CHUNK_OVERLAP if end <
# len(text) else end`. This boundary has a genuine infinite-loop failure
# mode (see _chunk_text_with_timeout's docstring below) -- a hung suite has
# caught a broken version of it before, but only by accident: nothing here
# pins the actual window positions the stride is supposed to produce, so a
# future change that avoided the hang while still breaking the stride math
# (wrong overlap amount, off-by-one drift, etc.) would have nothing to catch
# it.
# ---------------------------------------------------------------------------


def _chunk_text_with_timeout(*args, timeout: int = 10, **kwargs):
    """Run chunk_text() under a hard wall-clock deadline.

    A broken overlap-continuation boundary doesn't just miscount chunks, it
    never returns: `end` is always `<= len(text)` by construction (computed
    via `min(..., len(text))`), so widening `end < len(text)` to `end <=
    len(text)` makes the stride recompute the identical start/end pair on
    every remaining iteration, forever. A plain call to chunk_text() here
    would hang the whole suite on a regression instead of failing this one
    test.

    SIGALRM interrupts a pure-Python loop cleanly (no leaked thread or
    subprocess to clean up) and is available on both of this project's CI
    runners (macos-latest, ubuntu-latest -- see .github/workflows/ci.yml);
    it degrades to no timeout on platforms without it (e.g. Windows) rather
    than erroring there.
    """
    if not hasattr(signal, "SIGALRM"):
        return chunk_text(*args, **kwargs)

    def _on_alarm(signum, frame):
        raise TimeoutError(
            f"chunk_text() did not return within {timeout}s -- likely an infinite "
            f"loop in the overlap-continuation stride (chunker.py's "
            f"`end < len(text)` check)"
        )

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout)
    try:
        return chunk_text(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def test_overlap_stride_matches_target_minus_overlap():
    """Pins the exact window positions chunk_text's overlap-continuation
    stride produces, not just "chunking happens" (the existing tests'
    level of coverage). Uses text with no "\\n\\n" anywhere, so the separate
    paragraph-break heuristic (`last_newline > start + CHUNK_TARGET_SIZE //
    2`) can never fire -- rfind always returns -1, which is never greater
    than a non-negative threshold -- and can't interfere with the pure
    stride arithmetic being pinned here.

    Chosen length gives three windows with a deliberately uneven last one,
    so both the repeated `start = end - CHUNK_OVERLAP` stride AND the
    terminal `else end` branch (which must NOT subtract the overlap again)
    are independently exercised and checked against ground-truth slices of
    the source text.
    """
    stride = CHUNK_TARGET_SIZE - CHUNK_OVERLAP
    text = "A" * (2 * stride + CHUNK_TARGET_SIZE - 500)

    chunks = _chunk_text_with_timeout(text, "ep-1", "sess-1", ContentType.PROMPT, timeout=10)

    expected_windows = [
        (0, CHUNK_TARGET_SIZE),
        (stride, stride + CHUNK_TARGET_SIZE),
        (2 * stride, len(text)),
    ]
    assert len(chunks) == len(
        expected_windows
    ), "wrong number of windows -- the stride or the terminal case is off"
    for chunk, (start, end) in zip(chunks, expected_windows, strict=True):
        assert chunk.text == text[start:end]
    assert len(chunks[-1].text) < CHUNK_TARGET_SIZE, "fixture must exercise the shorter last chunk"
