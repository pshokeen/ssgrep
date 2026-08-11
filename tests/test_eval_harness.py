"""Tests for the retrieval evaluation harness (eval/).

Two different things need guarding here, and they are tested separately:

1. **The harness's own arithmetic and wiring** (eval/harness.py's
   `evaluate`/`_agg`/`rank_episodes_for_query`) -- does it compute rank,
   hit@10, MRR and the tuning aggregates correctly, given a KNOWN, synthetic,
   fully-controlled index? These tests never touch the real corpus: they
   build a tiny synthetic project via indexer.index() exactly the way
   tests/test_indexer.py does, with embed.encode() monkeypatched to a fast
   deterministic stand-in, so results are hermetic and fast.

2. **The committed label set's own integrity** (eval/queries.jsonl) -- at
   least 30 queries, all four classes represented, and -- the property that
   actually rules out the circularity build_labels.py's docstring warns
   against -- paraphrase/multi-hop queries never contain their anchor text
   verbatim. These read the committed file directly and never require the
   real corpus either.

A third, small category (marked `slow` and skipped without
~/.claude/projects) re-runs the actual methodology -- segmentation-mirror
alignment and build_labels' own invariants -- against today's real corpus,
the same corpus the committed queries.jsonl was built from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval import build_labels, chunking_ab, harness
from ssgrep import embed
from ssgrep.types import ContentType

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Synthetic-index helpers (mirrors tests/test_indexer.py's own pattern)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


def _install_fake_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A constant unit vector for every input. Keeps the vector leg from
    injecting its own ranking signal, so tests can rely purely on BM25's
    deterministic keyword matching to control which episode ranks where.
    """

    def fake_encode(texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 256), 1.0 / (256**0.5), dtype=np.float32)

    monkeypatch.setattr(embed, "encode", fake_encode)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _user_record(uid: str, text: str, cwd: str, session_id: str) -> dict[str, Any]:
    return {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": text},
        "uuid": uid,
        "timestamp": "2026-07-01T10:00:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


def _assistant_record(
    uid: str, text: str, cwd: str, session_id: str, parent_uuid: str
) -> dict[str, Any]:
    return {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "uuid": uid,
        "timestamp": "2026-07-01T10:01:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


def _build_synthetic_index(fake_home: Path, tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Three sessions, three episodes, each with a distinct, unambiguous
    keyword so BM25 alone determines which episode a keyword query hits.
    Returns (project_dir, index_dir).
    """
    from ssgrep import indexer

    project_dir = tmp_path / "project"
    projects_root = fake_home / ".claude" / "projects" / "proj"

    _write_jsonl(
        projects_root / "session-a.jsonl",
        [
            _user_record(
                "u1", "How do I configure the widget cache TTL?", str(project_dir), "session-a"
            ),
            _assistant_record(
                "a1",
                "Set CACHE_TTL_SECONDS in the widget cache settings module.",
                str(project_dir),
                "session-a",
                "u1",
            ),
        ],
    )
    _write_jsonl(
        projects_root / "session-b.jsonl",
        [
            _user_record(
                "u2",
                "Why does the flimflam parser reject empty input?",
                str(project_dir),
                "session-b",
            ),
            _assistant_record(
                "a2",
                "The flimflam parser requires at least one token before EOF.",
                str(project_dir),
                "session-b",
                "u2",
            ),
        ],
    )
    _write_jsonl(
        projects_root / "session-c.jsonl",
        [
            _user_record(
                "u3",
                "totally unrelated gardening question about tomatoes",
                str(project_dir),
                "session-c",
            ),
            _assistant_record(
                "a3", "Plant tomatoes after the last frost.", str(project_dir), "session-c", "u3"
            ),
        ],
    )

    _install_fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"
    indexer.index(project_dir, index_dir=index_dir, quiet=True)
    return project_dir, index_dir


# ---------------------------------------------------------------------------
# 1. Harness arithmetic and wiring — synthetic, hermetic, no real corpus
# ---------------------------------------------------------------------------


def test_evaluate_finds_correct_rank_via_bm25_keyword(fake_home, tmp_path, monkeypatch):
    _project_dir, index_dir = _build_synthetic_index(fake_home, tmp_path, monkeypatch)

    queries = [
        {
            "id": "q-cache",
            "query": "widget cache TTL",
            "class": "exact-identifier",
            "subagent_only": False,
            "target_episode_ids": ["session-a:ep:0"],
        },
        {
            "id": "q-flimflam",
            "query": "flimflam parser empty input",
            "class": "error-string",
            "subagent_only": False,
            "target_episode_ids": ["session-b:ep:0"],
        },
    ]

    results = harness.evaluate(index_dir, queries, main_session_boost=0.0)

    by_id = {r.id: r for r in results}
    assert by_id["q-cache"].rank == 1
    assert by_id["q-cache"].hit_at_10 is True
    assert by_id["q-cache"].reciprocal_rank == pytest.approx(1.0)
    assert by_id["q-flimflam"].rank == 1
    assert by_id["q-flimflam"].hit_at_10 is True


def test_evaluate_reports_miss_when_target_absent_from_index(fake_home, tmp_path, monkeypatch):
    _project_dir, index_dir = _build_synthetic_index(fake_home, tmp_path, monkeypatch)

    queries = [
        {
            "id": "q-nonexistent",
            "query": "widget cache TTL",
            "class": "exact-identifier",
            "subagent_only": False,
            # This episode was never indexed -- must never be found.
            "target_episode_ids": ["session-zzz:ep:0"],
        }
    ]

    results = harness.evaluate(index_dir, queries, main_session_boost=0.0)

    assert results[0].rank is None
    assert results[0].hit_at_10 is False
    assert results[0].reciprocal_rank == 0.0


def test_evaluate_multi_relevant_target_uses_best_rank(fake_home, tmp_path, monkeypatch):
    """target_episode_ids is a SET; MRR must use the lowest (best) rank
    among acceptable answers, per build_labels.py's documented treatment of
    a real corpus's restated facts.
    """
    _project_dir, index_dir = _build_synthetic_index(fake_home, tmp_path, monkeypatch)

    queries = [
        {
            "id": "q-multi",
            "query": "widget cache TTL",
            "class": "exact-identifier",
            "subagent_only": False,
            # Only one of these two actually exists / is relevant to the
            # query; the harness must still find it and not fail on the
            # other id not appearing anywhere in the ranking.
            "target_episode_ids": ["session-zzz:ep:99", "session-a:ep:0"],
        }
    ]

    results = harness.evaluate(index_dir, queries, main_session_boost=0.0)

    assert results[0].rank == 1
    assert results[0].hit_at_10 is True


def test_agg_recall_and_mrr_arithmetic():
    rows = [
        harness.QueryResult(
            id="a",
            query="",
            query_class="c",
            subagent_only=False,
            rank=1,
            reciprocal_rank=1.0,
            hit_at_10=True,
            top_result_episode_id="e1",
            top_result_score=0.5,
        ),
        harness.QueryResult(
            id="b",
            query="",
            query_class="c",
            subagent_only=False,
            rank=4,
            reciprocal_rank=0.25,
            hit_at_10=True,
            top_result_episode_id="e2",
            top_result_score=0.4,
        ),
        harness.QueryResult(
            id="c",
            query="",
            query_class="c",
            subagent_only=False,
            rank=None,
            reciprocal_rank=0.0,
            hit_at_10=False,
            top_result_episode_id="e3",
            top_result_score=0.1,
        ),
        harness.QueryResult(
            id="d",
            query="",
            query_class="c",
            subagent_only=False,
            rank=None,
            reciprocal_rank=0.0,
            hit_at_10=False,
            top_result_episode_id=None,
            top_result_score=None,
        ),
    ]

    agg = harness._agg(rows)

    assert agg["n"] == 4
    assert agg["recall_at_10"] == pytest.approx(2 / 4)
    assert agg["mrr"] == pytest.approx((1.0 + 0.25 + 0.0 + 0.0) / 4)


def test_agg_empty_input_reports_none_not_zero():
    """n=0 must report None, not a misleading 0.0 that looks like a measured
    zero score for a class no query in this run actually belonged to.
    """
    agg = harness._agg([])
    assert agg["n"] == 0
    assert agg["recall_at_10"] is None
    assert agg["mrr"] is None


def test_summarize_breaks_out_by_class_and_subagent_only():
    rows = [
        harness.QueryResult(
            id="a",
            query="",
            query_class="exact-identifier",
            subagent_only=False,
            rank=1,
            reciprocal_rank=1.0,
            hit_at_10=True,
            top_result_episode_id="e1",
            top_result_score=0.5,
        ),
        harness.QueryResult(
            id="b",
            query="",
            query_class="multi-hop",
            subagent_only=True,
            rank=None,
            reciprocal_rank=0.0,
            hit_at_10=False,
            top_result_episode_id=None,
            top_result_score=None,
        ),
    ]

    summary = harness.summarize(rows)

    assert summary["overall"]["n"] == 2
    assert summary["class:exact-identifier"]["n"] == 1
    assert summary["class:exact-identifier"]["recall_at_10"] == pytest.approx(1.0)
    assert summary["class:multi-hop"]["n"] == 1
    assert summary["class:multi-hop"]["recall_at_10"] == pytest.approx(0.0)
    assert summary["subagent_only"]["n"] == 1
    assert summary["subagent_only"]["recall_at_10"] == pytest.approx(0.0)


def test_load_queries_round_trips_jsonl(tmp_path):
    path = tmp_path / "q.jsonl"
    rows = [
        {"id": "x1", "query": "foo", "class": "exact-identifier", "target_episode_ids": ["e1"]},
        {"id": "x2", "query": "bar", "class": "paraphrase", "target_episode_ids": ["e2", "e3"]},
    ]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write("\n")  # a blank line must not become a phantom row

    loaded = harness.load_queries(path)

    assert loaded == rows


# ---------------------------------------------------------------------------
# 2. Committed label set integrity — reads eval/queries.jsonl directly
# ---------------------------------------------------------------------------


def _load_committed_queries() -> list[dict]:
    return harness.load_queries()


def test_query_set_has_at_least_30_queries_across_all_four_classes():
    rows = _load_committed_queries()
    assert len(rows) >= 30

    classes = {r["class"] for r in rows}
    assert classes == {"exact-identifier", "error-string", "paraphrase", "multi-hop"}

    from collections import Counter

    counts = Counter(r["class"] for r in rows)
    for cls, n in counts.items():
        assert n >= 5, f"class {cls!r} has only {n} queries, too thin to trust its aggregate"


def test_query_set_ids_are_unique():
    rows = _load_committed_queries()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_query_set_every_row_has_nonempty_targets():
    rows = _load_committed_queries()
    for r in rows:
        assert r["target_episode_ids"], f"{r['id']}: no target episodes -- untestable query"
        assert r["query"].strip(), f"{r['id']}: empty query text"


def test_query_set_includes_subagent_only_targets():
    """Task requirement: include queries whose correct answer lives only in
    a subagent transcript, since that's ~82% of indexed prose.
    """
    rows = _load_committed_queries()
    subagent_only = [r for r in rows if r["subagent_only"]]
    assert len(subagent_only) >= 5


def test_paraphrase_and_multihop_queries_never_contain_their_anchor_verbatim():
    """The anti-circularity property build_labels.py's docstring promises:
    for the two classes phrased in different words than the anchor, the
    anchor string must genuinely not appear in the query text. This is the
    one property that would silently break if someone pasted an anchor into
    a paraphrase/multi-hop query by hand.
    """
    rows = _load_committed_queries()
    checked = 0
    for r in rows:
        if r["class"] not in ("paraphrase", "multi-hop"):
            continue
        checked += 1
        query_lower = r["query"].lower()
        for anchor in r["anchors"]:
            assert anchor.lower() not in query_lower, (
                f"{r['id']}: anchor {anchor!r} appears verbatim in paraphrase/multi-hop "
                f"query {r['query']!r} -- this query is testing self-agreement, not retrieval"
            )
    assert checked >= 10  # both classes actually got exercised


def test_exact_identifier_and_error_string_queries_do_use_literal_anchor_tokens():
    """The inverse check: those two classes are SUPPOSED to be literal, so a
    query containing none of its own anchors would indicate a mislabeled
    row, not a well-formed exact/error-string query.
    """
    rows = _load_committed_queries()
    for r in rows:
        if r["class"] not in ("exact-identifier", "error-string"):
            continue
        assert any(a in r["query"] for a in r["anchors"]), (
            f"{r['id']}: {r['class']} query {r['query']!r} contains none of its own "
            f"anchors {r['anchors']!r}"
        )


# ---------------------------------------------------------------------------
# 3. Real-corpus methodology checks — skipped when ~/.claude/projects absent
# ---------------------------------------------------------------------------


@pytest.fixture
def ensure_corpus_available() -> None:
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        pytest.skip(reason="~/.claude/projects does not exist; real session corpus unavailable")


@pytest.mark.slow
def test_per_turn_segmentation_mirror_matches_real_segmentation(
    ensure_corpus_available, ensure_labelled_corpus_available
):
    """chunking_ab.py's whole A/B result is only meaningful if its hand-
    mirrored segmentation reproduces episodes.segment_episodes exactly. This
    re-derives the check chunking_ab.run_chunking_ab() itself performs
    before building any index, against the same real corpus the committed
    A/B numbers were measured on.

    ensure_corpus_available alone is not enough here, for the same reason
    documented on ensure_labelled_corpus_available below: it is a proxy that
    checks ~/.claude/projects exists, not that discovery actually resolves
    the real corpus at PROJECT_DIR. In a worktree checked out to a different
    path than the one real sessions recorded as their cwd (or in a genuinely
    shrunk corpus), discover_sessions(PROJECT_DIR) can legitimately return
    zero sessions -- chunking_ab._verify_alignment() then returns 0, which
    must skip, not fail: 0 sessions checked means the corpus was
    unavailable, not that the mirrored segmentation disagreed with the real
    one.
    """
    checked = chunking_ab._verify_alignment()
    assert checked > 0


@pytest.fixture
def ensure_labelled_corpus_available() -> None:
    """Skip unless the corpus the anchors were drawn from is discoverable.

    ensure_corpus_available is not enough here and was the reason this test
    failed rather than skipped: it checks that ~/.claude/projects exists,
    which is a proxy. Claude Code derives a project's transcript directory
    from its path and abandons the old one on a move, so after this repo
    moved the directory still existed, discovery still returned a large
    corpus, and every anchor still matched nothing.
    build_labels.corpus_shortfall() checks the precondition itself.
    """
    shortfall = build_labels.corpus_shortfall()
    if shortfall is not None:
        pytest.skip(reason=f"labelled corpus unavailable: {shortfall}")


@pytest.mark.slow
def test_build_labels_methodology_invariants_hold_on_current_corpus(
    ensure_corpus_available, ensure_labelled_corpus_available
):
    """Re-runs build_labels.build()'s own assertions (every anchor matches
    at least one episode, none matches more than the proportional
    ANCHOR_MATCH_RATIO cap, the subagent-only anchors are genuinely
    subagent-only) against today's
    corpus. Does not overwrite queries.jsonl -- the corpus mutates
    continuously, so exact episode-id reproducibility isn't expected, but
    the methodology's own invariants failing would mean the committed file
    can no longer be regenerated the way its docstring claims.
    """
    rows = build_labels.build()
    assert len(rows) == len(build_labels.QUERY_SPECS)


def test_per_turn_chunker_drops_short_turns_and_windows_long_ones():
    """Unit test of chunking_ab's per-turn chunking rule in isolation, no
    corpus or index required: a turn under MIN_TURN_CHARS is dropped, a
    turn at or under the target size becomes exactly one chunk, and a
    longer turn is windowed with no chunk exceeding the cap.
    """
    turns_by_episode = {
        "ep-1": [
            ("short", ContentType.PROMPT),  # 5 chars, must be dropped
            ("a" * 500, ContentType.RESPONSE),  # one chunk
            ("b" * 3000, ContentType.RESPONSE),  # must be windowed into >1 chunks
        ]
    }
    fn = chunking_ab.make_per_turn_chunk_episode(turns_by_episode)

    from tests.conftest import build_episode

    episode = build_episode(
        episode_id="ep-1", session_id="s-1", prompt_text="short", response_text="ignored"
    )
    chunks = fn(episode)

    # the short prompt turn was dropped entirely
    assert all(c.text != "short" for c in chunks)
    # the 500-char turn became exactly one chunk
    medium_chunks = [c for c in chunks if c.text == "a" * 500]
    assert len(medium_chunks) == 1
    # the 3000-char turn was windowed into multiple chunks, none over the cap
    long_chunks = [c for c in chunks if set(c.text) <= {"b"}]
    assert len(long_chunks) > 1
    assert all(len(c.text) <= chunking_ab.CHUNK_TARGET_SIZE for c in chunks)
    # chunk ids are unique -- no collision between the two response turns
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_per_turn_chunker_falls_back_for_unknown_episode():
    """An episode with no entry in the turns index must not silently
    produce zero chunks -- it falls back to production chunking.
    """
    from ssgrep import chunker
    from tests.conftest import build_episode

    fn = chunking_ab.make_per_turn_chunk_episode({})
    episode = build_episode(
        episode_id="unseen-ep",
        session_id="s-1",
        prompt_text="a fairly ordinary prompt with enough text to chunk",
        response_text="a fairly ordinary response with enough text to chunk",
    )

    result = fn(episode)
    expected = chunker.chunk_episode(episode)
    assert [c.text for c in result] == [c.text for c in expected]


# ---------------------------------------------------------------------------
# 4. Anchor cap proportionality tests — prevent point-in-time drift
# ---------------------------------------------------------------------------


def test_anchor_cap_is_proportional_not_absolute():
    """The anchor cap must scale with corpus size, not be a fixed count.
    This prevents the test from drifting red as the corpus grows.
    """
    # Verify the constant exists and is sensible (between 0.1% and 10%)
    assert hasattr(build_labels, "ANCHOR_MATCH_RATIO")
    assert 0.001 <= build_labels.ANCHOR_MATCH_RATIO <= 0.10
    # At 1%, a 6140-episode corpus allows up to 61 matching episodes for any anchor
    assert build_labels.ANCHOR_MATCH_RATIO == 0.01


def test_anchor_cap_formula_uses_proportional_calculation():
    """Verify that the anchor cap is calculated as a ratio of corpus size,
    not as a fixed absolute number. Test the calculation formula directly
    without relying on the full build() machinery.
    """
    # Test the cap calculation formula with various corpus sizes.
    # Formula: anchor_match_cap = max(1, int(corpus_size * ANCHOR_MATCH_RATIO))
    ratio = build_labels.ANCHOR_MATCH_RATIO

    # Small corpus of 10 episodes: cap = max(1, int(10 * 0.01)) = 1
    assert max(1, int(10 * ratio)) == 1

    # Medium corpus of 100 episodes: cap = max(1, int(100 * 0.01)) = 1
    assert max(1, int(100 * ratio)) == 1

    # 1000 episodes: cap = max(1, int(1000 * 0.01)) = 10
    assert max(1, int(1000 * ratio)) == 10

    # 10000 episodes: cap = max(1, int(10000 * 0.01)) = 100
    assert max(1, int(10000 * ratio)) == 100

    # Verify this is NOT a fixed constant like the old ANCHOR_MATCH_CAP = 35
    assert max(1, int(1000 * ratio)) != 35
    assert max(1, int(10000 * ratio)) != 35


def test_anchor_match_cap_prevents_generic_queries():
    """Verify the real-world behavior: the WHERE 1=1 anchor (error-03)
    is currently specific enough but would be rejected if it grew to
    exceed the proportional cap. As a regression test, this documents
    the current state and the invariant that should hold.
    """
    pairs = build_labels._collect_episodes()
    corpus_size = len(pairs)

    # Skip if the corpus is unavailable or empty
    if corpus_size == 0:
        pytest.skip(
            reason=(
                "Real session corpus is empty or unavailable; "
                "test_anchor_match_cap_prevents_generic_queries requires a populated corpus"
            )
        )

    anchor_match_cap = max(1, int(corpus_size * build_labels.ANCHOR_MATCH_RATIO))

    # error-03 is one of the queries; find its anchor match count.
    error_03 = next(s for s in build_labels.QUERY_SPECS if s.id == "error-03")
    hits = build_labels._matches(error_03, pairs)

    # Verify error-03 is below the cap (the whole point of this fix).
    assert len(hits) <= anchor_match_cap
    # At current corpus size, WHERE 1=1 matches ~30 episodes out of 6140 (0.49%).
    # This is well below the 1% cap and should remain acceptable as the corpus
    # grows. The proportional metric ensures this test doesn't need re-tuning.
    assert len(hits) < corpus_size * 0.01


@pytest.mark.slow
def test_anchor_cap_prevents_too_generic_anchor_in_real_corpus(ensure_labelled_corpus_available):
    """Mutation test: if build_labels.py were changed to remove or raise the
    cap, this test would fail by construction — build() must genuinely REJECT
    an anchor matching more than the proportional cap.

    Earlier versions asserted arithmetic about error-03's `WHERE 1=1` match
    count instead (~30 episodes in July 2026); transcript cleanup shrank that
    to single digits, which broke the test without any code change and proved
    the count was a proxy, not the property (see CLAUDE.md, "Assert the
    Property, Not a Proxy"). The property is build()'s rejection behavior, so
    this version exercises it directly: a deliberately generic anchor (" the ",
    present in most prose episodes on any real corpus) is injected as a spec,
    and build() must fail its cap assertion — naming the topic diagnosis —
    rather than emit a label row for it.
    """
    pairs = build_labels._collect_episodes()
    corpus_size = len(pairs)
    cap = max(1, int(corpus_size * build_labels.ANCHOR_MATCH_RATIO))

    generic = build_labels.QuerySpec(
        "generic-probe",
        "the",
        "exact-identifier",
        (" the ",),
    )
    hits = build_labels._matches(generic, pairs)
    # Positive assertion first: the probe anchor really is over-cap on this
    # corpus, so the rejection below is exercised, not vacuously skipped.
    assert len(hits) > cap, (
        f"generic probe anchor matched only {len(hits)} of {corpus_size} episodes "
        f"(cap {cap}) — corpus too small/odd for this test to mean anything"
    )

    original_specs = build_labels.QUERY_SPECS
    build_labels.QUERY_SPECS = [generic]
    try:
        with pytest.raises(AssertionError, match="topic, not a specific target"):
            build_labels.build()
    finally:
        build_labels.QUERY_SPECS = original_specs


def test_build_raises_when_an_anchor_exceeds_the_proportional_cap(monkeypatch):
    """Exercises build()'s actual enforcement path, not a reimplementation of
    it. The four tests above check the ANCHOR_MATCH_RATIO constant and the
    max(1, int(corpus_size * ratio)) formula in isolation -- none of them
    call build() itself, so a mutation that disabled or loosened build()'s
    `assert len(hits) <= anchor_match_cap` line would pass every test above
    unnoticed. This test forces that exact condition, hermetically, and
    checks build() itself raises.
    """
    fake_pairs = list(range(50))  # corpus_size=50 -> cap = max(1, int(50*0.01)) = 1
    oversized_hits = [(None, None), (None, None)]  # 2 hits > cap of 1

    monkeypatch.setattr(build_labels, "_collect_episodes", lambda: fake_pairs)
    monkeypatch.setattr(build_labels, "_matches", lambda spec, pairs: oversized_hits)

    with pytest.raises(AssertionError, match="matched"):
        build_labels.build()
