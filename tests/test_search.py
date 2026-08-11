"""Tests for hybrid BM25 + vector retrieval (search.py).

Fixture content in FIXTURE_EPISODES is drawn from the same fictional
widget-cache/allocator domain as the committed fixtures under
tests/fixtures/ (see manifest.md) -- several sentences are copied verbatim
from tests/fixtures/main-session.jsonl. The two acceptance-scenario queries
below (exact-identifier and paraphrase) were validated empirically against
the real embed.py/model2vec pipeline before being committed here: BM25
returns zero hits for the paraphrase query (confirmed directly against a
real FTS5 table), and the vector leg ranks ep-allocator's best chunk #1 by
cosine similarity ahead of every decoy, including ep-retry (whose response
chunk contains the identifier-scenario's literal token and would otherwise
be the strongest single chunk in the corpus). A query that shares no
literal tokens with the target text still shares occasional stopwords or
one domain word ("allocator") with a decoy in a six-episode fixture this
small; what actually matters, and what is asserted directly, is that BM25's
implicit-AND over every query token still returns zero rows.

Fixture indexes are built with the real store.py/vectors.py/chunker.py/
embed.py calls (never search.py's own code, which must never index) and
batch every vector into a single vectors.append() call: appending more than
once to the same store currently crashes (a separate, already-reported bug
in vectors.py, not touched by this file), and single-batch append is also
simply the realistic way an indexer would embed a batch of chunks.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep import api, embed, search, staleness, store
from ssgrep import vectors as vecstore
from ssgrep.chunker import chunk_episode
from ssgrep.types import (
    Chunk,
    ContentType,
    EmptyQueryError,
    Episode,
    FileCursor,
    IndexNotFoundError,
    IndexNotReadyError,
    ResultCard,
    SearchFilters,
)

# ---------------------------------------------------------------------------
# Shared fixture corpus
# ---------------------------------------------------------------------------

FIXTURE_EPISODES: list[Episode] = [
    Episode(
        episode_id="ep-cache",
        session_id="s1",
        prompt_text="How do I configure the widget cache TTL?",
        response_text=(
            "You can set the widget cache TTL in the cache settings module.\n"
            "The current widget cache TTL is 300 seconds. Edit CACHE_TTL_SECONDS "
            "in src/cache/settings.py to change it."
        ),
        title="Configure the widget cache TTL",
        timestamp=datetime(2026, 7, 1, 10, 5, tzinfo=UTC),
        files_touched=("src/cache/settings.py",),
        git_branch="main",
    ),
    Episode(
        episode_id="ep-allocator",
        session_id="s1",
        prompt_text=(
            "Please also check whether the allocator frobnication path handles " "empty pools."
        ),
        response_text=(
            "The frobnication path returns early on empty pools; all allocator "
            "tests pass.\nConfirmed: the empty-pool path is covered by the "
            "passing tests."
        ),
        title="Allocator empty-pool audit",
        timestamp=datetime(2026, 7, 5, 10, 12, tzinfo=UTC),
        files_touched=("src/allocator/pool.py",),
        git_branch="main",
    ),
    Episode(
        episode_id="ep-metrics",
        session_id="s1",
        prompt_text="Does pool resize emit a metric?",
        response_text=(
            "Pool resize emits one gauge metric: allocator.pool.size. It is "
            "registered in src/allocator/metrics.py during module init."
        ),
        title="Pool resize metric",
        timestamp=datetime(2026, 7, 10, 10, 20, tzinfo=UTC),
        files_touched=("src/allocator/metrics.py",),
        git_branch="feature/pool-metrics",
    ),
    Episode(
        episode_id="ep-audit",
        session_id="s1",
        prompt_text="Run the legacy audit against the cache module.",
        response_text=(
            "Legacy audit pass complete; no drift found in the cache module. "
            "The cache module matches its documented behavior."
        ),
        title="Legacy cache audit",
        timestamp=datetime(2026, 7, 15, 10, 25, tzinfo=UTC),
        files_touched=(),
        git_branch="main",
    ),
    Episode(
        episode_id="ep-phases",
        session_id="s1",
        prompt_text="Explain the algorithm phases.",
        response_text=(
            "The algorithm has three phases: seed, shuffle, and settle. Settle "
            "reconciles pending pool entries and releases unused handles."
        ),
        title="Algorithm phases",
        timestamp=datetime(2026, 7, 20, 10, 30, tzinfo=UTC),
        files_touched=(),
        git_branch="feature/algo",
    ),
    Episode(
        episode_id="ep-retry",
        session_id="s1",
        prompt_text="How do I reduce the number of retries during a network blip?",
        response_text=(
            "Set retry_backoff_ms to a higher value to slow down repeated "
            "attempts after a failure."
        ),
        title="Tune retry backoff",
        timestamp=datetime(2026, 7, 25, 10, 30, tzinfo=UTC),
        files_touched=("src/net/retry.py",),
        git_branch="main",
    ),
    Episode(
        episode_id="ep-subagent-deploy",
        session_id="s1:agent:deadbeef",
        prompt_text=(
            "Investigate why the deployment pipeline occasionally hangs after " "the build step."
        ),
        response_text=(
            "The hang was caused by a dangling background process holding "
            "stdout open; the fix waits on an explicit sentinel instead of EOF."
        ),
        title="Debug hanging deploy pipeline",
        timestamp=datetime(2026, 7, 28, 9, 0, tzinfo=UTC),
        files_touched=("scripts/deploy.sh",),
        git_branch="main",
        is_subagent=True,
        agent_type="general-purpose",
        agent_name="deploy-debugger",
        agent_description="Investigate CI pipeline hang",
        parent_session_id="s1",
    ),
]

# Validated empirically against the real embed.py pipeline and this exact
# seven-episode corpus: the BM25 leg returns zero hits for this query, while
# the vector leg ranks ep-allocator's best chunk #1 by cosine similarity
# with a real margin over every decoy, including ep-retry (whose response
# chunk is otherwise the strongest single chunk in the corpus, since it
# contains the literal token from the identifier scenario below).
PARAPHRASE_QUERY = "Is it safe to call the allocator on a pool that has no remaining free slots?"


def _build_index(index_dir: Path, episodes: list[Episode]) -> None:
    """Build a real on-disk index from episodes via the real sibling modules."""
    conn = store.init_db(index_dir / "index.db")
    vstore = vecstore.open_vectors(index_dir / "vectors.f32", dimension=embed.DIMENSION)

    all_chunks: list[Chunk] = []
    for ep in episodes:
        store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)
        all_chunks.extend(chunk_episode(ep))

    if all_chunks:
        vecs = embed.encode([c.text for c in all_chunks])
        rows = vecstore.append(vstore, vecs)
        for chunk, row in zip(all_chunks, rows, strict=True):
            store.insert_chunk(conn, chunk, row)

    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def indexed_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real, module-scoped, read-only index built once and shared.

    search() never mutates what it reads, so sharing across tests in this
    module is safe. Tests needing a different index state use their own
    function-scoped tmp_path instead.
    """
    project_dir = tmp_path_factory.mktemp("search-project")
    _build_index(project_dir / search.INDEX_DIRNAME, FIXTURE_EPISODES)
    return project_dir


# ---------------------------------------------------------------------------
# The two DONE-WHEN acceptance scenarios
# ---------------------------------------------------------------------------


class TestAcceptanceScenarios:
    def test_exact_identifier_query_retrieves_right_episode(self, indexed_project: Path) -> None:
        response = search.search(indexed_project, "retry_backoff_ms")
        assert response.results, "expected at least one result"
        assert response.results[0].ref == "ep-retry"
        assert "retry_backoff_ms" in response.results[0].excerpt

    def test_paraphrase_query_retrieves_right_episode(self, indexed_project: Path) -> None:
        response = search.search(indexed_project, PARAPHRASE_QUERY)
        assert response.results, "expected at least one result"
        assert response.results[0].ref == "ep-allocator"

    def test_paraphrase_query_bm25_leg_alone_returns_nothing(self, indexed_project: Path) -> None:
        """Proves the paraphrase scenario needs the vector leg: BM25 alone
        finds zero matches (Scenario: Paraphrased conceptual query...)."""
        db_path, _vec_path = search._index_paths(indexed_project)
        conn = sqlite3.connect(str(db_path))
        hits = store.search_fts(conn, search._fts_query(PARAPHRASE_QUERY), limit=500)
        conn.close()
        assert hits == []

    def test_bm25_leg_matches_only_the_literal_identifier(self, indexed_project: Path) -> None:
        db_path, _vec_path = search._index_paths(indexed_project)
        conn = sqlite3.connect(str(db_path))
        hits = store.search_fts(conn, search._fts_query("retry_backoff_ms"), limit=500)
        chunk_hits = search._fetch_chunk_hits(conn, [chunk_id for chunk_id, _r in hits])
        conn.close()
        matched_episodes = {hit.episode_id for hit in chunk_hits.values()}
        assert matched_episodes == {"ep-retry"}

    def test_single_leg_only_match_is_surfaced_regardless_of_the_other_leg(self) -> None:
        """Model-independent complement to the identifier scenario: a chunk
        present ONLY in the bm25 leg's ranked list -- e.g. because the
        vector leg's top-k pool never surfaced it at all, the worst case
        "the vector leg alone would rank them poorly" describes -- is still
        fused, rolled up, and returned. Proven here at the RRF/rollup level,
        independent of what any specific embedding model happens to do with
        any specific corpus (which is exactly the axis the two tests above
        already nail down empirically for this fixture).
        """
        chunk_hits = {
            "c-bm25-only": search._ChunkHit(
                "c-bm25-only", "ep-target", "response", "retry_backoff_ms lives here", "available"
            ),
        }
        fused = search.reciprocal_rank_fusion(["c-bm25-only"], [])  # vector leg: empty pool
        rolled = search.roll_up_to_episodes(fused, chunk_hits)
        assert "ep-target" in rolled
        assert rolled["ep-target"][0] == pytest.approx(1 / 61)

    def test_both_legs_execute_for_a_literal_shaped_query(
        self, indexed_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Single token: the OR leg would be byte-identical to the AND leg
        # and no adjacent-token phrase exists, so both are skipped --
        # exactly one FTS call.
        self._assert_legs_run(indexed_project, "retry_backoff_ms", monkeypatch, expected_fts=1)

    def test_both_legs_execute_for_a_prose_shaped_query(
        self, indexed_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Multi-token: AND leg, the recall-oriented OR leg, and the
        # phrase-proximity leg -- three FTS calls, one cosine call.
        self._assert_legs_run(indexed_project, PARAPHRASE_QUERY, monkeypatch, expected_fts=3)

    @staticmethod
    def _assert_legs_run(
        project_dir: Path,
        query: str,
        monkeypatch: pytest.MonkeyPatch,
        *,
        expected_fts: int,
    ) -> None:
        """search.py has no query-shape classifier that could skip a leg
        entirely -- the keyword and vector legs always run. Proven directly,
        per query, by spying on the two leg entry points rather than
        trusting the absence of a branch that could be reintroduced later
        without any test noticing. The exact FTS call count is asserted so
        the multi-token OR leg (and its single-token skip) cannot silently
        disappear either.
        """
        calls = {"fts": 0, "cosine": 0}
        real_fts = store.search_fts
        real_cosine = vecstore.cosine_top_k

        def spy_fts(*args: object, **kwargs: object) -> object:
            calls["fts"] += 1
            return real_fts(*args, **kwargs)  # type: ignore[arg-type]

        def spy_cosine(*args: object, **kwargs: object) -> object:
            calls["cosine"] += 1
            return real_cosine(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(store, "search_fts", spy_fts)
        monkeypatch.setattr(vecstore, "cosine_top_k", spy_cosine)
        response = search.search(project_dir, query)
        assert response.index_exists
        assert calls == {"fts": expected_fts, "cosine": 1}


# ---------------------------------------------------------------------------
# Staleness wiring: verifying staleness detector is called
# ---------------------------------------------------------------------------


class TestSearchStalenessWiring:
    """Verify staleness is detected and reported in SearchResponse.

    Tests construct distinguishable worlds (fresh vs. stale) and assert
    exact staleness values. Mutation tests prove the wiring is load-bearing.
    """

    def test_search_fresh_index_returns_stale_false_count_zero(
        self, indexed_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fresh index: stale=False, stale_count=0.

        All discovered files match their cursors.
        """
        from ssgrep import discovery

        def mock_discover(project_dir, scope=None):
            return []

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        response = search.search(indexed_project, "retry_backoff_ms")
        assert response.stale is False
        assert response.stale_count == 0

    def test_search_stale_index_returns_exact_count(
        self, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stale index: stale=True with stale_count == exact number of changed files.

        Build index, track 2 files, report them as appended. Assert count is 2.
        """
        from ssgrep import discovery

        project_dir = tmp_path_factory.mktemp("stale-search-project")
        index_dir = project_dir / search.INDEX_DIRNAME
        _build_index(index_dir, FIXTURE_EPISODES)

        file_a = project_dir / "session-a.jsonl"
        file_b = project_dir / "session-b.jsonl"
        file_a.write_text("x" * 100)
        file_b.write_text("y" * 100)

        db_path = index_dir / "index.db"
        conn = sqlite3.connect(str(db_path))
        for path in [file_a, file_b]:
            cursor = FileCursor(
                path=path, size=100, mtime=1000.0, byte_offset=0, first_line_hash="old"
            )
            store.upsert_session_file(conn, cursor)
        conn.commit()
        conn.close()

        def mock_discover(project_dir, scope=None):
            return [
                discovery.SessionFile(
                    path=file_a, session_id="s-a", is_main=True, size=200, mtime=2000.0
                ),
                discovery.SessionFile(
                    path=file_b, session_id="s-b", is_main=True, size=200, mtime=2000.0
                ),
            ]

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        response = search.search(project_dir, "retry_backoff_ms")
        assert response.stale is True
        assert response.stale_count == 2

    def test_search_mutation_staleness_call_removed_produces_stale_false_count_zero(
        self, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation test: removing staleness call leaves stale=False, count=0.

        Proves the wiring is load-bearing by showing before/after differ.
        """
        from ssgrep import discovery

        project_dir = tmp_path_factory.mktemp("mutation-search-project")
        index_dir = project_dir / search.INDEX_DIRNAME
        _build_index(index_dir, FIXTURE_EPISODES)

        file_a = project_dir / "session-a.jsonl"
        file_a.write_text("x" * 100)

        db_path = index_dir / "index.db"
        conn = sqlite3.connect(str(db_path))
        cursor = FileCursor(
            path=file_a, size=100, mtime=1000.0, byte_offset=0, first_line_hash="old"
        )
        store.upsert_session_file(conn, cursor)
        conn.commit()
        conn.close()

        def mock_discover(project_dir, scope=None):
            return [
                discovery.SessionFile(
                    path=file_a, session_id="s-a", is_main=True, size=200, mtime=2000.0
                ),
            ]

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        # Baseline: staleness detected
        response_before = search.search(project_dir, "retry_backoff_ms")
        assert response_before.stale is True
        assert response_before.stale_count == 1

        # Simulate mutation: replace staleness_summary with one that always returns unchanged
        def mock_staleness_summary_unchanged(project_dir, discovered_files=None):
            return staleness.StalenessReport(
                unchanged_count=1,
                appended_count=0,
                rewritten_count=0,
                vanished_count=0,
                stale_files=[],
            )

        monkeypatch.setattr(search, "staleness_summary", mock_staleness_summary_unchanged)

        # Mutated: staleness not detected
        response_after = search.search(project_dir, "retry_backoff_ms")
        assert response_after.stale is False
        assert response_after.stale_count == 0

        # Verify before/after are different (proves the call matters)
        assert response_before.stale != response_after.stale
        assert response_before.stale_count != response_after.stale_count


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (pure function)
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_fused_score_matches_the_documented_formula(self) -> None:
        # 'a' is rank 2 in the bm25 leg, rank 5 in the vector leg.
        bm25 = ["x", "a", "y"]
        vector = ["p", "q", "r", "s", "a"]
        result = search.reciprocal_rank_fusion(bm25, vector, k=60)
        assert result["a"] == pytest.approx(1 / 62 + 1 / 65)

    def test_single_leg_match_is_still_scored_and_returned(self) -> None:
        result = search.reciprocal_rank_fusion(["only-here"], [])
        assert result["only-here"] == pytest.approx(1 / 61)
        assert "only-here" in result

    def test_both_leg_match_outranks_equal_rank_single_leg_match(self) -> None:
        # Both 'both' and 'solo' are rank 1 in the leg they each appear in;
        # 'both' additionally appears at rank 1 in the second leg.
        result = search.reciprocal_rank_fusion(["both", "solo"], ["both"])
        assert result["both"] > result["solo"]

    def test_changing_k_changes_the_fused_score(self) -> None:
        default_k = search.reciprocal_rank_fusion(["a"], k=60)
        other_k = search.reciprocal_rank_fusion(["a"], k=10)
        assert default_k["a"] == pytest.approx(1 / 61)
        assert other_k["a"] == pytest.approx(1 / 11)
        assert default_k["a"] != other_k["a"]

    def test_dropping_a_leg_changes_the_fused_score(self) -> None:
        both_legs = search.reciprocal_rank_fusion(["a"], ["a"])
        one_leg = search.reciprocal_rank_fusion(["a"])
        assert both_legs["a"] == pytest.approx(2 / 61)
        assert one_leg["a"] == pytest.approx(1 / 61)
        assert both_legs["a"] != one_leg["a"]

    def test_empty_legs_produce_empty_result(self) -> None:
        assert search.reciprocal_rank_fusion([], []) == {}

    def test_rank_position_governs_score_not_list_order_of_arguments(self) -> None:
        scores = search.reciprocal_rank_fusion(["a", "b"], k=60)
        assert scores["a"] > scores["b"]
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62)

    def test_weight_scales_that_legs_contribution_to_the_fused_score(self) -> None:
        # 'a' is rank 1 in both legs. Halving the second leg's weight must
        # halve exactly that leg's term in the sum, not the whole score.
        unweighted = search.reciprocal_rank_fusion(["a"], ["a"], k=60)
        half_second_leg = search.reciprocal_rank_fusion(["a"], ["a"], k=60, weights=(1.0, 0.5))
        assert unweighted["a"] == pytest.approx(2 / 61)
        assert half_second_leg["a"] == pytest.approx(1 / 61 + 0.5 * (1 / 61))
        assert half_second_leg["a"] < unweighted["a"]

    def test_mismatched_weight_count_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            search.reciprocal_rank_fusion(["a"], ["b"], weights=(1.0,))
        with pytest.raises(ValueError):
            search.reciprocal_rank_fusion(["a"], weights=(1.0, 1.0))

    def test_zero_weight_leg_is_computed_but_contributes_nothing(self) -> None:
        # A zero-weighted leg's own exclusive hits must still be *returned*
        # (rolled up downstream, e.g. for excerpting) but score zero -- this
        # is different from dropping the leg entirely.
        both_legs = search.reciprocal_rank_fusion(["a"], ["a"], weights=(1.0, 0.0))
        assert both_legs["a"] == pytest.approx(1 / 61)
        zero_leg_only = search.reciprocal_rank_fusion([], ["solo"], weights=(1.0, 0.0))
        assert "solo" in zero_leg_only
        assert zero_leg_only["solo"] == pytest.approx(0.0)

    def test_negative_weight_subtracts_from_the_fused_score(self) -> None:
        # The function does not itself forbid negative weights; a leg
        # weighted -1.0 must cancel an equal positive-weighted leg exactly.
        result = search.reciprocal_rank_fusion(["a"], ["a"], weights=(1.0, -1.0))
        assert result["a"] == pytest.approx(0.0)

    def test_production_five_leg_weight_tuple_is_applied_as_documented(self) -> None:
        # Pins the exact weight tuple search() fuses its five legs with
        # (AND-BM25, vector, OR-BM25, trigram, phrase) so an accidental
        # change to any of the three named constants, or to how weights
        # are applied, is caught here rather than only surfacing as an
        # unexplained recall regression in the eval harness.
        weights = (
            1.0,
            1.0,
            search.OR_LEG_WEIGHT,
            search.TRIGRAM_LEG_WEIGHT,
            search.PHRASE_LEG_WEIGHT,
        )
        assert weights == (1.0, 1.0, 0.9, 1.0, 0.5)
        legs = (["a"], ["a"], ["a"], ["a"], ["a"])
        result = search.reciprocal_rank_fusion(*legs, k=60, weights=weights)
        assert result["a"] == pytest.approx(sum(w / 61 for w in weights))


# ---------------------------------------------------------------------------
# Episode-level roll-up (pure function)
# ---------------------------------------------------------------------------


class TestRollUpToEpisodes:
    def test_episode_score_is_max_plus_bounded_tail_not_open_sum(self) -> None:
        chunk_hits = {
            "c1": search._ChunkHit("c1", "ep-A", "prompt", "chunk one", "available"),
            "c2": search._ChunkHit("c2", "ep-A", "response", "chunk two", "available"),
            "c3": search._ChunkHit("c3", "ep-B", "prompt", "chunk three", "available"),
        }
        # ep-A has two weak matching chunks; ep-B has one strong one. The
        # roll-up requirement's property: one clearly stronger chunk beats
        # any pile of weak ones -- the tail is bounded, never an open sum.
        chunk_scores = {"c1": 0.01, "c2": 0.01, "c3": 0.05}
        rolled = search.roll_up_to_episodes(chunk_scores, chunk_hits)
        assert rolled["ep-A"][0] == pytest.approx(0.01 + search.ROLLUP_TAIL_WEIGHT * 0.01)
        assert rolled["ep-B"][0] == pytest.approx(0.05)
        assert rolled["ep-B"][0] > rolled["ep-A"][0]

    def test_tail_contribution_is_capped_at_rollup_tail_chunks(self) -> None:
        # Ten equal-scored chunks in one episode: only the best plus
        # ROLLUP_TAIL_CHUNKS - 1 tail chunks may count. An open sum would
        # score 10x the single-chunk episode; the cap keeps it bounded.
        chunk_hits = {
            f"c{i}": search._ChunkHit(f"c{i}", "ep-many", "prompt", f"t{i}", "available")
            for i in range(10)
        }
        chunk_scores = {f"c{i}": 0.01 for i in range(10)}
        rolled = search.roll_up_to_episodes(chunk_scores, chunk_hits)
        expected = 0.01 + search.ROLLUP_TAIL_WEIGHT * 0.01 * (search.ROLLUP_TAIL_CHUNKS - 1)
        assert rolled["ep-many"][0] == pytest.approx(expected)
        assert rolled["ep-many"][0] < 0.05  # still loses to one 5x-stronger chunk

    def test_excerpt_is_the_winning_chunks_text(self) -> None:
        chunk_hits = {
            "lo": search._ChunkHit("lo", "ep-A", "prompt", "low score text", "available"),
            "hi": search._ChunkHit("hi", "ep-A", "response", "high score text", "available"),
        }
        chunk_scores = {"lo": 0.01, "hi": 0.05}
        rolled = search.roll_up_to_episodes(chunk_scores, chunk_hits)
        assert rolled["ep-A"][1].text == "high score text"

    def test_three_matching_chunks_collapse_to_one_episode_entry(self) -> None:
        chunk_hits = {
            f"c{i}": search._ChunkHit(f"c{i}", "ep-A", "prompt", f"text {i}", "available")
            for i in range(3)
        }
        chunk_scores = {"c0": 0.01, "c1": 0.02, "c2": 0.015}
        rolled = search.roll_up_to_episodes(chunk_scores, chunk_hits)
        assert list(rolled.keys()) == ["ep-A"]
        assert rolled["ep-A"][0] == pytest.approx(0.02 + search.ROLLUP_TAIL_WEIGHT * (0.015 + 0.01))

    def test_chunk_with_no_metadata_is_dropped_not_crashed_on(self) -> None:
        rolled = search.roll_up_to_episodes({"missing": 0.5}, {})
        assert rolled == {}


# ---------------------------------------------------------------------------
# Main-session rank preference (pure function)
# ---------------------------------------------------------------------------


class TestMainSessionBoost:
    def test_boost_zero_disables_the_preference(self) -> None:
        scores = {"main": 0.02, "sub": 0.015}
        is_subagent = {"main": False, "sub": True}
        result = search.apply_main_session_boost(scores, is_subagent, boost=0.0)
        assert result == scores

    def test_equal_scores_favor_the_main_session(self) -> None:
        """A positive boost breaks ties toward main session.

        Uses an explicit positive boost rather than search.MAIN_SESSION_BOOST:
        the measured shipped default is 0.0 (see the provenance comment on
        that constant), which by design does NOT break ties -- that fact is
        covered by test_boost_zero_disables_the_preference above. This test
        is about the mechanism's general tie-breaking behavior at some
        positive boost, independent of whatever today's default happens to
        be.
        """
        scores = {"main": 0.02, "sub": 0.02}
        is_subagent = {"main": False, "sub": True}
        result = search.apply_main_session_boost(scores, is_subagent, boost=0.01)
        assert result["main"] > result["sub"]

    def test_bounded_boost_cannot_overturn_a_substantially_better_match(self) -> None:
        """Uses an explicit positive boost; see
        test_equal_scores_favor_the_main_session's docstring for why -- at
        the measured shipped default of 0.0 this would degenerate to a
        trivial 0.05 > 0.005 check that never exercises bounding at all.
        """
        scores = {"main": 0.005, "sub": 0.05}  # 10x gap, far larger than the boost
        is_subagent = {"main": False, "sub": True}
        result = search.apply_main_session_boost(scores, is_subagent, boost=0.01)
        assert result["sub"] > result["main"]

    def test_subagent_scores_are_never_boosted(self) -> None:
        """Uses an explicit positive boost; see
        test_equal_scores_favor_the_main_session's docstring for why -- at
        boost=0.0 nothing is ever boosted, main-session included, so this
        wouldn't distinguish "subagents are excluded" from "the boost is
        off".
        """
        scores = {"sub": 0.02}
        is_subagent = {"sub": True}
        result = search.apply_main_session_boost(scores, is_subagent, boost=0.01)
        assert result["sub"] == pytest.approx(0.02)

    def test_shipped_default_matches_measured_optimum(self) -> None:
        """Pin MAIN_SESSION_BOOST to the eval-harness-measured optimum.

        The eval harness was built specifically to tune this constant
        rather than guess it, and found that 0.0 strictly dominates every
        positive boost tested on recall@10 and MRR alike -- but that
        finding was never applied to the shipped constant, which stayed
        at the original 0.01 design-time estimate. See the provenance
        comment directly above MAIN_SESSION_BOOST's definition in
        search.py for the full curve, and
        eval/results/boost_sweep_2026-07-27.json for the raw per-boost,
        per-class numbers it's drawn from.

        This test is the guard against that exact failure mode recurring:
        a future change to MAIN_SESSION_BOOST should come with a fresh
        harness run justifying the new value, which means updating this
        assertion (and the comment it points at) is expected. A change
        that does neither is a silent regression, and this test exists to
        make it a loud one instead -- recall@10 fell from 77.8% to 58.3%,
        and subagent-only recall fell from 69.2% to 30.8%, the last time
        this constant drifted from the measured optimum.
        """
        assert search.MAIN_SESSION_BOOST == 0.0


# ---------------------------------------------------------------------------
# Deterministic ordering and tie-breaking (pure function)
# ---------------------------------------------------------------------------


def _episode_row(episode_id: str, timestamp: datetime | None) -> search._EpisodeRow:
    return search._EpisodeRow(
        episode_id=episode_id,
        session_id="s1",
        title="t",
        timestamp=timestamp,
        git_branch=None,
        files_touched=(),
        is_subagent=False,
        agent_name=None,
        agent_description=None,
        parent_session_id=None,
        source_status="available",
    )


class TestRankEpisodes:
    def test_orders_by_score_descending(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        scores = {"a": 0.1, "b": 0.3, "c": 0.2}
        rows = {eid: _episode_row(eid, ts) for eid in scores}
        assert search._rank_episodes(scores, rows) == ["b", "c", "a"]

    def test_equal_score_orders_by_recency(self) -> None:
        scores = {"old": 0.1, "new": 0.1}
        rows = {
            "old": _episode_row("old", datetime(2020, 1, 1, tzinfo=UTC)),
            "new": _episode_row("new", datetime(2026, 1, 1, tzinfo=UTC)),
        }
        assert search._rank_episodes(scores, rows) == ["new", "old"]

    def test_equal_score_and_timestamp_resolves_by_episode_id(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        scores = {"zzz": 0.1, "aaa": 0.1}
        rows = {eid: _episode_row(eid, ts) for eid in scores}
        assert search._rank_episodes(scores, rows) == ["aaa", "zzz"]

    def test_ordering_is_deterministic_across_repeated_calls(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        scores = {f"ep-{i}": float(i % 3) * 0.01 for i in range(30)}
        rows = {eid: _episode_row(eid, ts) for eid in scores}
        first = search._rank_episodes(scores, rows)
        second = search._rank_episodes(dict(scores), rows)
        assert first == second

    def test_missing_timestamp_does_not_crash_and_sorts_last_among_ties(self) -> None:
        scores = {"has_ts": 0.1, "no_ts": 0.1}
        rows = {
            "has_ts": _episode_row("has_ts", datetime(2020, 1, 1, tzinfo=UTC)),
            "no_ts": _episode_row("no_ts", None),
        }
        assert search._rank_episodes(scores, rows) == ["has_ts", "no_ts"]


# ---------------------------------------------------------------------------
# Composable filters (end-to-end against the real index)
# ---------------------------------------------------------------------------


class TestComposableFilters:
    # Broad enough that the vector leg (LEG_POOL_SIZE=500 exceeds this tiny
    # corpus) surfaces every episode regardless of literal wording, so
    # unfiltered total_matches always equals the full fixture count.
    QUERY = "cache"

    def test_no_filters_matches_every_episode(self, indexed_project: Path) -> None:
        response = search.search(indexed_project, self.QUERY, limit=search.MAX_RESULT_COUNT)
        assert response.total_matches == len(FIXTURE_EPISODES)

    def test_branch_filter(self, indexed_project: Path) -> None:
        response = search.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(branch="main"),
        )
        refs = {r.ref for r in response.results}
        assert refs == {"ep-cache", "ep-allocator", "ep-audit", "ep-retry", "ep-subagent-deploy"}

    def test_date_from_filter(self, indexed_project: Path) -> None:
        response = search.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(date_from=datetime(2026, 7, 8, tzinfo=UTC)),
        )
        refs = {r.ref for r in response.results}
        assert refs == {"ep-metrics", "ep-audit", "ep-phases", "ep-retry", "ep-subagent-deploy"}

    def test_date_range_filter(self, indexed_project: Path) -> None:
        response = search.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(
                date_from=datetime(2026, 7, 4, tzinfo=UTC),
                date_to=datetime(2026, 7, 16, tzinfo=UTC),
            ),
        )
        refs = {r.ref for r in response.results}
        assert refs == {"ep-allocator", "ep-metrics", "ep-audit"}

    def test_date_filter_naive_and_aware_datetimes_compare_safely(
        self, indexed_project: Path
    ) -> None:
        """episode.timestamp round-trips through the store as tz-aware
        (ISO-8601 with offset); a caller-supplied naive filter datetime
        must not raise TypeError when compared against it."""
        response = search.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(date_from=datetime(2026, 7, 8)),  # naive, no tzinfo
        )
        refs = {r.ref for r in response.results}
        assert refs == {"ep-metrics", "ep-audit", "ep-phases", "ep-retry", "ep-subagent-deploy"}

    def test_file_path_filter(self, indexed_project: Path) -> None:
        response = search.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(file_path="settings.py"),
        )
        refs = {r.ref for r in response.results}
        assert refs == {"ep-cache"}

    def test_content_type_filter_changes_excerpt_not_episode_membership(
        self, indexed_project: Path
    ) -> None:
        prompt_only = search.search(
            indexed_project,
            "widget cache TTL",
            filters=SearchFilters(content_type=ContentType.PROMPT),
        )
        response_only = search.search(
            indexed_project,
            "widget cache TTL",
            filters=SearchFilters(content_type=ContentType.RESPONSE),
        )
        cache_prompt = next(r for r in prompt_only.results if r.ref == "ep-cache")
        cache_response = next(r for r in response_only.results if r.ref == "ep-cache")
        assert cache_prompt.content_type == ContentType.PROMPT
        assert cache_response.content_type == ContentType.RESPONSE
        assert cache_prompt.excerpt != cache_response.excerpt

    def test_combined_filters_narrow_to_their_intersection(self, indexed_project: Path) -> None:
        response = search.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(branch="main", date_from=datetime(2026, 7, 8, tzinfo=UTC)),
        )
        refs = {r.ref for r in response.results}
        assert refs == {"ep-audit", "ep-retry", "ep-subagent-deploy"}

    def test_all_three_named_filters_at_once(self, indexed_project: Path) -> None:
        """date, path, and content type together -- the spec names these
        three explicitly as the filters that must compose."""
        response = search.search(
            indexed_project,
            "widget cache",
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(
                date_from=datetime(2026, 6, 1, tzinfo=UTC),
                file_path="settings.py",
                content_type=ContentType.PROMPT,
            ),
        )
        assert {r.ref for r in response.results} == {"ep-cache"}
        assert response.results[0].content_type == ContentType.PROMPT

    def test_over_restrictive_filter_combination_yields_empty_not_error(
        self, indexed_project: Path
    ) -> None:
        response = search.search(
            indexed_project, self.QUERY, filters=SearchFilters(branch="nonexistent-branch")
        )
        assert response.results == []
        assert response.index_exists is True
        assert response.index_empty is False
        assert response.total_matches == 0


class TestApiSearchForwardsFilters:
    """ssgrep.api.search() is the frozen contract boundary every CLI command
    and MCP tool actually calls -- TestComposableFilters above proves
    search.search() (the module-level implementation) honors filters
    correctly, but every one of those tests calls search.search() directly
    and never once goes through api.search(). A caller-supplied `filters`
    could be silently dropped at the api.py boundary (e.g. a call passing
    `filters=None` instead of `filters=filters` through to search.search())
    and every test in this file, and every other test in the suite, would
    keep passing: confirmed directly by planting exactly that mutation in
    api.py and running the full suite (602 passed, only an unrelated,
    already-failing concurrency test affected; every filter-related test,
    including all of TestComposableFilters, stayed green). These tests
    close that gap by asserting on api.search()'s own return value, not
    search.search()'s.
    """

    QUERY = "cache"

    def test_unfiltered_and_filtered_api_search_return_different_result_sets(
        self, indexed_project: Path
    ) -> None:
        """The load-bearing assertion for this gap: if api.search() silently
        dropped `filters` (e.g. forwarding None instead of the caller's
        value), a filtered call would return the exact same result set as
        an unfiltered one. It must not.
        """
        unfiltered = api.search(indexed_project, self.QUERY, limit=search.MAX_RESULT_COUNT)
        filtered = api.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(branch="main"),
        )
        unfiltered_refs = {r.ref for r in unfiltered.results}
        filtered_refs = {r.ref for r in filtered.results}

        assert unfiltered_refs == {
            r.episode_id for r in FIXTURE_EPISODES
        }, "sanity check: an unfiltered api.search() call should match every fixture episode"
        assert filtered_refs != unfiltered_refs, (
            "api.search() with a branch filter returned the SAME result set as an unfiltered "
            "call -- the filter had no effect, meaning api.search() is not forwarding `filters` "
            "through to search.search()"
        )

    def test_api_search_branch_filter_matches_the_known_correct_set(
        self, indexed_project: Path
    ) -> None:
        """Cross-checks api.search()'s filtered output against the exact
        expected set TestComposableFilters.test_branch_filter already
        proves is correct for search.search() with the identical filter --
        confirming api.search() is a true, value-preserving passthrough,
        not merely "narrows to something."
        """
        response = api.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(branch="main"),
        )
        refs = {r.ref for r in response.results}
        assert refs == {"ep-cache", "ep-allocator", "ep-audit", "ep-retry", "ep-subagent-deploy"}

    def test_api_search_file_path_filter_narrows_via_the_api_boundary(
        self, indexed_project: Path
    ) -> None:
        """A second, independently-shaped filter (file_path rather than
        branch) through the same api.search() boundary, so this gap isn't
        closed for exactly one filter field and left open for the rest.
        """
        response = api.search(
            indexed_project,
            self.QUERY,
            limit=search.MAX_RESULT_COUNT,
            filters=SearchFilters(file_path="settings.py"),
        )
        refs = {r.ref for r in response.results}
        assert refs == {"ep-cache"}

    def test_api_search_over_restrictive_filter_yields_empty_not_full_set(
        self, indexed_project: Path
    ) -> None:
        """If filters were dropped at the api.py boundary, an intentionally
        impossible filter (a branch that matches nothing) would still
        return the full unfiltered set instead of zero results.
        """
        response = api.search(
            indexed_project, self.QUERY, filters=SearchFilters(branch="nonexistent-branch")
        )
        assert response.results == []
        assert response.total_matches == 0


# ---------------------------------------------------------------------------
# Default and maximum result count
# ---------------------------------------------------------------------------


class TestResultCountLimits:
    def test_default_count_is_small_and_not_clamped(self, indexed_project: Path) -> None:
        response = search.search(indexed_project, "cache")
        assert len(response.results) <= search.DEFAULT_RESULT_COUNT
        assert response.clamped is False

    def test_override_within_range_is_honored(self, indexed_project: Path) -> None:
        response = search.search(indexed_project, "cache", limit=3)
        assert len(response.results) <= 3
        assert response.clamped is False

    def test_override_above_max_is_clamped_and_stated(self, indexed_project: Path) -> None:
        response = search.search(indexed_project, "cache", limit=search.MAX_RESULT_COUNT + 500)
        assert response.clamped is True
        assert len(response.results) <= search.MAX_RESULT_COUNT

    def test_default_limit_caps_a_large_synthetic_match_set(self) -> None:
        """The fixture corpus only has 7 episodes, too few to prove a
        default cap of 10 by itself; this drives _build_response directly
        with 25 synthetic episodes to prove the cap for real."""
        rolled_up = {}
        episode_rows = {}
        for i in range(25):
            eid = f"ep-{i}"
            hit = search._ChunkHit(f"c{i}", eid, "prompt", f"text {i}", "available")
            rolled_up[eid] = (1.0 - i * 0.001, hit)
            episode_rows[eid] = _episode_row(eid, datetime(2026, 1, 1 + (i % 27), tzinfo=UTC))
        response = search._build_response(
            rolled_up,
            episode_rows,
            limit=search.DEFAULT_RESULT_COUNT,
            token_budget=1_000_000,
            clamped=False,
        )
        assert response.total_matches == 25
        assert len(response.results) == search.DEFAULT_RESULT_COUNT


# ---------------------------------------------------------------------------
# Degenerate query and index handling
# ---------------------------------------------------------------------------


class TestDegenerateQueryAndIndex:
    def test_empty_query_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EmptyQueryError):
            search.search(tmp_path, "")

    def test_whitespace_only_query_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EmptyQueryError):
            search.search(tmp_path, "   \n\t  ")

    def test_empty_query_never_touches_the_index(self, tmp_path: Path) -> None:
        # No .ssgrep/ exists at all; if EmptyQueryError is genuinely raised
        # before any index access, nothing gets created as a side effect.
        with pytest.raises(EmptyQueryError):
            search.search(tmp_path, "")
        assert not (tmp_path / search.INDEX_DIRNAME).exists()

    def test_no_index_raises_index_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(IndexNotFoundError):
            search.search(tmp_path, "anything")

    def test_no_index_search_never_creates_one(self, tmp_path: Path) -> None:
        with pytest.raises(IndexNotFoundError):
            search.search(tmp_path, "anything")
        assert not (tmp_path / search.INDEX_DIRNAME).exists()

    def test_empty_index_returns_index_empty_not_error(self, tmp_path: Path) -> None:
        index_dir = tmp_path / search.INDEX_DIRNAME
        store.init_db(index_dir / "index.db").close()
        response = search.search(tmp_path, "anything")
        assert response.index_exists is True
        assert response.index_empty is True
        assert response.results == []
        assert response.total_matches == 0

    def test_schema_mismatch_raises_index_not_ready(self, tmp_path: Path) -> None:
        index_dir = tmp_path / search.INDEX_DIRNAME
        conn = store.init_db(index_dir / "index.db")
        store.set_meta(conn, "schema_version", "999")
        conn.commit()
        conn.close()
        with pytest.raises(IndexNotReadyError):
            search.search(tmp_path, "anything")

    def test_vector_alignment_corruption_raises_index_not_ready(self, tmp_path: Path) -> None:
        index_dir = tmp_path / search.INDEX_DIRNAME
        conn = store.init_db(index_dir / "index.db")
        chunk = Chunk(
            chunk_id="c1",
            episode_id="e1",
            session_id="s1",
            text="hello",
            content_type=ContentType.PROMPT,
        )
        store.insert_chunk(conn, chunk, vec_row=999)  # no such vector row exists
        conn.commit()
        conn.close()
        with pytest.raises(IndexNotReadyError):
            search.search(tmp_path, "anything")

    def test_well_formed_query_matching_nothing_is_not_an_error(self, tmp_path: Path) -> None:
        # An index that exists, has chunks, but whose only content is
        # filtered out entirely (content_type=RESPONSE excludes the sole
        # PROMPT chunk) is the reachable, real way to force a genuine
        # zero-match response under this design: brute-force cosine always
        # ranks the full corpus, so the vector leg alone never organically
        # returns nothing while any chunk exists (see search.py's
        # LEG_POOL_SIZE comment) -- filters are what make "matches nothing"
        # observable.
        index_dir = tmp_path / search.INDEX_DIRNAME
        conn = store.init_db(index_dir / "index.db")
        vstore = vecstore.open_vectors(index_dir / "vectors.f32", dimension=embed.DIMENSION)
        chunk = Chunk(
            chunk_id="c1",
            episode_id="e1",
            session_id="s1",
            text="only a prompt chunk exists in this tiny index",
            content_type=ContentType.PROMPT,
        )
        vecs = embed.encode([chunk.text])
        rows = vecstore.append(vstore, vecs)
        store.insert_chunk(conn, chunk, rows[0])
        conn.commit()
        conn.close()

        response = search.search(
            tmp_path, "prompt chunk", filters=SearchFilters(content_type=ContentType.RESPONSE)
        )
        assert response.results == []
        assert response.index_exists is True
        assert response.index_empty is False
        assert response.total_matches == 0


# ---------------------------------------------------------------------------
# Result card composition and subagent attribution
# ---------------------------------------------------------------------------


class TestResultCardComposition:
    def test_result_card_has_all_required_fields(self, indexed_project: Path) -> None:
        response = search.search(indexed_project, "widget cache TTL settings")
        assert response.results
        card = response.results[0]
        assert card.ref
        assert card.title
        assert card.score > 0
        assert isinstance(card.files_touched, tuple)
        assert card.excerpt

    def test_result_card_score_normalized_is_derived_from_the_live_ceiling(
        self, indexed_project: Path
    ) -> None:
        """score_normalized must equal score / search.SCORE_CEILING (imported,
        not copied -- a retune that moves the ceiling must move this test's
        expectation with it), clamped to (0, 1]."""
        response = search.search(indexed_project, "widget cache TTL settings")
        assert response.results
        card = response.results[0]
        assert 0.0 < card.score_normalized <= 1.0
        expected = min(card.score / search.SCORE_CEILING, 1.0)
        assert card.score_normalized == pytest.approx(expected), (
            "score_normalized must be score / SCORE_CEILING, computed from the "
            "live fusion constants"
        )

    def test_score_ceiling_derives_from_the_live_fusion_constants(self) -> None:
        """The ceiling is import-not-copy: recompute it from the shipped
        constants and require exact agreement, so a leg-weight or roll-up
        retune that forgets the ceiling goes red here."""
        expected = (
            (
                1.0
                + 1.0
                + search.OR_LEG_WEIGHT
                + search.TRIGRAM_LEG_WEIGHT
                + search.PHRASE_LEG_WEIGHT
                + search.RERANK_LEG_WEIGHT
            )
            / (search.RRF_K + 1)
            * (1.0 + search.ROLLUP_TAIL_WEIGHT * (search.ROLLUP_TAIL_CHUNKS - 1))
        )
        assert search.SCORE_CEILING == pytest.approx(expected)

    def test_episode_with_no_files_touched_still_produces_valid_card(
        self, indexed_project: Path
    ) -> None:
        response = search.search(indexed_project, "legacy audit cache module", limit=50)
        card = next(c for c in response.results if c.ref == "ep-audit")
        assert card.files_touched == ()
        assert card.title

    def test_excerpt_is_exactly_one_chunks_text_not_the_full_episode(
        self, indexed_project: Path
    ) -> None:
        response = search.search(indexed_project, "widget cache TTL settings")
        card = next(c for c in response.results if c.ref == "ep-cache")
        full_prompt = "How do I configure the widget cache TTL?"
        full_response = "You can set the widget cache TTL in the cache settings module."
        assert not (full_prompt in card.excerpt and full_response in card.excerpt)


class TestSubagentAttribution:
    def test_subagent_card_carries_attribution(self, tmp_path: Path) -> None:
        episode = Episode(
            episode_id="ep-sub",
            session_id="parent-session-1:agent:abc",
            prompt_text="investigate the flaky test",
            response_text="Root cause: a shared fixture leaked state between test runs.",
            title="Investigate flaky test",
            timestamp=datetime(2026, 7, 1, tzinfo=UTC),
            is_subagent=True,
            agent_type="general-purpose",
            agent_name="flaky-test-investigator",
            agent_description="Investigate why test_foo is flaky",
            parent_session_id="parent-session-1",
        )
        _build_index(tmp_path / search.INDEX_DIRNAME, [episode])
        response = search.search(tmp_path, "flaky test")
        assert len(response.results) == 1
        card = response.results[0]
        assert card.is_subagent is True
        assert card.agent_name == "flaky-test-investigator"
        assert card.agent_description == "Investigate why test_foo is flaky"
        assert card.parent_session_id == "parent-session-1"

    def test_shared_fixture_subagent_card_carries_attribution(self, indexed_project: Path) -> None:
        response = search.search(
            indexed_project, "deployment pipeline hangs stdout sentinel", limit=50
        )
        card = next(c for c in response.results if c.ref == "ep-subagent-deploy")
        assert card.is_subagent is True
        assert card.agent_name == "deploy-debugger"
        assert card.agent_description == "Investigate CI pipeline hang"
        assert card.parent_session_id == "s1"

    def test_main_session_card_has_no_fabricated_attribution(self, indexed_project: Path) -> None:
        response = search.search(indexed_project, "cache")
        card = next(r for r in response.results if r.ref == "ep-cache")
        assert card.is_subagent is False
        assert card.agent_name is None
        assert card.agent_description is None
        assert card.parent_session_id is None


# ---------------------------------------------------------------------------
# Determinism end-to-end
# ---------------------------------------------------------------------------


class TestDeterminismEndToEnd:
    def test_repeated_identical_query_yields_identical_order(self, indexed_project: Path) -> None:
        first = search.search(indexed_project, "pool")
        second = search.search(indexed_project, "pool")
        assert [c.ref for c in first.results] == [c.ref for c in second.results]
        assert [c.score for c in first.results] == [c.score for c in second.results]


# ---------------------------------------------------------------------------
# Token-budgeted response shaping
# ---------------------------------------------------------------------------


def _card(ref: str, score: float, excerpt: str, title: str = "Title") -> ResultCard:
    return ResultCard(
        ref=ref,
        title=title,
        timestamp=None,
        score=score,
        excerpt=excerpt,
        files_touched=(),
        is_subagent=False,
    )


class TestApplyTokenBudget:
    def test_response_under_budget_is_returned_unmodified(self) -> None:
        cards = [_card("a", 0.9, "a short excerpt")]
        kept, omitted, truncated = search._apply_token_budget(cards, 1500)
        assert kept == cards
        assert omitted == 0
        assert truncated is False

    def test_over_budget_truncates_the_excerpt_before_dropping(self) -> None:
        cards = [_card("a", 0.9, "word " * 400), _card("b", 0.8, "short excerpt")]
        kept, _omitted, truncated = search._apply_token_budget(cards, 120)
        assert truncated is True
        assert len(kept[0].excerpt) < len(cards[0].excerpt)
        assert kept[0].ref == "a"  # truncated, not dropped

    def test_drops_are_a_contiguous_ranked_tail(self) -> None:
        cards = [_card(f"c{i}", 1.0 - i * 0.01, "x" * 4000) for i in range(5)]
        kept, omitted, _truncated = search._apply_token_budget(cards, 80)
        assert omitted > 0
        assert [c.ref for c in kept] == [c.ref for c in cards[: len(kept)]]

    def test_omitted_count_equals_dropped_count(self) -> None:
        cards = [_card(f"c{i}", 1.0, "y" * 50) for i in range(20)]
        kept, omitted, _truncated = search._apply_token_budget(cards, 200)
        assert omitted == len(cards) - len(kept)

    def test_dropped_results_are_reported_not_silent(self) -> None:
        cards = [_card(f"c{i}", 1.0, "z" * 4000) for i in range(10)]
        _kept, omitted, _truncated = search._apply_token_budget(cards, 100)
        assert omitted > 0

    def test_configurable_budget_changes_how_much_survives(self) -> None:
        cards = [_card(f"c{i}", 1.0, "x" * 400) for i in range(10)]
        kept_small, _o1, _t1 = search._apply_token_budget(cards, 100)
        kept_large, _o2, _t2 = search._apply_token_budget(cards, 5000)
        assert len(kept_large) > len(kept_small)

    def test_truncate_text_handles_tiny_max_chars_without_crashing(self) -> None:
        result, truncated = search._truncate_text("hello world", 2)
        assert truncated is True
        assert result == "he"

    def test_truncate_text_under_limit_is_unmodified(self) -> None:
        result, truncated = search._truncate_text("short", 100)
        assert truncated is False
        assert result == "short"

    def test_estimate_tokens_is_at_least_one_for_any_nonempty_text(self) -> None:
        assert search._estimate_tokens("") == 1
        assert search._estimate_tokens("hi") == 1
        assert search._estimate_tokens("x" * 40) == 10


# ---------------------------------------------------------------------------
# FTS5 query sanitization
# ---------------------------------------------------------------------------


class TestFtsQuerySanitizer:
    def test_literal_error_string_does_not_raise(self, tmp_path: Path) -> None:
        conn = store.init_db(tmp_path / "test.db")
        query = search._fts_query("TypeError: cannot read property 'x' of undefined")
        # Must not raise sqlite3.OperationalError.
        conn.execute("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ?", (query,))
        conn.close()

    def test_config_key_matches_literally(self, tmp_path: Path) -> None:
        conn = store.init_db(tmp_path / "test.db")
        chunk = Chunk(
            chunk_id="c1",
            episode_id="e1",
            session_id="s1",
            text="set retry_backoff_ms to 500 to slow down retries",
            content_type=ContentType.PROMPT,
        )
        store.insert_chunk(conn, chunk, 0)
        conn.commit()
        results = store.search_fts(conn, search._fts_query("retry_backoff_ms"))
        conn.close()
        assert [r[0] for r in results] == ["c1"]

    def test_unescaped_raw_query_is_what_the_sanitizer_avoids(self, tmp_path: Path) -> None:
        """Documents *why* _fts_query exists: run the raw literal straight
        through MATCH with no sanitizing and it breaks FTS5 syntax."""
        conn = store.init_db(tmp_path / "test.db")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ?",
                ("TypeError: cannot read property 'x' of undefined",),
            )
        conn.close()

    def test_tokens_combine_with_and_not_or(self) -> None:
        # A multi-word query only matches a chunk containing every token --
        # this is what makes zero-shared-token paraphrases correctly return
        # nothing from this leg rather than loosely matching on stray words.
        query = search._fts_query("alpha zzz_absent_token")
        assert query == '"alpha" "zzz_absent_token"'

    def test_embedded_double_quote_is_escaped_by_doubling(self) -> None:
        assert search._fts_query('foo"bar') == '"foo""bar"'

    def test_embedded_quote_query_does_not_raise_and_matches_nothing_spurious(
        self, tmp_path: Path
    ) -> None:
        conn = store.init_db(tmp_path / "test.db")
        chunk = Chunk(
            chunk_id="c1",
            episode_id="e1",
            session_id="s1",
            text="ordinary text with no quotes",
            content_type=ContentType.PROMPT,
        )
        store.insert_chunk(conn, chunk, 0)
        conn.commit()
        results = store.search_fts(conn, search._fts_query('say "hi" now'))
        conn.close()
        assert results == []

    ADVERSARIAL_QUERIES = [
        "TypeError: cannot read property 'x' of undefined",
        'a " b """ c ""',
        "/ // /// -- - ~ !",
        "NEAR AND OR NOT MATCH",
        "near(a b, 5)",
        "text:foo chunk_id:bar",
        "(unbalanced ( parens",
        "*star* pre* *post",
        "colons: everywhere: here:",
        "emoji \U0001f680 unicode \u00fcn\u00efc\u00f6d\u00e9 \u4e2d\u6587\u5b57\u7b26",
        'back\\slash \\"escaped\\"',
        "^caret $dollar %percent",
        "a" * 300 + " huge token",
        '"quoted phrase" plus more',
        "single 'quotes' everywhere",
        ".... .... ....",
        "?? !! [] {} <>",
        "tab\thand\nnewline",
        "-9223372036854775808 1e308",
        '"',
    ]

    def test_all_four_match_builders_survive_adversarial_input(self, tmp_path: Path) -> None:
        """Every MATCH-expression builder must neutralize FTS5 syntax.

        _fts_query has always had this guarantee; the OR, trigram, and
        phrase legs added in schema v4 build their own MATCH expressions
        from the same raw user input and would crash search() with
        sqlite3.OperationalError if their quoting discipline slipped. Run
        each builder over inputs full of FTS5 syntax hazards (keywords,
        stray quotes, column filters, unbalanced parens, globs) and execute
        the result against real FTS5 tables -- the assertion is that no
        expression raises. A builder returning an empty expression for an
        input is fine (search() skips the leg); raising is the defect.
        """
        conn = store.init_db(tmp_path / "test.db")
        chunk = Chunk(
            chunk_id="c1",
            episode_id="e1",
            session_id="s1",
            text="ordinary indexed text so MATCH has something to scan",
            content_type=ContentType.PROMPT,
        )
        store.insert_chunk(conn, chunk, 0)
        conn.commit()
        builders = [
            (search._fts_query, store.search_fts),
            (search._fts_query_any, store.search_fts),
            (search._trigram_query, store.search_fts_trigram),
            (search._phrase_query, store.search_fts),
        ]
        exercised = 0
        for query in self.ADVERSARIAL_QUERIES:
            for build, run_match in builders:
                expression = build(query)
                if expression:
                    run_match(conn, expression, limit=5)  # must not raise
                    exercised += 1
        conn.close()
        # Positive assertion: the loop genuinely executed MATCH statements,
        # it did not vacuously skip everything via empty expressions.
        assert exercised > 40


# ---------------------------------------------------------------------------
# staleness.py adapter
# ---------------------------------------------------------------------------


class TestStalenessSummary:
    def test_reports_appended_when_disk_outgrew_the_cursor(self, tmp_path: Path) -> None:
        index_dir = tmp_path / search.INDEX_DIRNAME
        conn = store.init_db(index_dir / "index.db")
        store.upsert_session_file(
            conn,
            FileCursor(
                path=tmp_path / "transcript.jsonl",
                size=100,
                mtime=1000.0,
                byte_offset=0,
                first_line_hash="abc",
            ),
        )
        conn.commit()
        conn.close()

        discovered = [(tmp_path / "transcript.jsonl", 250, 1000.0)]  # grew since cursor
        report = search.staleness_summary(tmp_path, discovered_files=discovered)
        assert report.appended_count == 1
        assert report.unchanged_count == 0

    def test_reports_unchanged_when_cursor_matches_disk(self, tmp_path: Path) -> None:
        index_dir = tmp_path / search.INDEX_DIRNAME
        conn = store.init_db(index_dir / "index.db")
        store.upsert_session_file(
            conn,
            FileCursor(
                path=tmp_path / "t.jsonl",
                size=100,
                mtime=1000.0,
                byte_offset=0,
                first_line_hash="abc",
            ),
        )
        conn.commit()
        conn.close()

        discovered = [(tmp_path / "t.jsonl", 100, 1000.0)]
        report = search.staleness_summary(tmp_path, discovered_files=discovered)
        assert report.unchanged_count == 1
        assert report.appended_count == 0

    def test_no_index_treats_every_discovered_file_as_appended(self, tmp_path: Path) -> None:
        discovered = [(tmp_path / "t.jsonl", 100, 1000.0)]
        report = search.staleness_summary(tmp_path, discovered_files=discovered)
        assert report.appended_count == 1

    def test_injected_discovered_files_bypass_the_real_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing discovered_files explicitly must short-circuit
        discovery.discover_sessions() entirely -- proven by pointing
        Path.home() at an empty directory and confirming the report still
        reflects only the injected data."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "fake-home"))
        discovered = [(tmp_path / "a.jsonl", 100, 1.0)]
        report = search.staleness_summary(tmp_path, discovered_files=discovered)
        assert report.appended_count == 1


class TestResidueRefusesBeforeRepair:
    """Residue is refused upstream of repair (detect-and-refuse, D13).

    Pins the index-durability requirement "Residue is refused upstream of
    repair, not recovered by it": when a committed chunk row references a
    vec_row the vector file does not actually hold, the index-open step
    raises IndexNotReadyError before staleness detection -- and therefore
    before repair_current_session_tail() -- ever runs. Repair itself never
    calls recover(); this gate is what guarantees it can never append onto
    undetected crash residue.
    """

    def test_residue_bearing_index_refuses_search_before_repair_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ssgrep import repair

        index_dir = tmp_path / search.INDEX_DIRNAME
        _build_index(index_dir, FIXTURE_EPISODES)

        # Create crash residue: drop the final vector row so at least one
        # committed chunk references a vec_row past the file's actual end
        # (the same shape test_repair.py's crash-simulation test builds by
        # truncating vectors.f32 after a repair).
        vec_path = index_dir / "vectors.f32"
        row_bytes = embed.DIMENSION * 4
        vec_path.write_bytes(vec_path.read_bytes()[:-row_bytes])

        calls = {"count": 0}
        real_repair = repair.repair_current_session_tail

        def spy(*args: object, **kwargs: object) -> object:
            calls["count"] += 1
            return real_repair(*args, **kwargs)

        monkeypatch.setattr(repair, "repair_current_session_tail", spy)
        # Positive control (a bare `== 0` spy assertion passes vacuously if
        # the patched binding is bypassed): search/__init__.py resolves
        # `repair.repair_current_session_tail` through this exact module
        # attribute at call time, so the spy IS the binding search would
        # invoke on any path that reaches repair.
        assert search.repair.repair_current_session_tail is spy

        with pytest.raises(IndexNotReadyError):
            search.search(tmp_path, "retry_backoff_ms")

        assert calls["count"] == 0, "repair must never be invoked against residue"
