"""Tests for global search orchestration and row normalization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import ssgrep.search as search_module
from ssgrep.search.rows import _EpisodeRow
from ssgrep.utilities.types import (
    ContentType,
    EmptyQueryError,
    IndexNotFoundError,
    IndexNotReadyError,
    InvalidPredicateError,
    SearchFilters,
)


class FakeLanceStore:
    schema_version = 6

    def __init__(
        self,
        *,
        exists: bool = True,
        state: str | None = "ready",
        schema_ok: bool = True,
        chunk_count: int = 1,
        rows: list[dict] | None = None,
        search_error: Exception | None = None,
    ) -> None:
        self.exists_result = exists
        self.schema_ok = schema_ok
        self.chunk_count = chunk_count
        self.multivector_rows = [] if rows is None else rows
        self.search_error = search_error
        self.meta: dict[str, str | None] = {
            "index_state": state,
            "schema_version": str(self.schema_version),
            "model_id": search_module.MODEL_ID,
            "model_revision": search_module.MODEL_REVISION,
            "vector_dimension": str(search_module.DIMENSION),
        }
        self.count_calls: list[str] = []
        self.multivector_calls: list[tuple[np.ndarray, int, str | None]] = []
        self.two_stage_calls: list[dict] = []

    def exists(self) -> bool:
        return self.exists_result

    def get_meta(self, key: str) -> str | None:
        return self.meta.get(key)

    def schema_matches(self) -> bool:
        return self.schema_ok

    def count(self, table: str) -> int:
        self.count_calls.append(table)
        return self.chunk_count

    def multivector_search(
        self,
        query_matrix: np.ndarray,
        *,
        limit: int,
        where: str | None = None,
    ) -> list[dict]:
        self.multivector_calls.append((query_matrix, limit, where))
        if self.search_error is not None:
            raise self.search_error
        return self.multivector_rows

    def two_stage_search(
        self,
        query_matrix: np.ndarray,
        *,
        query_proxy: np.ndarray,
        limit: int,
        where: str | None = None,
        pool_size: int | None = None,
    ) -> list[dict]:
        self.two_stage_calls.append(
            {
                "query_matrix": query_matrix,
                "query_proxy": query_proxy,
                "limit": limit,
                "where": where,
                "pool_size": pool_size,
            }
        )
        if self.search_error is not None:
            raise self.search_error
        return self.multivector_rows


def use_store(monkeypatch: pytest.MonkeyPatch, store: FakeLanceStore) -> None:
    monkeypatch.setattr(search_module, "LanceStore", lambda: store)
    monkeypatch.setattr(
        search_module,
        "_query_matrix",
        lambda query: np.zeros((3, search_module.DIMENSION), dtype=np.float32),
    )
    # Keep search hermetic: a stub embedder that cannot encode keeps the
    # semantic snippet path on its per-chunk lexical fallback, so search tests
    # never load the real embedding model or depend on a local HF cache.
    monkeypatch.setattr(search_module, "load_embedder", lambda: _BrokenEmbedder())


class _BrokenEmbedder:
    """An embedder-shaped object whose encode always fails.

    Lets the semantic snippet path exercise its lexical fallback without a
    real model: ``semantic_windows`` catches the raised ``NotImplementedError``
    and routes every long chunk to ``fallback`` (the lexical window).
    """

    def encode(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError("hermetic test embedder cannot encode")


def searchable_row(**overrides: object) -> dict:
    row: dict[str, object] = {
        "episode_id": "session:ep:0",
        "content_type": "response",
        "text": "Use pytest.",
        "_distance": -0.75,
        "title": "Testing",
        "timestamp": "2025-01-02T03:04:05",
        "git_branch": "main",
        "files_touched": ["tests/test_example.py"],
        "is_subagent": False,
        "project": "/project",
        "source_path": "/session.jsonl",
    }
    row.update(overrides)
    return row


def test_as_datetime_normalizes_supported_values() -> None:
    value = datetime(2025, 1, 2, 3, 4)

    assert search_module._as_datetime(value) is value
    assert search_module._as_datetime("2025-01-02T03:04:05") == datetime(2025, 1, 2, 3, 4, 5)
    assert search_module._as_datetime("invalid") is None
    assert search_module._as_datetime(123) is None
    assert search_module._as_datetime("") is None


def test_as_items_normalizes_sequences_newline_strings_and_empty_values() -> None:
    assert search_module._as_items(["one", "", 2, None]) == ("one", "2")
    assert search_module._as_items(("one", "two")) == ("one", "two")
    assert search_module._as_items("one\ntwo") == ("one", "two")
    assert search_module._as_items(123) == ()
    assert search_module._as_items("") == ()


def test_query_matrix_embeds_query_via_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """_query_matrix embeds through load_embedder and returns a float32 matrix."""

    class _FakeEmbedder:
        def encode(
            self,
            texts: list[str],
            *,
            is_query: bool,
            normalize_embeddings: bool,
            pool_factor: int = 1,
        ) -> list[np.ndarray]:
            assert texts == ["the query"]
            assert is_query is True
            assert normalize_embeddings is True
            assert pool_factor == 1
            return [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]

    monkeypatch.setattr(search_module, "load_embedder", lambda: _FakeEmbedder())
    matrix = search_module._query_matrix("the query")
    assert matrix.shape == (1, 3)
    assert matrix.dtype == np.float32


def test_query_matrix_never_pools_regardless_of_pool_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries are NEVER pooled: SSGREP_POOL_FACTOR must not change Q's shape."""

    class _FakeEmbedder:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def encode(self, texts, **kwargs):
            self.calls.append(kwargs)
            return [np.full((7, search_module.DIMENSION), 0.5, dtype=np.float32)]

    fake = _FakeEmbedder()
    monkeypatch.setattr(search_module, "load_embedder", lambda: fake)

    monkeypatch.setenv("SSGREP_POOL_FACTOR", "3")
    pooled_env = search_module._query_matrix("the query")
    monkeypatch.setenv("SSGREP_POOL_FACTOR", "1")
    off_env = search_module._query_matrix("the query")

    assert pooled_env.shape == off_env.shape == (7, search_module.DIMENSION)
    assert len(fake.calls) == 2
    assert all(call.get("pool_factor") == 1 for call in fake.calls)
    assert all(call["is_query"] is True for call in fake.calls)


def test_utc_literal_normalizes_aware_values_and_preserves_naive_values() -> None:
    aware = datetime(2025, 1, 2, 8, 34, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    naive = datetime(2025, 1, 2, 3, 4)

    assert search_module._utc_literal(aware) == "2025-01-02T03:04:00"
    assert search_module._utc_literal(naive) == "2025-01-02T03:04:00"


def test_contains_quotes_literals_without_like_wildcard_semantics() -> None:
    assert search_module._contains("files", "50%_'quoted") == ("strpos(files, '50%_''quoted') > 0")


def test_build_where_compiles_every_public_filter_and_raw_predicate() -> None:
    filters = SearchFilters(
        date_from=datetime(2025, 1, 1, 12, tzinfo=timezone(timedelta(hours=2))),
        date_to=datetime(2025, 1, 2, 12),
        file_path="src/o'hare.py",
        content_type=ContentType.RESPONSE,
        branch="main",
        project="/project",
        source_path="/source.jsonl",
        session_id="session",
        is_subagent=True,
        agent_type="general",
        agent_model="model",
        tool_name="Read%",
        runtime="pi",
    )

    predicate = search_module.build_where(filters, "score > 0")

    assert predicate == " AND ".join(
        [
            "timestamp >= timestamp '2025-01-01T10:00:00'",
            "timestamp <= timestamp '2025-01-02T12:00:00'",
            "strpos(files_touched, 'src/o''hare.py') > 0",
            "content_type = 'response'",
            "git_branch = 'main'",
            "project = '/project'",
            "source_path = '/source.jsonl'",
            "session_id = 'session'",
            "is_subagent = true",
            "agent_type = 'general'",
            "agent_model = 'model'",
            "strpos(tool_names, 'Read%') > 0",
            "runtime = 'pi'",
            "(score > 0)",
        ]
    )


def test_build_where_handles_false_subagent_raw_only_and_no_filters() -> None:
    assert search_module.build_where(SearchFilters(is_subagent=False)) == "is_subagent = false"
    assert search_module.build_where(where="project IS NOT NULL") == "(project IS NOT NULL)"
    assert search_module.build_where() is None


def test_rows_to_episodes_rolls_up_best_scores_and_normalizes_metadata() -> None:
    rows = [
        searchable_row(
            _distance=-0.4,
            source_status="absent",
            agent_name="first",
            files_touched=["first.py"],
        ),
        searchable_row(
            _distance=-0.2,
            text="lower ranked duplicate",
            title="overwritten by each projection",
        ),
        searchable_row(
            _distance=-0.8,
            text="best chunk",
            title="Final metadata",
            timestamp="invalid",
            files_touched="a.py\nb.py",
            is_subagent=True,
            agent_name="worker",
            agent_description="task",
            parent_session_id="parent",
            project="/final",
            source_path="/final.jsonl",
            source_project="source-project",
            agent_model="model",
        ),
        searchable_row(
            episode_id="fallback:ep:0",
            _distance=None,
            title=None,
            timestamp=None,
            git_branch=None,
            files_touched=None,
            project=None,
            source_path=None,
        ),
        searchable_row(episode_id="missing-score:ep:0", _distance=None),
        searchable_row(episode_id="nan:ep:0", _distance=float("nan")),
        searchable_row(episode_id="inf:ep:0", _distance=float("inf")),
    ]
    # Removing the key (rather than setting None) exercises the rank fallback.
    rows[4].pop("_distance")

    rolled, episodes = search_module._rows_to_episodes(rows, num_query_tokens=3)

    # Best chunk maxsim only: weight 0.0 (additive bonus term vanishes).
    assert rolled["session:ep:0"][0] == pytest.approx(1.8)
    assert rolled["session:ep:0"][1].text == "best chunk"
    assert rolled["session:ep:0"][1].chunk_id != ""
    assert rolled["fallback:ep:0"][0] == pytest.approx(1 / 4)
    assert rolled["missing-score:ep:0"][0] == pytest.approx(1 / 5)
    assert rolled["nan:ep:0"][0] == 0.0
    assert rolled["inf:ep:0"][0] == 0.0
    assert episodes["session:ep:0"] == _EpisodeRow(
        title="Final metadata",
        timestamp=None,
        git_branch="main",
        files_touched=("a.py", "b.py"),
        is_subagent=True,
        agent_name="worker",
        agent_description="task",
        parent_session_id="parent",
        project="/final",
        source_path="/final.jsonl",
        source_project="source-project",
        agent_model="model",
    )
    assert episodes["fallback:ep:0"].title == "fallback:ep:0"
    assert episodes["fallback:ep:0"].files_touched == ()


def test_rows_to_episodes_converts_distance_to_maxsim() -> None:
    """Refined-scale conversion: maxsim = DISTANCE_TO_MAXSIM_OFFSET - _distance.

    Identity case: an exact-match doc with T=2 tokens reports _distance=-1.0
    on the refined scale (lancedb 0.37.1 measures flat scans and
    refine_factor(1) rescores both onto _distance = 1 - MaxSim).
    """
    assert search_module.DISTANCE_TO_MAXSIM_OFFSET == 1.0

    rows = [searchable_row(_distance=-1.0)]

    rolled, _episodes = search_module._rows_to_episodes(rows, num_query_tokens=2)

    assert rolled["session:ep:0"][0] == pytest.approx(2.0)

    partial = [searchable_row(_distance=0.25)]

    rolled, _episodes = search_module._rows_to_episodes(partial, num_query_tokens=12)

    assert rolled["session:ep:0"][0] == pytest.approx(0.75)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query_before_opening_index(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    def unexpected_store() -> FakeLanceStore:
        raise AssertionError("empty queries must not access the index")

    monkeypatch.setattr(search_module, "LanceStore", unexpected_store)

    with pytest.raises(EmptyQueryError, match="must not be empty") as caught:
        search_module.search(query)

    assert caught.value.condition == "empty_query"


def test_search_raises_for_missing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    use_store(monkeypatch, FakeLanceStore(exists=False))

    with pytest.raises(IndexNotFoundError, match="No global index found"):
        search_module.search("query")


def test_search_raises_for_incomplete_index(monkeypatch: pytest.MonkeyPatch) -> None:
    use_store(monkeypatch, FakeLanceStore(state="building"))

    with pytest.raises(IndexNotReadyError, match="incomplete") as caught:
        search_module.search("query")

    assert caught.value.condition == "index_incomplete"
    assert caught.value.command == "ssgrep index"


def test_search_rejects_table_schema_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    use_store(monkeypatch, FakeLanceStore(schema_ok=False))

    with pytest.raises(IndexNotReadyError, match="embedding model is incompatible"):
        search_module.search("query")


def test_search_rejects_embedding_metadata_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeLanceStore()
    store.meta["model_revision"] = "old-revision"
    use_store(monkeypatch, store)

    with pytest.raises(IndexNotReadyError, match="--rebuild"):
        search_module.search("query")


def test_search_rejects_legacy_v4_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A v4 index (768-d mpnet, schema 4) must fail the compatibility gate."""
    store = FakeLanceStore()
    store.meta["schema_version"] = "4"
    store.meta["model_id"] = "sentence-transformers/all-mpnet-base-v2"
    store.meta["vector_dimension"] = "768"
    use_store(monkeypatch, store)

    with pytest.raises(IndexNotReadyError, match="--rebuild"):
        search_module.search("query")


def test_search_rejects_stale_v5_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A v5 index (pre-pooling, no proxy_vector column) fails the gate."""
    store = FakeLanceStore()
    store.meta["schema_version"] = "5"
    use_store(monkeypatch, store)

    with pytest.raises(IndexNotReadyError, match="--rebuild"):
        search_module.search("query")


def test_search_accepts_fresh_v6_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A freshly rebuilt pooled/v6 index passes the compatibility gate."""
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)

    response = search_module.search("query")

    assert response.total_matches == 1
    assert store.meta["schema_version"] == "6"
    assert store.meta["model_id"] == search_module.MODEL_ID
    assert store.meta["vector_dimension"] == str(search_module.DIMENSION)


def test_search_empty_index_returns_empty_response_and_clamp_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeLanceStore(chunk_count=0)
    use_store(monkeypatch, store)

    response = search_module.search("query", limit=search_module.MAX_RESULT_COUNT + 1)

    assert response.results == []
    assert response.index_empty is True
    assert response.total_matches == 0
    assert response.omitted_count == 0
    assert response.excerpts_truncated is False
    assert response.clamped is True
    assert store.count_calls == [search_module.CHUNKS_TABLE]
    assert store.multivector_calls == []


def test_search_runs_multivector_query_and_builds_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)
    filters = SearchFilters(project="/project")

    response = search_module.search("  pytest  ", filters=filters, where="git_branch = 'main'")

    assert [card.ref for card in response.results] == ["session:ep:0"]
    assert response.results[0].excerpt == "Use pytest."
    # Refined scale: _distance=-0.75 -> maxsim = 1 - (-0.75) = 1.75.
    assert response.results[0].score == pytest.approx(1 - (-0.75))
    assert response.total_matches == 1
    assert response.index_empty is False
    assert response.clamped is False
    assert len(store.multivector_calls) == 1
    query_matrix, limit, where = store.multivector_calls[0]
    assert query_matrix.shape == (3, search_module.DIMENSION)
    assert limit == search_module.DEFAULT_RESULT_COUNT * search_module.OVERSAMPLE_FACTOR
    assert where == "project = '/project' AND (git_branch = 'main')"


def test_search_multiplies_engine_limit_by_oversample_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine fetch is limit x OVERSAMPLE_FACTOR; the rollup consumes the pool.

    Multiplication happens BEFORE rollup (never after): the extra candidates
    exist so per-episode rollup sees multiple chunks per episode instead of
    letting one dominant episode starve the rest.
    """
    monkeypatch.delenv("SSGREP_OVERSAMPLE", raising=False)
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)

    search_module.search("query", limit=5)

    assert len(store.multivector_calls) == 1
    _query_matrix, limit, _where = store.multivector_calls[0]
    assert limit == 5 * search_module.OVERSAMPLE_FACTOR


def test_search_response_stays_within_requested_limit_despite_oversampled_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI-visible results never exceed the requested limit after rollup."""
    monkeypatch.delenv("SSGREP_OVERSAMPLE", raising=False)
    rows = [
        searchable_row(episode_id=f"session-{index}:ep:0", text=f"chunk {index}")
        for index in range(12)
    ]
    store = FakeLanceStore(rows=rows)
    use_store(monkeypatch, store)

    response = search_module.search("query", limit=3)

    _query_matrix, engine_limit, _where = store.multivector_calls[0]
    assert engine_limit == 3 * search_module.OVERSAMPLE_FACTOR
    assert len(response.results) <= 3
    # The oversampled pool fed the rollup: every fetched episode was scored.
    assert response.total_matches == 12


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, 8, id="unset-default"),
        pytest.param("garbage", 8, id="unparsable-default"),
        pytest.param("0", 1, id="clamped-low"),
        pytest.param("-3", 1, id="negative-clamped-low"),
        pytest.param("99", 20, id="clamped-high"),
        pytest.param("7", 7, id="in-range-honored"),
    ],
)
def test_oversample_factor_env_clamps_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    """SSGREP_OVERSAMPLE clamps to [1, 20]; unparsable values keep the default."""
    if raw is None:
        monkeypatch.delenv("SSGREP_OVERSAMPLE", raising=False)
    else:
        monkeypatch.setenv("SSGREP_OVERSAMPLE", raw)

    assert search_module._resolved_oversample_factor() == expected


def test_search_normalizes_negative_limit_and_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)

    response = search_module.search("query", limit=-10, token_budget=-1)

    assert response.results == []
    assert response.total_matches == 1
    assert response.clamped is False
    assert len(store.multivector_calls) == 1
    query_matrix, limit, where = store.multivector_calls[0]
    assert query_matrix.shape == (3, search_module.DIMENSION)
    assert limit == 0
    assert where is None


@pytest.mark.parametrize(
    ("raw", "enabled"),
    [
        pytest.param(None, False, id="unset-off"),
        pytest.param("", False, id="empty-off"),
        pytest.param("0", False, id="zero-off"),
        pytest.param("false", False, id="false-off"),
        pytest.param("garbage", False, id="garbage-off"),
        pytest.param("1", True, id="one-on"),
        pytest.param("true", True, id="true-on"),
        pytest.param("YES", True, id="yes-case-insensitive-on"),
        pytest.param("on", True, id="on-on"),
    ],
)
def test_two_stage_flag_resolution(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, enabled: bool
) -> None:
    """SSGREP_TWO_STAGE defaults OFF; only truthy tokens enable it."""
    if raw is None:
        monkeypatch.delenv(search_module.TWO_STAGE_ENV, raising=False)
    else:
        monkeypatch.setenv(search_module.TWO_STAGE_ENV, raw)

    assert search_module._two_stage_enabled() is enabled


def test_search_flag_off_keeps_single_stage_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default env: the engine call is byte-for-byte today's single-stage path."""
    monkeypatch.delenv(search_module.TWO_STAGE_ENV, raising=False)
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)

    search_module.search("query", limit=5)

    assert len(store.multivector_calls) == 1
    assert store.two_stage_calls == []


def test_search_flag_on_routes_to_two_stage_with_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(search_module.TWO_STAGE_ENV, "1")
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)
    proxy_sentinel = np.full(search_module.DIMENSION, 0.5, dtype=np.float32)
    monkeypatch.setattr(
        search_module, "_query_proxy_vector", lambda _query, _matrix: proxy_sentinel
    )

    response = search_module.search("query", limit=5, where="runtime = 'pi'")

    assert [card.ref for card in response.results] == ["session:ep:0"]
    assert store.multivector_calls == []
    assert len(store.two_stage_calls) == 1
    call = store.two_stage_calls[0]
    assert call["query_proxy"] is proxy_sentinel
    assert call["where"] == "(runtime = 'pi')"
    # Candidate budget: max(200, limit x OVERSAMPLE x 4) — deeper than the
    # single-stage pool because the prefilter must not drop true hits. The
    # rollup pool stays at the single-stage size (limit x OVERSAMPLE).
    assert call["limit"] == max(200, 5 * search_module.OVERSAMPLE_FACTOR * 4)
    assert call["pool_size"] == 5 * search_module.OVERSAMPLE_FACTOR


@pytest.mark.parametrize(
    ("limit", "oversample", "expected"),
    [
        pytest.param(5, 4, 200, id="floor-wins"),
        pytest.param(50, 4, 800, id="formula-wins"),
        pytest.param(10, 1, 200, id="small-oversample-floored"),
        pytest.param(50, 20, 4000, id="maxed-out"),
    ],
)
def test_two_stage_candidate_budget_formula(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
    oversample: int,
    expected: int,
) -> None:
    monkeypatch.setenv(search_module.TWO_STAGE_ENV, "1")
    monkeypatch.setenv("SSGREP_OVERSAMPLE", str(oversample))
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)
    monkeypatch.setattr(
        search_module,
        "_query_proxy_vector",
        lambda _query, _matrix: np.zeros(search_module.DIMENSION, dtype=np.float32),
    )

    search_module.search("query", limit=limit)

    assert store.two_stage_calls[0]["limit"] == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, 200, id="unset-default-floor"),
        pytest.param("garbage", 200, id="unparsable-default-floor"),
        pytest.param("50", 200, id="below-floor-clamped"),
        pytest.param("999999", 10000, id="above-ceiling-clamped"),
        pytest.param("1000", 1000, id="in-range-honored"),
    ],
)
def test_two_stage_candidate_floor_env_clamps(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    """SSGREP_TWO_STAGE_CANDIDATES widens the stage-1 floor; clamps to [200, 10000]."""
    if raw is None:
        monkeypatch.delenv(search_module.TWO_STAGE_CANDIDATES_ENV, raising=False)
    else:
        monkeypatch.setenv(search_module.TWO_STAGE_CANDIDATES_ENV, raw)

    assert search_module._two_stage_candidate_floor() == expected


def test_query_proxy_vector_uses_only_real_token_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pad rows are excluded: pylate pads queries to 32 mask-token rows.

    A mean over all rows would be dominated by that constant padding and
    carry almost no query signal, so the tokenizer's attention mask marks
    the real prefix and only those rows are averaged.
    """

    class _FakeEmbedder:
        def tokenize(self, texts: list[str], *, is_query: bool) -> dict:
            assert texts == ["the query"]
            assert is_query is True
            return {"attention_mask": np.array([[1, 1, 1, 0, 0]], dtype=np.int64)}

    monkeypatch.setattr(search_module, "load_embedder", lambda: _FakeEmbedder())
    matrix = np.arange(10, dtype=np.float32).reshape(5, 2)
    matrix[3:] = 99.0

    proxy = search_module._query_proxy_vector("the query", matrix)

    expected = matrix[:3].mean(axis=0)
    np.testing.assert_allclose(proxy, expected / np.linalg.norm(expected), rtol=1e-6)


def test_query_proxy_vector_falls_back_to_full_matrix_without_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake embedders (no tokenize) yield unpadded matrices; use them as-is."""

    class _FakeEmbedder:
        def tokenize(self, *_args: object, **_kwargs: object) -> dict:
            raise AttributeError("no tokenizer on this fake")

    monkeypatch.setattr(search_module, "load_embedder", lambda: _FakeEmbedder())
    matrix = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)

    proxy = search_module._query_proxy_vector("q", matrix)

    expected = matrix.mean(axis=0)
    np.testing.assert_allclose(proxy, expected / np.linalg.norm(expected), rtol=1e-6)


def test_two_stage_failure_translates_like_single_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(search_module.TWO_STAGE_ENV, "1")
    store = FakeLanceStore(search_error=ValueError("bad syntax"))
    use_store(monkeypatch, store)
    monkeypatch.setattr(
        search_module,
        "_query_proxy_vector",
        lambda _query, _matrix: np.zeros(search_module.DIMENSION, dtype=np.float32),
    )

    with pytest.raises(InvalidPredicateError, match="Invalid metadata predicate: bad syntax"):
        search_module.search("query", where="bad (")


def test_search_zero_token_budget_drops_matching_card(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)

    response = search_module.search("query", limit=1, token_budget=0)

    assert response.results == []
    assert response.total_matches == 1
    assert response.omitted_count == 1


def test_search_translates_value_error_with_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeLanceStore(search_error=ValueError("bad syntax"))
    use_store(monkeypatch, store)

    with pytest.raises(
        InvalidPredicateError, match="Invalid metadata predicate: bad syntax"
    ) as caught:
        search_module.search("query", where="bad (")

    assert caught.value.__cause__ is store.search_error


def test_search_translates_value_error_without_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeLanceStore(search_error=ValueError("fts unavailable"))
    use_store(monkeypatch, store)

    with pytest.raises(
        IndexNotReadyError, match="multivector search failed: fts unavailable"
    ) as caught:
        search_module.search("query")

    assert caught.value.__cause__ is store.search_error


def test_search_translates_unexpected_multivector_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeLanceStore(search_error=RuntimeError("database closed"))
    use_store(monkeypatch, store)

    with pytest.raises(
        IndexNotReadyError, match="multivector search failed: database closed"
    ) as caught:
        search_module.search("query", where="valid = true")

    assert caught.value.__cause__ is store.search_error


def test_search_surfaces_model_download_failure_as_ready_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ssgrep.indexing.embed import ModelDownloadError

    store = FakeLanceStore(
        search_error=ModelDownloadError("Model google/embeddinggemma-300m is gated on Hugging Face")
    )
    use_store(monkeypatch, store)

    with pytest.raises(IndexNotReadyError, match="gated on Hugging Face") as caught:
        search_module.search("query")

    assert caught.value.condition == "model_unavailable"
    assert caught.value.command is None
    assert isinstance(caught.value.__cause__, ModelDownloadError)


def test_search_wires_semantic_window_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default env: response shaping receives the semantic batch window fn."""
    monkeypatch.delenv("SSGREP_SEMANTIC_SNIPPET", raising=False)
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)
    captured: dict[str, object] = {}
    original = search_module._build_response

    def capture_build_response(rolled_up, episode_rows, **kwargs):
        captured.update(kwargs)
        return original(rolled_up, episode_rows, **kwargs)

    monkeypatch.setattr(search_module, "_build_response", capture_build_response)

    search_module.search("query")

    assert callable(captured["window"])


def test_search_wires_no_semantic_window_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSGREP_SEMANTIC_SNIPPET=0: response shaping keeps the lexical window."""
    monkeypatch.setenv("SSGREP_SEMANTIC_SNIPPET", "0")
    store = FakeLanceStore(rows=[searchable_row()])
    use_store(monkeypatch, store)
    captured: dict[str, object] = {}
    original = search_module._build_response

    def capture_build_response(rolled_up, episode_rows, **kwargs):
        captured.update(kwargs)
        return original(rolled_up, episode_rows, **kwargs)

    monkeypatch.setattr(search_module, "_build_response", capture_build_response)

    search_module.search("query")

    assert captured["window"] is None


def test_search_semantic_window_falls_back_to_lexical_on_embed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long chunk whose encode raises still renders via the lexical window."""
    long_text = " ".join(["filler", "target"] * 60)  # ~720 chars > 400
    store = FakeLanceStore(rows=[searchable_row(text=long_text)])
    use_store(monkeypatch, store)

    response = search_module.search("target")

    assert response.excerpts_truncated is True
    assert len(response.results[0].excerpt) <= 450
    assert "target" in response.results[0].excerpt
