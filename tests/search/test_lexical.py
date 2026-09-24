"""Unit tests for the hybrid-retrieval lexical module (:mod:`ssgrep.search.lexical`)."""

from __future__ import annotations

import pytest

from ssgrep.search import lexical
from ssgrep.search.lexical import (
    bm25_chunk,
    corpus_stats,
    dump_stats,
    fuse_rrf,
    hybrid_scores,
    load_stats,
    tokenize,
)

# A tiny deterministic corpus: three chunks with known token distributions.
_CORPUS = [
    {"chunk_id": "a", "episode_id": "ep:0", "text": "def check_database_connection timeout retry"},
    {"chunk_id": "b", "episode_id": "ep:0", "text": "connection refused database timeout"},
    {"chunk_id": "c", "episode_id": "ep:1", "text": "unrelated notes about the weather today"},
]


def test_tokenize_lowercases_and_filters_short_tokens() -> None:
    assert tokenize("Foo_bar 123 ERROR!") == ["foo_bar", "123", "error"]
    assert tokenize("a b c") == []  # all single-char
    assert tokenize("") == []
    assert tokenize("snake_case-name.v2") == ["snake_case", "name", "v2"]


def test_corpus_stats_counts_df_once_per_doc_and_avgdl() -> None:
    stats = corpus_stats(_CORPUS)
    assert stats["n_docs"] == 3
    # "timeout" appears in docs a, b -> df 2
    assert stats["df"]["timeout"] == 2
    # "database" appears only in b (a has the compound check_database_connection) -> df 1
    assert stats["df"]["database"] == 1
    # "weather" appears only in c -> df 1
    assert stats["df"]["weather"] == 1
    # doc a has 4 tokens, b has 4, c has 6 -> avgdl 14/3
    assert stats["avgdl"] == pytest.approx(14 / 3)


def test_corpus_stats_empty_corpus_no_division_error() -> None:
    stats = corpus_stats([])
    assert stats["n_docs"] == 0
    assert stats["df"] == {}
    assert stats["avgdl"] == 0.0


def test_bm25_chunk_scores_verbatim_rare_token_higher() -> None:
    stats = corpus_stats(_CORPUS)
    # "check_database_connection" is unique to chunk a and matches verbatim
    a = bm25_chunk(_CORPUS[0]["text"], ["check_database_connection"], stats)
    c = bm25_chunk(_CORPUS[2]["text"], ["check_database_connection"], stats)
    assert a > 0.0
    assert c == 0.0  # token absent from this chunk


def test_bm25_chunk_empty_text_returns_zero() -> None:
    stats = corpus_stats(_CORPUS)
    assert bm25_chunk("", ["anything"], stats) == 0.0
    assert bm25_chunk("   ", ["anything"], stats) == 0.0


def test_bm25_chunk_token_not_in_query_contributes_nothing() -> None:
    stats = corpus_stats(_CORPUS)
    score = bm25_chunk(_CORPUS[0]["text"], ["timeout", "zzz_not_there"], stats)
    # only "timeout" matches; "zzz_not_there" is absent from the chunk
    timeout_only = bm25_chunk(_CORPUS[0]["text"], ["timeout"], stats)
    assert score == pytest.approx(timeout_only)


def test_bm25_chunk_unknown_token_gets_maximum_idf() -> None:
    # A token never seen in the corpus should still match verbatim with the
    # strongest idf (df=0 -> max), so rare identifiers are not suppressed.
    stats = {"df": {}, "avgdl": 5.0, "n_docs": 100}
    score = bm25_chunk("check_database_connection failed", ["check_database_connection"], stats)
    assert score > 0.0


def test_fuse_rrf_combines_both_signals_and_keeps_union() -> None:
    maxsim = {"ep:0": 30.0, "ep:1": 20.0, "ep:2": 10.0}
    lexical = {"ep:1": 5.0, "ep:3": 3.0}
    fused = fuse_rrf(maxsim, lexical, k=60)
    assert set(fused) == {"ep:0", "ep:1", "ep:2", "ep:3"}  # union
    # ep:1 ranks 2nd in both -> highest fused value
    assert fused["ep:1"] > fused["ep:0"]
    assert fused["ep:1"] > fused["ep:2"]
    assert fused["ep:3"] > 0.0  # lexical-only participant still present


def test_fuse_rrf_default_k_resolves_env(monkeypatch: pytest.MonkeyPatch) -> None:
    maxsim = {"a": 30.0, "b": 29.0}
    lexical = {"b": 1.0}
    import ssgrep.search.lexical as lex_module

    monkeypatch.delenv("SSGREP_RRF_K", raising=False)
    default = fusion_default = lex_module.fuse_rrf(maxsim, lexical)
    assert fusion_default["a"] == pytest.approx(1 / 61)
    # Override: k=60 is the shipped default constant (no env)
    monkeypatch.setenv("SSGREP_RRF_K", "100")
    overridden = lex_module.fuse_rrf(maxsim, lexical)
    assert overridden != default


def test_fuse_rrf_resolved_k_env_clamps_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ssgrep.search.lexical as lex_module

    monkeypatch.setenv("SSGREP_RRF_K", "not-a-number")
    assert lex_module._resolved_rrf_k() == lex_module.RRF_K
    monkeypatch.setenv("SSGREP_RRF_K", "0")
    assert lex_module._resolved_rrf_k() == 1
    monkeypatch.setenv("SSGREP_RRF_K", "99999")
    assert lex_module._resolved_rrf_k() == 500
    monkeypatch.setenv("SSGREP_RRF_K", "30")
    assert lex_module._resolved_rrf_k() == 30


def test_resolved_dense_weight_env_clamps_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ssgrep.search.lexical as lex_module

    monkeypatch.delenv("SSGREP_DENSE_WEIGHT", raising=False)
    assert lex_module._resolved_dense_weight() == 1.0
    monkeypatch.setenv("SSGREP_DENSE_WEIGHT", "not-a-number")
    assert lex_module._resolved_dense_weight() == 1.0
    monkeypatch.setenv("SSGREP_DENSE_WEIGHT", "0.1")
    assert lex_module._resolved_dense_weight() == 0.25
    monkeypatch.setenv("SSGREP_DENSE_WEIGHT", "99.0")
    assert lex_module._resolved_dense_weight() == 4.0
    monkeypatch.setenv("SSGREP_DENSE_WEIGHT", "2.0")
    assert lex_module._resolved_dense_weight() == 2.0


def test_fuse_rrf_deterministic_and_rank_weighted() -> None:
    maxsim = {"a": 30.0, "b": 29.0}
    lexical = {"b": 1.0, "a": 0.5}
    first = fuse_rrf(maxsim, lexical, k=60)
    second = fuse_rrf(maxsim, lexical, k=60)
    assert first == second
    # a: rank1 in maxsim, rank2 in lexical; b: rank2 in maxsim, rank1 in lexical
    assert first["a"] == 1 / 61 + 1 / 62
    assert first["b"] == 1 / 62 + 1 / 61


def test_hybrid_scores_fuses_when_stats_present() -> None:
    stats = corpus_stats(_CORPUS)
    maxsim = {"ep:0": 30.0, "ep:1": 20.0}
    fused = hybrid_scores(_CORPUS, maxsim, stats, "database connection timeout")
    assert set(fused) == {"ep:0", "ep:1"}
    # ep:0 has the lexical tokens -> boosted above its pure MaxSim rank
    assert fused["ep:0"] > 1 / 61


def test_hybrid_scores_none_stats_returns_maxsim_unchanged() -> None:
    maxsim = {"ep:0": 30.0, "ep:1": 20.0}
    assert hybrid_scores(_CORPUS, maxsim, None, "anything") == maxsim


def test_hybrid_scores_empty_query_returns_maxsim_unchanged() -> None:
    stats = corpus_stats(_CORPUS)
    maxsim = {"ep:0": 30.0}
    assert hybrid_scores(_CORPUS, maxsim, stats, "") == maxsim
    assert hybrid_scores(_CORPUS, maxsim, stats, "a b") == maxsim  # only short tokens


def test_hybrid_scores_no_lexical_matches_returns_maxsim_unchanged() -> None:
    stats = corpus_stats(_CORPUS)
    maxsim = {"ep:0": 30.0, "ep:1": 20.0}
    # query tokens that appear in no chunk text
    result = hybrid_scores(_CORPUS, maxsim, stats, "zzzqqq xxxyyy")
    assert result == maxsim


def test_dump_and_load_stats_round_trip() -> None:
    stats = corpus_stats(_CORPUS)
    raw = dump_stats(stats)
    assert isinstance(raw, str)
    assert load_stats(raw) == stats


def test_load_stats_handles_absent_and_malformed() -> None:
    assert load_stats(None) is None
    assert load_stats("") is None
    assert load_stats("not json {") is None
    assert load_stats('{"df": {}}') is None  # missing avgdl
    assert load_stats("[]") is None  # not a dict
    assert load_stats("[1,2") is None  # JSONDecodeError path


def test_corpus_stats_from_repository_reads_meta() -> None:
    class _FakeRepo:
        def __init__(self, raw: str | None) -> None:
            self._raw = raw

        def get_meta(self, key: str) -> str | None:  # noqa: ARG002 - fake signature
            return self._raw

    stats = corpus_stats(_CORPUS)
    repo_with = _FakeRepo(dump_stats(stats))
    assert lexical.corpus_stats_from_repository(repo_with) == stats
    repo_without = _FakeRepo(None)
    assert lexical.corpus_stats_from_repository(repo_without) is None
