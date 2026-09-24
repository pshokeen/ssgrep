"""Unit contract tests for :mod:`ssgrep.services.api`."""

from __future__ import annotations

from ssgrep.services import api
from ssgrep.utilities.types import ContentType, SearchFilters


def test_index_is_a_lossless_adapter(monkeypatch, sample_stats):
    calls = []

    def fake_index(**kwargs):
        calls.append(kwargs)
        return sample_stats

    monkeypatch.setattr(api.indexer, "index", fake_index)

    result = api.index(
        rebuild=True,
        no_subagents=True,
        allow_shrink=True,
        scope="/work/repo",
        live=True,
        full_reprocess=True,
    )

    assert result is sample_stats
    assert calls == [
        {
            "rebuild": True,
            "no_subagents": True,
            "allow_shrink": True,
            "scope": "/work/repo",
            "quiet": False,
            "live": True,
            "full_reprocess": True,
        }
    ]


def test_index_forwards_defaults(monkeypatch, sample_stats):
    calls = []
    monkeypatch.setattr(api.indexer, "index", lambda **kwargs: calls.append(kwargs) or sample_stats)

    assert api.index() is sample_stats
    assert calls == [
        {
            "rebuild": False,
            "no_subagents": False,
            "allow_shrink": False,
            "scope": None,
            "quiet": False,
            "live": False,
            "full_reprocess": False,
        }
    ]


def test_search_is_a_lossless_adapter(monkeypatch, sample_response):
    calls = []
    filters = SearchFilters(content_type=ContentType.RESPONSE, project="acme")

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return sample_response

    monkeypatch.setattr(api.search_module, "search", fake_search)

    result = api.search(
        "how was this fixed?",
        limit=4,
        token_budget=900,
        filters=filters,
        where="git_branch = 'main'",
    )

    assert result is sample_response
    assert calls == [
        (
            "how was this fixed?",
            {
                "limit": 4,
                "token_budget": 900,
                "filters": filters,
                "where": "git_branch = 'main'",
            },
        )
    ]


def test_show_and_status_delegate_without_transforming(monkeypatch, sample_detail, sample_stats):
    monkeypatch.setattr(api.detail, "show", lambda ref: sample_detail if ref == "known" else None)
    monkeypatch.setattr(api.observability, "status", lambda: sample_stats)

    assert api.show("known") is sample_detail
    assert api.show("missing") is None
    assert api.status() is sample_stats


def test_sources_forwards_scope_and_no_subagents(monkeypatch) -> None:
    calls = []
    discovered = [object(), object()]

    def fake_discover(*, scope, no_subagents):
        calls.append((scope, no_subagents))
        return discovered

    monkeypatch.setattr(api.transcript_adapters, "discover_sources", fake_discover)
    monkeypatch.setattr(
        api.transcript_adapters,
        "source_counts",
        lambda sources: (("pi", len(sources)),),
    )

    assert api.sources(scope="/work", no_subagents=True) == (("pi", 2),)
    assert calls == [("/work", True)]
